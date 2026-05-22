// hpx_server_stream_disconnect_smoke.cpp — M7d in-process disconnect
// smoke for llama-hpx-server.
//
// Boots model + engine + cpp-httplib server in-process, drives a
// streaming `POST /completion` whose client aborts mid-response, then
// posts a normal greedy/default `POST /completion` to prove the
// server stayed healthy. After the engine drains, the smoke asserts
// against `engine_result` counters that the disconnect path issued
// the expected fire-and-forget cancellation.
//
// Mechanism (vendored cpp-httplib semantics, verified directly):
//   - Client side: `Client::Post(..., ContentReceiver)` invokes the
//     receiver per response chunk. Returning `false` from the
//     receiver triggers `Error::Canceled` and tears down the socket.
//   - Server side: `data_sink.write` is a closure that internally
//     calls `write_data(strm, chunk, ...)`. On socket teardown,
//     `write_data` returns false → the closure flips its captured
//     `ok` to false → every subsequent `sink.write(...)` returns
//     false. This is the M7d-canonical signal; on first false return
//     the server-side provider issues `submit_handle::cancel(eng)`
//     once, drains the per-request stream channel in-place until
//     `stream_event_kind::closed`, awaits `submit_handle.result` to
//     observe final lifecycle, and returns false from the provider.
//
// What this smoke asserts:
//   - the disconnecting client observes either no Result (with
//     `Error::Canceled` or `Error::Read`) or a truncated 200 response
//     (cpp-httplib versions vary on which Error code surfaces) —
//     both are acceptable evidence that the client tore down before
//     the SSE `done` record arrived;
//   - a subsequent non-streaming POST returns HTTP 200 with the
//     budget-8 greedy canonical hash `0x0619d4d1900c2365`, proving
//     the server is alive and the engine slot was recycled;
//   - after engine drain, `engine_result.cancel_request_calls >= 1`,
//     `cancel_unknown_request_id == 0`, `decode_failures == 0`, and
//     `residual_kv_ok == true`.
//   - the smoke does NOT pin the disconnected request's final
//     `request_status` because cancel/complete is engine-timing.
//
// HPX-nativity boundary (mirrors hpx-server.cpp):
//   cpp-httplib is the explicit non-HPX network adapter boundary;
//   this TU contains no `std::thread`, `std::mutex`,
//   `std::condition_variable`, or `std::this_thread::sleep_for`. The
//   chunked-provider lambda runs on a cpp-httplib worker thread and
//   only touches: `submit_handle.stream->get(hpx::launch::sync)`,
//   `common_detokenize(const llama_vocab *, ...)`,
//   `httplib::DataSink::write/done`, `submit_handle::cancel(eng)`
//   (fire-and-forget), and `submit_handle.result.get()`. It never
//   touches `llama_context *`, `llama_decode`,
//   `llama_get_logits_ith`, `llama_sampler_*`, `llama_memory_seq_*`,
//   KV APIs, engine internals, or promise internals.

#include "common.h"
#include "llama.h"

#include "engine.h"
#include "hpx_runtime.h"
#include "trace.h"
#include "types.h"

#include <hpx/hpx.hpp>

#include <cpp-httplib/httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

struct smoke_args {
    std::string model_path;
};

void print_usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s --model <path>\n"
        "  M7d HPX server stream-disconnect smoke.\n",
        argv0);
}

bool parse_args(int argc, char ** argv, smoke_args & out) {
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--model") {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: --model requires a value\n");
                return false;
            }
            out.model_path = argv[++i];
        } else if (a == "-h" || a == "--help") {
            print_usage(argv[0]);
            return false;
        } else {
            fprintf(stderr, "error: unknown arg '%s'\n", a.c_str());
            return false;
        }
    }
    if (out.model_path.empty()) {
        fprintf(stderr, "error: --model is required\n");
        return false;
    }
    return true;
}

void emit_fail(const std::string & reason) {
    fprintf(stdout, "HPX_SERVER_STREAM_DISCONNECT_SMOKE: FAIL: %s\n",
            reason.c_str());
    fflush(stdout);
}

void emit_pass() {
    fprintf(stdout, "HPX_SERVER_STREAM_DISCONNECT_SMOKE: PASS\n");
    fflush(stdout);
}

void send_error(httplib::Response & res, int status,
                const std::string & code,
                const std::string & message) {
    nlohmann::json body;
    body["error"]["code"]    = code;
    body["error"]["message"] = message;
    res.status = status;
    res.set_content(body.dump(), "application/json");
}

std::string hex_hash(uint64_t h) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "0x%016llx",
                  static_cast<unsigned long long>(h));
    return std::string(buf);
}

}  // namespace

int main(int argc, char ** argv) {
    smoke_args args;
    if (!parse_args(argc, argv, args)) return 2;

    trace::init();

    if (!hpx_runtime::start_once(/*os_threads=*/2)) {
        emit_fail("hpx runtime start failed");
        return 1;
    }

    common_init();
    llama_backend_init();

    llama_model *   model = nullptr;
    llama_context * ctx   = nullptr;

    auto cleanup_and_stop = [&](int rc) -> int {
        if (ctx)   llama_free(ctx);
        if (model) llama_model_free(model);
        llama_backend_free();
        hpx_runtime::stop();
        return rc;
    };

    auto fail_and_cleanup = [&](const std::string & reason) -> int {
        emit_fail(reason);
        return cleanup_and_stop(1);
    };

    llama_model_params model_params = llama_model_default_params();
    model = llama_model_load_from_file(args.model_path.c_str(),
                                       model_params);
    if (model == nullptr) {
        return fail_and_cleanup("model load failed");
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx           = 2048;
    ctx_params.n_batch         = 1024;
    ctx_params.n_seq_max       = 2;
    ctx_params.n_threads       = 2;
    ctx_params.n_threads_batch = 2;
    ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        return fail_and_cleanup("context create failed");
    }

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);

    constexpr int32_t k_n_seq_max         = 2;
    constexpr int32_t k_max_prompt_tokens = 512;
    constexpr int32_t k_batch_capacity    =
        k_n_seq_max * k_max_prompt_tokens;
    // Large enough that the engine still has many decode iters ahead
    // of it when the client tears down — gives the cancel inbox a
    // wide window to be observed before the sequence naturally
    // completes.
    constexpr int32_t k_disconnect_decode_budget = 128;
    constexpr int32_t k_sanity_decode_budget     = 8;
    constexpr const char * k_canonical_greedy_b8 =
        "0x0619d4d1900c2365";

    std::vector<waiting_request> empty_waiting;

    engine_options opts;
    opts.lib.ctx                  = ctx;
    opts.lib.vocab                = vocab;
    opts.lib.n_vocab              = n_vocab;
    opts.lib.batch_capacity       = k_batch_capacity;
    opts.lib.n_seq_max            = k_n_seq_max;
    opts.lib.initial_idle_slots   = k_n_seq_max;
    opts.lib.keep_alive           = true;
    opts.preload.prompt_tokens    = nullptr;
    opts.preload.budgets          = {};
    opts.preload.waiting_queue    = &empty_waiting;
    opts.preload.reuse_completed  = false;
    opts.preload.stream_all       = false;
    opts.gate_test.cancel_after     = -1;
    opts.gate_test.max_decode_iters = 0;

    try {
        engine eng(std::move(opts));

        hpx::future<void> engine_fut =
            hpx::async([&] { eng.run(); });

        std::atomic<int32_t> next_rid{1};

        httplib::Server srv;
        // Handler mirrors the M7d shape of hpx-server.cpp. Inlined
        // here so the smoke is self-contained.
        srv.Post("/completion",
                 [&](const httplib::Request & req,
                     httplib::Response &       res) {
            nlohmann::json body;
            try {
                body = nlohmann::json::parse(req.body);
            } catch (const std::exception &) {
                send_error(res, 400, "bad_request",
                           "request body must be valid JSON");
                return;
            }
            if (!body.is_object()
             || !body.contains("prompt")
             || !body["prompt"].is_string()
             || !body.contains("decode_budget")
             || !body["decode_budget"].is_number_integer()) {
                send_error(res, 400, "bad_request",
                           "missing required fields");
                return;
            }
            const std::string prompt        = body["prompt"];
            const int32_t     decode_budget =
                body["decode_budget"].get<int32_t>();
            if (decode_budget <= 0) {
                send_error(res, 422, "invalid_argument",
                           "'decode_budget' must be > 0");
                return;
            }

            bool stream_requested = false;
            if (body.contains("stream")) {
                if (!body["stream"].is_boolean()) {
                    send_error(res, 400, "bad_request",
                        "'stream' must be a boolean");
                    return;
                }
                stream_requested = body["stream"].get<bool>();
            }

            std::vector<llama_token> tokens = common_tokenize(
                vocab, prompt, /*add_special=*/true,
                /*parse_special=*/true);
            if (tokens.empty()) {
                send_error(res, 422, "invalid_argument",
                           "tokenization produced zero tokens");
                return;
            }
            if (static_cast<int32_t>(tokens.size())
                  > k_max_prompt_tokens) {
                send_error(res, 413, "payload_too_large",
                           "prompt exceeds max-prompt-tokens");
                return;
            }

            const int32_t rid =
                next_rid.fetch_add(1, std::memory_order_relaxed);

            submit_request sr;
            sr.request_id    = rid;
            sr.prompt_tokens = std::move(tokens);
            sr.decode_budget = decode_budget;
            sr.want_stream   = stream_requested;

            submit_handle h = eng.submit_request(std::move(sr));

            if (!stream_requested) {
                request_result r;
                try {
                    r = h.result.get();
                } catch (const std::exception & e) {
                    send_error(res, 500, "engine_error", e.what());
                    return;
                }
                const std::string text =
                    common_detokenize(vocab, r.generated_tokens,
                                      /*special=*/false);
                nlohmann::json out;
                out["request_id"] = r.request_id;
                out["status"]     = status_name(r.status);
                out["n_decoded"]  = r.n_decoded;
                out["hash"]       = hex_hash(r.hash);
                out["text"]       = text;
                res.status = 200;
                res.set_content(out.dump(), "application/json");
                return;
            }

            if (!h.stream.has_value()) {
                send_error(res, 500, "engine_error",
                    "engine did not provide a stream channel for "
                    "want_stream=true");
                return;
            }

            struct stream_state {
                submit_handle h;
                bool          sink_alive        = true;
                bool          cancel_issued     = false;
                bool          result_consumed   = false;
                bool          completed_cleanly = false;
            };
            auto state = std::make_shared<stream_state>();
            state->h   = std::move(h);

            res.status = 200;
            res.set_chunked_content_provider(
                "text/event-stream",
                [state, vocab, &eng](size_t /*offset*/,
                                     httplib::DataSink & sink) -> bool {
                    token_stream_event ev;
                    try {
                        ev = state->h.stream->get(hpx::launch::sync);
                    } catch (const std::exception &) {
                        if (state->sink_alive) sink.done();
                        return false;
                    }

                    if (ev.kind == stream_event_kind::token) {
                        if (state->sink_alive) {
                            const std::string text = common_detokenize(
                                vocab,
                                std::vector<llama_token>{ev.token_id},
                                /*special=*/false);
                            nlohmann::json data;
                            data["token"]    = text;
                            data["token_id"] = ev.token_id;
                            std::string payload =
                                "event: token\ndata: " + data.dump()
                              + "\n\n";
                            if (sink.write(payload.data(),
                                           payload.size())) {
                                return true;
                            }
                            state->sink_alive = false;
                            if (!state->cancel_issued) {
                                state->h.cancel(eng);
                                state->cancel_issued = true;
                            }
                        }
                        while (true) {
                            token_stream_event drain_ev;
                            try {
                                drain_ev = state->h.stream->get(
                                    hpx::launch::sync);
                            } catch (const std::exception &) {
                                break;
                            }
                            if (drain_ev.kind ==
                                  stream_event_kind::closed) {
                                break;
                            }
                        }
                        if (!state->result_consumed) {
                            try {
                                (void)state->h.result.get();
                            } catch (const std::exception &) {}
                            state->result_consumed = true;
                        }
                        return false;
                    }

                    // kind == closed (happy path)
                    request_result r;
                    bool result_ok = true;
                    try {
                        r = state->h.result.get();
                    } catch (const std::exception &) {
                        result_ok = false;
                    }
                    state->result_consumed = true;
                    if (state->sink_alive) {
                        nlohmann::json data;
                        if (result_ok) {
                            data["request_id"] = r.request_id;
                            data["status"]     = status_name(r.status);
                            data["n_decoded"]  = r.n_decoded;
                            data["hash"]       = hex_hash(r.hash);
                        } else {
                            data["request_id"] = -1;
                            data["status"]     = "failed_reserved";
                            data["n_decoded"]  = 0;
                            data["hash"]       = hex_hash(0);
                        }
                        std::string payload =
                            "event: done\ndata: " + data.dump()
                          + "\n\n";
                        sink.write(payload.data(), payload.size());
                        sink.done();
                    }
                    state->completed_cleanly = true;
                    return false;
                },
                [state, &eng](bool /*success*/) {
                    if (state->completed_cleanly) return;
                    state->sink_alive = false;
                    if (!state->cancel_issued) {
                        state->h.cancel(eng);
                        state->cancel_issued = true;
                    }
                    while (true) {
                        token_stream_event drain_ev;
                        try {
                            drain_ev = state->h.stream->get(
                                hpx::launch::sync);
                        } catch (const std::exception &) {
                            break;
                        }
                        if (drain_ev.kind ==
                              stream_event_kind::closed) {
                            break;
                        }
                    }
                    if (!state->result_consumed) {
                        try {
                            (void)state->h.result.get();
                        } catch (const std::exception &) {}
                        state->result_consumed = true;
                    }
                });
        });

        const int port = srv.bind_to_any_port("127.0.0.1");
        if (port < 0) {
            eng.request_shutdown();
            engine_fut.get();
            return fail_and_cleanup("bind_to_any_port failed");
        }

        hpx::future<bool> listen_fut = hpx::async([&] {
            return srv.listen_after_bind();
        });
        srv.wait_until_ready();

        auto drain_engine_and_fail =
            [&](const std::string & reason) -> int {
            srv.stop();
            listen_fut.get();
            eng.request_shutdown();
            engine_fut.get();
            return fail_and_cleanup(reason);
        };

        // ---- Phase 1: streaming POST + mid-response client abort ---
        {
            httplib::Client cli("127.0.0.1", port);
            cli.set_read_timeout(60, 0);
            cli.set_write_timeout(30, 0);

            nlohmann::json req_body;
            req_body["prompt"]        = "Hello, my name is";
            req_body["decode_budget"] = k_disconnect_decode_budget;
            req_body["stream"]        = true;
            const std::string req_str = req_body.dump();

            // Abort after we've seen at least one full SSE record
            // (an `event: token\ndata: {...}\n\n` block ends in a
            // double newline; ~64 bytes is a safe lower bound).
            constexpr size_t k_abort_after_bytes = 64;
            size_t received = 0;
            bool   abort_triggered = false;

            httplib::ContentReceiver receiver =
                [&](const char * /*data*/, size_t data_len) -> bool {
                received += data_len;
                if (received >= k_abort_after_bytes) {
                    abort_triggered = true;
                    return false;  // tear down the socket
                }
                return true;
            };

            const httplib::Headers headers;
            auto resp = cli.Post(
                "/completion", headers, req_str,
                "application/json", std::move(receiver));

            // The client should either return !resp (Error::Canceled
            // / Error::Read) or a truncated 200. Both indicate the
            // server began streaming and the connection tore down.
            // We refuse only the case where the receiver never fired.
            if (!abort_triggered && received == 0) {
                return drain_engine_and_fail(
                    "client receiver never saw any bytes — server "
                    "may not have started streaming");
            }

            if (resp) {
                if (resp->status != 200) {
                    char buf[256];
                    std::snprintf(buf, sizeof(buf),
                        "disconnect POST returned non-200 "
                        "status=%d body=%s",
                        resp->status, resp->body.c_str());
                    return drain_engine_and_fail(buf);
                }
            } else {
                const httplib::Error err = resp.error();
                if (err != httplib::Error::Canceled
                 && err != httplib::Error::Read
                 && err != httplib::Error::Write) {
                    char buf[200];
                    std::snprintf(buf, sizeof(buf),
                        "disconnect POST unexpected error=%d",
                        static_cast<int>(err));
                    return drain_engine_and_fail(buf);
                }
            }
        }

        // ---- Phase 2: sanity non-streaming POST, server alive ------
        {
            httplib::Client cli("127.0.0.1", port);
            cli.set_read_timeout(60, 0);
            cli.set_write_timeout(30, 0);

            nlohmann::json req_body;
            req_body["prompt"]        = "Hello, my name is";
            req_body["decode_budget"] = k_sanity_decode_budget;
            // omit "stream" → non-streaming greedy

            auto resp = cli.Post("/completion", req_body.dump(),
                                 "application/json");
            if (!resp) {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                    "sanity POST failed (error=%d)",
                    static_cast<int>(resp.error()));
                return drain_engine_and_fail(buf);
            }
            if (resp->status != 200) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "sanity POST status=%d (expected 200) body=%s",
                    resp->status, resp->body.c_str());
                return drain_engine_and_fail(buf);
            }
            nlohmann::json out;
            try {
                out = nlohmann::json::parse(resp->body);
            } catch (const std::exception & e) {
                return drain_engine_and_fail(
                    std::string("sanity body parse failed: ")
                  + e.what());
            }
            if (!out.contains("hash")
             || !out["hash"].is_string()
             || out["hash"].get<std::string>()
                  != std::string(k_canonical_greedy_b8)) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "sanity hash=%s != canonical %s",
                    out.contains("hash") && out["hash"].is_string()
                        ? out["hash"].get<std::string>().c_str()
                        : "<missing>",
                    k_canonical_greedy_b8);
                return drain_engine_and_fail(buf);
            }
            if (!out.contains("n_decoded")
             || out["n_decoded"].get<int>() != 8) {
                return drain_engine_and_fail(
                    "sanity n_decoded != 8");
            }
        }

        srv.stop();
        listen_fut.get();
        eng.request_shutdown();
        engine_fut.get();

        // ---- Phase 3: engine counter assertions -------------------
        const engine_result & er = eng.result();
        if (er.cancel_request_calls < 1) {
            char buf[200];
            std::snprintf(buf, sizeof(buf),
                "cancel_request_calls=%d (expected >= 1)",
                er.cancel_request_calls);
            return fail_and_cleanup(buf);
        }
        if (er.cancel_unknown_request_id != 0) {
            char buf[200];
            std::snprintf(buf, sizeof(buf),
                "cancel_unknown_request_id=%d (expected 0)",
                er.cancel_unknown_request_id);
            return fail_and_cleanup(buf);
        }
        if (er.decode_failures != 0) {
            char buf[200];
            std::snprintf(buf, sizeof(buf),
                "decode_failures=%d (expected 0)",
                er.decode_failures);
            return fail_and_cleanup(buf);
        }
        if (!er.residual_kv_ok) {
            return fail_and_cleanup(
                "residual_kv_ok == false (expected true)");
        }

        fprintf(stdout,
            "cancel_request_calls=%d cancel_unknown_request_id=%d "
            "cancel_request_duplicates=%d decode_failures=%d "
            "residual_kv_ok=%d\n",
            er.cancel_request_calls,
            er.cancel_unknown_request_id,
            er.cancel_request_duplicates,
            er.decode_failures,
            er.residual_kv_ok ? 1 : 0);
        emit_pass();
    } catch (const std::exception & e) {
        return fail_and_cleanup(
            std::string("exception: ") + e.what());
    } catch (...) {
        return fail_and_cleanup("unknown exception");
    }

    return cleanup_and_stop(0);
}
