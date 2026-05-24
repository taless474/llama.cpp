// hpx_server_engine_pool_smoke.cpp — N4 in-process placement smoke for
// llama-hpx-server.
//
// Same in-process server/engine harness as `hpx_server_smoke.cpp` and
// `hpx_server_stream_disconnect_smoke.cpp`, with one difference: the
// HPX runtime is started with the named single-PU "engine" thread pool
// (`hpx_runtime::start_once(os_threads=2,
// {enable_engine_pool=true})`), the engine option
// `cooperative_yield_on_pump` is forced false (mirrors the gate's
// --engine-pool wiring; the yield is load-bearing on default-pool
// placement but causes a scheduler livelock on a dedicated single-PU
// named pool — N2.6c evidence), and the engine HPX task is spawned via
// `hpx_runtime::async_on_engine(...)` instead of bare `hpx::async`.
//
// Phase 1: canonical greedy round-trip. Drives one non-streaming
//   POST /completion ("Hello, my name is", decode_budget=8) and
//   asserts HTTP 200 + canonical greedy hash 0x0619d4d1900c2365 +
//   non-empty detokenized text. Proves engine-pool placement does not
//   break the basic HTTP path.
//
// Phase 2: SSE client-disconnect. Drives a streaming POST that the
//   client aborts mid-response via a ContentReceiver returning false
//   (~64 bytes received). Asserts either no Result or a truncated
//   200, then runs a sanity non-streaming POST that must return 200
//   with the canonical greedy hash — proves the server stayed alive
//   and the engine slot was recycled under engine-pool placement.
//
// Phase 3: engine counter assertions. After server stop and engine
//   drain, asserts cancel_request_calls >= 1,
//   cancel_unknown_request_id == 0, decode_failures == 0,
//   residual_kv_ok == true. Same shape as
//   hpx_server_stream_disconnect_smoke; proves the cancel path that
//   the gate's engine-pool queued-cancel smoke already validates
//   also works through the cpp-httplib adapter under engine-pool
//   placement.
//
// Placement evidence:
//   When run with LLAMA_HPX_PLACEMENT_TRACE=1, hpx_runtime emits
//     [hpx-cb-gate] engine_pool=created pool_name=engine engine_pus=1 default_pus=N
//     [hpx-cb-gate] engine_task_placement pool=engine
//   The validation script greps for the second line as positive
//   proof that the engine HPX task ran on the named pool. The smoke
//   itself does not require the env var to PASS.
//
// HPX-nativity boundary (mirrors hpx-server.cpp and the other server
// smokes): cpp-httplib is the explicit non-HPX network adapter
// boundary; this TU introduces no std::thread / std::mutex /
// std::condition_variable / std::this_thread::sleep_for. The handler
// thread only does JSON parse, common_{tokenize,detokenize}, push to
// engine inbox via submit_request, and blocks on hpx::future::get.

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
        "  N4 HPX server engine-pool placement smoke.\n",
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
    fprintf(stdout, "HPX_SERVER_ENGINE_POOL_SMOKE: FAIL: %s\n",
            reason.c_str());
    fflush(stdout);
}

void emit_pass() {
    fprintf(stdout, "HPX_SERVER_ENGINE_POOL_SMOKE: PASS\n");
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

    // N4: opt into the named single-PU "engine" thread pool. Requires
    // os_threads >= 2; hpx_runtime::start_once's preflight fails
    // closed otherwise (we pass 2 here, so the path is valid).
    hpx_runtime::runtime_config rt_cfg;
    rt_cfg.enable_engine_pool = true;
    if (!hpx_runtime::start_once(/*os_threads=*/2, rt_cfg)) {
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

    constexpr int32_t k_n_seq_max               = 2;
    constexpr int32_t k_max_prompt_tokens       = 512;
    constexpr int32_t k_batch_capacity          =
        k_n_seq_max * k_max_prompt_tokens;
    constexpr int32_t k_disconnect_decode_budget = 128;
    constexpr int32_t k_sanity_decode_budget     = 8;
    constexpr const char * k_canonical_greedy_b8 =
        "0x0619d4d1900c2365";

    std::vector<waiting_request> empty_waiting;

    engine_options opts;
    opts.lib.ctx                       = ctx;
    opts.lib.vocab                     = vocab;
    opts.lib.n_vocab                   = n_vocab;
    opts.lib.batch_capacity            = k_batch_capacity;
    opts.lib.n_seq_max                 = k_n_seq_max;
    opts.lib.initial_idle_slots        = k_n_seq_max;
    opts.lib.keep_alive                = true;
    // N4 / N2.7: required when the engine runs on the named single-PU
    // engine pool. Skipping the yield avoids the N2.6c scheduler
    // livelock; the engine still observes the channel via
    // pending_msg_.wait_for(0ns).
    opts.lib.cooperative_yield_on_pump = false;
    opts.preload.prompt_tokens         = nullptr;
    opts.preload.budgets               = {};
    opts.preload.waiting_queue         = &empty_waiting;
    opts.preload.reuse_completed       = false;
    opts.preload.stream_all            = false;
    opts.gate_test.cancel_after        = -1;
    opts.gate_test.max_decode_iters    = 0;

    try {
        engine eng(std::move(opts));

        // N4: spawn the engine HPX task on the named "engine" pool.
        // Throws std::runtime_error if the pool is unavailable; the
        // outer catch turns that into a fail-closed cleanup. No
        // promises or streams have been created yet.
        hpx::future<void> engine_fut =
            hpx_runtime::async_on_engine([&] { eng.run(); });

        std::atomic<int32_t> next_rid{1};

        httplib::Server srv;
        // Handler mirrors the M7d/M7a shape of hpx-server.cpp. Inlined
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

        // ---- Phase 1: canonical greedy round-trip ------------------
        {
            httplib::Client cli("127.0.0.1", port);
            cli.set_read_timeout(30, 0);
            cli.set_write_timeout(30, 0);

            nlohmann::json req_body;
            req_body["prompt"]        = "Hello, my name is";
            req_body["decode_budget"] = k_sanity_decode_budget;

            auto resp = cli.Post("/completion", req_body.dump(),
                                 "application/json");
            if (!resp) {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                    "round-trip POST failed (error=%d)",
                    static_cast<int>(resp.error()));
                return drain_engine_and_fail(buf);
            }
            if (resp->status != 200) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "round-trip POST status=%d (expected 200) body=%s",
                    resp->status, resp->body.c_str());
                return drain_engine_and_fail(buf);
            }
            nlohmann::json out;
            try {
                out = nlohmann::json::parse(resp->body);
            } catch (const std::exception & e) {
                return drain_engine_and_fail(
                    std::string("round-trip body parse failed: ")
                  + e.what());
            }
            if (!out.contains("n_decoded")
             || !out["n_decoded"].is_number_integer()
             || out["n_decoded"].get<int>() != 8) {
                return drain_engine_and_fail(
                    "round-trip n_decoded != 8");
            }
            if (!out.contains("hash") || !out["hash"].is_string()
             || out["hash"].get<std::string>()
                  != std::string(k_canonical_greedy_b8)) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "round-trip hash=%s != canonical %s",
                    out.contains("hash") && out["hash"].is_string()
                        ? out["hash"].get<std::string>().c_str()
                        : "<missing>",
                    k_canonical_greedy_b8);
                return drain_engine_and_fail(buf);
            }
            if (!out.contains("text") || !out["text"].is_string()
             || out["text"].get<std::string>().empty()) {
                return drain_engine_and_fail(
                    "round-trip text missing or empty");
            }
            if (!out.contains("status") || !out["status"].is_string()
             || out["status"].get<std::string>() != "completed") {
                return drain_engine_and_fail(
                    "round-trip status != completed");
            }
            fprintf(stdout,
                "phase1_round_trip: request_id=%d status=%s "
                "n_decoded=%d hash=%s\n",
                out["request_id"].get<int>(),
                out["status"].get<std::string>().c_str(),
                out["n_decoded"].get<int>(),
                out["hash"].get<std::string>().c_str());
        }

        // ---- Phase 2: streaming POST + mid-response client abort ---
        {
            httplib::Client cli("127.0.0.1", port);
            cli.set_read_timeout(60, 0);
            cli.set_write_timeout(30, 0);

            nlohmann::json req_body;
            req_body["prompt"]        = "Hello, my name is";
            req_body["decode_budget"] = k_disconnect_decode_budget;
            req_body["stream"]        = true;
            const std::string req_str = req_body.dump();

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
            fprintf(stdout,
                "phase2_disconnect: received=%zu abort_triggered=%d "
                "have_resp=%d\n",
                received, abort_triggered ? 1 : 0, resp ? 1 : 0);
        }

        // ---- Phase 3: sanity non-streaming POST, server alive ------
        {
            httplib::Client cli("127.0.0.1", port);
            cli.set_read_timeout(60, 0);
            cli.set_write_timeout(30, 0);

            nlohmann::json req_body;
            req_body["prompt"]        = "Hello, my name is";
            req_body["decode_budget"] = k_sanity_decode_budget;

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
            fprintf(stdout,
                "phase3_sanity: hash=%s n_decoded=%d\n",
                out["hash"].get<std::string>().c_str(),
                out["n_decoded"].get<int>());
        }

        srv.stop();
        listen_fut.get();
        eng.request_shutdown();
        engine_fut.get();

        // ---- Phase 4: engine counter assertions -------------------
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
            "phase4_counters: cancel_request_calls=%d "
            "cancel_unknown_request_id=%d "
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
