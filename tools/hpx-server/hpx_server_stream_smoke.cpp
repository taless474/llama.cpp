// hpx_server_stream_smoke.cpp — in-process SSE round-trip smoke for
// llama-hpx-server. Covers the cumulative-delta streaming
// detokenization contract.
//
// Boots model + engine + cpp-httplib server in-process. For each of
// two canonical TinyLlama greedy shapes — `p0_b8` (budget=8,
// hash=0x0619d4d1900c2365) and `p0_b16` (budget=16,
// hash=0x833045f1e2ebf49f) — drives one streaming `POST /completion`
// and one direct same-shape non-streaming `eng.submit_request`
// (in-process engine API; no second HTTP path), then asserts:
//   - HTTP 200, `Content-Type: text/event-stream` on the streaming
//     response,
//   - every non-terminal SSE record is `event: token` with `data:`
//     containing an integer `token_id` and a string `token` (the
//     cumulative-detokenization delta — may be ""),
//   - prefix stability across the streamed prefix: for every i,
//     `common_detokenize(vocab, streamed_token_ids[0..i], false)`
//     starts with the previously assembled cumulative text and
//     extends it by exactly the i-th `data["token"]` delta,
//   - in-stream identity: assembled streamed text equals
//     `common_detokenize(vocab, streamed_token_ids, false)`,
//   - same-shape equality: assembled streamed text equals
//     `common_detokenize(vocab, r.generated_tokens, false)`
//     byte-for-byte against the direct engine `request_result`,
//   - token-id equality: streamed `token_id` sequence equals the
//     direct engine `r.generated_tokens` (deterministic greedy →
//     exact equality),
//   - terminal SSE record is `event: done` with
//     `status:"completed"`, `n_decoded == decode_budget`, and
//     `hash == canonical` for the shape; the engine `request_result`
//     hash also equals the canonical.
//
// HPX-nativity boundary (mirrors hpx-server.cpp):
//   cpp-httplib is the explicit non-HPX network adapter boundary; this
//   TU contains no `std::thread`, `std::mutex`,
//   `std::condition_variable`, or `std::this_thread::sleep_for`. The
//   chunked-content-provider lambda runs on a cpp-httplib worker
//   thread and only:
//     - reads `token_stream_event` from `submit_handle.stream` via
//       `get(hpx::launch::sync)` (foreign-thread channel wait, same
//       shared-state-cv mechanism the M7a path already exercises),
//     - calls `common_detokenize(const llama_vocab *, ...)` against
//       the immutable post-load vocab pointer,
//     - writes bytes through `httplib::DataSink::write`,
//     - awaits `submit_handle.result` once after `kind == closed`.
//   It never touches `llama_context *`, `llama_decode`,
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
        "  M7c HPX server SSE round-trip smoke.\n",
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
    fprintf(stdout, "HPX_SERVER_STREAM_SMOKE: FAIL: %s\n",
            reason.c_str());
    fflush(stdout);
}

void emit_pass() {
    fprintf(stdout, "HPX_SERVER_STREAM_SMOKE: PASS\n");
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

// Minimal SSE record splitter: scans `body` and yields each non-empty
// SSE record (separated by a blank line, i.e. `\n\n`). Each record is
// returned with its trailing terminator stripped.
std::vector<std::string> split_sse_records(const std::string & body) {
    std::vector<std::string> out;
    size_t start = 0;
    while (start < body.size()) {
        const size_t sep = body.find("\n\n", start);
        const size_t end = (sep == std::string::npos)
                           ? body.size() : sep;
        if (end > start) {
            out.emplace_back(body.substr(start, end - start));
        }
        if (sep == std::string::npos) break;
        start = sep + 2;
    }
    return out;
}

// Per-record line scan: extracts the `event:` and `data:` line values.
// Leading whitespace after the `:` is trimmed (single space is the
// SSE-conventional separator).
struct sse_record {
    std::string event_name;  // e.g. "token", "done"
    std::string data;        // raw payload after `data: `
};

sse_record parse_sse_record(const std::string & raw) {
    sse_record r;
    size_t line_start = 0;
    while (line_start < raw.size()) {
        const size_t eol = raw.find('\n', line_start);
        const size_t line_end = (eol == std::string::npos)
                                ? raw.size() : eol;
        std::string line = raw.substr(line_start, line_end - line_start);
        // Trim a possible \r from CRLF (cpp-httplib emits LF-only,
        // but defensive parsing is cheap).
        if (!line.empty() && line.back() == '\r') line.pop_back();
        auto stripped_value = [](const std::string & line,
                                 size_t prefix_len) -> std::string {
            std::string v = line.substr(prefix_len);
            if (!v.empty() && v.front() == ' ') v.erase(0, 1);
            return v;
        };
        if (line.rfind("event:", 0) == 0) {
            r.event_name = stripped_value(line, 6);
        } else if (line.rfind("data:", 0) == 0) {
            // SSE allows multiple `data:` lines per record; we
            // concatenate them with `\n` per spec. Single-line is the
            // only shape M7c emits, but defensive concat is cheap.
            const std::string v = stripped_value(line, 5);
            if (r.data.empty()) r.data = v;
            else { r.data += "\n"; r.data += v; }
        }
        if (eol == std::string::npos) break;
        line_start = eol + 1;
    }
    return r;
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

    constexpr const char * k_canonical_greedy_b8 =
        "0x0619d4d1900c2365";
    constexpr const char * k_canonical_greedy_b16 =
        "0x833045f1e2ebf49f";

    try {
        engine eng(std::move(opts));

        hpx::future<void> engine_fut =
            hpx::async([&] { eng.run(); });

        std::atomic<int32_t> next_rid{1};

        httplib::Server srv;
        // Handler mirrors the SSE branch of hpx-server.cpp. Inlined
        // here for the smoke to be self-contained (no shared TU
        // with the production server).
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
                bool          sink_alive = true;
                // Cumulative-delta streaming detokenization state,
                // mirroring hpx-server.cpp. Touched only from the
                // single cpp-httplib worker that owns this response's
                // provider lambda.
                std::vector<llama_token> emitted_tokens;
                std::string              emitted_text;
            };
            auto state = std::make_shared<stream_state>();
            state->h   = std::move(h);

            res.status = 200;
            res.set_chunked_content_provider(
                "text/event-stream",
                [state, vocab](size_t /*offset*/,
                               httplib::DataSink & sink) -> bool {
                    token_stream_event ev;
                    try {
                        ev = state->h.stream->get(hpx::launch::sync);
                    } catch (const std::exception &) {
                        if (state->sink_alive) sink.done();
                        return false;
                    }

                    if (ev.kind == stream_event_kind::token) {
                        if (!state->sink_alive) return true;
                        // Cumulative-detokenization delta scheme,
                        // mirroring hpx-server.cpp. The cumulative
                        // `common_detokenize` owns every
                        // tokenizer-specific rule; the handler only
                        // emits the new visible byte-substring.
                        state->emitted_tokens.push_back(ev.token_id);
                        std::string full = common_detokenize(
                            vocab, state->emitted_tokens,
                            /*special=*/false);
                        const bool prefix_ok =
                            full.size() >= state->emitted_text.size()
                         && std::memcmp(
                                full.data(),
                                state->emitted_text.data(),
                                state->emitted_text.size()) == 0;
                        if (!prefix_ok) {
                            // Mirror the production fail-closed
                            // intent: stop emitting and let the
                            // client-side prefix-stability gate
                            // surface the failure. The smoke handler
                            // does not exercise the full M7d
                            // cancel/drain routine, so we simply
                            // mark the sink dead and return false;
                            // the missing `done` record will trip
                            // the outer assertions.
                            fprintf(stderr,
                                "[smoke handler] "
                                "stream_detokenize_prefix_violation"
                                " request=%d token_id=%d"
                                " emitted_bytes=%zu"
                                " new_full_bytes=%zu\n",
                                state->h.token.request_id,
                                ev.token_id,
                                state->emitted_text.size(),
                                full.size());
                            state->sink_alive = false;
                            return false;
                        }
                        const std::string delta =
                            full.substr(state->emitted_text.size());
                        state->emitted_text = std::move(full);
                        nlohmann::json data;
                        data["token"]    = delta;
                        data["token_id"] = ev.token_id;
                        std::string payload =
                            "event: token\ndata: " + data.dump()
                          + "\n\n";
                        if (!sink.write(payload.data(),
                                        payload.size())) {
                            state->sink_alive = false;
                        }
                        return true;
                    }

                    request_result r;
                    bool result_ok = true;
                    try {
                        r = state->h.result.get();
                    } catch (const std::exception &) {
                        result_ok = false;
                    }
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
                    return false;
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

        httplib::Client cli("127.0.0.1", port);
        cli.set_read_timeout(60, 0);
        cli.set_write_timeout(30, 0);

        auto end_test_cleanup = [&]() {
            srv.stop();
            listen_fut.get();
            eng.request_shutdown();
            engine_fut.get();
        };

        auto cleanup_and_fail_round =
            [&](const std::string & reason) -> int {
            end_test_cleanup();
            return fail_and_cleanup(reason);
        };

        struct round_spec {
            int32_t      decode_budget;
            const char * canonical_hash;
            const char * label;
        };
        const round_spec rounds[] = {
            { 8,  k_canonical_greedy_b8,  "p0_b8"  },
            { 16, k_canonical_greedy_b16, "p0_b16" },
        };

        // Per-round: HTTP streaming POST → walk SSE records with the
        // prefix-stability + in-stream-identity assertions; then a
        // direct in-process engine non-streaming submit for the same
        // shape to obtain the authoritative `r.generated_tokens` and
        // `r.hash`, against which the streamed token-id sequence and
        // assembled streamed text are asserted byte-for-byte.
        auto do_round = [&](const round_spec & rs) -> std::string {
            const std::string prompt = "Hello, my name is";

            nlohmann::json sreq;
            sreq["prompt"]        = prompt;
            sreq["decode_budget"] = rs.decode_budget;
            sreq["stream"]        = true;
            auto sresp = cli.Post("/completion", sreq.dump(),
                                  "application/json");
            if (!sresp) {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                    "streaming client.Post failed (error=%d)",
                    static_cast<int>(sresp.error()));
                return buf;
            }
            if (sresp->status != 200) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "streaming HTTP status=%d (expected 200), body=%s",
                    sresp->status, sresp->body.c_str());
                return buf;
            }
            const std::string ctype =
                sresp->get_header_value("Content-Type");
            if (ctype.rfind("text/event-stream", 0) != 0) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "Content-Type='%s' (expected prefix "
                    "text/event-stream)", ctype.c_str());
                return buf;
            }

            const std::vector<std::string> records =
                split_sse_records(sresp->body);
            if (records.empty()) {
                return "SSE body parsed to zero records";
            }

            std::vector<llama_token> streamed_token_ids;
            std::string              assembled;
            size_t                   token_event_count = 0;

            for (size_t i = 0; i + 1 < records.size(); i++) {
                const sse_record rec = parse_sse_record(records[i]);
                if (rec.event_name != "token") {
                    char buf[256];
                    std::snprintf(buf, sizeof(buf),
                        "record[%zu] event='%s' (expected 'token')",
                        i, rec.event_name.c_str());
                    return buf;
                }
                nlohmann::json data;
                try {
                    data = nlohmann::json::parse(rec.data);
                } catch (const std::exception & e) {
                    char buf[256];
                    std::snprintf(buf, sizeof(buf),
                        "record[%zu] data parse failed: %s",
                        i, e.what());
                    return buf;
                }
                if (!data.contains("token_id")
                 || !data["token_id"].is_number_integer()) {
                    char buf[200];
                    std::snprintf(buf, sizeof(buf),
                        "record[%zu] missing integer 'token_id'", i);
                    return buf;
                }
                if (!data.contains("token")
                 || !data["token"].is_string()) {
                    char buf[200];
                    std::snprintf(buf, sizeof(buf),
                        "record[%zu] missing string 'token'", i);
                    return buf;
                }
                const int32_t     tid   =
                    data["token_id"].get<int32_t>();
                const std::string delta =
                    data["token"].get<std::string>();

                // Client-side prefix-stability gate:
                //   common_detokenize(vocab, ids[0..=i], false) must
                //   start with `assembled` and extend it by exactly
                //   the wire `delta`. Either failure implies a
                //   handler bug or a tokenizer edge case the slice
                //   has not characterized.
                streamed_token_ids.push_back(tid);
                const std::string local_full = common_detokenize(
                    vocab, streamed_token_ids, /*special=*/false);
                if (local_full.size() < assembled.size()
                 || std::memcmp(local_full.data(), assembled.data(),
                                assembled.size()) != 0) {
                    char buf[320];
                    std::snprintf(buf, sizeof(buf),
                        "prefix-stability violation at record[%zu] "
                        "token_id=%d: local_full(len=%zu) does not "
                        "start with assembled(len=%zu)",
                        i, tid, local_full.size(),
                        assembled.size());
                    return buf;
                }
                const size_t expected_delta_len =
                    local_full.size() - assembled.size();
                if (delta.size() != expected_delta_len
                 || std::memcmp(
                        local_full.data() + assembled.size(),
                        delta.data(), delta.size()) != 0) {
                    char buf[320];
                    std::snprintf(buf, sizeof(buf),
                        "delta mismatch at record[%zu] token_id=%d: "
                        "wire_delta(len=%zu) != local_full tail "
                        "(len=%zu)",
                        i, tid, delta.size(), expected_delta_len);
                    return buf;
                }
                assembled += delta;
                token_event_count++;
            }

            if (token_event_count == 0) {
                return "no SSE token records";
            }

            // In-stream identity gate.
            {
                const std::string final_full = common_detokenize(
                    vocab, streamed_token_ids, /*special=*/false);
                if (assembled != final_full) {
                    return "assembled streamed text != "
                           "common_detokenize(streamed_token_ids)";
                }
            }

            // done record
            const sse_record done_rec =
                parse_sse_record(records.back());
            if (done_rec.event_name != "done") {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                    "terminal event='%s' (expected 'done')",
                    done_rec.event_name.c_str());
                return buf;
            }
            nlohmann::json done_data;
            try {
                done_data = nlohmann::json::parse(done_rec.data);
            } catch (const std::exception & e) {
                return std::string("done.data parse failed: ")
                     + e.what();
            }
            if (!done_data.contains("status")
             || !done_data["status"].is_string()
             || done_data["status"].get<std::string>()
                  != "completed") {
                return "done.status != \"completed\"";
            }
            if (!done_data.contains("n_decoded")
             || !done_data["n_decoded"].is_number_integer()
             || done_data["n_decoded"].get<int>()
                  != rs.decode_budget) {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                    "done.n_decoded=%d != decode_budget=%d",
                    done_data.contains("n_decoded")
                     && done_data["n_decoded"].is_number_integer()
                       ? done_data["n_decoded"].get<int>()
                       : -1,
                    rs.decode_budget);
                return buf;
            }
            if (!done_data.contains("hash")
             || !done_data["hash"].is_string()
             || done_data["hash"].get<std::string>()
                  != std::string(rs.canonical_hash)) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "done.hash=%s != canonical %s",
                    done_data.contains("hash")
                      && done_data["hash"].is_string()
                        ? done_data["hash"]
                              .get<std::string>().c_str()
                        : "<missing>",
                    rs.canonical_hash);
                return buf;
            }

            // Direct in-process engine non-streaming submit for the
            // same shape. This gives us the authoritative
            // `r.generated_tokens` and `r.hash` against which the
            // streamed token-id sequence and assembled text are
            // checked. Not routed through HTTP — the non-streaming
            // HTTP path is already covered by hpx_server_smoke; this
            // smoke isolates the streaming contract.
            std::vector<llama_token> ref_prompt_tokens =
                common_tokenize(vocab, prompt,
                                /*add_special=*/true,
                                /*parse_special=*/true);
            if (ref_prompt_tokens.empty()) {
                return "ref tokenization produced zero tokens";
            }

            submit_request ref_req;
            ref_req.request_id    = -10000 - rs.decode_budget;
            ref_req.prompt_tokens = std::move(ref_prompt_tokens);
            ref_req.decode_budget = rs.decode_budget;
            ref_req.want_stream   = false;
            submit_handle ref_h =
                eng.submit_request(std::move(ref_req));
            request_result r;
            try {
                r = ref_h.result.get();
            } catch (const std::exception & e) {
                return std::string(
                    "direct engine submit failed: ") + e.what();
            }
            if (r.status != request_status::completed) {
                return "direct engine r.status != completed";
            }
            if (r.n_decoded != rs.decode_budget) {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                    "direct engine r.n_decoded=%d != %d",
                    r.n_decoded, rs.decode_budget);
                return buf;
            }
            if (hex_hash(r.hash) != std::string(rs.canonical_hash)) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "direct engine r.hash=%s != canonical %s",
                    hex_hash(r.hash).c_str(), rs.canonical_hash);
                return buf;
            }
            if (streamed_token_ids != r.generated_tokens) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "streamed_token_ids (n=%zu) != "
                    "r.generated_tokens (n=%zu)",
                    streamed_token_ids.size(),
                    r.generated_tokens.size());
                return buf;
            }
            const std::string ref_text = common_detokenize(
                vocab, r.generated_tokens, /*special=*/false);
            if (assembled != ref_text) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "assembled(len=%zu) != "
                    "common_detokenize(r.generated_tokens) (len=%zu)",
                    assembled.size(), ref_text.size());
                return buf;
            }

            fprintf(stdout,
                "[%s] records=%zu token_events=%zu "
                "assembled_bytes=%zu done.hash=%s "
                "ref_n_decoded=%d ref_hash=%s\n",
                rs.label,
                records.size(), token_event_count,
                assembled.size(),
                done_data["hash"].get<std::string>().c_str(),
                r.n_decoded, hex_hash(r.hash).c_str());

            return std::string();  // success
        };

        for (const auto & rs : rounds) {
            const std::string reason = do_round(rs);
            if (!reason.empty()) {
                return cleanup_and_fail_round(
                    std::string(rs.label) + ": " + reason);
            }
        }

        emit_pass();
        end_test_cleanup();
    } catch (const std::exception & e) {
        return fail_and_cleanup(
            std::string("exception: ") + e.what());
    } catch (...) {
        return fail_and_cleanup("unknown exception");
    }

    return cleanup_and_stop(0);
}
