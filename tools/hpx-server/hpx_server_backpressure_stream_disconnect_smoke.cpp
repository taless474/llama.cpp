// hpx_server_backpressure_stream_disconnect_smoke.cpp — M7f
// composition smoke for llama-hpx-server.
//
// Composes M7d (SSE client-disconnect → cancel) and M7e (HTTP-door
// capacity lease). Proves that a streaming request which the client
// aborts mid-response (a) holds capacity for the full SSE lifecycle,
// (b) releases capacity exactly once after the disconnect / cancel /
// drain finalize path has fully unwound, and (c) the server still
// produces a byte-correct canonical greedy decode on the next
// admitted request.
//
// Sequence:
//   1. Streaming holder POST (decode_budget=64) launched via
//      hpx::async; ContentReceiver returns false after ~64 bytes →
//      cpp-httplib tears down the socket → server-side disconnect
//      detection fires (in-provider sink.write returns false OR the
//      ContentProviderResourceReleaser runs from ~Response). Both
//      paths share the same finalize routine: cancel via
//      h.cancel(eng), drain the stream channel, consume the result
//      future. The lease, moved into `stream_state` before
//      set_chunked_content_provider returned, releases when the
//      shared_ptr<stream_state> refcount drops to zero (after both
//      the provider lambda and the releaser lambda have released
//      their captures, which happens during ~Response, AFTER the
//      releaser body has run).
//   2. While the holder is past try_acquire (observed via
//      cap.in_flight >= 1, the M7e determinism anchor), a
//      non-streaming challenger POST is rejected with HTTP 503 /
//      server_busy.
//   3. Once cap.in_flight == 0 (proving the lease released), a
//      serial canonical non-streaming POST returns HTTP 200 with
//      the budget-8 greedy canonical hash 0x0619d4d1900c2365.
//
// Engine-side asserts mirror the M7d disconnect smoke:
//   cancel_request_calls    >= 1
//   cancel_unknown_request_id == 0
//   decode_failures         == 0
//   residual_kv_ok          == true
//
// Capacity-side asserts mirror the M7e backpressure smoke:
//   accepted        >= 2     (holder + serial canonical)
//   rejected_503    >= 1     (challenger)
//   peak            == 1
//   in_flight_final == 0
//
// HPX-nativity boundary (mirrors hpx-server.cpp):
//   cpp-httplib is the explicit non-HPX network adapter boundary;
//   this TU contains no `std::thread`, `std::mutex`,
//   `std::condition_variable`, or `std::this_thread::sleep_for`.
//   Holder concurrency uses `hpx::async`; the capacity-wait loop
//   uses `hpx::this_thread::yield` from an HPX-thread context
//   acquired via `hpx::async(...).get()` (the M7e lesson). The
//   in-flight capacity counter is a `std::atomic<int32_t>` at the
//   adapter boundary, following the same precedent as `next_rid`
//   and the M7e counter in `hpx-server.cpp`. The handler is
//   inlined here so the smoke is self-contained AND so the smoke
//   driver can observe `cap` directly.

#include "common.h"
#include "llama.h"

#include "engine.h"
#include "hpx_runtime.h"
#include "trace.h"
#include "types.h"

#include <hpx/hpx.hpp>

#include <cpp-httplib/httplib.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
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
        "  M7f HPX server backpressure stream-disconnect smoke.\n",
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
    fprintf(stdout,
        "HPX_SERVER_BACKPRESSURE_STREAM_DISCONNECT_SMOKE: FAIL: %s\n",
        reason.c_str());
    fflush(stdout);
}

void emit_pass() {
    fprintf(stdout,
        "HPX_SERVER_BACKPRESSURE_STREAM_DISCONNECT_SMOKE: PASS\n");
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

// ---- M7e capacity types (mirror of hpx-server.cpp) ---------------------
struct capacity {
    std::atomic<int32_t> in_flight{0};
    int32_t              max          = 0;
    std::atomic<int32_t> peak{0};
    std::atomic<int64_t> accepted{0};
    std::atomic<int64_t> rejected_503{0};
};

struct capacity_lease {
    capacity * cap      = nullptr;
    bool       released = true;

    capacity_lease() = default;
    explicit capacity_lease(capacity & c) noexcept
        : cap(&c), released(false) {}

    capacity_lease(const capacity_lease &)             = delete;
    capacity_lease & operator=(const capacity_lease &) = delete;

    capacity_lease(capacity_lease && o) noexcept
        : cap(o.cap), released(o.released) {
        o.cap      = nullptr;
        o.released = true;
    }
    capacity_lease & operator=(capacity_lease && o) noexcept {
        if (this != &o) {
            release();
            cap        = o.cap;
            released   = o.released;
            o.cap      = nullptr;
            o.released = true;
        }
        return *this;
    }

    ~capacity_lease() { release(); }

    void release() noexcept {
        if (cap != nullptr && !released) {
            cap->in_flight.fetch_sub(1, std::memory_order_acq_rel);
            released = true;
        }
    }
};

capacity_lease try_acquire(capacity & cap) {
    int32_t cur = cap.in_flight.load(std::memory_order_acquire);
    while (true) {
        if (cur >= cap.max) {
            cap.rejected_503.fetch_add(1, std::memory_order_relaxed);
            return capacity_lease{};
        }
        if (cap.in_flight.compare_exchange_weak(
                cur, cur + 1,
                std::memory_order_acq_rel,
                std::memory_order_acquire)) {
            const int32_t new_val = cur + 1;
            int32_t p = cap.peak.load(std::memory_order_relaxed);
            while (p < new_val
                && !cap.peak.compare_exchange_weak(
                       p, new_val,
                       std::memory_order_relaxed)) {}
            cap.accepted.fetch_add(1, std::memory_order_relaxed);
            return capacity_lease{cap};
        }
    }
}

// Bounded HPX-native poll for a predicate over the capacity counter.
// Returns true on success, false on deadline expiry. The poll body
// uses `hpx::this_thread::yield`, which REQUIRES an HPX thread
// context — calling it on the OS main thread aborts with
// `null_thread_id`. To stay HPX-native from any caller, the loop is
// wrapped in `hpx::async(...).get()` so it always runs on an HPX
// worker (the M7e lesson, ported verbatim).
template <typename Pred>
bool wait_for_capacity(capacity & cap, Pred pred,
                       std::chrono::seconds timeout) {
    return hpx::async([&cap, pred, timeout]() {
        const auto deadline =
            std::chrono::steady_clock::now() + timeout;
        while (!pred(cap.in_flight.load(std::memory_order_acquire))) {
            if (std::chrono::steady_clock::now() > deadline) {
                return false;
            }
            hpx::this_thread::yield();
        }
        return true;
    }).get();
}

}  // namespace

int main(int argc, char ** argv) {
    smoke_args args;
    if (!parse_args(argc, argv, args)) return 2;

    trace::init();

    // HPX worker budget: engine_fut + listen_fut + holder_fut each
    // hold a worker; wait_for_capacity wraps run briefly on a fourth.
    // 4 matches the M7e smoke.
    if (!hpx_runtime::start_once(/*os_threads=*/4)) {
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
    ctx_params.n_seq_max       = 1;
    ctx_params.n_threads       = 2;
    ctx_params.n_threads_batch = 2;
    ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        return fail_and_cleanup("context create failed");
    }

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);

    constexpr int32_t k_n_seq_max         = 1;
    constexpr int32_t k_max_prompt_tokens = 512;
    constexpr int32_t k_batch_capacity    =
        k_n_seq_max * k_max_prompt_tokens;
    constexpr int32_t k_max_concurrent    = 1;
    constexpr int32_t k_holder_budget     = 64;
    constexpr int32_t k_challenger_budget = 8;
    constexpr int32_t k_canonical_budget  = 8;
    constexpr size_t  k_abort_after_bytes = 64;
    constexpr const char * k_canonical_greedy_b8 =
        "0x0619d4d1900c2365";
    constexpr const char * k_prompt = "Hello, my name is";

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
        capacity cap;
        cap.max = k_max_concurrent;

        httplib::Server srv;
        // Handler = union of:
        //   - M7e backpressure handler (try_acquire / lease / 503)
        //   - M7d disconnect handler (streaming + in-provider fail
        //     handling + ContentProviderResourceReleaser cancel/drain)
        // Inlined so the smoke is self-contained AND the smoke driver
        // can observe `cap` directly.
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

            // M7e door reservation: validation first, capacity next,
            // tokenize + submit_request last.
            capacity_lease lease = try_acquire(cap);
            if (lease.cap == nullptr) {
                res.set_header("Retry-After", "1");
                send_error(res, 503, "server_busy",
                    "server is at capacity");
                return;
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
                submit_handle  h;
                bool           sink_alive        = true;
                bool           cancel_issued     = false;
                bool           result_consumed   = false;
                bool           completed_cleanly = false;
                // M7f: capacity lease moved here once we are committed
                // to the SSE path. ~capacity_lease fires when the
                // shared_ptr<stream_state> refcount drops to zero —
                // after both the chunked-content provider lambda and
                // the resource-releaser lambda have released their
                // captures, which happens during ~Response, AFTER the
                // releaser body has run its cancel/drain finalize.
                capacity_lease lease;
            };
            auto state   = std::make_shared<stream_state>();
            state->h     = std::move(h);
            state->lease = std::move(lease);

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
                            // In-provider disconnect: sink.write
                            // reported peer is gone. Finalize inline.
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

        // ---- Phase 1: streaming holder occupies the slot ----------
        // The holder is a SSE POST that the client aborts after
        // ~64 bytes via ContentReceiver returning false. budget=64
        // gives a wide streaming window (~700ms) so the abort lands
        // during decode rather than after natural completion.
        hpx::future<httplib::Result> holder_fut =
            hpx::async([port, holder_budget = k_holder_budget]() {
                httplib::Client cli("127.0.0.1", port);
                cli.set_read_timeout(60, 0);
                cli.set_write_timeout(30, 0);
                nlohmann::json req_body;
                req_body["prompt"]        = "Hello, my name is";
                req_body["decode_budget"] = holder_budget;
                req_body["stream"]        = true;
                const std::string req_str = req_body.dump();
                size_t received        = 0;
                bool   abort_triggered = false;
                httplib::ContentReceiver receiver =
                    [&](const char * /*data*/, size_t data_len) -> bool {
                    received += data_len;
                    if (received >= k_abort_after_bytes) {
                        abort_triggered = true;
                        return false;
                    }
                    return true;
                };
                const httplib::Headers headers;
                auto resp = cli.Post(
                    "/completion", headers, req_str,
                    "application/json", std::move(receiver));
                // `abort_triggered` is for diagnostics only; the
                // returned Result already captures the outcome.
                (void)abort_triggered;
                return resp;
            });

        // ---- Determinism anchor: holder is past try_acquire -------
        if (!wait_for_capacity(cap,
                [](int32_t v) { return v >= 1; },
                std::chrono::seconds(30))) {
            return drain_engine_and_fail(
                "holder never occupied capacity within 30s");
        }

        // ---- Phase 1 (cont): challenger must be rejected ----------
        {
            httplib::Client cli("127.0.0.1", port);
            cli.set_read_timeout(60, 0);
            cli.set_write_timeout(30, 0);
            nlohmann::json req_body;
            req_body["prompt"]        = k_prompt;
            req_body["decode_budget"] = k_challenger_budget;
            auto resp = cli.Post("/completion", req_body.dump(),
                                 "application/json");
            if (!resp) {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                    "challenger POST failed (error=%d)",
                    static_cast<int>(resp.error()));
                return drain_engine_and_fail(buf);
            }
            if (resp->status != 503) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "challenger status=%d (expected 503) body=%s",
                    resp->status, resp->body.c_str());
                return drain_engine_and_fail(buf);
            }
            const std::string retry_after =
                resp->get_header_value("Retry-After");
            if (retry_after != "1") {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "challenger Retry-After=\"%s\" (expected \"1\")",
                    retry_after.c_str());
                return drain_engine_and_fail(buf);
            }
            nlohmann::json err;
            try {
                err = nlohmann::json::parse(resp->body);
            } catch (const std::exception & e) {
                return drain_engine_and_fail(
                    std::string("challenger body parse failed: ")
                  + e.what());
            }
            if (!err.is_object()
             || !err.contains("error")
             || !err["error"].is_object()) {
                return drain_engine_and_fail(
                    std::string("challenger body missing error: ")
                  + resp->body);
            }
            const std::string code =
                err["error"].value("code", std::string());
            if (code != "server_busy") {
                return drain_engine_and_fail(
                    std::string("challenger error.code=\"")
                  + code
                  + "\" (expected \"server_busy\") body="
                  + resp->body);
            }
            const std::string message =
                err["error"].value("message", std::string());
            if (message != "server is at capacity") {
                return drain_engine_and_fail(
                    std::string("challenger error.message=\"")
                  + message
                  + "\" (expected \"server is at capacity\") body="
                  + resp->body);
            }
        }

        // ---- Phase 1 (cont): observe holder tear-down -------------
        // The disconnecting client either sees no Result (with
        // Error::Canceled / Read / Write) or a truncated 200 — both
        // are acceptable per M7d convention.
        {
            auto resp = holder_fut.get();
            if (resp) {
                if (resp->status != 200) {
                    char buf[256];
                    std::snprintf(buf, sizeof(buf),
                        "holder disconnect POST returned non-200 "
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
                        "holder disconnect POST unexpected error=%d",
                        static_cast<int>(err));
                    return drain_engine_and_fail(buf);
                }
            }
        }

        // ---- Phase 1 (cont): wait for the lease to release --------
        // After ~Response runs, both lambda captures drop their
        // shared_ptr<stream_state>, the refcount hits zero,
        // ~stream_state runs, ~capacity_lease decrements in_flight.
        // The 30s deadline is purely defensive.
        if (!wait_for_capacity(cap,
                [](int32_t v) { return v == 0; },
                std::chrono::seconds(30))) {
            return drain_engine_and_fail(
                "capacity never returned to 0 within 30s after "
                "streaming disconnect");
        }

        // ---- Phase 2: serial canonical follow-up ------------------
        {
            httplib::Client cli("127.0.0.1", port);
            cli.set_read_timeout(60, 0);
            cli.set_write_timeout(30, 0);
            nlohmann::json req_body;
            req_body["prompt"]        = k_prompt;
            req_body["decode_budget"] = k_canonical_budget;
            auto resp = cli.Post("/completion", req_body.dump(),
                                 "application/json");
            if (!resp) {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                    "serial canonical POST failed (error=%d)",
                    static_cast<int>(resp.error()));
                return drain_engine_and_fail(buf);
            }
            if (resp->status != 200) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "serial canonical status=%d (expected 200) "
                    "body=%s",
                    resp->status, resp->body.c_str());
                return drain_engine_and_fail(buf);
            }
            nlohmann::json out;
            try {
                out = nlohmann::json::parse(resp->body);
            } catch (const std::exception & e) {
                return drain_engine_and_fail(
                    std::string("serial canonical body parse failed: ")
                  + e.what());
            }
            if (!out.contains("hash")
             || !out["hash"].is_string()
             || out["hash"].get<std::string>()
                  != std::string(k_canonical_greedy_b8)) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "serial canonical hash=%s != %s",
                    out.contains("hash") && out["hash"].is_string()
                        ? out["hash"].get<std::string>().c_str()
                        : "<missing>",
                    k_canonical_greedy_b8);
                return drain_engine_and_fail(buf);
            }
            if (!out.contains("n_decoded")
             || out["n_decoded"].get<int>() != k_canonical_budget) {
                return drain_engine_and_fail(
                    "serial canonical n_decoded != 8");
            }
        }

        srv.stop();
        listen_fut.get();
        eng.request_shutdown();
        engine_fut.get();

        // ---- Engine cancellation counter asserts (M7d style) ------
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

        // ---- Capacity counter asserts (M7e style) -----------------
        const int64_t accepted_final =
            cap.accepted.load(std::memory_order_relaxed);
        const int64_t rejected_final =
            cap.rejected_503.load(std::memory_order_relaxed);
        const int32_t peak_final =
            cap.peak.load(std::memory_order_relaxed);
        const int32_t in_flight_final =
            cap.in_flight.load(std::memory_order_acquire);
        if (accepted_final < 2) {
            char buf[200];
            std::snprintf(buf, sizeof(buf),
                "cap.accepted=%lld (expected >= 2)",
                static_cast<long long>(accepted_final));
            return fail_and_cleanup(buf);
        }
        if (rejected_final < 1) {
            char buf[200];
            std::snprintf(buf, sizeof(buf),
                "cap.rejected_503=%lld (expected >= 1)",
                static_cast<long long>(rejected_final));
            return fail_and_cleanup(buf);
        }
        if (peak_final != 1) {
            char buf[200];
            std::snprintf(buf, sizeof(buf),
                "cap.peak=%d (expected 1)", peak_final);
            return fail_and_cleanup(buf);
        }
        if (in_flight_final != 0) {
            char buf[200];
            std::snprintf(buf, sizeof(buf),
                "cap.in_flight_final=%d (expected 0)",
                in_flight_final);
            return fail_and_cleanup(buf);
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
        fprintf(stdout,
            "cap accepted=%lld rejected_503=%lld peak=%d "
            "in_flight_final=%d\n",
            static_cast<long long>(accepted_final),
            static_cast<long long>(rejected_final),
            peak_final, in_flight_final);
        emit_pass();
    } catch (const std::exception & e) {
        return fail_and_cleanup(
            std::string("exception: ") + e.what());
    } catch (...) {
        return fail_and_cleanup("unknown exception");
    }

    return cleanup_and_stop(0);
}
