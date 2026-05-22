// hpx-server.cpp — M7a/M7b/M7c/M7d: minimal HPX-driven HTTP adapter
// for the llama-hpx-engine library.
//
// Surface: one endpoint, POST /completion. M7b adds optional nested
// "sampling"; M7c adds an optional top-level "stream":
//   { "prompt": "...", "decode_budget": <int>,
//     "stream": <bool>,                            // M7c, optional
//     "sampling": { "mode": "greedy"|"stochastic",
//                   "seed": <uint32>,
//                   "temperature": <float>,
//                   "top_k": <int32>,
//                   "top_p": <float>,
//                   "top_p_min_keep": <uint32> } } // M7b, optional
// Omitted/false "stream" => existing M7b non-streaming JSON response.
// Omitted "sampling" => greedy defaults. Invalid sampling => HTTP 422
// `invalid_sampling` via the shared `validate_sampling_config` helper.
//
// Non-streaming response (unchanged from M7a/M7b):
//   { "request_id": <int>, "status": "completed", "n_decoded": <int>,
//     "hash": "0x...", "text": "..." }
//
// Streaming response (M7c, when "stream": true):
//   Content-Type: text/event-stream, transfer-encoding: chunked.
//   One SSE record per emitted token followed by one terminal `done`:
//     event: token
//     data: {"token":"<best-effort utf8>","token_id":<int>}
//
//     event: done
//     data: {"request_id":<int>,"status":"<completed|cancelled|
//             failed_reserved>","n_decoded":<int>,"hash":"0x..."}
//   Per-token text is best-effort: a single-piece `common_detokenize`
//   may return "" when a multi-byte codepoint splits across pieces.
//   `token_id` is the reliable field in M7c (no UTF-8 chunk buffering
//   yet).
//
//   M7d wires client-disconnect cancellation through two cooperating
//   detection paths and one shared finalize routine:
//
//   (a) In-provider detection: when `DataSink::write` returns false
//       inside our chunked-content provider (peer reset surfaced
//       through the active write), the provider performs the
//       finalize routine inline and returns false.
//
//   (b) Out-of-band detection via `ContentProviderResourceReleaser`:
//       cpp-httplib's chunked write loop also checks
//       `strm.is_peer_alive()` BEFORE re-invoking the provider; on a
//       peer disconnect between provider invocations the loop exits
//       with `Error::Write` and the provider is never called again.
//       To catch that case we register a resource releaser (3rd arg
//       to `set_chunked_content_provider`) which fires from
//       `~Response` regardless of how the loop ended and re-runs the
//       same finalize routine. It is idempotent via per-request
//       flags.
//
//   The finalize routine:
//     1. marks the sink dead,
//     2. issues `submit_handle::cancel(engine &)` once (fire-and-
//        forget; idempotent on the engine side), guarded by a per-
//        request `cancel_issued` bool so the engine's duplicate
//        counter stays at zero,
//     3. drains the per-request stream channel in-place until
//        `stream_event_kind::closed`,
//     4. awaits `submit_handle.result` once (guarded by
//        `result_consumed`) to observe the engine-final snapshot —
//        cancellation observed by the engine resolves the promise to
//        `request_status::cancelled`; a race where the sequence
//        completes before the cancel lands resolves to `completed`,
//        both are acceptable. M7d only asserts that cancellation was
//        issued on disconnect and the engine remained healthy.
//
//   No `done` SSE record is written on the disconnect path; the
//   client cannot read it. Engine cancellation semantics and stream-
//   close ownership are unchanged — the engine task remains the sole
//   owner of `llama_*` state, KV mutation, channel close, and promise
//   fulfillment.
//
// M7e adds a single in-flight cap at the cpp-httplib adapter boundary,
// strictly before tokenize / submit_request:
//   --max-concurrent N    default = --n-seq-max
// `try_acquire` does a lock-free CAS-bump-if-below-cap on a single
// `std::atomic<int32_t> in_flight`. On success the handler holds an
// RAII `capacity_lease` that decrements exactly once on destruction.
// On failure the handler short-circuits with HTTP 503 + `Retry-After:
// 1` and the body:
//   { "error": { "code": "server_busy",
//                "message": "server is at capacity" } }
// Non-streaming: the lease is a local in the handler, so every exit
// path (200, 4xx, 5xx, exception) releases capacity. Streaming: the
// lease is moved into the `stream_state` shared_ptr so capacity is
// held for the full SSE lifecycle, including the M7d disconnect/cancel
// finalize path. When both the chunked-content provider lambda and the
// `ContentProviderResourceReleaser` lambda have released their
// captures, the shared_ptr refcount drops to zero, `stream_state`
// destructs, and the lease releases. No engine / types.h / gate
// changes; canonical greedy 0x0619d4d1900c2365 and stochastic seed=42
// 0xa8e14acb4094aa3f are preserved on the admitted path.
//
// HPX-nativity boundary (CRITICAL — keep this comment current):
//   cpp-httplib is NOT HPX-native. It spawns its own std::thread
//   workers internally to accept and dispatch HTTP requests. M7a
//   accepts this as a deliberate, explicit boundary on the network
//   edge. The invariant we still enforce in our own sources is:
//     - our files introduce NO std::thread / std::mutex /
//       std::condition_variable / std::this_thread::sleep_for;
//     - HPX continues to own the request lifecycle and engine
//       control plane (engine_fut runs eng.run() on hpx::async);
//     - cpp-httplib worker threads MUST NOT touch llama.cpp mutable
//       execution state. The only llama.cpp calls allowed from a
//       handler thread are common_tokenize / common_detokenize via
//       the `const llama_vocab *` overloads (vocab is immutable
//       post-load) — the handler never holds a llama_context * and
//       never reaches llama_decode / llama_get_logits_ith /
//       llama_sampler_* / llama_memory_seq_* / KV APIs;
//     - request_id is minted from a std::atomic<int32_t> (same
//       pattern the engine already uses for cancel_requested; not
//       a new synchronization primitive);
//     - M7e adds an `std::atomic<int32_t> in_flight` capacity
//       counter at the same adapter boundary, plus three relaxed-
//       atomic observational counters (peak, accepted, rejected_503).
//       Lock-free CAS-bump on acquire; idempotent fetch_sub on
//       lease release. No new synchronization primitive;
//     - no POSIX signal handler is installed in M7a — server
//       shutdown is driven by srv.stop() via normal control flow
//       (see the two server smokes for the smoke-driven pattern).
//
// Foreign-thread future wait:
//   The /completion handler runs on a cpp-httplib worker (a foreign
//   thread relative to HPX). It calls submit_request() — which only
//   pushes onto the engine inbox under hpx::spinlock and returns a
//   submit_handle whose `result` is an hpx::future<request_result>
//   — and then blocks on `h.result.get()` from that foreign thread.
//   HPX supports foreign-thread waits on shared futures by parking
//   the calling thread on the shared-state cv until the engine task
//   fulfils the promise. If this regresses on a future HPX version
//   the fallback documented in the M7a plan is `hpx::async([f =
//   std::move(h.result)]() mutable { return f.get(); }).get()`.

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
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

struct server_args {
    std::string model_path;
    std::string host              = "127.0.0.1";
    int32_t     port              = 8089;
    int32_t     n_seq_max         = 2;
    int32_t     max_prompt_tokens = 512;
    int32_t     n_threads         = 2;
    int32_t     n_ctx             = 2048;
    // M7e: HTTP-layer in-flight cap. 0 here is the "unset" sentinel;
    // `parse_args` normalizes a zero/negative value to n_seq_max at the
    // end of argv parsing so the default tracks the engine concurrency.
    int32_t     max_concurrent    = 0;
};

void print_usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s --model <path> [options]\n"
        "  --model <path>            (required) path to .gguf model\n"
        "  --host <addr>             default: 127.0.0.1\n"
        "  --port <int>              default: 8089\n"
        "  --n-seq-max <int>         default: 2\n"
        "  --max-prompt-tokens <int> default: 512\n"
        "  --n-threads <int>         default: 2  (libllama compute)\n"
        "  --max-concurrent <int>    default: --n-seq-max  (M7e: HTTP-layer in-flight cap)\n"
        "  --ctx-size <int>          default: 2048  (llama_context n_ctx)\n",
        argv0);
}

bool parse_args(int argc, char ** argv, server_args & out) {
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto need_value = [&](const char * flag) {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: %s requires a value\n", flag);
                return false;
            }
            return true;
        };
        if (a == "--model") {
            if (!need_value("--model")) return false;
            out.model_path = argv[++i];
        } else if (a == "--host") {
            if (!need_value("--host")) return false;
            out.host = argv[++i];
        } else if (a == "--port") {
            if (!need_value("--port")) return false;
            out.port = std::atoi(argv[++i]);
        } else if (a == "--n-seq-max") {
            if (!need_value("--n-seq-max")) return false;
            out.n_seq_max = std::atoi(argv[++i]);
        } else if (a == "--max-prompt-tokens") {
            if (!need_value("--max-prompt-tokens")) return false;
            out.max_prompt_tokens = std::atoi(argv[++i]);
        } else if (a == "--n-threads") {
            if (!need_value("--n-threads")) return false;
            out.n_threads = std::atoi(argv[++i]);
        } else if (a == "--max-concurrent") {
            if (!need_value("--max-concurrent")) return false;
            out.max_concurrent = std::atoi(argv[++i]);
        } else if (a == "--ctx-size") {
            if (!need_value("--ctx-size")) return false;
            out.n_ctx = std::atoi(argv[++i]);
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
    if (out.n_seq_max < 1) {
        fprintf(stderr, "error: --n-seq-max must be >= 1\n");
        return false;
    }
    if (out.max_prompt_tokens < 1) {
        fprintf(stderr, "error: --max-prompt-tokens must be >= 1\n");
        return false;
    }
    if (out.n_ctx < 1) {
        fprintf(stderr, "error: --ctx-size must be >= 1\n");
        return false;
    }
    // M7e: zero/negative => default to n_seq_max (one HTTP slot per
    // engine slot, no HTTP-side queueing). The strict default keeps
    // M7e to "prove serving admission"; HTTP-side queueing is a later
    // explicit feature.
    if (out.max_concurrent <= 0) {
        out.max_concurrent = out.n_seq_max;
    }
    return true;
}

// Error response helper. Format mirrors a typical compact JSON-API
// error body so callers can branch on `error.code`.
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

// M7b: JSON-string → sampling_mode. JSON-edge concern; intentionally
// not promoted into types.h. Returns false on unknown strings; the
// caller surfaces a 422 with a descriptive message.
bool parse_sampling_mode(const std::string & s, sampling_mode & out) {
    if (s == "greedy")     { out = sampling_mode::greedy;     return true; }
    if (s == "stochastic") { out = sampling_mode::stochastic; return true; }
    return false;
}

// ---- M7e: HTTP-layer capacity accounting -------------------------------
// Single in-flight cap applied at the cpp-httplib adapter boundary,
// strictly before tokenize / submit_request. `in_flight` is the live
// counter; `max` is the cap; the remaining three are observational
// (peak + cumulative accepted / rejected). All four are `std::atomic`
// at the adapter boundary only — see the file-top HPX-nativity block.
// Engine control-plane primitives are unchanged.
struct capacity {
    std::atomic<int32_t> in_flight{0};
    int32_t              max          = 0;
    std::atomic<int32_t> peak{0};
    std::atomic<int64_t> accepted{0};
    std::atomic<int64_t> rejected_503{0};
};

// RAII release token returned by `try_acquire`. Move-only; the
// destructor calls `fetch_sub` exactly once (idempotent via the
// `released` flag). A default-constructed lease is disengaged
// (`cap == nullptr`) and a no-op on destruction — that is the
// representation `try_acquire` returns on a saturated cap.
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

// Lock-free CAS-bump-if-below-cap. On success returns an engaged
// lease and bumps `peak` (best-effort, relaxed) + `accepted`. On
// failure returns a disengaged lease and bumps `rejected_503`. No
// blocking, no system call; safe on cpp-httplib worker threads.
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

}  // namespace

int main(int argc, char ** argv) {
    server_args args;
    if (!parse_args(argc, argv, args)) return 2;

    trace::init();

    // os_threads=2 leaves one HPX worker for the engine task and one
    // for other HPX work (e.g. promise fulfilment continuations).
    // cpp-httplib's internal worker pool is separate and not driven
    // by these.
    if (!hpx_runtime::start_once(/*os_threads=*/2)) {
        fprintf(stderr, "error: hpx runtime start failed\n");
        return 1;
    }

    common_init();
    llama_backend_init();

    llama_model *   model = nullptr;
    llama_context * ctx   = nullptr;

    auto cleanup = [&](int rc) -> int {
        if (ctx)   llama_free(ctx);
        if (model) llama_model_free(model);
        llama_backend_free();
        hpx_runtime::stop();
        return rc;
    };

    llama_model_params model_params = llama_model_default_params();
    model = llama_model_load_from_file(args.model_path.c_str(),
                                       model_params);
    if (model == nullptr) {
        fprintf(stderr, "error: model load failed\n");
        return cleanup(1);
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx           = args.n_ctx;
    ctx_params.n_batch         = 1024;
    ctx_params.n_seq_max       = args.n_seq_max;
    ctx_params.n_threads       = args.n_threads;
    ctx_params.n_threads_batch = args.n_threads;
    ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        fprintf(stderr, "error: context create failed\n");
        return cleanup(1);
    }

    // Vocab is captured ONCE at boot and held for the entire server
    // lifetime. cpp-httplib handler threads receive only this
    // const-pointer — `ctx` is never visible outside main() and the
    // engine task.
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);

    const int32_t batch_capacity =
        args.n_seq_max * args.max_prompt_tokens;

    std::vector<waiting_request> empty_waiting;

    engine_options opts;
    opts.lib.ctx                  = ctx;
    opts.lib.vocab                = vocab;
    opts.lib.n_vocab              = n_vocab;
    opts.lib.batch_capacity       = batch_capacity;
    opts.lib.n_seq_max            = args.n_seq_max;
    opts.lib.initial_idle_slots   = args.n_seq_max;
    opts.lib.keep_alive           = true;
    opts.preload.prompt_tokens    = nullptr;
    opts.preload.budgets          = {};
    opts.preload.waiting_queue    = &empty_waiting;
    opts.preload.reuse_completed  = false;
    opts.preload.stream_all       = false;
    opts.gate_test.cancel_after     = -1;
    opts.gate_test.max_decode_iters = 0;

    int rc = 0;
    try {
        engine eng(std::move(opts));

        hpx::future<void> engine_fut =
            hpx::async([&] { eng.run(); });

        std::atomic<int32_t> next_rid{1};

        // M7e: HTTP-layer in-flight cap. Captured by reference into
        // the POST handler lambda along with `next_rid` and `args`.
        // Lifetime: this scope, which encloses srv.listen() — the
        // server is guaranteed to be stopped (srv.stop() inside a
        // smoke driver or external SIGINT) before `cap` goes out of
        // scope. `in_flight` is guaranteed to be zero at that point
        // because every accepted request has either finished (lease
        // released on handler exit / stream_state destruction) or
        // never acquired (try_acquire failed).
        capacity cap;
        cap.max = args.max_concurrent;

        httplib::Server srv;

        // Handler runs on a cpp-httplib worker thread. The four
        // allowed operations: JSON parse, common_{tokenize,
        // detokenize}(const llama_vocab *, ...), submit_request,
        // hpx::future::get. Nothing else.
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
             || !body["prompt"].is_string()) {
                send_error(res, 400, "bad_request",
                    "missing or non-string 'prompt'");
                return;
            }
            if (!body.contains("decode_budget")
             || !body["decode_budget"].is_number_integer()) {
                send_error(res, 400, "bad_request",
                    "missing or non-integer 'decode_budget'");
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

            // ---- M7c: optional top-level "stream" boolean ---------
            // Omitted or false => existing M7b non-streaming JSON
            // response (byte-identical). True => SSE chunked branch
            // wired via `submit_request.want_stream = true` and
            // `submit_handle.stream`. Type errors => 400 bad_request.
            bool stream_requested = false;
            if (body.contains("stream")) {
                if (!body["stream"].is_boolean()) {
                    send_error(res, 400, "bad_request",
                        "'stream' must be a boolean");
                    return;
                }
                stream_requested = body["stream"].get<bool>();
            }

            // ---- M7b: optional nested "sampling" parsing ----------
            // Omitted "sampling" => greedy defaults (M7a contract).
            // Type errors on individual fields => 422 invalid_sampling
            // (parse-edge). Value-range failures => 422
            // invalid_sampling via the shared validate_sampling_config
            // helper. Bad-shape "sampling" => 400 bad_request.
            sampling_config cfg;
            if (body.contains("sampling")) {
                const auto & s = body["sampling"];
                if (!s.is_object()) {
                    send_error(res, 400, "bad_request",
                        "'sampling' must be an object");
                    return;
                }
                if (s.contains("mode")) {
                    if (!s["mode"].is_string()
                     || !parse_sampling_mode(
                            s["mode"].get<std::string>(), cfg.mode)) {
                        send_error(res, 422, "invalid_sampling",
                            "mode must be 'greedy' or 'stochastic'");
                        return;
                    }
                }
                if (s.contains("seed")) {
                    if (!s["seed"].is_number_unsigned()) {
                        send_error(res, 422, "invalid_sampling",
                            "seed must be a non-negative integer");
                        return;
                    }
                    const uint64_t v = s["seed"].get<uint64_t>();
                    if (v > std::numeric_limits<uint32_t>::max()) {
                        send_error(res, 422, "invalid_sampling",
                            "seed exceeds uint32_t range");
                        return;
                    }
                    cfg.seed = static_cast<uint32_t>(v);
                }
                if (s.contains("temperature")) {
                    if (!s["temperature"].is_number()) {
                        send_error(res, 422, "invalid_sampling",
                            "temperature must be a number");
                        return;
                    }
                    cfg.temperature = s["temperature"].get<float>();
                }
                if (s.contains("top_k")) {
                    if (!s["top_k"].is_number_integer()) {
                        send_error(res, 422, "invalid_sampling",
                            "top_k must be an integer");
                        return;
                    }
                    cfg.top_k = s["top_k"].get<int32_t>();
                }
                if (s.contains("top_p")) {
                    if (!s["top_p"].is_number()) {
                        send_error(res, 422, "invalid_sampling",
                            "top_p must be a number");
                        return;
                    }
                    cfg.top_p = s["top_p"].get<float>();
                }
                if (s.contains("top_p_min_keep")) {
                    if (!s["top_p_min_keep"].is_number_unsigned()) {
                        send_error(res, 422, "invalid_sampling",
                            "top_p_min_keep must be a non-negative "
                            "integer");
                        return;
                    }
                    const uint64_t v =
                        s["top_p_min_keep"].get<uint64_t>();
                    if (v > std::numeric_limits<uint32_t>::max()) {
                        send_error(res, 422, "invalid_sampling",
                            "top_p_min_keep exceeds uint32_t range");
                        return;
                    }
                    cfg.top_p_min_keep = static_cast<uint32_t>(v);
                }

                std::string verr;
                if (!validate_sampling_config(cfg, verr)) {
                    send_error(res, 422, "invalid_sampling", verr);
                    return;
                }
            }

            // ---- M7e: HTTP-door capacity reservation --------------
            // Acquired AFTER JSON parse + shape + sampling validation
            // (cheap, no engine touch) so malformed requests do not
            // consume a slot, and BEFORE tokenize + submit_request so
            // the engine inbox / waiting queue cannot grow unbounded
            // from concurrent clients. The lease is a stack local on
            // the non-streaming path (releases on handler exit via
            // RAII, including the exception path); on the streaming
            // path it is moved into `stream_state` and released only
            // after the SSE lifecycle — including the M7d disconnect/
            // cancel finalize path — has fully unwound.
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
                  > args.max_prompt_tokens) {
                char msg[160];
                std::snprintf(msg, sizeof(msg),
                    "prompt tokenized to %d tokens, exceeds "
                    "--max-prompt-tokens=%d",
                    static_cast<int>(tokens.size()),
                    args.max_prompt_tokens);
                send_error(res, 413, "payload_too_large", msg);
                return;
            }

            const int32_t rid =
                next_rid.fetch_add(1, std::memory_order_relaxed);

            submit_request sr;
            sr.request_id    = rid;
            sr.prompt_tokens = std::move(tokens);
            sr.decode_budget = decode_budget;
            sr.want_stream   = stream_requested;
            sr.sampling      = std::move(cfg);

            submit_handle h = eng.submit_request(std::move(sr));

            if (!stream_requested) {
                // ---- M7b non-streaming path (byte-identical) -------
                // Foreign-thread wait on the engine-owned future.
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

            // ---- M7c streaming path (SSE / chunked) ----------------
            // The engine guarantees `submit_handle.stream` is engaged
            // whenever `want_stream=true`; the guard below is a
            // belt-and-suspenders fail-closed check.
            if (!h.stream.has_value()) {
                send_error(res, 500, "engine_error",
                    "engine did not provide a stream channel for "
                    "want_stream=true");
                return;
            }

            // The chunked-content provider lambda is invoked
            // repeatedly by a cpp-httplib worker thread after this
            // handler returns. The `submit_handle` is move-only, so we
            // wrap it in a `shared_ptr<stream_state>` and capture the
            // pointer by value into the lambda — this extends the
            // handle's lifetime past the handler's return and through
            // every provider invocation. No new synchronization
            // primitive is introduced (shared_ptr is ownership, not
            // sync). Foreign-thread `h.stream->get(hpx::launch::sync)`
            // mirrors the M7a foreign-thread `h.result.get()` pattern;
            // both park on the shared-state cv inside the HPX local
            // channel / future. The handler thread never touches
            // llama_context, llama_decode, llama_get_logits_ith,
            // llama_sampler_*, llama_memory_seq_*, or KV APIs — only
            // `common_detokenize(const llama_vocab *, ...)`, which
            // operates against the immutable post-load vocab pointer.
            // M7d: per-request flags. All three are accessed only
            // from a single cpp-httplib worker thread at a time
            // (the chunked-provider loop is single-threaded per
            // response, and the resource releaser fires from
            // `~Response` AFTER the loop has exited). No
            // synchronization primitive is needed.
            //
            //   sink_alive       — false once a sink.write returned
            //                      false (peer reset surfaced through
            //                      the write path) OR the resource
            //                      releaser has fired with non-clean
            //                      end.
            //   cancel_issued    — true after the first
            //                      submit_handle::cancel(eng) call,
            //                      so a second call site is a no-op.
            //   result_consumed  — true after submit_handle.result
            //                      has been awaited once.
            //   completed_cleanly — true iff the closed branch of the
            //                      provider ran to completion
            //                      (including sink.done()). The
            //                      releaser uses this — not the
            //                      `success` arg cpp-httplib passes —
            //                      because the provider returns
            //                      `false` from the closed branch to
            //                      terminate the SSE stream, which
            //                      makes cpp-httplib's success arg
            //                      `false` even on the happy path.
            struct stream_state {
                submit_handle h;
                bool          sink_alive        = true;
                bool          cancel_issued     = false;
                bool          result_consumed   = false;
                bool          completed_cleanly = false;
                // M7e: capacity lease moved here once we are committed
                // to the SSE path. Destructor releases capacity exactly
                // once when the shared_ptr<stream_state> refcount drops
                // to zero — i.e. after both the chunked-content
                // provider lambda and the resource-releaser lambda
                // have released their captures (which happens during
                // ~Response, AFTER cpp-httplib has invoked the releaser
                // body so the M7d cancel/drain finalize routine has
                // already run).
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
                        // Receiver closed unexpectedly (engine
                        // shutdown / channel error). Best-effort
                        // terminate the SSE stream; do NOT issue a
                        // cancel here — the engine is already past
                        // promise/stream finalization.
                        if (state->sink_alive) {
                            sink.done();
                        }
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
                            // ---- M7d (in-provider disconnect) ------
                            // sink.write reported the peer is gone.
                            // Finalize inline in case cpp-httplib's
                            // chunked write loop chooses not to call
                            // us again.
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

                    // kind == closed (happy path). Drain the final
                    // snapshot and emit the terminal `done` SSE
                    // record. Per M7d rules, a failed final `done`
                    // write does NOT issue a cancel — the request has
                    // already completed.
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
                // M7d resource releaser: fires from `~Response`
                // regardless of how `write_content_with_provider`
                // ended. If the chunked write loop exited between
                // provider invocations (e.g. cpp-httplib observed
                // `!strm.is_peer_alive()` before our provider's next
                // call), neither the token branch nor the closed
                // branch above will have had a chance to issue the
                // cancel / drain — so we run the finalize routine
                // here. Idempotent via `cancel_issued`,
                // `result_consumed`, and `completed_cleanly`.
                [state, &eng](bool /*success*/) {
                    if (state->completed_cleanly) {
                        return;
                    }
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

        // listen() blocks the calling thread (main) until somebody
        // calls srv.stop(). M7a has no signal handler; an operator
        // running the binary by hand terminates it externally. The
        // server smokes drive srv.stop() through normal control
        // flow from a separate task — see hpx_server_smoke.cpp.
        fprintf(stderr,
            "[hpx-server] listening on %s:%d  "
            "(n_seq_max=%d max_prompt_tokens=%d batch_capacity=%d "
            "max_concurrent=%d n_ctx=%d)\n",
            args.host.c_str(), args.port,
            args.n_seq_max, args.max_prompt_tokens, batch_capacity,
            args.max_concurrent, args.n_ctx);
        fflush(stderr);
        const bool ok = srv.listen(args.host, args.port);
        if (!ok) {
            fprintf(stderr,
                "[hpx-server] listen() returned false "
                "(bind/listen failed?)\n");
            rc = 1;
        }

        eng.request_shutdown();
        engine_fut.get();
    } catch (const std::exception & e) {
        fprintf(stderr, "[hpx-server] fatal: %s\n", e.what());
        rc = 1;
    } catch (...) {
        fprintf(stderr, "[hpx-server] fatal: unknown exception\n");
        rc = 1;
    }

    return cleanup(rc);
}
