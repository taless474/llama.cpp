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
//     data: {"token":"<incremental utf8 delta>","token_id":<int>}
//
//     event: done
//     data: {"request_id":<int>,"status":"<completed|cancelled|
//             failed_reserved>","n_decoded":<int>,"hash":"0x..."}
//   `token` is the cumulative-detokenization *delta*: per stream we
//   keep `emitted_tokens` (running token-id vector) and `emitted_text`
//   (cumulative UTF-8 already written to the wire). On each engine
//   token event we append the new `token_id`, recompute
//   `full = common_detokenize(vocab, emitted_tokens, false)`, assert
//   the prefix-stability invariant `full.starts_with(emitted_text)`,
//   and emit `delta = full.substr(emitted_text.size())`. `delta` may
//   legitimately be "" when a Unicode codepoint splits across BPE /
//   byte-fallback tokens; the next token event delivers the assembled
//   bytes in one go. Concatenation of all `token` chunks across a
//   stream equals `common_detokenize(vocab, streamed_token_ids,
//   false)` for that same generated token sequence — an in-stream
//   identity by construction of the delta. In deterministic
//   same-shape smoke tests the assembled stream therefore matches
//   the same-shape non-streaming `text` field whenever token IDs /
//   anchors match; cross-request equality is not asserted in the
//   general case. The `token_id` field is unchanged and remains the
//   authoritative per-event identifier. On a prefix-stability violation the handler
//   fails closed (cancel + drain + no `done` record) rather than emit
//   corrupt streaming text.
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
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
    // Slice F Option B: per-seq prefill row cap forwarded to
    // engine_options::lib.prefill_budget_rows. Default 0 leaves the
    // engine in its current observable behavior (unbounded /
    // whole-prompt prefill in the live-admission build path). A value
    // > 0 enables the experimental HPX-owned chunked-prefill cap. This
    // is a diagnostic surface for the PrefillBudgetPolicy investigation
    // (see docs/hpx/prefill_budget_policy_design.md /
    // prefill_budget_policy_result.md); the slice does not recommend
    // any specific value.
    int32_t     prefill_budget_rows = 0;
    // N4: HPX worker count for hpx_runtime::start_once. Default 2
    // preserves M7a behavior byte-for-byte. Independent of --n-threads
    // (libllama compute). Must be >= 2 when --engine-pool is on; the
    // hpx_runtime::start_once preflight enforces that.
    int32_t     hpx_os_threads    = 2;
    // N4: opt into the single-PU named HPX "engine" thread pool created
    // by hpx_runtime::start_once. Default OFF — async_on_engine falls
    // through to bare hpx::async, byte-identical to M7a. ON requires
    // --hpx-os-threads >= 2.
    bool        engine_pool       = false;
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
        "  --ctx-size <int>          default: 2048  (llama_context n_ctx)\n"
        "  --prefill-budget-rows <int>  default: 0  (disabled / unbounded:\n"
        "                            whole-prompt prefill in the live-admission\n"
        "                            build path, current observable behavior).\n"
        "                            > 0 enables the experimental HPX prefill\n"
        "                            row cap (per-seq, per-iter) under the\n"
        "                            PrefillBudgetPolicy investigation. No value\n"
        "                            is recommended; provided as a diagnostic\n"
        "                            surface only.\n"
        "  --hpx-os-threads <int>    default: 2  (N4: HPX worker count;\n"
        "                            independent of --n-threads. Must be\n"
        "                            >= 2 when --engine-pool is on.)\n"
        "  --engine-pool             default: OFF  (N4: opt into the\n"
        "                            single-PU named HPX 'engine' thread\n"
        "                            pool. Engine HPX task runs on that\n"
        "                            pool; cpp-httplib workers and request\n"
        "                            futures stay on default. OFF preserves\n"
        "                            M7a spawn behavior byte-for-byte.)\n",
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
        } else if (a == "--prefill-budget-rows") {
            if (!need_value("--prefill-budget-rows")) return false;
            out.prefill_budget_rows = std::atoi(argv[++i]);
        } else if (a == "--hpx-os-threads") {
            if (!need_value("--hpx-os-threads")) return false;
            out.hpx_os_threads = std::atoi(argv[++i]);
        } else if (a == "--engine-pool") {
            // N4 boolean flag, no value. Validation against
            // --hpx-os-threads happens inside
            // hpx_runtime::start_once (requires os_threads >= 2).
            out.engine_pool = true;
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
    if (out.prefill_budget_rows < 0) {
        fprintf(stderr, "error: --prefill-budget-rows must be >= 0\n");
        return false;
    }
    if (out.n_ctx < 1) {
        fprintf(stderr, "error: --ctx-size must be >= 1\n");
        return false;
    }
    if (out.hpx_os_threads < 1) {
        fprintf(stderr, "error: --hpx-os-threads must be >= 1\n");
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

// ---- Phase 1 server-side serving-overhead diagnostics ------------------
// Default-OFF. With LLAMA_HPX_DIAG_METRICS unset, server_diag_enabled()
// returns false and every guarded site (chrono call, counter bump, JSON
// formatting, file I/O) is a single not-taken branch. Identical env-
// switch contract to the engine TU's diag_enabled / diag_path: same
// variable names, same semantics.
//
// Per-request state lives on the stack for the non-streaming path (a
// `request_metrics` POD local to the handler) and inside `stream_state`
// for the streaming path (one `request_metrics` field plus a
// `metrics_emitted` dedupe flag). No mutable file-scope state is
// introduced anywhere — the only file-scope additions below are the
// read-only env-switch caches and the pure dump helper.
//
// Emission policy: one JSONL row per request that reached
// `eng.submit_request`. Pre-submit failures (400/422/payload-too-large,
// 503 capacity-rejected, tokenization-empty) do NOT emit, because the
// row's submit_us / n_decoded fields would be undefined. Document this
// in the row schema rather than emit half-populated rows.
//
// Time axis: absolute steady_clock microseconds. Matches engine.cpp's
// `now_us()` helper used by Exp-13 responsiveness timing, so engine-
// side and server-side rows share a single process-level steady_clock
// epoch. Engine per-iter rows use a *relative* axis
// (t_us_from_engine_start_per_iter is delta from engine `t_start_`);
// cross-axis alignment is a Phase 2 detail and can be done later by
// stamping one engine-start absolute value into the engine_summary
// row (out of scope here — would require touching engine.cpp again).
bool server_diag_enabled() {
    static const bool v = []() -> bool {
        const char * e = std::getenv("LLAMA_HPX_DIAG_METRICS");
        return e != nullptr && std::strcmp(e, "1") == 0;
    }();
    return v;
}

const char * server_diag_path() {
    static const char * const v =
        std::getenv("LLAMA_HPX_DIAG_METRICS_PATH");
    return v;
}

// Phase 1 graceful-shutdown precursor. The real llama-hpx-server has no
// signal handler — `srv.listen()` blocks until something inside the
// process calls `srv.stop()`. Without that, killing the binary with
// SIGINT skips the post-listen cleanup (`eng.request_shutdown` +
// `engine_fut.get`), `engine::run()` never reaches its tail, and the
// engine's `dump_metrics_jsonl(result_)` site never fires — leaving
// every captured JSONL with `server_request` rows only and zero engine
// `iter` / `engine_summary` rows (the Phase 1 caveat surfaced in the
// real-server demo at local/runs/n6-phase1-diag/pass2-real-server/).
//
// This helper gates a diagnostic-only `POST /shutdown` endpoint behind
// two env vars in conjunction:
//   LLAMA_HPX_DIAG_METRICS         must be "1"
//   LLAMA_HPX_DIAG_ENABLE_SHUTDOWN must be "1"
//
// Both required: setting only DIAG_METRICS does NOT enable the
// endpoint. The two-key gate keeps the shutdown surface invisible to
// any production deployment that turns on metrics without explicitly
// opting into the operator-facing kill switch. The endpoint is
// intentionally NOT registered when this returns false — the route
// simply does not exist, so an unauthenticated POST gets the cpp-
// httplib default 404. No signal handler is installed; cpp-httplib's
// `srv.stop()` is called from the handler thread, which is the
// supported safe entry point.
bool server_diag_shutdown_enabled() {
    static const bool v = []() -> bool {
        if (!server_diag_enabled()) return false;
        const char * e = std::getenv("LLAMA_HPX_DIAG_ENABLE_SHUTDOWN");
        return e != nullptr && std::strcmp(e, "1") == 0;
    }();
    return v;
}

int64_t server_now_us() noexcept {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Per-request server-side metrics. POD-friendly. All fields default-
// init to safe sentinels so a partial row (e.g. failed_reserved before
// first token) is still well-formed JSONL. Used by both paths:
//   * non-streaming: stack-local in the handler;
//   * streaming: stored inside `stream_state` so the chunked-content
//     provider, the in-provider sad-path finalize, and the M7d
//     resource releaser all share the same accumulator.
//
// final_status is a pointer-to-static-string-literal (e.g.
// "completed", "cancelled", "failed_reserved", "engine_error",
// "stream_error", "prefix_violation"). No std::string allocation on
// the hot path. The status_name() return value used elsewhere in this
// TU is also a static string literal, so it is safe to copy into this
// pointer without lifetime worries.
struct request_metrics {
    int32_t      request_id              = -1;
    bool         stream                  = false;
    int64_t      submit_us               = 0;
    int64_t      first_token_us          = 0;
    int64_t      completion_us           = 0;
    int32_t      stream_channel_get_count = 0;
    int32_t      sse_write_count         = 0;
    int64_t      sse_write_bytes         = 0;
    bool         disconnect_observed     = false;
    bool         cancel_issued           = false;
    const char * final_status            = "unknown";
    int32_t      n_decoded               = 0;
};

// Emit one JSONL row to LLAMA_HPX_DIAG_METRICS_PATH (O_APPEND, line-
// buffered) when set, stderr otherwise. Bare fprintf (no nlohmann::json
// allocation on the hot path; mirrors engine.cpp's dump_metrics_jsonl).
// Caller must have already verified server_diag_enabled().
void dump_request_metrics_jsonl(const request_metrics & m) {
    const char * path = server_diag_path();
    FILE * fp = stderr;
    bool   need_close = false;
    if (path != nullptr && *path != '\0') {
        FILE * f = std::fopen(path, "a");
        if (f != nullptr) {
            std::setvbuf(f, nullptr, _IOLBF, 0);
            fp = f;
            need_close = true;
        }
    }
    std::fprintf(fp,
        "{\"kind\":\"server_request\","
        "\"request_id\":%d,"
        "\"stream\":%s,"
        "\"submit_us\":%lld,"
        "\"first_token_us\":%lld,"
        "\"completion_us\":%lld,"
        "\"stream_channel_get_count\":%d,"
        "\"sse_write_count\":%d,"
        "\"sse_write_bytes\":%lld,"
        "\"disconnect_observed\":%s,"
        "\"cancel_issued\":%s,"
        "\"final_status\":\"%s\","
        "\"n_decoded\":%d}\n",
        m.request_id,
        m.stream ? "true" : "false",
        static_cast<long long>(m.submit_us),
        static_cast<long long>(m.first_token_us),
        static_cast<long long>(m.completion_us),
        m.stream_channel_get_count,
        m.sse_write_count,
        static_cast<long long>(m.sse_write_bytes),
        m.disconnect_observed ? "true" : "false",
        m.cancel_issued ? "true" : "false",
        m.final_status != nullptr ? m.final_status : "unknown",
        m.n_decoded);
    if (need_close) {
        std::fclose(fp);
    } else {
        std::fflush(fp);
    }
}

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

    // os_threads >= 2 leaves one HPX worker for the engine task and at
    // least one for other HPX work (e.g. promise fulfilment
    // continuations). cpp-httplib's internal worker pool is separate
    // and not driven by these.
    //
    // N4: when --engine-pool is on, start_once installs an rp_callback
    // that creates a single-PU named "engine" thread pool. The
    // preflight inside start_once fails closed if os_threads < 2 while
    // engine_pool is requested, so the bad config never reaches
    // hpx::start.
    hpx_runtime::runtime_config rt_cfg;
    rt_cfg.enable_engine_pool = args.engine_pool;
    if (!hpx_runtime::start_once(args.hpx_os_threads, rt_cfg)) {
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
    // N4: disable the pump cooperativity yield when this engine will
    // run on the named single-PU engine pool. Mirrors the gate's
    // decision at the spawn site: yield is load-bearing on default
    // placement; it causes a scheduler livelock on a dedicated single-
    // PU named pool (N2.6c evidence).
    opts.lib.cooperative_yield_on_pump = !args.engine_pool;
    // Slice F Option B: forward the CLI-supplied prefill row cap to
    // the engine. Default 0 leaves the engine in its current
    // observable behavior (unbounded prefill). The engine ctor treats
    // any value <= 0 as unbounded (see Slice C invariant in
    // engine_options::lib.prefill_budget_rows).
    opts.lib.prefill_budget_rows  = args.prefill_budget_rows;
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

        // N4: async_on_engine spawns on the named "engine" pool when
        // --engine-pool was passed (and start_once created it); falls
        // through to bare hpx::async otherwise — byte-identical to
        // the legacy M7a spawn form. Throws std::runtime_error if the
        // engine pool was requested but is unavailable at spawn; the
        // surrounding catch (const std::exception &) sets rc=1 and
        // cleanup() tears everything down before any handler is
        // registered or any per-request promise/stream exists.
        hpx::future<void> engine_fut =
            hpx_runtime::async_on_engine([&] { eng.run(); });

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

            // Phase 1 server diagnostics: stack-local accumulator,
            // populated only when LLAMA_HPX_DIAG_METRICS=1. Submit
            // timestamp is captured immediately before submit_request
            // so it bookends the engine's view (the engine's per-iter
            // t_us_from_engine_start uses its own t_start_ baseline;
            // both ride the same process-level steady_clock epoch).
            // For the non-streaming branch this stays a stack local
            // released on handler return. For the streaming branch
            // the populated copy is moved into stream_state below so
            // the chunked provider, in-provider sad-path finalize, and
            // the M7d resource releaser can all update / emit it under
            // the same `metrics_emitted` dedupe flag.
            request_metrics rmet;
            if (server_diag_enabled()) {
                rmet.request_id = rid;
                rmet.stream     = stream_requested;
                rmet.submit_us  = server_now_us();
            }

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
                    if (server_diag_enabled()) {
                        rmet.completion_us = server_now_us();
                        rmet.final_status  = "engine_error";
                        dump_request_metrics_jsonl(rmet);
                    }
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
                if (server_diag_enabled()) {
                    rmet.completion_us = server_now_us();
                    rmet.final_status  = status_name(r.status);
                    rmet.n_decoded     = r.n_decoded;
                    if (r.request_id >= 0) rmet.request_id = r.request_id;
                    dump_request_metrics_jsonl(rmet);
                }
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
                // Cumulative-delta streaming detokenization state.
                // Both fields are touched only from the cpp-httplib
                // worker that owns this response's provider lambda
                // (single-threaded per response; the resource releaser
                // fires from ~Response strictly after the loop has
                // exited). No new synchronization primitive.
                std::vector<llama_token> emitted_tokens;
                std::string              emitted_text;
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
                // Phase 1 server diagnostics. Lives in stream_state
                // because four sites can finalize a streaming request:
                // (a) the closed branch (happy path); (b) the in-
                // provider prefix-violation finalize; (c) the in-
                // provider sink.write-returned-false finalize; (d) the
                // M7d resource releaser. metrics_emitted is the dedupe
                // flag — whichever site finishes first stamps rmet,
                // emits the JSONL row, and sets metrics_emitted=true;
                // subsequent sites skip emission. Both fields are
                // touched only from the cpp-httplib worker that owns
                // this response (provider loop is single-threaded per
                // response; the releaser fires strictly after the loop
                // has exited). No new synchronization primitive.
                request_metrics rmet;
                bool            metrics_emitted = false;
            };
            auto state   = std::make_shared<stream_state>();
            state->h     = std::move(h);
            state->lease = std::move(lease);
            if (server_diag_enabled()) {
                state->rmet = rmet;
            }

            res.status = 200;
            res.set_chunked_content_provider(
                "text/event-stream",
                [state, vocab, &eng](size_t /*offset*/,
                                     httplib::DataSink & sink) -> bool {
                    token_stream_event ev;
                    if (server_diag_enabled()) {
                        state->rmet.stream_channel_get_count++;
                    }
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
                        // Phase 1: leave metrics_emitted=false so the
                        // M7d resource releaser owns the JSONL row.
                        // final_status will be set there if r is
                        // unavailable (releaser tags "stream_error"
                        // when result.get() also fails).
                        return false;
                    }

                    if (ev.kind == stream_event_kind::token) {
                        if (state->sink_alive) {
                            // Cumulative-detokenization delta scheme.
                            // The cumulative `common_detokenize` owns
                            // every tokenizer-specific rule (byte-
                            // fallback assembly, leading-space glue,
                            // special-token suppression under
                            // special=false, trailing-whitespace
                            // policy). The handler only emits the new
                            // visible byte-substring.
                            //
                            // Prefix-stability invariant: each new
                            // cumulative detokenization must extend
                            // the previously emitted cumulative text
                            // byte-for-byte. A future detokenizer
                            // edge case that violates the invariant
                            // fails closed below rather than corrupt
                            // the wire stream.
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
                                // Surface through the existing env-
                                // gated trace (free-form printf-style
                                // `fmt`; no trace.{h,cpp} edits). Run
                                // the standard M7d cancel/drain
                                // finalize routine and return false
                                // without emitting a `done` record —
                                // the wire bytes are already
                                // unrecoverable, so finalizing as
                                // "completed" would be a lie.
                                trace::event(
                                    "stream_detokenize_prefix_violation"
                                    " request=%d token_id=%d"
                                    " emitted_bytes=%zu"
                                    " new_full_bytes=%zu",
                                    state->h.token.request_id,
                                    ev.token_id,
                                    state->emitted_text.size(),
                                    full.size());
                                state->sink_alive = false;
                                if (!state->cancel_issued) {
                                    state->h.cancel(eng);
                                    state->cancel_issued = true;
                                    if (server_diag_enabled()) {
                                        state->rmet.cancel_issued = true;
                                    }
                                }
                                while (true) {
                                    token_stream_event drain_ev;
                                    if (server_diag_enabled()) {
                                        state->rmet
                                            .stream_channel_get_count++;
                                    }
                                    try {
                                        drain_ev =
                                            state->h.stream->get(
                                                hpx::launch::sync);
                                    } catch (
                                        const std::exception &) {
                                        break;
                                    }
                                    if (drain_ev.kind ==
                                          stream_event_kind::closed) {
                                        break;
                                    }
                                }
                                if (!state->result_consumed) {
                                    try {
                                        request_result fr =
                                            state->h.result.get();
                                        if (server_diag_enabled()) {
                                            state->rmet.n_decoded =
                                                fr.n_decoded;
                                            state->rmet.final_status =
                                                status_name(fr.status);
                                            if (fr.request_id >= 0) {
                                                state->rmet.request_id =
                                                    fr.request_id;
                                            }
                                        }
                                    } catch (
                                        const std::exception &) {}
                                    state->result_consumed = true;
                                }
                                // Phase 1: in-provider prefix-violation
                                // exit. Leave metrics_emitted=false so
                                // the M7d resource releaser emits the
                                // JSONL row exactly once — keeps the
                                // dedupe contract centralized in one
                                // place rather than scattered across
                                // every sad-path return.
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
                            if (sink.write(payload.data(),
                                           payload.size())) {
                                if (server_diag_enabled()) {
                                    state->rmet.sse_write_count++;
                                    state->rmet.sse_write_bytes +=
                                        static_cast<int64_t>(
                                            payload.size());
                                    if (state->rmet.first_token_us == 0) {
                                        state->rmet.first_token_us =
                                            server_now_us();
                                    }
                                }
                                return true;
                            }
                            // ---- M7d (in-provider disconnect) ------
                            // sink.write reported the peer is gone.
                            // Finalize inline in case cpp-httplib's
                            // chunked write loop chooses not to call
                            // us again.
                            state->sink_alive = false;
                            if (server_diag_enabled()) {
                                state->rmet.disconnect_observed = true;
                            }
                            if (!state->cancel_issued) {
                                state->h.cancel(eng);
                                state->cancel_issued = true;
                                if (server_diag_enabled()) {
                                    state->rmet.cancel_issued = true;
                                }
                            }
                        }
                        while (true) {
                            token_stream_event drain_ev;
                            if (server_diag_enabled()) {
                                state->rmet.stream_channel_get_count++;
                            }
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
                                request_result fr =
                                    state->h.result.get();
                                if (server_diag_enabled()) {
                                    state->rmet.n_decoded = fr.n_decoded;
                                    state->rmet.final_status =
                                        status_name(fr.status);
                                    if (fr.request_id >= 0) {
                                        state->rmet.request_id =
                                            fr.request_id;
                                    }
                                }
                            } catch (const std::exception &) {}
                            state->result_consumed = true;
                        }
                        // Phase 1: in-provider sink.write-returned-false
                        // exit. Leave metrics_emitted=false so the M7d
                        // resource releaser emits the JSONL row exactly
                        // once.
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
                        if (server_diag_enabled()) {
                            // The `done` SSE record is unchecked
                            // (sink.write return value ignored per M7d
                            // contract), so the wire bytes may not
                            // have actually landed. Count it the same
                            // way: one chunked-write attempt with the
                            // full payload size, matching how analysis
                            // code interprets sse_write_bytes (bytes
                            // handed to cpp-httplib, not bytes ACKed).
                            state->rmet.sse_write_count++;
                            state->rmet.sse_write_bytes +=
                                static_cast<int64_t>(payload.size());
                        }
                    }
                    state->completed_cleanly = true;
                    // Phase 1: happy-path JSONL emission. metrics_emitted
                    // dedupes against the M7d resource releaser, which
                    // also fires on this path but short-circuits via
                    // completed_cleanly before reaching its own
                    // emission site.
                    if (server_diag_enabled()
                     && !state->metrics_emitted) {
                        state->rmet.completion_us = server_now_us();
                        if (result_ok) {
                            state->rmet.n_decoded    = r.n_decoded;
                            state->rmet.final_status =
                                status_name(r.status);
                            if (r.request_id >= 0) {
                                state->rmet.request_id = r.request_id;
                            }
                        } else if (state->rmet.final_status
                                   == request_metrics{}.final_status) {
                            state->rmet.final_status = "engine_error";
                        }
                        dump_request_metrics_jsonl(state->rmet);
                        state->metrics_emitted = true;
                    }
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
                    if (server_diag_enabled()) {
                        // Releaser-without-clean-close means cpp-httplib
                        // either observed !strm.is_peer_alive() between
                        // provider invocations, or the provider's own
                        // in-provider sad-path already set
                        // disconnect_observed=true (idempotent re-set).
                        // Either way, the server saw the request go
                        // away without delivering a `done` record.
                        state->rmet.disconnect_observed = true;
                    }
                    if (!state->cancel_issued) {
                        state->h.cancel(eng);
                        state->cancel_issued = true;
                        if (server_diag_enabled()) {
                            state->rmet.cancel_issued = true;
                        }
                    }
                    while (true) {
                        token_stream_event drain_ev;
                        if (server_diag_enabled()) {
                            state->rmet.stream_channel_get_count++;
                        }
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
                            request_result fr =
                                state->h.result.get();
                            if (server_diag_enabled()) {
                                state->rmet.n_decoded = fr.n_decoded;
                                state->rmet.final_status =
                                    status_name(fr.status);
                                if (fr.request_id >= 0) {
                                    state->rmet.request_id =
                                        fr.request_id;
                                }
                            }
                        } catch (const std::exception &) {}
                        state->result_consumed = true;
                    }
                    // Phase 1: catch-all JSONL emission. Fires for
                    // every streaming exit path except the happy
                    // closed branch (which short-circuited above via
                    // completed_cleanly). metrics_emitted dedupe is
                    // belt-and-suspenders here — the happy path is
                    // the only other site that emits, and it sets
                    // completed_cleanly before doing so, so this
                    // branch can only reach the emit site when
                    // metrics_emitted is still false. Kept for
                    // defense-in-depth against future refactors.
                    if (server_diag_enabled()
                     && !state->metrics_emitted) {
                        state->rmet.completion_us = server_now_us();
                        if (state->rmet.final_status
                            == request_metrics{}.final_status) {
                            // Result.get() either threw or was never
                            // available (e.g. provider's stream.get
                            // threw at line ~849 and the result
                            // promise also failed). Tag explicitly.
                            state->rmet.final_status = "stream_error";
                        }
                        dump_request_metrics_jsonl(state->rmet);
                        state->metrics_emitted = true;
                    }
                });
        });

        // Phase 1 graceful-shutdown precursor. Endpoint registration
        // is GATED by server_diag_shutdown_enabled() — true iff BOTH
        // LLAMA_HPX_DIAG_METRICS=1 AND LLAMA_HPX_DIAG_ENABLE_SHUTDOWN=1.
        // When the gate is closed the route is not registered at all,
        // so cpp-httplib returns its default 404 to any POST /shutdown.
        // When open: handler emits a small JSON ack, then calls
        // srv.stop() from the cpp-httplib handler thread — that is
        // the supported safe entry point per cpp-httplib's contract.
        // srv.stop() sets the accept-loop's atomic stop flag; the
        // in-flight response still flushes, srv.listen() returns
        // cleanly, the post-listen cleanup (eng.request_shutdown +
        // engine_fut.get) runs, engine::run() reaches its tail, and
        // dump_metrics_jsonl(result_) emits the engine iter rows
        // and engine_summary row that were previously inaccessible
        // when the operator killed the binary with SIGINT.
        //
        // No signal handler is installed. No always-on production
        // shutdown endpoint is exposed.
        if (server_diag_shutdown_enabled()) {
            fprintf(stderr,
                "[hpx-server] diagnostic /shutdown endpoint enabled "
                "(LLAMA_HPX_DIAG_METRICS=1 AND "
                "LLAMA_HPX_DIAG_ENABLE_SHUTDOWN=1)\n");
            fflush(stderr);
            srv.Post("/shutdown",
                     [&srv](const httplib::Request & /*req*/,
                            httplib::Response &       res) {
                nlohmann::json out;
                out["status"] = "shutting_down";
                res.status = 200;
                res.set_content(out.dump(), "application/json");
                // cpp-httplib's srv.stop() is thread-safe and is the
                // documented way to drive a graceful exit from a
                // request handler. The accept loop observes the stop
                // flag and exits AFTER this handler's response has
                // been flushed by the worker thread.
                srv.stop();
            });
        }

        // listen() blocks the calling thread (main) until somebody
        // calls srv.stop(). M7a has no signal handler; an operator
        // running the binary by hand terminates it externally. The
        // server smokes drive srv.stop() through normal control
        // flow from a separate task — see hpx_server_smoke.cpp.
        // Phase 1 diagnostic precursor (above): when the env-gated
        // /shutdown endpoint is registered, a POST request drives
        // srv.stop() through normal control flow without any signal
        // involvement.
        fprintf(stderr,
            "[hpx-server] listening on %s:%d  "
            "(n_seq_max=%d max_prompt_tokens=%d batch_capacity=%d "
            "max_concurrent=%d n_ctx=%d prefill_budget_rows=%d)\n",
            args.host.c_str(), args.port,
            args.n_seq_max, args.max_prompt_tokens, batch_capacity,
            args.max_concurrent, args.n_ctx, args.prefill_budget_rows);
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
