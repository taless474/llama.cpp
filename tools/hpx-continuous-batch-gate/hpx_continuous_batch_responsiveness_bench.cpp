// hpx_continuous_batch_responsiveness_bench.cpp — N3.1 / Experiment 13:
// control-plane responsiveness benchmark for the llama-hpx-engine.
//
// Phase 1 scope: W2 queued-cancel and W3 multi-request streaming
// workloads. The Python driver under
// hpx-bench/experiments/13_control_plane_responsiveness/ invokes this
// binary once per (mode, workload) cell, parses the emitted JSONL,
// and aggregates percentiles. Each invocation emits one JSONL line
// per request plus a terminal "RESPONSIVENESS_BENCH: PASS|FAIL" line
// that lets a Python or shell smoke check the binary in isolation.
//
// HPX-native boundary: this TU calls llama.cpp APIs only for backend /
// model / context setup, tokenization, and final cleanup. All llama
// execution and KV mutation live inside eng.run() running on the
// hpx_runtime::async_on_engine task this file spawns. No
// llama_decode, llama_batch_*, llama_memory_seq_*,
// llama_get_logits_ith, or common_batch_add call sites exist here.
//
// Time domain: submitter-side timestamps and engine-side timestamps
// (request_result.t_admitted_us / t_first_publish_us /
//  t_complete_us / t_cancel_observed_us) all use
//   std::chrono::steady_clock::now().time_since_epoch() us
// so the two sides can be subtracted directly with no conversion.
//
// Stream drain model:
//   W2 drains its single stream on the calling (foreign) thread
//     (drain_mode="single").
//   W3 (Phase 2) drains the K streams CONCURRENTLY, one foreign
//     std::thread per stream consumer (drain_mode="concurrent"). The
//     per-token timestamp token_receive_us is stamped when the consumer
//     DEQUEUES the event from its HPX channel — i.e. it is the
//     adapter/consumer-observed receive time, not engine-side
//     publish_token() time. inter_token_gap_us derived from it is the
//     adapter-observed streaming receive cadence: engine production plus
//     HPX channel wakeup, OS scheduling, and foreign-thread consumer
//     wake latency. That is exactly what a client-side responsiveness
//     benchmark should measure. The consumer threads are Layer-1 adapter
//     code (like hpx-server / cpp-httplib): they touch only their own
//     HPX channel-receive half and result future, never llama.cpp state,
//     and never an engine internal. std::thread is confined to this TU;
//     it does not appear in engine.cpp/engine.h/types.h.

#include "common.h"
#include "llama.h"

#include "engine.h"
#include "hpx_runtime.h"
#include "trace.h"
#include "types.h"

#include <hpx/hpx.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

// ---- absolute steady_clock microseconds (matches engine's now_us) -------
inline int64_t now_us() noexcept {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

enum class bench_mode {
    default_os1,
    default_os2,
    engine_pool_os2,
};

enum class bench_workload {
    w2_queued_cancel,
    w3_multi_stream,
};

struct bench_args {
    std::string    model_path;
    bench_mode     mode             = bench_mode::default_os2;
    bench_workload workload         = bench_workload::w2_queued_cancel;
    int32_t        trials           = 1;
    int32_t        warmup_trials    = 0;
    int32_t        decode_budget    = 8;
    int32_t        n_streams        = 4;
    std::string    prompt           = "Hello, my name is";
    std::string    output_path;             // empty => stdout
    std::string    trial_id_prefix  = "trial";
    std::string    mode_label;              // populated from --mode
    std::string    workload_label;          // populated from --workload
};

const char * mode_to_label(bench_mode m) {
    switch (m) {
        case bench_mode::default_os1:     return "default_os1";
        case bench_mode::default_os2:     return "default_os2";
        case bench_mode::engine_pool_os2: return "engine_pool_os2";
    }
    return "unknown";
}

const char * workload_to_label(bench_workload w) {
    switch (w) {
        case bench_workload::w2_queued_cancel: return "w2";
        case bench_workload::w3_multi_stream:  return "w3";
    }
    return "unknown";
}

const char * status_label(request_status s) {
    switch (s) {
        case request_status::completed:       return "completed";
        case request_status::cancelled:       return "cancelled";
        case request_status::failed_reserved: return "failed_reserved";
    }
    return "unknown";
}

const char * stream_close_label(stream_close_reason r) {
    switch (r) {
        case stream_close_reason::completed: return "completed";
        case stream_close_reason::cancelled: return "cancelled";
        case stream_close_reason::error:     return "error";
    }
    return "unknown";
}

void print_usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s --model <path> --mode <mode> --workload <w>\n"
        "       [--trials N] [--warmup-trials N]\n"
        "       [--decode-budget B] [--n-streams K] [--prompt S]\n"
        "       [--output PATH] [--trial-id PREFIX]\n"
        "\n"
        "  Experiment 13 control-plane responsiveness benchmark.\n"
        "  Emits one JSONL line per request on stdout (or --output)\n"
        "  and a terminal RESPONSIVENESS_BENCH: PASS|FAIL line.\n"
        "\n"
        "  --mode one of:\n"
        "    default_os1      default-pool, hpx-os-threads=1\n"
        "    default_os2      default-pool, hpx-os-threads=2\n"
        "    engine_pool_os2  engine-pool, hpx-os-threads=2 + named pool\n"
        "\n"
        "  --workload one of:\n"
        "    w2  queued-cancel responsiveness (single submit+cancel)\n"
        "    w3  multi-request streaming (K concurrent streams)\n",
        argv0);
}

bool parse_mode(const std::string & s, bench_mode & out) {
    if (s == "default_os1")     { out = bench_mode::default_os1;     return true; }
    if (s == "default_os2")     { out = bench_mode::default_os2;     return true; }
    if (s == "engine_pool_os2") { out = bench_mode::engine_pool_os2; return true; }
    return false;
}

bool parse_workload(const std::string & s, bench_workload & out) {
    if (s == "w2") { out = bench_workload::w2_queued_cancel; return true; }
    if (s == "w3") { out = bench_workload::w3_multi_stream;  return true; }
    return false;
}

bool parse_args(int argc, char ** argv, bench_args & a) {
    for (int i = 1; i < argc; i++) {
        const std::string s = argv[i];
        auto need = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: %s requires a value\n", name);
                return nullptr;
            }
            return argv[++i];
        };
        if (s == "-h" || s == "--help") {
            print_usage(argv[0]);
            return false;
        } else if (s == "--model") {
            const char * v = need("--model"); if (!v) return false;
            a.model_path = v;
        } else if (s == "--mode") {
            const char * v = need("--mode"); if (!v) return false;
            if (!parse_mode(v, a.mode)) {
                fprintf(stderr, "error: unknown --mode '%s'\n", v);
                return false;
            }
        } else if (s == "--workload") {
            const char * v = need("--workload"); if (!v) return false;
            if (!parse_workload(v, a.workload)) {
                fprintf(stderr, "error: unknown --workload '%s'\n", v);
                return false;
            }
        } else if (s == "--trials") {
            const char * v = need("--trials"); if (!v) return false;
            a.trials = std::atoi(v);
        } else if (s == "--warmup-trials") {
            const char * v = need("--warmup-trials"); if (!v) return false;
            a.warmup_trials = std::atoi(v);
        } else if (s == "--decode-budget") {
            const char * v = need("--decode-budget"); if (!v) return false;
            a.decode_budget = std::atoi(v);
        } else if (s == "--n-streams") {
            const char * v = need("--n-streams"); if (!v) return false;
            a.n_streams = std::atoi(v);
        } else if (s == "--prompt") {
            const char * v = need("--prompt"); if (!v) return false;
            a.prompt = v;
        } else if (s == "--output") {
            const char * v = need("--output"); if (!v) return false;
            a.output_path = v;
        } else if (s == "--trial-id") {
            const char * v = need("--trial-id"); if (!v) return false;
            a.trial_id_prefix = v;
        } else {
            fprintf(stderr, "error: unknown arg '%s'\n", s.c_str());
            return false;
        }
    }
    if (a.model_path.empty()) {
        fprintf(stderr, "error: --model is required\n");
        return false;
    }
    if (a.trials < 1) {
        fprintf(stderr, "error: --trials must be >= 1\n");
        return false;
    }
    if (a.warmup_trials < 0) {
        fprintf(stderr, "error: --warmup-trials must be >= 0\n");
        return false;
    }
    if (a.workload == bench_workload::w3_multi_stream && a.n_streams < 1) {
        fprintf(stderr, "error: --n-streams must be >= 1 for w3\n");
        return false;
    }
    a.mode_label     = mode_to_label(a.mode);
    a.workload_label = workload_to_label(a.workload);
    return true;
}

// ---- JSON helpers (small, dependency-free) ----------------------------------

void json_int64(std::ostringstream & o, const char * k, int64_t v,
                bool nullable_neg1 = true) {
    o << "\"" << k << "\":";
    if (nullable_neg1 && v < 0) o << "null";
    else                        o << v;
}

void json_int(std::ostringstream & o, const char * k, int32_t v) {
    o << "\"" << k << "\":" << v;
}

void json_str(std::ostringstream & o, const char * k, const std::string & v) {
    o << "\"" << k << "\":\"" << v << "\"";
}

void json_bool(std::ostringstream & o, const char * k, bool v) {
    o << "\"" << k << "\":" << (v ? "true" : "false");
}

void json_int64_vec(std::ostringstream & o, const char * k,
                    const std::vector<int64_t> & v) {
    o << "\"" << k << "\":[";
    for (size_t i = 0; i < v.size(); i++) {
        if (i) o << ",";
        o << v[i];
    }
    o << "]";
}

// ---- per-request record produced by the bench -------------------------------

struct trial_record {
    std::string                trial_id;
    std::string                mode_label;
    std::string                workload_label;
    int32_t                    request_id     = -1;
    int32_t                    decode_budget  = 0;
    bool                       want_stream    = false;
    int32_t                    is_warmup      = 0;
    std::string                status         = "unknown";
    int32_t                    n_decoded      = 0;
    // How this request's stream was drained: "single" (W2, drained on
    // the calling thread) or "concurrent" (W3 Phase 2, drained on a
    // dedicated foreign consumer thread). Recorded in JSONL so Phase 1
    // (serial) and Phase 2 (concurrent) data are separable.
    std::string                drain_mode     = "single";

    // submitter-side stamps (absolute us; -1 means not recorded).
    // token_receive_us holds the CONSUMER-observed receive time of each
    // token event (when the drain loop dequeues it from the channel),
    // NOT the engine-side publish_token() time. inter_token_gap_us is
    // derived from it.
    int64_t                    submit_us         = -1;
    int64_t                    cancel_us         = -1;
    int64_t                    future_ready_us   = -1;
    int64_t                    stream_close_us   = -1;
    std::vector<int64_t>       token_receive_us;

    // engine-side stamps copied from request_result
    int64_t                    t_admitted_us        = -1;
    int64_t                    t_first_publish_us   = -1;
    int64_t                    t_complete_us        = -1;
    int64_t                    t_cancel_observed_us = -1;

    // observed terminal stream close reason (only for streaming requests)
    std::string                stream_close_reason  = "unknown";

    // derived deltas (negative => "not applicable"); written by finalize()
    int64_t                    submit_to_admitted_us     = -1;
    int64_t                    submit_to_first_token_us  = -1;
    int64_t                    submit_to_complete_us     = -1;
    int64_t                    cancel_to_observed_us     = -1;
    int64_t                    cancel_to_future_ready_us = -1;
    int64_t                    cancel_to_stream_close_us = -1;
    std::vector<int64_t>       inter_token_gap_us;
};

void finalize_deltas(trial_record & r) {
    if (r.submit_us >= 0 && r.t_admitted_us >= 0)
        r.submit_to_admitted_us = r.t_admitted_us - r.submit_us;
    if (r.submit_us >= 0 && r.t_first_publish_us >= 0)
        r.submit_to_first_token_us = r.t_first_publish_us - r.submit_us;
    if (r.submit_us >= 0 && r.t_complete_us >= 0)
        r.submit_to_complete_us = r.t_complete_us - r.submit_us;
    if (r.cancel_us >= 0 && r.t_cancel_observed_us >= 0)
        r.cancel_to_observed_us = r.t_cancel_observed_us - r.cancel_us;
    if (r.cancel_us >= 0 && r.future_ready_us >= 0)
        r.cancel_to_future_ready_us = r.future_ready_us - r.cancel_us;
    if (r.cancel_us >= 0 && r.stream_close_us >= 0)
        r.cancel_to_stream_close_us = r.stream_close_us - r.cancel_us;
    r.inter_token_gap_us.clear();
    for (size_t i = 1; i < r.token_receive_us.size(); i++) {
        r.inter_token_gap_us.push_back(
            r.token_receive_us[i] - r.token_receive_us[i-1]);
    }
}

std::string to_jsonl(const trial_record & r) {
    std::ostringstream o;
    o << "{";
    json_str  (o, "trial_id",       r.trial_id);       o << ",";
    json_str  (o, "mode",           r.mode_label);     o << ",";
    json_str  (o, "workload",       r.workload_label); o << ",";
    json_int  (o, "request_id",     r.request_id);     o << ",";
    json_int  (o, "decode_budget",  r.decode_budget);  o << ",";
    json_bool (o, "want_stream",    r.want_stream);    o << ",";
    json_int  (o, "is_warmup",      r.is_warmup);      o << ",";
    json_str  (o, "status",         r.status);         o << ",";
    json_str  (o, "drain_mode",     r.drain_mode);     o << ",";
    json_int  (o, "n_decoded",      r.n_decoded);      o << ",";
    json_int64(o, "submit_us",         r.submit_us);          o << ",";
    json_int64(o, "cancel_us",         r.cancel_us);          o << ",";
    json_int64(o, "future_ready_us",   r.future_ready_us);    o << ",";
    json_int64(o, "stream_close_us",   r.stream_close_us);    o << ",";
    json_int64(o, "t_admitted_us",          r.t_admitted_us);          o << ",";
    json_int64(o, "t_first_publish_us",     r.t_first_publish_us);     o << ",";
    json_int64(o, "t_complete_us",          r.t_complete_us);          o << ",";
    json_int64(o, "t_cancel_observed_us",   r.t_cancel_observed_us);   o << ",";
    json_str  (o, "stream_close_reason",    r.stream_close_reason);    o << ",";
    json_int64(o, "submit_to_admitted_us",     r.submit_to_admitted_us);     o << ",";
    json_int64(o, "submit_to_first_token_us",  r.submit_to_first_token_us);  o << ",";
    json_int64(o, "submit_to_complete_us",     r.submit_to_complete_us);     o << ",";
    json_int64(o, "cancel_to_observed_us",     r.cancel_to_observed_us);     o << ",";
    json_int64(o, "cancel_to_future_ready_us", r.cancel_to_future_ready_us); o << ",";
    json_int64(o, "cancel_to_stream_close_us", r.cancel_to_stream_close_us); o << ",";
    json_int64_vec(o, "token_receive_us",   r.token_receive_us);    o << ",";
    json_int64_vec(o, "inter_token_gap_us", r.inter_token_gap_us);
    o << "}";
    return o.str();
}

// ---- monotonicity invariants (per-record) -----------------------------------

struct invariant_failure {
    std::string trial_id;
    int32_t     request_id;
    std::string reason;
};

bool check_invariants(const trial_record & r,
                      std::vector<invariant_failure> & failures) {
    auto fail = [&](const char * why) {
        failures.push_back({r.trial_id, r.request_id, why});
    };
    bool ok = true;

    if (r.submit_us < 0) {
        fail("submit_us not recorded");
        ok = false;
    }
    if (r.t_admitted_us >= 0 && r.submit_us > r.t_admitted_us) {
        fail("submit_us > t_admitted_us");
        ok = false;
    }
    if (r.t_first_publish_us >= 0 && r.t_admitted_us > r.t_first_publish_us) {
        fail("t_admitted_us > t_first_publish_us");
        ok = false;
    }
    if (r.t_complete_us >= 0 && r.t_first_publish_us > r.t_complete_us) {
        fail("t_first_publish_us > t_complete_us");
        ok = false;
    }
    if (r.t_complete_us >= 0 && r.t_admitted_us > r.t_complete_us) {
        fail("t_admitted_us > t_complete_us");
        ok = false;
    }
    if (r.cancel_us >= 0) {
        if (r.t_cancel_observed_us >= 0
         && r.cancel_us > r.t_cancel_observed_us) {
            fail("cancel_us > t_cancel_observed_us");
            ok = false;
        }
        if (r.future_ready_us >= 0
         && r.t_cancel_observed_us > r.future_ready_us) {
            fail("t_cancel_observed_us > future_ready_us");
            ok = false;
        }
        if (r.stream_close_us >= 0
         && r.t_cancel_observed_us > r.stream_close_us) {
            fail("t_cancel_observed_us > stream_close_us");
            ok = false;
        }
    }
    for (size_t i = 1; i < r.token_receive_us.size(); i++) {
        if (r.token_receive_us[i] < r.token_receive_us[i-1]) {
            fail("token_receive_us not monotonically non-decreasing");
            ok = false;
            break;
        }
    }
    return ok;
}

// ---- BENCH-ONLY temporary debug tracing (Phase 1 hang investigation) -------
// Enabled by env LLAMA_HPX_BENCH_TRACE=1. All output goes to stderr so
// stdout (PASS/FAIL line + JSONL) stays clean. Not an engine trace —
// these prints live only in this TU and never touch engine state.
#define BENCH_TRACE_ENABLED() (std::getenv("LLAMA_HPX_BENCH_TRACE") != nullptr)
#define BENCH_TRACE(fmt, ...) do { \
    if (BENCH_TRACE_ENABLED()) { \
        fprintf(stderr, "[bench-trace t=%lld] " fmt "\n", \
                (long long) now_us(), ##__VA_ARGS__); \
        fflush(stderr); \
    } \
} while (0)

// ---- W2: queued-cancel responsiveness ---------------------------------------

// Builds an engine with keep_alive=true, initial_idle_slots=0, empty
// budgets, and the responsiveness timing option ON. Each measured
// trial submits one request then cancels it before any admission can
// happen, then awaits the cancelled result and drains the stream.
//
// Cancel ordering: cancel_request(rid) is issued immediately after
// submit_request returns the handle. The engine's iteration-boundary
// cancel path drains the staged cancel and resolves the request as
// queued-cancelled. The test does NOT race admission — there are no
// admission sources (initial_idle_slots=0, budgets={}, no
// completion-freed pool entries), so the request stays queued
// regardless of timing.
int run_w2(const bench_args &           args,
           llama_context *              ctx,
           const llama_vocab *          vocab,
           int32_t                      n_vocab,
           const std::vector<llama_token> & shared_prompt,
           std::vector<trial_record> &  out) {
    const int32_t batch_capacity =
        std::max<int32_t>(static_cast<int32_t>(shared_prompt.size()), 1);

    std::vector<waiting_request> empty_waiting;

    engine_options opts;
    opts.lib.ctx                          = ctx;
    opts.lib.vocab                        = vocab;
    opts.lib.n_vocab                      = n_vocab;
    opts.lib.batch_capacity               = batch_capacity;
    opts.lib.n_seq_max                    = 1;
    opts.lib.initial_idle_slots           = 0;
    opts.lib.keep_alive                   = true;
    opts.lib.cooperative_yield_on_pump    =
        (args.mode != bench_mode::engine_pool_os2);
    opts.lib.enable_responsiveness_timing = true;
    opts.preload.prompt_tokens            = &shared_prompt;
    opts.preload.budgets                  = {};
    opts.preload.waiting_queue            = &empty_waiting;
    opts.preload.reuse_completed          = false;
    opts.preload.stream_all               = false;
    opts.gate_test.cancel_after           = -1;
    opts.gate_test.max_decode_iters       = 0;

    engine eng(std::move(opts));
    BENCH_TRACE("w2: engine constructed");

    hpx::future<void> engine_fut =
        hpx_runtime::async_on_engine([&] { eng.run(); });
    BENCH_TRACE("w2: engine task spawned");

    const int32_t total_trials = args.warmup_trials + args.trials;
    for (int32_t i = 0; i < total_trials; i++) {
        trial_record r;
        r.trial_id       = args.trial_id_prefix + "_" + std::to_string(i);
        r.mode_label     = args.mode_label;
        r.workload_label = args.workload_label;
        r.request_id     = 1000 + i;
        r.decode_budget  = args.decode_budget;
        r.want_stream    = true;
        r.is_warmup      = (i < args.warmup_trials) ? 1 : 0;
        BENCH_TRACE("w2: trial %d rid=%d is_warmup=%d begin",
                    i, r.request_id, r.is_warmup);

        submit_request req;
        req.request_id    = r.request_id;
        req.prompt_tokens = shared_prompt;
        req.decode_budget = args.decode_budget;
        req.want_stream   = true;

        BENCH_TRACE("w2: before submit_request rid=%d", r.request_id);
        r.submit_us = now_us();
        submit_handle h = eng.submit_request(std::move(req));
        BENCH_TRACE("w2: after submit_request rid=%d", r.request_id);

        BENCH_TRACE("w2: before cancel_request rid=%d", r.request_id);
        r.cancel_us = now_us();
        eng.cancel_request(r.request_id);
        BENCH_TRACE("w2: after cancel_request rid=%d", r.request_id);

        request_result rr;
        try {
            BENCH_TRACE("w2: before h.result.get rid=%d", r.request_id);
            rr = h.result.get();
            BENCH_TRACE("w2: after h.result.get rid=%d status=%d",
                        r.request_id, (int) rr.status);
        } catch (const std::exception & e) {
            BENCH_TRACE("w2: h.result.get threw rid=%d: %s",
                        r.request_id, e.what());
            fprintf(stderr,
                "[w2] trial %s: h.result.get threw: %s\n",
                r.trial_id.c_str(), e.what());
            r.status = "exception";
            finalize_deltas(r);
            out.push_back(std::move(r));
            continue;
        }
        r.future_ready_us = now_us();

        // Drain stream to terminal closed event.
        if (h.stream.has_value()) {
            BENCH_TRACE("w2: drain stream rid=%d begin", r.request_id);
            try {
                while (true) {
                    BENCH_TRACE("w2: drain rid=%d before stream->get()", r.request_id);
                    hpx::future<token_stream_event> fev = h.stream->get();
                    BENCH_TRACE("w2: drain rid=%d after stream->get(), before fev.get()", r.request_id);
                    token_stream_event ev = fev.get();
                    BENCH_TRACE("w2: drain rid=%d after fev.get() kind=%d",
                                r.request_id, (int) ev.kind);
                    if (ev.kind == stream_event_kind::token) {
                        r.token_receive_us.push_back(now_us());
                    } else {
                        r.stream_close_us     = now_us();
                        r.stream_close_reason = stream_close_label(ev.close_reason);
                        BENCH_TRACE("w2: drain rid=%d stream closed reason=%s",
                                    r.request_id, r.stream_close_reason.c_str());
                        break;
                    }
                }
            } catch (...) {
                BENCH_TRACE("w2: drain rid=%d caught exception", r.request_id);
                // Channel closed by engine; bench treats this as terminal
                // already accounted for; leave stream_close_us at -1 if
                // never observed (invariant check will flag).
            }
        }

        r.status      = status_label(rr.status);
        r.n_decoded   = rr.n_decoded;
        r.t_admitted_us        = rr.t_admitted_us;
        r.t_first_publish_us   = rr.t_first_publish_us;
        r.t_complete_us        = rr.t_complete_us;
        r.t_cancel_observed_us = rr.t_cancel_observed_us;

        finalize_deltas(r);
        out.push_back(std::move(r));
        BENCH_TRACE("w2: trial %d rid=%d end", i, r.request_id);
    }

    BENCH_TRACE("w2: before request_shutdown");
    eng.request_shutdown();
    BENCH_TRACE("w2: after request_shutdown, before engine_fut.get");
    try { engine_fut.get(); } catch (...) { /* surface below */ }
    BENCH_TRACE("w2: after engine_fut.get");
    return 0;
}

// ---- W3: multi-request streaming --------------------------------------------
//
// Engine: keep_alive=true, initial_idle_slots=K, stream_all=false (per
// request opt-in), budgets={}. Each measured trial submits K requests
// concurrently (rapid sequential submit_request calls before any
// h.result.get), then drains each stream in turn, then awaits each
// result. Records per-token publish times so first-token and
// inter-token gaps can be derived.
int run_w3(const bench_args &           args,
           llama_context *              ctx,
           const llama_vocab *          vocab,
           int32_t                      n_vocab,
           const std::vector<llama_token> & shared_prompt,
           std::vector<trial_record> &  out) {
    // W3 admits n_streams requests concurrently, so a single iter
    // can carry n_streams * prompt_size prefill rows. Size the
    // engine-visible batch capacity to fit that peak.
    const int32_t batch_capacity =
        std::max<int32_t>(
            static_cast<int32_t>(shared_prompt.size()) * args.n_streams, 1);

    std::vector<waiting_request> empty_waiting;

    engine_options opts;
    opts.lib.ctx                          = ctx;
    opts.lib.vocab                        = vocab;
    opts.lib.n_vocab                      = n_vocab;
    opts.lib.batch_capacity               = batch_capacity;
    opts.lib.n_seq_max                    = args.n_streams;
    opts.lib.initial_idle_slots           = args.n_streams;
    opts.lib.keep_alive                   = true;
    opts.lib.cooperative_yield_on_pump    =
        (args.mode != bench_mode::engine_pool_os2);
    opts.lib.enable_responsiveness_timing = true;
    opts.preload.prompt_tokens            = &shared_prompt;
    opts.preload.budgets                  = {};
    opts.preload.waiting_queue            = &empty_waiting;
    opts.preload.reuse_completed          = false;
    opts.preload.stream_all               = false;
    opts.gate_test.cancel_after           = -1;
    opts.gate_test.max_decode_iters       = 0;

    engine eng(std::move(opts));
    BENCH_TRACE("w3: engine constructed");

    hpx::future<void> engine_fut =
        hpx_runtime::async_on_engine([&] { eng.run(); });
    BENCH_TRACE("w3: engine task spawned");

    const int32_t total_trials = args.warmup_trials + args.trials;
    int32_t next_rid = 2000;
    for (int32_t t = 0; t < total_trials; t++) {
        const bool is_warmup = (t < args.warmup_trials);
        BENCH_TRACE("w3: trial %d is_warmup=%d begin", t, is_warmup ? 1 : 0);

        std::vector<submit_handle> handles;
        handles.reserve(args.n_streams);
        std::vector<trial_record> rs(args.n_streams);

        // Submit K requests rapidly before draining any of them.
        for (int32_t k = 0; k < args.n_streams; k++) {
            trial_record & r = rs[k];
            r.trial_id       = args.trial_id_prefix + "_" + std::to_string(t)
                             + "_s" + std::to_string(k);
            r.mode_label     = args.mode_label;
            r.workload_label = args.workload_label;
            r.request_id     = next_rid++;
            r.decode_budget  = args.decode_budget;
            r.want_stream    = true;
            r.is_warmup      = is_warmup ? 1 : 0;

            submit_request req;
            req.request_id    = r.request_id;
            req.prompt_tokens = shared_prompt;
            req.decode_budget = args.decode_budget;
            req.want_stream   = true;

            BENCH_TRACE("w3: before submit_request rid=%d k=%d", r.request_id, k);
            r.submit_us = now_us();
            handles.push_back(eng.submit_request(std::move(req)));
            BENCH_TRACE("w3: after submit_request rid=%d k=%d", r.request_id, k);
        }

        // Phase 2: drain the K streams CONCURRENTLY. One foreign
        // std::thread per stream consumer, started immediately after
        // submit so the engine sees K live consumers draining from the
        // first decode iteration (this models concurrent SSE clients).
        // Each consumer drains exactly one stream, stamps
        // token_receive_us as it DEQUEUES each event, observes its own
        // terminal close, then awaits its own result.get() and stamps
        // future_ready_us — preserving the per-request order the serial
        // path used (drain-to-close, then result.get), now parallel.
        //
        // Each thread writes only its own rs[k]/handles[k]; main joins
        // all threads before reading, so there is no shared mutable
        // state and no lock is needed. The consumer threads are Layer-1
        // adapter code: they touch only their channel-receive half and
        // their result future, never llama.cpp state or an engine
        // internal. Foreign std::threads (not HPX tasks) keep the
        // engine's HPX scheduling undisturbed — HPX-task consumers would
        // contend for the same worker pool, especially under os_threads=1.
        // Per-thread try/catch keeps any exception from escaping the
        // thread (an escaped exception would std::terminate the process).
        auto drain_one = [&](int32_t k) {
            trial_record & r  = rs[k];
            submit_handle & h = handles[k];
            r.drain_mode = "concurrent";
            try {
                BENCH_TRACE("w3: drain k=%d rid=%d begin", k, r.request_id);
                if (h.stream.has_value()) {
                    bool got_first = false;
                    try {
                        while (true) {
                            hpx::future<token_stream_event> fev = h.stream->get();
                            token_stream_event ev = fev.get();
                            if (ev.kind == stream_event_kind::token) {
                                r.token_receive_us.push_back(now_us());
                                if (!got_first) {
                                    BENCH_TRACE("w3: drain k=%d first token", k);
                                    got_first = true;
                                }
                            } else {
                                r.stream_close_us     = now_us();
                                r.stream_close_reason =
                                    stream_close_label(ev.close_reason);
                                BENCH_TRACE("w3: drain k=%d stream closed reason=%s",
                                            k, r.stream_close_reason.c_str());
                                break;
                            }
                        }
                    } catch (...) {
                        BENCH_TRACE("w3: drain k=%d caught stream exception", k);
                        // Channel closed by engine; leave stream_close_us
                        // at -1 if never observed.
                    }
                }
                try {
                    request_result rr = h.result.get();
                    r.future_ready_us      = now_us();
                    BENCH_TRACE("w3: after h.result.get k=%d status=%d",
                                k, (int) rr.status);
                    r.status               = status_label(rr.status);
                    r.n_decoded            = rr.n_decoded;
                    r.t_admitted_us        = rr.t_admitted_us;
                    r.t_first_publish_us   = rr.t_first_publish_us;
                    r.t_complete_us        = rr.t_complete_us;
                    r.t_cancel_observed_us = rr.t_cancel_observed_us;
                } catch (const std::exception & e) {
                    fprintf(stderr,
                        "[w3] trial %s: h.result.get threw: %s\n",
                        r.trial_id.c_str(), e.what());
                    r.status = "exception";
                }
                finalize_deltas(r);
                BENCH_TRACE("w3: drain k=%d rid=%d end", k, r.request_id);
            } catch (...) {
                // Last-resort guard: never let an exception escape the
                // consumer thread.
                r.status = "exception";
                finalize_deltas(r);
            }
        };

        std::vector<std::thread> consumers;
        consumers.reserve(args.n_streams);
        for (int32_t k = 0; k < args.n_streams; k++) {
            consumers.emplace_back(drain_one, k);
        }
        for (auto & th : consumers) th.join();

        for (auto & r : rs) out.push_back(std::move(r));
        BENCH_TRACE("w3: trial %d end", t);
    }

    BENCH_TRACE("w3: before request_shutdown");
    eng.request_shutdown();
    BENCH_TRACE("w3: after request_shutdown, before engine_fut.get");
    try { engine_fut.get(); } catch (...) { /* surface below */ }
    BENCH_TRACE("w3: after engine_fut.get");
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    bench_args args;
    if (!parse_args(argc, argv, args)) return 2;

    trace::init();

    // Mode -> HPX runtime config.
    hpx_runtime::runtime_config rt_cfg;
    int32_t os_threads = 1;
    switch (args.mode) {
        case bench_mode::default_os1:
            os_threads = 1;
            rt_cfg.enable_engine_pool = false;
            break;
        case bench_mode::default_os2:
            os_threads = 2;
            rt_cfg.enable_engine_pool = false;
            break;
        case bench_mode::engine_pool_os2:
            os_threads = 2;
            rt_cfg.enable_engine_pool = true;
            break;
    }
    if (!hpx_runtime::start_once(os_threads, rt_cfg)) {
        fprintf(stdout,
            "RESPONSIVENESS_BENCH: FAIL: hpx runtime start failed "
            "(mode=%s)\n", args.mode_label.c_str());
        fflush(stdout);
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

    auto emit_fail = [&](const std::string & reason) -> int {
        fprintf(stdout,
            "RESPONSIVENESS_BENCH: FAIL: %s\n", reason.c_str());
        fflush(stdout);
        return cleanup_and_stop(1);
    };

    llama_model_params model_params = llama_model_default_params();
    model = llama_model_load_from_file(args.model_path.c_str(), model_params);
    if (!model) return emit_fail("model load failed");

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx           = 2048;
    ctx_params.n_batch         = 256;
    ctx_params.n_seq_max       =
        (args.workload == bench_workload::w3_multi_stream)
            ? args.n_streams : 1;
    ctx_params.n_threads       = 2;
    ctx_params.n_threads_batch = 2;
    ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) return emit_fail("context create failed");

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> shared_prompt = common_tokenize(
        ctx, args.prompt.c_str(), /*add_special=*/true,
        /*parse_special=*/true);
    if (shared_prompt.empty()) {
        return emit_fail("tokenization produced 0 tokens");
    }

    std::vector<trial_record> records;
    try {
        switch (args.workload) {
            case bench_workload::w2_queued_cancel:
                run_w2(args, ctx, vocab, n_vocab, shared_prompt, records);
                break;
            case bench_workload::w3_multi_stream:
                run_w3(args, ctx, vocab, n_vocab, shared_prompt, records);
                break;
        }
    } catch (const std::exception & e) {
        return emit_fail(std::string("workload exception: ") + e.what());
    } catch (...) {
        return emit_fail("workload unknown exception");
    }

    // Open output sink (stdout by default).
    std::ofstream of;
    std::ostream * out = &std::cout;
    if (!args.output_path.empty()) {
        of.open(args.output_path, std::ios::out | std::ios::trunc);
        if (!of.is_open()) {
            return emit_fail("could not open --output path");
        }
        out = &of;
    }

    // Emit JSONL.
    for (const auto & r : records) {
        (*out) << to_jsonl(r) << "\n";
    }
    out->flush();

    // Invariants pass over MEASURED records only (exclude warmups).
    std::vector<invariant_failure> failures;
    int32_t measured = 0;
    for (const auto & r : records) {
        if (r.is_warmup) continue;
        measured++;
        check_invariants(r, failures);
    }

    // PASS / FAIL summary line on stdout (always; never to --output).
    if (!failures.empty()) {
        fprintf(stdout,
            "RESPONSIVENESS_BENCH: FAIL: %zu invariant violations "
            "across %d measured records (mode=%s workload=%s)\n",
            failures.size(), measured,
            args.mode_label.c_str(), args.workload_label.c_str());
        const size_t k_show = std::min<size_t>(failures.size(), 5);
        for (size_t i = 0; i < k_show; i++) {
            fprintf(stdout,
                "  invariant_fail trial=%s request=%d reason=\"%s\"\n",
                failures[i].trial_id.c_str(),
                failures[i].request_id,
                failures[i].reason.c_str());
        }
        fflush(stdout);
        return cleanup_and_stop(1);
    }

    fprintf(stdout,
        "RESPONSIVENESS_BENCH: PASS: %d measured records (mode=%s "
        "workload=%s)\n", measured,
        args.mode_label.c_str(), args.workload_label.c_str());
    fflush(stdout);
    return cleanup_and_stop(0);
}
