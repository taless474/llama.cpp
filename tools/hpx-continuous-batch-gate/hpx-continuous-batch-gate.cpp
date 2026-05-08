// HPX continuous-batching prototype — Cancel Slice 4: closeout of
// the cooperative-cancellation line. Adds reporting/metrics polish
// only — no behavioral change vs Cancel Slice 3.
//
// Cancel Slice 4 changes:
//   - Per-iter metrics block now reports completed/cancelled splits
//     correctly. The legacy `completed[budget=B]` line counted every
//     seq of that budget, which was misleading once cancellation
//     became real (budget-64 showed 33 even though 3 were cancelled).
//     The line now counts only `status == completed`, and a paired
//     `cancelled[budget=B]` line is emitted for budgets with any
//     cancelled seqs.
//   - `ttc_ms[budget=B]` is split into `ttc_ms_completed[budget=B]`
//     and (when present) `ttc_ms_cancelled[budget=B]`. Both remain
//     descriptive only and are not part of the determinism contract.
//   - Top-level metrics fields added (after engine_task_count):
//       completed_count
//       cancelled_count
//       decode_failures
//       wasted_decode_rows_after_cancel
//       residual_kv_empty
//   - Final emit is HPX_CB_CANCEL_STEP4: PASS / FAIL: <reason>.
// All Cancel Slice 2 / 3 correctness gates still apply unchanged.
//
// HPX continuous-batching prototype — Cancel Slice 3: result/status
// path polish on top of the Cancel Slice 2 cooperative-cancellation
// behavior.
//
// Cancel Slice 3 changes (no new behavior):
//   - Cancel-family trace events are aligned to the spec format
//     so they are grep-friendly and self-describing:
//       event=cancel_requested seq=<id> budget=<int> cancel_after=<int>
//       event=cancel_observed  seq=<id> iter=<int> n_decoded=<int>
//       event=cancel_kv_cleared seq=<id> pos_max_at_clear=<int> cross_talk_ok=1
//       event=cancel_future_fulfilled seq=<id> status=cancelled ttc_us=<int>
//   - Main prints an explicit per-iter status_summary line
//     (completed/cancelled/total) so the result split is obvious in
//     the capture without re-counting per-budget rows.
//   - Main adds an explicit cancelled-result gate
//     `n_decoded == n_decoded_at_cancel` (previously transitive
//     through both equalling --cancel-after).
//   - Engine emits seq_complete / kv_cleared / promise_fulfilled
//     ONLY on the completion path; the cancellation path emits
//     cancel_observed / cancel_kv_cleared / cancel_future_fulfilled
//     instead. Expected counts under the default plan:
//       request_admitted = 99
//       seq_prefilled = 99
//       seq_complete = 93
//       kv_cleared = 93
//       promise_fulfilled = 93
//       cancel_requested = 6
//       cancel_observed = 6
//       cancel_kv_cleared = 6
//       cancel_future_fulfilled = 6
//   - Final emit is HPX_CB_CANCEL_STEP3: PASS / FAIL: <reason>.
// All Cancel Slice 2 correctness gates still apply unchanged.
//
// HPX continuous-batching prototype — Cancel Slice 2: deterministic
// cooperative cancellation enabled inside the engine loop.
//
// Cancel Slice 2 turns on the data model that Cancel Slice 1
// plumbed:
//   - The plan from --cancel-plan is propagated into seq_state at
//     engine construction (cancel_after_decoded_tokens per seq).
//   - The engine task observes cancellation only at iteration
//     boundaries: once before the decode loop (post-prefill) and
//     once at the top of each decode iteration, BEFORE building
//     the active_idx for that iter. No mid-llama_decode
//     interruption.
//   - On observation the engine: marks cancel_observed[_iter]
//     and n_decoded_at_cancel; runs the existing
//     clear_and_check (KV remove + cross-talk verify); fulfills
//     the per-request promise once, with status=cancelled.
//   - Cancelled seqs do not appear in any subsequent decode batch
//     because they become seq.done=true via clear_and_check, and
//     the active_idx builder already skips done seqs. This makes
//     wasted_decode_rows_after_cancel structurally 0.
//   - Cancel-family trace events (cancel_requested,
//     cancel_observed, cancel_kv_cleared,
//     cancel_future_fulfilled) fire under LLAMA_HPX_CB_TRACE=1
//     using the same gating as Slice 4.
//   - --cancel-plan parser extended to accept seq_id 0
//     (previously rejected by parse_csv_int_list); seq_id 0 is
//     not used in the default plan but is now expressible.
// All non-cancelled Slice-3/4/5 correctness gates still hold;
// per-class hash anchors apply to NON-CANCELLED seqs only and
// the cancelled-run budget-64/256 hashes are NOT comparable to
// the all-99-running Slice-5 hashes (batch shape changes after
// iter 16 once the 6 cancelled seqs leave).
//
// HPX continuous-batching prototype — Slice 4: lifecycle traces +
// descriptive metrics on top of the Slice-3 promise/future engine.
//
// Slice 4 adds:
//   - env-gated trace events on stderr (LLAMA_HPX_CB_TRACE=1):
//       engine_start, request_admitted, seq_prefilled, decode_row,
//       seq_complete, kv_cleared, promise_fulfilled, engine_stop
//   - per-iteration descriptive metrics block on stdout:
//       wall_ms, decode_calls, update_iterations,
//       rows_per_batch (p50/p95/max), active_seqs_per_iter (p50/p95/max),
//       completed counts by budget,
//       time_to_completion_ms (mean / p95) by budget,
//       futures_created / promises_fulfilled / futures_completed,
//       engine_task_count
//
// Metrics are descriptive only; no comparison or speedup language.
// Both trace and metrics paths must not change correctness behavior or
// batch shape. With trace disabled (default), the trace path is one
// atomic load + early return per call site.
//
// All earlier Slice-3 invariants still hold:
//   - exactly one HPX engine task per repeat iteration;
//   - one hpx::promise<request_result> per seq, fulfilled only after
//     KV clear + cross-talk check pass;
//   - main validates correctness only from request_result snapshots
//     and never touches llama_context / llama_batch /
//     llama_decode / llama_memory_seq_* / llama_get_logits_ith.
//
// See docs/hpx/hpx_continuous_batching_prototype_design.md.

#include "common.h"
#include "llama.h"

#include <hpx/hpx.hpp>
#include <hpx/hpx_finalize.hpp>
#include <hpx/hpx_start.hpp>
#include <hpx/init.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr const char * STEP = "HPX_CB_CANCEL_STEP4";

void emit_pass() {
    fprintf(stdout, "%s: PASS\n", STEP);
}

void emit_fail(const char * reason) {
    fprintf(stdout, "%s: FAIL: %s\n", STEP, reason);
}

// ---- Trace (env-gated lifecycle events on stderr) -----------------------
namespace trace {

std::atomic<bool> g_enabled{false};
std::once_flag    g_init_flag;

void init() noexcept {
    std::call_once(g_init_flag, []() {
        const char * v = std::getenv("LLAMA_HPX_CB_TRACE");
        const bool on = v != nullptr && v[0] == '1' && v[1] == '\0';
        g_enabled.store(on, std::memory_order_release);
    });
}

bool enabled() noexcept {
    return g_enabled.load(std::memory_order_acquire);
}

// Lifecycle event. Format: `[hpx-cb-gate] event=<fmt-output>`.
// `fmt` should start with the event name and continue with
// space-separated key=value pairs (grep-friendly, stable).
void event(const char * fmt, ...) noexcept {
    if (!enabled()) return;
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "[hpx-cb-gate] event=");
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    fflush(stderr);
}

}  // namespace trace

// ---- Token hashing (byte-identical to multiseq-batch-gate) --------------
constexpr uint64_t k_token_hash_init  = 0xcbf29ce484222325ull;
constexpr uint64_t k_token_hash_empty = 0ull;
constexpr uint64_t k_token_hash_prime = 0x100000001b3ull;

uint64_t fold_token_hash(uint64_t state, int32_t token_id) noexcept {
    state ^= static_cast<uint64_t>(static_cast<uint32_t>(token_id));
    state *= k_token_hash_prime;
    return state;
}

llama_token argmax(const float * logits, int32_t n_vocab) {
    llama_token best   = 0;
    float       best_v = logits[0];
    for (int32_t i = 1; i < n_vocab; i++) {
        if (logits[i] > best_v) {
            best_v = logits[i];
            best   = static_cast<llama_token>(i);
        }
    }
    return best;
}

// Linear-interp percentile over a copy of the input (caller-friendly).
double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    if (v.size() == 1) return v[0];
    if (p <= 0.0) return v.front();
    if (p >= 1.0) return v.back();
    const double idx  = p * static_cast<double>(v.size() - 1);
    const size_t lo   = static_cast<size_t>(idx);
    const size_t hi   = std::min(lo + 1, v.size() - 1);
    const double frac = idx - static_cast<double>(lo);
    return v[lo] + frac * (v[hi] - v[lo]);
}

double mean_of(const std::vector<double> & v) {
    if (v.empty()) return 0.0;
    double s = 0.0;
    for (double x : v) s += x;
    return s / static_cast<double>(v.size());
}

// ---- Request lifecycle status (Cancel Slice 1: data model only) --------
enum class request_status : uint8_t {
    completed       = 0,
    cancelled       = 1,
    failed_reserved = 2,  // reserved for future slices; not produced now
};

const char * status_name(request_status s) noexcept {
    switch (s) {
        case request_status::completed:        return "completed";
        case request_status::cancelled:        return "cancelled";
        case request_status::failed_reserved:  return "failed_reserved";
    }
    return "unknown";
}

// ---- Per-seq state (engine-internal) ------------------------------------
// Cancellation fields are present in Cancel Slice 1 but dormant: the
// plan is not propagated into seq_state and the engine does not
// observe cancellation. Slice 2 will wire those.
struct seq_state {
    int32_t                  seq_id                       = 0;
    int32_t                  decode_budget                = 0;
    int32_t                  n_decoded                    = 0;
    int32_t                  pos_next                     = 0;
    int32_t                  i_batch                      = -1;
    llama_token              last_token                   = 0;
    bool                     done                         = false;
    int32_t                  done_iter                    = -1;
    llama_pos                pos_max_at_clear             = -1;
    bool                     kv_cleared                   = false;
    bool                     promise_fulfilled            = false;
    uint64_t                 hash_state                   = k_token_hash_init;
    std::vector<llama_token> generated_tokens;

    // ---- cancellation data model (Cancel Slice 1: not exercised) -------
    std::atomic<bool>        cancel_requested{false};
    int32_t                  cancel_after_decoded_tokens  = -1;  // -1 = never
    bool                     cancel_observed              = false;
    int32_t                  cancel_observed_iter         = -1;
    int32_t                  n_decoded_at_cancel          = -1;

    uint64_t finalize_hash() const noexcept {
        return (n_decoded == 0) ? k_token_hash_empty : hash_state;
    }
};

// ---- Per-request snapshot (the only thing crossing the engine boundary) -
struct request_result {
    int32_t                  request_id           = 0;
    int32_t                  seq_id               = 0;
    int32_t                  decode_budget        = 0;
    int32_t                  n_decoded            = 0;
    uint64_t                 hash                 = 0;
    int32_t                  done_iter            = -1;
    llama_pos                pos_max_at_clear     = -1;
    bool                     kv_cleared           = false;
    int64_t                  ttc_us               = 0;
    std::vector<llama_token> generated_tokens;

    // ---- cancellation snapshot (Cancel Slice 1: defaults always) -------
    request_status           status               = request_status::completed;
    int32_t                  cancel_observed_iter = -1;
    int32_t                  n_decoded_at_cancel  = -1;
};

// ---- Engine metrics -----------------------------------------------------
struct engine_metrics {
    double               wall_ms              = 0.0;
    int32_t              decode_calls         = 0;
    int32_t              update_iterations    = 0;
    std::vector<int32_t> rows_per_batch;       // one per llama_decode call
    std::vector<int32_t> active_seqs_per_iter; // one per decode iteration
};

// ---- Engine result (engine-task-level diagnostics) ----------------------
struct engine_result {
    bool           ok                 = false;
    std::string    error;
    int32_t        decode_calls       = 0;   // mirror of metrics.decode_calls
    int32_t        decode_failures    = 0;
    int32_t        engine_task_count  = 0;
    int32_t        promises_fulfilled = 0;
    int32_t        cancelled_count    = 0;   // seqs fulfilled with status=cancelled
    bool           residual_kv_ok     = false;
    std::string    residual_kv_error;
    engine_metrics metrics;
};

// ---- Engine class -------------------------------------------------------
class engine {
public:
    engine(llama_context *                  ctx,
           const llama_vocab *              vocab,
           int32_t                          n_vocab,
           const std::vector<llama_token> & prompt_tokens,
           std::vector<int32_t>             budgets,
           int32_t                          batch_capacity,
           const std::vector<int32_t> &     cancel_plan,
           int32_t                          cancel_after)
        : ctx_(ctx),
          vocab_(vocab),
          n_vocab_(n_vocab),
          prompt_tokens_(prompt_tokens),
          budgets_(std::move(budgets)),
          batch_capacity_(batch_capacity),
          promises_(budgets_.size()),
          seqs_(budgets_.size())
    {
        for (size_t s = 0; s < budgets_.size(); s++) {
            seqs_[s].seq_id        = static_cast<int32_t>(s);
            seqs_[s].decode_budget = budgets_[s];
            seqs_[s].generated_tokens.reserve(
                static_cast<size_t>(budgets_[s]));
        }
        // Cancel Slice 2: propagate the deterministic plan into
        // seq_state. Out-of-range seq_ids are silently skipped here
        // because main has already validated the plan against
        // [0, n_seqs) and printed the per-seq budget assertion.
        for (int32_t cs : cancel_plan) {
            if (cs >= 0 && static_cast<size_t>(cs) < seqs_.size()) {
                seqs_[static_cast<size_t>(cs)]
                    .cancel_after_decoded_tokens = cancel_after;
            }
        }
    }

    // Pull the futures *before* scheduling run(). Called from main.
    std::vector<hpx::future<request_result>> take_futures() {
        std::vector<hpx::future<request_result>> futs;
        futs.reserve(promises_.size());
        for (auto & p : promises_) {
            futs.emplace_back(p.get_future());
        }
        return futs;
    }

    // Sole entry point that touches llama_context / llama_batch /
    // llama_decode / llama_memory_seq_* / llama_get_logits_ith.
    // Designed to run inside exactly one HPX task per engine instance.
    void run() {
        result_.engine_task_count++;
        run_body();
        // Drain any unfulfilled promises so main's wait_all can never
        // deadlock if the decode loop bailed out early.
        for (size_t s = 0; s < promises_.size(); s++) {
            if (seqs_[s].promise_fulfilled) continue;
            const std::string msg = result_.error.empty()
                ? "engine ended without fulfilling promise"
                : result_.error;
            try {
                promises_[s].set_exception(
                    std::make_exception_ptr(std::runtime_error(msg)));
            } catch (...) {
                // promise already satisfied; ignore.
            }
        }
    }

    const engine_result & result() const noexcept { return result_; }
    int32_t n_seqs() const noexcept {
        return static_cast<int32_t>(seqs_.size());
    }

private:
    bool clear_and_check(seq_state & seq, int32_t iter,
                         llama_memory_t mem) {
        seq.done             = true;
        seq.done_iter        = iter;
        seq.pos_max_at_clear = llama_memory_seq_pos_max(mem, seq.seq_id);

        std::vector<llama_pos> sib_min(seqs_.size(), -1);
        std::vector<llama_pos> sib_max(seqs_.size(), -1);
        for (size_t t = 0; t < seqs_.size(); t++) {
            const seq_state & ot = seqs_[t];
            if (ot.seq_id == seq.seq_id || ot.kv_cleared || ot.done) continue;
            sib_min[t] = llama_memory_seq_pos_min(mem, ot.seq_id);
            sib_max[t] = llama_memory_seq_pos_max(mem, ot.seq_id);
        }

        if (!llama_memory_seq_rm(mem, seq.seq_id, /*p0=*/-1, /*p1=*/-1)) {
            result_.error = "llama_memory_seq_rm returned false";
            return false;
        }
        seq.kv_cleared = true;

        const llama_pos pmin = llama_memory_seq_pos_min(mem, seq.seq_id);
        const llama_pos pmax = llama_memory_seq_pos_max(mem, seq.seq_id);
        if (pmin != -1 || pmax != -1) {
            char buf[200];
            std::snprintf(buf, sizeof(buf),
                "seq %d not empty after seq_rm: pos_min=%d pos_max=%d",
                seq.seq_id, pmin, pmax);
            result_.error = buf;
            return false;
        }

        for (size_t t = 0; t < seqs_.size(); t++) {
            const seq_state & ot = seqs_[t];
            if (ot.seq_id == seq.seq_id || ot.kv_cleared || ot.done) continue;
            const llama_pos tmin = llama_memory_seq_pos_min(mem, ot.seq_id);
            const llama_pos tmax = llama_memory_seq_pos_max(mem, ot.seq_id);
            if (tmin != sib_min[t] || tmax != sib_max[t]) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "cross-talk: clearing seq %d disturbed seq %d "
                    "(pos_min=%d pos_max=%d, expected %d %d)",
                    seq.seq_id, ot.seq_id, tmin, tmax,
                    sib_min[t], sib_max[t]);
                result_.error = buf;
                return false;
            }
        }
        return true;
    }

    // Build the request_result snapshot and call set_value on the
    // matching promise exactly once. Shared by completion and
    // cancellation paths; the per-status fields and trace events
    // are owned by the callers.
    bool fulfill_promise(seq_state &     seq,
                         request_status  status,
                         int64_t         ttc_us) {
        request_result rr;
        rr.request_id       = seq.seq_id;
        rr.seq_id           = seq.seq_id;
        rr.decode_budget    = seq.decode_budget;
        rr.n_decoded        = seq.n_decoded;
        rr.hash             = seq.finalize_hash();
        rr.done_iter        = seq.done_iter;
        rr.pos_max_at_clear = seq.pos_max_at_clear;
        rr.kv_cleared       = seq.kv_cleared;
        rr.ttc_us           = ttc_us;
        rr.generated_tokens = seq.generated_tokens;
        rr.status           = status;
        if (status == request_status::cancelled) {
            rr.cancel_observed_iter = seq.cancel_observed_iter;
            rr.n_decoded_at_cancel  = seq.n_decoded_at_cancel;
        } else {
            rr.cancel_observed_iter = -1;
            rr.n_decoded_at_cancel  = -1;
        }

        try {
            promises_[static_cast<size_t>(seq.seq_id)]
                .set_value(std::move(rr));
        } catch (const std::exception & e) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "promise.set_value threw for seq %d: %s",
                seq.seq_id, e.what());
            result_.error = buf;
            return false;
        }
        seq.promise_fulfilled = true;
        result_.promises_fulfilled++;
        return true;
    }

    // Completion path: emit seq_complete, clear KV (cross-talk
    // check), emit kv_cleared, fulfill promise with status=completed,
    // emit promise_fulfilled. Cancellation never reaches this path.
    bool finalize_and_fulfill(seq_state & seq, int32_t iter,
                              llama_memory_t mem) {
        const uint64_t hash_now = seq.finalize_hash();
        trace::event(
            "seq_complete seq=%d budget=%d done_iter=%d hash=0x%016llx",
            seq.seq_id, seq.decode_budget, iter,
            static_cast<unsigned long long>(hash_now));

        if (!clear_and_check(seq, iter, mem)) return false;

        trace::event(
            "kv_cleared seq=%d pos_max_at_clear=%d cross_talk_ok=1",
            seq.seq_id, seq.pos_max_at_clear);

        const auto    now    = std::chrono::steady_clock::now();
        const int64_t ttc_us = std::chrono::duration_cast<
            std::chrono::microseconds>(now - t_start_).count();

        if (!fulfill_promise(seq, request_status::completed, ttc_us)) {
            return false;
        }

        trace::event("promise_fulfilled seq=%d ttc_us=%lld",
                     seq.seq_id, static_cast<long long>(ttc_us));
        return true;
    }

    // Cooperative cancellation observation predicate. Called only at
    // iteration boundaries (top of decode iter, post-prefill).
    bool cancel_should_observe(const seq_state & seq) const noexcept {
        if (seq.done || seq.cancel_observed) return false;
        if (seq.cancel_requested.load(std::memory_order_acquire)) {
            return true;
        }
        if (seq.cancel_after_decoded_tokens >= 0
         && seq.n_decoded >= seq.cancel_after_decoded_tokens) {
            return true;
        }
        return false;
    }

    // Cancellation path: record observation, clear KV (cross-talk
    // check), fulfill promise with status=cancelled. Mirrors
    // finalize_and_fulfill but emits cancel-family trace events.
    bool cancel_and_fulfill(seq_state & seq, int32_t iter,
                            llama_memory_t mem) {
        seq.cancel_observed       = true;
        seq.cancel_observed_iter  = iter;
        seq.n_decoded_at_cancel   = seq.n_decoded;

        trace::event(
            "cancel_observed seq=%d iter=%d n_decoded=%d",
            seq.seq_id, iter, seq.n_decoded_at_cancel);

        if (!clear_and_check(seq, iter, mem)) return false;

        trace::event(
            "cancel_kv_cleared seq=%d pos_max_at_clear=%d "
            "cross_talk_ok=1",
            seq.seq_id, seq.pos_max_at_clear);

        const auto    now    = std::chrono::steady_clock::now();
        const int64_t ttc_us = std::chrono::duration_cast<
            std::chrono::microseconds>(now - t_start_).count();

        if (!fulfill_promise(seq, request_status::cancelled, ttc_us)) {
            return false;
        }
        result_.cancelled_count++;

        trace::event(
            "cancel_future_fulfilled seq=%d status=cancelled ttc_us=%lld",
            seq.seq_id, static_cast<long long>(ttc_us));
        return true;
    }

    void run_body() {
        result_.ok = false;
        result_.error.clear();
        result_.decode_calls       = 0;
        result_.decode_failures    = 0;
        result_.promises_fulfilled = 0;
        result_.cancelled_count    = 0;
        result_.residual_kv_ok     = false;
        result_.residual_kv_error.clear();
        result_.metrics = engine_metrics{};

        auto * mem = llama_get_memory(ctx_);
        llama_memory_clear(mem, /*data=*/false);

        const int32_t n_seqs   = static_cast<int32_t>(budgets_.size());
        const int32_t n_prompt = static_cast<int32_t>(prompt_tokens_.size());
        int32_t       max_budget = 0;
        for (int32_t b : budgets_) {
            if (b > max_budget) max_budget = b;
        }

        // Reset per-seq dynamic state for this run (--repeat re-uses
        // the same engine's seqs_; configured fields stay set).
        for (auto & seq : seqs_) {
            seq.n_decoded         = 0;
            seq.pos_next          = 0;
            seq.i_batch           = -1;
            seq.last_token        = 0;
            seq.done              = false;
            seq.done_iter         = -1;
            seq.pos_max_at_clear  = -1;
            seq.kv_cleared        = false;
            seq.promise_fulfilled = false;
            seq.hash_state        = k_token_hash_init;
            seq.generated_tokens.clear();
        }

        t_start_ = std::chrono::steady_clock::now();

        trace::event("engine_start n_seqs=%d prompt_tokens=%d max_budget=%d",
                     n_seqs, n_prompt, max_budget);
        for (int32_t s = 0; s < n_seqs; s++) {
            trace::event("request_admitted seq=%d budget=%d",
                         seqs_[s].seq_id, seqs_[s].decode_budget);
        }
        // Cancel Slice 3: log every seq carrying a deterministic
        // cancel trigger, in the spec-aligned grep-friendly format.
        // External-cancel (atomic flag) sources would emit
        // cancel_requested at flag-set time; this slice does not
        // exercise that path.
        for (int32_t s = 0; s < n_seqs; s++) {
            if (seqs_[s].cancel_after_decoded_tokens >= 0) {
                trace::event(
                    "cancel_requested seq=%d budget=%d cancel_after=%d",
                    seqs_[s].seq_id, seqs_[s].decode_budget,
                    seqs_[s].cancel_after_decoded_tokens);
            }
        }

        llama_batch batch = llama_batch_init(batch_capacity_, /*embd=*/0,
                                             /*n_seq_max=*/1);

        // ---- Prefill pass (one shared batch over all seqs) -------------
        common_batch_clear(batch);
        for (int32_t s = 0; s < n_seqs; s++) {
            seq_state & seq = seqs_[s];
            for (int32_t p = 0; p < n_prompt; p++) {
                const bool last = (p == n_prompt - 1);
                common_batch_add(batch, prompt_tokens_[p], /*pos=*/p,
                                 /*seq_ids=*/{seq.seq_id}, /*logits=*/last);
                if (last) {
                    seq.i_batch = batch.n_tokens - 1;
                }
            }
            seq.pos_next = n_prompt;
        }

        result_.metrics.rows_per_batch.push_back(batch.n_tokens);
        result_.decode_calls++;
        result_.metrics.decode_calls = result_.decode_calls;
        if (llama_decode(ctx_, batch) != 0) {
            result_.decode_failures++;
            result_.error = "llama_decode failed during prefill";
            llama_batch_free(batch);
            finalize_wall_ms();
            return;
        }
        llama_synchronize(ctx_);

        const int32_t prefill_iter = 0;
        for (int32_t s = 0; s < n_seqs; s++) {
            seq_state & seq = seqs_[s];
            const float * logits = llama_get_logits_ith(ctx_, seq.i_batch);
            if (logits == nullptr) {
                result_.error = "llama_get_logits_ith returned null after prefill";
                llama_batch_free(batch);
                finalize_wall_ms();
                return;
            }
            const llama_token next_id = argmax(logits, n_vocab_);
            trace::event("seq_prefilled seq=%d first_token=%d",
                         seq.seq_id, static_cast<int>(next_id));
            if (llama_vocab_is_eog(vocab_, next_id)) {
                if (!finalize_and_fulfill(seq, prefill_iter, mem)) {
                    llama_batch_free(batch);
                    finalize_wall_ms();
                    return;
                }
                continue;
            }
            seq.generated_tokens.push_back(next_id);
            seq.hash_state = fold_token_hash(seq.hash_state,
                                             static_cast<int32_t>(next_id));
            seq.n_decoded++;
            seq.last_token = next_id;
            if (seq.n_decoded >= seq.decode_budget) {
                if (!finalize_and_fulfill(seq, prefill_iter, mem)) {
                    llama_batch_free(batch);
                    finalize_wall_ms();
                    return;
                }
            }
        }

        auto any_active = [&]() {
            for (const auto & seq : seqs_) {
                if (!seq.done) return true;
            }
            return false;
        };

        // Cancellation observation #1: post-prefill boundary. Equivalent
        // to "before decode iter 1 builds rows". For the design's
        // smoke shape (cancel_after=16) this is a no-op because
        // n_decoded == 1 after prefill, but a smaller cancel_after
        // (e.g. 0 or 1) would fire here.
        for (int32_t s = 0; s < n_seqs; s++) {
            seq_state & seq = seqs_[s];
            if (seq.done) continue;
            if (cancel_should_observe(seq)) {
                if (!cancel_and_fulfill(seq, /*iter=*/0, mem)) {
                    llama_batch_free(batch);
                    finalize_wall_ms();
                    return;
                }
            }
        }

        // ---- Decode loop (one row per still-active seq per iter) -------
        int32_t iter = 0;
        while (any_active()) {
            iter++;

            // Cancellation observation #2: top of each decode iter,
            // BEFORE building active_idx. A cancelled seq becomes
            // seq.done=true via clear_and_check, so the active_idx
            // loop below skips it naturally. This makes
            // wasted_decode_rows_after_cancel structurally 0.
            for (int32_t s = 0; s < n_seqs; s++) {
                seq_state & seq = seqs_[s];
                if (seq.done) continue;
                if (cancel_should_observe(seq)) {
                    if (!cancel_and_fulfill(seq, iter, mem)) {
                        llama_batch_free(batch);
                        result_.metrics.update_iterations = iter;
                        finalize_wall_ms();
                        return;
                    }
                }
            }
            // If every still-active seq was just cancelled, end the
            // loop cleanly without trying to decode an empty batch.
            if (!any_active()) {
                result_.metrics.update_iterations = iter;
                break;
            }

            common_batch_clear(batch);
            std::vector<int32_t> active_idx;
            active_idx.reserve(static_cast<size_t>(n_seqs));
            for (int32_t s = 0; s < n_seqs; s++) {
                seq_state & seq = seqs_[s];
                if (seq.done) continue;
                common_batch_add(batch, seq.last_token, /*pos=*/seq.pos_next,
                                 /*seq_ids=*/{seq.seq_id}, /*logits=*/true);
                seq.i_batch = batch.n_tokens - 1;
                seq.pos_next++;
                active_idx.push_back(s);
                trace::event("decode_row iter=%d seq=%d pos=%d",
                             iter, seq.seq_id, seq.pos_next - 1);
            }
            if (batch.n_tokens == 0) break;  // defensive

            result_.metrics.active_seqs_per_iter.push_back(
                static_cast<int32_t>(active_idx.size()));
            result_.metrics.rows_per_batch.push_back(batch.n_tokens);
            result_.decode_calls++;
            result_.metrics.decode_calls = result_.decode_calls;

            if (llama_decode(ctx_, batch) != 0) {
                result_.decode_failures++;
                result_.error = "llama_decode failed during decode loop";
                llama_batch_free(batch);
                result_.metrics.update_iterations = iter;
                finalize_wall_ms();
                return;
            }
            llama_synchronize(ctx_);

            for (int32_t s : active_idx) {
                seq_state & seq = seqs_[s];
                const float * logits = llama_get_logits_ith(ctx_, seq.i_batch);
                if (logits == nullptr) {
                    result_.error = "llama_get_logits_ith returned null during decode";
                    llama_batch_free(batch);
                    result_.metrics.update_iterations = iter;
                    finalize_wall_ms();
                    return;
                }
                const llama_token next_id = argmax(logits, n_vocab_);
                if (llama_vocab_is_eog(vocab_, next_id)) {
                    if (!finalize_and_fulfill(seq, iter, mem)) {
                        llama_batch_free(batch);
                        result_.metrics.update_iterations = iter;
                        finalize_wall_ms();
                        return;
                    }
                    continue;
                }
                seq.generated_tokens.push_back(next_id);
                seq.hash_state = fold_token_hash(seq.hash_state,
                                                 static_cast<int32_t>(next_id));
                seq.n_decoded++;
                seq.last_token = next_id;
                if (seq.n_decoded >= seq.decode_budget) {
                    if (!finalize_and_fulfill(seq, iter, mem)) {
                        llama_batch_free(batch);
                        result_.metrics.update_iterations = iter;
                        finalize_wall_ms();
                        return;
                    }
                }
            }
        }

        result_.metrics.update_iterations = iter;
        llama_batch_free(batch);

        // ---- Residual-KV-empty check (engine-side; main never touches
        //      llama_memory_seq_*). -----------------------------------
        bool all_clear = true;
        for (int32_t s = 0; s < n_seqs; s++) {
            const llama_pos pmin = llama_memory_seq_pos_min(mem, s);
            const llama_pos pmax = llama_memory_seq_pos_max(mem, s);
            if (pmin != -1 || pmax != -1) {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                    "seq %d residual KV not empty: pos_min=%d pos_max=%d",
                    s, pmin, pmax);
                result_.residual_kv_error = buf;
                all_clear = false;
                break;
            }
        }
        result_.residual_kv_ok = all_clear;

        if (!all_clear) {
            if (result_.error.empty()) {
                result_.error = result_.residual_kv_error;
            }
            finalize_wall_ms();
            return;
        }

        finalize_wall_ms();

        trace::event(
            "engine_stop decode_calls=%d update_iterations=%d wall_ms=%.3f",
            result_.decode_calls, result_.metrics.update_iterations,
            result_.metrics.wall_ms);

        result_.ok = true;
    }

    void finalize_wall_ms() {
        const auto now = std::chrono::steady_clock::now();
        const auto us  = std::chrono::duration_cast<std::chrono::microseconds>(
            now - t_start_).count();
        result_.metrics.wall_ms = static_cast<double>(us) / 1000.0;
    }

private:
    llama_context *                  ctx_;
    const llama_vocab *              vocab_;
    int32_t                          n_vocab_;
    const std::vector<llama_token> & prompt_tokens_;
    std::vector<int32_t>             budgets_;
    int32_t                          batch_capacity_;
    std::vector<hpx::promise<request_result>> promises_;
    std::vector<seq_state>           seqs_;
    engine_result                    result_;
    std::chrono::steady_clock::time_point t_start_{};
};

// ---- Process-wide HPX runtime (mirrors serving-bench/runtime_hpx.cpp) ---
namespace hpx_runtime {

std::once_flag    g_start_flag;
std::atomic<bool> g_running{false};

bool trace_enabled() noexcept {
    const char * v = std::getenv("LLAMA_HPX_CB_TRACE");
    return v != nullptr && v[0] == '1' && v[1] == '\0';
}

bool do_start(int32_t os_threads) noexcept {
    try {
        hpx::init_params init_args;
        if (os_threads > 0) {
            init_args.cfg.push_back(
                std::string("hpx.os_threads=") +
                std::to_string(os_threads));
        }
        if (!hpx::start(nullptr, 0, nullptr, init_args)) {
            fprintf(stderr,
                "[hpx-cb-gate] hpx::start returned false\n");
            return false;
        }
        return true;
    } catch (const std::exception & e) {
        fprintf(stderr,
            "[hpx-cb-gate] hpx::start threw: %s\n", e.what());
        return false;
    } catch (...) {
        fprintf(stderr, "[hpx-cb-gate] hpx::start threw unknown\n");
        return false;
    }
}

bool start_once(int32_t os_threads) {
    std::call_once(g_start_flag, [&]() {
        if (trace_enabled()) {
            fprintf(stderr,
                "[hpx-cb-gate] hpx_runtime_start_once: starting "
                "(os_threads=%d)\n", os_threads);
        }
        if (do_start(os_threads)) {
            g_running.store(true);
        }
    });
    return g_running.load();
}

void stop() {
    if (!g_running.exchange(false)) {
        return;
    }
    if (trace_enabled()) {
        fprintf(stderr, "[hpx-cb-gate] hpx_runtime_stop\n");
    }
    try {
        hpx::post([]() { hpx::finalize(); });
        hpx::stop();
    } catch (const std::exception & e) {
        fprintf(stderr,
            "[hpx-cb-gate] hpx::stop threw: %s\n", e.what());
    } catch (...) {
        fprintf(stderr, "[hpx-cb-gate] hpx::stop threw unknown\n");
    }
}

}  // namespace hpx_runtime

// ---- CLI args -----------------------------------------------------------
struct cli_args {
    std::string          model_path;
    std::string          prompt        = "Hello, my name is";
    int32_t              ctx_size      = 32768;
    int32_t              n_seq_max     = 99;
    int32_t              n_batch       = 1024;
    int32_t              n_threads     = 2;
    int32_t              n_seqs        = 99;
    int32_t              hpx_os_threads = 1;
    int32_t              repeat        = 1;
    std::vector<int32_t> decode_budget_mix = {8, 64, 256};

    // Cancel Slice 1: plan is parsed and printed, asserted against the
    // round-robin budget mapping, but NOT propagated into seq_state and
    // NOT observed by the engine. Defaults match the design's smoke
    // shape: cancel 3 budget-64 seqs (1,4,7) and 3 budget-256 seqs
    // (2,5,8) after 16 decoded tokens each.
    std::vector<int32_t> cancel_plan   = {1, 4, 7, 2, 5, 8};
    int32_t              cancel_after  = 16;
};

void print_usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s --model <path> [options]\n"
        "  --model <path>            (required) path to .gguf model file\n"
        "  --prompt <string>         default: \"Hello, my name is\"\n"
        "  --ctx-size <int>          default: 32768\n"
        "  --n-seq-max <int>         default: 99\n"
        "  --n-batch <int>           default: 1024\n"
        "  --n-threads <int>         default: 2  (libllama compute threads)\n"
        "  --n-seqs <int>            default: 99\n"
        "  --decode-budget-mix <csv> default: \"8,64,256\"\n"
        "  --hpx-os-threads <int>    default: 1  (HPX orchestration threads)\n"
        "  --repeat <int>            default: 1  (engine task per repeat)\n"
        "  --cancel-plan <csv>       default: \"1,4,7,2,5,8\"\n"
        "                            seq_ids (>=0) cancelled at iteration\n"
        "                            boundaries once they reach\n"
        "                            --cancel-after decoded tokens. Use\n"
        "                            \"none\" for an explicit empty plan.\n"
        "  --cancel-after <int>      default: 16\n"
        "                            (decoded-token threshold for the plan)\n",
        argv0);
}

bool parse_int(const char * s, int32_t & out) {
    if (s == nullptr || *s == '\0') return false;
    char * end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (end == s || *end != '\0') return false;
    if (v < INT32_MIN || v > INT32_MAX) return false;
    out = static_cast<int32_t>(v);
    return true;
}

bool parse_csv_int_list(const char * s, std::vector<int32_t> & out,
                        int32_t min_v = 1) {
    out.clear();
    if (s == nullptr || *s == '\0') return false;
    const std::string str = s;
    size_t i = 0;
    while (i <= str.size()) {
        size_t j = str.find(',', i);
        if (j == std::string::npos) j = str.size();
        const std::string tok = str.substr(i, j - i);
        if (tok.empty()) return false;
        int32_t v = 0;
        if (!parse_int(tok.c_str(), v)) return false;
        if (v < min_v) return false;
        out.push_back(v);
        if (j == str.size()) break;
        i = j + 1;
    }
    return !out.empty();
}

bool parse_args(int argc, char ** argv, cli_args & args) {
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto need = [&](const char * flag) -> const char * {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: %s requires a value\n", flag);
                return nullptr;
            }
            return argv[++i];
        };
        if (a == "--model") {
            const char * v = need("--model");
            if (!v) return false;
            args.model_path = v;
        } else if (a == "--prompt") {
            const char * v = need("--prompt");
            if (!v) return false;
            args.prompt = v;
        } else if (a == "--ctx-size") {
            const char * v = need("--ctx-size");
            if (!v || !parse_int(v, args.ctx_size)) return false;
        } else if (a == "--n-seq-max") {
            const char * v = need("--n-seq-max");
            if (!v || !parse_int(v, args.n_seq_max)) return false;
        } else if (a == "--n-batch") {
            const char * v = need("--n-batch");
            if (!v || !parse_int(v, args.n_batch)) return false;
        } else if (a == "--n-threads") {
            const char * v = need("--n-threads");
            if (!v || !parse_int(v, args.n_threads)) return false;
        } else if (a == "--n-seqs") {
            const char * v = need("--n-seqs");
            if (!v || !parse_int(v, args.n_seqs)) return false;
            if (args.n_seqs <= 0) return false;
        } else if (a == "--decode-budget-mix") {
            const char * v = need("--decode-budget-mix");
            if (!v || !parse_csv_int_list(v, args.decode_budget_mix)) {
                fprintf(stderr,
                    "error: --decode-budget-mix must be a non-empty CSV "
                    "of positive integers\n");
                return false;
            }
        } else if (a == "--hpx-os-threads") {
            const char * v = need("--hpx-os-threads");
            if (!v || !parse_int(v, args.hpx_os_threads)) return false;
            if (args.hpx_os_threads <= 0) return false;
        } else if (a == "--repeat") {
            const char * v = need("--repeat");
            if (!v || !parse_int(v, args.repeat)) return false;
            if (args.repeat <= 0) return false;
        } else if (a == "--cancel-plan") {
            const char * v = need("--cancel-plan");
            if (!v) return false;
            // Cancel Slice 2: --cancel-plan now accepts seq_id 0
            // (parse_csv_int_list takes min_v=0). The literal "none"
            // is an explicit empty-plan sentinel.
            if (std::strcmp(v, "none") == 0) {
                args.cancel_plan.clear();
            } else if (!parse_csv_int_list(v, args.cancel_plan,
                                           /*min_v=*/0)) {
                fprintf(stderr,
                    "error: --cancel-plan must be a non-empty CSV "
                    "of non-negative seq_ids, or the literal \"none\"\n");
                return false;
            }
        } else if (a == "--cancel-after") {
            const char * v = need("--cancel-after");
            if (!v || !parse_int(v, args.cancel_after)) return false;
            if (args.cancel_after < 0) {
                fprintf(stderr,
                    "error: --cancel-after must be >= 0\n");
                return false;
            }
        } else if (a == "-h" || a == "--help") {
            print_usage(argv[0]);
            return false;
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", a.c_str());
            print_usage(argv[0]);
            return false;
        }
    }
    if (args.model_path.empty()) {
        fprintf(stderr, "error: --model is required\n");
        print_usage(argv[0]);
        return false;
    }
    if (args.n_seq_max < args.n_seqs) {
        args.n_seq_max = args.n_seqs;
    }
    return true;
}

}  // namespace

int main(int argc, char ** argv) {
    cli_args args;
    if (!parse_args(argc, argv, args)) {
        return 2;
    }

    trace::init();

    // ---- Start HPX runtime first; fail closed if unavailable. -----------
    if (!hpx_runtime::start_once(args.hpx_os_threads)) {
        emit_fail("hpx runtime start failed");
        return 1;
    }

    common_init();
    llama_backend_init();

    auto cleanup_llama = [&](llama_context * ctx, llama_model * model) {
        if (ctx)   llama_free(ctx);
        if (model) llama_model_free(model);
        llama_backend_free();
    };

    // ---- Load model + create one context. ------------------------------
    llama_model_params model_params = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(
        args.model_path.c_str(), model_params);
    if (model == nullptr) {
        emit_fail("model load failed");
        cleanup_llama(nullptr, nullptr);
        hpx_runtime::stop();
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx           = static_cast<uint32_t>(args.ctx_size);
    ctx_params.n_batch         = static_cast<uint32_t>(args.n_batch);
    ctx_params.n_seq_max       = static_cast<uint32_t>(args.n_seq_max);
    ctx_params.n_threads       = args.n_threads;
    ctx_params.n_threads_batch = args.n_threads;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        emit_fail("context create failed");
        cleanup_llama(nullptr, model);
        hpx_runtime::stop();
        return 1;
    }

    const uint32_t actual_n_ctx     = llama_n_ctx(ctx);
    const uint32_t actual_n_seq_max = llama_n_seq_max(ctx);
    const uint32_t actual_n_batch   = llama_n_batch(ctx);

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> prompt_tokens =
        common_tokenize(ctx, args.prompt, /*add_special=*/true,
                        /*parse_special=*/true);
    const int32_t n_prompt_tokens =
        static_cast<int32_t>(prompt_tokens.size());

    std::vector<int32_t> budgets(static_cast<size_t>(args.n_seqs));
    {
        const size_t m = args.decode_budget_mix.size();
        for (int32_t i = 0; i < args.n_seqs; i++) {
            budgets[static_cast<size_t>(i)] =
                args.decode_budget_mix[static_cast<size_t>(i) % m];
        }
    }
    const int32_t max_budget =
        *std::max_element(budgets.begin(), budgets.end());

    fprintf(stdout, "model_path:          %s\n",  args.model_path.c_str());
    fprintf(stdout, "prompt:              \"%s\"\n", args.prompt.c_str());
    fprintf(stdout, "requested ctx_size:  %d\n", args.ctx_size);
    fprintf(stdout, "requested n_seq_max: %d\n", args.n_seq_max);
    fprintf(stdout, "requested n_batch:   %d\n", args.n_batch);
    fprintf(stdout, "actual n_ctx:        %u\n", actual_n_ctx);
    fprintf(stdout, "actual n_seq_max:    %u\n", actual_n_seq_max);
    fprintf(stdout, "actual n_batch:      %u\n", actual_n_batch);
    fprintf(stdout, "n_vocab:             %d\n", n_vocab);
    fprintf(stdout, "prompt_tokens:       %d\n", n_prompt_tokens);
    fprintf(stdout, "n_seqs:              %d\n", args.n_seqs);
    fprintf(stdout, "decode_budget_mix:   [");
    for (size_t i = 0; i < args.decode_budget_mix.size(); i++) {
        fprintf(stdout, "%s%d", i == 0 ? "" : ", ",
                args.decode_budget_mix[i]);
    }
    fprintf(stdout, "]\n");
    fprintf(stdout, "hpx_os_threads:      %d\n", args.hpx_os_threads);
    fprintf(stdout, "repeat:              %d\n", args.repeat);
    fprintf(stdout, "trace_enabled:       %d\n",
            trace::enabled() ? 1 : 0);
    fflush(stdout);

    auto fail_with = [&](const std::string & reason) {
        emit_fail(reason.c_str());
        cleanup_llama(ctx, model);
        hpx_runtime::stop();
    };

    // ---- Cancellation plan (Cancel Slice 4: closeout — no new
    //      behavior, polished metrics/reporting). ----------------------
    // Parse and print the plan. Assert each plan seq's actual budget
    // matches the round-robin mix prediction so the design's smoke
    // shape (seqs 1,4,7 -> budget 64; seqs 2,5,8 -> budget 256) is
    // verified against the current budget vector. The plan IS
    // propagated into seq_state and the engine observes cancellation
    // cooperatively at iteration boundaries.
    fprintf(stdout, "cancellation:        observed=1 "
                    "(Cancel Slice 4: cooperative)\n");
    fprintf(stdout, "cancel_after:        %d\n", args.cancel_after);
    fprintf(stdout, "cancel_plan:         [");
    for (size_t i = 0; i < args.cancel_plan.size(); i++) {
        fprintf(stdout, "%s%d", i == 0 ? "" : ", ",
                args.cancel_plan[i]);
    }
    fprintf(stdout, "]%s\n",
            args.cancel_plan.empty() ? " (empty)" : "");
    if (!args.cancel_plan.empty()) {
        fprintf(stdout, "planned cancel seqs:\n");
        for (int32_t cs : args.cancel_plan) {
            if (cs < 0 || cs >= args.n_seqs) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "cancel_plan seq_id=%d outside [0, n_seqs=%d)",
                    cs, args.n_seqs);
                fail_with(buf); return 1;
            }
            const int32_t actual_b   = budgets[static_cast<size_t>(cs)];
            const int32_t expected_b = args.decode_budget_mix[
                static_cast<size_t>(cs) % args.decode_budget_mix.size()];
            fprintf(stdout, "  seq=%d budget=%d\n", cs, actual_b);
            if (actual_b != expected_b) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "cancel_plan seq=%d: budget=%d != round-robin "
                    "expected %d (mix=[8,64,256])",
                    cs, actual_b, expected_b);
                fail_with(buf); return 1;
            }
        }
    }
    fflush(stdout);

    // ---- Structural prerequisites (Step 1 gates carried over). ---------
    if (actual_n_seq_max < static_cast<uint32_t>(args.n_seq_max)) {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "actual n_seq_max %u < requested n_seq_max %d",
            actual_n_seq_max, args.n_seq_max);
        fail_with(buf); return 1;
    }
    if (actual_n_ctx < static_cast<uint32_t>(n_prompt_tokens + 256)) {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "actual n_ctx %u < prompt_tokens %d + 256",
            actual_n_ctx, n_prompt_tokens);
        fail_with(buf); return 1;
    }
    if (actual_n_batch <
        static_cast<uint32_t>(args.n_seqs * n_prompt_tokens)) {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "actual n_batch %u < n_seqs %d * prompt_tokens %d",
            actual_n_batch, args.n_seqs, n_prompt_tokens);
        fail_with(buf); return 1;
    }
    if (actual_n_ctx <
        static_cast<uint32_t>(n_prompt_tokens + max_budget)) {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "actual n_ctx %u < prompt_tokens %d + max_budget %d",
            actual_n_ctx, n_prompt_tokens, max_budget);
        fail_with(buf); return 1;
    }

    const int32_t batch_capacity =
        std::max<int32_t>(args.n_seqs * n_prompt_tokens, args.n_seqs);

    constexpr uint64_t k_canonical_budget_8 = 0x0619d4d1900c2365ull;

    std::vector<request_result> first_results;

    for (int32_t r = 0; r < args.repeat; r++) {
        // One engine instance per iteration (own promises/futures set).
        engine eng(ctx, vocab, n_vocab, prompt_tokens, budgets,
                   batch_capacity, args.cancel_plan, args.cancel_after);

        std::vector<hpx::future<request_result>> futs = eng.take_futures();
        const int32_t futures_created =
            static_cast<int32_t>(futs.size());

        // Schedule engine::run() as exactly one HPX task; main thread
        // waits via wait_all on per-request futures + engine_fut.get().
        hpx::future<void> engine_fut = hpx::async([&eng]() {
            eng.run();
        });

        hpx::wait_all(futs);

        try {
            engine_fut.get();
        } catch (const std::exception & e) {
            std::string msg = std::string("engine task threw: ") + e.what();
            fail_with(msg); return 1;
        }

        const engine_result & er = eng.result();

        // Gate: engine_task_count == 1
        if (er.engine_task_count != 1) {
            char buf[128];
            std::snprintf(buf, sizeof(buf),
                "iter %d: engine_task_count=%d (expected 1)",
                r, er.engine_task_count);
            fail_with(buf); return 1;
        }
        // Gate: engine HPX task completed successfully
        if (!er.ok) {
            std::string msg = "iter " + std::to_string(r)
                + ": engine error: " + er.error;
            fail_with(msg); return 1;
        }
        // Gate: every llama_decode call returns 0
        if (er.decode_failures != 0) {
            char buf[128];
            std::snprintf(buf, sizeof(buf),
                "iter %d: decode_failures=%d (expected 0)",
                r, er.decode_failures);
            fail_with(buf); return 1;
        }
        // Gate: residual KV at end is empty (engine-side)
        if (!er.residual_kv_ok) {
            std::string msg = "iter " + std::to_string(r)
                + ": residual_kv_error: " + er.residual_kv_error;
            fail_with(msg); return 1;
        }

        // Gate: futures_created == n_seqs
        if (futures_created != args.n_seqs) {
            char buf[128];
            std::snprintf(buf, sizeof(buf),
                "iter %d: futures_created=%d (expected %d)",
                r, futures_created, args.n_seqs);
            fail_with(buf); return 1;
        }
        // Gate: promises_fulfilled == n_seqs
        if (er.promises_fulfilled != args.n_seqs) {
            char buf[128];
            std::snprintf(buf, sizeof(buf),
                "iter %d: promises_fulfilled=%d (expected %d)",
                r, er.promises_fulfilled, args.n_seqs);
            fail_with(buf); return 1;
        }

        // Gate: every request future is ready post-engine.
        int32_t futures_completed = 0;
        for (size_t i = 0; i < futs.size(); i++) {
            if (futs[i].is_ready()) {
                futures_completed++;
            } else {
                char buf[128];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: future %zu not ready after engine",
                    r, i);
                fail_with(buf); return 1;
            }
        }
        if (futures_completed != args.n_seqs) {
            char buf[128];
            std::snprintf(buf, sizeof(buf),
                "iter %d: futures_completed=%d (expected %d)",
                r, futures_completed, args.n_seqs);
            fail_with(buf); return 1;
        }

        // Extract request_result snapshots from every future.
        std::vector<request_result> results;
        results.reserve(futs.size());
        try {
            for (auto & f : futs) {
                results.push_back(f.get());
            }
        } catch (const std::exception & e) {
            std::string msg =
                std::string("future.get() threw: ") + e.what();
            fail_with(msg); return 1;
        }

        std::sort(results.begin(), results.end(),
                  [](const request_result & a, const request_result & b) {
                      return a.seq_id < b.seq_id;
                  });

        // Duplicate-seq-id guard.
        {
            std::vector<int32_t> seen;
            seen.reserve(results.size());
            for (const auto & rr : results) {
                for (int32_t s : seen) {
                    if (s == rr.seq_id) {
                        char buf[128];
                        std::snprintf(buf, sizeof(buf),
                            "iter %d: duplicate seq_id=%d in results",
                            r, rr.seq_id);
                        fail_with(buf); return 1;
                    }
                }
                seen.push_back(rr.seq_id);
            }
        }

        // Gate: kv_cleared == true on every fulfilled result.
        for (const auto & rr : results) {
            if (!rr.kv_cleared) {
                char buf[128];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: seq %d kv_cleared=false at fulfillment",
                    r, rr.seq_id);
                fail_with(buf); return 1;
            }
        }

        // ---- Cancel-Slice-2 per-result validation ---------------------
        // Build a lookup for planned-cancellation seq_ids.
        auto is_planned_cancel = [&](int32_t seq_id) {
            for (int32_t cs : args.cancel_plan) {
                if (cs == seq_id) return true;
            }
            return false;
        };

        // Expected anchor values for the smoke shape:
        //   At top of decode iter K, n_decoded == K (1 from prefill +
        //   K-1 from prior decode iters). The deterministic threshold
        //   `cancel_after_decoded_tokens` fires when n_decoded reaches
        //   that value; for cancel_after >= 1 the first iter at which
        //   the predicate is true is iter == cancel_after itself.
        //   (cancel_after == 0 fires at the post-prefill site with
        //   iter == 0; this slice's smoke shape uses cancel_after = 16.)
        const int32_t expected_n_decoded_at_cancel  = args.cancel_after;
        const int32_t expected_cancel_observed_iter = args.cancel_after;

        for (const auto & rr : results) {
            const bool planned = is_planned_cancel(rr.seq_id);
            if (planned) {
                if (rr.status != request_status::cancelled) {
                    char buf[200];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: seq %d status=%s (expected cancelled)",
                        r, rr.seq_id, status_name(rr.status));
                    fail_with(buf); return 1;
                }
                if (rr.n_decoded_at_cancel != expected_n_decoded_at_cancel) {
                    char buf[200];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: seq %d n_decoded_at_cancel=%d "
                        "(expected %d)", r, rr.seq_id,
                        rr.n_decoded_at_cancel,
                        expected_n_decoded_at_cancel);
                    fail_with(buf); return 1;
                }
                if (rr.cancel_observed_iter != expected_cancel_observed_iter) {
                    char buf[200];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: seq %d cancel_observed_iter=%d "
                        "(expected %d)", r, rr.seq_id,
                        rr.cancel_observed_iter,
                        expected_cancel_observed_iter);
                    fail_with(buf); return 1;
                }
                if (rr.n_decoded != expected_n_decoded_at_cancel) {
                    char buf[200];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: seq %d n_decoded=%d "
                        "(expected %d for cancelled)", r, rr.seq_id,
                        rr.n_decoded, expected_n_decoded_at_cancel);
                    fail_with(buf); return 1;
                }
                // Cancel Slice 3: explicit invariant — the decoded
                // token count at the moment of cancellation must
                // equal the snapshot's final n_decoded, since no
                // decode rows are added after observation.
                if (rr.n_decoded != rr.n_decoded_at_cancel) {
                    char buf[200];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: seq %d n_decoded=%d != "
                        "n_decoded_at_cancel=%d", r, rr.seq_id,
                        rr.n_decoded, rr.n_decoded_at_cancel);
                    fail_with(buf); return 1;
                }
                if (rr.n_decoded >= rr.decode_budget) {
                    char buf[200];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: seq %d cancelled but n_decoded=%d "
                        ">= budget=%d", r, rr.seq_id,
                        rr.n_decoded, rr.decode_budget);
                    fail_with(buf); return 1;
                }
                // wasted_decode_rows_after_cancel == 0 invariant
                // expressed per-seq: pos_max_at_clear must equal
                // n_prompt + n_decoded - 2 (no row was added at any
                // iter > cancel_observed_iter).
                const llama_pos expected_pos =
                    n_prompt_tokens + rr.n_decoded - 2;
                if (rr.pos_max_at_clear != expected_pos) {
                    char buf[256];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: seq %d cancelled pos_max_at_clear=%d "
                        "!= n_prompt+n_decoded-2=%d (extra rows added "
                        "after cancel observation)",
                        r, rr.seq_id, rr.pos_max_at_clear,
                        static_cast<int>(expected_pos));
                    fail_with(buf); return 1;
                }
            } else {
                if (rr.status != request_status::completed) {
                    char buf[200];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: seq %d status=%s (expected completed)",
                        r, rr.seq_id, status_name(rr.status));
                    fail_with(buf); return 1;
                }
                if (rr.cancel_observed_iter != -1) {
                    char buf[200];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: seq %d cancel_observed_iter=%d "
                        "(expected -1)", r, rr.seq_id,
                        rr.cancel_observed_iter);
                    fail_with(buf); return 1;
                }
                if (rr.n_decoded_at_cancel != -1) {
                    char buf[200];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: seq %d n_decoded_at_cancel=%d "
                        "(expected -1)", r, rr.seq_id,
                        rr.n_decoded_at_cancel);
                    fail_with(buf); return 1;
                }
                if (rr.n_decoded != rr.decode_budget) {
                    char buf[256];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: seq %d n_decoded %d != budget %d",
                        r, rr.seq_id, rr.n_decoded, rr.decode_budget);
                    fail_with(buf); return 1;
                }
            }
        }

        // ---- Per-budget completed summary + class gates. ---------------
        // Hash uniqueness, done_iter set, pos_max set are computed over
        // COMPLETED seqs only. Cancelled seqs of the same budget have
        // partial hashes and a different pos_max_at_clear and are
        // summarized in their own block below.
        std::vector<int32_t> uniq_budgets;
        for (const auto & rr : results) {
            bool seen = false;
            for (int32_t u : uniq_budgets) if (u == rr.decode_budget) { seen = true; break; }
            if (!seen) uniq_budgets.push_back(rr.decode_budget);
        }
        std::sort(uniq_budgets.begin(), uniq_budgets.end());

        for (int32_t b : uniq_budgets) {
            int32_t  completed_b = 0;
            int32_t  cancelled_b = 0;
            uint64_t any_hash    = 0;
            std::vector<uint64_t>  hashes;
            std::vector<int32_t>   done_iters;
            std::vector<llama_pos> pos_maxes;
            for (const auto & rr : results) {
                if (rr.decode_budget != b) continue;
                if (rr.status == request_status::completed) {
                    completed_b++;
                    any_hash = rr.hash;
                    bool h_seen = false;
                    for (uint64_t hh : hashes) if (hh == rr.hash) { h_seen = true; break; }
                    if (!h_seen) hashes.push_back(rr.hash);
                    bool d_seen = false;
                    for (int32_t dd : done_iters) if (dd == rr.done_iter) { d_seen = true; break; }
                    if (!d_seen) done_iters.push_back(rr.done_iter);
                    bool p_seen = false;
                    for (llama_pos pp : pos_maxes) if (pp == rr.pos_max_at_clear) { p_seen = true; break; }
                    if (!p_seen) pos_maxes.push_back(rr.pos_max_at_clear);
                } else if (rr.status == request_status::cancelled) {
                    cancelled_b++;
                }
            }
            std::sort(done_iters.begin(), done_iters.end());
            std::sort(pos_maxes.begin(),  pos_maxes.end());
            const int32_t expected_done_iter = b - 1;
            const int32_t expected_pos_max   = n_prompt_tokens + b - 2;

            fprintf(stdout,
                "iter[%d] budget=%d completed=%d cancelled=%d "
                "unique_completed_hashes=%zu hash=0x%016llx "
                "done_iter_set={", r, b, completed_b, cancelled_b,
                hashes.size(),
                static_cast<unsigned long long>(any_hash));
            for (size_t i = 0; i < done_iters.size(); i++) {
                fprintf(stdout, "%s%d", i == 0 ? "" : ",", done_iters[i]);
            }
            fprintf(stdout, "} pos_max_at_clear_set={");
            for (size_t i = 0; i < pos_maxes.size(); i++) {
                fprintf(stdout, "%s%d", i == 0 ? "" : ",", pos_maxes[i]);
            }
            fprintf(stdout, "}\n");

            if (completed_b > 0) {
                if (hashes.size() != 1) {
                    char buf[256];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: budget %d: %zu distinct hashes among "
                        "%d completed seqs (expected 1)",
                        r, b, hashes.size(), completed_b);
                    fail_with(buf); return 1;
                }
                if (done_iters.size() != 1
                    || done_iters[0] != expected_done_iter) {
                    char buf[256];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: budget %d completed: done_iter set "
                        "has %zu values, expected {%d}",
                        r, b, done_iters.size(), expected_done_iter);
                    fail_with(buf); return 1;
                }
                if (pos_maxes.size() != 1
                    || pos_maxes[0] != expected_pos_max) {
                    char buf[256];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: budget %d completed: pos_max set has "
                        "%zu values, expected {%d}",
                        r, b, pos_maxes.size(), expected_pos_max);
                    fail_with(buf); return 1;
                }
                if (b == 8 && any_hash != k_canonical_budget_8) {
                    char buf[256];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: budget 8 completed hash 0x%016llx "
                        "!= canonical 0x%016llx", r,
                        static_cast<unsigned long long>(any_hash),
                        static_cast<unsigned long long>(k_canonical_budget_8));
                    fail_with(buf); return 1;
                }
            }
        }

        // ---- Per-budget cancellation summary --------------------------
        for (int32_t b : uniq_budgets) {
            int32_t cancelled_b = 0;
            std::vector<int32_t> cancel_iters;
            std::vector<int32_t> cancel_n_decodes;
            for (const auto & rr : results) {
                if (rr.decode_budget != b) continue;
                if (rr.status != request_status::cancelled) continue;
                cancelled_b++;
                bool ci_seen = false;
                for (int32_t v : cancel_iters)
                    if (v == rr.cancel_observed_iter) { ci_seen = true; break; }
                if (!ci_seen) cancel_iters.push_back(rr.cancel_observed_iter);
                bool nd_seen = false;
                for (int32_t v : cancel_n_decodes)
                    if (v == rr.n_decoded_at_cancel) { nd_seen = true; break; }
                if (!nd_seen) cancel_n_decodes.push_back(rr.n_decoded_at_cancel);
            }
            if (cancelled_b == 0) continue;
            std::sort(cancel_iters.begin(),     cancel_iters.end());
            std::sort(cancel_n_decodes.begin(), cancel_n_decodes.end());
            fprintf(stdout,
                "iter[%d] cancelled budget=%d count=%d "
                "cancel_observed_iter_set={", r, b, cancelled_b);
            for (size_t i = 0; i < cancel_iters.size(); i++) {
                fprintf(stdout, "%s%d", i == 0 ? "" : ",",
                        cancel_iters[i]);
            }
            fprintf(stdout, "} n_decoded_at_cancel_set={");
            for (size_t i = 0; i < cancel_n_decodes.size(); i++) {
                fprintf(stdout, "%s%d", i == 0 ? "" : ",",
                        cancel_n_decodes[i]);
            }
            fprintf(stdout, "}\n");
        }

        // ---- Cancel Slice 3 status summary -----------------------------
        // Single-line, grep-friendly summary of the completed/cancelled
        // split. Computed from the request_result snapshots; the
        // top-level cross-check below confirms it matches the
        // engine-side counter.
        {
            int32_t comp_s = 0, canc_s = 0;
            for (const auto & rr : results) {
                if (rr.status == request_status::completed)      comp_s++;
                else if (rr.status == request_status::cancelled) canc_s++;
            }
            fprintf(stdout,
                "iter[%d] status_summary: completed=%d cancelled=%d "
                "total=%d\n", r, comp_s, canc_s, comp_s + canc_s);
        }

        // 33-per-class gate (completed + cancelled together) on
        // uniform-divisible mixes.
        if (args.n_seqs % static_cast<int32_t>(args.decode_budget_mix.size()) == 0) {
            const int32_t per_class =
                args.n_seqs / static_cast<int32_t>(args.decode_budget_mix.size());
            for (int32_t b : uniq_budgets) {
                int32_t cnt = 0;
                for (const auto & rr : results) {
                    if (rr.decode_budget == b) cnt++;
                }
                if (cnt != per_class) {
                    char buf[256];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: budget %d count=%d (expected %d)",
                        r, b, cnt, per_class);
                    fail_with(buf); return 1;
                }
            }
        }

        // Top-level cancelled / completed totals, cross-checked
        // against the engine-side counter.
        int32_t completed_total = 0;
        int32_t cancelled_total = 0;
        for (const auto & rr : results) {
            if (rr.status == request_status::completed)      completed_total++;
            else if (rr.status == request_status::cancelled) cancelled_total++;
        }
        if (cancelled_total !=
            static_cast<int32_t>(args.cancel_plan.size())) {
            char buf[200];
            std::snprintf(buf, sizeof(buf),
                "iter %d: cancelled_total=%d != cancel_plan size %zu",
                r, cancelled_total, args.cancel_plan.size());
            fail_with(buf); return 1;
        }
        if (cancelled_total != er.cancelled_count) {
            char buf[200];
            std::snprintf(buf, sizeof(buf),
                "iter %d: cancelled_total=%d (snapshots) != "
                "engine.cancelled_count=%d",
                r, cancelled_total, er.cancelled_count);
            fail_with(buf); return 1;
        }
        if (completed_total + cancelled_total != args.n_seqs) {
            char buf[200];
            std::snprintf(buf, sizeof(buf),
                "iter %d: completed=%d + cancelled=%d != n_seqs=%d",
                r, completed_total, cancelled_total, args.n_seqs);
            fail_with(buf); return 1;
        }

        // Aggregate wasted-decode-rows-after-cancel computed from
        // snapshots. Per-seq equality is already gated above, so this
        // sum is structurally 0; keeping the metric makes a regression
        // visible if the per-seq gate is ever relaxed.
        int32_t wasted_rows = 0;
        for (const auto & rr : results) {
            if (rr.status != request_status::cancelled) continue;
            const llama_pos expected_pos =
                n_prompt_tokens + rr.n_decoded - 2;
            const llama_pos delta = rr.pos_max_at_clear - expected_pos;
            if (delta > 0) wasted_rows += static_cast<int32_t>(delta);
        }
        if (wasted_rows != 0) {
            char buf[200];
            std::snprintf(buf, sizeof(buf),
                "iter %d: wasted_decode_rows_after_cancel=%d "
                "(expected 0)", r, wasted_rows);
            fail_with(buf); return 1;
        }

        fprintf(stdout,
            "iter[%d] residual_kv: all %d seqs cleared "
            "(pos_min=-1, pos_max=-1)\n", r, args.n_seqs);

        // ---- Descriptive metrics block ---------------------------------
        {
            const engine_metrics & m = er.metrics;

            std::vector<double> rpb;
            rpb.reserve(m.rows_per_batch.size());
            int32_t rpb_max = 0;
            for (int32_t v : m.rows_per_batch) {
                rpb.push_back(static_cast<double>(v));
                if (v > rpb_max) rpb_max = v;
            }
            std::vector<double> asp;
            asp.reserve(m.active_seqs_per_iter.size());
            int32_t asp_max = 0;
            for (int32_t v : m.active_seqs_per_iter) {
                asp.push_back(static_cast<double>(v));
                if (v > asp_max) asp_max = v;
            }
            const double rpb_p50 = percentile(rpb, 0.50);
            const double rpb_p95 = percentile(rpb, 0.95);
            const double asp_p50 = percentile(asp, 0.50);
            const double asp_p95 = percentile(asp, 0.95);

            fprintf(stdout, "iter[%d] metrics:\n", r);
            fprintf(stdout, "  wall_ms                 = %.3f\n",
                    m.wall_ms);
            fprintf(stdout, "  decode_calls            = %d\n",
                    m.decode_calls);
            fprintf(stdout, "  update_iterations       = %d\n",
                    m.update_iterations);
            fprintf(stdout,
                "  rows_per_batch          p50=%.0f p95=%.0f max=%d\n",
                rpb_p50, rpb_p95, rpb_max);
            fprintf(stdout,
                "  active_seqs_per_iter    p50=%.0f p95=%.0f max=%d\n",
                asp_p50, asp_p95, asp_max);

            // Per-budget completed/cancelled counts. The earlier
            // version of this loop printed `completed[budget=B] = <total>`
            // which was misleading once cancellation became real;
            // Cancel Slice 4 fixes the count and emits a paired
            // `cancelled[budget=B]` line for budgets with any
            // cancelled seqs.
            for (int32_t b : uniq_budgets) {
                int32_t comp_cnt = 0;
                int32_t canc_cnt = 0;
                for (const auto & rr : results) {
                    if (rr.decode_budget != b) continue;
                    if (rr.status == request_status::completed)      comp_cnt++;
                    else if (rr.status == request_status::cancelled) canc_cnt++;
                }
                fprintf(stdout,
                    "  completed[budget=%d]    = %d\n", b, comp_cnt);
                if (canc_cnt > 0) {
                    fprintf(stdout,
                        "  cancelled[budget=%d]    = %d\n", b, canc_cnt);
                }
            }
            // ttc_ms is descriptive only (excluded from the
            // determinism contract). Cancel Slice 4 splits it by
            // status so completion-time and cancellation-time
            // distributions are not blurred together.
            for (int32_t b : uniq_budgets) {
                std::vector<double> ttc_ms_comp;
                std::vector<double> ttc_ms_canc;
                ttc_ms_comp.reserve(static_cast<size_t>(33));
                ttc_ms_canc.reserve(static_cast<size_t>(8));
                for (const auto & rr : results) {
                    if (rr.decode_budget != b) continue;
                    const double v =
                        static_cast<double>(rr.ttc_us) / 1000.0;
                    if (rr.status == request_status::completed) {
                        ttc_ms_comp.push_back(v);
                    } else if (rr.status == request_status::cancelled) {
                        ttc_ms_canc.push_back(v);
                    }
                }
                if (!ttc_ms_comp.empty()) {
                    fprintf(stdout,
                        "  ttc_ms_completed[budget=%d]  mean=%.2f p95=%.2f\n",
                        b, mean_of(ttc_ms_comp),
                        percentile(ttc_ms_comp, 0.95));
                }
                if (!ttc_ms_canc.empty()) {
                    fprintf(stdout,
                        "  ttc_ms_cancelled[budget=%d]  mean=%.2f p95=%.2f\n",
                        b, mean_of(ttc_ms_canc),
                        percentile(ttc_ms_canc, 0.95));
                }
            }
            fprintf(stdout, "  futures_created         = %d\n",
                    futures_created);
            fprintf(stdout, "  promises_fulfilled      = %d\n",
                    er.promises_fulfilled);
            fprintf(stdout, "  futures_completed       = %d\n",
                    futures_completed);
            fprintf(stdout, "  engine_task_count       = %d\n",
                    er.engine_task_count);
            // Cancel Slice 4 closeout fields. Numeric/boolean only;
            // the underlying invariants are already gated above and
            // these lines exist for grep-friendly closeout reporting.
            fprintf(stdout, "  completed_count         = %d\n",
                    completed_total);
            fprintf(stdout, "  cancelled_count         = %d\n",
                    cancelled_total);
            fprintf(stdout, "  decode_failures         = %d\n",
                    er.decode_failures);
            fprintf(stdout, "  wasted_decode_rows_after_cancel = %d\n",
                    wasted_rows);
            fprintf(stdout, "  residual_kv_empty       = %s\n",
                    er.residual_kv_ok ? "true" : "false");
        }

        if (r == 0) {
            first_results = results;
        } else {
            // Determinism gate: identical n_decoded / generated_tokens /
            // hash / done_iter / pos_max_at_clear across repeats. ttc_us
            // and timing-derived metrics are not part of the determinism
            // contract.
            if (results.size() != first_results.size()) {
                char buf[128];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: %zu results vs first iter %zu",
                    r, results.size(), first_results.size());
                fail_with(buf); return 1;
            }
            for (size_t i = 0; i < results.size(); i++) {
                const request_result & a0 = first_results[i];
                const request_result & a  = results[i];
                if (a.seq_id            != a0.seq_id
                 || a.n_decoded         != a0.n_decoded
                 || a.generated_tokens  != a0.generated_tokens
                 || a.hash              != a0.hash
                 || a.done_iter         != a0.done_iter
                 || a.pos_max_at_clear  != a0.pos_max_at_clear) {
                    char buf[320];
                    std::snprintf(buf, sizeof(buf),
                        "non-deterministic: iter %d seq %d hash "
                        "0x%016llx differs from iter 0 hash 0x%016llx",
                        r, a.seq_id,
                        static_cast<unsigned long long>(a.hash),
                        static_cast<unsigned long long>(a0.hash));
                    fail_with(buf); return 1;
                }
            }
            fprintf(stdout, "iter[%d] determinism: matches iter 0\n", r);
        }
        fflush(stdout);
    }

    cleanup_llama(ctx, model);
    hpx_runtime::stop();

    emit_pass();
    return 0;
}
