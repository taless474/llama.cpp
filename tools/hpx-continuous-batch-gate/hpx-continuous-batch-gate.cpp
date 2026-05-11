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
#include <deque>
#include <exception>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr const char * STEP = "HPX_CB_ADMIT_STEP7";

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

// ---- Live admission source (Live Admission Slice 1: data model only) ----
// `none` is the default for every result. `cancel_freed` marks an admitted
// request that reused a slot popped from the engine's free_due_to_cancel
// queue. `completion_freed` (Slice 5) marks a slot popped from
// free_due_to_completion. Slice 1 only ever sets `none`; Slice 3
// introduces `cancel_freed`; Slice 5 introduces `completion_freed`.
enum class admission_source : uint8_t {
    none             = 0,
    cancel_freed     = 1,
    completion_freed = 2,
};

const char * admission_source_name(admission_source s) noexcept {
    switch (s) {
        case admission_source::none:             return "none";
        case admission_source::cancel_freed:     return "cancel_freed";
        case admission_source::completion_freed: return "completion_freed";
    }
    return "unknown";
}

// ---- Arrival source (Live Admission Slice 6: async external arrivals) --
// Distinguishes a waiting request that came preloaded at engine
// construction (via --n-waiting, the Slice 2..5 path) from one that
// was submitted at runtime by an external HPX submitter task (via
// engine::submit, the Slice 6 path). Orthogonal to admission_source:
// an external arrival can be admitted via either cancel_freed or
// completion_freed; in the Slice 6 smoke it is cancel_freed.
enum class arrival_source : uint8_t {
    preloaded = 0,
    external  = 1,
};

const char * arrival_source_name(arrival_source s) noexcept {
    switch (s) {
        case arrival_source::preloaded: return "preloaded";
        case arrival_source::external:  return "external";
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

    // ---- live admission data model (Live Admission Slice 1: not
    //      exercised). `request_id` is initialized to `seq_id` by the
    //      engine constructor; no admission rebinds it in Slice 1, so
    //      `request_id == seq_id` is an enforced invariant on every
    //      result. `reused_seq_id` lives only on `request_result`, not
    //      here. -----------------------------------------------------
    int32_t                  request_id                   = -1;
    int32_t                  admitted_at_iter             = -1;
    int32_t                  previous_request_id          = -1;
    admission_source         admission_src                = admission_source::none;

    // ---- arrival data model (Live Admission Slice 6) --------------------
    // Records whether this request entered the engine via the preloaded
    // construction-time waiting queue (--n-waiting) or via an external
    // submitter through engine::submit() at runtime. Default for all
    // initially-bound active seqs is preloaded; admission overwrites
    // this from the waiting_request that was bound. -----------------------
    arrival_source           arrival_src                  = arrival_source::preloaded;

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

    // ---- live admission snapshot (Live Admission Slice 1: defaults
    //      always). All four fields are populated on every completion
    //      and cancellation path; in Slice 1 the values are the
    //      defaults below for every result.
    int32_t                  admitted_at_iter     = -1;
    int32_t                  reused_seq_id        = -1;
    int32_t                  previous_request_id  = -1;
    admission_source         admission_src        = admission_source::none;

    // ---- arrival snapshot (Live Admission Slice 6) ----------------------
    // preloaded for every initially-bound active seq AND for preloaded
    // waiters admitted via the Slice 2..5 path. external only for
    // arrivals that entered through engine::submit(). ---------------------
    arrival_source           arrival_src          = arrival_source::preloaded;
};

// ---- Waiting request (Live Admission Slice 2: queue-only) --------------
// Constructed once by main and passed to the engine as a const-ref handle
// (preloaded path), OR constructed inside drain_external_inbox() from an
// arrival_msg (external path). Slice 6: `src` records which path; the
// admission step uses it to decide promise ownership at bind time.
struct waiting_request {
    int32_t        request_id    = -1;
    int32_t        decode_budget = 0;
    arrival_source src           = arrival_source::preloaded;
};

// ---- Async external arrival (Live Admission Slice 6) -------------------
// arrival_msg crosses from the external submitter HPX task to the
// engine task via engine::submit(), which acquires the inbox spinlock
// and pushes. The promise is created by the submitter; the submitter
// holds the corresponding future. At admission, the engine moves the
// promise into the slot's promise position and fulfills it via the
// existing fulfill_promise() path.
struct arrival_msg {
    int32_t                      request_id    = -1;
    int32_t                      decode_budget = 0;
    hpx::promise<request_result> promise;
    arrival_source               src           = arrival_source::external;
};

// ---- Pre-run release/ack handle (Live Admission Slice 6) ---------------
// Returned by engine::register_external_release_iter(K) before run().
// `release_future` is awaited by the submitter at the start of its
// per-K block. `ack_promise` is set_value()'d by the submitter after
// all K-arrivals have been pushed; the engine suspends on the matching
// future and only resumes once the ack fires (release+ack barrier).
struct external_release_handle {
    hpx::future<void>  release_future;
    hpx::promise<void> ack_promise;
};

// ---- Submitter script entry (Live Admission Slice 6) -------------------
// One per scripted arrival. The submitter walks the script in order;
// arrivals sharing the same release_iter are pushed in script order
// under a single release+ack barrier.
struct scripted_arrival {
    int32_t request_id    = -1;
    int32_t decode_budget = 0;
    int32_t release_iter  = -1;
};

// ---- Engine metrics -----------------------------------------------------
struct engine_metrics {
    double               wall_ms              = 0.0;
    int32_t              decode_calls         = 0;
    int32_t              update_iterations    = 0;
    std::vector<int32_t> rows_per_batch;       // one per llama_decode call
    std::vector<int32_t> active_seqs_per_iter; // one per decode iteration
    // Live Admission Slice 4: queue depth sampled AFTER this iter's
    // admission loop runs. For the smoke shape, this stays at
    // n_waiting for iters 1..16 (cancel pushes free_due_to_cancel
    // entries during iter 16, but admission consumes them at iter 17),
    // then drops to 0 at iter 17 and stays 0 for the remaining decode
    // iters. One sample per decode iter; size == update_iterations.
    std::vector<int32_t> waiting_queue_depth_after_admission_per_iter;
    // Live Admission Slice 4: distinct decode-iter values at which
    // admission fired (one push per iter when the admission loop
    // bound at least one waiting request). Insertion order is
    // monotonically increasing, so it doubles as the iter set.
    std::vector<int32_t> admission_iter_set;
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
    // Live Admission Slice 3: queue visibility + admission counters.
    // queued_count is sampled at engine start; waiting_queue_size_at_engine_end
    // is sampled at engine end. consumed = queued_count - end_size must
    // equal admitted_count. reused_seq_id_set lists the slot ids that
    // served an admitted request, in admission order (deterministic
    // ascending).
    int32_t              queued_count                     = 0;
    int32_t              waiting_queue_size_at_engine_end = 0;
    int32_t              admitted_count                   = 0;
    std::vector<int32_t> reused_seq_id_set;
    // Live Admission Slice 4: structural counter incremented next to
    // the admitted_prefilled trace site (unconditionally, regardless
    // of trace gating). Main asserts admitted_prefill_events ==
    // admitted_count at end of each repeat — catches "exactly one
    // admitted_prefilled per admitted request" without depending on
    // trace being enabled.
    int32_t              admitted_prefill_events          = 0;
    // Live Admission Slice 5: residual size of free_due_to_completion_
    // at engine end. Demand-gated push semantics mean unused freed
    // slots from the first wave remain pooled and KV-empty. For the
    // Slice 5 smoke shape (n_active=90, n_waiting=9, waiting_budget=8,
    // round-robin {8,64,256}) the expected residual is 21 (30 budget-8
    // slots completed at iter 7, 9 consumed by admission at iter 8).
    int32_t              completion_freed_pool_size_at_run_end = 0;

    // Live Admission Slice 6: async external arrival counters.
    // arrival_drained_count is incremented by the engine task in
    // drain_external_inbox() — engine::submit() never mutates result_
    // (Correction 1). external_admitted_count is incremented at the
    // admission step when a waiter with src=external is bound to a
    // freed slot. first_external_drain_iter records the iter of the
    // first arrival_drained event (-1 if none). iter_release_fired_set
    // and submitter_ack_set are pushed by the end-of-iter release+ack
    // barrier; their insertion order is monotonically ascending.
    int32_t              arrival_drained_count       = 0;
    int32_t              external_admitted_count     = 0;
    int32_t              first_external_drain_iter   = -1;
    std::vector<int32_t> iter_release_fired_set;
    std::vector<int32_t> submitter_ack_set;
    engine_metrics       metrics;
};

// ---- Engine class -------------------------------------------------------
class engine {
public:
    engine(llama_context *                       ctx,
           const llama_vocab *                   vocab,
           int32_t                               n_vocab,
           const std::vector<llama_token> &      prompt_tokens,
           std::vector<int32_t>                  budgets,
           int32_t                               batch_capacity,
           const std::vector<int32_t> &          cancel_plan,
           int32_t                               cancel_after,
           int32_t                               n_seq_max,
           const std::vector<waiting_request> &  waiting_queue,
           bool                                  reuse_completed,
           std::set<int32_t>                     release_iter_set,
           int32_t                               max_decode_iters)
        : ctx_(ctx),
          vocab_(vocab),
          n_vocab_(n_vocab),
          prompt_tokens_(prompt_tokens),
          budgets_(std::move(budgets)),
          batch_capacity_(batch_capacity),
          promises_(budgets_.size()),
          seqs_(budgets_.size()),
          n_seq_max_(n_seq_max),
          waiting_queue_(&waiting_queue),
          reuse_completed_(reuse_completed),
          iter_release_set_(std::move(release_iter_set)),
          max_decode_iters_(max_decode_iters)
    {
        // Live Admission Slice 6: pre-allocate release/ack vectors so
        // engine::register_external_release_iter(K) can take a slot
        // before any HPX task is spawned. Vectors are sized to cover
        // the largest K we might fire; non-watched entries are
        // default-constructed and never touched by the engine loop.
        const int32_t release_slots =
            std::max<int32_t>(max_decode_iters_ + 1, 0);
        iter_release_promises_.resize(
            static_cast<size_t>(release_slots));
        submitter_ack_futures_.resize(
            static_cast<size_t>(release_slots));
        for (size_t s = 0; s < budgets_.size(); s++) {
            seqs_[s].seq_id        = static_cast<int32_t>(s);
            // Live Admission Slice 1: request_id is initialized to seq_id
            // and stays equal for the life of this slice (no admission
            // rebinds it). Slice 3 will distinguish them.
            seqs_[s].request_id    = static_cast<int32_t>(s);
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
    // Returns the n_active "original" futures only; admitted-request
    // futures are created inside the engine task and collected via
    // take_admitted_futures() AFTER engine_fut.get().
    std::vector<hpx::future<request_result>> take_futures() {
        std::vector<hpx::future<request_result>> futs;
        futs.reserve(promises_.size());
        for (auto & p : promises_) {
            futs.emplace_back(p.get_future());
        }
        return futs;
    }

    // Live Admission Slice 3: drain the admitted-request futures the
    // engine produced. MUST be called from main only AFTER
    // engine_fut.get() returns — by that point every admitted promise
    // is already fulfilled, so each future is ready and the follow-on
    // hpx::wait_all in main is a no-op. The mutex matches the engine-
    // side push site; in practice both sides are sequenced by
    // engine_fut.get(), so contention is impossible.
    std::vector<hpx::future<request_result>> take_admitted_futures() {
        std::lock_guard<std::mutex> lk(admitted_futures_mtx_);
        std::vector<hpx::future<request_result>> out;
        out.swap(admitted_futures_);
        return out;
    }

    // Live Admission Slice 6: async external arrival inbox push.
    // Called from the external submitter HPX task — never from the
    // engine task. Per Correction 1, this method is intentionally
    // read-only on engine_result: it only acquires the inbox spinlock,
    // moves the message into inbox_, and (optionally) emits a trace
    // event. Counters are bumped by the engine when it drains.
    // Hard rule: this body must not call any llama_* API. The grep
    // gate in main asserts that.
    void submit(arrival_msg msg) {
        const int32_t rid = msg.request_id;
        const int32_t bud = msg.decode_budget;
        {
            std::lock_guard<hpx::spinlock> lk(inbox_mtx_);
            inbox_.push_back(std::move(msg));
        }
        trace::event(
            "request_submitted_external request=%d budget=%d "
            "arrival_source=external",
            rid, bud);
    }

    // Live Admission Slice 6: pre-run registration of a release+ack
    // barrier at decode iter K. Must be called once per watched K
    // BEFORE the engine task is scheduled (so the engine sees a
    // ready ack_future to .get() on at end of iter K). Returns the
    // matching release_future (for the submitter to await) and the
    // ack_promise (for the submitter to set_value() after pushing all
    // K-arrivals). The ack_promise is move-only and ownership transfers
    // out of the engine here.
    external_release_handle register_external_release_iter(int32_t K) {
        if (K < 0 || static_cast<size_t>(K) >= iter_release_promises_.size()) {
            throw std::runtime_error(
                "register_external_release_iter: K out of range");
        }
        if (!iter_release_set_.count(K)) {
            throw std::runtime_error(
                "register_external_release_iter: K not declared in "
                "release_iter_set passed to engine ctor");
        }
        external_release_handle h;
        h.release_future =
            iter_release_promises_[static_cast<size_t>(K)].get_future();
        hpx::promise<void> ack_promise_local;
        submitter_ack_futures_[static_cast<size_t>(K)] =
            ack_promise_local.get_future();
        h.ack_promise = std::move(ack_promise_local);
        return h;
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
        // Live Admission Slice 6: similarly drain any pre-registered
        // release promises that were not fired (engine bailed before
        // reaching iter K). Without this, the submitter task would
        // hang on release_future.get() forever.
        std::set<int32_t> fired(
            result_.iter_release_fired_set.begin(),
            result_.iter_release_fired_set.end());
        for (int32_t K : iter_release_set_) {
            if (fired.count(K)) continue;
            if (K < 0
             || static_cast<size_t>(K)
                    >= iter_release_promises_.size()) {
                continue;
            }
            const std::string msg = result_.error.empty()
                ? "engine ended without firing release_iter"
                : result_.error;
            try {
                iter_release_promises_[static_cast<size_t>(K)]
                    .set_exception(std::make_exception_ptr(
                        std::runtime_error(msg)));
            } catch (...) {
                // already satisfied; ignore.
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
        rr.request_id       = seq.request_id;
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
        // Live Admission Slice 3: copy admission fields from seq_state on
        // every completion AND every cancellation path. reused_seq_id is
        // populated for admitted results only — it is the slot id this
        // request was bound to at the admission boundary, which equals
        // seq_id at fulfillment time. Non-admitted results keep -1.
        rr.admitted_at_iter    = seq.admitted_at_iter;
        rr.previous_request_id = seq.previous_request_id;
        rr.admission_src       = seq.admission_src;
        rr.reused_seq_id       = (seq.admission_src != admission_source::none)
                                 ? seq.seq_id : -1;
        // Live Admission Slice 6: propagate arrival source onto the
        // snapshot. preloaded for original actives and preloaded
        // waiters; external for arrivals that came through engine::submit().
        rr.arrival_src         = seq.arrival_src;

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
        // Live Admission Slice 4: admitted-only completion event
        // mirroring seq_complete. Fires IN ADDITION TO seq_complete so
        // the existing 1:1 completion-trace invariant is preserved.
        if (seq.admission_src != admission_source::none) {
            trace::event(
                "admitted_complete request=%d seq_id=%d budget=%d "
                "done_iter=%d hash=0x%016llx admission_source=%s",
                seq.request_id, seq.seq_id, seq.decode_budget, iter,
                static_cast<unsigned long long>(hash_now),
                admission_source_name(seq.admission_src));
        }

        if (!clear_and_check(seq, iter, mem)) return false;

        trace::event(
            "kv_cleared seq=%d pos_max_at_clear=%d cross_talk_ok=1",
            seq.seq_id, seq.pos_max_at_clear);

        // Live Admission Slice 5: demand-gated push onto the completion-
        // freed pool. Push only when --reuse-completed is on AND the
        // waiting queue still has unadmitted entries. Once waiters are
        // drained, later natural completions stop pooling so the
        // residual size remains a tight invariant for the first-wave
        // proof. Push happens after KV-clear succeeds so the slot is
        // truly KV-empty at queue-entry time.
        if (reuse_completed_ && !waiting_queue_consumable_.empty()) {
            free_due_to_completion_.push_back(seq.seq_id);
        }

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

        // Live Admission Slice 3: enqueue this slot for reuse. The
        // cancellation-observation loop walks seqs_ in ascending seq_id
        // order, so push_back here yields the design's required
        // deterministic ascending eligibility order ({1,2,4,5,7,8} for
        // the smoke shape). Admission is delayed by one iter via the
        // freeze-count snapshot at the top of the next decode iter.
        free_due_to_cancel_.push_back(seq.seq_id);

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

    // Live Admission Slice 6: drain the async-arrival inbox at the top
    // of each decode iter. Engine task only — never call from the
    // submitter side. Swaps the entire inbox out under the spinlock
    // (single critical section, O(1)), then iterates the local deque
    // without the lock: stashes the submitter-created promise in
    // external_promises_ keyed by request_id, pushes a waiting_request
    // tagged src=external onto waiting_queue_consumable_, bumps
    // result_.arrival_drained_count, latches first_external_drain_iter
    // on first drain, and emits the arrival_drained / request_queued
    // trace events.
    void drain_external_inbox(int32_t iter) {
        std::deque<arrival_msg> drained;
        {
            std::lock_guard<hpx::spinlock> lk(inbox_mtx_);
            drained.swap(inbox_);
        }
        while (!drained.empty()) {
            arrival_msg msg = std::move(drained.front());
            drained.pop_front();
            const int32_t rid = msg.request_id;
            const int32_t bud = msg.decode_budget;
            external_promises_.emplace(rid, std::move(msg.promise));
            waiting_request w;
            w.request_id    = rid;
            w.decode_budget = bud;
            w.src           = arrival_source::external;
            waiting_queue_consumable_.push_back(w);
            result_.arrival_drained_count++;
            if (result_.first_external_drain_iter == -1) {
                result_.first_external_drain_iter = iter;
            }
            trace::event(
                "arrival_drained request=%d iter=%d budget=%d",
                rid, iter, bud);
            trace::event(
                "request_queued request=%d budget=%d "
                "arrival_source=external", rid, bud);
        }
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
        result_.queued_count                     = 0;
        result_.waiting_queue_size_at_engine_end = 0;
        result_.admitted_count                   = 0;
        result_.reused_seq_id_set.clear();
        result_.admitted_prefill_events          = 0;
        result_.completion_freed_pool_size_at_run_end = 0;
        result_.arrival_drained_count                 = 0;
        result_.external_admitted_count               = 0;
        result_.first_external_drain_iter             = -1;
        result_.iter_release_fired_set.clear();
        result_.submitter_ack_set.clear();
        result_.metrics = engine_metrics{};

        // Live Admission Slice 3: reset the cancel-freed slot queue and
        // re-seed the consumable waiting queue from the const-ref handle
        // at every repeat. Default mode (--n-waiting=0) leaves
        // free_due_to_cancel_ to accumulate cancel-plan entries that
        // are never consumed; that residue is now legal and must NOT
        // leak across repeats.
        // Live Admission Slice 5: free_due_to_completion_ must also be
        // cleared between repeats so the residual count is per-run.
        free_due_to_cancel_.clear();
        free_due_to_completion_.clear();
        waiting_queue_consumable_.assign(
            waiting_queue_->begin(), waiting_queue_->end());
        {
            std::lock_guard<std::mutex> lk(admitted_futures_mtx_);
            admitted_futures_.clear();
        }

        result_.queued_count =
            static_cast<int32_t>(waiting_queue_consumable_.size());

        auto * mem = llama_get_memory(ctx_);
        llama_memory_clear(mem, /*data=*/false);

        const int32_t n_seqs   = static_cast<int32_t>(budgets_.size());
        const int32_t n_prompt = static_cast<int32_t>(prompt_tokens_.size());
        int32_t       max_budget = 0;
        for (int32_t b : budgets_) {
            if (b > max_budget) max_budget = b;
        }

        // Reset per-seq dynamic state for this run (--repeat re-uses
        // the same engine's seqs_; configured fields stay set). Slice 3:
        // also reset the admission fields and restore decode_budget to
        // the original active value, so a slot that was admitted on a
        // prior repeat starts the next repeat as its original active
        // request again.
        for (size_t s = 0; s < seqs_.size(); s++) {
            seq_state & seq = seqs_[s];
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
            seq.decode_budget        = budgets_[s];
            seq.request_id           = static_cast<int32_t>(s);
            seq.admitted_at_iter     = -1;
            seq.previous_request_id  = -1;
            seq.admission_src        = admission_source::none;
            // Live Admission Slice 6: every initially-bound active slot
            // is preloaded at engine_start. admit_one overwrites this
            // from the bound waiting_request's `src` field.
            seq.arrival_src          = arrival_source::preloaded;
            seq.cancel_observed      = false;
            seq.cancel_observed_iter = -1;
            seq.n_decoded_at_cancel  = -1;
            seq.cancel_requested.store(false, std::memory_order_release);
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
        while (any_active() || !waiting_queue_consumable_.empty()) {
            iter++;

            // Live Admission Slice 6: drain the async-arrival inbox at
            // TOP of this iter, BEFORE the cancel-freed snapshot. Pairs
            // with the release+ack barrier at the END of iter K
            // (release-at-K, drain-at-K+1 semantics): an arrival pushed
            // by the submitter during the iter-K barrier is guaranteed
            // visible here at iter K+1 because the engine task only
            // resumes after the submitter's ack fires.
            drain_external_inbox(iter);

            // Live Admission Slice 3: snapshot the cancel-freed slot
            // count BEFORE this iter's cancellation observation runs.
            // Admission below will only consume up to this many entries
            // — anything pushed by the current iter's cancellation pass
            // is held over for next iter. This produces the design's
            // one-iter delay (cancel observed @ iter K -> admit @ K+1)
            // and yields admitted_at_iter == cancel_after + 1 on the
            // smoke shape (== 17 for cancel_after = 16).
            const size_t admission_eligible_count =
                free_due_to_cancel_.size();

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

            // Live Admission Slice 3: admission boundary. After the
            // cancellation observation pass for this iter and after
            // every cancelled seq's KV clear has succeeded, before any
            // batch build. Consume at most admission_eligible_count
            // entries (snapshot taken above) from free_due_to_cancel_
            // in deterministic ascending order, binding FIFO from
            // waiting_queue_consumable_. The KV-empty assertion is
            // fail-closed (design §3 invariant).
            // Helper lambda: bind a freshly freed seq_id to the FIFO
            // head of waiting_queue_consumable_ and emit the Slice-4
            // admission trace events (now Slice-5: payloads carry an
            // explicit admission_source=<value> key so cancel_freed and
            // completion_freed admissions can be distinguished without
            // adding new event names). Returns true on success; on the
            // KV-not-empty failure path it stamps result_.error and the
            // caller frees the batch + returns.
            auto admit_one = [&](int32_t reuse_seq,
                                 admission_source src,
                                 const char *      src_label) -> bool {
                const llama_pos pmin =
                    llama_memory_seq_pos_min(mem, reuse_seq);
                const llama_pos pmax =
                    llama_memory_seq_pos_max(mem, reuse_seq);
                if (pmin != -1 || pmax != -1) {
                    char buf[200];
                    std::snprintf(buf, sizeof(buf),
                        "admission: reuse seq %d KV not empty before "
                        "binding: pos_min=%d pos_max=%d",
                        reuse_seq, pmin, pmax);
                    result_.error = buf;
                    return false;
                }

                const waiting_request w =
                    waiting_queue_consumable_.front();
                waiting_queue_consumable_.pop_front();

                seq_state & rseq =
                    seqs_[static_cast<size_t>(reuse_seq)];
                const int32_t prior_owner = rseq.request_id;

                trace::event(
                    "seq_reused seq_id=%d previous_owner=%d "
                    "new_owner=%d iter=%d admission_source=%s "
                    "arrival_source=%s",
                    reuse_seq, prior_owner, w.request_id, iter,
                    src_label, arrival_source_name(w.src));
                trace::event(
                    "request_admitted_live request=%d reused_seq_id=%d "
                    "iter=%d admission_source=%s arrival_source=%s",
                    w.request_id, reuse_seq, iter, src_label,
                    arrival_source_name(w.src));

                rseq.previous_request_id    = prior_owner;
                rseq.request_id             = w.request_id;
                rseq.admitted_at_iter       = iter;
                rseq.admission_src          = src;
                rseq.arrival_src            = w.src;
                rseq.decode_budget          = w.decode_budget;
                rseq.n_decoded              = 0;
                rseq.pos_next               = 0;
                rseq.i_batch                = -1;
                rseq.last_token             = 0;
                rseq.done                   = false;
                rseq.done_iter              = -1;
                rseq.pos_max_at_clear       = -1;
                rseq.kv_cleared             = false;
                rseq.promise_fulfilled      = false;
                rseq.hash_state             = k_token_hash_init;
                rseq.generated_tokens.clear();
                rseq.cancel_observed             = false;
                rseq.cancel_observed_iter        = -1;
                rseq.n_decoded_at_cancel         = -1;
                rseq.cancel_after_decoded_tokens = -1;
                rseq.cancel_requested.store(
                    false, std::memory_order_release);

                // Promise rebinding splits by arrival source:
                //   preloaded waiter -> create a fresh engine-owned
                //     promise and push its future onto admitted_futures_
                //     for main to drain (Slice 2..5 path, unchanged).
                //   external arrival -> move the submitter-created
                //     promise out of external_promises_ into the slot;
                //     the submitter already holds the future, so do NOT
                //     push to admitted_futures_.
                if (w.src == arrival_source::external) {
                    auto it = external_promises_.find(w.request_id);
                    if (it == external_promises_.end()) {
                        char buf[200];
                        std::snprintf(buf, sizeof(buf),
                            "admission: external arrival req=%d missing "
                            "promise in external_promises_ at bind time",
                            w.request_id);
                        result_.error = buf;
                        return false;
                    }
                    promises_[static_cast<size_t>(reuse_seq)] =
                        std::move(it->second);
                    external_promises_.erase(it);
                    result_.external_admitted_count++;
                } else {
                    promises_[static_cast<size_t>(reuse_seq)] =
                        hpx::promise<request_result>{};
                    {
                        std::lock_guard<std::mutex> lk(
                            admitted_futures_mtx_);
                        admitted_futures_.emplace_back(
                            promises_[static_cast<size_t>(reuse_seq)]
                                .get_future());
                    }
                }
                result_.admitted_count++;
                result_.reused_seq_id_set.push_back(reuse_seq);
                return true;
            };

            int32_t admitted_this_iter = 0;
            // Source priority: drain free_due_to_cancel_ first so the
            // Slice-3 cancel-freed mapping is preserved when both
            // sources are present (out-of-scope for Slice 5 itself,
            // which uses only completion_freed).
            for (size_t i = 0; i < admission_eligible_count; i++) {
                if (free_due_to_cancel_.empty()) break;
                if (waiting_queue_consumable_.empty()) break;
                const int32_t reuse_seq = free_due_to_cancel_.front();
                free_due_to_cancel_.pop_front();

                if (!admit_one(reuse_seq, admission_source::cancel_freed,
                               "cancel_freed")) {
                    llama_batch_free(batch);
                    result_.metrics.update_iterations = iter;
                    finalize_wall_ms();
                    return;
                }
                admitted_this_iter++;
            }

            // Live Admission Slice 5: drain free_due_to_completion_
            // AFTER the cancel queue. Entries here were pushed at the
            // end of an earlier iter's decode (or post-prefill argmax),
            // so no snapshot is required — the queue is settled at the
            // top of this iter. Gated by --reuse-completed; off by
            // default to preserve Slice-4 behavior.
            if (reuse_completed_) {
                while (!free_due_to_completion_.empty()
                    && !waiting_queue_consumable_.empty()) {
                    const int32_t reuse_seq =
                        free_due_to_completion_.front();
                    free_due_to_completion_.pop_front();

                    if (!admit_one(reuse_seq,
                                   admission_source::completion_freed,
                                   "completion_freed")) {
                        llama_batch_free(batch);
                        result_.metrics.update_iterations = iter;
                        finalize_wall_ms();
                        return;
                    }
                    admitted_this_iter++;
                }
            }

            // Live Admission Slice 4: record this iter in
            // admission_iter_set (one push per iter that admitted at
            // least one waiting request). Then sample the queue depth
            // AFTER admission, so iter 17 in the smoke records 0
            // (admission consumed all 6 waiters this iter).
            if (admitted_this_iter > 0) {
                result_.metrics.admission_iter_set.push_back(iter);
            }
            result_.metrics.waiting_queue_depth_after_admission_per_iter
                .push_back(static_cast<int32_t>(
                    waiting_queue_consumable_.size()));

            // If every still-active seq was just cancelled AND no
            // admission happened, end the loop cleanly without trying
            // to decode an empty batch. Admitted seqs re-trigger
            // any_active() = true via done=false above.
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
                // Live Admission Slice 3: a freshly admitted seq has
                // admitted_at_iter == iter && n_decoded == 0 in the
                // same iter as its admission. Emit prompt/prefill rows
                // (logits=true on the last only) instead of a single
                // decode row. The post-decode argmax then produces its
                // first token via the same path as a normal
                // post-prefill argmax, after which n_decoded == 1.
                if (seq.admitted_at_iter == iter && seq.n_decoded == 0) {
                    for (int32_t p = 0; p < n_prompt; p++) {
                        const bool last = (p == n_prompt - 1);
                        common_batch_add(batch, prompt_tokens_[p],
                                         /*pos=*/p,
                                         /*seq_ids=*/{seq.seq_id},
                                         /*logits=*/last);
                        if (last) {
                            seq.i_batch = batch.n_tokens - 1;
                        }
                    }
                    seq.pos_next = n_prompt;
                    active_idx.push_back(s);
                    // No decode_row trace for prefill rows; Slice 4
                    // owns admitted-prefill trace events.
                } else {
                    common_batch_add(batch, seq.last_token,
                                     /*pos=*/seq.pos_next,
                                     /*seq_ids=*/{seq.seq_id},
                                     /*logits=*/true);
                    seq.i_batch = batch.n_tokens - 1;
                    seq.pos_next++;
                    active_idx.push_back(s);
                    trace::event("decode_row iter=%d seq=%d pos=%d",
                                 iter, seq.seq_id, seq.pos_next - 1);
                    // Live Admission Slice 4: fine-grained event for
                    // admitted seqs only. Fires IN ADDITION TO the
                    // generic decode_row above so the 1:1
                    // decode_row-per-row invariant is preserved.
                    if (seq.admission_src != admission_source::none) {
                        trace::event(
                            "admitted_decode_row request=%d seq_id=%d "
                            "iter=%d pos=%d admission_source=%s",
                            seq.request_id, seq.seq_id, iter,
                            seq.pos_next - 1,
                            admission_source_name(seq.admission_src));
                    }
                }
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
                // Live Admission Slice 4: capture the admitted-prefill-
                // argmax predicate BEFORE n_decoded changes, so the
                // event fires exactly once per admitted request (at the
                // post-decode argmax of its admission iter, where it
                // saw its prefill rows). Mirrors the existing
                // seq_prefilled placement (before EOG check).
                const bool is_admitted_prefill_argmax =
                    seq.admission_src != admission_source::none
                 && seq.admitted_at_iter == iter
                 && seq.n_decoded == 0;
                if (is_admitted_prefill_argmax) {
                    trace::event(
                        "admitted_prefilled request=%d seq_id=%d "
                        "first_token=%d admission_source=%s",
                        seq.request_id, seq.seq_id,
                        static_cast<int>(next_id),
                        admission_source_name(seq.admission_src));
                    result_.admitted_prefill_events++;
                }
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

            // Live Admission Slice 6: release + ack barrier at END of
            // iter K. If K is in iter_release_set_, set the release
            // promise (the submitter is waiting on its future), then
            // suspend the engine task on the matching ack future. The
            // submitter pushes all K-arrivals via engine::submit() and
            // calls ack_promise.set_value() to resume us. With the ack
            // barrier, scheduling order is deterministic even on a
            // single HPX worker: engine only proceeds past this point
            // after every K-arrival is in inbox_. The drain at the top
            // of iter K+1 then exposes them to admission.
            if (iter_release_set_.count(iter)) {
                trace::event("iter_release_fired iter=%d", iter);
                iter_release_promises_[static_cast<size_t>(iter)]
                    .set_value();
                result_.iter_release_fired_set.push_back(iter);
                try {
                    submitter_ack_futures_[
                        static_cast<size_t>(iter)].get();
                } catch (const std::exception & e) {
                    char buf[256];
                    std::snprintf(buf, sizeof(buf),
                        "submitter ack future for iter %d threw: %s",
                        iter, e.what());
                    if (result_.error.empty()) result_.error = buf;
                    llama_batch_free(batch);
                    result_.metrics.update_iterations = iter;
                    finalize_wall_ms();
                    return;
                }
                result_.submitter_ack_set.push_back(iter);
                trace::event("submitter_ack_observed iter=%d", iter);
            }
        }

        result_.metrics.update_iterations = iter;
        llama_batch_free(batch);

        // Live Admission Slice 3: residue in free_due_to_cancel_ at
        // end of run is allowed (default mode with no waiting requests
        // leaves cancel_plan.size() entries here). The earlier Slice 1
        // "must remain empty" guard is retired.

        // Live Admission Slice 3: queue size at engine end. consumed
        // entries == admitted_count is the engine-side cross-check
        // for "every consumed waiting request became an admission".
        // Live Admission Slice 6: queued_count snapshots the PRELOADED
        // waiter population at engine_start (drain pushes external
        // arrivals onto waiting_queue_consumable_ *after* the snapshot),
        // so the cross-check holds against the preloaded-admission
        // count: admitted_count - external_admitted_count.
        result_.waiting_queue_size_at_engine_end =
            static_cast<int32_t>(waiting_queue_consumable_.size());
        const int32_t consumed =
            result_.queued_count -
            result_.waiting_queue_size_at_engine_end;
        const int32_t preloaded_admissions =
            result_.admitted_count - result_.external_admitted_count;
        if (consumed != preloaded_admissions) {
            char buf[200];
            std::snprintf(buf, sizeof(buf),
                "consumed waiting=%d != preloaded_admissions=%d "
                "(admitted=%d external_admitted=%d queued=%d end_size=%d)",
                consumed, preloaded_admissions,
                result_.admitted_count, result_.external_admitted_count,
                result_.queued_count,
                result_.waiting_queue_size_at_engine_end);
            if (result_.error.empty()) result_.error = buf;
            finalize_wall_ms();
            return;
        }

        // Live Admission Slice 5: snapshot the residual completion pool
        // BEFORE the residual-KV-empty sweep so the metric reflects
        // unconsumed naturally-freed slots at run end. With the demand
        // gate ("push only while waiting queue is non-empty") this stays
        // at (first_wave_completions - admitted_count) — for the smoke
        // shape that is 30 - 9 = 21 and does not grow when later natural
        // completions occur with an empty waiting queue.
        result_.completion_freed_pool_size_at_run_end =
            static_cast<int32_t>(free_due_to_completion_.size());

        // Live Admission Slice 6: inbox and external_promises_ MUST be
        // empty at engine end. Non-empty inbox means an arrival was
        // pushed after the engine's last drain (a barrier-discipline
        // bug). Non-empty external_promises_ means an arrival was
        // drained but never admitted, which would leak the submitter's
        // future. Fail closed before the residual-KV sweep so the
        // diagnostic is unambiguous.
        {
            size_t inbox_residual = 0;
            {
                std::lock_guard<hpx::spinlock> lk(inbox_mtx_);
                inbox_residual = inbox_.size();
            }
            if (inbox_residual != 0) {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                    "inbox has %zu residual arrivals at run end",
                    inbox_residual);
                if (result_.error.empty()) result_.error = buf;
                finalize_wall_ms();
                return;
            }
        }
        if (!external_promises_.empty()) {
            char buf[200];
            std::snprintf(buf, sizeof(buf),
                "external_promises_ has %zu residual entries at run end",
                external_promises_.size());
            if (result_.error.empty()) result_.error = buf;
            finalize_wall_ms();
            return;
        }

        // ---- Residual-KV-empty check (engine-side; main never touches
        //      llama_memory_seq_*). Sweeps n_seq_max_ rather than the
        //      bound population, so the not-yet-bound slots
        //      [n_active, n_seq_max) are also asserted empty. ---------
        bool all_clear = true;
        for (int32_t s = 0; s < n_seq_max_; s++) {
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
    // Live Admission Slice 3: cancel-freed-slot reuse queue. cancel_and_fulfill
    // appends to this after its KV clear succeeds. At the top of each decode
    // iter, the admission loop pops up to the snapshot count captured BEFORE
    // this iter's cancel-observation pass — that one-iter delay is what
    // produces admitted_at_iter == cancel_after + 1 for the smoke shape.
    std::deque<int32_t>              free_due_to_cancel_;
    // Live Admission Slice 2: capacity of the underlying llama_context
    // (== args.n_seqs). Used by the residual-KV-empty sweep so it
    // covers the not-yet-bound slots [n_active, n_seq_max).
    int32_t                          n_seq_max_      = 0;
    // Live Admission Slice 3: read-only handle to the construction-time
    // waiting queue. run_body() seeds waiting_queue_consumable_ from it
    // at each repeat. Engine consumes from the working copy only; the
    // const-ref handle is never mutated.
    const std::vector<waiting_request> * waiting_queue_ = nullptr;
    std::deque<waiting_request>      waiting_queue_consumable_;
    // Live Admission Slice 5: completion-freed-slot reuse queue. The
    // completion path (finalize_and_fulfill) pushes here AFTER its KV
    // clear succeeds, gated by --reuse-completed AND non-empty
    // waiting_queue_consumable_ (demand-gate). Admission consumes here
    // AFTER draining free_due_to_cancel_ so cancel-freed remains the
    // primary source when both are present.
    bool                             reuse_completed_ = false;
    std::deque<int32_t>              free_due_to_completion_;
    // Live Admission Slice 3: per-admission promise/future pairs created
    // inside the engine task. Main drains this after engine_fut.get()
    // returns; by that point every entry's promise is fulfilled, so a
    // follow-on wait_all is a no-op (design §11.3). The mutex is
    // defensive — the engine task is single-threaded and main reads
    // strictly after engine_fut.get() — but it makes the ordering
    // invariant explicit.
    std::vector<hpx::future<request_result>> admitted_futures_;
    std::mutex                       admitted_futures_mtx_;
    // Live Admission Slice 6: async external arrival inbox + release/ack
    // barrier state. inbox_ is a passive engine-owned deque guarded by
    // an HPX-aware spinlock (no new std::mutex per Correction 1/2 rules).
    // Submitter pushes via engine::submit(); engine task drains under
    // the spinlock at the top of each decode iter. external_promises_
    // stashes the submitter-created promise keyed by request_id between
    // drain and admission; the admission step moves it into the slot's
    // promise position. iter_release_promises_/submitter_ack_futures_
    // are sized at ctor time and indexed by decode iter K.
    hpx::spinlock                                   inbox_mtx_;
    std::deque<arrival_msg>                                      inbox_;
    std::unordered_map<int32_t, hpx::promise<request_result>>    external_promises_;
    std::vector<hpx::promise<void>>                              iter_release_promises_;
    std::vector<hpx::future<void>>                               submitter_ack_futures_;
    std::set<int32_t>                                            iter_release_set_;
    int32_t                                                      max_decode_iters_ = 0;
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

    // Live Admission Slice 2: queue-only data model.
    //   --n-active        size of the bound active set; resolved post-
    //                     parse to args.n_seqs when 0 (= "use n_seqs").
    //   --n-waiting       number of waiting requests queued at
    //                     construction time; engine never consumes.
    //   --waiting-budget  uniform decode_budget for every waiting
    //                     request (only used by Slice 3+).
    int32_t              n_active        = 0;
    int32_t              n_waiting       = 0;
    int32_t              waiting_budget  = 64;

    // Live Admission Slice 5: enable completion-freed-slot admission.
    // OFF preserves Slice 4 semantics (engine never touches the
    // completion pool). ON pushes naturally completed slots onto
    // free_due_to_completion_ under a demand-gate (push only while the
    // waiting queue still has unadmitted entries) and the admission
    // loop drains the cancel queue first, then the completion queue.
    bool                 reuse_completed = false;

    // Live Admission Slice 6: async external arrivals.
    //   --n-external-arrivals    number of scripted arrivals the
    //                            submitter HPX task will push at the
    //                            release barrier. 0 = inactive (no
    //                            submitter spawned, no release/ack
    //                            barrier installed; the run is byte-
    //                            equivalent to Slice 3..5 outputs).
    //   --external-arrival-budget uniform decode budget for every
    //                            external arrival.
    //   --external-release-iter  decode iter K at which the engine
    //                            fires the release promise; the
    //                            submitter pushes its K-block then
    //                            acks. Drain happens at top of iter
    //                            K+1, admission at iter K+1+1 (cancel-
    //                            freed path) for the smoke shape.
    int32_t              n_external_arrivals     = 0;
    int32_t              external_arrival_budget = 64;
    int32_t              external_release_iter   = 0;
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
        "                            (decoded-token threshold for the plan)\n"
        "  --n-active <int>          default: n_seqs (when 0)\n"
        "                            size of the bound active set; must\n"
        "                            satisfy 1 <= n_active <= n_seqs and\n"
        "                            n_active + n_waiting <= n_seqs.\n"
        "  --n-waiting <int>         default: 0\n"
        "                            number of waiting requests queued at\n"
        "                            construction time. Slice 2: queue-\n"
        "                            only, the engine never consumes.\n"
        "  --waiting-budget <int>    default: 64\n"
        "                            uniform decode budget for every\n"
        "                            waiting request.\n"
        "  --reuse-completed         default: OFF\n"
        "                            Slice 5: enable completion-freed-\n"
        "                            slot admission. Naturally completed\n"
        "                            slots are pushed to a demand-gated\n"
        "                            pool (only while the waiting queue\n"
        "                            has unadmitted entries) and the\n"
        "                            admission loop drains the cancel\n"
        "                            queue first, then the completion\n"
        "                            queue. Off preserves Slice 4\n"
        "                            semantics (label may still advance).\n"
        "  --n-external-arrivals <int>      default: 0\n"
        "                            Slice 6: number of async external\n"
        "                            arrivals the scripted submitter HPX\n"
        "                            task will push under the release+ack\n"
        "                            barrier. 0 disables the path entirely\n"
        "                            (no submitter spawned).\n"
        "  --external-arrival-budget <int>  default: 64\n"
        "                            Slice 6: uniform decode budget for\n"
        "                            every external arrival.\n"
        "  --external-release-iter <int>    default: 0\n"
        "                            Slice 6: decode iter K at which the\n"
        "                            engine fires the release promise.\n"
        "                            Submitter pushes at K, drain occurs\n"
        "                            at top of K+1.\n",
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
        } else if (a == "--n-active") {
            const char * v = need("--n-active");
            if (!v || !parse_int(v, args.n_active)) return false;
            if (args.n_active < 0) {
                fprintf(stderr, "error: --n-active must be >= 0\n");
                return false;
            }
        } else if (a == "--n-waiting") {
            const char * v = need("--n-waiting");
            if (!v || !parse_int(v, args.n_waiting)) return false;
            if (args.n_waiting < 0) {
                fprintf(stderr, "error: --n-waiting must be >= 0\n");
                return false;
            }
        } else if (a == "--waiting-budget") {
            const char * v = need("--waiting-budget");
            if (!v || !parse_int(v, args.waiting_budget)) return false;
            if (args.waiting_budget < 1) {
                fprintf(stderr,
                    "error: --waiting-budget must be >= 1\n");
                return false;
            }
        } else if (a == "--reuse-completed") {
            // Slice 5 boolean flag, no value.
            args.reuse_completed = true;
        } else if (a == "--n-external-arrivals") {
            const char * v = need("--n-external-arrivals");
            if (!v || !parse_int(v, args.n_external_arrivals)) return false;
            if (args.n_external_arrivals < 0) {
                fprintf(stderr,
                    "error: --n-external-arrivals must be >= 0\n");
                return false;
            }
        } else if (a == "--external-arrival-budget") {
            const char * v = need("--external-arrival-budget");
            if (!v || !parse_int(v, args.external_arrival_budget)) return false;
            if (args.external_arrival_budget < 1) {
                fprintf(stderr,
                    "error: --external-arrival-budget must be >= 1\n");
                return false;
            }
        } else if (a == "--external-release-iter") {
            const char * v = need("--external-release-iter");
            if (!v || !parse_int(v, args.external_release_iter)) return false;
            if (args.external_release_iter < 0) {
                fprintf(stderr,
                    "error: --external-release-iter must be >= 0\n");
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
    // Live Admission Slice 2: resolve and validate the active/waiting
    // partition. Default mode (--n-active 0) inherits n_seqs so the
    // run is byte-identical to Slice 1 except for the relabel.
    if (args.n_active == 0) {
        args.n_active = args.n_seqs;
    }
    if (args.n_active < 1 || args.n_active > args.n_seqs) {
        fprintf(stderr,
            "error: --n-active=%d must satisfy 1 <= n_active <= "
            "n_seqs=%d\n", args.n_active, args.n_seqs);
        return false;
    }
    if (args.n_active + args.n_waiting > args.n_seqs) {
        fprintf(stderr,
            "error: n_active=%d + n_waiting=%d > n_seqs=%d\n",
            args.n_active, args.n_waiting, args.n_seqs);
        return false;
    }
    // Live Admission Slice 6: external arrivals occupy ids
    // [n_active + n_waiting, n_active + n_waiting + n_external_arrivals);
    // each will be bound to a freed slot already counted in n_seqs, so
    // we only require enough id headroom (request_id space). The seq_id
    // they end up on comes from the freed-slot pool already populated
    // by cancellation, not from new context slots.
    if (args.n_active + args.n_waiting + args.n_external_arrivals
        > args.n_seqs) {
        fprintf(stderr,
            "error: n_active=%d + n_waiting=%d + "
            "n_external_arrivals=%d > n_seqs=%d\n",
            args.n_active, args.n_waiting,
            args.n_external_arrivals, args.n_seqs);
        return false;
    }
    return true;
}

// ---- Slice 6 scripted submitter (HPX-native, llama-free) ---------------
// One submitter_release_block per distinct release_iter K. Per-arrival
// promises are pre-created in main and moved in here by value; main keeps
// the matching futures so it can wait_all on them after engine_fut.get().
// Arrivals are pushed to engine::submit() in scripted (intra-block) order.
struct submitter_release_block {
    int32_t                                   release_iter = -1;
    external_release_handle                   handle;
    std::vector<scripted_arrival>             arrivals;
    std::vector<hpx::promise<request_result>> promises;
};

// Helper invoked as exactly one HPX task per repeat. HARD RULE: this
// body must not call any llama_* API. It only:
//   - awaits release_future for each block (in ascending release_iter
//     order),
//   - constructs arrival_msg messages and calls engine::submit(),
//   - sets ack_promise after pushing the K-block.
// The corresponding futures live in main; the submitter never touches
// them. No std::thread / condition_variable / sleep / new std::mutex
// is used here — the only synchronisation primitives are the
// hpx::future/hpx::promise pairs and the engine's internal spinlock.
void run_scripted_submitter(
    engine & eng,
    std::vector<submitter_release_block> blocks)
{
    std::sort(blocks.begin(), blocks.end(),
              [](const submitter_release_block & a,
                 const submitter_release_block & b) {
                  return a.release_iter < b.release_iter;
              });
    for (auto & blk : blocks) {
        // Suspend until the engine fires the release promise at end of
        // iter K. If the engine errored before the barrier, mark every
        // arrival's promise broken via set_exception so main's wait_all
        // does not deadlock.
        try {
            blk.handle.release_future.get();
        } catch (const std::exception & e) {
            const std::string what = e.what();
            for (auto & p : blk.promises) {
                try {
                    p.set_exception(std::make_exception_ptr(
                        std::runtime_error(
                            "submitter: release_future threw: " + what)));
                } catch (...) {
                    // already satisfied
                }
            }
            // Best-effort ack so the engine does not hang on the
            // ack future if it managed to set release before failing.
            try { blk.handle.ack_promise.set_value(); } catch (...) {}
            throw;
        }

        for (size_t i = 0; i < blk.arrivals.size(); i++) {
            arrival_msg msg;
            msg.request_id    = blk.arrivals[i].request_id;
            msg.decode_budget = blk.arrivals[i].decode_budget;
            msg.promise       = std::move(blk.promises[i]);
            msg.src           = arrival_source::external;
            eng.submit(std::move(msg));
        }

        try {
            blk.handle.ack_promise.set_value();
        } catch (...) {
            // already satisfied — engine's get() will simply observe
            // ready; ignore.
        }
    }
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

    // Live Admission Slice 2: budgets are per-active-seq only; the
    // bound population is n_active. Slots [n_active, n_seqs) are
    // not-yet-bound and get no budget (Slice 3 will assign on
    // admission).
    std::vector<int32_t> budgets(static_cast<size_t>(args.n_active));
    {
        const size_t m = args.decode_budget_mix.size();
        for (int32_t i = 0; i < args.n_active; i++) {
            budgets[static_cast<size_t>(i)] =
                args.decode_budget_mix[static_cast<size_t>(i) % m];
        }
    }
    const int32_t max_budget =
        budgets.empty()
            ? 0
            : *std::max_element(budgets.begin(), budgets.end());

    // Live Admission Slice 3: build the waiting queue once at
    // construction time. Each waiting request_id starts at n_active
    // and runs through n_active + n_waiting - 1; uniform decode
    // budget per --waiting-budget. Engine takes a const-ref handle
    // and consumes the FIFO head at admission boundaries.
    //
    // Live Admission Slice 4: emit one request_queued trace per
    // waiting request, so the queued population is visible in trace
    // output BEFORE the engine starts. trace::event is gated and is
    // silent when LLAMA_HPX_CB_TRACE is unset.
    std::vector<waiting_request> waiting_queue;
    waiting_queue.reserve(static_cast<size_t>(args.n_waiting));
    for (int32_t i = 0; i < args.n_waiting; i++) {
        waiting_request w;
        w.request_id    = args.n_active + i;
        w.decode_budget = args.waiting_budget;
        w.src           = arrival_source::preloaded;
        waiting_queue.push_back(w);
        // Live Admission Slice 6: tag the preloaded path explicitly so
        // grep on `arrival_source=` exhaustively partitions queued
        // requests across preloaded and external sources.
        trace::event("request_queued request=%d budget=%d queue_pos=%d "
                     "arrival_source=preloaded",
                     args.n_active + i, args.waiting_budget, i);
    }

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
    fprintf(stdout, "n_active:            %d\n", args.n_active);
    fprintf(stdout, "n_waiting:           %d\n", args.n_waiting);
    fprintf(stdout, "waiting_budget:      %d\n", args.waiting_budget);
    fprintf(stdout, "n_external_arrivals: %d\n",
            args.n_external_arrivals);
    fprintf(stdout, "external_arrival_budget: %d\n",
            args.external_arrival_budget);
    fprintf(stdout, "external_release_iter:   %d\n",
            args.external_release_iter);
    fprintf(stdout, "decode_budget_mix:   [");
    for (size_t i = 0; i < args.decode_budget_mix.size(); i++) {
        fprintf(stdout, "%s%d", i == 0 ? "" : ", ",
                args.decode_budget_mix[i]);
    }
    fprintf(stdout, "]\n");
    fprintf(stdout, "hpx_os_threads:      %d\n", args.hpx_os_threads);
    fprintf(stdout, "repeat:              %d\n", args.repeat);
    fprintf(stdout, "reuse_completed:     %d\n",
            args.reuse_completed ? 1 : 0);
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
            if (cs < 0 || cs >= args.n_active) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "cancel_plan seq_id=%d outside [0, n_active=%d) "
                    "(cancellation must target a bound active seq)",
                    cs, args.n_active);
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
        // Live Admission Slice 6: derive the release barrier set and
        // the max decode iter the engine must size release/ack slots
        // for. With a single release_iter K (the smoke shape), the set
        // is {K} and max_decode_iters_for_ctor = K. With 0 external
        // arrivals, the set is empty and the path is fully inert.
        std::set<int32_t> release_iter_set;
        if (args.n_external_arrivals > 0) {
            release_iter_set.insert(args.external_release_iter);
        }
        int32_t max_decode_iters_for_ctor = 0;
        for (int32_t k : release_iter_set) {
            if (k > max_decode_iters_for_ctor) {
                max_decode_iters_for_ctor = k;
            }
        }

        // One engine instance per iteration (own promises/futures set).
        engine eng(ctx, vocab, n_vocab, prompt_tokens, budgets,
                   batch_capacity, args.cancel_plan, args.cancel_after,
                   /*n_seq_max=*/args.n_seqs,
                   /*waiting_queue=*/waiting_queue,
                   /*reuse_completed=*/args.reuse_completed,
                   /*release_iter_set=*/release_iter_set,
                   /*max_decode_iters=*/max_decode_iters_for_ctor);

        // Live Admission Slice 6: build scripted arrivals + per-arrival
        // promise/future pairs + release blocks. Per-arrival request_ids
        // start at n_active + n_waiting so they sit immediately above
        // any preloaded waiter. For the smoke shape all share a single
        // release_iter so we end up with exactly one block of 6.
        std::vector<scripted_arrival> scripted_arrivals;
        std::vector<hpx::future<request_result>> external_futs;
        std::vector<submitter_release_block> blocks;
        if (args.n_external_arrivals > 0) {
            scripted_arrivals.reserve(
                static_cast<size_t>(args.n_external_arrivals));
            for (int32_t i = 0; i < args.n_external_arrivals; i++) {
                scripted_arrival sa;
                sa.request_id    = args.n_active + args.n_waiting + i;
                sa.decode_budget = args.external_arrival_budget;
                sa.release_iter  = args.external_release_iter;
                scripted_arrivals.push_back(sa);
            }
            std::vector<hpx::promise<request_result>>
                per_arrival_promises(scripted_arrivals.size());
            external_futs.reserve(scripted_arrivals.size());
            for (auto & p : per_arrival_promises) {
                external_futs.emplace_back(p.get_future());
            }
            // Group by release_iter (set is sorted; single entry in the
            // smoke shape). For each K, register a release handle with
            // the engine and pack matching script entries + promises.
            for (int32_t K : release_iter_set) {
                submitter_release_block blk;
                blk.release_iter = K;
                try {
                    blk.handle = eng.register_external_release_iter(K);
                } catch (const std::exception & e) {
                    std::string msg =
                        std::string("register_external_release_iter(")
                        + std::to_string(K) + ") threw: " + e.what();
                    fail_with(msg); return 1;
                }
                for (size_t i = 0; i < scripted_arrivals.size(); i++) {
                    if (scripted_arrivals[i].release_iter != K) continue;
                    blk.arrivals.push_back(scripted_arrivals[i]);
                    blk.promises.push_back(
                        std::move(per_arrival_promises[i]));
                }
                blocks.push_back(std::move(blk));
            }
        }

        std::vector<hpx::future<request_result>> futs = eng.take_futures();
        const int32_t initial_futures =
            static_cast<int32_t>(futs.size());

        // Live Admission Slice 6: spawn the scripted submitter as
        // exactly one HPX task BEFORE the engine task starts, so the
        // release barrier is already being awaited when the engine
        // reaches end of iter K. No-op when blocks is empty.
        hpx::future<void> submitter_fut;
        if (!blocks.empty()) {
            submitter_fut = hpx::async(
                [&eng, blks = std::move(blocks)]() mutable {
                    run_scripted_submitter(eng, std::move(blks));
                });
        }

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

        // Live Admission Slice 6: join the submitter task. By this
        // point the engine has fired+observed every release/ack and
        // the submitter has set every ack — it should already be done.
        if (submitter_fut.valid()) {
            try {
                submitter_fut.get();
            } catch (const std::exception & e) {
                std::string msg =
                    std::string("submitter task threw: ") + e.what();
                fail_with(msg); return 1;
            }
        }

        const engine_result & er = eng.result();

        // Live Admission Slice 3: drain admitted-request futures and
        // append to the wait list. Engine has already fulfilled every
        // admitted promise before returning, so each appended future
        // is ready and the follow-on wait_all is a no-op.
        {
            auto admitted = eng.take_admitted_futures();
            futs.reserve(futs.size() + admitted.size());
            for (auto & f : admitted) {
                futs.push_back(std::move(f));
            }
            if (!admitted.empty()) {
                hpx::wait_all(futs);
            }
        }
        // Live Admission Slice 6: append external arrival futures. The
        // engine fulfills them on the same admission/decode path used
        // by preloaded waiters; main holds the futures directly here
        // (no admitted_futures_ push for external arrivals).
        if (!external_futs.empty()) {
            futs.reserve(futs.size() + external_futs.size());
            for (auto & f : external_futs) {
                futs.push_back(std::move(f));
            }
            hpx::wait_all(futs);
        }
        const int32_t futures_created = static_cast<int32_t>(futs.size());
        const int32_t expected_total  = args.n_active + er.admitted_count;

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

        // Slice 3 gates: counts include both initial actives and any
        // admitted-completed requests.
        if (initial_futures != args.n_active) {
            char buf[128];
            std::snprintf(buf, sizeof(buf),
                "iter %d: initial_futures=%d (expected n_active=%d)",
                r, initial_futures, args.n_active);
            fail_with(buf); return 1;
        }
        if (futures_created != expected_total) {
            char buf[128];
            std::snprintf(buf, sizeof(buf),
                "iter %d: futures_created=%d (expected n_active+"
                "admitted_count=%d+%d=%d)",
                r, futures_created, args.n_active,
                er.admitted_count, expected_total);
            fail_with(buf); return 1;
        }
        if (er.promises_fulfilled != expected_total) {
            char buf[128];
            std::snprintf(buf, sizeof(buf),
                "iter %d: promises_fulfilled=%d (expected %d)",
                r, er.promises_fulfilled, expected_total);
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
        if (futures_completed != expected_total) {
            char buf[128];
            std::snprintf(buf, sizeof(buf),
                "iter %d: futures_completed=%d (expected %d)",
                r, futures_completed, expected_total);
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

        // Live Admission Slice 3: sort by request_id, not seq_id.
        // Cancelled originals and admitted reusers can share a seq_id;
        // request_id is unique per result.
        std::sort(results.begin(), results.end(),
                  [](const request_result & a, const request_result & b) {
                      return a.request_id < b.request_id;
                  });

        // Duplicate-request_id guard (Slice 3: replaces duplicate-seq_id
        // guard; seq_id can legitimately repeat across cancelled+admitted
        // pairs).
        {
            std::vector<int32_t> seen;
            seen.reserve(results.size());
            for (const auto & rr : results) {
                for (int32_t r_id : seen) {
                    if (r_id == rr.request_id) {
                        char buf[128];
                        std::snprintf(buf, sizeof(buf),
                            "iter %d: duplicate request_id=%d in results",
                            r, rr.request_id);
                        fail_with(buf); return 1;
                    }
                }
                seen.push_back(rr.request_id);
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

        // ---- Per-result validation (Slice 3: partitioned by admission)
        // For a non-admitted result, request_id == seq_id (constructor
        // invariant) and the planned-cancel lookup uses request_id so
        // that it does not falsely match an admitted result that
        // happens to live on a cancel_plan slot.
        auto is_planned_cancel = [&](int32_t request_id) {
            for (int32_t cs : args.cancel_plan) {
                if (cs == request_id) return true;
            }
            return false;
        };

        const int32_t expected_n_decoded_at_cancel  = args.cancel_after;
        const int32_t expected_cancel_observed_iter = args.cancel_after;
        const int32_t expected_admitted_at_iter     = args.cancel_after + 1;

        // Live Admission Slice 5: expected admission iter for the
        // completion-freed path. With the demand-gated push policy
        // and round-robin {8,64,256} actives, the first wave of
        // completions happens at done_iter = min_active_budget - 1.
        // Admission consumes them at top of the next iter, so
        // admitted_at_iter == min_active_budget.
        const int32_t min_active_budget = budgets.empty()
            ? 0
            : *std::min_element(budgets.begin(), budgets.end());
        const int32_t expected_admitted_at_iter_completion =
            min_active_budget;

        for (const auto & rr : results) {
            if (rr.admission_src == admission_source::none) {
                // Non-admitted (original active request).
                // request_id == seq_id is the invariant for this group.
                if (rr.request_id != rr.seq_id) {
                    char buf[200];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: non-admitted request_id=%d != seq_id=%d",
                        r, rr.request_id, rr.seq_id);
                    fail_with(buf); return 1;
                }
                if (rr.admitted_at_iter != -1
                 || rr.reused_seq_id != -1
                 || rr.previous_request_id != -1) {
                    char buf[256];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: non-admitted req=%d carries admission "
                        "fields (admitted_at_iter=%d reused_seq_id=%d "
                        "previous_request_id=%d)",
                        r, rr.request_id, rr.admitted_at_iter,
                        rr.reused_seq_id, rr.previous_request_id);
                    fail_with(buf); return 1;
                }

                const bool planned = is_planned_cancel(rr.request_id);
                if (planned) {
                    if (rr.status != request_status::cancelled) {
                        char buf[200];
                        std::snprintf(buf, sizeof(buf),
                            "iter %d: req %d status=%s (expected cancelled)",
                            r, rr.request_id, status_name(rr.status));
                        fail_with(buf); return 1;
                    }
                    if (rr.n_decoded_at_cancel != expected_n_decoded_at_cancel) {
                        char buf[200];
                        std::snprintf(buf, sizeof(buf),
                            "iter %d: req %d n_decoded_at_cancel=%d "
                            "(expected %d)", r, rr.request_id,
                            rr.n_decoded_at_cancel,
                            expected_n_decoded_at_cancel);
                        fail_with(buf); return 1;
                    }
                    if (rr.cancel_observed_iter != expected_cancel_observed_iter) {
                        char buf[200];
                        std::snprintf(buf, sizeof(buf),
                            "iter %d: req %d cancel_observed_iter=%d "
                            "(expected %d)", r, rr.request_id,
                            rr.cancel_observed_iter,
                            expected_cancel_observed_iter);
                        fail_with(buf); return 1;
                    }
                    if (rr.n_decoded != expected_n_decoded_at_cancel) {
                        char buf[200];
                        std::snprintf(buf, sizeof(buf),
                            "iter %d: req %d n_decoded=%d "
                            "(expected %d for cancelled)", r, rr.request_id,
                            rr.n_decoded, expected_n_decoded_at_cancel);
                        fail_with(buf); return 1;
                    }
                    if (rr.n_decoded != rr.n_decoded_at_cancel) {
                        char buf[200];
                        std::snprintf(buf, sizeof(buf),
                            "iter %d: req %d n_decoded=%d != "
                            "n_decoded_at_cancel=%d", r, rr.request_id,
                            rr.n_decoded, rr.n_decoded_at_cancel);
                        fail_with(buf); return 1;
                    }
                    if (rr.n_decoded >= rr.decode_budget) {
                        char buf[200];
                        std::snprintf(buf, sizeof(buf),
                            "iter %d: req %d cancelled but n_decoded=%d "
                            ">= budget=%d", r, rr.request_id,
                            rr.n_decoded, rr.decode_budget);
                        fail_with(buf); return 1;
                    }
                    const llama_pos expected_pos =
                        n_prompt_tokens + rr.n_decoded - 2;
                    if (rr.pos_max_at_clear != expected_pos) {
                        char buf[256];
                        std::snprintf(buf, sizeof(buf),
                            "iter %d: req %d cancelled pos_max_at_clear=%d "
                            "!= n_prompt+n_decoded-2=%d", r, rr.request_id,
                            rr.pos_max_at_clear,
                            static_cast<int>(expected_pos));
                        fail_with(buf); return 1;
                    }
                } else {
                    if (rr.status != request_status::completed) {
                        char buf[200];
                        std::snprintf(buf, sizeof(buf),
                            "iter %d: req %d status=%s (expected completed)",
                            r, rr.request_id, status_name(rr.status));
                        fail_with(buf); return 1;
                    }
                    if (rr.cancel_observed_iter != -1) {
                        char buf[200];
                        std::snprintf(buf, sizeof(buf),
                            "iter %d: req %d cancel_observed_iter=%d "
                            "(expected -1)", r, rr.request_id,
                            rr.cancel_observed_iter);
                        fail_with(buf); return 1;
                    }
                    if (rr.n_decoded_at_cancel != -1) {
                        char buf[200];
                        std::snprintf(buf, sizeof(buf),
                            "iter %d: req %d n_decoded_at_cancel=%d "
                            "(expected -1)", r, rr.request_id,
                            rr.n_decoded_at_cancel);
                        fail_with(buf); return 1;
                    }
                    if (rr.n_decoded != rr.decode_budget) {
                        char buf[256];
                        std::snprintf(buf, sizeof(buf),
                            "iter %d: req %d n_decoded %d != budget %d",
                            r, rr.request_id, rr.n_decoded, rr.decode_budget);
                        fail_with(buf); return 1;
                    }
                }
            } else if (rr.admission_src == admission_source::cancel_freed) {
                // Live Admission Slice 3: admitted-result gates.
                if (rr.status != request_status::completed) {
                    char buf[200];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: admitted req %d status=%s "
                        "(expected completed)",
                        r, rr.request_id, status_name(rr.status));
                    fail_with(buf); return 1;
                }
                if (rr.admitted_at_iter != expected_admitted_at_iter) {
                    char buf[200];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: admitted req %d admitted_at_iter=%d "
                        "(expected cancel_after+1=%d)",
                        r, rr.request_id, rr.admitted_at_iter,
                        expected_admitted_at_iter);
                    fail_with(buf); return 1;
                }
                if (rr.reused_seq_id != rr.seq_id) {
                    char buf[200];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: admitted req %d reused_seq_id=%d != "
                        "seq_id=%d",
                        r, rr.request_id, rr.reused_seq_id, rr.seq_id);
                    fail_with(buf); return 1;
                }
                if (!is_planned_cancel(rr.reused_seq_id)) {
                    char buf[200];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: admitted req %d reused_seq_id=%d not "
                        "in cancel_plan (cancel-freed-only rule)",
                        r, rr.request_id, rr.reused_seq_id);
                    fail_with(buf); return 1;
                }
                // For this prototype the original active set has
                // request_id == seq_id, so the prior owner of a cancel-
                // freed slot has request_id == seq_id == reused_seq_id.
                if (rr.previous_request_id != rr.reused_seq_id) {
                    char buf[200];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: admitted req %d previous_request_id=%d "
                        "!= reused_seq_id=%d",
                        r, rr.request_id, rr.previous_request_id,
                        rr.reused_seq_id);
                    fail_with(buf); return 1;
                }
                if (rr.n_decoded != rr.decode_budget) {
                    char buf[200];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: admitted req %d n_decoded=%d != "
                        "budget=%d", r, rr.request_id,
                        rr.n_decoded, rr.decode_budget);
                    fail_with(buf); return 1;
                }
                const int32_t expected_done_iter_admit =
                    rr.admitted_at_iter + rr.decode_budget - 1;
                if (rr.done_iter != expected_done_iter_admit) {
                    char buf[200];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: admitted req %d done_iter=%d "
                        "(expected admitted_at_iter+budget-1=%d)",
                        r, rr.request_id, rr.done_iter,
                        expected_done_iter_admit);
                    fail_with(buf); return 1;
                }
                const llama_pos expected_pos_admit =
                    n_prompt_tokens + rr.decode_budget - 2;
                if (rr.pos_max_at_clear != expected_pos_admit) {
                    char buf[200];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: admitted req %d pos_max_at_clear=%d "
                        "(expected n_prompt+budget-2=%d)",
                        r, rr.request_id, rr.pos_max_at_clear,
                        static_cast<int>(expected_pos_admit));
                    fail_with(buf); return 1;
                }
                if (rr.cancel_observed_iter != -1
                 || rr.n_decoded_at_cancel != -1) {
                    char buf[200];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: admitted req %d carries cancel fields "
                        "(observed_iter=%d n_decoded_at_cancel=%d)",
                        r, rr.request_id, rr.cancel_observed_iter,
                        rr.n_decoded_at_cancel);
                    fail_with(buf); return 1;
                }
            } else if (rr.admission_src == admission_source::completion_freed) {
                // Live Admission Slice 5: completion-freed admitted
                // result gates. The slot was reused after its prior
                // owner finished naturally, so cancel-family fields
                // must be defaults and is_planned_cancel must NOT
                // hold for reused_seq_id (cross-source confusion check).
                if (!args.reuse_completed) {
                    char buf[200];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: completion_freed req %d but "
                        "--reuse-completed is OFF",
                        r, rr.request_id);
                    fail_with(buf); return 1;
                }
                if (rr.status != request_status::completed) {
                    char buf[200];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: completion_freed req %d status=%s "
                        "(expected completed)",
                        r, rr.request_id, status_name(rr.status));
                    fail_with(buf); return 1;
                }
                if (rr.admitted_at_iter !=
                    expected_admitted_at_iter_completion) {
                    char buf[200];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: completion_freed req %d "
                        "admitted_at_iter=%d (expected "
                        "min_active_budget=%d)",
                        r, rr.request_id, rr.admitted_at_iter,
                        expected_admitted_at_iter_completion);
                    fail_with(buf); return 1;
                }
                if (rr.reused_seq_id != rr.seq_id) {
                    char buf[200];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: completion_freed req %d "
                        "reused_seq_id=%d != seq_id=%d",
                        r, rr.request_id, rr.reused_seq_id, rr.seq_id);
                    fail_with(buf); return 1;
                }
                if (is_planned_cancel(rr.reused_seq_id)) {
                    char buf[200];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: completion_freed req %d "
                        "reused_seq_id=%d is in cancel_plan "
                        "(source confusion)",
                        r, rr.request_id, rr.reused_seq_id);
                    fail_with(buf); return 1;
                }
                // Slice-1 invariant: the original active set has
                // request_id == seq_id, so the prior owner of a
                // completion-freed slot has previous_request_id ==
                // reused_seq_id.
                if (rr.previous_request_id != rr.reused_seq_id) {
                    char buf[200];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: completion_freed req %d "
                        "previous_request_id=%d != reused_seq_id=%d",
                        r, rr.request_id, rr.previous_request_id,
                        rr.reused_seq_id);
                    fail_with(buf); return 1;
                }
                if (rr.n_decoded != rr.decode_budget) {
                    char buf[200];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: completion_freed req %d n_decoded=%d"
                        " != budget=%d",
                        r, rr.request_id, rr.n_decoded,
                        rr.decode_budget);
                    fail_with(buf); return 1;
                }
                const int32_t expected_done_iter_admit =
                    rr.admitted_at_iter + rr.decode_budget - 1;
                if (rr.done_iter != expected_done_iter_admit) {
                    char buf[200];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: completion_freed req %d done_iter=%d"
                        " (expected admitted_at_iter+budget-1=%d)",
                        r, rr.request_id, rr.done_iter,
                        expected_done_iter_admit);
                    fail_with(buf); return 1;
                }
                const llama_pos expected_pos_admit =
                    n_prompt_tokens + rr.decode_budget - 2;
                if (rr.pos_max_at_clear != expected_pos_admit) {
                    char buf[200];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: completion_freed req %d "
                        "pos_max_at_clear=%d (expected "
                        "n_prompt+budget-2=%d)",
                        r, rr.request_id, rr.pos_max_at_clear,
                        static_cast<int>(expected_pos_admit));
                    fail_with(buf); return 1;
                }
                if (rr.cancel_observed_iter != -1
                 || rr.n_decoded_at_cancel != -1) {
                    char buf[200];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: completion_freed req %d carries "
                        "cancel fields (observed_iter=%d "
                        "n_decoded_at_cancel=%d)",
                        r, rr.request_id, rr.cancel_observed_iter,
                        rr.n_decoded_at_cancel);
                    fail_with(buf); return 1;
                }
            } else {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: req %d unknown admission_source=%s",
                    r, rr.request_id,
                    admission_source_name(rr.admission_src));
                fail_with(buf); return 1;
            }
        }

        // ---- Per-budget completed summary + partitioned hash gates ----
        // Slice 3: hash uniqueness / done_iter / pos_max are validated
        // within (admission_src, decode_budget) partitions. Surviving
        // active budget-64 and admitted budget-64 share a budget but
        // not a batch-shape history, so their hashes are not required
        // to match each other (design §6, §11.7).
        std::vector<int32_t> uniq_budgets;
        for (const auto & rr : results) {
            bool seen = false;
            for (int32_t u : uniq_budgets) if (u == rr.decode_budget) { seen = true; break; }
            if (!seen) uniq_budgets.push_back(rr.decode_budget);
        }
        std::sort(uniq_budgets.begin(), uniq_budgets.end());

        struct partition_key {
            admission_source src;
            int32_t          budget;
        };
        const partition_key partitions[] = {
            {admission_source::none,             0},
            {admission_source::cancel_freed,     0},
            {admission_source::completion_freed, 0},
        };

        for (int32_t b : uniq_budgets) {
            for (const auto & pk_template : partitions) {
                const admission_source src = pk_template.src;
                int32_t  completed_b = 0;
                int32_t  cancelled_b = 0;
                uint64_t any_hash    = 0;
                std::vector<uint64_t>  hashes;
                std::vector<int32_t>   done_iters;
                std::vector<llama_pos> pos_maxes;
                for (const auto & rr : results) {
                    if (rr.decode_budget != b)  continue;
                    if (rr.admission_src != src) continue;
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
                if (completed_b == 0 && cancelled_b == 0) continue;
                std::sort(done_iters.begin(), done_iters.end());
                std::sort(pos_maxes.begin(),  pos_maxes.end());

                // Expected anchors:
                //   admission_src == none: standard prefill at iter 0,
                //     done_iter = budget - 1, pos_max = n_prompt+budget-2.
                //   admission_src == cancel_freed: prefilled at
                //     cancel_after + 1, done_iter = admitted_at_iter +
                //     budget - 1, pos_max = n_prompt + budget - 2.
                //   admission_src == completion_freed (Slice 5):
                //     prefilled at min_active_budget, done_iter =
                //     admitted_at_iter + budget - 1, pos_max =
                //     n_prompt + budget - 2.
                int32_t expected_done_iter = b - 1;
                if (src == admission_source::cancel_freed) {
                    expected_done_iter = expected_admitted_at_iter + b - 1;
                } else if (src == admission_source::completion_freed) {
                    expected_done_iter =
                        expected_admitted_at_iter_completion + b - 1;
                }
                const int32_t expected_pos_max =
                    n_prompt_tokens + b - 2;

                fprintf(stdout,
                    "iter[%d] partition src=%s budget=%d "
                    "completed=%d cancelled=%d "
                    "unique_completed_hashes=%zu hash=0x%016llx "
                    "done_iter_set={", r,
                    admission_source_name(src), b,
                    completed_b, cancelled_b,
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
                            "iter %d: src=%s budget %d: %zu distinct "
                            "hashes among %d completed seqs (expected 1)",
                            r, admission_source_name(src), b,
                            hashes.size(), completed_b);
                        fail_with(buf); return 1;
                    }
                    if (done_iters.size() != 1
                        || done_iters[0] != expected_done_iter) {
                        char buf[256];
                        std::snprintf(buf, sizeof(buf),
                            "iter %d: src=%s budget %d completed: "
                            "done_iter set has %zu values, expected {%d}",
                            r, admission_source_name(src), b,
                            done_iters.size(), expected_done_iter);
                        fail_with(buf); return 1;
                    }
                    if (pos_maxes.size() != 1
                        || pos_maxes[0] != expected_pos_max) {
                        char buf[256];
                        std::snprintf(buf, sizeof(buf),
                            "iter %d: src=%s budget %d completed: "
                            "pos_max set has %zu values, expected {%d}",
                            r, admission_source_name(src), b,
                            pos_maxes.size(), expected_pos_max);
                        fail_with(buf); return 1;
                    }
                    // Canonical budget-8 anchor only applies to the
                    // untouched batch shape (admission_src == none).
                    // The smoke shape never admits a budget-8 request
                    // (budget-8 slots aren't in cancel_plan), so the
                    // anchor still rides on the original active set.
                    if (src == admission_source::none
                        && b == 8
                        && any_hash != k_canonical_budget_8) {
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
        }

        // Per-class gate, restricted to the original active set
        // (admission_src == none). With admission, total per-budget
        // counts include admitted reusers and would no longer divide
        // evenly; the gate's intent is "the original active mix is
        // round-robin", which is unchanged.
        if (args.n_active % static_cast<int32_t>(args.decode_budget_mix.size()) == 0) {
            const int32_t per_class =
                args.n_active / static_cast<int32_t>(args.decode_budget_mix.size());
            for (int32_t b : uniq_budgets) {
                int32_t cnt = 0;
                for (const auto & rr : results) {
                    if (rr.admission_src != admission_source::none) continue;
                    if (rr.decode_budget == b) cnt++;
                }
                if (cnt != per_class) {
                    char buf[256];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: budget %d (admission_src=none) count=%d"
                        " (expected %d)", r, b, cnt, per_class);
                    fail_with(buf); return 1;
                }
            }
        }

        // Top-level cancelled / completed totals, partitioned by
        // admission source. Cancellations only affect the original
        // active set (cancel-freed-only rule). Admitted requests
        // always complete.
        int32_t orig_completed_total = 0;
        int32_t orig_cancelled_total = 0;
        int32_t admitted_completed_total = 0;
        for (const auto & rr : results) {
            if (rr.admission_src == admission_source::none) {
                if (rr.status == request_status::completed)      orig_completed_total++;
                else if (rr.status == request_status::cancelled) orig_cancelled_total++;
            } else if (rr.admission_src == admission_source::cancel_freed
                    || rr.admission_src ==
                       admission_source::completion_freed) {
                if (rr.status == request_status::completed) admitted_completed_total++;
            }
        }
        if (orig_cancelled_total !=
            static_cast<int32_t>(args.cancel_plan.size())) {
            char buf[200];
            std::snprintf(buf, sizeof(buf),
                "iter %d: orig_cancelled_total=%d != cancel_plan size %zu",
                r, orig_cancelled_total, args.cancel_plan.size());
            fail_with(buf); return 1;
        }
        if (orig_cancelled_total != er.cancelled_count) {
            char buf[200];
            std::snprintf(buf, sizeof(buf),
                "iter %d: orig_cancelled_total=%d (snapshots) != "
                "engine.cancelled_count=%d",
                r, orig_cancelled_total, er.cancelled_count);
            fail_with(buf); return 1;
        }
        if (orig_completed_total + orig_cancelled_total != args.n_active) {
            char buf[200];
            std::snprintf(buf, sizeof(buf),
                "iter %d: orig completed=%d + cancelled=%d != n_active=%d",
                r, orig_completed_total, orig_cancelled_total,
                args.n_active);
            fail_with(buf); return 1;
        }
        if (admitted_completed_total != er.admitted_count) {
            char buf[200];
            std::snprintf(buf, sizeof(buf),
                "iter %d: admitted_completed_total=%d != "
                "engine.admitted_count=%d",
                r, admitted_completed_total, er.admitted_count);
            fail_with(buf); return 1;
        }
        const int32_t completed_total =
            orig_completed_total + admitted_completed_total;
        const int32_t cancelled_total = orig_cancelled_total;

        // Aggregate wasted-decode-rows-after-cancel.
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

        // ---- Live Admission Slice 3 gates ------------------------------
        // Live Admission Slice 4: structural gate — exactly one
        // admitted_prefilled trace site fires per admitted request.
        // Catches bugs in the trace predicate (e.g., misplaced after
        // n_decoded mutation) WITHOUT requiring trace to be enabled.
        if (er.admitted_prefill_events != er.admitted_count) {
            char buf[200];
            std::snprintf(buf, sizeof(buf),
                "iter %d: admitted_prefill_events=%d != admitted_count=%d",
                r, er.admitted_prefill_events, er.admitted_count);
            fail_with(buf); return 1;
        }

        if (er.queued_count != args.n_waiting) {
            char buf[200];
            std::snprintf(buf, sizeof(buf),
                "iter %d: engine queued_count=%d (expected n_waiting=%d)",
                r, er.queued_count, args.n_waiting);
            fail_with(buf); return 1;
        }
        // Live Admission Slice 6: admitted_count covers preloaded
        // waiters AND external arrivals; the upper bound is the sum.
        // preloaded_admissions == admitted_count - external_admitted_count
        // and is the count that maps 1:1 to queued_count drain.
        const int32_t preloaded_admissions =
            er.admitted_count - er.external_admitted_count;
        const int32_t admitted_upper_bound =
            args.n_waiting + args.n_external_arrivals;
        if (er.admitted_count < 0
         || er.admitted_count > admitted_upper_bound) {
            char buf[200];
            std::snprintf(buf, sizeof(buf),
                "iter %d: admitted_count=%d outside "
                "[0, n_waiting+n_external_arrivals=%d]",
                r, er.admitted_count, admitted_upper_bound);
            fail_with(buf); return 1;
        }
        if (preloaded_admissions < 0
         || preloaded_admissions > args.n_waiting) {
            char buf[200];
            std::snprintf(buf, sizeof(buf),
                "iter %d: preloaded_admissions=%d outside "
                "[0, n_waiting=%d] (admitted=%d external=%d)",
                r, preloaded_admissions, args.n_waiting,
                er.admitted_count, er.external_admitted_count);
            fail_with(buf); return 1;
        }
        const int32_t expected_end_size =
            args.n_waiting - preloaded_admissions;
        if (er.waiting_queue_size_at_engine_end != expected_end_size) {
            char buf[200];
            std::snprintf(buf, sizeof(buf),
                "iter %d: waiting_queue_size_at_engine_end=%d "
                "(expected n_waiting - preloaded_admissions = %d)",
                r, er.waiting_queue_size_at_engine_end,
                expected_end_size);
            fail_with(buf); return 1;
        }
        if (static_cast<int32_t>(results.size()) != expected_total) {
            char buf[200];
            std::snprintf(buf, sizeof(buf),
                "iter %d: results.size()=%zu (expected n_active+admitted"
                "=%d)", r, results.size(), expected_total);
            fail_with(buf); return 1;
        }

        // Live Admission Slice 7: reused_seq_id_set property checks.
        //   1. size == admitted_count
        //   2. NO DUPLICATES globally (the engine must never reuse the
        //      same slot twice in one run).
        //   3. STRICTLY ASCENDING WITHIN EACH (admitted_at_iter,
        //      admission_src) group. The earlier global strictly-
        //      ascending gate was Slice-3 specific: when admissions
        //      happen at a single iter from a single source, the cancel
        //      pool pops in seq_id-ascending order, so the global set
        //      is also ascending. In Slice 7, completion-freed
        //      admissions at iter 8 ({0,3,…,24}) are followed by
        //      cancel-freed admissions at iter 17 ({1,2,4,5,7,8}), so
        //      the global set is no longer monotonic. The cancel-pool
        //      and completion-pool FIFO invariants still produce
        //      ascending seq_ids within each (iter, src) group, which
        //      is what we gate.
        if (static_cast<int32_t>(er.reused_seq_id_set.size())
            != er.admitted_count) {
            char buf[200];
            std::snprintf(buf, sizeof(buf),
                "iter %d: reused_seq_id_set size=%zu != admitted_count=%d",
                r, er.reused_seq_id_set.size(), er.admitted_count);
            fail_with(buf); return 1;
        }
        // No-duplicate check (was implied by global ascending pre-Slice 7).
        {
            std::vector<int32_t> sorted_seqs = er.reused_seq_id_set;
            std::sort(sorted_seqs.begin(), sorted_seqs.end());
            for (size_t i = 1; i < sorted_seqs.size(); i++) {
                if (sorted_seqs[i - 1] == sorted_seqs[i]) {
                    char buf[200];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: reused_seq_id_set has duplicate "
                        "seq_id=%d", r, sorted_seqs[i]);
                    fail_with(buf); return 1;
                }
            }
        }
        // Per-(admitted_at_iter, admission_src) ascending check, ordered
        // by request_id (the waiting-queue FIFO order). Both cancel pool
        // and completion pool pop in seq_id-ascending order under the
        // existing engine invariants; within an admission iter the cancel
        // pass runs before the completion pass, and within each pass FIFO
        // == seq_id ascending. We do not assume the cancel and completion
        // sub-sequences interleave monotonically, so the check is
        // per-(iter, src) rather than per-iter.
        {
            const admission_source sources[] = {
                admission_source::cancel_freed,
                admission_source::completion_freed,
            };
            for (int32_t K : er.metrics.admission_iter_set) {
                for (admission_source src : sources) {
                    std::vector<std::pair<int32_t, int32_t>> ordered;
                    for (const auto & rr : results) {
                        if (rr.admitted_at_iter != K)  continue;
                        if (rr.admission_src   != src) continue;
                        ordered.emplace_back(rr.request_id,
                                             rr.reused_seq_id);
                    }
                    if (ordered.empty()) continue;
                    std::sort(ordered.begin(), ordered.end());
                    for (size_t i = 1; i < ordered.size(); i++) {
                        if (ordered[i - 1].second >= ordered[i].second) {
                            char buf[256];
                            std::snprintf(buf, sizeof(buf),
                                "iter %d: reused_seq_id sequence at "
                                "admitted_at_iter=%d src=%s not strictly "
                                "ascending (req %d seq %d after req %d "
                                "seq %d)",
                                r, K, admission_source_name(src),
                                ordered[i].first, ordered[i].second,
                                ordered[i - 1].first,
                                ordered[i - 1].second);
                            fail_with(buf); return 1;
                        }
                    }
                }
            }
        }

        // Live Admission Slice 5: per-source reused_seq_id constraints.
        //   cancel_freed     -> reused_seq_id MUST be in cancel_plan
        //   completion_freed -> reused_seq_id MUST NOT be in cancel_plan
        // The earlier blanket "must be in cancel_plan" gate fired
        // against all entries regardless of source and is now scoped.
        for (const auto & rr : results) {
            if (rr.admission_src == admission_source::cancel_freed) {
                if (!is_planned_cancel(rr.reused_seq_id)) {
                    char buf[200];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: cancel_freed req %d reused_seq_id=%d"
                        " not in cancel_plan",
                        r, rr.request_id, rr.reused_seq_id);
                    fail_with(buf); return 1;
                }
            } else if (rr.admission_src ==
                       admission_source::completion_freed) {
                if (is_planned_cancel(rr.reused_seq_id)) {
                    char buf[200];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: completion_freed req %d "
                        "reused_seq_id=%d is in cancel_plan",
                        r, rr.request_id, rr.reused_seq_id);
                    fail_with(buf); return 1;
                }
            }
        }

        // Count admissions by source so the smoke-shape strict gates
        // can fire conditionally without firing the wrong one.
        int32_t cancel_freed_admissions     = 0;
        int32_t completion_freed_admissions = 0;
        for (const auto & rr : results) {
            if (rr.admission_src == admission_source::cancel_freed)
                cancel_freed_admissions++;
            else if (rr.admission_src ==
                     admission_source::completion_freed)
                completion_freed_admissions++;
        }

        // Slice 3 smoke-shape strict gates (cancel-freed full admission):
        // fire only when cancel_plan is non-empty AND all admissions
        // were cancel_freed AND consumed the full waiting set.
        const bool slice3_strict =
            !args.cancel_plan.empty()
            && args.n_waiting > 0
            && er.admitted_count == args.n_waiting
            && cancel_freed_admissions == er.admitted_count
            && completion_freed_admissions == 0;

        if (slice3_strict) {
            // Sorted cancel_plan should equal reused_seq_id_set
            // when admitted_count == cancel_plan.size(). (Smoke
            // shape: both are 6.)
            if (static_cast<size_t>(er.admitted_count)
                <= args.cancel_plan.size()) {
                std::vector<int32_t> sorted_plan(
                    args.cancel_plan.begin(), args.cancel_plan.end());
                std::sort(sorted_plan.begin(), sorted_plan.end());
                for (int32_t i = 0; i < er.admitted_count; i++) {
                    if (er.reused_seq_id_set[
                            static_cast<size_t>(i)]
                        != sorted_plan[static_cast<size_t>(i)]) {
                        char buf[256];
                        std::snprintf(buf, sizeof(buf),
                            "iter %d: reused_seq_id_set[%d]=%d != "
                            "sorted(cancel_plan)[%d]=%d",
                            r, i,
                            er.reused_seq_id_set[
                                static_cast<size_t>(i)],
                            i, sorted_plan[static_cast<size_t>(i)]);
                        fail_with(buf); return 1;
                    }
                }
            }
            // Admitted request_ids must equal [n_active, n_active +
            // admitted_count).
            std::vector<int32_t> admitted_request_ids;
            for (const auto & rr : results) {
                if (rr.admission_src == admission_source::cancel_freed) {
                    admitted_request_ids.push_back(rr.request_id);
                }
            }
            std::sort(admitted_request_ids.begin(),
                      admitted_request_ids.end());
            for (int32_t i = 0; i < er.admitted_count; i++) {
                const int32_t expected_req = args.n_active + i;
                if (admitted_request_ids[static_cast<size_t>(i)]
                    != expected_req) {
                    char buf[200];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: admitted_request_ids[%d]=%d "
                        "(expected %d)",
                        r, i,
                        admitted_request_ids[static_cast<size_t>(i)],
                        expected_req);
                    fail_with(buf); return 1;
                }
            }
            // FIFO mapping: request n_active+i bound to
            // sorted(cancel_plan)[i]. Already enforced because both
            // lists are deterministic and ascending; this is the
            // explicit cross-check.
            std::vector<int32_t> sorted_plan_for_map(
                args.cancel_plan.begin(), args.cancel_plan.end());
            std::sort(sorted_plan_for_map.begin(),
                      sorted_plan_for_map.end());
            for (const auto & rr : results) {
                if (rr.admission_src != admission_source::cancel_freed)
                    continue;
                const int32_t i_in =
                    rr.request_id - args.n_active;
                if (i_in < 0 || i_in >= er.admitted_count) continue;
                const int32_t expected_seq =
                    sorted_plan_for_map[static_cast<size_t>(i_in)];
                if (rr.reused_seq_id != expected_seq) {
                    char buf[200];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: req %d reused_seq_id=%d "
                        "(expected sorted(cancel_plan)[%d]=%d)",
                        r, rr.request_id, rr.reused_seq_id,
                        i_in, expected_seq);
                    fail_with(buf); return 1;
                }
                if (rr.previous_request_id != expected_seq) {
                    char buf[200];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: req %d previous_request_id=%d "
                        "(expected %d)",
                        r, rr.request_id, rr.previous_request_id,
                        expected_seq);
                    fail_with(buf); return 1;
                }
            }
        }

        // Live Admission Slice 5 smoke-shape strict gates: fire only
        // when --reuse-completed is on, cancel_plan is empty, and all
        // admissions were completion_freed AND consumed the full
        // waiting set. The expected reused_seq_id_set is the first
        // n_waiting min-budget actives in round-robin (seq_id) order.
        const bool slice5_strict =
            args.reuse_completed
            && args.cancel_plan.empty()
            && args.n_waiting > 0
            && er.admitted_count == args.n_waiting
            && completion_freed_admissions == er.admitted_count
            && cancel_freed_admissions == 0;

        if (slice5_strict) {
            // Build the expected reused-seq list: every active slot
            // with budget == min_active_budget, in ascending seq_id
            // order, take the first admitted_count of those.
            std::vector<int32_t> min_budget_slots;
            min_budget_slots.reserve(
                static_cast<size_t>(args.n_active));
            for (int32_t s = 0; s < args.n_active; s++) {
                if (budgets[static_cast<size_t>(s)] ==
                    min_active_budget) {
                    min_budget_slots.push_back(s);
                }
            }
            if (static_cast<int32_t>(min_budget_slots.size())
                < er.admitted_count) {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: only %zu min-budget(%d) slots available "
                    "but admitted_count=%d",
                    r, min_budget_slots.size(), min_active_budget,
                    er.admitted_count);
                fail_with(buf); return 1;
            }
            for (int32_t i = 0; i < er.admitted_count; i++) {
                if (er.reused_seq_id_set[static_cast<size_t>(i)]
                    != min_budget_slots[static_cast<size_t>(i)]) {
                    char buf[256];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: reused_seq_id_set[%d]=%d != "
                        "min_budget_slots[%d]=%d",
                        r, i,
                        er.reused_seq_id_set[static_cast<size_t>(i)],
                        i,
                        min_budget_slots[static_cast<size_t>(i)]);
                    fail_with(buf); return 1;
                }
            }
            // Admitted request_ids must equal [n_active, n_active +
            // admitted_count). Same shape as the cancel-freed strict
            // gate; the source is the only difference.
            std::vector<int32_t> admitted_request_ids;
            for (const auto & rr : results) {
                if (rr.admission_src ==
                    admission_source::completion_freed) {
                    admitted_request_ids.push_back(rr.request_id);
                }
            }
            std::sort(admitted_request_ids.begin(),
                      admitted_request_ids.end());
            for (int32_t i = 0; i < er.admitted_count; i++) {
                const int32_t expected_req = args.n_active + i;
                if (admitted_request_ids[static_cast<size_t>(i)]
                    != expected_req) {
                    char buf[200];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: completion_freed admitted_request_"
                        "ids[%d]=%d (expected %d)",
                        r, i,
                        admitted_request_ids[static_cast<size_t>(i)],
                        expected_req);
                    fail_with(buf); return 1;
                }
            }
            // FIFO mapping: request n_active+i -> min_budget_slots[i].
            for (const auto & rr : results) {
                if (rr.admission_src !=
                    admission_source::completion_freed) continue;
                const int32_t i_in =
                    rr.request_id - args.n_active;
                if (i_in < 0 || i_in >= er.admitted_count) continue;
                const int32_t expected_seq =
                    min_budget_slots[static_cast<size_t>(i_in)];
                if (rr.reused_seq_id != expected_seq) {
                    char buf[200];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: completion_freed req %d "
                        "reused_seq_id=%d (expected "
                        "min_budget_slots[%d]=%d)",
                        r, rr.request_id, rr.reused_seq_id,
                        i_in, expected_seq);
                    fail_with(buf); return 1;
                }
                if (rr.previous_request_id != expected_seq) {
                    char buf[200];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: completion_freed req %d "
                        "previous_request_id=%d (expected %d)",
                        r, rr.request_id, rr.previous_request_id,
                        expected_seq);
                    fail_with(buf); return 1;
                }
            }
            // Pool residual gate: the first wave of min-budget natural
            // completions (size == round-robin count of that budget)
            // should leave exactly (wave_size - admitted_count) entries
            // pooled at run end. With the demand gate, later natural
            // completions (budget-64, budget-256) MUST NOT add to the
            // pool because the waiting queue is empty by then.
            const int32_t first_wave_size =
                static_cast<int32_t>(min_budget_slots.size());
            const int32_t expected_pool_residual =
                first_wave_size - er.admitted_count;
            if (er.completion_freed_pool_size_at_run_end !=
                expected_pool_residual) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: completion_freed_pool_size_at_run_end="
                    "%d (expected first_wave=%d - admitted=%d = %d)",
                    r, er.completion_freed_pool_size_at_run_end,
                    first_wave_size, er.admitted_count,
                    expected_pool_residual);
                fail_with(buf); return 1;
            }
        }

        // Live Admission Slice 5: when --reuse-completed is OFF the
        // engine MUST NOT touch the completion pool. Gate fail-closed.
        if (!args.reuse_completed
            && er.completion_freed_pool_size_at_run_end != 0) {
            char buf[200];
            std::snprintf(buf, sizeof(buf),
                "iter %d: completion_freed_pool_size_at_run_end=%d "
                "but --reuse-completed is OFF",
                r, er.completion_freed_pool_size_at_run_end);
            fail_with(buf); return 1;
        }
        if (!args.reuse_completed && completion_freed_admissions != 0) {
            char buf[200];
            std::snprintf(buf, sizeof(buf),
                "iter %d: %d completion_freed admissions seen but "
                "--reuse-completed is OFF",
                r, completion_freed_admissions);
            fail_with(buf); return 1;
        }

        // ---- Live Admission Slice 6 gates ------------------------------
        // Inert path (n_external_arrivals == 0): every Slice 6 counter
        // and set must stay at its default. This catches accidental
        // activation of the release/ack barrier or inbox drain when
        // the user did not opt in.
        if (args.n_external_arrivals == 0) {
            if (er.arrival_drained_count != 0
             || er.external_admitted_count != 0
             || er.first_external_drain_iter != -1
             || !er.iter_release_fired_set.empty()
             || !er.submitter_ack_set.empty()) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: external path active without "
                    "--n-external-arrivals "
                    "(drained=%d ext_admitted=%d first_drain=%d "
                    "release_set_size=%zu ack_set_size=%zu)",
                    r, er.arrival_drained_count,
                    er.external_admitted_count,
                    er.first_external_drain_iter,
                    er.iter_release_fired_set.size(),
                    er.submitter_ack_set.size());
                fail_with(buf); return 1;
            }
        } else {
            // arrival_drained_count == n_external_arrivals
            if (er.arrival_drained_count != args.n_external_arrivals) {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: arrival_drained_count=%d (expected %d)",
                    r, er.arrival_drained_count,
                    args.n_external_arrivals);
                fail_with(buf); return 1;
            }
            // external_admitted_count == n_external_arrivals
            if (er.external_admitted_count != args.n_external_arrivals) {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: external_admitted_count=%d (expected %d)",
                    r, er.external_admitted_count,
                    args.n_external_arrivals);
                fail_with(buf); return 1;
            }
            // first_external_drain_iter == external_release_iter + 1
            const int32_t expected_first_drain =
                args.external_release_iter + 1;
            if (er.first_external_drain_iter != expected_first_drain) {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: first_external_drain_iter=%d "
                    "(expected release_iter+1=%d)",
                    r, er.first_external_drain_iter,
                    expected_first_drain);
                fail_with(buf); return 1;
            }
            // iter_release_fired_set == {release_iter}
            if (er.iter_release_fired_set.size() != 1
             || er.iter_release_fired_set[0]
                    != args.external_release_iter) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: iter_release_fired_set size=%zu "
                    "(expected {%d})",
                    r, er.iter_release_fired_set.size(),
                    args.external_release_iter);
                fail_with(buf); return 1;
            }
            // submitter_ack_set == {release_iter}
            if (er.submitter_ack_set.size() != 1
             || er.submitter_ack_set[0]
                    != args.external_release_iter) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: submitter_ack_set size=%zu (expected {%d})",
                    r, er.submitter_ack_set.size(),
                    args.external_release_iter);
                fail_with(buf); return 1;
            }
            // External admissions must use admission_src=cancel_freed
            // for the smoke shape (no completion-freed external path
            // exercised yet) AND must carry arrival_src=external.
            int32_t external_results_seen = 0;
            for (const auto & rr : results) {
                if (rr.arrival_src != arrival_source::external) continue;
                external_results_seen++;
                if (rr.admission_src != admission_source::cancel_freed) {
                    char buf[256];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: external req %d admission_src=%s "
                        "(expected cancel_freed)",
                        r, rr.request_id,
                        admission_source_name(rr.admission_src));
                    fail_with(buf); return 1;
                }
                if (rr.admitted_at_iter != expected_admitted_at_iter) {
                    char buf[256];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: external req %d admitted_at_iter=%d "
                        "(expected cancel_after+1=%d)",
                        r, rr.request_id, rr.admitted_at_iter,
                        expected_admitted_at_iter);
                    fail_with(buf); return 1;
                }
            }
            if (external_results_seen != args.n_external_arrivals) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: external_results_seen=%d (expected %d)",
                    r, external_results_seen, args.n_external_arrivals);
                fail_with(buf); return 1;
            }
            // Every NON-external result must carry arrival_src=preloaded.
            for (const auto & rr : results) {
                if (rr.arrival_src == arrival_source::external) continue;
                if (rr.arrival_src != arrival_source::preloaded) {
                    char buf[256];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: non-external req %d arrival_src=%s "
                        "(expected preloaded)",
                        r, rr.request_id,
                        arrival_source_name(rr.arrival_src));
                    fail_with(buf); return 1;
                }
            }

            // Slice 6 strict gate: the smoke shape. cancel-plan size
            // matches n_external_arrivals, no preloaded waiters, all
            // admissions are external + cancel_freed. Verify the FIFO
            // request->seq mapping and the canonical budget-64 hash.
            const bool slice6_strict =
                !args.cancel_plan.empty()
                && args.n_waiting == 0
                && static_cast<int32_t>(args.cancel_plan.size())
                       == args.n_external_arrivals
                && er.admitted_count == args.n_external_arrivals
                && er.external_admitted_count == args.n_external_arrivals
                && cancel_freed_admissions == er.admitted_count
                && completion_freed_admissions == 0;
            if (slice6_strict) {
                std::vector<int32_t> sorted_plan(
                    args.cancel_plan.begin(), args.cancel_plan.end());
                std::sort(sorted_plan.begin(), sorted_plan.end());
                // request n_active + i -> sorted(cancel_plan)[i]
                for (const auto & rr : results) {
                    if (rr.arrival_src != arrival_source::external) continue;
                    const int32_t i_in = rr.request_id - args.n_active;
                    if (i_in < 0 || i_in >= args.n_external_arrivals) {
                        char buf[256];
                        std::snprintf(buf, sizeof(buf),
                            "iter %d: external req %d outside expected "
                            "[%d, %d) id range",
                            r, rr.request_id, args.n_active,
                            args.n_active + args.n_external_arrivals);
                        fail_with(buf); return 1;
                    }
                    const int32_t expected_seq =
                        sorted_plan[static_cast<size_t>(i_in)];
                    if (rr.reused_seq_id != expected_seq) {
                        char buf[256];
                        std::snprintf(buf, sizeof(buf),
                            "iter %d: external req %d reused_seq_id=%d "
                            "(expected sorted(cancel_plan)[%d]=%d)",
                            r, rr.request_id, rr.reused_seq_id,
                            i_in, expected_seq);
                        fail_with(buf); return 1;
                    }
                }
                // budget-64 canonical hash anchor for the slice 6 smoke
                // shape (cancel-freed external admissions only).
                constexpr uint64_t k_canonical_slice6_budget_64 =
                    0x3b15a0474dfe11beull;
                if (args.external_arrival_budget == 64) {
                    for (const auto & rr : results) {
                        if (rr.arrival_src != arrival_source::external) continue;
                        if (rr.decode_budget != 64) continue;
                        if (rr.hash != k_canonical_slice6_budget_64) {
                            char buf[256];
                            std::snprintf(buf, sizeof(buf),
                                "iter %d: external req %d budget=64 "
                                "hash=0x%016llx != canonical 0x%016llx",
                                r, rr.request_id,
                                static_cast<unsigned long long>(rr.hash),
                                static_cast<unsigned long long>(
                                    k_canonical_slice6_budget_64));
                            fail_with(buf); return 1;
                        }
                    }
                }
            }
        }

        // ---- Live Admission Slice 7 strict mixed-source gate ----------
        // Fires only on the canonical Slice 7 smoke shape: reuse_completed
        // ON, cancel-plan non-empty, both preloaded waiters AND external
        // arrivals present, completion_freed admissions matched n_waiting
        // (phase 1), cancel_freed admissions matched n_external_arrivals
        // (phase 2), and admission fired at exactly two iters
        // {min_active_budget, cancel_after+1}. The body asserts the
        // mixed-source priority rule: at the iter where both pools are
        // non-empty (phase 2), cancel_freed drains first, completion_freed
        // residual is unchanged.
        const bool slice7_strict =
            args.reuse_completed
            && !args.cancel_plan.empty()
            && args.n_waiting           > 0
            && args.n_external_arrivals > 0
            && completion_freed_admissions == args.n_waiting
            && cancel_freed_admissions     == args.n_external_arrivals
            && er.external_admitted_count  == args.n_external_arrivals
            && er.admitted_count           == args.n_waiting
                                              + args.n_external_arrivals;
        if (slice7_strict) {
            const int32_t expected_phase1_iter =
                expected_admitted_at_iter_completion;  // == min_active_budget
            const int32_t expected_phase2_iter =
                expected_admitted_at_iter;             // == cancel_after + 1

            // admission_iter_set must be exactly {phase1_iter, phase2_iter}.
            const auto & ais = er.metrics.admission_iter_set;
            std::set<int32_t> ais_set(ais.begin(), ais.end());
            std::set<int32_t> expected_ais{expected_phase1_iter,
                                           expected_phase2_iter};
            if (ais_set != expected_ais) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: slice7 admission_iter_set size=%zu does "
                    "not equal {phase1=%d, phase2=%d}",
                    r, ais.size(), expected_phase1_iter,
                    expected_phase2_iter);
                fail_with(buf); return 1;
            }

            // Phase 1: every admitted result with admitted_at_iter ==
            // phase1_iter must be completion_freed + arrival_src=preloaded.
            // Phase 2: every admitted result with admitted_at_iter ==
            // phase2_iter must be cancel_freed + arrival_src=external.
            int32_t phase1_seen = 0;
            int32_t phase2_seen = 0;
            for (const auto & rr : results) {
                if (rr.admission_src == admission_source::none) continue;
                if (rr.admitted_at_iter == expected_phase1_iter) {
                    phase1_seen++;
                    if (rr.admission_src
                            != admission_source::completion_freed) {
                        char buf[256];
                        std::snprintf(buf, sizeof(buf),
                            "iter %d: slice7 phase1 req %d "
                            "admission_src=%s (expected completion_freed)",
                            r, rr.request_id,
                            admission_source_name(rr.admission_src));
                        fail_with(buf); return 1;
                    }
                    if (rr.arrival_src != arrival_source::preloaded) {
                        char buf[256];
                        std::snprintf(buf, sizeof(buf),
                            "iter %d: slice7 phase1 req %d "
                            "arrival_src=%s (expected preloaded)",
                            r, rr.request_id,
                            arrival_source_name(rr.arrival_src));
                        fail_with(buf); return 1;
                    }
                } else if (rr.admitted_at_iter == expected_phase2_iter) {
                    phase2_seen++;
                    if (rr.admission_src
                            != admission_source::cancel_freed) {
                        char buf[256];
                        std::snprintf(buf, sizeof(buf),
                            "iter %d: slice7 phase2 req %d "
                            "admission_src=%s (expected cancel_freed)",
                            r, rr.request_id,
                            admission_source_name(rr.admission_src));
                        fail_with(buf); return 1;
                    }
                    if (rr.arrival_src != arrival_source::external) {
                        char buf[256];
                        std::snprintf(buf, sizeof(buf),
                            "iter %d: slice7 phase2 req %d "
                            "arrival_src=%s (expected external)",
                            r, rr.request_id,
                            arrival_source_name(rr.arrival_src));
                        fail_with(buf); return 1;
                    }
                } else {
                    char buf[256];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: slice7 admitted req %d at unexpected "
                        "iter %d (expected %d or %d)",
                        r, rr.request_id, rr.admitted_at_iter,
                        expected_phase1_iter, expected_phase2_iter);
                    fail_with(buf); return 1;
                }
            }
            if (phase1_seen != args.n_waiting) {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: slice7 phase1_seen=%d (expected n_waiting=%d)",
                    r, phase1_seen, args.n_waiting);
                fail_with(buf); return 1;
            }
            if (phase2_seen != args.n_external_arrivals) {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: slice7 phase2_seen=%d (expected "
                    "n_external_arrivals=%d)",
                    r, phase2_seen, args.n_external_arrivals);
                fail_with(buf); return 1;
            }

            // Phase 1 mapping: request n_active + i -> min_budget_slots[i],
            // where min_budget_slots are the budget-min actives in
            // ascending seq_id order. For the smoke shape: budget-8 slots
            // {0,3,6,9,12,15,18,21,24,…}; the first n_waiting of those
            // are the phase 1 admissions.
            std::vector<int32_t> min_budget_slots;
            min_budget_slots.reserve(static_cast<size_t>(args.n_active));
            for (int32_t s = 0; s < args.n_active; s++) {
                if (budgets[static_cast<size_t>(s)] == min_active_budget) {
                    min_budget_slots.push_back(s);
                }
            }
            if (static_cast<int32_t>(min_budget_slots.size())
                    < args.n_waiting) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: slice7 only %zu min-budget(%d) actives "
                    "available but n_waiting=%d",
                    r, min_budget_slots.size(), min_active_budget,
                    args.n_waiting);
                fail_with(buf); return 1;
            }
            for (const auto & rr : results) {
                if (rr.admission_src != admission_source::completion_freed)
                    continue;
                const int32_t i_in = rr.request_id - args.n_active;
                if (i_in < 0 || i_in >= args.n_waiting) {
                    char buf[256];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: slice7 phase1 req %d outside expected "
                        "[%d, %d) preloaded id range",
                        r, rr.request_id, args.n_active,
                        args.n_active + args.n_waiting);
                    fail_with(buf); return 1;
                }
                const int32_t expected_seq =
                    min_budget_slots[static_cast<size_t>(i_in)];
                if (rr.reused_seq_id != expected_seq) {
                    char buf[256];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: slice7 phase1 req %d reused_seq_id=%d "
                        "(expected min_budget_slots[%d]=%d)",
                        r, rr.request_id, rr.reused_seq_id, i_in,
                        expected_seq);
                    fail_with(buf); return 1;
                }
            }

            // Phase 2 mapping: request n_active + n_waiting + i ->
            // sorted(cancel_plan)[i]. Same shape as Slice 6, but offset
            // by n_waiting in the request_id space.
            std::vector<int32_t> sorted_plan(
                args.cancel_plan.begin(), args.cancel_plan.end());
            std::sort(sorted_plan.begin(), sorted_plan.end());
            for (const auto & rr : results) {
                if (rr.admission_src != admission_source::cancel_freed)
                    continue;
                const int32_t i_in = rr.request_id
                                      - args.n_active
                                      - args.n_waiting;
                if (i_in < 0 || i_in >= args.n_external_arrivals) {
                    char buf[256];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: slice7 phase2 req %d outside expected "
                        "[%d, %d) external id range",
                        r, rr.request_id,
                        args.n_active + args.n_waiting,
                        args.n_active + args.n_waiting
                            + args.n_external_arrivals);
                    fail_with(buf); return 1;
                }
                const int32_t expected_seq =
                    sorted_plan[static_cast<size_t>(i_in)];
                if (rr.reused_seq_id != expected_seq) {
                    char buf[256];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: slice7 phase2 req %d reused_seq_id=%d "
                        "(expected sorted(cancel_plan)[%d]=%d)",
                        r, rr.request_id, rr.reused_seq_id, i_in,
                        expected_seq);
                    fail_with(buf); return 1;
                }
            }

            // No phase-2 admission may bind a residual completion-freed
            // seq_id (a min-budget slot NOT consumed in phase 1). The
            // residual set is min_budget_slots[n_waiting:].
            std::set<int32_t> residual_seqs;
            for (size_t i = static_cast<size_t>(args.n_waiting);
                 i < min_budget_slots.size(); i++) {
                residual_seqs.insert(min_budget_slots[i]);
            }
            for (const auto & rr : results) {
                if (rr.admitted_at_iter != expected_phase2_iter) continue;
                if (residual_seqs.count(rr.reused_seq_id)) {
                    char buf[256];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: slice7 phase2 req %d reused_seq_id=%d "
                        "is in completion-freed residual pool — cancel-"
                        "freed priority violated",
                        r, rr.request_id, rr.reused_seq_id);
                    fail_with(buf); return 1;
                }
            }

            // Completion-freed pool residual: first_wave_size = count of
            // min-budget actives (28 in the smoke); completion_freed
            // admissions == n_waiting. Demand gate stops later pushes
            // because waiting_queue is empty by the time admitted
            // budget-8 requests complete and surviving budget-64/256
            // actives complete. So residual = first_wave - n_waiting.
            const int32_t first_wave_size =
                static_cast<int32_t>(min_budget_slots.size());
            const int32_t expected_pool_residual =
                first_wave_size - args.n_waiting;
            if (er.completion_freed_pool_size_at_run_end
                    != expected_pool_residual) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: slice7 completion_freed_pool_size_at_run_end"
                    "=%d (expected first_wave=%d - n_waiting=%d = %d)",
                    r, er.completion_freed_pool_size_at_run_end,
                    first_wave_size, args.n_waiting,
                    expected_pool_residual);
                fail_with(buf); return 1;
            }
        }

        fprintf(stdout,
            "iter[%d] admit_step5: orig completed=%d cancelled=%d, "
            "admitted=%d (cancel_freed=%d completion_freed=%d), "
            "total_results=%d, queued=%d waiting_end=%d, "
            "completion_pool_residual=%d\n",
            r, orig_completed_total, orig_cancelled_total,
            er.admitted_count,
            cancel_freed_admissions, completion_freed_admissions,
            expected_total,
            er.queued_count, er.waiting_queue_size_at_engine_end,
            er.completion_freed_pool_size_at_run_end);

        fprintf(stdout,
            "iter[%d] admit_step6: external arrivals "
            "drained=%d admitted=%d first_drain_iter=%d "
            "release_set_size=%zu ack_set_size=%zu\n",
            r, er.arrival_drained_count, er.external_admitted_count,
            er.first_external_drain_iter,
            er.iter_release_fired_set.size(),
            er.submitter_ack_set.size());

        // Live Admission Slice 7 audit line. Sources counts from the
        // results vector (status-true cancel_freed_admissions and
        // completion_freed_admissions are already computed above) and
        // from engine_result. admission_iter_set is rendered as a
        // sorted ascending sequence — the engine pushes it in iter
        // order so it is already monotonic, but we format it
        // explicitly to make the audit line grep-friendly.
        {
            fprintf(stdout,
                "iter[%d] admit_step7: phase1@iter=%d completion_freed=%d "
                "phase2@iter=%d cancel_freed=%d pool_residual=%d "
                "admission_iter_set={",
                r, expected_admitted_at_iter_completion,
                completion_freed_admissions,
                expected_admitted_at_iter,
                cancel_freed_admissions,
                er.completion_freed_pool_size_at_run_end);
            for (size_t i = 0;
                 i < er.metrics.admission_iter_set.size(); i++) {
                fprintf(stdout, "%s%d", i == 0 ? "" : ",",
                        er.metrics.admission_iter_set[i]);
            }
            fprintf(stdout, "} first_external_drain_iter=%d\n",
                    er.first_external_drain_iter);
        }

        fprintf(stdout,
            "iter[%d] residual_kv: all %d seqs cleared "
            "(pos_min=-1, pos_max=-1)\n", r, args.n_seqs);

        // status_summary: Slice 3 retains the grep-friendly summary.
        fprintf(stdout,
            "iter[%d] status_summary: completed=%d cancelled=%d "
            "total=%d (orig_completed=%d admitted_completed=%d)\n",
            r, completed_total, cancelled_total,
            completed_total + cancelled_total,
            orig_completed_total, admitted_completed_total);

        // Per-budget cancellation summary (unchanged Slice-2 output;
        // useful when --cancel-plan is non-default).
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
            // Live Admission Slice 3: descriptive only.
            fprintf(stdout, "  queued_count            = %d\n",
                    er.queued_count);
            fprintf(stdout, "  admitted_count          = %d\n",
                    er.admitted_count);
            fprintf(stdout, "  waiting_queue_size_at_engine_end = %d\n",
                    er.waiting_queue_size_at_engine_end);
            fprintf(stdout, "  reused_seq_id_set       = {");
            for (size_t i = 0; i < er.reused_seq_id_set.size(); i++) {
                fprintf(stdout, "%s%d", i == 0 ? "" : ",",
                        er.reused_seq_id_set[i]);
            }
            fprintf(stdout, "}\n");
            // Live Admission Slice 4: descriptive only.
            fprintf(stdout, "  reused_seq_id_count     = %zu\n",
                    er.reused_seq_id_set.size());
            fprintf(stdout, "  admitted_prefill_events = %d\n",
                    er.admitted_prefill_events);
            fprintf(stdout, "  admission_iter_set      = {");
            for (size_t i = 0;
                 i < m.admission_iter_set.size(); i++) {
                fprintf(stdout, "%s%d", i == 0 ? "" : ",",
                        m.admission_iter_set[i]);
            }
            fprintf(stdout, "}\n");
            // waiting_queue_depth_after_admission_per_iter is sampled
            // AFTER each iter's admission loop, so for the smoke shape
            // this is n_waiting for iters 1..16 and 0 from iter 17 on.
            std::vector<double> wqd;
            wqd.reserve(
                m.waiting_queue_depth_after_admission_per_iter.size());
            int32_t wqd_max = 0;
            for (int32_t v :
                 m.waiting_queue_depth_after_admission_per_iter) {
                wqd.push_back(static_cast<double>(v));
                if (v > wqd_max) wqd_max = v;
            }
            const double wqd_p50 = percentile(wqd, 0.50);
            const double wqd_p95 = percentile(wqd, 0.95);
            fprintf(stdout,
                "  waiting_queue_depth_after_admission_per_iter "
                "p50=%.0f p95=%.0f max=%d (samples=%zu)\n",
                wqd_p50, wqd_p95, wqd_max, wqd.size());
            // admitted_ttc_ms[budget=B]: completion-time distribution
            // for admitted-completed results, segmented by budget AND
            // by admission source (so cancel_freed and completion_freed
            // admissions are not blurred together). Descriptive only.
            for (int32_t b : uniq_budgets) {
                std::vector<double> ttc_cf;   // cancel_freed
                std::vector<double> ttc_pf;   // completion_freed (Slice 5)
                for (const auto & rr : results) {
                    if (rr.decode_budget != b) continue;
                    if (rr.status != request_status::completed) continue;
                    const double v =
                        static_cast<double>(rr.ttc_us) / 1000.0;
                    if (rr.admission_src ==
                        admission_source::cancel_freed) {
                        ttc_cf.push_back(v);
                    } else if (rr.admission_src ==
                               admission_source::completion_freed) {
                        ttc_pf.push_back(v);
                    }
                }
                if (!ttc_cf.empty()) {
                    fprintf(stdout,
                        "  admitted_ttc_ms[src=cancel_freed,"
                        "budget=%d]  mean=%.2f p95=%.2f\n",
                        b, mean_of(ttc_cf),
                        percentile(ttc_cf, 0.95));
                }
                if (!ttc_pf.empty()) {
                    fprintf(stdout,
                        "  admitted_ttc_ms[src=completion_freed,"
                        "budget=%d]  mean=%.2f p95=%.2f\n",
                        b, mean_of(ttc_pf),
                        percentile(ttc_pf, 0.95));
                }
            }
            // Live Admission Slice 5: descriptive — residual size of
            // free_due_to_completion_ at engine end. Demand-gated, so
            // this stays at (first_wave - admitted) for the Slice 5
            // smoke shape (== 21).
            fprintf(stdout,
                "  completion_freed_pool_size_at_run_end = %d\n",
                er.completion_freed_pool_size_at_run_end);
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
                if (a.seq_id              != a0.seq_id
                 || a.request_id          != a0.request_id
                 || a.n_decoded           != a0.n_decoded
                 || a.generated_tokens    != a0.generated_tokens
                 || a.hash                != a0.hash
                 || a.done_iter           != a0.done_iter
                 || a.pos_max_at_clear    != a0.pos_max_at_clear
                 || a.admitted_at_iter    != a0.admitted_at_iter
                 || a.reused_seq_id       != a0.reused_seq_id
                 || a.previous_request_id != a0.previous_request_id
                 || a.admission_src       != a0.admission_src) {
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
