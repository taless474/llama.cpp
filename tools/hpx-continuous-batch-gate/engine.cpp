#include "engine.h"

#include "token_hash.h"
#include "trace.h"

#include "common.h"
#include "llama.h"

#include <hpx/hpx.hpp>
#include <hpx/lcos_local/channel.hpp>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <thread>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

using namespace token_hash;

// ---- Phase 1 serving-overhead diagnostics --------------------------------
// Default-OFF. With LLAMA_HPX_DIAG_METRICS unset, diag_enabled() returns
// false and every guarded site is a single not-taken branch — no chrono
// call, no vector push, no JSON formatting, no file I/O.
//
// Per-engine state: the only file-scope state below is the env-switch
// caches (diag_enabled() / diag_path()) which are read-only after first
// access. The per-iter scratch accumulator and every output vector live
// inside engine_result::metrics, owned by each engine instance. Multiple
// engines running concurrently in the same process (engine-pool framing,
// future replication) therefore never share Phase 1 state — every read
// and write goes through result_.metrics.cur_iter_diag.* and
// result_.metrics.*_per_iter, both of which are per-engine.
//
// Vector alignment: Phase 1 per-iter vectors are pushed at the same
// site as result_.metrics.active_seqs_per_iter (inside iter_run_decode).
// One push per iter that actually called llama_decode via the per-iter
// site. Iters that hit the no-active early break do not push, matching
// the existing active_seqs_per_iter semantics.
//
// Two llama_decode sites exist:
//   (a) the per-iter site in iter_run_decode (engine.cpp:iter_run_decode)
//   (b) a pre-loop preloaded-prefill site in run_body (fires only when
//       budgets_ is non-empty, before any iter)
// Phase 1 deliberately instruments site (a) only. Site (b) is excluded
// by design, for two reasons:
//   1. The hpx-server path constructs engines with budgets_={}, so site
//      (b) never fires in the production serving path that the c > 1
//      diagnostics are meant to analyze.
//   2. Gate/smoke runs with preloaded actives DO hit site (b), and
//      including it would create a half-iter row (no admission, no
//      cancellation, no per-iter scratch valid). The cleaner contract
//      is "per-iter rows describe iters from the inner-while loop only."
// The exclusion is surfaced to consumers in the engine_summary JSONL
// row via the "preload_prefill_llama_decode_included" key (always 0
// in Phase 1) so analysis code can detect the omission unambiguously
// rather than infer it. The dump's rows_per_batch offset code below
// handles gate/smoke runs where rows_per_batch has the extra leading
// preload entry.
bool diag_enabled() {
    static const bool v = []() -> bool {
        const char * e = std::getenv("LLAMA_HPX_DIAG_METRICS");
        return e != nullptr && std::strcmp(e, "1") == 0;
    }();
    return v;
}

const char * diag_path() {
    static const char * const v =
        std::getenv("LLAMA_HPX_DIAG_METRICS_PATH");
    return v;
}

// Emit one JSONL row per iter plus one engine_summary row. Called at
// the tail of engine::run() when diag_enabled(). Bare fprintf (no
// nlohmann::json in engine TU). Writes to LLAMA_HPX_DIAG_METRICS_PATH
// if set (O_APPEND, line-buffered), stderr otherwise.
void dump_metrics_jsonl(const engine_result & r) {
    const char * path = diag_path();
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

    const auto &  m = r.metrics;
    const size_t  n = m.active_seqs_per_iter.size();
    // rows_per_batch may have an extra leading entry from the
    // preloaded-prefill site; compute the per-iter offset.
    const size_t  rpb_offset =
        (m.rows_per_batch.size() > n) ? (m.rows_per_batch.size() - n) : 0;

    for (size_t i = 0; i < n; i++) {
        const int32_t batch_n_tokens =
            (i + rpb_offset < m.rows_per_batch.size())
              ? m.rows_per_batch[i + rpb_offset] : 0;
        const int32_t prefill =
            (i < m.prefill_rows_per_iter.size())
              ? m.prefill_rows_per_iter[i] : 0;
        const int32_t decode_r =
            (i < m.decode_rows_per_iter.size())
              ? m.decode_rows_per_iter[i] : 0;
        const int32_t admitted =
            (i < m.admitted_per_iter.size())
              ? m.admitted_per_iter[i] : 0;
        const int32_t completed =
            (i < m.completed_per_iter.size())
              ? m.completed_per_iter[i] : 0;
        const int32_t cancelled =
            (i < m.cancelled_per_iter.size())
              ? m.cancelled_per_iter[i] : 0;
        const int32_t tok_emit =
            (i < m.tokens_emitted_per_iter.size())
              ? m.tokens_emitted_per_iter[i] : 0;
        const int32_t wq_depth =
            (i < m.waiting_queue_depth_after_admission_per_iter.size())
              ? m.waiting_queue_depth_after_admission_per_iter[i] : 0;
        const int64_t decode_us =
            (i < m.llama_decode_wall_us_per_iter.size())
              ? m.llama_decode_wall_us_per_iter[i] : 0;
        const int64_t iter_us =
            (i < m.iter_wall_us_per_iter.size())
              ? m.iter_wall_us_per_iter[i] : 0;
        const int64_t idle_us =
            (i < m.idle_wait_us_per_iter.size())
              ? m.idle_wait_us_per_iter[i] : 0;
        const int64_t t_from_start_us =
            (i < m.t_us_from_engine_start_per_iter.size())
              ? m.t_us_from_engine_start_per_iter[i] : 0;
        std::fprintf(fp,
            "{\"kind\":\"iter\",\"iter\":%zu,"
            "\"t_us_from_engine_start\":%lld,"
            "\"batch_n_tokens\":%d,"
            "\"active_seq_count\":%d,"
            "\"prefill_rows_in_iter\":%d,"
            "\"decode_rows_in_iter\":%d,"
            "\"admitted_in_iter\":%d,"
            "\"completed_in_iter\":%d,"
            "\"cancelled_in_iter\":%d,"
            "\"tokens_emitted_in_iter\":%d,"
            "\"waiting_queue_depth_after_admission\":%d,"
            "\"llama_decode_wall_us\":%lld,"
            "\"iter_wall_us\":%lld,"
            "\"idle_wait_us\":%lld}\n",
            i,
            static_cast<long long>(t_from_start_us),
            batch_n_tokens,
            m.active_seqs_per_iter[i],
            prefill, decode_r,
            admitted, completed, cancelled, tok_emit,
            wq_depth,
            static_cast<long long>(decode_us),
            static_cast<long long>(iter_us),
            static_cast<long long>(idle_us));
    }

    // Phase 1: preload_prefill_llama_decode_included is always 0 in
    // this slice — the preloaded-prefill llama_decode call site (the
    // pre-inner-while site in run_body, fired only when budgets_ is
    // non-empty) is intentionally excluded from per-iter rows. Surfacing
    // it as an explicit key in engine_summary lets analysis code detect
    // the omission unambiguously rather than infer it from row counts.
    std::fprintf(fp,
        "{\"kind\":\"engine_summary\","
        "\"decode_calls\":%d,"
        "\"update_iterations\":%d,"
        "\"wall_ms\":%.3f,"
        "\"admitted_count\":%d,"
        "\"promises_fulfilled\":%d,"
        "\"cancelled_count\":%d,"
        "\"streams_closed_completed\":%d,"
        "\"streams_closed_cancelled\":%d,"
        "\"streams_closed_error\":%d,"
        "\"engine_t_start_us_absolute\":%lld,"
        "\"preload_prefill_llama_decode_included\":0}\n",
        r.decode_calls,
        r.metrics.update_iterations,
        r.metrics.wall_ms,
        r.admitted_count,
        r.promises_fulfilled,
        r.cancelled_count,
        r.streams_closed_completed,
        r.streams_closed_cancelled,
        r.streams_closed_error,
        static_cast<long long>(r.engine_t_start_us_absolute));

    if (need_close) {
        std::fclose(fp);
    } else {
        std::fflush(fp);
    }
}

// N3.1 (Exp 13): absolute steady_clock microseconds since the
// system's steady_clock epoch. Called only when
// engine.enable_responsiveness_timing_ is true; with the flag off
// no engine path executes this. Producers (benchmark binary) use
// the same expression so engine-side and submitter-side stamps
// share a single time domain.
inline int64_t now_us() noexcept {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
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

// M6c: validates `cfg` and builds a per-seq stochastic sampler chain.
// Returns a non-null llama_sampler_ptr on success, or an empty
// llama_sampler_ptr with `err` populated on validation/init failure.
// Engine-task-only — invoked exclusively from admit_one. The chain
// owns every sub-sampler added via llama_sampler_chain_add; sub-
// samplers MUST NOT be wrapped in a separate llama_sampler_ptr.
//
// M7b: validation rules moved to `validate_sampling_config` in
// types.h so the hpx-server can fail-fast at the network edge with
// the same rules. The engine remains the ultimate safety net — if
// any caller routes around the server-side validation, the
// pre-build check here still rejects invalid configs and admit_one
// fails closed via result_.error.
llama_sampler_ptr build_sampler_chain(const sampling_config & cfg,
                                      std::string &           err) {
    if (!validate_sampling_config(cfg, err)) {
        return llama_sampler_ptr{};
    }

    llama_sampler_chain_params cp =
        llama_sampler_chain_default_params();
    cp.no_perf = true;
    llama_sampler * chain = llama_sampler_chain_init(cp);
    if (chain == nullptr) {
        err = "llama_sampler_chain_init returned null";
        return llama_sampler_ptr{};
    }
    llama_sampler_ptr owner(chain);

    if (cfg.top_k > 0) {
        llama_sampler_chain_add(
            chain, llama_sampler_init_top_k(cfg.top_k));
    }
    if (cfg.top_p < 1.0f) {
        llama_sampler_chain_add(
            chain,
            llama_sampler_init_top_p(
                cfg.top_p, static_cast<size_t>(cfg.top_p_min_keep)));
    }
    if (cfg.temperature != 1.0f) {
        llama_sampler_chain_add(
            chain, llama_sampler_init_temp(cfg.temperature));
    }
    llama_sampler_chain_add(
        chain, llama_sampler_init_dist(cfg.seed));

    return owner;
}

}  // namespace

engine::engine(engine_options opts)
    : ctx_(opts.lib.ctx),
      vocab_(opts.lib.vocab),
      n_vocab_(opts.lib.n_vocab),
      // M4c: prompt_tokens_ is now a const-pointer (optional). It is
      // guaranteed non-null whenever budgets_ is non-empty (see the
      // ctor body guard); the only read sites are gated accordingly.
      prompt_tokens_(opts.preload.prompt_tokens),
      budgets_(std::move(opts.preload.budgets)),
      batch_capacity_(opts.lib.batch_capacity),
      // M2f: promises_ and seqs_ are sized to the total slot population
      // (initial actives + idle pool). With initial_idle_slots == 0 the
      // sum collapses to budgets_.size(), matching the pre-M2f layout
      // byte-for-byte. Slots in [budgets_.size(), seqs_.size()) are
      // never bound to initial actives; admit_one rebinds them when
      // the admission step consumes from free_idle_.
      promises_(budgets_.size()
                + static_cast<size_t>(std::max<int32_t>(
                    opts.lib.initial_idle_slots, 0))),
      seqs_(budgets_.size()
            + static_cast<size_t>(std::max<int32_t>(
                opts.lib.initial_idle_slots, 0))),
      n_seq_max_(opts.lib.n_seq_max),
      waiting_queue_(opts.preload.waiting_queue),
      reuse_completed_(opts.preload.reuse_completed),
      keep_alive_(opts.lib.keep_alive),
      cooperative_yield_on_pump_(opts.lib.cooperative_yield_on_pump),
      enable_responsiveness_timing_(opts.lib.enable_responsiveness_timing),
      prefill_budget_rows_(opts.lib.prefill_budget_rows),
      iter_release_set_(std::move(opts.gate_test.release_iter_set)),
      max_decode_iters_(opts.gate_test.max_decode_iters),
      stream_all_(opts.preload.stream_all)
{
    if (opts.lib.ctx == nullptr) {
        throw std::invalid_argument(
            "engine_options::lib.ctx must be non-null");
    }
    if (opts.lib.vocab == nullptr) {
        throw std::invalid_argument(
            "engine_options::lib.vocab must be non-null");
    }
    // M4c: prompt_tokens is required only when budgets is non-empty
    // (preloaded active requests). submit_request-only consumers can
    // leave it null. budgets_ has already been move-initialized above,
    // so we check the engine-owned copy.
    if (!budgets_.empty() && opts.preload.prompt_tokens == nullptr) {
        throw std::invalid_argument(
            "engine_options::preload.prompt_tokens must be non-null "
            "when preload.budgets is non-empty");
    }
    // M4b: waiting_queue may be null. A null pointer is treated as
    // an empty preloaded waiting queue; run_body()'s per-repeat
    // reseed leaves waiting_queue_consumable_ empty in that case.
    if (opts.lib.initial_idle_slots < 0) {
        throw std::invalid_argument(
            "engine_options::lib.initial_idle_slots must be >= 0");
    }
    if (static_cast<size_t>(opts.lib.n_seq_max) < seqs_.size()) {
        throw std::invalid_argument(
            "engine_options::lib.n_seq_max must be >= "
            "preload.budgets.size() + lib.initial_idle_slots");
    }
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
        // M1b: populate the per-seq owned prompt vector for each
        // initial active seq from the existing shared prompt source.
        // The initial-prefill loop in run_body() reads from this
        // owned copy instead of the shared prompt_tokens_ borrow.
        // M1c added the symmetric path for waiters/external arrivals
        // — admit_one moves the bound waiter's prompt vector into
        // the slot's seq_state::prompt_tokens, and the admitted-
        // prefill loop now reads it. M1d adds the per-active-prompt
        // override: when main supplied per_active_prompt_tokens
        // (file mode), seed each initial seq from the matching
        // file-tokenized vector. Otherwise fall back to the shared
        // prompt_tokens_ borrow exactly as M1b/M1c did. The shared
        // borrow is still retained so the no-file path remains
        // byte-identical.
        if (opts.preload.per_active_prompt_tokens != nullptr
         && opts.preload.per_active_prompt_tokens->size() >= budgets_.size()) {
            seqs_[s].prompt_tokens =
                (*opts.preload.per_active_prompt_tokens)[s];
        } else {
            // M4c: prompt_tokens_ is guaranteed non-null here because
            // this branch only runs when budgets_ is non-empty, which
            // is the exact condition under which the ctor guard
            // requires opts.preload.prompt_tokens != nullptr.
            seqs_[s].prompt_tokens = *prompt_tokens_;
        }
    }
    // M2f: initialize the idle slots ([budgets_.size(), seqs_.size()))
    // to clean "available, never bound" state. seq_id is the slot
    // index; done=true so the prefill / decode_row / cancel-observe
    // / any_active() loops skip the slot; decode_budget=0 and
    // request_id=-1 are sentinels until admit_one rebinds them.
    // free_idle_ is seeded here in ctor and re-seeded in run_body's
    // per-repeat reset so the pool starts fresh on every run().
    for (size_t s = budgets_.size(); s < seqs_.size(); s++) {
        seqs_[s].seq_id        = static_cast<int32_t>(s);
        seqs_[s].request_id    = -1;
        seqs_[s].decode_budget = 0;
        seqs_[s].done          = true;
        free_idle_.push_back(static_cast<int32_t>(s));
    }
    // Cancel Slice 2: propagate the deterministic plan into
    // seq_state. Out-of-range seq_ids are silently skipped here
    // because main has already validated the plan against
    // [0, n_seqs) and printed the per-seq budget assertion.
    // M2f: idle slots can technically appear in the plan (cs in
    // [budgets_.size(), seqs_.size())) but are inert because
    // cancel_should_observe short-circuits on seq.done=true, and
    // admit_one resets cancel_after_decoded_tokens=-1 at rebind.
    for (int32_t cs : opts.gate_test.cancel_plan) {
        if (cs >= 0 && static_cast<size_t>(cs) < seqs_.size()) {
            seqs_[static_cast<size_t>(cs)]
                .cancel_after_decoded_tokens = opts.gate_test.cancel_after;
        }
    }
    // Streaming Slice 8: when --stream-all is on, mark every bound
    // active seq as stream_enabled, default-construct its
    // engine-owned full channel handle, and pull a matching
    // receive_channel for main. Main collects the receivers once
    // via take_stream_receivers() BEFORE scheduling run(); the
    // engine publishes events through the channel during the run.
    // Engine is reconstructed per repeat, so this fires once per
    // repeat naturally.
    // M2g: iterate budgets_.size() rather than seqs_.size() so initial
    // idle slots are excluded from the bulk receiver vector. With
    // initial_idle_slots == 0 (every existing gate run) this collapses
    // to seqs_.size() and stream_receivers_ remains byte-identical;
    // with idle slots present, the bulk path is for initial actives
    // only, and an admitted-into-idle request opting into streaming
    // via submit_request(want_stream=true) gets its caller-supplied
    // channel installed by admit_one instead.
    if (stream_all_) {
        stream_receivers_.reserve(budgets_.size());
        for (size_t s = 0; s < budgets_.size(); s++) {
            seqs_[s].stream_enabled = true;
            seqs_[s].stream_channel = token_stream_channel{};
            stream_receivers_.emplace_back(
                seqs_[s].stream_channel);
        }
    }
}

std::vector<hpx::future<request_result>> engine::take_futures() {
    std::vector<hpx::future<request_result>> futs;
    futs.reserve(promises_.size());
    for (auto & p : promises_) {
        futs.emplace_back(p.get_future());
    }
    return futs;
}

std::vector<hpx::future<request_result>> engine::take_admitted_futures() {
    std::lock_guard<hpx::spinlock> lk(admitted_futures_mtx_);
    std::vector<hpx::future<request_result>> out;
    out.swap(admitted_futures_);
    return out;
}

std::vector<admitted_stream_handoff> engine::take_admitted_stream_handoffs() {
    std::lock_guard<hpx::spinlock> lk(admitted_futures_mtx_);
    std::vector<admitted_stream_handoff> out;
    out.swap(admitted_stream_handoffs_);
    return out;
}

std::vector<token_stream_receiver> engine::take_stream_receivers() {
    std::vector<token_stream_receiver> out;
    out.swap(stream_receivers_);
    return out;
}

void engine::submit(arrival_msg msg) {
    const int32_t rid = msg.request_id;
    const int32_t bud = msg.decode_budget;
    // N1: publish a `submission` inbox_msg onto the HPX inbox
    // channel. The engine task pumps the channel at its next
    // predicate-evaluation site (inbox_has_pending or the outer-
    // loop tail). channel<T>::set takes T by value with HPX_MOVE
    // forwarding, supporting move-only inbox_msg (arrival is move-
    // only via hpx::promise + optional<channel>).
    inbox_msg im;
    im.k = inbox_msg_kind::submission;
    im.arrival.emplace(std::move(msg));
    inbox_chan_.set(std::move(im));
    trace::event(
        "request_submitted_external request=%d budget=%d "
        "arrival_source=external",
        rid, bud);
}

// M2f: engine-task-only. N1: pumps the HPX inbox channel non-
// blocking into the staged_* deques, then reports whether any
// staged arrival is now visible. Non-const because the pump
// mutates pending_msg_ and staged_arrivals_. MUST NOT be called
// from foreign/server threads — staged_* deques are unguarded and
// accessed exclusively by run().
bool engine::inbox_has_pending() {
    pump_inbox_nonblocking();
    return !staged_arrivals_.empty();
}

// N1: stage one pumped inbox_msg into the engine-task-only
// staged_* deque or the staged_shutdown_ latch. Engine task only.
void engine::stage_one(inbox_msg m) {
    switch (m.k) {
        case inbox_msg_kind::submission:
            // arrival is guaranteed populated for k=submission;
            // every producer that publishes a submission emplaces
            // the arrival_msg before set().
            staged_arrivals_.push_back(std::move(*m.arrival));
            break;
        case inbox_msg_kind::cancel_rid:
            staged_cancel_rids_.push_back(m.rid);
            break;
        case inbox_msg_kind::cancel_token:
            staged_cancel_tokens_.push_back(m.token);
            break;
        case inbox_msg_kind::shutdown:
            staged_shutdown_ = true;
            break;
    }
}

// N1: drain every channel-buffered message into staged_* without
// blocking. Maintains the invariant that pending_msg_ is valid
// (queued on the channel for the next not-yet-arrived message) at
// function exit. Engine task only.
//
// N2.7 / N2.7b: placement-selected cooperativity. Two operating
// modes, decided at spawn time by the caller (Layer 2) and carried
// into the engine as `cooperative_yield_on_pump_` from
// engine_options::lib.cooperative_yield_on_pump:
//
//   - Default-pool placement (default true). Under small HPX
//     worker-thread counts (especially os_threads=1) the engine
//     task and the producer/cancel tasks share workers; without
//     yielding, a foreign-thread or HPX-task call to
//     engine::cancel_request / engine::submit may not be observed
//     in time and the engine_queued_cancel_smoke pattern can hang
//     (N2.6 evidence). The yield gives the scheduler a chance to
//     dispatch producer work between pump calls.
//
//   - Named engine-pool placement (caller sets false). When the
//     engine runs on a dedicated single-PU HPX pool, yield causes
//     a scheduler livelock (N2.6b / N2.6c evidence). Removing the
//     yield alone is insufficient: pump becomes pure-userspace
//     polling that can lose the race against a foreign-thread
//     inbox_chan_.set(...) before the engine suspends in
//     wait_inbox_blocking() (N2.7b evidence: 1/3 hang without any
//     synchronization point). The engine-pool branch therefore
//     issues one nonblocking pending_msg_.wait_for(0ns) as a
//     future/channel readiness synchronization point. wait_for(0)
//     touches the future's shared state, which propagates any
//     just-published value into is_ready without re-pending the
//     engine task.
//
// Cancellation semantics are unchanged in either mode: inbox
// messages are still staged/pumped only at iteration boundaries
// here and in wait_inbox_blocking(), and cancellation is still
// not observed from inside llama_decode.
void engine::pump_inbox_nonblocking() {
    if (cooperative_yield_on_pump_) {
        hpx::this_thread::yield();
    }
    if (!pending_msg_.valid()) {
        pending_msg_ = inbox_chan_.get();
    }
    if (!cooperative_yield_on_pump_) {
        pending_msg_.wait_for(std::chrono::nanoseconds(0));
    }
    while (pending_msg_.is_ready()) {
        stage_one(pending_msg_.get());
        pending_msg_ = inbox_chan_.get();
    }
}

// N1: HPX-suspend the engine task on the inbox channel until at
// least one new message arrives, stage it, then drain any follow-
// up ready messages. Replaces the M3a condition_variable_any
// idle-wait. Engine task only.
void engine::wait_inbox_blocking() {
    if (!pending_msg_.valid()) {
        pending_msg_ = inbox_chan_.get();
    }
    // pending_msg_.get() parks the engine HPX task on the channel's
    // shared state if no message is ready. HPX-native suspension —
    // not std::condition_variable::wait.
    stage_one(pending_msg_.get());
    pending_msg_ = inbox_chan_.get();
    while (pending_msg_.is_ready()) {
        stage_one(pending_msg_.get());
        pending_msg_ = inbox_chan_.get();
    }
}

submit_handle engine::submit_request(struct submit_request req) {
    // Create the engine-side promise; the matching future is moved
    // into the returned handle so the caller owns it directly.
    hpx::promise<request_result> promise;
    hpx::future<request_result>  fut = promise.get_future();

    // Build the engine-internal arrival_msg. All llama-touching work
    // happens later inside the engine task; this method only moves
    // POD/owned data onto the inbox via the existing submit() path.
    arrival_msg msg;
    msg.request_id    = req.request_id;
    msg.decode_budget = req.decode_budget;
    msg.prompt_tokens = std::move(req.prompt_tokens);
    msg.src           = arrival_source::external;
    msg.promise       = std::move(promise);
    msg.want_stream   = req.want_stream;
    // M6a: carry per-request sampling configuration through the inbox.
    // `sampling_config` is a small POD; a copy here is cheap and keeps
    // `submit_request` reusable by the caller after this call returns.
    // No sampling site reads this field in M6a — both argmax sites
    // remain unchanged.
    msg.sampling      = req.sampling;

    submit_handle h;
    h.result            = std::move(fut);
    h.stream            = std::nullopt;
    h.token.request_id  = req.request_id;
    // M2g: per-request opt-in streaming. Build the HPX channel on the
    // caller's side, hand the receiver to the caller via
    // submit_handle.stream, and move the channel into arrival_msg.
    // drain_external_inbox will stash it in external_stream_channels_
    // keyed by request_id, and admit_one will move it into the bound
    // slot's seq_state::stream_channel. No llama.cpp API call sites
    // here — channel construction is pure HPX-side wiring.
    if (req.want_stream) {
        token_stream_channel  chan;
        token_stream_receiver rx(chan);
        msg.stream_channel.emplace(std::move(chan));
        h.stream.emplace(std::move(rx));
    }

    // N1: bump next_epoch_ AND publish the submission onto the HPX
    // inbox channel under a single hpx::spinlock critical section.
    // Holding the lock across submit() (which calls
    // inbox_chan_.set(...)) preserves the M5a invariant that drain
    // order matches epoch order for concurrent same-rid
    // submissions. live_epoch_by_rid_ semantics are therefore
    // unchanged — the older incarnation is always staged and
    // processed before the newer one. Other producers
    // (cancel_request rid/token, request_shutdown,
    // engine::submit(arrival_msg) direct) do not bump next_epoch_
    // and do not acquire submit_publish_mtx_. Epoch starts at 1
    // (0 is the "unset" sentinel for arrivals that never came
    // through submit_request).
    {
        std::lock_guard<hpx::spinlock> lk(submit_publish_mtx_);
        const uint64_t issued_epoch = next_epoch_++;
        msg.epoch          = issued_epoch;
        h.token.epoch      = issued_epoch;
        submit(std::move(msg));
    }
    return h;
}

// M3a: public shutdown signal for keep-alive mode. N1: publishes a
// `shutdown` inbox_msg onto the HPX inbox channel. Idempotent:
// multiple shutdown messages are folded into the engine-task-only
// staged_shutdown_ latch by stage_one. Hard rule: no llama_* call
// sites; pure HPX-side control-plane wiring. Safe to call from any
// HPX or foreign thread.
void engine::request_shutdown() {
    inbox_msg im;
    im.k = inbox_msg_kind::shutdown;
    inbox_chan_.set(std::move(im));
}

// M3b: public queued-before-admission cancellation. N1: publishes a
// `cancel_rid` inbox_msg onto the HPX inbox channel. The engine
// pumps it into staged_cancel_rids_ at the next predicate-pump
// site (inbox_has_pending or outer-loop tail);
// drain_cancel_inbox() promotes it into cancelled_request_ids_ at
// iteration boundaries (top of each iter AND outer-loop tail
// before idle-wait / shutdown-drain). Idempotent — duplicates are
// deduped by cancelled_request_ids_ (unordered_set) and bump
// cancel_request_duplicates. Hard rule: no llama_* call sites; no
// seq_state lookup; no promise fulfillment. Safe to call from any
// HPX or foreign thread.
void engine::cancel_request(int32_t rid) {
    inbox_msg im;
    im.k   = inbox_msg_kind::cancel_rid;
    im.rid = rid;
    inbox_chan_.set(std::move(im));
}

// M5a: epoch-aware cancellation. N1: publishes a `cancel_token`
// inbox_msg onto the HPX inbox channel. The engine pumps it into
// staged_cancel_tokens_ at the next predicate-pump site;
// drain_cancel_inbox() promotes it into cancelled_tokens_ keyed by
// (rid, epoch). Hard rule: no llama_* call sites; no seq_state
// lookup; no promise fulfillment. Safe to call from any HPX or
// foreign thread.
void engine::cancel_request(const cancel_token & tok) {
    inbox_msg im;
    im.k     = inbox_msg_kind::cancel_token;
    im.token = tok;
    inbox_chan_.set(std::move(im));
}

// M5b: explicit-engine convenience forwarder. Equivalent to
// `eng.cancel_request(this->token);`. Fire-and-forget; final truth
// is observed through `submit_handle::result`. Stores no engine
// observer on the handle — the caller passes a live reference, so
// destructor/lifetime safety reduces to the standard C++ rule "do
// not pass a reference to a dead object". No new cancellation
// semantic, no llama.cpp call sites, no new HPX synchronization
// primitive. Defined here (not inline in types.h) so types.h need
// not depend on the full `engine` definition.
void submit_handle::cancel(engine & eng) const noexcept {
    eng.cancel_request(token);
}

external_release_handle engine::register_external_release_iter(int32_t K) {
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

void engine::run() {
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
    // Streaming Slice 8: drain any open token streams so the caller's
    // chain-walk cannot hang if the decode loop bailed out early.
    // Closes with reason=error — never reported as completed when the
    // engine aborted (see design §3). No-op when --stream-all is OFF
    // or when finalize_and_fulfill / cancel_and_fulfill already
    // closed the stream.
    for (size_t s = 0; s < seqs_.size(); s++) {
        if (!seqs_[s].stream_enabled) continue;
        if (seqs_[s].stream_closed)   continue;
        close_stream(seqs_[s], stream_close_reason::error);
    }
    // M2g: close any per-request stream channels submitted via
    // submit_request(want_stream=true) that the engine never bound to
    // a slot (e.g., engine bailed before admission, or admission
    // failed). The caller is holding the matching receiver via
    // submit_handle.stream and would otherwise hang on .get(). Set a
    // closed{reason=error} event then close the channel; bump
    // streams_closed_error so the counter reflects orphan closures
    // symmetrically with the bound-but-unfinished path above.
    for (auto & kv : external_stream_channels_) {
        try {
            kv.second.set(token_stream_event{
                stream_event_kind::closed, 0,
                stream_close_reason::error});
        } catch (...) {
            // Already closed; fall through to close().
        }
        try {
            kv.second.close();
        } catch (...) {
            // Already closed.
        }
        result_.streams_closed_error++;
        trace::event(
            "token_stream_closed request=%d seq_id=-1 "
            "n_tokens=0 reason=error",
            kv.first);
    }
    external_stream_channels_.clear();
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
    // Phase 1 diagnostics dump. Fires exactly once per engine::run()
    // tail when LLAMA_HPX_DIAG_METRICS=1. Writes to
    // LLAMA_HPX_DIAG_METRICS_PATH if set (O_APPEND, line-buffered),
    // stderr otherwise. Zero-cost when the env switch is unset.
    if (diag_enabled()) {
        dump_metrics_jsonl(result_);
    }
}

bool engine::clear_and_check(seq_state & seq, int32_t iter,
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

void engine::publish_token(seq_state & seq, int32_t token_id) {
    // Phase 1: count every token emission attributed to this iter,
    // regardless of streaming status. Bumped BEFORE the streaming
    // early-return so non-streaming requests are also counted. Guarded
    // by !empty so preloaded-prefill calls (which fire before
    // iter_run_decode pushes the first row) do not underflow.
    if (diag_enabled()
     && !result_.metrics.tokens_emitted_per_iter.empty()) {
        result_.metrics.tokens_emitted_per_iter.back()++;
    }
    if (!seq.stream_enabled || seq.stream_closed) return;
    // N3.1 (Exp 13): stamp the first non-terminal token publish on
    // this seq when the option is on. Guarded by the seq's prior
    // -1 sentinel so each request's first publish is captured
    // exactly once. Cost is one steady_clock::now() at most once
    // per request. With the option off, zero added work.
    if (enable_responsiveness_timing_ && seq.t_first_publish_us == -1) {
        seq.t_first_publish_us = now_us();
    }
    try {
        seq.stream_channel.set(
            token_stream_event{stream_event_kind::token, token_id,
                               stream_close_reason::completed});
    } catch (...) {
        // Channel closed unexpectedly (single-producer discipline
        // guarantees this does not happen); drop the publish.
        return;
    }
    seq.stream_tokens_emitted++;
    result_.stream_tokens_emitted_total++;
    trace::event(
        "token_stream_token request=%d seq_id=%d pos=%d token=%d",
        seq.request_id, seq.seq_id, seq.stream_tokens_emitted - 1,
        token_id);
}

void engine::close_stream(seq_state & seq, stream_close_reason reason) {
    if (!seq.stream_enabled || seq.stream_closed) return;
    try {
        seq.stream_channel.set(
            token_stream_event{stream_event_kind::closed, 0, reason});
    } catch (...) {
        // Already closed; fall through to bookkeeping.
    }
    try {
        seq.stream_channel.close();
    } catch (...) {
        // Already closed (defensive). Counters and trace fire regardless.
    }
    seq.stream_closed = true;
    switch (reason) {
        case stream_close_reason::completed:
            result_.streams_closed_completed++;
            break;
        case stream_close_reason::cancelled:
            result_.streams_closed_cancelled++;
            break;
        case stream_close_reason::error:
            result_.streams_closed_error++;
            break;
    }
    trace::event(
        "token_stream_closed request=%d seq_id=%d n_tokens=%d "
        "reason=%s",
        seq.request_id, seq.seq_id, seq.stream_tokens_emitted,
        stream_close_reason_name(reason));
}

bool engine::fulfill_promise(seq_state &     seq,
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
    // N3.1 (Exp 13): propagate the responsiveness timestamps from
    // the engine-internal seq_state carriers onto the public
    // request_result snapshot. All four are -1 unless the engine
    // was constructed with
    // engine_options::lib.enable_responsiveness_timing == true and
    // the corresponding write site fired for this request.
    rr.t_admitted_us        = seq.t_admitted_us;
    rr.t_first_publish_us   = seq.t_first_publish_us;
    rr.t_complete_us        = seq.t_complete_us;
    rr.t_cancel_observed_us = seq.t_cancel_observed_us;

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

bool engine::finalize_and_fulfill(seq_state & seq, int32_t iter,
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

    // Streaming Slice 8: close the per-seq token stream with
    // reason=completed before fulfilling the request_result promise.
    // No-op when streaming is disabled.
    close_stream(seq, stream_close_reason::completed);

    // Live Admission Slice 5: demand-gated push onto the completion-
    // freed pool. Push only when --reuse-completed is on AND the
    // waiting queue still has unadmitted entries. Once waiters are
    // drained, later natural completions stop pooling so the
    // residual size remains a tight invariant for the first-wave
    // proof. Push happens after KV-clear succeeds so the slot is
    // truly KV-empty at queue-entry time.
    if (reuse_completed_ && !waiting_queue_consumable_.empty()) {
        free_due_to_completion_.push_back(seq.seq_id);
    } else if (seq.seq_id >= static_cast<int32_t>(budgets_.size())) {
        // N5a: idle-origin slots occupy seq_id range
        // [budgets_.size(), seqs_.size()); after a natural completion
        // that is NOT routed to free_due_to_completion_, return them to
        // free_idle_. Keying on the seq_id range rather than
        // admission_src also recovers a slot admitted via cancel_freed
        // (idle slot -> active-cancel -> cancel_freed -> natural
        // completion), which the old initial_idle-only predicate leaked.
        // This relies on today's seq_id == slot index invariant. Gates
        // use initial_idle_slots=0 (budgets_.size()==seqs_.size()) so
        // this branch stays inert for canonical gate shapes.
        free_idle_.push_back(seq.seq_id);
    }

    const auto    now    = std::chrono::steady_clock::now();
    const int64_t ttc_us = std::chrono::duration_cast<
        std::chrono::microseconds>(now - t_start_).count();
    // N3.1 (Exp 13): stamp completion time on the seq just before
    // fulfill_promise copies it to the request_result. Use the
    // same `now` instant as ttc_us above, just converted into the
    // absolute time-since-epoch domain that producers also use.
    if (enable_responsiveness_timing_) {
        seq.t_complete_us = std::chrono::duration_cast<
            std::chrono::microseconds>(now.time_since_epoch()).count();
    }

    if (!fulfill_promise(seq, request_status::completed, ttc_us)) {
        return false;
    }

    // M5a: guarded erase of live_epoch_by_rid_. Only erases if the
    // map entry still points at this completion's (rid, epoch) — a
    // newer same-rid submission would have overwritten the entry to
    // a different epoch, in which case we must NOT wipe the live
    // pointer to that newer incarnation.
    if (seq.epoch != 0) {
        auto live_it = live_epoch_by_rid_.find(seq.request_id);
        if (live_it != live_epoch_by_rid_.end()
         && live_it->second == seq.epoch) {
            live_epoch_by_rid_.erase(live_it);
        }
    }

    // M6b: release the per-seq sampler chain (if any) once this
    // request's owning slot is no longer producing tokens. unique_ptr
    // deleter runs `llama_sampler_free`, which also frees any sub-
    // samplers added to the chain. A no-op on greedy admissions
    // (sampler_chain stayed null) so the greedy path remains byte-
    // identical. Engine-task-only — finalize_and_fulfill is reached
    // exclusively from the engine task.
    seq.sampler_chain.reset();

    trace::event("promise_fulfilled seq=%d ttc_us=%lld",
                 seq.seq_id, static_cast<long long>(ttc_us));
    // Phase 1: bump the current iter's completed counter. Guarded by
    // !empty so preloaded-prefill calls (which fire before iter_run_decode
    // pushes the first row) do not underflow.
    if (diag_enabled()
     && !result_.metrics.completed_per_iter.empty()) {
        result_.metrics.completed_per_iter.back()++;
    }
    return true;
}

bool engine::cancel_should_observe(const seq_state & seq) const noexcept {
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

bool engine::cancel_and_fulfill(seq_state & seq, int32_t iter,
                                llama_memory_t mem) {
    seq.cancel_observed       = true;
    seq.cancel_observed_iter  = iter;
    seq.n_decoded_at_cancel   = seq.n_decoded;

    trace::event(
        "cancel_observed seq=%d iter=%d n_decoded=%d",
        seq.seq_id, iter, seq.n_decoded_at_cancel);

    if (!clear_and_check(seq, iter, mem)) return false;

    // Streaming Slice 8: defensive close with reason=cancelled. Not
    // exercised in the Slice 8 smoke (--cancel-plan none) but the
    // wiring must exist so a future mixed shape does not leak a
    // dangling tail. No-op when streaming is disabled.
    close_stream(seq, stream_close_reason::cancelled);

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
    // N3.1 (Exp 13): stamp the cancel-observed instant on the seq
    // before fulfill_promise copies it onto the request_result.
    // Engine task only; iteration-boundary observation per design.
    if (enable_responsiveness_timing_) {
        seq.t_cancel_observed_us = std::chrono::duration_cast<
            std::chrono::microseconds>(now.time_since_epoch()).count();
    }

    if (!fulfill_promise(seq, request_status::cancelled, ttc_us)) {
        return false;
    }
    result_.cancelled_count++;

    // M5a: guarded erase of live_epoch_by_rid_, symmetric with
    // finalize_and_fulfill. A late same-rid resubmission's live
    // pointer is preserved if the epoch no longer matches.
    if (seq.epoch != 0) {
        auto live_it = live_epoch_by_rid_.find(seq.request_id);
        if (live_it != live_epoch_by_rid_.end()
         && live_it->second == seq.epoch) {
            live_epoch_by_rid_.erase(live_it);
        }
    }

    // M6b: release the per-seq sampler chain on the cancellation
    // path, symmetric with finalize_and_fulfill. No-op on greedy
    // admissions. Engine-task-only — cancel_and_fulfill is reached
    // exclusively from the engine task.
    seq.sampler_chain.reset();

    trace::event(
        "cancel_future_fulfilled seq=%d status=cancelled ttc_us=%lld",
        seq.seq_id, static_cast<long long>(ttc_us));
    if (diag_enabled()) result_.metrics.cur_iter_diag.cancelled++;
    return true;
}

void engine::drain_external_inbox(int32_t iter) {
    // N1: swap from the engine-task-only staged_arrivals_ deque
    // (already populated by pump_inbox_nonblocking() at the
    // predicate site). No channel pump here — cancels arriving on
    // the channel mid-drain stay buffered and are observed only at
    // the next cancel phase. No lock — engine task is the sole
    // accessor of staged_arrivals_.
    std::deque<arrival_msg> drained;
    drained.swap(staged_arrivals_);
    while (!drained.empty()) {
        arrival_msg msg = std::move(drained.front());
        drained.pop_front();
        const int32_t rid = msg.request_id;
        const int32_t bud = msg.decode_budget;
        // M5a: token-aware cancel-before-submit-drain shortcut.
        // Resolved BEFORE the legacy rid-only path because the token
        // path requires an exact (rid, epoch) match — a stale token
        // (rid known, epoch mismatch) MUST NOT short-circuit the new
        // incarnation's arrival. The token-aware short-circuit only
        // fires when the (rid, epoch) pair the caller used at
        // submit_request time was cancelled before the drain reached
        // it. Engine task only; no llama API.
        const uint64_t arrival_epoch = msg.epoch;
        if (arrival_epoch != 0) {
            const auto tok_it = cancelled_tokens_.find(
                std::make_pair(rid, arrival_epoch));
            if (tok_it != cancelled_tokens_.end()) {
                cancelled_tokens_.erase(tok_it);
                result_.arrival_drained_count++;
                if (result_.first_external_drain_iter == -1) {
                    result_.first_external_drain_iter = iter;
                }
                trace::event(
                    "arrival_drained request=%d iter=%d budget=%d",
                    rid, iter, bud);
                fulfill_queued_cancelled(
                    rid, bud,
                    std::move(msg.promise),
                    std::move(msg.stream_channel),
                    "inbox_token");
                continue;
            }
        }
        // M3b: cancel-before-submit-drain shortcut. If a prior
        // cancel_request(rid) had already populated
        // cancelled_request_ids_, resolve this arrival as queued-
        // cancelled immediately — do not push to waiting queue, do
        // not stash promise/channel into the external_* maps. The
        // arrival_drained_count IS still bumped so the engine-side
        // cross-check stays accurate: every arrival_msg that left
        // the inbox is accounted for. The request_queued trace is
        // suppressed because the request never enters the waiting
        // queue. Engine task only.
        if (cancelled_request_ids_.erase(rid) > 0) {
            result_.arrival_drained_count++;
            if (result_.first_external_drain_iter == -1) {
                result_.first_external_drain_iter = iter;
            }
            trace::event(
                "arrival_drained request=%d iter=%d budget=%d",
                rid, iter, bud);
            fulfill_queued_cancelled(
                rid, bud,
                std::move(msg.promise),
                std::move(msg.stream_channel),
                "inbox");
            continue;
        }
        external_promises_.emplace(rid, std::move(msg.promise));
        // M2g: stash any caller-supplied stream channel for this
        // request. admit_one's external branch checks the map at
        // bind time and moves the channel into the slot. Always
        // engine-task-only — no synchronization needed.
        if (msg.stream_channel.has_value()) {
            external_stream_channels_.emplace(
                rid, std::move(*msg.stream_channel));
        }
        // M5a: register the live (rid, epoch) pair if the arrival
        // came through submit_request (epoch != 0). The gate's
        // direct engine::submit(arrival_msg) path leaves epoch at 0
        // and skips this map entirely, so existing gate shapes are
        // unaffected. Note: if a prior arrival for the same rid is
        // still live (e.g. two same-rid submit_requests queued in
        // quick succession), this overwrites the live entry to the
        // newer epoch. Guarded erases at finalize/cancel sites
        // protect against the older completion wiping the newer
        // entry.
        if (arrival_epoch != 0) {
            live_epoch_by_rid_[rid] = arrival_epoch;
        }
        waiting_request w;
        w.request_id    = rid;
        w.decode_budget = bud;
        w.src           = arrival_source::external;
        w.epoch         = arrival_epoch;
        // M1c: preserve the per-arrival prompt by moving it onto
        // the new waiting_request. admit_one will move it again
        // into the bound slot's seq_state::prompt_tokens.
        w.prompt_tokens = std::move(msg.prompt_tokens);
        // M6a: same plumbing for sampling_config. `sampling_config` is
        // a small POD with no owning members, but the move expression
        // is kept symmetric with `prompt_tokens` so the surface stays
        // ready for M6b, when a non-trivially-movable `llama_sampler_ptr`
        // may eventually ride alongside or replace this carrier.
        w.sampling      = std::move(msg.sampling);
        waiting_queue_consumable_.push_back(std::move(w));
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

// M3b + M5a: transfer of staged cancels -> cancelled_request_ids_
// and cancelled_tokens_. Engine task only. N1: swaps from the
// engine-task-only staged_cancel_rids_ and staged_cancel_tokens_
// deques (already populated by pump_inbox_nonblocking() at the
// predicate site). No channel pump here — cancels arriving on the
// channel mid-drain stay buffered and are observed only at the
// next cancel phase. No lock. Each drained entry bumps
// cancel_request_calls (counter is shared between rid-only and
// token-aware paths since it measures drained cancel work).
// Duplicates on either side bump cancel_request_duplicates — the
// rid-only path keys by rid, the token-aware path keys by
// (rid, epoch).
void engine::drain_cancel_inbox() {
    std::deque<int32_t>      drained_rids;
    std::deque<cancel_token> drained_tokens;
    drained_rids.swap(staged_cancel_rids_);
    drained_tokens.swap(staged_cancel_tokens_);
    while (!drained_rids.empty()) {
        const int32_t rid = drained_rids.front();
        drained_rids.pop_front();
        result_.cancel_request_calls++;
        const auto ins = cancelled_request_ids_.insert(rid);
        if (!ins.second) {
            result_.cancel_request_duplicates++;
            trace::event(
                "cancel_request_duplicate request=%d", rid);
            continue;
        }
        trace::event("cancel_drained request=%d", rid);
    }
    while (!drained_tokens.empty()) {
        const cancel_token tok = drained_tokens.front();
        drained_tokens.pop_front();
        result_.cancel_request_calls++;
        const auto ins = cancelled_tokens_.insert(
            std::make_pair(tok.request_id, tok.epoch));
        if (!ins.second) {
            result_.cancel_request_duplicates++;
            trace::event(
                "cancel_request_duplicate_token request=%d epoch=%llu",
                tok.request_id,
                static_cast<unsigned long long>(tok.epoch));
            continue;
        }
        trace::event(
            "cancel_drained_token request=%d epoch=%llu",
            tok.request_id,
            static_cast<unsigned long long>(tok.epoch));
    }
}

// M3b: queued-cancellation resolver. Engine task only. No KV touch,
// no llama API. Builds a request_result with status=cancelled and
// the default "never decoded" sentinels (n_decoded=0, hash=
// k_token_hash_empty, admitted_at_iter=-1, kv_cleared=false), closes
// the optional stream with reason=cancelled (incrementing
// streams_opened so the existing streams_opened == sum(closed_*)
// identity holds), and fulfills the moved-in promise exactly once.
bool engine::fulfill_queued_cancelled(
    int32_t                             request_id,
    int32_t                             decode_budget,
    hpx::promise<request_result>        promise,
    std::optional<token_stream_channel> channel,
    const char *                        from_label) {
    // Stream half: send a terminal closed event with reason=cancelled
    // then close the channel. set/close are best-effort — the close
    // path matches engine::close_stream(...)'s try/catch discipline
    // so a defensive double-close is silent. The single-producer
    // (engine task) discipline guarantees no race here.
    if (channel.has_value()) {
        try {
            channel->set(token_stream_event{
                stream_event_kind::closed, 0,
                stream_close_reason::cancelled});
        } catch (...) {
            // Receiver may have dropped its half; nothing to do.
        }
        try {
            channel->close();
        } catch (...) {
            // Already closed; bookkeeping below fires regardless.
        }
        result_.streams_opened++;
        result_.streams_closed_cancelled++;
        trace::event(
            "token_stream_opened request=%d seq_id=-1", request_id);
        trace::event(
            "token_stream_closed request=%d seq_id=-1 n_tokens=0 "
            "reason=cancelled", request_id);
    }

    const auto    now    = std::chrono::steady_clock::now();
    const int64_t ttc_us = std::chrono::duration_cast<
        std::chrono::microseconds>(now - t_start_).count();

    request_result rr;
    rr.request_id          = request_id;
    rr.seq_id              = -1;
    rr.decode_budget       = decode_budget;
    rr.n_decoded           = 0;
    rr.hash                = token_hash::k_token_hash_empty;
    rr.done_iter           = -1;
    rr.pos_max_at_clear    = -1;
    rr.kv_cleared          = false;
    rr.ttc_us              = ttc_us;
    rr.generated_tokens.clear();
    rr.status              = request_status::cancelled;
    rr.cancel_observed_iter = -1;
    rr.n_decoded_at_cancel  = 0;
    rr.admitted_at_iter    = -1;
    rr.reused_seq_id       = -1;
    rr.previous_request_id = -1;
    rr.admission_src       = admission_source::none;
    rr.arrival_src         = arrival_source::external;
    // N3.1 (Exp 13): on the queued-cancel path the request never
    // bound to a seq, so t_admitted_us / t_first_publish_us /
    // t_complete_us stay at their -1 defaults (correct semantics:
    // "not admitted; never published; not completed"). Only
    // t_cancel_observed_us is meaningful here — stamped at the
    // instant the engine fulfills the queued-cancel promise.
    if (enable_responsiveness_timing_) {
        rr.t_cancel_observed_us = now_us();
    }

    try {
        promise.set_value(std::move(rr));
    } catch (const std::exception & e) {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "fulfill_queued_cancelled.set_value threw for "
            "request %d: %s", request_id, e.what());
        if (result_.error.empty()) result_.error = buf;
        return false;
    }
    result_.queued_cancelled++;
    result_.promises_fulfilled++;
    trace::event(
        "queued_cancelled request=%d from=%s ttc_us=%lld",
        request_id, from_label, static_cast<long long>(ttc_us));
    return true;
}

// N5b: shutdown-aborted queued-request resolver. Engine task only. No
// KV touch, no llama API. A request drained into
// waiting_queue_consumable_ but never admitted is resolved on shutdown
// with status=failed_reserved — shutdown-aborted queued work, NOT user
// cancellation (so it does NOT bump queued_cancelled and stays
// distinguishable from the M3b queued-cancel path). The "never decoded /
// never admitted" sentinels match the queued-cancel result (n_decoded=0,
// hash=empty, admitted_at_iter=-1, kv_cleared=false). A queued stream,
// if present, is closed with reason=error: the request neither completed
// nor was user-cancelled, so among the existing stream_close_reason
// values `error` is the least-misleading (a `completed` close would be a
// lie; `cancelled` would imply user intent). streams_opened/streams_
// closed_error are bumped symmetrically so streams_opened == sum(closed_*)
// still holds. Fulfills the moved-in promise exactly once.
bool engine::fulfill_queued_shutdown_aborted(
    int32_t                             request_id,
    int32_t                             decode_budget,
    hpx::promise<request_result>        promise,
    std::optional<token_stream_channel> channel,
    const char *                        from_label) {
    if (channel.has_value()) {
        try {
            channel->set(token_stream_event{
                stream_event_kind::closed, 0,
                stream_close_reason::error});
        } catch (...) {
            // Receiver may have dropped its half; nothing to do.
        }
        try {
            channel->close();
        } catch (...) {
            // Already closed; bookkeeping below fires regardless.
        }
        result_.streams_opened++;
        result_.streams_closed_error++;
        trace::event(
            "token_stream_opened request=%d seq_id=-1", request_id);
        trace::event(
            "token_stream_closed request=%d seq_id=-1 n_tokens=0 "
            "reason=error", request_id);
    }

    const auto    now    = std::chrono::steady_clock::now();
    const int64_t ttc_us = std::chrono::duration_cast<
        std::chrono::microseconds>(now - t_start_).count();

    request_result rr;
    rr.request_id          = request_id;
    rr.seq_id              = -1;
    rr.decode_budget       = decode_budget;
    rr.n_decoded           = 0;
    rr.hash                = token_hash::k_token_hash_empty;
    rr.done_iter           = -1;
    rr.pos_max_at_clear    = -1;
    rr.kv_cleared          = false;
    rr.ttc_us              = ttc_us;
    rr.generated_tokens.clear();
    rr.status              = request_status::failed_reserved;
    rr.cancel_observed_iter = -1;
    rr.n_decoded_at_cancel  = -1;
    rr.admitted_at_iter    = -1;
    rr.reused_seq_id       = -1;
    rr.previous_request_id = -1;
    rr.admission_src       = admission_source::none;
    rr.arrival_src         = arrival_source::external;

    try {
        promise.set_value(std::move(rr));
    } catch (const std::exception & e) {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "fulfill_queued_shutdown_aborted.set_value threw for "
            "request %d: %s", request_id, e.what());
        if (result_.error.empty()) result_.error = buf;
        return false;
    }
    result_.promises_fulfilled++;
    trace::event(
        "queued_shutdown_aborted request=%d from=%s ttc_us=%lld",
        request_id, from_label, static_cast<long long>(ttc_us));
    return true;
}

// N5b: drain ALL queued-but-unadmittable requests on shutdown. Engine
// task only. For every entry still in waiting_queue_consumable_, move its
// promise out of external_promises_ and any stream channel out of
// external_stream_channels_ (both rid-keyed, populated by
// drain_external_inbox), erase its live_epoch_by_rid_ entry, and resolve
// it via fulfill_queued_shutdown_aborted. Leaves waiting_queue_consumable_
// empty and the corresponding external_promises_ / external_stream_
// channels_ / live_epoch_by_rid_ entries erased, so the run-end
// "external_promises_ must be empty" guard holds. Preloaded gate waiters
// carry no external promise and are simply dropped (nothing is owed to
// them on this path; gates never reach shutdown with unadmitted preloaded
// waiters).
void engine::drain_waiting_queue_for_shutdown() {
    while (!waiting_queue_consumable_.empty()) {
        waiting_request w =
            std::move(waiting_queue_consumable_.front());
        waiting_queue_consumable_.pop_front();
        const int32_t rid = w.request_id;

        auto it_pr = external_promises_.find(rid);
        if (it_pr == external_promises_.end()) {
            // Preloaded waiter: no external promise to fulfill.
            live_epoch_by_rid_.erase(rid);
            continue;
        }
        hpx::promise<request_result> promise = std::move(it_pr->second);
        external_promises_.erase(it_pr);

        std::optional<token_stream_channel> channel;
        auto it_ch = external_stream_channels_.find(rid);
        if (it_ch != external_stream_channels_.end()) {
            channel.emplace(std::move(it_ch->second));
            external_stream_channels_.erase(it_ch);
        }

        live_epoch_by_rid_.erase(rid);

        fulfill_queued_shutdown_aborted(
            rid, w.decode_budget, std::move(promise),
            std::move(channel), "shutdown");
    }
}

// M3b/M3c: walk cancelled_request_ids_ and resolve each rid against
// the queued/active populations. Match in waiting_queue_consumable_:
// pull promise + channel from the external_* maps, fulfill cancelled
// and erase rid from set (M3b path). Match in live non-done active
// seqs: set seq.cancel_requested under the engine task, bump
// cancel_active_observed, emit trace, erase rid from set (M3c path —
// the existing cancel_should_observe → cancel_and_fulfill iter-
// boundary pipeline handles KV clear, stream close, freed-slot
// bookkeeping, and promise fulfillment). No match: leave rid in the
// set so a later arrival can still be resolved as queued
// cancellation. Engine task only. No llama API; no KV mutation.
void engine::apply_queued_cancellations() {
    // ---- M5a: token-aware resolver (runs BEFORE the legacy rid-only
    //      pass so a (rid, epoch) match is always preferred over a
    //      rid-only match for the same rid). Walks cancelled_tokens_
    //      and for each (rid, epoch):
    //        - rid not currently live in live_epoch_by_rid_: keep the
    //          entry in the set for a potential later arrival to
    //          short-circuit at drain_external_inbox (symmetric with
    //          the legacy "cancel-before-submit" semantic). Any
    //          residue at engine end bumps cancel_unknown_request_id.
    //        - rid live, epoch MISMATCH: this is the M5a stale-token
    //          path. Bump cancel_stale_epoch and erase the token from
    //          the set. The live request is left untouched; the
    //          caller's stale token has no power over the new
    //          incarnation.
    //        - rid live, epoch MATCH: route to the M3 mechanics —
    //          waiting_queue_consumable_ first (M3b queued cancel),
    //          then live non-done active seqs (M3c active cancel).
    //          Promise/stream fulfillment, KV clear, and seq freed-
    //          slot bookkeeping stay on the engine task at the
    //          existing iter boundaries.
    if (!cancelled_tokens_.empty()) {
        std::vector<std::pair<int32_t, uint64_t>> tok_to_erase;
        tok_to_erase.reserve(cancelled_tokens_.size());
        for (const auto & ke : cancelled_tokens_) {
            const int32_t  rid_t   = ke.first;
            const uint64_t epoch_t = ke.second;
            const auto live_it = live_epoch_by_rid_.find(rid_t);
            if (live_it == live_epoch_by_rid_.end()) {
                // Not currently live — keep the entry for a future
                // arrival (cancel-before-drain) to match against.
                continue;
            }
            if (live_it->second != epoch_t) {
                result_.cancel_stale_epoch++;
                trace::event(
                    "cancel_stale_epoch request=%d "
                    "token_epoch=%llu live_epoch=%llu",
                    rid_t,
                    static_cast<unsigned long long>(epoch_t),
                    static_cast<unsigned long long>(live_it->second));
                tok_to_erase.push_back(ke);
                continue;
            }
            // (a) waiting_queue_consumable_ match.
            auto it_q = std::find_if(
                waiting_queue_consumable_.begin(),
                waiting_queue_consumable_.end(),
                [rid_t](const waiting_request & w) {
                    return w.request_id == rid_t;
                });
            if (it_q != waiting_queue_consumable_.end()) {
                const int32_t  bud_q     = it_q->decode_budget;
                const uint64_t waiter_ep = it_q->epoch;
                waiting_queue_consumable_.erase(it_q);
                auto it_pr = external_promises_.find(rid_t);
                if (it_pr == external_promises_.end()) {
                    char buf[200];
                    std::snprintf(buf, sizeof(buf),
                        "apply_queued_cancellations(token): "
                        "request %d missing in external_promises_",
                        rid_t);
                    if (result_.error.empty()) result_.error = buf;
                    tok_to_erase.push_back(ke);
                    continue;
                }
                hpx::promise<request_result> pr =
                    std::move(it_pr->second);
                external_promises_.erase(it_pr);
                std::optional<token_stream_channel> chan;
                auto it_ch = external_stream_channels_.find(rid_t);
                if (it_ch != external_stream_channels_.end()) {
                    chan.emplace(std::move(it_ch->second));
                    external_stream_channels_.erase(it_ch);
                }
                fulfill_queued_cancelled(
                    rid_t, bud_q, std::move(pr), std::move(chan),
                    "waiting_queue_token");
                // Guarded erase against live_epoch_by_rid_.
                auto live_erase_it = live_epoch_by_rid_.find(rid_t);
                if (live_erase_it != live_epoch_by_rid_.end()
                 && live_erase_it->second == waiter_ep) {
                    live_epoch_by_rid_.erase(live_erase_it);
                }
                tok_to_erase.push_back(ke);
                continue;
            }
            // (b) Active seq match.
            seq_state * active_match = nullptr;
            for (seq_state & s : seqs_) {
                if (!s.done && s.request_id == rid_t) {
                    active_match = &s;
                    break;
                }
            }
            if (active_match != nullptr) {
                active_match->cancel_requested.store(
                    true, std::memory_order_release);
                result_.cancel_active_observed++;
                trace::event(
                    "cancel_active_observed_token "
                    "request=%d seq_id=%d epoch=%llu",
                    rid_t, active_match->seq_id,
                    static_cast<unsigned long long>(epoch_t));
                tok_to_erase.push_back(ke);
                continue;
            }
            // (c) live[rid]==epoch but no waiter / no active match.
            // Should not happen — live is only populated at drain
            // and erased at finalize/cancel-fulfill. Leave in set;
            // engine-end residue resolution accounts for it.
        }
        for (const auto & ke : tok_to_erase) {
            cancelled_tokens_.erase(ke);
        }
    }

    if (cancelled_request_ids_.empty()) return;
    std::vector<int32_t> to_erase;
    to_erase.reserve(cancelled_request_ids_.size());
    for (int32_t rid : cancelled_request_ids_) {
        // (a) Search waiting_queue_consumable_ for matching rid.
        auto it_q = std::find_if(
            waiting_queue_consumable_.begin(),
            waiting_queue_consumable_.end(),
            [rid](const waiting_request & w) {
                return w.request_id == rid;
            });
        if (it_q != waiting_queue_consumable_.end()) {
            const int32_t bud = it_q->decode_budget;
            waiting_queue_consumable_.erase(it_q);
            auto it_pr = external_promises_.find(rid);
            if (it_pr == external_promises_.end()) {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                    "apply_queued_cancellations: request %d "
                    "missing in external_promises_", rid);
                if (result_.error.empty()) result_.error = buf;
                to_erase.push_back(rid);
                continue;
            }
            hpx::promise<request_result> pr = std::move(it_pr->second);
            external_promises_.erase(it_pr);
            std::optional<token_stream_channel> chan;
            auto it_ch = external_stream_channels_.find(rid);
            if (it_ch != external_stream_channels_.end()) {
                chan.emplace(std::move(it_ch->second));
                external_stream_channels_.erase(it_ch);
            }
            fulfill_queued_cancelled(
                rid, bud, std::move(pr), std::move(chan),
                "waiting_queue");
            to_erase.push_back(rid);
            continue;
        }
        // (b) M3c: active-request cancellation. Find the live
        // non-done seq carrying this request_id and set its
        // cancel_requested atomic from the engine task. The normal
        // iter-boundary cancellation observation pass
        // (cancel_should_observe → cancel_and_fulfill) runs later
        // in this iter (or at the top of a subsequent iter when the
        // match happens at the outer-loop tail) and handles the
        // KV clear, stream close, freed-slot bookkeeping, and
        // promise fulfillment. No llama API call site here; no KV
        // touch; no batch mutation. Engine-task-only mutator on
        // seq_state::cancel_requested.
        seq_state * active_match = nullptr;
        for (seq_state & s : seqs_) {
            if (!s.done && s.request_id == rid) {
                active_match = &s;
                break;
            }
        }
        if (active_match != nullptr) {
            active_match->cancel_requested.store(
                true, std::memory_order_release);
            result_.cancel_active_observed++;
            trace::event(
                "cancel_active_observed request=%d seq_id=%d",
                rid, active_match->seq_id);
            to_erase.push_back(rid);
            continue;
        }
        // (c) No match this iter — keep rid in the set for a
        // future cancel-before-submit arrival to match against.
    }
    for (int32_t rid : to_erase) {
        cancelled_request_ids_.erase(rid);
    }
}

// Bind a freshly freed seq_id to the FIFO head of
// waiting_queue_consumable_ and emit the admission trace events
// (payloads carry an explicit admission_source=<value> key so
// cancel_freed and completion_freed admissions can be distinguished
// without adding new event names). Returns true on success; on the
// KV-not-empty failure path it stamps result_.error and the caller
// frees the batch + returns.
bool engine::admit_one(int32_t              reuse_seq,
                       admission_source     src,
                       const char *         src_label,
                       int32_t              iter,
                       llama_memory_t       mem) {
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

    // M1c: move the waiter out so its prompt_tokens vector can be
    // moved into the bound slot below without an extra copy.
    waiting_request w =
        std::move(waiting_queue_consumable_.front());
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
    // N3.1 (Exp 13): always reset the responsiveness stamps on
    // every rebind so a prior owner of this slot cannot leak its
    // timestamps into the new owner's request_result. Then stamp
    // t_admitted_us when the option is on. With the option off,
    // all four stay -1, byte-identical to pre-N3.1 behavior.
    rseq.t_admitted_us          = -1;
    rseq.t_first_publish_us     = -1;
    rseq.t_complete_us          = -1;
    rseq.t_cancel_observed_us   = -1;
    if (enable_responsiveness_timing_) {
        rseq.t_admitted_us = now_us();
    }
    // M5a: carry the waiter's engine-issued epoch into the bound seq.
    // Used by guarded erase against live_epoch_by_rid_ at
    // finalize/cancel sites. Stays 0 for preloaded waiters and idle
    // slots that never went through submit_request.
    rseq.epoch                  = w.epoch;
    rseq.decode_budget          = w.decode_budget;
    rseq.n_decoded              = 0;
    rseq.pos_next               = 0;
    // Slice B: a reused slot starts a fresh prefill. The admitted-prefill
    // pass in iter_build_batch sets these to (size, true) in the admission
    // iter, before sampling, so behavior is unchanged.
    rseq.prefill_cursor         = 0;
    rseq.prefill_complete       = false;
    rseq.i_batch                = -1;
    rseq.last_token             = 0;
    rseq.done                   = false;
    rseq.done_iter              = -1;
    rseq.pos_max_at_clear       = -1;
    rseq.kv_cleared             = false;
    rseq.promise_fulfilled      = false;
    rseq.hash_state             = k_token_hash_init;
    rseq.generated_tokens.clear();
    // M1c: hand the waiter's per-request prompt vector to the
    // engine's per-seq slot. The admitted-prefill loop in
    // run_body() now reads this owned copy. In M1c every waiter's
    // prompt is still a copy of the shared prompt source, so the
    // resulting prefill rows are byte-identical to M1b / M0.
    rseq.prompt_tokens          = std::move(w.prompt_tokens);
    // M6a: move the waiter's sampling_config into the bound slot.
    // Symmetric with the prompt_tokens hand-off; both are per-request
    // inputs fixed at admission.
    rseq.sampling               = std::move(w.sampling);

    // M6b/M6c: per-seq sampler chain (re)binding for the new owner.
    //
    // Step 1 — unconditionally drop any chain left over from the
    // prior owner of this slot. unique_ptr deleter runs
    // `llama_sampler_free` (no-op on an already-null chain). This is
    // belt-and-suspenders alongside the resets in
    // finalize_and_fulfill / cancel_and_fulfill; together they ensure
    // there is no path that re-uses a stale chain across admissions.
    //
    // Step 2 — when the new owner requested stochastic sampling,
    // build a fresh chain via the TU-local `build_sampler_chain`
    // helper. M6c wires top_k / top_p / top_p_min_keep / temperature
    // alongside dist(seed); a default-knob stochastic config produces
    // a dist(seed)-only chain (byte-equivalent to the M6b shape).
    // Validation failure or chain_init failure returns an empty
    // llama_sampler_ptr with `err` populated; we fail closed via
    // `result_.error` and never partially install a chain on the seq.
    //
    // Greedy / default admissions leave `sampler_chain` null, so the
    // two sampling sites in run_body() take the existing
    // `llama_get_logits_ith` + local argmax path bit-identically.
    rseq.sampler_chain.reset();
    if (rseq.sampling.mode == sampling_mode::stochastic) {
        std::string       err;
        llama_sampler_ptr chain = build_sampler_chain(rseq.sampling, err);
        if (!chain) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "admission: %s for stochastic request rid=%d seq=%d",
                err.c_str(), w.request_id, reuse_seq);
            result_.error = buf;
            return false;
        }
        rseq.sampler_chain = std::move(chain);
    }
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
        // M2g: caller-supplied stream channel takes precedence over
        // the gate's `stream_all_` admitted-handoff path. When the
        // caller submitted with want_stream=true, the receiver is
        // already on the caller's side via submit_handle.stream; the
        // engine moves the matching channel into the slot here and
        // skips pushing onto admitted_stream_handoffs_ (which is the
        // gate's bulk-collection mechanism, not a public API). The
        // engine's existing publish_token / close_stream paths drive
        // the channel from here exactly like the stream_all_ path.
        // No mutex — external_stream_channels_ is engine-task-only.
        auto it_stream =
            external_stream_channels_.find(w.request_id);
        if (it_stream != external_stream_channels_.end()) {
            rseq.stream_channel        = std::move(it_stream->second);
            rseq.stream_enabled        = true;
            rseq.stream_closed         = false;
            rseq.stream_tokens_emitted = 0;
            result_.streams_opened++;
            trace::event(
                "token_stream_opened request=%d seq_id=%d",
                w.request_id, reuse_seq);
            external_stream_channels_.erase(it_stream);
        }
        // Streaming Slice 5 / Slice 6: rebind the slot's
        // stream channel for this externally-arriving
        // admitted request when --stream-all is on AND the
        // admission source is completion_freed (Slice 5
        // scope) OR cancel_freed (Slice 6 scope —
        // structurally symmetric to Slice 4 on the
        // preloaded branch: the cancel-freed slot's state
        // at admission is symmetric to a completion-freed
        // slot, with stream_closed=true left by the prior
        // occupant's close_stream(cancelled) call). The
        // submitter holds the result future directly; the
        // stream receiver is pushed onto
        // admitted_stream_handoffs_ keyed by request_id
        // so main drains it via the same Slice 3/4 path.
        // The push, the streams_opened++, and the trace
        // event all happen inside the same existing
        // admitted_futures_mtx_ critical section (M4d:
        // hpx::spinlock — HPX-native result-handoff primitive).
        // No new synchronization primitive, no new CLI flag.
        // The lock guards engine→main result-handoff
        // metadata only; it does NOT guard llama.cpp
        // execution state.
        // M2g: only fall into the gate's admitted-handoff path when
        // no caller-supplied channel was installed above. With every
        // existing gate smoke (`want_stream=false`), the new
        // short-circuit is silent and this block fires identically
        // to pre-M2g.
        else if (stream_all_
            && (src == admission_source::completion_freed
             || src == admission_source::cancel_freed)) {
            std::lock_guard<hpx::spinlock> lk(
                admitted_futures_mtx_);
            rseq.stream_channel = token_stream_channel{};
            rseq.stream_closed  = false;
            rseq.stream_tokens_emitted = 0;
            admitted_stream_handoffs_.push_back(
                admitted_stream_handoff{
                    w.request_id,
                    token_stream_receiver(
                        rseq.stream_channel)});
            result_.streams_opened++;
            trace::event(
                "token_stream_opened request=%d seq_id=%d",
                w.request_id, reuse_seq);
        }
    } else {
        promises_[static_cast<size_t>(reuse_seq)] =
            hpx::promise<request_result>{};
        {
            std::lock_guard<hpx::spinlock> lk(
                admitted_futures_mtx_);
            admitted_futures_.emplace_back(
                promises_[static_cast<size_t>(reuse_seq)]
                    .get_future());
            // Streaming Slice 3 / Slice 4: rebind the
            // slot's stream channel for this admitted
            // preloaded waiter when --stream-all is on
            // AND the admission source is either
            // completion_freed (Slice 3 scope) or
            // cancel_freed (Slice 4 scope — added without
            // any new HPX primitive or new mutex; the
            // cancel-freed slot's state at admission is
            // symmetric to a completion-freed slot, with
            // stream_closed=true left by the prior
            // occupant's close_stream(cancelled) call).
            // External-arrival admitted streaming is
            // handled in the external-arrival branch
            // above with the same
            // completion_freed || cancel_freed predicate
            // (Slice 5 added external × completion_freed,
            // Slice 6 added external × cancel_freed); this
            // preloaded branch mirrors that policy for
            // queued preloaded waiters. The prior
            // occupant's close_stream(...) already flipped
            // stream_closed=true, so we assign a fresh
            // token_stream_channel and clear the flag.
            // stream_enabled stays true (set at ctor).
            // streams_opened bumps once per fresh stream.
            // The handoff push, the streams_opened++, and
            // the trace event all happen inside the same
            // existing admitted_futures_mtx_ critical
            // section (M4d: hpx::spinlock). No new
            // synchronization primitive, no expansion of
            // the locked region beyond the existing
            // result-handoff scope. No llama.cpp state is
            // touched here.
            if (stream_all_
                && (src == admission_source::completion_freed
                 || src == admission_source::cancel_freed)) {
                rseq.stream_channel = token_stream_channel{};
                rseq.stream_closed  = false;
                // Streaming Slice 3: the cumulative per-
                // slot stream_tokens_emitted counter is NOT
                // reset by the existing admit_one metadata
                // reset block, so the close-event trace
                // n_tokens would otherwise read prev+
                // admitted (e.g. 8+16=24) and be confusing
                // in evidence. Main-side correctly counts
                // events on the fresh receiver, but reset
                // the per-slot counter so the
                // token_stream_closed trace n_tokens
                // matches the admitted request's actual
                // emitted count.
                rseq.stream_tokens_emitted = 0;
                admitted_stream_handoffs_.push_back(
                    admitted_stream_handoff{
                        w.request_id,
                        token_stream_receiver(
                            rseq.stream_channel)});
                result_.streams_opened++;
                trace::event(
                    "token_stream_opened request=%d seq_id=%d",
                    w.request_id, reuse_seq);
            }
        }
    }
    result_.admitted_count++;
    result_.reused_seq_id_set.push_back(reuse_seq);
    if (diag_enabled()) result_.metrics.cur_iter_diag.admitted++;
    return true;
}

// N3.0 P6: per-iter cancellation observation walk. Iteration-
// boundary only — cancel is never observed inside llama_decode.
// Returns false if cancel_and_fulfill bails (KV clear failure);
// caller frees the batch and finalizes.
bool engine::iter_observe_cancellations(int32_t iter,
                                        llama_memory_t mem) {
    // Cancellation observation #2: top of each decode iter,
    // BEFORE building active_idx. A cancelled seq becomes
    // seq.done=true via clear_and_check, so the active_idx
    // loop below skips it naturally. This makes
    // wasted_decode_rows_after_cancel structurally 0.
    // M2f: iterate over the full slot set so admitted-into-idle
    // slots are reachable here too. Idle slots that have not
    // been admitted carry done=true and short-circuit on the
    // `if (seq.done) continue` line, so behavior is identical
    // when initial_idle_slots == 0.
    for (size_t s = 0; s < seqs_.size(); s++) {
        seq_state & seq = seqs_[s];
        if (seq.done) continue;
        if (cancel_should_observe(seq)) {
            if (!cancel_and_fulfill(seq, iter, mem)) {
                return false;
            }
        }
    }
    return true;
}

// N3.0 P7: three-source admission priority drain. Order
// cancel_freed → completion_freed → initial_idle is frozen
// (Live Admission Slices 3 / 5 / M2f). The cancel_freed source
// is bounded by admission_eligible_count — the snapshot the
// caller took BEFORE iter_observe_cancellations — which is what
// produces the one-iter cancel-to-admit delay
// (admitted_at_iter == cancel_after + 1 on the smoke shape).
engine::iter_admission_result engine::iter_run_admissions(
    int32_t iter, llama_memory_t mem,
    size_t admission_eligible_count) {
    iter_admission_result r;
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
                       "cancel_freed", iter, mem)) {
            r.ok = false;
            return r;
        }
        r.admitted++;
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
                           "completion_freed", iter, mem)) {
                r.ok = false;
                return r;
            }
            r.admitted++;
        }
    }

    // M2f: drain free_idle_ AFTER cancel-freed and completion-freed
    // so the existing source priority for the gate's smokes is
    // preserved. No --reuse-completed gate: idle slots have no
    // prior occupant to demand-pair against. With
    // initial_idle_slots == 0 (every existing gate run), free_idle_
    // is empty and this loop is a no-op, so no new
    // `admission_source=initial_idle` trace event ever fires on
    // the seven canonical gate smokes.
    while (!free_idle_.empty()
        && !waiting_queue_consumable_.empty()) {
        const int32_t reuse_seq = free_idle_.front();
        free_idle_.pop_front();

        if (!admit_one(reuse_seq,
                       admission_source::initial_idle,
                       "initial_idle", iter, mem)) {
            r.ok = false;
            return r;
        }
        r.admitted++;
    }
    return r;
}

// N3.0 P9: build the per-iter llama_batch. Emits prefill rows
// for freshly admitted seqs (admitted_at_iter == iter &&
// n_decoded == 0), regular decode rows otherwise. Sets
// seq.i_batch / seq.pos_next as it goes. Returns the resulting
// batch row count — 0 means "no rows" and the caller breaks
// defensively. No bail path.
int32_t engine::iter_build_batch(int32_t iter, llama_batch & batch,
                                 std::vector<int32_t> & active_idx) {
    common_batch_clear(batch);
    // M2f: walk the full slot set so admitted-into-idle slots get
    // picked up for prefill/decode. Initial active loops at engine
    // start (request_admitted trace, prefill batch build) still use
    // n_seqs = budgets_.size() because those events only describe
    // the preloaded initial population.
    active_idx.reserve(seqs_.size());
    for (size_t s = 0; s < seqs_.size(); s++) {
        seq_state & seq = seqs_[s];
        if (seq.done) continue;
        // Live Admission Slice 3: a freshly admitted seq has
        // admitted_at_iter == iter && n_decoded == 0 in the
        // same iter as its admission. Emit prompt/prefill rows
        // (logits=true on the last only) instead of a single
        // decode row. The post-decode argmax then produces its
        // first token via the same path as a normal
        // post-prefill argmax, after which n_decoded == 1.
        // Slice C: live-admission prefill (Path A) is keyed on prefill
        // STATE, not the admission iter, so a chunked prefill
        // (prefill_budget_rows_ > 0) can span multiple iters. For B=0
        // this places the whole prompt in the admission iter, preserving
        // the previous `admitted_at_iter == iter && n_decoded == 0`
        // selection: a freshly admitted seq has prefill_complete == false
        // (set in admit_one), and n_decoded only advances after
        // prefill_complete (the sample loop skips mid-prefill seqs), so
        // the two predicates pick the same seqs for B=0. Path B
        // (preloaded) sets prefill_complete in run_body before the inner
        // loop and is therefore never selected by this branch.
        if (!seq.prefill_complete) {
            // M1c: admitted prefill reads the per-seq owned prompt vector
            // that admit_one moved in from the waiter.
            const auto &  prompt    = seq.prompt_tokens;
            const int32_t n         = static_cast<int32_t>(prompt.size());
            const int32_t start     = seq.prefill_cursor;
            const int32_t remaining = n - start;
            // B <= 0 means unbounded (place the whole remaining prompt this
            // iter); B > 0 caps the chunk. min(remaining, B) is >= 1
            // whenever start < n and B >= 1, so the cursor always advances
            // and the engine task cannot livelock on a zero-row prefill
            // iter. The only chunk == 0 case is an empty/exhausted prompt
            // (start >= n), which flips prefill_complete immediately below.
            int32_t chunk = remaining;
            if (prefill_budget_rows_ > 0 && chunk > prefill_budget_rows_) {
                chunk = prefill_budget_rows_;
            }
            assert((start < n) ? (chunk > 0) : (chunk == 0));
            for (int32_t l = 0; l < chunk; l++) {
                const int32_t p = start + l;
                // logits only on the final prompt token of the FINAL
                // chunk, keyed on the absolute index so any chunk-boundary
                // alignment is handled. Intermediate-chunk rows carry
                // logits=false and never have i_batch read (the sample
                // loop skips this seq while !prefill_complete).
                const bool final = (p == n - 1);
                common_batch_add(batch, prompt[p],
                                 /*pos=*/start + l,
                                 /*seq_ids=*/{seq.seq_id},
                                 /*logits=*/final);
                if (final) {
                    seq.i_batch = batch.n_tokens - 1;
                }
            }
            seq.prefill_cursor   = start + chunk;
            seq.pos_next         = seq.prefill_cursor;
            seq.prefill_complete = (seq.prefill_cursor >= n);
            assert(seq.prefill_cursor
                   <= static_cast<int32_t>(prompt.size()));
            // B <= 0 (and B >= n) must finish the whole prompt in one pass.
            assert(prefill_budget_rows_ > 0 || seq.prefill_complete);
            active_idx.push_back(static_cast<int32_t>(s));
            if (diag_enabled()) {
                result_.metrics.cur_iter_diag.prefill_rows += chunk;
            }
            // No decode_row trace for prefill rows; Slice 4
            // owns admitted-prefill trace events.
        } else {
            common_batch_add(batch, seq.last_token,
                             /*pos=*/seq.pos_next,
                             /*seq_ids=*/{seq.seq_id},
                             /*logits=*/true);
            seq.i_batch = batch.n_tokens - 1;
            seq.pos_next++;
            active_idx.push_back(static_cast<int32_t>(s));
            if (diag_enabled()) {
                result_.metrics.cur_iter_diag.decode_rows++;
            }
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
    return batch.n_tokens;
}

// N3.0 P10: per-iter decode site. Records the per-iter
// active_seqs_per_iter / rows_per_batch / decode_calls metrics,
// then calls llama_decode + llama_synchronize. Returns false on
// llama_decode != 0 with result_.decode_failures and
// result_.error set; caller frees the batch and finalizes.
bool engine::iter_run_decode(llama_batch & batch,
                             const std::vector<int32_t> & active_idx) {
    result_.metrics.active_seqs_per_iter.push_back(
        static_cast<int32_t>(active_idx.size()));
    result_.metrics.rows_per_batch.push_back(batch.n_tokens);
    result_.decode_calls++;
    result_.metrics.decode_calls = result_.decode_calls;

    // Phase 1 diagnostics: push per-iter aggregates aligned 1:1 with
    // active_seqs_per_iter. admitted/cancelled/prefill_rows/decode_rows
    // come from the per-engine cur_iter_diag accumulator populated by
    // earlier-in-iter phases (iter_observe_cancellations via
    // cancel_and_fulfill, iter_run_admissions via admit_one, and
    // iter_build_batch). completed/tokens_emitted are pushed as 0 here
    // and bumped in finalize_and_fulfill/publish_token below. iter_wall
    // and idle_wait are pushed as 0 and back-patched at iter bottom
    // (idle_wait stays 0 in Phase 1; deferred to Phase 2).
    // t_us_from_engine_start is the snapshot taken at iter top in
    // run_body (microseconds since t_start_), enabling server-side
    // JSONL request events to align to a common engine-relative axis.
    if (diag_enabled()) {
        const auto & acc = result_.metrics.cur_iter_diag;
        result_.metrics.prefill_rows_per_iter.push_back(acc.prefill_rows);
        result_.metrics.decode_rows_per_iter.push_back(acc.decode_rows);
        result_.metrics.admitted_per_iter.push_back(acc.admitted);
        result_.metrics.cancelled_per_iter.push_back(acc.cancelled);
        result_.metrics.t_us_from_engine_start_per_iter.push_back(
            acc.t0_us);
        result_.metrics.completed_per_iter.push_back(0);
        result_.metrics.tokens_emitted_per_iter.push_back(0);
        result_.metrics.llama_decode_wall_us_per_iter.push_back(0);
        result_.metrics.iter_wall_us_per_iter.push_back(0);
        // c=4 attribution: credit any idle wait accumulated at the
        // keep-alive tail (wait_inbox_blocking) since the last decode
        // iter to THIS iter — the iter the wake produced — then reset
        // the accumulator. Stays 0 for non-keep-alive engines (they
        // never reach the wait site) and for back-to-back iters with no
        // intervening idle wait.
        result_.metrics.idle_wait_us_per_iter.push_back(
            result_.metrics.pending_idle_wait_us);
        result_.metrics.pending_idle_wait_us = 0;
    }

    std::chrono::steady_clock::time_point decode_t0;
    if (diag_enabled()) decode_t0 = std::chrono::steady_clock::now();
    if (llama_decode(ctx_, batch) != 0) {
        result_.decode_failures++;
        result_.error = "llama_decode failed during decode loop";
        return false;
    }
    llama_synchronize(ctx_);
    if (diag_enabled()
     && !result_.metrics.llama_decode_wall_us_per_iter.empty()) {
        const auto t1 = std::chrono::steady_clock::now();
        result_.metrics.llama_decode_wall_us_per_iter.back() =
            std::chrono::duration_cast<std::chrono::microseconds>(
                t1 - decode_t0).count();
    }
    return true;
}

// N3.0 P11: per-iter sample + EOG + publish + finalize. For each
// active seq: sample (greedy argmax or per-seq sampler chain),
// check EOG and finalize, otherwise publish the token, fold the
// hash, bump n_decoded, and finalize on budget. The ordering
// inside the loop body is hash-relevant — publish_token runs
// BEFORE n_decoded mutates (Streaming Slice 8). Returns false on
// logits/finalize bail; caller frees the batch and finalizes.
bool engine::iter_sample_and_finalize(
    int32_t iter,
    const std::vector<int32_t> & active_idx,
    llama_memory_t mem) {
    for (int32_t s : active_idx) {
        seq_state & seq = seqs_[s];
        // Slice C: a mid-prefill seq is active for KV construction (its
        // chunk rows were placed and decoded this iter) but must never be
        // sampled. Skip it BEFORE any logits read, sampling, hash fold,
        // publish_token, generated_tokens push, n_decoded bump, or
        // finalize. With B=0 prefill always completes in iter_build_batch
        // within the same iter, so this skip is never taken on the
        // whole-prompt path; once prefill_complete is true the seq is
        // sampled exactly once per iter as before.
        if (!seq.prefill_complete) continue;
        // M6b: select the next token via per-seq sampler chain
        // when stochastic, otherwise fall through to the
        // bit-identical greedy argmax path. This is the site
        // where stochastic requests actually pick tokens in M6b
        // (the post-prefill site stays greedy because preloaded
        // actives carry no sampling_config). On greedy slots,
        // `seq.sampler_chain` is null and the path is byte-
        // identical to the pre-M6b argmax. `llama_sampler_sample`
        // internally calls `llama_sampler_accept`; do NOT call
        // accept separately or the dist RNG double-advances and
        // breaks same-seed reproducibility.
        llama_token next_id;
        if (seq.sampler_chain) {
            next_id = llama_sampler_sample(
                seq.sampler_chain.get(), ctx_, seq.i_batch);
        } else {
            const float * logits = llama_get_logits_ith(ctx_, seq.i_batch);
            if (logits == nullptr) {
                result_.error =
                    "llama_get_logits_ith returned null during decode";
                return false;
            }
            next_id = argmax(logits, n_vocab_);
        }
        // Live Admission Slice 4 / Slice C: capture the admitted-
        // prefill-argmax predicate BEFORE n_decoded changes, so the
        // event fires exactly once per admitted request (at the
        // post-decode argmax of the iter in which its prefill
        // completed and the first token is sampled). With chunked
        // prefill that iter may be LATER than the admission iter, so
        // the predicate is keyed on prefill having just finished
        // (n_decoded == 0, and the skip above guarantees
        // prefill_complete) rather than admitted_at_iter == iter. For
        // B=0 the first sample is in the admission iter where
        // admitted_at_iter == iter also held, so this selects the same
        // single event. Mirrors the seq_prefilled placement (before
        // the EOG check).
        const bool is_admitted_prefill_argmax =
            seq.admission_src != admission_source::none
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
                return false;
            }
            continue;
        }
        // Streaming Slice 8: publish the same int32 fed to the
        // hash BEFORE n_decoded mutates.
        publish_token(seq, static_cast<int32_t>(next_id));
        seq.generated_tokens.push_back(next_id);
        seq.hash_state = fold_token_hash(seq.hash_state,
                                         static_cast<int32_t>(next_id));
        seq.n_decoded++;
        seq.last_token = next_id;
        if (seq.n_decoded >= seq.decode_budget) {
            if (!finalize_and_fulfill(seq, iter, mem)) {
                return false;
            }
        }
    }
    return true;
}

// N3.0 P12: gate-test release/ack barrier at end of iter K. If
// K is in iter_release_set_, fire the release promise and block
// on the matching submitter ack future. The .get() on the ack
// future is intentionally synchronous on the engine task —
// scheduling order is deterministic on a single HPX worker
// because the engine only proceeds past this point after every
// K-arrival is in the inbox. Returns false on ack-future
// exception with result_.error set; caller frees the batch and
// finalizes.
bool engine::iter_fire_release_ack_barrier(int32_t iter) {
    if (!iter_release_set_.count(iter)) {
        return true;
    }
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
        return false;
    }
    result_.submitter_ack_set.push_back(iter);
    trace::event("submitter_ack_observed iter=%d", iter);
    return true;
}

// run_body — single-engine-task control loop. The inner loop follows
// named HPX control-plane phases (INTAKE, CANCEL, ADMIT, BUILD, DECODE,
// SAMPLE_PUBLISH_FINALIZE, RELEASE, WAIT). llama_decode is the single
// llama.cpp execution boundary and is crossed exactly once per iter.
// Detailed dataflow and ownership tables live in
// docs/hpx/hpx_native_serving_control_plane_design.md.
void engine::run_body() {
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
    // Streaming Slice 8: reset stream counters per repeat.
    result_.streams_opened              = 0;
    result_.streams_closed_completed    = 0;
    result_.streams_closed_cancelled    = 0;
    result_.streams_closed_error        = 0;
    result_.stream_tokens_emitted_total = 0;
    // M3a: keep-alive counters reset per repeat. Both stay at zero
    // for every keep_alive=false run, so the seven canonical gate
    // smokes are byte-identical to M2.
    result_.engine_idle_waits           = 0;
    result_.engine_shutdown_observed    = 0;
    // M3a: clear the engine-observed shutdown latch so a
    // request_shutdown() observed at the end of a prior repeat does
    // not poison the next repeat. N1: staged_shutdown_ is engine-
    // task-only; no lock. The inbox_chan_ buffer and the staged_*
    // arrival/cancel deques are intentionally NOT cleared (they are
    // symmetric with the legacy `inbox_` / `cancel_inbox_` / `cancel_
    // token_inbox_` deques that were also not cleared), so
    // submit()/cancel_request() calls made BEFORE the engine task
    // actually picks up run() are honored — clearing them would
    // race with the producer and silently drop pre-run work. End-
    // of-run residue is drained into cancelled_request_ids_ and
    // accounted as cancel_unknown_request_id at the run_body tail,
    // then cancelled_request_ids_ is cleared, so no entries leak
    // into a subsequent repeat through this path.
    // cancelled_request_ids_ is engine-task-only and is reset here
    // defensively in case a prior run_body bailed out early before
    // its tail cleanup. live_epoch_by_rid_ is engine-task-only and
    // reset defensively. next_epoch_ is NOT reset — it is
    // monotonic for the lifetime of the engine so tokens issued on
    // one repeat cannot collide with submissions on the next.
    staged_shutdown_ = false;
    cancelled_request_ids_.clear();
    cancelled_tokens_.clear();
    live_epoch_by_rid_.clear();
    result_.queued_cancelled            = 0;
    result_.cancel_request_calls        = 0;
    result_.cancel_active_not_supported = 0;
    result_.cancel_unknown_request_id   = 0;
    result_.cancel_request_duplicates   = 0;
    result_.cancel_stale_epoch          = 0;
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
    // M2f: re-seed the idle-slot pool per run. Idle slots also have
    // their dynamic state reset below (done=true, decode_budget=0,
    // request_id=-1, prompt_tokens cleared) so a slot returning to
    // idle between repeats starts the next run as a fresh idle slot.
    free_idle_.clear();
    for (size_t s = budgets_.size(); s < seqs_.size(); s++) {
        free_idle_.push_back(static_cast<int32_t>(s));
    }
    // M2g: defensively clear any leftover per-request stream channels
    // from a prior repeat. With `--repeat 1` this is a no-op; with
    // `--repeat N` and a submission that never got admitted in repeat
    // r, the engine cleanup at the end of repeat r already closed and
    // dropped the channel, so the map is empty at this point. Clear
    // is the safe invariant either way.
    external_stream_channels_.clear();
    // M4b: waiting_queue_ may be null (normal-library users that only
    // use submit_request). Null is treated as an empty preloaded queue;
    // the deref is guarded strictly here so the rest of the engine
    // continues to operate on waiting_queue_consumable_ unchanged.
    if (waiting_queue_ != nullptr) {
        waiting_queue_consumable_.assign(
            waiting_queue_->begin(), waiting_queue_->end());
    } else {
        waiting_queue_consumable_.clear();
    }
    {
        std::lock_guard<hpx::spinlock> lk(admitted_futures_mtx_);
        admitted_futures_.clear();
        // Streaming Slice 3: admitted-stream handoff vector lives
        // under the same critical section as admitted_futures_.
        admitted_stream_handoffs_.clear();
    }

    result_.queued_count =
        static_cast<int32_t>(waiting_queue_consumable_.size());

    auto * mem = llama_get_memory(ctx_);
    llama_memory_clear(mem, /*data=*/false);

    const int32_t n_seqs   = static_cast<int32_t>(budgets_.size());
    // M4c: prompt_tokens_ may be null on submit_request-only engines
    // (budgets_.empty()); use 0 in that case. n_prompt feeds the
    // engine_start trace event only — it never indexes prompt content.
    const int32_t n_prompt = (prompt_tokens_ != nullptr)
        ? static_cast<int32_t>(prompt_tokens_->size())
        : 0;
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
    // M2f: idle slots (s >= budgets_.size()) reset to the "available,
    // never bound" state: done=true so the prefill / decode_row /
    // cancel-observe / any_active() loops skip them; decode_budget=0
    // and request_id=-1 as sentinels until admit_one rebinds them.
    // prompt_tokens is cleared so a slot that was admitted in a prior
    // repeat returns to a fresh-idle vector instead of carrying the
    // prior occupant's tokens. With initial_idle_slots == 0 the loop
    // never enters the else branch and behavior is byte-identical.
    for (size_t s = 0; s < seqs_.size(); s++) {
        seq_state & seq = seqs_[s];
        seq.n_decoded         = 0;
        seq.pos_next          = 0;
        // Slice B: reset prefill progress for the new repeat. With the
        // whole-prompt path these are immediately set to (size, true) in
        // the prefill pass below, so behavior is unchanged.
        seq.prefill_cursor    = 0;
        seq.prefill_complete  = false;
        seq.i_batch           = -1;
        seq.last_token        = 0;
        seq.done_iter         = -1;
        seq.pos_max_at_clear  = -1;
        seq.kv_cleared        = false;
        seq.promise_fulfilled = false;
        seq.hash_state        = k_token_hash_init;
        seq.generated_tokens.clear();
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
        // M5a: reset per-seq epoch for the new repeat. Initial actives
        // and idle slots both start at 0 (the "no engine-issued
        // token" sentinel); admit_one overwrites this with the
        // bound waiter's `epoch` field at admission time when the
        // waiter came from submit_request.
        seq.epoch                = 0;
        if (s < budgets_.size()) {
            seq.done          = false;
            seq.decode_budget = budgets_[s];
            seq.request_id    = static_cast<int32_t>(s);
        } else {
            seq.done          = true;
            seq.decode_budget = 0;
            seq.request_id    = -1;
            seq.prompt_tokens.clear();
        }
    }

    t_start_ = std::chrono::steady_clock::now();
    // c=4 attribution: stamp t_start_ in the absolute steady_clock epoch
    // so engine per-iter rows (relative t_us_from_engine_start) can be
    // overlaid on server_request rows (absolute server_now_us). Reuses
    // the already-captured time_point — no extra clock read. Diag-gated;
    // emitted in the engine_summary JSONL row.
    if (diag_enabled()) {
        result_.engine_t_start_us_absolute =
            std::chrono::duration_cast<std::chrono::microseconds>(
                t_start_.time_since_epoch()).count();
    }

    trace::event("engine_start n_seqs=%d prompt_tokens=%d max_budget=%d",
                 n_seqs, n_prompt, max_budget);
    for (int32_t s = 0; s < n_seqs; s++) {
        trace::event("request_admitted seq=%d budget=%d",
                     seqs_[s].seq_id, seqs_[s].decode_budget);
    }
    // Streaming Slice 8: emit token_stream_opened for every
    // stream_enabled seq and bump streams_opened. Engine task only.
    for (int32_t s = 0; s < n_seqs; s++) {
        if (!seqs_[s].stream_enabled) continue;
        result_.streams_opened++;
        trace::event(
            "token_stream_opened request=%d seq_id=%d",
            seqs_[s].request_id, seqs_[s].seq_id);
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
        // M1b: initial prefill reads the per-seq owned prompt vector
        // populated by the engine ctor. In M1b every initial seq's
        // seq.prompt_tokens is a copy of the same shared prompt
        // source, so prompt.size() equals the outer n_prompt and the
        // emitted rows / pos / logits selection are byte-identical to
        // the M1a / M0 path. M1c will diverge when waiters/external
        // arrivals carry their own prompt vectors.
        const auto &  prompt       = seq.prompt_tokens;
        const int32_t n_seq_prompt =
            static_cast<int32_t>(prompt.size());
        for (int32_t p = 0; p < n_seq_prompt; p++) {
            const bool last = (p == n_seq_prompt - 1);
            common_batch_add(batch, prompt[p], /*pos=*/p,
                             /*seq_ids=*/{seq.seq_id}, /*logits=*/last);
            if (last) {
                seq.i_batch = batch.n_tokens - 1;
            }
        }
        seq.pos_next = n_seq_prompt;
        // Slice B: preloaded/initial whole-prompt prefill completes here,
        // before the post-prefill argmax pass below. Inert state only.
        seq.prefill_cursor   = n_seq_prompt;
        seq.prefill_complete = true;
        assert(seq.prefill_cursor <= static_cast<int32_t>(prompt.size()));
        assert(seq.prefill_complete);
    }

    // M2f: with zero initial actives (budgets_.empty()), the prefill
    // batch above adds no rows. Skip the prefill llama_decode +
    // post-prefill argmax pass entirely so an empty llama_batch is
    // never submitted. With at least one initial active, batch.n_tokens
    // is strictly > 0 and the body runs byte-identically to pre-M2f.
    if (batch.n_tokens > 0) {
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
            // Slice B invariant (preloaded path): the initial prefill loop
            // above set prefill_complete = true for every preloaded active
            // before this post-prefill argmax. Debug-only; inert in NDEBUG.
            assert(seq.prefill_complete);
            // M6b: select the next token via per-seq sampler chain
            // when stochastic, otherwise fall through to the
            // bit-identical greedy argmax path. The post-prefill site
            // fires for preloaded actives only; in M6b those are
            // always greedy (engine ctor does not build a chain for
            // them, and the scripted submitter path doesn't carry
            // sampling), so `seq.sampler_chain` is always null here
            // in M6b — the branch is added for symmetry with the
            // per-iter site and to keep the surface ready for future
            // slices that may route stochastic preloaded actives.
            // `llama_sampler_sample` internally calls
            // `llama_sampler_accept`; do NOT call accept separately.
            llama_token next_id;
            if (seq.sampler_chain) {
                next_id = llama_sampler_sample(
                    seq.sampler_chain.get(), ctx_, seq.i_batch);
            } else {
                const float * logits = llama_get_logits_ith(ctx_, seq.i_batch);
                if (logits == nullptr) {
                    result_.error = "llama_get_logits_ith returned null after prefill";
                    llama_batch_free(batch);
                    finalize_wall_ms();
                    return;
                }
                next_id = argmax(logits, n_vocab_);
            }
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
            // Streaming Slice 8: publish the same int32 fed to the hash
            // BEFORE n_decoded mutates, so streamed-hash == rr.hash and
            // stream_tokens_emitted == rr.n_decoded hold by construction.
            publish_token(seq, static_cast<int32_t>(next_id));
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
    // M2f: predicate widened to also peek the async inbox so the loop
    // can enter / re-enter even when there are no active seqs yet and
    // no waiters have been drained. N1: inbox_has_pending() pumps the
    // HPX inbox channel into staged_arrivals_ and reports
    // !staged_arrivals_.empty(); no llama.cpp API call sites. With the
    // gate's existing smokes (initial_idle_slots=0 and inbox empty
    // outside the release-barrier window), the new term flips only
    // during the one-iter drain window where any_active() is already
    // true, so the predicate is behaviorally identical and the gate's
    // smoke output remains byte-for-byte the same.
    int32_t iter = 0;
    // M3a: keep-alive outer loop. With keep_alive_=false this body
    // executes once — the inner finite while drains all work, the
    // !keep_alive_ short-circuit fires, and the outer loop exits.
    // Behavior in that case is byte-identical to M2: no idle-wait,
    // no counter bumps, just the inner predicate's pump call.
    // N1: with keep_alive_=true the engine returns to the outer
    // tail when there is no work, pumps the channel, observes
    // staged_shutdown_, and either reports engine_shutdown_observed=1
    // + breaks (drained shutdown) or increments engine_idle_waits and
    // HPX-suspends on the inbox channel via wait_inbox_blocking()
    // until a new arrival or shutdown wakes us. The inner finite
    // loop is intentionally NOT reindented — it is the unchanged M2
    // decode/admission body wrapped verbatim.
    while (true) {
    while (any_active()
        || !waiting_queue_consumable_.empty()
        || inbox_has_pending()) {
        iter++;

        // Phase 1 diagnostics: reset per-engine per-iter accumulator
        // at iter top and stamp t0_us as microseconds since this
        // engine's t_start_ baseline. cur_iter_diag is a member of
        // this engine's result_.metrics, so concurrent engines never
        // share this state. All zero-cost when LLAMA_HPX_DIAG_METRICS
        // is unset.
        if (diag_enabled()) {
            auto & acc = result_.metrics.cur_iter_diag;
            acc.admitted     = 0;
            acc.cancelled    = 0;
            acc.prefill_rows = 0;
            acc.decode_rows  = 0;
            acc.t0_us        =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - t_start_).count();
        }

        // ---- Phase INTAKE — pump inbox; drain external arrivals +
        // cancels; apply queued cancellations. HPX control plane only;
        // no KV touch. Phase 1 counters attach here (arrival_drained_count,
        // cancel_request_calls, queued_cancelled, cancel_active_observed).

        // N1: unconditional channel pump at iter top. The inner-
        // while predicate short-circuits on any_active() and never
        // calls inbox_has_pending() during a mid-decode iter, so
        // cancels/submissions published on the channel while the
        // engine is actively decoding would not be staged without
        // this call. drain_cancel_inbox + drain_external_inbox
        // below then operate on the snapshot of staged_* taken
        // here. Pump is NOT called inside drain_external_inbox,
        // so cancels arriving on the channel mid-arrival-drain
        // stay buffered and are observed only at the next iter's
        // cancel phase.
        pump_inbox_nonblocking();

        // M3b: drain the public cancel inbox BEFORE the external
        // arrival inbox, so any cancel_request(rid) for an rid that
        // is ALSO in this iter's arrival batch is observed in time
        // for drain_external_inbox()'s arrival-side guard to short-
        // circuit the queue push. apply_queued_cancellations runs
        // AFTER drain_external_inbox so a fresh arrival admitted
        // into the queue can be cancelled in the same iter if the
        // matching cancel arrived during the same wake window.
        drain_cancel_inbox();
        // Live Admission Slice 6: drain the async-arrival inbox at
        // TOP of this iter, BEFORE the cancel-freed snapshot. Pairs
        // with the release+ack barrier at the END of iter K
        // (release-at-K, drain-at-K+1 semantics): an arrival pushed
        // by the submitter during the iter-K barrier is guaranteed
        // visible here at iter K+1 because the engine task only
        // resumes after the submitter's ack fires.
        drain_external_inbox(iter);
        // M3b: resolve any cancellations against waiting queue or
        // live active seqs. Engine-task-only; no llama API; no KV
        // mutation. Runs every iter so a cancel arriving during
        // active decode is observed promptly (active matches bump
        // cancel_active_not_supported in M3b; M3c will replace).
        apply_queued_cancellations();

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

        // ---- Phase CANCEL — observe active-seq cancellations at iter
        // boundary; KV clear + stream close + promise fulfill. Pushes
        // the freed slot to free_due_to_cancel_ for next-iter admission.
        // Phase 1 counter attaches here (cur_iter_diag.cancelled).
        // P6: cancellation observation #2.
        if (!iter_observe_cancellations(iter, mem)) {
            llama_batch_free(batch);
            result_.metrics.update_iterations = iter;
            finalize_wall_ms();
            return;
        }

        // ---- Phase ADMIT — three-source priority drain (cancel_freed
        // -> completion_freed -> initial_idle) binds waiting_request to
        // a KV-empty slot. Engine task only; no llama_decode here.
        // Phase 1 counter attaches here (cur_iter_diag.admitted).
        // P7: three-source admission priority drain.
        const iter_admission_result adm =
            iter_run_admissions(iter, mem, admission_eligible_count);
        if (!adm.ok) {
            llama_batch_free(batch);
            result_.metrics.update_iterations = iter;
            finalize_wall_ms();
            return;
        }

        // P8: admission metrics + early break if no active. Live
        // Admission Slice 4: one push to admission_iter_set per iter
        // that admitted at least one waiting request; then sample
        // queue depth AFTER admission. If every still-active seq was
        // just cancelled AND no admission happened, end the loop
        // cleanly without trying to decode an empty batch.
        if (adm.admitted > 0) {
            result_.metrics.admission_iter_set.push_back(iter);
        }
        result_.metrics.waiting_queue_depth_after_admission_per_iter
            .push_back(static_cast<int32_t>(
                waiting_queue_consumable_.size()));
        if (!any_active()) {
            result_.metrics.update_iterations = iter;
            break;
        }

        // ---- Phase BUILD — compose llama_batch rows: admitted-prefill
        // rows plus one decode row per still-active seq. llama_batch
        // shape mutation only; no llama_decode here. Phase 1 counters
        // attach here (prefill_rows / decode_rows per iter; rows_per_batch).
        // P9: build per-iter llama_batch (admitted-prefill rows OR
        // regular decode rows). Returns 0 → defensive break.
        std::vector<int32_t> active_idx;
        if (iter_build_batch(iter, batch, active_idx) == 0) break;

        // ---- Phase DECODE — THE single llama_decode site per iter.
        // llama.cpp execution boundary. Engine task only; never
        // parallelized on the same llama_context. Phase 1 counter
        // attaches here (llama_decode_wall_us_per_iter).
        // P10: per-iter decode site + per-iter decode metrics.
        if (!iter_run_decode(batch, active_idx)) {
            llama_batch_free(batch);
            result_.metrics.update_iterations = iter;
            finalize_wall_ms();
            return;
        }

        // ---- Phase SAMPLE_PUBLISH_FINALIZE — per-seq sampler/argmax,
        // publish_token to per-request stream channel, finalize_and_fulfill
        // on EOG/budget (KV clear + promise). Phase 1 counters attach
        // here (tokens_emitted_per_iter, completed_per_iter).
        // P11: sample + EOG + publish + finalize per active seq.
        if (!iter_sample_and_finalize(iter, active_idx, mem)) {
            llama_batch_free(batch);
            result_.metrics.update_iterations = iter;
            finalize_wall_ms();
            return;
        }

        // ---- Phase RELEASE — gate-test release/ack barrier (no-op
        // when the release_iter set is empty) and iter-wall diagnostic
        // backpatch onto the row pushed in DECODE. HPX control plane only.
        // P12: gate-test release/ack barrier at end of iter K.
        if (!iter_fire_release_ack_barrier(iter)) {
            llama_batch_free(batch);
            result_.metrics.update_iterations = iter;
            finalize_wall_ms();
            return;
        }

        // Phase 1 diagnostics: back-patch this iter's wall time into
        // the entry pushed by iter_run_decode. Computed as
        // (now_us_from_t_start_ - cur_iter_diag.t0_us), so it shares
        // the engine-relative time axis with t_us_from_engine_start.
        // Guarded by !empty so an iter that hit the no-active early-
        // break (which does not push) cannot corrupt a prior iter's
        // entry.
        if (diag_enabled()
         && !result_.metrics.iter_wall_us_per_iter.empty()) {
            const int64_t now_us_from_start =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - t_start_).count();
            result_.metrics.iter_wall_us_per_iter.back() =
                now_us_from_start
                  - result_.metrics.cur_iter_diag.t0_us;
        }
    }
    // ---- Phase WAIT — keep-alive outer-loop tail. Observe shutdown,
    // continue if staged work exists, else HPX-native suspend on the
    // inbox channel via wait_inbox_blocking() (no condition_variable,
    // no busy-poll). Phase 1 counter attaches here (engine_idle_waits).
    // M3a: keep-alive outer-loop tail. With keep_alive_=false this
    // breaks immediately. With keep_alive_=true: N1: pump the HPX
    // inbox channel into the staged_* deques and observe
    // staged_shutdown_ on an engine with no active sequences.
    // any_active() is false here (either the inner predicate returned
    // false or the no-active early break fired); waiting_queue_
    // consumable_ may still be NON-empty when the early break fired
    // with a queued-but-unadmittable request, so N5b drains that queue
    // on the shutdown branch rather than gating shutdown on it. If
    // shutdown is observed, set engine_shutdown_observed=1 and break.
    // Otherwise increment
    // engine_idle_waits and HPX-suspend on the inbox channel via
    // wait_inbox_blocking() — the engine task parks on the
    // channel's shared state until a new message arrives, with no
    // condition_variable_any.
    // M3b: process any pending queued cancellations BEFORE the
    // shutdown-drain / idle-wait decision. This is required for
    // the smoke shape (budgets={}, initial_idle_slots=0, one
    // request queued and never admissible) — the inner-while
    // breaks with a non-empty waiting queue and shutdown not yet
    // requested; the only way the queued request can be cancelled
    // is at the outer-loop tail before the engine idle-waits.
    // drain_cancel_inbox + apply_queued_cancellations are engine-
    // task-only. The pump call after them folds any messages that
    // arrived while we were applying queued cancellations into the
    // predicate.
    if (!keep_alive_) break;
    drain_cancel_inbox();
    apply_queued_cancellations();
    pump_inbox_nonblocking();
    if (staged_shutdown_
        && staged_arrivals_.empty()
        && staged_cancel_rids_.empty()
        && staged_cancel_tokens_.empty()
        && !any_active()) {
        // N5b: once there are no active sequences and no staged work,
        // shutdown is total — a queued-but-unadmittable request must
        // NOT suppress it (the inner loop's no-active early break can
        // reach here with waiting_queue_consumable_ non-empty). Resolve
        // every still-queued waiting request with status=failed_reserved
        // (shutdown-aborted) and close any queued stream before exiting,
        // so no submitter future is left unfulfilled and the run-end
        // external_promises_-empty guard holds. With an empty waiting
        // queue (every existing gate and smoke shape) this drain is a
        // no-op and the break fires exactly as before, preserving all
        // canonical anchors.
        drain_waiting_queue_for_shutdown();
        result_.engine_shutdown_observed = 1;
        break;
    }
    // The pump above may stage arrivals/cancels that landed AFTER this
    // tail's drain_cancel_inbox()/apply_queued_cancellations() pass. Do
    // not park on the inbox while actionable staged work exists — loop
    // back so the inner-loop drain/apply/drain_external sites consume it
    // (actor-style: never suspend on a mailbox with pending work). A
    // late queued cancel would otherwise be stranded until some future
    // message wakes the engine. staged_shutdown_ is intentionally
    // excluded — shutdown stays governed by the predicate above.
    if (!staged_arrivals_.empty()
        || !staged_cancel_rids_.empty()
        || !staged_cancel_tokens_.empty()) {
        continue;
    }
    result_.engine_idle_waits++;
    // N1: HPX-native suspension on the inbox channel. Wakes on any
    // submission, cancel (rid or token), or shutdown — channel
    // FIFO; no lost-wakeup risk because the producer publishes
    // before its set() returns.
    // c=4 attribution: time the (already-suspending) wait only when
    // diagnostics are on, and accumulate it into pending_idle_wait_us.
    // No behavior change — wait_inbox_blocking() is unchanged; the
    // chrono pair brackets an existing suspension point and is credited
    // to the next decode iter at the iter_run_decode push site.
    if (diag_enabled()) {
        const auto idle_t0 = std::chrono::steady_clock::now();
        wait_inbox_blocking();
        const auto idle_t1 = std::chrono::steady_clock::now();
        result_.metrics.pending_idle_wait_us +=
            std::chrono::duration_cast<std::chrono::microseconds>(
                idle_t1 - idle_t0).count();
    } else {
        wait_inbox_blocking();
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
    // empty at engine end. Non-empty staged arrivals means an
    // arrival was pushed after the engine's last drain (a barrier-
    // discipline bug). Non-empty external_promises_ means an
    // arrival was drained but never admitted, which would leak the
    // submitter's future. Fail closed before the residual-KV sweep
    // so the diagnostic is unambiguous. N1: pump the channel first
    // to catch any arrival that landed since the last predicate
    // pump; then read staged_arrivals_ without a lock.
    {
        pump_inbox_nonblocking();
        size_t inbox_residual = staged_arrivals_.size();
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

    // M3b: account for cancelled_request_ids_ residue. Any rid that
    // was cancel_request()'d but never matched a queued or active
    // request is bookkeeping (cancel-before-submit that never
    // received a matching submission, or cancel-after-completion).
    // Each residue entry bumps cancel_unknown_request_id and emits
    // a trace; do NOT fail-close on residue so existing smokes
    // (which never call cancel_request) remain byte-identical. N1:
    // pump the channel once more to catch any cancel that arrived
    // since the last predicate pump; then drain the staged cancel
    // deques as residue. No lock — staged_* are engine-task-only.
    {
        pump_inbox_nonblocking();
        while (!staged_cancel_rids_.empty()) {
            const int32_t rid = staged_cancel_rids_.front();
            staged_cancel_rids_.pop_front();
            result_.cancel_request_calls++;
            cancelled_request_ids_.insert(rid);
        }
        // M5a: symmetric flush of staged_cancel_tokens_ as residue.
        while (!staged_cancel_tokens_.empty()) {
            const cancel_token tok = staged_cancel_tokens_.front();
            staged_cancel_tokens_.pop_front();
            result_.cancel_request_calls++;
            cancelled_tokens_.insert(
                std::make_pair(tok.request_id, tok.epoch));
        }
    }
    for (int32_t rid : cancelled_request_ids_) {
        result_.cancel_unknown_request_id++;
        trace::event("cancel_unknown_request_id request=%d", rid);
    }
    cancelled_request_ids_.clear();
    // M5a: residue from cancelled_tokens_ is accounted as
    // cancel_unknown_request_id when the token's rid is not (or no
    // longer) live, and as cancel_stale_epoch when it is live but
    // for a different epoch — the latter is rare here because the
    // resolution pass would normally bump cancel_stale_epoch and
    // erase the entry during the run. Either accounting form
    // surfaces the residue without double-counting.
    for (const auto & ke : cancelled_tokens_) {
        const int32_t  rid   = ke.first;
        const uint64_t epoch = ke.second;
        auto live_it = live_epoch_by_rid_.find(rid);
        if (live_it != live_epoch_by_rid_.end()
         && live_it->second != epoch) {
            result_.cancel_stale_epoch++;
            trace::event(
                "cancel_stale_epoch_residue request=%d "
                "token_epoch=%llu live_epoch=%llu",
                rid,
                static_cast<unsigned long long>(epoch),
                static_cast<unsigned long long>(live_it->second));
        } else {
            result_.cancel_unknown_request_id++;
            trace::event(
                "cancel_unknown_request_id_token "
                "request=%d epoch=%llu",
                rid,
                static_cast<unsigned long long>(epoch));
        }
    }
    cancelled_tokens_.clear();
    live_epoch_by_rid_.clear();

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

void engine::finalize_wall_ms() {
    const auto now = std::chrono::steady_clock::now();
    const auto us  = std::chrono::duration_cast<std::chrono::microseconds>(
        now - t_start_).count();
    result_.metrics.wall_ms = static_cast<double>(us) / 1000.0;
}
