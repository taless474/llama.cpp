#include "engine.h"

#include "token_hash.h"
#include "trace.h"

#include "common.h"
#include "llama.h"

#include <hpx/hpx.hpp>
#include <hpx/lcos_local/channel.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
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

using namespace token_hash;

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

}  // namespace

engine::engine(engine_options opts)
    : ctx_(opts.ctx),
      vocab_(opts.vocab),
      n_vocab_(opts.n_vocab),
      prompt_tokens_(*opts.prompt_tokens),
      budgets_(std::move(opts.budgets)),
      batch_capacity_(opts.batch_capacity),
      promises_(budgets_.size()),
      seqs_(budgets_.size()),
      n_seq_max_(opts.n_seq_max),
      waiting_queue_(opts.waiting_queue),
      reuse_completed_(opts.reuse_completed),
      iter_release_set_(std::move(opts.release_iter_set)),
      max_decode_iters_(opts.max_decode_iters),
      stream_all_(opts.stream_all)
{
    if (opts.ctx == nullptr) {
        throw std::invalid_argument(
            "engine_options::ctx must be non-null");
    }
    if (opts.vocab == nullptr) {
        throw std::invalid_argument(
            "engine_options::vocab must be non-null");
    }
    if (opts.prompt_tokens == nullptr) {
        throw std::invalid_argument(
            "engine_options::prompt_tokens must be non-null");
    }
    if (opts.waiting_queue == nullptr) {
        throw std::invalid_argument(
            "engine_options::waiting_queue must be non-null");
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
    }
    // Cancel Slice 2: propagate the deterministic plan into
    // seq_state. Out-of-range seq_ids are silently skipped here
    // because main has already validated the plan against
    // [0, n_seqs) and printed the per-seq budget assertion.
    for (int32_t cs : opts.cancel_plan) {
        if (cs >= 0 && static_cast<size_t>(cs) < seqs_.size()) {
            seqs_[static_cast<size_t>(cs)]
                .cancel_after_decoded_tokens = opts.cancel_after;
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
    if (stream_all_) {
        stream_receivers_.reserve(seqs_.size());
        for (size_t s = 0; s < seqs_.size(); s++) {
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
    std::lock_guard<std::mutex> lk(admitted_futures_mtx_);
    std::vector<hpx::future<request_result>> out;
    out.swap(admitted_futures_);
    return out;
}

std::vector<admitted_stream_handoff> engine::take_admitted_stream_handoffs() {
    std::lock_guard<std::mutex> lk(admitted_futures_mtx_);
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
    {
        std::lock_guard<hpx::spinlock> lk(inbox_mtx_);
        inbox_.push_back(std::move(msg));
    }
    trace::event(
        "request_submitted_external request=%d budget=%d "
        "arrival_source=external",
        rid, bud);
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
    if (!seq.stream_enabled || seq.stream_closed) return;
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

    if (!fulfill_promise(seq, request_status::cancelled, ttc_us)) {
        return false;
    }
    result_.cancelled_count++;

    trace::event(
        "cancel_future_fulfilled seq=%d status=cancelled ttc_us=%lld",
        seq.seq_id, static_cast<long long>(ttc_us));
    return true;
}

void engine::drain_external_inbox(int32_t iter) {
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
        // admitted_futures_mtx_ critical section — no new
        // std::mutex, no new spinlock, no new HPX
        // primitive, no new CLI flag. The lock guards
        // engine→main result-handoff metadata only; it
        // does NOT guard llama.cpp execution state.
        if (stream_all_
            && (src == admission_source::completion_freed
             || src == admission_source::cancel_freed)) {
            std::lock_guard<std::mutex> lk(
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
            std::lock_guard<std::mutex> lk(
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
            // section — no new mutex, no new spinlock, no
            // expansion of the locked region beyond the
            // existing result-handoff scope. No llama.cpp
            // state is touched here.
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
    return true;
}

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
        // Streaming Slice 3: admitted-stream handoff vector lives
        // under the same critical section as admitted_futures_.
        admitted_stream_handoffs_.clear();
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
        // fail-closed (design §3 invariant). Admission body lives
        // in the private engine::admit_one method (see below).

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
                           "cancel_freed", iter, mem)) {
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
                               "completion_freed", iter, mem)) {
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

void engine::finalize_wall_ms() {
    const auto now = std::chrono::steady_clock::now();
    const auto us  = std::chrono::duration_cast<std::chrono::microseconds>(
        now - t_start_).count();
    result_.metrics.wall_ms = static_cast<double>(us) / 1000.0;
}
