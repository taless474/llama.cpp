// HPX continuous-batch engine class for the gate binary.
//
// Ownership boundary: exactly one HPX task per engine instance may
// touch llama.cpp mutable execution state (`llama_context`,
// `llama_batch`, `llama_decode`, `llama_memory_seq_*`,
// `llama_get_logits_ith`). The engine task is the one spawned via
// `hpx::async([&eng]{ eng.run(); })` in main; no other HPX task may
// call any engine method that mutates llama.cpp state.
//
// Public surface used by main + submitter:
//   construction       — `explicit engine(const engine_options &)`
//   future drain       — `take_futures`, `take_admitted_futures`,
//                        `take_admitted_stream_handoffs`,
//                        `take_stream_receivers`
//   external arrival   — `submit(arrival_msg)`,
//                        `register_external_release_iter(int32_t K)`
//   engine entry point — `run()`
//   results / sizing   — `result()`, `n_seqs()`
//
// All private members (queues, mutexes, the per-seq state vector, the
// admitted-handoff vector, the inbox, the release/ack barrier slots,
// the streaming receivers vector, the per-engine result struct, etc.)
// are declared here because they must be visible to compile any TU
// that constructs an `engine`; their definitions and the bodies of
// every method live in `engine.cpp`.

#pragma once

#include "types.h"

#include "llama.h"

#include <hpx/hpx.hpp>
#include <hpx/lcos_local/channel.hpp>

#include <chrono>
#include <cstdint>
#include <deque>
#include <mutex>
#include <set>
#include <unordered_map>
#include <vector>

// ---- engine_options ----------------------------------------------------
//
// Construction-time options for the engine.
//
// Pointer fields (`ctx`, `vocab`, `prompt_tokens`, `waiting_queue`) are
// *borrowed* — the engine stores either the pointer directly or a
// const-reference bound to `*ptr`, so the caller must keep the pointee
// alive for the engine's lifetime. Engine construction throws
// `std::invalid_argument` if any required pointer is null.
//
// Value fields (`budgets`, `cancel_plan`, `release_iter_set`) are copied
// into engine-owned storage during construction.
struct engine_options {
    llama_context *                       ctx              = nullptr;
    const llama_vocab *                   vocab            = nullptr;
    int32_t                               n_vocab          = 0;
    const std::vector<llama_token> *      prompt_tokens    = nullptr;
    std::vector<int32_t>                  budgets;
    int32_t                               batch_capacity   = 0;
    std::vector<int32_t>                  cancel_plan;
    int32_t                               cancel_after     = -1;
    int32_t                               n_seq_max        = 0;
    const std::vector<waiting_request> *  waiting_queue    = nullptr;
    bool                                  reuse_completed  = false;
    std::set<int32_t>                     release_iter_set;
    int32_t                               max_decode_iters = 0;
    bool                                  stream_all       = false;
    // M1d: optional per-active-seq prompt vectors. When non-null and
    // sized >= budgets.size(), the engine ctor seeds each initial
    // seq's prompt_tokens from the matching entry instead of from
    // the shared prompt_tokens_ borrow. When null (default), the M1b
    // shared-prompt seeding path is used and behavior matches M1c
    // byte-for-byte. Borrowed pointer — main must keep the pointee
    // alive for the engine's lifetime.
    const std::vector<std::vector<llama_token>> *
                                          per_active_prompt_tokens = nullptr;
    // M2f: count of additional seq slots reserved at engine startup as
    // an idle pool. Slots have ids
    // [budgets.size(), budgets.size() + initial_idle_slots) and start
    // done=true / decode_budget=0 / request_id=-1. They are consumed by
    // the admission step from `free_idle_` after cancel-freed and
    // completion-freed sources, allowing submit_request to be admitted
    // without any prior active completion or cancellation. Default 0
    // preserves existing behavior byte-for-byte. The engine ctor
    // requires `n_seq_max >= budgets.size() + initial_idle_slots`.
    int32_t                               initial_idle_slots = 0;
};

class engine {
public:
    explicit engine(engine_options opts);

    // Pull the futures *before* scheduling run(). Called from main.
    // Returns the n_active "original" futures only; admitted-request
    // futures are created inside the engine task and collected via
    // take_admitted_futures() AFTER engine_fut.get().
    std::vector<hpx::future<request_result>> take_futures();

    // Live Admission Slice 3: drain the admitted-request futures the
    // engine produced. MUST be called from main only AFTER
    // engine_fut.get() returns — by that point every admitted promise
    // is already fulfilled, so each future is ready and the follow-on
    // hpx::wait_all in main is a no-op. The mutex matches the engine-
    // side push site; in practice both sides are sequenced by
    // engine_fut.get(), so contention is impossible.
    std::vector<hpx::future<request_result>> take_admitted_futures();

    // Streaming Slice 3: symmetric drain of the per-admission stream
    // handoff bundles. Called by main once per repeat, strictly AFTER
    // engine_fut.get() returns — by that point every admitted promise
    // is fulfilled and every receiver's channel has been closed by
    // close_stream(). The lock matches the engine-side push site and
    // is the same admitted_futures_mtx_ used for futures handoff; no
    // new lock is introduced. Returns an empty vector when --stream-all
    // is off or when no completion-freed admission fired.
    std::vector<admitted_stream_handoff> take_admitted_stream_handoffs();

    // Streaming Slice 8: pull the per-seq stream receivers collected at
    // engine construction time. Returns one receive_channel handle per
    // bound active seq (in seq_id order) when --stream-all is on; empty
    // when streaming is off. Main calls this once per repeat BEFORE
    // scheduling run(), the same way take_futures() works.
    std::vector<token_stream_receiver> take_stream_receivers();

    // Live Admission Slice 6: async external arrival inbox push.
    // Called from the external submitter HPX task — never from the
    // engine task. Per Correction 1, this method is intentionally
    // read-only on engine_result: it only acquires the inbox spinlock,
    // moves the message into inbox_, and (optionally) emits a trace
    // event. Counters are bumped by the engine when it drains.
    // Hard rule: this body must not call any llama_* API. The grep
    // gate in main asserts that.
    void submit(arrival_msg msg);

    // M2b: public submit API. Converts the public submit_request into
    // the engine-internal arrival_msg (with src=external and a fresh
    // engine-owned hpx::promise) and routes through the existing
    // submit(arrival_msg) inbox path. Returns a move-only handle
    // carrying the per-request HPX future; submit_handle.stream is
    // std::nullopt in M2b (req.want_stream=true throws — stream
    // wiring is deferred to M2c, where it can be added alongside the
    // gate's scripted-submitter migration). Safe to call from any
    // HPX task. Hard rule: this body must not call any llama_* API.
    // `struct submit_request` is an elaborated type specifier; it
    // disambiguates the parameter type from the enclosing member
    // function name (both spell `submit_request`).
    submit_handle submit_request(struct submit_request req);

    // Live Admission Slice 6: pre-run registration of a release+ack
    // barrier at decode iter K. Must be called once per watched K
    // BEFORE the engine task is scheduled (so the engine sees a
    // ready ack_future to .get() on at end of iter K). Returns the
    // matching release_future (for the submitter to await) and the
    // ack_promise (for the submitter to set_value() after pushing all
    // K-arrivals). The ack_promise is move-only and ownership transfers
    // out of the engine here.
    external_release_handle register_external_release_iter(int32_t K);

    // Sole entry point that touches llama_context / llama_batch /
    // llama_decode / llama_memory_seq_* / llama_get_logits_ith.
    // Designed to run inside exactly one HPX task per engine instance.
    void run();

    const engine_result & result() const noexcept { return result_; }
    int32_t n_seqs() const noexcept {
        return static_cast<int32_t>(seqs_.size());
    }

private:
    bool clear_and_check(seq_state & seq, int32_t iter,
                         llama_memory_t mem);

    void publish_token(seq_state & seq, int32_t token_id);

    void close_stream(seq_state & seq, stream_close_reason reason);

    bool fulfill_promise(seq_state &     seq,
                         request_status  status,
                         int64_t         ttc_us);

    bool finalize_and_fulfill(seq_state & seq, int32_t iter,
                              llama_memory_t mem);

    bool cancel_should_observe(const seq_state & seq) const noexcept;

    bool cancel_and_fulfill(seq_state & seq, int32_t iter,
                            llama_memory_t mem);

    void drain_external_inbox(int32_t iter);

    bool admit_one(int32_t              reuse_seq,
                   admission_source     src,
                   const char *         src_label,
                   int32_t              iter,
                   llama_memory_t       mem);

    // M2f: non-const because it acquires `inbox_mtx_` (the existing
    // engine-internal spinlock). Engine task only. Pure metadata
    // probe — no llama.cpp API call sites in the body.
    bool inbox_has_pending();

    void run_body();

    void finalize_wall_ms();

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
    // M2f: initial-idle slot reuse queue. Populated in the ctor (and
    // re-seeded in run_body's per-repeat reset) with the seq_ids of
    // every idle slot allocated via engine_options::initial_idle_slots.
    // Drained by the admission step AFTER free_due_to_cancel_ and
    // free_due_to_completion_, with no demand gate — idle slots have no
    // prior occupant to demand-pair against. Empty when
    // initial_idle_slots == 0, so existing gate runs are inert.
    std::deque<int32_t>              free_idle_;
    // Live Admission Slice 3: per-admission promise/future pairs created
    // inside the engine task. Main drains this after engine_fut.get()
    // returns; by that point every entry's promise is fulfilled, so a
    // follow-on wait_all is a no-op (design §11.3). The mutex is
    // defensive — the engine task is single-threaded and main reads
    // strictly after engine_fut.get() — but it makes the ordering
    // invariant explicit.
    std::vector<hpx::future<request_result>> admitted_futures_;
    // Streaming Slice 3: parallel per-admission stream handoff vector,
    // guarded by the SAME admitted_futures_mtx_ critical section that
    // already serializes admitted-future handoff. No new std::mutex and
    // no hpx::spinlock is introduced; this is gate-local synchronization
    // around result handoff only, never around llama.cpp execution
    // state. Populated by admit_one when stream_all_ is on and
    // src == admission_source::completion_freed; drained by main via
    // take_admitted_stream_handoffs() strictly after engine_fut.get().
    std::vector<admitted_stream_handoff>     admitted_stream_handoffs_;
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
    // M2g: engine-task-only stash of per-request stream channels
    // submitted via `submit_request(want_stream=true)`. Populated in
    // `drain_external_inbox` (moves the channel out of `arrival_msg`)
    // and consumed in `admit_one`'s external branch (moves the channel
    // into the bound `seq_state`). No mutex — every reader/writer
    // lives on the single engine task. Defensively cleared in
    // `run_body`'s per-repeat reset; the `run()` cleanup tail closes
    // any leftover channels with reason=error so the caller's receiver
    // cannot hang if the engine bails before admission.
    std::unordered_map<int32_t, token_stream_channel>            external_stream_channels_;
    std::vector<hpx::promise<void>>                              iter_release_promises_;
    std::vector<hpx::future<void>>                               submitter_ack_futures_;
    std::set<int32_t>                                            iter_release_set_;
    int32_t                                                      max_decode_iters_ = 0;
    // Streaming Slice 8: --stream-all latch, receiver vector populated
    // at ctor time and drained by main via take_stream_receivers().
    bool                                                         stream_all_ = false;
    std::vector<token_stream_receiver>                           stream_receivers_;
    engine_result                    result_;
    std::chrono::steady_clock::time_point t_start_{};
};
