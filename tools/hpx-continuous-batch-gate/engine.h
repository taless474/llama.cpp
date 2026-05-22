// HPX continuous-batch engine class for the gate binary.
//
// Ownership boundary: exactly one HPX task per engine instance may
// touch llama.cpp mutable execution state (`llama_context`,
// `llama_batch`, `llama_decode`, `llama_memory_seq_*`,
// `llama_get_logits_ith`). The engine task is the one spawned via
// `hpx::async([&eng]{ eng.run(); })` in main; no other HPX task may
// call any engine method that mutates llama.cpp state.
//
// Public surface used by main + submitter, classified per M4a buckets
// (see engine_options below for the bucket definitions):
//   LIBRARY     — stable runtime API: `engine(engine_options)`, `run()`,
//                 `submit_request()`, `cancel_request()`,
//                 `request_shutdown()`, `result()`, `n_seqs()`.
//   GATE_COMPAT — temporary public compatibility methods used by the
//                 existing gate/preload harness; intended to move
//                 behind an adapter later: `take_futures()`,
//                 `take_admitted_futures()`,
//                 `take_admitted_stream_handoffs()`,
//                 `take_stream_receivers()`.
//   GATE_TEST   — deterministic gate/scripted-test hooks:
//                 `submit(arrival_msg)`,
//                 `register_external_release_iter(K)`.
// PRELOAD applies to ctor-time fields only (see engine_options).
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
#include <hpx/synchronization/condition_variable.hpp>

#include <chrono>
#include <cstdint>
#include <deque>
#include <optional>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// ---- engine_options ----------------------------------------------------
//
// Construction-time options for the engine, organized into three nested
// sub-structs that reify the M4a classification (LIBRARY / PRELOAD /
// GATE_TEST). The GATE_COMPAT bucket continues to apply to *methods*
// only — see the file-top public-surface comment.
//
//   lib       (library_options)   — stable runtime API surface a
//                                   submit_request consumer needs.
//   preload   (preload_options)   — ctor-time seeding for the gate /
//                                   preload harness (initial actives,
//                                   waiting queue, prompt borrows,
//                                   reuse + stream_all latches).
//                                   Meaningful only when the consumer
//                                   seeds work at construction time
//                                   (gate --n-active / --n-waiting /
//                                   prompt-file) or opts into preload-
//                                   only behavior (reuse_completed,
//                                   stream_all). Normal library users
//                                   go through
//                                   submit_request(want_stream=true)
//                                   instead.
//   gate_test (gate_test_options) — deterministic gate/test-only
//                                   plumbing (scripted cancel,
//                                   scripted release/ack barrier at
//                                   iter K, bounded max_decode_iters).
//                                   Not part of the library surface.
//
// Pointer fields (`lib.ctx`, `lib.vocab`, `preload.prompt_tokens`,
// `preload.waiting_queue`, `preload.per_active_prompt_tokens`) are
// *borrowed* — the engine stores either the pointer directly or a
// const-reference bound to `*ptr`, so the caller must keep the pointee
// alive for the engine's lifetime. Engine construction throws
// `std::invalid_argument` if any *required* pointer is null.
//   `lib.ctx`, `lib.vocab`         — always required (non-null).
//   `preload.waiting_queue` (M4b)  — optional. Null is treated as an
//                                    empty preloaded waiting queue.
//   `preload.prompt_tokens` (M4c)  — optional iff `preload.budgets` is
//                                    empty. Required (non-null) when
//                                    `preload.budgets` is non-empty,
//                                    because preloaded active requests
//                                    must be seeded with a shared
//                                    prompt.
//
// Value fields (`preload.budgets`, `gate_test.cancel_plan`,
// `gate_test.release_iter_set`) are copied into engine-owned storage
// during construction.
struct engine_options {
    // M4e: library bucket. Stable runtime API surface a submit_request
    // consumer needs. Required pointers (ctx, vocab) are validated in
    // the engine ctor and reported as
    // "engine_options::lib.ctx must be non-null" etc.
    struct library_options {
        llama_context *     ctx                = nullptr;
        const llama_vocab * vocab              = nullptr;
        int32_t             n_vocab            = 0;
        int32_t             batch_capacity     = 0;
        int32_t             n_seq_max          = 0;
        // M2f: count of additional seq slots reserved at engine startup
        // as an idle pool. Slots have ids
        // [budgets.size(), budgets.size() + initial_idle_slots) and
        // start done=true / decode_budget=0 / request_id=-1. They are
        // consumed by the admission step from `free_idle_` after
        // cancel-freed and completion-freed sources, allowing
        // submit_request to be admitted without any prior active
        // completion or cancellation. Default 0 preserves existing
        // behavior byte-for-byte. The engine ctor requires
        // `lib.n_seq_max >=
        //    preload.budgets.size() + lib.initial_idle_slots`.
        int32_t             initial_idle_slots = 0;
        // M3a: long-running keep-alive mode. With keep_alive=false
        // (default) the decode loop exits as soon as no active seq,
        // waiting queue empty, and inbox empty — byte-identical to M2.
        // With keep_alive=true the engine remains in run() and
        // idle-waits on an HPX condition variable when there is no work
        // to do; it exits only when `engine::request_shutdown()` has
        // been called AND all work is drained. No new behavior unless
        // the caller opts in.
        bool                keep_alive         = false;
    } lib;

    // M4e: preload bucket. Ctor-time seeding for the gate / preload
    // harness. submit_request-only consumers can leave these defaults
    // in place (null prompt_tokens, empty budgets, null waiting_queue,
    // null per_active_prompt_tokens, reuse_completed=false,
    // stream_all=false).
    struct preload_options {
        // M4c: optional iff `budgets` is empty. Null is allowed for
        // submit_request-only engines; the ctor rejects null whenever
        // `budgets` is non-empty (preloaded actives require a shared
        // prompt).
        const std::vector<llama_token> *      prompt_tokens = nullptr;
        std::vector<int32_t>                  budgets;
        // M4b: optional. Null is treated as an empty preloaded waiting
        // queue (no preloaded waiting requests). Normal library users
        // that only use submit_request() can leave this as nullptr.
        const std::vector<waiting_request> *  waiting_queue = nullptr;
        // M1d: optional per-active-seq prompt vectors. When non-null
        // and sized >= budgets.size(), the engine ctor seeds each
        // initial seq's prompt_tokens from the matching entry instead
        // of from the shared prompt_tokens borrow. When null (default),
        // the M1b shared-prompt seeding path is used and behavior
        // matches M1c byte-for-byte. Borrowed pointer — main must keep
        // the pointee alive for the engine's lifetime.
        const std::vector<std::vector<llama_token>> *
                                              per_active_prompt_tokens = nullptr;
        bool                                  reuse_completed = false;
        bool                                  stream_all      = false;
    } preload;

    // M4e: gate_test bucket. Deterministic gate/test-only plumbing
    // (scripted cancel, scripted release/ack barrier at iter K,
    // bounded max_decode_iters for the smoke shape). Not part of the
    // library surface; library consumers should leave these defaults
    // in place.
    struct gate_test_options {
        std::vector<int32_t> cancel_plan;
        int32_t              cancel_after     = -1;
        std::set<int32_t>    release_iter_set;
        int32_t              max_decode_iters = 0;
    } gate_test;
};

class engine {
public:
    // M4a: LIBRARY — stable engine construction.
    explicit engine(engine_options opts);

    // M4a: GATE_COMPAT — preloaded-actives future drain; future adapter target.
    // Pull the futures *before* scheduling run(). Called from main.
    // Returns the n_active "original" futures only; admitted-request
    // futures are created inside the engine task and collected via
    // take_admitted_futures() AFTER engine_fut.get().
    std::vector<hpx::future<request_result>> take_futures();

    // M4a: GATE_COMPAT — admitted-request future drain; future adapter target.
    // Live Admission Slice 3: drain the admitted-request futures the
    // engine produced. MUST be called from main only AFTER
    // engine_fut.get() returns — by that point every admitted promise
    // is already fulfilled, so each future is ready and the follow-on
    // hpx::wait_all in main is a no-op. The mutex matches the engine-
    // side push site; in practice both sides are sequenced by
    // engine_fut.get(), so contention is impossible.
    std::vector<hpx::future<request_result>> take_admitted_futures();

    // M4a: GATE_COMPAT — bulk admitted-stream handoff drain; future adapter target.
    // Streaming Slice 3: symmetric drain of the per-admission stream
    // handoff bundles. Called by main once per repeat, strictly AFTER
    // engine_fut.get() returns — by that point every admitted promise
    // is fulfilled and every receiver's channel has been closed by
    // close_stream(). The lock matches the engine-side push site and
    // is the same admitted_futures_mtx_ used for futures handoff; no
    // new lock is introduced. Returns an empty vector when --stream-all
    // is off or when no completion-freed admission fired.
    std::vector<admitted_stream_handoff> take_admitted_stream_handoffs();

    // M4a: GATE_COMPAT — preloaded per-seq stream-receiver drain; future adapter target.
    // Streaming Slice 8: pull the per-seq stream receivers collected at
    // engine construction time. Returns one receive_channel handle per
    // bound active seq (in seq_id order) when --stream-all is on; empty
    // when streaming is off. Main calls this once per repeat BEFORE
    // scheduling run(), the same way take_futures() works.
    std::vector<token_stream_receiver> take_stream_receivers();

    // M4a: GATE_TEST — scripted-arrival inbox push (engine-internal arrival_msg).
    // Live Admission Slice 6: async external arrival inbox push.
    // Called from the external submitter HPX task — never from the
    // engine task. Per Correction 1, this method is intentionally
    // read-only on engine_result: it only acquires the inbox spinlock,
    // moves the message into inbox_, and (optionally) emits a trace
    // event. Counters are bumped by the engine when it drains.
    // Hard rule: this body must not call any llama_* API. The grep
    // gate in main asserts that.
    void submit(arrival_msg msg);

    // M4a: LIBRARY — stable public submit API.
    // M2b: public submit API. Converts the public submit_request into
    // the engine-internal arrival_msg (with src=external and a fresh
    // engine-owned hpx::promise) and routes through the existing
    // submit(arrival_msg) inbox path. Returns a move-only handle
    // carrying the per-request HPX future. When req.want_stream is
    // true, the handle's `stream` member carries the receiver half of
    // a per-request HPX local channel; the engine moves the channel
    // object through the inbox into external_stream_channels_ and
    // then into the bound slot at admit_one (M2g). Safe to call from
    // any HPX task. Hard rule: this body must not call any llama_* API.
    // `struct submit_request` is an elaborated type specifier; it
    // disambiguates the parameter type from the enclosing member
    // function name (both spell `submit_request`).
    submit_handle submit_request(struct submit_request req);

    // M4a: LIBRARY — stable shutdown signal for keep-alive engines.
    // M3a: public shutdown signal for long-running keep-alive mode.
    // Sets the engine's `shutdown_requested_` flag under inbox_mtx_
    // and notifies inbox_cv_, so the engine task wakes from an idle
    // wait. Idempotent and safe to call from any HPX task (including
    // the same task that submitted requests). Does not touch any
    // llama_* state. When `engine_options::lib.keep_alive` was false this
    // method is harmless (the engine exits naturally on work drain);
    // when keep_alive was true, the engine drains any active work at
    // iteration boundaries and then exits the decode loop cleanly.
    // Never interrupts a llama_decode in progress.
    void request_shutdown();

    // M4a: LIBRARY — stable public cancellation.
    // M3b/M3c: public cancellation. Pushes `request_id` onto
    // cancel_inbox_ under inbox_mtx_ and notifies inbox_cv_. The
    // engine task drains cancel_inbox_ at iteration boundaries and
    // at the outer-loop tail, then matches the rid against
    // (a) arrivals not yet drained from inbox_ — M3b queued path:
    //     drain_external_inbox short-circuits the queue push and
    //     fulfill_queued_cancelled resolves the promise + stream
    //     with status=cancelled, n_decoded=0, kv_cleared=false,
    //     admitted_at_iter=-1;
    // (b) waiting requests not yet admitted from
    //     waiting_queue_consumable_ — M3b queued path: same
    //     fulfill_queued_cancelled, but the rid is pulled from the
    //     queue inside apply_queued_cancellations;
    // (c) live non-done active seqs — M3c active path:
    //     apply_queued_cancellations sets
    //     seq.cancel_requested.store(true, release) under the
    //     engine task, and the existing iter-boundary
    //     cancel_should_observe → cancel_and_fulfill pipeline
    //     clears KV, closes the stream with reason=cancelled,
    //     pushes the slot onto free_due_to_cancel_, and fulfills
    //     the promise with status=cancelled. No interruption
    //     inside llama_decode — observation is iter-boundary only.
    // Cancel-before-submit is supported: rids that match no
    // current request are held in cancelled_request_ids_ until a
    // matching arrival drains. Idempotent (duplicate calls bump
    // cancel_request_duplicates), non-blocking, callable from any
    // HPX task. Hard rule: no llama_* call sites; no KV mutation;
    // promise/stream fulfillment happens strictly on the engine
    // task.
    void cancel_request(int32_t request_id);

    // M5a: LIBRARY — epoch-aware cancellation. Matches both
    // `tok.request_id` AND `tok.epoch` against the engine's
    // engine-task-owned `live_epoch_by_rid_` map. Behavior versus the
    // legacy rid-only overload:
    //   (a) rid currently live with matching epoch — routes to the
    //       existing M3 queued/active cancel pipeline, byte-identical
    //       to the legacy overload for the same request.
    //   (b) rid currently live with NON-matching epoch — stale token
    //       for an older incarnation. Bumps
    //       `engine_result::cancel_stale_epoch` and does NOT cancel
    //       the live request. Caller's stale handle observes whatever
    //       outcome the live request reaches naturally.
    //   (c) rid not currently live (already finalized, never
    //       submitted) — bumps `cancel_unknown_request_id`, identical
    //       to the legacy overload's behavior in the same case.
    // Idempotent: pushing a `(rid, epoch)` already present in the
    // engine-task-owned `cancelled_tokens_` set bumps
    // `cancel_request_duplicates`. Token-aware duplicate detection is
    // keyed by `(rid, epoch)` and is independent of the legacy
    // rid-only set — calling the legacy `cancel_request(rid)` does
    // NOT inhibit a subsequent token-aware cancel with the same rid,
    // and vice versa. Non-blocking, callable from any HPX task. Hard
    // rule: no llama_* call sites; no KV mutation; promise/stream
    // fulfillment happens strictly on the engine task at the existing
    // iter boundaries.
    void cancel_request(const cancel_token & tok);

    // M4a: GATE_TEST — scripted release/ack barrier at iter K.
    // Live Admission Slice 6: pre-run registration of a release+ack
    // barrier at decode iter K. Must be called once per watched K
    // BEFORE the engine task is scheduled (so the engine sees a
    // ready ack_future to .get() on at end of iter K). Returns the
    // matching release_future (for the submitter to await) and the
    // ack_promise (for the submitter to set_value() after pushing all
    // K-arrivals). The ack_promise is move-only and ownership transfers
    // out of the engine here.
    external_release_handle register_external_release_iter(int32_t K);

    // M4a: LIBRARY — stable engine entry point.
    // Sole entry point that touches llama_context / llama_batch /
    // llama_decode / llama_memory_seq_* / llama_get_logits_ith.
    // Designed to run inside exactly one HPX task per engine instance.
    void run();

    // M4a: LIBRARY — stable result / sizing accessors.
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

    // M3b: cancel_inbox_ → cancelled_request_ids_ transfer. Engine
    // task only. Acquires inbox_mtx_ to swap cancel_inbox_ into a
    // local deque, then folds every rid into cancelled_request_ids_
    // (unordered_set: duplicates collapse, bumping
    // cancel_request_duplicates if a rid was already present).
    // Bumps cancel_request_calls once per drained entry. No llama
    // call sites; no promise fulfillment here — see
    // apply_queued_cancellations() and the drain_external_inbox()
    // arrival-side guard.
    void drain_cancel_inbox();

    // M3b/M3c: walk cancelled_request_ids_ and resolve each rid
    // against (1) waiting_queue_consumable_ — match removes the
    // entry, pulls promise + optional stream channel from
    // external_promises_ / external_stream_channels_, calls
    // fulfill_queued_cancelled, and erases rid from the set
    // (M3b queued path); (2) live non-done active seqs — sets
    // seq.cancel_requested.store(true, release) under the engine
    // task, bumps cancel_active_observed, emits trace, and erases
    // rid (M3c active path; the existing iter-boundary
    // cancel_should_observe → cancel_and_fulfill pipeline performs
    // the KV clear, stream close with reason=cancelled, and
    // promise fulfillment); (3) no match — leaves rid in the set
    // so a later arrival can still be resolved as queued
    // cancellation (cancel-before-submit). Engine task only. No
    // llama call sites; no KV mutation.
    void apply_queued_cancellations();

    // M3b: build a request_result with status=cancelled / n_decoded=0,
    // close the optional stream with reason=cancelled, and fulfill
    // the moved-in promise exactly once. Engine task only. Used by
    // both the drain-time arrival guard and apply_queued_cancellations
    // for the post-drain queue case. No llama call sites; no KV
    // mutation; no seq_state involved. Returns true on successful
    // set_value, false if set_value throws (the caller then stamps
    // result_.error per the existing fail-closed pattern).
    bool fulfill_queued_cancelled(
        int32_t                             request_id,
        int32_t                             decode_budget,
        hpx::promise<request_result>        promise,
        std::optional<token_stream_channel> channel,
        const char *                        from_label);

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
    // M4c: held as a pointer (was a reference). May be null when
    // budgets_ is empty (submit_request-only engines); guaranteed
    // non-null whenever budgets_ is non-empty by the ctor guard.
    const std::vector<llama_token> * prompt_tokens_;
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
    // every idle slot allocated via engine_options::lib.initial_idle_slots.
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
    // already serializes admitted-future handoff. As of M4d that
    // primitive is hpx::spinlock (matches inbox_mtx_); this is
    // gate-local synchronization around result handoff only, never
    // around llama.cpp execution state. Populated by admit_one when
    // stream_all_ is on and
    // src == admission_source::completion_freed; drained by main via
    // take_admitted_stream_handoffs() strictly after engine_fut.get().
    std::vector<admitted_stream_handoff>     admitted_stream_handoffs_;
    // M4d: was std::mutex; swapped to hpx::spinlock to match the
    // HPX-native primitive already used by inbox_mtx_. Guards the
    // same gate-local result-handoff vectors as before; no behavior
    // change beyond the primitive type.
    hpx::spinlock                    admitted_futures_mtx_;
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
    // M3a: condition variable paired with inbox_mtx_ for keep-alive
    // mode. condition_variable_any accepts any BasicLockable, so
    // inbox_mtx_ stays an hpx::spinlock (the existing four
    // lock-guard sites are unchanged). Both `submit()` and
    // `request_shutdown()` notify on this cv; the engine task waits
    // on it via the predicated overload only when keep_alive_ is on
    // and there is no other work to do. Engine-task-only mutator
    // outside the lock; lock-guarded mutators are documented below.
    hpx::condition_variable_any                     inbox_cv_;
    // M3a: public shutdown signal. Guarded by inbox_mtx_ on every
    // writer and on the cv waiter's predicate check; readers outside
    // the wait (the decode-loop while-predicate and the
    // post-no-active continue path) read it through the same lock
    // path the inbox itself uses (`inbox_has_pending()` already
    // acquires inbox_mtx_, so reading shutdown_requested_ right
    // before/after it pays for the lock once per iter at most).
    // request_shutdown() sets it under inbox_mtx_ and notifies the
    // cv; run_body's per-repeat reset clears it.
    bool                                            shutdown_requested_ = false;
    // M3a: keep-alive latch, copied once from engine_options::
    // lib.keep_alive at construction. With keep_alive_=false the decode
    // loop is byte-identical to M2. With keep_alive_=true the loop
    // does not exit on natural work drain and the engine idle-waits
    // on inbox_cv_ until new work arrives or shutdown is requested.
    bool                                            keep_alive_         = false;
    std::deque<arrival_msg>                                      inbox_;
    // M3b: queued-before-admission cancellation transport. cancel_inbox_
    // is the producer-side deque that engine::cancel_request() pushes
    // request_ids onto under inbox_mtx_ (sharing the existing M3a
    // lock); the same notify on inbox_cv_ wakes a keep-alive engine.
    // cancelled_request_ids_ is the engine-task-only persistent set
    // populated by drain_cancel_inbox(). Both are cleared in
    // run_body's per-repeat reset. The cv-wait predicate and the
    // drained-shutdown predicate both treat cancel_inbox_ as work,
    // so cancellations can never be lost across an idle suspend.
    std::deque<int32_t>                                          cancel_inbox_;
    std::unordered_set<int32_t>                                  cancelled_request_ids_;
    // M5a: epoch-aware cancellation transport. Sits alongside the
    // legacy M3b rid-only structures above; no field on those is
    // repurposed. `cancel_token_inbox_` is the producer-side deque
    // that `engine::cancel_request(const cancel_token &)` pushes
    // tokens onto under the existing `inbox_mtx_`; the same notify
    // on `inbox_cv_` wakes a keep-alive engine. `cancelled_tokens_`
    // is the engine-task-only persistent set populated by
    // `drain_cancel_inbox()` after the legacy rid drain — duplicate
    // `(rid, epoch)` insertions bump `cancel_request_duplicates`.
    // Both are cleared in `run_body`'s per-repeat reset. The cv-wait
    // predicate and the drained-shutdown predicate treat
    // `cancel_token_inbox_` as work alongside the legacy
    // `cancel_inbox_`, so token cancellations cannot be lost across
    // an idle suspend.
    std::deque<cancel_token>                                     cancel_token_inbox_;
    std::set<std::pair<int32_t, uint64_t>>                       cancelled_tokens_;
    // M5a: engine-task-owned "currently live" map keyed by rid. An
    // entry exists iff the rid has an arrival that has been drained
    // from the inbox and not yet finalized (queued in
    // waiting_queue_consumable_ OR admitted to a seq AND not yet
    // completed/cancelled). The value is the engine-issued epoch of
    // the live submission. Inserted in `drain_external_inbox` when
    // `arrival_msg::epoch != 0` (engine-issued tokens only — the
    // gate's `engine::submit(arrival_msg)` direct path leaves epoch
    // at 0 and never touches this map). Erased GUARDED at finalize /
    // cancel / queued-cancel fulfill sites: an entry is removed only
    // if `live_epoch_by_rid_[rid] == seq.epoch` (or the equivalent
    // pre-admission epoch). The guard prevents a late completion of
    // an older same-rid incarnation from wiping the live entry of a
    // newer submission. Engine task only; no locking.
    std::unordered_map<int32_t, uint64_t>                        live_epoch_by_rid_;
    // M5a: monotonic engine-issued epoch source. Bumped under
    // `inbox_mtx_` inside `engine::submit_request` so the same lock
    // that serializes inbox pushes also serializes epoch issuance.
    // Starts at 1; the value 0 is the "unset" sentinel reserved for
    // default-constructed `cancel_token` and for arrival_msg /
    // seq_state / waiting_request paths that never went through
    // `submit_request`. Monotonic for the lifetime of the engine
    // (not reset across `--repeat` iterations) so tokens issued on
    // one repeat cannot collide with submissions on the next.
    uint64_t                                                     next_epoch_ = 1;
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
