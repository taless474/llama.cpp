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
// All private members (queues, the per-seq state vector, the
// admitted-handoff vector, the HPX inbox channel + staged deques,
// the release/ack barrier slots, the streaming receivers vector,
// the per-engine result struct, etc.) are declared here because
// they must be visible to compile any TU that constructs an
// `engine`; their definitions and the bodies of every method live
// in `engine.cpp`.

#pragma once

#include "types.h"

#include "llama.h"

#include <hpx/hpx.hpp>
#include <hpx/lcos_local/channel.hpp>

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
        // N2.7: placement-selected cooperativity yield in
        // pump_inbox_nonblocking(). Default true matches the legacy
        // behavior for every default-pool spawn site (gate without
        // --engine-pool, all existing engine smokes, hpx-server): the
        // yield is the foreign-thread → engine cooperativity point
        // that lets producer/cancel tasks make progress under
        // os_threads=1. When the caller spawns this engine on the
        // named HPX `engine` pool via hpx_runtime::async_on_engine
        // (i.e. start_once was called with enable_engine_pool=true),
        // set this to false: N2.6b/N2.6c proved that
        // hpx::this_thread::yield() on a single-PU named pool causes
        // a scheduler-livelock that prevents channel observation
        // from foreign-thread producers. The decision is made at the
        // spawn site (Layer 2 placement); the engine itself does not
        // query runtime placement state.
        bool                cooperative_yield_on_pump = true;
        // N3.1 (Exp 13): opt into control-plane responsiveness
        // timestamp collection on `request_result.t_admitted_us`,
        // `t_first_publish_us`, `t_complete_us`, and
        // `t_cancel_observed_us`. Default false matches every
        // existing call site (smokes, gate, hpx-server, M8 configs)
        // byte-for-byte: with the option off, no `steady_clock::now()`
        // calls are added on any hot engine path, and all four
        // request_result fields stay -1.
        // Set this true ONLY from the dedicated responsiveness
        // benchmark binary so existing gates do not pay extra clock
        // reads.
        bool                enable_responsiveness_timing = false;
        // Slice C: per-seq chunked-prefill row budget for the live-
        // admission build path (engine.cpp iter_build_batch). Default 0
        // means unbounded — the whole prompt is placed in the admission
        // iter, preserving current observable behavior (same smokes, same
        // b8 anchor, same streaming/cancel behavior, same diagnostics).
        // A value B > 0 caps each iter at B prompt rows for a given seq,
        // so a long prompt is prefilled across multiple iters; the seq
        // is not sampled until its prefill completes. Consulted ONLY in
        // the live-admission Path A; the preloaded whole-prompt path is
        // unchanged and ignores this. No server CLI/env surface in this
        // slice — set directly via engine_options. Values <= 0 are
        // treated as unbounded. See
        // docs/hpx/prefill_budget_policy_design.md §4.
        int32_t             prefill_budget_rows = 0;
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
    // read-only on engine_result: N1: it publishes a `submission`
    // inbox_msg onto the HPX inbox channel and emits a trace event.
    // Counters are bumped by the engine when it drains. Hard rule:
    // this body must not call any llama_* API. The grep gate in
    // main asserts that.
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
    // N1: publishes a `shutdown` inbox_msg onto the HPX inbox channel.
    // The engine task observes it during its next pump (inner-loop
    // predicate via inbox_has_pending, or outer-loop tail pump). Same
    // iteration-boundary semantics as before. Idempotent: multiple
    // shutdown messages are folded into the engine-task-only
    // staged_shutdown_ latch by stage_one(). Safe to call from any
    // HPX or foreign thread. Does not touch any llama_* state. When
    // `engine_options::lib.keep_alive` was false the method is
    // harmless (the engine exits naturally on work drain); when
    // keep_alive was true, the engine drains any active work at
    // iteration boundaries and then exits the decode loop cleanly.
    // Never interrupts a llama_decode in progress.
    void request_shutdown();

    // M4a: LIBRARY — stable public cancellation.
    // M3b/M3c: public cancellation. N1: publishes a `cancel_rid`
    // inbox_msg onto the HPX inbox channel. The engine task pumps
    // it into staged_cancel_rids_ at the next predicate-pump site;
    // drain_cancel_inbox() promotes the rid into
    // cancelled_request_ids_ at iteration boundaries and at the
    // outer-loop tail, then matches the rid against
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

    // M3b + M5a: staged-cancel → cancelled_request_ids_ /
    // cancelled_tokens_ transfer. Engine task only. N1: swaps
    // staged_cancel_rids_ and staged_cancel_tokens_ (already pumped
    // off the HPX inbox channel by inbox_has_pending() or the
    // outer-loop pump) into local deques, then folds every rid /
    // token into the engine-task-only sets (unordered_set / set:
    // duplicates collapse, bumping cancel_request_duplicates if
    // already present). Bumps cancel_request_calls once per drained
    // entry. No channel pump inside this function — pump happens
    // only at predicate sites, so cancels arriving on the channel
    // during this drain stay buffered until the next pump and are
    // observed only at the next cancel phase. No llama call sites;
    // no promise fulfillment here — see apply_queued_cancellations()
    // and the drain_external_inbox() arrival-side guard.
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

    // N5b: shutdown-aborted queued-request resolver. Engine task only.
    // Modeled on fulfill_queued_cancelled but resolves with status=
    // failed_reserved (shutdown-aborted queued work, NOT user
    // cancellation) and closes any queued stream with reason=error. No
    // KV touch, no llama API. Returns true on successful set_value,
    // false if set_value throws (caller stamps result_.error).
    bool fulfill_queued_shutdown_aborted(
        int32_t                             request_id,
        int32_t                             decode_budget,
        hpx::promise<request_result>        promise,
        std::optional<token_stream_channel> channel,
        const char *                        from_label);

    // N5b: drain ALL queued-but-unadmittable requests on shutdown.
    // Engine task only. For each entry in waiting_queue_consumable_,
    // moves its promise out of external_promises_ and any stream channel
    // out of external_stream_channels_, erases its live_epoch_by_rid_
    // entry, and resolves it via fulfill_queued_shutdown_aborted. Leaves
    // waiting_queue_consumable_ empty so the outer-tail shutdown break
    // can no longer be suppressed by a stranded queued request.
    void drain_waiting_queue_for_shutdown();

    // M2f: engine-task-only. N1: pumps the HPX inbox channel
    // non-blocking into the staged_* deques, then reports whether
    // any staged arrival is now visible. Non-const because the pump
    // mutates pending_msg_ and staged_arrivals_. Must NOT be called
    // from foreign/server threads — staged_* deques are unguarded
    // and accessed exclusively by run().
    bool inbox_has_pending();

    // N3.0: per-iter phase helpers extracted from run_body()'s
    // inner-while body. Engine task only. Call order each iter:
    //   P1  pump_inbox_nonblocking
    //   P2  drain_cancel_inbox
    //   P3  drain_external_inbox(iter)
    //   P4  apply_queued_cancellations
    //   P5  admission_eligible_count snapshot   (inline in run_body)
    //   P6  iter_observe_cancellations
    //   P7  iter_run_admissions
    //   P8  admission metrics + !any_active break (inline in run_body)
    //   P9  iter_build_batch
    //   P10 iter_run_decode
    //   P11 iter_sample_and_finalize
    //   P12 iter_fire_release_ack_barrier
    // Bodies are verbatim moves of the prior inline blocks. Helpers
    // return false / ok=false on bail; the orchestrator performs the
    // shared llama_batch_free + update_iterations + finalize_wall_ms
    // cleanup at each call site (matches the legacy bail pattern).
    struct iter_admission_result {
        int32_t admitted = 0;
        bool    ok       = true;
    };

    bool iter_observe_cancellations(int32_t iter, llama_memory_t mem);

    iter_admission_result iter_run_admissions(
        int32_t iter, llama_memory_t mem,
        size_t admission_eligible_count);

    int32_t iter_build_batch(int32_t iter, llama_batch & batch,
                             std::vector<int32_t> & active_idx);

    bool iter_run_decode(llama_batch & batch,
                         const std::vector<int32_t> & active_idx);

    bool iter_sample_and_finalize(
        int32_t iter,
        const std::vector<int32_t> & active_idx,
        llama_memory_t mem);

    bool iter_fire_release_ack_barrier(int32_t iter);

    void run_body();

    void finalize_wall_ms();

    // N1: HPX inbox protocol — engine-internal types and helpers.
    // inbox_msg is a tagged-union over the four producer kinds. It
    // is move-only (because arrival_msg is move-only) and travels
    // through hpx::lcos::local::channel<inbox_msg>. The type is a
    // private nested member so the wire format never leaks into
    // public engine headers.
    enum class inbox_msg_kind : uint8_t {
        submission   = 0,
        cancel_rid   = 1,
        cancel_token = 2,
        shutdown     = 3,
    };
    struct inbox_msg {
        inbox_msg_kind             k       = inbox_msg_kind::shutdown;
        std::optional<arrival_msg> arrival;        // valid iff k == submission
        int32_t                    rid     = -1;   // valid iff k == cancel_rid
        ::cancel_token             token   = {};   // valid iff k == cancel_token
    };

    // pump_inbox_nonblocking: pull every channel message that is
    // currently ready into the staged_* deques. Engine-task-only.
    // Maintains the invariant that pending_msg_ is a valid future
    // for the next-not-yet-arrived message at function exit.
    void pump_inbox_nonblocking();

    // wait_inbox_blocking: park the engine HPX task on the channel
    // until at least one new message arrives, stage it, then drain
    // any follow-up ready messages. Engine-task-only. Replaces the
    // old condition_variable_any wait.
    void wait_inbox_blocking();

    // stage_one: route a single inbox_msg into the appropriate
    // staged_* deque (or set the staged_shutdown_ latch). Engine-
    // task-only.
    void stage_one(inbox_msg m);

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
    // primitive is hpx::spinlock; this is gate-local synchronization
    // around result handoff only, never around llama.cpp execution
    // state. Populated by admit_one when stream_all_ is on and
    // src == admission_source::completion_freed; drained by main via
    // take_admitted_stream_handoffs() strictly after engine_fut.get().
    std::vector<admitted_stream_handoff>     admitted_stream_handoffs_;
    // M4d: was std::mutex; swapped to hpx::spinlock to keep the
    // result-handoff synchronization HPX-native. Guards the same
    // gate-local result-handoff vectors as before; no behavior
    // change beyond the primitive type.
    hpx::spinlock                    admitted_futures_mtx_;
    // N1: HPX-native async inbox. Replaces the M3a/M3b/M5a
    // spinlock+cv+deque triad with a single move-only-capable HPX
    // local channel. Producers (submit, submit_request,
    // cancel_request rid/token, request_shutdown) publish typed
    // inbox_msg values into inbox_chan_; the engine task pumps the
    // channel into the engine-task-only staged_* deques at every
    // predicate-evaluation site (inbox_has_pending() and the outer-
    // loop tail). drain_external_inbox / drain_cancel_inbox swap
    // from staged_* into local deques without pumping, so cancels
    // arriving on the channel mid-arrival-drain stay buffered until
    // the next predicate pump — preserving the existing
    // iteration-boundary cancel-then-arrival phase order.
    //
    // pending_msg_ is the next-message future used by
    // pump_inbox_nonblocking / wait_inbox_blocking. Engine-task-
    // only; not accessed outside run().
    //
    // submit_publish_mtx_ is held briefly around the next_epoch_
    // bump AND the channel publish inside engine::submit_request().
    // This preserves M5a's "drain order matches epoch order"
    // invariant for concurrent same-rid submissions without
    // changing live_epoch_by_rid_ semantics. Other producers
    // (cancel rid/token, request_shutdown, engine::submit) do not
    // bump next_epoch_ and do not acquire this lock.
    hpx::lcos::local::channel<inbox_msg>            inbox_chan_;
    hpx::future<inbox_msg>                          pending_msg_;
    std::deque<arrival_msg>                         staged_arrivals_;
    std::deque<int32_t>                             staged_cancel_rids_;
    std::deque<cancel_token>                        staged_cancel_tokens_;
    bool                                            staged_shutdown_ = false;
    hpx::spinlock                                   submit_publish_mtx_;
    // M3a: keep-alive latch, copied once from engine_options::
    // lib.keep_alive at construction. With keep_alive_=false the decode
    // loop exits as soon as no active seq, waiting queue empty, and
    // inbox empty — byte-identical to M2. With keep_alive_=true the
    // loop does not exit on natural work drain and the engine
    // HPX-suspends on the inbox channel via wait_inbox_blocking()
    // until new work arrives or shutdown is requested.
    bool                                            keep_alive_         = false;
    // N2.7: copied once from engine_options::lib.cooperative_yield_on_pump
    // at construction. Default true matches legacy behavior for every
    // default-pool spawn site. The gate sets this to false when
    // --engine-pool is on so the engine task running on the named
    // single-PU engine pool skips the yield in pump_inbox_nonblocking
    // (N2.6c-proved scheduler livelock).
    bool                                            cooperative_yield_on_pump_ = true;
    // N3.1 (Exp 13): copied once from
    // engine_options::lib.enable_responsiveness_timing at construction.
    // Gates the four responsiveness timestamp writes
    // (t_admitted_us / t_first_publish_us / t_complete_us /
    // t_cancel_observed_us). False on every existing caller (smokes,
    // gate, hpx-server, M8 configs); set true only by the dedicated
    // responsiveness benchmark binary.
    bool                                            enable_responsiveness_timing_ = false;
    // Slice C: copied once from engine_options::lib.prefill_budget_rows
    // at construction. 0 (and any <= 0) means unbounded whole-prompt
    // prefill — current behavior. A value B > 0 caps the live-admission
    // prefill build (iter_build_batch) at B prompt rows per iter for a
    // seq, so prefill spans multiple iters and the seq is held out of
    // sampling until prefill_complete. Engine-task-only; read only on
    // the Path A build site.
    int32_t                                         prefill_budget_rows_ = 0;
    // M3b: queued-before-admission cancellation set, engine-task-
    // only. Populated by drain_cancel_inbox() from
    // staged_cancel_rids_; consumed by apply_queued_cancellations
    // and the drain_external_inbox() arrival-side guard. Cleared in
    // run_body's per-repeat reset.
    std::unordered_set<int32_t>                                  cancelled_request_ids_;
    // M5a: epoch-aware cancellation set, engine-task-only. Populated
    // by drain_cancel_inbox() from staged_cancel_tokens_. Cleared in
    // run_body's per-repeat reset.
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
    // M5a: monotonic engine-issued epoch source. N1: bumped under
    // `submit_publish_mtx_` inside `engine::submit_request`. The
    // same critical section also publishes the matching `submission`
    // inbox_msg onto `inbox_chan_`, so two concurrent
    // submit_request callers produce a totally ordered
    // (epoch, channel-position) pair. This preserves the M5a
    // assumption that drain order matches epoch order, keeping
    // `live_epoch_by_rid_` semantically correct without any
    // tracking changes. Starts at 1; the value 0 is the "unset"
    // sentinel reserved for default-constructed `cancel_token` and
    // for arrival_msg / seq_state / waiting_request paths that
    // never went through `submit_request`. Monotonic for the
    // lifetime of the engine (not reset across `--repeat`
    // iterations) so tokens issued on one repeat cannot collide
    // with submissions on the next.
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
