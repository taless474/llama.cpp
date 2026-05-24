# N5 — slot-recovery note (N5b fixed, N5a deferred)

Status: **N5b (shutdown liveness) fixed; N5a (slot recovery) still
deferred.** No performance claim. This records the N5 design so N5a can
be resumed cleanly.

## 1. Observed bug

On an idle-pool engine (`reuse_completed=false`, `initial_idle_slots>0`),
this sequence leaves a later request stuck:

    active cancel  ->  cancel_freed admission  ->  natural completion
                   ->  later request not admitted

A request admitted from the cancel-freed slot completes normally, after
which a subsequent `submit_request` reaches `arrival_drained` +
`request_queued` but is never admitted.

## 2. Suspected root cause

`finalize_and_fulfill` only returns **initial_idle**-admitted slots to
`free_idle_` when `reuse_completed_ == false`. A slot whose
`admission_src` is `cancel_freed` (or `completion_freed`) that then
completes naturally matches neither branch, so its `seq_id` is dropped
from every admission pool (`free_due_to_cancel_`,
`free_due_to_completion_`, `free_idle_`). The slot is leaked.

## 3. Proposed future fix

Return idle-origin slots to `free_idle_` using a per-slot predicate
instead of the most-recent `admission_src`. Idle slots occupy seq_ids
`>= budgets_.size()` today, so:

    else if (seq.seq_id >= static_cast<int32_t>(budgets_.size())) {
        free_idle_.push_back(seq.seq_id);
    }

This is more precise than an engine-wide size check and preserves
canonical pure-preloaded gate shapes (`initial_idle_slots == 0`, all
seq_ids `< budgets_.size()` → branch inert). Relies on today's
`seq_id == slot index` invariant.

## 4. Why it was not implemented now

A clean pre-fix reproducer was built and confirmed Phase A (active
cancel) and Phase B (cancel_freed → natural completion), and that a
later Phase C request does not complete pre-fix. But its cleanup
exposed a **second, separate** problem: when Phase C is queued but
unadmittable, `request_shutdown()` / `engine_fut.get()` did not wake or
join the engine task (hang on shutdown). Applying the slot-recovery fix
now would mask that issue rather than establish a clean red first.

## 5. Split

- **N5a** — slot recovery after `cancel_freed` → natural completion
  (the `finalize_and_fulfill` fix above). **Still deferred, not fixed.**
- **N5b** — `request_shutdown` / engine-join behavior when a request is
  queued but unadmittable (engine-task liveness, independent of N5a).
  **Fixed** (see §6).

## 6. N5b — fixed

Isolated and fixed. The shutdown-liveness bug was the outer-tail
shutdown-break in `run_body` being gated on
`waiting_queue_consumable_.empty()`: a queued-but-unadmittable request
kept the queue non-empty, so `staged_shutdown_` was never observed and
the engine re-parked in `wait_inbox_blocking()` forever.

Fix (engine.cpp / engine.h only):

- The outer-tail shutdown break now fires when `staged_shutdown_` is set
  and there are no active sequences and no staged work — it no longer
  requires an empty waiting queue.
- Before breaking, `drain_waiting_queue_for_shutdown()` resolves every
  still-queued request via `fulfill_queued_shutdown_aborted(...)` with
  status `request_status::failed_reserved` (shutdown-aborted queued
  work, explicitly **not** user `cancelled`), closing any queued stream
  with `reason=error`, and erasing the matching `external_promises_` /
  `external_stream_channels_` / `live_epoch_by_rid_` entries so the
  run-end `external_promises_`-empty guard holds. No queued promise is
  left unfulfilled.
- With an empty waiting queue (every prior gate/smoke shape) the drain
  is a no-op and the break fires exactly as before — canonical anchors
  (p0_b8 `0x0619d4d1900c2365`) are unchanged, verified on the default
  and placement-on gates.

Regression guard: `engine_shutdown_queued_unadmittable_smoke`
(`llama-hpx-engine-shutdown-queued-unadmittable-smoke`).

**This N5b fix does NOT address N5a.** It does not change
`finalize_and_fulfill`, does not touch the `cancel_freed` → natural
completion slot-repush, and makes no claim about slot recovery. N5a
remains deferred; the unaccepted N5a draft reproducer is archived under
`local/n5_deferred/` (gitignored). No performance claim.
