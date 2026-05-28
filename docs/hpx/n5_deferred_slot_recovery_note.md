# N5 — slot-recovery note (N5b fixed, N5a fixed)

Status: **N5b (shutdown liveness) fixed; N5a (slot recovery) fixed.** No
performance claim. This records the N5 design and both fixes.

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

## 3. Fix (applied — see §7)

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
  (the `finalize_and_fulfill` fix above). **Fixed** (see §7).
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

**This N5b fix did NOT itself address N5a.** It did not change
`finalize_and_fulfill` or touch the `cancel_freed` → natural-completion
slot-repush. N5a was fixed separately on top of the N5b fix (see §7),
which made a clean pre-fix red possible: a queued-but-unadmittable
Phase C now drains as `failed_reserved` on shutdown instead of hanging.
No performance claim.

## 7. N5a — fixed

Fixed on top of N5b, in `engine.cpp` only (`finalize_and_fulfill`). The
`else if` that returns a naturally-completed slot to `free_idle_` was
re-keyed from `admission_src == initial_idle` to the seq_id-range
predicate documented in §3:

    } else if (seq.seq_id >= static_cast<int32_t>(budgets_.size())) {
        free_idle_.push_back(seq.seq_id);
    }

Because `seq_id == slot index` and idle-origin slots occupy
`[budgets_.size(), seqs_.size())`, this recovers a slot regardless of
whether it was last admitted via `initial_idle` **or** `cancel_freed`,
closing the `cancel_freed` → natural-completion → later-admission leak.
Preload-active slots (`seq_id < budgets_.size()`) are unaffected, and
canonical pure-preloaded gates (`initial_idle_slots == 0`,
`budgets_.size() == seqs_.size()`) leave the branch inert — so the
anchors are unchanged. No `home_pool` state was added; `cancel_and_fulfill`
and the N5b shutdown behavior were not touched.

Regression guard:
`engine_cancel_freed_completion_admit_smoke`
(`llama-hpx-engine-cancel-freed-completion-admit-smoke`). The smoke
classifies pre-fix vs post-fix on one single-slot engine: Phase A active
cancel → Phase B `cancel_freed` natural completion → Phase C admission
probe. Pre-fix, Phase C never admits and resolves `failed_reserved` on
the N5b shutdown drain (controlled FAIL, clean join). Post-fix, Phase C
admits from the recovered `free_idle_` slot and completes with the
canonical b8 hash `0x0619d4d1900c2365`, `admitted_count == 3`.

Verified: smoke 1× + 10/10 PASS; N5b shutdown smoke PASS
(`failed_reserved` intact); active-cancel, queued-cancel, and
keepalive-multi-submit regressions PASS; default and placement-on
(`--engine-pool --hpx-os-threads 2`) gates PASS with p0_b8 unchanged at
`0x0619d4d1900c2365`; `llama-hpx-server-engine-pool-smoke` PASS. No
performance claim.

## 8. Final N5a + Exp14 W3 refresh report

### N5a pre-fix red

- Smoke: `llama-hpx-engine-cancel-freed-completion-admit-smoke`.
- Phase A active-cancelled while decoding: `status=cancelled`,
  `n_decoded=2`.
- Phase B admitted from `cancel_freed`, completed with hash
  `0x0619d4d1900c2365`.
- Phase C was not admitted pre-fix and was resolved as
  `failed_reserved` by the N5b shutdown drain.
- Clean failure: no hang, no SIGKILL, no `_Exit`.

### N5a fix

- File: `tools/hpx-continuous-batch-gate/engine.cpp`.
- Function: `finalize_and_fulfill`.
- Fallback changed from
  `seq.admission_src == admission_source::initial_idle` to
  `seq.seq_id >= static_cast<int32_t>(budgets_.size())`.
- Rationale: idle-origin slots occupy
  `[budgets_.size(), seqs_.size())`; this recovers a naturally
  completed idle-origin slot even if its last admission source was
  `cancel_freed`.
- Note: relies on today's `seq_id == slot index` invariant.

### N5a post-fix validation

- N5a smoke: 1× PASS, 10/10 PASS. Phase C now completes with hash
  `0x0619d4d1900c2365` and `admitted_count == 3`.
- N5b shutdown smoke still PASS.
- Focused regressions: queued-cancel PASS, active-cancel PASS,
  keepalive-multi-submit PASS.
- Default gate PASS, p0_b8 unchanged: `0x0619d4d1900c2365`.
- Placement-on gate PASS, p0_b8 unchanged, engine-pool path
  exercised.
- `hpx-server-engine-pool-smoke` PASS.

### Exp14 W3 refresh

- Harness: added `--w3-between-trial-mode {sleep,capfree}`. Default
  remains `sleep`. Capfree reuses `wait_for_cap_free` and records
  `w3_probe` rows; `summary.csv` skips `w3_probe` and `w3_sanity`.
- Capfree validation
  (`run_id=20260524-142251-n5-refresh-w3-capfree-validate`): PASS in
  all three modes.
- Sleep-vs-capfree comparison
  (`run_id=20260524-142531-n5-refresh-w3-sleep-compare`,
  `run_id=20260524-142655-n5-refresh-w3-capfree-compare`): both PASS.
- In capfree mode every `wait_for_cap_free` probe returned true; max
  probe time ~72 ms vs the 30 s timeout. W3 sanity hash matched
  `0x0619d4d1900c2365`.

### Interpretation

- N5a and N5b are now fixed.
- The original Exp14 W3 cap-free failure path is no longer observed
  on this harness shape.
- Capfree gives cleaner W3 measurements by avoiding queue-wait
  contamination.
- This is correctness/harness evidence, not a performance claim.
- `sleep` remains the Phase 1 default for compatibility with
  recorded Phase 1 results.
- No llama-server comparison was performed.
- No full Exp14 rebaseline was performed.
- No concurrent-client claim.
