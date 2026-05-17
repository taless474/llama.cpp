# Continuous batching — Streaming Slice 5 design

## 1. Purpose

Streaming Slice 5 closes the external-arrival admission arm
of the HPX-native per-request token stream on the
**completion-freed** slot only. Slices 1 and 2 gated
streaming on the completion and cancellation paths for
original-active seqs. Slice 3 extended streaming to
**preloaded** waiters admitted via the completion-freed
slot. Slice 4 extended streaming to **preloaded** waiters
admitted via the cancel-freed slot. Slice 5 extends
streaming to **external arrivals** — requests that entered
through the scripted HPX submitter task via
`engine::submit()` — admitted via the **completion-freed**
slot. After Slice 5, every admission path is streamed
end-to-end **except** external × cancel-freed, which is
deferred.

Slice 5 is a gate-only / orchestration-only advance. The
engine-side delta is one new rebind block at the end of the
external branch of `admit_one`, gated on
`stream_all_ && src == admission_source::completion_freed`.
No new HPX primitive, no new `std::mutex`, no new CLI flag,
no new trace event name, no new channel type, no CMake
change. The streaming substrate, ownership rules, payload
(`int32_t` token id + close reason), and trace event set
are unchanged from Slices 1–4.

This is a correctness/lifecycle gate. No performance claim
is made.

## 2. Relation to Streaming Slices 1–4

Slices 1–7 (the admission gate sequence,
`HPX_CB_ADMIT_STEP1` … `HPX_CB_ADMIT_STEP7`) remain a
carry-over invariant: every admission, cancellation, and
lifecycle invariant gated there still holds under
`HPX_CB_STREAM_STEP5`. The `--stream-all` OFF regression
gate on the canonical Slice 7 admission shape continues to
pass with all engine-side stream counters at exactly zero.

Streaming Slice 1 (`HPX_CB_STREAM_STEP1: PASS`) proved the
completion arm for an original active seq. Streaming Slice
2 (`HPX_CB_STREAM_STEP2: PASS`) proved the cancellation
arm for an original active seq. Streaming Slice 3
(`HPX_CB_STREAM_STEP3: PASS`) proved preloaded-waiter
streaming over a completion-freed slot. Streaming Slice 4
(`HPX_CB_STREAM_STEP4: PASS`) proved preloaded-waiter
streaming over a cancel-freed slot.

Streaming Slice 5 (`HPX_CB_STREAM_STEP5: PASS`) is the
external-arrival analogue of Slice 3: same admission
source (`completion_freed`), different arrival source
(`external` instead of `preloaded`). The Slice 1 / Slice 2
streaming invariants (single producer, single consumer,
terminal `kind=closed` before `channel.close()`, no
llama.cpp state crossing the channel, unbounded
`channel<T>`, trace-off quietness) and all Slice 3 / Slice
4 admitted-streaming invariants are strict carry-overs.
The Slice 3 / Slice 4 smokes and gates are unchanged.

## 3. Target invariant

An externally-arriving request admitted into a
completion-freed slot under `--stream-all` must:

```text
- open a fresh hpx::lcos::local::channel<token_stream_event>
  on the recycled seq_id (the prior occupant's closed
  channel must not be inherited);
- have stream_closed reset to false at admission;
- have the per-slot cumulative stream_tokens_emitted reset
  to zero at admission (so the admitted close trace
  reports the admitted-request count, not prev + admitted
  cumulative);
- emit exactly rr.n_decoded token_stream_event{kind=token}
  events on its own fresh channel;
- close exactly once with kind=closed, reason=completed;
- satisfy streamed_hash == rr.hash, where streamed_hash is
  the FNV-1a fold of the int32_t token ids the admitted
  request received over its own channel;
- satisfy rr.status == completed,
  rr.admission_source == completion_freed, and
  rr.arrival_source == external;
- carry rr.previous_request_id == the completed sibling's
  request_id and rr.reused_seq_id == the freed seq_id;
- leave residual KV empty across all n_seq_max slots at
  engine end.
```

The submitter-held `hpx::future<request_result>` carries
the result snapshot; the stream receiver is pushed into
`admitted_stream_handoffs_` keyed by `request_id` and is
drained by main via the same Slice 3 / Slice 4 path. Main
does **not** push to `admitted_futures_` for external
arrivals (the submitter already holds the future); only
the stream-handoff push happens under the existing
`admitted_futures_mtx_`.

## 4. Scope decision: completion_freed × external only

Slice 5 deliberately narrows the rebind predicate in the
external branch to `src == admission_source::completion_freed`
**only**. The cancel-freed × external path is intentionally
out of scope and remains deferred. The narrowing has two
motivations:

1. **Minimal-surface advance.** The smallest engine-side
   change that proves external-arrival streaming end-to-end
   is one rebind block in the external branch on one
   admission source. Cancel-freed × external requires the
   same rebind shape but is orthogonal to the smoke; a
   later slice will broaden the predicate.
2. **Tight coverage gate.** A coverage gate scoped to
   `--reuse-completed && cancel_plan empty && n_external_arrivals > 0`
   fail-closes on the exact path Slice 5 gates. A broader
   gate would weaken the Slice 5 evidence and conflate it
   with cancel-freed × external coverage.

When Slice 6 broadens the rebind predicate in the external
branch to `src == cancel_freed`, the coverage gate will
gain a parallel cancel-freed × external arm. Both
together close the external-arrival admission surface.

## 5. Smoke shape

```text
--stream-all
--n-seqs 3
--n-active 2
--n-waiting 0
--n-external-arrivals 1
--decode-budget-mix 8,256
--external-arrival-budget 16
--external-release-iter 3
--reuse-completed
--cancel-plan none
--repeat 2
```

Notes on the shape:

- `--n-active 2` keeps the decode loop alive across the
  admission boundary (seq 1, budget 256, is the
  survivor).
- `--decode-budget-mix 8,256` makes seq 0 complete at
  `done_iter=7`, freeing slot 0 for admission at iter 8
  (analogue of the Slice 3 smoke timing).
- `--external-release-iter 3` puts the external arrival
  in `waiting_queue_consumable_` at top of iter 4 — well
  before seq 0 completes at iter 7. The engine sets the
  release promise at end of iter 3, the scripted HPX
  submitter pushes via `engine::submit()`, the engine
  drains the inbox at top of iter 4.
- `--external-arrival-budget 16` keeps the admitted
  external stream a different length from the prior
  occupant's 8-token stream — preserves the existing
  Slice-3-style length-based independence-gate inequality
  (8 ≠ 16).
- `--reuse-completed` enables the completion-freed
  admission queue.
- `--cancel-plan none` keeps the cancel arm out of the
  smoke.
- `--repeat 2` re-asserts per-result determinism and
  per-seq streamed token vector determinism across two
  consecutive engine runs in the same process.

The `--stream-all` OFF regression continues to use the
canonical Slice 7 admission shape; all stream counters
remain exactly zero.

## 6. Expected per-request behavior

```text
request 0 (original active, seq_id=0):
  admission_source          = none
  arrival_source            = preloaded
  budget                    = 8
  status                    = completed
  done_iter                 = 7
  close_reason              = completed
  streamed count            = 8
  hash                      = 0x0619d4d1900c2365   (canonical anchor)

request 1 (original active, seq_id=1):
  admission_source          = none
  arrival_source            = preloaded
  budget                    = 256
  status                    = completed
  close_reason              = completed
  streamed count            = 256
  hash                      = 0x8790fbe5a60c9ae6
                              (within-shape uniqueness +
                               --repeat 2 determinism)

request 2 (external arrival, admitted into seq_id=0):
  reused_seq_id             = 0
  previous_request_id       = 0
  admitted_at_iter          = 8                    (== done_iter + 1 of seq 0)
  admission_source          = completion_freed
  arrival_source            = external
  budget                    = 16
  status                    = completed
  close_reason              = completed
  streamed count            = 16
  streamed_hash             = 0x833045f1e2ebf49f
                              (equals Slice 3 / Slice 4
                               preloaded admitted budget-16
                               hashes for the same prompt
                               and budget under greedy
                               decoding; NOT a new
                               cross-shape canonical
                               anchor)
  done_iter                 = 23
  pos_max_at_clear          = 20
```

`first_external_drain_iter` is expected to equal 4
(`= external_release_iter + 1`). The submitter ack set
is `{3}` and the release-fire iter set is `{3}`.

`--repeat 2` is deterministic on the per-seq streamed
token vectors, the streamed hashes, the close reasons,
and the per-result tuple.

## 7. Required gates

Per streamed request (original active and admitted alike):

```text
streamed_close[req]              == expected_close_from(rr.status)

streamed_tokens[req].size()      == rr.n_decoded

streamed_hash[req]               == rr.hash    (FNV-1a)
```

Slice 5-specific gates on the external-arrival admitted
request:

```text
rr.arrival_source                == external
rr.admission_source              == completion_freed
rr.reused_seq_id                 == prior occupant's seq_id
rr.previous_request_id           == prior occupant's request_id
rr.admitted_at_iter              == prior occupant's done_iter + 1
                                    (Slice 3/5 admit-result gates
                                     enforce this)

admitted_streamed_close[rr.request_id]
                                 == completed

streamed_tokens[admitted]        is the admitted request's
                                 own token vector (the
                                 rebound receiver's reads),
                                 populated through
                                 admitted_streamed_tokens
                                 keyed by request_id, not
                                 the seq-id-indexed
                                 streamed_tokens vector;

stream_tokens_emitted[slot] at admitted close == admitted
   request's own decoded token count (the per-slot
   cumulative counter was reset on rebind).
```

Engine-side stream-counter gates (derived from the per-
`request_result` status of the union of original-active
and admitted-streamed requests; the Slice 4 form, with
admitted-streamed defined as
`admission_src == completion_freed || cancel_freed`):

```text
streams_closed_error     == 0

streams_closed_completed == count(rr.status == completed
                                  over streamed requests)

streams_closed_cancelled == count(rr.status == cancelled
                                  over streamed requests)

streams_opened           == streams_closed_completed
                          + streams_closed_cancelled
                          + streams_closed_error

stream_tokens_emitted_total
                         == sum(rr.n_decoded over streamed)
```

Slice 5 coverage gate (new):

```text
streamed_admitted_external_arrival_count >= 1   when
   --stream-all
   && args.n_external_arrivals > 0
   && args.reuse_completed
   && args.cancel_plan.empty()
```

The Slice 3 completion-freed coverage gate
(`streamed_admitted_completion_freed_count >= 1` when
`--stream-all && --reuse-completed && --n-waiting > 0`)
and the Slice 4 cancel-freed coverage gate
(`streamed_admitted_cancel_freed_count >= 1` when
`--stream-all && !cancel_plan.empty() && --n-waiting > 0`)
are **unchanged**.

Carry-over invariants (unchanged from Slices 1–4):

- residual KV empty across all `n_seq_max` slots at engine
  end;
- with `LLAMA_HPX_CB_TRACE` unset, zero `[hpx-cb-gate]
  event=` lines on stderr (trace-off is one atomic load +
  early return per call site);
- with `--stream-all` OFF, every stream counter is exactly
  zero, the receiver vector is empty, and the admitted-
  stream handoff vector is empty;
- `--repeat 2` deterministic on the per-result tuple, the
  per-seq streamed token vectors, the streamed hashes,
  and the close reasons;
- single engine task per repeat, single `llama_context`
  ownership, no `llama_decode` call interrupted, every
  `llama_decode` returns 0;
- engine asserts `inbox_.empty()` and
  `external_promises_.empty()` before the residual-KV
  sweep.

## 8. Stream counter expectations

Per repeat (the smoke runs `--repeat 2`, so trace totals
are double):

```text
streams_opened              = 3        (2 ctor + 1 external admit)
streams_closed_completed    = 3        (orig 0 + orig 1 + admitted 2)
streams_closed_cancelled    = 0
streams_closed_error        = 0
stream_tokens_emitted_total = 280      (= 8 + 256 + 16)
```

External-arrival metrics (per repeat):

```text
admitted_count              = 1
external_admitted_count     = 1
arrival_drained_count       = 1
first_external_drain_iter   = 4
iter_release_fired_set      = {3}
submitter_ack_set           = {3}
```

Across `--repeat 2`, engine-side counters double exactly.

## 9. Trace evidence

Slice 5 uses the same trace event set as Slices 1–4; no
new event name is introduced. Under
`LLAMA_HPX_CB_TRACE=1` the smoke (`--repeat 2`) emits:

```text
token_stream_opened           == 6    (3 per repeat × 2 repeats)
token_stream_token            == 560  (280 per repeat × 2)
token_stream_closed           == 6
  reason=completed            == 6
  reason=cancelled            == 0
arrival_drained               == 2
request_queued                == 2    (with arrival_source=external)
iter_release_fired            == 2
submitter_ack_observed        == 2
request_admitted_live         == 2    (with
                                        admission_source=completion_freed,
                                        arrival_source=external)
seq_reused                    == 2
admitted_prefilled            == 2
```

Note: the source emits the release-fire trace event as
`iter_release_fired`, **not** `release_fired`. The count of
2 across `--repeat 2` matches the design's expected
release-fire total of 2.

Trace event ordering on the external admission boundary,
as captured in
`local/slice12_stream_step5_smoke_traceon.stderr` (one
repeat; gated only as "the trace shows", not as a
strict-ordering correctness invariant):

```text
event=iter_release_fired iter=3
event=submitter_ack_observed iter=3
event=arrival_drained request=2 iter=4 budget=16
... (seq 0 decodes 8 tokens, completes at iter 7) ...
event=seq_reused seq_id=0 previous_owner=0 new_owner=2 iter=8 admission_source=completion_freed arrival_source=external
event=request_admitted_live request=2 reused_seq_id=0 iter=8 admission_source=completion_freed arrival_source=external
event=token_stream_opened request=2 seq_id=0
event=admitted_prefilled request=2 seq_id=0 first_token=2259 admission_source=completion_freed
... 16 × event=token_stream_token request=2 seq_id=0 ...
event=token_stream_closed request=2 seq_id=0 n_tokens=16 reason=completed
```

`token_stream_opened request=2` follows the rebind, and
`token_stream_closed request=2` carries the admitted
request's own token count (`n_tokens=16`), not the prior
occupant + admitted cumulative.

## 10. Results

Streaming Slice 5 lands `HPX_CB_STREAM_STEP5: PASS` on all
four runs in the close-out matrix (Metal+HPX build,
TinyLlama Q4_K_M, greedy, `"Hello, my name is"`):

```text
stream-off regression on Slice 7 shape : HPX_CB_STREAM_STEP5: PASS
Slice 5 stream smoke (trace off)       : HPX_CB_STREAM_STEP5: PASS
Slice 5 trace-on stream smoke          : HPX_CB_STREAM_STEP5: PASS
Slice 5 cross-invocation repeat        : HPX_CB_STREAM_STEP5: PASS
```

Closeout captures:

```text
local/slice12_stream_step5_regression_streamoff.{stdout,stderr}
local/slice12_stream_step5_smoke.{stdout,stderr}
local/slice12_stream_step5_smoke_traceon.{stdout,stderr}
local/slice12_stream_step5_smoke_repeat2.{stdout,stderr}
```

Detailed evidence is in
`tools/hpx-continuous-batch-gate/results.md` under the
*Streaming Slice 5 results — external-arrival streaming
over completion-freed slot* section.

Carry-over gates passed on every Slice 5 run:

- residual KV empty across all `n_seq_max` slots at engine
  end (`residual_kv_empty = true`);
- trace-off quietness: zero `[hpx-cb-gate] event=` lines
  on stderr with `LLAMA_HPX_CB_TRACE` unset;
- `--repeat 2` deterministic on per-result tuple and per-
  seq streamed token vectors;
- admitted stream did not inherit the prior occupant's
  close reason or stream state.

## 11. HPX-native design note

Streaming Slice 5 stays HPX-native at the orchestration /
streaming-gate boundary:

- The streaming substrate remains
  `hpx::lcos::local::channel<token_stream_event>`.
- The engine HPX task is the sole producer of stream
  events for every slot, including the external-admitted
  request after rebind.
- `main()` (the gate consumer task) is the sole consumer
  via the matching `receive_channel`.
- Stream events carry only an `int32_t` token id and a
  close reason. No `llama_context`, KV state, logits, or
  any other llama.cpp object ever crosses the channel.
- Only the engine HPX task touches llama.cpp execution
  state (`llama_context`, `llama_batch`, `llama_decode`,
  `llama_memory_seq_*`, `llama_get_logits_ith`). The
  rebind happens inside the engine task on the
  external-admission boundary.
- The scripted HPX submitter task remains a pure
  message-pusher: it calls `engine::submit()` to push
  `arrival_msg` values into `inbox_` under
  `hpx::spinlock`, sets the ack promise, and returns. It
  calls **no** `llama_*` API (grep-gate enforced at
  review).
- Release+ack barrier remains HPX-promise-driven.
  `hpx::promise<void>` / `hpx::future<void>` carry the
  release+ack signals. No `std::thread`, no
  `std::condition_variable`, no
  `std::this_thread::sleep_for`, no wall-clock sleep.

Synchronization invariants under Slice 5:

- No new `std::mutex` is introduced.
- No new HPX primitive (channel, promise, future,
  spinlock) is introduced.
- The existing `admitted_futures_mtx_` critical section
  is reused only for the stream-handoff push in the
  external branch — same lock as Slices 3 and 4. This
  lock guards orchestration metadata, not llama.cpp
  execution state.
- The submitter-held
  `hpx::future<request_result>` is the result route for
  external arrivals; main does **not** push to
  `admitted_futures_` for those.
- The single-shot `hpx::promise<request_result>` is still
  fulfilled only after the channel closes, after per-seq
  KV clear, and after the cross-talk check passes.

Failure mode hardness:

- HPX runtime startup/shutdown remains process-level; the
  engine object does not start or stop HPX.
- Slice 5 fails closed on HPX/runtime/model/decode errors;
  no silent fallback to a non-HPX path is introduced.

## 12. Risks and caveats

1. **Slice 5 covers only the completion-freed × external
   path.** The Slice 5 rebind predicate in the external
   branch of `admit_one` is
   `src == admission_source::completion_freed` only.
   External arrivals admitted via the cancel-freed slot
   still produce admitted requests whose slots inherit
   `stream_closed = true` from the prior occupant and
   silently skip streaming. Streaming that path is
   deferred to a later slice (see §13).

2. **Pre-existing Live Admission Slice 6 gate adjusted.**
   The Slice 6 results-validation block originally
   required every external arrival to come via
   `admission_src=cancel_freed` (its comment admitted "no
   completion-freed external path exercised yet"). Slice
   5 relaxed it to accept either `cancel_freed` (when
   `cancel_plan` non-empty) or `completion_freed` (when
   `--reuse-completed` is on), and skipped the
   `cancel_after+1` `admitted_at_iter` recheck for the
   completion-freed case (the per-result Slice 3/5
   admit-result gates already check the completion-freed
   `admitted_at_iter` invariant). The Slice 7
   mixed-source `slice7_strict` block was **not**
   touched: its preconditions
   (`reuse_completed && !cancel_plan.empty() && n_waiting > 0
    && n_external_arrivals > 0`) exclude the Slice 5
   smoke shape, so the Slice 7 phase1/phase2 invariants
   are unmoved.

3. **Trace event name is `iter_release_fired`.** Some
   external-facing summaries (including the original
   Slice 5 source-task description) used the shorthand
   `release_fired`. The source emits the event under the
   name `iter_release_fired`. The Slice 5 trace-count
   total of 2 across `--repeat 2` matches the design's
   expected release-fire total.

4. **Independence evidence is length-based.** The
   pre-existing Slice 3 admitted-only independence gate
   is a full token-vector compare. Greedy decoding on
   the same prompt produces the same first-N tokens
   regardless of admission path or arrival source, so a
   16-vs-16 shape (prior occupant size 16 vs admitted
   budget 16) would make the vectors byte-identical and
   trip the gate as a false positive. Sizing prior
   occupant to 8 and admitted budget to 16 makes the
   vectors differ in length, so equality is structurally
   impossible. The Slice 5 invariant is unaffected, but
   readers should not interpret the gate's inequality as
   evidence of semantic token divergence — it is
   length-based.

5. **External admitted hash is shape-scoped.** The
   observed budget-16 admitted hash
   `0x833045f1e2ebf49f` equals the Slice 3
   completion-freed and Slice 4 cancel-freed preloaded
   admitted budget-16 hashes for the same prompt /
   policy / budget. This is expected under greedy
   decoding — the first 16 generated tokens are
   deterministic regardless of which admission path or
   arrival source bound the slot. It is **not** a new
   cross-shape canonical anchor; it is the budget-16
   fingerprint for this specific prompt / policy /
   batch-shape combination and is gated only as
   `streamed_hash == rr.hash` within the Slice 5 smoke.

Hard invariant: only the engine HPX task touches
`llama_context`, `llama_batch`, `llama_decode`,
`llama_memory_seq_*`, or `llama_get_logits_ith`. Stream
events carry only an `int32_t` token id and a close
reason; no llama.cpp state ever crosses the channel.

## 13. Deferred work

Out of scope for Streaming Slice 5:

- **External × cancel-freed admitted streaming.** A
  later slice will broaden the rebind predicate in the
  external branch of `admit_one` to also fire on
  `src == admission_source::cancel_freed`.
- **HTTP / gRPC / Unix-socket streaming.** The streaming
  surface remains an in-process HPX local channel
  between the engine task and `main()`. No network
  framing, no chat-template assembly, no `tools/server`
  integration. See
  `docs/hpx/continuous_batching_gate_vs_serving_layer.md`
  for the gate-vs-serving-layer boundary.
- **Backpressure / bounded-channel policy.** `channel<T>`
  is still unbounded so the engine never suspends on the
  consumer. A bounded-channel slice is deferred.
- **Routing the receiver to the submitter task.** In a
  serving layer the receiver should travel to the
  submitter alongside the future. The gate-only scope of
  Slice 5 keeps main as the consumer (via the existing
  `admitted_stream_handoffs_` path); a serving-layer
  slice will move the receiver to the submitter.
- **Multi-cycle slot reuse with streaming.** The Slice 5
  smoke admits one external arrival into one freed slot
  per run.
- **Per-request prompts and per-request sampling.** The
  smoke still uses TinyLlama, `"Hello, my name is"`, and
  greedy decoding.
- **Engine-failure stream semantics.** The defensive
  `error` close path exists in source; no smoke
  exercises it.
- **Performance comparisons.** Streaming Slice 5 is
  correctness-only.

## 14. Safe claim language

Supported by Streaming Slice 5 evidence and safe to use
verbatim:

- "External-arrival admitted-request streaming is gated
  at the HPX continuous-batching gate boundary on the
  completion-freed admission path: an externally
  arriving request submitted through the scripted HPX
  submitter task via `engine::submit()` and admitted
  into a slot freed by a sibling's natural completion
  opens a fresh HPX local-channel stream, emits exactly
  `rr.n_decoded` token events whose FNV-1a fold equals
  `rr.hash`, and closes with `reason=completed`."
- "The Slice 5 `admit_one` rebind predicate in the
  external branch covers `admission_source::completion_freed`
  only. External-arrival admitted streaming over
  cancel-freed remains deferred."
- "Streaming Slice 5 introduces no new HPX primitive,
  no new `std::mutex`, no new CLI flag, no new trace
  event name, no new channel type, no CMake change, no
  HTTP / server streaming, no backpressure, and no
  performance claim. The engine-to-main handoff of the
  external admitted request's stream reuses the
  existing `admitted_stream_handoffs_` vector and the
  existing `admitted_futures_mtx_` critical section
  introduced in Slice 3."
- "The submitter-held `hpx::future<request_result>`
  continues to carry the result snapshot for external
  arrivals; main does not push to `admitted_futures_`
  for external arrivals. The stream receiver is pushed
  onto the existing `admitted_stream_handoffs_` vector
  and drained by main via the same Slice 3 / Slice 4
  path."
- "Engine-side stream counters
  (`streams_closed_completed`, `streams_closed_cancelled`,
  `streams_closed_error`, `streams_opened`,
  `stream_tokens_emitted_total`) match the corresponding
  counts derived from the per-`request_result` status of
  the union of original-active and admitted-streamed
  requests on every gated run."
- "All Slice 1–7 admission/cancellation/lifecycle
  invariants and all Streaming Slice 1 / Slice 2 /
  Slice 3 / Slice 4 streaming invariants remain strict
  carry-overs under `HPX_CB_STREAM_STEP5`."

Not supported by Slice 5 and to be avoided:

- "Streaming Slice 5 covers external-arrival admission
  streaming on the cancel-freed slot." (The `admit_one`
  external-branch rebind predicate is scoped to
  `completion_freed` only.)
- "Streaming Slice 5 improves performance." (No
  comparative benchmark is run; this is a correctness /
  lifecycle gate.)
- "Streaming Slice 5 integrates with `llama-server`."
  (No network adapter; the stream is in-process only.)
- "The Slice 5 independence gate proves semantic token
  divergence between the prior occupant's stream and
  the admitted external arrival's stream." (The gate
  is a full token-vector compare; the Slice 5 smoke
  makes that comparison inequality-by-length. See §12.)
- "The trace event for the release barrier is
  `release_fired`." (The source emits it as
  `iter_release_fired`.)
- "The admitted budget-16 hash `0x833045f1e2ebf49f` is
  a new canonical cross-shape anchor." (It equals the
  Slice 3 / Slice 4 preloaded admitted budget-16
  hashes for the same prompt/policy/budget under
  greedy decoding and is not a cross-shape correctness
  invariant under CLAUDE.md.)
