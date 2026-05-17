# Continuous batching — Streaming Slice 6 design

## 1. Purpose

Streaming Slice 6 closes the external-arrival admission arm
of the HPX-native per-request token stream on the
**cancel-freed** slot. Slices 1 and 2 gated streaming on
the completion and cancellation paths for original-active
seqs. Slice 3 extended streaming to **preloaded** waiters
admitted via the completion-freed slot. Slice 4 extended
streaming to **preloaded** waiters admitted via the
cancel-freed slot. Slice 5 extended streaming to
**external arrivals** admitted via the completion-freed
slot. Slice 6 extends streaming to **external arrivals**
admitted via the **cancel-freed** slot. With Slice 6
closed, every admission × arrival-source combination —
preloaded × {completion_freed, cancel_freed} and external ×
{completion_freed, cancel_freed} — is gated end-to-end as a
streaming surface inside the HPX continuous-batching gate.

Slice 6 is a gate-only / orchestration-only advance. The
engine-side delta is a single-disjunct broadening of the
external-branch rebind predicate in `admit_one`:

```text
old (Slice 5):
  stream_all_ && src == admission_source::completion_freed

new (Slice 6):
  stream_all_ && (src == admission_source::completion_freed
              || src == admission_source::cancel_freed)
```

This is structurally symmetric to Slice 4 on the preloaded
branch (which broadened the preloaded-branch predicate from
`completion_freed` to `completion_freed || cancel_freed`).
No new HPX primitive, no new `std::mutex`, no new CLI flag,
no new trace event name, no new channel type, no CMake
change. The streaming substrate, ownership rules, payload
(`int32_t` token id + close reason), and trace event set
are unchanged from Slices 1–5.

This is a correctness/lifecycle gate. No performance claim
is made.

## 2. Relation to Streaming Slices 1–5

Slices 1–7 (the admission gate sequence,
`HPX_CB_ADMIT_STEP1` … `HPX_CB_ADMIT_STEP7`) remain a
carry-over invariant: every admission, cancellation, and
lifecycle invariant gated there still holds under
`HPX_CB_STREAM_STEP6`. The `--stream-all` OFF regression
gate on the canonical Slice 7 admission shape continues to
pass with all engine-side stream counters at exactly zero.

Streaming Slice 1 (`HPX_CB_STREAM_STEP1: PASS`) proved the
completion arm for an original active seq. Streaming Slice
2 (`HPX_CB_STREAM_STEP2: PASS`) proved the cancellation
arm for an original active seq. Streaming Slice 3
(`HPX_CB_STREAM_STEP3: PASS`) proved preloaded-waiter
streaming over a completion-freed slot. Streaming Slice 4
(`HPX_CB_STREAM_STEP4: PASS`) proved preloaded-waiter
streaming over a cancel-freed slot. Streaming Slice 5
(`HPX_CB_STREAM_STEP5: PASS`) proved external-arrival
streaming over a completion-freed slot.

Streaming Slice 6 (`HPX_CB_STREAM_STEP6: PASS`) is the
external-arrival analogue of Slice 4: same admission source
(`cancel_freed`), different arrival source (`external`
instead of `preloaded`). The Slice 1 / Slice 2 streaming
invariants (single producer, single consumer, terminal
`kind=closed` before `channel.close()`, no llama.cpp state
crossing the channel, unbounded `channel<T>`, trace-off
quietness) and all Slice 3 / Slice 4 / Slice 5 admitted-
streaming invariants are strict carry-overs. The Slice 3 /
Slice 4 / Slice 5 smokes and gates are unchanged.

## 3. Target invariant

An externally-arriving request admitted into a cancel-freed
slot under `--stream-all` must:

```text
- open a fresh hpx::lcos::local::channel<token_stream_event>
  on the recycled seq_id (the prior cancelled occupant's
  closed channel must not be inherited);
- have stream_closed reset to false at admission;
- have the per-slot cumulative stream_tokens_emitted reset
  to zero at admission (so the admitted close trace
  reports the admitted-request count, not prev + admitted
  cumulative);
- emit exactly rr.n_decoded token_stream_event{kind=token}
  events on its own fresh channel;
- close exactly once with kind=closed, reason=completed —
  not inheriting the cancelled previous occupant's
  reason=cancelled close;
- satisfy streamed_hash == rr.hash, where streamed_hash is
  the FNV-1a fold of the int32_t token ids the admitted
  request received over its own channel;
- satisfy rr.status == completed,
  rr.admission_source == cancel_freed, and
  rr.arrival_source == external;
- carry rr.previous_request_id == the cancelled sibling's
  request_id and rr.reused_seq_id == the freed seq_id;
- satisfy rr.admitted_at_iter == cancel_after + 1 (the
  one-iter cancel→admit delay invariant from the
  Live Admission Slice 6 / Slice 3 admit-result gates,
  carried over);
- leave residual KV empty across all n_seq_max slots at
  engine end.
```

The submitter-held `hpx::future<request_result>` carries
the result snapshot; the stream receiver is pushed into
`admitted_stream_handoffs_` keyed by `request_id` and is
drained by main via the same Slice 3 / Slice 4 / Slice 5
path. Main does **not** push to `admitted_futures_` for
external arrivals (the submitter already holds the future);
only the stream-handoff push happens under the existing
`admitted_futures_mtx_`.

## 4. Scope

Slice 6 covers exactly one new admission × arrival-source
combination on the streaming surface:

```text
external × cancel_freed   (new in Slice 6)
```

Carried over from earlier slices (still gated and tested
on their respective smokes):

```text
original active completion              (Slice 1)
original active cancellation            (Slice 2)
preloaded × completion_freed admitted   (Slice 3)
preloaded × cancel_freed     admitted   (Slice 4)
external × completion_freed  admitted   (Slice 5)
```

With Slice 6 closed, the streaming gate covers every
admission × arrival-source combination the gate exercises.
No further admission paths are introduced.

The Slice 5 narrow comment ("cancel-freed × external
admitted streaming is deferred to a later slice") is
removed by Slice 6. The Slice 6 source change is one
single-disjunct extension of the Slice 5 predicate; the
push site, lock scope, handoff vector, drain path, and
trace event names are all carry-overs.

## 5. Smoke shape

```text
--stream-all
--n-seqs 3
--n-active 2
--n-waiting 0
--n-external-arrivals 1
--decode-budget-mix 64,256
--external-arrival-budget 16
--external-release-iter 3
--cancel-plan 0
--cancel-after 8
--repeat 2
```

Notes on the shape:

- `--n-active 2` keeps the decode loop alive across the
  cancel→admit boundary (seq 1, budget 256, is the
  surviving original).
- `--decode-budget-mix 64,256` round-robins seq 0 to
  budget 64 and seq 1 to budget 256. With
  `--cancel-after 8`, seq 0 is cancelled at iter 8 and
  frees slot 0 — the cancel-freed slot the external
  arrival lands on.
- `--cancel-plan 0` plans cancellation on seq 0 only.
  The cancel-freed pool ends up with exactly one entry
  for one repeat.
- `--cancel-after 8` is small enough that the cancellation
  fires before seq 0 reaches its budget of 64; the
  cancellation happens on the decode-iter boundary.
- `--external-release-iter 3` puts the external arrival
  in `waiting_queue_consumable_` at top of iter 4 — well
  before the cancellation observation at iter 8. The
  engine sets the release promise at end of iter 3, the
  scripted HPX submitter pushes via `engine::submit()`,
  the engine drains the inbox at top of iter 4. The
  external arrival waits in the consumable queue until
  the cancel-freed slot becomes available at iter 9
  (`cancel_after + 1`).
- `--external-arrival-budget 16` keeps the admitted
  external stream a different length from the cancelled
  prior occupant's 8-token stream — preserves the
  Slice-3-style length-based independence-gate
  inequality (8 ≠ 16).
- `--n-waiting 0` keeps preloaded waiters out of the
  smoke; the only admission that fires is external ×
  cancel_freed.
- `--repeat 2` re-asserts per-result determinism and
  per-seq streamed token vector determinism across two
  consecutive engine runs in the same process.

The `--stream-all` OFF regression continues to use the
canonical Slice 7 admission shape; all stream counters
remain exactly zero.

## 6. Expected per-request behavior

```text
request 0 (original active, seq_id=0, cancelled):
  admission_source          = none
  arrival_source            = preloaded
  budget                    = 64
  status                    = cancelled
  cancel_observed_iter      = 8
  n_decoded_at_cancel       = 8
  close_reason              = cancelled
  streamed count            = 8
  hash                      = (no completion hash; cancelled)

request 1 (original active, seq_id=1, surviving):
  admission_source          = none
  arrival_source            = preloaded
  budget                    = 256
  status                    = completed
  close_reason              = completed
  streamed count            = 256
  hash                      = 0x8790fbe5a60c9ae6
                              (within-shape uniqueness +
                               --repeat 2 determinism)

request 2 (external arrival, admitted into seq_id=0 from
            cancel-freed pool):
  reused_seq_id             = 0
  previous_request_id       = 0
  admitted_at_iter          = 9                    (== cancel_after + 1)
  admission_source          = cancel_freed
  arrival_source            = external
  budget                    = 16
  status                    = completed
  close_reason              = completed
  streamed count            = 16
  streamed_hash             = 0x833045f1e2ebf49f
                              (equals Slice 3 / Slice 4 /
                               Slice 5 admitted budget-16
                               hashes for the same prompt
                               and budget under greedy
                               decoding; NOT a new
                               cross-shape canonical
                               anchor)
  done_iter                 = 24
  pos_max_at_clear          = 20
```

`first_external_drain_iter` is expected to equal 4
(`= external_release_iter + 1`). The submitter ack set
is `{3}` and the release-fire iter set is `{3}`. The
release-fire trace event name is **`iter_release_fired`**
(not `release_fired`) — carry-over from Slice 5.

`--repeat 2` is deterministic on the per-seq streamed
token vectors, the streamed hashes, the close reasons,
and the per-result tuple.

## 7. Required gates

Per streamed request (original active and admitted alike):

```text
streamed_close[req]              == expected_close_from(rr.status)

streamed_tokens[req].size()      == rr.n_decoded
                                    (== n_decoded_at_cancel
                                     when rr.status == cancelled)

streamed_hash[req]               == rr.hash    (FNV-1a)
```

Slice 6-specific gates on the external-arrival admitted
request:

```text
rr.arrival_source                == external
rr.admission_source              == cancel_freed
rr.reused_seq_id                 == prior cancelled occupant's seq_id
rr.previous_request_id           == prior cancelled occupant's request_id
rr.admitted_at_iter              == cancel_after + 1
                                    (Live Admission Slice 6
                                     admit-result gate carry-over)

admitted_streamed_close[rr.request_id]
                                 == completed
                                    (NOT cancelled — admitted
                                     close is emitted on the
                                     rebound channel, not
                                     inherited from the
                                     cancelled occupant's close)

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

Per cancelled original request (Slice 2 carry-over):

```text
streamed_close[req=0]            == cancelled
streamed_tokens[seq=0].size()    == n_decoded_at_cancel  (== 8)
streamed_hash[req=0]             == rr.hash
rr.status                        == cancelled
```

Engine-side stream-counter gates (derived from the per-
`request_result` status of the union of original-active
and admitted-streamed requests; same form as Slice 4 /
Slice 5, with admitted-streamed defined as
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

Slice 6 coverage gate (new):

```text
streamed_admitted_external_cancel_freed_count >= 1   when
   --stream-all
   && args.n_external_arrivals > 0
   && !args.cancel_plan.empty()
```

The gate lives inside the enclosing `if (args.stream_all)`
block so stream-off regression runs (`--stream-all` OFF)
bypass it entirely. The Slice 3 completion-freed coverage
gate, the Slice 4 cancel-freed (preloaded) coverage gate,
and the Slice 5 external × completion_freed coverage gate
are **unchanged**.

Carry-over invariants (unchanged from Slices 1–5):

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
streams_opened              = 3        (1 ctor cancelled
                                       + 1 ctor surviving
                                       + 1 external admit-rebind)
streams_closed_completed    = 2        (req 1 + req 2)
streams_closed_cancelled    = 1        (req 0)
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

Cancellation metrics (per repeat):

```text
cancel_observed             = 1
cancel_kv_cleared           = 1
cancel_future_fulfilled     = 1
seq_reused                  = 1   (cancel-freed seq 0 → ext req 2)
admitted_prefilled          = 1
request_admitted_live       = 1   (with admission_source=cancel_freed,
                                          arrival_source=external)
```

Across `--repeat 2`, engine-side counters double exactly.

## 9. Trace evidence

Slice 6 uses the same trace event set as Slices 1–5; no
new event name is introduced. Under
`LLAMA_HPX_CB_TRACE=1` the smoke (`--repeat 2`) emits:

```text
token_stream_opened           == 6    (3 per repeat × 2 repeats)
token_stream_token            == 560  (280 per repeat × 2)
token_stream_closed           == 6
  reason=completed            == 4    (req 1 + req 2 across 2 repeats)
  reason=cancelled            == 2    (req 0 across 2 repeats)
arrival_drained               == 2
request_queued                == 2    (with arrival_source=external)
iter_release_fired            == 2
submitter_ack_observed        == 2
cancel_observed               == 2
cancel_kv_cleared             == 2
cancel_future_fulfilled       == 2
request_admitted_live         == 2    (with
                                        admission_source=cancel_freed,
                                        arrival_source=external)
seq_reused                    == 2
admitted_prefilled            == 2
```

Note: the source emits the release-fire trace event as
`iter_release_fired`, **not** `release_fired`. The count of
2 across `--repeat 2` matches the design's expected
release-fire total of 2.

A descriptive ordering hint for the external admission
boundary (read off the trace-on capture; gated only as
"the trace shows", **not** as a strict-ordering correctness
invariant):

```text
event=iter_release_fired iter=3
event=submitter_ack_observed iter=3
event=arrival_drained request=2 iter=4 budget=16
... (seq 0 decodes 8 tokens, then cancel observed at iter 8) ...
event=cancel_observed seq=0 iter=8 n_decoded=8
event=cancel_kv_cleared seq=0 pos_max_at_clear=12 cross_talk_ok=1
event=cancel_future_fulfilled seq=0 status=cancelled ttc_us=...
event=token_stream_closed request=0 seq_id=0 n_tokens=8 reason=cancelled
... (admission iter 9: cancel-freed pool drains, external req 2 binds) ...
event=seq_reused seq_id=0 previous_owner=0 new_owner=2 iter=9 admission_source=cancel_freed arrival_source=external
event=request_admitted_live request=2 reused_seq_id=0 iter=9 admission_source=cancel_freed arrival_source=external
event=token_stream_opened request=2 seq_id=0
event=admitted_prefilled request=2 seq_id=0 first_token=... admission_source=cancel_freed
... 16 × event=token_stream_token request=2 seq_id=0 ...
event=token_stream_closed request=2 seq_id=0 n_tokens=16 reason=completed
```

`token_stream_opened request=2` follows the rebind, and
`token_stream_closed request=2` carries the admitted
request's own token count (`n_tokens=16`), not the
cancelled occupant + admitted cumulative. The cancelled
occupant's `token_stream_closed request=0 ... reason=cancelled`
fires before the admitted request's `token_stream_opened`
on the same `seq_id`, so the sequence of close-then-open
on the recycled slot is structurally well-defined.

## 10. Results

Streaming Slice 6 lands `HPX_CB_STREAM_STEP6: PASS` on all
five runs in the close-out matrix (Metal+HPX build,
TinyLlama Q4_K_M, greedy, `"Hello, my name is"`):

```text
stream-off regression on Slice 7 shape : HPX_CB_STREAM_STEP6: PASS
Slice 6 stream smoke (trace off)       : HPX_CB_STREAM_STEP6: PASS
Slice 6 trace-on stream smoke          : HPX_CB_STREAM_STEP6: PASS
Slice 6 cross-invocation repeat        : HPX_CB_STREAM_STEP6: PASS
Slice 5 regression (optional)          : HPX_CB_STREAM_STEP6: PASS
```

The fifth run is the **optional Slice 5 regression**: the
Slice 5 smoke shape (`--decode-budget-mix 8,256
--reuse-completed --cancel-plan none --external-arrival-budget 16`)
re-run under the Slice 6 binary to confirm the broadened
external-branch rebind predicate did not regress the Slice
5 path. The streamed admitted request lands with
`admission_src=completion_freed` exactly as in Slice 5.

Closeout captures:

```text
local/slice13_stream_step6_regression_streamoff.{stdout,stderr}
local/slice13_stream_step6_smoke.{stdout,stderr}
local/slice13_stream_step6_smoke_traceon.{stdout,stderr}
local/slice13_stream_step6_smoke_repeat2.{stdout,stderr}
local/slice13_stream_step6_slice5_regression.{stdout,stderr}
```

Detailed evidence is in
`tools/hpx-continuous-batch-gate/results.md` under the
*Streaming Slice 6 results — external-arrival streaming
over cancel-freed slot* section.

Carry-over gates passed on every Slice 6 run:

- residual KV empty across all `n_seq_max` slots at engine
  end (`residual_kv_empty = true`);
- trace-off quietness: zero `[hpx-cb-gate] event=` lines
  on stderr with `LLAMA_HPX_CB_TRACE` unset;
- `--repeat 2` deterministic on per-result tuple and per-
  seq streamed token vectors;
- admitted stream did not inherit the cancelled prior
  occupant's close reason or stream state.

## 11. HPX-native design note

Streaming Slice 6 stays HPX-native at the orchestration /
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

Synchronization invariants under Slice 6:

- No new `std::mutex` is introduced.
- No new HPX primitive (channel, promise, future,
  spinlock) is introduced.
- The existing `admitted_futures_mtx_` critical section
  is reused only for the stream-handoff push in the
  external branch — same lock, same scope as Slices 3,
  4, and 5. This lock guards orchestration metadata, not
  llama.cpp execution state.
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
- Slice 6 fails closed on HPX/runtime/model/decode errors;
  no silent fallback to a non-HPX path is introduced.

## 12. Risks and caveats

1. **Slice 6 covers exactly the external × cancel_freed
   streaming surface.** All other admission × arrival-
   source combinations remain as gated by Slices 1–5.
   The Slice 5 external × completion_freed gate is
   carried over and the optional fifth run regression-
   tests it under the Slice 6 binary.

2. **Independence evidence is length-based.** The Slice 3
   admitted-only independence gate compares full token
   vectors. In the Slice 6 smoke the cancelled previous
   occupant's stream is 8 tokens vs the admitted
   external's 16, so equality is structurally
   impossible. The gate proves channel-rebind structural
   independence (a fresh
   `hpx::lcos::local::channel<token_stream_event>` was
   bound on admission, the per-slot
   `stream_tokens_emitted` counter was reset on rebind,
   and the admitted close was emitted on the rebound
   channel rather than inheriting the cancelled
   occupant's `reason=cancelled` close). It is **not**
   evidence of semantic token divergence — greedy
   decoding on the same prompt produces identical
   first-N tokens regardless of admission path or
   arrival source.

3. **External admitted hash is shape-scoped.** The
   observed budget-16 admitted hash
   `0x833045f1e2ebf49f` equals the Slice 3 / Slice 4 /
   Slice 5 admitted budget-16 hashes for the same prompt
   / policy / budget. This is expected under greedy
   decoding — the first 16 generated tokens are
   deterministic regardless of which admission path or
   arrival source bound the slot. It is **not** a new
   cross-shape canonical anchor; it is the budget-16
   fingerprint for this specific prompt / policy /
   batch-shape combination and is gated only as
   `streamed_hash == rr.hash` within the Slice 6 smoke.

4. **Trace event name is `iter_release_fired`.** Some
   external-facing summaries used the shorthand
   `release_fired`. The source emits the event under the
   name `iter_release_fired`. The Slice 6 trace-count
   total of 2 across `--repeat 2` matches the design's
   expected release-fire total.

5. **Trace-ordering hint is descriptive only.** Slice 6
   does **not** introduce a strict trace-order
   correctness invariant; the per-event count totals
   are the gated form. The ordering shown in §9 is
   descriptive evidence read off the trace-on capture.

6. **Pre-existing Live Admission Slice 6 results-
   validation block already accepts Slice 6's smoke
   shape.** The Slice 5 closeout already relaxed the
   per-result external-arrival validation to accept
   either `cancel_freed` (when `cancel_plan` non-empty)
   or `completion_freed` (when `--reuse-completed` is
   on). The Slice 6 smoke (`cancel_plan` non-empty,
   `--reuse-completed` off) takes the cancel-freed arm
   and the existing `cancel_after+1`
   `admitted_at_iter` recheck is exercised. The Slice 7
   mixed-source `slice7_strict` block is unaffected.

Hard invariant: only the engine HPX task touches
`llama_context`, `llama_batch`, `llama_decode`,
`llama_memory_seq_*`, or `llama_get_logits_ith`. Stream
events carry only an `int32_t` token id and a close
reason; no llama.cpp state ever crosses the channel.

## 13. Deferred work

Out of scope for Streaming Slice 6:

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
  Slice 6 keeps main as the consumer (via the existing
  `admitted_stream_handoffs_` path); a serving-layer
  slice will move the receiver to the submitter.
- **Multi-cycle slot reuse with streaming.** The Slice 6
  smoke admits one external arrival into one cancel-
  freed slot per run; chaining a second
  cancel-then-admit cycle on the same slot within one
  run is out of scope.
- **Per-request prompts and per-request sampling.** The
  smoke still uses TinyLlama, `"Hello, my name is"`, and
  greedy decoding.
- **Engine-failure stream semantics.** The defensive
  `error` close path exists in source; no smoke
  exercises it.
- **Performance comparisons.** Streaming Slice 6 is
  correctness-only.

## 14. Safe claim language

Supported by Streaming Slice 6 evidence and safe to use
verbatim:

- "External-arrival admitted streaming is gated on the
  cancel-freed path: after an original streamed request
  is cancelled and frees a slot, an externally arriving
  request submitted through the scripted HPX submitter
  task via `engine::submit()` and admitted into that
  slot opens a fresh HPX local-channel stream, emits
  exactly `rr.n_decoded` token events whose FNV-1a fold
  equals `rr.hash`, closes with `reason=completed`, and
  does not inherit the cancelled occupant's stream
  state or close reason."
- "The Slice 6 `admit_one` rebind predicate in the
  external branch covers
  `admission_source::completion_freed` ∪
  `admission_source::cancel_freed` — the Slice 5
  predicate broadened by one disjunct, structurally
  symmetric to the Slice 4 broadening on the preloaded
  branch."
- "Streaming Slice 6 introduces no new HPX primitive,
  no new `std::mutex`, no new CLI flag, no new trace
  event name, no new channel type, no CMake change, no
  HTTP / server streaming, no backpressure, no
  multi-cycle slot reuse, no engine-error stream
  semantics, and no performance claim. The
  engine-to-main handoff of the external admitted
  request's stream reuses the existing
  `admitted_stream_handoffs_` vector and the existing
  `admitted_futures_mtx_` critical section introduced
  in Slice 3."
- "The submitter-held `hpx::future<request_result>`
  continues to carry the result snapshot for external
  arrivals; main does not push to `admitted_futures_`
  for external arrivals. The stream receiver is pushed
  onto the existing `admitted_stream_handoffs_` vector
  and drained by main via the same Slice 3 / Slice 4 /
  Slice 5 path."
- "Engine-side stream counters
  (`streams_closed_completed`, `streams_closed_cancelled`,
  `streams_closed_error`, `streams_opened`,
  `stream_tokens_emitted_total`) match the corresponding
  counts derived from the per-`request_result` status of
  the union of original-active and admitted-streamed
  requests on every gated run."
- "All Slice 1–7 admission/cancellation/lifecycle
  invariants and all Streaming Slice 1 / Slice 2 /
  Slice 3 / Slice 4 / Slice 5 streaming invariants
  remain strict carry-overs under
  `HPX_CB_STREAM_STEP6`."
- "With Slice 6 closed, every admission × arrival-
  source combination — preloaded ×
  {completion_freed, cancel_freed} and external ×
  {completion_freed, cancel_freed} — is gated end-to-end
  as a streaming surface inside the HPX continuous-
  batching gate."

Not supported by Slice 6 and to be avoided:

- "Streaming Slice 6 covers HTTP, gRPC, Unix-socket, or
  any network streaming." (The stream is in-process
  only.)
- "Streaming Slice 6 introduces backpressure or a
  bounded `channel<T>`." (The channel remains
  unbounded.)
- "Streaming Slice 6 improves performance." (No
  comparative benchmark is run; this is a correctness /
  lifecycle gate.)
- "Streaming Slice 6 integrates with `llama-server`."
  (No network adapter; the stream is in-process only.)
- "Streaming Slice 6 covers multi-cycle slot reuse." (A
  `seq_id` is reused at most once per run.)
- "Streaming Slice 6 covers engine-failure
  `reason=error` stream semantics." (The defensive
  `error` close path exists in source; no smoke
  exercises it.)
- "The Slice 6 independence gate proves semantic token
  divergence between the cancelled occupant's stream
  and the admitted external arrival's stream." (The
  gate is a full token-vector compare; the Slice 6
  smoke makes that comparison inequality-by-length. See
  §12.)
- "The trace event for the release barrier is
  `release_fired`." (The source emits it as
  `iter_release_fired`.)
- "The admitted budget-16 hash `0x833045f1e2ebf49f` is
  a new canonical cross-shape anchor." (It equals the
  Slice 3 / Slice 4 / Slice 5 admitted budget-16
  hashes for the same prompt/policy/budget under
  greedy decoding and is not a cross-shape correctness
  invariant under CLAUDE.md.)
- "The Slice 6 trace ordering on the cancel→admit
  boundary is a strict correctness invariant." (It is
  descriptive evidence; the per-event count totals are
  the gated form.)
