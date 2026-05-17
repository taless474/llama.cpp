# Continuous batching — Streaming Slice 3 design

## 1. Purpose

Streaming Slice 3 closes the admitted-request arm of the
HPX-native per-request token stream introduced in Streaming
Slice 1 and extended on the cancellation path in Streaming
Slice 2. Slice 1 proved completion-path streaming for an
original active seq; Slice 2 proved cancellation-path
streaming for an original active seq. Slice 3 proves that a
**waiting request admitted into a slot freed by a sibling's
natural completion** is itself streamed end-to-end: it opens
a fresh HPX local-channel stream on the recycled slot,
emits exactly `rr.n_decoded` token events, satisfies
`streamed_hash == rr.hash`, and closes with
`reason=completed`. The admitted request's streamed token
vector must not inherit the previous occupant's token
vector or close reason.

Slice 3 is a gate-only / orchestration-only advance with
the following engine-side delta inside the existing
`admit_one` path:

- the slot's `token_stream_channel` is rebound to a fresh
  `hpx::lcos::local::channel<token_stream_event>`;
- `stream_closed` is reset to `false`;
- the per-slot cumulative `stream_tokens_emitted` counter
  is reset to `0` so `token_stream_closed` reports the
  admitted request's own token count, not previous+admitted
  cumulative;
- an explicit `{request_id, receiver}` bundle is pushed
  under the existing `admitted_futures_mtx_` critical
  section so the gate side can pick up the admitted
  request's stream by request_id.

No new HPX primitive, no new `std::mutex`, no new CLI
flag, no new CMake change, no new trace event name, no new
channel type. The streaming substrate, ownership model,
and event payload (`int32_t` token id + close reason) are
unchanged from Slice 1.

This is a correctness/lifecycle gate. No performance claim
is made.

## 2. Relation to Streaming Slices 1 and 2

Slices 1–7 (the admission gate sequence,
`HPX_CB_ADMIT_STEP1` … `HPX_CB_ADMIT_STEP7`) remain a
carry-over invariant: every admission, cancellation, and
lifecycle invariant gated there still holds under
`HPX_CB_STREAM_STEP3`. The `--stream-all` OFF regression
gate on the canonical Slice 7 admission shape continues to
pass with all engine-side stream counters at exactly zero.

Streaming Slice 1 (`HPX_CB_STREAM_STEP1: PASS`) proved the
completion arm for an original active seq. Streaming Slice
2 (`HPX_CB_STREAM_STEP2: PASS`) proved the cancellation
arm for an original active seq.

Streaming Slice 3 (`HPX_CB_STREAM_STEP3: PASS`) extends
the completion arm to cover an **admitted** request on the
**completion-freed admission path**: the prior occupant
completes naturally, the slot is recycled, the admitted
waiter is prefilled, decoded, and streamed on a fresh
channel bound to the same `seq_id`. The Slice 1 and Slice 2
streaming invariants (single producer, single consumer,
terminal `kind=closed` before `channel.close()`, no
llama.cpp state crossing the channel, unbounded
`channel<T>`, trace-off quietness) are strict carry-overs.

Slice 3 does **not** extend streaming to cancel-freed or
external-arrival admission paths. See §12.

## 3. Target invariant

A waiting request admitted into a completion-freed slot
under `--stream-all` must:

```text
- open a fresh hpx::lcos::local::channel<token_stream_event>
  on the recycled seq_id (the prior occupant's closed
  channel must not be inherited);
- have stream_closed reset to false at admission;
- have the per-slot cumulative stream_tokens_emitted reset
  to zero at admission;
- emit exactly rr.n_decoded token_stream_event{kind=token}
  events on its own fresh channel;
- close exactly once with kind=closed,
  reason=completed;
- satisfy streamed_hash == rr.hash, where streamed_hash is
  the FNV-1a fold of the int32_t token ids the admitted
  request received over its own channel;
- satisfy rr.status == completed and
  rr.admission_source == completion_freed;
- carry rr.previous_request_id == the completed sibling's
  request_id and rr.reused_seq_id == the freed seq_id;
- leave residual KV empty across all n_seq_max slots at
  engine end.
```

The admitted request's streamed token vector is gated
**independent of the previous occupant's vector**: a
distinct vector identity (the rebound `receiver`) is what
the gate consumes, and the gate asserts that the admitted
vector length equals `rr.n_decoded` and its hash equals
`rr.hash`, not the previous occupant's hash.

## 4. Smoke shape

The Slice 3 smoke is the completion-freed admission
analogue of the Slice 1 smoke. Same model, prompt, decode
policy; the admission surface is exercised via a single
preloaded waiter that the engine binds into the slot freed
by the original active seq's natural completion.

```text
--stream-all
--n-seqs 2
--n-active 1
--n-waiting 1
--decode-budget-mix 8
--waiting-budget 16
--reuse-completed
--cancel-plan none
--n-external-arrivals 0
--repeat 2
```

Notes on the shape:

- `--n-active 1` + `--n-waiting 1` produces one original
  active seq and one preloaded waiter.
- `--decode-budget-mix 8` makes the active seq's budget 8.
- `--waiting-budget 16` makes the waiter's budget 16.
- `--reuse-completed` enables completion-freed admission.
- `--cancel-plan none` keeps cancel-freed admission out of
  the smoke.
- `--n-external-arrivals 0` keeps the external-arrival
  path out of the smoke.
- `--repeat 2` re-asserts per-result determinism and
  per-seq streamed token vector determinism across two
  consecutive engine runs in the same process.

The `--stream-all` OFF regression continues to use the
canonical Slice 7 admission shape; every stream counter
remains exactly zero.

## 5. Expected per-request behavior

```text
request 0 (original active):
  seq_id                    = 0
  budget                    = 8
  admission_source          = none
  arrival_source            = preloaded
  status                    = completed
  close_reason              = completed
  streamed count            = 8
  streamed_hash             = 0x0619d4d1900c2365   (canonical anchor)

request 1 (admitted waiter):
  reused_seq_id             = 0
  previous_request_id       = 0
  admitted_at_iter          = 8
  admission_source          = completion_freed
  arrival_source            = preloaded
  budget                    = 16
  status                    = completed
  close_reason              = completed
  streamed count            = 16
  streamed_hash             = 0x833045f1e2ebf49f
```

The budget-16 admitted hash `0x833045f1e2ebf49f` is
observed and repeat-deterministic for this exact smoke
shape. It is **not** a new cross-shape canonical anchor:
its value is contingent on prompt, decoding policy, batch
shape, scheduling policy, and the specific completion-
freed admission boundary. Do not compare it to any other
budget-16 hash captured under a different shape.

`--repeat 2` is deterministic on the per-seq streamed
token vectors, the streamed hashes, the close reasons, and
the per-result tuple
`(seq_id, request_id, n_decoded, generated_tokens, hash,
done_iter, pos_max_at_clear, admitted_at_iter,
reused_seq_id, previous_request_id, admission_src,
arrival_src)`.

## 6. Required gates

Per streamed request (original active and admitted alike):

```text
streamed_close[req] == stream_close_reason::completed   # both reqs in this smoke

streamed_tokens[req].size() == rr.n_decoded

streamed_hash[req] == rr.hash   # FNV-1a over int32_t token ids
```

Slice 3-specific gates on the admitted request:

```text
rr.admission_source       == completion_freed
rr.arrival_source         == preloaded
rr.reused_seq_id          == previous occupant's seq_id
rr.previous_request_id    == previous occupant's request_id
rr.admitted_at_iter       == iter at which admit_one rebound the slot

streamed_tokens[admitted] is the admitted request's own
   token vector (the rebound receiver's reads), not the
   previous occupant's vector;

streamed_close[admitted]  == completed, distinct from the
   previous occupant's already-closed channel state;

stream_tokens_emitted[slot] at admitted close == admitted
   request's own decoded token count (the per-slot
   cumulative counter was reset on rebind, see §11).
```

Engine-side stream-counter gates (derived from the per-
`request_result` status of the union of original-active
and admitted-streamed requests):

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

Carry-over invariants (unchanged from Slices 1 and 2):

- residual KV empty across all `n_seq_max` slots at engine
  end;
- with `LLAMA_HPX_CB_TRACE` unset, zero `[hpx-cb-gate]
  event=` lines on stderr (trace-off is one atomic load +
  early return per call site);
- with `--stream-all` OFF, every stream counter is exactly
  zero and the receiver vector is empty;
- `--repeat 2` deterministic on the per-result tuple, the
  per-seq streamed token vectors, the streamed hashes,
  and the close reasons;
- single engine task per repeat, single `llama_context`
  ownership, no `llama_decode` call interrupted, every
  `llama_decode` returns 0.

## 7. Stream counter expectations

Per repeat (the smoke runs `--repeat 2`, so trace totals
are double):

```text
streams_opened              = 2
streams_closed_completed    = 2
streams_closed_cancelled    = 0
streams_closed_error        = 0
stream_tokens_emitted_total = 24       (= 8 + 16)
```

Across `--repeat 2`, engine-side counters double exactly:

```text
streams_opened              = 4
streams_closed_completed    = 4
streams_closed_cancelled    = 0
streams_closed_error        = 0
stream_tokens_emitted_total = 48
```

## 8. Trace evidence

Slice 3 uses the same trace event set as Slices 1 and 2;
no new event name is introduced. Under
`LLAMA_HPX_CB_TRACE=1` the smoke (`--repeat 2`) emits
exactly:

```text
token_stream_opened           == 4     (2 per repeat × 2 repeats)
token_stream_token            == 48    (24 per repeat × 2 repeats)
token_stream_closed           == 4
  reason=completed            == 4
  reason=cancelled            == 0
request_admitted_live         == 2
seq_reused                    == 2
admitted_prefilled            == 2
seq_complete                  == 4     (1 original + 1 admitted, per repeat × 2)
kv_cleared                    == 4
promise_fulfilled             == 4
```

`token_stream_token` event count equals
`stream_tokens_emitted_total` across the run
(`48 == 48`); the Slice 1 1:1
`token_stream_token`-per-streamed-token invariant carries
over, including across the rebind boundary.

The trace event ordering as captured in
`local/slice10_stream_step3_smoke_traceon.stderr` (one
repeat) is:

```text
event=token_stream_opened request=0 seq_id=0
... (8 × event=token_stream_token request=0 seq_id=0) ...
event=seq_complete seq=0 budget=8 done_iter=7 hash=0x0619d4d1900c2365
event=kv_cleared seq=0 pos_max_at_clear=12 cross_talk_ok=1
event=token_stream_closed request=0 seq_id=0 n_tokens=8 reason=completed
event=promise_fulfilled seq=0 ttc_us=<int>
event=seq_reused seq_id=0 previous_owner=0 new_owner=1 iter=8 admission_source=completion_freed arrival_source=preloaded
event=request_admitted_live request=1 reused_seq_id=0 iter=8 admission_source=completion_freed arrival_source=preloaded
event=token_stream_opened request=1 seq_id=0
event=admitted_prefilled request=1 seq_id=0 first_token=2259 admission_source=completion_freed
... (16 × event=token_stream_token request=1 seq_id=0) ...
event=seq_complete seq=0 budget=16 done_iter=23 hash=0x833045f1e2ebf49f
event=kv_cleared seq=0 pos_max_at_clear=20 cross_talk_ok=1
event=token_stream_closed request=1 seq_id=0 n_tokens=16 reason=completed
event=promise_fulfilled seq=0 ttc_us=<int>
```

Three ordering observations from the capture, gated only
as "the trace shows", not as a strict-ordering correctness
invariant:

- For each lifecycle close, the engine emits
  `seq_complete` → `kv_cleared` → `token_stream_closed` →
  `promise_fulfilled`. The terminal `kind=closed` stream
  event is therefore sent **after** the per-seq KV clear
  and cross-talk check have already passed, and the
  promise is fulfilled **after** `channel.close()`.
- On the admission boundary the engine emits
  `seq_reused` → `request_admitted_live` →
  `token_stream_opened` → `admitted_prefilled`: the
  rebound stream is opened **before** the admitted
  request's prefill row is published.
- `token_stream_opened request=1` is distinct from
  `token_stream_opened request=0`; the admitted request's
  fresh channel is opened on the same `seq_id=0` after
  the previous occupant's `token_stream_closed` and
  `promise_fulfilled` events, and the admitted request's
  `token_stream_closed` carries `n_tokens=16` (the
  admitted request's own decoded count), not 24
  (previous occupant + admitted cumulative).

## 9. Results

Streaming Slice 3 lands `HPX_CB_STREAM_STEP3: PASS` on all
four runs in the close-out matrix (Metal+HPX build,
TinyLlama Q4_K_M, greedy, `"Hello, my name is"`):

```text
stream-off regression on Slice 7 shape : HPX_CB_STREAM_STEP3: PASS
Slice 3 stream smoke (trace off)       : HPX_CB_STREAM_STEP3: PASS
Slice 3 trace-on stream smoke          : HPX_CB_STREAM_STEP3: PASS
Slice 3 repeat (cross-invocation)      : HPX_CB_STREAM_STEP3: PASS
```

Closeout captures:

```text
local/slice10_stream_step3_regression_streamoff.{stdout,stderr}
local/slice10_stream_step3_smoke.{stdout,stderr}
local/slice10_stream_step3_smoke_traceon.{stdout,stderr}
local/slice10_stream_step3_smoke_repeat2.{stdout,stderr}
```

Detailed evidence is in
`tools/hpx-continuous-batch-gate/results.md` under the
*Streaming Slice 3 results — admitted-request streaming on
the completion-freed admission path* section.

Carry-over gates passed on every Slice 3 run:

- residual KV empty across all `n_seq_max` slots at engine
  end (`residual_kv_empty = true`);
- trace-off quietness: zero `[hpx-cb-gate] event=` lines
  on stderr with `LLAMA_HPX_CB_TRACE` unset;
- `--repeat 2` deterministic on per-result tuple and per-
  seq streamed token vectors;
- admitted stream did not inherit the previous occupant's
  token vector or close reason.

## 10. HPX-native design note

Streaming Slice 3 stays HPX-native at the orchestration /
streaming-gate boundary:

- The streaming substrate remains
  `hpx::lcos::local::channel<token_stream_event>`.
- The engine HPX task is the sole producer of stream
  events for every slot, including the admitted request
  after rebind.
- `main()` (the gate consumer task) is the sole consumer
  via the matching `receive_channel`.
- Stream events carry only an `int32_t` token id and a
  close reason. No `llama_context`, KV state, logits, or
  any other llama.cpp object ever crosses the channel.
- Only the engine HPX task touches llama.cpp execution
  state (`llama_context`, `llama_batch`, `llama_decode`,
  `llama_memory_seq_*`, `llama_get_logits_ith`). The
  rebind happens inside the engine task on the admission
  boundary.

Synchronization invariants under Slice 3:

- No new `std::mutex` is introduced.
- No new HPX primitive (channel, promise, future, spinlock)
  is introduced.
- The existing `admitted_futures_mtx_` critical section is
  reused only for engine-to-main handoff of the
  `{request_id, receiver}` bundle. This lock guards
  orchestration metadata, not llama.cpp execution state.
- The `hpx::promise<request_result>` per slot is still
  single-shot and is fulfilled only after the channel
  closes, after per-seq KV clear, and after the cross-talk
  check passes.

Failure mode hardness:

- HPX runtime startup/shutdown remains process-level; the
  engine object does not start or stop HPX.
- Slice 3 fails closed on HPX/runtime/model/decode errors;
  no silent fallback to a non-HPX path is introduced.

## 11. Risks and caveats

Source-side caveats surfaced or addressed in the Slice 3
work:

1. **Pre-existing budget round-robin gate bug (fixed).**
   The gate's "original active mix round-robin" comment
   said the loop iterated over the requested decode budget
   mix, but it actually iterated over `uniq_budgets`, which
   conflated original-active and admitted budgets. Under
   the Slice 3 Candidate A shape this caused the original-
   active loop to consider the admitted (waiting) budget.
   The loop was changed to iterate over
   `args.decode_budget_mix`. The fix is bundled with Slice
   3 because the Slice 3 smoke is the shape that surfaced
   it; it does not change behavior on any Slice 1 / Slice
   2 / Slices 1–7 admission smoke.

2. **`stream_tokens_emitted` reset on rebind.** The per-
   slot cumulative counter is reset to zero in `admit_one`
   so that `token_stream_closed` for the admitted request
   reports the admitted request's own token count, not the
   previous occupant + admitted cumulative. Without this
   reset, the trace-on counter gate
   (`stream_tokens_emitted_total == sum(rr.n_decoded over
   streamed)`) would over-count by the previous occupant's
   length on every recycled slot.

3. **Rebind currently gated on
   `admission_source == completion_freed` only.** The
   `admit_one` rebind predicate is currently scoped to the
   completion-freed admission path. Cancel-freed admission
   and external-arrival admission still produce
   `request_result`s whose slots inherit `stream_closed =
   true` from the prior occupant and silently skip
   streaming for the admitted request. Streaming those
   paths is deferred to a later slice (see §12).

4. **Admitted-stream hash is shape-scoped.** The observed
   budget-16 admitted hash `0x833045f1e2ebf49f` is
   repeat-deterministic for the Candidate A shape only.
   It is not a new cross-shape canonical anchor; do not
   compare it to other budget-16 hashes captured under
   different shapes, prompts, or scheduling policies.

Hard invariant: only the engine HPX task touches
`llama_context`, `llama_batch`, `llama_decode`,
`llama_memory_seq_*`, or `llama_get_logits_ith`. Stream
events carry only an `int32_t` token id and a close
reason; no llama.cpp state ever crosses the channel.

## 12. Deferred work

Out of scope for Streaming Slice 3:

- **Cancel-freed admission streaming.** A waiter admitted
  into a slot freed by a sibling's cooperative
  cancellation still inherits the prior occupant's
  `stream_closed = true` and is not streamed. The
  `admit_one` rebind predicate must be extended to cover
  `admission_source == cancel_freed`. Deferred.
- **External-arrival admission streaming.** A waiter
  admitted from the `arrival_msg` inbox (Slice 6 / 7
  surface) is also not streamed at present. The
  `admit_one` rebind predicate must be extended to cover
  `arrival_source == external` admissions. Deferred.
- **HTTP / gRPC / Unix-socket streaming.** The streaming
  surface remains an in-process HPX local channel between
  the engine task and `main()`. No network framing, no
  chat-template assembly, no `tools/server` integration.
  See `docs/hpx/continuous_batching_gate_vs_serving_layer.md`
  for the gate-vs-serving-layer boundary.
- **Backpressure / bounded-channel policy.** `channel<T>`
  is still unbounded so the engine never suspends on the
  consumer. A bounded-channel slice is deferred.
- **Multi-cycle slot reuse with streaming.** The Slice 3
  smoke admits one waiter into one freed slot per run.
  Chaining waiting → admitted → completed → second-
  waiting → second-admitted on the same slot, each with
  its own fresh stream, is out of scope.
- **Per-request prompts and per-request sampling.** The
  smoke still uses TinyLlama, `"Hello, my name is"`, and
  greedy decoding. Per-request prompts and per-request
  sampling configuration are deferred.
- **Performance comparisons.** Streaming Slice 3 is
  correctness-only.

## 13. Safe claim language

Supported by Streaming Slice 3 evidence and safe to use
verbatim:

- "Admitted-request streaming is gated at the HPX
  continuous-batching gate boundary on the completion-
  freed admission path: a waiting request bound to a slot
  freed by a sibling's natural completion opens a fresh
  HPX local-channel stream, emits exactly `rr.n_decoded`
  token events whose FNV-1a fold equals `rr.hash`, and
  closes with `reason=completed`."
- "The admitted request's streamed token vector does not
  inherit the previous occupant's vector or close reason;
  the per-slot stream channel is rebound and the per-slot
  cumulative `stream_tokens_emitted` counter is reset at
  admission."
- "Streaming Slice 3 introduces no HTTP / server
  streaming, no backpressure, no new CLI flag, no new HPX
  primitive, no new `std::mutex`, and no performance
  claim. The engine-to-main handoff of the admitted
  request's stream uses the existing
  `admitted_futures_mtx_` critical section for an explicit
  `{request_id, receiver}` bundle."
- "Engine-side stream counters
  (`streams_closed_completed`, `streams_closed_cancelled`,
  `streams_closed_error`, `streams_opened`,
  `stream_tokens_emitted_total`) match the corresponding
  counts derived from the per-`request_result` status of
  the union of original-active and admitted-streamed
  requests on every gated run."
- "All Slice 1–7 admission/cancellation/lifecycle
  invariants and all Streaming Slice 1 / Slice 2
  streaming invariants remain strict carry-overs under
  `HPX_CB_STREAM_STEP3`."

Not supported by Slice 3 and to be avoided:

- "Streaming Slice 3 covers cancel-freed admission
  streaming." (The `admit_one` rebind predicate is scoped
  to `admission_source == completion_freed`; cancel-freed
  admitted requests still silently skip streaming.)
- "Streaming Slice 3 covers external-arrival streaming."
  (External admissions are also not yet rebound; see §12.)
- "Streaming Slice 3 improves performance." (No
  comparative benchmark is run; this is a correctness /
  lifecycle gate.)
- "Streaming Slice 3 integrates with `llama-server`."
  (No network adapter; the stream is in-process only.)
- "The admitted budget-16 hash `0x833045f1e2ebf49f` is a
  new canonical cross-shape anchor." (It is repeat-
  deterministic for the Candidate A shape only and is
  not a cross-shape correctness invariant under
  CLAUDE.md.)
