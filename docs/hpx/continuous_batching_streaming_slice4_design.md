# Continuous batching — Streaming Slice 4 design

## 1. Purpose

Streaming Slice 4 closes the cancel-freed admission arm of
the HPX-native per-request token stream. Slices 1 and 2
gated streaming on the completion and cancellation paths
for original-active seqs. Slice 3 extended streaming to
**admitted** requests on the **completion-freed** admission
path. Slice 4 extends streaming to **admitted** requests on
the **cancel-freed** admission path: when a streamed
original request is cancelled and frees its slot, the
waiting request admitted into that slot is itself streamed
end-to-end on a fresh HPX local-channel stream, emits
exactly `rr.n_decoded` token events, has
`streamed_hash == rr.hash`, and closes with
`reason=completed` without inheriting the cancelled
occupant's close reason or stream state.

Slice 4 is a gate-only / orchestration-only advance with a
single engine-side delta inside the existing `admit_one`
rebind block: the predicate that was
`src == admission_source::completion_freed` becomes
`src == admission_source::completion_freed ||
 src == admission_source::cancel_freed`. No new HPX
primitive, no new `std::mutex`, no new CLI flag, no new
trace event name, no new channel type, no CMake change.
The streaming substrate, ownership rules, payload
(`int32_t` token id + close reason), and trace event set
are unchanged from Slices 1–3.

This is a correctness/lifecycle gate. No performance claim
is made.

## 2. Relation to Streaming Slices 1–3

Slices 1–7 (the admission gate sequence,
`HPX_CB_ADMIT_STEP1` … `HPX_CB_ADMIT_STEP7`) remain a
carry-over invariant: every admission, cancellation, and
lifecycle invariant gated there still holds under
`HPX_CB_STREAM_STEP4`. The `--stream-all` OFF regression
gate on the canonical Slice 7 admission shape continues to
pass with all engine-side stream counters at exactly zero.

Streaming Slice 1 (`HPX_CB_STREAM_STEP1: PASS`) proved the
completion arm for an original active seq. Streaming Slice
2 (`HPX_CB_STREAM_STEP2: PASS`) proved the cancellation
arm for an original active seq. Streaming Slice 3
(`HPX_CB_STREAM_STEP3: PASS`) proved admitted-request
streaming over a completion-freed slot.

Streaming Slice 4 (`HPX_CB_STREAM_STEP4: PASS`) is the
symmetric extension of Slice 3 to the cancel-freed
admission path. The Slice 1 and Slice 2 streaming
invariants (single producer, single consumer, terminal
`kind=closed` before `channel.close()`, no llama.cpp state
crossing the channel, unbounded `channel<T>`, trace-off
quietness) and all Slice 3 admitted-streaming invariants
are strict carry-overs. The Slice 3 completion-freed
smoke and its gates are unchanged.

Slice 4 does **not** extend streaming to the external-
arrival admission path. See §12.

## 3. Target invariant

A waiting request admitted into a cancel-freed slot under
`--stream-all` must:

```text
- open a fresh hpx::lcos::local::channel<token_stream_event>
  on the recycled seq_id (the cancelled prior occupant's
  closed channel must not be inherited);
- have stream_closed reset to false at admission;
- have the per-slot cumulative stream_tokens_emitted reset
  to zero at admission (so the admitted close trace
  reports the admitted-request count, not prev_cancelled +
  admitted cumulative);
- emit exactly rr.n_decoded token_stream_event{kind=token}
  events on its own fresh channel;
- close exactly once with kind=closed, reason=completed
  (the cancelled occupant's reason=cancelled close is NOT
  inherited);
- satisfy streamed_hash == rr.hash, where streamed_hash is
  the FNV-1a fold of the int32_t token ids the admitted
  request received over its own channel;
- satisfy rr.status == completed and
  rr.admission_source == cancel_freed;
- carry rr.previous_request_id == the cancelled sibling's
  request_id and rr.reused_seq_id == the freed seq_id;
- leave residual KV empty across all n_seq_max slots at
  engine end.
```

The admitted request's streamed token vector is drained
through a distinct receiver (the rebound `receiver`) that
the gate consumes; the cancelled occupant's vector is
drained through the seq-id-indexed `streamed_tokens[seq]`
vector populated by the Slice 1 receiver pump. The two
collections are populated independently, so structural
independence of the admitted stream is by construction.

## 4. Smoke-shape adjustment

The originally-sketched `--n-active 1` shape (one active,
one waiter, one cancel) **cannot** fire cancel-freed
admission. The engine's admission boundary uses a
deterministic one-iter delay:
`admission_eligible_count` is snapshotted at the top of
iter K **before** the cancellation observation pass runs,
so a cancel observed at iter K is admissible at iter K+1,
not iter K. With `--n-active 1`, the single active seq is
cancelled at iter K and the engine then exits via
`if (!any_active())` at the end of iter K before iter K+1
runs. No admission happens. The Slice 4 coverage gate
correctly flags this as misconfigured.

The smallest viable shape uses **two active seqs**: one
gets cancelled at iter K, the other has a long enough
budget to keep the decode loop alive across iter K+1.
Additionally, because greedy decoding from the same
prompt produces the same first-N generated tokens
regardless of admission path, choosing cancel-after `8`
versus admitted budget `16` makes the cancelled-prefix
token vector and the admitted-streamed vector differ in
length — which is what the existing Slice 3 admitted-only
independence gate (a full token-vector compare) actually
checks. The Slice 4 independence evidence is therefore
**length-based** in this smoke. See §11 for the
implications.

## 5. Smoke shape

```text
--stream-all
--n-seqs 3
--n-active 2
--n-waiting 1
--decode-budget-mix 64,256
--waiting-budget 16
--cancel-plan 0
--cancel-after 8
--n-external-arrivals 0
--repeat 2
```

Notes on the shape:

- `--n-active 2` + `--n-waiting 1` produces two original
  active seqs (seq 0 budget 64, seq 1 budget 256) and one
  preloaded waiter (waiting_budget 16).
- `--cancel-plan 0 --cancel-after 8` cancels seq 0 at the
  iter where seq 0's `n_decoded` reaches 8.
- The surviving active seq 1 (budget 256) keeps the
  decode loop alive across the cancel→admit boundary at
  iter 9.
- `--n-external-arrivals 0` keeps the external-arrival
  path out of the smoke.
- `--repeat 2` re-asserts per-result determinism and per-
  seq streamed token vector determinism across two
  consecutive engine runs in the same process.

The `--stream-all` OFF regression continues to use the
canonical Slice 7 admission shape; every stream counter
remains exactly zero.

## 6. Expected per-request behavior

```text
request 0 (original active, seq_id=0):
  admission_source         = none
  arrival_source            = preloaded
  budget                    = 64
  status                    = cancelled
  cancel_observed_iter      = 8
  n_decoded                 = 8
  n_decoded_at_cancel       = 8
  close_reason              = cancelled
  streamed count            = 8

request 1 (original active, seq_id=1):
  admission_source          = none
  arrival_source            = preloaded
  budget                    = 256
  status                    = completed
  close_reason              = completed
  streamed count            = 256
  streamed_hash             = 0x8790fbe5a60c9ae6
                              (within-shape uniqueness +
                               --repeat 2 determinism)

request 2 (admitted waiter, seq_id=0):
  reused_seq_id             = 0
  previous_request_id       = 0
  admitted_at_iter          = 9                     (== cancel_after + 1)
  admission_source          = cancel_freed
  arrival_source            = preloaded
  budget                    = 16
  status                    = completed
  close_reason              = completed
  streamed count            = 16
  streamed_hash             = 0x833045f1e2ebf49f
                              (observed and
                               repeat-deterministic for
                               this smoke; matches the
                               Slice 3 completion-freed
                               admitted budget-16 hash for
                               the same prompt/policy/
                               budget under greedy
                               decoding; NOT a new
                               cross-shape canonical
                               anchor)
```

`--repeat 2` is deterministic on the per-seq streamed
token vectors, the streamed hashes, the close reasons, and
the per-result tuple.

## 7. Required gates

Per streamed request (original active and admitted alike):

```text
streamed_close[req]              == expected_close_from(rr.status)

streamed_tokens[req].size()      == rr.n_decoded

if rr.status == cancelled:
  streamed_tokens[req].size()    == rr.n_decoded_at_cancel

streamed_hash[req]               == rr.hash    (FNV-1a)
```

Slice 4-specific gates on the cancel-freed admitted
request:

```text
rr.admission_source              == cancel_freed
rr.arrival_source                == preloaded
rr.reused_seq_id                 == the cancelled occupant's seq_id
rr.previous_request_id           == the cancelled occupant's request_id
rr.admitted_at_iter              == iter at which admit_one rebound the slot
                                     (== cancel_after + 1 for this smoke)

admitted_streamed_close[rr.request_id]
                                 == completed
                                    (the cancelled
                                     occupant's close
                                     reason `cancelled`
                                     is NOT inherited)

streamed_tokens[admitted]        is the admitted request's
                                 own token vector (the
                                 rebound receiver's
                                 reads), populated through
                                 admitted_streamed_tokens
                                 keyed by request_id, not
                                 the seq-id-indexed
                                 streamed_tokens vector;

stream_tokens_emitted[slot] at admitted close == admitted
   request's own decoded token count (the per-slot
   cumulative counter was reset on rebind; see §11).
```

Engine-side stream-counter gates (derived from the per-
`request_result` status of the union of original-active
and admitted-streamed requests; the Slice 3 form extended
to include `admission_source::cancel_freed`):

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

Slice 4 coverage gate (new):

```text
streamed_admitted_cancel_freed_count >= 1   when
   --stream-all && !args.cancel_plan.empty() && --n-waiting > 0
```

The Slice 3 completion-freed coverage gate
(`streamed_admitted_completion_freed_count >= 1` when
`--stream-all && --reuse-completed && --n-waiting > 0`) is
**unchanged**.

Carry-over invariants (unchanged from Slices 1–3):

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
  `llama_decode` returns 0.

## 8. Stream counter expectations

Per repeat (the smoke runs `--repeat 2`, so trace totals
are double):

```text
streams_opened              = 3        (orig 0 + orig 1 + admitted 2)
streams_closed_completed    = 2        (orig 1 + admitted 2)
streams_closed_cancelled    = 1        (orig 0)
streams_closed_error        = 0
stream_tokens_emitted_total = 280      (= 8 + 256 + 16)
```

Across `--repeat 2`, engine-side counters double exactly.

## 9. Trace evidence

Slice 4 uses the same trace event set as Slices 1–3; no
new event name is introduced. Under
`LLAMA_HPX_CB_TRACE=1` the smoke (`--repeat 2`) emits:

```text
token_stream_opened           == 6    (3 per repeat × 2 repeats)
token_stream_token            == 560  (280 per repeat × 2 repeats)
token_stream_closed           == 6
  reason=completed            == 4
  reason=cancelled            == 2
cancel_observed               == 2
cancel_kv_cleared             == 2
cancel_future_fulfilled       == 2
request_admitted_live         == 2
seq_reused                    == 2
admitted_prefilled            == 2
```

Trace event ordering on the cancel→admit boundary, as
captured in
`local/slice11_stream_step4_smoke_traceon.stderr` (one
repeat; gated only as "the trace shows", not as a
strict-ordering correctness invariant):

```text
event=token_stream_opened request=0 seq_id=0
event=token_stream_opened request=1 seq_id=1
... 8 × event=token_stream_token request=0 seq_id=0 ...
event=cancel_observed seq=0 iter=8 n_decoded=8
event=token_stream_closed request=0 seq_id=0 n_tokens=8 reason=cancelled
event=cancel_kv_cleared seq=0 pos_max_at_clear=12 cross_talk_ok=1
event=cancel_future_fulfilled seq=0 status=cancelled ttc_us=<int>
event=seq_reused seq_id=0 previous_owner=0 new_owner=2 iter=9 admission_source=cancel_freed arrival_source=preloaded
event=request_admitted_live request=2 reused_seq_id=0 iter=9 admission_source=cancel_freed arrival_source=preloaded
event=token_stream_opened request=2 seq_id=0
event=admitted_prefilled request=2 seq_id=0 first_token=2259 admission_source=cancel_freed
... 16 × event=token_stream_token request=2 seq_id=0 ...
event=token_stream_closed request=2 seq_id=0 n_tokens=16 reason=completed
... event=token_stream_token request=1 seq_id=1 continue throughout ...
event=token_stream_closed request=1 seq_id=1 n_tokens=256 reason=completed
```

`token_stream_opened request=2` follows the rebind, and
`token_stream_closed request=2` carries the admitted
request's own token count (`n_tokens=16`), not 24
(cancelled-prefix + admitted cumulative).

## 10. Results

Streaming Slice 4 lands `HPX_CB_STREAM_STEP4: PASS` on all
four runs in the close-out matrix (Metal+HPX build,
TinyLlama Q4_K_M, greedy, `"Hello, my name is"`):

```text
stream-off regression on Slice 7 shape : HPX_CB_STREAM_STEP4: PASS
Slice 4 stream smoke (trace off)       : HPX_CB_STREAM_STEP4: PASS
Slice 4 trace-on stream smoke          : HPX_CB_STREAM_STEP4: PASS
Slice 4 cross-invocation repeat        : HPX_CB_STREAM_STEP4: PASS
```

Closeout captures:

```text
local/slice11_stream_step4_regression_streamoff.{stdout,stderr}
local/slice11_stream_step4_smoke.{stdout,stderr}
local/slice11_stream_step4_smoke_traceon.{stdout,stderr}
local/slice11_stream_step4_smoke_repeat2.{stdout,stderr}
```

Detailed evidence is in
`tools/hpx-continuous-batch-gate/results.md` under the
*Streaming Slice 4 results — admitted-request streaming
over cancel-freed slot* section.

Carry-over gates passed on every Slice 4 run:

- residual KV empty across all `n_seq_max` slots at engine
  end (`residual_kv_empty = true`);
- trace-off quietness: zero `[hpx-cb-gate] event=` lines
  on stderr with `LLAMA_HPX_CB_TRACE` unset;
- `--repeat 2` deterministic on per-result tuple and per-
  seq streamed token vectors;
- admitted stream did not inherit the cancelled
  occupant's close reason or stream state.

## 11. HPX-native design note

Streaming Slice 4 stays HPX-native at the orchestration /
streaming-gate boundary:

- The streaming substrate remains
  `hpx::lcos::local::channel<token_stream_event>`.
- The engine HPX task is the sole producer of stream
  events for every slot, including the cancel-freed
  admitted request after rebind.
- `main()` (the gate consumer task) is the sole consumer
  via the matching `receive_channel`.
- Stream events carry only an `int32_t` token id and a
  close reason. No `llama_context`, KV state, logits, or
  any other llama.cpp object ever crosses the channel.
- Only the engine HPX task touches llama.cpp execution
  state (`llama_context`, `llama_batch`, `llama_decode`,
  `llama_memory_seq_*`, `llama_get_logits_ith`). The
  rebind happens inside the engine task on the cancel-
  freed admission boundary.

Synchronization invariants under Slice 4:

- No new `std::mutex` is introduced.
- No new HPX primitive (channel, promise, future,
  spinlock) is introduced.
- The existing `admitted_futures_mtx_` critical section
  is reused only for engine-to-main handoff of the
  `{request_id, receiver}` bundle, the same way Slice 3
  reused it. This lock guards orchestration metadata, not
  llama.cpp execution state.
- The `hpx::promise<request_result>` per slot is still
  single-shot and is fulfilled only after the channel
  closes, after per-seq KV clear, and after the cross-
  talk check passes.

Failure mode hardness:

- HPX runtime startup/shutdown remains process-level; the
  engine object does not start or stop HPX.
- Slice 4 fails closed on HPX/runtime/model/decode errors;
  no silent fallback to a non-HPX path is introduced.

## 12. Risks and caveats

1. **The original `--n-active 1` sketch cannot fire
   cancel-freed admission.** The engine's one-iter
   cancel→admit delay combined with the `!any_active()`
   exit means a single active seq's cancellation tears
   down the engine before the admit-at-K+1 boundary. The
   smoke must keep ≥ 1 surviving active seq across the
   boundary. The Slice 4 coverage gate fail-closes if a
   misconfigured shape silently skips the path.

2. **The smoke uses `--cancel-after 8` rather than 16.**
   The pre-existing Slice 3 admitted-only independence
   gate is a full token-vector compare. Greedy decoding
   on the same prompt produces the same first-N
   generated tokens regardless of admission path, so a
   16-vs-16-token shape (cancelled prefix == admitted
   budget) would make the vectors byte-identical and
   trip the gate as a false positive. Sizing
   cancel-after to 8 and admitted budget to 16 makes the
   vectors differ in length, so equality is structurally
   impossible. The Slice 4 invariant is unaffected.

3. **The vector-inequality independence evidence in this
   smoke is length-based.** Do not overstate it as
   semantic token divergence. What the gate actually
   proves is structural independence:

   - a freshly default-constructed
     `hpx::lcos::local::channel<token_stream_event>` was
     bound on admission;
   - the admitted request's close reason did **not**
     inherit the cancelled occupant's `cancelled`
     reason — the admitted close is `completed`;
   - the engine-side per-slot `stream_tokens_emitted`
     counter was reset on rebind so the admitted close
     trace reports the admitted count, not previous +
     admitted cumulative;
   - the admitted vector is drained through
     `admitted_streamed_tokens[request_id]`, a distinct
     map from the seq-id-indexed `streamed_tokens[seq]`
     that holds the cancelled-prefix vector.

   At equal indices the admitted-stream tokens may match
   the cancelled-prefix tokens (greedy decoding, same
   prompt). That is not a defect; it is a property of
   greedy decoding.

4. **Cancel-freed admitted hash is shape-scoped.** The
   observed budget-16 admitted hash
   `0x833045f1e2ebf49f` equals the Slice 3 completion-
   freed admitted budget-16 hash for the same prompt /
   policy / budget. This is expected under greedy
   decoding on `"Hello, my name is"` — the first 16
   generated tokens are deterministic regardless of
   which admission path bound the slot. It is **not** a
   new cross-shape canonical anchor; it is the
   budget-16 fingerprint for this specific prompt /
   policy / batch-shape combination and is gated only as
   `streamed_hash == rr.hash` within the Slice 4 smoke.

Hard invariant: only the engine HPX task touches
`llama_context`, `llama_batch`, `llama_decode`,
`llama_memory_seq_*`, or `llama_get_logits_ith`. Stream
events carry only an `int32_t` token id and a close
reason; no llama.cpp state ever crosses the channel.

## 13. Deferred work

Out of scope for Streaming Slice 4:

- **External-arrival admitted streaming.** A waiter
  admitted from the `arrival_msg` inbox (Slice 6 / 7
  surface) is still not streamed. The `admit_one` rebind
  predicate now covers `completion_freed` (Slice 3) and
  `cancel_freed` (Slice 4) but excludes
  `arrival_source::external` admissions. Those slots
  inherit `stream_closed = true` from the prior occupant
  and silently skip streaming. A later slice will extend
  the rebind to cover external arrivals.
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
- **Multi-cycle slot reuse with streaming.** The Slice 4
  smoke admits one waiter into one freed slot per run.
  Chaining waiting → admitted → cancelled →
  second-waiting → second-admitted on the same slot,
  each with its own fresh stream, is out of scope.
- **Per-request prompts and per-request sampling.** The
  smoke still uses TinyLlama, `"Hello, my name is"`, and
  greedy decoding.
- **Engine-failure stream semantics.** The defensive
  `error` close path exists in source; no smoke
  exercises it.
- **Performance comparisons.** Streaming Slice 4 is
  correctness-only.

## 14. Safe claim language

Supported by Streaming Slice 4 evidence and safe to use
verbatim:

- "Cancel-freed admitted-request streaming is gated at
  the HPX continuous-batching gate boundary: after a
  streamed original request is cancelled and frees its
  slot, a waiting request admitted into that slot
  receives a fresh HPX local-channel stream, emits
  exactly `rr.n_decoded` token events whose FNV-1a fold
  equals `rr.hash`, closes with `reason=completed`, and
  does not inherit the cancelled occupant's close
  reason or stream state."
- "The Slice 4 `admit_one` rebind predicate covers
  both `admission_source::completion_freed` (Slice 3
  scope) and `admission_source::cancel_freed` (Slice 4
  scope). External-arrival admitted streaming remains
  deferred."
- "Streaming Slice 4 introduces no new HPX primitive,
  no new `std::mutex`, no new CLI flag, no new trace
  event name, no new channel type, no CMake change, no
  HTTP / server streaming, no backpressure, and no
  performance claim. The engine-to-main handoff of the
  cancel-freed admitted request's stream reuses the
  existing `admitted_stream_handoffs_` vector and the
  existing `admitted_futures_mtx_` critical section
  introduced in Slice 3."
- "Engine-side stream counters
  (`streams_closed_completed`, `streams_closed_cancelled`,
  `streams_closed_error`, `streams_opened`,
  `stream_tokens_emitted_total`) match the corresponding
  counts derived from the per-`request_result` status of
  the union of original-active and admitted-streamed
  requests (with admitted-streamed extended to include
  `admission_source::cancel_freed`) on every gated run."
- "All Slice 1–7 admission/cancellation/lifecycle
  invariants and all Streaming Slice 1 / Slice 2 /
  Slice 3 streaming invariants remain strict carry-overs
  under `HPX_CB_STREAM_STEP4`."

Not supported by Slice 4 and to be avoided:

- "Streaming Slice 4 covers external-arrival admission
  streaming." (The `admit_one` rebind predicate is
  scoped to `completion_freed || cancel_freed`;
  external-arrival admitted requests still silently
  skip streaming.)
- "Streaming Slice 4 improves performance." (No
  comparative benchmark is run; this is a correctness /
  lifecycle gate.)
- "Streaming Slice 4 integrates with `llama-server`."
  (No network adapter; the stream is in-process only.)
- "The independence gate proves semantic token
  divergence between the cancelled prefix and the
  admitted stream." (The gate is a full token-vector
  compare; the Slice 4 smoke makes that comparison
  inequality-by-length. See §11.)
- "The admitted budget-16 hash `0x833045f1e2ebf49f` is
  a new canonical cross-shape anchor." (It equals the
  Slice 3 completion-freed admitted budget-16 hash for
  the same prompt/policy/budget under greedy decoding
  and is not a cross-shape correctness invariant under
  CLAUDE.md.)
