# Continuous batching — Streaming Slice 2 design

## 1. Purpose

Streaming Slice 2 closes the cancellation arm of the
HPX-native per-request token stream introduced in Streaming
Slice 1. Slice 1 proved that, on the completion path, the
engine task is the sole producer of `token_stream_event`
values onto an `hpx::lcos::local::channel`, that main is the
sole consumer via the matching `receive_channel`, that
streamed token count equals `rr.n_decoded`, that
`streamed_hash == rr.hash`, and that the channel closes
exactly once with `reason=completed`. Slice 2 raises the
same correctness bar on the cancellation path: a streamed
request that the engine cancels at an iteration boundary
must close its channel with `reason=cancelled` after exactly
`rr.n_decoded_at_cancel` token events, satisfy
`rr.status == cancelled` with `streamed_hash == rr.hash`,
and still leave KV empty. Slice 2 gates these properties
semantically — through stream length, close reason,
`rr.status`, and `streamed_hash == rr.hash` — not through a
per-token trace-order assertion.

Streaming Slice 2 is a **gate-only** advance. No new HPX
primitive, no new CLI flag, no new engine-side behavior, no
new channel type, no new trace event name, no CMake change.
The engine-side `cancel_and_fulfill` flow already (since
Slice 1) sends a `kind=closed`, `reason=cancelled` event
onto the channel before calling `channel.close()` and
fulfilling `request_result` with `status = cancelled`. What
Slice 2 changes is the gate logic in
`tools/hpx-continuous-batch-gate/hpx-continuous-batch-gate.cpp`:
the close-reason check becomes status-aware, an explicit
`streamed_tokens.size() == rr.n_decoded_at_cancel` gate is
added for cancelled streamed requests, and the absolute
"completion-only" engine-side counter gates are replaced
with expected-from-`request_result` gates.

This is a correctness/lifecycle gate. No performance claim
is made.

## 2. Relation to Slices 1–7 and Streaming Slice 1

Slices 1–7 (the admission gate sequence,
`HPX_CB_ADMIT_STEP1` … `HPX_CB_ADMIT_STEP7`) remain a
carry-over invariant: every admission, cancellation, and
lifecycle invariant gated there still holds under
`HPX_CB_STREAM_STEP2`. The `--stream-all` OFF regression
gate on the canonical Slice 7 admission shape continues to
pass with all engine-side stream counters at exactly zero.

Streaming Slice 1 (`HPX_CB_STREAM_STEP1: PASS`) opened the
streaming gate sequence and proved the completion arm. Slice
1's lifecycle invariants — single producer (engine task),
single consumer (main), terminal `kind=closed` event sent
**before** `channel.close()`, no llama.cpp state crossing
the channel, unbounded `channel<T>` so the engine never
suspends on the consumer, trace events
(`token_stream_opened` / `token_stream_token` /
`token_stream_closed`) gated on `LLAMA_HPX_CB_TRACE=1` with
trace-off being one atomic load + early return — all carry
over unchanged in Slice 2.

Streaming Slice 2 inherits Slice 1's engine wiring. The
delta is purely on the gate side and is scoped to the
streamed request loop and the engine-side stream-counter
block.

## 3. Cancellation-aware streaming policy

Engine-side cancellation flow (already implemented in Slice
1, unchanged in Slice 2):

```text
cancel_and_fulfill(seq, iter, mem):
  1. record cancel_observed, cancel_observed_iter,
     n_decoded_at_cancel
  2. clear_and_check(seq, iter, mem)          # per-seq KV clear + cross-talk
  3. close_stream(seq, stream_close_reason::cancelled)
  4. fulfill_promise(seq, request_status::cancelled, ttc_us)

close_stream(seq, reason):
  - send token_stream_event{kind=closed, close_reason=reason}
  - channel.close()
  - increment streams_closed_<reason> on engine_result
  - emit token_stream_closed trace event with reason
```

Why no token id is emitted after `cancel_observed_iter`:

```text
Cancellation observation runs at the top of each decode
iteration boundary, **before** the active row set is built.
Once a seq is marked cancel-observed it becomes seq.done in
the shared clear path, so it cannot reach llama_decode /
argmax / publish_token for that iter or any later iter.
The last token id published is therefore the one emitted at
iter == cancel_observed_iter - 1 (i.e. iter (cancel_observed_iter - 1)
publish_token), and the streamed token vector for the
cancelled seq has length exactly n_decoded_at_cancel.
```

Slice 2's gates assert that semantic invariant without
needing a trace-order check.

## 4. Smoke shape

The Slice 2 smoke is the cancellation-aware analogue of the
Slice 1 smoke. Same model, prompt, decode policy, and budget
mix; one streamed seq is forced onto the cancellation path
via `--cancel-plan`.

```text
--stream-all
--n-seqs 3
--decode-budget-mix 8,64,256
--cancel-plan 1
--cancel-after 16
--n-waiting 0
--n-external-arrivals 0
--repeat 2
```

`--cancel-plan 1` schedules cancellation for `seq_id 1`
once it reaches `--cancel-after = 16` decoded tokens.
Because the budget mix assigns budgets round-robin
(`{8, 64, 256}`), seq 1 carries budget 64 and is the only
cancellation in the smoke.

The `--stream-all` OFF regression continues to use the
canonical Slice 7 admission shape (`--n-active 93`,
`--n-external-arrivals 6`, `--external-release-iter 8`,
`--cancel-plan 1,4,7,2,5,8`, `--cancel-after 16`) and
re-asserts that every stream counter is zero.

## 5. Expected per-request behavior

```text
seq=0  budget=8    status=completed  close=completed  streamed=8
seq=1  budget=64   status=cancelled  close=cancelled  streamed=16
                   n_decoded=16
                   n_decoded_at_cancel=16
                   cancel_observed_iter=16
                   cancel_kv_cleared: pos_max_at_clear=20
                                       cross_talk_ok=1
seq=2  budget=256  status=completed  close=completed  streamed=256
```

Engine-side stream counters per repeat:

```text
streams_opened              = 3
streams_closed_completed    = 2
streams_closed_cancelled    = 1
streams_closed_error        = 0
stream_tokens_emitted_total = 280       (= 8 + 16 + 256)
```

`--repeat 2` is deterministic on per-seq streamed token
vectors and close reasons; the smoke and a separate
cross-invocation repeat run produce byte-identical streamed
counts and close reasons.

## 6. Required gates

Per streamed request (`admission_src == none`,
`request_id == seq_id`, `seq_id ∈ [0, n_active)`):

```text
expected_close_reason = (rr.status == request_status::cancelled)
                          ? stream_close_reason::cancelled
                          : stream_close_reason::completed

streamed_close[seq] == expected_close_reason

streamed_tokens[seq].size() == rr.n_decoded

if rr.status == request_status::cancelled:
  streamed_tokens[seq].size() == rr.n_decoded_at_cancel

streamed_hash == rr.hash         # FNV-1a over int32_t token
                                 # ids; applies to both
                                 # completed and cancelled
                                 # streamed requests
```

Engine-side stream-counter gates (replaces the Slice 1
completion-only block):

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

Carry-over invariants (unchanged from Slice 1):

- residual KV empty across all `n_seq_max` slots at engine
  end;
- with `LLAMA_HPX_CB_TRACE` unset, zero `[hpx-cb-gate]
  event=` lines on stderr (trace-off is one atomic load +
  early return per call site);
- with `--stream-all` OFF, every stream counter is exactly
  zero and the receiver vector is empty;
- `--repeat 2` deterministic on the per-result tuple, the
  per-seq streamed token vectors, and the close reasons;
- single engine task per repeat, single
  `llama_context` ownership, no `llama_decode` call
  interrupted, every `llama_decode` returns 0.

The "no token id after `cancel_observed_iter`" property is
**not** gated as a trace-order check. The semantic gates
above (`streamed_tokens.size() == rr.n_decoded_at_cancel`,
`close_reason == cancelled`, `rr.status == cancelled`) cover
it without requiring `token_stream_token` to carry an
`iter` payload.

## 7. Trace evidence

Slice 2 uses the same trace event set as Slice 1; no new
event name is introduced. Under `LLAMA_HPX_CB_TRACE=1` the
smoke (`--repeat 2`) emits (per repeat — the smoke runs two
repeats, so totals are double):

```text
token_stream_opened           == 3
token_stream_token            == 280
token_stream_closed           == 3
  reason=completed            == 2
  reason=cancelled            == 1
cancel_requested              == 1
cancel_observed               == 1
cancel_kv_cleared             == 1
cancel_future_fulfilled       == 1
```

Cancel-arm event ordering on the cancelled seq (one
repeat):

```text
event=cancel_observed seq=1 iter=16 n_decoded=16
event=cancel_kv_cleared seq=1 pos_max_at_clear=20 cross_talk_ok=1
event=token_stream_closed request=1 seq_id=1 n_tokens=16 reason=cancelled
event=cancel_future_fulfilled seq=1 status=cancelled ttc_us=<int>
```

`token_stream_token` event count equals
`stream_tokens_emitted_total` per repeat
(`280 == 280`); the Slice 1 1:1
`token_stream_token`-per-streamed-token invariant carries
over.

## 8. Results

Streaming Slice 2 lands `HPX_CB_STREAM_STEP2: PASS` on all
four runs in the close-out matrix (Metal+HPX build,
TinyLlama Q4_K_M, greedy, `"Hello, my name is"`):

```text
stream-off regression on Slice 7 shape : HPX_CB_STREAM_STEP2: PASS
Slice 2 stream smoke (trace off)       : HPX_CB_STREAM_STEP2: PASS
Slice 2 trace-on stream smoke          : HPX_CB_STREAM_STEP2: PASS
Slice 2 repeat (cross-invocation)      : HPX_CB_STREAM_STEP2: PASS
```

Observed hashes:

```text
budget 8   completed         hash=0x0619d4d1900c2365   (canonical anchor; strict-gated)
budget 256 completed         hash=0x8790fbe5a60c9ae6   (within-run uniqueness only;
                                                         carry-over value from Slice 1)
budget 64  cancelled-prefix  streamed_hash == rr.hash gated; concrete value not
                              surfaced in stdout (the partition row prints only
                              unique_completed_hashes; cancelled-prefix hash is not
                              separately emitted by the gate).
```

The budget-64 cancelled-prefix hash is intentionally not
compared to Slice 1's 64-token completed hash
(`0x3b15a0474dfe11be`): the Slice 1 hash is over 64
tokens, the Slice 2 cancelled-prefix is over 16 tokens.
Cross-shape long-budget hash equality is not a safe
correctness invariant under CLAUDE.md.

Closeout captures: `local/slice9_stream_step2_*.{stdout,stderr}`.
Detailed evidence is in
`tools/hpx-continuous-batch-gate/results.md` under the
*Streaming Slice 2 results — cancellation-aware streaming*
section.

## 9. Risks and deferred work

Out of scope for Streaming Slice 2:

- HTTP / gRPC / Unix-socket streaming. The streaming
  surface remains an in-process HPX local channel between
  the engine task and `main()`. No network framing, no
  chat-template assembly, no `tools/server` integration.
  See `docs/hpx/continuous_batching_gate_vs_serving_layer.md`
  for the gate-vs-serving-layer boundary.
- Backpressure / bounded-channel policy. `channel<T>` is
  still unbounded so the engine never suspends on the
  consumer. A bounded-channel slice is deferred.
- Engine-failure stream semantics. The defensive
  `engine::run()` drain still closes pending streams with
  `error` when the decode loop bailed early; no smoke
  exercises it. An error-path streaming slice is deferred.
- Tokenizer / prompt generalization. The smoke still uses
  TinyLlama, `"Hello, my name is"`, and greedy decoding.
  Per-request prompts and per-request sampling
  configuration are deferred.
- Streaming through admitted reuser slots in the smoke.
  The streaming wiring is admission-agnostic, but the
  smoke has no admission and the cancelled seq is an
  original active seq, not a freed/reused slot.
- Multiple concurrent engine tasks. The
  single-engine-task-per-process invariant still holds.
- Performance comparisons. Streaming Slice 2 is
  correctness-only.

Hard invariant: only the engine HPX task touches
`llama_context`, `llama_batch`, `llama_decode`,
`llama_memory_seq_*`, or `llama_get_logits_ith`. Stream
events carry only an `int32_t` token id and a close
reason; no llama.cpp state ever crosses the channel.

## 10. Safe claim language

Supported by Streaming Slice 2 evidence and safe to use
verbatim:

- "Cancellation-aware streaming is gated at the HPX
  continuous-batching gate boundary: a cancelled streamed
  request closes its HPX local channel with
  `reason=cancelled` after exactly `n_decoded_at_cancel`
  token events, the streamed hash equals `rr.hash`, and
  `rr.status == cancelled`."
- "Engine-side stream counters
  (`streams_closed_completed`, `streams_closed_cancelled`,
  `streams_closed_error`, `streams_opened`,
  `stream_tokens_emitted_total`) match the corresponding
  counts derived from the per-request
  `request_result` status on every gated run."
- "Streaming Slice 2 introduces no new HPX primitive, no
  new CLI flag, no engine-side behavior change, and no
  CMake change. The advance is gate-side: status-aware
  close-reason and token-count checks, plus
  expected-from-`request_result` engine-counter gates."
- "All Slice 1–7 admission/cancellation/lifecycle
  invariants and all Streaming Slice 1 streaming
  invariants remain strict carry-overs under
  `HPX_CB_STREAM_STEP2`."

Not supported by Slice 2 and to be avoided:

- "Streaming Slice 2 integrates with `llama-server` or
  exposes an HTTP/gRPC streaming endpoint." (No network
  adapter; the stream is in-process only.)
- "Streaming Slice 2 demonstrates a performance benefit
  from HPX." (No comparative benchmark is run; this is a
  correctness/lifecycle gate.)
- "The cancelled budget-64 prefix hashes to
  `0x...`." (The concrete cancelled-prefix hash value is
  not surfaced in the gate's stdout — only the equality
  `streamed_hash == rr.hash` is gated. Do not assert a
  concrete value without a separate emit.)
- "Cancellation-aware streaming includes backpressure."
  (The channel is unbounded by design; bounded
  channels are a deferred slice.)
