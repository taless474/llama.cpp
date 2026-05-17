# Streaming Slice 7 design — multi-cycle slot reuse with streaming

## 1. Purpose

Streaming Slice 7 closes the next correctness gap after the first-admission streaming surface. Slices 1–6 gated the width of streaming coverage: original active completion, original active cancellation, preloaded admitted requests over `completion_freed` and `cancel_freed`, and external admitted requests over `completion_freed` and `cancel_freed`.

Slice 7 gates the depth of the surface: a single `seq_id` slot hosting more than one admitted streamed request in the same engine run.

The tested chain is:

```text
request 0 -> request 1 -> request 2
```

All three occupants use the same slot and complete normally.

This remains a correctness/lifecycle gate. No performance claim is made.

## 2. Relation to Streaming Slices 1–6

Slice 1 introduced HPX local-channel token streaming for original active completed requests.

Slice 2 made original-active streaming cancellation-aware.

Slice 3 proved fresh stream rebinding for a preloaded waiting request admitted through `completion_freed`.

Slice 4 extended admitted streaming to the preloaded `cancel_freed` path.

Slice 5 extended admitted streaming to external arrival over `completion_freed`.

Slice 6 extended admitted streaming to external arrival over `cancel_freed`.

Slice 7 keeps the admission source simple (`completion_freed`, preloaded), but asks a different question: can the same slot be reused more than once while preserving independent stream lifecycles?

## 3. Target invariant

A single `seq_id` slot can host a three-occupant completion-freed chain in one engine run:

```text
request 0 -> request 1 -> request 2
```

For each occupant:

- a stream lifecycle exists;
- close reason is `completed`;
- streamed token count equals `rr.n_decoded`;
- `streamed_hash == rr.hash`;
- KV is empty before the next occupant binds;
- residual KV is empty at engine end.

For admitted occupants:

- `reused_seq_id` points to the reused slot;
- `previous_request_id` points to the immediately previous occupant;
- `admitted_at_iter == previous_occupant.done_iter + 1`.

## 4. Scope

In scope:

- completion-freed multi-cycle reuse;
- preloaded waiters only;
- a three-occupant chain;
- Path alpha structural independence;
- existing HPX local-channel stream substrate;
- existing request-result and stream-counter gates;
- existing trace event names.

Out of scope:

- cancel-freed multi-cycle reuse;
- external-arrival multi-cycle reuse;
- cancellation of second-or-later admitted occupants;
- engine-error `reason=error` stream semantics;
- HTTP / gRPC / Unix-socket / WebSocket streaming;
- backpressure / bounded channels;
- performance comparison.

## 5. Smoke shape

The conceptual shape is one active request and two waiting requests. The CLI requires:

```text
n_active + n_waiting <= n_seqs
```

So the smoke uses `--n-seqs 3`, even though only slot 0 is live at any moment.

```text
--n-seqs 3
--n-active 1
--n-waiting 2
--waiting-budget 8
--decode-budget-mix 8
--reuse-completed
--cancel-plan none
--stream-all
--repeat 2
```

## 6. Expected per-request behavior

### Request 0, original active

```text
request_id          = 0
seq_id              = 0
admission_src       = none
arrival_src         = preloaded
budget              = 8
status              = completed
done_iter           = 7
close_reason        = completed
streamed_count      = 8
hash                = 0x0619d4d1900c2365
```

### Request 1, first admitted occupant

```text
request_id          = 1
seq_id              = 0
admission_src       = completion_freed
arrival_src         = preloaded
reused_seq_id       = 0
previous_request_id = 0
admitted_at_iter    = 8
budget              = 8
status              = completed
done_iter           = 15
close_reason        = completed
streamed_count      = 8
hash                = 0x0619d4d1900c2365
```

### Request 2, second admitted occupant

```text
request_id          = 2
seq_id              = 0
admission_src       = completion_freed
arrival_src         = preloaded
reused_seq_id       = 0
previous_request_id = 1
admitted_at_iter    = 16
budget              = 8
status              = completed
done_iter           = 23
close_reason        = completed
streamed_count      = 8
hash                = 0x0619d4d1900c2365
```

## 7. Required gates

Carry-over gates from Slices 1–6:

- stream closes with the expected reason;
- streamed token count equals `rr.n_decoded`;
- `streamed_hash == rr.hash`;
- stream counters match streamed `request_result` rows;
- trace-off quietness;
- residual KV empty;
- repeat determinism.

New Slice 7 gate:

When the Slice 7 multi-cycle predicate holds:

```text
stream_all
reuse_completed
cancel_plan empty
n_waiting > n_active
n_external_arrivals == 0
```

require:

- at least one slot has two or more admitted occupants;
- each admitted occupant is status `completed`;
- each admitted occupant has a stream;
- each admitted stream closes with `completed`;
- each admitted streamed token count equals `rr.n_decoded`;
- each admitted `streamed_hash == rr.hash`;
- `previous_request_id` links to the immediately previous occupant;
- `admitted_at_iter == previous.done_iter + 1`.

## 8. Path alpha: structural independence

Slice 7 intentionally does not use token-vector inequality for the multi-cycle chain.

All three occupants use the same prompt, model, greedy policy, and budget. With clean KV and the same prompt, the correct model behavior is to produce the same 8-token sequence three times. Equal token vectors are expected.

Independence is proved structurally:

1. fresh channel per admission;
2. per-slot stream counter reset;
3. KV-empty assertion before each bind;
4. request-id chain walk;
5. per-cycle `streamed_hash == rr.hash`;
6. independent stream open/close counters.

This is stronger and more accurate than requiring artificial token-vector differences.

## 9. Stream counter expectations

Per repeat:

```text
streams_opened              = 3
streams_closed_completed    = 3
streams_closed_cancelled    = 0
streams_closed_error        = 0
stream_tokens_emitted_total = 24
admitted_count              = 2
external_admitted_count     = 0
arrival_drained_count       = 0
first_external_drain_iter   = -1
iter_release_fired_set      = {}
submitter_ack_set           = {}
reused_seq_id_set           = {0,0}
```

The duplicate `reused_seq_id` is expected in this slice. It is the multi-cycle evidence.

## 10. Trace evidence

Across `--repeat 2`:

```text
token_stream_opened      = 6
token_stream_token       = 48
token_stream_closed      = 6
reason=completed         = 6
reason=cancelled         = 0
arrival_drained          = 0
iter_release_fired       = 0
submitter_ack_observed   = 0
cancel_observed          = 0
cancel_kv_cleared        = 0
cancel_future_fulfilled  = 0
seq_complete             = 6
kv_cleared               = 6
promise_fulfilled        = 6
seq_reused               = 4
request_admitted_live    = 4
admitted_prefilled       = 4
```

The `request_admitted_live` events carry:

```text
admission_source=completion_freed
arrival_source=preloaded
```

Trace ordering is descriptive only. The strict correctness form is the chain walk plus per-request stream/hash/status gates.

## 11. Results

All five runs passed:

```text
local/slice14_stream_step7_regression_streamoff.stdout      HPX_CB_STREAM_STEP7: PASS
local/slice14_stream_step7_smoke.stdout                     HPX_CB_STREAM_STEP7: PASS
local/slice14_stream_step7_smoke_traceon.stdout             HPX_CB_STREAM_STEP7: PASS
local/slice14_stream_step7_smoke_repeat2.stdout             HPX_CB_STREAM_STEP7: PASS
local/slice14_stream_step7_slice6_regression.stdout         HPX_CB_STREAM_STEP7: PASS
```

The representative Slice 6 regression passing under the Slice 7 binary shows that Slice 7 validation relaxations are conditional and do not weaken the prior external + cancel-freed path.

## 12. HPX-native design note

Slice 7 remains HPX-native at the orchestration/streaming boundary:

- stream substrate remains `hpx::lcos::local::channel<token_stream_event>`;
- no new HPX primitive is added;
- no new mutex is added;
- no `std::thread`;
- no `std::condition_variable`;
- no `std::this_thread::sleep_for`;
- engine HPX task remains the sole stream producer;
- main remains the consumer in this gate;
- stream payload remains `int32_t token_id` plus close reason;
- no llama.cpp state crosses the channel;
- only the engine HPX task touches llama.cpp execution state.

The Slice 7 source change is validation-only. Engine-side admission, rebind, streaming, and KV-clear behavior are unchanged.

## 13. Risks and caveats

- Slice 7 covers completion-freed multi-cycle reuse only.
- Cancel-freed multi-cycle reuse is deferred.
- External-arrival multi-cycle reuse is deferred.
- The smoke uses uniform budget 8, so token-vector equality is expected.
- Path alpha proves structural stream independence, not token-vector difference.
- `--n-seqs 3` is required by the CLI parser even though only slot 0 is live.
- The same canonical budget-8 hash appears three times by design.
- No engine-error stream semantics are tested.
- No performance result is claimed.

## 14. Deferred work

After Slice 7, remaining streaming-gate surfaces include:

- cancel-freed multi-cycle reuse;
- external-arrival multi-cycle reuse;
- cancellation of admitted occupants in later cycles;
- engine-error `reason=error` stream close;
- serving-layer receiver routing;
- HTTP/server/network streaming;
- backpressure or bounded channels;
- performance measurement.

## 15. Safe claim language

Supported:

```text
Streaming Slice 7 gates completion-freed multi-cycle slot reuse with streaming. A single
seq_id slot hosts a three-occupant request chain in one engine run, and each occupant
gets a fresh HPX local-channel stream lifecycle, closes with reason=completed, preserves
streamed_hash == rr.hash, and links to the immediately previous occupant through
previous_request_id.
```

Not supported:

```text
Slice 7 does not cover cancel-freed multi-cycle reuse, external-arrival multi-cycle reuse,
engine-error stream semantics, HTTP/server streaming, backpressure, or performance.
Slice 7 also does not claim token-vector divergence across identical greedy runs.
```
