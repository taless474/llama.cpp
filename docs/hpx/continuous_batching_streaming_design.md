# Continuous batching — consolidated streaming design (Streaming Slices 1–7)

## 1. Purpose

This document consolidates the design notes for Streaming Slices 1–7 of the HPX continuous-batching gate. These slices add and validate an HPX-native, per-request token streaming surface inside the gate.

The streaming sequence starts after the closed admission gate sequence:

```text
HPX_CB_ADMIT_STEP7: PASS
```

and advances through:

```text
HPX_CB_STREAM_STEP1: PASS
HPX_CB_STREAM_STEP2: PASS
HPX_CB_STREAM_STEP3: PASS
HPX_CB_STREAM_STEP4: PASS
HPX_CB_STREAM_STEP5: PASS
HPX_CB_STREAM_STEP6: PASS
HPX_CB_STREAM_STEP7: PASS
```

The whole sequence is a correctness/lifecycle gate, not a serving layer and not a performance benchmark. The consumer is still the gate's `main()` path, not an HTTP, gRPC, Unix-socket, WebSocket, or `llama-server` integration.

The streaming gates prove that generated token ids can be published through an HPX-native channel, reconstructed by the consumer, and checked against the existing `request_result` snapshot without moving any llama.cpp execution state across the stream boundary.

## 2. Common streaming substrate

Streaming uses `hpx::lcos::local::channel<token_stream_event>`.

```cpp
enum class stream_event_kind : uint8_t {
    token  = 0,
    closed = 1,
};

enum class stream_close_reason : uint8_t {
    completed = 0,
    cancelled = 1,
    error     = 2,
};

struct token_stream_event {
    stream_event_kind    kind;          // token or closed
    int32_t              token_id;      // valid iff kind == token
    stream_close_reason  close_reason;  // valid iff kind == closed
};

using token_stream_channel =
    hpx::lcos::local::channel<token_stream_event>;
using token_stream_sender =
    hpx::lcos::local::send_channel<token_stream_event>;
using token_stream_receiver =
    hpx::lcos::local::receive_channel<token_stream_event>;
```

The channel replaced an earlier manual future-chain prototype. The future-chain passed the initial runs, but `hpx::lcos::local::channel` is the better fit because it is an HPX LCO, provides split producer/consumer handles, owns its lifetime through intrusive pointers, and has explicit close semantics.

No new CMake target or link dependency was needed for the channel path.

## 3. Shared ownership and HPX-native boundary

The engine task is the sole producer. The gate consumer is the sole receiver.

| Side | Owns | Rules |
|---|---|---|
| Engine task | `seq_state.stream_channel`; llama.cpp mutable execution state | Publishes token events, sends the terminal close event, closes the channel, clears KV, and fulfills the request promise. |
| Gate consumer (`main`) | Matching `token_stream_receiver` handles | Drains events with `rx.get(hpx::launch::sync)` and reconstructs streamed token vectors. It never touches llama.cpp execution state. |

Hard boundary:

```text
Only the engine HPX task touches:
  llama_context
  llama_batch
  llama_decode
  llama_get_logits_ith
  llama_memory_seq_*

Stream events carry only:
  int32_t token_id
  stream_close_reason
```

Streaming adds no `std::thread`, no `std::condition_variable`, no wall-clock sleep, and no new `std::mutex`. Synchronization for the stream itself is the HPX local channel implementation.

## 4. Shared stream invariants

Each streamed request has exactly one terminal close event.

```text
token events... → token_stream_event{kind=closed, close_reason=...} → channel.close()
```

Required ordering:

- The terminal `kind=closed` event is sent before `channel.close()`.
- Calling `channel.close()` before sending the terminal event would make the final `set()` fail.
- Consumers stop on `kind == closed`.
- A consumer that calls `get()` after the terminal event observes `hpx::error::invalid_status`, which is an error, not a hang.

Per streamed request:

```text
streamed_tokens.size() == rr.n_decoded
streamed_hash == rr.hash
close_count == 1
close_reason follows rr.status:
  completed → completed
  cancelled → cancelled
```

For cancelled requests:

```text
streamed_tokens.size() == rr.n_decoded_at_cancel
rr.status == cancelled
streamed_hash == rr.hash
```

For admitted/reused requests:

```text
the admitted request uses a fresh channel
stream_closed is reset
stream_tokens_emitted is reset for the slot
the admitted stream is keyed by request_id
the previous occupant's close reason and token vector are not inherited
```

Engine-side stream counters are derived from streamed `request_result` rows:

```text
streams_closed_error == 0       // for these smokes

streams_opened ==
    streams_closed_completed
  + streams_closed_cancelled
  + streams_closed_error

stream_tokens_emitted_total ==
    sum(rr.n_decoded over streamed requests)
```

With `--stream-all` OFF:

```text
streams_opened              == 0
streams_closed_completed    == 0
streams_closed_cancelled    == 0
streams_closed_error        == 0
stream_tokens_emitted_total == 0
stream receiver collections are empty
```

Carry-over gates across the sequence:

- residual KV is empty at engine end;
- `--repeat 2` is deterministic on result snapshots, streamed token vectors, and close reasons;
- exactly one engine task owns llama.cpp mutable execution state per repeat;
- every `llama_decode` returns 0;
- trace-off runs emit zero `[hpx-cb-gate] event=` lines;
- the engine inbox and external promise maps are empty at run end when external arrivals are exercised.

## 5. Trace events

The streaming trace event set is stable across Streaming Slices 1–7:

```text
[hpx-cb-gate] event=token_stream_opened request=<id> seq_id=<id>
[hpx-cb-gate] event=token_stream_token  request=<id> seq_id=<id> pos=<int> token=<int>
[hpx-cb-gate] event=token_stream_closed request=<id> seq_id=<id> n_tokens=<int> reason=completed|cancelled|error
```

All are gated by `LLAMA_HPX_CB_TRACE=1`. The trace-off path remains quiet.

`token_stream_token` is high-cardinality: one event per emitted streamed token. The strict correctness form is the result/stream/hash/counter gates. Trace ordering snippets in the slice notes are descriptive evidence, not a separate strict ordering contract unless explicitly stated.

## 6. Slice evolution summary

| Slice | PASS label | New coverage | Main source delta | Key result |
|---|---|---|---|---|
| Streaming Slice 1 | `HPX_CB_STREAM_STEP1: PASS` | Original active completion streaming | Introduces HPX local-channel stream path behind `--stream-all` | 3 completed streams, budgets 8/64/256, 328 tokens |
| Streaming Slice 2 | `HPX_CB_STREAM_STEP2: PASS` | Original active cancellation streaming | Gate becomes status-aware; engine cancel close path already wired | 2 completed + 1 cancelled stream, 280 tokens |
| Streaming Slice 3 | `HPX_CB_STREAM_STEP3: PASS` | Preloaded waiter admitted via `completion_freed` | Rebind fresh stream in `admit_one` for completion-freed admission | Original 8-token stream + admitted 16-token stream |
| Streaming Slice 4 | `HPX_CB_STREAM_STEP4: PASS` | Preloaded waiter admitted via `cancel_freed` | Broaden preloaded rebind predicate to `completion_freed || cancel_freed` | Cancelled original + completed admitted waiter |
| Streaming Slice 5 | `HPX_CB_STREAM_STEP5: PASS` | External arrival admitted via `completion_freed` | Add external-branch rebind for `completion_freed` | External request gets fresh completed stream |
| Streaming Slice 6 | `HPX_CB_STREAM_STEP6: PASS` | External arrival admitted via `cancel_freed` | Broaden external-branch rebind to `completion_freed || cancel_freed` | Every admission × arrival-source combination is covered |
| Streaming Slice 7 | `HPX_CB_STREAM_STEP7: PASS` | Multi-cycle completion-freed reuse | Validation-only; engine behavior unchanged | One slot hosts request 0 → 1 → 2 with independent stream lifecycles |

## 7. Streaming Slice 1 — original active completion

### Purpose

Slice 1 opens the streaming gate sequence. It proves that active requests can emit generated token ids into HPX local channels and that the consumer can reconstruct the same token sequence recorded in `request_result`.

The path is opt-in via:

```text
--stream-all
```

With `--stream-all` OFF, the streaming path is fail-closed and all stream counters remain zero.

### Engine flow

At construction, when streaming is enabled:

```text
mark bound active seq stream_enabled
construct token_stream_channel
derive token_stream_receiver
make the receiver available to the consumer
```

On each generated token:

```cpp
publish_token(seq, static_cast<int32_t>(next_id));
seq.generated_tokens.push_back(next_id);
seq.hash_state = fold_token_hash(seq.hash_state,
                                 static_cast<int32_t>(next_id));
seq.n_decoded++;
seq.last_token = next_id;
```

On completion:

```cpp
clear_and_check(seq, iter, mem);
close_stream(seq, completed);
fulfill_promise(seq, request_status::completed, ttc_us);
```

Cancellation and error close paths are present defensively but are not exercised by the Slice 1 smoke.

### Smoke shape

```text
--stream-all
--n-seqs 3
--decode-budget-mix 8,64,256
--cancel-plan none
--n-waiting 0
--n-external-arrivals 0
```

Implicit defaults:

```text
--prompt "Hello, my name is"
--hpx-os-threads 1
--reuse-completed OFF
```

Expected streams:

| seq_id | budget | status | streamed tokens | close reason |
|---|---:|---|---:|---|
| 0 | 8 | completed | 8 | completed |
| 1 | 64 | completed | 64 | completed |
| 2 | 256 | completed | 256 | completed |

Expected total:

```text
stream_tokens_emitted_total = 328
```

Trace-on counts:

```text
token_stream_opened = 3
token_stream_token  = 328
token_stream_closed = 3
```

Known hash anchors for this shape:

```text
budget 8   hash=0x0619d4d1900c2365
budget 64  hash=0x3b15a0474dfe11be
budget 256 hash=0x8790fbe5a60c9ae6
```

## 8. Streaming Slice 2 — original active cancellation

### Purpose

Slice 2 closes the cancellation arm for original active streamed requests. It proves that a cancelled streamed request closes with `reason=cancelled`, emits exactly `n_decoded_at_cancel` token events, satisfies `streamed_hash == rr.hash`, and leaves KV empty.

No new HPX primitive, channel type, trace event, CLI flag, CMake change, or engine-side stream primitive is introduced. The change is primarily gate-side: close-reason checks become status-aware and stream counters are derived from `request_result` status.

### Cancellation policy

Cancellation is observed at an iteration boundary before the active row set is built. After observation, the sequence is marked done and cannot publish another token.

```text
cancel_and_fulfill:
  record cancel_observed_iter and n_decoded_at_cancel
  clear KV and check cross-talk
  close stream with reason=cancelled
  fulfill promise with status=cancelled
```

### Smoke shape

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

Expected behavior:

```text
seq 0: budget=8    status=completed  streamed=8    close=completed
seq 1: budget=64   status=cancelled  streamed=16   close=cancelled
seq 2: budget=256  status=completed  streamed=256  close=completed
```

Expected per-repeat counters:

```text
streams_opened              = 3
streams_closed_completed    = 2
streams_closed_cancelled    = 1
streams_closed_error        = 0
stream_tokens_emitted_total = 280
```

Trace-on per-repeat cancellation evidence:

```text
cancel_requested        = 1
cancel_observed         = 1
cancel_kv_cleared       = 1
cancel_future_fulfilled = 1
```

The concrete cancelled-prefix hash is not surfaced as a standalone stdout anchor. The gated invariant is `streamed_hash == rr.hash`, not equality with the 64-token completed hash from Slice 1.

## 9. Streaming Slice 3 — preloaded completion-freed admitted request

### Purpose

Slice 3 proves admitted-request streaming on the `completion_freed` path for preloaded waiters. A waiting request admitted into a slot freed by natural completion must get a fresh stream, emit its own token vector, and close independently of the previous occupant.

### Source delta

Inside `admit_one`, the completion-freed rebind path:

```text
constructs a fresh token_stream_channel
resets stream_closed
resets stream_tokens_emitted for the slot
pushes {request_id, receiver} to the admitted stream handoff path
```

No new HPX primitive, CLI flag, trace event, channel type, or CMake change is introduced.

### Smoke shape

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

Expected behavior:

```text
request 0:
  seq_id=0
  admission_source=none
  arrival_source=preloaded
  budget=8
  status=completed
  streamed=8
  hash=0x0619d4d1900c2365

request 1:
  reused_seq_id=0
  previous_request_id=0
  admitted_at_iter=8
  admission_source=completion_freed
  arrival_source=preloaded
  budget=16
  status=completed
  streamed=16
  hash=0x833045f1e2ebf49f
```

Expected per-repeat counters:

```text
streams_opened              = 2
streams_closed_completed    = 2
streams_closed_cancelled    = 0
streams_closed_error        = 0
stream_tokens_emitted_total = 24
```

Trace-on counts across `--repeat 2`:

```text
token_stream_opened = 4
token_stream_token  = 48
token_stream_closed = 4
seq_reused          = 2
request_admitted_live = 2
admitted_prefilled  = 2
```

### Caveats

A pre-existing round-robin validation bug was surfaced and fixed here: original-active budget validation now iterates over `args.decode_budget_mix`, not `uniq_budgets`.

The admitted budget-16 hash is shape-scoped. It must not be promoted to a cross-shape canonical anchor.

At this point, cancel-freed and external-arrival admitted streaming were still deferred.

## 10. Streaming Slice 4 — preloaded cancel-freed admitted request

### Purpose

Slice 4 extends the admitted preloaded waiter path from `completion_freed` to `cancel_freed`. A waiter admitted into a slot freed by cooperative cancellation must receive a fresh stream and must not inherit the cancelled occupant's close reason or stream state.

### Source delta

The preloaded-branch rebind predicate in `admit_one` broadens from:

```text
src == admission_source::completion_freed
```

to:

```text
src == admission_source::completion_freed
|| src == admission_source::cancel_freed
```

No new HPX primitive, CLI flag, trace event, channel type, mutex, or CMake change is introduced.

### Smoke shape

A one-active shape cannot trigger cancel-freed admission because cancellation is observed at iter K and the freed slot is eligible at iter K+1. If the cancelled sequence is the only active one, the engine exits before K+1. Slice 4 therefore uses one cancelled active sequence and one surviving active sequence.

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

Expected behavior:

```text
request 0:
  seq_id=0
  budget=64
  status=cancelled
  cancel_observed_iter=8
  n_decoded_at_cancel=8
  close=cancelled
  streamed=8

request 1:
  seq_id=1
  budget=256
  status=completed
  close=completed
  streamed=256
  hash=0x8790fbe5a60c9ae6

request 2:
  reused_seq_id=0
  previous_request_id=0
  admitted_at_iter=9
  admission_source=cancel_freed
  arrival_source=preloaded
  budget=16
  status=completed
  close=completed
  streamed=16
  hash=0x833045f1e2ebf49f
```

Expected per-repeat counters:

```text
streams_opened              = 3
streams_closed_completed    = 2
streams_closed_cancelled    = 1
streams_closed_error        = 0
stream_tokens_emitted_total = 280
```

Trace-on counts across `--repeat 2`:

```text
token_stream_opened     = 6
token_stream_token      = 560
token_stream_closed     = 6
reason=completed        = 4
reason=cancelled        = 2
cancel_observed         = 2
cancel_kv_cleared       = 2
cancel_future_fulfilled = 2
request_admitted_live   = 2
seq_reused              = 2
admitted_prefilled      = 2
```

### Caveats

The independence evidence in this smoke is structural and length-based. The cancelled prefix has 8 tokens and the admitted stream has 16. This proves fresh channel binding, counter reset, and non-inheritance of close reason/state. It does not prove semantic token divergence. Under greedy decoding, matching prefixes can be expected when prompt and policy are identical.

External-arrival admitted streaming remained deferred after Slice 4.

## 11. Streaming Slice 5 — external completion-freed admitted request

### Purpose

Slice 5 extends admitted-request streaming to external arrivals submitted through the scripted HPX submitter task with `engine::submit()`, admitted via a `completion_freed` slot.

After Slice 5, all admission paths were streamed except external × cancel-freed.

### Source delta

The external branch of `admit_one` gains a stream rebind block gated on:

```text
stream_all_ && src == admission_source::completion_freed
```

The submitter-held `hpx::future<request_result>` remains the result route for external arrivals. Main does not push an external-arrival future into `admitted_futures_`; only the stream receiver handoff is added.

No new HPX primitive, CLI flag, trace event, channel type, mutex, or CMake change is introduced.

### Smoke shape

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

Expected behavior:

```text
request 0:
  seq_id=0
  budget=8
  status=completed
  done_iter=7
  streamed=8
  hash=0x0619d4d1900c2365

request 1:
  seq_id=1
  budget=256
  status=completed
  streamed=256
  hash=0x8790fbe5a60c9ae6

request 2:
  arrival_source=external
  admission_source=completion_freed
  reused_seq_id=0
  previous_request_id=0
  admitted_at_iter=8
  budget=16
  status=completed
  streamed=16
  hash=0x833045f1e2ebf49f
```

External-arrival metrics per repeat:

```text
arrival_drained_count     = 1
first_external_drain_iter = 4
iter_release_fired_set    = {3}
submitter_ack_set         = {3}
external_admitted_count   = 1
```

Expected stream counters per repeat:

```text
streams_opened              = 3
streams_closed_completed    = 3
streams_closed_cancelled    = 0
streams_closed_error        = 0
stream_tokens_emitted_total = 280
```

Trace-on counts across `--repeat 2`:

```text
token_stream_opened      = 6
token_stream_token       = 560
token_stream_closed      = 6
reason=completed         = 6
arrival_drained          = 2
request_queued           = 2
iter_release_fired       = 2
submitter_ack_observed   = 2
request_admitted_live    = 2
seq_reused               = 2
admitted_prefilled       = 2
```

### Caveats

Slice 5 covers only external × completion_freed. External × cancel_freed remained deferred.

The source trace event is named `iter_release_fired`, not `release_fired`.

The Slice 5 independence evidence is length-based for the same reason as Slice 4: the prior occupant's stream has 8 tokens and the admitted external request has 16.

The budget-16 hash remains shape-scoped and should not be treated as a cross-shape canonical anchor.

## 12. Streaming Slice 6 — external cancel-freed admitted request

### Purpose

Slice 6 extends external-arrival streaming to the `cancel_freed` admission path. With Slice 6 closed, every admission × arrival-source combination exercised by the gate is streamed end-to-end:

```text
preloaded × completion_freed
preloaded × cancel_freed
external  × completion_freed
external  × cancel_freed
```

### Source delta

The external-branch rebind predicate broadens from:

```text
stream_all_ && src == admission_source::completion_freed
```

to:

```text
stream_all_
&& (src == admission_source::completion_freed
 || src == admission_source::cancel_freed)
```

This mirrors the Slice 4 broadening on the preloaded branch. No new HPX primitive, CLI flag, trace event, channel type, mutex, or CMake change is introduced.

### Smoke shape

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

Expected behavior:

```text
request 0:
  seq_id=0
  budget=64
  status=cancelled
  cancel_observed_iter=8
  n_decoded_at_cancel=8
  close=cancelled
  streamed=8

request 1:
  seq_id=1
  budget=256
  status=completed
  close=completed
  streamed=256
  hash=0x8790fbe5a60c9ae6

request 2:
  arrival_source=external
  admission_source=cancel_freed
  reused_seq_id=0
  previous_request_id=0
  admitted_at_iter=9
  budget=16
  status=completed
  close=completed
  streamed=16
  hash=0x833045f1e2ebf49f
```

External-arrival metrics per repeat:

```text
arrival_drained_count     = 1
first_external_drain_iter = 4
iter_release_fired_set    = {3}
submitter_ack_set         = {3}
external_admitted_count   = 1
```

Cancellation/admission metrics per repeat:

```text
cancel_observed         = 1
cancel_kv_cleared       = 1
cancel_future_fulfilled = 1
seq_reused              = 1
admitted_prefilled      = 1
request_admitted_live   = 1
```

Expected stream counters per repeat:

```text
streams_opened              = 3
streams_closed_completed    = 2
streams_closed_cancelled    = 1
streams_closed_error        = 0
stream_tokens_emitted_total = 280
```

Trace-on counts across `--repeat 2`:

```text
token_stream_opened      = 6
token_stream_token       = 560
token_stream_closed      = 6
reason=completed         = 4
reason=cancelled         = 2
arrival_drained          = 2
request_queued           = 2
iter_release_fired       = 2
submitter_ack_observed   = 2
cancel_observed          = 2
cancel_kv_cleared        = 2
cancel_future_fulfilled  = 2
request_admitted_live    = 2
seq_reused               = 2
admitted_prefilled       = 2
```

The optional Slice 5 regression was also run under the Slice 6 binary to confirm the broadened predicate did not regress external × completion_freed.

### Caveats

The independence evidence is still structural and length-based, not semantic token divergence.

The descriptive trace ordering is not the strict correctness form. The gated form is the per-event count totals plus result/stream/hash/status checks.

The Slice 6 trace event remains `iter_release_fired`, not `release_fired`.

## 13. Streaming Slice 7 — multi-cycle completion-freed slot reuse

### Purpose

Slices 1–6 close the width of the streaming surface across original active requests and first admitted requests. Slice 7 checks depth: one `seq_id` slot hosts more than one admitted streamed request in a single engine run.

The tested chain is:

```text
request 0 -> request 1 -> request 2
```

All three occupants use the same slot and complete normally.

The Slice 7 source change is validation-only. Engine-side admission, rebind, streaming, and KV-clear behavior are unchanged.

### Scope

In scope:

```text
completion-freed multi-cycle reuse
preloaded waiters
three occupants on one slot
structural stream independence
```

Out of scope:

```text
cancel-freed multi-cycle reuse
external-arrival multi-cycle reuse
cancellation of second-or-later admitted occupants
engine-error reason=error
serving-layer receiver routing
HTTP/network streaming
backpressure
performance
```

### Smoke shape

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

The CLI requires `n_active + n_waiting <= n_seqs`, so the smoke uses `--n-seqs 3` even though only slot 0 is live at a time.

### Expected request chain

```text
request 0:
  seq_id=0
  admission_src=none
  arrival_src=preloaded
  budget=8
  status=completed
  done_iter=7
  streamed=8
  hash=0x0619d4d1900c2365

request 1:
  seq_id=0
  admission_src=completion_freed
  arrival_src=preloaded
  reused_seq_id=0
  previous_request_id=0
  admitted_at_iter=8
  budget=8
  status=completed
  done_iter=15
  streamed=8
  hash=0x0619d4d1900c2365

request 2:
  seq_id=0
  admission_src=completion_freed
  arrival_src=preloaded
  reused_seq_id=0
  previous_request_id=1
  admitted_at_iter=16
  budget=8
  status=completed
  done_iter=23
  streamed=8
  hash=0x0619d4d1900c2365
```

### Slice 7-specific gate

When the multi-cycle predicate holds:

```text
stream_all
reuse_completed
cancel_plan empty
n_waiting > n_active
n_external_arrivals == 0
```

the gate requires:

```text
at least one slot has two or more admitted occupants
each admitted occupant completed
each admitted occupant has a stream
each admitted stream closes with completed
streamed token count equals rr.n_decoded
streamed_hash == rr.hash
previous_request_id links to the immediately previous occupant
admitted_at_iter == previous.done_iter + 1
```

Expected per-repeat counters:

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

The duplicate `reused_seq_id` is expected and is part of the multi-cycle evidence.

Trace-on counts across `--repeat 2`:

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

### Structural independence

Slice 7 intentionally does not use token-vector inequality. All occupants use the same prompt, model, greedy policy, and budget. With clean KV and the same prompt, the correct behavior is to produce the same 8-token sequence three times.

Independence is instead proved structurally:

```text
fresh channel per admission
per-slot stream counter reset
KV-empty check before each bind
request-id chain walk
per-cycle streamed_hash == rr.hash
independent stream open/close counters
```

This is more accurate than forcing artificial token-vector differences.

## 14. Results and evidence locations

Streaming Slice 1 results are in:

```text
tools/hpx-continuous-batch-gate/results.md
  Streaming Slice 1 results — HPX local-channel token stream
```

Streaming Slice 2 results are in:

```text
tools/hpx-continuous-batch-gate/results.md
  Streaming Slice 2 results — cancellation-aware streaming
```

Streaming Slice 3 results are in:

```text
tools/hpx-continuous-batch-gate/results.md
  Streaming Slice 3 results — admitted-request streaming on the completion-freed admission path
```

Streaming Slice 4 results are in:

```text
tools/hpx-continuous-batch-gate/results.md
  Streaming Slice 4 results — admitted-request streaming over cancel-freed slot
```

Streaming Slice 5 results are in:

```text
tools/hpx-continuous-batch-gate/results.md
  Streaming Slice 5 results — external-arrival streaming over completion-freed slot
```

Streaming Slice 6 results are in:

```text
tools/hpx-continuous-batch-gate/results.md
  Streaming Slice 6 results — external-arrival streaming over cancel-freed slot
```

Streaming Slice 7 closeout captures:

```text
local/slice14_stream_step7_regression_streamoff.stdout
local/slice14_stream_step7_smoke.stdout
local/slice14_stream_step7_smoke_traceon.stdout
local/slice14_stream_step7_smoke_repeat2.stdout
local/slice14_stream_step7_slice6_regression.stdout
```

All five Slice 7 runs passed:

```text
HPX_CB_STREAM_STEP7: PASS
```

The Slice 6 regression under the Slice 7 binary shows that the Slice 7 validation relaxations are conditional and do not weaken the earlier external × cancel_freed path.

## 15. Shared caveats

### This is still a gate, not a serving layer

The stream consumer is the gate's main path. There is no network framing, no HTTP/gRPC/Unix-socket/WebSocket adapter, no `tools/server` integration, and no `llama-server` claim.

### No backpressure

The HPX local channel is unbounded in these gates. The engine does not suspend on the consumer. Bounded-channel policy and backpressure are deferred.

### No performance claim

All seven streaming slices are correctness/lifecycle gates. No comparative benchmark is run.

### Error close path is wired but not gated

`stream_close_reason::error` exists and the engine has a defensive drain path for unfinished streams on bailout, but none of these smokes gates engine-error stream semantics.

### Hashes are shape-scoped unless explicitly canonical

The budget-8 hash `0x0619d4d1900c2365` is used as a canonical anchor in these shapes. Other observed hashes, especially the admitted budget-16 hash `0x833045f1e2ebf49f`, are shape-scoped and must not be promoted to cross-shape correctness anchors.

### Structural independence is not semantic token divergence

Several admitted-stream independence checks are length-based or structural. Greedy decoding with the same prompt, model, policy, and budget can produce identical prefixes across different admission paths. That is expected. The gate proves stream lifecycle independence: fresh channel, reset counters, independent close reason, correct request-id keying, and clean KV between occupants.

## 16. Safe claim language

Supported:

```text
Streaming Slices 1–7 add and gate an HPX-native per-request token streaming
surface inside the continuous-batching gate. Streams use
hpx::lcos::local::channel<token_stream_event>; the engine task is the sole
producer, and the gate consumer drains receive_channel handles. Stream events
carry only int32_t token ids and close reasons; no llama.cpp execution state
crosses the channel boundary.
```

```text
The streaming gate covers original active completion, original active
cancellation, preloaded admitted requests over completion_freed and
cancel_freed, external admitted requests over completion_freed and
cancel_freed, and completion_freed multi-cycle slot reuse.
```

```text
For every streamed request, the gate checks streamed token count against
rr.n_decoded, streamed_hash against rr.hash, exactly one close event, and the
expected close reason. Stream counters are checked against request_result-derived
counts, and trace-off runs remain quiet.
```

```text
Streaming Slice 7 proves structural stream independence across a three-occupant
completion-freed chain on one seq_id slot. Equal token vectors are expected in
that smoke because the prompt, model, greedy policy, and budget are identical.
```

Not supported:

```text
The streaming slices do not implement HTTP, gRPC, Unix-socket, WebSocket, or
llama-server streaming.
```

```text
The streaming slices do not implement backpressure or bounded channels.
```

```text
The streaming slices do not claim a performance improvement.
```

```text
The streaming slices do not gate engine-error reason=error stream semantics.
```

```text
The streaming slices do not prove semantic token divergence between streams that
share the same prompt and greedy decoding policy.
```

```text
The admitted budget-16 hash 0x833045f1e2ebf49f is not a new cross-shape
canonical anchor.
```
