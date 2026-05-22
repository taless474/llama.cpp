# HPX continuous-batching lifecycle design: cancellation and live admission

This consolidated note merges the original cancellation and live-admission design notes for `tools/hpx-continuous-batch-gate/`.

It is a correctness-and-lifecycle design. It does not make performance claims.

The two features are intentionally connected:

```text
cooperative cancellation proves the leave path:
  active request -> cancel observed -> KV cleared -> future fulfilled

live admission proves the enter path:
  waiting request -> admitted into freed slot -> prefilled -> decoded -> future fulfilled
```

Together they establish that HPX can own request lifecycle around a real `llama.cpp` continuous-batching primitive while preserving the `llama.cpp` execution boundary.

---

## 1. Baseline architecture

Both designs build on the same HPX continuous-batching prototype:

```text
one llama_model
one llama_context
one shared llama_batch
one HPX engine task per repeat iteration
one hpx::promise<request_result> per active request
main validates only request_result snapshots
```

Hard boundary:

```text
Only the engine task may touch:
  llama_context
  llama_batch
  llama_decode
  llama_get_logits_ith
  llama_memory_seq_*
```

Future consumers receive snapshots only. They must not touch `llama.cpp` mutable execution state.

The engine must preserve:

```text
no parallel llama_decode on the same llama_context
KV clear before promise fulfillment
sibling cross-talk checks before fulfillment
residual KV empty at end
every promise fulfilled exactly once
repeat determinism for the same schedule
trace-off quietness
no speedup / latency / throughput claims
```

---

## 2. Shared lifecycle model

### Cancellation

Cancellation is cooperative and iteration-boundary only. The engine never interrupts an in-flight `llama_decode`.

```text
running
  -> cancel_requested
  -> cancel_observed at an iteration boundary
  -> KV cleared by the engine task
  -> future fulfilled with status=cancelled
```

A cancelled request:

```text
must not appear in any decode batch after cancel_observed_iter
must not have KV cleared before sibling cross-talk is checked
must not fulfill its promise before KV clear
must not fulfill twice
must complete its future with status=cancelled
may have a partial token sequence and partial hash
```

### Live admission

Live admission is also iteration-boundary only. Admission never happens after rows are added for an iteration and before `llama_decode` returns.

```text
waiting request
  -> admission boundary
  -> bind to a KV-empty seq_id slot
  -> prefill rows added to the shared llama_batch
  -> decode continues in later iterations
  -> future fulfilled from a request_result snapshot
```

A reused slot must satisfy:

```text
llama_memory_seq_pos_min(mem, seq_id) == -1
llama_memory_seq_pos_max(mem, seq_id) == -1
```

before a new request is bound to it.

---

## 3. Data model additions

### Request status

```cpp
enum class request_status : uint8_t {
    completed = 0,
    cancelled = 1,
    failed    = 2,   // reserved
};
```

Snapshot invariants:

```text
status == completed:
  n_decoded == decode_budget
  n_decoded_at_cancel == -1
  cancel_observed_iter == -1
  kv_cleared == true

status == cancelled:
  0 <= n_decoded_at_cancel < decode_budget
  n_decoded == n_decoded_at_cancel
  cancel_observed_iter >= 0
  kv_cleared == true
```

### Cancellation fields

Engine-internal `seq_state` gains:

```text
cancel_requested
cancel_after_decoded_tokens
cancel_observed
cancel_observed_iter
n_decoded_at_cancel
```

`request_result` gains:

```text
status
n_decoded_at_cancel
cancel_observed_iter
```

`cancel_requested` is atomic only because an external control path may set it while the engine reads it. The engine observes it only at iteration boundaries.

### Admission fields

`seq_state` / `request_result` gain:

```text
request_id
admitted_at_iter
previous_request_id
admission_source
reused_seq_id        // result-side field
```

Admission sources:

```text
none
cancel_freed
completion_freed
```

Arrival sources introduced later:

```text
preloaded
external
```

Defaults for non-admitted original requests:

```text
admitted_at_iter == -1
reused_seq_id == -1
previous_request_id == -1
admission_source == none
```

---

## 4. Engine-loop semantics

### Cancellation observation

Cancellation is checked:

```text
after prefill
at the top of each decode iteration, before building active rows
```

It is not checked inside the post-decode argmax loop. That avoids a same-iteration collision between completion and cancellation.

Pseudocode:

```text
after prefill:
  for each active seq:
    if cancel_should_observe(seq):
      cancel_and_fulfill(seq, prefill_iter)

for each decode iter:
  for each active seq:
    if cancel_should_observe(seq):
      cancel_and_fulfill(seq, iter)

  build rows only for surviving active seqs
  llama_decode(...)
  argmax / hash / n_decoded++
  if complete:
    finalize_and_fulfill(seq, iter, status=completed)
```

`cancel_should_observe(seq)` is true when:

```text
!seq.cancel_observed
and (
  seq.cancel_requested == true
  or
  (seq.cancel_after_decoded_tokens >= 0
   and seq.n_decoded >= seq.cancel_after_decoded_tokens)
)
```

`cancel_and_fulfill(seq, iter)`:

```text
records cancel_observed_iter and n_decoded_at_cancel
clears KV through the same clear_and_check path as completion
builds request_result(status=cancelled)
sets the promise exactly once
updates cancellation counters
```

### Admission boundary

Live admission runs at the top of a decode iteration:

```text
1. observe cancellation
2. collect slots whose KV clear succeeded
3. admit waiting requests into eligible freed seq_ids
4. add admitted prefill rows to the same batch as surviving decode rows
5. call llama_decode once for that mixed batch
```

The first admitted token comes from the prefill argmax in the admission iteration. Later tokens come from normal decode iterations.

---

## 5. Cancellation smoke

Canonical cancellation shape:

```text
n_seqs       = 99
budget mix   = {8, 64, 256}
prompt       = "Hello, my name is"
ctx_size     = 32768  (actual n_ctx 50688)
n_batch      = 1024
n_threads    = 2
```

Round-robin budget layout:

```text
seq_id % 3 == 0 -> budget 8
seq_id % 3 == 1 -> budget 64
seq_id % 3 == 2 -> budget 256
```

Cancel plan:

```text
cancel seq_ids {1, 4, 7}  from budget 64
cancel seq_ids {2, 5, 8}  from budget 256
cancel_after_decoded_tokens = 16
budget 8 is not cancelled
```

Expected outcome:

```text
completed_count = 93
cancelled_count = 6

budget 8:
  completed = 33
  cancelled = 0

budget 64:
  completed = 30
  cancelled = 3

budget 256:
  completed = 30
  cancelled = 3
```

Iteration math:

```text
first token comes from prefill
tokens 2..16 come from decode iters 1..15
top of iter 16 observes n_decoded == 16
cancel_observed_iter == 16
n_decoded_at_cancel == 16
```

Required anchors:

```text
cancel_observed_iter_set = {16}
n_decoded_at_cancel_set = {16}
wasted_decode_rows_after_cancel = 0
residual_kv_empty = true
decode_failures = 0
```

Cross-shape caveat:

```text
budget-64 and budget-256 hashes in the cancellation run may differ
from the full Slice-5 run because the batch shape changes after
the cancelled seqs leave. Do not compare those hashes across shapes.
Within-run uniqueness and repeat determinism are the gates.
```

---

## 6. Cancellation validation gates

The cancellation smoke must prove:

```text
engine task completes
all 99 futures complete
engine_task_count == 1
exactly 6 cancelled results
exactly 93 completed results
cancelled results have status=cancelled
completed results have status=completed
cancelled results have n_decoded == n_decoded_at_cancel == 16
cancelled results have cancel_observed_iter == 16
completed results have n_decoded == decode_budget
budget-8 completed hash == 0x0619d4d1900c2365
budget-64 completed results have one unique hash within the run
budget-256 completed results have one unique hash within the run
cancelled seqs are not decoded after cancellation
wasted_decode_rows_after_cancel == 0
KV clear succeeds for completed and cancelled results
sibling cross-talk checks pass
residual KV is empty for all seq_ids
every llama_decode returns 0
every promise is fulfilled exactly once
KV clear happens before promise fulfillment
--repeat 2 is deterministic except timing fields
```

---

## 7. Cancellation metrics and traces

Metrics are descriptive only:

```text
completed_count
cancelled_count
cancel_observed_iter_set
decoded_tokens_before_cancel
wasted_decode_rows_after_cancel
active_seqs_per_iter_after_cancellation
decode_calls_saved
```

For the canonical cancellation smoke:

```text
decode_calls_saved = 3*(64-16) + 3*(256-16) = 864
```

This number is not a speedup metric.

Trace events added behind `LLAMA_HPX_CB_TRACE=1`:

```text
cancel_requested
cancel_observed
cancel_kv_cleared
cancel_future_fulfilled
```

Completion-path events remain distinct:

```text
seq_complete
kv_cleared
promise_fulfilled
```

Trace-off runs must emit zero `[hpx-cb-gate] event=` lines.

---

## 8. Cancellation implementation slices

| Slice | Purpose | Final label |
|---|---|---|
| Cancel Slice 1 | Add cancellation data model only, no behavior change | `HPX_CB_CANCEL_STEP1: PASS` |
| Cancel Slice 2 | Add deterministic trigger and engine observation | `HPX_CB_CANCEL_STEP2: PASS` |
| Cancel Slice 3 | Add `status=cancelled` futures and cancellation traces | `HPX_CB_CANCEL_STEP3: PASS` |
| Cancel Slice 4 | Add cancellation metrics and results closeout | `HPX_CB_CANCEL_STEP4: PASS` |

After Cancel Slice 4, the prototype demonstrates a real HPX serving invariant: a request can leave early, clean its KV, and complete its future with a non-completed status while siblings continue.

---

## 9. Live-admission vocabulary

```text
request
  user-level lifecycle object; request_id is not seq_id once admission exists

seq_id / slot
  llama.cpp KV owner in [0, n_seq_max)

active request
  currently bound to a seq_id

waiting request
  queued but not yet bound to a seq_id

admission boundary
  top of a decode iteration after cancellation observation and before row construction

slot reuse
  binding a waiting request to a verified-empty seq_id

free_due_to_cancel
  ordered queue of seq_ids freed by cancellation KV clear

free_due_to_completion
  ordered queue of seq_ids freed by normal completion, used only when enabled

admission_source
  none | cancel_freed | completion_freed

arrival_source
  preloaded | external
```

`free_due_to_cancel` is not a global free-list. In the first live-admission smoke, naturally completed budget-8 slots are deliberately ignored so the smoke proves reuse-after-cancellation only.

---

## 10. Live Admission Slice 3 canonical smoke: cancel-freed slot reuse

The first full live-admission smoke reuses cancellation-freed slots only.

Shape:

```text
n_seq_max   = 99
n_active    = 93       request_ids 0..92, mapped 1:1 to seq_ids 0..92
n_waiting   = 6        request_ids 93..98
total       = 99

active budgets:
  request_id % 3 == 0 -> budget 8
  request_id % 3 == 1 -> budget 64
  request_id % 3 == 2 -> budget 256

waiting requests:
  request_ids 93..98
  decode_budget = 64
  FIFO order

cancel plan:
  request_ids {1,4,7,2,5,8}
  cancel_after = 16
```

Cancellation frees seq_ids:

```text
{1, 2, 4, 5, 7, 8}
```

Admission mapping at iter 17:

| waiting request | reused seq_id |
|---:|---:|
| 93 | 1 |
| 94 | 2 |
| 95 | 4 |
| 96 | 5 |
| 97 | 7 |
| 98 | 8 |

Why restrict to cancel-freed slots:

```text
budget-8 active seqs finish around done_iter=7.
By iter 17, generic lowest-free-slot reuse would pick seq_ids
{0,3,6,...}, accidentally testing completion-freed reuse.
The first smoke must test cancellation-freed reuse only.
```

Iter 17 mixed batch:

```text
36 prefill rows = 6 admitted requests * 6 prompt tokens
56 decode rows  = 28 surviving budget-64 + 28 surviving budget-256
total rows      = 92 <= n_batch 1024
```

Expected admitted completion:

```text
admitted_at_iter = 17
decode_budget = 64
done_iter = 17 + 64 - 1 = 80
pos_max_at_clear = 6 + 64 - 2 = 68
admission_source = cancel_freed
```

Surviving active budget-64 and budget-256 hashes may differ from the cancellation-only run because iter 17 is now mixed prefill+decode. Within-run uniqueness remains the gate.

---

## 11. Live-admission expected statuses

For the cancel-freed smoke:

```text
original completed:
  budget 8   -> 31
  budget 64  -> 28
  budget 256 -> 28
  subtotal   -> 87

original cancelled:
  budget 64  -> 3
  budget 256 -> 3
  subtotal   -> 6

admitted completed:
  budget 64  -> 6
  subtotal   -> 6

totals:
  completed = 93
  cancelled = 6
  futures   = 99
```

Per-result expectations:

```text
original completed:
  status=completed
  n_decoded == decode_budget
  admitted_at_iter == -1
  reused_seq_id == -1
  admission_source == none

original cancelled:
  status=cancelled
  n_decoded == 16
  n_decoded_at_cancel == 16
  cancel_observed_iter == 16
  pos_max_at_clear == 20
  admission_source == none

admitted completed:
  status=completed
  n_decoded == 64
  admitted_at_iter == 17
  done_iter == 80
  reused_seq_id in {1,2,4,5,7,8}
  admission_source == cancel_freed
  previous_request_id == reused_seq_id
```

---

## 12. Live-admission gates

The cancel-freed live-admission smoke must prove:

```text
engine task completes
engine_task_count == 1
all 99 futures complete
completed_count == 93
cancelled_count == 6
admitted_count == 6
admission_iter_set == {17}
reused_seq_id_set == {1,2,4,5,7,8}
no duplicate reused_seq_id
only cancel-freed slots are reused
no naturally completed budget-8 slot is reused
admission_source == cancel_freed for admitted results
admission_source == none for original results
previous_request_id mapping matches the cancel plan
admitted requests are not decoded before admission
admitted requests prefill exactly once
KV is verified empty before slot reuse
cancelled seqs are not decoded after cancellation
wasted_decode_rows_after_cancel == 0
budget-8 canonical hash remains 0x0619d4d1900c2365
surviving budget-64 / budget-256 partitions have within-run unique hashes
admitted budget-64 partition has within-run unique hash
KV clear / cross-talk / residual KV gates still pass
every llama_decode returns 0
every promise is fulfilled exactly once
KV clear happens before promise fulfillment
--repeat 2 is deterministic except timing fields
```

---

## 13. Live-admission traces and metrics

New trace events:

```text
request_queued
request_admitted_live
seq_reused
admitted_prefilled
admitted_decode_row
admitted_complete
```

Existing admission-independent events remain:

```text
engine_start
request_admitted       // original active only
seq_prefilled
decode_row
seq_complete
kv_cleared
promise_fulfilled
engine_stop
cancel_requested
cancel_observed
cancel_kv_cleared
cancel_future_fulfilled
```

For the cancel-freed smoke, expected trace counts include:

```text
request_admitted      = 93
request_queued        = 6
request_admitted_live = 6
seq_reused            = 6
seq_prefilled         = 93
admitted_prefilled    = 6
admitted_decode_row   = 378    // 6 admitted * 63 decode iters
seq_complete          = 87
admitted_complete     = 6
kv_cleared            = 93
promise_fulfilled     = 93
cancel_requested      = 6
cancel_observed       = 6
cancel_kv_cleared     = 6
cancel_future_fulfilled = 6
```

Admission metrics:

```text
queued_count
admitted_count
admission_iter_set
reused_seq_id_count
reused_seq_id_set
admitted_ttc_ms[budget=*]
waiting_queue_depth_after_admission_per_iter
```

For the cancel-freed smoke:

```text
queued_count = 6
admitted_count = 6
admission_iter_set = {17}
reused_seq_id_set = {1,2,4,5,7,8}
waiting queue depth after admission:
  p50=0 p95=6 max=6 samples=255
```

Metrics are descriptive only.

---

## 14. Live-admission implementation slices

| Slice | Purpose | Final label |
|---|---|---|
| Live Admission Slice 1 | Add admission data model only, no behavior change | `HPX_CB_ADMIT_STEP1: PASS` |
| Live Admission Slice 2 | Add waiting queue, no admission yet | `HPX_CB_ADMIT_STEP2: PASS` |
| Live Admission Slice 3 | Admit waiting requests into cancel-freed slots | `HPX_CB_ADMIT_STEP3: PASS` |
| Live Admission Slice 4 | Add traces, metrics, and results closeout | `HPX_CB_ADMIT_STEP4: PASS` |
| Live Admission Slice 5 | Add completion-freed slot reuse under `--reuse-completed` | `HPX_CB_ADMIT_STEP5: PASS` |
| Live Admission Slice 6 | Add deterministic async external arrivals | `HPX_CB_ADMIT_STEP6: PASS` |
| Live Admission Slice 7 | Prove mixed-source admission priority | `HPX_CB_ADMIT_STEP7: PASS` |

---

## 15. Live Admission Slice 5: completion-freed slot reuse

Slice 5 adds `completion_freed` as a second admission source.

CLI:

```text
--reuse-completed    // default OFF
--cancel-plan none   // explicit no-cancellation smoke
```

Source priority:

```text
1. free_due_to_cancel
2. free_due_to_completion
```

Completion-freed pooling is demand-gated:

```text
push completed slots only if:
  --reuse-completed is ON
  waiting_queue_consumable_ is non-empty
```

Smoke shape:

```text
n_seq_max      = 99
n_active       = 90
n_waiting      = 9
waiting_budget = 8
active budgets = round-robin {8,64,256}
cancel_plan    = none
```

Mapping:

```text
request 90 -> seq  0
request 91 -> seq  3
request 92 -> seq  6
request 93 -> seq  9
request 94 -> seq 12
request 95 -> seq 15
request 96 -> seq 18
request 97 -> seq 21
request 98 -> seq 24
```

Expected anchors:

```text
admission_iter_set == {8}
reused_seq_id_set == {0,3,6,9,12,15,18,21,24}
admission_source == completion_freed for every admitted result
admitted_count == 9
completion_freed_pool_size_at_run_end == 21
free_due_to_cancel remains empty
with --reuse-completed OFF:
  zero completion_freed admissions
  completion_freed_pool_size_at_run_end == 0
```

Hash policy:

```text
original budget-8 partition is strictly gated to 0x0619d4d1900c2365
surviving budget-64/256 and admitted budget-8 partitions are gated by
within-run uniqueness and repeat determinism, not cross-shape anchors
```

---

## 16. Live Admission Slice 6: async external arrivals

Slice 6 adds deterministic external arrivals using HPX primitives.

Path:

```text
scripted HPX submitter task
  -> waits on release_future
  -> engine::submit(arrival_msg)
  -> sets ack_promise
  -> engine waits for ack at release iter
  -> engine drains inbox at top of next iter
  -> existing admission path binds arrivals
```

HPX-native constraints:

```text
no std::thread
no std::condition_variable
no std::this_thread::sleep_for
no wall-clock timing
no new std::mutex
only new lock is hpx::spinlock inbox_mtx_
coordination uses hpx::promise<void> / hpx::future<void>
submitter does not call llama_* APIs
engine::submit() does not mutate engine_result
only the engine task touches llama.cpp state
```

New types:

```text
arrival_source ::= preloaded | external
arrival_msg { request_id, decode_budget, promise, arrival_source }
external_release_handle { release_future, ack_promise }
scripted_arrival { request_id, decode_budget, release_iter }
```

Smoke shape:

```text
n_seq_max               = 99
n_active                = 93
n_waiting               = 0
n_external_arrivals     = 6
external_arrival_budget = 64
external_release_iter   = 8
cancel_plan             = 1,4,7,2,5,8
cancel_after            = 16
--reuse-completed       = OFF
```

Expected timing:

```text
release_iter = 8
drain_iter   = 9
admit_iter   = 17
```

Expected mapping:

```text
request 93 -> seq 1
request 94 -> seq 2
request 95 -> seq 4
request 96 -> seq 5
request 97 -> seq 7
request 98 -> seq 8
```

Key gates:

```text
arrival_drained_count == n_external_arrivals
external_admitted_count == n_external_arrivals
first_external_drain_iter == external_release_iter + 1
iter_release_fired_set == {external_release_iter}
submitter_ack_set == {external_release_iter}
arrival_src == external for external admitted results
arrival_src == preloaded for all other results
admission_src == cancel_freed for external admitted results
inbox and external_promises_ empty at engine end
n_external_arrivals == 0 is inert
```

---

## 17. Live Admission Slice 7: mixed-source admission priority

Slice 7 proves the source-priority rule when both `free_due_to_cancel_` and `free_due_to_completion_` are non-empty at an admission boundary.

Priority rule:

```text
cancel_freed first
completion_freed second
```

Smoke shape:

```text
n_seq_max               = 99
n_active                = 84
n_waiting               = 9        // preloaded, budget 8
waiting_budget          = 8
n_external_arrivals     = 6        // external, budget 64
external_arrival_budget = 64
external_release_iter   = 16
--reuse-completed       = ON
cancel_plan             = 1,4,7,2,5,8
cancel_after            = 16
active budgets          = round-robin {8,64,256}
```

Timeline:

```text
iter 7:
  28 budget-8 actives complete
  free_due_to_completion_ gets 28 entries

iter 8:
  phase 1 admits 9 preloaded waiters via completion_freed
  residual completion pool = 19

iter 16:
  cancel plan fires
  free_due_to_cancel_ gets {1,2,4,5,7,8}
  external release/ack submits 6 arrivals

iter 17:
  phase 2 sees both pools non-empty
  source priority admits all 6 external arrivals via cancel_freed
  completion pool remains 19
```

Mappings:

```text
Phase 1, iter 8, completion_freed, preloaded:
  req 84 -> seq  0
  req 85 -> seq  3
  req 86 -> seq  6
  req 87 -> seq  9
  req 88 -> seq 12
  req 89 -> seq 15
  req 90 -> seq 18
  req 91 -> seq 21
  req 92 -> seq 24

Phase 2, iter 17, cancel_freed, external:
  req 93 -> seq 1
  req 94 -> seq 2
  req 95 -> seq 4
  req 96 -> seq 5
  req 97 -> seq 7
  req 98 -> seq 8
```

Strict gates:

```text
admission_iter_set == {8,17}

iter 8 admitted results:
  admission_src == completion_freed
  arrival_src == preloaded

iter 17 admitted results:
  admission_src == cancel_freed
  arrival_src == external

phase 1 mapping matches first n_waiting min-budget slots
phase 2 mapping matches sorted cancel_plan
no iter-17 result uses completion-freed residual slots
completion_freed_pool_size_at_run_end == 19
no duplicate reused_seq_id across the run
per-(admitted_at_iter, admission_src) reused_seq_ids are ascending by request_id
```

No new CLI flags, trace event names, or HPX primitives are introduced in Slice 7.

---

## 18. Shared out of scope

Across cancellation and admission, the following are out of scope for these designs:

```text
HTTP / network routes
client disconnect handling
production external cancellation
priority scheduling
preemption / requeue
deadline / timeout API
streaming partial responses
multiple llama_context instances
distributed serving
multi-host orchestration
named HPX orchestration pool / resource partitioner
performance claims
interrupting llama_decode
modifying tools/server/
modifying tools/serving-bench/
modifying tools/multiseq-batch-gate/
```

Slice 6 introduces a deterministic HPX submitter for external arrivals, but not a production network ingress path.

---

## 19. Risks and safeguards

### Cancellation risks

```text
double fulfillment
cancellation and completion colliding in the same iteration
decode rows emitted after cancellation
KV clear before sibling cross-talk snapshot
hash comparisons across changed batch shapes
```

Safeguards:

```text
observe cancellation only at iteration boundaries
fulfill cancelled futures only after KV clear
track wasted_decode_rows_after_cancel
gate residual KV empty
gate repeat determinism
avoid cross-shape hash claims for long budgets
```

### Admission risks

```text
mixed prefill+decode i_batch mistakes
stale promise reuse when a seq_id is reused
future delivery race for admitted requests
slot reuse before KV is actually empty
unclear request_admitted trace semantics
source-priority ambiguity
```

Safeguards:

```text
verify KV empty before binding
replace the slot's promise at admission after the prior owner is fulfilled
deliver admitted futures through an engine-owned collection
keep request_admitted for original active requests only
use request_admitted_live for live admissions
gate exact mappings and source labels
gate residual pool sizes
gate no duplicate reused_seq_id
```

---

## 20. Safe claims

Supported by these designs once their slices pass:

```text
HPX can represent completion and cancellation through request futures.
Cancellation can be cooperative and engine-owned.
Cancelled requests can leave decode batches without disturbing siblings.
KV cleanup can remain engine-task-owned.
Waiting requests can be admitted into verified-empty slots.
Cancel-freed and completion-freed slot reuse can be tested independently.
External arrivals can be modeled with HPX futures/promises and an engine inbox.
Source priority can be made deterministic and validated.
```

Not supported:

```text
HPX is faster than std
HPX is faster than upstream llama-server
the prototype is production-ready
HTTP behavior is implemented
client disconnect behavior is implemented
priority scheduling is implemented
streaming partial responses are implemented
distributed serving is implemented
```

---

## 21. Source documents consolidated

This file consolidates:

```text
docs/hpx/continuous_batching_cancellation_design.md
docs/hpx/continuous_batching_live_admission_design.md
```

The original documents contained full per-slice implementation prompts. Those prompts were intentionally omitted here to keep this file as a design reference rather than a task script.
