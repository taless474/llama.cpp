# HPX continuous-batching gate closeout

## Purpose

This note closes out the current `tools/hpx-continuous-batch-gate/` line of work.

The goal of this line was not to prove an HPX speedup. The goal was to prove that HPX can own request lifecycle and serving orchestration around a real `llama.cpp` continuous-batching primitive without violating the core `llama.cpp` execution invariants.

The completed prototype shows:

```text
HPX owns request orchestration.
llama.cpp owns model execution.
A single engine task owns llama_context / llama_batch / llama_decode / KV operations.
Per-request HPX futures represent request completion.
Cooperative cancellation is handled at iteration boundaries.
Completed and cancelled request futures both resolve with clean result snapshots.
```

This is a correctness and lifecycle closeout, not a performance closeout.

---

## Background

The earlier serving-level HPX experiment used HPX as a FIFO context-pool replacement around opaque `llama_decode` calls.

That path was closed out because the HPX backend was correct and robust, but did not outperform the simpler std backend on the tested CPU-only TinyLlama serving workloads.

The key conclusion from the FIFO context-pool work was:

```text
Replacing a simple std context pool with HPX is not enough.
HPX needs to own a real serving-level lifecycle invariant to be meaningful.
```

The continuous-batching direction was chosen because upstream `llama-server` is organized around:

```text
one shared llama_context
many active seq_ids / slots
one shared llama_batch
an update loop that admits, decodes, completes, and clears per-seq state
```

That shape is much closer to production LLM serving than a FIFO pool of independent contexts.

---

## Evidence chain before the HPX gate

Before writing the HPX prototype, two preparatory gates were completed.

### 1. Simulator

The simulator under:

```text
hpx-bench/sim/continuous_batching/
```

showed that continuous batching does not matter much on perfectly symmetric all-short workloads. In those cases, static batching and continuous batching mostly tie.

It did show meaningful separation on mixed or bursty workloads, especially mixed decode lengths.

The selected first real target was:

```text
mixed_decode_only
prompt_tokens = 6 for every request
decode budget mix = {8, 64, 256}
round-robin class assignment
all requests admitted at t = 0
```

This target isolates decode-length heterogeneity without adding long-prompt prefill complexity.

### 2. Pure llama.cpp multi-seq gate

The pure `llama.cpp` reference gate under:

```text
tools/multiseq-batch-gate/
```

proved the real primitive needed by the HPX prototype.

The important Step-5 shape was:

```text
n_seqs = 99
budget mix = {8, 64, 256}
33 seqs per budget class
prompt_tokens = 6
one llama_model
one llama_context
one shared llama_batch
shared prefill
shared decode loop
per-seq KV clear
```

The Step-5 reference passed with:

```text
actual n_ctx = 50688
actual n_seq_max = 99
actual n_batch = 1024

budget 8   hash = 0x0619d4d1900c2365
budget 64  hash = 0x88a4dc75a31d4325
budget 256 hash = 0x8a1a3bd01360aada

done_iter sets:
  8   -> {7}
  64  -> {63}
  256 -> {255}

pos_max_at_clear sets:
  8   -> {12}
  64  -> {68}
  256 -> {260}
```

The reference gate also proved repeat determinism for the same batch shape and clean per-seq KV lifecycle.

One important caveat came out of that work:

```text
Long-budget hashes are batch-shape dependent.
Do not use budget-64 or budget-256 hashes across different batch shapes as correctness anchors.
```

The safe correctness rules are:

```text
same-shape repeat determinism
within-run same-budget-class hash equality
expected token counts
expected done_iter
expected pos_max_at_clear
per-seq KV clear
no sibling KV cross-talk
every llama_decode returns 0
```

---

## HPX prototype boundary

The HPX prototype lives under:

```text
tools/hpx-continuous-batch-gate/
```

The pure reference gate remains separate under:

```text
tools/multiseq-batch-gate/
```

This separation is intentional. The reference gate is the pure `llama.cpp` closeout. The HPX gate is the orchestration prototype.

### HPX owns

```text
request metadata
request lifecycle state
per-request promises/futures
one engine task
result snapshots
trace events
descriptive metrics
future extension points for cancellation, priority, admission, and streaming
```

### llama.cpp owns

```text
llama_model
llama_context
llama_batch
llama_decode
llama_memory_seq_*
llama_get_logits_ith
tokenization
logits / greedy argmax inputs
```

### Hard rule

```text
Only the engine task may touch llama_context, llama_batch, llama_decode, llama_get_logits_ith, or llama_memory_seq_*.
No parallel llama_decode calls on the same llama_context.
```

Futures carry snapshots only. A future consumer must never touch `llama.cpp` context, batch, logits, or KV state.

---

## Prototype slices completed

### Slice 1: HPX runtime skeleton

Slice 1 proved that the new tool can start and stop HPX in the same process as `libllama`, load the model, create one context, negotiate the requested capacity, and build 99 metadata-only request objects.

Result:

```text
HPX_CB_STEP1: PASS
```

Observed capacity:

```text
actual n_ctx = 50688
actual n_seq_max = 99
actual n_batch = 1024
prompt_tokens = 6
hpx_os_threads = 1
```

No `llama_batch` or `llama_decode` work was performed in this slice.

---

### Slice 2: one HPX engine task owns the Step-5 loop

Slice 2 moved the proven Step-5 loop into an `engine` object and ran that engine as exactly one HPX task.

Result:

```text
HPX_CB_STEP2: PASS
```

This proved that HPX could own the engine-task lifecycle without disturbing the pure `llama.cpp` Step-5 batch shape.

The engine task was the sole code path touching:

```text
llama_context
llama_batch
llama_decode
llama_get_logits_ith
llama_memory_seq_*
```

Step-5 correctness was reproduced:

```text
budget 8   hash = 0x0619d4d1900c2365
budget 64  hash = 0x88a4dc75a31d4325
budget 256 hash = 0x8a1a3bd01360aada

done_iter sets:
  8   -> {7}
  64  -> {63}
  256 -> {255}

pos_max_at_clear sets:
  8   -> {12}
  64  -> {68}
  256 -> {260}
```

Repeat-2 determinism passed.

---

### Slice 3: per-request HPX futures/promises

Slice 3 introduced one HPX promise/future pair per sequence.

Result:

```text
HPX_CB_STEP3: PASS
```

The engine fulfilled each promise only after:

```text
the sequence reached its decode budget
generated-token state was finalized
pos_max_at_clear was recorded
per-seq KV clear succeeded
sibling cross-talk check passed
```

Main waited on the request futures and validated only from `request_result` snapshots.

Key counters:

```text
futures_created = 99
promises_fulfilled = 99
futures_completed = 99
engine_task_count = 1
decode_failures = 0
```

This was the first truly HPX-native serving milestone in the prototype: request completion became future-based while the single-engine ownership boundary remained intact.

---

### Slice 4: lifecycle traces and descriptive metrics

Slice 4 added env-gated lifecycle traces and a descriptive metrics block.

Result:

```text
HPX_CB_STEP4: PASS
```

Trace env var:

```text
LLAMA_HPX_CB_TRACE=1
```

The trace stream uses stable key/value events on stderr.

Expected trace counts in the non-cancelled 99-seq run were:

```text
engine_start       = 1
engine_stop        = 1
request_admitted   = 99
seq_prefilled      = 99
decode_row         = 10725
seq_complete       = 99
kv_cleared         = 99
promise_fulfilled  = 99
```

The metrics block included:

```text
wall_ms
decode_calls
update_iterations
rows_per_batch p50/p95/max
active_seqs_per_iter p50/p95/max
completed counts by budget
ttc_ms by budget class
futures_created
promises_fulfilled
futures_completed
engine_task_count
```

All metrics are descriptive. They are not speedup evidence.

---

### Slice 5: same-shape HPX-off vs HPX-on comparison

Slice 5 compared the pure `llama.cpp` reference gate against the HPX prototype using the same batch shape.

Result in the HPX closeout evidence:

```text
HPX_CB_PROTO: PASS
```

The pure reference emitted:

```text
GATE_STEP5: PASS
```

The HPX prototype emitted:

```text
HPX_CB_STEP4: PASS
```

Same-shape hash equivalence:

```text
budget 8:   0x0619d4d1900c2365 == 0x0619d4d1900c2365
budget 64:  0x88a4dc75a31d4325 == 0x88a4dc75a31d4325
budget 256: 0x8a1a3bd01360aada == 0x8a1a3bd01360aada
```

Same-shape lifecycle equivalence:

```text
done_iter:
  8   -> {7}
  64  -> {63}
  256 -> {255}

pos_max_at_clear:
  8   -> {12}
  64  -> {68}
  256 -> {260}
```

The comparison was correctness equivalence only, not performance comparison.

---

## Cooperative cancellation

After the base HPX prototype passed, cooperative cancellation was added as the first real HPX-native serving capability.

Cancellation is cooperative:

```text
no interruption inside llama_decode
observed only at iteration boundaries
checked at the top of each decode iteration
cancelled seqs are removed from later decode batches
cancelled seq KV is cleared by the engine task
cancelled futures are fulfilled with status = cancelled
non-cancelled seqs continue
```

Initial cancellation plan:

```text
cancel seqs {1, 4, 7} from budget 64
cancel seqs {2, 5, 8} from budget 256
cancel_after_decoded_tokens = 16
budget 8 is not cancelled
```

Expected final shape:

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

---

### Cancel Slice 1: data model only

Cancel Slice 1 added the cancellation data model without enabling cancellation behavior.

Result:

```text
HPX_CB_CANCEL_STEP1: PASS
```

Added model fields included:

```text
request_status
cancel_requested
cancel_after_decoded_tokens
cancel_observed
cancel_observed_iter
n_decoded_at_cancel
status on request_result
```

The cancellation plan was printed and validated, but not propagated into active behavior.

All prior Slice 3/4/5 correctness gates remained green.

---

### Cancel Slice 2: deterministic cancellation behavior

Cancel Slice 2 enabled deterministic cancellation inside the engine.

Result:

```text
HPX_CB_CANCEL_STEP2: PASS
```

Observed result:

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

cancel_observed_iter_set = {16}
n_decoded_at_cancel_set = {16}
wasted_decode_rows_after_cancel = 0
residual_kv_empty = true
decode_failures = 0
```

Cancelled futures were fulfilled only after KV clear and sibling cross-talk checks.

---

### Cancel Slice 3: explicit status and cancellation trace path

Cancel Slice 3 polished the status path and trace events.

Result:

```text
HPX_CB_CANCEL_STEP3: PASS
```

Status validation:

```text
completed results:
  status == completed
  n_decoded == decode_budget
  cancel_observed_iter == -1
  n_decoded_at_cancel == -1
  kv_cleared == true

cancelled results:
  status == cancelled
  n_decoded == n_decoded_at_cancel
  n_decoded < decode_budget
  cancel_observed_iter == 16
  n_decoded_at_cancel == 16
  kv_cleared == true
  pos_max_at_clear == 20
```

Cancellation trace events were added under `LLAMA_HPX_CB_TRACE=1`:

```text
cancel_requested
cancel_observed
cancel_kv_cleared
cancel_future_fulfilled
```

Trace counts in the cancellation smoke:

```text
engine_start             = 1
engine_stop              = 1
request_admitted         = 99
seq_prefilled            = 99
decode_row               = 9861
seq_complete             = 93
kv_cleared               = 93
promise_fulfilled        = 93
cancel_requested         = 6
cancel_observed          = 6
cancel_kv_cleared        = 6
cancel_future_fulfilled  = 6
```

---

### Cancel Slice 4: metrics and cancellation closeout

Cancel Slice 4 fixed and expanded the metrics/reporting path.

Result:

```text
HPX_CB_CANCEL_STEP4: PASS
```

The metrics block now correctly separates completed and cancelled counts by budget.

Example compact-run metrics:

```text
status_summary: completed=93 cancelled=6 total=99
residual_kv: all 99 seqs cleared

decode_calls = 256
update_iterations = 255
rows_per_batch p50=30 p95=66 max=594
active_seqs_per_iter p50=30 p95=66 max=99

completed[budget=8] = 33
completed[budget=64] = 30
cancelled[budget=64] = 3
completed[budget=256] = 30
cancelled[budget=256] = 3

futures_created = 99
promises_fulfilled = 99
futures_completed = 99
engine_task_count = 1
completed_count = 93
cancelled_count = 6
decode_failures = 0
wasted_decode_rows_after_cancel = 0
residual_kv_empty = true
```

Cancellation is now closed out as a correctness/lifecycle feature.

---

## What was proven

The completed gate proves:

```text
HPX can own the request lifecycle around a real llama.cpp continuous-batching primitive.

A single HPX-owned engine task can preserve the llama.cpp batch/decode/KV invariants.

Per-request HPX futures can represent independent request completion.

Request futures can complete with status = completed or status = cancelled.

Cancellation can be observed cooperatively at iteration boundaries.

Cancelled seqs can be removed from future decode batches.

Cancelled seq KV can be cleared safely by the engine task.

Cancelled futures can be fulfilled only after KV cleanup.

Non-cancelled sibling seqs can continue to completion.

Final residual KV can be empty for all seqs.

The HPX prototype can match the pure llama.cpp same-shape reference on correctness fields.
```

This is a real HPX serving-runtime milestone.

---

## What was deliberately not proven

This line does not prove:

```text
HPX is faster than std
HPX is faster than upstream llama-server
the prototype is production-ready
HTTP behavior
streaming behavior
client disconnect behavior
external async cancellation
priority scheduling
live admission of new requests
multiple llama_context instances
multi-node or distributed serving
GPU/Metal performance behavior
general performance claims
```

The current prototype is a correctness-first orchestration gate.

---

## HPX nativity assessment

This work is HPX-native in the serving-runtime sense:

```text
HPX owns request lifecycle.
HPX futures represent request completion.
HPX promises are fulfilled at precise lifecycle boundaries.
HPX controls the engine task.
HPX traces describe serving lifecycle events.
HPX cancellation semantics are represented in request results.
```

It is not yet HPX-rich in the scheduling sense:

```text
there is no live admission yet
there is no priority queue yet
there is no external cancellation source yet
there is no streaming/result channel yet
there is no dedicated orchestration executor/pool yet
```

That is intentional. The current work first established safe ownership and lifecycle semantics around `llama.cpp`.

A named HPX orchestration pool/resource-partitioner is deferred. With only one engine task, a named pool does not change the runtime architecture meaningfully. It becomes useful once the prototype has multiple HPX-side tasks, such as admission, cancellation-source tasks, priority scheduling, streaming dispatch, or result-processing continuations.

---

## Recommended next feature

The strongest next feature is **live admission**.

Cancellation proved that a request can leave the active batch early:

```text
active request -> cancelled -> KV cleared -> future fulfilled
```

Live admission would prove the other half of continuous batching:

```text
waiting request -> admitted into freed capacity -> participates in later decode iterations
```

Recommended next question:

```text
Can the HPX engine admit new requests while the decode loop is already running, without breaking batch-shape accounting, per-seq KV lifecycle, or request-future semantics?
```

A good first live-admission smoke could be:

```text
start with fewer than 99 active seqs
keep additional requests waiting
when cancelled seqs leave at iter 16, admit waiting requests into freed seq_id/slot capacity
new requests prefill at an iteration boundary
new requests then join decode batches
all original completed/cancelled/new-admitted futures resolve cleanly
residual KV is empty at end
```

This would move the prototype closer to real continuous batching while preserving the same core engine ownership rule.

---

## Closeout statement

The HPX continuous-batching gate has reached its intended correctness milestone.

The prototype now demonstrates:

```text
same-shape equivalence to the pure llama.cpp multi-seq reference
future-based request completion
engine-owned KV lifecycle
env-gated lifecycle traces
descriptive metrics
cooperative cancellation with clean cancelled futures
no residual KV
no sibling cross-talk
repeat determinism for the tested shapes
```

This is enough to close the gate and move to the next HPX-serving capability.
