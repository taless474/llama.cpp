# Continuous batching foundation

This document consolidates the pre-streaming HPX continuous-batching foundation:

```text
docs/hpx/continuous_batching_phase3_target.md
docs/hpx/multiseq_llama_batch_gate.md
docs/hpx/continuous_batching_prototype_design.md
docs/hpx/continuous_batching_gate_closeout.md
```

It is a foundation and closeout note for the path that led to `tools/hpx-continuous-batch-gate/`. It is not a streaming-slice design, not an HTTP/server design, and not a performance claim.

## 1. Scope and boundary

The direction was chosen after the FIFO context-pool experiment closed out. Replacing a simple std context pool with HPX around opaque `llama_decode` calls was correct, but did not show a useful latency benefit on the tested CPU-only TinyLlama serving workloads.

The new direction was:

```text
HPX owns request orchestration.
llama.cpp owns model execution.
```

The hard boundary is `llama_decode`:

```text
HPX schedules around llama_decode.
HPX does not parallelize inside llama_decode.
Only one engine task touches llama_context, llama_batch, llama_decode,
llama_get_logits_ith, or llama_memory_seq_*.
```

This boundary stays constant across the foundation documents.

## 2. Phase 3 target workload

The simulator suggested that the first real prototype should target mixed decode lengths, not symmetric all-short workloads.

### Primary target: mixed_decode_only

```text
all requests arrive at t = 0
prompt_tokens = 6 for every request
decode budget mix = {8, 64, 256}
class assignment = request_id % 3
initial simulator config = n_slots 4, n_batch 128
suggested real size = 99 requests
```

Reason: this isolates decode-length heterogeneity while avoiding long-prompt prefill complexity. The simulator showed a large scheduling-level separation on this shape: makespan around 30% lower, TTFT p50 around 36% lower, tokens per decode call around 125% higher, and decode calls around 55% lower for continuous batching versus static batching.

### Secondary target: mixed_prompt_and_decode

```text
round-robin classes:
  (prompt=16,   decode=8)
  (prompt=128,  decode=64)
  (prompt=1024, decode=256)
```

This was more realistic but intentionally deferred because long prompts add multi-iteration prefill and `n_batch` complexity.

### Not selected first

```text
A: exp11_like_all_short
B: staggered_identical_short
D: mixed_prompt_only
F: bursty_mixed_deterministic
```

Those shapes either showed too little simulator separation or introduced confounds that were not needed for the first real gate.

## 3. Pure llama.cpp multi-sequence gate

Before adding HPX, the project needed to prove the real `llama.cpp` primitive:

```text
one llama_model
one llama_context
many active seq_ids
one shared llama_batch
shared prefill
shared decode loop
per-seq logits readback
per-seq KV clear
deterministic per-seq hashes
```

This became the separate pure reference tool:

```text
tools/multiseq-batch-gate/
```

It intentionally contained no HPX code, no server code, no streaming, no cancellation, and no performance claim.

### Minimal technical shape

The gate needed to prove:

```text
multiple llama_seq_id values active in one context
one shared llama_batch reused across iterations
same fixed prompt for every seq
greedy argmax via llama_get_logits_ith
one decode row per active seq per iteration
positions tracked per seq_id
per-seq KV clear is effective
clearing one seq does not corrupt siblings
repeat runs are deterministic
```

The core operation order was:

```text
load model and context
tokenize prompt once
assign seq_ids
allocate one shared llama_batch
prefill all seqs in one shared batch
decode one row per active seq per iteration
read logits by recorded i_batch row
argmax next token
mark seq done at budget
clear that seq's KV
validate no sibling cross-talk
```

### Small smoke

The smallest meaningful smoke was:

```text
seqs = 3
prompt = "Hello, my name is"
decode budgets = {8, 64, 256}
TinyLlama 1.1B Q4_K_M
CPU-only
greedy argmax
```

This exercises staggered completion and per-seq KV clear.

### Scale-up target

The Phase 3 reference target was:

```text
n_seqs = 99
budget mix = {8, 64, 256}
33 seqs per budget class
prompt_tokens = 6
n_batch = 1024
```

The completed Step-5 reference shape passed with:

```text
actual n_ctx = 50688
actual n_seq_max = 99
actual n_batch = 1024

budget 8   hash = 0x0619d4d1900c2365
budget 64  hash = 0x88a4dc75a31d4325
budget 256 hash = 0x8a1a3bd01360aada

done_iter:
  8   -> {7}
  64  -> {63}
  256 -> {255}

pos_max_at_clear:
  8   -> {12}
  64  -> {68}
  256 -> {260}
```

The reference also proved repeat determinism for the same batch shape and clean per-seq KV lifecycle.

### Hash caveat

Long-budget hashes are batch-shape dependent.

Safe rules:

```text
budget-8 hash is the only currently validated cross-shape anchor
budget-64 and budget-256 hashes are same-shape anchors only
compare long-budget hashes only against the same prompt, same batch shape,
same policy, same n_seqs, same n_batch, and same per-iteration composition
```

A future change that mixes prefill and decode rows differently can validly change long-budget hashes.

## 4. HPX continuous-batching prototype

After the pure gate passed, the HPX prototype was placed in a separate tool:

```text
tools/hpx-continuous-batch-gate/
```

The pure gate remained frozen as the closeout reference:

```text
tools/multiseq-batch-gate/
```

The HPX prototype was intentionally a correctness-first orchestration gate.

### HPX owns

```text
request metadata
request lifecycle state
per-request promises and futures
one engine task
result snapshots
trace events
descriptive metrics
future extension points for cancellation, admission, priority, and streaming
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
logits and greedy argmax inputs
```

### First prototype shape

The first HPX prototype preserved the pure Step-5 execution shape:

```text
one llama_model
one llama_context
many seq_ids
one shared llama_batch
mixed budgets {8, 64, 256}
same prompt for every seq
greedy argmax
per-seq completion at n_decoded == budget
per-seq llama_memory_seq_rm(..., -1, -1)
```

The important change was ownership, not model behavior:

```text
request completion became hpx::future<request_result>
the engine loop ran as one HPX task
main waited on futures and validated request_result snapshots
```

## 5. HPX prototype slice sequence

### Slice 1: HPX runtime skeleton

Goal: prove the new tool can start/stop HPX in the same process as `libllama`, load the model, create one context, negotiate capacity, and build 99 metadata-only request objects.

No `llama_batch` or `llama_decode` work was performed.

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

### Slice 2: one HPX engine task owns the Step-5 loop

Goal: move the proven Step-5 loop into an `engine` object and run it as exactly one HPX task.

Result:

```text
HPX_CB_STEP2: PASS
```

The engine task was the only code path touching `llama_context`, `llama_batch`, `llama_decode`, `llama_get_logits_ith`, and `llama_memory_seq_*`.

Step-5 correctness was reproduced:

```text
budget 8   hash = 0x0619d4d1900c2365
budget 64  hash = 0x88a4dc75a31d4325
budget 256 hash = 0x8a1a3bd01360aada

done_iter:
  8   -> {7}
  64  -> {63}
  256 -> {255}

pos_max_at_clear:
  8   -> {12}
  64  -> {68}
  256 -> {260}
```

### Slice 3: per-request HPX futures/promises

Goal: represent each request completion through an HPX promise/future pair.

Result:

```text
HPX_CB_STEP3: PASS
```

The engine fulfilled each promise only after:

```text
request reached decode budget
generated-token state was finalized
pos_max_at_clear was recorded
per-seq KV clear succeeded
sibling cross-talk check passed
```

Key counters:

```text
futures_created = 99
promises_fulfilled = 99
futures_completed = 99
engine_task_count = 1
decode_failures = 0
```

This was the first HPX-native serving milestone in the prototype: request completion became future-based while the single-engine ownership boundary held.

### Slice 4: traces and descriptive metrics

Goal: add env-gated lifecycle traces and descriptive metrics.

Result:

```text
HPX_CB_STEP4: PASS
```

Trace env var:

```text
LLAMA_HPX_CB_TRACE=1
```

Expected trace counts in the non-cancelled 99-seq run:

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

Metrics included:

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

All metrics were descriptive. They were not speedup evidence.

### Slice 5: same-shape HPX-off vs HPX-on comparison

Goal: compare the pure `llama.cpp` reference gate against the HPX prototype using the same batch shape.

Result:

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

This was correctness equivalence only, not performance comparison.

## 6. Cooperative cancellation closeout

After the base prototype passed, cooperative cancellation was added as the first real HPX-owned serving capability.

Cancellation semantics:

```text
no interruption inside llama_decode
observed only at iteration boundaries
checked before active decode rows are built
cancelled seqs leave future decode batches
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

Final expected shape:

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

### Cancel Slice 1: data model only

Result:

```text
HPX_CB_CANCEL_STEP1: PASS
```

Added fields included:

```text
request_status
cancel_requested
cancel_after_decoded_tokens
cancel_observed
cancel_observed_iter
n_decoded_at_cancel
status on request_result
```

The plan was printed and validated, but cancellation behavior was not enabled.

### Cancel Slice 2: deterministic cancellation behavior

Result:

```text
HPX_CB_CANCEL_STEP2: PASS
```

Observed:

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

### Cancel Slice 3: status and traces

Result:

```text
HPX_CB_CANCEL_STEP3: PASS
```

Status validation:

```text
completed:
  status == completed
  n_decoded == decode_budget
  cancel_observed_iter == -1
  n_decoded_at_cancel == -1
  kv_cleared == true

cancelled:
  status == cancelled
  n_decoded == n_decoded_at_cancel
  n_decoded < decode_budget
  cancel_observed_iter == 16
  n_decoded_at_cancel == 16
  kv_cleared == true
  pos_max_at_clear == 20
```

Cancellation trace events:

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

### Cancel Slice 4: metrics and closeout

Result:

```text
HPX_CB_CANCEL_STEP4: PASS
```

The metrics/reporting path separated completed and cancelled counts by budget.

Compact-run metrics:

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

Cancellation is closed as a correctness/lifecycle feature.

## 7. Correctness gates

Across the pure gate and HPX prototype, these are the core gates:

```text
every request future completes
every seq reaches the expected status and token count
budget-class counts are exact
budget-8 hash matches the canonical anchor
long-budget hashes are same-shape only
done_iter sets match expected values
pos_max_at_clear sets match expected values
per-seq KV clear is effective
clearing one seq does not disturb siblings
residual KV is empty at end
every llama_decode returns 0
repeat-2 is deterministic for the tested shape
the final stdout line is the expected PASS or specific FAIL label
```

The HPX-specific additions are:

```text
one engine task per run
one promise/future pair per request
promise fulfilled only after KV cleanup and cross-talk check
future consumers validate from snapshots only
cancellation futures resolve as completed or cancelled
```

## 8. Metrics and trace policy

Metrics are descriptive, not acceptance claims about performance.

Trace policy:

```text
LLAMA_HPX_CB_TRACE unset:
  trace-off path should be quiet

LLAMA_HPX_CB_TRACE=1:
  emit lifecycle key/value events on stderr
```

Trace events describe request lifecycle and debugging evidence. They do not replace the correctness gates.

## 9. What was proven

The foundation work proved:

```text
real llama.cpp can run the 99-seq shared-batch mixed-budget target
per-seq KV clear and sibling cross-talk checks are safe on that shape
HPX can own the engine-task lifecycle around that primitive
HPX futures can represent per-request completion
request results can be validated from snapshots
HPX-on matches the pure llama.cpp same-shape reference on correctness fields
cooperative cancellation works at iteration boundaries
completed and cancelled futures both resolve cleanly
cancelled seqs are removed from later decode batches
residual KV is empty at engine end
```

This is a correctness and lifecycle milestone for HPX serving orchestration.

## 10. What was not proven

This foundation does not prove:

```text
HPX is faster than std
HPX is faster than upstream llama-server
the prototype is production-ready
HTTP behavior
streaming behavior
client disconnect behavior
external async cancellation
priority scheduling
live admission
multiple llama_context instances
multi-node or distributed serving
GPU/Metal performance behavior
general performance claims
```

Those are separate axes.

## 11. HPX nativity assessment

This work is HPX-native in the serving-runtime sense:

```text
HPX owns request lifecycle
HPX futures represent request completion
HPX promises are fulfilled at lifecycle boundaries
HPX controls the engine task
HPX traces describe serving lifecycle events
HPX cancellation semantics appear in request results
```

It is not yet HPX-rich in the broader scheduling sense:

```text
no live admission yet
no priority queue yet
no external cancellation source yet
no streaming channel yet
no dedicated orchestration executor/pool yet
```

That was intentional. The first milestone established safe ownership and lifecycle semantics around `llama.cpp`.

A named HPX orchestration pool/resource partitioner was deferred. With only one engine task, a named pool does not change the architecture meaningfully. It becomes useful once the prototype has multiple HPX-side tasks such as admission, cancellation sources, priority scheduling, streaming dispatch, or result-processing continuations.

## 12. Recommended next feature from this closeout

The recommended next feature after the foundation closeout was live admission.

Cancellation proved:

```text
active request -> cancelled -> KV cleared -> future fulfilled
```

Live admission should prove:

```text
waiting request -> admitted into freed capacity -> participates in later decode iterations
```

Recommended question:

```text
Can the HPX engine admit new requests while the decode loop is already running,
without breaking batch-shape accounting, per-seq KV lifecycle, or request-future semantics?
```

A first live-admission smoke could:

```text
start with fewer than 99 active seqs
keep additional requests waiting
cancel some seqs at iter 16
admit waiting requests into freed seq_ids
prefill new requests at an iteration boundary
let new requests join decode batches
resolve all original completed/cancelled/new-admitted futures
leave residual KV empty at end
```

## 13. Closeout statement

The foundation reached its intended correctness milestone.

The prototype demonstrated:

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

This closed the pre-streaming continuous-batching gate and created the foundation for later live admission and streaming work.
