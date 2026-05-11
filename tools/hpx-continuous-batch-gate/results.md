# hpx-continuous-batch-gate — results

This document records the slice-by-slice correctness evidence for
the HPX continuous-batching prototype in
`tools/hpx-continuous-batch-gate/`.

> **This is a correctness-equivalence record, not a performance
> comparison.** No HPX-vs-std speedup or slowdown is claimed.
> Walltime/ttc fields are descriptive only and are not part of any
> determinism or equivalence contract.

Common shape used across all slices (Phase 3 target):

```text
model:       /Users/unick/Desktop/hpx/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
prompt:      "Hello, my name is"
n_seqs:      99
budget mix:  {8, 64, 256}, round-robin → 33 + 33 + 33
ctx_size:    32768  (actual n_ctx = 50688 after llama.cpp rounding)
n_batch:     1024
n_threads:   2     (libllama compute)
hpx_threads: 1     (HPX orchestration)
greedy argmax, single shared llama_batch, one-shot prefill,
per-step decode, per-seq KV clear on completion.
```

Same-shape canonical hashes (only valid within this shape):

```text
budget   8: 0x0619d4d1900c2365   (cross-shape canonical anchor)
budget  64: 0x88a4dc75a31d4325
budget 256: 0x8a1a3bd01360aada
```

The budget-8 hash is the cross-shape canonical anchor inherited
from the multiseq-batch-gate (Q7 Step 2). The budget-64 and
budget-256 hashes are only comparable when the batch shape is
identical to this run; do **not** compare them across runs with
different `n_seqs` or different scheduling policies.

---

## Slice 1 — HPX runtime skeleton

Final stdout line: `HPX_CB_STEP1: PASS`

- HPX runtime starts and stops cleanly within the same process as
  `libllama` (mirrors `tools/serving-bench/runtime_hpx.cpp`).
- Loads the model, creates one `llama_context`, prints structural
  capacity fields, builds 99 metadata-only request objects.
- No `llama_decode`, no `llama_batch` allocated.

Structural actuals: `actual n_ctx=50688`, `actual n_seq_max=99`,
`actual n_batch=1024`, `prompt_tokens=6`, `hpx_os_threads=1`.

Capture: `local/hpx_cb_step1.stdout`.

---

## Slice 2 — engine as one HPX task

Final stdout line: `HPX_CB_STEP2: PASS` (single + `--repeat 2`).

- An `engine` class owns `llama_context*`, the shared
  `llama_batch`, the per-seq state vector, and decode-loop state.
- `engine::run()` is the sole code path that touches
  `llama_context` / `llama_batch` / `llama_decode` /
  `llama_memory_seq_*` / `llama_get_logits_ith`.
- `main()` schedules `engine::run()` via
  `hpx::async([&eng]{eng.run();})` and waits on the returned
  future. Exactly one HPX engine task per repeat iteration.
- Reproduces all Step-5 correctness gates on the 99-seq mixed
  shape and confirms `engine_task_count == 1`.

Per-budget actuals (single run; iter[1] identical with `--repeat 2`):

| budget | count | unique_hash_count | hash                 | done_iter_set | pos_max_at_clear_set |
|-------:|------:|------------------:|----------------------|--------------:|---------------------:|
|     8  |    33 |                 1 | `0x0619d4d1900c2365` |          `{7}` |               `{12}` |
|    64  |    33 |                 1 | `0x88a4dc75a31d4325` |         `{63}` |               `{68}` |
|   256  |    33 |                 1 | `0x8a1a3bd01360aada` |        `{255}` |              `{260}` |

`decode_calls = 256`, `decode_failures = 0`, residual KV empty for
all 99 seqs. Captures: `local/hpx_cb_step2.stdout`,
`local/hpx_cb_step2_repeat2.stdout`.

---

## Slice 3 — per-request HPX promise/future

Final stdout line: `HPX_CB_STEP3: PASS` (single + `--repeat 2`).

- One `hpx::promise<request_result>` per seq, with futures handed
  out before the engine task is scheduled.
- The engine fulfills each promise only after that seq has
  reached its budget, recorded `pos_max_at_clear`, had its KV
  cleared, and passed the cross-talk-against-still-active-siblings
  check.
- `main()` uses `hpx::wait_all` on the per-request futures, calls
  `engine_fut.get()` to surface engine-task exceptions, and
  validates correctness exclusively from `request_result`
  snapshots.
- Boundary tightened: `main()` no longer touches
  `llama_context` / `llama_batch` / `llama_decode` /
  `llama_memory_seq_*` / `llama_get_logits_ith`. The
  residual-KV-empty check moved into the engine task.

Per-iter HPX gates: `engine_task_count = 1`,
`futures_created = 99`, `promises_fulfilled = 99`,
`futures_completed = 99`. No promise fulfilled twice. Every
fulfilled `request_result` carries `kv_cleared = true`.

Captures: `local/hpx_cb_step3.stdout`,
`local/hpx_cb_step3_repeat2.stdout`.

---

## Slice 3b — orchestration pool / resource partitioner

**Optional later. Not scheduled.** With only one engine task, a
named HPX orchestration pool would not change runtime behavior in
a meaningful way, and adding `hpx::resource::partitioner` /
executor wiring now would expand the failure surface against
gates that are about futures/promises, not thread topology. This
slice becomes useful only when there are multiple HPX-side tasks
that need real separation from libllama compute (admission,
cancellation, priority, streaming/result dispatch, metrics
continuations).

---

## Slice 4 — lifecycle traces and descriptive metrics

Final stdout line: `HPX_CB_STEP4: PASS` (compact, `--repeat 2`,
and trace-on smoke).

- env-gated trace events on stderr (`LLAMA_HPX_CB_TRACE=1`):
  `engine_start`, `request_admitted`, `seq_prefilled`,
  `decode_row`, `seq_complete`, `kv_cleared`, `promise_fulfilled`,
  `engine_stop`. With trace disabled, the path is one atomic load
  + early return per call site.
- per-iter descriptive metrics block on stdout: `wall_ms`,
  `decode_calls`, `update_iterations`,
  `rows_per_batch` (p50/p95/max),
  `active_seqs_per_iter` (p50/p95/max), per-budget completed
  counts, per-budget `ttc_ms` (mean / p95), `futures_created`,
  `promises_fulfilled`, `futures_completed`, `engine_task_count`.
- All Slice-3 correctness gates still pass. Trace and metrics do
  not change correctness behavior or batch shape.

Trace event counts on the 99-seq run (`LLAMA_HPX_CB_TRACE=1`):

```text
engine_start         1
request_admitted    99
seq_prefilled       99
decode_row       10725   (= 7×99 + 56×66 + 192×33)
seq_complete        99
kv_cleared          99
promise_fulfilled   99
engine_stop          1
```

Compact runs (env unset) emit zero `event=` lines on stderr.

Captures: `local/hpx_cb_step4.stdout`,
`local/hpx_cb_step4_repeat2.stdout`,
`local/hpx_cb_step4_trace.stdout`,
`local/hpx_cb_step4_trace.stderr`.

---

## Slice 5 — same-shape HPX-off vs HPX-on equivalence

**Correctness equivalence only. Not a performance comparison.**

Same-shape inputs were sent to:

1. Pure llama.cpp reference gate
   (`tools/multiseq-batch-gate/`, binary
   `/Users/unick/Desktop/HPX/builds/llama-base/bin/llama-multiseq-batch-gate`).
2. HPX prototype
   (`tools/hpx-continuous-batch-gate/`, binary
   `/Users/unick/Desktop/HPX/builds/llama-hpx-hpx-on/bin/llama-hpx-continuous-batch-gate`).

Commands:

```sh
# Reference
/Users/unick/Desktop/HPX/builds/llama-base/bin/llama-multiseq-batch-gate \
  --model /Users/unick/Desktop/hpx/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --n-seqs 99 --decode-budget-mix 8,64,256 \
  --ctx-size 32768 --n-batch 1024 --n-threads 2 \
  > local/hpx_cb_slice5_reference.stdout 2> local/hpx_cb_slice5_reference.stderr

# HPX prototype
/Users/unick/Desktop/HPX/builds/llama-hpx-hpx-on/bin/llama-hpx-continuous-batch-gate \
  --model /Users/unick/Desktop/hpx/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --n-seqs 99 --decode-budget-mix 8,64,256 \
  --ctx-size 32768 --n-batch 1024 --n-threads 2 \
  > local/hpx_cb_slice5_hpx.stdout 2> local/hpx_cb_slice5_hpx.stderr
```

### Self-check final lines

- Reference: `GATE_STEP5: PASS`
- HPX:       `HPX_CB_STEP4: PASS`

The HPX binary continues to emit its own self-check label
(`HPX_CB_STEP4`); the comparison verdict `HPX_CB_PROTO: PASS` is
declared from this evidence file rather than from the binary.

### Side-by-side correctness fields

| Field                       | Reference                          | HPX                                | Match |
|-----------------------------|------------------------------------|------------------------------------|-------|
| `actual n_ctx`              | 50688                              | 50688                              | yes   |
| `actual n_seq_max`          | 99                                 | 99                                 | yes   |
| `actual n_batch`            | 1024                               | 1024                               | yes   |
| `prompt_tokens`             | 6                                  | 6                                  | yes   |
| `n_seqs`                    | 99                                 | 99                                 | yes   |
| budget=8 count              | 33                                 | 33                                 | yes   |
| budget=64 count             | 33                                 | 33                                 | yes   |
| budget=256 count            | 33                                 | 33                                 | yes   |
| budget=8 unique_hashes      | 1                                  | 1                                  | yes   |
| budget=64 unique_hashes     | 1                                  | 1                                  | yes   |
| budget=256 unique_hashes    | 1                                  | 1                                  | yes   |
| budget=8 hash               | `0x0619d4d1900c2365`               | `0x0619d4d1900c2365`               | yes   |
| budget=64 hash              | `0x88a4dc75a31d4325`               | `0x88a4dc75a31d4325`               | yes   |
| budget=256 hash             | `0x8a1a3bd01360aada`               | `0x8a1a3bd01360aada`               | yes   |
| budget=8 done_iter set      | `{7}`                              | `{7}`                              | yes   |
| budget=64 done_iter set     | `{63}`                             | `{63}`                             | yes   |
| budget=256 done_iter set    | `{255}`                            | `{255}`                            | yes   |
| budget=8 pos_max set        | `{12}`                             | `{12}`                             | yes   |
| budget=64 pos_max set       | `{68}`                             | `{68}`                             | yes   |
| budget=256 pos_max set      | `{260}`                            | `{260}`                            | yes   |
| residual KV                 | all 99 seqs `(-1, -1)`             | all 99 seqs `(-1, -1)`             | yes   |
| llama_decode failures       | 0                                  | 0                                  | yes   |
| HPX `engine_task_count`     | n/a                                | 1                                  | n/a   |
| HPX `futures_created`       | n/a                                | 99                                 | n/a   |
| HPX `promises_fulfilled`    | n/a                                | 99                                 | n/a   |
| HPX `futures_completed`     | n/a                                | 99                                 | n/a   |

All compared correctness fields match. The HPX-only future/promise
gates also pass on the HPX side.

### Trace gating

`LLAMA_HPX_CB_TRACE` was unset for both Slice-5 runs.
`grep -c 'event=' local/hpx_cb_slice5_hpx.stderr` returns `0`,
confirming the gate stayed off and no trace events leaked into the
reference comparison.

### Verdict

`HPX_CB_PROTO: PASS` — same-shape correctness equivalence between
the pure llama.cpp reference and the HPX prototype on the
99-seq / `{8,64,256}` decode-budget shape.

This is correctness equivalence only. No performance, throughput,
latency, or scaling claim is made.

Captures:

```text
local/hpx_cb_slice5_reference.stdout
local/hpx_cb_slice5_reference.stderr
local/hpx_cb_slice5_hpx.stdout
local/hpx_cb_slice5_hpx.stderr
```

---

## Cooperative cancellation results

This section closes out the cooperative-cancellation line on the
HPX continuous-batching prototype. **Correctness/lifecycle
evidence only — not a performance claim.** Any "saved" or
"after-cancel" field below is descriptive.

### Slice-by-slice summary

- **Cancel Slice 1 — data model only.** Added `request_status`
  enum (`completed` / `cancelled` / `failed_reserved`); cancel
  fields on `seq_state` (`std::atomic<bool> cancel_requested`,
  `cancel_after_decoded_tokens`, `cancel_observed`,
  `cancel_observed_iter`, `n_decoded_at_cancel`); cancel fields
  on `request_result` (`status`, `cancel_observed_iter`,
  `n_decoded_at_cancel`); CLI `--cancel-plan <csv>` (default
  `1,4,7,2,5,8`) and `--cancel-after <int>` (default `16`). Plan
  was parsed/printed and asserted against the round-robin
  budget mapping but was **not** propagated into the engine; no
  cancellation behavior. All Slice-3/4/5 gates remained green.
  Final emit was `HPX_CB_CANCEL_STEP1: PASS`.
- **Cancel Slice 2 — deterministic cancellation observed at
  iteration boundaries.** The plan is propagated into
  `seq_state` at engine construction. The engine task observes
  cancellation only at iteration boundaries — once post-prefill,
  and once at the top of each decode iteration **before**
  building the active row set. No `llama_decode` call is
  interrupted. On observation the engine records
  `cancel_observed`, `cancel_observed_iter`,
  `n_decoded_at_cancel`; runs the existing KV-clear +
  cross-talk check; and fulfills the request promise once with
  `status = cancelled`. Cancelled seqs are not added to any
  later decode batch (they become `seq.done = true` via the
  shared clear path), so `wasted_decode_rows_after_cancel` is
  structurally `0`. The `--cancel-plan` parser was generalized
  to accept `seq_id 0` while still rejecting negative seq ids.
  Final emit was `HPX_CB_CANCEL_STEP2: PASS`.
- **Cancel Slice 3 — status path and trace events.** No
  behavior change. Cancel-family trace events were aligned to a
  spec-friendly format
  (`cancel_requested seq=<id> budget=<int> cancel_after=<int>`,
  `cancel_observed seq=<id> iter=<int> n_decoded=<int>`,
  `cancel_kv_cleared seq=<id> pos_max_at_clear=<int> cross_talk_ok=1`,
  `cancel_future_fulfilled seq=<id> status=cancelled ttc_us=<int>`).
  `seq_complete`, `kv_cleared`, and `promise_fulfilled` fire
  only on the completion path; the cancellation path emits its
  own `cancel_*` events. Main prints an explicit per-iter
  `status_summary: completed=<> cancelled=<> total=<>` line and
  gates `n_decoded == n_decoded_at_cancel` on every cancelled
  result. Final emit was `HPX_CB_CANCEL_STEP3: PASS`.
- **Cancel Slice 4 — metrics/report closeout.** No new
  behavior. The per-iter metrics block now reports
  completed/cancelled splits correctly: `completed[budget=B]`
  counts only `status == completed` (the legacy form printed
  every seq of that budget), with a paired `cancelled[budget=B]`
  line for budgets with any cancelled seqs. The single
  `ttc_ms[budget=B]` field is split into
  `ttc_ms_completed[budget=B]` and (when present)
  `ttc_ms_cancelled[budget=B]`. Top-level metrics added:
  `completed_count`, `cancelled_count`, `decode_failures`,
  `wasted_decode_rows_after_cancel`, `residual_kv_empty`. Final
  emit is `HPX_CB_CANCEL_STEP4: PASS`.

### Default cancellation plan

```text
seqs {1, 4, 7}: budget 64
seqs {2, 5, 8}: budget 256
cancel_after_decoded_tokens = 16
budget-8 seqs are NOT cancelled in the smoke shape
```

### Final cancellation outcome

Captures: `local/hpx_cb_cancel_step4.stdout` (compact) and
`local/hpx_cb_cancel_step4_repeat2.stdout` (`--repeat 2`). Both
final stdout lines: `HPX_CB_CANCEL_STEP4: PASS`. Both stderr
captures contain zero `event=` lines (trace stays off when
`LLAMA_HPX_CB_TRACE` is unset) and no `warn|error|fail|panic|threw`
matches.

| Field                                | Value                  |
|--------------------------------------|------------------------|
| `completed_count`                    | 93                     |
| `cancelled_count`                    | 6                      |
| budget=8 completed / cancelled       | 33 / 0                 |
| budget=64 completed / cancelled      | 30 / 3                 |
| budget=256 completed / cancelled     | 30 / 3                 |
| `cancel_observed_iter_set`           | `{16}`                 |
| `n_decoded_at_cancel_set`            | `{16}`                 |
| `wasted_decode_rows_after_cancel`    | 0                      |
| `residual_kv_empty`                  | true                   |
| `decode_failures`                    | 0                      |
| `engine_task_count`                  | 1                      |
| `futures_created`                    | 99                     |
| `promises_fulfilled`                 | 99                     |
| `futures_completed`                  | 99                     |
| `--repeat 2` determinism             | matches iter 0         |
| budget=8 completed hash              | `0x0619d4d1900c2365`   |
| budget=64 completed unique hashes    | 1                      |
| budget=256 completed unique hashes   | 1                      |

The budget-8 hash is the cross-shape canonical anchor (budget-8
seqs are not in the cancel plan, so the shape is unchanged for
that class). Budget-64 / budget-256 *cancelled-run* hashes are
not directly comparable to the all-99-running Slice-5 hashes
because the batch shape changes after iter 16 once the 6
cancelled seqs leave; only the within-class uniqueness gate
(one unique hash among completed seqs of that budget) is
asserted.

### Trace event counts

Captured under `LLAMA_HPX_CB_TRACE=1` from
`local/hpx_cb_cancel_step3_trace.stderr` (the trace event
format and engine code paths are unchanged in Cancel Slice 4,
so this capture is reused as the trace-counts evidence):

```text
engine_start              =     1
engine_stop               =     1
request_admitted          =    99
seq_prefilled             =    99
decode_row                =  9861
seq_complete              =    93
kv_cleared                =    93
promise_fulfilled         =    93
cancel_requested          =     6
cancel_observed           =     6
cancel_kv_cleared         =     6
cancel_future_fulfilled   =     6
```

Order in the trace stream confirms `cancel_kv_cleared` precedes
`cancel_future_fulfilled` for every cancelled seq, i.e. the
cancelled future is fulfilled only after engine-owned KV
cleanup and the cross-talk check have succeeded.

### Captures

```text
local/hpx_cb_cancel_step1.stdout
local/hpx_cb_cancel_step1.stderr
local/hpx_cb_cancel_step1_repeat2.stdout
local/hpx_cb_cancel_step1_repeat2.stderr
local/hpx_cb_cancel_step2.stdout
local/hpx_cb_cancel_step2.stderr
local/hpx_cb_cancel_step2_repeat2.stdout
local/hpx_cb_cancel_step2_repeat2.stderr
local/hpx_cb_cancel_step3.stdout
local/hpx_cb_cancel_step3.stderr
local/hpx_cb_cancel_step3_repeat2.stdout
local/hpx_cb_cancel_step3_repeat2.stderr
local/hpx_cb_cancel_step3_trace.stdout
local/hpx_cb_cancel_step3_trace.stderr
local/hpx_cb_cancel_step4.stdout
local/hpx_cb_cancel_step4.stderr
local/hpx_cb_cancel_step4_repeat2.stdout
local/hpx_cb_cancel_step4_repeat2.stderr
```

### Interpretation

Cooperative cancellation is now a real HPX serving capability
in this prototype. It is **not** a performance claim. It proves
that:

- request futures can complete with `status = cancelled` after
  engine-owned KV cleanup and the cross-talk-against-still-active-
  siblings check, while non-cancelled sequences continue and
  reach their full decode budgets;
- cancellation is observed strictly at iteration boundaries
  (never inside `llama_decode`), so the single-owner
  `llama_context` / `llama_batch` discipline is preserved;
- the cancellation plan is fully deterministic on the smoke
  shape (`cancel_observed_iter = 16`, `n_decoded_at_cancel = 16`
  for every cancelled seq) and survives `--repeat 2`;
- residual KV at end of run is empty for all 99 seqs across
  both completed and cancelled paths;
- `wasted_decode_rows_after_cancel = 0` — no decode row was ever
  added for a seq after it was observed as cancelled.

The `ttc_ms_cancelled[budget=*]` numbers reflect the wall time
at which the cancelled futures were fulfilled (driven by the
shared engine loop reaching iter 16, not by any per-seq
optimization) and are descriptive only.

---

## Live Admission Slice 1 results — data model only

Slice 1 extends the HPX serving-gate result schema with
live-admission metadata while preserving baseline Metal
execution, canonical output hashes, cancel-gate behavior, and
repeat-run determinism. The current HPX request orchestrator is
building the outer serving/runtime layer first: request
identity, sequence identity, admission metadata, cancellation
accounting, and deterministic result routing. It still
delegates model execution to upstream llama.cpp.

Implementation reference:
[`docs/hpx/continuous_batching_live_admission_design.md`](../../docs/hpx/continuous_batching_live_admission_design.md)
(§9 Slice 1 — data model only, no behavior change).

### Build configuration

A separate Metal-enabled build directory was used so the
recorded canonical hashes (which were captured against a
Metal-offloaded run) remain reproducible. The CPU-only build
directory at `/Users/Ashk/Desktop/HPX/builds/llama-hpx-hpx-on`
was left untouched.

```sh
cmake -S /Users/Ashk/Desktop/HPX/llama-hpx \
      -B /Users/Ashk/Desktop/HPX/builds/llama-hpx-hpx-on-metal \
      -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON \
      -DGGML_ACCELERATE=ON -DGGML_LLAMAFILE=ON -DGGML_NATIVE=ON \
      -DLLAMA_BUILD_TOOLS=ON \
      -DLLAMA_BUILD_HPX_CONTINUOUS_BATCH_GATE=ON \
      -DLLAMA_BUILD_SERVING_BENCH=ON \
      -DLLAMA_BUILD_MULTISEQ_GATE=OFF \
      -DLLAMA_CURL=OFF \
      -DHPX_DIR=/Users/Ashk/Desktop/HPX/hpx-install/lib/cmake/HPX
```

CMakeCache assertions:

```
GGML_METAL:BOOL=ON
LLAMA_BUILD_HPX_CONTINUOUS_BATCH_GATE:BOOL=ON
HPX_DIR:UNINITIALIZED=/Users/Ashk/Desktop/HPX/hpx-install/lib/cmake/HPX
```

Exact target built:

```sh
cmake --build /Users/Ashk/Desktop/HPX/builds/llama-hpx-hpx-on-metal \
      --target llama-hpx-continuous-batch-gate
```

Backend confirmation (every run's stderr):

```
ggml_metal_device_init: GPU name:   MTL0 (Apple M4)
ggml_metal_device_init: GPU family: MTLGPUFamilyApple9 (1009)
load_tensors: offloading 23/23 layers to GPU
load_tensors:  MTL0_Mapped model buffer size = 636.18 MiB
llama_kv_cache:       MTL0 KV buffer size  = 1089.00 MiB
```

No `CPU_REPACK` lines appear; offload is full-model.

### Run commands

```sh
# Clean-HEAD baseline against the Metal build (Cancel Slice 4
# regression check — the gate at this commit emits the
# HPX_CB_CANCEL_STEP4 label when the cpp WIP is not in the tree).
/Users/Ashk/Desktop/HPX/builds/llama-hpx-hpx-on-metal/bin/llama-hpx-continuous-batch-gate \
  --model /Users/Ashk/Desktop/HPX/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --n-seqs 99 --decode-budget-mix 8,64,256 \
  --ctx-size 32768 --n-batch 1024 --n-threads 2 \
  > local/hpx_cb_cancel_step4_metal_repro.stdout \
  2> local/hpx_cb_cancel_step4_metal_repro.stderr

# Slice 1 compact (cpp WIP restored; binary now emits
# HPX_CB_ADMIT_STEP1).
/Users/Ashk/Desktop/HPX/builds/llama-hpx-hpx-on-metal/bin/llama-hpx-continuous-batch-gate \
  --model /Users/Ashk/Desktop/HPX/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --n-seqs 99 --decode-budget-mix 8,64,256 \
  --ctx-size 32768 --n-batch 1024 --n-threads 2 \
  > local/hpx_cb_admit_step1_metal.stdout \
  2> local/hpx_cb_admit_step1_metal.stderr

# Slice 1 --repeat 2.
/Users/Ashk/Desktop/HPX/builds/llama-hpx-hpx-on-metal/bin/llama-hpx-continuous-batch-gate \
  --model /Users/Ashk/Desktop/HPX/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --n-seqs 99 --decode-budget-mix 8,64,256 \
  --ctx-size 32768 --n-batch 1024 --n-threads 2 --repeat 2 \
  > local/hpx_cb_admit_step1_metal_repeat2.stdout \
  2> local/hpx_cb_admit_step1_metal_repeat2.stderr
```

`LLAMA_HPX_CB_TRACE` was unset for every run.

### Final stdout lines

| Run                       | Final line                       |
|---------------------------|----------------------------------|
| Clean Metal baseline      | `HPX_CB_CANCEL_STEP4: PASS`      |
| Slice 1 compact           | `HPX_CB_ADMIT_STEP1: PASS`       |
| Slice 1 `--repeat 2`      | `HPX_CB_ADMIT_STEP1: PASS`       |

### Canonical hashes (preserved against Cancel Slice 4)

The same-shape Slice-5 / Cancel-Slice-4 anchors reproduce
exactly on this Metal build, both with and without the Slice 1
WIP applied:

| Budget | Count | Unique hashes (completed) | Canonical hash         | `done_iter` set | `pos_max_at_clear` set |
|--------|-------|---------------------------|------------------------|-----------------|------------------------|
| 8      | 33    | 1                         | `0x0619d4d1900c2365`   | `{7}`           | `{12}`                 |
| 64     | 30 / 3 (completed / cancelled) | 1            | `0x88a4dc75a31d4325`   | `{63}`          | `{68}`                 |
| 256    | 30 / 3 (completed / cancelled) | 1            | `0x8a1a3bd01360aada`   | `{255}`         | `{260}`                |

Cancellation invariants from Cancel Slice 4 also hold:
`completed_count = 93`, `cancelled_count = 6`,
`cancel_observed_iter_set = {16}`,
`n_decoded_at_cancel_set = {16}`,
`wasted_decode_rows_after_cancel = 0`,
`residual_kv_empty = true`, `decode_failures = 0`.

### Slice 1 field-gate evidence

The compact run's per-iter audit line:

```
iter[0] admit_step1: all 99 results carry
   admitted_at_iter=-1 reused_seq_id=-1
   previous_request_id=-1 admission_source=none
   request_id==seq_id; free_due_to_cancel_violations=0
```

The `--repeat 2` run prints the same audit line for both
`iter[0]` and `iter[1]`, plus
`iter[1] determinism: matches iter 0` from the determinism
gate (now extended with the five new admission fields).

| Gate                                                              | Compact | `--repeat 2` |
|-------------------------------------------------------------------|:-------:|:------------:|
| every result `admitted_at_iter == -1`                             | PASS    | PASS         |
| every result `reused_seq_id == -1`                                | PASS    | PASS         |
| every result `previous_request_id == -1`                          | PASS    | PASS         |
| every result `admission_source == none`                           | PASS    | PASS         |
| every result `request_id == seq_id`                               | PASS    | PASS         |
| `free_due_to_cancel` placeholder empty/unused (`free_due_to_cancel_violations == 0`) | PASS | PASS |
| All Cancel Slice 4 gates still pass (engine_task_count, futures_created/promises_fulfilled/futures_completed = 99, residual KV empty, decode_failures = 0, etc.) | PASS | PASS |
| Determinism tuple includes `request_id`, `admitted_at_iter`, `reused_seq_id`, `previous_request_id`, `admission_src` | n/a | PASS |

### Captures

```text
local/hpx_cb_admit_step1_wip.diff
local/hpx_cb_cancel_step4_metal_repro.stdout
local/hpx_cb_cancel_step4_metal_repro.stderr
local/hpx_cb_admit_step1_metal.stdout
local/hpx_cb_admit_step1_metal.stderr
local/hpx_cb_admit_step1_metal_repeat2.stdout
local/hpx_cb_admit_step1_metal_repeat2.stderr
```

`hpx_cb_admit_step1_wip.diff` is the cpp working-tree diff
preserved before the diagnostic stash dance; it matches the
restored working tree byte-for-byte (298 lines).

### Working-tree state at slice closeout

```text
M  tools/hpx-continuous-batch-gate/hpx-continuous-batch-gate.cpp
M  docs/hpx/continuous_batching_live_admission_design.md
M  tools/hpx-continuous-batch-gate/README.md
M  tools/hpx-continuous-batch-gate/results.md
```

Not committed. Out-of-scope edits in the working tree
(`CLAUDE.md` path renames; `.claude/skills/write-handoff/SKILL.md`)
are unrelated to Slice 1 and originate outside this slice's
work.

### Interpretation

Slice 1 plumbs the live-admission data model end-to-end —
engine-internal state, the `request_result` snapshot, the
deterministic output gate, and the placeholder reuse queue —
without changing decode behavior or batch shape. Every value
the model produces is identical to Cancel Slice 4 on the
recorded Metal anchors, every new field carries its
slice-1-default, and the placeholder reuse queue is wired but
provably empty. Slice 3 will activate the queue and bind
admitted requests to its head; Slice 1 only proves that the
schema, the engine guard, the snapshot copy, and the
determinism contract all work without surprising the existing
correctness gates.

---

## Live Admission Slice 2 results — waiting queue, no admission

Slice 2 introduces a construction-time waiting queue that lives
beside the active population. The engine receives a read-only
handle, samples its size at run start and run end, and fails
closed if the size changes. Active decode/cancel behavior is
unchanged; the canonical Cancel-Slice-4 hashes still hold for
the active subset on the Metal build.

Implementation reference:
[`docs/hpx/continuous_batching_live_admission_design.md`](../../docs/hpx/continuous_batching_live_admission_design.md)
(§9 Slice 2 — waiting queue, no admission yet).

### Build configuration

The Metal-enabled build directory established for Slice 1 is
reused (canonical hashes were captured against this build):

- Metal build directory:
  `/Users/Ashk/Desktop/HPX/builds/llama-hpx-hpx-on-metal`

Build command:

```sh
cmake --build /Users/Ashk/Desktop/HPX/builds/llama-hpx-hpx-on-metal \
      --target llama-hpx-continuous-batch-gate
```

Configure flags (unchanged from Slice 1; `GGML_METAL=ON`,
`LLAMA_BUILD_HPX_CONTINUOUS_BATCH_GATE=ON`,
`HPX_DIR=/Users/Ashk/Desktop/HPX/hpx-install/lib/cmake/HPX`).

`LLAMA_HPX_CB_TRACE` was unset for every run (no new trace
events were added by Slice 2).

### Run commands

```sh
# 1. Default-mode regression — no Slice 2 flags.
#    n_active resolves to n_seqs (= 99); n_waiting = 0.
#    Behavior must match Slice 1 except for the relabel and the
#    new header / audit / metric lines (all additive output).
/Users/Ashk/Desktop/HPX/builds/llama-hpx-hpx-on-metal/bin/llama-hpx-continuous-batch-gate \
  --model /Users/Ashk/Desktop/HPX/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --n-seqs 99 --decode-budget-mix 8,64,256 \
  --ctx-size 32768 --n-batch 1024 --n-threads 2 \
  > local/hpx_cb_admit_step2_default.stdout \
  2> local/hpx_cb_admit_step2_default.stderr

# 2. Slice 2 compact smoke — n_active=93, n_waiting=6.
/Users/Ashk/Desktop/HPX/builds/llama-hpx-hpx-on-metal/bin/llama-hpx-continuous-batch-gate \
  --model /Users/Ashk/Desktop/HPX/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --n-seqs 99 --n-active 93 --n-waiting 6 --waiting-budget 64 \
  --decode-budget-mix 8,64,256 \
  --ctx-size 32768 --n-batch 1024 --n-threads 2 \
  > local/hpx_cb_admit_step2.stdout \
  2> local/hpx_cb_admit_step2.stderr

# 3. Slice 2 --repeat 2.
/Users/Ashk/Desktop/HPX/builds/llama-hpx-hpx-on-metal/bin/llama-hpx-continuous-batch-gate \
  --model /Users/Ashk/Desktop/HPX/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --n-seqs 99 --n-active 93 --n-waiting 6 --waiting-budget 64 \
  --decode-budget-mix 8,64,256 \
  --ctx-size 32768 --n-batch 1024 --n-threads 2 --repeat 2 \
  > local/hpx_cb_admit_step2_repeat2.stdout \
  2> local/hpx_cb_admit_step2_repeat2.stderr
```

### Final stdout lines

| Run                          | Final line                       |
|------------------------------|----------------------------------|
| Default-mode regression      | `HPX_CB_ADMIT_STEP2: PASS`       |
| Slice 2 compact              | `HPX_CB_ADMIT_STEP2: PASS`       |
| Slice 2 `--repeat 2`         | `HPX_CB_ADMIT_STEP2: PASS`       |

### Default-mode regression — counts

`--n-active` unset → resolves to `n_seqs = 99`; `--n-waiting`
defaults to `0`. The run is behaviorally identical to Slice 1.

| Field                        | Value                  |
|------------------------------|------------------------|
| `n_active`                   | 99                     |
| `n_waiting`                  | 0                      |
| `completed_count`            | 93                     |
| `cancelled_count`            | 6                      |
| `queued_count`               | 0                      |
| `waiting_queue_size_at_engine_end` | 0                |
| budget=8 unique hash         | `0x0619d4d1900c2365`   |
| budget=64 unique hash        | `0x88a4dc75a31d4325`   |
| budget=256 unique hash       | `0x8a1a3bd01360aada`   |
| `done_iter` sets             | `{7}/{63}/{255}`       |
| `pos_max_at_clear` sets      | `{12}/{68}/{260}`      |
| `residual_kv_empty`          | true (sweep over 99 slots) |
| `decode_failures`            | 0                      |

Canonical hashes match the Cancel-Slice-4 / Slice-1 anchors
exactly; no FP drift from threading the bound population
through the new active/`n_seq_max` split.

### Slice 2 compact smoke — counts

| Field                              | Value |
|------------------------------------|-------|
| `n_active`                         | 93    |
| `n_waiting`                        | 6     |
| `results.size()`                   | 93    |
| `futures_created`                  | 93    |
| `promises_fulfilled`               | 93    |
| `futures_completed`                | 93    |
| `completed_count`                  | 87    |
| `cancelled_count`                  | 6     |
| `queued_count`                     | 6     |
| `waiting_queue_size_at_engine_end` | 6     |

Per-budget split (across the 93-active population):

| Budget | completed | cancelled | unique hash | sample hash             |
|--------|-----------|-----------|-------------|-------------------------|
| 8      | 31        | 0         | 1           | `0x0619d4d1900c2365`    |
| 64     | 28        | 3         | 1           | `0x88a4dc75a31d4325`    |
| 256    | 28        | 3         | 1           | `0x8a1a3bd01360aada`    |

Same canonical hashes as the default mode and the recorded
Cancel Slice 4 anchors — proving that removing the inactive 6
slots from the bound population does not change FP order on
Metal.

### Slice 1 + 2 field-gate evidence

Per-iter audit line:

```
iter[0] admit_step2: all 93 results carry
   admitted_at_iter=-1 reused_seq_id=-1
   previous_request_id=-1 admission_source=none
   request_id==seq_id; free_due_to_cancel_violations=0;
   queued_count=6 waiting_queue_size_at_engine_end=6
```

`--repeat 2` emits the same audit line at `iter[0]` and
`iter[1]`, plus `iter[1] determinism: matches iter 0`.

| Gate                                                                                            | Default | Compact | `--repeat 2` |
|-------------------------------------------------------------------------------------------------|:-------:|:-------:|:------------:|
| every result `request_id == seq_id`                                                             | PASS    | PASS    | PASS         |
| every result `admitted_at_iter == -1`                                                           | PASS    | PASS    | PASS         |
| every result `reused_seq_id == -1`                                                              | PASS    | PASS    | PASS         |
| every result `previous_request_id == -1`                                                        | PASS    | PASS    | PASS         |
| every result `admission_source == none`                                                         | PASS    | PASS    | PASS         |
| `free_due_to_cancel_violations == 0`                                                            | PASS    | PASS    | PASS         |
| `queued_count == n_waiting`                                                                     | 0/0 ✓   | 6/6 ✓   | 6/6 ✓        |
| `waiting_queue_size_at_engine_end == n_waiting` (no consumption)                                | 0/0 ✓   | 6/6 ✓   | 6/6 ✓        |
| `results.size() == n_active`                                                                    | 99 ✓    | 93 ✓    | 93 ✓         |
| `futures_created` / `promises_fulfilled` / `futures_completed == n_active`                      | 99 ✓    | 93 ✓    | 93 ✓         |
| Cancel-Slice-4 carry-over (`cancel_observed_iter_set={16}`, `n_decoded_at_cancel_set={16}`, `wasted_decode_rows_after_cancel=0`, `decode_failures=0`) | PASS | PASS | PASS |

### Residual KV

The engine-side residual-KV-empty sweep now covers the full
`n_seq_max` range, not just the bound population:

```
iter[N] residual_kv: all 99 seqs cleared (pos_min=-1, pos_max=-1)
residual_kv_empty       = true
```

Slots `[93, 99)` are valid llama `seq_id`s that the engine
never wrote to in the Slice 2 smoke; they nonetheless appear in
the sweep with `pos_min == pos_max == -1`. This rules out any
leftover state on the not-yet-bound slots and prepares the
engine for Slice-3 admission, which will bind into exactly
those slots.

### `--repeat 2` determinism

`iter[1] determinism: matches iter 0` — the per-result tuple
`(seq_id, request_id, n_decoded, generated_tokens, hash,
done_iter, pos_max_at_clear, admitted_at_iter, reused_seq_id,
previous_request_id, admission_src)` is byte-identical between
the two repeats. Engine-side queue size samples are also
byte-identical (`queued_count = 6`,
`waiting_queue_size_at_engine_end = 6` in both repeats), and
the cancel-plan + admission-field defaults all match.

### Captures

```text
local/hpx_cb_admit_step2_default.stdout
local/hpx_cb_admit_step2_default.stderr
local/hpx_cb_admit_step2.stdout
local/hpx_cb_admit_step2.stderr
local/hpx_cb_admit_step2_repeat2.stdout
local/hpx_cb_admit_step2_repeat2.stderr
```

### Caveat

Slice 2 does not prove live admission yet. It only proves that
a waiting queue can exist beside the active population without
changing active decode/cancel behavior — the data path is
plumbed, the engine sees the queue but does not consume from
it, and every Slice-1 field default still holds. Slice 3 is
where the engine will pop the FIFO head at the admission
boundary, bind the popped request to a cancel-freed slot, and
turn the queue into a real serving primitive.

## Live Admission Slice 3 results — cancel-freed-slot admission

Live Admission Slice 3 turns the Slice 2 waiting queue into a real
serving primitive. The engine now consumes the FIFO head at each
admission boundary, binds it to a slot freed by cooperative
cancellation, prefills it in a mixed prefill+decode batch, and
fulfills the admitted request's promise after KV clear and
cross-talk verification — exactly like a completed original
active request.

### Build

```sh
cmake --build /Users/Ashk/Desktop/HPX/builds/llama-hpx-hpx-on-metal \
  --target llama-hpx-continuous-batch-gate
```

(Metal-enabled build dir, matching the recorded canonical-hash
baseline. The CPU-only build would not reproduce the
`0x88a4dc75a31d4325` / `0x8a1a3bd01360aada` surviving-active
hashes — a Metal vs CPU/CPU_REPACK divergence noted for the
Cancel Slice 4 evidence and carried over here.)

### Runs

Default-mode regression — same default plan as Cancel Slice 4,
with `--n-active` defaulted to `n_seqs` and `--n-waiting=0` so
no admission can fire:

```sh
/Users/Ashk/Desktop/HPX/builds/llama-hpx-hpx-on-metal/bin/llama-hpx-continuous-batch-gate \
  --model /Users/Ashk/Desktop/HPX/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --n-seqs 99 --decode-budget-mix 8,64,256 \
  --ctx-size 32768 --n-batch 1024 --n-threads 2 \
  > local/hpx_cb_admit_step3_default.stdout \
  2> local/hpx_cb_admit_step3_default.stderr
```

Slice 3 compact smoke — the canonical cancel-freed-slot admission
shape (`n_active=93`, `n_waiting=6`, `waiting_budget=64`, default
cancel plan):

```sh
/Users/Ashk/Desktop/HPX/builds/llama-hpx-hpx-on-metal/bin/llama-hpx-continuous-batch-gate \
  --model /Users/Ashk/Desktop/HPX/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --n-seqs 99 --n-active 93 --n-waiting 6 --waiting-budget 64 \
  --decode-budget-mix 8,64,256 \
  --ctx-size 32768 --n-batch 1024 --n-threads 2 \
  > local/hpx_cb_admit_step3.stdout \
  2> local/hpx_cb_admit_step3.stderr
```

Slice 3 repeat 2 — same shape, two engine iterations, validates
determinism over the extended admission tuple:

```sh
/Users/Ashk/Desktop/HPX/builds/llama-hpx-hpx-on-metal/bin/llama-hpx-continuous-batch-gate \
  --model /Users/Ashk/Desktop/HPX/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --n-seqs 99 --n-active 93 --n-waiting 6 --waiting-budget 64 \
  --decode-budget-mix 8,64,256 \
  --ctx-size 32768 --n-batch 1024 --n-threads 2 --repeat 2 \
  > local/hpx_cb_admit_step3_repeat2.stdout \
  2> local/hpx_cb_admit_step3_repeat2.stderr
```

### Final lines

| Run                     | Final stdout line             |
|-------------------------|-------------------------------|
| Default-mode regression | `HPX_CB_ADMIT_STEP3: PASS`    |
| Slice 3 compact smoke   | `HPX_CB_ADMIT_STEP3: PASS`    |
| Slice 3 repeat 2        | `HPX_CB_ADMIT_STEP3: PASS`    |

### Slice 3 smoke counts

```text
orig completed     = 87
original cancelled = 6
admitted completed = 6
completed total    = 93   (87 original + 6 admitted)
cancelled total    = 6
total results      = 99
queued             = 6
waiting_end        = 0
```

`admit_step3` audit line (from `local/hpx_cb_admit_step3.stdout`):

```text
iter[0] admit_step3: orig completed=87 cancelled=6, admitted=6,
        total_results=99, queued=6 waiting_end=0
iter[0] status_summary: completed=93 cancelled=6 total=99
        (orig_completed=87 admitted_completed=6)
```

### Reused slot set

```text
reused_seq_id_set = {1,2,4,5,7,8}
```

The set equals the cancellation plan in ascending `seq_id`
order. No naturally completed budget-8 slot
(`{0, 3, 6, 9, …, 90}`) is reused — the engine consumes from
`free_due_to_cancel` only, and `free_due_to_cancel` is
populated only by `cancel_and_fulfill` after a successful
cancellation KV clear.

### Admission mapping

The waiting queue is consumed FIFO; cancel-freed slots are
consumed in ascending `seq_id`. With waiting requests
`{93, 94, 95, 96, 97, 98}` and ascending cancel-freed slots
`{1, 2, 4, 5, 7, 8}`:

| waiting request_id | bound to seq_id | previous_request_id |
|--------------------|-----------------|---------------------|
| 93                 | 1               | 1                   |
| 94                 | 2               | 2                   |
| 95                 | 4               | 4                   |
| 96                 | 5               | 5                   |
| 97                 | 7               | 7                   |
| 98                 | 8               | 8                   |

`previous_request_id == reused_seq_id` because the original
active set has `request_id == seq_id` (no admission rebinds
that mapping).

### Partitioned hash table

Hash uniqueness, `done_iter`, and `pos_max_at_clear` are
validated within `(admission_src, decode_budget)` partitions:

| `admission_src` | budget | completed | cancelled | hash                   | `done_iter` | `pos_max_at_clear` |
|-----------------|--------|-----------|-----------|------------------------|-------------|--------------------|
| `none`          | 8      | 31        | 0         | `0x0619d4d1900c2365`   | 7           | 12                 |
| `none`          | 64     | 28        | 3         | `0x88a4dc75a31d4325`   | 63          | 68                 |
| `none`          | 256    | 28        | 3         | `0x8a1a3bd01360aada`   | 255         | 260                |
| `cancel_freed`  | 64     | 6         | 0         | `0x3b15a0474dfe11be`   | 80          | 68                 |

The admitted budget-64 hash `0x3b15a0474dfe11be` differs from
the surviving-active budget-64 hash `0x88a4dc75a31d4325`
because iter 17 is now a mixed prefill+decode batch (36
prefill rows for the 6 admitted seqs + 56 decode rows for the
surviving-active budget-64/256 seqs). Floating-point order is
batch-shape-dependent, so any seq whose KV is updated in iter
17 sees a different micro-result. Within-partition uniqueness
still holds: all 6 admitted budget-64 hashes are identical to
each other, and all 28 surviving-active budget-64 hashes are
identical to each other; the two values are not required to
match.

The budget-8 canonical anchor `0x0619d4d1900c2365` is
preserved because budget-8 slots are never cancelled and never
admitted — their iter-17 batch shape is unchanged from the
cancellation-only run. The surviving-active budget-256 hash
`0x8a1a3bd01360aada` matches the Cancel Slice 4 anchor for the
same reason this prototype's design predicts: surviving
budget-256 seqs ride iter 17 as a single decode row each, so
their per-row composition is identical to the
cancellation-only path… up to the iter-17 mixed batch's
floating-point side effects on co-resident rows. (The match
above is the empirical outcome on this Metal build; it is not
gated as an across-shape anchor and is reported descriptively.)

`done_iter == 80` for the admitted budget-64 partition equals
`admitted_at_iter (17) + decode_budget (64) − 1`, matching the
design's expected anchor exactly. `pos_max_at_clear == 68`
equals `n_prompt (6) + decode_budget (64) − 2`.

### Cancellation gates (still hold)

```text
cancelled budget=64  count=3 cancel_observed_iter_set={16}
                     n_decoded_at_cancel_set={16}
cancelled budget=256 count=3 cancel_observed_iter_set={16}
                     n_decoded_at_cancel_set={16}
```

Cancellation fires at iter 16 for the 6 cancel-plan seqs;
their KV is cleared in the same iter; `free_due_to_cancel`
gains `{1, 2, 4, 5, 7, 8}` after these clears succeed; the
engine snapshots `free_due_to_cancel.size() = 6` at the top
of iter 17 (before iter 17's no-op cancellation pass) and
admits all 6 waiting requests. `wasted_decode_rows_after_cancel`
remains structurally `0` and `decode_failures = 0`.

### Determinism (`--repeat 2`)

```text
iter[1] determinism: matches iter 0
```

The per-result tuple `(seq_id, request_id, n_decoded,
generated_tokens, hash, done_iter, pos_max_at_clear,
admitted_at_iter, reused_seq_id, previous_request_id,
admission_src)` is byte-identical between iter 0 and iter 1.
`admitted_count = 6`, `reused_seq_id_set = {1,2,4,5,7,8}`,
`waiting_queue_size_at_engine_end = 0`, and the
admitted-result mapping all repeat verbatim.

### Residual KV

```text
iter[N] residual_kv: all 99 seqs cleared (pos_min=-1, pos_max=-1)
residual_kv_empty       = true
```

The engine-side sweep covers all 99 `seq_id`s — the 6
cancel-freed slots that were re-used by admitted requests, the
87 still-mapped original active slots, and the
budget-8-completed slots that were never re-used in this
smoke. Every slot ends the run with `pos_min == pos_max ==
-1`.

### Default mode and `free_due_to_cancel` residue

In the default-mode run (`--n-waiting=0`), the cancellation
plan still fires for `{1, 4, 7, 2, 5, 8}` and pushes those
`seq_id`s into `free_due_to_cancel`. With no waiting requests
to consume them, the deque holds `cancel_plan.size()` entries
at end of run. **This is expected in Slice 3 and not a gate
violation:** the Slice 1 / Slice 2 "must remain empty" guard
is retired, and `run_body()` resets `free_due_to_cancel`,
`waiting_queue_consumable_`, and `admitted_futures_` at the
start of every repeat so residue never leaks between repeats.

Default-mode partitioned table (no admitted partition exists
because `admitted_count = 0`):

```text
iter[0] partition src=none budget=8   completed=33 cancelled=0
        hash=0x0619d4d1900c2365 done_iter_set={7}   pos_max_at_clear_set={12}
iter[0] partition src=none budget=64  completed=30 cancelled=3
        hash=0x88a4dc75a31d4325 done_iter_set={63}  pos_max_at_clear_set={68}
iter[0] partition src=none budget=256 completed=30 cancelled=3
        hash=0x8a1a3bd01360aada done_iter_set={255} pos_max_at_clear_set={260}
iter[0] admit_step3: orig completed=93 cancelled=6, admitted=0,
        total_results=99, queued=0 waiting_end=0
```

### Captures

```text
local/hpx_cb_admit_step3_default.stdout
local/hpx_cb_admit_step3_default.stderr
local/hpx_cb_admit_step3.stdout
local/hpx_cb_admit_step3.stderr
local/hpx_cb_admit_step3_repeat2.stdout
local/hpx_cb_admit_step3_repeat2.stderr
```

### Caveat

Slice 3 is correctness-only. No new trace event names are
emitted (`request_queued`, `request_admitted_live`,
`seq_reused`, `admitted_prefilled`, `admitted_decode_row`,
`admitted_complete`) — the existing `decode_row`,
`seq_complete`, `kv_cleared`, `promise_fulfilled` lifecycle
events fire for admitted seqs through the same paths. The
admission-specific traces and the extended descriptive
metrics (`admission_iter_set`, `reused_seq_id_count`,
`admitted_ttc_ms[budget=*]`,
`waiting_queue_depth_per_iter`) are deferred to Slice 4.

## Live Admission Slice 4 results — traces and metrics closeout

Live Admission Slice 4 wires the admission-specific trace events
and descriptive metrics promised by the live-admission design
(§7, §8). **Slice 4 did not change admission behavior; it only
added observability.** Every Slice 3 anchor (87 orig completed,
6 cancelled, 6 admitted, total 99, residual KV empty,
partitioned hashes including the admitted-budget-64 anchor
`0x3b15a0474dfe11be`) reproduces exactly. Trace-off captures
remain byte-quiet: zero `[hpx-cb-gate] event=` lines on stderr
in default, compact, and repeat-2 trace-off runs.

### Build

```sh
cmake --build /Users/Ashk/Desktop/HPX/builds/llama-hpx-hpx-on-metal \
  --target llama-hpx-continuous-batch-gate
```

### Runs

Default-mode regression (trace off):

```sh
/Users/Ashk/Desktop/HPX/builds/llama-hpx-hpx-on-metal/bin/llama-hpx-continuous-batch-gate \
  --model /Users/Ashk/Desktop/HPX/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --n-seqs 99 --decode-budget-mix 8,64,256 \
  --ctx-size 32768 --n-batch 1024 --n-threads 2 \
  > local/hpx_cb_admit_step4_default.stdout \
  2> local/hpx_cb_admit_step4_default.stderr
```

Compact smoke (trace off):

```sh
/Users/Ashk/Desktop/HPX/builds/llama-hpx-hpx-on-metal/bin/llama-hpx-continuous-batch-gate \
  --model /Users/Ashk/Desktop/HPX/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --n-seqs 99 --n-active 93 --n-waiting 6 --waiting-budget 64 \
  --decode-budget-mix 8,64,256 \
  --ctx-size 32768 --n-batch 1024 --n-threads 2 \
  > local/hpx_cb_admit_step4.stdout \
  2> local/hpx_cb_admit_step4.stderr
```

Compact smoke (`LLAMA_HPX_CB_TRACE=1`):

```sh
LLAMA_HPX_CB_TRACE=1 \
/Users/Ashk/Desktop/HPX/builds/llama-hpx-hpx-on-metal/bin/llama-hpx-continuous-batch-gate \
  --model /Users/Ashk/Desktop/HPX/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --n-seqs 99 --n-active 93 --n-waiting 6 --waiting-budget 64 \
  --decode-budget-mix 8,64,256 \
  --ctx-size 32768 --n-batch 1024 --n-threads 2 \
  > local/hpx_cb_admit_step4_trace.stdout \
  2> local/hpx_cb_admit_step4_trace.stderr
```

Repeat 2 (trace off):

```sh
/Users/Ashk/Desktop/HPX/builds/llama-hpx-hpx-on-metal/bin/llama-hpx-continuous-batch-gate \
  --model /Users/Ashk/Desktop/HPX/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --n-seqs 99 --n-active 93 --n-waiting 6 --waiting-budget 64 \
  --decode-budget-mix 8,64,256 \
  --ctx-size 32768 --n-batch 1024 --n-threads 2 --repeat 2 \
  > local/hpx_cb_admit_step4_repeat2.stdout \
  2> local/hpx_cb_admit_step4_repeat2.stderr
```

### Final lines

| Run                       | Final stdout line             |
|---------------------------|-------------------------------|
| Default-mode trace-off    | `HPX_CB_ADMIT_STEP4: PASS`    |
| Compact smoke trace-off   | `HPX_CB_ADMIT_STEP4: PASS`    |
| Compact smoke trace-on    | `HPX_CB_ADMIT_STEP4: PASS`    |
| Repeat 2 trace-off        | `HPX_CB_ADMIT_STEP4: PASS`    |

### Trace-off quietness

```text
$ grep -c "[hpx-cb-gate] event=" local/hpx_cb_admit_step4_default.stderr
0
$ grep -c "[hpx-cb-gate] event=" local/hpx_cb_admit_step4.stderr
0
$ grep -c "[hpx-cb-gate] event=" local/hpx_cb_admit_step4_repeat2.stderr
0
```

All three trace-off captures emit zero `[hpx-cb-gate] event=`
lines on stderr (the 145 lines per stderr are the standard
llama.cpp / Metal init noise). Trace gating is preserved from
prior slices.

### Trace event counts (compact, `LLAMA_HPX_CB_TRACE=1`)

| Event                        | Expected | Observed | Note         |
|------------------------------|----------|----------|--------------|
| `engine_start`               | 1        | 1        |              |
| `engine_stop`                | 1        | 1        |              |
| `request_admitted`           | 93       | 93       | original active set |
| `request_queued`             | 6        | 6        | main, pre-engine |
| `request_admitted_live`      | 6        | 6        | admission boundary |
| `seq_reused`                 | 6        | 6        | admission boundary |
| `seq_prefilled`              | 93       | 93       | original active prefill |
| `admitted_prefilled`         | 6        | 6        | iter-17 post-decode argmax |
| `decode_row`                 | derived  | 9589     | descriptive (high-cardinality) |
| `admitted_decode_row`        | 378      | 378      | 6 admitted × 63 decode iters |
| `seq_complete`               | 93       | 93       | every completion path |
| `admitted_complete`          | 6        | 6        | admitted completion path only |
| `kv_cleared`                 | 93       | 93       | every successful KV clear |
| `promise_fulfilled`          | 93       | 93       | every completion fulfillment |
| `cancel_requested`           | 6        | 6        | engine-start cancel-plan emit |
| `cancel_observed`            | 6        | 6        | iter-16 cancel pass |
| `cancel_kv_cleared`          | 6        | 6        | cancel path |
| `cancel_future_fulfilled`    | 6        | 6        | cancel path |

### Trace event examples

`seq_reused` carries the prior owner (captured before
`rseq.request_id` is overwritten) and the new owner:

```text
[hpx-cb-gate] event=seq_reused seq_id=1 previous_owner=1 new_owner=93 iter=17
```

`admitted_complete` mirrors `seq_complete` and carries the
budget / done_iter / hash — the hash matches the admitted
budget-64 anchor `0x3b15a0474dfe11be`:

```text
[hpx-cb-gate] event=admitted_complete request=93 seq_id=1 budget=64 done_iter=80 hash=0x3b15a0474dfe11be
```

### Structural gate (independent of trace flag)

`admitted_prefill_events == admitted_count` per repeat. The
counter is incremented next to the `admitted_prefilled` trace
site, gated on a predicate captured BEFORE `n_decoded` is
mutated, so it fires exactly once per admitted request even
when trace is disabled:

| Run               | admitted_prefill_events | admitted_count | OK |
|-------------------|-------------------------|----------------|----|
| Default trace-off | 0                       | 0              | ✓  |
| Compact trace-off | 6                       | 6              | ✓  |
| Repeat 2 iter[0]  | 6                       | 6              | ✓  |
| Repeat 2 iter[1]  | 6                       | 6              | ✓  |

### Slice 4 metric lines (compact smoke)

```text
admitted_count          = 6
reused_seq_id_set       = {1,2,4,5,7,8}
reused_seq_id_count     = 6
admitted_prefill_events = 6
admission_iter_set      = {17}
waiting_queue_depth_after_admission_per_iter p50=0 p95=6 max=6 (samples=255)
admitted_ttc_ms[budget=64]  mean=29610.24 p95=29610.43
```

### Queue-depth metric explanation

`waiting_queue_depth_after_admission_per_iter` is sampled
**after** each decode iter's admission loop completes. One
sample per decode iter (255 samples for the smoke shape, equal
to `update_iterations`):

- iters 1..16: depth = 6 — admission has not yet had a freed
  slot to consume (cancellation fires at iter 16's top, but the
  freeze-count snapshot delays admission by one iter).
- iter 17: depth = 0 — admission consumed all 6 waiters this
  iter.
- iters 18..255: depth = 0 — queue stays empty.

That gives `max = 6` (the 16 pre-admission iters), `p50 = 0`
(most iters are post-admission), and `p95 = 6` (16/255 ≈ 6.3%
of samples are 6, fitting the upper tail). The "after
admission" sampling point is the design's intent: at iter 17,
the engine consumed the queue, so the engine-end depth at iter
17 is `0`, not `6`.

### Repeat determinism

```text
iter[1] determinism: matches iter 0
```

The per-result tuple `(seq_id, request_id, n_decoded,
generated_tokens, hash, done_iter, pos_max_at_clear,
admitted_at_iter, reused_seq_id, previous_request_id,
admission_src)` is byte-identical between iter 0 and iter 1.
Every Slice 4 metric is also stable across repeats:
`admitted_count = 6`, `reused_seq_id_set = {1,2,4,5,7,8}`,
`reused_seq_id_count = 6`, `admitted_prefill_events = 6`,
`admission_iter_set = {17}`,
`waiting_queue_depth_after_admission_per_iter p50=0 p95=6 max=6
(samples=255)` — all observed verbatim in both repeats.

### Captures

```text
local/hpx_cb_admit_step4_default.stdout
local/hpx_cb_admit_step4_default.stderr
local/hpx_cb_admit_step4.stdout
local/hpx_cb_admit_step4.stderr
local/hpx_cb_admit_step4_trace.stdout
local/hpx_cb_admit_step4_trace.stderr
local/hpx_cb_admit_step4_repeat2.stdout
local/hpx_cb_admit_step4_repeat2.stderr
```

### Caveat

Slice 4 is observability-only. No admission-behavior change, no
new CLI flags, no scheduling-policy change, no batch-shape
change. The Slice 3 admission flow is intact end-to-end; the
new traces and metrics are descriptive surfaces over the same
underlying engine state.

---

## Live Admission Slice 5 results — completion-freed slot reuse

Slice 5 adds completion-freed-slot admission on top of the
Slice 3 cancel-freed-slot path. A new default-OFF CLI flag
`--reuse-completed` enables the second admission source; a
new engine deque `free_due_to_completion` is filled under a
**demand gate** (push only while the waiting queue still has
unadmitted entries); and admission source priority is
`cancel_freed` → `completion_freed`. Slice 3 cancel-freed
behavior is preserved semantically when `--reuse-completed`
is OFF. Slice 5 does **not** test combined cancel + completion
source admission in one run — that is a future mixed-source
slice and is out of scope here.

Implementation reference:
[`docs/hpx/continuous_batching_live_admission_design.md`](../../docs/hpx/continuous_batching_live_admission_design.md)
(§9 Slice 5 — completion-freed-slot admission).

### Build

Same Metal-enabled build as Slice 1–4:

```sh
cmake --build /Users/Ashk/Desktop/HPX/builds/llama-hpx-hpx-on-metal \
  --target llama-hpx-continuous-batch-gate
```

Result: clean rebuild, `[2/2] Linking CXX executable
bin/llama-hpx-continuous-batch-gate`. The binary embeds
`HPX_CB_ADMIT_STEP5`, `completion_freed`, and
`--reuse-completed` strings (verified via `strings`).

### Run commands

1. Default regression (no `--n-active/--n-waiting`, no
   `--reuse-completed`, default cancel plan):

   ```bash
   /Users/Ashk/Desktop/HPX/builds/llama-hpx-hpx-on-metal/bin/llama-hpx-continuous-batch-gate \
     --model /Users/Ashk/Desktop/HPX/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
     --n-seqs 99 --decode-budget-mix 8,64,256 \
     --ctx-size 32768 --n-batch 1024 --n-threads 2 \
     > local/hpx_cb_admit_step5_default.stdout \
     2> local/hpx_cb_admit_step5_default.stderr
   ```

2. Slice 3 semantic regression with `--reuse-completed` OFF
   (compact shape, default cancel plan):

   ```bash
   /Users/Ashk/Desktop/HPX/builds/llama-hpx-hpx-on-metal/bin/llama-hpx-continuous-batch-gate \
     --model /Users/Ashk/Desktop/HPX/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
     --n-seqs 99 --n-active 93 --n-waiting 6 --waiting-budget 64 \
     --decode-budget-mix 8,64,256 \
     --ctx-size 32768 --n-batch 1024 --n-threads 2 \
     > local/hpx_cb_admit_step5_slice3_regression.stdout \
     2> local/hpx_cb_admit_step5_slice3_regression.stderr
   ```

3. Slice 5 compact smoke (completion-freed, explicit empty
   cancel plan):

   ```bash
   /Users/Ashk/Desktop/HPX/builds/llama-hpx-hpx-on-metal/bin/llama-hpx-continuous-batch-gate \
     --model /Users/Ashk/Desktop/HPX/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
     --n-seqs 99 --n-active 90 --n-waiting 9 --waiting-budget 8 \
     --decode-budget-mix 8,64,256 \
     --cancel-plan none --reuse-completed \
     --ctx-size 32768 --n-batch 1024 --n-threads 2 \
     > local/hpx_cb_admit_step5.stdout \
     2> local/hpx_cb_admit_step5.stderr
   ```

4. Slice 5 trace-on compact:

   ```bash
   LLAMA_HPX_CB_TRACE=1 \
   /Users/Ashk/Desktop/HPX/builds/llama-hpx-hpx-on-metal/bin/llama-hpx-continuous-batch-gate \
     --model /Users/Ashk/Desktop/HPX/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
     --n-seqs 99 --n-active 90 --n-waiting 9 --waiting-budget 8 \
     --decode-budget-mix 8,64,256 \
     --cancel-plan none --reuse-completed \
     --ctx-size 32768 --n-batch 1024 --n-threads 2 \
     > local/hpx_cb_admit_step5_trace.stdout \
     2> local/hpx_cb_admit_step5_trace.stderr
   ```

5. Slice 5 repeat 2:

   ```bash
   /Users/Ashk/Desktop/HPX/builds/llama-hpx-hpx-on-metal/bin/llama-hpx-continuous-batch-gate \
     --model /Users/Ashk/Desktop/HPX/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
     --n-seqs 99 --n-active 90 --n-waiting 9 --waiting-budget 8 \
     --decode-budget-mix 8,64,256 \
     --cancel-plan none --reuse-completed \
     --ctx-size 32768 --n-batch 1024 --n-threads 2 --repeat 2 \
     > local/hpx_cb_admit_step5_repeat2.stdout \
     2> local/hpx_cb_admit_step5_repeat2.stderr
   ```

### Final stdout lines

| Run | Final line |
|---|---|
| 1. Default regression | `HPX_CB_ADMIT_STEP5: PASS` |
| 2. Slice 3 semantic regression (`--reuse-completed` OFF) | `HPX_CB_ADMIT_STEP5: PASS` |
| 3. Slice 5 compact smoke | `HPX_CB_ADMIT_STEP5: PASS` |
| 4. Slice 5 trace-on compact | `HPX_CB_ADMIT_STEP5: PASS` |
| 5. Slice 5 repeat 2 | `HPX_CB_ADMIT_STEP5: PASS` |

### Slice 5 smoke evidence (run 3)

```text
iter[0] admit_step5: orig completed=90 cancelled=0, admitted=9
        (cancel_freed=0 completion_freed=9), total_results=99,
        queued=9 waiting_end=0, completion_pool_residual=21
iter[0] residual_kv: all 99 seqs cleared (pos_min=-1, pos_max=-1)
  reused_seq_id_set       = {0,3,6,9,12,15,18,21,24}
  admission_iter_set      = {8}
  completion_freed_pool_size_at_run_end = 21
```

Key counts:

```text
orig completed                          = 90
cancelled                               = 0
admitted                                = 9
total_results                           = 99
queued                                  = 9
waiting_end                             = 0
reused_seq_id_set                       = {0,3,6,9,12,15,18,21,24}
admission_iter_set                      = {8}
completion_freed_pool_size_at_run_end   = 21
residual KV empty (all 99 seq_ids)      = yes
```

### Slice 5 admission mapping (from trace-on, run 4)

| request_id | reused seq_id | admission_source | iter |
|---:|---:|:---|---:|
| 90 | 0  | completion_freed | 8 |
| 91 | 3  | completion_freed | 8 |
| 92 | 6  | completion_freed | 8 |
| 93 | 9  | completion_freed | 8 |
| 94 | 12 | completion_freed | 8 |
| 95 | 15 | completion_freed | 8 |
| 96 | 18 | completion_freed | 8 |
| 97 | 21 | completion_freed | 8 |
| 98 | 24 | completion_freed | 8 |

Every admitted `request_result`: `budget=8`,
`admitted_at_iter=8`, `done_iter=15`,
`pos_max_at_clear=12` (= `n_prompt + budget − 2 = 6 + 8 − 2`),
`admission_src=completion_freed`,
`previous_request_id == reused_seq_id`.

### Slice 5 partition hashes (run 3)

```text
iter[0] partition src=none             budget=8   completed=30
                  unique_completed_hashes=1
                  hash=0x0619d4d1900c2365
                  done_iter_set={7}  pos_max_at_clear_set={12}

iter[0] partition src=completion_freed budget=8   completed=9
                  unique_completed_hashes=1
                  hash=0x0619d4d1900c2365
                  done_iter_set={15} pos_max_at_clear_set={12}

iter[0] partition src=none             budget=64  completed=30
                  unique_completed_hashes=1
                  hash=0x88a4dc75a31d4325
                  done_iter_set={63} pos_max_at_clear_set={68}

iter[0] partition src=none             budget=256 completed=30
                  unique_completed_hashes=1
                  hash=0x8a1a3bd01360aada
                  done_iter_set={255} pos_max_at_clear_set={260}
```

Hash-gating policy applied in this run:

- **Natural budget-8 partition** (`src=none, budget=8`,
  30 results) completes entirely before admission iter 8.
  Its canonical Metal anchor `0x0619d4d1900c2365` is **gated
  strictly**.
- **Surviving budget-64 and budget-256 partitions**
  (`src=none, budget={64,256}`, 30 results each) cross
  iter 8's mixed prefill+decode batch shape. They matched
  canonical anchors `0x88a4dc75a31d4325` and
  `0x8a1a3bd01360aada` in this run, but per the approved
  plan they are **not** gated against canonical — only
  against within-run uniqueness (one hash per partition) and
  `--repeat 2` determinism. The canonical matches are
  recorded here as observed, not required.
- **Admitted `completion_freed` budget-8 partition**
  (9 results) is **descriptive only**: the observed hash
  `0x0619d4d1900c2365` is recorded for reference but is not
  canonical-gated. Within-run uniqueness (single hash across
  the 9 admitted) IS gated.

### Slice 3 semantic regression (run 2)

With `--reuse-completed` OFF and the default cancel plan
`{1,4,7,2,5,8}` / `cancel_after=16`, the Slice 3 / 4
cancel-freed behavior is preserved. The four observed
partition lines:

```text
iter[0] partition src=none         budget=8   completed=31
                  hash=0x0619d4d1900c2365  done_iter_set={7}
                  pos_max_at_clear_set={12}

iter[0] partition src=none         budget=64  completed=28
                  cancelled=3   hash=0x88a4dc75a31d4325
                  done_iter_set={63}  pos_max_at_clear_set={68}

iter[0] partition src=cancel_freed budget=64  completed=6
                  hash=0x3b15a0474dfe11be  done_iter_set={80}
                  pos_max_at_clear_set={68}

iter[0] partition src=none         budget=256 completed=28
                  cancelled=3   hash=0x8a1a3bd01360aada
                  done_iter_set={255} pos_max_at_clear_set={260}
```

The cancel-freed admitted budget-64 hash
`0x3b15a0474dfe11be`, the `reused_seq_id_set =
{1,2,4,5,7,8}`, and `admission_iter_set = {17}` reproduce
exactly. `completion_freed_pool_size_at_run_end = 0` (engine
never touched the pool) and `cancel_freed=6
completion_freed=0` on the audit line. Comparison to Slice 4
is semantic, not byte-for-byte: only the label
(`HPX_CB_ADMIT_STEP4` → `HPX_CB_ADMIT_STEP5`) and the audit
line name (`admit_step3:` → `admit_step5:` with the new
source-split / pool-residual fields appended) differ.

### Trace counts (run 4, trace-on)

```text
engine_start              1
engine_stop               1
request_admitted         90
request_queued            9
request_admitted_live     9
seq_reused                9     payload: admission_source=completion_freed
seq_prefilled            90
admitted_prefilled        9     payload: admission_source=completion_freed
decode_row             9813     descriptive
admitted_decode_row      63     payload: admission_source=completion_freed
seq_complete             99
admitted_complete         9     payload: admission_source=completion_freed
kv_cleared               99
promise_fulfilled        99
cancel_requested          0
cancel_observed           0
cancel_kv_cleared         0
cancel_future_fulfilled   0
```

Notes:

- `seq_complete = 99` (all 99 completions: 90 originals + 9
  admitted) and `admitted_complete = 9` (admitted subset) are
  distinct counters, not redundant — same naming convention as
  `request_admitted = 90` (generic) vs `request_admitted_live
  = 9` (admission-source-aware) from Slice 4.
- `admitted_decode_row = 63` is 9 admitted × 7 post-prefill
  decode rows. The first row of each admitted seq is a prefill
  row (counted by `admitted_prefilled`, not `admitted_decode_row`)
  and the post-decode argmax of iter 8 produces the first
  generated token; the remaining 7 decode iters each emit one
  `admitted_decode_row` per admitted seq.
- Cancel-family event counts are `0` (Slice 5 smoke uses
  `--cancel-plan none`).

### Trace-off quietness

All four trace-off captures emit zero `[hpx-cb-gate] event=`
lines on stderr:

```text
local/hpx_cb_admit_step5_default.stderr            0
local/hpx_cb_admit_step5_slice3_regression.stderr  0
local/hpx_cb_admit_step5.stderr                    0
local/hpx_cb_admit_step5_repeat2.stderr            0
```

### Repeat determinism (run 5)

```text
iter[0] admit_step5: … admitted=9 (cancel_freed=0 completion_freed=9)
                       total_results=99 queued=9 waiting_end=0
                       completion_pool_residual=21
  reused_seq_id_set       = {0,3,6,9,12,15,18,21,24}
  admission_iter_set      = {8}
  completion_freed_pool_size_at_run_end = 21

iter[1] admit_step5: … admitted=9 (cancel_freed=0 completion_freed=9)
                       total_results=99 queued=9 waiting_end=0
                       completion_pool_residual=21
  reused_seq_id_set       = {0,3,6,9,12,15,18,21,24}
  admission_iter_set      = {8}
  completion_freed_pool_size_at_run_end = 21

iter[1] determinism: matches iter 0
```

The per-result tuple `(seq_id, request_id, n_decoded,
generated_tokens, hash, done_iter, pos_max_at_clear,
admitted_at_iter, reused_seq_id, previous_request_id,
admission_src)` is byte-identical across iter 0 and iter 1.
`reused_seq_id_set`, `admission_iter_set`, and
`completion_freed_pool_size_at_run_end` are stable across
repeats.

### Captures

```text
local/hpx_cb_admit_step5_default.stdout
local/hpx_cb_admit_step5_default.stderr
local/hpx_cb_admit_step5_slice3_regression.stdout
local/hpx_cb_admit_step5_slice3_regression.stderr
local/hpx_cb_admit_step5.stdout
local/hpx_cb_admit_step5.stderr
local/hpx_cb_admit_step5_trace.stdout
local/hpx_cb_admit_step5_trace.stderr
local/hpx_cb_admit_step5_repeat2.stdout
local/hpx_cb_admit_step5_repeat2.stderr
```

### Scope

Slice 5 proves completion-freed-slot reuse (the
`waiting → queued → naturally completed slot freed → KV
cleared → seq_id pooled → waiting request admitted →
prefilled in a mixed batch → decoded to completion → future
fulfilled → residual KV empty` path), and it preserves the
Slice 3 cancel-freed-slot reuse semantics with
`--reuse-completed` OFF.

Slice 5 does **not** prove **mixed cancel + completion source
admission in one run** — that is a future mixed-source slice.
Slice 5 deliberately exercises exactly one admission source
per smoke so the failure-mode space remains separable.

---

## Live Admission Slice 6 results — async external arrivals

Closeout for the Live Admission Slice 6 surface. Adds
deterministic async external arrivals on top of the existing
Slice 3 cancel-freed admission path: a single scripted HPX
submitter task pushes external arrivals into an engine-owned
inbox under a release+ack barrier, the engine drains the
inbox at the top of the next iter, and the arrivals admit
through the existing cancel-freed FIFO admission path with
`arrival_source = external` on every snapshot.

### Build

- Clean build, one unused warning, zero errors.
- Build dir: `/Users/Ashk/Desktop/HPX/builds/llama-hpx-hpx-on-metal`.
- Binary:
  `/Users/Ashk/Desktop/HPX/builds/llama-hpx-hpx-on-metal/bin/llama-hpx-continuous-batch-gate`.
- HPX-native correctness check confirmed at build time: the
  inbox lock uses **`hpx::spinlock`** (not
  `hpx::lcos::local::spinlock`, which is not the public alias
  exported by this HPX install). No `std::thread`, no
  `std::condition_variable`, no `std::this_thread::sleep_for`,
  no new `std::mutex`.

### Smoke shape

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
admission source under test = cancel_freed (external arrivals
                              bound to cancel-freed slots)
```

Derived deterministic timing:

```text
release_iter = 8       (engine end-of-iter-8 release set,
                        submitter pushes 6-block, sets ack)
drain_iter   = 9       (engine drains inbox at top of iter 9)
admit_iter   = 17      (cancel_after + 1; 6 cancelled slots
                        freed at iter 16 admit external arrivals)
```

External request IDs: `93, 94, 95, 96, 97, 98`
(`n_active + n_waiting + i` for `i = 0..5`).

### Captures

```text
local/slice6_default.stdout
local/slice6_default.stderr
local/slice6_smoke.stdout
local/slice6_smoke.stderr
local/slice6_trace.stdout
local/slice6_trace.stderr
local/slice6_repeat2.stdout
local/slice6_repeat2.stderr
```

### Run commands

Default regression (no Slice 6 args; verifies the inert path):

```sh
/Users/Ashk/Desktop/HPX/builds/llama-hpx-hpx-on-metal/bin/llama-hpx-continuous-batch-gate \
    --model /Users/Ashk/Desktop/HPX/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
    > local/slice6_default.stdout 2> local/slice6_default.stderr
```

Slice 6 compact smoke:

```sh
/Users/Ashk/Desktop/HPX/builds/llama-hpx-hpx-on-metal/bin/llama-hpx-continuous-batch-gate \
    --model /Users/Ashk/Desktop/HPX/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
    --n-seqs 99 --n-active 93 --n-waiting 0 \
    --n-external-arrivals 6 --external-arrival-budget 64 \
    --external-release-iter 8 \
    --cancel-plan 1,4,7,2,5,8 --cancel-after 16 \
    > local/slice6_smoke.stdout 2> local/slice6_smoke.stderr
```

Slice 6 trace-on compact:

```sh
LLAMA_HPX_CB_TRACE=1 \
/Users/Ashk/Desktop/HPX/builds/llama-hpx-hpx-on-metal/bin/llama-hpx-continuous-batch-gate \
    --model /Users/Ashk/Desktop/HPX/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
    --n-seqs 99 --n-active 93 --n-waiting 0 \
    --n-external-arrivals 6 --external-arrival-budget 64 \
    --external-release-iter 8 \
    --cancel-plan 1,4,7,2,5,8 --cancel-after 16 \
    > local/slice6_trace.stdout 2> local/slice6_trace.stderr
```

Slice 6 repeat 2 (determinism):

```sh
/Users/Ashk/Desktop/HPX/builds/llama-hpx-hpx-on-metal/bin/llama-hpx-continuous-batch-gate \
    --model /Users/Ashk/Desktop/HPX/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
    --n-seqs 99 --n-active 93 --n-waiting 0 \
    --n-external-arrivals 6 --external-arrival-budget 64 \
    --external-release-iter 8 \
    --cancel-plan 1,4,7,2,5,8 --cancel-after 16 --repeat 2 \
    > local/slice6_repeat2.stdout 2> local/slice6_repeat2.stderr
```

### Final-line table

```text
default regression  HPX_CB_ADMIT_STEP6: PASS
compact smoke       HPX_CB_ADMIT_STEP6: PASS
trace-on compact    HPX_CB_ADMIT_STEP6: PASS
repeat 2            HPX_CB_ADMIT_STEP6: PASS
```

### External-arrival gates (smoke + repeat 2; per iter)

```text
arrival_drained_count       = 6
external_admitted_count     = 6
first_external_drain_iter   = 9
iter_release_fired_set      = {8}
submitter_ack_set           = {8}
```

The per-iter `admit_step6:` audit line confirms this directly:

```text
iter[0] admit_step6: external arrivals drained=6 admitted=6 first_drain_iter=9 release_set_size=1 ack_set_size=1
iter[1] admit_step6: external arrivals drained=6 admitted=6 first_drain_iter=9 release_set_size=1 ack_set_size=1
```

### Request → seq mapping (FIFO over sorted cancel-plan)

```text
request 93 -> seq 1
request 94 -> seq 2
request 95 -> seq 4
request 96 -> seq 5
request 97 -> seq 7
request 98 -> seq 8
```

Source for this evidence (from `local/slice6_trace.stderr`):

```text
[hpx-cb-gate] event=request_admitted_live request=93 reused_seq_id=1 iter=17 admission_source=cancel_freed arrival_source=external
[hpx-cb-gate] event=request_admitted_live request=94 reused_seq_id=2 iter=17 admission_source=cancel_freed arrival_source=external
[hpx-cb-gate] event=request_admitted_live request=95 reused_seq_id=4 iter=17 admission_source=cancel_freed arrival_source=external
[hpx-cb-gate] event=request_admitted_live request=96 reused_seq_id=5 iter=17 admission_source=cancel_freed arrival_source=external
[hpx-cb-gate] event=request_admitted_live request=97 reused_seq_id=7 iter=17 admission_source=cancel_freed arrival_source=external
[hpx-cb-gate] event=request_admitted_live request=98 reused_seq_id=8 iter=17 admission_source=cancel_freed arrival_source=external
```

### Per-result snapshot state (every external arrival)

```text
arrival_source       = external
admission_source     = cancel_freed
admitted_at_iter     = 17
admitted budget      = 64
admitted budget-64 hash = 0x3b15a0474dfe11be
```

The hash is the same anchor recorded by Slice 3 / Slice 4 /
Slice 5 for the cancel-freed admitted budget-64 partition,
proving that routing the same payload through the async
external-arrival surface yields a byte-identical completed
hash.

### Trace counts (trace-on smoke; per repeat)

```text
event=request_submitted_external                          = 6
event=arrival_drained                                     = 6
event=iter_release_fired                                  = 1
event=submitter_ack_observed                              = 1
event=request_admitted_live arrival_source=external       = 6
```

### Trace-off quietness

```text
local/slice6_smoke.stderr     [hpx-cb-gate] event=     0
local/slice6_default.stderr   [hpx-cb-gate] event=     0
local/slice6_repeat2.stderr   [hpx-cb-gate] event=     0
```

Default and trace-off runs emit zero `[hpx-cb-gate] event=`
lines; the trace path stays one atomic load + early return
per call site when `LLAMA_HPX_CB_TRACE` is unset.

### Residual / inbox cleanup

```text
residual_kv_empty = true   (default, smoke, trace-on, repeat 2 — both repeats)
```

The engine asserts both `inbox_.empty()` and
`external_promises_.empty()` BEFORE the residual-KV sweep at
the end of `run_body()`; the run fails closed with an
explicit reason if either is non-empty. Across all four
captures both checks pass on every repeat.

### Repeat determinism

```text
iter[0] admit_step6: external arrivals drained=6 admitted=6 first_drain_iter=9 release_set_size=1 ack_set_size=1
iter[1] admit_step6: external arrivals drained=6 admitted=6 first_drain_iter=9 release_set_size=1 ack_set_size=1
iter[1] determinism: matches iter 0
```

The per-result determinism tuple `(seq_id, request_id,
n_decoded, generated_tokens, hash, done_iter,
pos_max_at_clear, admitted_at_iter, reused_seq_id,
previous_request_id, admission_src)` is byte-identical
across iter 0 and iter 1, including for every external
arrival. The new Slice 6 counters
(`arrival_drained_count`, `external_admitted_count`,
`first_external_drain_iter`, `iter_release_fired_set`,
`submitter_ack_set`) are also stable across repeats.

### Inert path (default regression, no Slice 6 flags)

With `--n-external-arrivals` left at its default of 0:

```text
arrival_drained_count       = 0
external_admitted_count     = 0
first_external_drain_iter   = -1
iter_release_fired_set      = {}
submitter_ack_set           = {}
```

No submitter task is spawned, no release/ack handle is
registered, the inbox is never written, and the run is
semantically equivalent to Slice 5 with `--reuse-completed`
OFF. `HPX_CB_ADMIT_STEP6: PASS` confirms the inert-path gate
fires correctly when the external surface is not exercised.

### Scope

Slice 6 proves the `external HPX submitter task -> release/ack
barrier -> engine inbox (hpx::spinlock) -> drain at iter K+1 ->
waiting queue -> existing cancel_freed admission -> external
future fulfilled` path under deterministic (non-wall-clock)
timing, with `arrival_source = external` propagated end-to-end
on every snapshot, and with the engine remaining the sole
owner of `llama_context`, `llama_batch`, `llama_decode`,
`llama_memory_seq_*`, and `llama_get_logits_ith`.

Slice 6 does **not** exercise:

- Multiple distinct release iters in a single run (the smoke
  uses one release barrier at iter 8).
- External arrivals admitted via `completion_freed` (the
  smoke shape's `--reuse-completed` is OFF; this is left for
  a later mixed-source slice).
- Wall-clock arrival schedules / streaming.
- More than one engine task per process (the single-engine-
  task invariant still holds end-to-end).

---

## Live Admission Slice 7 results — mixed-source admission priority

Closeout for the Live Admission Slice 7 surface. Proves the
engine's source-priority rule end-to-end: when
`free_due_to_cancel_` and `free_due_to_completion_` are
**both** non-empty at the same admission boundary, admission
drains **cancel-freed first, completion-freed second**, and
the residual completion-freed pool stays untouched.

### Build

- Clean build, zero errors. Source-only edits, four targeted
  changes: STEP bump to `HPX_CB_ADMIT_STEP7`, soften the
  reused-seq-id ordering gate (global no-duplicate +
  per-`(admitted_at_iter, admission_src)` strictly ascending),
  add `slice7_strict` gate, add `admit_step7:` audit line.
- No new CLI flags, no new trace event names, no new HPX
  primitives. The Slice 6 `hpx::spinlock` inbox and release+ack
  barrier are reused unchanged.
- Build dir:
  `/Users/Ashk/Desktop/HPX/builds/llama-hpx-hpx-on-metal`.
- Binary:
  `/Users/Ashk/Desktop/HPX/builds/llama-hpx-hpx-on-metal/bin/llama-hpx-continuous-batch-gate`.

### Smoke shape

```text
n_seq_max               = 99
n_active                = 84
n_waiting               = 9
waiting_budget          = 8
n_external_arrivals     = 6
external_arrival_budget = 64
external_release_iter   = 16
--reuse-completed       = ON
cancel_plan             = 1,4,7,2,5,8
cancel_after            = 16
active budgets          = round-robin {8,64,256}
```

Active slot partition (per the round-robin):

```text
budget-8   slots: 0,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48,51,54,57,60,63,66,69,72,75,78,81   (28 slots)
budget-64  slots: 1,4,7,10,13,16,19,22,25,28,31,34,37,40,43,46,49,52,55,58,61,64,67,70,73,76,79,82  (28 slots)
budget-256 slots: 2,5,8,11,14,17,20,23,26,29,32,35,38,41,44,47,50,53,56,59,62,65,68,71,74,77,80,83  (28 slots)
```

cancel_plan `{1,4,7,2,5,8}` ⊂ budget-64 ∪ budget-256, all in
`[0, 84)`.

Derived deterministic timing:

```text
phase 1 admission iter = 8     (min_active_budget)
release_iter           = 16
drain_iter             = 17    (top of release_iter + 1)
phase 2 admission iter = 17    (cancel_after + 1)
admitted budget-64 done_iter   = 80   (17 + 64 − 1)
```

### Captures

```text
local/slice7_default.stdout
local/slice7_default.stderr
local/slice7_smoke.stdout
local/slice7_smoke.stderr
local/slice7_trace.stdout
local/slice7_trace.stderr
local/slice7_repeat2.stdout
local/slice7_repeat2.stderr
```

### Run commands

Default regression (no Slice 7 args; verifies the rest of the
pipeline is unaffected):

```sh
/Users/Ashk/Desktop/HPX/builds/llama-hpx-hpx-on-metal/bin/llama-hpx-continuous-batch-gate \
    --model /Users/Ashk/Desktop/HPX/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
    > local/slice7_default.stdout 2> local/slice7_default.stderr
```

Slice 7 compact smoke:

```sh
/Users/Ashk/Desktop/HPX/builds/llama-hpx-hpx-on-metal/bin/llama-hpx-continuous-batch-gate \
    --model /Users/Ashk/Desktop/HPX/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
    --n-seqs 99 --n-active 84 --n-waiting 9 --waiting-budget 8 \
    --n-external-arrivals 6 --external-arrival-budget 64 \
    --external-release-iter 16 \
    --reuse-completed \
    --cancel-plan 1,4,7,2,5,8 --cancel-after 16 \
    > local/slice7_smoke.stdout 2> local/slice7_smoke.stderr
```

Slice 7 trace-on compact:

```sh
LLAMA_HPX_CB_TRACE=1 \
/Users/Ashk/Desktop/HPX/builds/llama-hpx-hpx-on-metal/bin/llama-hpx-continuous-batch-gate \
    --model /Users/Ashk/Desktop/HPX/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
    --n-seqs 99 --n-active 84 --n-waiting 9 --waiting-budget 8 \
    --n-external-arrivals 6 --external-arrival-budget 64 \
    --external-release-iter 16 \
    --reuse-completed \
    --cancel-plan 1,4,7,2,5,8 --cancel-after 16 \
    > local/slice7_trace.stdout 2> local/slice7_trace.stderr
```

Slice 7 repeat 2 (determinism):

```sh
/Users/Ashk/Desktop/HPX/builds/llama-hpx-hpx-on-metal/bin/llama-hpx-continuous-batch-gate \
    --model /Users/Ashk/Desktop/HPX/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
    --n-seqs 99 --n-active 84 --n-waiting 9 --waiting-budget 8 \
    --n-external-arrivals 6 --external-arrival-budget 64 \
    --external-release-iter 16 \
    --reuse-completed \
    --cancel-plan 1,4,7,2,5,8 --cancel-after 16 --repeat 2 \
    > local/slice7_repeat2.stdout 2> local/slice7_repeat2.stderr
```

### Final-line table

```text
default regression  HPX_CB_ADMIT_STEP7: PASS
compact smoke       HPX_CB_ADMIT_STEP7: PASS
trace-on compact    HPX_CB_ADMIT_STEP7: PASS
repeat 2            HPX_CB_ADMIT_STEP7: PASS
```

### Phase 1 mapping (iter 8, completion_freed, preloaded)

```text
req 84 -> seq  0
req 85 -> seq  3
req 86 -> seq  6
req 87 -> seq  9
req 88 -> seq 12
req 89 -> seq 15
req 90 -> seq 18
req 91 -> seq 21
req 92 -> seq 24
```

Every phase-1 admitted result snapshot has
`admission_source = completion_freed`,
`arrival_source = preloaded`, `admitted_at_iter = 8`,
`decode_budget = 8`, `done_iter = 15`,
`pos_max_at_clear = 12`.

Trace evidence (from `local/slice7_trace.stderr`):

```text
[hpx-cb-gate] event=request_admitted_live request=84 reused_seq_id=0  iter=8 admission_source=completion_freed arrival_source=preloaded
[hpx-cb-gate] event=request_admitted_live request=85 reused_seq_id=3  iter=8 admission_source=completion_freed arrival_source=preloaded
[hpx-cb-gate] event=request_admitted_live request=86 reused_seq_id=6  iter=8 admission_source=completion_freed arrival_source=preloaded
[hpx-cb-gate] event=request_admitted_live request=87 reused_seq_id=9  iter=8 admission_source=completion_freed arrival_source=preloaded
[hpx-cb-gate] event=request_admitted_live request=88 reused_seq_id=12 iter=8 admission_source=completion_freed arrival_source=preloaded
[hpx-cb-gate] event=request_admitted_live request=89 reused_seq_id=15 iter=8 admission_source=completion_freed arrival_source=preloaded
[hpx-cb-gate] event=request_admitted_live request=90 reused_seq_id=18 iter=8 admission_source=completion_freed arrival_source=preloaded
[hpx-cb-gate] event=request_admitted_live request=91 reused_seq_id=21 iter=8 admission_source=completion_freed arrival_source=preloaded
[hpx-cb-gate] event=request_admitted_live request=92 reused_seq_id=24 iter=8 admission_source=completion_freed arrival_source=preloaded
```

### Phase 2 mapping (iter 17, cancel_freed, external)

```text
req 93 -> seq 1
req 94 -> seq 2
req 95 -> seq 4
req 96 -> seq 5
req 97 -> seq 7
req 98 -> seq 8
```

Every phase-2 admitted result snapshot has
`admission_source = cancel_freed`,
`arrival_source = external`, `admitted_at_iter = 17`,
`decode_budget = 64`, `done_iter = 80`,
`pos_max_at_clear = 68`.

Trace evidence:

```text
[hpx-cb-gate] event=request_admitted_live request=93 reused_seq_id=1 iter=17 admission_source=cancel_freed arrival_source=external
[hpx-cb-gate] event=request_admitted_live request=94 reused_seq_id=2 iter=17 admission_source=cancel_freed arrival_source=external
[hpx-cb-gate] event=request_admitted_live request=95 reused_seq_id=4 iter=17 admission_source=cancel_freed arrival_source=external
[hpx-cb-gate] event=request_admitted_live request=96 reused_seq_id=5 iter=17 admission_source=cancel_freed arrival_source=external
[hpx-cb-gate] event=request_admitted_live request=97 reused_seq_id=7 iter=17 admission_source=cancel_freed arrival_source=external
[hpx-cb-gate] event=request_admitted_live request=98 reused_seq_id=8 iter=17 admission_source=cancel_freed arrival_source=external
```

### Priority proof

At the discriminating iter 17 boundary, both pools are
non-empty (`free_due_to_cancel_` = 6 entries,
`free_due_to_completion_` = 19 entries) and the waiting queue
has exactly 6 entries (the just-drained external arrivals).
The engine must consume cancel-freed first; the smoke records:

```text
iter 8  admissions: completion_freed = 9   cancel_freed = 0
iter 17 admissions: cancel_freed     = 6   completion_freed = 0

iter 17 reused_seq_id set = {1, 2, 4, 5, 7, 8}
                         == sorted(cancel_plan)

no iter-17 admission has reused_seq_id in the residual
completion-freed pool:
    {27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66,
     69, 72, 75, 78, 81}
```

If the priority were reversed (completion-freed first), the
smoke would instead admit the 6 externals onto
`{27, 30, 33, 36, 39, 42}` and `free_due_to_cancel_` would
end with 6 residual entries. The slice7_strict gate fails
closed against both shapes.

`admit_step7:` audit line (smoke + both repeats):

```text
iter[r] admit_step7: phase1@iter=8 completion_freed=9 phase2@iter=17 cancel_freed=6 pool_residual=19 admission_iter_set={8,17} first_external_drain_iter=17
```

### Residual completion-freed pool

```text
completion_freed_pool_size_at_run_end = 19
```

Computed from the engine and asserted both by the generic
gate (Slice 5) and by `slice7_strict` (Slice 7). The 19 slots
are the 28 budget-8 actives minus the 9 consumed at iter 8;
the demand gate stops subsequent pushes from naturally
completing admitted-budget-8 (iter 15) and from later
budget-64 / budget-256 actives (iter 63 / 255) because the
waiting queue is empty at those iters.

The 19 residual seq_ids stay KV-empty under the existing
all-`n_seq_max`-slots residual sweep at engine end.

### Engine counters (smoke + both repeats)

| Counter | Expected | Observed |
|---|---|---|
| `admitted_count` | 15 | 15 |
| `completion_freed_admissions` | 9 | 9 |
| `cancel_freed_admissions` | 6 | 6 |
| `external_admitted_count` | 6 | 6 |
| `arrival_drained_count` | 6 | 6 |
| `first_external_drain_iter` | 17 | 17 |
| `iter_release_fired_set` | `{16}` | `{16}` |
| `submitter_ack_set` | `{16}` | `{16}` |
| `admission_iter_set` | `{8, 17}` | `{8, 17}` |
| `waiting_queue_size_at_engine_end` | 0 | 0 |
| `promises_fulfilled` | 99 | 99 |
| `results.size()` | 99 | 99 |
| `completion_freed_pool_size_at_run_end` | 19 | 19 |
| `cancelled_count` | 6 | 6 |
| `residual_kv_empty` | true | true |

Status / audit split:

```text
orig_completed_total      = 78   (28 b-8 + 25 b-64 + 25 b-256)
orig_cancelled_total      = 6
admitted_completed_total  = 15   (9 completion_freed + 6 cancel_freed)
completed_total           = 93   (orig + admitted)
cancelled_total           = 6
```

### Trace counts (trace-on capture)

```text
event=request_queued   arrival_source=preloaded   =  9
event=request_queued   arrival_source=external    =  6
event=request_submitted_external                  =  6
event=arrival_drained                             =  6
event=iter_release_fired                          =  1
event=submitter_ack_observed                      =  1
event=request_admitted_live (total)               = 15
event=seq_complete                                = 93
event=admitted_complete                           = 15
event=kv_cleared                                  = 93
event=promise_fulfilled                           = 93
event=cancel_requested                            =  6
event=cancel_observed                             =  6
event=cancel_kv_cleared                           =  6
event=cancel_future_fulfilled                     =  6
```

All 15 trace-count expectations match the design exactly. The
generic completion-trace family (`seq_complete`, `kv_cleared`,
`promise_fulfilled` at 93 each) covers both the 78 original
completions and the 15 admitted completions; the cancel-family
events fire only on the cancellation path (6 each).

### Trace-off quietness

```text
local/slice7_default.stderr    [hpx-cb-gate] event=    0
local/slice7_smoke.stderr      [hpx-cb-gate] event=    0
local/slice7_repeat2.stderr    [hpx-cb-gate] event=    0
```

Every trace-off capture emits zero `[hpx-cb-gate] event=`
lines; the trace path stays one atomic load + early return
per call site when `LLAMA_HPX_CB_TRACE` is unset.

### Residual KV

`residual_kv_empty = true` across default, smoke, trace-on,
and both iters of the repeat-2 run. The Slice 6 engine-side
gate ensures `inbox_` and `external_promises_` are also
asserted empty BEFORE the residual-KV sweep.

### Repeat determinism

```text
iter[0] admit_step7: phase1@iter=8 completion_freed=9 phase2@iter=17 cancel_freed=6 pool_residual=19 admission_iter_set={8,17} first_external_drain_iter=17
iter[1] admit_step7: phase1@iter=8 completion_freed=9 phase2@iter=17 cancel_freed=6 pool_residual=19 admission_iter_set={8,17} first_external_drain_iter=17
iter[1] determinism: matches iter 0
```

The per-result determinism tuple `(seq_id, request_id,
n_decoded, generated_tokens, hash, done_iter,
pos_max_at_clear, admitted_at_iter, reused_seq_id,
previous_request_id, admission_src)` is byte-identical
across iter 0 and iter 1, including for every phase-1
completion-freed admission and every phase-2 cancel-freed
external admission. The Slice 6/7 counters
(`arrival_drained_count`, `external_admitted_count`,
`first_external_drain_iter`, `iter_release_fired_set`,
`submitter_ack_set`, `admission_iter_set`,
`completion_freed_pool_size_at_run_end`) are stable across
repeats.

### Hash observations

The following hashes are recorded **descriptively** for the
Slice 7 batch shape. Per the agreed hash policy they are
gated only by within-partition uniqueness (1 unique hash per
admitted partition) and `--repeat 2` determinism (iter 1
matches iter 0); they are **not** strict-gated against a
canonical anchor.

```text
src=completion_freed  budget=8   hash=0x0619d4d1900c2365   done_iter={15}  pos_max_at_clear={12}
src=cancel_freed      budget=64  hash=0x3b15a0474dfe11be   done_iter={80}  pos_max_at_clear={68}
```

The completion-freed admitted budget-8 hash happens to equal
the canonical budget-8 anchor `0x0619d4d1900c2365`. The
cancel-freed admitted budget-64 hash happens to equal the
Slice 6 observed value `0x3b15a0474dfe11be`. Both equalities
are *observed, not required* — the Slice 7 batch shape (84
actives, `--reuse-completed` ON, mixed-source admission at
iter 17) is not a strict anchor for any of the prior slice
hashes; the equalities are recorded as evidence rather than
gated.

### Scope

Slice 7 proves the **mixed-source admission priority**
invariant under a deterministic two-phase smoke. After
Slice 7, the prototype demonstrates every pairwise
combination of the three admission feed-in sources
(preloaded waiters, cancel-freed slots, completion-freed
slots, external arrivals) that the v1 admission loop is
designed to handle.

Slice 7 deliberately does **not** exercise:

- More than one release barrier in a single run (single
  `release_iter` only; multi-K release schedules left for
  a later slice).
- External arrivals admitted via `completion_freed` (the
  smoke binds externals to cancel-freed slots; an external
  arrival admitted via completion-freed is left for a
  future surface).
- Wall-clock arrival schedules or streaming.
- More than one engine task per process (the single-engine-
  task invariant still holds end-to-end).
- Any per-request priority queue; admission within each
  source is FIFO by request_id.
- Performance comparisons; Slice 7 is correctness-only.
