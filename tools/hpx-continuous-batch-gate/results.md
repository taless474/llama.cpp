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
