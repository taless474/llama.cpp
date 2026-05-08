# hpx-continuous-batch-gate

The HPX continuous-batching prototype. Wraps the proven Step 5
multi-seq `llama_batch` primitive (see
`tools/multiseq-batch-gate/results.md`) with HPX-owned request and
slot orchestration.

This tool is a **correctness-first prototype**, not a benchmark. No
HPX-vs-std performance claim is made.

## Status

- **Slice 1 (done):** HPX runtime skeleton. Starts the HPX runtime,
  loads the model, creates one `llama_context`, prints the
  structural capacity fields, builds N metadata-only request
  objects, stops the HPX runtime, and exits with `HPX_CB_STEP1: PASS`.
- **Slice 2 (done):** an `engine` class owns `llama_context*`, the
  shared `llama_batch`, the per-seq state vector, and decode-loop
  state. `engine::run()` is the sole code path that touches
  `llama_context` / `llama_batch` / `llama_decode` /
  `llama_memory_seq_*` / `llama_get_logits_ith`, and is launched
  as exactly one HPX task per repeat iteration via `hpx::async`;
  `main()` waits on the returned future.
- **Slice 3 (done):** per-request HPX promise/future on top of the
  Slice-2 engine. One HPX promise/future per seq; futures are
  handed out before the engine task is scheduled; the engine
  fulfills each promise only after the seq reaches its budget,
  records `pos_max_at_clear`, has its KV cleared, and passes the
  cross-talk-against-still-active-siblings check. Main uses
  `hpx::wait_all` on the per-request futures, calls
  `engine_fut.get()` to surface engine-task exceptions, and
  validates correctness only from `request_result` snapshots.
  Main does not touch `llama_context`, `llama_batch`,
  `llama_decode`, `llama_get_logits_ith`, or `llama_memory_seq_*`
  — the residual-KV-empty check is performed inside the engine
  task.
- **Slice 3b (optional later — not scheduled):** named HPX
  orchestration pool / resource-partitioner setup. See
  *Why no orchestration pool yet?* below.
- **Slice 4 (done):** lifecycle trace events
  (`LLAMA_HPX_CB_TRACE=1`) and a descriptive metrics block. Trace
  emits `engine_start`, `request_admitted`, `seq_prefilled`,
  `decode_row`, `seq_complete`, `kv_cleared`, `promise_fulfilled`,
  `engine_stop` to stderr (off by default; off-path is one atomic
  load + early return per call site). Each repeat iteration prints
  a metrics block to stdout: `wall_ms`, `decode_calls`,
  `update_iterations`, `rows_per_batch` (p50/p95/max),
  `active_seqs_per_iter` (p50/p95/max), per-budget completed
  counts, per-budget `ttc_ms` (mean / p95), `futures_created`,
  `promises_fulfilled`, `futures_completed`, `engine_task_count`.
  Metrics are descriptive only — no comparisons or speedup
  language. Final emit is `HPX_CB_STEP4: PASS` / `FAIL: <reason>`.
- **Slice 5 (done):** same-shape correctness equivalence between
  the pure llama.cpp reference (`tools/multiseq-batch-gate/`,
  `GATE_STEP5: PASS`) and the HPX prototype (`HPX_CB_STEP4: PASS`)
  on the 99-seq / `{8,64,256}` shape. Verdict
  `HPX_CB_PROTO: PASS` is recorded in
  `tools/hpx-continuous-batch-gate/results.md`. Correctness
  equivalence only; not a performance comparison.
- **Cancel Slice 1 (done):** cancellation **data model only**.
  Added the `request_status` enum
  (`completed` / `cancelled` / `failed_reserved`) and new cancel
  fields on `seq_state` (`std::atomic<bool> cancel_requested`,
  `cancel_after_decoded_tokens`, `cancel_observed`,
  `cancel_observed_iter`, `n_decoded_at_cancel`) and on
  `request_result` (`status`, `cancel_observed_iter`,
  `n_decoded_at_cancel`). Added CLI: `--cancel-plan <csv>`
  (default `1,4,7,2,5,8`) and `--cancel-after <int>`
  (default `16`). The plan was parsed, printed, and each plan
  seq's actual budget was asserted against the round-robin mix
  prediction. The plan was **not** propagated into `seq_state`
  and the engine did **not** observe cancellation in that slice.
  Final emit was `HPX_CB_CANCEL_STEP1: PASS` / `FAIL: <reason>`.
- **Cancel Slice 2 (done):** cooperative cancellation enabled
  inside the engine loop. The plan from `--cancel-plan` is
  propagated into `seq_state` at engine construction
  (`cancel_after_decoded_tokens` per seq). The engine task
  observes cancellation only at iteration boundaries: once
  post-prefill, and once at the top of each decode iteration
  **before** building the active row set. No `llama_decode`
  call is interrupted. On observation the engine records
  `cancel_observed`, `cancel_observed_iter`,
  `n_decoded_at_cancel`; runs the existing KV-clear +
  cross-talk check; and fulfills the request promise once with
  `status = cancelled`. Cancelled seqs do not appear in any
  later decode batch (they become `seq.done = true` via the
  shared clear path), so `wasted_decode_rows_after_cancel` is
  structurally `0`. The `--cancel-plan` parser is generalized
  to accept `seq_id 0` while still rejecting negative seq ids.
  Final emit was `HPX_CB_CANCEL_STEP2: PASS` / `FAIL: <reason>`.
- **Cancel Slice 3 (done):** result/status path polish on top
  of Cancel Slice 2 — no behavioral change. Cancel-family
  trace events were aligned to the spec format
  (`cancel_requested seq=<id> budget=<int> cancel_after=<int>`,
  `cancel_observed seq=<id> iter=<int> n_decoded=<int>`,
  `cancel_kv_cleared seq=<id> pos_max_at_clear=<int> cross_talk_ok=1`,
  `cancel_future_fulfilled seq=<id> status=cancelled ttc_us=<int>`).
  Lifecycle events `seq_complete`, `kv_cleared`, and
  `promise_fulfilled` fire only on the completion path; the
  cancellation path emits its own `cancel_*` events. Default-plan
  trace event counts: `seq_complete=93`, `kv_cleared=93`,
  `promise_fulfilled=93`, `cancel_requested=6`,
  `cancel_observed=6`, `cancel_kv_cleared=6`,
  `cancel_future_fulfilled=6`. Main also prints an explicit
  per-iter `status_summary: completed=<> cancelled=<> total=<>`
  line and gates `n_decoded == n_decoded_at_cancel` for
  cancelled results. Final emit was `HPX_CB_CANCEL_STEP3: PASS`
  / `FAIL: <reason>`.
- **Cancel Slice 4 (this version):** closeout of the
  cooperative-cancellation line — no new behavior. The per-iter
  metrics block now reports completed/cancelled splits
  correctly: the `completed[budget=B]` line counts only
  `status == completed` (the legacy form printed every seq of
  that budget, which was misleading once cancellation became
  real), and a paired `cancelled[budget=B]` line is emitted for
  budgets with any cancelled seqs. The single
  `ttc_ms[budget=B]` field is split into
  `ttc_ms_completed[budget=B]` and (when present)
  `ttc_ms_cancelled[budget=B]`, both descriptive only and not
  part of the determinism contract. Top-level metrics added
  after `engine_task_count`: `completed_count`,
  `cancelled_count`, `decode_failures`,
  `wasted_decode_rows_after_cancel`, `residual_kv_empty`. All
  Cancel Slice 2 / 3 correctness gates still apply unchanged.
  The cooperative-cancellation evidence is summarized in
  `tools/hpx-continuous-batch-gate/results.md` under the
  *Cooperative cancellation results* section. Final emit is
  `HPX_CB_CANCEL_STEP4: PASS` / `FAIL: <reason>`.

See `docs/hpx/hpx_continuous_batching_prototype_design.md` for the
full slice plan, correctness gates, and out-of-scope list.

## Why no orchestration pool yet?

A named HPX orchestration pool / resource-partitioner setup is
intentionally deferred. With only one engine task, it would not
change runtime behavior in a meaningful way. It becomes useful once
the prototype has multiple HPX-side tasks, such as admission,
cancellation, priority scheduling, streaming/result dispatch, or
metrics continuations. Until then, adding pool/executor wiring
would increase failure surface without testing a new serving
invariant.

Slice 3b may be added later with gates such as:

- engine task submitted through a named orchestration executor
- engine records pool/thread identity
- no llama.cpp context access escapes the engine task
- all Slice 3 correctness gates still pass

## Build

The target is opt-in (default OFF) so default builds are unaffected.
HPX install path is provided through `HPX_DIR`.

```sh
cmake -S . -B <build_dir> \
    -DLLAMA_BUILD_HPX_CONTINUOUS_BATCH_GATE=ON \
    -DHPX_DIR=/Users/unick/Desktop/hpx/hpx-install/lib/cmake/HPX
cmake --build <build_dir> --target llama-hpx-continuous-batch-gate
```

(Adjust `HPX_DIR` to your local HPX install. The path used by the
existing HPX-on serving-bench build dir is shown above.)

## Usage

```text
llama-hpx-continuous-batch-gate --model <path> [options]

  --model <path>            (required) path to .gguf model file
  --prompt <string>         default: "Hello, my name is"
  --ctx-size <int>          default: 32768
  --n-seq-max <int>         default: 99
  --n-batch <int>           default: 1024
  --n-threads <int>         default: 2  (libllama compute threads)
  --n-seqs <int>            default: 99
  --decode-budget-mix <csv> default: "8,64,256"
  --hpx-os-threads <int>    default: 1  (HPX orchestration threads)
  --repeat <int>            default: 1
  --cancel-plan <csv>       default: "1,4,7,2,5,8" (use "none" for empty)
                            non-negative seq_ids cancelled at the
                            iteration boundary once they reach
                            --cancel-after decoded tokens.
  --cancel-after <int>      default: 16
                            (decoded-token threshold; 0 fires post-prefill)
```

Set `LLAMA_HPX_CB_TRACE=1` to enable HPX runtime startup/shutdown
trace lines plus the Slice-4 lifecycle events (`engine_start`,
`request_admitted`, `seq_prefilled`, `decode_row`, `seq_complete`,
`kv_cleared`, `promise_fulfilled`, `engine_stop`) and the
Cancel-Slice-2 events (`cancel_requested`, `cancel_observed`,
`cancel_kv_cleared`, `cancel_future_fulfilled`) on stderr.

### Trace event format

Each lifecycle event is a single stderr line:

```
[hpx-cb-gate] event=<name> key=value [key=value ...]
```

Examples:

```
[hpx-cb-gate] event=engine_start n_seqs=99 prompt_tokens=6 max_budget=256
[hpx-cb-gate] event=request_admitted seq=0 budget=8
[hpx-cb-gate] event=seq_prefilled seq=0 first_token=29871
[hpx-cb-gate] event=decode_row iter=1 seq=0 pos=6
[hpx-cb-gate] event=seq_complete seq=0 budget=8 done_iter=7 hash=0x0619d4d1900c2365
[hpx-cb-gate] event=kv_cleared seq=0 pos_max_at_clear=12 cross_talk_ok=1
[hpx-cb-gate] event=promise_fulfilled seq=0 ttc_us=...
[hpx-cb-gate] event=engine_stop decode_calls=256 update_iterations=255 wall_ms=...
```

`decode_row` is high-cardinality (one event per row, per decode
iter). All trace output is gated on `LLAMA_HPX_CB_TRACE=1` and is
silent by default; the off-path is a single atomic load per call
site.

### Metrics block

A descriptive metrics block is printed to stdout near the end of
every repeat iteration, before the final PASS/FAIL line. Fields:

```
iter[N] metrics:
  wall_ms                 = <float>
  decode_calls            = <int>
  update_iterations       = <int>
  rows_per_batch          p50=<int> p95=<int> max=<int>
  active_seqs_per_iter    p50=<int> p95=<int> max=<int>
  completed[budget=B]     = <int>            (one line per budget class)
  ttc_ms[budget=B]        mean=<float> p95=<float>
  futures_created         = <int>
  promises_fulfilled      = <int>
  futures_completed       = <int>
  engine_task_count       = <int>
```

Metrics are descriptive only. No comparisons or speedup language.
Timing-derived fields (`wall_ms`, `ttc_ms`) are not part of the
`--repeat` determinism contract.

### Final stdout line

Current final stdout line:

- `HPX_CB_CANCEL_STEP4: PASS`

  or

- `HPX_CB_CANCEL_STEP4: FAIL: <reason>`

Historical slice labels (kept for reference only — the binary now
emits the Cancel Slice 4 label):

- `HPX_CB_STEP1` — Slice 1
- `HPX_CB_STEP2` — Slice 2
- `HPX_CB_STEP3` — Slice 3
- `HPX_CB_STEP4` — Slice 4
- `HPX_CB_CANCEL_STEP1` — Cancel Slice 1
- `HPX_CB_CANCEL_STEP2` — Cancel Slice 2
- `HPX_CB_CANCEL_STEP3` — Cancel Slice 3
- `HPX_CB_CANCEL_STEP4` — Cancel Slice 4 (current)

## Structural prerequisites

- `llama_n_seq_max(ctx) >= --n-seq-max`
- `llama_n_ctx(ctx) >= prompt_tokens + max_decode_budget`
- `llama_n_batch(ctx) >= --n-seqs * prompt_tokens` for one-shot
  prefill
- HPX runtime starts and stops cleanly within the same process as
  `libllama`

Note: for the 99-seq Phase 3 shape, actual `n_ctx` may be larger
than requested because llama.cpp rounds/partitions context capacity
internally. Treat actual capacity as authoritative.

## Slice 4 correctness gates

All Slice-3 correctness gates still apply, listed below. Slice 4
adds trace and metrics output but does not change correctness
behavior or batch shape.

### Slice 3 correctness gates (still apply)

- engine task completes
- `futures_created = 99`
- `promises_fulfilled = 99`
- `futures_completed = 99`
- no promise fulfilled twice
- all seqs reach requested budgets
- exactly 33 seqs per budget class
- budget 8 hash == `0x0619d4d1900c2365`
- budget 64 has one unique hash within the run
- budget 256 has one unique hash within the run
- `done_iter` sets:
    - 8 → `{7}`
    - 64 → `{63}`
    - 256 → `{255}`
- `pos_max_at_clear` sets:
    - 8 → `{12}`
    - 64 → `{68}`
    - 256 → `{260}`
- per-seq KV clear succeeds before promise fulfillment
- clearing one seq does not disturb still-active siblings
- residual KV empty check is performed inside the engine task
- all `llama_decode` calls return 0
- `--repeat 2` is deterministic

## HPX ownership invariant

Futures carry `request_result` snapshots only. No future consumer
may touch `llama_context`, `llama_batch`, `llama_decode`,
`llama_get_logits_ith`, or `llama_memory_seq_*`.

## Out of scope (every slice)

Same as the design note: HTTP, streaming, cancellation, priority,
prompt cache, LoRA, speculative decoding, multimodal, `n_cmpl > 1`,
distributed serving, modifying `tools/server/` /
`tools/serving-bench/` / `tools/multiseq-batch-gate/`, performance
comparisons, multi-backend toggling, context shift, SWA.
