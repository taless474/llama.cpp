# Facts — deep-queue short-request stress experiment

This file records the stable, design-time facts of the experiment.
Run results are appended (or added as a separate section) only after
the real run completes.

## Workload

Cell:

```text
n_contexts   = 2
n_concurrent = 32         (waiter pressure: n_concurrent / n_contexts = 16:1)
n_requests   = 200
n_threads    = 2
ctx_size     = 2048
batch_size   = 512
prompt       = "Hello, my name is"
seed_base    = 1234
```

Per-request budget plan (uniform):

```text
plan = [8] * 200

class                = all "short"   (8 ≤ 16)
total tokens / trial = 1600
plan-multiset        = {8: 200}
```

Per-request seed: `seed[i] = seed_base + i` (1234..1433). Decoding is
greedy / argmax, so the seed does not affect the generated token
stream; every request produces the same 8 tokens with hash
`0x0619d4d1900c2365` (the canonical 8-token hash from experiment 10).

Fit-check: `n_prompt_tokens(6) + max(plan)(8) = 14 ≪ ctx_size(2048)`.

Waiter pressure: `n_concurrent (32) > n_contexts (2)`. Two requests
are in flight at any moment; the other 30 wait in the FIFO queue
during the bulk phase. The harness submits up to 32 requests at
startup and refills as completions arrive, so the in-flight count
stays at 32 through ~95% of the trial wall.

## Backend surface used

Both backends consume per-request budgets through the existing public
field `serving_bench::request_params::max_tokens`. `backend_std.cpp`
and `backend_hpx.cpp` are not modified for this experiment. The
experiment is enabled only by the existing harness CLI plumbing
(`--max-tokens-plan` from experiment 10, here used with a uniform
plan of length 200).

## Harness output convention

Same as in experiment 10. The harness output line

```text
[serving-bench] req[i] status=... n_tokens_generated=... ...
```

prints the loop variable `i` from result-vector iteration (completion
order), not the original `request_index`. Validation gates in this
experiment are completion-order-safe and do not assume that `req[i]`
in the output corresponds to the i-th submitted request.

## Path discipline

Helpers compute paths from `Path(__file__).resolve()` and the repo's
grandparent directory. No `/Users/...` literal appears in any helper.
Two environment variables override defaults:

```text
LLAMA_SERVING_BENCH_BIN   absolute path to llama-serving-bench
                          default: <repo>/../builds/llama-hpx-hpx-on/bin/llama-serving-bench

LLAMA_MODEL               absolute path to the GGUF
                          default: <repo>/../models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

Generated outputs stay under the experiment directory:

```text
runs/        gitignored      raw trial artifacts, schedule, driver logs
summaries/   tracked         summarized evidence
```

## Protocol parameters

```text
trial granularity              one binary invocation per trial
trials per (layer × backend)   K = 11
trials retained for timing     10 (trial_00 dropped from timing aggregation)
schedule seed                  1234
schedule scope                 per layer; reseeds, same seed
layer ordering                 correctness_trace_on  →  timing_trace_off
backend ordering within layer  randomized; std/hpx interleaved
total invocations              1 cell × 2 layers × 2 backends × 11 = 44
```

Layer environment:

```text
correctness_trace_on   LLAMA_SERVING_BENCH_HPX_TRACE = 1
timing_trace_off       LLAMA_SERVING_BENCH_HPX_TRACE unset
```

Trial 0 of every (layer, backend) is correctness-gated but excluded
from timing aggregation.

## Per-trial gates

These gates apply to every trial in every (layer, backend) and are
enforced by `_summarize_layer.py`:

```text
bench.exit_code == 0
per_repeat.csv has 200 data rows
bench.stdout aggregate line: "n_ok=200 n_cancelled=0 n_error=0"
all rows status == ok
bench.stderr contains   "prompt fits: 6 prompt tokens + 8 max_tokens <= 2048 ctx_size"
bench.stderr lacks      "[serving-bench] error"
bench.stderr lacks      "failed"

multiset(n_tokens_generated over the 200 rows) == {8: 200}
   The intended correctness shape assumes no early-EOS at budget 8 on
   this prompt at greedy/argmax. If the observed multiset differs, the
   trial FAILS this gate. The harness output line does not print the
   original request_index, so a shortened row cannot be mapped back to
   a specific plan slot; no per-row early-EOS classification is
   attempted.

per-budget hash consistency within a trial:
   all 200 rows share one generated_token_hash. Restricted to budget
   values in set(plan) = {8}; rows whose n_tokens_generated is not 8
   only appear when the multiset gate above already fails.
```

The fit-check line uses `max(plan)` (= 8) because `main.cpp` switches
the fit-check to `max(plan)` whenever `--max-tokens-plan` is passed.

## Layer-specific gates

`correctness_trace_on, backend=hpx` (per trial):

```text
1 hpx_runtime_start_once line
1 engine_hpx ready line     (n_contexts=2 pool_size=2)
1 hpx_runtime_stop line
200 acquire / 200 release pairs
acquire ctx ids and release ctx ids in {0, 1}
release ctx == acquire ctx per request
acquire precedes release per request
acquire ctx-id set == {0, 1}
total filtered HPX trace lines == 403
```

The 403 expected trace lines are
`1 start + 1 ready + 1 stop + 200 acquire + 200 release`. They are
independent of the budget plan: HPX runtime carriers are sized by
`os_threads_for(cfg) = max(1, min(n_concurrent, n_contexts)) = 2`.

`correctness_trace_on, backend=std` (per trial):

```text
0 matching HPX lifecycle / pool lines
```

`timing_trace_off, both backends` (per trial):

```text
0 matching HPX lifecycle / pool lines
```

## Cross-backend hash equality

Because the plan is uniform at budget 8, per-budget cross-backend
equality reduces to a single comparison:

```text
std.canonical_hash[8] == hpx.canonical_hash[8]
```

This must equal `0x0619d4d1900c2365` if the harness, model, and prompt
are unchanged from experiment 10's std correctness layer. A mismatch
indicates a real backend divergence — a bug, not a performance result.

`summaries/canonical_hashes_by_budget.json` records the std-side
canonical hash. It is a regression aid only.

## Layer / condition / experiment pass policy

```text
LAYER_PASS              every scheduled trial present, every per-trial
                        gate PASS

CONDITION_PASS          both backend layer summaries PASS, AND
                        per-budget cross-backend hash equality PASS
                        for budget 8

EXPERIMENT_OVERALL      both condition summaries PASS
```

Timing is descriptive throughout the layer summarizers and is
interpreted only when `EXPERIMENT_OVERALL == PASS`.

## Metrics

Per-row metrics (parsed from the `[serving-bench] req[i] ...` lines):

```text
total_ms            queue wait + decode
ttft_ms             submit → first generated token
total_minus_ttft_ms decode-only (= total_ms − ttft_ms)
tokens_per_second   8 / total_ms × 1000
```

Per-request pool aggregates (over measured trials × 200 rows; n=2000
per backend across 10 measured trials):

```text
short pool: median, p90, p95, p99, mean, CV
            of total_ms, total_minus_ttft_ms, ttft_ms
```

Tail percentiles (p90, p95, p99) are headlined because the deep queue
produces a staircase of completion times. Early-position rows finish
fast; late-position rows wait for the pool to drain. Per-request
scheduling overhead, if any, accumulates into the tail.

Per-trial trial-level metrics:

```text
makespan_ms          max over per-trial total_ms[i] within the trial
process_wall_ms      outer time.monotonic() around the binary
agg_tokens_per_sec   1600 generated tokens / wall_s
```

`queue_drain_ms` from experiment 10 is omitted: it is defined relative
to a long-class median that does not exist here.

HPX-vs-std deltas:

```text
delta = hpx − std
delta_pct = (delta / std) × 100
```

Headline metrics, treated co-equally:

```text
short.total_ms                 median and p99
trial makespan_ms              median
```

## Interpretation rules

Branches are evaluated only when `EXPERIMENT_OVERALL == PASS`.

```text
hpx_better:
    short.total_ms.delta_pct (median) ≤ −1.0%, AND
    makespan.delta_pct       (median) ≤ +0.5%, AND
    short.total_ms.delta_pct (p99)    ≤ −1.0%

equivalent:
    |short.total_ms.delta_pct (median)| ≤ 1.0%, AND
    |makespan.delta_pct       (median)| ≤ 1.0%, AND
    |short.total_ms.delta_pct (p99)|    ≤ 2.0%

hpx_worse:
    short.total_ms.delta_pct (median) ≥ +1.0%, OR
    short.total_ms.delta_pct (p99)    ≥ +2.0%

mixed:
    sign of delta_pct disagrees between median and p99, OR
    short.total_ms and makespan disagree at the median

incomplete:
    EXPERIMENT_OVERALL ≠ PASS, OR any required delta is unavailable.
```

Threshold rationale: prior experiments treated sub-1% deltas inside
the CV band as not interpretable. p99 has higher CV than the median
on a deep-queue staircase, so its bound is widened to 2.0%. The
median p99 jointly catches both location and tail behaviour, which
is what the deep-queue setup is designed to expose.

## Caveats

```text
The deep-queue staircase produces a wide CV at the per-row level;
   median total_ms aggregates over both early and late completion
   positions and so spans a large range. Tail percentiles (p90, p95,
   p99) are more informative than the median for backend comparison
   and are headlined accordingly.
The cross-backend hash equality reduces to a single hash comparison
   at budget 8, so its information value is lower than in
   experiment 10. The gate's purpose is to confirm the harness and
   model state did not change between backends, not to differentiate.
The harness submits up to n_concurrent requests at startup and refills
   as completions arrive. With n_concurrent=32 and n_requests=200,
   the in-flight count stays at 32 through ~95% of the trial wall
   and falls off only as the queue empties. This is the intended
   shape; trial-level metrics (makespan) are dominated by the bulk
   phase rather than queue startup or drain.
The expected outcome from prior experiments is `equivalent` or
   `hpx_worse`. The experiment is not designed to guarantee an HPX
   advantage; its purpose is to expose a structural difference if
   one exists at higher queue depth.
n=2000 measured short rows per backend across 10 trials is a
   substantial pool, but trials are not iid at the row level (rows
   within a trial share a queue) and trial-level n=10 is small.
   Cross-trial CV of trial-level metrics matters more than per-row
   CV for backend comparison.
Single machine, single prompt, single model, single quantization,
   single CPU-only build, single budget, K=10 measured trials.
   Conclusions do not generalize beyond that scope without
   additional evidence.
```
