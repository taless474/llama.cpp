# Facts — heterogeneous budgets experiment

This file records the stable, design-time facts of the experiment.
Run results are appended (or added as a separate section) only after
the real run completes.

## Workload

Cell:

```text
n_contexts   = 2
n_concurrent = 4
n_requests   = 12
n_threads    = 2
ctx_size     = 2048
batch_size   = 512
prompt       = "Hello, my name is"
seed_base    = 1234
```

Per-request budget plan (fixed across the protocol):

```text
plan = [64, 8, 16, 8, 64, 16, 8, 64, 8, 16, 8, 32]

class    plan[i] in    request_index set in submission order   count
short    {8, 16}       {1, 2, 3, 5, 6, 8, 9, 10}               8
medium   {32}          {11}                                    1
long     {64}          {0, 4, 7}                               3

total tokens / trial = 320
min / max budget     = 8 / 64

plan-multiset (sorted) = [8, 8, 8, 8, 8, 16, 16, 16, 32, 64, 64, 64]
```

Per-request seed: `seed[i] = seed_base + i` (1234..1245). Decoding is
greedy / argmax, so the seed does not affect the generated token
stream; the output of any request depends only on `(prompt, plan[i])`.

Fit-check: `n_prompt_tokens(6) + max(plan)(64) = 70 ≪ ctx_size(2048)`.

Waiter pressure: `n_concurrent (4) > n_contexts (2)`. Two requests are
in flight at any moment; the rest wait in the FIFO queue until a
context is released.

## Backend surface used

Both backends consume per-request budgets through the existing public
field `serving_bench::request_params::max_tokens`. `backend_std.cpp`
and `backend_hpx.cpp` are not modified for this experiment. The
experiment is enabled only by harness-level CLI plumbing
(`--max-tokens-plan`) which writes the plan into
`request_params::max_tokens` per submission index in
`main.cpp::submit_one`.

## Harness output convention

The harness output line

```text
[serving-bench] req[i] status=... n_tokens_generated=... ...
```

is produced by iterating `results` in completion order and printing
the loop variable `i`, not the request's original `request_index`. As
a consequence:

```text
the i in "req[i]" is a completion-order index, not the submission index
n_tokens_generated[i]   is bound to the budget of whichever request
                        was submitted with that submission index, which
                        the output line does not record
the multiset of n_tokens_generated values for a trial still equals the
multiset of plan budgets (when no early-EOS occurs)
greedy / argmax decoding makes the generated token stream a function
of (prompt, budget) only, so all rows of a given budget class within
a trial share the same generated_token_hash
```

Validation gates in this experiment are completion-order-safe: they
work without any assumption that `req[i]` in the output corresponds to
`plan[i]`. If the harness is later changed to print the
`request_index`, the gate set in this file should be revised before
relying on per-request-index identities.

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
per_repeat.csv has 12 data rows
bench.stdout aggregate line: "n_ok=12 n_cancelled=0 n_error=0"
all rows status == ok
bench.stderr contains   "prompt fits: 6 prompt tokens + 64 max_tokens <= 2048 ctx_size"
bench.stderr lacks      "[serving-bench] error"
bench.stderr lacks      "failed"

multiset(n_tokens_generated over the 12 rows) == multiset(plan)
   The intended correctness shape assumes no early-EOS for the
   fixed prompt and the budget set {8, 16, 32, 64}. If the observed
   multiset differs from the plan multiset, the trial FAILS this
   gate (and therefore fails the per-trial gate set). The layer
   summary records the observed multiset alongside the plan
   multiset as the diagnostic. Because the harness output line
   does not print the original request_index, a shortened row
   cannot be mapped back to a specific max_tokens_plan slot, so no
   per-row early-EOS classification is attempted and the trial is
   not silently passed.

per-budget hash consistency within a trial:
   for every budget b in set(plan), all rows of the trial whose
   n_tokens_generated == b share one generated_token_hash. This
   must hold at greedy/argmax with a fixed prompt; failure
   indicates a real divergence. Rows whose n_tokens_generated is
   not in set(plan) do not participate in this gate (they only
   appear when the multiset gate already fails) and do not
   contribute to any per-budget canonical hash.
```

The fit-check line uses `max(plan)` (= 64) because `main.cpp` switches
the fit-check to `max(plan)` whenever `--max-tokens-plan` is passed.

## Layer-specific gates

`correctness_trace_on, backend=hpx` (per trial):

```text
1 hpx_runtime_start_once line
1 engine_hpx ready line     (n_contexts=2 pool_size=2)
1 hpx_runtime_stop line
12 acquire / 12 release pairs
acquire ctx ids and release ctx ids in {0, 1}
release ctx == acquire ctx per request
acquire precedes release per request
acquire ctx-id set == {0, 1}
total filtered HPX trace lines == 27
```

The 27 expected trace lines are
`1 start + 1 ready + 1 stop + 12 acquire + 12 release`. They are
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

Because the harness output is in completion order rather than
submission order, std and hpx are compared by budget class:

```text
For each budget b in the plan:
    std.canonical_hash[b] == hpx.canonical_hash[b]

where canonical_hash[b] is the unique hash observed in any std (resp.
hpx) row of any PASS trial whose n_tokens_generated equals b.
```

This gate is the primary correctness gate of the experiment. Per
the harness output convention above, per-request-index hash matching
is not a meaningful gate without harness changes and is therefore
not used.

`summaries/canonical_hashes_by_budget.json` records the std-side
canonical hash per budget. It is a regression aid only.

## Layer / condition / experiment pass policy

```text
LAYER_PASS              every scheduled trial present, every per-trial
                        gate PASS

CONDITION_PASS          both backend layer summaries PASS, AND
                        per-budget cross-backend hash equality PASS
                        for every budget in the plan

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
tokens_per_second   n_tokens_generated / total_ms × 1000
```

Per-class aggregates (over measured trials × rows in the class):

```text
short  pool: median, p95, CV   of total_ms, total_minus_ttft_ms, ttft_ms
medium pool: median, p95, CV   (descriptive — n is small)
long   pool: median, p95, CV
```

Per-trial trial-level metrics:

```text
makespan_ms          max over per-trial total_ms[i] within the trial
queue_drain_ms       makespan_ms − long-class median total_ms in the trial
                     (positive ⇒ short or medium rows still waiting after
                      the longest long row finished)
process_wall_ms      outer time.monotonic() around the binary
agg_tokens_per_sec   320 generated tokens / wall_s
```

HPX-vs-std deltas:

```text
delta_median = hpx.median - std.median
delta_pct    = (delta_median / std.median) × 100
```

Headline metrics, treated co-equally:

```text
short-class total_ms        median, p95
trial makespan_ms           median
trial queue_drain_ms        median (read jointly with makespan)
```

## Interpretation rules

Branches are evaluated only when `EXPERIMENT_OVERALL == PASS`.

```text
hpx_short_better:
    short.total_ms.delta_pct ≤ −1.0%, AND
    makespan.delta_pct       ≤ +0.5%, AND
    long.total_ms.delta_pct  ≤ +1.0%

equivalent:
    |short.total_ms.delta_pct| ≤ 1.0%, AND
    |makespan.delta_pct|       ≤ 1.0%, AND
    |long.total_ms.delta_pct|  ≤ 1.0%

hpx_short_worse:
    short.total_ms.delta_pct ≥ +1.0%

mixed:
    sign of delta_pct disagrees across short / medium / long, OR
    short.total_ms and makespan disagree

incomplete:
    EXPERIMENT_OVERALL ≠ PASS, OR any required delta is unavailable.
```

Threshold rationale: prior experiments treated sub-1% deltas inside
the CV band as not interpretable. Makespan is per-trial (n=10 iid)
with typically lower CV than per-request pool stats, so its bound in
`hpx_short_better` is tighter at 0.5%.

## Caveats

```text
Per-class n is small per trial: 8 short rows + 1 medium + 3 long.
   Across 10 measured trials: 80 short / 10 medium / 30 long
   measured rows. The medium row is descriptive only.
The harness output convention above ties hash identity to budget
   class, not to submission index. Per-request-index hash matching
   is therefore not a valid gate in this experiment without harness
   changes.
At greedy/argmax with a fixed prompt, every row of a given budget
   class shares a single generated_token_hash. The hash-equality
   gate works at the budget-class granularity, which is the
   strongest gate available without harness changes.
The intended correctness shape assumes no early-EOS for budgets
   {8, 16, 32, 64} on this prompt at greedy / argmax. If a trial's
   per-row n_tokens_generated multiset differs from the plan
   multiset, the trial fails the budget gate. The condition (and
   therefore the experiment) is then treated as non-comparable and
   not interpreted, because the harness output convention prevents
   mapping a shortened row back to a specific plan slot.
makespan_ms and queue_drain_ms are per-trial metrics with n=10. CV
   tends to be lower than per-request pool stats but the sample is
   still small.
The current HPX backend exposes no scheduling primitive that the std
   backend lacks. The most-likely outcome based on prior matrix and
   sweep results is `equivalent`. The experiment is not designed to
   guarantee an HPX win.
```
