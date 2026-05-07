# Heterogeneous request budgets — std vs hpx

## Purpose

Compare the std and hpx serving backends under requests of mixed
generation lengths. Both backends today are structurally
FIFO-over-context-pool. Uniform-budget workloads keep the pool
balanced and mask any difference between their wake mechanisms. Mixing
budgets puts variance into the pipeline and is the cleanest way to
surface a difference, if any exists, within the existing harness.

## Question

When requests carry mixed generation budgets, do the two backends
behave differently in:

```text
short-class request total_ms
trial makespan / queue-drain shape
long-class request total_ms
```

Equivalence on these metrics is publishable as a closeout: it says
the current HPX backend is, on this workload, a thread-pool rename.
A real divergence — typically on short-class tail or makespan — is a
signal worth investigating.

## Shape

One cell, fixed:

```text
n_contexts   = 2
n_concurrent = 4         (waiter pressure: n_concurrent > n_contexts)
n_requests   = 12
n_threads    = 2
ctx_size     = 2048
batch_size   = 512
prompt       = "Hello, my name is"
seed_base    = 1234
```

Per-request budget plan (same for every trial):

```text
plan = [64, 8, 16, 8, 64, 16, 8, 64, 8, 16, 8, 32]

class    plan[i] in    count    request_index set in submission order
short    {8, 16}       8        {1, 2, 3, 5, 6, 8, 9, 10}
medium   {32}          1        {11}
long     {64}          3        {0, 4, 7}

total tokens / trial = 320
```

Per-request seed: `seed[i] = seed_base + i`. At greedy / argmax
decoding the seed does not affect the generated token stream; the
output of a request depends only on the prompt and its budget.

Fit-check: `n_prompt_tokens(6) + max(plan)(64) = 70 ≪ ctx_size(2048)`.

## Protocol

Two layers, two backends, K=11 process-level trials per (layer,
backend):

```text
correctness_trace_on   LLAMA_SERVING_BENCH_HPX_TRACE=1
timing_trace_off       env unset

backends   std, hpx
trials     trial_00..trial_10        (trial_00 correctness-checked but
                                      excluded from timing aggregation)
```

Trials within a layer are interleaved deterministically: each layer
reseeds `random.Random(1234)` and shuffles a fresh 22-entry list of
`(backend, trial_index)`.

Total invocations: `1 × 2 × 2 × 11 = 44`.

## Result

Correctness PASS. Branch label `hpx_short_worse` — no HPX speedup
claim. The HPX backend remained correct and robust under the
heterogeneous budget plan, but did not improve short-request latency
or trial makespan over the std backend on this CPU-only TinyLlama
workload. Long-class median slightly favoured HPX without a
corresponding p95 gain. `queue_drain_ms` showed the largest relative
movement but a small absolute delta and high CV; it should not be
read as a structural difference.

| metric                    | std median | hpx median | delta (median) |
|---------------------------|------------|------------|----------------|
| short total_ms (n=80)     | 633.03 ms  | 645.67 ms  | +12.64 ms (+2.00%) |
| medium total_ms (n=10)    | 979.89 ms  | 995.48 ms  | +15.59 ms (+1.59%) |
| long total_ms (n=30)      | 1566.96 ms | 1561.85 ms |  −5.11 ms (−0.33%) |
| makespan_ms (n=10)        | 1598.37 ms | 1614.35 ms | +15.98 ms (+1.00%) |
| queue_drain_ms (n=10)     |   16.99 ms |   45.05 ms | +28.06 ms (noisy)  |

Detailed report: `results.md`. Summarized evidence:
`summaries/`. Raw per-trial artifacts: `runs/` (gitignored,
reproducible from `_run_experiment.py`).

## How to run

The experiment is self-contained under
`hpx-bench/experiments/10_perf_heterogeneous_budgets/`. Helpers compute
paths from `Path(__file__).resolve()`. The binary and model can be
overridden via environment variables:

```text
LLAMA_SERVING_BENCH_BIN   absolute path to llama-serving-bench
                          default: <repo>/../builds/llama-hpx-hpx-on/bin/llama-serving-bench

LLAMA_MODEL               absolute path to the GGUF
                          default: <repo>/../models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

Steps:

```text
1.  python3 _make_schedule.py
2.  python3 _run_experiment.py
3.  python3 _summarize_layer.py --layer correctness_trace_on --backend std
3a. python3 _summarize_layer.py --layer correctness_trace_on --backend hpx
3b. python3 _summarize_layer.py --layer timing_trace_off    --backend std
3c. python3 _summarize_layer.py --layer timing_trace_off    --backend hpx
4.  python3 _summarize_condition.py --layer correctness_trace_on
4a. python3 _summarize_condition.py --layer timing_trace_off
5.  python3 _summarize_experiment.py
```

`_run_experiment.py` is idempotent: an entry whose
`runs/<layer>/<backend>/trial_NN/bench.exit_code.txt` exists is
skipped unless `--force` is passed. The driver stops on the first
nonzero helper return code.

## Output layout

```text
runs/                                   (gitignored)
    _schedule.json
    schedule_summary.txt
    correctness/std/trial_00../trial_10/
    correctness/hpx/trial_00../trial_10/
    timing/std/trial_00../trial_10/
    timing/hpx/trial_00../trial_10/
    driver_logs/run-<timestamp>/

summaries/                              (tracked)
    layer_correctness_std.txt
    layer_correctness_hpx.txt
    layer_timing_std.txt
    layer_timing_hpx.txt
    condition_correctness.txt
    condition_timing.txt
    experiment_summary.txt
    canonical_hashes_by_budget.json
```

Per-trial artifacts under each `trial_NN/`:

```text
bench.stdout             raw harness stdout
bench.stderr             raw harness stderr (incl. HPX traces when on)
bench.exit_code.txt      harness exit code
process_wall_ms.txt      outer wall around the binary
per_repeat.csv           one row per output line, in completion order
                         (req_index in this CSV is the harness's printed
                          completion index, NOT the original request_index;
                          see facts.md "Harness output convention")
request_summary.txt      same content, human-readable
hpx_trace.txt            filtered HPX lifecycle/pool lines
binary_used.txt          binary path, size, sha256
git_head.txt             rev, last commit, status --short
build_info.txt           build dir, flags, schedule fields
schedule_entry.json      the schedule entry as run
```

## How outputs are interpreted

The summarizers run in three layers and write a machine-readable
`KEY: VALUE` header (terminated by `# END HEADER`) so each layer can
compose without re-applying the gates of the previous layer.

### Per-trial gates

```text
bench.exit_code == 0
per_repeat.csv has 12 data rows
bench.stdout aggregate line: "n_ok=12 n_cancelled=0 n_error=0"
all rows status == ok
bench.stderr contains  "prompt fits: 6 prompt tokens + 64 max_tokens <= 2048 ctx_size"
bench.stderr lacks     "[serving-bench] error"
bench.stderr lacks     "failed"

multiset of per-trial n_tokens_generated values  ==  multiset of plan
   The intended correctness shape assumes no early-EOS for budgets
   {8, 16, 32, 64} on this prompt at greedy / argmax. A multiset
   mismatch fails the trial; the layer summary surfaces the observed
   multiset alongside the plan multiset. The harness output line
   does not print the original request_index, so a shortened row
   cannot be mapped back to a specific plan slot; the trial is not
   silently passed.

per-budget hash consistency within a trial:
   for each budget b in set(plan), all rows with n_tokens_generated == b
   share one generated_token_hash. At greedy/argmax with a fixed prompt
   this must hold; failure indicates a real divergence. Rows whose
   n_tokens_generated is not in set(plan) only appear when the multiset
   gate already fails and do not participate in this gate or in any
   per-budget canonical hash.
```

### Layer-specific gates

```text
correctness_trace_on, hpx:
    1 hpx_runtime_start_once line
    1 engine_hpx ready line     (n_contexts=2 pool_size=2)
    1 hpx_runtime_stop line
    12 acquire / 12 release pairs
    acquire ctx-id set == {0, 1}
    same-ctx acquire/release pairing per request
    acquire precedes release per request
    total filtered HPX trace lines == 27

correctness_trace_on, std:           0 matching HPX trace lines
timing_trace_off,    both backends:  0 matching HPX trace lines
```

### Cross-backend hash equality (primary correctness gate)

Because the harness prints rows in completion order, not in original
request submission order, std and hpx are compared by budget class
rather than by request_index:

```text
For each budget b that appears in the plan:
    std.canonical_hash[b] == hpx.canonical_hash[b]

where canonical_hash[b] is the unique hash observed in any std (resp.
hpx) row whose n_tokens_generated equals b. (The per-budget hash
consistency gate ensures uniqueness within a backend; this gate
ensures equivalence across backends.)
```

`summaries/canonical_hashes_by_budget.json` records the std-side
canonical hash per budget as a regression aid. It is not a primary
gate; the primary cross-backend gate is the equality above.

### Pass policy

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

### Metrics

Per-class request pools, over measured trials × rows in the class:

```text
short  pool: median, p95, CV  of total_ms, total_minus_ttft_ms, ttft_ms
medium pool: median, p95, CV  (descriptive — n is small)
long   pool: median, p95, CV
```

Per-trial trial-level metrics:

```text
makespan_ms        max over per-trial total_ms[i] in the trial
queue_drain_ms     makespan_ms − long-class median total_ms in the trial
process_wall_ms    outer time.monotonic() around the binary
agg_tokens_per_sec 320 generated tokens / wall_s
```

HPX-vs-std deltas per metric:

```text
delta_median = hpx.median - std.median
delta_pct    = (delta_median / std.median) × 100
```

### Branch labels

Branches are evaluated only when `EXPERIMENT_OVERALL == PASS`.
Thresholds match prior experiments' "sub-1% delta inside CV is not
interpretable" rule. The 0.5% makespan bound in `hpx_short_better`
reflects makespan's lower per-trial CV.

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

## Scope

This experiment measures one cell, one prompt, one model, one
quantization, one machine, one CPU-only build, one budget plan,
K=10 measured trials per backend. Reading the result outside that
scope is not supported by this evidence.
