# Deep-queue short-request stress - std vs hpx

## Purpose

Compare the std and hpx serving backends under a deep queue of short-budget requests. Earlier experiments, including the matrix at uniform 16 tokens, the thread sweep at C_2x4, and the mixed-budget C_2x4 experiment, all measured either equivalence or small-but-consistent HPX overhead. None exposed a structural advantage for HPX.

This experiment increases queue depth and reduces per-request decode time so that per-request queue/wake overhead is a larger fraction of total work and accumulates over many events.

## Question

When 30 short requests sit waiting at any moment during the bulk phase, do the two backends behave differently in:

```text
short-class total_ms tail (p90, p95, p99)
trial makespan
short-class median total_ms
```

The deep queue plus uniform short budget is the cleanest workload for exposing per-request scheduling overhead within the existing harness. If a structural difference between the FIFO-pool implementations exists, this is where it should appear.

## Shape

One cell, fixed:

```text
n_contexts   = 2
n_concurrent = 32        (waiter pressure: n_concurrent / n_contexts = 16:1)
n_requests   = 200
n_threads    = 2
ctx_size     = 2048
batch_size   = 512
prompt       = "Hello, my name is"
seed_base    = 1234
```

Per-request budget plan, uniform:

```text
plan = [8] * 200    (every request asks for 8 generated tokens)

total tokens / trial = 1600
classes              = all "short" (8 <= 16)
```

Per-request seed: `seed[i] = seed_base + i`. At greedy / argmax decoding, the seed does not affect the generated token stream; every request produces the same 8 generated tokens with hash `0x0619d4d1900c2365`, the canonical 8-token hash from experiment 10.

Fit-check:

```text
n_prompt_tokens(6) + max(plan)(8) = 14 << ctx_size(2048)
```

## Protocol

Two layers, two backends, K=11 process-level trials per `(layer, backend)`:

```text
correctness_trace_on   LLAMA_SERVING_BENCH_HPX_TRACE=1
timing_trace_off       env unset

backends   std, hpx
trials     trial_00..trial_10        (trial_00 correctness-checked but
                                      excluded from timing aggregation)
```

Trials within a layer are interleaved deterministically: each layer reseeds `random.Random(1234)` and shuffles a fresh 22-entry list of `(backend, trial_index)`.

Total invocations:

```text
1 × 2 × 2 × 11 = 44
```

Estimated wall time:

```text
~8 minutes
```

## Result

The experiment passed all correctness gates.

| Item | Result |
|---|---:|
| `EXPERIMENT_OVERALL` | PASS |
| Branch label | `hpx_worse` |
| Completed invocations | 44 / 44 |
| Budget-8 hash equality | PASS |
| Canonical budget-8 hash | `0x0619d4d1900c2365` |
| Expected HPX trace lines | 403 |
| HPX correctness trials matching trace expectation | 11 / 11 |

Performance deltas are reported as `hpx - std`.

| Metric | Delta |
|---|---:|
| short.total_ms median | +2.07% |
| short.total_ms p90 | +0.13% |
| short.total_ms p95 | +0.43% |
| short.total_ms p99 | +1.20% |
| makespan median | -0.14% |

The `hpx_worse` branch label was triggered by the short-request median threshold: HPX was **+2.07%** slower than std at the median. The p99 delta was **+1.20%**, which stayed inside the configured +2.0% p99 threshold, and makespan was essentially unchanged at **-0.14%**.

Overall, this experiment supports a conservative conclusion: HPX serving orchestration is correct and robust under deep waiter pressure, but the current HPX FIFO context-pool backend does not show a latency or makespan advantage over the simpler std backend in this CPU-only TinyLlama workload.

The detailed report is in `results.md`. Raw trial artifacts are under `runs/` and ignored by git; summarized evidence is under `summaries/`.

## How to run

The experiment is self-contained under:

```text
hpx-bench/experiments/11_perf_deep_queue_short_requests/
```

Helpers compute paths from `Path(__file__).resolve()`. The binary and model can be overridden via environment variables:

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

`_run_experiment.py` is idempotent: an entry whose `runs/<layer>/<backend>/trial_NN/bench.exit_code.txt` exists is skipped unless `--force` is passed. The driver stops on the first nonzero helper return code.

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
    layer_*_per_trial.csv
    layer_*_all_requests.csv
```

Per-trial artifacts under each `trial_NN/`:

```text
bench.stdout             raw harness stdout (200 req lines)
bench.stderr             raw harness stderr (incl. HPX traces when on)
bench.exit_code.txt      harness exit code
process_wall_ms.txt      outer wall around the binary
per_repeat.csv           one row per output line, in completion order
hpx_trace.txt            filtered HPX lifecycle/pool lines
binary_used.txt          binary path, size, sha256
git_head.txt             rev, last commit, status --short
build_info.txt           build dir, flags, schedule fields
schedule_entry.json      the schedule entry as run
```

## How outputs are interpreted

The summarizers run in three layers and write a machine-readable `KEY: VALUE` header, terminated by `# END HEADER`, so each layer can compose without re-applying the gates of the previous layer.

### Per-trial gates

```text
bench.exit_code == 0
per_repeat.csv has 200 data rows
bench.stdout aggregate line: "n_ok=200 n_cancelled=0 n_error=0"
all rows status == ok
bench.stderr contains  "prompt fits: 6 prompt tokens + 8 max_tokens <= 2048 ctx_size"
bench.stderr lacks     "[serving-bench] error"
bench.stderr lacks     "failed"

multiset(n_tokens_generated over the 200 rows) == {8: 200}
```

The intended correctness shape assumes no early EOS at budget 8 on this prompt at greedy/argmax. A multiset mismatch fails the trial; the layer summary surfaces the observed multiset. The harness output line does not print the original `request_index`, so a shortened row cannot be mapped back to a specific submission slot; the trial is not silently passed.

Per-budget hash consistency within a trial:

```text
all 200 rows share one generated_token_hash
```

At greedy/argmax with a fixed prompt, this must hold. Failure indicates a real divergence.

### Layer-specific gates

```text
correctness_trace_on, hpx:
    1 hpx_runtime_start_once line
    1 engine_hpx ready line     (n_contexts=2 pool_size=2)
    1 hpx_runtime_stop line
    200 acquire / 200 release pairs
    acquire ctx-id set == {0, 1}
    same-ctx acquire/release pairing per request
    acquire precedes release per request
    total filtered HPX trace lines == 403

correctness_trace_on, std:           0 matching HPX trace lines
timing_trace_off,    both backends:  0 matching HPX trace lines
```

### Cross-backend hash equality

The plan is uniform at budget 8, so per-budget cross-backend equality reduces to:

```text
std.canonical_hash[8] == hpx.canonical_hash[8]
```

`summaries/canonical_hashes_by_budget.json` records the std-side canonical hash for budget 8.

### Pass policy

```text
LAYER_PASS              every scheduled trial present, every per-trial
                        gate PASS

CONDITION_PASS          both backend layer summaries PASS, AND
                        per-budget cross-backend hash equality PASS
                        for budget 8

EXPERIMENT_OVERALL      both condition summaries PASS
```

Timing is descriptive throughout the layer summarizers and is interpreted only when `EXPERIMENT_OVERALL == PASS`.

### Metrics

Per-row metrics:

```text
total_ms            queue wait + decode
ttft_ms             submit -> first generated token
total_minus_ttft_ms decode-only (= total_ms - ttft_ms)
tokens_per_second   8 / total_ms × 1000
```

Per-request pool aggregates over measured trials × 200 rows, n=2000 measured rows per backend across 10 measured trials:

```text
short pool: median, p90, p95, p99, CV
            of total_ms, total_minus_ttft_ms, ttft_ms
```

Tail percentiles are emphasized because the deep queue produces a staircase of completion times: early-position rows finish fast, and late-position rows wait for the entire pool to drain. The tail is the metric that exposes per-request scheduling overhead.

Per-trial trial-level metrics:

```text
makespan_ms         max over per-trial total_ms[i] within the trial
process_wall_ms     outer time.monotonic() around the binary
agg_tokens_per_sec  1600 generated tokens / wall_s
```

`queue_drain_ms` from experiment 10 does not apply here because no long class exists.

HPX-vs-std deltas per metric:

```text
delta = hpx - std
delta_pct = (delta / std) × 100
```

### Branch labels

Branches are evaluated only when `EXPERIMENT_OVERALL == PASS`.

```text
hpx_better:
    short.total_ms.delta_pct (median) <= -1.0%, AND
    makespan.delta_pct       (median) <= +0.5%, AND
    short.total_ms.delta_pct (p99)    <= -1.0%

equivalent:
    |short.total_ms.delta_pct (median)| <= 1.0%, AND
    |makespan.delta_pct       (median)| <= 1.0%, AND
    |short.total_ms.delta_pct (p99)|    <= 2.0%

hpx_worse:
    short.total_ms.delta_pct (median) >= +1.0%, OR
    short.total_ms.delta_pct (p99)    >= +2.0%

mixed:
    sign of delta_pct disagrees between median and p99, OR
    short.total_ms and makespan disagree at the median

incomplete:
    EXPERIMENT_OVERALL != PASS, OR any required delta is unavailable.
```

The 1.0% threshold matches prior experiments' rule that a sub-1% delta inside the CV band is not interpretable. The p99 bound is wider, at 2.0%, because tail percentiles have higher per-pool CV than medians do.

## Scope

This experiment measures one cell, one prompt, one model, one quantization, one machine, one CPU-only build, one budget, and K=10 measured trials per backend. Reading the result outside that scope is not supported by this evidence.
