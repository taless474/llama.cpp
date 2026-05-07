# Results - deep-queue short-requests experiment

## Summary

`EXPERIMENT_OVERALL`: **PASS**

Branch label: **hpx_worse**

The deep-queue short-request experiment passed all correctness gates. Both backends completed the full schedule, preserved the canonical budget-8 token hash, and the HPX correctness layer produced the expected context-pool lifecycle trace.

The performance result does not show an HPX advantage. The `hpx_worse` label was triggered by short-request median `total_ms`: HPX was **+2.07%** slower than std at the median. Tail latency did not show a large HPX regression: p99 was **+1.20%**, which is inside the configured +2.0% p99 threshold. Trial makespan was essentially equivalent, with HPX slightly lower at **-0.14%**.

Conservative conclusion:

> The HPX backend remained correct and robust under deep waiter pressure, but it did not outperform the simpler std backend. This run shows a small per-request median overhead for the current HPX FIFO context-pool design, without a compensating makespan or tail-latency improvement.

## Workload

This experiment used one fixed deep-queue, all-short-request shape:

```text
n_contexts   = 2
n_concurrent = 32
n_requests   = 200
n_threads    = 2
plan         = [8] * 200
```

This creates 16:1 waiter pressure:

```text
n_concurrent / n_contexts = 32 / 2 = 16
```

During the bulk phase, two requests can hold contexts while the remaining requests wait in the queue. Since each request generates only eight tokens, the request wake/acquire/release path is more visible than in longer decode workloads.

Each trial generates:

```text
200 requests × 8 tokens/request = 1600 generated tokens
```

The HPX correctness trace expectation is:

```text
1 start + 1 ready + 1 stop + 200 acquire + 200 release = 403 trace lines
```

## Correctness

Correctness passed cleanly.

```text
44 / 44 invocations PASS
4 / 4 layer summaries PASS
```

The budget-8 canonical hash matched across std and hpx:

```text
8 -> 0x0619d4d1900c2365
```

The HPX correctness layer also passed the expected lifecycle and context-pool checks:

```text
11 / 11 HPX correctness trials had the expected 403 trace lines
context pool ids observed: {0, 1}
```

Because every request has the same budget, the harness's completion-order output convention is not a blocker for budget validation. The required token-count multiset is simply:

```text
{8: 200}
```

Cross-backend hash equality reduces to:

```text
std unique hash for budget 8 == hpx unique hash for budget 8
```

That gate passed.

## Timing results

Timing deltas are reported as `hpx - std`.

| Metric | Delta |
|---|---:|
| short.total_ms median | +2.07% |
| short.total_ms p90 | +0.13% |
| short.total_ms p95 | +0.43% |
| short.total_ms p99 | +1.20% |
| makespan median | -0.14% |

The branch label was triggered by the median threshold:

```text
short.total_ms median delta = +2.07% >= +1.0%
```

The p99 result did not trigger the tail threshold:

```text
short.total_ms p99 delta = +1.20% < +2.0%
```

Makespan was effectively unchanged:

```text
makespan median delta = -0.14%
```

## Interpretation

This was the cleanest workload so far for exposing serving-queue overhead:

```text
16:1 waiter pressure
200 short requests per trial
200 acquire/release events per trial
all requests use the same 8-token budget
```

If the current HPX future/promise wake path were going to produce a serving-level advantage over the std FIFO context pool, this workload should have made that advantage easier to see. Instead, HPX showed about **+2% median per-request overhead** and no makespan improvement.

The p90, p95, and p99 results were much closer to std than the median result:

```text
p90: +0.13%
p95: +0.43%
p99: +1.20%
```

So this should not be described as a broad tail-latency failure. The more precise interpretation is:

> HPX showed a small median per-request orchestration overhead in this deep-queue short-request workload, while tail latency and makespan were approximately equivalent.

This strengthens the closeout conclusion for the current design:

> As currently implemented, HPX as a FIFO serving-level context-pool replacement is correct, but it does not outperform the simpler std backend on this CPU-only TinyLlama serving harness.

This result should not be generalized to HPX as a whole or to LLM serving generally. The current HPX backend does not yet use HPX-native advantages such as priority scheduling, cancellation, richer future composition, work stealing, or distributed execution. It is mainly a FIFO context-pool orchestrator around the same opaque `llama_decode` path used by the std backend.

## Evidence files

The summarized evidence for this experiment is under:

```text
hpx-bench/experiments/11_perf_deep_queue_short_requests/summaries/
```

Expected summary files include:

```text
experiment_summary.txt
condition_correctness.txt
condition_timing.txt

layer_correctness_std.txt
layer_correctness_hpx.txt
layer_timing_std.txt
layer_timing_hpx.txt

layer_correctness_std_per_trial.csv
layer_correctness_hpx_per_trial.csv
layer_timing_std_per_trial.csv
layer_timing_hpx_per_trial.csv

layer_correctness_std_all_requests.csv
layer_correctness_hpx_all_requests.csv
layer_timing_std_all_requests.csv
layer_timing_hpx_all_requests.csv

canonical_hashes_by_budget.json
```

Raw trial artifacts are kept under `runs/` and ignored by git. The `summaries/` files are the shareable evidence package.
