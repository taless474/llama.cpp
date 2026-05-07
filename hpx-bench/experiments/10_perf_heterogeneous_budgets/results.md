# Results — heterogeneous budgets experiment

## Summary

- EXPERIMENT_OVERALL: PASS
- Branch label: `hpx_short_worse`
- 44/44 invocations completed
- 0 n_tok multiset mismatches
- std-vs-hpx per-budget hash equality: PASS

Correctness is strong. Performance does not show an HPX advantage on
this CPU-only TinyLlama workload.

The HPX backend remained correct and robust, but did not improve
short-request latency or makespan versus the std backend in this
fixed-shape mixed-budget test.

## Correctness

All four layer summaries passed:

```text
layer_correctness_std    LAYER_OVERALL = PASS   (11/11 trials, 0 mismatches)
layer_correctness_hpx    LAYER_OVERALL = PASS   (11/11 trials, 0 mismatches)
layer_timing_std         LAYER_OVERALL = PASS   (11/11 trials, 0 mismatches)
layer_timing_hpx         LAYER_OVERALL = PASS   (11/11 trials, 0 mismatches)
```

Per-trial gates: every trial produced 12 ok rows whose
`n_tokens_generated` multiset equalled the plan multiset
`{8:5, 16:3, 32:1, 64:3}`. Per-budget hash consistency held within
every PASS trial.

Cross-backend equality at the budget-class granularity was PASS in
both conditions (`condition_correctness` and `condition_timing`):
every budget in the plan produced one canonical hash on each backend,
and the four std-side hashes equalled the four hpx-side hashes
exactly.

Canonical hashes (`summaries/canonical_hashes_by_budget.json`):

```text
8  -> 0x0619d4d1900c2365
16 -> 0x833045f1e2ebf49f
32 -> 0x6794e47fe0f84af1
64 -> 0x7efcc69b4edadb70
```

The 16-token hash matches the canonical fingerprint pinned in
`docs/hpx/serving_bench_acceptance.md` for the uniform-16-token
shape; the other three are observed canonicals for this prompt and
this experiment's budgets.

This validates budget-class correctness under the current
completion-order output convention. Per-request-index identities are
not asserted here — see `facts.md` "Harness output convention" for
why.

## Timing results

All values from the timing layer (`condition_timing.txt`), measured
trials only (10 per backend after dropping trial 0). Per-class rows
are pooled across trials, so n=row_count_per_trial × 10.

short-class total_ms (n=80):

```text
median   std=633.03 ms   hpx=645.67 ms   delta = +12.64 ms (+1.9967%)
p95      std=953.22 ms   hpx=950.93 ms   delta =  -2.30 ms (-0.24%)
CV       std=0.39        hpx=0.39
```

medium-class total_ms (n=10):

```text
median   std=979.89 ms   hpx=995.48 ms   delta = +15.59 ms (+1.59%)
p95      std=1020.37 ms  hpx=1056.60 ms  delta = +36.23 ms (+3.55%)
CV       std=0.026       hpx=0.033
```

long-class total_ms (n=30):

```text
median   std=1566.96 ms  hpx=1561.85 ms  delta =  -5.11 ms (-0.33%)
p95      std=1634.42 ms  hpx=1733.41 ms  delta = +98.98 ms (+6.06%)
CV       std=0.17        hpx=0.18
```

trial makespan_ms (n=10):

```text
median   std=1598.37 ms  hpx=1614.35 ms  delta = +15.98 ms (+1.00%)
p95      std=1650.99 ms  hpx=1785.15 ms  delta = +134.16 ms (+8.13%)
CV       std=0.020       hpx=0.055
```

trial queue_drain_ms (n=10):

```text
median   std=16.99 ms    hpx=45.05 ms    delta = +28.06 ms (+165.19%)
p95      std=57.31 ms    hpx=91.88 ms    delta = +34.57 ms (+60.33%)
CV       std=0.86        hpx=0.64
```

Note that the relative delta on `queue_drain_ms` is large, but the
absolute delta is small (tens of ms on a ~1.6 s makespan) and CV is
high.

## Interpretation

The mechanical branch label is `hpx_short_worse` because short-class
median `total_ms` was worse by more than +1.0%
(`DELTA_PCT_SHORT_TOTAL_MED = +1.9967`).

Qualifying that label:

```text
short  median moved against HPX by ~2%, but p95 was essentially flat
       (-0.24%) and slightly favoured HPX. Per-request pool CV bands
       overlap substantially (std cv ~0.39, hpx cv ~0.39 on medians
       of 633 ms and 646 ms), so the +2% median delta sits inside the
       within-pool variability.
makespan moved against HPX at the median (+1.00%), with a wider
       p95 gap (+8.13%) and higher hpx-side CV (0.055 vs 0.020).
long   median slightly favoured HPX (-0.33%), while long p95 favoured
       std (+6.06%).
medium descriptive only — n=10 across 10 trials means a single
       measured row per trial, and the per-trial CV is small but the
       cross-trial variance is what matters.
queue_drain showed the largest relative movement but the smallest
       absolute movement; the metric is also the noisiest in the
       suite (CV 0.86 / 0.64) so the relative number should not be
       read as a structural difference.
```

Conclusion: no evidence of a serving-level HPX latency benefit in
this mixed-budget CPU TinyLlama experiment. The HPX backend is
correct and robust under the heterogeneous budget plan, but the
short-class total_ms moved slightly against it and makespan moved
slightly against it. Long-class median slightly favoured HPX without
a corresponding p95 gain. The aggregate picture is consistent with
the prior matrix and thread-sweep results: small,
configuration-dependent HPX overhead, no HPX advantage at uniform or
mixed budgets on this shape.

## Evidence files

Under `summaries/`:

```text
experiment_summary.txt
condition_correctness.txt
condition_timing.txt
layer_correctness_std.txt          layer_correctness_hpx.txt
layer_timing_std.txt               layer_timing_hpx.txt
layer_correctness_std_per_trial.csv         layer_correctness_hpx_per_trial.csv
layer_timing_std_per_trial.csv              layer_timing_hpx_per_trial.csv
layer_correctness_std_all_requests.csv      layer_correctness_hpx_all_requests.csv
layer_timing_std_all_requests.csv           layer_timing_hpx_all_requests.csv
canonical_hashes_by_budget.json
```

Raw per-trial artifacts (`bench.stdout`, `bench.stderr`, `per_repeat.csv`,
`hpx_trace.txt`, etc.) are reproducible from `_run_experiment.py` and
live under `runs/`; that directory is gitignored.
