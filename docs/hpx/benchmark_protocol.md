# HPX benchmark protocol

Use this protocol for any HPX-related performance comparison in this repository.
It is not decode-specific and it is not packet-specific.

Applies to comparisons such as:

- scheduler vs HPX
- HPX off vs HPX on
- run planner off vs on
- executor policy A vs B
- packet off vs on
- fallback coalescing off vs on
- decode benchmarks
- prefill benchmarks
- model A/B comparisons, only when the model difference is the experiment

The purpose is to avoid confusing real performance changes with machine noise, rebuild drift, thermal drift, or different workloads.

---

## 1. Quiet machine check

Before each benchmark batch, check that the machine is quiet.

```bash
ps -A -o %cpu,comm | awk 'NR==1 || $1+0 > 5.0'
pmset -g therm
```

Expected:

- no unexpected CPU-heavy processes
- no thermal warnings
- no stray `llama-bench`, `llama-simple`, `mds_stores`, or indexing process consuming significant CPU

If the machine is busy, stop and wait. Do not record benchmark results from a contended machine unless the contention itself is the experiment.

---

## 2. Same binary / same build

For an A/B comparison, rebuild once and use the same binary for every condition in the batch.

Do not rebuild between arms of the same comparison.

Good:

```text
build once
run A
run B
run B
run A
```

Bad:

```text
build A
run A
rebuild B
run B
```

Record:

- git commit hash
- build directory
- binary path
- CMake configure command
- important env vars
- HPX install path and HPX commit if HPX is used

---

## 3. Back-to-back comparison

Run conditions back-to-back and alternate order across batches.

Example:

```text
batch 1: A then B
batch 2: B then A
```

If both batches agree on the direction of the result, drift is less likely to be the explanation.

---

## 4. Same workload

Keep the workload identical across conditions unless changing the workload is the experiment.

For all benchmarks, keep fixed:

- model file
- prompt/input
- thread count
- batch / ubatch settings
- context length
- GPU offload settings
- quantization / model variant
- environment variables unrelated to the condition being tested

For decode:

- keep `-n` the same across conditions
- use enough generated tokens to reach a stable plateau
- do not extend so long that thermal drift dominates

For prefill:

- keep prompt token count the same
- keep batch and ubatch settings the same
- report prompt eval time and prompt tokens

For model comparisons:

- state clearly that the model is the variable
- do not mix model comparison with executor comparison in the same table unless the table is explicitly designed for that

---

## 5. Report mean and stdev

Do not report only one run.

For each condition, report:

- number of runs
- mean
- standard deviation
- tokens/sec when applicable
- raw timing logs or CSV path

Prefer tables like:

```text
condition | runs | mean ms | stdev ms | tok/s | vs baseline
```

A result with high variance is not a stable result. Repeat on a quieter machine or shorten the benchmark if thermal drift is suspected.

---

## 6. Report path-specific counters when available

If the code emits counters, include them.

Examples:

- total graph nodes
- run count by kind
- fallback runs
- lowered nodes/runs
- packet matches/nodes/runs
- scheduler splits
- native-to-HPX crossings, if measured
- cache hits/misses
- per-bucket timing, if available

For packet experiments, report packet counters.

For run-level executor experiments, report run-plan counters.

For prefill experiments, report region count and dependency shape if available.

For fallback coalescing experiments, report fallback node count and fallback run count.

---

## 7. Sanity-check timing buckets

If timing buckets are available, check that the bucket deltas explain the total delta.

Example:

```text
observed_total_delta ≈ bucket_delta_1 + bucket_delta_2 + ...
```

If the bucket math does not approximately match, do not claim a speedup until the accounting is understood.

---

## 8. Do not compare across contention states

Never compare a run from a quiet machine to a run from a contended machine as if they are equivalent.

This matters especially for HPX/selective paths because dispatch-heavy paths can amplify CPU contention differently from the baseline scheduler.

If a run was taken under contention, label it clearly and do not mix it with quiet-machine results.

---

## 9. Result directory convention

Save results inside the repository, not in `/tmp`.

Preferred layout:

```text
hpx-bench/results/<date>-<slug>/
```

Minimum contents:

```text
README.md       short summary, commands, commit hash, key numbers
stdout.txt      stdout capture, if relevant
stderr.txt      stderr capture, if relevant
bench.csv       raw CSV, if produced
```

The `README.md` should include:

- goal of the benchmark
- exact command(s)
- git commit hash
- binary path
- model path
- HPX install path and HPX commit, if HPX is used
- machine notes
- key results
- conclusion
- whether the result is actionable or only exploratory

---

## 10. Evidence gate

A performance optimization is not considered meaningful until it has:

1. path engagement proof
2. correctness validation when output is affected
3. fair A/B benchmark under this protocol
4. counters showing what changed structurally
5. comparison against the plain scheduler or current baseline

Do not continue optimizing a path just because it is correct. First prove that it engages, changes the intended execution structure, and has a plausible route to beating the baseline.

