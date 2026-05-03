# HPX benchmark protocol

Use this protocol for any HPX-related performance comparison in this repository.
It is not decode-specific and it is not packet-specific.

Applies to comparisons such as:

- scheduler vs HPX
- HPX off vs HPX on
- serving `std` backend vs serving `hpx` backend
- run planner off vs on
- executor policy A vs B
- packet off vs on
- fallback coalescing off vs on
- decode benchmarks
- prefill benchmarks
- model A/B comparisons, only when the model difference is the experiment

The purpose is to avoid confusing real performance changes with machine noise,
rebuild drift, thermal drift, correctness drift, or different workloads.

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
- no stray `llama-bench`, `llama-simple`, `llama-serving-bench`,
  `mds_stores`, or indexing process consuming significant CPU

If the machine is busy, stop and wait. Do not record benchmark results
from a contended machine unless the contention itself is the experiment.

---

## 2. Same binary / same build

For an A/B comparison, rebuild once and use the same binary for every
condition in the batch.

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
- working tree status
- build directory
- binary path
- CMake configure command
- important env vars
- HPX install path and HPX commit if HPX is used

For local exploratory runs, record `git status --short` before and after
the matrix. The status should not change during the benchmark.

---

## 3. Back-to-back comparison

Run conditions back-to-back and alternate order across batches.

Example:

```text
batch 1: A then B
batch 2: B then A
```

If both batches agree on the direction of the result, drift is less likely
to be the explanation.

For serving-bench matrices, alternate `std` and `hpx` within each workload
shape so temporal drift affects both backends similarly.

---

## 4. Same workload

Keep the workload identical across conditions unless changing the workload
is the experiment.

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

For serving-bench:

- keep `--n-contexts`, `--n-concurrent`, `--n-requests`, `--max-tokens`,
  model, prompt, and backend build identical except for the backend being
  tested
- compare `--backend std` and `--backend hpx` from the same HPX-capable
  binary when testing serving orchestration
- verify token hashes before interpreting timing

For model comparisons:

- state clearly that the model is the variable
- do not mix model comparison with executor comparison in the same table
  unless the table is explicitly designed for that

---

## 5. Correctness before performance

Performance numbers are invalid if correctness fails.

For output-affecting paths, verify:

- no request errors
- no cancellations unless cancellation is the experiment
- stable generated-token hash or equivalent output check
- std-vs-HPX structural equality when comparing serving backends
- same prompt/model/settings across both arms

For `llama-serving-bench`, a repeated benchmark run must satisfy:

```text
n_error=0
n_cancelled=0
expected canonical generated-token hash count
```

If any run violates correctness, stop treating the matrix as benchmark
data and debug correctness first.

---

## 6. Report mean and stdev

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

A result with high variance is not a stable result. Repeat on a quieter
machine or shorten the benchmark if thermal drift is suspected.

For early serving-bench sanity checks, it is acceptable to first show the
raw repeated spread before computing mean/stdev. Do not make speedup
claims from the raw spread alone.

---

## 7. Report path-specific counters when available

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
- serving request count
- serving context count
- serving concurrency level
- context-pool acquire/release count, if traced

For packet experiments, report packet counters.

For run-level executor experiments, report run-plan counters.

For prefill experiments, report region count and dependency shape if available.

For fallback coalescing experiments, report fallback node count and fallback
run count.

For serving-bench experiments, report enough request/context/concurrency
metadata that the shape is reproducible.

---

## 8. Sanity-check timing buckets

If timing buckets are available, check that the bucket deltas explain the
total delta.

Example:

```text
observed_total_delta ≈ bucket_delta_1 + bucket_delta_2 + ...
```

If the bucket math does not approximately match, do not claim a speedup
until the accounting is understood.

---

## 9. Do not compare across contention states

Never compare a run from a quiet machine to a run from a contended machine
as if they are equivalent.

This matters especially for HPX/selective paths because dispatch-heavy
paths can amplify CPU contention differently from the baseline scheduler.

If a run was taken under contention, label it clearly and do not mix it
with quiet-machine results.

---

## 10. Result directory convention

Save results inside the repository, not in `/tmp`.

For local exploratory runs, use:

```text
local/<descriptive-result-dir>/
```

For shareable benchmark evidence, use:

```text
hpx-bench/results/<date>-<slug>/
```

Minimum contents for shareable evidence:

```text
README.md       short summary, commands, commit hash, key numbers
stdout.txt      stdout capture, if relevant
stderr.txt      stderr capture, if relevant
bench.csv       raw CSV, if produced
```

The `README.md` should include:

- goal of the benchmark
- exact command(s) or script path
- git commit hash
- working tree status
- binary path
- model path
- HPX install path and HPX commit, if HPX is used
- machine notes
- key results
- conclusion
- whether the result is actionable or only exploratory

Do not publish large raw `local/` logs directly. Convert them into a
small result summary under `docs/hpx/` or a structured result directory
under `hpx-bench/results/`.

---

## 11. Serving-bench repeated sanity matrix

Use this matrix for early local sanity checks of the HPX-native
`llama-serving-bench` backend after correctness gates pass.

This is not a final performance protocol. It is a small repeated sanity
matrix to check stability, obvious overhead, and correctness preservation.

### Preconditions

- `OFF` / std acceptance checks are green.
- HPX-ON serving gates are green.
- The HPX backend runs real decode.
- std and HPX produce the same canonical generated-token hash for the
  canonical prompt.
- Use the same HPX-capable binary for both `--backend std` and
  `--backend hpx`.
- Do not rebuild between runs in the same matrix.
- Record `git status --short` before and after the matrix.

### Matrix

| Shape | n_contexts | n_concurrent | n_requests | max_tokens | Expected canonical-hash count |
|---|---:|---:|---:|---:|---:|
| A | 1 | 1 | 8 | 16 | 8 |
| B | 2 | 2 | 16 | 16 | 16 |
| C | 1 | 4 | 16 | 16 | 16 |

Backends:

```text
std
hpx
```

Repeats:

```text
5 per (shape, backend)
30 total runs
```

### Run order

Alternate std and HPX within each shape:

```text
A_std_r1, A_hpx_r1, A_std_r2, A_hpx_r2, ...
B_std_r1, B_hpx_r1, B_std_r2, B_hpx_r2, ...
C_std_r1, C_hpx_r1, C_std_r2, C_hpx_r2, ...
```

The goal is to keep each std/hpx pair temporally adjacent so thermal and
background-load drift affects both backends similarly.

### Command template

Use local paths appropriate for your machine:

```bash
/path/to/llama-serving-bench \
  -m /path/to/model.gguf \
  -p "Hello, my name is" \
  --n-contexts <N_CTX> \
  --n-concurrent <N_CONC> \
  --n-requests <N_REQ> \
  --max-tokens 16 \
  --backend std|hpx
```

Capture stdout and stderr per run.

Suggested local layout:

```text
local/bench_repeat/<shape>_<backend>_r<N>.stdout
local/bench_repeat/<shape>_<backend>_r<N>.stderr
local/bench_repeat/git_status_pre.txt
local/bench_repeat/git_status_post.txt
```

A local shell script is acceptable if it is saved and included with the
local evidence. For public docs, report the matrix and template rather
than machine-specific absolute paths.

### Per-run correctness check

Each run must independently satisfy:

```text
n_error=0
n_cancelled=0
canonical hash count matches the expected count for the shape
```

Expected canonical hash count:

```text
A: 8
B: 16
C: 16
```

If a single run violates correctness, the matrix is not valid benchmark
data.

### Metrics to extract

For each run, extract:

- `n_ok`, `n_cancelled`, `n_error`
- wall seconds
- aggregate tokens/sec
- TTFT p50 / p95
- total latency p50 / p95
- per-request TPS coefficient of variation
- canonical hash count

Report one raw table per shape before making any aggregate claims.

### Interpretation

This matrix can support limited statements such as:

```text
The HPX backend remained correct across 30 local runs.
HPX did or did not show obvious catastrophic overhead in this small matrix.
The result was noisy or workload-dependent.
```

It cannot support broad statements such as:

```text
HPX is faster than std.
HPX is better than upstream llama-server.
This result predicts larger-model behavior.
```

---

## 12. Evidence gate

A performance optimization is not considered meaningful until it has:

1. path engagement proof
2. correctness validation when output is affected
3. fair A/B benchmark under this protocol
4. counters showing what changed structurally
5. comparison against the plain scheduler or current baseline

Do not continue optimizing a path just because it is correct. First prove
that it engages, changes the intended execution structure, and has a
plausible route to beating the baseline.
