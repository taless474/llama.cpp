# FACTS — C_2x4 n_threads sensitivity protocol setup

This file records current-state facts before any run of the
n_threads-sensitivity protocol. It is not a benchmark report and not
a design authority by itself.

## 1. HPX install state

HPX is built and installed:

```text
HPX install dir:  /Users/Ashk/Desktop/HPX/hpx-install/
HPX cmake config: /Users/Ashk/Desktop/HPX/hpx-install/lib/cmake/HPX/HPXConfig.cmake
HPX libs:         /Users/Ashk/Desktop/HPX/hpx-install/lib/libhpx.dylib
                  /Users/Ashk/Desktop/HPX/hpx-install/lib/libhpx_core.dylib
```

No HPX rebuild is required for this protocol.

## 2. HPX-on llama binary

```text
/Users/Ashk/Desktop/HPX/builds/llama-hpx-hpx-on/bin/llama-serving-bench
```

Same binary used by the performance matrix and the prior correctness
slices. No llama rebuild and no source change is in scope for this
protocol.

## 3. Prior PASS evidence — context only, not reused as trials

This protocol is independent of prior runs. They are listed for context
only; their artifacts are not read by the sweep helpers.

```text
local/baselines/comparison_hpx_waiters/
  shape: (n_contexts=2, n_concurrent=4, n_requests=12, n_threads=4, max_tokens=16)
  result: OVERALL: PASS
  scope:  capacity correctness / no-starvation under n_concurrent > n_contexts

local/baselines/perf_hpx_vs_std_matrix/
  shape:  matrix of 4 cells (A_1x1, B_2x2, C_2x4, D_4x4); n_threads=4 fixed
  result: MATRIX_OVERALL_CORRECTNESS: PASS
          16/16 layer summaries PASS
          4/4 condition summaries PASS
          496/496 trial files at exit_code 0
  C_2x4 timing (descriptive only):
          total_ms median:            std  475.062 ms, hpx  511.560 ms (+7.68%)
          total_minus_ttft_ms median: std  200.026 ms, hpx  213.831 ms (+6.90%)
          ttft_ms median:             std  275.767 ms, hpx  295.075 ms (+7.00%)
          process_wall_ms median:     std 1794.560 ms, hpx 1906.228 ms (+6.22%)
          tokens_per_second median:   std   33.680    , hpx   31.277    (-7.13%)
  scope:  cross-cell HPX-vs-std at n_threads=4 fixed
```

These two slices establish that the C_2x4 cell is structurally correct
at n_threads=4 and provide the n_threads=4 timing baseline that this
sweep should reproduce within statistical noise.

## 4. Canonical hash

```text
0x833045f1e2ebf49f
```

This is the canonical generated-token hash for the fixed shape:

```text
model:   tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
prompt:  "Hello, my name is"
tokens:  16 generated tokens
decode:  greedy / argmax
CPU:     yes
```

The hash is independent of `n_threads`: sampling is greedy/argmax,
seed_base is fixed, and the quantized model is the same. All three
n_threads settings must produce this hash on every trial.

The hash is also pinned in `docs/hpx/serving_bench_acceptance.md` for
`max_tokens=16`.

## 5. Sweep shape

Cell (fixed across sweep):

```text
n_contexts=2
n_concurrent=4
n_requests=12   (waiter-pressure cell: n_concurrent > n_contexts)
```

Sweep variable:

```text
n_threads ∈ {1, 2, 4}
```

Common shape, pinned across all (n_threads, backend) combinations:

```text
binary:        /Users/Ashk/Desktop/HPX/builds/llama-hpx-hpx-on/bin/llama-serving-bench
model:         /Users/Ashk/Desktop/HPX/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
prompt:        "Hello, my name is"
max_tokens:    16
ctx_size:      2048
batch_size:    512
seed_base:     1234
canonical hash: 0x833045f1e2ebf49f
```

## 6. Protocol parameters

```text
trial granularity:                                process-level (one binary invocation per trial)
trials per (n_threads × backend × layer):        K = 11
trials retained for timing per (n_threads × bk): 10 (trial 0 dropped)
schedule seed:                                    1234
schedule scope:                                   per (n_threads × layer); fresh schedule, same seed
schedule recording:                               _schedule.json written before any trial runs
layer ordering per n_threads:                     correctness layer first; timing layer only if correctness PASS
n_threads ordering:                               n=1 → n=2 → n=4 (block by setting)
within-layer ordering:                            randomized, std/hpx interleaved (paired)
total invocations:                                3 × 2 × 2 × 11 = 132
estimated wall (warm):                            ~5–7 minutes on an idle machine
```

## 7. Two-layer expectations

### Correctness layer (trace ON)

```text
env:             LLAMA_SERVING_BENCH_HPX_TRACE=1
gates per trial: full common gates + lifecycle / pool / trace gates (hpx variant)
                 inverted-trace gate (std variant)
purpose:         prove (n_threads × backend) is structurally correct under our shape
output:          per-trial artifacts + correctness_summary.txt + layer_pass.txt
timing use:      none — this layer's per-request timings are NOT pooled
                 into the timing summary
```

Expected per-trial filtered HPX trace count, hpx variant only:

```text
1 start + 1 ready + 1 stop + 12 acquire + 12 release = 27
   (independent of n_threads; HPX runtime carriers are not changed by the sweep)
```

Expected ctx-id usage, hpx variant only:

```text
acquire ctx-id set == {0, 1}
```

Expected `os_threads` and `pool_size`, hpx variant only:

```text
os_threads = 2
pool_size  = 2
   (independent of n_threads)
```

Expected std variant trace, all n_threads:

```text
0 matching HPX lifecycle / pool lines
```

### Timing layer (trace OFF)

```text
env:             LLAMA_SERVING_BENCH_HPX_TRACE unset
gates per trial: reduced common gates (hash, status, n_tokens, aggregate counts,
                 prompt-fits line, no error/failed); NO lifecycle / pool / trace
                 gates (no trace lines emitted)
purpose:         measure per-request and per-trial timing free of HPX-only
                 fprintf trace overhead
output:          per-trial artifacts + per_trial_summary.csv + all_requests.csv
                 + timing_summary.txt + condition_comparison.txt
trial 0:         correctness-checked AND excluded from timing aggregation
trials measured: 10 (trials 1..10) per (n_threads, backend)
```

## 8. Scope of this protocol

This protocol measures:

```text
per-request latency (total_ms, ttft_ms, total_minus_ttft_ms) at C_2x4
  for n_threads ∈ {1, 2, 4}, paired std-vs-hpx, with median and p95 reported
process_wall_ms per trial
aggregate_tokens_per_sec per trial
within-(n_threads, cell) std-vs-hpx timing delta (median, p95, stdev, CV)
HPX correctness (lifecycle + pool + RAII pairing) at K=11 trials per
  (n_threads, layer, backend)
trend of HPX-vs-std delta_pct(total_ms median) across n_threads ∈ {1, 2, 4}
```

This protocol does NOT measure:

```text
cells other than C_2x4
n_threads outside {1, 2, 4}
general llama.cpp performance
HPX scheduler quality in general
performance of other prompts, generation lengths, models, or quantizations
behavior of n_threads=1 as a deployment shape (the single-thread ggml
  path is a different code path; cf. README.md §8 caveats)
waiter-path / queueing overhead in isolation (waiter pressure is held
  constant in this protocol; lowering n_threads does not change it)
behavior at K > 10 measured trials; sub-1% deltas inside the CV band
  are not interpretable as "different"
```
