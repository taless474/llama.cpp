# C_2x4 n_threads sensitivity — `local/baselines/perf_thread_sweep/`

This is a focused follow-up to the HPX-vs-std performance matrix
(`local/baselines/perf_hpx_vs_std_matrix/`).

The matrix showed:

```text
A_1x1: HPX and std indistinguishable
B_2x2: HPX +4.33% slower (total_ms median)
C_2x4: HPX +7.68% slower (total_ms median)
D_4x4: mixed/noisy
```

This experiment varies `n_threads` only, on cell C_2x4, to test whether the HPX overhead in C_2x4 is amplified by kernel-thread oversubscription or is intrinsic to serving / waiter-path orchestration.

## 1. Question

Does the HPX overhead observed in C_2x4 persist when we reduce per-context llama kernel threads?

## 2. Rationale

The original matrix used:

```text
n_threads=4
```

For C_2x4 that means:

```text
2 contexts × 4 kernel threads = 8 kernel threads
4 concurrent requests on a 4-core machine
2 waiters at any time (n_concurrent=4 > n_contexts=2)
```

This may amplify scheduler-side effects and confound the HPX-vs-std comparison.

This sweep isolates the kernel-thread variable while keeping waiter pressure constant.

## 3. Shape

Cell fixed:

```text
n_contexts=2
n_concurrent=4
n_requests=12
```

Common settings fixed:

```text
binary:         /Users/Ashk/Desktop/HPX/builds/llama-hpx-hpx-on/bin/llama-serving-bench
model:          /Users/Ashk/Desktop/HPX/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
prompt:         "Hello, my name is"
max_tokens:     16
ctx_size:       2048
batch_size:     512
seed_base:      1234
canonical hash: 0x833045f1e2ebf49f
```

Sweep variable:

```text
n_threads ∈ {1, 2, 4}
```

Backends:

```text
std
hpx
```

Layers:

```text
correctness_trace_on
timing_trace_off
```

Trial count:

```text
K = 11 process-level trials per (n_threads × backend × layer)
trial 0 correctness-checked but excluded from timing aggregation
10 measured timing trials per (n_threads, backend)
```

Total invocations:

```text
3 thread settings × 2 backends × 2 layers × 11 trials = 132
```

## 4. Artifacts

Experiment root:

```text
local/baselines/perf_thread_sweep/
```

Important files:

```text
README.md
FACTS.md
RESULTS.md
_schedule.json
schedule_summary.txt
sweep_summary.txt
```

Helper scripts:

```text
_make_schedule.py
_run_one_trial.py
_run_sweep.py
_summarize_layer.py
_summarize_thread_setting.py
_summarize_sweep.py
```

Raw trial tree:

```text
thread_settings/
  n1/
    correctness/
      std/
      hpx/
    timing/
      std/
      hpx/
    condition_comparison.txt
  n2/
  n4/
```

Each layer/backend directory contains:

```text
trial_00/
...
trial_10/
all_requests.csv
per_trial_summary.csv
condition_summary.txt
```

Top-level summary:

```text
sweep_summary.txt
```

## 5. Acceptance gates

Common per-trial gates:

```text
harness exit code = 0
all 12 expected request rows parsed
aggregate line: n_ok=12 n_cancelled=0 n_error=0
all request statuses = ok
all n_tokens_generated = 16
all generated_token_hash = 0x833045f1e2ebf49f
prompt-fits line present
no serving-bench error/failed diagnostics
```

Correctness layer, HPX variant:

```text
filtered HPX trace count = 27
   (1 start + 1 ready + 1 stop + 12 acquire + 12 release)
os_threads = 2
pool_size  = 2
acquire ctx-id set = {0, 1}
same-ctx acquire/release pairing
acquire precedes release per request
```

Correctness layer, std variant:

```text
0 matching HPX lifecycle / pool lines
```

Timing layer:

```text
trace off
reduced common gates only
no lifecycle / pool / trace gates
trial 0 correctness-checked and excluded from timing aggregation
trials 1..10 measured per (n_threads, backend)
```

A thread setting passes if both layers pass for both backends. The sweep passes if all three thread settings pass.

## 6. Correctness result

All correctness gates passed:

```text
12 / 12 layer summaries: PASS
3 / 3 thread-setting summaries: PASS
SWEEP_OVERALL_CORRECTNESS: PASS
```

Layer summary decisions:

```text
n_threads=1  correctness_trace_on  std=PASS   hpx=PASS
n_threads=1  timing_trace_off      std=PASS   hpx=PASS
n_threads=2  correctness_trace_on  std=PASS   hpx=PASS
n_threads=2  timing_trace_off      std=PASS   hpx=PASS
n_threads=4  correctness_trace_on  std=PASS   hpx=PASS
n_threads=4  timing_trace_off      std=PASS   hpx=PASS
```

Timing interpretation is allowed because all correctness gates passed.

## 7. Timing result

### total_ms median, pool view

```text
n_threads  std median   hpx median   delta       delta %   std CV   hpx CV
1          741.918 ms   751.691 ms   +9.773 ms   +1.32%    0.200    0.218
2          541.571 ms   580.258 ms   +38.687 ms  +7.14%    0.223    0.227
4          488.700 ms   506.925 ms   +18.226 ms  +3.73%    0.208    0.255
```

### total_minus_ttft_ms median, pool view

```text
n_threads  std median   hpx median   delta       delta %
1          296.796 ms   300.538 ms   +3.743 ms   +1.26%
2          220.489 ms   236.490 ms   +16.001 ms  +7.26%
4          204.109 ms   211.719 ms   +7.609 ms   +3.73%
```

### ttft_ms median, pool view

```text
n_threads  std median   hpx median   delta       delta %
1          445.502 ms   450.861 ms   +5.359 ms   +1.20%
2          318.880 ms   340.977 ms   +22.098 ms  +6.93%
4          283.055 ms   293.406 ms   +10.351 ms  +3.66%
```

### process_wall_ms median, per-trial view

```text
n_threads  std median    hpx median    delta       delta %   std CV   hpx CV
1          2528.322 ms   2590.280 ms   +61.958 ms  +2.45%    0.017    0.034
2          1979.565 ms   2077.892 ms   +98.326 ms  +4.97%    0.024    0.024
4          1804.687 ms   1889.323 ms   +84.636 ms  +4.69%    0.026    0.057
```

## 8. Trend classification

Primary metric:

```text
metric: total_ms median, pool view

delta_pct(n_threads=1) = +1.3172%
delta_pct(n_threads=2) = +7.1435%
delta_pct(n_threads=4) = +3.7294%
range                    =  5.8263%
trend                    = mixed
```

All four trend metrics were classified as mixed:

```text
metric                         n=1      n=2      n=4      range    trend
total_ms (pool)                +1.32%   +7.14%   +3.73%   5.83%    mixed
total_minus_ttft_ms (pool)     +1.26%   +7.26%   +3.73%   6.00%    mixed
ttft_ms (pool)                 +1.20%   +6.93%   +3.66%   5.73%    mixed
process_wall_ms (per-trial)    +2.45%   +4.97%   +4.69%   2.52%    mixed
```

The overhead is non-monotonic:

```text
smallest at n_threads=1
largest at n_threads=2
intermediate at n_threads=4
```

## 9. Interpretation

The sweep does not support a clean oversubscription-only explanation.

If kernel-thread oversubscription were the whole explanation, we would expect HPX overhead to grow monotonically with `n_threads`:

```text
n_threads=1 < n_threads=2 < n_threads=4
```

But the observed trend is:

```text
n_threads=1: small overhead
n_threads=2: largest overhead
n_threads=4: medium overhead
```

The sweep also does not support a flat orchestration-only explanation, because the delta range is much larger than the 1% flat threshold.

The best interpretation is:

```text
The C_2x4 HPX overhead comes from a mixed interaction between HPX serving orchestration / waiter pressure and llama/ggml kernel-thread behavior. It is not explained cleanly by oversubscription alone, and it is not a fixed HPX orchestration overhead independent of n_threads.
```

A more cautious operational reading:

```text
HPX overhead is small at n_threads=1, peaks at n_threads=2, and drops again at n_threads=4. The n_threads=1 result is promising but must be interpreted carefully because ggml may use a different single-thread execution path.
```

## 10. Caveats

```text
Single machine
single CPU topology
single CPU-only build
single model: TinyLlama Q4_K_M
single prompt: "Hello, my name is"
single generation length: 16 tokens
single cell: C_2x4
K=10 measured trials per (n_threads, backend)
```

The sweep keeps waiter pressure constant:

```text
n_contexts=2
n_concurrent=4
```

Lowering `n_threads` reduces kernel-thread pressure, not queueing pressure.

`n_threads=1` may use a different ggml single-thread path. If overhead disappears or shrinks mainly at `n_threads=1`, interpretation must remain cautious.

This is not a general llama.cpp benchmark and not a general HPX scheduler benchmark.

Timing deltas are descriptive. Correctness is the only OVERALL gate.
