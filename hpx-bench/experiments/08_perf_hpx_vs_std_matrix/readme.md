# HPX vs std performance matrix — `llama-serving-bench`

This experiment measures the behavior difference between:

```text
llama-serving-bench --backend std
llama-serving-bench --backend hpx
```

under a controlled CPU-only TinyLlama serving harness.

The matrix was run only after the correctness foundation had already passed:

```text
single-context HPX structural correctness: PASS
(2,2) multi-context capacity correctness: PASS
(2,4) waiter-pressure / no-starvation correctness: PASS
```

This matrix is the first benchmark-grade timing protocol for the HPX serving backend. The matrix-level PASS is a correctness PASS. Timing is descriptive and must be interpreted with variance/CV.

## 1. Central claim

Correctness passed across the full matrix.

```text
MATRIX_OVERALL_CORRECTNESS: PASS
16 / 16 layer summaries: PASS
4 / 4 condition summaries: PASS
496 / 496 raw trials completed
```

Timing interpretation is allowed because all correctness gates passed.

The measured timing result is:

```text
A_1x1: HPX and std are indistinguishable.
B_2x2: HPX shows a small consistent overhead.
C_2x4: HPX overhead is larger under waiter pressure.
D_4x4: signal is mixed and noisy; no reliable speedup claim.
```

The honest conclusion:

```text
HPX correctness is solid. This CPU-only TinyLlama serving-bench harness does not show a reliable HPX speedup. It shows negligible overhead at 1x1, modest overhead at 2x2 and 2x4, and high-noise mixed behavior at 4x4.
```

## 2. Question

What is the measured overhead or behavior difference between `std` and `hpx` backends under controlled serving shapes, after correctness gates already pass?

More specifically:

```text
For each (n_contexts, n_concurrent) cell, what are the median and tail-latency differences between HPX and std on this machine, this build, this model, this prompt, and this generation length?
```

This experiment does not ask whether HPX is generally faster than std. It measures this harness on this machine.

## 3. Matrix shape

Common settings:

```text
binary:         /Users/Ashk/Desktop/HPX/builds/llama-hpx-hpx-on/bin/llama-serving-bench
model:          /Users/Ashk/Desktop/HPX/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
prompt:         "Hello, my name is"
max_tokens:     16
ctx_size:       2048
batch_size:     512
n_threads:      4
seed_base:      1234
canonical hash: 0x833045f1e2ebf49f
```

Cells:

```text
A_1x1: n_contexts=1, n_concurrent=1, n_requests=6
B_2x2: n_contexts=2, n_concurrent=2, n_requests=8
C_2x4: n_contexts=2, n_concurrent=4, n_requests=12
D_4x4: n_contexts=4, n_concurrent=4, n_requests=16
```

Expected HPX lifecycle counts:

```text
A_1x1: os_threads=1, pool_size=1, expected_trace_lines=15
B_2x2: os_threads=2, pool_size=2, expected_trace_lines=19
C_2x4: os_threads=2, pool_size=2, expected_trace_lines=27
D_4x4: os_threads=4, pool_size=4, expected_trace_lines=35
```

## 4. Protocol

The matrix used two layers:

```text
correctness_trace_on
timing_trace_off
```

The correctness layer uses `LLAMA_SERVING_BENCH_HPX_TRACE=1` and gates lifecycle correctness.

The timing layer runs with trace off so timing does not include HPX-only trace `fprintf` overhead.

Trial count:

```text
K=31 process-level trials per backend per cell per layer
trial 0 correctness-checked but excluded from timing aggregation
30 measured timing trials per backend per cell
```

Total raw invocations:

```text
4 cells × 2 layers × 2 backends × 31 trials = 496
```

The full matrix completed across two driver sessions:

```text
first session:   388 RUN entries completed cleanly
resume session:  108 RUN entries completed cleanly, 388 SKIP entries
combined:        496/496 trial directories with bench.exit_code.txt = 0
```

The split across sessions is acceptable because each trial is an independent process invocation from the deterministic schedule.

## 5. Artifacts

Experiment root:

```text
local/baselines/perf_hpx_vs_std_matrix/
```

Important files:

```text
README.md
FACTS.md
RESULTS.md
_schedule.json
schedule_summary.txt
matrix_summary.txt
```

Helper scripts:

```text
_make_schedule.py
_run_one_trial.py
_run_matrix.py
_summarize_layer.py
_summarize_condition.py
_summarize_matrix.py
```

Raw trial tree:

```text
conditions/
  A_1x1/
  B_2x2/
  C_2x4/
  D_4x4/
```

Each condition contains:

```text
correctness/
  std/
  hpx/
timing/
  std/
  hpx/
condition_comparison.txt
```

Each layer/backend directory contains:

```text
trial_00/
...
trial_30/
all_requests.csv
per_trial_summary.csv
condition_summary.txt
```

Top-level summary:

```text
matrix_summary.txt
```

## 6. Correctness result

All summarizers passed:

```text
Layer summaries:     16 / 16 PASS
Condition summaries: 4 / 4 PASS
Matrix summary:      MATRIX_OVERALL_CORRECTNESS: PASS
```

Correctness guarantees checked include:

```text
all harness exit codes are 0
all expected request rows parsed
all aggregate lines match n_ok=N n_cancelled=0 n_error=0
all request statuses are ok
all n_tokens_generated values are 16
all generated_token_hash values equal 0x833045f1e2ebf49f
prompt-fits line present
no serving-bench error/failed diagnostics
HPX lifecycle counts match each cell
std remains HPX-trace-free when required
```

This means timing tables are publishable under the protocol.

## 7. Top-line timing result

### total_ms median, pool view

```text
cell    n_ctx  n_conc  n_req  std median   hpx median   delta       delta %
A_1x1   1      1       6      174.075 ms   174.210 ms   +0.136 ms   +0.08%
B_2x2   2      2       8      234.417 ms   244.556 ms   +10.140 ms  +4.33%
C_2x4   2      4       12     475.062 ms   511.560 ms   +36.498 ms  +7.68%
D_4x4   4      4       16     1217.710 ms  1200.312 ms  -17.398 ms  -1.43%
```

### total_minus_ttft_ms median, pool view

```text
cell    std median   hpx median   delta       delta %
A_1x1   146.084 ms   146.170 ms   +0.087 ms   +0.06%
B_2x2   195.235 ms   203.228 ms   +7.993 ms   +4.09%
C_2x4   200.026 ms   213.831 ms   +13.804 ms  +6.90%
D_4x4   740.449 ms   705.610 ms   -34.839 ms  -4.71%
```

### ttft_ms median, pool view

```text
cell    std median   hpx median   delta       delta %
A_1x1   27.372 ms    27.610 ms    +0.238 ms   +0.87%
B_2x2   39.847 ms    41.182 ms    +1.335 ms   +3.35%
C_2x4   275.767 ms   295.075 ms   +19.308 ms  +7.00%
D_4x4   391.589 ms   410.622 ms   +19.033 ms  +4.86%
```

### process_wall_ms median, per-trial view

```text
cell    std median    hpx median    delta        delta %
A_1x1   1333.215 ms   1350.292 ms   +17.077 ms   +1.28%
B_2x2   1263.887 ms   1335.703 ms   +71.817 ms   +5.68%
C_2x4   1794.560 ms   1906.228 ms   +111.668 ms  +6.22%
D_4x4   5296.890 ms   5487.855 ms   +190.965 ms  +3.61%
```

### aggregate tokens/sec median, per-trial view

```text
cell    std median    hpx median    delta       delta %
A_1x1   91.280 tps    91.035 tps    -0.245      -0.27%
B_2x2   135.175 tps   129.120 tps   -6.055      -4.48%
C_2x4   131.150 tps   123.860 tps   -7.290      -5.56%
D_4x4   51.620 tps    50.425 tps    -1.195      -2.31%
```

### total_ms p95, pool view

```text
cell    std p95       hpx p95
A_1x1   182.116 ms    185.203 ms
B_2x2   276.998 ms    291.206 ms
C_2x4   528.949 ms    600.734 ms
D_4x4   2094.725 ms   2091.855 ms
```

## 8. Interpretation by cell

### A_1x1

HPX and std are indistinguishable.

The total_ms delta is:

```text
+0.136 ms, +0.08%
```

The decode-only delta is:

```text
+0.087 ms, +0.06%
```

This is inside noise.

### B_2x2

HPX shows a small consistent overhead.

```text
total_ms median:             +10.140 ms, +4.33%
total_minus_ttft_ms median:  +7.993 ms, +4.09%
process_wall_ms median:      +71.817 ms, +5.68%
aggregate tokens/sec median: -6.055 tps, -4.48%
```

The sign is consistent across metrics, so this is likely a real modest overhead in this harness.

### C_2x4

HPX overhead grows under waiter pressure.

```text
total_ms median:             +36.498 ms, +7.68%
total_minus_ttft_ms median:  +13.804 ms, +6.90%
ttft_ms median:              +19.308 ms, +7.00%
process_wall_ms median:      +111.668 ms, +6.22%
aggregate tokens/sec median: -7.290 tps, -5.56%
```

This is the clearest overhead cell.

### D_4x4

D_4x4 is mixed and noisy.

Some median metrics favor HPX:

```text
total_ms median:            -17.398 ms, -1.43%
total_minus_ttft_ms median: -34.839 ms, -4.71%
```

But others favor std:

```text
ttft_ms median:             +19.033 ms, +4.86%
process_wall_ms median:     +190.965 ms, +3.61%
aggregate tokens/sec:       -1.195 tps, -2.31%
```

The coefficient of variation is high in D_4x4, so this cell does not support a reliable speedup claim.

## 9. Overall interpretation

Best project-level conclusion:

```text
Across a 496-trial matrix, HPX preserved correctness and lifecycle invariants in every tested shape. In this CPU-only TinyLlama serving-bench harness, HPX performance was indistinguishable from std at 1x1, modestly slower at 2x2 and 2x4, and too noisy to interpret confidently at 4x4. The matrix does not show evidence that HPX improves performance for this workload; it shows that HPX orchestration is correct and its overhead is measurable but modest in the moderate-concurrency cells.
```

Shorter conclusion:

```text
HPX correctness is solid. Performance is not better in this CPU-only harness; the measured effect is mostly small overhead, with high-noise behavior at 4x4.
```

## 10. Caveats

```text
TinyLlama 1.1B Chat Q4_K_M only
single prompt
short generation: 16 tokens
CPU-only build
one Mac / one CPU topology
laptop thermal envelope
serving-bench harness only
not a general llama.cpp benchmark
not a general HPX scheduler benchmark
```

Cells C and D are intentionally oversubscribed:

```text
n_threads=4 per context
C: 2 contexts, 4 concurrent requests
D: 4 contexts, 4 concurrent requests
```

The protocol measures this harness and workload. It does not measure realistic production serving.

Timing deltas are not gates. Correctness is the only matrix OVERALL gate.

