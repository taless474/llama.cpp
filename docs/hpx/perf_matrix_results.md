# HPX serving performance matrix results

This report summarizes the HPX-vs-std `llama-serving-bench` performance matrix on the `hpx-run-level-analyzer` branch.

## Summary

The full 496-trial matrix completed and all correctness gates passed.

```text
MATRIX_OVERALL_CORRECTNESS: PASS
16 / 16 layer summaries: PASS
4 / 4 condition summaries: PASS
```

The timing result does not show an HPX speedup in this CPU-only TinyLlama harness. It shows:

```text
A_1x1: no meaningful difference
B_2x2: small HPX overhead
C_2x4: larger HPX overhead under waiter pressure
D_4x4: mixed / noisy, no reliable speedup claim
```

## Experimental setup

Branch:

```text
hpx-run-level-analyzer
```

Latest known HEAD:

```text
63bae8a49
```

Model:

```text
tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

Prompt:

```text
Hello, my name is
```

Generation:

```text
max_tokens=16
greedy / argmax
CPU-only
```

Canonical generated-token hash:

```text
0x833045f1e2ebf49f
```

## Protocol

The matrix used two layers:

```text
correctness_trace_on
timing_trace_off
```

The correctness layer gates hash, request status, token count, and HPX lifecycle traces.

The timing layer runs with trace off to avoid including HPX-only trace logging overhead.

Trial count:

```text
31 process-level trials per backend per cell per layer
trial 0 correctness-checked but excluded from timing
30 measured timing trials per backend per cell
```

Total invocations:

```text
4 cells × 2 layers × 2 backends × 31 trials = 496
```

Cells:

```text
A_1x1: n_contexts=1, n_concurrent=1, n_requests=6
B_2x2: n_contexts=2, n_concurrent=2, n_requests=8
C_2x4: n_contexts=2, n_concurrent=4, n_requests=12
D_4x4: n_contexts=4, n_concurrent=4, n_requests=16
```

## Results

### total_ms median

```text
cell    std median   hpx median   delta       delta %
A_1x1   174.075 ms   174.210 ms   +0.136 ms   +0.08%
B_2x2   234.417 ms   244.556 ms   +10.140 ms  +4.33%
C_2x4   475.062 ms   511.560 ms   +36.498 ms  +7.68%
D_4x4   1217.710 ms  1200.312 ms  -17.398 ms  -1.43%
```

### total_minus_ttft_ms median

```text
cell    std median   hpx median   delta       delta %
A_1x1   146.084 ms   146.170 ms   +0.087 ms   +0.06%
B_2x2   195.235 ms   203.228 ms   +7.993 ms   +4.09%
C_2x4   200.026 ms   213.831 ms   +13.804 ms  +6.90%
D_4x4   740.449 ms   705.610 ms   -34.839 ms  -4.71%
```

### process_wall_ms median

```text
cell    std median    hpx median    delta        delta %
A_1x1   1333.215 ms   1350.292 ms   +17.077 ms   +1.28%
B_2x2   1263.887 ms   1335.703 ms   +71.817 ms   +5.68%
C_2x4   1794.560 ms   1906.228 ms   +111.668 ms  +6.22%
D_4x4   5296.890 ms   5487.855 ms   +190.965 ms  +3.61%
```

### aggregate tokens/sec median

```text
cell    std median    hpx median    delta       delta %
A_1x1   91.280 tps    91.035 tps    -0.245      -0.27%
B_2x2   135.175 tps   129.120 tps   -6.055      -4.48%
C_2x4   131.150 tps   123.860 tps   -7.290      -5.56%
D_4x4   51.620 tps    50.425 tps    -1.195      -2.31%
```

## Interpretation

The key engineering finding is correctness, not speed.

HPX preserved lifecycle and context-ownership invariants across all tested cells. The canonical generated-token hash was preserved across the matrix, and std stayed HPX-trace-free where required.

The timing result is more conservative:

```text
No reliable speedup was observed.
A_1x1 is indistinguishable.
B_2x2 and C_2x4 show modest HPX overhead.
D_4x4 is too noisy to interpret as a speedup.
```

This suggests that HPX serving-level orchestration is correct but does not improve performance in this small CPU-only workload.

## Caveats

This result is specific to:

```text
one machine
one CPU-only build
one small quantized model
one short prompt
one short generation length
one serving-bench harness
```

It is not a general llama.cpp benchmark and not a general HPX scheduler benchmark.

Cells C and D are oversubscribed and should be interpreted as stress shapes, not recommended deployment configurations.
