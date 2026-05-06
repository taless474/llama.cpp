# RESULTS — HPX vs std performance matrix

Snapshot date: 2026-05-05.  
Branch: `hpx-run-level-analyzer`.  
Latest known HEAD: `63bae8a49`.

## Result status

```text
MATRIX_OVERALL_CORRECTNESS: PASS
```

Raw matrix coverage:

```text
496 / 496 trials completed
16 / 16 layer summaries PASS
4 / 4 condition summaries PASS
```

Timing interpretation is allowed by the protocol because correctness passed across the whole matrix.

## Main conclusion

```text
HPX correctness is solid. Performance is not better in this CPU-only TinyLlama serving-bench harness; the measured effect is mostly small overhead, with high-noise behavior at 4x4.
```

More precise:

```text
Across the full 496-trial matrix, HPX preserved correctness and lifecycle invariants in every tested shape. Performance was indistinguishable from std at 1x1, showed a small consistent overhead at 2x2 and 2x4, and became too noisy to interpret confidently at 4x4 under heavy oversubscription.
```

## Matrix

```text
A_1x1: n_contexts=1, n_concurrent=1, n_requests=6
B_2x2: n_contexts=2, n_concurrent=2, n_requests=8
C_2x4: n_contexts=2, n_concurrent=4, n_requests=12
D_4x4: n_contexts=4, n_concurrent=4, n_requests=16
```

Common setup:

```text
model:      tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
prompt:     "Hello, my name is"
max_tokens: 16
ctx_size:   2048
batch_size: 512
n_threads:  4
seed_base:  1234
canonical hash: 0x833045f1e2ebf49f
```

## Top-line timing table

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

A_1x1 is indistinguishable from std. HPX overhead is effectively noise.

B_2x2 shows a small consistent HPX overhead around 4–6% across median latency, process wall time, and throughput.

C_2x4 shows a larger HPX overhead around 6–8% under waiter pressure.

D_4x4 is mixed and noisy. Some per-request medians favor HPX, but process wall time and throughput favor std. This cell does not support a reliable speedup claim.

## What this means for the project

The HPX serving backend is correct and robust across tested shapes.

The CPU-only TinyLlama harness does not show a performance win for HPX.

The useful engineering result is:

```text
HPX can orchestrate llama-serving-bench correctly with clean lifecycle and context ownership, but its current serving-level orchestration does not outperform the simpler std backend in this small CPU-only workload.
```

## What not to claim

Do not claim:

```text
HPX is faster.
HPX scales better.
HPX improves llama.cpp performance.
D_4x4 proves a speedup.
This result generalizes to larger models or GPUs.
```

## Suggested next step

Package this result and stop the current performance push.

A good next project decision would be one of:

```text
1. Write a concise final report and use this as a portfolio-quality systems experiment.
2. Add queued/wake trace instrumentation to prove FIFO waiter ordering.
3. Design a different workload where HPX could plausibly help:
   - heavier request orchestration
   - heterogeneous prompts
   - cancellation/backpressure
   - batching/pipelining
   - distributed or multi-node scheduling
```
