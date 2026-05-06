# C_2x4 n_threads sensitivity results

This report summarizes the `n_threads` sensitivity sweep for the HPX-vs-std `llama-serving-bench` comparison.

## Summary

The sweep completed and all correctness gates passed.

```text
SWEEP_OVERALL_CORRECTNESS: PASS
12 / 12 layer summaries: PASS
3 / 3 thread-setting summaries: PASS
```

The timing result is mixed:

```text
n_threads=1: HPX overhead is small
n_threads=2: HPX overhead is largest
n_threads=4: HPX overhead is intermediate
```

This does not support a clean oversubscription-only explanation, and it does not support a flat orchestration-only explanation.

## Context

The prior performance matrix showed HPX overhead at C_2x4 with `n_threads=4`:

```text
total_ms median:            +7.68%
total_minus_ttft_ms median: +6.90%
ttft_ms median:             +7.00%
process_wall_ms median:     +6.22%
```

The sweep tested whether that overhead persists when per-context kernel threads are reduced.

## Fixed shape

```text
n_contexts=2
n_concurrent=4
n_requests=12
max_tokens=16
ctx_size=2048
batch_size=512
seed_base=1234
canonical hash=0x833045f1e2ebf49f
```

Sweep variable:

```text
n_threads ∈ {1, 2, 4}
```

## Results

### total_ms median

```text
n_threads  std median   hpx median   delta       delta %
1          741.918 ms   751.691 ms   +9.773 ms   +1.32%
2          541.571 ms   580.258 ms   +38.687 ms  +7.14%
4          488.700 ms   506.925 ms   +18.226 ms  +3.73%
```

### total_minus_ttft_ms median

```text
n_threads  std median   hpx median   delta       delta %
1          296.796 ms   300.538 ms   +3.743 ms   +1.26%
2          220.489 ms   236.490 ms   +16.001 ms  +7.26%
4          204.109 ms   211.719 ms   +7.609 ms   +3.73%
```

### ttft_ms median

```text
n_threads  std median   hpx median   delta       delta %
1          445.502 ms   450.861 ms   +5.359 ms   +1.20%
2          318.880 ms   340.977 ms   +22.098 ms  +6.93%
4          283.055 ms   293.406 ms   +10.351 ms  +3.66%
```

### process_wall_ms median

```text
n_threads  std median    hpx median    delta       delta %
1          2528.322 ms   2590.280 ms   +61.958 ms  +2.45%
2          1979.565 ms   2077.892 ms   +98.326 ms  +4.97%
4          1804.687 ms   1889.323 ms   +84.636 ms  +4.69%
```

## Trend

All four trend metrics were classified as mixed.

```text
metric                         n=1      n=2      n=4      trend
total_ms                       +1.32%   +7.14%   +3.73%   mixed
total_minus_ttft_ms            +1.26%   +7.26%   +3.73%   mixed
ttft_ms                        +1.20%   +6.93%   +3.66%   mixed
process_wall_ms                +2.45%   +4.97%   +4.69%   mixed
```

## Interpretation

The sweep rules out a simple explanation.

It is not cleanly oversubscription-only, because overhead does not increase monotonically with `n_threads`.

It is not flat orchestration-only overhead, because overhead changes significantly across thread settings.

The best reading is:

```text
C_2x4 HPX overhead appears to come from an interaction between HPX serving orchestration / waiter pressure and llama/ggml kernel-thread behavior. The interaction is worst at n_threads=2 in this run.
```

The small overhead at `n_threads=1` is interesting but not conclusive because `n_threads=1` may use a different ggml single-thread path.

## Project implication

The previous performance matrix showed HPX had modest overhead at C_2x4. This sweep shows that the overhead is sensitive to llama kernel-thread configuration and is not simply caused by “too many threads.”

This makes the project result more nuanced:

```text
HPX serving correctness is robust. Performance overhead depends on the interaction between serving orchestration, waiter pressure, and backend kernel-thread behavior. The current CPU-only harness still does not show a reliable HPX speedup.
```

## Caveats

```text
single machine
single CPU-only build
single TinyLlama Q4_K_M model
single prompt
16 generated tokens
single C_2x4 cell
K=10 measured trials per thread/backend
```

Waiter pressure remains constant throughout the sweep:

```text
n_contexts=2
n_concurrent=4
```

Lowering `n_threads` reduces kernel-thread pressure, not queueing pressure.
