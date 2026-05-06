# RESULTS — C_2x4 n_threads sensitivity

## Result status

```text
SWEEP_OVERALL_CORRECTNESS: PASS
```

Raw sweep coverage:

```text
132 / 132 trials completed
12 / 12 layer summaries PASS
3 / 3 thread-setting summaries PASS
```

Timing interpretation is allowed by the protocol because correctness passed across the full sweep.

## Main conclusion

```text
The n_threads sweep produced a mixed trend. HPX overhead was smallest at n_threads=1, largest at n_threads=2, and intermediate at n_threads=4. This does not support a clean oversubscription-only explanation or a flat orchestration-only explanation for the matrix's C_2x4 HPX overhead.
```

## Why this experiment was run

The prior HPX-vs-std performance matrix showed that C_2x4 had the clearest HPX overhead:

```text
C_2x4 at n_threads=4:
total_ms median:            +7.68%
total_minus_ttft_ms median: +6.90%
ttft_ms median:             +7.00%
process_wall_ms median:     +6.22%
```

This sweep asks whether that overhead was mostly due to per-context kernel-thread oversubscription.

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

## Top-line result

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

## Trend classification

All four trend metrics are classified as mixed:

```text
metric                         n=1      n=2      n=4      trend
total_ms                       +1.32%   +7.14%   +3.73%   mixed
total_minus_ttft_ms            +1.26%   +7.26%   +3.73%   mixed
ttft_ms                        +1.20%   +6.93%   +3.66%   mixed
process_wall_ms                +2.45%   +4.97%   +4.69%   mixed
```

## Interpretation

Oversubscription alone does not explain the C_2x4 overhead.

A clean oversubscription explanation would predict overhead increasing as `n_threads` increases:

```text
n_threads=1 < n_threads=2 < n_threads=4
```

The observed pattern is:

```text
n_threads=1: smallest overhead
n_threads=2: largest overhead
n_threads=4: intermediate overhead
```

A fixed HPX orchestration / waiter-path explanation also does not fit because the overhead is not flat across thread counts.

The best cautious interpretation is:

```text
HPX overhead at C_2x4 appears to be a mixed interaction between serving orchestration, waiter pressure, and llama/ggml kernel-thread behavior. The worst point is n_threads=2, not the most oversubscribed n_threads=4.
```

## What this changes from the matrix result

Before this sweep, the C_2x4 matrix result was:

```text
HPX is slower at C_2x4 by about 7%.
```

After this sweep, the better statement is:

```text
HPX overhead at C_2x4 depends on n_threads. It is small at n_threads=1, large at n_threads=2, and moderate at n_threads=4. The overhead is not simply a function of more kernel threads.
```

## Caveats

```text
Single machine
single CPU-only build
single TinyLlama Q4_K_M model
single prompt
single short generation length
single C_2x4 cell
K=10 measured timing trials per thread/backend
```

`n_threads=1` may use a different ggml single-thread path. The small overhead at `n_threads=1` cannot be read as proof that HPX is fine whenever oversubscription is removed.

Waiter pressure is held constant:

```text
n_contexts=2
n_concurrent=4
```

So the sweep reduces kernel-thread pressure, not queueing pressure.

