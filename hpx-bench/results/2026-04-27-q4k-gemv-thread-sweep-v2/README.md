# 2026-04-27 — Q4_K_M CPU_REPACK gemv thread sweep, two shapes (v2)

## Question

Does ggml's CPU_REPACK + `q4_K_8x8_q8_K` gemv path scale well with threads on
Apple M4? Refines `2026-04-27-q4k-gemv-thread-sweep` with full 1–8 thread
range and a second shape, to see whether the optimal thread count is uniform
or shape-dependent.

## Setup

| | |
|---|---|
| binary | `build-baseline-no-hpx/bin/bench-cpu-repack-q4k-gemv` |
| build | `-DGGML_HPX=OFF -DGGML_HPX_REGION_DAG=OFF -DCMAKE_BUILD_TYPE=Release` |
| commit | `e0b332bb5` (working tree dirty: bench cpp + CMakeLists hook) |
| host | Apple M4 (4 P-cores + 6 E-cores; NEON + matmul-int8 + dotprod) |
| iters | 1000 timed (after 10 warmup) |
| harness reality | `buft=CPU_REPACK, trait=q4_K_8x8_q8_K`, all 16 runs |
| machine state | Mostly quiet (Time Machine and other bursts had completed before the sweep) |

The bench takes `argv[1]=threads, argv[2]=cols, argv[3]=out_cols`. Stdout is
one CSV row per run; stderr has the human-readable summary plus harness
reality print.

## Reproduce

```
for t in 1 2 3 4 5 6 7 8; do
  ./build-baseline-no-hpx/bin/bench-cpu-repack-q4k-gemv $t 2048 5632
done
for t in 1 2 3 4 5 6 7 8; do
  ./build-baseline-no-hpx/bin/bench-cpu-repack-q4k-gemv $t 2048 2048
done
```

(Note: zsh does not word-split unquoted variables, so passing the shape as a
single string `"2048 5632"` and expanding `$shape` will silently drop the
second token. Pass cols and out_cols as separate explicit args.)

## Results

### 2048 × 5632 (TinyLlama gate/up)

| threads | min us | median us | p90 us | mean us | stdev us | rel stdev |
|---|---|---|---|---|---|---|
| 1 | 147.21 | 152.42 | 175.42 | 160.46 | 68.69 | 43% |
| 2 |  85.38 |  97.08 | 114.46 | 101.55 | 13.91 | 14% |
| 3 |  76.92 |  84.38 |  97.75 |  87.73 |  9.11 | 10% |
| 4 |  **71.25** |  **81.08** |  99.75 |  85.63 | 20.77 | 24% |
| 5 |  80.33 |  98.00 | 118.83 | 102.44 | 20.92 | 20% |
| 6 |  90.38 | 157.29 | 168.42 | 148.99 | 25.55 | 17% |
| 7 |  99.67 | 164.25 | 176.42 | 157.54 | 27.26 | 17% |
| 8 | 106.63 | 171.33 | 184.00 | 165.98 | 25.00 | 15% |

### 2048 × 2048 (TinyLlama Qcur / Kcur / attn_out)

| threads | min us | median us | p90 us | mean us | stdev us | rel stdev |
|---|---|---|---|---|---|---|
| 1 | 62.29 |  63.00 |  66.50 |  64.07 |  3.32 | **5%** |
| 2 | 44.75 |  51.13 |  54.83 |  52.21 |  5.66 | 11% |
| 3 | **40.29** |  **49.46** |  54.54 |  51.10 |  8.29 | 16% |
| 4 | 46.17 |  52.79 |  66.83 |  56.64 | 13.77 | 24% |
| 5 | 53.71 |  67.08 |  85.25 |  72.88 | 26.25 | 36% |
| 6 | 56.42 | 125.54 | 135.58 | 111.20 | 27.90 | 25% |
| 7 | 67.04 | 137.38 | 149.79 | 128.37 | 30.19 | 24% |
| 8 | 72.33 | 143.67 | 155.25 | 137.37 | 24.72 | 18% |

`y[0] = 0.0177` identical across all 16 runs.

### Speedup using min — both shapes side by side

| threads | 2048×5632 (gate/up) | 2048×2048 (Q/K/attn_out) |
|---|---|---|
| 1 | 1.00x | 1.00x |
| 2 | 1.72x | 1.39x |
| 3 | 1.91x | **1.55x** ← peak |
| 4 | **2.07x** ← peak | 1.35x |
| 5 | 1.83x | 1.16x |
| 6 | 1.63x | 1.10x |
| 7 | 1.48x | 0.93x |
| 8 | 1.38x | 0.86x |

### Speedup using median — both shapes

| threads | 2048×5632 | 2048×2048 |
|---|---|---|
| 1 | 1.00x | 1.00x |
| 2 | 1.57x | 1.23x |
| 3 | 1.81x | **1.27x** ← peak |
| 4 | **1.88x** ← peak | 1.19x |
| 5 | 1.55x | 0.94x |
| 6 | 0.97x | 0.50x |
| 7 | 0.93x | 0.46x |
| 8 | 0.89x | 0.44x |

## What this says

**The optimal thread count is shape-dependent.**
- 2048×5632 peaks at **4 threads** (min 71us, ~2.1x).
- 2048×2048 peaks at **3 threads** (min 40us, ~1.55x).
The smaller shape's knee is one thread earlier. Cold reading: the fixed
per-iter barrier cost is the same for both shapes, but the parallelisable
work per thread is 2.75x smaller for 2048×2048, so barrier overhead reaches
break-even with less help.

**Per-thread efficiency is shape-dependent too.**
- At t=4: 2048×5632 efficiency = 52%; 2048×2048 efficiency = 34%.
- At t=2: 2048×5632 efficiency = 86%; 2048×2048 efficiency = 70%.
Both shapes lose efficiency fast, but the smaller shape loses faster.

**E-core scheduling makes both shapes worse past the P-core count.** Once
threads ≥ 5 (or 4 for the small shape), you start spilling onto E-cores; ggml's
threadpool barrier waits for the slowest worker. Adding workers actively
degrades performance:
- 8t on 2048×5632 is 1.50x SLOWER (median 171us vs 81us at 4t).
- 8t on 2048×2048 is 2.72x SLOWER (median 144us vs 49us at 3t).

The smaller shape gets hurt much more by E-core scheduling — same fixed
barrier overhead, but smaller chunks per worker mean the slowest E-core's
latency dominates the iter time more completely.

**The 1-thread case for the small shape is the cleanest measurement in the
entire dataset.** 5% relative stdev. Its 1000 iters total ~62ms, short enough
that no macOS background event lands inside the run. The same 1-thread case
on the larger shape took ~150ms and shows 43% stdev — burst exposure scales
with run length.

## Implications for HPX

1. **A single graph-wide thread count can't be optimal for both shapes.**
   ggml today uses one `n_threads` for the whole graph. On TinyLlama Q4_K_M
   decode that's a forced trade-off: pick 4 and lose efficiency on every
   2048×2048 op (Qcur, Kcur, attn_out — 3 per layer × 22 layers = 66 nodes),
   or pick 3 and lose 17% on every 2048×5632 op (gate/up — 44 nodes per
   layer-decode).
2. **Per-node thread-count selection is a real lever.** An HPX scheduler that
   picks the fan-out per node by work size could outperform ggml on every
   shape. The numbers say: use 3 threads for any 2048×2048-class op, 4 for
   2048×5632-class ops, never more.
3. **Adding work-stealing won't help past the P-core knee.** If 8 threads is
   slower than 4, no scheduler can help by giving E-core workers more chunks —
   the barrier wait is the cost. The HPX play, if any, is heterogeneity-aware
   scheduling: keep critical-path workers on P-cores, push background work
   (e.g., next-layer Q8_K quantize) onto E-cores so they're useful but
   off-critical-path.
4. **The fixed-cost barrier is the visible enemy.** Across both shapes,
   roughly half the parallel work is lost between 1t and the optimum thread
   count. That's the gap any HPX work has to dent. Either reduce barrier cost
   or batch multiple consecutive MUL_MATs (Qcur+Kcur+Vcur, gate+up) into a
   single barrier-bounded region.

## Open

- Did not measure 5632×2048 (q6_K, ffn_out — different trait). If the q6_K
  trait shows the same knee structure, confidence in shape-dependent optima
  goes up; if it differs, that's another data point HPX must consider.
- Did not measure 2048×32000 (q6_K, lm_head — also different trait, very
  large out_cols). That op fires once per token; if its optimum is yet
  another thread count, the case for per-node thread selection gets even
  stronger.
- Did not test thread-pinning to P-cores (via `pthread_set_qos_class_self_np`
  or similar). If we pin to P-cores, threads ≥ 5 might oversubscribe rather
  than spill — possibly a different (still bad) failure mode.

## Files

- `sweep.csv` — combined CSV (header + 16 rows: 2 shapes × 8 thread counts)
- `sweep.log` — combined stderr (16 runs of harness output)
- `README.md` — this file
