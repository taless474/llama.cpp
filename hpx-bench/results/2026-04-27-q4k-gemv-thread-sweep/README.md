# 2026-04-27 — Q4_K_M CPU_REPACK gemv thread-scaling sweep

## Question

Does ggml's CPU_REPACK + `q4_K_8x8_q8_K` gemv path scale well with threads on
Apple M4? Specifically for the gate/up shape that fires 44 times per decode
step on TinyLlama Q4_K_M.

## Setup

| | |
|---|---|
| binary | `build-baseline-no-hpx/bin/bench-cpu-repack-q4k-gemv` |
| build | `-DGGML_HPX=OFF -DGGML_HPX_REGION_DAG=OFF -DCMAKE_BUILD_TYPE=Release` |
| commit | `e0b332bb5` (working tree dirty: bench cpp + CMakeLists hook) |
| host | Apple M4 (4 P-cores + 6 E-cores; NEON + matmul-int8 + dotprod) |
| shape | cols=2048, out_cols=5632, rows=1 (TinyLlama gate/up) |
| iters | 1000 timed (after 10 warmup) |
| harness reality check | `buft=CPU_REPACK, trait=q4_K_8x8_q8_K`, all runs |
| machine state | Time Machine `backupd` was active (~25% combined CPU) |

## Reproduce

```
for t in 1 2 4 8; do
  ./build-baseline-no-hpx/bin/bench-cpu-repack-q4k-gemv $t
done
```

CSV row to stdout, human-readable status to stderr. The bench takes thread
count as `argv[1]` and defaults to 4 if absent.

## Results

| threads | min us | median us | p90 us | mean us | stdev us | rel stdev |
|---|---|---|---|---|---|---|
| 1 | 147.38 | 151.67 | 165.58 | 158.52 | 74.05 | **47%** |
| 2 |  90.58 |  96.75 | 110.67 |  99.99 |  9.88 | 10% |
| 4 |  **71.17** |  **82.33** |  97.96 |  85.39 | 13.53 | 16% |
| 8 | 100.58 | 167.96 | 179.62 | 162.65 | 25.03 | 15% |

`y[0] = 0.0177` identical across all four runs — same numerical work.

### Scaling — speedup relative to threads=1

|  | min | median | mean |
|---|---|---|---|
| 2t | 1.63x | 1.57x | 1.59x |
| 4t | **2.07x** | **1.84x** | **1.86x** |
| 8t | 1.47x | 0.90x | 0.97x |

### Parallel efficiency (speedup / N)

|  | min | median |
|---|---|---|
| 2t | 81% | 78% |
| 4t | 52% | 46% |
| 8t | 18% | 11% |

## What this says

**Scaling is sub-linear and stops at 4 threads.** Going from 1→2→4 threads
gives 1.63x → 2.07x speedup (using the per-iter min, the cleanest signal).
Efficiency drops from 81% to 52% across that range — already losing nearly
half the parallel work to overhead by 4 threads.

**8 threads is a clear regression.** Min jumps from 71us back to 101us, and
median nearly doubles from 82us to 168us. This is the canonical Apple Silicon
heterogeneous-core problem: 4 P-cores + 6 E-cores. ggml's threadpool barrier
waits for the slowest worker; once threads spill onto E-cores (2-3x slower
per core) every barrier-bound chunk pays the E-core latency. For the
gate/up shape with `nrows=1` and tile-grid hand-out via
`ggml_threadpool_chunk_set` + atomic-fetch-add, every iter is barrier-bound.

**threads=1 has high stdev (47%) but stable median.** With a 150us per-iter
floor and 1000 iters, that run lasts ~150ms — long enough that Time Machine's
~25% background CPU usage clipped tail iterations heavily. The mean and stdev
are corrupted by this; the min/median/p90 are not (152us median, 165us p90).
This validates the "report all five, trust min and median" methodology choice.

## Implications for HPX scheduling

These numbers define the floor any HPX work must beat for the gate/up shape:

- 4 threads, **min 71us**, median 82us — that's the bar.
- Anything above 4 threads is dead weight; HPX should not target N > 4 for
  this kernel on M4 (or should be heterogeneity-aware, e.g. NUMA/QoS-aware
  scheduling that avoids E-cores for tight-barrier kernels).
- The 1→2→4 scaling efficiency (81% → 52%) means there's ~half the parallel
  work being lost to **scheduling and barriers** at 4 threads. That's exactly
  the surface area an HPX work-stealing replacement would target.

The reframed strategic question stays the same: can HPX schedule around these
gemv kernels better than `ggml_threadpool_chunk_set` + atomic-fetch-add +
`ggml_barrier`? But now we know the *gap to chase*: **~half the parallel
work lost between 1t and 4t.** If HPX can recover even a fraction of that —
say from 52% efficiency to 70% — the 4-thread min would drop from 71us
toward 53us, and gate/up alone would shave ~800us off every decode step
(44 ops × 18us).

## Open

- Did not measure other shapes (Qcur 2048×2048, ffn_out 5632×2048 q6_K,
  result_output 2048×32000 q6_K). Gate/up was prioritised because it's the
  most-invoked Q4_K shape; expanding to other shapes would tell us whether
  the 4-thread plateau is shape-uniform or shape-dependent.
- Did not measure with the machine fully quiet (Time Machine off).
  threads=1 numbers especially would tighten up.
- 6 threads (4 P + 2 E) was not measured. If there's any value in spilling to
  E-cores at all, it might appear at 5 or 6 before fully degrading by 8.

## Files

- `sweep.csv`  — combined CSV (header + 4 rows)
- `sweep.log`  — combined stderr (4 runs of harness output)
- `README.md`  — this file
