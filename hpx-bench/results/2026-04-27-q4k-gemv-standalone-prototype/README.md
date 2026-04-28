# 2026-04-27 — Standalone Q4_K + CPU_REPACK gemv prototype bench

## Goal

Build the smallest possible harness that **demonstrably** runs the same kernel
real TinyLlama Q4_K_M decode runs on M4: `ggml_gemv_q4_K_8x8_q8_K` via the
CPU_REPACK extra path. Use one shape (gate/up: 2048×5632, rows=1), default
threads (4), no HPX, no env vars.

## Harness reality check

Before timing, the bench asserts:
- `W->extra != nullptr` (CPU_REPACK init_tensor populated extra)
- `ggml_repack_extra_traits_name(y) == "q4_K_8x8_q8_K"` (op tensor's src[0] uses
  the gate/up trait)

Run output prints both. All three runs printed:
```
buft: CPU_REPACK, trait: q4_K_8x8_q8_K, extra: 0x...
```
and `y[0] = 0.0177` bit-identical across all runs. The harness IS real — we
are timing the gemv kernel, not the dead-code standard `vec_dot_q4_K_q8_K`.

## Build

| | |
|---|---|
| binary | `build-baseline-no-hpx/bin/bench-cpu-repack-q4k-gemv` |
| build dir | `build-baseline-no-hpx/` |
| CMake flags | `-DGGML_HPX=OFF -DGGML_HPX_REGION_DAG=OFF -DCMAKE_BUILD_TYPE=Release` |
| commit | `e0b332bb5` (working tree dirty: bench source + CMakeLists hook) |
| host | Apple M4 (NEON + matmul-int8 + dotprod) |

Source: `tests/bench-cpu-repack-q4k-gemv.cpp`.
Build hook: 2-line block in `tests/CMakeLists.txt` after `export-graph-ops`.

## Reproduce

```
./build-baseline-no-hpx/bin/bench-cpu-repack-q4k-gemv
```

No flags, no env vars. Hard-coded: cols=2048, out_cols=5632, rows=1, threads=4,
warmup=5, timed=100.

## Results — three back-to-back runs

| run | mean us | stdev us | rel | min us | max us | machine state |
|---|---|---|---|---|---|---|
| 1 | 84.35  | 6.69  | 7.9%  | 78.21 | 110.75 | quiet |
| 2 | 110.10 | 64.66 | 58.7% | 79.46 | 512.83 | system burst |
| 3 | 83.08  | 10.70 | 12.9% | 78.29 | 147.21 | tail of burst |

`y[0] = 0.0177` identical across all three — same work, same numerical result.

### What happened between runs

Pre-run quiet check before run1: nothing over 5% except WindowServer (29%).
Post-run check after run3 showed:

```
74.8% CacheDelete deleted
52.1% triald_system
45.2% cloudd
23.8% pkd
```

These are macOS background maintenance processes that fired between run1 and
run2. Run2 timed during the burst and got a 58% stdev with a 6.5x max-vs-min
outlier. Run3 timed in the tail of the burst, recovered most of the way, but
still 12.9% stdev.

## What this proves

**The kernel floor is rock-solid: ~78us per iter on M4.** That number is the
same across all three runs, regardless of machine state. The kernel itself is
not noisy.

**Mean+stdev with 100 iters of an ~85us kernel is fragile** to per-iteration
preemption events. One 0.5ms hiccup in 100 iters drags the mean by 5%. A
sustained burst of macOS background work shows up as a long tail, not as a
floor shift.

**Min is the most informative single number for a kernel A/B**, especially on
a non-RT macOS host. Median is the second-best (insensitive to upper tail).

## Implications for HPX A/B protocol

Before doing any HPX-vs-baseline comparison on top of this kernel:
- compare *min* per condition, not mean
- run >= 1000 iters to dilute outlier influence on mean/median
- alternate condition order across batches (per `CLAUDE.local.md`)
- record any high-CPU process snapshot taken at run start AND at run end —
  one snapshot before isn't enough; bursts can start mid-run

## Open

- The current bench reports mean+stdev+min+max. Median and percentile
  reporting is not implemented yet. Adding p50/p90 and increasing iters to
  1000 is the next protocol-fix step before doing any HPX comparison.
- Thread pinning / QoS class pinning was not attempted. macOS may schedule
  small bursts to E-cores; pinning to P-cores via `pthread_set_qos_class_self_np`
  could reduce variance but adds platform-specific code.

## Files

- `run1.log` — quiet machine, mean=84.35±6.69 us
- `run2.log` — system burst, mean=110.10±64.66 us
- `run3.log` — tail of burst, mean=83.08±10.70 us
- `README.md` — this file
