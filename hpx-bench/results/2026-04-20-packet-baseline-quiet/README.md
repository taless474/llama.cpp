# Exp 5 — Honest PACKET=0 vs PACKET=1 comparison on quiet machine

**Date:** 2026-04-20
**Commit:** 26578ce22 + uncommitted working tree (GLU packet path enabled)
**Binary:** `build-hpx-dag/bin/llama-simple` (single rebuild; same binary for both runs)
**Model:** `models/tinyllama/tinyllama-1.1b-chat-v1.0.F32.gguf`
**Invocation:** `-ngl 0 -n 16 "Hello"` on both runs
**Machine prep:** `pmset -g therm` clean, no processes >10% CPU at run start, VS Code renderer at 20–40% (typical for active editor; not a core hog).

This run is the PACKET=0 baseline. PACKET=1 data is the sibling directory
`2026-04-20-packet-warmup-trace-v2/`.

## Per-token breakdown (n=15 decode tokens)

| bucket            | PACKET=0       | PACKET=1       | delta (0→1)       |
|-------------------|---------------:|---------------:|-------------------:|
| `lowered_ms`      | 213.4 ± 57.2   | 144.2 ± 47.8   | −69.2 ms           |
| `fallback_ms`     |  33.5 ± 14.9   |  34.0 ± 16.3   |  +0.5 ms           |
| `packet_ms`       |   0.0          |  41.4 ± 3.2    | +41.4 ms           |
| **total**         | **246.9 ± 58.9** | **219.6 ± 55.6** | **−27.3 ms**      |

Graph composition:
- PACKET=0: `lowered=222  fallback=467  packet=0`
- PACKET=1: `lowered=156  fallback=467  packet=22(66 nodes)`

## Headline

| metric                          | PACKET=0      | PACKET=1      | speedup  |
|---------------------------------|--------------:|--------------:|---------:|
| eval tok/s (from llama_perf)    | 4.05          | 4.55          | **1.12×** |
| ms/token (from llama_perf)      | 247.12        | 219.79        | **11.1% faster** |
| mean per-token total (computed) | 246.9         | 219.6         | 1.12×    |

**Honest decode speedup for B.1 GLU packet path: ~1.12× (11%).**
Not the 2× claimed in the Apr 18 smoke.

## What the 2× claim was

The Apr 18 smoke reported PACKET=0 at 1.27 tok/s and PACKET=1 at 2.57 tok/s.
On a quiet machine today both runs are significantly faster (4.05 and 4.55
tok/s respectively). The PACKET=0 number benefited more from a quiet machine
than PACKET=1 did, which collapses the apparent speedup from 2× to 1.12×.

Mechanism: PACKET=0 has 66 more nodes going through the lowered path
(`MUL_MAT`, `SiLU`, `MUL`, `GGML_OP_GLU`), which is more sensitive to thread
contention and scheduler jitter. When the machine is contested, PACKET=0
suffers disproportionately, inflating the apparent PACKET=1 win. Remove the
contention and the gap shrinks.

## Why the ~27 ms/token win is believable

- Moving 66 nodes from lowered to packet removed 69.2 ms of lowered work per
  token, at a cost of 41.4 ms of packet dispatch. Net: 27.8 ms saved, matches
  the measured 27.3 ms total delta within rounding.
- `fallback_ms` is unchanged (+0.5 ms) — the packet path does not touch the
  fallback bucket, as expected.
- `packet_ms` stdev is only 3.2 ms on a 41.4 ms mean (8%). Very stable; no
  warmup cliff visible.

## Caveats

- n=15 decode tokens per run. `lowered_ms` stdev is ~25% of its mean, so a
  larger n (e.g. -n 48–64) would tighten the total-ms estimate — but risks
  thermal drift on M4. -n 16 is a reasonable compromise.
- Runs were back-to-back. Baseline ran after packet; we did not measure
  thermal state between runs. Running them in the opposite order on a fresh
  cooled machine and confirming the same 1.12× is a useful sanity check.
- Single sample per condition. Pair-repeating (packet, baseline, packet,
  baseline, ...) a few times and taking medians would reduce run-to-run
  variance.

## Protocol established for future B.1 comparisons

1. Confirm no process >5% CPU except the assistant and editor UI.
2. Confirm `pmset -g therm` shows no warnings.
3. Single rebuild; use the same binary for all conditions in a batch.
4. Same `-n` (recommend 16 on M4; extend only if plateau unclear).
5. Back-to-back runs, alternating condition order across batches.
6. Report mean ± stdev per bucket, not just eval tok/s.
7. Cross-check: `packet_ms` should be low-variance (it's the pure packet
   dispatch). `lowered_ms` will still show ~25% stdev — that's baseline
   scheduler noise, not packet behavior.

## Consequence

The packet-warmup-cliff investigation (Exps 2–4) is downgraded. There is no
observable cliff on a quiet machine. The honest B.1 gain is ~12% decode
speedup on CPU-only F32 TinyLlama — a real but modest win from moving the
MLP_GLU sublayer into a packet.
