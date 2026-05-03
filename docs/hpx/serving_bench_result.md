# Serving-bench results

This document records correctness and local benchmark results for
`llama-serving-bench`.

These results are useful as local engineering evidence. They are not final
performance claims.

## Correctness status

Before benchmark interpretation, the serving backend passed the correctness
and lifecycle gates:

```text
OFF Tests 1-7: PASS
HPX-ON Gates 1, 2, 3, 4, 5, 6, 7, 9: PASS
```

Most important structural-fidelity result:

```text
std hash == hpx hash == 0x833045f1e2ebf49f
```

Canonical smoke input:

```text
Model:  TinyLlama 1.1B Chat Q4_K_M
Prompt: Hello, my name is
Decode: greedy
```

Pinned hash values:

```text
max_tokens=16: generated_token_hash=0x833045f1e2ebf49f
max_tokens=0:  generated_token_hash=0x0000000000000000
max_tokens=32: generated_token_hash=0x6794e47fe0f84af1
```

## First local sanity benchmark

This was a single-run sanity check after Slice 5 correctness passed.

Matrix:

```text
A: 1 context / 1 concurrent / 8 requests / 16 tokens
B: 2 contexts / 2 concurrent / 16 requests / 16 tokens
C: 1 context / 4 concurrent / 16 requests / 16 tokens
Backends: std, hpx
Repeats: 1 per cell
```

Result files:

```text
local/bench_small/{A,B,C}_{std,hpx}.{stdout,stderr}
```

All six runs had:

```text
n_error=0
n_cancelled=0
canonical hash counts matched expectations
```

First-pass aggregate throughput:

| Shape | std agg tok/s | HPX agg tok/s |
|---|---:|---:|
| A | 58.87 | 72.61 |
| B | 118.56 | 123.69 |
| C | 65.70 | 74.31 |

This single-shot matrix only showed that HPX had no obvious overhead in
that run. It is not final performance evidence.

## Repeated local sanity benchmark

A repeatable local benchmark was then run with five repeats per cell.

Matrix:

| Shape | n_contexts | n_concurrent | n_requests | max_tokens | expected canonical-hash count |
|---|---:|---:|---:|---:|---:|
| A | 1 | 1 | 8 | 16 | 8 |
| B | 2 | 2 | 16 | 16 | 16 |
| C | 1 | 4 | 16 | 16 | 16 |

Backends: `std`, `hpx`.

Repeats: five per `(shape, backend)` cell, 30 total runs.

Result files:

```text
local/bench_repeat/
```

Correctness summary:

```text
30 stdout files
30 stderr files
all runs: n_cancelled=0 and n_error=0
all A runs: canonical hash count = 8
all B runs: canonical hash count = 16
all C runs: canonical hash count = 16
git status before and after the matrix was identical
```

## Repeated benchmark raw results

### Shape A: 1 context / 1 concurrent / 8 requests

| Run | Backend | wall s | agg tok/s | TTFT p50 | TTFT p95 | total p50 | total p95 | CV |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| r1 | std | 1.54 | 83.29 | 24.91 | 47.95 | 155.82 | 364.40 | 0.2663 |
| r1 | hpx | 1.21 | 105.71 | 22.58 | 36.71 | 142.83 | 178.99 | 0.1012 |
| r2 | std | 1.30 | 98.56 | 20.48 | 32.30 | 135.01 | 246.85 | 0.2188 |
| r2 | hpx | 1.25 | 102.16 | 20.19 | 51.23 | 138.10 | 220.73 | 0.1713 |
| r3 | std | 1.18 | 108.27 | 27.07 | 47.60 | 142.35 | 171.87 | 0.1012 |
| r3 | hpx | 1.12 | 114.70 | 20.37 | 35.93 | 133.67 | 152.10 | 0.0522 |
| r4 | std | 1.41 | 91.10 | 20.21 | 31.74 | 138.02 | 324.29 | 0.2452 |
| r4 | hpx | 1.16 | 110.05 | 20.03 | 31.50 | 140.82 | 167.44 | 0.0842 |
| r5 | std | 1.13 | 113.29 | 20.47 | 34.49 | 137.28 | 150.73 | 0.0540 |
| r5 | hpx | 1.41 | 90.56 | 21.07 | 178.98 | 146.16 | 310.51 | 0.2377 |

### Shape B: 2 contexts / 2 concurrent / 16 requests

| Run | Backend | wall s | agg tok/s | TTFT p50 | TTFT p95 | total p50 | total p95 | CV |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| r1 | std | 1.84 | 138.93 | 36.94 | 47.27 | 229.03 | 240.46 | 0.0272 |
| r1 | hpx | 2.08 | 123.10 | 35.45 | 49.98 | 228.98 | 460.64 | 0.1816 |
| r2 | std | 1.85 | 138.66 | 35.00 | 39.09 | 225.31 | 252.10 | 0.0433 |
| r2 | hpx | 1.93 | 132.56 | 34.31 | 54.84 | 228.61 | 294.38 | 0.1006 |
| r3 | std | 1.85 | 138.52 | 33.74 | 39.58 | 230.03 | 243.69 | 0.0414 |
| r3 | hpx | 1.96 | 130.62 | 34.26 | 58.38 | 230.55 | 306.31 | 0.1114 |
| r4 | std | 1.82 | 140.24 | 32.04 | 38.88 | 216.26 | 261.13 | 0.0828 |
| r4 | hpx | 1.93 | 132.63 | 32.20 | 42.78 | 223.42 | 334.76 | 0.1247 |
| r5 | std | 1.79 | 142.90 | 33.13 | 38.70 | 221.00 | 230.97 | 0.0358 |
| r5 | hpx | 1.93 | 132.52 | 34.78 | 79.41 | 226.88 | 305.32 | 0.1054 |

### Shape C: 1 context / 4 concurrent / 16 requests

| Run | Backend | wall s | agg tok/s | TTFT p50 | TTFT p95 | total p50 | total p95 | CV |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| r1 | std | 2.66 | 96.26 | 493.25 | 630.04 | 620.40 | 751.54 | 0.6519 |
| r1 | hpx | 2.56 | 100.00 | 436.65 | 693.37 | 567.98 | 839.85 | 0.5693 |
| r2 | std | 2.21 | 115.57 | 424.54 | 439.20 | 543.03 | 557.41 | 0.5894 |
| r2 | hpx | 2.87 | 89.15 | 464.98 | 978.56 | 588.44 | 1098.36 | 0.6026 |
| r3 | std | 2.57 | 99.74 | 440.25 | 723.84 | 566.02 | 841.01 | 0.6263 |
| r3 | hpx | 2.36 | 108.62 | 420.31 | 510.57 | 534.39 | 626.28 | 0.5590 |
| r4 | std | 2.55 | 100.56 | 418.74 | 765.91 | 533.50 | 894.07 | 0.6835 |
| r4 | hpx | 2.21 | 115.71 | 420.60 | 436.19 | 539.65 | 550.26 | 0.5428 |
| r5 | std | 2.52 | 101.76 | 430.32 | 745.12 | 548.69 | 860.70 | 0.6755 |
| r5 | hpx | 2.24 | 114.36 | 427.75 | 453.25 | 543.57 | 573.38 | 0.5944 |

## Five-repeat summary

Mean ± sample standard deviation across five repeats:

| Shape | Backend | wall s | agg tok/s | TTFT p50 | TTFT p95 | total p50 | total p95 | CV |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | std | 1.31 ± 0.17 | 98.90 ± 12.24 | 22.63 ± 3.16 | 38.82 ± 8.24 | 141.70 ± 8.33 | 251.63 ± 92.95 | 0.1771 ± 0.0939 |
| A | hpx | 1.23 ± 0.11 | 104.64 ± 9.17 | 20.85 ± 1.05 | 66.87 ± 63.11 | 140.32 ± 4.74 | 205.95 ± 63.76 | 0.1293 ± 0.0746 |
| B | std | 1.83 ± 0.02 | 139.85 ± 1.84 | 34.17 ± 1.88 | 40.70 ± 3.69 | 224.33 ± 5.74 | 245.67 ± 11.48 | 0.0461 ± 0.0214 |
| B | hpx | 1.97 ± 0.06 | 130.29 ± 4.11 | 34.20 ± 1.22 | 57.08 ± 13.79 | 227.69 ± 2.72 | 340.28 ± 68.92 | 0.1247 ± 0.0330 |
| C | std | 2.50 ± 0.17 | 102.78 ± 7.44 | 441.42 ± 30.04 | 660.82 ± 134.34 | 562.33 ± 34.55 | 780.95 ± 135.65 | 0.6453 ± 0.0384 |
| C | hpx | 2.45 ± 0.27 | 105.57 ± 11.07 | 434.06 ± 18.53 | 614.39 ± 227.68 | 554.81 ± 22.79 | 737.63 ± 231.87 | 0.5736 ± 0.0248 |

## Reading the repeated benchmark

The repeated benchmark is mixed, which is useful.

Observed in this local run:

- Shape A: HPX had slightly higher mean aggregate throughput than std,
  but both paths had noisy outliers.
- Shape B: std had higher mean aggregate throughput and tighter p95 total
  latency than HPX.
- Shape C: HPX and std were close on mean aggregate throughput; HPX had
  lower mean TTFT p50 and lower mean total p50, but both paths showed
  high tail variability under queueing.

These are local sanity results only. They should not be described as a
general performance conclusion.

## What this result supports

This result supports the following limited claims:

```text
The HPX backend is structurally correct for the tested matrix.
The repeated benchmark completed without request errors or hash drift.
HPX does not show catastrophic overhead in these small local runs.
The performance picture is workload-dependent and needs broader testing.
```

It does not support:

```text
HPX is faster than std.
HPX is better than upstream llama-server.
The current HPX thread-count formula is optimal.
The current small TinyLlama matrix predicts larger-model behavior.
```

## Next benchmark questions

Useful follow-up benchmark questions:

- Does the pattern hold with longer decode, such as `max_tokens=64`?
- Does the pattern hold with longer prompts?
- Does the shape change with larger request counts?
- Does adding one extra HPX carrier for orchestration improve queueing?
- How does this compare against upstream `llama-server` behavior?
