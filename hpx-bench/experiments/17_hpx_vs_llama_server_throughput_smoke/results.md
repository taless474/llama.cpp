# Experiment 17 — results

Small external throughput smoke: `llama-server` vs `llama-hpx-server`
(default/B=0), TinyLlama 1.1B F32, fixed short prompt, greedy, n_predict=8,
concurrency c ∈ {1, 2}, 20 measured + 1 warmup per cell, non-streaming, one
machine. Run date 2026-05-27. External request-level metrics only.

Both servers exited 0; all 80 measured requests completed with no failures.
hpx produced the canonical greedy b8 hash (`0x0619d4d1900c2365`).

## 1. Throughput

| backend | c | completed | failed | total_tokens | wall_s | tokens/s | req/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| llama-server | 1 | 20 | 0 | 160 | 6.98 | 22.9 | 2.87 |
| llama-server | 2 | 20 | 0 | 160 | 3.73 | 42.9 | 5.36 |
| hpx-B0 | 1 | 20 | 0 | 160 | 6.88 | 23.2 | 2.91 |
| hpx-B0 | 2 | 20 | 0 | 160 | 4.77 | 33.6 | 4.20 |

## 2. Latency

| backend | c | p50_ms | p95_ms | ttft_p50 | ttft_p95 |
|---|---:|---:|---:|---:|---:|
| llama-server | 1 | 344 | 373 | n/a | n/a |
| llama-server | 2 | 371 | 384 | n/a | n/a |
| hpx-B0 | 1 | 340 | 368 | n/a | n/a |
| hpx-B0 | 2 | 467 | 531 | n/a | n/a |

TTFT is not collected (non-streaming run); with budget 8 it would closely track
completion latency.

## 3. Interpretation

- **c=1: tied.** Both backends are within noise of each other — ~23 tokens/s
  and ~340 ms p50. Single-stream decode is dominated by the same underlying
  llama.cpp execution, so there is no meaningful external difference.
- **c=2: llama-server has higher external throughput on this shape** —
  42.9 vs 33.6 tokens/s (and 5.36 vs 4.20 req/s). It scales ~1.87× from c=1
  while hpx scales ~1.45×.
- **HPX c=2 also shows higher latency** — p50 467 / p95 531 ms vs llama-server's
  371 / 384 ms.
- The c=1 parity plus the c=2 gap suggests the next performance investigation
  should target the **c=2 concurrent path — serving-control overhead, batching
  cadence, and wakeups — rather than `llama_decode` itself**, which is shared
  and already tied at c=1. (See the caveat in §4 — the diag-metrics confound
  must be ruled out first.)

## 4. Important caveat — diag metrics enabled on HPX

The hpx-server run was launched with `LLAMA_HPX_DIAG_METRICS=1` (together with
`LLAMA_HPX_DIAG_ENABLE_SHUTDOWN=1`), which was required for the `/shutdown`
endpoint to exit cleanly in this driver. Per-iteration diagnostics may add
overhead on the hpx side, and llama-server had no equivalent instrumentation
enabled. Therefore the c=2 hpx gap is **not yet attributable** to
serving-control logic: a diag-metrics-off rerun (using SIGTERM, or a shutdown
path that does not require the metrics flag) should be the first step of any
follow-up before drawing conclusions about batching cadence or wakeups.

## 5. Non-claims

- Not a production benchmark.
- No claim that HPX is generally faster or slower.
- No Llama 3 (or other model/quantization) generalization.
- No PrefillBudgetPolicy claim (B=0 / default only; no B>0 in this run).
- No streaming / TTFT claim (non-streaming run).
- No Exp14 / Exp15 conclusion.
- HPX internal JSONL is not used as a llama-server comparison metric.

## 6. Artifact references

Raw run (gitignored, not copied into the tree):

```text
local/runs/hpx-vs-llama-server-throughput-2026-05-27/
  per_request.tsv
  summary.tsv
  final_report.md
  plots/throughput_tokens_per_sec.png
  plots/latency_p50_p95.png
  driver.py, analyze.py
  <backend>/c{1,2}/client.jsonl, <backend>/run_meta.json, server.std*
```

## 7. Diag-off and scaling follow-up

Two follow-up runs on 2026-05-27 address the §4 caveat (diag-metrics confound)
and extend concurrency past c=2. Both are still small external request-level
smokes; no servers were rerun for the plots — they are regenerated from the
recorded `summary.tsv` only.

### 7.1 Diagnostics-off control

llama-hpx-server only, B=0, `LLAMA_HPX_DIAG_*` unset (no `diag.jsonl`), SIGTERM
stop, otherwise the §1 shape (20 measured + 1 warmup/cell):

| backend | c | tokens/s | p50_ms | p95_ms |
|---|---:|---:|---:|---:|
| hpx diag-OFF | 1 | 24.1 | 332 | 338 |
| hpx diag-OFF | 2 | 35.3 | 449 | 475 |

Turning diagnostics OFF recovered about **+1.7 tok/s at c=2** (33.6 → 35.3 vs the
§1 diag-ON hpx), i.e. roughly 18% of the original c=2 gap to llama-server. p50/p95
also dropped modestly. So diagnostics **explain part but not most** of the c=2
gap; with the confound removed, 7.6 tok/s of the c=2 gap to llama-server (35.3 vs
42.9) remains.

### 7.2 Scaling c ∈ {1, 2, 4, 8}, diagnostics OFF

Both backends, B=0, diagnostics OFF, capacity matched to c
(llama `--parallel c`, hpx `--n-seq-max c --max-concurrent c`), SIGTERM stop, 40
measured requests/cell (+max(1,c) warmup). **`c` is client concurrency —
requests in flight — NOT a CPU thread count.**

Throughput:

| backend | c | completed | failed | tokens/s | req/s |
|---|---:|---:|---:|---:|---:|
| llama-server | 1 | 40 | 0 | 23.6 | 2.95 |
| llama-server | 2 | 40 | 0 | 38.4 | 4.80 |
| llama-server | 4 | 40 | 0 | 75.5 | 9.44 |
| llama-server | 8 | 40 | 0 | 87.0 | 10.88 |
| hpx-B0 | 1 | 40 | 0 | 23.4 | 2.93 |
| hpx-B0 | 2 | 40 | 0 | 35.7 | 4.46 |
| hpx-B0 | 4 | 40 | 0 | 59.3 | 7.41 |
| hpx-B0 | 8 | 40 | 0 | 82.1 | 10.27 |

Latency:

| backend | c | p50_ms | p95_ms | p99_ms |
|---|---:|---:|---:|---:|
| llama-server | 1 | 338 | 343 | 359 |
| llama-server | 2 | 439 | 461 | 481 |
| llama-server | 4 | 411 | 538 | 540 |
| llama-server | 8 | 738 | 744 | 747 |
| hpx-B0 | 1 | 335 | 375 | 388 |
| hpx-B0 | 2 | 447 | 453 | 464 |
| hpx-B0 | 4 | 538 | 558 | 558 |
| hpx-B0 | 8 | 778 | 781 | 781 |

Scaling efficiency, efficiency(c) = tokens_per_sec(c) / (tokens_per_sec(c=1) · c),
1.0 = ideal linear scaling:

| backend | c=1 | c=2 | c=4 | c=8 |
|---|---:|---:|---:|---:|
| llama-server | 1.00 | 0.81 | 0.80 | 0.46 |
| hpx-B0 | 1.00 | 0.76 | 0.63 | 0.44 |

### 7.3 Interpretation

- **c=1 is tied** (llama 23.6 vs hpx 23.4 tok/s). The shared single-stream
  `llama_decode` path is not the source of any gap.
- **c=2 is a small gap** (38.4 vs 35.7 tok/s, +2.7), consistent with the
  diag-off control: diagnostics removed, the residual c=2 gap persists.
- **c=4 is the largest gap**: llama-server **75.5** vs HPX **59.3** tok/s
  (+16.3). HPX efficiency has dropped to 0.63 here vs llama-server's 0.80, and
  HPX p50 sits at 538 ms vs llama-server's 411 ms.
- **c=8 partially reconverges**: 87.0 vs 82.1 tok/s (gap +4.9), with both
  backends near efficiency ~0.45 (both saturating).
- **Likely next attribution target:** serving-control batching cadence /
  wakeups / request-lifecycle handling under moderate concurrency — i.e. how
  HPX forms and dispatches batches at **c=4** — not `llama_decode`, which is
  shared and tied at c=1.

### 7.4 Plots

Regenerated from `summary.tsv` only (no servers rerun):

```text
local/runs/hpx-vs-llama-server-throughput-scaling-2026-05-27/plots_meeting/
  throughput_vs_concurrency.png
  scaling_efficiency_vs_concurrency.png
  throughput_gap_vs_concurrency.png
  latency_p50_p95_vs_concurrency.png
  normalized_throughput_vs_concurrency.png
```

### 7.5 Non-claims (follow-up)

- Still a small smoke, not a production benchmark.
- No general claim that HPX is faster or slower; no final verdict.
- No Llama 3 (or other model/quantization) generalization.
- No PrefillBudgetPolicy claim (B=0 / default only).
- External request-level metrics only; HPX internal JSONL not used as a
  cross-backend comparison metric.

### 7.6 Follow-up artifact references

```text
local/runs/hpx-vs-llama-server-throughput-2026-05-27/hpx-diag-off-control/
  summary.tsv, final_report.md, per_request.tsv, run_meta.json
  c{1,2}/, server.std*, driver.std*
local/runs/hpx-vs-llama-server-throughput-scaling-2026-05-27/
  summary.tsv, final_report.md, per_request.tsv
  driver.py, analyze.py, make_meeting_plots.py
  <backend>/c{1,2,4,8}/, plots/, plots_meeting/
```
