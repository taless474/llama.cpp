# HPX serving-bench manifest

This directory packages the shareable benchmark evidence for the HPX serving-bench work.

The raw trial artifacts remain under `local/baselines/` and are intentionally not copied here. This package keeps the protocol documents, helper scripts, deterministic schedules, compact summaries, and result reports needed to understand and reproduce the experiments without committing bulky raw logs.

In this directory `experiments/` contains the chronological experiment packages. Raw per-trial directories such as `conditions/`, `thread_settings/`, `driver_logs/`, and `trial_*/` are excluded.

## Experiment chronology

| Order | Packaged directory | Source local directory | Status | What it proved | What it did not prove |
|---:|---|---|---|---|---|
| 00 | `experiments/00_server_baseline/` | `local/baselines/server/` | Smoke / setup | Basic llama-server startup, health, metrics, and one request path worked on the local machine. | Did not compare against `llama-serving-bench`; did not establish repeatability or performance. |
| 01 | `experiments/01_server_repeat/` | `local/baselines/server_repeat/` | PASS | llama-server single-request behavior was repeatable enough to use as a baseline reference. | Did not compare HPX or std backends; did not provide benchmark-grade performance evidence. |
| 02 | `experiments/02_server_timing/` | `local/baselines/server_timing/` | PASS | Collected a small llama-server timing baseline with warmup/measured requests and summary output. | Did not provide an apples-to-apples comparison with `llama-serving-bench`; server wall time and harness-internal timing remained different surfaces. |
| 03 | `experiments/03_serving_bench_std_timing/` | `local/baselines/serving_bench_std_timing/` | PASS | Built and ran `llama-serving-bench --backend std`; confirmed deterministic token-hash behavior and harness timing output. | Did not involve HPX; did not yet prove cross-backend equivalence. |
| 04 | `experiments/04_comparison_aligned/` | `local/baselines/comparison_aligned/` | PASS | Established an aligned comparison between llama-server and `llama-serving-bench --backend std`, including prompt/tokenization parity and timing-surface caveats. | Did not claim direct wall-clock equivalence between server and harness; did not test HPX. |
| 05 | `experiments/05_hpx_vs_std_single_context/` | `local/baselines/comparison_hpx_vs_std/` | PASS | Verified HPX-vs-std structural correctness at `n_contexts=1`, `n_concurrent=1`; HPX lifecycle was clean and std stayed HPX-trace-free in the HPX-capable binary. | Did not test multi-context concurrency or waiter pressure; timing was descriptive only. |
| 06 | `experiments/06_hpx_concurrency_2x2/` | `local/baselines/comparison_hpx_concurrency/` | PASS | Verified HPX correctness with `n_contexts=2`, `n_concurrent=2`; both context slots were exercised and canonical hashes matched. | Did not test `n_concurrent > n_contexts`; did not prove waiter behavior under pressure. |
| 07 | `experiments/07_hpx_waiters_2x4/` | `local/baselines/comparison_hpx_waiters/` | PASS | Verified HPX capacity/no-starvation behavior with `n_contexts=2`, `n_concurrent=4`, `n_requests=12`; canonical hashes matched and lifecycle traces were clean. | Did not prove FIFO waiter ordering because queued/wake trace events were not instrumented. |
| 08 | `experiments/08_perf_hpx_vs_std_matrix/` | `local/baselines/perf_hpx_vs_std_matrix/` | PASS | Ran and summarized the full 496-trial HPX-vs-std performance matrix. Correctness passed across all cells and timing interpretation was allowed. | Did not show a reliable HPX speedup. Results apply only to this CPU-only TinyLlama harness, prompt, model, and machine. |
| 09 | `experiments/09_perf_thread_sweep/` | `local/baselines/perf_thread_sweep/` | PASS | Ran and summarized the C_2x4 `n_threads` sensitivity sweep. Correctness passed; timing trend was mixed across `n_threads=1,2,4`. | Did not isolate a single cause for HPX overhead. It ruled out both a clean oversubscription-only explanation and a flat orchestration-only explanation. |
| 10 | `experiments/10_perf_heterogeneous_budgets_design/` | `local/baselines/perf_heterogeneous_budgets/` | Design / pre-run | Designed a future HPX-favoring workload with heterogeneous generation budgets. Identified that per-request `max_tokens` already exists at the request layer and needs a small CLI/call-site feature. | No source change, build, helper scripts, schedule, run, or result yet. This is not performance evidence. |
| 11 | `experiments/11_perf_deep_queue_short_requests/` | `experiments/11_perf_deep_queue_short_requests/runs/` (gitignored, self-contained) | PASS | Verified correctness under deep-queue waiter pressure (`n_contexts=2`, `n_concurrent=32`, `n_requests=200`, `plan=[8]*200`, 16:1 waiter pressure). Canonical budget-8 hash `0x0619d4d1900c2365` matched across std and hpx; HPX correctness layer produced the expected 403-line lifecycle trace per trial; context-pool ids `{0, 1}` exercised. | Did not show an HPX advantage. Short-class median `total_ms` was +2.07% on HPX; p99 was +1.20% (inside the +2.0% threshold); trial makespan was within ±0.14%. Applies only to this CPU-only TinyLlama serving-bench harness, prompt, model, and machine. |
| 12 | `experiments/12_hpx_vs_llama_server_pair/` | `experiments/12_hpx_vs_llama_server_pair/results/` (gitignored, self-contained) | PASS | Built a matched-condition pair-run harness comparing `hpx-server` and `llama-server` over `POST /completion` (non-streaming and SSE). Three independent runs PASS: `20260521-231710-pair` (non-streaming 2×2 matrix, harness `m8d.0`), `20260521-233521-pair` (canonical streaming, 10 repeats, harness `m8e.0`), `20260521-234615-pair` (streaming 2×2 matrix, harness `m8e.0`). Canonical HPX anchors held: `p0_b8` hash `0x0619d4d1900c2365`, `p0_b32` hash `0x6794e47fe0f84af1`. Per-shape stability, EOG-stop handling, and streaming row/anchor gates all PASS; forbidden-words guard clean. | Did not produce any cross-server performance comparison. No aggregation, no percentiles, no averaged TTFT, no throughput. Cross-server `text` equality and `n_decoded` equality are recorded, not gated. `cpp-httplib` remains a non-HPX HTTP adapter boundary. |

## Project-level reports

| Report | Description |
|---|---|
| `docs/perf_matrix_results.md` | Shareable report for the 496-trial HPX-vs-std performance matrix. |
| `docs/perf_thread_sweep_results.md` | Shareable report for the C_2x4 `n_threads` sensitivity sweep. |

## Main result summary

The HPX serving backend passed correctness checks across increasingly difficult workloads:

```text
single-context HPX-vs-std correctness: PASS
2-context concurrency correctness: PASS
2-context / 4-concurrent waiter-pressure correctness: PASS
496-trial performance matrix correctness: PASS
132-trial n_threads sweep correctness: PASS
```

The performance result is conservative:

```text
HPX did not show a reliable speedup in the fixed-shape CPU TinyLlama serving-bench harness.
```

The matrix result was:

```text
A_1x1: HPX and std were indistinguishable.
B_2x2: HPX showed modest overhead.
C_2x4: HPX showed larger overhead under waiter pressure.
D_4x4: behavior was mixed and noisy.
```

The `n_threads` sweep result was:

```text
C_2x4 HPX overhead was smallest at n_threads=1, largest at n_threads=2, and intermediate at n_threads=4.
```

This means the C_2x4 overhead is not explained cleanly by kernel-thread oversubscription alone, and it is not a flat HPX orchestration cost independent of `n_threads`.

## Evidence standard used

Across the packaged experiments, timing interpretation is allowed only after correctness gates pass.

The recurring evidence rules are:

```text
correctness gates first
trace-on layer for HPX lifecycle checks
trace-off layer for timing
canonical or cross-backend token-hash checks
trial 0 excluded from timing aggregation where applicable
no speedup claims from smoke tests
no broad performance claims from one machine / one prompt / one model
```


