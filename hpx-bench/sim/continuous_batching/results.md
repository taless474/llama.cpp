# Results - continuous batching simulator v1

## Summary

State:

- Phase 2 simulator implemented.
- Pure Python.
- No HPX dependency.
- No llama.cpp / GGUF / real `llama_decode`.
- 13 tests passed, with 9 subtests.
- All 45 exp11 grid combinations completed.
- No invariant failures.
- Deterministic rerun produced the same `run_id` and byte-identical per-policy CSVs.

## What the simulator models

- **Requests** — `(request_id, arrival_time_ms, prompt_tokens, decode_tokens)`. No real text, no logits, no tokens.
- **Slots** — fixed-size pool of `n_slots`. `slot.id` doubles as `seq_id`. One request per slot at a time.
- **Waiting queue** — FIFO; arrived but not-yet-admitted requests.
- **Logical batch rows** — per iteration, the simulator builds a list of `(slot_id, kind=prefill|decode)` rows. PASS A appends one decode row per slot in DECODE phase; PASS B appends prefill rows per slot in PREFILL phase, capped at `n_batch`.
- **Update iterations** — admission → batch build → cost model → row effects → metrics → invariant check. Mirrors upstream `update_slots()` shape.
- **`fifo_context_pool`** — each active slot pays its own decode-call base cost (PER_SLOT cost mode); rows from different slots never share a decode call. Approximates the current `llama-serving-bench` shape.
- **`static_batching`** — admission only when ALL slots are waiting; within a group, rows from multiple slots share one decode call.
- **`continuous_batching`** — admission as soon as any slot frees; rows from multiple slots share one decode call. Prompt and decode rows can coexist in the same iteration.
- **Synthetic cost model** — linear: `iter_cost_ms = base + per_prompt × n_prompt + per_decode × n_decode + per_slot × active_slots + per_stream × n_decode`. Defaults are placeholders; values must be calibrated against a real run before drawing conclusions.

These are scheduling-model results, not measured llama.cpp or HPX performance.

## Test coverage

13 unittest cases (auto-collected by pytest), 9 subtests:

1. one request, one slot — every policy completes a single request
2. two requests, one slot — second request must wait for the first
3. two slots, two requests — both run in parallel, neither waits
4. n_batch smaller than desired rows — prefill correctly chunks across iterations
5. static batching group barrier — admission only when all slots are waiting
6. continuous immediate admission — freed slots pick up new work before peers finish
7. long prefill plus short request mixed iteration — prefill and decode rows coexist when cont_batching is on
8. double-ownership invariant — `assign_request_to_slot` rejects a busy slot
9. deterministic output — same config + seed produces identical completion sequences (continuous + mixed_realistic)
10. tokens_per_decode_call sanity — continuous batching on exp11 with `n_slots > 1` has `tokens_per_decode_call > 1.0`

## Smoke result

Workload: `exp11_like_all_short` (200 requests at `t=0`, P=6, D=8). Config: `n_slots=4`, `n_batch=128`, `seed=42`. Default cost params.

| policy              | makespan_ms | tokens_per_decode_call |
|---------------------|-------------|------------------------|
| fifo_context_pool   | 2660.0      | 1.556                  |
| static_batching     | 1310.0      | 6.222                  |
| continuous_batching | 1310.0      | 6.222                  |

Static and continuous tie on this symmetric all-at-once workload because all requests are identical and there is no staggered arrival or length variance for continuous admission to exploit. Continuous batching's mechanism — admit a freed slot before the rest of the group drains — has nothing to do here, since all four slots finish their identical work at the same iteration.

Run directory: `results/d26c1609bcc4/`.

## Exp11 grid result

Sweep:

- policies: `fifo_context_pool`, `static_batching`, `continuous_batching`
- `n_slots`: 1, 2, 4, 8, 16
- `n_batch`: 32, 128, 512
- 45 (policy × n_slots × n_batch) combinations
- no invariant violations

Continuous matches static across the entire symmetric exp11 grid (their makespan and `tokens_per_decode_call` are identical for every `(n_slots, n_batch)` pair). FIFO loses once `n_slots > 1` because the simulator's cost model charges one `base_decode_step_cost_ms` per active slot per iteration in PER_SLOT mode, while static/continuous pay it once per shared iteration.

This is a consequence of the simulator's cost model and should not be read as a measured llama.cpp speedup.

Selected rows from the sweep summary (full numbers in each `results/<run_id>/summary.csv`):

| n_slots | n_batch | fifo makespan | static makespan | continuous makespan | static tokens/call |
|---------|---------|---------------|------------------|---------------------|--------------------|
| 1       | 128     | 2660.0        | 2660.0           | 2660.0              | 1.556              |
| 2       | 128     | 2660.0        | 1760.0           | 1760.0              | 3.111              |
| 4       | 128     | 2660.0        | 1310.0           | 1310.0              | 6.222              |
| 8       | 128     | 2660.0        | 1085.0           | 1085.0              | 12.444             |
| 16      | 128     | 2660.0        | 977.0            | 977.0               | 23.932             |
| 16      | 32      | 2660.0        | 1014.0           | 979.0               | 18.182             |

The single `n_slots=16, n_batch=32` cell is the one place where continuous and static diverge slightly on this workload: the small `n_batch` cap means a group cannot fully fit into one shared iteration, and continuous admission gets to slip a freed slot into the next iteration earlier than static's group barrier allows. The makespan delta is ~3.5%.

## Mixed realistic result

Workload: `mixed_realistic` with `seed=7`, `total_cap=50`. Config: `n_slots=4`, `n_batch=128`. Default cost params. Run directory: `results/a2fa5ef0ddcd/`.

| policy              | makespan_ms | total_latency p50 | total_latency p95 | total_latency p99 | ttft p50 | ttft p95 | ttft p99 | tokens/call | decode_calls |
|---------------------|-------------|-------------------|-------------------|-------------------|----------|----------|----------|-------------|---------------|
| fifo_context_pool   | 1496.40     | 470.97            | 673.66            | 771.25            | 419.83   | 522.09   | 527.70   | 2.843       | 954           |
| static_batching     | 1120.40     | 264.85            | 354.58            | 371.89            | 214.20   | 326.28   | 350.89   | 4.692       | 578           |
| continuous_batching | 953.77      | 55.73             | 240.44            | 250.24            | 19.38    | 93.34    | 100.07   | 7.194       | 377           |

Under the simulator's cost model, on this 50-request bursty Poisson workload with three length classes, continuous batching has lower makespan, lower latency percentiles, and higher `tokens_per_decode_call` than the other two policies. This separation appears here — and not on `exp11_like_all_short` — because mixed lengths and staggered Poisson arrivals create moments where some slots finish early while others are still working, which is exactly the situation continuous admission is designed to exploit.

These are scheduling-model numbers, not measured llama.cpp performance.

## Determinism

- Same args produced same `run_id`: `d26c1609bcc4`.
- SHA-256 of every per-policy `requests.csv` and `iterations.csv` was byte-identical before and after a re-run.

Hash files retained at `local/sim_hashes_before.txt` and `local/sim_hashes_after.txt`.

## Interpretation

The simulator validates the scheduling model and gives us a safe way to reason about continuous batching before touching llama.cpp or HPX code.

Key lesson:

- Continuous batching is not meaningful just because it exists.
- It helps when workload structure gives it something to exploit: staggered arrivals, heterogeneous prompt/decode lengths, cancellations, or priority classes.

The exp11 grid demonstrates the inverse case: when arrivals and lengths are uniform, continuous and static behave identically because there is nothing to exploit. The `n_slots=16, n_batch=32` cell is the lone exception, where `n_batch` becomes the constraining edge and a different scheduling slack appears.

## Next design implication

The next useful simulator work is not HPX integration yet. It is to analyze mixed/bursty workloads and identify the smallest policy or workload where continuous batching has a clear scheduling advantage over static batching. That gives us a falsifiable target for a future HPX-on-llama.cpp prototype: reproduce the same shape of separation on real hardware before claiming the prototype "works."
