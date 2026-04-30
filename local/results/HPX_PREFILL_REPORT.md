# HPX Prefill Findings — Final Report

**Date:** 2026-04-30  
**Commit:** `7f05dcc2b70d8231ff8b10305fb6c2765c15bb2d`  
**Binary:** `build-hpx-dag/bin/llama-simple`  
**Hardware:** Apple M4, 10 logical CPUs, CPU-only (`-ngl 0`)

Raw data:

- `2026-04-30-prefill-hpx-benchmark/` — TinyLlama 1.1B Q4_K_M
- `2026-04-30-prefill-hpx-benchmark-llama31-8b/` — Llama 3.1 8B Q4_K_M

---

## 1. Current HPX prefill region executor

This is the path engaged when `LLAMA_USE_HPX=1` and `n_tokens >= 16`:
`ggml_hpx_exec_run_prefill` in `ggml-hpx-exec.cpp`. It divides the compute graph into regions
(alternating CPU and BLAS blocks) and dispatches them either serially or, with
`LLAMA_HPX_PARALLEL_PROJ=1`, attempts to run Q/K/V chains within large CPU regions as concurrent
HPX tasks.

### Results — 3 runs each, mean ± stdev

#### TinyLlama 1.1B Q4_K_M — 22 layers, GQA-4

| config | tokens | mean ms | stdev ms | tok/s | vs A |
|---|---:|---:|---:|---:|---:|
| A no-HPX | 566 | 1646 | 42 | 344 | — |
| B HPX serial | 566 | 1676 | 11 | 338 | +1.8% |
| C HPX parallel-proj | 566 | 2312 | 12 | 245 | **+40%** |
| A no-HPX | 1130 | 3396 | 30 | 333 | — |
| B HPX serial | 1130 | 3438 | 35 | 329 | +1.2% |
| C HPX parallel-proj | 1130 | 4782 | 68 | 236 | **+41%** |

#### Llama 3.1 8B Q4_K_M — 32 layers, GQA-8

| config | tokens | mean ms | stdev ms | tok/s | vs A |
|---|---:|---:|---:|---:|---:|
| A no-HPX | 472 | 9322 | 184 | 50.6 | — |
| B HPX serial | 472 | 9872 | 253 | 47.8 | +5.9% |
| C HPX parallel-proj | 472 | 13742 | 405 | 34.3 | **+47%** |
| A no-HPX | 942 | 20805 | 1136 | 45.3 | — |
| B HPX serial | 942 | 21041 | 545 | 44.8 | +1.1% |
| C HPX parallel-proj | 942 | 28701 | 539 | 32.8 | **+38%** |

### Interpretation

The current HPX prefill region executor does not provide a useful speedup on either TinyLlama
Q4_K_M or Llama 3.1 8B Q4_K_M CPU-only. HPX serial is roughly equal to or slower than the
scheduler path, while HPX parallel-proj is consistently much slower. Topology dumps show fully
linear region chains for both models, so larger model size increases the number of regions but
does not create exploitable region-level parallelism.

**HPX serial (B):** On TinyLlama the overhead is within noise (±30–42 ms). On Llama 3.1 8B it
is more visible (~550 ms at ~470 tokens, ~3 stdevs) because the region analysis pass over 65
regions and 999 nodes costs more in absolute terms. In both cases the direction is neutral-to-
negative: no run of B was faster than the corresponding run of A.

**HPX parallel-proj (C):** Consistently 38–47% slower across both models and both prompt sizes.
The within-run stdev for C is small (12–539 ms) relative to the excess (665–7896 ms), so the
degradation is not noise — it is stable and model-size-independent. The cause is HPX task
dispatch and future-synchronization overhead applied per-layer: even with GQA-8 (Q/KV ratio 4×)
instead of GQA-4 (8×), the Q projection is still substantially larger than K/V and the critical
path is unchanged.

### Topology

`LLAMA_HPX_DUMP_TOPO=1` confirms the structural cause on both models:

| model | layers | n_nodes | n_regions | sched_splits | chain |
|---|---:|---:|---:|---:|---|
| TinyLlama 1.1B | 22 | 689 | 45 | 1 | fully linear |
| Llama 3.1 8B | 32 | 999 | 65 | 1 | fully linear |

Both models produce an alternating CPU (~27 nodes) / BLAS (4 nodes) pattern with every region
depending on the previous one (`prev = i−1` for all i). The `dependency chain: fully linear —
no independent regions` message appeared in both topology dumps. This is a property of the
transformer architecture on a single CPU-backend forward pass, not a model-size artifact.

---

## 2. Selective lowering / fine-region DAG / frozen packet path

This benchmark does not evaluate selective lowering, fine-region DAG lowering, or frozen packets.
Those paths are structurally unreachable for prefill because the selective block is guarded by
`!batched`, while prefill has `n_tokens > 1` / `rows > 1`. Zero selective/packet stats appeared
in the logs.

Specifically, in `llama-context.cpp`:

```cpp
// line 2261
const bool batched = n_tokens > 1;

// line 2303 — selective path entry
if (hpx_selective_mul_mat && !batched) {   // ← never true for prefill
    ...
    ggml_hpx_exec_graph_selective_mul_mat(...)   // lowering + fine-region DAG
    // prints: lowered_nodes, fallback_nodes, fallback_runs,
    //         packet_matches, packet_nodes, lowered_ms, fallback_ms, packet_ms
}

// line 2365 — packet eligibility
const bool packet_eligible = hpx_mlp_gate_up_packet && !batched;  // ← always false for prefill
```

A search across all 24 run stderr files (12 TinyLlama + 12 Llama 3.1 8B) for `hpx-selective`,
`lowered_nodes`, `fallback_nodes`, `packet_match`, `packet_dispatch` returned **zero hits**.
This is expected: the code paths that emit those strings are inside the `!batched` block and were
never entered.

This result says nothing about whether the selective lowering, fine-region DAG, or packetization
work is correct or useful for decode. Those are evaluated in decode-mode runs where `n_tokens == 1`
and `!batched` is true. The prefill benchmarks here are orthogonal to that work.

---

## 3. Decode packetization results

Decode packetization (selective MUL_MAT, GLU packet, Q4_Kx8 lowering) was developed and tested
separately in decode-mode runs (`n_tokens == 1`). Those results are not recorded in this report.
See the decode benchmark directories and the selective lowering test suite in `tests/hpx/` for
that evidence. The key finding from earlier decode work is that `lookup_or_compile_mlp_glu` fires
on real TinyLlama decode, confirming live packet engagement on the GLU path.

The current state of decode packetization (as of the commit above):

- Selective MUL_MAT lowering: implemented and tested for Q4_K weights on CPU-only decode
- GLU packet path: confirmed live on TinyLlama decode (`LLAMA_HPX_SELECTIVE_MUL_MAT=1`)
- Q4_Kx8 lowering: implemented with CPU_REPACK guard and fallback
- QKV packetization: work stopped (not started for rows > 1; prefill benchmarks confirmed no
  prefill path benefit that would motivate it)

---

## Final recommendation

**Stop current prefill executor work.**  
**Stop QKV packetizing for now.**  
**Write up the decode/selective/packetization findings separately.**

If HPX is revisited later for prefill, it should be at a different granularity: request batching,
multi-sequence scheduling, server-level orchestration, or another design that creates independent
work. Do not continue the current linear region-chain prefill executor. The transformer prefill
graph on a single CPU backend is fully serial at the region level regardless of model size, and
the current HPX layer adds overhead without access to any parallelism.
