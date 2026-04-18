# Selective executor 3-case comparison

**Date:** 2026-04-17  
**Commit:** ec4f2b6cae6d7c685c9649e9d5778174e21fe572  
**Binary:** build-hpx-dag/bin/llama-simple  
**Model:** models/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf  
**Flags:** `LLAMA_USE_HPX=1 LLAMA_HPX_SELECTIVE_MUL_MAT=1 LLAMA_HPX_SELECTIVE_STATS=1`

---

## Summary table

| Case | prompt tokens | generated | lowered/graph | fallback/graph | prefill lowered_ms | prefill fallback_ms | decode lowered_ms (avg) | decode fallback_ms (avg) |
|---|---|---|---|---|---|---|---|---|
| decode-heavy | 3 | 64 | 45 | 644 | 3.1 | 272 | ~47 | ~40 |
| small prefill | 16 | 1 | 2 (prefill) / 45 (decode) | 687 / 644 | 3.3 | 126 | n/a (1 decode) | n/a |
| large prefill | 141 | 1 | 2 (prefill) / 45 (decode) | 687 / 644 | 3.0 | 913 | n/a (1 decode) | n/a |

---

## What is actually lowered

**lowered=45 per decode graph, lowered=2 per prefill graph.**

The 45 are the elementwise activation ops in TinyLlama's 22 transformer layers:
- 22 × SILU_F32 (gated MLP gate activation)
- 22 × MUL_F32 (gate × up elementwise product)
- 1 additional F32 op (likely output-path)

Total = 45. These lower successfully because they have no REDUCTION regions and operate on F32 activations.
The reduction guard correctly blocks all RMS_NORM lowering (22 attn_norm + 22 ffn_norm + 1 output_norm).

The prefill graph shows only 2 lowered nodes. This is likely because the prefill batch shape
changes the op types or sizes seen by `ggml_hpx_lower_op` (batched MUL_MAT is Q4_K, not F32).

---

## Interpretation: are we measuring real lowering or fallback correctness?

**lowered > 0 in all cases → the stats plumbing works and real lowering is happening.**

However, there is a structural issue with this test configuration:

The model runs 23/23 layers on Metal (MTL0). The selective path replaces
`ggml_backend_sched_graph_compute_async` entirely, feeding the full 689-node graph
(which includes Metal-assigned nodes) through a single CPU backend. On M4 unified memory,
Metal-mapped buffers have valid CPU pointers, so the CPU backend can attempt to execute
Metal nodes. The result is that all Metal ops run on CPU instead of Metal.

Evidence:
1. All generated tokens are `<unk>` — degenerate output consistent with wrong computation
2. Decode speed is ~10 t/s vs the ~100+ t/s Metal would provide
3. The fallback_ms includes Metal-assigned q4_K MUL_MAT being run on CPU

**What the timing shows:**
- `lowered_ms` (~47ms avg decode) = 45 elementwise ops through HPX fine-region DAG
- `fallback_ms` (~40ms avg decode) = 644 nodes through cpu_be, including all Metal MUL_MAT
- Lowered is slightly *slower* than fallback per-node for decode — HPX dispatch overhead dominates these tiny elementwise ops

**What this run confirms:**
- Stats plumbing is correct
- Live CPU backend routing (no fresh `ggml_backend_cpu_init()` per call) works
- Reduction guard is working (no RMS_NORM lowering attempted)
- No crashes despite running an entirely Metal-assigned graph through the CPU path
- The selective path fires on every graph_compute call regardless of backend assignment

---

## What this run does NOT measure

- Selective lowering benefit vs Metal: the selective path displaces Metal with CPU, so there is no comparison
- Prefill speedup from HPX on large batches: the prefill graph lowered=2 (almost all fallback)
- Correct model quality: `<unk>` output throughout

---

## Required before clean measurement

To measure selective lowering vs CPU-only baseline meaningfully, run with:
- `LLAMA_NO_METAL=1` or `-ngl 0` to force CPU-only backend
- OR gate the selective path to only engage when the scheduled graph has no non-CPU splits

With a CPU-only run, the fallback nodes would be real CPU ops, lowered nodes would be
HPX-dispatched elementwise, and the timing ratio would answer whether HPX dispatch
overhead is worth it for the 45-node elementwise subset.

---

## Open questions

1. Why does the prefill graph show lowered=2 instead of 45? Are SILU/MUL shaped differently in batched mode?
2. Should `hpx_selective_mul_mat` be gated on `graph_splits == 1` (CPU-only graph) to avoid the Metal displacement?
3. For decode, lowered_ms > fallback_ms per-node. Is this pure HPX dispatch overhead or also unified-memory cache effects from the Metal ops running on CPU?
