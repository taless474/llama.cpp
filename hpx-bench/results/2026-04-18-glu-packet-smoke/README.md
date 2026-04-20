# B.1 GLU packet smoke — selective path, CPU-only F32-dequant TinyLlama

**Date:** 2026-04-18
**Commit:** 26578ce22 + Item 5/6 (hpx-prefill-orchestrator branch, uncommitted)
**Binary:** `build-hpx-dag/bin/llama-simple`
**Model:** `models/tinyllama/tinyllama-1.1b-chat-v1.0.F32.gguf` (dequantized from Q4_K_M)
**Flags common to both:** `LLAMA_USE_HPX=1 LLAMA_HPX_SELECTIVE_MUL_MAT=1 LLAMA_HPX_SELECTIVE_STATS=1 -ngl 0 -n 16 "Hello"`

---

## Summary

| Variant | packet_matches | packet_nodes | lowered/graph | packet_ms=0 | eval tok/s |
|---|---|---|---|---|---|
| MLP_PACKET=0 (no packet) | 0 | 0 | 222 | — | **1.27** |
| MLP_PACKET=1 (GLU packet) | 22 | 66 | 156 | ~80 ms avg | **2.57** |

**Decode speedup: ~2.0×**  
**Prefill speedup: 5178 ms → 1327 ms ≈ 3.9×**

---

## What fired

TinyLlama has 22 transformer layers, each using `GGML_OP_GLU[SWIGLU]` for its MLP block
(a fused gate+up+silu node, not the explicit MUL_MAT→SiLU→MUL 4-node pattern).
The GLU packet matcher (`prescan_mlp_glu_matches`) finds the 3-node pattern:

```
MUL_MAT (gate) + MUL_MAT (up) + GGML_OP_GLU[SWIGLU] (trigger)
```

22 matches × 3 nodes = **66 nodes** per decode graph moved from `lowered` to `packet`.
The 4-node gate/up packet never fired on this model (confirmed: `mlp_cache` was still null
for gate/up patterns; `mlp_glu_cache` handled all 22 MLP blocks).

### Node accounting (decode graph, PACKET=1)

| Bucket | Count | Delta vs PACKET=0 |
|---|---|---|
| lowered | 156 | −66 (moved to packet) |
| fallback | 467 | 0 (unchanged) |
| packet | 66 (22 matches) | +66 |

The fallback=467 bucket is unchanged — the packet path only claims the GLU sublayer nodes.

---

## Log excerpts

**PACKET=0 (decode steady-state):**
```
[hpx-selective] lowered=222 fallback=467 packet=0(0 nodes) lowered_ms=355.302 fallback_ms=61.523 packet_ms=0.000
eval time =  11781.58 ms / 15 runs  (785.44 ms/token, 1.27 tok/s)
```

**PACKET=1 (first decode — cache miss + lazy-create):**
```
graph_compute: HPX MLP packet dispatch ready (gate/up + GLU, n_lanes=1)
[hpx-selective] lowered=156 fallback=467 packet=22(66 nodes) lowered_ms=432.841 fallback_ms=48.084 packet_ms=225.830
```

**PACKET=1 (decode steady-state):**
```
[hpx-selective] lowered=156 fallback=467 packet=22(66 nodes) lowered_ms=~230 fallback_ms=~45 packet_ms=~60
eval time =   5830.29 ms / 15 runs  (388.69 ms/token, 2.57 tok/s)
```

---

## Open questions

- The packet_ms range (43–254 ms) is wide; the 254 ms outlier on the last decode token
  is notable. Cache warming / JIT effects or scheduling variance — needs a longer run to
  distinguish.
- lowered_ms in PACKET=1 is still ~230 ms avg (the remaining 156 lowered nodes: RMS_NORM,
  attention ops, etc.). Lowering the remaining HPX-dispatchable ops is future work.
- The gate/up 4-node packet never fires on this model (TinyLlama uses GGML_OP_GLU, not
  the explicit SiLU+MUL pattern). The 4-node path remains exercisable via the isolation
  test only, until a model with that layout is tested.
