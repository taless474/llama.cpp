# MLP gate/up packet smoke — CPU-only, packet flag on, TinyLlama Q4_K_M

**Date:** 2026-04-17
**Commit:** ec4f2b6ca + uncommitted v1 packet integration (files 1–6)
**Binary:** `build-hpx-dag/bin/llama-simple`
**Model:** `models/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`
**Flags:** `LLAMA_USE_HPX=1 LLAMA_HPX_SELECTIVE_MUL_MAT=1 LLAMA_HPX_SELECTIVE_MLP_PACKET=1 LLAMA_HPX_SELECTIVE_STATS=1`
**Args:** `-ngl 0 -n 16 "Hello"`

---

## What this smoke is for

Prove that enabling the v1 frozen-packet flag on the current smoke model
produces a clean, honest integration boundary. On TinyLlama Q4_K_M the
MLP gate and up projections are Q4_K MUL_MAT, which the lowering layer
rejects, so the packet matcher must find nothing to compile and every
selective-stats line must report `packet=0(0 nodes)`. This is the
predicted outcome documented in `ggml-hpx-exec-selective.h`
("First-deployment scope").

---

## Observed results

- **Exit status:** 0
- **Output:** coherent — `"Hello, World!\n\n5. Python:\n\n\`\`\`python\nprint"` (16 tokens)
- **Packet counters:** every `[hpx-selective]` line shows
  `packet=0(0 nodes) packet_ms=0.000`
- **Lowered / fallback:** unchanged from the previous selective runs
  (pre-packet-flag) on the same model:
  - prefill call (ubatch size 3): `lowered=2 fallback=687`
  - every decode call:           `lowered=45 fallback=644`
- **Lazy-init log:** `HPX MLP gate/up packet dispatch ready (n_lanes=1)`
  appears exactly once — on the first eligible (decode, CPU-only)
  `graph_compute` call.
- **No new error spam, no new warnings, no regressions.**

Representative stats lines:

```
[hpx-selective] lowered=2 fallback=687 packet=0(0 nodes)  lowered_ms=3.464  fallback_ms=59.680 packet_ms=0.000
[hpx-selective] lowered=45 fallback=644 packet=0(0 nodes) lowered_ms=78.073 fallback_ms=54.176 packet_ms=0.000
[hpx-selective] lowered=45 fallback=644 packet=0(0 nodes) lowered_ms=50.205 fallback_ms=49.001 packet_ms=0.000
…
```

---

## Interpretation

The four success criteria for this smoke hold:

| Criterion                                | Result |
|------------------------------------------|--------|
| coherent output                          | ✅ `"Hello, World!…"` |
| no regressions / extra spam              | ✅ one new info line (lazy-init), once |
| `packet=0(0 nodes)` on every stats line  | ✅ 16 decode + 1 prefill |
| existing lowered / fallback unchanged    | ✅ 45/644 decode, 2/687 prefill (matches the 2026-04-17 pre-packet run) |

The v1 flag is wired correctly all the way through:
- env var is read and AND-gated on the selective path
- runtime + cache lazy-create succeeds on first eligible call (single
  INFO line proves this)
- selective executor's matcher runs, finds no compile-able pattern on
  Q4_K_M, returns zero packet dispatches
- per-call stats count the packet bucket as 0 with 0 ms
- destructor cleans up without error (no crash on context teardown)

This matches the "honest integration surface, not an immediate speedup
on TinyLlama Q4_K_M" framing from section 10 / the header doc block.

---

## What this run does NOT prove

- End-to-end packet speedup on llama (no packets actually fired)
- Correctness of packet dispatch on a real llama graph (proven in
  isolation by `test_hpx_selective_mlp_gate_up_packet`, but not yet
  through a real model's 22 MLP layers)
- Behaviour under multi-lane packet execution (v1 is n_lanes=1 only)

Exercising the packet dispatch through llama requires a model with F32
MLP projections. That is the next milestone, outside the scope of this
smoke.

---

## Reproduction

```
LLAMA_USE_HPX=1 \
LLAMA_HPX_SELECTIVE_MUL_MAT=1 \
LLAMA_HPX_SELECTIVE_MLP_PACKET=1 \
LLAMA_HPX_SELECTIVE_STATS=1 \
  build-hpx-dag/bin/llama-simple \
    -m models/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
    -ngl 0 -n 16 "Hello"
```
