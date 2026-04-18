# Milestone A — packet dispatch activation proof (CPU-only, F32 weights)

**Date:** 2026-04-17
**Commit:** ec4f2b6ca + uncommitted v1 packet integration (files 1–6) + uncommitted A milestone artifacts
**Binary:** `build-hpx-dag/bin/llama-simple`
**Model:** `models/tinyllama/tinyllama-1.1b-chat-v1.0.F32.gguf` (produced via A.1, see below)
**Flags:** `LLAMA_USE_HPX=1 LLAMA_HPX_SELECTIVE_MUL_MAT=1 LLAMA_HPX_SELECTIVE_MLP_PACKET=1 LLAMA_HPX_SELECTIVE_STATS=1`
**Args:** `-ngl 0 -n 16 "Hello"`

---

## How the F32 model was produced (A.1)

Dequantized from the existing Q4_K_M GGUF using the stock `llama-quantize`
tool with `--allow-requantize`:

```
build-hpx-dag/bin/llama-quantize --allow-requantize \
    models/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
    models/tinyllama/tinyllama-1.1b-chat-v1.0.F32.gguf \
    F32
```

Produced in 5.3 s.  Input 636 MiB → output 4196 MiB (32.00 BPW).

**Important scope note:** this is a dequantization, not a recovered F32
baseline.  Every MLP `ffn_gate.weight` and `ffn_up.weight` tensor was
converted from `q4_K` to `f32` (logged line by line during quantize).
The *values* are still the Q4_K_M-rounded ones — only the tensor dtype
changes.  The generation quality of this F32 GGUF matches Q4_K_M; the
only thing it buys is that every MUL_MAT's src[0] is now F32-typed, which
is what `ggml_hpx_lower_op` (and therefore `ggml_hpx_compose_mlp_gate_up_group`)
requires.

This GGUF is for the packet-dispatch activation proof only.  It is NOT a
performance or accuracy baseline.

---

## Observed result

- **Exit status:** 0
- **Output:** coherent — `"Hello, World!\n\n5. Python:\n\n\`\`\`python\nprint"` (16 tokens, matches the Q4_K_M smoke)
- **Lazy-init log:** `HPX MLP gate/up packet dispatch ready (n_lanes=1)` fires once, on the first decode-side `graph_compute`
- **Per-call stats (representative):**

```
[hpx-selective] lowered=157 fallback=532 packet=0(0 nodes) lowered_ms=4042.7  fallback_ms=101.4 packet_ms=0.000    (prefill)
[hpx-selective] lowered=200 fallback=489 packet=0(0 nodes) lowered_ms=4375.8  fallback_ms= 83.2 packet_ms=0.000    (decode 1)
[hpx-selective] lowered=200 fallback=489 packet=0(0 nodes) lowered_ms=4419.2  fallback_ms= 67.5 packet_ms=0.000    (decode 2)
…
```

### Comparison to the Q4_K_M baseline at the same flags

| run                | prefill `lowered` | decode `lowered` | decode `fallback` | `packet_matches` |
|--------------------|------------------:|-----------------:|------------------:|-----------------:|
| Q4_K_M (prior run) |                 2 |               45 |               644 |                0 |
| F32 (this run)     |               157 |              200 |               489 |                0 |

The jump in `lowered` confirms that switching to F32 weights made the MLP
MUL_MATs (and all other F32-typed MUL_MATs) lower-able: 45 → 200 per
decode is roughly +155 nodes, consistent with enabling ~22 layers of
attention + FFN MUL_MAT lowering on top of the pre-existing 45
elementwise ops.

**But `packet_matches` stayed at 0.**

---

## Why packet dispatch did not fire

The v1 matcher (`prescan_mlp_matches` in
`ggml-hpx-exec-selective.cpp`) looks for the following node shape at
every `GGML_OP_MUL`:

```
mul.src[0]       = GGML_OP_UNARY with subop GGML_UNARY_OP_SILU (silu)
  silu.src[0]    = GGML_OP_MUL_MAT                               (gate)
mul.src[1]       = GGML_OP_MUL_MAT                               (up)
  gate.src[1] == up.src[1]                                       (shared x)
```

That pattern exists in the isolation test
(`test_hpx_selective_mlp_gate_up_packet.cpp`), which constructs gate +
up + silu + mul as four separate ggml ops.  It does **not** exist in real
llama graphs.

`src/llama-graph.cpp:1074` calls:

```
cur = ggml_swiglu_split(ctx0, cur, tmp);
```

which emits a **single** `GGML_OP_GLU` node with subop
`GGML_GLU_OP_SWIGLU` in place of the `silu(gate) → mul(silu, up)` tail.
The fused GLU op is a ggml-level optimization that predates the frozen-
packet work.

So the real per-layer MLP tail in TinyLlama's graph is:

```
gate = MUL_MAT(W_gate, x)
up   = MUL_MAT(W_up,   x)
cur  = GLU[SWIGLU](gate, up)           <-- one node, not three
out  = MUL_MAT(W_down, cur)
```

and no fragment of that contains the 3-op `(MUL_MAT, SiLU, MUL)` tail the
matcher scans for.  `packet_matches` is therefore 0 by construction.

---

## What this milestone proves

- The v1 end-to-end integration path runs on a real model without
  regression: env flag honoured, lazy creation fires once, matcher runs
  on every decode graph, selective stats include the packet bucket,
  teardown clean.
- Switching MLP weights from Q4_K to F32 does materially change lower_op
  acceptance — `lowered` goes from 45 → 200 per decode.  The fine-region
  path works on a realistic graph once the dtype constraint is lifted.
- Packet dispatch is mechanism-complete but shape-incomplete for real
  llama.  The gap is a ggml-level op fusion, not an HPX-layer bug.

## What it does NOT prove (and was never scoped to)

- Packet dispatch actually firing in llama — did not happen on this
  model because the graph does not contain the v1-assumed pattern.
- Any speedup claim.

---

## Closing A and scoping B

The activation-proof goal was: *does `packet_matches > 0` in real
llama?*  The honest answer on TinyLlama at ec4f2b6ca is: **no, because
the graph has no 3-op SiLU+MUL tail**.  The v1 integration is sound;
the matcher's pattern is simply narrower than any real llama graph.

This was worth discovering as its own milestone.  It converts B from
*"widen lowering/composition so quantized models can reach packet
dispatch"* into something more precise:

**B — realistic workload path, specifically including:**

1. Recognize the fused `GLU[SWIGLU]` op as the semantic equivalent of
   `silu(gate) × up`.  The matcher either gains a second shape (two
   MUL_MAT + one GLU) or the composer gains a GLU-aware variant.
2. A new sublayer id (e.g. `GGML_HPX_PACKET_SUBLAYER_MLP_GATE_UP_GLU_F32
   = 3`) with its own compose + compile path, so the plan key stays
   structurally distinct from the existing 4-op shape.
3. Q4_K-aware `MUL_MAT` lowering (the original B scope), so dequant is
   not required just to activate.  This is the substantive payoff —
   once both (1) and (3) land, quantized TinyLlama will actually fire
   packet dispatch with no model conversion needed.

A and B together close the full Section 7–9 provenance arc from frozen
packet → real llama workload.  This README closes A.

---

## Reproduction

```
# 1. Produce the F32 GGUF (one-time, ~5 s, ~4 GB output):
build-hpx-dag/bin/llama-quantize --allow-requantize \
    models/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
    models/tinyllama/tinyllama-1.1b-chat-v1.0.F32.gguf \
    F32

# 2. Smoke run:
LLAMA_USE_HPX=1 \
LLAMA_HPX_SELECTIVE_MUL_MAT=1 \
LLAMA_HPX_SELECTIVE_MLP_PACKET=1 \
LLAMA_HPX_SELECTIVE_STATS=1 \
  build-hpx-dag/bin/llama-simple \
    -m models/tinyllama/tinyllama-1.1b-chat-v1.0.F32.gguf \
    -ngl 0 -n 16 "Hello"
```
