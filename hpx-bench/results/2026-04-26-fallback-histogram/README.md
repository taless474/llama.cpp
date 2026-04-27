# Fallback-Bucket Histogram on TinyLlama Q4_K_M Decode (Step 2)

**Date:** 2026-04-26
**Base commit:** `e0b332bb5` ("Guard Q4_K lowering against repacked CPU tensors")
**Working tree:** dirty — Step 1 (selective guard) + Step 2 (histogram) uncommitted:
- `ggml/src/ggml-hpx/ggml-hpx-exec-selective.h` — `should_engage` + `log_node_histogram` declarations
- `ggml/src/ggml-hpx/ggml-hpx-exec-selective.cpp` — both implementations
- `src/llama-context.cpp` — second skip arm + histogram call

**Branch:** `hpx-prefill-orchestrator`
**Binary:** `build-hpx-dag/bin/llama-simple`
**Model:** `models/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`
**Invocation:** `-ngl 0 -n 16 "Hello"`
**Env:** `LLAMA_USE_HPX=1 LLAMA_HPX_SELECTIVE_MUL_MAT=1 LLAMA_HPX_SELECTIVE_HIST=1`
**Machine:** Apple M4

## Short answer

Implementing repacked Q4_Kx8 lowering (Step 3 / Option B) would convert **134
of 155 MUL_MATs (87%)** in TinyLlama Q4_K_M decode from fallback to the
lowered HPX path. Non-MUL_MAT fallbacks are dominated by 264 metadata
ops (VIEW/RESHAPE/PERMUTE) that are essentially free and ~200 light ops
(ADD/MUL/RMS_NORM/ROPE) that the existing CPU path already handles
efficiently. **B is the right next step**; widening non-MUL_MAT lowering
coverage is not gating M4 perf.

## Histogram (n_nodes = 689)

| bucket                       | count | nature                                  |
|------------------------------|------:|-----------------------------------------|
| **MUL_MAT / Q4_K-repacked**  | **134** | **discriminating — would move to lowered** |
| MUL_MAT / Q4_K-nonrepacked   |     0 | (impossible on M4 — repack always fires) |
| MUL_MAT / F32×F32            |     0 |                                          |
| MUL_MAT / other-quant        |    21 | likely Q6_K (attn_v + ffn_down + tok_embd / output head) |
| MUL_MAT / other              |     0 |                                          |
| ADD                          |    44 | light, vectorized                        |
| MUL                          |    45 | light, vectorized                        |
| RMS_NORM                     |    45 | light                                    |
| CPY                          |     1 | metadata-ish                             |
| RESHAPE                      |    88 | metadata, free                           |
| VIEW                         |   110 | metadata, free                           |
| PERMUTE                      |    66 | metadata, free                           |
| GET_ROWS                     |     3 | embedding lookup                         |
| SET_ROWS                     |    44 | KV-cache writeback                       |
| ROPE                         |    44 | rotary, fused                            |
| FLASH_ATTN_EXT               |    22 | fused attention compute                  |
| GLU                          |    22 | SWIGLU, fused                            |

Subtotal MUL_MAT: 155 (134 repacked Q4_K + 21 other-quant). 22 layers ×
~7 projection matmuls/layer ≈ 154, matching the count.

Sum check: 155 + 44 + 45 + 45 + 1 + 88 + 110 + 66 + 3 + 44 + 44 + 22 + 22 = 689. ✓

## Decomposition by mass

The 134 repacked Q4_K MUL_MATs are the body projections — Wq, Wk, Wo per layer
plus the FFN gate/up/down (typical Q4_K_M recipe pushes attn_v and ffn_down to
Q6_K, which lands them in the "other-quant" bucket). These matrices are
[hidden_dim × hidden_dim] = 2048 × 2048 = 4.2M params each (or 2048 × 5632 =
11.5M for the FFN). They dominate the FLOPs of one decode step.

Non-MUL_MAT fallback nodes break down as:
- **264 metadata** (VIEW + RESHAPE + PERMUTE): no compute, just stride games.
  These are not real fallback cost — ggml processes them in microseconds.
- **~200 light compute** (ADD + MUL + RMS_NORM + ROPE + GLU = 200): vectorized,
  per-token cost is small. Lowering them in HPX would parallelize them, but
  decode-row work (rows = 1) limits the parallel speedup.
- **22 FLASH_ATTN_EXT**: this is the only non-MUL_MAT op with non-trivial per-call
  cost. A fused kernel that processes the full attention block per layer.
  Already well-optimized in ggml-cpu.
- **44 SET_ROWS + 3 GET_ROWS + 1 CPY**: bookkeeping for the KV cache + embedding
  lookup, small.

Conclusion: there is no large hidden cost in non-MUL_MAT fallbacks. The
selective-tax problem we hit in B4 was per-node *dispatch* overhead amplified
by 622 graph_compute calls into HPX threadpool, not per-node *compute*
overhead. Whole-graph compute through the same threadpool (the post-Step-1
behaviour) handles all of these without issue.

## What this means for Step 3

**Worth doing.** 87% of the heavy MUL_MAT mass is in one bucket, gated by one
specific layout decision (`q4_K_8x8_q8_K`). Once HPX can read that layout,
selective will own the body matmul pipeline on M4 quantized models, and the
Step 1 guard will stop firing on the production path.

**Not blocked by anything else.** Non-MUL_MAT lowering coverage doesn't need
to expand — the existing CPU fallback handles all 200 light compute nodes
efficiently when run as part of the whole graph. After B, selective would
own the matmuls, and the remaining ~50 light non-metadata fallbacks per token
would dispatch at one-node-per-call into HPX for the duration of the layer's
non-matmul prefix/suffix. That's still ~50 dispatches per token — small enough
that it likely won't reintroduce a perceptible regression, but it should be
re-measured once B is in.

**Open scoping question for Step 3:** the 21 "other-quant" matmuls. If
TinyLlama Q4_K_M's attn_v and ffn_down are Q6_K, those won't benefit from
Q4_Kx8 work alone — they need a separate repacked-Q6_K path or a more general
"any quantized × Q8_K" lowering. Not gating Step 3, but worth scoping before
starting: do we land Q4_Kx8 first and accept that ~14% of matmuls still fall
back, or expand the work to cover Q6_K too?

## Reproducing

```sh
LLAMA_USE_HPX=1 LLAMA_HPX_SELECTIVE_MUL_MAT=1 LLAMA_HPX_SELECTIVE_HIST=1 \
  ./build-hpx-dag/bin/llama-simple \
  -m models/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  -ngl 0 -n 16 "Hello"
```

The histogram is warned-once per process — one decode produces one block.
Set the env to `0` (or unset it) to disable the diagnostic entirely.

Files in this directory:
- `histogram.txt` — extracted histogram block, easy to grep
- `bench.log` — full stderr from the run including the histogram
- `histogram_shapes.txt` — shape-weighted view (per-MUL_MAT-bucket shape multiplicities)
- `bench_shapes.log` — full stderr from the shape-weighted run

## Update: shape-weighted view

The histogram diagnostic was extended to record `(cols × out_cols × rows)` for
every MUL_MAT, dedup'd per bucket. This answers the scoping question "are
the 21 other-quant nodes small residuals or large tensors like ffn_down?"

### Q4_K-repacked shapes (134 nodes)

| cols | out_cols | rows | count | most likely identity                |
|-----:|---------:|-----:|------:|-------------------------------------|
| 2048 |     2048 |    1 |    44 | Wq + Wo per layer (2 × 22)          |
| 2048 |      256 |    1 |    34 | Wk on all layers + Wv on 12 layers  |
| 2048 |     5632 |    1 |    44 | gate + up per layer (2 × 22)        |
| 5632 |     2048 |    1 |    12 | ffn_down on 12 layers               |

Sum: 44 + 34 + 44 + 12 = 134. ✓

### Other-quant shapes (21 nodes)

| cols | out_cols | rows | count | most likely identity                |
|-----:|---------:|-----:|------:|-------------------------------------|
| 2048 |      256 |    1 |    10 | Wv on 10 layers (Q6_K from Q4_K_M recipe) |
| 5632 |     2048 |    1 |    10 | ffn_down on 10 layers (Q6_K)        |
| 2048 |    32000 |    1 |     1 | lm_head / output projection         |

Sum: 10 + 10 + 1 = 21. ✓

So the Q4_K_M recipe applies Q6_K to attn_v + ffn_down on 10 of 22 layers and
Q4_K on the other 12, plus Q6_K on the lm_head. The split is consistent with
llama.cpp's quant-strategy upgrades for "more important" layers.

### FLOP-weighted ownership

Approximate per-token MUL_MAT FLOPs (proportional to `cols × out_cols × rows`,
ignoring the ×2 for fma):

| bucket             | FLOPs (M) | share |
|--------------------|----------:|------:|
| Q4_K-repacked      |     ~847  | **82%** |
| other-quant        |     ~186  |  18%  |
| **total MUL_MAT**  |    ~1033  | 100%  |

By node count Q4_K-repacked is 87%; FLOP-weighted it's 82%. The 5-percentage-
point gap is the lm_head: a single 2048×32000 Q6_K MUL_MAT is ~6% of total
MUL_MAT FLOPs by itself. After Q4_Kx8 lands, that single node will be the
largest residual matmul — worth noting if a follow-up Q6_K path is later
considered.

### Confirms B5A scoping

134 nodes / 82% of MUL_MAT FLOPs in one discriminating bucket. Implementing
repacked Q4_Kx8 lowering (Step 3 / Option B) is the right scope. Q6_K can
be a separate follow-up after we measure the residual histogram with the
Q4_K_M Q6_K layers + lm_head still on the fallback side.
