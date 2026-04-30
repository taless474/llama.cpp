# Selective graph-map diagnostic — first decode graph

## What was measured

One-shot dump of the first decode graph as seen by the selective executor,
classifying every node as **L** (lowered via fine-region path), **F**
(CPU-backend fallback), or **P** (frozen-packet member).

Diagnostic added in `ggml/src/ggml-hpx/ggml-hpx-exec-selective.cpp`,
gated on `LLAMA_HPX_SELECTIVE_GRAPH_MAP=1`. Prints once per process
right after the gate/up and GLU prescans complete and before the main
execution loop. Classifier mirrors the main loop:

- `consumed[i]` ⇒ **P**
- `!ggml_hpx_lower_op` ⇒ **F** (`not_lowerable`)
- has REDUCTION region with `uses_resources=1` ⇒ **F** (`resource_reduction`)
- otherwise ⇒ **L** (`lowered`)

## Reproduce

- commit: `7f05dcc2b` (+ uncommitted graph-map patch)
- binary: `build-hpx-dag/bin/llama-simple`
- model:  `models/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`

```
LLAMA_USE_HPX=1 \
LLAMA_HPX_SELECTIVE_MUL_MAT=1 \
LLAMA_HPX_SELECTIVE_GRAPH_MAP=1 \
LLAMA_HPX_SELECTIVE_STATS=1 \
./build-hpx-dag/bin/llama-simple \
  -m models/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  -n 4 -ngl 0 \
  "Hello, my name is" \
  > stdout.txt 2> stderr.txt
```

## Key numbers

```
[hpx-map-summary] nodes=689
                  lowered_nodes=201   fallback_nodes=488   packet_nodes=0
                  lowered_runs=101    fallback_runs=102    packet_runs=0
```

- **689 ggml nodes** in one decode graph (22 layers × ~31 nodes/layer + tail).
- **29% lowered, 71% fallback, 0% packet** by node count.
- **101 L-islands, 102 F-islands** — runs alternate almost one-for-one.
- **packet_runs=0** is expected here: this run did not enable the GLU /
  gate-up packet caches. The map shows where packets *would* fire if engaged.

## Run-length distribution (top buckets)

| count | cat | len | shape of island                                     |
|-------|-----|-----|-----------------------------------------------------|
|   57  |  L  |  1  | isolated MUL_MAT (Q-, K-, output projection)        |
|   46  |  F  |  2  | `ADD,RMS_NORM` (residual + norm)                    |
|   22  |  L  |  2  | `MUL,MUL_MAT` (norm-mul + Q proj)                   |
|   21  |  F  | 14  | KV-cache metadata: `RESHAPE,ROPE,VIEW,SET_ROWS,...` |
|   13  |  F  |  1  | single fallback (often V MUL_MAT or stray RESHAPE)  |
|   12  |  L  |  5  | **GLU sublayer**: `MUL,MUL_MAT,MUL_MAT,GLU,MUL_MAT` |
|   11  |  F  |  4  | `RESHAPE,ROPE,MUL_MAT(V),RESHAPE` (V projection)    |
|   10  |  L  |  4  | **GLU sublayer (no down)**: `MUL,MUL_MAT,MUL_MAT,GLU` |

## Findings (answers to the diagnostic questions)

- **102 fallback islands is real.** They alternate with L-islands, so almost
  every L-run is bracketed by an F-run on each side.
- **Lowered nodes are not isolated 1-by-1 in general** — but a large minority
  (57 out of 101 L-islands) are length-1 MUL_MATs surrounded by metadata.
- **Fallback is dominated by KV-cache plumbing**, not by matmuls. The
  recurring 14-node F-burst (21 occurrences ≈ once per layer) is
  `RESHAPE,ROPE,VIEW,SET_ROWS,VIEW,SET_ROWS,VIEW,PERMUTE,...` around
  the K/V cache write.
- **GLU pattern is visible end-to-end.** `MUL_MAT,MUL_MAT,GLU` shows up
  inside L-islands of length 4 and 5: 22 occurrences total, exactly matching
  TinyLlama's 22 layers. **This is where the GLU packet should fire.**
- **Q/K/V projection asymmetry is visible.**
  - **Q proj** (`q4_K_8x8_q8_K`) → lowered, lands in an L-len=2 (norm-mul + Q).
  - **K proj** (`q4_K_8x8_q8_K`) → lowered, isolated L-len=1 between two
    F-islands carrying the rope/view bookkeeping.
  - **V proj** (`q6_K`, repacked but not 8x8) → **falls back**, sitting
    inside an F-len=4 island: `RESHAPE,ROPE,MUL_MAT(V),RESHAPE`.
    `[hpx-map] idx=6 ... op=MUL_MAT name=Vcur-0 src0=q6_K trait=repacked
     reason=not_lowerable`.
- **Where packets should exist:** at every `MUL_MAT,MUL_MAT,GLU` cluster
  (22 sites). The graph-map confirms the GLU prescan would find these and
  mark them P if `mlp_glu_cache` were wired in by the caller.
- **Q4_K coverage gap to close next:** the Q4_K_8x8_q8_K trait is being
  hit (Q and K), but the V projection's q6_K (repacked, non-8x8 trait)
  is the recurring single-MUL_MAT fallback.

## Files

- `stdout.txt` — model output ("Hello, my name is" + 4 generated tokens).
- `stderr.txt` — 689 `[hpx-map]` + 203 `[hpx-map-run]` + 1 `[hpx-map-summary]`
  + the surrounding llama/HPX log lines.
- `README.md` — this file.

## Open questions

- Re-run with the GLU packet path engaged (cache wired by caller) and
  confirm the 22 GLU sites move from L (len=4/5) to P (len=3).
- Investigate whether the V proj `q6_K` repacked trait can be added to the
  lowered path, which would convert ~22 F-len=4 islands into shorter F-len=2
  bookkeeping islands plus an L-len=1 V MUL_MAT — likely also enabling
  packet coverage of Q/K/V together in a fused attention sublayer later.
- Investigate the recurring 14-node KV-cache F-island. Most ops in it are
  metadata that probably can't be lowered profitably, but if `SET_ROWS` or
  `PERMUTE` could be lowered, those islands would shorten further.
