# Selective Guard: Validate the Graph-Entry Bailout

**Date:** 2026-04-26
**Base commit:** `e0b332bb5` ("Guard Q4_K lowering against repacked CPU tensors")
**Working tree:** dirty — three uncommitted files implement the guard:
- `ggml/src/ggml-hpx/ggml-hpx-exec-selective.h` — declare `ggml_hpx_selective_should_engage`
- `ggml/src/ggml-hpx/ggml-hpx-exec-selective.cpp` — implement field-only MUL_MAT predicate
- `src/llama-context.cpp` — second skip arm at the selective gate

**Branch:** `hpx-prefill-orchestrator`
**Binary:** `build-hpx-dag/bin/llama-simple` (single rebuild)
**Model:** `models/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`
**Invocation:** `-ngl 0 -n 16 "Hello"`
**Env:** `LLAMA_USE_HPX=1 LLAMA_HPX_SELECTIVE_MUL_MAT=1 LLAMA_HPX_SELECTIVE_DEBUG=1`
**Machine:** Apple M4, quiet (WindowServer + VSCode + claude only at check time, no thermal warnings)

## Short answer

The graph-entry guard correctly bypasses the per-node selective dispatch on M4 Q4_K_M
decode. The 9.5× regression observed in the prior B4 measurement is gone; throughput
returns to plain-CPU baseline within run-to-run noise.

## What changed

Selective entry in `llama_context::graph_compute` now requires
`ggml_hpx_selective_should_engage(gf)` in addition to the existing CPU-only-graph check.
The new predicate is a single linear pass over `gf->nodes` that mirrors
`ggml_hpx_lower_op`'s MUL_MAT acceptance gate at the field level: output `F32`, plus
either F32×F32 or non-repacked Q4_K (`w->extra == nullptr`) with `rows == 1`.

When the graph contains MUL_MATs but none of them would lower (the M4 Q4_K_M case —
every Q4_K weight is repacked), the gate falls through to
`ggml_backend_sched_graph_compute_async`, which runs the whole graph as one unit on
the configured CPU threadpool. No more 622 single-op compute calls per token.

## Observed signals

| Expected                                                                          | Observed                                                                                | Verdict |
|-----------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|---------|
| Skip log `[hpx-selective] disabled: no lowerable MUL_MAT in graph` fires once     | Fires once per process invocation (warned-once)                                         | OK      |
| Throughput recovers to ~plain-CPU baseline (~100 tok/s)                           | 101.9 ± 2.5 tok/s across 4 reps                                                         | OK      |
| Output coherent, matches baseline prefix                                          | `<s> Hello, World! 5. Python: \`\`\`python print("...` — same prefix as B4 baseline    | OK      |

## Timing data (4 reps, -ngl 0, CPU-only)

| arm                                  | eval ms/tok | eval tok/s     |
|--------------------------------------|------------:|---------------:|
| **hpx_selective + guard (this run)** | 9.81 ± 0.25 | **101.9 ± 2.5** |
| baseline_cpu (B4)                    | 9.97 ± 0.6  | 100.2 ± 5.3    |
| hpx_selective (B4, no guard)         | 93.7 ± 6.4  | 10.7 ± 0.8     |

The guarded selective path is statistically indistinguishable from the plain-CPU
baseline (overlapping confidence intervals, 1.7% mean delta), and 9.5× faster than
the unguarded selective path. This is the expected outcome: with all MUL_MATs
bailing out, the guard delegates the entire graph to one whole-graph compute on
the same threadpool the unguarded selective path would have used per-node.

Per-rep raw values are in `bench_selective_guard.csv`. Full stderr from one rep
(including the warned-once skip log) is in `bench.log`.

## Why the fix is the right shape

The 10× regression was not a Q4_K-math problem and never was. With every Q4_K
matmul rejected by the `w->extra` guard, the selective executor was paying its
per-node dispatch tax 622× per token while owning zero heavy work. The fix
declines to engage selective when the predicted heavy-op ownership is zero, and
otherwise leaves the engaged path completely unchanged.

The predicate is a strict subset of `ggml_hpx_lower_op` — field reads only, no
contiguity / stride / scratch-budget checks. False positives (engage when lowering
later rejects the matmul) are harmless; false negatives are impossible by
construction in the regression case.

## Test status

All 27 HPX tests pass (`ctest -I 52,78`), including the four selective-path tests
and the Q4_K selective tests. The guarded path's behavior on the engaged side
is provably untouched: `ggml_hpx_selective_should_engage` returns true for every
graph that has a lowerable MUL_MAT, and every existing test builds such a graph.

## Open questions / next step

This run validates Step 1 of the post-B4 plan: prevent regression. It does not
unlock M4 Q4_K perf.

Step 2 is fallback-bucket instrumentation — counting the fallback nodes by op
type on a single TinyLlama Q4_K_M decode. The number that determines whether
implementing repacked Q4_Kx8 support is worth doing is: how many of the 622
fallback nodes are repacked-Q4_K MUL_MATs, vs. miscellaneous small ops (RoPE,
softmax, attention compute, RMS finalize, etc.)?

- If repacked Q4_K MUL_MATs dominate → implementing `q4_K_8x8_q8_K` lowering moves
  most of the 622 to the lowered path, and the guard stops firing on real models.
- If miscellaneous ops dominate → repacked support alone won't move the needle and
  we'd need to widen non-MUL_MAT lowering coverage too.

The instrumentation is local to the selective TU and does not require changes to
the dispatch logic. It just tags every fallback with `node->op` and emits a
histogram at the end.
