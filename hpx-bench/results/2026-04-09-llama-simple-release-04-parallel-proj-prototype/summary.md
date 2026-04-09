# Experiment 04 — Outer-region parallel chain dispatch: negative result

## Hypothesis
Detecting independent op groups (FFN gate+up, Q/K/V projections) inside a
CPU prefill region and dispatching each group as a separate HPX async task
will reduce prefill time by running the groups concurrently.

## Setup
- date: 2026-04-09
- branch: hpx-prefill-orchestrator / Release build
- model: TinyLlama-1.1B-Chat-v1.0.Q4_K_M
- prompt: "Hello " × 100 (102 tokens, -n 1, -ngl 0)
- prototype: `detect_parallel_chains` + `run_parallel_chains` in ggml-hpx-exec.cpp
- env vars: LLAMA_HPX_PARALLEL_PROJ=1, LLAMA_HPX_PAR_CHAIN_THREADS=N

## What the prototype does

**detect_parallel_chains**: Kahn's BFS on each sched_split CPU region with
≥ 20 nodes. Stops at the first topological level where all ready nodes are
MUL_MAT and width ≥ 2. On this machine (BLAS accelerates Q/K/V), it finds
FFN gate+up (width=2, level 7, CONTIGUOUS) in every layer.

Chain decomposition for each CPU region:
- before  [node_begin, gate)  serial, original sched backend
- chain 0 [gate, up)          1-node HPX async task, fresh cpu backend
- chain 1 [up,  merge)        1-node HPX async task, fresh cpu backend
- after   [merge, node_end)   serial, original sched backend

merge = first node with cross-chain deps (GLU, depends on both gate and up).

**run_parallel_chains**: calls `ggml_backend_cpu_init()` per chain, sets
n_threads = LLAMA_HPX_PAR_CHAIN_THREADS, fires hpx::async per chain,
hpx::wait_all, frees backends.

## Results

### Correctness: PASS
All variants produce bit-exact output vs baseline on all scenarios.
The chain detection, boundary derivation, and hpx::wait_all dispatch
are correct. No data races, no divergence.

### Timing: FAIL — parallel is consistently slower

| config                    | ms  | t/s | vs HPX serial |
|---------------------------|-----|-----|---------------|
| baseline (no HPX)         | 316 | 323 | —             |
| HPX serial                | 410 | 249 | 1.0×          |
| parallel 1 t/chain        | 679 | 150 | −1.7×         |
| parallel 2 t/chain (2+2)  | 631 | 162 | −1.5×         |
| parallel 3 t/chain        | 589 | 173 | −1.4×         |
| parallel 4 t/chain        | 585 | 174 | −1.4×         |

The **2+2 shared-budget check** (2 chains × 2 threads = 4 total, matching the
serial backend's thread count) gives 631ms vs 410ms serial. The hypothesis
is falsified: equal thread budget does not yield equal or better performance.

## Root cause

The fundamental problem is **thread pool fragmentation**.

`ggml_backend_cpu_init()` allocates a new ggml internal threadpool.
It does not reuse or share the calling backend's pool. So for each region:

```
serial path:
  original backend (4 threads) → gate → up   [sequential, 4 threads each]

parallel path (2+2):
  original backend (4 threads) → before/after [4 threads, as before]
  new backend A (2 threads) → gate             ← these run in parallel
  new backend B (2 threads) → up               ←
```

The original 4 threads are **idle** during the gate+up phase. The 2+2 new
threads are *additional* threads, not a redistribution of the original 4.
The total active thread count during gate+up drops from 4 (serial) to 2+2
on separate pools, which is never better and often worse (cache fragmentation,
OS scheduler pressure).

The 2+2 budget is additive, not shared. The hypothesis requires shared.

Secondary factors:
- FLASH_ATTN_EXT dominates the before-slice (~60% of region time); it runs
  serially regardless, capping the achievable speedup from gate+up parallelism.
- Each of the 22 layers triggers backend init+free for 2 chains; measured at
  0.8ms total for 44 cycles — negligible, not the cause.
- HPX task dispatch overhead: also negligible at ~44 task posts per prefill.

## What this experiment established

1. **Outer orchestration is not sufficient.** Wrapping independent ggml ops
   in separate ggml backends and running them via hpx::async produces correct
   results but cannot achieve a shared thread budget. The two thread pools are
   independent and always run in addition to, not instead of, the original.

2. **The graph decomposition works.** detect_parallel_chains correctly identifies
   FFN gate+up and derives before/merge/after boundaries. The chain ranges are
   contiguous ggml_graph_view slices. The scattered-dispatch machinery (proven
   in BackendComputeScatteredNodeListMatchesBaseline) is not actually needed for
   this model — the chains happen to be contiguous in node index order.

3. **The right interface is inside ggml's threadpool, not above it.** To share
   thread resources across concurrent subgraphs, HPX must become the executor
   that ggml's per-op parallelism dispatches through — not a separate layer
   above backends. This is option B: replace ggml_graph_compute's threadpool
   with an HPX-backed executor.

## Disposition

Prototype code (`detect_parallel_chains`, `run_parallel_chains`, ggml-cpu link
in CMakeLists, LLAMA_HPX_PARALLEL_PROJ env var) is left in place as a reference
implementation behind the env-var guard. It does not affect any path unless
LLAMA_HPX_PARALLEL_PROJ=1 is set. All existing tests pass.

Next step (option B): design a plan for replacing ggml's internal threadpool
with an HPX executor in a targeted CPU backend, enabling N concurrent subgraph
dispatches to share the same thread pool.
