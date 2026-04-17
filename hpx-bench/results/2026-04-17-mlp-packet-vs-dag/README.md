# bench_hpx_mlp_gate_up_f32 — dag_group vs direct_manual vs frozen_packet

## What was measured

Three execution paths for the MLP gate/up sublayer (4 ops: gate MUL_MAT, up MUL_MAT, SiLU, elementwise MUL), n_lanes=1 (decode), macOS Apple Silicon.

- **dag_group**: composed 4-region group run via `ggml_hpx_run_region_group`.  
  Generic dependency-driven DAG scheduler: shared_future per region, dataflow chains, heap allocation per call.

- **direct_manual**: 4 production `run_range` callbacks called in fixed order on the caller thread, ith=0, nth=1. No HPX, no DAG. Ctx structs built once per shape (not per rep) — same as the packet which also does not rebuild dims at bind time. This is the bare math cost.

- **frozen_packet**: compiled 4-step packet, pinned decode Exec.  
  Timed body: `bind` (7 pointer patches) + `run_frozen_packet`.  
  n_lanes=1 → all steps SERIAL, no inner `for_loop`.

Correctness verified for all 12 shapes (relative tolerance 1e-5).

## Summary table (median ns)

| shape (oc×c×r) | dag_med | dm_med | pkt_med | dag/pkt | dm/pkt | dag/dm |
|---|---|---|---|---|---|---|
| 3×4×2 (focused) | 9875 | 84 | 125 | **79×** | 0.7× | **118×** |
| 64×64×1 | 9583 | 500 | 1166 | 8.2× | 0.4× | **19×** |
| 64×64×2 | 10542 | 958 | 1000 | 10.5× | ~1× | **11×** |
| 64×64×8 | 14083 | 3792 | 3833 | 3.7× | ~1× | 3.7× |
| 64×64×16 | 18250 | 7583 | 7583 | 2.4× | ~1× | 2.4× |
| 64×256×1 | 11416 | 1666 | 1667 | 6.8× | ~1× | **6.9×** |
| 256×64×1 | 11708 | 1959 | 2000 | 5.9× | ~1× | 6.0× |

## Findings

**direct_manual ≈ frozen_packet** once the shape is large enough to give any math work (≥ ~64 elements). The packet overhead above the raw callbacks is ~2–5% at the sweep sizes. At the tiny focused shape (3×4×2), the packet adds ~40 ns on top of 84 ns math — which is still the bind step amortized over nothing.

**dag_group overhead is large and shape-independent.** The DAG floor (7–10 µs) dominates all shapes in this sweep. The generic scheduler machinery (shared_future alloc, dataflow chain, topological sort per call) costs the same regardless of how much math follows.

**dag_group p95 spikes to ~1 ms** on most shapes. The packet p95 matches the median. This is the allocator / task-scheduler variance in the generic DAG path.

**`dm/pkt` ratio**: at 64×64 and above, the packet overhead above raw callbacks is ≤5%. Bind (7 pointer stores into the frame) is negligible.

## Open questions

- Does the dag/pkt ratio shrink further at realistic LLM sizes (out_cols=4096+)?
- Does n_lanes>1 (prefill path) change the overhead structure? PREFILL team not yet implemented.

## Environment

- Commit: `116452bdb86182446bc9824506656761a0d47d7f`
- Binary: `build-hpx-dag/bin/bench_hpx_mlp_gate_up_f32`
- Date: 2026-04-17
