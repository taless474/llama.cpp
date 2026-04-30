# Packet design note — `MLP_GATE_UP_GLU_Q4K8`

Companion to `README.md` and `stderr.txt` in this directory. Drafted off the
graph-map evidence: 22 structurally identical GLU sites per decode graph,
all with `q4_K_8x8_q8_K` gate and up weights.

## Why this packet (vs QBRIDGE)

`QBRIDGE` (already in tree, see `lookup_or_compile_mlp_glu`'s
`GGML_HPX_PACKET_SUBLAYER_MLP_GLU_QBRIDGE` branch) keeps the SWIGLU step in
a packet but routes gate_mm and up_mm through `ggml_backend_graph_compute`
on the CPU backend. Two consequences make that wrong for our perf goal:

- The two CPU dispatches are coarse-grained substrate ggml-graph_view +
  graph_compute calls — exactly the per-node tax that CLAUDE.md's
  "Current performance truth" identifies as the remaining decode blocker.
- The Q8_K-quantized activation produced for the gate gemv is thrown away,
  then immediately re-derived for the up gemv — the same `x` is quantized
  twice in two unrelated CPU calls.

`MLP_GATE_UP_GLU_Q4K8` does both gemvs inside one HPX entry, sharing a
single quantized `x` across them, and runs SWIGLU before returning. One
HPX entry → three sublayer nodes consumed → no per-node coarse dispatch.

## Target pattern

```
MUL_MAT  ffn_gate-N        src0=q4_K  trait=q4_K_8x8_q8_K
MUL_MAT  ffn_up-N          src0=q4_K  trait=q4_K_8x8_q8_K
GLU      ffn_swiglu-N      op=SWIGLU
```

Count: **22 sites per decode graph** on TinyLlama Q4_K_M (one per layer).
All sites share the same shape — exactly one cache entry across the run.

## First-cut match conditions (prescan rejects anything else)

The prescan must return a match only when ALL hold:

1. Trigger node `op == GGML_OP_GLU` and `glu_op == GGML_GLU_OP_SWIGLU`.
2. `gate_t = trigger->src[0]` and `up_t = trigger->src[1]` are both
   `GGML_OP_MUL_MAT`.
3. `gate_t->src[0]->type == GGML_TYPE_Q4_K`
   and `gate_t->src[0]->extra != nullptr`
   and `ggml_hpx_is_q4k_8x8_repacked(gate_t)` —
   note: `gate_t` is the **MUL_MAT op node**, not the weight tensor;
   the predicate's existing signature
   (`ggml-hpx-lower.h:131`) takes the op and inspects `op->src[0]`
   internally. The whole packet is valid only for the
   `q4_K_8x8_q8_K` repacked trait, so this condition is load-bearing.
4. Same three conditions on `up_t` (also the MUL_MAT op node).
5. `gate_t->src[1] == up_t->src[1]` (identical activation tensor).
6. Activation `x = gate_t->src[1]` has `x->ne[1] == 1` (decode row).
7. Output width `gate_t->ne[0] == up_t->ne[0]` (we'll call it `out_cols`).
8. Input width `gate_t->src[0]->ne[0] == up_t->src[0]->ne[0]`
   (we'll call it `cols`).
9. GLU output type is `GGML_TYPE_F32`.
10. **Exclusive consumer**: no node in `gf` other than the SWIGLU trigger
    reads `gate_t` or `up_t`. Required because packet dispatch happens
    at the GLU trigger node — the original gate_mm / up_mm nodes are
    *skipped* at their own positions in graph order. If any other node
    consumes gate or up between the MUL_MAT positions and the GLU
    position, that consumer would observe the gate/up tensors before the
    packet has written them, breaking ggml graph order. First cut only
    accepts gate/up tensors whose sole graph consumer is the SWIGLU
    trigger; widening this is a future extension and would require
    moving packet dispatch earlier (or rewriting the graph), neither of
    which is in scope here.
11. Topological precedence: `i_gate < i_glu` and `i_up < i_glu`, and
    none of the three are already consumed by an earlier match.

If any condition fails the prescan does not consume the nodes; they fall
through to the existing per-node selective routing unchanged.

## Plan key

```
sublayer       = GGML_HPX_PACKET_SUBLAYER_MLP_GATE_UP_GLU_Q4K8   (new value)
team           = GGML_HPX_PACKET_TEAM_DECODE
n_lanes        = cache->n_lanes
dtype          = GGML_TYPE_F32                       (output dtype)
seq_regime     = cache->seq_regime
policy_version = 1
shape[0]       = out_cols                            (5632 here)
shape[1]       = cols                                (2048 here)
shape[2]       = 1                                   (rows; locked at first cut)
shape[3]       = w_row_stride                        (gate->src[0]->nb[1])
extra          = 0                                   (gate/up weight type
                                                      both fixed by sublayer)
```

`w_row_stride` is the repacked weight's row stride in bytes
(`w->nb[1]` of the gate weight; the composer asserts gate and up
agree). It is **structural identity**, not a per-call parameter: it is
baked into the compiled gemv ctx and cannot be patched at bind time
without violating the binding contract (six pointers only, no stride
fields). Two sites with the same `(out_cols, cols)` but different
repacked layouts (different repack trait variant, different backend,
different model) would silently miscompile if they shared a cached
packet. The cache key therefore includes it.

Two sites with the same `(out_cols, cols, rows, w_row_stride)` tuple
produce the same key → one cached compiled packet. With our 22 sites
all `(5632, 2048, 1, w_row_stride)` and a single repack trait
(`q4_K_8x8_q8_K`), `w_row_stride` is constant across the run, so we
still expect 1 compile and 21 hits per process.

## Binding (caller fills these at dispatch time)

```
struct ggml_hpx_mlp_gate_up_glu_q4k8_binding {
    const void *  w_gate;     // q4_K repacked weight, gate_t->src[0]->data
    const void *  w_up;       // q4_K repacked weight, up_t  ->src[0]->data
    const float * x;          // f32 activation row, gate_t->src[1]->data
    float *       gate;       // gate_t->data — optional write-back
    float *       up;         // up_t  ->data — optional write-back
    float *       out;        // glu_t ->data — required, this is the live output
};
```

`gate` and `up` are populated only if the prescan's exclusive-consumer check
returned "consumed by GLU only" but we choose to keep them addressable for
future rematerialization or debug. First cut: write them through (matches
`MLP_GLU_F32`'s behavior, costs 2× 22528 B writes per dispatch, keeps
ggml-side state coherent).

## Resource model

Per-dispatch buffers:

| buffer            | size (TinyLlama)                  | lifetime           |
|-------------------|-----------------------------------|--------------------|
| `q8k_x` shared    | `ggml_row_size(Q8_K, 2048)` = 2336 B | lives across regions |
| `gate` output     | `out_cols * sizeof(float)` = 22528 B | written through    |
| `up` output       | `out_cols * sizeof(float)` = 22528 B | written through    |
| `out` (GLU)       | `out_cols * sizeof(float)` = 22528 B | written through    |
| `lane_scratch[i]` | none                              | unused              |
| `reduction_buffer`| none                              | unused              |

`q8k_x` lives in `shared_scratch` (the ctx the packet allocates inside
its lowered group). `gate`, `up`, `out` are pointers from the binding —
no allocation. **No REDUCTION region with `uses_resources=1`.**

## Region group (composed at packet compile)

Four regions, four dep edges:

```
R0  ELEMENTWISE   quantize_x_to_q8k        : (x, cols)            -> q8k_x
R1  GEMV         q4_K_8x8_q8_K_gemv (gate): (W_gate, q8k_x, ...) -> gate
R2  GEMV         q4_K_8x8_q8_K_gemv (up)  : (W_up,   q8k_x, ...) -> up
R3  ELEMENTWISE   swiglu                   : (gate, up, out_cols) -> out

deps:  R0 -> R1
       R0 -> R2
       R1 -> R3
       R2 -> R3
```

### First-cut execution: single outer HPX entry, straight-line body

The first cut does NOT build a nested HPX DAG inside the packet. It uses
one outer `hpx::async` (the same dispatch shape the existing GLU and
gate/up packets use) and a straight-line single-lane body:

1. Bind `b` (caller fills the six pointers).
2. `quantize_x_to_q8k(b.x, cols, q8k_x)`.
3. `q4_K_8x8_q8_K_gemv_prequantized(b.w_gate, q8k_x, b.gate, out_cols, cols)`.
4. `q4_K_8x8_q8_K_gemv_prequantized(b.w_up,   q8k_x, b.up,   out_cols, cols)`.
5. `swiglu(b.gate, b.up, b.out, out_cols)`.

Total: **one HPX entry, four straight-line CPU calls**, no
`ggml_backend_graph_compute`, no per-node coarse dispatch, no nested
HPX tasks. This is what we measure first — the design proves out only
when correctness and dispatch-count reduction are demonstrated on this
shape.

The four-region group above is the *composed structural description* the
packet carries (so the packet system, stats, and future extensions see
the same DAG every other packet sees), not a recipe for spawning four
tasks. A single-lane packet runner walks the DAG in topological order
and calls each region's kernel inline; this is the same way the existing
single-lane MLP_GLU packets work today.

### Region-DAG variant (follow-up, not first cut)

Once correctness ships green, the natural next step is to fan R1 and R2
out: they are independent (both read shared `q8k_x`, write disjoint
regions of `gate` and `up`), and partitioning the 5632 output rows into
`n_lanes` 8-row-aligned chunks parallelizes each gemv. R3 (swiglu) then
partitions `out_cols` across `n_lanes`. Avoid spawning a pile of internal
HPX tasks at first — this is a deliberate sequencing choice, not a
limitation of the packet contract.

## Caller-side wiring (selective executor)

Mirrors the existing GLU branch in `ggml_hpx_exec_graph_selective_mul_mat`:

- Prescan (`prescan_mlp_gate_up_glu_q4k8_matches`) runs after the gate/up
  prescan but before the existing GLU prescan, so the new packet wins on
  any site it can match. Sites where any condition (e.g. q6_K weight or
  exclusive-consumer fails) drop through to the existing GLU paths
  unchanged.
- Trigger dispatch is at the GLU node (same as current GLU branch).
  Member nodes (gate_mm, up_mm) are skipped silently when consumed.
- Stats: `packet_matches += 1`, `packet_nodes += 3` per dispatch.
  No `bridge_fallback_nodes` increment — that field stays 0 for this
  sublayer.

Cache lifecycle: a new `ggml_hpx_mlp_gate_up_glu_q4k8_packet_cache`
parallel to the existing two caches. Plugged into
`ggml_hpx_selective_packet_env` as a fourth optional pointer, so a caller
that creates only the existing caches loses nothing.

## What this packet does NOT include

- **ffn_out / down projection.** Stays as a separate node, post-GLU.
  Routes per its existing trait: 12 of 22 layers go to the lowered path
  (q4_K_8x8_q8_K), 10 go to fallback (q6_K repacked, non-8x8). Folding
  ffn_out into the same packet is a future extension once this one ships
  green.
- **F32 weights.** That case keeps using the existing `MLP_GLU_F32`
  packet. No code is moved out of that path.
- **Multi-row decode (`ne[1] > 1`).** First cut matches `rows == 1` only.
  Prefill ubatches do not engage packet dispatch anyway (caller passes
  `nullptr` env on prefill).
- **Q4_K standard layout (`extra == nullptr`).** First cut requires the
  `q4_K_8x8_q8_K` repack trait so we can call the existing repacked
  gemv directly. Standard-layout Q4_K can be added later via a different
  inner kernel; key would diverge by `extra` field.
- **Other repacked Q4_K traits** (`q4_K_8x4_q8_K`, RISC-V 16x1). Excluded
  for the same reason.

## Why this matters for the live-speedup story

CLAUDE.md's current performance truth: selective decode is 10–12 tok/s
vs ~99 tok/s baseline; the blocker is "selective executor dispatch
structure, especially one HPX entry per lowered node." This packet
collapses three lowered/dispatched MUL/GLU events per layer into one
HPX entry, 22 times per graph — a direct dent in the dispatch tax that
is the named blocker. It does not by itself fix the K-projection
isolated-MUL_MAT or the V-projection q6_K fallback; those are separate
follow-ups.

## First implementation check — resolved

Result: **READY.**

The Q4_Kx8 lowering path already exposes the packet shape we need.

The required kernels exist as standalone `run_range` helpers:

- `ggml_hpx_quantize_q8_k_f32_run_range`
  - F32 activation row → Q8_K scratch
- `ggml_hpx_mul_mat_q4_k_8x8_q8_k_run_range`
  - repacked q4_K_8x8 weight + prequantized Q8_K row → F32 output
- `ggml_hpx_swiglu_f32_run_range`
  - gate + up → SWIGLU output

No refactor is required before packet implementation.

The packet should allocate Q8_K scratch in the packet frame / shared
scratch, run quantize once, then call the existing Q4_Kx8 gemv helper
twice: once for gate and once for up.

The previous conditional branch

- if prequantized-x helper exists, implement packet directly
- otherwise refactor `lower.cpp` first

is now resolved to the first path.

## First-cut implementation target

Implement the smallest version behind an env flag (e.g.
`LLAMA_HPX_PACKET_MLP_GATE_UP_GLU_Q4K8=1`). The flag both enables the
new prescan and enables the cache wiring in
`ggml_hpx_selective_packet_env`; default off so the binary is risk-free
to ship.

First success criteria (one decode graph on TinyLlama Q4_K_M):

```
packet_matches = 22
packet_nodes   = 66
output         = bit-identical to selective-off
Q4_K fallback  = no regression vs current selective-on
```

Stretch criterion (do not block on this for the first cut): measurable
reduction in HPX-entry count per token, since the named blocker in
CLAUDE.md is "one HPX entry per lowered node" — 22 fewer entries per
graph is the minimum we expect this packet to deliver, before any
multi-lane fan-out work.

## Other open questions

- **Exclusive-consumer check cost.** Naive: O(n_nodes × 4) per prescan
  pass. Prescan already does linear scans for `find_index`; one more
  is acceptable but worth measuring. Alternative: build a single
  `consumers[i]` map at the top of `ggml_hpx_exec_graph_selective_mul_mat`
  and reuse for all prescans.
- **Write-through of gate / up.** Confirmed safe under the
  exclusive-consumer check, but doing it is the safer first cut. Cost
  is ~45 KB of writes per dispatch — small relative to the gemv
  arithmetic.
- **Cache invalidation.** Bump `policy_version` to `1` for the new
  sublayer so a stale-cached older packet (none exists yet) cannot
  be reused.
- **Prescan ordering vs existing GLU prescan.** New prescan runs FIRST;
  existing GLU prescan must skip any node already consumed (already does
  this via the `consumed[]` check). No code change there.
