// ggml-hpx-compose.h
//
// Composer: concatenates multiple per-op ggml_hpx_lowering results into one
// ggml_hpx_cpu_region_group with cross-op dep edges wired.
//
// Currently ships two sublayer composers:
//
//   MLP gate/up  (4 ops: MUL_MAT gate, MUL_MAT up, SiLU, elementwise MUL)
//
//     Region layout (fixed):
//       0  gate   = MUL_MAT(W_gate, x)   MATMUL
//       1  up     = MUL_MAT(W_up,   x)   MATMUL
//       2  silu   = SiLU(gate)            ELEMENTWISE
//       3  out    = MUL(silu, up)         ELEMENTWISE
//
//     Cross-op dep edges (global region indices):
//       0 → 2   (gate → silu)
//       1 → 3   (up   → out)
//       2 → 3   (silu → out)
//
//     gate and up MUL_MAT are independent and may execute in parallel.
//
//   MLP GLU      (3 ops: MUL_MAT gate, MUL_MAT up, GLU[SWIGLU])
//
//     Real llama MLP uses ggml_swiglu_split which emits GGML_OP_GLU with
//     GGML_GLU_OP_SWIGLU — a single fused node replacing the separate SiLU +
//     elementwise MUL of the gate/up composer above.
//
//     Region layout (fixed):
//       0  gate   = MUL_MAT(W_gate, x)          MATMUL
//       1  up     = MUL_MAT(W_up,   x)          MATMUL
//       2  glu    = GLU[SWIGLU](gate, up)        ELEMENTWISE
//
//     Cross-op dep edges (global region indices):
//       0 → 2   (gate → glu)
//       1 → 2   (up   → glu)
//
//     gate and up MUL_MAT are independent and may execute in parallel.
//
// Gated on GGML_HPX_REGION_DAG.

#pragma once

#ifdef GGML_HPX_REGION_DAG

#include "ggml-hpx-lower.h"    // ggml_hpx_lowering, ggml_hpx_lowering_init

#include "ggml.h"

#include <cstring>

// ---------------------------------------------------------------------------
// MLP gate/up composed group
// ---------------------------------------------------------------------------
//
// Ownership rules (same invariant as ggml_hpx_lowering):
//   - lowers[i] owns the ctx storage for region i.
//     combined_regions[i].ctx always points into lowers[i].ctx_buf[0].
//   - This struct must NOT be copied or moved after compose returns.
//     group.regions and group.deps contain absolute pointers into the inline
//     arrays; copying produces dangling pointers in the copy.
//   - Callers pass &grp->group to ggml_hpx_run_region_group or
//     ggml_hpx_compile_packet. Both must deep-copy any state they retain.
//   - Lifetime: must remain valid until after any compile/run call returns.
//
// Call ggml_hpx_mlp_gate_up_group_init before passing to the composer.

#define GGML_HPX_MLP_GATE_UP_N_OPS     4    // gate, up, silu, mul
#define GGML_HPX_MLP_GATE_UP_N_REGIONS 4    // one region per op (all single-region)
#define GGML_HPX_MLP_GATE_UP_N_DEPS    3    // 0→2, 1→3, 2→3

typedef struct ggml_hpx_mlp_gate_up_group
{
    // Per-op ctx storage. lowers[i].ctx_buf[0] is the live ctx for region i.
    // Not used for scheduling; only the ctx_buf fields are accessed at run time.
    ggml_hpx_lowering  lowers[GGML_HPX_MLP_GATE_UP_N_OPS];

    // Flat combined region array. Populated by the composer from lowers[i].regions[0].
    // ctx pointers remain stable: they point into lowers[i].ctx_buf[0] above.
    ggml_hpx_cpu_region  combined_regions[GGML_HPX_MLP_GATE_UP_N_REGIONS];

    // Cross-op dep edges in global combined_regions[] indices.
    ggml_hpx_dep_edge    combined_deps[GGML_HPX_MLP_GATE_UP_N_DEPS];

    // The combined group ready for run_region_group or compile_packet.
    // group.regions == combined_regions, group.deps == combined_deps.
    ggml_hpx_cpu_region_group  group;
} ggml_hpx_mlp_gate_up_group;

// Zero the struct and wire all internal pointers. Must be called before
// ggml_hpx_compose_mlp_gate_up_group. Safe to call again for reuse.
static inline void ggml_hpx_mlp_gate_up_group_init(ggml_hpx_mlp_gate_up_group * g)
{
    std::memset(g, 0, sizeof(*g));
    for (int i = 0; i < GGML_HPX_MLP_GATE_UP_N_OPS; ++i)
    {
        // lower_op sets lowers[i].group.regions itself, but init them here
        // for clarity so the struct is self-consistent before compose.
        g->lowers[i].group.regions = g->lowers[i].regions;
    }
    g->group.regions = g->combined_regions;
    g->group.deps    = g->combined_deps;
}

// ---------------------------------------------------------------------------
// Composer
// ---------------------------------------------------------------------------
//
// Lower each of the four ops, copy their single regions into the combined
// flat array, and wire the three cross-op dep edges.
//
// Input nodes:
//   node_gate     : result of ggml_mul_mat(W_gate, x)
//   node_up       : result of ggml_mul_mat(W_up,   x)
//   node_gate_act : result of ggml_silu(node_gate)
//   node_out      : result of ggml_mul(node_gate_act, node_up)
//
// Preconditions:
//   - ggml_hpx_mlp_gate_up_group_init(grp) was called.
//   - All four nodes have type GGML_TYPE_F32.
//   - All data pointers are set and contiguous.
//   - Each op must lower to exactly 1 region (MUL_MAT, SiLU, elementwise MUL
//     are all single-region; multi-region ops like RMS_NORM are rejected).
//
// Returns true on success. On failure, grp->group.n_regions is 0.
// grp is not valid for use if false is returned.

#ifdef __cplusplus
extern "C" {
#endif

bool ggml_hpx_compose_mlp_gate_up_group(
    const struct ggml_tensor *      node_gate,
    const struct ggml_tensor *      node_up,
    const struct ggml_tensor *      node_gate_act,
    const struct ggml_tensor *      node_out,
    ggml_hpx_mlp_gate_up_group *    grp);

#ifdef __cplusplus
}    // extern "C"
#endif

// ---------------------------------------------------------------------------
// MLP GLU (SWIGLU) composed group
// ---------------------------------------------------------------------------
//
// Ownership rules: identical to ggml_hpx_mlp_gate_up_group.
//   - lowers[i] owns the ctx storage for region i.
//     combined_regions[i].ctx always points into lowers[i].ctx_buf[0].
//   - Must NOT be copied or moved after compose returns.
//   - Callers pass &grp->group to ggml_hpx_run_region_group or
//     ggml_hpx_compile_packet. Both must deep-copy any state they retain.
//   - Lifetime: must remain valid until after any compile/run call returns.
//
// Call ggml_hpx_mlp_glu_group_init before passing to the composer.

#define GGML_HPX_MLP_GLU_N_OPS     3    // gate, up, glu
#define GGML_HPX_MLP_GLU_N_REGIONS 3    // one region per op (all single-region)
#define GGML_HPX_MLP_GLU_N_DEPS    2    // 0→2, 1→2

typedef struct ggml_hpx_mlp_glu_group
{
    ggml_hpx_lowering  lowers[GGML_HPX_MLP_GLU_N_OPS];

    ggml_hpx_cpu_region  combined_regions[GGML_HPX_MLP_GLU_N_REGIONS];

    ggml_hpx_dep_edge    combined_deps[GGML_HPX_MLP_GLU_N_DEPS];

    ggml_hpx_cpu_region_group  group;
} ggml_hpx_mlp_glu_group;

static inline void ggml_hpx_mlp_glu_group_init(ggml_hpx_mlp_glu_group * g)
{
    std::memset(g, 0, sizeof(*g));
    for (int i = 0; i < GGML_HPX_MLP_GLU_N_OPS; ++i)
    {
        g->lowers[i].group.regions = g->lowers[i].regions;
    }
    g->group.regions = g->combined_regions;
    g->group.deps    = g->combined_deps;
}

// ---------------------------------------------------------------------------
// Composer
// ---------------------------------------------------------------------------
//
// Lower each of the three ops, copy their single regions into the combined
// flat array, and wire the two cross-op dep edges.
//
// Input nodes:
//   node_gate : result of ggml_mul_mat(W_gate, x)
//   node_up   : result of ggml_mul_mat(W_up,   x)
//   node_glu  : result of ggml_swiglu_split(node_gate, node_up)
//               (op == GGML_OP_GLU, subop == GGML_GLU_OP_SWIGLU,
//                src[0] == node_gate, src[1] == node_up)
//
// Preconditions:
//   - ggml_hpx_mlp_glu_group_init(grp) was called.
//   - All three nodes have type GGML_TYPE_F32.
//   - All data pointers are set and contiguous.
//   - Each op must lower to exactly 1 region.
//   - Operand order is strict: node_glu->src[0] must be node_gate and
//     node_glu->src[1] must be node_up. A reversed order is rejected even
//     though it is mathematically equivalent.
//
// Returns true on success. On failure, grp->group.n_regions is 0.

#ifdef __cplusplus
extern "C" {
#endif

bool ggml_hpx_compose_mlp_glu_group(
    const struct ggml_tensor *    node_gate,
    const struct ggml_tensor *    node_up,
    const struct ggml_tensor *    node_glu,
    ggml_hpx_mlp_glu_group *      grp);

#ifdef __cplusplus
}    // extern "C"
#endif

#endif    // GGML_HPX_REGION_DAG
