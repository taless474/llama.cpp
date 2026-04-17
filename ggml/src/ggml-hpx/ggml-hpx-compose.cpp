// ggml-hpx-compose.cpp

#ifndef GGML_HPX_REGION_DAG
#  error "ggml-hpx-compose.cpp requires -DGGML_HPX_REGION_DAG"
#endif

#include "ggml-hpx-compose.h"
#include "ggml.h"    // ggml_get_unary_op, GGML_OP_*, GGML_UNARY_OP_*

// Stable symbolic indices for the four ops in ggml_hpx_mlp_gate_up_group.
enum {
    IDX_GATE     = 0,
    IDX_UP       = 1,
    IDX_GATE_ACT = 2,
    IDX_OUT      = 3,
};

bool ggml_hpx_compose_mlp_gate_up_group(
    const ggml_tensor *          node_gate,
    const ggml_tensor *          node_up,
    const ggml_tensor *          node_gate_act,
    const ggml_tensor *          node_out,
    ggml_hpx_mlp_gate_up_group * grp)
{
    if (!grp) return false;

    // Reset to a self-consistent failure state.  Rewire internal pointers
    // unconditionally so the struct is usable even if the caller skipped init.
    grp->group.n_regions = 0;
    grp->group.n_deps    = 0;
    grp->group.regions   = grp->combined_regions;
    grp->group.deps      = grp->combined_deps;

    if (!node_gate || !node_up || !node_gate_act || !node_out) return false;

    // ── Op-kind checks ────────────────────────────────────────────────────────
    // Reject before touching data pointers or calling lower_op.
    if (node_gate->op != GGML_OP_MUL_MAT) return false;
    if (node_up->op   != GGML_OP_MUL_MAT) return false;
    if (node_gate_act->op != GGML_OP_UNARY
        || ggml_get_unary_op(node_gate_act) != GGML_UNARY_OP_SILU) return false;
    if (node_out->op  != GGML_OP_MUL)     return false;

    // ── Topology checks ───────────────────────────────────────────────────────
    // This is a canonical pattern match: gate_act = silu(gate), out = mul(gate_act, up).
    // mul(up, gate_act) is mathematically equivalent but is not this pattern and
    // is rejected so callers cannot accidentally mis-order the arguments.
    if (node_gate_act->src[0] != node_gate)     return false;
    if (node_out->src[0]      != node_gate_act) return false;
    if (node_out->src[1]      != node_up)       return false;

    // ── Lower each op ─────────────────────────────────────────────────────────
    const ggml_tensor * nodes[GGML_HPX_MLP_GATE_UP_N_OPS] = {
        node_gate, node_up, node_gate_act, node_out,
    };

    for (int i = 0; i < GGML_HPX_MLP_GATE_UP_N_OPS; ++i)
    {
        if (!ggml_hpx_lower_op(nodes[i], &grp->lowers[i])) return false;
        // Composer requires single-region, zero-dep ops only.
        if (grp->lowers[i].group.n_regions != 1) return false;
        if (grp->lowers[i].group.n_deps    != 0) return false;
        grp->combined_regions[i] = grp->lowers[i].regions[0];
    }

    // ── Cross-op dep edges (global combined_regions[] indices) ────────────────
    //   IDX_GATE     → IDX_GATE_ACT   (gate feeds silu)
    //   IDX_UP       → IDX_OUT        (up feeds elementwise mul)
    //   IDX_GATE_ACT → IDX_OUT        (silu feeds elementwise mul)
    grp->combined_deps[0] = {IDX_GATE,     IDX_GATE_ACT};
    grp->combined_deps[1] = {IDX_UP,       IDX_OUT};
    grp->combined_deps[2] = {IDX_GATE_ACT, IDX_OUT};

    grp->group.n_regions = GGML_HPX_MLP_GATE_UP_N_REGIONS;
    grp->group.n_deps    = GGML_HPX_MLP_GATE_UP_N_DEPS;
    return true;
}
