// ggml-hpx-compose.cpp

#ifndef GGML_HPX_REGION_DAG
#  error "ggml-hpx-compose.cpp requires -DGGML_HPX_REGION_DAG"
#endif

#include "ggml-hpx-compose.h"
#include "ggml-hpx-region-exec.h"   // ctx structs + run_range fns for the Q4_Kx8 path
#include "ggml.h"                   // ggml_get_unary_op, GGML_OP_*, GGML_UNARY_OP_*

#include <new>     // placement new for in-arena ctx construction

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

// ---------------------------------------------------------------------------
// ggml_hpx_compose_mlp_glu_group
// ---------------------------------------------------------------------------

namespace {
enum {
    GLU_IDX_GATE = 0,
    GLU_IDX_UP   = 1,
    GLU_IDX_GLU  = 2,
};
}    // namespace

bool ggml_hpx_compose_mlp_glu_group(
    const ggml_tensor *       node_gate,
    const ggml_tensor *       node_up,
    const ggml_tensor *       node_glu,
    ggml_hpx_mlp_glu_group *  grp)
{
    if (!grp) return false;

    grp->group.n_regions = 0;
    grp->group.n_deps    = 0;
    grp->group.regions   = grp->combined_regions;
    grp->group.deps      = grp->combined_deps;

    if (!node_gate || !node_up || !node_glu) return false;

    // ── Op-kind checks ────────────────────────────────────────────────────────
    if (node_gate->op != GGML_OP_MUL_MAT) return false;
    if (node_up->op   != GGML_OP_MUL_MAT) return false;
    if (node_glu->op  != GGML_OP_GLU
        || ggml_get_glu_op(node_glu) != GGML_GLU_OP_SWIGLU) return false;

    // ── Topology checks ───────────────────────────────────────────────────────
    // Operand order is strict: src[0]=gate, src[1]=up.  A reversed pairing is
    // rejected even though SiLU(gate)*up == SiLU(up)*gate only by coincidence;
    // callers must supply nodes in the canonical order.
    if (node_glu->src[0] != node_gate) return false;
    if (node_glu->src[1] != node_up)   return false;

    // ── Lower each op ─────────────────────────────────────────────────────────
    const ggml_tensor * nodes[GGML_HPX_MLP_GLU_N_OPS] = {
        node_gate, node_up, node_glu,
    };

    for (int i = 0; i < GGML_HPX_MLP_GLU_N_OPS; ++i)
    {
        if (!ggml_hpx_lower_op(nodes[i], &grp->lowers[i])) return false;
        if (grp->lowers[i].group.n_regions != 1) return false;
        if (grp->lowers[i].group.n_deps    != 0) return false;
        grp->combined_regions[i] = grp->lowers[i].regions[0];
    }

    // ── Cross-op dep edges (global combined_regions[] indices) ────────────────
    //   GLU_IDX_GATE → GLU_IDX_GLU   (gate feeds SWIGLU)
    //   GLU_IDX_UP   → GLU_IDX_GLU   (up   feeds SWIGLU)
    grp->combined_deps[0] = {GLU_IDX_GATE, GLU_IDX_GLU};
    grp->combined_deps[1] = {GLU_IDX_UP,   GLU_IDX_GLU};

    grp->group.n_regions = GGML_HPX_MLP_GLU_N_REGIONS;
    grp->group.n_deps    = GGML_HPX_MLP_GLU_N_DEPS;
    return true;
}

// ---------------------------------------------------------------------------
// ggml_hpx_compose_mlp_gate_up_glu_q4k8_group
// ---------------------------------------------------------------------------
//
// Hand-built 4-region group for the Q4_Kx8 gate/up/GLU sublayer. Does NOT
// call ggml_hpx_lower_op (would produce two redundant quantize regions).
// Mirrors the ctx field conventions used by ggml-hpx-lower.cpp's Q4_K
// branch exactly — verified field-by-field against lower.cpp:359-373.

namespace {
enum {
    Q4K8_IDX_QUANTIZE = 0,
    Q4K8_IDX_GATE     = 1,
    Q4K8_IDX_UP       = 2,
    Q4K8_IDX_GLU      = 3,
};
}    // namespace

bool ggml_hpx_compose_mlp_gate_up_glu_q4k8_group(
    const ggml_tensor *                       node_gate,
    const ggml_tensor *                       node_up,
    const ggml_tensor *                       node_glu,
    ggml_hpx_mlp_gate_up_glu_q4k8_group *    grp)
{
    if (!grp) return false;

    grp->group.n_regions = 0;
    grp->group.n_deps    = 0;
    grp->group.regions   = grp->combined_regions;
    grp->group.deps      = grp->combined_deps;

    if (!node_gate || !node_up || !node_glu) return false;

    // ── Op-kind checks ────────────────────────────────────────────────────────
    if (node_gate->op != GGML_OP_MUL_MAT) return false;
    if (node_up  ->op != GGML_OP_MUL_MAT) return false;
    if (node_glu ->op != GGML_OP_GLU
        || ggml_get_glu_op(node_glu) != GGML_GLU_OP_SWIGLU) return false;

    // ── Topology checks ───────────────────────────────────────────────────────
    if (node_glu->src[0] != node_gate) return false;
    if (node_glu->src[1] != node_up)   return false;

    const ggml_tensor * x      = node_gate->src[1];
    const ggml_tensor * w_gate = node_gate->src[0];
    const ggml_tensor * w_up   = node_up  ->src[0];
    if (!x || !w_gate || !w_up) return false;
    if (node_up->src[1] != x)   return false;    // gate and up share x

    // ── Type / trait checks ───────────────────────────────────────────────────
    if (node_gate->type != GGML_TYPE_F32) return false;
    if (node_up  ->type != GGML_TYPE_F32) return false;
    if (node_glu ->type != GGML_TYPE_F32) return false;
    if (x->type         != GGML_TYPE_F32) return false;

    if (w_gate->type != GGML_TYPE_Q4_K) return false;
    if (w_up  ->type != GGML_TYPE_Q4_K) return false;
    if (w_gate->extra == nullptr || w_up->extra == nullptr) return false;
    if (!ggml_hpx_is_q4k_8x8_repacked(node_gate)) return false;
    if (!ggml_hpx_is_q4k_8x8_repacked(node_up))   return false;

    // ── Shape / alignment checks ──────────────────────────────────────────────
    const int64_t cols     = w_gate->ne[0];
    const int64_t out_cols = w_gate->ne[1];
    const int64_t rows     = x->ne[1];
    if (cols <= 0 || out_cols <= 0 || rows <= 0)         return false;
    if (rows != 1)                                       return false;
    if (cols % ggml_blck_size(GGML_TYPE_Q4_K) != 0)      return false;    // 256
    if (out_cols % 8 != 0)                               return false;    // NB_COLS

    // gate and up must agree structurally — different repacked-weight nb[1]
    // would produce two compiled packets with different cache keys; reject
    // early so the caller does not see a silently-incompatible packet.
    if (w_up->ne[0] != cols || w_up->ne[1] != out_cols)  return false;
    if (w_up->nb[1] != w_gate->nb[1])                    return false;

    // Output strides — gemv writes element j at y + j * nb[0]; SWIGLU
    // assumes contiguous F32 dst. nb[0] != sizeof(float) is undefined.
    if (node_gate->nb[0] != sizeof(float))               return false;
    if (node_up  ->nb[0] != sizeof(float))               return false;
    if (node_glu ->nb[0] != sizeof(float))               return false;

    // Activation x — rows == 1 so we only read one row; nb[1] is irrelevant
    // but nb[0] must be a contiguous F32 element stride and ne[2,3] must be 1
    // (the kernel does not handle batched / multi-channel x).
    if (x->nb[0] != sizeof(float))                       return false;
    if (x->ne[2] != 1 || x->ne[3] != 1)                  return false;

    // Q8_K row must fit in lowers[0].scratch (8192 B).
    const size_t q8k_row = ggml_row_size(GGML_TYPE_Q8_K, cols);
    if (q8k_row > GGML_HPX_LOWERING_SCRATCH_BYTES)       return false;

    // ── Build the four regions ────────────────────────────────────────────────
    //
    // ctx values are templates: per-call pointers (x, w, y, gate/up/dst) get
    // patched by ggml_hpx_bind_mlp_gate_up_glu_q4k8_packet at bind time, and
    // x_q8 is rewired to the frame's scratch slot at the same point. The
    // values set here keep the group runnable as-is via ggml_hpx_run_region_group
    // for direct testing, sharing lowers[0].scratch across R0/R1/R2.

    void * shared_scratch = grp->lowers[Q4K8_IDX_QUANTIZE].scratch;

    // R0: quantize x → shared_scratch (Q8_K row).
    ::new (grp->lowers[Q4K8_IDX_QUANTIZE].ctx_buf[0])
        ggml_hpx_quantize_q8_k_f32_ctx{
            static_cast<const float *>(x->data),
            shared_scratch,
            cols,
        };
    grp->combined_regions[Q4K8_IDX_QUANTIZE] = {
        GGML_HPX_CPU_REGION_KIND_REDUCTION,
        0, cols, 0,
        grp->lowers[Q4K8_IDX_QUANTIZE].ctx_buf[0],
        ggml_hpx_quantize_q8_k_f32_run_range,
        0,    // self-sufficient: writes only to ctx->x_q8
    };

    // R1: gate gemv. Field convention copied from lower.cpp:359-366.
    ::new (grp->lowers[Q4K8_IDX_GATE].ctx_buf[0])
        ggml_hpx_mul_mat_q4_k_8x8_q8_k_ctx{
            w_gate->data,
            shared_scratch,
            static_cast<float *>(node_gate->data),
            cols, out_cols,
            w_gate->nb[1],     // repacked weight row stride (NOT block size)
            node_gate->nb[0],  // == sizeof(float), validated above
        };
    grp->combined_regions[Q4K8_IDX_GATE] = {
        GGML_HPX_CPU_REGION_KIND_MATMUL,
        0, out_cols, 0,
        grp->lowers[Q4K8_IDX_GATE].ctx_buf[0],
        ggml_hpx_mul_mat_q4_k_8x8_q8_k_run_range,
        0,
    };

    // R2: up gemv. Same field convention as R1; different weight + output.
    ::new (grp->lowers[Q4K8_IDX_UP].ctx_buf[0])
        ggml_hpx_mul_mat_q4_k_8x8_q8_k_ctx{
            w_up->data,
            shared_scratch,
            static_cast<float *>(node_up->data),
            cols, out_cols,
            w_up->nb[1],       // == w_gate->nb[1] by check above
            node_up->nb[0],
        };
    grp->combined_regions[Q4K8_IDX_UP] = {
        GGML_HPX_CPU_REGION_KIND_MATMUL,
        0, out_cols, 0,
        grp->lowers[Q4K8_IDX_UP].ctx_buf[0],
        ggml_hpx_mul_mat_q4_k_8x8_q8_k_run_range,
        0,
    };

    // R3: SWIGLU. n is total element count; rows == 1 so n == out_cols.
    ::new (grp->lowers[Q4K8_IDX_GLU].ctx_buf[0])
        ggml_hpx_swiglu_f32_ctx{
            static_cast<const float *>(node_gate->data),
            static_cast<const float *>(node_up  ->data),
            static_cast<float *>(node_glu->data),
            out_cols,
        };
    grp->combined_regions[Q4K8_IDX_GLU] = {
        GGML_HPX_CPU_REGION_KIND_ELEMENTWISE,
        0, out_cols, 0,
        grp->lowers[Q4K8_IDX_GLU].ctx_buf[0],
        ggml_hpx_swiglu_f32_run_range,
        0,
    };

    // ── Cross-op dep edges (global combined_regions[] indices) ────────────────
    //   QUANTIZE → GATE   (R1 reads x_q8 written by R0)
    //   QUANTIZE → UP     (R2 reads x_q8 written by R0)
    //   GATE     → GLU    (R3 reads gate written by R1)
    //   UP       → GLU    (R3 reads up   written by R2)
    grp->combined_deps[0] = {Q4K8_IDX_QUANTIZE, Q4K8_IDX_GATE};
    grp->combined_deps[1] = {Q4K8_IDX_QUANTIZE, Q4K8_IDX_UP};
    grp->combined_deps[2] = {Q4K8_IDX_GATE,     Q4K8_IDX_GLU};
    grp->combined_deps[3] = {Q4K8_IDX_UP,       Q4K8_IDX_GLU};

    grp->group.n_regions = GGML_HPX_MLP_GATE_UP_GLU_Q4K8_N_REGIONS;
    grp->group.n_deps    = GGML_HPX_MLP_GATE_UP_GLU_Q4K8_N_DEPS;
    return true;
}
