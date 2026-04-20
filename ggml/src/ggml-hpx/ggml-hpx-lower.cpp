// ggml-hpx-lower.cpp
//
// Op lowering: ggml_hpx_lower_op translates one ggml_tensor compute node
// into a ggml_hpx_lowering (a self-contained fine-region group + inline ctx
// storage).  See docs/HPX_LOWER_OP.md for the full contract.
//
// Supported ops (first implementation):
//   GGML_OP_UNARY / GGML_UNARY_OP_SILU   → 1-region ELEMENTWISE
//   GGML_OP_MUL                           → 1-region ELEMENTWISE
//   GGML_OP_MUL_MAT                       → 1-region MATMUL
//   GGML_OP_RMS_NORM (ne[1]==1 only)      → 3-region DAG (partial/finalize/apply)
//   GGML_OP_GLU / GGML_GLU_OP_SWIGLU      → 1-region ELEMENTWISE (fused SwiGLU)

#ifndef GGML_HPX_REGION_DAG
#  error "ggml-hpx-lower.cpp requires -DGGML_HPX_REGION_DAG"
#endif

#include "ggml-hpx-lower.h"
#include "ggml-hpx-region-exec.h"

#include "ggml.h"

#include <cassert>
#include <cstring>
#include <new>

// ---------------------------------------------------------------------------
// Compile-time size guards — every emitted ctx must fit in one slot.
// Bump GGML_HPX_LOWERING_CTX_BYTES_PER_REGION (and update HPX_LOWER_OP.md)
// if any assert fires; do NOT just delete the assert.
// ---------------------------------------------------------------------------

static_assert(sizeof(ggml_hpx_mul_mat_f32_ctx)          <= GGML_HPX_LOWERING_CTX_BYTES_PER_REGION,
    "ggml_hpx_mul_mat_f32_ctx too large for lowering arena; bump CTX_BYTES_PER_REGION");
static_assert(sizeof(ggml_hpx_silu_f32_ctx)             <= GGML_HPX_LOWERING_CTX_BYTES_PER_REGION,
    "ggml_hpx_silu_f32_ctx too large for lowering arena; bump CTX_BYTES_PER_REGION");
static_assert(sizeof(ggml_hpx_mul_f32_ctx)              <= GGML_HPX_LOWERING_CTX_BYTES_PER_REGION,
    "ggml_hpx_mul_f32_ctx too large for lowering arena; bump CTX_BYTES_PER_REGION");
static_assert(sizeof(ggml_hpx_swiglu_f32_ctx)           <= GGML_HPX_LOWERING_CTX_BYTES_PER_REGION,
    "ggml_hpx_swiglu_f32_ctx too large for lowering arena; bump CTX_BYTES_PER_REGION");
static_assert(sizeof(ggml_hpx_rms_norm_partial_f32_ctx) <= GGML_HPX_LOWERING_CTX_BYTES_PER_REGION,
    "ggml_hpx_rms_norm_partial_f32_ctx too large for lowering arena; bump CTX_BYTES_PER_REGION");
static_assert(sizeof(ggml_hpx_rms_norm_finalize_f32_ctx)<= GGML_HPX_LOWERING_CTX_BYTES_PER_REGION,
    "ggml_hpx_rms_norm_finalize_f32_ctx too large for lowering arena; bump CTX_BYTES_PER_REGION");
static_assert(sizeof(ggml_hpx_rms_norm_apply_f32_ctx)   <= GGML_HPX_LOWERING_CTX_BYTES_PER_REGION,
    "ggml_hpx_rms_norm_apply_f32_ctx too large for lowering arena; bump CTX_BYTES_PER_REGION");

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

// True iff the tensor's innermost stride equals the element size, i.e. the
// elements are stored contiguously in memory. Necessary for kernels that
// cast data to float* and index with a bare integer.
static bool is_contiguous_f32(const ggml_tensor * t)
{
    return t->nb[0] == sizeof(float);
}

// True iff the tensor is row-major contiguous in 2D: elements within a row
// are adjacent and rows are not padded. Required by MUL_MAT which accesses
// x_row = data + row * ne[0] and w_row = data + col * ne[0].
static bool is_contiguous_2d_f32(const ggml_tensor * t)
{
    return t->nb[0] == sizeof(float)
        && t->nb[1] == static_cast<size_t>(t->ne[0]) * sizeof(float);
}

// ---------------------------------------------------------------------------
// ggml_hpx_lower_op
// ---------------------------------------------------------------------------

bool ggml_hpx_lower_op(
    const ggml_tensor * node,
    ggml_hpx_lowering * out)
{
    assert(node != nullptr);
    assert(out  != nullptr);

    // Reset to failure state. Wire internal pointers unconditionally so the
    // struct is self-consistent even if the caller skipped lowering_init.
    out->group.regions   = out->regions;
    out->group.n_regions = 0;
    out->group.deps      = nullptr;
    out->group.n_deps    = 0;

    if (node->type != GGML_TYPE_F32) return false;

    switch (node->op)
    {
    // ── SILU ──────────────────────────────────────────────────────────────
    case GGML_OP_UNARY:
    {
        if (ggml_get_unary_op(node) != GGML_UNARY_OP_SILU) return false;

        const ggml_tensor * src = node->src[0];
        if (!src || !src->data || !node->data) return false;
        if (src->type != GGML_TYPE_F32) return false;
        if (!is_contiguous_f32(src) || !is_contiguous_f32(node)) return false;

        const int64_t n = ggml_nelements(node);

        ::new (out->ctx_buf[0]) ggml_hpx_silu_f32_ctx{
            static_cast<const float *>(src->data),
            static_cast<float *>(node->data),
            n,
        };
        out->regions[0] = {
            GGML_HPX_CPU_REGION_KIND_ELEMENTWISE,
            0, n, 0,
            out->ctx_buf[0],
            ggml_hpx_silu_f32_run_range,
        };
        out->group.n_regions = 1;
        return true;
    }

    // ── Elementwise MUL ───────────────────────────────────────────────────
    case GGML_OP_MUL:
    {
        const ggml_tensor * a = node->src[0];
        const ggml_tensor * b = node->src[1];
        if (!a || !b || !a->data || !b->data || !node->data) return false;
        if (a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32) return false;

        const int64_t n = ggml_nelements(node);
        // Both sources must cover exactly the same element count as the output.
        if (ggml_nelements(a) != n || ggml_nelements(b) != n) return false;
        if (!is_contiguous_f32(a) || !is_contiguous_f32(b) || !is_contiguous_f32(node))
            return false;

        ::new (out->ctx_buf[0]) ggml_hpx_mul_f32_ctx{
            static_cast<const float *>(a->data),
            static_cast<const float *>(b->data),
            static_cast<float *>(node->data),
            n,
        };
        out->regions[0] = {
            GGML_HPX_CPU_REGION_KIND_ELEMENTWISE,
            0, n, 0,
            out->ctx_buf[0],
            ggml_hpx_mul_f32_run_range,
        };
        out->group.n_regions = 1;
        return true;
    }

    // ── GLU[SWIGLU] (fused SiLU(gate) * up) ───────────────────────────────
    //
    // ggml emits this as a single GGML_OP_GLU node with subop SWIGLU when the
    // MLP is built via ggml_swiglu_split (src/llama-graph.cpp:1074).  v1's
    // matcher expects three nodes (SiLU, MUL_MAT, MUL) and therefore does
    // not fire on real llama graphs.  B.1 lowers the fused op into a single
    // ELEMENTWISE region backed by the SWIGLU_F32 kernel.
    case GGML_OP_GLU:
    {
        if (ggml_get_glu_op(node) != GGML_GLU_OP_SWIGLU) return false;

        const ggml_tensor * gate = node->src[0];
        const ggml_tensor * up   = node->src[1];
        if (!gate || !up || !gate->data || !up->data || !node->data) return false;
        if (gate->type != GGML_TYPE_F32 || up->type != GGML_TYPE_F32)  return false;

        const int64_t n = ggml_nelements(node);
        if (ggml_nelements(gate) != n || ggml_nelements(up) != n) return false;
        if (!is_contiguous_f32(gate) || !is_contiguous_f32(up) || !is_contiguous_f32(node))
            return false;

        ::new (out->ctx_buf[0]) ggml_hpx_swiglu_f32_ctx{
            static_cast<const float *>(gate->data),
            static_cast<const float *>(up  ->data),
            static_cast<float *>(node->data),
            n,
        };
        out->regions[0] = {
            GGML_HPX_CPU_REGION_KIND_ELEMENTWISE,
            0, n, 0,
            out->ctx_buf[0],
            ggml_hpx_swiglu_f32_run_range,
        };
        out->group.n_regions = 1;
        return true;
    }

    // ── MUL_MAT ───────────────────────────────────────────────────────────
    //
    // ggml convention: src[0]=weight [cols × out_cols], src[1]=input [cols × rows].
    // The kernel iterates output columns in [0, out_cols) as its work range.
    case GGML_OP_MUL_MAT:
    {
        const ggml_tensor * w = node->src[0];    // weight
        const ggml_tensor * x = node->src[1];    // activations
        if (!w || !x || !w->data || !x->data || !node->data) return false;
        if (w->type != GGML_TYPE_F32 || x->type != GGML_TYPE_F32) return false;

        const int64_t cols     = w->ne[0];    // shared (reduction) dim
        const int64_t out_cols = w->ne[1];    // output columns; also node->ne[0]
        const int64_t rows     = x->ne[1];    // batch; also node->ne[1]
        if (cols <= 0 || out_cols <= 0 || rows <= 0) return false;
        // Verify the output tensor shape is consistent with the sources.
        if (node->ne[0] != out_cols || node->ne[1] != rows) return false;
        // Shared dimension must match between weight and activation.
        if (x->ne[0] != cols) return false;
        if (!is_contiguous_2d_f32(w) || !is_contiguous_2d_f32(x) || !is_contiguous_2d_f32(node))
            return false;

        ::new (out->ctx_buf[0]) ggml_hpx_mul_mat_f32_ctx{
            static_cast<const float *>(x->data),
            static_cast<const float *>(w->data),
            static_cast<float *>(node->data),
            rows,
            cols,
            out_cols,
        };
        out->regions[0] = {
            GGML_HPX_CPU_REGION_KIND_MATMUL,
            0, out_cols, 0,
            out->ctx_buf[0],
            ggml_hpx_mul_mat_f32_run_range,
        };
        out->group.n_regions = 1;
        return true;
    }

    // ── RMS_NORM ──────────────────────────────────────────────────────────
    //
    // Three-region decomposition (see ggml-hpx-region-exec.h for the protocol):
    //   R0 (ELEMENTWISE): lane-parallel partial sumsq → lane_scratch[ith]
    //   R1 (REDUCTION)  : single-thread finalize      → reduction_buffer
    //   R2 (ELEMENTWISE): lane-parallel apply         → dst
    //
    // Prototype limitation: single-row tensors only (ne[1] == 1).
    case GGML_OP_RMS_NORM:
    {
        const ggml_tensor * src = node->src[0];
        if (!src || !src->data || !node->data) return false;
        if (src->type != GGML_TYPE_F32) return false;
        if (node->ne[1] != 1) return false;    // multi-row not yet supported
        if (!is_contiguous_f32(src) || !is_contiguous_f32(node)) return false;

        const int64_t n = node->ne[0];
        float eps = 0.f;
        std::memcpy(&eps, node->op_params, sizeof(float));

        ::new (out->ctx_buf[0]) ggml_hpx_rms_norm_partial_f32_ctx{
            static_cast<const float *>(src->data),
            n,
        };
        ::new (out->ctx_buf[1]) ggml_hpx_rms_norm_finalize_f32_ctx{n, eps};
        ::new (out->ctx_buf[2]) ggml_hpx_rms_norm_apply_f32_ctx{
            static_cast<const float *>(src->data),
            static_cast<float *>(node->data),
            n,
        };

        out->regions[0] = {
            GGML_HPX_CPU_REGION_KIND_ELEMENTWISE, 0, n, 0,
            out->ctx_buf[0], ggml_hpx_rms_norm_partial_f32_run_range,
        };
        out->regions[1] = {
            GGML_HPX_CPU_REGION_KIND_REDUCTION, 0, n, 0,
            out->ctx_buf[1], ggml_hpx_rms_norm_finalize_f32_run_range,
        };
        out->regions[2] = {
            GGML_HPX_CPU_REGION_KIND_ELEMENTWISE, 0, n, 0,
            out->ctx_buf[2], ggml_hpx_rms_norm_apply_f32_run_range,
        };

        out->deps[0] = {0, 1};
        out->deps[1] = {1, 2};

        out->group.n_regions = 3;
        out->group.n_deps    = 2;
        out->group.deps      = out->deps;
        return true;
    }

    default:
        return false;
    }
}
