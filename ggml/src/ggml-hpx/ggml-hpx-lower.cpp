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

static_assert(sizeof(ggml_hpx_mul_mat_f32_ctx)              <= GGML_HPX_LOWERING_CTX_BYTES_PER_REGION,
    "ggml_hpx_mul_mat_f32_ctx too large for lowering arena; bump CTX_BYTES_PER_REGION");
static_assert(sizeof(ggml_hpx_quantize_q8_k_f32_ctx)        <= GGML_HPX_LOWERING_CTX_BYTES_PER_REGION,
    "ggml_hpx_quantize_q8_k_f32_ctx too large for lowering arena; bump CTX_BYTES_PER_REGION");
static_assert(sizeof(ggml_hpx_mul_mat_q4_k_q8_k_ctx)        <= GGML_HPX_LOWERING_CTX_BYTES_PER_REGION,
    "ggml_hpx_mul_mat_q4_k_q8_k_ctx too large for lowering arena; bump CTX_BYTES_PER_REGION");
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
            0,
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
            0,
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
            0,
        };
        out->group.n_regions = 1;
        return true;
    }

    // ── MUL_MAT ───────────────────────────────────────────────────────────
    //
    // ggml convention: src[0]=weight [cols × out_cols], src[1]=input [cols × rows].
    // The kernel iterates output columns in [0, out_cols) as its work range.
    //
    // Supported weight types:
    //   GGML_TYPE_F32  — 1-region F32 mul_mat (existing path)
    //   GGML_TYPE_Q4_K — 2-region Q4_K decode pipeline, rows == 1 only
    case GGML_OP_MUL_MAT:
    {
        const ggml_tensor * w = node->src[0];    // weight
        const ggml_tensor * x = node->src[1];    // activations
        if (!w || !x || !w->data || !x->data || !node->data) return false;

        const int64_t cols     = w->ne[0];    // shared (reduction) dim
        const int64_t out_cols = w->ne[1];    // output columns
        const int64_t rows     = x->ne[1];    // batch
        if (cols <= 0 || out_cols <= 0 || rows <= 0) return false;
        if (node->ne[0] != out_cols || node->ne[1] != rows) return false;
        if (x->ne[0] != cols) return false;

        // ── F32 × F32 path ────────────────────────────────────────────────
        if (w->type == GGML_TYPE_F32 && x->type == GGML_TYPE_F32)
        {
            if (!is_contiguous_2d_f32(w) || !is_contiguous_2d_f32(x)
             || !is_contiguous_2d_f32(node))
                return false;

            ::new (out->ctx_buf[0]) ggml_hpx_mul_mat_f32_ctx{
                static_cast<const float *>(x->data),
                static_cast<const float *>(w->data),
                static_cast<float *>(node->data),
                rows, cols, out_cols,
            };
            out->regions[0] = {
                GGML_HPX_CPU_REGION_KIND_MATMUL,
                0, out_cols, 0,
                out->ctx_buf[0],
                ggml_hpx_mul_mat_f32_run_range,
                0,
            };
            out->group.n_regions = 1;
            return true;
        }

        // ── Q4_K weight × F32 activation → F32 output, rows == 1 ─────────
        //
        // Two-region pipeline:
        //   R0 (REDUCTION, serial) : quantize F32 x → Q8_K scratch
        //   R1 (MATMUL, parallel)  : Q4_K × Q8_K dot products
        //   dep: 0 → 1
        if (w->type == GGML_TYPE_Q4_K && x->type == GGML_TYPE_F32)
        {
            if (std::getenv("LLAMA_HPX_SELECTIVE_NO_Q4K")) return false;
            if (rows != 1) return false;
            if (cols % ggml_blck_size(GGML_TYPE_Q4_K) != 0) return false;

            // Weight contiguity: nb[0] is one block's byte size, nb[1] is one row.
            const size_t blk_sz  = ggml_type_size(GGML_TYPE_Q4_K);
            const size_t blk_cnt = static_cast<size_t>(
                cols / ggml_blck_size(GGML_TYPE_Q4_K));
            if (w->nb[0] != blk_sz || w->nb[1] != blk_cnt * blk_sz) return false;

            // Input x must be fully contiguous (both element and row strides).
            // Output y: nb[0]==sizeof(float) is sufficient for rows=1 because
            // the kernel writes element j at (char*)y + j*nb[0]; nb[1] may be
            // larger than ne[0]*sizeof(float) for KV-cache view tensors.
            if (!is_contiguous_2d_f32(x)) return false;
            if (node->nb[0] != sizeof(float)) return false;

            // Reject if the Q8_K scratch row won't fit in the lowering arena.
            const size_t q8k_row = ggml_row_size(GGML_TYPE_Q8_K, cols);
            if (q8k_row > GGML_HPX_LOWERING_SCRATCH_BYTES) return false;

            const size_t w_row_stride = blk_cnt * blk_sz;
            void * scratch = out->scratch;   // shared between the two regions

            // R0: serial quantize
            ::new (out->ctx_buf[0]) ggml_hpx_quantize_q8_k_f32_ctx{
                static_cast<const float *>(x->data),
                scratch,
                cols,
            };
            out->regions[0] = {
                GGML_HPX_CPU_REGION_KIND_REDUCTION,
                0, cols, 0,
                out->ctx_buf[0],
                ggml_hpx_quantize_q8_k_f32_run_range,
                0,  // self-sufficient: writes only to ctx->x_q8 (lo.scratch), no external resources
            };

            // R1: parallel dot products
            ::new (out->ctx_buf[1]) ggml_hpx_mul_mat_q4_k_q8_k_ctx{
                w->data,
                scratch,
                static_cast<float *>(node->data),
                cols, out_cols,
                w_row_stride,
                q8k_row,
                node->nb[0],
            };
            out->regions[1] = {
                GGML_HPX_CPU_REGION_KIND_MATMUL,
                0, out_cols, 0,
                out->ctx_buf[1],
                ggml_hpx_mul_mat_q4_k_q8_k_run_range,
                0,
            };

            out->deps[0] = {0, 1};

            out->group.n_regions = 2;
            out->group.n_deps    = 1;
            out->group.deps      = out->deps;
            return true;
        }

        return false;
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
            out->ctx_buf[0], ggml_hpx_rms_norm_partial_f32_run_range, 0,
        };
        out->regions[1] = {
            GGML_HPX_CPU_REGION_KIND_REDUCTION, 0, n, 0,
            out->ctx_buf[1], ggml_hpx_rms_norm_finalize_f32_run_range,
            1,  // reads lane_scratch[0..n_lanes) + writes reduction_buffer
        };
        out->regions[2] = {
            GGML_HPX_CPU_REGION_KIND_ELEMENTWISE, 0, n, 0,
            out->ctx_buf[2], ggml_hpx_rms_norm_apply_f32_run_range, 0,
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
