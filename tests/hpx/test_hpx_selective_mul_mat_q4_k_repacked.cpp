// test_hpx_selective_mul_mat_q4_k_repacked.cpp
//
// CPU_REPACK Q4_K (q4_K_8x8_q8_K trait) MUL_MAT through the HPX lowered path.
//
// Companion to test_hpx_selective_mul_mat_q4_k.cpp, which covers the
// non-repacked Q4_K layout.  This test allocates the weight via the
// CPU_REPACK buffer type so init_tensor populates `extra` to the
// q4_K_8x8_q8_K trait instance and set_tensor repacks the data into
// block_q4_Kx8 layout.  The selective executor must then route the
// MUL_MAT through the new R0 (quantize) + R1 (gemv_q4_K_8x8_q8_K) group
// rather than falling back to the CPU backend, and the output must match
// the CPU backend reference (which itself runs the same gemv kernel).
//
// Reality assertions (each test cell, before the executor runs):
//   - W->extra != nullptr
//   - ggml_repack_extra_traits_name(y) == "q4_K_8x8_q8_K"
//
// Selective-executor assertions:
//   - stats.lowered_nodes  == 1
//   - stats.fallback_nodes == 0
//
// Numerical: HPX output bit-equal to ggml CPU backend on the same
// repacked weight (same kernel body).
//
// Gated on GGML_HPX_REGION_DAG.

#include <gtest/gtest.h>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include "ggml-hpx-exec-selective.h"
#include "ggml-hpx-runtime.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#ifdef GGML_HPX_REGION_DAG

// repack.h declares this with C++ linkage (not inside extern "C").  Same
// pattern as bench-hpx-route-q4k-gemv.cpp.
ggml_backend_buffer_type_t ggml_backend_cpu_repack_buffer_type(void);

extern "C" {
void         quantize_row_q4_K           (const float * x, void * y, int64_t k);
const char * ggml_repack_extra_traits_name(const struct ggml_tensor * op);
}

namespace {

// Build one MUL_MAT graph whose Q4_K weight is allocated through the
// CPU_REPACK buffer type.  The buffer's init_tensor populates W->extra
// with the trait pointer, and set_tensor repacks Q4_K block bytes into
// the trait-specific (e.g. block_q4_Kx8) layout in place.
struct RepackQ4KMulMatGraph
{
    ggml_context *        ctx   = nullptr;
    ggml_cgraph *         gf    = nullptr;
    ggml_backend_buffer_t W_buf = nullptr;

    ggml_tensor * W = nullptr;
    ggml_tensor * x = nullptr;
    ggml_tensor * y = nullptr;

    RepackQ4KMulMatGraph(int64_t cols, int64_t out_cols, int64_t rows)
    {
        ggml_init_params p{};
        p.mem_size   = 4 * 1024 * 1024;
        p.mem_buffer = nullptr;
        p.no_alloc   = true;
        ctx = ggml_init(p);

        W = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, cols, out_cols);
        x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,  cols, rows);
        y = ggml_mul_mat(ctx, W, x);

        gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);

        ggml_backend_buffer_type_t buft = ggml_backend_cpu_repack_buffer_type();
        if (buft == nullptr) return;

        W_buf = ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(W));
        if (W_buf == nullptr) return;

        ggml_tallocr alloc = ggml_tallocr_new(W_buf);
        ggml_tallocr_alloc(&alloc, W);
    }

    bool valid() const { return ctx && W_buf && W && W->buffer; }

    // Push standard Q4_K row bytes; the repack buffer's set_tensor
    // re-lays them into the trait's block layout.
    void load_weight(const uint8_t * src_q4k, size_t bytes)
    {
        ggml_backend_tensor_set(W, src_q4k, 0, bytes);
    }

    ~RepackQ4KMulMatGraph()
    {
        if (W_buf) ggml_backend_buffer_free(W_buf);
        if (ctx)   ggml_free(ctx);
    }

    RepackQ4KMulMatGraph(const RepackQ4KMulMatGraph &)             = delete;
    RepackQ4KMulMatGraph & operator=(const RepackQ4KMulMatGraph &) = delete;
};

// Build a deterministic F32 weight, quantize to standard Q4_K bytes, and
// build a deterministic F32 activation row.  The pattern matches what
// bench-cpu-repack-q4k-gemv.cpp uses, so anyone debugging numerical
// drift can correlate output values across the two harnesses.
void build_inputs(int64_t cols, int64_t out_cols, int64_t rows,
                  std::vector<uint8_t> & W_q4k_out,
                  std::vector<float>   & x_out)
{
    std::vector<float> W_f32(static_cast<size_t>(cols * out_cols));
    for (int64_t j = 0; j < out_cols; ++j)
    {
        for (int64_t k = 0; k < cols; ++k)
        {
            W_f32[static_cast<size_t>(j * cols + k)] =
                0.01f * static_cast<float>((j + 1) * ((k % 17) - 8));
        }
    }

    const size_t row_bytes = ggml_row_size(GGML_TYPE_Q4_K, cols);
    W_q4k_out.assign(row_bytes * static_cast<size_t>(out_cols), 0);
    for (int64_t j = 0; j < out_cols; ++j)
    {
        quantize_row_q4_K(
            W_f32.data() + static_cast<size_t>(j * cols),
            W_q4k_out.data() + static_cast<size_t>(j) * row_bytes,
            cols);
    }

    x_out.assign(static_cast<size_t>(cols * rows), 0.0f);
    for (int64_t k = 0; k < cols * rows; ++k)
    {
        x_out[static_cast<size_t>(k)] =
            0.02f * static_cast<float>((k % 13) - 6);
    }
}

// Test body shared across shapes.  Builds two graphs (ref / hpx), each
// with its own repack buffer for W, pushes the same Q4_K bytes into both
// (so each gets independently repacked into the same block_q4_Kx8
// layout), runs ref via the CPU backend and hpx via the selective
// executor, and asserts the trait, the lowered/fallback counts, and
// numerical equality.
void run_repacked_q4k_test(int64_t cols, int64_t out_cols, int64_t rows)
{
    ASSERT_EQ(cols % 256, 0)   << "cols must be a multiple of QK_K=256";
    ASSERT_EQ(out_cols % 8, 0) << "out_cols must be a multiple of NB_COLS=8 for the 8x8 trait";

    RepackQ4KMulMatGraph ref(cols, out_cols, rows);
    RepackQ4KMulMatGraph hpx(cols, out_cols, rows);
    ASSERT_TRUE(ref.valid()) << "CPU_REPACK buffer alloc failed for ref graph";
    ASSERT_TRUE(hpx.valid()) << "CPU_REPACK buffer alloc failed for hpx graph";

    std::vector<uint8_t> W_q4k_blob;
    std::vector<float>   x_data;
    build_inputs(cols, out_cols, rows, W_q4k_blob, x_data);

    ref.load_weight(W_q4k_blob.data(), W_q4k_blob.size());
    hpx.load_weight(W_q4k_blob.data(), W_q4k_blob.size());

    // Reality check: both graphs must end up on the q4_K_8x8_q8_K trait
    // for this test to be meaningful.  If the host doesn't satisfy
    // (NEON + matmul-int8) or (AVX2), the repack buffer may pick a
    // different trait or skip repack entirely; in that case there is no
    // 8x8 path to exercise and the test should be considered N/A
    // rather than failing the assertion.
    const char * ref_trait = ggml_repack_extra_traits_name(ref.y);
    const char * hpx_trait = ggml_repack_extra_traits_name(hpx.y);
    ASSERT_NE(ref_trait, nullptr) << "ref W->extra not populated by repack init_tensor";
    ASSERT_NE(hpx_trait, nullptr) << "hpx W->extra not populated by repack init_tensor";
    ASSERT_STREQ(ref_trait, "q4_K_8x8_q8_K");
    ASSERT_STREQ(hpx_trait, "q4_K_8x8_q8_K");

    std::vector<float> y_ref(static_cast<size_t>(out_cols * rows), 0.0f);
    std::vector<float> y_hpx(static_cast<size_t>(out_cols * rows), 0.0f);
    ref.x->data = x_data.data();
    ref.y->data = y_ref.data();
    hpx.x->data = x_data.data();
    hpx.y->data = y_hpx.data();

    ggml_backend_t cpu_be = ggml_backend_cpu_init();
    ASSERT_NE(cpu_be, nullptr);

    // Reference: ggml CPU backend on the repacked W.  The CPU backend
    // dispatches via tensor_traits' repack_cpu_op; this calls the same
    // ggml_gemv_q4_K_8x8_q8_K kernel the HPX path uses, so outputs must
    // be bit-equal — there is no kernel-level FP nondeterminism here.
    ASSERT_EQ(ggml_backend_graph_compute(cpu_be, ref.gf), GGML_STATUS_SUCCESS);

    // HPX selective path.
    ggml_hpx_tpool_start();

    ggml_hpx_selective_stats stats{};
    ASSERT_TRUE(ggml_hpx_exec_graph_selective_mul_mat(
        hpx.gf,
        cpu_be,
        /*n_lanes=*/0,         // 0 ⇒ use hpx::get_num_worker_threads()
        &stats,
        /*packet_env=*/nullptr));

    EXPECT_EQ(stats.lowered_nodes,  1u)
        << "lowered_nodes should be 1; the trait-aware Q4_K branch in "
           "ggml_hpx_lower_op did not accept this node";
    EXPECT_EQ(stats.fallback_nodes, 0u)
        << "fallback_nodes should be 0; if non-zero, the executor's "
           "prescan classifier (q4k_decode_safe in selective.cpp) "
           "is out of sync with lower_op's acceptance";

    // Bit-equal output check.  Same kernel body; if these differ we
    // either have a chunk-alignment bug, a stride bug, or are dispatching
    // to the wrong kernel.
    for (int64_t j = 0; j < out_cols * rows; ++j)
    {
        EXPECT_FLOAT_EQ(y_hpx[static_cast<size_t>(j)],
                        y_ref[static_cast<size_t>(j)])
            << "j=" << j;
    }

    ggml_backend_free(cpu_be);
}

}    // namespace

// Smallest valid shape under the 8x8 trait: cols=256 (1 QK_K block),
// out_cols=8 (1 NB_COLS tile).  Exercises the boundary path where the
// chunk grid produces nchunk0 == 1 with dr0 == 8.
TEST(SelectiveMulMatQ4KRepacked, TinyShape_256x8)
{
    run_repacked_q4k_test(/*cols=*/256, /*out_cols=*/8, /*rows=*/1);
}

// TinyLlama Q/K/V/attn_out shape (66 ops/step, 1.59x bench speedup at
// 3 workers): cols=2048, out_cols=2048.
TEST(SelectiveMulMatQ4KRepacked, QKVProjectionShape_2048x2048)
{
    run_repacked_q4k_test(/*cols=*/2048, /*out_cols=*/2048, /*rows=*/1);
}

// TinyLlama gate/up shape (44 ops/step, 1.30x bench speedup at 4 workers):
// cols=2048, out_cols=5632.
TEST(SelectiveMulMatQ4KRepacked, MLPProjectionShape_2048x5632)
{
    run_repacked_q4k_test(/*cols=*/2048, /*out_cols=*/5632, /*rows=*/1);
}

// Off-shape: out_cols=64 — 8-aligned but neither the 2048×N nor the
// 256-block boundary case.  Catches alignment bugs that only surface
// when nchunk0 falls between the per-worker break-even and the tile
// minimum.
TEST(SelectiveMulMatQ4KRepacked, OffTileShape_2048x64)
{
    run_repacked_q4k_test(/*cols=*/2048, /*out_cols=*/64, /*rows=*/1);
}

// TinyLlama MLP down-projection shape (12 ops/step on real decode):
// cols=5632, out_cols=2048.  This is the *reduction-dim* large case
// (cols=5632 needs Q8_K row of 22 × 292 = 6424 bytes) — pre-bump the
// 4 KB lowering scratch arena rejected this shape and silently routed
// the 12 nodes/step to the CPU fallback (see
// hpx-bench/results/2026-04-28-q4k-repack-real-tinyllama/README.md).
// Locks GGML_HPX_LOWERING_SCRATCH_BYTES against future regressions
// below 6424 bytes, and locks the cols=5632 path against future
// shape-derived stride or alignment bugs.
TEST(SelectiveMulMatQ4KRepacked, MLPDownProjectionShape_5632x2048)
{
    run_repacked_q4k_test(/*cols=*/5632, /*out_cols=*/2048, /*rows=*/1);
}

#endif    // GGML_HPX_REGION_DAG
