// test_hpx_selective_mul_mat_q4_k.cpp
//
// B4: Q4_K MUL_MAT through the HPX lowered path.
//
// Proves that ggml_hpx_exec_graph_selective_mul_mat routes a single
// Q4_K × F32 MUL_MAT node through the HPX fine-region path rather than
// the CPU-backend fallback:
//
//   stats.lowered_nodes  == 1  (2-region Q4_K group ran via HPX)
//   stats.fallback_nodes == 0  (no CPU fallback)
//
// Correctness is verified by comparing the HPX output element-wise against
// the ggml CPU backend reference on the same quantized weight tensor.
//
// Shape: cols=256 (one QK_K block), out_cols=4, rows=1.
//
// Gated on GGML_HPX_REGION_DAG.

#include <gtest/gtest.h>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include "ggml-hpx-exec-selective.h"
#include "ggml-hpx-runtime.h"

#include <cstddef>
#include <vector>

#ifdef GGML_HPX_REGION_DAG

extern "C" {
void quantize_row_q4_K(const float * x, void * y, int64_t k);
}

namespace {

struct Q4KMulMatGraph
{
    ggml_context * ctx = nullptr;
    ggml_cgraph *  gf  = nullptr;

    ggml_tensor * W_q4k = nullptr;
    ggml_tensor * x     = nullptr;
    ggml_tensor * y     = nullptr;

    Q4KMulMatGraph(int64_t cols, int64_t out_cols, int64_t rows)
    {
        ggml_init_params p{};
        p.mem_size   = 2 * 1024 * 1024;
        p.mem_buffer = nullptr;
        p.no_alloc   = true;
        ctx = ggml_init(p);

        W_q4k = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, cols, out_cols);
        x     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,  cols, rows);
        y     = ggml_mul_mat(ctx, W_q4k, x);

        gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
    }

    ~Q4KMulMatGraph()
    {
        if (ctx)
        {
            ggml_free(ctx);
        }
    }

    Q4KMulMatGraph(const Q4KMulMatGraph &)             = delete;
    Q4KMulMatGraph & operator=(const Q4KMulMatGraph &) = delete;
};

}    // namespace

TEST(SelectiveMulMatQ4K, DecodeRowsOneLowersAndMatchesCpuBackend)
{
    constexpr int64_t cols     = 256;    // Q4_K requires multiples of QK_K = 256
    constexpr int64_t out_cols = 4;
    constexpr int64_t rows     = 1;

    Q4KMulMatGraph ref(cols, out_cols, rows);
    Q4KMulMatGraph hpx(cols, out_cols, rows);

    const size_t w_row_bytes = ggml_row_size(GGML_TYPE_Q4_K, cols);
    const size_t w_bytes     = w_row_bytes * static_cast<size_t>(out_cols);

    std::vector<float>   W_f32(static_cast<size_t>(cols * out_cols));
    std::vector<uint8_t> W_q4k(w_bytes);
    std::vector<float>   x_data(static_cast<size_t>(cols * rows));
    std::vector<float>   y_ref(static_cast<size_t>(out_cols * rows), 0.0f);
    std::vector<float>   y_hpx(static_cast<size_t>(out_cols * rows), 0.0f);

    for (int64_t j = 0; j < out_cols; ++j)
    {
        for (int64_t k = 0; k < cols; ++k)
        {
            W_f32[static_cast<size_t>(j * cols + k)] =
                0.01f * static_cast<float>((j + 1) * ((k % 17) - 8));
        }
    }

    for (int64_t k = 0; k < cols; ++k)
    {
        x_data[static_cast<size_t>(k)] =
            0.02f * static_cast<float>((k % 13) - 6);
    }

    for (int64_t j = 0; j < out_cols; ++j)
    {
        quantize_row_q4_K(
            W_f32.data()  + static_cast<size_t>(j * cols),
            W_q4k.data()  + static_cast<size_t>(j) * w_row_bytes,
            cols);
    }

    ref.W_q4k->data = W_q4k.data();
    ref.x->data     = x_data.data();
    ref.y->data     = y_ref.data();

    hpx.W_q4k->data = W_q4k.data();
    hpx.x->data     = x_data.data();
    hpx.y->data     = y_hpx.data();

    ggml_backend_t cpu_be = ggml_backend_cpu_init();
    ASSERT_NE(cpu_be, nullptr);

    // Reference: ggml CPU backend.
    ASSERT_EQ(ggml_backend_graph_compute(cpu_be, ref.gf), GGML_STATUS_SUCCESS);

    // HPX selective path.
    ggml_hpx_tpool_start();

    ggml_hpx_selective_stats stats{};
    ASSERT_TRUE(ggml_hpx_exec_graph_selective_mul_mat(
        hpx.gf,
        cpu_be,
        /*n_lanes=*/1,
        &stats,
        /*packet_env=*/nullptr));

    EXPECT_EQ(stats.lowered_nodes,  1u);
    EXPECT_EQ(stats.fallback_nodes, 0u);

    constexpr float kTol = 1e-4f;
    for (int64_t j = 0; j < out_cols * rows; ++j)
    {
        EXPECT_NEAR(y_hpx[static_cast<size_t>(j)],
                    y_ref[static_cast<size_t>(j)],
                    kTol)
            << "j=" << j;
    }

    ggml_backend_free(cpu_be);
}

// K/V projection shape: cols=2048 (8 QK_K blocks), out_cols=256, rows=1.
TEST(SelectiveMulMatQ4K, KVProjectionShape_2048x256)
{
    constexpr int64_t cols     = 2048;    // 8 × QK_K
    constexpr int64_t out_cols = 256;
    constexpr int64_t rows     = 1;

    Q4KMulMatGraph ref(cols, out_cols, rows);
    Q4KMulMatGraph hpx(cols, out_cols, rows);

    const size_t w_row_bytes = ggml_row_size(GGML_TYPE_Q4_K, cols);
    const size_t w_bytes     = w_row_bytes * static_cast<size_t>(out_cols);

    std::vector<float>   W_f32(static_cast<size_t>(cols * out_cols));
    std::vector<uint8_t> W_q4k(w_bytes);
    std::vector<float>   x_data(static_cast<size_t>(cols * rows));
    std::vector<float>   y_ref(static_cast<size_t>(out_cols * rows), 0.0f);
    std::vector<float>   y_hpx(static_cast<size_t>(out_cols * rows), 0.0f);

    for (int64_t j = 0; j < out_cols; ++j)
        for (int64_t k = 0; k < cols; ++k)
            W_f32[static_cast<size_t>(j * cols + k)] =
                0.01f * static_cast<float>((j + 1) * ((k % 17) - 8));

    for (int64_t k = 0; k < cols; ++k)
        x_data[static_cast<size_t>(k)] =
            0.02f * static_cast<float>((k % 13) - 6);

    for (int64_t j = 0; j < out_cols; ++j)
        quantize_row_q4_K(
            W_f32.data()  + static_cast<size_t>(j * cols),
            W_q4k.data()  + static_cast<size_t>(j) * w_row_bytes,
            cols);

    ref.W_q4k->data = W_q4k.data();
    ref.x->data     = x_data.data();
    ref.y->data     = y_ref.data();

    hpx.W_q4k->data = W_q4k.data();
    hpx.x->data     = x_data.data();
    hpx.y->data     = y_hpx.data();

    ggml_backend_t cpu_be = ggml_backend_cpu_init();
    ASSERT_NE(cpu_be, nullptr);

    ASSERT_EQ(ggml_backend_graph_compute(cpu_be, ref.gf), GGML_STATUS_SUCCESS);

    ggml_hpx_tpool_start();

    ggml_hpx_selective_stats stats{};
    ASSERT_TRUE(ggml_hpx_exec_graph_selective_mul_mat(
        hpx.gf,
        cpu_be,
        /*n_lanes=*/1,
        &stats,
        /*packet_env=*/nullptr));

    EXPECT_EQ(stats.lowered_nodes,  1u);
    EXPECT_EQ(stats.fallback_nodes, 0u);

    constexpr float kTol = 1e-4f;
    for (int64_t j = 0; j < out_cols * rows; ++j)
        EXPECT_NEAR(y_hpx[static_cast<size_t>(j)],
                    y_ref[static_cast<size_t>(j)],
                    kTol) << "j=" << j;

    ggml_backend_free(cpu_be);
}

// Full-rank square: cols=2048, out_cols=2048, rows=1.
TEST(SelectiveMulMatQ4K, SquareShape_2048x2048)
{
    constexpr int64_t cols     = 2048;
    constexpr int64_t out_cols = 2048;
    constexpr int64_t rows     = 1;

    Q4KMulMatGraph ref(cols, out_cols, rows);
    Q4KMulMatGraph hpx(cols, out_cols, rows);

    const size_t w_row_bytes = ggml_row_size(GGML_TYPE_Q4_K, cols);
    const size_t w_bytes     = w_row_bytes * static_cast<size_t>(out_cols);

    std::vector<float>   W_f32(static_cast<size_t>(cols * out_cols));
    std::vector<uint8_t> W_q4k(w_bytes);
    std::vector<float>   x_data(static_cast<size_t>(cols * rows));
    std::vector<float>   y_ref(static_cast<size_t>(out_cols * rows), 0.0f);
    std::vector<float>   y_hpx(static_cast<size_t>(out_cols * rows), 0.0f);

    for (int64_t j = 0; j < out_cols; ++j)
        for (int64_t k = 0; k < cols; ++k)
            W_f32[static_cast<size_t>(j * cols + k)] =
                0.01f * static_cast<float>((j + 1) * ((k % 17) - 8));

    for (int64_t k = 0; k < cols; ++k)
        x_data[static_cast<size_t>(k)] =
            0.02f * static_cast<float>((k % 13) - 6);

    for (int64_t j = 0; j < out_cols; ++j)
        quantize_row_q4_K(
            W_f32.data()  + static_cast<size_t>(j * cols),
            W_q4k.data()  + static_cast<size_t>(j) * w_row_bytes,
            cols);

    ref.W_q4k->data = W_q4k.data();
    ref.x->data     = x_data.data();
    ref.y->data     = y_ref.data();

    hpx.W_q4k->data = W_q4k.data();
    hpx.x->data     = x_data.data();
    hpx.y->data     = y_hpx.data();

    ggml_backend_t cpu_be = ggml_backend_cpu_init();
    ASSERT_NE(cpu_be, nullptr);

    ASSERT_EQ(ggml_backend_graph_compute(cpu_be, ref.gf), GGML_STATUS_SUCCESS);

    ggml_hpx_tpool_start();

    ggml_hpx_selective_stats stats{};
    ASSERT_TRUE(ggml_hpx_exec_graph_selective_mul_mat(
        hpx.gf,
        cpu_be,
        /*n_lanes=*/1,
        &stats,
        /*packet_env=*/nullptr));

    EXPECT_EQ(stats.lowered_nodes,  1u);
    EXPECT_EQ(stats.fallback_nodes, 0u);

    constexpr float kTol = 1e-4f;
    for (int64_t j = 0; j < out_cols * rows; ++j)
        EXPECT_NEAR(y_hpx[static_cast<size_t>(j)],
                    y_ref[static_cast<size_t>(j)],
                    kTol) << "j=" << j;

    ggml_backend_free(cpu_be);
}

// Non-contiguous (strided) output — simulates a KV-cache view.
//
// y is shaped [out_cols=256, rows=1] but has nb[1] = 2*out_cols*sizeof(float),
// as if it were a slice of a larger tensor row.  For rows=1 nb[1] is irrelevant
// for addressing (only row 0 is written), but the is_contiguous_2d check in
// lower_op previously rejected such tensors, routing them to the CPU fallback
// instead of the HPX lowered path.
TEST(SelectiveMulMatQ4K, StridedOutputView_2048x256)
{
    constexpr int64_t cols     = 2048;
    constexpr int64_t out_cols = 256;
    constexpr int64_t rows     = 1;
    constexpr size_t  y_nb1    = 2 * static_cast<size_t>(out_cols) * sizeof(float);

    Q4KMulMatGraph ref(cols, out_cols, rows);
    Q4KMulMatGraph hpx_g(cols, out_cols, rows);

    // Override nb[1..3] on both output tensors to simulate a KV-cache view.
    // nb[2] and nb[3] must be >= nb[1] to satisfy ggml's internal invariant
    // (GGML_ASSERT(nb1 <= nb2) in ggml_compute_forward_mul_mat).
    // For rows=1, ne[1]=ne[2]=ne[3]=1, so nb[2]=nb[3]=nb[1] is correct.
    ref.y->nb[1]   = ref.y->nb[2]   = ref.y->nb[3]   = y_nb1;
    hpx_g.y->nb[1] = hpx_g.y->nb[2] = hpx_g.y->nb[3] = y_nb1;

    const size_t w_row_bytes = ggml_row_size(GGML_TYPE_Q4_K, cols);
    const size_t w_bytes     = w_row_bytes * static_cast<size_t>(out_cols);

    std::vector<float>   W_f32(static_cast<size_t>(cols * out_cols));
    std::vector<uint8_t> W_q4k(w_bytes);
    std::vector<float>   x_data(static_cast<size_t>(cols * rows));
    // Padded output buffers: 2×out_cols floats; only [0..out_cols) are written.
    std::vector<float>   y_ref_buf(2 * static_cast<size_t>(out_cols), 0.0f);
    std::vector<float>   y_hpx_buf(2 * static_cast<size_t>(out_cols), 0.0f);

    for (int64_t j = 0; j < out_cols; ++j)
        for (int64_t k = 0; k < cols; ++k)
            W_f32[static_cast<size_t>(j * cols + k)] =
                0.01f * static_cast<float>((j + 1) * ((k % 17) - 8));

    for (int64_t k = 0; k < cols; ++k)
        x_data[static_cast<size_t>(k)] =
            0.02f * static_cast<float>((k % 13) - 6);

    for (int64_t j = 0; j < out_cols; ++j)
        quantize_row_q4_K(
            W_f32.data()  + static_cast<size_t>(j * cols),
            W_q4k.data()  + static_cast<size_t>(j) * w_row_bytes,
            cols);

    ref.W_q4k->data = W_q4k.data();
    ref.x->data     = x_data.data();
    ref.y->data     = y_ref_buf.data();

    hpx_g.W_q4k->data = W_q4k.data();
    hpx_g.x->data     = x_data.data();
    hpx_g.y->data     = y_hpx_buf.data();

    ggml_backend_t cpu_be = ggml_backend_cpu_init();
    ASSERT_NE(cpu_be, nullptr);

    ASSERT_EQ(ggml_backend_graph_compute(cpu_be, ref.gf), GGML_STATUS_SUCCESS);

    ggml_hpx_tpool_start();

    ggml_hpx_selective_stats stats{};
    ASSERT_TRUE(ggml_hpx_exec_graph_selective_mul_mat(
        hpx_g.gf,
        cpu_be,
        /*n_lanes=*/1,
        &stats,
        /*packet_env=*/nullptr));

    EXPECT_EQ(stats.lowered_nodes,  1u);
    EXPECT_EQ(stats.fallback_nodes, 0u);

    constexpr float kTol = 1e-4f;
    for (int64_t j = 0; j < out_cols; ++j)
        EXPECT_NEAR(y_hpx_buf[static_cast<size_t>(j)],
                    y_ref_buf[static_cast<size_t>(j)],
                    kTol) << "j=" << j;

    ggml_backend_free(cpu_be);
}

// Gate/up projection shape (TinyLlama MLP): cols=2048, out_cols=5632, rows=1.
TEST(SelectiveMulMatQ4K, MLPProjectionShape_2048x5632)
{
    constexpr int64_t cols     = 2048;
    constexpr int64_t out_cols = 5632;
    constexpr int64_t rows     = 1;

    Q4KMulMatGraph ref(cols, out_cols, rows);
    Q4KMulMatGraph hpx(cols, out_cols, rows);

    const size_t w_row_bytes = ggml_row_size(GGML_TYPE_Q4_K, cols);
    const size_t w_bytes     = w_row_bytes * static_cast<size_t>(out_cols);

    std::vector<float>   W_f32(static_cast<size_t>(cols * out_cols));
    std::vector<uint8_t> W_q4k(w_bytes);
    std::vector<float>   x_data(static_cast<size_t>(cols * rows));
    std::vector<float>   y_ref(static_cast<size_t>(out_cols * rows), 0.0f);
    std::vector<float>   y_hpx(static_cast<size_t>(out_cols * rows), 0.0f);

    for (int64_t j = 0; j < out_cols; ++j)
        for (int64_t k = 0; k < cols; ++k)
            W_f32[static_cast<size_t>(j * cols + k)] =
                0.01f * static_cast<float>((j + 1) * ((k % 17) - 8));

    for (int64_t k = 0; k < cols; ++k)
        x_data[static_cast<size_t>(k)] =
            0.02f * static_cast<float>((k % 13) - 6);

    for (int64_t j = 0; j < out_cols; ++j)
        quantize_row_q4_K(
            W_f32.data()  + static_cast<size_t>(j * cols),
            W_q4k.data()  + static_cast<size_t>(j) * w_row_bytes,
            cols);

    ref.W_q4k->data = W_q4k.data();
    ref.x->data     = x_data.data();
    ref.y->data     = y_ref.data();

    hpx.W_q4k->data = W_q4k.data();
    hpx.x->data     = x_data.data();
    hpx.y->data     = y_hpx.data();

    ggml_backend_t cpu_be = ggml_backend_cpu_init();
    ASSERT_NE(cpu_be, nullptr);

    ASSERT_EQ(ggml_backend_graph_compute(cpu_be, ref.gf), GGML_STATUS_SUCCESS);

    ggml_hpx_tpool_start();

    ggml_hpx_selective_stats stats{};
    ASSERT_TRUE(ggml_hpx_exec_graph_selective_mul_mat(
        hpx.gf,
        cpu_be,
        /*n_lanes=*/1,
        &stats,
        /*packet_env=*/nullptr));

    EXPECT_EQ(stats.lowered_nodes,  1u);
    EXPECT_EQ(stats.fallback_nodes, 0u);

    constexpr float kTol = 1e-4f;
    for (int64_t j = 0; j < out_cols * rows; ++j)
        EXPECT_NEAR(y_hpx[static_cast<size_t>(j)],
                    y_ref[static_cast<size_t>(j)],
                    kTol) << "j=" << j;

    ggml_backend_free(cpu_be);
}

#else    // GGML_HPX_REGION_DAG

TEST(SelectiveMulMatQ4K, RegionDagDisabled)
{
    GTEST_SKIP() << "GGML_HPX_REGION_DAG not defined";
}

#endif    // GGML_HPX_REGION_DAG
