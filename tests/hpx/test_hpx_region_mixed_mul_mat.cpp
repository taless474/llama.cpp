// test_hpx_region_mixed_mul_mat.cpp
//
// Tests the mixed fine-region / CPU-fallback execution path.
//
// Graph topology:
//   w, x (leaf tensors)
//     -> mm  = ggml_mul_mat(w, x)   [supported: lowered to fine-region path]
//     -> neg = ggml_neg(mm)          [unsupported: falls back to CPU backend]
//
// After execution neg->data must equal -mm element-wise.
// mm is driven by identity-like weights so expected values are known exactly.
//
// Instrumentation counters (GGML_HPX_EXEC_SELECTIVE_TESTING) assert that the
// fine-region path actually ran for mm and was not silently bypassed.
// The flag is injected via target_compile_definitions in CMakeLists.txt so
// both the test TU and the re-compiled selective impl TU see it.

#include <gtest/gtest.h>

#include <cstring>

#include "ggml-hpx-exec-selective.h"
#include "ggml-hpx-runtime.h"
#include "ggml.h"

#ifdef GGML_HPX_REGION_DAG

namespace
{

class HPXRegionMixedMulMat : public ::testing::Test
{
protected:
    static void SetUpTestSuite()
    {
        ggml_hpx_tpool_start();
    }
};

TEST_F(HPXRegionMixedMulMat, MulMatLoweredNegFallback)
{
    constexpr int64_t rows     = 2;
    constexpr int64_t cols     = 4;
    constexpr int64_t out_cols = 3;

    ggml_init_params params{};
    params.mem_size   = 16 * 1024 * 1024;
    params.mem_buffer = nullptr;
    params.no_alloc   = false;

    ggml_context * ctx = ggml_init(params);
    ASSERT_NE(ctx, nullptr);

    // ggml MUL_MAT: src[0]=w [cols × out_cols], src[1]=x [cols × rows]
    // ne[0]=cols, ne[1]=out_cols for w; ne[0]=cols, ne[1]=rows for x.
    ggml_tensor * w   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, out_cols);
    ggml_tensor * x   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, rows);
    ggml_tensor * mm  = ggml_mul_mat(ctx, w, x);
    ggml_tensor * neg = ggml_neg(ctx, mm);
    ASSERT_NE(w,   nullptr);
    ASSERT_NE(x,   nullptr);
    ASSERT_NE(mm,  nullptr);
    ASSERT_NE(neg, nullptr);

    // Identity-like weights: w[out_col,:] picks input column out_col.
    // w is stored row-major: out_cols rows, each cols wide.
    const float w_vals[out_cols * cols] = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
    };
    const float x_vals[rows * cols] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
    };
    std::memcpy(w->data, w_vals, sizeof(w_vals));
    std::memcpy(x->data, x_vals, sizeof(x_vals));

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ASSERT_NE(gf, nullptr);
    ggml_build_forward_expand(gf, neg);

    g_hpx_selective_lowered_count.store(0);
    g_hpx_selective_executed_count.store(0);

    ASSERT_TRUE(ggml_hpx_exec_graph_selective_mul_mat_arena(gf));

    // mm[row][out_col] = dot(x[row,:], w[out_col,:])
    //   row 0: [1,2,3]   row 1: [5,6,7]
    // neg = -mm:
    //   row 0: [-1,-2,-3]  row 1: [-5,-6,-7]
    const float expected[rows * out_cols] = {
        -1.f, -2.f, -3.f,
        -5.f, -6.f, -7.f,
    };

    const float * out = static_cast<const float *>(neg->data);
    ASSERT_NE(out, nullptr);
    for (int64_t i = 0; i < rows * out_cols; ++i)
    {
        EXPECT_FLOAT_EQ(out[i], expected[i]) << "flat index=" << i;
    }

    // mm was the one supported node; neg fell back to CPU.
    EXPECT_EQ(g_hpx_selective_lowered_count.load(),  1);
    EXPECT_EQ(g_hpx_selective_executed_count.load(), 1);

    ggml_free(ctx);
}

} // namespace

#endif // GGML_HPX_REGION_DAG
