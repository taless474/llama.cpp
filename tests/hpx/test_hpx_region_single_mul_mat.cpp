// test_hpx_region_single_mul_mat.cpp
//
// Correctness test for ggml_hpx_run_single_region with the F32 mul_mat kernel.
//
// Weight matrix is an identity-like selection:
//   w[0] = [1 0 0 0]    selects x column 0
//   w[1] = [0 1 0 0]    selects x column 1
//   w[2] = [0 0 1 0]    selects x column 2
//
// So output[row][col] = x[row][col] for col in {0,1,2}.
//
// The work range over output COLUMNS is [0, out_cols), NOT [0, rows).
//
// HPX note: ggml_hpx_run_single_region constructs a fork_join_executor
// internally and must run on an HPX thread.  The test body is wrapped in
// hpx::async().get() after ggml_hpx_tpool_start().

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <cstring>

#include "ggml-hpx-region-dag.h"
#include "ggml-hpx-region-exec.h"
#include "ggml-hpx-runtime.h"    // ggml_hpx_tpool_start

#include <hpx/future.hpp>

#ifdef GGML_HPX_REGION_DAG

#ifdef GGML_HPX_REGION_DAG_TESTING
extern std::atomic<int> g_ggml_graph_compute_thread_run_calls;
#endif

namespace
{

TEST(HPXRegionSingleMulMat, ProducesExpectedOutput)
{
    constexpr int64_t rows     = 2;
    constexpr int64_t cols     = 4;
    constexpr int64_t out_cols = 3;
    constexpr int     kNLanes  = 2;

    const float x[rows * cols] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
    };

    // w[out_col][col] — each row selects one input column
    const float w[out_cols * cols] = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
    };

    float y[rows * out_cols];
    std::memset(y, 0, sizeof(y));

    const float expected[rows * out_cols] = {
        1, 2, 3,
        5, 6, 7,
    };

    ggml_hpx_mul_mat_f32_ctx ctx{};
    ctx.x        = x;
    ctx.w        = w;
    ctx.y        = y;
    ctx.rows     = rows;
    ctx.cols     = cols;
    ctx.out_cols = out_cols;

    // Work range is output columns [0, out_cols), NOT [0, rows).
    ggml_hpx_cpu_region region{
        GGML_HPX_CPU_REGION_KIND_MATMUL,
        0,
        out_cols,
        0,
        &ctx,
        ggml_hpx_mul_mat_f32_run_range,
    };

    void * lane_scratch[kNLanes] = {nullptr, nullptr};

    ggml_hpx_region_resources resources{};
    resources.shared_scratch   = nullptr;
    resources.lane_scratch     = lane_scratch;
    resources.reduction_buffer = nullptr;
    resources.n_lanes          = kNLanes;

#ifdef GGML_HPX_REGION_DAG_TESTING
    g_ggml_graph_compute_thread_run_calls.store(0);
#endif

    // fork_join_executor ctor requires an HPX thread context.
    ggml_hpx_tpool_start();
    hpx::async([&]() {
        ggml_hpx_run_single_region(&region, &resources);
    }).get();

    for (int64_t i = 0; i < rows * out_cols; ++i)
    {
        EXPECT_FLOAT_EQ(y[i], expected[i]);
    }

#ifdef GGML_HPX_REGION_DAG_TESTING
    EXPECT_EQ(g_ggml_graph_compute_thread_run_calls.load(), 0);
#endif
}

}    // namespace

#endif    // GGML_HPX_REGION_DAG
