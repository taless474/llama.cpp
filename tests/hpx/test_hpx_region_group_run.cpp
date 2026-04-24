// test_hpx_region_group_run.cpp
//
// Correctness test for ggml_hpx_run_region_group.
// Runs a 3-node DAG:
//   region 0: y[i] = x[i] + 1
//   region 1: z[i] = x[i] * 2
//   region 2: out[i] = y[i] + z[i]   (depends on 0 and 1)
//
// Expected: out[i] = 3*x[i] + 1
//
// HPX note: ggml_hpx_run_region_group constructs a fork_join_executor
// internally and must run on an HPX thread.  The test body is wrapped in
// hpx::async().get() after ggml_hpx_tpool_start().

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>

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

struct AddOneCtx
{
    const float * x;
    float *       y;
};

struct TimesTwoCtx
{
    const float * x;
    float *       z;
};

struct CombineCtx
{
    const float * y;
    const float * z;
    float *       out;
};

static void add_one_run_range(
    void *                      ctx_void,
    int                         /*ith*/,
    int                         /*nth*/,
    int64_t                     begin,
    int64_t                     end,
    ggml_hpx_region_resources * /*resources*/)
{
    auto * ctx = static_cast<AddOneCtx *>(ctx_void);
    for (int64_t i = begin; i < end; ++i)
    {
        ctx->y[i] = ctx->x[i] + 1.0f;
    }
}

static void times_two_run_range(
    void *                      ctx_void,
    int                         /*ith*/,
    int                         /*nth*/,
    int64_t                     begin,
    int64_t                     end,
    ggml_hpx_region_resources * /*resources*/)
{
    auto * ctx = static_cast<TimesTwoCtx *>(ctx_void);
    for (int64_t i = begin; i < end; ++i)
    {
        ctx->z[i] = ctx->x[i] * 2.0f;
    }
}

static void combine_run_range(
    void *                      ctx_void,
    int                         /*ith*/,
    int                         /*nth*/,
    int64_t                     begin,
    int64_t                     end,
    ggml_hpx_region_resources * /*resources*/)
{
    auto * ctx = static_cast<CombineCtx *>(ctx_void);
    for (int64_t i = begin; i < end; ++i)
    {
        ctx->out[i] = ctx->y[i] + ctx->z[i];
    }
}

TEST(HPXRegionGroupRun, ThreeRegionDAGProducesExpectedOutput)
{
    constexpr int64_t N       = 16;
    constexpr int     kNLanes = 4;

    float x[N];
    float y[N];
    float z[N];
    float out[N];

    for (int64_t i = 0; i < N; ++i)
    {
        x[i]   = static_cast<float>(i);
        y[i]   = 0.0f;
        z[i]   = 0.0f;
        out[i] = 0.0f;
    }

    AddOneCtx  c0{x, y};
    TimesTwoCtx c1{x, z};
    CombineCtx  c2{y, z, out};

    ggml_hpx_cpu_region regions[3] = {
        {GGML_HPX_CPU_REGION_KIND_ELEMENTWISE, 0, N, 0, &c0,
         add_one_run_range},
        {GGML_HPX_CPU_REGION_KIND_ELEMENTWISE, 0, N, 0, &c1,
         times_two_run_range},
        {GGML_HPX_CPU_REGION_KIND_ELEMENTWISE, 0, N, 0, &c2,
         combine_run_range},
    };

    ggml_hpx_dep_edge deps[2] = {{0, 2}, {1, 2}};

    ggml_hpx_cpu_region_group group{regions, 3, deps, 2};

    void * lane_scratch[kNLanes] = {nullptr, nullptr, nullptr, nullptr};

    ggml_hpx_region_resources resources{};
    resources.shared_scratch   = nullptr;
    resources.lane_scratch     = lane_scratch;
    resources.reduction_buffer = nullptr;
    resources.n_lanes          = kNLanes;

#ifdef GGML_HPX_REGION_DAG_TESTING
    g_ggml_graph_compute_thread_run_calls.store(0);
#endif

    const char * err = ggml_hpx_validate_region_group(&group, true);
    ASSERT_EQ(err, nullptr);

    // fork_join_executor ctor requires an HPX thread context.
    ggml_hpx_tpool_start();
    hpx::async([&]() {
        ggml_hpx_run_region_group(&group, &resources);
    }).get();

    for (int64_t i = 0; i < N; ++i)
    {
        EXPECT_FLOAT_EQ(y[i],   x[i] + 1.0f);
        EXPECT_FLOAT_EQ(z[i],   x[i] * 2.0f);
        EXPECT_FLOAT_EQ(out[i], 3.0f * x[i] + 1.0f);
    }

#ifdef GGML_HPX_REGION_DAG_TESTING
    EXPECT_EQ(g_ggml_graph_compute_thread_run_calls.load(), 0);
#endif
}

}    // namespace

#endif    // GGML_HPX_REGION_DAG
