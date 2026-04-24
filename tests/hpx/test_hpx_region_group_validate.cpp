// test_hpx_region_group_validate.cpp
//
// Unit tests for ggml_hpx_validate_region_group.
// Pure C-compatible path — no HPX thread context required.

#include <gtest/gtest.h>

#include "ggml-hpx-region-dag.h"
#include "ggml-hpx-region-exec.h"

#ifdef GGML_HPX_REGION_DAG

namespace
{

struct DummyCtx
{
    float * x;
};

static void dummy_run_range(
    void *                      ctx_void,
    int                         /*ith*/,
    int                         /*nth*/,
    int64_t                     begin,
    int64_t                     end,
    ggml_hpx_region_resources * /*resources*/)
{
    auto * ctx = static_cast<DummyCtx *>(ctx_void);
    for (int64_t i = begin; i < end; ++i)
    {
        ctx->x[i] += 1.0f;
    }
}

TEST(HPXRegionGroupValidate, BrokenEdges)
{
    constexpr int64_t N = 8;
    float             x[N] = {};
    DummyCtx          ctx{x};

    ggml_hpx_cpu_region regions[1] = {
        {GGML_HPX_CPU_REGION_KIND_ELEMENTWISE, 0, N, 0, &ctx,
         dummy_run_range}};

    ggml_hpx_dep_edge deps[1] = {{0, 2}};    // dst=2 out of range for 1 region

    ggml_hpx_cpu_region_group group{regions, 1, deps, 1};

    const char * err = ggml_hpx_validate_region_group(&group, true);
    ASSERT_NE(err, nullptr);
}

TEST(HPXRegionGroupValidate, SelfLoop)
{
    constexpr int64_t N = 8;
    float             x[N] = {};
    DummyCtx          ctx{x};

    ggml_hpx_cpu_region regions[1] = {
        {GGML_HPX_CPU_REGION_KIND_ELEMENTWISE, 0, N, 0, &ctx,
         dummy_run_range}};

    ggml_hpx_dep_edge deps[1] = {{0, 0}};

    ggml_hpx_cpu_region_group group{regions, 1, deps, 1};

    const char * err = ggml_hpx_validate_region_group(&group, true);
    ASSERT_NE(err, nullptr);
}

TEST(HPXRegionGroupValidate, NullCallback)
{
    constexpr int64_t N = 8;
    float             x[N] = {};
    DummyCtx          ctx{x};

    ggml_hpx_cpu_region regions[1] = {
        {GGML_HPX_CPU_REGION_KIND_ELEMENTWISE, 0, N, 0, &ctx, nullptr}};

    ggml_hpx_cpu_region_group group{regions, 1, nullptr, 0};

    const char * err = ggml_hpx_validate_region_group(&group, true);
    ASSERT_NE(err, nullptr);
}

}    // namespace

#endif    // GGML_HPX_REGION_DAG
