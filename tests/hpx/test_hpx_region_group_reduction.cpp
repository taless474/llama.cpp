// test_hpx_region_group_reduction.cpp
//
// Phase 3 resource-model proof: reduction_buffer visibility across dependent regions.
//
// DAG:
//   R0 (REDUCTION)   — sums x[0..n) into reduction_buffer
//   R1 (ELEMENTWISE) — reads reduction_buffer, writes out[i] = x[i] + sum
//   dep: R0 -> R1
//
// Proves:
//   - reduction_buffer carries the value produced by R0 to R1
//   - dependency ordering guarantees R0 completes before R1 reads the buffer
//   - REDUCTION runs single-threaded (ith == 0, nth == 1)
//   - ELEMENTWISE fans out to all n_lanes lanes (consumer_calls == n_lanes)

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <vector>

#include <hpx/future.hpp>

#include "ggml-hpx-region-dag.h"
#include "ggml-hpx-region-exec.h"
#include "ggml-hpx-runtime.h"

#ifdef GGML_HPX_REGION_DAG

namespace
{

struct ReductionCtx
{
    const float *       x;
    int64_t             n;
    std::atomic<int> *  phase;
    std::atomic<int> *  calls;
};

struct AddCtx
{
    const float *       x;
    float *             out;
    int64_t             n;
    std::atomic<int> *  phase;
    std::atomic<int> *  calls;
};

// R0: sum x[begin..end) into reduction_buffer.
// Contract: REDUCTION runs single-threaded (ith == 0, nth == 1, full range).
void sum_to_reduction_buffer(
    void *                      ctx_void,
    int                         ith,
    int                         nth,
    int64_t                     begin,
    int64_t                     end,
    ggml_hpx_region_resources * resources)
{
    auto * ctx = static_cast<ReductionCtx *>(ctx_void);

    // Single-threaded REDUCTION contract.
    ASSERT_EQ(ith, 0);
    ASSERT_EQ(nth, 1);
    ASSERT_EQ(begin, 0);
    ASSERT_EQ(end, ctx->n);
    ASSERT_NE(resources, nullptr);
    ASSERT_NE(resources->reduction_buffer, nullptr);

    float acc = 0.0f;
    for (int64_t i = begin; i < end; ++i)
    {
        acc += ctx->x[i];
    }

    *static_cast<float *>(resources->reduction_buffer) = acc;
    ctx->calls->fetch_add(1, std::memory_order_relaxed);
    ctx->phase->store(1, std::memory_order_release);
}

// R1: out[i] = x[i] + sum, where sum was written to reduction_buffer by R0.
// The dependency edge guarantees phase == 1 (R0 complete) before R1 runs.
void add_sum_to_vector(
    void *                      ctx_void,
    int                         /*ith*/,
    int                         /*nth*/,
    int64_t                     begin,
    int64_t                     end,
    ggml_hpx_region_resources * resources)
{
    auto * ctx = static_cast<AddCtx *>(ctx_void);

    ASSERT_NE(resources, nullptr);
    ASSERT_NE(resources->reduction_buffer, nullptr);
    // Dependency must guarantee R0 already completed.
    ASSERT_EQ(ctx->phase->load(std::memory_order_acquire), 1);
    // Sanity-check the work range the executor handed us.
    ASSERT_GE(begin, 0);
    ASSERT_LE(begin, end);
    ASSERT_LE(end, ctx->n);

    const float sum = *static_cast<float *>(resources->reduction_buffer);
    for (int64_t i = begin; i < end; ++i)
    {
        ctx->out[i] = ctx->x[i] + sum;
    }

    ctx->calls->fetch_add(1, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------

TEST(HPXRegionGroupReduction, ReductionBufferVisibleToDependentRegion)
{
    constexpr int kNLanes = 4;

    const std::vector<float> x   = {1.f, 2.f, 3.f, 4.f, 5.f};
    std::vector<float>       out(x.size(), -1.f);

    float reduction_value = -123.f;

    std::atomic<int> phase{0};
    std::atomic<int> reduction_calls{0};
    std::atomic<int> consumer_calls{0};

    ReductionCtx r0_ctx{x.data(), static_cast<int64_t>(x.size()), &phase, &reduction_calls};
    AddCtx       r1_ctx{x.data(), out.data(), static_cast<int64_t>(x.size()), &phase, &consumer_calls};

    ggml_hpx_cpu_region regions[2] = {
        {GGML_HPX_CPU_REGION_KIND_REDUCTION,   0, static_cast<int64_t>(x.size()), 0, &r0_ctx, sum_to_reduction_buffer},
        {GGML_HPX_CPU_REGION_KIND_ELEMENTWISE, 0, static_cast<int64_t>(x.size()), 0, &r1_ctx, add_sum_to_vector},
    };

    ggml_hpx_dep_edge deps[1] = {{0, 1}};

    ggml_hpx_cpu_region_group group{regions, 2, deps, 1};

    void * lane_scratch[kNLanes] = {nullptr, nullptr, nullptr, nullptr};

    ggml_hpx_region_resources resources{};
    resources.shared_scratch   = nullptr;
    resources.lane_scratch     = lane_scratch;
    resources.reduction_buffer = &reduction_value;
    resources.n_lanes          = kNLanes;

    const char * err = ggml_hpx_validate_region_group(&group, true);
    ASSERT_EQ(err, nullptr);

    ggml_hpx_tpool_start();
    hpx::async([&]() {
        ggml_hpx_run_region_group(&group, &resources);
    }).get();

    if (::testing::Test::HasFatalFailure())
    {
        return;
    }

    // R0 ran exactly once (single-threaded REDUCTION).
    EXPECT_EQ(reduction_calls.load(std::memory_order_relaxed), 1);

    // R1 fanned out across all lanes (span=5 > n_lanes=4, no empty chunks).
    EXPECT_EQ(consumer_calls.load(std::memory_order_relaxed), kNLanes);

    // R0 wrote the correct sum into reduction_buffer.
    EXPECT_NEAR(reduction_value, 15.f, 1e-6f);

    // R1 read reduction_buffer and produced the correct output.
    const float expected[] = {16.f, 17.f, 18.f, 19.f, 20.f};
    for (int i = 0; i < static_cast<int>(x.size()); ++i)
    {
        EXPECT_NEAR(out[i], expected[i], 1e-6f);
    }
}

}    // namespace

#endif    // GGML_HPX_REGION_DAG
