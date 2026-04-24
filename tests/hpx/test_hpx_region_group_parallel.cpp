// test_hpx_region_group_parallel.cpp
//
// Proves that same-level regions in ggml_hpx_run_region_group execute
// concurrently by observing mutual overlap.
//
// DAG:  R0 ─┐
//            └─► R2
//       R1 ─┘
//
// R0 and R1 are at level 0.  Each probe region:
//   1. Sets self_running = true.
//   2. Polls other_running for 20 × 1 ms; if it sees true, sets overlap_seen.
//   3. Sets self_running = false.
//
// With serial execution R0 finishes (self_running goes false) before R1 starts,
// so neither region ever observes the other alive: overlap_seen stays false and
// the test FAILS.
//
// With concurrent execution both regions are live simultaneously; at least one
// sees the other's flag and sets overlap_seen: the test PASSES.
//
// Expected result against the current serial run_region_group: FAIL.
// Expected result after same-level concurrency is implemented: PASS.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>

#include <hpx/include/async.hpp>

#include "ggml-hpx-region-dag.h"
#include "ggml-hpx-region-exec.h"
#include "ggml-hpx-runtime.h"    // ggml_hpx_tpool_start

#ifdef GGML_HPX_REGION_DAG

namespace
{

class HPXRegionGroupParallelTest : public ::testing::Test
{
protected:
    static void SetUpTestSuite()
    {
        ggml_hpx_tpool_start();
    }
};

struct OverlapCtx
{
    std::atomic<bool> * self_running;
    std::atomic<bool> * other_running;
    std::atomic<bool> * overlap_seen;
    float *             out;
    float               value;
};

struct JoinCtx
{
    float * out0;
    float * out1;
    float * out2;
};

static void overlap_region_run_range(
    void *                      ctx_void,
    int                         /*ith*/,
    int                         /*nth*/,
    int64_t                     begin,
    int64_t                     end,
    ggml_hpx_region_resources * /*resources*/)
{
    auto * ctx = static_cast<OverlapCtx *>(ctx_void);

    ctx->self_running->store(true, std::memory_order_release);

    // Stay alive long enough for concurrent overlap to be observable.
    for (int k = 0; k < 20; ++k)
    {
        if (ctx->other_running->load(std::memory_order_acquire))
        {
            ctx->overlap_seen->store(true, std::memory_order_release);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    for (int64_t i = begin; i < end; ++i)
    {
        ctx->out[i] = ctx->value;
    }

    ctx->self_running->store(false, std::memory_order_release);
}

static void join_run_range(
    void *                      ctx_void,
    int                         /*ith*/,
    int                         /*nth*/,
    int64_t                     begin,
    int64_t                     end,
    ggml_hpx_region_resources * /*resources*/)
{
    auto * ctx = static_cast<JoinCtx *>(ctx_void);
    for (int64_t i = begin; i < end; ++i)
    {
        ctx->out2[i] = ctx->out0[i] + ctx->out1[i];
    }
}

// ---------------------------------------------------------------------------

TEST_F(HPXRegionGroupParallelTest, SameLevelRegionsActuallyOverlap)
{
    constexpr int64_t N       = 16;
    constexpr int     kNLanes = 4;

    float out0[N] = {};
    float out1[N] = {};
    float out2[N] = {};

    std::atomic<bool> r0_running{false};
    std::atomic<bool> r1_running{false};
    std::atomic<bool> overlap_seen{false};

    OverlapCtx c0{
        &r0_running,
        &r1_running,
        &overlap_seen,
        out0,
        10.0f,
    };

    OverlapCtx c1{
        &r1_running,
        &r0_running,
        &overlap_seen,
        out1,
        20.0f,
    };

    JoinCtx c2{out0, out1, out2};

    ggml_hpx_cpu_region regions[3] = {
        {GGML_HPX_CPU_REGION_KIND_ELEMENTWISE, 0, N, 0, &c0,
         overlap_region_run_range},
        {GGML_HPX_CPU_REGION_KIND_ELEMENTWISE, 0, N, 0, &c1,
         overlap_region_run_range},
        {GGML_HPX_CPU_REGION_KIND_ELEMENTWISE, 0, N, 0, &c2, join_run_range},
    };

    ggml_hpx_dep_edge deps[2] = {{0, 2}, {1, 2}};

    ggml_hpx_cpu_region_group group{regions, 3, deps, 2};

    void * lane_scratch[kNLanes] = {nullptr, nullptr, nullptr, nullptr};

    ggml_hpx_region_resources resources{};
    resources.shared_scratch   = nullptr;
    resources.lane_scratch     = lane_scratch;
    resources.reduction_buffer = nullptr;
    resources.n_lanes          = kNLanes;

    const char * err = ggml_hpx_validate_region_group(&group, true);
    ASSERT_EQ(err, nullptr);

    hpx::async([&]() {
        ggml_hpx_run_region_group(&group, &resources);
    }).get();

    EXPECT_TRUE(overlap_seen.load(std::memory_order_acquire));

    for (int64_t i = 0; i < N; ++i)
    {
        EXPECT_FLOAT_EQ(out0[i], 10.0f);
        EXPECT_FLOAT_EQ(out1[i], 20.0f);
        EXPECT_FLOAT_EQ(out2[i], 30.0f);
    }
}

}    // namespace

#endif    // GGML_HPX_REGION_DAG
