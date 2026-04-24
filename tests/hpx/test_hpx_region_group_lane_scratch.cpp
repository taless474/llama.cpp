// test_hpx_region_group_lane_scratch.cpp
//
// Phase 3 resource-model proof: per-lane scratch plumbing.
//
// One ELEMENTWISE region with span=8, n_lanes=4.
// Each lane writes into its own lane_scratch[ith] slot.
//
// Proves:
//   - resources->lane_scratch[ith] is correctly threaded to each lane callback
//   - each lane is invoked exactly once
//   - ith correctly identifies the lane (slot->lane == ith)
//   - nth == n_lanes (fan-out count matches lane count for this shape)
//   - output is copied correctly across all lanes
//   - per-lane partial sums confirm each lane operated on its own chunk

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

struct LaneSlot
{
    std::atomic<int> hits{0};
    int              lane{-1};
    int              nth{-1};
    float            partial_sum{-1.f};
};

struct LaneScratchCtx
{
    const float * x;
    float *       out;
};

void lane_scratch_copy_run_range(
    void *                      ctx_void,
    int                         ith,
    int                         nth,
    int64_t                     begin,
    int64_t                     end,
    ggml_hpx_region_resources * resources)
{
    auto * ctx = static_cast<LaneScratchCtx *>(ctx_void);

    ASSERT_NE(resources, nullptr);
    ASSERT_NE(resources->lane_scratch, nullptr);
    ASSERT_GE(ith, 0);
    ASSERT_LT(ith, resources->n_lanes);

    auto * slot = static_cast<LaneSlot *>(resources->lane_scratch[ith]);
    ASSERT_NE(slot, nullptr);

    slot->lane = ith;
    slot->nth  = nth;

    float acc = 0.0f;
    for (int64_t i = begin; i < end; ++i)
    {
        ctx->out[i] = ctx->x[i];
        acc         += ctx->x[i];
    }
    slot->partial_sum = acc;
    slot->hits.fetch_add(1, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------

TEST(HPXRegionGroupLaneScratch, EachLaneWritesItsOwnSlot)
{
    constexpr int kNLanes = 4;

    const std::vector<float> x   = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
    std::vector<float>       out(x.size(), -1.f);

    LaneSlot slots[kNLanes];

    void * lane_scratch[kNLanes] = {&slots[0], &slots[1], &slots[2], &slots[3]};

    LaneScratchCtx ctx{x.data(), out.data()};

    ggml_hpx_cpu_region regions[1] = {
        {GGML_HPX_CPU_REGION_KIND_ELEMENTWISE, 0, static_cast<int64_t>(x.size()), 0, &ctx,
         lane_scratch_copy_run_range},
    };

    ggml_hpx_cpu_region_group group{regions, 1, nullptr, 0};

    ggml_hpx_region_resources resources{};
    resources.shared_scratch   = nullptr;
    resources.lane_scratch     = lane_scratch;
    resources.reduction_buffer = nullptr;
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

    // Output copied correctly across all lanes.
    for (int i = 0; i < static_cast<int>(x.size()); ++i)
    {
        EXPECT_FLOAT_EQ(out[i], x[i]);
    }

    // Per-lane scratch plumbing: each lane invoked exactly once, ith correct,
    // and each lane accumulated a positive partial sum (all x[i] > 0).
    for (int i = 0; i < kNLanes; ++i)
    {
        EXPECT_EQ(slots[i].hits.load(std::memory_order_relaxed), 1) << "lane " << i;
        EXPECT_EQ(slots[i].lane, i)                                  << "lane " << i;
        EXPECT_EQ(slots[i].nth, kNLanes)                             << "lane " << i;
        EXPECT_GT(slots[i].partial_sum, 0.f)                         << "lane " << i;
    }

    // Total sum across all lanes equals the full-array sum (no overlap, no gap).
    // Does not assume any specific chunk boundary; invariant under splitter changes.
    float total = 0.f;
    for (int i = 0; i < kNLanes; ++i)
    {
        total += slots[i].partial_sum;
    }
    EXPECT_FLOAT_EQ(total, 36.f);
}

}    // namespace

#endif    // GGML_HPX_REGION_DAG
