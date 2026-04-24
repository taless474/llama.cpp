// test_hpx_region_group_rms_norm_f32.cpp
//
// Phase 4: first real ggml-style CPU op on the fine-region DAG.
//
// One F32 RMS_NORM row lowered into a 3-region DAG:
//
//   R0 (ELEMENTWISE): lane-parallel partial sumsq into lane_scratch[ith]
//   R1 (REDUCTION)  : single-thread finalize, writes sumsq + scale
//                     into reduction_buffer
//   R2 (ELEMENTWISE): lane-parallel apply, dst[i] = x[i] * scale
//
// Proves:
//   - lane_scratch is used for real per-lane computation
//   - reduction_buffer carries finalized state between dependent regions
//   - dependency ordering R0 -> R1 -> R2 is sufficient for correctness
//   - the fine-region DAG can execute one real ggml-style CPU op correctly

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include <hpx/future.hpp>

#include "ggml-hpx-region-dag.h"
#include "ggml-hpx-region-exec.h"
#include "ggml-hpx-runtime.h"

#ifdef GGML_HPX_REGION_DAG

namespace
{

static void rms_norm_ref(
    const float * x,
    float *       dst,
    int64_t       n,
    float         eps,
    float *       out_sumsq,
    float *       out_scale)
{
    float sumsq = 0.0f;
    for (int64_t i = 0; i < n; ++i)
    {
        sumsq += x[i] * x[i];
    }

    const float scale = 1.0f / std::sqrt(sumsq / static_cast<float>(n) + eps);

    for (int64_t i = 0; i < n; ++i)
    {
        dst[i] = x[i] * scale;
    }

    *out_sumsq = sumsq;
    *out_scale = scale;
}

TEST(HPXRegionGroupRMSNormF32, ComputesCorrectOutput)
{
    constexpr int   kNLanes = 4;
    constexpr float kEps    = 1e-5f;

    const std::vector<float> x = {
        1.f, -2.f, 3.f, -4.f, 5.f, -6.f, 7.f, -8.f,
    };
    ASSERT_EQ(static_cast<int>(x.size()), 8);

    std::vector<float> dst(x.size(), 0.0f);
    std::vector<float> ref(x.size(), 0.0f);

    float ref_sumsq = 0.0f;
    float ref_scale = 0.0f;
    rms_norm_ref(
        x.data(),
        ref.data(),
        static_cast<int64_t>(x.size()),
        kEps,
        &ref_sumsq,
        &ref_scale);

    float partials[kNLanes] = {0.f, 0.f, 0.f, 0.f};
    void * lane_scratch[kNLanes] = {
        &partials[0],
        &partials[1],
        &partials[2],
        &partials[3],
    };

    ggml_hpx_rms_norm_f32_reduce_buffer reduce_buf{};
    reduce_buf.sumsq = 0.0f;
    reduce_buf.scale = 0.0f;

    ggml_hpx_rms_norm_partial_f32_ctx partial_ctx{
        x.data(),
        static_cast<int64_t>(x.size()),
    };

    ggml_hpx_rms_norm_finalize_f32_ctx finalize_ctx{
        static_cast<int64_t>(x.size()),
        kEps,
    };

    ggml_hpx_rms_norm_apply_f32_ctx apply_ctx{
        x.data(),
        dst.data(),
        static_cast<int64_t>(x.size()),
    };

    ggml_hpx_cpu_region regions[3] = {
        {
            GGML_HPX_CPU_REGION_KIND_ELEMENTWISE,
            0,
            static_cast<int64_t>(x.size()),
            0,
            &partial_ctx,
            ggml_hpx_rms_norm_partial_f32_run_range,
        },
        {
            // R1: single-threaded finalize.  Aggregates resources->lane_scratch[0 ..
            // resources->n_lanes) (each slot holds the float partial sumsq written
            // by R0) into reduce_buf.sumsq, then derives scale.  The region's
            // begin/end is the full work range [0, n) but the cross-lane aggregation
            // walks [0, resources->n_lanes), not the element range.
            GGML_HPX_CPU_REGION_KIND_REDUCTION,
            0,
            static_cast<int64_t>(x.size()),
            0,
            &finalize_ctx,
            ggml_hpx_rms_norm_finalize_f32_run_range,
        },
        {
            GGML_HPX_CPU_REGION_KIND_ELEMENTWISE,
            0,
            static_cast<int64_t>(x.size()),
            0,
            &apply_ctx,
            ggml_hpx_rms_norm_apply_f32_run_range,
        },
    };

    ggml_hpx_dep_edge deps[2] = {
        {0, 1},
        {1, 2},
    };

    ggml_hpx_cpu_region_group group{
        regions,
        3,
        deps,
        2,
    };

    ggml_hpx_region_resources resources{};
    resources.shared_scratch   = nullptr;
    resources.lane_scratch     = lane_scratch;
    resources.reduction_buffer = &reduce_buf;
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

    // Exact per-lane partial sums of squares for R0.
    // Depends on the current splitter: span=8, n_lanes=4 produces 4 non-empty
    // 2-element chunks.  If the splitter changes these values will need updating.
    //
    //   lane 0: x[0]² + x[1]² = 1  +  4  =   5
    //   lane 1: x[2]² + x[3]² = 9  + 16  =  25
    //   lane 2: x[4]² + x[5]² = 25 + 36  =  61
    //   lane 3: x[6]² + x[7]² = 49 + 64  = 113
    const float expected_partials[kNLanes] = {5.f, 25.f, 61.f, 113.f};

    for (int i = 0; i < kNLanes; ++i)
    {
        EXPECT_NEAR(partials[i], expected_partials[i], 1e-6f) << "lane " << i;
    }

    float partial_total = 0.0f;
    for (int i = 0; i < kNLanes; ++i)
    {
        partial_total += partials[i];
    }

    EXPECT_NEAR(partial_total, ref_sumsq, 1e-6f);
    EXPECT_NEAR(reduce_buf.sumsq, ref_sumsq, 1e-6f);
    EXPECT_NEAR(reduce_buf.scale, ref_scale, 1e-6f);
    EXPECT_TRUE(std::isfinite(reduce_buf.scale));

    for (int i = 0; i < static_cast<int>(x.size()); ++i)
    {
        EXPECT_NEAR(dst[i], ref[i], 1e-6f) << "i=" << i;
    }
}

} // namespace

#endif // GGML_HPX_REGION_DAG
