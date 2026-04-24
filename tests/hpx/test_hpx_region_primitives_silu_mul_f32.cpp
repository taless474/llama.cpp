// test_hpx_region_primitives_silu_mul_f32.cpp
//
// Correctness tests for the SILU_F32 and MUL_F32 fine-region callbacks.
//
// Each test:
//   1. Computes a scalar reference result.
//   2. Builds a ggml_hpx_cpu_region pointing at the matching ctx struct.
//   3. Executes the region via ggml_hpx_run_region_group (single-region group,
//      n_deps == 0) inside hpx::async().get() after ggml_hpx_tpool_start().
//   4. Compares element-wise against the reference within float tolerance.
//
// Both ops are pure elementwise and use no resources (no lane_scratch, no
// reduction_buffer). The resource struct is populated with n_lanes == 1 and
// null scratch pointers to match the zero-resource contract.

#include <gtest/gtest.h>

#include <cassert>
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

// Helper: build a minimal single-region group with n_deps == 0, execute it
// inside an HPX context, and return. The region's ctx must outlive the call.
void run_single(ggml_hpx_cpu_region & r, ggml_hpx_region_resources & res)
{
    ggml_hpx_cpu_region_group group{&r, 1, nullptr, 0};
    ggml_hpx_tpool_start();
    hpx::async([&]() {
        ggml_hpx_run_region_group(&group, &res);
    }).get();
}

// Zero-resource bundle used by both tests. Both kernels ignore lane_scratch
// and reduction_buffer; null is safe here.
ggml_hpx_region_resources zero_res()
{
    ggml_hpx_region_resources r{};
    r.shared_scratch   = nullptr;
    r.lane_scratch     = nullptr;
    r.reduction_buffer = nullptr;
    r.n_lanes          = 1;
    return r;
}

} // namespace

// ---------------------------------------------------------------------------
// SiLU tests
// ---------------------------------------------------------------------------

TEST(HPXRegionSiluF32, ComputesCorrectOutput)
{
    const std::vector<float> x = {
        0.f, 1.f, -1.f, 2.f, -2.f, 0.5f, -0.5f, 4.f,
    };
    const int64_t n = static_cast<int64_t>(x.size());
    std::vector<float> dst(static_cast<std::size_t>(n), 0.f);

    // Scalar reference: silu(v) = v / (1 + exp(-v))
    std::vector<float> ref(static_cast<std::size_t>(n));
    for (int64_t i = 0; i < n; ++i)
    {
        const float v = x[static_cast<std::size_t>(i)];
        ref[static_cast<std::size_t>(i)] = v / (1.f + std::exp(-v));
    }

    ggml_hpx_silu_f32_ctx ctx{x.data(), dst.data(), n};
    ggml_hpx_cpu_region r{
        GGML_HPX_CPU_REGION_KIND_ELEMENTWISE,
        0, n, 0,
        &ctx,
        ggml_hpx_silu_f32_run_range,
    };
    auto res = zero_res();
    run_single(r, res);

    for (int64_t i = 0; i < n; ++i)
    {
        EXPECT_NEAR(dst[static_cast<std::size_t>(i)],
                    ref[static_cast<std::size_t>(i)],
                    1e-6f)
            << "i=" << i;
    }
}

TEST(HPXRegionSiluF32, ZeroInput)
{
    // silu(0) == 0
    const int64_t n = 16;
    std::vector<float> x(static_cast<std::size_t>(n), 0.f);
    std::vector<float> dst(static_cast<std::size_t>(n), 1.f);   // pre-fill with non-zero

    ggml_hpx_silu_f32_ctx ctx{x.data(), dst.data(), n};
    ggml_hpx_cpu_region r{
        GGML_HPX_CPU_REGION_KIND_ELEMENTWISE,
        0, n, 0,
        &ctx,
        ggml_hpx_silu_f32_run_range,
    };
    auto res = zero_res();
    run_single(r, res);

    for (int64_t i = 0; i < n; ++i)
    {
        EXPECT_NEAR(dst[static_cast<std::size_t>(i)], 0.f, 1e-7f) << "i=" << i;
    }
}

TEST(HPXRegionSiluF32, LargePositiveApproachesIdentity)
{
    // For large positive x, silu(x) ≈ x (sigmoid → 1).
    const int64_t n = 4;
    const std::vector<float> x = {10.f, 20.f, 50.f, 100.f};
    std::vector<float> dst(static_cast<std::size_t>(n), 0.f);

    ggml_hpx_silu_f32_ctx ctx{x.data(), dst.data(), n};
    ggml_hpx_cpu_region r{
        GGML_HPX_CPU_REGION_KIND_ELEMENTWISE,
        0, n, 0,
        &ctx,
        ggml_hpx_silu_f32_run_range,
    };
    auto res = zero_res();
    run_single(r, res);

    for (int64_t i = 0; i < n; ++i)
    {
        const float xi = x[static_cast<std::size_t>(i)];
        // silu(x) / x should be very close to 1.0 for large positive x
        EXPECT_NEAR(dst[static_cast<std::size_t>(i)] / xi, 1.f, 1e-4f) << "i=" << i;
    }
}

// ---------------------------------------------------------------------------
// Elementwise MUL tests
// ---------------------------------------------------------------------------

TEST(HPXRegionMulF32, ComputesCorrectOutput)
{
    const std::vector<float> a = {1.f, -2.f, 3.f, -4.f, 0.f, 0.5f, -0.5f, 2.f};
    const std::vector<float> b = {2.f,  3.f, -1.f, 0.f, 5.f, 2.f,  2.f, -3.f};
    const int64_t n = static_cast<int64_t>(a.size());
    std::vector<float> dst(static_cast<std::size_t>(n), 0.f);

    std::vector<float> ref(static_cast<std::size_t>(n));
    for (int64_t i = 0; i < n; ++i)
    {
        ref[static_cast<std::size_t>(i)] =
            a[static_cast<std::size_t>(i)] * b[static_cast<std::size_t>(i)];
    }

    ggml_hpx_mul_f32_ctx ctx{a.data(), b.data(), dst.data(), n};
    ggml_hpx_cpu_region r{
        GGML_HPX_CPU_REGION_KIND_ELEMENTWISE,
        0, n, 0,
        &ctx,
        ggml_hpx_mul_f32_run_range,
    };
    auto res = zero_res();
    run_single(r, res);

    for (int64_t i = 0; i < n; ++i)
    {
        EXPECT_NEAR(dst[static_cast<std::size_t>(i)],
                    ref[static_cast<std::size_t>(i)],
                    1e-7f)
            << "i=" << i;
    }
}

TEST(HPXRegionMulF32, MultiLaneProducesCorrectOutput)
{
    // Run with n_lanes > 1 to exercise the lane fan-out path inside
    // ggml_hpx_run_region_group → launch_region_async.
    constexpr int kNLanes = 4;
    const int64_t n = 16;

    std::vector<float> a(static_cast<std::size_t>(n));
    std::vector<float> b(static_cast<std::size_t>(n));
    for (int64_t i = 0; i < n; ++i)
    {
        a[static_cast<std::size_t>(i)] = static_cast<float>(i + 1);
        b[static_cast<std::size_t>(i)] = static_cast<float>(n - i);
    }
    std::vector<float> dst(static_cast<std::size_t>(n), 0.f);
    std::vector<float> ref(static_cast<std::size_t>(n));
    for (int64_t i = 0; i < n; ++i)
    {
        ref[static_cast<std::size_t>(i)] =
            a[static_cast<std::size_t>(i)] * b[static_cast<std::size_t>(i)];
    }

    ggml_hpx_mul_f32_ctx ctx{a.data(), b.data(), dst.data(), n};
    ggml_hpx_cpu_region r{
        GGML_HPX_CPU_REGION_KIND_ELEMENTWISE,
        0, n, 0,
        &ctx,
        ggml_hpx_mul_f32_run_range,
    };

    // Manually allocate lane_scratch even though MUL_F32 doesn't use it;
    // n_lanes must be consistent with the resources contract.
    float lane_store[kNLanes] = {};
    void * lane_ptrs[kNLanes] = {
        &lane_store[0], &lane_store[1], &lane_store[2], &lane_store[3],
    };
    ggml_hpx_region_resources res{};
    res.shared_scratch   = nullptr;
    res.lane_scratch     = lane_ptrs;
    res.reduction_buffer = nullptr;
    res.n_lanes          = kNLanes;

    ggml_hpx_cpu_region_group group{&r, 1, nullptr, 0};
    ggml_hpx_tpool_start();
    hpx::async([&]() {
        ggml_hpx_run_region_group(&group, &res);
    }).get();

    for (int64_t i = 0; i < n; ++i)
    {
        EXPECT_NEAR(dst[static_cast<std::size_t>(i)],
                    ref[static_cast<std::size_t>(i)],
                    1e-7f)
            << "i=" << i;
    }
}

TEST(HPXRegionMulF32, ZeroOperandProducesZero)
{
    const int64_t n = 8;
    const std::vector<float> a = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
    const std::vector<float> b(static_cast<std::size_t>(n), 0.f);
    std::vector<float> dst(static_cast<std::size_t>(n), 99.f);

    ggml_hpx_mul_f32_ctx ctx{a.data(), b.data(), dst.data(), n};
    ggml_hpx_cpu_region r{
        GGML_HPX_CPU_REGION_KIND_ELEMENTWISE,
        0, n, 0,
        &ctx,
        ggml_hpx_mul_f32_run_range,
    };
    auto res = zero_res();
    run_single(r, res);

    for (int64_t i = 0; i < n; ++i)
    {
        EXPECT_EQ(dst[static_cast<std::size_t>(i)], 0.f) << "i=" << i;
    }
}

#endif // GGML_HPX_REGION_DAG
