// test_hpx_region_primitives_swiglu_f32.cpp
//
// Correctness test for the SWIGLU_F32 fused fine-region callback.
//
// SwiGLU is the fused equivalent of the two-step SiLU(gate) * up pipeline:
//     dst[i] = (gate[i] / (1 + exp(-gate[i]))) * up[i]
//
// This is the primitive layer that Milestone B.1 adds so the MLP gate/up
// frozen-packet path can express the real llama graph shape, which emits
// a single GGML_OP_GLU node (subop GGML_GLU_OP_SWIGLU) instead of the
// separate SiLU + MUL nodes v1's matcher was looking for.
//
// Each test:
//   1. Computes a scalar reference result.
//   2. Builds a ggml_hpx_cpu_region pointing at ggml_hpx_swiglu_f32_ctx.
//   3. Executes the region via ggml_hpx_run_region_group (single-region
//      group, n_deps == 0) inside hpx::async().get() after
//      ggml_hpx_tpool_start().
//   4. Compares element-wise against the reference within float tolerance.
//
// The kernel is pure elementwise and uses no resources (no lane_scratch,
// no reduction_buffer).

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

void run_single(ggml_hpx_cpu_region & r, ggml_hpx_region_resources & res)
{
    ggml_hpx_cpu_region_group group{&r, 1, nullptr, 0};
    ggml_hpx_tpool_start();
    hpx::async([&]() {
        ggml_hpx_run_region_group(&group, &res);
    }).get();
}

ggml_hpx_region_resources zero_res()
{
    ggml_hpx_region_resources r{};
    r.shared_scratch   = nullptr;
    r.lane_scratch     = nullptr;
    r.reduction_buffer = nullptr;
    r.n_lanes          = 1;
    return r;
}

float swiglu_ref(float g, float u)
{
    return (g / (1.0f + std::exp(-g))) * u;
}

}    // namespace

// ---------------------------------------------------------------------------
// Correctness across a range of gate / up values, including negatives.
// ---------------------------------------------------------------------------
TEST(HPXRegionSwigluF32, ComputesFusedSiluTimesUp)
{
    const std::vector<float> gate = {
         0.0f, 1.0f, -1.0f,  2.0f, -2.0f, 0.5f, -0.5f,  4.0f,
    };
    const std::vector<float> up = {
         1.0f, 2.0f,  0.5f, -1.0f,  3.0f, 0.0f,  2.5f, -0.25f,
    };
    ASSERT_EQ(gate.size(), up.size());

    const int64_t n = static_cast<int64_t>(gate.size());
    std::vector<float> dst(static_cast<std::size_t>(n), 0.0f);

    std::vector<float> ref(static_cast<std::size_t>(n));
    for (int64_t i = 0; i < n; ++i)
    {
        ref[static_cast<std::size_t>(i)] = swiglu_ref(
            gate[static_cast<std::size_t>(i)],
            up  [static_cast<std::size_t>(i)]);
    }

    ggml_hpx_swiglu_f32_ctx ctx{gate.data(), up.data(), dst.data(), n};
    ggml_hpx_cpu_region r{
        GGML_HPX_CPU_REGION_KIND_ELEMENTWISE,
        0, n, 0,
        &ctx,
        ggml_hpx_swiglu_f32_run_range,
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

// ---------------------------------------------------------------------------
// Aliasing: dst == gate.  swiglu(0, u) == 0 regardless of u because
// silu(0) == 0.  Verifies the kernel reads gate[i] before writing dst[i].
// ---------------------------------------------------------------------------
TEST(HPXRegionSwigluF32, AllowsDstAliasesGate)
{
    std::vector<float> buf = { 0.0f, 1.0f, -1.0f, 2.0f };
    std::vector<float> up  = { 3.0f, 4.0f,  5.0f, 6.0f };
    const int64_t n = static_cast<int64_t>(buf.size());

    std::vector<float> ref(static_cast<std::size_t>(n));
    for (int64_t i = 0; i < n; ++i)
    {
        ref[static_cast<std::size_t>(i)] = swiglu_ref(
            buf[static_cast<std::size_t>(i)],
            up [static_cast<std::size_t>(i)]);
    }

    ggml_hpx_swiglu_f32_ctx ctx{buf.data(), up.data(), buf.data(), n};
    ggml_hpx_cpu_region r{
        GGML_HPX_CPU_REGION_KIND_ELEMENTWISE,
        0, n, 0,
        &ctx,
        ggml_hpx_swiglu_f32_run_range,
    };
    auto res = zero_res();
    run_single(r, res);

    for (int64_t i = 0; i < n; ++i)
    {
        EXPECT_NEAR(buf[static_cast<std::size_t>(i)],
                    ref[static_cast<std::size_t>(i)],
                    1e-6f)
            << "i=" << i;
    }
}

// ---------------------------------------------------------------------------
// Partial range: only [begin, end) is written; the rest of dst is untouched.
// ---------------------------------------------------------------------------
TEST(HPXRegionSwigluF32, RespectsPartialRange)
{
    const std::vector<float> gate = { 1.0f,  1.0f,  1.0f,  1.0f };
    const std::vector<float> up   = { 2.0f,  2.0f,  2.0f,  2.0f };
    const int64_t n = static_cast<int64_t>(gate.size());

    constexpr float sentinel = -7.5f;
    std::vector<float> dst(static_cast<std::size_t>(n), sentinel);

    ggml_hpx_swiglu_f32_ctx ctx{gate.data(), up.data(), dst.data(), n};
    ggml_hpx_cpu_region r{
        GGML_HPX_CPU_REGION_KIND_ELEMENTWISE,
        1, 3, 0,    // only indices 1 and 2 are in range
        &ctx,
        ggml_hpx_swiglu_f32_run_range,
    };
    auto res = zero_res();
    run_single(r, res);

    const float expected = swiglu_ref(1.0f, 2.0f);

    EXPECT_EQ  (dst[0], sentinel);
    EXPECT_NEAR(dst[1], expected, 1e-6f);
    EXPECT_NEAR(dst[2], expected, 1e-6f);
    EXPECT_EQ  (dst[3], sentinel);
}

#else    // GGML_HPX_REGION_DAG

TEST(HPXRegionSwigluF32, RegionDagDisabled)
{
    GTEST_SKIP() << "GGML_HPX_REGION_DAG not defined";
}

#endif    // GGML_HPX_REGION_DAG
