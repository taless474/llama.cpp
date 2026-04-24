// test_hpx_selective_mlp_glu_packet.cpp
//
// B.1 Item 5 isolation test — proves the selective graph executor's GLU
// frozen-packet dispatch path end-to-end:
//
//   1. prescan_mlp_glu_matches recognises the 3-node GLU pattern
//      (MUL_MAT gate, MUL_MAT up, GGML_OP_GLU[SWIGLU])
//   2. first call compiles the packet exactly once (cache miss)
//   3. second call reuses the cached packet (compile count stays at 1)
//   4. binds from live ggml tensors at each dispatch
//   5. output matches the scalar SWIGLU reference
//
// Structural assertions per call:
//   packet_matches == 1
//   packet_nodes   == 3   (gate MUL_MAT + up MUL_MAT + GLU trigger)
//   lowered_nodes  == 0
//   fallback_nodes == 0
//
// Gated on GGML_HPX_REGION_DAG.  Compiled with GGML_HPX_EXEC_SELECTIVE_TESTING
// so the extern atomics
//   g_hpx_mlp_glu_packet_compile_count
//   g_hpx_mlp_glu_packet_dispatch_count
// are defined in the re-compiled ggml-hpx-exec-selective.cpp TU.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include "ggml-hpx-exec-selective.h"
#include "ggml-hpx-packet.h"
#include "ggml-hpx-runtime.h"

#ifdef GGML_HPX_REGION_DAG

namespace
{

static float silu_ref(float v)
{
    return v / (1.0f + std::exp(-v));
}

// Build a 3-op MLP GLU subgraph with no_alloc=true so the caller owns every
// tensor's data buffer. The selective executor reads ->data directly at bind
// time, so external ownership is required.
struct GluSubgraph
{
    ggml_context * ctx    = nullptr;
    ggml_cgraph *  gf     = nullptr;
    ggml_tensor *  W_gate = nullptr;
    ggml_tensor *  W_up   = nullptr;
    ggml_tensor *  x      = nullptr;
    ggml_tensor *  gate   = nullptr;
    ggml_tensor *  up     = nullptr;
    ggml_tensor *  glu    = nullptr;

    GluSubgraph(int64_t cols, int64_t out_cols, int64_t rows)
    {
        ggml_init_params p{};
        p.mem_size   = 1 * 1024 * 1024;
        p.mem_buffer = nullptr;
        p.no_alloc   = true;
        ctx = ggml_init(p);

        W_gate = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, out_cols);
        W_up   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, out_cols);
        x      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, rows);
        gate   = ggml_mul_mat(ctx, W_gate, x);
        up     = ggml_mul_mat(ctx, W_up,   x);
        glu    = ggml_swiglu_split(ctx, gate, up);

        gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, glu);
    }

    ~GluSubgraph() { if (ctx) ggml_free(ctx); }

    GluSubgraph(const GluSubgraph &)             = delete;
    GluSubgraph & operator=(const GluSubgraph &) = delete;
};

}    // namespace

TEST(SelectiveMlpGluPacket, MatchesCompilesOnceDispatchesTwice)
{
    constexpr int64_t cols     = 4;
    constexpr int64_t out_cols = 3;
    // rows = 1 is the decode-only shape: v1 is intentionally narrow to the
    // single-lane / team=DECODE path; batched prefill is out of scope here.
    constexpr int64_t rows     = 1;

    GluSubgraph mlp(cols, out_cols, rows);

    // External buffers. no_alloc=true: ggml owns no storage; we set ->data
    // manually so the selective executor's bind step reads from these vectors.
    std::vector<float> wg(cols * out_cols);
    std::vector<float> wu(cols * out_cols);
    for (int64_t j = 0; j < out_cols; ++j)
    {
        for (int64_t k = 0; k < cols; ++k)
        {
            wg[j * cols + k] = 0.1f * static_cast<float>(j * cols + k + 1);
            wu[j * cols + k] = 0.2f * static_cast<float>(j * cols + k + 1);
        }
    }
    std::vector<float> xb      = { 1.0f, 2.0f, 3.0f, 4.0f };    // rows=1, cols=4
    std::vector<float> gate_buf(rows * out_cols, 0.0f);
    std::vector<float> up_buf  (rows * out_cols, 0.0f);
    std::vector<float> out_buf (rows * out_cols, 0.0f);

    mlp.W_gate->data = wg      .data();
    mlp.W_up  ->data = wu      .data();
    mlp.x     ->data = xb      .data();
    mlp.gate  ->data = gate_buf.data();
    mlp.up    ->data = up_buf  .data();
    mlp.glu   ->data = out_buf .data();

    // HPX substrate: tpool → packet runtime → GLU cache.  All at n_lanes=1
    // per the v1 scope note in ggml-hpx-exec-selective.h.
    ggml_hpx_tpool_start();

    ggml_hpx_packet_runtime * rt =
        ggml_hpx_packet_runtime_create(/*n_decode_threads=*/1);
    ASSERT_NE(rt, nullptr);

    // policy_version=1 is the current MLP_GLU_F32 compiler policy; bump in
    // both the create call and lookup_or_compile_mlp_glu if lowering changes.
    ggml_hpx_mlp_glu_packet_cache * glu_cache =
        ggml_hpx_mlp_glu_packet_cache_create(
            /*n_lanes=*/        1,
            /*seq_regime=*/     0,
            /*policy_version=*/ 1);
    ASSERT_NE(glu_cache, nullptr);

    // cpu_be is required by the entry point but should never be invoked here:
    // lowered_nodes == 0 and fallback_nodes == 0 guarantee that all dispatch
    // goes through the GLU frozen-packet path, not the fine-region or fallback
    // paths. Any routing to cpu_be would be a matcher regression.
    ggml_backend_t cpu_be = ggml_backend_cpu_init();
    ASSERT_NE(cpu_be, nullptr);

    // Scalar reference: gate[j] = dot(x, W_gate[j]), up[j] = dot(x, W_up[j])
    // out[j] = silu(gate[j]) * up[j]
    auto scalar_ref = [&](int r, int j) -> float
    {
        float g = 0.0f;
        float u = 0.0f;
        for (int k = 0; k < static_cast<int>(cols); ++k)
        {
            g += xb[r * cols + k] * wg[j * cols + k];
            u += xb[r * cols + k] * wu[j * cols + k];
        }
        return silu_ref(g) * u;
    };

    auto check_output_matches_ref = [&](const char * tag)
    {
        for (int r = 0; r < static_cast<int>(rows); ++r)
        {
            for (int j = 0; j < static_cast<int>(out_cols); ++j)
            {
                const float expected = scalar_ref(r, j);
                const float got      = out_buf[r * out_cols + j];
                const float tol      = std::max(
                    1e-5f, std::abs(expected) * 1e-5f);
                EXPECT_NEAR(got, expected, tol)
                    << tag << " r=" << r << " j=" << j;
            }
        }
    };

    // Reset all counters before measuring.
    g_hpx_mlp_glu_packet_compile_count  = 0;
    g_hpx_mlp_glu_packet_dispatch_count = 0;
    g_hpx_selective_lowered_count       = 0;
    g_hpx_selective_executed_count      = 0;

    // Build the env. mlp_cache is null: gate/up path is not under test here,
    // and leaving it null confirms the GLU prescan fires independently.
    ggml_hpx_selective_packet_env penv{};
    penv.rt            = rt;
    penv.mlp_cache     = nullptr;    // gate/up path not under test here
    penv.mlp_glu_cache = glu_cache;

    // ── Call 1: cache miss ────────────────────────────────────────────────
    ggml_hpx_selective_stats s1{};
    ASSERT_TRUE(ggml_hpx_exec_graph_selective_mul_mat(
        mlp.gf, cpu_be, /*n_lanes=*/0, &s1, &penv));

    EXPECT_EQ(s1.packet_matches,  1u);
    EXPECT_EQ(s1.packet_nodes,    3u);    // gate + up + glu trigger
    EXPECT_EQ(s1.lowered_nodes,   0u);
    EXPECT_EQ(s1.fallback_nodes,  0u);
    EXPECT_GT(s1.packet_dispatch_ns, 0ull);
    EXPECT_EQ(g_hpx_mlp_glu_packet_compile_count .load(), 1);
    EXPECT_EQ(g_hpx_mlp_glu_packet_dispatch_count.load(), 1);

    // Also confirm the non-GLU counters were not touched.
    EXPECT_EQ(g_hpx_selective_lowered_count .load(), 0);
    EXPECT_EQ(g_hpx_selective_executed_count.load(), 0);

    check_output_matches_ref("call1");

    // ── Call 2: cache hit ─────────────────────────────────────────────────
    // Zero the output so a skipped dispatch would leave zeros behind.
    std::fill(out_buf.begin(), out_buf.end(), 0.0f);

    ggml_hpx_selective_stats s2{};
    ASSERT_TRUE(ggml_hpx_exec_graph_selective_mul_mat(
        mlp.gf, cpu_be, /*n_lanes=*/0, &s2, &penv));

    EXPECT_EQ(s2.packet_matches,  1u);
    EXPECT_EQ(s2.packet_nodes,    3u);
    EXPECT_EQ(s2.lowered_nodes,   0u);
    EXPECT_EQ(s2.fallback_nodes,  0u);
    EXPECT_GT(s2.packet_dispatch_ns, 0ull);

    // Load-bearing cache-reuse assertion: compile_count must NOT bump on the
    // second call, but dispatch_count must reach 2.
    EXPECT_EQ(g_hpx_mlp_glu_packet_compile_count .load(), 1);
    EXPECT_EQ(g_hpx_mlp_glu_packet_dispatch_count.load(), 2);

    check_output_matches_ref("call2");

    // Teardown (cache before runtime — consistent with the runtime-is-substrate
    // layering; entry dtors free frozen packets, which hold no runtime ptr).
    ggml_hpx_mlp_glu_packet_cache_destroy(glu_cache);
    ggml_hpx_packet_runtime_destroy(rt);
    ggml_backend_free(cpu_be);
}

#else    // GGML_HPX_REGION_DAG

TEST(SelectiveMlpGluPacket, RegionDagDisabled)
{
    GTEST_SKIP() << "GGML_HPX_REGION_DAG not defined";
}

#endif    // GGML_HPX_REGION_DAG
