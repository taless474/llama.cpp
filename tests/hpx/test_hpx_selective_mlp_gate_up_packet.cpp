// test_hpx_selective_mlp_gate_up_packet.cpp
//
// Phase 5 M6 isolation test — proves the selective graph executor's
// frozen-packet dispatch path:
//
//   1. matches the 4-op MLP gate/up pattern via prescan_mlp_matches
//   2. compiles the packet exactly once on cache miss
//   3. reuses the cached packet on the second call (compile count stays 1)
//   4. binds from live ggml tensors at each dispatch (no cached data ptrs)
//   5. produces the same output as a scalar reference
//
// Gated on GGML_HPX_REGION_DAG.  Compiled with GGML_HPX_EXEC_SELECTIVE_TESTING
// (see tests/hpx/CMakeLists.txt) so the extern atomics
//   g_hpx_mlp_packet_compile_count
//   g_hpx_mlp_packet_dispatch_count
// are defined in the re-compiled ggml-hpx-exec-selective.cpp TU and
// declared visible to this test.

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

float silu_ref(float v)
{
    return v / (1.0f + std::exp(-v));
}

// Build a 4-op MLP gate/up subgraph with no_alloc=true so the caller owns
// every tensor's data buffer (required because the selective executor
// reads ->data directly at bind time).
struct MlpSubgraph
{
    ggml_context * ctx      = nullptr;
    ggml_cgraph *  gf       = nullptr;
    ggml_tensor *  W_gate   = nullptr;
    ggml_tensor *  W_up     = nullptr;
    ggml_tensor *  x        = nullptr;
    ggml_tensor *  gate     = nullptr;
    ggml_tensor *  up       = nullptr;
    ggml_tensor *  gate_act = nullptr;
    ggml_tensor *  out      = nullptr;

    MlpSubgraph(int64_t cols, int64_t out_cols, int64_t rows)
    {
        ggml_init_params p{};
        p.mem_size   = 1 * 1024 * 1024;
        p.mem_buffer = nullptr;
        p.no_alloc   = true;
        ctx = ggml_init(p);

        W_gate   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, out_cols);
        W_up     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, out_cols);
        x        = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, rows);
        gate     = ggml_mul_mat(ctx, W_gate, x);
        up       = ggml_mul_mat(ctx, W_up,   x);
        gate_act = ggml_silu  (ctx, gate);
        out      = ggml_mul   (ctx, gate_act, up);

        gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
    }

    ~MlpSubgraph() { if (ctx) ggml_free(ctx); }

    MlpSubgraph(const MlpSubgraph &)             = delete;
    MlpSubgraph & operator=(const MlpSubgraph &) = delete;
};

}    // namespace

TEST(SelectiveMlpGateUpPacket, MatchesCompilesOnceDispatchesTwice)
{
    constexpr int64_t cols     = 4;
    constexpr int64_t out_cols = 3;
    // rows = 1 is the decode-only shape that v1 is intentionally narrow
    // to: the first-deployment packet path is single-lane / team=DECODE
    // and only exercises the dispatch mechanism, not batched prefill.
    constexpr int64_t rows     = 1;

    MlpSubgraph mlp(cols, out_cols, rows);

    // External buffers. no_alloc=true means the ggml context doesn't own
    // any storage; we assign ->data manually so the selective executor's
    // bind step reads from these vectors.
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
    std::vector<float> xb       = { 1.0f, 2.0f, 3.0f, 4.0f };    // rows=1, cols=4
    std::vector<float> gate_buf (rows * out_cols, 0.0f);
    std::vector<float> up_buf   (rows * out_cols, 0.0f);
    std::vector<float> ga_buf   (rows * out_cols, 0.0f);
    std::vector<float> out_buf  (rows * out_cols, 0.0f);

    mlp.W_gate  ->data = wg      .data();
    mlp.W_up    ->data = wu      .data();
    mlp.x       ->data = xb      .data();
    mlp.gate    ->data = gate_buf.data();
    mlp.up      ->data = up_buf  .data();
    mlp.gate_act->data = ga_buf  .data();
    mlp.out     ->data = out_buf .data();

    // HPX substrate: tpool → packet runtime → MLP cache.  All at n_lanes=1
    // per the v1 scope note in ggml-hpx-exec-selective.h.
    ggml_hpx_tpool_start();

    ggml_hpx_packet_runtime * rt =
        ggml_hpx_packet_runtime_create(/*n_decode_threads=*/1);
    ASSERT_NE(rt, nullptr);

    ggml_hpx_mlp_gate_up_packet_cache * cache =
        ggml_hpx_mlp_gate_up_packet_cache_create(
            /*n_lanes=*/        1,
            /*seq_regime=*/     0,
            /*policy_version=*/ 1);
    ASSERT_NE(cache, nullptr);

    // cpu_be is supplied only because the selective entry point's signature
    // requires it; the CPU-fallback and fine-region paths are NOT the
    // intended paths under test here.  The lowered_nodes == 0 and
    // fallback_nodes == 0 assertions below guarantee this test exercises
    // only the frozen-packet dispatch path.
    ggml_backend_t cpu_be = ggml_backend_cpu_init();
    ASSERT_NE(cpu_be, nullptr);

    // Scalar reference for the 4-op MLP gate/up math.
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

    // Reset counters before we start.
    g_hpx_mlp_packet_compile_count  = 0;
    g_hpx_mlp_packet_dispatch_count = 0;
    g_hpx_selective_lowered_count   = 0;
    g_hpx_selective_executed_count  = 0;

    // ── Call 1: cache miss ────────────────────────────────────────────
    ggml_hpx_selective_packet_env penv{};
    penv.rt            = rt;
    penv.mlp_cache     = cache;
    penv.mlp_glu_cache = nullptr;    // GLU path not under test here

    ggml_hpx_selective_stats s1{};
    ASSERT_TRUE(ggml_hpx_exec_graph_selective_mul_mat(
        mlp.gf, cpu_be, /*n_lanes=*/0, &s1, &penv));

    EXPECT_EQ(s1.packet_matches,  1u);
    EXPECT_EQ(s1.packet_nodes,    4u);
    EXPECT_EQ(s1.lowered_nodes,   0u);
    EXPECT_EQ(s1.fallback_nodes,  0u);
    EXPECT_GT(s1.packet_dispatch_ns, 0ull);
    EXPECT_EQ(g_hpx_mlp_packet_compile_count .load(), 1);
    EXPECT_EQ(g_hpx_mlp_packet_dispatch_count.load(), 1);

    check_output_matches_ref("call1");

    // ── Call 2: cache hit ─────────────────────────────────────────────
    // Clear the output so a skipped dispatch would leave zeros behind.
    std::fill(out_buf.begin(), out_buf.end(), 0.0f);

    ggml_hpx_selective_stats s2{};
    ASSERT_TRUE(ggml_hpx_exec_graph_selective_mul_mat(
        mlp.gf, cpu_be, /*n_lanes=*/0, &s2, &penv));

    EXPECT_EQ(s2.packet_matches,  1u);
    EXPECT_EQ(s2.packet_nodes,    4u);
    EXPECT_EQ(s2.lowered_nodes,   0u);
    EXPECT_EQ(s2.fallback_nodes,  0u);
    EXPECT_GT(s2.packet_dispatch_ns, 0ull);

    // The load-bearing cache-reuse assertion: compile_count must NOT bump
    // on the second call, but dispatch_count must.
    EXPECT_EQ(g_hpx_mlp_packet_compile_count .load(), 1);
    EXPECT_EQ(g_hpx_mlp_packet_dispatch_count.load(), 2);

    check_output_matches_ref("call2");

    // Teardown (cache destroy runs before runtime destroy — entry dtors
    // free frozen packets, which don't reference the runtime, but we keep
    // the ordering consistent with the runtime-is-substrate layering).
    ggml_hpx_mlp_gate_up_packet_cache_destroy(cache);
    ggml_hpx_packet_runtime_destroy(rt);
    ggml_backend_free(cpu_be);
}

#else    // GGML_HPX_REGION_DAG

TEST(SelectiveMlpGateUpPacket, RegionDagDisabled)
{
    GTEST_SKIP() << "GGML_HPX_REGION_DAG not defined";
}

#endif    // GGML_HPX_REGION_DAG
