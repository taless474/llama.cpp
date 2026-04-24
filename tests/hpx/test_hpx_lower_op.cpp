// test_hpx_lower_op.cpp
//
// Correctness tests for ggml_hpx_lower_op.
//
// Each test builds a tiny hand-constructed ggml graph using no_alloc=true
// (tensor structs only; data pointers set manually), calls lower_op, then:
//
//   1. Verifies group structure: n_regions, n_deps, region kinds, work ranges,
//      dep edges, and that each region's ctx points into ctx_buf[i].
//   2. Executes the lowered group via ggml_hpx_run_region_group inside
//      hpx::async().get() and compares output element-wise to a scalar ref.
//
// Covered ops: SILU_F32, MUL_F32, MUL_MAT_F32, MUL_MAT_Q4K_F32, RMS_NORM_F32.
// Aliasing: in-place SILU (dst==src), MUL with dst==a, MUL with dst==b.
// Rejection: unsupported op, non-F32 type, UNARY non-SILU subop,
//            RMS_NORM multi-row, null src/dst data, null MUL second source,
//            null MUL_MAT weight, null MUL_MAT activation,
//            MUL_MAT Q4_K with rows > 1.

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <vector>

#include <hpx/future.hpp>

#include "ggml-hpx-lower.h"
#include "ggml-hpx-region-exec.h"
#include "ggml-hpx-runtime.h"
#include "ggml.h"

#ifdef GGML_HPX_REGION_DAG

// Forward declarations — symbols provided by the linked ggml library.
// Signatures match ggml/src/ggml-cpu/quants.h exactly. No header is included
// because quants.h is an internal header not reachable from ggml/include/.
// This follows the same pattern as ggml_vec_dot_f32 in ggml-hpx-region-exec.cpp.
extern "C"
{
    void quantize_row_q4_K(const float * x, void * y, int64_t k);
    void quantize_row_q8_K(const float * x, void * y, int64_t k);
    void ggml_vec_dot_q4_K_q8_K(int n, float * s, size_t bs,
                                  const void * vx, size_t bx,
                                  const void * vy, size_t by,
                                  int nrc);
}

namespace
{

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

// RAII ggml context. no_alloc=true: only tensor structs, no data.
struct GCtx
{
    ggml_context * ctx;

    explicit GCtx(size_t mem_bytes = 64 * 1024)
    {
        ggml_init_params p{};
        p.mem_size   = mem_bytes;
        p.mem_buffer = nullptr;
        p.no_alloc   = true;
        ctx = ggml_init(p);
    }
    ~GCtx() { if (ctx) ggml_free(ctx); }

    GCtx(const GCtx &)            = delete;
    GCtx & operator=(const GCtx &) = delete;
};

// Run a lowered group inside an HPX context on n_lanes lanes.
// lane_scratch / reduction_buffer may be null for ops that don't use them.
void run_group(
    ggml_hpx_lowering *     lo,
    int                     n_lanes,
    void **                 lane_scratch,
    void *                  reduction_buffer)
{
    ggml_hpx_region_resources res{};
    res.shared_scratch   = nullptr;
    res.lane_scratch     = lane_scratch;
    res.reduction_buffer = reduction_buffer;
    res.n_lanes          = n_lanes;

    ggml_hpx_tpool_start();
    hpx::async([&]() {
        ggml_hpx_run_region_group(&lo->group, &res);
    }).get();
}

// ---------------------------------------------------------------------------
// SILU F32
// ---------------------------------------------------------------------------

TEST(LowerOpSiluF32, StructureAndExecution)
{
    GCtx gc;
    const int64_t n = 16;

    ggml_tensor * src  = ggml_new_tensor_1d(gc.ctx, GGML_TYPE_F32, n);
    ggml_tensor * node = ggml_silu(gc.ctx, src);

    std::vector<float> xd(static_cast<std::size_t>(n));
    std::vector<float> dstd(static_cast<std::size_t>(n), 0.f);
    for (int64_t i = 0; i < n; ++i)
    {
        xd[static_cast<std::size_t>(i)] = static_cast<float>(i) - static_cast<float>(n) / 2.f;
    }
    src->data  = xd.data();
    node->data = dstd.data();

    ggml_hpx_lowering lo;
    ggml_hpx_lowering_init(&lo);
    ASSERT_TRUE(ggml_hpx_lower_op(node, &lo));

    // ── structural checks ─────────────────────────────────────────────────
    EXPECT_EQ(lo.group.n_regions, 1);
    EXPECT_EQ(lo.group.n_deps,    0);
    EXPECT_EQ(lo.group.regions,   lo.regions);
    EXPECT_EQ(lo.group.deps,      nullptr);    // zero-dep ops must set deps to nullptr

    EXPECT_EQ(lo.regions[0].kind,      GGML_HPX_CPU_REGION_KIND_ELEMENTWISE);
    EXPECT_EQ(lo.regions[0].begin,     int64_t{0});
    EXPECT_EQ(lo.regions[0].end,       n);
    EXPECT_EQ(lo.regions[0].ctx,       static_cast<void *>(lo.ctx_buf[0]));
    EXPECT_EQ(lo.regions[0].run_range, ggml_hpx_silu_f32_run_range);

    // ── execution + output check ──────────────────────────────────────────
    run_group(&lo, /*n_lanes=*/1, nullptr, nullptr);

    for (int64_t i = 0; i < n; ++i)
    {
        const float v   = xd[static_cast<std::size_t>(i)];
        const float ref = v / (1.f + std::exp(-v));
        EXPECT_NEAR(dstd[static_cast<std::size_t>(i)], ref, 1e-6f) << "i=" << i;
    }
}

// ---------------------------------------------------------------------------
// Elementwise MUL F32
// ---------------------------------------------------------------------------

TEST(LowerOpMulF32, StructureAndExecution)
{
    GCtx gc;
    const int64_t n = 12;

    ggml_tensor * a    = ggml_new_tensor_1d(gc.ctx, GGML_TYPE_F32, n);
    ggml_tensor * b    = ggml_new_tensor_1d(gc.ctx, GGML_TYPE_F32, n);
    ggml_tensor * node = ggml_mul(gc.ctx, a, b);

    std::vector<float> ad(static_cast<std::size_t>(n));
    std::vector<float> bd(static_cast<std::size_t>(n));
    std::vector<float> dstd(static_cast<std::size_t>(n), 0.f);
    for (int64_t i = 0; i < n; ++i)
    {
        ad[static_cast<std::size_t>(i)] = static_cast<float>(i + 1);
        bd[static_cast<std::size_t>(i)] = static_cast<float>(n - i);
    }
    a->data    = ad.data();
    b->data    = bd.data();
    node->data = dstd.data();

    ggml_hpx_lowering lo;
    ggml_hpx_lowering_init(&lo);
    ASSERT_TRUE(ggml_hpx_lower_op(node, &lo));

    // ── structural checks ─────────────────────────────────────────────────
    EXPECT_EQ(lo.group.n_regions, 1);
    EXPECT_EQ(lo.group.n_deps,    0);
    EXPECT_EQ(lo.group.regions,   lo.regions);

    EXPECT_EQ(lo.group.deps,      nullptr);    // zero-dep ops must set deps to nullptr

    EXPECT_EQ(lo.regions[0].kind,      GGML_HPX_CPU_REGION_KIND_ELEMENTWISE);
    EXPECT_EQ(lo.regions[0].begin,     int64_t{0});
    EXPECT_EQ(lo.regions[0].end,       n);
    EXPECT_EQ(lo.regions[0].ctx,       static_cast<void *>(lo.ctx_buf[0]));
    EXPECT_EQ(lo.regions[0].run_range, ggml_hpx_mul_f32_run_range);

    // ── execution + output check ──────────────────────────────────────────
    run_group(&lo, 1, nullptr, nullptr);

    for (int64_t i = 0; i < n; ++i)
    {
        const float ref = ad[static_cast<std::size_t>(i)] * bd[static_cast<std::size_t>(i)];
        EXPECT_NEAR(dstd[static_cast<std::size_t>(i)], ref, 1e-7f) << "i=" << i;
    }
}

// ---------------------------------------------------------------------------
// MUL_MAT F32
// ---------------------------------------------------------------------------

TEST(LowerOpMulMatF32, StructureAndExecution)
{
    // rows=2, cols=4, out_cols=3.
    // w is identity-like: w[out_col][col] selects input column out_col.
    // Expected: y[row][col] = x[row][col] for col in {0,1,2}.
    GCtx gc;
    constexpr int64_t rows     = 2;
    constexpr int64_t cols     = 4;
    constexpr int64_t out_cols = 3;

    // ggml MUL_MAT: src[0]=w [cols × out_cols], src[1]=x [cols × rows]
    ggml_tensor * w    = ggml_new_tensor_2d(gc.ctx, GGML_TYPE_F32, cols, out_cols);
    ggml_tensor * x    = ggml_new_tensor_2d(gc.ctx, GGML_TYPE_F32, cols, rows);
    ggml_tensor * node = ggml_mul_mat(gc.ctx, w, x);

    const float wd[out_cols * cols] = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
    };
    const float xd[rows * cols] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
    };
    float yd[rows * out_cols];
    std::memset(yd, 0, sizeof(yd));

    w->data    = const_cast<float *>(wd);
    x->data    = const_cast<float *>(xd);
    node->data = yd;

    ggml_hpx_lowering lo;
    ggml_hpx_lowering_init(&lo);
    ASSERT_TRUE(ggml_hpx_lower_op(node, &lo));

    // ── structural checks ─────────────────────────────────────────────────
    EXPECT_EQ(lo.group.n_regions, 1);
    EXPECT_EQ(lo.group.n_deps,    0);
    EXPECT_EQ(lo.group.regions,   lo.regions);

    EXPECT_EQ(lo.group.deps,      nullptr);    // zero-dep ops must set deps to nullptr

    EXPECT_EQ(lo.regions[0].kind,      GGML_HPX_CPU_REGION_KIND_MATMUL);
    EXPECT_EQ(lo.regions[0].begin,     int64_t{0});
    EXPECT_EQ(lo.regions[0].end,       out_cols);
    EXPECT_EQ(lo.regions[0].ctx,       static_cast<void *>(lo.ctx_buf[0]));
    EXPECT_EQ(lo.regions[0].run_range, ggml_hpx_mul_mat_f32_run_range);

    // Verify ctx was wired correctly via the exposed fields.
    auto * ctx = reinterpret_cast<ggml_hpx_mul_mat_f32_ctx *>(lo.ctx_buf[0]);
    EXPECT_EQ(ctx->rows,     rows);
    EXPECT_EQ(ctx->cols,     cols);
    EXPECT_EQ(ctx->out_cols, out_cols);

    // ── execution + output check ──────────────────────────────────────────
    run_group(&lo, 1, nullptr, nullptr);

    // y[row][col] = x[row][col] for col in {0,1,2}
    const float expected[rows * out_cols] = {
        1, 2, 3,
        5, 6, 7,
    };
    for (int64_t i = 0; i < rows * out_cols; ++i)
    {
        EXPECT_NEAR(yd[i], expected[i], 1e-5f) << "flat index=" << i;
    }
}

// ---------------------------------------------------------------------------
// RMS_NORM F32 (single row)
// ---------------------------------------------------------------------------

TEST(LowerOpRmsNormF32, StructureAndExecution)
{
    GCtx gc;
    constexpr int     kNLanes = 4;
    constexpr float   kEps    = 1e-5f;
    constexpr int64_t n       = 8;

    ggml_tensor * src  = ggml_new_tensor_1d(gc.ctx, GGML_TYPE_F32, n);
    ggml_tensor * node = ggml_rms_norm(gc.ctx, src, kEps);

    const std::vector<float> xd = {1.f, -2.f, 3.f, -4.f, 5.f, -6.f, 7.f, -8.f};
    std::vector<float> dstd(static_cast<std::size_t>(n), 0.f);
    src->data  = const_cast<float *>(xd.data());
    node->data = dstd.data();

    ggml_hpx_lowering lo;
    ggml_hpx_lowering_init(&lo);
    ASSERT_TRUE(ggml_hpx_lower_op(node, &lo));

    // ── structural checks ─────────────────────────────────────────────────
    EXPECT_EQ(lo.group.n_regions, 3);
    EXPECT_EQ(lo.group.n_deps,    2);
    EXPECT_EQ(lo.group.regions,   lo.regions);
    EXPECT_EQ(lo.group.deps,      lo.deps);

    EXPECT_EQ(lo.regions[0].kind,      GGML_HPX_CPU_REGION_KIND_ELEMENTWISE);
    EXPECT_EQ(lo.regions[0].begin,     int64_t{0});
    EXPECT_EQ(lo.regions[0].end,       n);
    EXPECT_EQ(lo.regions[0].ctx,       static_cast<void *>(lo.ctx_buf[0]));
    EXPECT_EQ(lo.regions[0].run_range, ggml_hpx_rms_norm_partial_f32_run_range);

    EXPECT_EQ(lo.regions[1].kind,      GGML_HPX_CPU_REGION_KIND_REDUCTION);
    EXPECT_EQ(lo.regions[1].begin,     int64_t{0});
    EXPECT_EQ(lo.regions[1].end,       n);
    EXPECT_EQ(lo.regions[1].ctx,       static_cast<void *>(lo.ctx_buf[1]));
    EXPECT_EQ(lo.regions[1].run_range, ggml_hpx_rms_norm_finalize_f32_run_range);

    EXPECT_EQ(lo.regions[2].kind,      GGML_HPX_CPU_REGION_KIND_ELEMENTWISE);
    EXPECT_EQ(lo.regions[2].begin,     int64_t{0});
    EXPECT_EQ(lo.regions[2].end,       n);
    EXPECT_EQ(lo.regions[2].ctx,       static_cast<void *>(lo.ctx_buf[2]));
    EXPECT_EQ(lo.regions[2].run_range, ggml_hpx_rms_norm_apply_f32_run_range);

    // Dep edges: 0→1 and 1→2 (order not guaranteed; check as a set).
    bool saw_0_1 = false;
    bool saw_1_2 = false;
    for (int i = 0; i < 2; ++i)
    {
        if (lo.deps[i].src == 0 && lo.deps[i].dst == 1) saw_0_1 = true;
        if (lo.deps[i].src == 1 && lo.deps[i].dst == 2) saw_1_2 = true;
    }
    EXPECT_TRUE(saw_0_1) << "missing dep edge 0→1";
    EXPECT_TRUE(saw_1_2) << "missing dep edge 1→2";

    // ── execution + output check ──────────────────────────────────────────
    float partials[kNLanes] = {};
    void * lane_ptrs[kNLanes] = {
        &partials[0], &partials[1], &partials[2], &partials[3],
    };
    ggml_hpx_rms_norm_f32_reduce_buffer reduce_buf{};
    run_group(&lo, kNLanes, lane_ptrs, &reduce_buf);

    // Scalar reference.
    float sumsq = 0.f;
    for (int64_t i = 0; i < n; ++i) sumsq += xd[static_cast<std::size_t>(i)] * xd[static_cast<std::size_t>(i)];
    const float scale = 1.f / std::sqrt(sumsq / static_cast<float>(n) + kEps);
    std::vector<float> ref(static_cast<std::size_t>(n));
    for (int64_t i = 0; i < n; ++i)
        ref[static_cast<std::size_t>(i)] = xd[static_cast<std::size_t>(i)] * scale;

    EXPECT_NEAR(reduce_buf.sumsq, sumsq, 1e-5f);
    EXPECT_NEAR(reduce_buf.scale, scale, 1e-6f);
    for (int64_t i = 0; i < n; ++i)
    {
        EXPECT_NEAR(dstd[static_cast<std::size_t>(i)], ref[static_cast<std::size_t>(i)], 1e-6f)
            << "i=" << i;
    }
}

// ---------------------------------------------------------------------------
// Rejection cases
// ---------------------------------------------------------------------------

TEST(LowerOpRejects, NonF32TypeReturnsFalse)
{
    GCtx gc;
    ggml_tensor * src  = ggml_new_tensor_1d(gc.ctx, GGML_TYPE_F16, 8);
    ggml_tensor * node = ggml_silu(gc.ctx, src);
    float dummy = 0.f;
    src->data  = &dummy;
    node->data = &dummy;

    ggml_hpx_lowering lo;
    ggml_hpx_lowering_init(&lo);
    EXPECT_FALSE(ggml_hpx_lower_op(node, &lo));
    EXPECT_EQ(lo.group.n_regions, 0);
}

TEST(LowerOpRejects, NonSiluUnaryReturnsFalse)
{
    GCtx gc;
    ggml_tensor * src  = ggml_new_tensor_1d(gc.ctx, GGML_TYPE_F32, 8);
    ggml_tensor * node = ggml_relu(gc.ctx, src);    // GGML_UNARY_OP_RELU, not SILU
    float dummy = 0.f;
    src->data  = &dummy;
    node->data = &dummy;

    ggml_hpx_lowering lo;
    ggml_hpx_lowering_init(&lo);
    EXPECT_FALSE(ggml_hpx_lower_op(node, &lo));
    EXPECT_EQ(lo.group.n_regions, 0);
}

TEST(LowerOpRejects, RmsNormMultiRowReturnsFalse)
{
    GCtx gc;
    // 2-row tensor: ne[0]=8, ne[1]=2 → should be rejected
    ggml_tensor * src  = ggml_new_tensor_2d(gc.ctx, GGML_TYPE_F32, 8, 2);
    ggml_tensor * node = ggml_rms_norm(gc.ctx, src, 1e-5f);
    float dummy = 0.f;
    src->data  = &dummy;
    node->data = &dummy;

    ggml_hpx_lowering lo;
    ggml_hpx_lowering_init(&lo);
    EXPECT_FALSE(ggml_hpx_lower_op(node, &lo));
    EXPECT_EQ(lo.group.n_regions, 0);
}

TEST(LowerOpRejects, NullSrcDataReturnsFalse)
{
    GCtx gc;
    ggml_tensor * src  = ggml_new_tensor_1d(gc.ctx, GGML_TYPE_F32, 8);
    ggml_tensor * node = ggml_silu(gc.ctx, src);
    // src->data left null; node->data set so only src is the culprit
    float dummy = 0.f;
    node->data = &dummy;

    ggml_hpx_lowering lo;
    ggml_hpx_lowering_init(&lo);
    EXPECT_FALSE(ggml_hpx_lower_op(node, &lo));
    EXPECT_EQ(lo.group.n_regions, 0);
}

TEST(LowerOpRejects, NullNodeDataReturnsFalse)
{
    GCtx gc;
    ggml_tensor * src  = ggml_new_tensor_1d(gc.ctx, GGML_TYPE_F32, 8);
    ggml_tensor * node = ggml_silu(gc.ctx, src);
    float dummy = 0.f;
    src->data = &dummy;
    // node->data left null

    ggml_hpx_lowering lo;
    ggml_hpx_lowering_init(&lo);
    EXPECT_FALSE(ggml_hpx_lower_op(node, &lo));
    EXPECT_EQ(lo.group.n_regions, 0);
}

TEST(LowerOpRejects, MulNullSecondSourceReturnsFalse)
{
    GCtx gc;
    ggml_tensor * a    = ggml_new_tensor_1d(gc.ctx, GGML_TYPE_F32, 8);
    ggml_tensor * b    = ggml_new_tensor_1d(gc.ctx, GGML_TYPE_F32, 8);
    ggml_tensor * node = ggml_mul(gc.ctx, a, b);
    float dummy = 0.f;
    a->data    = &dummy;
    // b->data left null
    node->data = &dummy;

    ggml_hpx_lowering lo;
    ggml_hpx_lowering_init(&lo);
    EXPECT_FALSE(ggml_hpx_lower_op(node, &lo));
    EXPECT_EQ(lo.group.n_regions, 0);
}

TEST(LowerOpRejects, MulMatNullWeightReturnsFalse)
{
    GCtx gc;
    ggml_tensor * w    = ggml_new_tensor_2d(gc.ctx, GGML_TYPE_F32, 4, 3);
    ggml_tensor * x    = ggml_new_tensor_2d(gc.ctx, GGML_TYPE_F32, 4, 2);
    ggml_tensor * node = ggml_mul_mat(gc.ctx, w, x);
    float dummy = 0.f;
    // w->data left null
    x->data    = &dummy;
    node->data = &dummy;

    ggml_hpx_lowering lo;
    ggml_hpx_lowering_init(&lo);
    EXPECT_FALSE(ggml_hpx_lower_op(node, &lo));
    EXPECT_EQ(lo.group.n_regions, 0);
}

TEST(LowerOpRejects, MulMatNullActivationReturnsFalse)
{
    GCtx gc;
    ggml_tensor * w    = ggml_new_tensor_2d(gc.ctx, GGML_TYPE_F32, 4, 3);
    ggml_tensor * x    = ggml_new_tensor_2d(gc.ctx, GGML_TYPE_F32, 4, 2);
    ggml_tensor * node = ggml_mul_mat(gc.ctx, w, x);
    float dummy = 0.f;
    w->data = &dummy;
    // x->data left null
    node->data = &dummy;

    ggml_hpx_lowering lo;
    ggml_hpx_lowering_init(&lo);
    EXPECT_FALSE(ggml_hpx_lower_op(node, &lo));
    EXPECT_EQ(lo.group.n_regions, 0);
}

// ---------------------------------------------------------------------------
// Aliasing guarantees
// ---------------------------------------------------------------------------

TEST(LowerOpSiluF32, InPlaceAliasWorks)
{
    // dst == src: lower_op must accept this and the kernel must produce the
    // same result as the non-aliased path because it reads x[i] before
    // writing dst[i] at the same index.
    GCtx gc;
    const int64_t n = 8;

    ggml_tensor * src  = ggml_new_tensor_1d(gc.ctx, GGML_TYPE_F32, n);
    ggml_tensor * node = ggml_silu(gc.ctx, src);

    std::vector<float> buf = {0.f, 1.f, -1.f, 2.f, -2.f, 0.5f, -0.5f, 4.f};
    src->data  = buf.data();
    node->data = buf.data();    // same buffer — in-place

    ggml_hpx_lowering lo;
    ggml_hpx_lowering_init(&lo);
    ASSERT_TRUE(ggml_hpx_lower_op(node, &lo));

    run_group(&lo, 1, nullptr, nullptr);

    const std::vector<float> orig = {0.f, 1.f, -1.f, 2.f, -2.f, 0.5f, -0.5f, 4.f};
    for (int64_t i = 0; i < n; ++i)
    {
        const float v   = orig[static_cast<std::size_t>(i)];
        const float ref = v / (1.f + std::exp(-v));
        EXPECT_NEAR(buf[static_cast<std::size_t>(i)], ref, 1e-6f) << "i=" << i;
    }
}

TEST(LowerOpMulF32, InPlaceAliasDstEqualsA)
{
    GCtx gc;
    const int64_t n = 8;

    ggml_tensor * a    = ggml_new_tensor_1d(gc.ctx, GGML_TYPE_F32, n);
    ggml_tensor * b    = ggml_new_tensor_1d(gc.ctx, GGML_TYPE_F32, n);
    ggml_tensor * node = ggml_mul(gc.ctx, a, b);

    std::vector<float> ad = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
    std::vector<float> bd = {8.f, 7.f, 6.f, 5.f, 4.f, 3.f, 2.f, 1.f};
    const std::vector<float> ad_orig = ad;

    a->data    = ad.data();
    b->data    = bd.data();
    node->data = ad.data();    // dst aliases a

    ggml_hpx_lowering lo;
    ggml_hpx_lowering_init(&lo);
    ASSERT_TRUE(ggml_hpx_lower_op(node, &lo));

    run_group(&lo, 1, nullptr, nullptr);

    for (int64_t i = 0; i < n; ++i)
    {
        const float ref = ad_orig[static_cast<std::size_t>(i)] * bd[static_cast<std::size_t>(i)];
        EXPECT_NEAR(ad[static_cast<std::size_t>(i)], ref, 1e-7f) << "i=" << i;
    }
}

// ---------------------------------------------------------------------------
// MUL_MAT Q4_K × F32 → F32 (rows == 1)
// ---------------------------------------------------------------------------

TEST(LowerOpMulMatQ4KF32, StructureAndExecution)
{
    // Minimal valid shape: one QK_K block wide, 2 output columns, 1 row.
    GCtx gc;
    constexpr int64_t rows     = 1;
    constexpr int64_t cols     = 256;   // one QK_K block
    constexpr int64_t out_cols = 2;

    ggml_tensor * w    = ggml_new_tensor_2d(gc.ctx, GGML_TYPE_Q4_K, cols, out_cols);
    ggml_tensor * x    = ggml_new_tensor_2d(gc.ctx, GGML_TYPE_F32,  cols, rows);
    ggml_tensor * node = ggml_mul_mat(gc.ctx, w, x);

    // Build F32 weights and quantize to Q4_K.
    const size_t q4k_row = ggml_row_size(GGML_TYPE_Q4_K, cols);
    std::vector<float> w_f32(static_cast<std::size_t>(out_cols * cols));
    for (int64_t c = 0; c < out_cols; ++c)
        for (int64_t i = 0; i < cols; ++i)
            w_f32[static_cast<std::size_t>(c * cols + i)] =
                static_cast<float>((c * 17 + i) % 9 - 4) * 0.1f;

    std::vector<char> w_q4k(static_cast<std::size_t>(out_cols) * q4k_row);
    for (int64_t c = 0; c < out_cols; ++c)
        quantize_row_q4_K(
            w_f32.data() + c * cols,
            w_q4k.data() + static_cast<std::size_t>(c) * q4k_row,
            cols);

    // Build F32 activation (one row).
    std::vector<float> x_f32(static_cast<std::size_t>(rows * cols));
    for (int64_t i = 0; i < rows * cols; ++i)
        x_f32[static_cast<std::size_t>(i)] = static_cast<float>(i % 5 + 1) * 0.2f;

    std::vector<float> y_hpx(static_cast<std::size_t>(rows * out_cols), 0.0f);

    w->data    = w_q4k.data();
    x->data    = x_f32.data();
    node->data = y_hpx.data();

    ggml_hpx_lowering lo;
    ggml_hpx_lowering_init(&lo);
    ASSERT_TRUE(ggml_hpx_lower_op(node, &lo));

    // ── structural checks ─────────────────────────────────────────────────
    EXPECT_EQ(lo.group.n_regions, 2);
    EXPECT_EQ(lo.group.n_deps,    1);
    EXPECT_EQ(lo.group.regions,   lo.regions);
    EXPECT_EQ(lo.group.deps,      lo.deps);

    // Region 0: serial quantize F32 → Q8_K
    EXPECT_EQ(lo.regions[0].kind,      GGML_HPX_CPU_REGION_KIND_REDUCTION);
    EXPECT_EQ(lo.regions[0].begin,     int64_t{0});
    EXPECT_EQ(lo.regions[0].end,       cols);
    EXPECT_EQ(lo.regions[0].ctx,       static_cast<void *>(lo.ctx_buf[0]));
    EXPECT_EQ(lo.regions[0].run_range, ggml_hpx_quantize_q8_k_f32_run_range);

    auto * qctx = reinterpret_cast<ggml_hpx_quantize_q8_k_f32_ctx *>(lo.ctx_buf[0]);
    EXPECT_EQ(qctx->cols,  cols);
    EXPECT_EQ(qctx->x,     x_f32.data());
    EXPECT_EQ(qctx->x_q8,  static_cast<void *>(lo.scratch));

    // Region 1: parallel Q4_K × Q8_K dot products
    EXPECT_EQ(lo.regions[1].kind,      GGML_HPX_CPU_REGION_KIND_MATMUL);
    EXPECT_EQ(lo.regions[1].begin,     int64_t{0});
    EXPECT_EQ(lo.regions[1].end,       out_cols);
    EXPECT_EQ(lo.regions[1].ctx,       static_cast<void *>(lo.ctx_buf[1]));
    EXPECT_EQ(lo.regions[1].run_range, ggml_hpx_mul_mat_q4_k_q8_k_run_range);

    auto * dctx = reinterpret_cast<ggml_hpx_mul_mat_q4_k_q8_k_ctx *>(lo.ctx_buf[1]);
    EXPECT_EQ(dctx->cols,          cols);
    EXPECT_EQ(dctx->out_cols,      out_cols);
    EXPECT_EQ(dctx->w_row_stride,  q4k_row);
    EXPECT_EQ(dctx->q8k_row_bytes, ggml_row_size(GGML_TYPE_Q8_K, cols));
    EXPECT_EQ(dctx->x_q8,          static_cast<void *>(lo.scratch));

    // Dep edge 0 → 1
    bool saw_dep = false;
    for (int i = 0; i < lo.group.n_deps; ++i)
        if (lo.deps[i].src == 0 && lo.deps[i].dst == 1) saw_dep = true;
    EXPECT_TRUE(saw_dep) << "missing dep edge 0→1";

    // ── execution + output check ──────────────────────────────────────────
    run_group(&lo, /*n_lanes=*/1, nullptr, nullptr);

    // Reference: same call sequence as the kernel — quantize x to Q8_K then
    // call ggml_vec_dot_q4_K_q8_K directly.  Both the kernel and the reference
    // call the same deterministic functions with the same inputs, so results
    // must be bitwise identical.
    const size_t q8k_row = ggml_row_size(GGML_TYPE_Q8_K, cols);
    std::vector<char> x_q8k(q8k_row);
    quantize_row_q8_K(x_f32.data(), x_q8k.data(), cols);

    for (int64_t c = 0; c < out_cols; ++c)
    {
        float ref = 0.0f;
        ggml_vec_dot_q4_K_q8_K(
            static_cast<int>(cols),
            &ref, 0,
            w_q4k.data() + static_cast<std::size_t>(c) * q4k_row, 0,
            x_q8k.data(), 0,
            1);
        EXPECT_FLOAT_EQ(y_hpx[static_cast<std::size_t>(c)], ref) << "col=" << c;
    }
}

TEST(LowerOpRejects, Q4KMultiRowReturnsFalse)
{
    // rows=2: Q4_K lower_op only supports rows==1 (initial implementation).
    GCtx gc;
    ggml_tensor * w    = ggml_new_tensor_2d(gc.ctx, GGML_TYPE_Q4_K, 256, 3);
    ggml_tensor * x    = ggml_new_tensor_2d(gc.ctx, GGML_TYPE_F32,  256, 2);
    ggml_tensor * node = ggml_mul_mat(gc.ctx, w, x);

    std::vector<char> w_buf(3 * ggml_row_size(GGML_TYPE_Q4_K, 256), 0);
    float x_dummy[256 * 2] = {};
    float y_dummy[3 * 2]   = {};

    w->data    = w_buf.data();
    x->data    = x_dummy;
    node->data = y_dummy;

    ggml_hpx_lowering lo;
    ggml_hpx_lowering_init(&lo);
    EXPECT_FALSE(ggml_hpx_lower_op(node, &lo));
    EXPECT_EQ(lo.group.n_regions, 0);
}

TEST(LowerOpMulF32, InPlaceAliasDstEqualsB)
{
    GCtx gc;
    const int64_t n = 8;

    ggml_tensor * a    = ggml_new_tensor_1d(gc.ctx, GGML_TYPE_F32, n);
    ggml_tensor * b    = ggml_new_tensor_1d(gc.ctx, GGML_TYPE_F32, n);
    ggml_tensor * node = ggml_mul(gc.ctx, a, b);

    std::vector<float> ad = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
    std::vector<float> bd = {8.f, 7.f, 6.f, 5.f, 4.f, 3.f, 2.f, 1.f};
    const std::vector<float> bd_orig = bd;

    a->data    = ad.data();
    b->data    = bd.data();
    node->data = bd.data();    // dst aliases b

    ggml_hpx_lowering lo;
    ggml_hpx_lowering_init(&lo);
    ASSERT_TRUE(ggml_hpx_lower_op(node, &lo));

    run_group(&lo, 1, nullptr, nullptr);

    for (int64_t i = 0; i < n; ++i)
    {
        const float ref = ad[static_cast<std::size_t>(i)] * bd_orig[static_cast<std::size_t>(i)];
        EXPECT_NEAR(bd[static_cast<std::size_t>(i)], ref, 1e-7f) << "i=" << i;
    }
}

} // namespace

#endif // GGML_HPX_REGION_DAG
