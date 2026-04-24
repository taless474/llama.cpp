// test_hpx_mlp_gate_up_compose.cpp
//
// Correctness tests for ggml_hpx_compose_mlp_gate_up_group.
//
// Builds a 4-op MLP gate/up subgraph via ggml op builders, composes into a
// single 4-region group with 3 cross-op dep edges, verifies structure, executes,
// and compares against a scalar reference.
//
// Topology:
//   gate     = ggml_mul_mat(W_gate, x)   region 0  MATMUL
//   up       = ggml_mul_mat(W_up,   x)   region 1  MATMUL
//   gate_act = ggml_silu(gate)           region 2  ELEMENTWISE
//   out      = ggml_mul(gate_act, up)    region 3  ELEMENTWISE
//
// Cross-op dep edges (src/dst): {0,2}, {1,3}, {2,3}

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <vector>

#include <hpx/future.hpp>

#include "ggml-hpx-compose.h"
#include "ggml-hpx-region-exec.h"
#include "ggml-hpx-runtime.h"
#include "ggml.h"

#ifdef GGML_HPX_REGION_DAG

namespace
{

// RAII ggml context, no_alloc=true: tensor structs only, no data pool.
struct GCtx
{
    ggml_context * ctx;

    explicit GCtx(size_t mem_bytes = 256 * 1024)
    {
        ggml_init_params p{};
        p.mem_size   = mem_bytes;
        p.mem_buffer = nullptr;
        p.no_alloc   = true;
        ctx = ggml_init(p);
    }
    ~GCtx() { if (ctx) ggml_free(ctx); }

    GCtx(const GCtx &)             = delete;
    GCtx & operator=(const GCtx &) = delete;
};

// Run a composed group (no per-region resources needed for MATMUL/ELEMENTWISE).
void run_composed(ggml_hpx_mlp_gate_up_group * grp)
{
    ggml_hpx_region_resources res{};
    res.n_lanes = 1;

    ggml_hpx_tpool_start();
    hpx::async([&]() {
        ggml_hpx_run_region_group(&grp->group, &res);
    }).get();
}

// Scalar SiLU.
static float silu(float v) { return v / (1.0f + std::exp(-v)); }

}    // namespace

// ---------------------------------------------------------------------------
// Structure tests
// ---------------------------------------------------------------------------

TEST(ComposeMlpGateUp, StructureIsCorrect)
{
    // hidden=4, out=3, batch=1.
    constexpr int64_t cols     = 4;
    constexpr int64_t out_cols = 3;
    constexpr int64_t rows     = 1;

    GCtx gc;

    // ggml_mul_mat(W, x): W is src[0] [cols × out_cols], x is src[1] [cols × rows].
    ggml_tensor * W_gate = ggml_new_tensor_2d(gc.ctx, GGML_TYPE_F32, cols, out_cols);
    ggml_tensor * W_up   = ggml_new_tensor_2d(gc.ctx, GGML_TYPE_F32, cols, out_cols);
    ggml_tensor * x      = ggml_new_tensor_2d(gc.ctx, GGML_TYPE_F32, cols, rows);

    // Build op graph. Intermediate output tensors are named explicitly for clarity.
    ggml_tensor * gate     = ggml_mul_mat(gc.ctx, W_gate, x);
    ggml_tensor * up       = ggml_mul_mat(gc.ctx, W_up,   x);
    ggml_tensor * gate_act = ggml_silu(gc.ctx, gate);
    ggml_tensor * node_out = ggml_mul(gc.ctx, gate_act, up);

    // Provide actual data buffers (ggml strides are computed correctly even in
    // no_alloc mode — only data pointers need to be set manually).
    std::vector<float> wg_buf(cols * out_cols, 1.0f);
    std::vector<float> wu_buf(cols * out_cols, 1.0f);
    std::vector<float> x_buf (cols * rows,     1.0f);
    std::vector<float> gate_buf  (out_cols * rows, 0.0f);
    std::vector<float> up_buf    (out_cols * rows, 0.0f);
    std::vector<float> gate_act_buf(out_cols * rows, 0.0f);
    std::vector<float> out_buf   (out_cols * rows, 0.0f);

    W_gate->data   = wg_buf.data();
    W_up->data     = wu_buf.data();
    x->data        = x_buf.data();
    gate->data     = gate_buf.data();
    up->data       = up_buf.data();
    gate_act->data = gate_act_buf.data();
    node_out->data = out_buf.data();

    ggml_hpx_mlp_gate_up_group grp;
    ggml_hpx_mlp_gate_up_group_init(&grp);

    bool ok = ggml_hpx_compose_mlp_gate_up_group(gate, up, gate_act, node_out, &grp);
    ASSERT_TRUE(ok);

    // Combined group shape.
    EXPECT_EQ(grp.group.n_regions, 4);
    EXPECT_EQ(grp.group.n_deps,    3);
    EXPECT_EQ(grp.group.regions,   grp.combined_regions);
    EXPECT_EQ(grp.group.deps,      grp.combined_deps);

    // Region kinds.
    EXPECT_EQ(grp.combined_regions[0].kind, GGML_HPX_CPU_REGION_KIND_MATMUL);
    EXPECT_EQ(grp.combined_regions[1].kind, GGML_HPX_CPU_REGION_KIND_MATMUL);
    EXPECT_EQ(grp.combined_regions[2].kind, GGML_HPX_CPU_REGION_KIND_ELEMENTWISE);
    EXPECT_EQ(grp.combined_regions[3].kind, GGML_HPX_CPU_REGION_KIND_ELEMENTWISE);

    // Cross-op dep edges (field names: src/dst).
    bool saw_0_2 = false, saw_1_3 = false, saw_2_3 = false;
    for (int i = 0; i < grp.group.n_deps; ++i)
    {
        const auto & d = grp.combined_deps[i];
        if (d.src == 0 && d.dst == 2) saw_0_2 = true;
        if (d.src == 1 && d.dst == 3) saw_1_3 = true;
        if (d.src == 2 && d.dst == 3) saw_2_3 = true;
    }
    EXPECT_TRUE(saw_0_2);
    EXPECT_TRUE(saw_1_3);
    EXPECT_TRUE(saw_2_3);

    // Each combined region's ctx must point into its corresponding lowering slot.
    for (int i = 0; i < 4; ++i)
        EXPECT_EQ(grp.combined_regions[i].ctx, (void *)grp.lowers[i].ctx_buf[0]);
}

// ---------------------------------------------------------------------------
// Execution correctness
// ---------------------------------------------------------------------------

TEST(ComposeMlpGateUp, ExecutionCorrectness)
{
    constexpr int64_t cols     = 4;
    constexpr int64_t out_cols = 3;
    constexpr int64_t rows     = 2;

    GCtx gc;

    ggml_tensor * W_gate   = ggml_new_tensor_2d(gc.ctx, GGML_TYPE_F32, cols, out_cols);
    ggml_tensor * W_up     = ggml_new_tensor_2d(gc.ctx, GGML_TYPE_F32, cols, out_cols);
    ggml_tensor * x        = ggml_new_tensor_2d(gc.ctx, GGML_TYPE_F32, cols, rows);
    ggml_tensor * gate     = ggml_mul_mat(gc.ctx, W_gate, x);
    ggml_tensor * up       = ggml_mul_mat(gc.ctx, W_up,   x);
    ggml_tensor * gate_act = ggml_silu(gc.ctx, gate);
    ggml_tensor * node_out = ggml_mul(gc.ctx, gate_act, up);

    // W_gate[j][k] = 0.1*(j*cols+k+1), W_up[j][k] = 0.2*(j*cols+k+1).
    std::vector<float> wg(cols * out_cols), wu(cols * out_cols);
    for (int j = 0; j < (int)out_cols; ++j)
        for (int k = 0; k < (int)cols; ++k)
        {
            wg[j * cols + k] = 0.1f * (float)(j * cols + k + 1);
            wu[j * cols + k] = 0.2f * (float)(j * cols + k + 1);
        }

    // x = all-ones.
    std::vector<float> xb(cols * rows, 1.0f);
    std::vector<float> gate_buf  (out_cols * rows, 0.0f);
    std::vector<float> up_buf    (out_cols * rows, 0.0f);
    std::vector<float> gate_act_buf(out_cols * rows, 0.0f);
    std::vector<float> out_buf   (out_cols * rows, 0.0f);

    W_gate->data   = wg.data();
    W_up->data     = wu.data();
    x->data        = xb.data();
    gate->data     = gate_buf.data();
    up->data       = up_buf.data();
    gate_act->data = gate_act_buf.data();
    node_out->data = out_buf.data();

    ggml_hpx_mlp_gate_up_group grp;
    ggml_hpx_mlp_gate_up_group_init(&grp);
    ASSERT_TRUE(ggml_hpx_compose_mlp_gate_up_group(gate, up, gate_act, node_out, &grp));

    run_composed(&grp);

    // Scalar reference. MUL_MAT: result[r][j] = sum_k x[r][k] * W[j][k].
    // Since x is all-ones: result[r][j] = sum_k W[j][k].
    for (int r = 0; r < (int)rows; ++r)
        for (int j = 0; j < (int)out_cols; ++j)
        {
            float sg = 0.0f, su = 0.0f;
            for (int k = 0; k < (int)cols; ++k)
            {
                sg += wg[j * cols + k];
                su += wu[j * cols + k];
            }
            float expected = silu(sg) * su;
            EXPECT_NEAR(out_buf[r * out_cols + j], expected, 1e-4f)
                << "r=" << r << " j=" << j;
        }
}

// ---------------------------------------------------------------------------
// Rejection: null pointers
// ---------------------------------------------------------------------------

TEST(ComposeMlpGateUp, RejectsNullNodes)
{
    GCtx gc;
    constexpr int64_t cols = 4, out_cols = 3, rows = 1;

    ggml_tensor * W = ggml_new_tensor_2d(gc.ctx, GGML_TYPE_F32, cols, out_cols);
    ggml_tensor * x = ggml_new_tensor_2d(gc.ctx, GGML_TYPE_F32, cols, rows);
    ggml_tensor * g = ggml_mul_mat(gc.ctx, W, x);
    ggml_tensor * u = ggml_mul_mat(gc.ctx, W, x);
    ggml_tensor * a = ggml_silu(gc.ctx, g);
    ggml_tensor * o = ggml_mul(gc.ctx, a, u);

    std::vector<float> wb(cols * out_cols, 1.0f), xb(cols * rows, 1.0f);
    std::vector<float> gb(out_cols * rows), ub(out_cols * rows), ab(out_cols * rows), ob(out_cols * rows);
    W->data = wb.data(); x->data = xb.data();
    g->data = gb.data(); u->data = ub.data();
    a->data = ab.data(); o->data = ob.data();

    auto try_compose = [&](ggml_tensor * ng, ggml_tensor * nu,
                            ggml_tensor * na, ggml_tensor * no) -> bool {
        ggml_hpx_mlp_gate_up_group grp;
        ggml_hpx_mlp_gate_up_group_init(&grp);
        return ggml_hpx_compose_mlp_gate_up_group(ng, nu, na, no, &grp);
    };

    EXPECT_FALSE(try_compose(nullptr, u, a, o));
    EXPECT_FALSE(try_compose(g, nullptr, a, o));
    EXPECT_FALSE(try_compose(g, u, nullptr, o));
    EXPECT_FALSE(try_compose(g, u, a, nullptr));
}

// ---------------------------------------------------------------------------
// Rejection: structurally wrong topology (non-null, but mis-wired)
// ---------------------------------------------------------------------------

TEST(ComposeMlpGateUp, RejectsMalformedTopology)
{
    GCtx gc;
    constexpr int64_t cols = 4, out_cols = 3, rows = 1;

    ggml_tensor * W_gate = ggml_new_tensor_2d(gc.ctx, GGML_TYPE_F32, cols, out_cols);
    ggml_tensor * W_up   = ggml_new_tensor_2d(gc.ctx, GGML_TYPE_F32, cols, out_cols);
    ggml_tensor * x      = ggml_new_tensor_2d(gc.ctx, GGML_TYPE_F32, cols, rows);

    ggml_tensor * gate = ggml_mul_mat(gc.ctx, W_gate, x);
    ggml_tensor * up   = ggml_mul_mat(gc.ctx, W_up,   x);
    // Correct path.
    ggml_tensor * gate_act = ggml_silu(gc.ctx, gate);
    ggml_tensor * node_out = ggml_mul(gc.ctx, gate_act, up);
    // Wrong paths for substitution.
    ggml_tensor * silu_of_up  = ggml_silu(gc.ctx, up);          // silu feeds from up, not gate
    ggml_tensor * mul_gate_up = ggml_mul(gc.ctx, gate, up);     // out feeds from gate, not gate_act
    ggml_tensor * mul_act_act = ggml_mul(gc.ctx, gate_act, gate_act); // out's src[1] is gate_act, not up

    std::vector<float> wg(cols * out_cols, 1.0f), wu(cols * out_cols, 1.0f);
    std::vector<float> xb(cols * rows, 1.0f);
    std::vector<float> gbuf(out_cols * rows), ubuf(out_cols * rows);
    std::vector<float> abuf(out_cols * rows), obuf(out_cols * rows);
    std::vector<float> sbuf(out_cols * rows), mbuf(out_cols * rows), m2buf(out_cols * rows);

    W_gate->data = wg.data(); W_up->data = wu.data(); x->data = xb.data();
    gate->data = gbuf.data(); up->data = ubuf.data();
    gate_act->data = abuf.data(); node_out->data = obuf.data();
    silu_of_up->data  = sbuf.data();
    mul_gate_up->data = mbuf.data();
    mul_act_act->data = m2buf.data();

    auto try_compose = [&](ggml_tensor * ng, ggml_tensor * nu,
                            ggml_tensor * na, ggml_tensor * no) -> bool {
        ggml_hpx_mlp_gate_up_group grp;
        ggml_hpx_mlp_gate_up_group_init(&grp);
        return ggml_hpx_compose_mlp_gate_up_group(ng, nu, na, no, &grp);
    };

    // gate_act = silu(up) instead of silu(gate) — src[0] mismatch.
    EXPECT_FALSE(try_compose(gate, up, silu_of_up, node_out));

    // out = mul(gate, up) instead of mul(gate_act, up) — src[0] mismatch.
    EXPECT_FALSE(try_compose(gate, up, gate_act, mul_gate_up));

    // out = mul(gate_act, gate_act) instead of mul(gate_act, up) — src[1] mismatch.
    EXPECT_FALSE(try_compose(gate, up, gate_act, mul_act_act));

    // Commuted operand order: mul(up, gate_act) is not the canonical pattern.
    ggml_tensor * mul_up_act = ggml_mul(gc.ctx, up, gate_act);
    std::vector<float> m3buf(out_cols * rows);
    mul_up_act->data = m3buf.data();
    EXPECT_FALSE(try_compose(gate, up, gate_act, mul_up_act));
}

// ---------------------------------------------------------------------------
// Rejection: wrong op kinds (non-null, topology-consistent, but wrong op type)
// ---------------------------------------------------------------------------

TEST(ComposeMlpGateUp, RejectsWrongOpKinds)
{
    GCtx gc;
    constexpr int64_t n = 3, cols = 4, out_cols = 3, rows = 1;

    // Build nodes with the right shape but wrong ops for gate/up positions.
    ggml_tensor * W  = ggml_new_tensor_2d(gc.ctx, GGML_TYPE_F32, cols, out_cols);
    ggml_tensor * x  = ggml_new_tensor_2d(gc.ctx, GGML_TYPE_F32, cols, rows);
    ggml_tensor * a1 = ggml_new_tensor_1d(gc.ctx, GGML_TYPE_F32, n);
    ggml_tensor * a2 = ggml_new_tensor_1d(gc.ctx, GGML_TYPE_F32, n);

    // Correct nodes used as baselines.
    ggml_tensor * gate     = ggml_mul_mat(gc.ctx, W, x);
    ggml_tensor * up       = ggml_mul_mat(gc.ctx, W, x);
    ggml_tensor * gate_act = ggml_silu(gc.ctx, gate);
    ggml_tensor * node_out = ggml_mul(gc.ctx, gate_act, up);

    // A relu node: GGML_OP_UNARY / GGML_UNARY_OP_RELU — not SiLU.
    ggml_tensor * relu_of_gate = ggml_relu(gc.ctx, gate);

    // An add node to substitute in gate/up positions: GGML_OP_ADD, not MUL_MAT.
    ggml_tensor * add_node = ggml_add(gc.ctx, a1, a2);

    std::vector<float> wb(cols * out_cols, 1.0f), xb(cols * rows, 1.0f);
    std::vector<float> gb(out_cols * rows), ub(out_cols * rows);
    std::vector<float> ab(out_cols * rows), ob(out_cols * rows);
    std::vector<float> rb(out_cols * rows);
    std::vector<float> a1b(n, 1.0f), a2b(n, 1.0f), addb(n);

    W->data = wb.data(); x->data = xb.data();
    gate->data = gb.data(); up->data = ub.data();
    gate_act->data = ab.data(); node_out->data = ob.data();
    relu_of_gate->data = rb.data();
    a1->data = a1b.data(); a2->data = a2b.data(); add_node->data = addb.data();

    auto try_compose = [&](ggml_tensor * ng, ggml_tensor * nu,
                            ggml_tensor * na, ggml_tensor * no) -> bool {
        ggml_hpx_mlp_gate_up_group grp;
        ggml_hpx_mlp_gate_up_group_init(&grp);
        return ggml_hpx_compose_mlp_gate_up_group(ng, nu, na, no, &grp);
    };

    // gate position is not MUL_MAT.
    EXPECT_FALSE(try_compose(add_node, up, gate_act, node_out));

    // up position is not MUL_MAT.
    EXPECT_FALSE(try_compose(gate, add_node, gate_act, node_out));

    // gate_act is UNARY/RELU, not UNARY/SILU.
    // Rewire relu_of_gate->src[0] to gate so the topology check passes.
    // The op-kind check fires before the topology check for gate_act only
    // if we check op before src[] — which we do. Either way it must reject.
    EXPECT_FALSE(try_compose(gate, up, relu_of_gate, node_out));
}

#endif    // GGML_HPX_REGION_DAG

int main(int argc, char ** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
