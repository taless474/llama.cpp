// test_hpx_packet_mlp_glu_f32.cpp
//
// Packet correctness test for GGML_HPX_PACKET_SUBLAYER_MLP_GLU_F32.
//
// Builds the 3-op ggml subgraph via ggml op builders (two MUL_MAT + one
// ggml_swiglu_split), composes it into a 3-region group, compiles a frozen
// packet, binds one invocation, runs it, and checks the output against a
// scalar reference.
//
// Parallels tests/hpx/test_hpx_packet_mlp_gate_up_f32.cpp; the math is
// identical but the fused GLU form drops the intermediate gate_act buffer
// and collapses the SiLU + elementwise MUL into a single kernel region.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <new>
#include <vector>

#include <hpx/future.hpp>

#include "ggml.h"
#include "ggml-hpx-compose.h"
#include "ggml-hpx-packet.h"
#include "ggml-hpx-runtime.h"

#ifdef GGML_HPX_REGION_DAG

namespace
{

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

static float silu_ref(float v)
{
    return v / (1.0f + std::exp(-v));
}

}    // namespace

TEST(PacketMlpGluF32, CompileBindRunMatchesScalarReference)
{
    constexpr int64_t cols     = 4;
    constexpr int64_t out_cols = 3;
    constexpr int64_t rows     = 2;

    GCtx gc;

    // ── Build ggml subgraph ───────────────────────────────────────────────
    ggml_tensor * W_gate = ggml_new_tensor_2d(gc.ctx, GGML_TYPE_F32, cols, out_cols);
    ggml_tensor * W_up   = ggml_new_tensor_2d(gc.ctx, GGML_TYPE_F32, cols, out_cols);
    ggml_tensor * x      = ggml_new_tensor_2d(gc.ctx, GGML_TYPE_F32, cols, rows);
    ggml_tensor * gate   = ggml_mul_mat(gc.ctx, W_gate, x);
    ggml_tensor * up     = ggml_mul_mat(gc.ctx, W_up,   x);
    ggml_tensor * out    = ggml_swiglu_split(gc.ctx, gate, up);

    // ── Buffers ───────────────────────────────────────────────────────────
    std::vector<float> wg(cols * out_cols);
    std::vector<float> wu(cols * out_cols);
    for (int j = 0; j < (int)out_cols; ++j)
        for (int k = 0; k < (int)cols; ++k)
        {
            wg[j * cols + k] = 0.1f * (float)(j * cols + k + 1);
            wu[j * cols + k] = 0.2f * (float)(j * cols + k + 1);
        }

    std::vector<float> xb = {
        1.f, 1.f, 1.f, 1.f,
        1.f, 2.f, 3.f, 4.f,
    };

    std::vector<float> gate_buf(rows * out_cols, 0.0f);
    std::vector<float> up_buf  (rows * out_cols, 0.0f);
    std::vector<float> out_buf (rows * out_cols, 0.0f);

    W_gate->data = wg.data();
    W_up->data   = wu.data();
    x->data      = xb.data();
    gate->data   = gate_buf.data();
    up->data     = up_buf.data();
    out->data    = out_buf.data();

    // ── Compose ───────────────────────────────────────────────────────────
    ggml_hpx_mlp_glu_group grp;
    ggml_hpx_mlp_glu_group_init(&grp);
    ASSERT_TRUE(ggml_hpx_compose_mlp_glu_group(gate, up, out, &grp));

    // ── Runtime ───────────────────────────────────────────────────────────
    ggml_hpx_tpool_start();
    ggml_hpx_packet_runtime * rt = ggml_hpx_packet_runtime_create(/*n_decode_threads=*/1);
    ASSERT_NE(rt, nullptr);

    // ── Plan key ─────────────────────────────────────────────────────────
    ggml_hpx_packet_plan_key key{};
    key.sublayer       = GGML_HPX_PACKET_SUBLAYER_MLP_GLU_F32;
    key.team           = GGML_HPX_PACKET_TEAM_DECODE;
    key.n_lanes        = 1;
    key.dtype          = GGML_TYPE_F32;
    key.seq_regime     = 0;
    key.policy_version = 1;
    key.shape[0]       = out_cols;
    key.shape[1]       = cols;
    key.shape[2]       = rows;
    key.shape[3]       = 0;
    key.extra          = 0;

    // ── Compile ───────────────────────────────────────────────────────────
    const char * err = nullptr;
    ggml_hpx_frozen_packet * packet = ggml_hpx_compile_packet(&grp.group, &key, &err);
    ASSERT_NE(packet, nullptr) << (err ? err : "compile failed");

    // ── Frame ─────────────────────────────────────────────────────────────
    const size_t frame_sz = ggml_hpx_packet_frame_size(packet);
    const size_t frame_al = ggml_hpx_packet_frame_align(packet);
    void * raw = ::operator new(frame_sz, std::align_val_t{frame_al});
    auto * frame = static_cast<ggml_hpx_packet_frame *>(raw);
    ggml_hpx_packet_frame_init(frame, packet);

    // ── Bind ─────────────────────────────────────────────────────────────
    ggml_hpx_mlp_glu_binding bind{};
    bind.w_gate = wg.data();
    bind.w_up   = wu.data();
    bind.x      = xb.data();
    bind.gate   = gate_buf.data();
    bind.up     = up_buf.data();
    bind.out    = out_buf.data();
    ggml_hpx_bind_mlp_glu_packet(frame, &bind);

    // ── Resources ─────────────────────────────────────────────────────────
    // MLP_GLU has no REDUCTION regions; all resource requirements are 0
    // except n_lanes.
    ggml_hpx_packet_resource_requirements req{};
    ggml_hpx_packet_get_resource_requirements(packet, &req);

    std::vector<std::vector<unsigned char>> scratch_storage(req.n_lanes);
    std::vector<void *> lane_ptrs(req.n_lanes, nullptr);
    for (uint32_t i = 0; i < req.n_lanes; ++i)
    {
        scratch_storage[i].resize(req.lane_scratch_bytes_per_lane);
        lane_ptrs[i] = scratch_storage[i].empty() ? nullptr : scratch_storage[i].data();
    }
    std::vector<unsigned char> reduce_storage(req.reduction_buffer_bytes);
    std::vector<unsigned char> shared_storage(req.shared_scratch_bytes);

    ggml_hpx_region_resources resources{};
    resources.n_lanes          = static_cast<int>(req.n_lanes);
    resources.lane_scratch     = lane_ptrs.empty()    ? nullptr : lane_ptrs.data();
    resources.reduction_buffer = reduce_storage.empty() ? nullptr : reduce_storage.data();
    resources.shared_scratch   = shared_storage.empty() ? nullptr : shared_storage.data();

    // ── Run ───────────────────────────────────────────────────────────────
    hpx::async([&]() {
        ggml_hpx_run_frozen_packet(rt, packet, frame, &resources);
    }).get();

    // ── Scalar reference ──────────────────────────────────────────────────
    // gate[r][j] = sum_k x[r][k] * W_gate[j][k]
    // up  [r][j] = sum_k x[r][k] * W_up  [j][k]
    // out [r][j] = silu(gate[r][j]) * up[r][j]
    for (int r = 0; r < (int)rows; ++r)
        for (int j = 0; j < (int)out_cols; ++j)
        {
            float g_ref = 0.0f, u_ref = 0.0f;
            for (int k = 0; k < (int)cols; ++k)
            {
                g_ref += xb[r * cols + k] * wg[j * cols + k];
                u_ref += xb[r * cols + k] * wu[j * cols + k];
            }
            const float expected = silu_ref(g_ref) * u_ref;
            EXPECT_NEAR(out_buf[r * out_cols + j], expected, 1e-4f)
                << "r=" << r << " j=" << j;
        }

    // ── Cleanup ───────────────────────────────────────────────────────────
    ::operator delete(raw, std::align_val_t{frame_al});
    ggml_hpx_free_packet(packet);
    ggml_hpx_packet_runtime_destroy(rt);
}

// ── Hardening: compile rejection for wrong key.shape[] ───────────────────────
//
// Tests that compile returns nullptr (with a non-null error string) when:
//   (a) shape[0] == 0 (zero out_cols is invalid)
//   (b) shape dimensions are inconsistent with the composed group's work ranges
//       (e.g. shape[2] == rows+1 while the group was built for rows)
TEST(PacketMlpGluF32, CompileRejectsWrongKeyShape)
{
    constexpr int64_t cols     = 4;
    constexpr int64_t out_cols = 3;
    constexpr int64_t rows     = 2;

    GCtx gc;

    ggml_tensor * W_gate = ggml_new_tensor_2d(gc.ctx, GGML_TYPE_F32, cols, out_cols);
    ggml_tensor * W_up   = ggml_new_tensor_2d(gc.ctx, GGML_TYPE_F32, cols, out_cols);
    ggml_tensor * x      = ggml_new_tensor_2d(gc.ctx, GGML_TYPE_F32, cols, rows);
    ggml_tensor * gate   = ggml_mul_mat(gc.ctx, W_gate, x);
    ggml_tensor * up     = ggml_mul_mat(gc.ctx, W_up,   x);
    ggml_tensor * out    = ggml_swiglu_split(gc.ctx, gate, up);

    std::vector<float> wg(cols * out_cols, 1.f);
    std::vector<float> wu(cols * out_cols, 1.f);
    std::vector<float> xb(cols * rows,     1.f);
    std::vector<float> gate_buf(rows * out_cols, 0.f);
    std::vector<float> up_buf  (rows * out_cols, 0.f);
    std::vector<float> out_buf (rows * out_cols, 0.f);

    W_gate->data = wg.data();
    W_up->data   = wu.data();
    x->data      = xb.data();
    gate->data   = gate_buf.data();
    up->data     = up_buf.data();
    out->data    = out_buf.data();

    ggml_hpx_mlp_glu_group grp;
    ggml_hpx_mlp_glu_group_init(&grp);
    ASSERT_TRUE(ggml_hpx_compose_mlp_glu_group(gate, up, out, &grp));

    // (a) shape[0] == 0 — out_cols must be > 0
    {
        ggml_hpx_packet_plan_key key{};
        key.sublayer       = GGML_HPX_PACKET_SUBLAYER_MLP_GLU_F32;
        key.team           = GGML_HPX_PACKET_TEAM_DECODE;
        key.n_lanes        = 1;
        key.dtype          = GGML_TYPE_F32;
        key.policy_version = 1;
        key.shape[0]       = 0;          // invalid: zero out_cols
        key.shape[1]       = cols;
        key.shape[2]       = rows;
        key.shape[3]       = 0;

        const char * err    = nullptr;
        ggml_hpx_frozen_packet * p = ggml_hpx_compile_packet(&grp.group, &key, &err);
        EXPECT_EQ(p, nullptr)   << "compile must reject shape[0]==0";
        EXPECT_NE(err, nullptr) << "must set out_err on rejection";
    }

    // (b) shape[2] wrong — rows+1 disagrees with the group's elementwise work range
    {
        ggml_hpx_packet_plan_key key{};
        key.sublayer       = GGML_HPX_PACKET_SUBLAYER_MLP_GLU_F32;
        key.team           = GGML_HPX_PACKET_TEAM_DECODE;
        key.n_lanes        = 1;
        key.dtype          = GGML_TYPE_F32;
        key.policy_version = 1;
        key.shape[0]       = out_cols;
        key.shape[1]       = cols;
        key.shape[2]       = rows + 1;   // disagrees with the composed group
        key.shape[3]       = 0;

        const char * err    = nullptr;
        ggml_hpx_frozen_packet * p = ggml_hpx_compile_packet(&grp.group, &key, &err);
        EXPECT_EQ(p, nullptr)   << "compile must reject mismatched rows";
        EXPECT_NE(err, nullptr) << "must set out_err on rejection";
    }
}

// ── Hardening: compile rejection for malformed group ─────────────────────────
//
// Tests that compile returns nullptr when the group itself is structurally
// wrong, independent of the key:
//   (a) empty group (n_regions == 0) with a valid key
//   (b) valid MLP_GLU group but key.sublayer is MLP_GATE_UP_F32 (sibling
//       sublayer mismatch — the most adversarial case because shape[] is
//       identical in convention)
TEST(PacketMlpGluF32, CompileRejectsMalformedGroup)
{
    constexpr int64_t cols     = 4;
    constexpr int64_t out_cols = 3;
    constexpr int64_t rows     = 2;

    GCtx gc;

    ggml_tensor * W_gate = ggml_new_tensor_2d(gc.ctx, GGML_TYPE_F32, cols, out_cols);
    ggml_tensor * W_up   = ggml_new_tensor_2d(gc.ctx, GGML_TYPE_F32, cols, out_cols);
    ggml_tensor * x      = ggml_new_tensor_2d(gc.ctx, GGML_TYPE_F32, cols, rows);
    ggml_tensor * gate   = ggml_mul_mat(gc.ctx, W_gate, x);
    ggml_tensor * up     = ggml_mul_mat(gc.ctx, W_up,   x);
    ggml_tensor * out    = ggml_swiglu_split(gc.ctx, gate, up);

    std::vector<float> wg(cols * out_cols, 1.f);
    std::vector<float> wu(cols * out_cols, 1.f);
    std::vector<float> xb(cols * rows,     1.f);
    std::vector<float> gate_buf(rows * out_cols, 0.f);
    std::vector<float> up_buf  (rows * out_cols, 0.f);
    std::vector<float> out_buf (rows * out_cols, 0.f);

    W_gate->data = wg.data();
    W_up->data   = wu.data();
    x->data      = xb.data();
    gate->data   = gate_buf.data();
    up->data     = up_buf.data();
    out->data    = out_buf.data();

    ggml_hpx_mlp_glu_group valid_grp;
    ggml_hpx_mlp_glu_group_init(&valid_grp);
    ASSERT_TRUE(ggml_hpx_compose_mlp_glu_group(gate, up, out, &valid_grp));

    // (a) empty group — n_regions == 0 fails the generic structural validator
    {
        ggml_hpx_cpu_region_group empty{};   // zero-init: n_regions == 0

        ggml_hpx_packet_plan_key key{};
        key.sublayer       = GGML_HPX_PACKET_SUBLAYER_MLP_GLU_F32;
        key.team           = GGML_HPX_PACKET_TEAM_DECODE;
        key.n_lanes        = 1;
        key.dtype          = GGML_TYPE_F32;
        key.policy_version = 1;
        key.shape[0]       = out_cols;
        key.shape[1]       = cols;
        key.shape[2]       = rows;
        key.shape[3]       = 0;

        const char * err    = nullptr;
        ggml_hpx_frozen_packet * p = ggml_hpx_compile_packet(&empty, &key, &err);
        EXPECT_EQ(p, nullptr)   << "compile must reject empty group";
        EXPECT_NE(err, nullptr) << "must set out_err on rejection";
    }

    // (b) valid MLP_GLU group but key says MLP_GATE_UP_F32 — sibling sublayer
    // mismatch. shape[] convention is identical between the two sublayers so
    // this specifically exercises the per-sublayer validator dispatch: the
    // MLP_GATE_UP validator rejects the group because it has 3 regions, not 4.
    {
        ggml_hpx_packet_plan_key key{};
        key.sublayer       = GGML_HPX_PACKET_SUBLAYER_MLP_GATE_UP_F32;  // wrong sublayer
        key.team           = GGML_HPX_PACKET_TEAM_DECODE;
        key.n_lanes        = 1;
        key.dtype          = GGML_TYPE_F32;
        key.policy_version = 1;
        key.shape[0]       = out_cols;
        key.shape[1]       = cols;
        key.shape[2]       = rows;
        key.shape[3]       = 0;

        const char * err    = nullptr;
        ggml_hpx_frozen_packet * p =
            ggml_hpx_compile_packet(&valid_grp.group, &key, &err);
        EXPECT_EQ(p, nullptr)   << "compile must reject MLP_GLU group under MLP_GATE_UP key";
        EXPECT_NE(err, nullptr) << "must set out_err on rejection";
    }
}

#endif    // GGML_HPX_REGION_DAG

int main(int argc, char ** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
