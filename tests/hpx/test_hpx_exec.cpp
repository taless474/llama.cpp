#include <gtest/gtest.h>

#include "ggml-hpx-adapter.h"
#include "ggml-hpx-exec.h"
#include "ggml-hpx-region.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace {

// RAII owner for a decode backend bundle.
// Initialises cpu via ggml_backend_cpu_init(); blas stays null (CPU-only).
// Destructor calls ggml_backend_free so each test body stays leak-free.
struct cpu_backends_guard
{
    ggml_hpx_decode_backends b{};

    explicit cpu_backends_guard()
    {
        b.cpu = ggml_backend_cpu_init();
    }

    ~cpu_backends_guard()
    {
        ggml_backend_free(b.cpu);
    }

    cpu_backends_guard(cpu_backends_guard const&)            = delete;
    cpu_backends_guard& operator=(cpu_backends_guard const&) = delete;
};

struct ggml_context_deleter
{
    void operator()(ggml_context* ctx) const noexcept
    {
        if (ctx != nullptr)
        {
            ggml_free(ctx);
        }
    }
};

struct empty_graph_owner
{
    std::vector<uint8_t> storage;
    std::unique_ptr<ggml_context, ggml_context_deleter> ctx;
    ggml_cgraph* gf = nullptr;
};

empty_graph_owner make_empty_graph()
{
    empty_graph_owner out{};
    out.storage.resize(1 << 14);

    ggml_init_params params{};
    params.mem_size   = out.storage.size();
    params.mem_buffer = out.storage.data();
    params.no_alloc   = true;

    out.ctx.reset(ggml_init(params));
    if (!out.ctx)
    {
        return out;
    }

    out.gf = ggml_new_graph_custom(out.ctx.get(), 8, false);
    return out;
}

ggml_hpx_exec_params make_exec_params()
{
    ggml_hpx_exec_params params{};
    params.runtime.n_decode_threads      = 2;
    params.runtime.n_prefill_threads     = 2;
    params.runtime.initial_scratch_bytes = 0;
    params.policy_version                = 1;
    return params;
}

}    // namespace

TEST(HpxExec, CreateDestroySucceeds)
{
    auto const params = make_exec_params();
    auto* const exec  = ggml_hpx_exec_create(params);
    ASSERT_NE(exec, nullptr);
    ggml_hpx_exec_destroy(exec);
}

TEST(HpxExec, AbortSetsFlagAndRunReturnsAborted)
{
    auto const params = make_exec_params();
    auto* const exec  = ggml_hpx_exec_create(params);
    ASSERT_NE(exec, nullptr);

    auto graph = make_empty_graph();
    ASSERT_NE(graph.gf, nullptr);

    cpu_backends_guard bg;
    ggml_hpx_exec_abort(exec);

    ggml_hpx_exec_status const status =
        ggml_hpx_exec_run_decode(exec, graph.gf, bg.b);
    EXPECT_EQ(status, ggml_hpx_exec_status::aborted);

    ggml_hpx_exec_destroy(exec);
}

TEST(HpxExec, DecodeRunOnEmptyGraphReturnsOk)
{
    auto const params = make_exec_params();
    auto* const exec  = ggml_hpx_exec_create(params);
    ASSERT_NE(exec, nullptr);

    auto graph = make_empty_graph();
    ASSERT_NE(graph.gf, nullptr);

    cpu_backends_guard bg;
    ggml_hpx_exec_status const status =
        ggml_hpx_exec_run_decode(exec, graph.gf, bg.b);
    EXPECT_EQ(status, ggml_hpx_exec_status::ok);

    ggml_hpx_exec_destroy(exec);
}

// Two identical runs on the same graph both return ok.
// Verifies that the executor does not get stuck in an error state
// between runs (no assertion about cache hit/miss, which has no
// public observable API at this level).
TEST(HpxExec, DecodeRunTwiceOnEmptyGraphBothReturnOk)
{
    auto const params = make_exec_params();
    auto* const exec  = ggml_hpx_exec_create(params);
    ASSERT_NE(exec, nullptr);

    auto graph = make_empty_graph();
    ASSERT_NE(graph.gf, nullptr);

    cpu_backends_guard bg;
    EXPECT_EQ(ggml_hpx_exec_run_decode(exec, graph.gf, bg.b),
        ggml_hpx_exec_status::ok);
    EXPECT_EQ(ggml_hpx_exec_run_decode(exec, graph.gf, bg.b),
        ggml_hpx_exec_status::ok);

    ggml_hpx_exec_destroy(exec);
}

TEST(HpxExec, PrefillRunOnEmptyGraphReturnsOk)
{
    auto const params = make_exec_params();
    auto* const exec  = ggml_hpx_exec_create(params);
    ASSERT_NE(exec, nullptr);

    auto graph = make_empty_graph();
    ASSERT_NE(graph.gf, nullptr);

    ggml_hpx_exec_status const status =
        ggml_hpx_exec_run_prefill(exec, graph.gf, /*sched=*/nullptr);
    EXPECT_EQ(status, ggml_hpx_exec_status::ok);

    ggml_hpx_exec_destroy(exec);
}

// Abort is requested before run1, which must observe it and return aborted.
// run must reset the abort token internally, so run2 on the same graph
// returns ok without any explicit reset call from the caller.
TEST(HpxExec, ResetAbortAllowsSubsequentRunToSucceed)
{
    auto const params = make_exec_params();
    auto* const exec  = ggml_hpx_exec_create(params);
    ASSERT_NE(exec, nullptr);

    auto graph = make_empty_graph();
    ASSERT_NE(graph.gf, nullptr);

    cpu_backends_guard bg;
    ggml_hpx_exec_abort(exec);

    EXPECT_EQ(ggml_hpx_exec_run_decode(exec, graph.gf, bg.b),
        ggml_hpx_exec_status::aborted);

    EXPECT_EQ(ggml_hpx_exec_run_decode(exec, graph.gf, bg.b),
        ggml_hpx_exec_status::ok);

    ggml_hpx_exec_destroy(exec);
}

// CPU-only integration test: mul_mat + neg on the decode path.
// All tensors land on a host buffer type, so the adapter produces one
// cpu_contiguous region covering the whole graph.  The HPX exec result
// is compared against a plain ggml_backend_graph_compute baseline.
TEST(HpxExec, DecodeRunMatchesBaselineForCpuOnlyMulMatNeg)
{
    ggml_backend_t cpu = ggml_backend_init_by_type(
        GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ASSERT_NE(cpu, nullptr);

    // Lambda captures cpu so gallocr can use its buffer type.
    auto make_graph = [cpu]()
    {
        struct fixture
        {
            ggml_context* ctx    = nullptr;
            ggml_cgraph*  gf     = nullptr;
            ggml_tensor*  w      = nullptr;
            ggml_tensor*  x      = nullptr;
            ggml_tensor*  out    = nullptr;
            ggml_gallocr_t gallocr = nullptr;
        } f{};

        ggml_init_params params = {};
        params.mem_size   = 2 * 1024 * 1024;
        params.mem_buffer = nullptr;
        params.no_alloc   = true;

        f.ctx = ggml_init(params);
        EXPECT_NE(f.ctx, nullptr);

        // w: [3 x 2], x: [3 x 4]
        f.w = ggml_new_tensor_2d(f.ctx, GGML_TYPE_F32, 3, 2);
        f.x = ggml_new_tensor_2d(f.ctx, GGML_TYPE_F32, 3, 4);
        EXPECT_NE(f.w, nullptr);
        EXPECT_NE(f.x, nullptr);

        ggml_set_input(f.w);
        ggml_set_input(f.x);

        ggml_tensor* mm = ggml_mul_mat(f.ctx, f.w, f.x);
        EXPECT_NE(mm, nullptr);

        f.out = ggml_neg(f.ctx, mm);
        EXPECT_NE(f.out, nullptr);
        ggml_set_output(f.out);

        f.gf = ggml_new_graph(f.ctx);
        EXPECT_NE(f.gf, nullptr);
        ggml_build_forward_expand(f.gf, f.out);

        ggml_backend_buffer_type_t buft =
            ggml_backend_get_default_buffer_type(cpu);
        EXPECT_NE(buft, nullptr);

        f.gallocr = ggml_gallocr_new(buft);
        EXPECT_NE(f.gallocr, nullptr);

        EXPECT_TRUE(ggml_gallocr_reserve(f.gallocr, f.gf));
        EXPECT_TRUE(ggml_gallocr_alloc_graph(f.gallocr, f.gf));

        return f;
    };

    auto free_graph = [](auto& f)
    {
        if (f.gallocr != nullptr)
        {
            ggml_gallocr_free(f.gallocr);
            f.gallocr = nullptr;
        }
        if (f.ctx != nullptr)
        {
            ggml_free(f.ctx);
            f.ctx = nullptr;
        }
    };

    auto baseline   = make_graph();
    auto under_test = make_graph();

    std::vector<float> const w_data = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
    };
    std::vector<float> const x_data = {
         0.5f, -1.0f,  2.0f,
         1.5f,  0.0f, -0.5f,
         2.5f,  3.0f,  1.0f,
        -2.0f,  1.0f,  0.25f,
    };

    ggml_backend_tensor_set(
        baseline.w,   w_data.data(), 0, w_data.size() * sizeof(float));
    ggml_backend_tensor_set(
        baseline.x,   x_data.data(), 0, x_data.size() * sizeof(float));
    ggml_backend_tensor_set(
        under_test.w, w_data.data(), 0, w_data.size() * sizeof(float));
    ggml_backend_tensor_set(
        under_test.x, x_data.data(), 0, x_data.size() * sizeof(float));

    // CPU-only: all tensors are on host buffers so the adapter collapses
    // the whole graph into one cpu_contiguous region.
    {
        constexpr uint32_t k_policy = 1;
        ggml_hpx_adapter_result ar =
            ggml_hpx_adapt_decode(under_test.gf, k_policy);

        ASSERT_EQ(ar.topo.regions.size(), 1u);
        EXPECT_EQ(ar.topo.regions.front().type,
            ggml_hpx_region_type::cpu_contiguous);
        EXPECT_EQ(ar.topo.regions.front().node_begin, 0u);
        EXPECT_EQ(ar.topo.regions.front().node_end,
            static_cast<uint32_t>(ggml_graph_n_nodes(under_test.gf)));
    }

    // Baseline: plain ggml compute on the whole graph.
    ASSERT_EQ(ggml_backend_graph_compute(cpu, baseline.gf),
        GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(cpu);

    std::vector<float> expected(
        ggml_nbytes(baseline.out) / sizeof(float));
    ggml_backend_tensor_get(
        baseline.out, expected.data(), 0,
        expected.size() * sizeof(float));

    // HPX exec path: same backend, through adapt→plan→cache→dispatch.
    ggml_hpx_exec_params exec_params = {};
    exec_params.runtime.n_decode_threads      = 1;
    exec_params.runtime.n_prefill_threads     = 4;
    exec_params.runtime.initial_scratch_bytes = 0;
    exec_params.policy_version                = 1;

    ggml_hpx_exec* exec = ggml_hpx_exec_create(exec_params);
    ASSERT_NE(exec, nullptr);

    ggml_hpx_decode_backends backends = {};
    backends.cpu  = cpu;
    backends.blas = nullptr;    // exercise cpu fallback for any blas region

    EXPECT_EQ(
        ggml_hpx_exec_run_decode(exec, under_test.gf, backends),
        ggml_hpx_exec_status::ok);
    ggml_backend_synchronize(cpu);

    std::vector<float> got(ggml_nbytes(under_test.out) / sizeof(float));
    ggml_backend_tensor_get(
        under_test.out, got.data(), 0, got.size() * sizeof(float));

    ASSERT_EQ(got.size(), expected.size());
    for (size_t i = 0; i < got.size(); ++i)
    {
        EXPECT_FLOAT_EQ(got[i], expected[i]) << "mismatch at index " << i;
    }

    ggml_hpx_exec_destroy(exec);
    free_graph(under_test);
    free_graph(baseline);
    ggml_backend_free(cpu);
}

// Verify that decode actually runs ggml ops, not just the orchestration path.
// Graph: b = neg(a) where a = [1, -2, 3, -4].
// Expected: b = [-1, 2, -3, 4].
TEST(HpxExec, DecodeRunExecutesNegateOp)
{
    auto const params = make_exec_params();
    auto* const exec  = ggml_hpx_exec_create(params);
    ASSERT_NE(exec, nullptr);

    cpu_backends_guard bg;

    // Build a tiny ggml context. no_alloc=true: tensor storage is
    // provided by the gallocr below, not by the ggml context itself.
    std::vector<uint8_t> storage(1 << 16);
    ggml_init_params init{};
    init.mem_size   = storage.size();
    init.mem_buffer = storage.data();
    init.no_alloc   = true;
    std::unique_ptr<ggml_context, ggml_context_deleter> ctx(
        ggml_init(init));
    ASSERT_NE(ctx.get(), nullptr);

    ggml_tensor* a  = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 4);
    ggml_tensor* b  = ggml_neg(ctx.get(), a);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx.get(), 8, false);
    ggml_build_forward_expand(gf, b);

    // Allocate compute buffers through the CPU backend.
    ggml_gallocr_t galloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(bg.b.cpu));
    ASSERT_NE(galloc, nullptr);
    ASSERT_TRUE(ggml_gallocr_alloc_graph(galloc, gf));

    // Write input.
    float const in[4] = {1.0f, -2.0f, 3.0f, -4.0f};
    ggml_backend_tensor_set(a, in, 0, sizeof(in));

    // Execute the decode path.
    EXPECT_EQ(ggml_hpx_exec_run_decode(exec, gf, bg.b),
        ggml_hpx_exec_status::ok);

    // Verify output: neg flips sign of each element.
    float out[4] = {};
    ggml_backend_tensor_get(b, out, 0, sizeof(out));
    EXPECT_NEAR(out[0], -1.0f, 1e-6f);
    EXPECT_NEAR(out[1],  2.0f, 1e-6f);
    EXPECT_NEAR(out[2], -3.0f, 1e-6f);
    EXPECT_NEAR(out[3],  4.0f, 1e-6f);

    ggml_gallocr_free(galloc);
    ggml_hpx_exec_destroy(exec);
}

// Prefill integration test: mul_mat + neg through the scheduler-driven path.
// A single CPU backend produces one sched_split region. The HPX exec result
// is compared against ggml_backend_sched_graph_compute baseline.
TEST(HpxExec, PrefillRunMatchesBaselineForCpuOnlyMulMatNeg)
{
    struct fixture
    {
        ggml_backend_t      cpu   = nullptr;
        ggml_backend_sched_t sched = nullptr;
        ggml_context*        ctx   = nullptr;
        ggml_cgraph*         gf    = nullptr;
        ggml_tensor*         w     = nullptr;
        ggml_tensor*         x     = nullptr;
        ggml_tensor*         out   = nullptr;
    };

    auto make_graph = []() -> fixture
    {
        fixture f{};

        f.cpu = ggml_backend_init_by_type(
            GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        EXPECT_NE(f.cpu, nullptr);

        ggml_backend_t backends[1] = {f.cpu};
        f.sched = ggml_backend_sched_new(
            backends, nullptr, 1,
            GGML_DEFAULT_GRAPH_SIZE,
            /*parallel=*/false,
            /*op_offload=*/true);
        EXPECT_NE(f.sched, nullptr);

        ggml_init_params params = {};
        params.mem_size   = 2 * 1024 * 1024;
        params.mem_buffer = nullptr;
        params.no_alloc   = true;

        f.ctx = ggml_init(params);
        EXPECT_NE(f.ctx, nullptr);

        f.w = ggml_new_tensor_2d(f.ctx, GGML_TYPE_F32, 3, 2);
        f.x = ggml_new_tensor_2d(f.ctx, GGML_TYPE_F32, 3, 4);
        EXPECT_NE(f.w, nullptr);
        EXPECT_NE(f.x, nullptr);

        ggml_set_input(f.w);
        ggml_set_input(f.x);

        ggml_tensor* mm = ggml_mul_mat(f.ctx, f.w, f.x);
        EXPECT_NE(mm, nullptr);

        f.out = ggml_neg(f.ctx, mm);
        EXPECT_NE(f.out, nullptr);
        ggml_set_output(f.out);

        f.gf = ggml_new_graph(f.ctx);
        EXPECT_NE(f.gf, nullptr);
        ggml_build_forward_expand(f.gf, f.out);

        EXPECT_TRUE(ggml_backend_sched_reserve(f.sched, f.gf));
        EXPECT_TRUE(ggml_backend_sched_alloc_graph(f.sched, f.gf));

        return f;
    };

    auto free_graph = [](fixture& f)
    {
        if (f.sched != nullptr)
        {
            ggml_backend_sched_free(f.sched);
            f.sched = nullptr;
        }
        if (f.ctx != nullptr)
        {
            ggml_free(f.ctx);
            f.ctx = nullptr;
        }
        if (f.cpu != nullptr)
        {
            ggml_backend_free(f.cpu);
            f.cpu = nullptr;
        }
    };

    fixture baseline   = make_graph();
    fixture under_test = make_graph();

    std::vector<float> const w_data = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
    };
    std::vector<float> const x_data = {
         0.5f, -1.0f,  2.0f,
         1.5f,  0.0f, -0.5f,
         2.5f,  3.0f,  1.0f,
        -2.0f,  1.0f,  0.25f,
    };

    ggml_backend_tensor_set(
        baseline.w,   w_data.data(), 0, w_data.size() * sizeof(float));
    ggml_backend_tensor_set(
        baseline.x,   x_data.data(), 0, x_data.size() * sizeof(float));
    ggml_backend_tensor_set(
        under_test.w, w_data.data(), 0, w_data.size() * sizeof(float));
    ggml_backend_tensor_set(
        under_test.x, x_data.data(), 0, x_data.size() * sizeof(float));

    // CPU-only: one sched_split region covering the whole graph.
    // Avoid asserting exact split count in case the scheduler ever
    // subdivides differently; check coverage instead.
    {
        constexpr uint32_t k_policy = 1;
        ggml_hpx_adapter_result ar = ggml_hpx_adapt_prefill(
            under_test.gf, under_test.sched, k_policy);

        ASSERT_FALSE(ar.topo.regions.empty());
        EXPECT_EQ(ar.topo.regions.front().node_begin, 0u);
        EXPECT_EQ(ar.topo.regions.back().node_end,
            static_cast<uint32_t>(
                ggml_graph_n_nodes(under_test.gf)));
    }

    // Baseline: scheduler's own compute path.
    ASSERT_EQ(
        ggml_backend_sched_graph_compute(baseline.sched, baseline.gf),
        GGML_STATUS_SUCCESS);

    std::vector<float> expected(ggml_nbytes(baseline.out) / sizeof(float));
    ggml_backend_tensor_get(
        baseline.out, expected.data(), 0,
        expected.size() * sizeof(float));

    // HPX exec path: adapt → plan → cache → dispatch per region.
    ggml_hpx_exec_params exec_params = {};
    exec_params.runtime.n_decode_threads      = 1;
    exec_params.runtime.n_prefill_threads     = 4;
    exec_params.runtime.initial_scratch_bytes = 0;
    exec_params.policy_version                = 1;

    ggml_hpx_exec* exec = ggml_hpx_exec_create(exec_params);
    ASSERT_NE(exec, nullptr);

    EXPECT_EQ(
        ggml_hpx_exec_run_prefill(exec, under_test.gf, under_test.sched),
        ggml_hpx_exec_status::ok);
    ggml_backend_sched_synchronize(under_test.sched);

    std::vector<float> got(ggml_nbytes(under_test.out) / sizeof(float));
    ggml_backend_tensor_get(
        under_test.out, got.data(), 0, got.size() * sizeof(float));

    ASSERT_EQ(got.size(), expected.size());
    for (size_t i = 0; i < got.size(); ++i)
    {
        EXPECT_FLOAT_EQ(got[i], expected[i]) << "mismatch at index " << i;
    }

    ggml_hpx_exec_destroy(exec);
    free_graph(under_test);
    free_graph(baseline);
}

// Verify that ggml_backend_graph_compute executes exactly the nodes in
// cgraph->nodes[], regardless of whether they form a contiguous sub-range
// of some larger graph.
//
// Full graph (4 nodes):
//   x  = [1, -2, 3, -4]   (input)
//   A  = neg(x)            = [-1,  2, -3,  4]   index 0
//   B  = neg(A)            = [ 1, -2,  3, -4]   index 1  (interleaved dep)
//   C  = relu(x)           = [ 1,  0,  3,  0]   index 2
//   D  = abs(x)            = [ 1,  2,  3,  4]   index 3
//
// Scattered cgraph: [A, C, D]  — skips B (index 1, non-contiguous)
//
// Expected after scattered run:
//   A, C, D  — match baseline values (recomputed)
//   B        — stays zero           (skipped; proves node-list semantics)
//
// NOTE: allocates tensors via ggml_backend_alloc_ctx_tensors, NOT gallocr.
// gallocr's greedy in-place policy aliases x->data with D->data and
// A->data with B->data, so zeroing outputs would corrupt the input x.
// ggml_backend_alloc_ctx_tensors assigns each tensor a unique buffer.
TEST(HpxExec, BackendComputeScatteredNodeListMatchesBaseline)
{
    ggml_backend_t cpu = ggml_backend_init_by_type(
        GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ASSERT_NE(cpu, nullptr);

    // ----------------------------------------------------------------
    // Build graph: x -> A=neg(x), B=neg(A), C=relu(x), D=abs(x).
    // Allocate with ggml_backend_alloc_ctx_tensors so every tensor
    // gets its own address (no in-place aliasing from gallocr).
    // ----------------------------------------------------------------
    ggml_init_params params = {};
    params.mem_size   = 4 * 1024 * 1024;
    params.mem_buffer = nullptr;
    params.no_alloc   = true;

    ggml_context* ctx = ggml_init(params);
    ASSERT_NE(ctx, nullptr);

    ggml_tensor* x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_set_input(x);

    ggml_tensor* A = ggml_neg(ctx, x);
    ggml_tensor* B = ggml_neg(ctx, A);
    ggml_tensor* C = ggml_relu(ctx, x);
    ggml_tensor* D = ggml_abs(ctx, x);

    // Allocate: each tensor gets its own slice, no aliasing.
    ggml_backend_buffer_t buf =
        ggml_backend_alloc_ctx_tensors(ctx, cpu);
    ASSERT_NE(buf, nullptr);

    // Sanity: all five tensors must have distinct data pointers.
    ASSERT_NE(x->data, A->data);
    ASSERT_NE(x->data, B->data);
    ASSERT_NE(x->data, C->data);
    ASSERT_NE(x->data, D->data);
    ASSERT_NE(A->data, B->data);

    // ----------------------------------------------------------------
    // Full graph includes all four compute nodes in order A, B, C, D.
    // ----------------------------------------------------------------
    ggml_cgraph* full_graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(full_graph, A);
    ggml_build_forward_expand(full_graph, B);
    ggml_build_forward_expand(full_graph, C);
    ggml_build_forward_expand(full_graph, D);

    // ----------------------------------------------------------------
    // Baseline: run the full graph, capture A/B/C/D outputs.
    // ----------------------------------------------------------------
    float const x_in[4] = {1.0f, -2.0f, 3.0f, -4.0f};
    ggml_backend_tensor_set(x, x_in, 0, sizeof(x_in));

    ASSERT_EQ(ggml_backend_graph_compute(cpu, full_graph),
        GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(cpu);

    float baseline_A[4] = {};
    float baseline_B[4] = {};
    float baseline_C[4] = {};
    float baseline_D[4] = {};
    ggml_backend_tensor_get(A, baseline_A, 0, sizeof(baseline_A));
    ggml_backend_tensor_get(B, baseline_B, 0, sizeof(baseline_B));
    ggml_backend_tensor_get(C, baseline_C, 0, sizeof(baseline_C));
    ggml_backend_tensor_get(D, baseline_D, 0, sizeof(baseline_D));

    // Sanity-check baseline values.
    EXPECT_NEAR(baseline_A[0], -1.0f, 1e-6f);    // neg(1)
    EXPECT_NEAR(baseline_B[0],  1.0f, 1e-6f);    // neg(neg(1))
    EXPECT_NEAR(baseline_C[0],  1.0f, 1e-6f);    // relu(1)
    EXPECT_NEAR(baseline_D[0],  1.0f, 1e-6f);    // abs(1)

    // ----------------------------------------------------------------
    // Zero all four output buffers so any non-execution is detectable.
    // x is separate from all outputs (unique buffer) so it is unaffected.
    // ----------------------------------------------------------------
    float const zeros[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    ggml_backend_tensor_set(A, zeros, 0, sizeof(zeros));
    ggml_backend_tensor_set(B, zeros, 0, sizeof(zeros));
    ggml_backend_tensor_set(C, zeros, 0, sizeof(zeros));
    ggml_backend_tensor_set(D, zeros, 0, sizeof(zeros));

    // ----------------------------------------------------------------
    // Scattered cgraph: only [A, C, D] — B intentionally skipped.
    // ggml_graph_add_node is a direct append; no recursive expansion.
    // ----------------------------------------------------------------
    ggml_cgraph* scattered = ggml_new_graph_custom(ctx, 4, false);
    ggml_graph_add_node(scattered, A);
    ggml_graph_add_node(scattered, C);
    ggml_graph_add_node(scattered, D);

    EXPECT_EQ(ggml_graph_n_nodes(scattered), 3);

    ASSERT_EQ(ggml_backend_graph_compute(cpu, scattered),
        GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(cpu);

    // ----------------------------------------------------------------
    // A, C, D must match baseline; B must remain zero (was not executed).
    // ----------------------------------------------------------------
    float got_A[4] = {};
    float got_B[4] = {};
    float got_C[4] = {};
    float got_D[4] = {};
    ggml_backend_tensor_get(A, got_A, 0, sizeof(got_A));
    ggml_backend_tensor_get(B, got_B, 0, sizeof(got_B));
    ggml_backend_tensor_get(C, got_C, 0, sizeof(got_C));
    ggml_backend_tensor_get(D, got_D, 0, sizeof(got_D));

    for (int i = 0; i < 4; ++i)
    {
        EXPECT_FLOAT_EQ(got_A[i], baseline_A[i]) << "A mismatch at " << i;
        EXPECT_FLOAT_EQ(got_C[i], baseline_C[i]) << "C mismatch at " << i;
        EXPECT_FLOAT_EQ(got_D[i], baseline_D[i]) << "D mismatch at " << i;
        EXPECT_FLOAT_EQ(got_B[i], 0.0f)
            << "B should be zero (skipped) at " << i;
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(cpu);
}
