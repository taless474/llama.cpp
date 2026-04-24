#include <gtest/gtest.h>

#include "ggml-hpx-adapter.h"
#include "ggml-hpx-region.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace {

struct ggml_context_deleter
{
    void operator()(ggml_context* ctx) const noexcept
    {
        if (ctx != nullptr)
            ggml_free(ctx);
    }
};

struct graph_owner
{
    std::vector<uint8_t> storage;
    std::unique_ptr<ggml_context, ggml_context_deleter> ctx;
    ggml_cgraph* gf = nullptr;
};

// Owns a graph, two CPU backends, and a scheduler for prefill tests.
// The graph is embedded as a graph_owner; context and storage lifetime
// are managed by graph_owner, not duplicated here.
struct prefill_graph_fixture
{
    graph_owner graph;

    ggml_backend_t backend0 = nullptr;
    ggml_backend_t backend1 = nullptr;
    ggml_backend_sched_t sched = nullptr;

    ~prefill_graph_fixture()
    {
        if (sched != nullptr)
            ggml_backend_sched_free(sched);
        if (backend1 != nullptr)
            ggml_backend_free(backend1);
        if (backend0 != nullptr)
            ggml_backend_free(backend0);
    }

    ggml_cgraph* gf() const noexcept { return graph.gf; }
    ggml_context* ctx() const noexcept { return graph.ctx.get(); }
};

// ---------------------------------------------------------------------------
// Graph helpers
// ---------------------------------------------------------------------------

// Base helper: allocate storage + context, create an empty graph.
// Returns with gf == nullptr on any allocation failure.
static graph_owner make_base_graph(
    size_t mem_size = 1 << 16, int max_nodes = 16)
{
    graph_owner out{};
    out.storage.resize(mem_size);

    ggml_init_params params{};
    params.mem_size   = out.storage.size();
    params.mem_buffer = out.storage.data();
    params.no_alloc   = true;

    out.ctx.reset(ggml_init(params));
    if (!out.ctx)
        return out;

    out.gf = ggml_new_graph_custom(out.ctx.get(), max_nodes, false);
    return out;
}

// Zero-node graph: context and graph object created, no forward expand.
static graph_owner make_empty_graph()
{
    return make_base_graph(1 << 14, 8);
}

// Single compute node: a + b (one ADD op, two leaf inputs).
static graph_owner make_single_node_graph()
{
    auto out = make_base_graph(1 << 14, 8);
    if (!out.gf)
        return out;

    ggml_tensor* a = ggml_new_tensor_1d(out.ctx.get(), GGML_TYPE_F32, 8);
    ggml_tensor* b = ggml_new_tensor_1d(out.ctx.get(), GGML_TYPE_F32, 8);
    if (!a || !b)
    {
        out.gf = nullptr;
        return out;
    }

    ggml_tensor* c = ggml_add(out.ctx.get(), a, b);
    if (!c)
    {
        out.gf = nullptr;
        return out;
    }

    ggml_build_forward_expand(out.gf, c);
    return out;
}

// Two-node graph: a + b → mul by b. Matches the shape used in key-stability
// and topology tests.
static graph_owner make_decode_graph(
    size_t mem_size = 1 << 16, int max_nodes = 16)
{
    auto out = make_base_graph(mem_size, max_nodes);
    if (!out.gf)
        return out;

    ggml_tensor* a = ggml_new_tensor_1d(out.ctx.get(), GGML_TYPE_F32, 8);
    ggml_tensor* b = ggml_new_tensor_1d(out.ctx.get(), GGML_TYPE_F32, 8);
    if (!a || !b)
    {
        out.gf = nullptr;
        return out;
    }

    ggml_tensor* c = ggml_add(out.ctx.get(), a, b);
    ggml_tensor* d = ggml_mul(out.ctx.get(), c, b);
    if (!c || !d)
    {
        out.gf = nullptr;
        return out;
    }

    ggml_build_forward_expand(out.gf, d);
    return out;
}

// Four-node chain: add → mul → add → mul.
// Used for prefill split tests: nodes 0-1 assigned to one backend,
// nodes 2-3 to another, producing two distinct scheduler splits.
static graph_owner make_prefill_shape_graph()
{
    auto out = make_base_graph(1 << 15, 16);
    if (!out.gf)
        return out;

    ggml_tensor* a = ggml_new_tensor_1d(out.ctx.get(), GGML_TYPE_F32, 8);
    ggml_tensor* b = ggml_new_tensor_1d(out.ctx.get(), GGML_TYPE_F32, 8);
    if (!a || !b)
    {
        out.gf = nullptr;
        return out;
    }

    ggml_tensor* n0 = ggml_add(out.ctx.get(), a, b);
    ggml_tensor* n1 = ggml_mul(out.ctx.get(), n0, b);
    ggml_tensor* n2 = ggml_add(out.ctx.get(), n1, a);
    ggml_tensor* n3 = ggml_mul(out.ctx.get(), n2, b);
    if (!n0 || !n1 || !n2 || !n3)
    {
        out.gf = nullptr;
        return out;
    }

    ggml_build_forward_expand(out.gf, n3);
    return out;
}

// ---------------------------------------------------------------------------
// Scheduler helper
// ---------------------------------------------------------------------------

// Create a two-backend scheduler inside fx. Returns false if creation fails.
// Both backend handles are transferred into fx; fx destructor owns them.
static bool make_two_backend_sched(
    prefill_graph_fixture& fx,
    ggml_backend_t backend0,
    ggml_backend_t backend1)
{
    fx.backend0 = backend0;
    fx.backend1 = backend1;

    ggml_backend_t backends[2] = {fx.backend0, fx.backend1};
    fx.sched = ggml_backend_sched_new(
        backends,
        /*bufts=*/nullptr,
        /*n_backends=*/2,
        /*graph_size=*/16,
        /*parallel=*/false,
        /*op_offload=*/true);

    return fx.sched != nullptr;
}

}    // namespace

// ---------------------------------------------------------------------------
// Decode: key stability
// ---------------------------------------------------------------------------

TEST(HpxAdapterDecode, SameGraphAndPolicyProduceStableKey)
{
    auto g1 = make_decode_graph();
    auto g2 = make_decode_graph();

    ASSERT_NE(g1.ctx.get(), nullptr);
    ASSERT_NE(g1.gf, nullptr);
    ASSERT_NE(g2.ctx.get(), nullptr);
    ASSERT_NE(g2.gf, nullptr);

    ggml_hpx_adapter_result const r1 =
        ggml_hpx_adapt_decode(g1.gf, /*policy_version=*/7);
    ggml_hpx_adapter_result const r2 =
        ggml_hpx_adapt_decode(g2.gf, /*policy_version=*/7);

    EXPECT_EQ(r1.key.mode, ggml_hpx_mode::decode);
    EXPECT_EQ(r2.key.mode, ggml_hpx_mode::decode);
    EXPECT_EQ(r1.key.policy_version, 7u);
    EXPECT_EQ(r2.key.policy_version, 7u);

    EXPECT_EQ(r1.key.graph_shape_hash, r2.key.graph_shape_hash);
    EXPECT_EQ(r1.key.backend_assign_hash, r2.key.backend_assign_hash);
    EXPECT_EQ(r1.key.workspace_layout_sig, r2.key.workspace_layout_sig);
}

TEST(HpxAdapterDecode, PolicyVersionFeedsThroughToKey)
{
    auto g = make_decode_graph();

    ASSERT_NE(g.ctx.get(), nullptr);
    ASSERT_NE(g.gf, nullptr);

    ggml_hpx_adapter_result const r1 =
        ggml_hpx_adapt_decode(g.gf, /*policy_version=*/7);
    ggml_hpx_adapter_result const r2 =
        ggml_hpx_adapt_decode(g.gf, /*policy_version=*/8);

    EXPECT_EQ(r1.key.mode, ggml_hpx_mode::decode);
    EXPECT_EQ(r2.key.mode, ggml_hpx_mode::decode);
    EXPECT_EQ(r1.key.policy_version, 7u);
    EXPECT_EQ(r2.key.policy_version, 8u);

    // Structural hashes must be stable across policy-version changes.
    EXPECT_EQ(r1.key.graph_shape_hash, r2.key.graph_shape_hash);
    EXPECT_EQ(r1.key.backend_assign_hash, r2.key.backend_assign_hash);
    EXPECT_EQ(r1.key.workspace_layout_sig, r2.key.workspace_layout_sig);
}

// ---------------------------------------------------------------------------
// Decode: topology shape
// ---------------------------------------------------------------------------

TEST(HpxAdapterDecode, ProducesTopologyWithBoundedRegionsAndNoSchedSplitRegions)
{
    auto g = make_decode_graph();

    ASSERT_NE(g.ctx.get(), nullptr);
    ASSERT_NE(g.gf, nullptr);

    ggml_hpx_adapter_result const r =
        ggml_hpx_adapt_decode(g.gf, /*policy_version=*/7);

    EXPECT_GT(r.topo.n_nodes, 0u);
    ASSERT_FALSE(r.topo.regions.empty());

    for (ggml_hpx_region const& region : r.topo.regions)
    {
        EXPECT_LT(region.node_begin, region.node_end);
        EXPECT_LE(region.node_end, r.topo.n_nodes);
        EXPECT_NE(region.type, ggml_hpx_region_type::sched_split);
    }
}

TEST(HpxAdapterDecode, ResultOwnsItsTopologyByValue)
{
    auto g = make_decode_graph();

    ASSERT_NE(g.ctx.get(), nullptr);
    ASSERT_NE(g.gf, nullptr);

    ggml_hpx_adapter_result r =
        ggml_hpx_adapt_decode(g.gf, /*policy_version=*/7);

    ASSERT_FALSE(r.topo.regions.empty());

    uint32_t const original_begin = r.topo.regions.front().node_begin;

    ggml_hpx_adapter_result copy = r;
    r.topo.regions.front().node_begin = original_begin + 100;

    EXPECT_EQ(copy.topo.regions.front().node_begin, original_begin);
}

// ---------------------------------------------------------------------------
// Decode: edge cases — empty graph and single-node graph
// ---------------------------------------------------------------------------

TEST(HpxAdapterDecode, EmptyGraphProducesEmptyTopology)
{
    auto g = make_empty_graph();

    ASSERT_NE(g.ctx.get(), nullptr);
    ASSERT_NE(g.gf, nullptr);

    ggml_hpx_adapter_result const r =
        ggml_hpx_adapt_decode(g.gf, /*policy_version=*/0);

    EXPECT_EQ(r.topo.n_nodes, 0u);
    EXPECT_TRUE(r.topo.regions.empty());
    EXPECT_EQ(r.key.mode, ggml_hpx_mode::decode);
    EXPECT_EQ(r.key.policy_version, 0u);
}

TEST(HpxAdapterDecode, SingleNodeGraphProducesOneRegionWithZeroDependencies)
{
    auto g = make_single_node_graph();

    ASSERT_NE(g.ctx.get(), nullptr);
    ASSERT_NE(g.gf, nullptr);

    ggml_hpx_adapter_result const r =
        ggml_hpx_adapt_decode(g.gf, /*policy_version=*/1);

    EXPECT_EQ(r.topo.n_nodes, 1u);
    ASSERT_EQ(r.topo.regions.size(), 1u);

    EXPECT_EQ(r.topo.regions[0].node_begin, 0u);
    EXPECT_EQ(r.topo.regions[0].node_end, 1u);
    // First region has no predecessors.
    EXPECT_EQ(r.topo.regions[0].prev_idx, UINT32_MAX);
    // With no_alloc=true all buffers are null → cpu_contiguous.
    EXPECT_EQ(r.topo.regions[0].type,
        ggml_hpx_region_type::cpu_contiguous);
}

// ---------------------------------------------------------------------------
// Prefill: split-distinctness contracts
// ---------------------------------------------------------------------------

TEST(HpxAdapterPrefill, AdjacentCpuSplitsRemainDistinct)
{
    prefill_graph_fixture fx{};
    fx.graph = make_prefill_shape_graph();

    ASSERT_NE(fx.ctx(), nullptr);
    ASSERT_NE(fx.gf(), nullptr);

    // Two distinct CPU backend handles force a split boundary between them.
    ggml_backend_t b0 = ggml_backend_cpu_init();
    ggml_backend_t b1 = ggml_backend_cpu_init();
    ASSERT_NE(b0, nullptr);
    ASSERT_NE(b1, nullptr);

    ASSERT_TRUE(make_two_backend_sched(fx, b0, b1));

    int const n           = ggml_graph_n_nodes(fx.gf());
    ggml_tensor** const nodes = ggml_graph_nodes(fx.gf());

    ASSERT_GE(n, 4);

    ggml_backend_sched_set_tensor_backend(fx.sched, nodes[0], fx.backend0);
    ggml_backend_sched_set_tensor_backend(fx.sched, nodes[1], fx.backend0);
    ggml_backend_sched_set_tensor_backend(fx.sched, nodes[2], fx.backend1);
    ggml_backend_sched_set_tensor_backend(fx.sched, nodes[3], fx.backend1);

    // Materialise the split state from the manual backend assignments above.
    // adapt_prefill no longer calls split_graph internally.
    ggml_backend_sched_split_graph(fx.sched, fx.gf());

    ggml_hpx_adapter_result const r =
        ggml_hpx_adapt_prefill(fx.gf(), fx.sched, /*policy_version=*/0);

    EXPECT_EQ(r.key.mode, ggml_hpx_mode::prefill);
    ASSERT_GE(r.topo.regions.size(), 2u);
    // Each scheduler split must become its own region even when both are
    // CPU-bound (sched_split type).
    EXPECT_EQ(r.topo.regions[0].type, ggml_hpx_region_type::sched_split);
    EXPECT_EQ(r.topo.regions[1].type, ggml_hpx_region_type::sched_split);
    // The two regions must cover disjoint, contiguous node ranges.
    EXPECT_EQ(r.topo.regions[0].node_end, r.topo.regions[1].node_begin);
}

TEST(HpxAdapterPrefill, AdjacentDelegatedOnDifferentBackendsRemainDistinct)
{
    GTEST_SKIP()
        << "Requires two distinct non-CPU ggml_backend_t objects (e.g. BLAS "
           "or GPU). Add make_prefill_graph_with_two_delegated_backends() "
           "when a second accelerator backend is available in the test env.";
}

TEST(HpxAdapterPrefill, PrefillTopologyRegionsAreBounded)
{
    prefill_graph_fixture fx{};
    fx.graph = make_prefill_shape_graph();

    ASSERT_NE(fx.ctx(), nullptr);
    ASSERT_NE(fx.gf(), nullptr);

    ggml_backend_t b0 = ggml_backend_cpu_init();
    ggml_backend_t b1 = ggml_backend_cpu_init();
    ASSERT_NE(b0, nullptr);
    ASSERT_NE(b1, nullptr);

    ASSERT_TRUE(make_two_backend_sched(fx, b0, b1));

    int const n           = ggml_graph_n_nodes(fx.gf());
    ggml_tensor** const nodes = ggml_graph_nodes(fx.gf());

    ASSERT_GE(n, 4);

    ggml_backend_sched_set_tensor_backend(fx.sched, nodes[0], fx.backend0);
    ggml_backend_sched_set_tensor_backend(fx.sched, nodes[1], fx.backend0);
    ggml_backend_sched_set_tensor_backend(fx.sched, nodes[2], fx.backend1);
    ggml_backend_sched_set_tensor_backend(fx.sched, nodes[3], fx.backend1);

    ggml_backend_sched_split_graph(fx.sched, fx.gf());

    ggml_hpx_adapter_result const r =
        ggml_hpx_adapt_prefill(fx.gf(), fx.sched, /*policy_version=*/7);

    EXPECT_EQ(r.key.mode, ggml_hpx_mode::prefill);
    EXPECT_EQ(r.key.policy_version, 7u);
    ASSERT_FALSE(r.topo.regions.empty());

    for (ggml_hpx_region const& region : r.topo.regions)
    {
        EXPECT_LT(region.node_begin, region.node_end);
        EXPECT_LE(region.node_end, r.topo.n_nodes);
    }
}
