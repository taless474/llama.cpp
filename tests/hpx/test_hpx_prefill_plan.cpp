#include <gtest/gtest.h>

#include "ggml-hpx-plan.h"
#include "test_hpx_test_utils.h"

using namespace hpx_test;

// ---------------------------------------------------------------------------
// Prefill plan: build
// ---------------------------------------------------------------------------

TEST(HpxPrefillPlanBuild, CopiesTopologyAndCreatesRegionPolicies)
{
    auto const topo = make_prefill_topology();
    auto const key  = make_key(ggml_hpx_mode::prefill);

    auto const plan = ggml_hpx_build_prefill_plan(topo, key);

    EXPECT_EQ(plan.key.mode, ggml_hpx_mode::prefill);
    EXPECT_EQ(plan.key, key);
    EXPECT_EQ(plan.topo.n_nodes, topo.n_nodes);
    ASSERT_EQ(plan.topo.regions.size(), topo.regions.size());
    ASSERT_EQ(plan.region_policies.size(), topo.regions.size());
    EXPECT_EQ(plan.overlap, ggml_hpx_overlap_policy::none);

    for (size_t i = 0; i < topo.regions.size(); ++i)
    {
        EXPECT_EQ(
            plan.topo.regions[i].type, topo.regions[i].type);
        EXPECT_EQ(
            plan.topo.regions[i].node_begin,
            topo.regions[i].node_begin);
        EXPECT_EQ(
            plan.topo.regions[i].node_end, topo.regions[i].node_end);
        EXPECT_EQ(
            plan.topo.regions[i].prev_idx, topo.regions[i].prev_idx);

        EXPECT_EQ(
            plan.region_policies[i].chunk.policy,
            ggml_hpx_chunk_policy::auto_size);
        EXPECT_EQ(plan.region_policies[i].chunk.param, 0u);
    }
}

TEST(HpxPrefillPlanBuild, PlanOwnsItsTopologyByValue)
{
    auto topo       = make_prefill_topology();
    auto const key  = make_key(ggml_hpx_mode::prefill);
    auto const plan = ggml_hpx_build_prefill_plan(topo, key);

    topo.n_nodes           = 123;
    topo.regions[0].node_end = 42;

    EXPECT_EQ(plan.topo.n_nodes, 10u);
    EXPECT_EQ(plan.topo.regions[0].node_end, 3u);
}

// ---------------------------------------------------------------------------
// Prefill plan: check
// ---------------------------------------------------------------------------

TEST(HpxPrefillPlanCheck, ReturnsValidForExactKeyMatch)
{
    auto const topo = make_prefill_topology();
    auto const key  = make_key(ggml_hpx_mode::prefill);
    auto const plan = ggml_hpx_build_prefill_plan(topo, key);

    EXPECT_EQ(
        ggml_hpx_prefill_plan_check(plan, key),
        ggml_hpx_plan_check_result::valid);
}

TEST(HpxPrefillPlanCheck, ReportsWhichKeyFieldWentStale)
{
    auto const topo = make_prefill_topology();
    auto const key  = make_key(ggml_hpx_mode::prefill);
    auto const plan = ggml_hpx_build_prefill_plan(topo, key);

    EXPECT_EQ(
        ggml_hpx_prefill_plan_check(
            plan, with_graph_hash(key, key.graph_shape_hash + 1)),
        ggml_hpx_plan_check_result::stale_graph_shape);

    EXPECT_EQ(
        ggml_hpx_prefill_plan_check(
            plan,
            with_backend_hash(key, key.backend_assign_hash + 1)),
        ggml_hpx_plan_check_result::stale_backend_assign);

    EXPECT_EQ(
        ggml_hpx_prefill_plan_check(
            plan,
            with_workspace_sig(key, key.workspace_layout_sig + 1)),
        ggml_hpx_plan_check_result::stale_workspace);

    EXPECT_EQ(
        ggml_hpx_prefill_plan_check(
            plan, with_policy_version(key, key.policy_version + 1)),
        ggml_hpx_plan_check_result::stale_policy);
}

// ---------------------------------------------------------------------------
// Prefill plan: death tests
// (enabled once plan.cpp uses GGML_ASSERT for malformed topology)
// ---------------------------------------------------------------------------

#if GTEST_HAS_DEATH_TEST
TEST(HpxPrefillPlanBuildDeathTest, RejectsEmptySpanTopology)
{
    auto const bad = make_bad_topology_empty_span();
    auto const key = make_key(ggml_hpx_mode::prefill);

    EXPECT_DEATH_IF_SUPPORTED(
        { (void) ggml_hpx_build_prefill_plan(bad, key); },
        ".*");
}
#endif
