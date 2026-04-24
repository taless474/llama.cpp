#include <gtest/gtest.h>

#include "ggml-hpx-plan.h"
#include "test_hpx_test_utils.h"

using namespace hpx_test;

// ---------------------------------------------------------------------------
// Plan key: equality and inequality operators
// (ggml_hpx_plan_key is mode-independent; tested once here)
// ---------------------------------------------------------------------------

TEST(HpxPlanKey, EqualityAndInequalityWork)
{
    auto const a = make_key(ggml_hpx_mode::decode);
    auto const b = make_key(ggml_hpx_mode::decode);
    auto const c = with_policy_version(a, a.policy_version + 1);

    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a != b);

    EXPECT_FALSE(a == c);
    EXPECT_TRUE(a != c);
}

// ---------------------------------------------------------------------------
// Decode plan: build
// ---------------------------------------------------------------------------

TEST(HpxDecodePlanBuild, CopiesRegionsAndCreatesParallelChunkSpecs)
{
    auto const topo = make_decode_topology();
    auto const key  = make_key(ggml_hpx_mode::decode);

    auto const plan = ggml_hpx_build_decode_plan(topo, key);

    EXPECT_EQ(plan.key.mode, ggml_hpx_mode::decode);
    EXPECT_EQ(plan.key, key);
    ASSERT_EQ(plan.regions.size(), topo.regions.size());
    ASSERT_EQ(plan.chunk_specs.size(), topo.regions.size());

    for (size_t i = 0; i < topo.regions.size(); ++i)
    {
        EXPECT_EQ(plan.regions[i].type, topo.regions[i].type);
        EXPECT_EQ(
            plan.regions[i].node_begin, topo.regions[i].node_begin);
        EXPECT_EQ(plan.regions[i].node_end, topo.regions[i].node_end);
        EXPECT_EQ(plan.regions[i].prev_idx, topo.regions[i].prev_idx);
    }

    for (auto const& chunk : plan.chunk_specs)
    {
        EXPECT_EQ(chunk.policy, ggml_hpx_chunk_policy::auto_size);
        EXPECT_EQ(chunk.param, 0u);
    }
}

TEST(HpxDecodePlanBuild, PlanDoesNotAliasInputTopologyStorage)
{
    auto topo       = make_decode_topology();
    auto const key  = make_key(ggml_hpx_mode::decode);
    auto const plan = ggml_hpx_build_decode_plan(topo, key);

    topo.regions[0].node_begin = 99;
    topo.regions[1].prev_idx   = UINT32_MAX;

    EXPECT_EQ(plan.regions[0].node_begin, 0u);
    EXPECT_EQ(plan.regions[1].prev_idx, 0u);  // region 1's predecessor is region 0
}

// ---------------------------------------------------------------------------
// Decode plan: check
// ---------------------------------------------------------------------------

TEST(HpxDecodePlanCheck, ReturnsValidForExactKeyMatch)
{
    auto const topo = make_decode_topology();
    auto const key  = make_key(ggml_hpx_mode::decode);
    auto const plan = ggml_hpx_build_decode_plan(topo, key);

    EXPECT_EQ(
        ggml_hpx_decode_plan_check(plan, key),
        ggml_hpx_plan_check_result::valid);
}

TEST(HpxDecodePlanCheck, ReportsWhichKeyFieldWentStale)
{
    auto const topo = make_decode_topology();
    auto const key  = make_key(ggml_hpx_mode::decode);
    auto const plan = ggml_hpx_build_decode_plan(topo, key);

    EXPECT_EQ(
        ggml_hpx_decode_plan_check(
            plan, with_graph_hash(key, key.graph_shape_hash + 1)),
        ggml_hpx_plan_check_result::stale_graph_shape);

    EXPECT_EQ(
        ggml_hpx_decode_plan_check(
            plan,
            with_backend_hash(key, key.backend_assign_hash + 1)),
        ggml_hpx_plan_check_result::stale_backend_assign);

    EXPECT_EQ(
        ggml_hpx_decode_plan_check(
            plan,
            with_workspace_sig(key, key.workspace_layout_sig + 1)),
        ggml_hpx_plan_check_result::stale_workspace);

    EXPECT_EQ(
        ggml_hpx_decode_plan_check(
            plan, with_policy_version(key, key.policy_version + 1)),
        ggml_hpx_plan_check_result::stale_policy);
}

// ---------------------------------------------------------------------------
// Decode topology: validate
// ---------------------------------------------------------------------------

TEST(HpxDecodePlanValidate, RejectsSchedSplitRegion)
{
    auto const bad = make_bad_topology_sched_split();
    EXPECT_FALSE(ggml_hpx_validate_decode_topology(bad));
}

// ---------------------------------------------------------------------------
// Decode plan: build death tests
// ---------------------------------------------------------------------------

#if GTEST_HAS_DEATH_TEST
TEST(HpxDecodePlanBuildDeathTest, RejectsOutOfBoundsTopology)
{
    auto const bad = make_bad_topology_out_of_bounds();
    auto const key = make_key(ggml_hpx_mode::decode);

    EXPECT_DEATH_IF_SUPPORTED(
        { (void) ggml_hpx_build_decode_plan(bad, key); },
        ".*");
}

TEST(HpxDecodePlanBuildDeathTest, RejectsSchedSplitRegion)
{
    auto const bad = make_bad_topology_sched_split();
    auto const key = make_key(ggml_hpx_mode::decode);

    EXPECT_DEATH_IF_SUPPORTED(
        { (void) ggml_hpx_build_decode_plan(bad, key); },
        ".*");
}
#endif
