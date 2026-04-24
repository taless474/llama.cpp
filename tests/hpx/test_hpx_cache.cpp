#include <gtest/gtest.h>

#include "ggml-hpx-cache.h"
#include "ggml-hpx-plan.h"
#include "test_hpx_test_utils.h"

using namespace hpx_test;

// Cache API (ggml-hpx-cache.h):
//   ggml_hpx_plan_cache             -- value struct; default-construct in place
//   ggml_hpx_decode_plan const*  ggml_hpx_cache_lookup_decode(cache, key)
//   ggml_hpx_decode_plan const*  ggml_hpx_cache_insert_decode(cache, plan)
//   ggml_hpx_prefill_plan const* ggml_hpx_cache_lookup_prefill(cache, key)
//   ggml_hpx_prefill_plan const* ggml_hpx_cache_insert_prefill(cache, plan)
//   void ggml_hpx_cache_clear_decode(cache)
//   void ggml_hpx_cache_clear_prefill(cache)
//   void ggml_hpx_cache_clear(cache)

class HpxCacheTest : public ::testing::Test
{
protected:
    ggml_hpx_plan_cache cache{};
};

TEST_F(HpxCacheTest, LookupMissReturnsNullptrForDecode)
{
    auto const key = make_key(ggml_hpx_mode::decode);
    EXPECT_EQ(ggml_hpx_cache_lookup_decode(cache, key), nullptr);
}

TEST_F(HpxCacheTest, LookupMissReturnsNullptrForPrefill)
{
    auto const key = make_key(ggml_hpx_mode::prefill);
    EXPECT_EQ(ggml_hpx_cache_lookup_prefill(cache, key), nullptr);
}

TEST_F(HpxCacheTest, InsertDecodeReturnsStablePointerAndMakesSubsequentLookupHit)
{
    auto const topo = make_decode_topology();
    auto const key = make_key(ggml_hpx_mode::decode);
    auto plan = ggml_hpx_build_decode_plan(topo, key);

    auto const* inserted = ggml_hpx_cache_insert_decode(cache, plan);
    ASSERT_NE(inserted, nullptr);

    auto const* looked_up = ggml_hpx_cache_lookup_decode(cache, key);
    ASSERT_NE(looked_up, nullptr);
    EXPECT_EQ(looked_up, inserted);
    EXPECT_EQ(looked_up->key, key);
    ASSERT_EQ(looked_up->regions.size(), topo.regions.size());
    ASSERT_EQ(looked_up->chunk_specs.size(), topo.regions.size());
}

TEST_F(HpxCacheTest, InsertPrefillReturnsStablePointerAndMakesSubsequentLookupHit)
{
    auto const topo = make_prefill_topology();
    auto const key = make_key(ggml_hpx_mode::prefill);
    auto plan = ggml_hpx_build_prefill_plan(topo, key);

    auto const* inserted = ggml_hpx_cache_insert_prefill(cache, plan);
    ASSERT_NE(inserted, nullptr);

    auto const* looked_up = ggml_hpx_cache_lookup_prefill(cache, key);
    ASSERT_NE(looked_up, nullptr);
    EXPECT_EQ(looked_up, inserted);
    EXPECT_EQ(looked_up->key, key);
    ASSERT_EQ(looked_up->topo.regions.size(), topo.regions.size());
    ASSERT_EQ(looked_up->region_policies.size(), topo.regions.size());
}

TEST_F(HpxCacheTest, CacheOwnsACopyOfInsertedDecodePlan)
{
    auto const topo = make_decode_topology();
    auto const key = make_key(ggml_hpx_mode::decode);
    auto plan = ggml_hpx_build_decode_plan(topo, key);

    auto const* inserted = ggml_hpx_cache_insert_decode(cache, plan);
    ASSERT_NE(inserted, nullptr);

    plan.regions[0].node_begin = 99;
    plan.chunk_specs[0].param = 77;
    plan.chunk_specs[0].policy = ggml_hpx_chunk_policy::fixed_count;

    auto const* looked_up = ggml_hpx_cache_lookup_decode(cache, key);
    ASSERT_NE(looked_up, nullptr);
    EXPECT_EQ(looked_up->regions[0].node_begin, 0u);
    EXPECT_EQ(
        looked_up->chunk_specs[0].policy, ggml_hpx_chunk_policy::auto_size);
    EXPECT_EQ(looked_up->chunk_specs[0].param, 0u);
}

TEST_F(HpxCacheTest, CacheOwnsACopyOfInsertedPrefillPlan)
{
    auto const topo = make_prefill_topology();
    auto const key = make_key(ggml_hpx_mode::prefill);
    auto plan = ggml_hpx_build_prefill_plan(topo, key);

    auto const* inserted = ggml_hpx_cache_insert_prefill(cache, plan);
    ASSERT_NE(inserted, nullptr);

    plan.topo.regions[0].node_end = 99;
    plan.region_policies[0].chunk.param = 55;
    plan.overlap = ggml_hpx_overlap_policy::opportunistic;

    auto const* looked_up = ggml_hpx_cache_lookup_prefill(cache, key);
    ASSERT_NE(looked_up, nullptr);
    EXPECT_EQ(looked_up->topo.regions[0].node_end, 3u);
    EXPECT_EQ(looked_up->region_policies[0].chunk.param, 0u);
    EXPECT_EQ(looked_up->overlap, ggml_hpx_overlap_policy::none);
}

TEST_F(HpxCacheTest, ClearDecodeRemovesOnlyDecodePlans)
{
    auto const d_topo = make_decode_topology();
    auto const p_topo = make_prefill_topology();
    auto const d_key = make_key(ggml_hpx_mode::decode);
    auto const p_key = make_key(ggml_hpx_mode::prefill);

    ggml_hpx_cache_insert_decode(
        cache, ggml_hpx_build_decode_plan(d_topo, d_key));
    ggml_hpx_cache_insert_prefill(
        cache, ggml_hpx_build_prefill_plan(p_topo, p_key));

    ASSERT_NE(ggml_hpx_cache_lookup_decode(cache, d_key), nullptr);
    ASSERT_NE(ggml_hpx_cache_lookup_prefill(cache, p_key), nullptr);

    ggml_hpx_cache_clear_decode(cache);

    EXPECT_EQ(ggml_hpx_cache_lookup_decode(cache, d_key), nullptr);
    EXPECT_NE(ggml_hpx_cache_lookup_prefill(cache, p_key), nullptr);
}

TEST_F(HpxCacheTest, ClearPrefillRemovesOnlyPrefillPlans)
{
    auto const d_topo = make_decode_topology();
    auto const p_topo = make_prefill_topology();
    auto const d_key = make_key(ggml_hpx_mode::decode);
    auto const p_key = make_key(ggml_hpx_mode::prefill);

    ggml_hpx_cache_insert_decode(
        cache, ggml_hpx_build_decode_plan(d_topo, d_key));
    ggml_hpx_cache_insert_prefill(
        cache, ggml_hpx_build_prefill_plan(p_topo, p_key));

    ggml_hpx_cache_clear_prefill(cache);

    EXPECT_NE(ggml_hpx_cache_lookup_decode(cache, d_key), nullptr);
    EXPECT_EQ(ggml_hpx_cache_lookup_prefill(cache, p_key), nullptr);
}

TEST_F(HpxCacheTest, ClearAllRemovesEverything)
{
    auto const d_topo = make_decode_topology();
    auto const p_topo = make_prefill_topology();
    auto const d_key = make_key(ggml_hpx_mode::decode);
    auto const p_key = make_key(ggml_hpx_mode::prefill);

    ggml_hpx_cache_insert_decode(
        cache, ggml_hpx_build_decode_plan(d_topo, d_key));
    ggml_hpx_cache_insert_prefill(
        cache, ggml_hpx_build_prefill_plan(p_topo, p_key));

    ggml_hpx_cache_clear(cache);

    EXPECT_EQ(ggml_hpx_cache_lookup_decode(cache, d_key), nullptr);
    EXPECT_EQ(ggml_hpx_cache_lookup_prefill(cache, p_key), nullptr);
}

TEST_F(HpxCacheTest, InsertSecondDecodePlanReplacesFirst)
{
    auto const topo1 = make_decode_topology();
    auto const key1  = make_key(
        ggml_hpx_mode::decode,
        /*graph_shape_hash=*/0x101,
        /*backend_assign_hash=*/0x201,
        /*workspace_layout_sig=*/0x301,
        /*policy_version=*/7);

    auto const topo2 = make_decode_topology();
    auto const key2  = make_key(
        ggml_hpx_mode::decode,
        /*graph_shape_hash=*/0x102,
        /*backend_assign_hash=*/0x202,
        /*workspace_layout_sig=*/0x302,
        /*policy_version=*/8);

    auto const* first =
        ggml_hpx_cache_insert_decode(
            cache, ggml_hpx_build_decode_plan(topo1, key1));
    ASSERT_NE(first, nullptr);
    ASSERT_NE(ggml_hpx_cache_lookup_decode(cache, key1), nullptr);

    auto const* second =
        ggml_hpx_cache_insert_decode(
            cache, ggml_hpx_build_decode_plan(topo2, key2));
    ASSERT_NE(second, nullptr);

    // Old entry is gone, new one is present.
    EXPECT_EQ(ggml_hpx_cache_lookup_decode(cache, key1), nullptr);

    auto const* looked_up = ggml_hpx_cache_lookup_decode(cache, key2);
    ASSERT_NE(looked_up, nullptr);
    EXPECT_EQ(looked_up, second);
    EXPECT_EQ(looked_up->key, key2);
}

TEST_F(HpxCacheTest, InsertSecondPrefillPlanReplacesFirst)
{
    auto const topo1 = make_prefill_topology();
    auto const key1  = make_key(
        ggml_hpx_mode::prefill,
        /*graph_shape_hash=*/0x401,
        /*backend_assign_hash=*/0x501,
        /*workspace_layout_sig=*/0x601,
        /*policy_version=*/7);

    auto const topo2 = make_prefill_topology();
    auto const key2  = make_key(
        ggml_hpx_mode::prefill,
        /*graph_shape_hash=*/0x402,
        /*backend_assign_hash=*/0x502,
        /*workspace_layout_sig=*/0x602,
        /*policy_version=*/8);

    auto const* first =
        ggml_hpx_cache_insert_prefill(
            cache, ggml_hpx_build_prefill_plan(topo1, key1));
    ASSERT_NE(first, nullptr);
    ASSERT_NE(ggml_hpx_cache_lookup_prefill(cache, key1), nullptr);

    auto const* second =
        ggml_hpx_cache_insert_prefill(
            cache, ggml_hpx_build_prefill_plan(topo2, key2));
    ASSERT_NE(second, nullptr);

    // Old entry is gone, new one is present.
    EXPECT_EQ(ggml_hpx_cache_lookup_prefill(cache, key1), nullptr);

    auto const* looked_up = ggml_hpx_cache_lookup_prefill(cache, key2);
    ASSERT_NE(looked_up, nullptr);
    EXPECT_EQ(looked_up, second);
    EXPECT_EQ(looked_up->key, key2);
}
