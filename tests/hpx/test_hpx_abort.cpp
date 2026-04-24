#include <gtest/gtest.h>

#include "ggml-hpx-abort.h"

// ggml_hpx_abort_token contract:
//   default state  -- check() returns false
//   request()      -- sets the flag; subsequent check() returns true
//   reset()        -- clears the flag; subsequent check() returns false
//   request()      -- idempotent; multiple calls leave check() true
//   reset()        -- harmless when flag is already clear

TEST(HpxAbortToken, DefaultStateIsNotAborted)
{
    ggml_hpx_abort_token token{};
    EXPECT_FALSE(token.check());
}

TEST(HpxAbortToken, RequestSetsFlag)
{
    ggml_hpx_abort_token token{};
    token.request();
    EXPECT_TRUE(token.check());
}

TEST(HpxAbortToken, ResetClearsFlag)
{
    ggml_hpx_abort_token token{};
    token.request();
    ASSERT_TRUE(token.check());

    token.reset();
    EXPECT_FALSE(token.check());
}

TEST(HpxAbortToken, ResetWithoutPriorRequestIsHarmless)
{
    ggml_hpx_abort_token token{};
    token.reset();
    EXPECT_FALSE(token.check());
}

TEST(HpxAbortToken, MultipleRequestsAreIdempotent)
{
    ggml_hpx_abort_token token{};
    token.request();
    token.request();
    token.request();
    EXPECT_TRUE(token.check());
}

TEST(HpxAbortToken, RequestResetRequestCycle)
{
    ggml_hpx_abort_token token{};

    token.request();
    EXPECT_TRUE(token.check());

    token.reset();
    EXPECT_FALSE(token.check());

    // A second request after reset must work exactly like the first.
    token.request();
    EXPECT_TRUE(token.check());
}

TEST(HpxAbortToken, ResetAfterMultipleRequestsClearsFlag)
{
    ggml_hpx_abort_token token{};
    token.request();
    token.request();

    token.reset();
    EXPECT_FALSE(token.check());
}
