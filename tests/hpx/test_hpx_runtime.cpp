#include <gtest/gtest.h>

#include "ggml-hpx-runtime.h"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <future>
#include <thread>
#include <vector>

namespace
{

ggml_hpx_runtime_params make_params(
    uint32_t n_decode_threads = 2,
    uint32_t n_prefill_threads = 3,
    size_t initial_scratch_bytes = 0)
{
    ggml_hpx_runtime_params params{};
    params.n_decode_threads    = n_decode_threads;
    params.n_prefill_threads   = n_prefill_threads;
    params.initial_scratch_bytes = initial_scratch_bytes;
    return params;
}

struct dispatch_state
{
    explicit dispatch_state(uint32_t n_chunks)
        : counts(n_chunks)
    {
    }

    std::vector<std::atomic<uint32_t>> counts;
    std::atomic<uint32_t> total{0};
    std::atomic<uint32_t> out_of_range{0};
};

void dispatch_counter_fn(uint32_t chunk_idx, void* user_data)
{
    auto* const state = static_cast<dispatch_state*>(user_data);

    state->total.fetch_add(1, std::memory_order_relaxed);

    if (static_cast<size_t>(chunk_idx) >= state->counts.size())
    {
        state->out_of_range.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    state->counts[chunk_idx].fetch_add(1, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// Helpers for blocking / single-chunk dispatch tests
// ---------------------------------------------------------------------------

template <typename Pred>
bool spin_until(
    Pred pred,
    std::chrono::milliseconds timeout = std::chrono::milliseconds(2000))
{
    auto const deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline)
    {
        if (pred())
        {
            return true;
        }
        std::this_thread::yield();
    }
    return pred();
}

struct blocking_dispatch_state
{
    std::atomic<int> started{0};
    std::atomic<int> finished{0};
    std::atomic<int> in_flight{0};
    std::atomic<int> peak_in_flight{0};

    std::promise<void>      release_promise;
    std::shared_future<void> release_future{
        release_promise.get_future().share()};
};

void blocking_dispatch_fn(uint32_t /*i*/, void* user_data)
{
    auto* const ctx = static_cast<blocking_dispatch_state*>(user_data);

    int const now =
        1 + ctx->in_flight.fetch_add(1, std::memory_order_acq_rel);

    int peak = ctx->peak_in_flight.load(std::memory_order_relaxed);
    while (now > peak &&
           !ctx->peak_in_flight.compare_exchange_weak(
               peak, now,
               std::memory_order_acq_rel,
               std::memory_order_relaxed))
    {
    }

    ctx->started.fetch_add(1, std::memory_order_acq_rel);
    ctx->release_future.wait();
    ctx->finished.fetch_add(1, std::memory_order_acq_rel);
    ctx->in_flight.fetch_sub(1, std::memory_order_acq_rel);
}

struct single_chunk_state
{
    std::atomic<uint32_t> calls{0};
    std::atomic<uint32_t> last_chunk{UINT32_C(0xffffffff)};
};

void single_chunk_fn(uint32_t i, void* user_data)
{
    auto* const ctx = static_cast<single_chunk_state*>(user_data);
    ctx->calls.fetch_add(1, std::memory_order_acq_rel);
    ctx->last_chunk.store(i, std::memory_order_release);
}

}    // namespace

TEST(HpxRuntime, CreateDestroySucceeds)
{
    auto const params = make_params();

    auto* const rt = ggml_hpx_runtime_create(params);
    ASSERT_NE(rt, nullptr);

    ggml_hpx_runtime_destroy(rt);
}

TEST(HpxRuntime, ScratchEnsureAllocatesAndGrows)
{
    auto const params = make_params(
        /*n_decode_threads=*/2,
        /*n_prefill_threads=*/2,
        /*initial_scratch_bytes=*/0);

    auto* const rt = ggml_hpx_runtime_create(params);
    ASSERT_NE(rt, nullptr);

    EXPECT_EQ(ggml_hpx_runtime_scratch_ptr(rt), nullptr);

    ggml_hpx_runtime_scratch_ensure(rt, 64);
    void* const p1 = ggml_hpx_runtime_scratch_ptr(rt);
    ASSERT_NE(p1, nullptr);

    // If scratch_ensure(64) worked, writing 64 bytes should be valid.
    std::memset(p1, 0xA5, 64);

    ggml_hpx_runtime_scratch_ensure(rt, 4096);
    void* const p2 = ggml_hpx_runtime_scratch_ptr(rt);
    ASSERT_NE(p2, nullptr);

    // Growth may reallocate or may grow in place. Both are valid.
    // Writing 4096 bytes checks that the requested capacity exists.
    std::memset(p2, 0x5A, 4096);

    ggml_hpx_runtime_destroy(rt);
}

TEST(HpxRuntime, ScratchPtrIsStableUntilNextGrow)
{
    auto const params = make_params(
        /*n_decode_threads=*/2,
        /*n_prefill_threads=*/2,
        /*initial_scratch_bytes=*/0);

    auto* const rt = ggml_hpx_runtime_create(params);
    ASSERT_NE(rt, nullptr);

    ggml_hpx_runtime_scratch_ensure(rt, 128);
    void* const p1 = ggml_hpx_runtime_scratch_ptr(rt);
    ASSERT_NE(p1, nullptr);

    void* const p2 = ggml_hpx_runtime_scratch_ptr(rt);
    EXPECT_EQ(p1, p2);

    // No grow required, so the pointer must stay stable.
    ggml_hpx_runtime_scratch_ensure(rt, 64);
    void* const p3 = ggml_hpx_runtime_scratch_ptr(rt);
    EXPECT_EQ(p1, p3);

    ggml_hpx_runtime_scratch_ensure(rt, 128);
    void* const p4 = ggml_hpx_runtime_scratch_ptr(rt);
    EXPECT_EQ(p1, p4);

    // After a real grow request, the pointer may stay the same or change.
    ggml_hpx_runtime_scratch_ensure(rt, 512);
    void* const p5 = ggml_hpx_runtime_scratch_ptr(rt);
    ASSERT_NE(p5, nullptr);

    ggml_hpx_runtime_destroy(rt);
}

TEST(HpxRuntime, DispatchDecodeRunsEveryChunkExactlyOnce)
{
    auto const params = make_params(
        /*n_decode_threads=*/3,
        /*n_prefill_threads=*/2,
        /*initial_scratch_bytes=*/0);

    auto* const rt = ggml_hpx_runtime_create(params);
    ASSERT_NE(rt, nullptr);

    uint32_t const n_chunks = 17;
    dispatch_state state(n_chunks);

    ggml_hpx_runtime_dispatch_decode(
        rt, n_chunks, dispatch_counter_fn, &state);

    EXPECT_EQ(state.total.load(std::memory_order_relaxed), n_chunks);
    EXPECT_EQ(state.out_of_range.load(std::memory_order_relaxed), 0U);

    for (uint32_t i = 0; i < n_chunks; ++i)
    {
        EXPECT_EQ(
            state.counts[i].load(std::memory_order_relaxed), 1U);
    }

    ggml_hpx_runtime_destroy(rt);
}

TEST(HpxRuntime, DispatchPrefillRunsEveryChunkExactlyOnce)
{
    auto const params = make_params(
        /*n_decode_threads=*/2,
        /*n_prefill_threads=*/4,
        /*initial_scratch_bytes=*/0);

    auto* const rt = ggml_hpx_runtime_create(params);
    ASSERT_NE(rt, nullptr);

    uint32_t const n_chunks = 23;
    dispatch_state state(n_chunks);

    ggml_hpx_runtime_dispatch_prefill(
        rt, n_chunks, dispatch_counter_fn, &state);

    EXPECT_EQ(state.total.load(std::memory_order_relaxed), n_chunks);
    EXPECT_EQ(state.out_of_range.load(std::memory_order_relaxed), 0U);

    for (uint32_t i = 0; i < n_chunks; ++i)
    {
        EXPECT_EQ(
            state.counts[i].load(std::memory_order_relaxed), 1U);
    }

    ggml_hpx_runtime_destroy(rt);
}

// Prefill dispatch is currently sequential (parallel dispatch requires
// dependency-aware scheduling; see ggml-hpx-runtime.cpp).
// Verify: all 3 chunks run in order (0, 1, 2) and dispatch is synchronous.
TEST(HpxRuntime, DispatchPrefillRunsChunksSequentiallyAndSynchronously)
{
    auto const params = make_params(
        /*n_decode_threads=*/2,
        /*n_prefill_threads=*/4,
        /*initial_scratch_bytes=*/0);

    auto* const rt = ggml_hpx_runtime_create(params);
    ASSERT_NE(rt, nullptr);

    // Record the order in which chunks execute.
    std::vector<uint32_t> order;
    order.reserve(3);

    auto const fn = [](uint32_t chunk_idx, void* user_data) {
        auto* v = static_cast<std::vector<uint32_t>*>(user_data);
        v->push_back(chunk_idx);
    };

    ggml_hpx_runtime_dispatch_prefill(rt, /*n_chunks=*/3, fn, &order);

    // Dispatch is synchronous: all chunks must have finished by the time
    // dispatch_prefill returns.
    ASSERT_EQ(order.size(), 3u);
    EXPECT_EQ(order[0], 0u);
    EXPECT_EQ(order[1], 1u);
    EXPECT_EQ(order[2], 2u);

    ggml_hpx_runtime_destroy(rt);
}

TEST(HpxRuntime, DispatchDecodeSingleChunkRunsExactlyOnce)
{
    auto const params = make_params(
        /*n_decode_threads=*/3,
        /*n_prefill_threads=*/2,
        /*initial_scratch_bytes=*/0);

    auto* const rt = ggml_hpx_runtime_create(params);
    ASSERT_NE(rt, nullptr);

    single_chunk_state state;

    ggml_hpx_runtime_dispatch_decode(
        rt, /*n_chunks=*/1, single_chunk_fn, &state);

    EXPECT_EQ(state.calls.load(std::memory_order_relaxed), 1U);
    EXPECT_EQ(state.last_chunk.load(std::memory_order_relaxed), 0U);

    ggml_hpx_runtime_destroy(rt);
}

TEST(HpxRuntime, DispatchPrefillSingleChunkRunsExactlyOnce)
{
    auto const params = make_params(
        /*n_decode_threads=*/2,
        /*n_prefill_threads=*/4,
        /*initial_scratch_bytes=*/0);

    auto* const rt = ggml_hpx_runtime_create(params);
    ASSERT_NE(rt, nullptr);

    single_chunk_state state;

    ggml_hpx_runtime_dispatch_prefill(
        rt, /*n_chunks=*/1, single_chunk_fn, &state);

    EXPECT_EQ(state.calls.load(std::memory_order_relaxed), 1U);
    EXPECT_EQ(state.last_chunk.load(std::memory_order_relaxed), 0U);

    ggml_hpx_runtime_destroy(rt);
}
