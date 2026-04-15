// ggml-hpx-tpool.cpp
//
// HPX executor ops — bulk-region variant.
//
// The HPX backend owns a reusable execution context (scheduler_executor
// wrapping a thread_pool_scheduler) created once at init time with an
// explicit thread-count placement hint.
//
// Execution model
// ───────────────
// run_job submits one hpx::experimental::for_loop over logical worker IDs
// [0, n_threads) to the owned executor.  Each HPX task calls
// ggml_graph_compute_thread_run for its worker ID, which iterates the full
// graph and synchronises between nodes via ggml's atomic spin barrier.
//
// hpx::experimental::for_loop with hpx::execution::par dispatches all
// n_threads tasks before any complete (bulk semantics), so every barrier
// cycle sees all n_active_threads participants arrive.  The call blocks the
// caller until all iterations complete, so run_job returns only after the
// final barrier has passed.
//
// There are no persistent worker tasks, no semaphores, and no coroutine
// guard.  Thread placement is delegated to the HPX runtime via the
// with_processing_units_count property on the scheduler.

#include "ggml-hpx-tpool.h"
#include "ggml-hpx-runtime.h"    // ggml_hpx_tpool_start

#include <hpx/algorithm.hpp>
#include <hpx/execution.hpp>
#include <hpx/executors/scheduler_executor.hpp>
#include <hpx/executors/thread_pool_scheduler.hpp>

#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>

// ---------------------------------------------------------------------------
// Owned execution context
// ---------------------------------------------------------------------------

namespace
{

using Scheduler =
    hpx::execution::experimental::thread_pool_scheduler;

using Exec =
    hpx::execution::experimental::scheduler_executor<Scheduler>;

// ---------------------------------------------------------------------------
// Dispatch histogram
//
// Records cgraph->n_nodes × n_threads for every run_job call.
// Printed to stderr in hpx_destroy.  Purely observational; zero effect on
// dispatch behaviour.
// ---------------------------------------------------------------------------

static constexpr int kNThreadTiers = 4;   // t=1, t=2, t=4, other

// n_nodes histogram (graph topology; batch-size invariant for llama graphs)
static constexpr int kNNodeBuckets                     = 7;
static constexpr int kNodeEdges[kNNodeBuckets - 1]     = {8, 16, 32, 64, 128, 256};
static constexpr char const* kNodeLabels[kNNodeBuckets] = {
    "    <8", "  8-15", " 16-31", " 32-63", " 64-127", "128-255", "   >=256",
};

// work_size histogram (scratch bytes; varies with batch/sequence length)
static constexpr int    kNWorkBuckets                      = 6;
static constexpr size_t kWorkEdges[kNWorkBuckets - 1]      = {
    1, 4096, 65536, 1048576, 16777216
};   // 0 / 1–4K / 4K–64K / 64K–1M / 1M–16M / >=16M
static constexpr char const* kWorkLabels[kNWorkBuckets] = {
    "       0", "  1-4K", " 4K-64K", " 64K-1M", " 1M-16M", "  >=16M",
};

static int thread_tier(int n_threads) noexcept
{
    if (n_threads == 1) return 0;
    if (n_threads == 2) return 1;
    if (n_threads == 4) return 2;
    return 3;
}

static int node_bucket(int n_nodes) noexcept
{
    for (int k = 0; k < kNNodeBuckets - 1; ++k)
        if (n_nodes < kNodeEdges[k])
            return k;
    return kNNodeBuckets - 1;
}

static int work_bucket(std::size_t work_size) noexcept
{
    for (int k = 0; k < kNWorkBuckets - 1; ++k)
        if (work_size < kWorkEdges[k])
            return k;
    return kNWorkBuckets - 1;
}

struct HpxTpoolState
{
    // Reusable execution context: n_threads workers, HPX-topology placement.
    // Constructed once in hpx_init; every run_job dispatch uses this object.
    Exec exec;

    // n_nodes histogram: [thread_tier][node_bucket] dispatch counts.
    std::int64_t node_hist[kNThreadTiers][kNNodeBuckets] = {};
    // work_size histogram: [thread_tier][work_bucket] dispatch counts.
    std::int64_t work_hist[kNThreadTiers][kNWorkBuckets] = {};
    std::int64_t tier_total[kNThreadTiers]               = {};

    explicit HpxTpoolState(int n_threads)
      : exec(hpx::parallel::execution::with_processing_units_count(
            Scheduler{},
            static_cast<std::size_t>(n_threads)))
    {
    }
};

}    // namespace

// ---------------------------------------------------------------------------
// Ops
// ---------------------------------------------------------------------------

static void hpx_init(
    struct ggml_threadpool *        tp,
    struct ggml_threadpool_params * /*tpp*/)
{
    int n = ggml_threadpool_n_threads(tp);
    ggml_threadpool_set_priv(tp, n > 0 ? new HpxTpoolState{n} : nullptr);
}

static void hpx_run_job(struct ggml_threadpool * tp, int n_threads)
{
    auto * s = static_cast<HpxTpoolState *>(ggml_threadpool_get_priv(tp));

    // Hoist ws and n_nodes so both the probe and the fallback gate share them.
    std::size_t const ws      = ggml_threadpool_work_size(tp);
    int const         n_nodes = ggml_threadpool_n_nodes(tp);

    // Probe: record every dispatch in the histograms before any early return.
    if (s)
    {
        int const ti = thread_tier(n_threads);
        if (n_nodes >= 0)
            ++s->node_hist[ti][node_bucket(n_nodes)];
        if (ws != SIZE_MAX)
            ++s->work_hist[ti][work_bucket(ws)];
        ++s->tier_total[ti];
    }

    // Fast path: no HPX overhead for single-threaded dispatch.
    if (n_threads == 1)
    {
        ggml_graph_compute_thread_run(ggml_threadpool_worker(tp, 0));
        return;
    }

    // Bulk dispatch: all n_threads tasks are submitted simultaneously so the
    // spin barrier inside ggml_graph_compute_thread_run sees every participant.
    hpx::experimental::for_loop(
        hpx::execution::par.on(s->exec),
        0,
        n_threads,
        [=](int j) {
            ggml_graph_compute_thread_run(ggml_threadpool_worker(tp, j));
        });
}

static void hpx_destroy(struct ggml_threadpool * tp)
{
    auto * s = static_cast<HpxTpoolState *>(ggml_threadpool_get_priv(tp));
    if (s)
    {
        // Compute overall total.
        std::int64_t grand_total = 0;
        for (int ti = 0; ti < kNThreadTiers; ++ti)
            grand_total += s->tier_total[ti];

        static constexpr char const* kTierLabels[kNThreadTiers] = {
            "t=1", "t=2", "t=4", "t=other"
        };

        std::fprintf(stderr,
            "[hpx-tpool] dispatch histogram  grand_total=%" PRId64 "\n",
            grand_total);

        for (int ti = 0; ti < kNThreadTiers; ++ti)
        {
            if (s->tier_total[ti] == 0)
                continue;

            double const tot = static_cast<double>(s->tier_total[ti]);
            std::fprintf(stderr, "  %s  dispatches=%" PRId64 "\n",
                kTierLabels[ti], s->tier_total[ti]);

            std::fprintf(stderr, "    -- n_nodes --\n");
            for (int bi = 0; bi < kNNodeBuckets; ++bi)
            {
                std::int64_t const cnt = s->node_hist[ti][bi];
                std::fprintf(stderr,
                    "      %-8s  %6" PRId64 "  (%5.1f%%)\n",
                    kNodeLabels[bi], cnt,
                    static_cast<double>(cnt) / tot * 100.0);
            }

            std::fprintf(stderr, "    -- work_size --\n");
            for (int bi = 0; bi < kNWorkBuckets; ++bi)
            {
                std::int64_t const cnt = s->work_hist[ti][bi];
                std::fprintf(stderr,
                    "      %-8s  %6" PRId64 "  (%5.1f%%)\n",
                    kWorkLabels[bi], cnt,
                    static_cast<double>(cnt) / tot * 100.0);
            }
        }
        std::fflush(stderr);
    }

    delete s;
    ggml_threadpool_set_priv(tp, nullptr);
    ggml_threadpool_destroy_substrate(tp);
}

// ---------------------------------------------------------------------------
// Ops table
// ---------------------------------------------------------------------------

static const struct ggml_cpu_executor_ops ggml_hpx_executor_ops = {
    /* .init    = */ hpx_init,
    /* .run_job = */ hpx_run_job,
    /* .destroy = */ hpx_destroy,
};

extern "C" const struct ggml_cpu_executor_ops * ggml_hpx_tpool_get_ops()
{
    return &ggml_hpx_executor_ops;
}

struct ggml_threadpool * ggml_hpx_tpool_create(
    struct ggml_threadpool_params * tpp)
{
    // Start HPX runtime if not already running (idempotent).
    // Does not touch g_executor_ops.
    ggml_hpx_tpool_start();
    return ggml_threadpool_new_with_ops(tpp, &ggml_hpx_executor_ops);
}
