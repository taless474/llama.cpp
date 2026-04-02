// ggml-cpu-hpx.cpp — HPX thread pool implementation for ggml
//
// Compiled only when GGML_USE_HPX is defined (set by CMake via GGML_HPX=ON).
// Provides extern "C" replacements for all pthread-specific functions that are
// guarded by #if !defined(GGML_USE_HPX) in ggml-cpu.c.
//
// Design mirrors the pthread baseline:
//   - Workers 0..N-1 all run as HPX tasks per graph dispatch.
//   - ggml_barrier() uses hpx::barrier<> (cooperative suspend, not spin-wait).
//     hpx::barrier requires the caller to be an HPX thread — worker 0 is
//     therefore also an HPX task (not run inline on the calling OS thread).
//   - Pool-level control (pause/resume/stop/kickoff) uses std::mutex +
//     std::condition_variable so it is callable from any thread, including the
//     main OS thread.  hpx::mutex/condition_variable require HPX thread context.
//
// Note: do NOT include <stdatomic.h> here.  ggml-cpu-threads.h provides
// atomic_int / atomic_bool as std::atomic<> aliases in C++ mode, which avoids
// the libc++ / clang <stdatomic.h> macro conflict with <atomic>.

#include "ggml-cpu-threads.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"

#include <condition_variable>
#include <cstring>
#include <mutex>

#include <hpx/future.hpp>
#include <hpx/init.hpp>
#include <hpx/synchronization/barrier.hpp>

// ── Accessor helpers ──────────────────────────────────────────────────────────
// hpx_mutex / hpx_cond are std::mutex* / std::condition_variable* in this impl.

static std::mutex & tp_mutex(ggml_threadpool * tp) {
    return *static_cast<std::mutex *>(tp->hpx_mutex);
}

static std::condition_variable & tp_cond(ggml_threadpool * tp) {
    return *static_cast<std::condition_variable *>(tp->hpx_cond);
}

static hpx::barrier<> & tp_barrier(ggml_threadpool * tp) {
    return *static_cast<hpx::barrier<> *>(tp->hpx_barrier);
}

// ── HPX runtime singleton ─────────────────────────────────────────────────────
//
// Constructed at static-init time (before main()), destroyed at exit.
// The (void)g_hpx_runtime reference in ggml_threadpool_new_impl_hpx prevents
// the linker from dead-stripping this TU.

namespace {

struct HpxRuntime {
    HpxRuntime() {
        hpx::init_params p;
        hpx::start(nullptr, 0, nullptr, p);
    }

    ~HpxRuntime() {
        hpx::finalize();
        hpx::stop();
    }
};

static HpxRuntime g_hpx_runtime;  // NOLINT(cert-err58-cpp)

}  // namespace

// ── ggml_barrier ─────────────────────────────────────────────────────────────
// Must be called from an HPX thread — guaranteed because all workers including
// worker 0 are dispatched as HPX tasks during graph compute.

extern "C" void ggml_barrier(ggml_threadpool * tp) {
    int n_threads = tp->n_graph.load(std::memory_order_relaxed) & GGML_THREADPOOL_N_THREADS_MASK;
    if (n_threads == 1) {
        return;
    }
    tp_barrier(tp).arrive_and_wait();
}

// ── Worker loop (persistent workers 1..N-1) ───────────────────────────────────

// Wait until a new graph is kicked, the pool is stopped, or pause is cleared.
// Returns state->pending.
static bool hpx_check_for_work(ggml_compute_state * state) {
    ggml_threadpool * tp = state->threadpool;
    std::unique_lock<std::mutex> lk(tp_mutex(tp));
    tp_cond(tp).wait(lk, [&] {
        if (tp->stop.load(std::memory_order_relaxed) || tp->pause.load(std::memory_order_relaxed)) {
            return true;
        }
        int ng = tp->n_graph.load(std::memory_order_relaxed);
        if (ng != state->last_graph) {
            int n_threads     = ng & GGML_THREADPOOL_N_THREADS_MASK;
            state->pending    = (state->ith < n_threads);
            state->last_graph = ng;
            return true;
        }
        return false;
    });
    return state->pending;
}

// Persistent HPX task body for workers 1..N-1.
static void hpx_worker_thread(ggml_compute_state * state) {
    while (true) {
        // Wait while paused.
        {
            std::unique_lock<std::mutex> lk(tp_mutex(state->threadpool));
            tp_cond(state->threadpool).wait(lk, [&] {
                return !state->threadpool->pause.load(std::memory_order_relaxed);
            });
        }

        if (state->threadpool->stop.load(std::memory_order_relaxed)) {
            break;
        }

        hpx_check_for_work(state);
        if (state->pending) {
            state->pending = false;
            ggml_graph_compute_thread(state);
        }
    }
}

// ── Kickoff ───────────────────────────────────────────────────────────────────

static void ggml_graph_compute_kickoff_hpx(ggml_threadpool * tp, int n_threads) {
    std::unique_lock<std::mutex> lk(tp_mutex(tp));

    int n_graph = tp->n_graph.load(std::memory_order_relaxed) >> GGML_THREADPOOL_N_THREADS_BITS;
    n_graph     = ((n_graph + 1) << GGML_THREADPOOL_N_THREADS_BITS) | (n_threads & GGML_THREADPOOL_N_THREADS_MASK);
    tp->n_graph.store(n_graph, std::memory_order_seq_cst);

    // hpx::barrier count is fixed at construction; recreate when n_threads changes.
    // Cheap: once per graph dispatch, not per node barrier.
    delete static_cast<hpx::barrier<> *>(tp->hpx_barrier);
    tp->hpx_barrier = new hpx::barrier<>(static_cast<std::ptrdiff_t>(n_threads));

    tp_cond(tp).notify_all();
}

// ── ggml_graph_compute_hpx (entry point called from ggml-cpu.c) ──────────────

extern "C" enum ggml_status ggml_graph_compute_hpx(struct ggml_cgraph     * /*cgraph*/,
                                                    struct ggml_cplan      * /*cplan*/,
                                                    struct ggml_threadpool * tp,
                                                    int                      n_threads) {
    // cgraph/cplan are already set on tp by ggml_graph_compute() before calling here.
    if (n_threads > tp->n_threads) {
        GGML_LOG_WARN("cplan requested more threads (%d) than available (%d)\n", n_threads, tp->n_threads);
        n_threads = tp->n_threads;
    }

    ggml_graph_compute_kickoff_hpx(tp, n_threads);

    // Worker 0 MUST run as an HPX task, not inline on this OS thread, because
    // ggml_barrier() calls hpx::barrier::arrive_and_wait() which requires an
    // HPX thread context.  We block this OS thread on fut0.get() — hpx::future
    // supports get() from non-HPX threads (OS-level block).
    auto fut0 = hpx::async([tp] { ggml_graph_compute_thread(&tp->workers[0]); });
    fut0.get();

    return tp->ec;
}

// ── ggml_threadpool_new / free ────────────────────────────────────────────────

static ggml_threadpool * ggml_threadpool_new_impl_hpx(
    ggml_threadpool_params * tpp, ggml_cgraph * cgraph, ggml_cplan * cplan) {
    (void) g_hpx_runtime;  // keep TU alive — prevents linker dead-stripping

    auto * tp = static_cast<ggml_threadpool *>(ggml_aligned_malloc(sizeof(ggml_threadpool)));
    memset(tp, 0, sizeof(*tp));

    tp->cgraph    = cgraph;
    tp->cplan     = cplan;
    tp->n_threads = tpp->n_threads;
    tp->prio      = tpp->prio;
    tp->poll      = tpp->poll;
    tp->ec        = GGML_STATUS_SUCCESS;
    tp->n_graph.store(0, std::memory_order_relaxed);
    tp->current_chunk.store(0, std::memory_order_relaxed);
    tp->abort.store(-1, std::memory_order_relaxed);
    tp->stop.store(false, std::memory_order_relaxed);
    tp->pause.store(static_cast<bool>(tpp->paused), std::memory_order_relaxed);

    tp->hpx_mutex   = new std::mutex();
    tp->hpx_cond    = new std::condition_variable();
    tp->hpx_barrier = new hpx::barrier<>(static_cast<std::ptrdiff_t>(tpp->n_threads));

    const size_t wsz     = sizeof(ggml_compute_state) * tpp->n_threads;
    auto *       workers = static_cast<ggml_compute_state *>(ggml_aligned_malloc(wsz));
    memset(workers, 0, wsz);

    for (int j = 0; j < tpp->n_threads; j++) {
        workers[j].threadpool = tp;
        workers[j].ith        = j;
    }
    tp->workers = workers;

    // Spawn persistent HPX tasks for workers 1..N-1.
    // Worker 0 is dispatched per-graph in ggml_graph_compute_hpx.
    for (int j = 1; j < tpp->n_threads; j++) {
        auto fut              = hpx::async(&hpx_worker_thread, &workers[j]);
        workers[j].hpx_future = new hpx::future<void>(std::move(fut));
    }

    return tp;
}

extern "C" ggml_threadpool * ggml_threadpool_new(ggml_threadpool_params * tpp) {
    return ggml_threadpool_new_impl_hpx(tpp, nullptr, nullptr);
}

extern "C" void ggml_threadpool_free(ggml_threadpool * tp) {
    if (!tp) {
        return;
    }

    {
        std::unique_lock<std::mutex> lk(tp_mutex(tp));
        tp->stop.store(true, std::memory_order_relaxed);
        tp->pause.store(false, std::memory_order_relaxed);
        tp_cond(tp).notify_all();
    }

    for (int j = 1; j < tp->n_threads; j++) {
        auto * fut = static_cast<hpx::future<void> *>(tp->workers[j].hpx_future);
        if (fut) {
            fut->get();
            delete fut;
            tp->workers[j].hpx_future = nullptr;
        }
    }

    delete static_cast<hpx::barrier<> *>(tp->hpx_barrier);
    delete static_cast<std::condition_variable *>(tp->hpx_cond);
    delete static_cast<std::mutex *>(tp->hpx_mutex);

    const size_t wsz = sizeof(ggml_compute_state) * tp->n_threads;
    ggml_aligned_free(tp->workers, wsz);
    ggml_aligned_free(tp, sizeof(ggml_threadpool));
}

// ── pause / resume ────────────────────────────────────────────────────────────

extern "C" void ggml_threadpool_pause(ggml_threadpool * tp) {
    std::unique_lock<std::mutex> lk(tp_mutex(tp));
    tp->pause.store(true, std::memory_order_relaxed);
    tp_cond(tp).notify_all();
}

extern "C" void ggml_threadpool_resume(ggml_threadpool * tp) {
    std::unique_lock<std::mutex> lk(tp_mutex(tp));
    tp->pause.store(false, std::memory_order_relaxed);
    tp_cond(tp).notify_all();
}
