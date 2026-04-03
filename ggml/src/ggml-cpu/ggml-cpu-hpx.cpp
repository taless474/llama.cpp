// ggml-cpu-hpx.cpp — HPX thread pool implementation for ggml
//
// Compiled only when GGML_USE_HPX is defined (set by CMake via GGML_HPX=ON).
// Provides extern "C" replacements for all pthread-specific functions that are
// guarded by #if !defined(GGML_USE_HPX) in ggml-cpu.c.
//
// Design:
//   - All N workers are spawned fresh as HPX tasks per graph dispatch.
//     There are no persistent idle worker tasks between dispatches.
//   - ggml_barrier() uses atomic n_barrier / n_barrier_passed counters (same
//     fields as the pthread implementation) with hpx::this_thread::yield() for
//     cooperative waiting.  This suspends the HPX *task* (coroutine) without
//     blocking the underlying OS thread, so the OS thread remains available to
//     run the other N-1 worker tasks.
//   - Pool-level pause/resume is a simple atomic flag checked at dispatch entry.
//     There are no idle workers to signal.
//
// Why not persistent workers?
//   Persistent HPX tasks that yield-loop while idle hold scheduler slots and
//   generate heavy std::mutex contention (lock+unlock on every yield iteration).
//   std::mutex contention causes OS-thread blocking which can starve the HPX
//   scheduler — the same root cause as the original condvar crash, just slower.
//   Per-dispatch tasks eliminate this entirely.
//
// Why not hpx::barrier<>?
//   hpx::barrier<> requires a fixed participant count at construction.  ggml
//   supports variable n_threads per dispatch, so the old code deleted and
//   recreated the barrier every graph — fragile and allocation-heavy.  The
//   atomic counter approach has zero allocation cost and no fixed-count issue.

#include "ggml-cpu-threads.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"

#include <cstring>
#include <thread>
#include <vector>

#include <hpx/future.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/program_options.hpp>
#include <hpx/thread.hpp>

// ── HPX runtime lifecycle ─────────────────────────────────────────────────────
//
// HPX is started once per process on the first ggml_threadpool_new call and is
// NOT stopped from ggml_threadpool_free.  With per-dispatch tasks there are no
// lingering HPX tasks between dispatches, so atexit shutdown is clean.

namespace {

std::once_flag g_hpx_start_flag;

void hpx_start_once() {
    std::call_once(g_hpx_start_flag, []() {
        hpx::init_params p;
        hpx::start([](hpx::program_options::variables_map&) -> int { return 0; },
                   0, nullptr, p);
    });
}

}  // namespace

// ── ggml_barrier ─────────────────────────────────────────────────────────────
//
// Mirrors the pthread barrier implementation but replaces the OS-level
// spin/sleep with hpx::this_thread::yield().  yield() suspends only the HPX
// task (coroutine), not the OS thread, so the OS thread can immediately run
// the next ready HPX task (e.g., another worker in the same dispatch).
//
// n_barrier and n_barrier_passed are the same atomic fields used by the pthread
// path — now unconditional in the struct so both backends can use them.

extern "C" void ggml_barrier(ggml_threadpool * tp) {
    int n_threads = tp->n_graph.load(std::memory_order_relaxed) & GGML_THREADPOOL_N_THREADS_MASK;
    if (n_threads == 1) {
        return;
    }

    int pass_before = tp->n_barrier_passed.load(std::memory_order_relaxed);

    if (tp->n_barrier.fetch_add(1, std::memory_order_acq_rel) + 1 == n_threads) {
        // Last to arrive: reset arrival counter and advance the generation.
        tp->n_barrier.store(0, std::memory_order_release);
        tp->n_barrier_passed.fetch_add(1, std::memory_order_acq_rel);
    } else {
        // Wait for the last arrival to advance the generation.
        while (tp->n_barrier_passed.load(std::memory_order_acquire) == pass_before) {
            hpx::this_thread::yield();
        }
    }

    std::atomic_thread_fence(std::memory_order_acquire);
}

// ── ggml_graph_compute_hpx (entry point called from ggml-cpu.c) ──────────────
//
// Spawns N fresh HPX tasks (one per thread slot) and waits for all to complete.
// Worker tasks run ggml_graph_compute_thread() which calls ggml_barrier() at
// inter-node boundaries; ggml_barrier() uses hpx::this_thread::yield() so no
// OS thread is ever blocked inside an HPX task.

extern "C" enum ggml_status ggml_graph_compute_hpx(struct ggml_cgraph     * /*cgraph*/,
                                                    struct ggml_cplan      * /*cplan*/,
                                                    struct ggml_threadpool * tp,
                                                    int                      n_threads) {
    // Honour pause: spin on the main (OS) thread until resumed.
    // Spinning on the main thread is fine — it is not an HPX task.
    while (tp->pause.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }

    if (n_threads > tp->n_threads) {
        GGML_LOG_WARN("cplan requested more threads (%d) than available (%d)\n", n_threads, tp->n_threads);
        n_threads = tp->n_threads;
    }

    // Update n_graph: increment generation counter, set active thread count.
    int n_graph = tp->n_graph.load(std::memory_order_relaxed) >> GGML_THREADPOOL_N_THREADS_BITS;
    n_graph     = ((n_graph + 1) << GGML_THREADPOOL_N_THREADS_BITS) | (n_threads & GGML_THREADPOOL_N_THREADS_MASK);
    tp->n_graph.store(n_graph, std::memory_order_seq_cst);

    // Reset barrier arrival counter for this dispatch.
    tp->n_barrier.store(0, std::memory_order_seq_cst);

    // Spawn all N tasks.  HPX task creation is sub-microsecond, so even N=32
    // tasks add only ~32 µs — negligible vs real inference workloads.
    std::vector<hpx::future<void>> futures;
    futures.reserve(static_cast<std::size_t>(n_threads));
    for (int i = 0; i < n_threads; i++) {
        futures.push_back(hpx::async([tp, i] {
            ggml_graph_compute_thread(&tp->workers[i]);
        }));
    }

    // hpx::wait_all() from a non-HPX thread does an OS-level block, which is
    // correct here — the caller (ggml_graph_compute) is on the main OS thread.
    hpx::wait_all(futures);

    return tp->ec;
}

// ── ggml_threadpool_new / free ────────────────────────────────────────────────

extern "C" ggml_threadpool * ggml_threadpool_new(ggml_threadpool_params * tpp) {
    hpx_start_once();

    auto * tp = static_cast<ggml_threadpool *>(ggml_aligned_malloc(sizeof(ggml_threadpool)));
    memset(tp, 0, sizeof(*tp));

    tp->n_threads = tpp->n_threads;
    tp->prio      = tpp->prio;
    tp->poll      = tpp->poll;
    tp->ec        = GGML_STATUS_SUCCESS;
    tp->n_graph.store(0, std::memory_order_relaxed);
    tp->n_barrier.store(0, std::memory_order_relaxed);
    tp->n_barrier_passed.store(0, std::memory_order_relaxed);
    tp->current_chunk.store(0, std::memory_order_relaxed);
    tp->abort.store(-1, std::memory_order_relaxed);
    tp->stop.store(false, std::memory_order_relaxed);
    tp->pause.store(static_cast<bool>(tpp->paused), std::memory_order_relaxed);

    const size_t wsz     = sizeof(ggml_compute_state) * tpp->n_threads;
    auto *       workers = static_cast<ggml_compute_state *>(ggml_aligned_malloc(wsz));
    memset(workers, 0, wsz);

    for (int j = 0; j < tpp->n_threads; j++) {
        workers[j].threadpool = tp;
        workers[j].ith        = j;
    }
    tp->workers = workers;

    return tp;
}

extern "C" void ggml_threadpool_free(ggml_threadpool * tp) {
    if (!tp) {
        return;
    }
    // No persistent worker tasks to join — all tasks are scoped to each
    // ggml_graph_compute_hpx() call and are done before it returns.
    const size_t wsz = sizeof(ggml_compute_state) * tp->n_threads;
    ggml_aligned_free(tp->workers, wsz);
    ggml_aligned_free(tp, sizeof(ggml_threadpool));
    // HPX runtime is NOT stopped here — atexit handler manages shutdown.
}

// ── pause / resume ────────────────────────────────────────────────────────────
//
// pause/resume only affect the entry to the next dispatch; in-progress tasks
// are not interrupted.  With per-dispatch tasks, the pool is naturally "idle"
// between dispatches, so pause just gates the next call to ggml_graph_compute_hpx.

extern "C" void ggml_threadpool_pause(ggml_threadpool * tp) {
    tp->pause.store(true, std::memory_order_seq_cst);
}

extern "C" void ggml_threadpool_resume(ggml_threadpool * tp) {
    tp->pause.store(false, std::memory_order_seq_cst);
}
