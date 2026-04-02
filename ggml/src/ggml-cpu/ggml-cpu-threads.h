#pragma once

// Shared struct definitions for the ggml CPU thread pool.
// Included by both ggml-cpu.c (pthread impl) and ggml-cpu-hpx.cpp (HPX impl).
//
// Threading typedefs (ggml_mutex_t, ggml_cond_t, ggml_thread_t) are defined
// in ggml-cpu.c before this header is included. In HPX mode those fields are
// replaced with void* handles so the typedefs are not required here.

#include "ggml.h"       // GGML_MAX_N_THREADS, enum ggml_status, struct ggml_cgraph/cplan
#include <stdint.h>
#include <stdbool.h>

// atomic_int / atomic_bool
//
// C mode  : use C11 <stdatomic.h> types directly.
// C++ mode: alias std::atomic<> to the C11 names so the struct definition is
//           shared.  This avoids the libc++ / clang <stdatomic.h> macro conflict
//           that occurs when <atomic> and <stdatomic.h> are both visible.
//           (std::atomic<int> and _Atomic int are ABI-compatible on GCC/Clang.)
#if defined(__cplusplus)
#  include <atomic>
using atomic_int  = std::atomic<int>;
using atomic_bool = std::atomic<bool>;
#elif !defined(_MSC_VER) || defined(__clang__)
#  include <stdatomic.h>
#endif

// ── Cache line alignment ──────────────────────────────────────────────────────

#ifndef GGML_CACHE_LINE
#define GGML_CACHE_LINE 64
#endif

#ifndef GGML_CACHE_ALIGN
#  if defined(__clang__) || defined(__GNUC__)
#    define GGML_CACHE_ALIGN __attribute__((aligned(GGML_CACHE_LINE)))
#  elif defined(_MSC_VER)
#    define GGML_CACHE_ALIGN __declspec(align(GGML_CACHE_LINE))
#  else
#    define GGML_CACHE_ALIGN
#  endif
#endif

// ── Thread count packing helpers ──────────────────────────────────────────────

#define GGML_THREADPOOL_N_THREADS_MASK (0xffffU)
#define GGML_THREADPOOL_N_THREADS_BITS (16)

// ── Threadpool ────────────────────────────────────────────────────────────────

struct ggml_threadpool {
#if !defined(GGML_USE_HPX)
    ggml_mutex_t mutex;       // mutex for cond.var
    ggml_cond_t  cond;        // cond.var for waiting for new work
#endif

    struct ggml_cgraph * cgraph;
    struct ggml_cplan  * cplan;

    // synchronization primitives
    atomic_int n_graph;       // updated when there is work to be done; holds graph and active thread counts
#if !defined(GGML_USE_HPX)
    atomic_int GGML_CACHE_ALIGN n_barrier;
    atomic_int GGML_CACHE_ALIGN n_barrier_passed;
#endif
    atomic_int GGML_CACHE_ALIGN current_chunk; // currently processing chunk during Mat_Mul, shared between all the threads

    // these are atomic as an annotation for thread-sanitizer
    atomic_bool stop;         // Used for stopping the threadpool altogether
    atomic_bool pause;        // Used for pausing the threadpool or individual threads
    atomic_int  abort;        // Used for aborting processing of a graph

    struct ggml_compute_state * workers;   // per thread state
    int          n_threads;   // Number of threads in the pool
    int32_t      prio;        // Scheduling priority
    uint32_t     poll;        // Polling level (0 - no polling)

    enum ggml_status ec;

#if defined(GGML_USE_HPX)
    void * hpx_mutex;         // std::mutex*              (callable from any thread)
    void * hpx_cond;          // std::condition_variable* (callable from any thread)
    void * hpx_barrier;       // hpx::barrier<>*          (HPX threads only)
#endif
};

// ── Per-thread state ──────────────────────────────────────────────────────────

struct ggml_compute_state {
#if !defined(GGML_USE_OPENMP) && !defined(GGML_USE_HPX)
    ggml_thread_t thrd;
#endif
#if defined(GGML_USE_HPX)
    void * hpx_future;        // hpx::future<void>*
#endif
#if !defined(GGML_USE_OPENMP)
    int  last_graph;
    bool pending;
#endif
    bool cpumask[GGML_MAX_N_THREADS];
    struct ggml_threadpool * threadpool;
    int ith;
};

// ── HPX-mode extras ───────────────────────────────────────────────────────────

#if defined(GGML_USE_HPX)
// thread_ret_t matches the definition in ggml-cpu.c
#  if defined(_WIN32) && !defined(__MINGW32__)
typedef DWORD thread_ret_t;
#  else
typedef void * thread_ret_t;
#  endif

// ggml_graph_compute_thread is non-static in HPX mode so ggml-cpu-hpx.cpp can call it.
// Use extern "C" in C++ mode to match the C linkage of the definition in ggml-cpu.c.
#if defined(__cplusplus)
extern "C" {
#endif
thread_ret_t ggml_graph_compute_thread(void * data);
#if defined(__cplusplus)
}
#endif
#endif
