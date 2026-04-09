// Internal header — not part of the public API.
//
// Exposes struct ggml_compute_job and struct ggml_threadpool for:
//   - ggml/src/ggml-cpu/ggml-cpu.c   (include after platform blocks, with prereqs flag)
//   - white-box tests that need direct field access
//     (add ${PROJECT_SOURCE_DIR}/ggml/src/ggml-cpu to include dirs)
//
// Usage from ggml-cpu.c:
//   #define GGML_CPU_THREADPOOL_PREREQS_DONE  // platform types already set up
//   #include "ggml-cpu-threadpool.h"
//
// Usage from tests (standalone):
//   #include "ggml-cpu-threadpool.h"          // header sets everything up

#pragma once

#include "ggml.h" // ggml_cgraph, ggml_cplan, ggml_status

// ---------------------------------------------------------------------------
// Platform prerequisites (skipped when included from ggml-cpu.c which sets
// them up earlier in the translation unit).
// ---------------------------------------------------------------------------

#ifndef GGML_CPU_THREADPOOL_PREREQS_DONE

#define GGML_CACHE_LINE 64

#if defined(_WIN32)

#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>

#  if defined(_MSC_VER) && !defined(__clang__)
// MSVC native: provide the atomic subset needed to read struct fields.
#    define GGML_CACHE_ALIGN __declspec(align(GGML_CACHE_LINE))

typedef volatile LONG atomic_int;
typedef atomic_int    atomic_bool;

typedef enum ggml_tpool_memory_order {
    memory_order_relaxed,
    memory_order_consume,
    memory_order_acquire,
    memory_order_release,
    memory_order_acq_rel,
    memory_order_seq_cst
} memory_order;

static LONG ggml_tpool_atomic_load_expl(volatile LONG * ptr, memory_order mo) {
    (void)mo;
    return InterlockedCompareExchange((LONG *)ptr, 0, 0);
}
#    define atomic_load_explicit(ptr, mo) \
        ggml_tpool_atomic_load_expl((volatile LONG *)(ptr), (mo))

#  else // clang-cl or MinGW
#    define GGML_CACHE_ALIGN __attribute__((aligned(GGML_CACHE_LINE)))
#    include <stdatomic.h>
#  endif

typedef CONDITION_VARIABLE ggml_cond_t;
typedef SRWLOCK            ggml_mutex_t;

#else // POSIX

#  if defined(__clang__) || defined(__GNUC__)
#    define GGML_CACHE_ALIGN __attribute__((aligned(GGML_CACHE_LINE)))
#  endif

#  include <stdatomic.h>
#  include <pthread.h>

typedef pthread_cond_t  ggml_cond_t;
typedef pthread_mutex_t ggml_mutex_t;

#endif // _WIN32

#endif // GGML_CPU_THREADPOOL_PREREQS_DONE

// ---------------------------------------------------------------------------
// Forward declaration — full definition is in ggml-cpu.c.
// Only used as a pointer inside struct ggml_threadpool.
// ---------------------------------------------------------------------------

struct ggml_compute_state;

// ---------------------------------------------------------------------------
// Per-dispatch job state.
// Initialized completely by the caller before being published to workers
// via executor->current_job and then executor->n_graph.
// Lifetime: must remain live until all workers pass the final barrier
// and ggml_graph_compute_thread returns.
// ---------------------------------------------------------------------------

struct ggml_compute_job {
    struct ggml_cgraph * cgraph;
    struct ggml_cplan  * cplan;
    int                  n_active_threads; // threads participating in this dispatch

    atomic_int GGML_CACHE_ALIGN n_barrier;
    atomic_int GGML_CACHE_ALIGN n_barrier_passed;
    atomic_int GGML_CACHE_ALIGN current_chunk; // MatMul chunk index, shared across threads

    atomic_int       abort; // abort-at-node index; -1 = no abort
    enum ggml_status ec;    // dispatch result
};

// ---------------------------------------------------------------------------
// Executor: owns the worker substrate — threads, wakeup policy, affinity, lifetime.
// Multiple CPU backend instances may share one executor; the executor is created
// and freed independently of any backend (see ggml_backend_cpu_attach_threadpool).
//
// Relationship to per-dispatch state:
//   current_job — pointer to the active ggml_compute_job during a dispatch;
//                 always points to the embedded `job` field between dispatches
//                 so the pointer is never left dangling.
//   n_graph     — publication channel: seq# in high bits, active-thread-count in
//                 low bits; workers poll this to detect new work; updated last in
//                 kickoff after current_job and all job fields are written.
// ---------------------------------------------------------------------------

struct ggml_threadpool {
    ggml_mutex_t mutex; // mutex for cond.var
    ggml_cond_t  cond;  // cond.var for waiting for new work

    struct ggml_compute_job   job;         // embedded job; home for current_job between dispatches
    struct ggml_compute_job * current_job; // points to active job during dispatch

    atomic_int n_graph; // publication channel (see comment above)

    // these are atomic as an annotation for thread-sanitizer
    atomic_bool stop;  // Used for stopping the threadpool altogether
    atomic_bool pause; // Used for pausing the threadpool or individual threads

    struct ggml_compute_state * workers; // per thread state
    int      n_threads;                  // Number of threads in the pool
    int32_t  prio;                       // Scheduling priority
    uint32_t poll;                       // Polling level (0 - no polling)
};
