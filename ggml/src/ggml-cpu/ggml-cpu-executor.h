// ggml-cpu-executor.h
//
// Executor ops interface — zero platform-header dependencies.
// Safe to include from both C and C++ translation units.
//
// Included by:
//   ggml-cpu-threadpool.h  (gets the ops struct used inside struct ggml_threadpool)
//   ggml-hpx-tpool.h       (HPX executor implementation)

#pragma once

#include <stdint.h>    // int64_t

#ifdef __cplusplus
extern "C"
{
#endif

// ---------------------------------------------------------------------------
// Forward declarations — full definitions are elsewhere.
// ---------------------------------------------------------------------------

struct ggml_compute_state;
struct ggml_threadpool;
struct ggml_threadpool_params;

// ---------------------------------------------------------------------------
// Executor ops vtable.
//
// Isolates the four substrate-specific operations so that alternative
// compute substrates (e.g. HPX) can be wired in without touching the
// barrier, chunk-dispatch, or kernel-execution logic.
//
// The default implementation (ggml_pthread_executor_ops) wraps the existing
// pthreads + condition-variable substrate.  No behavior changes; the ops are
// one indirection point between ggml_threadpool_new_impl / kickoff /
// secondary-thread loop / ggml_threadpool_free and the actual
// create/wake/sleep/join primitives.
// ---------------------------------------------------------------------------

struct ggml_cpu_executor_ops
{
    // Launch persistent workers (pthreads) and apply CPU placement.
    // Called from ggml_threadpool_new_impl after mutex/cond are initialized.
    // HPX: no-op (tasks are submitted on each kickoff instead).
    void (*init)(struct ggml_threadpool*       tp,
                 struct ggml_threadpool_params* tpp);

    // Wake workers for a new dispatch.
    // Called with tp->mutex held (exclusive), after n_graph has been
    // published with a seq_cst store and current_job is set.
    void (*kickoff)(struct ggml_threadpool* tp, int n_threads);

    // Persistent worker's full idle cycle: block until new work is ready
    // (state->pending == true) or the pool is stopping (tp->stop == true).
    // Called from the secondary-thread loop.
    // HPX: unreachable — no persistent workers exist.
    void (*worker_wait)(struct ggml_threadpool*    tp,
                        struct ggml_compute_state* state);

    // Signal all workers to exit, join them, and tear down the substrate
    // (mutex, cond, etc.).  Called from ggml_threadpool_free.
    void (*destroy)(struct ggml_threadpool* tp);
};

// ---------------------------------------------------------------------------
// Functions implemented in ggml-cpu.c — safe to call from C++.
// ---------------------------------------------------------------------------

// Override the executor ops for all future ggml_threadpool_new() calls.
// Pass NULL to restore the default pthread ops.
// Called by the HPX backend during initialisation.
void ggml_cpu_set_executor_ops(const struct ggml_cpu_executor_ops* ops);

// Non-static wrapper: lets HPX code in a separate TU submit one worker
// as an HPX task without crossing the static-linkage boundary.
void ggml_graph_compute_thread_run(struct ggml_compute_state* state);

// Return &tp->workers[j].  Avoids exposing the full layout of
// struct ggml_compute_state to callers that only need the pointer.
struct ggml_compute_state* ggml_threadpool_worker(
    struct ggml_threadpool* tp, int j);

// Return tp->n_threads.  Avoids exposing the full layout of
// struct ggml_threadpool to C++ callers that cannot include
// ggml-cpu-threadpool.h (stdatomic.h / HPX macro conflict).
int ggml_threadpool_n_threads(const struct ggml_threadpool* tp);

// Set tp->stop = true, then destroy tp->mutex and tp->cond.
// Used by the HPX destroy op which has no direct access to the
// platform mutex/cond macros.
void ggml_threadpool_destroy_substrate(struct ggml_threadpool* tp);

// Override the active-thread count for the current dispatch to n.
// Used by hpx_kickoff to downgrade to single-thread execution when
// called from an HPX coroutine (to avoid barrier deadlock).
void ggml_threadpool_set_active_threads(struct ggml_threadpool* tp, int n);

// Attach or retrieve opaque executor-private state on a threadpool.
// The pointer is set once in executor->init and read in kickoff/destroy.
// The executor owns the allocation; ggml does not inspect it.
void  ggml_threadpool_set_priv(struct ggml_threadpool* tp, void* priv);
void* ggml_threadpool_get_priv(const struct ggml_threadpool* tp);

// Return nodes[n_nodes-1]->ne[1] for the graph attached to the current job.
// Returns -1 if tp, current_job, or the graph is NULL, or n_nodes == 0.
// Safe to call from C++ TUs that cannot include ggml-cpu-threadpool.h
// (which pulls in <stdatomic.h> and conflicts with HPX memory-order macros).
int64_t ggml_threadpool_last_node_ne1(const struct ggml_threadpool* tp);

#ifdef __cplusplus
}
#endif
