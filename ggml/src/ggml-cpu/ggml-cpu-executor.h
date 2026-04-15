// ggml-cpu-executor.h
//
// Executor ops interface — zero platform-header dependencies.
// Safe to include from both C and C++ translation units.
//
// Included by:
//   ggml-cpu-threadpool.h  (gets the ops struct used inside struct ggml_threadpool)
//   ggml-hpx-tpool.h       (HPX executor implementation)

#pragma once

#include <stddef.h>    /* size_t */

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
// Isolates substrate-specific operations so that alternative compute
// substrates (e.g. HPX) can be wired in without touching the barrier,
// chunk-dispatch, or kernel-execution logic.
//
// Two execution models coexist:
//
//   pthread (default): init spawns persistent workers; kickoff wakes them;
//     the common layer runs worker-0 inline; workers synchronise via the
//     spin barrier in ggml_graph_compute_thread.  run_job is NULL.
//
//   HPX: init creates a reusable scheduler_executor; run_job submits a
//     bulk region over logical worker IDs [0, n_threads) to that executor
//     and waits for completion.  kickoff/worker_wait are not used.
// ---------------------------------------------------------------------------

struct ggml_cpu_executor_ops
{
    // Called from ggml_threadpool_new_impl after mutex/cond are initialized.
    // pthread: spawns n_threads-1 worker threads and applies CPU placement.
    // HPX:     creates the owned scheduler_executor (N threads, HPX topology).
    void (*init)(struct ggml_threadpool*       tp,
                 struct ggml_threadpool_params* tpp);

    // Execute one full graph dispatch across n_threads logical workers.
    // current_job (cgraph, cplan, n_active_threads) is already set when
    // this is called.  Returns only after all workers have completed.
    //
    // pthread: NULL — the common layer uses its own kickoff + inline worker-0.
    // HPX:     submits hpx::experimental::for_loop over [0, n_threads) on
    //          the owned executor; each iteration calls
    //          ggml_graph_compute_thread_run(&tp->workers[j]).
    void (*run_job)(struct ggml_threadpool* tp, int n_threads);

    // Signal all workers to exit, join them, and tear down the substrate
    // (mutex, cond, executor state).  Called from ggml_threadpool_free.
    void (*destroy)(struct ggml_threadpool* tp);
};

// ---------------------------------------------------------------------------
// Functions implemented in ggml-cpu.c — safe to call from C++.
// ---------------------------------------------------------------------------

// Override the executor ops for all future ggml_threadpool_new() calls.
// Pass NULL to restore the default pthread ops.
// Prefer ggml_threadpool_new_with_ops() for new code; this function modifies
// global state and is retained only for the legacy default-path override.
void ggml_cpu_set_executor_ops(const struct ggml_cpu_executor_ops* ops);

// Create a threadpool with an explicitly supplied executor ops table.
// ops must not be NULL.  Does not read or modify g_executor_ops.
struct ggml_threadpool* ggml_threadpool_new_with_ops(
    struct ggml_threadpool_params*      tpp,
    const struct ggml_cpu_executor_ops* ops);

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

// Return the number of nodes in the graph currently loaded into tp.
// Returns -1 if tp, tp->current_job, or tp->current_job->cgraph is NULL.
// In normal execution all three are valid when run_job is called, but the
// defensive check makes the contract explicit for callers that probe this
// value (e.g. histogram logging, future size-gating policies).
int ggml_threadpool_n_nodes(const struct ggml_threadpool* tp);

// Return the scratch work_size (bytes) from the cplan of the currently
// active job.  work_size is computed by ggml_graph_plan() and reflects the
// total scratch needed for this graph at the current batch size; it varies
// with batch/sequence length even when n_nodes is invariant.
// Returns SIZE_MAX if tp, tp->current_job, or tp->current_job->cplan is NULL.
// (0 is a valid work_size for graphs with no scratch requirement, so SIZE_MAX
// is the sentinel rather than 0.)
size_t ggml_threadpool_work_size(const struct ggml_threadpool* tp);

// Set tp->stop = true, then destroy tp->mutex and tp->cond.
// Used by the HPX destroy op which has no direct access to the
// platform mutex/cond macros.
void ggml_threadpool_destroy_substrate(struct ggml_threadpool* tp);

// Attach or retrieve opaque executor-private state on a threadpool.
// The pointer is set once in executor->init and read in kickoff/destroy.
// The executor owns the allocation; ggml does not inspect it.
void  ggml_threadpool_set_priv(struct ggml_threadpool* tp, void* priv);
void* ggml_threadpool_get_priv(const struct ggml_threadpool* tp);

#ifdef __cplusplus
}
#endif
