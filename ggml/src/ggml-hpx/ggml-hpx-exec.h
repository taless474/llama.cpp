#pragma once

// ggml-hpx-exec.h
//
// Top-level executor interface for the HPX execution layer.
//
// The executor owns the HPX runtime, the plan cache, the abort token, and
// instrumentation hooks. It is created once per llama context and destroyed
// when the context is destroyed.
//
// Per-run flow (handled inside exec.cpp):
//   1. reset abort token
//   2. call adapter to obtain topology + key (decode: graph walk;
//      prefill: scheduler-driven split translation)
//   3. look up plan in cache; build and insert if miss
//   4. iterate regions: check abort, dispatch to runtime, track dependencies
//   5. fire instrumentation hooks at run and region boundaries
//
// exec.cpp is the orchestration layer. It does not manage HPX thread pools
// directly (that is runtime's responsibility) and does not read scheduler
// state directly (that is the adapter's responsibility).
//
// Forward declarations for ggml_cgraph, ggml_backend, and ggml_backend_sched
// are provided by ggml-hpx-fwd.h. ggml-backend.h must not be included here.
//
// Note: both run entry points take ggml_cgraph* (not const). The ggml graph
// API is not const-correct: ggml_graph_view and ggml_graph_n_nodes both
// require a non-const pointer even when only reading.

#include "ggml-hpx-fwd.h"        // ggml_cgraph, ggml_backend_t, ggml_backend_sched_t
#include "ggml-hpx-instrument.h" // ggml_hpx_metrics_hooks
#include "ggml-hpx-runtime.h"    // ggml_hpx_runtime_params

#include <cstdint>

// ---------------------------------------------------------------------------
// Opaque executor handle
// ---------------------------------------------------------------------------

// Full definition is in ggml-hpx-exec.cpp. Callers hold a pointer only.
struct ggml_hpx_exec;

// ---------------------------------------------------------------------------
// Creation parameters
// ---------------------------------------------------------------------------

struct ggml_hpx_exec_params
{
    ggml_hpx_runtime_params runtime;       // worker team sizes, initial scratch size
    uint32_t policy_version;               // written into every plan key; bump when
                                           // planner logic changes to force rebuild
    ggml_hpx_metrics_hooks hooks;          // optional benchmark/instrumentation callbacks;
                                           // all function pointers default to null
};

// ---------------------------------------------------------------------------
// Run status
// ---------------------------------------------------------------------------

enum class ggml_hpx_exec_status : uint8_t
{
    ok,       // all regions completed successfully
    aborted,  // cooperative abort was observed; some regions may not have run
};

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

// Create the executor: start the HPX runtime, create worker teams, and
// pre-allocate scratch. Returns a non-null pointer on success.
//
// Defined in ggml-hpx-exec.cpp.
ggml_hpx_exec* ggml_hpx_exec_create(ggml_hpx_exec_params const& params);

// Shut down the HPX runtime, join worker teams, release scratch and all
// cached plans. The pointer is invalid after this call.
//
// Defined in ggml-hpx-exec.cpp.
void ggml_hpx_exec_destroy(ggml_hpx_exec* exec);

// ---------------------------------------------------------------------------
// Abort
// ---------------------------------------------------------------------------

// Signal a cooperative abort. Safe to call from any thread, including from
// a ggml abort callback. The current or next run will observe the signal and
// return aborted without starting new regions.
//
// The abort token is reset automatically at the start of each run, so the
// caller does not need to clear it between runs.
//
// Defined in ggml-hpx-exec.cpp.
void ggml_hpx_exec_abort(ggml_hpx_exec* exec);

// ---------------------------------------------------------------------------
// Backend bundle — decode
// ---------------------------------------------------------------------------

// Live backend handles passed to each decode run. Plans are structural and
// carry no handles; callers supply the bundle per-run so backend ownership
// stays with the caller.
//
// cpu  - required: used for all cpu_contiguous regions and as the fallback
//        when blas is null or when blas_delegated regions appear in a run
//        where blas was not provided
// blas - optional (nullable): used for blas_delegated regions when non-null;
//        omitting it silently falls back to cpu (slower, but correct)
struct ggml_hpx_decode_backends
{
    ggml_backend_t cpu  = nullptr;  // required
    ggml_backend_t blas = nullptr;  // nullable — BLAS is optional
};

// ---------------------------------------------------------------------------
// Run entry points
// ---------------------------------------------------------------------------

// Execute the graph on the decode path (low-latency single-token).
// Uses a topology derived from an independent graph walk; does not touch
// the scheduler.
//
// graph    - the graph to execute; used only during this call, not retained
// backends - live backend handles; cpu must be non-null (asserted on entry)
//
// Defined in ggml-hpx-exec.cpp.
ggml_hpx_exec_status ggml_hpx_exec_run_decode(
    ggml_hpx_exec*                  exec,
    ggml_cgraph*                    graph,
    ggml_hpx_decode_backends const& backends);

// Execute the graph on the prefill path (throughput-oriented prompt/batch).
// Uses a topology derived from a scheduler-driven split translation performed
// inside the adapter. The scheduler pointer is not retained after the call.
//
// graph - the graph to execute; used only during this call, not retained
// sched - caller-owned scheduler; the adapter may invoke split-related
//         functions on it but does not retain, reset, or own it
//
// Defined in ggml-hpx-exec.cpp.
ggml_hpx_exec_status ggml_hpx_exec_run_prefill(
    ggml_hpx_exec* exec, ggml_cgraph* graph, ggml_backend_sched_t sched);
