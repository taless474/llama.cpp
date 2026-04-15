// ggml-hpx-region-exec.h
//
// Validator, kernel context/callback, and HPX executor API for the
// ggml_hpx_cpu_region contract defined in ggml-hpx-region-dag.h.
//
// All declarations here are guarded by GGML_HPX_REGION_DAG.
//
// Layered interface
// ─────────────────
//   1. ggml_hpx_validate_region_group   — pre-flight contract check
//   2. ggml_hpx_mul_mat_f32_run_range   — direct F32 mul_mat kernel
//      ggml_hpx_mul_mat_f32_ctx         — context struct for that kernel
//   3. ggml_hpx_run_single_region       — one region, fork_join fan-out
//   4. ggml_hpx_run_region_group        — dependency-ordered group runner
//
// None of the functions in this header call:
//   ggml_graph_compute_thread_run
//   ggml_graph_compute
//   ggml_backend_graph_compute
//
// C / C++ split
// ─────────────
//   Items 1–2 are declared inside extern "C" and may be called from C or C++.
//   Items 3–4 are C++ only (fork_join_executor) and require an HPX thread
//   context (they construct fork_join_executor, which calls
//   this_thread::get_pool() internally).
//
//   Wrap in hpx::async(...).get() when calling from the main thread after
//   hpx::start but before hpx::stop.

#pragma once

#ifdef GGML_HPX_REGION_DAG

#include "ggml-hpx-region-dag.h"

// ---------------------------------------------------------------------------
// C-compatible section (validator + F32 mul_mat kernel)
// ---------------------------------------------------------------------------

#ifdef __cplusplus
extern "C"
{
#endif

// ── Validator ────────────────────────────────────────────────────────────

// Pre-flight check for a ggml_hpx_cpu_region_group.
//
// Always verifies:
//   - group != nullptr
//   - n_regions > 0  →  regions != nullptr
//   - n_deps   > 0  →  deps   != nullptr
//   - every region: run_range != nullptr, begin < end
//   - every dep edge: 0 <= src, dst < n_regions, src != dst
//
// If check_acyclic is true, also runs Kahn's algorithm on the dep graph
// (O(n_regions + n_deps)).  A cycle returns a non-null error string.
//
// Returns nullptr on success, or a static C-string describing the first
// violation.  Does not allocate heap memory.
const char * ggml_hpx_validate_region_group(
    const ggml_hpx_cpu_region_group * group,
    bool                              check_acyclic);

// ── F32 mul_mat kernel ────────────────────────────────────────────────────

// Context for the direct F32 mul_mat kernel.
//
// Memory layout (all row-major):
//   x — [rows × cols]      input activations
//   w — [out_cols × cols]  weight matrix; row i is weight column i
//   y — [rows × out_cols]  output
//
// y[row][col] = dot(x[row, :], w[col, :])
//
// The run_range begin/end index output columns in [0, out_cols).
typedef struct ggml_hpx_mul_mat_f32_ctx
{
    const float * x;        // input activations  [rows × cols]
    const float * w;        // weight matrix      [out_cols × cols]
    float *       y;        // output             [rows × out_cols]
    int64_t       rows;     // batch dimension
    int64_t       cols;     // shared (reduction) dimension
    int64_t       out_cols; // output columns; work range is [0, out_cols)
} ggml_hpx_mul_mat_f32_ctx;

// run_range callback for F32 mul_mat.
//
// begin/end are output column indices in [0, n_out).
// Calls ggml_vec_dot_f32 directly; does not enter ggml_graph_compute,
// ggml_graph_compute_thread_run, or ggml_backend_graph_compute.
//
// Satisfies the ggml_hpx_run_range_fn contract:
//   reentrant for distinct (ith, begin, end) tuples;
//   resources parameter is accepted but not used by this kernel.
void ggml_hpx_mul_mat_f32_run_range(
    void *                      ctx,
    int                         ith,
    int                         nth,
    int64_t                     begin,
    int64_t                     end,
    ggml_hpx_region_resources * resources);

#ifdef __cplusplus
}    // extern "C"
#endif

// ---------------------------------------------------------------------------
// C++ executor API (requires HPX thread context)
// ---------------------------------------------------------------------------

#ifdef __cplusplus

// Execute one fine CPU region across at most resources->n_lanes HPX workers.
//
// Splits [region.begin, region.end) into n_lanes non-empty chunks and
// dispatches each chunk to one HPX worker via fork_join_executor:
//
//   region.run_range(ctx, ith, n_lanes, chunk_begin, chunk_end, resources)
//
// Exception: REDUCTION regions (kind == GGML_HPX_CPU_REGION_KIND_REDUCTION)
// and n_lanes == 1 bypass fan-out and run single-threaded (ith = 0).
//
// Constructs a new fork_join_executor per call.  For hot paths the caller
// should use the lower-level internal run_on_exec helper directly.
// resources must not be null.
void ggml_hpx_run_single_region(
    const ggml_hpx_cpu_region * region,
    ggml_hpx_region_resources * resources);

// Execute a ggml_hpx_cpu_region_group in dependency order.
//
// Computes topological levels from the dep edges (O(n_regions + n_deps)).
// At each level all regions are sequentially dispatched through one shared
// fork_join_executor so the executor is never driven by concurrent callers
// (fork_join_executor documents concurrent callers as UB).
//
// Intra-region parallelism is preserved: each region fans out across
// resources->n_lanes workers via for_loop.  Genuine inter-region concurrency
// at the same level (independent regions running simultaneously) is a TODO —
// it requires either separate executors per region or a different executor
// type that supports concurrent submission.
//
// Assumes the group has passed ggml_hpx_validate_region_group.
// resources must not be null.
void ggml_hpx_run_region_group(
    const ggml_hpx_cpu_region_group * group,
    ggml_hpx_region_resources *       resources);

#endif    // __cplusplus

#endif    // GGML_HPX_REGION_DAG
