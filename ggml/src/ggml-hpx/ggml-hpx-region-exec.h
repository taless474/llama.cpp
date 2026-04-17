// ggml-hpx-region-exec.h
//
// Validator, direct kernel callback, and HPX execution API for the
// ggml_hpx_cpu_region contract defined in ggml-hpx-region-dag.h.
//
// All declarations here are guarded by GGML_HPX_REGION_DAG.
//
// Layered interface
// ─────────────────
//   1. ggml_hpx_validate_region_group   : pre-flight contract check
//   2. ggml_hpx_mul_mat_f32_run_range   : direct F32 mul_mat kernel
//      ggml_hpx_mul_mat_f32_ctx         : context for that kernel
//   3. ggml_hpx_run_single_region       : execute one region
//   4. ggml_hpx_run_region_group        : dependency-driven group runner
//
// None of the functions declared here call:
//   ggml_graph_compute_thread_run
//   ggml_graph_compute
//   ggml_backend_graph_compute
//
// C / C++ split
// ─────────────
//   Items 1-2 are declared inside extern "C" and may be called from C or C++.
//   Items 3-4 are C++ only and require an HPX thread context.
//
//   Wrap calls in hpx::async(...).get() when invoking from a non-HPX thread
//   after hpx::start() and before hpx::stop().

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

// ── F32 RMS_NORM kernels ─────────────────────────────────────────────────
//
// Three-region decomposition of one F32 RMS_NORM row:
//
//   R0 (ELEMENTWISE) : lane-parallel partial sumsq into lane_scratch[ith]
//   R1 (REDUCTION)   : single-thread finalize → reduction_buffer
//   R2 (ELEMENTWISE) : lane-parallel apply, dst[i] = x[i] * scale
//
// Data flow through resources, not ctx structs:
//   R0 writes  lane_scratch[ith]           (float partial sumsq)
//   R1 reads   lane_scratch[0 .. n_lanes)  → writes reduction_buffer
//   R2 reads   reduction_buffer->scale     → writes dst

typedef struct ggml_hpx_rms_norm_f32_reduce_buffer
{
    float sumsq;
    float scale;
} ggml_hpx_rms_norm_f32_reduce_buffer;

typedef struct ggml_hpx_rms_norm_partial_f32_ctx
{
    const float * x;
    int64_t       n;
} ggml_hpx_rms_norm_partial_f32_ctx;

typedef struct ggml_hpx_rms_norm_finalize_f32_ctx
{
    int64_t n;
    float   eps;
} ggml_hpx_rms_norm_finalize_f32_ctx;

typedef struct ggml_hpx_rms_norm_apply_f32_ctx
{
    const float * x;
    float *       dst;
    int64_t       n;
} ggml_hpx_rms_norm_apply_f32_ctx;

void ggml_hpx_rms_norm_partial_f32_run_range(
    void *                      ctx,
    int                         ith,
    int                         nth,
    int64_t                     begin,
    int64_t                     end,
    ggml_hpx_region_resources * resources);

void ggml_hpx_rms_norm_finalize_f32_run_range(
    void *                      ctx,
    int                         ith,
    int                         nth,
    int64_t                     begin,
    int64_t                     end,
    ggml_hpx_region_resources * resources);

void ggml_hpx_rms_norm_apply_f32_run_range(
    void *                      ctx,
    int                         ith,
    int                         nth,
    int64_t                     begin,
    int64_t                     end,
    ggml_hpx_region_resources * resources);

// ── F32 SiLU kernel ──────────────────────────────────────────────────────────
//
// Elementwise SiLU: dst[i] = x[i] * sigmoid(x[i]) = x[i] / (1 + exp(-x[i]))
//
// The run_range begin/end index elements in [0, n).
// Uses no resources (no lane_scratch, no reduction_buffer).

typedef struct ggml_hpx_silu_f32_ctx
{
    const float * x;    // input  [n]
    float *       dst;  // output [n]; may alias x
    int64_t       n;    // total element count
} ggml_hpx_silu_f32_ctx;

void ggml_hpx_silu_f32_run_range(
    void *                      ctx,
    int                         ith,
    int                         nth,
    int64_t                     begin,
    int64_t                     end,
    ggml_hpx_region_resources * resources);

// ── F32 elementwise MUL kernel ───────────────────────────────────────────────
//
// Elementwise multiply: dst[i] = a[i] * b[i]
//
// The run_range begin/end index elements in [0, n).
// Uses no resources (no lane_scratch, no reduction_buffer).

typedef struct ggml_hpx_mul_f32_ctx
{
    const float * a;    // first input  [n]
    const float * b;    // second input [n]
    float *       dst;  // output       [n]; may alias a or b
    int64_t       n;    // total element count
} ggml_hpx_mul_f32_ctx;

void ggml_hpx_mul_f32_run_range(
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

// Execute one fine CPU region using the current HPX region runner.
//
// For non-REDUCTION regions, the runner splits [region.begin, region.end)
// into up to resources->n_lanes non-empty chunks and launches one HPX task
// per non-empty chunk:
//
//   region.run_range(ctx, ith, n_lanes, chunk_begin, chunk_end, resources)
//
// REDUCTION regions are currently executed single-threaded:
//   ith = 0, nth = 1, begin = region.begin, end = region.end
//
// Assumes region != nullptr and resources != nullptr.
void ggml_hpx_run_single_region(
    const ggml_hpx_cpu_region * region,
    ggml_hpx_region_resources * resources);

// Execute a ggml_hpx_cpu_region_group as a dependency-driven DAG.
//
// The group must already satisfy ggml_hpx_validate_region_group().
//
// Execution model:
//   - predecessor lists are built from group->deps
//   - regions are topologically ordered with Kahn's algorithm
//   - one hpx::shared_future<void> is created per region
//   - a region with no predecessors launches immediately
//   - a region with predecessors is launched via hpx::dataflow(...) after
//     all predecessor futures become ready
//
// This preserves true inter-region concurrency for independent regions.
// There is no explicit level loop.
//
// Each region still uses ggml_hpx_run_single_region semantics internally:
// non-REDUCTION regions may fan out across resources->n_lanes, while
// REDUCTION regions currently run single-threaded.
//
// resources must not be null.
void ggml_hpx_run_region_group(
    const ggml_hpx_cpu_region_group * group,
    ggml_hpx_region_resources *       resources);

#endif    // __cplusplus

#endif    // GGML_HPX_REGION_DAG
