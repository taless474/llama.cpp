#pragma once

// ggml-hpx-region-dag.h
//
// Fine-grained CPU work region types for the HPX executor DAG path.
//
// Guarded by GGML_HPX_REGION_DAG. All definitions here are inactive unless
// that flag is set at compile time. No existing dispatch path includes this
// header; it exists solely to define the contract for the experimental DAG
// executor layer.
//
// Three-level hierarchy:
//   graph nodes
//     -> ggml_hpx_region          (coarse: a slice of graph nodes, existing)
//        -> ggml_hpx_cpu_region   (fine: a parallel work range within one op)
//
// A heavy node (e.g. mul_mat) may lower to one or more ggml_hpx_cpu_region
// instances. HPX executes those fine regions natively via block_fork_join_executor
// or futures; it does not call ggml_graph_compute_thread_run for this path.
//
// Inter-region dependencies:
//   ggml_hpx_dep_edge { src, dst }  means dst cannot start until src completes.
//   src is the producer / prerequisite index into the region group's region array.
//   dst is the dependent / consumer index.
//
// Scratch and reduction buffers:
//   ggml_hpx_region_resources carries all mutable per-dispatch state that
//   fine regions may need: a shared scratch slab, per-lane scratch (one
//   pointer per HPX worker in the team, indexed by lane/ith), and a reduction
//   buffer for reductions that must accumulate across lanes.
//
//   lane_scratch[i] is valid for i in [0, n_lanes). It corresponds to the
//   i-th HPX worker in the dispatch team. This is stable only when the HPX
//   runtime uses static scheduling (--hpx:queuing=static), which is the
//   configured default in ggml-hpx-runtime.cpp.
//
// run_range callback:
//   void run_range(ctx, ith, nth, begin, end, resources)
//   Called by the HPX executor for each chunk of [begin, end).
//     ith       - chunk index (0-based), analogous to thread index
//     nth       - total chunk count, analogous to thread count
//     begin/end - half-open work range for this chunk
//   run_range must be reentrant across different (ith, begin, end) tuples.
//   It must not call back into ggml_graph_compute or ggml_backend_graph_compute.
//
// This is a C-compatible header so it may be included by both C and C++ TUs.

#ifdef GGML_HPX_REGION_DAG

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

// ---------------------------------------------------------------------------
// Region kind
// ---------------------------------------------------------------------------

// Coarse classification of the operation a fine region computes.
// Used by the executor to select dispatch policy (e.g. chunk size heuristics,
// reduction strategy). Does not affect correctness; only policy.
typedef enum ggml_hpx_cpu_region_kind
{
    GGML_HPX_CPU_REGION_KIND_INVALID     = 0,
    GGML_HPX_CPU_REGION_KIND_MATMUL      = 1,    // mul_mat, outer product
    GGML_HPX_CPU_REGION_KIND_ELEMENTWISE = 2,    // rope, scale, add, gelu, silu
    GGML_HPX_CPU_REGION_KIND_REDUCTION   = 3,    // rms_norm, softmax, sum
    GGML_HPX_CPU_REGION_KIND_CUSTOM      = 4,    // anything not covered above
} ggml_hpx_cpu_region_kind;

// ---------------------------------------------------------------------------
// Dependency edge
// ---------------------------------------------------------------------------

// Directed edge in the region group DAG.
// src -> dst  means:  dst may not begin execution until src has completed.
// Indices refer to positions in ggml_hpx_cpu_region_group.regions[].
// Edges must form a DAG; cycles produce undefined behaviour.
typedef struct ggml_hpx_dep_edge
{
    int src;    // producer / prerequisite region index
    int dst;    // dependent / consumer region index
} ggml_hpx_dep_edge;

// ---------------------------------------------------------------------------
// Per-dispatch resources
// ---------------------------------------------------------------------------

// Mutable state passed to every run_range invocation in a dispatch.
// Owned by the executor; regions must not free or reallocate any pointer here.
typedef struct ggml_hpx_region_resources
{
    void*  shared_scratch;      // single contiguous scratch slab, shared across all lanes
    void** lane_scratch;        // lane_scratch[i]: per-lane scratch for lane i in [0, n_lanes)
    void*  reduction_buffer;    // accumulation buffer for cross-lane reductions
    int    n_lanes;             // number of HPX workers in the dispatch team
} ggml_hpx_region_resources;

// ---------------------------------------------------------------------------
// run_range callback type
// ---------------------------------------------------------------------------

// Kernel entry point for a fine CPU region.
//
// Parameters:
//   ctx       - opaque per-region context (set in ggml_hpx_cpu_region.ctx)
//   ith       - chunk index in [0, nth); analogous to thread id
//   nth       - total chunk count; analogous to thread count
//   begin     - inclusive start of this chunk's work range
//   end       - exclusive end of this chunk's work range
//   resources - executor-owned scratch and reduction buffers; non-null
//
// Constraints:
//   - Must be reentrant for distinct (ith, begin, end) tuples.
//   - Must not call ggml_graph_compute, ggml_backend_graph_compute,
//     or any function that re-enters the HPX executor.
//   - begin < end is guaranteed; zero-width chunks are not dispatched.
typedef void (*ggml_hpx_run_range_fn)(
    void*                        ctx,
    int                          ith,
    int                          nth,
    int64_t                      begin,
    int64_t                      end,
    ggml_hpx_region_resources*   resources);

// ---------------------------------------------------------------------------
// Fine CPU region
// ---------------------------------------------------------------------------

// One parallel work range within a single op's kernel.
// The executor fans this out across HPX workers using block_fork_join_executor
// or equivalent, calling run_range once per chunk.
//
// grain: minimum work unit size. The executor will not split a chunk smaller
// than grain. Set to 0 to let the executor decide.
typedef struct ggml_hpx_cpu_region
{
    ggml_hpx_cpu_region_kind    kind;           // dispatch policy hint
    int64_t                     begin;          // inclusive start of work range
    int64_t                     end;            // exclusive end of work range
    int64_t                     grain;          // minimum chunk size; 0 = executor decides
    void*                       ctx;            // opaque kernel context
    ggml_hpx_run_range_fn       run_range;      // kernel entry point; must be non-null
} ggml_hpx_cpu_region;

// ---------------------------------------------------------------------------
// Fine CPU region group
// ---------------------------------------------------------------------------

// A set of fine CPU regions connected by an explicit dependency DAG.
// The executor may run independent regions in parallel; regions with
// unsatisfied predecessors (via deps) are deferred until all src regions
// in their incoming edges have completed.
//
// Ownership: regions and deps are borrowed — the group does not own them.
// Both arrays must remain valid for the lifetime of the dispatch call.
//
// n_deps == 0 is valid: all regions are independent and may run fully in
// parallel.
typedef struct ggml_hpx_cpu_region_group
{
    ggml_hpx_cpu_region*    regions;      // array of fine regions
    int                     n_regions;    // length of regions[]

    ggml_hpx_dep_edge*      deps;         // DAG edges; may be null if n_deps == 0
    int                     n_deps;       // length of deps[]
} ggml_hpx_cpu_region_group;

#ifdef __cplusplus
}
#endif

#endif    // GGML_HPX_REGION_DAG
