// ggml-hpx-exec-selective.h
//
// Graph-level selective executor: walks a ggml_cgraph node-by-node and
// routes each node to one of three paths:
//
//   - fine-region path       (ggml_hpx_lower_op + ggml_hpx_run_region_group)
//   - CPU-backend fallback   (1-node ggml_graph_view on a caller-owned cpu_be)
//   - frozen-packet path     (optional; recognized repeated sublayers only)
//
// The frozen-packet path is off by default. It engages only when the caller
// supplies both a packet_runtime and a sublayer-specific cache (currently
// only MLP gate/up). Nothing about the fine-region or fallback paths changes
// when packets are disabled.
//
// First-deployment scope
// ----------------------
// MLP gate/up packet dispatch requires F32 gate and up MUL_MAT. Quantized
// models (including TinyLlama Q4_K_M, the current smoke target) produce
// quantized MUL_MAT that ggml_hpx_lower_op rejects, so the composer fails,
// so the cache records no entry, and packet_matches stays 0. On those
// models the selective path behaves exactly as it does today.
//
// The first-deployment packet path uses a single-lane decode runtime
// (n_lanes = 1) to validate matcher, cache, bind, and llama-integrated
// dispatch before widening to multi-lane packet execution. Proving
// end-to-end speedup on llama requires a model whose MLP projections are F32.
//
// Gated on GGML_HPX_REGION_DAG.

#pragma once

#ifdef GGML_HPX_REGION_DAG

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>

// Opaque forward-declarations. Full definitions live in:
//   ggml_hpx_packet_runtime           — ggml-hpx-packet.h
//   ggml_hpx_mlp_gate_up_packet_cache — this TU's .cpp (internal)
//   ggml_hpx_mlp_glu_packet_cache     — this TU's .cpp (internal)
struct ggml_hpx_packet_runtime;
struct ggml_hpx_mlp_gate_up_packet_cache;
struct ggml_hpx_mlp_glu_packet_cache;

// Per-call counters filled by ggml_hpx_exec_graph_selective_mul_mat.
// Times are in nanoseconds.
//
// When packet dispatch is enabled (both packet args non-null) and the
// matcher pre-scan runs, every node in gf is counted in exactly one of
// lowered_nodes, fallback_nodes, or packet_nodes. When packet dispatch is
// disabled, packet_matches and packet_nodes stay 0 and every node lands
// in lowered_nodes or fallback_nodes.
struct ggml_hpx_selective_stats {
    uint32_t lowered_nodes      = 0; // nodes through the fine-region path
    uint32_t fallback_nodes     = 0; // nodes through the CPU-backend fallback
    uint32_t packet_matches     = 0; // repeated sublayer patterns dispatched
    uint32_t packet_nodes       = 0; // ggml nodes consumed by packet matches
                                     //   (MLP gate/up: 4 per match; MLP GLU: 3 per match)
    uint64_t lowered_ns         = 0; // wall time for all lowered dispatches
    uint64_t fallback_ns        = 0; // wall time for all fallback dispatches
    uint64_t packet_dispatch_ns = 0; // bind + run time across packet matches
                                     //   (cache-miss compile cost is NOT included)
};

// ---------------------------------------------------------------------------
// MLP gate/up packet cache
// ---------------------------------------------------------------------------
//
// Plan store. Amortizes the cost of compose + compile + frame allocation
// across repeated MLP gate/up subgraphs that share (out_cols, cols, rows).
//
// Not a generic packet registry: specialized for one sublayer. Additional
// sublayers get additional caches rather than widening this one.
//
// The cache does NOT hold a reference to a ggml_hpx_packet_runtime. The
// runtime is an execution substrate passed in at dispatch time; the cache
// is strictly a plan store. Layering is enforced by keeping them as two
// separate parameters on the selective entry point below.
//
// n_lanes must equal the n_decode_threads the accompanying packet runtime
// was created with; it flows into every cached packet's plan key through
// ggml_hpx_packet_plan_key::n_lanes.
//
// seq_regime and policy_version are fixed at create time; they flow into
// every cached packet's plan key. Bump policy_version whenever the
// compiler's lowering rules change so older cached packets are not reused.
ggml_hpx_mlp_gate_up_packet_cache * ggml_hpx_mlp_gate_up_packet_cache_create(
    uint32_t n_lanes,
    uint32_t seq_regime,
    uint32_t policy_version);

void ggml_hpx_mlp_gate_up_packet_cache_destroy(
    ggml_hpx_mlp_gate_up_packet_cache * cache);

// ---------------------------------------------------------------------------
// MLP GLU (SWIGLU) packet cache
// ---------------------------------------------------------------------------
//
// Plan store for the fused-SWIGLU sublayer. Real llama MLPs emit
// GGML_OP_GLU with subop SWIGLU — a single node that fuses the separate
// SiLU + elementwise MUL of the gate/up form. This cache is the GLU
// counterpart to ggml_hpx_mlp_gate_up_packet_cache.
//
// Ownership, layering, and n_lanes / seq_regime / policy_version semantics
// are identical to the gate/up cache above. The current compiler policy
// for MLP_GLU_F32 is policy_version = 1; bump if lowering rules change.
ggml_hpx_mlp_glu_packet_cache * ggml_hpx_mlp_glu_packet_cache_create(
    uint32_t n_lanes,
    uint32_t seq_regime,
    uint32_t policy_version);

void ggml_hpx_mlp_glu_packet_cache_destroy(
    ggml_hpx_mlp_glu_packet_cache * cache);

// ---------------------------------------------------------------------------
// Packet env — bundles runtime + all sublayer caches
// ---------------------------------------------------------------------------
//
// Passed as a single optional pointer to the selective entry point. Pass
// nullptr to disable all packet dispatch (identical to the pre-packet path).
// Individual cache fields may be null to disable that sublayer's packet path
// while leaving the other enabled; rt must be non-null whenever any cache
// field is non-null.
//
// Caller owns all three pointers; lifetimes must exceed every call that
// receives this env.
struct ggml_hpx_selective_packet_env
{
    ggml_hpx_packet_runtime *             rt;           // shared execution substrate
    ggml_hpx_mlp_gate_up_packet_cache *   mlp_cache;    // null → gate/up path disabled
    ggml_hpx_mlp_glu_packet_cache *       mlp_glu_cache; // null → GLU path disabled
};

// ---------------------------------------------------------------------------
// Selective graph executor
// ---------------------------------------------------------------------------
//
// Walks gf->nodes[0..n_nodes). When packet dispatch is enabled via
// packet_env, two pre-scans run (gate/up first, GLU second) over a shared
// consumed[] bitmap, then the main loop dispatches in node order:
//
//   - gate/up final MUL    -> bind + run MLP_GATE_UP_F32 frozen packet
//   - GLU final node       -> bind + run MLP_GLU_F32    frozen packet
//   - other consumed node  -> skip (non-trigger packet member)
//   - ggml_hpx_lower_op succeeds     -> ggml_hpx_run_region_group
//   - ggml_hpx_lower_op returns false -> ggml_backend_graph_compute via cpu_be
//
// cpu_be must be a live CPU backend owned by the caller. The caller retains
// ownership; this function does not free it.
//
// n_lanes controls fine-region fan-out for the lowered path only. Pass 0
// to use hpx::get_num_worker_threads() at call time; pass > 0 to fix the
// lane count (useful in tests). Independent of packet_env->rt's n_lanes.
//
// stats_out, if non-null, is filled with per-call counters.
//
// packet_env is optional. Pass nullptr to disable all packet dispatch.
// Decode-only gating is a caller responsibility: pass nullptr on prefill
// ubatches. The packet runtime is team=DECODE only.
//
// Requires the HPX runtime to be started (ggml_hpx_tpool_start()).
// Returns true if all nodes complete without error.
bool ggml_hpx_exec_graph_selective_mul_mat(
    ggml_cgraph *                              gf,
    ggml_backend_t                             cpu_be,
    int                                        n_lanes    = 0,
    ggml_hpx_selective_stats *                 stats_out  = nullptr,
    const ggml_hpx_selective_packet_env *      packet_env = nullptr);

// Convenience wrapper for arena-style unit tests (ggml_init with no_alloc=false).
// Creates a fresh CPU backend internally, runs the graph, then frees the backend.
// Not suitable for sched-managed graphs (tensors must have data already allocated).
//
// This wrapper does NOT support packet dispatch: it is a narrow helper for
// existing one-op lowering tests. Callers wanting packet dispatch must use
// the full entry point with their own cpu_be.
bool ggml_hpx_exec_graph_selective_mul_mat_arena(
    ggml_cgraph *              gf,
    int                        n_lanes   = 1,
    ggml_hpx_selective_stats * stats_out = nullptr);

// ---------------------------------------------------------------------------
// Test-only instrumentation — compiled in only when
// GGML_HPX_EXEC_SELECTIVE_TESTING is defined.
//
// g_hpx_selective_lowered_count        — ++ when ggml_hpx_lower_op succeeds
//                                        for a graph node (lowered path)
// g_hpx_selective_executed_count       — ++ when the fine-region group for a
//                                        lowered node is dispatched
// g_hpx_mlp_packet_compile_count       — ++ on every MLP gate/up cache miss
//                                        (compose + compile path)
// g_hpx_mlp_packet_dispatch_count      — ++ on every MLP gate/up packet
//                                        dispatch (miss + hit alike)
// g_hpx_mlp_glu_packet_compile_count   — ++ on every MLP GLU cache miss
// g_hpx_mlp_glu_packet_dispatch_count  — ++ on every MLP GLU packet
//                                        dispatch (miss + hit alike)
//
// Reset to 0 before each test that checks them.
// ---------------------------------------------------------------------------
#ifdef GGML_HPX_EXEC_SELECTIVE_TESTING
#include <atomic>
extern std::atomic<int> g_hpx_selective_lowered_count;
extern std::atomic<int> g_hpx_selective_executed_count;
extern std::atomic<int> g_hpx_mlp_packet_compile_count;
extern std::atomic<int> g_hpx_mlp_packet_dispatch_count;
extern std::atomic<int> g_hpx_mlp_glu_packet_compile_count;
extern std::atomic<int> g_hpx_mlp_glu_packet_dispatch_count;
#endif

#endif // GGML_HPX_REGION_DAG
