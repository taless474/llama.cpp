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
struct ggml_hpx_packet_runtime;
struct ggml_hpx_mlp_gate_up_packet_cache;

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
                                     //   (MLP gate/up: 4 * packet_matches)
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
// Selective graph executor
// ---------------------------------------------------------------------------
//
// Walks gf->nodes[0..n_nodes). When packet dispatch is enabled, a pre-scan
// recognizes repeated sublayer patterns and marks their nodes for packet
// dispatch. Then, for each node in order:
//
//   - packet-marked                  -> dispatched as part of its packet
//                                       match at the final node of the
//                                       pattern; other pattern nodes skip
//   - ggml_hpx_lower_op succeeds     -> ggml_hpx_run_region_group
//   - ggml_hpx_lower_op returns false -> ggml_backend_graph_compute on a
//                                        1-node ggml_graph_view via cpu_be
//
// cpu_be must be a live CPU backend owned by the caller (e.g. the
// llama_context::backend_cpu that already participates in the scheduler).
// The caller retains ownership; this function does not free it.
//
// n_lanes controls the fine-region fan-out for the lowered path only.
// Pass 0 (or omit) to use hpx::get_num_worker_threads() at call time.
// Pass a positive integer to fix the lane count explicitly (useful in
// tests for reproducibility). This is independent of the packet runtime's
// n_lanes.
//
// stats_out, if non-null, is filled with per-call counters.
//
// packet_rt and mlp_cache are optional and must be supplied together.
// Exactly one of these conditions holds on entry:
//   (a) both null     -> packet dispatch is disabled; behaviour is
//                        identical to a pure selective call
//   (b) both non-null -> packet dispatch is enabled for MLP gate/up
//   (c) exactly one   -> debug-asserts; release treats as (a)
//
// Decode-only gating is a caller responsibility: pass null packet args on
// prefill ubatches. The packet runtime is team=DECODE only, so non-null
// packet args on a prefill graph is unsupported in v1.
//
// Requires the HPX runtime to be started (ggml_hpx_tpool_start()).
// Returns true if all nodes complete without error.
bool ggml_hpx_exec_graph_selective_mul_mat(
    ggml_cgraph *                         gf,
    ggml_backend_t                        cpu_be,
    int                                   n_lanes    = 0,
    ggml_hpx_selective_stats *            stats_out  = nullptr,
    ggml_hpx_packet_runtime *             packet_rt  = nullptr,
    ggml_hpx_mlp_gate_up_packet_cache *   mlp_cache  = nullptr);

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
// g_hpx_selective_lowered_count    — ++ when ggml_hpx_lower_op succeeds
//                                    for a graph node (lowered path)
// g_hpx_selective_executed_count   — ++ when the fine-region group for a
//                                    lowered node is dispatched
// g_hpx_mlp_packet_compile_count   — ++ on every MLP gate/up cache miss
//                                    (compose + compile path)
// g_hpx_mlp_packet_dispatch_count  — ++ on every MLP gate/up packet
//                                    dispatch (miss + hit alike)
//
// Reset to 0 before each test that checks them.
// ---------------------------------------------------------------------------
#ifdef GGML_HPX_EXEC_SELECTIVE_TESTING
#include <atomic>
extern std::atomic<int> g_hpx_selective_lowered_count;
extern std::atomic<int> g_hpx_selective_executed_count;
extern std::atomic<int> g_hpx_mlp_packet_compile_count;
extern std::atomic<int> g_hpx_mlp_packet_dispatch_count;
#endif

#endif // GGML_HPX_REGION_DAG
