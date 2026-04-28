// ggml-hpx-lower.h
//
// Op lowering: translate one ggml_tensor compute node into a fine-region group.
//
// See docs/HPX_LOWER_OP.md for the full contract.
//
// Gated on GGML_HPX_REGION_DAG.

#pragma once

#ifdef GGML_HPX_REGION_DAG

#include "ggml-hpx-region-dag.h"

#include "ggml.h"

#include <cstddef>
#include <cstring>

// ---------------------------------------------------------------------------
// Storage constants
// ---------------------------------------------------------------------------
//
// MAX_REGIONS / MAX_DEPS: upper bounds on what any supported op emits.
//   Currently: RMS_NORM emits 3 regions + 2 dep edges (the largest).
//
// CTX_BYTES_PER_REGION: per-region ctx slot size.
//   Compile-time asserts in ggml-hpx-lower.cpp verify every emitted ctx fits.
//   Currently: ggml_hpx_mul_mat_f32_ctx is the largest at ~48 bytes.

#define GGML_HPX_LOWERING_MAX_REGIONS            4
#define GGML_HPX_LOWERING_MAX_DEPS               6
#define GGML_HPX_LOWERING_CTX_BYTES_PER_REGION 128

// Scratch arena shared across regions within one lowered op.
// Currently used by the Q4_K two-region pipeline: region 0 writes a Q8_K
// quantized activation row here; region 1 reads from it.
// Size: 8192 bytes covers Q8_K rows up to cols=7168
// (ggml_row_size(Q8_K, 7168) = 28 blocks × 292 bytes = 8176 bytes).
// Bumped from 4096 to fit TinyLlama's MLP down-projection (cols=5632 →
// 6424 bytes) and similar shapes; observed in
// hpx-bench/results/2026-04-28-q4k-repack-real-tinyllama where the
// 4096 limit silently dropped 12 nodes/step into the CPU fallback.
// lower_op still rejects Q4_K nodes whose Q8_K row exceeds this limit.
#define GGML_HPX_LOWERING_SCRATCH_BYTES 8192

// ---------------------------------------------------------------------------
// Output arena
// ---------------------------------------------------------------------------
//
// ggml_hpx_lowering owns temporary storage for one lowered op.
//
// Ownership rules:
//   - group.regions[i].ctx always points into ctx_buf[i].
//     It must never be redirected to an external buffer.
//   - group.regions points at regions[]; group.deps points at deps[]
//     (or nullptr when n_deps == 0).
//   - Callers may pass &out->group directly to ggml_hpx_compile_packet or
//     ggml_hpx_run_region_group. Both helpers read the group during the call
//     only; they must not retain raw pointers into it.
//   - Anything the packet compiler needs persistently must be deep-copied
//     into packet-owned storage before compile_packet returns.
//   - This struct is NOT safe to move or copy after lower_op returns, because
//     group.regions and group.deps contain absolute pointers into the inline
//     arrays. Use ggml_hpx_lowering_init to reset before reuse.
//
// Lifetime: must remain valid until after ggml_hpx_compile_packet() returns.
// After compile, the struct may be destroyed.

typedef struct ggml_hpx_lowering
{
    ggml_hpx_cpu_region_group  group;
    ggml_hpx_cpu_region        regions[GGML_HPX_LOWERING_MAX_REGIONS];
    ggml_hpx_dep_edge          deps   [GGML_HPX_LOWERING_MAX_DEPS];
    alignas(max_align_t) unsigned char
        ctx_buf[GGML_HPX_LOWERING_MAX_REGIONS][GGML_HPX_LOWERING_CTX_BYTES_PER_REGION];
    // Scratch arena for intermediate data shared across regions.
    // Zero-initialised by ggml_hpx_lowering_init via memset.
    alignas(max_align_t) unsigned char scratch[GGML_HPX_LOWERING_SCRATCH_BYTES];
} ggml_hpx_lowering;

// Zero the struct and re-wire the internal pointers. Call this before every
// use (first-time init or reuse). Ensures lower_op always starts from a known
// state regardless of prior contents.
static inline void ggml_hpx_lowering_init(ggml_hpx_lowering * l)
{
    std::memset(l, 0, sizeof(*l));
    l->group.regions = l->regions;
    l->group.deps    = l->deps;
}

// ---------------------------------------------------------------------------
// Lower
// ---------------------------------------------------------------------------
//
// Translate one ggml compute node into a fine-region group.
//
// On success:
//   - out->group.n_regions > 0.
//   - out->group.regions == out->regions, out->group.deps == out->deps (or
//     nullptr when n_deps == 0).
//   - Each out->regions[i].ctx points into out->ctx_buf[i].
//   - Returns true.
//
// On failure:
//   - out->group.n_regions is set to 0.
//   - Returns false.
//
// Failure conditions:
//   - op not in the supported set (see docs/HPX_LOWER_OP.md).
//   - node->type != GGML_TYPE_F32 (only F32 is implemented).
//   - GGML_OP_UNARY: unary subop is not GGML_UNARY_OP_SILU.
//   - GGML_OP_RMS_NORM: ne[1] != 1 (multi-row not yet supported).
//   - Any required src pointer is null.
//
// node and out must both be non-null.

#ifdef __cplusplus
extern "C" {
#endif

bool ggml_hpx_lower_op(
    const struct ggml_tensor * node,
    ggml_hpx_lowering *        out);

// Trait predicate: true iff op->src[0] is a CPU_REPACK Q4_K weight using
// the q4_K_8x8_q8_K trait — the only repacked Q4_K trait the HPX lowered
// path can drive (via ggml_gemv_q4_K_8x8_q8_K). False on null op,
// null src[0], null src[0]->extra, any other trait name, or when the
// repack-traits helper returns null.
bool ggml_hpx_is_q4k_8x8_repacked(const struct ggml_tensor * op);

#ifdef __cplusplus
}    // extern "C"
#endif

#endif    // GGML_HPX_REGION_DAG
