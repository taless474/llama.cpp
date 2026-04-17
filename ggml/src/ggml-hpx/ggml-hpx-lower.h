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

#ifdef __cplusplus
}    // extern "C"
#endif

#endif    // GGML_HPX_REGION_DAG
