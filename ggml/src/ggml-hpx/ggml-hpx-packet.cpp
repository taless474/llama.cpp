// ggml-hpx-packet.cpp
//
// Frozen-packet prototype. See ggml-hpx-packet.h for the full contract.
//
// This translation unit:
//   - Owns the struct definitions behind the opaque handles
//     (ggml_hpx_frozen_packet / ggml_hpx_packet_frame / ggml_hpx_packet_runtime).
//   - Implements packet compile, run, and all introspection/lifecycle entry
//     points.
//   - Currently supports two sublayers:
//       GGML_HPX_PACKET_SUBLAYER_RMS_NORM_F32    (3-region)
//       GGML_HPX_PACKET_SUBLAYER_MLP_GATE_UP_F32 (4-region)
//   - Currently supports exactly one team: GGML_HPX_PACKET_TEAM_DECODE.
//
// Execution model (invariant I4): the run loop never constructs hpx::async,
// hpx::dataflow, hpx::shared_future, or wait_all. Lane fan-out is a single
// hpx::experimental::for_loop over [0, n_lanes) on the packet runtime's
// pinned decode Exec. SERIAL steps run on the caller thread.
//
// Ownership (invariant I7): frame storage is caller-owned. frame_init copies
// the compiled ctx-arena template into the caller's buffer; no heap
// allocation happens per invocation.

#ifdef GGML_HPX_REGION_DAG

#include "ggml-hpx-packet.h"
#include "ggml-hpx-region-dag.h"
#include "ggml-hpx-region-exec.h"

#include <hpx/algorithm.hpp>
#include <hpx/execution.hpp>
#include <hpx/executors/scheduler_executor.hpp>
#include <hpx/executors/thread_pool_scheduler.hpp>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

// ---------------------------------------------------------------------------
// Scheduler / Exec type aliases
// ---------------------------------------------------------------------------
//
// Match the pattern used by ggml-hpx-tpool.cpp exactly so the packet runtime
// and the bulk-region threadpool share the same persistent-worker substrate.

namespace
{

using Scheduler =
    hpx::execution::experimental::thread_pool_scheduler;

using Exec =
    hpx::execution::experimental::scheduler_executor<Scheduler>;

// Internal step kind mirror of the (test-only) public enum. The public enum
// lives behind GGML_HPX_PACKET_TEST_API, so the implementation needs its own
// constants to stay compilable in the production build.
constexpr uint32_t kStepKindSerial     = 0;
constexpr uint32_t kStepKindLaneFanout = 1;

// The compiler's private step record. Layout matches the (test-only) public
// ggml_hpx_packet_step so callers compiled with GGML_HPX_PACKET_TEST_API see
// the same bytes.
struct PacketStep
{
    uint32_t               kind;
    ggml_hpx_run_range_fn  run_range;
    uint32_t               ctx_offset;     // absolute byte offset from frame base
    uint32_t               ctx_size;       // diagnostics / bounds checks
    int64_t                work_begin;
    int64_t                work_end;
    uint16_t               n_lanes;
    uint16_t               flags;
};

// ---------------------------------------------------------------------------
// RMS_NORM_F32 arena layout
// ---------------------------------------------------------------------------
//
// The three ctx structs from ggml-hpx-region-exec.h are laid out sequentially
// inside the frame, right after the frame header (packet back-pointer).
//
//   frame base + 0    : ggml_hpx_packet_frame header (back-pointer, 8 B)
//   frame base + 8    : partial_ctx   { const float* x; int64_t n; }            (16 B)
//   frame base + 24   : finalize_ctx  { int64_t n; float eps; pad4; }           (16 B)
//   frame base + 40   : apply_ctx     { const float* x; float* dst; int64_t n; }(24 B)
//   total frame size  : 64 B, 8-byte aligned
//
// All ctx fields are 8-byte aligned; the arena alignment is alignof(int64_t) = 8.

constexpr size_t kRmsFrameHeaderSize = 8;    // sizeof(ggml_hpx_packet_frame)
constexpr size_t kRmsPartialCtxSize  = sizeof(ggml_hpx_rms_norm_partial_f32_ctx);
constexpr size_t kRmsFinalizeCtxSize = sizeof(ggml_hpx_rms_norm_finalize_f32_ctx);
constexpr size_t kRmsApplyCtxSize    = sizeof(ggml_hpx_rms_norm_apply_f32_ctx);

constexpr size_t kRmsPartialOff  = kRmsFrameHeaderSize;
constexpr size_t kRmsFinalizeOff = kRmsPartialOff  + kRmsPartialCtxSize;
constexpr size_t kRmsApplyOff    = kRmsFinalizeOff + kRmsFinalizeCtxSize;
constexpr size_t kRmsFrameBytes  = kRmsApplyOff    + kRmsApplyCtxSize;
constexpr size_t kRmsFrameAlign  = 8;

// Field-level offsets for the typed RMS_NORM bind. These are small enough
// that the compiler resolves them at compile time.
constexpr size_t kRmsPartialXFieldOff    =
    offsetof(ggml_hpx_rms_norm_partial_f32_ctx, x);
constexpr size_t kRmsFinalizeEpsFieldOff =
    offsetof(ggml_hpx_rms_norm_finalize_f32_ctx, eps);
constexpr size_t kRmsApplyXFieldOff      =
    offsetof(ggml_hpx_rms_norm_apply_f32_ctx, x);
constexpr size_t kRmsApplyDstFieldOff    =
    offsetof(ggml_hpx_rms_norm_apply_f32_ctx, dst);

// ---------------------------------------------------------------------------
// MLP_GATE_UP_F32 arena layout
// ---------------------------------------------------------------------------
//
// Four ctx structs laid out sequentially after the frame header:
//
//   frame base + 0    : ggml_hpx_packet_frame header (back-pointer, 8 B)
//   frame base + 8    : gate MUL_MAT ctx  { x*; w*; y*; rows; cols; out_cols } (48 B)
//   frame base + 56   : up   MUL_MAT ctx  { x*; w*; y*; rows; cols; out_cols } (48 B)
//   frame base + 104  : SiLU ctx          { x*; dst*; n }                      (24 B)
//   frame base + 128  : elementwise MUL ctx { a*; b*; dst*; n }                (32 B)
//   total frame size  : 160 B, 8-byte aligned
//
// Dimensions are baked at compile time; pointer fields are zeroed in the
// template and patched per-invocation by ggml_hpx_bind_mlp_gate_up_packet.

constexpr size_t kMlpFrameHeaderSize = 8;

constexpr size_t kMlpGateCtxSize  = sizeof(ggml_hpx_mul_mat_f32_ctx);    // 48
constexpr size_t kMlpUpCtxSize    = sizeof(ggml_hpx_mul_mat_f32_ctx);    // 48
constexpr size_t kMlpSiluCtxSize  = sizeof(ggml_hpx_silu_f32_ctx);       // 24
constexpr size_t kMlpEwMulCtxSize = sizeof(ggml_hpx_mul_f32_ctx);        // 32

constexpr size_t kMlpGateCtxOff  = kMlpFrameHeaderSize;
constexpr size_t kMlpUpCtxOff    = kMlpGateCtxOff  + kMlpGateCtxSize;
constexpr size_t kMlpSiluCtxOff  = kMlpUpCtxOff    + kMlpUpCtxSize;
constexpr size_t kMlpEwMulCtxOff = kMlpSiluCtxOff  + kMlpSiluCtxSize;
constexpr size_t kMlpFrameBytes  = kMlpEwMulCtxOff + kMlpEwMulCtxSize;
constexpr size_t kMlpFrameAlign  = 8;

// Field-level offsets for bind patching (shared by gate/up slots for MUL_MAT).
constexpr size_t kMlpMulMatXOff  = offsetof(ggml_hpx_mul_mat_f32_ctx, x);
constexpr size_t kMlpMulMatWOff  = offsetof(ggml_hpx_mul_mat_f32_ctx, w);
constexpr size_t kMlpMulMatYOff  = offsetof(ggml_hpx_mul_mat_f32_ctx, y);
constexpr size_t kMlpSiluXOff    = offsetof(ggml_hpx_silu_f32_ctx, x);
constexpr size_t kMlpSiluDstOff  = offsetof(ggml_hpx_silu_f32_ctx, dst);
constexpr size_t kMlpEwMulAOff   = offsetof(ggml_hpx_mul_f32_ctx, a);
constexpr size_t kMlpEwMulBOff   = offsetof(ggml_hpx_mul_f32_ctx, b);
constexpr size_t kMlpEwMulDstOff = offsetof(ggml_hpx_mul_f32_ctx, dst);

// ---------------------------------------------------------------------------
// MLP_GLU_F32 arena layout
// ---------------------------------------------------------------------------
//
// Fused-SWIGLU counterpart to MLP_GATE_UP_F32. Three ctx structs laid out
// sequentially after the frame header:
//
//   frame base + 0    : ggml_hpx_packet_frame header (back-pointer, 8 B)
//   frame base + 8    : gate MUL_MAT ctx   { x*; w*; y*; rows; cols; out_cols } (48 B)
//   frame base + 56   : up   MUL_MAT ctx   { x*; w*; y*; rows; cols; out_cols } (48 B)
//   frame base + 104  : SWIGLU ctx         { gate*; up*; dst*; n }              (32 B)
//   total frame size  : 136 B, 8-byte aligned
//
// Dimensions are baked at compile time; pointer fields are zeroed in the
// template and patched per-invocation by ggml_hpx_bind_mlp_glu_packet.

constexpr size_t kGluFrameHeaderSize = 8;

constexpr size_t kGluGateCtxSize   = sizeof(ggml_hpx_mul_mat_f32_ctx);    // 48
constexpr size_t kGluUpCtxSize     = sizeof(ggml_hpx_mul_mat_f32_ctx);    // 48
constexpr size_t kGluSwigluCtxSize = sizeof(ggml_hpx_swiglu_f32_ctx);     // 32

constexpr size_t kGluGateCtxOff   = kGluFrameHeaderSize;
constexpr size_t kGluUpCtxOff     = kGluGateCtxOff   + kGluGateCtxSize;
constexpr size_t kGluSwigluCtxOff = kGluUpCtxOff     + kGluUpCtxSize;
constexpr size_t kGluFrameBytes   = kGluSwigluCtxOff + kGluSwigluCtxSize;
constexpr size_t kGluFrameAlign   = 8;

// Field-level offsets for bind patching. MUL_MAT offsets are reused from the
// MLP_GATE_UP layout above (kMlpMulMatX/W/YOff) since the ctx struct is the
// same type; only SWIGLU is distinct.
constexpr size_t kGluSwigluGateOff = offsetof(ggml_hpx_swiglu_f32_ctx, gate);
constexpr size_t kGluSwigluUpOff   = offsetof(ggml_hpx_swiglu_f32_ctx, up);
constexpr size_t kGluSwigluDstOff  = offsetof(ggml_hpx_swiglu_f32_ctx, dst);

}    // namespace

// ---------------------------------------------------------------------------
// Opaque type definitions
// ---------------------------------------------------------------------------

struct ggml_hpx_frozen_packet
{
    ggml_hpx_packet_plan_key               key;
    uint32_t                               n_steps;
    PacketStep                           * steps;           // heap-allocated, owned
    void                                 * ctx_template;    // heap-allocated, owned
    size_t                                 arena_offset;    // template copy dest in frame
    size_t                                 arena_bytes;     // length of ctx_template
    size_t                                 frame_bytes;
    size_t                                 frame_alignment;
    ggml_hpx_packet_resource_requirements  res;
};

struct ggml_hpx_packet_frame
{
    const ggml_hpx_frozen_packet * packet;
    // Ctx arena follows at offset `packet->arena_offset` (== sizeof header
    // aligned up to arena alignment). Fields at absolute offsets
    // step.ctx_offset from this base.
};

struct ggml_hpx_packet_runtime
{
    Exec     decode_exec;
    uint32_t n_decode_threads;

    explicit ggml_hpx_packet_runtime(uint32_t n_decode)
      : decode_exec(hpx::parallel::execution::with_processing_units_count(
            Scheduler{},
            static_cast<std::size_t>(n_decode)))
      , n_decode_threads(n_decode)
    {
    }
};

// ---------------------------------------------------------------------------
// Compile-time layout asserts
// ---------------------------------------------------------------------------

static_assert(sizeof(ggml_hpx_packet_frame) == kRmsFrameHeaderSize,
    "frame header size assumption baked into arena layout");

// ---------------------------------------------------------------------------
// Runtime lifecycle
// ---------------------------------------------------------------------------

ggml_hpx_packet_runtime * ggml_hpx_packet_runtime_create(
    uint32_t n_decode_threads)
{
    if (n_decode_threads == 0)
    {
        return nullptr;
    }
    return new ggml_hpx_packet_runtime{n_decode_threads};
}

void ggml_hpx_packet_runtime_destroy(ggml_hpx_packet_runtime * runtime)
{
    delete runtime;
}

// ---------------------------------------------------------------------------
// Compile — helpers
// ---------------------------------------------------------------------------

namespace
{

// Set *out_err if caller asked for it. Always returns nullptr so the caller
// can `return report_err(out_err, "...");` at failure sites.
ggml_hpx_frozen_packet * report_err(const char * * out_err, const char * msg)
{
    if (out_err != nullptr)
    {
        *out_err = msg;
    }
    return nullptr;
}

// Validate that `group` describes a 3-region RMS_NORM_F32 DAG with the
// expected kinds, production callbacks, matching work range, and deps
// 0 -> 1 -> 2 (in any input order).
const char * validate_rms_norm_group(
    const ggml_hpx_cpu_region_group * group,
    int64_t                           expected_n)
{
    if (group == nullptr)
    {
        return "fine_group is null";
    }
    if (group->n_regions != 3)
    {
        return "RMS_NORM_F32 requires exactly 3 regions";
    }
    if (group->regions == nullptr)
    {
        return "regions is null";
    }

    const ggml_hpx_cpu_region & r0 = group->regions[0];
    const ggml_hpx_cpu_region & r1 = group->regions[1];
    const ggml_hpx_cpu_region & r2 = group->regions[2];

    if (r0.kind != GGML_HPX_CPU_REGION_KIND_ELEMENTWISE)
    {
        return "region 0 must be ELEMENTWISE (partial sumsq)";
    }
    if (r1.kind != GGML_HPX_CPU_REGION_KIND_REDUCTION)
    {
        return "region 1 must be REDUCTION (finalize)";
    }
    if (r2.kind != GGML_HPX_CPU_REGION_KIND_ELEMENTWISE)
    {
        return "region 2 must be ELEMENTWISE (apply)";
    }

    if (r0.run_range != ggml_hpx_rms_norm_partial_f32_run_range)
    {
        return "region 0 run_range must be ggml_hpx_rms_norm_partial_f32_run_range";
    }
    if (r1.run_range != ggml_hpx_rms_norm_finalize_f32_run_range)
    {
        return "region 1 run_range must be ggml_hpx_rms_norm_finalize_f32_run_range";
    }
    if (r2.run_range != ggml_hpx_rms_norm_apply_f32_run_range)
    {
        return "region 2 run_range must be ggml_hpx_rms_norm_apply_f32_run_range";
    }

    if (r0.begin != 0 || r1.begin != 0 || r2.begin != 0)
    {
        return "all regions must have begin == 0 (work range covers [0, n))";
    }
    if (r0.end != expected_n || r1.end != expected_n || r2.end != expected_n)
    {
        return "region.end must equal key.shape[0]";
    }

    if (group->n_deps != 2)
    {
        return "RMS_NORM_F32 requires exactly 2 dep edges";
    }
    if (group->deps == nullptr)
    {
        return "deps is null";
    }

    bool saw_0_1 = false;
    bool saw_1_2 = false;
    for (int i = 0; i < 2; ++i)
    {
        const ggml_hpx_dep_edge & d = group->deps[i];
        if (d.src == 0 && d.dst == 1)
        {
            saw_0_1 = true;
        }
        else if (d.src == 1 && d.dst == 2)
        {
            saw_1_2 = true;
        }
        else
        {
            return "unexpected dep edge (expected 0->1 and 1->2)";
        }
    }
    if (!saw_0_1 || !saw_1_2)
    {
        return "missing required dep edge (need 0->1 and 1->2)";
    }

    return nullptr;
}

// Validate that `group` is a 4-region MLP_GATE_UP_F32 DAG:
//   regions[0]: MATMUL, run_range = mul_mat_f32, work [0, out_cols)
//   regions[1]: MATMUL, run_range = mul_mat_f32, work [0, out_cols)
//   regions[2]: ELEMENTWISE, run_range = silu_f32, work [0, out_cols*rows)
//   regions[3]: ELEMENTWISE, run_range = mul_f32,  work [0, out_cols*rows)
//   deps: {0→2, 1→3, 2→3}
const char * validate_mlp_gate_up_group(
    const ggml_hpx_cpu_region_group * group,
    int64_t                           out_cols,
    int64_t                           cols,
    int64_t                           rows)
{
    (void)cols;    // cols is structural identity only; not visible in region metadata

    if (group == nullptr)      return "fine_group is null";
    if (group->n_regions != 4) return "MLP_GATE_UP_F32 requires exactly 4 regions";
    if (group->regions == nullptr) return "regions is null";

    const int64_t n_elem = out_cols * rows;

    const ggml_hpx_cpu_region & r0 = group->regions[0];
    const ggml_hpx_cpu_region & r1 = group->regions[1];
    const ggml_hpx_cpu_region & r2 = group->regions[2];
    const ggml_hpx_cpu_region & r3 = group->regions[3];

    if (r0.kind != GGML_HPX_CPU_REGION_KIND_MATMUL)
        return "region 0 must be MATMUL (gate MUL_MAT)";
    if (r1.kind != GGML_HPX_CPU_REGION_KIND_MATMUL)
        return "region 1 must be MATMUL (up MUL_MAT)";
    if (r2.kind != GGML_HPX_CPU_REGION_KIND_ELEMENTWISE)
        return "region 2 must be ELEMENTWISE (SiLU)";
    if (r3.kind != GGML_HPX_CPU_REGION_KIND_ELEMENTWISE)
        return "region 3 must be ELEMENTWISE (elementwise MUL)";

    if (r0.run_range != ggml_hpx_mul_mat_f32_run_range)
        return "region 0 run_range must be ggml_hpx_mul_mat_f32_run_range";
    if (r1.run_range != ggml_hpx_mul_mat_f32_run_range)
        return "region 1 run_range must be ggml_hpx_mul_mat_f32_run_range";
    if (r2.run_range != ggml_hpx_silu_f32_run_range)
        return "region 2 run_range must be ggml_hpx_silu_f32_run_range";
    if (r3.run_range != ggml_hpx_mul_f32_run_range)
        return "region 3 run_range must be ggml_hpx_mul_f32_run_range";

    if (r0.begin != 0 || r0.end != out_cols)
        return "region 0 work range must be [0, out_cols)";
    if (r1.begin != 0 || r1.end != out_cols)
        return "region 1 work range must be [0, out_cols)";
    if (r2.begin != 0 || r2.end != n_elem)
        return "region 2 work range must be [0, out_cols*rows)";
    if (r3.begin != 0 || r3.end != n_elem)
        return "region 3 work range must be [0, out_cols*rows)";

    if (group->n_deps != 3)     return "MLP_GATE_UP_F32 requires exactly 3 dep edges";
    if (group->deps == nullptr) return "deps is null";

    bool saw_0_2 = false, saw_1_3 = false, saw_2_3 = false;
    for (int i = 0; i < 3; ++i)
    {
        const ggml_hpx_dep_edge & d = group->deps[i];
        if      (d.src == 0 && d.dst == 2) saw_0_2 = true;
        else if (d.src == 1 && d.dst == 3) saw_1_3 = true;
        else if (d.src == 2 && d.dst == 3) saw_2_3 = true;
        else return "unexpected dep edge (expected {0→2, 1→3, 2→3})";
    }
    if (!saw_0_2 || !saw_1_3 || !saw_2_3)
        return "missing required dep edge in MLP_GATE_UP_F32 group";

    return nullptr;
}

// Validate that `group` is a 3-region MLP_GLU_F32 DAG:
//   regions[0]: MATMUL,      run_range = mul_mat_f32, work [0, out_cols)
//   regions[1]: MATMUL,      run_range = mul_mat_f32, work [0, out_cols)
//   regions[2]: ELEMENTWISE, run_range = swiglu_f32,  work [0, out_cols*rows)
//   deps: {0→2, 1→2}
const char * validate_mlp_glu_group(
    const ggml_hpx_cpu_region_group * group,
    int64_t                           out_cols,
    int64_t                           cols,
    int64_t                           rows)
{
    (void)cols;    // cols is structural identity only; not visible in region metadata

    if (group == nullptr)          return "fine_group is null";
    if (group->n_regions != 3)     return "MLP_GLU_F32 requires exactly 3 regions";
    if (group->regions == nullptr) return "regions is null";

    const int64_t n_elem = out_cols * rows;

    const ggml_hpx_cpu_region & r0 = group->regions[0];
    const ggml_hpx_cpu_region & r1 = group->regions[1];
    const ggml_hpx_cpu_region & r2 = group->regions[2];

    if (r0.kind != GGML_HPX_CPU_REGION_KIND_MATMUL)
        return "region 0 must be MATMUL (gate MUL_MAT)";
    if (r1.kind != GGML_HPX_CPU_REGION_KIND_MATMUL)
        return "region 1 must be MATMUL (up MUL_MAT)";
    if (r2.kind != GGML_HPX_CPU_REGION_KIND_ELEMENTWISE)
        return "region 2 must be ELEMENTWISE (fused SWIGLU)";

    if (r0.run_range != ggml_hpx_mul_mat_f32_run_range)
        return "region 0 run_range must be ggml_hpx_mul_mat_f32_run_range";
    if (r1.run_range != ggml_hpx_mul_mat_f32_run_range)
        return "region 1 run_range must be ggml_hpx_mul_mat_f32_run_range";
    if (r2.run_range != ggml_hpx_swiglu_f32_run_range)
        return "region 2 run_range must be ggml_hpx_swiglu_f32_run_range";

    if (r0.begin != 0 || r0.end != out_cols)
        return "region 0 work range must be [0, out_cols)";
    if (r1.begin != 0 || r1.end != out_cols)
        return "region 1 work range must be [0, out_cols)";
    if (r2.begin != 0 || r2.end != n_elem)
        return "region 2 work range must be [0, out_cols*rows)";

    if (group->n_deps != 2)     return "MLP_GLU_F32 requires exactly 2 dep edges";
    if (group->deps == nullptr) return "deps is null";

    bool saw_0_2 = false, saw_1_2 = false;
    for (int i = 0; i < 2; ++i)
    {
        const ggml_hpx_dep_edge & d = group->deps[i];
        if      (d.src == 0 && d.dst == 2) saw_0_2 = true;
        else if (d.src == 1 && d.dst == 2) saw_1_2 = true;
        else return "unexpected dep edge (expected {0→2, 1→2})";
    }
    if (!saw_0_2 || !saw_1_2)
        return "missing required dep edge in MLP_GLU_F32 group";

    return nullptr;
}

}    // namespace

// ---------------------------------------------------------------------------
// Compile
// ---------------------------------------------------------------------------

ggml_hpx_frozen_packet * ggml_hpx_compile_packet(
    const ggml_hpx_cpu_region_group * fine_group,
    const ggml_hpx_packet_plan_key *  key,
    const char * *                    out_err)
{
    if (key == nullptr)
    {
        return report_err(out_err, "key is null");
    }
    if (fine_group == nullptr)
    {
        return report_err(out_err, "fine_group is null");
    }

    // Prototype scope gates.
    if (key->team == GGML_HPX_PACKET_TEAM_PREFILL)
    {
        return report_err(out_err, "PREFILL team is reserved but not implemented in prototype");
    }
    if (key->team != GGML_HPX_PACKET_TEAM_DECODE)
    {
        return report_err(out_err, "unsupported team value");
    }
    if (key->n_lanes == 0)
    {
        return report_err(out_err, "key.n_lanes must be > 0");
    }

    // Generic fine-region validator (sublayer-independent structural check).
    if (const char * msg = ggml_hpx_validate_region_group(fine_group, /*check_acyclic=*/true))
    {
        return report_err(out_err, msg);
    }

    // ── Sublayer dispatch ─────────────────────────────────────────────────────

    if (key->sublayer == GGML_HPX_PACKET_SUBLAYER_RMS_NORM_F32)
    {
        const int64_t n = key->shape[0];
        if (n <= 0)
            return report_err(out_err, "key.shape[0] must be > 0 for RMS_NORM_F32");

        if (const char * msg = validate_rms_norm_group(fine_group, n))
            return report_err(out_err, msg);

        auto * packet = new ggml_hpx_frozen_packet{};
        packet->key             = *key;
        packet->n_steps         = 3;
        packet->steps           = new PacketStep[3];
        packet->arena_offset    = kRmsFrameHeaderSize;
        packet->arena_bytes     = kRmsFrameBytes - kRmsFrameHeaderSize;
        packet->frame_bytes     = kRmsFrameBytes;
        packet->frame_alignment = kRmsFrameAlign;

        packet->res.n_lanes                     = key->n_lanes;
        packet->res.lane_scratch_bytes_per_lane =
            static_cast<uint32_t>(sizeof(float));
        packet->res.reduction_buffer_bytes      =
            static_cast<uint32_t>(sizeof(ggml_hpx_rms_norm_f32_reduce_buffer));
        packet->res.shared_scratch_bytes        = 0;

        packet->ctx_template = std::malloc(packet->arena_bytes);
        if (packet->ctx_template == nullptr)
        {
            delete[] packet->steps;
            delete packet;
            return report_err(out_err, "ctx_template allocation failed");
        }
        std::memset(packet->ctx_template, 0, packet->arena_bytes);

        auto * tmpl_partial =
            reinterpret_cast<ggml_hpx_rms_norm_partial_f32_ctx *>(
                static_cast<char *>(packet->ctx_template)
                + (kRmsPartialOff - kRmsFrameHeaderSize));
        tmpl_partial->x = nullptr;
        tmpl_partial->n = n;

        auto * tmpl_finalize =
            reinterpret_cast<ggml_hpx_rms_norm_finalize_f32_ctx *>(
                static_cast<char *>(packet->ctx_template)
                + (kRmsFinalizeOff - kRmsFrameHeaderSize));
        tmpl_finalize->n   = n;
        tmpl_finalize->eps = 0.0f;

        auto * tmpl_apply =
            reinterpret_cast<ggml_hpx_rms_norm_apply_f32_ctx *>(
                static_cast<char *>(packet->ctx_template)
                + (kRmsApplyOff - kRmsFrameHeaderSize));
        tmpl_apply->x   = nullptr;
        tmpl_apply->dst = nullptr;
        tmpl_apply->n   = n;

        const bool serial = (key->n_lanes == 1);
        packet->steps[0] = PacketStep{
            serial ? kStepKindSerial : kStepKindLaneFanout,
            ggml_hpx_rms_norm_partial_f32_run_range,
            static_cast<uint32_t>(kRmsPartialOff),
            static_cast<uint32_t>(kRmsPartialCtxSize),
            0, n,
            static_cast<uint16_t>(serial ? 1 : key->n_lanes), 0,
        };
        packet->steps[1] = PacketStep{
            kStepKindSerial,
            ggml_hpx_rms_norm_finalize_f32_run_range,
            static_cast<uint32_t>(kRmsFinalizeOff),
            static_cast<uint32_t>(kRmsFinalizeCtxSize),
            0, n, 1, 0,
        };
        packet->steps[2] = PacketStep{
            serial ? kStepKindSerial : kStepKindLaneFanout,
            ggml_hpx_rms_norm_apply_f32_run_range,
            static_cast<uint32_t>(kRmsApplyOff),
            static_cast<uint32_t>(kRmsApplyCtxSize),
            0, n,
            static_cast<uint16_t>(serial ? 1 : key->n_lanes), 0,
        };
        return packet;
    }

    if (key->sublayer == GGML_HPX_PACKET_SUBLAYER_MLP_GATE_UP_F32)
    {
        const int64_t out_cols = key->shape[0];
        const int64_t cols     = key->shape[1];
        const int64_t rows     = key->shape[2];
        if (out_cols <= 0 || cols <= 0 || rows <= 0)
            return report_err(out_err, "key.shape[0..2] must be > 0 for MLP_GATE_UP_F32");

        if (const char * msg = validate_mlp_gate_up_group(fine_group, out_cols, cols, rows))
            return report_err(out_err, msg);

        auto * packet = new ggml_hpx_frozen_packet{};
        packet->key             = *key;
        packet->n_steps         = 4;
        packet->steps           = new PacketStep[4];
        packet->arena_offset    = kMlpFrameHeaderSize;
        packet->arena_bytes     = kMlpFrameBytes - kMlpFrameHeaderSize;
        packet->frame_bytes     = kMlpFrameBytes;
        packet->frame_alignment = kMlpFrameAlign;

        // MLP_GATE_UP has no REDUCTION regions: no lane_scratch or
        // reduction_buffer needed.
        packet->res.n_lanes                     = key->n_lanes;
        packet->res.lane_scratch_bytes_per_lane = 0;
        packet->res.reduction_buffer_bytes      = 0;
        packet->res.shared_scratch_bytes        = 0;

        packet->ctx_template = std::malloc(packet->arena_bytes);
        if (packet->ctx_template == nullptr)
        {
            delete[] packet->steps;
            delete packet;
            return report_err(out_err, "ctx_template allocation failed");
        }
        std::memset(packet->ctx_template, 0, packet->arena_bytes);

        // Helper to get a template pointer at an absolute frame offset.
        auto tmpl_at = [&](size_t abs_off) -> char * {
            return static_cast<char *>(packet->ctx_template)
                   + (abs_off - kMlpFrameHeaderSize);
        };

        // Bake dimensions; data pointers stay null (patched by bind).
        auto * gate_ctx =
            reinterpret_cast<ggml_hpx_mul_mat_f32_ctx *>(tmpl_at(kMlpGateCtxOff));
        gate_ctx->rows     = rows;
        gate_ctx->cols     = cols;
        gate_ctx->out_cols = out_cols;

        auto * up_ctx =
            reinterpret_cast<ggml_hpx_mul_mat_f32_ctx *>(tmpl_at(kMlpUpCtxOff));
        up_ctx->rows     = rows;
        up_ctx->cols     = cols;
        up_ctx->out_cols = out_cols;

        reinterpret_cast<ggml_hpx_silu_f32_ctx *>(tmpl_at(kMlpSiluCtxOff))->n =
            out_cols * rows;

        reinterpret_cast<ggml_hpx_mul_f32_ctx *>(tmpl_at(kMlpEwMulCtxOff))->n =
            out_cols * rows;

        // All 4 regions are MATMUL or ELEMENTWISE — no REDUCTION.
        // Steps emitted in region-index order (0,1,2,3), which is a valid
        // topological ordering of the {0→2, 1→3, 2→3} dep edges.
        const bool   serial    = (key->n_lanes == 1);
        const uint32_t kind    = serial ? kStepKindSerial : kStepKindLaneFanout;
        const uint16_t nlanes  = static_cast<uint16_t>(serial ? 1 : key->n_lanes);
        const int64_t  n_elem  = out_cols * rows;

        packet->steps[0] = PacketStep{
            kind, ggml_hpx_mul_mat_f32_run_range,
            static_cast<uint32_t>(kMlpGateCtxOff),
            static_cast<uint32_t>(kMlpGateCtxSize),
            0, out_cols, nlanes, 0,
        };
        packet->steps[1] = PacketStep{
            kind, ggml_hpx_mul_mat_f32_run_range,
            static_cast<uint32_t>(kMlpUpCtxOff),
            static_cast<uint32_t>(kMlpUpCtxSize),
            0, out_cols, nlanes, 0,
        };
        packet->steps[2] = PacketStep{
            kind, ggml_hpx_silu_f32_run_range,
            static_cast<uint32_t>(kMlpSiluCtxOff),
            static_cast<uint32_t>(kMlpSiluCtxSize),
            0, n_elem, nlanes, 0,
        };
        packet->steps[3] = PacketStep{
            kind, ggml_hpx_mul_f32_run_range,
            static_cast<uint32_t>(kMlpEwMulCtxOff),
            static_cast<uint32_t>(kMlpEwMulCtxSize),
            0, n_elem, nlanes, 0,
        };
        return packet;
    }

    if (key->sublayer == GGML_HPX_PACKET_SUBLAYER_MLP_GLU_F32)
    {
        const int64_t out_cols = key->shape[0];
        const int64_t cols     = key->shape[1];
        const int64_t rows     = key->shape[2];
        if (out_cols <= 0 || cols <= 0 || rows <= 0)
            return report_err(out_err, "key.shape[0..2] must be > 0 for MLP_GLU_F32");

        if (const char * msg = validate_mlp_glu_group(fine_group, out_cols, cols, rows))
            return report_err(out_err, msg);

        auto * packet = new ggml_hpx_frozen_packet{};
        packet->key             = *key;
        packet->n_steps         = 3;
        packet->steps           = new PacketStep[3];
        packet->arena_offset    = kGluFrameHeaderSize;
        packet->arena_bytes     = kGluFrameBytes - kGluFrameHeaderSize;
        packet->frame_bytes     = kGluFrameBytes;
        packet->frame_alignment = kGluFrameAlign;

        // MLP_GLU has no REDUCTION regions: no lane_scratch or
        // reduction_buffer needed.
        packet->res.n_lanes                     = key->n_lanes;
        packet->res.lane_scratch_bytes_per_lane = 0;
        packet->res.reduction_buffer_bytes      = 0;
        packet->res.shared_scratch_bytes        = 0;

        packet->ctx_template = std::malloc(packet->arena_bytes);
        if (packet->ctx_template == nullptr)
        {
            delete[] packet->steps;
            delete packet;
            return report_err(out_err, "ctx_template allocation failed");
        }
        std::memset(packet->ctx_template, 0, packet->arena_bytes);

        auto tmpl_at = [&](size_t abs_off) -> char * {
            return static_cast<char *>(packet->ctx_template)
                   + (abs_off - kGluFrameHeaderSize);
        };

        // Bake dimensions; data pointers stay null (patched by bind).
        auto * gate_ctx =
            reinterpret_cast<ggml_hpx_mul_mat_f32_ctx *>(tmpl_at(kGluGateCtxOff));
        gate_ctx->rows     = rows;
        gate_ctx->cols     = cols;
        gate_ctx->out_cols = out_cols;

        auto * up_ctx =
            reinterpret_cast<ggml_hpx_mul_mat_f32_ctx *>(tmpl_at(kGluUpCtxOff));
        up_ctx->rows     = rows;
        up_ctx->cols     = cols;
        up_ctx->out_cols = out_cols;

        reinterpret_cast<ggml_hpx_swiglu_f32_ctx *>(tmpl_at(kGluSwigluCtxOff))->n =
            out_cols * rows;

        // All 3 regions are MATMUL or ELEMENTWISE — no REDUCTION.
        // Steps emitted in region-index order (0,1,2), which is a valid
        // topological ordering of the {0→2, 1→2} dep edges.
        const bool     serial  = (key->n_lanes == 1);
        const uint32_t kind    = serial ? kStepKindSerial : kStepKindLaneFanout;
        const uint16_t nlanes  = static_cast<uint16_t>(serial ? 1 : key->n_lanes);
        const int64_t  n_elem  = out_cols * rows;

        packet->steps[0] = PacketStep{
            kind, ggml_hpx_mul_mat_f32_run_range,
            static_cast<uint32_t>(kGluGateCtxOff),
            static_cast<uint32_t>(kGluGateCtxSize),
            0, out_cols, nlanes, 0,
        };
        packet->steps[1] = PacketStep{
            kind, ggml_hpx_mul_mat_f32_run_range,
            static_cast<uint32_t>(kGluUpCtxOff),
            static_cast<uint32_t>(kGluUpCtxSize),
            0, out_cols, nlanes, 0,
        };
        packet->steps[2] = PacketStep{
            kind, ggml_hpx_swiglu_f32_run_range,
            static_cast<uint32_t>(kGluSwigluCtxOff),
            static_cast<uint32_t>(kGluSwigluCtxSize),
            0, n_elem, nlanes, 0,
        };
        return packet;
    }

    return report_err(out_err, "unsupported sublayer");
}

void ggml_hpx_free_packet(ggml_hpx_frozen_packet * packet)
{
    if (packet == nullptr)
    {
        return;
    }
    std::free(packet->ctx_template);
    delete[] packet->steps;
    delete packet;
}

// ---------------------------------------------------------------------------
// Introspection
// ---------------------------------------------------------------------------

size_t ggml_hpx_packet_frame_size(const ggml_hpx_frozen_packet * packet)
{
    assert(packet != nullptr);
    return packet->frame_bytes;
}

size_t ggml_hpx_packet_frame_align(const ggml_hpx_frozen_packet * packet)
{
    assert(packet != nullptr);
    return packet->frame_alignment;
}

void ggml_hpx_packet_get_resource_requirements(
    const ggml_hpx_frozen_packet *          packet,
    ggml_hpx_packet_resource_requirements * out)
{
    assert(packet != nullptr);
    assert(out != nullptr);
    *out = packet->res;
}

const ggml_hpx_packet_plan_key * ggml_hpx_packet_key(
    const ggml_hpx_frozen_packet * packet)
{
    assert(packet != nullptr);
    return &packet->key;
}

#ifdef GGML_HPX_PACKET_TEST_API

uint32_t ggml_hpx_packet_n_steps(const ggml_hpx_frozen_packet * packet)
{
    assert(packet != nullptr);
    return packet->n_steps;
}

const ggml_hpx_packet_step * ggml_hpx_packet_steps(
    const ggml_hpx_frozen_packet * packet)
{
    assert(packet != nullptr);
    // PacketStep matches ggml_hpx_packet_step layout by construction.
    return reinterpret_cast<const ggml_hpx_packet_step *>(packet->steps);
}

#endif    // GGML_HPX_PACKET_TEST_API

// ---------------------------------------------------------------------------
// Frame init
// ---------------------------------------------------------------------------

void ggml_hpx_packet_frame_init(
    ggml_hpx_packet_frame *         frame,
    const ggml_hpx_frozen_packet *  packet)
{
    assert(frame != nullptr);
    assert(packet != nullptr);

    frame->packet = packet;
    std::memcpy(
        reinterpret_cast<char *>(frame) + packet->arena_offset,
        packet->ctx_template,
        packet->arena_bytes);
}

// ---------------------------------------------------------------------------
// Typed bind — RMS_NORM_F32
// ---------------------------------------------------------------------------

void ggml_hpx_bind_rms_norm_packet(
    ggml_hpx_packet_frame *           frame,
    const ggml_hpx_rms_norm_binding * binding)
{
    assert(frame != nullptr);
    assert(binding != nullptr);
    assert(frame->packet != nullptr);
    assert(frame->packet->key.sublayer == GGML_HPX_PACKET_SUBLAYER_RMS_NORM_F32);
    assert(binding->n == frame->packet->key.shape[0]);
    assert(binding->x != nullptr);
    assert(binding->dst != nullptr);

    char * base = reinterpret_cast<char *>(frame);

    // partial_ctx.x
    *reinterpret_cast<const float **>(base + kRmsPartialOff + kRmsPartialXFieldOff) =
        binding->x;

    // finalize_ctx.eps
    *reinterpret_cast<float *>(base + kRmsFinalizeOff + kRmsFinalizeEpsFieldOff) =
        binding->eps;

    // apply_ctx.x, apply_ctx.dst
    *reinterpret_cast<const float **>(base + kRmsApplyOff + kRmsApplyXFieldOff) =
        binding->x;
    *reinterpret_cast<float **>(base + kRmsApplyOff + kRmsApplyDstFieldOff) =
        binding->dst;
}

// ---------------------------------------------------------------------------
// Typed bind — MLP_GATE_UP_F32
// ---------------------------------------------------------------------------

void ggml_hpx_bind_mlp_gate_up_packet(
    ggml_hpx_packet_frame *                  frame,
    const ggml_hpx_mlp_gate_up_binding *     binding)
{
    assert(frame != nullptr);
    assert(binding != nullptr);
    assert(frame->packet != nullptr);
    assert(frame->packet->key.sublayer == GGML_HPX_PACKET_SUBLAYER_MLP_GATE_UP_F32);
    assert(binding->w_gate   != nullptr);
    assert(binding->w_up     != nullptr);
    assert(binding->x        != nullptr);
    assert(binding->gate     != nullptr);
    assert(binding->up       != nullptr);
    assert(binding->gate_act != nullptr);
    assert(binding->out      != nullptr);

    char * const base = reinterpret_cast<char *>(frame);

    // gate MUL_MAT: x = input activations, w = W_gate, y = gate output
    *reinterpret_cast<const float **>(base + kMlpGateCtxOff + kMlpMulMatXOff) = binding->x;
    *reinterpret_cast<const float **>(base + kMlpGateCtxOff + kMlpMulMatWOff) = binding->w_gate;
    *reinterpret_cast<float **>      (base + kMlpGateCtxOff + kMlpMulMatYOff) = binding->gate;

    // up MUL_MAT: x = input activations, w = W_up, y = up output
    *reinterpret_cast<const float **>(base + kMlpUpCtxOff + kMlpMulMatXOff) = binding->x;
    *reinterpret_cast<const float **>(base + kMlpUpCtxOff + kMlpMulMatWOff) = binding->w_up;
    *reinterpret_cast<float **>      (base + kMlpUpCtxOff + kMlpMulMatYOff) = binding->up;

    // SiLU: x = gate output, dst = gate_act output
    *reinterpret_cast<const float **>(base + kMlpSiluCtxOff + kMlpSiluXOff)   =
        binding->gate;
    *reinterpret_cast<float **>      (base + kMlpSiluCtxOff + kMlpSiluDstOff) =
        binding->gate_act;

    // elementwise MUL: a = gate_act, b = up output, dst = final output
    *reinterpret_cast<const float **>(base + kMlpEwMulCtxOff + kMlpEwMulAOff)   =
        static_cast<const float *>(binding->gate_act);
    *reinterpret_cast<const float **>(base + kMlpEwMulCtxOff + kMlpEwMulBOff)   =
        static_cast<const float *>(binding->up);
    *reinterpret_cast<float **>      (base + kMlpEwMulCtxOff + kMlpEwMulDstOff) =
        binding->out;
}

// ---------------------------------------------------------------------------
// Typed bind — MLP_GLU_F32
// ---------------------------------------------------------------------------

void ggml_hpx_bind_mlp_glu_packet(
    ggml_hpx_packet_frame *            frame,
    const ggml_hpx_mlp_glu_binding *   binding)
{
    assert(frame != nullptr);
    assert(binding != nullptr);
    assert(frame->packet != nullptr);
    assert(frame->packet->key.sublayer == GGML_HPX_PACKET_SUBLAYER_MLP_GLU_F32);
    assert(binding->w_gate != nullptr);
    assert(binding->w_up   != nullptr);
    assert(binding->x      != nullptr);
    assert(binding->gate   != nullptr);
    assert(binding->up     != nullptr);
    assert(binding->out    != nullptr);

    char * const base = reinterpret_cast<char *>(frame);

    // gate MUL_MAT: x = input activations, w = W_gate, y = gate output
    *reinterpret_cast<const float **>(base + kGluGateCtxOff + kMlpMulMatXOff) = binding->x;
    *reinterpret_cast<const float **>(base + kGluGateCtxOff + kMlpMulMatWOff) = binding->w_gate;
    *reinterpret_cast<float **>      (base + kGluGateCtxOff + kMlpMulMatYOff) = binding->gate;

    // up MUL_MAT: x = input activations, w = W_up, y = up output
    *reinterpret_cast<const float **>(base + kGluUpCtxOff + kMlpMulMatXOff) = binding->x;
    *reinterpret_cast<const float **>(base + kGluUpCtxOff + kMlpMulMatWOff) = binding->w_up;
    *reinterpret_cast<float **>      (base + kGluUpCtxOff + kMlpMulMatYOff) = binding->up;

    // Fused SWIGLU: gate = gate MUL_MAT output, up = up MUL_MAT output,
    // dst = final output. The kernel computes
    //   dst[i] = silu(gate[i]) * up[i]
    // in a single pass, replacing the separate SiLU + elementwise MUL of the
    // MLP_GATE_UP_F32 variant.
    *reinterpret_cast<const float **>(base + kGluSwigluCtxOff + kGluSwigluGateOff) =
        static_cast<const float *>(binding->gate);
    *reinterpret_cast<const float **>(base + kGluSwigluCtxOff + kGluSwigluUpOff)   =
        static_cast<const float *>(binding->up);
    *reinterpret_cast<float **>      (base + kGluSwigluCtxOff + kGluSwigluDstOff)  =
        binding->out;
}

// ---------------------------------------------------------------------------
// Run
// ---------------------------------------------------------------------------
//
// I4 compliance notes:
//   - No hpx::async, hpx::dataflow, hpx::shared_future, wait_all.
//   - No per-step heap.
//   - LANE_FANOUT uses hpx::experimental::for_loop on the pinned decode Exec.
//   - SERIAL runs on the caller thread; the preceding LANE_FANOUT's
//     synchronous for_loop join provides the happens-before edge.

void ggml_hpx_run_frozen_packet(
    ggml_hpx_packet_runtime *       runtime,
    const ggml_hpx_frozen_packet *  packet,
    ggml_hpx_packet_frame *         frame,
    ggml_hpx_region_resources *     resources)
{
    assert(runtime != nullptr);
    assert(packet != nullptr);
    assert(frame != nullptr);
    assert(resources != nullptr);
    assert(frame->packet == packet);
    assert(resources->n_lanes == static_cast<int>(packet->key.n_lanes));
    assert(packet->key.team == GGML_HPX_PACKET_TEAM_DECODE);
    assert(static_cast<uint32_t>(resources->n_lanes) == runtime->n_decode_threads);

    char * const frame_base = reinterpret_cast<char *>(frame);

    for (uint32_t s = 0; s < packet->n_steps; ++s)
    {
        const PacketStep & step = packet->steps[s];
        void * const       ctx  = frame_base + step.ctx_offset;

        if (step.kind == kStepKindSerial)
        {
            step.run_range(ctx,
                /*ith=*/0,
                /*nth=*/1,
                step.work_begin,
                step.work_end,
                resources);
            continue;
        }

        // LANE_FANOUT. Match ggml-hpx-region-exec.cpp's partitioning exactly:
        //   lane i gets [work_begin + span*i/n_lanes, work_begin + span*(i+1)/n_lanes).
        // Unlike the generic runner, every lane is always dispatched (no
        // skip-on-empty). The run_range callbacks handle an empty range as a
        // zero-contribution no-op, so lane_scratch[i] is always written.

        const int     n_lanes = step.n_lanes;
        const int64_t span    = step.work_end - step.work_begin;
        const int64_t base    = step.work_begin;

        hpx::experimental::for_loop(
            hpx::execution::par.on(runtime->decode_exec),
            0,
            n_lanes,
            [=](int ith)
            {
                const int64_t b = base + (span *  ith)      / n_lanes;
                const int64_t e = base + (span * (ith + 1)) / n_lanes;
                step.run_range(ctx, ith, n_lanes, b, e, resources);
            });
    }
}

#endif    // GGML_HPX_REGION_DAG
