#pragma once

// ggml-hpx-packet.h
//
// ============================================================================
// Purpose
// ============================================================================
//
// The fine-region DAG defined in ggml-hpx-region-dag.h is correct and
// expressive, but Phase 5 benchmarking showed that the generic per-call
// DAG machinery (topological sort, per-region shared_future, dataflow
// composition, per-region lane-vector heap allocation) dominates cost at
// the RMS_NORM-class granularity. The math, the callbacks, the outer HPX
// entry, and per-call heap churn were all individually ruled out as the
// bottleneck.
//
// This header defines a second execution surface that keeps the fine-region
// DAG as the planning / legality / resource-flow representation but replaces
// the generic per-call scheduler with a compiled, linearized, reusable
// execution unit — a "frozen packet" — sized at the sublayer level.
//
// A frozen packet is:
//   - compiled once from a ggml_hpx_cpu_region_group
//   - immutable afterwards (shareable across threads and invocations)
//   - executed as a straight-line for-loop over precomputed steps
//   - free of hpx::async, hpx::dataflow, hpx::shared_future, wait_all,
//     and any per-step heap allocation at run time
//
// The fine-region DAG remains the authoritative source of truth. Packets
// are a compiled projection of it. `ggml_hpx_run_region_group` stays in
// place (correctness reference, debug path, prototype source input).
//
// ============================================================================
// Locked invariants
// ============================================================================
//
//  I1. Immutable packet, mutable frame, runtime state
//      - ggml_hpx_frozen_packet is immutable once compiled; shareable across
//        threads and invocations.
//      - ggml_hpx_packet_frame is per-invocation mutable state (patched ctx
//        arena). Concurrent invocations require separate frames.
//      - ggml_hpx_packet_runtime carries executor handles only; no packet
//        cache in this prototype.
//
//  I2. Bind never mutates the packet
//      - Every bind function takes (frame *, binding *). None take a
//        non-const packet *. This is enforced by the signatures below.
//
//  I3. Team is part of structural identity
//      - A packet is compiled for exactly one persistent worker team
//        (decode or prefill). The team is a field of ggml_hpx_packet_plan_key
//        and selects which pinned per-team executor the runtime uses for
//        lane fan-out.
//      - packet->key.n_lanes must equal the target team's worker count.
//      - Prototype scope: only the DECODE team is wired. PREFILL is reserved
//        in the enum for structural identity completeness but is not
//        executed by the first implementation. Prefill requires dep-aware
//        scheduling (see ggml-hpx-runtime.cpp) that is out of scope here.
//
//  I4. Run loop has no generic DAG machinery
//      - No hpx::async, hpx::dataflow, hpx::shared_future, wait_all.
//      - No per-step heap.
//      - Lane fan-out uses hpx::experimental::for_loop on a pinned per-team
//        Exec held by ggml_hpx_packet_runtime. The Exec is built once at
//        runtime_create via
//          hpx::parallel::execution::with_processing_units_count(
//              Scheduler{}, n_team_workers)
//        and reused across calls. No per-step Exec construction, no
//        threadpool lookup, no ggml_hpx_runtime_dispatch_* indirection.
//      - SERIAL steps run on the caller thread; the synchronous for_loop
//        join supplies the happens-before needed by REDUCTION finalizers.
//
//  I5. Resources are caller-passed (prototype)
//      - ggml_hpx_region_resources is supplied on every run_frozen_packet
//        call. Caller is responsible for ensuring resources->n_lanes equals
//        packet->key.n_lanes. Runtime-owned resources may be added later.
//
//  I6. Typed bind per sublayer, not a generic slot walker
//      - Every sublayer kind ships one typed binding struct and one bind
//        function (e.g. ggml_hpx_rms_norm_binding +
//        ggml_hpx_bind_rms_norm_packet). No tagged-union slots, no runtime
//        type dispatch at bind time.
//
//  I7. Frame storage is caller-owned, size-opaque
//      - Size is queried from the packet via ggml_hpx_packet_frame_size.
//      - Storage is supplied by the caller (stack, heap, arena — caller's
//        choice). frame_init initializes in place; no heap allocation.
//
// ============================================================================
// Typical call sequence (prototype)
// ============================================================================
//
//   auto * rt = ggml_hpx_packet_runtime_create(/*n_decode_threads=*/4);
//
//   // Build the same 3-region RMS_NORM group the fine-DAG path would use.
//   ggml_hpx_cpu_region_group fine = build_rms_norm_group(...);
//
//   ggml_hpx_packet_plan_key key{ ... };    // includes team + shape + n_lanes
//   auto * packet = ggml_hpx_compile_packet(&fine, &key, /*out_err=*/nullptr);
//
//   // Query size + alignment from the compiled packet. Do NOT assume a fixed
//   // alignment; sublayers with SIMD ctx fields may require more than
//   // alignof(std::max_align_t).
//   const size_t frame_sz = ggml_hpx_packet_frame_size(packet);
//   const size_t frame_al = ggml_hpx_packet_frame_align(packet);
//
//   // Standard C++17 over-aligned allocation. Variable-length arrays are
//   // non-standard C++ and intentionally avoided here.
//   void * raw = ::operator new(frame_sz, std::align_val_t{frame_al});
//   auto * frame = static_cast<ggml_hpx_packet_frame *>(raw);
//   ggml_hpx_packet_frame_init(frame, packet);
//
//   // Size caller-owned resources per the packet's requirements.
//   ggml_hpx_packet_resource_requirements req{};
//   ggml_hpx_packet_get_resource_requirements(packet, &req);
//   ggml_hpx_region_resources resources = build_resources(req);
//
//   // Per invocation:
//   ggml_hpx_rms_norm_binding b{ x, dst, n, eps };
//   ggml_hpx_bind_rms_norm_packet(frame, &b);
//   ggml_hpx_run_frozen_packet(rt, packet, frame, &resources);
//
//   ::operator delete(raw, std::align_val_t{frame_al});
//   ggml_hpx_free_packet(packet);
//   ggml_hpx_packet_runtime_destroy(rt);
//
// Note: the packet runtime in this prototype is self-sufficient — it owns
// one pinned decode Exec and nothing else. No ggml_hpx_runtime is required.
// The runtime connection will be reinstated once something downstream
// (runtime-owned resources, scratch, team-aware dispatch) genuinely needs
// it; until then the packet runtime's only input is the decode team size.
//
// ============================================================================

#ifdef GGML_HPX_REGION_DAG

#include "ggml-hpx-region-dag.h"    // ggml_hpx_cpu_region_group,
                                    // ggml_hpx_run_range_fn,
                                    // ggml_hpx_region_resources

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

// ---------------------------------------------------------------------------
// Opaque handles
// ---------------------------------------------------------------------------
//
// Full definitions live in ggml-hpx-packet.cpp. Callers hold pointers only.
//
// - ggml_hpx_frozen_packet   : immutable; shareable across threads and calls.
// - ggml_hpx_packet_frame    : mutable per-invocation; one per concurrent call.
// - ggml_hpx_packet_runtime  : owns the pinned per-team Exec (prototype:
//                              decode only); no external runtime pointer.

struct ggml_hpx_frozen_packet;
struct ggml_hpx_packet_frame;
struct ggml_hpx_packet_runtime;

// ---------------------------------------------------------------------------
// Persistent worker team selector
// ---------------------------------------------------------------------------
//
// Identifies which persistent team a packet was compiled against. The packet
// runtime holds one pinned Exec per team (see runtime lifecycle below) and
// selects the matching one at dispatch time based on packet->key.team.
//
// Prototype scope: only DECODE is wired. PREFILL is reserved in the enum so
// that structural identity (plan_key) is stable across the prototype and
// follow-up work, but compiling a packet with team == PREFILL returns
// nullptr in the first implementation.

typedef enum ggml_hpx_packet_team
{
    GGML_HPX_PACKET_TEAM_DECODE  = 0,
    GGML_HPX_PACKET_TEAM_PREFILL = 1,    // reserved; not implemented in prototype
} ggml_hpx_packet_team;

// ---------------------------------------------------------------------------
// Sublayer identity
// ---------------------------------------------------------------------------
//
// Every compiled packet is for exactly one sublayer kind. The id selects the
// ctx-arena layout convention that the matching bind function relies on.
//
// This enum is append-only: new sublayers get new ids; existing ids do not
// shift. Callers compiling their own sublayers define ids in the custom
// range (>= GGML_HPX_PACKET_SUBLAYER_CUSTOM_BASE).

typedef enum ggml_hpx_packet_sublayer
{
    GGML_HPX_PACKET_SUBLAYER_INVALID         = 0,
    GGML_HPX_PACKET_SUBLAYER_RMS_NORM_F32    = 1,    // 3-region: partial / finalize / apply
    GGML_HPX_PACKET_SUBLAYER_MLP_GATE_UP_F32 = 2,    // 4-region: gate MUL_MAT / up MUL_MAT / SiLU / MUL
    GGML_HPX_PACKET_SUBLAYER_MLP_GLU_F32     = 3,    // 3-region: gate MUL_MAT / up MUL_MAT / fused SWIGLU
    // future: ATTENTION_QKV, ...

    GGML_HPX_PACKET_SUBLAYER_CUSTOM_BASE     = 1 << 16,
} ggml_hpx_packet_sublayer;

// ---------------------------------------------------------------------------
// Plan key
// ---------------------------------------------------------------------------
//
// Structural identity of a packet. If any field differs, a distinct packet
// must be compiled. This is also the cache key (when a cache is added).
//
// Fields:
//   sublayer     : which sublayer kind; drives ctx-arena layout convention
//   team         : persistent worker team the packet targets
//   n_lanes        : fan-out width; must match team worker count
//   dtype          : ggml_type-equivalent for the packet's primary tensor
//   seq_regime     : caller-defined bucket (e.g. 0 = decode-1, 1 = prefill-N)
//   policy_version : bump whenever the compiler's lowering rules change
//                    (step kind selection, ctx layout convention, fan-out
//                    policy). Included in the structural identity so cached
//                    packets compiled under older rules are never reused.
//   shape          : up to four sublayer-specific structural dimensions
//                    (e.g. for RMS_NORM_F32 : { n, 0, 0, 0 })
//   extra          : packed caller-defined structural bits (e.g. has_bias,
//                    rope_base, head_dim). Bit layout is per-sublayer.
//
// Two keys compare equal iff all fields compare equal byte-for-byte.
// No padding should be present; the struct is declared with explicitly
// sized fields for stable hashing.

typedef struct ggml_hpx_packet_plan_key
{
    uint32_t    sublayer;       // ggml_hpx_packet_sublayer
    uint32_t    team;           // ggml_hpx_packet_team
    uint32_t    n_lanes;
    uint32_t    dtype;          // ggml_type
    uint32_t    seq_regime;
    uint32_t    policy_version; // bump to invalidate stale compiled packets
    int64_t     shape[4];
    uint64_t    extra;
} ggml_hpx_packet_plan_key;

// Layout contract, enforced at compile time.
// 6 * sizeof(uint32_t) + 4 * sizeof(int64_t) + sizeof(uint64_t) = 64 bytes.
// Any change here is a structural identity change — bump policy_version
// in every caller whose compile output depends on this layout.
#ifdef __cplusplus
static_assert(sizeof(ggml_hpx_packet_plan_key) == 64,
    "ggml_hpx_packet_plan_key must be exactly 64 bytes (no padding)");
static_assert(alignof(ggml_hpx_packet_plan_key) == 8,
    "ggml_hpx_packet_plan_key must be 8-byte aligned");
#else
_Static_assert(sizeof(ggml_hpx_packet_plan_key) == 64,
    "ggml_hpx_packet_plan_key must be exactly 64 bytes (no padding)");
_Static_assert(_Alignof(ggml_hpx_packet_plan_key) == 8,
    "ggml_hpx_packet_plan_key must be 8-byte aligned");
#endif

// ---------------------------------------------------------------------------
// Step kind and step struct — TEST / DEBUG ONLY
// ---------------------------------------------------------------------------
//
// The packet's linearized program is an implementation detail. The step
// enum and struct exist so tests and debuggers can walk the compiled
// schedule and so documentation can reference the kinds by name. They are
// NOT part of the production ABI:
//
//   - the struct exposes ggml_hpx_run_range_fn directly, which is a stronger
//     internal coupling than any other type in this header
//   - callers never construct these; the compiler owns their layout
//   - field additions / renames do not require a policy_version bump
//
// Gated behind GGML_HPX_PACKET_TEST_API together with the step accessors
// (below). Production code must not depend on either.
//
// SERIAL      : one call on the caller thread. No HPX crossing. Used for
//               REDUCTION regions and any other single-threaded finalizer.
// LANE_FANOUT : one hpx::experimental::for_loop on the packet runtime's
//               pinned per-team Exec, with bounds [0, step.n_lanes). The
//               loop index is passed to run_range as `ith`. The for_loop
//               returns only when every lane has completed, so REDUCTION
//               steps after a LANE_FANOUT see a total happens-before edge
//               without explicit wait_all or shared_future.

#ifdef GGML_HPX_PACKET_TEST_API

typedef enum ggml_hpx_packet_step_kind
{
    GGML_HPX_PACKET_STEP_SERIAL      = 0,
    GGML_HPX_PACKET_STEP_LANE_FANOUT = 1,
} ggml_hpx_packet_step_kind;

typedef struct ggml_hpx_packet_step
{
    uint32_t               kind;           // ggml_hpx_packet_step_kind
    ggml_hpx_run_range_fn  run_range;
    uint32_t               ctx_offset;     // byte offset into frame->ctx_arena
    uint32_t               ctx_size;       // diagnostics / bounds checks
    int64_t                work_begin;     // inclusive; copied from region
    int64_t                work_end;       // exclusive
    uint16_t               n_lanes;        // fan-out width; 1 for SERIAL
    uint16_t               flags;          // reserved; must be 0
} ggml_hpx_packet_step;

#endif    // GGML_HPX_PACKET_TEST_API

// ---------------------------------------------------------------------------
// Runtime lifecycle
// ---------------------------------------------------------------------------
//
// The packet runtime owns the per-team Exec objects used for lane fan-out.
// The decode Exec is built once at runtime_create via
//   hpx::parallel::execution::with_processing_units_count(
//       Scheduler{}, n_decode_threads)
// and pinned for the lifetime of the packet runtime. It is reused across
// every run_frozen_packet call; no per-step Exec construction, no
// threadpool lookup, no ggml_hpx_runtime_dispatch_* crossing.
//
// The prototype packet runtime is self-sufficient: it takes the decode team
// size as its only input and owns no other state (no ggml_hpx_runtime
// pointer, no scratch, no packet cache). HPX itself must already be started
// (e.g. via ggml_hpx_tpool_start()) before runtime_create is called.
//
// Prototype scope: only the DECODE team's Exec is constructed. run_frozen_packet
// with packet->key.team == PREFILL aborts (debug) / is undefined (release).
//
// Preconditions:
//   - HPX runtime is running (ggml_hpx_tpool_start() has been called)
//   - n_decode_threads > 0

struct ggml_hpx_packet_runtime * ggml_hpx_packet_runtime_create(
    uint32_t n_decode_threads);

void ggml_hpx_packet_runtime_destroy(
    struct ggml_hpx_packet_runtime * runtime);

// ---------------------------------------------------------------------------
// Compile
// ---------------------------------------------------------------------------
//
// Produce an immutable frozen packet from a fine-region group.
//
//   fine_group : source of truth; must already satisfy
//                ggml_hpx_validate_region_group(fine_group, true).
//                Read during compilation only; not retained.
//   key        : structural identity. key.sublayer determines the
//                ctx-arena layout convention. All key fields must be
//                populated; unused shape[] slots should be zero.
//
// Compilation:
//   - topologically orders the fine regions (Kahn)
//   - maps REDUCTION-kind regions to SERIAL steps
//   - maps other kinds to LANE_FANOUT steps (with n_lanes = key.n_lanes)
//   - lays out a ctx arena per the sublayer layout convention, baking
//     structural constants (shape, eps-if-in-key, etc.) into place and
//     leaving per-invocation fields zero-initialized for bind to patch
//   - precomputes step.ctx_offset and step.ctx_size
//
// Returns a non-null pointer to an immutable packet on success, or null
// if the fine group violates sublayer expectations (e.g. wrong region count
// or kind sequence for key.sublayer, or the group fails the fine-region
// validator). In the prototype, key.team == PREFILL also returns null
// (with a matching out_err string); PREFILL is reserved for follow-up work.
//
// Error reporting:
//   out_err is optional. If non-null, on failure it is set to a static
//   C-string describing the first violation (no allocation). On success
//   it is left untouched. Callers may pass NULL to suppress reporting;
//   the function then asserts in debug builds and returns null in release.
//
// Callers own the returned pointer and must free it with
// ggml_hpx_free_packet when done.

struct ggml_hpx_frozen_packet * ggml_hpx_compile_packet(
    const ggml_hpx_cpu_region_group *  fine_group,
    const ggml_hpx_packet_plan_key *   key,
    const char * *                     out_err);      // optional; may be NULL

void ggml_hpx_free_packet(
    struct ggml_hpx_frozen_packet * packet);

// ---------------------------------------------------------------------------
// Packet introspection (production)
// ---------------------------------------------------------------------------
//
// All helpers are const; they read the immutable packet state only. These
// are the only introspection production code should depend on.
//
// frame_size / frame_align : required for caller-owned frame storage.
//   The frame buffer must be at least frame_size bytes and aligned to at
//   least frame_align bytes. Alignment may exceed alignof(max_align_t)
//   when the sublayer's ctx arena contains SIMD-aligned fields; callers
//   must not assume a fixed alignment value.
//
// get_resource_requirements : required for sizing the caller-owned
//   ggml_hpx_region_resources that will be passed to run_frozen_packet.
//   Writes the exact n_lanes, lane_scratch_bytes_per_lane,
//   reduction_buffer_bytes, and shared_scratch_bytes this packet assumes.
//   Passing resources sized below any of these is undefined behaviour.
//   run_frozen_packet debug-asserts the check; release builds do not.
//
// key : returns a stable pointer valid for the packet's lifetime.

size_t ggml_hpx_packet_frame_size(
    const struct ggml_hpx_frozen_packet * packet);

size_t ggml_hpx_packet_frame_align(
    const struct ggml_hpx_frozen_packet * packet);

typedef struct ggml_hpx_packet_resource_requirements
{
    uint32_t n_lanes;                       // must equal resources->n_lanes
    uint32_t lane_scratch_bytes_per_lane;   // bytes addressable at lane_scratch[i]
    uint32_t reduction_buffer_bytes;        // bytes addressable at reduction_buffer
    uint32_t shared_scratch_bytes;          // bytes addressable at shared_scratch
} ggml_hpx_packet_resource_requirements;

void ggml_hpx_packet_get_resource_requirements(
    const struct ggml_hpx_frozen_packet *     packet,
    ggml_hpx_packet_resource_requirements *   out);

const ggml_hpx_packet_plan_key * ggml_hpx_packet_key(
    const struct ggml_hpx_frozen_packet * packet);

// ---------------------------------------------------------------------------
// Packet introspection (tests / debug)
// ---------------------------------------------------------------------------
//
// These accessors expose the compiled step array. They exist for tests
// and debugging and may change without notice. Production code must not
// depend on them. Gated to keep them out of the production ABI surface.

#ifdef GGML_HPX_PACKET_TEST_API

uint32_t ggml_hpx_packet_n_steps(
    const struct ggml_hpx_frozen_packet * packet);

const ggml_hpx_packet_step * ggml_hpx_packet_steps(
    const struct ggml_hpx_frozen_packet * packet);

#endif    // GGML_HPX_PACKET_TEST_API

// ---------------------------------------------------------------------------
// Frame lifecycle
// ---------------------------------------------------------------------------
//
// Storage is caller-owned.
//
//   frame_storage : at least ggml_hpx_packet_frame_size(packet) bytes,
//                   aligned to at least ggml_hpx_packet_frame_align(packet)
//                   bytes. Callers must not assume alignof(max_align_t) is
//                   sufficient — sublayers with SIMD-aligned ctx fields may
//                   require a larger alignment.
//
// frame_init copies the packet's ctx_arena template into the frame. No
// heap allocation. After init the frame is ready for bind.
//
// A frame is valid for any number of bind+run cycles as long as the
// referenced packet remains alive. A frame is NOT safe to share across
// concurrent invocations: one frame per in-flight call.

void ggml_hpx_packet_frame_init(
    struct ggml_hpx_packet_frame *         frame,
    const struct ggml_hpx_frozen_packet *  packet);

// ---------------------------------------------------------------------------
// Typed binding — RMS_NORM_F32 (prototype)
// ---------------------------------------------------------------------------
//
// One binding struct per sublayer. This is the complete, typed set of
// per-invocation values for a packet with
//   key.sublayer == GGML_HPX_PACKET_SUBLAYER_RMS_NORM_F32.
//
// Fields:
//   x    : [n] input activations
//   dst  : [n] output; may alias x iff the ggml op semantics allow it
//   n    : length; must equal packet->key.shape[0]
//   eps  : numerical stabilizer
//
// Hard preconditions (violating any of these is undefined behaviour;
// debug-asserted, not checked in release):
//
//   - The frame must have been ggml_hpx_packet_frame_init'd against a packet
//     whose key.sublayer == GGML_HPX_PACKET_SUBLAYER_RMS_NORM_F32. Passing a
//     frame from a different sublayer's packet corrupts the ctx arena
//     because each sublayer has its own private layout convention.
//   - binding->n must equal that packet's key.shape[0].
//   - binding->x and binding->dst must point to at least n floats.
//
// The bind function patches these values into the frame's ctx arena at
// offsets fixed by the RMS_NORM_F32 layout convention. It must not read
// or modify the packet.

typedef struct ggml_hpx_rms_norm_binding
{
    const float *  x;
    float *        dst;
    int64_t        n;
    float          eps;
} ggml_hpx_rms_norm_binding;

void ggml_hpx_bind_rms_norm_packet(
    struct ggml_hpx_packet_frame *      frame,
    const ggml_hpx_rms_norm_binding *   binding);

// ---------------------------------------------------------------------------
// Typed binding — MLP_GATE_UP_F32
// ---------------------------------------------------------------------------
//
// One binding struct for
//   key.sublayer == GGML_HPX_PACKET_SUBLAYER_MLP_GATE_UP_F32.
//
// Shape convention (must match key.shape[] used at compile time):
//   shape[0] = out_cols  — output columns; work range for the two MUL_MAT regions
//   shape[1] = cols      — shared (reduction) dimension
//   shape[2] = rows      — batch dimension
//   shape[3] = 0         — unused
//
// Fields (per-invocation data pointers only — dimensions are structural,
// baked into the packet at compile time, and must NOT be re-patched per call):
//
//   w_gate   : [out_cols × cols]  gate weight matrix   (read-only)
//   w_up     : [out_cols × cols]  up weight matrix     (read-only)
//   x        : [rows × cols]      input activations    (read-only)
//   gate     : [rows × out_cols]  gate MUL_MAT output  (write; also SiLU input)
//   up       : [rows × out_cols]  up MUL_MAT output    (write; also MUL b-input)
//   gate_act : [rows × out_cols]  SiLU output          (write; also MUL a-input)
//   out      : [rows × out_cols]  final MUL output     (write)
//
// Hard preconditions (debug-asserted; undefined behaviour if violated):
//   - frame must have been frame_init'd against a packet with
//     key.sublayer == MLP_GATE_UP_F32.
//   - All seven pointers must be non-null and validly sized.
//   - gate, up, gate_act, out must not alias each other (the four steps
//     consume them in dependency order, but the caller owns the buffers).

typedef struct ggml_hpx_mlp_gate_up_binding
{
    const float *  w_gate;      // gate weight matrix     [out_cols × cols]
    const float *  w_up;        // up weight matrix       [out_cols × cols]
    const float *  x;           // input activations      [rows × cols]
    float *        gate;        // gate MUL_MAT output    [rows × out_cols]
    float *        up;          // up MUL_MAT output      [rows × out_cols]
    float *        gate_act;    // SiLU output            [rows × out_cols]
    float *        out;         // final output           [rows × out_cols]
} ggml_hpx_mlp_gate_up_binding;

void ggml_hpx_bind_mlp_gate_up_packet(
    struct ggml_hpx_packet_frame *           frame,
    const ggml_hpx_mlp_gate_up_binding *     binding);

// ---------------------------------------------------------------------------
// Typed binding — MLP_GLU_F32
// ---------------------------------------------------------------------------
//
// One binding struct for
//   key.sublayer == GGML_HPX_PACKET_SUBLAYER_MLP_GLU_F32.
//
// This is the fused-SWIGLU counterpart to MLP_GATE_UP_F32. Real llama MLPs
// emit GGML_OP_GLU with subop SWIGLU (via ggml_swiglu_split), collapsing the
// separate SiLU + elementwise MUL of the gate/up form into one region. The
// compiled packet therefore has 3 steps, not 4, and the binding has 6
// pointers, not 7 (there is no separate gate_act buffer — the SWIGLU kernel
// reads `gate` directly as its SiLU source).
//
// Shape convention (must match key.shape[] used at compile time), identical
// to MLP_GATE_UP_F32:
//   shape[0] = out_cols  — output columns; work range for the two MUL_MAT regions
//   shape[1] = cols      — shared (reduction) dimension
//   shape[2] = rows      — batch dimension
//   shape[3] = 0         — unused
//
// Fields (per-invocation data pointers only — dimensions are structural,
// baked into the packet at compile time, and must NOT be re-patched per call):
//
//   w_gate : [out_cols × cols]  gate weight matrix   (read-only)
//   w_up   : [out_cols × cols]  up weight matrix     (read-only)
//   x      : [rows × cols]      input activations    (read-only)
//   gate   : [rows × out_cols]  gate MUL_MAT output  (write; also SWIGLU gate input)
//   up     : [rows × out_cols]  up   MUL_MAT output  (write; also SWIGLU up input)
//   out    : [rows × out_cols]  fused SWIGLU output  (write; may alias gate or up
//                                                    per the SWIGLU kernel contract)
//
// Hard preconditions (debug-asserted; undefined behaviour if violated):
//   - frame must have been frame_init'd against a packet with
//     key.sublayer == MLP_GLU_F32.
//   - All six pointers must be non-null and validly sized.
//   - gate and up must not alias each other (the two MUL_MAT regions write
//     them independently and SWIGLU reads both). `out` may alias either per
//     the SWIGLU kernel's aliasing contract, but the caller owns that choice.

typedef struct ggml_hpx_mlp_glu_binding
{
    const float *  w_gate;    // gate weight matrix    [out_cols × cols]
    const float *  w_up;      // up weight matrix      [out_cols × cols]
    const float *  x;         // input activations     [rows × cols]
    float *        gate;      // gate MUL_MAT output   [rows × out_cols]
    float *        up;        // up   MUL_MAT output   [rows × out_cols]
    float *        out;       // fused SWIGLU output   [rows × out_cols]
} ggml_hpx_mlp_glu_binding;

void ggml_hpx_bind_mlp_glu_packet(
    struct ggml_hpx_packet_frame *      frame,
    const ggml_hpx_mlp_glu_binding *    binding);

#ifdef __cplusplus
}    // extern "C"
#endif

// ---------------------------------------------------------------------------
// Run (requires HPX runtime to be up)
// ---------------------------------------------------------------------------
//
// Execute the packet.
//
//   runtime   : live packet runtime; owns the pinned decode Exec used for
//               lane fan-out steps.
//   packet    : immutable; must match the frame it's called with.
//   frame     : previously frame_init'd against this packet, and
//               already bind-patched with the current invocation's values.
//   resources : caller-owned. Must satisfy
//                 resources->n_lanes == packet->key.n_lanes
//               and be sized per
//                 ggml_hpx_packet_get_resource_requirements(packet, ...)
//               for lane_scratch bytes, reduction_buffer bytes, and
//               shared_scratch bytes.
//
// Debug-asserted preconditions (enabled in debug builds; release builds
// skip these checks — callers must still satisfy them):
//
//   - runtime, packet, frame, resources all non-null
//   - frame was frame_init'd against this packet (the frame stores a
//     back-pointer to its packet at init time so this is cheap to check)
//   - resources->n_lanes == packet->key.n_lanes
//   - packet->key.n_lanes == n_decode_threads passed to runtime_create
//
// These preconditions are documented here as the contract a future
// ggml_hpx_packet_validate_resources(packet, resources) helper would
// formalize; the prototype does not ship that external validator.
//
// Execution model (I4):
//   Exec & exec = runtime->exec_for(packet->key.team);     // pinned per-team
//   for each step in packet->steps:
//     if step.kind == SERIAL:
//         step.run_range(ctx, /*ith=*/0, /*nth=*/1,
//                        step.work_begin, step.work_end, resources)
//     else:   // LANE_FANOUT
//         hpx::experimental::for_loop(
//             hpx::execution::par.on(exec),
//             0, step.n_lanes,
//             [&](int ith) {
//                 step.run_range(ctx, ith, step.n_lanes,
//                                step.work_begin, step.work_end, resources);
//             });
//
// No hpx::async, hpx::dataflow, shared_future, wait_all, or per-step heap.
// The for_loop is synchronous: control returns only after every lane has
// completed, providing the happens-before edge that any subsequent SERIAL
// REDUCTION step relies on.
//
// Thread safety: two concurrent calls must use two distinct frames. The
// packet, runtime, and resources pointers may be shared across concurrent
// calls provided the caller has arranged resource disjointness.

#ifdef __cplusplus

void ggml_hpx_run_frozen_packet(
    struct ggml_hpx_packet_runtime *       runtime,
    const struct ggml_hpx_frozen_packet *  packet,
    struct ggml_hpx_packet_frame *         frame,
    ggml_hpx_region_resources *            resources);

#endif    // __cplusplus

#endif    // GGML_HPX_REGION_DAG
