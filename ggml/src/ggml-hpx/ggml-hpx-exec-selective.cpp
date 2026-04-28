// ggml-hpx-exec-selective.cpp
//
// Selective graph executor for the mixed fine-region / CPU-fallback /
// frozen-packet path.
//
// Includes ggml-backend.h (third TU in this directory to do so, after
// ggml-hpx-adapter.cpp and ggml-hpx-exec.cpp).

#ifndef GGML_HPX_REGION_DAG
#  error "ggml-hpx-exec-selective.cpp requires -DGGML_HPX_REGION_DAG"
#endif

#include "ggml-hpx-exec-selective.h"
#include "ggml-hpx-compose.h"
#include "ggml-hpx-lower.h"
#include "ggml-hpx-packet.h"
#include "ggml-hpx-region-exec.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "ggml.h"

#include <hpx/future.hpp>
#include <hpx/runtime.hpp>

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <memory>
#include <new>
#include <unordered_map>
#include <vector>

#ifdef GGML_HPX_EXEC_SELECTIVE_TESTING
#include <atomic>
std::atomic<int> g_hpx_selective_lowered_count      {0};
std::atomic<int> g_hpx_selective_executed_count     {0};
std::atomic<int> g_hpx_mlp_packet_compile_count     {0};
std::atomic<int> g_hpx_mlp_packet_dispatch_count    {0};
std::atomic<int> g_hpx_mlp_glu_packet_compile_count {0};
std::atomic<int> g_hpx_mlp_glu_packet_dispatch_count{0};
#endif

// ---------------------------------------------------------------------------
// MLP gate/up packet cache — internal types
// ---------------------------------------------------------------------------

namespace {

// LLAMA_HPX_PACKET_COMPILE_LOG=1 emits one stderr line per packet cache miss
// with matcher tag, shape, and compile duration in ns. Read exactly once.
bool packet_compile_log_enabled() noexcept
{
    static const bool enabled = []() {
        const char * e = std::getenv("LLAMA_HPX_PACKET_COMPILE_LOG");
        return e && std::atoi(e) != 0;
    }();
    return enabled;
}

struct mlp_gate_up_cache_key
{
    int64_t out_cols;
    int64_t cols;
    int64_t rows;

    bool operator==(const mlp_gate_up_cache_key & o) const noexcept
    {
        return out_cols == o.out_cols && cols == o.cols && rows == o.rows;
    }
};

struct mlp_gate_up_cache_key_hash
{
    std::size_t operator()(const mlp_gate_up_cache_key & k) const noexcept
    {
        std::size_t h = std::hash<int64_t>{}(k.out_cols);
        h ^= std::hash<int64_t>{}(k.cols) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int64_t>{}(k.rows) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

// One cached sublayer: the compiled packet plus the caller-owned frame
// storage that was aligned-allocated at compile time and reused across
// every subsequent dispatch for the same structural key.
struct mlp_gate_up_packet_entry
{
    ggml_hpx_frozen_packet *  packet      = nullptr;
    ggml_hpx_packet_frame *   frame       = nullptr;
    void *                    frame_raw   = nullptr;
    std::size_t               frame_align = 0;

    mlp_gate_up_packet_entry() noexcept = default;

    ~mlp_gate_up_packet_entry() noexcept
    {
        if (frame_raw)
        {
            ::operator delete(frame_raw, std::align_val_t{frame_align});
        }
        if (packet)
        {
            ggml_hpx_free_packet(packet);
        }
    }

    mlp_gate_up_packet_entry(const mlp_gate_up_packet_entry &)             = delete;
    mlp_gate_up_packet_entry & operator=(const mlp_gate_up_packet_entry &) = delete;
};

} // namespace

// Full definition of the opaque cache type. unordered_map is held by
// unique_ptr value so rehash does not disturb pointers we hand back.
struct ggml_hpx_mlp_gate_up_packet_cache
{
    uint32_t n_lanes        = 0;
    uint32_t seq_regime     = 0;
    uint32_t policy_version = 0;

    std::unordered_map<mlp_gate_up_cache_key,
                       std::unique_ptr<mlp_gate_up_packet_entry>,
                       mlp_gate_up_cache_key_hash> entries;
};

ggml_hpx_mlp_gate_up_packet_cache * ggml_hpx_mlp_gate_up_packet_cache_create(
    uint32_t n_lanes,
    uint32_t seq_regime,
    uint32_t policy_version)
{
    auto * c = new ggml_hpx_mlp_gate_up_packet_cache;
    c->n_lanes        = n_lanes;
    c->seq_regime     = seq_regime;
    c->policy_version = policy_version;
    return c;
}

void ggml_hpx_mlp_gate_up_packet_cache_destroy(
    ggml_hpx_mlp_gate_up_packet_cache * cache)
{
    delete cache;
}

// ---------------------------------------------------------------------------
// Internal: lookup-or-compile (compose + compile_packet + frame allocate)
// ---------------------------------------------------------------------------

namespace {

// Returns a cache-owned entry pointer valid until the cache is destroyed.
// Returns nullptr if the ggml nodes cannot be composed (e.g. non-F32
// MUL_MAT inputs) or if packet compile rejects the plan key.
const mlp_gate_up_packet_entry * lookup_or_compile_mlp(
    ggml_hpx_mlp_gate_up_packet_cache * cache,
    const ggml_tensor *                 node_gate,
    const ggml_tensor *                 node_up,
    const ggml_tensor *                 node_gate_act,
    const ggml_tensor *                 node_out)
{
    // Structural key: MUL_MAT(W_gate, x) produces [out_cols, rows].
    //   gate->ne[0]        == out_cols
    //   gate->ne[1]        == rows
    //   gate->src[1]->ne[0] == cols (the x input's leading dimension)
    mlp_gate_up_cache_key key{};
    key.out_cols = node_gate->ne[0];
    key.rows     = node_gate->ne[1];
    key.cols     = node_gate->src[1]->ne[0];
    if (key.out_cols <= 0 || key.cols <= 0 || key.rows <= 0)
    {
        return nullptr;
    }

    auto it = cache->entries.find(key);
    if (it != cache->entries.end())
    {
        return it->second.get();
    }

    // Cache miss: compose → compile → frame allocate.
    ggml_hpx_mlp_gate_up_group grp;
    ggml_hpx_mlp_gate_up_group_init(&grp);
    if (!ggml_hpx_compose_mlp_gate_up_group(
            node_gate, node_up, node_gate_act, node_out, &grp))
    {
        return nullptr;
    }

    ggml_hpx_packet_plan_key pkey{};
    pkey.sublayer       = GGML_HPX_PACKET_SUBLAYER_MLP_GATE_UP_F32;
    pkey.team           = GGML_HPX_PACKET_TEAM_DECODE;
    pkey.n_lanes        = cache->n_lanes;
    pkey.dtype          = GGML_TYPE_F32;
    pkey.seq_regime     = cache->seq_regime;
    pkey.policy_version = cache->policy_version;
    pkey.shape[0]       = key.out_cols;
    pkey.shape[1]       = key.cols;
    pkey.shape[2]       = key.rows;
    pkey.shape[3]       = 0;
    pkey.extra          = 0;

    const char *             err    = nullptr;
    const auto               t0     = std::chrono::steady_clock::now();
    ggml_hpx_frozen_packet * packet = ggml_hpx_compile_packet(&grp.group, &pkey, &err);
    const auto               t1     = std::chrono::steady_clock::now();
    if (!packet)
    {
        return nullptr;
    }
    if (packet_compile_log_enabled())
    {
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        fprintf(stderr,
            "[hpx-packet-compile] matcher=gate_up out_cols=%lld cols=%lld rows=%lld compile_ns=%lld\n",
            (long long) key.out_cols, (long long) key.cols, (long long) key.rows, (long long) ns);
    }

    const std::size_t sz = ggml_hpx_packet_frame_size (packet);
    const std::size_t al = ggml_hpx_packet_frame_align(packet);
    void *            raw   = ::operator new(sz, std::align_val_t{al});
    auto *            frame = static_cast<ggml_hpx_packet_frame *>(raw);
    ggml_hpx_packet_frame_init(frame, packet);

    auto entry          = std::make_unique<mlp_gate_up_packet_entry>();
    entry->packet       = packet;
    entry->frame        = frame;
    entry->frame_raw    = raw;
    entry->frame_align  = al;

    mlp_gate_up_packet_entry * ret = entry.get();
    cache->entries.emplace(key, std::move(entry));

#ifdef GGML_HPX_EXEC_SELECTIVE_TESTING
    ++g_hpx_mlp_packet_compile_count;
#endif

    return ret;
}

// True iff the tensor is a GGML_OP_UNARY whose subop is SILU.
bool is_silu_op(const ggml_tensor * t) noexcept
{
    return t
        && t->op == GGML_OP_UNARY
        && ggml_get_unary_op(t) == GGML_UNARY_OP_SILU;
}

// Per-match record: the four ggml node indices (0 = gate_mm, 1 = up_mm,
// 2 = silu, 3 = final mul) plus the cache entry that will execute them.
struct mlp_match
{
    int                              nodes[4];
    const mlp_gate_up_packet_entry * entry;
};

// One pre-scan pass over gf->nodes.  For each GGML_OP_MUL that is the
// final node of the 4-op MLP gate/up pattern and whose shape is lookup-
// or-compile-able in the cache, record a match and mark all four node
// indices in `consumed`.
//
// Post-condition per index i:
//   !consumed[i]                           → standard lowered/fallback routing
//   consumed[i] && match_at_final[i] <  0  → non-trigger packet member
//   consumed[i] && match_at_final[i] >= 0  → final MUL (dispatch trigger)
void prescan_mlp_matches(
    ggml_cgraph *                         gf,
    ggml_hpx_mlp_gate_up_packet_cache *   cache,
    std::vector<bool> &                    consumed,
    std::vector<int>  &                    match_at_final,
    std::vector<mlp_match> &               matches)
{
    // Linear tensor→index lookup. n_nodes is small in the sublayers we
    // care about; a pointer→index hash map is premature optimization.
    auto find_index = [&](const ggml_tensor * t) -> int
    {
        for (int i = 0; i < gf->n_nodes; ++i)
        {
            if (gf->nodes[i] == t)
            {
                return i;
            }
        }
        return -1;
    };

    for (int i = 0; i < gf->n_nodes; ++i)
    {
        const ggml_tensor * mul_t = gf->nodes[i];
        if (mul_t->op != GGML_OP_MUL)          continue;
        if (!mul_t->src[0] || !mul_t->src[1])  continue;

        const ggml_tensor * silu_t = mul_t->src[0];
        const ggml_tensor * up_t   = mul_t->src[1];
        if (!is_silu_op(silu_t))               continue;
        if (up_t->op != GGML_OP_MUL_MAT)       continue;

        const ggml_tensor * gate_t = silu_t->src[0];
        if (!gate_t || gate_t->op != GGML_OP_MUL_MAT) continue;

        // gate and up must share the same x input.
        if (gate_t->src[1] != up_t->src[1])    continue;

        // All four must appear in the graph, topologically before the MUL.
        const int i_gate = find_index(gate_t);
        const int i_up   = find_index(up_t);
        const int i_silu = find_index(silu_t);
        if (i_gate < 0 || i_up < 0 || i_silu < 0)         continue;
        if (i_gate >= i || i_up >= i || i_silu >= i)      continue;

        // Don't steal a node already claimed by an earlier match.
        if (consumed[i_gate] || consumed[i_up]
         || consumed[i_silu] || consumed[i])              continue;

        const mlp_gate_up_packet_entry * entry =
            lookup_or_compile_mlp(cache, gate_t, up_t, silu_t, mul_t);
        if (!entry)                                       continue;

        mlp_match m{};
        m.nodes[0] = i_gate;
        m.nodes[1] = i_up;
        m.nodes[2] = i_silu;
        m.nodes[3] = i;
        m.entry    = entry;

        match_at_final[i] = static_cast<int>(matches.size());
        matches.push_back(m);
        consumed[i_gate] = true;
        consumed[i_up  ] = true;
        consumed[i_silu] = true;
        consumed[i     ] = true;
    }
}

} // namespace

// ---------------------------------------------------------------------------
// MLP GLU (SWIGLU) cache — internal types
// ---------------------------------------------------------------------------
//
// In an anonymous namespace so they are TU-private, exactly like the gate/up
// types above. ggml_hpx_mlp_glu_packet_cache (at file scope below) references
// them; that is fine because anonymous-namespace members are visible throughout
// the same TU.

namespace {

struct mlp_glu_cache_key
{
    int64_t  out_cols;
    int64_t  cols;
    int64_t  rows;
    uint32_t w_gate_type;    // ggml_type of gate weight; 0 = GGML_TYPE_F32
    uint32_t w_up_type;      // ggml_type of up weight;   0 = GGML_TYPE_F32

    bool operator==(const mlp_glu_cache_key & o) const noexcept
    {
        return out_cols    == o.out_cols
            && cols        == o.cols
            && rows        == o.rows
            && w_gate_type == o.w_gate_type
            && w_up_type   == o.w_up_type;
    }
};

struct mlp_glu_cache_key_hash
{
    std::size_t operator()(const mlp_glu_cache_key & k) const noexcept
    {
        std::size_t h = std::hash<int64_t>{}(k.out_cols);
        h ^= std::hash<int64_t>{}(k.cols)          + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int64_t>{}(k.rows)          + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<uint32_t>{}(k.w_gate_type)  + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<uint32_t>{}(k.w_up_type)    + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct mlp_glu_packet_entry
{
    ggml_hpx_frozen_packet *  packet      = nullptr;
    ggml_hpx_packet_frame *   frame       = nullptr;
    void *                    frame_raw   = nullptr;
    std::size_t               frame_align = 0;

    mlp_glu_packet_entry() noexcept = default;

    ~mlp_glu_packet_entry() noexcept
    {
        if (frame_raw)
        {
            ::operator delete(frame_raw, std::align_val_t{frame_align});
        }
        if (packet)
        {
            ggml_hpx_free_packet(packet);
        }
    }

    mlp_glu_packet_entry(const mlp_glu_packet_entry &)             = delete;
    mlp_glu_packet_entry & operator=(const mlp_glu_packet_entry &) = delete;
};

} // namespace

// ---------------------------------------------------------------------------
// MLP GLU packet cache — full definition and lifecycle
// ---------------------------------------------------------------------------
//
// Placed after the anonymous-namespace types above and before the lookup /
// prescan functions below. This ordering is required: lookup_or_compile_mlp_glu
// accesses cache->entries and needs the complete type.

struct ggml_hpx_mlp_glu_packet_cache
{
    uint32_t n_lanes        = 0;
    uint32_t seq_regime     = 0;
    uint32_t policy_version = 0;

    std::unordered_map<mlp_glu_cache_key,
                       std::unique_ptr<mlp_glu_packet_entry>,
                       mlp_glu_cache_key_hash> entries;
};

ggml_hpx_mlp_glu_packet_cache * ggml_hpx_mlp_glu_packet_cache_create(
    uint32_t n_lanes,
    uint32_t seq_regime,
    uint32_t policy_version)
{
    auto * c = new ggml_hpx_mlp_glu_packet_cache;
    c->n_lanes        = n_lanes;
    c->seq_regime     = seq_regime;
    c->policy_version = policy_version;
    return c;
}

void ggml_hpx_mlp_glu_packet_cache_destroy(ggml_hpx_mlp_glu_packet_cache * cache)
{
    delete cache;
}

// ---------------------------------------------------------------------------
// Internal: lookup-or-compile for MLP_GLU_F32, and pre-scan
// ---------------------------------------------------------------------------

namespace {

const mlp_glu_packet_entry * lookup_or_compile_mlp_glu(
    ggml_hpx_mlp_glu_packet_cache * cache,
    const ggml_tensor *              node_gate,
    const ggml_tensor *              node_up,
    const ggml_tensor *              node_glu)
{
    // Structural key from the gate MUL_MAT output tensor.
    //   gate->ne[0]         == out_cols
    //   gate->ne[1]         == rows
    //   gate->src[1]->ne[0] == cols (x input's leading dimension)
    mlp_glu_cache_key key{};
    key.out_cols    = node_gate->ne[0];
    key.rows        = node_gate->ne[1];
    key.cols        = node_gate->src[1]->ne[0];
    key.w_gate_type = static_cast<uint32_t>(node_gate->src[0]->type);
    key.w_up_type   = static_cast<uint32_t>(node_up  ->src[0]->type);
    if (key.out_cols <= 0 || key.cols <= 0 || key.rows <= 0)
    {
        return nullptr;
    }

    auto it = cache->entries.find(key);
    if (it != cache->entries.end())
    {
        return it->second.get();
    }

    // ---------------------------------------------------------------------------
    // Cache miss. Branch on weight dtype.
    //
    //  F32 weights: full 3-step MLP_GLU_F32 packet (gate_mm + up_mm + swiglu).
    //  Quantized:   1-step MLP_GLU_QBRIDGE packet (swiglu only; gate_mm + up_mm
    //               execute via CPU fallback before this packet fires).
    // ---------------------------------------------------------------------------

    const bool is_f32_weights = (key.w_gate_type == GGML_TYPE_F32)
                              && (key.w_up_type   == GGML_TYPE_F32);

    ggml_hpx_frozen_packet * packet = nullptr;
    const char * tag = nullptr;

    if (is_f32_weights)
    {
        ggml_hpx_mlp_glu_group grp;
        ggml_hpx_mlp_glu_group_init(&grp);
        if (!ggml_hpx_compose_mlp_glu_group(node_gate, node_up, node_glu, &grp))
        {
            return nullptr;
        }

        ggml_hpx_packet_plan_key pkey{};
        pkey.sublayer       = GGML_HPX_PACKET_SUBLAYER_MLP_GLU_F32;
        pkey.team           = GGML_HPX_PACKET_TEAM_DECODE;
        pkey.n_lanes        = cache->n_lanes;
        pkey.dtype          = GGML_TYPE_F32;
        pkey.seq_regime     = cache->seq_regime;
        pkey.policy_version = cache->policy_version;
        pkey.shape[0]       = key.out_cols;
        pkey.shape[1]       = key.cols;
        pkey.shape[2]       = key.rows;
        pkey.shape[3]       = 0;
        pkey.extra          = 0;

        const char * err = nullptr;
        packet = ggml_hpx_compile_packet(&grp.group, &pkey, &err);
        tag    = "glu_f32";
    }
    else
    {
        // QBRIDGE: lower the GLU node alone into a 1-region group.
        // gate->data and up->data are populated by the ggml scheduler before
        // graph_compute is called, so lower_op can read their pointers now.
        ggml_hpx_lowering lo;
        ggml_hpx_lowering_init(&lo);
        if (!ggml_hpx_lower_op(node_glu, &lo))
        {
            return nullptr;
        }

        // Pack weight types into pkey.extra (bits 0-7 = w_gate, bits 8-15 = w_up).
        const uint64_t extra =
              static_cast<uint64_t>(key.w_gate_type)
            | (static_cast<uint64_t>(key.w_up_type) << 8);

        ggml_hpx_packet_plan_key pkey{};
        pkey.sublayer       = GGML_HPX_PACKET_SUBLAYER_MLP_GLU_QBRIDGE;
        pkey.team           = GGML_HPX_PACKET_TEAM_DECODE;
        pkey.n_lanes        = cache->n_lanes;
        pkey.dtype          = GGML_TYPE_F32;
        pkey.seq_regime     = cache->seq_regime;
        pkey.policy_version = cache->policy_version;
        pkey.shape[0]       = key.out_cols;
        pkey.shape[1]       = key.cols;
        pkey.shape[2]       = key.rows;
        pkey.shape[3]       = 0;
        pkey.extra          = extra;

        const char * err = nullptr;
        packet = ggml_hpx_compile_packet(&lo.group, &pkey, &err);
        tag    = "glu_qbridge";
    }

    if (!packet)
    {
        return nullptr;
    }

    if (packet_compile_log_enabled())
    {
        const auto t1_now = std::chrono::steady_clock::now();
        (void)t1_now;
        fprintf(stderr,
            "[hpx-packet-compile] matcher=%s out_cols=%lld cols=%lld rows=%lld"
            " w_gate=%u w_up=%u\n",
            tag,
            (long long) key.out_cols, (long long) key.cols, (long long) key.rows,
            key.w_gate_type, key.w_up_type);
    }

    const std::size_t sz    = ggml_hpx_packet_frame_size (packet);
    const std::size_t al    = ggml_hpx_packet_frame_align(packet);
    void *            raw   = ::operator new(sz, std::align_val_t{al});
    auto *            frame = static_cast<ggml_hpx_packet_frame *>(raw);
    ggml_hpx_packet_frame_init(frame, packet);

    auto entry         = std::make_unique<mlp_glu_packet_entry>();
    entry->packet      = packet;
    entry->frame       = frame;
    entry->frame_raw   = raw;
    entry->frame_align = al;

    mlp_glu_packet_entry * ret = entry.get();
    cache->entries.emplace(key, std::move(entry));

#ifdef GGML_HPX_EXEC_SELECTIVE_TESTING
    ++g_hpx_mlp_glu_packet_compile_count;
#endif

    return ret;
}

// Per-match record: the three ggml node indices (0 = gate_mm, 1 = up_mm,
// 2 = glu trigger) plus the cache entry that will execute them.
struct mlp_glu_match
{
    int                          nodes[3];   // [0]=gate, [1]=up, [2]=glu (trigger)
    const mlp_glu_packet_entry * entry;
};

// One pre-scan pass over gf->nodes. For each GGML_OP_GLU[SWIGLU] node that
// is the final node of the 3-op GLU pattern and whose shape is lookup-or-
// compile-able in the cache, record a match and mark all three node indices
// in `consumed`.
//
// Ordering rule: trigger (GLU node) is always at index i; both gate_mm and
// up_mm must appear strictly before i in gf->nodes (i_gate < i, i_up < i).
// Only mark consumed[] after lookup_or_compile_mlp_glu succeeds — mirrors
// the gate/up pattern exactly.
void prescan_mlp_glu_matches(
    ggml_cgraph *                    gf,
    ggml_hpx_mlp_glu_packet_cache *  cache,
    std::vector<bool> &               consumed,
    std::vector<int>  &               glu_match_at_final,
    std::vector<mlp_glu_match> &      matches)
{
    auto find_index = [&](const ggml_tensor * t) -> int
    {
        for (int i = 0; i < gf->n_nodes; ++i)
        {
            if (gf->nodes[i] == t)
            {
                return i;
            }
        }
        return -1;
    };

    for (int i = 0; i < gf->n_nodes; ++i)
    {
        const ggml_tensor * glu_t = gf->nodes[i];
        if (glu_t->op != GGML_OP_GLU)                          continue;
        if (ggml_get_glu_op(glu_t) != GGML_GLU_OP_SWIGLU)     continue;
        if (!glu_t->src[0] || !glu_t->src[1])                  continue;

        const ggml_tensor * gate_t = glu_t->src[0];
        const ggml_tensor * up_t   = glu_t->src[1];
        if (gate_t->op != GGML_OP_MUL_MAT)                     continue;
        if (up_t->op   != GGML_OP_MUL_MAT)                     continue;

        // Gate and up MUL_MAT outputs must be F32 — the SWIGLU kernel reads
        // them directly, regardless of what weight dtype was used.
        if (gate_t->type != GGML_TYPE_F32)                      continue;
        if (up_t->type   != GGML_TYPE_F32)                      continue;

        // Gate and up must share the same x input (src[1] in ggml convention).
        if (gate_t->src[1] != up_t->src[1])                    continue;

        const int i_gate = find_index(gate_t);
        const int i_up   = find_index(up_t);

        // Both must appear in the graph and strictly before the trigger.
        if (i_gate < 0 || i_up < 0)                            continue;
        if (i_gate >= i || i_up >= i)                           continue;

        // Don't steal nodes already claimed by an earlier match.
        if (consumed[i_gate] || consumed[i_up] || consumed[i]) continue;

        // Only mark consumed after a successful lookup/compile.
        const mlp_glu_packet_entry * entry =
            lookup_or_compile_mlp_glu(cache, gate_t, up_t, glu_t);
        if (!entry)                                             continue;

        mlp_glu_match m{};
        m.nodes[0] = i_gate;
        m.nodes[1] = i_up;
        m.nodes[2] = i;
        m.entry    = entry;

        glu_match_at_final[i] = static_cast<int>(matches.size());
        matches.push_back(m);
        consumed[i_gate] = true;
        consumed[i_up  ] = true;
        consumed[i     ] = true;
    }
}

} // namespace

// ---------------------------------------------------------------------------
// Graph-entry policy guard
// ---------------------------------------------------------------------------
//
// Predicate is intentionally conservative: it considers only MUL_MAT
// ownership. A graph whose MUL_MATs all fall back is 100% selective tax with
// zero heavy-op win, regardless of how many small non-MUL_MAT ops would
// lower (norms, activations).
//
// Mirrors ggml_hpx_lower_op's MUL_MAT acceptance gates, but with field reads
// only (no contiguity / stride / scratch-budget checks). Any divergence is at
// worst a false positive — we engage selective on a graph that ends up
// all-fallback. It cannot be a false negative for the regression case the
// guard targets, because that case has zero lowerable MUL_MATs by
// construction (every Q4_K weight has w->extra != nullptr on M4).

bool ggml_hpx_selective_should_engage(const ggml_cgraph * gf)
{
    if (!gf || gf->n_nodes == 0)
    {
        return true;
    }

    uint32_t n_mul_mat   = 0;
    uint32_t n_lowerable = 0;

    for (int i = 0; i < gf->n_nodes; ++i)
    {
        const ggml_tensor * node = gf->nodes[i];
        if (!node || node->op != GGML_OP_MUL_MAT)
        {
            continue;
        }
        ++n_mul_mat;

        if (node->type != GGML_TYPE_F32)
        {
            continue;
        }
        const ggml_tensor * w = node->src[0];
        const ggml_tensor * x = node->src[1];
        if (!w || !x)
        {
            continue;
        }

        const bool f32xf32 =
            (w->type == GGML_TYPE_F32) && (x->type == GGML_TYPE_F32);

        // Q4_K is lowerable if it is either:
        //   - standard layout (extra == null)              → vec_dot path, or
        //   - CPU_REPACK with the q4_K_8x8_q8_K trait      → gemv path.
        // Other repacked traits (q4_K_8x4_q8_K, RISC-V 16x1) stay on the
        // CPU fallback because the lowered path has no kernel for them.
        // Must stay in lockstep with the Q4_K branch in
        // ggml_hpx_lower_op (ggml-hpx-lower.cpp).
        const bool q4k_decode_safe =
            (w->type == GGML_TYPE_Q4_K)
            && (node->ne[1] == 1)
            && (w->extra == nullptr || ggml_hpx_is_q4k_8x8_repacked(node));

        if (f32xf32 || q4k_decode_safe)
        {
            ++n_lowerable;
        }
    }

    return (n_mul_mat == 0) || (n_lowerable > 0);
}

// ---------------------------------------------------------------------------
// Node-histogram diagnostic
// ---------------------------------------------------------------------------
//
// Walks the graph once, buckets every node, emits a single block to stderr.
// MUL_MAT is sub-classified; everything else is tallied per ggml_op into a
// fixed-size array indexed by the op id. No heap allocation.

void ggml_hpx_selective_log_node_histogram(const ggml_cgraph * gf)
{
    static const bool enabled = [] {
        const char * e = getenv("LLAMA_HPX_SELECTIVE_HIST");
        return e && atoi(e) != 0;
    }();
    if (!enabled)
    {
        return;
    }

    static bool warned = false;
    if (warned)
    {
        return;
    }
    warned = true;

    if (!gf || gf->n_nodes == 0)
    {
        fprintf(stderr,
            "[hpx-selective-hist] empty graph — nothing to bucket\n");
        return;
    }

    uint32_t mm_q4k_repacked    = 0;
    uint32_t mm_q4k_nonrepacked = 0;
    uint32_t mm_f32xf32         = 0;
    uint32_t mm_other_quant     = 0;
    uint32_t mm_other           = 0;

    uint32_t op_counts[GGML_OP_COUNT] = {};

    // Shape-weighted view: per-bucket dedup'd list of (cols, out_cols, rows)
    // tuples. Convention matches ggml-hpx-lower.cpp's MUL_MAT case:
    //   cols = w->ne[0]   (shared / reduction dim)
    //   out_cols = w->ne[1]   (output columns)
    //   rows = x->ne[1]   (batch)
    // 16 entries is enough for any single transformer graph (TinyLlama uses 4).
    struct mm_shape { int64_t cols; int64_t out_cols; int64_t rows; uint32_t count; };
    constexpr int kMaxShapes = 16;
    mm_shape shapes_q4k_repacked   [kMaxShapes] = {};
    mm_shape shapes_q4k_nonrepacked[kMaxShapes] = {};
    mm_shape shapes_f32xf32        [kMaxShapes] = {};
    mm_shape shapes_other_quant    [kMaxShapes] = {};
    int n_shapes_q4k_repacked    = 0;
    int n_shapes_q4k_nonrepacked = 0;
    int n_shapes_f32xf32         = 0;
    int n_shapes_other_quant     = 0;

    auto record_shape = [](mm_shape * arr, int & n,
                           int64_t cols, int64_t out_cols, int64_t rows) {
        for (int j = 0; j < n; ++j)
        {
            if (arr[j].cols == cols && arr[j].out_cols == out_cols
                && arr[j].rows == rows)
            {
                ++arr[j].count;
                return;
            }
        }
        if (n < kMaxShapes)
        {
            arr[n] = mm_shape{cols, out_cols, rows, 1};
            ++n;
        }
    };

    for (int i = 0; i < gf->n_nodes; ++i)
    {
        const ggml_tensor * node = gf->nodes[i];
        if (!node)
        {
            continue;
        }
        if (node->op != GGML_OP_MUL_MAT)
        {
            if (node->op >= 0 && node->op < GGML_OP_COUNT)
            {
                ++op_counts[node->op];
            }
            continue;
        }

        const ggml_tensor * w = node->src[0];
        const ggml_tensor * x = node->src[1];
        if (!w || !x)
        {
            ++mm_other;
            continue;
        }

        const int64_t cols     = w->ne[0];
        const int64_t out_cols = w->ne[1];
        const int64_t rows     = x->ne[1];

        if (w->type == GGML_TYPE_Q4_K)
        {
            if (w->extra != nullptr)
            {
                ++mm_q4k_repacked;
                record_shape(shapes_q4k_repacked, n_shapes_q4k_repacked,
                             cols, out_cols, rows);
            }
            else
            {
                ++mm_q4k_nonrepacked;
                record_shape(shapes_q4k_nonrepacked, n_shapes_q4k_nonrepacked,
                             cols, out_cols, rows);
            }
        }
        else if (w->type == GGML_TYPE_F32 && x->type == GGML_TYPE_F32)
        {
            ++mm_f32xf32;
            record_shape(shapes_f32xf32, n_shapes_f32xf32,
                         cols, out_cols, rows);
        }
        else if (ggml_is_quantized(w->type))
        {
            ++mm_other_quant;
            record_shape(shapes_other_quant, n_shapes_other_quant,
                         cols, out_cols, rows);
        }
        else
        {
            ++mm_other;
        }
    }

    fprintf(stderr, "[hpx-selective-hist] n_nodes=%d\n", gf->n_nodes);

    auto print_bucket = [](const char * label, uint32_t count,
                           const mm_shape * arr, int n) {
        fprintf(stderr, "[hpx-selective-hist]   %-24s = %u",
                label, count);
        if (n > 0)
        {
            fprintf(stderr, "  shapes(cols x out_cols x rows = count):");
            for (int i = 0; i < n; ++i)
            {
                fprintf(stderr, " %lldx%lldx%lld=%u",
                        (long long) arr[i].cols,
                        (long long) arr[i].out_cols,
                        (long long) arr[i].rows,
                        arr[i].count);
            }
        }
        fprintf(stderr, "\n");
    };

    print_bucket("MUL_MAT/Q4_K-repacked",    mm_q4k_repacked,
                 shapes_q4k_repacked, n_shapes_q4k_repacked);
    print_bucket("MUL_MAT/Q4_K-nonrepacked", mm_q4k_nonrepacked,
                 shapes_q4k_nonrepacked, n_shapes_q4k_nonrepacked);
    print_bucket("MUL_MAT/F32xF32",          mm_f32xf32,
                 shapes_f32xf32, n_shapes_f32xf32);
    print_bucket("MUL_MAT/other-quant",      mm_other_quant,
                 shapes_other_quant, n_shapes_other_quant);
    print_bucket("MUL_MAT/other",            mm_other, nullptr, 0);

    for (int op = 0; op < GGML_OP_COUNT; ++op)
    {
        if (op_counts[op] == 0)
        {
            continue;
        }
        fprintf(stderr,
            "[hpx-selective-hist]   %-24s = %u\n",
            ggml_op_name(static_cast<ggml_op>(op)),
            op_counts[op]);
    }
}

// ---------------------------------------------------------------------------
// Main selective entry point
// ---------------------------------------------------------------------------

bool ggml_hpx_exec_graph_selective_mul_mat(
    ggml_cgraph *                             gf,
    ggml_backend_t                            cpu_be,
    int                                       n_lanes,
    ggml_hpx_selective_stats *                stats_out,
    const ggml_hpx_selective_packet_env *     packet_env)
{
    if (!gf || gf->n_nodes == 0)
    {
        return true;
    }

    const bool gate_up_enabled = packet_env
        && packet_env->rt
        && packet_env->mlp_cache;
    const bool glu_enabled = packet_env
        && packet_env->rt
        && packet_env->mlp_glu_cache;

    const int lanes = (n_lanes > 0)
        ? n_lanes
        : static_cast<int>(hpx::get_num_worker_threads());

    ggml_hpx_region_resources res{};
    res.n_lanes = lanes;

    ggml_hpx_selective_stats stats{};
    bool ok = true;

    using clock    = std::chrono::steady_clock;
    using nanosecs = std::chrono::nanoseconds;

    // Pre-scan. Both scans share consumed[] so a node claimed by gate/up
    // cannot be re-claimed by GLU and vice versa.
    // Gate/up runs first (preserves the older path), GLU second.
    // When both paths are disabled, all vectors stay at their default values
    // and the main loop behaves exactly as it did before the packet path.
    std::vector<bool>          consumed          (gf->n_nodes, false);
    std::vector<int>           match_at_final    (gf->n_nodes, -1);
    std::vector<int>           glu_match_at_final(gf->n_nodes, -1);
    std::vector<mlp_match>     matches;
    std::vector<mlp_glu_match> glu_matches;

    if (gate_up_enabled)
    {
        prescan_mlp_matches(gf, packet_env->mlp_cache,
                            consumed, match_at_final, matches);
    }
    if (glu_enabled)
    {
        prescan_mlp_glu_matches(gf, packet_env->mlp_glu_cache,
                                consumed, glu_match_at_final, glu_matches);
    }

    static const bool dbg = [] {
        const char * e = getenv("LLAMA_HPX_SELECTIVE_DEBUG");
        return e && atoi(e) != 0;
    }();

    // Forward-scan a contiguous run of fallback-eligible nodes starting at
    // `start`, returning the first index that is NOT fallback-eligible (or
    // gf->n_nodes). A node k is fallback-eligible iff:
    //   - !consumed[k]                                     (not packet-owned)
    //   - AND ( !ggml_hpx_lower_op(node[k], &lo_peek)      (lower_op rejected)
    //           OR  any region in lo_peek.group is a REDUCTION with
    //               uses_resources == 1 )                  (resource-rejected)
    //
    // Used to coalesce a run of fallback nodes into one
    // ggml_backend_graph_compute(view) dispatch instead of one per node.
    // Stops at the first lowerable-and-safe node (run boundary) and at the
    // first packet-consumed node (graph order must be preserved across packet
    // dispatches).
    auto scan_fallback_run_end = [&](int start) -> int {
        ggml_hpx_lowering lo_peek;
        int j = start;
        for (; j < gf->n_nodes; ++j)
        {
            if (consumed[j]) break;

            ggml_hpx_lowering_init(&lo_peek);
            if (!ggml_hpx_lower_op(gf->nodes[j], &lo_peek))
            {
                continue;  // lower_op rejected → fallback-eligible
            }

            bool has_resource_reduction = false;
            for (int r = 0; r < lo_peek.group.n_regions; ++r)
            {
                if (lo_peek.group.regions[r].kind == GGML_HPX_CPU_REGION_KIND_REDUCTION
                 && lo_peek.group.regions[r].uses_resources)
                {
                    has_resource_reduction = true;
                    break;
                }
            }
            if (has_resource_reduction)
            {
                continue;  // resource-rejected → fallback-eligible
            }

            break;  // would lower safely → end of fallback run
        }
        return j;
    };

    for (int i = 0; i < gf->n_nodes && ok; ++i)
    {
        // Packet-consumed node: dispatch if this is the trigger of any match,
        // otherwise skip silently (non-trigger member owned by its match).
        // A node cannot simultaneously be the trigger for both a gate/up match
        // and a GLU match — that is a pre-scan bug.
        if (consumed[i])
        {
            assert(!((match_at_final[i] >= 0) && (glu_match_at_final[i] >= 0))
                && "node is trigger for both gate/up and GLU match — pre-scan overlap bug");

            if (match_at_final[i] >= 0)
            {
                // Gate/up packet dispatch. Trigger is the final MUL node.
                const mlp_match &   m      = matches[match_at_final[i]];
                const ggml_tensor * gate_t = gf->nodes[m.nodes[0]];
                const ggml_tensor * up_t   = gf->nodes[m.nodes[1]];
                const ggml_tensor * silu_t = gf->nodes[m.nodes[2]];
                ggml_tensor *       mul_t  = gf->nodes[m.nodes[3]];

                ggml_hpx_mlp_gate_up_binding b{};
                b.w_gate   = static_cast<const float *>(gate_t->src[0]->data);
                b.w_up     = static_cast<const float *>(up_t  ->src[0]->data);
                b.x        = static_cast<const float *>(gate_t->src[1]->data);
                b.gate     = static_cast<float *>(gate_t->data);
                b.up       = static_cast<float *>(up_t  ->data);
                b.gate_act = static_cast<float *>(silu_t->data);
                b.out      = static_cast<float *>(mul_t ->data);
                ggml_hpx_bind_mlp_gate_up_packet(m.entry->frame, &b);

                // MLP gate/up has no REDUCTION regions; lane_scratch /
                // reduction_buffer / shared_scratch byte counts are all 0.
                // run_frozen_packet still reads resources->n_lanes, so a
                // valid (though unused) lane pointer array is supplied.
                ggml_hpx_packet_resource_requirements req{};
                ggml_hpx_packet_get_resource_requirements(m.entry->packet, &req);
                std::vector<void *> lane_ptrs(req.n_lanes, nullptr);
                ggml_hpx_region_resources pkt_res{};
                pkt_res.n_lanes          = static_cast<int>(req.n_lanes);
                pkt_res.lane_scratch     = lane_ptrs.empty() ? nullptr : lane_ptrs.data();
                pkt_res.reduction_buffer = nullptr;
                pkt_res.shared_scratch   = nullptr;

                const auto t0 = clock::now();
                hpx::async([&]() {
                    ggml_hpx_run_frozen_packet(
                        packet_env->rt, m.entry->packet, m.entry->frame, &pkt_res);
                }).get();
                const auto t1 = clock::now();

                stats.packet_matches     += 1;
                stats.packet_nodes       += 4;
                stats.packet_dispatch_ns += static_cast<uint64_t>(
                    std::chrono::duration_cast<nanosecs>(t1 - t0).count());

#ifdef GGML_HPX_EXEC_SELECTIVE_TESTING
                ++g_hpx_mlp_packet_dispatch_count;
#endif
            }
            else if (glu_match_at_final[i] >= 0)
            {
                // GLU packet dispatch. Trigger is the GGML_OP_GLU node.
                const mlp_glu_match & m      = glu_matches[glu_match_at_final[i]];
                const ggml_tensor *   gate_t = gf->nodes[m.nodes[0]];
                const ggml_tensor *   up_t   = gf->nodes[m.nodes[1]];
                ggml_tensor *         glu_t  = gf->nodes[m.nodes[2]];

                const uint32_t sublayer =
                    ggml_hpx_packet_key(m.entry->packet)->sublayer;

                if (sublayer == GGML_HPX_PACKET_SUBLAYER_MLP_GLU_QBRIDGE)
                {
                    // QBRIDGE: gate_mm and up_mm have quantized weights and
                    // must run via the CPU backend before the SWIGLU packet.
                    // Time each CPU call separately so bridge_fallback_ns
                    // captures the true quantized MUL_MAT cost.
                    {
                        ggml_cgraph view = ggml_graph_view(gf, m.nodes[0], m.nodes[0] + 1);
                        const auto tb0 = clock::now();
                        if (ggml_backend_graph_compute(cpu_be, &view) != GGML_STATUS_SUCCESS)
                        {
                            ok = false;
                        }
                        const auto tb1 = clock::now();
                        stats.bridge_fallback_nodes += 1;
                        stats.bridge_fallback_ns    += static_cast<uint64_t>(
                            std::chrono::duration_cast<nanosecs>(tb1 - tb0).count());
                    }
                    if (ok)
                    {
                        ggml_cgraph view = ggml_graph_view(gf, m.nodes[1], m.nodes[1] + 1);
                        const auto tb0 = clock::now();
                        if (ggml_backend_graph_compute(cpu_be, &view) != GGML_STATUS_SUCCESS)
                        {
                            ok = false;
                        }
                        const auto tb1 = clock::now();
                        stats.bridge_fallback_nodes += 1;
                        stats.bridge_fallback_ns    += static_cast<uint64_t>(
                            std::chrono::duration_cast<nanosecs>(tb1 - tb0).count());
                    }
                    if (!ok) continue;

                    ggml_hpx_mlp_glu_qbridge_binding b{};
                    b.gate = static_cast<float *>(gate_t->data);
                    b.up   = static_cast<float *>(up_t  ->data);
                    b.out  = static_cast<float *>(glu_t ->data);
                    ggml_hpx_bind_mlp_glu_qbridge_packet(m.entry->frame, &b);
                }
                else
                {
                    // MLP_GLU_F32: full 3-step packet; gate_mm + up_mm execute
                    // inside the packet, no separate fallback needed.
                    ggml_hpx_mlp_glu_binding b{};
                    b.w_gate = static_cast<const float *>(gate_t->src[0]->data);
                    b.w_up   = static_cast<const float *>(up_t  ->src[0]->data);
                    b.x      = static_cast<const float *>(gate_t->src[1]->data);
                    b.gate   = static_cast<float *>(gate_t->data);
                    b.up     = static_cast<float *>(up_t  ->data);
                    b.out    = static_cast<float *>(glu_t ->data);
                    ggml_hpx_bind_mlp_glu_packet(m.entry->frame, &b);
                }

                // No REDUCTION regions in either variant; null lane-scratch.
                ggml_hpx_packet_resource_requirements req{};
                ggml_hpx_packet_get_resource_requirements(m.entry->packet, &req);
                std::vector<void *> lane_ptrs(req.n_lanes, nullptr);
                ggml_hpx_region_resources pkt_res{};
                pkt_res.n_lanes          = static_cast<int>(req.n_lanes);
                pkt_res.lane_scratch     = lane_ptrs.empty() ? nullptr : lane_ptrs.data();
                pkt_res.reduction_buffer = nullptr;
                pkt_res.shared_scratch   = nullptr;

                const auto t0 = clock::now();
                hpx::async([&]() {
                    ggml_hpx_run_frozen_packet(
                        packet_env->rt, m.entry->packet, m.entry->frame, &pkt_res);
                }).get();
                const auto t1 = clock::now();

                // QBRIDGE: only the GLU trigger node runs inside the packet.
                // gate_mm + up_mm are already counted in bridge_fallback_nodes.
                // MLP_GLU_F32: all 3 nodes run inside the packet.
                const bool is_qbridge =
                    (sublayer == GGML_HPX_PACKET_SUBLAYER_MLP_GLU_QBRIDGE);
                stats.packet_matches     += 1;
                stats.packet_nodes       += is_qbridge ? 1 : 3;
                stats.packet_dispatch_ns += static_cast<uint64_t>(
                    std::chrono::duration_cast<nanosecs>(t1 - t0).count());

#ifdef GGML_HPX_EXEC_SELECTIVE_TESTING
                ++g_hpx_mlp_glu_packet_dispatch_count;
#endif
            }
            // else: non-trigger packet member; owned by its match, nothing to do.
            continue;
        }

        ggml_hpx_lowering lo;
        ggml_hpx_lowering_init(&lo);

        if (ggml_hpx_lower_op(gf->nodes[i], &lo))
        {
            // A REDUCTION region is safe to run via HPX only when its kernel
            // is self-sufficient (uses_resources == 0), i.e. it writes to its
            // own ctx scratch rather than resources->lane_scratch /
            // resources->reduction_buffer.  Example: Q4_K quantize-to-Q8_K
            // writes to the lowering struct's scratch arena — no external
            // resources needed.  RMS_NORM finalize sets uses_resources = 1
            // because it reads lane_scratch[0..n_lanes) and writes
            // reduction_buffer; those are null in res{} and cannot run safely.
            bool has_resource_reduction = false;
            for (int r = 0; r < lo.group.n_regions; ++r)
            {
                if (lo.group.regions[r].kind == GGML_HPX_CPU_REGION_KIND_REDUCTION
                 && lo.group.regions[r].uses_resources)
                {
                    has_resource_reduction = true;
                }
            }
            if (dbg)
            {
                fprintf(stderr,
                    "[hpx-selective-lower] op=%s n_regions=%d has_resource_reduction=%d\n",
                    ggml_op_name(gf->nodes[i]->op),
                    lo.group.n_regions,
                    (int)has_resource_reduction);
            }

            // Reject groups with resource-dependent REDUCTION regions.
            if (has_resource_reduction)
            {
                const int j = scan_fallback_run_end(i + 1);
                ggml_cgraph view = ggml_graph_view(gf, i, j);
                const auto t0 = clock::now();
                if (ggml_backend_graph_compute(cpu_be, &view) != GGML_STATUS_SUCCESS)
                {
                    ok = false;
                }
                const auto t1 = clock::now();
                stats.fallback_nodes += static_cast<uint32_t>(j - i);
                stats.fallback_runs  += 1;
                stats.fallback_ns    += static_cast<uint64_t>(
                    std::chrono::duration_cast<nanosecs>(t1 - t0).count());
                i = j - 1;  // for-loop's ++i lands on j
                continue;
            }

#ifdef GGML_HPX_EXEC_SELECTIVE_TESTING
            ++g_hpx_selective_lowered_count;
#endif
            // lo is on the stack of this loop iteration; .get() blocks until
            // the async body completes, so lo remains valid throughout.
            const auto t0 = clock::now();
            hpx::async([&lo, &res]() {
#ifdef GGML_HPX_EXEC_SELECTIVE_TESTING
                ++g_hpx_selective_executed_count;
#endif
                ggml_hpx_run_region_group(&lo.group, &res);
            }).get();
            const auto t1 = clock::now();
            ++stats.lowered_nodes;
            stats.lowered_ns += static_cast<uint64_t>(
                std::chrono::duration_cast<nanosecs>(t1 - t0).count());
        }
        else
        {
            const int j = scan_fallback_run_end(i + 1);
            ggml_cgraph view = ggml_graph_view(gf, i, j);
            const auto t0 = clock::now();
            if (ggml_backend_graph_compute(cpu_be, &view) != GGML_STATUS_SUCCESS)
            {
                ok = false;
            }
            const auto t1 = clock::now();
            stats.fallback_nodes += static_cast<uint32_t>(j - i);
            stats.fallback_runs  += 1;
            stats.fallback_ns    += static_cast<uint64_t>(
                std::chrono::duration_cast<nanosecs>(t1 - t0).count());
            i = j - 1;  // for-loop's ++i lands on j
        }
    }

    if (stats_out)
    {
        *stats_out = stats;
    }
    return ok;
}

bool ggml_hpx_exec_graph_selective_mul_mat_arena(
    ggml_cgraph *              gf,
    int                        n_lanes,
    ggml_hpx_selective_stats * stats_out)
{
    ggml_backend_t cpu_be = ggml_backend_cpu_init();
    if (!cpu_be)
    {
        return false;
    }
    const bool ok = ggml_hpx_exec_graph_selective_mul_mat(gf, cpu_be, n_lanes, stats_out);
    ggml_backend_free(cpu_be);
    return ok;
}
