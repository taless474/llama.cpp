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
#include <cstdlib>
#include <functional>
#include <memory>
#include <new>
#include <unordered_map>
#include <vector>

#ifdef GGML_HPX_EXEC_SELECTIVE_TESTING
#include <atomic>
std::atomic<int> g_hpx_selective_lowered_count  {0};
std::atomic<int> g_hpx_selective_executed_count {0};
std::atomic<int> g_hpx_mlp_packet_compile_count {0};
std::atomic<int> g_hpx_mlp_packet_dispatch_count{0};
#endif

// ---------------------------------------------------------------------------
// MLP gate/up packet cache — internal types
// ---------------------------------------------------------------------------

namespace {

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
    ggml_hpx_frozen_packet * packet = ggml_hpx_compile_packet(&grp.group, &pkey, &err);
    if (!packet)
    {
        return nullptr;
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
// Main selective entry point
// ---------------------------------------------------------------------------

bool ggml_hpx_exec_graph_selective_mul_mat(
    ggml_cgraph *                         gf,
    ggml_backend_t                        cpu_be,
    int                                   n_lanes,
    ggml_hpx_selective_stats *            stats_out,
    ggml_hpx_packet_runtime *             packet_rt,
    ggml_hpx_mlp_gate_up_packet_cache *   mlp_cache)
{
    if (!gf || gf->n_nodes == 0)
    {
        return true;
    }

    // Packet args must come in together; mismatched input is debug-asserted
    // and treated as "packet dispatch disabled" in release.
    assert(((packet_rt == nullptr) == (mlp_cache == nullptr))
        && "packet_rt and mlp_cache must both be null or both non-null");
    const bool packet_enabled = (packet_rt != nullptr) && (mlp_cache != nullptr);

    const int lanes = (n_lanes > 0)
        ? n_lanes
        : static_cast<int>(hpx::get_num_worker_threads());

    ggml_hpx_region_resources res{};
    res.n_lanes = lanes;

    ggml_hpx_selective_stats stats{};
    bool ok = true;

    using clock    = std::chrono::steady_clock;
    using nanosecs = std::chrono::nanoseconds;

    // Pre-scan. When packet dispatch is disabled, consumed[] stays all-false
    // and match_at_final[] stays all -1; the main loop below behaves exactly
    // as it did before the packet path existed.
    std::vector<bool>      consumed      (gf->n_nodes, false);
    std::vector<int>       match_at_final(gf->n_nodes, -1);
    std::vector<mlp_match> matches;
    if (packet_enabled)
    {
        prescan_mlp_matches(gf, mlp_cache, consumed, match_at_final, matches);
    }

    static const bool dbg = [] {
        const char * e = getenv("LLAMA_HPX_SELECTIVE_DEBUG");
        return e && atoi(e) != 0;
    }();

    for (int i = 0; i < gf->n_nodes && ok; ++i)
    {
        // Packet-consumed node: dispatch if this is the final MUL of the
        // match, otherwise skip silently. This is the only place the
        // "final MUL owns dispatch" invariant is enforced.
        if (consumed[i])
        {
            if (match_at_final[i] >= 0)
            {
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
                        packet_rt, m.entry->packet, m.entry->frame, &pkt_res);
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
            // Non-trigger packet member: owned by its match; nothing to do.
            continue;
        }

        ggml_hpx_lowering lo;
        ggml_hpx_lowering_init(&lo);

        if (ggml_hpx_lower_op(gf->nodes[i], &lo))
        {
            bool has_reduction = false;
            for (int r = 0; r < lo.group.n_regions; ++r)
            {
                if (lo.group.regions[r].kind == GGML_HPX_CPU_REGION_KIND_REDUCTION)
                {
                    has_reduction = true;
                }
            }
            if (dbg)
            {
                fprintf(stderr,
                    "[hpx-selective-lower] op=%s n_regions=%d has_reduction=%d\n",
                    ggml_op_name(gf->nodes[i]->op),
                    lo.group.n_regions,
                    (int)has_reduction);
            }

            // Reject groups that contain REDUCTION regions: those require
            // lane_scratch and reduction_buffer which are null in res{}.
            if (has_reduction)
            {
                ggml_cgraph view = ggml_graph_view(gf, i, i + 1);
                const auto t0 = clock::now();
                if (ggml_backend_graph_compute(cpu_be, &view) != GGML_STATUS_SUCCESS)
                {
                    ok = false;
                }
                const auto t1 = clock::now();
                ++stats.fallback_nodes;
                stats.fallback_ns += static_cast<uint64_t>(
                    std::chrono::duration_cast<nanosecs>(t1 - t0).count());
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
            const auto t0 = clock::now();
            ggml_cgraph view = ggml_graph_view(gf, i, i + 1);
            if (ggml_backend_graph_compute(cpu_be, &view) != GGML_STATUS_SUCCESS)
            {
                ok = false;
            }
            const auto t1 = clock::now();
            ++stats.fallback_nodes;
            stats.fallback_ns += static_cast<uint64_t>(
                std::chrono::duration_cast<nanosecs>(t1 - t0).count());
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
