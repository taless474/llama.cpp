# HPX op lowering contract

## Purpose

`ggml_hpx_lower_op` translates one `ggml_tensor` compute node into a
`ggml_hpx_cpu_region_group` that the frozen-packet compiler or the fine-region
DAG runner can consume directly.

Single-region ops (SiLU, elementwise MUL) produce `group.n_regions == 1`.
Multi-stage ops (RMS_NORM) produce `group.n_regions > 1` with explicit dep edges.
The packet compiler always receives a group; 1-region groups are just the
degenerate case.

## Output struct — `ggml_hpx_lowering`

```c
#define GGML_HPX_LOWERING_MAX_REGIONS            4
#define GGML_HPX_LOWERING_MAX_DEPS               6
#define GGML_HPX_LOWERING_CTX_BYTES_PER_REGION 128

typedef struct ggml_hpx_lowering {
    ggml_hpx_cpu_region_group  group;
    ggml_hpx_cpu_region        regions[GGML_HPX_LOWERING_MAX_REGIONS];
    ggml_hpx_dep_edge          deps   [GGML_HPX_LOWERING_MAX_DEPS];
    alignas(max_align_t) unsigned char
        ctx_buf[GGML_HPX_LOWERING_MAX_REGIONS][GGML_HPX_LOWERING_CTX_BYTES_PER_REGION];
} ggml_hpx_lowering;
```

`lower_op` placement-news each op's ctx struct into `ctx_buf[i]` and sets
`regions[i].ctx = ctx_buf[i]`.  `group.regions` and `group.deps` point at the
inline arrays.

The 128-byte-per-region budget accounts for the largest ctx struct across all
supported ops (currently `ggml_hpx_mul_mat_f32_ctx` at ~48 bytes).  A
compile-time `static_assert` in `ggml-hpx-lower.cpp` enforces that every
emitted ctx fits within this budget.

## Init helper

Always call `ggml_hpx_lowering_init` before passing `out` to `lower_op`:

```c
// Zero the struct and wire group.regions → regions[], group.deps → deps[].
// Required before first use and before reuse.
static inline void ggml_hpx_lowering_init(ggml_hpx_lowering * l);
```

This puts the struct in a known state and ensures the internal pointer wiring
is correct even if the struct was declared uninitialised or previously used.

## Function signature

```c
bool ggml_hpx_lower_op(const ggml_tensor * node, ggml_hpx_lowering * out);
```

Returns `true` on success, `false` for unsupported ops or invalid
tensor/layout combinations (wrong type, non-contiguous strides, batch dim > 1
for ops that only support single-row lowering in this prototype).

`out` is written atomically: on failure, `out->group.n_regions` is set to 0
and no ctx constructors run.

## Supported ops (first implementation)

| ggml op                                  | `n_regions` | dep edges | work range     |
|------------------------------------------|-------------|-----------|----------------|
| `GGML_OP_UNARY` / `GGML_UNARY_OP_SILU`  | 1           | 0         | `[0, ne[0])`   |
| `GGML_OP_MUL`                            | 1           | 0         | `[0, ne[0])`   |
| `GGML_OP_MUL_MAT`                        | 1           | 0         | `[0, out_cols)`|
| `GGML_OP_RMS_NORM`                       | 3           | 0→1, 1→2  | `[0, ne[0])`   |

**RMS_NORM prototype limitation**: only single-row tensors (`ne[1] == 1`) are
supported in this first implementation.  Multi-row support (one lowering per
row, or a batched region) is deferred.

**SiLU dispatch**: `GGML_OP_UNARY` covers many unary ops; `lower_op` checks
`ggml_get_unary_op(node) == GGML_UNARY_OP_SILU` and rejects any other unary op.

## Ctx lifetime and packet-compile contract

`ggml_hpx_lowering` must remain valid (not destroyed, not moved) until after
`ggml_hpx_compile_packet` returns.  Packet compile is required to deep-copy any
ctx data it retains into its own allocation.  After compile returns, the caller
owns the lowering struct and may destroy it freely.

Rationale: `ggml_hpx_lowering` is typically stack-allocated at the call site.
Making packet compile own a copy keeps the caller's lifetime reasoning trivial.

## Relationship to packet compiler

Packet compile (`ggml_hpx_compile_packet`) takes a `const ggml_hpx_cpu_region_group *`
— pass `&out->group`.  The compiler validates structure and callback identity
before using any ctx data.

## Adding a new op

1. Add ctx struct to `ggml-hpx-region-exec.h` and kernel to `ggml-hpx-region-exec.cpp`.
2. Add a `case` in `ggml_hpx_lower_op` in `ggml-hpx-lower.cpp`.
3. Add a compile-time assert alongside the others at the top of `ggml-hpx-lower.cpp`:
   `static_assert(sizeof(MyCtx) <= GGML_HPX_LOWERING_CTX_BYTES_PER_REGION, "ctx too large");`
4. Add correctness tests in `tests/hpx/`.
5. If multi-region: add a sublayer enum value in `ggml-hpx-packet.h` and a
   validator + compile path in `ggml-hpx-packet.cpp`.

---

## Appendix — what `ggml-cpu` actually does on Q4_K_M decode

Before deciding what HPX should lower, we mapped what the baseline already
runs. Strict baseline: `-DGGML_HPX=OFF -DGGML_HPX_REGION_DAG=OFF`, no
`LLAMA_HPX_*` env vars, only `GGML_CPU_LOG_MULMAT_PATH=1` instrumentation
in `ggml/src/ggml-cpu/ggml-cpu.c`. TinyLlama-1.1B Q4_K_M on Apple M4
(NEON + matmul-int8 + dotprod). Full data:
`hpx-bench/results/2026-04-26-baseline-execution-map/`.

### Three buckets, separated

For every `MUL_MAT` in the decode graph:

| bucket | observation |
|---|---|
| **logical** (`src0->type`) | `q4_K` for Q/K/Out + FFN gate/up; `q6_K` for V, ffn_out, lm_head |
| **physical** (`src0->extra`, buft) | every weight uses `buft = CPU_REPACK`, `extra != NULL`, with trait `q4_K_8x8_q8_K` or `q6_K_8x8_q8_K` |
| **kernel** | the trait family — `ggml_gemv_q*_K_8x8_q8_K` on decode; `ggml_gemm_q*_K_8x8_q8_K` on prefill |

### Why this matters

The standard `ggml_compute_forward_mul_mat` switch
(`ggml-cpu.c:1791-1794` → `ggml-cpu.c:1215`) — the path that uses
`type_traits_cpu[Q4_K].vec_dot = ggml_vec_dot_q4_K_q8_K` — **never runs**
for Q4_K_M weights on this hardware. `ggml_cpu_extra_compute_forward`
(`traits.cpp:12`) short-circuits earlier in `ggml_compute_forward`
(`ggml-cpu.c:1674`) because `CPU_REPACK`'s `init_tensor` populates
`tensor->extra` for every weight.

The repack trait's gemv-vs-gemm split is structural, not in the log:

```
ggml/src/ggml-cpu/repack.cpp:4241
    if (nrows > 3) gemm<...>(...)
    for (iter = nrows - (nrows % 4); iter < nrows; iter++)
        gemv<...>(..., 1 /* nrows */, ...)
```

`nrows = src1_end - src1_start`, ultimately bounded by `src1->ne[1]`. For
pure decode `ne11 = 1`, so `nrows = 1`, so only the gemv kernel fires.
A per-chunk runtime hook to confirm gemv vs gemm was considered and
deliberately skipped — the structural argument is sufficient.

### What this changes for HPX

Earlier reasoning sometimes treated the standard `vec_dot_q4_K_q8_K` as the
baseline HPX would replace or compete with. That comparison is meaningless
because the standard path doesn't run for Q4_K_M. The real baseline is:

> Repacked Q4_K weights, F32 src1 quantized to Q8_K wdata in the existing
> per-thread loop, then per-chunk `ggml_gemv_q4_K_8x8_q8_K` (or `_q6_K_`)
> across `nchunk0 × nchunk1` tiles assigned by `ggml_threadpool_chunk_set` /
> `atomic_fetch_add` (see `repack.cpp:4347-4406`).

The kernels themselves are NEON-tuned and not HPX's business to rewrite.
The chunking and barrier (`ggml_barrier(params->threadpool)` at
`repack.cpp:4352`) are.

### Reframed question for the next pass

**Can HPX schedule around the existing CPU_REPACK gemv kernels better than
`ggml-cpu` already does?** Concretely:

- can we replace the `ggml_threadpool_chunk_set` + atomic-fetch-add chunk
  hand-out with HPX work-stealing across the same tile grid, without
  touching the gemv body?
- can we overlap the F32→Q8_K quantization (`repack.cpp:4296-4309`) with
  the next layer's K/V projection by lifting the global barrier into a
  per-tile dependency edge?
- can we batch multiple consecutive `MUL_MAT` nodes (`Qcur` + `Kcur` +
  `Vcur` share `src1`; `ffn_gate` + `ffn_up` likewise share their input)
  into a single HPX schedule that reuses the quantized wdata?

These are scheduling questions on top of the existing kernels, not
replacement-of-kernel questions. The packet/sublayer composer in
`ggml-hpx-exec-selective.cpp` is the natural place if any of these turn
out to be wins.

### Open

- We have not yet measured a baseline tok/s for this exact path on a
  quiet machine (per the fair-comparison protocol in `CLAUDE.local.md`).
  That number is the floor any HPX scheduling change has to beat.
- The 4 nodes that show `rows=1` even during prefill (`ffn_gate-21`,
  `ffn_up-21`, `ffn_out-21`, `result_output`) are last-position-only by
  llama.cpp design — they're already gemv on prefill. If HPX is going to
  help anywhere in the FFN chain, those last-layer nodes are atypical
  and should not be the proxy.
