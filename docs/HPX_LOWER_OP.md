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
