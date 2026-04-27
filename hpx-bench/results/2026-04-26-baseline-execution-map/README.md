# 2026-04-26 — TinyLlama Q4_K_M baseline ggml-cpu execution map

## Goal

Before deciding whether HPX belongs anywhere on the Q4_K_M decode path, map
what `ggml-cpu` actually does for every `MUL_MAT` node — separating three
things that have been getting conflated:

1. **logical** — what the tensor says it is (`ggml_type_name(src0->type)`)
2. **physical** — what buffer type and `tensor_traits` are installed in
   `src0->extra` (the repacked path identity)
3. **kernel** — what gemv/gemm or vec_dot family will actually run

This run is a strict baseline — no HPX in the binary, no HPX env vars,
only the new `GGML_CPU_LOG_MULMAT_PATH` instrumentation.

## Build

| | |
|---|---|
| commit | `e0b332bb5` (working tree dirty: instrumentation + repack hoist) |
| build dir | `build-baseline-no-hpx/` |
| CMake flags | `-DGGML_HPX=OFF -DGGML_HPX_REGION_DAG=OFF -DCMAKE_BUILD_TYPE=Release` |
| binary | `build-baseline-no-hpx/bin/llama-simple` |
| model | `models/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` |
| host | Apple M4 (NEON + matmul-int8 + dotprod) |

Build-time HPX availability and runtime HPX engagement are deliberately both
off. `GGML_HPX=OFF` ensures `ggml-hpx` is not linked at all, so even if some
future wiring tried to use an HPX path it would not exist in this binary.

## Reproduce

Two captures, fresh process each time so the pointer-keyed dedupe state resets.

```
GGML_CPU_LOG_MULMAT_PATH=1 \
  ./build-baseline-no-hpx/bin/llama-simple \
  -m models/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  -ngl 0 -n 4 "The capital of France is" \
  > decode.stdout 2> bench.log

GGML_CPU_LOG_MULMAT_PATH=1 \
  ./build-baseline-no-hpx/bin/llama-simple \
  -m models/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  -ngl 0 -n 1 "Hi" \
  > decode_only.stdout 2> decode_only.log
```

`-ngl 0` keeps everything on the CPU backend (Metal is otherwise auto-attached
by the build).

## Findings — the three-bucket answer

For real TinyLlama Q4_K_M baseline CPU decode on M4:

| bucket | value |
|---|---|
| logical | `q4_K` for attention Q/K/Output and FFN gate/up; `q6_K` for V, FFN-out, and `result_output` |
| physical | `buft = CPU_REPACK`, `extra != NULL`, `trait = q4_K_8x8_q8_K` (Q4_K) or `q6_K_8x8_q8_K` (Q6_K) |
| kernel | the trait itself — same `q4_K_8x8_q8_K` / `q6_K_8x8_q8_K` family |

**Confirmed: Q4_K_M decode goes through `CPU_REPACK` + `q4_K_8x8_q8_K`.** Not
through the standard non-repacked `ggml_vec_dot_q4_K_q8_K` path. Every Q4_K
weight in the model has `tensor->extra` populated by the repack buffer's
`init_tensor`, and `ggml_cpu_extra_compute_forward` short-circuits the
standard switch.

### gemv vs gemm — structural, not in the log directly

The trait runs **both** kernels and decides per chunk:

```
ggml/src/ggml-cpu/repack.cpp:4241
    if (nrows > 3) {
        gemm<...>(...)
    }
    for (int iter = nrows - (nrows % 4); iter < nrows; iter++) {
        gemv<...>(..., 1 /* nrows */, ...)
    }
```

`nrows = src1_end - src1_start`, ultimately bounded by `src1->ne[1]`. For pure
decode, `ne11 = 1`, so `nrows = 1` and **only `ggml_gemv_q4_K_8x8_q8_K` /
`ggml_gemv_q6_K_8x8_q8_K` fire**. Prefill of N>3 tokens additionally invokes
the `_gemm_` variant.

The instrumentation logs the first `src1->ne[1]` it sees per tensor. Both
captures show ≤3 (rows=6 prefill of "The capital of France is"; rows=2 prefill
of "Hi"); both fall in the trait's gemv branch. The 4 entries showing rows=1
in both captures (`ffn_gate-21`, `ffn_up-21`, `ffn_out-21`, `result_output`)
are the tensors llama.cpp short-circuits to the last token even during prefill
because only last-position logits are needed.

### Per-tensor map (one decode step, layer 0 shown — every layer is identical)

| name | logical | trait | shape (cols × out_cols) |
|---|---|---|---|
| `Qcur-L` | `q4_K` | `q4_K_8x8_q8_K` | 2048 × 2048 |
| `Kcur-L` | `q4_K` | `q4_K_8x8_q8_K` | 2048 × 256 |
| `Vcur-L` | `q6_K` | `q6_K_8x8_q8_K` | 2048 × 256 |
| `attn_out-L` | `q4_K` | `q4_K_8x8_q8_K` | 2048 × 2048 |
| `ffn_gate-L` | `q4_K` | `q4_K_8x8_q8_K` | 2048 × 5632 |
| `ffn_up-L` | `q4_K` | `q4_K_8x8_q8_K` | 2048 × 5632 |
| `ffn_out-L` | `q6_K` | `q6_K_8x8_q8_K` | 5632 × 2048 |
| `result_output` (final) | `q6_K` | `q6_K_8x8_q8_K` | 2048 × 32000 |

Note the Q4_K_M mix: most weights are Q4_K, but `Vcur`, `ffn_out`, and the
output head are Q6_K. Both classes are repacked.

### Counts

| capture | rows=6 | rows=2 | rows=1 | total unique nodes |
|---|---|---|---|---|
| decode.stdout (prompt 6 tok, n=4) | 151 | 0 | 4 | 155 |
| decode_only.stdout (prompt "Hi" 2 tok, n=1) | 0 | 151 | 4 | 155 |

Same 155 unique tensor pointers across both runs → llama.cpp reuses the
compute graph; tensor identity is stable across decode steps. The pointer
dedupe gives one log line per node; `src1->ne[1]` shown is whichever shape
was active on first compute (always prefill, by construction).

## What this confirms vs. what is still open

**Confirmed:**
- The `ggml_cpu_extra_compute_forward` extra path is the *only* path that
  fires for Q4_K_M weights in this binary. The non-repacked
  `ggml_vec_dot_q4_K_q8_K` standard branch is dead for this model.
- Trait selection on M4 is `q4_K_8x8_q8_K` (NEON + matmul-int8, ne[1] % 8 == 0
  → branch in `ggml_repack_get_optimal_repack_type`, `repack.cpp:4606-4609`).
- The trait dispatches `gemv` vs `gemm` per chunk on `src1->ne[1]`. Decode is
  `gemv` deterministically.

**Open / next:**
- The instrumentation logs first-seen shape; to capture the actual
  decode-step `nrows` per chunk we would need a second hook inside
  `forward_mul_mat_one_chunk`. Not done — out of scope for this baseline pass.
- We have not yet verified what HPX would do with these nodes. That's a
  separate question now that we know the baseline is CPU_REPACK gemv, not the
  standard vec_dot path.

## Files

- `bench.log` — stderr from the 6-token prompt + 4 decode steps run (155 nodes × ~7 lines)
- `decode.stdout` — model output of that run
- `decode_only.log` — stderr from the "Hi" + 1 decode step run
- `decode_only.stdout` — model output of that run
- `README.md` — this file

## Implementation notes

- Logger lives in `ggml/src/ggml-cpu/ggml-cpu.c`, env-gated, fires once per
  unique `tensor` pointer, gated by `params->ith == 0`.
- Trait names come from a new helper `ggml_repack_extra_traits_name` in
  `ggml/src/ggml-cpu/repack.cpp`. To make the trait pointer identifiable from
  outside its dispatch function, the static instances were hoisted out of
  `ggml_repack_get_optimal_repack_type` into a file-scope anonymous namespace.
  Same internal linkage, same lifetime, same addresses.
- Standard-path kernel resolution uses `dladdr` guarded by
  `__APPLE__ / __linux__ / __FreeBSD__`. Not exercised in this run because
  every Q4_K_M MUL_MAT goes through the trait path.
