# Q4_K Non-Repacked Benchmark: Does HPX Own Q4_K Matmul?

**Date:** 2026-04-26
**Commit:** e0b332bb5 ("Guard Q4_K lowering against repacked CPU tensors")
**Binary:** `build-hpx-dag/bin/llama-simple` (single rebuild, same binary for both arms)
**Model:** `models/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`
**Invocation:** `-ngl 0 -n 16 "Hello"` on both arms
**Machine:** Apple M4, quiet (WindowServer + VSCode only at check time)

## Short answer

HPX cannot own Q4_K matmul on this machine. The guard `if (w->extra != nullptr) return false`
fires for every Q4_K weight in TinyLlama. On Apple M4 (NEON + matmul_int8), ggml-cpu repack
all Q4_K tensors whose `ne[1] % 8 == 0` into Q4_Kx8 layout and sets `tensor->extra`. Every
TinyLlama Q4_K weight satisfies this condition (2048, 256, 5632 are all divisible by 8).

## Observed signals vs expected

| Expected                        | Observed                          | Verdict |
|---------------------------------|-----------------------------------|---------|
| normal block_q4_K tensors       | all repacked (extra != nullptr)   | BLOCKED |
| lowered count increases         | lowered=67 (same as F32 run)      | no change |
| fallback count decreases        | fallback=622 (MUL_MAT all CPU)    | no change |
| packet may still be 0           | packet=0                          | CORRECT |
| TinyLlama output matches        | "Hello, World! 5. Python: print..." consistent | CORRECT |

## Timing data (3 reps, -ngl 0, CPU-only)

| arm             | eval ms/tok | eval tok/s |
|-----------------|------------:|-----------:|
| hpx_selective   | 93.7 ± 6.4  | 10.7 ± 0.8 |
| baseline_cpu    |  9.97 ± 0.6 | 100.2 ± 5.3 |

HPX selective path is ~9x slower than plain CPU. This is NOT caused by Q4_K lowering
(which never fires). It is the per-node dispatch overhead of the selective executor
calling `ggml_backend_graph_compute(cpu_be, &view)` 622 times with 1-node graph views,
instead of once for all 689 nodes.

## Why the guard is correct

`ggml_repack_get_optimal_repack_type` in `ggml/src/ggml-cpu/repack.cpp` returns
`&q4_K_8x8_q8_K` for any Q4_K tensor with `ne[1] % 8 == 0` on ARM with
`ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()`. This repack converts from
`block_q4_K[]` layout to `block_q4_Kx8[]` and sets `tensor->extra` to the traits pointer.
Our Q4_K kernel reads `block_q4_K` layout — reading repacked memory would corrupt output.
The guard prevents this correctly.

## What "HPX owns Q4_K" requires

To get non-repacked Q4_K tensors, one of the following would be needed:
- A custom `ggml_backend_buffer_type_t` that skips repack (e.g. plain CPU host buffer)
- A model whose Q4_K weight `ne[1]` is not divisible by 8 (unusual for transformers)
- The unit test path (`tests/hpx/test_hpx_selective_mul_mat_q4_k.cpp`) which builds
  synthetic tensors with correct block layout and no repack

The unit test is the right vehicle to verify Q4_K lowering correctness in isolation.
