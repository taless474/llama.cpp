# Exp 4 — Packet compile cost (env-gated compile log)

**Date:** 2026-04-20
**Commit:** 26578ce22 + uncommitted working tree + compile-log instrumentation
(ggml-hpx-exec-selective.cpp, env-gated by `LLAMA_HPX_PACKET_COMPILE_LOG=1`,
both `lookup_or_compile_mlp` and `lookup_or_compile_mlp_glu` sites)
**Binary:** `build-hpx-dag/bin/llama-simple` (rebuilt with compile-log edit)
**Model:** `models/tinyllama/tinyllama-1.1b-chat-v1.0.F32.gguf`
**Env:** `LLAMA_USE_HPX=1 LLAMA_HPX_SELECTIVE_MUL_MAT=1 LLAMA_HPX_SELECTIVE_MLP_PACKET=1 LLAMA_HPX_SELECTIVE_STATS=1 LLAMA_HPX_PACKET_COMPILE_LOG=1`
**Invocation:** `-ngl 0 -n 16 "Hello"`

## Compile-log output

Exactly one line fired:

```
[hpx-packet-compile] matcher=glu out_cols=5632 cols=2048 rows=1 compile_ns=1083
```

- **matcher=glu** confirms the GLU path is the only live compile site on real
  TinyLlama decode. `matcher=gate_up` never fires — as expected, since
  GGML_OP_GLU[SWIGLU] is TinyLlama's MLP trigger.
- **compile_ns=1083** = **1.083 µs ≈ 0.001 ms**.

## First-token breakdown

Token 1 `packet_ms` in this run: **42.452 ms** (line 403). Of that:

| component                | value      |
|--------------------------|-----------:|
| compile (one-time)       | 0.001 ms   |
| first-dispatch residual  | ~42.45 ms  |

The compile cost is four orders of magnitude below the dispatch cost. No
hidden "warmup compilation" exists.

## Cross-check with Exp 1 v2 and Exp 5

Per-token `packet_ms` in this run: mean 40.8 ms, first token 42.5 ms, last
token 42.5 ms. Consistent with v2 (mean 41.4, stdev 3.2). The compile-log
instrumentation does not perturb steady-state packet dispatch.

## Findings

1. **Compile is free.** ~1 µs is well below any measurable cliff.
2. **Only the GLU matcher fires** — gate/up is dead code on real llama until
   a model without SWIGLU is tested.
3. **No warmup cliff remains unexplained.** Combined with Exp 1 v2 (no cliff)
   and Exp 5 (honest 1.12× speedup), the B.1 investigation is closed.

## Instrumentation left in place

The `LLAMA_HPX_PACKET_COMPILE_LOG` env var and the two-site log are off by
default (single `std::getenv` + `atoi` read once per process, cached in a
function-local static). Zero overhead when unset. Keep for future
diagnostics; removing is optional.
