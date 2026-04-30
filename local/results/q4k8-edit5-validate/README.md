# Edit 5 validation — Q4_Kx8 packet sublayer

Validation that wiring `penv.mlp_gate_up_glu_q4k8_cache` in `src/llama-context.cpp`
makes the new Q4_Kx8 packet sublayer reachable, with no impact on output and a
measurable speedup within the selective regime.

## Setup

- Branch: `hpx-prefill-orchestrator`
- Binary: `build-hpx-dag/bin/llama-simple` (`GGML_HPX=ON`, `GGML_HPX_REGION_DAG=ON`)
- Model: `models/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`
- Prompt: `"Hello, my name is"`
- Flags: `-n 16 -ngl 0`
- Date: 2026-04-29
- Host: Apple M4

## Configs

| log | env (in addition to `LLAMA_USE_HPX=1`) |
|---|---|
| `C-baseline-no-selective.log` | (none — plain scheduler path) |
| `A-q4k8-off.log` | `LLAMA_HPX_SELECTIVE_MUL_MAT=1`, `LLAMA_HPX_SELECTIVE_MLP_PACKET=1`, `LLAMA_HPX_SELECTIVE_STATS=1`, `LLAMA_HPX_PACKET_MLP_GATE_UP_GLU_Q4K8=0` |
| `B-q4k8-on.log` | as A, but `LLAMA_HPX_PACKET_MLP_GATE_UP_GLU_Q4K8=1` |
| `D-stability-n64.log` | as B, with `-n 64` |
| `E-stability-n128.log` | as B, with `-n 128` |

## Tokens — bit-identical across all three configs

```
<s> Hello, my name is John Smith. I am a software engineer at XYZ Company. I have
```

md5: `239988b2f1ef09a88af084e37e473632` (see `*.tokens.final`).

## Numbers

Per-decode counts and timings from `[hpx-selective] lowered=...` lines (averaged
over the 15 decode steps; counts are constant across decodes):

| metric | C: scheduler | A: Q4K8 off | B: Q4K8 on |
|---|---:|---:|---:|
| lowered nodes (per graph) | n/a | 135 | 135 |
| fallback nodes (per graph) | n/a | 488 | 488 |
| fallback runs (per graph) | n/a | 102 | 102 |
| packet matches (per graph) | n/a | 22 | 22 |
| **packet nodes (per graph)** | n/a | **22** | **66** |
| avg lowered_ms | n/a | 94.26 | 61.46 |
| avg fallback_ms | n/a | 36.70 | 26.43 |
| avg packet_ms | n/a | 3.90 | 13.73 |
| eval time (15 decodes, ms) | 145.94 | 2247.99 | 1533.34 |
| **tok/s** | **102.78** | **6.67** | **9.78** |

## Reading

### Headline: A → B

- Packet match count is unchanged (22). Each match bundles 2 additional nodes
  when Q4K8 is on, for **+44 packet nodes per decode graph (22 × 2)**.
- Those 44 nodes were previously hitting the fallback path; absorbing them into
  packets shifts ~10 ms/decode out of fallback/lowered and ~10 ms/decode into
  packet dispatch. Net: 25% less per-decode dispatch time, +47% tok/s.
- Lowered/fallback node and run counts are unchanged — Q4K8 only changes
  packet bundling, not the selective lowering decision.
- Edit 5 (`penv.mlp_gate_up_glu_q4k8_cache = packet_active ? hpx_mlp_gate_up_glu_q4k8_cache : nullptr;`)
  is what makes this path reachable.

### Context: vs C

- The plain scheduler path is ~10× faster than selective on this graph.
- Selective leaves 488 fallback nodes per decode, **coalesced into 102 fallback
  runs**. Dispatch cost is paid per fallback run (not per fallback node), but
  102 runs/decode is still a lot of substrate-entry overhead that the
  scheduler avoids by running the whole CPU graph in a single compute call.
- The Q4K8 packet recovers some of that loss but does not close the gap to C.

This is consistent with the known baseline path: every Q4_K/Q6_K weight uses
CPU_REPACK + q*_K_8x8_q8_K gemv on decode, which the plain scheduler runs as
one whole-graph compute on the configured CPU threadpool.

## Stability — longer runs at config B (Q4K8 on)

Same env as B, increased token budget. Goal: confirm packet shape stays
`packet=22(66 nodes)` per decode and output stays coherent through to the
last token.

| run | n | decode steps | stat lines `packet=22(66 nodes)` | off-target shapes | exit | tok/s |
|---|---:|---:|---:|---:|---:|---:|
| D | 64 | 63 | 63/63 | 0 | 0 | 13.07 |
| E | 128 | 127 | 127/127 | 0 | 0 | 12.95 |

- Output is coherent end-to-end at both lengths; no late-token failure or
  repetition. Tail of E:
  `... I have been using MySQL to store the data. However, I am not sure how to design the database to store the data efficiently.`
- Throughput is slightly higher than at n=16 (9.78 → ~13 t/s), so per-graph
  fixed overhead amortizes; nothing creeps as the runtime/cache age.

## Files

- `*.log` — full run output (5 configs: C, A, B, D, E)
- `*.tokens.final` — extracted token streams used for bit-identical check
- `run.log`, `run-full-gates.log` — earlier exploratory runs from this session
- `F-hist-attention-shapes.log` — node-histogram diagnostic for attention-block packet candidate review
