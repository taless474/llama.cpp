# Exp 1 — Per-token packet_ms trace (PACKET=1, -n 16)

**Date:** 2026-04-20
**Commit:** 26578ce22 (hpx-prefill-orchestrator, with uncommitted Apr 18–20 changes in working tree)
**Binary:** `build-hpx-dag/bin/llama-simple` (rebuilt 2026-04-20, sources up to Apr 18 23:20)
**Model:** `models/tinyllama/tinyllama-1.1b-chat-v1.0.F32.gguf`
**Env:** `LLAMA_USE_HPX=1 LLAMA_HPX_SELECTIVE_MUL_MAT=1 LLAMA_HPX_SELECTIVE_MLP_PACKET=1 LLAMA_HPX_SELECTIVE_STATS=1`
**Invocation:** `-ngl 0 -n 16 "Hello"`

## Per-token results

See `bench_per_token.csv`. Summary (all values in ms):

| token | lowered | fallback | packet  |
|------:|--------:|---------:|--------:|
| 1     | 2049    | 62       | **3150** |
| 2     | 371     | 56       | **3232** |
| 3     | 587     | 66       | 2071    |
| 4     | 2181    | 53       | 1848    |
| 5     | 357     | 59       | 1184    |
| 6     | 370     | 54       | 1467    |
| 7     | 531     | 57       | 1470    |
| 8     | 1840    | 63       | 1128    |
| 9     | 1676    | 52       | 1620    |
| 10    | 1688    | 57       | 2354    |
| 11    | 501     | 65       | 1822    |
| 12    | 1058    | 57       | 1858    |
| 13    | 394     | 70       | 1472    |
| 14    | 1459    | 57       | **2971** |
| 15    | 444     | 62       | 2116    |

Prefill (token 0): `packet=0` — packet env gated off during prefill as expected.
Eval: `3078 ms/token (0.32 tok/s)`.

## Findings

- `packet_ms` does **not** show a clean monotonic warmup. Tokens 1–2 are the
  peak, but token 14 spikes back to 2971 ms. Values stay in 1100–3200 ms range
  throughout.
- `lowered_ms` is also highly erratic (357–2181 ms), varying 6× token-to-token
  on the same work.
- Both magnitudes are **~10× worse** than the Apr 18 smoke run
  (PACKET=1 steady was ~60 ms packet, ~230 ms lowered).

## Interpretation

This run is **noise-dominated**, not warmup-dominated. The fact that a pure-CPU
work chunk (`lowered_ms`) — which has nothing to do with the packet path —
varies 6× on identical work strongly suggests **machine-level confound**:
thermal throttling, DVFS oscillation, or background load.

We cannot see a warmup plateau through this much noise. Exp 1 needs to be
re-run on a cooled, quiesced machine before it is useful for cliff shape
analysis.

## Open questions

- Was the Apr 18 smoke-run binary built from the same source tree? If
  `build-hpx-dag/bin/llama-simple` timestamp was Apr 17 and sources changed
  Apr 18, the smoke-run binary was from an earlier snapshot. Running that
  binary vs this one on the same hardware would separate "code changed" from
  "machine state changed".
- Current M4 thermal state unknown. Next run should start from a known-cold
  machine (fans up or idle period), log `powermetrics`/`pmset -g therm` if
  accessible, and compare `lowered_ms` variance as a machine-health proxy.
