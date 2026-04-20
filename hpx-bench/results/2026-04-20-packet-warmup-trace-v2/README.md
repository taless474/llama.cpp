# Exp 1 v2 — Per-token packet_ms trace (PACKET=1, -n 16, quiet machine)

**Date:** 2026-04-20
**Commit:** 26578ce22 + uncommitted working tree (GLU packet path enabled)
**Binary:** `build-hpx-dag/bin/llama-simple` (rebuilt 2026-04-20, post-unstash)
**Model:** `models/tinyllama/tinyllama-1.1b-chat-v1.0.F32.gguf`
**Env:** `LLAMA_USE_HPX=1 LLAMA_HPX_SELECTIVE_MUL_MAT=1 LLAMA_HPX_SELECTIVE_MLP_PACKET=1 LLAMA_HPX_SELECTIVE_STATS=1`
**Invocation:** `-ngl 0 -n 16 "Hello"`

## Machine prep
Previous run of this directory (`2026-04-20-packet-warmup-trace/`) was
contaminated by a stray `build-hpx-cpu/bin/llama-bench` at 98% CPU and by
`mds_stores` spotlight indexing post-build. Both eliminated before this run.
`pmset -g therm` reports no warnings.

## Per-token results (see `bench_per_token.csv`)

| token | lowered_ms | fallback_ms | **packet_ms** |
|------:|-----------:|------------:|--------------:|
| 1     | 148.7      | 26.5        | **39.4** |
| 2     | 153.3      | 60.6        | 40.3 |
| 3     | 132.8      | 50.4        | 38.4 |
| 4     | 179.7      | 49.8        | 37.1 |
| 5     |  79.8      | 14.0        | 39.9 |
| 6     |  72.8      | 11.3        | 42.8 |
| 7     |  61.6      | 18.1        | 43.3 |
| 8     | 100.7      | 42.2        | 46.8 |
| 9     | 171.5      | 34.9        | 45.2 |
| 10    | 201.1      | 13.9        | 44.8 |
| 11    | 217.6      | 43.9        | 43.0 |
| 12    | 154.9      | 14.0        | 41.0 |
| 13    | 178.5      | 38.3        | 43.5 |
| 14    | 184.0      | 44.4        | 35.4 |
| 15    | 126.0      | 47.1        | 40.1 |

Eval: **4.55 tok/s** (vs Apr 18 smoke's 2.57 tok/s).

## Stats
- `packet_ms`: n=15, min=35.4, max=46.8, **mean=41.4**, stdev=3.2
- `lowered_ms`: n=15, min=61.6, max=217.6, mean=144.2, stdev=47.8

## Finding: no warmup cliff at -n 16 on a quiet machine

- First decode token's `packet_ms` is 39.4 ms — **indistinguishable** from
  the rest of the run.
- `packet_ms` stdev is only 3.2 ms on a ~41 ms mean (~8%). Tight.
- Tokens 1–3 mean (39.4) ≈ tokens 13–15 mean (39.7). Zero detectable trend.

This contradicts both (a) the "225 ms first token / ~60 ms steady" reported
in `2026-04-18-glu-packet-smoke/` and (b) the "warmup cliff lasting several
tokens" the user originally described.

## Interpretation

The earlier "HPX packet warmup cliff" is most likely **not a real code-level
warmup**. It was observed on machines with CPU contention and/or unstable
thermal/DVFS state, and disappears on a quiet, cool machine.

The only plausible genuine one-time cost (lazy packet-runtime init +
first-shape compile) is folded into token 1's 39.4 ms. That number is already
at steady state, meaning the one-time cost is at most a few ms — not
hundreds.

## Implications for the original plan

- Exp 2 (back-to-back driver) and Exp 3 (dummy warmup) become low-priority.
  They were designed to distinguish intra-process from inter-process warmup;
  if there is no warmup, there is nothing to distinguish.
- Exp 4 (compile-miss log) is still worth doing, as a small insurance:
  confirm compile ns is indeed negligible relative to the ~40 ms steady-state
  packet dispatch.
- Exp 5 (fair comparison protocol) becomes the real priority: we now need a
  clean PACKET=0 baseline on the same quiet machine to honestly compare
  against this 41.4 ms mean / 4.55 tok/s number.

## Next step proposal

Run the matching PACKET=0 baseline immediately (same machine state, same
invocation, only `LLAMA_HPX_SELECTIVE_MLP_PACKET` toggled to 0), and compare
eval tok/s. That gives an honest B.1 speedup number, not the confounded one.

## Open question

`lowered_ms` stdev is 47.8 on mean 144 (33%). Much noisier than packet_ms.
That's the pure-lowered portion of the work and may reflect graph-reuse
cache behavior or scheduler variance on the non-packet path. Noted for
later; not blocking here.
