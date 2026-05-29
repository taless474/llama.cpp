# HPX exact-session persistent-slot prefix-reuse POC (Design B / B1)

## Motivation

The occupancy / offered-load closeout
([hpx_server_occupancy_offered_load_closeout.md](hpx_server_occupancy_offered_load_closeout.md))
stopped the current branch as a performance-competition project: under offered
load the existing HPX backlog/admission machinery already fills all active
slots, so the throughput ceiling is single-context decode and batch density, not
HPX orchestration overhead.

The one plausible future performance lever identified there was **Design B:
persistent slots / prefix-KV reuse** — keep useful slot/KV state alive across
related requests so a shared prefix is not re-prefilled every request.

**B0** measured whether that opportunity was worth pursuing, using upstream
`llama-server`'s `cache_prompt` as a realized reference and HPX per-iteration
diagnostics as a cross-check. **B1** then implemented a deliberately minimal
slice of Design B inside the HPX engine and measured it end-to-end.

## B0 opportunity result

`llama-server` `cache_prompt=false` vs `true` (TinyLlama F32, greedy), which
reuses a shared prefix via longest-common-prefix matching:

| Workload | TTFT off→on | TTFT improvement | tok/s off→on | tok/s improvement |
| --- | ---: | ---: | ---: | ---: |
| W-long-sys | 399 → 53 ms | 86.8% | 15.7 → 23.5 | +50.3% |
| W-repeat-prefix | 407 → 56 ms | 86.3% | 15.6 → 23.9 | +53.4% |
| W-chat | 177 → 54 ms | 69.3% | 22.6 → 23.8 | +5.1% |
| W-control | — | none | — | none |

B0 showed prefix reuse is worth pursuing. Important framing: B0 measured an
**upper-bound / reference behavior** using `llama-server`'s mature prompt cache,
**not** the cost or behavior of an HPX implementation. It bounds the opportunity;
it does not predict what a given HPX implementation captures.

## B1 scope

B1 intentionally implemented only:

- optional `session_id` on the request
- exact-session persistent-slot reuse
- same `session_id` only
- same resident slot only
- exact token-extension match only (resident tokens must be a full prefix of the
  new prompt, with at least one new token to prefill)
- suffix prefill only (skip the matched prefix)
- full-prefill fallback on mismatch
- inactive resident-slot eviction (oldest-LRU) when no free slot is available

B1 explicitly did **not** implement:

- cross-session reuse
- global longest-common-prefix matching
- `llama_memory_seq_cp`
- context shift

## Implementation slices

- **Slice 1 — `session_id` plumbing.** Optional `session_id` carried
  `submit_request → arrival_msg → waiting_request → seq_state::owning_session_id`.
  Empty string means "no session" and is byte-identical to prior behavior.
- **Slice 2 — resident finalize + teardown.** A successful `session_id`
  completion keeps its slot resident (KV not cleared at finalize) and records
  `resident_tokens` equal to exactly the KV-resident sequence (prompt ++
  generated, truncated to `pos_max+1`; the final sampled token is never placed
  in KV). All resident KV is cleared at engine teardown before the residual-KV
  sweep.
- **Slice 3 — exact-session reuse.** An admission pre-pass binds a same-session
  waiter whose prompt exactly extends a resident slot back onto that slot, sets
  `prefill_cursor` to the matched length, and prefills only the suffix.
- **Slice 4 — eviction + fallback.** When no ordinary free slot exists and a
  waiter cannot exact-reuse, the oldest-LRU inactive resident slot is reclaimed
  (KV cleared, returned to the free pool) and the waiter is admitted with a
  normal full prefill. Mismatches fall back to full prefill.

Key smoke results (engine-level, `tools/hpx-continuous-batch-gate/`):

- no-session behavior unchanged: canonical budget-8 hash
  `0x0619d4d1900c2365` preserved across the plumbing, residency, eviction, and
  server smokes.
- Slice 3 reuse skipped 13 prefix tokens and produced output **byte-identical**
  to the full-prefill baseline (hash `0x886f58b78a7e4f99`).
- Slice 4 eviction passed with oldest-LRU selection; the mismatch-fallback path
  did not reuse and matched its full-prefill baseline hash
  `0x8161ebdda277f73e`.

These are correctness fingerprints under fixed model/prompt/decode/batch shape,
not performance claims.

## B1 benchmark result

End-to-end through `llama-hpx-server` (TinyLlama F32, greedy, `--n-threads 4
--hpx-os-threads 2 --ctx-size 4096`, N=40 measured/cell), HPX no-session vs HPX
exact-session (stable `session_id` per logical session). Reuse is observed via
the per-iteration `prefill_rows_in_iter` diagnostic (total prompt rows
prefilled).

| Workload | TTFT p50 ns→se | TTFT | tok/s ns→se | tok/s | prefill rows ns→se | skipped | reuse fired | failures | hash mismatch ns/se |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: | :---: |
| W-chat | 567.6 → 78.3 ms | +86.2% | 17.5 → 19.3 | +10.7% | 14800 → 2788 | 12012 (81%) | yes | 0/0 | 0/40 |
| W-long-sys | 395.7 → 438.0 ms | −10.7% | 15.4 → 14.6 | −5.6% | 19480 → 19480 | 0 | no | 0/0 | 0/40 |
| W-repeat-prefix | 518.4 → 681.7 ms | −31.5% | 13.8 → 11.7 | −15.0% | 18934 → 18934 | 0 | no | 0/0 | 0/40 |
| W-control | 68.6 → 67.8 ms | +1.1% | 19.3 → 19.6 | +1.4% | 370 → 370 | 0 | no | 0/0 | 0/40 |

Correctness: zero failures in all cells, and **0 of 40 per-request hash
mismatches** between the no-session and session cells in every workload — the
reuse path (W-chat, 81% of prefill rows skipped) produces byte-identical output
to full prefill.

## Interpretation

- B1 was **demonstrated successfully on the measured exact multi-turn
  continuation workload**: where the next prompt genuinely extends the resident
  token sequence (W-chat), reuse fires, skips the matched prefix prefill, and
  preserves output exactly.
- B1 is **not a general shared-prefix cache.** It does not capture the
  W-long-sys / W-repeat-prefix upside that B0 measured, because those workloads
  share a prefix but have **divergent suffixes** — the resident sequence
  includes the previous request's divergent suffix, so the new prompt does not
  exact-extend it. Capturing those requires longest-common-prefix reuse, which
  B1 does not implement.
- Applying `session_id` to **non-extending** workloads can be neutral or
  harmful: it creates resident slots that are never reused, consuming slot/KV
  capacity and incurring mismatch/eviction overhead (e.g. W-repeat-prefix here
  ran slower with a session_id than without).
- **Usage guidance:** set `session_id` only for genuine continuations
  (multi-turn chat where each turn extends the prior prompt). Leave it unset
  otherwise; the no-session path is unchanged.

This is a scoped correctness-and-feasibility result on TinyLlama F32 / greedy /
short decode. It is not a production-readiness claim, and it is not a claim that
HPX broadly outperforms `llama-server`.

## Future work

- **Stop B1 here as complete.** The exact-session POC is correct, validated, and
  measured.
- The natural follow-on is **Design B+ / LCP**, as a separate decision rather
  than a continuation of B1:
  - longest-common-prefix matching between the incoming tokenized prompt and the
    resident tokens (reuse the common prefix even when suffixes diverge), which
    is what would address the W-long-sys / W-repeat-prefix workloads;
  - possibly cross-session prefix sharing later;
  - possibly `llama_memory_seq_cp` later, but not as the first step.
- LCP is the plausible next lever, not a guaranteed win: it carries its own
  correctness obligations (tokenization-boundary handling, KV safety, eviction
  policy) and would need its own opportunity/cost measurement before commitment.

## Artifacts

```text
local/runs/designB-b0-prefix-opportunity-2026-05-28/
local/runs/designB-b1-exact-session-benchmark-2026-05-28/
```

The B1 engine behavior is exercised by the
`tools/hpx-continuous-batch-gate/engine_session_*` smokes (session-id plumbing,
resident-finalize, exact-reuse, evict-inactive, mismatch-fallback).
