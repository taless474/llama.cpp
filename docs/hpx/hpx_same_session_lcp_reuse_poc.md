# HPX same-session longest-common-prefix reuse POC (Design B / B+1)

## Question

B1 ([hpx_exact_session_prefix_reuse_poc.md](hpx_exact_session_prefix_reuse_poc.md))
implemented **exact-session** persistent-slot reuse: a same-session request
whose prompt *exactly extends* a resident slot skips the matched-prefix
prefill. That helped genuine continuations (W-chat: TTFT p50 567.6 ms → 78.3 ms,
tok/s +10.7%, byte-identical output), but it did **not** help shared-prefix
*divergent-suffix* workloads (W-long-sys, W-repeat-prefix). Exact reuse requires
the new prompt to extend the entire resident token sequence, so two requests
that share a long common prefix but then diverge cannot reuse anything.

B+1 asks: can **same-session longest-common-prefix (LCP) reuse** recover that
missed shared-prefix upside — keep the matched common prefix, trim the divergent
resident tail, prefill only the new suffix — **without** cross-session reuse,
`seq_cp`, or context shift?

## Scope

B+1 is deliberately minimal:

* same-session LCP only — a resident slot is reused only by its own
  `session_id`
* no cross-session LCP — a different session never matches a resident prefix
* no `seq_cp`
* no context shift
* no tokenizer / sampler / `llama_decode` changes
* not a production-readiness claim
* not a general shared-prefix cache across users or sessions

## Implementation summary

* A resident slot carries `resident_tokens` (the exact token sequence held in
  its KV) plus its `session_id`.
* An incoming same-session prompt is compared against same-session resident
  slots by **token-level longest common prefix**
  (`find_best_lcp_resident_slot`).
* **Exact extension** (the resident tokens are a full prefix of the new prompt)
  remains the B1 fast path: no threshold, no trim, suffix prefilled from the
  resident length onward.
* **Partial LCP** (matched < resident length, with at least one new token to
  prefill) requires a useful match threshold to be worthwhile, currently
  `LCP_MIN_TOKENS = 32`. Below threshold the request falls through to a normal
  full prefill.
* For a partial match, the divergent resident tail `[matched, end)` is removed
  with `llama_memory_seq_rm(seq_id, matched, -1)`, and the new suffix is
  prefilled from `matched` onward at contiguous positions, continuing from the
  retained prefix KV `[0, matched)`.
* No cross-session resident is ever considered.
* B1's resident-slot eviction (oldest inactive resident) and mismatch fallback
  behavior are unchanged.

## Smoke validation

Engine-level smokes (TinyLlama F32, greedy, fixed batch shape) validate the
mechanics. In a fixed batch shape the LCP-reuse output is byte-identical to the
fresh full-prefill baseline computed in the same shape, so these smokes assert
hash equality directly:

* **LCP divergent-suffix smoke**
  (`engine_session_lcp_divergent_suffix_smoke.cpp`):
  * matched tokens = 43, trimmed tokens = 12
  * `lcp_reuse_admitted_count = 1` (no exact reuse)
  * output matched the full-prefill baseline hash `0x917f8acbae3746cf`
* **Threshold tiny-prefix smoke**
  (`engine_session_lcp_threshold_no_reuse_smoke.cpp`):
  * common prefix below `LCP_MIN_TOKENS` → no reuse
  * output matched baseline hash `0x88f68c24302e81fc`
* **Cross-session same-prefix smoke**
  (`engine_session_cross_session_no_lcp_smoke.cpp`):
  * identical prefix in a *different* session → no reuse
  * output matched baseline hash `0x917f8acbae3746cf`
* **B1 exact reuse** smokes still passed.
* **No-session b8 anchor** still passed: `0x0619d4d1900c2365`.

## Benchmark result

End-to-end via `llama-hpx-server` (TinyLlama F32, greedy), comparing an HPX
no-session cell against an HPX session cell that supplies a stable `session_id`
per logical session.

Small W-long-sys smoke (LCP firing confirmed):

* TTFT ≈ 395 ms → 77 ms (≈ 80% TTFT improvement)
* tok/s +48.6%
* ≈ 78% prefill skipped

Full benchmark:

| Workload | tok/s improvement | reuse |
| --- | ---: | --- |
| W-long-sys | +40.2% | fired, ≈ 88–97% prefill skipped |
| W-repeat-prefix | +52.0% | fired, ≈ 88–97% prefill skipped |
| W-chat | +26.8% | fired (exact or LCP) |
| W-control | neutral | none (distinct sessions) |

Correctness vs the no-session full-prefill baseline (hash comparison):

* W-long-sys: 0/40 mismatches
* W-repeat-prefix: 1/40 mismatches
* W-chat: 0/40 mismatches
* W-control: 0/40 mismatches

## Mismatch investigation

The single W-repeat-prefix mismatch was investigated in detail
(artifacts under `local/runs/designB-bplus1-lcp-repro-2026-05-29/`).

* **Reproducible across 4 runs** (original + 3 reruns): always exactly one
  mismatch, always at measured index 30 / request `i=32`.
* Prompt: `_REPEAT_PREFIX + " Variation 32. Continue:"`.
* no-session hash: `0x35fcec78e7ef42d3`
* session/LCP hash: `0x7c6c2a5fa91b4c42`
* Both completed cleanly; both `n_decoded = 16`.
* Outputs shared `"1."` then diverged at the second content token:
  * no-session chose token `16585`, text `" Background"`
    (echoing the literal repeated prefix)
  * session/LCP chose token `349`, text `" P"` → `"refix"`
    (echoing `"prefix-reuse benchmark"`)
  * Both are coherent literal phrases from the prompt, which repeats
    `"Background reference text for the prefix-reuse benchmark."` 40 times.
* **KV positions were correct on both paths.** A trace
  (`LLAMA_HPX_CB_TRACE=1`) shows the first generated token at KV position 451 on
  *both* paths, then 452, 453, 454, … contiguously. The trace confirms the old
  resident tail was trimmed and the suffix was prefilled contiguously from the
  matched prefix.
* 39 of 40 reuse outputs were byte-identical to fresh full-prefill; only this
  prompt diverged, even though all 40 took the identical trim + suffix-prefill
  code path.

**Conclusion:** this is deterministic cross-shape floating-point / greedy
near-tie sensitivity, **not** a stale-KV, position, or tail-trim bug. There is
no evidence of KV contamination or a trim/position bug.

**Interpretation (precise wording):** partial-LCP reuse is **structurally
correct** and **performance-positive**, but **byte-identical output versus fresh
full-prefill is not guaranteed**, because the reused prefix KV may have been
computed under a different batch shape than a fresh full prefill. When two
candidate next tokens are a near-tie under greedy decoding, that cross-shape
floating-point difference can flip the argmax into a different — still coherent —
continuation. This is the same class of cross-shape FP non-determinism that
CLAUDE.md already excludes from correctness gates. (Note: the engine smokes
above *do* assert byte-identity because their baseline and reuse run in the same
fixed batch shape; the benchmark mismatch arises precisely because the resident
prefix was computed under a prior request's batch shape.)

This is an observation of deterministic cross-shape floating-point sensitivity.
It is **not** a claim that semantic equivalence has been formally proven.

## Correctness policy

* Byte-identity remains appropriate for **no-session fixed-shape anchors**
  (e.g. the b8 anchor `0x0619d4d1900c2365`).
* Byte-identity remains appropriate for the **B1 exact-reuse smoke** and the
  **B+1 LCP smokes**, where it passed under the tested fixed shape.
* For **partial-LCP reuse against fresh full-prefill across varying batch
  shapes**, byte-identity is too strict: cross-shape FP sensitivity can flip
  greedy near-ties.
* The gates for partial-LCP reuse should instead verify:
  * token-level prefix match
  * correct tail trim
  * contiguous positions
  * no cross-session reuse
  * no stale-KV contamination
  * no failures or KV leaks
  * mismatch rate and concrete examples are reported when comparing against
    fresh full-prefill (rather than asserting byte-identity)

## Final decision

B+1 same-session LCP is a **successful, performance-positive POC with a
determinism caveat**. It recovers the shared-prefix upside that B1 missed
(W-long-sys, W-repeat-prefix), while remaining same-session only and avoiding
cross-session privacy/policy issues. **Stop B+1 here and document it.**

Future work (separate decision, not in scope here):

* **B+2 cross-session LCP** could recover shared prefixes across independent
  sessions/users. It should require an explicit policy decision, because
  cross-session cache hits can create a timing side channel.
* `seq_cp` and context shift remain out of scope for this doc.

## Artifacts

* `local/runs/designB-bplus1-lcp-benchmark-2026-05-28/` — full B+1 benchmark
* `local/runs/designB-bplus1-lcp-repro-2026-05-29/` — mismatch reproduction,
  token-id probe, and `FINDINGS.md`
