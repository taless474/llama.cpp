# PrefillBudgetPolicy — candidate design note

Status: **design record.** Originally landed as a candidate design; Slices B–F
and the Option B server-surface (`--prefill-budget-rows`) have since been
implemented. The as-built result lives in
`prefill_budget_policy_result.md` and the server-path confirmation in
`hpx-bench/experiments/16_prefill_budget_policy_server_w1c/`. This document is
preserved as the original design record (problem statement, policy idea,
invariants, and acceptance gates against which the implementation was checked).

Revision note (post-review): §3, §4, §5, §7, §8, and §9 were corrected against
the accepted skeptical review `local/reviews/prefill_budget_policy_review.md`.
The most important change: the earlier claim that chunked prefill must reproduce
the whole-prompt token hash has been **removed** — it conflicts with `CLAUDE.md`'s
batch-shape floating-point rule. §4 now states the two previously-missing engine
mechanics (BUILD-branch re-keying and the sample-loop skip) explicitly.

This note is the detailed expansion of the `PrefillBudgetPolicy` hook already
named in `docs/hpx/hpx_native_serving_control_plane_design.md` §7 (policy hooks),
§8 (prefill/decode split), and §12 item 1 (ranked improvements). It does not
supersede that doc; it grounds the hook in the W1 diagnostic evidence.

## 1. Problem statement

The engine task composes one shared `llama_batch` per iter in its BUILD phase
(`iter_build_batch`) and calls `llama_decode` exactly once per iter. Today a
newly admitted request contributes **all** of its prompt-token rows to the batch
of the single iter in which it is admitted — whole-prompt prefill, bounded only
by `batch_capacity`. There is no policy that caps prompt rows per iter.

Consequence, measured (not asserted): when a long prompt is admitted **while
another sequence is already decoding**, that decode's next token is produced in
the same iter as the whole prefill, so it waits for roughly the entire prefill
cost before advancing.

### Evidence (local artifacts; observational, TinyLlama F32, one machine)

- **W1a** — `local/runs/w1-mixed-short-long-2026-05-26/`. Every request used the
  fixed 6-token anchor prompt. Two negative results: no weak-batch-filling
  pathology, and no measurable HPX service-layer overhead. But because prefill
  was fixed at 6 tokens, W1a could not exercise long-prefill interference at all.
- **W1b** — `local/runs/w1b-varied-prefill-2026-05-26/`. Varying prompt length
  showed larger `prefill_rows_in_iter` increases `llama_decode_wall_us`, while
  HPX `non_decode_us` stayed tiny (~100 µs) and shape-independent. The headline
  interference cell was labelled **NOISY**: the random mixed workload produced
  too-close low-prefill cells (pf=6 vs pf=31 effectively tied) and uneven sample
  counts, so the curve could not be called cleanly.
- **W1c** — `local/runs/w1c-staggered-prefill-2026-05-26/`. A deliberate stagger
  (decode-holding anchor + probe submitted 120 ms later) forced the interference
  shape `(prefill_rows>0, decode_rows>0, active_seq_count==2)` with a 1.00
  stagger-hit rate (48/48). Across measured prefill rows [11, 79, 251, 977]:
  - interference `llama_decode_wall_us` is **INCREASING** (monotonic,
    slope ≈ 823 µs/row, R² = 0.998) — this resolves W1b's NOISY;
  - it is **ADDITIVE** for all four classes (residual 0.5–2.6%):
    `interference_med ≈ prefill_alone_med + decode_alone_med` (one decode ≈
    40.5 ms);
  - HPX `non_decode_us` stayed ~95–118 µs — **no measurable HPX cross-cost**.

The additive result is the load-bearing motivation: the concurrent decode is
delayed by ~the full isolated prefill duration, and that delay is attributable to
`llama_decode` batch work, not to HPX orchestration. A policy that bounds prompt
rows per iter is therefore the lever that could bound the per-iter delay seen by
an active decoder — and the lever lives entirely in HPX-owned batch composition,
not in `llama.cpp`.

## 2. Policy idea (HPX-native)

Add a `PrefillBudgetPolicy`: a control-plane object, consulted on the engine task
during the BUILD phase, that decides **how many prompt-token rows may be added to
this iter's batch** (across all in-progress prefills), instead of always adding a
whole prompt at once. A request whose prompt exceeds the per-iter budget is
prefilled across multiple consecutive iters ("chunked prefill"); its decode does
not begin until its prompt is fully prefilled.

This is row composition, not preemption and not parallelism: still one
`llama_batch`, still one `llama_decode` per iter, still one engine task.

## 3. Boundary preservation (hard invariants, unchanged)

The policy must preserve every existing ownership and correctness invariant from
`CLAUDE.md` and the control-plane design:

- **No `llama_decode` change.** The policy only changes which/how many rows the
  engine puts in the batch before the single per-iter `llama_decode`.
- **No `ggml` / tokenizer / sampler / `llama_memory_seq_*` / KV-internals
  change.** Prompt tokens are already produced by the tokenizer up front; the
  policy only meters their entry into the batch.
- **Engine task remains the sole mutator of `llama_context`** and the sole caller
  of `llama_decode` / `llama_get_logits_ith` / `llama_memory_seq_*`.
- **Exactly one `llama_decode` boundary per iter.** Decisions are
  iteration-boundary only; no in-flight decode is preempted.
- **Correctness invariant (corrected — not whole-prompt hash equality).**
  Chunked prefill is a deliberate batch-shape change, and `CLAUDE.md` is explicit
  that batch shape can change floating-point behavior. Therefore the invariant is
  **not** "chunked prefill reproduces the whole-prompt hash." It is:
  - **Null/unbounded policy** (`max_prefill_rows_per_iter = ∞`) must preserve the
    current canonical anchors *exactly* (`0x0619d4d1900c2365` budget-8,
    `0x833045f1e2ebf49f` budget-16) — there is no batch-shape change in this
    configuration.
  - **Chunked config** correctness is **within-shape determinism**: the same prompt
    at the same chunk budget reproduces the same generated-token sequence and hash
    across repeats.
  - No residual KV for any completed or cancelled sequence.
  - Every `llama_decode` returns 0.
  - The generated-token sequence / hash under a chunked budget **may legitimately
    differ from the whole-prompt result** because re-starting `llama_decode` at
    chunk boundaries changes internal ubatch partitioning and K/V batching. Such a
    difference is **not, by itself, a failure** — only a within-shape
    non-determinism, a residual-KV leak, or a non-zero `llama_decode` is.
  - Separately, a co-resident decoding sequence's tokens must **not** change when a
    chunked prefill shares its iters (attention is per-seq masked); that *is* a
    hard correctness property (see §8).

## 4. Per-request state needed for chunked prefill

The current `seq_state` (`tools/hpx-continuous-batch-gate/types.h`) already
carries the fields chunked prefill builds on:

- `prompt_tokens` — the full tokenized prompt (already moved into the slot at
  `admit_one`);
- `pos_next` — next absolute KV position for this seq;
- `n_decoded` — decode count (gates completion);
- `i_batch` — this seq's logits row index for the current iter;
- `last_token`, `hash_state`, `generated_tokens` — decode-side state.

A chunked-prefill candidate would add (design only — not edited here):

- `prefill_cursor` (int): index into `prompt_tokens` of the next prompt row not
  yet submitted to a batch. `0` at admission; advances by the rows actually placed
  each iter.
- `prefill_complete` (bool): `true` once `prefill_cursor == prompt_tokens.size()`.
  Equivalent to `prefill_cursor == size`, but an explicit flag keeps the BUILD/
  sampling branch cheap and readable.
### 4.1 BUILD-branch re-keying (the single most error-prone future edit)

Today the prefill-vs-decode branch in `iter_build_batch` is keyed effectively on
`admitted_at_iter == iter && n_decoded == 0` (engine.cpp:2221). This is correct
**only because prefill never spans more than one iter today**. Under chunking, a
second chunk runs on an iter where `admitted_at_iter != iter`, so the sequence
would fall into the **decode** branch (engine.cpp:2250) and emit a single
`last_token` decode row — i.e. it would be mistaken for a decoder mid-prompt.

Chunked prefill **must re-key this branch** on a prefill-incomplete predicate:

```
if (prefill_cursor < prompt_tokens.size())   // equivalently: !prefill_complete
    -> emit up to (remaining budget) prompt rows for this seq
else
    -> emit one decode row (existing path)
```

The `admitted_at_iter == iter` test and the matching trace predicate at
engine.cpp:2381 must both be re-pointed off "admitted this iter" and onto the
prefill-cursor/`prefill_complete` state. This is the highest-risk edit in the
whole policy; call it out in any implementation plan.

### 4.2 Sample-loop skip (a real behavior change, not a passive ignore)

`iter_sample_and_finalize` (engine.cpp:2349) currently samples **every** sequence
in `active_idx` unconditionally and, per sequence, reads logits via `seq.i_batch`,
publishes, folds the hash, and bumps `n_decoded`. There is no skip path.

A mid-prefill sequence must remain in `active_idx` (so its chunk rows are decoded
into KV) **but must be skipped entirely by sampling/finalization/publish** until
`prefill_complete`. Explicitly:

> No token may be sampled, hashed (`hash_state`), streamed (`publish_token`), or
> counted in `n_decoded` before the full prompt prefill is complete.

Concretely the sample loop needs a guard `if (!seq.prefill_complete) continue;`,
and the `admitted_prefilled` trace (engine.cpp:2381–2384) must fire on the final
chunk's first sample rather than on "admitted this iter."

### 4.3 Exact position and logits rules

- **Position:** a chunk row's position is `prefill_cursor + local_chunk_index`
  (the absolute prompt index). `pos_next` advances per chunk by the number of rows
  placed. (Today engine.cpp:2237 passes `pos = p` where `p` is the whole-prompt
  loop index and happens to equal the absolute index; chunking must use the
  absolute index explicitly.)
- **Intermediate chunks:** set `logits = false` on **all** rows; do not register
  `i_batch`.
- **Final chunk:** only the **final prompt token of the final chunk** may set
  `logits = true` and update `i_batch`; that is the row the existing post-prefill
  argmax samples.

### 4.4 Degenerate (null) case

Today's whole-prompt prefill is the degenerate case `max_prefill_rows_per_iter =
∞`: `prefill_cursor` jumps from 0 to `size` in one iter and `prefill_complete`
becomes true immediately. A policy of "unbounded" must reproduce today's behavior
exactly (and the same canonical anchors) — the natural Slice B null check.

## 5. Candidate policy knobs

All knobs are control-plane configuration consulted in BUILD; none are
`llama.cpp` parameters.

- `max_prefill_rows_per_iter` (int): cap on total prompt rows added to one iter's
  batch. The core knob. `∞`/0 = today's whole-prompt behavior.
- `max_new_prefill_requests_per_iter` (int): cap on how many *distinct* requests
  may begin prefill in one iter. Bounds the co-admission shape W1c observed for
  L64 (two prompts in one iter). Interacts with `AdmissionPolicy` (§7 of the
  control-plane doc), which decides admission count; this knob meters prompt-row
  entry after admission.
- `prefill_first` vs `decode_first` (enum): row ordering when both prefill rows
  and active decode rows compete for batch capacity in one iter. `decode_first`
  favors latency of in-flight decoders; `prefill_first` favors prompt completion.
- `reserve_decode_row_when_active` (bool): if any sequence is mid-decode, reserve
  at least one decode row in the batch so a long prefill can never fully starve an
  active decoder within a single iter. This is the knob most directly motivated by
  the W1c additive interference result.

### 5.1 Knob precedence (resolved to avoid deadlock-prone ambiguity)

Two pairs of knobs overlap and must have a fixed precedence:

- **`reserve_decode_row_when_active` vs `prefill_first`/`decode_first`.**
  Resolution: **`reserve_decode_row_when_active` wins.** When any sequence is
  mid-decode, the engine first places one decode row per active decoder, then
  applies the prefill budget to the *remaining* capacity, ordered by
  `prefill_first`/`decode_first`. So the ordering knob only governs how leftover
  capacity is filled; it can never starve an active decoder of its reserved row.
- **`max_new_prefill_requests_per_iter` vs `AdmissionPolicy` authority.**
  Resolution: **`AdmissionPolicy` is authoritative for *admission* (binding a
  request to a slot); `max_new_prefill_requests_per_iter` only meters how many
  *already-admitted* sequences may *begin emitting prompt rows* in a given iter.**
  To avoid the deadlock where admission binds a request that prefill metering then
  refuses to advance, the rule is: a sequence that has been admitted is always
  allowed to make prefill progress on some later iter (the budget defers, never
  permanently blocks), and `prefill_cursor` only ever moves forward. An admitted
  sequence can wait for prefill capacity but cannot be indefinitely denied it.

The policy emits a structured decision record per iter (rows placed, requests
advanced, budget hit yes/no) for diagnostics — *recorded only*, no new metric is
added by this note.

## 6. Expected benefit (bounded, not a throughput claim)

The intended and **only** claimed benefit: **bound the per-iter disruption a long
prefill imposes on an already-decoding sequence.** With
`max_prefill_rows_per_iter = B`, the worst-case extra wall time an active decoder
waits in any single iter is ~the cost of `B` prefill rows plus its own decode row,
instead of ~the whole prompt. The W1c additive model predicts this directly:
per-iter interference cost ≈ (cost of rows placed) + (one decode), so capping rows
placed caps the per-iter delay.

Explicit non-benefit: this does **not** claim total throughput or total
wall-time improvement. Chunking a long prompt into N iters can increase that
prompt's own latency and adds fixed per-iter overhead N times (see §7).

## 7. Risks

- **More iterations for long prompts.** A 977-row prompt at `B=128` becomes ~8
  prefill iters; each iter pays the fixed per-iter cost (~40 ms decode-step base
  observed in W1c) and the ~100 µs HPX overhead. Net wall time for that prompt
  may rise.
- **Total wall-time tradeoff.** Bounding per-iter interference can lengthen total
  time; the tradeoff is latency-fairness vs aggregate completion, and must be
  measured (Slice E), not assumed.
- **More complex per-seq state machine.** `prefill_cursor` / `prefill_complete`
  add a partial-prefill state between "admitted" and "decoding"; every path that
  inspects a seq (completion, KV cleanup, streaming, cancellation) must handle it.
- **Cancellation during partial prefill.** The KV layer already handles this; the
  risk is state-machine bookkeeping. Required behavior when a request is cancelled
  after some but not all prompt rows are in KV:
  - **KV removal uses the existing whole-sequence remove path** —
    `llama_memory_seq_rm(mem, seq_id, -1, -1)` (engine.cpp:847) removes the
    sequence's KV regardless of how many positions were filled, so partial prefill
    needs no new clear logic.
  - **`n_decoded_at_cancel` is 0** if no token was sampled (the common case during
    prefill, since sampling is skipped until `prefill_complete`).
  - **`generated_tokens` is empty**, and the result resolves as
    `status = cancelled`.
  - **The stream closes `cancelled` with no token events** if cancellation happens
    before the first sample (no `publish_token` ran).
  - **The freed slot must pass the existing reuse / no-residual-KV checks** — the
    N5a seq_id-range free-list predicate handles a partially-prefilled cancel like
    any other (see `[[project_exp14_deferred_admission_slice]]`).
- **Token-correctness vs legitimate FP variation (do not conflate).** Two distinct
  things can change the generated tokens, and only one is a bug:
  - *Bug:* wrong positions (not `prefill_cursor + local_index`), logits on the
    wrong row, or sampling before `prefill_complete`. These corrupt the result and
    must fail their gate.
  - *Not a bug:* a token/hash difference vs the **whole-prompt** run caused purely
    by chunk-boundary ubatch/K-V batching floating-point effects. Per the corrected
    §3/§8 invariant, this is expected and is **not** a failure.
  The validation that separates them is within-shape determinism at a fixed chunk
  budget (§8 gate 2b), not equality to the whole-prompt anchor. The null/unbounded
  config still must reproduce the canonical anchors exactly (§8 gate 2a).
- **Batch capacity and `n_seq_max` interaction.** Per-iter row budget must respect
  `batch_capacity` and the active-seq count; with `n_seq_max=2` a reserved decode
  row plus a prefill chunk must both fit. The policy must degrade safely when
  capacity is tighter than the configured budget.

## 8. Acceptance gates (any future implementation slice must clear these)

1. The existing smoke suite (the 24 smokes per
   `hpx_serving_layer_m0_m8_milestone_summary.md`) still passes unchanged.
2. **(a) Null-config anchor equality:** with the policy unbounded
   (`max_prefill_rows_per_iter = ∞`) the current canonical anchors reproduce
   *exactly* — budget-8 `0x0619d4d1900c2365`, budget-16 `0x833045f1e2ebf49f`, and
   the W1c per-class within-shape probe hashes. This is the strong regression gate.
   **(b) Chunked-config correctness:** within-shape determinism (same prompt at the
   same chunk budget → same generated-token sequence and hash across repeats),
   plus no residual KV and every `llama_decode` returning 0. Equality of the
   chunked hash to the **whole-prompt** hash is **not** a gate — a difference is
   expected from batch-shape floating-point effects and is not a failure.
3. Streaming gates hold (`local/runs/streaming-detokenize/` coverage): the
   completed/cancelled/error close partition is unchanged, and the streamed-token
   sequence is **deterministic within shape** (same chunk budget across repeats).
   The streamed sequence may differ from the whole-prompt run for the same reason
   as 2(b); only within-shape non-determinism is a failure.
4. A W1c-like diagnostic shows the per-iter interference is **bounded** by the
   configured `max_prefill_rows_per_iter` (i.e., the interference cell's
   `prefill_rows_in_iter` never exceeds the budget, and per-iter
   `llama_decode_wall_us` tracks the capped row count).
5. No new HPX overhead dominates `non_decode_us` — it must stay in the W1c
   ~100 µs band; chunking must not move cost from `llama_decode_wall_us` into HPX.
6. **Cancellation during partial prefill is covered by a new smoke before any
   benchmarking** — a request cancelled mid-prefill must leave no residual KV, must
   resolve `status=cancelled` with `n_decoded_at_cancel=0` and empty
   `generated_tokens`, and must not corrupt a later admission into the freed slot
   (see §7).
7. **Co-resident decoder unaffected:** a sequence that is decoding while another
   sequence's chunked prefill shares its iters must produce the **same** tokens it
   would have produced undisturbed (attention is per-seq masked). This is a hard
   user-visible correctness property and a within-shape equality gate.

## 9. Implementation slices (as planned and as built)

The slices below were planned sequentially, each stopping for approval. All are
done; the as-built per-slice evidence is in `prefill_budget_policy_result.md`
§2 and the new smokes registered in
`tools/hpx-continuous-batch-gate/CMakeLists.txt`.

- **A — Revise this design note.** [done] Correctness invariant fixed (no
  whole-prompt hash equality), BUILD-branch re-keying and sample-loop skip made
  explicit, knob precedence resolved, cancellation specified.
- **B — No-behavior prep slice.** [done] `prefill_cursor` / `prefill_complete`
  added to `seq_state` (`types.h`) and a null `PrefillBudgetPolicy`
  (`prefill_budget_rows = 0` ≡ unbounded) consulted in BUILD, with the §4
  invariants encoded as assertions/comments. Gate: byte-identical canonical
  anchors and the same 24 smokes; whole-prompt path unchanged.
- **C — Gated single-request chunked-prefill experiment.** [done]
  `engine_chunked_prefill_smoke`. One sequence, no concurrency, finite `B`,
  behind `engine_options::lib.prefill_budget_rows` (0 = unbounded). Validates
  positions, completion step, `n_decoded`, KV-empty after completion, and
  within-shape determinism at the same chunk budget — never against the
  whole-prompt hash.
- **D — Cancel-during-partial-prefill smoke.** [done]
  `engine_cancel_mid_prefill_smoke`. Per §7 / gate 6, landed before any
  benchmark.
- **E — Concurrent chunked smoke under `n_seq_max=2`.** [done]
  `engine_concurrent_chunked_prefill_smoke`. Chunked prefill of one seq while
  another decodes; per-seq causal masking gate (gate 7) holds.
- **F — W1c policy-enabled comparison.** [done] Engine-only Option A:
  `engine_w1c_policy_smoke` (bound holds `max(prefill_rows_in_iter) ≤ B` across
  64 scenarios). Server-path Option B: `--prefill-budget-rows` CLI flag in
  `tools/hpx-server/hpx-server.cpp`, validated by Exp16
  (`hpx-bench/experiments/16_prefill_budget_policy_server_w1c/`). Observational
  only.

Safe order, as actually executed: **fix design invariant → add no-behavior
state → prove null config unchanged → single-request chunking → cancellation →
concurrent chunking → engine-only policy result → server-surface policy
result.**

## 10. Non-claims

- No `llama-server` comparison.
- No production claim.
- No Llama 3 (or any non-TinyLlama) generalization yet.
- No claim that chunking improves total throughput or total wall time.
- No scheduling implementation is approved by this document. It is a candidate
  design; the decision to build any slice is separate and explicit.

## References

- `local/runs/w1-mixed-short-long-2026-05-26/` (W1a)
- `local/runs/w1b-varied-prefill-2026-05-26/analysis/summary.md` (W1b)
- `local/runs/w1c-staggered-prefill-2026-05-26/analysis/final_report.md` (W1c)
- `docs/hpx/hpx_native_serving_control_plane_design.md` §7, §8, §12
- `docs/hpx/serving_overhead_diagnostics_roadmap.md`
- `tools/hpx-continuous-batch-gate/types.h` (`seq_state`),
  `tools/hpx-continuous-batch-gate/engine.cpp` (BUILD phase / admission step)
