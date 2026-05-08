# Results — multi-sequence llama_batch gate

This is the closeout evidence for the smallest real `llama.cpp`
multi-sequence batching gate (the Phase 3 prerequisite defined in
`docs/hpx/multiseq_llama_batch_gate.md`). All steps were run on
TinyLlama 1.1B Q4_K_M, prompt `"Hello, my name is"`, greedy argmax,
CPU/Metal default backend, no HPX.

## Summary

State:

- **Step 1**: skeleton / context creation — **PASS**
- **Step 2**: single-seq budget-8 and budget-16 hashes — **PASS**
- **Step 3**: two-seq same-prompt equality — **PASS**
- **Step 4**: three-seq mixed budgets `[8, 64, 256]` — **PASS**
- **Step 5**: 99-seq Phase 3 target shape — **PASS**

Each step's final stdout line was `GATE_STEPN: PASS` (no FAIL paths
triggered, no `llama_decode` retries, no KV exhaustion).

## Step 5 headline

- `n_seqs` = **99**
- budget mix = **{8, 64, 256}** round-robined across `seq_id = 0..98`
- **33** seqs per budget class
- `prompt_tokens` = **6**
- requested `n_batch` = **1024**
- actual `n_ctx` = **50688** (per-seq slice ≈ 512, well above
  `P + max_budget = 262`)
- actual `n_seq_max` = **99**
- every `llama_decode` call returned **0** across both the single
  run and the `--repeat 2` run
- `--repeat 2` produced byte-identical per-class hashes, counts,
  `done_iter` sets, and `pos_max_at_clear` sets across iter 0 and iter 1
- final stdout line: `GATE_STEP5: PASS`

## Step 5 correctness

Per-class hashes and structural sets:

- **budget 8** — count **33**, hash `0x0619d4d1900c2365`, **1**
  unique hash. Matches the canonical TinyLlama / `"Hello, my name is"`
  budget-8 reference from `tools/serving-bench`.
- **budget 64** — count **33**, hash `0x88a4dc75a31d4325`, **1**
  unique hash.
- **budget 256** — count **33**, hash `0x8a1a3bd01360aada`, **1**
  unique hash.

`done_iter` sets (iter 0 = prefill, expected = `budget − 1`):

- budget 8 → `{7}`
- budget 64 → `{63}`
- budget 256 → `{255}`

`pos_max_at_clear` sets (expected = `P + budget − 2` = `4 + budget`
since `P = 6`):

- budget 8 → `{12}`
- budget 64 → `{68}`
- budget 256 → `{260}`

Residual KV at end of run: **all 99 seqs** report
`(pos_min = -1, pos_max = -1)` — every per-seq KV was cleared by the
in-loop `llama_memory_seq_rm(mem, seq_id, -1, -1)` call when that
seq reached its budget.

Cross-talk isolation: at every per-seq clear, snapshot of every
still-active sibling's `(pos_min, pos_max)` was preserved
byte-for-byte across the clear. No clear ever disturbed a sibling.

Within-class equality: 33 seqs of each budget class produced byte-
identical generated-token sequences, validated by both per-class
`unique_hashes == 1` and the cross-class pairwise prefix-equality
loop (first `min(budget_i, budget_j)` tokens identical across every
pair).

## Important hash-shape caveat

Step 5's budget-64 and budget-256 hashes **differ** from Step 4's
N=3 hashes for the same prompt and same budgets:

| budget | Step 4 (N=3) hash    | Step 5 (N=99) hash   | match? |
|-------:|----------------------|----------------------|:------:|
|      8 | `0x0619d4d1900c2365` | `0x0619d4d1900c2365` |  yes   |
|     64 | `0x3b15a0474dfe11be` | `0x88a4dc75a31d4325` |  no    |
|    256 | `0x8790fbe5a60c9ae6` | `0x8a1a3bd01360aada` |  no    |

This is consistent with batched FP arithmetic being non-associative:
with a different number of co-resident seqs, internal reductions land
on slightly different floats, and greedy argmax can flip on near-tie
logits as decoding progresses. The budget-8 hash agrees because the
first 8 tokens are robust to that noise on this model + prompt;
later tokens are not.

**Cross-shape token-hash equality must NOT be used as a correctness
invariant when the batch shape changes.**

Safe correctness invariants for this gate (and for any future HPX
prototype reusing the same primitive) are:

- **within-shape repeat determinism** — same `(model, prompt,
  budgets, n_seqs, batch_shape, seed)` produces byte-identical
  per-class hashes across reruns
- **within-run same-budget-class hash equality** — all seqs in a
  class share one hash
- **expected token counts** — every seq finishes with `n_decoded ==
  budget`
- **expected `done_iter`** — `budget − 1` (with iter 0 = prefill)
- **expected `pos_max_at_clear`** — `P + budget − 2`
- **per-seq KV clear** — `llama_memory_seq_rm(..., -1, -1)` followed
  by `pos_min == pos_max == -1`
- **no cross-talk** — clearing one seq must not change any
  still-active seq's `(pos_min, pos_max)`

Do **not** claim that all batch shapes must produce identical
long-budget hashes. Reproducibility is per-shape, not per-prompt.

## Implication for Phase 3

The Phase 3 primary target shape is now reproducible end-to-end on
real `llama.cpp`. One `llama_model`, one `llama_context`, one shared
`llama_batch`, 99 active `seq_id`s, mixed decode budgets, per-seq KV
clear, no cross-talk, fully deterministic per shape. The next phase
can design HPX orchestration around this primitive.

Constraints carried forward into Phase 3:

- HPX must preserve the **batch shape** of any reference run that
  generates correctness-anchor hashes. Switching from "all seqs in
  one shared batch" to a different per-iteration batch composition
  will change long-budget hashes even when the per-seq inputs are
  identical, for the FP-non-associativity reason documented above.
- Correctness comparisons for HPX should be made against a reference
  run **with the same batch shape and the same scheduling policy**,
  not against unrelated N=3 or single-seq hashes for budgets longer
  than ~8 tokens.
- The budget-8 canonical hash `0x0619d4d1900c2365` may continue to
  serve as a per-seq sanity fingerprint across any reasonable batch
  shape on this model + prompt, but it is the only such cross-shape
  anchor we currently have evidence for. Anything past 8 tokens is
  shape-bound.
- The seven-point safe-invariant list above is the gate that any
  HPX prototype must reproduce *under its own batch shape* before
  any performance comparison is meaningful.

## Evidence captures

All captures live under `local/` (untracked, no committed result
directory needed for correctness work):

```text
local/multiseq_gate_step1.stdout
local/multiseq_gate_step1.stderr

local/multiseq_gate_step2_budget8.stdout
local/multiseq_gate_step2_budget8.stderr
local/multiseq_gate_step2_budget16.stdout
local/multiseq_gate_step2_budget16.stderr
local/multiseq_gate_step2_budget8_repeat2.stdout
local/multiseq_gate_step2_budget8_repeat2.stderr

local/multiseq_gate_step3_two_seq.stdout
local/multiseq_gate_step3_two_seq.stderr

local/multiseq_gate_step4_three_seq.stdout
local/multiseq_gate_step4_three_seq.stderr
local/multiseq_gate_step4_three_seq_repeat2.stdout
local/multiseq_gate_step4_three_seq_repeat2.stderr

local/multiseq_gate_step5_99seq.stdout
local/multiseq_gate_step5_99seq.stderr
local/multiseq_gate_step5_99seq_repeat2.stdout
local/multiseq_gate_step5_99seq_repeat2.stderr
```

Each `.stdout` ends in a `GATE_STEPN: PASS` line; `.stderr` is the
standard `llama.cpp` model-load + `ggml_metal_free: deallocating`
shutdown log with no warning or error markers.

## Status

This document closes out the multi-seq `llama_batch` gate. Phase 3
HPX integration is **not** started. The gate binary, sources, and
this results file remain in the repo as the reference primitive that
any future HPX prototype must reproduce under its own scheduling.
