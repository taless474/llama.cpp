# Experiment 16 — results

Server-driven `--prefill-budget-rows` policy, W1c d2-staggered, B in
{0, 32, 64, 128}, one repeat each. Fresh in-tree `B=0` is the numeric baseline;
the 2026-05-26 W1c run is historical context only. Observational — no
throughput, total-latency, production, default-enable, llama-server, Llama 3,
or Exp14/Exp15 claim.

Run date: 2026-05-27. All four `B` runs exited 0. No source tree was modified
for this run; raw artifacts are gitignored under `local/runs`.

## 1. Bound gate — `max(prefill_rows_in_iter) <= B`

For every `B>0` and every prompt class, the maximum prefill rows in any
interference cell respected the budget. **PASS.** `B=0` rows show the natural
unbounded prefill size for reference.

| B | class | n_interference | max_pf | bound | within_bound |
|---:|---|---:|---:|---:|---|
| 0 | L8 | 4 | 11 | 11 | YES |
| 0 | L64 | 4 | 79 | 79 | YES |
| 0 | L256 | 4 | 251 | 251 | YES |
| 0 | L1024 | 4 | 977 | 977 | YES |
| 32 | L8 | 4 | 11 | 32 | YES |
| 32 | L64 | 12 | 32 | 32 | YES |
| 32 | L256 | 20 | 32 | 32 | YES |
| 32 | L1024 | 20 | 32 | 32 | YES |
| 64 | L8 | 4 | 11 | 64 | YES |
| 64 | L64 | 8 | 64 | 64 | YES |
| 64 | L256 | 16 | 64 | 64 | YES |
| 64 | L1024 | 20 | 64 | 64 | YES |
| 128 | L8 | 4 | 11 | 128 | YES |
| 128 | L64 | 4 | 79 | 128 | YES |
| 128 | L256 | 8 | 128 | 128 | YES |
| 128 | L1024 | 20 | 128 | 128 | YES |

Prompt classes shorter than `B` keep their natural `max_pf` (L8 stays 11 at
every `B`; L64 stays 79 at `B=128`) and are admitted in a single iteration —
expected, not a bound violation.

## 2. Headline — L1024 interference cells

The clearest case: a 1024-class probe arriving mid-decode. At `B=0` the probe's
prefill lands in one ~977-row iteration that dominates the decode wall. The
budget splits it into many capped iterations.

| B | max_pf | n_interference | med_dec | p95_dec |
|---:|---:|---:|---:|---:|
| 0 | 977 | 4 | 868.8 ms | 870.9 ms |
| 32 | 32 | 20 | 84.7 ms | 85.1 ms |
| 64 | 64 | 20 | 101.6 ms | 103.3 ms |
| 128 | 128 | 20 | 151.5 ms | 157.7 ms |

(`med_dec` / `p95_dec` are the decode-wall distribution over the L1024
interference cells; values rounded from microseconds in
`analysis/cross_B/interference_compare.tsv`.)

`B=0` L1024 has a single large interference iteration per cycle (4 cycles -> 4
interference iters, `max_pf=977`). Every `B>0` L1024 cell instead has 20 capped
interference iters per the chunked schedule.

## 3. Decoder progress — not starved

In every `(B, class)` cell the anchor decoder emitted at least one token in
every interference iteration: `iters_with_token == n_interference`. The decoder
keeps advancing while the probe's prefill is chunked; the budget redistributes
prefill work without starving decode.

| B | class | n_interference | iters_with_token | tokens_emitted_total |
|---:|---|---:|---:|---:|
| 0 | L8/L64/L256/L1024 | 4 each | 4 each | 8 each |
| 32 | L8 | 4 | 4 | 8 |
| 32 | L64 | 12 | 12 | 16 |
| 32 | L256 | 20 | 20 | 20 |
| 32 | L1024 | 20 | 20 | 20 |
| 64 | L8 | 4 | 4 | 8 |
| 64 | L64 | 8 | 8 | 12 |
| 64 | L256 | 16 | 16 | 20 |
| 64 | L1024 | 20 | 20 | 20 |
| 128 | L8 | 4 | 4 | 8 |
| 128 | L64 | 4 | 4 | 8 |
| 128 | L256 | 8 | 8 | 12 |
| 128 | L1024 | 20 | 20 | 20 |

## 4. Interpretation

`--prefill-budget-rows` changes the **per-iteration distribution** of prefill
work. It splits one large interference iteration into multiple smaller capped
interference iterations. The bound `max_pf <= B` holds in every `B>0` cell, and
in every cell the concurrent decoder keeps making token progress, so the
redistribution does not starve decode.

That is the entire claim. The L1024 decode-wall figures fall as `B` shrinks
because each interference iteration carries less prefill work, but this is a
per-iteration interference-bounding observation, **not** a throughput or
total-latency result.

## 5. Non-claims

- No throughput claim.
- No total-latency improvement claim.
- No production-readiness claim.
- No default-enable recommendation.
- No llama-server comparison.
- No Llama 3 (or other model/quantization) generalization.
- No Exp14 / Exp15 conclusion.
- Numeric baseline is the fresh in-tree `B=0` row; 2026-05-26 W1c is historical
  context only.
- Cross-B token hashes are not gates: chunk shape can change floating-point
  behavior.

## 6. Artifact references

Raw run (gitignored, not copied into the tree):

```text
local/runs/w1c-staggered-prefill-policy-2026-05-27/
  analysis/cross_B/max_pf_table.tsv          bound gate, section 1
  analysis/cross_B/interference_compare.tsv  timing, sections 2-3
  analysis/cross_B/final_report.md           generated cross-B report
  analysis/per_B/B<n>/{per_iter,interference_cells,per_request}.tsv
  analysis/per_B/B<n>/summary.md
  B<n>/d2-staggered/r1/{diag.jsonl, client.jsonl, server.stderr, ...}
```

Historical context only (not a baseline):

```text
local/runs/w1c-staggered-prefill-2026-05-26/
```

## 7. Relationship to previous work

- **W1c baseline (2026-05-26)** found long-prefill interference: a long probe
  prompt arriving mid-decode produces one large prefill iteration that dominates
  the anchor's decode wall.
- **Slice F Option A** proved the budget-policy effect in an engine-only harness
  (`prefill_budget_rows` capping per-iteration prefill rows).
- **Exp16 (this run)** reproduces that bound through the real server path using
  the `--prefill-budget-rows` CLI flag, in the same W1c d2-staggered shape.

## 8. Gate status

```text
1. all four B runs exit 0                                  PASS
2. max_pf <= B for every B>0 / class interference cell     PASS
3. interference cells exist (pf>0, dec>0, active_seq==2)   PASS (B0=16, B32=56, B64=48, B128=36)
4. L1024 B=0 max_pf ~= 977, single large interference iter PASS (977; 4 iters = 1/cycle)
5. L1024 B>0 multiple capped interference iters            PASS (20 each)
6. decoder not starved (iters_with_token == n_interference) PASS
```
