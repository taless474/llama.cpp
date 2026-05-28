# Experiment 16 — server-driven `--prefill-budget-rows` policy, W1c d2-staggered

## Question

> Does the `llama-hpx-server --prefill-budget-rows` policy bound per-iteration
> prefill/decode interference in the **server-driven** W1c d2-staggered shape?

Concretely: with a staggered long-prompt probe arriving while an anchor decode
is in flight, does setting a per-iteration prefill row budget `B` cap
`max(prefill_rows_in_iter)` to `<= B` in the interference cells, and how does
that redistribute the per-iteration work?

## Important distinction

This is **not** the engine-only Slice F Option A result. Slice F proved the
policy effect inside the engine harness (`engine_w1c_policy_smoke`), directly
exercising `engine_options::lib.prefill_budget_rows`. **Exp16 is the
server-driven follow-up**: the workload runs through the real
`llama-hpx-server` HTTP path, and the budget is set through the actual CLI flag
`--prefill-budget-rows B`. Same W1c d2-staggered driver shape, real server
control path.

## Experiment shape

```text
path             server-driven (llama-hpx-server HTTP), engine task internal diag
model            tinyllama-1.1b-chat-v1.0.F32.gguf
condition        d2-staggered only
B values         {0, 32, 64, 128}   (B passed via --prefill-budget-rows)
repeats          1 each (r1)
prompt classes   L8, L64, L256, L1024
cycles           K = 4 per prompt class
stagger          timing mode, STAGGER_DELAY_MS = 120
server flags     --n-seq-max 2 --max-concurrent 8
                 --max-prompt-tokens 1280 --ctx-size 4096
                 [--prefill-budget-rows B]   (omitted entirely for B=0)
```

`B=0` means the flag is absent — natural unbounded prefill, the **fresh in-tree
baseline** for this comparison. The 2026-05-26 W1c run is historical context
only, never the numeric baseline.

Sample-count accounting: 1 repeat x K=4 = 4 cycles per `(B, class)`; 16 cycles
per `B` across all four classes. The number of *interference rows* per
`(B, class)` is **not** the cycle count — `B=0` typically has one big
interference iter per cycle, while `B>0` long prompts produce several capped
interference iters per cycle from chunking.

## What is measured

The driver sets `LLAMA_HPX_DIAG_METRICS=1` and writes the engine's
per-iteration diagnostic stream to `diag.jsonl`; client-side records go to
`client.jsonl`. An **interference cell** is an engine iteration with both
`prefill_rows_in_iter > 0` and `decode_rows_in_iter > 0` while
`active_seq_count == 2` — the probe's prefill chunk and the anchor's decode
sharing one `llama_decode`.

Key quantities per `(B, class)`:

- `max_pf` = `max(prefill_rows_in_iter)` over interference cells — the policy
  bound check.
- `med_dec_us` / `p95_dec_us` = decode-wall distribution over interference
  cells.
- decoder-progress = count of interference iters that emitted >= 1 token.

## How this was run (not to be rerun)

The run is complete. Raw artifacts live under `local/runs` (gitignored); they
are **not** copied into the source tree. Recorded numbers and the curated
tables are in `results.md`.

```text
local/runs/w1c-staggered-prefill-policy-2026-05-27/
  driver.py                     copied + extended with --policy-B N
                                (forwards --prefill-budget-rows N to the server)
  cross_B.py                    stdlib-only cross-B analyzer
  B{0,32,64,128}/d2-staggered/r1/{diag.jsonl, client.jsonl, server.stderr, ...}
  analysis/per_B/B<n>/{per_iter,interference_cells,per_request}.tsv + summary.md
  analysis/cross_B/{max_pf_table,interference_compare}.tsv + final_report.md
```

For reproduction the run shape was, per `B`:

```text
python3 driver.py --policy-B <B> --condition d2-staggered \
  --repeat-dir <run-dir>/B<B>/d2-staggered/r1 --port 8090
# then:
python3 cross_B.py
```

## Boundary / non-claims

This experiment makes a **per-iteration interference-bounding** statement only.
It does **not** claim:

- throughput improvement
- total-latency improvement
- "HPX is faster"
- production readiness
- a default-enable recommendation
- any llama-server comparison
- Llama 3 (or other model) generalization
- any Exp14 / Exp15 conclusion

Cross-B generated-token hashes are not gates: chunk shape can change
floating-point behavior, so hashes may legitimately differ across `B`.

## Relationship to previous work

- **W1c baseline** characterized long-prefill interference: a long probe prompt
  arriving mid-decode produces one large prefill iteration that dominates the
  anchor's decode wall.
- **Slice F Option A** proved the budget policy effect in an engine-only
  harness (`prefill_budget_rows` capping per-iteration prefill).
- **Exp16 (this run)** reproduces the same bound through the server path using
  the real `--prefill-budget-rows` CLI flag.

See `facts.md` for stable design-time facts and field definitions; see
`results.md` for the recorded numbers and interpretation.
