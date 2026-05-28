# PrefillBudgetPolicy — result note

Status: result summary for the PrefillBudgetPolicy evidence arc. Pairs
with `docs/hpx/prefill_budget_policy_design.md`. Observational only —
no production claim, no default-enable recommendation, no llama-server
comparison.

## 1. Conclusion

In the engine-only W1c-style comparison (Slice F Option A),
`engine_options::lib.prefill_budget_rows` (B) bounds per-iteration
prefill/decode interference: with B>0 the maximum `prefill_rows_in_iter`
in every interference cell `(prefill_rows>0 ∧ decode_rows>0 ∧
active_seq_count==2)` is ≤ B across all four probe classes (L8, L64,
L256, L1024), four B values (0, 32, 64, 128), and four cycles per cell,
on the engine-only `llama-hpx-engine-w1c-policy-smoke` harness over
TinyLlama 1.1B F32. The single ~874 ms B=0 L1024 interference iter is
replaced by 28 smaller interference iters of ~90–157 ms median
`llama_decode_wall_us` for B ∈ {32, 64, 128}; total per-cycle prefill
work is conserved across iters, only the per-iter distribution changes.

## 2. Evidence chain

- **W1a** (`local/runs/w1-mixed-short-long-2026-05-26/`) — every request
  used the fixed 6-token anchor prompt. Two negative results: no
  weak-batch-filling pathology, and no measurable HPX service-layer
  overhead. Prefill-present iters were heavier than pure-decode iters,
  but because prefill was fixed at 6 tokens, W1a could not test
  long-prefill interference.

- **W1b** (`local/runs/w1b-varied-prefill-2026-05-26/`) — varied prompt
  length and confirmed larger `prefill_rows_in_iter` increases
  `llama_decode_wall_us`, while HPX `non_decode_us` stayed small
  (~100 µs) and shape-independent. The headline interference cell was
  NOISY (low-pf classes effectively tied; uneven sample counts).

- **W1c** (`local/runs/w1c-staggered-prefill-2026-05-26/`) — deliberate
  120 ms stagger forced the interference shape with a 1.00 stagger-hit
  rate (48/48). Across prefill rows [11, 79, 251, 977]: interference
  `llama_decode_wall_us` INCREASING (slope ≈ 823 µs/row, R² = 0.998) and
  ADDITIVE for all four classes (residual 0.5–2.6%):
  `interference_med ≈ prefill_alone_med + decode_alone_med`. HPX
  `non_decode_us` stayed 95–118 µs — no measurable HPX cross-cost. This
  is the load-bearing motivation for PrefillBudgetPolicy.

- **Slice B** (types.h/engine.cpp/engine.h, no behavior change) — added
  `prefill_cursor` and `prefill_complete` state fields to the per-seq
  record in `types.h`, with engine.cpp/engine.h plumbing. Full
  24-registered smoke matrix passed; canonical b8 anchor held; no
  observable change.

- **Slice C** (`engine_chunked_prefill_smoke.cpp`) — gated
  single-request chunked prefill on the live-admission build path with
  `engine_options::lib.prefill_budget_rows` set. Default B=0 means
  whole-prompt (unchanged). B>0 caps each iter at B prompt rows for a
  given seq. Verified: B=0 and B>=prompt_len collapse to canonical b8;
  B=2 split [2,2,2]; B=4 split [4,2]; no token emitted before prefill
  completed; same-B repeat-deterministic; residual KV clean.

- **Slice D** (`engine_cancel_mid_prefill_smoke.cpp`) — cancellation
  during partial prefill. Used a deterministic release barrier (not
  sleeps) to park the engine mid-prefill, then cancel. Verified:
  `prefill_rows_per_iter=[2]`, `tokens_emitted_per_iter=[0]`,
  `status=cancelled`, `n_decoded=0`, terminal stream close
  `reason=cancelled`, `residual_kv_ok=1`. A follow-up request was
  admitted from `cancel_freed` and completed with b8.

- **Slice E** (`engine_concurrent_chunked_prefill_smoke.cpp`) —
  concurrent decode + chunked prefill (`n_seq_max=2`). Proved the
  real continuous-batching shape: one sequence decoding while another
  chunk-prefills in the same `llama_decode`. Per-seq content isolation
  gate: A=P/B=P → both b8; A=P/B=Q → A invariant (b8), B differs;
  A=Q/B=P → B invariant (b8), A differs. Per-seq causal masking
  preserved under overlap.

- **Slice F Option A** (`engine_w1c_policy_smoke.cpp`,
  `local/runs/slice-f-engine-A-2026-05-27/`) — engine-only W1c-style
  policy comparison. Mirrors the W1c d2-staggered SHAPE without a
  server: anchor "Hello, my name is" decoding while a long-prefill
  probe is admitted, via the existing release/ack barrier at iter K=1.
  64 scenarios across (B, class, cycle); every interference iter had
  `max(prefill_rows_in_iter) ≤ B` for B>0. Anchor canonical b8 held
  across all 64 scenarios; probe hashes for B=0 match the W1c
  isolated-decode references for all four classes.

## 3. Key Slice F numbers — L1024 interference cells

(Engine-only harness, 4 cycles per B, 28 interference iters across
cycles for L1024 with B>0 and 4 for B=0. `max_pf` is the max
`prefill_rows_in_iter` observed in any interference cell.)

| B   | max_pf | med `llama_decode_wall_us` | p95 `llama_decode_wall_us` |
|----:|-------:|---------------------------:|---------------------------:|
| 0   | 977    | ~874 ms                    | ~877 ms                    |
| 32  | 32     | ~90 ms                     | ~93 ms                     |
| 64  | 64     | ~105 ms                    | ~112 ms                    |
| 128 | 128    | ~157 ms                    | ~163 ms                    |

Across **every** (B, class) interference cell with B>0 (16 cells × 4
cycles = 64 scenarios), `max(prefill_rows_in_iter) ≤ B`. Asserted
fail-loudly in the harness AND re-shown by the analyzer
(`local/runs/slice-f-engine-A-2026-05-27/interference_cells.tsv`,
`summary.md`).

## 4. What was proven

- Per-iteration interference can be bounded by an HPX-owned
  scheduling-policy knob (`prefill_budget_rows`) without changing any
  llama.cpp execution path.
- The concurrent decoder continues making progress across smaller
  interference iterations: under B>0 every interference iter emits ≥ 1
  anchor token in the harness; the decoder never sits behind a single
  ~977-row prefill iter.
- HPX `non_decode_us` remains small (50–270 µs) relative to
  `llama_decode_wall_us` (90 ms – 874 ms) in interference cells —
  consistent with W1b/W1c.
- Engine invariants under chunked prefill: `decode_failures==0`,
  `cancelled_count==0`, `residual_kv_ok` across all 64 scenarios.
  Same-shape repeat determinism holds within B (anchor canonical b8 and
  probe hashes identical across the 4 cycles per cell).

## 5. What was NOT proven

- No throughput claim.
- No total-latency improvement claim. Total prefill work is conserved
  across iters; only the per-iter distribution changes.
- No server end-to-end claim. This is an engine-only harness; there is
  no HTTP, SSE, or `llama-hpx-server` path involved.
- No llama-server comparison.
- No default-enable recommendation for PrefillBudgetPolicy.
- No Llama 3 (or any other model) generalization. TinyLlama 1.1B F32
  only.
- No Exp14 / Exp15 conclusion.
- The numeric comparison baseline is the new engine-only B=0 harness
  run, NOT the 2026-05-26 server-driven W1c numbers.

## 6. Server-surface follow-up — done as Exp16

Option B (server-surface exposure of the policy) has since been
landed and validated. See
`hpx-bench/experiments/16_prefill_budget_policy_server_w1c/` for the
server-path W1c d2-staggered confirmation.

- CLI: `tools/hpx-server/hpx-server.cpp` exposes
  `--prefill-budget-rows <int>` and wires it to
  `engine_options::lib.prefill_budget_rows`. Default 0 leaves engine
  behavior unchanged; non-negative is enforced; the resolved value is
  logged at startup.
- W1c d2-staggered driver was re-run through the real
  `llama-hpx-server` at B ∈ {0, 32, 64, 128}, four cycles per
  `(B, class)`. For every `B>0` interference cell,
  `max(prefill_rows_in_iter) ≤ B` (Exp16 §1). The L1024 single
  ~977-row interference iter at B=0 is replaced by 20 capped
  interference iters at B>0 (Exp16 §2); the concurrent decoder
  continues to emit ≥1 token in every interference iter (Exp16 §3).

Exp16 keeps the same boundary as this note: per-iteration interference
bounding only — no throughput, total-latency, default-enable,
llama-server, or Llama 3 claim.

## 7. Artifact index

Run directories:
- `local/runs/w1c-staggered-prefill-2026-05-26/` — original
  server-driven W1c baseline (motivation/context only; not the Slice F
  numeric baseline).
- `local/runs/slice-f-engine-A-2026-05-27/` — Slice F Option A
  engine-only run dir: per-B `diag-B{0,32,64,128}.jsonl`, stdout/stderr
  logs, `per_iter.tsv`, `interference_cells.tsv`, `per_request.tsv`,
  `summary.md`, `analyze.py`.
- `local/runs/w1c-staggered-prefill-policy-2026-05-27/` — Exp16
  server-path Option B run dir: per-B `B{0,32,64,128}/d2-staggered/r1/`
  `{diag.jsonl,client.jsonl,server.stderr}`, `analysis/per_B/`,
  `analysis/cross_B/{max_pf_table.tsv,interference_compare.tsv,
  final_report.md}`.

Design / policy docs:
- `docs/hpx/prefill_budget_policy_design.md` — candidate design.
- `docs/hpx/hpx_native_serving_control_plane_design.md` — the broader
  control-plane note where the policy hook was first named.
- `docs/hpx/serving_overhead_diagnostics_roadmap.md` — Phase-1
  diagnostics that made the per-iter measurements possible.
- `hpx-bench/experiments/16_prefill_budget_policy_server_w1c/` —
  server-path follow-up that reproduces the bound through the real
  `--prefill-budget-rows` CLI flag.

Smoke files (all under `tools/hpx-continuous-batch-gate/`):
- `engine_chunked_prefill_smoke.cpp` — Slice C.
- `engine_cancel_mid_prefill_smoke.cpp` — Slice D.
- `engine_concurrent_chunked_prefill_smoke.cpp` — Slice E.
- `engine_w1c_policy_smoke.cpp` — Slice F Option A.

Build artifact:
- `/Users/unick/Desktop/hpx/builds/llama-hpx-hpx-on/bin/llama-hpx-engine-w1c-policy-smoke`
