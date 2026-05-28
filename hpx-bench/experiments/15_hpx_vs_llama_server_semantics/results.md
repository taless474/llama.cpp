# Experiment 15 — Phase 0 results

Status: **complete**. Both Phase 0 runs PASS, all canonical
hpx-server anchors hold post-N5a/N5b, and the documented hpx vs
llama-server semantic mismatches are reproduced.

## Phase 0 runs

### Non-streaming alignment

- Driver: `hpx-bench/experiments/12_hpx_vs_llama_server_pair/bench.py`
  (Exp12, unchanged) with
  `--config ../15_hpx_vs_llama_server_semantics/config.phase0.nonstreaming.json`
  and
  `--results-root ../15_hpx_vs_llama_server_semantics/results`.
- `run_id`: `20260524-145817-pair`
- Results dir: `results/20260524-145817-pair/`
- Driver capture: `local/runs/exp15/phase0-nonstreaming/driver.{stdout,stderr}`
- Gates: **PASS**. 18 client rows (3 workloads × 3 iters × 2 servers).
  No stuck `llama-server` or `llama-hpx-server` process after the run.

### Streaming alignment

- Driver: same Exp12 `bench.py` with
  `--config ../15_hpx_vs_llama_server_semantics/config.phase0.streaming.json`
  and same `--results-root`.
- `run_id`: `20260524-145834-pair`
- Results dir: `results/20260524-145834-pair/`
- Driver capture: `local/runs/exp15/phase0-streaming/driver.{stdout,stderr}`
- Gates: **PASS**. 20 client rows (1 workload × 10 iters × 2 servers).
  No stuck server process after the run.

## Per-shape hpx-server anchors

All anchors stable across every iteration of the corresponding
shape (per Exp12 `evaluate_gates` hash-stability gate, plus the
explicit Phase 0 anchor check).

| run         | shape   | server     | iters | hash                  | n_decoded | text_normalized_sha256 |
|-------------|---------|------------|------:|:----------------------|----------:|:----------------------:|
| nonstream   | p0_b8   | hpx_server |     3 | `0x0619d4d1900c2365`  | 8         | stable                 |
| nonstream   | p0_b32  | hpx_server |     3 | `0x6794e47fe0f84af1`  | 32        | stable                 |
| nonstream   | p1_b8   | hpx_server |     3 | `0x0000000000000000`  | 0         | stable (empty text)    |
| streaming   | p0_b8   | hpx_server |    10 | `0x0619d4d1900c2365`  | 8         | stable                 |

All three canonical anchors (`p0_b8`, `p0_b32`, `p1_b8` EOG-stop)
match the values recorded in `docs/hpx/provenance.md` §9. N5a and
N5b changes do not alter greedy decode output on this shape, as
expected.

## llama-server outputs (recorded only, not gated)

| run         | shape   | server       | iters | n_decoded | server_status | text_normalized_sha256 | text (first row, trimmed)                  |
|-------------|---------|--------------|------:|----------:|:--------------|:----------------------:|:--------------------------------------------|
| nonstream   | p0_b8   | llama_server |     3 | 8         | `stop`        | stable                 | `John Smith. I am a software engineer` (37 chars; leading space in raw) |
| nonstream   | p0_b32  | llama_server |     3 | 32        | `stop`        | stable                 | `John Smith. I am a software engineer at XYZ Company. I have` (145 chars; leading space in raw) |
| nonstream   | p1_b8   | llama_server |     3 | 1         | `stop`        | stable (empty content) | empty (`stopped_eos=true` upstream) |
| streaming   | p0_b8   | llama_server |    10 | 8         | `limit`       | stable                 | `John Smith. I am a software engineer` (37 chars; leading space in raw) |

llama-server's response does not carry an `hash` field; the hash
column is empty for every llama-server row by adapter design.
`text` rendered above is trimmed for readability; the raw
`content` field begins with a leading space in every llama-server
row (documented SentencePiece convention).

## EOG/EOS observations (`p1_b8` shape)

```text
hpx_server p1_b8:
  status     = "completed"
  n_decoded  = 0
  hash       = 0x0000000000000000
  text       = ""
  ok         = False   (non-empty-text predicate; not an error)

llama_server p1_b8:
  server_status      = "stop"
  raw_json.stopped_eos = true
  tokens_predicted   = 1
  content            = ""
  ok                 = False   (non-empty-text predicate; not an error)
```

Both servers terminate at EOG immediately after prefill on this
prompt. They differ in how they count the terminating step
(`hpx` returns `n_decoded=0` because `engine.cpp`'s EOG branch
finalizes the request without appending the EOG token;
`llama-server` reports `tokens_predicted=1` and `stopped_eos=true`).
This divergence is **recorded**, not failed. It matches the value
already pinned in `docs/hpx/provenance.md` §9.5.

## Streaming reconstruction observations (`p0_b8`)

```text
hpx_server streaming text (per iter):
  "JohnSmith.Iamasoftwareengineer"             (30 chars, no inter-token spaces)

hpx_server non-streaming text (per iter):
  "John Smith. I am a software engineer"        (36 chars, inter-token spaces)

llama_server streaming text (per iter):
  " John Smith. I am a software engineer"       (37 chars, leading space, inter-token spaces)
```

hpx-server's SSE path emits one `event: token` per decoded token,
each carrying a raw single-token SentencePiece detokenization.
Single-token detokenization loses the leading-space glue between
words; the per-token concatenation therefore drops inter-token
spaces. The non-streaming path detokenizes the full sequence at
the end and recovers the spacing.

llama-server's SSE path emits `data:` records produced by its
incremental detokenizer, which preserves spacing across chunks.

Cross-server streaming-text byte equality is **not meaningful and
not gated**. Both servers are internally stable across iterations
(`text_normalized_sha256` is unique per server per shape).

Streaming TTFTs (`first_event_ms` ≈ `first_token_ms`, since both
servers' first SSE record is a content record) sit in the 38–53 ms
range on this hardware. They are recorded as raw monotonic deltas
only and are not aggregated, averaged, or compared across servers.

## What is gated vs recorded

Gated (PASS required, per Exp12 `evaluate_gates` plus Phase 0
anchor expectations):

- Per row: `http_status==200`, `parse_error==""`,
  `0 <= n_decoded <= decode_budget`. Streaming rows additionally
  require `done_seen`, `stream_parse_error==""`,
  `first_event_latency_ms>0`, `total_latency_ms>0`,
  `first_token_latency_ms>0` when `token_event_count>0`.
- Per `(server, workload_id)`: `text_normalized_sha256` stable
  across iterations; `n_decoded` stable across iterations;
  hpx-server `hash` stable across iterations.
- hpx p0_b8 `hash==0x0619d4d1900c2365` AND `n_decoded==8`
  (canonical anchor).
- hpx p0_b32 `hash==0x6794e47fe0f84af1` AND `n_decoded==32`.
- hpx p1_b8 `n_decoded==0` AND `hash==0x0000000000000000` AND
  empty `text` (EOG-stop signature).
- `run_notes.txt` free of forbidden comparative words (`faster`,
  `slower`, `speedup`, `regression`, `wins`, `beats`,
  `outperforms`, `better`, `worse`).

Recorded only (NOT gated, divergence documented):

- llama-server `hash` field (absent by adapter design).
- Cross-server `text_sha256` / `text_normalized_sha256` equality.
- Cross-server `n_decoded` equality (EOG-stop divergence on `p1`).
- llama-server `tokens_evaluated` vs hpx-server `prompt_tokens=-1`.
- Streaming wire-format differences (event-typed vs bare `data:`).
- Streaming detokenization differences (per-token vs incremental).
- TTFTs and first-event latencies (raw, single-machine, single-client).

## Valid claims

- The Phase 0 canonical hpx-server anchors hold post-N5a/N5b. The
  greedy decode path is unaffected by the N5a/N5b lifecycle fixes,
  as expected.
- Within each server, every `(workload, mode)` is stable across
  iterations (text, `n_decoded`, hpx-server `hash`).
- The documented hpx vs llama-server semantic mismatches —
  leading-space convention, missing `prompt_tokens`, divergent
  sampler chains, EOG-stop counting, streaming detokenization —
  are all reproduced exactly as Exp12's M8 evidence pinned them.

## Non-claims

- No performance claim. No throughput, latency, TTFT, or
  responsiveness comparison between the two servers.
- No concurrent-client claim. Single client only.
- No cross-server text-equality gate.
- No cross-server `n_decoded` equality gate.
- No production-server claim. Single machine (Apple M4 Pro,
  darwin), single model (TinyLlama 1.1B Q4_K_M), greedy only.
- Phase 0 does not authorize any Phase 1 timing comparison;
  Phase 1 would need a separate design slice with its own
  scope, guardrails, and forbidden-language audit.

# Experiment 15 — Phase 1 results

Status: **complete**. Both Phase 1 runs PASS. All hpx-server
canonical anchors hold across all 51 iterations per shape.
Phase 1 records side-by-side single-client timing values under
matched settings; no winner/performance language is asserted.

## Phase 1 runs

### Non-streaming timing

- Driver: `hpx-bench/experiments/12_hpx_vs_llama_server_pair/bench.py`
  (Exp12, unchanged) with
  `--config ../15_hpx_vs_llama_server_semantics/config.phase1.nonstreaming.json`
  and
  `--results-root ../15_hpx_vs_llama_server_semantics/results`.
- `run_id`: `20260524-151044-pair`
- Results dir: `results/20260524-151044-pair/`
- Driver capture: `local/runs/exp15/phase1-nonstreaming/driver.{stdout,stderr}`
- Workloads: `p0_b8`, `p0_b32` (p0-only; `p1` / EOG excluded from
  Phase 1 timing).
- 51 iterations per workload × 2 workloads × 2 servers = 204 rows.
- Gates: **PASS**. No stuck `llama-server` or `llama-hpx-server`
  process after the run.

### Streaming timing

- Driver: same Exp12 `bench.py` with
  `--config ../15_hpx_vs_llama_server_semantics/config.phase1.streaming.json`
  and same `--results-root`.
- `run_id`: `20260524-151137-pair`
- Results dir: `results/20260524-151137-pair/`
- Driver capture: `local/runs/exp15/phase1-streaming/driver.{stdout,stderr}`
- Workloads: `p0_b8`, `p0_b32` (p0-only).
- 51 iterations per workload × 2 workloads × 2 servers = 204 rows.
- Gates: **PASS**. No stuck server process after the run.

## Warm-up rule

The Exp12 harness numbers iterations `1..N` (`for i in range(1,
iters_per+1)`), so the first iteration per `(server, workload)`
is `iteration=1`. Phase 1 treats this first iteration as a
warm-up row for timing aggregates.

- Timing tables use `iteration >= 2`, giving `n=50` per
  `(server, workload)`.
- Correctness/stability gates use **all 51 rows** per
  `(server, workload)` (warm-up included).

Both runs report `readiness_ms` and `warmup_status=200` for both
servers; the harness's own pre-iteration warm-up `/completion`
already drives the model through one decode before iteration 1
begins.

## Non-streaming timing — `response_complete_ms`

`response_complete_ms` is the per-iteration `latency_ms` field in
`client_results.jsonl` (the wall-clock duration of the whole
`POST /completion` request from send to body close). Values are
in milliseconds; `n=50` per cell after dropping `iteration=1`.

| workload | server       |  p50 |  p95 |  p99 |
|:---------|:-------------|-----:|-----:|-----:|
| p0_b8    | hpx_server   |  91.0 |  93.5 |  96.0 |
| p0_b8    | llama_server |  97.0 | 101.5 | 103.5 |
| p0_b32   | hpx_server   | 274.0 | 279.0 | 285.1 |
| p0_b32   | llama_server | 298.5 | 314.6 | 343.4 |

Percentiles use linear interpolation on the sorted sample of
size 50.

## Streaming timing

Source: `results/20260524-151137-pair/client_results.jsonl`.
Per-row fields used: `first_event_latency_ms`,
`first_token_latency_ms`, `total_latency_ms`, and
`token_event_times_ms` (for inter-event gaps). `n=50` per cell
after dropping `iteration=1`.

| workload | server       | first_event_ms p50 | first_event_ms p95 | first_token_ms p50 | first_token_ms p95 | total_ms p50 | total_ms p95 | total_ms p99 |
|:---------|:-------------|-------------------:|-------------------:|-------------------:|-------------------:|-------------:|-------------:|-------------:|
| p0_b8    | hpx_server   |  37.0 |  38.0 |  37.0 |  38.0 |  91.0 |  93.5 |  96.5 |
| p0_b8    | llama_server |  40.0 |  41.0 |  40.0 |  41.0 | 101.0 | 104.5 | 105.0 |
| p0_b32   | hpx_server   |  37.0 |  38.0 |  37.0 |  38.0 | 274.0 | 278.6 | 284.1 |
| p0_b32   | llama_server |  40.0 |  41.0 |  40.0 |  41.0 | 299.0 | 306.0 | 313.6 |

`first_event_latency_ms` and `first_token_latency_ms` collapse to
the same value here because every Phase 1 row's first SSE record
is a content record for both servers (no `event: started` /
`event: prelude` prefix is emitted in this configuration).

### Inter-event gaps (`inter_event_gaps_ms`, flattened)

Adjacent differences of `token_event_times_ms` per row,
flattened across the 50 timing rows per `(server, workload)`.
Counts: `p0_b8` → `(8-1) × 50 = 350` gaps per server;
`p0_b32` → `(32-1) × 50 = 1550` gaps per server.

| workload | server       |    n | p50 | p95 |
|:---------|:-------------|-----:|----:|----:|
| p0_b8    | hpx_server   |  350 | 8.0 |  8.0 |
| p0_b8    | llama_server |  350 | 8.0 | 10.0 |
| p0_b32   | hpx_server   | 1550 | 8.0 |  8.0 |
| p0_b32   | llama_server | 1550 | 8.0 |  9.0 |

These are client-observed gaps between consecutive SSE token
records on the wire; they do not isolate model decode time from
network/stream-framing time and they include the harness's
per-line read loop. They are recorded descriptively.

## Derived tokens-per-second (client-side approximate)

Not computed for Phase 1. The harness records raw latency fields
only; any client-side `tokens / total_latency_ms × 1000` figure
would conflate prefill, decode, network, and stream framing and
would not be comparable to the upstream `prompt_eval_count` /
`tokens_per_second` fields that only `llama_server` reports.

## Correctness / stability gates (all 51 rows)

Across both Phase 1 runs:

| run         | shape   | server       | iters | hash                  | n_decoded | text_normalized_sha256                                                |
|:------------|:--------|:-------------|------:|:----------------------|----------:|:----------------------------------------------------------------------|
| nonstream   | p0_b8   | hpx_server   |    51 | `0x0619d4d1900c2365`  | 8         | `4bc48f3540cf5ed22e1485a46878b848af8e8bf6ce72bf9c145fe0e2276fe222`    |
| nonstream   | p0_b8   | llama_server |    51 | (absent)              | 8         | `4bc48f3540cf5ed22e1485a46878b848af8e8bf6ce72bf9c145fe0e2276fe222`    |
| nonstream   | p0_b32  | hpx_server   |    51 | `0x6794e47fe0f84af1`  | 32        | `2714e119e167066851e43feba6ebdfc5685a75a44aa6b71b089580bac7f7baa0`    |
| nonstream   | p0_b32  | llama_server |    51 | (absent)              | 32        | `2714e119e167066851e43feba6ebdfc5685a75a44aa6b71b089580bac7f7baa0`    |
| streaming   | p0_b8   | hpx_server   |    51 | `0x0619d4d1900c2365`  | 8         | `c2cb4b4ddddeab5d4b1148e91efa08cd3c098e8816d5ffeb505914290056899e`    |
| streaming   | p0_b8   | llama_server |    51 | (absent)              | 8         | `4bc48f3540cf5ed22e1485a46878b848af8e8bf6ce72bf9c145fe0e2276fe222`    |
| streaming   | p0_b32  | hpx_server   |    51 | `0x6794e47fe0f84af1`  | 32        | `5f5f2b818a0331c688ba61ebec19e3400853d3c95ed3de03a561c8d27a6e79d8`    |
| streaming   | p0_b32  | llama_server |    51 | (absent)              | 32        | `2714e119e167066851e43feba6ebdfc5685a75a44aa6b71b089580bac7f7baa0`    |

Confirmations across all 51 rows per `(server, workload, mode)`:

- hpx `p0_b8` `hash == 0x0619d4d1900c2365` (matches canonical
  anchor in `docs/hpx/provenance.md` §9), in both non-streaming
  and streaming.
- hpx `p0_b32` `hash == 0x6794e47fe0f84af1` (matches canonical
  anchor in `docs/hpx/provenance.md` §9), in both non-streaming
  and streaming.
- Per-server `text_normalized_sha256` stable across all 51 rows
  for every `(server, workload, mode)` cell (8 cells total). The
  streaming hpx-server text hashes differ from the non-streaming
  ones, as expected: the SSE path detokenizes per token and the
  non-streaming path detokenizes the whole sequence at the end
  (documented in Phase 0).
- `n_decoded` stable across all 51 rows for every cell (8 = b8,
  32 = b32).
- `parse_error == ""` and `stream_parse_error == ""` for every
  row in both runs (`parse_errors: 0`, `stream_parse_errors: 0`).
- No stuck server processes after either run.

## Semantic mismatches carried forward from Phase 0

Phase 1 reproduces every mismatch already recorded in Phase 0
and pinned in `docs/hpx/provenance.md` §9.5 and Exp12's M8
evidence:

- Leading-space convention: llama-server's raw `content` begins
  with a leading space (SentencePiece convention); hpx-server's
  non-streaming `text` is presented without that leading space.
  `text_normalized_sha256` strips leading whitespace before
  hashing, so non-streaming p0 cells agree at the
  `text_normalized_sha256` level on both servers.
- `hash` field present on hpx-server, absent on llama-server (by
  adapter design; the "(absent)" cells above are empty strings).
- `prompt_tokens=-1` on hpx-server vs `tokens_evaluated` on
  llama-server. hpx-server does not surface prompt-token count
  in its `/completion` response shape today.
- Streaming detokenization divergence: hpx-server's SSE path
  emits one `event: token` per decoded token with a raw
  single-token detokenization, which loses inter-token spaces on
  this prompt (`JohnSmith.Iamasoftwareengineer`). llama-server's
  SSE path uses incremental detokenization and preserves spacing
  (` John Smith. I am a software engineer`).
- EOG/`p1` is **excluded** from Phase 1 timing per the design
  slice. Phase 0 already documented the EOG-stop count
  divergence (`hpx` `n_decoded=0`, `llama_server`
  `tokens_predicted=1`) and that divergence is not gated here.

None of these mismatches affect the Phase 1 correctness gates;
they are recorded descriptively.

## Valid claims

- Single-client `response_complete_ms`, `first_event_latency_ms`,
  `first_token_latency_ms`, `total_latency_ms`, and
  `inter_event_gaps_ms` values are recorded under matched Phase 1
  settings (same model, same prompt, same `decode_budget`, same
  context size, same threads, sequential single-client load,
  one server process per run, single machine).
- The hpx-server canonical anchors (`p0_b8` and `p0_b32` hashes)
  remain stable across all 51 iterations in both non-streaming
  and streaming Phase 1 runs.
- Both servers complete the `p0_b8` and `p0_b32` shapes
  consistently across all 51 iterations (`n_decoded` and
  `text_normalized_sha256` stable per server per cell).
- The Phase 0 semantic mismatches reproduce under the Phase 1
  workload matrix at higher iteration count.

## Non-claims

- No production-throughput claim. Phase 1 measures one client
  sending one request at a time.
- No concurrency claim. Phase 1 runs a single client; there is
  no concurrent-client load test in this slice.
- No semantic-equality claim. The streaming detokenization and
  leading-space divergences listed above mean the two servers'
  wire-level output bytes are not equal even on `p0` shapes.
- No EOG timing claim. `p1` / EOG-stop is excluded from Phase 1
  timing by design.
- No generalization beyond this machine (Apple M4 Pro, darwin),
  this model (TinyLlama 1.1B Q4_K_M), this prompt
  (`"Hello, my name is"`), these budgets (8, 32), or this
  scheduling policy. Different shape, different hardware, or
  different model can move every timing value.
- The Phase 1 tables are recorded side-by-side timing values
  only. They do not constitute a winner determination, a
  throughput ranking, or a fixed performance baseline.

# Experiment 15 — Phase 2a-S0 results

Status: **harness-correctness slice complete; recorded as a
semantic finding.** S0 was a tiny smoke run of the new concurrent
driver. Its original "all rows return 200" gate failed under
matched single-slot admission settings, but the failure is a real
**admission-policy divergence** between hpx-server and llama-server,
not a harness bug. S0 is therefore accepted as Phase 2 evidence and
the future Phase 2 plan is reshaped accordingly (see `facts.md`
"Phase 2 gate policy" and `readme.md` "Phase 2 — future work").

## S0 run

- Driver:
  `hpx-bench/experiments/15_hpx_vs_llama_server_semantics/concurrent_bench.py`
  (new; pure stdlib; reuses Exp12 adapters and Exp12 lifecycle
  helpers unmodified).
- Config:
  `hpx-bench/experiments/15_hpx_vs_llama_server_semantics/config.phase2a.smoke.p0_b8.json`.
- `run_id`: `20260524-221428-c-smoke`
- Result dir: `results/20260524-221428-c-smoke/`
- Driver capture: `local/runs/exp15/phase2a-smoke-p0_b8-nonstreaming/driver.{stdout,stderr}`

## Shape

```text
workload          p0_b8 (prompt "Hello, my name is", decode_budget=8)
mode              non-streaming
concurrency_levels [1, 2]
iterations_per_client 6
warm-up rule       iteration == 1 per client is warm-up for any future
                   aggregate; correctness gates use all rows
server order       hpx-server first, then llama-server
server lifecycle   each server booted exactly once per run; both
                   concurrency cells share that boot
server settings    hpx-server: --n-seq-max 1 --max-concurrent 1
                   llama-server: --parallel 1
```

## Result

`S0_GATES = FAIL` under the original "every row http_status == 200"
gate. Driver exit code 1. The failure is concentrated in a single
cell (`hpx_server c=2`), with a single failure mode (HTTP 503), and
is structurally explained by the admission-policy divergence below.
Every other cell passed.

## Per-cell summary

| server       | c | rows | completed | failed | timeouts | batch_wall_clock_ms | client_observed_rps (approx) |
|:-------------|--:|-----:|----------:|-------:|---------:|--------------------:|-----------------------------:|
| hpx_server   | 1 |   6  |     6     |   0    |    0     |        553          | 10.85                        |
| hpx_server   | 2 |  12  |     6     |   6    |    0     |        555          | 10.81                        |
| llama_server | 1 |   6  |     6     |   0    |    0     |        594          | 10.10                        |
| llama_server | 2 |  12  |    12     |   0    |    0     |       1204          |  9.97                        |

`client_observed_rps` is approximate / client-observed / includes
loopback / **not** a server-throughput claim.

## Finding

```text
hpx_server c=1  client_id=0 iters 1..6  → HTTP 200, hash 0x0619d4d1900c2365, n_decoded=8
hpx_server c=2  client_id=0 iters 1..6  → HTTP 200, hash 0x0619d4d1900c2365, n_decoded=8
hpx_server c=2  client_id=1 iters 1..6  → HTTP 503, empty body, n_decoded=-1
                                          (latency_ms <= 1 ms each; immediate rejection)

llama_server c=1  client_id=0  iters 1..6        → HTTP 200, 6/6 completed
llama_server c=2  client_id=0/1 iters 1..6 (x2)  → HTTP 200, 12/12 completed
                                                   batch_wall_clock_ms ≈ 2× c=1
                                                   (consistent with sequential
                                                    admission on a single slot)
```

In the failing cell, hpx-server admitted client_id=0 immediately and
rejected client_id=1's requests with HTTP 503 for every iteration,
back-to-back, without enqueueing them. In the matching llama-server
cell, both clients' requests completed: llama-server held the second
client's requests until the single slot freed, and the cell's
wall-clock roughly doubled.

## Server / process state

- hpx-server: `exit=-15` (SIGTERM-driven shutdown), `pid_still_running=False`.
- llama-server: `exit=0`, `pid_still_running=False`.
- No stuck `llama-server` or `llama-hpx-server` after either cell.
- hpx-server stderr at default verbosity contains the startup line
  `[hpx-server] listening on 127.0.0.1:9087 (n_seq_max=1 …
  max_concurrent=1 n_ctx=2048)` and **no** admission /
  rejection / queue / 503 log lines — the rejection path is silent
  at the current log level in this build.

## Forbidden-word audit

`run_notes.txt` clean. `grep` matches against the audit prose in
`readme.md` / `facts.md` / `results.md` are inside the explicit
forbidden-word list itself, not in claims.

## Interpretation

This is an **admission-policy semantic divergence**, not a harness
failure:

- Under matched "one externally admitted request at a time" settings
  (`hpx --max-concurrent 1`, `llama --parallel 1`), hpx-server
  **rejects fast** (HTTP 503) and llama-server **queues and
  serializes** (HTTP 200 after a wait).
- Phase 0 and Phase 1 were single-client and could not have observed
  this. S0 surfaces it as soon as `c > max_concurrent`.
- The harness gates correctly identified the rejected rows; the
  original Phase 2a gate over-assumed that both servers queue.

S0 is therefore accepted as Phase 2 evidence. Future Phase 2 work is
split into two tracks (see `facts.md` "Phase 2 gate policy"):

1. **Phase 2a — admission / rejection semantics under overload.**
   503 from hpx-server is a recorded, allowed outcome when the
   client count exceeds `--max-concurrent`. Hash / text / `n_decoded`
   gates apply only to HTTP-200 rows. Reported descriptors include
   `completed_count`, `rejected_count`, `503_rate`, and (record-only)
   per-row latencies of completed requests.
2. **Phase 2b — admitted parallelism with matched N.**
   hpx-server `--n-seq-max N --max-concurrent N` vs llama-server
   `--parallel N`. Every row is expected to return HTTP 200. This
   is the correct track for timing completed requests under
   admitted parallelism.

Retry-on-503 is **not** enabled in the harness because it would mask
the very divergence S0 surfaced.

## Valid claims (S0)

- The new concurrent harness boots, runs, shuts down cleanly, and
  produces the documented schema for both servers.
- Under matched `--max-concurrent 1` / `--parallel 1` settings at
  `c = 2`, the two servers' externally observable overload
  responses differ: hpx-server rejects with HTTP 503; llama-server
  queues with HTTP 200.
- For all HTTP-200 rows in the S0 run, the hpx-server canonical
  anchor (`p0_b8` hash `0x0619d4d1900c2365`, `n_decoded == 8`) holds.
- No stuck server processes; the harness's launch / readiness /
  shutdown / forbidden-word audit all work as designed.

## Non-claims (S0)

- S0 does **not** compare throughput between the two servers.
- S0 does **not** rank or evaluate either admission policy; it
  records that the two servers respond differently to overload
  under these settings.
- Timing values from S0 are **record-only**. The successful-request
  populations differ between servers (`hpx_server c=2` completed
  only client_id=0; `llama_server c=2` completed both clients), so
  any cross-server latency comparison from S0 would be comparing
  unequal samples.
- S0 does not generalize beyond this machine (Apple M4 Pro, darwin),
  this model (TinyLlama 1.1B Q4_K_M), this prompt, this budget, or
  these admission settings.
- S0 does not authorize the full Phase 2a / Phase 2b matrix; that
  authorization is gated on the reshaped plan being approved
  separately.

# Experiment 15 — Phase 2b-S0 results

Status: **PASS (Case A — canonical anchor held).** Both servers
admitted both concurrent clients cleanly; every row returned HTTP
200; the hpx-server canonical anchor `0x0619d4d1900c2365` held at
both `c=1` and `c=2`; per-server `text_normalized_sha256` is stable
across every cell; no stuck server processes; forbidden-word audit
clean.

## S0 run

- Driver:
  `hpx-bench/experiments/15_hpx_vs_llama_server_semantics/concurrent_bench.py`
  (extended with optional `server_args` config block; absent →
  falls back to `adapter.build_args(cfg, port)`, preserving
  Phase 2a-S0 byte-for-byte).
- Config:
  `hpx-bench/experiments/15_hpx_vs_llama_server_semantics/config.phase2b.smoke.p0_b8.json`.
- `run_id`: `20260524-222733-c-smoke`
- Result dir: `results/20260524-222733-c-smoke/`
- Driver capture: `local/runs/exp15/phase2b-smoke-p0_b8-nonstreaming/driver.{stdout,stderr}`

## Shape

```text
workload          p0_b8 (prompt "Hello, my name is", decode_budget=8)
mode              non-streaming
concurrency_levels [1, 2]
iterations_per_client 6
server order       hpx-server first, then llama-server
server lifecycle   each server booted exactly once per run; both cells share that boot
server settings    hpx-server: --n-seq-max 2 --max-concurrent 2
                                --max-prompt-tokens 512 --n-threads 2
                                --ctx-size 2048   (no --engine-pool → default OFF)
                   llama-server: --parallel 2 --threads 2 --ctx-size 2048
                                 --no-context-shift --seed 0
```

## Per-cell summary

| server       | c | rows | completed | failed | timeouts | batch_wall_clock_ms | client_observed_rps (approx) |
|:-------------|--:|-----:|----------:|-------:|---------:|--------------------:|-----------------------------:|
| hpx_server   | 1 |   6  |     6     |   0    |    0     |        571          | 10.51                        |
| hpx_server   | 2 |  12  |    12     |   0    |    0     |       1037          | 11.57                        |
| llama_server | 1 |   6  |     6     |   0    |    0     |        586          | 10.24                        |
| llama_server | 2 |  12  |    12     |   0    |    0     |        809          | 14.83                        |

`client_observed_rps` is approximate / client-observed / includes
loopback / **not** a server-throughput claim.

## Gate status

```text
gate: all rows http=200, no parse errors, no failed reasons, n_decoded == 8
gate: hpx_server hash == 0x0619d4d1900c2365 across all hpx rows
gate: hpx_server   c=1 text_normalized_sha256 stable: 4bc48f3540cf5ed2…
gate: hpx_server   c=2 text_normalized_sha256 stable: 4bc48f3540cf5ed2…
gate: llama_server c=1 text_normalized_sha256 stable: 4bc48f3540cf5ed2…
gate: llama_server c=2 text_normalized_sha256 stable: 4bc48f3540cf5ed2…
S0_GATES: PASS
```

`text_normalized_sha256` matches the Phase 0 / Phase 1
non-streaming `p0_b8` value (`4bc48f3540cf5ed22e1485a46878b848af8e8bf6ce72bf9c145fe0e2276fe222`)
on every cell of either server. Both clients in every `c=2` cell
returned the same canonical greedy output.

## c=1 / c=2 completion counts (anchor confirmation)

```text
hpx_server   c=1: 6/6 completed   hash 0x0619d4d1900c2365 on every row
hpx_server   c=2: 12/12 completed hash 0x0619d4d1900c2365 on every row  (Case A)
llama_server c=1: 6/6 completed
llama_server c=2: 12/12 completed
```

Outcome bucketed against the Phase 2b-S0 gate logic: **Case A** —
the canonical hpx-server anchor held under N=2 admitted parallelism
on this prompt/budget/sampling/hardware combination. There is no
Case B "alternate stable hash" finding to record from this run.

## Concurrent-client overlap (descriptive)

`c=2` rows for both servers show both clients submitting at
`submit_monotonic_ms ≈ 0` (the harness's `threading.Barrier` start
fence) and completing within a few milliseconds of each other on
every iteration, confirming the two concurrent clients actually
overlapped in flight rather than serializing.

```text
hpx_server   c=2 iter=1 (cid=0/cid=1):  submit≈0/0      complete≈168/176 ms
hpx_server   c=2 iter=2 (cid=0/cid=1):  submit≈168/176  complete≈340/348 ms
…
llama_server c=2 iter=1 (cid=0/cid=1):  submit≈0/0      complete≈135/135 ms
llama_server c=2 iter=2 (cid=0/cid=1):  submit≈135/135  complete≈268/268 ms
```

These are recorded as descriptive overlap evidence; they are **not**
aggregated as percentiles, **not** compared between servers, and
**not** interpreted as performance claims.

## Server / process state

- hpx-server: `exit=-15` (SIGTERM-driven shutdown), `pid_still_running=False`.
- llama-server: `exit=0`, `pid_still_running=False`.
- No stuck `llama-server` or `llama-hpx-server` after either run.

## Forbidden-word audit

`run_notes.txt` clean. Doc-level matches are confined to the
explicit audit-rule prose in `readme.md` / `facts.md` / `results.md`.

## Server_args fallback verification

Prior to the run, the harness was sanity-checked to confirm that:

- Loading `config.phase2a.smoke.p0_b8.json` (no `server_args`)
  routes through `adapter.build_args(cfg, port)` and produces the
  Phase 2a-S0 argv byte-for-byte (`--n-seq-max 1`,
  `--max-concurrent 1`, `--parallel 1`).
- Loading `config.phase2b.smoke.p0_b8.json` (with `server_args`)
  routes through the new harness builders and produces
  `--n-seq-max 2 --max-concurrent 2` on hpx-server and
  `--parallel 2` on llama-server, with all other flags identical
  to the Phase 2a-S0 argv.

The Phase 2a path remains unchanged.

## Valid claims (Phase 2b-S0)

- The harness extension correctly routes argv via `server_args`
  when present and via the Exp12 adapter when absent.
- Under matched `--n-seq-max 2 / --max-concurrent 2` on hpx-server
  and `--parallel 2` on llama-server, both servers admit two
  concurrent clients and complete every request with HTTP 200.
- For this specific shape (TinyLlama 1.1B Q4_K_M, prompt
  `"Hello, my name is"`, `decode_budget=8`, greedy, M4 Pro,
  darwin, `--n-seq-max 2`), the hpx-server canonical greedy hash
  `0x0619d4d1900c2365` is invariant across the c=1 and c=2 cells.
- Per-server `text_normalized_sha256` matches the Phase 0 / Phase
  1 anchor across every cell, confirming the same canonical greedy
  output reaches both clients in both servers' `c=2` cells.

## Non-claims (Phase 2b-S0)

- Not a production-throughput claim.
- Not a winner/performance comparison. Per-row latency and per-cell
  wall-clock are recorded but not aggregated as percentiles, not
  compared between servers, and not interpreted.
- Not a fairness analysis of either server's scheduler.
- Not a generalization beyond this machine (Apple M4 Pro, darwin),
  this model (TinyLlama 1.1B Q4_K_M), this prompt, this budget,
  greedy sampling, or `N=2` admission. A different shape, hardware,
  or N can move both the timing values and the canonical hash.
- Not the full Phase 2b matrix. S0 is `c ∈ {1, 2}`,
  `iterations_per_client = 6`, `p0_b8` non-streaming only.
- Does not authorize the full Phase 2b matrix (`N ∈ {2, 4}`,
  `c ∈ {1, 2, 4, 8}`, 51 iterations per client, `p0_b8` and
  `p0_b32`, streaming deferred) without a separate design slice
  and approval.

# Experiment 15 — Phase 2b-S1 results (p0_b8 non-streaming timing)

Status: **PASS.** Admitted-parallelism timing run on the
`p0_b8` non-streaming shape with hpx-server `--n-seq-max 2
--max-concurrent 2` and llama-server `--parallel 2`. All gates
green; canonical hpx-server anchor held; side-by-side timing
values recorded descriptively.

## S1 run

- Driver: `concurrent_bench.py` unchanged from Phase 2b-S0.
- Config:
  `hpx-bench/experiments/15_hpx_vs_llama_server_semantics/config.phase2b.p0_b8.json`
  (same shape as Phase 2b-S0 except
  `iterations_per_client = 51`).
- `run_id`: `20260524-223346-c-smoke`
- Result dir: `results/20260524-223346-c-smoke/`
- Driver capture: `local/runs/exp15/phase2b-p0_b8-nonstreaming/driver.{stdout,stderr}`

## Shape

```text
workload                p0_b8 (prompt "Hello, my name is", decode_budget=8)
mode                    non-streaming
concurrency_levels      [1, 2]
iterations_per_client   51
warm-up rule            iteration == 1 per client is warm-up for timing aggregates;
                        correctness/stability gates use all rows.
timing-row counts:
  c=1, per server:      n = 50          (1 client × (51 − 1) iters)
  c=2, per server:      n = 100         (2 clients × (51 − 1) iters each)

server order            hpx-server first, then llama-server
server lifecycle        each server booted once per run; both cells share that boot
server settings         hpx-server   --n-seq-max 2 --max-concurrent 2
                                     --max-prompt-tokens 512 --n-threads 2
                                     --ctx-size 2048 (no --engine-pool → default OFF)
                        llama-server --parallel 2 --threads 2 --ctx-size 2048
                                     --no-context-shift --seed 0
```

## Per-cell completion summary

| server       | c | rows | completed | failed | timeouts | batch_wall_clock_ms | client_observed_rps (approx) |
|:-------------|--:|-----:|----------:|-------:|---------:|--------------------:|-----------------------------:|
| hpx_server   | 1 |   51 |    51     |   0    |    0     |        4726         | 10.79                        |
| hpx_server   | 2 |  102 |   102     |   0    |    0     |        8810         | 11.58                        |
| llama_server | 1 |   51 |    51     |   0    |    0     |        5029         | 10.14                        |
| llama_server | 2 |  102 |   102     |   0    |    0     |        8846         | 11.53                        |

`client_observed_rps` is approximate / client-observed / includes
loopback / **not** a server-throughput claim.

## Gate status

```text
gate: all rows http=200, no parse errors, no failed reasons, n_decoded == 8
gate: hpx_server hash == 0x0619d4d1900c2365 across all hpx rows
gate: hpx_server   c=1 text_normalized_sha256 stable: 4bc48f3540cf5ed2…
gate: hpx_server   c=2 text_normalized_sha256 stable: 4bc48f3540cf5ed2…
gate: llama_server c=1 text_normalized_sha256 stable: 4bc48f3540cf5ed2…
gate: llama_server c=2 text_normalized_sha256 stable: 4bc48f3540cf5ed2…
S0_GATES: PASS
```

- All 306 rows returned HTTP 200 with `parse_error == ""` and
  `failed_reason == ""`.
- `n_decoded == 8` on every row.
- hpx-server `hash == 0x0619d4d1900c2365` on every hpx row (51 at
  `c=1`; 102 at `c=2`). The Phase 1 / Phase 2b-S0 canonical anchor
  held under N=2 admitted parallelism at the 153-row sample size.
- Per-`(server, concurrency)` `text_normalized_sha256` matches the
  Phase 0 / Phase 1 anchor (`4bc48f3540cf5ed2…`) on all four cells.
- No timeouts. No stuck server process.
- `run_notes.txt` forbidden-word audit clean.

## HPX canonical anchor confirmation

```text
hpx_server c=1: 51/51 rows  hash 0x0619d4d1900c2365
hpx_server c=2: 102/102 rows hash 0x0619d4d1900c2365
```

The N=1 canonical anchor remains the same-shape anchor under
`--n-seq-max 2 --max-concurrent 2` for this prompt / budget /
sampling / hardware combination across the larger S1 sample.

## Timing table — `latency_ms` (per-request wall-clock)

Computed over `iteration >= 2` rows per client (warm-up dropped).
Percentiles use linear interpolation on the sorted sample.

| server       | c |   n |   p50 |   p95 |   p99 |
|:-------------|--:|----:|------:|------:|------:|
| hpx_server   | 1 |  50 |  91.0 |  97.1 | 106.1 |
| hpx_server   | 2 | 100 | 172.0 | 173.0 | 185.0 |
| llama_server | 1 |  50 |  98.5 | 103.0 | 103.0 |
| llama_server | 2 | 100 | 182.0 | 187.3 | 217.0 |

These are side-by-side timing values recorded under matched
Phase 2b-S1 settings. They are not aggregated cross-server, not
interpreted as a winner determination, and not used as a
performance baseline.

Descriptive observations only (no winner language):

- At `c=1` on this shape, observed hpx-server p50 is lower than
  observed llama-server p50; recorded p50 latency differs by
  approximately 7–8 ms. The two values sit within roughly 10 % of
  each other and are well inside one decode-step of variation on
  this prompt.
- At `c=2` on this shape, observed hpx-server p50 is lower than
  observed llama-server p50; recorded p50 latency differs by
  approximately 10 ms. The pattern within each server is that p50
  at `c=2` is roughly twice p50 at `c=1`. This is consistent with
  two clients sharing the same underlying decode pipeline and is
  recorded descriptively, not interpreted.
- `batch_wall_clock_ms` is recorded within roughly 5 % across the
  two servers at both `c=1` and `c=2`.
- `client_observed_rps` is within roughly 1 unit between the two
  servers at `c=2`. It is recorded as approximate / client-observed
  / includes loopback and is **not** a server-throughput claim.

## Server / process state

- hpx-server: `exit=-15` (SIGTERM-driven shutdown), `pid_still_running=False`.
- llama-server: `exit=0`, `pid_still_running=False`.
- No stuck `llama-server` or `llama-hpx-server` after either run.

## Forbidden-word audit

`run_notes.txt` clean. Document hits in `readme.md` / `facts.md` /
`results.md` are confined to the explicit audit-rule prose.

## Valid claims (Phase 2b-S1)

- Single-machine, matched-N=2-admission `latency_ms` p50 / p95 /
  p99 values are recorded for hpx-server and llama-server on the
  `p0_b8` non-streaming shape at `c ∈ {1, 2}` over `n=50` / `n=100`
  non-warm-up samples per cell.
- The hpx-server canonical anchor `0x0619d4d1900c2365` is stable
  across 153 hpx rows (`c=1` and `c=2` combined) at
  `--n-seq-max 2 --max-concurrent 2` for this shape.
- Both servers complete every iteration of the
  `(workload, concurrency)` matrix with HTTP 200, stable
  `n_decoded`, and stable per-cell `text_normalized_sha256`.

## Non-claims (Phase 2b-S1)

- No production-throughput claim.
- No winner / performance comparison. Latency tables are
  side-by-side values, not a ranking.
- No claim that either server's admission policy is preferable.
- No semantic-equality claim — every Phase 0 mismatch
  (leading space, missing `prompt_tokens`, streaming
  detokenization, EOG counting) still applies; Phase 2b-S1 does
  not exercise streaming or `p1`.
- No scheduler-fairness claim. Phase 2b-S1 does not measure
  cross-client tail latency or starvation.
- No generalization beyond this machine (Apple M4 Pro, darwin),
  this model (TinyLlama 1.1B Q4_K_M), this prompt, this budget
  (8), greedy sampling, or `N=2` admission. A different shape,
  hardware, model, or N can move every value in the timing
  tables.
- Not the full Phase 2b matrix. S1 is `c ∈ {1, 2}`, `p0_b8`
  non-streaming, `N=2` only. `c=4` / `c=8`, `N=4`, `p0_b32`, and
  streaming are deferred to separate design slices.

# Experiment 15 — Phase 2b-S2 results (p0_b32 non-streaming timing)

Status: **PASS.** Admitted-parallelism timing run on the
`p0_b32` non-streaming shape under the same Phase 2b settings as
S1. All gates green; canonical hpx-server `p0_b32` anchor held;
side-by-side timing values recorded descriptively.

## S2 run

- Driver: `concurrent_bench.py` unchanged from Phase 2b-S0 / S1.
- Config:
  `hpx-bench/experiments/15_hpx_vs_llama_server_semantics/config.phase2b.p0_b32.json`
  (same shape as the S1 config except `workload.workload_id =
  "p0_b32"` and `workload.decode_budget = 32`).
- `run_id`: `20260524-223815-c-smoke`
- Result dir: `results/20260524-223815-c-smoke/`
- Driver capture: `local/runs/exp15/phase2b-p0_b32-nonstreaming/driver.{stdout,stderr}`

## Shape

```text
workload                p0_b32 (prompt "Hello, my name is", decode_budget=32)
mode                    non-streaming
concurrency_levels      [1, 2]
iterations_per_client   51
warm-up rule            iteration == 1 per client is warm-up for timing aggregates;
                        correctness/stability gates use all rows.
timing-row counts:
  c=1, per server:      n = 50          (1 client × (51 − 1) iters)
  c=2, per server:      n = 100         (2 clients × (51 − 1) iters each)

server order            hpx-server first, then llama-server
server lifecycle        each server booted once per run; both cells share that boot
server settings         hpx-server   --n-seq-max 2 --max-concurrent 2
                                     --max-prompt-tokens 512 --n-threads 2
                                     --ctx-size 2048 (no --engine-pool → default OFF)
                        llama-server --parallel 2 --threads 2 --ctx-size 2048
                                     --no-context-shift --seed 0
```

## Per-cell completion summary

| server       | c | rows | completed | failed | timeouts | batch_wall_clock_ms | client_observed_rps (approx) |
|:-------------|--:|-----:|----------:|-------:|---------:|--------------------:|-----------------------------:|
| hpx_server   | 1 |   51 |    51     |   0    |    0     |       14110         | 3.61                         |
| hpx_server   | 2 |  102 |   102     |   0    |    0     |       24830         | 4.11                         |
| llama_server | 1 |   51 |    51     |   0    |    0     |       15150         | 3.37                         |
| llama_server | 2 |  102 |   102     |   0    |    0     |       23467         | 4.35                         |

`client_observed_rps` is approximate / client-observed / includes
loopback / **not** a server-throughput claim.

## Gate status

```text
gate: all rows http=200, no parse errors, no failed reasons, n_decoded == 32
gate: hpx_server hash == 0x6794e47fe0f84af1 across all hpx rows
gate: hpx_server   c=1 text_normalized_sha256 stable: 2714e119e1670668…
gate: hpx_server   c=2 text_normalized_sha256 stable: 2714e119e1670668…
gate: llama_server c=1 text_normalized_sha256 stable: 2714e119e1670668…
gate: llama_server c=2 text_normalized_sha256 stable: 2714e119e1670668…
S0_GATES: PASS
```

- All 306 rows returned HTTP 200 with `parse_error == ""` and
  `failed_reason == ""`.
- `n_decoded == 32` on every row.
- hpx-server `hash == 0x6794e47fe0f84af1` on every hpx row (51 at
  `c=1`; 102 at `c=2`). The Phase 1 canonical `p0_b32` anchor held
  under N=2 admitted parallelism at this sample size.
- Per-`(server, concurrency)` `text_normalized_sha256` matches the
  Phase 1 anchor (`2714e119e167066851e43feba6ebdfc5685a75a44aa6b71b089580bac7f7baa0`)
  on every cell of either server.
- No timeouts. No stuck server process.
- `run_notes.txt` forbidden-word audit clean.

## HPX canonical anchor confirmation

```text
hpx_server c=1: 51/51 rows  hash 0x6794e47fe0f84af1
hpx_server c=2: 102/102 rows hash 0x6794e47fe0f84af1
```

The Phase 1 N=1 canonical anchor for `p0_b32` remains the
same-shape anchor under `--n-seq-max 2 --max-concurrent 2` for
this prompt / budget / sampling / hardware combination across the
S2 sample.

## Timing table — `latency_ms` (per-request wall-clock)

Computed over `iteration >= 2` rows per client (warm-up dropped).
Percentiles use linear interpolation on the sorted sample.

| server       | c |   n |   p50 |   p95 |   p99 |
|:-------------|--:|----:|------:|------:|------:|
| hpx_server   | 1 |  50 | 273.0 | 289.6 | 298.6 |
| hpx_server   | 2 | 100 | 484.0 | 489.1 | 502.0 |
| llama_server | 1 |  50 | 297.5 | 308.6 | 317.1 |
| llama_server | 2 | 100 | 459.0 | 466.0 | 467.0 |

These are side-by-side timing values recorded under matched
Phase 2b-S2 settings. They are not aggregated cross-server, not
interpreted as a winner determination, and not used as a
performance baseline.

## Descriptive observations (S2 alone)

- At `c=1` on `p0_b32`, observed hpx-server p50 is lower than
  observed llama-server p50; recorded p50 latency differs by
  approximately 24 ms. Recorded p95 difference is approximately
  19 ms; recorded p99 difference is approximately 19 ms.
- At `c=2` on `p0_b32`, observed hpx-server p50 is higher than
  observed llama-server p50; recorded p50 latency differs by
  approximately 25 ms. Recorded p95 difference is approximately
  23 ms; recorded p99 difference is approximately 35 ms.
- Recorded p99 − p50 spread is tighter on llama-server than on
  hpx-server at both `c=1` and `c=2` in this run
  (llama: 19.6 / 8.0 ms vs hpx: 25.6 / 18.0 ms).
- `batch_wall_clock_ms` is recorded within roughly 7 % across the
  two servers at `c=1` and within roughly 6 % at `c=2`.

## Descriptive observations vs Phase 2b-S1 (`p0_b8`)

Recorded p50 ratios `p0_b32 / p0_b8` per cell (same server, same
concurrency):

| server       | c | `p0_b8` p50 (ms) | `p0_b32` p50 (ms) | ratio |
|:-------------|--:|-----------------:|------------------:|------:|
| hpx_server   | 1 |  91.0            | 273.0             |  3.00 |
| hpx_server   | 2 | 172.0            | 484.0             |  2.81 |
| llama_server | 1 |  98.5            | 297.5             |  3.02 |
| llama_server | 2 | 182.0            | 459.0             |  2.52 |

Recorded p50 ratios `c=2 / c=1` per cell (same server, same shape):

| server       | shape   | `c=1` p50 (ms) | `c=2` p50 (ms) | ratio |
|:-------------|:--------|---------------:|---------------:|------:|
| hpx_server   | p0_b8   |  91.0          | 172.0          |  1.89 |
| hpx_server   | p0_b32  | 273.0          | 484.0          |  1.77 |
| llama_server | p0_b8   |  98.5          | 182.0          |  1.85 |
| llama_server | p0_b32  | 297.5          | 459.0          |  1.54 |

Recorded only, neutral. Two patterns are noted:

- The `p0_b32 / p0_b8` p50 ratio is approximately 3× on this
  hardware/model at `c=1` for both servers, which is recorded
  consistent with `p0_b32` performing 4× the decode steps of
  `p0_b8` with a fixed per-request prefill / network overhead
  amortized across more decode steps.
- The `c=2 / c=1` p50 ratio is closer to 2× on `p0_b8` and lower
  than 2× on `p0_b32`. This pattern is recorded; it is not
  interpreted as a server property and is not compared between
  servers as a ranking.

These cross-shape and cross-concurrency descriptors are recorded
only. They do not constitute a winner determination, a throughput
ranking, or a fixed performance baseline.

## Server / process state

- hpx-server: `exit=-15` (SIGTERM-driven shutdown), `pid_still_running=False`.
- llama-server: `exit=0`, `pid_still_running=False`.
- No stuck `llama-server` or `llama-hpx-server` after either run.

## Forbidden-word audit

`run_notes.txt` clean. Document hits in `readme.md` / `facts.md` /
`results.md` are confined to the explicit audit-rule prose.

## Valid claims (Phase 2b-S2)

- Single-machine, matched-N=2-admission `latency_ms` p50 / p95 /
  p99 values are recorded for hpx-server and llama-server on the
  `p0_b32` non-streaming shape at `c ∈ {1, 2}` over `n=50` /
  `n=100` non-warm-up samples per cell.
- The hpx-server canonical anchor `0x6794e47fe0f84af1` is stable
  across 153 hpx rows (`c=1` and `c=2` combined) at
  `--n-seq-max 2 --max-concurrent 2` for this shape.
- Both servers complete every iteration of the
  `(workload, concurrency)` matrix with HTTP 200, stable
  `n_decoded`, and stable per-cell `text_normalized_sha256`.

## Non-claims (Phase 2b-S2)

- No production-throughput claim.
- No winner / performance comparison. Latency tables and the
  cross-shape / cross-concurrency ratio tables are side-by-side
  values, not a ranking.
- No claim that either server's admission policy is preferable,
  and no claim that either server scales preferentially with
  concurrency or with decode budget.
- No semantic-equality claim — every Phase 0 mismatch still
  applies; Phase 2b-S2 does not exercise streaming or `p1`.
- No scheduler-fairness claim. Phase 2b-S2 does not measure
  cross-client tail latency or starvation.
- No generalization beyond this machine (Apple M4 Pro, darwin),
  this model (TinyLlama 1.1B Q4_K_M), this prompt, these budgets
  (8 cross-referenced, 32 measured), greedy sampling, or `N=2`
  admission. A different shape, hardware, model, or N can move
  every value in the timing tables.
- Not the full Phase 2b matrix. S2 is `c ∈ {1, 2}`, `p0_b32`
  non-streaming, `N=2` only. `c=4` / `c=8`, `N=4`, and streaming
  are deferred to separate design slices.

## Phase 2b-S1/S2 c=2 timing-shape analysis

Read-only analysis of the existing Phase 2b-S1 (`p0_b8`) and
Phase 2b-S2 (`p0_b32`) result JSONL files, before adding new
shapes (streaming, higher concurrency). No new runs. No harness
changes. Goal: characterize the `c=2` timing pattern at the
client side and explain why the `c=2 / c=1` ratio differs between
`p0_b8` and `p0_b32`. All client-side observations only; no
scheduler-internal claim.

Inputs:

- `p0_b8`:  `results/20260524-223346-c-smoke/client_results.jsonl`
- `p0_b32`: `results/20260524-223815-c-smoke/client_results.jsonl`

### Hash/text stability re-check

Recomputed across all rows of all four cells in both runs:

| run    | server       | c | rows | hash                  | n_decoded | text_normalized_sha256 |
|:-------|:-------------|--:|-----:|:----------------------|----------:|:----------------------:|
| p0_b8  | hpx_server   | 1 |   51 | `0x0619d4d1900c2365`  | 8         | stable (Phase 1 anchor) |
| p0_b8  | hpx_server   | 2 |  102 | `0x0619d4d1900c2365`  | 8         | stable                  |
| p0_b8  | llama_server | 1 |   51 | (absent)              | 8         | stable (Phase 1 anchor) |
| p0_b8  | llama_server | 2 |  102 | (absent)              | 8         | stable                  |
| p0_b32 | hpx_server   | 1 |   51 | `0x6794e47fe0f84af1`  | 32        | stable (Phase 1 anchor) |
| p0_b32 | hpx_server   | 2 |  102 | `0x6794e47fe0f84af1`  | 32        | stable                  |
| p0_b32 | llama_server | 1 |   51 | (absent)              | 32        | stable (Phase 1 anchor) |
| p0_b32 | llama_server | 2 |  102 | (absent)              | 32        | stable                  |

No correctness drift observed during this analysis.

### Per-client c=2 latency tables (iter ≥ 2, n=50 per client)

p0_b8:

| server       | client_id | n  |   p50 |   p95 |
|:-------------|----------:|---:|------:|------:|
| hpx_server   |     0     | 50 | 172.0 | 172.6 |
| hpx_server   |     1     | 50 | 172.0 | 173.0 |
| llama_server |     0     | 50 | 182.0 | 187.0 |
| llama_server |     1     | 50 | 182.5 | 190.3 |

p0_b32:

| server       | client_id | n  |   p50 |   p95 |
|:-------------|----------:|---:|------:|------:|
| hpx_server   |     0     | 50 | 484.0 | 491.3 |
| hpx_server   |     1     | 50 | 484.0 | 488.6 |
| llama_server |     0     | 50 | 459.0 | 466.0 |
| llama_server |     1     | 50 | 459.0 | 466.0 |

Both client IDs record near-identical p50 in every cell.
Inter-client p50 difference is at most 0.5 ms across all four
cells, which is within single-millisecond noise on this hardware.
No systematic per-client asymmetry observed.

### Completion ordering at c=2 (first iterations)

Submission times relative to cell start; sorted by
`submit_monotonic_ms` (rows sampled iter 1–6 per client).

p0_b8 hpx_server c=2:

```text
iter   cid sub   cmp   lat
  1    0   0    165   165   (warm-up)
  1    1   0    173   173   (warm-up)
  2    0   165  338   172
  2    1   173  345   172
  3    0   338  510   172
  3    1   345  518   172
  4    0   510  682   171
  4    1   518  690   172
  5    0   682  855   172
  5    1   690  862   172
  6    0   855 1027   172
  6    1   862 1035   172
```

Pattern: both clients submit at the barrier; `cid=1` completes
roughly 7–8 ms after `cid=0` on every iteration; per-iteration
latency is ≈ 172 ms and very stable. Iterations are paired.

p0_b8 llama_server c=2:

```text
iter   cid sub   cmp   lat
  1    0   0    133   133   (warm-up)
  1    1   0    134   134   (warm-up)
  2    0   134  269   135
  2    1   134  269   134
  3    0   269  396   127
  3    1   269  396   127
  4    0   396  527   131
  4    1   396  528   131
  5    0   528  657   129
  5    1   528  657   129
  6    0   657  790   132
  6    1   657  790   132
```

Pattern: both clients submit at the barrier; completions land
within ≤ 1 ms of each other on every iteration in the first
several iterations; per-iteration latency is ≈ 127–135 ms early
in the cell. The cell-wide `p50` for `iter ≥ 2` is 182 ms
(reported in the Phase 2b-S1 timing table), so per-iteration
latency drifts upward over the cell's 51 iterations. The
client-side data does not isolate the source of this drift.

p0_b32 hpx_server c=2:

```text
iter   cid sub   cmp   lat
  1    0   0    544   544   (warm-up)
  1    1   0    555   555   (warm-up)
  2    0   544 1040   495
  2    1   555 1048   492
  3    0   1040 1525  484
  3    1   1048 1533  485
  4    0   1525 2010  485
  4    1   1533 2018  484
  5    0   2010 2494  483
  5    1   2018 2501  483
```

Pattern: paired; per-iteration latency ≈ 484 ms; `cid=1`
completes 6–11 ms after `cid=0` on every iteration.

p0_b32 llama_server c=2:

```text
iter   cid sub   cmp   lat
  1    0   0    458   458   (warm-up)
  1    1   0    458   458   (warm-up)
  2    0   458  920   462
  2    1   458  920   462
  3    1   920 1381   460
  3    0   921 1382   461
  4    1   1381 1849  467
  4    0   1382 1849  466
  5    0   1849 2309  459
  5    1   1849 2308  459
```

Pattern: paired; completions land within ≤ 1 ms of each other on
every iteration across the entire cell; per-iteration latency
≈ 458–467 ms. No transient drift visible (unlike p0_b8 llama
c=2).

### Tail rows — top 5 latencies per cell (iter ≥ 2)

p0_b8:

```text
hpx_server   c=1: 112, 100, 98, 96, 95           (all on cid=0)
hpx_server   c=2: 185, 185, 173, 173, 173        (top two: cid=0/cid=1 of iter=10)
llama_server c=1: 103, 103, 103, 103, 103        (all on cid=0)
llama_server c=2: 218, 217, 203, 195, 193        (top two: cid=0/cid=1 of iter=21;
                                                    next two: cid=0/cid=1 of iter=36)
```

p0_b32:

```text
hpx_server   c=1: 302, 295, 291, 288, 287        (all on cid=0)
hpx_server   c=2: 506, 502, 495, 494, 492        (top two: cid=0/cid=1 of iter=43;
                                                    495/492 are iter=2 post-warmup transient)
llama_server c=1: 320, 314, 310, 307, 306        (all on cid=0)
llama_server c=2: 469, 467, 467, 467, 466        (paired iter=8 and paired iter=4)
```

c=2 tails are **paired across clients**: both `cid=0` and
`cid=1` of the same iteration appear together in the top 5 for
every cell. This is consistent with iteration-level system
behavior (e.g. a single batch step taking longer and affecting
both decoded sequences) rather than per-client outliers, but the
client-side data does not identify a cause.

### Batch wall-clock vs per-row latency (cell-level)

| run    | server       | c | n(iter≥2) | sum_lat_ms | p50 (ms) | batch_wc_ms |   sum_lat / batch_wc |
|:-------|:-------------|--:|----------:|-----------:|---------:|------------:|---------------------:|
| p0_b8  | hpx_server   | 1 |     50    |    4592    |   91.0   |    4726     | 0.97 (sequential)    |
| p0_b8  | hpx_server   | 2 |    100    |   17223    |  172.0   |    8810     | 1.96 (≈ 2× overlap)  |
| p0_b8  | llama_server | 1 |     50    |    4905    |   98.5   |    5029     | 0.98 (sequential)    |
| p0_b8  | llama_server | 2 |    100    |   17352    |  182.0   |    8845     | 1.96 (≈ 2× overlap)  |
| p0_b32 | hpx_server   | 1 |     50    |   13798    |  273.0   |   14110     | 0.98 (sequential)    |
| p0_b32 | hpx_server   | 2 |    100    |   48501    |  484.0   |   24830     | 1.95 (≈ 2× overlap)  |
| p0_b32 | llama_server | 1 |     50    |   14834    |  297.5   |   15150     | 0.98 (sequential)    |
| p0_b32 | llama_server | 2 |    100    |   45956    |  459.0   |   23466     | 1.96 (≈ 2× overlap)  |

`c=1` rows: `sum_lat ≈ batch_wc`, consistent with one client
issuing back-to-back requests with negligible idle time between
them.
`c=2` rows: `sum_lat ≈ 2 × batch_wc` on every cell, consistent
with two clients fully overlapping their wall-clock occupancy.
Both servers admit both concurrent clients across the full cell
duration.

### `c=2 / c=1` p50 ratio recap

| server       | p0_b8 c=2/c=1 | p0_b32 c=2/c=1 |
|:-------------|--------------:|---------------:|
| hpx_server   |          1.89 |           1.77 |
| llama_server |          1.85 |           1.54 |

`c=2 / c=1` batch wall-clock ratios show the same pattern:

| server       | p0_b8 wc c=2/c=1 | p0_b32 wc c=2/c=1 |
|:-------------|-----------------:|------------------:|
| hpx_server   |             1.86 |              1.76 |
| llama_server |             1.76 |              1.55 |

The ratio collapse is therefore not a percentile artifact — both
the median per-request latency and the batch wall-clock confirm
the same pattern, and the divergence between the two servers is
larger at `p0_b32` than at `p0_b8`.

### Interpretation (cautious)

Client-side data supports the following observations, stated
deliberately without server-internal commitments:

- At `c=2`, both clients run with effectively 100 % temporal
  overlap on every cell of both servers
  (`sum_lat ≈ 2 × batch_wc`). The admitted-parallelism premise
  holds in every cell.
- On `p0_b8`, the `c=2 / c=1` p50 ratio sits between 1.85 and
  1.89 on both servers. This is consistent with each request
  occupying roughly half of the per-iteration compute when two
  are admitted together; the per-request latency at `c=2` is
  recorded close to twice the `c=1` latency on both servers.
- On `p0_b32`, the `c=2 / c=1` p50 ratio is 1.77 on hpx-server
  and 1.54 on llama-server. The smaller ratio on llama-server
  suggests more useful compute overlap at the longer decode
  budget on this hardware and model; the larger ratio on
  hpx-server suggests less recovery from the `c=2` overhead at
  the longer decode budget. The client-side data alone does not
  identify whether the difference originates in batch-step
  composition, KV-cache reuse, sampler-chain layout, or some
  other scheduling detail.
- p0_b8 `llama_server c=2` shows a recorded per-iteration
  latency drift from ≈ 130 ms in the first several iterations to
  ≈ 180 ms by mid-cell. p0_b32 `llama_server c=2` does not show
  this drift over its 51 iterations. The client-side data does
  not isolate the source.
- Tail rows at `c=2` are paired across clients on every cell.
  This is consistent with iteration-level system effects rather
  than per-client outliers.

These are recorded patterns, not server-internal claims. Phrases
like "consistent with" / "suggests" mark the boundary between
what is measured client-side and what is being inferred. No
scheduler-fairness, scheduler-policy, or production-throughput
conclusion is drawn here.

### Non-claims (analysis subsection)

- No production-throughput claim.
- No fairness claim. Per-client tables show near-identical p50
  in every cell, but this is one prompt / one budget / one
  hardware / one model / one N — not a generalizable fairness
  characterization.
- No generalization beyond this machine (Apple M4 Pro, darwin),
  this model (TinyLlama 1.1B Q4_K_M), this prompt, these budgets
  (8 and 32), greedy sampling, or `N=2` admission. Different
  prompts, longer decodes, larger models, or different hardware
  can move every value and every ratio in these tables.
- No server-internal timing claim. The analysis uses only
  client-side `submit_monotonic_ms`, `complete_monotonic_ms`, and
  `latency_ms`; no scheduler, batch-composition, or KV-cache
  field is inspected here.
- No winner / performance comparison. The `c=2 / c=1` p50 ratio
  divergence at `p0_b32` is recorded as a pattern to keep in
  mind when interpreting future Phase 2b results, not as a
  ranking of either server.

# Experiment 15 — Phase 2b-S3 results (p0_b8 SSE streaming timing)

Status: **PASS.** Admitted-parallelism **streaming** (SSE) run on
the `p0_b8` shape under the same Phase 2b settings as S1/S2
(hpx-server `--n-seq-max 2 --max-concurrent 2`, llama-server
`--parallel 2`). Every row returned HTTP 200; the hpx-server
canonical anchor `0x0619d4d1900c2365` held across all 153 hpx
rows in the streaming terminal event; per-server
`text_normalized_sha256` is stable across every cell; client-side
streaming timing values are recorded descriptively. The
documented streaming detokenization mismatch reproduces and is
record-only.

## S3 run

- Driver: `concurrent_bench.py` extended with an SSE streaming
  path; reuses Exp12's `bench.http_post_streaming` and each
  adapter's `stream_request_body` / `parse_stream_events`
  unmodified (no SSE parser logic duplicated in the harness). The
  non-streaming `_request_once` path is unchanged.
- Config:
  `hpx-bench/experiments/15_hpx_vs_llama_server_semantics/config.phase2b.streaming.p0_b8.json`
  (same Phase 2b shape as S1 except `streaming: true`, plus
  `stream_line_timeout_seconds = 30.0` and
  `stream_overall_timeout_seconds = 60.0`).
- `run_id`: `20260524-225622-c-smoke`
- Result dir: `results/20260524-225622-c-smoke/`
- Driver capture: `local/runs/exp15/phase2b-streaming-p0_b8/driver.{stdout,stderr}`

Backward-compat: the existing non-streaming configs
(`config.phase2a.smoke.p0_b8.json`,
`config.phase2b.smoke.p0_b8.json`, `config.phase2b.p0_b8.json`,
`config.phase2b.p0_b32.json`) all default to `streaming = False`
and produce the same argv and the same non-streaming row schema
as before this extension.

## Shape

```text
workload                p0_b8 (prompt "Hello, my name is", decode_budget=8)
mode                    streaming (SSE)
concurrency_levels      [1, 2]
iterations_per_client   51
warm-up rule            iteration == 1 per client is warm-up for timing aggregates;
                        correctness/stability gates use all rows.
timing-row counts:
  c=1, per server:      n = 50          (1 client × (51 − 1) iters)
  c=2, per server:      n = 100         (2 clients × (51 − 1) iters each)
  warm-up rows dropped: 6 total         (per server: 1 at c=1 + 2 at c=2)

server order            hpx-server first, then llama-server
server lifecycle        each server booted once per run; both cells share that boot
server settings         hpx-server   --n-seq-max 2 --max-concurrent 2
                                     --max-prompt-tokens 512 --n-threads 2
                                     --ctx-size 2048 (no --engine-pool → default OFF)
                        llama-server --parallel 2 --threads 2 --ctx-size 2048
                                     --no-context-shift --seed 0
```

## Per-cell completion summary

| server       | c | rows | completed | failed | timeouts | batch_wall_clock_ms | client_observed_rps (approx) |
|:-------------|--:|-----:|----------:|-------:|---------:|--------------------:|-----------------------------:|
| hpx_server   | 1 |   51 |    51     |   0    |    0     |        4690         | 10.87                        |
| hpx_server   | 2 |  102 |   102     |   0    |    0     |        8790         | 11.60                        |
| llama_server | 1 |   51 |    51     |   0    |    0     |        4984         | 10.23                        |
| llama_server | 2 |  102 |   102     |   0    |    0     |        8786         | 11.61                        |

`client_observed_rps` is approximate / client-observed / includes
loopback / **not** a server-throughput claim.

## Gate status

```text
gate: all rows http=200, no parse errors, no failed reasons, n_decoded == 8,
      stream_parse_error empty, done_seen true, token_event_count > 0
gate: hpx_server hash == 0x0619d4d1900c2365 across all hpx rows
gate: hpx_server   c=1 text_normalized_sha256 stable: c2cb4b4ddddeab5d…
gate: hpx_server   c=2 text_normalized_sha256 stable: c2cb4b4ddddeab5d…
gate: llama_server c=1 text_normalized_sha256 stable: 4bc48f3540cf5ed2…
gate: llama_server c=2 text_normalized_sha256 stable: 4bc48f3540cf5ed2…
S0_GATES: PASS
```

Re-verified over all 306 rows of `client_results.jsonl`:

- All 306 rows returned HTTP 200 with `parse_error == ""`,
  `stream_parse_error == ""`, and `failed_reason == ""`.
- `done_seen == true` on every row.
- `token_event_count > 0` on every row.
- `n_decoded == 8` on every row.
- hpx-server `hash == 0x0619d4d1900c2365` on every hpx row (153
  total: 51 at `c=1`, 102 at `c=2`); the hash set across all hpx
  streaming rows is the single canonical value with zero
  off-anchor rows. The hash is carried in the streaming terminal
  event.
- Per-`(server, concurrency)` `text_normalized_sha256` stable
  across every cell. hpx-server streaming cells hash to
  `c2cb4b4ddddeab5d4b1148e91efa08cd3c098e8816d5ffeb505914290056899e`
  (matching the Phase 1 **streaming** hpx-server `p0_b8` value);
  llama-server streaming cells hash to
  `4bc48f3540cf5ed22e1485a46878b848af8e8bf6ce72bf9c145fe0e2276fe222`.
- No timeouts. No stuck server process.
- `run_notes.txt` forbidden-word audit clean.

## HPX canonical anchor confirmation

```text
hpx_server c=1: 51/51 rows  hash 0x0619d4d1900c2365
hpx_server c=2: 102/102 rows hash 0x0619d4d1900c2365
```

The Phase 0 / Phase 1 / Phase 2b-S0/S1 canonical `p0_b8` anchor
holds under N=2 admitted parallelism in the SSE streaming path
across the 153-row S3 hpx sample. The anchor is emitted in the
hpx-server streaming terminal event for every row.

## Streaming timing table

Computed over `iteration >= 2` rows per client (warm-up dropped);
`n = 50` per cell at `c=1`, `n = 100` per cell at `c=2`.
Percentiles use linear interpolation on the sorted sample.

| server       | c |   n | first_event_ms p50 | first_event_ms p95 | first_token_ms p50 | first_token_ms p95 | total_ms p50 | total_ms p95 | total_ms p99 |
|:-------------|--:|----:|-------------------:|-------------------:|-------------------:|-------------------:|-------------:|-------------:|-------------:|
| hpx_server   | 1 |  50 |  37.0 |  38.0 |  37.0 |  38.0 |  91.0 |  94.5 |  96.5 |
| hpx_server   | 2 | 100 |  67.5 |  86.0 |  67.5 |  86.0 | 172.0 | 174.1 | 179.1 |
| llama_server | 1 |  50 |  39.0 |  40.5 |  39.0 |  40.5 |  97.0 | 103.1 | 104.0 |
| llama_server | 2 | 100 |  49.0 |  90.0 |  49.0 |  90.0 | 181.0 | 187.0 | 192.0 |

`first_event_latency_ms` and `first_token_latency_ms` collapse to
the same value in every cell, because every row's first SSE
record is a content record for both servers (no `event: started`
/ `event: prelude` prefix is emitted in this configuration). This
matches the Phase 0 / Phase 1 streaming observation.

### Inter-event gaps (`inter_event_gaps_ms`, flattened)

Adjacent differences of `token_event_times_ms` per row, flattened
across the non-warm-up timing rows per `(server, concurrency)`.
Counts: `c=1` → `(8-1) × 50 = 350` gaps per server; `c=2` →
`(8-1) × 100 = 700` gaps per server.

| server       | c |   n | p50 | p95 |
|:-------------|--:|----:|----:|----:|
| hpx_server   | 1 | 350 |  8.0 |  8.0 |
| hpx_server   | 2 | 700 | 13.0 | 49.0 |
| llama_server | 1 | 350 |  8.0 | 10.0 |
| llama_server | 2 | 700 | 14.0 | 49.0 |

These are client-observed gaps between consecutive SSE token
records on the wire. They do not isolate model decode time from
network / stream-framing time and they include the harness's
per-line read loop. They are recorded descriptively, are not
aggregated cross-server, and are not interpreted as a winner
determination or a performance baseline.

## Known streaming semantic mismatch observed (record-only)

The documented per-token vs incremental detokenization divergence
reproduces in S3 exactly as in Phase 0 / Phase 1:

```text
hpx_server streaming row:    text_len_chars = 30   sha = c2cb4b4ddddeab5d…
llama_server streaming row:  text_len_chars = 37   sha = 4bc48f3540cf5ed2…
```

hpx-server's SSE path emits one token event per decoded token,
each carrying a raw single-token SentencePiece detokenization;
single-token detokenization drops the inter-token space glue, so
the per-token concatenation has fewer characters
(`JohnSmith.Iamasoftwareengineer`, 30 chars). llama-server's SSE
path uses incremental detokenization and preserves spacing
(` John Smith. I am a software engineer`, 37 chars). The raw
streamed text is not persisted in `client_results.jsonl` (only
`text_len_chars` and `text_normalized_sha256` are recorded); the
character-count and hash difference is the observable signature.

Cross-server streaming-text byte equality is **not meaningful and
not gated**. Each server is internally stable across all its rows
(`text_normalized_sha256` unique and stable per server per cell).

## Server / process state

- hpx-server: `exit=-15` (SIGTERM-driven shutdown), `pid_still_running=False`.
- llama-server: `exit=0`, `pid_still_running=False`.
- No stuck `llama-server` or `llama-hpx-server` after the run
  (`pgrep` clean before and after).

## Forbidden-word audit

`run_notes.txt` clean (`forbidden_word_hits=[]`). Document hits in
`readme.md` / `facts.md` / `results.md` are confined to the
explicit audit-rule prose.

## Valid claims (Phase 2b-S3)

- The streaming admitted-parallelism run completed under this
  shape: both servers admit two concurrent SSE clients and
  complete every request with HTTP 200, `done_seen == true`, and
  `token_event_count > 0`, across all 306 rows.
- The hpx-server `p0_b8` canonical anchor `0x0619d4d1900c2365`
  held in the streaming path across all 153 hpx rows (`c=1` and
  `c=2` combined) at `--n-seq-max 2 --max-concurrent 2` for this
  shape.
- Client-side streaming timing values
  (`first_event_latency_ms`, `first_token_latency_ms`,
  `total_latency_ms`, flattened `inter_event_gaps_ms`) are
  recorded for hpx-server and llama-server at `c ∈ {1, 2}` over
  `n=50` / `n=100` non-warm-up samples per cell.
- Per-server `text_normalized_sha256` is stable across every cell;
  the hpx-server streaming text hash matches the Phase 1 streaming
  `p0_b8` value.
- The harness streaming extension reuses Exp12's
  `http_post_streaming` and the adapters' `parse_stream_events`
  unmodified; the non-streaming path and the four non-streaming
  configs are unchanged (same argv, same row schema).

## Non-claims (Phase 2b-S3)

- No production-throughput claim. The recorded `batch_wall_clock_ms`
  and `client_observed_rps` are client-observed, include loopback,
  and are not server-throughput figures.
- No scheduler-fairness claim. S3 does not measure cross-client
  tail latency or starvation; per-client breakdowns are not
  asserted as a fairness characterization.
- No semantic-equality claim. The streaming detokenization
  divergence (and every other Phase 0 mismatch) means the two
  servers' streamed bytes are not equal even on `p0_b8`; the
  streaming reconstruction mismatch is **record-only**.
- No winner / performance comparison. The streaming timing and
  inter-event-gap tables are side-by-side values, not a ranking.
- No generalization beyond this machine (Apple M4 Pro, darwin),
  this model (TinyLlama 1.1B Q4_K_M), this prompt
  (`"Hello, my name is"`), this budget (8), greedy sampling, or
  `N=2` admission. A different shape, hardware, model, or N can
  move every value in the timing tables and the streamed output.
- Not the full Phase 2b matrix. S3 is `c ∈ {1, 2}`, `p0_b8`
  streaming, `N=2` only. `c=4` / `c=8`, `N=4`, and `p0_b32`
  streaming are deferred to separate design slices.

## Phase 2b-S1/S3 p0_b8 streaming vs non-streaming analysis

Read-only analysis of the already-recorded Phase 2b-S1 (`p0_b8`
non-streaming) and Phase 2b-S3 (`p0_b8` streaming) result JSONL
files. No new runs, no harness changes, no config changes.

Inputs:

- non-streaming: `results/20260524-223346-c-smoke/client_results.jsonl`
- streaming:     `results/20260524-225622-c-smoke/client_results.jsonl`

Both runs: 306 rows, 6 warm-up rows dropped for timing (1 at
`c=1` + 2 at `c=2` per server); `n=50` per `(server)` cell at
`c=1`, `n=100` at `c=2`. Percentiles use linear interpolation on
the sorted sample. In the streaming run, `latency_ms` and
`total_latency_ms` are byte-identical on all 306 rows
(`max |Δ| = 0`), so "total latency" below is the same
send→close wall-clock field in both modes.

### Total latency — non-streaming vs streaming (ms)

| server       | c | nonstream p50 | nonstream p95 | nonstream p99 | streaming p50 | streaming p95 | streaming p99 | p50 Δ (stream − nonstream) |
|:-------------|--:|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|---------------------------:|
| hpx_server   | 1 |  91.0 |  97.1 | 106.1 |  91.0 |  94.5 |  96.5 |  0.0 |
| hpx_server   | 2 | 172.0 | 173.0 | 185.0 | 172.0 | 174.1 | 179.1 |  0.0 |
| llama_server | 1 |  98.5 | 103.0 | 103.0 |  97.0 | 103.1 | 104.0 | −1.5 |
| llama_server | 2 | 182.0 | 187.3 | 217.0 | 181.0 | 187.0 | 192.0 | −1.0 |

The streaming total-latency p50 lands on the non-streaming p50 to
within ≤ 1.5 ms in every cell (identical for both hpx cells). p95
agrees to within ≈ 2.5 ms across all cells. The largest spread is
at the p99 of `llama_server c=2` (non-streaming 217.0 vs streaming
192.0); this is a single-run tail value across two separate runs
and is recorded descriptively only.

### Streaming first-event / first-token (ms)

`first_event_latency_ms == first_token_latency_ms` on 300/300
non-warm-up streaming rows (every row's first SSE record is a
content record; no `event: started` / `event: prelude` prefix in
this configuration), so one table covers both.

| server       | c | p50 | p95 |
|:-------------|--:|----:|----:|
| hpx_server   | 1 | 37.0 | 38.0 |
| hpx_server   | 2 | 67.5 | 86.0 |
| llama_server | 1 | 39.0 | 40.5 |
| llama_server | 2 | 49.0 | 90.0 |

At `c=1`, first-content p50 sits at 37–39 ms on both servers
(close to one prefill + first-decode step on this shape). At
`c=2`, the cell-level p50 is 67.5 ms (hpx) and 49.0 ms (llama);
the per-client breakdown below explains why these cell medians
differ even though both servers exhibit the same two-cluster
first-content pattern.

### Inter-event gaps (streaming, flattened, non-warm-up)

| server       | c |   n |  p50 |  p95 |
|:-------------|--:|----:|-----:|-----:|
| hpx_server   | 1 | 350 |  8.0 |  8.0 |
| hpx_server   | 2 | 700 | 13.0 | 49.0 |
| llama_server | 1 | 350 |  8.0 | 10.0 |
| llama_server | 2 | 700 | 14.0 | 49.0 |

At `c=1`, adjacent token events are ≈ 8 ms apart on both servers.
At `c=2`, the median gap roughly doubles (13–14 ms) and the p95
gap rises to ≈ 49 ms on both servers — consistent with two
admitted sequences interleaving their per-token emissions in a
shared decode step. Recorded only; no server-internal cause is
inferred.

### Per-client symmetry at c=2

Total latency is symmetric across the two clients in both modes —
per-client p50 agrees within ≤ 0.5 ms in every cell:

| mode       | server       | cid | total p50 | total p95 |
|:-----------|:-------------|----:|----------:|----------:|
| nonstream  | hpx_server   |  0  | 172.0 | 172.6 |
| nonstream  | hpx_server   |  1  | 172.0 | 173.0 |
| nonstream  | llama_server |  0  | 182.0 | 187.0 |
| nonstream  | llama_server |  1  | 182.5 | 190.3 |
| streaming  | hpx_server   |  0  | 172.0 | 174.6 |
| streaming  | hpx_server   |  1  | 172.0 | 174.0 |
| streaming  | llama_server |  0  | 181.0 | 187.0 |
| streaming  | llama_server |  1  | 181.0 | 186.6 |

Streaming `first_token_latency_ms` at `c=2`, by contrast, is
**two-clustered** on both servers — one client's first content
lands early (≈ 45–48 ms) and the other's lands ≈ 40 ms later
(≈ 86–89 ms):

| server       | cid | first_token p50 | first_token p95 | min | max |
|:-------------|----:|----------------:|----------------:|----:|----:|
| hpx_server   |  0  | 45.0 | 46.0 | 37 |  50 |
| hpx_server   |  1  | 86.0 | 86.0 | 85 |  87 |
| llama_server |  0  | 48.0 | 50.5 | 35 |  60 |
| llama_server |  1  | 89.0 | 91.5 | 35 | 100 |

The two servers differ in how that clustering maps to client IDs:

- hpx_server: the split is clean and fixed by `client_id` —
  `cid=0` records the early first content (45–46 ms, min 37,
  max 50) and `cid=1` records the late first content (85–87 ms)
  on essentially every iteration. The merged cell median
  therefore falls on the boundary between the two clusters
  (67.5 ms).
- llama_server: the clusters overlap — `cid=1`'s
  `first_token_latency_ms` ranges down to 35 ms on some
  iterations (and `cid=0` up to 60 ms), so the early/late role is
  not fixed to a single client. More than half of the merged
  c=2 rows (59/100) fall below 60 ms, which pulls the merged cell
  median into the low cluster (49.0 ms).

The cell-level `c=2` first-content p50 difference (hpx 67.5 ms vs
llama 49.0 ms) is therefore an artifact of which client each
server's early-content cluster attaches to, not a uniform
per-row gap. Both servers stagger the two clients' first-content
events by ≈ 40 ms. The client-side data does not identify the
cause of the staggering or of the per-client assignment.

### Stability (all rows, both runs)

| run        | hpx hash                | off-anchor | hpx text_sha (c1/c2) | llama text_sha (c1/c2) | n_decoded |
|:-----------|:------------------------|-----------:|:---------------------|:-----------------------|:----------|
| nonstream  | `0x0619d4d1900c2365`    |     0      | `4bc48f3540cf…` (1/1) | `4bc48f3540cf…` (1/1) | {8}       |
| streaming  | `0x0619d4d1900c2365`    |     0      | `c2cb4b4dddde…` (1/1) | `4bc48f3540cf…` (1/1) | {8}       |

The hpx-server `p0_b8` canonical hash held across all 153 hpx
rows in both runs (zero off-anchor rows). Per-`(server,
concurrency)` `text_normalized_sha256` is unique within each cell
in both runs. The streaming hpx-server text hash
(`c2cb4b4dddde…`) differs from the non-streaming one
(`4bc48f3540cf…`), as documented: the SSE path detokenizes per
token (dropping inter-token spaces) and the non-streaming path
detokenizes the whole sequence at the end. The llama-server text
hash is the same (`4bc48f3540cf…`) in both modes here because its
incremental SSE detokenizer preserves spacing and its
`text_normalized_sha256` normalizes the leading space. No
correctness drift observed during this analysis.

### Interpretation (cautious)

- Streaming total latency closely tracks non-streaming total
  latency for `p0_b8` on this setup: p50 matches to within
  ≤ 1.5 ms in every cell and is identical for both hpx cells.
  Streaming does not add a recorded total-completion cost on this
  shape.
- First-content timing (`first_event` / `first_token`) is a
  distinct signal that the non-streaming rows do not carry. It
  records when the first SSE content record reaches the client,
  separate from when the full response completes.
- At `c=2`, llama-server records earlier first content at the
  cell-median level (49.0 ms vs 67.5 ms), while hpx-server records
  slightly lower total completion (p50 172.0 ms vs 181.0 ms) in
  this shape. The first-content difference is tied to the
  per-client clustering described above (llama's early-content
  cluster is shared across both clients; hpx's is fixed to
  `cid=0`), not to a uniform per-row gap.
- No server-internal cause is inferred for the total-latency
  match, the first-content clustering, or the inter-event gap
  growth at `c=2`. The analysis uses only client-side fields
  (`latency_ms` / `total_latency_ms`, `first_event_latency_ms`,
  `first_token_latency_ms`, `token_event_times_ms`,
  `submit_monotonic_ms`).

### Non-claims (analysis subsection)

- No throughput claim. All values are client-observed, include
  loopback, and are not server-throughput figures.
- No semantic-equality claim. The streaming detokenization
  divergence (hpx per-token vs llama incremental) means the two
  servers' streamed bytes are not equal even on `p0_b8`; it
  remains record-only.
- No scheduler-fairness claim. The total-latency per-client
  symmetry and the first-content clustering are recorded for this
  one shape / budget / hardware / model / N; they are not a
  generalizable fairness characterization.
- No winner / performance comparison. The total-latency,
  first-content, inter-event-gap, and per-client tables are
  side-by-side recorded values, not a ranking.
- No generalization beyond this machine (Apple M4 Pro, darwin),
  this model (TinyLlama 1.1B Q4_K_M), this prompt
  (`"Hello, my name is"`), this budget (8), greedy sampling, or
  `N=2` admission. A different shape, hardware, model, or N can
  move every value in these tables.

# Experiment 15 — Phase 2b interim summary

Consolidates the recorded results of Phase 2a-S0, Phase 2b-S0,
Phase 2b-S1, Phase 2b-S2, the Phase 2b-S1/S2 c=2 timing-shape
analysis, Phase 2b-S3, and the Phase 2b-S1/S3 streaming vs
non-streaming analysis. No new runs. No new claims beyond the
results already recorded in the sections above.

## Phase 2b interim summary

### 1. Correctness / stability

Across every admitted-parallelism `N=2` run (Phase 2b-S0, S1, S2,
S3):

- All cells completed cleanly: every row HTTP 200, all clients
  completed, `failed_count == 0`.
- No HTTP 503 in any Phase 2b run. (The only 503s in this
  experiment were in Phase 2a-S0 under single-slot overload; see
  §2.)
- No timeouts.
- No `parse_error` and no `stream_parse_error`.
- HPX canonical hashes held on every hpx row of every shape:
  - `p0_b8`:  `0x0619d4d1900c2365`
  - `p0_b32`: `0x6794e47fe0f84af1`
- Per-`(server, concurrency)` `text_normalized_sha256` stable
  within each cell of every run.
- No stuck `llama-server` or `llama-hpx-server` process after any
  run.

### 2. Admission-policy result (Phase 2a-S0)

Under matched single-slot settings (hpx-server `--max-concurrent 1`,
llama-server `--parallel 1`) at `c = 2`:

- hpx-server admitted one client and rejected the excess client's
  requests with HTTP 503 (immediate, per iteration).
- llama-server queued the excess client's requests and served them
  after the slot freed (HTTP 200 after a wait).

This is a **semantic admission-policy divergence**, not a harness
failure: the harness gates correctly flagged the rejected rows.
It motivated splitting Phase 2 into Phase 2a (admission / rejection
semantics under overload) and Phase 2b (admitted parallelism with
matched `N`). Retry-on-503 is deliberately not enabled, because it
would mask this divergence.

### 3. Admitted-parallelism timing (recorded values)

`latency_ms` (per-request wall-clock), warm-up dropped, `n = 50`
per cell at `c=1` and `n = 100` at `c=2`:

| shape  | server       | c |   p50 |   p95 |   p99 |
|:-------|:-------------|--:|------:|------:|------:|
| p0_b8  | hpx_server   | 1 |  91.0 |  97.1 | 106.1 |
| p0_b8  | hpx_server   | 2 | 172.0 | 173.0 | 185.0 |
| p0_b8  | llama_server | 1 |  98.5 | 103.0 | 103.0 |
| p0_b8  | llama_server | 2 | 182.0 | 187.3 | 217.0 |
| p0_b32 | hpx_server   | 1 | 273.0 | 289.6 | 298.6 |
| p0_b32 | hpx_server   | 2 | 484.0 | 489.1 | 502.0 |
| p0_b32 | llama_server | 1 | 297.5 | 308.6 | 317.1 |
| p0_b32 | llama_server | 2 | 459.0 | 466.0 | 467.0 |

- `p0_b8` non-streaming: hpx-server recorded lower p50, p95, and
  p99 than llama-server at both `c=1` and `c=2`.
- `p0_b32` non-streaming: hpx-server recorded lower p50/p95/p99 at
  `c=1`; llama-server recorded lower p50/p95/p99 at `c=2`.

These are side-by-side recorded values, not a ranking.

### 4. c=2 timing-shape analysis (Phase 2b-S1/S2)

- At `c=2`, both clients run with effectively full temporal
  overlap on every cell of both servers (`sum_lat ≈ 2 ×
  batch_wall_clock`). The admitted-parallelism premise holds in
  every cell.
- Per-client latencies are symmetric: per-client p50 differs by at
  most 0.5 ms across all `c=2` cells in both shapes.
- `c=2` tails are paired across the two clients (both clients of
  the same iteration appear together in each cell's top-5),
  consistent with iteration-level system effects rather than
  per-client outliers.
- The `c=2 / c=1` p50 ratio is ≈ 1.85–1.89 for both servers on
  `p0_b8`, and lower on `p0_b32` (hpx 1.77, llama 1.54). The
  divergence between the two servers is larger at `p0_b32`. Batch
  wall-clock ratios confirm the same pattern, so it is not a
  percentile artifact.
- The existing client-side data cannot identify the server-internal
  cause (batch-step composition, KV-cache reuse, sampler-chain
  layout, or other scheduling detail).

### 5. Streaming p0_b8 summary (Phase 2b-S3 + S1/S3 analysis)

- Streaming total latency closely tracks non-streaming total
  latency on `p0_b8`: p50 matches to within ≤ 1.5 ms in every cell
  (identical for both hpx cells); `latency_ms == total_latency_ms`
  on all 306 streaming rows.
- `first_event_latency_ms == first_token_latency_ms` on 300/300
  non-warm-up streaming rows (every first SSE record is a content
  record in this shape).
- At `c=2`, streaming first-token timing is two-clustered on both
  servers (one client ≈ 45–48 ms, the other ≈ 86–89 ms). The
  cell-level first-token median differs (hpx 67.5 ms vs llama
  49.0 ms) partly because the early-content cluster maps to a fixed
  client on hpx-server (`cid=0`) but overlaps both clients on
  llama-server — not a uniform per-row gap.
- Known streaming text mismatch remains **record-only**:
  hpx-server's per-token SSE detokenization drops inter-token
  spaces (`text_len_chars = 30`); llama-server's incremental SSE
  detokenization preserves spaces (`text_len_chars = 37`). Each
  server is internally stable; cross-server streamed-byte equality
  is not gated.

### 6. Valid claims (Phase 2b interim)

- The Phase 2b `N=2` admitted-parallelism harness works for three
  shapes: `p0_b8` non-streaming, `p0_b32` non-streaming, and
  `p0_b8` streaming. All complete with HTTP 200 across all cells.
- The HPX canonical anchors (`p0_b8` `0x0619d4d1900c2365`,
  `p0_b32` `0x6794e47fe0f84af1`) remain stable under `N=2`
  admitted parallelism across `c ∈ {1, 2}` in these shapes.
- Client-side timing values (`latency_ms` percentiles;
  streaming `first_event` / `first_token` / inter-event gaps) are
  recorded for these shapes.
- The streaming path adds first-content timing visibility that the
  non-streaming path does not carry.

### 7. Non-claims (Phase 2b interim)

- No production-throughput claim.
- No scheduler-fairness claim.
- No semantic-equality claim (streaming detokenization and the
  other Phase 0 mismatches still apply).
- No generalization beyond this machine (Apple M4 Pro, darwin),
  this model (TinyLlama 1.1B Q4_K_M), this prompt, these budgets
  (8, 32), greedy sampling, or these concurrency levels.
- No server-internal causal claim for any timing pattern.
- No `c=4` / `c=8` claim.
- No `p0_b32` streaming claim yet (not run).

### 8. Recommended next branch of work

Two options, neither run here:

- **Option A — `p0_b32` streaming at `N=2`, `c ∈ {1, 2}`.**
  Completes the `p0_b32` streaming surface (the missing fourth
  cell of the {`p0_b8`, `p0_b32`} × {non-streaming, streaming}
  matrix at `N=2`). Reuses the already-validated streaming path
  and the already-validated `p0_b32` shape; no new server config
  and no new concurrency regime.
- **Option B — Phase 2b-N4 smoke, `p0_b8` non-streaming.**
  hpx-server `--n-seq-max 4 --max-concurrent 4`, llama-server
  `--parallel 4`, `c ∈ {1, 2, 4}`. Tests higher admitted
  parallelism. Introduces a new server config and a new
  concurrency regime (`c=4`) at once.

**Recommendation: Option A next.** It is the lower-variable step:
it closes a coherent surface (all four shape×mode cells at the
same `N=2`) using two components that are already individually
validated, with no new server flags and no new concurrency level.
It also yields richer streaming detail at the longer budget — 31
inter-event gaps per row versus 7 on `p0_b8`, and first-content /
clustering data over a 32-token decode — which extends the
existing streaming-vs-non-streaming and c=2 timing-shape analyses
on a known-stable anchor (`0x6794e47fe0f84af1`). Option B opens a
new parallelism regime while the `N=2` `p0_b32` `c=2/c=1` ratio
divergence noted in §4 is still only partly characterized; it is
the recommended step only once the `N=2` surface is complete.

This recommendation does not authorize either run; both remain
separate design slices requiring their own approval.

# Experiment 15 — Phase 2b-S4 results (p0_b32 SSE streaming timing)

Status: **PASS.** Admitted-parallelism **streaming** (SSE) run on
the `p0_b32` shape under the same Phase 2b settings as S1/S2/S3
(hpx-server `--n-seq-max 2 --max-concurrent 2`, llama-server
`--parallel 2`). Every row returned HTTP 200; the hpx-server
canonical anchor `0x6794e47fe0f84af1` held across all 153 hpx
rows in the streaming terminal event; per-server
`text_normalized_sha256` is stable across every cell; client-side
streaming timing values are recorded descriptively. The
documented streaming detokenization mismatch reproduces at the
longer budget and is record-only. **This completes the `N=2`
TinyLlama 2×2 surface** ({`p0_b8`, `p0_b32`} × {non-streaming,
streaming}).

## S4 run

- Driver: `concurrent_bench.py` unchanged (the existing streaming
  path expresses `p0_b32` via `decode_budget = 32`; no harness
  modification needed). The canonical hash for `(p0_b32, 32)` is
  already pinned in the harness as `0x6794e47fe0f84af1`.
- Config (new):
  `hpx-bench/experiments/15_hpx_vs_llama_server_semantics/config.phase2b.streaming.p0_b32.json`
  (mirrors `config.phase2b.streaming.p0_b8.json` except
  `workload.workload_id = "p0_b32"` and
  `workload.decode_budget = 32`).
- `run_id`: `20260524-231545-c-smoke`
- Result dir: `results/20260524-231545-c-smoke/`
- Driver capture: `local/runs/exp15/phase2b-streaming-p0_b32/driver.{stdout,stderr}`

## Shape

```text
workload                p0_b32 (prompt "Hello, my name is", decode_budget=32)
mode                    streaming (SSE)
concurrency_levels      [1, 2]
iterations_per_client   51
warm-up rule            iteration == 1 per client is warm-up for timing aggregates;
                        correctness/stability gates use all rows.
timing-row counts:
  c=1, per server:      n = 50          (1 client × (51 − 1) iters)
  c=2, per server:      n = 100         (2 clients × (51 − 1) iters each)
  warm-up rows dropped: 6 total         (per server: 1 at c=1 + 2 at c=2)

server order            hpx-server first, then llama-server
server lifecycle        each server booted once per run; both cells share that boot
server settings         hpx-server   --n-seq-max 2 --max-concurrent 2
                                     --max-prompt-tokens 512 --n-threads 2
                                     --ctx-size 2048 (no --engine-pool → default OFF)
                        llama-server --parallel 2 --threads 2 --ctx-size 2048
                                     --no-context-shift --seed 0
```

## Per-cell completion summary

| server       | c | rows | completed | failed | timeouts | batch_wall_clock_ms | client_observed_rps (approx) |
|:-------------|--:|-----:|----------:|-------:|---------:|--------------------:|-----------------------------:|
| hpx_server   | 1 |   51 |    51     |   0    |    0     |       14035         | 3.63                         |
| hpx_server   | 2 |  102 |   102     |   0    |    0     |       24769         | 4.12                         |
| llama_server | 1 |   51 |    51     |   0    |    0     |       15157         | 3.37                         |
| llama_server | 2 |  102 |   102     |   0    |    0     |       24911         | 4.10                         |

`client_observed_rps` is approximate / client-observed / includes
loopback / **not** a server-throughput claim.

## Gate status

```text
gate: all rows http=200, no parse errors, no failed reasons, n_decoded == 32,
      stream_parse_error empty, done_seen true, token_event_count > 0
gate: hpx_server hash == 0x6794e47fe0f84af1 across all hpx rows
gate: hpx_server   c=1 text_normalized_sha256 stable: 5f5f2b818a0331c6…
gate: hpx_server   c=2 text_normalized_sha256 stable: 5f5f2b818a0331c6…
gate: llama_server c=1 text_normalized_sha256 stable: 2714e119e1670668…
gate: llama_server c=2 text_normalized_sha256 stable: 2714e119e1670668…
S0_GATES: PASS
```

Re-verified over all 306 rows of `client_results.jsonl`:

- All 306 rows returned HTTP 200 with `parse_error == ""`,
  `stream_parse_error == ""`, and `failed_reason == ""`.
- `done_seen == true` on every row.
- `token_event_count > 0` on every row.
- `n_decoded == 32` on every row.
- hpx-server `hash == 0x6794e47fe0f84af1` on every hpx row (153
  total: 51 at `c=1`, 102 at `c=2`); the hash set across all hpx
  streaming rows is the single canonical value with zero
  off-anchor rows. The hash is carried in the streaming terminal
  event.
- Per-`(server, concurrency)` `text_normalized_sha256` stable
  across every cell. hpx-server streaming cells hash to
  `5f5f2b818a0331c688ba61ebec19e3400853d3c95ed3de03a561c8d27a6e79d8`
  (matching the Phase 1 **streaming** hpx-server `p0_b32` value);
  llama-server streaming cells hash to
  `2714e119e167066851e43feba6ebdfc5685a75a44aa6b71b089580bac7f7baa0`
  (the `p0_b32` non-streaming / llama anchor).
- No timeouts. No stuck server process.
- `run_notes.txt` forbidden-word audit clean.

## HPX canonical anchor confirmation

```text
hpx_server c=1: 51/51 rows  hash 0x6794e47fe0f84af1
hpx_server c=2: 102/102 rows hash 0x6794e47fe0f84af1
```

The Phase 1 / Phase 2b-S2 canonical `p0_b32` anchor holds under
N=2 admitted parallelism in the SSE streaming path across the
153-row S4 hpx sample.

## Streaming timing table

Computed over `iteration >= 2` rows per client (warm-up dropped);
`n = 50` per cell at `c=1`, `n = 100` per cell at `c=2`.
Percentiles use linear interpolation on the sorted sample. In the
streaming run, `latency_ms == total_latency_ms` on all 306 rows
(`max |Δ| = 0`).

| server       | c |   n | first_event_ms p50 | first_event_ms p95 | first_token_ms p50 | first_token_ms p95 | total_ms p50 | total_ms p95 | total_ms p99 |
|:-------------|--:|----:|-------------------:|-------------------:|-------------------:|-------------------:|-------------:|-------------:|-------------:|
| hpx_server   | 1 |  50 |  37.0 |  38.0 |  37.0 |  38.0 | 273.0 | 281.1 | 296.1 |
| hpx_server   | 2 | 100 |  66.5 |  86.0 |  66.5 |  86.0 | 484.0 | 486.1 | 516.0 |
| llama_server | 1 |  50 |  40.0 |  41.0 |  40.0 |  41.0 | 298.0 | 308.1 | 311.5 |
| llama_server | 2 | 100 |  48.0 |  91.0 |  48.0 |  91.0 | 506.5 | 512.0 | 519.1 |

`first_event_latency_ms == first_token_latency_ms` on 300/300
non-warm-up rows (every first SSE record is a content record; no
`event: started` / `event: prelude` prefix in this
configuration), matching the Phase 0 / Phase 1 / S3 observation.
The hpx-server `total_ms` p50 values (273.0 at `c=1`, 484.0 at
`c=2`) land on the Phase 2b-S2 non-streaming `p0_b32` p50 values
recorded in the S2 section; the llama-server `total_ms` p50
values are recorded here for the streaming run. A full
streaming-vs-non-streaming `p0_b32` comparison is deferred to a
separate analysis slice (analogous to the Phase 2b-S1/S3 pass).

### Inter-event gaps (`inter_event_gaps_ms`, flattened)

Adjacent differences of `token_event_times_ms` per row, flattened
across the non-warm-up timing rows per `(server, concurrency)`.
Counts: `c=1` → `(32-1) × 50 = 1550` gaps per server; `c=2` →
`(32-1) × 100 = 3100` gaps per server.

| server       | c |    n |  p50 |  p95 |
|:-------------|--:|-----:|-----:|-----:|
| hpx_server   | 1 | 1550 |  8.0 |  8.0 |
| hpx_server   | 2 | 3100 | 13.0 | 14.0 |
| llama_server | 1 | 1550 |  8.0 |  9.0 |
| llama_server | 2 | 3100 | 14.0 | 14.0 |

At `c=1`, adjacent token events are ≈ 8 ms apart on both servers.
At `c=2`, the median gap is ≈ 13–14 ms on both servers, with a
tight p95 (14 ms) over the 31-gap-per-row, 3100-gap sample. These
are client-observed gaps between consecutive SSE token records on
the wire; they do not isolate model decode time from network /
stream-framing time and they include the harness's per-line read
loop. Recorded descriptively; no server-internal cause inferred.

## Per-client symmetry at c=2

Total latency is symmetric across the two clients: per-client p50
is identical to 0.5 ms in every cell.

| server       | cid | total p50 | total p95 |
|:-------------|----:|----------:|----------:|
| hpx_server   |  0  | 484.0 | 486.0 |
| hpx_server   |  1  | 484.0 | 486.1 |
| llama_server |  0  | 506.5 | 512.0 |
| llama_server |  1  | 506.5 | 512.0 |

Streaming `first_token_latency_ms` at `c=2` shows the same
two-cluster staggering observed on `p0_b8` in S3 (one client's
first content ≈ 45–48 ms, the other ≈ 86–89 ms):

| server       | cid | first_token p50 | first_token p95 | min | max |
|:-------------|----:|----------------:|----------------:|----:|----:|
| hpx_server   |  0  | 45.0 | 46.0 | 45 | 48 |
| hpx_server   |  1  | 86.0 | 86.0 | 85 | 95 |
| llama_server |  0  | 89.0 | 91.0 | 35 | 91 |
| llama_server |  1  | 48.0 | 50.0 | 35 | 51 |

As on `p0_b8`: hpx-server's early-content cluster is fixed to one
client (`cid=0`, 45–48 ms), so the cell-level first-content p50
(66.5 ms) falls on the boundary between the two clusters;
llama-server's clusters overlap (both clients reach `min = 35` ms
on some iterations), so the merged cell median (48.0 ms) sits in
the low cluster. The cell-level first-content p50 difference is
therefore an artifact of cluster/client mapping, not a uniform
per-row gap. The client-side data does not identify the cause.

## Known streaming semantic mismatch observed (record-only)

The per-token vs incremental detokenization divergence reproduces
at `decode_budget = 32`:

```text
hpx_server streaming row:    text_len_chars = 118   sha = 5f5f2b818a03…
llama_server streaming row:  text_len_chars = 145   sha = 2714e119e167…
```

hpx-server's SSE path emits one token event per decoded token,
each carrying a raw single-token SentencePiece detokenization that
drops the inter-token space glue, so the per-token concatenation
has fewer characters (118). llama-server's SSE path uses
incremental detokenization and preserves spacing (145). The
character-count and hash difference is the observable signature
(the raw streamed text is not persisted in `client_results.jsonl`;
only `text_len_chars` and `text_normalized_sha256` are recorded).
Cross-server streaming-text byte equality is **not meaningful and
not gated**; each server is internally stable per cell.

## Server / process state

- hpx-server: `exit=-15` (SIGTERM-driven shutdown), `pid_still_running=False`.
- llama-server: `exit=0`, `pid_still_running=False`.
- No stuck `llama-server` or `llama-hpx-server` after the run
  (`pgrep` clean before and after).

## Forbidden-word audit

`run_notes.txt` clean (`forbidden_word_hits=[]`). Document hits in
`readme.md` / `facts.md` / `results.md` are confined to the
explicit audit-rule prose.

## N=2 TinyLlama 2×2 surface complete

With S4, all four cells of the `N=2` admitted-parallelism surface
on TinyLlama 1.1B Q4_K_M are recorded and green:

| shape  | non-streaming        | streaming            |
|:-------|:---------------------|:---------------------|
| p0_b8  | Phase 2b-S1 (PASS)   | Phase 2b-S3 (PASS)   |
| p0_b32 | Phase 2b-S2 (PASS)   | Phase 2b-S4 (PASS)   |

All four cells held their canonical hpx-server anchor
(`p0_b8` `0x0619d4d1900c2365`, `p0_b32` `0x6794e47fe0f84af1`),
completed every row with HTTP 200, and recorded stable per-cell
`text_normalized_sha256`. This completion does **not** authorize a
model switch (Llama 3) or higher admitted parallelism (`N=4`,
`c ∈ {4, 8}`); each remains a separate design slice.

## Valid claims (Phase 2b-S4)

- The streaming admitted-parallelism run completed under this
  shape: both servers admit two concurrent SSE clients and
  complete every request with HTTP 200, `done_seen == true`, and
  `token_event_count > 0`, across all 306 rows.
- The hpx-server `p0_b32` canonical anchor `0x6794e47fe0f84af1`
  held in the streaming path across all 153 hpx rows (`c=1` and
  `c=2` combined) at `--n-seq-max 2 --max-concurrent 2`.
- Client-side streaming timing values (`first_event_latency_ms`,
  `first_token_latency_ms`, `total_latency_ms`, flattened
  `inter_event_gaps_ms`) are recorded for hpx-server and
  llama-server at `c ∈ {1, 2}` over `n=50` / `n=100` non-warm-up
  samples per cell.
- Per-server `text_normalized_sha256` is stable across every cell;
  the hpx-server streaming text hash matches the Phase 1 streaming
  `p0_b32` value.
- The `N=2` TinyLlama 2×2 surface ({`p0_b8`, `p0_b32`} ×
  {non-streaming, streaming}) is now recorded and green.

## Non-claims (Phase 2b-S4)

- No production-throughput claim. The recorded `batch_wall_clock_ms`
  and `client_observed_rps` are client-observed, include loopback,
  and are not server-throughput figures.
- No scheduler-fairness claim. S4 does not measure cross-client
  tail latency or starvation.
- No semantic-equality claim. The streaming detokenization
  divergence (and every other Phase 0 mismatch) means the two
  servers' streamed bytes are not equal even on `p0_b32`; the
  streaming reconstruction mismatch is **record-only**.
- No winner / performance comparison. The streaming timing,
  inter-event-gap, and per-client tables are side-by-side recorded
  values, not a ranking.
- No generalization beyond this machine (Apple M4 Pro, darwin),
  this model (TinyLlama 1.1B Q4_K_M), this prompt
  (`"Hello, my name is"`), this budget (32), greedy sampling, or
  `N=2` admission. A different shape, hardware, model, or N can
  move every value in the timing tables and the streamed output.
- Surface completion does not authorize a Llama 3 model switch or
  `N=4` / `c ∈ {4, 8}`; those remain separate design slices.

# Experiment 15 — Phase 2b N=2 surface summary

Read-only consolidation of the completed TinyLlama `N=2`
admitted-parallelism 2×2 surface ({`p0_b8`, `p0_b32`} ×
{non-streaming, streaming}). No new runs. No new claims beyond the
values already recorded in the S1–S4 sections above.

Source runs:

- Phase 2b-S1 — `p0_b8` non-streaming — `run_id=20260524-223346-c-smoke`
- Phase 2b-S2 — `p0_b32` non-streaming — `run_id=20260524-223815-c-smoke`
- Phase 2b-S3 — `p0_b8` streaming — `run_id=20260524-225622-c-smoke`
- Phase 2b-S4 — `p0_b32` streaming — `run_id=20260524-231545-c-smoke`

## Phase 2b N=2 surface summary

### 1. Surface completion

| shape  | mode          | slice | status | HPX canonical hash    |
|:-------|:--------------|:------|:-------|:----------------------|
| p0_b8  | non-streaming | S1    | PASS   | `0x0619d4d1900c2365`  |
| p0_b8  | streaming     | S3    | PASS   | `0x0619d4d1900c2365`  |
| p0_b32 | non-streaming | S2    | PASS   | `0x6794e47fe0f84af1`  |
| p0_b32 | streaming     | S4    | PASS   | `0x6794e47fe0f84af1`  |

All four cells of the `N=2` surface are recorded and green under
matched settings (hpx-server `--n-seq-max 2 --max-concurrent 2`,
llama-server `--parallel 2`, `c ∈ {1, 2}`, 51 iters/client).

### 2. Correctness summary

- All four runs passed (`S0_GATES: PASS`).
- No HTTP 503 in any of the four runs.
- No timeouts.
- No `parse_error` and no `stream_parse_error`.
- No stuck `llama-server` or `llama-hpx-server` process after any
  run.
- The HPX canonical anchor held on every hpx row of every shape:
  `p0_b8` `0x0619d4d1900c2365`, `p0_b32` `0x6794e47fe0f84af1`.
- Per-`(server, concurrency)` `text_normalized_sha256` stable
  within each cell of all four runs.

### 3. Total latency summary (`latency_ms` p50, warm-up dropped)

| server       | shape  | mode          |  c=1 |  c=2 |
|:-------------|:-------|:--------------|-----:|-----:|
| hpx_server   | p0_b8  | non-streaming |  91.0 | 172.0 |
| hpx_server   | p0_b8  | streaming     |  91.0 | 172.0 |
| hpx_server   | p0_b32 | non-streaming | 273.0 | 484.0 |
| hpx_server   | p0_b32 | streaming     | 273.0 | 484.0 |
| llama_server | p0_b8  | non-streaming |  98.5 | 182.0 |
| llama_server | p0_b8  | streaming     |  97.0 | 181.0 |
| llama_server | p0_b32 | non-streaming | 297.5 | 459.0 |
| llama_server | p0_b32 | streaming     | 298.0 | 506.5 |

Values are p50 of the per-request wall-clock (`latency_ms`;
`latency_ms == total_latency_ms` on all streaming rows), `n=50`
per `c=1` cell and `n=100` per `c=2` cell. Side-by-side recorded
values, not a ranking.

### 4. Streaming first-token summary (`first_token_latency_ms` p50)

| server       | shape  | c=1 | c=2 |
|:-------------|:-------|----:|----:|
| hpx_server   | p0_b8  | 37.0 | 67.5 |
| hpx_server   | p0_b32 | 37.0 | 66.5 |
| llama_server | p0_b8  | 39.0 | 49.0 |
| llama_server | p0_b32 | 40.0 | 48.0 |

- `first_event_latency_ms == first_token_latency_ms` on every
  streaming row of both S3 and S4 (each first SSE record is a
  content record in this configuration).
- At `c=2`, streaming first-token timing is **two-clustered** on
  both servers (one client's first content ≈ 45–48 ms, the
  other's ≈ 86–89 ms) in both shapes.
- The cell-level first-token median is affected by how that
  clustering maps to client IDs: hpx-server's early-content
  cluster is fixed to one client, so the cell median falls on the
  cluster boundary (≈ 66–68 ms); llama-server's clusters overlap
  across both clients, so the cell median sits in the low cluster
  (≈ 48–49 ms).
- The `c=2` first-token median alone should not be over-interpreted;
  the per-client breakdown in the S3 and S4 sections is the more
  complete view.

### 5. Main observed patterns (neutral)

- HPX total p50 is stable between non-streaming and streaming in
  both shapes: identical at `p0_b8` (91.0 / 172.0) and identical
  at `p0_b32` (273.0 / 484.0).
- llama-server total p50 is also close between non-streaming and
  streaming on `p0_b8` (c=1 98.5 vs 97.0; c=2 182.0 vs 181.0).
- llama-server `p0_b32` `c=2` total p50 differs between
  non-streaming and streaming in these runs (459.0 vs 506.5);
  `c=1` stays close (297.5 vs 298.0).
- `p0_b32` `c=2` is the cell where the non-streaming and streaming
  patterns diverge most across these runs. These are separate
  runs; the client-side data does not isolate the cause.
- Streaming adds first-content timing visibility
  (`first_event_latency_ms` / `first_token_latency_ms` /
  inter-event gaps) that the non-streaming path cannot provide.
- The known streaming text mismatch remains **record-only**:
  hpx-server per-token SSE detokenization drops inter-token spaces
  (`p0_b8` 30 chars, `p0_b32` 118 chars); llama-server incremental
  SSE detokenization preserves spaces (`p0_b8` 37 chars, `p0_b32`
  145 chars).

### 6. Valid claims (N=2 surface)

- The TinyLlama `N=2` admitted-parallelism surface is complete for
  {`p0_b8`, `p0_b32`} × {non-streaming, streaming}.
- The HPX canonical anchors (`p0_b8` `0x0619d4d1900c2365`,
  `p0_b32` `0x6794e47fe0f84af1`) held across the completed surface.
- Client-side timing values are recorded for this exact surface.
- Streaming first-content timing is now recorded for both `p0_b8`
  and `p0_b32`.

### 7. Non-claims (N=2 surface)

- No production-throughput claim.
- No scheduler-fairness claim.
- No semantic-equality claim.
- No server-internal causal claim for any timing pattern,
  including the `p0_b32` `c=2` non-streaming/streaming divergence.
- No `c=4` / `c=8` claim.
- No Llama 3 / larger-model claim.
- No generalization beyond this machine (Apple M4 Pro, darwin),
  this model (TinyLlama 1.1B Q4_K_M), this prompt, these budgets
  (8, 32), greedy sampling, or `c ∈ {1, 2}` at `N=2`.

### 8. Recommended next experiment

**Phase 2b-N4-S0** — TinyLlama `p0_b8` non-streaming, higher
admitted parallelism:

- hpx-server `--n-seq-max 4 --max-concurrent 4`
- llama-server `--parallel 4`
- `c ∈ {1, 2, 4}`
- small `iterations_per_client` first (smoke), before any full
  timing matrix

Rationale: this tests whether the `N=2` admitted-parallelism
findings (clean admission, paired full overlap, stable anchors,
the `c=2` timing-shape patterns) remain stable as `N` grows beyond
2 and as `c` reaches the admission ceiling (`c = N = 4`). It keeps
the model, prompt, budget, and mode fixed so the only new variable
is admitted parallelism.

A Llama 3 (or other larger-model) switch should wait until the
`N=4` smoke passes on TinyLlama, so that admitted-parallelism
behavior at higher `N` is characterized on the known-anchored
model before introducing a new model as an additional variable.
This summary does not authorize either step; each remains a
separate design slice requiring its own approval.

# Experiment 15 — Phase 2b-N4-S0 results (p0_b8 non-streaming, N=4 smoke)

Status: **PASS (Case A — canonical anchor held at N=4).** A small
admitted-parallelism smoke raising `N` from 2 to 4. Both servers
admitted all clients at `c ∈ {1, 2, 4}` with HTTP 200; the
hpx-server canonical `p0_b8` anchor `0x0619d4d1900c2365` held in
**every** cell including `c=4` (where `c == N == max_concurrent`);
per-server `text_normalized_sha256` is stable within every cell;
no HTTP 503; no timeouts; no stuck processes; forbidden-word audit
clean. There is no N=4 alternate-anchor finding to record — the
N=1/N=2 canonical hash held at N=4.

## S0 run

- Driver: `concurrent_bench.py` unchanged (the existing
  `server_args` schema expresses `N=4` as plain integers; no
  harness modification needed).
- Config (new):
  `hpx-bench/experiments/15_hpx_vs_llama_server_semantics/config.phase2b.n4.smoke.p0_b8.json`
  (mirrors `config.phase2b.smoke.p0_b8.json` except
  `concurrency_levels = [1, 2, 4]`, hpx-server `n_seq_max = 4` /
  `max_concurrent = 4`, llama-server `parallel = 4`).
- `run_id`: `20260524-232423-c-smoke`
- Result dir: `results/20260524-232423-c-smoke/`
- Driver capture: `local/runs/exp15/phase2b-n4-smoke-p0_b8/driver.{stdout,stderr}`

## Shape

```text
workload                p0_b8 (prompt "Hello, my name is", decode_budget=8)
mode                    non-streaming
concurrency_levels      [1, 2, 4]
iterations_per_client   6   (smoke; not a full timing result)
warm-up rule            iteration == 1 per client is warm-up for any aggregate;
                        correctness gates use all rows
server order            hpx-server first, then llama-server
server lifecycle        each server booted once per run; all three cells share that boot
server settings         hpx-server   --n-seq-max 4 --max-concurrent 4
                                     --max-prompt-tokens 512 --n-threads 2
                                     --ctx-size 2048 (no --engine-pool → default OFF)
                        llama-server --parallel 4 --threads 2 --ctx-size 2048
                                     --no-context-shift --seed 0
```

Server argv confirmed from `hpx_server.args` / `llama_server.args`:
hpx-server `--n-seq-max 4 --max-concurrent 4`, llama-server
`--parallel 4`.

## Per-cell completion summary

| server       | c | rows | completed | failed | timeouts | batch_wall_clock_ms | client_observed_rps (approx) |
|:-------------|--:|-----:|----------:|-------:|---------:|--------------------:|-----------------------------:|
| hpx_server   | 1 |   6  |     6     |   0    |    0     |         568         | 10.56                        |
| hpx_server   | 2 |  12  |    12     |   0    |    0     |        1073         | 11.18                        |
| hpx_server   | 4 |  24  |    24     |   0    |    0     |        1767         | 13.58                        |
| llama_server | 1 |   6  |     6     |   0    |    0     |         591         | 10.15                        |
| llama_server | 2 |  12  |    12     |   0    |    0     |         818         | 14.67                        |
| llama_server | 4 |  24  |    24     |   0    |    0     |        1677         | 14.31                        |

`client_observed_rps` is approximate / client-observed / includes
loopback / **not** a server-throughput claim.

## Gate status

```text
gate: all rows http=200, no parse errors, no failed reasons, n_decoded == 8
gate: hpx_server hash == 0x0619d4d1900c2365 across all hpx rows
gate: hpx_server   c=1 text_normalized_sha256 stable: 4bc48f3540cf5ed2…
gate: hpx_server   c=2 text_normalized_sha256 stable: 4bc48f3540cf5ed2…
gate: hpx_server   c=4 text_normalized_sha256 stable: 4bc48f3540cf5ed2…
gate: llama_server c=1 text_normalized_sha256 stable: 4bc48f3540cf5ed2…
gate: llama_server c=2 text_normalized_sha256 stable: 4bc48f3540cf5ed2…
gate: llama_server c=4 text_normalized_sha256 stable: 4bc48f3540cf5ed2…
S0_GATES: PASS
```

Re-verified over all 84 rows of `client_results.jsonl`: HTTP 200
on every row; `parse_error == ""` and `failed_reason == ""` on
every row; `n_decoded == 8` on every row; no timeouts. Per-cell
`text_normalized_sha256` is unique (stable) within each cell and
equals the canonical `p0_b8` non-streaming value
(`4bc48f3540cf5ed22e1485a46878b848af8e8bf6ce72bf9c145fe0e2276fe222`)
on both servers across all three concurrency levels.

## HPX hash result (per cell)

```text
hpx_server c=1:  6/6 rows  hash 0x0619d4d1900c2365
hpx_server c=2: 12/12 rows hash 0x0619d4d1900c2365
hpx_server c=4: 24/24 rows hash 0x0619d4d1900c2365
```

- `c=1` hash matches the canonical anchor `0x0619d4d1900c2365`
  (expected; the N=1 anchor is unchanged at `--n-seq-max 4`).
- `c=2` and `c=4` hashes are identical to the `c=1` hash and stable
  within their cells. The canonical anchor held under N=4 admitted
  parallelism — **Case A**. No alternate stable N=4 same-shape
  hash was produced, so there is no N=4 anchor finding to record,
  and no within-cell hash variation (which would have been a
  correctness concern).

## N=4-specific observations

- At `c=4`, where the client count equals the admission ceiling
  (`c == N == max_concurrent == 4`), both servers admitted all four
  concurrent clients and returned HTTP 200 on every request. No
  HTTP 503 occurred at any concurrency level. (The only 503s in
  this experiment remain those from Phase 2a-S0 under single-slot
  overload, where `c > max_concurrent`.)
- The hpx-server canonical greedy hash is invariant as `N` grows
  from 2 to 4 on this shape: the same `0x0619d4d1900c2365`
  appeared at every concurrency level of every Phase 2b run so far
  (N=2 and N=4).
- `batch_wall_clock_ms` rises with `c` on both servers (roughly
  3× from `c=1` to `c=4`). This is recorded as a sanity check
  only; with `iterations_per_client = 6` it is not a timing
  result.

## Smoke latency (record-only sanity, not a timing result)

`latency_ms` p50 over all rows (warm-up included; `n = 6 / 12 / 24`
at `c = 1 / 2 / 4`):

| server       | c=1  | c=2   | c=4   |
|:-------------|-----:|------:|------:|
| hpx_server   | 91.0 | 175.5 | 294.0 |
| llama_server | 97.5 | 136.0 | 287.0 |

These are smoke values from 6 iterations per client (warm-up not
dropped) and are recorded only to confirm the run behaved sanely.
They are **not** aggregated as a timing result, not compared
across servers, and not interpreted.

## Server / process state

- hpx-server: `exit=-15` (SIGTERM-driven shutdown), `pid_still_running=False`.
- llama-server: `exit=0`, `pid_still_running=False`.
- No stuck `llama-server` or `llama-hpx-server` after the run
  (`pgrep` clean before and after).

## Forbidden-word audit

`run_notes.txt` clean (`forbidden_word_hits=[]`). Document hits in
`readme.md` / `facts.md` / `results.md` are confined to the
explicit audit-rule prose.

## Valid claims (Phase 2b-N4-S0)

- The admitted-parallelism setup remains stable when `N` grows
  from 2 to 4 on this shape: both servers admit `c ∈ {1, 2, 4}`
  clients (including `c == N == 4`) with HTTP 200 on every row,
  no 503, no timeout.
- The hpx-server canonical `p0_b8` anchor `0x0619d4d1900c2365`
  held under N=4 admitted parallelism at every concurrency level,
  with stable per-cell `text_normalized_sha256` and
  `n_decoded == 8`.
- The harness expresses `N=4` through the existing `server_args`
  schema with no code change.
- No stuck processes; launch / readiness / shutdown / audit all
  worked as designed at N=4.

## Non-claims (Phase 2b-N4-S0)

- No production-throughput claim. `batch_wall_clock_ms` and
  `client_observed_rps` are client-observed, include loopback, and
  are not server-throughput figures.
- No timing claim. This is a 6-iteration smoke; the `latency_ms`
  values are sanity-only and are not a timing result.
- No scheduler-fairness claim.
- No semantic-equality claim (every Phase 0 mismatch still
  applies).
- No server-internal causal claim.
- No `c=8` claim and no full N=4 timing-matrix claim; those remain
  separate design slices.
- No Llama 3 / larger-model claim.
- No generalization beyond this machine (Apple M4 Pro, darwin),
  this model (TinyLlama 1.1B Q4_K_M), this prompt, this budget (8),
  greedy sampling, or `c ∈ {1, 2, 4}` at `N=4`.

# Experiment 15 — Phase 2b-N4-S1 results (p0_b8 non-streaming, N=4 timing)

Status: **PASS.** Full admitted-parallelism timing run at `N=4` on
the `p0_b8` non-streaming shape (hpx-server `--n-seq-max 4
--max-concurrent 4`, llama-server `--parallel 4`), `c ∈ {1, 2, 4}`,
51 iterations per client. Every row returned HTTP 200; the
hpx-server canonical anchor `0x0619d4d1900c2365` held at `c=1`,
`c=2`, and `c=4`; per-server `text_normalized_sha256` stable within
every cell; no 503; no timeouts; no stuck processes; forbidden-word
audit clean. Side-by-side timing values recorded descriptively.

## S1 run

- Driver: `concurrent_bench.py` unchanged (the existing
  `server_args` schema expresses `N=4`; no harness modification
  needed).
- Config (new):
  `hpx-bench/experiments/15_hpx_vs_llama_server_semantics/config.phase2b.n4.p0_b8.json`
  (mirrors `config.phase2b.n4.smoke.p0_b8.json` except
  `iterations_per_client = 51`).
- `run_id`: `20260524-232816-c-smoke`
- Result dir: `results/20260524-232816-c-smoke/`
- Driver capture: `local/runs/exp15/phase2b-n4-p0_b8/driver.{stdout,stderr}`

## Shape

```text
workload                p0_b8 (prompt "Hello, my name is", decode_budget=8)
mode                    non-streaming
concurrency_levels      [1, 2, 4]
iterations_per_client   51
warm-up rule            iteration == 1 per client is warm-up for timing aggregates;
                        correctness/stability gates use all rows.
timing-row counts:
  c=1, per server:      n = 50          (1 client × (51 − 1) iters)
  c=2, per server:      n = 100         (2 clients × (51 − 1) iters each)
  c=4, per server:      n = 200         (4 clients × (51 − 1) iters each)
  warm-up rows dropped: 14 total        (per server: 1 + 2 + 4 = 7)

server order            hpx-server first, then llama-server
server lifecycle        each server booted once per run; all cells share that boot
server settings         hpx-server   --n-seq-max 4 --max-concurrent 4
                                     --max-prompt-tokens 512 --n-threads 2
                                     --ctx-size 2048 (no --engine-pool → default OFF)
                        llama-server --parallel 4 --threads 2 --ctx-size 2048
                                     --no-context-shift --seed 0
```

## Per-cell completion summary

| server       | c | rows | completed | failed | timeouts | batch_wall_clock_ms | client_observed_rps (approx) |
|:-------------|--:|-----:|----------:|-------:|---------:|--------------------:|-----------------------------:|
| hpx_server   | 1 |   51 |    51     |   0    |    0     |        4687         | 10.88                        |
| hpx_server   | 2 |  102 |   102     |   0    |    0     |        8816         | 11.57                        |
| hpx_server   | 4 |  204 |   204     |   0    |    0     |       16831         | 12.12                        |
| llama_server | 1 |   51 |    51     |   0    |    0     |        5046         | 10.11                        |
| llama_server | 2 |  102 |   102     |   0    |    0     |        8357         | 12.21                        |
| llama_server | 4 |  204 |   204     |   0    |    0     |       15554         | 13.12                        |

`client_observed_rps` is approximate / client-observed / includes
loopback / **not** a server-throughput claim.

## Gate status

```text
gate: all rows http=200, no parse errors, no failed reasons, n_decoded == 8
gate: hpx_server hash == 0x0619d4d1900c2365 across all hpx rows
gate: hpx_server   c=1 text_normalized_sha256 stable: 4bc48f3540cf5ed2…
gate: hpx_server   c=2 text_normalized_sha256 stable: 4bc48f3540cf5ed2…
gate: hpx_server   c=4 text_normalized_sha256 stable: 4bc48f3540cf5ed2…
gate: llama_server c=1 text_normalized_sha256 stable: 4bc48f3540cf5ed2…
gate: llama_server c=2 text_normalized_sha256 stable: 4bc48f3540cf5ed2…
gate: llama_server c=4 text_normalized_sha256 stable: 4bc48f3540cf5ed2…
S0_GATES: PASS
```

Re-verified over all 714 rows of `client_results.jsonl`: HTTP 200
on every row; `parse_error == ""` and `failed_reason == ""` on
every row; `n_decoded == 8` on every row; no timeouts. Per-cell
`text_normalized_sha256` is unique within each cell and equals the
canonical `p0_b8` non-streaming value
(`4bc48f3540cf5ed22e1485a46878b848af8e8bf6ce72bf9c145fe0e2276fe222`)
on both servers across all three concurrency levels.

## HPX canonical anchor confirmation (per cell)

```text
hpx_server c=1:  51/51 rows  hash 0x0619d4d1900c2365
hpx_server c=2: 102/102 rows hash 0x0619d4d1900c2365
hpx_server c=4: 204/204 rows hash 0x0619d4d1900c2365
```

The canonical `p0_b8` anchor held under N=4 admitted parallelism at
every concurrency level, including `c=4` where `c == N ==
max_concurrent`, across the 357-row hpx sample. No off-anchor rows
and no within-cell variation.

## Timing table — `latency_ms` (per-request wall-clock)

Computed over `iteration >= 2` rows per client (warm-up dropped).
Percentiles use linear interpolation on the sorted sample.

| server       | c |   n |   p50 |   p95 |   p99 |
|:-------------|--:|----:|------:|------:|------:|
| hpx_server   | 1 |  50 |  91.0 |  93.0 |  96.5 |
| hpx_server   | 2 | 100 | 172.0 | 173.0 | 174.0 |
| hpx_server   | 4 | 200 | 329.0 | 330.0 | 331.0 |
| llama_server | 1 |  50 |  98.0 | 104.6 | 121.8 |
| llama_server | 2 | 100 | 181.0 | 185.1 | 186.0 |
| llama_server | 4 | 200 | 304.0 | 307.0 | 308.0 |

These are side-by-side timing values recorded under matched
Phase 2b-N4-S1 settings. They are not aggregated cross-server, not
interpreted as a winner determination, and not used as a
performance baseline.

Descriptive observations only (neutral):

- At `c=4` on this shape, recorded hpx-server p50 (329.0 ms) is
  higher than recorded llama-server p50 (304.0 ms); recorded p50
  differs by approximately 25 ms.
- Recorded p99 − p50 spread at `c=4` is tight on both servers
  (hpx 2.0 ms, llama 4.0 ms). The widest recorded tail in this run
  is `llama_server c=1` (p99 121.8 ms vs p50 98.0 ms).
- Recorded `c=4 / c=1` p50 ratios: hpx ≈ 3.62, llama ≈ 3.10.
  Recorded `c=4 / c=2` p50 ratios: hpx ≈ 1.91, llama ≈ 1.68.
  Recorded descriptively, not interpreted as a server property.
- `batch_wall_clock_ms` at `c=4` is recorded within roughly 8 %
  across the two servers (hpx 16831, llama 15554).

## N=2 vs N=4 descriptive comparison (p0_b8 non-streaming)

Recorded `latency_ms` p50 (warm-up dropped). N=2 values are from
Phase 2b-S1 (`run_id=20260524-223346-c-smoke`); N=4 values are
from this run. `c=4` exists only at N=4.

| server       | c | N=2 p50 | N=4 p50 | recorded difference |
|:-------------|--:|--------:|--------:|--------------------:|
| hpx_server   | 1 |  91.0   |  91.0   | 0.0 ms              |
| hpx_server   | 2 | 172.0   | 172.0   | 0.0 ms              |
| hpx_server   | 4 |   —     | 329.0   | new at N=4          |
| llama_server | 1 |  98.5   |  98.0   | −0.5 ms             |
| llama_server | 2 | 182.0   | 181.0   | −1.0 ms             |
| llama_server | 4 |   —     | 304.0   | new at N=4          |

Descriptive notes (neutral):

- At `c=1` and `c=2`, the recorded p50 values under N=4 settings
  match the N=2 values to within ≤ 1 ms on both servers (identical
  for hpx-server). Raising the admission ceiling from 2 to 4 did
  not change the recorded low-concurrency latency on this shape.
- `c=4` is the new cell. Recorded p50 is 329.0 ms (hpx-server) and
  304.0 ms (llama-server). At `c=4`, recorded hpx-server p50 is
  higher than recorded llama-server p50 on this shape.
- The hpx-server canonical anchor `0x0619d4d1900c2365` is invariant
  across both N values and all concurrency levels.

These are recorded values across two separate runs; the
client-side data does not isolate any server-internal cause.

## Server / process state

- hpx-server: `exit=-15` (SIGTERM-driven shutdown), `pid_still_running=False`.
- llama-server: `exit=0`, `pid_still_running=False`.
- No stuck `llama-server` or `llama-hpx-server` after the run.

## Forbidden-word audit

`run_notes.txt` clean (`forbidden_word_hits=[]`). Document hits in
`readme.md` / `facts.md` / `results.md` are confined to the
explicit audit-rule prose.

## Valid claims (Phase 2b-N4-S1)

- Single-machine, matched-N=4-admission `latency_ms` p50 / p95 /
  p99 values are recorded for hpx-server and llama-server on the
  `p0_b8` non-streaming shape at `c ∈ {1, 2, 4}` over `n=50` /
  `n=100` / `n=200` non-warm-up samples per cell.
- The hpx-server canonical anchor `0x0619d4d1900c2365` is stable
  across 357 hpx rows (`c=1`, `c=2`, `c=4` combined) at
  `--n-seq-max 4 --max-concurrent 4`.
- Both servers complete every iteration of the
  `(concurrency)` matrix with HTTP 200, stable `n_decoded`, and
  stable per-cell `text_normalized_sha256`, with no 503 and no
  timeout.
- At `c=1` and `c=2`, the recorded p50 under N=4 settings matches
  the N=2 recorded p50 to within ≤ 1 ms on this shape.

## Non-claims (Phase 2b-N4-S1)

- No production-throughput claim.
- No winner / performance comparison. Latency tables and the
  N=2/N=4 comparison are side-by-side recorded values, not a
  ranking.
- No claim that either server's admission policy is preferable, and
  no claim that either server scales preferentially with
  concurrency or with `N`.
- No scheduler-fairness claim. This run does not measure
  cross-client tail latency or starvation.
- No semantic-equality claim (every Phase 0 mismatch still
  applies; this run does not exercise streaming or `p1`).
- No server-internal causal claim for any timing value, the
  `c=4` cross-server p50 difference, or the N=2/N=4 match at low
  concurrency.
- No `c=8` claim, no full N=4 `p0_b32` or streaming claim, and no
  Llama 3 / larger-model claim; those remain separate design
  slices.
- No generalization beyond this machine (Apple M4 Pro, darwin),
  this model (TinyLlama 1.1B Q4_K_M), this prompt, this budget (8),
  greedy sampling, or `c ∈ {1, 2, 4}` at `N=4`. A different shape,
  hardware, model, or N can move every value in the timing tables.

## Phase 2b-N4-S1 c=4 timing-shape analysis

Read-only analysis of the Phase 2b-N4-S1 result
(`run_id=20260524-232816-c-smoke`, `p0_b8` non-streaming,
`--n-seq-max 4 --max-concurrent 4` / `--parallel 4`). No new runs,
no harness changes. Question: at `c=4`, why does the timing
pattern differ from `c=1` / `c=2`? Input:
`results/20260524-232816-c-smoke/client_results.jsonl`. All
observations are client-side; `n=50` per client at `c=4` after
dropping `iteration=1`.

### Per-client latency at c=4

| server       | client_id | n  |   p50 |   p95 | min | max |
|:-------------|----------:|---:|------:|------:|----:|----:|
| hpx_server   |     0     | 50 | 329.0 | 330.0 | 328 | 331 |
| hpx_server   |     1     | 50 | 329.0 | 330.0 | 329 | 331 |
| hpx_server   |     2     | 50 | 329.0 | 330.0 | 329 | 331 |
| hpx_server   |     3     | 50 | 329.0 | 330.0 | 329 | 331 |
| llama_server |     0     | 50 | 304.0 | 307.0 | 298 | 308 |
| llama_server |     1     | 50 | 304.0 | 307.0 | 298 | 308 |
| llama_server |     2     | 50 | 304.0 | 306.6 | 299 | 307 |
| llama_server |     3     | 50 | 304.0 | 307.0 | 298 | 308 |

All four clients record identical p50 within each server (hpx 329.0,
llama 304.0) and tight per-client spreads. No client systematically
differs. The `c=4` per-client symmetry matches the per-client
symmetry already observed at `c=2`.

### Completion ordering at c=4

The four clients submit together at the start barrier and complete
clustered within ≈ 20–23 ms of each other on every iteration; each
client re-submits as soon as its previous request completes, so
later iterations show the four submits staggered by that same small
span. Completions are clustered, not serialized.

```text
hpx_server c=4 (sub → cmp, ms):
  iter=2  sub=[311,332,333,333]  cmp=[641,662,662,662]  span_cmp=21
  iter=3  sub=[641,662,662,662]  cmp=[972,993,993,993]  span_cmp=21
  iter=4  sub=[972,993,993,993]  cmp=[1302,1322,1322,1323] span_cmp=21

llama_server c=4 (sub → cmp, ms):
  iter=2  sub=[305,328,328,328]  cmp=[612,634,634,635]  span_cmp=23
  iter=3  sub=[612,634,634,635]  cmp=[918,938,938,938]  span_cmp=20
  iter=4  sub=[918,938,938,938]  cmp=[1217,1237,1237,1237] span_cmp=20
```

Per-iteration latency is ≈ 329 ms (hpx) / ≈ 304 ms (llama) and very
stable. The four requests of an iteration are in flight together and
land within a ≈ 20 ms window. This is the same clustered-overlap
pattern observed at `c=2`, extended to four clients.

### Batch wall-clock vs sum of per-row latencies (overlap ratio)

`sum_latency_ms / batch_wall_clock_ms`, non-warm-up rows:

| server       | c |   n | sum_lat_ms | batch_wc_ms | overlap ratio |
|:-------------|--:|----:|-----------:|------------:|--------------:|
| hpx_server   | 1 |  50 |     4570   |    4687     | 0.98          |
| hpx_server   | 2 | 100 |    17232   |    8816     | 1.95          |
| hpx_server   | 4 | 200 |    65884   |   16831     | 3.91          |
| llama_server | 1 |  50 |     4918   |    5046     | 0.97          |
| llama_server | 2 | 100 |    16376   |    8357     | 1.96          |
| llama_server | 4 | 200 |    60791   |   15554     | 3.91          |

The overlap ratio tracks the concurrency level on both servers:
≈ 1 at `c=1` (sequential single client), ≈ 2 at `c=2`, and ≈ 3.9 at
`c=4`. At `c=4` the four clients occupy the cell wall-clock with
near-full temporal overlap on both servers; the admitted-parallelism
premise holds at the admission ceiling (`c == N == max_concurrent`).

### Tail rows (top latency at c=4)

```text
hpx_server   c=4: 331 ms rows are grouped by iteration — iter=17 carries all four
                  clients (cid 0/1/2/3) at 331; iter=51 carries cid1/cid3 at 331;
                  the remaining top rows are cid=0 (the earliest submitter each iter).
llama_server c=4: 308/307 ms rows are grouped in pairs by iteration — e.g. iter=6
                  cid0/cid1, iter=7 cid0/cid1, iter=20 cid0/cid1.
```

The `c=4` tails are grouped across clients of the same iteration
rather than isolated to one client, consistent with iteration-level
system effects (a single batch step taking longer affecting the
sequences decoded together). This matches the paired-tail pattern
recorded at `c=2` in the Phase 2b-S1/S2 analysis. The client-side
data does not identify a cause.

### Scaling ratios

`latency_ms` p50 and `batch_wall_clock_ms`, non-warm-up:

| server       | metric | c=1 | c=2 | c=4 | c2/c1 | c4/c1 | c4/c2 |
|:-------------|:-------|----:|----:|----:|------:|------:|------:|
| hpx_server   | p50    |  91.0 | 172.0 | 329.0 | 1.89 | 3.62 | 1.91 |
| hpx_server   | bwc    |  4687 |  8816 | 16831 | 1.88 | 3.59 | 1.91 |
| llama_server | p50    |  98.0 | 181.0 | 304.0 | 1.85 | 3.10 | 1.68 |
| llama_server | bwc    |  5046 |  8357 | 15554 | 1.66 | 3.08 | 1.86 |

For each server, the p50 ratio and the batch-wall-clock ratio agree,
so the scaling is not a percentile artifact. The per-doubling ratio
(`c2/c1`, `c4/c2`) sits near 1.9 on hpx-server and below 1.9 on
llama-server (`c4/c2 ≈ 1.68` for llama p50). The `c4/c1` ratio is
≈ 3.6 (hpx) and ≈ 3.1 (llama). This extends the same per-server
pattern noted at N=2: hpx-server ratios stay close to the naive
per-doubling value, llama-server ratios sit lower.

### Correctness re-check

Recomputed across all rows of all `c=4`/`c=2`/`c=1` cells:

- hpx-server `hash == 0x0619d4d1900c2365` at every cell (`c=1`,
  `c=2`, `c=4`); single value, no off-anchor rows.
- `text_normalized_sha256` unique (stable) within every cell of
  both servers.
- `n_decoded == 8` on every cell.
- Zero failed or non-200 rows.

No correctness drift observed during this analysis.

### Interpretation (cautious)

- `c=1` and `c=2` reproduce the N=2 pattern: `c=1` recorded p50
  matches the N=2 `c=1` value, `c=2` matches the N=2 `c=2` value,
  and the overlap ratios (≈ 1 and ≈ 2) are unchanged.
- `c=4` does not introduce a new *kind* of timing shape; it is the
  same admitted-parallelism pattern (full overlap, clustered
  completions, per-client symmetry, iteration-grouped tails)
  extended to the admission ceiling (`c == N == max_concurrent ==
  4`). What changes is the magnitude: with four sequences decoded
  together the per-request latency rises to ≈ 329 ms (hpx) / ≈ 304
  ms (llama), and the overlap ratio rises to ≈ 3.9.
- The per-doubling scaling differs between the two servers (hpx
  near 1.9, llama below 1.9 at `c4/c2`), continuing the N=2
  observation; the client-side data alone does not identify whether
  this originates in batch-step composition, KV-cache reuse,
  sampler-chain layout, or other scheduling detail.
- Client-side data can show clustering, overlap, symmetry, and tail
  grouping, but cannot identify the server-internal cause of any of
  these.

### Non-claims (analysis subsection)

- No production-throughput claim. The overlap ratios and
  wall-clock values are client-observed, include loopback, and are
  not server-throughput figures.
- No fairness claim. Per-client tables are near-identical in every
  `c=4` cell, but this is one prompt / one budget / one hardware /
  one model / one N — not a generalizable fairness
  characterization.
- No server-internal causal claim for the per-request latency at
  `c=4`, the per-doubling scaling difference, or the tail grouping.
- No generalization beyond this machine (Apple M4 Pro, darwin),
  this model (TinyLlama 1.1B Q4_K_M), this prompt, this budget (8),
  greedy sampling, or `c ∈ {1, 2, 4}` at `N=4`.
- No `p0_b32`-at-N4 claim and no streaming-at-N4 claim; neither was
  run.
- No Llama 3 / larger-model claim.

# Experiment 15 — Phase 2b-N4-S2 results (p0_b32 non-streaming, N=4 smoke)

Status: **PASS (Case A — canonical anchor held at N=4).** A small
`p0_b32` non-streaming smoke at `N=4`, the longer-budget companion
to the Phase 2b-N4-S0 `p0_b8` smoke. Both servers admitted all
clients at `c ∈ {1, 2, 4}` with HTTP 200; the hpx-server canonical
`p0_b32` anchor `0x6794e47fe0f84af1` held in **every** cell
including `c=4` (where `c == N == max_concurrent`); per-server
`text_normalized_sha256` is stable within every cell; no HTTP 503;
no timeouts; no stuck processes; forbidden-word audit clean. There
is no N=4 `p0_b32` alternate-anchor finding to record — the
N=1/N=2 canonical hash held at N=4.

## S2 run

- Driver: `concurrent_bench.py` unchanged (the existing
  `server_args` schema expresses `N=4` and the `(p0_b32, 32)`
  canonical hash `0x6794e47fe0f84af1` is already pinned; no harness
  modification needed).
- Config (new):
  `hpx-bench/experiments/15_hpx_vs_llama_server_semantics/config.phase2b.n4.smoke.p0_b32.json`
  (mirrors `config.phase2b.n4.smoke.p0_b8.json` except
  `workload.workload_id = "p0_b32"` and
  `workload.decode_budget = 32`).
- `run_id`: `20260524-234307-c-smoke`
- Result dir: `results/20260524-234307-c-smoke/`
- Driver capture: `local/runs/exp15/phase2b-n4-smoke-p0_b32/driver.{stdout,stderr}`

## Shape

```text
workload                p0_b32 (prompt "Hello, my name is", decode_budget=32)
mode                    non-streaming
concurrency_levels      [1, 2, 4]
iterations_per_client   6   (smoke; not a full timing result)
warm-up rule            iteration == 1 per client is warm-up for any aggregate;
                        correctness gates use all rows
server order            hpx-server first, then llama-server
server lifecycle        each server booted once per run; all three cells share that boot
server settings         hpx-server   --n-seq-max 4 --max-concurrent 4
                                     --max-prompt-tokens 512 --n-threads 2
                                     --ctx-size 2048 (no --engine-pool → default OFF)
                        llama-server --parallel 4 --threads 2 --ctx-size 2048
                                     --no-context-shift --seed 0
```

Server argv confirmed from `hpx_server.args` / `llama_server.args`:
hpx-server `--n-seq-max 4 --max-concurrent 4`, llama-server
`--parallel 4`.

## Per-cell completion summary

| server       | c | rows | completed | failed | timeouts | batch_wall_clock_ms | client_observed_rps (approx) |
|:-------------|--:|-----:|----------:|-------:|---------:|--------------------:|-----------------------------:|
| hpx_server   | 1 |   6  |     6     |   0    |    0     |        1677         | 3.58                         |
| hpx_server   | 2 |  12  |    12     |   0    |    0     |        3202         | 3.75                         |
| hpx_server   | 4 |  24  |    24     |   0    |    0     |        5978         | 4.02                         |
| llama_server | 1 |   6  |     6     |   0    |    0     |        1871         | 3.21                         |
| llama_server | 2 |  12  |    12     |   0    |    0     |        2728         | 4.40                         |
| llama_server | 4 |  24  |    24     |   0    |    0     |        5989         | 4.01                         |

`client_observed_rps` is approximate / client-observed / includes
loopback / **not** a server-throughput claim.

## Gate status

```text
gate: all rows http=200, no parse errors, no failed reasons, n_decoded == 32
gate: hpx_server hash == 0x6794e47fe0f84af1 across all hpx rows
gate: hpx_server   c=1 text_normalized_sha256 stable: 2714e119e1670668…
gate: hpx_server   c=2 text_normalized_sha256 stable: 2714e119e1670668…
gate: hpx_server   c=4 text_normalized_sha256 stable: 2714e119e1670668…
gate: llama_server c=1 text_normalized_sha256 stable: 2714e119e1670668…
gate: llama_server c=2 text_normalized_sha256 stable: 2714e119e1670668…
gate: llama_server c=4 text_normalized_sha256 stable: 2714e119e1670668…
S0_GATES: PASS
```

Re-verified over all 84 rows of `client_results.jsonl`: HTTP 200
on every row; `parse_error == ""` and `failed_reason == ""` on
every row; `n_decoded == 32` on every row; no timeouts. Per-cell
`text_normalized_sha256` is unique (stable) within each cell and
equals the canonical `p0_b32` value
(`2714e119e167066851e43feba6ebdfc5685a75a44aa6b71b089580bac7f7baa0`)
on both servers across all three concurrency levels.

## HPX hash result (per cell)

```text
hpx_server c=1:  6/6 rows  hash 0x6794e47fe0f84af1
hpx_server c=2: 12/12 rows hash 0x6794e47fe0f84af1
hpx_server c=4: 24/24 rows hash 0x6794e47fe0f84af1
```

- `c=1` hash matches the canonical anchor `0x6794e47fe0f84af1`
  (expected; the N=1 `p0_b32` anchor is unchanged at
  `--n-seq-max 4`).
- `c=2` and `c=4` hashes are identical to the `c=1` hash and stable
  within their cells. The canonical anchor held under N=4 admitted
  parallelism — **Case A**. No alternate stable N=4 `p0_b32`
  same-shape hash was produced, so there is no N=4 anchor finding
  to record, and no within-cell hash variation.

## N=4 p0_b32-specific observations

- At `c=4`, where the client count equals the admission ceiling
  (`c == N == max_concurrent == 4`), both servers admitted all four
  concurrent clients and returned HTTP 200 on every request. No
  HTTP 503 occurred at any concurrency level.
- The hpx-server canonical greedy hash for `p0_b32` is invariant as
  `N` grows from 2 to 4 on this shape: the same
  `0x6794e47fe0f84af1` appeared at every concurrency level (N=2 in
  Phase 2b-S2/S4, N=4 here).
- Both canonical anchors now hold at N=4: `p0_b8`
  `0x0619d4d1900c2365` (Phase 2b-N4-S0/S1) and `p0_b32`
  `0x6794e47fe0f84af1` (this run).
- `batch_wall_clock_ms` rises with `c` on both servers (roughly
  3.6× from `c=1` to `c=4`). Recorded as a sanity check only; with
  `iterations_per_client = 6` it is not a timing result.

## Smoke latency (record-only sanity, not a timing result)

`latency_ms` p50 over all rows (warm-up included; `n = 6 / 12 / 24`
at `c = 1 / 2 / 4`):

| server       |  c=1  |  c=2  |  c=4   |
|:-------------|------:|------:|-------:|
| hpx_server   | 274.5 | 526.5 |  994.0 |
| llama_server | 310.0 | 455.0 | 1011.0 |

These are smoke values from 6 iterations per client (warm-up not
dropped) and are recorded only to confirm the run behaved sanely.
They are **not** aggregated as a timing result, not compared across
servers, and not interpreted.

## Server / process state

- hpx-server: `exit=-15` (SIGTERM-driven shutdown), `pid_still_running=False`.
- llama-server: `exit=0`, `pid_still_running=False`.
- No stuck `llama-server` or `llama-hpx-server` after the run
  (`pgrep` clean before and after).

## Forbidden-word audit

`run_notes.txt` clean (`forbidden_word_hits=[]`). Document hits in
`readme.md` / `facts.md` / `results.md` are confined to the
explicit audit-rule prose.

## Valid claims (Phase 2b-N4-S2)

- The admitted-parallelism setup is stable for `p0_b32` at `N=4`:
  both servers admit `c ∈ {1, 2, 4}` clients (including `c == N ==
  4`) with HTTP 200 on every row, no 503, no timeout.
- The hpx-server canonical `p0_b32` anchor `0x6794e47fe0f84af1`
  held under N=4 admitted parallelism at every concurrency level,
  with stable per-cell `text_normalized_sha256` and
  `n_decoded == 32`.
- The harness expresses this N=4 `p0_b32` run through the existing
  config schema with no code change.
- No stuck processes; launch / readiness / shutdown / audit all
  worked as designed.

## Non-claims (Phase 2b-N4-S2)

- No production-throughput claim. `batch_wall_clock_ms` and
  `client_observed_rps` are client-observed, include loopback, and
  are not server-throughput figures.
- No timing claim. This is a 6-iteration smoke; the `latency_ms`
  values are sanity-only and are not a timing result.
- No scheduler-fairness claim.
- No semantic-equality claim (every Phase 0 mismatch still
  applies).
- No server-internal causal claim.
- No full N=4 `p0_b32` timing-matrix claim and no streaming-at-N4
  claim; those remain separate design slices.
- No Llama 3 / larger-model claim.
- No generalization beyond this machine (Apple M4 Pro, darwin),
  this model (TinyLlama 1.1B Q4_K_M), this prompt, this budget (32),
  greedy sampling, or `c ∈ {1, 2, 4}` at `N=4`.

# Experiment 15 — Phase 2b-N4-S3 results (p0_b32 non-streaming, N=4 timing)

Status: **PASS.** Full admitted-parallelism timing run at `N=4` on
the `p0_b32` non-streaming shape (hpx-server `--n-seq-max 4
--max-concurrent 4`, llama-server `--parallel 4`), `c ∈ {1, 2, 4}`,
51 iterations per client. Every row returned HTTP 200; the
hpx-server canonical anchor `0x6794e47fe0f84af1` held at `c=1`,
`c=2`, and `c=4`; per-server `text_normalized_sha256` stable within
every cell; no 503; no timeouts; no stuck processes; forbidden-word
audit clean. Side-by-side timing values recorded descriptively.

## S3 run

- Driver: `concurrent_bench.py` unchanged (existing `server_args`
  schema expresses `N=4`; canonical hash for `(p0_b32, 32)` already
  pinned; no harness modification needed).
- Config (new):
  `hpx-bench/experiments/15_hpx_vs_llama_server_semantics/config.phase2b.n4.p0_b32.json`
  (mirrors `config.phase2b.n4.smoke.p0_b32.json` except
  `iterations_per_client = 51`).
- `run_id`: `20260524-234618-c-smoke`
- Result dir: `results/20260524-234618-c-smoke/`
- Driver capture: `local/runs/exp15/phase2b-n4-p0_b32/driver.{stdout,stderr}`

## Shape

```text
workload                p0_b32 (prompt "Hello, my name is", decode_budget=32)
mode                    non-streaming
concurrency_levels      [1, 2, 4]
iterations_per_client   51
warm-up rule            iteration == 1 per client is warm-up for timing aggregates;
                        correctness/stability gates use all rows.
timing-row counts:
  c=1, per server:      n = 50          (1 client × (51 − 1) iters)
  c=2, per server:      n = 100         (2 clients × (51 − 1) iters each)
  c=4, per server:      n = 200         (4 clients × (51 − 1) iters each)
  warm-up rows dropped: 14 total        (per server: 1 + 2 + 4 = 7)

server order            hpx-server first, then llama-server
server lifecycle        each server booted once per run; all cells share that boot
server settings         hpx-server   --n-seq-max 4 --max-concurrent 4
                                     --max-prompt-tokens 512 --n-threads 2
                                     --ctx-size 2048 (no --engine-pool → default OFF)
                        llama-server --parallel 4 --threads 2 --ctx-size 2048
                                     --no-context-shift --seed 0
```

## Per-cell completion summary

| server       | c | rows | completed | failed | timeouts | batch_wall_clock_ms | client_observed_rps (approx) |
|:-------------|--:|-----:|----------:|-------:|---------:|--------------------:|-----------------------------:|
| hpx_server   | 1 |   51 |    51     |   0    |    0     |       14238         | 3.58                         |
| hpx_server   | 2 |  102 |   102     |   0    |    0     |       25013         | 4.08                         |
| hpx_server   | 4 |  204 |   204     |   0    |    0     |       53253         | 3.83                         |
| llama_server | 1 |   51 |    51     |   0    |    0     |       15237         | 3.35                         |
| llama_server | 2 |  102 |   102     |   0    |    0     |       26300         | 3.88                         |
| llama_server | 4 |  204 |   204     |   0    |    0     |       54234         | 3.76                         |

`client_observed_rps` is approximate / client-observed / includes
loopback / **not** a server-throughput claim.

## Gate status

```text
gate: all rows http=200, no parse errors, no failed reasons, n_decoded == 32
gate: hpx_server hash == 0x6794e47fe0f84af1 across all hpx rows
gate: hpx_server   c=1 text_normalized_sha256 stable: 2714e119e1670668…
gate: hpx_server   c=2 text_normalized_sha256 stable: 2714e119e1670668…
gate: hpx_server   c=4 text_normalized_sha256 stable: 2714e119e1670668…
gate: llama_server c=1 text_normalized_sha256 stable: 2714e119e1670668…
gate: llama_server c=2 text_normalized_sha256 stable: 2714e119e1670668…
gate: llama_server c=4 text_normalized_sha256 stable: 2714e119e1670668…
S0_GATES: PASS
```

Re-verified over all 714 rows of `client_results.jsonl`: HTTP 200
on every row; `parse_error == ""` and `failed_reason == ""` on
every row; `n_decoded == 32` on every row; no timeouts. Per-cell
`text_normalized_sha256` is unique within each cell and equals the
canonical `p0_b32` value
(`2714e119e167066851e43feba6ebdfc5685a75a44aa6b71b089580bac7f7baa0`)
on both servers across all three concurrency levels.

## HPX canonical anchor confirmation (per cell)

```text
hpx_server c=1:  51/51 rows  hash 0x6794e47fe0f84af1
hpx_server c=2: 102/102 rows hash 0x6794e47fe0f84af1
hpx_server c=4: 204/204 rows hash 0x6794e47fe0f84af1
```

The canonical `p0_b32` anchor held under N=4 admitted parallelism
at every concurrency level, including `c=4` where `c == N ==
max_concurrent`, across the 357-row hpx sample. No off-anchor rows
and no within-cell variation.

## Timing table — `latency_ms` (per-request wall-clock)

Computed over `iteration >= 2` rows per client (warm-up dropped).
Percentiles use linear interpolation on the sorted sample.

| server       | c |   n |    p50 |    p95 |    p99 |
|:-------------|--:|----:|-------:|-------:|-------:|
| hpx_server   | 1 |  50 |  278.5 |  282.1 |  284.5 |
| hpx_server   | 2 | 100 |  490.0 |  495.0 |  503.0 |
| hpx_server   | 4 | 200 | 1043.0 | 1047.0 | 1075.0 |
| llama_server | 1 |  50 |  298.0 |  306.6 |  313.1 |
| llama_server | 2 | 100 |  516.5 |  531.0 |  543.0 |
| llama_server | 4 | 200 | 1062.0 | 1133.0 | 1138.0 |

These are side-by-side timing values recorded under matched
Phase 2b-N4-S3 settings. They are not aggregated cross-server, not
interpreted as a winner determination, and not used as a
performance baseline.

Descriptive observations only (neutral):

- At `c=1` and `c=2` on this shape, recorded hpx-server p50 is
  lower than recorded llama-server p50 (c=1: 278.5 vs 298.0; c=2:
  490.0 vs 516.5).
- At `c=4` on this shape, recorded hpx-server p50 (1043.0 ms) is
  lower than recorded llama-server p50 (1062.0 ms); recorded p50
  differs by approximately 19 ms.
- Recorded p99 − p50 spread at `c=4` is tighter on hpx-server
  (32.0 ms) than on llama-server (76.0 ms) in this run.
- Recorded `c=4 / c=1` p50 ratios: hpx ≈ 3.75, llama ≈ 3.56.
  Recorded `c=4 / c=2` p50 ratios: hpx ≈ 2.13, llama ≈ 2.06.
  Recorded `c=2 / c=1` p50 ratios: hpx ≈ 1.76, llama ≈ 1.73.
  Recorded descriptively, not interpreted as a server property.

## N=2 vs N=4 descriptive comparison (p0_b32 non-streaming)

Recorded `latency_ms` p50. N=2 values are from Phase 2b-S2
(`run_id=20260524-223815-c-smoke`); N=4 values are from this run.
`c=4` exists only at N=4.

| server       | c | N=2 p50 | N=4 p50 | recorded difference |
|:-------------|--:|--------:|--------:|--------------------:|
| hpx_server   | 1 | 273.0   | 278.5   | +5.5 ms             |
| hpx_server   | 2 | 484.0   | 490.0   | +6.0 ms             |
| hpx_server   | 4 |   —     | 1043.0  | new at N=4          |
| llama_server | 1 | 297.5   | 298.0   | +0.5 ms             |
| llama_server | 2 | 459.0   | 516.5   | +57.5 ms            |
| llama_server | 4 |   —     | 1062.0  | new at N=4          |

Descriptive notes (neutral):

- hpx-server `p0_b32` recorded p50 at `c=1` and `c=2` is close
  between N=2 and N=4 (within ≈ 6 ms). The hpx-server `c=2`
  `p0_b32` p50 has stayed near 484–490 ms across the S2 (484.0),
  S4 streaming (484.0), and this N=4 (490.0) runs.
- llama-server `p0_b32` recorded p50 at `c=1` is close between N=2
  and N=4 (within ≈ 0.5 ms), but at `c=2` it differs by ≈ 57 ms
  across these two runs (459.0 vs 516.5). The llama-server `c=2`
  `p0_b32` p50 has been recorded at 459.0 (S2), 506.5 (S4
  streaming), and 516.5 (this N=4 run). This is recorded
  run-to-run variation; the client-side data does not isolate the
  cause.
- `c=4` is the new cell. Recorded p50 is 1043.0 ms (hpx-server) and
  1062.0 ms (llama-server).
- The hpx-server canonical anchor `0x6794e47fe0f84af1` is invariant
  across both N values and all concurrency levels.

## p0_b8 vs p0_b32 descriptive comparison (at N=4)

Recorded `latency_ms` p50. p0_b8 values are from Phase 2b-N4-S1;
p0_b32 values are from this run. Ratio is `p0_b32 / p0_b8`.

| server       | c | p0_b8 p50 | p0_b32 p50 | ratio |
|:-------------|--:|----------:|-----------:|------:|
| hpx_server   | 1 |  91.0     |  278.5     | 3.06  |
| hpx_server   | 2 | 172.0     |  490.0     | 2.85  |
| hpx_server   | 4 | 329.0     | 1043.0     | 3.17  |
| llama_server | 1 |  98.0     |  298.0     | 3.04  |
| llama_server | 2 | 181.0     |  516.5     | 2.85  |
| llama_server | 4 | 304.0     | 1062.0     | 3.49  |

Descriptive notes (neutral):

- The recorded `p0_b32 / p0_b8` p50 ratio is ≈ 2.85–3.49 across the
  cells of both servers at N=4. `p0_b32` performs 4× the decode
  steps of `p0_b8` with a fixed per-request prefill / network
  overhead amortized across more steps, which is recorded
  consistent with the ≈ 3× pattern already noted at N=2 (Phase
  2b-S2).
- The cross-server direction at `c=4` differs by budget in these
  runs: recorded hpx-server p50 is higher than recorded
  llama-server p50 on `p0_b8` (329.0 vs 304.0) and lower than
  recorded llama-server p50 on `p0_b32` (1043.0 vs 1062.0).
  Recorded descriptively; the client-side data does not isolate a
  cause.

## Server / process state

- hpx-server: `exit=-15` (SIGTERM-driven shutdown), `pid_still_running=False`.
- llama-server: `exit=0`, `pid_still_running=False`.
- No stuck `llama-server` or `llama-hpx-server` after the run.

## Forbidden-word audit

`run_notes.txt` clean (`forbidden_word_hits=[]`). Document hits in
`readme.md` / `facts.md` / `results.md` are confined to the
explicit audit-rule prose.

## Valid claims (Phase 2b-N4-S3)

- Single-machine, matched-N=4-admission `latency_ms` p50 / p95 /
  p99 values are recorded for hpx-server and llama-server on the
  `p0_b32` non-streaming shape at `c ∈ {1, 2, 4}` over `n=50` /
  `n=100` / `n=200` non-warm-up samples per cell.
- The hpx-server canonical anchor `0x6794e47fe0f84af1` is stable
  across 357 hpx rows (`c=1`, `c=2`, `c=4` combined) at
  `--n-seq-max 4 --max-concurrent 4`.
- Both servers complete every iteration of the `(concurrency)`
  matrix with HTTP 200, stable `n_decoded`, and stable per-cell
  `text_normalized_sha256`, with no 503 and no timeout.
- The N=4 non-streaming surface on TinyLlama now covers both
  `p0_b8` (Phase 2b-N4-S1) and `p0_b32` (this run), both with their
  canonical anchors held.

## Non-claims (Phase 2b-N4-S3)

- No production-throughput claim.
- No winner / performance comparison. Latency tables and the
  N=2/N=4 and p0_b8/p0_b32 comparisons are side-by-side recorded
  values, not a ranking.
- No claim that either server's admission policy is preferable, and
  no claim that either server scales preferentially with
  concurrency, with `N`, or with decode budget.
- No scheduler-fairness claim.
- No semantic-equality claim (every Phase 0 mismatch still applies;
  this run does not exercise streaming or `p1`).
- No server-internal causal claim for any timing value, the `c=4`
  cross-server p50 difference, the llama `c=2` N=2/N=4 difference,
  or the cross-budget direction flip at `c=4`.
- No `c=8` claim, no streaming-at-N4 claim, and no Llama 3 /
  larger-model claim; those remain separate design slices.
- No generalization beyond this machine (Apple M4 Pro, darwin),
  this model (TinyLlama 1.1B Q4_K_M), this prompt, this budget (32),
  greedy sampling, or `c ∈ {1, 2, 4}` at `N=4`. A different shape,
  hardware, model, or N can move every value in the timing tables.

# Experiment 15 — Phase 2b N=4 non-streaming summary

Read-only consolidation of the completed TinyLlama `N=4`
admitted-parallelism non-streaming surface ({`p0_b8`, `p0_b32`} at
`c ∈ {1, 2, 4}`). No new runs. No new claims beyond the values
already recorded in the N4-S0…S3 sections above.

Source runs:

- Phase 2b-N4-S0 — `p0_b8` smoke — `run_id=20260524-232423-c-smoke`
- Phase 2b-N4-S1 — `p0_b8` timing — `run_id=20260524-232816-c-smoke`
- Phase 2b-N4-S2 — `p0_b32` smoke — `run_id=20260524-234307-c-smoke`
- Phase 2b-N4-S3 — `p0_b32` timing — `run_id=20260524-234618-c-smoke`

## Phase 2b N=4 non-streaming summary

### 1. Surface completion

| shape  | mode          | smoke | timing | status |
|:-------|:--------------|:------|:-------|:-------|
| p0_b8  | non-streaming | N4-S0 | N4-S1  | PASS   |
| p0_b32 | non-streaming | N4-S2 | N4-S3  | PASS   |

Both shapes completed at `N=4` (hpx-server `--n-seq-max 4
--max-concurrent 4`, llama-server `--parallel 4`) with both their
smoke precursors and full timing runs green. Across all four runs:
every row HTTP 200, no HTTP 503, no timeouts, no `parse_error` /
`failed_reason`, no stuck `llama-server` or `llama-hpx-server`
process.

### 2. HPX anchor summary

- `p0_b8`:  `0x0619d4d1900c2365` held at `c=1`, `c=2`, `c=4`
  (N4-S0 smoke and N4-S1 timing).
- `p0_b32`: `0x6794e47fe0f84af1` held at `c=1`, `c=2`, `c=4`
  (N4-S2 smoke and N4-S3 timing).

Both canonical anchors held at every concurrency level including
`c=4` (`c == N == max_concurrent`), with stable per-cell
`text_normalized_sha256` and stable `n_decoded` (8 for `p0_b8`, 32
for `p0_b32`). No off-anchor rows and no within-cell hash variation.

### 3. Timing table (`latency_ms`, non-warm-up; n=50/100/200 at c=1/2/4)

From the N4-S1 (`p0_b8`) and N4-S3 (`p0_b32`) timing runs.

| shape  | server       | c |    p50 |    p95 |    p99 |
|:-------|:-------------|--:|-------:|-------:|-------:|
| p0_b8  | hpx_server   | 1 |   91.0 |   93.0 |   96.5 |
| p0_b8  | hpx_server   | 2 |  172.0 |  173.0 |  174.0 |
| p0_b8  | hpx_server   | 4 |  329.0 |  330.0 |  331.0 |
| p0_b8  | llama_server | 1 |   98.0 |  104.6 |  121.8 |
| p0_b8  | llama_server | 2 |  181.0 |  185.1 |  186.0 |
| p0_b8  | llama_server | 4 |  304.0 |  307.0 |  308.0 |
| p0_b32 | hpx_server   | 1 |  278.5 |  282.1 |  284.5 |
| p0_b32 | hpx_server   | 2 |  490.0 |  495.0 |  503.0 |
| p0_b32 | hpx_server   | 4 | 1043.0 | 1047.0 | 1075.0 |
| p0_b32 | llama_server | 1 |  298.0 |  306.6 |  313.1 |
| p0_b32 | llama_server | 2 |  516.5 |  531.0 |  543.0 |
| p0_b32 | llama_server | 4 | 1062.0 | 1133.0 | 1138.0 |

Side-by-side recorded values, not a ranking.

### 4. Observed patterns (neutral)

- `c=1` and `c=2` `p0_b8` recorded p50 under N=4 settings reproduce
  the N=2 recorded values closely (c=1 identical at 91.0; c=2
  identical at 172.0 for hpx-server; llama within ≤ 1 ms). Raising
  the admission ceiling from 2 to 4 did not change the recorded
  low-concurrency `p0_b8` latency.
- N=4 `p0_b32` is stable across smoke and timing; both anchors hold
  at every concurrency level.
- At `c=4`, the recorded p50 direction differs by budget:
  - `p0_b8`: recorded hpx-server p50 is higher than recorded
    llama-server p50 (329.0 vs 304.0).
  - `p0_b32`: recorded hpx-server p50 is lower than recorded
    llama-server p50 (1043.0 vs 1062.0).
- The recorded `p0_b32 / p0_b8` p50 ratio is roughly 3× across the
  cells of both servers (≈ 2.85–3.49), with the largest recorded
  ratio at `llama_server c=4` (≈ 3.49).
- The recorded p99 − p50 spread at `c=4` is tight on both servers
  for `p0_b8` (hpx 2.0 ms, llama 4.0 ms); for `p0_b32` it is 32.0
  ms (hpx) and 76.0 ms (llama) in this run.
- Client-side data does not identify the server-internal cause of
  the `c=4` cross-server direction difference, the cross-budget
  ratios, or the tail spreads.

### 5. Valid claims (N=4 non-streaming surface)

- The TinyLlama `N=4` non-streaming surface is complete for `p0_b8`
  and `p0_b32` at `c ∈ {1, 2, 4}`.
- The HPX canonical anchors (`p0_b8` `0x0619d4d1900c2365`,
  `p0_b32` `0x6794e47fe0f84af1`) hold at `c=1`, `c=2`, and `c=4`.
- Client-side `latency_ms` p50 / p95 / p99 values are recorded for
  this exact N=4 non-streaming surface, over `n=50` / `n=100` /
  `n=200` non-warm-up samples per cell.

### 6. Non-claims (N=4 non-streaming surface)

- No production-throughput claim.
- No scheduler-fairness claim.
- No semantic-equality claim.
- No streaming-at-N=4 claim (not yet run).
- No Llama 3 / larger-model claim.
- No server-internal causal claim for any recorded timing pattern.
- No generalization beyond this machine (Apple M4 Pro, darwin),
  this model (TinyLlama 1.1B Q4_K_M), this prompt, these budgets
  (8, 32), greedy sampling, or `c ∈ {1, 2, 4}` at `N=4`.

### 7. Recommended next experiment

**Phase 2b-N4-S4** — TinyLlama `p0_b8` SSE streaming at `N=4`:

- hpx-server `--n-seq-max 4 --max-concurrent 4`
- llama-server `--parallel 4`
- `c ∈ {1, 2, 4}`
- small smoke first (low `iterations_per_client`), before any full
  streaming timing matrix

Rationale: this checks whether the SSE streaming path that passed
at `N=2` (Phase 2b-S3 / S4) remains stable at `N=4` — every row
HTTP 200, `done_seen`, `token_event_count > 0`, anchors held, and
the known per-token detokenization mismatch still record-only — as
`c` reaches the admission ceiling (`c == N == 4`). It keeps the
model, prompt, budget, and `N` fixed so the only new variable is
the streaming transport at higher admitted parallelism.

A Llama 3 (or other larger-model) switch should wait until the
`N=4` streaming smoke passes on TinyLlama, so that streaming
behavior at higher `N` is characterized on the known-anchored model
before a new model is introduced as an additional variable. This
summary does not authorize either step; each remains a separate
design slice requiring its own approval.

# Experiment 15 — Phase 2b-N4-S4 results (p0_b8 SSE streaming, N=4 smoke)

Status: **PASS.** A small `p0_b8` **streaming** (SSE) smoke at
`N=4`, checking whether the SSE path that passed at `N=2` remains
stable as admitted parallelism grows to 4. Both servers admitted
all clients at `c ∈ {1, 2, 4}` with HTTP 200; every row has
`done_seen` and `token_event_count > 0`; the hpx-server canonical
`p0_b8` anchor `0x0619d4d1900c2365` held in the streaming terminal
event at `c=1`, `c=2`, and `c=4`; per-server
`text_normalized_sha256` stable within every cell; no HTTP 503; no
timeouts; no stuck processes; forbidden-word audit clean. The known
per-token detokenization mismatch reproduced and remains
record-only.

## S4 run

- Driver: `concurrent_bench.py` unchanged (the existing streaming
  path and `server_args` schema express N=4 streaming; no harness
  modification needed).
- Config (new):
  `hpx-bench/experiments/15_hpx_vs_llama_server_semantics/config.phase2b.n4.smoke.streaming.p0_b8.json`
  (mirrors `config.phase2b.n4.smoke.p0_b8.json` plus
  `streaming: true`, `stream_line_timeout_seconds = 30.0`,
  `stream_overall_timeout_seconds = 60.0`).
- `run_id`: `20260524-235810-c-smoke`
- Result dir: `results/20260524-235810-c-smoke/`
- Driver capture: `local/runs/exp15/phase2b-n4-smoke-streaming-p0_b8/driver.{stdout,stderr}`

## Shape

```text
workload                p0_b8 (prompt "Hello, my name is", decode_budget=8)
mode                    streaming (SSE)
concurrency_levels      [1, 2, 4]
iterations_per_client   6   (smoke; not a full timing result)
warm-up rule            iteration == 1 per client is warm-up for any aggregate;
                        correctness gates use all rows
server order            hpx-server first, then llama-server
server lifecycle        each server booted once per run; all three cells share that boot
server settings         hpx-server   --n-seq-max 4 --max-concurrent 4
                                     --max-prompt-tokens 512 --n-threads 2
                                     --ctx-size 2048 (no --engine-pool → default OFF)
                        llama-server --parallel 4 --threads 2 --ctx-size 2048
                                     --no-context-shift --seed 0
```

## Per-cell completion summary

| server       | c | rows | completed | failed | timeouts | batch_wall_clock_ms | client_observed_rps (approx) |
|:-------------|--:|-----:|----------:|-------:|---------:|--------------------:|-----------------------------:|
| hpx_server   | 1 |   6  |     6     |   0    |    0     |         557         | 10.77                        |
| hpx_server   | 2 |  12  |    12     |   0    |    0     |        1076         | 11.15                        |
| hpx_server   | 4 |  24  |    24     |   0    |    0     |        1775         | 13.52                        |
| llama_server | 1 |   6  |     6     |   0    |    0     |         595         | 10.08                        |
| llama_server | 2 |  12  |    12     |   0    |    0     |         961         | 12.49                        |
| llama_server | 4 |  24  |    24     |   0    |    0     |        1681         | 14.28                        |

`client_observed_rps` is approximate / client-observed / includes
loopback / **not** a server-throughput claim.

## Streaming gate status

```text
gate: all rows http=200, no parse errors, no failed reasons, n_decoded == 8,
      stream_parse_error empty, done_seen true, token_event_count > 0
gate: hpx_server hash == 0x0619d4d1900c2365 across all hpx rows
gate: hpx_server   c=1 text_normalized_sha256 stable: c2cb4b4ddddeab5d…
gate: hpx_server   c=2 text_normalized_sha256 stable: c2cb4b4ddddeab5d…
gate: hpx_server   c=4 text_normalized_sha256 stable: c2cb4b4ddddeab5d…
gate: llama_server c=1 text_normalized_sha256 stable: 4bc48f3540cf5ed2…
gate: llama_server c=2 text_normalized_sha256 stable: 4bc48f3540cf5ed2…
gate: llama_server c=4 text_normalized_sha256 stable: 4bc48f3540cf5ed2…
S0_GATES: PASS
```

Re-verified over all 84 rows of `client_results.jsonl`: HTTP 200,
`parse_error == ""`, `stream_parse_error == ""`,
`failed_reason == ""`, `done_seen == true`,
`token_event_count > 0`, and `n_decoded == 8` on every row; no
timeouts. `first_event_latency_ms == first_token_latency_ms` on
84/84 rows (each first SSE record is a content record). Per-cell
`text_normalized_sha256` is unique within each cell: hpx-server
cells hash to `c2cb4b4dddde…` (the Phase 1 / Phase 2b-S3 streaming
`p0_b8` value), llama-server cells to `4bc48f3540cf…`.

## HPX hash result (per cell)

```text
hpx_server c=1:  6/6 rows  hash 0x0619d4d1900c2365
hpx_server c=2: 12/12 rows hash 0x0619d4d1900c2365
hpx_server c=4: 24/24 rows hash 0x0619d4d1900c2365
```

The canonical `p0_b8` anchor held in the streaming terminal event
at every concurrency level, including `c=4` (`c == N ==
max_concurrent`). No off-anchor rows and no within-cell variation.

## Known streaming mismatch (record-only)

Reproduced at N=4 exactly as at N=2:

```text
hpx_server streaming row:    text_len_chars = 30   sha = c2cb4b4dddde…
llama_server streaming row:  text_len_chars = 37   sha = 4bc48f3540cf…
```

hpx-server's per-token SSE detokenization drops inter-token spaces
(30 chars); llama-server's incremental SSE detokenization preserves
spacing (37 chars). Each server is internally stable per cell;
cross-server streamed-byte equality is not gated. This is the same
mismatch recorded in Phase 0 / Phase 1 / Phase 2b-S3, unchanged at
`N=4`.

## Smoke timing (record-only sanity, not a timing result)

`first_event_latency_ms` / `first_token_latency_ms` /
`total_latency_ms` p50 over all rows (warm-up included;
`n = 6 / 12 / 24` at `c = 1 / 2 / 4`):

| server       | c | first_event | first_token | total |
|:-------------|--:|------------:|------------:|------:|
| hpx_server   | 1 |  37.0 |  37.0 |  92.0 |
| hpx_server   | 2 |  75.0 |  75.0 | 178.0 |
| hpx_server   | 4 |  99.0 |  99.0 | 294.5 |
| llama_server | 1 |  39.5 |  39.5 |  99.0 |
| llama_server | 2 |  37.0 |  37.0 | 155.0 |
| llama_server | 4 |  46.0 |  46.0 | 292.0 |

These are smoke values from 6 iterations per client (warm-up not
dropped) and are recorded only to confirm the run behaved sanely.
They are **not** aggregated as a timing result, not compared across
servers, and not interpreted. (`first_event == first_token` in
every cell, consistent with the N=2 streaming runs.)

## Server / process state

- hpx-server: `exit=-15` (SIGTERM-driven shutdown), `pid_still_running=False`.
- llama-server: `exit=0`, `pid_still_running=False`.
- No stuck `llama-server` or `llama-hpx-server` after the run
  (`pgrep` clean before and after).

## Forbidden-word audit

`run_notes.txt` clean (`forbidden_word_hits=[]`). Document hits in
`readme.md` / `facts.md` / `results.md` are confined to the
explicit audit-rule prose.

## Valid claims (Phase 2b-N4-S4)

- The SSE streaming path is stable for `p0_b8` at `N=4`: both
  servers admit `c ∈ {1, 2, 4}` concurrent SSE clients (including
  `c == N == 4`) and complete every request with HTTP 200,
  `done_seen == true`, and `token_event_count > 0`, with no 503
  and no timeout.
- The hpx-server canonical `p0_b8` anchor `0x0619d4d1900c2365` held
  in the streaming path at every concurrency level, with stable
  per-cell `text_normalized_sha256` and `n_decoded == 8`.
- The harness expresses N=4 streaming through the existing config
  schema with no code change.
- No stuck processes; launch / readiness / shutdown / audit all
  worked as designed.

## Non-claims (Phase 2b-N4-S4)

- No production-throughput claim. `batch_wall_clock_ms` and
  `client_observed_rps` are client-observed, include loopback, and
  are not server-throughput figures.
- No timing claim. This is a 6-iteration smoke; the timing values
  are sanity-only.
- No scheduler-fairness claim.
- No semantic-equality claim. The streaming detokenization
  divergence (and every other Phase 0 mismatch) means the two
  servers' streamed bytes are not equal even on `p0_b8`; it remains
  record-only.
- No server-internal causal claim.
- No `p0_b32`-streaming-at-N4 claim and no full N=4 streaming
  timing-matrix claim; those remain separate design slices.
- No `c=8` claim.
- No Llama 3 / larger-model claim.
- No generalization beyond this machine (Apple M4 Pro, darwin),
  this model (TinyLlama 1.1B Q4_K_M), this prompt, this budget (8),
  greedy sampling, or `c ∈ {1, 2, 4}` at `N=4`.

# Experiment 15 — Phase 2b-N4-S5 results (p0_b8 SSE streaming, N=4 timing)

Status: **PASS.** Full client-side **streaming** (SSE) timing run on
the `p0_b8` shape at admitted parallelism `N=4`, following the N4-S4
smoke. Every row returned HTTP 200; the hpx-server canonical anchor
`0x0619d4d1900c2365` held across all 357 hpx rows
(`c ∈ {1, 2, 4}`) in the streaming terminal event; per-server
`text_normalized_sha256` is stable across every cell; client-side
streaming timing values are recorded descriptively. The documented
streaming detokenization mismatch reproduces and is record-only.

## S5 run

- Driver: `concurrent_bench.py` unchanged (the existing streaming
  path expresses the N=4 timing matrix with no harness change).
- Config (new):
  `hpx-bench/experiments/15_hpx_vs_llama_server_semantics/config.phase2b.n4.streaming.p0_b8.json`
  (mirrors `config.phase2b.n4.smoke.streaming.p0_b8.json` exactly
  except `iterations_per_client: 51` in place of `6`).
- `run_id`: `20260525-160314-c-smoke`
- Result dir: `results/20260525-160314-c-smoke/`
- Driver capture: `local/runs/exp15/phase2b-n4-streaming-p0_b8/driver.{stdout,stderr}`

## Shape

```text
workload                p0_b8 (prompt "Hello, my name is", decode_budget=8)
mode                    streaming (SSE)
concurrency_levels      [1, 2, 4]
iterations_per_client   51
warm-up rule            iteration == 1 per client is warm-up for timing aggregates;
                        correctness/stability gates use all rows.
timing-row counts:
  c=1, per server:      n = 50          (1 client × (51 − 1) iters)
  c=2, per server:      n = 100         (2 clients × (51 − 1) iters each)
  c=4, per server:      n = 200         (4 clients × (51 − 1) iters each)
  warm-up rows dropped: 7 per server    (1 at c=1 + 2 at c=2 + 4 at c=4)

server order            hpx-server first, then llama-server
server lifecycle        each server booted once per run; all three cells share that boot
server settings         hpx-server   --n-seq-max 4 --max-concurrent 4
                                     --max-prompt-tokens 512 --n-threads 2
                                     --ctx-size 2048 (no --engine-pool → default OFF)
                        llama-server --parallel 4 --threads 2 --ctx-size 2048
                                     --no-context-shift --seed 0
```

## Per-cell completion summary

| server       | c | rows | completed | failed | timeouts | batch_wall_clock_ms | client_observed_rps (approx) |
|:-------------|--:|-----:|----------:|-------:|---------:|--------------------:|-----------------------------:|
| hpx_server   | 1 |  51  |    51     |   0    |    0     |        4679         | 10.90                        |
| hpx_server   | 2 | 102  |   102     |   0    |    0     |        8799         | 11.59                        |
| hpx_server   | 4 | 204  |   204     |   0    |    0     |       17737         | 11.50                        |
| llama_server | 1 |  51  |    51     |   0    |    0     |        4784         | 10.66                        |
| llama_server | 2 | 102  |   102     |   0    |    0     |        8862         | 11.51                        |
| llama_server | 4 | 204  |   204     |   0    |    0     |       15254         | 13.37                        |

`client_observed_rps` is approximate / client-observed / includes
loopback / **not** a server-throughput claim. llama-server
`done_status` is `"limit"` (its label for hitting the requested
decode budget); `n_decoded == 8` on every row confirms the decode
completed as configured.

## Streaming gate status

```text
gate: all rows http=200, no parse errors, no failed reasons, n_decoded == 8,
      stream_parse_error empty, done_seen true, token_event_count > 0
gate: hpx_server hash == 0x0619d4d1900c2365 across all hpx rows
gate: hpx_server   c=1 text_normalized_sha256 stable: c2cb4b4ddddeab5d…
gate: hpx_server   c=2 text_normalized_sha256 stable: c2cb4b4ddddeab5d…
gate: hpx_server   c=4 text_normalized_sha256 stable: c2cb4b4ddddeab5d…
gate: llama_server c=1 text_normalized_sha256 stable: 4bc48f3540cf5ed2…
gate: llama_server c=2 text_normalized_sha256 stable: 4bc48f3540cf5ed2…
gate: llama_server c=4 text_normalized_sha256 stable: 4bc48f3540cf5ed2…
S0_GATES: PASS
```

Re-verified over all 714 rows of `client_results.jsonl`:

- All 714 rows returned HTTP 200 with `parse_error == ""`,
  `stream_parse_error == ""`, and `failed_reason == ""`.
- `done_seen == true` on every row.
- `token_event_count > 0` on every row.
- `n_decoded == 8` on every row.
- hpx-server `hash == 0x0619d4d1900c2365` on every hpx row (357
  total: 51 at `c=1`, 102 at `c=2`, 204 at `c=4`); the hash set
  across all hpx streaming rows is the single canonical value with
  zero off-anchor rows. The hash is carried in the streaming
  terminal event.
- Per-`(server, concurrency)` `text_normalized_sha256` stable
  across every cell. hpx-server streaming cells hash to
  `c2cb4b4ddddeab5d4b1148e91efa08cd3c098e8816d5ffeb505914290056899e`
  (matching the Phase 1 / Phase 2b-S3 / Phase 2b-N4-S4 streaming
  hpx-server `p0_b8` value); llama-server streaming cells hash to
  `4bc48f3540cf5ed22e1485a46878b848af8e8bf6ce72bf9c145fe0e2276fe222`.
- No timeouts. No stuck server process.
- `run_notes.txt` forbidden-word audit clean.

## HPX hash result (per cell)

```text
hpx_server c=1:  51/51 rows  hash 0x0619d4d1900c2365
hpx_server c=2: 102/102 rows hash 0x0619d4d1900c2365
hpx_server c=4: 204/204 rows hash 0x0619d4d1900c2365
```

The canonical `p0_b8` anchor held in the streaming terminal event
at every concurrency level, including `c == N == max_concurrent ==
4`. No off-anchor rows and no within-cell variation across the
357-row hpx sample.

## Streaming timing table

Computed over `iteration >= 2` rows per client (warm-up dropped);
`n = 50` per cell at `c=1`, `n = 100` per cell at `c=2`,
`n = 200` per cell at `c=4`. Percentiles use linear interpolation
on the sorted sample.

| server       | c |   n | first_event_ms p50 | first_event_ms p95 | first_token_ms p50 | first_token_ms p95 | total_ms p50 | total_ms p95 | total_ms p99 |
|:-------------|--:|----:|-------------------:|-------------------:|-------------------:|-------------------:|-------------:|-------------:|-------------:|
| hpx_server   | 1 |  50 |  37.0 |  37.0 |  37.0 |  37.0 |  90.0 |  92.0 |  93.0 |
| hpx_server   | 2 | 100 |  65.0 |  85.0 |  65.0 |  85.0 | 171.0 | 172.0 | 172.0 |
| hpx_server   | 4 | 200 | 132.0 | 174.0 | 132.0 | 174.0 | 330.0 | 365.0 | 365.0 |
| llama_server | 1 |  50 |  38.0 |  38.0 |  38.0 |  38.0 |  93.0 |  94.0 |  97.6 |
| llama_server | 2 | 100 |  46.0 |  87.0 |  46.0 |  87.0 | 176.0 | 177.0 | 177.0 |
| llama_server | 4 | 200 | 100.0 | 100.0 | 100.0 | 100.0 | 298.0 | 299.0 | 315.0 |

`first_event_latency_ms` and `first_token_latency_ms` collapse to
the same value in every cell, because every row's first SSE
record is a content record for both servers (no `event: started` /
`event: prelude` prefix is emitted in this configuration). This
matches the Phase 0 / Phase 1 / Phase 2b-S3 / Phase 2b-N4-S4
streaming observation.

### Inter-event gaps (`inter_event_gaps_ms`, flattened)

Adjacent differences of `token_event_times_ms` per row, flattened
across the non-warm-up timing rows per `(server, concurrency)`.
Counts: `c=1` → `(8-1) × 50 = 350` gaps per server; `c=2` →
`(8-1) × 100 = 700` gaps per server; `c=4` → `(8-1) × 200 =
1400` gaps per server.

| server       | c |    n | p50 | p95 |
|:-------------|--:|-----:|----:|----:|
| hpx_server   | 1 |  350 |  8.0 |  8.0 |
| hpx_server   | 2 |  700 | 13.0 | 48.0 |
| hpx_server   | 4 | 1400 | 29.0 | 95.0 |
| llama_server | 1 |  350 |  8.0 |  8.0 |
| llama_server | 2 |  700 | 13.0 | 49.0 |
| llama_server | 4 | 1400 | 30.0 | 30.0 |

These are client-observed gaps between consecutive SSE token
records on the wire. They do not isolate model decode time from
network / stream-framing time and they include the harness's
per-line read loop. They are recorded descriptively, are not
aggregated cross-server, and are not interpreted as a winner
determination or a performance baseline.

## Sanity check vs Phase 2b-N4-S4 smoke (not a performance comparison)

The N4-S4 smoke values are over `n = 6 / 12 / 24` rows
(warm-up included); the N4-S5 timing values are over
`n = 50 / 100 / 200` rows with warm-up dropped. Side-by-side
`total_latency_ms` p50 values, recorded only to confirm the
streaming shape is reproducible at N=4:

| server       | c | N4-S4 smoke p50 (n incl warm-up) | N4-S5 timing p50 (warm-up dropped) |
|:-------------|--:|---------------------------------:|-----------------------------------:|
| hpx_server   | 1 |  92.0 (n=6)   |  90.0 (n=50)  |
| hpx_server   | 2 | 178.0 (n=12)  | 171.0 (n=100) |
| hpx_server   | 4 | 294.5 (n=24)  | 330.0 (n=200) |
| llama_server | 1 |  99.0 (n=6)   |  93.0 (n=50)  |
| llama_server | 2 | 155.0 (n=12)  | 176.0 (n=100) |
| llama_server | 4 | 292.0 (n=24)  | 298.0 (n=200) |

The N4-S5 values are in the same neighborhood as the smoke values
on a per-cell basis. Sample sizes, warm-up handling, and per-cell
elapsed time differ, so the per-cell deltas are not interpreted.
No performance claim is made from this comparison; it is a
reproducibility sanity check.

## Descriptive comparison vs N=2 p0_b8 streaming (Phase 2b-S3)

Recorded `total_latency_ms` p50 from N=2 streaming (Phase 2b-S3,
`run_id=20260524-225622-c-smoke`, hpx `--n-seq-max 2 --max-concurrent 2`,
llama `--parallel 2`) and this N=4 streaming run. `c=4` exists
only at N=4.

| server       | c | N=2 p50 | N=4 p50 | recorded difference |
|:-------------|--:|--------:|--------:|--------------------:|
| hpx_server   | 1 |  91.0   |  90.0   | −1.0 ms             |
| hpx_server   | 2 | 172.0   | 171.0   | −1.0 ms             |
| hpx_server   | 4 |   —     | 330.0   | new at N=4          |
| llama_server | 1 |  97.0   |  93.0   | −4.0 ms             |
| llama_server | 2 | 181.0   | 176.0   | −5.0 ms             |
| llama_server | 4 |   —     | 298.0   | new at N=4          |

Descriptive notes (neutral):

- At `c=1` and `c=2`, the recorded streaming p50 values under N=4
  settings stay within ≤ 5 ms of the N=2 streaming values on both
  servers. Raising the admission ceiling from 2 to 4 did not move
  the recorded low-concurrency streaming latency on this shape;
  this mirrors the non-streaming N=2/N=4 match observed in Phase
  2b-N4-S1.
- `c=4` is the new cell. Recorded streaming p50 is 330.0 ms
  (hpx-server) and 298.0 ms (llama-server). The recorded hpx-server
  p50 at `c=4` is higher than the recorded llama-server p50 on this
  shape; the same direction was observed in the non-streaming N4-S1
  `c=4` cell (hpx 329.0 ms vs llama 304.0 ms).
- The hpx-server canonical anchor `0x0619d4d1900c2365` is invariant
  across both N values and all concurrency levels in the streaming
  path.

These are recorded values across two separate runs; the
client-side data does not isolate any server-internal cause.

## Known streaming mismatch (record-only)

Reproduced at N=4 timing exactly as at N=2 / N=4 smoke:

```text
hpx_server streaming row:    text_len_chars = 30   sha = c2cb4b4ddddeab5d…
llama_server streaming row:  text_len_chars = 37   sha = 4bc48f3540cf5ed2…
```

hpx-server's per-token SSE detokenization drops inter-token spaces
(30 chars); llama-server's incremental SSE detokenization preserves
spacing (37 chars). Each server is internally stable per cell;
cross-server streamed-byte equality is not gated. This is the same
mismatch recorded in Phase 0 / Phase 1 / Phase 2b-S3 / Phase
2b-N4-S4, unchanged at N=4 with `iterations_per_client = 51`.

## Server / process state

- hpx-server: `exit=-15` (SIGTERM-driven shutdown), `pid_still_running=False`.
- llama-server: `exit=0`, `pid_still_running=False`.
- No stuck `llama-server` or `llama-hpx-server` after the run
  (`pgrep` clean before and after).

## Forbidden-word audit

`run_notes.txt` clean (`forbidden_word_hits=[]`). Document hits in
`readme.md` / `facts.md` / `results.md` are confined to the
explicit audit-rule prose.

## Valid claims (Phase 2b-N4-S5)

- Single-machine, matched-N=4-admission **streaming** (SSE) timing
  values (`first_event_latency_ms`, `first_token_latency_ms`,
  `total_latency_ms`, flattened `inter_event_gaps_ms`) are
  recorded for hpx-server and llama-server on the `p0_b8` shape at
  `c ∈ {1, 2, 4}` over `n = 50 / 100 / 200` non-warm-up samples
  per cell.
- The hpx-server canonical `p0_b8` anchor `0x0619d4d1900c2365` is
  stable across 357 hpx streaming rows (`c=1`, `c=2`, `c=4`
  combined) at `--n-seq-max 4 --max-concurrent 4`, carried in the
  streaming terminal event.
- Both servers complete every iteration of the `(concurrency)`
  matrix with HTTP 200, `done_seen == true`,
  `token_event_count > 0`, and `n_decoded == 8`, with no 503 and
  no timeout, on this shape.
- Per-server `text_normalized_sha256` is stable across every cell;
  the hpx-server streaming text hash matches the Phase 1 / Phase
  2b-S3 / Phase 2b-N4-S4 streaming `p0_b8` value.
- At `c=1` and `c=2`, the recorded streaming p50 under N=4 settings
  stays within ≤ 5 ms of the N=2 streaming p50 on this shape.

## Non-claims (Phase 2b-N4-S5)

- No production-throughput claim. `batch_wall_clock_ms` and
  `client_observed_rps` are client-observed, include loopback, and
  are not server-throughput figures.
- No winner / performance comparison. Streaming timing,
  inter-event-gap, N4-S4 sanity, and N=2 vs N=4 tables are
  side-by-side recorded values, not a ranking.
- No claim that either server's admission policy is preferable, and
  no claim that either server scales preferentially with
  concurrency or with `N`.
- No scheduler-fairness claim. This run does not measure
  cross-client tail latency or starvation.
- No semantic-equality claim. The streaming detokenization
  divergence (and every other Phase 0 mismatch) means the two
  servers' streamed bytes are not equal even on `p0_b8`; it
  remains record-only.
- No server-internal causal claim for any timing value, the `c=4`
  cross-server p50 difference, or the N=2/N=4 match at low
  concurrency.
- No `p0_b32`-streaming-at-N=4 claim, no `c=8` claim, and no Llama
  3 / larger-model claim; those remain separate design slices.
- No generalization beyond this machine (Apple M4 Pro, darwin),
  this model (TinyLlama 1.1B Q4_K_M), this prompt, this budget (8),
  greedy sampling, or `c ∈ {1, 2, 4}` at `N=4`. A different shape,
  hardware, model, or N can move every value in the timing tables.

## Phase 2b-N4-S5 c=4 streaming timing-shape analysis

Read-only analysis of the Phase 2b-N4-S5 result
(`run_id=20260525-160314-c-smoke`, `p0_b8` streaming,
`--n-seq-max 4 --max-concurrent 4` / `--parallel 4`,
`iterations_per_client=51`). No new runs, no harness changes.
Question: at `c=4`, what does the per-client streaming timing
shape look like, and how does it compare to the non-streaming
`c=4` cell of Phase 2b-N4-S1? Inputs:
`results/20260525-160314-c-smoke/client_results.jsonl` (N4
streaming) and `results/20260524-232816-c-smoke/client_results.jsonl`
(N4 non-streaming). All observations are client-side; `n=50` per
client at `c=4` after dropping `iteration=1`.

### Per-client first_token at c=4 (streaming)

| server       | client_id | n  |  min |   p50 |   p95 | max |
|:-------------|----------:|---:|-----:|------:|------:|----:|
| hpx_server   |     0     | 50 |  74  | 132.0 | 132.0 | 134 |
| hpx_server   |     1     | 50 | 132  | 134.0 | 174.0 | 174 |
| hpx_server   |     2     | 50 | 132  | 133.5 | 174.0 | 174 |
| hpx_server   |     3     | 50 |  58  |  58.5 |  90.0 |  95 |
| llama_server |     0     | 50 |  99  | 100.0 | 100.0 | 118 |
| llama_server |     1     | 50 |  99  | 100.0 | 100.0 | 118 |
| llama_server |     2     | 50 |  99  | 100.0 | 100.0 | 118 |
| llama_server |     3     | 50 |  56  |  57.0 |  57.0 |  76 |

The per-client `first_token_latency_ms` is asymmetric on both
servers: `client_id=3` records a single-digit-tens p50 (≈ 58 ms
hpx, ≈ 57 ms llama) while `client_id ∈ {0, 1, 2}` record a higher
p50 (≈ 132–134 ms hpx, ≈ 100 ms llama). The cell median
(`132.0` hpx, `100.0` llama, see Phase 2b-N4-S5 timing table)
reflects the 3-client cohort, which holds 150 of the 200
non-warm-up rows.

### Per-client total at c=4 (streaming)

| server       | client_id | n  |  min |   p50 |   p95 | max |
|:-------------|----------:|---:|-----:|------:|------:|----:|
| hpx_server   |     0     | 50 | 328  | 330.0 | 364.0 | 365 |
| hpx_server   |     1     | 50 | 328  | 330.0 | 365.0 | 365 |
| hpx_server   |     2     | 50 | 328  | 330.0 | 365.0 | 365 |
| hpx_server   |     3     | 50 | 328  | 331.0 | 365.0 | 370 |
| llama_server |     0     | 50 | 295  | 298.0 | 299.0 | 315 |
| llama_server |     1     | 50 | 295  | 298.0 | 299.0 | 315 |
| llama_server |     2     | 50 | 295  | 298.0 | 299.0 | 315 |
| llama_server |     3     | 50 | 295  | 298.0 | 300.1 | 315 |

All four clients record the same per-client `latency_ms` p50
within each server (hpx ≈ 330 ms, llama ≈ 298 ms) and tight
per-client spreads. The streaming `c=4` per-client total is
symmetric across clients; the asymmetry in the
`first_token_latency_ms` table does not propagate to the
per-client total.

### Completion ordering at c=4

Per-iteration `submit_monotonic_ms` and `complete_monotonic_ms`
spreads across the four clients, computed across all 50
non-warm-up iteration rounds:

| run            | server       | submit-spread p50 | complete-spread p50 | all-4-overlap | fully serialized |
|:---------------|:-------------|------------------:|--------------------:|:--------------|:-----------------|
| N4 streaming   | hpx_server   |       21.0 ms     |        21.0 ms      |   50/50       |   0/50           |
| N4 streaming   | llama_server |       20.0 ms     |        20.0 ms      |   50/50       |   0/50           |
| N4 non-stream  | hpx_server   |       21.0 ms     |        21.0 ms      |   50/50       |   0/50           |
| N4 non-stream  | llama_server |       22.0 ms     |        22.0 ms      |   50/50       |   0/50           |

On every iteration of every cell above, the four client request
windows overlap in time (max submit < min complete on all 50
iterations on all four cells). None of the 50 iterations is fully
serialized. The complete-spread tracks the submit-spread to
within ≤ 1 ms in every cell, so the per-iteration completion order
mirrors the per-iteration submit order. This is the same
clustered-overlap pattern recorded in the Phase 2b-N4-S1
non-streaming `c=4` analysis, observed here in the streaming
path. The harness's start-barrier and re-submit loop produce a
≈ 20 ms client-side stagger that persists across iterations
because each client re-submits as soon as its previous request
completes, and clients arrive at the next submit point in
roughly the same order they finished the previous one.

Sample iteration round (`iteration=2`, monotonic ms):

```text
hpx_server c=4   iter=2:
  client=0  submit=430  first_event=132  complete=759  lat=328
  client=1  submit=430  first_event=132  complete=759  lat=328
  client=2  submit=430  first_event=132  complete=759  lat=328
  client=3  submit=409  first_event= 58  complete=738  lat=328

llama_server c=4 iter=2:
  client=0  submit=322  first_event=100  complete=621  lat=298
  client=1  submit=322  first_event=100  complete=621  lat=298
  client=2  submit=322  first_event=100  complete=621  lat=299
  client=3  submit=300  first_event= 59  complete=601  lat=301
```

`client_id=3` submits about 20 ms before clients 0/1/2 in this
round (and in every subsequent iteration). Its
`first_event_latency_ms` is correspondingly ≈ 74 ms (hpx) /
≈ 41 ms (llama) shorter than clients 0/1/2's
`first_event_latency_ms`. All four clients complete within ≈ 20 ms
of each other, with `client_id=3` finishing first, so the per-row
`latency_ms` (`complete - submit`) is essentially identical across
the four clients.

### Streaming vs non-streaming total latency at N=4

`latency_ms` percentiles (warm-up dropped), same N=4 admission
settings, `p0_b8` shape. Streaming values are this run's; the
non-streaming column is the Phase 2b-N4-S1 result
(`run_id=20260524-232816-c-smoke`).

| server       | c |  stream p50 |  stream p95 |  stream p99 | non-stream p50 | non-stream p95 | non-stream p99 |
|:-------------|--:|------------:|------------:|------------:|---------------:|---------------:|---------------:|
| hpx_server   | 1 |        90.0 |        92.0 |        93.0 |          91.0  |          93.0  |          96.5  |
| hpx_server   | 2 |       171.0 |       172.0 |       172.0 |         172.0  |         173.0  |         174.0  |
| hpx_server   | 4 |       330.0 |       365.0 |       365.0 |         329.0  |         330.0  |         331.0  |
| llama_server | 1 |        93.0 |        94.0 |        97.6 |          98.0  |         104.6  |         121.8  |
| llama_server | 2 |       176.0 |       177.0 |       177.0 |         181.0  |         185.1  |         186.0  |
| llama_server | 4 |       298.0 |       299.0 |       315.0 |         304.0  |         307.0  |         308.0  |

Descriptive notes (neutral):

- At `c=1` and `c=2`, the streaming p50 is within ≤ 5 ms of the
  non-streaming p50 on both servers, with the non-streaming column
  recording a slightly higher value on llama-server at `c=1` / `c=2`
  and an essentially identical value on hpx-server.
- At `c=4`, the recorded p50 values are essentially identical on
  hpx-server (330.0 streaming vs 329.0 non-streaming) and the
  llama-server streaming p50 is 6.0 ms below its non-streaming p50
  (298.0 vs 304.0).
- The hpx-server `c=4` streaming p95/p99 is ≈ 35 ms above its
  non-streaming p95/p99 (365 vs 330–331), while the llama-server
  `c=4` streaming p95 is ≈ 8 ms below its non-streaming p95
  (299 vs 307). The streaming and non-streaming tails differ in
  opposite directions on the two servers; the client-side data
  alone does not identify a cause.
- The streaming path adds per-token SSE framing on top of decode,
  so a 1:1 latency match is not expected; the recorded c=1/c=2 values
  are nonetheless close to the non-streaming values on this shape.

### Inter-event gaps at c=4 (per-client)

Adjacent differences of `token_event_times_ms` per row, per
client (`n = 7 × 50 = 350` gaps per client per cell):

| server       | client_id |  n  | min |  p50 |  p95 | max |
|:-------------|----------:|----:|----:|-----:|-----:|----:|
| hpx_server   |     0     | 350 |  20 | 29.0 | 99.0 | 100 |
| hpx_server   |     1     | 350 |  15 | 29.0 | 30.0 |  30 |
| hpx_server   |     2     | 350 |  15 | 29.0 | 30.0 |  30 |
| hpx_server   |     3     | 350 |  29 | 29.0 | 99.0 | 100 |
| llama_server |     0     | 350 |  19 | 30.0 | 30.0 |  34 |
| llama_server |     1     | 350 |  19 | 30.0 | 30.0 |  35 |
| llama_server |     2     | 350 |  19 | 30.0 | 30.0 |  34 |
| llama_server |     3     | 350 |  29 | 30.0 | 63.0 |  63 |

The flattened p50 is ≈ 29 ms (hpx) / 30 ms (llama) on every
client. The per-client p95 is asymmetric: on hpx-server, clients
0 and 3 record p95 ≈ 99 ms while clients 1 and 2 record p95
≈ 30 ms; on llama-server, clients 0/1/2 record p95 ≈ 30 ms while
`client_id=3` records p95 ≈ 63 ms. The wider-tailed clients on the
two servers are not the same client_ids. Client-observed gaps
include the harness's per-line SSE read loop and any network /
stream-framing time, and do not isolate any server-internal step.
This is recorded descriptively only.

### Correctness re-check

Recomputed across all 714 N4 streaming rows (warm-up included):

- hpx-server `hash == 0x0619d4d1900c2365` at every `c=1` / `c=2` /
  `c=4` row; single value, no off-anchor rows.
- llama-server `hash` field is empty on every row (llama-server
  does not emit this field on the SSE path), so no cross-server
  hash comparison is made.
- `text_normalized_sha256` unique (stable) within every cell of
  both servers (hpx cells:
  `c2cb4b4ddddeab5d4b1148e91efa08cd3c098e8816d5ffeb505914290056899e`;
  llama cells:
  `4bc48f3540cf5ed22e1485a46878b848af8e8bf6ce72bf9c145fe0e2276fe222`).
- `n_decoded == 8` on every row.
- `done_seen == true`, `token_event_count > 0`,
  `stream_parse_error == ""`, `parse_error == ""`,
  `failed_reason == ""`, `http_status == 200` on every row.

No correctness drift observed during this analysis.

### Interpretation (cautious)

- The N=4 streaming `p0_b8` result is correct and stable: the
  canonical hpx-server `p0_b8` anchor `0x0619d4d1900c2365` holds
  across all 357 hpx rows, per-cell `text_normalized_sha256` is
  stable, and every row completes with HTTP 200, `done_seen`, and
  `n_decoded == 8`.
- At `c=1` and `c=2`, streaming `total_latency_ms` p50 tracks
  non-streaming `latency_ms` p50 to within ≤ 5 ms on both servers,
  matching the pattern from Phase 2b-N4-S1.
- At `c=4`, the per-row `latency_ms` is essentially symmetric
  across the four clients (≈ 330 ms hpx, ≈ 298 ms llama on every
  client), and the four request windows of each iteration overlap
  on every iteration, mirroring the non-streaming `c=4` clustered-
  overlap pattern.
- The streaming `c=4` cell exposes a per-client
  `first_token_latency_ms` asymmetry that the non-streaming view
  does not surface (it has no `first_token_latency_ms` field). The
  asymmetry tracks the ≈ 20 ms client-side submit stagger: the
  early-submitter (`client_id=3`) records a lower
  `first_token_latency_ms` p50 on both servers, while the three
  later-submitters cluster at the cell median. The cell-level
  `first_token_latency_ms` p50 is therefore driven by the
  3-of-4-client cohort and is not a single per-decode-step value.
- The streaming `c=4` per-client inter-event gap p95 is asymmetric
  in opposite ways on the two servers (hpx wider on clients 0/3,
  llama wider on client 3). The client-side data shows the pattern
  but cannot identify the server-internal cause.

Client-side data can show clustering, overlap, per-client
symmetry/asymmetry, and tail grouping, but cannot identify the
server-internal cause of any of these.

### Non-claims (analysis subsection)

- No production-throughput claim. The recorded `latency_ms`,
  `first_token_latency_ms`, and inter-event-gap values are
  client-observed, include loopback and SSE framing, and are not
  server-throughput figures.
- No scheduler-fairness claim. Per-client tables show client-side
  symmetry (in `latency_ms`) and client-side asymmetry (in
  `first_token_latency_ms` and per-client gap p95), but these are
  one prompt / one budget / one hardware / one model / one N — not
  a fairness characterization.
- No server-internal causal claim for the per-client
  `first_token_latency_ms` asymmetry, the streaming vs
  non-streaming p95 / p99 differences at `c=4`, or the per-client
  inter-event-gap differences.
- No semantic-equality claim. The streaming detokenization
  divergence (and every other Phase 0 mismatch) means the two
  servers' streamed bytes are not equal even on `p0_b8`; it
  remains record-only.
- No `p0_b32`-streaming-at-N=4 claim; the run has not been
  performed.
- No `c=8` claim and no `N=8` claim.
- No Llama 3 / larger-model claim.
- No generalization beyond this machine (Apple M4 Pro, darwin),
  this model (TinyLlama 1.1B Q4_K_M), this prompt, this budget
  (8), greedy sampling, or `c ∈ {1, 2, 4}` at `N=4`.

# Experiment 15 — Llama3-S0 results (p0_b8 non-streaming, N=1 c=1, semantic-anchor smoke)

Status: **PASS.** First Llama 3 model-generalization probe in this
experiment. A small `p0_b8` **non-streaming** semantic-anchor smoke
on `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` at `N=1`, `c=1`. Both
servers admitted every request with HTTP 200; `n_decoded == 8` on
every row; per-server `text_normalized_sha256` stable across all
rows; hpx-server hash stable across all rows (no canonical Llama 3
anchor was pinned in the harness, so the gate ran in stability-only
mode); no timeouts, no stuck processes; forbidden-word audit clean.

This run records a **candidate** hpx-server Llama 3 `p0_b8` hash and
text-hash signature; nothing is canonicalized in the harness yet.

## S0 run

- Driver: `concurrent_bench.py` unchanged. The harness already
  treats `(workload_id, decode_budget)` not present in its
  `CANONICAL_HPX_HASH` table as stability-only — see line 712 of
  `concurrent_bench.py` (`"note: no canonical hash pinned …;
  stability-only gate"`). To avoid colliding with the TinyLlama
  `p0_b8` anchor, this config uses
  `workload["workload_id"] = "p0_b8_llama3"` and
  `workload["is_canonical_anchor"] = false`. No harness code change
  was needed.
- Config (new):
  `hpx-bench/experiments/15_hpx_vs_llama_server_semantics/config.llama3.semantic.p0_b8.json`
- `run_id`: `20260525-161913-c-smoke`
- Result dir: `results/20260525-161913-c-smoke/`
- Driver capture: `local/runs/exp15/llama3-semantic-p0_b8/driver.{stdout,stderr}`

## Shape

```text
model                   /Users/unick/Desktop/hpx/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
                        (Llama 3.1 8B Instruct, Q4_K_M, ~4.9 GB on disk)
workload                p0_b8_llama3 (prompt "Hello, my name is", decode_budget=8)
mode                    non-streaming
concurrency_levels      [1]
iterations_per_client   6
warm-up rule            iteration == 1 per client is warm-up for any aggregate;
                        correctness/stability gates use all rows
server order            hpx-server first, then llama-server
server lifecycle        each server booted once per run; one cell each
server settings         hpx-server   --n-seq-max 1 --max-concurrent 1
                                     --max-prompt-tokens 512 --n-threads 2
                                     --ctx-size 2048 (no --engine-pool → default OFF)
                        llama-server --parallel 1 --threads 2 --ctx-size 2048
                                     --no-context-shift --seed 0
```

## Per-server completion summary

| server       | c | rows | completed | failed | timeouts | batch_wall_clock_ms | client_observed_rps (approx) |
|:-------------|--:|-----:|----------:|-------:|---------:|--------------------:|-----------------------------:|
| hpx_server   | 1 |  6   |     6     |   0    |    0     |        3593         | 1.670                        |
| llama_server | 1 |  6   |     6     |   0    |    0     |        3563         | 1.684                        |

`client_observed_rps` is approximate / client-observed / includes
loopback / **not** a server-throughput claim. llama-server
`server_status` is `"stop"` (its label for hitting the requested
decode budget); `n_decoded == 8` on every row confirms the decode
completed as configured.

## Gate status

```text
gate: all rows http=200, no parse errors, no failed reasons, n_decoded == 8
note: no canonical hash pinned for workload=p0_b8_llama3 budget=8; stability-only gate
gate: hpx_server   c=1 text_normalized_sha256 stable: 3494b8623e8bc760…
gate: llama_server c=1 text_normalized_sha256 stable: 3494b8623e8bc760…
S0_GATES: PASS
```

Re-verified independently over all 12 rows of
`client_results.jsonl`:

- All rows returned HTTP 200 with `parse_error == ""` and
  `failed_reason == ""`.
- `n_decoded == 8` on every row (both servers).
- `server_status`: hpx-server `"completed"` on 6/6 rows;
  llama-server `"stop"` on 6/6 rows (its label for hitting the
  requested decode budget).
- `text_len_chars == 21` on every row, both servers.
- hpx-server `hash == "0xcaaa7ff70cbabd53"` on every hpx row
  (6/6, including warm-up). Single value, no off-anchor rows.
- llama-server `hash` field is empty on every row (llama-server
  does not emit this field on the non-streaming completion
  response in this configuration). Cross-server hash equality is
  therefore not checked.
- `text_normalized_sha256 ==
  "3494b8623e8bc760884545ec06e87931726ee87df4fd470768b7ffdff339e0af"`
  on every row of both servers.
- No timeouts. No stuck server process (`pgrep` clean before and
  after).
- `run_notes.txt` forbidden-word audit clean
  (`forbidden_word_hits=[]`).

## Candidate HPX Llama 3 `p0_b8` anchor (not canonicalized)

This run produced a single stable hpx-server hash across all 6
rows:

```text
candidate hpx-server hash for Llama 3.1 8B Q4_K_M, p0_b8, greedy:
  0xcaaa7ff70cbabd53
```

This is **recorded as a candidate**, not added to the harness's
`CANONICAL_HPX_HASH` table. Canonicalization is deferred to a
separate approved slice once the result is reproduced under more
than one matrix point (e.g., N=2 or higher iteration count).

## llama-server output stability

The llama-server non-streaming response on this configuration does
not include the hpx-style 64-bit hash, but it does emit
`text_normalized_sha256`. Across the 6 llama-server rows, both
`text_normalized_sha256` and `text_len_chars` were a single stable
value:

```text
llama-server p0_b8 (Llama 3.1 8B Q4_K_M, greedy, N=1, c=1):
  text_normalized_sha256 = 3494b8623e8bc760884545ec06e87931726ee87df4fd470768b7ffdff339e0af
  text_len_chars         = 21
```

## Cross-server text observation (record-only, not gated)

The hpx-server and llama-server `text_normalized_sha256` happened
to coincide on this run
(`3494b8623e8bc760884545ec06e87931726ee87df4fd470768b7ffdff339e0af`
on both). Cross-server text equality is **not** a gate in this
experiment (and remains explicitly disclaimed for streaming),
because the two servers use independent detokenization paths and
the TinyLlama streaming runs documented a 30 vs 37 char
divergence. The matching non-streaming `p0_b8` Llama 3 text-hash
is recorded here as an observation only; it is not asserted as a
property of either server.

## Server / process state

- hpx-server: `exit=-15` (SIGTERM-driven shutdown), `pid_still_running=False`.
- llama-server: `exit=0`, `pid_still_running=False`.
- No stuck `llama-server` or `llama-hpx-server` after the run
  (`pgrep` clean before and after).

## Forbidden-word audit

`run_notes.txt` clean (`forbidden_word_hits=[]`). Document hits in
`readme.md` / `facts.md` / `results.md` are confined to the
explicit audit-rule prose.

## Valid claims (Llama3-S0)

- A Llama 3.1 8B Q4_K_M `p0_b8` non-streaming semantic-anchor
  smoke completed on this machine under `N=1`, `c=1`,
  `iterations_per_client=6`.
- The hpx-server output on `p0_b8` Llama 3 was stable across all 6
  rows: a single `hash` (`0xcaaa7ff70cbabd53`), a single
  `text_normalized_sha256`, and a single `text_len_chars` value,
  with `n_decoded == 8` on every row.
- The llama-server output on the same shape was stable across all
  6 rows: a single `text_normalized_sha256`, a single
  `text_len_chars`, and `n_decoded == 8` on every row.
- A candidate hpx-server Llama 3 `p0_b8` hash
  (`0xcaaa7ff70cbabd53`) is recorded for future reference.
- The existing harness expressed the Llama 3 model path and the
  unknown-anchor case via the existing config schema and the
  existing stability-only fallback in
  `concurrent_bench.py:712`; no code change was needed.

## Non-claims (Llama3-S0)

- No performance claim. `batch_wall_clock_ms` and
  `client_observed_rps` are client-observed, include loopback and
  the cold-boot warm-up iteration, and are not server-throughput
  figures.
- No concurrency claim. Only `c=1` was exercised.
- No streaming claim. Only the non-streaming path was exercised.
- No `N=2` / `N=4` claim. Only `N=1` was exercised.
- No thread-scaling claim. `--threads 2` is fixed.
- No semantic-equality claim. The coincident
  `text_normalized_sha256` between the two servers on this
  non-streaming shape is recorded as an observation only; it is
  not asserted as a server-internal property and is not extended
  to other shapes, modes, or prompts.
- No canonical-anchor claim. `0xcaaa7ff70cbabd53` is a candidate
  from this single smoke and is **not** added to the harness's
  `CANONICAL_HPX_HASH` table.
- No generalization beyond this machine (Apple M4 Pro, darwin),
  this model (Llama 3.1 8B Q4_K_M), this prompt
  (`"Hello, my name is"`), this budget (8), greedy sampling, or
  `c == N == 1` admission.
- No `p0_b32` Llama 3 claim, no Llama 3 streaming claim, no Llama
  3 N=2 / N=4 claim.

# Experiment 15 — Llama3-S1 results (p0_b8 non-streaming, N=2 admitted-parallelism smoke)

Status: **PASS.** Llama 3 admitted-parallelism probe at `N=2`,
non-streaming `p0_b8` smoke on
`Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`. The Llama3-S0 candidate
hpx-server hash `0xcaaa7ff70cbabd53` held across all 18 hpx rows
at `c=1` and `c=2`. Per-`(server, concurrency)`
`text_normalized_sha256` stable across every cell of both servers
at the same Llama 3 `p0_b8` value recorded in S0. Every row
returned HTTP 200; `n_decoded == 8` on every row; no timeouts; no
stuck processes; forbidden-word audit clean. The candidate hash
is still **not** added to the harness's `CANONICAL_HPX_HASH`
table.

## S1 run

- Driver: `concurrent_bench.py` unchanged. Continues to use the
  stability-only fallback for `workload_id="p0_b8_llama3"`,
  `decode_budget=8` (no canonical Llama 3 anchor pinned).
- Config (new):
  `hpx-bench/experiments/15_hpx_vs_llama_server_semantics/config.llama3.n2.smoke.p0_b8.json`
  (mirrors `config.llama3.semantic.p0_b8.json` exactly except
  `hpx n_seq_max / max_concurrent = 2`, `llama parallel = 2`, and
  `concurrency_levels = [1, 2]`).
- `run_id`: `20260525-162327-c-smoke`
- Result dir: `results/20260525-162327-c-smoke/`
- Driver capture: `local/runs/exp15/llama3-n2-smoke-p0_b8/driver.{stdout,stderr}`

## Shape

```text
model                   /Users/unick/Desktop/hpx/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
                        (Llama 3.1 8B Instruct, Q4_K_M)
workload                p0_b8_llama3 (prompt "Hello, my name is", decode_budget=8)
mode                    non-streaming
concurrency_levels      [1, 2]
iterations_per_client   6 (smoke; not a timing result)
warm-up rule            iteration == 1 per client is warm-up for any aggregate;
                        correctness/stability gates use all rows
server order            hpx-server first, then llama-server
server lifecycle        each server booted once per run; both cells share that boot
server settings         hpx-server   --n-seq-max 2 --max-concurrent 2
                                     --max-prompt-tokens 512 --n-threads 2
                                     --ctx-size 2048 (no --engine-pool → default OFF)
                        llama-server --parallel 2 --threads 2 --ctx-size 2048
                                     --no-context-shift --seed 0
```

## Per-cell completion summary

| server       | c | rows | completed | failed | timeouts | batch_wall_clock_ms | client_observed_rps (approx) |
|:-------------|--:|-----:|----------:|-------:|---------:|--------------------:|-----------------------------:|
| hpx_server   | 1 |   6  |     6     |   0    |    0     |        3554         | 1.688                        |
| hpx_server   | 2 |  12  |    12     |   0    |    0     |        6387         | 1.879                        |
| llama_server | 1 |   6  |     6     |   0    |    0     |        3568         | 1.682                        |
| llama_server | 2 |  12  |    12     |   0    |    0     |        5743         | 2.090                        |

`client_observed_rps` is approximate / client-observed / includes
loopback / **not** a server-throughput claim. llama-server
`server_status` is `"stop"` (its label for hitting the requested
decode budget); `n_decoded == 8` on every row confirms the decode
completed as configured.

Sanity `latency_ms` p50 across all rows (warm-up included; not a
timing result):

| server       | c |  n |  min |   p50 |  max |
|:-------------|--:|---:|-----:|------:|-----:|
| hpx_server   | 1 |  6 |  584 |  589.0|  611 |
| hpx_server   | 2 | 12 |  774 | 1121.0| 1123 |
| llama_server | 1 |  6 |  590 |  593.5|  601 |
| llama_server | 2 | 12 |  781 |  935.5| 1129 |

These are sanity values only and are not aggregated as a timing
result. The hpx-server `c=1` p50 (589.0 ms) tracks the Llama3-S0
`c=1` shape on the same model and prompt; the `c=2` cells are
new under this run.

## Gate status

```text
gate: all rows http=200, no parse errors, no failed reasons, n_decoded == 8
note: no canonical hash pinned for workload=p0_b8_llama3 budget=8; stability-only gate
gate: hpx_server   c=1 text_normalized_sha256 stable: 3494b8623e8bc760…
gate: hpx_server   c=2 text_normalized_sha256 stable: 3494b8623e8bc760…
gate: llama_server c=1 text_normalized_sha256 stable: 3494b8623e8bc760…
gate: llama_server c=2 text_normalized_sha256 stable: 3494b8623e8bc760…
S0_GATES: PASS
```

Re-verified independently over all 36 rows of
`client_results.jsonl`:

- All rows returned HTTP 200 with `parse_error == ""` and
  `failed_reason == ""`.
- `n_decoded == 8` on every row.
- `server_status`: hpx-server `"completed"` on 18/18 rows;
  llama-server `"stop"` on 18/18 rows.
- `text_len_chars == 21` on every row, both servers, both cells.
- hpx-server `hash == "0xcaaa7ff70cbabd53"` on every hpx row at
  `c=1` (6/6) and at `c=2` (12/12). Single value, no off-anchor
  rows, no within-cell variation.
- llama-server `hash` field is empty on every row (llama-server
  does not emit this field on the non-streaming completion
  response in this configuration); cross-server hash equality is
  therefore not checked.
- `text_normalized_sha256 ==
  "3494b8623e8bc760884545ec06e87931726ee87df4fd470768b7ffdff339e0af"`
  on every row of both servers (matching the Llama3-S0 value).
- No timeouts. No stuck server process (`pgrep` clean before and
  after).
- `run_notes.txt` forbidden-word audit clean
  (`forbidden_word_hits=[]`).

## HPX candidate hash at N=2 (per cell)

```text
hpx_server c=1:  6/6 rows  hash 0xcaaa7ff70cbabd53
hpx_server c=2: 12/12 rows hash 0xcaaa7ff70cbabd53
```

**Candidate anchor held at N=2.** The Llama3-S0 hpx-server
candidate `p0_b8` hash `0xcaaa7ff70cbabd53` reappeared across all
18 hpx rows of this run, with no off-candidate rows and no
within-cell variation. The candidate is still **not** added to
the harness's `CANONICAL_HPX_HASH` table; that step is deferred to
a separate approved slice.

## llama-server output stability

```text
llama_server c=1:  6/6 rows  text_normalized_sha256 = 3494b8623e8bc760…  text_len_chars = 21
llama_server c=2: 12/12 rows text_normalized_sha256 = 3494b8623e8bc760…  text_len_chars = 21
```

llama-server's non-streaming response does not include the
hpx-style 64-bit `hash` field on this configuration, but the
`text_normalized_sha256` is a single stable value across every
cell, matching the Llama3-S0 value (`3494b86…`). `text_len_chars`
is 21 on every row.

## Cross-server text observation (record-only, not gated)

Cross-server `text_normalized_sha256` continues to coincide on the
non-streaming Llama 3 `p0_b8` shape:

```text
all cells of both servers (S1): text_normalized_sha256 = 3494b8623e8bc760…  text_len_chars = 21
```

The Llama3-S0 observation extends to `N=2`, `c=1` and `c=2`. This
remains **record-only**, not a semantic-equality claim and not a
gate. Streaming-mode cross-server byte equality is still
explicitly disclaimed and would not be expected to hold given the
TinyLlama streaming detokenization mismatch.

## Server / process state

- hpx-server: `exit=-15` (SIGTERM-driven shutdown), `pid_still_running=False`.
- llama-server: `exit=0`, `pid_still_running=False`.
- No stuck `llama-server` or `llama-hpx-server` after the run
  (`pgrep` clean before and after).

## Forbidden-word audit

`run_notes.txt` clean (`forbidden_word_hits=[]`). Document hits in
`readme.md` / `facts.md` / `results.md` are confined to the
explicit audit-rule prose.

## Valid claims (Llama3-S1)

- A Llama 3.1 8B Q4_K_M `p0_b8` non-streaming `N=2`
  admitted-parallelism smoke completed on this machine at
  `c ∈ {1, 2}`, `iterations_per_client=6`.
- The Llama3-S0 candidate hpx-server `p0_b8` hash
  (`0xcaaa7ff70cbabd53`) held across all 18 hpx rows of this run,
  at `c=1` and `c=2`.
- Per-`(server, concurrency)` `text_normalized_sha256` is stable
  across every cell of both servers, at the same value recorded
  in Llama3-S0.
- The existing harness expressed the `N=2` admitted-parallelism
  Llama 3 shape via the existing config schema and the existing
  stability-only fallback in `concurrent_bench.py:712`; no code
  change was needed.

## Non-claims (Llama3-S1)

- No performance claim. `batch_wall_clock_ms`,
  `client_observed_rps`, and the sanity `latency_ms` table are
  client-observed, include loopback and the cold-boot warm-up
  iteration, and are not server-throughput figures or a timing
  result.
- No streaming claim. Only the non-streaming path was exercised.
- No `N=4` claim. Only `N=2` was exercised.
- No `c=4` claim. Only `c ∈ {1, 2}` was exercised.
- No thread-scaling claim. `--threads 2` is fixed.
- No semantic-equality claim. The coincident
  `text_normalized_sha256` between the two servers on this
  non-streaming shape continues to be recorded as an observation
  only; it is not asserted as a server-internal property and is
  not extended to other shapes, modes, or prompts.
- No canonical-anchor claim. `0xcaaa7ff70cbabd53` is still a
  candidate from S0 confirmed at S1 and is **not** added to the
  harness's `CANONICAL_HPX_HASH` table.
- No `p0_b32` Llama 3 claim, no Llama 3 streaming claim, no Llama
  3 `N=4` claim.
- No generalization beyond this machine (Apple M4 Pro, darwin),
  this model (Llama 3.1 8B Q4_K_M), this prompt
  (`"Hello, my name is"`), this budget (8), greedy sampling, or
  `c ∈ {1, 2}` at `N=2`.

# Experiment 15 — Llama3-S2 results (p0_b8 non-streaming, N=2 timing baseline)

Status: **PASS.** Llama 3 timing-sized baseline at `N=2`,
non-streaming `p0_b8` on
`Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` with
`iterations_per_client = 51`. Every row returned HTTP 200;
`n_decoded == 8` on every row; the Llama3-S0/S1 candidate
hpx-server hash `0xcaaa7ff70cbabd53` held across all 153 hpx
rows at `c=1` and `c=2`; per-`(server, concurrency)`
`text_normalized_sha256` is stable across every cell at the same
Llama 3 `p0_b8` value recorded in S0; no timeouts; no stuck
processes; forbidden-word audit clean. The candidate hash is
still **not** added to the harness's `CANONICAL_HPX_HASH` table.

## S2 run

- Driver: `concurrent_bench.py` unchanged. Continues to use the
  stability-only fallback for `workload_id="p0_b8_llama3"`,
  `decode_budget=8` (no canonical Llama 3 anchor pinned).
- Config (new):
  `hpx-bench/experiments/15_hpx_vs_llama_server_semantics/config.llama3.n2.p0_b8.json`
  (mirrors `config.llama3.n2.smoke.p0_b8.json` exactly except
  `iterations_per_client: 51` in place of `6`).
- `run_id`: `20260525-162757-c-smoke`
- Result dir: `results/20260525-162757-c-smoke/`
- Driver capture: `local/runs/exp15/llama3-n2-p0_b8/driver.{stdout,stderr}`

## Shape

```text
model                   /Users/unick/Desktop/hpx/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
                        (Llama 3.1 8B Instruct, Q4_K_M)
workload                p0_b8_llama3 (prompt "Hello, my name is", decode_budget=8)
mode                    non-streaming
concurrency_levels      [1, 2]
iterations_per_client   51
warm-up rule            iteration == 1 per client is warm-up for timing aggregates;
                        correctness/stability gates use all rows.
timing-row counts:
  c=1, per server:      n = 50          (1 client × (51 − 1) iters)
  c=2, per server:      n = 100         (2 clients × (51 − 1) iters each)
  warm-up rows dropped: 3 per server    (1 at c=1 + 2 at c=2)

server order            hpx-server first, then llama-server
server lifecycle        each server booted once per run; both cells share that boot
server settings         hpx-server   --n-seq-max 2 --max-concurrent 2
                                     --max-prompt-tokens 512 --n-threads 2
                                     --ctx-size 2048 (no --engine-pool → default OFF)
                        llama-server --parallel 2 --threads 2 --ctx-size 2048
                                     --no-context-shift --seed 0
```

## Per-cell completion summary

| server       | c | rows | completed | failed | timeouts | batch_wall_clock_ms | client_observed_rps (approx) |
|:-------------|--:|-----:|----------:|-------:|---------:|--------------------:|-----------------------------:|
| hpx_server   | 1 |  51  |    51     |   0    |    0     |       29968         | 1.702                        |
| hpx_server   | 2 | 102  |   102     |   0    |    0     |       57419         | 1.776                        |
| llama_server | 1 |  51  |    51     |   0    |    0     |       30099         | 1.694                        |
| llama_server | 2 | 102  |   102     |   0    |    0     |       56160         | 1.816                        |

`client_observed_rps` is approximate / client-observed / includes
loopback / **not** a server-throughput claim. llama-server
`server_status` is `"stop"` (its label for hitting the requested
decode budget); `n_decoded == 8` on every row confirms the decode
completed as configured.

## Gate status

```text
gate: all rows http=200, no parse errors, no failed reasons, n_decoded == 8
note: no canonical hash pinned for workload=p0_b8_llama3 budget=8; stability-only gate
gate: hpx_server   c=1 text_normalized_sha256 stable: 3494b8623e8bc760…
gate: hpx_server   c=2 text_normalized_sha256 stable: 3494b8623e8bc760…
gate: llama_server c=1 text_normalized_sha256 stable: 3494b8623e8bc760…
gate: llama_server c=2 text_normalized_sha256 stable: 3494b8623e8bc760…
S0_GATES: PASS
```

Re-verified independently over all 306 rows of
`client_results.jsonl`:

- All rows returned HTTP 200 with `parse_error == ""` and
  `failed_reason == ""`.
- `n_decoded == 8` on every row.
- `server_status`: hpx-server `"completed"` on 153/153 rows;
  llama-server `"stop"` on 153/153 rows.
- `text_len_chars == 21` on every row, both servers, both cells.
- hpx-server `hash == "0xcaaa7ff70cbabd53"` on every hpx row at
  `c=1` (51/51) and at `c=2` (102/102). Single value, no
  off-candidate rows, no within-cell variation.
- llama-server `hash` field is empty on every row (llama-server
  does not emit this field on the non-streaming completion
  response in this configuration); cross-server hash equality is
  therefore not checked.
- `text_normalized_sha256 ==
  "3494b8623e8bc760884545ec06e87931726ee87df4fd470768b7ffdff339e0af"`
  on every row of both servers (matching the Llama3-S0 / Llama3-S1
  value).
- No timeouts. No stuck server process (`pgrep` clean before and
  after).
- `run_notes.txt` forbidden-word audit clean
  (`forbidden_word_hits=[]`).

## HPX candidate hash confirmation (per cell)

```text
hpx_server c=1:  51/51 rows  hash 0xcaaa7ff70cbabd53
hpx_server c=2: 102/102 rows hash 0xcaaa7ff70cbabd53
```

**Candidate anchor held in timing-sized N=2 baseline.** The
Llama3-S0 candidate, confirmed in Llama3-S1, reappeared across all
153 hpx rows of this 51-iter/client run, including the warm-up
rows, with no off-candidate rows and no within-cell variation. The
candidate is still **not** added to the harness's
`CANONICAL_HPX_HASH` table; canonicalization remains a separately
approved slice.

## llama-server stability confirmation

```text
llama_server c=1:  51/51 rows  text_normalized_sha256 = 3494b8623e8bc760…  text_len_chars = 21
llama_server c=2: 102/102 rows text_normalized_sha256 = 3494b8623e8bc760…  text_len_chars = 21
```

llama-server's non-streaming response continues to omit the
hpx-style 64-bit `hash` field on this configuration, but the
`text_normalized_sha256` is a single stable value across every
cell, matching the Llama3-S0 / Llama3-S1 value
(`3494b86…`). `text_len_chars` is 21 on every row.

## Timing table

Computed over `iteration >= 2` rows per client (warm-up dropped);
`n = 50` per cell at `c=1`, `n = 100` per cell at `c=2`.
Percentiles use linear interpolation on the sorted sample.

| server       | c |   n |   latency_ms p50 |   latency_ms p95 |   latency_ms p99 |   min |   max | batch_wall_clock_ms |
|:-------------|--:|----:|-----------------:|-----------------:|-----------------:|------:|------:|--------------------:|
| hpx_server   | 1 |  50 |            586.0 |            593.0 |            595.0 |   584 |   596 |               29968 |
| hpx_server   | 2 | 100 |           1123.0 |           1135.0 |           1137.0 |  1121 |  1137 |               57419 |
| llama_server | 1 |  50 |            589.0 |            592.0 |            596.5 |   588 |   597 |               30099 |
| llama_server | 2 | 100 |           1131.0 |           1154.0 |           1173.0 |   799 |  1173 |               56160 |

Descriptive notes (neutral):

- At `c=1`, recorded p50 differs by ≤ 3 ms across the two servers
  on this shape (hpx 586.0, llama 589.0). Per-cell `min`–`max`
  spread is tight on both (12–9 ms).
- At `c=2`, recorded p50 differs by 8 ms across the two servers
  (hpx 1123.0, llama 1131.0). The hpx-server `min`–`max` spread
  at `c=2` is 16 ms; the llama-server `min`–`max` spread at `c=2`
  is 374 ms because of a single low outlier
  (`latency_ms = 799 ms`); the llama-server p95 and p99 sit
  closer to the cell median (1154.0 and 1173.0).
- Per-server `c=2 / c=1` recorded p50 ratios: hpx ≈ 1.92, llama
  ≈ 1.92. Recorded descriptively, not interpreted as a server
  property.
- `batch_wall_clock_ms` at `c=1` is recorded within roughly
  0.4 % across the two servers (hpx 29968, llama 30099). At
  `c=2` it is within roughly 2.2 % (hpx 57419, llama 56160).

## Sanity comparison to Llama3-S1 smoke

The Llama3-S1 (`run_id=20260525-162327-c-smoke`) smoke used
`iterations_per_client=6`; this run uses 51. `batch_wall_clock_ms`
ratios are recorded as sanity only:

| server       | c | S1 wall ms (n=6) | S2 wall ms (n=51) | S2/S1 ratio |
|:-------------|--:|-----------------:|------------------:|------------:|
| hpx_server   | 1 |             3554 |             29968 |       8.43  |
| hpx_server   | 2 |             6387 |             57419 |       8.99  |
| llama_server | 1 |             3568 |             30099 |       8.44  |
| llama_server | 2 |             5743 |             56160 |       9.78  |

The naive linear expectation is `51 / 6 ≈ 8.50`. Recorded ratios
sit between `8.43` and `9.78`; the difference between the timing-
sized and smoke-sized wall-clocks is dominated by the constant
single-iteration cost and the larger denominator (warm-up is a
larger fraction of a 6-iter cell than of a 51-iter cell). This is
recorded as a sanity check that the timing run did not produce a
qualitatively different shape from the smoke; it is **not** a
performance comparison.

## Server / process state

- hpx-server: `exit=-15` (SIGTERM-driven shutdown), `pid_still_running=False`.
- llama-server: `exit=0`, `pid_still_running=False`.
- No stuck `llama-server` or `llama-hpx-server` after the run
  (`pgrep` clean before and after).

## Forbidden-word audit

`run_notes.txt` clean (`forbidden_word_hits=[]`). Document hits in
`readme.md` / `facts.md` / `results.md` are confined to the
explicit audit-rule prose.

## Valid claims (Llama3-S2)

- A Llama 3.1 8B Q4_K_M `p0_b8` non-streaming `N=2`
  timing-sized baseline completed on this machine at
  `c ∈ {1, 2}`, `iterations_per_client=51`.
- The Llama3-S0 / Llama3-S1 candidate hpx-server `p0_b8` hash
  (`0xcaaa7ff70cbabd53`) held across all 153 hpx rows of this
  timing-sized run, at `c=1` and `c=2`, with no off-candidate
  rows.
- Per-`(server, concurrency)` `text_normalized_sha256` is stable
  across every cell of both servers, at the same value recorded
  in Llama3-S0 / Llama3-S1.
- Single-machine, matched-N=2-admission `latency_ms` p50 / p95 /
  p99 values are recorded for hpx-server and llama-server on the
  `p0_b8` non-streaming shape at `c ∈ {1, 2}` over `n=50` /
  `n=100` non-warm-up samples per cell.
- Both servers complete every iteration of the matrix with HTTP
  200, stable `n_decoded`, and stable per-cell
  `text_normalized_sha256`, with no 503 and no timeout.
- The existing harness expressed the timing-sized N=2 Llama 3
  shape via the existing config schema and the existing
  stability-only fallback in `concurrent_bench.py:712`; no code
  change was needed.

## Non-claims (Llama3-S2)

- No production-throughput claim. `batch_wall_clock_ms`,
  `client_observed_rps`, and the timing table are
  client-observed, include loopback, and are not server-throughput
  figures.
- No winner / performance comparison. The latency table and the
  S1 sanity comparison are side-by-side recorded values, not a
  ranking.
- No claim that either server's admission policy is preferable,
  and no claim that either server scales preferentially with
  concurrency on this shape.
- No scheduler-fairness claim. This run does not measure
  cross-client tail latency or starvation.
- No semantic-equality claim. The coincident
  `text_normalized_sha256` between the two servers on this
  non-streaming Llama 3 shape continues to be a record-only
  observation and is not asserted as a server-internal property.
- No canonical-anchor claim. `0xcaaa7ff70cbabd53` remains a
  candidate confirmed at S0, S1, and S2; it is **not** added to
  the harness's `CANONICAL_HPX_HASH` table.
- No streaming claim, no `N=4` claim, no `c=4` claim, no
  `p0_b32` Llama 3 claim. Those remain separate design slices.
- No thread-scaling claim. `--threads 2` is fixed.
- No generalization beyond this machine (Apple M4 Pro, darwin),
  this model (Llama 3.1 8B Q4_K_M), this prompt
  (`"Hello, my name is"`), this budget (8), greedy sampling, or
  `c ∈ {1, 2}` at `N=2`.

# Experiment 15 — Llama3-S3 results (p0_b8 non-streaming, N=2 timing baseline, threads=4)

Status: **PASS.** Same Llama 3 timing-sized N=2 baseline shape as
Llama3-S2 but with both servers configured at
`--n-threads 4` / `--threads 4`. Every row returned HTTP 200;
`n_decoded == 8` on every row; the Llama3-S0 / S1 / S2 candidate
hpx-server hash `0xcaaa7ff70cbabd53` held across all 153 hpx
rows at `c=1` and `c=2`; per-`(server, concurrency)`
`text_normalized_sha256` is stable across every cell at the same
Llama 3 `p0_b8` value recorded in S0; no timeouts; no stuck
processes; forbidden-word audit clean. The candidate hash is
still **not** added to the harness's `CANONICAL_HPX_HASH` table.

## S3 run

- Driver: `concurrent_bench.py` unchanged. The harness reads the
  top-level `threads` field and wires it into the per-server argv
  (`--n-threads` for hpx-server,
  `concurrent_bench.py:518`; `--threads` for llama-server,
  `concurrent_bench.py:538`).
- Config (new):
  `hpx-bench/experiments/15_hpx_vs_llama_server_semantics/config.llama3.n2.p0_b8.threads4.json`
  (mirrors `config.llama3.n2.p0_b8.json` exactly except the
  top-level `threads` field is `4` in place of `2`).
- `run_id`: `20260525-163545-c-smoke`
- Result dir: `results/20260525-163545-c-smoke/`
- Driver capture: `local/runs/exp15/llama3-n2-p0_b8-threads4/driver.{stdout,stderr}`

Confirmed in `run_notes.txt` launch argv:
`hpx-server --n-threads 4`, `llama-server --threads 4`.

## Shape

```text
model                   /Users/unick/Desktop/hpx/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
                        (Llama 3.1 8B Instruct, Q4_K_M)
workload                p0_b8_llama3 (prompt "Hello, my name is", decode_budget=8)
mode                    non-streaming
concurrency_levels      [1, 2]
iterations_per_client   51
warm-up rule            iteration == 1 per client is warm-up for timing aggregates;
                        correctness/stability gates use all rows.
timing-row counts:
  c=1, per server:      n = 50          (1 client × (51 − 1) iters)
  c=2, per server:      n = 100         (2 clients × (51 − 1) iters each)
  warm-up rows dropped: 3 per server    (1 at c=1 + 2 at c=2)

server order            hpx-server first, then llama-server
server lifecycle        each server booted once per run; both cells share that boot
server settings         hpx-server   --n-seq-max 2 --max-concurrent 2
                                     --max-prompt-tokens 512 --n-threads 4
                                     --ctx-size 2048 (no --engine-pool → default OFF)
                        llama-server --parallel 2 --threads 4 --ctx-size 2048
                                     --no-context-shift --seed 0
```

## Per-cell completion summary

| server       | c | rows | completed | failed | timeouts | batch_wall_clock_ms | client_observed_rps (approx) |
|:-------------|--:|-----:|----------:|-------:|---------:|--------------------:|-----------------------------:|
| hpx_server   | 1 |  51  |    51     |   0    |    0     |       30164         | 1.691                        |
| hpx_server   | 2 | 102  |   102     |   0    |    0     |       57502         | 1.774                        |
| llama_server | 1 |  51  |    51     |   0    |    0     |       30149         | 1.692                        |
| llama_server | 2 | 102  |   102     |   0    |    0     |       59368         | 1.718                        |

`client_observed_rps` is approximate / client-observed / includes
loopback / **not** a server-throughput claim. llama-server
`server_status` is `"stop"` (its label for hitting the requested
decode budget); `n_decoded == 8` on every row confirms the decode
completed as configured.

## Gate status

```text
gate: all rows http=200, no parse errors, no failed reasons, n_decoded == 8
note: no canonical hash pinned for workload=p0_b8_llama3 budget=8; stability-only gate
gate: hpx_server   c=1 text_normalized_sha256 stable: 3494b8623e8bc760…
gate: hpx_server   c=2 text_normalized_sha256 stable: 3494b8623e8bc760…
gate: llama_server c=1 text_normalized_sha256 stable: 3494b8623e8bc760…
gate: llama_server c=2 text_normalized_sha256 stable: 3494b8623e8bc760…
S0_GATES: PASS
```

Re-verified independently over all 306 rows of
`client_results.jsonl`:

- All rows returned HTTP 200 with `parse_error == ""` and
  `failed_reason == ""`.
- `n_decoded == 8` on every row.
- `server_status`: hpx-server `"completed"` on 153/153 rows;
  llama-server `"stop"` on 153/153 rows.
- `text_len_chars == 21` on every row, both servers, both cells.
- hpx-server `hash == "0xcaaa7ff70cbabd53"` on every hpx row at
  `c=1` (51/51) and at `c=2` (102/102). Single value, no
  off-candidate rows, no within-cell variation.
- llama-server `hash` field is empty on every row (llama-server
  does not emit this field on the non-streaming completion
  response in this configuration); cross-server hash equality is
  therefore not checked.
- `text_normalized_sha256 ==
  "3494b8623e8bc760884545ec06e87931726ee87df4fd470768b7ffdff339e0af"`
  on every row of both servers (matching the Llama3-S0 / S1 / S2
  value).
- No timeouts. No stuck server process (`pgrep` clean before and
  after).
- `run_notes.txt` forbidden-word audit clean
  (`forbidden_word_hits=[]`).

## HPX candidate hash confirmation (per cell)

```text
hpx_server c=1:  51/51 rows  hash 0xcaaa7ff70cbabd53
hpx_server c=2: 102/102 rows hash 0xcaaa7ff70cbabd53
```

**Candidate anchor held under threads=4.** The Llama3-S0
candidate, confirmed at S1 and S2, reappeared across all 153 hpx
rows of this `--n-threads 4` run, including warm-up rows, with no
off-candidate rows and no within-cell variation. Doubling the
compute-thread count from 2 to 4 did not perturb the hpx-server
`p0_b8` hash. The candidate is still **not** added to the
harness's `CANONICAL_HPX_HASH` table.

## llama-server stability confirmation

```text
llama_server c=1:  51/51 rows  text_normalized_sha256 = 3494b8623e8bc760…  text_len_chars = 21
llama_server c=2: 102/102 rows text_normalized_sha256 = 3494b8623e8bc760…  text_len_chars = 21
```

llama-server's non-streaming response continues to omit the
hpx-style 64-bit `hash` field on this configuration, but the
`text_normalized_sha256` is a single stable value across every
cell, matching the Llama3-S0 / S1 / S2 value (`3494b86…`).
`text_len_chars` is 21 on every row.

## Timing table (threads=4)

Computed over `iteration >= 2` rows per client (warm-up dropped);
`n = 50` per cell at `c=1`, `n = 100` per cell at `c=2`.
Percentiles use linear interpolation on the sorted sample.

| server       | c |   n |   latency_ms p50 |   latency_ms p95 |   latency_ms p99 |   min |   max | batch_wall_clock_ms |
|:-------------|--:|----:|-----------------:|-----------------:|-----------------:|------:|------:|--------------------:|
| hpx_server   | 1 |  50 |            588.0 |            604.5 |            606.5 |   583 |   607 |               30164 |
| hpx_server   | 2 | 100 |           1131.0 |           1153.0 |           1157.0 |  1076 |  1157 |               57502 |
| llama_server | 1 |  50 |            590.0 |            593.0 |            602.2 |   588 |   611 |               30149 |
| llama_server | 2 | 100 |           1163.5 |           1267.0 |           1292.0 |   825 |  1293 |               59368 |

Per-server `c=2 / c=1` recorded p50 ratios: hpx ≈ 1.92, llama
≈ 1.97. Recorded descriptively, not interpreted as a server
property.

## Descriptive comparison: threads=2 (Llama3-S2) vs threads=4 (Llama3-S3)

Side-by-side `latency_ms` percentiles for matched cells across the
two runs. Δ columns are `S3 − S2` (positive means the recorded
threads=4 value sits above the recorded threads=2 value). Same
model, prompt, decode budget, N=2 admission, concurrency level,
and number of non-warm-up samples per cell.

| server       | c |   t2 p50 |   t4 p50 |   Δp50 |   t2 p95 |   t4 p95 |   Δp95 |   t2 p99 |   t4 p99 |   Δp99 |
|:-------------|--:|---------:|---------:|-------:|---------:|---------:|-------:|---------:|---------:|-------:|
| hpx_server   | 1 |    586.0 |    588.0 |   +2.0 |    593.0 |    604.5 |  +11.5 |    595.0 |    606.5 |  +11.5 |
| hpx_server   | 2 |   1123.0 |   1131.0 |   +8.0 |   1135.0 |   1153.0 |  +18.0 |   1137.0 |   1157.0 |  +20.0 |
| llama_server | 1 |    589.0 |    590.0 |   +1.0 |    592.0 |    593.0 |   +1.0 |    596.5 |    602.2 |   +5.7 |
| llama_server | 2 |   1131.0 |   1163.5 |  +32.5 |   1154.0 |   1267.0 | +113.0 |   1173.0 |   1292.0 | +119.0 |

`batch_wall_clock_ms` side-by-side:

| server       | c | S2 wall (t=2) | S3 wall (t=4) | Δ wall |
|:-------------|--:|--------------:|--------------:|-------:|
| hpx_server   | 1 |        29968  |        30164  |   +196 |
| hpx_server   | 2 |        57419  |        57502  |    +83 |
| llama_server | 1 |        30099  |        30149  |    +50 |
| llama_server | 2 |        56160  |        59368  |  +3208 |

Descriptive notes (neutral):

- At `c=1`, recorded p50 values move by ≤ 2 ms across the two
  thread settings on both servers; p95 / p99 move by ≤ 11.5 ms.
  The recorded change at low concurrency is small.
- At `c=2`, the hpx-server recorded p50 moves by +8.0 ms, p95 by
  +18.0 ms, p99 by +20.0 ms. The llama-server recorded p50 moves
  by +32.5 ms, p95 by +113.0 ms, p99 by +119.0 ms. The recorded
  llama-server tail change at `c=2` is the largest single delta
  in the table.
- `batch_wall_clock_ms` moves by ≤ 200 ms on three of the four
  cells (hpx `c=1` / hpx `c=2` / llama `c=1`). The llama-server
  `c=2` `batch_wall_clock_ms` recorded value sits +3208 ms above
  the threads=2 value (5.7 % above the S2 value).
- The recorded `c=2 / c=1` p50 ratio is ≈ 1.92 (hpx) and ≈ 1.97
  (llama) at threads=4; the same ratio at threads=2 was ≈ 1.92 /
  ≈ 1.92. The recorded admitted-parallelism shape is unchanged in
  direction.
- This is a recorded comparison across two single runs. The
  client-side data does not isolate any server-internal cause for
  the recorded threads=4 deltas (e.g., whether the change
  originates in compute-thread contention on this 8B model at
  this prompt/budget, in OS scheduling, in cache behavior, or
  elsewhere).

## Server / process state

- hpx-server: `exit=-15` (SIGTERM-driven shutdown), `pid_still_running=False`.
- llama-server: `exit=0`, `pid_still_running=False`.
- No stuck `llama-server` or `llama-hpx-server` after the run
  (`pgrep` clean before and after).

## Forbidden-word audit

`run_notes.txt` clean (`forbidden_word_hits=[]`). Document hits in
`readme.md` / `facts.md` / `results.md` are confined to the
explicit audit-rule prose.

## Valid claims (Llama3-S3)

- A Llama 3.1 8B Q4_K_M `p0_b8` non-streaming `N=2`
  timing-sized run completed on this machine at `c ∈ {1, 2}`,
  `iterations_per_client=51`, with both servers configured at
  `--n-threads 4` / `--threads 4`.
- The Llama3-S0 / S1 / S2 candidate hpx-server `p0_b8` hash
  (`0xcaaa7ff70cbabd53`) held across all 153 hpx rows of this
  threads=4 run, at `c=1` and `c=2`, with no off-candidate rows.
- Per-`(server, concurrency)` `text_normalized_sha256` is stable
  across every cell of both servers at the same value recorded in
  the threads=2 baseline.
- Single-machine `latency_ms` p50 / p95 / p99 and
  `batch_wall_clock_ms` values are recorded for hpx-server and
  llama-server at `c ∈ {1, 2}` under threads=4.
- Side-by-side recorded `Δp50 / Δp95 / Δp99 / Δ wall`
  values vs Llama3-S2 are documented above for matched cells.
- The existing harness expressed the threads=4 shape via the
  existing config schema (top-level `threads` field) and the
  existing argv builders; no code change was needed.

## Non-claims (Llama3-S3)

- No production-throughput claim. `batch_wall_clock_ms`,
  `client_observed_rps`, and the timing table are
  client-observed, include loopback, and are not server-throughput
  figures.
- No winner / performance comparison. The latency table and the
  S2-vs-S3 side-by-side are recorded values, not a ranking.
- No claim that `--n-threads 4` (or `--threads 4`) is preferable
  or non-preferable on this shape, on this machine, or in
  general. The recorded comparison is a single matched run; it
  is not a thread-scaling recommendation.
- No claim that either server's admission policy is preferable.
- No scheduler-fairness claim.
- No semantic-equality claim. The coincident
  `text_normalized_sha256` between the two servers on this
  non-streaming Llama 3 shape continues to be a record-only
  observation and is not asserted as a server-internal property.
- No canonical-anchor claim. `0xcaaa7ff70cbabd53` remains a
  candidate confirmed at S0, S1, S2, and S3; it is **not** added
  to the harness's `CANONICAL_HPX_HASH` table.
- No streaming claim, no `N=4` claim, no `c=4` claim, no
  `p0_b32` Llama 3 claim, no `threads=8` claim. Those remain
  separate design slices.
- No generalization beyond this machine (Apple M4 Pro, darwin),
  this model (Llama 3.1 8B Q4_K_M), this prompt
  (`"Hello, my name is"`), this budget (8), greedy sampling,
  `c ∈ {1, 2}` at `N=2`, or `--n-threads 4` / `--threads 4`.

## Llama 3 interim summary

Read-only roll-up of the Llama 3 results recorded so far in this
experiment (Llama3-S0 / S1 / S2 / S3). No new runs, no harness
changes, no harness `CANONICAL_HPX_HASH` modification, no
canonicalization step.

### 1. Correctness / stability summary

- Llama 3 model path used:
  `/Users/unick/Desktop/hpx/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`
  (Llama 3.1 8B Instruct, Q4_K_M).
- All four Llama 3 runs (S0, S1, S2, S3) passed: `S0_GATES: PASS`
  on each, and independent re-audit of each `client_results.jsonl`
  showed zero gate failures.
- No HTTP failures. Every row of every Llama 3 run returned
  `http_status == 200`.
- No parse errors. `parse_error == ""` on every row across all
  four runs.
- No `failed_reason` on any row across all four runs.
- No timeouts on any row across all four runs.
- No stuck server processes. `pgrep -lf 'llama-server|llama-hpx-server'`
  returned no servers before and after each run; the hpx-server
  always exited under SIGTERM, the llama-server always exited
  cleanly.
- HPX candidate hash remained stable at
  `0xcaaa7ff70cbabd53` across every hpx row of every Llama 3 run
  (semantic smoke S0, N=2 smoke S1, N=2 timing baseline S2 at
  threads=2, and N=2 timing baseline S3 at threads=4).
- HPX candidate hash is **still not added** to
  `concurrent_bench.py`'s `CANONICAL_HPX_HASH` table.
  Canonicalization remains a separately approved slice.
- llama-server `text_normalized_sha256` remained stable per cell
  across every Llama 3 run, at the single value
  `3494b8623e8bc760884545ec06e87931726ee87df4fd470768b7ffdff339e0af`.
  `text_len_chars == 21` on every row in every cell.
- Cross-server `text_normalized_sha256` happened to coincide on
  every cell of every Llama 3 run on this non-streaming `p0_b8`
  shape (both servers produced
  `3494b86…` with `text_len_chars = 21`). This continues to be
  **record-only**, not a semantic-equality gate, and not extended
  to streaming or other shapes / prompts.

### 2. Candidate anchor table

The hpx-server `p0_b8` candidate hash `0xcaaa7ff70cbabd53` has
held across every hpx row of every Llama 3 run to date:

| run        | run_id                       | shape                             | HPX rows |        hash |
|:-----------|:-----------------------------|:----------------------------------|---------:|------------:|
| Llama3-S0  | `20260525-161913-c-smoke`    | N=1, c=1, threads=2, 6 iters      |        6 | `0xcaaa7ff70cbabd53` |
| Llama3-S1  | `20260525-162327-c-smoke`    | N=2, c={1,2}, threads=2, 6 iters  |       18 | `0xcaaa7ff70cbabd53` |
| Llama3-S2  | `20260525-162757-c-smoke`    | N=2, c={1,2}, threads=2, 51 iters |      153 | `0xcaaa7ff70cbabd53` |
| Llama3-S3  | `20260525-163545-c-smoke`    | N=2, c={1,2}, threads=4, 51 iters |      153 | `0xcaaa7ff70cbabd53` |

Total: 330 hpx rows, every one carrying the same candidate hash.
No off-candidate row in any Llama 3 cell to date.

### 3. Timing table (threads=2 vs threads=4, N=2)

Recorded `latency_ms` percentiles, warm-up dropped
(`iteration >= 2` per client). `n = 50` per cell at `c=1`,
`n = 100` per cell at `c=2`. Linear-interpolation percentiles.

**threads=2 (Llama3-S2, `run_id=20260525-162757-c-smoke`):**

| server       | c |    p50 |    p95 |    p99 |
|:-------------|--:|-------:|-------:|-------:|
| hpx_server   | 1 |  586.0 |  593.0 |  595.0 |
| hpx_server   | 2 | 1123.0 | 1135.0 | 1137.0 |
| llama_server | 1 |  589.0 |  592.0 |  596.5 |
| llama_server | 2 | 1131.0 | 1154.0 | 1173.0 |

**threads=4 (Llama3-S3, `run_id=20260525-163545-c-smoke`):**

| server       | c |    p50 |    p95 |    p99 |
|:-------------|--:|-------:|-------:|-------:|
| hpx_server   | 1 |  588.0 |  604.5 |  606.5 |
| hpx_server   | 2 | 1131.0 | 1153.0 | 1157.0 |
| llama_server | 1 |  590.0 |  593.0 |  602.2 |
| llama_server | 2 | 1163.5 | 1267.0 | 1292.0 |

### 4. Observed patterns (neutral)

- hpx-server and llama-server recorded close `c=1` p50 values on
  both thread settings (within ≤ 3 ms across all four `c=1`
  cells).
- The HPX candidate anchor `0xcaaa7ff70cbabd53` stayed stable
  when moving from threads=2 to threads=4; doubling the
  compute-thread count did not perturb the hpx-server `p0_b8`
  hash on this shape.
- Increasing `--n-threads` from 2 to 4 did not lower the recorded
  hpx-server `latency_ms` p50, p95, or p99 on this `p0_b8` /
  `N=2` shape (every recorded threads=4 percentile sits at or
  slightly above the corresponding threads=2 value, by ≤ 20 ms).
- The llama-server `c=2` cell recorded the largest single
  movement in the table when moving from threads=2 to threads=4:
  recorded p95 sits +113 ms above the threads=2 value and p99
  sits +119 ms above. The hpx-server `c=2` movement at the same
  percentiles is +18 / +20 ms.
- Per-server `c=2 / c=1` p50 ratios are essentially unchanged
  across the two thread settings (hpx ≈ 1.92 at both threads=2
  and threads=4; llama ≈ 1.92 at threads=2 and ≈ 1.97 at
  threads=4). The recorded admitted-parallelism shape is
  unchanged in direction.
- Llama3-S2 / S3 are still narrow recorded results: one Llama 3
  model, one prompt, one decode budget, `p0_b8`, `N=2`,
  non-streaming, `c ∈ {1, 2}`, two specific thread settings.
- Neither thread setting is asserted as preferable on this shape,
  on this machine, or in general. The threads=2 / threads=4
  comparison is a single matched pair of runs.

### 5. Valid claims (Llama 3 interim)

- Llama 3.1 8B Q4_K_M `p0_b8` non-streaming works through `N=2`
  admitted parallelism at `c ∈ {1, 2}` on this machine, under
  both `--n-threads 2` and `--n-threads 4`, with every iteration
  completing with HTTP 200, `n_decoded == 8`, and stable per-cell
  `text_normalized_sha256`.
- The hpx-server produced a single stable candidate `p0_b8` hash
  (`0xcaaa7ff70cbabd53`) across all 330 hpx rows recorded so far,
  spanning the semantic smoke (S0), the N=2 smoke (S1), the
  threads=2 N=2 timing baseline (S2), and the threads=4 N=2
  timing baseline (S3).
- Client-side `latency_ms` p50 / p95 / p99 and
  `batch_wall_clock_ms` values are recorded for hpx-server and
  llama-server on this shape at `c ∈ {1, 2}` for both
  `--n-threads 2` and `--n-threads 4`, over `n = 50` / `n = 100`
  non-warm-up samples per cell.

### 6. Non-claims (Llama 3 interim)

- No production-throughput claim. Recorded `latency_ms`,
  `batch_wall_clock_ms`, and `client_observed_rps` are
  client-observed, include loopback, and are not server-throughput
  figures.
- No semantic-equality claim. The coincident
  `text_normalized_sha256` between the two servers on the
  non-streaming `p0_b8` Llama 3 shape remains a record-only
  observation; it is not extended to streaming, to other prompts,
  or to other budgets.
- No streaming claim for Llama 3; only the non-streaming path
  has been exercised.
- No `N=4` Llama 3 claim; only `N ∈ {1, 2}` has been exercised.
- No thread-scaling conclusion beyond this narrow run. The
  recorded threads=2 vs threads=4 comparison is a single matched
  pair on one shape; it is not a sweep and not a recommendation.
- No recommendation to use `--n-threads 4` permanently on the
  basis of this data.
- No generalization to larger prompts, larger decode budgets,
  other Llama 3 variants, or other model families.
- No canonical-anchor claim. `0xcaaa7ff70cbabd53` remains a
  candidate and is still **not** added to the harness's
  `CANONICAL_HPX_HASH` table.

### 7. Recommended next experiment

**Option A — Llama3-S4, N=4 smoke at threads=2,
`p0_b8` non-streaming, `c ∈ {1, 2, 4}`, small iterations.**
(Recommended.)

Rationale: threads=2 is the cleaner Llama 3 baseline so far —
both the S0 semantic smoke and the S2 timing baseline ran on
threads=2, and the candidate hash held there. An N=4 smoke at
threads=2 isolates admitted-parallelism scaling on Llama 3
without introducing a second variable. Adding threads=8 should
follow once N=4 correctness is known on this model; mixing the
two changes in one slice would make it harder to attribute any
recorded difference.

**Option B — Llama3-S4, threads=8 smoke at N=2, `p0_b8`
non-streaming, `c ∈ {1, 2}`, small iterations.**
(Defer.)

Rationale: a threads=8 smoke would extend the thread-scaling
view past S3's threads=4 point, but would not exercise any new
admitted-parallelism behavior on Llama 3. Pursue this after the
N=4 correctness smoke (Option A) holds, so the next thread sweep
runs on a known-anchored Llama 3 `N=4` shape.

Neither option authorizes additional changes; each remains a
separate design slice that requires its own approval.

# Llama3-S4 — N=4 admitted-parallelism smoke, threads=2

Recommended-Option-A run from the Llama 3 interim summary §7.
Goal: exercise Llama 3.1 8B Q4_K_M `p0_b8` non-streaming at
admitted parallelism `N=4` on this machine, at
`c ∈ {1, 2, 4}`, holding all other variables (decode budget,
prompt, model, threads, ctx size, sampling) at the Llama3-S2
baseline. This is a smoke, not a timing result.

## S4 run

- Driver: `concurrent_bench.py` unchanged. The harness reads the
  top-level `threads` field and the `server_args.hpx_server.n_seq_max`
  / `server_args.hpx_server.max_concurrent` / `server_args.llama_server.parallel`
  fields and wires them into the per-server argv.
- Config (new):
  `hpx-bench/experiments/15_hpx_vs_llama_server_semantics/config.llama3.n4.smoke.p0_b8.json`
  (mirrors `config.llama3.n2.smoke.p0_b8.json` except
  `concurrency_levels = [1, 2, 4]`,
  `hpx_server.n_seq_max = 4`,
  `hpx_server.max_concurrent = 4`,
  `llama_server.parallel = 4`).
- `run_id`: `20260525-214921-c-smoke`
- Result dir: `results/20260525-214921-c-smoke/`
- Driver capture: `local/runs/exp15/llama3-n4-smoke-p0_b8/driver.{stdout,stderr}`

Confirmed in `run_notes.txt` launch argv:
`hpx-server --n-seq-max 4 --max-concurrent 4 --n-threads 2`,
`llama-server --parallel 4 --threads 2`.

## Shape

```text
model                   /Users/unick/Desktop/hpx/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
                        (Llama 3.1 8B Instruct, Q4_K_M)
workload                p0_b8_llama3 (prompt "Hello, my name is", decode_budget=8)
mode                    non-streaming
concurrency_levels      [1, 2, 4]
iterations_per_client   6
warm-up rule            iteration == 1 per client is warm-up for timing aggregates;
                        correctness/stability gates use all rows.
row counts:
  c=1, per server:      6  rows (1 warm-up + 5 timed)
  c=2, per server:      12 rows (2 warm-up + 10 timed)
  c=4, per server:      24 rows (4 warm-up + 20 timed)
  total per server:     42 rows
server order            hpx-server first, then llama-server
server lifecycle        each server booted once per run; all three cells share that boot
server settings         hpx-server   --n-seq-max 4 --max-concurrent 4
                                     --max-prompt-tokens 512 --n-threads 2
                                     --ctx-size 2048 (no --engine-pool → default OFF)
                        llama-server --parallel 4 --threads 2 --ctx-size 2048
                                     --no-context-shift --seed 0
```

## Per-cell completion summary

| server       | c | rows | completed | failed | timeouts | batch_wall_clock_ms | client_observed_rps (approx) |
|:-------------|--:|-----:|----------:|-------:|---------:|--------------------:|-----------------------------:|
| hpx_server   | 1 |   6  |     6     |   0    |    0     |        3595         | 1.669                        |
| hpx_server   | 2 |  12  |    12     |   0    |    0     |        7080         | 1.695                        |
| hpx_server   | 4 |  24  |    24     |   0    |    0     |       11950         | 2.008                        |
| llama_server | 1 |   6  |     6     |   0    |    0     |        3697         | 1.623                        |
| llama_server | 2 |  12  |    12     |   0    |    0     |        4752         | 2.525                        |
| llama_server | 4 |  24  |    24     |   0    |    0     |       10402         | 2.307                        |

`client_observed_rps` is approximate / client-observed / includes
loopback / **not** a server-throughput claim. llama-server
`server_status` is `"stop"` (its label for hitting the requested
decode budget); `n_decoded == 8` on every row confirms the decode
completed as configured.

## Gate status

```text
gate: all rows http=200, no parse errors, no failed reasons, n_decoded == 8
note: no canonical hash pinned for workload=p0_b8_llama3 budget=8; stability-only gate
gate: hpx_server   c=1 text_normalized_sha256 stable: 3494b8623e8bc760…
gate: hpx_server   c=2 text_normalized_sha256 stable: 3494b8623e8bc760…
gate: hpx_server   c=4 text_normalized_sha256 stable: 3494b8623e8bc760…
gate: llama_server c=1 text_normalized_sha256 stable: 3494b8623e8bc760…
gate: llama_server c=2 text_normalized_sha256 stable: 3494b8623e8bc760…
gate: llama_server c=4 text_normalized_sha256 stable: 3494b8623e8bc760…
S0_GATES: PASS
```

Re-verified independently over all 84 rows of
`client_results.jsonl`:

- All rows returned HTTP 200 with `parse_error == ""` and
  `failed_reason == ""`.
- `n_decoded == 8` on every row, both servers, all three cells.
- `server_status`: hpx-server `"completed"` on 42/42 rows;
  llama-server `"stop"` on 42/42 rows.
- hpx-server `raw_json.hash == "0xcaaa7ff70cbabd53"` on every hpx
  row at `c=1` (6/6), at `c=2` (12/12), and at `c=4` (24/24).
  Single value, no off-candidate rows, no within-cell variation.
- llama-server `hash` field is empty on every row (llama-server
  does not emit this field on the non-streaming completion
  response in this configuration); cross-server hash equality is
  therefore not checked.
- `text_normalized_sha256 ==
  "3494b8623e8bc760884545ec06e87931726ee87df4fd470768b7ffdff339e0af"`
  on every row of both servers (matching the Llama3-S0 / S1 / S2
  / S3 value).
- No timeouts. No stuck server process (`pgrep` clean before and
  after).
- `run_notes.txt` forbidden-word audit clean
  (`forbidden_word_hits=[]`).

## HPX candidate hash at N=4 (per cell)

```text
hpx_server c=1:  6/6   rows  hash 0xcaaa7ff70cbabd53
hpx_server c=2: 12/12  rows  hash 0xcaaa7ff70cbabd53
hpx_server c=4: 24/24  rows  hash 0xcaaa7ff70cbabd53
```

**Candidate anchor held at N=4.** The Llama3-S0 candidate, confirmed
across S1 / S2 / S3, reappeared across all 42 hpx rows of this
`--n-seq-max 4 --max-concurrent 4` run, including warm-up rows,
with no off-candidate rows and no within-cell variation. Lifting
admitted parallelism from N=2 to N=4 (with `c=4` exercising the
new admission ceiling) did not perturb the hpx-server `p0_b8`
hash. The candidate is still **not** added to the harness's
`CANONICAL_HPX_HASH` table.

## llama-server stability at N=4

```text
llama_server c=1:  6/6   rows  text_normalized_sha256 = 3494b8623e8bc760…  text_len_chars = 21
llama_server c=2: 12/12  rows  text_normalized_sha256 = 3494b8623e8bc760…  text_len_chars = 21
llama_server c=4: 24/24  rows  text_normalized_sha256 = 3494b8623e8bc760…  text_len_chars = 21
```

llama-server's non-streaming response continues to omit the
hpx-style 64-bit `hash` field on this configuration, but the
`text_normalized_sha256` is a single stable value across every
cell, matching the Llama3-S0 / S1 / S2 / S3 value (`3494b86…`).
`text_len_chars` is 21 on every row.

## Cross-server text observation (record-only, not gated)

On every cell of this Llama 3 `N=4` non-streaming `p0_b8` smoke,
hpx-server and llama-server produced the same
`text_normalized_sha256`
(`3494b8623e8bc760884545ec06e87931726ee87df4fd470768b7ffdff339e0af`)
with identical `text_len_chars` (21). Recorded for completeness.
Cross-server text coincidence is **not** treated as a
semantic-equality gate (per CLAUDE.md correctness invariants); it
is **not** extended to streaming, other prompts, other budgets,
or other models.

## Timing (smoke, not a timing result)

Computed over `iteration >= 2` rows per client (warm-up dropped);
`n = 5` per cell at `c=1`, `n = 10` per cell at `c=2`,
`n = 20` per cell at `c=4`. Percentiles use linear interpolation
on the sorted sample. Sample sizes are too small to support
percentile claims; this is sanity-only.

| server       | c |   n |   latency_ms p50 |   latency_ms p95 |   latency_ms p99 | batch_wall_clock_ms |
|:-------------|--:|----:|-----------------:|-----------------:|-----------------:|--------------------:|
| hpx_server   | 1 |   5 |            596.0 |            610.8 |            611.8 |                3595 |
| hpx_server   | 2 |  10 |           1145.0 |           1227.4 |           1230.3 |                7080 |
| hpx_server   | 4 |  20 |           1986.0 |           2005.0 |           2005.0 |               11950 |
| llama_server | 1 |   5 |            620.0 |            634.0 |            636.4 |                3697 |
| llama_server | 2 |  10 |            786.5 |            798.5 |            798.9 |                4752 |
| llama_server | 4 |  20 |           1650.0 |           1904.0 |           1904.0 |               10402 |

Recorded sanity-only:

- Both servers completed all `c=4` iterations; per-cell row
  counts match the configured shape on both sides.
- hpx-server recorded `c=4` p50 sits roughly at 2 × the recorded
  `c=2` p50 (1986 / 1145 ≈ 1.73), and the `c=2` p50 sits roughly
  at 2 × the recorded `c=1` p50 (1145 / 596 ≈ 1.92). Direction is
  consistent with admitted-parallelism scaling on this shape.
- Timing is **not** interpreted beyond sanity; this is a 6-iter
  smoke (`n = 5 / 10 / 20` non-warm-up rows per cell).

## Server / process state

- hpx-server: `exit=-15` (SIGTERM-driven shutdown), `pid_still_running=False`.
- llama-server: `exit=0`, `pid_still_running=False`.
- No stuck `llama-server` or `llama-hpx-server` after the run
  (`pgrep` clean before and after).

## Forbidden-word audit

`run_notes.txt` clean (`forbidden_word_hits=[]`). Document hits in
`readme.md` / `facts.md` / `results.md` are confined to the
explicit audit-rule prose.

## Valid claims (Llama3-S4)

- A Llama 3.1 8B Q4_K_M `p0_b8` non-streaming `N=4`
  admitted-parallelism smoke completed on this machine at
  `c ∈ {1, 2, 4}`, `iterations_per_client=6`, `threads=2`.
- The Llama3-S0 / S1 / S2 / S3 candidate hpx-server `p0_b8` hash
  (`0xcaaa7ff70cbabd53`) held across all 42 hpx rows of this
  `N=4` run, at `c=1`, `c=2`, and `c=4`, with no off-candidate
  rows.
- Per-`(server, concurrency)` `text_normalized_sha256` is stable
  across every cell of both servers at the same value recorded in
  the prior Llama 3 runs.
- Larger-model admitted parallelism was exercised at
  `c == N == 4`: 24 concurrent client requests landed on
  hpx-server with `--n-seq-max 4 --max-concurrent 4` and on
  llama-server with `--parallel 4`, every iteration completing
  with HTTP 200 and `n_decoded == 8`.
- Single-machine `latency_ms` p50 / p95 / p99 and
  `batch_wall_clock_ms` values are recorded for hpx-server and
  llama-server at `c ∈ {1, 2, 4}` for sanity only.

## Non-claims (Llama3-S4)

- No production-throughput claim. `batch_wall_clock_ms`,
  `client_observed_rps`, and the timing table are client-observed,
  include loopback, and are not server-throughput figures.
- No performance / comparative claim. The smoke's
  `n = 5 / 10 / 20` non-warm-up samples per cell are not
  sufficient for percentile claims and are recorded as sanity
  only.
- No streaming claim for Llama 3; only the non-streaming path
  has been exercised.
- No thread-scaling claim. Llama3-S4 fixes `threads=2`; the
  threads=8 sweep remains a separate, unauthorized slice.
- No `p0_b32` Llama 3 claim, no `N>4` Llama 3 claim, no
  alternative-sampler claim.
- No semantic-equality claim. The coincident
  `text_normalized_sha256` between the two servers on this
  non-streaming Llama 3 shape continues to be a record-only
  observation and is not asserted as a server-internal property.
- No canonical-anchor claim. `0xcaaa7ff70cbabd53` remains a
  candidate confirmed at S0, S1, S2, S3, and S4; it is **not**
  added to the harness's `CANONICAL_HPX_HASH` table.
- No generalization beyond this machine (Apple M4 Pro, darwin),
  this model (Llama 3.1 8B Q4_K_M), this prompt
  (`"Hello, my name is"`), this budget (8), greedy sampling,
  `c ∈ {1, 2, 4}` at `N=4`, or `--n-threads 2` / `--threads 2`.
