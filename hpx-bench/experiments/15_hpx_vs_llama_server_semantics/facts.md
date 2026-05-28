# Experiment 15 — Phase 0 facts

Design-time facts for the post-N5a/N5b semantic-alignment
revalidation. All raw values only. No performance claim. No
cross-server equality is asserted beyond the gates listed below.

## Driver

`hpx-bench/experiments/12_hpx_vs_llama_server_pair/bench.py`,
unchanged. Python 3 stdlib only. Boots exactly one server at a time
(`hpx-server` first, then `llama-server`), never both concurrently.
The adapters
(`adapters/hpx_server.py`, `adapters/llama_server.py`) are also
inherited unchanged.

The harness is descriptive-only. `run_notes.txt` is scanned for
forbidden comparative words (`faster`, `slower`, `speedup`,
`regression`, `wins`, `beats`, `outperforms`, `better`, `worse`) by
`bench.py`; presence of any of those is a failure mode.

## Servers

```text
hpx-server
  /Users/unick/Desktop/hpx/builds/llama-hpx-hpx-on/bin/llama-hpx-server
  argv (built by adapters/hpx_server.py:build_args):
    --model <path>
    --host 127.0.0.1
    --port <PORT>
    --n-seq-max 1
    --max-prompt-tokens 512
    --n-threads 2
    --max-concurrent 1
    --ctx-size 2048
  Defaults not overridden:
    --hpx-os-threads 2       (the "default_os2" mode from Exp14)
    --engine-pool OFF        (no --engine-pool flag on the cmdline)
  Includes N5a and N5b fixes (engine.cpp finalize_and_fulfill +
  request_shutdown drain).

llama-server
  /Users/unick/Desktop/hpx/builds/llama-base/bin/llama-server
  argv (built by adapters/llama_server.py:build_args):
    --model <path>
    --host 127.0.0.1
    --port <PORT>
    --ctx-size 2048
    --parallel 1
    --threads 2
    --no-context-shift
    --seed 0
  Upstream binary unchanged for Phase 0.
```

Both servers are bound to `127.0.0.1`. Phase 0 uses fixed config
ports `9087` (hpx) and `9088` (llama). Cells run strictly
sequentially, so port reuse across phases is not a concern.

## Model and prompts

- Model: TinyLlama 1.1B Chat v1.0 Q4_K_M
  (`/Users/unick/Desktop/hpx/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`).
- Prompt `p0` (canonical): `Hello, my name is`.
- Prompt `p1` (EOG-sensitive): `Write one short sentence about the moon.`
  Trailing sentence-final punctuation; greedy argmax on TinyLlama
  often lands on EOG immediately after prefill.

## Workload matrices

### Phase 0 (semantic alignment)

Non-streaming (`config.phase0.nonstreaming.json`):

```text
iterations_per_workload: 3
workloads:
  p0_b8   "Hello, my name is"                          budget=8   is_canonical_anchor=true
  p0_b32  "Hello, my name is"                          budget=32  is_canonical_anchor=false
  p1_b8   "Write one short sentence about the moon."   budget=8   is_canonical_anchor=false
```

Streaming (`config.phase0.streaming.json`, `streaming: true`):

```text
iterations_per_workload: 10
workloads:
  p0_b8   "Hello, my name is"                          budget=8   is_canonical_anchor=true
```

`p1_b32` is intentionally omitted from Phase 0 — `p1` lands on EOG
at or very close to position 0 on TinyLlama (recorded in
`docs/hpx/provenance.md` §9.5), so `p1_b32` adds no information
over `p1_b8`.

### Phase 1 (single-client timing comparison)

Non-streaming (`config.phase1.nonstreaming.json`):

```text
iterations_per_workload: 51
workloads:
  p0_b8   "Hello, my name is"                          budget=8   is_canonical_anchor=true
  p0_b32  "Hello, my name is"                          budget=32  is_canonical_anchor=false
```

Streaming (`config.phase1.streaming.json`, `streaming: true`):

```text
iterations_per_workload: 51
workloads:
  p0_b8   "Hello, my name is"                          budget=8   is_canonical_anchor=true
  p0_b32  "Hello, my name is"                          budget=32  is_canonical_anchor=false
```

Phase 1 is `p0`-only. `p1`/EOG is excluded from Phase 1 by
construction — both servers stop at or very near `n_decoded=0`, so
timing "essentially no decode" yields no useful signal. Phase 0
already pinned `p1` semantics.

## Request payload shapes

```text
hpx-server /completion (per adapters/hpx_server.py:request_body):
  {"prompt": "<prompt>", "decode_budget": <budget>}
  Streaming variant adds "stream": true.

llama-server /completion (per adapters/llama_server.py:_pinned_sampler_body):
  {
    "prompt": "<prompt>",
    "n_predict": <budget>,
    "temperature": 0, "top_k": 1, "top_p": 1.0, "min_p": 1.0,
    "typical_p": 1.0, "repeat_penalty": 1.0,
    "presence_penalty": 0.0, "frequency_penalty": 0.0,
    "dry_multiplier": 0.0, "xtc_probability": 0.0,
    "mirostat": 0, "n_probs": 0,
    "samplers": ["top_k", "temperature"], "stop": [],
    "seed": 0, "stream": false, "cache_prompt": false
  }
  Streaming variant sets "stream": true.
  ignore_eos is intentionally ABSENT (M8d fix). Pinning ignore_eos on
  llama-server while hpx-server has no equivalent created a behavioral
  asymmetry on EOG-sensitive prompts. Both servers now stop on EOG.
```

## Canonical hpx-server anchors

```text
hpx p0_b8   hash = 0x0619d4d1900c2365   n_decoded = 8
hpx p0_b32  hash = 0x6794e47fe0f84af1   n_decoded = 32
hpx p1_b8   hash = 0x0000000000000000   n_decoded = 0   empty text
```

The `p1_b8` row is the canonical EOG-stop signature: TinyLlama
greedy emits EOG immediately at the post-prefill site, the engine's
EOG branch finalizes the request as `completed` without appending
the EOG token, and `engine.cpp` returns the all-zero zero-length
hash. This is documented engine behavior, not a failure.

## Gated fields (Exp12 inherits, plus Phase 0 anchors)

Inherited row-level gates from Exp12 `evaluate_gates`:

- `http_status == 200`
- `parse_error == ""`
- `0 <= n_decoded <= decode_budget`
- Streaming additions: `done_seen`,
  `stream_parse_error == ""`, `first_event_latency_ms > 0`,
  `total_latency_ms > 0`, `first_token_latency_ms > 0` when
  `token_event_count > 0`.

Inherited per-(server, workload) stability gates:

- `text_normalized_sha256` stable across iterations.
- `n_decoded` stable across iterations.
- hpx-server `hash` stable across iterations.

Phase 0 anchor expectations (verified post-run, recorded in
`results.md`):

- hpx p0_b8 `hash == 0x0619d4d1900c2365` AND `n_decoded == 8` for
  every iteration.
- hpx p0_b32 `hash == 0x6794e47fe0f84af1` AND `n_decoded == 32` for
  every iteration.
- hpx p1_b8 `n_decoded == 0` AND `hash == 0x0000000000000000` AND
  empty `text` for every iteration.

Cross-server hash equality is **not** asserted at any shape.

## Phase 1 metric inventory (single-client timing)

Phase 1 produces aggregate percentiles from the same JSONL rows that
Phase 0 records. Exp12's `bench.py` writes iterations as 1-indexed
(`iteration ∈ [1, 51]`). **The first measured iteration
(`iteration == 1`) is treated as warm-up** and excluded from timing
percentiles. Reported percentiles are computed over
`iteration ∈ [2, 51]` (n=50). Correctness, anchor, and per-server
stability gates are evaluated over all 51 rows.

Comparable metrics (client-side monotonic, include loopback; field
names are exactly as Exp12's `client_results.jsonl` writes them):

```text
Non-streaming:
  latency_ms                  p50 / p95 / p99 per (server, workload)

Streaming:
  first_event_latency_ms      p50 / p95           (≈ TTFT; first SSE record)
  first_token_latency_ms      p50 / p95           (first content-bearing record)
  total_latency_ms            p50 / p95 / p99
  inter_event_gap_ms          p50 / p95           (derived per row from
                                                   consecutive differences of
                                                   token_event_times_ms,
                                                   then flattened across rows)
  (derived, caveated)
  tokens/sec ≈ n_decoded * 1000 / total_latency_ms
```

The derived tokens/sec is meaningful only on `p0_b8` and `p0_b32`,
where both servers report `n_decoded == decode_budget`. It is
reported as a descriptive ratio, not a throughput claim.

All Phase 1 percentiles are reported descriptively — no winner
language, no comparative adjectives beyond "lower"/"higher"/
"differs by" applied to specific numeric values.

## Recorded-only fields (not gated)

- llama-server `tokens_evaluated` (prompt-token count). hpx-server
  returns `prompt_tokens = -1`; not comparable.
- Cross-server `text_sha256` and `text_normalized_sha256` equality.
  Acknowledged sources of asymmetry: implicit BOS on llama-server,
  leading-space convention on first generated token in llama-server
  `content`, and divergent sampler-chain implementations even at
  `top_k=1, temperature=0`. See Exp12's auto-generated
  `known_mismatches.txt` for the canonical list.
- Cross-server `n_decoded` equality. EOG-stop positions can differ
  between servers (`p1` is the obvious case).
- Streaming wire-format differences:
  - hpx-server SSE: `event: token` + `event: done` records.
  - llama-server SSE: bare `data:` records, terminal record carries
    `stop: true` and `stop_type`.
- Streaming detokenization: hpx-server emits per-token raw
  detokenization (no SentencePiece word-boundary glue across tokens);
  llama-server emits incremental detokenizer chunks. Cross-server
  streaming text equality is not meaningful and not gated.
- TTFT (`first_token_latency_ms`) and `first_event_latency_ms` —
  raw monotonic deltas, client-observed, include loopback. Not
  aggregated, averaged, or compared across servers.
- All raw response JSONs (`raw_json` field) — kept for debugging.

## EOG/EOS observation contract

```text
hpx-server EOG-stop signature (p1_b8):
  status   = completed
  n_decoded = 0
  hash      = 0x0000000000000000
  text      = ""

llama-server EOG-stop signature (p1_b8):
  stopped_eos = true   (mapped to server_status = "stopped_eos")
  tokens_predicted = 1
  content     = ""     (empty content even though tokens_predicted=1)
  stop_type   = "eos"  (in the terminal streaming record)
```

These differ. They are **recorded**; they are not failures. The
`ok` field has divergent meaning between modes — non-streaming sets
`ok` on non-empty `text`, streaming sets `ok` on a clean terminal
record. An EOG-stop row can be non-streaming `ok=False` and
streaming `ok=True` simultaneously; this is not a contradiction.

## Output layout

Each `bench.py` invocation creates:

```text
results/<YYYYMMDD-HHMMSS>-pair/
  config.json
  client_results.jsonl
  summary.csv
  known_mismatches.txt
  match_conditions.json
  run_notes.txt
  hpx_server.{stdout,stderr,args}
  llama_server.{stdout,stderr,args}
```

The Phase 0 `results.md` references the run IDs and the per-shape
anchor outcomes; it does not duplicate the raw JSONL or summary.csv.

## Caveats baked into every Phase 0 output

- Single machine: Apple M4 Pro, darwin, TinyLlama 1.1B Q4_K_M.
- Single client; servers run strictly sequentially.
- No performance claim. No concurrent-client claim. No
  llama-server-vs-hpx-server timing comparison.
- Phase 0 evidence does not generalize beyond the workload matrix
  above on this hardware and this model.

## Phase 2a-S0 — concurrent harness schema and gates

S0 is a harness-correctness slice for the new
`concurrent_bench.py` driver. Server build, model, ports, sampler
pinning, and the Exp12 adapters are inherited unchanged. The S0
config is `config.phase2a.smoke.p0_b8.json`.

### Driver reuse

```text
sys.path.insert(0, ".../12_hpx_vs_llama_server_pair")
from adapters import hpx_server, llama_server
import bench as exp12_bench   # for tcp_wait, http_post_json,
                              # launch_server, stop_server,
                              # write_args_file, server_pid_running
```

`concurrent_bench.py` adds its own:

- threaded client pool with a `threading.Barrier` start fence,
- per-cell wall-clock and `(completed, failed, timeout)` counters,
- gate evaluator that asserts the S0 invariants,
- output writers for `client_results.jsonl`, `summary.csv`, and
  `run_notes.txt`.

No semantic parsing logic is duplicated from Exp12.

### S0 server settings

Inherited from Exp12 adapters (Phase 0 / Phase 1 settings), unchanged:

```text
hpx-server (adapters/hpx_server.py:build_args):
  --n-seq-max 1 --max-prompt-tokens 512 --max-concurrent 1
  --n-threads 2 --ctx-size 2048

llama-server (adapters/llama_server.py:build_args):
  --parallel 1 --threads 2 --ctx-size 2048
  --no-context-shift --seed 0
```

S0 keeps both servers at the same admission limits as Phase 0 / Phase
1 (`--n-seq-max 1` / `--parallel 1`). At `c = 2`, the second client
queues behind the first by design; that is exactly the lifecycle path
S0 is validating in the harness.

Each server boots **once per run**; both concurrency cells (`c=1`
then `c=2`) for that server run against the same boot. No port
reuse occurs.

### S0 workload

```text
iterations_per_client: 6
concurrency_levels:    [1, 2]
workload:
  p0_b8   "Hello, my name is"   budget=8   is_canonical_anchor=true
```

The first iteration per client (`iteration == 1`) is treated as
warm-up for any aggregate that follows. S0 itself does not compute
aggregates — it only checks that all rows satisfy the row-level and
per-cell stability gates, including the warm-up row.

### S0 per-row schema (`client_results.jsonl`)

```text
server                    str   "hpx_server" | "llama_server"
workload_id               str   "p0_b8"
prompt_id                 str   "p0"
is_canonical_anchor       bool  true
concurrency               int   cell concurrency (1 or 2)
client_id                 int   0..concurrency-1
iteration                 int   1..iterations_per_client
is_warmup                 bool  (iteration == 1)
decode_budget_requested   int   8
submit_monotonic_ms       int   client submit time relative to cell start
complete_monotonic_ms     int   client complete time relative to cell start
latency_ms                int   complete - submit (wall-clock)
http_status               int   HTTP status; -1 on transport failure
n_decoded                 int   from response; -1 if absent
hash                      str   "0x..." (hpx only); "" for llama
text_normalized_sha256    str   sha256 of stripped response text
text_len_chars            int   length of response text in chars
server_status             str   adapter-normalized status
parse_error               str   "" on clean parse
failed_reason             str   "" if completed; otherwise reason
ts_utc                    str   ISO-8601 timestamp
raw_json                  obj   parsed response body
request_body              obj   client request body actually sent
```

`summary.csv` flattens a subset of these columns (no `raw_json` or
`request_body`).

### S0 per-cell aggregates (`run_notes.txt` and `summary` output)

```text
server                    str
workload_id               str
concurrency               int
rows                      int     concurrency * iterations_per_client
completed_count           int     http_status==200 AND parse_error=="" AND
                                  failed_reason==""
failed_count              int     failed_reason set AND no "timeout"
timeout_count             int     failed_reason set AND contains "timeout"
batch_wall_clock_ms       int     max(complete) - min(submit) across rows
client_observed_rps       float   completed_count * 1000 / batch_wall_clock_ms
client_observed_rps_note  str     "approximate; client-observed; ..."
```

`client_observed_rps` is labelled approximate/client-observed and is
**not** a server-throughput claim. It includes loopback transport
time and the harness's own per-thread submit jitter.

### S0 gates (as originally defined; superseded by Phase 2 gate policy below)

The S0 gates below are recorded as-defined for historical traceability.
**S0 failed the row-level `http_status == 200` gate under matched
single-slot admission settings**, and that failure is now accepted as
an admission-policy semantic finding rather than a harness defect.
See "Phase 2 gate policy" at the end of this file for the reshaped
gates that govern any subsequent Phase 2 work.

Row-level (every row in either server's set):

- `http_status == 200`
- `parse_error == ""`
- `failed_reason == ""`
- `n_decoded == decode_budget` (== 8 for `p0_b8`)

hpx-server canonical anchor (every hpx_server row, warm-up included):

- `hash == 0x0619d4d1900c2365`

Per-(server, concurrency) stability (every cell in either server):

- `text_normalized_sha256` stable across the cell's rows.

Process:

- No stuck `llama-server` or `llama-hpx-server` after either cell.

Notes audit:

- `run_notes.txt` contains no forbidden comparative words
  (`faster`, `slower`, `speedup`, `regression`, `wins`, `beats`,
  `outperforms`, `better`, `worse`).

### S0 record-only fields

Carried forward from Phase 0 / Phase 1; not gated:

- `latency_ms`, `submit_monotonic_ms`, `complete_monotonic_ms`,
  `batch_wall_clock_ms`, `client_observed_rps` — recorded so the
  schema is exercised; **not** aggregated as percentiles and not
  compared between servers in S0.
- Cross-server `text_normalized_sha256` equality.
- Cross-server `n_decoded` equality.
- llama-server `hash` field absent (by adapter design).
- llama-server `tokens_evaluated` vs hpx-server `prompt_tokens=-1`.

### S0 output layout

```text
results/<YYYYMMDD-HHMMSS>-c-smoke/
  config.json
  client_results.jsonl
  summary.csv
  run_notes.txt
  hpx_server.{stdout,stderr,args}
  llama_server.{stdout,stderr,args}
```

Driver `stdout` / `stderr` capture lives outside the run directory:
`local/runs/exp15/phase2a-smoke-p0_b8-nonstreaming/driver.{stdout,stderr}`.

## Phase 2 gate policy (post-S0)

S0 surfaced a real admission-policy divergence: under matched
`hpx-server --max-concurrent 1` / `llama-server --parallel 1`,
hpx-server rejects the excess concurrent client with HTTP 503 while
llama-server queues it and serves it later. The original S0 "every
row is HTTP 200" gate over-assumed both servers queue. Phase 2 work
is therefore split into two tracks, each with its own gate policy.

### Phase 2a — admission / rejection semantics under overload

Purpose: record what each server does when the externally-issued
client count exceeds the externally-admitted slot count.

Server settings (matched; carried forward from Phase 0 / Phase 1):

```text
hpx-server   --n-seq-max 1 --max-concurrent 1
llama-server --parallel 1
```

Gates (PASS required):

- For every row with `http_status == 200`:
  - `parse_error == ""`
  - `n_decoded == decode_budget`
  - For hpx-server: `hash` matches the canonical anchor for the
    workload (`p0_b8 → 0x0619d4d1900c2365`,
    `p0_b32 → 0x6794e47fe0f84af1`).
- For hpx-server rows with `http_status == 503`:
  - `n_decoded == -1`, `hash == ""`, and `failed_reason` is the 503
    indicator. These rows are recorded, not gated against the 200
    invariants.
- Per `(server, concurrency)`, `text_normalized_sha256` is stable
  across the cell's HTTP-200 rows.
- No timeouts; no stuck server process; `run_notes.txt` free of
  forbidden comparative wording.

Reported descriptors per `(server, concurrency)`:

```text
completed_count       int     http_status == 200
rejected_count        int     http_status == 503  (hpx-server only by design)
failed_count          int     other non-200 / parse errors / transport errors
timeout_count         int
rejected_rate         float   rejected_count / rows
batch_wall_clock_ms   int
```

Record-only (not gated, not aggregated as percentiles in Phase 2a):

- Per-row `latency_ms`, `submit_monotonic_ms`, `complete_monotonic_ms`,
  `client_observed_rps`. Comparing cross-server latency in 2a is
  not meaningful because the successful-request populations differ
  between servers under overload.

Retry-on-503 is **not** enabled in Phase 2a; enabling it would mask
the very divergence 2a is designed to record.

### Phase 2b — admitted parallelism with matched N

Purpose: time completed requests when both servers are configured
to admit the same number of in-flight requests, so the
successful-request populations match.

Server settings (matched; `N ∈ {2, 4}` first cut):

```text
hpx-server   --n-seq-max N --max-concurrent N
llama-server --parallel N
```

Gates (PASS required):

- Every row: `http_status == 200`, `parse_error == ""`,
  `n_decoded == decode_budget`.
- Every hpx-server row: `hash` matches the canonical anchor.
- Per `(server, concurrency)`, `text_normalized_sha256` is stable
  across the cell's rows.
- No 503; no timeouts; no stuck server process;
  `run_notes.txt` free of forbidden comparative wording.

Reported timing aggregates per `(server, concurrency)`:

- `latency_ms` p50 / p95 / p99 over `iteration >= 2` rows per client
  (warm-up rule unchanged from Phase 1).
- `batch_wall_clock_ms`, `client_observed_rps` (descriptive only).

Phase 2b is the correct track for any latency-under-load
side-by-side. It does not run until Phase 2a is documented and
the matched-N settings are validated by a smoke (2b-S0).

### Canonical hashes are HTTP-200 invariants only

Canonical hpx-server `hash` and `n_decoded` are correctness
invariants of completed greedy decode runs. HTTP-503 rejection rows
never reach the engine and therefore carry empty `hash` and
`n_decoded == -1`; this is not anchor drift.

### Forbidden-word audit applies to both tracks

Same word list as Phase 0 / Phase 1 / S0: `faster`, `slower`,
`speedup`, `regression`, `wins`, `beats`, `outperforms`, `better`,
`worse`. Applies to `run_notes.txt`, `results.md`, `readme.md`, and
`facts.md`. Matches inside the audit prose that quotes the word
list itself are allowed.

## Phase 2b-S0 — schema and gates

Phase 2b-S0 is the admitted-parallelism smoke for the Phase 2b
track. Server build, model, prompt, sampler pinning, and adapter
imports are inherited unchanged from Phase 2a-S0. The S0 config is
`config.phase2b.smoke.p0_b8.json`.

### Config schema extension (`server_args`)

`concurrent_bench.py` gained an optional top-level `server_args`
block. When present, it routes argv assembly to harness-side
builders. When absent, the harness falls back to
`adapter.build_args(cfg, port)`, preserving the Phase 2a-S0
config byte-for-byte.

```jsonc
"server_args": {
  "hpx_server": {
    "n_seq_max":         <int, required>,
    "max_concurrent":    <int, required>,
    "max_prompt_tokens": <int, optional, default 512>
  },
  "llama_server": {
    "parallel":          <int, required>,
    "no_context_shift":  <bool, optional, default false>,
    "seed":              <int, optional>
  }
}
```

The shared knobs (`model_path`, `binaries`, `ports`, `ctx_size`,
`threads`, `workload`, `concurrency_levels`,
`iterations_per_client`, `readiness`, `shutdown`,
`request_timeout_seconds`) carry over from the Phase 2a-S0 schema
unchanged. Harness builders only assemble argv; semantic-parsing
imports from Exp12 adapters (`request_body`, `warmup_body`,
`parse_response`) are reused as-is.

### Phase 2b-S0 server settings (matched)

```text
hpx-server   --n-seq-max 2 --max-concurrent 2 --max-prompt-tokens 512
             --n-threads 2 --ctx-size 2048
             (no --engine-pool flag → default OFF)

llama-server --parallel 2 --threads 2 --ctx-size 2048
             --no-context-shift --seed 0
```

### Phase 2b-S0 workload

```text
iterations_per_client:  6
concurrency_levels:     [1, 2]
workload:
  p0_b8   "Hello, my name is"   budget=8   is_canonical_anchor=true
```

Warm-up rule unchanged: `iteration == 1` per client is warm-up for
any aggregate. Correctness gates use all rows.

### Phase 2b-S0 gates (PASS required)

Strict; every row in either server's set, both concurrency cells:

- `http_status == 200` (no HTTP 503 — distinguishing from Phase 2a)
- `parse_error == ""`
- `failed_reason == ""`
- `n_decoded == 8`
- For every hpx-server row: `hash == 0x0619d4d1900c2365`
- Per `(server, concurrency)`: `text_normalized_sha256` stable
- No stuck `llama-server` or `llama-hpx-server` after either cell
- `run_notes.txt` free of forbidden comparative words

### Canonical-hash outcomes at `c = 2`

The hpx-server canonical anchor `0x0619d4d1900c2365` was pinned in
Phase 1 with `--n-seq-max 1`. At `--n-seq-max 2 --max-concurrent 2`
the engine's batch composition can include N=2 at decode steps
when two clients overlap. CLAUDE.md anticipates this:

> If batch composition, seq ordering, prompt/decode mixing, or
> scheduling policy changes, regenerate same-shape reference
> results before comparing hashes.

Phase 2b-S0 therefore handles three outcomes explicitly:

- **Case A** — every hpx `c=2` row has `hash ==
  0x0619d4d1900c2365`. PASS candidate. The canonical anchor holds
  under N=2 admitted parallelism for this prompt/budget/sampling/
  hardware combination.
- **Case B** — every hpx `c=2` row has a single, stable,
  non-canonical hash. Controlled finding, not an automatic failure.
  Record the new hash as a candidate N=2 same-shape anchor; do not
  declare PASS; do not canonicalize without explicit approval.
- **Case C** — hpx `c=2` hash varies across rows. Correctness
  concern; stop and report.

### Record-only metrics

Per-row `latency_ms`, `submit_monotonic_ms`,
`complete_monotonic_ms` and per-cell `batch_wall_clock_ms`,
`client_observed_rps` are recorded so the schema is exercised;
they are **not** aggregated as percentiles in S0 and **not**
compared between servers in S0. Per-row `prompt_tokens` for
llama-server is record-only (asymmetry already pinned).

### Phase 2b-S0 output layout

```text
results/<YYYYMMDD-HHMMSS>-c-smoke/
  config.json
  client_results.jsonl
  summary.csv
  run_notes.txt
  hpx_server.{stdout,stderr,args}
  llama_server.{stdout,stderr,args}
```

The run-id suffix `-c-smoke` is shared with Phase 2a-S0; the
timestamp disambiguates the two runs, and `config.json` echoes the
exact config used so the run can be associated with its phase
unambiguously. Driver `stdout` / `stderr` capture lives outside
the run directory:
`local/runs/exp15/phase2b-smoke-p0_b8-nonstreaming/driver.{stdout,stderr}`.
