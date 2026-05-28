# Experiment 14 — facts

Stable design-time facts about the Phase 1 driver and its outputs. Field
names and units are the contract that `bench.py`, the JSONL writer, and
`summary.csv` all share.

## Driver scope

- Pure Python 3 stdlib. No third-party deps. Modules used:
  `argparse`, `csv`, `datetime`, `hashlib`, `http.client`, `json`, `os`,
  `pathlib`, `signal`, `socket`, `subprocess`, `sys`, `time`.
- Does not import any code from `hpx-bench/experiments/12_*`. Driver and
  adapter are self-contained under this directory.
- Does not modify or rebuild any binary. The hpx-server binary path is
  taken from `--binary` and used as-is.
- Does not touch the engine, runtime, hpx-server source, or any smoke.

## Cell isolation

A "cell" is one (mode, workload, decode_budget) combination. In the main
run there are 3 modes × (2 W1 budgets + 2 W2 budgets + 1 W3 budget) =
15 cells. Each cell:

1. Picks a free ephemeral TCP port via `socket.bind(('127.0.0.1', 0))`.
2. Launches one hpx-server subprocess with mode-specific extra args:
   - `default_os1`: `--hpx-os-threads 1`
   - `default_os2`: `--hpx-os-threads 2`
   - `engine_pool_os2`: `--hpx-os-threads 2 --engine-pool`
3. Waits for TCP connect, then a warm-up POST (`decode_budget=1`).
4. Runs `--warmup-trials` warmup trials (flagged `is_warmup=true`) of
   the cell's workload at the cell's budget.
5. Runs `--trials` measured trials.
6. For W3 cells, runs one extra non-streaming sanity request (canonical
   prompt, budget=8) and records its hash check result in the manifest.
7. SIGTERMs the server with a grace period, falls back to SIGKILL on
   timeout, closes stdout/stderr files.

Cells run strictly sequentially. Never two model-loading processes at
once. Server reuse within a cell across trials is intentional and matches
real serving behavior.

## Mode definitions

```text
default_os1      hpx_os_threads=1   engine_pool=false
default_os2      hpx_os_threads=2   engine_pool=false
engine_pool_os2  hpx_os_threads=2   engine_pool=true
```

`engine_pool_os2` requires `hpx_os_threads >= 2`; the hpx-server
preflight enforces it. `default_os1` is the only mode that runs the
engine HPX task on a single shared OS worker.

## Workload definitions

```text
W1  POST /completion  non-streaming
    body = {"prompt": <prompt>, "decode_budget": <budget>}
    metrics:
      response_complete_ms = body fully read - request_start
      http_status, response_bytes, n_decoded, hash

W2  POST /completion  streaming
    body = {"prompt": <prompt>, "decode_budget": <budget>, "stream": true}
    metrics:
      first_event_ms       = first complete SSE record terminator
      first_token_event_ms = first `event: token` record
      response_complete_ms = terminal `event: done` record
      inter_event_gap_ms   = consecutive event-arrival deltas (derived)
      token_event_count, done_seen, done_status, n_decoded, hash

W3  POST /completion  streaming + client abort
    body = {"prompt": <prompt>, "decode_budget": <w3_budget>,
            "stream": true}
    abort condition: cumulative bytes read >= 64
    metrics:
      first_event_ms       = first complete SSE record terminator
      disconnect_ms        = local socket close return - request_start
      bytes_received
    sanity (once per W3 cell, after the trial loop):
      one non-streaming POST budget=8 must HTTP 200 with canonical
      greedy hash. Sanity outcome stored in the manifest, not in the
      raw JSONL.
```

All timings are monotonic deltas captured by `time.monotonic()`, in
**milliseconds** (float). Wall-clock UTC ISO timestamps are recorded
separately for trace.

## Canonical correctness anchors

- Prompt: `"Hello, my name is"`
- Budget-8 greedy hash: `0x0619d4d1900c2365`
- Budget-64 greedy hash: not asserted in Exp 14 (no within-run
  comparison would be cross-shape, but Exp 14 keeps W1/W2 hashes
  recorded as observations — see W1/W2 rows in JSONL).

The budget-8 anchor is the only **hash gate** in Phase 1: W1 budget=8
must match, W2 budget=8 terminal-done hash must match, W3 post-cell
sanity request must match. All three guard against an accidental engine
change between phases.

## Readiness

Two-stage:

1. TCP connect poll on `127.0.0.1:<port>`.
2. Warm-up POST `/completion` with `decode_budget=1` polled until HTTP
   200.

Both bounded by `--readiness-timeout-seconds` (default 30s). Failure to
reach readiness aborts the cell; the manifest records `readiness_ok=false`
and `readiness_error=<message>`; the cell's trials are skipped.

## Output layout (per run)

```text
results/<run_id>/
  manifest.json
  commands.txt
  raw/<mode>_<workload>_b<budget>.jsonl
  logs/<mode>_<workload>_b<budget>.stdout
  logs/<mode>_<workload>_b<budget>.stderr
  logs/<mode>_<workload>_b<budget>.args
  summary.csv
```

`run_id` format: `YYYYMMDD-HHMMSS-<label>` (e.g. `20260523-160100-phase1`).

## manifest.json schema

```text
{
  "harness_version":  "exp14.phase1.v1",
  "run_id":           "<run_id>",
  "label":            "<--label arg>",
  "started_utc":      "<iso>",
  "ended_utc":        "<iso>",
  "git_sha":          "<short sha>",
  "git_branch":       "<branch>",
  "git_dirty":        true|false,
  "machine": {
    "platform":       "<sys.platform>",
    "uname":          "<os.uname()>",
    "python":         "<sys.version>"
  },
  "binary":           "<absolute path>",
  "binary_sha256":    "<hex>",
  "model":            "<absolute path>",
  "model_sha256":     "<hex>"          // computed if model size < 5 GiB
  "modes":            ["..."],
  "workloads":        ["..."],
  "decode_budgets":   [8, 64],
  "w3_decode_budget": 128,
  "w3_post_disconnect_sleep_s": 2.0,
  "w3_between_trial_mode":      "sleep" | "capfree",
  "trials":           30,
  "warmup_trials":    1,
  "placement_trace_engine_pool": true|false,
  "cells": [
    {
      "mode":               "<mode>",
      "workload":           "<w1|w2|w3>",
      "decode_budget":      <int>,
      "port":               <int>,
      "argv":               ["..."],
      "env_extra":          {"LLAMA_HPX_PLACEMENT_TRACE": "1"} | {},
      "started_utc":        "<iso>",
      "ended_utc":          "<iso>",
      "readiness_ok":       true|false,
      "readiness_error":    "",
      "trials_attempted":   <int>,
      "trials_ok":          <int>,
      "trials_hash_ok":     <int>      // W1+W2 budget=8 only
      "trials_done_seen":   <int>      // W2 only
      "w3_sanity_ok":       true|false // W3 only
      "w3_sanity_hash_ok":  true|false // W3 only
      "w3_probe_attempts":  <int>|null // W3 + capfree only; null otherwise
      "w3_probe_ok":        <int>|null // W3 + capfree only; null otherwise
      "w3_probe_max_ms":    <float>|null // max observed wait_for_cap_free
                                         // elapsed_ms across the cell;
                                         // null until the first probe
      "exit_code":          <int>,
      "term_method":        "sigterm|sigkill|already_exited",
      "placement_trace_seen": true|false  // engine_pool_os2 only
    }
  ]
}
```

## raw/*.jsonl row schema

One row per request (warmups + measured + the W3 sanity request).

```text
{
  "mode":                 "<mode>",
  "workload":             "<w1|w2|w3|w3_sanity|w3_probe>",
  "decode_budget":        <int>,
  "iteration":            <int>             // 0 = warmup if --warmup-trials > 0
                                            // w3_probe pre_sanity: -1
                                            // w3_probe between_trial: index of trial just completed
  "is_warmup":            true|false,       // w3_probe between_trial: matches the trial just completed
                                            // w3_probe pre_sanity: false
  "ts_utc":               "<iso>",
  "request_start_monotonic": <float seconds>,
  "request_body":         {...},
  "http_status":          <int>,
  "response_bytes":       <int>,
  "response_complete_ms": <float>,
  // W2/W3 only:
  "first_byte_ms":        <float|null>,
  "first_event_ms":       <float|null>,
  "first_token_event_ms": <float|null>,
  "token_event_count":    <int>,
  "done_seen":            true|false,
  "done_status":          "<str>",
  "inter_event_gaps_ms":  [<float>, ...]   // empty for W1
  // W3 only:
  "disconnect_ms":        <float>,
  "bytes_received":       <int>,
  // w3_probe only (capfree mode):
  "probe_phase":          "<between_trial|pre_sanity>",
  "probe_ok":             true|false,       // wait_for_cap_free return
  "probe_elapsed_ms":     <float>,          // monotonic call duration
  "probe_timeout_s":      <float>,          // 30.0 in the current harness
  // correctness fingerprints:
  "n_decoded":            <int>,
  "hash":                 "<str>",
  "server_request_id":    <int>,
  "server_status":        "<str>",
  "parse_error":          "",
  "stream_parse_error":   ""
}
```

`w3_probe` and `w3_sanity` rows are excluded from `summary.csv`
aggregation. Probe outcomes are summarized into the per-cell manifest
counters (`w3_probe_attempts`, `w3_probe_ok`, `w3_probe_max_ms`); sanity
outcomes into `w3_sanity_ok` / `w3_sanity_hash_ok`.

## summary.csv schema

Per-(mode, workload, decode_budget, metric) row.

```text
mode,workload,decode_budget,metric,n,min_ms,p50_ms,p95_ms,p99_ms,max_ms,mean_ms
```

Warmups excluded. Trials with `parse_error != ""` or `http_status != 200`
(except W3, where non-200 / no response is the expected disconnect
outcome) are excluded from latency aggregation but counted in
manifest `trials_ok`.

`metric` values:

```text
W1:  response_complete_ms
W2:  first_event_ms, first_token_event_ms, response_complete_ms,
     inter_event_gap_ms
W3:  first_event_ms, disconnect_ms, bytes_received
```

## Caveats baked into every output

- Single-machine Apple M4 darwin, TinyLlama 1.1B Q4_K_M, single client.
- cpp-httplib worker threads are foreign to HPX placement; engine-pool
  placement affects HPX engine-task and HPX-resolved future paths only.
- Decode dominates total latency for budget>=64. Mode-effect on W1/W2
  budget=64 is expected to be small.
- `disconnect_ms` is client-side socket close, not server-observed
  cancel time.
- `LLAMA_HPX_PLACEMENT_TRACE=1` perturbs timing via extra stderr writes
  and is only enabled in the validation `engine_pool_os2` legs.
- Phase 1 makes no llama-server comparison and no concurrent-client
  claim.

## Known issue / harness decision (W3 post-disconnect spacing)

- W3 originally used a `wait_for_cap_free` sanity POST between
  disconnect trials. The probe issued a `decode_budget=1` natural
  completion to verify the cpp-httplib capacity lease had been
  released by the previous cancelled request.
- Targeted debug showed the disconnect itself is handled correctly:
  the Python client tears the socket down, the server observes the
  disconnect / cancel, and the engine finalizes the cancelled
  request.
- The probe, however, inserted a natural-completion request between
  disconnects and exposed a separate admission issue. The reproducer
  pattern is:
    `disconnect -> natural completion probe -> next disconnect`
  Under that pattern the third request is `arrival_drained` and
  `request_queued` but never admitted. The benign scenario
  `disconnect -> sleep -> disconnect` does not reproduce it.
- Exp 14 Phase 1 measures client-visible disconnect behavior, not
  multi-cycle admission. To keep Phase 1 honest, the harness avoids
  the probe and uses a fixed `--w3-post-disconnect-sleep` (default
  2.0 s) between W3 disconnect trials and before the W3 post-cell
  sanity POST. The sleep value is recorded in `manifest.json` and
  the `commands.txt` header.
- The cancelled-then-completed admission issue should become a future
  correctness/regression slice — likely a dedicated
  `hpx_server_multi_cycle_disconnect_smoke` or an engine-level
  reproducer — and is tracked outside Phase 1.
- Update: this admission issue was tracked as **N5a** (cancel_freed
  → natural completion slot recovery) and is now **fixed**
  (`finalize_and_fulfill` returns idle-origin slots to `free_idle_` by
  seq_id range, so a `cancel_freed`-admitted slot that completes
  naturally is recovered for later admission). A separate engine
  shutdown-liveness issue found while building the N5 reproducer
  (**N5b** — a queued-but-unadmittable request blocking
  `request_shutdown`) was **isolated and fixed** first; the N5b
  `failed_reserved` shutdown drain is what makes the N5a pre-fix red
  clean (`docs/hpx/n5_deferred_slot_recovery_note.md`). The W3
  fixed-sleep workaround below predates the N5a fix and is retained for
  Phase 1; it has not been re-tuned and this records no performance
  claim.
- Post-N5a/N5b, `bench.py` accepts `--w3-between-trial-mode {sleep,
  capfree}`. Default remains `sleep` to preserve Phase 1 behavior and
  the `--w3-post-disconnect-sleep` knob. The new `capfree` mode
  restores the original `wait_for_cap_free` probe between W3
  disconnect trials and before the W3 post-cell sanity POST. Each
  probe writes a `workload="w3_probe"` row to the cell JSONL and
  updates the per-cell `w3_probe_attempts` / `w3_probe_ok` /
  `w3_probe_max_ms` manifest counters; a single failed probe fails
  the cell loudly via the PASS/FAIL gate. The `capfree` mode is
  intended for refresh validation that the N5a fix removed the
  admission bug that originally motivated the fixed sleep; no
  performance claim is attached to it.
- This is a harness decision and not a performance claim.
