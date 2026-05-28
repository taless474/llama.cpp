# Experiment 14 Phase 1 — results

End-to-end client-visible responsiveness of `hpx-server` across three
HPX placement modes. Phase 1 is a single-machine, single-client run on
Apple M4, TinyLlama 1.1B Q4_K_M, canonical prompt `"Hello, my name is"`.

## 1. Files changed in this slice

```text
M  hpx-bench/experiments/14_hpx_server_end_to_end_responsiveness/bench.py
M  hpx-bench/experiments/14_hpx_server_end_to_end_responsiveness/facts.md
A  hpx-bench/experiments/14_hpx_server_end_to_end_responsiveness/results.md   (this file)
```

(Both `bench.py` and `facts.md` live inside the still-untracked Exp 14
directory, so `git status --short` reports them as part of the
`??` entry on `hpx-bench/experiments/14_hpx_server_end_to_end_responsiveness/`.)

No engine, runtime, hpx-server, smoke, or build-system source files
were touched. No commits, no staging, no pushes.

## 2. Validation run

- run_id: `20260523-161300-validate`
- dir: `results/20260523-161300-validate/`
- result: `EXP14_PHASE1: PASS`
- cells: 9 / 9 PASS (3 modes x {w1 b8, w2 b8, w3 b64})
- per cell: 30/30 trials_ok=3/3 (warmup excluded)
- W1 b8 canonical hash `0x0619d4d1900c2365` matched 3/3 in every mode
- W2 b8 done_seen=3/3, canonical hash matched 3/3 in every mode
- W3 b64 disconnect trials completed without the 30 s line timeout
- W3 post-cell sanity HTTP 200 + canonical hash in every mode
- `engine_pool_os2` argv shows `--engine-pool`; server started/served
  successfully (manifest `placement_trace_seen=false` because the
  validation command did not set `--placement-trace-engine-pool`)
- no stuck `hpx-server` process after the run
- `w3_post_disconnect_sleep_s=2.0` recorded in `manifest.json`

## 3. Main Phase 1 run

- run_id: `20260523-161417-phase1`
- dir: `results/20260523-161417-phase1/`
- result: `EXP14_PHASE1: PASS`
- 15 cells (3 modes x {w1 b8, w1 b64, w2 b8, w2 b64, w3 b128})
- 30 measured trials per cell + 1 warmup
- elapsed: 2026-05-23T23:14:17Z -> 2026-05-23T23:19:31Z (~5 min 14 s)
- every cell terminated with `sigterm`, no `sigkill`
- W1/W2 b8 canonical hash matched 30/30 in every mode
- W2 done_seen=30/30 in every mode (b8 and b64)
- W3 post-cell sanity HTTP 200 + canonical hash in every mode
- `w3_post_disconnect_sleep_s=2.0` recorded in `manifest.json`
  and the header of `commands.txt`

## 4. W1 (non-streaming) — `response_complete_ms`

n = 30 measured trials per cell.

```text
mode             budget   p50      p95      p99
default_os1      8         98.29   102.98   103.77
default_os1      64       568.89   585.35   613.40
default_os2      8         89.85    91.99    97.24
default_os2      64       507.56   510.24   524.33
engine_pool_os2  8         93.38    97.04    99.84
engine_pool_os2  64       535.54   547.69   552.03
```

All values in milliseconds.

## 5. W2 (streaming, drained to terminal `done`)

n = 30 measured trials per cell. `inter_event_gap_ms` aggregates
consecutive event-arrival deltas across all events in all measured
trials (n = budget x 30, e.g. 64 x 30 = 1920 for b64).

```text
mode             budget   metric                p50      p95      p99
default_os1      8        first_event_ms         38.77    40.49    40.89
default_os1      8        first_token_event_ms   38.77    40.49    40.89
default_os1      8        response_complete_ms   96.96   103.43   112.53
default_os1      8        inter_event_gap_ms      8.03     9.35    11.07
default_os1      64       first_event_ms         40.32    40.90    40.97
default_os1      64       first_token_event_ms   40.32    40.90    40.97
default_os1      64       response_complete_ms  597.74   610.01   622.62
default_os1      64       inter_event_gap_ms      8.73     9.40    10.66

default_os2      8        first_event_ms         37.59    37.78    38.42
default_os2      8        first_token_event_ms   37.59    37.78    38.42
default_os2      8        response_complete_ms   90.22    93.24    94.53
default_os2      8        inter_event_gap_ms      7.44     7.91     8.27
default_os2      64       first_event_ms         37.66    37.82    37.98
default_os2      64       first_token_event_ms   37.66    37.82    37.98
default_os2      64       response_complete_ms  507.49   527.02   533.24
default_os2      64       inter_event_gap_ms      7.43     7.78     8.98

engine_pool_os2  8        first_event_ms         37.85    38.02    38.03
engine_pool_os2  8        first_token_event_ms   37.85    38.02    38.03
engine_pool_os2  8        response_complete_ms   93.74    97.83   100.31
engine_pool_os2  8        inter_event_gap_ms      7.92     8.60     9.16
engine_pool_os2  64       first_event_ms         37.90    38.29    38.33
engine_pool_os2  64       first_token_event_ms   37.90    38.29    38.33
engine_pool_os2  64       response_complete_ms  519.00   560.09   592.71
engine_pool_os2  64       inter_event_gap_ms      7.61     8.33     9.24
```

All values in milliseconds. `first_event_ms == first_token_event_ms`
in every cell because the very first SSE record the client sees is
already an `event: token` record on the current `hpx-server` adapter.

## 6. W3 (streaming + client abort after ~64 bytes received)

n = 30 measured trials per cell at decode budget 128.

```text
mode             metric              p50      p95      p99
default_os1      first_event_ms       49.62    55.18    56.92
default_os1      disconnect_ms        57.95    64.80    66.04
default_os2      first_event_ms       54.37    57.06    61.27
default_os2      disconnect_ms        62.18    64.93    69.17
engine_pool_os2  first_event_ms       54.76    57.29    58.21
engine_pool_os2  disconnect_ms        62.77    65.25    66.22
```

All values in milliseconds. `bytes_received` was constant 107 in
every trial (the SSE framing of one `event: token` record carrying
the first sampled token, then the abort).

Post-cell sanity status (one budget=8 non-streaming POST per W3 cell,
after the 2.0 s post-disconnect sleep):

```text
default_os1       w3_sanity_ok=True   w3_sanity_hash_ok=True
default_os2       w3_sanity_ok=True   w3_sanity_hash_ok=True
engine_pool_os2   w3_sanity_ok=True   w3_sanity_hash_ok=True
```

## 7. Interpretation — `default_os1` vs `default_os2`

Going from 1 to 2 HPX OS worker threads improves end-to-end decode
latency in W1 and W2 but is essentially neutral on the W3
client-visible disconnect path.

- W1 b64 p50: `568.89 -> 507.56 ms` (-10.8 %)
- W1 b8 p50: ` 98.29 ->  89.85 ms` (-8.6 %)
- W2 b64 `response_complete_ms` p50: `597.74 -> 507.49 ms` (-15.1 %)
- W2 b8 `response_complete_ms` p50: ` 96.96 ->  90.22 ms` (-7.0 %)
- W2 `first_event_ms` (b64) p50: ` 40.32 ->  37.66 ms` (-6.6 %)
- W2 `inter_event_gap_ms` (b64) p50: `  8.73 ->   7.43 ms` (-14.9 %)
- W3 `first_event_ms` p50: ` 49.62 ->  54.37 ms` (+9.6 %)
- W3 `disconnect_ms` p50: ` 57.95 ->  62.18 ms` (+7.3 %)

The W3 numbers go the "wrong" way (slightly worse with 2 OS threads)
but the absolute deltas are within ~5 ms at p50 / p99 and the client
abort path is dominated by network/socket teardown timing, not by
HPX scheduling. We do not draw a conclusion from this delta in
Phase 1.

## 8. Interpretation — `default_os2` vs `engine_pool_os2`

Engine-pool placement is **slightly negative-to-neutral** on every
end-to-end client-visible metric in Phase 1.

- W1 b64 p50: `507.56 -> 535.54 ms` (+5.5 %)
- W1 b8 p50: ` 89.85 ->  93.38 ms` (+3.9 %)
- W2 b64 `response_complete_ms` p50: `507.49 -> 519.00 ms` (+2.3 %)
- W2 b8 `response_complete_ms` p50: ` 90.22 ->  93.74 ms` (+3.9 %)
- W2 b64 `inter_event_gap_ms` p50: `  7.43 ->   7.61 ms` (+2.4 %)
- W2 `first_event_ms` (b64) p50: ` 37.66 ->  37.90 ms` (+0.6 %)
- W3 `first_event_ms` p50: ` 54.37 ->  54.76 ms` (+0.7 %)
- W3 `disconnect_ms` p50: ` 62.18 ->  62.77 ms` (+0.9 %)

The shifts are small (single-digit %) and the W3 metrics are
essentially equal at p50 and p99. We do not see an end-to-end gain
from engine-pool placement on this workload, but we also do not see
a regression at the precision Phase 1 supports.

## 9. Does engine-pool help, hurt, or stay neutral end-to-end?

In Phase 1 it is **neutral-to-slightly-negative** on every measured
end-to-end client-visible metric. This is consistent with Exp 13:

- Exp 13 found that engine-pool placement helps the
  control-plane / queued-cancel tail (HPX-internal scheduling),
  which is not on the client-visible critical path measured here.
- Exp 13 also found that engine-pool does not help and may slightly
  worsen W3 streaming completion on a decode-dominated workload —
  again consistent with the small positive deltas seen here on
  `response_complete_ms` for b64.
- No end-to-end client metric in Phase 1 contradicts Exp 13.

Recommendation as of this run: keep `--engine-pool` opt-in and
diagnostic. Do not enable it by default for end-to-end serving on
this hardware/workload until a workload appears where the
control-plane wins outweigh the small end-to-end overhead.

## 10. Caveats

- Single machine: Apple M4 Pro, darwin, TinyLlama 1.1B Q4_K_M.
- Single client (one Python `http.client` connection at a time;
  cells run strictly sequentially; never two model-loading
  processes overlap).
- `disconnect_ms` is the client-side socket close return time,
  not a server-observed cancel timestamp. Phase 1 has no
  server-side cancel timestamp yet.
- W3 between-trial spacing (`--w3-between-trial-mode`, see
  section 11):
  - fixed sleep remains the Phase 1 default for compatibility with
    recorded Phase 1 results;
  - capfree mode is available for post-N5a/N5b refresh validation;
  - this does not re-baseline Phase 1.
- No `LLAMA_HPX_PLACEMENT_TRACE` was set in the main run. The
  `engine_pool_os2` cells' argv contains `--engine-pool`, the
  binary launched successfully, and Exp 13 already demonstrated
  `engine_task_placement pool=engine` under the same binary.
- No llama-server comparison in Phase 1. (`hpx-server`-only.)
- W1/W2 b64 hashes are recorded but not asserted (no canonical
  b64 anchor); the hash gate is b8 only.
- Decode dominates W1/W2 b64 totals. Mode effects on b64 are
  expected to be small relative to decode time.

## 11. Known issue — `cancelled -> completed -> queued-not-admitted`

(Full note in `facts.md`.)

- W3 originally used a `wait_for_cap_free` probe between disconnect
  trials. The probe was a `decode_budget=1` natural-completion POST
  intended to verify that the cpp-httplib capacity lease from the
  previous cancelled request had been released.
- Targeted debug showed the disconnect itself is handled
  correctly: the Python client tears the socket down, the server
  observes the disconnect / cancel, and the engine finalizes the
  cancelled request.
- The probe exposed a separate admission issue with the pattern
  `disconnect -> natural completion -> next disconnect`. Under
  that pattern the third request reaches `arrival_drained` and
  `request_queued` but is never admitted, and the trial line-
  timeouts at ~30 s. The benign pattern `disconnect -> sleep ->
  disconnect` does not reproduce it.
- Phase 1 measures client-visible disconnect behavior only, not
  multi-cycle admission. To keep Phase 1 honest, the harness now
  uses a fixed `--w3-post-disconnect-sleep` (default 2.0 s)
  between W3 trials and before the W3 post-cell sanity POST.
- This admission issue should become a future correctness /
  regression slice — likely a dedicated
  `hpx_server_multi_cycle_disconnect_smoke` or an engine-level
  reproducer — and is tracked outside Phase 1.
- Update: this admission issue was tracked as **N5a** and is now
  **fixed** — `finalize_and_fulfill` recovers a `cancel_freed`-admitted
  slot that completes naturally, so a later request is admitted. A
  separate engine shutdown-liveness issue (**N5b** — a
  queued-but-unadmittable request blocking `request_shutdown`) was
  isolated and **fixed** first
  (`docs/hpx/n5_deferred_slot_recovery_note.md`). The W3 fixed-sleep
  workaround predates the N5a fix and is retained for Phase 1; it has
  not been re-tuned and this records no performance claim.
- Refresh: a small post-N5a/N5b W3-only validation
  (`run_id=20260524-142251-n5-refresh-w3-capfree-validate`, 1 warmup +
  3 trials per mode) and a 10-trial sleep-vs-capfree comparison
  (`run_id=20260524-142531-n5-refresh-w3-sleep-compare`,
  `run_id=20260524-142655-n5-refresh-w3-capfree-compare`) both PASS in
  all three modes. In capfree mode every `wait_for_cap_free` probe
  returned True (max ~72 ms vs the 30 s timeout) and the W3 sanity hash
  matched, confirming that the admission bug originally tracked as N5a
  is no longer observed on this harness shape. The 10-trial capfree
  distributions are visibly tighter than the matching sleep
  distributions; this is consistent with the capfree probe removing
  queue-wait contamination from per-trial measurements rather than a
  performance change. The fixed `--w3-post-disconnect-sleep 2.0`
  workaround remains the Phase 1 default; capfree is available via
  `--w3-between-trial-mode capfree` for refresh validation. No
  performance claim, and no llama-server comparison was performed.
- This is a harness decision and not a performance claim.

## 12. `git status --short` (end of run)

```text
 M README.md
 M tools/hpx-continuous-batch-gate/CMakeLists.txt
 M tools/hpx-continuous-batch-gate/cli.cpp
 M tools/hpx-continuous-batch-gate/cli.h
 M tools/hpx-continuous-batch-gate/engine.cpp
 M tools/hpx-continuous-batch-gate/engine.h
 M tools/hpx-continuous-batch-gate/hpx-continuous-batch-gate.cpp
 M tools/hpx-continuous-batch-gate/hpx_runtime.cpp
 M tools/hpx-continuous-batch-gate/hpx_runtime.h
 M tools/hpx-continuous-batch-gate/types.h
 M tools/hpx-server/CMakeLists.txt
 M tools/hpx-server/hpx-server.cpp
?? .claude/scheduled_tasks.lock
?? docs/hpx/hpx_serving_layer_architecture.md
?? hpx-bench/experiments/13_control_plane_responsiveness/
?? hpx-bench/experiments/14_hpx_server_end_to_end_responsiveness/
?? tools/hpx-continuous-batch-gate/engine_queued_cancel_engine_pool_smoke.cpp
?? tools/hpx-continuous-batch-gate/hpx_continuous_batch_responsiveness_bench.cpp
?? tools/hpx-server/hpx_server_engine_pool_smoke.cpp
```

The `results/` subdirectory under
`hpx-bench/experiments/14_hpx_server_end_to_end_responsiveness/`
is gitignored and does not appear in `git status`. Nothing has been
staged, committed, or pushed.

## 13. Exp14 Phase 1R — post-N5a/N5b capfree refresh

`run_id`: `20260524-144031-phase1r-capfree`. Results dir:
`results/20260524-144031-phase1r-capfree/`. Driver capture:
`local/runs/exp14/phase1r-capfree/driver.{stdout,stderr}`.

Same shape as the original Phase 1 main except W3 uses
`--w3-between-trial-mode capfree`:

```text
python3 bench.py \
  --modes default_os1,default_os2,engine_pool_os2 \
  --workloads w1,w2,w3 \
  --trials 30 \
  --warmup-trials 1 \
  --decode-budgets 8,64 \
  --w3-decode-budget 128 \
  --w3-between-trial-mode capfree \
  --label phase1r-capfree
```

15 cells = 3 modes × (2 W1 budgets + 2 W2 budgets + 1 W3 budget).
Every cell `trials_ok=30/30 term=sigterm exit=-15`. No stuck
`llama-hpx-server` process after the run.

PASS/FAIL: `EXP14_PHASE1: PASS run_id=20260524-144031-phase1r-capfree`.

### 13.1 W3 capfree probe summary

31 probes per W3 cell = 30 `between_trial` + 1 `pre_sanity`. All
probes returned `ok=true`; max wait well under the 30 s timeout.

| mode             | w3_probe_attempts | w3_probe_ok | w3_probe_max_ms |
|------------------|------------------:|------------:|----------------:|
| default_os1      | 31                | 31          | 69.57           |
| default_os2      | 31                | 31          | 69.16           |
| engine_pool_os2  | 31                | 31          | 75.92           |

### 13.2 W3 sanity hash summary

| mode             | w3_sanity_ok | w3_sanity_hash_ok |
|------------------|:------------:|:-----------------:|
| default_os1      | True         | True              |
| default_os2      | True         | True              |
| engine_pool_os2  | True         | True              |

Canonical b8 hash `0x0619d4d1900c2365` matched in all three sanity
POSTs.

### 13.3 W1 — `response_complete_ms` (ms, n=30, warmup excluded)

| mode             |  b | p50    | p95    | p99    |
|------------------|---:|-------:|-------:|-------:|
| default_os1      |  8 |  96.47 | 102.96 | 103.22 |
| default_os1      | 64 | 569.02 | 583.66 | 594.40 |
| default_os2      |  8 |  91.41 |  94.48 | 104.81 |
| default_os2      | 64 | 517.64 | 521.30 | 525.04 |
| engine_pool_os2  |  8 |  92.03 |  93.18 |  93.63 |
| engine_pool_os2  | 64 | 523.83 | 544.36 | 548.05 |

W1 b=8 `trials_hash_ok = 30/30` in every mode. W1 b=64 hashes are
recorded only (no canonical b64 anchor).

### 13.4 W2 — first event, first token, completion (ms, n=30)

| mode             |  b | first_event p50/p95 | first_token p50/p95 | response_complete p50/p95 |
|------------------|---:|--------------------:|--------------------:|--------------------------:|
| default_os1      |  8 | 39.28 / 40.52       | 39.28 / 40.52       |  96.97 / 103.19           |
| default_os1      | 64 | 39.84 / 41.20       | 39.84 / 41.20       | 566.59 / 623.07           |
| default_os2      |  8 | 37.89 / 38.27       | 37.89 / 38.27       |  91.80 /  92.77           |
| default_os2      | 64 | 38.06 / 38.45       | 38.06 / 38.45       | 517.23 / 521.87           |
| engine_pool_os2  |  8 | 38.07 / 38.54       | 38.07 / 38.54       |  92.89 /  96.13           |
| engine_pool_os2  | 64 | 38.57 / 42.59       | 38.57 / 42.59       | 538.18 / 595.92           |

`first_event_ms == first_token_event_ms` in every W2 cell — the SSE
stream emits the first token event as the first record. W2 b=8
`trials_hash_ok = 30/30` per mode; b=64 hashes recorded only.
`inter_event_gap_ms` p50/p95 at b=64: default_os1 8.26 / 9.31;
default_os2 7.55 / 7.94; engine_pool_os2 7.87 / 8.83.

### 13.5 W3 — first event, disconnect, bytes (b=128, capfree)

| mode             | first_event p50 / p95 | disconnect p50 / p95 | bytes_received |
|------------------|----------------------:|---------------------:|---------------:|
| default_os1      | 38.36 / 38.71         | 46.87 / 47.53        | 107            |
| default_os2      | 37.92 / 38.63         | 45.92 / 46.81        | 107            |
| engine_pool_os2  | 38.02 / 38.64         | 46.19 / 48.69        | 107            |

`bytes_received` is constant 107 per trial (the W3 abort threshold).

### 13.6 Interpretation

- Refreshed hpx-server-only baseline after N5a/N5b. Every cell PASS,
  no hangs, no SIGKILLs, no stuck processes; the W3 capfree path is
  stable at full Phase 1 size (31 probes per W3 cell, all
  `ok=true`).
- `default_os1` is the slowest on long-decode workloads (W1/W2 b=64
  `response_complete_ms` p50 about +50 ms vs `default_os2`),
  consistent with one shared OS worker contending with the engine
  task.
- `default_os2` is the fastest in most metrics on this shape.
- `engine_pool_os2` is approximately neutral relative to
  `default_os2` on end-to-end client metrics: first-event /
  first-token p50 essentially equal; W1/W2 b=64
  `response_complete_ms` p50 is slightly higher (about 5–20 ms)
  with a longer p95 tail on W2 b=64. Consistent with the existing
  Phase 1 and Exp 13 finding that engine-pool placement helps the
  HPX control plane but is not visible — and may slightly worsen —
  on decode-dominated end-to-end metrics.
- W3 distributions across all three modes are tightly clustered
  (`first_event_ms` p50 ~38 ms, `disconnect_ms` p50 ~46 ms) and the
  original W3 cap-free admission failure path is no longer
  observed.
- No llama-server comparison. No concurrent-client claim. No
  performance claim beyond this hpx-server-only refresh.

### 13.7 Caveats

- Single machine: Apple M4 Pro, darwin, TinyLlama 1.1B Q4_K_M.
- Single client; cells run strictly sequentially.
- W3 used `capfree` for harness cleanliness. Phase 1 main numbers
  (§3–§9 above) used the fixed-sleep workaround; Phase 1R W3
  timings are therefore **not directly comparable** to those W3
  numbers as a performance delta.
- W1/W2 do not use the between-trial spacing knob; their Phase 1R
  numbers are directly comparable in shape to the corresponding
  Phase 1 main W1/W2 numbers, though run-to-run noise and minor
  scheduling differences are expected.
- `disconnect_ms` is client-side socket close, not server-observed
  cancel time.
- No llama-server comparison. No broader production-server claim.
