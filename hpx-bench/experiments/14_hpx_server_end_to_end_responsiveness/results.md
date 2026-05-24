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
- W3 uses a fixed `--w3-post-disconnect-sleep 2.0` between
  disconnect trials and before the post-cell sanity POST instead
  of the older `wait_for_cap_free` probe. See section 11.
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
- Update: this admission issue is now tracked as **N5a** and remains
  **deferred / not fixed**. A separate engine shutdown-liveness issue
  (**N5b** — a queued-but-unadmittable request blocking
  `request_shutdown`) was isolated and **fixed**
  (`docs/hpx/n5_deferred_slot_recovery_note.md`). The W3 fixed-sleep
  workaround remains; N5a, not N5b, is what keeps it necessary.
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
