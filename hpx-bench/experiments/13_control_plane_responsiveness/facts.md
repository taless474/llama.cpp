# Facts — Experiment 13 control-plane responsiveness (Phase 1)

This file records the stable, design-time facts. Run-specific numbers
live under `results/<run_id>/` and in a curated `results.md`.

## What is measured

Latency of HPX-owned control-plane events, in microseconds. Not
throughput, not makespan, not token-hash correctness.

## Time domain

Submitter-side and engine-side stamps both use:

```text
std::chrono::steady_clock::now().time_since_epoch() in microseconds
```

so the two sides subtract directly with no conversion. The engine-side
stamps are opt-in: the bench sets
`engine_options::lib.enable_responsiveness_timing = true` (default false
everywhere else).

## Engine-side timestamps (from request_result)

```text
t_admitted_us         set at the admit_one bind site (live admission)
t_first_publish_us    set at the first publish_token for the request
t_complete_us         set in finalize_and_fulfill, before fulfilling
t_cancel_observed_us  set in cancel_and_fulfill (engine observes cancel)
```

A queued-cancelled request (W2) keeps `t_admitted_us = -1` and
`t_first_publish_us = -1`: it is never admitted and never publishes. It
gets `t_cancel_observed_us`. JSONL renders `-1` stamps as `null`.

## Submitter-side timestamps (recorded in the bench TU)

```text
submit_us         just before eng.submit_request returns the handle
cancel_us         just before eng.cancel_request (W2 only)
future_ready_us   just after h.result.get() returns
stream_close_us   when the terminal closed stream event is observed
token_receive_us  one stamp per token event, taken when the CONSUMER
                  dequeues it from the channel (consumer-observed receive
                  time, NOT engine-side publish_token() time)
```

Field-name note: `token_receive_us` was named `token_publish_us` in the
Phase 1 binary. It always recorded consumer dequeue time, so the name was
misleading; Phase 2 renames it to `token_receive_us` (the JSONL key
changes accordingly). Old Phase 1 raw files still carry the legacy
`token_publish_us` key.

## Derived metrics

W2 (queued-cancel), all `cancel`-relative:

```text
cancel_to_observed_us      = t_cancel_observed_us - cancel_us
cancel_to_future_ready_us  = future_ready_us      - cancel_us
cancel_to_stream_close_us  = stream_close_us      - cancel_us
```

W3 (multi-request streaming), `submit`-relative plus cadence:

```text
submit_to_admitted_us      = t_admitted_us      - submit_us
submit_to_first_token_us   = t_first_publish_us - submit_us
submit_to_complete_us      = t_complete_us      - submit_us
submit_to_stream_close_us  = stream_close_us    - submit_us   (driver-derived)
inter_token_gap_us[i]      = token_receive_us[i] - token_receive_us[i-1]
```

`inter_token_gap_us` is therefore the **adapter-observed receive cadence**
between tokens: it includes engine production time plus HPX channel
wakeup, OS scheduling, and foreign-thread consumer wake latency. It is not
pure engine-side publish cadence.

`submit_to_stream_close_us` is computed by the Python driver from the raw
`submit_us` / `stream_close_us` fields; the binary does not emit it
directly. The binary does emit `cancel_to_stream_close_us` for W2.

## Invariants enforced by the binary (measured records only)

```text
submit_us >= 0
submit_us         <= t_admitted_us       (when admitted)
t_admitted_us     <= t_first_publish_us  (when first token published)
t_first_publish_us<= t_complete_us       (when completed)
t_admitted_us     <= t_complete_us
cancel_us         <= t_cancel_observed_us
t_cancel_observed_us <= future_ready_us
t_cancel_observed_us <= stream_close_us
token_receive_us monotonically non-decreasing
```

A violation makes the cell print `RESPONSIVENESS_BENCH: FAIL` and exit
non-zero. Warmup records are excluded from the invariant pass.

## Engine configuration per workload

W2:

```text
batch_capacity        = max(prompt_size, 1)
n_seq_max             = 1
initial_idle_slots    = 0        (no admission source -> always queued)
keep_alive            = true
budgets               = {}
stream_all            = false    (per-request want_stream opt-in)
cooperative_yield_on_pump = (mode != engine_pool_os2)
```

W3:

```text
batch_capacity        = prompt_size * n_streams   (fits the prefill peak)
n_seq_max             = n_streams
initial_idle_slots    = n_streams
keep_alive            = true
budgets               = {}
stream_all            = false
cooperative_yield_on_pump = (mode != engine_pool_os2)
```

## Mode -> HPX runtime config

```text
default_os1      os_threads=1, enable_engine_pool=false
default_os2      os_threads=2, enable_engine_pool=false
engine_pool_os2  os_threads=2, enable_engine_pool=true
```

Context params (both workloads): `n_ctx=2048`, `n_batch=256`,
`n_threads=2`, `n_threads_batch=2`, `n_seq_max = n_streams` for W3 else 1.

## Fixed workload constants (Phase 1)

```text
prompt        = "Hello, my name is"   (greedy decode)
decode_budget = 8
n_streams     = 4   (W3)
model         = tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

## Trial / warmup semantics

`total_trials = warmup_trials + trials`. Warmups run first and are
emitted to JSONL with `is_warmup=1` but excluded from both the binary's
invariant pass and the driver's percentile aggregation. For W3, each
trial submits `n_streams` requests, so a W3 cell yields
`trials * n_streams` measured request records.

## Record counts (Phase 1 main run, per cell)

```text
W2 cell: 30 measured records (1 request/trial)
W3 cell: 30 * 4 = 120 measured records (4 requests/trial)
```

## Stream-drain model and drain_mode (Phase 1 vs Phase 2)

Each record carries `drain_mode`:

```text
single      W2: the one stream is drained on the calling thread.
concurrent  W3 Phase 2: the K streams are drained CONCURRENTLY, one
            foreign std::thread per stream consumer, started right after
            submit so the engine sees K live consumers from the first
            decode iteration. Each consumer drains its own stream to
            terminal close, then awaits its own result.get().
legacy      (driver-assigned) old Phase 1 raw with no drain_mode field.
```

Phase 1 drained the K W3 streams serially on one consumer; tokens were
already buffered in their channels by the time each stream was reached, so
`inter_token_gap_us` collapsed to a ~1 µs buffer-drain artifact. Phase 2
changes the **measurement only** — concurrent foreign-thread consumers —
so `inter_token_gap_us` reflects real adapter-observed receive cadence.

Boundary: the consumer threads are Layer-1 adapter code (like
`hpx-server` / cpp-httplib). They touch only their own channel-receive
half and result future — never `llama_context`/`batch`/`decode`/`memory`
or any engine internal. `std::thread` is confined to the benchmark
binary's W3 workload; it does not appear in `engine.cpp`/`engine.h`/
`types.h`. Engine semantics, cancellation semantics, and stream-close
ordering are unchanged. The single-consumer-per-channel contract holds:
exactly one thread drains each stream.

## Aggregation

`summary.csv` reports `N, mean, min, p50, p95, p99, max` per
(mode, workload, drain_mode, metric), measured records only. Percentiles
use linear interpolation (numpy "linear"). `inter_token_gap_us` is
flattened across all measured records of the cell before aggregation.

## Provenance

This experiment depends on the accepted outer-tail staged-work fix in
`tools/hpx-continuous-batch-gate/engine.cpp` (the engine never suspends
on the inbox while actionable staged work exists). W2's cancel-path
timing is only trustworthy because that race is closed.
