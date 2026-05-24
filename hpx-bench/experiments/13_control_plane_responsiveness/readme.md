# Experiment 13 — control-plane responsiveness (Phase 1)

## Purpose

Earlier experiments (05–11) measured **throughput / makespan** of the
serving backends and found HPX correct but not faster on the old FIFO
context-pool path. This experiment asks a different question on the
current continuous-batch engine: **how responsive is the HPX-owned
control plane**, and does engine-task placement change it?

"Control plane" here means the latency of the orchestration events HPX
owns, *not* model execution speed:

- how fast a **queued cancellation** is observed, resolves the request
  future, and closes the stream, and
- how fast a freshly submitted streaming request is **admitted**,
  produces its **first token**, **completes**, and what its
  **inter-token cadence** looks like under K concurrent streams.

These are latency-of-orchestration measurements. They are not
correctness gates and make no token-hash or throughput claims.

## Question

For each runtime mode:

```text
W2 (queued-cancel):
  cancel_to_observed_us         cancel -> engine observes cancel
  cancel_to_future_ready_us     cancel -> request future ready
  cancel_to_stream_close_us     cancel -> terminal stream close

W3 (multi-request streaming, K concurrent):
  submit_to_admitted_us         submit -> engine admits request
  submit_to_first_token_us      submit -> consumer receives first token
  submit_to_complete_us         submit -> request future ready
  submit_to_stream_close_us     submit -> terminal stream close
  inter_token_gap_us            adapter-observed receive cadence between
                                tokens (derived from token_receive_us)
```

The W3 token timings are **adapter/consumer-observed**: `token_receive_us`
is stamped when the consumer dequeues a token event from its channel, so
`inter_token_gap_us` reflects engine production *plus* HPX channel wakeup,
OS scheduling, and foreign-thread consumer wake latency — i.e. what a
streaming client actually sees, not pure engine-side `publish_token()`
timing.

And the cross-mode question:

> Does running the engine task on a **named HPX pool**
> (`engine_pool_os2`) change control-plane responsiveness versus the
> default-pool 2-OS-thread baseline (`default_os2`)?

## Modes

All three modes run the **same** engine and workload code. Only the HPX
runtime configuration differs (set in `main()` of the bench binary):

```text
default_os1      default pool, hpx os-threads = 1, cooperative yield on pump
default_os2      default pool, hpx os-threads = 2, cooperative yield on pump
engine_pool_os2  named engine pool, hpx os-threads = 2, no cooperative yield
```

`default_os1` is the most contended case (engine task and submitter share
one OS thread, so the engine relies on cooperative yields). `default_os2`
gives the engine room on a second OS thread. `engine_pool_os2` pins the
engine task to its own named pool.

## Workloads

```text
W2  queued-cancel:    submit one request, immediately cancel it before any
                      admission source exists (initial_idle_slots=0,
                      budgets={}). The request is always resolved as a
                      queued cancellation; timing is the cancel path only.

W3  multi-request:    submit K requests rapidly (initial_idle_slots=K),
                      then drain the K streams CONCURRENTLY — one foreign
                      std::thread per stream consumer (Phase 2). Each
                      consumer drains its own stream to terminal close,
                      then awaits its own result. Measures admission,
                      first-token, completion, and inter-token receive
                      cadence under concurrency.
```

### W3 stream-drain model (Phase 2)

Phase 1 drained the K W3 streams **serially** on one consumer, which made
`inter_token_gap_us` a meaningless ~1 µs buffer-drain artifact (by the
time each stream was drained its tokens were already buffered in the
channel). Phase 2 fixes the **measurement only**: it drains the K streams
concurrently, one **foreign `std::thread`** per consumer, started right
after submit so the engine sees K live consumers from the first decode
iteration.

Foreign `std::thread`s (not HPX tasks) are deliberate: consumers are
Layer-1 adapter code (like `hpx-server` / cpp-httplib) and touch only
their own channel-receive half and result future — never llama.cpp state
or an engine internal. HPX-task consumers would contend for the same
worker pool the engine runs on (especially under `os_threads=1`),
perturbing the very scheduling being measured. `std::thread` is confined
to the benchmark binary's W3 workload; engine semantics, cancellation
semantics, and stream-close ordering are unchanged. Each record carries
`drain_mode` (`"single"` for W2, `"concurrent"` for W3 Phase 2) so the two
phases stay separable in the data.

W2 and W3 cover both terminal control-plane outcomes — queued-cancel and
full streaming completion. W1 (single-request baseline) and W4
(saturation / backpressure) are intentionally **not** part of Phase 1:
they answer steady-state / overload questions, not the
control-plane-responsiveness question above, and would dilute the W2/W3
comparison. They are deferred to a later phase.

## Shape (Phase 1 main run)

```text
modes          = default_os1, default_os2, engine_pool_os2
workloads      = w2, w3
trials         = 30
warmup-trials  = 1     (excluded from aggregation)
n-streams      = 4     (W3 only)
decode-budget  = 8
prompt         = "Hello, my name is"
model          = tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

Decoding is greedy. Decode budget is an upper bound; W2 requests never
decode (they are cancelled while queued). These are responsiveness
timings, so token hashes are not asserted here — see the gate tools for
correctness fingerprints.

## How to run

The driver runs every (mode, workload) cell **strictly sequentially** as
its own process. There is never more than one model-loading process at a
time — this is required to keep the microsecond-scale timings clean.

Validation run (fast):

```text
python3 bench.py \
  --binary /Users/unick/Desktop/hpx/builds/llama-hpx-hpx-on/bin/llama-hpx-continuous-batch-responsiveness-bench \
  --model  /Users/unick/Desktop/hpx/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --modes default_os1,default_os2,engine_pool_os2 \
  --workloads w2,w3 \
  --trials 3 --warmup-trials 1 --n-streams 4 --decode-budget 8 \
  --label validate
```

Main Phase 1 collection:

```text
python3 bench.py \
  --modes default_os1,default_os2,engine_pool_os2 \
  --workloads w2,w3 \
  --trials 30 --warmup-trials 1 --n-streams 4 --decode-budget 8 \
  --label phase1
```

(`--binary` and `--model` default to the common local layout; override
them if your checkout differs.)

## Output layout

```text
results/<run_id>/
  manifest.json                  run metadata, git SHA, model fingerprint,
                                 per-cell command + PASS/FAIL status
  commands.txt                   exact command line per cell
  raw/<mode>_<workload>.jsonl    per-request JSONL (warmups included)
  logs/<mode>_<workload>.stdout  binary stdout (PASS/FAIL line)
  logs/<mode>_<workload>.stderr  binary stderr (llama/ggml chatter)
  summary.csv                    p50/p95/p99/max/mean/min/N per metric,
                                 measured records only (warmups excluded)
```

`results/` is gitignored; commit a curated `results.md` if a run is worth
keeping.

## Boundary notes

- This driver and the bench binary do not change engine semantics,
  cancellation semantics, or stream-close ordering.
- All llama execution and KV mutation happen on the engine task inside
  the binary. The Python driver only launches processes and aggregates
  JSONL.
- See `facts.md` for the stable design-time facts and field definitions.
