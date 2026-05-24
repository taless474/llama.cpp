# Experiment 14 — hpx-server end-to-end client-visible responsiveness (Phase 1)

## Purpose

N4 made `llama-hpx-server` able to run its engine HPX task on either the
default HPX pool or on a single-PU named `engine` thread pool (`--engine-pool`
+ `--hpx-os-threads`). Experiment 13 measured the **engine-internal** effect
of placement (timestamps stamped inside the engine task). Experiment 14
measures the **end-to-end client-visible** effect through the cpp-httplib
HTTP/SSE adapter.

The question is **within hpx-server only**:

> Does HPX engine-pool placement change what an HTTP/SSE client actually
> observes — round-trip latency, first-event latency, full-stream completion
> latency, and client-side disconnect/cancel latency?

llama-server is intentionally not in Phase 1. The first question is placement
inside hpx-server now that N4 exists; including llama-server would re-run the
Exp 12 cross-server question and dilute the placement signal.

## Modes

All three modes start the same hpx-server binary, change only the HPX runtime
config and engine-pool placement:

```text
default_os1      --hpx-os-threads 1             (no --engine-pool)
default_os2     --hpx-os-threads 2             (no --engine-pool)
engine_pool_os2  --hpx-os-threads 2 --engine-pool
```

`default_os1` is the contended case (one HPX worker shared by engine task
and any HPX-resolved promise/stream continuation). `default_os2` gives a
second OS worker. `engine_pool_os2` pins the engine task to its own named
single-PU HPX pool. Across all modes, cpp-httplib worker threads are
foreign to HPX placement.

## Workloads

```text
W1  round-trip:          POST /completion non-streaming. Measures
                         response_complete_ms (request_start -> body fully
                         read). Anchor for HTTP path under each mode.

W2  stream-full:         POST /completion stream=true, drained to the
                         terminal `event: done`. Measures
                         first_event_ms / first_token_event_ms /
                         response_complete_ms and inter-event cadence.

W3  stream-disconnect:   POST /completion stream=true. Client aborts
                         after the first complete SSE record (>=64
                         bytes). Measures first_event_ms and
                         disconnect_ms (request_start -> client socket
                         close completion). After the trial loop, one
                         sanity non-streaming POST verifies the server
                         survived (canonical greedy hash unchanged).
```

Concurrent clients, additional disconnect points (e.g. mid-token), and
backpressure scenarios are out of scope for Phase 1.

## Shape (Phase 1 main run)

```text
modes                = default_os1, default_os2, engine_pool_os2
workloads            = w1, w2, w3
trials               = 30
warmup-trials        = 1     (excluded from aggregation)
decode-budgets       = 8, 64  (W1 + W2 only)
w3-decode-budget     = 128    (long enough for cancel to matter)
prompt               = "Hello, my name is"
model                = tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
canonical-hash (b8)  = 0x0619d4d1900c2365
```

**One server process per (mode, workload) cell** in the main run. Cells
run strictly sequentially; never two model-loading processes at once.
This is slower than reusing the same server across W1/W2/W3 for a mode
but isolates each workload from any state a previous workload (streaming
buffers, half-cancelled requests in flight at SIGTERM) might leave behind.

Cell count: 3 modes x (2 budgets W1 + 2 budgets W2 + 1 budget W3) =
3 x 5 = 15 cells. Each cell = one launch + 1 warmup + 30 measured trials
(+ 1 sanity request for W3 cells).

## How to run

Validation run (fast, single budget):

```text
python3 bench.py \
  --binary /Users/unick/Desktop/hpx/builds/llama-hpx-hpx-on/bin/llama-hpx-server \
  --model  /Users/unick/Desktop/hpx/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --modes default_os1,default_os2,engine_pool_os2 \
  --workloads w1,w2,w3 \
  --trials 3 --warmup-trials 1 \
  --decode-budgets 8 \
  --w3-decode-budget 128 \
  --label validate \
  --placement-trace-engine-pool
```

`--placement-trace-engine-pool` enables `LLAMA_HPX_PLACEMENT_TRACE=1`
**only** for the `engine_pool_os2` legs, so the validation manifest
captures `engine_task_placement pool=engine` evidence without perturbing
the other modes' timing.

Main Phase 1 collection (no placement-trace env, two budgets for W1/W2):

```text
python3 bench.py \
  --modes default_os1,default_os2,engine_pool_os2 \
  --workloads w1,w2,w3 \
  --trials 30 --warmup-trials 1 \
  --decode-budgets 8,64 \
  --w3-decode-budget 128 \
  --label phase1
```

(`--binary` and `--model` default to the common local layout; override
them if your checkout differs.)

## Output layout

```text
results/<run_id>/
  manifest.json                       run metadata, git SHA, model+binary
                                      fingerprints, per-cell command +
                                      readiness/exit status
  commands.txt                        exact command line per cell
  raw/<mode>_<workload>_b<budget>.jsonl
                                      per-request JSONL (warmups included,
                                      flagged by `is_warmup`)
  logs/<mode>_<workload>_b<budget>.stdout
  logs/<mode>_<workload>_b<budget>.stderr
  logs/<mode>_<workload>_b<budget>.args
  summary.csv                         p50/p95/p99/max/mean/min/N per
                                      metric per (mode, workload, budget);
                                      measured records only (warmups
                                      excluded)
```

`results/` is gitignored; commit a curated `results.md` after a run worth
keeping.

## PASS/FAIL

Phase 1 PASS/FAIL is correctness, not latency:

- server starts and reaches HTTP readiness in each cell;
- W1 trials all HTTP 200; for budget=8, hash equals `0x0619d4d1900c2365`;
- W2 trials all see a terminal `event: done` with `status=completed`;
  for budget=8, the done-record hash equals `0x0619d4d1900c2365`;
- W3 trials all complete the disconnect (socket closed locally);
  the post-trial sanity request returns HTTP 200 with the canonical
  greedy hash for budget=8;
- server exits cleanly (SIGTERM grace OK).

Latency values are observations, not pass conditions. Phase 1 makes no
"X is faster" claim from this data.

## Boundary notes

- Phase 1 does not change `hpx-server`, the engine, the runtime, or any
  smoke. The driver only launches processes and aggregates JSONL.
- cpp-httplib worker threads are foreign to HPX placement regardless of
  mode; HPX engine-pool placement affects the engine task and
  HPX-resolved promise/stream paths only.
- Decode dominates total latency for long budgets; observed mode
  differences for W2 budget=64 are likely small and inside noise. This
  is consistent with the Exp 13 W3 result and is expected, not a bug.
- W3 `disconnect_ms` is client-side socket close, not server-observed
  cancel time. Surfacing server-observed cancel timing would require
  small hpx-server instrumentation; that is a Phase 2 candidate.
- `LLAMA_HPX_PLACEMENT_TRACE=1` is enabled only for the validation
  engine_pool_os2 leg, never in main timing runs.

See `facts.md` for the stable design-time facts and JSONL/CSV field
definitions. See `results.md` (written after the main run) for the
recorded numbers and per-mode interpretation.
