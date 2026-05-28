# Experiment 17 — facts

Stable design-time facts and field definitions for the HPX vs llama-server
external throughput smoke. Numbers live in `results.md`.

## Scope

- External request-level comparison only: client-observed completion and
  client-observed throughput. No server-internal timing is used to compare the
  two backends.
- llama-server exposes no per-iteration JSONL; HPX does, but it is deliberately
  not read here. The comparison is the closest external request-level shape,
  not an internally equivalent benchmark.

## Backends

```text
llama-server    /Users/.../builds/llama-base/bin/llama-server
                --parallel 2, --no-context-shift, --seed 0, pinned greedy body
                stopped via SIGTERM (no /shutdown endpoint)
hpx-server      /Users/.../builds/llama-hpx-hpx-on/bin/llama-hpx-server
                --n-seq-max 2 --max-concurrent 8 --max-prompt-tokens 1280
                default/unbounded prefill (no --prefill-budget-rows)
                stopped via POST /shutdown
```

Both use `--ctx-size 4096` and the same F32 TinyLlama model. Neither sets a
thread count; each uses its own default. This is a known non-equivalence.

## Workload constants

```text
prompt           "Hello, my name is"   (fixed; ~6 tokens incl. BOS)
decode budget    8  (n_predict for llama, decode_budget for hpx)
sampling         greedy / deterministic
concurrency      c ∈ {1, 2}
N_MEASURED       20 per cell
warmup           1 readiness warmup + 1 explicit warmup, both excluded
slots            2 fixed for both cells; client controls in-flight concurrency
```

The hpx greedy b8 anchor hash (`0x0619d4d1900c2365`) is expected and observed,
confirming deterministic greedy decode on the hpx side.

## Concurrency / wall-clock model

A cell with concurrency `c` dispatches `N_MEASURED` requests through a thread
pool of `c` workers (so up to `c` requests are in flight at once). The cell
wall clock is measured from the moment the batch is dispatched to the moment
the last request returns:

```text
batch_wall_us = t_cell_end_us - t_cell_start_us
```

## Field definitions

### client.jsonl (per request)

```text
backend, idx
client_send_us / client_close_us   monotonic µs around the POST
http_status                        HTTP status (200 expected)
server_status                      hpx: status field; llama: derived
                                   "completed" from stop/stopped_eos/limit
n_decoded                          hpx: n_decoded; llama: tokens_predicted
hash                               hpx engine hash (llama: null)
text_sha256                        client-computed sha256(text)[:16]
error                              transport/parse error or null
```

### client.jsonl cell_meta line

```text
cell_meta=true, concurrency, n_requests,
batch_wall_us, t_cell_start_us, t_cell_end_us
```

### Derived per-cell metrics (analyze.py)

```text
completed        http_status==200 and no error
failed           http_status!=200 or error set
total_tokens     sum(n_decoded) over completed
wall_s           batch_wall_us / 1e6
tokens_per_s     total_tokens / wall_s   (external, client-observed)
req_per_s        completed / wall_s
p50_ms / p95_ms  percentiles of per-request completion latency (completed)
ttft             not collected (non-streaming run)
```

## Caveats baked into interpretation

- **Diag-metrics overhead.** The original §1–§4 hpx-server run had
  `LLAMA_HPX_DIAG_METRICS=1` enabled (required, with
  `LLAMA_HPX_DIAG_ENABLE_SHUTDOWN=1`, for the `/shutdown` endpoint to exit
  cleanly), while llama-server had no equivalent. The diag-off control in
  `results.md` §7 rules this out: diagnostics accounted for only ~18% of the
  c=2 gap (+1.7 tok/s recovered), so the residual gap is not the diag-metrics
  path. The §7 scaling follow-up runs both backends with diagnostics OFF.
- Tiny sample: 20 requests/cell, single repeat per cell, one machine.
- TTFT unavailable (non-streaming); with budget 8 it would track completion.
- Default thread counts differ between servers.
- F32 TinyLlama only.
