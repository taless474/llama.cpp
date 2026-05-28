# Experiment 17 — HPX vs llama-server external throughput smoke

## Question

> For a short fixed prompt, greedy 8-token generation, and concurrency
> c ∈ {1, 2}, how do `llama-server` and `llama-hpx-server` (default/B=0)
> compare on **external request-level** throughput and latency?

This is a small smoke, not a benchmark: 80 measured requests total, one
machine, one model, one fixed prompt. It exists to situate the two servers'
external behavior and to point the next performance investigation in the right
direction.

A follow-up (see `results.md` §7) removes the diag-metrics confound (hpx
diagnostics OFF) and extends concurrency to c ∈ {1, 2, 4, 8} at 40 requests/cell,
locating the widest external gap at c=4.

## Prior-reference note

No existing experiment is a clean HPX-vs-llama-server throughput benchmark.

- **Exp14** is `hpx-server`-only end-to-end *responsiveness* (round-trip /
  first-event / completion / disconnect latency). Its readme scopes the
  question "within hpx-server only," excludes llama-server from Phase 1, and
  makes no "X is faster" claim. It is not a cross-server throughput experiment.
- **Exp15** is the *closest* cross-server reference — same model, prompts,
  greedy sampler pinning via the Exp12 pair-run harness, and a Phase-2b
  `concurrent_bench.py` that even computes a `client_observed_rps`. But Exp15
  is a **semantic-alignment** experiment, and its rps field is explicitly
  annotated *"approximate; client-observed; includes loopback; not a server
  throughput claim."*
- **Exp17 (this)** is a new small throughput smoke. It reuses the Exp12/Exp15
  llama-server invocation shape and the Exp16/W1c hpx-server shape, with a
  purpose-built external driver. It does not rerun or modify Exp14/15/16.

## Run shape

```text
model            tinyllama-1.1b-chat-v1.0.F32.gguf
prompt           "Hello, my name is"  (fixed, short; NOT L1024)
decode budget    n_predict / decode_budget = 8
sampling         greedy / deterministic
concurrency      c ∈ {1, 2}   (client-controlled in-flight count)
requests         20 measured + 1 warmup per cell
slots            2  (hpx --n-seq-max 2 / llama --parallel 2), fixed both cells
ctx-size         4096
streaming        no (non-streaming /completion)
threads          each server at its own default (neither flag set)
backends         llama-server, llama-hpx-server (default/B=0, no policy)
cells            2 backends × 2 concurrency = 4 cells, 80 requests total
boots            one server boot per backend serving both cells, sequential
```

External metrics only. HPX internal per-iteration JSONL is **not** used as a
cross-backend comparison metric.

## Exact commands

llama-server (one boot, stopped via SIGTERM — no `/shutdown` endpoint):

```text
llama-server --model <tinyllama-1.1b-chat-v1.0.F32.gguf> \
  --host 127.0.0.1 --port 8090 --ctx-size 4096 --parallel 2 \
  --no-context-shift --seed 0
```

llama-hpx-server (one boot, clean POST `/shutdown`):

```text
LLAMA_HPX_DIAG_METRICS=1 LLAMA_HPX_DIAG_ENABLE_SHUTDOWN=1 \
LLAMA_HPX_DIAG_METRICS_PATH=<out>/diag.jsonl \
llama-hpx-server --model <...F32.gguf> --port 8090 \
  --n-seq-max 2 --max-concurrent 8 --max-prompt-tokens 1280 --ctx-size 4096
```

No `--prefill-budget-rows` (default/unbounded == B=0). The diag env is set only
so `/shutdown` exits cleanly; the resulting `diag.jsonl` is **not** read by the
analyzer. See the caveat in `results.md` — diag metrics may add overhead.

Request bodies / sampling:

```text
llama-server  POST /completion : pinned greedy body —
  temperature=0, top_k=1, top_p=1.0, min_p=1.0, repeat_penalty=1.0,
  samplers=["top_k","temperature"], seed=0, n_predict=8,
  stream=false, cache_prompt=false
hpx-server    POST /completion : {"prompt": "Hello, my name is",
  "decode_budget": 8}   (hpx greedy default)
```

## How this was run (not to be rerun)

The run is complete; raw artifacts live under `local/runs` (gitignored), not in
the source tree. Driver and analyzer:

```text
local/runs/hpx-vs-llama-server-throughput-2026-05-27/
  driver.py     backend-aware (one boot, c=1 then c=2), external-only
  analyze.py    TSVs + 2 plots (run under local/.venv-plot for matplotlib)
  <backend>/c{1,2}/client.jsonl, <backend>/run_meta.json, server.std*
```

See `facts.md` for stable design-time facts and field definitions, and
`results.md` for the recorded numbers, interpretation, caveat, and non-claims.
