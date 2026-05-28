# Experiment 16 — facts

Stable design-time facts and field definitions for the server-driven
`--prefill-budget-rows` W1c d2-staggered policy run. Numbers belong in
`results.md`; this file is the contract.

## Policy surface

- `llama-hpx-server` accepts `--prefill-budget-rows <int>`. When set, it
  forwards into the engine as `engine_options::lib.prefill_budget_rows` and
  caps the number of prefill rows admitted into any single `llama_decode`
  iteration to `B`.
- The flag absent (== `B=0` in this experiment) leaves engine behavior
  unchanged: prefill is unbounded per iteration. This is the **baseline**, and
  it is the fresh in-tree baseline, not the 2026-05-26 run.
- The driver's `--policy-B N` appends `--prefill-budget-rows N` to the server
  command iff `N` is given. `--policy-B 0` is treated as "set the flag to 0",
  which is the no-op natural-prefill case for this server build.

## Ownership boundary (unchanged)

Only the engine task touches `llama_context`, `llama_batch`, `llama_decode`,
`llama_memory_seq_*`. The budget policy changes *how prefill rows are packed
into the shared batch per iteration*, not who owns the context. No parallel
`llama_decode`. The server adapter only sets the option and reads diagnostics.

## Run-shape constants

```text
model               tinyllama-1.1b-chat-v1.0.F32.gguf
server binary       builds/llama-hpx-hpx-on/bin/llama-hpx-server
condition           d2-staggered (timing mode)
STAGGER_DELAY_MS    120
prompt classes      L8, L64, L256, L1024 (approx prompt token lengths)
K_CYCLES            4 per prompt class
repeats             1 (r1)
B values            0, 32, 64, 128
server flags        --n-seq-max 2 --max-concurrent 8
                    --max-prompt-tokens 1280 --ctx-size 4096
diag env            LLAMA_HPX_DIAG_METRICS=1
                    LLAMA_HPX_DIAG_ENABLE_SHUTDOWN=1
                    LLAMA_HPX_DIAG_METRICS_PATH=<repeat-dir>/diag.jsonl
```

A warmup L8 non-streaming request precedes each cell and is excluded from
analysis. After the workload the driver POSTs `/shutdown` and waits for clean
server exit.

## d2-staggered shape

For each prompt class and cycle: an **anchor** decode request is submitted; after
`STAGGER_DELAY_MS` a **probe** request of that prompt class is submitted so its
prefill lands while the anchor is mid-decode. The overlap window is where
interference cells appear. `--n-seq-max 2` admits at most the anchor + probe
pair concurrently.

## Field definitions

### diag.jsonl (engine per-iteration), key fields

```text
iter                       engine iteration index
active_seq_count           sequences live in this iteration
prefill_rows_in_iter       prompt rows packed into this iteration's batch
decode_rows_in_iter        decode (1-token) rows in this iteration's batch
tokens_emitted_in_iter     tokens produced this iteration
llama_decode_wall_us       wall time of the llama_decode call
iter_wall_us               full engine-iteration wall
non_decode_us              iter_wall_us minus decode wall (engine overhead)
prompt_class               L8 / L64 / L256 / L1024 tag
is_interference            iteration flagged as an overlap cell
```

### Derived analysis quantities

```text
interference cell          iteration with prefill_rows_in_iter > 0
                           AND decode_rows_in_iter > 0
                           AND active_seq_count == 2
max_pf                     max(prefill_rows_in_iter) over a (B, class)'s
                           interference cells  -> the policy bound check
bound                      B for B>0; for B=0 it is the natural max_pf
within_bound               max_pf <= bound
med_dec_us / p95_dec_us    decode-wall percentiles over interference cells
iters_with_token           interference iters with tokens_emitted_in_iter >= 1
                           -> decoder-not-starved evidence
```

## Invariants this run relies on

- The budget caps prefill rows per iteration; it must never raise `max_pf`
  above `B` in any interference cell for `B>0`.
- A prompt class shorter than `B` is admitted in a single iteration and its
  `max_pf` stays at the natural prompt size (e.g. L8 -> 11 for all `B`,
  L64 -> 79 for `B=128`). This is expected, not a bound violation.
- Every submitted request completes or fails explicitly; per-seq KV is cleared
  on completion; every `llama_decode` returns 0.
- The decoder is not blocked behind prefill: in overlap cells the anchor must
  keep emitting tokens while the probe's prefill is chunked.

## Caveats baked into interpretation

- Single repeat per `B`; timings are observations, not a distribution estimate.
- Decode-wall figures are over interference cells only, not whole-request
  latency.
- Cross-B token hashes are **not** correctness gates: chunk shape changes the
  batch composition and can change floating-point behavior.
- F32 TinyLlama only; no claim generalizes to other models or quantizations.
- The numeric baseline is `B=0` in this run directory. The 2026-05-26 W1c run
  is historical context only.
