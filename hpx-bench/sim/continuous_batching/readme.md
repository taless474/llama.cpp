# Continuous-batching simulator

Pure-Python discrete-event simulator that models the upstream `llama-server`
`update_slots()` loop at the level of slots, batch construction, and
scheduling policy. It is **not** an inference engine: no real tokens, no
logits, no KV cache, no llama.cpp linkage, no HPX.

Used to compare three policies side-by-side under the same workload:

- `fifo_context_pool` — each active slot pays its own decode-call base cost.
- `static_batching` — shared batch within a group; admit only when the group drains.
- `continuous_batching` — shared batch; admit as soon as a slot frees.

## Run the tests

```sh
cd hpx-bench/sim/continuous_batching
python3 -m unittest discover tests -v
```

## Run the exp11-like workload

```sh
cd hpx-bench/sim/continuous_batching
python3 run_experiment.py --workload exp11_like_all_short --n-slots 4 --n-batch 128 --seed 42
```

This runs all three policies on 200 short requests at `t=0` and writes
results to `results/<run_id>/`, where `run_id` is a 12-char config hash by
default (deterministic — same config produces the same dir). Use
`--timestamped` for ad-hoc sessions.

## Run the mixed-realistic workload

```sh
python3 run_experiment.py --workload mixed_realistic --seed 7 --workload-cap 100
```

Bursty Poisson arrivals across short / medium / long classes. The summary
includes per-class p50 / p95 latency.

## Re-summarize an existing run

```sh
python3 summarize.py results/<run_id>/
```

## Outputs

```
results/<run_id>/
  config.json                              # full config + git rev + command
  summary.csv                              # one row per policy
  summary.md                               # repro line + side-by-side table
  fifo_context_pool/
    requests.csv                           # per-request timings
    iterations.csv                         # per-iteration metrics
  static_batching/
    requests.csv
    iterations.csv
  continuous_batching/
    requests.csv
    iterations.csv
```

`requests.csv` columns: `request_id, arrival_time_ms, assigned_at_ms,
first_token_time_ms, finish_time_ms, prompt_tokens, decode_tokens,
queue_wait_ms, time_to_first_token_ms, total_latency_ms, prefill_ms,
decode_ms, class_label`.

`iterations.csv` columns: `iteration_index, t_iter_start_ms,
t_iter_end_ms, prefill_rows, decode_rows, batch_size, active_slots,
decode_calls_in_iteration, iter_cost_ms, admitted_this_iteration,
completed_this_iteration`.

## Cost model

Linear, configurable from CLI:

```
shared (static / continuous):
  iter_cost_ms = base + per_prompt * #prompt_rows + per_decode * #decode_rows
                 + per_slot * #active_slots + per_stream * #decode_rows

per_slot (fifo):
  iter_cost_ms = sum over contributing slots of (base + per_prompt * np + per_decode * nd + per_stream * nd)
                 + per_slot * #active_slots
```

CLI flags: `--cost-base`, `--cost-prompt`, `--cost-decode`, `--cost-slot`,
`--cost-stream`. Defaults are placeholder values; calibrate against a real
serving-bench run before drawing conclusions.

## What this is not

This is a **scheduling simulator**, not a performance benchmark. It does
not measure HPX, llama.cpp, KV cache mechanics, real prompt processing,
streaming socket overhead, or anything else that touches the actual model.
Numbers it produces are functions of the cost model, not measurements of
real hardware.
