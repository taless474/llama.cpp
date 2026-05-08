# Simulator stable facts

This document records the v1 simulator's design contract. It is not a
running log; update it only when the contract changes.

## v1 scope

Models the upstream `llama-server` `update_slots()` loop at the level of:

- slots (one per `seq_id`, fixed at startup)
- a shared logical batch built per iteration
- iteration-boundary phase transitions
- linear cost model parameterized by CostParams
- three scheduling policies side-by-side under the same workload

## Intentionally not modeled in v1

- HPX runtime (no `hpx::*`, no future-based plumbing)
- llama.cpp linkage (no `llama_decode`, no `llama_batch`, no GGUF)
- real KV cache, KV memory pressure, fragmentation, eviction
- real token values, logits, sampling
- prefix cache reuse (`server_prompt_cache`, LCP similarity)
- context shifting, SWA / recurrent state checkpoints
- speculative decoding
- LoRA / aLoRA
- multimodal (MTMD)
- `n_cmpl > 1` parent / child fan-out
- HTTP layer (no SSE, chunked encoding, sockets)
- request cancellation (the `cancel_at_ms` field is reserved for v2)
- priority scheduling (the `priority` field is reserved for v2)

## Policy definitions

- `fifo_context_pool` — each active slot pays its own decode-call base
  cost (PER_SLOT cost mode); rows from different slots are never folded
  into a single shared call. Approximates the current
  `llama-serving-bench` shape.
- `static_batching` — admission only when ALL slots are waiting; within a
  group, rows from multiple slots share one decode call (SHARED cost
  mode).
- `continuous_batching` — admission as soon as any slot frees; rows from
  multiple slots share one decode call. Prompt and decode rows can
  coexist in the same iteration.

## Default cost params

Placeholder values; not calibrated. Override from CLI before drawing any
conclusion.

| param                          | default |
| ------------------------------ | ------- |
| `base_decode_step_cost_ms`     | 1.0     |
| `per_prompt_token_cost_ms`     | 0.05    |
| `per_decode_token_cost_ms`     | 0.5     |
| `per_active_slot_overhead_ms`  | 0.0     |
| `streaming_emit_cost_ms`       | 0.0     |

## Sanity gates (asserted at every iteration boundary)

1. every submitted request completes exactly once
2. no slot owns more than one request at a time
3. `generated_tokens <= decode_tokens`
4. `completed.size() == submitted.size()` at end of run
5. no negative `prompt_remaining` or `decode_remaining`
6. `active_slots <= n_slots`
7. `batch_size <= n_batch` per call (per-slot for `fifo_context_pool`)
8. continuous batching admits only into free slots
9. fifo never shares rows from multiple slots in one decode call
10. static batching does not admit a new group until the active group drains
11. for each request: `first_token_time >= assigned_at`,
    `finish_time >= first_token_time`
12. `current_time_ms` is non-decreasing

A violation aborts the run with `SanityGateViolation`.

## First-token timing

The simulator sets `first_token_time_ms` at the end of the iteration that
produces the **first decode row** for a slot, not at the end of the
prefill iteration. This is one iteration later than upstream's "logits of
the last prefill row produce the first sampled token." The difference is
small for the cost-model parameters in use and is documented here so that
direct comparisons against measured ttft on a real engine account for it.

## Output layout

```
results/<run_id>/
  config.json
  summary.csv
  summary.md
  <policy>/requests.csv
  <policy>/iterations.csv
```

`run_id` is a 12-char SHA-256 prefix of the canonical config payload
(workload, policies, n_slots, n_batch, seed, cost_params). Same config →
same run_id → same outputs. `--timestamped` is opt-in for ad-hoc sessions.
