# HPX Serving Layer — Architecture

This document is the durable layering reference for the HPX Serving Layer
for llama.cpp. It is not an implementation log, milestone tracker, or
benchmark report. Future slices should declare which layer they touch and
should justify any change that crosses layer boundaries.

It records what was already built (M0–M8, N1, N2.5, N3.0) and what the
next slices are allowed to assume.

---

## Layer 0 — Process / HPX runtime lifecycle

This layer owns:

- `hpx::start` / `hpx::finalize`
- `hpx_runtime::start_once` / `hpx_runtime::stop`
- HPX `resource_partitioner` setup
- creation of named pools (currently: the `engine` pool)

Hard rule:

```text
Engine objects never start or stop HPX.
Engine objects never own HPX runtime lifecycle.
```

Process entry brings HPX up, configures the partitioner, and tears HPX
down at exit. Engine instances may be constructed and destroyed many
times within a single HPX runtime; they are not coupled to it.

---

## Layer 1 — Adapter boundary

This layer owns:

- the CLI / continuous-batch gate binary
- `hpx-server` / HTTP adapter
- benchmark-harness adapter behavior
- request submission
- cancellation requests
- stream consumption

It does not touch llama.cpp mutable execution state. Adapters never call
`llama_decode`, `llama_memory_seq_*`, `llama_batch_*`, or any other API
that mutates a `llama_context`. The only sanctioned interaction with the
engine actor is through the typed inbox channel and the per-request
stream channel.

Adapters may run on foreign threads (cpp-httplib, benchmark drivers).
The boundary between foreign threads and the engine actor is the inbox
channel; the boundary the other direction is the per-request stream
channel.

---

## Layer 2 — Runtime placement

This layer owns:

- the default HPX pool, which hosts adapter, producer, and cancel tasks
- the optional named `engine` pool, which hosts the engine actor task
- any future stream/I/O pool (deferred; not committed)

The engine pool is opt-in (today, behind `--engine-pool` in the gate).
When it is off, the engine actor runs on the default pool and is
byte-identical to the legacy spawn form.

N4 extends this same opt-in placement to the `hpx-server` HTTP/SSE
adapter: `--engine-pool` (with `--hpx-os-threads >= 2`) runs the engine
actor on the named pool behind the live server, guarded by
`llama-hpx-server-engine-pool-smoke`. Experiment 13 measures the
engine-internal effect of this placement; Experiment 14 measures its
end-to-end client-visible effect.

Engine-pool placement is not cosmetic. It decouples engine scheduling
from producer/cancel scheduling and is the precondition for removing the
temporary cooperativity yield currently held in
`pump_inbox_nonblocking()`. The yield is load-bearing today only because
under default-pool placement (and especially `os_threads=1`) the engine
task and the producer/cancel tasks share a single worker, and the engine
inner-while body does not yield naturally. Once the engine actor lives
on a dedicated pool, that shared-worker condition disappears.

---

## Layer 3 — Engine actor

This layer owns:

- the typed inbox channel
- the engine-task-only staged queues
- `llama_context`
- `llama_batch`
- KV state and KV mutation
- per-seq sampler state
- per-request stream channel: publish and close
- per-request promise fulfillment

The engine actor is a single-mailbox, single-owner construct. All
llama.cpp mutation happens through the engine-owned path. The actor
receives typed messages (`submit`, `cancel rid`, `cancel token`,
`shutdown`) on the inbox channel, stages them into engine-task-only
deques, and applies them at well-defined phase boundaries.

**External mutation contract.** External code may submit, cancel, or
shut down through the inbox channel. External code must not mutate
`llama_context`, `llama_batch`, KV state, sampler state, stream close
state, or engine-owned request state directly. The inbox channel is the
only sanctioned write path into the engine.

**Cancellation contract.** Cancels are submitted as inbox messages but
observed only at engine iteration boundaries. Cancels never interrupt
`llama_decode`. A cancel arriving during decode is staged and acted on
at the next iteration's cancel-observation phase.

---

## Layer 4 — Engine phase contracts

This layer owns the named per-iteration phase sequence created by N3.0.
Each phase has a stable name and a stable ordering position:

1. pump inbox
2. drain cancels
3. drain arrivals
4. apply queued cancellations
5. observe cancellations
6. admit requests
7. build batch
8. `llama_decode`
9. sample / publish / finalize
10. release/ack barrier
11. idle wait / shutdown checks (outer-loop tail)

The following order constraints are load-bearing and must not move:

- cancel observation must not move inside `llama_decode`
- the three-source admission ordering (`cancel_freed` → `completion_freed`
  → `initial_idle`) must remain stable
- `publish_token` stays before `n_decoded++`
- `close_stream` stays before `set_value` on the per-request promise
- `llama_decode` stays between batch build and sampling, with no
  llama-touching work in between

These ordering rules are what make Layer 5 possible. They are also what
keeps canonical-hash anchors from drifting when phase composition
changes.

---

## Layer 5 — Future / dataflow evolution

This layer is future work only. Today the engine runs as one imperative
HPX task per repeat. A future slice may rewire the iteration into a
sequential continuation chain on the engine executor.

Entry conditions:

- Layer 4 phase contracts must already be stable.
- Use sequential continuations on the engine executor
  (`then(engine_exec, ...)`).
- Do not parallelize llama-touching phases.
- Prefer then-style sequencing for in-iteration order. Do not introduce
  `dataflow` or `when_all` around mutable llama.cpp state.

Dataflow as composition is allowed. Dataflow as parallelism over the
llama-touching phases is not.

---

## Streaming direction

The two channel directions are intentionally asymmetric:

- **Inbox channel:** adapter / producer  →  engine.
  Typed messages: submit, cancel rid, cancel token, shutdown.
  One channel per engine instance.

- **Token stream channel:** engine  →  adapter / consumer.
  Per request. Engine publishes tokens and closes the stream; the
  adapter consumes events and observes the close reason.

A future stream/I/O pool, if introduced, is a Layer 2 placement decision
for the consumer side of token-stream channels. It is independent of the
inbox channel path and must not be conflated with engine placement.

---

## Review rule for future slices

Every future slice must declare which layer it touches.

Any slice that touches multiple layers must justify why in the slice
description. Cross-layer slices are not forbidden, but they are reviewed
with stricter scrutiny than within-layer slices.

Examples:

- A change to the inbox message format is a Layer 3 slice.
- A change to the resource partitioner is a Layer 0 slice.
- Adding a new named pool is Layer 0 (creation) plus Layer 2 (placement
  policy) and must call this out.
- Converting the phase loop to dataflow is a Layer 5 slice and is gated
  on Layer 4 stability.

---

## Mapping to completed work

- **M0–M8.** Discovered and validated serving-layer behavior, including
  request lifecycle, admission, cancellation, sampling carry, and per-
  request streaming. Established the canonical correctness fingerprints.

- **N1.** Layer 3 inbox became an HPX local-channel actor. Replaced the
  spinlock + condition_variable + std::deque triad with
  `hpx::lcos::local::channel<inbox_msg>` and engine-task-only staged
  queues. Producers publish typed messages; the engine pumps the channel
  at predicate sites.

- **N2 / N2.5.** Layer 2 runtime placement was investigated and an
  opt-in named `engine` pool was added for the continuous-batch gate
  through the resource partitioner. Default-off behavior is byte-
  identical to the legacy path. Placement is evidenced by a one-shot
  `engine_task_placement` trace.

- **N3.0.** Layer 4 phase contracts were extracted. The per-iteration
  body of the engine actor was decomposed into named phase helpers with
  no behavior change; the canonical hash anchors held.

- **N2.6 (evidence only).** A temporary no-yield experiment showed that
  removing the `pump_inbox_nonblocking()` yield globally is unsafe under
  default-pool placement and `os_threads=1`: the queued-cancel pattern
  hangs because the engine and producer share a single worker. Under
  `--engine-pool` placement, the no-yield path passed. Conclusion:
  conditional gating on engine-pool-active is the right shape for a
  future implementation slice; global removal is not.
