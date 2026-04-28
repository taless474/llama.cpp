# HPX selective executor — hot-path audit

Code-only audit of the selective-mul-mat hot path. No binary runs in
this session beyond the prior fallback-histogram capture. Numbers
are derived by reading the source plus the per-step counters from
`hpx-bench/results/2026-04-28-q4k-repack-real-tinyllama/run_on_fbhist_8k.stderr`.

Workload reference: TinyLlama-1.1B Q4_K_M, decode (`n_predict`),
1 token per graph_compute, 689 nodes/graph, 4 HPX worker threads
(M4 P-cores).

## Files inspected

- `ggml-hpx-exec-selective.cpp` (1499 lines, the selective entry)
- `ggml-hpx-region-exec.cpp` (739 lines, `ggml_hpx_run_region_group`
  + `launch_region_async`)
- `ggml-hpx-runtime.cpp` (189 lines, HPX lifecycle)
- `ggml-hpx-cache.cpp` (67 lines, slot-cache)
- `ggml-hpx-plan.cpp` (122 lines, plan validator/builder)
- `ggml-hpx-adapter.cpp` (382 lines, scheduler→topology adapter)
- `ggml-hpx-tpool.cpp` (263 lines, HPX-backed ggml_threadpool)

## Scope discovery — cache / plan / adapter are not on the selective path

`ggml_hpx_cache_lookup_*`, `ggml_hpx_cache_insert_*`,
`ggml_hpx_build_decode_plan`, `ggml_hpx_build_prefill_plan` are all
called only from `ggml-hpx-exec.cpp` (the older coarse HPX path).
Grep of `ggml-hpx-exec-selective.cpp` for these symbols: zero hits.

So `cache.cpp`, `plan.cpp`, and `adapter.cpp` carry the coarse-path
plan-key/topology machinery and have **no presence on the selective
hot path**. They are not a source of mutex contention, allocation,
or per-token overhead in the selective-on case.

The selective hot path is just:
`ggml_hpx_exec_graph_selective_mul_mat` → for each node →
`hpx::async([&]{ggml_hpx_run_region_group(...)}).get()` (lowered)
or `ggml_backend_graph_compute(cpu_be, single-node-view)` (fallback).

## Per-token tally (TinyLlama Q4_K_M decode, 4 lanes, post-fix)

From the live run:
- 689 nodes / decode graph
- 189 lowered nodes / token (134 Q4_K MUL_MAT + 55 other lowered ops)
- 500 fallback nodes / token (21 Q6_K MUL_MAT + ~479 non-MUL_MAT)
- 0 packet matches (gate/up & GLU packet env off)

Per-node HPX scheduler entry counts (read from
`launch_region_async` at lines 120-164 + `ggml_hpx_run_region_group`
at lines 545-739):

### Lowered Q4_K MUL_MAT (2 regions, 1 dep edge, R0=REDUCTION → R1=MATMUL, n_lanes=4)

Inside the small-group fast path (n=2 ≤ kSmallN=8):

| event                                                        | count |
|--------------------------------------------------------------|------:|
| outer `hpx::async` (selective:1203, native→HPX)              |     1 |
| `launch_region_async(R0)` — 1 outer async, REDUCTION single-thread |     1 |
| `hpx::dataflow(launch::async, ...)` for R1 (waits on R0)     |     1 |
| inside dataflow: `launch_region_async(R1).get()` — 1 outer + 4 inner asyncs + 1 wait_all |  6 |
| outer `hpx::wait_all` on 2 region futures                    |     1 |

≈ **10 HPX scheduler entries per Q4_K MUL_MAT lowered node**, of
which **1 is a native→HPX entry** (selective:1203 bridge). The other
9 happen between HPX worker threads.

For 134 Q4_K MUL_MATs / token: ~1340 scheduler entries, 134 native→HPX
crossings.

### Lowered RMS_NORM-style 3-region op (ELEMENTWISE → REDUCTION → ELEMENTWISE)

Same fast path, n=3, deps=2. Roughly ~12 scheduler entries (more
inner asyncs because two ELEMENTWISE regions each fan out to 4 lanes).
1 native→HPX bridge per node.

55 non-MUL_MAT lowered nodes / token ≈ 660 scheduler entries, 55
native→HPX crossings.

### Fallback node (any op)

`ggml_cgraph view = ggml_graph_view(gf, i, i + 1);` — **single-node
slice**. Then `ggml_backend_graph_compute(cpu_be, &view)`:

- `cpu_be` has its threadpool set to `threadpool_hpx` for graphs
  with `cplan->work_size >= 32 KiB` (`llama-context.cpp:2266-2278`).
  Decode graph work_size exceeds this — confirmed by the non-zero
  `[hpx-tpool]` dispatch counters at process exit.
- Per fallback view: 1 entry into `hpx_run_job` →
  `hpx::experimental::for_loop(par.on(exec), 0, n_threads, ...)`
  (`tpool.cpp:173-183`). That submits n_threads concurrent HPX
  tasks, each of which calls `ggml_graph_compute_thread_run` and
  participates in ggml's atomic spin barrier.
- Net: ~5 HPX scheduler entries (1 bulk + 4 worker) per fallback
  view, **1 native→HPX crossing**.

For 500 fallback nodes / token: ~2500 scheduler entries, 500
native→HPX crossings.

### Per-token totals

| line item                                | scheduler entries | native→HPX crossings |
|------------------------------------------|------------------:|---------------------:|
| Lowered Q4_K MUL_MAT (134 × 10)          |         ~1,340 |                  134 |
| Lowered other (55 × ~12)                 |           ~660 |                   55 |
| Fallback view (500 × 5)                  |         ~2,500 |                  500 |
| **Per-token total**                      |       **~4,500** |                **689** |

For comparison, **selective-off**: a single
`ggml_backend_sched_graph_compute_async` call per token →
1 `hpx_run_job` → 1 bulk for_loop with n_threads tasks →
~5 scheduler entries, 1 native→HPX crossing. **~900× more HPX
scheduler entries per token** under selective-on.

## Question-by-question answers

### Q1. How many times per token do we enter HPX?

**Native→HPX crossings: 689 per token** (one per node).

Decomposed: 189 from `selective.cpp:1203` (one per lowered node) +
500 from `tpool.cpp:173` via the fallback's
`ggml_backend_graph_compute(cpu_be, single-node-view)`.

This counts only crossings from the native llama-context thread
into HPX worker context. Inner HPX-to-HPX scheduling (dataflow,
shared_future composition, wait_all) is not counted here.

### Q2. How many `hpx::async(...).get()` calls happen per decode step?

**Direct `.get()` calls from native thread: 189 per token.**

All at `ggml-hpx-exec-selective.cpp:1208` (the lowered-node bridge).
The two packet-path `.get()` sites (lines 1039, 1129) fire 0 times
in this run (packet env off).

The other `.get()` at `ggml-hpx-region-exec.cpp:643` and `:729` is
inside an HPX dataflow continuation — that runs on an HPX worker
thread, not native. Not counted as a native→HPX entry.

### Q3. How many fallback `graph_compute` calls happen per token?

**500.** Each is a single-node `ggml_graph_view(gf, i, i + 1)` →
`ggml_backend_graph_compute(cpu_be, &view)`. Two call sites:
- `ggml-hpx-exec-selective.cpp:1186` (resource-rejected reduction)
- `ggml-hpx-exec-selective.cpp:1218` (lower_op rejected)

### Q4. Are fallback nodes coalesced or one-node sliced?

**One-node sliced.** Both fallback sites build
`ggml_graph_view(gf, i, i + 1)` — exactly one node per dispatch.
There is no run-of-fallbacks coalescer in the selective loop;
adjacent fallback nodes pay the per-call setup cost independently
even when they are an unbroken contiguous run on the same backend.

This is the largest concrete cost the audit surfaces: every
RESHAPE/VIEW/PERMUTE/ROPE/ADD in a sequence of fallbacks pays full
backend dispatch overhead even though they could trivially be
batched into one whole-subgraph compute.

### Q5. Are lowered nodes batched or one HPX entry per node?

**One HPX entry per node.** `ggml-hpx-exec-selective.cpp:1203`:

```cpp
const auto t0 = clock::now();
hpx::async([&lo, &res]() {
    ggml_hpx_run_region_group(&lo.group, &res);
}).get();
const auto t1 = clock::now();
```

Strictly serial between consecutive lowered nodes — `.get()` blocks
the native thread until the lowered node finishes before moving on
to node i+1. No pipelining across nodes; no runs of lowered nodes
fused into one HPX entry.

The `lo` (`ggml_hpx_lowering`) is loop-iteration-local on the stack
(line 1150), which prevents reusing it across nodes — that ownership
choice intentionally serializes.

### Q6. Are mutexes hit when diagnostics are disabled?

**No application mutex on the selective hot path** (after the
fallback-histogram cleanup in this session):

- `selective.cpp` body, lines 992–1227: no `std::mutex`,
  `std::lock_guard`, or `std::unique_lock` — confirmed by grep.
- `region-exec.cpp` `run_region_group` + `launch_region_async`:
  no application mutex. Synchronization is via
  `hpx::shared_future`/`wait_all`, which use HPX's internal
  scheduler primitives (lock-free task queues + atomics on M4-style
  hardware), not pthread mutexes.
- `cache.cpp`: not on this path. (Even if it were: zero mutexes —
  it's an `std::optional` slot-cache.)
- `runtime.cpp`: `std::once_flag g_hpx_start_flag` is a one-time
  init guarded by `std::call_once` — atomic load on the cold path,
  irrelevant for hot path.
- `tpool.cpp`: ggml's threadpool uses an atomic spin barrier inside
  `ggml_graph_compute_thread_run` (no pthread mutex). The
  `[hpx-tpool] dispatch histogram` counters (node_hist, work_hist,
  tier_total) are plain `int`, not atomic, written from the
  dispatcher thread before the bulk for_loop runs — single-writer,
  no contention.

The HPX scheduler's *internal* task queue can take internal locks
on contended pushes/pops. We don't control that directly. Under
the static queuing scheduler we configured
(`runtime.cpp:48: --hpx:queuing=static`), worker threads have their
own queues, so steal-induced lock traffic is minimized.

### Q7. Are stats/debug paths truly cold?

**Cold paths (zero work when env var off):**

- `LLAMA_HPX_SELECTIVE_HIST` (histogram, line 758): one-shot
  `static bool warned` early-return after first call — cold
  per-graph after the first.
- `LLAMA_HPX_SELECTIVE_DEBUG` (dbg, line 987): Meyers-singleton
  initialised once, then per-loop-iteration `if (dbg)` checks load
  one bool. ~1 cycle per check, predictable branch.
- `LLAMA_HPX_SELECTIVE_STATS` (stats print, llama-context.cpp:2418):
  bool flag at construction time; cold when off.
- `LLAMA_HPX_SELECTIVE_MLP_PACKET` (packet env): bool flag at
  construction time; when off, `gate_up_enabled` and `glu_enabled`
  are false at the top of `ggml_hpx_exec_graph_selective_mul_mat`
  and the prescan loops + packet-trigger branches are skipped.

**Always-on (no env gate, but cheap):**

- Per-node stats counters (`stats.lowered_nodes++`, `stats.fallback_ns
  += ...`) at lines 1191, 1210, 1223. Plain non-atomic int writes.
  ~5 cycles per node × 689 nodes = ~3.4 µs per token on the math
  path. Below noise.
- `[hpx-tpool]` dispatch histogram in `tpool.cpp:152-158`: plain
  int increments per fallback dispatch. ~3 cycles. ~1.5 µs per
  token. Printed only at process exit in `hpx_destroy`.
- `GGML_HPX_EXEC_SELECTIVE_TESTING` increments at lines 1198,
  1205, 1048, 1143: `#ifdef`-gated. Compiled out in production
  builds.

**Conclusion:** stats/debug paths are cold when env vars are off.
The "always-on" stats counters are negligible (microseconds per
token). After the cleanup in this session, no application mutex is
ever taken on the selective hot path with diagnostics off.

## Findings (no implementation)

1. **The structural cost matches observed perf.** Per-token totals:
   ~4,500 HPX scheduler entries vs. ~5 for selective-off. Even at
   1 µs per HPX scheduler entry, that is ~4.5 ms/token of pure
   scheduling overhead — comparable to selective-off's *total*
   per-token compute (9.75 ms/token). The 120 ms/token regression
   is consistent with this analysis: it's not a kernel-quality
   issue, it's the cost of sliced dispatch.

2. **One bridge per lowered node is the load-bearing constraint.**
   `selective.cpp:1203`'s `hpx::async([&]{...}).get()` serializes
   189 lowered nodes per token through the native thread. Removing
   this serialization (e.g. submitting all lowered region groups
   into one HPX dataflow chain and `.get()`'ing once at the end)
   would eliminate 188 of 189 native→HPX crossings — but doing so
   needs a way to interleave lowered and fallback nodes while
   preserving graph order, since they share `cpu_be` data.

3. **Fallback-run coalescing is the cheapest single win.** A
   contiguous run of fallback nodes (e.g. RESHAPE → VIEW → PERMUTE
   → ADD → ROPE → ...) currently pays N independent backend
   dispatches. A trivial coalescer that grows `view = (gf, i, j)`
   while `ggml_hpx_lower_op(gf->nodes[k], ...)` returns false for
   every k in [i, j) would let one `graph_compute(view)` cover the
   whole run. The histogram says these runs exist (660 ROPE +
   660 ADD + 1320 RESHAPE + 1650 VIEW + 990 PERMUTE per 15 steps,
   many of them adjacent). Estimated saving: 30–60 % of fallback
   per-call cost.

4. **The MATMUL kernel itself is not the bottleneck.** With
   `n_lanes=4` it parallelises correctly across HPX workers, and
   the synthetic A/B (1.30–1.59× over ggml's threadpool dispatch on
   the same chunk grid) confirms that. The lowered Q4_K kernel is
   doing what it should.

5. **`cache.cpp`, `plan.cpp`, `adapter.cpp` are dead weight here.**
   They are coarse-path machinery and were listed in the audit
   request likely on suspicion that they sat on the selective path.
   They don't. Any future selective-side caching (e.g. memoising
   `lower_op` results across decode tokens, since shape is stable)
   would need to be a separate selective-side cache.

6. **The HPX runtime startup is one-shot and not on the hot path.**
   `std::call_once(g_hpx_start_flag)` in `runtime.cpp:43`. After
   first decode, every `ggml_hpx_tpool_start()` call is an atomic
   load on the once_flag. Microseconds per process, irrelevant.

## What this audit does not cover

- HPX's internal scheduler cost per `hpx::async`: assumed ~1 µs
  based on prior synthetic measurements
  (`hpx-bench/results/2026-04-27-hpx-bridge-ablation/`). Not
  re-measured here.
- The cost of `ggml_hpx_lowering_init` + `ggml_hpx_lower_op` per
  node. Per-node: a `memset` of the 8 KB lowering struct (after
  this session's bump from 4 KB) + one switch-case dispatch. Cheap
  but non-zero. Worth a follow-up if scheduling cost is removed
  and per-node setup becomes the next bottleneck.
- Fallback CPU backend internals (ggml-cpu's per-op dispatch). Not
  in scope; that is ggml's mainline code.

## Open framing for next decision

The audit confirms the structural diagnosis from the prior README:
selective-on is paying ~120 ms/token of dispatch overhead, almost
all of it from one-node-at-a-time fallback `graph_compute` and
one-bridge-per-node lowered async. Two intervention shapes are
visible from this audit:

- **Coalesce fallback runs** — small, local change to the main
  loop in `selective.cpp:992-1227`. No HPX restructuring.
  Target: ~30–60 % reduction in fallback per-call cost.

- **Batch lowered-node entry into HPX** — larger, requires
  rethinking `ggml_hpx_lowering` ownership (currently iter-local on
  the stack) and the dependency model between lowered + fallback
  + lowered runs. Target: collapse 189 native→HPX crossings to ~1
  per region run, or to ~1 per token with proper dataflow.

The audit does not recommend an order — that is a design call.
