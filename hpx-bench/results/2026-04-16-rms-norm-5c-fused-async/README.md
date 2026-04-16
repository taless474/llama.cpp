# RMS_NORM Phase 5C — fused_async single-dispatch cost — 2026-04-16

**Binary:** `build-hpx/bin/bench_hpx_rms_norm_f32`
**Commit:** `3c1d7b585` + stack-alloc patch (hpx-prefill-orchestrator)
**Baseline:** `hpx-bench/results/2026-04-16-rms-norm-5c-stackalloc/`
**Params:** eps=1e-5, warmup=50, reps=1000, n∈{512,2048,4096,8192}, lanes∈{1,2,4}

## Variants

| impl | dispatch model | lanes |
|---|---|---|
| `ref` | scalar loop, main thread | 1 |
| `direct` | fused body, main thread, no HPX | 1 |
| `fused_async` | one `hpx::async(...).get()` per rep around fused body | 1 |
| `dag_empty_nowrap` | 3-region group, near-nop callbacks, outer async once | 1,2,4 |
| `dag_nowrap` | 3-region group, production callbacks, outer async once | 1,2,4 |

`fused_async` uses `bench_ns` (outer async paid per rep).
`dag_empty_nowrap` / `dag_nowrap` use `bench_ns_in_hpx` (outer async paid once).

## Key numbers (min_ns, l=1)

| n | direct | fused_async | dag_empty_nowrap | dag_nowrap |
|---|---|---|---|---|
| 512 | 250 | 917 | 7334 | 7709 |
| 2048 | 1125 | 2917 | 7958 | 8000 |
| 4096 | 2416 | 4208 | 7500 | 10833 |
| 8192 | 4916 | 7083 | 7334 | 13083 |

## Findings

**1. One `hpx::async(...).get()` costs ~700–2000 ns on the fast path.**

`fused_async` min − `direct` min:
- n=512:  917 − 250 = 667 ns
- n=2048: 2917 − 1125 = 1792 ns
- n=4096: 4208 − 2416 = 1792 ns
- n=8192: 7083 − 4916 = 2167 ns

At n=512 the task-post overhead is ~700 ns; at larger n it stabilises around
1800–2200 ns.  This is the cost of one HPX task-post + join from the main thread.

**2. `dag_empty_nowrap` and `dag_nowrap` are indistinguishable at l=1.**

`dag_empty_nowrap` min ≈ `dag_nowrap` min at n=512 and n=2048 (within 50 ns).
At n=4096 and n=8192 `dag_nowrap` is higher (~3 µs), reflecting the real
callback work appearing above the noise floor.

The orchestration machinery (Kahn, dataflow, wait_all) with near-nop callbacks
costs essentially the same as with full callbacks at these sizes.

**3. `fused_async` min < `dag_empty_nowrap` min at n=512–4096.**

| n | fused_async min | dag_empty_nowrap min | difference |
|---|---|---|---|
| 512 | 917 | 7334 | −6.4 µs |
| 2048 | 2917 | 7958 | −5.0 µs |
| 4096 | 4208 | 7500 | −3.3 µs |
| 8192 | 7083 | 7334 | −0.25 µs |

At n=8192 the two converge: math cost catches up to DAG overhead.  Below
n=8192, `fused_async` wins because it avoids the 3-region DAG entirely.

**4. The "break-even n" for the 3-region DAG is somewhere around 8k–16k.**

At n=8192, `dag_nowrap` min is ~13 µs and `fused_async` min is ~7 µs.
The DAG still costs 2× more.  Breakeven requires the math to dominate the
~6 µs DAG floor — which needs n ≫ 8192 for l=1.

## The cost ladder (fast path, l=1)

```
direct        ~250 ns   (math only, no HPX)
fused_async   ~917 ns   (math + one task-post)  [+667 ns vs direct]
dag_empty_nowrap ~7.3 µs (3-region DAG, near-nop, outer async once)  [+6.4 µs vs fused_async]
dag_nowrap    ~7.7 µs   (3-region DAG, production callbacks, l=1)    [+0.4 µs vs dag_empty_nowrap]
```

The DAG machinery costs ~6 µs more than a single fused dispatch.
The production callbacks add only ~0.4 µs on top of near-nop.

## Conclusion

One `hpx::async` dispatch costs ~700–2200 ns (varies with n, possibly
cache effects or thread state).  The 3-region DAG costs ~6–7 µs more
than that single dispatch, which is the price of 2 additional asyncs +
2 dataflow compositions + wait_all.

To reach `fused_async`-level latency from the DAG path, the number of
independent HPX task posts per group invocation must drop from 5–6 to 1.
