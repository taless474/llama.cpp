# FACTS — heterogeneous request budgets protocol setup

This file records current-state facts before any source change or run
of the heterogeneous-budgets protocol. It is not a benchmark report
and not a design authority by itself.

## 1. HPX install state

HPX is built and installed:

```text
HPX install dir:  /Users/Ashk/Desktop/HPX/hpx-install/
HPX cmake config: /Users/Ashk/Desktop/HPX/hpx-install/lib/cmake/HPX/HPXConfig.cmake
HPX libs:         /Users/Ashk/Desktop/HPX/hpx-install/lib/libhpx.dylib
                  /Users/Ashk/Desktop/HPX/hpx-install/lib/libhpx_core.dylib
```

No HPX rebuild is required for this protocol. HPX itself is unchanged.

## 2. HPX-on llama binary

```text
/Users/Ashk/Desktop/HPX/builds/llama-hpx-hpx-on/bin/llama-serving-bench
```

This is the same binary used by the matrix and the n_threads sweep.
**It must be rebuilt after the source change in README.md §4 is
applied.** Until that rebuild, the harness has no `--max-tokens-plan`
flag and cannot run this protocol.

## 3. Prior PASS evidence — context only, not reused as trials

This protocol is independent of prior runs. They are listed for
context only; their artifacts are not read by the helpers in this
experiment.

```text
local/baselines/comparison_hpx_waiters/
  shape:  (n_contexts=2, n_concurrent=4, n_requests=12, n_threads=4, max_tokens=16 uniform)
  result: OVERALL: PASS
  scope:  capacity correctness / no-starvation under n_concurrent > n_contexts

local/baselines/perf_hpx_vs_std_matrix/
  shape:  matrix of 4 cells; n_threads=4 fixed; max_tokens=16 uniform
  result: MATRIX_OVERALL_CORRECTNESS: PASS
          16/16 layer summaries PASS
          4/4 condition summaries PASS
          496/496 trials at exit_code 0
  C_2x4 (relevant cell): HPX +7.68% on total_ms median (descriptive)
  scope:  cross-cell HPX-vs-std at uniform 16-token budgets

local/baselines/perf_thread_sweep/
  shape:  C_2x4 fixed; n_threads ∈ {1, 2, 4}; max_tokens=16 uniform
  result: SWEEP_OVERALL_CORRECTNESS: PASS
          12/12 layer summaries PASS
          3/3 thread-setting summaries PASS
          132/132 trials at exit_code 0
  trend on total_ms median: mixed
          delta_pct = +1.32% / +7.14% / +3.73% at n_threads=1/2/4
          peak at n_threads=2
  scope:  n_threads sensitivity at uniform 16-token budgets, fixed cell
```

These three results establish:

```text
- C_2x4 is structurally correct under uniform budgets.
- HPX-vs-std overhead at uniform budgets is small (1–7%) and
  configuration-dependent, with peak at n_threads=2 (+7.14%).
- No fixed-shape configuration shows an HPX advantage.
```

This experiment is the first to vary `max_tokens` per request, to
test whether budget heterogeneity exposes any difference between the
backends' FIFO-pool implementations.

## 4. Canonical hash situation

Only one canonical hash is currently pinned in this project:

```text
0x833045f1e2ebf49f
```

That hash is the canonical fingerprint for:

```text
model:      tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
prompt:     "Hello, my name is"
max_tokens: 16
decode:     greedy / argmax
seed_base:  1234
CPU:        yes
```

Same shape used by the matrix, the sweep, and the prior comparison
slices, and pinned in `docs/hpx/serving_bench_acceptance.md` for
`max_tokens=16`.

This experiment uses budgets 8, 16, 32, 64 across 12 requests with
12 different seeds (`seed_base + request_index`). For request_index=0
through 11 with budgets other than 16, **no canonical hash is pinned
today**.

How correctness gates handle this:

```text
Primary gate (cross-backend hash equality):
   For every (trial_index t, request_index i):
      std_request_hash[t][i] == hpx_request_hash[t][i]
   Both backends are deterministic argmax over the same vocab, model,
   prompt, and per-request seed (seed_base + i). They MUST produce the
   same generated tokens, and therefore the same hash, given the same
   per-request budget plan[i]. Any mismatch is a real backend
   divergence — a bug, not a perf result.

Discovered-canonical secondary gate (regression-only):
   After a clean std-only run, the layer summarizer records each
   (plan[i], seed_base + i) → discovered_hash mapping into:
      discovered_canonical_hashes.json
   Subsequent runs can compare against this file as a regression aid.
   This is NOT a primary correctness gate; the primary gate stays
   cross-backend hash equality.
```

The 16-token canonical hash `0x833045f1e2ebf49f` is preserved as a
sanity check for the request_index slots in the plan that happen to
use budget 16 (request_index ∈ {2, 5, 9}). If those slots reproduce
the canonical hash under the same per-request seed (seed_base + i for
i in {2, 5, 9}), that is informative; if they do not, that does NOT
mean a failure — different per-request seeds produce different valid
hashes, even at the same budget.

## 5. Workload shape

Cell (fixed across the protocol):

```text
n_contexts=2
n_concurrent=4
n_requests=12   (waiter-pressure: n_concurrent > n_contexts)
```

Common shape, pinned across both backends:

```text
binary:        /Users/Ashk/Desktop/HPX/builds/llama-hpx-hpx-on/bin/llama-serving-bench
                (must be rebuilt after the source change in README.md §4)
model:         /Users/Ashk/Desktop/HPX/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
prompt:        "Hello, my name is"
ctx_size:      2048
batch_size:    512
n_threads:     2
seed_base:     1234
```

Per-request budget plan (fixed):

```text
plan = [64, 8, 16, 8, 64, 16, 8, 64, 8, 16, 8, 32]
size:    12   (matches n_requests)
total:  320 generated tokens per trial
min:      8
max:     64
classes: short {8, 16} ×8 ; medium {32} ×1 ; long {64} ×3
```

Per-request seed (fixed by the existing harness logic):

```text
seed[i] = seed_base + i   (i.e. 1234..1245)
```

Per-request fit-check at runtime:

```text
n_prompt_tokens(6) + max(plan)(64) = 70  ≪  ctx_size(2048)   ✓
```

## 6. Protocol parameters

```text
trial granularity:                    process-level (one binary invocation per trial)
trials per (layer × backend):         K = 11
trials retained for timing per bk:    10 (trial 0 dropped)
schedule seed:                        1234
schedule scope:                       per layer; fresh schedule, same seed
schedule recording:                   _schedule.json written before any trial runs
layer ordering:                       correctness layer first; timing layer only if correctness PASS
backend ordering within layer:        randomized, std/hpx interleaved (paired)
total invocations:                    1 cell × 1 plan × 2 backends × 2 layers × 11 trials = 44
estimated wall (warm):                ~5–8 minutes on an idle machine
                                       (320 tokens/trial; ~10 ms/tok ≈ 3.2 s decode-floor; pool overlap ≈ 1.6 s)
```

## 7. Two-layer expectations

### Correctness layer (trace ON)

```text
env:             LLAMA_SERVING_BENCH_HPX_TRACE=1
gates per trial: full common gates + lifecycle/pool/trace gates (hpx variant)
                 inverted-trace gate (std variant)
                 per-request budget gate (n_tokens_generated[i] == plan[i],
                                          unless EOS appears early)
                 cross-backend hash equality (deferred to layer summarizer)
purpose:         prove that the heterogeneous-budget workload is
                 structurally correct under both backends
output:          per-trial artifacts + condition_summary.txt + layer_pass.txt
timing use:      none — this layer's per-request timings are NOT pooled
                 into the timing summary
```

Expected per-trial filtered HPX trace count, hpx variant only:

```text
1 start + 1 ready + 1 stop + 12 acquire + 12 release = 27
   (independent of the budget plan; HPX runtime carriers are not
    affected by budget variance)
```

Expected ctx-id usage, hpx variant only:

```text
acquire ctx-id set == {0, 1}
```

Expected `os_threads` and `pool_size`, hpx variant only:

```text
os_threads = 2
pool_size  = 2
   (matches main.cpp's os_threads_for(cfg) = max(1, min(n_concurrent, n_contexts)))
```

Expected std variant trace, all layers:

```text
0 matching HPX lifecycle / pool lines
```

### Timing layer (trace OFF)

```text
env:             LLAMA_SERVING_BENCH_HPX_TRACE unset
gates per trial: reduced common gates (status, n_tokens against plan,
                 aggregate counts, prompt-fits line, no error/failed);
                 NO lifecycle / pool / trace gates (no trace lines emitted)
purpose:         measure per-request and per-trial timing free of HPX-only
                 fprintf trace overhead
output:          per-trial artifacts + per_trial_summary.csv + all_requests.csv
                 + timing_summary.txt + condition_comparison.txt
trial 0:         correctness-checked AND excluded from timing aggregation
trials measured: 10 (trials 1..10) per backend
```

## 8. Source-change requirement (precondition)

This protocol cannot run on the current HPX-on binary. A small,
harness-level source change is required before the binary is rebuilt.

What already exists in the harness:

```text
serving_bench::request_params has a per-request max_tokens field.
main.cpp::submit_one constructs a fresh request_params per request.
backend_std.cpp::run_one reads req.max_tokens directly.
backend_hpx.cpp::run_decode reads req.max_tokens directly.
```

What needs to change (harness-level only):

```text
A. tools/serving-bench/harness.h
   - add std::vector<int32_t> max_tokens_plan to harness_config.

B. tools/serving-bench/harness.cpp
   - parse_cli: add --max-tokens-plan CSV taking comma-separated int32_t
     values into harness_config::max_tokens_plan.
   - parse_cli: validate plan size equals n_requests when non-empty
     (after both flags have been parsed; report error and exit 1 on mismatch).
   - print_config: print the parsed plan if non-empty.

C. tools/serving-bench/main.cpp
   - update the fit-check at line ~117: when a plan is given, check
     n_prompt_tokens + max(plan) > cfg.ctx_size; otherwise keep the
     existing cfg.max_tokens check.
   - update submit_one (line ~145): set
       rp.max_tokens = plan.empty() ? cfg.max_tokens : plan[next_idx];
```

What does NOT need to change:

```text
backend_std.cpp     no change — already reads req.max_tokens
backend_hpx.cpp     no change — already reads req.max_tokens
runtime_hpx.cpp     no change — runtime carrier count is independent of budgets
```

This is the smallest source change that enables the experiment. It
preserves the apples-to-apples shape of the std-vs-hpx comparison
(both backends consume the per-request budget through the same
`request_params::max_tokens` field they already read today).

## 9. Scope of this protocol

This protocol measures:

```text
per-request total_ms, ttft_ms, total_minus_ttft_ms tagged by budget
   class (short ≤ 16, medium = 32, long = 64), paired std-vs-hpx,
   under n_contexts=2 / n_concurrent=4 / n_threads=2
trial-level makespan_ms and queue_drain_ms
process_wall_ms per trial
agg_tokens_per_sec per trial
within-class std-vs-hpx delta (median, p95, stdev, CV)
HPX correctness (lifecycle + pool + RAII pairing) at K=11 trials per
   (layer, backend)
cross-backend hash equality per (trial_index, request_index)
```

This protocol does NOT measure:

```text
priority scheduling (the HPX backend has none today)
work-stealing or any-of future composition
mid-decode cancellation or backpressure
heterogeneous prompts (the prompt is fixed; only budgets vary)
behaviour at any cell other than C_2x4
behaviour at any n_threads other than 2
behaviour at any n_concurrent other than 4
general llama.cpp performance
HPX scheduler quality in general
budget-sensitivity sweeps (this protocol uses one fixed plan)
behaviour at K > 10 measured trials; sub-1% deltas inside the CV
   band remain not interpretable as "different"
```
