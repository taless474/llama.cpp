# Heterogeneous request budgets — `local/baselines/perf_heterogeneous_budgets/`

Snapshot date: 2026-05-06.  
Branch: `hpx-run-level-analyzer`.  
Latest known HEAD: `63bae8a49` (`tools: add HPX-native serving backend and gates`).  
Status: **DESIGN / PRE-CHANGE / PRE-RUN.**

This is the first HPX-favouring workload after three rounds of fixed-shape
results showed no HPX speedup. The matrix and the n_threads sweep both
returned correctness PASS with small, configuration-dependent HPX
overhead. Heterogeneous request budgets test whether mixed generation
lengths expose different queue-drain or wake behaviour in the current
HPX vs std backends — implementations that, to date, are structurally
similar FIFO-over-context-pool.

## 1. Question

Does HPX serving orchestration behave better than std when requests
have mixed generation lengths, so some requests finish much earlier
than others?

More precisely: does the current HPX backend's future-based pool wake
produce different per-request latency, queue-drain shape, or
makespan than the current std backend's condition-variable pool wake,
when n_concurrent > n_contexts and request budgets vary?

## 2. Rationale

All prior fixed-shape experiments used a single `max_tokens` for every
request, so all in-flight requests took roughly the same time. That
shape masks any difference between FIFO wake mechanisms because the
pool stays balanced.

A heterogeneous workload puts variance into the pipeline: a short
request that arrives behind a long one waits much longer than its own
work, and the pool's wake behaviour interacts with that wait directly.
This is the cleanest way to surface any latent difference between the
two backends' pool implementations within the existing harness.

The honest framing matters here:

```text
- Both backends are structurally FIFO-over-context-pool.
- engine_std uses a std::queue + std::condition_variable + one std::thread per ctx.
- engine_hpx uses a deque<hpx::promise> + std::mutex + hpx::future continuations.
- Neither backend currently has priority scheduling, work stealing,
  cancellation, or any-of future composition.
- This experiment tests whether mixed budgets expose different
  queue-drain or wake behavior in the current implementations.
- It is NOT designed to guarantee an HPX win, and the design doc must
  not present an HPX win as the expected outcome.
```

The two interesting outcomes are roughly equally informative:

```text
1. HPX and std behave equivalently under mixed budgets too.
   Reading: the HPX backend is, on this workload class, a thread-pool
   rename, and any future HPX-vs-std experiment must change the HPX
   backend (priority pool, future composition, cancellation) before
   running. This is a clean closeout to the implementation-comparison
   line of work.

2. HPX and std behave differently — typically on short-request
   tail or makespan, since FIFO wake interacts with budget variance
   only on those metrics.
   Reading: the future-based pool's wake path has measurable behaviour
   the CV-based pool doesn't, and that becomes a thing to study.
```

Either reading is publishable.

## 3. Shape

Cell (fixed):

```text
n_contexts=2
n_concurrent=4
n_requests=12   (waiter-pressure: n_concurrent > n_contexts)
```

Common settings (fixed):

```text
binary:        /Users/Ashk/Desktop/HPX/builds/llama-hpx-hpx-on/bin/llama-serving-bench
                (must be rebuilt after the source change in §4)
model:         /Users/Ashk/Desktop/HPX/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
prompt:        "Hello, my name is"
ctx_size:      2048
batch_size:    512
n_threads:     2
seed_base:     1234
```

`n_threads=2` is chosen because the n_threads sweep at C_2x4 placed the
largest HPX-vs-std Δ% there (+7.14% on `total_ms` median). If any
backend divergence under mixed budgets exists, n_threads=2 is the most
likely point of best signal-to-noise.

Budget plan (per-request `max_tokens`, fixed across the protocol):

```text
plan = [64, 8, 16, 8, 64, 16, 8, 64, 8, 16, 8, 32]
```

Composition:

```text
short  (≤ 16 tokens):  request_index ∈ {1, 2, 3, 5, 6, 8, 9, 10}     (count=8)
medium (= 32 tokens):  request_index ∈ {11}                          (count=1)
long   (= 64 tokens):  request_index ∈ {0, 4, 7}                     (count=3)
total tokens / trial:  320
min / max:             8 / 64
```

Fit-check at runtime:

```text
n_prompt_tokens(6) + max(plan)(64) = 70  ≪  ctx_size(2048)   ✓
```

This is a one-off design; no sweep yet. If correctness passes and the
timing read is interesting, a follow-up could vary the plan
(e.g. all-short, all-long, sorted, reverse-sorted).

## 4. Source change required

This experiment requires a small, harness-level source change before
it can run. **It is not yet approved.** This README documents the
change so the design is honest about what the next gate will be.

What already exists in the harness:

```text
serving_bench::request_params has a per-request max_tokens field.
main.cpp::submit_one already constructs a fresh request_params per request.
backend_std.cpp::run_one reads req.max_tokens directly.
backend_hpx.cpp::run_decode reads req.max_tokens directly.
```

So neither backend needs a change. Both already accept per-request
budgets through the existing public surface.

What needs to change (harness-level only):

```text
A. tools/serving-bench/harness.cpp
   - parse_cli: add a --max-tokens-plan flag taking comma-separated
     int32_t values into a new harness_config field (e.g.
     std::vector<int32_t> max_tokens_plan).
   - print_config: print the parsed plan if non-empty.
   - validate plan size equals cfg.n_requests when non-empty.

B. tools/serving-bench/harness.h
   - Add the std::vector<int32_t> max_tokens_plan field on harness_config.

C. tools/serving-bench/main.cpp
   - Update the fit-check at line ~117 to use max(plan) when a plan
     is given; fall back to cfg.max_tokens when the plan is empty.
   - Update submit_one to set
       rp.max_tokens = plan.empty() ? cfg.max_tokens : plan[next_idx];

What does NOT need to change:

```text
backend_std.cpp     no change
backend_hpx.cpp     no change
runtime_hpx.cpp     no change
```

This keeps the std-vs-hpx comparison apples-to-apples by construction:
the only thing the harness does differently is plumb a per-index
budget into request_params; both backends consume that budget through
the same field they already read today.

After the source change is approved and applied, the HPX-on llama
binary at:

```text
/Users/Ashk/Desktop/HPX/builds/llama-hpx-hpx-on/bin/llama-serving-bench
```

must be rebuilt before any trial in this experiment runs.

## 5. Artifacts

Experiment root:

```text
local/baselines/perf_heterogeneous_budgets/
```

Planned files (NOT YET CREATED beyond this README and FACTS.md):

```text
README.md                              (this file)
FACTS.md                               (current-state facts; sibling design doc)
_schedule.json                         (deterministic schedule; pending)
schedule_summary.txt
discovered_canonical_hashes.json       (filled after the first trusted std run)
condition_comparison.txt
experiment_summary.txt
```

Planned helper scripts (NOT YET CREATED):

```text
_make_schedule.py
_run_one_trial.py
_run_experiment.py
_summarize_layer.py
_summarize_condition.py
_summarize_experiment.py
```

Planned raw trial tree:

```text
correctness/
  std/    trial_00..trial_10
  hpx/    trial_00..trial_10
timing/
  std/    trial_00..trial_10
  hpx/    trial_00..trial_10
condition_comparison.txt
```

## 6. Acceptance gates

Per-trial gates (every layer / backend, both correctness and timing):

```text
harness exit code = 0
all 12 expected request rows parsed
aggregate line: n_ok=12 n_cancelled=0 n_error=0
all request statuses = ok
no serving-bench error/failed diagnostics
prompt-fits line present (with max_tokens = max(plan))
```

Per-request gates under the budget plan:

```text
n_tokens_generated[i] == plan[i]
   unless EOS appears early — see exception below.
```

Early-EOS exception:

```text
If n_tokens_generated[i] < plan[i]:
   the request must still report status=ok
   the trial does NOT silently fail
   the layer summarizer records the early-EOS event:
      trial_index, request_index, plan[i], n_tokens_generated[i]
   diagnose before interpreting timing
   on this prompt at greedy/argmax with this seed,
   we expect EOS not to fire below 64 tokens, but the gate must
   not assume so without evidence
```

Cross-backend hash equality (primary correctness gate):

```text
For every request_index i, in every (trial_index t) where the same
seed_base produces the same per-request seed under both backends:
   std_request_hash[t][i] == hpx_request_hash[t][i]
```

Both backends are deterministic argmax over the same vocab, model,
prompt, and per-request seed. Differences here would mean a real
divergence (real bug), not a performance result. This is the gate
that replaces the canonical-hash gate from prior experiments, since
budgets 8, 32, and 64 have no pinned canonical hash today.

Discovered-canonical secondary gate (optional, after first trusted std run):

```text
After a clean std-only run produces deterministic hashes per
(plan[i], seed_base + i) pair, we record:
   discovered_canonical_hashes.json:
     "8,1234":  "0x...."
     "8,1235":  "0x...."
     "16,1236": "0x...."
     ...
A subsequent run can be gated against these as a secondary check.
This is a regression aid, not a primary correctness gate.
```

Correctness layer (trace on, hpx variant):

```text
filtered HPX trace count = 27
   (1 start + 1 ready + 1 stop + 12 acquire + 12 release)
os_threads = 2
pool_size  = 2
acquire ctx-id set = {0, 1}
same-ctx acquire/release pairing
acquire precedes release per request
```

Note: HPX `os_threads` and `pool_size` are HPX runtime carriers and
remain fixed by `os_threads_for(cfg) = max(1, min(n_concurrent, n_contexts)) = 2`,
independent of the budget plan.

Correctness layer (trace on, std variant):

```text
0 matching HPX lifecycle / pool lines
```

Timing layer (trace off):

```text
reduced common gates only
no lifecycle / pool / trace gates (no trace lines emitted)
trial 0 correctness-checked AND excluded from timing aggregation
trials 1..10 measured per backend
```

A backend layer passes if BOTH layers pass for that backend AND the
cross-backend hash equality gate holds for every measured trial.
The experiment passes if both backend-layer gates pass.

## 7. Metrics

Per-request, tagged by budget class:

```text
class    plan[i] in    request_index set in this plan
short    {8, 16}       {1, 2, 3, 5, 6, 8, 9, 10}
medium   {32}          {11}
long     {64}          {0, 4, 7}
```

Per-request metrics (every measured trial):

```text
total_ms[i]                    // queue wait + decode
ttft_ms[i]                     // submit → first generated token
total_minus_ttft_ms[i]         // decode-only
tokens_per_second[i]           // n_tokens_generated[i] / total_ms[i] in seconds
```

Per-class aggregates (over all measured trials × all requests in the class):

```text
short   pool: median, p95, CV    of total_ms, total_minus_ttft_ms, ttft_ms
medium  pool: median, p95, CV
long    pool: median, p95, CV
```

Per-trial trial-level metrics:

```text
makespan_ms          = max over i of total_ms[i] under common t0
                       (i.e., last-request finish time relative to first
                        submit)
queue_drain_ms       = makespan_ms − long_class median total_ms
                       (a positive value indicates short or medium
                        requests are still waiting after the longest
                        long request finishes; negative or zero means
                        the long requests dominated)
process_wall_ms      // outer time.monotonic() around the binary
agg_tokens_per_sec   // 320 generated tokens / wall_s
```

HPX-vs-std deltas (per metric, per class where applicable):

```text
delta_median = hpx.median - std.median
delta_pct    = delta_median / std.median × 100
```

## 8. Headline metrics

Two headline metrics, treated as co-equal:

```text
1. short-class total_ms (pool view): median and p95
   — does HPX shorten time-spent-in-pool for short requests when
     they queue behind longs?

2. trial makespan / queue_drain_ms (per-trial view): median
   — does HPX produce a different wall-clock finish for the trial,
     or a different drain shape after the longest long request
     completes?
```

Neither is privileged over the other. The interpretation rules below
read both jointly so a one-sided result can't be sold as "HPX wins"
or "HPX loses" without the other metric agreeing.

## 9. Interpretation rules (a-priori)

These branches are fixed before the data exists. All branches require
SWEEP_OVERALL_CORRECTNESS = PASS (incl. cross-backend hash equality)
to be evaluated.

```text
hpx_short_better:
   short.total_ms.delta_pct ≤ −1.0%, AND
   makespan.delta_pct       ≤ +0.5%, AND
   long.total_ms.delta_pct  ≤ +1.0%
   Reading: HPX shortens short-request pool time without paying
   meaningful cost on the long class or makespan. The future-based
   wake interacts well with budget variance.

equivalent:
   |short.total_ms.delta_pct| ≤ 1.0%, AND
   |makespan.delta_pct|       ≤ 1.0%, AND
   |long.total_ms.delta_pct|  ≤ 1.0%
   Reading: heterogeneous budgets do NOT expose any backend
   difference. The homogeneous result generalizes. Strong evidence
   that the current HPX backend is, on this workload, a thread-pool
   rename. Closeout-grade conclusion.

hpx_short_worse:
   short.total_ms.delta_pct ≥ +1.0%, AND CV-disjoint
   Reading: the future-based wake path adds latency to short-request
   queue exits when there are concurrent long-class decodes. This is
   real signal worth investigating before any further HPX work.

mixed:
   sign of delta_pct disagrees across short / medium / long, OR
   short and makespan disagree (one favours hpx, one favours std)
   Reading: cannot cleanly attribute. Likely too small a sample
   (K=10 measured) to interpret confidently; do not extrapolate.

incomplete:
   any layer fails or any per-trial gate fails.
   Reading: do not interpret timing.
```

A flat threshold of 1.0% matches prior experiments' "sub-1% delta
inside CV is not interpretable" rule. The makespan threshold is
slightly tighter (0.5%) in the `hpx_short_better` branch because
makespan has lower per-trial CV than per-request pool stats.

## 10. Caveats

```text
The experiment requires a source change in tools/serving-bench/.
   Results from any binary built before that change are invalid.
Per-class n is small (per-trial: 8 short rows + 1 medium + 3 long).
   Across 10 measured trials: 80 short / 10 medium / 30 long
   measured rows. Medium class CV will be unreliable; treat the
   medium row as descriptive only.
The makespan and queue_drain_ms metrics are per-trial (n=10), iid,
   and tend to have lower CV than pool views.
Single machine, single CPU-only build, single TinyLlama Q4_K_M,
   single prompt, single budget plan, K=10 measured trials.
This is not a general llama.cpp benchmark and not a general HPX
   scheduler benchmark.
The current HPX backend exposes no scheduling primitive that the std
   backend lacks. This experiment is NOT designed to guarantee an HPX
   win and should not be presented as one. The most-likely outcome
   based on the prior matrix and sweep is `equivalent`.
EOS-early behaviour is not pre-validated for this prompt at budgets
   {8, 32, 64} on greedy/argmax. If EOS appears in the middle of a
   request, the gate records it; the layer summarizer must surface
   it; a real interpretation requires per-budget reasoning before
   trusting the timing.
For budgets other than 16, no canonical hash is pinned. The primary
   correctness gate is cross-backend hash equality per request_index,
   not absolute hash match.
```

## 11. Result

```text
NOT YET RUN
```

To be filled in after the run, with:

```text
correctness gate state per (layer, backend), incl. hash-equality count
short.total_ms.delta_pct (median, p95)
makespan.delta_pct (median)
long.total_ms.delta_pct
medium.total_ms.delta_pct (descriptive)
queue_drain_ms median per backend
discovered canonical hashes per (budget, seed)
branch label: hpx_short_better / equivalent / hpx_short_worse / mixed
```

## 12. Next action

```text
1. Approve this design and FACTS.md.
2. Approve the source change described in §4 (one-file behaviour
   change in tools/serving-bench/harness.cpp + harness.h + main.cpp).
3. Apply the source change.
4. Rebuild the HPX-on llama binary.
5. Approve helper scripts (_make_schedule.py, _run_one_trial.py,
   _run_experiment.py, _summarize_*.py).
6. Generate _schedule.json.
7. Run correctness layer for both backends; stop on first gate
   failure. Verify cross-backend hash equality per request_index
   before allowing the timing layer to run.
8. Run timing layer for both backends.
9. Run summarizers.
10. Fill in §11 Result.
11. Decide whether to add an index row in local/baselines/README.md.
```

## 13. Things not to do

```text
Do not modify any source under tools/serving-bench/ before §4 is
   approved.
Do not rebuild HPX or the llama HPX-on binary before §4 is applied.
Do not modify any prior baseline directory.
Do not write helper scripts before this design and the source change
   are approved.
Do not run the experiment before helpers and the rebuild are approved.
Do not interpret timing if any per-trial correctness gate failed.
Do not interpret timing if any cross-backend hash equality fails;
   that would be a real divergence, not a performance result.
Do not silently treat an early-EOS event as a failure or as ok
   without recording it.
Do not include trial 0 in timing aggregation.
Do not claim HPX is faster (or slower) outside the noise-aware
   language defined in §9.
Do not extrapolate to other prompts, generation lengths, models,
   quantizations, cells, or machines.
Do not write outputs to /tmp.
Do not commit until the run completes and any index decisions are
   resolved.
```
