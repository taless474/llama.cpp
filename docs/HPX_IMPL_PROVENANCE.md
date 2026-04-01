# HPX Implementation Provenance

Goal: evaluate an HPX-backed `ggml` thread pool against the existing pthread-based thread pool in `llama.cpp` on Apple Silicon, and understand **where** HPX helps, where it hurts, and why.

This plan now reflects the current state of the project:
- the HPX backend has been implemented
- correctness and stress tests exist
- tiny reusable-dispatch stalls were investigated
- the main pathological stall mode was traced to the worker sleep path
- the next phase is to measure realistic workloads

---

## 1. Current project state

We started with a narrow question:

**Can we replace the pthread-based orchestration in `ggml/llama.cpp` with HPX and learn something useful about CPU LLM inference execution?**


### What is already done

- HPX-backed thread-pool path implemented in `ggml-cpu-hpx.cpp`
- Contract/correctness tests added for:
  - lifecycle
  - single-thread correctness
  - multi-thread correctness
  - consistency across thread counts
  - multiple graphs on the same pool
  - pause/resume
  - barrier-heavy add chains
  - variable `n_threads` on one pool
  - disposable threadpool path
  - `mul_mat` correctness and stress
- Microbenchmarks added for:
  - dispatch-overhead
  - size ladder
  - reusable pool vs disposable pool
- Additional targeted repro/debug tests added for:
  - fresh-pool startup
  - repeated reuse on same pool
  - warmup effects
  - barrier churn


The current question is:

**How does the HPX path behave, and what do the results actually mean?**

---

## 2. What we learned so far

### 2.1 The first result: HPX was much slower on tiny CPU dispatches

The earliest benchmark result was not that HPX gave a clean speedup or slowdown across realistic inference. Instead, the HPX path looked dramatically worse on tiny repeated CPU dispatches.

That told us an important first fact:

A direct substitution of HPX for pthread orchestration in `ggml` does not
automatically improve performance.

In the smallest cases, dispatch overhead dominates.

---

### 2.2 The bad behavior was not just “HPX is slower”

The tiny reusable benchmark did not merely show slower numbers. It also exposed pathological behavior:
- apparent hangs
- very long stalls on some iterations
- high sensitivity to warmup and runtime state

This pushed the project into a debugging phase.

---

### 2.3 The original “startup race” idea became less convincing

At first, it looked possible that the pool returned before workers were ready, and that the first tiny compute reached a barrier before all HPX workers had entered a stable wait/spin state.

But smaller repros changed that picture:
- fresh-pool one-shot tests passed
- a single immediate first dispatch passed
- the problem reproduced much more strongly in **repeated reuse** of the same graph/plan/pool

So the bug was probably **not** a pure first-dispatch startup issue.

---

### 2.4 The key stall diagnosis

The strongest debugging result so far is this:

**The catastrophic multi-second stall mode was caused by the Phase 2 sleep path.**

Specifically:
- the worker wait path used `std::condition_variable::wait()`
- persistent HPX worker tasks blocking on `std::condition_variable` blocked the
  underlying **OS thread**
- if enough workers blocked simultaneously, HPX had too few free OS threads to
  schedule the remaining worker needed to complete the barrier
- this produced very long stalls that could look like hangs

This explains why disabling that path removed the catastrophic stall mode.

---

### 2.5 What the Phase 2 change means

Replacing the blocking wait path with cooperative yielding removed the pathological multi-second stall behavior in the reuse test.

That gives us a much better current interpretation:

- the original catastrophic behavior was not “HPX is inherently broken”
- it was tied to using an **OS-thread-blocking wait path** inside HPX worker tasks
- once that was removed, the backend still showed significant dispatch overhead,
  but not the same catastrophic stall mode

So now we have to distinguish between:

1. **correctness/robustness**
2. **performance**

The stall diagnosis improved robustness.  
It did **not** prove HPX is competitive on tiny workloads.

---

## 3. Important lesson: tiny benchmark graphs are a microscope, not “all inference”

A lot of the debugging centered on a very small synthetic graph like:

- `N = 32`
- `chain_len = 2`

This graph is intentionally tiny.

It is useful because it magnifies:
- dispatch cost
- barrier cost
- wakeup/scheduling overhead
- reusable-pool state bugs

But it should not be confused with “all inference graphs are tiny.”

### Better interpretation

- **Tiny synthetic graphs** are a stress test for orchestration overhead.
- **Real inference** includes heavier work, especially during prompt processing.
- **Decode/token generation** is often more overhead-sensitive than prefill, but
  it is still not equivalent to `N=32, chain_len=2`.

So the tiny benchmark should remain in the project, but only as:

**a dispatch-overhead microscope**

not as the sole performance verdict.

---

## 4. The updated research questions

The project is now trying to answer three distinct questions.

### Q1. Can the HPX backend run correctly and robustly?
This is answered by the contract and stress suite.

### Q2. What is the dispatch/synchronization overhead of HPX compared with pthread?
This is answered by tiny synthetic microbenchmarks.

### Q3. What happens on realistic inference-related workloads?
This must be answered by kernel-level tests and end-to-end `llama-bench` runs.

---

## 5. Updated benchmark strategy

We should now evaluate the backend at **three levels**.

---

### Level A — Microbenchmarks

Purpose: isolate dispatch, wakeup, and barrier overhead.

Use:
- dispatch-overhead microbenchmark
- size ladder
- reusable vs disposable pool
- barrier churn

Interpretation:
- this level tells us whether HPX orchestration is expensive
- it also acts as a regression detector for pathological reusable-dispatch behavior

But this level should not be treated as the entire inference story.

---

### Level B — Kernel-level ggml work 

Purpose: test whether HPX overhead amortizes when there is real work.

Use:
- `mul_mat` correctness + benchmark-sized variants
- possibly a few additional representative ggml kernels or graph shapes

This is the bridge between tiny synthetic tests and real model inference.

Questions:
- does HPX remain terrible once the compute is bigger?
- or does the overhead shrink to a tolerable fraction?

---

### Level C — End-to-end inference benchmarking

Purpose: evaluate actual llama.cpp workloads.

Use `llama-bench` and report separately:
- **pp** = prompt processing / prefill
- **tg** = token generation / decode

This separation matters.

Why:
- prompt processing is heavier and should amortize scheduler overhead better
- token generation is more overhead-sensitive and may expose runtime costs more sharply

This is the level that answers the real project question.

---

## 6. Updated build matrix

We still want four build variants.

### Pair A — BLAS ON
More realistic macOS baseline where large GEMMs are offloaded to Accelerate.

- `build-pthread-blas/`
- `build-hpx-blas/`

Question:
**Does HPX help or hurt under more realistic inference conditions?**

### Pair B — BLAS OFF
Thread pool is responsible for more of the compute path.

- `build-pthread-noblas/`
- `build-hpx-noblas/`

Question:
**How does HPX behave when the thread pool is responsible for much more of the work?**

Caution:
BLAS OFF changes both threading and compute behavior, so it is not a pure scheduler-only comparison.

---

## 7. Correctness gate before any timing claims

No performance conclusion is trustworthy unless the HPX build passes correctness and
stress checks first.

### Minimum correctness/stability gate

- contract tests pass
- barrier churn passes
- reusable-pool stress passes
- `mul_mat` correctness passes
- no catastrophic multi-second reusable-dispatch stall under the current worker wait path

If this gate fails, timing numbers are not meaningful.

---

## 8. Immediate next steps

### Step 1 — Re-run the core validation suite on the current HPX path
After the worker wait-path change:

- run contract tests
- run stress tests
- run barrier churn
- run reuse tests

Goal:
confirm that the catastrophic stall mode is gone and nothing else regressed.

---

### Step 2 — Reduce tiny benchmark run counts for fast iteration
The current tiny benchmark can take too long to finish because even modest per-dispatch
overhead accumulates over very large run counts.

For development:
- reduce tiny run counts
- add progress prints or shorter quick-run variants

Goal:
get usable post-fix numbers quickly.

---

### Step 3 — Benchmark kernel-level `mul_mat` workloads
Before jumping to persistent worker redesigns, measure whether real compute amortizes
the HPX overhead.

Goal:
see whether HPX remains poor only on dispatch microscopes or also on more realistic kernels.

---

### Step 4 — Run `llama-bench` with `pp` and `tg` reported separately
This is the most important next measurement.

Recommended:
- same model
- same CPU-only settings
- same thread sweep
- report prompt processing separately from token generation

Goal:
recover the big picture.

---



The current best project summary is:

We implemented an HPX-backed `ggml` thread-pool path, added correctness, stress,
and benchmarking infrastructure, discovered that a direct HPX substitution performs
poorly on tiny repeated CPU dispatches, and traced the worst pathological stall mode
to an OS-thread-blocking worker wait path. With that stall mode removed, the next task
is to evaluate HPX on more realistic kernel-level and end-to-end inference workloads.

