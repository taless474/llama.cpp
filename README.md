# llama-hpx

This repository is an experimental fork of `llama.cpp` used to study where HPX can help LLM inference and serving workloads.

The project started with the question:

```text
Can HPX improve parts of llama.cpp execution or serving by adding better runtime orchestration?
```

Across the branches, the work moved through three levels:

```text
1. thread-pool/runtime experiments around llama.cpp
2. HPX near ggml graph execution and prefill/decode orchestration
3. HPX outside ggml, at the serving request-orchestration layer
```

The current evidence suggests that HPX is useful for building and validating runtime ideas, but simply inserting HPX as a replacement scheduler around the existing `llama.cpp` execution path does not automatically create a speedup. The most promising future direction is adding a capability the baseline does not have.

## Branch map

### `hpx-threadpool-experiment`

First exploration branch.

URL:

```text
https://github.com/taless474/llama.cpp/tree/hpx-threadpool-experiment
```

This branch explored early HPX/thread-pool ideas around llama.cpp execution. It was the first attempt to understand where external runtime orchestration could fit and what kinds of overheads appear when llama.cpp work is wrapped by another scheduling layer.

Main lesson:

```text
Wrapping existing llama.cpp execution with another runtime is not enough by itself. The design has to expose real parallelism, overlap, cancellation, priority, or another useful scheduling capability.
```

### `hpx-prefill-orchestrator`

Second exploration branch.

This branch moved closer to ggml graph execution. It explored HPX prefill orchestration, selective execution, packetized fine-region execution, and run-level graph ideas.

Topics explored included:

```text
ggml graph structure
prefill vs decode behavior
selective lowering
fine-region DAGs
packetized repeated subgraphs
run-level execution planning
interaction with ggml scheduler and CPU backend paths
```

Main lesson:

```text
HPX could execute some lowered or packetized regions correctly, but the tested CPU-only designs did not beat the existing llama.cpp / ggml scheduler. The overhead and granularity were not favorable when HPX was inserted inside or near ggml graph execution.
```

### `hpx-run-level-analyzer`

Current evidence branch.

This branch moved HPX out of ggml graph execution and evaluated it at the serving orchestration layer.

The serving design is:

```text
shared llama_model
multiple prewarmed llama_context objects
one leased context per active request
normal llama_decode inside the leased context
std backend vs hpx backend
token-hash correctness checks
trace-based lifecycle and context-pool validation
```

The main comparison is:

```text
llama-serving-bench --backend std
vs.
llama-serving-bench --backend hpx
```

Both backends use the same model, prompt, request shape, context-pool shape, greedy decoding, and `llama_decode` path.

Main lesson:

```text
The HPX serving backend is correct and robust, but the current HPX design does not outperform the simpler std backend on the tested CPU-only TinyLlama serving workloads.
```

The reason is structural: both backends are FIFO context-pool orchestrators around the same opaque `llama_decode` path. The HPX backend changes the orchestration mechanism, but it does not yet add a new HPX-native capability such as cancellation, priority scheduling, richer future composition, work stealing, or distributed execution.

## What this repository is studying

This repository is not trying to replace `llama.cpp` kernels.

The main questions are:

```text
Where can HPX sit around llama.cpp without breaking correctness?
What granularity is too fine for HPX to help?
When does HPX orchestration become overhead instead of useful scheduling?
What evidence is needed before claiming a runtime-level improvement?
What HPX-native features could create a real functional advantage?
```

The project has deliberately kept correctness checks central:

```text
same prompt
same model
same request shape
same decoding path
same token hashes
trace-based lifecycle checks
repeatable benchmark protocols
trial-0 excluded from timing aggregation
```


## Repository guide

### HPX design and notes

HPX-related design notes and reports live under:

```text
docs/hpx/
```

Useful documents include design notes, benchmark protocols, result summaries, executor contracts, provenance notes, and closeout notes.

Start here when you want the reasoning and conclusions.

### Benchmark evidence package

Benchmark evidence lives under:

```text
hpx-bench/
```

This directory contains packaged experiments, scripts, schedules, summaries, and reproducible evidence for the serving-bench work.

Important subdirectory:

```text
hpx-bench/experiments/
```

Examples of packaged serving experiments include:

```text
08_perf_hpx_vs_std_matrix/
09_perf_thread_sweep/
10_perf_heterogeneous_budgets/
11_perf_deep_queue_short_requests/
```

Each experiment directory is intended to contain:

```text
readme.md        experiment purpose, protocol, and compact result
facts.md         stable design facts and validation rules
results.md       post-run interpretation, when available
summaries/       shareable summarized evidence
runs/            raw local trial artifacts, ignored by git
scripts          schedule generation, trial runner, and summarizers
```

### Serving benchmark source

The serving benchmark source lives under:

```text
tools/serving-bench/
```

This is where the std and hpx serving backends, request harness, and benchmark CLI live.

The key benchmark comparison remains:

```text
llama-serving-bench --backend std
llama-serving-bench --backend hpx
```

