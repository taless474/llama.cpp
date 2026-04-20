## Claude Guidance for HPX / ggml Integration

### Read in this order

1. README_HPX.md
2. docs/HPX_EXECUTOR_CONTRACT.md
3. docs/HPX_PROVENANCE.md (optional, historical only, only read if needed and start from the last section)

### Core principles

- Prefer current architecture over historical designs
- Follow explicit contracts, not inferred behavior
- Keep changes minimal and localized
- Preserve separation of concerns

### Critical invariants

- Two execution substrates:
  - pthread → small work / decode
  - HPX → large work / prefill

- Selection is based on:
  cplan->work_size

- Scheduler logic ONLY in:
  ggml-hpx-adapter.cpp

- Plans must be structural only

- BLAS is opaque and not decomposed

### Forbidden

- Do not merge pthread and HPX into one executor
- Do not route decode work to HPX casually
- Do not leak scheduler outside adapter
- Do not store runtime pointers in plans
- Do not use provenance as design truth

### Preferred direction

- Move toward run_range(...)
- Use fine-grained region DAG
- Reduce thread-centric execution

### One-line model

Small work → pthread
Large work → HPX
Future → region DAG

### Saving results

Never write benchmark output to `/tmp`.  All results stay inside the repo.

- `hpx-bench/results/<date>-<slug>/` — benchmark CSVs and logs
- `local/results/` — experiment summaries, environment notes, anything not suitable for git

Each run gets its own directory.  Minimum contents:
- `bench_<name>.csv` — raw CSV (stdout)
- `bench.log` — stderr (legend, done marker, any errors)
- `README.md` — short markdown summary: what was measured, key numbers, findings, open questions

Redirect output at the invocation step, not with a post-run `cp`.
Include the git commit hash and binary path in the summary so results are reproducible.
