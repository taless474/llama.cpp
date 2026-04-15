## Claude Guidance for HPX / ggml Integration

### Read in this order

1. README_HPX.md
2. docs/HPX_EXECUTOR_CONTRACT.md
3. docs/HPX_PROVENANCE.md (optional, historical only, only read if needed)

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
