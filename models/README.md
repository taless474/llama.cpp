# Models

The `ggml-vocab-*.gguf` files and `templates/` in this directory are test fixtures
tracked by the repo. Everything else is gitignored.

Benchmark models are NOT tracked. Download them locally:

```bash
# install hf CLI (one-time)
brew install pipx && pipx install huggingface_hub

# download Llama 3.1 8B Instruct Q4_K_M (~4.7 GB)
hf download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \
  Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  --local-dir models/llama3.1-8b
```

See `HPX_BENCHMARK_PLAN.md` in the repo root for the full benchmarking workflow.
