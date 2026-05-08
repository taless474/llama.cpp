# Multi-sequence `llama_batch` gate — design note

This is a **design note**, not an implementation. It defines the
smallest real `llama.cpp` technical gate that must pass before any HPX
integration starts.

No HPX code, no `tools/server/` change, no `tools/serving-bench/`
change, no full server, no real performance claim is made here.

## What this gate must prove

Concretely, with a single tiny binary that links `libllama` and
`libcommon`:

- One `llama_model`, loaded once.
- One `llama_context`, with `n_seq_max` ≥ the number of concurrent
  sequences in the smoke run.
- Multiple `llama_seq_id`s active at the same time (call them
  `seq=0..N-1`).
- One shared `llama_batch`, reused across iterations
  (`common_batch_clear` at the start of each iteration).
- A fixed prompt that is **the same** for every sequence in the
  initial smoke run.
- Greedy argmax sampling (no `common_sampler` chain), bypassing
  temperature / top-k / top-p / penalties.
- Multiple sequences decoding **in the same iteration** — one decode
  row per active sequence per step.
- Logits read back via `llama_get_logits_ith(ctx, i)` where `i` is the
  row index that was set as the slot's `i_batch` at row-append time.
- Positions tracked per-`seq_id` correctly (prefill rows at
  `pos=0..P-1`, decode rows at `pos=P+k`).
- `llama_memory_seq_rm(mem, seq_id, -1, -1)` clears KV for a finished
  `seq_id`, and the cleared `seq_id` can be reused or simply left idle
  without affecting other live sequences.
- Deterministic per-sequence generated-token hashes across re-runs of
  the same binary on the same model.

## Q1. Where should this gate live?

**Recommendation: a new small tool under `tools/`, not inside
`tools/serving-bench/`.**

Suggested path:

```
tools/multiseq-batch-gate/
    CMakeLists.txt
    multiseq-batch-gate.cpp
    README.md
```

(Final name to be decided when implementation starts.
`multiseq-batch-gate` is descriptive and not collision-prone.)

Reasons:

- `tools/serving-bench/` is the FIFO-pool benchmark and its closeout
  evidence is committed. Mixing a multi-`seq_id` gate into it would
  conflate two different scope levels.
- The gate is a fixed-prompt, fixed-budget, no-HTTP, no-sampler
  binary. It has no shared code path with the existing serving-bench.
- `tools/server/` is upstream infrastructure and is out of scope.
- An entry under `tools/` matches existing layout (one tool = one
  directory). It cleanly follows the project convention used by
  `serving-bench`, `main`, `bench`, etc.
- The binary's CMake target should not be added to the existing
  `serving-bench` build target — it should be its own target so it
  can be built and run independently.

## Q2. What existing `llama.cpp` APIs are needed?

Minimum surface area:

| Concern | API (header) |
|---|---|
| Model load | `llama_model_load_from_file` (`llama.h`) |
| Context create | `llama_init_from_model` (`llama.h`) |
| Tokenize | `llama_tokenize` (`llama.h`) or `common_tokenize` (`common/common.h`) |
| Vocab size | `llama_vocab_n_tokens` / `llama_n_vocab` (`llama.h`) |
| Context query | `llama_n_seq_max(ctx)`, `llama_n_batch(ctx)`, `llama_n_ctx(ctx)` (`llama.h`) |
| Batch alloc | `llama_batch_init` (`llama.h`) |
| Batch helpers | `common_batch_clear`, `common_batch_add` (`common/common.h`) |
| Decode | `llama_decode(ctx, batch_view)` (`llama.h`) |
| Logits readback | `llama_get_logits_ith(ctx, row_idx)` (`llama.h`) |
| Memory handle | `llama_get_memory(ctx)` (`llama.h`) |
| KV per-seq clear | `llama_memory_seq_rm(mem, seq_id, p0, p1)` (`llama.h`) |
| KV per-seq pos | `llama_memory_seq_pos_min/max` (advisory; `llama.h`) |
| Cleanup | `llama_batch_free`, `llama_free`, `llama_model_free` (`llama.h`) |

Notes:

- The full sampler chain (`common_sampler_*`) is **not** needed.
  Greedy argmax over `llama_get_logits_ith` is enough and removes a
  large source of behavioral nondeterminism.
- No `llama_state_seq_*`, no checkpointing, no SWA-specific paths.
- No `llama_memory_seq_cp` / `seq_add` / `seq_div` — those are for
  context shift, prefix reuse, and `n_cmpl > 1` fan-out, all out of
  scope.
- Hashing the generated tokens can use any stable per-process hash
  function; for example, FNV-1a over `little-endian u32` token ids.
  The choice is an implementation detail. The same canonical token
  hash mechanism that `serving-bench` already uses is fine.

## Q3. What is the exact minimal sequence of operations?

In order:

1. **Init.** Load the `llama_model` once.  Build one `llama_context`
   with `n_seq_max ≥ N`, where `N` is the number of concurrent
   sequences (initially 3). Verify `llama_n_seq_max(ctx) >= N`,
   `llama_n_ctx(ctx) >= P + max(decode_budgets)`, and
   `llama_n_batch(ctx) >= N * P` (so the all-at-once prefill fits).
2. **Tokenize the prompt once.** The prompt string is fixed for the
   smoke run. Call `common_tokenize(model, prompt, /*add_special=*/true,
   /*parse_special=*/false)` once and reuse the token vector for all
   sequences.
3. **Assign `seq_id`s.** Use `seq=0..N-1`. Maintain a small per-seq
   table holding: `decode_budget`, `n_decoded`, `pos_next`,
   `i_batch`, `done`, and the generated-token vector.
4. **Allocate the shared `llama_batch`.** Allocate it once via
   `llama_batch_init(n_batch_max, /*embd=*/0, /*n_seq_max=*/N)` where
   `n_batch_max = max(N * P, n_batch_query)`.
5. **Prefill all sequences in one iteration (smoke case).**
   1. `common_batch_clear(batch)`.
   2. For `seq` in `0..N-1` and for each prompt token at position `p`:
      append a row with `common_batch_add(batch, prompt_tokens[p], p,
      { seq }, /*logits=*/false)`. Set `logits=true` only on the last
      prompt row of each sequence.
   3. Record per-seq `i_batch = batch.n_tokens - 1` immediately after
      its last prompt row is appended.
   4. Call `llama_decode(ctx, batch)`. Verify return code == 0.
   5. For each `seq`, read `llama_get_logits_ith(ctx, i_batch[seq])`,
      take argmax over the vocab to get the first generated token
      `t0[seq]`. Append to that seq's generated-token vector. Bump
      `n_decoded[seq]` to 1, set `pos_next[seq] = P + 1`.
6. **Per-step decode iteration loop.** While at least one sequence is
   not yet at its budget:
   1. `common_batch_clear(batch)`.
   2. For each non-`done` `seq`: append exactly one decode row
      `common_batch_add(batch, last_token[seq], pos_next[seq], { seq },
      /*logits=*/true)`. Record this row's index as `i_batch[seq] =
      batch.n_tokens - 1`. Bump `pos_next[seq]`.
   3. Call `llama_decode(ctx, batch)`. Verify return code == 0.
   4. For each non-`done` `seq`: argmax over
      `llama_get_logits_ith(ctx, i_batch[seq])` → next token. Append to
      that seq's generated-token vector. Bump `n_decoded[seq]`.
      `last_token[seq] = next`.
   5. For any `seq` with `n_decoded[seq] >= decode_budget[seq]`: mark
      it `done` (do not append rows for it next iteration).
7. **Per-seq cleanup on completion.** When a `seq` becomes `done`,
   call `llama_memory_seq_rm(mem, seq, -1, -1)`. Verify
   `llama_memory_seq_pos_min(mem, seq) == -1` afterwards. The slot is
   then conceptually free (the gate does not re-use it; that is
   slot-pool territory and out of scope here).
8. **Termination.** When all `N` sequences are `done`:
   - Compute the per-seq generated-token hash.
   - Print one line per `seq`:
     `seq=<id> budget=<D> n_decoded=<D> hash=0x<...>`.
   - Free batch, context, model. Exit 0.

Step 5 (one-shot multi-seq prefill) is the smoke shortcut. The
"more correct" prefill would chunk by `n_batch`; the smoke case
deliberately stays inside `n_batch` so that the only multi-iteration
work is the decode loop. This isolates the multi-`seq_id` decode
behavior from prompt-chunking complexity.

## Q4. What correctness checks prove the gate?

The gate must verify all of the following before exiting 0:

1. **All sequences finish at their requested budget.** For every
   `seq`, `n_decoded[seq] == decode_budget[seq]`. The gate's exit
   code is non-zero if any `seq` ends short.
2. **Per-seq deterministic token hash.** Re-running the same binary
   with the same model and same seed (no seed needed for greedy
   argmax) produces byte-identical hashes for every `seq`. The gate
   should print the hashes; an external script (or a second invocation
   inside the same binary using `--repeat 2`) compares them.
3. **No `seq_id` cross-talk.** With identical prompts, the prefill
   logits at the last prompt row should be identical across `seq_id`s
   (modulo positional encoding effects, which are zero since `pos`
   is identical for the same position across sequences). For the
   decode loop, with the same prompt and greedy argmax, **all**
   sequences produce the same `t0`. After `t0`, each seq's trajectory
   is fully determined by its own decode-token sequence; if budgets
   differ, sequences with longer budgets simply continue past the
   shared prefix. Concretely: for budgets `[8, 64, 256]` and a shared
   prompt, the first 8 tokens of every seq's hash-input must be
   pairwise equal; the first 64 of seqs with budget ≥ 64 must be
   pairwise equal; etc.
4. **Different decode budgets can finish at different times.** The
   per-seq `done` flag flips at different iterations. The gate must
   show the iteration count at which each seq became `done` and these
   counts must be `decode_budget[seq] - 0` (since the smoke case
   collapses prefill into iteration 0; iteration `k > 0` produces the
   `(k+1)`-th decoded token for every still-active seq).
5. **KV-clear is effective.** After `llama_memory_seq_rm(mem, seq,
   -1, -1)` on a completed `seq`, `llama_memory_seq_pos_min(mem,
   seq)` returns the "no entries" sentinel (`-1` in the upstream
   convention). Other live `seq_id`s' `seq_pos_min/max` are unchanged
   by the clear of a sibling.

   **KV position convention.** For prompt length `P` and decode
   budget `D`, the expected per-seq KV state immediately before
   clear is:

   - `pos_min == 0` (the BOS / first prompt row)
   - `pos_max == P + D - 2`, **not** `P + D - 1`

   The reason: generated token #1 comes from the final prompt row's
   logits and does not occupy a new KV cell. Only `D - 1` of the
   generated tokens are fed back as decode rows, so KV holds `P + (D
   - 1) = P + D - 1` cells at positions `0 .. P + D - 2`.
6. **No overflow / no return-code surprises.** Every `llama_decode`
   call returns `0`. If any returns non-zero (`1` = ctx exceeded,
   `-1` = invalid input, `2` = retryable KV exhaustion), the gate
   fails with a clear diagnostic naming the iteration index and the
   total number of rows in the batch.
7. **Repeated runs match.** A second run of the same binary, same
   model, same args, produces byte-identical per-seq hashes. (The
   gate may run twice internally and compare, or rely on an external
   harness — the choice is an implementation detail.)

The gate emits a single-line `GATE: PASS` (or `GATE: FAIL: <reason>`)
on the last line of stdout so an external script can grep for it.

## Q5. What is explicitly out of scope?

The gate must **not** introduce any of the following. Each is its
own axis of complexity and will be handled in a later phase:

- HPX runtime / HPX async / HPX channels of any kind.
- HTTP server, REST routing, JSON parsing, OAI/Anthropic compat.
- Streaming partial responses, SSE.
- Cancellation tokens, request-disconnect detection.
- Priority classes / per-request priority.
- Prompt cache (host-RAM `server_prompt_cache`), prompt prefix reuse
  (`llama_memory_seq_add`).
- LoRA adapters, alora.
- Speculative decoding (no draft model).
- Multimodal / MTMD inputs.
- `n_cmpl > 1` parent/child KV fan-out (`llama_memory_seq_cp`).
- Distributed serving / multi-host.
- Sampler chain (`common_sampler_*`), temperature, top-k, top-p,
  penalties, grammar, JSON-schema constrained decoding.
- Chat templates, tokenizer special-tokens experiments beyond the
  default add-BOS behavior.
- Performance comparisons of any kind. The gate is a **correctness
  binary**, not a benchmark.
- Multi-backend toggling (CUDA/Metal/etc). The smoke case is CPU-only.
- Context shift, SWA, recurrent paths.

## Q6. What is the smallest smoke test?

The smallest run that satisfies the gate:

- **Sequences:** 3 (`seq=0,1,2`).
- **Prompt:** the canonical `"Hello, my name is"` already used by the
  serving-bench evidence (or any short, fixed prompt; the prompt is
  not the variable under test).
- **`prompt_tokens`:** 6 (matches the existing TinyLlama smoke shape;
  exact value is whatever the chosen tokenizer produces, the gate
  records it).
- **Decode budgets:** `[8, 64, 256]` (one per seq, taken from the
  Phase 3 primary target's decode-length classes).
- **Backing model:** TinyLlama 1.1B Q4_K_M (the same GGUF the
  serving-bench evidence uses).
- **`llama_context` parameters:**
  - `n_seq_max = 3` (or higher).
  - `n_ctx >= P + max(budgets) = 6 + 256 + slack`. Use 1024 for slack.
  - `n_batch >= N * P = 18` for the smoke prefill; using the upstream
    default (e.g. 512 or 2048) is fine.
- **CPU-only.** No GPU offload. Match the existing CPU_REPACK Q4_K_M
  baseline path documented in project memory.
- **Greedy argmax.** No sampler. No seed needed.
- **Iteration cap:** `max_iter = 1 + max(budgets) = 257`. Exceeding
  this is a fail.
- **Acceptance:** all seven correctness checks above pass; binary
  exits 0; stdout's last line is `GATE: PASS`.

This run is small enough to finish in seconds on TinyLlama-1.1B
CPU-only. It exercises:

- multi-`seq_id` shared prefill in iteration 0,
- multi-`seq_id` shared decode in iterations 1..256,
- staggered completion (seq 0 done after iter 7, seq 1 done after iter
  63, seq 2 done after iter 255),
- per-`seq_id` KV clear at each completion.

After this passes, the gate scales to `n=99` round-robined over
`{8, 64, 256}` (the Phase 3 primary target) by simply iterating the
same loop with a larger `seq_id` table. No structural changes
required.

## Q7. What is the next implementation step after this design note?

In order, with each step a stop-and-check gate:

1. **Create the tool skeleton** under `tools/multiseq-batch-gate/`:
   `CMakeLists.txt` linking against `libllama` and `libcommon`,
   minimal `main()` that loads model + ctx, prints
   `n_seq_max / n_batch / n_ctx`, and exits. No batch work yet.
2. **Single-seq baseline** (no multi-seq yet): one `seq_id`, fixed
   prompt, greedy argmax. Two runs:
   - **First run, decode budget = 8.** Print the generated token
     hash. Expected:
     `hash == 0x0619d4d1900c2365`.
   - **Optional second sanity run, decode budget = 16.** Print the
     generated token hash. Expected:
     `hash == 0x833045f1e2ebf49f` (the canonical 16-token TinyLlama /
     `"Hello, my name is"` fingerprint from `CLAUDE.md`).

   Each hash is **shape-specific**: a budget-8 run cannot produce the
   16-token canonical hash, and vice versa. The 16-token hash is
   referenced only for the budget-16 run and must not be claimed as
   evidence for the budget-8 run.
3. **Two-seq case**: same prompt for both seqs, both budget 8.
   Verify both seqs' hashes are equal (Q4 check #3). This is the
   smallest non-trivial multi-`seq_id` shape.
4. **Three-seq smoke (Q6)**: budgets `[8, 64, 256]`. All seven
   correctness checks active. `GATE: PASS` on stdout.
5. **Scale-up to 99 seqs** round-robined over `{8, 64, 256}`. Same
   correctness checks; expect PASS. This produces evidence at the
   Phase 3 primary target shape.
6. **Hand off**: at this point the gate has proven that real
   `llama.cpp` can host multiple `seq_id`s in one shared `llama_batch`
   with a single shared `llama_context` and that per-`seq_id` KV
   cleanup works. HPX integration may then begin, owning the
   request/slot lifecycle around this same primitive.

Each step writes its evidence (stdout capture + the per-seq token
hash table) under `local/` for inspection. None of these steps writes
under `hpx-bench/results/` — that path is reserved for committed
benchmark evidence, and this is correctness work, not benchmarking.

## Status

This document is **design only**. Implementation has not started.

When implementation begins, the first step is the tool skeleton (Q7
step 1). Each subsequent step requires the previous one to pass.

Source-of-truth signals when revisiting this note:

- Upstream API surface: `docs/hpx/continuous_batching_upstream_notes.md`.
- Phase 3 target choice: `docs/hpx/continuous_batching_phase3_target.md`.
- Simulator evidence backing the C target:
  `hpx-bench/sim/continuous_batching/phase2b_mixed_workload.md`.

If any of those change in a way that invalidates the smoke shape or
the API list, this note must be updated before the gate is built.
