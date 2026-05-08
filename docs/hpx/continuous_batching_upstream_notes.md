# Continuous Batching — Upstream `llama-server` Notes

Phase 1 reading note for the HPX continuous-batching direction. This document describes how upstream `tools/server/` works *as it stands today*. It is not a design for HPX. It is the reference point we will compare an HPX prototype against.

Citations point at `tools/server/` files in this tree (snapshot 2026-04-30).

---

## 1. Main components of `llama-server`

From `tools/server/README-dev.md` and the headers, the inference-mode server is built from these layers:

```
HTTP threads (cpp-httplib, many)
  └─ server_http_context  (server-http.{h,cpp})
        └─ server_routes  (server-context.cpp, init_routes())
              ├─ parses JSON, applies chat template, tokenizes
              ├─ creates server_task(s)
              └─ hands them to a server_response_reader

         queue_tasks (server_queue)   <─── HTTP → engine
         queue_results (server_response) ──> engine → HTTP

server_context (single dedicated thread)
  └─ server_context_impl  (server-context.cpp)
        ├─ owns the one llama_model, llama_context, mtmd_context
        ├─ owns std::vector<server_slot> slots
        ├─ owns the single shared llama_batch
        ├─ owns server_prompt_cache (host-RAM prompt cache)
        ├─ runs the main loop:
        │     callback_new_task(task)        → process_single_task()
        │     callback_update_slots()        → update_slots()
        │     callback_sleeping_state(bool)  → handle_sleeping_state()
        └─ pushes server_task_result into queue_results
```

Other components mentioned in `README-dev.md` and used in the source:

- `server_slot` — per-sequence inference state (the unit of "parallel request"; `slots.size() == params.n_parallel`).
- `server_task` / `server_task_result` — units of work pushed/popped through the two queues. Polymorphic `server_task_result` (`*_cmpl_partial`, `*_cmpl_final`, `*_embd`, `*_rerank`, `*_metrics`, `*_error`, …).
- `server_response_reader` — generator-like wrapper around `(queue_tasks, queue_results)`. Owns `id_tasks`, drains results, posts `SERVER_TASK_TYPE_CANCEL` on disconnect.
- `server_tokens` — unified token sequence (text + multimodal chunks). Used both as `task.tokens` (input) and as `slot.prompt.tokens` (cached / KV-resident).
- `server_prompt` / `server_prompt_checkpoint` / `server_prompt_cache` — host-RAM cache of (token-sequence, serialized seq state). Lets a different request reuse a previous slot's KV state when prefixes match.
- `server_models` — only used in router mode (process-per-model proxy). Out of scope for continuous batching.
- `server_metrics` — counters surfaced via `/metrics` and `/slots`.

Router mode (`tools/server/server-models.cpp`) is a separate concern: it proxies HTTP to per-model child server processes. Everything below describes single-model inference mode (`server_context_impl`).

## 2. Lifecycle of one completion request

Trace, mostly mirroring `README-dev.md` §"Example trace of a request" but with file:line:

1. **HTTP thread:** request arrives at `cpp-httplib`, dispatched in `server-http.cpp:process_handler_response`. The request struct includes `req.is_connection_closed` as a `std::function<bool()>` named `should_stop`.
2. **HTTP thread:** routed to `server_routes::handle_completions_impl` (`server-context.cpp:3185`). It:
   - parses JSON (`json_value`, `params_from_json_cmpl`),
   - tokenizes via `tokenize_input_prompts` (or MTMD path),
   - constructs one `server_task` per prompt; for `n_cmpl>1` it adds child tasks via `task.add_child(...)`,
   - calls `rd.post_tasks(...)` on the `server_response_reader`. That registers `id_tasks` in `queue_results`'s waiting set, then `queue_tasks.post(...)` pushes the tasks.
3. **Engine thread:** `server_queue::start_loop` (`server-queue.cpp:125`) wakes, pops tasks, calls `callback_new_task` → `process_single_task` (`server-context.cpp:1847`). For COMPLETION/INFILL/EMBEDDING/RERANK it picks a slot via `get_available_slot` (LCP-similarity → LRU) and calls `launch_slot_with_task` (sets samplers, lora, alora invocation, moves task into slot, transitions `SLOT_STATE_IDLE → SLOT_STATE_STARTED`). If no slot is free or requested slot is busy, the task is `queue_tasks.defer(...)`'d.
4. **Engine thread:** after every batch of new-task processing the loop calls `callback_update_slots` → `update_slots()` (`server-context.cpp:2132`). One iteration:
   - apply ctx-shift to any GENERATING slot at risk of overrunning `n_ctx`,
   - clear shared `batch`,
   - append the **already-sampled token** (and any speculative draft) of every GENERATING slot,
   - if `cont_batching` (or batch is empty), append **prompt tokens** of every PROCESSING_PROMPT/STARTED slot up to `n_batch`,
   - call `llama_decode(ctx, batch_view)` (potentially chunked by `n_batch`),
   - for each slot whose `i_batch` falls in the just-decoded slice:
     - if EMBEDDING/RERANK: emit result, release slot,
     - else: `common_sampler_sample(...)` → `process_token(...)` → if streaming, immediately `send_partial_response`. On stop: `send_final_response` + `slot.release()`.
5. **HTTP thread (concurrently):** sits inside `chunked_content_provider` (streaming, `server-http.cpp:395`) or `wait_for_all` (non-streaming). Each `next(should_stop)` call pulls the next `server_task_result_ptr` off `queue_results`. For SSE responses it formats and writes to `httplib::DataSink`.
6. **Termination:**
   - normal stop → `*_cmpl_final` posted; reader's `received_count` reaches `id_tasks.size()`; `has_next()` returns false; HTTP layer writes the closing `data: [DONE]\n\n`.
   - client disconnect → `req.is_connection_closed` flips true → `should_stop` returns true → reader returns `nullptr` → `server_response_reader::stop()` posts `SERVER_TASK_TYPE_CANCEL` tasks to the FRONT of the queue. Engine thread sees the cancel, finds the slot whose `task->id == id_target`, calls `slot.release()`. `cleanup_pending_task()` (called from inside `queue_tasks.post(CANCEL)`) also purges any not-yet-launched tasks with that id.

Single physical thread runs steps 3–4. All HTTP threads pump into the same two queues.

## 3. What a slot is

Defined in `server-context.cpp:79` (`struct server_slot`). A slot is a **persistent claim on one `llama_seq_id`** plus all the per-request state that hangs off it. There are exactly `params.n_parallel` of them, created once at startup. Key fields:

- `int id` — both the slot index and the `llama_seq_id` used in `llama_batch.seq_id[*]` and in every `llama_memory_seq_*` call. So slot id ≡ sequence id, by construction (`server-context.cpp:895-898`).
- `llama_context * ctx` — pointer to the **shared** `llama_context` (all slots share one).
- `slot_state state` — `IDLE | WAIT_OTHER | STARTED | PROCESSING_PROMPT | DONE_PROMPT | GENERATING`.
- `unique_ptr<const server_task> task` — currently launched task; null when IDLE.
- `server_prompt prompt` — the slot's **view of what is currently in the KV cache**: token vector + serialized state + a `std::list<server_prompt_checkpoint>` for SWA/recurrent fallback.
- `int32_t n_ctx` — per-slot context size (`llama_n_ctx_seq(ctx)`, capped at `n_ctx_train`).
- `int32_t n_decoded`, `n_prompt_tokens_processed`, `n_prompt_tokens_cache` — counters/timing.
- `int32_t i_batch` — index *inside the shared batch* of the row whose logits this slot wants. Set when prompt is finished (`batch.n_tokens - 1`) and when the next decode token is added. Used to map decoder output back to slot in step 4.
- `common_sampler_ptr smpl`, `llama_token sampled` — sampler chain and last-sampled token.
- `common_speculative_ptr spec`, `llama_tokens spec_draft`, `spec_i_batch`, `spec_ckpt` — speculative decoding state when a draft model is configured.
- `std::vector<common_adapter_lora_info> lora`, `int32_t alora_invocation_start` — per-slot LoRA scales.
- generation/output: `generated_text`, `generated_tokens`, `generated_token_probs`, `last_nl_pos`, `has_next_token`, `has_new_line`, `truncated`, `stop`, `stopping_word`, `n_sent_text`.
- timing: `t_start_process_prompt`, `t_start_generation`, `t_prompt_processing`, `t_token_generation`, `t_last_used`.
- `callback_on_release` — set in `load_model` to `queue_tasks.pop_deferred_task(id_slot)`, so freeing a slot pulls another task in.

Methods worth noting: `update_batch(batch)` (append next decode token, used in pass 1 of `update_slots`), `release()` (transition to IDLE, fire callback, reset stats), `reset()`, `prompt_save/prompt_load/prompt_clear`, `can_batch_with(other)` (same task type AND identical LoRA), `can_split()` (false for embedding without LAST pooling), `copy_state_to(other)` (used by `n_cmpl>1` to fan out the parent's KV to N children via `llama_memory_seq_cp`).

## 4. How `n_slots`, `n_batch`, `n_ubatch`, `n_ctx`, and continuous batching interact

- **`n_slots = params.n_parallel`** (`server-context.cpp:888`). Number of concurrent sequences. Caps maximum in-flight requests.
- **`n_ctx`** is the *total* KV cache, shared across all sequences. Per-slot context is `n_ctx_slot = llama_n_ctx_seq(ctx)` (`server-context.cpp:870`). With `--kv-unified` all slots share one pool; without it `n_ctx` is split per-slot.
- **`n_batch`** = `llama_n_batch(ctx)` (`server-context.cpp:2248`). The maximum number of token rows that may be submitted to a single `llama_decode` call. The shared `batch` itself is allocated to `max(n_batch, n_parallel)` rows (`server-context.cpp:944`) so that the per-iteration "one decode token per GENERATING slot" pass always fits even when `n_batch < n_parallel`.
- **`n_ubatch`** = `llama_n_ubatch(ctx)`. The physical microbatch size used by the backend; prompt processing splits at this internally. The server uses it as the safety margin for two things:
  1. embedding/rerank tasks (`!can_split()`) must fit in one ubatch (`server-context.cpp:2324`),
  2. checkpoint placement: server tries to break out of prompt-batching `4` and `4 + n_ubatch` tokens before the end of a long prompt to make a meaningful checkpoint possible (`server-context.cpp:2664-2678`).
- **`cont_batching`** is the boolean gate on the second pass of `update_slots`:

  ```cpp
  if (params_base.cont_batching || batch.n_tokens == 0) {
      for (auto & slot : slots) { ... append prompt tokens ... }
  }
  ```

  (`server-context.cpp:2255`). With it ON, prompt prefill and decode can share one `llama_decode` call. With it OFF, prompts only go in when no slot is currently generating — i.e. behave like batched-but-not-continuous.
- Embedding requirement (in `tools/server/server.cpp:88-93`): if `--embedding` is set and `n_batch > n_ubatch`, both are forced down to `n_ubatch`. Embedding pooling needs the whole input in a single ubatch.

## 5. How `update_slots()` works

Reading order matches `server-context.cpp:2132-3070`. Single function, ~940 lines.

```
0. early return if all slots are IDLE                                     (2132-2148)
1. post a NEXT_RESPONSE token to keep the loop turning                    (2151-2157)
2. ctx-shift any GENERATING slot whose prompt+1 ≥ n_ctx                   (2161-2218)
   - llama_memory_seq_rm + llama_memory_seq_add to slide tokens left
   - sets slot.truncated=true
3. common_batch_clear(batch)                                              (2221)
4. PASS A — append already-sampled tokens for every GENERATING slot       (2232-2245)
   - via slot.update_batch(batch); chooses one canonical "slot_batched"
   - skips slots that fail can_batch_with(slot_batched) (different LoRA / type)
5. PASS B — if cont_batching OR batch empty: append PROMPT tokens         (2255-2751)
   For each slot in PROCESSING_PROMPT / STARTED:
     - skip if !can_batch_with(slot_batched) or state==WAIT_OTHER
     - if STARTED: stamp t_start_process_prompt; switch to PROCESSING_PROMPT.
     - prefix matching:
         n_past = slot.prompt.tokens.get_common_prefix(input_tokens)      (2358)
         (clamped before alora_invocation_start)
         optionally KV-shift reuse via llama_memory_seq_rm / seq_add      (2390-2424)
       Then for SWA/recurrent models: try to restore a checkpoint;
       fall back to full reprocess if no usable checkpoint                (2435-2538)
     - guarantee at least 1 token to evaluate (n_past-- if all matched)   (2542-2546)
     - llama_memory_seq_rm(memory, slot.id, p0=pos_next, -1)              (2572) — drop tail
     - emit MTMD chunks via input_tokens.process_chunk if present         (2611-2631)
     - while (slot.prompt.n_tokens() < task.n_tokens && batch.n_tokens < n_batch):
         common_batch_add(batch, tok, pos, {slot.id}, slot.task->need_embd())
         slot.prompt.tokens.push_back(tok)
         slot.n_prompt_tokens_processed++
         possibly break to leave room for an end-of-prompt checkpoint     (2664-2678)
     - if prompt complete: state→DONE_PROMPT; batch.logits[last]=true;
       slot.i_batch=batch.n_tokens-1; slot.init_sampler()                 (2685-2697)
     - optionally create a partial state checkpoint                       (2723-2740)
     - break outer loop if batch.n_tokens >= n_batch                      (2747-2749)
6. set adapter LoRA + embeddings flag from slot_batched                   (2755-2767)
7. CHUNKED DECODE:                                                        (2782-2845)
     for (i = 0; i < batch.n_tokens; i = i_next):
        n_tokens = min(n_batch, batch.n_tokens - i)
        batch_view = subrange(batch, i, n_tokens)
        ret = llama_decode(ctx, batch_view)
        if ret != 0:
          - ret==1 with n_batch==1 → "Context size has been exceeded"; release all
          - ret==-1 → "Invalid input batch"; release all
          - ret==2 (KV exhausted, retryable):
              try_clear_idle_slots() (purges idle slot KV) OR n_batch /= 2
              continue   ← the same i is retried with a smaller view
        i_next = i + n_tokens
8. PASS C — for each slot whose i_batch in [i, i+n_tokens):                (2877-2960)
     - DONE_PROMPT + EMBEDDING → send_embedding, release
     - DONE_PROMPT + RERANK    → send_rerank, release
     - DONE_PROMPT + completion → state→GENERATING; common_speculative_begin
     - sample: id = common_sampler_sample(slot.smpl, ctx, tok_idx)
     - common_sampler_accept(...)
     - bump n_decoded; first-token timing flips t_start_generation
     - process_token(result, slot):
         appends to generated_text, fires send_partial_response if streaming,
         decides has_next_token (EOS/limit/stopword/n_predict/ctx-overflow/indent)
     - if !has_next_token: slot.print_timings(); send_final_response;
       metrics.on_prediction; slot.release()
9. PASS D — speculative-only: verify accepted draft, restore checkpoint
   on partial acceptance, run process_token for each accepted token.        (2962-3067)
```

Important loop invariant (PASS C): a slot is mapped back to its row in the just-decoded batch view via `slot.i_batch`. If `slot.i_batch ∉ [i, i+n_tokens)` the slot is skipped this iteration; this happens naturally when the batch was chunked.

## 6. How the shared `llama_batch` is built

There is exactly one `llama_batch` per server (`server_context_impl::batch`, allocated once in `load_model`, freed in `destroy()`). It is reused — `common_batch_clear(batch)` at the start of every `update_slots()`.

Population is two passes inside `update_slots`:

1. **Decode rows** — for each GENERATING slot, `slot.update_batch(batch)` calls `common_batch_add(batch, sampled, pos_next, {slot.id}, /*logits=*/true)` (`server-context.cpp:381`). In speculative mode it also appends each draft token.
2. **Prompt rows** — for each non-GENERATING active slot, in the inner `while` loop, `common_batch_add(batch, cur_tok, pos_next, {slot.id}, /*logits=*/need_embd)` (`server-context.cpp:2650`). Logits are turned on only for the last row of a finished prompt (`batch.logits[batch.n_tokens-1] = true` at `server-context.cpp:2691`).

So each row carries `(token, pos, seq_id={slot.id}, logits_bool)`. There is **one `seq_id` per row** in the upstream code — no row is shared by two slots in the same batch.

## 7. Can prompt prefill and decode tokens from different slots be batched together?

Yes — when `--cont-batching` is on (`params_base.cont_batching == true`), or as a fallback when no GENERATING slot contributed any decode token in this iteration. PASS A (decode of GENERATING slots) and PASS B (prefill of PROCESSING_PROMPT slots) populate the *same* `batch` before the single `llama_decode` call. This is the upstream definition of "continuous batching" in this codebase.

Constraints on what can co-exist in one batch:
- `slot.can_batch_with(other)` requires same `task->type` and identical `lora` vector. The first contributing slot becomes `slot_batched`; later slots that fail `can_batch_with(slot_batched)` are skipped this iter (`server-context.cpp:2240, 2262`).
- Total rows ≤ `n_batch` (the prompt-fill loop breaks; the decode loop chunks).
- `WAIT_OTHER` slots (children of an `n_cmpl>1` parent) wait until the parent's `DONE_PROMPT` is reached before being filled, then the parent's seq state is `seq_cp`'d into them.

If `--cont-batching` is off, prompt rows are added only when `batch.n_tokens == 0` — so prefill and decode are temporally interleaved on different `update_slots` iterations rather than coexisting in one decode.

## 8. When exactly is `llama_decode` called?

Exactly once per chunk per `update_slots()` iteration, at `server-context.cpp:2795`. Chunking is determined by `n_batch` (`for (i = 0; i < batch.n_tokens; i = i_next)`). On retryable failure the same `i` is decoded again with smaller `n_batch` after either clearing an idle slot's KV or halving `n_batch`.

Surrounding sequence in one iteration:
```
common_batch_clear(batch);
... pass A and B populate batch ...
common_set_adapter_lora(ctx, slot_batched->lora);
llama_set_embeddings(ctx, slot_batched->task->need_embd());
for chunked views: llama_decode(ctx, batch_view); metrics.on_decoded(slots);
```
One `update_slots()` corresponds to one logical "macro batch"; `llama_decode` is the synchronous compute call for the slice of that macro batch that fits in `n_batch`.

## 9. After `llama_decode`, how are logits mapped back to slots?

Each slot stores `i_batch`, the row in the *full* shared batch where its target output sits. Two cases:

- For prompt completion of a slot: when its prompt finishes filling, `slot.i_batch = batch.n_tokens - 1` and `batch.logits[batch.n_tokens - 1] = true` (`server-context.cpp:2691-2694`). The single trailing logits row carries that slot's "first token" prediction.
- For an ongoing GENERATING slot: `slot.update_batch` records `i_batch = batch.n_tokens` *before* the one row is appended (`server-context.cpp:379`). For speculative, `spec_i_batch` is a vector of the rows.

After each chunked `llama_decode` (`server-context.cpp:2782` outer `for`):

```cpp
const int tok_idx = slot.i_batch - i;
llama_token id = common_sampler_sample(slot.smpl.get(), slot.ctx, tok_idx);
```

So the slot's row in the batch view is `slot.i_batch - i`, and that index is passed straight to the sampler / `llama_get_logits_ith` / `llama_get_embeddings_ith` / `llama_get_embeddings_seq` (the embeddings paths use `seq_id`, not `i_batch`, see `send_embedding` at `server-context.cpp:1671`).

The mapping is set BY the slot when populating the batch, READ by the slot post-decode. There is no global table; each slot tracks its own row.

## 10. KV cache management per slot

The KV cache is owned by the shared `llama_context` and addressed via `llama_get_memory(ctx)` plus a `seq_id`. The server uses these primitives:

- `llama_memory_seq_rm(mem, seq_id, p0, p1)` — drop KV cells for `seq_id` in `[p0, p1)`. `(p0=-1, p1=-1)` clears all cells. Used at `prompt_clear` (`:166`), tail truncation before re-fill (`:2572`), context shift (`:2197`), KV-cache reuse pre-shift (`:2408`), and post-decode tail trim after speculative checkpoint restore (`:3010`, `:3045`).
- `llama_memory_seq_add(mem, seq_id, p0, p1, delta)` — shift positions of cells. Used in context shift (`:2198`) and prompt-prefix-reuse KV shifting (`:2409`).
- `llama_memory_seq_cp(mem, src, dst, p0, p1)` — copy a range from one seq to another. Used by `slot::copy_state_to` (`:548`) for `n_cmpl>1` fan-out.
- `llama_memory_seq_pos_min/max(mem, seq_id)` — boundaries (used to set up checkpoints and validate prefix reuse).
- `llama_state_seq_get_data_ext` / `set_data_ext` / `save_file` / `load_file` — serialize partial seq state for prompt cache, SWA checkpoints, and `/slots/:id` save/restore.

Per-slot view: `slot.prompt.tokens` mirrors what is *currently* in the KV for `seq_id == slot.id`. After a `seq_rm` the server immediately updates `slot.prompt.tokens` to keep them coherent (`keep_first(n_past)` at `:2551`, `clear()` at `:167`, etc.). When this mirror drifts from the actual KV the assertions or the next decode catch it (e.g. the `pos_min == -1` abort at `:2439`).

Eviction: there is no automatic per-slot eviction. The host-RAM `server_prompt_cache` (`server-task.h:622`) optionally archives a slot's `(tokens, serialized state)` when the slot is reused (`get_available_slot` calls `prompt_save` / `prompt_load`), so future requests with a matching prefix can warm-restart a slot. With `--cache-idle-slots` (requires `--kv-unified --cache-ram`), idle slots' state is also moved out of KV after each new task arrives (`process_single_task` → `slot_save_and_clear`).

## 11. Relationship between slot and `llama_seq_id`

One-to-one. Set at slot construction:

```cpp
for (int i = 0; i < params_base.n_parallel; i++) {
    slot.id = i;                                // server-context.cpp:895
    slot.ctx = ctx;
    ...
}
```

Every `common_batch_add` call uses `{ slot.id }` as the `seq_id`. Every `llama_memory_seq_*` call uses `slot.id`. There is no remapping; a slot owns its sequence id for the lifetime of the process.

The unification happens at startup: with `n_parallel < 0`, `server.cpp` sets `n_parallel = 4` and `kv_unified = true`. With `kv_unified` the shared `llama_context`'s memory is one pool addressed by seq_id; without it, the context is partitioned `n_ctx / n_parallel` ways (one slice per seq_id).

## 12. Scheduling policy for waiting requests

In `process_single_task` (`server-context.cpp:1865-1909`):

1. If `task.id_slot != -1`, look that slot up. If it's busy, **defer**.
2. Else call `get_available_slot(task)`:
   - if `slot_prompt_similarity > 0`: scan idle slots with non-empty prompt cache, pick the one whose `tokens.get_common_prefix(task.tokens) / task.tokens.size()` is highest above the threshold (LCP-similarity).
   - else / fallback: scan idle slots, pick the one with smallest `t_last_used` (LRU).
3. If still nothing free, **defer**.
4. For `task.is_parent()` (n_cmpl>1): also need `n_child_tasks` other free slots via `get_free_slots(...)`. If not enough, defer the whole parent.
5. On launch, optionally call `slot_save_and_clear` on every other idle slot if `--cache-idle-slots`.

Deferred tasks live in `server_queue::queue_tasks_deferred`. They come back via `pop_deferred_task(id_slot)`, which is called from `slot.callback_on_release`. `pop_deferred_task` first looks for a deferred task that explicitly named this slot id; failing that, pops the front of the deferred queue. This means: as soon as a slot frees, exactly one deferred task is promoted back into the main queue.

`SERVER_TASK_TYPE_CANCEL` is special-cased: `queue_tasks.post(CANCEL)` calls `cleanup_pending_task(id_target)` first, which erases any not-yet-launched task with that id from both the main and deferred queues (`server-queue.cpp:211`).

Priority: cancel tasks and `metrics` requests are posted with `front=true`. Normal completion requests go to the back.

## 13. What happens when all slots are busy

Two interlocking behaviors:

- **HTTP layer (instant)**: the request is queued, not rejected. `handle_completions_impl` returns nothing visible until at least one result arrives; for streaming it begins emitting the SSE stream only after the first result. Client sees latency, not 503.
- **Engine layer**: `process_single_task` defers the task into `queue_tasks_deferred`. The slot freeing up triggers `pop_deferred_task`, which promotes one deferred task to the front of the main queue, and the next loop iteration calls `callback_new_task` on it.

There is no admission control limit beyond memory and the `cache-ram` budget. There is no per-client fairness (all deferred tasks are FIFO, with optional preference for ones that name a specific slot id).

In the rare case of decode-time KV exhaustion (`llama_decode` returns positive ret), the engine reactively tries `try_clear_idle_slots()` to free space, then halves `n_batch` and retries. That's the only place a "soft eviction of idle work" happens automatically.

## 14. How continuous batching affects short vs long requests

Short requests (small prompt, small `n_predict`):
- With many slots busy, a short request's prefill rides along in the same `llama_decode` call as everyone else's decode tokens, paying ~one decode worth of latency for first-token.
- After `DONE_PROMPT`, every iteration adds one decode row for it. Time-to-completion is roughly `n_predict × per-iter-latency`, where per-iter-latency grows with the number of co-resident slots (more rows per `llama_decode`).

Long requests (large prompt):
- Prefill chunks across multiple `update_slots` iterations because of the `batch.n_tokens < n_batch` cap and the per-iteration `n_batch` limit. While its prefill is in progress, other slots' decode rows are still appended in PASS A, so the long request does not stall short requests' generation.
- The "leave room for a checkpoint" break-out (`:2664`) intentionally stops the prefill 4 (or `4+n_ubatch`) tokens early so the SWA/recurrent paths can checkpoint.

Net effect: short requests get reasonable first-token latency even behind a long prefill, *as long as* `cont_batching` is on. With it off, prompts only enter when no slot is generating, which makes a long prefill block all decode for its duration.

This is the mechanism. There is no priority queue or work-stealing — just opportunistic packing per iteration.

## 15. Cancellation / disconnect detection

End-to-end flow:

1. **Detection**: `cpp-httplib` sets a `req.is_connection_closed` callback on every incoming request. The server wraps it as `should_stop` and threads it through `server_http_req::should_stop` (`server-http.h:52`), `server_response_reader::next(should_stop)`, and the streaming `chunked_content_provider` (`server-http.cpp:395`).
2. **Streaming path**: each iteration of `chunked_content_provider` checks `req.should_stop()` (`server-context.cpp:3330`). If true it returns `false`, which closes the SSE response. `rd.next` also re-checks `should_stop` between polling intervals (`server-queue.cpp:382`).
3. **Reader teardown**: `~server_response_reader()` calls `stop()`. `stop()` removes the task ids from `queue_results.waiting_task_ids` and posts `SERVER_TASK_TYPE_CANCEL` for each id to the FRONT of `queue_tasks` (`server-queue.cpp:431`).
4. **Engine response**: when the engine pops a CANCEL task, `cleanup_pending_task(id_target)` runs first inside `post()` to delete any non-launched copy. Then `process_single_task` finds the slot whose `task && task->id == id_target` and calls `slot.release()` (`server-context.cpp:1911-1920`). The slot transitions to IDLE and fires `callback_on_release`, pulling the next deferred task in.
5. **Non-streaming path**: `wait_for_all(should_stop)` returns `is_terminated=true` on disconnect (`server-queue.cpp:409`). Same teardown via the reader's destructor.

There is no "abort mid-decode": the engine's current `llama_decode` call runs to completion, then the next iteration's `update_slots` honors the cancellation by not appending a row for the released slot.

## 16. Streaming and the decode loop

Streaming output is produced **inline inside `update_slots`**, not on a separate emitter thread:

- `process_token(result, slot)` (`server-context.cpp:1351`) runs after each `common_sampler_sample`. If the slot's task asks for streaming (`task->params.stream == true`), it calls `send_partial_response(slot, result, false)` immediately (`server-context.cpp:1395`).
- `send_partial_response` builds a `server_task_result_cmpl_partial` and pushes it to `queue_results.send(...)`. That notifies the condition variable (`server-queue.cpp:319`).
- The HTTP thread, sitting in `server_response_reader::next` → `recv_with_timeout`, wakes up, picks up the result, formats it as SSE (`format_oai_sse` / `format_anthropic_sse` / `format_oai_resp_sse`), and writes a chunk into the `httplib::DataSink`.

Because emit happens inside the engine loop:
- the engine's single-threadedness imposes a serialization point — heavy JSON formatting in HTTP threads is fine, but heavy work in `process_token` would directly slow generation across all slots (warning in `README-dev.md` §"Thread Management"),
- partial chunks are emitted in the same order as they were sampled within an iteration, but interleavings *between slots* happen at single-token granularity within one `update_slots` call.

`send_partial_response` is also used to emit progress chunks during prefill of long prompts (`:2556`, `:2881`), gated on `task.params.return_progress`.

For non-streaming requests, only `send_final_response` runs at the end and `wait_for_all` aggregates the single result.

## 17. What tests exist under `tools/server/tests`

`tools/server/tests/` is a pytest suite (see `tests/README.md`). It builds the server binary via cmake, then `tests.sh` runs `pytest` against a child process spawned by `utils.py:ServerProcess`. Tiny default model is `tinyllamas/stories260K.gguf` from `ggml-org/models` HuggingFace repo, downloaded into `tmp/` (or `LLAMA_CACHE`).

Files in `tests/unit/`:

| File                              | Purpose                                                                 |
| --------------------------------- | ----------------------------------------------------------------------- |
| `test_basic.py`                   | `/health`, `/props`, `/models`, `/slots`, model aliases, split GGUF, no-webui |
| `test_completion.py`              | `/completion` non-stream + stream, parallel slots, KV-unified split, response_fields, n_probs, logit_bias, **cancel**, host-RAM prompt cache |
| `test_chat_completion.py`         | `/chat/completions` (incl. OAI library), JSON schema, grammar, logprobs, context-size-exceeded, return_progress |
| `test_embedding.py`               | `/embedding` and `/v1/embeddings`, pooling modes, prompt-too-long, OAI library, base64 encoding |
| `test_rerank.py`                  | `/rerank`, top_n, TEI format, usage tokens                              |
| `test_infill.py`                  | `/infill` with/without `input_extra`, Qwen FIM                          |
| `test_speculative.py`             | Draft model on/off, n_min/n_max, slot ctx not exceeded, ctx_shift compat, parallel speculative |
| `test_ctx_shift.py`               | ctx_shift enabled/disabled, short/long prompt, stream                  |
| `test_kv_keep_only_active.py`     | `--cache-idle-slots` behavior; flag-disabled fallback                  |
| `test_slot_save.py`               | `/slots/:id` save / restore / erase                                    |
| `test_lora.py`                    | LoRA scale parameter, per-request adapters, big-model toggle           |
| `test_template.py`                | Chat template reasoning toggles, date-in-prompt, add_generation_prompt |
| `test_tokenize.py`                | `/tokenize`, BOS handling, with_pieces                                 |
| `test_tool_call.py`               | OAI tool calls (large; multiple presets)                               |
| `test_compat_anthropic.py`        | `/v1/messages` Anthropic-compat                                        |
| `test_compat_oai_responses.py`    | `/v1/responses` OAI-compat                                             |
| `test_vision_api.py`              | MTMD image input                                                       |
| `test_security.py`                | API keys, public endpoints, CORS, local-media-file gating              |
| `test_router.py`                  | Router-mode multi-model proxy                                          |
| `test_sleep.py`                   | `--sleep-idle-seconds` enter/exit sleeping state                       |
| `test_proxy.py`                   | CORS proxy (experimental)                                              |
| `test_ignore_eos.py`              | `ignore_eos` populates logit_bias                                      |

`tests/utils.py` is the main fixture (`ServerProcess`) — it boots a server with the requested flags, polls `/health`, and offers `make_request` / `make_stream_request` helpers. `parallel_function_calls` is a `ThreadPoolExecutor` that lets a test fire N concurrent HTTP calls.

## 18. Tests covering each behavior

- **Slots / slot lifecycle**: `test_basic.test_server_slots` (slots endpoint), `test_completion.test_completion_parallel_slots` (asserts `is_processing` count), `test_completion.test_completion_unified` (`--kv-unified` parametrized over `n_ctx, n_slots, n_predict_vals`), `test_kv_keep_only_active.*`, `test_slot_save.*`.
- **Parallel requests / continuous batching**: `test_completion.test_completion_parallel_slots` (parametrized `(n_slots, n_requests)` including over-subscription), `test_completion.test_completion_unified`, `test_speculative.test_multi_requests_parallel`.
- **Cancellation**: `test_completion.test_cancel_request` — `n_predict=-1`, sends a request with `timeout=0.1`, then asserts `/slots[0]["is_processing"] == False` after a 2-second wait.
- **Streaming**: `test_completion.test_completion_stream`, `test_completion.test_completion_stream_vs_non_stream`, `test_chat_completion.test_chat_completion_stream`, `test_chat_completion.test_logprobs_stream`, `test_chat_completion.test_completion_stream_with_openai_library*`, `test_chat_completion.test_context_size_exceeded_stream`, `test_ctx_shift.test_ctx_shift_disabled_stream`, `test_completion.test_n_probs_stream`.
- **Embeddings**: all of `test_embedding.py` (pooling modes, FA, mixed input, error paths, OAI library, base64). `test_embedding.test_embedding_multiple_with_fa` exercises flash-attn.
- **Metrics**: `test_basic.test_server_props` and `test_completion.test_completion_parallel_slots` indirectly hit `/slots`. There is no dedicated `/metrics` correctness test in the unit set; the `bench/` directory has end-to-end `k6` benchmarking that consumes `/metrics`.
- **Reranking**: `test_rerank.py` (full set).
- **Tool calling**: `test_tool_call.py` (large file, multiple model presets).
- **Sleeping mode**: `test_sleep.test_server_sleep`.
- **Infill / FIM**: `test_infill.py`.

## 19. Test presets / tiny models we can reuse

- **Default tiny model**: `ggml-org/models / tinyllamas/stories260K.gguf` — single-file Stories260K, downloads in seconds. This is the default in `utils.py:51`, so any test that doesn't override `model_hf_repo/file` uses it. Good for slot-lifecycle, scheduling, parallel-request, cancel, streaming gates.
- **Alias / context defaults** in the same fixture: `model_alias="tinyllama-2"`, `n_predict` default unset (server default), `temperature=0.8`, `seed=42`. Override on the `ServerProcess` instance before calling `.start()`.
- **Larger / capability-specific models** appear only in tests that need them: `test_lora.test_with_big_model`, `test_infill.test_with_qwen_model`, `test_template.*` (Jinja-template variants), `test_vision_api.*` (MTMD), `test_speculative.*` (also pulls a draft). These are gated by `SLOW_TESTS=1`.
- **Test pattern we should mirror**: tests construct `server = ServerProcess()`, set `server.n_slots`, `server.n_ctx`, `server.server_continuous_batching = True`, etc., then `server.start()` boots a real binary and `server.make_request(...)` / `make_stream_request(...)` do the I/O. We can reuse the fixture verbatim for an HPX simulator that exposes the same HTTP surface, or we can call lower-level utilities (`tokenize_input_prompts`, `server_task::params_from_json_cmpl`) directly from a C++ harness.

Specifically for our work (TinyLlama-1.1B, deeper queues, short requests):
- the existing `(stories260K, n_slots∈{1,2,4}, n_requests∈{1..6})` parametrization in `test_completion_parallel_slots` is the closest match,
- our prompt fingerprint (canonical hash `0x833045f1e2ebf49f`) is already produced by `serving-bench` directly; the upstream server tests do not validate generated-token hashes, only string regexes.

## 20. Minimum subset of upstream behavior needed for a simulator

A simulator that *models* upstream continuous batching without doing real inference needs:

- A `slot_state` enum (IDLE / PROCESSING_PROMPT / DONE_PROMPT / GENERATING; we can drop WAIT_OTHER and STARTED initially).
- A `server_slot` analogue with `id`, `state`, `n_prompt_tokens_total`, `n_prompt_tokens_done`, `n_decode_done`, `n_predict`, `t_arrived`, `t_first_token`, `t_finished`.
- A `Request` analogue with `id`, `n_prompt`, `n_predict_target`, `t_submit`, `stream` flag.
- A `Queue` for arriving requests (FIFO + deferred).
- A scheduler matching `process_single_task`: pop task → assign to free slot via LRU (skip LCP-similarity, skip `n_cmpl>1`, skip cancel) → else defer.
- A `update_slots`-shaped loop:
  1. compute `batch_rows = sum_over_GENERATING_slots(1) + sum_over_PROCESSING_slots(min(n_remaining_prompt, n_batch_left))` capped at `n_batch`, with `cont_batching` toggle,
  2. advance time by `latency(batch_rows)` where `latency` is a function of total rows (model: linear in rows + fixed overhead, parameters fit from a real `serving-bench` run),
  3. for each slot whose row was a "last prompt token" → DONE_PROMPT in this step, GENERATING in the next,
  4. for each GENERATING slot whose row was its decode token: bump `n_decode_done`, fire stop logic (just `n_decode_done == n_predict_target` for the simulator).
- Cancellation: a cancel API that flips slot to IDLE before its next row.

What the simulator should **intentionally ignore**:

- Real KV cache mechanics, `seq_rm/seq_add/seq_cp`, checkpoints, SWA/recurrent paths, ctx-shift.
- Sampler chains (`common_sampler_*`), tokenization, chat templates, MTMD, LoRA, alora, reasoning, speculative decoding.
- Streaming text formatting, OAI / Anthropic SSE, JSON parsing.
- `server_prompt_cache` host-RAM eviction, `--cache-idle-slots`.
- `n_cmpl > 1` parent/child fan-out (use it later if we want to model it).
- Embedding and rerank request types.
- Sleeping mode, router mode, MCP proxy, /tools.

## 21. Minimum subset for a future HPX prototype

A useful HPX prototype must reproduce upstream's *correctness* in addition to its *scheduling shape*:

- HPX-owned: process-wide HPX runtime startup/shutdown, the engine's main loop, the task queue, the response queue, slot-acquisition futures, cancellation propagation.
- llama.cpp-owned (opaque): `llama_model`, `llama_context`, `llama_batch`, `llama_decode`, `llama_memory_seq_*`, sampler chain, tokenization, chat template, MTMD.
- Required upstream features to keep:
  - one `llama_model`, one `llama_context`, `n_parallel` slots each tied to a distinct `seq_id`,
  - per-slot `server_prompt` mirror of KV state,
  - shared `llama_batch` reused per iteration,
  - `update_slots`-style two-pass population (decode rows first, prompt rows second when `cont_batching`),
  - per-row `seq_id` set to `slot.id`, logits flag on last prompt row only,
  - `i_batch` mapping for sampler readback,
  - cancel via released slot transitions,
  - host-thread safety: only the engine thread touches `llama_context`, `llama_decode`, `llama_memory_seq_*`, and `llama_batch`.
- Optional / later:
  - LCP-similarity slot picking,
  - prompt cache (host-RAM) reuse,
  - SWA / recurrent / non-shiftable contexts,
  - speculative decoding (it changes `update_batch` and post-decode pass),
  - LoRA / alora,
  - context shift,
  - n_cmpl>1 parent/child slots,
  - MTMD chunks.

The first HPX prototype should restrict itself to: completion-only, single LoRA (or none), `kv_unified=true`, `--cont-batching` always on, no speculative, no MTMD, no ctx-shift. That matches the `serving-bench` bench shape we already validate.

## 22. Which parts should HPX own, which should remain llama.cpp-owned

| Concern                              | Upstream owner       | Proposed HPX-prototype owner |
| ------------------------------------ | -------------------- | ---------------------------- |
| Process / runtime lifecycle          | `main()` + signal h. | HPX (program-entry layer)    |
| Model load / unload                  | `server_context_impl::load_model` (calls `common_init_from_params`) | unchanged — load on engine startup |
| Engine main loop                     | `server_queue::start_loop` | HPX (an HPX-async loop or a single `hpx::thread`) |
| Task queue                           | `server_queue` (mutex+cv+deque) | HPX (concurrent queue or `hpx::lcos::local::channel`) |
| Response queue                       | `server_response`    | HPX (per-task `hpx::future`/`shared_future` over results) |
| Slot pool ownership                  | `std::vector<server_slot>` | HPX (pool of futures gating `seq_id` ownership) |
| `llama_context` access               | engine thread only    | engine thread only — HPX must not parallelize calls into it |
| `llama_decode` invocation            | engine thread, synchronous | engine thread, synchronous (opaque) |
| `llama_memory_seq_*` calls           | engine thread only    | engine thread only            |
| Sampler chain (`common_sampler_*`)   | engine thread, per slot | engine thread, per slot     |
| Tokenization                         | HTTP threads (cheap)  | any thread (it's reentrant)   |
| Chat template / JSON parsing         | HTTP threads          | any thread                    |
| Streaming text formatting (SSE)      | HTTP threads          | any thread                    |
| `server_prompt_cache`                | engine thread         | engine thread                 |
| Cancellation propagation             | request-side `should_stop` + `SERVER_TASK_TYPE_CANCEL` | HPX (`hpx::lcos::local::cancellation_token`?) plumbed into the same task-cancel path |
| HTTP serving                         | `cpp-httplib`         | unchanged for now             |

Rule of thumb: anything that touches `llama_context`, `llama_batch`, `llama_decode`, or `llama_memory_seq_*` is engine-thread-only and HPX must serialize access to it. Anything that's pure data (json, tokens, sampler outputs once produced) can fan out to any HPX worker.

---

## Implications for HPX

### What should HPX own

- The engine main loop (today `server_queue::start_loop`): make it an HPX-managed task that consumes a request channel and produces results.
- The slot pool: model each `seq_id` as an `hpx::future<slot_lease>` with RAII release. A request awaits a lease, holds it for its lifetime, drops it on completion or cancel.
- The request/response wiring: replace the two `std::deque`+mutex queues with HPX channels or shared futures. Each request becomes a `hpx::future<final_result>` plus a chunked `hpx::lcos::local::channel<partial_result>` for streaming.
- Cancellation: HPX cancellation tokens that propagate from the HTTP layer to the engine. The engine still only honors them at iteration boundaries — never mid-decode.
- Off-engine work: tokenization, JSON formatting, SSE serialization, prompt-cache load/save can run on HPX workers concurrently with the engine loop.

### What should llama.cpp own

- Everything inside `llama_context`: model weights, KV cache, attention, decode kernels.
- The `llama_batch` data layout, `llama_decode` semantics, and `llama_memory_seq_*` mutations.
- Sampler implementation (`common_sampler_*`).
- Tokenizer state (read-mostly; safe to call from any thread on the same `llama_model`).

We should not push HPX async into `llama_decode` itself — it is one synchronous compute call and must stay that way.

### What the simulator should model

- Slot lifecycle (`IDLE → PROCESSING_PROMPT → DONE_PROMPT → GENERATING → IDLE`).
- Request queue + deferred queue + LRU slot picking + slot release callback.
- The two-pass batch packing (decode rows first, then prompt rows when `cont_batching`).
- Per-iteration "row budget" capped at `n_batch`, decode chunked into `n_batch` views.
- Cancellation at iteration boundaries.
- Streaming partials produced inside the engine iteration.
- Approximate per-iteration latency as a function of total rows (calibrated from a real `serving-bench` run on the same TinyLlama shape).

### What the simulator should intentionally ignore

- Real KV mechanics, real KV reuse, SWA/recurrent paths, ctx-shift, checkpoints.
- Sampler internals, tokenizer correctness.
- LoRA, alora, MTMD, speculative decoding, n_cmpl>1.
- Prompt cache eviction (host-RAM).
- HTTP layer, JSON, chat template, MTMD, OAI/Anthropic compat.
- Embeddings, rerank, infill, slot save/restore.
- Sleeping mode, router mode.

### First correctness gates

Mirror the existing HPX-on slice (`local/baselines/comparison_hpx_vs_std/FACTS.md`):

1. With `n_slots=1, n_concurrent=1`, deterministic prompt, the simulator's emitted token sequence (mocked by request id, not real tokens) matches the reference order for trivially-ordered requests.
2. Slot lifecycle: each accepted request acquires a slot exactly once, releases it exactly once. No double-release. No slot left BUSY after all requests complete.
3. Cancellation: cancelling a request before the slot is assigned removes it from the deferred queue without ever entering a slot. Cancelling mid-generation transitions the slot to IDLE before the next iteration adds a row for it.
4. Continuous-batching switch:
   - with `cont_batching=true`, two slots can have rows in the same `update_slots` iteration (one decoding, one prefilling),
   - with `cont_batching=false`, prompt rows only appear in iterations where no slot is generating.
5. Backpressure: when `n_requests > n_slots`, deferred queue depth never exceeds `n_requests - n_slots`; every request eventually exits IDLE.

These mirror the upstream pytest signals (`test_completion_parallel_slots`, `test_cancel_request`, `test_completion_stream*`) but at the simulator level, not against a real model.

### First performance metrics (the simulator can produce)

Once correctness gates pass, the simulator can report:

- per-iteration row count distribution (mean, p50, p95, max),
- per-request: queue-wait time, prefill time, time-to-first-token, inter-token time, total latency,
- slot utilization: fraction of iterations with each slot busy,
- batch utilization: `batch.n_tokens / n_batch` distribution,
- effect of `cont_batching` on/off across the same arrival pattern,
- effect of `n_slots` and `n_batch` sweeps on tail latency and throughput.

These are the metrics we'd want a real HPX prototype to also report, so that HPX-vs-upstream is a same-units comparison.

---

## Proposed Phase 2 simulator scope (not implementing yet)

**Goal**: a discrete-event simulator, no real inference, that reproduces upstream `update_slots` scheduling on synthetic request traces.

**Inputs**:
- model parameters: `n_slots`, `n_batch`, `n_ubatch` (advisory only), `n_ctx`, `cont_batching` flag,
- per-row latency model: `latency(rows) = a + b*rows` calibrated from one `serving-bench` run (so a real HPX prototype can later be plugged in instead),
- request trace: list of `(t_arrive, n_prompt_tokens, n_predict_tokens, stream_bool)`.

**Components**:
- `Request`, `Slot`, `Scheduler::Queue`, `Engine::update_slots()`, `Clock`.
- One scheduler thread, no HPX yet — keep dependencies minimal so the simulator runs anywhere.
- Output: per-request latency table + per-iteration stats CSV, both writable to `local/sim/<run-id>/`.

**Out of scope for Phase 2**:
- Real `llama.cpp` linkage (deferred to Phase 3 when we wire HPX into a real engine).
- Speculative, multimodal, LoRA, n_cmpl>1.
- HTTP layer (drive the simulator from Python or a C++ test harness).
- Performance claims of any kind; this phase produces *expected behavior under a model*, not measured throughput.

**Acceptance for Phase 2**:
- The five correctness gates above all pass on synthetic traces.
- For a fixed synthetic trace, simulator outputs match a hand-calculated expected schedule for `n_slots∈{1,2,4}, cont_batching∈{on,off}, n_predict∈{1, 16, 128}`.
- A simple sweep over `n_slots` and `cont_batching` produces plots that match the qualitative claims in `tools/server/README-dev.md` §"Batching" (continuous batching reduces tail latency for short requests behind a long prefill).

Phase 3 (later, separate doc): replace the latency model with a real `llama_decode` call, keeping the simulator's scheduler intact, and rebuild as the HPX prototype.
