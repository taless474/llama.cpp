// HPX continuous-batching prototype — Cancel Slice 4: closeout of
// the cooperative-cancellation line. Adds reporting/metrics polish
// only — no behavioral change vs Cancel Slice 3.
//
// Cancel Slice 4 changes:
//   - Per-iter metrics block now reports completed/cancelled splits
//     correctly. The legacy `completed[budget=B]` line counted every
//     seq of that budget, which was misleading once cancellation
//     became real (budget-64 showed 33 even though 3 were cancelled).
//     The line now counts only `status == completed`, and a paired
//     `cancelled[budget=B]` line is emitted for budgets with any
//     cancelled seqs.
//   - `ttc_ms[budget=B]` is split into `ttc_ms_completed[budget=B]`
//     and (when present) `ttc_ms_cancelled[budget=B]`. Both remain
//     descriptive only and are not part of the determinism contract.
//   - Top-level metrics fields added (after engine_task_count):
//       completed_count
//       cancelled_count
//       decode_failures
//       wasted_decode_rows_after_cancel
//       residual_kv_empty
//   - Final emit is HPX_CB_CANCEL_STEP4: PASS / FAIL: <reason>.
// All Cancel Slice 2 / 3 correctness gates still apply unchanged.
//
// HPX continuous-batching prototype — Cancel Slice 3: result/status
// path polish on top of the Cancel Slice 2 cooperative-cancellation
// behavior.
//
// Cancel Slice 3 changes (no new behavior):
//   - Cancel-family trace events are aligned to the spec format
//     so they are grep-friendly and self-describing:
//       event=cancel_requested seq=<id> budget=<int> cancel_after=<int>
//       event=cancel_observed  seq=<id> iter=<int> n_decoded=<int>
//       event=cancel_kv_cleared seq=<id> pos_max_at_clear=<int> cross_talk_ok=1
//       event=cancel_future_fulfilled seq=<id> status=cancelled ttc_us=<int>
//   - Main prints an explicit per-iter status_summary line
//     (completed/cancelled/total) so the result split is obvious in
//     the capture without re-counting per-budget rows.
//   - Main adds an explicit cancelled-result gate
//     `n_decoded == n_decoded_at_cancel` (previously transitive
//     through both equalling --cancel-after).
//   - Engine emits seq_complete / kv_cleared / promise_fulfilled
//     ONLY on the completion path; the cancellation path emits
//     cancel_observed / cancel_kv_cleared / cancel_future_fulfilled
//     instead. Expected counts under the default plan:
//       request_admitted = 99
//       seq_prefilled = 99
//       seq_complete = 93
//       kv_cleared = 93
//       promise_fulfilled = 93
//       cancel_requested = 6
//       cancel_observed = 6
//       cancel_kv_cleared = 6
//       cancel_future_fulfilled = 6
//   - Final emit is HPX_CB_CANCEL_STEP3: PASS / FAIL: <reason>.
// All Cancel Slice 2 correctness gates still apply unchanged.
//
// HPX continuous-batching prototype — Cancel Slice 2: deterministic
// cooperative cancellation enabled inside the engine loop.
//
// Cancel Slice 2 turns on the data model that Cancel Slice 1
// plumbed:
//   - The plan from --cancel-plan is propagated into seq_state at
//     engine construction (cancel_after_decoded_tokens per seq).
//   - The engine task observes cancellation only at iteration
//     boundaries: once before the decode loop (post-prefill) and
//     once at the top of each decode iteration, BEFORE building
//     the active_idx for that iter. No mid-llama_decode
//     interruption.
//   - On observation the engine: marks cancel_observed[_iter]
//     and n_decoded_at_cancel; runs the existing
//     clear_and_check (KV remove + cross-talk verify); fulfills
//     the per-request promise once, with status=cancelled.
//   - Cancelled seqs do not appear in any subsequent decode batch
//     because they become seq.done=true via clear_and_check, and
//     the active_idx builder already skips done seqs. This makes
//     wasted_decode_rows_after_cancel structurally 0.
//   - Cancel-family trace events (cancel_requested,
//     cancel_observed, cancel_kv_cleared,
//     cancel_future_fulfilled) fire under LLAMA_HPX_CB_TRACE=1
//     using the same gating as Slice 4.
//   - --cancel-plan parser extended to accept seq_id 0
//     (previously rejected by parse_csv_int_list); seq_id 0 is
//     not used in the default plan but is now expressible.
// All non-cancelled Slice-3/4/5 correctness gates still hold;
// per-class hash anchors apply to NON-CANCELLED seqs only and
// the cancelled-run budget-64/256 hashes are NOT comparable to
// the all-99-running Slice-5 hashes (batch shape changes after
// iter 16 once the 6 cancelled seqs leave).
//
// HPX continuous-batching prototype — Slice 4: lifecycle traces +
// descriptive metrics on top of the Slice-3 promise/future engine.
//
// Slice 4 adds:
//   - env-gated trace events on stderr (LLAMA_HPX_CB_TRACE=1):
//       engine_start, request_admitted, seq_prefilled, decode_row,
//       seq_complete, kv_cleared, promise_fulfilled, engine_stop
//   - per-iteration descriptive metrics block on stdout:
//       wall_ms, decode_calls, update_iterations,
//       rows_per_batch (p50/p95/max), active_seqs_per_iter (p50/p95/max),
//       completed counts by budget,
//       time_to_completion_ms (mean / p95) by budget,
//       futures_created / promises_fulfilled / futures_completed,
//       engine_task_count
//
// Metrics are descriptive only; no comparison or speedup language.
// Both trace and metrics paths must not change correctness behavior or
// batch shape. With trace disabled (default), the trace path is one
// atomic load + early return per call site.
//
// All earlier Slice-3 invariants still hold:
//   - exactly one HPX engine task per repeat iteration;
//   - one hpx::promise<request_result> per seq, fulfilled only after
//     KV clear + cross-talk check pass;
//   - main validates correctness only from request_result snapshots
//     and never touches llama_context / llama_batch /
//     llama_decode / llama_memory_seq_* / llama_get_logits_ith.
//
// See docs/hpx/hpx_continuous_batching_prototype_design.md.

#include "common.h"
#include "llama.h"

#include "cli.h"
#include "engine.h"
#include "gate_emit.h"
#include "gate_validation.h"
#include "hpx_runtime.h"
#include "submitter.h"
#include "trace.h"
#include "types.h"

#include <hpx/hpx.hpp>
#include <hpx/lcos_local/channel.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

// Bring `gate_emit::*` into the gate's anonymous namespace so the
// fail/PASS sites in main resolve unqualified, matching the
// pre-extraction style. `trace::*` is always namespace-qualified.
using namespace gate_emit;

// ---- Enums, channel aliases, and POD types moved to types.h. ----------
// ---- Engine class moved to engine.h + engine.cpp. ---------------------
// ---- hpx_runtime namespace moved to hpx_runtime.h + hpx_runtime.cpp. --
// ---- CLI args / print_usage / parse_args moved to cli.h + cli.cpp. ----
// ---- run_scripted_submitter moved to submitter.cpp. -------------------
// ---- token_hash / percentile / mean_of and the per-repeat validation
//      harness moved to token_hash.h and gate_validation.h/.cpp. --------

}  // namespace

int main(int argc, char ** argv) {
    cli_args args;
    if (!parse_args(argc, argv, args)) {
        return 2;
    }

    trace::init();

    // ---- Start HPX runtime first; fail closed if unavailable. -----------
    if (!hpx_runtime::start_once(args.hpx_os_threads)) {
        emit_fail("hpx runtime start failed");
        return 1;
    }

    common_init();
    llama_backend_init();

    auto cleanup_llama = [&](llama_context * ctx, llama_model * model) {
        if (ctx)   llama_free(ctx);
        if (model) llama_model_free(model);
        llama_backend_free();
    };

    // ---- Load model + create one context. ------------------------------
    llama_model_params model_params = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(
        args.model_path.c_str(), model_params);
    if (model == nullptr) {
        emit_fail("model load failed");
        cleanup_llama(nullptr, nullptr);
        hpx_runtime::stop();
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx           = static_cast<uint32_t>(args.ctx_size);
    ctx_params.n_batch         = static_cast<uint32_t>(args.n_batch);
    ctx_params.n_seq_max       = static_cast<uint32_t>(args.n_seq_max);
    ctx_params.n_threads       = args.n_threads;
    ctx_params.n_threads_batch = args.n_threads;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        emit_fail("context create failed");
        cleanup_llama(nullptr, model);
        hpx_runtime::stop();
        return 1;
    }

    const uint32_t actual_n_ctx     = llama_n_ctx(ctx);
    const uint32_t actual_n_seq_max = llama_n_seq_max(ctx);
    const uint32_t actual_n_batch   = llama_n_batch(ctx);

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> prompt_tokens =
        common_tokenize(ctx, args.prompt, /*add_special=*/true,
                        /*parse_special=*/true);
    const int32_t n_prompt_tokens =
        static_cast<int32_t>(prompt_tokens.size());

    // Live Admission Slice 2: budgets are per-active-seq only; the
    // bound population is n_active. Slots [n_active, n_seqs) are
    // not-yet-bound and get no budget (Slice 3 will assign on
    // admission).
    std::vector<int32_t> budgets(static_cast<size_t>(args.n_active));
    {
        const size_t m = args.decode_budget_mix.size();
        for (int32_t i = 0; i < args.n_active; i++) {
            budgets[static_cast<size_t>(i)] =
                args.decode_budget_mix[static_cast<size_t>(i) % m];
        }
    }
    const int32_t max_budget =
        budgets.empty()
            ? 0
            : *std::max_element(budgets.begin(), budgets.end());

    // Live Admission Slice 3: build the waiting queue once at
    // construction time. Each waiting request_id starts at n_active
    // and runs through n_active + n_waiting - 1; uniform decode
    // budget per --waiting-budget. Engine takes a const-ref handle
    // and consumes the FIFO head at admission boundaries.
    //
    // Live Admission Slice 4: emit one request_queued trace per
    // waiting request, so the queued population is visible in trace
    // output BEFORE the engine starts. trace::event is gated and is
    // silent when LLAMA_HPX_CB_TRACE is unset.
    std::vector<waiting_request> waiting_queue;
    waiting_queue.reserve(static_cast<size_t>(args.n_waiting));
    for (int32_t i = 0; i < args.n_waiting; i++) {
        waiting_request w;
        w.request_id    = args.n_active + i;
        w.decode_budget = args.waiting_budget;
        w.src           = arrival_source::preloaded;
        // M1c: copy the existing shared prompt into each preloaded
        // waiter. The engine moves this vector into the bound slot's
        // seq_state::prompt_tokens at admission time, after which
        // the admitted-prefill loop reads it. Same prompt source as
        // M1b / M0 → batch is byte-identical.
        w.prompt_tokens = prompt_tokens;
        waiting_queue.push_back(std::move(w));
        // Live Admission Slice 6: tag the preloaded path explicitly so
        // grep on `arrival_source=` exhaustively partitions queued
        // requests across preloaded and external sources.
        trace::event("request_queued request=%d budget=%d queue_pos=%d "
                     "arrival_source=preloaded",
                     args.n_active + i, args.waiting_budget, i);
    }

    fprintf(stdout, "model_path:          %s\n",  args.model_path.c_str());
    fprintf(stdout, "prompt:              \"%s\"\n", args.prompt.c_str());
    fprintf(stdout, "requested ctx_size:  %d\n", args.ctx_size);
    fprintf(stdout, "requested n_seq_max: %d\n", args.n_seq_max);
    fprintf(stdout, "requested n_batch:   %d\n", args.n_batch);
    fprintf(stdout, "actual n_ctx:        %u\n", actual_n_ctx);
    fprintf(stdout, "actual n_seq_max:    %u\n", actual_n_seq_max);
    fprintf(stdout, "actual n_batch:      %u\n", actual_n_batch);
    fprintf(stdout, "n_vocab:             %d\n", n_vocab);
    fprintf(stdout, "prompt_tokens:       %d\n", n_prompt_tokens);
    fprintf(stdout, "n_seqs:              %d\n", args.n_seqs);
    fprintf(stdout, "n_active:            %d\n", args.n_active);
    fprintf(stdout, "n_waiting:           %d\n", args.n_waiting);
    fprintf(stdout, "waiting_budget:      %d\n", args.waiting_budget);
    fprintf(stdout, "n_external_arrivals: %d\n",
            args.n_external_arrivals);
    fprintf(stdout, "external_arrival_budget: %d\n",
            args.external_arrival_budget);
    fprintf(stdout, "external_release_iter:   %d\n",
            args.external_release_iter);
    fprintf(stdout, "decode_budget_mix:   [");
    for (size_t i = 0; i < args.decode_budget_mix.size(); i++) {
        fprintf(stdout, "%s%d", i == 0 ? "" : ", ",
                args.decode_budget_mix[i]);
    }
    fprintf(stdout, "]\n");
    fprintf(stdout, "hpx_os_threads:      %d\n", args.hpx_os_threads);
    fprintf(stdout, "repeat:              %d\n", args.repeat);
    fprintf(stdout, "reuse_completed:     %d\n",
            args.reuse_completed ? 1 : 0);
    fprintf(stdout, "stream_all:          %d\n",
            args.stream_all ? 1 : 0);
    fprintf(stdout, "trace_enabled:       %d\n",
            trace::enabled() ? 1 : 0);
    fflush(stdout);

    auto fail_with = [&](const std::string & reason) {
        emit_fail(reason.c_str());
        cleanup_llama(ctx, model);
        hpx_runtime::stop();
    };

    // ---- M1d: optional per-request prompt source ----------------------
    // When --request-prompts-file is unset, per_request_prompts stays
    // empty, max_prompt_tokens == n_prompt_tokens, every downstream
    // path falls back to the shared args.prompt tokenization, and the
    // run is byte-identical to M1c.
    //
    // When set, the file must contain exactly
    //     n_active + n_waiting + n_external_arrivals
    // lines, one prompt per line. Line i is the prompt for request_id
    // i. Each line is tokenized with the same common_tokenize call
    // used for args.prompt. max_prompt_tokens is taken across all
    // per-request prompts and used to size batch_capacity and the
    // context-fit gates below; the engine still binds prompt_tokens_
    // to the args.prompt tokenization (legacy borrow) but in file
    // mode no seq actually reads it — initial actives are seeded
    // from per_active_prompts (passed via engine_options) and
    // waiters/external arrivals carry their per-request prompt
    // directly. Sampling is still greedy/argmax everywhere; M1e
    // will add a sampling_config field.
    std::vector<std::vector<llama_token>> per_request_prompts;
    int32_t       max_prompt_tokens  = n_prompt_tokens;
    const bool    prompt_file_mode   = !args.request_prompts_file.empty();
    const int32_t expected_prompt_lines =
        args.n_active + args.n_waiting + args.n_external_arrivals;
    if (prompt_file_mode) {
        std::ifstream in(args.request_prompts_file);
        if (!in.is_open()) {
            char buf[512];
            std::snprintf(buf, sizeof(buf),
                "request-prompts-file: cannot open '%s'",
                args.request_prompts_file.c_str());
            fail_with(buf); return 1;
        }
        std::vector<std::string> lines;
        std::string line;
        while (std::getline(in, line)) {
            lines.push_back(std::move(line));
        }
        if (static_cast<int32_t>(lines.size()) != expected_prompt_lines) {
            char buf[512];
            std::snprintf(buf, sizeof(buf),
                "request-prompts-file: expected %d lines "
                "(n_active=%d + n_waiting=%d + n_external_arrivals=%d), "
                "got %zu",
                expected_prompt_lines, args.n_active,
                args.n_waiting, args.n_external_arrivals,
                lines.size());
            fail_with(buf); return 1;
        }
        per_request_prompts.reserve(
            static_cast<size_t>(expected_prompt_lines));
        max_prompt_tokens = 0;
        for (int32_t i = 0; i < expected_prompt_lines; i++) {
            std::vector<llama_token> toks = common_tokenize(
                ctx, lines[static_cast<size_t>(i)],
                /*add_special=*/true, /*parse_special=*/true);
            if (toks.empty()) {
                char buf[512];
                std::snprintf(buf, sizeof(buf),
                    "request-prompts-file: line %d tokenized to 0 tokens",
                    i);
                fail_with(buf); return 1;
            }
            const int32_t len = static_cast<int32_t>(toks.size());
            if (len > max_prompt_tokens) max_prompt_tokens = len;
            per_request_prompts.push_back(std::move(toks));
        }
        fprintf(stdout,
                "request_prompts_file: %s "
                "(lines=%d max_prompt_tokens=%d)\n",
                args.request_prompts_file.c_str(),
                expected_prompt_lines, max_prompt_tokens);
        fflush(stdout);
        // Overwrite the preloaded waiter prompts (built earlier from
        // the shared prompt) with their per-request file prompts.
        // Waiters occupy request_ids [n_active, n_active + n_waiting).
        for (int32_t i = 0; i < args.n_waiting; i++) {
            waiting_queue[static_cast<size_t>(i)].prompt_tokens =
                per_request_prompts[
                    static_cast<size_t>(args.n_active + i)];
        }
    }

    // Per-active prompts vector for the engine ctor. Empty (and
    // unused) in no-file mode; populated from the first n_active
    // entries of per_request_prompts in file mode. Pointer is
    // passed via engine_options; engine ctor falls back to the
    // shared prompt_tokens_ borrow when the pointer is null.
    std::vector<std::vector<llama_token>> per_active_prompts;
    if (prompt_file_mode) {
        per_active_prompts.reserve(static_cast<size_t>(args.n_active));
        for (int32_t i = 0; i < args.n_active; i++) {
            per_active_prompts.push_back(
                per_request_prompts[static_cast<size_t>(i)]);
        }
    }

    // ---- Cancellation plan (Cancel Slice 4: closeout — no new
    //      behavior, polished metrics/reporting). ----------------------
    // Parse and print the plan. Assert each plan seq's actual budget
    // matches the round-robin mix prediction so the design's smoke
    // shape (seqs 1,4,7 -> budget 64; seqs 2,5,8 -> budget 256) is
    // verified against the current budget vector. The plan IS
    // propagated into seq_state and the engine observes cancellation
    // cooperatively at iteration boundaries.
    fprintf(stdout, "cancellation:        observed=1 "
                    "(Cancel Slice 4: cooperative)\n");
    fprintf(stdout, "cancel_after:        %d\n", args.cancel_after);
    fprintf(stdout, "cancel_plan:         [");
    for (size_t i = 0; i < args.cancel_plan.size(); i++) {
        fprintf(stdout, "%s%d", i == 0 ? "" : ", ",
                args.cancel_plan[i]);
    }
    fprintf(stdout, "]%s\n",
            args.cancel_plan.empty() ? " (empty)" : "");
    if (!args.cancel_plan.empty()) {
        fprintf(stdout, "planned cancel seqs:\n");
        for (int32_t cs : args.cancel_plan) {
            if (cs < 0 || cs >= args.n_active) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "cancel_plan seq_id=%d outside [0, n_active=%d) "
                    "(cancellation must target a bound active seq)",
                    cs, args.n_active);
                fail_with(buf); return 1;
            }
            const int32_t actual_b   = budgets[static_cast<size_t>(cs)];
            const int32_t expected_b = args.decode_budget_mix[
                static_cast<size_t>(cs) % args.decode_budget_mix.size()];
            fprintf(stdout, "  seq=%d budget=%d\n", cs, actual_b);
            if (actual_b != expected_b) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "cancel_plan seq=%d: budget=%d != round-robin "
                    "expected %d (mix=[8,64,256])",
                    cs, actual_b, expected_b);
                fail_with(buf); return 1;
            }
        }
    }
    fflush(stdout);

    // ---- Structural prerequisites (Step 1 gates carried over). ---------
    if (actual_n_seq_max < static_cast<uint32_t>(args.n_seq_max)) {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "actual n_seq_max %u < requested n_seq_max %d",
            actual_n_seq_max, args.n_seq_max);
        fail_with(buf); return 1;
    }
    // M1d: context-fit and batch-capacity gates use max_prompt_tokens
    // (the maximum prompt length across all per-request prompts in
    // file mode, or n_prompt_tokens when no file is supplied). In
    // no-file mode max_prompt_tokens == n_prompt_tokens, so the
    // gate values and batch_capacity are byte-identical to M1c.
    if (actual_n_ctx < static_cast<uint32_t>(max_prompt_tokens + 256)) {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "actual n_ctx %u < prompt_tokens %d + 256",
            actual_n_ctx, max_prompt_tokens);
        fail_with(buf); return 1;
    }
    if (actual_n_batch <
        static_cast<uint32_t>(args.n_seqs * max_prompt_tokens)) {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "actual n_batch %u < n_seqs %d * prompt_tokens %d",
            actual_n_batch, args.n_seqs, max_prompt_tokens);
        fail_with(buf); return 1;
    }
    if (actual_n_ctx <
        static_cast<uint32_t>(max_prompt_tokens + max_budget)) {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "actual n_ctx %u < prompt_tokens %d + max_budget %d",
            actual_n_ctx, max_prompt_tokens, max_budget);
        fail_with(buf); return 1;
    }

    const int32_t batch_capacity =
        std::max<int32_t>(args.n_seqs * max_prompt_tokens, args.n_seqs);

    // Stage 6 (M0 extraction): cross-repeat snapshot state for the
    // validation harness. Constructed once and threaded through every
    // run_validation call. The validator stores r=0 results / streamed
    // tokens / admitted streams here and reads them for the r>0
    // determinism gate.
    validation_state vstate;

    for (int32_t r = 0; r < args.repeat; r++) {
        // Live Admission Slice 6: derive the release barrier set and
        // the max decode iter the engine must size release/ack slots
        // for. With a single release_iter K (the smoke shape), the set
        // is {K} and max_decode_iters_for_ctor = K. With 0 external
        // arrivals, the set is empty and the path is fully inert.
        std::set<int32_t> release_iter_set;
        if (args.n_external_arrivals > 0) {
            release_iter_set.insert(args.external_release_iter);
        }
        int32_t max_decode_iters_for_ctor = 0;
        for (int32_t k : release_iter_set) {
            if (k > max_decode_iters_for_ctor) {
                max_decode_iters_for_ctor = k;
            }
        }

        // One engine instance per iteration (own promises/futures set).
        engine_options eng_opts;
        eng_opts.ctx              = ctx;
        eng_opts.vocab            = vocab;
        eng_opts.n_vocab          = n_vocab;
        eng_opts.prompt_tokens    = &prompt_tokens;
        eng_opts.budgets          = budgets;
        eng_opts.batch_capacity   = batch_capacity;
        eng_opts.cancel_plan      = args.cancel_plan;
        eng_opts.cancel_after     = args.cancel_after;
        eng_opts.n_seq_max        = args.n_seqs;
        eng_opts.waiting_queue    = &waiting_queue;
        eng_opts.reuse_completed  = args.reuse_completed;
        eng_opts.release_iter_set = release_iter_set;
        eng_opts.max_decode_iters = max_decode_iters_for_ctor;
        eng_opts.stream_all       = args.stream_all;
        // M1d: in file mode, point the engine ctor at the per-active
        // prompt vector so initial seqs are seeded from their own
        // file-tokenized prompts. In no-file mode (pointer left
        // nullptr), the ctor falls back to the shared prompt_tokens_
        // borrow exactly as M1b/M1c did.
        eng_opts.per_active_prompt_tokens =
            prompt_file_mode ? &per_active_prompts : nullptr;
        engine eng(std::move(eng_opts));

        // Live Admission Slice 6: build scripted arrivals + per-arrival
        // promise/future pairs + release blocks. Per-arrival request_ids
        // start at n_active + n_waiting so they sit immediately above
        // any preloaded waiter. For the smoke shape all share a single
        // release_iter so we end up with exactly one block of 6.
        std::vector<scripted_arrival> scripted_arrivals;
        std::vector<hpx::future<request_result>> external_futs;
        std::vector<submitter_release_block> blocks;
        if (args.n_external_arrivals > 0) {
            scripted_arrivals.reserve(
                static_cast<size_t>(args.n_external_arrivals));
            for (int32_t i = 0; i < args.n_external_arrivals; i++) {
                scripted_arrival sa;
                sa.request_id    = args.n_active + args.n_waiting + i;
                sa.decode_budget = args.external_arrival_budget;
                sa.release_iter  = args.external_release_iter;
                // M1c: copy the existing shared prompt into each
                // scripted external arrival. M1d: in file mode,
                // use the per-request file prompt instead (indexed
                // by request_id). The submitter task moves the
                // chosen vector into the matching arrival_msg
                // before eng.submit().
                if (prompt_file_mode) {
                    sa.prompt_tokens = per_request_prompts[
                        static_cast<size_t>(sa.request_id)];
                } else {
                    sa.prompt_tokens = prompt_tokens;
                }
                scripted_arrivals.push_back(std::move(sa));
            }
            // M2c: pre-reserve external_futs so the submitter task
            // can push_back into it without reallocating. The vector
            // is populated inside the submitter task via
            // engine::submit_request().result; main reads it after
            // submitter_fut.get() joins, so no mutex is needed.
            external_futs.reserve(scripted_arrivals.size());
            // Group by release_iter (set is sorted; single entry in the
            // smoke shape). For each K, register a release handle with
            // the engine and pack matching script entries; per-arrival
            // promises are no longer pre-created here — the engine
            // owns them inside submit_request().
            for (int32_t K : release_iter_set) {
                submitter_release_block blk;
                blk.release_iter = K;
                try {
                    blk.handle = eng.register_external_release_iter(K);
                } catch (const std::exception & e) {
                    std::string msg =
                        std::string("register_external_release_iter(")
                        + std::to_string(K) + ") threw: " + e.what();
                    fail_with(msg); return 1;
                }
                for (size_t i = 0; i < scripted_arrivals.size(); i++) {
                    if (scripted_arrivals[i].release_iter != K) continue;
                    blk.arrivals.push_back(scripted_arrivals[i]);
                }
                blocks.push_back(std::move(blk));
            }
        }

        std::vector<hpx::future<request_result>> futs = eng.take_futures();
        const int32_t initial_futures =
            static_cast<int32_t>(futs.size());

        // Streaming Slice 8: collect per-seq stream receivers BEFORE
        // the engine task starts (mirrors take_futures discipline).
        // Empty when --stream-all is OFF. Receivers are in seq_id order
        // from the ctor; indexing by seq_id is direct.
        std::vector<token_stream_receiver> stream_receivers =
            eng.take_stream_receivers();

        // Live Admission Slice 6: spawn the scripted submitter as
        // exactly one HPX task BEFORE the engine task starts, so the
        // release barrier is already being awaited when the engine
        // reaches end of iter K. No-op when blocks is empty.
        hpx::future<void> submitter_fut;
        if (!blocks.empty()) {
            // M2c: capture &external_futs by reference so the
            // submitter task can push each submit_request handle's
            // result future as it submits. external_futs.reserve()
            // above bounds capacity so push_back never reallocates.
            submitter_fut = hpx::async(
                [&eng, &external_futs,
                 blks = std::move(blocks)]() mutable {
                    run_scripted_submitter(
                        eng, std::move(blks), external_futs);
                });
        }

        // Schedule engine::run() as exactly one HPX task; main thread
        // waits via wait_all on per-request futures + engine_fut.get().
        hpx::future<void> engine_fut = hpx::async([&eng]() {
            eng.run();
        });

        hpx::wait_all(futs);

        try {
            engine_fut.get();
        } catch (const std::exception & e) {
            std::string msg = std::string("engine task threw: ") + e.what();
            fail_with(msg); return 1;
        }

        // Live Admission Slice 6: join the submitter task. By this
        // point the engine has fired+observed every release/ack and
        // the submitter has set every ack — it should already be done.
        if (submitter_fut.valid()) {
            try {
                submitter_fut.get();
            } catch (const std::exception & e) {
                std::string msg =
                    std::string("submitter task threw: ") + e.what();
                fail_with(msg); return 1;
            }
        }

        // Streaming Slice 8: drain each stream receiver into a per-seq
        // token-id vector and record the observed close reason. The
        // engine task has already returned via engine_fut.get(), so
        // every event is already buffered in the channel; rx.get() with
        // launch::sync returns immediately. Hard rule: this loop only
        // touches int32_t token ids and the POD `token_stream_event`.
        // It must NOT call any llama_* API.
        std::vector<std::vector<int32_t>> streamed_tokens(
            static_cast<size_t>(args.n_active));
        std::vector<stream_close_reason>  streamed_close(
            static_cast<size_t>(args.n_active),
            stream_close_reason::completed);
        std::vector<bool>                 streamed_seen(
            static_cast<size_t>(args.n_active), false);
        if (args.stream_all) {
            if (static_cast<int32_t>(stream_receivers.size())
                != args.n_active) {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: stream_receivers.size()=%zu != n_active=%d",
                    r, stream_receivers.size(), args.n_active);
                fail_with(buf); return 1;
            }
            try {
                for (size_t s = 0; s < stream_receivers.size(); s++) {
                    auto & rx = stream_receivers[s];
                    for (;;) {
                        const token_stream_event ev =
                            rx.get(hpx::launch::sync);
                        if (ev.kind == stream_event_kind::closed) {
                            streamed_close[s] = ev.close_reason;
                            streamed_seen[s]  = true;
                            break;
                        }
                        streamed_tokens[s].push_back(ev.token_id);
                    }
                }
            } catch (const std::exception & e) {
                std::string msg =
                    std::string("stream channel drain threw: ") + e.what();
                fail_with(msg); return 1;
            }
        } else {
            if (!stream_receivers.empty()) {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: stream_receivers not empty with "
                    "--stream-all OFF (size=%zu)",
                    r, stream_receivers.size());
                fail_with(buf); return 1;
            }
        }

        const engine_result & er = eng.result();

        // Live Admission Slice 3: drain admitted-request futures and
        // append to the wait list. Engine has already fulfilled every
        // admitted promise before returning, so each appended future
        // is ready and the follow-on wait_all is a no-op.
        {
            auto admitted = eng.take_admitted_futures();
            futs.reserve(futs.size() + admitted.size());
            for (auto & f : admitted) {
                futs.push_back(std::move(f));
            }
            if (!admitted.empty()) {
                hpx::wait_all(futs);
            }
        }
        // Streaming Slice 3: drain per-admission stream handoffs.
        // Engine pushed each completion-freed preloaded-waiter bundle
        // inside admit_one under the same admitted_futures_mtx_
        // critical section that already guards admitted-future
        // handoff. Receivers are drained strictly after
        // engine_fut.get(), so rx.get(launch::sync) returns the
        // already-buffered events immediately. Keyed by request_id,
        // not seq_id, because admitted requests reuse a seq slot.
        // Hard rule: this loop only touches int32_t token ids and the
        // POD `token_stream_event` — it must NOT call any llama_* API.
        std::unordered_map<int32_t, std::vector<int32_t>>
            admitted_streamed_tokens;
        std::unordered_map<int32_t, stream_close_reason>
            admitted_streamed_close;
        std::unordered_map<int32_t, bool>
            admitted_streamed_seen;
        {
            auto handoffs = eng.take_admitted_stream_handoffs();
            if (!args.stream_all && !handoffs.empty()) {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: admitted_stream_handoffs not empty with "
                    "--stream-all OFF (size=%zu)",
                    r, handoffs.size());
                fail_with(buf); return 1;
            }
            if (args.stream_all) {
                try {
                    for (auto & h : handoffs) {
                        auto & toks =
                            admitted_streamed_tokens[h.request_id];
                        for (;;) {
                            const token_stream_event ev =
                                h.rx.get(hpx::launch::sync);
                            if (ev.kind == stream_event_kind::closed) {
                                admitted_streamed_close[h.request_id] =
                                    ev.close_reason;
                                admitted_streamed_seen[h.request_id]  =
                                    true;
                                break;
                            }
                            toks.push_back(ev.token_id);
                        }
                    }
                } catch (const std::exception & e) {
                    std::string msg =
                        std::string("admitted stream channel drain "
                                    "threw: ") + e.what();
                    fail_with(msg); return 1;
                }
            }
        }
        // Live Admission Slice 6: append external arrival futures. The
        // engine fulfills them on the same admission/decode path used
        // by preloaded waiters; main holds the futures directly here
        // (no admitted_futures_ push for external arrivals).
        if (!external_futs.empty()) {
            futs.reserve(futs.size() + external_futs.size());
            for (auto & f : external_futs) {
                futs.push_back(std::move(f));
            }
            hpx::wait_all(futs);
        }

        // Stage 6 (M0 extraction): delegate every per-repeat
        // correctness gate, the partition table / admit_step5/6/7
        // audit lines / residual_kv / status_summary / metrics
        // block, and the cross-repeat determinism check to the
        // validation harness. Main retains sole ownership of the
        // llama context/model and hpx_runtime teardown on the
        // failure path; the validator only emits the FAIL line and
        // returns false.
        if (!run_validation(args, r, n_prompt_tokens, budgets, er,
                            futs, initial_futures,
                            streamed_tokens, streamed_close,
                            streamed_seen,
                            admitted_streamed_tokens,
                            admitted_streamed_close,
                            admitted_streamed_seen,
                            vstate)) {
            cleanup_llama(ctx, model);
            hpx_runtime::stop();
            return 1;
        }
    }

    cleanup_llama(ctx, model);
    hpx_runtime::stop();

    emit_pass();
    return 0;
}
