// HPX continuous-batch gate — validation harness (Stage 6 extraction).
//
// Everything in this file moved verbatim from the per-repeat block of
// hpx-continuous-batch-gate.cpp:
//
//   - the engine-level structural gates (engine_task_count, decode
//     failures, residual KV, futures created vs expected),
//   - the per-result loop (non-admitted vs cancel_freed vs
//     completion_freed branches, slot-chain bookkeeping),
//   - the per-request stream gates and the coverage gates that ride on
//     them (Slice 3/4/5/6/7 multi-cycle),
//   - the engine-side stream counter cross-checks,
//   - the partitioned hash / done_iter / pos_max table and the
//     budget-class round-robin gate,
//   - the cancel/completed totals and wasted-decode-row aggregate,
//   - the Live Admission Slice 3-7 strict gates + admission counter
//     reconciliation,
//   - the residual_kv / status_summary / admit_step5/6/7 print lines,
//   - the descriptive metrics block, and
//   - the cross-repeat determinism check (results + streamed + admitted
//     streamed maps).
//
// No gate text, gate predicate, output line, or PASS/FAIL label is
// changed by this extraction. Behaviour is byte-identical to Stage 5
// across the six smoke runs documented in results.md, modulo timing
// fields (wall_ms, ttc_ms_*, admitted_ttc_ms[*]).
//
// Failure / cleanup ownership:
//   - `run_validation` reports gate failures by calling
//     `gate_emit::emit_fail(reason)` and returning false. It NEVER
//     touches the llama context, the model, or the HPX runtime — main
//     keeps sole ownership of `llama_free` / `llama_model_free` /
//     `llama_backend_free` and the matching `hpx_runtime::stop()` call.
//   - On PASS the function returns true; main runs the same teardown
//     sequence and then emits the terminal PASS line via
//     `gate_emit::emit_pass`.
//
// Determinism state (`validation_state`):
//   - Snapshots from the r == 0 iteration are stored in `vstate.first_*`
//     so r > 0 iterations can cross-check.
//   - Main constructs ONE `validation_state` outside the per-repeat
//     loop and passes it by mutable reference on every invocation.

#include "gate_validation.h"

#include "cli.h"
#include "gate_emit.h"
#include "token_hash.h"
#include "types.h"

#include "llama.h"

#include <hpx/hpx.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

// Mirrors the in-main `using namespace ...` directives so unqualified
// references (e.g. `k_canonical_budget_8`, `emit_fail`) resolve the
// same way they did inside main's anonymous namespace.
using namespace token_hash;
using namespace gate_emit;

// Linear-interp percentile over a copy of the input (validation-local).
double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    if (v.size() == 1) return v[0];
    if (p <= 0.0) return v.front();
    if (p >= 1.0) return v.back();
    const double idx  = p * static_cast<double>(v.size() - 1);
    const size_t lo   = static_cast<size_t>(idx);
    const size_t hi   = std::min(lo + 1, v.size() - 1);
    const double frac = idx - static_cast<double>(lo);
    return v[lo] + frac * (v[hi] - v[lo]);
}

double mean_of(const std::vector<double> & v) {
    if (v.empty()) return 0.0;
    double s = 0.0;
    for (double x : v) s += x;
    return s / static_cast<double>(v.size());
}

}  // namespace

bool run_validation(
    const cli_args &                                            args,
    int32_t                                                     repeat_index,
    int32_t                                                     n_prompt_tokens,
    const std::vector<int32_t> &                                budgets,
    const engine_result &                                       er,
    std::vector<hpx::future<request_result>> &                  futs,
    int32_t                                                     initial_futures,
    const std::vector<std::vector<int32_t>> &                   streamed_tokens,
    const std::vector<stream_close_reason> &                    streamed_close,
    const std::vector<bool> &                                   streamed_seen,
    const std::unordered_map<int32_t, std::vector<int32_t>> &   admitted_streamed_tokens,
    const std::unordered_map<int32_t, stream_close_reason> &    admitted_streamed_close,
    const std::unordered_map<int32_t, bool> &                   admitted_streamed_seen,
    validation_state &                                          vstate)
{
    // Local naming preserved from the in-main implementation. Keeping
    // `r` as the iteration variable means every snprintf format string,
    // every gate message, and every audit line is byte-identical to the
    // pre-extraction baseline.
    const int32_t r = repeat_index;

    // M1d: legacy_shape is true when --request-prompts-file is unset.
    // In that case every request carries a copy of the same shared
    // prompt, so the canonical-hash anchors and the
    // `n_prompt + budget - 2` pos_max equality gates apply and remain
    // enforced exactly as in M1c. In file mode prompts differ per
    // request, so per-partition unique-completed-hashes and the
    // pos_max equality gates are skipped — repeat determinism,
    // streamed_hash == rr.hash, residual-KV-empty, status summaries,
    // and the prompt-length-independent per-result gates remain in
    // force to catch the M1d invariants.
    const bool legacy_shape = args.request_prompts_file.empty();

    // Validation-local fail handler. Records the failure through the
    // same `emit_fail` channel main would have used, then propagates a
    // false return up to main. Main is responsible for tearing down the
    // llama context/model and the HPX runtime; this function does NOT
    // own that path. Mirrors the pre-extraction `fail_with(buf); return
    // 1;` two-statement pattern: each gate site now does
    // `fail_with(buf); return false;`.
    auto fail_with = [&](const std::string & reason) {
        emit_fail(reason.c_str());
    };

    const int32_t futures_created = static_cast<int32_t>(futs.size());
    const int32_t expected_total  = args.n_active + er.admitted_count;

    // Gate: engine_task_count == 1
    if (er.engine_task_count != 1) {
        char buf[128];
        std::snprintf(buf, sizeof(buf),
            "iter %d: engine_task_count=%d (expected 1)",
            r, er.engine_task_count);
        fail_with(buf); return false;
    }
    // Gate: engine HPX task completed successfully
    if (!er.ok) {
        std::string msg = "iter " + std::to_string(r)
            + ": engine error: " + er.error;
        fail_with(msg); return false;
    }
    // Gate: every llama_decode call returns 0
    if (er.decode_failures != 0) {
        char buf[128];
        std::snprintf(buf, sizeof(buf),
            "iter %d: decode_failures=%d (expected 0)",
            r, er.decode_failures);
        fail_with(buf); return false;
    }
    // Gate: residual KV at end is empty (engine-side)
    if (!er.residual_kv_ok) {
        std::string msg = "iter " + std::to_string(r)
            + ": residual_kv_error: " + er.residual_kv_error;
        fail_with(msg); return false;
    }

    // Slice 3 gates: counts include both initial actives and any
    // admitted-completed requests.
    if (initial_futures != args.n_active) {
        char buf[128];
        std::snprintf(buf, sizeof(buf),
            "iter %d: initial_futures=%d (expected n_active=%d)",
            r, initial_futures, args.n_active);
        fail_with(buf); return false;
    }
    if (futures_created != expected_total) {
        char buf[128];
        std::snprintf(buf, sizeof(buf),
            "iter %d: futures_created=%d (expected n_active+"
            "admitted_count=%d+%d=%d)",
            r, futures_created, args.n_active,
            er.admitted_count, expected_total);
        fail_with(buf); return false;
    }
    if (er.promises_fulfilled != expected_total) {
        char buf[128];
        std::snprintf(buf, sizeof(buf),
            "iter %d: promises_fulfilled=%d (expected %d)",
            r, er.promises_fulfilled, expected_total);
        fail_with(buf); return false;
    }

    // Gate: every request future is ready post-engine.
    int32_t futures_completed = 0;
    for (size_t i = 0; i < futs.size(); i++) {
        if (futs[i].is_ready()) {
            futures_completed++;
        } else {
            char buf[128];
            std::snprintf(buf, sizeof(buf),
                "iter %d: future %zu not ready after engine",
                r, i);
            fail_with(buf); return false;
        }
    }
    if (futures_completed != expected_total) {
        char buf[128];
        std::snprintf(buf, sizeof(buf),
            "iter %d: futures_completed=%d (expected %d)",
            r, futures_completed, expected_total);
        fail_with(buf); return false;
    }

    // Extract request_result snapshots from every future.
    std::vector<request_result> results;
    results.reserve(futs.size());
    try {
        for (auto & f : futs) {
            results.push_back(f.get());
        }
    } catch (const std::exception & e) {
        std::string msg =
            std::string("future.get() threw: ") + e.what();
        fail_with(msg); return false;
    }

    // Live Admission Slice 3: sort by request_id, not seq_id.
    // Cancelled originals and admitted reusers can share a seq_id;
    // request_id is unique per result.
    std::sort(results.begin(), results.end(),
              [](const request_result & a, const request_result & b) {
                  return a.request_id < b.request_id;
              });

    // Duplicate-request_id guard (Slice 3: replaces duplicate-seq_id
    // guard; seq_id can legitimately repeat across cancelled+admitted
    // pairs).
    {
        std::vector<int32_t> seen;
        seen.reserve(results.size());
        for (const auto & rr : results) {
            for (int32_t r_id : seen) {
                if (r_id == rr.request_id) {
                    char buf[128];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: duplicate request_id=%d in results",
                        r, rr.request_id);
                    fail_with(buf); return false;
                }
            }
            seen.push_back(rr.request_id);
        }
    }

    // Gate: kv_cleared == true on every fulfilled result.
    for (const auto & rr : results) {
        if (!rr.kv_cleared) {
            char buf[128];
            std::snprintf(buf, sizeof(buf),
                "iter %d: seq %d kv_cleared=false at fulfillment",
                r, rr.seq_id);
            fail_with(buf); return false;
        }
    }

    // ---- Per-result validation (Slice 3: partitioned by admission)
    // For a non-admitted result, request_id == seq_id (constructor
    // invariant) and the planned-cancel lookup uses request_id so
    // that it does not falsely match an admitted result that
    // happens to live on a cancel_plan slot.
    auto is_planned_cancel = [&](int32_t request_id) {
        for (int32_t cs : args.cancel_plan) {
            if (cs == request_id) return true;
        }
        return false;
    };

    const int32_t expected_n_decoded_at_cancel  = args.cancel_after;
    const int32_t expected_cancel_observed_iter = args.cancel_after;
    const int32_t expected_admitted_at_iter     = args.cancel_after + 1;

    // Live Admission Slice 5: expected admission iter for the
    // completion-freed path. With the demand-gated push policy
    // and round-robin {8,64,256} actives, the first wave of
    // completions happens at done_iter = min_active_budget - 1.
    // Admission consumes them at top of the next iter, so
    // admitted_at_iter == min_active_budget.
    const int32_t min_active_budget = budgets.empty()
        ? 0
        : *std::min_element(budgets.begin(), budgets.end());
    const int32_t expected_admitted_at_iter_completion =
        min_active_budget;

    // Streaming Slice 7: multi-cycle slot-reuse chain map. For
    // each seq_id slot, build an ordered list of occupants:
    // - index 0 is the original active occupant
    //   (admission_src == none, request_id == seq_id, seq_id in
    //    [0, n_active));
    // - indices 1..N are admitted occupants on that slot, sorted
    //   ascending by admitted_at_iter (deterministic for the
    //   smoke shapes since the engine drains cancel-/completion-
    //   freed pools in FIFO order at one admission iter at a
    //   time).
    // Used to replace the Slice-1-style
    // `previous_request_id == reused_seq_id` and
    // `admitted_at_iter == min_active_budget` checks in the
    // completion-freed branch with chain-aware checks
    // (Path α from the Slice 7 design): each admitted occupant's
    // previous_request_id equals the previous chain occupant's
    // request_id, and admitted_at_iter equals the previous
    // chain occupant's done_iter + 1. Built unconditionally
    // because the per-result gates fire regardless of
    // --stream-all.
    std::unordered_map<int32_t, std::vector<const request_result *>>
        slot_chain;
    for (const auto & rr : results) {
        if (rr.admission_src == admission_source::none
            && rr.request_id == rr.seq_id
            && rr.seq_id >= 0
            && rr.seq_id < args.n_active) {
            slot_chain[rr.seq_id].push_back(&rr);
        }
    }
    {
        std::vector<const request_result *> admitted_only;
        for (const auto & rr : results) {
            if (rr.admission_src != admission_source::none
                && rr.reused_seq_id >= 0
                && rr.reused_seq_id < args.n_active) {
                admitted_only.push_back(&rr);
            }
        }
        std::sort(admitted_only.begin(), admitted_only.end(),
            [](const request_result * a,
               const request_result * b) {
                return a->admitted_at_iter < b->admitted_at_iter;
            });
        for (const auto * p : admitted_only) {
            slot_chain[p->reused_seq_id].push_back(p);
        }
    }

    // Streaming Slice 7 multi-cycle smoke predicate. Used to
    // narrow gates that assume single-cycle reuse (the
    // reused_seq_id no-duplicate check and slice5_strict's
    // admitted_count<=|min_budget_slots| precondition). When
    // true, duplicate reused_seq_id values are expected and
    // the chain-walk gates above are the structural correctness
    // form. Mirrors the Slice 7 coverage gate's precondition
    // set (see below).
    const bool slice7_multicycle =
        args.stream_all
        && args.reuse_completed
        && args.cancel_plan.empty()
        && args.n_waiting > args.n_active
        && args.n_external_arrivals == 0;

    for (const auto & rr : results) {
        if (rr.admission_src == admission_source::none) {
            // Non-admitted (original active request).
            // request_id == seq_id is the invariant for this group.
            if (rr.request_id != rr.seq_id) {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: non-admitted request_id=%d != seq_id=%d",
                    r, rr.request_id, rr.seq_id);
                fail_with(buf); return false;
            }
            if (rr.admitted_at_iter != -1
             || rr.reused_seq_id != -1
             || rr.previous_request_id != -1) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: non-admitted req=%d carries admission "
                    "fields (admitted_at_iter=%d reused_seq_id=%d "
                    "previous_request_id=%d)",
                    r, rr.request_id, rr.admitted_at_iter,
                    rr.reused_seq_id, rr.previous_request_id);
                fail_with(buf); return false;
            }

            const bool planned = is_planned_cancel(rr.request_id);
            if (planned) {
                if (rr.status != request_status::cancelled) {
                    char buf[200];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: req %d status=%s (expected cancelled)",
                        r, rr.request_id, status_name(rr.status));
                    fail_with(buf); return false;
                }
                if (rr.n_decoded_at_cancel != expected_n_decoded_at_cancel) {
                    char buf[200];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: req %d n_decoded_at_cancel=%d "
                        "(expected %d)", r, rr.request_id,
                        rr.n_decoded_at_cancel,
                        expected_n_decoded_at_cancel);
                    fail_with(buf); return false;
                }
                if (rr.cancel_observed_iter != expected_cancel_observed_iter) {
                    char buf[200];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: req %d cancel_observed_iter=%d "
                        "(expected %d)", r, rr.request_id,
                        rr.cancel_observed_iter,
                        expected_cancel_observed_iter);
                    fail_with(buf); return false;
                }
                if (rr.n_decoded != expected_n_decoded_at_cancel) {
                    char buf[200];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: req %d n_decoded=%d "
                        "(expected %d for cancelled)", r, rr.request_id,
                        rr.n_decoded, expected_n_decoded_at_cancel);
                    fail_with(buf); return false;
                }
                if (rr.n_decoded != rr.n_decoded_at_cancel) {
                    char buf[200];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: req %d n_decoded=%d != "
                        "n_decoded_at_cancel=%d", r, rr.request_id,
                        rr.n_decoded, rr.n_decoded_at_cancel);
                    fail_with(buf); return false;
                }
                if (rr.n_decoded >= rr.decode_budget) {
                    char buf[200];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: req %d cancelled but n_decoded=%d "
                        ">= budget=%d", r, rr.request_id,
                        rr.n_decoded, rr.decode_budget);
                    fail_with(buf); return false;
                }
                // M1d: pos_max equality depends on per-seq prompt
                // length; only enforced in legacy shared-prompt mode.
                if (legacy_shape) {
                    const llama_pos expected_pos =
                        n_prompt_tokens + rr.n_decoded - 2;
                    if (rr.pos_max_at_clear != expected_pos) {
                        char buf[256];
                        std::snprintf(buf, sizeof(buf),
                            "iter %d: req %d cancelled pos_max_at_clear=%d "
                            "!= n_prompt+n_decoded-2=%d", r, rr.request_id,
                            rr.pos_max_at_clear,
                            static_cast<int>(expected_pos));
                        fail_with(buf); return false;
                    }
                }
            } else {
                if (rr.status != request_status::completed) {
                    char buf[200];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: req %d status=%s (expected completed)",
                        r, rr.request_id, status_name(rr.status));
                    fail_with(buf); return false;
                }
                if (rr.cancel_observed_iter != -1) {
                    char buf[200];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: req %d cancel_observed_iter=%d "
                        "(expected -1)", r, rr.request_id,
                        rr.cancel_observed_iter);
                    fail_with(buf); return false;
                }
                if (rr.n_decoded_at_cancel != -1) {
                    char buf[200];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: req %d n_decoded_at_cancel=%d "
                        "(expected -1)", r, rr.request_id,
                        rr.n_decoded_at_cancel);
                    fail_with(buf); return false;
                }
                if (rr.n_decoded != rr.decode_budget) {
                    char buf[256];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: req %d n_decoded %d != budget %d",
                        r, rr.request_id, rr.n_decoded, rr.decode_budget);
                    fail_with(buf); return false;
                }
            }
        } else if (rr.admission_src == admission_source::cancel_freed) {
            // Live Admission Slice 3: admitted-result gates.
            if (rr.status != request_status::completed) {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: admitted req %d status=%s "
                    "(expected completed)",
                    r, rr.request_id, status_name(rr.status));
                fail_with(buf); return false;
            }
            if (rr.admitted_at_iter != expected_admitted_at_iter) {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: admitted req %d admitted_at_iter=%d "
                    "(expected cancel_after+1=%d)",
                    r, rr.request_id, rr.admitted_at_iter,
                    expected_admitted_at_iter);
                fail_with(buf); return false;
            }
            if (rr.reused_seq_id != rr.seq_id) {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: admitted req %d reused_seq_id=%d != "
                    "seq_id=%d",
                    r, rr.request_id, rr.reused_seq_id, rr.seq_id);
                fail_with(buf); return false;
            }
            if (!is_planned_cancel(rr.reused_seq_id)) {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: admitted req %d reused_seq_id=%d not "
                    "in cancel_plan (cancel-freed-only rule)",
                    r, rr.request_id, rr.reused_seq_id);
                fail_with(buf); return false;
            }
            // For this prototype the original active set has
            // request_id == seq_id, so the prior owner of a cancel-
            // freed slot has request_id == seq_id == reused_seq_id.
            if (rr.previous_request_id != rr.reused_seq_id) {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: admitted req %d previous_request_id=%d "
                    "!= reused_seq_id=%d",
                    r, rr.request_id, rr.previous_request_id,
                    rr.reused_seq_id);
                fail_with(buf); return false;
            }
            if (rr.n_decoded != rr.decode_budget) {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: admitted req %d n_decoded=%d != "
                    "budget=%d", r, rr.request_id,
                    rr.n_decoded, rr.decode_budget);
                fail_with(buf); return false;
            }
            const int32_t expected_done_iter_admit =
                rr.admitted_at_iter + rr.decode_budget - 1;
            if (rr.done_iter != expected_done_iter_admit) {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: admitted req %d done_iter=%d "
                    "(expected admitted_at_iter+budget-1=%d)",
                    r, rr.request_id, rr.done_iter,
                    expected_done_iter_admit);
                fail_with(buf); return false;
            }
            // M1d: pos_max equality depends on per-seq prompt length;
            // only enforced in legacy shared-prompt mode.
            if (legacy_shape) {
                const llama_pos expected_pos_admit =
                    n_prompt_tokens + rr.decode_budget - 2;
                if (rr.pos_max_at_clear != expected_pos_admit) {
                    char buf[200];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: admitted req %d pos_max_at_clear=%d "
                        "(expected n_prompt+budget-2=%d)",
                        r, rr.request_id, rr.pos_max_at_clear,
                        static_cast<int>(expected_pos_admit));
                    fail_with(buf); return false;
                }
            }
            if (rr.cancel_observed_iter != -1
             || rr.n_decoded_at_cancel != -1) {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: admitted req %d carries cancel fields "
                    "(observed_iter=%d n_decoded_at_cancel=%d)",
                    r, rr.request_id, rr.cancel_observed_iter,
                    rr.n_decoded_at_cancel);
                fail_with(buf); return false;
            }
        } else if (rr.admission_src == admission_source::completion_freed) {
            // Live Admission Slice 5: completion-freed admitted
            // result gates. The slot was reused after its prior
            // owner finished naturally, so cancel-family fields
            // must be defaults and is_planned_cancel must NOT
            // hold for reused_seq_id (cross-source confusion check).
            if (!args.reuse_completed) {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: completion_freed req %d but "
                    "--reuse-completed is OFF",
                    r, rr.request_id);
                fail_with(buf); return false;
            }
            if (rr.status != request_status::completed) {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: completion_freed req %d status=%s "
                    "(expected completed)",
                    r, rr.request_id, status_name(rr.status));
                fail_with(buf); return false;
            }
            if (rr.reused_seq_id != rr.seq_id) {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: completion_freed req %d "
                    "reused_seq_id=%d != seq_id=%d",
                    r, rr.request_id, rr.reused_seq_id, rr.seq_id);
                fail_with(buf); return false;
            }
            if (is_planned_cancel(rr.reused_seq_id)) {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: completion_freed req %d "
                    "reused_seq_id=%d is in cancel_plan "
                    "(source confusion)",
                    r, rr.request_id, rr.reused_seq_id);
                fail_with(buf); return false;
            }
            // Slice 7: chain-aware previous_request_id +
            // admitted_at_iter check. For a multi-cycle slot
            // chain [O_0 (original), O_1, O_2, ...], each
            // admitted occupant O_k must satisfy:
            //   O_k.previous_request_id == O_{k-1}.request_id
            //   O_k.admitted_at_iter    == O_{k-1}.done_iter + 1
            // For k == 1 this collapses to the prior Slice-1
            // invariant (previous_request_id == reused_seq_id
            // because original.request_id == original.seq_id;
            // admitted_at_iter == original.done_iter + 1 ==
            // min_active_budget for the prior single-cycle
            // smoke shapes). For k >= 2 (Slice 7 multi-cycle)
            // the prior occupant is itself an admitted request,
            // so the simple equality and the
            // expected_admitted_at_iter_completion equality
            // would both fail; the chain form is the general
            // structural invariant.
            {
                auto it_chain = slot_chain.find(rr.reused_seq_id);
                if (it_chain == slot_chain.end()
                    || it_chain->second.empty()) {
                    char buf[200];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: completion_freed req %d "
                        "reused_seq_id=%d has no slot-chain "
                        "entry",
                        r, rr.request_id, rr.reused_seq_id);
                    fail_with(buf); return false;
                }
                const auto & chain = it_chain->second;
                size_t k = chain.size();
                for (size_t j = 0; j < chain.size(); j++) {
                    if (chain[j]->request_id == rr.request_id) {
                        k = j;
                        break;
                    }
                }
                if (k == 0 || k >= chain.size()) {
                    char buf[256];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: completion_freed req %d not "
                        "found as admitted occupant in "
                        "slot-chain[seq=%d] (k=%zu, "
                        "chain.size=%zu)",
                        r, rr.request_id, rr.reused_seq_id,
                        k, chain.size());
                    fail_with(buf); return false;
                }
                const request_result * prev = chain[k - 1];
                if (rr.previous_request_id != prev->request_id) {
                    char buf[256];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: completion_freed req %d "
                        "previous_request_id=%d != prior chain "
                        "occupant request_id=%d (k=%zu)",
                        r, rr.request_id, rr.previous_request_id,
                        prev->request_id, k);
                    fail_with(buf); return false;
                }
                if (rr.admitted_at_iter
                    != prev->done_iter + 1) {
                    char buf[256];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: completion_freed req %d "
                        "admitted_at_iter=%d != prior chain "
                        "occupant req %d done_iter+1=%d "
                        "(k=%zu)",
                        r, rr.request_id, rr.admitted_at_iter,
                        prev->request_id, prev->done_iter + 1,
                        k);
                    fail_with(buf); return false;
                }
            }
            if (rr.n_decoded != rr.decode_budget) {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: completion_freed req %d n_decoded=%d"
                    " != budget=%d",
                    r, rr.request_id, rr.n_decoded,
                    rr.decode_budget);
                fail_with(buf); return false;
            }
            const int32_t expected_done_iter_admit =
                rr.admitted_at_iter + rr.decode_budget - 1;
            if (rr.done_iter != expected_done_iter_admit) {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: completion_freed req %d done_iter=%d"
                    " (expected admitted_at_iter+budget-1=%d)",
                    r, rr.request_id, rr.done_iter,
                    expected_done_iter_admit);
                fail_with(buf); return false;
            }
            // M1d: pos_max equality depends on per-seq prompt length;
            // only enforced in legacy shared-prompt mode.
            if (legacy_shape) {
                const llama_pos expected_pos_admit =
                    n_prompt_tokens + rr.decode_budget - 2;
                if (rr.pos_max_at_clear != expected_pos_admit) {
                    char buf[200];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: completion_freed req %d "
                        "pos_max_at_clear=%d (expected "
                        "n_prompt+budget-2=%d)",
                        r, rr.request_id, rr.pos_max_at_clear,
                        static_cast<int>(expected_pos_admit));
                    fail_with(buf); return false;
                }
            }
            if (rr.cancel_observed_iter != -1
             || rr.n_decoded_at_cancel != -1) {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: completion_freed req %d carries "
                    "cancel fields (observed_iter=%d "
                    "n_decoded_at_cancel=%d)",
                    r, rr.request_id, rr.cancel_observed_iter,
                    rr.n_decoded_at_cancel);
                fail_with(buf); return false;
            }
        } else {
            char buf[200];
            std::snprintf(buf, sizeof(buf),
                "iter %d: req %d unknown admission_source=%s",
                r, rr.request_id,
                admission_source_name(rr.admission_src));
            fail_with(buf); return false;
        }
    }

    // ---- Streaming Slice 3: per-request stream gates --------------
    // Slice 3 generalizes the Slice 1/2 gates to range over the
    // UNION of original-active streamed requests (keyed by seq_id)
    // and completion-freed admitted streamed requests (keyed by
    // request_id). Status-aware close-reason / token-count / hash
    // gates from Slice 2 apply uniformly to both. Slice 3 adds:
    //   - the admitted request's stream channel closed independently
    //     of the previous slot occupant's stream (no inheritance);
    //   - at least one completion-freed admitted streamed request is
    //     witnessed when --stream-all AND --reuse-completed are on
    //     AND a waiting queue exists, so misconfigured smoke shapes
    //     fail-closed instead of silently proving nothing.
    // With --stream-all OFF every stream counter must be zero and no
    // streamed_seen flag may be true.
    if (args.stream_all) {
        int32_t streamed_admitted_completion_freed_count = 0;
        int32_t streamed_admitted_cancel_freed_count     = 0;
        int32_t streamed_admitted_external_arrival_count = 0;
        int32_t streamed_admitted_external_cancel_freed_count = 0;
        for (const auto & rr : results) {
            // Original-active streamed request: admission_src ==
            // none AND request_id == seq_id AND seq_id in
            // [0, n_active). Drained through the seq_id-indexed
            // stream_receivers vector (Slice 1 wiring).
            const bool is_original_active =
                   rr.admission_src == admission_source::none
                && rr.request_id == rr.seq_id
                && rr.seq_id >= 0
                && rr.seq_id < args.n_active;
            // Admitted streamed request: completion-freed (Slice 3
            // / Slice 5 scopes) OR cancel-freed (Slice 4 / Slice 6
            // scopes). Drained through the admitted-stream handoff
            // bundles, keyed by request_id. arrival_src is
            // orthogonal: preloaded admissions (Slices 3/4) and
            // external admissions (Slices 5/6) both reach this
            // point through the same handoff path.
            const bool is_admitted_streamed =
                   rr.admission_src == admission_source::completion_freed
                || rr.admission_src == admission_source::cancel_freed;
            if (!is_original_active && !is_admitted_streamed) continue;

            size_t                       streamed_count = 0;
            const int32_t *              streamed_data  = nullptr;
            stream_close_reason          streamed_reason =
                stream_close_reason::completed;
            bool                         streamed_seen_flag = false;

            if (is_original_active) {
                const size_t s = static_cast<size_t>(rr.seq_id);
                streamed_count     = streamed_tokens[s].size();
                streamed_data      = streamed_tokens[s].data();
                streamed_reason    = streamed_close[s];
                streamed_seen_flag = streamed_seen[s];
            } else {
                auto it_seen =
                    admitted_streamed_seen.find(rr.request_id);
                streamed_seen_flag =
                    (it_seen != admitted_streamed_seen.end())
                    && it_seen->second;
                if (streamed_seen_flag) {
                    const auto & toks =
                        admitted_streamed_tokens.at(rr.request_id);
                    streamed_count  = toks.size();
                    streamed_data   = toks.data();
                    streamed_reason =
                        admitted_streamed_close.at(rr.request_id);
                }
                if (rr.admission_src
                    == admission_source::completion_freed) {
                    streamed_admitted_completion_freed_count++;
                } else if (rr.admission_src
                    == admission_source::cancel_freed) {
                    streamed_admitted_cancel_freed_count++;
                }
                if (rr.arrival_src == arrival_source::external) {
                    streamed_admitted_external_arrival_count++;
                    if (rr.admission_src
                        == admission_source::cancel_freed) {
                        streamed_admitted_external_cancel_freed_count++;
                    }
                }
            }

            if (!streamed_seen_flag) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: req %d (seq %d, admission_src=%s) "
                    "stream did not close (no terminal event "
                    "observed)", r, rr.request_id, rr.seq_id,
                    admission_source_name(rr.admission_src));
                fail_with(buf); return false;
            }
            const stream_close_reason expected_close =
                (rr.status == request_status::cancelled)
                    ? stream_close_reason::cancelled
                    : stream_close_reason::completed;
            if (streamed_reason != expected_close) {
                char buf[320];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: req %d (seq %d, admission_src=%s) "
                    "stream close_reason=%s (expected %s, "
                    "rr.status=%s)", r, rr.request_id, rr.seq_id,
                    admission_source_name(rr.admission_src),
                    stream_close_reason_name(streamed_reason),
                    stream_close_reason_name(expected_close),
                    status_name(rr.status));
                fail_with(buf); return false;
            }
            if (static_cast<int32_t>(streamed_count)
                != rr.n_decoded) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: req %d (seq %d, admission_src=%s) "
                    "streamed=%zu != n_decoded=%d",
                    r, rr.request_id, rr.seq_id,
                    admission_source_name(rr.admission_src),
                    streamed_count, rr.n_decoded);
                fail_with(buf); return false;
            }
            if (rr.status == request_status::cancelled) {
                if (static_cast<int32_t>(streamed_count)
                    != rr.n_decoded_at_cancel) {
                    char buf[320];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: req %d (seq %d, admission_src=%s) "
                        "streamed=%zu != n_decoded_at_cancel=%d",
                        r, rr.request_id, rr.seq_id,
                        admission_source_name(rr.admission_src),
                        streamed_count, rr.n_decoded_at_cancel);
                    fail_with(buf); return false;
                }
            }
            uint64_t streamed_hash =
                (streamed_count == 0)
                    ? k_token_hash_empty
                    : k_token_hash_init;
            for (size_t i = 0; i < streamed_count; i++) {
                streamed_hash =
                    fold_token_hash(streamed_hash, streamed_data[i]);
            }
            if (streamed_hash != rr.hash) {
                char buf[320];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: req %d (seq %d, admission_src=%s) "
                    "streamed_hash=0x%016llx != rr.hash=0x%016llx",
                    r, rr.request_id, rr.seq_id,
                    admission_source_name(rr.admission_src),
                    static_cast<unsigned long long>(streamed_hash),
                    static_cast<unsigned long long>(rr.hash));
                fail_with(buf); return false;
            }

            // Slice 3 admitted-only independence gate: the admitted
            // request's streamed token vector must NOT equal the
            // previous occupant's streamed token vector. Fresh
            // channel objects are constructed per admission, so
            // structural independence is by construction; this gate
            // produces explicit evidence by comparing the drained
            // vectors. In single-cycle smokes (Slices 3-6) the
            // sizes differ by design (e.g., 8 vs 16), so equality
            // is impossible; in shapes where sizes match by
            // accident, content still differs because the admitted
            // request decodes from a fresh KV-cleared slot with a
            // different batch-shape history.
            //
            // Streaming Slice 7 (multi-cycle, Path α): when the
            // smoke uses uniform budgets and the same prompt under
            // greedy decoding, the cycle-1 admitted (B) and the
            // original (A) produce byte-identical streamed token
            // vectors — that is expected, NOT a stream-rebind
            // failure. Independence is structural under Path α:
            // a fresh hpx::lcos::local::channel<token_stream_event>
            // is constructed per admission, the per-slot
            // stream_tokens_emitted counter is reset, the KV-empty
            // assertion in admit_one fires per cycle, and the
            // chain-walk gates above enforce the request_id +
            // admitted_at_iter chain. Skip this vector-inequality
            // gate when slice7_multicycle is true.
            if (is_admitted_streamed && !slice7_multicycle) {
                if (rr.previous_request_id < 0) {
                    char buf[256];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: req %d admission_src=%s has "
                        "previous_request_id=%d (expected >= 0)",
                        r, rr.request_id,
                        admission_source_name(rr.admission_src),
                        rr.previous_request_id);
                    fail_with(buf); return false;
                }
                if (rr.seq_id < 0
                    || rr.seq_id >= args.n_active) {
                    char buf[256];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: req %d admission_src=%s seq_id=%d "
                        "out of original-active [0, n_active=%d)",
                        r, rr.request_id,
                        admission_source_name(rr.admission_src),
                        rr.seq_id, args.n_active);
                    fail_with(buf); return false;
                }
                const size_t prev_s =
                    static_cast<size_t>(rr.seq_id);
                const auto & prev_toks = streamed_tokens[prev_s];
                const auto & admitted_toks =
                    admitted_streamed_tokens.at(rr.request_id);
                if (admitted_toks == prev_toks) {
                    char buf[320];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: req %d (seq %d) admitted "
                        "streamed token vector equals previous "
                        "occupant (req %d) — stream rebind did "
                        "not produce an independent channel",
                        r, rr.request_id, rr.seq_id,
                        rr.previous_request_id);
                    fail_with(buf); return false;
                }
            }
        }

        // Slice 3 coverage gate: when streaming is on and
        // --reuse-completed is on and there is a non-empty waiting
        // queue, at least one completion-freed admitted streamed
        // request must have been witnessed. Catches misconfigured
        // smoke shapes that do not actually fire the path. Gated
        // on the conjunction so the Slice 7 stream-off regression
        // and the Slice 1/2 smokes (which have no waiting queue
        // or no --reuse-completed) remain valid.
        if (args.reuse_completed && args.n_waiting > 0
            && streamed_admitted_completion_freed_count <= 0) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "iter %d: --stream-all && --reuse-completed && "
                "--n-waiting=%d > 0 but no streamed admitted "
                "completion_freed request observed",
                r, args.n_waiting);
            fail_with(buf); return false;
        }

        // Slice 4 coverage gate: when streaming is on and a
        // cancel plan is configured and there is a non-empty
        // waiting queue, at least one cancel-freed admitted
        // streamed request must have been witnessed. Catches
        // misconfigured smoke shapes that do not actually fire
        // the cancel-freed admitted streaming path. Gated on the
        // conjunction so the Slice 7 stream-off regression and
        // the Slice 1/2/3 smokes (which either disable streaming
        // or have no cancel plan or no waiting queue) remain
        // valid.
        if (!args.cancel_plan.empty() && args.n_waiting > 0
            && streamed_admitted_cancel_freed_count <= 0) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "iter %d: --stream-all && cancel_plan non-empty "
                "&& --n-waiting=%d > 0 but no streamed admitted "
                "cancel_freed request observed",
                r, args.n_waiting);
            fail_with(buf); return false;
        }

        // Slice 5 coverage gate: when streaming is on and there
        // are external arrivals and --reuse-completed is enabled
        // and the cancel plan is empty, at least one streamed
        // external-arrival admitted request (on the
        // completion-freed admission path) must have been
        // witnessed. Catches misconfigured smoke shapes that do
        // not actually fire the external-arrival admitted
        // streaming path. Scoped to the Slice 5 narrow
        // (completion-freed × external) shape.
        if (args.n_external_arrivals > 0 && args.reuse_completed
            && args.cancel_plan.empty()
            && streamed_admitted_external_arrival_count <= 0) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "iter %d: --stream-all && "
                "--n-external-arrivals=%d > 0 && "
                "--reuse-completed && cancel_plan empty but no "
                "streamed admitted external-arrival request "
                "observed", r, args.n_external_arrivals);
            fail_with(buf); return false;
        }

        // Slice 6 coverage gate: when streaming is on and there
        // are external arrivals and a non-empty cancel_plan,
        // at least one streamed external-arrival admitted request
        // on the cancel-freed admission path must have been
        // witnessed. Catches misconfigured smoke shapes that do
        // not actually fire the external × cancel_freed admitted
        // streaming path that Slice 6 introduces. Lives inside
        // the enclosing `if (args.stream_all)` block above, so
        // stream-off regression runs (--stream-all OFF) bypass
        // this gate entirely.
        if (args.n_external_arrivals > 0 && !args.cancel_plan.empty()
            && streamed_admitted_external_cancel_freed_count <= 0) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "iter %d: --stream-all && "
                "--n-external-arrivals=%d > 0 && cancel_plan "
                "non-empty but no streamed admitted external+"
                "cancel_freed request observed",
                r, args.n_external_arrivals);
            fail_with(buf); return false;
        }

        // Slice 7 multi-cycle coverage gate. When the smoke is
        // shaped for multi-cycle slot reuse (more preloaded
        // waiters than original active slots, --reuse-completed
        // on, no cancel plan, no external arrivals), require:
        //   1. at least one slot S with >= 2 admitted occupants
        //      in slot_chain;
        //   2. the chain walk for that slot completes successfully
        //      (every admitted occupant is status=completed,
        //       has streamed close=completed, streamed token
        //       count == rr.n_decoded, and streamed_hash ==
        //       rr.hash). The chain link/timing checks were
        //       already enforced per occupant in the
        //       completion_freed branch of the per-result loop;
        //       this gate adds the existence assertion + the
        //       per-occupant streaming-side checks (the
        //       chain-walk gates above are admission-side).
        // Independence is structural under Path α: a fresh
        // hpx::lcos::local::channel<token_stream_event> is
        // bound at each admission, the per-slot
        // stream_tokens_emitted counter is reset, and the
        // KV-empty assertion in admit_one fires per cycle. With
        // uniform budgets and the same prompt under greedy
        // decoding, the streamed token vectors and hashes are
        // expected to be identical across cycles — vector
        // equality is NOT a failure mode for this gate. Lives
        // inside the enclosing `if (args.stream_all)` block so
        // stream-off regression is unaffected.
        if (slice7_multicycle) {
            int32_t slots_with_chain = 0;
            int32_t admitted_in_chains = 0;
            for (const auto & kv : slot_chain) {
                if (kv.second.size() < 2) continue;
                slots_with_chain++;
                // Skip index 0 (original); validate admitted
                // occupants in chain order.
                for (size_t k = 1; k < kv.second.size(); k++) {
                    const request_result * occ = kv.second[k];
                    admitted_in_chains++;
                    if (occ->status != request_status::completed) {
                        char buf[256];
                        std::snprintf(buf, sizeof(buf),
                            "iter %d: slice7_multicycle: "
                            "chain[seq=%d][k=%zu] req %d "
                            "status=%s (expected completed)",
                            r, kv.first, k, occ->request_id,
                            status_name(occ->status));
                        fail_with(buf); return false;
                    }
                    auto it_seen =
                        admitted_streamed_seen.find(
                            occ->request_id);
                    const bool seen =
                        (it_seen != admitted_streamed_seen.end())
                        && it_seen->second;
                    if (!seen) {
                        char buf[256];
                        std::snprintf(buf, sizeof(buf),
                            "iter %d: slice7_multicycle: "
                            "chain[seq=%d][k=%zu] req %d "
                            "stream not seen / not closed",
                            r, kv.first, k, occ->request_id);
                        fail_with(buf); return false;
                    }
                    const auto it_close =
                        admitted_streamed_close.find(
                            occ->request_id);
                    const stream_close_reason occ_close =
                        (it_close != admitted_streamed_close.end())
                            ? it_close->second
                            : stream_close_reason::completed;
                    if (occ_close
                        != stream_close_reason::completed) {
                        char buf[256];
                        std::snprintf(buf, sizeof(buf),
                            "iter %d: slice7_multicycle: "
                            "chain[seq=%d][k=%zu] req %d "
                            "streamed close=%s (expected "
                            "completed)",
                            r, kv.first, k, occ->request_id,
                            stream_close_reason_name(occ_close));
                        fail_with(buf); return false;
                    }
                    const auto it_toks =
                        admitted_streamed_tokens.find(
                            occ->request_id);
                    if (it_toks
                        == admitted_streamed_tokens.end()) {
                        char buf[256];
                        std::snprintf(buf, sizeof(buf),
                            "iter %d: slice7_multicycle: "
                            "chain[seq=%d][k=%zu] req %d "
                            "no admitted_streamed_tokens entry",
                            r, kv.first, k, occ->request_id);
                        fail_with(buf); return false;
                    }
                    const auto & toks = it_toks->second;
                    if (static_cast<int32_t>(toks.size())
                        != occ->n_decoded) {
                        char buf[256];
                        std::snprintf(buf, sizeof(buf),
                            "iter %d: slice7_multicycle: "
                            "chain[seq=%d][k=%zu] req %d "
                            "streamed=%zu != n_decoded=%d",
                            r, kv.first, k, occ->request_id,
                            toks.size(), occ->n_decoded);
                        fail_with(buf); return false;
                    }
                    uint64_t streamed_hash_chain =
                        (toks.empty())
                            ? k_token_hash_empty
                            : k_token_hash_init;
                    for (int32_t t : toks) {
                        streamed_hash_chain =
                            fold_token_hash(
                                streamed_hash_chain, t);
                    }
                    if (streamed_hash_chain != occ->hash) {
                        char buf[320];
                        std::snprintf(buf, sizeof(buf),
                            "iter %d: slice7_multicycle: "
                            "chain[seq=%d][k=%zu] req %d "
                            "streamed_hash=0x%016llx != "
                            "rr.hash=0x%016llx",
                            r, kv.first, k, occ->request_id,
                            static_cast<unsigned long long>(
                                streamed_hash_chain),
                            static_cast<unsigned long long>(
                                occ->hash));
                        fail_with(buf); return false;
                    }
                }
            }
            if (slots_with_chain == 0) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: slice7_multicycle predicate true "
                    "but no slot has >= 2 admitted occupants "
                    "(no multi-cycle reuse witnessed)", r);
                fail_with(buf); return false;
            }
            if (admitted_in_chains <= 0) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: slice7_multicycle: 0 admitted "
                    "occupants across chains", r);
                fail_with(buf); return false;
            }
        }

        // Engine-side stream counter gates (Slice 3/4 expected-
        // from-results form, extended in Slices 5/6). Streamed
        // requests are the union of original-active
        // (admission_src==none, request_id==seq_id, seq_id in
        // [0, n_active)) and admitted streams on either the
        // completion-freed (Slice 3 / Slice 5) or cancel-freed
        // (Slice 4 / Slice 6) admission path. arrival_src is
        // orthogonal: preloaded admitted streams (Slices 3/4) and
        // external admitted streams (Slices 5/6) both contribute
        // to streams_* through the same admit_one rebind path.
        int32_t expected_completed   = 0;
        int32_t expected_cancelled   = 0;
        int32_t expected_streams_opened = 0;
        int64_t expected_token_total = 0;
        for (const auto & rr : results) {
            const bool is_original_active =
                   rr.admission_src == admission_source::none
                && rr.request_id == rr.seq_id
                && rr.seq_id >= 0
                && rr.seq_id < args.n_active;
            const bool is_admitted_streamed =
                   rr.admission_src == admission_source::completion_freed
                || rr.admission_src == admission_source::cancel_freed;
            if (!is_original_active && !is_admitted_streamed) continue;
            expected_streams_opened++;
            if (rr.status == request_status::completed) {
                expected_completed++;
            } else if (rr.status == request_status::cancelled) {
                expected_cancelled++;
            }
            expected_token_total += rr.n_decoded;
        }
        if (er.streams_opened != expected_streams_opened) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "iter %d: streams_opened=%d != expected(from "
                "streamed request_results)=%d",
                r, er.streams_opened, expected_streams_opened);
            fail_with(buf); return false;
        }
        const int32_t closed_total =
            er.streams_closed_completed
          + er.streams_closed_cancelled
          + er.streams_closed_error;
        if (closed_total != er.streams_opened) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "iter %d: streams_opened=%d != streams_closed_"
                "completed+cancelled+error = %d+%d+%d = %d",
                r, er.streams_opened, er.streams_closed_completed,
                er.streams_closed_cancelled,
                er.streams_closed_error, closed_total);
            fail_with(buf); return false;
        }
        if (er.streams_closed_error != 0) {
            char buf[200];
            std::snprintf(buf, sizeof(buf),
                "iter %d: streams_closed_error=%d (expected 0)",
                r, er.streams_closed_error);
            fail_with(buf); return false;
        }
        if (er.streams_closed_completed != expected_completed) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "iter %d: streams_closed_completed=%d != "
                "expected(from rr.status=completed)=%d",
                r, er.streams_closed_completed, expected_completed);
            fail_with(buf); return false;
        }
        if (er.streams_closed_cancelled != expected_cancelled) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "iter %d: streams_closed_cancelled=%d != "
                "expected(from rr.status=cancelled)=%d",
                r, er.streams_closed_cancelled, expected_cancelled);
            fail_with(buf); return false;
        }
        if (er.stream_tokens_emitted_total != expected_token_total) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "iter %d: stream_tokens_emitted_total=%lld != "
                "sum(rr.n_decoded for streamed)=%lld",
                r,
                static_cast<long long>(er.stream_tokens_emitted_total),
                static_cast<long long>(expected_token_total));
            fail_with(buf); return false;
        }
    } else {
        if (er.streams_opened              != 0
         || er.streams_closed_completed    != 0
         || er.streams_closed_cancelled    != 0
         || er.streams_closed_error        != 0
         || er.stream_tokens_emitted_total != 0) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "iter %d: --stream-all OFF but stream counters "
                "non-zero (opened=%d completed=%d cancelled=%d "
                "error=%d tokens=%lld)",
                r, er.streams_opened,
                er.streams_closed_completed,
                er.streams_closed_cancelled,
                er.streams_closed_error,
                static_cast<long long>(
                    er.stream_tokens_emitted_total));
            fail_with(buf); return false;
        }
    }

    // ---- Per-budget completed summary + partitioned hash gates ----
    // Slice 3: hash uniqueness / done_iter / pos_max are validated
    // within (admission_src, decode_budget) partitions. Surviving
    // active budget-64 and admitted budget-64 share a budget but
    // not a batch-shape history, so their hashes are not required
    // to match each other (design §6, §11.7).
    std::vector<int32_t> uniq_budgets;
    for (const auto & rr : results) {
        bool seen = false;
        for (int32_t u : uniq_budgets) if (u == rr.decode_budget) { seen = true; break; }
        if (!seen) uniq_budgets.push_back(rr.decode_budget);
    }
    std::sort(uniq_budgets.begin(), uniq_budgets.end());

    struct partition_key {
        admission_source src;
        int32_t          budget;
    };
    const partition_key partitions[] = {
        {admission_source::none,             0},
        {admission_source::cancel_freed,     0},
        {admission_source::completion_freed, 0},
    };

    for (int32_t b : uniq_budgets) {
        for (const auto & pk_template : partitions) {
            const admission_source src = pk_template.src;
            int32_t  completed_b = 0;
            int32_t  cancelled_b = 0;
            uint64_t any_hash    = 0;
            std::vector<uint64_t>  hashes;
            std::vector<int32_t>   done_iters;
            std::vector<llama_pos> pos_maxes;
            for (const auto & rr : results) {
                if (rr.decode_budget != b)  continue;
                if (rr.admission_src != src) continue;
                if (rr.status == request_status::completed) {
                    completed_b++;
                    any_hash = rr.hash;
                    bool h_seen = false;
                    for (uint64_t hh : hashes) if (hh == rr.hash) { h_seen = true; break; }
                    if (!h_seen) hashes.push_back(rr.hash);
                    bool d_seen = false;
                    for (int32_t dd : done_iters) if (dd == rr.done_iter) { d_seen = true; break; }
                    if (!d_seen) done_iters.push_back(rr.done_iter);
                    bool p_seen = false;
                    for (llama_pos pp : pos_maxes) if (pp == rr.pos_max_at_clear) { p_seen = true; break; }
                    if (!p_seen) pos_maxes.push_back(rr.pos_max_at_clear);
                } else if (rr.status == request_status::cancelled) {
                    cancelled_b++;
                }
            }
            if (completed_b == 0 && cancelled_b == 0) continue;
            std::sort(done_iters.begin(), done_iters.end());
            std::sort(pos_maxes.begin(),  pos_maxes.end());

            // Expected anchors:
            //   admission_src == none: standard prefill at iter 0,
            //     done_iter = budget - 1, pos_max = n_prompt+budget-2.
            //   admission_src == cancel_freed: prefilled at
            //     cancel_after + 1, done_iter = admitted_at_iter +
            //     budget - 1, pos_max = n_prompt + budget - 2.
            //   admission_src == completion_freed (Slice 5):
            //     prefilled at min_active_budget, done_iter =
            //     admitted_at_iter + budget - 1, pos_max =
            //     n_prompt + budget - 2.
            int32_t expected_done_iter = b - 1;
            if (src == admission_source::cancel_freed) {
                expected_done_iter = expected_admitted_at_iter + b - 1;
            } else if (src == admission_source::completion_freed) {
                expected_done_iter =
                    expected_admitted_at_iter_completion + b - 1;
            }
            const int32_t expected_pos_max =
                n_prompt_tokens + b - 2;

            fprintf(stdout,
                "iter[%d] partition src=%s budget=%d "
                "completed=%d cancelled=%d "
                "unique_completed_hashes=%zu hash=0x%016llx "
                "done_iter_set={", r,
                admission_source_name(src), b,
                completed_b, cancelled_b,
                hashes.size(),
                static_cast<unsigned long long>(any_hash));
            for (size_t i = 0; i < done_iters.size(); i++) {
                fprintf(stdout, "%s%d", i == 0 ? "" : ",", done_iters[i]);
            }
            fprintf(stdout, "} pos_max_at_clear_set={");
            for (size_t i = 0; i < pos_maxes.size(); i++) {
                fprintf(stdout, "%s%d", i == 0 ? "" : ",", pos_maxes[i]);
            }
            fprintf(stdout, "}\n");

            if (completed_b > 0) {
                // M1d: unique-hash partition gate assumes every
                // request shares the same prompt; in file mode
                // prompts differ per request_id so distinct hashes
                // per partition are expected. Repeat determinism +
                // per-result hash stability still enforce that the
                // SAME prompt produces the SAME hash across repeats.
                if (legacy_shape && hashes.size() != 1) {
                    char buf[256];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: src=%s budget %d: %zu distinct "
                        "hashes among %d completed seqs (expected 1)",
                        r, admission_source_name(src), b,
                        hashes.size(), completed_b);
                    fail_with(buf); return false;
                }
                // The done_iter and pos_max equality checks are
                // single-cycle aggregates: under Slices 1-6 each
                // (admission_src, budget) partition has at most one
                // occupant per slot, so exactly one done_iter and
                // one pos_max value are expected. Streaming Slice 7
                // (multi-cycle reuse, completion-freed) admits
                // multiple occupants with the same budget on the
                // same slot, so the partition gathers >= 2
                // done_iters (one per cycle, e.g., {15, 23}) and
                // duplicate pos_maxes (e.g., [12, 12]). The
                // per-result gate
                //   rr.done_iter == rr.admitted_at_iter + budget - 1
                // (cpp:3491) already verifies the per-cycle
                // invariant for each occupant; skip the
                // partition-level aggregate equality under
                // slice7_multicycle.
                if (!slice7_multicycle) {
                    if (done_iters.size() != 1
                        || done_iters[0] != expected_done_iter) {
                        char buf[256];
                        std::snprintf(buf, sizeof(buf),
                            "iter %d: src=%s budget %d completed: "
                            "done_iter set has %zu values, "
                            "expected {%d}",
                            r, admission_source_name(src), b,
                            done_iters.size(), expected_done_iter);
                        fail_with(buf); return false;
                    }
                    // M1d: pos_max equality depends on prompt
                    // length being uniform across the partition;
                    // file mode breaks that assumption.
                    if (legacy_shape) {
                        if (pos_maxes.size() != 1
                            || pos_maxes[0] != expected_pos_max) {
                            char buf[256];
                            std::snprintf(buf, sizeof(buf),
                                "iter %d: src=%s budget %d completed: "
                                "pos_max set has %zu values, "
                                "expected {%d}",
                                r, admission_source_name(src), b,
                                pos_maxes.size(), expected_pos_max);
                            fail_with(buf); return false;
                        }
                    }
                }
                // Canonical budget-8 anchor only applies to the
                // untouched batch shape (admission_src == none).
                // The smoke shape never admits a budget-8 request
                // (budget-8 slots aren't in cancel_plan), so the
                // anchor still rides on the original active set.
                // M1d: only meaningful when every seq carries the
                // shared canonical prompt.
                if (legacy_shape
                    && src == admission_source::none
                    && b == 8
                    && any_hash != k_canonical_budget_8) {
                    char buf[256];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: budget 8 completed hash 0x%016llx "
                        "!= canonical 0x%016llx", r,
                        static_cast<unsigned long long>(any_hash),
                        static_cast<unsigned long long>(k_canonical_budget_8));
                    fail_with(buf); return false;
                }
            }
        }
    }

    // Per-class gate, restricted to the original active set
    // (admission_src == none). With admission, total per-budget
    // counts include admitted reusers and would no longer divide
    // evenly; the gate's intent is "the original active mix is
    // round-robin", which is unchanged. Streaming Slice 3: iterate
    // over args.decode_budget_mix directly rather than uniq_budgets
    // — admitted requests may carry --waiting-budget that is not in
    // the decode_budget_mix (e.g. Candidate A uses mix=[8] but
    // waiting-budget=16). The gate's intent applies only to mix
    // budgets, so iterating over the mix set is the precise read.
    if (args.n_active % static_cast<int32_t>(args.decode_budget_mix.size()) == 0) {
        const int32_t per_class =
            args.n_active / static_cast<int32_t>(args.decode_budget_mix.size());
        for (int32_t b : args.decode_budget_mix) {
            int32_t cnt = 0;
            for (const auto & rr : results) {
                if (rr.admission_src != admission_source::none) continue;
                if (rr.decode_budget == b) cnt++;
            }
            if (cnt != per_class) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: budget %d (admission_src=none) count=%d"
                    " (expected %d)", r, b, cnt, per_class);
                fail_with(buf); return false;
            }
        }
    }

    // Top-level cancelled / completed totals, partitioned by
    // admission source. Cancellations only affect the original
    // active set (cancel-freed-only rule). Admitted requests
    // always complete.
    int32_t orig_completed_total = 0;
    int32_t orig_cancelled_total = 0;
    int32_t admitted_completed_total = 0;
    for (const auto & rr : results) {
        if (rr.admission_src == admission_source::none) {
            if (rr.status == request_status::completed)      orig_completed_total++;
            else if (rr.status == request_status::cancelled) orig_cancelled_total++;
        } else if (rr.admission_src == admission_source::cancel_freed
                || rr.admission_src ==
                   admission_source::completion_freed) {
            if (rr.status == request_status::completed) admitted_completed_total++;
        }
    }
    if (orig_cancelled_total !=
        static_cast<int32_t>(args.cancel_plan.size())) {
        char buf[200];
        std::snprintf(buf, sizeof(buf),
            "iter %d: orig_cancelled_total=%d != cancel_plan size %zu",
            r, orig_cancelled_total, args.cancel_plan.size());
        fail_with(buf); return false;
    }
    if (orig_cancelled_total != er.cancelled_count) {
        char buf[200];
        std::snprintf(buf, sizeof(buf),
            "iter %d: orig_cancelled_total=%d (snapshots) != "
            "engine.cancelled_count=%d",
            r, orig_cancelled_total, er.cancelled_count);
        fail_with(buf); return false;
    }
    if (orig_completed_total + orig_cancelled_total != args.n_active) {
        char buf[200];
        std::snprintf(buf, sizeof(buf),
            "iter %d: orig completed=%d + cancelled=%d != n_active=%d",
            r, orig_completed_total, orig_cancelled_total,
            args.n_active);
        fail_with(buf); return false;
    }
    if (admitted_completed_total != er.admitted_count) {
        char buf[200];
        std::snprintf(buf, sizeof(buf),
            "iter %d: admitted_completed_total=%d != "
            "engine.admitted_count=%d",
            r, admitted_completed_total, er.admitted_count);
        fail_with(buf); return false;
    }
    const int32_t completed_total =
        orig_completed_total + admitted_completed_total;
    const int32_t cancelled_total = orig_cancelled_total;

    // Aggregate wasted-decode-rows-after-cancel.
    // M1d: this gate compares pos_max against (n_prompt + n_decoded - 2),
    // which assumes uniform prompt length. Compute the metric only in
    // legacy shared-prompt mode; in file mode the per-cancel "no
    // decode row after cancel observation" invariant is structurally
    // enforced by the engine (cancelled seqs immediately become
    // seq.done=true via clear_and_check, so the active_idx loop
    // skips them in subsequent decode iters). The metrics block
    // below still prints `wasted_decode_rows_after_cancel = 0` in
    // both modes so the line stays grep-friendly.
    int32_t wasted_rows = 0;
    if (legacy_shape) {
        for (const auto & rr : results) {
            if (rr.status != request_status::cancelled) continue;
            const llama_pos expected_pos =
                n_prompt_tokens + rr.n_decoded - 2;
            const llama_pos delta = rr.pos_max_at_clear - expected_pos;
            if (delta > 0) wasted_rows += static_cast<int32_t>(delta);
        }
        if (wasted_rows != 0) {
            char buf[200];
            std::snprintf(buf, sizeof(buf),
                "iter %d: wasted_decode_rows_after_cancel=%d "
                "(expected 0)", r, wasted_rows);
            fail_with(buf); return false;
        }
    }

    // ---- Live Admission Slice 3 gates ------------------------------
    // Live Admission Slice 4: structural gate — exactly one
    // admitted_prefilled trace site fires per admitted request.
    // Catches bugs in the trace predicate (e.g., misplaced after
    // n_decoded mutation) WITHOUT requiring trace to be enabled.
    if (er.admitted_prefill_events != er.admitted_count) {
        char buf[200];
        std::snprintf(buf, sizeof(buf),
            "iter %d: admitted_prefill_events=%d != admitted_count=%d",
            r, er.admitted_prefill_events, er.admitted_count);
        fail_with(buf); return false;
    }

    if (er.queued_count != args.n_waiting) {
        char buf[200];
        std::snprintf(buf, sizeof(buf),
            "iter %d: engine queued_count=%d (expected n_waiting=%d)",
            r, er.queued_count, args.n_waiting);
        fail_with(buf); return false;
    }
    // Live Admission Slice 6: admitted_count covers preloaded
    // waiters AND external arrivals; the upper bound is the sum.
    // preloaded_admissions == admitted_count - external_admitted_count
    // and is the count that maps 1:1 to queued_count drain.
    const int32_t preloaded_admissions =
        er.admitted_count - er.external_admitted_count;
    const int32_t admitted_upper_bound =
        args.n_waiting + args.n_external_arrivals;
    if (er.admitted_count < 0
     || er.admitted_count > admitted_upper_bound) {
        char buf[200];
        std::snprintf(buf, sizeof(buf),
            "iter %d: admitted_count=%d outside "
            "[0, n_waiting+n_external_arrivals=%d]",
            r, er.admitted_count, admitted_upper_bound);
        fail_with(buf); return false;
    }
    if (preloaded_admissions < 0
     || preloaded_admissions > args.n_waiting) {
        char buf[200];
        std::snprintf(buf, sizeof(buf),
            "iter %d: preloaded_admissions=%d outside "
            "[0, n_waiting=%d] (admitted=%d external=%d)",
            r, preloaded_admissions, args.n_waiting,
            er.admitted_count, er.external_admitted_count);
        fail_with(buf); return false;
    }
    const int32_t expected_end_size =
        args.n_waiting - preloaded_admissions;
    if (er.waiting_queue_size_at_engine_end != expected_end_size) {
        char buf[200];
        std::snprintf(buf, sizeof(buf),
            "iter %d: waiting_queue_size_at_engine_end=%d "
            "(expected n_waiting - preloaded_admissions = %d)",
            r, er.waiting_queue_size_at_engine_end,
            expected_end_size);
        fail_with(buf); return false;
    }
    if (static_cast<int32_t>(results.size()) != expected_total) {
        char buf[200];
        std::snprintf(buf, sizeof(buf),
            "iter %d: results.size()=%zu (expected n_active+admitted"
            "=%d)", r, results.size(), expected_total);
        fail_with(buf); return false;
    }

    // Live Admission Slice 7: reused_seq_id_set property checks.
    //   1. size == admitted_count
    //   2. NO DUPLICATES globally (the engine must never reuse the
    //      same slot twice in one run).
    //   3. STRICTLY ASCENDING WITHIN EACH (admitted_at_iter,
    //      admission_src) group. The earlier global strictly-
    //      ascending gate was Slice-3 specific: when admissions
    //      happen at a single iter from a single source, the cancel
    //      pool pops in seq_id-ascending order, so the global set
    //      is also ascending. In Slice 7, completion-freed
    //      admissions at iter 8 ({0,3,…,24}) are followed by
    //      cancel-freed admissions at iter 17 ({1,2,4,5,7,8}), so
    //      the global set is no longer monotonic. The cancel-pool
    //      and completion-pool FIFO invariants still produce
    //      ascending seq_ids within each (iter, src) group, which
    //      is what we gate.
    if (static_cast<int32_t>(er.reused_seq_id_set.size())
        != er.admitted_count) {
        char buf[200];
        std::snprintf(buf, sizeof(buf),
            "iter %d: reused_seq_id_set size=%zu != admitted_count=%d",
            r, er.reused_seq_id_set.size(), er.admitted_count);
        fail_with(buf); return false;
    }
    // No-duplicate check. Single-cycle reuse means each slot is
    // bound to at most one admitted occupant in a run, so
    // duplicates indicate a Slice 1–6 invariant violation.
    // Streaming Slice 7 (multi-cycle reuse on the
    // completion-freed admission path) intentionally produces
    // duplicates; the structural invariant for that case is the
    // chain walk gated above (per-occupant
    // previous_request_id == prior chain occupant request_id,
    // admitted_at_iter == prior occupant done_iter + 1). When
    // slice7_multicycle holds, skip the no-duplicate check.
    if (!slice7_multicycle) {
        std::vector<int32_t> sorted_seqs = er.reused_seq_id_set;
        std::sort(sorted_seqs.begin(), sorted_seqs.end());
        for (size_t i = 1; i < sorted_seqs.size(); i++) {
            if (sorted_seqs[i - 1] == sorted_seqs[i]) {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: reused_seq_id_set has duplicate "
                    "seq_id=%d", r, sorted_seqs[i]);
                fail_with(buf); return false;
            }
        }
    }
    // Per-(admitted_at_iter, admission_src) ascending check, ordered
    // by request_id (the waiting-queue FIFO order). Both cancel pool
    // and completion pool pop in seq_id-ascending order under the
    // existing engine invariants; within an admission iter the cancel
    // pass runs before the completion pass, and within each pass FIFO
    // == seq_id ascending. We do not assume the cancel and completion
    // sub-sequences interleave monotonically, so the check is
    // per-(iter, src) rather than per-iter.
    {
        const admission_source sources[] = {
            admission_source::cancel_freed,
            admission_source::completion_freed,
        };
        for (int32_t K : er.metrics.admission_iter_set) {
            for (admission_source src : sources) {
                std::vector<std::pair<int32_t, int32_t>> ordered;
                for (const auto & rr : results) {
                    if (rr.admitted_at_iter != K)  continue;
                    if (rr.admission_src   != src) continue;
                    ordered.emplace_back(rr.request_id,
                                         rr.reused_seq_id);
                }
                if (ordered.empty()) continue;
                std::sort(ordered.begin(), ordered.end());
                for (size_t i = 1; i < ordered.size(); i++) {
                    if (ordered[i - 1].second >= ordered[i].second) {
                        char buf[256];
                        std::snprintf(buf, sizeof(buf),
                            "iter %d: reused_seq_id sequence at "
                            "admitted_at_iter=%d src=%s not strictly "
                            "ascending (req %d seq %d after req %d "
                            "seq %d)",
                            r, K, admission_source_name(src),
                            ordered[i].first, ordered[i].second,
                            ordered[i - 1].first,
                            ordered[i - 1].second);
                        fail_with(buf); return false;
                    }
                }
            }
        }
    }

    // Live Admission Slice 5: per-source reused_seq_id constraints.
    //   cancel_freed     -> reused_seq_id MUST be in cancel_plan
    //   completion_freed -> reused_seq_id MUST NOT be in cancel_plan
    // The earlier blanket "must be in cancel_plan" gate fired
    // against all entries regardless of source and is now scoped.
    for (const auto & rr : results) {
        if (rr.admission_src == admission_source::cancel_freed) {
            if (!is_planned_cancel(rr.reused_seq_id)) {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: cancel_freed req %d reused_seq_id=%d"
                    " not in cancel_plan",
                    r, rr.request_id, rr.reused_seq_id);
                fail_with(buf); return false;
            }
        } else if (rr.admission_src ==
                   admission_source::completion_freed) {
            if (is_planned_cancel(rr.reused_seq_id)) {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: completion_freed req %d "
                    "reused_seq_id=%d is in cancel_plan",
                    r, rr.request_id, rr.reused_seq_id);
                fail_with(buf); return false;
            }
        }
    }

    // Count admissions by source so the smoke-shape strict gates
    // can fire conditionally without firing the wrong one.
    int32_t cancel_freed_admissions     = 0;
    int32_t completion_freed_admissions = 0;
    for (const auto & rr : results) {
        if (rr.admission_src == admission_source::cancel_freed)
            cancel_freed_admissions++;
        else if (rr.admission_src ==
                 admission_source::completion_freed)
            completion_freed_admissions++;
    }

    // Slice 3 smoke-shape strict gates (cancel-freed full admission):
    // fire only when cancel_plan is non-empty AND all admissions
    // were cancel_freed AND consumed the full waiting set.
    const bool slice3_strict =
        !args.cancel_plan.empty()
        && args.n_waiting > 0
        && er.admitted_count == args.n_waiting
        && cancel_freed_admissions == er.admitted_count
        && completion_freed_admissions == 0;

    if (slice3_strict) {
        // Sorted cancel_plan should equal reused_seq_id_set
        // when admitted_count == cancel_plan.size(). (Smoke
        // shape: both are 6.)
        if (static_cast<size_t>(er.admitted_count)
            <= args.cancel_plan.size()) {
            std::vector<int32_t> sorted_plan(
                args.cancel_plan.begin(), args.cancel_plan.end());
            std::sort(sorted_plan.begin(), sorted_plan.end());
            for (int32_t i = 0; i < er.admitted_count; i++) {
                if (er.reused_seq_id_set[
                        static_cast<size_t>(i)]
                    != sorted_plan[static_cast<size_t>(i)]) {
                    char buf[256];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: reused_seq_id_set[%d]=%d != "
                        "sorted(cancel_plan)[%d]=%d",
                        r, i,
                        er.reused_seq_id_set[
                            static_cast<size_t>(i)],
                        i, sorted_plan[static_cast<size_t>(i)]);
                    fail_with(buf); return false;
                }
            }
        }
        // Admitted request_ids must equal [n_active, n_active +
        // admitted_count).
        std::vector<int32_t> admitted_request_ids;
        for (const auto & rr : results) {
            if (rr.admission_src == admission_source::cancel_freed) {
                admitted_request_ids.push_back(rr.request_id);
            }
        }
        std::sort(admitted_request_ids.begin(),
                  admitted_request_ids.end());
        for (int32_t i = 0; i < er.admitted_count; i++) {
            const int32_t expected_req = args.n_active + i;
            if (admitted_request_ids[static_cast<size_t>(i)]
                != expected_req) {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: admitted_request_ids[%d]=%d "
                    "(expected %d)",
                    r, i,
                    admitted_request_ids[static_cast<size_t>(i)],
                    expected_req);
                fail_with(buf); return false;
            }
        }
        // FIFO mapping: request n_active+i bound to
        // sorted(cancel_plan)[i]. Already enforced because both
        // lists are deterministic and ascending; this is the
        // explicit cross-check.
        std::vector<int32_t> sorted_plan_for_map(
            args.cancel_plan.begin(), args.cancel_plan.end());
        std::sort(sorted_plan_for_map.begin(),
                  sorted_plan_for_map.end());
        for (const auto & rr : results) {
            if (rr.admission_src != admission_source::cancel_freed)
                continue;
            const int32_t i_in =
                rr.request_id - args.n_active;
            if (i_in < 0 || i_in >= er.admitted_count) continue;
            const int32_t expected_seq =
                sorted_plan_for_map[static_cast<size_t>(i_in)];
            if (rr.reused_seq_id != expected_seq) {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: req %d reused_seq_id=%d "
                    "(expected sorted(cancel_plan)[%d]=%d)",
                    r, rr.request_id, rr.reused_seq_id,
                    i_in, expected_seq);
                fail_with(buf); return false;
            }
            if (rr.previous_request_id != expected_seq) {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: req %d previous_request_id=%d "
                    "(expected %d)",
                    r, rr.request_id, rr.previous_request_id,
                    expected_seq);
                fail_with(buf); return false;
            }
        }
    }

    // Live Admission Slice 5 smoke-shape strict gates: fire only
    // when --reuse-completed is on, cancel_plan is empty, and all
    // admissions were completion_freed AND consumed the full
    // waiting set. The expected reused_seq_id_set is the first
    // n_waiting min-budget actives in round-robin (seq_id) order.
    // Streaming Slice 7 (multi-cycle): admitted_count > number of
    // min-budget original slots, so the strict block's
    // `admitted_count <= |min_budget_slots|` precondition would
    // fail with "only N min-budget slots available but
    // admitted_count=M". Exclude the multi-cycle smoke shape
    // here; the chain-walk gates above and the Slice 7 coverage
    // gate below are the structural correctness form.
    const bool slice5_strict =
        args.reuse_completed
        && args.cancel_plan.empty()
        && args.n_waiting > 0
        && args.n_waiting <= args.n_active
        && er.admitted_count == args.n_waiting
        && completion_freed_admissions == er.admitted_count
        && cancel_freed_admissions == 0;

    if (slice5_strict) {
        // Build the expected reused-seq list: every active slot
        // with budget == min_active_budget, in ascending seq_id
        // order, take the first admitted_count of those.
        std::vector<int32_t> min_budget_slots;
        min_budget_slots.reserve(
            static_cast<size_t>(args.n_active));
        for (int32_t s = 0; s < args.n_active; s++) {
            if (budgets[static_cast<size_t>(s)] ==
                min_active_budget) {
                min_budget_slots.push_back(s);
            }
        }
        if (static_cast<int32_t>(min_budget_slots.size())
            < er.admitted_count) {
            char buf[200];
            std::snprintf(buf, sizeof(buf),
                "iter %d: only %zu min-budget(%d) slots available "
                "but admitted_count=%d",
                r, min_budget_slots.size(), min_active_budget,
                er.admitted_count);
            fail_with(buf); return false;
        }
        for (int32_t i = 0; i < er.admitted_count; i++) {
            if (er.reused_seq_id_set[static_cast<size_t>(i)]
                != min_budget_slots[static_cast<size_t>(i)]) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: reused_seq_id_set[%d]=%d != "
                    "min_budget_slots[%d]=%d",
                    r, i,
                    er.reused_seq_id_set[static_cast<size_t>(i)],
                    i,
                    min_budget_slots[static_cast<size_t>(i)]);
                fail_with(buf); return false;
            }
        }
        // Admitted request_ids must equal [n_active, n_active +
        // admitted_count). Same shape as the cancel-freed strict
        // gate; the source is the only difference.
        std::vector<int32_t> admitted_request_ids;
        for (const auto & rr : results) {
            if (rr.admission_src ==
                admission_source::completion_freed) {
                admitted_request_ids.push_back(rr.request_id);
            }
        }
        std::sort(admitted_request_ids.begin(),
                  admitted_request_ids.end());
        for (int32_t i = 0; i < er.admitted_count; i++) {
            const int32_t expected_req = args.n_active + i;
            if (admitted_request_ids[static_cast<size_t>(i)]
                != expected_req) {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: completion_freed admitted_request_"
                    "ids[%d]=%d (expected %d)",
                    r, i,
                    admitted_request_ids[static_cast<size_t>(i)],
                    expected_req);
                fail_with(buf); return false;
            }
        }
        // FIFO mapping: request n_active+i -> min_budget_slots[i].
        for (const auto & rr : results) {
            if (rr.admission_src !=
                admission_source::completion_freed) continue;
            const int32_t i_in =
                rr.request_id - args.n_active;
            if (i_in < 0 || i_in >= er.admitted_count) continue;
            const int32_t expected_seq =
                min_budget_slots[static_cast<size_t>(i_in)];
            if (rr.reused_seq_id != expected_seq) {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: completion_freed req %d "
                    "reused_seq_id=%d (expected "
                    "min_budget_slots[%d]=%d)",
                    r, rr.request_id, rr.reused_seq_id,
                    i_in, expected_seq);
                fail_with(buf); return false;
            }
            if (rr.previous_request_id != expected_seq) {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: completion_freed req %d "
                    "previous_request_id=%d (expected %d)",
                    r, rr.request_id, rr.previous_request_id,
                    expected_seq);
                fail_with(buf); return false;
            }
        }
        // Pool residual gate: the first wave of min-budget natural
        // completions (size == round-robin count of that budget)
        // should leave exactly (wave_size - admitted_count) entries
        // pooled at run end. With the demand gate, later natural
        // completions (budget-64, budget-256) MUST NOT add to the
        // pool because the waiting queue is empty by then.
        const int32_t first_wave_size =
            static_cast<int32_t>(min_budget_slots.size());
        const int32_t expected_pool_residual =
            first_wave_size - er.admitted_count;
        if (er.completion_freed_pool_size_at_run_end !=
            expected_pool_residual) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "iter %d: completion_freed_pool_size_at_run_end="
                "%d (expected first_wave=%d - admitted=%d = %d)",
                r, er.completion_freed_pool_size_at_run_end,
                first_wave_size, er.admitted_count,
                expected_pool_residual);
            fail_with(buf); return false;
        }
    }

    // Live Admission Slice 5: when --reuse-completed is OFF the
    // engine MUST NOT touch the completion pool. Gate fail-closed.
    if (!args.reuse_completed
        && er.completion_freed_pool_size_at_run_end != 0) {
        char buf[200];
        std::snprintf(buf, sizeof(buf),
            "iter %d: completion_freed_pool_size_at_run_end=%d "
            "but --reuse-completed is OFF",
            r, er.completion_freed_pool_size_at_run_end);
        fail_with(buf); return false;
    }
    if (!args.reuse_completed && completion_freed_admissions != 0) {
        char buf[200];
        std::snprintf(buf, sizeof(buf),
            "iter %d: %d completion_freed admissions seen but "
            "--reuse-completed is OFF",
            r, completion_freed_admissions);
        fail_with(buf); return false;
    }

    // ---- Live Admission Slice 6 gates ------------------------------
    // Inert path (n_external_arrivals == 0): every Slice 6 counter
    // and set must stay at its default. This catches accidental
    // activation of the release/ack barrier or inbox drain when
    // the user did not opt in.
    if (args.n_external_arrivals == 0) {
        if (er.arrival_drained_count != 0
         || er.external_admitted_count != 0
         || er.first_external_drain_iter != -1
         || !er.iter_release_fired_set.empty()
         || !er.submitter_ack_set.empty()) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "iter %d: external path active without "
                "--n-external-arrivals "
                "(drained=%d ext_admitted=%d first_drain=%d "
                "release_set_size=%zu ack_set_size=%zu)",
                r, er.arrival_drained_count,
                er.external_admitted_count,
                er.first_external_drain_iter,
                er.iter_release_fired_set.size(),
                er.submitter_ack_set.size());
            fail_with(buf); return false;
        }
    } else {
        // arrival_drained_count == n_external_arrivals
        if (er.arrival_drained_count != args.n_external_arrivals) {
            char buf[200];
            std::snprintf(buf, sizeof(buf),
                "iter %d: arrival_drained_count=%d (expected %d)",
                r, er.arrival_drained_count,
                args.n_external_arrivals);
            fail_with(buf); return false;
        }
        // external_admitted_count == n_external_arrivals
        if (er.external_admitted_count != args.n_external_arrivals) {
            char buf[200];
            std::snprintf(buf, sizeof(buf),
                "iter %d: external_admitted_count=%d (expected %d)",
                r, er.external_admitted_count,
                args.n_external_arrivals);
            fail_with(buf); return false;
        }
        // first_external_drain_iter == external_release_iter + 1
        const int32_t expected_first_drain =
            args.external_release_iter + 1;
        if (er.first_external_drain_iter != expected_first_drain) {
            char buf[200];
            std::snprintf(buf, sizeof(buf),
                "iter %d: first_external_drain_iter=%d "
                "(expected release_iter+1=%d)",
                r, er.first_external_drain_iter,
                expected_first_drain);
            fail_with(buf); return false;
        }
        // iter_release_fired_set == {release_iter}
        if (er.iter_release_fired_set.size() != 1
         || er.iter_release_fired_set[0]
                != args.external_release_iter) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "iter %d: iter_release_fired_set size=%zu "
                "(expected {%d})",
                r, er.iter_release_fired_set.size(),
                args.external_release_iter);
            fail_with(buf); return false;
        }
        // submitter_ack_set == {release_iter}
        if (er.submitter_ack_set.size() != 1
         || er.submitter_ack_set[0]
                != args.external_release_iter) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "iter %d: submitter_ack_set size=%zu (expected {%d})",
                r, er.submitter_ack_set.size(),
                args.external_release_iter);
            fail_with(buf); return false;
        }
        // External admissions must carry arrival_src=external AND
        // land on a real freeing path: cancel_freed when a cancel
        // plan is configured (Live Admission Slice 6/7 path), or
        // completion_freed when --reuse-completed is on (Streaming
        // Slice 5 path). The cancel_after+1 admitted_at_iter
        // invariant is specific to the cancel-freed path; the
        // per-result Slice 3/5 admit-result gates at the top of
        // results processing already check the completion-freed
        // admitted_at_iter (prior occupant's done_iter + 1), so
        // we skip the iter recheck here for the completion-freed
        // external case.
        int32_t external_results_seen = 0;
        for (const auto & rr : results) {
            if (rr.arrival_src != arrival_source::external) continue;
            external_results_seen++;
            const bool ok_cancel =
                rr.admission_src == admission_source::cancel_freed
                && !args.cancel_plan.empty();
            const bool ok_completion =
                rr.admission_src == admission_source::completion_freed
                && args.reuse_completed;
            if (!ok_cancel && !ok_completion) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: external req %d admission_src=%s "
                    "(expected cancel_freed when cancel_plan "
                    "non-empty, or completion_freed when "
                    "--reuse-completed)",
                    r, rr.request_id,
                    admission_source_name(rr.admission_src));
                fail_with(buf); return false;
            }
            if (ok_cancel
                && rr.admitted_at_iter != expected_admitted_at_iter) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: external req %d admitted_at_iter=%d "
                    "(expected cancel_after+1=%d)",
                    r, rr.request_id, rr.admitted_at_iter,
                    expected_admitted_at_iter);
                fail_with(buf); return false;
            }
        }
        if (external_results_seen != args.n_external_arrivals) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "iter %d: external_results_seen=%d (expected %d)",
                r, external_results_seen, args.n_external_arrivals);
            fail_with(buf); return false;
        }
        // Every NON-external result must carry arrival_src=preloaded.
        for (const auto & rr : results) {
            if (rr.arrival_src == arrival_source::external) continue;
            if (rr.arrival_src != arrival_source::preloaded) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: non-external req %d arrival_src=%s "
                    "(expected preloaded)",
                    r, rr.request_id,
                    arrival_source_name(rr.arrival_src));
                fail_with(buf); return false;
            }
        }

        // Slice 6 strict gate: the smoke shape. cancel-plan size
        // matches n_external_arrivals, no preloaded waiters, all
        // admissions are external + cancel_freed. Verify the FIFO
        // request->seq mapping and the canonical budget-64 hash.
        const bool slice6_strict =
            !args.cancel_plan.empty()
            && args.n_waiting == 0
            && static_cast<int32_t>(args.cancel_plan.size())
                   == args.n_external_arrivals
            && er.admitted_count == args.n_external_arrivals
            && er.external_admitted_count == args.n_external_arrivals
            && cancel_freed_admissions == er.admitted_count
            && completion_freed_admissions == 0;
        if (slice6_strict) {
            std::vector<int32_t> sorted_plan(
                args.cancel_plan.begin(), args.cancel_plan.end());
            std::sort(sorted_plan.begin(), sorted_plan.end());
            // request n_active + i -> sorted(cancel_plan)[i]
            for (const auto & rr : results) {
                if (rr.arrival_src != arrival_source::external) continue;
                const int32_t i_in = rr.request_id - args.n_active;
                if (i_in < 0 || i_in >= args.n_external_arrivals) {
                    char buf[256];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: external req %d outside expected "
                        "[%d, %d) id range",
                        r, rr.request_id, args.n_active,
                        args.n_active + args.n_external_arrivals);
                    fail_with(buf); return false;
                }
                const int32_t expected_seq =
                    sorted_plan[static_cast<size_t>(i_in)];
                if (rr.reused_seq_id != expected_seq) {
                    char buf[256];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: external req %d reused_seq_id=%d "
                        "(expected sorted(cancel_plan)[%d]=%d)",
                        r, rr.request_id, rr.reused_seq_id,
                        i_in, expected_seq);
                    fail_with(buf); return false;
                }
            }
            // budget-64 canonical hash anchor for the slice 6 smoke
            // shape (cancel-freed external admissions only).
            // M1d: only meaningful when every external arrival
            // carries the canonical shared prompt.
            constexpr uint64_t k_canonical_slice6_budget_64 =
                0x3b15a0474dfe11beull;
            if (legacy_shape && args.external_arrival_budget == 64) {
                for (const auto & rr : results) {
                    if (rr.arrival_src != arrival_source::external) continue;
                    if (rr.decode_budget != 64) continue;
                    if (rr.hash != k_canonical_slice6_budget_64) {
                        char buf[256];
                        std::snprintf(buf, sizeof(buf),
                            "iter %d: external req %d budget=64 "
                            "hash=0x%016llx != canonical 0x%016llx",
                            r, rr.request_id,
                            static_cast<unsigned long long>(rr.hash),
                            static_cast<unsigned long long>(
                                k_canonical_slice6_budget_64));
                        fail_with(buf); return false;
                    }
                }
            }
        }
    }

    // ---- Live Admission Slice 7 strict mixed-source gate ----------
    // Fires only on the canonical Slice 7 smoke shape: reuse_completed
    // ON, cancel-plan non-empty, both preloaded waiters AND external
    // arrivals present, completion_freed admissions matched n_waiting
    // (phase 1), cancel_freed admissions matched n_external_arrivals
    // (phase 2), and admission fired at exactly two iters
    // {min_active_budget, cancel_after+1}. The body asserts the
    // mixed-source priority rule: at the iter where both pools are
    // non-empty (phase 2), cancel_freed drains first, completion_freed
    // residual is unchanged.
    const bool slice7_strict =
        args.reuse_completed
        && !args.cancel_plan.empty()
        && args.n_waiting           > 0
        && args.n_external_arrivals > 0
        && completion_freed_admissions == args.n_waiting
        && cancel_freed_admissions     == args.n_external_arrivals
        && er.external_admitted_count  == args.n_external_arrivals
        && er.admitted_count           == args.n_waiting
                                          + args.n_external_arrivals;
    if (slice7_strict) {
        const int32_t expected_phase1_iter =
            expected_admitted_at_iter_completion;  // == min_active_budget
        const int32_t expected_phase2_iter =
            expected_admitted_at_iter;             // == cancel_after + 1

        // admission_iter_set must be exactly {phase1_iter, phase2_iter}.
        const auto & ais = er.metrics.admission_iter_set;
        std::set<int32_t> ais_set(ais.begin(), ais.end());
        std::set<int32_t> expected_ais{expected_phase1_iter,
                                       expected_phase2_iter};
        if (ais_set != expected_ais) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "iter %d: slice7 admission_iter_set size=%zu does "
                "not equal {phase1=%d, phase2=%d}",
                r, ais.size(), expected_phase1_iter,
                expected_phase2_iter);
            fail_with(buf); return false;
        }

        // Phase 1: every admitted result with admitted_at_iter ==
        // phase1_iter must be completion_freed + arrival_src=preloaded.
        // Phase 2: every admitted result with admitted_at_iter ==
        // phase2_iter must be cancel_freed + arrival_src=external.
        int32_t phase1_seen = 0;
        int32_t phase2_seen = 0;
        for (const auto & rr : results) {
            if (rr.admission_src == admission_source::none) continue;
            if (rr.admitted_at_iter == expected_phase1_iter) {
                phase1_seen++;
                if (rr.admission_src
                        != admission_source::completion_freed) {
                    char buf[256];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: slice7 phase1 req %d "
                        "admission_src=%s (expected completion_freed)",
                        r, rr.request_id,
                        admission_source_name(rr.admission_src));
                    fail_with(buf); return false;
                }
                if (rr.arrival_src != arrival_source::preloaded) {
                    char buf[256];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: slice7 phase1 req %d "
                        "arrival_src=%s (expected preloaded)",
                        r, rr.request_id,
                        arrival_source_name(rr.arrival_src));
                    fail_with(buf); return false;
                }
            } else if (rr.admitted_at_iter == expected_phase2_iter) {
                phase2_seen++;
                if (rr.admission_src
                        != admission_source::cancel_freed) {
                    char buf[256];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: slice7 phase2 req %d "
                        "admission_src=%s (expected cancel_freed)",
                        r, rr.request_id,
                        admission_source_name(rr.admission_src));
                    fail_with(buf); return false;
                }
                if (rr.arrival_src != arrival_source::external) {
                    char buf[256];
                    std::snprintf(buf, sizeof(buf),
                        "iter %d: slice7 phase2 req %d "
                        "arrival_src=%s (expected external)",
                        r, rr.request_id,
                        arrival_source_name(rr.arrival_src));
                    fail_with(buf); return false;
                }
            } else {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: slice7 admitted req %d at unexpected "
                    "iter %d (expected %d or %d)",
                    r, rr.request_id, rr.admitted_at_iter,
                    expected_phase1_iter, expected_phase2_iter);
                fail_with(buf); return false;
            }
        }
        if (phase1_seen != args.n_waiting) {
            char buf[200];
            std::snprintf(buf, sizeof(buf),
                "iter %d: slice7 phase1_seen=%d (expected n_waiting=%d)",
                r, phase1_seen, args.n_waiting);
            fail_with(buf); return false;
        }
        if (phase2_seen != args.n_external_arrivals) {
            char buf[200];
            std::snprintf(buf, sizeof(buf),
                "iter %d: slice7 phase2_seen=%d (expected "
                "n_external_arrivals=%d)",
                r, phase2_seen, args.n_external_arrivals);
            fail_with(buf); return false;
        }

        // Phase 1 mapping: request n_active + i -> min_budget_slots[i],
        // where min_budget_slots are the budget-min actives in
        // ascending seq_id order. For the smoke shape: budget-8 slots
        // {0,3,6,9,12,15,18,21,24,…}; the first n_waiting of those
        // are the phase 1 admissions.
        std::vector<int32_t> min_budget_slots;
        min_budget_slots.reserve(static_cast<size_t>(args.n_active));
        for (int32_t s = 0; s < args.n_active; s++) {
            if (budgets[static_cast<size_t>(s)] == min_active_budget) {
                min_budget_slots.push_back(s);
            }
        }
        if (static_cast<int32_t>(min_budget_slots.size())
                < args.n_waiting) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "iter %d: slice7 only %zu min-budget(%d) actives "
                "available but n_waiting=%d",
                r, min_budget_slots.size(), min_active_budget,
                args.n_waiting);
            fail_with(buf); return false;
        }
        for (const auto & rr : results) {
            if (rr.admission_src != admission_source::completion_freed)
                continue;
            const int32_t i_in = rr.request_id - args.n_active;
            if (i_in < 0 || i_in >= args.n_waiting) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: slice7 phase1 req %d outside expected "
                    "[%d, %d) preloaded id range",
                    r, rr.request_id, args.n_active,
                    args.n_active + args.n_waiting);
                fail_with(buf); return false;
            }
            const int32_t expected_seq =
                min_budget_slots[static_cast<size_t>(i_in)];
            if (rr.reused_seq_id != expected_seq) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: slice7 phase1 req %d reused_seq_id=%d "
                    "(expected min_budget_slots[%d]=%d)",
                    r, rr.request_id, rr.reused_seq_id, i_in,
                    expected_seq);
                fail_with(buf); return false;
            }
        }

        // Phase 2 mapping: request n_active + n_waiting + i ->
        // sorted(cancel_plan)[i]. Same shape as Slice 6, but offset
        // by n_waiting in the request_id space.
        std::vector<int32_t> sorted_plan(
            args.cancel_plan.begin(), args.cancel_plan.end());
        std::sort(sorted_plan.begin(), sorted_plan.end());
        for (const auto & rr : results) {
            if (rr.admission_src != admission_source::cancel_freed)
                continue;
            const int32_t i_in = rr.request_id
                                  - args.n_active
                                  - args.n_waiting;
            if (i_in < 0 || i_in >= args.n_external_arrivals) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: slice7 phase2 req %d outside expected "
                    "[%d, %d) external id range",
                    r, rr.request_id,
                    args.n_active + args.n_waiting,
                    args.n_active + args.n_waiting
                        + args.n_external_arrivals);
                fail_with(buf); return false;
            }
            const int32_t expected_seq =
                sorted_plan[static_cast<size_t>(i_in)];
            if (rr.reused_seq_id != expected_seq) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: slice7 phase2 req %d reused_seq_id=%d "
                    "(expected sorted(cancel_plan)[%d]=%d)",
                    r, rr.request_id, rr.reused_seq_id, i_in,
                    expected_seq);
                fail_with(buf); return false;
            }
        }

        // No phase-2 admission may bind a residual completion-freed
        // seq_id (a min-budget slot NOT consumed in phase 1). The
        // residual set is min_budget_slots[n_waiting:].
        std::set<int32_t> residual_seqs;
        for (size_t i = static_cast<size_t>(args.n_waiting);
             i < min_budget_slots.size(); i++) {
            residual_seqs.insert(min_budget_slots[i]);
        }
        for (const auto & rr : results) {
            if (rr.admitted_at_iter != expected_phase2_iter) continue;
            if (residual_seqs.count(rr.reused_seq_id)) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: slice7 phase2 req %d reused_seq_id=%d "
                    "is in completion-freed residual pool — cancel-"
                    "freed priority violated",
                    r, rr.request_id, rr.reused_seq_id);
                fail_with(buf); return false;
            }
        }

        // Completion-freed pool residual: first_wave_size = count of
        // min-budget actives (28 in the smoke); completion_freed
        // admissions == n_waiting. Demand gate stops later pushes
        // because waiting_queue is empty by the time admitted
        // budget-8 requests complete and surviving budget-64/256
        // actives complete. So residual = first_wave - n_waiting.
        const int32_t first_wave_size =
            static_cast<int32_t>(min_budget_slots.size());
        const int32_t expected_pool_residual =
            first_wave_size - args.n_waiting;
        if (er.completion_freed_pool_size_at_run_end
                != expected_pool_residual) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "iter %d: slice7 completion_freed_pool_size_at_run_end"
                "=%d (expected first_wave=%d - n_waiting=%d = %d)",
                r, er.completion_freed_pool_size_at_run_end,
                first_wave_size, args.n_waiting,
                expected_pool_residual);
            fail_with(buf); return false;
        }
    }

    fprintf(stdout,
        "iter[%d] admit_step5: orig completed=%d cancelled=%d, "
        "admitted=%d (cancel_freed=%d completion_freed=%d), "
        "total_results=%d, queued=%d waiting_end=%d, "
        "completion_pool_residual=%d\n",
        r, orig_completed_total, orig_cancelled_total,
        er.admitted_count,
        cancel_freed_admissions, completion_freed_admissions,
        expected_total,
        er.queued_count, er.waiting_queue_size_at_engine_end,
        er.completion_freed_pool_size_at_run_end);

    fprintf(stdout,
        "iter[%d] admit_step6: external arrivals "
        "drained=%d admitted=%d first_drain_iter=%d "
        "release_set_size=%zu ack_set_size=%zu\n",
        r, er.arrival_drained_count, er.external_admitted_count,
        er.first_external_drain_iter,
        er.iter_release_fired_set.size(),
        er.submitter_ack_set.size());

    // Live Admission Slice 7 audit line. Sources counts from the
    // results vector (status-true cancel_freed_admissions and
    // completion_freed_admissions are already computed above) and
    // from engine_result. admission_iter_set is rendered as a
    // sorted ascending sequence — the engine pushes it in iter
    // order so it is already monotonic, but we format it
    // explicitly to make the audit line grep-friendly.
    {
        fprintf(stdout,
            "iter[%d] admit_step7: phase1@iter=%d completion_freed=%d "
            "phase2@iter=%d cancel_freed=%d pool_residual=%d "
            "admission_iter_set={",
            r, expected_admitted_at_iter_completion,
            completion_freed_admissions,
            expected_admitted_at_iter,
            cancel_freed_admissions,
            er.completion_freed_pool_size_at_run_end);
        for (size_t i = 0;
             i < er.metrics.admission_iter_set.size(); i++) {
            fprintf(stdout, "%s%d", i == 0 ? "" : ",",
                    er.metrics.admission_iter_set[i]);
        }
        fprintf(stdout, "} first_external_drain_iter=%d\n",
                er.first_external_drain_iter);
    }

    fprintf(stdout,
        "iter[%d] residual_kv: all %d seqs cleared "
        "(pos_min=-1, pos_max=-1)\n", r, args.n_seqs);

    // status_summary: Slice 3 retains the grep-friendly summary.
    fprintf(stdout,
        "iter[%d] status_summary: completed=%d cancelled=%d "
        "total=%d (orig_completed=%d admitted_completed=%d)\n",
        r, completed_total, cancelled_total,
        completed_total + cancelled_total,
        orig_completed_total, admitted_completed_total);

    // Per-budget cancellation summary (unchanged Slice-2 output;
    // useful when --cancel-plan is non-default).
    for (int32_t b : uniq_budgets) {
        int32_t cancelled_b = 0;
        std::vector<int32_t> cancel_iters;
        std::vector<int32_t> cancel_n_decodes;
        for (const auto & rr : results) {
            if (rr.decode_budget != b) continue;
            if (rr.status != request_status::cancelled) continue;
            cancelled_b++;
            bool ci_seen = false;
            for (int32_t v : cancel_iters)
                if (v == rr.cancel_observed_iter) { ci_seen = true; break; }
            if (!ci_seen) cancel_iters.push_back(rr.cancel_observed_iter);
            bool nd_seen = false;
            for (int32_t v : cancel_n_decodes)
                if (v == rr.n_decoded_at_cancel) { nd_seen = true; break; }
            if (!nd_seen) cancel_n_decodes.push_back(rr.n_decoded_at_cancel);
        }
        if (cancelled_b == 0) continue;
        std::sort(cancel_iters.begin(),     cancel_iters.end());
        std::sort(cancel_n_decodes.begin(), cancel_n_decodes.end());
        fprintf(stdout,
            "iter[%d] cancelled budget=%d count=%d "
            "cancel_observed_iter_set={", r, b, cancelled_b);
        for (size_t i = 0; i < cancel_iters.size(); i++) {
            fprintf(stdout, "%s%d", i == 0 ? "" : ",",
                    cancel_iters[i]);
        }
        fprintf(stdout, "} n_decoded_at_cancel_set={");
        for (size_t i = 0; i < cancel_n_decodes.size(); i++) {
            fprintf(stdout, "%s%d", i == 0 ? "" : ",",
                    cancel_n_decodes[i]);
        }
        fprintf(stdout, "}\n");
    }

    // ---- Descriptive metrics block ---------------------------------
    {
        const engine_metrics & m = er.metrics;

        std::vector<double> rpb;
        rpb.reserve(m.rows_per_batch.size());
        int32_t rpb_max = 0;
        for (int32_t v : m.rows_per_batch) {
            rpb.push_back(static_cast<double>(v));
            if (v > rpb_max) rpb_max = v;
        }
        std::vector<double> asp;
        asp.reserve(m.active_seqs_per_iter.size());
        int32_t asp_max = 0;
        for (int32_t v : m.active_seqs_per_iter) {
            asp.push_back(static_cast<double>(v));
            if (v > asp_max) asp_max = v;
        }
        const double rpb_p50 = percentile(rpb, 0.50);
        const double rpb_p95 = percentile(rpb, 0.95);
        const double asp_p50 = percentile(asp, 0.50);
        const double asp_p95 = percentile(asp, 0.95);

        fprintf(stdout, "iter[%d] metrics:\n", r);
        fprintf(stdout, "  wall_ms                 = %.3f\n",
                m.wall_ms);
        fprintf(stdout, "  decode_calls            = %d\n",
                m.decode_calls);
        fprintf(stdout, "  update_iterations       = %d\n",
                m.update_iterations);
        fprintf(stdout,
            "  rows_per_batch          p50=%.0f p95=%.0f max=%d\n",
            rpb_p50, rpb_p95, rpb_max);
        fprintf(stdout,
            "  active_seqs_per_iter    p50=%.0f p95=%.0f max=%d\n",
            asp_p50, asp_p95, asp_max);

        // Per-budget completed/cancelled counts. The earlier
        // version of this loop printed `completed[budget=B] = <total>`
        // which was misleading once cancellation became real;
        // Cancel Slice 4 fixes the count and emits a paired
        // `cancelled[budget=B]` line for budgets with any
        // cancelled seqs.
        for (int32_t b : uniq_budgets) {
            int32_t comp_cnt = 0;
            int32_t canc_cnt = 0;
            for (const auto & rr : results) {
                if (rr.decode_budget != b) continue;
                if (rr.status == request_status::completed)      comp_cnt++;
                else if (rr.status == request_status::cancelled) canc_cnt++;
            }
            fprintf(stdout,
                "  completed[budget=%d]    = %d\n", b, comp_cnt);
            if (canc_cnt > 0) {
                fprintf(stdout,
                    "  cancelled[budget=%d]    = %d\n", b, canc_cnt);
            }
        }
        // ttc_ms is descriptive only (excluded from the
        // determinism contract). Cancel Slice 4 splits it by
        // status so completion-time and cancellation-time
        // distributions are not blurred together.
        for (int32_t b : uniq_budgets) {
            std::vector<double> ttc_ms_comp;
            std::vector<double> ttc_ms_canc;
            ttc_ms_comp.reserve(static_cast<size_t>(33));
            ttc_ms_canc.reserve(static_cast<size_t>(8));
            for (const auto & rr : results) {
                if (rr.decode_budget != b) continue;
                const double v =
                    static_cast<double>(rr.ttc_us) / 1000.0;
                if (rr.status == request_status::completed) {
                    ttc_ms_comp.push_back(v);
                } else if (rr.status == request_status::cancelled) {
                    ttc_ms_canc.push_back(v);
                }
            }
            if (!ttc_ms_comp.empty()) {
                fprintf(stdout,
                    "  ttc_ms_completed[budget=%d]  mean=%.2f p95=%.2f\n",
                    b, mean_of(ttc_ms_comp),
                    percentile(ttc_ms_comp, 0.95));
            }
            if (!ttc_ms_canc.empty()) {
                fprintf(stdout,
                    "  ttc_ms_cancelled[budget=%d]  mean=%.2f p95=%.2f\n",
                    b, mean_of(ttc_ms_canc),
                    percentile(ttc_ms_canc, 0.95));
            }
        }
        fprintf(stdout, "  futures_created         = %d\n",
                futures_created);
        fprintf(stdout, "  promises_fulfilled      = %d\n",
                er.promises_fulfilled);
        fprintf(stdout, "  futures_completed       = %d\n",
                futures_completed);
        fprintf(stdout, "  engine_task_count       = %d\n",
                er.engine_task_count);
        // Cancel Slice 4 closeout fields. Numeric/boolean only;
        // the underlying invariants are already gated above and
        // these lines exist for grep-friendly closeout reporting.
        fprintf(stdout, "  completed_count         = %d\n",
                completed_total);
        fprintf(stdout, "  cancelled_count         = %d\n",
                cancelled_total);
        fprintf(stdout, "  decode_failures         = %d\n",
                er.decode_failures);
        fprintf(stdout, "  wasted_decode_rows_after_cancel = %d\n",
                wasted_rows);
        fprintf(stdout, "  residual_kv_empty       = %s\n",
                er.residual_kv_ok ? "true" : "false");
        // Live Admission Slice 3: descriptive only.
        fprintf(stdout, "  queued_count            = %d\n",
                er.queued_count);
        fprintf(stdout, "  admitted_count          = %d\n",
                er.admitted_count);
        fprintf(stdout, "  waiting_queue_size_at_engine_end = %d\n",
                er.waiting_queue_size_at_engine_end);
        fprintf(stdout, "  reused_seq_id_set       = {");
        for (size_t i = 0; i < er.reused_seq_id_set.size(); i++) {
            fprintf(stdout, "%s%d", i == 0 ? "" : ",",
                    er.reused_seq_id_set[i]);
        }
        fprintf(stdout, "}\n");
        // Live Admission Slice 4: descriptive only.
        fprintf(stdout, "  reused_seq_id_count     = %zu\n",
                er.reused_seq_id_set.size());
        fprintf(stdout, "  admitted_prefill_events = %d\n",
                er.admitted_prefill_events);
        fprintf(stdout, "  admission_iter_set      = {");
        for (size_t i = 0;
             i < m.admission_iter_set.size(); i++) {
            fprintf(stdout, "%s%d", i == 0 ? "" : ",",
                    m.admission_iter_set[i]);
        }
        fprintf(stdout, "}\n");
        // waiting_queue_depth_after_admission_per_iter is sampled
        // AFTER each iter's admission loop, so for the smoke shape
        // this is n_waiting for iters 1..16 and 0 from iter 17 on.
        std::vector<double> wqd;
        wqd.reserve(
            m.waiting_queue_depth_after_admission_per_iter.size());
        int32_t wqd_max = 0;
        for (int32_t v :
             m.waiting_queue_depth_after_admission_per_iter) {
            wqd.push_back(static_cast<double>(v));
            if (v > wqd_max) wqd_max = v;
        }
        const double wqd_p50 = percentile(wqd, 0.50);
        const double wqd_p95 = percentile(wqd, 0.95);
        fprintf(stdout,
            "  waiting_queue_depth_after_admission_per_iter "
            "p50=%.0f p95=%.0f max=%d (samples=%zu)\n",
            wqd_p50, wqd_p95, wqd_max, wqd.size());
        // admitted_ttc_ms[budget=B]: completion-time distribution
        // for admitted-completed results, segmented by budget AND
        // by admission source (so cancel_freed and completion_freed
        // admissions are not blurred together). Descriptive only.
        for (int32_t b : uniq_budgets) {
            std::vector<double> ttc_cf;   // cancel_freed
            std::vector<double> ttc_pf;   // completion_freed (Slice 5)
            for (const auto & rr : results) {
                if (rr.decode_budget != b) continue;
                if (rr.status != request_status::completed) continue;
                const double v =
                    static_cast<double>(rr.ttc_us) / 1000.0;
                if (rr.admission_src ==
                    admission_source::cancel_freed) {
                    ttc_cf.push_back(v);
                } else if (rr.admission_src ==
                           admission_source::completion_freed) {
                    ttc_pf.push_back(v);
                }
            }
            if (!ttc_cf.empty()) {
                fprintf(stdout,
                    "  admitted_ttc_ms[src=cancel_freed,"
                    "budget=%d]  mean=%.2f p95=%.2f\n",
                    b, mean_of(ttc_cf),
                    percentile(ttc_cf, 0.95));
            }
            if (!ttc_pf.empty()) {
                fprintf(stdout,
                    "  admitted_ttc_ms[src=completion_freed,"
                    "budget=%d]  mean=%.2f p95=%.2f\n",
                    b, mean_of(ttc_pf),
                    percentile(ttc_pf, 0.95));
            }
        }
        // Live Admission Slice 5: descriptive — residual size of
        // free_due_to_completion_ at engine end. Demand-gated, so
        // this stays at (first_wave - admitted) for the Slice 5
        // smoke shape (== 21).
        fprintf(stdout,
            "  completion_freed_pool_size_at_run_end = %d\n",
            er.completion_freed_pool_size_at_run_end);
        // Streaming Slice 8: descriptive stream counters.
        fprintf(stdout, "  stream_all                = %d\n",
                args.stream_all ? 1 : 0);
        fprintf(stdout, "  streams_opened            = %d\n",
                er.streams_opened);
        fprintf(stdout, "  streams_closed_completed  = %d\n",
                er.streams_closed_completed);
        fprintf(stdout, "  streams_closed_cancelled  = %d\n",
                er.streams_closed_cancelled);
        fprintf(stdout, "  streams_closed_error      = %d\n",
                er.streams_closed_error);
        fprintf(stdout,
            "  stream_tokens_emitted_total = %lld\n",
            static_cast<long long>(er.stream_tokens_emitted_total));
        if (args.stream_all) {
            for (size_t s = 0; s < streamed_tokens.size(); s++) {
                fprintf(stdout,
                    "  streamed[seq=%zu] count=%zu close=%s\n",
                    s, streamed_tokens[s].size(),
                    stream_close_reason_name(streamed_close[s]));
            }
            // Streaming Slice 3: per-admission streamed lines.
            // Sorted by request_id for deterministic output.
            std::vector<int32_t> admitted_rids;
            admitted_rids.reserve(admitted_streamed_tokens.size());
            for (const auto & kv : admitted_streamed_tokens) {
                admitted_rids.push_back(kv.first);
            }
            std::sort(admitted_rids.begin(), admitted_rids.end());
            for (int32_t rid : admitted_rids) {
                const auto & toks =
                    admitted_streamed_tokens.at(rid);
                auto itc = admitted_streamed_close.find(rid);
                const stream_close_reason cr =
                    (itc != admitted_streamed_close.end())
                        ? itc->second
                        : stream_close_reason::completed;
                admission_source asrc = admission_source::none;
                for (const auto & rr_ : results) {
                    if (rr_.request_id == rid) {
                        asrc = rr_.admission_src;
                        break;
                    }
                }
                fprintf(stdout,
                    "  streamed[req=%d admission_src=%s] "
                    "count=%zu close=%s\n",
                    rid, admission_source_name(asrc),
                    toks.size(),
                    stream_close_reason_name(cr));
            }
        }
    }

    if (r == 0) {
        vstate.first_results = results;
        vstate.first_streamed_tokens = streamed_tokens;
        vstate.first_streamed_close  = streamed_close;
        vstate.first_admitted_streamed_tokens = admitted_streamed_tokens;
        vstate.first_admitted_streamed_close  = admitted_streamed_close;
    } else {
        // Determinism gate: identical n_decoded / generated_tokens /
        // hash / done_iter / pos_max_at_clear across repeats. ttc_us
        // and timing-derived metrics are not part of the determinism
        // contract.
        if (results.size() != vstate.first_results.size()) {
            char buf[128];
            std::snprintf(buf, sizeof(buf),
                "iter %d: %zu results vs first iter %zu",
                r, results.size(), vstate.first_results.size());
            fail_with(buf); return false;
        }
        for (size_t i = 0; i < results.size(); i++) {
            const request_result & a0 = vstate.first_results[i];
            const request_result & a  = results[i];
            if (a.seq_id              != a0.seq_id
             || a.request_id          != a0.request_id
             || a.n_decoded           != a0.n_decoded
             || a.generated_tokens    != a0.generated_tokens
             || a.hash                != a0.hash
             || a.done_iter           != a0.done_iter
             || a.pos_max_at_clear    != a0.pos_max_at_clear
             || a.admitted_at_iter    != a0.admitted_at_iter
             || a.reused_seq_id       != a0.reused_seq_id
             || a.previous_request_id != a0.previous_request_id
             || a.admission_src       != a0.admission_src) {
                char buf[320];
                std::snprintf(buf, sizeof(buf),
                    "non-deterministic: iter %d seq %d hash "
                    "0x%016llx differs from iter 0 hash 0x%016llx",
                    r, a.seq_id,
                    static_cast<unsigned long long>(a.hash),
                    static_cast<unsigned long long>(a0.hash));
                fail_with(buf); return false;
            }
        }
        // Streaming Slice 8: bit-equal streamed-token vectors and
        // close reasons across repeats. With --stream-all OFF both
        // sides are empty and this loop is a no-op.
        if (streamed_tokens.size() != vstate.first_streamed_tokens.size()) {
            char buf[200];
            std::snprintf(buf, sizeof(buf),
                "iter %d: streamed_tokens.size()=%zu != iter 0 "
                "size=%zu", r, streamed_tokens.size(),
                vstate.first_streamed_tokens.size());
            fail_with(buf); return false;
        }
        for (size_t s = 0; s < streamed_tokens.size(); s++) {
            if (streamed_tokens[s] != vstate.first_streamed_tokens[s]) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: streamed_tokens[seq=%zu] differs "
                    "from iter 0 (size %zu vs %zu)",
                    r, s, streamed_tokens[s].size(),
                    vstate.first_streamed_tokens[s].size());
                fail_with(buf); return false;
            }
            if (streamed_close[s] != vstate.first_streamed_close[s]) {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: streamed_close[seq=%zu]=%s differs "
                    "from iter 0=%s", r, s,
                    stream_close_reason_name(streamed_close[s]),
                    stream_close_reason_name(vstate.first_streamed_close[s]));
                fail_with(buf); return false;
            }
        }
        // Streaming Slice 3: bit-equal admitted-stream maps across
        // repeats. Keyed by request_id. With --stream-all OFF or no
        // completion-freed admission, both maps are empty and this
        // block is a no-op.
        if (admitted_streamed_tokens.size()
            != vstate.first_admitted_streamed_tokens.size()) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "iter %d: admitted_streamed_tokens.size()=%zu != "
                "iter 0 size=%zu", r,
                admitted_streamed_tokens.size(),
                vstate.first_admitted_streamed_tokens.size());
            fail_with(buf); return false;
        }
        for (const auto & kv : admitted_streamed_tokens) {
            const int32_t rid = kv.first;
            auto it0 = vstate.first_admitted_streamed_tokens.find(rid);
            if (it0 == vstate.first_admitted_streamed_tokens.end()) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: admitted streamed req %d present "
                    "in iter %d but absent from iter 0",
                    r, rid, r);
                fail_with(buf); return false;
            }
            if (kv.second != it0->second) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: admitted_streamed_tokens[req=%d] "
                    "differs from iter 0 (size %zu vs %zu)",
                    r, rid, kv.second.size(), it0->second.size());
                fail_with(buf); return false;
            }
            auto itc  = admitted_streamed_close.find(rid);
            auto itc0 = vstate.first_admitted_streamed_close.find(rid);
            if (itc == admitted_streamed_close.end()
                || itc0 == vstate.first_admitted_streamed_close.end()
                || itc->second != itc0->second) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "iter %d: admitted_streamed_close[req=%d] "
                    "differs from iter 0", r, rid);
                fail_with(buf); return false;
            }
        }
        fprintf(stdout, "iter[%d] determinism: matches iter 0\n", r);
    }
    fflush(stdout);

    return true;
}
