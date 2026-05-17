// Gate-validation surface for the HPX continuous-batch gate.
//
// `run_validation` is invoked by main exactly once per `--repeat`
// iteration. It walks the per-result snapshots, partition tables,
// stream gates, slice-strict gates, admission and arrival-source
// invariants, residual-KV print, status_summary print, descriptive
// metrics block, and (for r > 0) the determinism cross-check.
//
// Failure ownership:
//   - On any gate failure the function calls `gate_emit::emit_fail`
//     with the same reason string the previous in-main `fail_with`
//     lambda used, then returns `false`.
//   - Main owns llama context/model teardown and the
//     `hpx_runtime::stop()` call. The validator does NOT touch those.
//   - On PASS the function prints the same per-iter audit / metrics /
//     determinism lines the in-main block used and returns `true`.
//
// Determinism state:
//   - `validation_state` snapshots the r==0 results so that r > 0
//     iterations can compare against them. Main owns one
//     `validation_state` for the lifetime of the per-repeat loop and
//     passes it by mutable reference on every call.

#pragma once

#include "cli.h"
#include "types.h"

#include <hpx/hpx.hpp>

#include <cstdint>
#include <unordered_map>
#include <vector>

struct validation_state {
    std::vector<request_result>                                 first_results;
    std::vector<std::vector<int32_t>>                           first_streamed_tokens;
    std::vector<stream_close_reason>                            first_streamed_close;
    std::unordered_map<int32_t, std::vector<int32_t>>           first_admitted_streamed_tokens;
    std::unordered_map<int32_t, stream_close_reason>            first_admitted_streamed_close;
};

bool run_validation(
    const cli_args &                                                          args,
    int32_t                                                                   repeat_index,
    int32_t                                                                   n_prompt_tokens,
    const std::vector<int32_t> &                                              budgets,
    const engine_result &                                                     er,
    std::vector<hpx::future<request_result>> &                                futs,
    int32_t                                                                   initial_futures,
    const std::vector<std::vector<int32_t>> &                                 streamed_tokens,
    const std::vector<stream_close_reason> &                                  streamed_close,
    const std::vector<bool> &                                                 streamed_seen,
    const std::unordered_map<int32_t, std::vector<int32_t>> &                 admitted_streamed_tokens,
    const std::unordered_map<int32_t, stream_close_reason> &                  admitted_streamed_close,
    const std::unordered_map<int32_t, bool> &                                 admitted_streamed_seen,
    validation_state &                                                        vstate);
