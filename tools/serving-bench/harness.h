#pragma once

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace serving_bench {

enum class request_status : uint8_t {
    ok,
    cancelled,
    error,
};

// Stable, dependency-free FNV-1a 64-bit fold over generated token ids.
// Both backends MUST use these exact constants and this exact arithmetic so
// that the hashes computed on the std and HPX paths can be compared bit-for-bit.
//
// Scope: structural-fidelity check (std vs HPX) only. Not a quality, security,
// or perceptual signal. Do not repurpose.
inline constexpr uint64_t k_token_hash_init  = 0xcbf29ce484222325ull; // FNV-1a offset basis
inline constexpr uint64_t k_token_hash_empty = 0;                    // n_tokens_generated == 0

inline uint64_t fold_token_hash(uint64_t state, int32_t token_id) noexcept {
    state ^= static_cast<uint64_t>(static_cast<uint32_t>(token_id));
    state *= 0x100000001b3ull; // FNV-1a 64-bit prime
    return state;
}

struct request_params {
    std::string prompt;
    int32_t     max_tokens;
    uint32_t    seed;
    // Stable per-submission index assigned by the harness (main.cpp).
    // Used by HPX trace lines (`req[i] acquire ctx=…`, `req[i] release ctx=…`)
    // to align with the per-request `req[i]` index that main.cpp prints on
    // stdout. Engines that do not trace per-request lifecycle (e.g.,
    // backend_std) may ignore this field.
    int32_t     request_index;
};

struct request_result {
    request_status           status;
    int32_t                  n_tokens_generated;
    std::chrono::nanoseconds ttft;
    std::chrono::nanoseconds total;
    // FNV-1a 64-bit fold over generated token ids in emission order.
    // Set to k_token_hash_empty (0) when n_tokens_generated == 0, and for
    // any non-ok result. See fold_token_hash() above.
    uint64_t                 generated_token_hash = 0;
    std::string              error_message;
};

struct aggregate_metrics {
    int32_t                  n_ok;
    int32_t                  n_cancelled;
    int32_t                  n_error;
    std::chrono::nanoseconds wall_time;
    double                   aggregate_tokens_per_sec;
    double                   ttft_ms_p50;
    double                   ttft_ms_p95;
    double                   ttft_ms_p99;
    double                   total_ms_p50;
    double                   total_ms_p95;
    double                   total_ms_p99;
    double                   per_request_tps_cv;
};

struct harness_config {
    std::string model_path;
    std::string prompt;
    int32_t     n_contexts;
    int32_t     n_concurrent;
    int32_t     n_requests;
    int32_t     max_tokens;
    int32_t     n_threads_per_ctx;
    int32_t     ctx_size;
    int32_t     batch_size;
    // Engine selector. Accepted values: "std" (default), "hpx".
    // Parser must reject any other value.
    std::string backend;
    uint32_t    seed_base;
    // Optional per-request max_tokens plan. Empty means every request uses
    // cfg.max_tokens (existing behavior). Non-empty must have exactly
    // n_requests entries, each strictly positive. main.cpp's submit_one
    // selects plan[next_idx] when non-empty; the fit-check uses
    // max(plan) when non-empty.
    std::vector<int32_t> max_tokens_plan;
};

enum class parse_result : uint8_t {
    ok,
    help,
    error,
};

aggregate_metrics compute_aggregate(
    const std::vector<request_result> & results,
    std::chrono::nanoseconds            wall_time);

void print_aggregate(FILE * out, const aggregate_metrics & m);

parse_result parse_cli(int argc, char ** argv, harness_config & out);

void print_config(FILE * out, const harness_config & c);

} // namespace serving_bench
