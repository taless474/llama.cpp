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

struct request_params {
    std::string prompt;
    int32_t     max_tokens;
    uint32_t    seed;
};

struct request_result {
    request_status           status;
    int32_t                  n_tokens_generated;
    std::chrono::nanoseconds ttft;
    std::chrono::nanoseconds total;
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
    uint32_t    seed_base;
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
