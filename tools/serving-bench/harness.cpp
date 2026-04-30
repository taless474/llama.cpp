#include "harness.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>

namespace serving_bench {

namespace {

double percentile(std::vector<double> values, double p) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    if (p <= 0.0) {
        return values.front();
    }
    if (p >= 1.0) {
        return values.back();
    }
    const double idx = p * static_cast<double>(values.size() - 1);
    const size_t lo  = static_cast<size_t>(std::floor(idx));
    const size_t hi  = static_cast<size_t>(std::ceil (idx));
    const double f   = idx - static_cast<double>(lo);
    return values[lo] * (1.0 - f) + values[hi] * f;
}

double coefficient_of_variation(const std::vector<double> & values) {
    if (values.size() < 2) {
        return 0.0;
    }
    double sum = 0.0;
    for (double v : values) {
        sum += v;
    }
    const double mean = sum / static_cast<double>(values.size());
    if (mean <= 0.0) {
        return 0.0;
    }
    double var = 0.0;
    for (double v : values) {
        var += (v - mean) * (v - mean);
    }
    var /= static_cast<double>(values.size());
    return std::sqrt(var) / mean;
}

void print_usage(FILE * out) {
    std::fprintf(out,
        "usage: llama-serving-bench [options]\n"
        "  -m, --model PATH         model file\n"
        "  -p, --prompt STR         prompt\n"
        "      --n-contexts N       pooled contexts (default 1)\n"
        "      --n-concurrent N     concurrent in-flight (default 1)\n"
        "      --n-requests N       total requests (default 1)\n"
        "  -n, --max-tokens N       max generated tokens (default 64)\n"
        "      --n-threads N        per-context kernel threads (0=auto)\n"
        "      --seed-base N        RNG seed base (default 1234)\n"
        "  -h, --help               show this message and exit\n");
}

} // namespace

aggregate_metrics compute_aggregate(
    const std::vector<request_result> & results,
    std::chrono::nanoseconds            wall_time) {
    aggregate_metrics m{};
    m.wall_time = wall_time;

    std::vector<double> ttft_ms;
    std::vector<double> total_ms;
    std::vector<double> per_req_tps;

    int64_t total_tokens = 0;

    for (const auto & r : results) {
        switch (r.status) {
            case request_status::ok:        m.n_ok++;        break;
            case request_status::cancelled: m.n_cancelled++; break;
            case request_status::error:     m.n_error++;     break;
        }
        if (r.status != request_status::ok) {
            continue;
        }
        ttft_ms .push_back(static_cast<double>(r.ttft .count()) / 1.0e6);
        total_ms.push_back(static_cast<double>(r.total.count()) / 1.0e6);
        total_tokens += r.n_tokens_generated;
        if (r.total.count() > 0) {
            per_req_tps.push_back(
                static_cast<double>(r.n_tokens_generated) /
                (static_cast<double>(r.total.count()) / 1.0e9));
        }
    }

    const double wall_s = static_cast<double>(wall_time.count()) / 1.0e9;
    m.aggregate_tokens_per_sec = wall_s > 0.0
        ? static_cast<double>(total_tokens) / wall_s
        : 0.0;

    m.ttft_ms_p50  = percentile(ttft_ms,  0.50);
    m.ttft_ms_p95  = percentile(ttft_ms,  0.95);
    m.ttft_ms_p99  = percentile(ttft_ms,  0.99);
    m.total_ms_p50 = percentile(total_ms, 0.50);
    m.total_ms_p95 = percentile(total_ms, 0.95);
    m.total_ms_p99 = percentile(total_ms, 0.99);
    m.per_request_tps_cv = coefficient_of_variation(per_req_tps);

    return m;
}

void print_aggregate(FILE * out, const aggregate_metrics & m) {
    std::fprintf(out, "[serving-bench] n_ok=%d n_cancelled=%d n_error=%d\n",
        m.n_ok, m.n_cancelled, m.n_error);
    std::fprintf(out, "[serving-bench] wall=%.3f s  agg_tok/s=%.2f\n",
        static_cast<double>(m.wall_time.count()) / 1.0e9,
        m.aggregate_tokens_per_sec);
    std::fprintf(out, "[serving-bench] ttft_ms  p50=%.2f p95=%.2f p99=%.2f\n",
        m.ttft_ms_p50, m.ttft_ms_p95, m.ttft_ms_p99);
    std::fprintf(out, "[serving-bench] total_ms p50=%.2f p95=%.2f p99=%.2f\n",
        m.total_ms_p50, m.total_ms_p95, m.total_ms_p99);
    std::fprintf(out, "[serving-bench] per_req_tps_cv=%.4f\n",
        m.per_request_tps_cv);
}

parse_result parse_cli(int argc, char ** argv, harness_config & out) {
    out = {};
    out.n_contexts        = 1;
    out.n_concurrent      = 1;
    out.n_requests        = 1;
    out.max_tokens        = 64;
    out.n_threads_per_ctx = 0;
    out.seed_base         = 1234;

    auto need = [&](int & idx) -> const char * {
        if (idx + 1 >= argc) {
            std::fprintf(stderr, "missing value for %s\n", argv[idx]);
            return nullptr;
        }
        return argv[++idx];
    };

    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--help" || a == "-h") {
            print_usage(stdout);
            return parse_result::help;
        } else if (a == "--model" || a == "-m") {
            const char * v = need(i); if (!v) return parse_result::error;
            out.model_path = v;
        } else if (a == "--prompt" || a == "-p") {
            const char * v = need(i); if (!v) return parse_result::error;
            out.prompt = v;
        } else if (a == "--n-contexts") {
            const char * v = need(i); if (!v) return parse_result::error;
            out.n_contexts = std::atoi(v);
        } else if (a == "--n-concurrent") {
            const char * v = need(i); if (!v) return parse_result::error;
            out.n_concurrent = std::atoi(v);
        } else if (a == "--n-requests") {
            const char * v = need(i); if (!v) return parse_result::error;
            out.n_requests = std::atoi(v);
        } else if (a == "--max-tokens" || a == "-n") {
            const char * v = need(i); if (!v) return parse_result::error;
            out.max_tokens = std::atoi(v);
        } else if (a == "--n-threads") {
            const char * v = need(i); if (!v) return parse_result::error;
            out.n_threads_per_ctx = std::atoi(v);
        } else if (a == "--seed-base") {
            const char * v = need(i); if (!v) return parse_result::error;
            out.seed_base = static_cast<uint32_t>(std::atoll(v));
        } else {
            std::fprintf(stderr, "unknown arg: %s\n", a.c_str());
            return parse_result::error;
        }
    }

    return parse_result::ok;
}

void print_config(FILE * out, const harness_config & c) {
    std::fprintf(out, "[serving-bench] config:\n");
    std::fprintf(out, "  model              = %s\n", c.model_path.empty() ? "(unset)" : c.model_path.c_str());
    std::fprintf(out, "  prompt             = %s\n", c.prompt.empty()     ? "(unset)" : c.prompt.c_str());
    std::fprintf(out, "  n_contexts         = %d\n", c.n_contexts);
    std::fprintf(out, "  n_concurrent       = %d\n", c.n_concurrent);
    std::fprintf(out, "  n_requests         = %d\n", c.n_requests);
    std::fprintf(out, "  max_tokens         = %d\n", c.max_tokens);
    std::fprintf(out, "  n_threads_per_ctx  = %d%s\n",
        c.n_threads_per_ctx, c.n_threads_per_ctx == 0 ? " (auto)" : "");
    std::fprintf(out, "  seed_base          = %u\n", c.seed_base);
}

} // namespace serving_bench
