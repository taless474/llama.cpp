// Standalone baseline benchmark: time the CPU_REPACK + q4_K_8x8_q8_K gemv
// path for one MUL_MAT shape, default threads, no HPX involvement.
//
// Why this exists: the 2026-04-26 baseline-execution-map work confirmed that
// real TinyLlama Q4_K_M decode never touches ggml_vec_dot_q4_K_q8_K — every
// Q4_K weight goes through the CPU_REPACK extra path and the q*_K_8x8_q8_K
// trait. To meaningfully compare any future HPX scheduling change against the
// real baseline, we need a harness that reliably exercises that exact path on
// a known shape. This is that harness.
//
// First-cut shape is the gate/up projection (cols=2048, out_cols=5632) — the
// most performance-relevant Q4_K MUL_MAT in TinyLlama (44 invocations across
// the 22 layers × 2 projections per decode step, all with rows=1).
//
// The harness asserts CPU_REPACK reality before timing:
//   W->extra != nullptr
//   ggml_repack_extra_traits_name(W) == "q4_K_8x8_q8_K"
// If either fails, the benchmark aborts. This guarantees the timed work is
// the gemv kernel we actually care about, not the dead standard vec_dot path.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// repack.h declares this with plain C++ linkage (not inside extern "C").
ggml_backend_buffer_type_t ggml_backend_cpu_repack_buffer_type(void);

extern "C" {
    void quantize_row_q4_K(const float * x, void * y, int64_t k);
    const char * ggml_repack_extra_traits_name(const struct ggml_tensor * op);
}

namespace {

constexpr int64_t kRows       = 1;
constexpr int     kWarmupIter = 10;
constexpr int     kTimedIter  = 1000;

double mean(const std::vector<double> & v)
{
    double s = 0.0;
    for (double x : v) s += x;
    return s / static_cast<double>(v.size());
}

double stdev(const std::vector<double> & v, double m)
{
    double s = 0.0;
    for (double x : v) { double d = x - m; s += d * d; }
    return std::sqrt(s / static_cast<double>(v.size()));
}

// percentile via sorted copy; rank by nearest-rank, no interpolation.
double percentile(std::vector<double> sorted, double p)
{
    if (sorted.empty()) return 0.0;
    size_t idx = static_cast<size_t>(p * static_cast<double>(sorted.size() - 1));
    return sorted[idx];
}

}    // namespace

int main(int argc, char ** argv)
{
    int     threads  = 4;
    int64_t cols     = 2048;     // tinyllama gate/up default
    int64_t out_cols = 5632;
    if (argc >= 2) { threads  = std::atoi(argv[1]); if (threads < 1) threads = 1; }
    if (argc >= 3) { cols     = std::atoll(argv[2]); }
    if (argc >= 4) { out_cols = std::atoll(argv[3]); }

    if (cols % 256 != 0) {
        std::fprintf(stderr, "cols (%lld) must be a multiple of QK_K=256\n", (long long) cols);
        return 1;
    }
    if (out_cols % 8 != 0) {
        std::fprintf(stderr, "out_cols (%lld) must be a multiple of 8 (repack tile)\n", (long long) out_cols);
        return 1;
    }

    ggml_init_params ip{};
    ip.mem_size   = 4 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        std::fprintf(stderr, "ggml_init failed\n");
        return 1;
    }

    ggml_tensor * W = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, cols, out_cols);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,  cols, kRows);
    ggml_tensor * y = ggml_mul_mat(ctx, W, x);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);

    // Allocate W through the CPU_REPACK buffer so init_tensor populates extra
    // and set_tensor repacks the data in place to the 8x8 trait layout.
    ggml_backend_buffer_type_t buft = ggml_backend_cpu_repack_buffer_type();
    if (!buft) {
        std::fprintf(stderr,
            "CPU_REPACK buffer type unavailable (build without GGML_USE_CPU_REPACK?)\n");
        ggml_free(ctx);
        return 1;
    }

    const size_t w_bytes = ggml_nbytes(W);
    ggml_backend_buffer_t W_buf = ggml_backend_buft_alloc_buffer(buft, w_bytes);
    if (!W_buf) {
        std::fprintf(stderr, "failed to allocate %zu bytes via CPU_REPACK\n", w_bytes);
        ggml_free(ctx);
        return 1;
    }

    ggml_tallocr W_alloc = ggml_tallocr_new(W_buf);
    if (ggml_tallocr_alloc(&W_alloc, W) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "ggml_tallocr_alloc(W) failed\n");
        ggml_backend_buffer_free(W_buf);
        ggml_free(ctx);
        return 1;
    }

    // Build deterministic F32 weight, quantize to standard Q4_K, then push
    // through backend_tensor_set so the repack buffer repacks in place.
    std::vector<float> W_f32(static_cast<size_t>(cols * out_cols));
    for (int64_t j = 0; j < out_cols; ++j) {
        for (int64_t k = 0; k < cols; ++k) {
            W_f32[static_cast<size_t>(j * cols + k)] =
                0.01f * static_cast<float>((j + 1) * ((k % 17) - 8));
        }
    }
    const size_t row_bytes = ggml_row_size(GGML_TYPE_Q4_K, cols);
    std::vector<uint8_t> W_q4k(row_bytes * static_cast<size_t>(out_cols));
    for (int64_t j = 0; j < out_cols; ++j) {
        quantize_row_q4_K(
            W_f32.data() + static_cast<size_t>(j * cols),
            W_q4k.data() + static_cast<size_t>(j) * row_bytes,
            cols);
    }
    ggml_backend_tensor_set(W, W_q4k.data(), 0, w_bytes);

    std::vector<float> x_data(static_cast<size_t>(cols * kRows));
    std::vector<float> y_data(static_cast<size_t>(out_cols * kRows));
    for (int64_t k = 0; k < cols; ++k) {
        x_data[static_cast<size_t>(k)] = 0.02f * static_cast<float>((k % 13) - 6);
    }
    x->data = x_data.data();
    y->data = y_data.data();

    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) {
        std::fprintf(stderr, "ggml_backend_cpu_init failed\n");
        ggml_backend_buffer_free(W_buf);
        ggml_free(ctx);
        return 1;
    }
    ggml_backend_cpu_set_n_threads(cpu, threads);

    // Reality check before timing — abort if we are not on the gemv path.
    if (W->extra == nullptr) {
        std::fprintf(stderr,
            "FATAL: W->extra is null after CPU_REPACK alloc — repack init_tensor did not fire\n");
        return 1;
    }
    // Helper takes the op tensor and looks at src[0]->extra; pass y (mul_mat).
    const char * trait = ggml_repack_extra_traits_name(y);
    if (!trait || std::strcmp(trait, "q4_K_8x8_q8_K") != 0) {
        std::fprintf(stderr,
            "FATAL: expected trait q4_K_8x8_q8_K, got %s\n",
            trait ? trait : "<null>");
        return 1;
    }

    std::fprintf(stderr,
        "shape: cols=%lld out_cols=%lld rows=%lld\n"
        "threads: %d, type: q4_K, buft: %s, trait: %s, extra: %p\n"
        "warmup: %d iters, timing: %d iters\n",
        (long long) cols, (long long) out_cols, (long long) kRows,
        threads,
        ggml_backend_buft_name(ggml_backend_buffer_get_type(W->buffer)),
        trait,
        W->extra,
        kWarmupIter, kTimedIter);

    for (int i = 0; i < kWarmupIter; ++i) {
        if (ggml_backend_graph_compute(cpu, gf) != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "warmup iter %d failed\n", i);
            return 1;
        }
    }

    std::vector<double> us;
    us.reserve(static_cast<size_t>(kTimedIter));
    for (int i = 0; i < kTimedIter; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        ggml_status st = ggml_backend_graph_compute(cpu, gf);
        auto t1 = std::chrono::steady_clock::now();
        if (st != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "timed iter %d failed\n", i);
            return 1;
        }
        us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }

    std::vector<double> sorted = us;
    std::sort(sorted.begin(), sorted.end());
    const double mn  = sorted.front();
    const double med = percentile(sorted, 0.50);
    const double p90 = percentile(sorted, 0.90);
    const double mx  = sorted.back();
    const double m   = mean(us);
    const double s   = stdev(us, m);

    std::fprintf(stderr,
        "result: min=%.2f  median=%.2f  p90=%.2f  mean=%.2f  stdev=%.2f (%.1f%%)  max=%.2f us  y[0]=%.4f\n",
        mn, med, p90, m, s, 100.0 * s / m, mx, static_cast<double>(y_data[0]));

    // CSV row to stdout — driver-friendly aggregation.
    // Columns: threads,iters,cols,out_cols,rows,trait,min_us,median_us,p90_us,mean_us,stdev_us
    std::fprintf(stdout,
        "%d,%d,%lld,%lld,%lld,%s,%.3f,%.3f,%.3f,%.3f,%.3f\n",
        threads, kTimedIter,
        (long long) cols, (long long) out_cols, (long long) kRows,
        trait,
        mn, med, p90, m, s);

    ggml_backend_free(cpu);
    ggml_backend_buffer_free(W_buf);
    ggml_free(ctx);
    return 0;
}
