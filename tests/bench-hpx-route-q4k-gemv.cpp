// HPX-route standalone benchmark for the same q4_K_8x8_q8_K gemv path the
// baseline bench (`bench-cpu-repack-q4k-gemv.cpp`) measures.  The kernel body
// is identical; only the per-chunk dispatcher changes:
//
//   baseline path:  ggml's threadpool — workers wake from spin barrier,
//                   pick chunks via ggml_threadpool_chunk_set + atomic_fetch_add,
//                   ggml_barrier at the end.
//
//   HPX path:       hpx::experimental::for_loop(par, 0, nchunk0, ...) over
//                   the same chunk grid.  Quantize step still runs serially
//                   on the calling thread (matches what ggml does for ne11=1
//                   — only ith=0 runs from_float, others spin in barrier).
//
// Chunk-size formula and per-chunk pointer math mirror repack.cpp:4314-4345
// and forward_mul_mat_one_chunk exactly, so worker count and dispatch
// granularity are apples-to-apples with the baseline at the same n_workers.
//
// What this answers: at the same chunk granularity, does HPX schedule the
// gemv kernel with lower per-iter time than ggml's threadpool?
//
// Reality checks before timing:
//   - W allocated through CPU_REPACK; W->extra populated; trait name is
//     "q4_K_8x8_q8_K".
//   - First HPX-driven iter's output bit-equal to ggml CPU backend reference
//     at 8 sentinel positions.  (Same kernel, same wdata, same W → must
//     match exactly.)

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <hpx/hpx_start.hpp>
#include <hpx/hpx_finalize.hpp>
#include <hpx/algorithm.hpp>
#include <hpx/execution.hpp>
#include <hpx/future.hpp>
#include <hpx/runtime.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

// Plain C++ linkage (matches repack.h:11):
ggml_backend_buffer_type_t ggml_backend_cpu_repack_buffer_type(void);

extern "C" {
    void quantize_row_q4_K(const float * x, void * y, int64_t k);
    void quantize_row_q8_K(const float * x, void * y, int64_t k);
    void ggml_gemv_q4_K_8x8_q8_K(int n, float * s, size_t bs,
                                 const void * vx, const void * vy,
                                 int nrows, int nc);
    const char * ggml_repack_extra_traits_name(const struct ggml_tensor * op);
}

namespace {

constexpr int     kRows       = 1;
constexpr int     kWarmupIter = 10;
constexpr int     kTimedIter  = 1000;
constexpr int64_t kNBCols     = 8;     // q4_K_8x8_q8_K trait NB_COLS

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

double percentile(std::vector<double> sorted, double p)
{
    if (sorted.empty()) return 0.0;
    size_t idx = static_cast<size_t>(p * static_cast<double>(sorted.size() - 1));
    return sorted[idx];
}

// HPX is process-singleton: hpx::start() once for the lifetime of the binary.
// We pin the thread count via --hpx:threads=N and the scheduler to "static" to
// match ggml-hpx-runtime.cpp's production setup.
void start_hpx_once(int n_workers)
{
    static std::once_flag once;
    static int            started_with = 0;
    std::call_once(once, [&]() {
        // Static buffers — HPX's argv contract requires writable storage.
        static char  arg0[]   = "bench-hpx-route-q4k-gemv";
        static char  arg1[]   = "--hpx:queuing=static";
        static char  arg2[64] = {};
        std::snprintf(arg2, sizeof(arg2), "--hpx:threads=%d", n_workers);
        static char * hpx_argv[] = {arg0, arg1, arg2};

        hpx::start(nullptr, 3, hpx_argv);
        started_with = n_workers;

        std::atexit([]() {
            hpx::post([]() { hpx::finalize(); });
            hpx::stop();
        });
    });

    if (started_with != n_workers) {
        std::fprintf(stderr,
            "WARNING: HPX already started with %d workers; requested %d ignored\n",
            started_with, n_workers);
    }
}

// Mirror of repack.cpp:4314-4345 — compute (nchunk0, dr0) for a given
// (nr0, n_workers).  Returns {nchunk0, dr0}.
std::pair<int64_t, int64_t> chunk_grid(int64_t nr0, int n_workers)
{
    const int64_t min_chunk_size = kNBCols;

    int     nth_scaled  = n_workers * 4;
    int64_t chunk_size0 = (nr0 + nth_scaled - 1) / nth_scaled;
    int64_t nchunk0     = (nr0 + chunk_size0 - 1) / chunk_size0;

    if (nchunk0 > 0 && (nr0 / nchunk0) < min_chunk_size && nr0 >= min_chunk_size) {
        nchunk0 = (nr0 + min_chunk_size - 1) / min_chunk_size;
    }

    int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
    if (n_workers == 1
        || (nchunk0 < n_workers
            && (nr0 + n_workers - 1) / n_workers >= min_chunk_size))
    {
        nchunk0 = n_workers;
        dr0     = (nr0 + nchunk0 - 1) / nchunk0;
    }

    const int64_t max_nchunk = (nr0 + min_chunk_size - 1) / min_chunk_size;
    nchunk0 = std::min(nchunk0, max_nchunk);
    return {nchunk0, dr0};
}

}    // namespace

enum class Mode { Bridge, InHpx };

int main(int argc, char ** argv)
{
    int     n_workers = 4;
    int64_t cols      = 2048;
    int64_t out_cols  = 5632;
    Mode    mode      = Mode::Bridge;
    if (argc >= 2) { n_workers = std::atoi(argv[1]); if (n_workers < 1) n_workers = 1; }
    if (argc >= 3) { cols      = std::atoll(argv[2]); }
    if (argc >= 4) { out_cols  = std::atoll(argv[3]); }
    if (argc >= 5) {
        if (std::strcmp(argv[4], "bridge") == 0) {
            mode = Mode::Bridge;
        } else if (std::strcmp(argv[4], "in_hpx") == 0) {
            mode = Mode::InHpx;
        } else {
            std::fprintf(stderr, "FATAL: unknown mode '%s' (expected: bridge | in_hpx)\n", argv[4]);
            return 1;
        }
    }
    const char * mode_str = (mode == Mode::Bridge) ? "bridge" : "in_hpx";

    if (cols % 256 != 0) {
        std::fprintf(stderr, "cols (%lld) must be a multiple of QK_K=256\n",
            (long long) cols);
        return 1;
    }
    if (out_cols % kNBCols != 0) {
        std::fprintf(stderr, "out_cols (%lld) must be a multiple of NB_COLS=%lld\n",
            (long long) out_cols, (long long) kNBCols);
        return 1;
    }

    ggml_init_params ip{};
    ip.mem_size   = 4 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) { std::fprintf(stderr, "ggml_init failed\n"); return 1; }

    ggml_tensor * W = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, cols, out_cols);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,  cols, kRows);
    ggml_tensor * y = ggml_mul_mat(ctx, W, x);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);

    ggml_backend_buffer_type_t buft = ggml_backend_cpu_repack_buffer_type();
    if (!buft) {
        std::fprintf(stderr, "CPU_REPACK buffer type unavailable\n");
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

    // Deterministic F32 weight → quantize to Q4_K → push through repack buffer.
    std::vector<float> W_f32(static_cast<size_t>(cols * out_cols));
    for (int64_t j = 0; j < out_cols; ++j) {
        for (int64_t k = 0; k < cols; ++k) {
            W_f32[static_cast<size_t>(j * cols + k)] =
                0.01f * static_cast<float>((j + 1) * ((k % 17) - 8));
        }
    }
    const size_t w_row_bytes = ggml_row_size(GGML_TYPE_Q4_K, cols);
    std::vector<uint8_t> W_q4k(w_row_bytes * static_cast<size_t>(out_cols));
    for (int64_t j = 0; j < out_cols; ++j) {
        quantize_row_q4_K(
            W_f32.data() + static_cast<size_t>(j * cols),
            W_q4k.data() + static_cast<size_t>(j) * w_row_bytes,
            cols);
    }
    ggml_backend_tensor_set(W, W_q4k.data(), 0, w_bytes);

    std::vector<float> x_data    (static_cast<size_t>(cols * kRows));
    std::vector<float> y_ref_data(static_cast<size_t>(out_cols * kRows), 0.0f);
    std::vector<float> y_hpx_data(static_cast<size_t>(out_cols * kRows), 0.0f);
    for (int64_t k = 0; k < cols; ++k) {
        x_data[static_cast<size_t>(k)] = 0.02f * static_cast<float>((k % 13) - 6);
    }
    x->data = x_data.data();
    y->data = y_ref_data.data();

    // Reality check on the harness.
    if (W->extra == nullptr) {
        std::fprintf(stderr,
            "FATAL: W->extra is null after CPU_REPACK alloc\n");
        return 1;
    }
    const char * trait = ggml_repack_extra_traits_name(y);
    if (!trait || std::strcmp(trait, "q4_K_8x8_q8_K") != 0) {
        std::fprintf(stderr,
            "FATAL: expected trait q4_K_8x8_q8_K, got %s\n",
            trait ? trait : "<null>");
        return 1;
    }

    // Reference output via ggml CPU backend (same kernel, ggml threadpool).
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) {
        std::fprintf(stderr, "ggml_backend_cpu_init failed\n");
        return 1;
    }
    ggml_backend_cpu_set_n_threads(cpu, n_workers);
    if (ggml_backend_graph_compute(cpu, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "reference graph_compute failed\n");
        return 1;
    }
    // y_ref_data is now populated. Re-target y to the HPX output buffer.
    y->data = y_hpx_data.data();

    // Chunk grid (mirrors ggml exactly).
    const int64_t nr0 = ggml_nrows(W);
    auto chunk_pair = chunk_grid(nr0, n_workers);
    const int64_t nchunk0 = chunk_pair.first;
    const int64_t dr0     = chunk_pair.second;
    const size_t       q8k_row = ggml_row_size(GGML_TYPE_Q8_K, cols);
    std::vector<uint8_t> wdata(q8k_row);

    // Start HPX with the requested worker count.  Singleton; first call wins.
    start_hpx_once(n_workers);

    // Query effective HPX OS-thread count.  hpx::get_num_worker_threads must
    // run inside HPX context, so bridge via async like the for_loop dispatch.
    std::size_t hpx_workers = hpx::async([]() {
        return hpx::get_num_worker_threads();
    }).get();

    std::fprintf(stderr,
        "shape: cols=%lld out_cols=%lld rows=%lld\n"
        "requested n_workers: %d, effective HPX worker threads: %zu\n"
        "type: q4_K, buft: %s, trait: %s, extra: %p\n"
        "chunk grid: nchunk0=%lld dr0=%lld (mirrors repack.cpp formula)\n"
        "mode: %s (bridge = hpx::async(.).get() per iter; in_hpx = single async around warmup/timed loops)\n"
        "warmup: %d iters, timing: %d iters\n",
        (long long) cols, (long long) out_cols, (long long) kRows,
        n_workers, hpx_workers,
        ggml_backend_buft_name(ggml_backend_buffer_get_type(W->buffer)),
        trait,
        W->extra,
        (long long) nchunk0, (long long) dr0,
        mode_str,
        kWarmupIter, kTimedIter);

    if (hpx_workers != static_cast<std::size_t>(n_workers)) {
        std::fprintf(stderr,
            "FATAL: requested %d HPX workers but runtime reports %zu — refusing to time\n",
            n_workers, hpx_workers);
        return 1;
    }

    // Bare iter body.  Must execute on an HPX worker thread because
    // hpx::experimental::for_loop is not callable from a native OS thread.
    // The two modes below differ only in *when* we enter HPX context:
    //   bridge: hpx::async([&]{ iter_body(); }).get() per iter
    //   in_hpx: one outer hpx::async wraps the whole warmup or timed loop,
    //           iter_body is called directly from inside that worker.
    auto iter_body = [&]() {
        // Step 1: serial F32→Q8_K (matches ggml ith=0-only path for ne11=1).
        quantize_row_q8_K(x_data.data(), wdata.data(), cols);
        // Step 2: parallel gemv chunks via HPX.
        hpx::experimental::for_loop(
            hpx::execution::par,
            int64_t{0}, nchunk0,
            [&](int64_t chunk_idx) {
                int64_t src0_start = dr0 * chunk_idx;
                int64_t src0_end   = std::min(src0_start + dr0, nr0);
                src0_start = (src0_start % kNBCols)
                    ? src0_start + kNBCols - (src0_start % kNBCols)
                    : src0_start;
                src0_end = (src0_end % kNBCols)
                    ? src0_end + kNBCols - (src0_end % kNBCols)
                    : src0_end;
                src0_end = std::min(src0_end, nr0);
                if (src0_start >= src0_end) return;

                ggml_gemv_q4_K_8x8_q8_K(
                    /* n     */ static_cast<int>(cols),
                    /* s     */ y_hpx_data.data() + src0_start,
                    /* bs    */ static_cast<size_t>(out_cols),
                    /* vx    */ static_cast<const char *>(W->data) + src0_start * W->nb[1],
                    /* vy    */ wdata.data(),
                    /* nrows */ 1,
                    /* nc    */ static_cast<int>(src0_end - src0_start));
            });
    };

    // Warmup.
    if (mode == Mode::Bridge) {
        for (int i = 0; i < kWarmupIter; ++i) {
            hpx::async([&]() { iter_body(); }).get();
        }
    } else {
        hpx::async([&]() {
            for (int i = 0; i < kWarmupIter; ++i) iter_body();
        }).get();
    }

    // Correctness check at sentinel positions.  y_hpx_data is regular memory;
    // writes by the HPX workers are visible here after .get() synchronizes.
    const int64_t sentinels[] = {0, 1, 7, 8, 100, out_cols / 2, out_cols - 8, out_cols - 1};
    int checked = 0;
    for (int64_t j : sentinels) {
        if (j < 0 || j >= out_cols) continue;
        const float a = y_hpx_data[static_cast<size_t>(j)];
        const float b = y_ref_data[static_cast<size_t>(j)];
        if (a != b) {
            std::fprintf(stderr,
                "FATAL: HPX/ref mismatch at j=%lld: hpx=%.6g ref=%.6g (delta=%.3g)\n",
                (long long) j, (double) a, (double) b, (double) (a - b));
            return 1;
        }
        ++checked;
    }
    std::fprintf(stderr, "correctness: HPX matches ggml CPU reference at %d/%d sentinels\n",
        checked, checked);

    // Timed.  In bridge mode, each iter pays one hpx::async hop.  In in_hpx
    // mode, the whole timed loop runs inside one HPX worker; per-iter timing
    // is taken inside that worker, so each iter measures pure for_loop
    // dispatch with no thread-hop.
    std::vector<double> us;
    us.reserve(static_cast<size_t>(kTimedIter));
    if (mode == Mode::Bridge) {
        for (int i = 0; i < kTimedIter; ++i) {
            auto t0 = std::chrono::steady_clock::now();
            hpx::async([&]() { iter_body(); }).get();
            auto t1 = std::chrono::steady_clock::now();
            us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        }
    } else {
        hpx::async([&]() {
            for (int i = 0; i < kTimedIter; ++i) {
                auto t0 = std::chrono::steady_clock::now();
                iter_body();
                auto t1 = std::chrono::steady_clock::now();
                us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
            }
        }).get();
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
        mn, med, p90, m, s, 100.0 * s / m, mx,
        static_cast<double>(y_hpx_data[0]));

    // CSV row to stdout.
    // Columns: route,mode,n_workers,iters,cols,out_cols,rows,trait,min,median,p90,mean,stdev
    std::fprintf(stdout,
        "hpx,%s,%d,%d,%lld,%lld,%lld,%s,%.3f,%.3f,%.3f,%.3f,%.3f\n",
        mode_str, n_workers, kTimedIter,
        (long long) cols, (long long) out_cols, (long long) kRows,
        trait,
        mn, med, p90, m, s);

    ggml_backend_free(cpu);
    ggml_backend_buffer_free(W_buf);
    ggml_free(ctx);
    return 0;
}
