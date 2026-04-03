/* hpx-bench/test_barrier_churn.c
 *
 * Targeted stress test for barrier recreation / thread-count churn on one
 * persistent ggml threadpool.
 *
 * Kept separate from test_ggml_threadpool.c because this can be slow on HPX.
 *
 * Optional env:
 *   RESULTS_CSV   -> append one summary row to this CSV file
 *   BACKEND_LABEL -> string written into CSV/log summary (e.g. pthread / hpx)
 *
 * Usage:
 *   ./test_barrier_churn            # defaults: iters=100, chain_len=6
 *   ./test_barrier_churn 500 6
 */

#include "ggml-cpu.h"
#include "ggml.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static void pass(const char * name) {
    printf("  PASS  %s\n", name);
}

static void fail(const char * name) {
    printf("  FAIL  %s\n", name);
    exit(1);
}

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ull + (uint64_t) ts.tv_nsec;
}

static FILE * open_csv_if_requested(void) {
    const char * path = getenv("RESULTS_CSV");
    if (!path || !*path) {
        return NULL;
    }

    FILE * fp = fopen(path, "a");
    if (!fp) {
        fprintf(stderr, "warning: failed to open RESULTS_CSV=%s\n", path);
        return NULL;
    }

    long pos = ftell(fp);
    if (pos == 0) {
        fprintf(fp,
                "backend,test_name,iters,chain_len,max_threads,total_ms,ms_per_dispatch,status,fail_iter,fail_threads,"
                "got,expected\n");
        fflush(fp);
    }

    return fp;
}

static void csv_write_summary(FILE *       fp,
                              const char * backend,
                              int          iters,
                              int          chain_len,
                              int          max_threads,
                              double       total_ms,
                              const char * status,
                              int          fail_iter,
                              int          fail_threads,
                              float        got,
                              float        expected) {
    if (!fp) {
        return;
    }

    fprintf(fp, "%s,test_barrier_churn,%d,%d,%d,%.3f,%.6f,%s,%d,%d,%.6f,%.6f\n", backend ? backend : "unknown", iters,
            chain_len, max_threads, total_ms, iters > 0 ? total_ms / (double) iters : 0.0, status, fail_iter,
            fail_threads, got, expected);
    fflush(fp);
}

/* Run a chain of `chain_len` sequential adds on the same tensor.
 * Graph has `chain_len` nodes -> `chain_len` barrier cycles per dispatch.
 * Result should be initial_val * (chain_len + 1).
 */
static float run_add_chain(int n, float val, int chain_len, int n_threads, struct ggml_threadpool * tp) {
    size_t                  mem  = (size_t) (n * sizeof(float) * (chain_len + 2)) + 2 * 1024 * 1024;
    struct ggml_init_params init = { mem, NULL, false };
    struct ggml_context *   ctx  = ggml_init(init);

    struct ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_set_f32(a, val);

    struct ggml_tensor * cur = a;
    for (int i = 0; i < chain_len; i++) {
        cur = ggml_add(ctx, cur, a);
    }

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, cur);

    struct ggml_cplan plan = ggml_graph_plan(gf, n_threads, tp);
    uint8_t *         work = NULL;
    if (plan.work_size > 0) {
        work           = (uint8_t *) malloc(plan.work_size);
        plan.work_data = work;
    }

    enum ggml_status st = ggml_graph_compute(gf, &plan);
    if (st != GGML_STATUS_SUCCESS) {
        free(work);
        ggml_free(ctx);
        return NAN;
    }

    float sum      = 0.0f;
    int   sample_n = n < 16 ? n : 16;
    for (int i = 0; i < sample_n; i++) {
        sum += ggml_get_f32_1d(cur, i);
    }

    free(work);
    ggml_free(ctx);
    return sum / sample_n;
}

int main(int argc, char ** argv) {
    int iters     = 100;
    int chain_len = 6;

    if (argc >= 2) {
        iters = atoi(argv[1]);
        if (iters < 1) {
            iters = 1;
        }
    }
    if (argc >= 3) {
        chain_len = atoi(argv[2]);
        if (chain_len < 1) {
            chain_len = 1;
        }
    }

    const int   N        = 16 * 1024;
    const float val      = 1.0f;
    const float expected = (float) (chain_len + 1) * val;

    int max_t = (int) sysconf(_SC_NPROCESSORS_ONLN);
    if (max_t < 4) {
        max_t = 4;
    }
    if (max_t > 10) {
        max_t = 10; /* keep runtime bounded */
    }

    const char * backend = getenv("BACKEND_LABEL");
    if (!backend || !*backend) {
        backend = "unknown";
    }

    FILE * csv_fp = open_csv_if_requested();

    printf("\n=== barrier/thread-count churn ===\n");
    printf("backend   : %s\n", backend);
    printf("iters     : %d\n", iters);
    printf("chain_len : %d\n", chain_len);
    printf("max_t     : %d\n", max_t);
    printf("N         : %d\n", N);
    printf("expected  : %.3f\n\n", expected);
    fflush(stdout);

    struct ggml_threadpool_params p  = ggml_threadpool_params_default(max_t);
    struct ggml_threadpool *      tp = ggml_threadpool_new(&p);
    if (!tp) {
        csv_write_summary(csv_fp, backend, iters, chain_len, max_t, 0.0, "tp_new_failed", -1, -1, NAN, expected);
        if (csv_fp) {
            fclose(csv_fp);
        }
        fail("threadpool_new returned NULL");
    }

    uint64_t t0             = now_ns();
    int      progress_every = iters <= 100 ? 10 : 100;

    for (int i = 0; i < iters; ++i) {
        int t;
        switch (i % 8) {
            case 0:
                t = 1;
                break;
            case 1:
                t = max_t;
                break;
            case 2:
                t = 2;
                break;
            case 3:
                t = max_t - 1;
                break;
            case 4:
                t = 4;
                break;
            case 5:
                t = max_t;
                break;
            case 6:
                t = 3;
                break;
            default:
                t = 1;
                break;
        }

        float result = run_add_chain(N, val, chain_len, t, tp);
        if (isnan(result)) {
            double total_ms = (double) (now_ns() - t0) / 1e6;
            printf("  FAIL  iter=%d t=%d graph_compute returned non-success\n", i, t);
            fflush(stdout);
            ggml_threadpool_free(tp);
            csv_write_summary(csv_fp, backend, iters, chain_len, max_t, total_ms, "graph_compute_failed", i, t, NAN,
                              expected);
            if (csv_fp) {
                fclose(csv_fp);
            }
            fail("graph_compute failed during churn");
        }

        if (fabsf(result - expected) >= 1e-2f) {
            double total_ms = (double) (now_ns() - t0) / 1e6;
            printf("  FAIL  iter=%d t=%d got=%f expected=%f\n", i, t, result, expected);
            fflush(stdout);
            ggml_threadpool_free(tp);
            csv_write_summary(csv_fp, backend, iters, chain_len, max_t, total_ms, "value_mismatch", i, t, result,
                              expected);
            if (csv_fp) {
                fclose(csv_fp);
            }
            fail("barrier/thread-count churn");
        }

        if ((i + 1) % progress_every == 0 || i + 1 == iters) {
            printf("  progress  %d/%d\n", i + 1, iters);
            fflush(stdout);
        }
    }

    double total_ms = (double) (now_ns() - t0) / 1e6;
    ggml_threadpool_free(tp);

    printf("\nsummary: total=%.3f ms, ms/dispatch=%.6f\n", total_ms, total_ms / (double) iters);
    fflush(stdout);

    csv_write_summary(csv_fp, backend, iters, chain_len, max_t, total_ms, "pass", -1, -1, expected, expected);

    if (csv_fp) {
        fclose(csv_fp);
    }

    pass("barrier/thread-count churn");
    printf("\n");
    return 0;
}
