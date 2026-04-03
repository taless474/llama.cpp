#include "ggml-cpu.h"
#include "ggml.h"

#include <errno.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

struct bench_case {
    struct ggml_context *    ctx;
    struct ggml_cgraph *     gf;
    struct ggml_cplan        plan;
    struct ggml_threadpool * tp;
    uint8_t *                work;
};

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ull + (uint64_t) ts.tv_nsec;
}

static void die(const char * msg) {
    fprintf(stderr, "ERROR: %s\n", msg);
    fflush(stderr);
    exit(1);
}

static void ensure_dir(const char * path) {
    if (mkdir(path, 0777) != 0 && errno != EEXIST) {
        perror("mkdir");
        die("failed to create results directory");
    }
}

/* Tiny repeated-add graph.
 * Graph has `chain_len` nodes, so each dispatch includes multiple node barriers.
 */
static struct bench_case build_add_chain_case(int n, int chain_len, int n_threads, bool disposable_pool) {
    struct bench_case bc;
    memset(&bc, 0, sizeof(bc));

    size_t                  mem  = (size_t) (n * sizeof(float) * (chain_len + 2)) + 2 * 1024 * 1024;
    struct ggml_init_params init = {
        .mem_size   = mem,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };

    bc.ctx = ggml_init(init);
    if (!bc.ctx) {
        die("ggml_init failed");
    }

    struct ggml_tensor * a = ggml_new_tensor_1d(bc.ctx, GGML_TYPE_F32, n);
    ggml_set_f32(a, 1.0f);

    struct ggml_tensor * cur = a;
    for (int i = 0; i < chain_len; ++i) {
        cur = ggml_add(bc.ctx, cur, a);
    }

    bc.gf = ggml_new_graph(bc.ctx);
    ggml_build_forward_expand(bc.gf, cur);

    if (!disposable_pool) {
        struct ggml_threadpool_params p = ggml_threadpool_params_default(n_threads);
        bc.tp                           = ggml_threadpool_new(&p);
    } else {
        bc.tp = NULL;
    }

    bc.plan = ggml_graph_plan(bc.gf, n_threads, bc.tp);

    if (disposable_pool) {
        bc.plan.threadpool = NULL;
    }

    if (bc.plan.work_size > 0) {
        bc.work = (uint8_t *) malloc(bc.plan.work_size);
        if (!bc.work) {
            die("malloc(work) failed");
        }
        bc.plan.work_data = bc.work;
    }

    return bc;
}

static void destroy_case(struct bench_case * bc) {
    if (bc->tp) {
        ggml_threadpool_free(bc->tp);
        bc->tp = NULL;
    }
    free(bc->work);
    bc->work = NULL;
    if (bc->ctx) {
        ggml_free(bc->ctx);
        bc->ctx = NULL;
    }
}

static void
run_case(FILE * csv, const char * label, int n, int chain_len, int n_threads, int runs, bool disposable_pool) {
    struct bench_case bc = build_add_chain_case(n, chain_len, n_threads, disposable_pool);

    /* warmup */
    for (int i = 0; i < 10; ++i) {
        enum ggml_status st = ggml_graph_compute(bc.gf, &bc.plan);
        if (st != GGML_STATUS_SUCCESS) {
            destroy_case(&bc);
            die("warmup ggml_graph_compute failed");
        }
    }

    uint64_t t0 = now_ns();
    for (int i = 0; i < runs; ++i) {
        enum ggml_status st = ggml_graph_compute(bc.gf, &bc.plan);
        if (st != GGML_STATUS_SUCCESS) {
            destroy_case(&bc);
            die("ggml_graph_compute failed");
        }
    }
    uint64_t t1 = now_ns();

    double total_ms    = (double) (t1 - t0) / 1e6;
    double us_dispatch = (double) (t1 - t0) / (double) runs / 1e3;

    printf("%-16s mode=%-10s t=%2d runs=%6d n=%7d chain=%2d  total=%9.3f ms  us/dispatch=%8.3f\n", label,
           disposable_pool ? "disposable" : "reusable", n_threads, runs, n, chain_len, total_ms, us_dispatch);
    fflush(stdout);

    if (csv) {
        fprintf(csv, "%s,%s,%d,%d,%d,%d,%.3f,%.3f\n", label, disposable_pool ? "disposable" : "reusable", n_threads,
                runs, n, chain_len, total_ms, us_dispatch);
        fflush(csv);
    }

    destroy_case(&bc);
}

int main(int argc, char ** argv) {
    setvbuf(stdout, NULL, _IOLBF, 0);

    int max_t = (int) sysconf(_SC_NPROCESSORS_ONLN);
    if (max_t < 1) {
        max_t = 4;
    }

    int t = 4;
    if (argc >= 2) {
        t = atoi(argv[1]);
        if (t < 1) {
            t = 1;
        }
        if (t > max_t) {
            t = max_t;
        }
    }

    /* CSV path: use explicit arg, or auto-generate a timestamped name. */
    char auto_csv_path[256];
    const char * csv_path = NULL;
    if (argc >= 3) {
        csv_path = argv[2];
    } else {
#ifndef BENCH_BACKEND
#define BENCH_BACKEND "unknown"
#endif
        time_t now = time(NULL);
        struct tm * tm_info = localtime(&now);
        ensure_dir("results");
        strftime(auto_csv_path, sizeof(auto_csv_path),
                 "results/bench_dispatch_overhead_" BENCH_BACKEND "_%Y%m%d_%H%M%S.csv",
                 tm_info);
        csv_path = auto_csv_path;
    }

    FILE * csv = NULL;
    {
        ensure_dir("results");
        csv = fopen(csv_path, "w");
        if (!csv) {
            perror("fopen");
            die("failed to open csv output");
        }
        fprintf(csv, "label,mode,threads,runs,n,chain,total_ms,us_per_dispatch\n");
        fflush(csv);
        printf("saving CSV → %s\n", csv_path);
    }

    printf("\n=== dispatch overhead / size ladder ===\n");
    printf("threads=%d  logical_cores=%d\n\n", t, max_t);
    fflush(stdout);

    printf("── Dispatch-overhead microbenchmark ──\n");
    fflush(stdout);
    run_case(csv, "tiny", 32, 2, t, 100, false);
    run_case(csv, "tiny", 32, 2, t, 100, true);

    printf("\n── Size ladder (same reusable pool shape) ──\n");
    fflush(stdout);
    run_case(csv, "tiny", 32, 2, t, 100, false);
    run_case(csv, "small", 256, 4, t, 100, false);
    run_case(csv, "medium", 4096, 4, t, 100, false);
    run_case(csv, "large", 65536, 4, t, 100, false);

    printf("\n");
    fflush(stdout);

    if (csv) {
        fclose(csv);
    }

    return 0;
}
