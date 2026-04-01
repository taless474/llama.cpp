/* bench_mul_mat.c — Level B benchmark: ggml_mul_mat at realistic inference sizes.
 *
 * Matrix dimensions match key layers in Llama 3.1-8B (hidden=4096, ffn=14336):
 *
 *   Attention Q/K/V proj:  K=4096, M=4096
 *   FFN gate / up proj:    K=4096, M=14336
 *   FFN down proj:         K=14336, M=4096
 *
 * Two batch sizes:
 *   N=1   → token generation (one token at a time)
 *   N=512 → prompt processing (heavy prefill)
 *
 * Weights stored as F16, activations as F32 — matches the ggml CPU path for
 * dequantized inference (real Q4_K_M inference also routes through here after
 * dequant, but F16 gives a cleaner scheduler comparison without quant overhead).
 *
 * Usage: bench_mul_mat [n_threads]  (default: 4)
 */

#include "ggml-cpu.h"
#include "ggml.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>

#ifndef BENCH_BACKEND
#define BENCH_BACKEND "unknown"
#endif

static void ensure_dir(const char * path) {
    if (mkdir(path, 0777) != 0 && errno != EEXIST) {
        perror("mkdir");
        fprintf(stderr, "failed to create results directory\n");
        exit(1);
    }
}

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

/* Run one mul_mat graph for `runs` iterations on a persistent pool.
 * Returns average µs per dispatch. */
static double bench_mul_mat(int K, int M, int N, int n_threads, int warmup, int runs) {
    /* Memory: weights (F16) + activations (F32) + output (F32) + overhead */
    size_t mem = (size_t)K * M * sizeof(uint16_t)   /* A: F16 weights  */
               + (size_t)K * N * sizeof(float)       /* B: F32 activations */
               + (size_t)M * N * sizeof(float)       /* C: F32 output   */
               + 16 * 1024 * 1024;                   /* ggml overhead   */

    struct ggml_init_params init = { .mem_size = mem, .mem_buffer = NULL, .no_alloc = false };
    struct ggml_context *ctx = ggml_init(init);
    if (!ctx) { fprintf(stderr, "ggml_init failed\n"); exit(1); }

    struct ggml_tensor *A = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, K, M);
    struct ggml_tensor *B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);

    /* Fill with small values to keep numerics clean */
    for (int i = 0; i < K * M; i++) ggml_set_f32_1d(A, i,  0.125f);
    for (int i = 0; i < K * N; i++) ggml_set_f32_1d(B, i, -0.125f);

    struct ggml_tensor *C = ggml_mul_mat(ctx, A, B);
    struct ggml_cgraph *gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, C);

    struct ggml_threadpool_params pp = ggml_threadpool_params_default(n_threads);
    struct ggml_threadpool *tp = ggml_threadpool_new(&pp);

    struct ggml_cplan plan = ggml_graph_plan(gf, n_threads, tp);
    uint8_t *work = NULL;
    if (plan.work_size > 0) {
        work = (uint8_t *)malloc(plan.work_size);
        plan.work_data = work;
    }

    /* Warmup */
    for (int i = 0; i < warmup; i++) {
        ggml_graph_compute(gf, &plan);
    }

    /* Timed runs */
    uint64_t t0 = now_ns();
    for (int i = 0; i < runs; i++) {
        ggml_graph_compute(gf, &plan);
    }
    uint64_t t1 = now_ns();

    free(work);
    ggml_threadpool_free(tp);
    ggml_free(ctx);

    return (double)(t1 - t0) / 1e3 / runs;  /* µs per dispatch */
}

int main(int argc, char **argv) {
    int n_threads = 4;
    if (argc > 1) n_threads = atoi(argv[1]);

    /* Auto-generate a timestamped CSV path. */
    char csv_path[256];
    time_t now = time(NULL);
    struct tm *tm_info = localtime(&now);
    ensure_dir("results");
    strftime(csv_path, sizeof(csv_path),
             "results/bench_mul_mat_" BENCH_BACKEND "_%Y%m%d_%H%M%S.csv",
             tm_info);

    FILE *csv = fopen(csv_path, "w");
    if (!csv) { perror("fopen"); exit(1); }
    fprintf(csv, "backend,threads,layer,K,M,N,warmup,runs,us_per_dispatch\n");
    fflush(csv);
    printf("saving CSV → %s\n", csv_path);

    printf("\n=== Level B: mul_mat at realistic Llama 3.1-8B layer sizes ===\n");
    printf("threads=%d\n\n", n_threads);

    /* Layer shapes */
    struct { const char *name; int K, M; } layers[] = {
        { "attn_qkv_proj ", 4096,  4096 },
        { "ffn_gate_up    ", 4096, 14336 },
        { "ffn_down       ", 14336, 4096 },
    };
    int nlayers = (int)(sizeof(layers) / sizeof(layers[0]));

    int batch_sizes[] = { 1, 4, 32, 512 };
    int nbatch = (int)(sizeof(batch_sizes) / sizeof(batch_sizes[0]));

    printf("%-18s  %5s  %7s  %7s  %12s\n",
           "layer", "N", "K", "M", "us/dispatch");
    printf("%-18s  %5s  %7s  %7s  %12s\n",
           "------------------", "-----", "-------", "-------", "------------");

    for (int l = 0; l < nlayers; l++) {
        int K = layers[l].K, M = layers[l].M;
        for (int b = 0; b < nbatch; b++) {
            int N = batch_sizes[b];

            long long flops = (long long)K * M * N * 2;
            int runs   = (int)(2e11 / ((double)flops < 1e6 ? 1e6 : (double)flops));
            if (runs < 5)   runs = 5;
            if (runs > 500) runs = 500;
            int warmup = runs / 5 < 3 ? 3 : runs / 5;

            double us = bench_mul_mat(K, M, N, n_threads, warmup, runs);

            printf("%-18s  %5d  %7d  %7d  %12.3f\n",
                   layers[l].name, N, K, M, us);
            fflush(stdout);

            fprintf(csv, "%s,%d,%s,%d,%d,%d,%d,%d,%.3f\n",
                    BENCH_BACKEND, n_threads, layers[l].name, K, M, N, warmup, runs, us);
            fflush(csv);
        }
        printf("\n");
    }

    fclose(csv);
    return 0;
}
