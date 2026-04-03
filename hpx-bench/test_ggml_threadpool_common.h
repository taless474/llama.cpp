#ifndef TEST_GGML_THREADPOOL_COMMON_H
#define TEST_GGML_THREADPOOL_COMMON_H

#include "ggml.h"
#include "ggml-cpu.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void pass_(const char *name) { printf("  PASS  %s\n", name); }
static void fail_(const char *name) { printf("  FAIL  %s\n", name); exit(1); }

#define CHECK_(cond, name) do { if (cond) pass_(name); else fail_(name); } while (0)

/* Run a single ggml_add graph: result[i] = a[i] + b[i].
 * Returns the average of the first few samples. */
static float run_add_(int n, float val_a, float val_b,
                      int n_threads, struct ggml_threadpool *tp) {
    struct ggml_init_params init = {
        .mem_size   = (size_t)(n * 4 * 3 + 512) * 1024,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    struct ggml_context *ctx = ggml_init(init);

    struct ggml_tensor *a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    struct ggml_tensor *b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_set_f32(a, val_a);
    ggml_set_f32(b, val_b);

    struct ggml_tensor *result = ggml_add(ctx, a, b);

    struct ggml_cgraph *gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, result);

    struct ggml_cplan plan = ggml_graph_plan(gf, n_threads, tp);
    uint8_t *work = NULL;
    if (plan.work_size > 0) {
        work = (uint8_t *) malloc(plan.work_size);
        if (!work) {
            ggml_free(ctx);
            fail_("malloc work buffer");
        }
        plan.work_data = work;
    }

    enum ggml_status st = ggml_graph_compute(gf, &plan);
    CHECK_(st == GGML_STATUS_SUCCESS, "graph_compute status OK");

    float sum = 0.0f;
    int sample_n = n < 16 ? n : 16;
    for (int i = 0; i < sample_n; ++i) {
        sum += ggml_get_f32_1d(result, i);
    }

    free(work);
    ggml_free(ctx);
    return sum / sample_n;
}

/* Run a chain of sequential adds on the same tensor.
 * Expected result is val * (chain_len + 1). */
static float run_add_chain_(int n, float val, int chain_len,
                            int n_threads, struct ggml_threadpool *tp) {
    size_t mem = (size_t) (n * sizeof(float) * (chain_len + 2)) + 2 * 1024 * 1024;
    struct ggml_init_params init = {
        .mem_size   = mem,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    struct ggml_context *ctx = ggml_init(init);

    struct ggml_tensor *a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_set_f32(a, val);

    struct ggml_tensor *cur = a;
    for (int i = 0; i < chain_len; ++i) {
        cur = ggml_add(ctx, cur, a);
    }

    struct ggml_cgraph *gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, cur);

    struct ggml_cplan plan = ggml_graph_plan(gf, n_threads, tp);
    uint8_t *work = NULL;
    if (plan.work_size > 0) {
        work = (uint8_t *) malloc(plan.work_size);
        if (!work) {
            ggml_free(ctx);
            fail_("malloc work buffer");
        }
        plan.work_data = work;
    }

    enum ggml_status st = ggml_graph_compute(gf, &plan);
    CHECK_(st == GGML_STATUS_SUCCESS, "chain graph_compute ok");

    float sum = 0.0f;
    int sample_n = n < 16 ? n : 16;
    for (int i = 0; i < sample_n; ++i) {
        sum += ggml_get_f32_1d(cur, i);
    }

    free(work);
    ggml_free(ctx);
    return sum / sample_n;
}

static float pattern_a_(int k, int m) {
    int v = ((k * 17 + m * 13) % 19) - 9;
    return 0.125f * (float) v;
}

static float pattern_b_(int k, int n) {
    int v = ((k * 11 + n * 7) % 23) - 11;
    return 0.125f * (float) v;
}

/* Run C = A^T * B via ggml_mul_mat(A, B) and compare against a scalar reference.
 * Returns the maximum absolute error. */
static float run_mul_mat_max_err_(int K, int M, int N,
                                  int n_threads, struct ggml_threadpool *tp) {
    size_t mem = (size_t) (K * M * 8 + K * N * 8 + M * N * 8) + 8 * 1024 * 1024;

    struct ggml_init_params init = {
        .mem_size   = mem,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    struct ggml_context *ctx = ggml_init(init);

    struct ggml_tensor *A = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, K, M);
    struct ggml_tensor *B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);

    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            ggml_set_f32_1d(A, m * K + k, pattern_a_(k, m));
        }
    }

    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            ggml_set_f32_1d(B, n * K + k, pattern_b_(k, n));
        }
    }

    struct ggml_tensor *C = ggml_mul_mat(ctx, A, B);

    struct ggml_cgraph *gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, C);

    struct ggml_cplan plan = ggml_graph_plan(gf, n_threads, tp);
    uint8_t *work = NULL;
    if (plan.work_size > 0) {
        work = (uint8_t *) malloc(plan.work_size);
        if (!work) {
            ggml_free(ctx);
            fail_("malloc work buffer");
        }
        plan.work_data = work;
    }

    enum ggml_status st = ggml_graph_compute(gf, &plan);
    CHECK_(st == GGML_STATUS_SUCCESS, "mul_mat graph_compute status OK");

    float max_err = 0.0f;
    for (int n = 0; n < N; ++n) {
        for (int m = 0; m < M; ++m) {
            float ref = 0.0f;
            for (int k = 0; k < K; ++k) {
                float a = ggml_get_f32_nd(A, k, m, 0, 0);
                float b = ggml_get_f32_nd(B, k, n, 0, 0);
                ref += a * b;
            }

            float got = ggml_get_f32_nd(C, m, n, 0, 0);
            float err = fabsf(got - ref);
            if (err > max_err) {
                max_err = err;
            }
        }
    }

    free(work);
    ggml_free(ctx);
    return max_err;
}

#endif /* TEST_GGML_THREADPOOL_COMMON_H */
