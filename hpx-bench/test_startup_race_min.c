/* hpx-bench/test_startup_race_min.c
 *
 * Minimal startup-race reproducer:
 *   - fresh pool every iteration
 *   - immediate first compute
 *   - tiny graph (N=32, chain_len=2)
 *   - fixed reusable pool path
 *
 * Defaults:
 *   iters    = 1000
 *   delay_us = 0
 *   threads  = 4
 *
 * Usage:
 *   ./test_startup_race_min
 *   ./test_startup_race_min 1000 0 4
 *   ./test_startup_race_min 1000 1000 4   # add 1 ms delay before first compute
 *
 * Expected:
 *   - With the startup race present, delay_us=0 may hang or timeout.
 *   - Adding a small delay often makes it pass, which strongly suggests
 *     workers were not ready at first dispatch.
 */

#include "ggml.h"
#include "ggml-cpu.h"

#include <math.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static volatile sig_atomic_t g_iter = -1;
static volatile sig_atomic_t g_delay_us = 0;
static volatile sig_atomic_t g_threads = 0;

static void fail(const char *msg) {
    printf("FAIL: %s\n", msg);
    fflush(stdout);
    exit(1);
}

static void on_alarm(int sig) {
    (void) sig;
    printf("\nFAIL: timeout/hang at iter=%d delay_us=%d threads=%d\n",
           (int) g_iter, (int) g_delay_us, (int) g_threads);
    fflush(stdout);
    _exit(2);
}

static float run_once(int n, int chain_len, int n_threads, int delay_us) {
    size_t mem = (size_t)(n * sizeof(float) * (chain_len + 2)) + 2 * 1024 * 1024;

    struct ggml_init_params init = {
        .mem_size   = mem,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };

    struct ggml_context *ctx = ggml_init(init);
    if (!ctx) {
        fail("ggml_init failed");
    }

    struct ggml_tensor *a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_set_f32(a, 1.0f);

    struct ggml_tensor *cur = a;
    for (int i = 0; i < chain_len; ++i) {
        cur = ggml_add(ctx, cur, a);
    }

    struct ggml_cgraph *gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, cur);

    struct ggml_threadpool_params p = ggml_threadpool_params_default(n_threads);
    struct ggml_threadpool *tp = ggml_threadpool_new(&p);
    if (!tp) {
        ggml_free(ctx);
        fail("ggml_threadpool_new failed");
    }

    struct ggml_cplan plan = ggml_graph_plan(gf, n_threads, tp);

    uint8_t *work = NULL;
    if (plan.work_size > 0) {
        work = (uint8_t *) malloc(plan.work_size);
        if (!work) {
            ggml_threadpool_free(tp);
            ggml_free(ctx);
            fail("malloc(work) failed");
        }
        plan.work_data = work;
    }

    if (delay_us > 0) {
        usleep((useconds_t) delay_us);
    }

    /* Bound each iteration so a hang becomes a test failure, not a stuck CI job. */
    alarm(5);
    enum ggml_status st = ggml_graph_compute(gf, &plan);
    alarm(0);

    if (st != GGML_STATUS_SUCCESS) {
        free(work);
        ggml_threadpool_free(tp);
        ggml_free(ctx);
        fail("ggml_graph_compute returned non-success");
    }

    float sum = 0.0f;
    int sample_n = n < 16 ? n : 16;
    for (int i = 0; i < sample_n; ++i) {
        sum += ggml_get_f32_1d(cur, i);
    }
    float avg = sum / sample_n;

    free(work);
    ggml_threadpool_free(tp);
    ggml_free(ctx);

    return avg;
}

int main(int argc, char **argv) {
    signal(SIGALRM, on_alarm);
    setvbuf(stdout, NULL, _IOLBF, 0);

    int iters    = 1000;
    int delay_us = 0;
    int threads  = 4;

    if (argc >= 2) {
        iters = atoi(argv[1]);
        if (iters < 1) iters = 1;
    }
    if (argc >= 3) {
        delay_us = atoi(argv[2]);
        if (delay_us < 0) delay_us = 0;
    }
    if (argc >= 4) {
        threads = atoi(argv[3]);
        if (threads < 1) threads = 1;
    }

    const int N = 32;
    const int CHAIN_LEN = 2;
    const float EXPECTED = (float)(CHAIN_LEN + 1); /* 3.0 */

    g_delay_us = delay_us;
    g_threads = threads;

    printf("\n=== minimal startup race test ===\n");
    printf("iters    : %d\n", iters);
    printf("delay_us : %d\n", delay_us);
    printf("threads  : %d\n", threads);
    printf("N        : %d\n", N);
    printf("chain    : %d\n", CHAIN_LEN);
    printf("expected : %.3f\n\n", EXPECTED);

    for (int i = 0; i < iters; ++i) {
        g_iter = i;

        float got = run_once(N, CHAIN_LEN, threads, delay_us);
        if (fabsf(got - EXPECTED) >= 1e-3f) {
            printf("FAIL: value mismatch at iter=%d got=%f expected=%f\n", i, got, EXPECTED);
            return 1;
        }

        if ((i + 1) % 10 == 0 || i + 1 == iters) {
            printf("progress %d/%d\n", i + 1, iters);
        }
    }

    printf("\nPASS: no hangs, all first-dispatch runs produced %.3f\n\n", EXPECTED);
    return 0;
}
