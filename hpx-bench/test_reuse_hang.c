/* Does the hang occur on repeated computes with the same pool/graph? */
#include "ggml.h"
#include "ggml-cpu.h"
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static volatile sig_atomic_t g_iter = -1;

static void on_alarm(int sig) {
    (void)sig;
    printf("\nFAIL: timeout/hang at iter=%d\n", (int)g_iter);
    fflush(stdout);
    _exit(2);
}

int main(void) {
    signal(SIGALRM, on_alarm);
    setvbuf(stdout, NULL, _IOLBF, 0);

    const int N = 32, CHAIN = 2, T = 4, RUNS = 30;
    size_t mem = (size_t)(N * sizeof(float) * (CHAIN + 2)) + 2*1024*1024;
    struct ggml_init_params ip = { .mem_size = mem, .mem_buffer = NULL, .no_alloc = false };
    struct ggml_context *ctx = ggml_init(ip);

    struct ggml_tensor *a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);
    ggml_set_f32(a, 1.0f);
    struct ggml_tensor *cur = a;
    for (int i = 0; i < CHAIN; ++i) cur = ggml_add(ctx, cur, a);
    struct ggml_cgraph *gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, cur);

    struct ggml_threadpool_params p = ggml_threadpool_params_default(T);
    struct ggml_threadpool *tp = ggml_threadpool_new(&p);
    struct ggml_cplan plan = ggml_graph_plan(gf, T, tp);

    printf("reuse test: N=%d chain=%d threads=%d runs=%d\n", N, CHAIN, T, RUNS);

    for (int i = 0; i < RUNS; ++i) {
        g_iter = i;
        alarm(5);
        enum ggml_status st = ggml_graph_compute(gf, &plan);
        alarm(0);
        if (st != GGML_STATUS_SUCCESS) { printf("FAIL: non-success at iter=%d\n", i); return 1; }
        if ((i+1) % 5 == 0) printf("ok %d/%d\n", i+1, RUNS);
    }
    printf("PASS\n");
    ggml_threadpool_free(tp);
    ggml_free(ctx);
    return 0;
}
