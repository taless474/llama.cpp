#include "ggml.h"
#include "ggml-cpu.h"
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static volatile sig_atomic_t g_iter = -1;
static int g_chain = 0;

static void on_alarm(int sig) {
    (void)sig;
    printf("HANG at iter=%d chain=%d (total barriers/worker=%d)\n",
           (int)g_iter, g_chain, (int)g_iter * (g_chain + 1));
    fflush(stdout);
    _exit(2);
}

static void run_series(int chain_len, int n_threads, int max_iters) {
    g_chain = chain_len;
    const int N = 32;
    size_t mem = (size_t)(N * sizeof(float) * (chain_len + 2)) + 2*1024*1024;
    struct ggml_init_params ip = { .mem_size = mem, .mem_buffer = NULL, .no_alloc = false };
    struct ggml_context *ctx = ggml_init(ip);

    struct ggml_tensor *a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);
    ggml_set_f32(a, 1.0f);
    struct ggml_tensor *cur = a;
    for (int i = 0; i < chain_len; ++i) cur = ggml_add(ctx, cur, a);
    struct ggml_cgraph *gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, cur);

    struct ggml_threadpool_params p = ggml_threadpool_params_default(n_threads);
    struct ggml_threadpool *tp = ggml_threadpool_new(&p);
    struct ggml_cplan plan = ggml_graph_plan(gf, n_threads, tp);

    printf("  chain=%d t=%d: ", chain_len, n_threads); fflush(stdout);
    int i;
    for (i = 0; i < max_iters; ++i) {
        g_iter = i;
        alarm(4);
        ggml_graph_compute(gf, &plan);
        alarm(0);
    }
    printf("PASS after %d iters (total barriers/worker=%d)\n", max_iters, max_iters * (chain_len + 1));

    ggml_threadpool_free(tp);
    ggml_free(ctx);
}

int main(void) {
    signal(SIGALRM, on_alarm);
    setvbuf(stdout, NULL, _IOLBF, 0);
    printf("threshold sweep: N=32 t=4\n");
    run_series(1, 4, 30);
    run_series(2, 4, 30);
    run_series(3, 4, 30);
    run_series(4, 4, 30);
    return 0;
}
