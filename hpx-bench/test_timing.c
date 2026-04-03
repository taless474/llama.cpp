#include "ggml.h"
#include "ggml-cpu.h"
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

static volatile sig_atomic_t g_iter = -1;
static void on_alarm(int sig) { (void)sig; printf("HANG at iter=%d\n", (int)g_iter); fflush(stdout); _exit(2); }

static uint64_t now_ns(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

int main(void) {
    signal(SIGALRM, on_alarm);
    setvbuf(stdout, NULL, _IOLBF, 0);

    const int N=32, CHAIN=2, T=4, RUNS=30;
    size_t mem = (size_t)(N*sizeof(float)*(CHAIN+2)) + 2*1024*1024;
    struct ggml_init_params ip = { .mem_size=mem, .no_alloc=false };
    struct ggml_context *ctx = ggml_init(ip);
    struct ggml_tensor *a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);
    ggml_set_f32(a, 1.0f);
    struct ggml_tensor *cur = a;
    for (int i=0; i<CHAIN; ++i) cur = ggml_add(ctx, cur, a);
    struct ggml_cgraph *gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, cur);
    struct ggml_threadpool_params p = ggml_threadpool_params_default(T);
    struct ggml_threadpool *tp = ggml_threadpool_new(&p);
    struct ggml_cplan plan = ggml_graph_plan(gf, T, tp);

    printf("iter  us_elapsed\n");
    for (int i=0; i<RUNS; ++i) {
        g_iter = i;
        alarm(30);
        uint64_t t0 = now_ns();
        ggml_graph_compute(gf, &plan);
        uint64_t t1 = now_ns();
        alarm(0);
        printf("%4d  %10.1f\n", i, (double)(t1-t0)/1e3);
        fflush(stdout);
    }
    ggml_threadpool_free(tp);
    ggml_free(ctx);
    return 0;
}
