/* Does running one compute with a prior pool prevent the chain=2 hang? */
#include "ggml.h"
#include "ggml-cpu.h"
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static volatile sig_atomic_t g_iter = -1;
static void on_alarm(int sig) {
    (void)sig;
    printf("HANG at iter=%d\n", (int)g_iter);
    fflush(stdout);
    _exit(2);
}

static struct ggml_threadpool *make_pool(int n_threads) {
    struct ggml_threadpool_params p = ggml_threadpool_params_default(n_threads);
    return ggml_threadpool_new(&p);
}

static struct ggml_cgraph *make_graph(struct ggml_context *ctx, int N, int chain) {
    struct ggml_tensor *a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);
    ggml_set_f32(a, 1.0f);
    struct ggml_tensor *cur = a;
    for (int i = 0; i < chain; ++i) cur = ggml_add(ctx, cur, a);
    struct ggml_cgraph *gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, cur);
    return gf;
}

int main(int argc, char **argv) {
    signal(SIGALRM, on_alarm);
    setvbuf(stdout, NULL, _IOLBF, 0);

    int do_warmup = (argc >= 2 && argv[1][0] == '1');
    printf("warmup=%d\n", do_warmup);

    if (do_warmup) {
        /* Run one chain=1 compute on a separate pool first */
        struct ggml_init_params ip = { .mem_size = 1*1024*1024, .no_alloc=false };
        struct ggml_context *ctx = ggml_init(ip);
        struct ggml_cgraph *gf = make_graph(ctx, 32, 1);
        struct ggml_threadpool *tp = make_pool(4);
        struct ggml_cplan plan = ggml_graph_plan(gf, 4, tp);
        alarm(4); ggml_graph_compute(gf, &plan); alarm(0);
        ggml_threadpool_free(tp);
        ggml_free(ctx);
        printf("warmup pool done\n");
    }

    /* Now run chain=2 pool for 30 iters */
    size_t mem = (size_t)(32 * sizeof(float) * 4) + 2*1024*1024;
    struct ggml_init_params ip = { .mem_size = mem, .no_alloc = false };
    struct ggml_context *ctx = ggml_init(ip);
    struct ggml_cgraph *gf = make_graph(ctx, 32, 2);
    struct ggml_threadpool *tp = make_pool(4);
    struct ggml_cplan plan = ggml_graph_plan(gf, 4, tp);

    for (int i = 0; i < 30; ++i) {
        g_iter = i;
        alarm(4);
        ggml_graph_compute(gf, &plan);
        alarm(0);
        if ((i+1) % 5 == 0) printf("  ok %d/30\n", i+1);
    }
    printf("PASS\n");
    ggml_threadpool_free(tp);
    ggml_free(ctx);
    return 0;
}
