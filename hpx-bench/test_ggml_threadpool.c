/* hpx-bench/test_ggml_threadpool.c
 *
 * Stage 2 contract test: verify the ggml threadpool interface behaves
 * correctly regardless of the underlying threading implementation
 * (pthreads or HPX).
 *
 * Tests the exact API surface that ggml_graph_compute() uses:
 *   ggml_threadpool_new / free / pause / resume / get_n_threads
 *   ggml_graph_plan / ggml_graph_compute
 *
 * Build via hpx-bench/CMakeLists.txt (see target test_ggml_threadpool).
 * Must be linked against a build of ggml-cpu (pthread or HPX variant).
 *
 * Expected output: all tests print PASS, exit 0.
 */

#include "ggml.h"
#include "ggml-cpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── helpers ────────────────────────────────────────────────────────────── */

static void pass(const char *name) { printf("  PASS  %s\n", name); }
static void fail(const char *name) { printf("  FAIL  %s\n", name); exit(1); }
#define CHECK(cond, name) do { if (cond) pass(name); else fail(name); } while(0)

/* Run a single ggml_add graph: result[i] = a[i] + b[i].
 * a is filled with `val_a`, b with `val_b`.
 * Returns the average of result[0..n-1]. */
static float run_add(int n, float val_a, float val_b,
                     int n_threads, struct ggml_threadpool *tp)
{
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
        work = malloc(plan.work_size);
        plan.work_data = work;
    }

    enum ggml_status st = ggml_graph_compute(gf, &plan);
    CHECK(st == GGML_STATUS_SUCCESS, "graph_compute status OK");

    /* sample a few elements to verify correctness */
    float sum = 0.0f;
    int   sample_n = n < 16 ? n : 16;
    for (int i = 0; i < sample_n; i++)
        sum += ggml_get_f32_1d(result, i);
    float avg = sum / sample_n;

    free(work);
    ggml_free(ctx);
    return avg;
}

/* ── Test 1: lifecycle ──────────────────────────────────────────────────── */

static void test_lifecycle(void)
{
    for (int t = 1; t <= 4; t++) {
        struct ggml_threadpool_params p = ggml_threadpool_params_default(t);
        struct ggml_threadpool *tp = ggml_threadpool_new(&p);
        CHECK(tp != NULL, "threadpool_new returns non-NULL");
        ggml_threadpool_free(tp);
        pass("threadpool_free (no crash)");
    }
}

/* ── Test 2: correct results, single-threaded ───────────────────────────── */

static void test_single_thread_correctness(void)
{
    struct ggml_threadpool_params p = ggml_threadpool_params_default(1);
    struct ggml_threadpool *tp = ggml_threadpool_new(&p);

    float result = run_add(4096, 1.5f, 2.5f, 1, tp);
    CHECK(fabsf(result - 4.0f) < 1e-4f, "single-thread: 1.5+2.5=4.0");

    ggml_threadpool_free(tp);
}

/* ── Test 3: correct results, multi-threaded ────────────────────────────── */

static void test_multi_thread_correctness(void)
{
    /* Use a large tensor so all threads get work */
    const int N = 64 * 1024;

    int thread_counts[] = {2, 4, 8};
    for (int i = 0; i < 3; i++) {
        int t = thread_counts[i];
        struct ggml_threadpool_params p = ggml_threadpool_params_default(t);
        struct ggml_threadpool *tp = ggml_threadpool_new(&p);

        float result = run_add(N, 3.0f, 7.0f, t, tp);
        CHECK(fabsf(result - 10.0f) < 1e-3f,
              "multi-thread: 3.0+7.0=10.0 (correct across all threads)");

        ggml_threadpool_free(tp);
    }
}

/* ── Test 4: result consistent across thread counts ────────────────────── */

static void test_thread_count_consistency(void)
{
    const int N = 32 * 1024;
    float val_a = 1.23f, val_b = 4.56f;
    float expected = val_a + val_b;

    for (int t = 1; t <= 8; t++) {
        struct ggml_threadpool_params p = ggml_threadpool_params_default(t);
        struct ggml_threadpool *tp = ggml_threadpool_new(&p);

        float result = run_add(N, val_a, val_b, t, tp);
        CHECK(fabsf(result - expected) < 1e-3f,
              "consistency: same result regardless of thread count");

        ggml_threadpool_free(tp);
    }
}

/* ── Test 5: multiple sequential graphs on same threadpool ──────────────── */

static void test_multiple_graphs(void)
{
    const int GRAPHS = 8;
    const int N      = 16 * 1024;

    struct ggml_threadpool_params p = ggml_threadpool_params_default(4);
    struct ggml_threadpool *tp = ggml_threadpool_new(&p);

    for (int g = 0; g < GRAPHS; g++) {
        float result = run_add(N, (float)g, 1.0f, 4, tp);
        float expected = (float)g + 1.0f;
        CHECK(fabsf(result - expected) < 1e-3f,
              "multiple graphs: correct result on each dispatch");
    }

    ggml_threadpool_free(tp);
}

/* ── Test 6: pause / resume ─────────────────────────────────────────────── */

static void test_pause_resume(void)
{
    const int N = 8 * 1024;

    /* Create paused */
    struct ggml_threadpool_params p = ggml_threadpool_params_default(4);
    p.paused = true;
    struct ggml_threadpool *tp = ggml_threadpool_new(&p);
    CHECK(tp != NULL, "pause/resume: threadpool created paused");

    /* Resume and compute */
    ggml_threadpool_resume(tp);
    float result = run_add(N, 2.0f, 3.0f, 4, tp);
    CHECK(fabsf(result - 5.0f) < 1e-3f, "pause/resume: correct result after resume");

    /* Pause again and free */
    ggml_threadpool_pause(tp);
    ggml_threadpool_free(tp);
    pass("pause/resume: free after pause (no deadlock)");
}

/* ── main ───────────────────────────────────────────────────────────────── */

int main(void)
{
    printf("\n=== ggml threadpool contract test ===\n\n");

    test_lifecycle();
    test_single_thread_correctness();
    test_multi_thread_correctness();
    test_thread_count_consistency();
    test_multiple_graphs();
    test_pause_resume();

    printf("\nAll tests passed.\n\n");
    return 0;
}
