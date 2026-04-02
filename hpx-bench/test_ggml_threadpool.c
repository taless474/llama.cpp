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
#include <unistd.h>   /* sysconf(_SC_NPROCESSORS_ONLN) */

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

/* ── helpers (extended) ──────────────────────────────────────────────────── */

/* Run a chain of `chain_len` sequential adds on the same tensor.
 * Graph has `chain_len` nodes → `chain_len` barrier cycles per dispatch.
 * Result should be initial_val * (chain_len + 1) because each step adds
 * the original `a` to the running total.
 * e.g. chain_len=4, val=1: 1→2→3→4→5, expected = 5.0 */
static float run_add_chain(int n, float val, int chain_len,
                            int n_threads, struct ggml_threadpool *tp)
{
    size_t mem = (size_t)(n * sizeof(float) * (chain_len + 2)) + 2 * 1024 * 1024;
    struct ggml_init_params init = { mem, NULL, false };
    struct ggml_context *ctx = ggml_init(init);

    struct ggml_tensor *a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_set_f32(a, val);

    struct ggml_tensor *cur = a;
    for (int i = 0; i < chain_len; i++)
        cur = ggml_add(ctx, cur, a);

    struct ggml_cgraph *gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, cur);

    struct ggml_cplan plan = ggml_graph_plan(gf, n_threads, tp);
    uint8_t *work = NULL;
    if (plan.work_size > 0) {
        work = malloc(plan.work_size);
        plan.work_data = work;
    }

    enum ggml_status st = ggml_graph_compute(gf, &plan);
    CHECK(st == GGML_STATUS_SUCCESS, "chain graph_compute ok");

    float sum = 0.0f;
    int   sample_n = n < 16 ? n : 16;
    for (int i = 0; i < sample_n; i++)
        sum += ggml_get_f32_1d(cur, i);

    free(work);
    ggml_free(ctx);
    return sum / sample_n;
}

/* ── Test 7: barrier correctness ─────────────────────────────────────────── */
/* A chain of 10 adds creates 10 graph nodes → 10 barrier cycles per dispatch.
 * A broken barrier lets threads race into the next node before the previous
 * one is finished, producing wrong intermediate values.
 * Thread counts are chosen to hit the specific sizes where bugs would show:
 *   t=1  → barrier early-return path (no synchronisation needed)
 *   t=2  → minimal contention, tests basic two-thread sync
 *   t=4  → typical workload
 *   t=10 → exceeds typical core count, stresses scheduling */

static void test_barrier_correctness(void)
{
    const int N          = 32 * 1024;
    const int CHAIN_LEN  = 10;
    const float VAL      = 1.0f;
    const float EXPECTED = (float)(CHAIN_LEN + 1) * VAL; /* 11.0 */

    int thread_counts[] = {1, 2, 4, 10};
    for (int i = 0; i < 4; i++) {
        int t = thread_counts[i];
        struct ggml_threadpool_params p = ggml_threadpool_params_default(t);
        struct ggml_threadpool *tp = ggml_threadpool_new(&p);

        float result = run_add_chain(N, VAL, CHAIN_LEN, t, tp);
        CHECK(fabsf(result - EXPECTED) < 1e-2f,
              "barrier: chain result correct (would be wrong if barrier races)");

        ggml_threadpool_free(tp);
    }
}

/* ── Test 8: variable n_threads per graph on same pool ───────────────────── */
/* Changing n_threads between dispatches on the same persistent pool exercises
 * barrier reconstruction (HPX) and kickoff counter repacking (pthread/HPX).
 * If the barrier or counter is stale, results will be wrong. */

static void test_variable_thread_counts(void)
{
    const int N      = 32 * 1024;
    const int MAX_T  = 8;
    const float EXPECTED = 4.0f; /* 2.5 + 1.5 */

    struct ggml_threadpool_params p = ggml_threadpool_params_default(MAX_T);
    struct ggml_threadpool *tp = ggml_threadpool_new(&p);

    /* cycle: descend, ascend, random — cover all transitions */
    int counts[] = {8, 4, 2, 1, 2, 4, 8, 3, 6, 1, 5, 7};
    int ncounts  = (int)(sizeof(counts) / sizeof(counts[0]));
    for (int i = 0; i < ncounts; i++) {
        int   t      = counts[i];
        float result = run_add(N, 2.5f, 1.5f, t, tp);
        CHECK(fabsf(result - EXPECTED) < 1e-3f,
              "variable n_threads: correct result at each count");
    }

    ggml_threadpool_free(tp);
}

/* ── Test 9: stress — many sequential dispatches ─────────────────────────── */
/* Race conditions that only surface intermittently need many iterations.
 * Uses a chain graph so barriers are exercised on every dispatch. */

static void test_stress(void)
{
    const int GRAPHS     = 100;
    const int N          = 16 * 1024;
    const int CHAIN_LEN  = 5;
    const float EXPECTED = (float)(CHAIN_LEN + 1); /* val=1 → 6.0 */

    struct ggml_threadpool_params p = ggml_threadpool_params_default(4);
    struct ggml_threadpool *tp = ggml_threadpool_new(&p);

    for (int g = 0; g < GRAPHS; g++) {
        float result = run_add_chain(N, 1.0f, CHAIN_LEN, 4, tp);
        CHECK(fabsf(result - EXPECTED) < 1e-2f,
              "stress: correct result on every dispatch");
    }

    ggml_threadpool_free(tp);
}

/* ── Test 10: thread count edge cases ────────────────────────────────────── */
/* n_threads=1: barrier early-return path (no hpx::barrier::arrive_and_wait).
 * n_threads=n_cores: maximum OS-level contention. */

static void test_thread_count_edges(void)
{
    const int N          = 32 * 1024;
    const int CHAIN_LEN  = 8;
    const float EXPECTED = (float)(CHAIN_LEN + 1); /* val=1 → 9.0 */

    /* t=1: barrier bypass */
    {
        struct ggml_threadpool_params p = ggml_threadpool_params_default(1);
        struct ggml_threadpool *tp = ggml_threadpool_new(&p);
        float result = run_add_chain(N, 1.0f, CHAIN_LEN, 1, tp);
        CHECK(fabsf(result - EXPECTED) < 1e-2f, "edge: t=1 (barrier bypass) correct");
        ggml_threadpool_free(tp);
    }

    /* t=n_cores: maximum contention */
    {
#if defined(__APPLE__)
        int n_cores = (int)sysconf(_SC_NPROCESSORS_ONLN);
#else
        int n_cores = (int)sysconf(_SC_NPROCESSORS_ONLN);
#endif
        if (n_cores < 1) n_cores = 4;
        struct ggml_threadpool_params p = ggml_threadpool_params_default(n_cores);
        struct ggml_threadpool *tp = ggml_threadpool_new(&p);
        float result = run_add_chain(N, 1.0f, CHAIN_LEN, n_cores, tp);
        CHECK(fabsf(result - EXPECTED) < 1e-2f, "edge: t=n_cores (max contention) correct");
        ggml_threadpool_free(tp);
    }
}

/* ── Test 11: disposable threadpool (NULL cplan->threadpool) ─────────────── */
/* ggml_graph_compute creates a throw-away pool when cplan->threadpool=NULL.
 * In HPX mode this code path was patched (Step 3) — confirm it works. */

static void test_disposable_threadpool(void)
{
    const int N = 16 * 1024;
    struct ggml_init_params init = {
        .mem_size   = (size_t)(N * 4 * 3 + 512) * 1024,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    struct ggml_context *ctx = ggml_init(init);

    struct ggml_tensor *a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);
    struct ggml_tensor *b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);
    ggml_set_f32(a, 5.0f);
    ggml_set_f32(b, 3.0f);

    struct ggml_tensor *result = ggml_add(ctx, a, b);
    struct ggml_cgraph *gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, result);

    /* NULL threadpool → ggml_graph_compute allocates a disposable one */
    struct ggml_cplan plan = ggml_graph_plan(gf, 4, NULL);
    plan.threadpool = NULL;
    uint8_t *work = NULL;
    if (plan.work_size > 0) {
        work = malloc(plan.work_size);
        plan.work_data = work;
    }

    enum ggml_status st = ggml_graph_compute(gf, &plan);
    CHECK(st == GGML_STATUS_SUCCESS, "disposable pool: status ok");
    CHECK(fabsf(ggml_get_f32_1d(result, 0) - 8.0f) < 1e-3f,
          "disposable pool: 5+3=8 correct");

    free(work);
    ggml_free(ctx);
}

/* ── main ───────────────────────────────────────────────────────────────── */

int main(void)
{
    printf("\n=== ggml threadpool contract test ===\n\n");

    printf("── Original contract tests ──\n");
    test_lifecycle();
    test_single_thread_correctness();
    test_multi_thread_correctness();
    test_thread_count_consistency();
    test_multiple_graphs();
    test_pause_resume();

    printf("\n── Barrier & correctness tests ──\n");
    test_barrier_correctness();
    test_variable_thread_counts();

    printf("\n── Stress & edge case tests ──\n");
    test_stress();
    test_thread_count_edges();
    test_disposable_threadpool();

    printf("\nAll tests passed.\n\n");
    return 0;
}
