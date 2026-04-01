#include "test_ggml_threadpool_common.h"

/* Reuse one persistent pool across different graph shapes and different
 * n_threads values so one workload cannot leave stale state for the next. */

static void test_mixed_workload_reuse_same_pool(void) {
    struct ggml_threadpool_params p = ggml_threadpool_params_default(8);
    struct ggml_threadpool *tp = ggml_threadpool_new(&p);
    CHECK_(tp != NULL, "mixed reuse: pool created");

    for (int r = 0; r < 20; ++r) {
        float add1 = run_add_(16 * 1024, 2.0f, 1.0f, 8, tp);
        CHECK_(fabsf(add1 - 3.0f) < 1e-3f,
               "mixed reuse: add graph correct");

        float chain = run_add_chain_(16 * 1024, 1.0f, 6, 3, tp);
        CHECK_(fabsf(chain - 7.0f) < 1e-2f,
               "mixed reuse: add-chain graph correct");

        float mm1 = run_mul_mat_max_err_(1024, 192, 96, 4, tp);
        CHECK_(mm1 < 1e-2f,
               "mixed reuse: mul_mat graph correct (phase 1)");

        float add2 = run_add_(8 * 1024, -2.5f, 4.0f, 1, tp);
        CHECK_(fabsf(add2 - 1.5f) < 1e-3f,
               "mixed reuse: single-thread add after mul_mat correct");

        float mm2 = run_mul_mat_max_err_(1024, 192, 96, 6, tp);
        CHECK_(mm2 < 1e-2f,
               "mixed reuse: mul_mat graph correct (phase 2)");
    }

    ggml_threadpool_free(tp);
}

int main(void) {
    printf("\n=== mixed workload reuse on same pool ===\n\n");
    test_mixed_workload_reuse_same_pool();
    printf("\nAll tests passed.\n\n");
    return 0;
}
