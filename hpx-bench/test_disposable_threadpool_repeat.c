#include "test_ggml_threadpool_common.h"

/* Repeatedly exercise the disposable threadpool path:
 *   ggml_graph_plan(..., NULL)
 *   plan.threadpool = NULL
 *   ggml_graph_compute(...)
 * so create/free/create is stressed, not just one-shot validated. */

static void test_disposable_threadpool_repeat(void) {
    const int N = 16 * 1024;
    const int RUNS = 250;
    int counts[] = {1, 4, 2, 8, 3, 6, 1, 5};
    const int ncounts = (int) (sizeof(counts) / sizeof(counts[0]));

    for (int r = 0; r < RUNS; ++r) {
        int t = counts[r % ncounts];

        struct ggml_init_params init = {
            .mem_size   = (size_t)(N * 4 * 3 + 512) * 1024,
            .mem_buffer = NULL,
            .no_alloc   = false,
        };
        struct ggml_context *ctx = ggml_init(init);

        struct ggml_tensor *a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);
        struct ggml_tensor *b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);
        ggml_set_f32(a, 5.0f + (float)(r % 3));
        ggml_set_f32(b, 3.0f);

        struct ggml_tensor *result = ggml_add(ctx, a, b);
        struct ggml_cgraph *gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, result);

        struct ggml_cplan plan = ggml_graph_plan(gf, t, NULL);
        plan.threadpool = NULL;

        uint8_t *work = NULL;
        if (plan.work_size > 0) {
            work = (uint8_t *) malloc(plan.work_size);
            if (!work) fail_("malloc work buffer");
            plan.work_data = work;
        }

        enum ggml_status st = ggml_graph_compute(gf, &plan);
        CHECK_(st == GGML_STATUS_SUCCESS, "disposable repeat: status ok");
        CHECK_(fabsf(ggml_get_f32_1d(result, 0) - (8.0f + (float)(r % 3))) < 1e-3f,
               "disposable repeat: result correct");

        free(work);
        ggml_free(ctx);
    }
}

int main(void) {
    printf("\n=== disposable threadpool repeat ===\n\n");
    test_disposable_threadpool_repeat();
    printf("\nAll tests passed.\n\n");
    return 0;
}
