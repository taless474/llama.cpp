#include "test_ggml_threadpool_common.h"

/* Keep two pools alive at once and interleave work across them.
 * This checks that pool-local state does not leak across instances. */

static void test_two_pools_isolation(void) {
    struct ggml_threadpool_params pa = ggml_threadpool_params_default(4);
    struct ggml_threadpool_params pb = ggml_threadpool_params_default(7);

    struct ggml_threadpool *tp_a = ggml_threadpool_new(&pa);
    struct ggml_threadpool *tp_b = ggml_threadpool_new(&pb);

    CHECK_(tp_a != NULL, "two pools: pool A created");
    CHECK_(tp_b != NULL, "two pools: pool B created");

    int counts_a[] = {4, 2, 1, 3, 4, 1};
    int counts_b[] = {7, 5, 3, 6, 2, 4};
    const int steps = 60;

    for (int i = 0; i < steps; ++i) {
        int ta = counts_a[i % (int)(sizeof(counts_a) / sizeof(counts_a[0]))];
        int tb = counts_b[i % (int)(sizeof(counts_b) / sizeof(counts_b[0]))];

        float add_expected = 10.0f + (float)(i % 5);
        float add_got = run_add_(32 * 1024, 9.0f + (float)(i % 5), 1.0f, ta, tp_a);
        CHECK_(fabsf(add_got - add_expected) < 1e-3f,
               "two pools: pool A result correct while pool B exists");

        float chain_expected = 5.0f; /* val=1, chain_len=4 -> 5 */
        float chain_got = run_add_chain_(24 * 1024, 1.0f, 4, tb, tp_b);
        CHECK_(fabsf(chain_got - chain_expected) < 1e-2f,
               "two pools: pool B result correct while pool A exists");
    }

    ggml_threadpool_free(tp_a);
    ggml_threadpool_free(tp_b);
}

int main(void) {
    printf("\n=== two pools isolation ===\n\n");
    test_two_pools_isolation();
    printf("\nAll tests passed.\n\n");
    return 0;
}
