#include "test_ggml_threadpool_common.h"

/* Same idea as variable n_threads on the add path, but on mul_mat using one
 * persistent pool. This exercises barrier/counter reconfiguration on the more
 * realistic compute kernel too. */

static void test_mul_mat_variable_thread_counts_same_pool(void) {
    const int K = 1024;
    const int M = 192;
    const int N = 96;

    struct ggml_threadpool_params p = ggml_threadpool_params_default(8);
    struct ggml_threadpool *tp = ggml_threadpool_new(&p);
    CHECK_(tp != NULL, "mul_mat variable counts: pool created");

    int counts[] = {8, 4, 2, 1, 2, 4, 8, 3, 6, 1, 5, 7};
    const int ncounts = (int) (sizeof(counts) / sizeof(counts[0]));

    for (int round = 0; round < 3; ++round) {
        for (int i = 0; i < ncounts; ++i) {
            float max_err = run_mul_mat_max_err_(K, M, N, counts[i], tp);
            CHECK_(max_err < 1e-2f,
                   "mul_mat variable counts: max abs error within tolerance");
        }
    }

    ggml_threadpool_free(tp);
}

int main(void) {
    printf("\n=== mul_mat variable thread counts on same pool ===\n\n");
    test_mul_mat_variable_thread_counts_same_pool();
    printf("\nAll tests passed.\n\n");
    return 0;
}
