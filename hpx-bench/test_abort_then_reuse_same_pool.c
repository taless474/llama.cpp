#include "test_ggml_threadpool_common.h"

/* This test assumes your local ggml_cplan exposes:
 *   - abort_callback
 *   - abort_callback_data
 * and that ggml_graph_compute() returns GGML_STATUS_ABORTED when the callback
 * requests abort.
 */

struct abort_state {
    int calls;
    int abort_after;
};

static bool abort_after_n_calls_(void *data) {
    struct abort_state *st = (struct abort_state *) data;
    st->calls += 1;
    return st->calls >= st->abort_after;
}

static enum ggml_status run_abortable_chain_(int n, int chain_len, int n_threads,
                                             struct ggml_threadpool *tp,
                                             struct abort_state *ab) {
    size_t mem = (size_t) (n * sizeof(float) * (chain_len + 2)) + 2 * 1024 * 1024;
    struct ggml_init_params init = {
        .mem_size   = mem,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    struct ggml_context *ctx = ggml_init(init);

    struct ggml_tensor *a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_set_f32(a, 1.0f);

    struct ggml_tensor *cur = a;
    for (int i = 0; i < chain_len; ++i) {
        cur = ggml_add(ctx, cur, a);
    }

    struct ggml_cgraph *gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, cur);

    struct ggml_cplan plan = ggml_graph_plan(gf, n_threads, tp);
    plan.abort_callback = abort_after_n_calls_;
    plan.abort_callback_data = ab;

    uint8_t *work = NULL;
    if (plan.work_size > 0) {
        work = (uint8_t *) malloc(plan.work_size);
        if (!work) {
            ggml_free(ctx);
            fail_("malloc work buffer");
        }
        plan.work_data = work;
    }

    enum ggml_status st = ggml_graph_compute(gf, &plan);

    free(work);
    ggml_free(ctx);
    return st;
}

static void test_abort_then_reuse_same_pool(void) {
    struct ggml_threadpool_params p = ggml_threadpool_params_default(4);
    struct ggml_threadpool *tp = ggml_threadpool_new(&p);
    CHECK_(tp != NULL, "abort/reuse: pool created");

    struct abort_state ab = {
        .calls = 0,
        .abort_after = 1,
    };

    enum ggml_status st = run_abortable_chain_(64 * 1024, 12, 4, tp, &ab);
    CHECK_(ab.calls > 0, "abort/reuse: abort callback was observed");
    CHECK_(st == GGML_STATUS_ABORTED, "abort/reuse: compute returned ABORTED");

    float got = run_add_chain_(32 * 1024, 1.0f, 4, 4, tp);
    CHECK_(fabsf(got - 5.0f) < 1e-2f,
           "abort/reuse: same pool works correctly after aborted graph");

    ggml_threadpool_free(tp);
}

int main(void) {
    printf("\n=== abort then reuse same pool ===\n\n");
    test_abort_then_reuse_same_pool();
    printf("\nAll tests passed.\n\n");
    return 0;
}
