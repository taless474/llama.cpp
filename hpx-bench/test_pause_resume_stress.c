#include "test_ggml_threadpool_common.h"

/* Stress pause/resume by repeatedly resuming, dispatching, and pausing the
 * same pool. This is much stronger than a one-shot sanity check. */

static void test_pause_resume_stress(void) {
    struct ggml_threadpool_params p = ggml_threadpool_params_default(4);
    p.paused = true;

    struct ggml_threadpool *tp = ggml_threadpool_new(&p);
    CHECK_(tp != NULL, "pause/resume stress: pool created paused");

    int counts[] = {4, 2, 1, 3, 4};
    const int ncounts = (int) (sizeof(counts) / sizeof(counts[0]));

    for (int i = 0; i < 100; ++i) {
        ggml_threadpool_resume(tp);

        float got = run_add_chain_(16 * 1024, 1.0f, 5, counts[i % ncounts], tp);
        CHECK_(fabsf(got - 6.0f) < 1e-2f,
               "pause/resume stress: correct result after resume");

        ggml_threadpool_pause(tp);
    }

    ggml_threadpool_free(tp);
    pass_("pause/resume stress: free after many pause/resume cycles (no deadlock)");
}

int main(void) {
    printf("\n=== pause/resume stress ===\n\n");
    test_pause_resume_stress();
    printf("\nAll tests passed.\n\n");
    return 0;
}
