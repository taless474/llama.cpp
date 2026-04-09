// internal/white-box test:
// requires visibility of struct ggml_threadpool from ggml-cpu-threadpool.h

#include <assert.h>
#include <math.h>
#include <stdatomic.h>
#include <string.h>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cpu-threadpool.h"

static int nearly_equal_f32(float a, float b) {
    return fabsf(a - b) < 1e-6f;
}

static int run_add_graph_on_backend(ggml_backend_t backend, float out[4]) {
    const size_t ctx_size = 256 * 1024;

    struct ggml_init_params params = {
        .mem_size   = ctx_size,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };

    struct ggml_context * ctx = ggml_init(params);
    if (ctx == NULL) {
        return 0;
    }

    struct ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    struct ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    struct ggml_tensor * c = ggml_add(ctx, a, b);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, c);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == NULL) {
        ggml_free(ctx);
        return 0;
    }

    const float av[4] = { 1.f, 2.f, 3.f, 4.f };
    const float bv[4] = { 10.f, 20.f, 30.f, 40.f };

    ggml_backend_tensor_set(a, av, 0, sizeof(av));
    ggml_backend_tensor_set(b, bv, 0, sizeof(bv));

    enum ggml_status st = ggml_backend_graph_compute(backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return 0;
    }

    memset(out, 0, 4 * sizeof(float));
    ggml_backend_tensor_get(c, out, 0, 4 * sizeof(float));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return 1;
}

static void test_shared_executor_not_paused_on_replace(void) {
    printf("  test_shared_executor_not_paused_on_replace ... ");

    struct ggml_threadpool_params shared_params  = ggml_threadpool_params_default(4);
    struct ggml_threadpool_params private_params = ggml_threadpool_params_default(2);

    ggml_threadpool_t tp_shared  = ggml_threadpool_new(&shared_params);
    ggml_threadpool_t tp_private = ggml_threadpool_new(&private_params);

    assert(tp_shared  != NULL);
    assert(tp_private != NULL);

    ggml_backend_t backend_a = ggml_backend_cpu_init();
    ggml_backend_t backend_b = ggml_backend_cpu_init();

    assert(backend_a != NULL);
    assert(backend_b != NULL);

    ggml_backend_cpu_set_n_threads(backend_a, 4);
    ggml_backend_cpu_set_n_threads(backend_b, 4);

    // Both backends borrow the same shared executor.
    ggml_backend_cpu_attach_threadpool(backend_a, tp_shared);
    ggml_backend_cpu_attach_threadpool(backend_b, tp_shared);

    // Sanity: shared executor should not start paused.
    assert(atomic_load_explicit(&tp_shared->pause, memory_order_seq_cst) == false);

    float out_a[4];
    float out_b[4];

    // First compute through A on the shared executor.
    assert(run_add_graph_on_backend(backend_a, out_a));
    assert(nearly_equal_f32(out_a[0], 11.f));
    assert(nearly_equal_f32(out_a[1], 22.f));
    assert(nearly_equal_f32(out_a[2], 33.f));
    assert(nearly_equal_f32(out_a[3], 44.f));

    // A switches away to a private managed executor.
    // Because A previously ATTACHED tp_shared, this must NOT pause tp_shared.
    ggml_backend_cpu_set_threadpool(backend_a, tp_private);

    // White-box check: Patch 3 invariant.
    assert(atomic_load_explicit(&tp_shared->pause, memory_order_seq_cst) == false);

    // B still uses the shared executor and must continue to work.
    assert(run_add_graph_on_backend(backend_b, out_b));
    assert(nearly_equal_f32(out_b[0], 11.f));
    assert(nearly_equal_f32(out_b[1], 22.f));
    assert(nearly_equal_f32(out_b[2], 33.f));
    assert(nearly_equal_f32(out_b[3], 44.f));

    ggml_backend_free(backend_a);
    ggml_backend_free(backend_b);
    ggml_threadpool_free(tp_private);
    ggml_threadpool_free(tp_shared);

    printf("PASS\n");
}

static void test_set_threadpool_managed_semantics_unchanged(void) {
    printf("  test_set_threadpool_managed_semantics_unchanged ... ");

    struct ggml_threadpool_params p1 = ggml_threadpool_params_default(4);
    struct ggml_threadpool_params p2 = ggml_threadpool_params_default(2);

    ggml_threadpool_t tp1 = ggml_threadpool_new(&p1);
    ggml_threadpool_t tp2 = ggml_threadpool_new(&p2);

    GGML_ASSERT(tp1 != NULL);
    GGML_ASSERT(tp2 != NULL);

    ggml_backend_t backend = ggml_backend_cpu_init();
    GGML_ASSERT(backend != NULL);

    float out1[4];
    float out2[4];

    // Managed association with tp1.
    ggml_backend_cpu_set_threadpool(backend, tp1);
    ggml_backend_cpu_set_n_threads(backend, 4);

    GGML_ASSERT(run_add_graph_on_backend(backend, out1));
    GGML_ASSERT(nearly_equal_f32(out1[0], 11.f));
    GGML_ASSERT(nearly_equal_f32(out1[1], 22.f));
    GGML_ASSERT(nearly_equal_f32(out1[2], 33.f));
    GGML_ASSERT(nearly_equal_f32(out1[3], 44.f));

    // tp1 should not be paused while still active.
    GGML_ASSERT(atomic_load_explicit(&tp1->pause, memory_order_seq_cst) == false);

    // Switch to a different managed executor.
    // Legacy behavior: the previously managed executor must be paused.
    ggml_backend_cpu_set_threadpool(backend, tp2);
    ggml_backend_cpu_set_n_threads(backend, 2);

    GGML_ASSERT(atomic_load_explicit(&tp1->pause, memory_order_seq_cst) == true);
    GGML_ASSERT(atomic_load_explicit(&tp2->pause, memory_order_seq_cst) == false);

    // Compute must now succeed on tp2.
    GGML_ASSERT(run_add_graph_on_backend(backend, out2));
    GGML_ASSERT(nearly_equal_f32(out2[0], 11.f));
    GGML_ASSERT(nearly_equal_f32(out2[1], 22.f));
    GGML_ASSERT(nearly_equal_f32(out2[2], 33.f));
    GGML_ASSERT(nearly_equal_f32(out2[3], 44.f));

    ggml_backend_free(backend);
    ggml_threadpool_free(tp2);
    ggml_threadpool_free(tp1);

    printf("PASS\n");
}

int main(void) {
    printf("test-executor-sharing-wb:\n");
    test_shared_executor_not_paused_on_replace();
    test_set_threadpool_managed_semantics_unchanged();
    printf("all tests passed\n");
    return 0;
}
