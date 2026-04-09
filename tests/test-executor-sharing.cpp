// Tests for ggml_backend_cpu_attach_threadpool and the shared-executor contract.
//
// Four tests, in increasing strength:
//
//  1. test_attach_same_executor_two_backends
//     Basic: attach one executor to two backends, compute on each, check results.
//
//  2. test_attach_alternating_dispatch
//     Black-box: interleave A/B dispatches to verify no state bleed across
//     per-dispatch job objects (local_job reset, current_job restore,
//     barrier counters consistent between dispatches on different backends).
//
//  3. test_attach_replace_no_pause_blackbox
//     Black-box: after attaching the same executor to A and B, backend A
//     replaces its executor via set_threadpool.  B must continue to compute
//     correctly.  NOTE: because kickoff always resumes a paused pool, this
//     test passes even with the bug present; it documents intent.
//
//  4. test_attach_pause_state_whitebox
//     White-box: same scenario as (3), but directly asserts that the shared
//     executor's pause flag is false after the replacement.  This is the
//     only test that can actually detect the bug.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static bool nearly_equal_f32(float a, float b) {
    return fabsf(a - b) < 1e-6f;
}

// Run a 4-element float add graph (a+b = [11,22,33,44]) on the given backend.
// Writes results to out[4].  Returns true on success.
static bool run_add_graph_on_backend(ggml_backend_t backend, float out[4]) {
    struct ggml_init_params params;
    params.mem_size   = 256 * 1024;
    params.mem_buffer = NULL;
    params.no_alloc   = true;

    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;

    struct ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    struct ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    struct ggml_tensor * c = ggml_add(ctx, a, b);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, c);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { ggml_free(ctx); return false; }

    const float av[4] = { 1.f, 2.f, 3.f, 4.f };
    const float bv[4] = { 10.f, 20.f, 30.f, 40.f };
    ggml_backend_tensor_set(a, av, 0, sizeof(av));
    ggml_backend_tensor_set(b, bv, 0, sizeof(bv));

    enum ggml_status st = ggml_backend_graph_compute(backend, gf);

    memset(out, 0, 4 * sizeof(float));
    if (st == GGML_STATUS_SUCCESS) {
        ggml_backend_tensor_get(c, out, 0, 4 * sizeof(float));
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return st == GGML_STATUS_SUCCESS;
}

static void check_add_result(const char * label, const float out[4]) {
    assert(nearly_equal_f32(out[0], 11.f));
    assert(nearly_equal_f32(out[1], 22.f));
    assert(nearly_equal_f32(out[2], 33.f));
    assert(nearly_equal_f32(out[3], 44.f));
    (void)label;
}

// ---------------------------------------------------------------------------
// Test 1: basic attach to two backends
// ---------------------------------------------------------------------------

static void test_attach_same_executor_two_backends(void) {
    printf("  test_attach_same_executor_two_backends ... ");

    struct ggml_threadpool_params tpp = ggml_threadpool_params_default(4);
    ggml_threadpool_t tp = ggml_threadpool_new(&tpp);
    assert(tp != NULL);

    ggml_backend_t ba = ggml_backend_cpu_init();
    ggml_backend_t bb = ggml_backend_cpu_init();
    assert(ba && bb);

    ggml_backend_cpu_set_n_threads(ba, 4);
    ggml_backend_cpu_set_n_threads(bb, 4);
    ggml_backend_cpu_attach_threadpool(ba, tp);
    ggml_backend_cpu_attach_threadpool(bb, tp);

    float out_a[4], out_b[4];
    assert(run_add_graph_on_backend(ba, out_a));
    assert(run_add_graph_on_backend(bb, out_b));
    check_add_result("A", out_a);
    check_add_result("B", out_b);

    ggml_backend_free(ba);
    ggml_backend_free(bb);
    ggml_threadpool_free(tp);

    printf("PASS\n");
}

// ---------------------------------------------------------------------------
// Test 2 (black-box): alternating dispatch — no state bleed between backends
// ---------------------------------------------------------------------------

static void test_attach_alternating_dispatch(void) {
    printf("  test_attach_alternating_dispatch ........ ");

    struct ggml_threadpool_params tpp = ggml_threadpool_params_default(4);
    ggml_threadpool_t tp = ggml_threadpool_new(&tpp);
    assert(tp != NULL);

    ggml_backend_t ba = ggml_backend_cpu_init();
    ggml_backend_t bb = ggml_backend_cpu_init();
    assert(ba && bb);

    ggml_backend_cpu_set_n_threads(ba, 4);
    ggml_backend_cpu_set_n_threads(bb, 4);
    ggml_backend_cpu_attach_threadpool(ba, tp);
    ggml_backend_cpu_attach_threadpool(bb, tp);

    // Interleave dispatches: A B A B ... for 20 rounds each.
    // Exercises that local_job is fully reset on each call and that
    // n_barrier/n_barrier_passed stay consistent across alternating backends.
    float out[4];
    for (int i = 0; i < 20; i++) {
        assert(run_add_graph_on_backend(ba, out));
        check_add_result("A", out);
        assert(run_add_graph_on_backend(bb, out));
        check_add_result("B", out);
    }

    ggml_backend_free(ba);
    ggml_backend_free(bb);
    ggml_threadpool_free(tp);

    printf("PASS\n");
}

// ---------------------------------------------------------------------------
// Test 3 (black-box): replace does not break the other backend's compute
// ---------------------------------------------------------------------------

static void test_attach_replace_no_pause_blackbox(void) {
    printf("  test_attach_replace_no_pause_blackbox ... ");

    // NOTE: because ggml_graph_compute_kickoff resumes a paused pool before
    // broadcasting, this test passes even if the bug is present (tp would be
    // paused but immediately resumed on next kickoff).  It documents intended
    // behaviour; test 4 provides the actual regression guard.

    struct ggml_threadpool_params tpp  = ggml_threadpool_params_default(4);
    struct ggml_threadpool_params tpp2 = ggml_threadpool_params_default(1);
    ggml_threadpool_t tp  = ggml_threadpool_new(&tpp);
    ggml_threadpool_t tp2 = ggml_threadpool_new(&tpp2);
    assert(tp && tp2);

    ggml_backend_t ba = ggml_backend_cpu_init();
    ggml_backend_t bb = ggml_backend_cpu_init();
    assert(ba && bb);

    ggml_backend_cpu_set_n_threads(ba, 4);
    ggml_backend_cpu_set_n_threads(bb, 4);
    ggml_backend_cpu_attach_threadpool(ba, tp);
    ggml_backend_cpu_attach_threadpool(bb, tp);

    float out[4];
    assert(run_add_graph_on_backend(ba, out));
    check_add_result("A pre-replace", out);

    // Replace ba's executor.  With correct code: tp is not paused.
    // With the bug: tp is paused, but kickoff resumes it, so B still succeeds.
    ggml_backend_cpu_set_threadpool(ba, tp2);

    assert(run_add_graph_on_backend(bb, out));
    check_add_result("B post-replace", out);

    ggml_backend_free(ba);
    ggml_backend_free(bb);
    ggml_threadpool_free(tp);
    ggml_threadpool_free(tp2);

    printf("PASS\n");
}

// ---------------------------------------------------------------------------
// Test 4 (white-box): pause flag must be false after replacing an external attachment
// ---------------------------------------------------------------------------

static void test_attach_pause_state_whitebox(void) {
    printf("  test_attach_pause_state_whitebox ........ ");

    struct ggml_threadpool_params tpp  = ggml_threadpool_params_default(4);
    struct ggml_threadpool_params tpp2 = ggml_threadpool_params_default(1);
    ggml_threadpool_t tp  = ggml_threadpool_new(&tpp);
    ggml_threadpool_t tp2 = ggml_threadpool_new(&tpp2);
    assert(tp && tp2);

    ggml_backend_t ba = ggml_backend_cpu_init();
    ggml_backend_t bb = ggml_backend_cpu_init();
    assert(ba && bb);

    ggml_backend_cpu_set_n_threads(ba, 4);
    ggml_backend_cpu_set_n_threads(bb, 4);
    ggml_backend_cpu_attach_threadpool(ba, tp);
    ggml_backend_cpu_attach_threadpool(bb, tp);

    float out[4];
    assert(run_add_graph_on_backend(ba, out));

    // ba replaces its external attachment with tp2.
    // Because ba held tp as external, set_threadpool must NOT call
    // ggml_threadpool_pause(tp).
    ggml_backend_cpu_set_threadpool(ba, tp2);

    // White-box assertion: tp->pause must still be false.
    assert(!ggml_threadpool_is_paused(tp));

    // Confirm tp is still live: bb must compute correctly.
    assert(run_add_graph_on_backend(bb, out));
    check_add_result("B post-replace", out);

    ggml_backend_free(ba);
    ggml_backend_free(bb);
    ggml_threadpool_free(tp);
    ggml_threadpool_free(tp2);

    printf("PASS\n");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(void) {
    printf("test-executor-sharing:\n");

    test_attach_same_executor_two_backends();
    test_attach_alternating_dispatch();
    test_attach_replace_no_pause_blackbox();
    test_attach_pause_state_whitebox();

    printf("all tests passed\n");
    return 0;
}
