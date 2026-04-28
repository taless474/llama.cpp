# HPX executor tests
#
# Included from tests/CMakeLists.txt when GGML_HPX is ON.
# CTest is already enabled in the parent scope.
#
# All test targets link ggml-hpx and HPX::hpx. Most tests exercise the HPX
# layer through its opaque interface (exec, plan, cache, abort) and do not
# use HPX APIs directly, but HPX::hpx is linked uniformly to avoid subtle
# build differences between targets.
#
# Include isolation (ggml-backend.h not included by plan/exec/cache TUs) is
# a compile-time contract. Enforce it in CI with a grep or include-graph
# check; this file does not replicate that enforcement.

find_package(HPX REQUIRED)
find_package(GTest REQUIRED)

function(hpx_add_test source)
    get_filename_component(TEST_TARGET ${source} NAME_WE)
    add_executable(${TEST_TARGET} ${source})
    target_link_libraries(${TEST_TARGET}
        PRIVATE
            ggml-hpx
            ggml
            HPX::hpx
            GTest::gtest_main
    )
    target_include_directories(${TEST_TARGET}
        PRIVATE
            ${PROJECT_SOURCE_DIR}/ggml/src/ggml-hpx
            ${PROJECT_SOURCE_DIR}/ggml/include
    )
    target_compile_features(${TEST_TARGET} PRIVATE cxx_std_17)
    add_test(NAME ${TEST_TARGET} COMMAND ${TEST_TARGET})
endfunction()

# Structural contract invariants (key comparison, mode enum, status values).
# hpx_add_test(test_hpx_contract.cpp)   # not yet written

# Unit tests per layer, bottom-up.
hpx_add_test(test_hpx_abort.cpp)
hpx_add_test(test_hpx_cache.cpp)
hpx_add_test(test_hpx_decode_plan.cpp)
hpx_add_test(test_hpx_prefill_plan.cpp)
hpx_add_test(test_hpx_adapter.cpp)
hpx_add_test(test_hpx_runtime.cpp)
hpx_add_test(test_hpx_exec.cpp)

# End-to-end llama.cpp smoke test.  Requires LLAMACPP_TEST_MODELFILE to be set;
# skipped (not failed) when absent.  Needs the llama library for model loading.
add_executable(test_hpx_llama_smoke test_hpx_llama_smoke.cpp)
target_link_libraries(test_hpx_llama_smoke
    PRIVATE
        llama
        ggml-hpx
        ggml
        HPX::hpx
        GTest::gtest_main
)
target_include_directories(test_hpx_llama_smoke
    PRIVATE
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-hpx
        ${PROJECT_SOURCE_DIR}/ggml/include
        ${PROJECT_SOURCE_DIR}/include
)
target_compile_features(test_hpx_llama_smoke PRIVATE cxx_std_17)
add_test(NAME test_hpx_llama_smoke COMMAND test_hpx_llama_smoke)

# Selective HPX mul_mat smoke test.  Same skip-on-missing-model semantics.
# Gated on GGML_HPX_REGION_DAG because LLAMA_HPX_SELECTIVE_MUL_MAT=1 is a
# no-op unless the region-dag path is compiled in.
if(GGML_HPX_REGION_DAG)
    add_executable(test_hpx_llama_selective_mul_mat_smoke
        test_hpx_llama_selective_mul_mat_smoke.cpp)
    target_compile_definitions(test_hpx_llama_selective_mul_mat_smoke
        PRIVATE GGML_HPX_REGION_DAG)
    target_link_libraries(test_hpx_llama_selective_mul_mat_smoke
        PRIVATE
            llama
            ggml-hpx
            ggml
            HPX::hpx
            GTest::gtest_main
    )
    target_include_directories(test_hpx_llama_selective_mul_mat_smoke
        PRIVATE
            ${PROJECT_SOURCE_DIR}/ggml/src/ggml-hpx
            ${PROJECT_SOURCE_DIR}/ggml/include
            ${PROJECT_SOURCE_DIR}/include
    )
    target_compile_features(test_hpx_llama_selective_mul_mat_smoke PRIVATE cxx_std_17)
    add_test(NAME test_hpx_llama_selective_mul_mat_smoke
             COMMAND test_hpx_llama_selective_mul_mat_smoke)
endif()

# Stress test: concurrent abort, repeated runs, cache churn.
# hpx_add_test(test_hpx_stress.cpp)     # not yet written

# Fine-grained region DAG tests — gated on GGML_HPX_REGION_DAG.
# These tests exercise ggml-hpx-region-exec.{h,cpp} directly.
# The executor tests (run and mul_mat) construct fork_join_executor, which
# requires an HPX thread context; each test body wraps the executor call in
# hpx::async().get() after calling ggml_hpx_tpool_start().
if(GGML_HPX_REGION_DAG)
    function(hpx_add_region_dag_test source)
        get_filename_component(TEST_TARGET ${source} NAME_WE)
        add_executable(${TEST_TARGET} ${source})
        target_compile_definitions(${TEST_TARGET} PRIVATE GGML_HPX_REGION_DAG)
        target_link_libraries(${TEST_TARGET}
            PRIVATE
                ggml-hpx
                ggml
                HPX::hpx
                GTest::gtest_main
        )
        target_include_directories(${TEST_TARGET}
            PRIVATE
                ${PROJECT_SOURCE_DIR}/ggml/include
        )
        target_compile_features(${TEST_TARGET} PRIVATE cxx_std_17)
        add_test(NAME ${TEST_TARGET} COMMAND ${TEST_TARGET})
    endfunction()

    hpx_add_region_dag_test(test_hpx_region_group_validate.cpp)
    hpx_add_region_dag_test(test_hpx_region_group_run.cpp)
    hpx_add_region_dag_test(test_hpx_region_single_mul_mat.cpp)
    hpx_add_region_dag_test(test_hpx_region_group_parallel.cpp)
    # Phase 3: resource-model proofs (reduction_buffer + lane_scratch).
    hpx_add_region_dag_test(test_hpx_region_group_reduction.cpp)
    hpx_add_region_dag_test(test_hpx_region_group_lane_scratch.cpp)
    # Phase 4: first real ggml-style CPU op on the fine-region DAG (RMS_NORM F32).
    hpx_add_region_dag_test(test_hpx_region_group_rms_norm_f32.cpp)
    # Phase 5 M1: new elementwise primitives (SiLU F32, elementwise MUL F32).
    hpx_add_region_dag_test(test_hpx_region_primitives_silu_mul_f32.cpp)
    # Milestone B.1 primitive: fused SwiGLU F32 (SiLU(gate) * up in one pass).
    hpx_add_region_dag_test(test_hpx_region_primitives_swiglu_f32.cpp)
    # Phase 5 M2: op lowering — ggml_hpx_lower_op structural + execution tests.
    hpx_add_region_dag_test(test_hpx_lower_op.cpp)
    # Phase 5 M3: MLP gate/up composer — structural + execution tests.
    hpx_add_region_dag_test(test_hpx_mlp_gate_up_compose.cpp)
    # Phase 5 M4: MLP gate/up frozen packet — compile + bind + run test.
    hpx_add_region_dag_test(test_hpx_packet_mlp_gate_up_f32.cpp)
    # Milestone B.1: fused-SWIGLU MLP GLU frozen packet — compile + bind + run test.
    hpx_add_region_dag_test(test_hpx_packet_mlp_glu_f32.cpp)
    # Phase 5 M5: mixed execution — mul_mat lowered to fine-region, neg to CPU fallback.
    # Re-compiles ggml-hpx-exec-selective.cpp with GGML_HPX_EXEC_SELECTIVE_TESTING so
    # the instrumentation atomics are present and the extern declarations resolve.
    add_executable(test_hpx_region_mixed_mul_mat
        test_hpx_region_mixed_mul_mat.cpp
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-hpx/ggml-hpx-exec-selective.cpp
    )
    target_compile_definitions(test_hpx_region_mixed_mul_mat
        PRIVATE GGML_HPX_REGION_DAG GGML_HPX_EXEC_SELECTIVE_TESTING)
    target_link_libraries(test_hpx_region_mixed_mul_mat
        PRIVATE ggml-hpx ggml HPX::hpx GTest::gtest_main)
    target_include_directories(test_hpx_region_mixed_mul_mat
        PRIVATE
            ${PROJECT_SOURCE_DIR}/ggml/include
            ${PROJECT_SOURCE_DIR}/ggml/src
            ${PROJECT_SOURCE_DIR}/ggml/src/ggml-hpx
            ${PROJECT_SOURCE_DIR}/ggml/src/ggml-cpu)
    target_compile_features(test_hpx_region_mixed_mul_mat PRIVATE cxx_std_17)
    add_test(NAME test_hpx_region_mixed_mul_mat COMMAND test_hpx_region_mixed_mul_mat)

    # Phase 5 M6: selective MLP gate/up packet dispatch — isolation test for
    # matcher + cache miss/hit + bind-per-call + frozen-packet dispatch
    # through ggml_hpx_exec_graph_selective_mul_mat.  Same re-compile pattern
    # as test_hpx_region_mixed_mul_mat so the MLP packet instrumentation
    # atomics are present.
    add_executable(test_hpx_selective_mlp_gate_up_packet
        test_hpx_selective_mlp_gate_up_packet.cpp
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-hpx/ggml-hpx-exec-selective.cpp
    )
    target_compile_definitions(test_hpx_selective_mlp_gate_up_packet
        PRIVATE GGML_HPX_REGION_DAG GGML_HPX_EXEC_SELECTIVE_TESTING)
    target_link_libraries(test_hpx_selective_mlp_gate_up_packet
        PRIVATE ggml-hpx ggml HPX::hpx GTest::gtest_main)
    target_include_directories(test_hpx_selective_mlp_gate_up_packet
        PRIVATE
            ${PROJECT_SOURCE_DIR}/ggml/include
            ${PROJECT_SOURCE_DIR}/ggml/src
            ${PROJECT_SOURCE_DIR}/ggml/src/ggml-hpx
            ${PROJECT_SOURCE_DIR}/ggml/src/ggml-cpu)
    target_compile_features(test_hpx_selective_mlp_gate_up_packet PRIVATE cxx_std_17)
    add_test(NAME test_hpx_selective_mlp_gate_up_packet
             COMMAND test_hpx_selective_mlp_gate_up_packet)

    # B.1 Item 5: selective MLP GLU (SWIGLU) packet dispatch — isolation test
    # for the GLU prescan + cache miss/hit + bind-per-call + frozen-packet
    # dispatch through ggml_hpx_exec_graph_selective_mul_mat.  Same re-compile
    # pattern so the GLU packet instrumentation atomics are present.
    add_executable(test_hpx_selective_mlp_glu_packet
        test_hpx_selective_mlp_glu_packet.cpp
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-hpx/ggml-hpx-exec-selective.cpp
    )
    target_compile_definitions(test_hpx_selective_mlp_glu_packet
        PRIVATE GGML_HPX_REGION_DAG GGML_HPX_EXEC_SELECTIVE_TESTING)
    target_link_libraries(test_hpx_selective_mlp_glu_packet
        PRIVATE ggml-hpx ggml HPX::hpx GTest::gtest_main)
    target_include_directories(test_hpx_selective_mlp_glu_packet
        PRIVATE
            ${PROJECT_SOURCE_DIR}/ggml/include
            ${PROJECT_SOURCE_DIR}/ggml/src
            ${PROJECT_SOURCE_DIR}/ggml/src/ggml-hpx
            ${PROJECT_SOURCE_DIR}/ggml/src/ggml-cpu)
    target_compile_features(test_hpx_selective_mlp_glu_packet PRIVATE cxx_std_17)
    add_test(NAME test_hpx_selective_mlp_glu_packet
             COMMAND test_hpx_selective_mlp_glu_packet)

    # B4: Q4_K MUL_MAT through the HPX lowered path.
    # Proves that the selective executor routes a Q4_K × F32 MUL_MAT through
    # the fine-region (lowered) path rather than the CPU-backend fallback, and
    # that the output matches the ggml CPU backend reference.
    add_executable(test_hpx_selective_mul_mat_q4_k
        test_hpx_selective_mul_mat_q4_k.cpp)
    target_compile_definitions(test_hpx_selective_mul_mat_q4_k
        PRIVATE GGML_HPX_REGION_DAG)
    target_link_libraries(test_hpx_selective_mul_mat_q4_k
        PRIVATE ggml-hpx ggml HPX::hpx GTest::gtest_main)
    target_include_directories(test_hpx_selective_mul_mat_q4_k
        PRIVATE
            ${PROJECT_SOURCE_DIR}/ggml/include
            ${PROJECT_SOURCE_DIR}/ggml/src
            ${PROJECT_SOURCE_DIR}/ggml/src/ggml-hpx
            ${PROJECT_SOURCE_DIR}/ggml/src/ggml-cpu)
    target_compile_features(test_hpx_selective_mul_mat_q4_k PRIVATE cxx_std_17)
    add_test(NAME test_hpx_selective_mul_mat_q4_k
             COMMAND test_hpx_selective_mul_mat_q4_k)

    # CPU_REPACK Q4_K (q4_K_8x8_q8_K trait) MUL_MAT through the HPX lowered
    # path.  Companion to test_hpx_selective_mul_mat_q4_k; allocates the
    # weight via the CPU_REPACK buffer type so init_tensor populates extra
    # and set_tensor repacks the bytes into block_q4_Kx8 layout.  The
    # selective executor must accept the node, run the new R0 (quantize)
    # + R1 (gemv_q4_K_8x8_q8_K) group, and produce output bit-equal to
    # the ggml CPU backend reference.
    add_executable(test_hpx_selective_mul_mat_q4_k_repacked
        test_hpx_selective_mul_mat_q4_k_repacked.cpp)
    target_compile_definitions(test_hpx_selective_mul_mat_q4_k_repacked
        PRIVATE GGML_HPX_REGION_DAG)
    target_link_libraries(test_hpx_selective_mul_mat_q4_k_repacked
        PRIVATE ggml-hpx ggml HPX::hpx GTest::gtest_main)
    target_include_directories(test_hpx_selective_mul_mat_q4_k_repacked
        PRIVATE
            ${PROJECT_SOURCE_DIR}/ggml/include
            ${PROJECT_SOURCE_DIR}/ggml/src
            ${PROJECT_SOURCE_DIR}/ggml/src/ggml-hpx
            ${PROJECT_SOURCE_DIR}/ggml/src/ggml-cpu)
    target_compile_features(test_hpx_selective_mul_mat_q4_k_repacked PRIVATE cxx_std_17)
    add_test(NAME test_hpx_selective_mul_mat_q4_k_repacked
             COMMAND test_hpx_selective_mul_mat_q4_k_repacked)

    # B.2 QBRIDGE: selective MLP GLU with quantized (F16) weights.
    # Proves the QBRIDGE prescan, 1-step packet compile, CPU-fallback gate/up,
    # SWIGLU packet dispatch, and cache-key separation from the F32 path.
    add_executable(test_hpx_selective_mlp_glu_qbridge
        test_hpx_selective_mlp_glu_qbridge.cpp
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-hpx/ggml-hpx-exec-selective.cpp
    )
    target_compile_definitions(test_hpx_selective_mlp_glu_qbridge
        PRIVATE GGML_HPX_REGION_DAG GGML_HPX_EXEC_SELECTIVE_TESTING)
    target_link_libraries(test_hpx_selective_mlp_glu_qbridge
        PRIVATE ggml-hpx ggml HPX::hpx GTest::gtest_main)
    target_include_directories(test_hpx_selective_mlp_glu_qbridge
        PRIVATE
            ${PROJECT_SOURCE_DIR}/ggml/include
            ${PROJECT_SOURCE_DIR}/ggml/src
            ${PROJECT_SOURCE_DIR}/ggml/src/ggml-hpx
            ${PROJECT_SOURCE_DIR}/ggml/src/ggml-cpu)
    target_compile_features(test_hpx_selective_mlp_glu_qbridge PRIVATE cxx_std_17)
    add_test(NAME test_hpx_selective_mlp_glu_qbridge
             COMMAND test_hpx_selective_mlp_glu_qbridge)
endif()
