cmake_minimum_required(VERSION 3.18)
project(hpx_bench CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)

# ── Stage 1: HPX primitives smoke test ───────────────────────────────────────
find_package(HPX REQUIRED)

add_executable(test_hpx_primitives test_hpx_primitives.cpp)
target_link_libraries(test_hpx_primitives PRIVATE HPX::hpx HPX::wrap_main)

# ── Stage 2: ggml threadpool contract test ────────────────────────────────────
# Set GGML_BUILD_DIR to the llama build directory to link against.
# Example:
#   cmake ... -DGGML_BUILD_DIR=../build-pthread-blas   (green: baseline)
#   cmake ... -DGGML_BUILD_DIR=../build-hpx-blas       (red until HPX impl done)

if(DEFINED GGML_BUILD_DIR)
    set(_bd "${GGML_BUILD_DIR}")

    add_executable(test_ggml_threadpool test_ggml_threadpool.c)

    target_include_directories(test_ggml_threadpool PRIVATE
        ${CMAKE_SOURCE_DIR}/../ggml/include
        ${CMAKE_SOURCE_DIR}/../include
    )

    target_link_directories(test_ggml_threadpool PRIVATE ${_bd}/bin)

    target_link_libraries(test_ggml_threadpool PRIVATE
        ggml-cpu
        ggml-base
        ggml
        m
    )

    set_target_properties(test_ggml_threadpool PROPERTIES
        BUILD_RPATH "${_bd}/bin"
    )

    message(STATUS "test_ggml_threadpool: linking against ${_bd}")

    add_executable(test_barrier_churn test_barrier_churn.c)
    target_include_directories(test_barrier_churn PRIVATE
        ${CMAKE_SOURCE_DIR}/../ggml/include
        ${CMAKE_SOURCE_DIR}/../include
    )
    target_link_directories(test_barrier_churn PRIVATE ${_bd}/bin)
    target_link_libraries(test_barrier_churn PRIVATE
        ggml-cpu
        ggml-base
        ggml
        m
    )
    set_target_properties(test_barrier_churn PROPERTIES
        BUILD_RPATH "${_bd}/bin"
    )

    add_executable(bench_dispatch_overhead bench_dispatch_overhead.c)
    target_include_directories(bench_dispatch_overhead PRIVATE
        ${CMAKE_SOURCE_DIR}/../ggml/include
        ${CMAKE_SOURCE_DIR}/../include
    )
    target_link_directories(bench_dispatch_overhead PRIVATE ${_bd}/bin)
    target_link_libraries(bench_dispatch_overhead PRIVATE ggml-cpu ggml-base ggml m)
    set_target_properties(bench_dispatch_overhead PROPERTIES BUILD_RPATH "${_bd}/bin")

    # ── New contract/stress tests ─────────────────────────────────────────────
    macro(add_ggml_test _tgt _src)
        add_executable(${_tgt} ${_src})
        target_include_directories(${_tgt} PRIVATE
            ${CMAKE_SOURCE_DIR}/../ggml/include
            ${CMAKE_SOURCE_DIR}/../include
        )
        target_link_directories(${_tgt} PRIVATE ${_bd}/bin)
        target_link_libraries(${_tgt} PRIVATE ggml-cpu ggml-base ggml m)
        set_target_properties(${_tgt} PROPERTIES BUILD_RPATH "${_bd}/bin")
    endmacro()

    add_ggml_test(test_startup_race_min                   test_startup_race_min.c)
    add_ggml_test(test_threshold                          test_threshold.c)
    add_ggml_test(test_warmup_effect                      test_warmup_effect.c)
    add_ggml_test(test_reuse_hang                         test_reuse_hang.c)
    add_ggml_test(test_disposable_threadpool_repeat       test_disposable_threadpool_repeat.c)
    add_ggml_test(test_abort_then_reuse_same_pool         test_abort_then_reuse_same_pool.c)
    add_ggml_test(test_mixed_workload_reuse_same_pool     test_mixed_workload_reuse_same_pool.c)
    add_ggml_test(test_mul_mat_variable_thread_counts_same_pool test_mul_mat_variable_thread_counts_same_pool.c)
    add_ggml_test(test_pause_resume_stress                test_pause_resume_stress.c)
    add_ggml_test(test_two_pools_isolation                test_two_pools_isolation.c)

else()
    message(STATUS "test_ggml_threadpool: skipped (set -DGGML_BUILD_DIR=<path>)")
endif()