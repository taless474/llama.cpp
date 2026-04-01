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
else()
    message(STATUS "test_ggml_threadpool: skipped (set -DGGML_BUILD_DIR=<path>)")
endif()
