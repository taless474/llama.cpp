# ggml-hpx backend CMakeLists.txt
#
# Gated by GGML_HPX in the parent CMakeLists.
# Included only when GGML_HPX is ON.
#
# Note:
#   Two translation units include ggml-backend.h:
#     ggml-hpx-adapter.cpp — scheduler coupling and split state
#     ggml-hpx-exec.cpp    — ggml_backend_graph_compute and
#                            ggml_backend_sched_get_tensor_backend at run time
#   All other files in this directory must not include ggml-backend.h directly.

find_package(HPX REQUIRED)

message(STATUS "HPX found: ${HPX_VERSION}")

ggml_add_backend_library(ggml-hpx
    ggml-hpx-adapter.cpp
    ggml-hpx-cache.cpp
    ggml-hpx-exec.cpp
    ggml-hpx-plan.cpp
    ggml-hpx-runtime.cpp
    ggml-hpx-tpool.cpp
    ggml-hpx-instrument.cpp
)

target_include_directories(ggml-hpx
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../ggml-cpu
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_link_libraries(ggml-hpx
    PRIVATE
        HPX::hpx
        # ggml-cpu: needed by the parallel-chain prototype in ggml-hpx-exec.cpp
        # to create per-chain CPU backends (ggml_backend_cpu_init /
        # ggml_backend_cpu_set_n_threads).  Remove when the prototype is
        # replaced by a registration-based backend factory.
        ggml-cpu
)

target_compile_features(ggml-hpx PRIVATE cxx_std_17)
