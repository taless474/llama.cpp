# ggml-hpx backend CMakeLists.txt
#
# Gated by GGML_HPX in the parent CMakeLists.
# Included only when GGML_HPX is ON.
#
# Note:
#   ggml-hpx-adapter.cpp is intended to be the only translation unit
#   that includes ggml-backend.h. Enforce this in CI or with a source
#   check target; this file only documents the rule.

find_package(HPX REQUIRED)

message(STATUS "HPX found: ${HPX_VERSION}")

ggml_add_backend_library(ggml-hpx
    ggml-hpx-adapter.cpp
    ggml-hpx-cache.cpp
    ggml-hpx-exec.cpp
    ggml-hpx-plan.cpp
    ggml-hpx-runtime.cpp
    ggml-hpx-instrument.cpp
)

target_include_directories(ggml-hpx
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_link_libraries(ggml-hpx
    PRIVATE
        HPX::hpx
)

target_compile_features(ggml-hpx PRIVATE cxx_std_17)
