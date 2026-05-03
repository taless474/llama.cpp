set(TARGET llama-serving-bench)

option(LLAMA_SERVING_BENCH_HPX
       "Build the HPX backend for llama-serving-bench" OFF)

add_executable(${TARGET}
    main.cpp
    harness.cpp
    harness.h
    backend.h
    backend_std.h
    backend_std.cpp
    backend_hpx.h
    backend_hpx.cpp
    runtime_hpx.h
    runtime_hpx.cpp
)

install(TARGETS ${TARGET} RUNTIME)

target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(${TARGET} PRIVATE llama-common ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if (LLAMA_SERVING_BENCH_HPX)
    find_package(HPX REQUIRED)
    target_compile_definitions(${TARGET} PRIVATE LLAMA_SERVING_BENCH_HPX=1)
    target_link_libraries(${TARGET} PRIVATE HPX::hpx)
endif()
