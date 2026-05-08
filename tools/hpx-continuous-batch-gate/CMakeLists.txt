set(TARGET llama-hpx-continuous-batch-gate)
add_executable(${TARGET} hpx-continuous-batch-gate.cpp)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_link_libraries(${TARGET}
    PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})

# HPX is required for this tool. The local install path is provided
# via -DHPX_DIR=... when the parent build is configured (see the
# tool's README).
find_package(HPX REQUIRED)
target_link_libraries(${TARGET} PRIVATE HPX::hpx)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
