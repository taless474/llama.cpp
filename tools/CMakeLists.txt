# dependencies

find_package(Threads REQUIRED)

# third-party

# ...

# flags

llama_add_compile_flags()

option(LLAMA_BUILD_SERVING_BENCH "Build the serving benchmark harness" OFF)
option(LLAMA_BUILD_MULTISEQ_GATE  "Build the multi-seq llama_batch gate" OFF)
option(LLAMA_BUILD_HPX_CONTINUOUS_BATCH_GATE
       "Build the HPX continuous-batching prototype" OFF)

# tools

if (EMSCRIPTEN)
else()
    add_subdirectory(batched-bench)
    add_subdirectory(gguf-split)
    add_subdirectory(imatrix)
    add_subdirectory(llama-bench)
    add_subdirectory(completion)
    add_subdirectory(perplexity)
    add_subdirectory(quantize)
    if (LLAMA_BUILD_SERVER)
        add_subdirectory(cli)
        add_subdirectory(server)
    endif()
    add_subdirectory(tokenize)
    add_subdirectory(parser)
    add_subdirectory(tts)
    add_subdirectory(mtmd)
    if (GGML_RPC)
        add_subdirectory(rpc)
    endif()
    if (NOT GGML_BACKEND_DL)
        # these examples use the backends directly and cannot be built with dynamic loading
        add_subdirectory(cvector-generator)
        add_subdirectory(export-lora)
    endif()
    add_subdirectory(fit-params)
    add_subdirectory(results)
    if (LLAMA_BUILD_SERVING_BENCH)
        add_subdirectory(serving-bench)
    endif()
    if (LLAMA_BUILD_MULTISEQ_GATE)
        add_subdirectory(multiseq-batch-gate)
    endif()
    if (LLAMA_BUILD_HPX_CONTINUOUS_BATCH_GATE)
        add_subdirectory(hpx-continuous-batch-gate)
    endif()
endif()
