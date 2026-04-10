// ggml-hpx-tpool.h
//
// HPX executor ops for the ggml_threadpool substrate.
//
// Call once during hpx_acquire() to redirect all future threadpool
// allocations to the HPX substrate:
//
//   ggml_cpu_set_executor_ops(ggml_hpx_tpool_get_ops());
//
// The returned pointer is valid for the lifetime of the process.

#pragma once

// struct ggml_cpu_executor_ops, ggml_cpu_set_executor_ops,
// ggml_graph_compute_thread_run, ggml_threadpool_worker,
// ggml_threadpool_destroy_substrate.
#include "ggml-cpu-executor.h"

#ifdef __cplusplus
extern "C"
{
#endif

// Returns a pointer to the static HPX executor ops table.
const struct ggml_cpu_executor_ops* ggml_hpx_tpool_get_ops(void);

#ifdef __cplusplus
}
#endif
