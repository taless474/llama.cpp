// ggml-hpx-exec.cpp
//
// Stub implementation — executor dispatch not yet wired.
// Satisfies the linker for test builds.

#include "ggml-hpx-exec.h"
#include "ggml-hpx-abort.h"
#include "ggml-hpx-runtime.h"

struct ggml_hpx_exec
{
    ggml_hpx_runtime*    runtime = nullptr;
    ggml_hpx_abort_token abort{};
};

ggml_hpx_exec* ggml_hpx_exec_create(
    ggml_hpx_exec_params const& params)
{
    auto* exec     = new ggml_hpx_exec{};
    exec->runtime  = ggml_hpx_runtime_create(params.runtime);
    return exec;
}

void ggml_hpx_exec_destroy(ggml_hpx_exec* exec)
{
    ggml_hpx_runtime_destroy(exec->runtime);
    delete exec;
}

void ggml_hpx_exec_abort(ggml_hpx_exec* exec)
{
    exec->abort.request();
}

ggml_hpx_exec_status ggml_hpx_exec_run_decode(
    ggml_hpx_exec* exec, ggml_cgraph const* /*graph*/)
{
    if (exec->abort.check())
    {
        exec->abort.reset();
        return ggml_hpx_exec_status::aborted;
    }
    return ggml_hpx_exec_status::ok;
}

ggml_hpx_exec_status ggml_hpx_exec_run_prefill(
    ggml_hpx_exec* exec, ggml_cgraph const* /*graph*/,
    ggml_backend_sched_t /*sched*/)
{
    if (exec->abort.check())
    {
        exec->abort.reset();
        return ggml_hpx_exec_status::aborted;
    }
    return ggml_hpx_exec_status::ok;
}
