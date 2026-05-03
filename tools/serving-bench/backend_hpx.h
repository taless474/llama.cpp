#pragma once

#include "backend.h"

#include <memory>

namespace serving_bench {

// HPX serving engine factory.
//
// The caller MUST ensure the process-wide HPX runtime has been started via
// hpx_runtime_start_once() (see runtime_hpx.h) before calling this.
// Engine instances do not start or stop the HPX runtime — that lifecycle is
// process-level and one-shot, per the project's HPX hard invariants.
//
// When LLAMA_SERVING_BENCH_HPX is OFF, this factory returns nullptr and
// logs a clean "HPX backend not built" message; the std backend is unaffected.
std::unique_ptr<engine> make_engine_hpx();

} // namespace serving_bench
