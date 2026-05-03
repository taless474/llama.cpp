#pragma once

#include "backend.h"

#include <memory>

namespace serving_bench {

std::unique_ptr<engine> make_engine_std();

} // namespace serving_bench
