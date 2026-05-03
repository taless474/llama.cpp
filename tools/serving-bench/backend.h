#pragma once

#include "harness.h"

#include <future>
#include <string>

namespace serving_bench {

struct engine_init_params {
    std::string model_path;
    int32_t     n_contexts;
    int32_t     n_threads_per_ctx;
    int32_t     ctx_size;
    int32_t     batch_size;
};

class engine {
public:
    engine() = default;
    virtual ~engine() noexcept = default;

    engine(const engine &)             = delete;
    engine & operator=(const engine &) = delete;

    virtual bool init(const engine_init_params & params) = 0;

    virtual std::future<request_result> submit(request_params req) = 0;

    // Returns the number of prompt tokens that the request path would produce
    // for `prompt`. The result MUST come from the same tokenization path
    // (same vocab, same add_special / parse_special flags) used when the
    // request is actually run by submit(); otherwise the prompt-fit check is
    // unreliable. A negative return value indicates tokenization failure.
    virtual int32_t prompt_token_count(const std::string & prompt) const = 0;
};

} // namespace serving_bench
