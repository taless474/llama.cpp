#include "backend_std.h"

#include "harness.h"

#include "llama.h"

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace serving_bench {

namespace {

using steady_clock = std::chrono::steady_clock;

struct work_item {
    request_params               req;
    steady_clock::time_point     t_submit;
    std::promise<request_result> promise;
};

std::vector<llama_token> tokenize_prompt(const llama_vocab * vocab, const std::string & prompt) {
    std::vector<llama_token> tokens(prompt.size() + 16);
    int32_t n = llama_tokenize(
        vocab,
        prompt.c_str(), static_cast<int32_t>(prompt.size()),
        tokens.data(), static_cast<int32_t>(tokens.size()),
        true, true);
    if (n < 0) {
        tokens.resize(static_cast<size_t>(-n));
        n = llama_tokenize(
            vocab,
            prompt.c_str(), static_cast<int32_t>(prompt.size()),
            tokens.data(), static_cast<int32_t>(tokens.size()),
            true, true);
        if (n < 0) {
            return {};
        }
    }
    tokens.resize(static_cast<size_t>(n));
    return tokens;
}

llama_token argmax(const float * logits, int32_t n_vocab) {
    llama_token best   = 0;
    float       best_v = logits[0];
    for (int32_t i = 1; i < n_vocab; i++) {
        if (logits[i] > best_v) {
            best_v = logits[i];
            best   = static_cast<llama_token>(i);
        }
    }
    return best;
}

class engine_std final : public engine {
public:
    engine_std() = default;

    ~engine_std() noexcept override {
        shutdown();
    }

    bool init(const engine_init_params & params) override {
        const int32_t n_workers = params.n_contexts > 0 ? params.n_contexts : 1;

        if (params.ctx_size <= 0) {
            std::fprintf(stderr, "[serving-bench] invalid ctx_size=%d\n", params.ctx_size);
            return false;
        }
        if (params.batch_size <= 0) {
            std::fprintf(stderr, "[serving-bench] invalid batch_size=%d\n", params.batch_size);
            return false;
        }

        int32_t n_threads = params.n_threads_per_ctx;
        if (n_threads <= 0) {
            const uint32_t hw = std::thread::hardware_concurrency();
            const int32_t hw_i = hw > 0 ? static_cast<int32_t>(hw) : 1;
            n_threads = hw_i / n_workers;
            if (n_threads < 1) {
                n_threads = 1;
            }
        }

        llama_model_params mparams = llama_model_default_params();
        mparams.n_gpu_layers = 0;

        model_ = llama_model_load_from_file(params.model_path.c_str(), mparams);
        if (model_ == nullptr) {
            std::fprintf(stderr, "[serving-bench] failed to load model: %s\n",
                         params.model_path.c_str());
            return false;
        }

        vocab_   = llama_model_get_vocab(model_);
        n_vocab_ = llama_vocab_n_tokens(vocab_);

        contexts_.resize(static_cast<size_t>(n_workers), nullptr);
        for (int32_t i = 0; i < n_workers; i++) {
            llama_context_params cparams = llama_context_default_params();
            cparams.n_ctx           = static_cast<uint32_t>(params.ctx_size);
            cparams.n_batch         = static_cast<uint32_t>(params.batch_size);
            cparams.n_ubatch        = static_cast<uint32_t>(params.batch_size);
            cparams.n_threads       = n_threads;
            cparams.n_threads_batch = n_threads;
            cparams.no_perf         = true;

            llama_context * ctx = llama_init_from_model(model_, cparams);
            if (ctx == nullptr) {
                std::fprintf(stderr, "[serving-bench] failed to init context %d\n", i);
                return false;
            }
            contexts_[i] = ctx;
        }

        std::fprintf(stderr,
                     "[serving-bench] engine_std ready: n_contexts=%d n_threads_per_ctx=%d "
                     "ctx_size=%d batch_size=%d\n",
                     n_workers, n_threads, params.ctx_size, params.batch_size);

        running_ = true;
        for (int32_t i = 0; i < n_workers; i++) {
            workers_.emplace_back(&engine_std::worker_loop, this, i);
        }

        return true;
    }

    int32_t prompt_token_count(const std::string & prompt) const override {
        if (vocab_ == nullptr) {
            return -1;
        }
        const std::vector<llama_token> tokens = tokenize_prompt(vocab_, prompt);
        if (tokens.empty()) {
            return -1;
        }
        return static_cast<int32_t>(tokens.size());
    }

    std::future<request_result> submit(request_params req) override {
        auto item      = std::make_shared<work_item>();
        item->req      = std::move(req);
        item->t_submit = steady_clock::now();
        auto fut = item->promise.get_future();
        {
            std::lock_guard<std::mutex> lock(mu_);
            queue_.push(std::move(item));
        }
        cv_.notify_one();
        return fut;
    }

private:
    void shutdown() noexcept {
        {
            std::lock_guard<std::mutex> lock(mu_);
            running_ = false;
        }
        cv_.notify_all();
        for (auto & t : workers_) {
            if (t.joinable()) {
                t.join();
            }
        }
        workers_.clear();
        for (auto * ctx : contexts_) {
            if (ctx != nullptr) {
                llama_free(ctx);
            }
        }
        contexts_.clear();
        if (model_ != nullptr) {
            llama_model_free(model_);
            model_ = nullptr;
        }
    }

    void worker_loop(int32_t wid) {
        llama_context * ctx = contexts_[static_cast<size_t>(wid)];
        for (;;) {
            std::shared_ptr<work_item> item;
            {
                std::unique_lock<std::mutex> lock(mu_);
                cv_.wait(lock, [&] { return !running_ || !queue_.empty(); });
                if (queue_.empty()) {
                    return;
                }
                item = std::move(queue_.front());
                queue_.pop();
            }
            run_one(ctx, *item);
        }
    }

    void run_one(llama_context * ctx, work_item & item) {
        const auto & req = item.req;
        request_result res{};
        res.status               = request_status::error;
        res.n_tokens_generated   = 0;
        res.generated_token_hash = k_token_hash_empty;

        llama_memory_clear(llama_get_memory(ctx), false);

        if (req.max_tokens <= 0) {
            res.status = request_status::ok;
            res.ttft   = std::chrono::nanoseconds(0);
            res.total  = steady_clock::now() - item.t_submit;
            item.promise.set_value(std::move(res));
            return;
        }

        std::vector<llama_token> prompt_tokens = tokenize_prompt(vocab_, req.prompt);
        if (prompt_tokens.empty()) {
            res.error_message = "tokenize failed";
            res.total         = steady_clock::now() - item.t_submit;
            item.promise.set_value(std::move(res));
            return;
        }

        const int32_t n_prompt  = static_cast<int32_t>(prompt_tokens.size());
        const int32_t n_predict = req.max_tokens;

        llama_batch batch = llama_batch_get_one(prompt_tokens.data(), n_prompt);

        int32_t     n_pos       = 0;
        int32_t     n_generated = 0;
        bool        first       = true;
        llama_token next_id     = 0;
        uint64_t    hash_state  = k_token_hash_init;

        while (n_pos + batch.n_tokens < n_prompt + n_predict) {
            const int32_t rc = llama_decode(ctx, batch);
            if (rc != 0) {
                res.error_message = "llama_decode failed";
                res.total         = steady_clock::now() - item.t_submit;
                item.promise.set_value(std::move(res));
                return;
            }
            llama_synchronize(ctx);

            n_pos += batch.n_tokens;

            const float * logits = llama_get_logits_ith(ctx, -1);
            if (logits == nullptr) {
                res.error_message = "no logits";
                res.total         = steady_clock::now() - item.t_submit;
                item.promise.set_value(std::move(res));
                return;
            }

            next_id = argmax(logits, n_vocab_);

            if (first) {
                res.ttft = steady_clock::now() - item.t_submit;
                first    = false;
            }

            if (llama_vocab_is_eog(vocab_, next_id)) {
                break;
            }
            n_generated++;
            hash_state = fold_token_hash(hash_state, static_cast<int32_t>(next_id));

            batch = llama_batch_get_one(&next_id, 1);
        }

        res.status               = request_status::ok;
        res.n_tokens_generated   = n_generated;
        res.generated_token_hash = (n_generated == 0) ? k_token_hash_empty : hash_state;
        res.total                = steady_clock::now() - item.t_submit;
        item.promise.set_value(std::move(res));
    }

    llama_model *                          model_   = nullptr;
    const llama_vocab *                    vocab_   = nullptr;
    int32_t                                n_vocab_ = 0;
    std::vector<llama_context *>           contexts_;
    std::vector<std::thread>               workers_;
    std::mutex                             mu_;
    std::condition_variable                cv_;
    std::queue<std::shared_ptr<work_item>> queue_;
    bool                                   running_ = false;
};

} // namespace

std::unique_ptr<engine> make_engine_std() {
    return std::unique_ptr<engine>(new engine_std());
}

} // namespace serving_bench
