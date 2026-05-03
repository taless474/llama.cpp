#include "backend_hpx.h"

#include "harness.h"

#include "llama.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <exception>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef LLAMA_SERVING_BENCH_HPX
#include <hpx/hpx.hpp>
#endif

namespace serving_bench {

#ifdef LLAMA_SERVING_BENCH_HPX

namespace {

using steady_clock = std::chrono::steady_clock;

bool trace_enabled() noexcept {
    const char * v = std::getenv("LLAMA_SERVING_BENCH_HPX_TRACE");
    return v != nullptr && v[0] == '1' && v[1] == '\0';
}

// Mirrors backend_std.cpp's argmax byte-for-byte. Greedy decode equality
// (Gate 4) requires identical sampling, including the tie-breaking shape
// (>, not >=, so the first-seen max wins).
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

// Mirrors backend_std.cpp's tokenize_prompt byte-for-byte: same vocab, same
// (add_special=true, parse_special=true), same retry-on-overflow shape. Gate
// 4 (std-vs-HPX hash equality) requires this to stay identical to the std
// helper.
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

// Future-based context pool with FIFO waiters.
//
// acquire() returns a ready future when a context is available, otherwise it
// enqueues a fresh hpx::promise on waiters_ and returns the corresponding
// future. After close() it returns an exceptional future immediately.
//
// release(ctx) wakes exactly one queued waiter (FIFO, popped from the front
// of waiters_) if any is queued, otherwise pushes ctx back to available_.
// Post-close, waiters_ is permanently empty, so release(ctx) just pushes
// to available_ — matching the design-doc contract that release on a closed
// pool is legal and never wakes anyone.
//
// Concurrency invariant: set_value / set_exception NEVER run under mu_.
// Both release() and close() pop the relevant promises out of waiters_
// while holding the lock and only resolve them after dropping it, so no
// .then continuation runs under our mutex.
class hpx_context_pool {
public:
    explicit hpx_context_pool() noexcept = default;

    void install(std::vector<llama_context *> contexts) {
        std::lock_guard<std::mutex> lock(mu_);
        available_ = std::move(contexts);
        size_      = static_cast<int32_t>(available_.size());
    }

    int32_t size() const noexcept { return size_; }

    hpx::future<llama_context *> acquire() {
        std::lock_guard<std::mutex> lock(mu_);
        if (closed_) {
            return make_closed_future();
        }
        if (!available_.empty()) {
            llama_context * ctx = available_.back();
            available_.pop_back();
            return hpx::make_ready_future<llama_context *>(ctx);
        }
        hpx::promise<llama_context *> p;
        hpx::future<llama_context *>  f = p.get_future();
        waiters_.emplace_back(std::move(p));
        return f;
    }

    void release(llama_context * ctx) noexcept {
        hpx::promise<llama_context *> next_waiter;
        bool                          have_waiter = false;
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (!waiters_.empty()) {
                next_waiter = std::move(waiters_.front());
                waiters_.pop_front();
                have_waiter = true;
            } else {
                available_.push_back(ctx);
            }
        }
        if (have_waiter) {
            try {
                next_waiter.set_value(ctx);
            } catch (...) {
                // Logic bug (promise already satisfied) or transport error.
                // Don't drop the context — push it back to available_ so a
                // future acquire can still find it.
                std::lock_guard<std::mutex> lock(mu_);
                available_.push_back(ctx);
            }
        }
    }

    void close() noexcept {
        std::deque<hpx::promise<llama_context *>> drain;
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (closed_) {
                return;
            }
            closed_ = true;
            drain   = std::move(waiters_);
            waiters_.clear();
        }
        if (!drain.empty()) {
            const std::exception_ptr eptr =
                std::make_exception_ptr(std::runtime_error("hpx_context_pool: closed"));
            for (auto & w : drain) {
                try {
                    w.set_exception(eptr);
                } catch (...) {
                    // Waiter already satisfied or torn down. Nothing useful
                    // to do here during shutdown.
                }
            }
        }
    }

private:
    static hpx::future<llama_context *> make_closed_future() {
        hpx::promise<llama_context *> p;
        hpx::future<llama_context *>  f = p.get_future();
        p.set_exception(
            std::make_exception_ptr(std::runtime_error("hpx_context_pool: closed")));
        return f;
    }

    mutable std::mutex                        mu_;
    std::vector<llama_context *>              available_;
    std::deque<hpx::promise<llama_context *>> waiters_;
    int32_t                                   size_   = 0;
    bool                                      closed_ = false;
};

// Move-only RAII guard. The destructor releases the held context on every
// path (success and error) and, when LLAMA_SERVING_BENCH_HPX_TRACE=1, prints
// a matching `release` line so trace acquire/release counts always agree.
//
// A default-constructed guard owns nothing and emits no trace; same for any
// guard that has been moved-from. Trace + release fire only when the guard
// holds a non-null pool/ctx pair.
class context_guard {
public:
    context_guard() noexcept = default;

    explicit context_guard(hpx_context_pool * pool, llama_context * ctx,
                           int32_t req_idx, int32_t ctx_id) noexcept
        : pool_(pool), ctx_(ctx), req_idx_(req_idx), ctx_id_(ctx_id) {}

    ~context_guard() noexcept {
        release_internal();
    }

    context_guard(const context_guard &)             = delete;
    context_guard & operator=(const context_guard &) = delete;

    context_guard(context_guard && other) noexcept
        : pool_(other.pool_),
          ctx_(other.ctx_),
          req_idx_(other.req_idx_),
          ctx_id_(other.ctx_id_) {
        other.pool_ = nullptr;
        other.ctx_  = nullptr;
    }

    context_guard & operator=(context_guard && other) noexcept {
        if (this != &other) {
            release_internal();
            pool_       = other.pool_;
            ctx_        = other.ctx_;
            req_idx_    = other.req_idx_;
            ctx_id_     = other.ctx_id_;
            other.pool_ = nullptr;
            other.ctx_  = nullptr;
        }
        return *this;
    }

    llama_context * get() const noexcept { return ctx_; }

private:
    void release_internal() noexcept {
        if (pool_ != nullptr && ctx_ != nullptr) {
            if (trace_enabled()) {
                std::fprintf(stderr,
                             "[serving-bench] req[%d] release ctx=%d\n",
                             req_idx_, ctx_id_);
            }
            pool_->release(ctx_);
        }
        pool_ = nullptr;
        ctx_  = nullptr;
    }

    hpx_context_pool * pool_    = nullptr;
    llama_context *    ctx_     = nullptr;
    int32_t            req_idx_ = -1;
    int32_t            ctx_id_  = -1;
};

// Per the HPX-ON acceptance plan: every terminal path (`finish_empty_ok`,
// `finish_error`, future `finish_ok`) resolves the std::promise via a
// single CAS on `finished`. With HPX continuations potentially running on
// different carriers, a load-then-store check would let two finishers both
// observe `false` and double-set the promise — undefined behavior on
// std::promise.
struct request_state {
    request_params               req;
    steady_clock::time_point     t_submit;
    std::promise<request_result> promise;
    std::atomic<bool>            finished{false};
};

class engine_hpx final : public engine {
public:
    explicit engine_hpx() noexcept = default;

    ~engine_hpx() noexcept override {
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
            const uint32_t hw   = std::thread::hardware_concurrency();
            const int32_t  hw_i = hw > 0 ? static_cast<int32_t>(hw) : 1;
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

        contexts_.assign(static_cast<size_t>(n_workers), nullptr);
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
            contexts_[static_cast<size_t>(i)] = ctx;
            ctx_id_[ctx]                      = i;
        }

        // Pool stores the same pointers (non-owning view). engine_hpx retains
        // ownership; pool's vector is destroyed when the engine destructs,
        // after shutdown() has already llama_free'd them.
        pool_.install(contexts_);

        if (trace_enabled()) {
            std::fprintf(stderr,
                         "[serving-bench] engine_hpx ready: n_contexts=%d pool_size=%d\n",
                         n_workers, pool_.size());
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
        auto state      = std::make_shared<request_state>();
        state->req      = std::move(req);
        state->t_submit = steady_clock::now();
        auto fut        = state->promise.get_future();

        if (state->req.max_tokens <= 0) {
            // Empty-generation path: bypass the pool by design (Gate 5).
            // The chain shape stays uniform — a ready null-context future
            // and a .then that resolves the promise as ok / empty.
            hpx::future<llama_context *> ctx_fut =
                hpx::make_ready_future<llama_context *>(nullptr);
            hpx::future<void> chain = ctx_fut.then(
                [state, this](hpx::future<llama_context *> /*f*/) {
                    this->finish_empty_ok(*state);
                });
            push_in_flight(std::move(chain));
            return fut;
        }

        // Non-empty path: real pool acquire. The continuation pulls ctx
        // from f (handling the closed-pool exception case), traces the
        // acquire, constructs a guard so the release path is RAII, then
        // routes to finish_error("not implemented yet") because real
        // decode lands in the next slice.
        hpx::future<llama_context *> ctx_fut = pool_.acquire();
        hpx::future<void> chain = ctx_fut.then(
            [state, this](hpx::future<llama_context *> f) {
                llama_context * ctx = nullptr;
                try {
                    ctx = f.get();
                } catch (const std::exception & e) {
                    this->finish_error(*state,
                                       std::string("hpx pool acquire failed: ") + e.what());
                    return;
                } catch (...) {
                    this->finish_error(*state, "hpx pool acquire failed: unknown");
                    return;
                }
                if (ctx == nullptr) {
                    this->finish_error(*state, "hpx pool acquire returned nullptr");
                    return;
                }
                const int32_t req_idx = state->req.request_index;
                const int32_t ctx_idx = this->ctx_id_at(ctx);
                if (trace_enabled()) {
                    std::fprintf(stderr,
                                 "[serving-bench] req[%d] acquire ctx=%d\n",
                                 req_idx, ctx_idx);
                }
                // Guard releases (and traces release) on every exit, so
                // trace acquire and release counts always agree even if
                // an unexpected throw escapes the decode body.
                context_guard guard(&this->pool_, ctx, req_idx, ctx_idx);
                // Wrap the decode body so any exception still resolves the
                // std::promise (via finish_error) and lets the guard
                // release the ctx. The CAS in finish_ok / finish_error
                // ensures the catch-side finish_error is a no-op when
                // run_decode already resolved on the success path.
                try {
                    this->run_decode(*state, ctx);
                } catch (const std::exception & e) {
                    this->finish_error(*state,
                                       std::string("hpx decode threw: ") + e.what());
                } catch (...) {
                    this->finish_error(*state, "hpx decode threw unknown");
                }
            });
        push_in_flight(std::move(chain));
        return fut;
    }

private:
    void shutdown() noexcept {
        // Per the design doc: close the pool BEFORE waiting on in-flight
        // chains. Any waiter still queued at close time gets resolved with
        // an exception, so its .then continuation routes through
        // finish_error rather than hanging forever.
        pool_.close();

        std::vector<hpx::future<void>> drain;
        {
            std::lock_guard<std::mutex> lock(in_flight_mu_);
            drain = std::move(in_flight_);
        }
        if (!drain.empty()) {
            try {
                hpx::wait_all(drain);
            } catch (const std::exception & e) {
                std::fprintf(stderr,
                             "[serving-bench] engine_hpx wait_all threw: %s\n",
                             e.what());
            } catch (...) {
                std::fprintf(stderr,
                             "[serving-bench] engine_hpx wait_all threw unknown\n");
            }
        }

        for (auto * ctx : contexts_) {
            if (ctx != nullptr) {
                llama_free(ctx);
            }
        }
        contexts_.clear();
        ctx_id_.clear();
        if (model_ != nullptr) {
            llama_model_free(model_);
            model_ = nullptr;
        }
    }

    void push_in_flight(hpx::future<void> f) {
        std::lock_guard<std::mutex> lock(in_flight_mu_);
        in_flight_.emplace_back(std::move(f));
    }

    // Mirrors backend_std.cpp::run_one's decode body byte-for-byte, with
    // the std-side res-mutation rewritten as calls to finish_ok /
    // finish_error so the CAS-resolved std::promise pattern is preserved.
    // The leased ctx comes from the pool; the guard in submit() releases
    // it after this returns or throws.
    void run_decode(request_state & s, llama_context * ctx) {
        const auto & req = s.req;

        llama_memory_clear(llama_get_memory(ctx), false);

        std::vector<llama_token> prompt_tokens = tokenize_prompt(vocab_, req.prompt);
        if (prompt_tokens.empty()) {
            finish_error(s, "tokenize failed");
            return;
        }

        const int32_t n_prompt  = static_cast<int32_t>(prompt_tokens.size());
        const int32_t n_predict = req.max_tokens;

        llama_batch batch = llama_batch_get_one(prompt_tokens.data(), n_prompt);

        int32_t                  n_pos       = 0;
        int32_t                  n_generated = 0;
        bool                     first       = true;
        llama_token              next_id     = 0;
        uint64_t                 hash_state  = k_token_hash_init;
        std::chrono::nanoseconds ttft        = std::chrono::nanoseconds(0);

        while (n_pos + batch.n_tokens < n_prompt + n_predict) {
            const int32_t rc = llama_decode(ctx, batch);
            if (rc != 0) {
                finish_error(s, "llama_decode failed");
                return;
            }
            llama_synchronize(ctx);

            n_pos += batch.n_tokens;

            const float * logits = llama_get_logits_ith(ctx, -1);
            if (logits == nullptr) {
                finish_error(s, "no logits");
                return;
            }

            next_id = argmax(logits, n_vocab_);

            if (first) {
                ttft  = steady_clock::now() - s.t_submit;
                first = false;
            }

            if (llama_vocab_is_eog(vocab_, next_id)) {
                break;
            }
            n_generated++;
            hash_state = fold_token_hash(hash_state, static_cast<int32_t>(next_id));

            batch = llama_batch_get_one(&next_id, 1);
        }

        const uint64_t hash =
            (n_generated == 0) ? k_token_hash_empty : hash_state;
        finish_ok(s, ttft, n_generated, hash);
    }

    void finish_empty_ok(request_state & s) {
        bool expected = false;
        if (!s.finished.compare_exchange_strong(expected, true)) {
            return;
        }
        request_result r{};
        r.status               = request_status::ok;
        r.n_tokens_generated   = 0;
        r.generated_token_hash = k_token_hash_empty;
        r.ttft                 = std::chrono::nanoseconds(0);
        r.total                = steady_clock::now() - s.t_submit;
        s.promise.set_value(std::move(r));
    }

    void finish_ok(request_state & s, std::chrono::nanoseconds ttft,
                   int32_t n_generated, uint64_t hash) {
        bool expected = false;
        if (!s.finished.compare_exchange_strong(expected, true)) {
            return;
        }
        request_result r{};
        r.status               = request_status::ok;
        r.n_tokens_generated   = n_generated;
        r.generated_token_hash = hash;
        r.ttft                 = ttft;
        r.total                = steady_clock::now() - s.t_submit;
        s.promise.set_value(std::move(r));
    }

    void finish_error(request_state & s, std::string msg) {
        bool expected = false;
        if (!s.finished.compare_exchange_strong(expected, true)) {
            return;
        }
        request_result r{};
        r.status               = request_status::error;
        r.n_tokens_generated   = 0;
        r.generated_token_hash = k_token_hash_empty;
        r.error_message        = std::move(msg);
        r.total                = steady_clock::now() - s.t_submit;
        s.promise.set_value(std::move(r));
    }

    int32_t ctx_id_at(llama_context * ctx) const {
        const auto it = ctx_id_.find(ctx);
        return it != ctx_id_.end() ? it->second : -1;
    }

    llama_model *                                model_   = nullptr;
    const llama_vocab *                          vocab_   = nullptr;
    int32_t                                      n_vocab_ = 0;
    std::vector<llama_context *>                 contexts_;
    std::unordered_map<llama_context *, int32_t> ctx_id_;
    hpx_context_pool                             pool_;
    std::mutex                                   in_flight_mu_;
    std::vector<hpx::future<void>>               in_flight_;
};

} // namespace

std::unique_ptr<engine> make_engine_hpx() {
    return std::unique_ptr<engine>(new engine_hpx());
}

#else

std::unique_ptr<engine> make_engine_hpx() {
    // Defensive fallback. Under the post-Slice-2 lifecycle, main.cpp short-
    // circuits on hpx_runtime_start_once() returning false, so this branch
    // is never reached when LLAMA_SERVING_BENCH_HPX is OFF.
    std::fprintf(stderr,
                 "[serving-bench] HPX backend not built "
                 "(LLAMA_SERVING_BENCH_HPX=OFF)\n");
    return nullptr;
}

#endif

} // namespace serving_bench
