// hpx_server_smoke.cpp — M7a in-process round-trip smoke for
// llama-hpx-server.
//
// Spins up the same server stack as `hpx-server.cpp` inside this
// process, runs `srv.listen_after_bind()` on an hpx::async task, and
// drives a single POST /completion via an in-process
// `httplib::Client`. Asserts HTTP 200 + canonical greedy hash + non-
// empty detokenized text, then stops the server through normal
// control flow (srv.stop()) and drains the engine.
//
// HPX-nativity boundary (mirrors hpx-server.cpp):
//   cpp-httplib is an explicit non-HPX network adapter boundary. Our
//   sources contain no std::thread / std::mutex /
//   std::condition_variable / std::this_thread::sleep_for. The
//   handler thread only parses JSON, calls common_{tokenize,
//   detokenize}(const llama_vocab *, ...), pushes onto the engine
//   inbox via submit_request, and blocks on hpx::future::get. The
//   handler never holds llama_context * and never reaches
//   llama_decode / llama_get_logits_ith / llama_sampler_* /
//   llama_memory_seq_* / KV APIs.

#include "common.h"
#include "llama.h"

#include "engine.h"
#include "hpx_runtime.h"
#include "trace.h"
#include "types.h"

#include <hpx/hpx.hpp>

#include <cpp-httplib/httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>
#include <utility>
#include <vector>

namespace {

struct smoke_args {
    std::string model_path;
};

void print_usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s --model <path>\n"
        "  M7a HPX server round-trip smoke.\n",
        argv0);
}

bool parse_args(int argc, char ** argv, smoke_args & out) {
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--model") {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: --model requires a value\n");
                return false;
            }
            out.model_path = argv[++i];
        } else if (a == "-h" || a == "--help") {
            print_usage(argv[0]);
            return false;
        } else {
            fprintf(stderr, "error: unknown arg '%s'\n", a.c_str());
            return false;
        }
    }
    if (out.model_path.empty()) {
        fprintf(stderr, "error: --model is required\n");
        return false;
    }
    return true;
}

void emit_fail(const std::string & reason) {
    fprintf(stdout, "HPX_SERVER_SMOKE: FAIL: %s\n", reason.c_str());
    fflush(stdout);
}

void emit_pass() {
    fprintf(stdout, "HPX_SERVER_SMOKE: PASS\n");
    fflush(stdout);
}

void send_error(httplib::Response & res, int status,
                const std::string & code,
                const std::string & message) {
    nlohmann::json body;
    body["error"]["code"]    = code;
    body["error"]["message"] = message;
    res.status = status;
    res.set_content(body.dump(), "application/json");
}

std::string hex_hash(uint64_t h) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "0x%016llx",
                  static_cast<unsigned long long>(h));
    return std::string(buf);
}

}  // namespace

int main(int argc, char ** argv) {
    smoke_args args;
    if (!parse_args(argc, argv, args)) return 2;

    trace::init();

    if (!hpx_runtime::start_once(/*os_threads=*/2)) {
        emit_fail("hpx runtime start failed");
        return 1;
    }

    common_init();
    llama_backend_init();

    llama_model *   model = nullptr;
    llama_context * ctx   = nullptr;

    auto cleanup_and_stop = [&](int rc) -> int {
        if (ctx)   llama_free(ctx);
        if (model) llama_model_free(model);
        llama_backend_free();
        hpx_runtime::stop();
        return rc;
    };

    auto fail_and_cleanup = [&](const std::string & reason) -> int {
        emit_fail(reason);
        return cleanup_and_stop(1);
    };

    llama_model_params model_params = llama_model_default_params();
    model = llama_model_load_from_file(args.model_path.c_str(),
                                       model_params);
    if (model == nullptr) {
        return fail_and_cleanup("model load failed");
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx           = 2048;
    ctx_params.n_batch         = 1024;
    ctx_params.n_seq_max       = 2;
    ctx_params.n_threads       = 2;
    ctx_params.n_threads_batch = 2;
    ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        return fail_and_cleanup("context create failed");
    }

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);

    constexpr int32_t k_n_seq_max         = 2;
    constexpr int32_t k_max_prompt_tokens = 512;
    constexpr int32_t k_batch_capacity    =
        k_n_seq_max * k_max_prompt_tokens;

    std::vector<waiting_request> empty_waiting;

    engine_options opts;
    opts.lib.ctx                  = ctx;
    opts.lib.vocab                = vocab;
    opts.lib.n_vocab              = n_vocab;
    opts.lib.batch_capacity       = k_batch_capacity;
    opts.lib.n_seq_max            = k_n_seq_max;
    opts.lib.initial_idle_slots   = k_n_seq_max;
    opts.lib.keep_alive           = true;
    opts.preload.prompt_tokens    = nullptr;
    opts.preload.budgets          = {};
    opts.preload.waiting_queue    = &empty_waiting;
    opts.preload.reuse_completed  = false;
    opts.preload.stream_all       = false;
    opts.gate_test.cancel_after     = -1;
    opts.gate_test.max_decode_iters = 0;

    try {
        engine eng(std::move(opts));

        hpx::future<void> engine_fut =
            hpx::async([&] { eng.run(); });

        std::atomic<int32_t> next_rid{1};

        httplib::Server srv;
        srv.Post("/completion",
                 [&](const httplib::Request & req,
                     httplib::Response &       res) {
            nlohmann::json body;
            try {
                body = nlohmann::json::parse(req.body);
            } catch (const std::exception &) {
                send_error(res, 400, "bad_request",
                           "request body must be valid JSON");
                return;
            }
            if (!body.is_object()
             || !body.contains("prompt")
             || !body["prompt"].is_string()
             || !body.contains("decode_budget")
             || !body["decode_budget"].is_number_integer()) {
                send_error(res, 400, "bad_request",
                           "missing required fields");
                return;
            }
            const std::string prompt        = body["prompt"];
            const int32_t     decode_budget =
                body["decode_budget"].get<int32_t>();
            if (decode_budget <= 0) {
                send_error(res, 422, "invalid_argument",
                           "'decode_budget' must be > 0");
                return;
            }

            std::vector<llama_token> tokens = common_tokenize(
                vocab, prompt, /*add_special=*/true,
                /*parse_special=*/true);
            if (tokens.empty()) {
                send_error(res, 422, "invalid_argument",
                           "tokenization produced zero tokens");
                return;
            }
            if (static_cast<int32_t>(tokens.size())
                  > k_max_prompt_tokens) {
                send_error(res, 413, "payload_too_large",
                           "prompt exceeds max-prompt-tokens");
                return;
            }

            const int32_t rid =
                next_rid.fetch_add(1, std::memory_order_relaxed);

            submit_request sr;
            sr.request_id    = rid;
            sr.prompt_tokens = std::move(tokens);
            sr.decode_budget = decode_budget;
            sr.want_stream   = false;

            submit_handle h = eng.submit_request(std::move(sr));

            request_result r;
            try {
                r = h.result.get();
            } catch (const std::exception & e) {
                send_error(res, 500, "engine_error", e.what());
                return;
            }

            const std::string text =
                common_detokenize(vocab, r.generated_tokens,
                                  /*special=*/false);

            nlohmann::json out;
            out["request_id"] = r.request_id;
            out["status"]     = status_name(r.status);
            out["n_decoded"]  = r.n_decoded;
            out["hash"]       = hex_hash(r.hash);
            out["text"]       = text;
            res.status = 200;
            res.set_content(out.dump(), "application/json");
        });

        // Bind first so we can capture an OS-assigned ephemeral port
        // (avoids smoke flakiness from a hard-coded port collision).
        const int port = srv.bind_to_any_port("127.0.0.1");
        if (port < 0) {
            eng.request_shutdown();
            engine_fut.get();
            return fail_and_cleanup("bind_to_any_port failed");
        }

        hpx::future<bool> listen_fut = hpx::async([&] {
            return srv.listen_after_bind();
        });

        // Blocks the main thread on cpp-httplib's own readiness
        // signal. No std::this_thread::sleep_for needed.
        srv.wait_until_ready();

        httplib::Client cli("127.0.0.1", port);
        cli.set_read_timeout(30, 0);
        cli.set_write_timeout(30, 0);

        nlohmann::json req_body;
        req_body["prompt"]        = "Hello, my name is";
        req_body["decode_budget"] = 8;

        auto resp = cli.Post("/completion", req_body.dump(),
                             "application/json");

        // Stop the server BEFORE running assertions so any failure
        // path still drains cleanly.
        srv.stop();
        listen_fut.get();

        if (!resp) {
            char buf[200];
            std::snprintf(buf, sizeof(buf),
                "client.Post failed (error=%d)",
                static_cast<int>(resp.error()));
            eng.request_shutdown();
            engine_fut.get();
            return fail_and_cleanup(buf);
        }
        if (resp->status != 200) {
            char buf[200];
            std::snprintf(buf, sizeof(buf),
                "HTTP status=%d (expected 200), body=%s",
                resp->status, resp->body.c_str());
            eng.request_shutdown();
            engine_fut.get();
            return fail_and_cleanup(buf);
        }

        nlohmann::json out;
        try {
            out = nlohmann::json::parse(resp->body);
        } catch (const std::exception & e) {
            eng.request_shutdown();
            engine_fut.get();
            return fail_and_cleanup(
                std::string("response body is not JSON: ") + e.what());
        }

        if (!out.contains("n_decoded")
         || !out["n_decoded"].is_number_integer()
         || out["n_decoded"].get<int>() != 8) {
            eng.request_shutdown();
            engine_fut.get();
            return fail_and_cleanup("n_decoded != 8");
        }
        constexpr const char * k_canonical_greedy_b8 =
            "0x0619d4d1900c2365";
        if (!out.contains("hash") || !out["hash"].is_string()
         || out["hash"].get<std::string>() != k_canonical_greedy_b8) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "hash=%s != canonical %s",
                out.contains("hash") && out["hash"].is_string()
                    ? out["hash"].get<std::string>().c_str()
                    : "<missing>",
                k_canonical_greedy_b8);
            eng.request_shutdown();
            engine_fut.get();
            return fail_and_cleanup(buf);
        }
        if (!out.contains("text") || !out["text"].is_string()
         || out["text"].get<std::string>().empty()) {
            eng.request_shutdown();
            engine_fut.get();
            return fail_and_cleanup("text missing or empty");
        }
        if (!out.contains("status") || !out["status"].is_string()
         || out["status"].get<std::string>() != "completed") {
            eng.request_shutdown();
            engine_fut.get();
            return fail_and_cleanup("status != completed");
        }

        fprintf(stdout,
            "request_id=%d status=%s n_decoded=%d hash=%s text=%s\n",
            out["request_id"].get<int>(),
            out["status"].get<std::string>().c_str(),
            out["n_decoded"].get<int>(),
            out["hash"].get<std::string>().c_str(),
            out["text"].get<std::string>().c_str());
        emit_pass();

        eng.request_shutdown();
        engine_fut.get();
    } catch (const std::exception & e) {
        return fail_and_cleanup(
            std::string("exception: ") + e.what());
    } catch (...) {
        return fail_and_cleanup("unknown exception");
    }

    return cleanup_and_stop(0);
}
