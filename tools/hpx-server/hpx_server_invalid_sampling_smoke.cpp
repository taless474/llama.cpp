// hpx_server_invalid_sampling_smoke.cpp — M7b negative-path
// sampling smoke for llama-hpx-server.
//
// One in-process server, six invalid-config POSTs followed by one
// valid greedy POST. Each invalid POST asserts:
//   - HTTP 422
//   - response body parses to JSON
//   - error.code == "invalid_sampling"
//   - error.message contains the expected substring (the
//     `validate_sampling_config` error wording lifted into types.h)
// The final POST asserts the server is still alive and serves
// canonical greedy under the M7a-shape (no "sampling") request,
// proving the rejection path didn't pollute engine state.
//
// HPX-nativity boundary: cpp-httplib remains the explicit non-HPX
// network adapter boundary. Our sources contain no std::thread /
// std::mutex / std::condition_variable / std::this_thread::sleep_for.
// Handler thread only parses JSON, validates sampling, tokenizes via
// const llama_vocab *, submits, and waits on hpx::future::get. Never
// touches llama.cpp mutable execution state.

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
#include <limits>
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
        "  M7b HPX server invalid-sampling rejection smoke.\n",
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
    fprintf(stdout, "HPX_SERVER_INVALID_SAMPLING_SMOKE: FAIL: %s\n",
            reason.c_str());
    fflush(stdout);
}

void emit_pass() {
    fprintf(stdout, "HPX_SERVER_INVALID_SAMPLING_SMOKE: PASS\n");
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

bool parse_sampling_mode(const std::string & s, sampling_mode & out) {
    if (s == "greedy")     { out = sampling_mode::greedy;     return true; }
    if (s == "stochastic") { out = sampling_mode::stochastic; return true; }
    return false;
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
    ctx_params.n_batch         = 64;
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

    constexpr const char * k_canonical_greedy_b8 =
        "0x0619d4d1900c2365";

    try {
        engine eng(std::move(opts));

        hpx::future<void> engine_fut =
            hpx::async([&] { eng.run(); });

        std::atomic<int32_t> next_rid{1};

        httplib::Server srv;
        // Handler identical in shape to hpx-server.cpp's. Inlined
        // here to keep the smoke self-contained (no shared TU).
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

            sampling_config cfg;
            if (body.contains("sampling")) {
                const auto & s = body["sampling"];
                if (!s.is_object()) {
                    send_error(res, 400, "bad_request",
                        "'sampling' must be an object");
                    return;
                }
                if (s.contains("mode")) {
                    if (!s["mode"].is_string()
                     || !parse_sampling_mode(
                            s["mode"].get<std::string>(), cfg.mode)) {
                        send_error(res, 422, "invalid_sampling",
                            "mode must be 'greedy' or 'stochastic'");
                        return;
                    }
                }
                if (s.contains("seed")) {
                    if (!s["seed"].is_number_unsigned()) {
                        send_error(res, 422, "invalid_sampling",
                            "seed must be a non-negative integer");
                        return;
                    }
                    const uint64_t v = s["seed"].get<uint64_t>();
                    if (v > std::numeric_limits<uint32_t>::max()) {
                        send_error(res, 422, "invalid_sampling",
                            "seed exceeds uint32_t range");
                        return;
                    }
                    cfg.seed = static_cast<uint32_t>(v);
                }
                if (s.contains("temperature")) {
                    if (!s["temperature"].is_number()) {
                        send_error(res, 422, "invalid_sampling",
                            "temperature must be a number");
                        return;
                    }
                    cfg.temperature = s["temperature"].get<float>();
                }
                if (s.contains("top_k")) {
                    if (!s["top_k"].is_number_integer()) {
                        send_error(res, 422, "invalid_sampling",
                            "top_k must be an integer");
                        return;
                    }
                    cfg.top_k = s["top_k"].get<int32_t>();
                }
                if (s.contains("top_p")) {
                    if (!s["top_p"].is_number()) {
                        send_error(res, 422, "invalid_sampling",
                            "top_p must be a number");
                        return;
                    }
                    cfg.top_p = s["top_p"].get<float>();
                }
                if (s.contains("top_p_min_keep")) {
                    if (!s["top_p_min_keep"].is_number_unsigned()) {
                        send_error(res, 422, "invalid_sampling",
                            "top_p_min_keep must be a non-negative "
                            "integer");
                        return;
                    }
                    const uint64_t v =
                        s["top_p_min_keep"].get<uint64_t>();
                    if (v > std::numeric_limits<uint32_t>::max()) {
                        send_error(res, 422, "invalid_sampling",
                            "top_p_min_keep exceeds uint32_t range");
                        return;
                    }
                    cfg.top_p_min_keep = static_cast<uint32_t>(v);
                }
                std::string verr;
                if (!validate_sampling_config(cfg, verr)) {
                    send_error(res, 422, "invalid_sampling", verr);
                    return;
                }
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
            sr.sampling      = std::move(cfg);

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

        const int port = srv.bind_to_any_port("127.0.0.1");
        if (port < 0) {
            eng.request_shutdown();
            engine_fut.get();
            return fail_and_cleanup("bind_to_any_port failed");
        }

        hpx::future<bool> listen_fut = hpx::async([&] {
            return srv.listen_after_bind();
        });
        srv.wait_until_ready();

        httplib::Client cli("127.0.0.1", port);
        cli.set_read_timeout(30, 0);
        cli.set_write_timeout(30, 0);

        auto stop_drain_fail = [&](const std::string & reason) -> int {
            srv.stop();
            listen_fut.get();
            eng.request_shutdown();
            engine_fut.get();
            return fail_and_cleanup(reason);
        };

        auto post_completion =
            [&](const nlohmann::json & body) -> httplib::Result {
            return cli.Post("/completion", body.dump(),
                            "application/json");
        };

        auto expect_422 =
            [&](const char *           label,
                const nlohmann::json & body,
                const char *           expected_msg_substr) -> std::string {
            auto resp = post_completion(body);
            if (!resp) {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                    "%s client.Post failed (error=%d)", label,
                    static_cast<int>(resp.error()));
                return std::string(buf);
            }
            if (resp->status != 422) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "%s HTTP status=%d (expected 422), body=%s",
                    label, resp->status, resp->body.c_str());
                return std::string(buf);
            }
            nlohmann::json out;
            try {
                out = nlohmann::json::parse(resp->body);
            } catch (const std::exception & e) {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                    "%s response body is not JSON: %s",
                    label, e.what());
                return std::string(buf);
            }
            if (!out.contains("error")
             || !out["error"].is_object()
             || !out["error"].contains("code")
             || !out["error"]["code"].is_string()
             || out["error"]["code"].get<std::string>()
                  != "invalid_sampling") {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "%s error.code != 'invalid_sampling' (body=%s)",
                    label, resp->body.c_str());
                return std::string(buf);
            }
            if (!out["error"].contains("message")
             || !out["error"]["message"].is_string()) {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                    "%s error.message missing/non-string", label);
                return std::string(buf);
            }
            const std::string msg =
                out["error"]["message"].get<std::string>();
            if (expected_msg_substr != nullptr
             && msg.find(expected_msg_substr) == std::string::npos) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "%s error.message='%s' missing substr '%s'",
                    label, msg.c_str(), expected_msg_substr);
                return std::string(buf);
            }
            return std::string();  // empty == pass
        };

        // ---- 6 rejection cases --------------------------------------
        struct case_def {
            const char *   label;
            nlohmann::json sampling;
            const char *   expected_substr;
        };
        const std::vector<case_def> cases = {
            { "case1_temp0",     {{"mode","stochastic"},{"temperature",0}},      "temperature" },
            { "case2_temp_neg",  {{"mode","stochastic"},{"temperature",-1.0}},   "temperature" },
            { "case3_topk_neg",  {{"mode","stochastic"},{"top_k",-1}},           "top_k"       },
            { "case4_topp0",     {{"mode","stochastic"},{"top_p",0}},            "top_p"       },
            { "case5_topp_high", {{"mode","stochastic"},{"top_p",1.5}},          "top_p"       },
            { "case6_mk0",       {{"mode","stochastic"},{"top_p_min_keep",0}},   "top_p_min_keep" },
        };

        for (const auto & c : cases) {
            nlohmann::json req;
            req["prompt"]        = "Hello, my name is";
            req["decode_budget"] = 8;
            req["sampling"]      = c.sampling;
            const std::string err =
                expect_422(c.label, req, c.expected_substr);
            if (!err.empty()) {
                return stop_drain_fail(err);
            }
        }

        // ---- Sanity case: greedy/default still succeeds --------------
        nlohmann::json req_ok;
        req_ok["prompt"]        = "Hello, my name is";
        req_ok["decode_budget"] = 8;
        auto resp_ok = post_completion(req_ok);
        if (!resp_ok) {
            return stop_drain_fail("sanity client.Post failed");
        }
        if (resp_ok->status != 200) {
            char buf[200];
            std::snprintf(buf, sizeof(buf),
                "sanity HTTP status=%d (expected 200), body=%s",
                resp_ok->status, resp_ok->body.c_str());
            return stop_drain_fail(buf);
        }
        nlohmann::json out_ok;
        try {
            out_ok = nlohmann::json::parse(resp_ok->body);
        } catch (const std::exception & e) {
            return stop_drain_fail(
                std::string("sanity body is not JSON: ") + e.what());
        }
        if (!out_ok.contains("hash")
         || out_ok["hash"].get<std::string>()
              != std::string(k_canonical_greedy_b8)) {
            return stop_drain_fail(
                "sanity hash != greedy canonical");
        }
        if (!out_ok.contains("n_decoded")
         || out_ok["n_decoded"].get<int>() != 8) {
            return stop_drain_fail("sanity n_decoded != 8");
        }

        srv.stop();
        listen_fut.get();

        for (const auto & c : cases) {
            fprintf(stdout,
                "%s status=422 invalid_sampling\n", c.label);
        }
        fprintf(stdout,
            "sanity hash=%s n_decoded=%d\n",
            out_ok["hash"].get<std::string>().c_str(),
            out_ok["n_decoded"].get<int>());
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
