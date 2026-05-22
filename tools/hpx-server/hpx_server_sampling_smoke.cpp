// hpx_server_sampling_smoke.cpp — M7b positive-path sampling smoke
// for llama-hpx-server.
//
// One in-process server, four sequential POSTs against /completion:
//
//   Case A — omit "sampling"
//     Expect HTTP 200, hash == 0x0619d4d1900c2365 (greedy canonical).
//     Proves the M7a-shape request continues to take the greedy
//     argmax path bit-identically after the M7b JSON additions.
//
//   Case B — {"mode":"stochastic","seed":42}, default knobs
//     Expect HTTP 200, hash == 0xa8e14acb4094aa3f (M6b anchor).
//     Proves default-knob stochastic over HTTP collapses to the same
//     dist(seed)-only chain the M6b sampling smokes pinned.
//
//   Case C — {"mode":"stochastic","seed":43}, default knobs
//     Expect HTTP 200, hash != greedy canonical AND != seed=42
//     anchor. Proves seed sensitivity over HTTP. No literal pin on
//     Case C's hash.
//
//   Case D — full non-default knobs (seed=42, t=0.7, k=40, p=0.95,
//            min_keep=1)
//     Expect HTTP 200, status completed, hash != greedy canonical
//     AND != seed=42 anchor. No literal pin on Case D's hash — same
//     discipline as the M6c carry smoke's Case B (non-default knobs
//     may EOG-terminate early; we accept any non-zero n_decoded).
//
// HPX-nativity boundary: cpp-httplib remains the explicit non-HPX
// network adapter boundary. Our sources contain no std::thread /
// std::mutex / std::condition_variable / std::this_thread::sleep_for.
// Handler thread only parses JSON, calls common_{tokenize,
// detokenize}(const llama_vocab *, ...), validates sampling, pushes
// onto the engine inbox via submit_request, and waits on
// hpx::future::get. Never holds llama_context * and never reaches
// llama_decode / llama_get_logits_ith / llama_sampler_* /
// llama_memory_seq_* / KV APIs.

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
        "  M7b HPX server positive-path sampling smoke.\n",
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
    fprintf(stdout, "HPX_SERVER_SAMPLING_SMOKE: FAIL: %s\n",
            reason.c_str());
    fflush(stdout);
}

void emit_pass() {
    fprintf(stdout, "HPX_SERVER_SAMPLING_SMOKE: PASS\n");
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

    constexpr const char * k_canonical_greedy_b8 =
        "0x0619d4d1900c2365";
    constexpr const char * k_default_stochastic_seed42_b8 =
        "0xa8e14acb4094aa3f";

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

        auto expect_200_with_hash =
            [&](httplib::Result & resp, const char * label,
                const char * expected_hash_or_nullptr,
                nlohmann::json & out_json) -> std::string {
            if (!resp) {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                    "%s client.Post failed (error=%d)", label,
                    static_cast<int>(resp.error()));
                return std::string(buf);
            }
            if (resp->status != 200) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "%s HTTP status=%d (expected 200), body=%s",
                    label, resp->status, resp->body.c_str());
                return std::string(buf);
            }
            try {
                out_json = nlohmann::json::parse(resp->body);
            } catch (const std::exception & e) {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                    "%s body is not JSON: %s", label, e.what());
                return std::string(buf);
            }
            if (!out_json.contains("hash")
             || !out_json["hash"].is_string()) {
                char buf[120];
                std::snprintf(buf, sizeof(buf),
                    "%s response missing 'hash'", label);
                return std::string(buf);
            }
            if (!out_json.contains("status")
             || out_json["status"].get<std::string>() != "completed") {
                char buf[120];
                std::snprintf(buf, sizeof(buf),
                    "%s status != completed", label);
                return std::string(buf);
            }
            if (!out_json.contains("n_decoded")
             || !out_json["n_decoded"].is_number_integer()
             || out_json["n_decoded"].get<int>() < 1) {
                char buf[120];
                std::snprintf(buf, sizeof(buf),
                    "%s n_decoded < 1 or missing", label);
                return std::string(buf);
            }
            if (expected_hash_or_nullptr != nullptr
             && out_json["hash"].get<std::string>()
                  != std::string(expected_hash_or_nullptr)) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "%s hash=%s != expected %s", label,
                    out_json["hash"].get<std::string>().c_str(),
                    expected_hash_or_nullptr);
                return std::string(buf);
            }
            return std::string();  // empty == success
        };

        // ---- Case A: omit "sampling" (greedy canonical) -------------
        nlohmann::json req_a;
        req_a["prompt"]        = "Hello, my name is";
        req_a["decode_budget"] = 8;
        auto resp_a = post_completion(req_a);
        nlohmann::json out_a;
        std::string    err_a = expect_200_with_hash(
            resp_a, "CaseA", k_canonical_greedy_b8, out_a);
        if (!err_a.empty()) return stop_drain_fail(err_a);
        if (out_a["n_decoded"].get<int>() != 8) {
            return stop_drain_fail("CaseA n_decoded != 8");
        }

        // ---- Case B: stochastic seed=42 default knobs --------------
        nlohmann::json req_b;
        req_b["prompt"]        = "Hello, my name is";
        req_b["decode_budget"] = 8;
        req_b["sampling"] = {
            {"mode", "stochastic"},
            {"seed", 42}
        };
        auto resp_b = post_completion(req_b);
        nlohmann::json out_b;
        std::string    err_b = expect_200_with_hash(
            resp_b, "CaseB", k_default_stochastic_seed42_b8, out_b);
        if (!err_b.empty()) return stop_drain_fail(err_b);
        if (out_b["n_decoded"].get<int>() != 8) {
            return stop_drain_fail("CaseB n_decoded != 8");
        }

        // ---- Case C: stochastic seed=43 (seed sensitivity) ---------
        nlohmann::json req_c;
        req_c["prompt"]        = "Hello, my name is";
        req_c["decode_budget"] = 8;
        req_c["sampling"] = {
            {"mode", "stochastic"},
            {"seed", 43}
        };
        auto resp_c = post_completion(req_c);
        nlohmann::json out_c;
        std::string    err_c = expect_200_with_hash(
            resp_c, "CaseC", nullptr, out_c);
        if (!err_c.empty()) return stop_drain_fail(err_c);
        if (out_c["hash"].get<std::string>()
              == std::string(k_default_stochastic_seed42_b8)) {
            return stop_drain_fail(
                "CaseC hash == seed=42 anchor (no seed sensitivity)");
        }
        if (out_c["hash"].get<std::string>()
              == std::string(k_canonical_greedy_b8)) {
            return stop_drain_fail(
                "CaseC hash == greedy canonical");
        }

        // ---- Case D: full non-default knobs ------------------------
        nlohmann::json req_d;
        req_d["prompt"]        = "Hello, my name is";
        req_d["decode_budget"] = 8;
        req_d["sampling"] = {
            {"mode", "stochastic"},
            {"seed", 42},
            {"temperature", 0.7},
            {"top_k", 40},
            {"top_p", 0.95},
            {"top_p_min_keep", 1}
        };
        auto resp_d = post_completion(req_d);
        nlohmann::json out_d;
        std::string    err_d = expect_200_with_hash(
            resp_d, "CaseD", nullptr, out_d);
        if (!err_d.empty()) return stop_drain_fail(err_d);
        if (out_d["hash"].get<std::string>()
              == std::string(k_default_stochastic_seed42_b8)) {
            return stop_drain_fail(
                "CaseD hash == default-stochastic anchor (knobs not "
                "wired)");
        }
        if (out_d["hash"].get<std::string>()
              == std::string(k_canonical_greedy_b8)) {
            return stop_drain_fail(
                "CaseD hash == greedy canonical");
        }

        srv.stop();
        listen_fut.get();

        fprintf(stdout, "CaseA  hash=%s n_decoded=%d\n",
            out_a["hash"].get<std::string>().c_str(),
            out_a["n_decoded"].get<int>());
        fprintf(stdout, "CaseB  hash=%s n_decoded=%d\n",
            out_b["hash"].get<std::string>().c_str(),
            out_b["n_decoded"].get<int>());
        fprintf(stdout, "CaseC  hash=%s n_decoded=%d\n",
            out_c["hash"].get<std::string>().c_str(),
            out_c["n_decoded"].get<int>());
        fprintf(stdout, "CaseD  hash=%s n_decoded=%d\n",
            out_d["hash"].get<std::string>().c_str(),
            out_d["n_decoded"].get<int>());
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
