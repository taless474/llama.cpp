// test_hpx_llama_smoke.cpp
//
// End-to-end smoke test: HPX exec path vs. reference scheduler path.
//
// Loads a model from LLAMACPP_TEST_MODELFILE, runs a single decode batch
// through both paths, and asserts that every logit matches exactly.
//
// The test is skipped (not failed) when the env var is absent, so it is safe
// to include in the suite unconditionally.

#include "llama.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace
{

TEST(HpxLlamaSmoke, DecodeSingleTokenMatchesReference)
{
    const char * model_path = getenv("LLAMACPP_TEST_MODELFILE");
    if (model_path == nullptr || model_path[0] == '\0')
    {
        GTEST_SKIP() << "LLAMACPP_TEST_MODELFILE not set — skipping smoke test";
    }

    // -----------------------------------------------------------------------
    // Load model once; share across both contexts.
    // -----------------------------------------------------------------------
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers       = 0; // CPU only

    llama_model * model = llama_model_load_from_file(model_path, mparams);
    ASSERT_NE(model, nullptr) << "failed to load model from " << model_path;

    // -----------------------------------------------------------------------
    // Short fixed prompt.
    // -----------------------------------------------------------------------
    const std::string prompt = "Hello";

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens(prompt.size() + 4);
    int n_tokens = llama_tokenize(
        vocab,
        prompt.c_str(), static_cast<int32_t>(prompt.size()),
        tokens.data(), static_cast<int32_t>(tokens.size()),
        /*add_special=*/true, /*parse_special=*/false);
    ASSERT_GT(n_tokens, 0) << "tokenization produced no tokens";
    tokens.resize(static_cast<size_t>(n_tokens));

    // -----------------------------------------------------------------------
    // Helper: run one decode and return logits for all tokens.
    // -----------------------------------------------------------------------
    auto run_decode = [&](llama_context * ctx) -> std::vector<float>
    {
        llama_batch batch = llama_batch_get_one(
            tokens.data(), static_cast<int32_t>(tokens.size()));

        const int ret = llama_decode(ctx, batch);
        EXPECT_EQ(ret, 0) << "llama_decode failed";

        const int n_vocab = llama_vocab_n_tokens(vocab);
        const float * logits = llama_get_logits(ctx);
        EXPECT_NE(logits, nullptr);

        return std::vector<float>(logits, logits + n_vocab);
    };

    // -----------------------------------------------------------------------
    // Reference context — standard scheduler path.
    // -----------------------------------------------------------------------
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx                = static_cast<uint32_t>(n_tokens + 4);
    cparams.n_batch              = static_cast<uint32_t>(n_tokens);
    cparams.n_threads            = 1;
    cparams.n_threads_batch      = 1;

    // Ensure LLAMA_USE_HPX is unset for the reference context.
#if defined(_WIN32)
    _putenv_s("LLAMA_USE_HPX", "");
#else
    unsetenv("LLAMA_USE_HPX");
#endif

    llama_context * ctx_ref = llama_init_from_model(model, cparams);
    ASSERT_NE(ctx_ref, nullptr) << "failed to create reference context";
    std::vector<float> logits_ref = run_decode(ctx_ref);
    llama_free(ctx_ref);

    // -----------------------------------------------------------------------
    // HPX context — LLAMA_USE_HPX=1.
    // -----------------------------------------------------------------------
#if defined(_WIN32)
    _putenv_s("LLAMA_USE_HPX", "1");
#else
    setenv("LLAMA_USE_HPX", "1", /*overwrite=*/1);
#endif

    llama_context * ctx_hpx = llama_init_from_model(model, cparams);
    ASSERT_NE(ctx_hpx, nullptr) << "failed to create HPX context";
    std::vector<float> logits_hpx = run_decode(ctx_hpx);
    llama_free(ctx_hpx);

    // Clean up env so later tests (if any) are not affected.
#if defined(_WIN32)
    _putenv_s("LLAMA_USE_HPX", "");
#else
    unsetenv("LLAMA_USE_HPX");
#endif

    // -----------------------------------------------------------------------
    // Compare.
    // -----------------------------------------------------------------------
    ASSERT_EQ(logits_ref.size(), logits_hpx.size());
    for (size_t i = 0; i < logits_ref.size(); ++i)
    {
        EXPECT_FLOAT_EQ(logits_ref[i], logits_hpx[i])
            << "logit mismatch at index " << i;
    }

    llama_model_free(model);
}

} // namespace
