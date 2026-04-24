// test_hpx_llama_selective_mul_mat_smoke.cpp
//
// End-to-end smoke test: HPX selective mul_mat path vs. reference path.
//
// Loads a model from LLAMACPP_TEST_MODELFILE, runs a prefill + single-token
// decode through both paths, and asserts that every logit from the decode
// step matches exactly.
//
// Reference context:  LLAMA_USE_HPX unset  (standard scheduler path)
// Selective context:  LLAMA_USE_HPX=1 + LLAMA_HPX_SELECTIVE_MUL_MAT=1
//
// Two-phase execution per context:
//   1. Prefill — all prompt tokens in one batch (batched=true).
//   2. Decode  — one fixed token (kDecodeToken) alone (batched=false).
//      With LLAMA_HPX_SELECTIVE_MLP_PACKET=1 this is the step where
//      packet_eligible=true and the QBRIDGE/GLU packet fires.
//
// Logits are captured only from the decode step and compared.
// Run with LLAMA_HPX_PACKET_COMPILE_LOG=1 to confirm matcher=glu_qbridge
// appears in stderr for the selective context.
//
// The test is skipped (not failed) when LLAMACPP_TEST_MODELFILE is absent.

#include "llama.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace
{

// Fixed decode token used by both paths — must be a valid vocab id.
// Token 1 (BOS) is always present in LLaMA-family models.
constexpr llama_token kDecodeToken = 1;

TEST(HpxLlamaSelectiveMulMatSmoke, DecodeSingleTokenMatchesReference)
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
    // Tokenize a short prompt.
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

    // n_ctx must hold the prompt plus the single decode token.
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx                = static_cast<uint32_t>(n_tokens + 4);
    cparams.n_batch              = static_cast<uint32_t>(n_tokens);
    cparams.n_threads            = 1;
    cparams.n_threads_batch      = 1;

    // -----------------------------------------------------------------------
    // Helper: prefill the prompt, then decode one fixed token.
    // Returns logits from the decode step only.
    // -----------------------------------------------------------------------
    auto run_prefill_then_decode = [&](llama_context * ctx) -> std::vector<float>
    {
        // Phase 1: prefill — all prompt tokens, batched=true.
        llama_batch prefill = llama_batch_get_one(
            tokens.data(), static_cast<int32_t>(tokens.size()));
        EXPECT_EQ(llama_decode(ctx, prefill), 0) << "prefill failed";

        // Phase 2: decode — one fixed token, batched=false so packet_eligible=true.
        llama_token decode_tok = kDecodeToken;
        llama_batch decode = llama_batch_get_one(&decode_tok, 1);
        EXPECT_EQ(llama_decode(ctx, decode), 0) << "decode failed";

        const int     n_vocab = llama_vocab_n_tokens(vocab);
        const float * logits  = llama_get_logits(ctx);
        EXPECT_NE(logits, nullptr);

        return std::vector<float>(logits, logits + n_vocab);
    };

    // -----------------------------------------------------------------------
    // Reference context — standard scheduler path, no HPX.
    // -----------------------------------------------------------------------
#if defined(_WIN32)
    _putenv_s("LLAMA_USE_HPX", "");
    _putenv_s("LLAMA_HPX_SELECTIVE_MUL_MAT", "");
#else
    unsetenv("LLAMA_USE_HPX");
    unsetenv("LLAMA_HPX_SELECTIVE_MUL_MAT");
#endif

    llama_context * ctx_ref = llama_init_from_model(model, cparams);
    ASSERT_NE(ctx_ref, nullptr) << "failed to create reference context";
    std::vector<float> logits_ref = run_prefill_then_decode(ctx_ref);
    llama_free(ctx_ref);

    // -----------------------------------------------------------------------
    // Selective HPX context — LLAMA_USE_HPX=1, LLAMA_HPX_SELECTIVE_MUL_MAT=1.
    // With LLAMA_HPX_SELECTIVE_MLP_PACKET=1 (set externally) the decode step
    // is packet_eligible and QBRIDGE fires for Q4_K_M weights.
    // -----------------------------------------------------------------------
#if defined(_WIN32)
    _putenv_s("LLAMA_USE_HPX", "1");
    _putenv_s("LLAMA_HPX_SELECTIVE_MUL_MAT", "1");
#else
    setenv("LLAMA_USE_HPX",               "1", /*overwrite=*/1);
    setenv("LLAMA_HPX_SELECTIVE_MUL_MAT", "1", /*overwrite=*/1);
#endif

    llama_context * ctx_sel = llama_init_from_model(model, cparams);
    ASSERT_NE(ctx_sel, nullptr) << "failed to create selective HPX context";
    std::vector<float> logits_sel = run_prefill_then_decode(ctx_sel);
    llama_free(ctx_sel);

    // Clean up env so later tests are not affected.
#if defined(_WIN32)
    _putenv_s("LLAMA_USE_HPX", "");
    _putenv_s("LLAMA_HPX_SELECTIVE_MUL_MAT", "");
#else
    unsetenv("LLAMA_USE_HPX");
    unsetenv("LLAMA_HPX_SELECTIVE_MUL_MAT");
#endif

    // -----------------------------------------------------------------------
    // Compare logits from the decode step only.
    // -----------------------------------------------------------------------
    // EXPECT_NEAR rather than EXPECT_FLOAT_EQ: the selective path runs our
    // scalar SWIGLU kernel while the reference uses ggml's SIMD kernel.
    // Both are mathematically identical but produce up to ~4 ULP differences
    // (observed max ~3.3e-6 on M4). kLogitTol is set just above that floor.
    constexpr float kLogitTol = 5e-6f;

    ASSERT_EQ(logits_ref.size(), logits_sel.size());
    float max_abs_diff = 0.0f;
    for (size_t i = 0; i < logits_ref.size(); ++i)
    {
        const float diff = std::abs(logits_ref[i] - logits_sel[i]);
        if (diff > max_abs_diff) max_abs_diff = diff;
    }
    for (size_t i = 0; i < logits_ref.size(); ++i)
    {
        EXPECT_NEAR(logits_ref[i], logits_sel[i], kLogitTol)
            << "logit mismatch at index " << i
            << " (max_abs_diff=" << max_abs_diff << ")";
    }

    llama_model_free(model);
}

} // namespace
