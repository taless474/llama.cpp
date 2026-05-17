// Token-id hash helpers shared across the HPX continuous-batch gate.
// Byte-identical to multiseq-batch-gate.
//
// FNV-1a-style fold over int32 token ids. `k_token_hash_init` is the
// FNV-1a 64-bit offset basis; `k_token_hash_empty` is the sentinel
// that finalize_hash() returns when no tokens have been folded.
//
// These constants and the fold function are correctness fingerprints,
// not performance metrics. Hash comparisons are only meaningful within
// the same model / prompt / decode policy / batch shape / scheduling
// policy.

#pragma once

#include <cstdint>

namespace token_hash {

inline constexpr uint64_t k_token_hash_init  = 0xcbf29ce484222325ull;
inline constexpr uint64_t k_token_hash_empty = 0ull;
inline constexpr uint64_t k_token_hash_prime = 0x100000001b3ull;

// Budget-8 canonical hash anchor for the TinyLlama-1.1B-Chat-v1.0 /
// "Hello, my name is" / greedy decoding shape on the current build.
// Used by the validation harness as an anchor; only meaningful under
// the documented (model, prompt, policy, batch shape) tuple.
inline constexpr uint64_t k_canonical_budget_8 = 0x0619d4d1900c2365ull;

inline uint64_t fold_token_hash(uint64_t state, int32_t token_id) noexcept {
    state ^= static_cast<uint64_t>(static_cast<uint32_t>(token_id));
    state *= k_token_hash_prime;
    return state;
}

}  // namespace token_hash
