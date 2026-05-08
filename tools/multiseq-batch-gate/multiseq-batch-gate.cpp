// Multi-sequence llama_batch gate.
//
// Steps implemented:
//   - Step 1: structural prerequisites (capacity).
//   - Step 2: single-seq baseline (--n-seqs 1, default).
//   - Step 3: N-seq same-prompt equality check (--n-seqs >= 2,
//             uniform --decode-budget).
//   - Step 4: N-seq mixed-budget smoke
//             (--decode-budgets <csv>, e.g. "8,64,256"). Sequences
//             finish at different iterations; KV is cleared per seq
//             as it completes; remaining seqs continue without
//             cross-talk.
//   - Step 5: scaled mixed-budget run
//             (--decode-budget-mix <csv> --n-seqs N). The mix is
//             round-robined across N seqs, e.g.
//             `--n-seqs 99 --decode-budget-mix 8,64,256`
//             yields 33+33+33 seqs with the Phase 3 primary target
//             shape. Output is collapsed to a per-budget-class
//             summary; per-seq lines only print for small N.
//
// All N-seq runs share one llama_model, one llama_context, and one
// shared llama_batch. Each row carries a single seq_id. Logits are
// requested only on the rows whose argmax we need (last prompt row
// per seq during prefill; every decode row during the decode loop).
//
// See docs/hpx/multiseq_llama_batch_gate.md.
//
// Hash convention is byte-identical to tools/serving-bench/harness.h:
//   k_token_hash_init = 0xcbf29ce484222325
//   prime             = 0x100000001b3
//   state ^= u32(int32_t(token_id)); state *= prime
//   n_decoded == 0 ⇒ hash = 0
// End-of-generation tokens, if sampled, terminate the loop without
// being folded into the hash.
//
// Canonical references for prompt = "Hello, my name is" on
// TinyLlama 1.1B Q4_K_M, greedy argmax:
//   budget  8 → 0x0619d4d1900c2365
//   budget 16 → 0x833045f1e2ebf49f
// Each is shape-specific.
//
// KV position convention for a successful run with prompt length P
// and decode budget D:
//   pos_min == 0
//   pos_max == P + D - 2  (NOT P + D - 1)
// because generated token #1 comes from the prefill's final logits
// row and does not occupy a new KV cell — only D - 1 generated
// tokens are fed back as decode rows.

#include "common.h"
#include "llama.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr uint64_t k_token_hash_init  = 0xcbf29ce484222325ull;
constexpr uint64_t k_token_hash_empty = 0ull;
constexpr uint64_t k_token_hash_prime = 0x100000001b3ull;

uint64_t fold_token_hash(uint64_t state, int32_t token_id) noexcept {
    state ^= static_cast<uint64_t>(static_cast<uint32_t>(token_id));
    state *= k_token_hash_prime;
    return state;
}

struct cli_args {
    std::string          model_path;
    std::string          prompt        = "Hello, my name is";
    int32_t              ctx_size      = 1024;
    int32_t              n_seq_max     = 3;
    int32_t              n_batch       = 512;
    int32_t              n_threads     = 2;
    int32_t              decode_budget = 8;
    int32_t              repeat        = 1;
    int32_t              n_seqs        = 1;
    std::vector<int32_t> decode_budgets;     // empty ⇒ use scalar decode_budget
    std::vector<int32_t> decode_budget_mix;  // round-robined over n_seqs
};

void print_usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s --model <path> [options]\n"
        "  --model <path>           (required) path to .gguf model file\n"
        "  --prompt <string>        default: \"Hello, my name is\"\n"
        "  --ctx-size <int>         default: 1024\n"
        "  --n-seq-max <int>        default: 3\n"
        "  --n-batch <int>          default: 512\n"
        "  --n-threads <int>        default: 2\n"
        "  --decode-budget <int>    default: 8 (used for every seq)\n"
        "  --repeat <int>           default: 1\n"
        "  --n-seqs <int>           default: 1\n"
        "  --decode-budgets <csv>   per-seq budgets, e.g. \"8,64,256\".\n"
        "                           If given, n_seqs = number of values\n"
        "                           and --decode-budget is ignored.\n"
        "  --decode-budget-mix <csv> mix to round-robin over --n-seqs,\n"
        "                           e.g. \"8,64,256\". Use with --n-seqs N\n"
        "                           to avoid typing an N-entry CSV.\n"
        "                           Triggers GATE_STEP5 emit.\n"
        "                           Step label:\n"
        "                             1 seq                  ⇒ GATE_STEP2\n"
        "                             N seqs, all equal      ⇒ GATE_STEP3\n"
        "                             N seqs, mixed budgets  ⇒ GATE_STEP4\n"
        "                             --decode-budget-mix    ⇒ GATE_STEP5\n",
        argv0);
}

bool parse_int(const char * s, int32_t & out) {
    if (s == nullptr || *s == '\0') {
        return false;
    }
    char * end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (end == s || *end != '\0') {
        return false;
    }
    if (v < INT32_MIN || v > INT32_MAX) {
        return false;
    }
    out = static_cast<int32_t>(v);
    return true;
}

bool parse_csv_int_list(const char * s, std::vector<int32_t> & out) {
    out.clear();
    if (s == nullptr || *s == '\0') {
        return false;
    }
    const std::string str = s;
    size_t i = 0;
    while (i <= str.size()) {
        size_t j = str.find(',', i);
        if (j == std::string::npos) j = str.size();
        const std::string tok = str.substr(i, j - i);
        if (tok.empty()) {
            return false;
        }
        int32_t v = 0;
        if (!parse_int(tok.c_str(), v)) {
            return false;
        }
        if (v <= 0) {
            return false;
        }
        out.push_back(v);
        if (j == str.size()) break;
        i = j + 1;
    }
    return !out.empty();
}

bool parse_args(int argc, char ** argv, cli_args & args) {
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto need_value = [&](const char * flag) -> const char * {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: %s requires a value\n", flag);
                return nullptr;
            }
            return argv[++i];
        };
        if (a == "--model") {
            const char * v = need_value("--model");
            if (!v) return false;
            args.model_path = v;
        } else if (a == "--prompt") {
            const char * v = need_value("--prompt");
            if (!v) return false;
            args.prompt = v;
        } else if (a == "--ctx-size") {
            const char * v = need_value("--ctx-size");
            if (!v) return false;
            if (!parse_int(v, args.ctx_size)) {
                fprintf(stderr, "error: --ctx-size must be an integer\n");
                return false;
            }
        } else if (a == "--n-seq-max") {
            const char * v = need_value("--n-seq-max");
            if (!v) return false;
            if (!parse_int(v, args.n_seq_max)) {
                fprintf(stderr, "error: --n-seq-max must be an integer\n");
                return false;
            }
        } else if (a == "--n-batch") {
            const char * v = need_value("--n-batch");
            if (!v) return false;
            if (!parse_int(v, args.n_batch)) {
                fprintf(stderr, "error: --n-batch must be an integer\n");
                return false;
            }
        } else if (a == "--n-threads") {
            const char * v = need_value("--n-threads");
            if (!v) return false;
            if (!parse_int(v, args.n_threads)) {
                fprintf(stderr, "error: --n-threads must be an integer\n");
                return false;
            }
        } else if (a == "--decode-budget") {
            const char * v = need_value("--decode-budget");
            if (!v) return false;
            if (!parse_int(v, args.decode_budget)) {
                fprintf(stderr, "error: --decode-budget must be an integer\n");
                return false;
            }
            if (args.decode_budget <= 0) {
                fprintf(stderr, "error: --decode-budget must be > 0\n");
                return false;
            }
        } else if (a == "--repeat") {
            const char * v = need_value("--repeat");
            if (!v) return false;
            if (!parse_int(v, args.repeat)) {
                fprintf(stderr, "error: --repeat must be an integer\n");
                return false;
            }
            if (args.repeat <= 0) {
                fprintf(stderr, "error: --repeat must be > 0\n");
                return false;
            }
        } else if (a == "--n-seqs") {
            const char * v = need_value("--n-seqs");
            if (!v) return false;
            if (!parse_int(v, args.n_seqs)) {
                fprintf(stderr, "error: --n-seqs must be an integer\n");
                return false;
            }
            if (args.n_seqs <= 0) {
                fprintf(stderr, "error: --n-seqs must be > 0\n");
                return false;
            }
        } else if (a == "--decode-budgets") {
            const char * v = need_value("--decode-budgets");
            if (!v) return false;
            if (!parse_csv_int_list(v, args.decode_budgets)) {
                fprintf(stderr,
                    "error: --decode-budgets must be a non-empty CSV "
                    "of positive integers, got: %s\n", v);
                return false;
            }
        } else if (a == "--decode-budget-mix") {
            const char * v = need_value("--decode-budget-mix");
            if (!v) return false;
            if (!parse_csv_int_list(v, args.decode_budget_mix)) {
                fprintf(stderr,
                    "error: --decode-budget-mix must be a non-empty CSV "
                    "of positive integers, got: %s\n", v);
                return false;
            }
        } else if (a == "-h" || a == "--help") {
            print_usage(argv[0]);
            return false;
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", a.c_str());
            print_usage(argv[0]);
            return false;
        }
    }
    if (args.model_path.empty()) {
        fprintf(stderr, "error: --model is required\n");
        print_usage(argv[0]);
        return false;
    }
    if (!args.decode_budgets.empty() && !args.decode_budget_mix.empty()) {
        fprintf(stderr,
            "error: --decode-budgets and --decode-budget-mix are "
            "mutually exclusive\n");
        return false;
    }
    // --decode-budgets fixes the per-seq budget vector and overrides
    // --n-seqs. --decode-budget-mix is round-robined across --n-seqs
    // (which the user must set explicitly when N is large).
    if (!args.decode_budgets.empty()) {
        args.n_seqs = static_cast<int32_t>(args.decode_budgets.size());
    }
    return true;
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

struct seq_state {
    int32_t                  seq_id           = 0;
    int32_t                  decode_budget    = 0;
    int32_t                  n_decoded        = 0;
    int32_t                  pos_next         = 0;
    int32_t                  i_batch          = -1;
    llama_token              last_token       = 0;
    bool                     done             = false;
    int32_t                  done_iter        = -1;  // 0 if from prefill
    llama_pos                pos_max_at_clear = -1;
    bool                     kv_cleared       = false;
    uint64_t                 hash_state       = k_token_hash_init;
    std::vector<llama_token> generated_tokens;

    uint64_t finalize_hash() const noexcept {
        return (n_decoded == 0) ? k_token_hash_empty : hash_state;
    }
};

struct multiseq_result {
    bool                   ok    = false;
    std::string            error;
    std::vector<seq_state> seqs;
};

// Mark `seq` as done at the given iter, snapshot its KV pos_max,
// clear its KV, then verify (a) the cleared seq reads back as empty
// and (b) every still-active seq is undisturbed. Returns false and
// fills `r.error` on any violation.
bool finalize_and_clear_seq(
    multiseq_result &        r,
    seq_state &              seq,
    int32_t                  iter,
    llama_memory_t           mem,
    const std::vector<seq_state> & seqs)
{
    seq.done             = true;
    seq.done_iter        = iter;
    seq.pos_max_at_clear = llama_memory_seq_pos_max(mem, seq.seq_id);

    // Snapshot still-active siblings before we clear `seq`.
    std::vector<llama_pos> sib_min(seqs.size(), -1);
    std::vector<llama_pos> sib_max(seqs.size(), -1);
    for (size_t t = 0; t < seqs.size(); t++) {
        const seq_state & ot = seqs[t];
        if (ot.seq_id == seq.seq_id || ot.kv_cleared || ot.done) continue;
        sib_min[t] = llama_memory_seq_pos_min(mem, ot.seq_id);
        sib_max[t] = llama_memory_seq_pos_max(mem, ot.seq_id);
    }

    if (!llama_memory_seq_rm(mem, seq.seq_id, /*p0=*/-1, /*p1=*/-1)) {
        r.error = "llama_memory_seq_rm returned false";
        return false;
    }
    seq.kv_cleared = true;

    const llama_pos pmin = llama_memory_seq_pos_min(mem, seq.seq_id);
    const llama_pos pmax = llama_memory_seq_pos_max(mem, seq.seq_id);
    if (pmin != -1 || pmax != -1) {
        char buf[200];
        std::snprintf(buf, sizeof(buf),
            "seq %d not empty after seq_rm: pos_min=%d pos_max=%d",
            seq.seq_id, pmin, pmax);
        r.error = buf;
        return false;
    }

    // Re-read still-active siblings and require unchanged pos_min/max.
    for (size_t t = 0; t < seqs.size(); t++) {
        const seq_state & ot = seqs[t];
        if (ot.seq_id == seq.seq_id || ot.kv_cleared || ot.done) continue;
        const llama_pos tmin = llama_memory_seq_pos_min(mem, ot.seq_id);
        const llama_pos tmax = llama_memory_seq_pos_max(mem, ot.seq_id);
        if (tmin != sib_min[t] || tmax != sib_max[t]) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "cross-talk: clearing seq %d disturbed seq %d "
                "(pos_min=%d pos_max=%d, expected %d %d)",
                seq.seq_id, ot.seq_id, tmin, tmax, sib_min[t], sib_max[t]);
            r.error = buf;
            return false;
        }
    }
    return true;
}

// Run prefill + per-step greedy-argmax decode for `n_seqs` seqs using
// the same `prompt_tokens`. Each seq has its own decode_budget. As a
// seq reaches its budget the gate immediately clears that seq's KV
// and asserts no cross-talk on remaining seqs.
multiseq_result run_multiseq(
    llama_context *                  ctx,
    const llama_vocab *              vocab,
    int32_t                          n_vocab,
    const std::vector<llama_token> & prompt_tokens,
    const std::vector<int32_t> &     budgets,
    int32_t                          batch_capacity)
{
    multiseq_result r{};
    r.ok = false;

    auto * mem = llama_get_memory(ctx);

    // Fresh KV across the whole memory (no-op on first run; needed
    // when --repeat > 1 or when a prior caller left state behind).
    llama_memory_clear(mem, /*data=*/false);

    const int32_t n_seqs   = static_cast<int32_t>(budgets.size());
    const int32_t n_prompt = static_cast<int32_t>(prompt_tokens.size());

    r.seqs.resize(static_cast<size_t>(n_seqs));
    for (int32_t s = 0; s < n_seqs; s++) {
        r.seqs[s].seq_id        = s;
        r.seqs[s].decode_budget = budgets[s];
        r.seqs[s].generated_tokens.reserve(
            static_cast<size_t>(budgets[s]));
    }

    llama_batch batch = llama_batch_init(batch_capacity, /*embd=*/0,
                                         /*n_seq_max=*/1);

    // ---- Prefill: append prompt rows for every seq in one shared batch.
    common_batch_clear(batch);
    for (int32_t s = 0; s < n_seqs; s++) {
        seq_state & seq = r.seqs[s];
        for (int32_t p = 0; p < n_prompt; p++) {
            const bool last = (p == n_prompt - 1);
            common_batch_add(batch, prompt_tokens[p], /*pos=*/p,
                             /*seq_ids=*/{seq.seq_id}, /*logits=*/last);
            if (last) {
                seq.i_batch = batch.n_tokens - 1;
            }
        }
        seq.pos_next = n_prompt;
    }

    if (llama_decode(ctx, batch) != 0) {
        r.error = "llama_decode failed during prefill";
        llama_batch_free(batch);
        return r;
    }
    llama_synchronize(ctx);

    const int32_t prefill_iter = 0;

    // First sampled token per seq; iter 0 (= prefill) may already
    // satisfy a seq with budget 1.
    for (int32_t s = 0; s < n_seqs; s++) {
        seq_state & seq = r.seqs[s];
        const float * logits = llama_get_logits_ith(ctx, seq.i_batch);
        if (logits == nullptr) {
            r.error = "llama_get_logits_ith returned null after prefill";
            llama_batch_free(batch);
            return r;
        }
        const llama_token next_id = argmax(logits, n_vocab);
        if (llama_vocab_is_eog(vocab, next_id)) {
            // Zero-token early eog. Still clear KV and finalize.
            if (!finalize_and_clear_seq(r, seq, prefill_iter, mem, r.seqs)) {
                llama_batch_free(batch);
                return r;
            }
            continue;
        }
        seq.generated_tokens.push_back(next_id);
        seq.hash_state = fold_token_hash(seq.hash_state,
                                         static_cast<int32_t>(next_id));
        seq.n_decoded++;
        seq.last_token = next_id;
        if (seq.n_decoded >= seq.decode_budget) {
            if (!finalize_and_clear_seq(r, seq, prefill_iter, mem, r.seqs)) {
                llama_batch_free(batch);
                return r;
            }
        }
    }

    auto any_active = [&]() {
        for (const auto & seq : r.seqs) {
            if (!seq.done) return true;
        }
        return false;
    };

    // ---- Decode loop: one row per still-active seq per iteration. ----
    int32_t iter = 0;  // iter 0 was prefill; the loop starts at iter 1
    while (any_active()) {
        iter++;
        common_batch_clear(batch);
        std::vector<int32_t> active_idx;
        active_idx.reserve(static_cast<size_t>(n_seqs));
        for (int32_t s = 0; s < n_seqs; s++) {
            seq_state & seq = r.seqs[s];
            if (seq.done) continue;
            common_batch_add(batch, seq.last_token, /*pos=*/seq.pos_next,
                             /*seq_ids=*/{seq.seq_id}, /*logits=*/true);
            seq.i_batch = batch.n_tokens - 1;
            seq.pos_next++;
            active_idx.push_back(s);
        }
        if (batch.n_tokens == 0) {
            break;  // defensive; any_active() said true but no rows
        }

        if (llama_decode(ctx, batch) != 0) {
            r.error = "llama_decode failed during decode loop";
            llama_batch_free(batch);
            return r;
        }
        llama_synchronize(ctx);

        for (int32_t s : active_idx) {
            seq_state & seq = r.seqs[s];
            const float * logits = llama_get_logits_ith(ctx, seq.i_batch);
            if (logits == nullptr) {
                r.error = "llama_get_logits_ith returned null during decode";
                llama_batch_free(batch);
                return r;
            }
            const llama_token next_id = argmax(logits, n_vocab);
            if (llama_vocab_is_eog(vocab, next_id)) {
                if (!finalize_and_clear_seq(r, seq, iter, mem, r.seqs)) {
                    llama_batch_free(batch);
                    return r;
                }
                continue;
            }
            seq.generated_tokens.push_back(next_id);
            seq.hash_state = fold_token_hash(seq.hash_state,
                                             static_cast<int32_t>(next_id));
            seq.n_decoded++;
            seq.last_token = next_id;
            if (seq.n_decoded >= seq.decode_budget) {
                if (!finalize_and_clear_seq(r, seq, iter, mem, r.seqs)) {
                    llama_batch_free(batch);
                    return r;
                }
            }
        }
    }

    llama_batch_free(batch);
    r.ok = true;
    return r;
}

void print_token_ids(const char * label,
                     const std::vector<llama_token> & toks,
                     int32_t max_print = 32) {
    fprintf(stdout, "%s [", label);
    const int32_t n =
        std::min<int32_t>(max_print, static_cast<int32_t>(toks.size()));
    for (int32_t i = 0; i < n; i++) {
        fprintf(stdout, "%s%d", i == 0 ? "" : ", ",
                static_cast<int>(toks[i]));
    }
    if (n < static_cast<int32_t>(toks.size())) {
        fprintf(stdout, ", ... (+%d more)",
                static_cast<int>(toks.size()) - n);
    }
    fprintf(stdout, "]\n");
}

void emit_pass(const char * step) {
    fprintf(stdout, "%s: PASS\n", step);
}

void emit_fail(const char * step, const char * reason) {
    fprintf(stdout, "%s: FAIL: %s\n", step, reason);
}

}  // namespace

int main(int argc, char ** argv) {
    cli_args args;
    if (!parse_args(argc, argv, args)) {
        return 2;
    }

    // Resolve per-seq budgets.
    std::vector<int32_t> budgets;
    const bool using_mix = !args.decode_budget_mix.empty();
    if (!args.decode_budgets.empty()) {
        budgets = args.decode_budgets;
    } else if (using_mix) {
        // Round-robin the mix across --n-seqs.
        budgets.resize(static_cast<size_t>(args.n_seqs));
        const size_t m = args.decode_budget_mix.size();
        for (int32_t i = 0; i < args.n_seqs; i++) {
            budgets[static_cast<size_t>(i)] =
                args.decode_budget_mix[static_cast<size_t>(i) % m];
        }
    } else {
        budgets.assign(static_cast<size_t>(args.n_seqs), args.decode_budget);
    }
    const int32_t n_seqs = static_cast<int32_t>(budgets.size());

    // Auto-bump requested n_seq_max so the user doesn't need to also pass
    // --n-seq-max alongside --n-seqs N.
    if (args.n_seq_max < n_seqs) {
        args.n_seq_max = n_seqs;
    }

    // Step label:
    //   --decode-budget-mix    ⇒ STEP5
    //   else mixed budgets     ⇒ STEP4
    //   else N >= 2 (uniform)  ⇒ STEP3
    //   else                   ⇒ STEP2
    bool mixed = false;
    for (int32_t s = 1; s < n_seqs; s++) {
        if (budgets[s] != budgets[0]) { mixed = true; break; }
    }
    const char * STEP =
        using_mix          ? "GATE_STEP5" :
        mixed              ? "GATE_STEP4" :
        (n_seqs >= 2)      ? "GATE_STEP3" :
                             "GATE_STEP2";

    common_init();
    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(
        args.model_path.c_str(), model_params);
    if (model == nullptr) {
        emit_fail(STEP, "model load failed");
        llama_backend_free();
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx     = static_cast<uint32_t>(args.ctx_size);
    ctx_params.n_batch   = static_cast<uint32_t>(args.n_batch);
    ctx_params.n_seq_max = static_cast<uint32_t>(args.n_seq_max);
    ctx_params.n_threads       = args.n_threads;
    ctx_params.n_threads_batch = args.n_threads;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        emit_fail(STEP, "context create failed");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    const uint32_t actual_n_ctx     = llama_n_ctx(ctx);
    const uint32_t actual_n_seq_max = llama_n_seq_max(ctx);
    const uint32_t actual_n_batch   = llama_n_batch(ctx);

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> prompt_tokens =
        common_tokenize(ctx, args.prompt, /*add_special=*/true,
                        /*parse_special=*/true);
    const int32_t n_prompt_tokens =
        static_cast<int32_t>(prompt_tokens.size());

    fprintf(stdout, "model_path:          %s\n",  args.model_path.c_str());
    fprintf(stdout, "prompt:              \"%s\"\n", args.prompt.c_str());
    fprintf(stdout, "requested ctx_size:  %d\n", args.ctx_size);
    fprintf(stdout, "requested n_seq_max: %d\n", args.n_seq_max);
    fprintf(stdout, "requested n_batch:   %d\n", args.n_batch);
    fprintf(stdout, "actual n_ctx:        %u\n", actual_n_ctx);
    fprintf(stdout, "actual n_seq_max:    %u\n", actual_n_seq_max);
    fprintf(stdout, "actual n_batch:      %u\n", actual_n_batch);
    fprintf(stdout, "n_vocab:             %d\n", n_vocab);
    fprintf(stdout, "prompt_tokens:       %d\n", n_prompt_tokens);
    fprintf(stdout, "n_seqs:              %d\n", n_seqs);
    fprintf(stdout, "decode_budgets:      [");
    {
        const int32_t k = std::min<int32_t>(16, n_seqs);
        for (int32_t s = 0; s < k; s++) {
            fprintf(stdout, "%s%d", s == 0 ? "" : ", ", budgets[s]);
        }
        if (k < n_seqs) {
            fprintf(stdout, ", ... (+%d more)", n_seqs - k);
        }
    }
    fprintf(stdout, "]\n");
    if (using_mix) {
        fprintf(stdout, "decode_budget_mix:   [");
        for (size_t i = 0; i < args.decode_budget_mix.size(); i++) {
            fprintf(stdout, "%s%d", i == 0 ? "" : ", ",
                    args.decode_budget_mix[i]);
        }
        fprintf(stdout, "]\n");
    }
    fprintf(stdout, "repeat:              %d\n", args.repeat);
    fprintf(stdout, "step_label:          %s\n", STEP);
    fflush(stdout);

    // ---- Step 1 structural prerequisites. ------------------------------
    auto cleanup = [&]() {
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
    };

    if (actual_n_seq_max < static_cast<uint32_t>(args.n_seq_max)) {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "actual n_seq_max %u < requested n_seq_max %d",
            actual_n_seq_max, args.n_seq_max);
        emit_fail(STEP, buf); cleanup(); return 1;
    }
    if (n_seqs > args.n_seq_max) {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "n_seqs %d > --n-seq-max %d", n_seqs, args.n_seq_max);
        emit_fail(STEP, buf); cleanup(); return 1;
    }
    if (actual_n_ctx < static_cast<uint32_t>(n_prompt_tokens + 256)) {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "actual n_ctx %u < prompt_tokens %d + 256",
            actual_n_ctx, n_prompt_tokens);
        emit_fail(STEP, buf); cleanup(); return 1;
    }
    if (actual_n_batch <
        static_cast<uint32_t>(args.n_seq_max * n_prompt_tokens)) {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "actual n_batch %u < n_seq_max %d * prompt_tokens %d",
            actual_n_batch, args.n_seq_max, n_prompt_tokens);
        emit_fail(STEP, buf); cleanup(); return 1;
    }

    // The largest decode budget defines how big n_ctx must be for the
    // longest seq's KV; check that the slack is sufficient.
    const int32_t max_budget = *std::max_element(budgets.begin(), budgets.end());
    if (actual_n_ctx <
        static_cast<uint32_t>(n_prompt_tokens + max_budget)) {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "actual n_ctx %u < prompt_tokens %d + max_decode_budget %d",
            actual_n_ctx, n_prompt_tokens, max_budget);
        emit_fail(STEP, buf); cleanup(); return 1;
    }

    // ---- Run the multi-seq engine, optionally repeated. ----------------
    const int32_t batch_capacity =
        std::max<int32_t>(n_seqs * n_prompt_tokens, n_seqs);

    std::vector<seq_state> first_seqs;

    // Suppress per-seq lines for large N; always emit budget-class summary.
    constexpr int32_t k_per_seq_print_threshold = 8;
    const bool verbose_per_seq = (n_seqs <= k_per_seq_print_threshold);

    for (int32_t r = 0; r < args.repeat; r++) {
        multiseq_result mr = run_multiseq(
            ctx, vocab, n_vocab, prompt_tokens,
            budgets, batch_capacity);
        if (!mr.ok) {
            char buf[320];
            std::snprintf(buf, sizeof(buf), "iteration %d: %s",
                          r, mr.error.c_str());
            emit_fail(STEP, buf); cleanup(); return 1;
        }

        if (verbose_per_seq) {
            for (int32_t s = 0; s < n_seqs; s++) {
                const seq_state & seq = mr.seqs[s];
                fprintf(stdout,
                    "iter[%d] seq=%d budget=%d n_decoded=%d done_iter=%d "
                    "pos_max_at_clear=%d hash=0x%016llx\n",
                    r, seq.seq_id, seq.decode_budget, seq.n_decoded,
                    seq.done_iter, seq.pos_max_at_clear,
                    static_cast<unsigned long long>(seq.finalize_hash()));
                if (r == 0) {
                    char label[64];
                    std::snprintf(label, sizeof(label),
                        "iter[%d] seq=%d tokens:", r, seq.seq_id);
                    print_token_ids(label, seq.generated_tokens);
                }
            }
        }

        // Per-budget-class summary (always printed). For each unique
        // budget value, report count, unique hash count, the hash
        // (asserts unique below), and observed vs expected
        // done_iter / pos_max_at_clear sets.
        std::vector<int32_t> uniq_budgets;
        for (int32_t s = 0; s < n_seqs; s++) {
            const int32_t b = mr.seqs[s].decode_budget;
            bool seen = false;
            for (int32_t u : uniq_budgets) if (u == b) { seen = true; break; }
            if (!seen) uniq_budgets.push_back(b);
        }
        std::sort(uniq_budgets.begin(), uniq_budgets.end());

        for (int32_t b : uniq_budgets) {
            int32_t  count            = 0;
            uint64_t any_hash         = 0;
            std::vector<uint64_t>  hashes;
            std::vector<int32_t>   done_iters;
            std::vector<llama_pos> pos_maxes;
            for (int32_t s = 0; s < n_seqs; s++) {
                const seq_state & seq = mr.seqs[s];
                if (seq.decode_budget != b) continue;
                count++;
                const uint64_t h = seq.finalize_hash();
                any_hash = h;
                bool h_seen = false;
                for (uint64_t hh : hashes) if (hh == h) { h_seen = true; break; }
                if (!h_seen) hashes.push_back(h);
                bool d_seen = false;
                for (int32_t dd : done_iters) if (dd == seq.done_iter) { d_seen = true; break; }
                if (!d_seen) done_iters.push_back(seq.done_iter);
                bool p_seen = false;
                for (llama_pos pp : pos_maxes) if (pp == seq.pos_max_at_clear) { p_seen = true; break; }
                if (!p_seen) pos_maxes.push_back(seq.pos_max_at_clear);
            }
            std::sort(done_iters.begin(), done_iters.end());
            std::sort(pos_maxes.begin(),  pos_maxes.end());
            const int32_t expected_done_iter = b - 1;
            const int32_t expected_pos_max   = n_prompt_tokens + b - 2;

            fprintf(stdout,
                "iter[%d] budget=%d count=%d unique_hashes=%zu hash=0x%016llx "
                "expected_done_iter=%d observed_done_iter={",
                r, b, count, hashes.size(),
                static_cast<unsigned long long>(any_hash),
                expected_done_iter);
            for (size_t i = 0; i < done_iters.size(); i++) {
                fprintf(stdout, "%s%d",
                        i == 0 ? "" : ",", done_iters[i]);
            }
            fprintf(stdout,
                "} expected_pos_max=%d observed_pos_max={",
                expected_pos_max);
            for (size_t i = 0; i < pos_maxes.size(); i++) {
                fprintf(stdout, "%s%d",
                        i == 0 ? "" : ",", pos_maxes[i]);
            }
            fprintf(stdout, "}\n");

            // Hard gates per class.
            if (hashes.size() != 1) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "budget %d: %zu distinct hashes among %d seqs "
                    "(expected 1)", b, hashes.size(), count);
                emit_fail(STEP, buf); cleanup(); return 1;
            }
            if (done_iters.size() != 1
                || done_iters[0] != expected_done_iter) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "budget %d: done_iter set has %zu values, expected "
                    "{%d}", b, done_iters.size(), expected_done_iter);
                emit_fail(STEP, buf); cleanup(); return 1;
            }
            if (pos_maxes.size() != 1
                || pos_maxes[0] != expected_pos_max) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "budget %d: pos_max_at_clear set has %zu values, "
                    "expected {%d}", b, pos_maxes.size(), expected_pos_max);
                emit_fail(STEP, buf); cleanup(); return 1;
            }
        }

        if (r == 0) {
            first_seqs = mr.seqs;
        } else {
            // Determinism check across repeats.
            for (int32_t s = 0; s < n_seqs; s++) {
                const seq_state & a0 = first_seqs[s];
                const seq_state & a  = mr.seqs[s];
                if (a.n_decoded != a0.n_decoded
                    || a.generated_tokens != a0.generated_tokens
                    || a.finalize_hash()  != a0.finalize_hash()
                    || a.done_iter        != a0.done_iter) {
                    char buf[320];
                    std::snprintf(buf, sizeof(buf),
                        "non-deterministic: iter %d seq %d hash 0x%016llx "
                        "differs from iter 0 hash 0x%016llx",
                        r, s,
                        static_cast<unsigned long long>(a.finalize_hash()),
                        static_cast<unsigned long long>(a0.finalize_hash()));
                    emit_fail(STEP, buf); cleanup(); return 1;
                }
            }
        }
        fflush(stdout);
    }

    // ---- Per-seq budget gate: every seq must hit its own budget. -----
    for (int32_t s = 0; s < n_seqs; s++) {
        const seq_state & seq = first_seqs[s];
        if (seq.n_decoded != seq.decode_budget) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "seq %d: n_decoded %d != decode_budget %d (early eog?)",
                seq.seq_id, seq.n_decoded, seq.decode_budget);
            emit_fail(STEP, buf); cleanup(); return 1;
        }
    }

    // ---- KV-clear position-convention gate. ---------------------------
    // pos_max_at_clear must equal P + budget - 2 for every seq.
    for (int32_t s = 0; s < n_seqs; s++) {
        const seq_state & seq = first_seqs[s];
        const llama_pos expected = n_prompt_tokens + seq.decode_budget - 2;
        if (seq.pos_max_at_clear != expected) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "seq %d unexpected pos_max_at_clear=%d "
                "(expected P+D-2 = %d)",
                seq.seq_id, seq.pos_max_at_clear, expected);
            emit_fail(STEP, buf); cleanup(); return 1;
        }
    }

    // ---- Pairwise prefix equality. ------------------------------------
    // For every pair (i, j) the first min(budget_i, budget_j) tokens
    // must be identical (same prompt + greedy argmax → same prefix).
    for (int32_t i = 0; i < n_seqs; i++) {
        for (int32_t j = i + 1; j < n_seqs; j++) {
            const auto & ti = first_seqs[i].generated_tokens;
            const auto & tj = first_seqs[j].generated_tokens;
            const int32_t k =
                std::min<int32_t>(budgets[i], budgets[j]);
            for (int32_t p = 0; p < k; p++) {
                if (ti[p] != tj[p]) {
                    char buf[256];
                    std::snprintf(buf, sizeof(buf),
                        "prefix mismatch: seq %d vs seq %d at index %d "
                        "(%d != %d), first %d tokens should be equal",
                        i, j, p, ti[p], tj[p], k);
                    emit_fail(STEP, buf); cleanup(); return 1;
                }
            }
        }
    }
    if (n_seqs >= 2) {
        fprintf(stdout,
            "prefix equality: pairwise first-min(budget_i,budget_j) "
            "tokens identical across all %d seqs\n", n_seqs);
    }

    // ---- Finish-order monotonicity (mixed budgets only). --------------
    // For every pair (i,j): if budget_i < budget_j then
    //   done_iter_i < done_iter_j; if equal, equal.
    for (int32_t i = 0; i < n_seqs; i++) {
        for (int32_t j = 0; j < n_seqs; j++) {
            if (i == j) continue;
            const seq_state & si = first_seqs[i];
            const seq_state & sj = first_seqs[j];
            if (si.decode_budget < sj.decode_budget
                && !(si.done_iter < sj.done_iter)) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "ordering: seq %d budget %d done_iter=%d not earlier "
                    "than seq %d budget %d done_iter=%d",
                    si.seq_id, si.decode_budget, si.done_iter,
                    sj.seq_id, sj.decode_budget, sj.done_iter);
                emit_fail(STEP, buf); cleanup(); return 1;
            }
            if (si.decode_budget == sj.decode_budget
                && si.done_iter != sj.done_iter) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "ordering: equal budgets but unequal done_iter "
                    "(seq %d=%d, seq %d=%d)",
                    si.seq_id, si.done_iter, sj.seq_id, sj.done_iter);
                emit_fail(STEP, buf); cleanup(); return 1;
            }
        }
    }

    // ---- Canonical-hash check (for any seq with budget 8 or 16). -----
    constexpr uint64_t k_canonical_budget_8  = 0x0619d4d1900c2365ull;
    constexpr uint64_t k_canonical_budget_16 = 0x833045f1e2ebf49full;
    for (int32_t s = 0; s < n_seqs; s++) {
        const seq_state & seq = first_seqs[s];
        uint64_t expected = 0;
        bool     known    = false;
        if (seq.decode_budget == 8) {
            expected = k_canonical_budget_8; known = true;
        } else if (seq.decode_budget == 16) {
            expected = k_canonical_budget_16; known = true;
        }
        if (known) {
            const uint64_t observed = seq.finalize_hash();
            if (verbose_per_seq) {
                fprintf(stdout, "seq=%d budget=%d expected=0x%016llx "
                                "observed=0x%016llx\n",
                        seq.seq_id, seq.decode_budget,
                        static_cast<unsigned long long>(expected),
                        static_cast<unsigned long long>(observed));
            }
            if (observed != expected) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "seq %d hash mismatch: observed 0x%016llx != "
                    "expected 0x%016llx (budget %d, prompt \"%s\")",
                    seq.seq_id,
                    static_cast<unsigned long long>(observed),
                    static_cast<unsigned long long>(expected),
                    seq.decode_budget, args.prompt.c_str());
                emit_fail(STEP, buf); cleanup(); return 1;
            }
        }
    }

    // ---- Final residual KV check: every seq should be empty. ----------
    auto * mem = llama_get_memory(ctx);
    for (int32_t s = 0; s < n_seqs; s++) {
        const llama_pos pmin = llama_memory_seq_pos_min(mem, s);
        const llama_pos pmax = llama_memory_seq_pos_max(mem, s);
        if (verbose_per_seq) {
            fprintf(stdout, "seq=%d residual pos_min=%d pos_max=%d\n",
                    s, pmin, pmax);
        }
        if (pmin != -1 || pmax != -1) {
            char buf[200];
            std::snprintf(buf, sizeof(buf),
                "seq %d residual KV not empty: pos_min=%d pos_max=%d",
                s, pmin, pmax);
            emit_fail(STEP, buf); cleanup(); return 1;
        }
    }
    if (!verbose_per_seq) {
        fprintf(stdout,
            "residual KV: all %d seqs cleared (pos_min=-1, pos_max=-1)\n",
            n_seqs);
    }

    cleanup();
    emit_pass(STEP);
    return 0;
}
