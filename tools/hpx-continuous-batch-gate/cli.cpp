#include "cli.h"

#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

bool parse_int(const char * s, int32_t & out) {
    if (s == nullptr || *s == '\0') return false;
    char * end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (end == s || *end != '\0') return false;
    if (v < INT32_MIN || v > INT32_MAX) return false;
    out = static_cast<int32_t>(v);
    return true;
}

bool parse_csv_int_list(const char * s, std::vector<int32_t> & out,
                        int32_t min_v = 1) {
    out.clear();
    if (s == nullptr || *s == '\0') return false;
    const std::string str = s;
    size_t i = 0;
    while (i <= str.size()) {
        size_t j = str.find(',', i);
        if (j == std::string::npos) j = str.size();
        const std::string tok = str.substr(i, j - i);
        if (tok.empty()) return false;
        int32_t v = 0;
        if (!parse_int(tok.c_str(), v)) return false;
        if (v < min_v) return false;
        out.push_back(v);
        if (j == str.size()) break;
        i = j + 1;
    }
    return !out.empty();
}

}  // namespace

void print_usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s --model <path> [options]\n"
        "  --model <path>            (required) path to .gguf model file\n"
        "  --prompt <string>         default: \"Hello, my name is\"\n"
        "  --ctx-size <int>          default: 32768\n"
        "  --n-seq-max <int>         default: 99\n"
        "  --n-batch <int>           default: 1024\n"
        "  --n-threads <int>         default: 2  (libllama compute threads)\n"
        "  --n-seqs <int>            default: 99\n"
        "  --decode-budget-mix <csv> default: \"8,64,256\"\n"
        "  --hpx-os-threads <int>    default: 1  (HPX orchestration threads)\n"
        "  --repeat <int>            default: 1  (engine task per repeat)\n"
        "  --cancel-plan <csv>       default: \"1,4,7,2,5,8\"\n"
        "                            seq_ids (>=0) cancelled at iteration\n"
        "                            boundaries once they reach\n"
        "                            --cancel-after decoded tokens. Use\n"
        "                            \"none\" for an explicit empty plan.\n"
        "  --cancel-after <int>      default: 16\n"
        "                            (decoded-token threshold for the plan)\n"
        "  --n-active <int>          default: n_seqs (when 0)\n"
        "                            size of the bound active set; must\n"
        "                            satisfy 1 <= n_active <= n_seqs and\n"
        "                            n_active + n_waiting <= n_seqs.\n"
        "  --n-waiting <int>         default: 0\n"
        "                            number of waiting requests queued at\n"
        "                            construction time. Slice 2: queue-\n"
        "                            only, the engine never consumes.\n"
        "  --waiting-budget <int>    default: 64\n"
        "                            uniform decode budget for every\n"
        "                            waiting request.\n"
        "  --reuse-completed         default: OFF\n"
        "                            Slice 5: enable completion-freed-\n"
        "                            slot admission. Naturally completed\n"
        "                            slots are pushed to a demand-gated\n"
        "                            pool (only while the waiting queue\n"
        "                            has unadmitted entries) and the\n"
        "                            admission loop drains the cancel\n"
        "                            queue first, then the completion\n"
        "                            queue. Off preserves Slice 4\n"
        "                            semantics (label may still advance).\n"
        "  --n-external-arrivals <int>      default: 0\n"
        "                            Slice 6: number of async external\n"
        "                            arrivals the scripted submitter HPX\n"
        "                            task will push under the release+ack\n"
        "                            barrier. 0 disables the path entirely\n"
        "                            (no submitter spawned).\n"
        "  --external-arrival-budget <int>  default: 64\n"
        "                            Slice 6: uniform decode budget for\n"
        "                            every external arrival.\n"
        "  --external-release-iter <int>    default: 0\n"
        "                            Slice 6: decode iter K at which the\n"
        "                            engine fires the release promise.\n"
        "                            Submitter pushes at K, drain occurs\n"
        "                            at top of K+1.\n"
        "  --stream-all              default: OFF\n"
        "                            Slice 8: enable HPX-native per-\n"
        "                            request token streaming. Each bound\n"
        "                            active seq gets a shared_future head\n"
        "                            published by the engine task; main\n"
        "                            drains the chain after engine_fut.\n"
        "                            OFF preserves Slice 7 semantics.\n",
        argv0);
}

bool parse_args(int argc, char ** argv, cli_args & args) {
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto need = [&](const char * flag) -> const char * {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: %s requires a value\n", flag);
                return nullptr;
            }
            return argv[++i];
        };
        if (a == "--model") {
            const char * v = need("--model");
            if (!v) return false;
            args.model_path = v;
        } else if (a == "--prompt") {
            const char * v = need("--prompt");
            if (!v) return false;
            args.prompt = v;
        } else if (a == "--ctx-size") {
            const char * v = need("--ctx-size");
            if (!v || !parse_int(v, args.ctx_size)) return false;
        } else if (a == "--n-seq-max") {
            const char * v = need("--n-seq-max");
            if (!v || !parse_int(v, args.n_seq_max)) return false;
        } else if (a == "--n-batch") {
            const char * v = need("--n-batch");
            if (!v || !parse_int(v, args.n_batch)) return false;
        } else if (a == "--n-threads") {
            const char * v = need("--n-threads");
            if (!v || !parse_int(v, args.n_threads)) return false;
        } else if (a == "--n-seqs") {
            const char * v = need("--n-seqs");
            if (!v || !parse_int(v, args.n_seqs)) return false;
            if (args.n_seqs <= 0) return false;
        } else if (a == "--decode-budget-mix") {
            const char * v = need("--decode-budget-mix");
            if (!v || !parse_csv_int_list(v, args.decode_budget_mix)) {
                fprintf(stderr,
                    "error: --decode-budget-mix must be a non-empty CSV "
                    "of positive integers\n");
                return false;
            }
        } else if (a == "--hpx-os-threads") {
            const char * v = need("--hpx-os-threads");
            if (!v || !parse_int(v, args.hpx_os_threads)) return false;
            if (args.hpx_os_threads <= 0) return false;
        } else if (a == "--repeat") {
            const char * v = need("--repeat");
            if (!v || !parse_int(v, args.repeat)) return false;
            if (args.repeat <= 0) return false;
        } else if (a == "--cancel-plan") {
            const char * v = need("--cancel-plan");
            if (!v) return false;
            // Cancel Slice 2: --cancel-plan now accepts seq_id 0
            // (parse_csv_int_list takes min_v=0). The literal "none"
            // is an explicit empty-plan sentinel.
            if (std::strcmp(v, "none") == 0) {
                args.cancel_plan.clear();
            } else if (!parse_csv_int_list(v, args.cancel_plan,
                                           /*min_v=*/0)) {
                fprintf(stderr,
                    "error: --cancel-plan must be a non-empty CSV "
                    "of non-negative seq_ids, or the literal \"none\"\n");
                return false;
            }
        } else if (a == "--cancel-after") {
            const char * v = need("--cancel-after");
            if (!v || !parse_int(v, args.cancel_after)) return false;
            if (args.cancel_after < 0) {
                fprintf(stderr,
                    "error: --cancel-after must be >= 0\n");
                return false;
            }
        } else if (a == "--n-active") {
            const char * v = need("--n-active");
            if (!v || !parse_int(v, args.n_active)) return false;
            if (args.n_active < 0) {
                fprintf(stderr, "error: --n-active must be >= 0\n");
                return false;
            }
        } else if (a == "--n-waiting") {
            const char * v = need("--n-waiting");
            if (!v || !parse_int(v, args.n_waiting)) return false;
            if (args.n_waiting < 0) {
                fprintf(stderr, "error: --n-waiting must be >= 0\n");
                return false;
            }
        } else if (a == "--waiting-budget") {
            const char * v = need("--waiting-budget");
            if (!v || !parse_int(v, args.waiting_budget)) return false;
            if (args.waiting_budget < 1) {
                fprintf(stderr,
                    "error: --waiting-budget must be >= 1\n");
                return false;
            }
        } else if (a == "--reuse-completed") {
            // Slice 5 boolean flag, no value.
            args.reuse_completed = true;
        } else if (a == "--n-external-arrivals") {
            const char * v = need("--n-external-arrivals");
            if (!v || !parse_int(v, args.n_external_arrivals)) return false;
            if (args.n_external_arrivals < 0) {
                fprintf(stderr,
                    "error: --n-external-arrivals must be >= 0\n");
                return false;
            }
        } else if (a == "--external-arrival-budget") {
            const char * v = need("--external-arrival-budget");
            if (!v || !parse_int(v, args.external_arrival_budget)) return false;
            if (args.external_arrival_budget < 1) {
                fprintf(stderr,
                    "error: --external-arrival-budget must be >= 1\n");
                return false;
            }
        } else if (a == "--external-release-iter") {
            const char * v = need("--external-release-iter");
            if (!v || !parse_int(v, args.external_release_iter)) return false;
            if (args.external_release_iter < 0) {
                fprintf(stderr,
                    "error: --external-release-iter must be >= 0\n");
                return false;
            }
        } else if (a == "--stream-all") {
            // Slice 8 boolean flag, no value.
            args.stream_all = true;
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
    if (args.n_seq_max < args.n_seqs) {
        args.n_seq_max = args.n_seqs;
    }
    // Live Admission Slice 2: resolve and validate the active/waiting
    // partition. Default mode (--n-active 0) inherits n_seqs so the
    // run is byte-identical to Slice 1 except for the relabel.
    if (args.n_active == 0) {
        args.n_active = args.n_seqs;
    }
    if (args.n_active < 1 || args.n_active > args.n_seqs) {
        fprintf(stderr,
            "error: --n-active=%d must satisfy 1 <= n_active <= "
            "n_seqs=%d\n", args.n_active, args.n_seqs);
        return false;
    }
    if (args.n_active + args.n_waiting > args.n_seqs) {
        fprintf(stderr,
            "error: n_active=%d + n_waiting=%d > n_seqs=%d\n",
            args.n_active, args.n_waiting, args.n_seqs);
        return false;
    }
    // Live Admission Slice 6: external arrivals occupy ids
    // [n_active + n_waiting, n_active + n_waiting + n_external_arrivals);
    // each will be bound to a freed slot already counted in n_seqs, so
    // we only require enough id headroom (request_id space). The seq_id
    // they end up on comes from the freed-slot pool already populated
    // by cancellation, not from new context slots.
    if (args.n_active + args.n_waiting + args.n_external_arrivals
        > args.n_seqs) {
        fprintf(stderr,
            "error: n_active=%d + n_waiting=%d + "
            "n_external_arrivals=%d > n_seqs=%d\n",
            args.n_active, args.n_waiting,
            args.n_external_arrivals, args.n_seqs);
        return false;
    }
    return true;
}
