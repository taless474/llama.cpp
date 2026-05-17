// CLI argument surface for the HPX continuous-batch gate.
//
// `cli_args` holds every flag the binary accepts. `print_usage`
// prints the help text. `parse_args` consumes argv, populates
// `cli_args`, and validates the partition (n_active + n_waiting +
// n_external_arrivals <= n_seqs). Both functions preserve the
// pre-extraction behavior byte-for-byte; the parser-local helpers
// (`parse_int`, `parse_csv_int_list`) live in cli.cpp's anonymous
// namespace.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct cli_args {
    std::string          model_path;
    std::string          prompt        = "Hello, my name is";
    int32_t              ctx_size      = 32768;
    int32_t              n_seq_max     = 99;
    int32_t              n_batch       = 1024;
    int32_t              n_threads     = 2;
    int32_t              n_seqs        = 99;
    int32_t              hpx_os_threads = 1;
    int32_t              repeat        = 1;
    std::vector<int32_t> decode_budget_mix = {8, 64, 256};

    // Cancel Slice 1: plan is parsed and printed, asserted against the
    // round-robin budget mapping, but NOT propagated into seq_state and
    // NOT observed by the engine. Defaults match the design's smoke
    // shape: cancel 3 budget-64 seqs (1,4,7) and 3 budget-256 seqs
    // (2,5,8) after 16 decoded tokens each.
    std::vector<int32_t> cancel_plan   = {1, 4, 7, 2, 5, 8};
    int32_t              cancel_after  = 16;

    // Live Admission Slice 2: queue-only data model.
    //   --n-active        size of the bound active set; resolved post-
    //                     parse to args.n_seqs when 0 (= "use n_seqs").
    //   --n-waiting       number of waiting requests queued at
    //                     construction time; engine never consumes.
    //   --waiting-budget  uniform decode_budget for every waiting
    //                     request (only used by Slice 3+).
    int32_t              n_active        = 0;
    int32_t              n_waiting       = 0;
    int32_t              waiting_budget  = 64;

    // Live Admission Slice 5: enable completion-freed-slot admission.
    // OFF preserves Slice 4 semantics (engine never touches the
    // completion pool). ON pushes naturally completed slots onto
    // free_due_to_completion_ under a demand-gate (push only while the
    // waiting queue still has unadmitted entries) and the admission
    // loop drains the cancel queue first, then the completion queue.
    bool                 reuse_completed = false;

    // Live Admission Slice 6: async external arrivals.
    //   --n-external-arrivals    number of scripted arrivals the
    //                            submitter HPX task will push at the
    //                            release barrier. 0 = inactive (no
    //                            submitter spawned, no release/ack
    //                            barrier installed; the run is byte-
    //                            equivalent to Slice 3..5 outputs).
    //   --external-arrival-budget uniform decode budget for every
    //                            external arrival.
    //   --external-release-iter  decode iter K at which the engine
    //                            fires the release promise; the
    //                            submitter pushes its K-block then
    //                            acks. Drain happens at top of iter
    //                            K+1, admission at iter K+1+1 (cancel-
    //                            freed path) for the smoke shape.
    int32_t              n_external_arrivals     = 0;
    int32_t              external_arrival_budget = 64;
    int32_t              external_release_iter   = 0;

    // Streaming Slice 8: enable HPX-native per-request token streaming.
    // OFF (default) preserves Slice 7 semantics; engine never allocates
    // stream promises and emits no token_stream_* events / counters.
    bool                 stream_all              = false;
};

void print_usage(const char * argv0);

bool parse_args(int argc, char ** argv, cli_args & args);
