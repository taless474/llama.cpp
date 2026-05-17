// Scripted submitter for the HPX continuous-batch gate.
//
// One `submitter_release_block` per distinct release_iter K. Per-arrival
// promises are pre-created in main and moved in here by value; main keeps
// the matching futures so it can wait_all on them after engine_fut.get().
// Arrivals are pushed to engine::submit() in scripted (intra-block) order.
//
// `run_scripted_submitter` is invoked as exactly one HPX task per
// repeat. HARD RULE: this body must not call any llama_* API. It only:
//   - awaits release_future for each block (in ascending release_iter
//     order),
//   - constructs arrival_msg messages and calls engine::submit(),
//   - sets ack_promise after pushing the K-block.
// The corresponding futures live in main; the submitter never touches
// them.

#pragma once

#include "types.h"

#include <hpx/hpx.hpp>

#include <cstdint>
#include <vector>

class engine;

struct submitter_release_block {
    int32_t                                   release_iter = -1;
    external_release_handle                   handle;
    std::vector<scripted_arrival>             arrivals;
    std::vector<hpx::promise<request_result>> promises;
};

void run_scripted_submitter(
    engine & eng,
    std::vector<submitter_release_block> blocks);
