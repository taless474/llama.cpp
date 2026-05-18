// Scripted submitter for the HPX continuous-batch gate.
//
// One `submitter_release_block` per distinct release_iter K. Arrivals
// are pushed to engine::submit_request() in scripted (intra-block)
// order; the engine owns the per-request promise and returns the
// future via submit_handle.result, which the submitter pushes into
// the main-owned `external_futs` vector. Main reads `external_futs`
// only after `submitter_fut.get()` joins this task — single-producer
// / sequential-consumer with a join boundary, so no mutex.
//
// `run_scripted_submitter` is invoked as exactly one HPX task per
// repeat. HARD RULE: this body must not call any llama_* API. It only:
//   - awaits release_future for each block (in ascending release_iter
//     order),
//   - constructs submit_request values and calls
//     engine::submit_request(), pushing each handle.result onto
//     external_futs,
//   - sets ack_promise after pushing the K-block.
// The corresponding futures live in main; the submitter never reads
// them after pushing.

#pragma once

#include "types.h"

#include <hpx/hpx.hpp>

#include <cstdint>
#include <vector>

class engine;

struct submitter_release_block {
    int32_t                       release_iter = -1;
    external_release_handle       handle;
    std::vector<scripted_arrival> arrivals;
    // M2c: per-arrival promises moved into the engine via
    // submit_request(); the submitter pushes the returned futures
    // into the main-owned external_futs vector instead.
};

void run_scripted_submitter(
    engine & eng,
    std::vector<submitter_release_block>       blocks,
    std::vector<hpx::future<request_result>> & external_futs);
