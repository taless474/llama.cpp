#include "submitter.h"

#include "engine.h"
#include "types.h"

#include <algorithm>
#include <exception>
#include <stdexcept>
#include <string>
#include <utility>

void run_scripted_submitter(
    engine & eng,
    std::vector<submitter_release_block>       blocks,
    std::vector<hpx::future<request_result>> & external_futs)
{
    std::sort(blocks.begin(), blocks.end(),
              [](const submitter_release_block & a,
                 const submitter_release_block & b) {
                  return a.release_iter < b.release_iter;
              });
    for (auto & blk : blocks) {
        // Suspend until the engine fires the release promise at end of
        // iter K. If the engine errored before the barrier, the per-
        // arrival promises for this block have not been created yet
        // (M2c: engine owns them and only allocates them inside
        // submit_request, which we never reach in this branch).
        // Best-effort ack so the engine does not hang on the ack
        // future if it managed to set release before failing.
        try {
            blk.handle.release_future.get();
        } catch (...) {
            try { blk.handle.ack_promise.set_value(); } catch (...) {}
            throw;
        }

        for (size_t i = 0; i < blk.arrivals.size(); i++) {
            // M2c: route through the new public submit API. The
            // engine constructs the matching arrival_msg with a
            // fresh engine-owned hpx::promise and pushes via the
            // same inbox path the legacy submitter used. want_stream
            // is false here because the gate's per-seq stream
            // channels are still allocated by admit_one when
            // engine_options.stream_all is on, and external-arrival
            // receivers continue to flow through
            // take_admitted_stream_handoffs() to preserve the
            // existing trace counts and stdout shape byte-for-byte.
            // submit_request streaming will be wired in a later
            // stage along with a smoke that drives it.
            submit_request req;
            req.request_id    = blk.arrivals[i].request_id;
            req.decode_budget = blk.arrivals[i].decode_budget;
            req.prompt_tokens =
                std::move(blk.arrivals[i].prompt_tokens);
            req.want_stream   = false;
            submit_handle h   = eng.submit_request(std::move(req));
            // h.stream is std::nullopt by construction here.
            external_futs.push_back(std::move(h.result));
        }

        try {
            blk.handle.ack_promise.set_value();
        } catch (...) {
            // already satisfied — engine's get() will simply observe
            // ready; ignore.
        }
    }
}
