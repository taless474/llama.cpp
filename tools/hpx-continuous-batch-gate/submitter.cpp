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
    std::vector<submitter_release_block> blocks)
{
    std::sort(blocks.begin(), blocks.end(),
              [](const submitter_release_block & a,
                 const submitter_release_block & b) {
                  return a.release_iter < b.release_iter;
              });
    for (auto & blk : blocks) {
        // Suspend until the engine fires the release promise at end of
        // iter K. If the engine errored before the barrier, mark every
        // arrival's promise broken via set_exception so main's wait_all
        // does not deadlock.
        try {
            blk.handle.release_future.get();
        } catch (const std::exception & e) {
            const std::string what = e.what();
            for (auto & p : blk.promises) {
                try {
                    p.set_exception(std::make_exception_ptr(
                        std::runtime_error(
                            "submitter: release_future threw: " + what)));
                } catch (...) {
                    // already satisfied
                }
            }
            // Best-effort ack so the engine does not hang on the
            // ack future if it managed to set release before failing.
            try { blk.handle.ack_promise.set_value(); } catch (...) {}
            throw;
        }

        for (size_t i = 0; i < blk.arrivals.size(); i++) {
            arrival_msg msg;
            msg.request_id    = blk.arrivals[i].request_id;
            msg.decode_budget = blk.arrivals[i].decode_budget;
            msg.promise       = std::move(blk.promises[i]);
            msg.src           = arrival_source::external;
            eng.submit(std::move(msg));
        }

        try {
            blk.handle.ack_promise.set_value();
        } catch (...) {
            // already satisfied — engine's get() will simply observe
            // ready; ignore.
        }
    }
}
