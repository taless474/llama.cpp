#!/usr/bin/env bash
# run_sched_sweep.sh
#
# Scheduler sweep for bench_run_job_overhead.
#
# For each HPX queuing policy in SCHEDULERS, runs N_LAUNCHES independent
# process launches.  Launches that crash (non-zero exit) are recorded as
# SKIPPED so the result file stays complete.
#
# Usage (from the repo root or hpx-bench/):
#   bash hpx-bench/run_sched_sweep.sh
#
# Tuning:
#   BINARY      — path to the benchmark binary
#   N_THREADS   — --hpx:threads value (controls the HPX thread pool size)
#   N_LAUNCHES  — independent process launches per scheduler
#   SCHEDULERS  — space-separated list of --hpx:queuing values to sweep

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BINARY="${BINARY:-$REPO_ROOT/build-hpx-cpu/bin/bench_run_job_overhead}"
N_THREADS="${N_THREADS:-4}"
N_LAUNCHES="${N_LAUNCHES:-3}"
SCHEDULERS="${SCHEDULERS:-local static static-priority local-priority shared-priority}"

TIMESTAMP="$(date +%Y-%m-%d-%H%M%S)"
OUTDIR="$SCRIPT_DIR/results/${TIMESTAMP}-sched-sweep"
mkdir -p "$OUTDIR"
OUTFILE="$OUTDIR/sweep.txt"

log() { printf '%s\n' "$*" | tee -a "$OUTFILE"; }

log "# bench_run_job_overhead scheduler sweep"
log "# date:       $TIMESTAMP"
log "# binary:     $BINARY"
log "# n_threads:  $N_THREADS"
log "# n_launches: $N_LAUNCHES"
log "# schedulers: $SCHEDULERS"
log ""

for sched in $SCHEDULERS; do
    log "###############################################################"
    log "### scheduler: $sched"
    log "###############################################################"
    log ""

    for run in $(seq 1 "$N_LAUNCHES"); do
        log "--- launch $run / $N_LAUNCHES  (sched=$sched) ---"
        log ""

        # Run the benchmark; tee stdout+stderr into the result file.
        # A non-zero exit (crash, HPX error) is caught and recorded.
        set +e
        "$BINARY" \
            --hpx:threads="$N_THREADS" \
            --hpx:queuing="$sched" \
            --sched-label="$sched" \
            2>&1 | tee -a "$OUTFILE"
        rc="${PIPESTATUS[0]}"
        set -e

        if [ "$rc" -ne 0 ]; then
            log ""
            log "!!! SKIPPED: binary exited with status $rc for sched=$sched launch=$run"
            log ""
            break  # no point repeating a scheduler that crashes
        fi

        log ""
    done

    log ""
done

log "# sweep complete: $OUTFILE"
printf '\nResults saved to: %s\n' "$OUTFILE"
