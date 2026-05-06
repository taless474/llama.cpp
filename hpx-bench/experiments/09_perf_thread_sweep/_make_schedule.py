"""
Phase 1 helper — generate the deterministic schedule for the C_2x4
n_threads sensitivity sweep. Does not run anything. Does not invoke
the binary.

Writes:
  local/baselines/perf_thread_sweep/_schedule.json
  local/baselines/perf_thread_sweep/schedule_summary.txt

Schedule shape:
  3 thread settings × 2 layers × 2 backends × 11 trials = 132 entries

Fixed cell: C_2x4 (n_contexts=2, n_concurrent=4, n_requests=12).
Sweep variable: n_threads ∈ {1, 2, 4}.

Determinism:
  Per (n_threads, layer) pair, a fresh random.Random(1234) shuffles a
  22-entry list of (backend, trial_index) where backend in {std, hpx}
  and trial_index in 0..10. Outer iteration order is fixed:
    n_threads: 1 → 2 → 4
    layers:    correctness_trace_on → timing_trace_off
  Rerunning this script must produce an identical _schedule.json.

measured_for_timing rules:
  correctness_trace_on:                     False
  timing_trace_off, trial_index == 0:       False
  timing_trace_off, trial_index in 1..10:   True

trace_enabled rules:
  correctness_trace_on: True   (LLAMA_SERVING_BENCH_HPX_TRACE=1)
  timing_trace_off:     False  (env var unset)
"""
import json
import random
from pathlib import Path

SEED = 1234

ROOT = Path(
    "/Users/Ashk/Desktop/HPX/llama-hpx/local/baselines/perf_thread_sweep"
)

# Cell is fixed at C_2x4 across the entire sweep.
CELL = {
    "cell_id":              "C_2x4",
    "n_contexts":           2,
    "n_concurrent":         4,
    "n_requests":           12,
    "expected_os_threads":  2,
    "expected_pool_size":   2,
    "expected_trace_lines": 27,
}

THREAD_SETTINGS = [1, 2, 4]
LAYERS   = ["correctness_trace_on", "timing_trace_off"]
BACKENDS = ["std", "hpx"]
TRIALS_PER_BACKEND = 11  # trial_index 0..10 inclusive

LAYER_DIR = {
    "correctness_trace_on": "correctness",
    "timing_trace_off":     "timing",
}


def trace_enabled_for(layer: str) -> bool:
    return layer == "correctness_trace_on"


def measured_for_timing(layer: str, trial_index: int) -> bool:
    if layer == "correctness_trace_on":
        return False
    return trial_index != 0  # timing_trace_off: drop trial 0 from aggregation


def build_schedule():
    entries = []
    global_index = 0
    for n_threads in THREAD_SETTINGS:
        for layer in LAYERS:
            rng = random.Random(SEED)
            pairs = []
            for backend in BACKENDS:
                for trial_index in range(TRIALS_PER_BACKEND):
                    pairs.append((backend, trial_index))
            rng.shuffle(pairs)
            for backend, trial_index in pairs:
                entry = {
                    "global_index":         global_index,
                    "cell_id":              CELL["cell_id"],
                    "n_threads":            n_threads,
                    "layer":                layer,
                    "backend":              backend,
                    "trial_index":          trial_index,
                    "n_contexts":           CELL["n_contexts"],
                    "n_concurrent":         CELL["n_concurrent"],
                    "n_requests":           CELL["n_requests"],
                    "max_tokens":           16,
                    "ctx_size":             2048,
                    "batch_size":           512,
                    "seed_base":            SEED,
                    "trace_enabled":        trace_enabled_for(layer),
                    "measured_for_timing":  measured_for_timing(layer, trial_index),
                    "expected_os_threads":  CELL["expected_os_threads"],
                    "expected_pool_size":   CELL["expected_pool_size"],
                    "expected_trace_lines": CELL["expected_trace_lines"],
                    "output_dir": (
                        f"thread_settings/n{n_threads}/"
                        f"{LAYER_DIR[layer]}/{backend}/"
                        f"trial_{trial_index:02d}"
                    ),
                }
                entries.append(entry)
                global_index += 1
    return entries


def write_schedule(entries, path: Path):
    payload = {
        "schedule_seed":      SEED,
        "cell_id":            CELL["cell_id"],
        "thread_settings":    THREAD_SETTINGS,
        "layers":             LAYERS,
        "backends":           BACKENDS,
        "trials_per_backend": TRIALS_PER_BACKEND,
        "trial_index_range":  [0, TRIALS_PER_BACKEND - 1],
        "total_entries":      len(entries),
        "schedule":           entries,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_summary(entries, path: Path):
    counts = {}
    measured_counts = {}
    for e in entries:
        key = (e["n_threads"], e["layer"], e["backend"])
        counts[key] = counts.get(key, 0) + 1
        if e["measured_for_timing"]:
            measured_counts[key] = measured_counts.get(key, 0) + 1

    lines = []
    lines.append("C_2x4 n_threads sensitivity sweep — schedule summary")
    lines.append("")
    lines.append(f"seed:                 {SEED}")
    lines.append(f"cell_id:              {CELL['cell_id']}")
    lines.append(f"thread_settings:      {THREAD_SETTINGS}")
    lines.append(f"layers:               {LAYERS}")
    lines.append(f"backends:             {BACKENDS}")
    lines.append(
        f"trials per backend:   {TRIALS_PER_BACKEND} "
        f"(indices 0..{TRIALS_PER_BACKEND - 1} inclusive)"
    )
    lines.append(f"total entries:        {len(entries)}")
    lines.append(
        "expected total:       3 × 2 × 2 × 11 = 132"
    )
    lines.append("")
    lines.append("Per (n_threads × layer × backend) trial counts:")
    lines.append(
        "  n_thr  layer                  bk   trials  measured_for_timing"
    )
    for n_threads in THREAD_SETTINGS:
        for layer in LAYERS:
            for backend in BACKENDS:
                key = (n_threads, layer, backend)
                lines.append(
                    f"  {n_threads:5d}  {layer:22s} {backend:3s}  "
                    f"{counts.get(key, 0):6d}  {measured_counts.get(key, 0):6d}"
                )
    lines.append("")
    lines.append("First 8 entries:")
    lines.append(
        "  gidx  n_thr  layer                  bk   trial  measured  trace"
    )
    for e in entries[:8]:
        lines.append(
            f"  {e['global_index']:4d}  {e['n_threads']:5d}  "
            f"{e['layer']:22s} {e['backend']:3s}  "
            f"{e['trial_index']:5d}  "
            f"{str(e['measured_for_timing']):8s}  "
            f"{str(e['trace_enabled'])}"
        )
    lines.append("")
    lines.append("Last 4 entries:")
    lines.append(
        "  gidx  n_thr  layer                  bk   trial  measured  trace"
    )
    for e in entries[-4:]:
        lines.append(
            f"  {e['global_index']:4d}  {e['n_threads']:5d}  "
            f"{e['layer']:22s} {e['backend']:3s}  "
            f"{e['trial_index']:5d}  "
            f"{str(e['measured_for_timing']):8s}  "
            f"{str(e['trace_enabled'])}"
        )
    lines.append("")
    lines.append(
        "Reproducibility: each (n_threads × layer) pair re-seeds random.Random(1234)"
    )
    lines.append(
        "and shuffles a fresh list of 22 (backend, trial_index) pairs. The seed"
    )
    lines.append(
        "is the only source of randomness; rerunning this script must produce an"
    )
    lines.append("identical _schedule.json byte-for-byte.")
    path.write_text("\n".join(lines) + "\n")


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    entries = build_schedule()
    write_schedule(entries, ROOT / "_schedule.json")
    write_summary(entries, ROOT / "schedule_summary.txt")
    print(f"wrote {ROOT / '_schedule.json'}")
    print(f"wrote {ROOT / 'schedule_summary.txt'}")
    print(f"total entries: {len(entries)}  (expected 132)")


if __name__ == "__main__":
    main()
