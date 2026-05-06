"""
Phase 1 helper — generate the deterministic schedule for the HPX-vs-std
performance matrix. Does not run anything. Does not invoke the binary.

Writes:
  local/baselines/perf_hpx_vs_std_matrix/_schedule.json
  local/baselines/perf_hpx_vs_std_matrix/schedule_summary.txt

Schedule shape:
  4 cells × 2 layers × 2 backends × 31 trials = 496 entries

Determinism:
  Per (cell, layer) pair, a fresh random.Random(1234) shuffles a 62-entry
  list of (backend, trial_index) where backend in {std, hpx} and
  trial_index in 0..30. The same seed (1234) is used for every (cell,
  layer); the schedules across pairs are independent because each pair
  re-seeds. Outer iteration order across pairs is fixed:
    cells:  A_1x1 -> B_2x2 -> C_2x4 -> D_4x4
    layers: correctness_trace_on -> timing_trace_off
  Rerunning this script must produce an identical _schedule.json.

measured_for_timing rules:
  correctness_trace_on:                    always False
  timing_trace_off, trial_index == 0:      False (correctness-checked but
                                                  excluded from timing
                                                  aggregation)
  timing_trace_off, trial_index in 1..30:  True

trace_enabled rules:
  correctness_trace_on:  True   (LLAMA_SERVING_BENCH_HPX_TRACE=1)
  timing_trace_off:      False  (env var unset)
"""
import json
import random
from pathlib import Path

SEED = 1234

ROOT = Path(
    "/Users/Ashk/Desktop/HPX/llama-hpx/local/baselines/perf_hpx_vs_std_matrix"
)

CELLS = [
    {
        "cell_id":              "A_1x1",
        "n_contexts":           1,
        "n_concurrent":         1,
        "n_requests":           6,
        "expected_os_threads":  1,
        "expected_pool_size":   1,
        "expected_trace_lines": 15,
    },
    {
        "cell_id":              "B_2x2",
        "n_contexts":           2,
        "n_concurrent":         2,
        "n_requests":           8,
        "expected_os_threads":  2,
        "expected_pool_size":   2,
        "expected_trace_lines": 19,
    },
    {
        "cell_id":              "C_2x4",
        "n_contexts":           2,
        "n_concurrent":         4,
        "n_requests":           12,
        "expected_os_threads":  2,
        "expected_pool_size":   2,
        "expected_trace_lines": 27,
    },
    {
        "cell_id":              "D_4x4",
        "n_contexts":           4,
        "n_concurrent":         4,
        "n_requests":           16,
        "expected_os_threads":  4,
        "expected_pool_size":   4,
        "expected_trace_lines": 35,
    },
]

LAYERS   = ["correctness_trace_on", "timing_trace_off"]
BACKENDS = ["std", "hpx"]
TRIALS_PER_BACKEND = 31  # trial_index 0..30 inclusive

# layer -> directory segment used inside output_dir; matches the
# README's planned `conditions/<cell>/correctness|timing/<backend>/...`
# tree so this helper aligns with the artifact layout described there.
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
    for cell in CELLS:
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
                    "cell_id":              cell["cell_id"],
                    "layer":                layer,
                    "backend":              backend,
                    "trial_index":          trial_index,
                    "n_contexts":           cell["n_contexts"],
                    "n_concurrent":         cell["n_concurrent"],
                    "n_requests":           cell["n_requests"],
                    "max_tokens":           16,
                    "ctx_size":             2048,
                    "batch_size":           512,
                    "n_threads":            4,
                    "seed_base":            SEED,
                    "trace_enabled":        trace_enabled_for(layer),
                    "measured_for_timing":  measured_for_timing(layer, trial_index),
                    "expected_os_threads":  cell["expected_os_threads"],
                    "expected_pool_size":   cell["expected_pool_size"],
                    "expected_trace_lines": cell["expected_trace_lines"],
                    "output_dir": (
                        f"conditions/{cell['cell_id']}/"
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
        "cells":              [c["cell_id"] for c in CELLS],
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
        key = (e["cell_id"], e["layer"], e["backend"])
        counts[key] = counts.get(key, 0) + 1
        if e["measured_for_timing"]:
            measured_counts[key] = measured_counts.get(key, 0) + 1

    lines = []
    lines.append("HPX-vs-std performance matrix — schedule summary")
    lines.append("")
    lines.append(f"seed:                 {SEED}")
    lines.append(f"cells:                {[c['cell_id'] for c in CELLS]}")
    lines.append(f"layers:               {LAYERS}")
    lines.append(f"backends:             {BACKENDS}")
    lines.append(
        f"trials per backend:   {TRIALS_PER_BACKEND} "
        f"(indices 0..{TRIALS_PER_BACKEND - 1} inclusive)"
    )
    lines.append(f"total entries:        {len(entries)}")
    lines.append(
        "expected total:       4 cells × 2 layers × 2 backends × 31 trials = 496"
    )
    lines.append("")
    lines.append("Per (cell × layer × backend) trial counts:")
    lines.append(
        "  cell    layer                  bk   trials  measured_for_timing"
    )
    for cell in CELLS:
        for layer in LAYERS:
            for backend in BACKENDS:
                key = (cell["cell_id"], layer, backend)
                lines.append(
                    f"  {cell['cell_id']:6s}  {layer:22s} {backend:3s}  "
                    f"{counts.get(key, 0):6d}  {measured_counts.get(key, 0):6d}"
                )
    lines.append("")
    lines.append("First 8 entries:")
    lines.append(
        "  gidx  cell    layer                  bk   trial  measured  trace"
    )
    for e in entries[:8]:
        lines.append(
            f"  {e['global_index']:4d}  {e['cell_id']:6s}  "
            f"{e['layer']:22s} {e['backend']:3s}  "
            f"{e['trial_index']:5d}  "
            f"{str(e['measured_for_timing']):8s}  "
            f"{str(e['trace_enabled'])}"
        )
    lines.append("")
    lines.append("Last 4 entries:")
    lines.append(
        "  gidx  cell    layer                  bk   trial  measured  trace"
    )
    for e in entries[-4:]:
        lines.append(
            f"  {e['global_index']:4d}  {e['cell_id']:6s}  "
            f"{e['layer']:22s} {e['backend']:3s}  "
            f"{e['trial_index']:5d}  "
            f"{str(e['measured_for_timing']):8s}  "
            f"{str(e['trace_enabled'])}"
        )
    lines.append("")
    lines.append(
        "Reproducibility: each (cell × layer) pair re-seeds random.Random(1234)"
    )
    lines.append(
        "and shuffles a fresh list of 62 (backend, trial_index) pairs. The seed"
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
    print(f"total entries: {len(entries)}  (expected 496)")


if __name__ == "__main__":
    main()
