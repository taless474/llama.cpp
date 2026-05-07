"""
Generate the deterministic schedule for the heterogeneous-budgets
experiment. Does not run anything; does not invoke the binary.

Writes (under <experiment>/runs/):
    _schedule.json
    schedule_summary.txt

Schedule shape:
    1 cell × 2 layers × 2 backends × K=11 trials = 44 entries

Determinism:
    Per layer, a fresh random.Random(1234) shuffles a 22-entry list of
    (backend, trial_index) pairs where backend in {std, hpx} and
    trial_index in 0..10. The same seed is used for both layers; the
    layers shuffle independently because each layer reseeds. Outer
    iteration order across layers is fixed:
        correctness_trace_on  →  timing_trace_off
    Rerunning this script must produce an identical _schedule.json.

measured_for_timing rules:
    correctness_trace_on:                  False
    timing_trace_off, trial_index == 0:    False
    timing_trace_off, trial_index in 1..10: True

trace_enabled rules:
    correctness_trace_on:  True   (LLAMA_SERVING_BENCH_HPX_TRACE=1)
    timing_trace_off:      False  (env var unset)
"""
import json
import random
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
RUNS    = EXP_DIR / "runs"

SEED = 1234

CELL = {
    "cell_id":              "C_2x4",
    "n_contexts":           2,
    "n_concurrent":         4,
    "n_requests":           12,
    "n_threads":            2,
    "ctx_size":             2048,
    "batch_size":           512,
    "expected_os_threads":  2,
    "expected_pool_size":   2,
    "expected_trace_lines": 27,
}

PLAN = [64, 8, 16, 8, 64, 16, 8, 64, 8, 16, 8, 32]

LAYERS   = ["correctness_trace_on", "timing_trace_off"]
BACKENDS = ["std", "hpx"]
TRIALS_PER_BACKEND = 11

LAYER_DIR = {
    "correctness_trace_on": "correctness",
    "timing_trace_off":     "timing",
}


def trace_enabled_for(layer: str) -> bool:
    return layer == "correctness_trace_on"


def measured_for_timing(layer: str, trial_index: int) -> bool:
    if layer == "correctness_trace_on":
        return False
    return trial_index != 0


def build_schedule():
    if len(PLAN) != CELL["n_requests"]:
        raise SystemExit(
            f"plan length ({len(PLAN)}) != n_requests ({CELL['n_requests']})"
        )
    entries = []
    global_index = 0
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
                "layer":                layer,
                "backend":              backend,
                "trial_index":          trial_index,
                "n_contexts":           CELL["n_contexts"],
                "n_concurrent":         CELL["n_concurrent"],
                "n_requests":           CELL["n_requests"],
                "max_tokens":           max(PLAN),
                "max_tokens_plan":      list(PLAN),
                "ctx_size":             CELL["ctx_size"],
                "batch_size":           CELL["batch_size"],
                "n_threads":            CELL["n_threads"],
                "seed_base":            SEED,
                "trace_enabled":        trace_enabled_for(layer),
                "measured_for_timing":  measured_for_timing(layer, trial_index),
                "expected_os_threads":  CELL["expected_os_threads"],
                "expected_pool_size":   CELL["expected_pool_size"],
                "expected_trace_lines": CELL["expected_trace_lines"],
                "output_dir": (
                    f"{LAYER_DIR[layer]}/{backend}/trial_{trial_index:02d}"
                ),
            }
            entries.append(entry)
            global_index += 1
    return entries


def write_schedule(entries, path: Path):
    payload = {
        "schedule_seed":      SEED,
        "cell":               CELL,
        "max_tokens_plan":    list(PLAN),
        "layers":             LAYERS,
        "backends":           BACKENDS,
        "trials_per_backend": TRIALS_PER_BACKEND,
        "trial_index_range":  [0, TRIALS_PER_BACKEND - 1],
        "total_entries":      len(entries),
        "schedule":           entries,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_summary(entries, path: Path):
    counts          = {}
    measured_counts = {}
    for e in entries:
        key = (e["layer"], e["backend"])
        counts[key] = counts.get(key, 0) + 1
        if e["measured_for_timing"]:
            measured_counts[key] = measured_counts.get(key, 0) + 1

    lines = []
    lines.append("Heterogeneous-budgets — schedule summary")
    lines.append("")
    lines.append(f"seed:                 {SEED}")
    lines.append(f"cell:                 {CELL['cell_id']}  "
                 f"(n_contexts={CELL['n_contexts']} "
                 f"n_concurrent={CELL['n_concurrent']} "
                 f"n_requests={CELL['n_requests']} "
                 f"n_threads={CELL['n_threads']})")
    lines.append(f"plan:                 {PLAN}")
    lines.append(f"max(plan):            {max(PLAN)}")
    lines.append(f"layers:               {LAYERS}")
    lines.append(f"backends:             {BACKENDS}")
    lines.append(
        f"trials per backend:   {TRIALS_PER_BACKEND} "
        f"(indices 0..{TRIALS_PER_BACKEND - 1} inclusive)"
    )
    lines.append(f"total entries:        {len(entries)}")
    lines.append(
        "expected total:       1 cell × 2 layers × 2 backends × "
        f"{TRIALS_PER_BACKEND} trials = "
        f"{2 * 2 * TRIALS_PER_BACKEND}"
    )
    lines.append("")
    lines.append("Per (layer × backend) trial counts:")
    lines.append("  layer                  bk   trials  measured_for_timing")
    for layer in LAYERS:
        for backend in BACKENDS:
            key = (layer, backend)
            lines.append(
                f"  {layer:22s} {backend:3s}  "
                f"{counts.get(key, 0):6d}  {measured_counts.get(key, 0):6d}"
            )
    lines.append("")
    lines.append("First 8 entries:")
    lines.append("  gidx  layer                  bk   trial  measured  trace")
    for e in entries[:8]:
        lines.append(
            f"  {e['global_index']:4d}  "
            f"{e['layer']:22s} {e['backend']:3s}  "
            f"{e['trial_index']:5d}  "
            f"{str(e['measured_for_timing']):8s}  "
            f"{str(e['trace_enabled'])}"
        )
    lines.append("")
    lines.append("Last 4 entries:")
    lines.append("  gidx  layer                  bk   trial  measured  trace")
    for e in entries[-4:]:
        lines.append(
            f"  {e['global_index']:4d}  "
            f"{e['layer']:22s} {e['backend']:3s}  "
            f"{e['trial_index']:5d}  "
            f"{str(e['measured_for_timing']):8s}  "
            f"{str(e['trace_enabled'])}"
        )
    lines.append("")
    lines.append(
        "Reproducibility: each layer reseeds random.Random(1234) and shuffles"
    )
    lines.append(
        "a fresh list of 22 (backend, trial_index) pairs. The seed is the only"
    )
    lines.append(
        "source of randomness; rerunning this script must produce an identical"
    )
    lines.append("_schedule.json byte-for-byte.")
    path.write_text("\n".join(lines) + "\n")


def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    entries = build_schedule()
    write_schedule(entries, RUNS / "_schedule.json")
    write_summary(entries, RUNS / "schedule_summary.txt")
    print(f"wrote {RUNS / '_schedule.json'}")
    print(f"wrote {RUNS / 'schedule_summary.txt'}")
    print(
        f"total entries: {len(entries)}  "
        f"(expected {2 * 2 * TRIALS_PER_BACKEND})"
    )


if __name__ == "__main__":
    main()
