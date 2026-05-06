"""
Phase 2 helper — summarize one cell across both layers and both backends.

Usage:
  python3 _summarize_condition.py --cell-id A_1x1

Reads:
  conditions/<cell_id>/correctness/std/condition_summary.txt
  conditions/<cell_id>/correctness/hpx/condition_summary.txt
  conditions/<cell_id>/timing/std/condition_summary.txt
  conditions/<cell_id>/timing/hpx/condition_summary.txt
  conditions/<cell_id>/timing/std/all_requests.csv
  conditions/<cell_id>/timing/hpx/all_requests.csv
  conditions/<cell_id>/timing/std/per_trial_summary.csv
  conditions/<cell_id>/timing/hpx/per_trial_summary.csv

Each layer summary file begins with a machine-readable `KEY: VALUE`
header section produced by _summarize_layer.py and terminated by
`# END HEADER`. We parse `LAYER_OVERALL` from each header to gate
correctness without re-applying per-trial gates.

Correctness gate (cell-level):
  - correctness_trace_on / std PASS  AND
  - correctness_trace_on / hpx PASS  AND
  - timing_trace_off     / std PASS  AND
  - timing_trace_off     / hpx PASS

Timing interpretation uses ONLY rows in:
  conditions/<cell_id>/timing/{std,hpx}/all_requests.csv
where measured_for_timing == true (i.e. trial_index in 1..30).
Pool view: every measured request row.
Per-trial view: one number per trial from per_trial_summary.csv
(measured_for_timing=true).

Compared metrics (std vs hpx):
  - total_ms                  pool view (n = n_requests * 30)
  - ttft_ms                   pool view
  - total_minus_ttft_ms       pool view
  - tokens_per_second         pool view
  - process_wall_ms           per-trial view (n = 30)
  - agg_tok_s                 per-trial view (n = 30)
  - median total_ms           per-trial-median view (n = 30)
  - median total_minus_ttft   per-trial-median view (n = 30)

For each, report mean, median, p95, p99, stdev_pop, CV (per backend),
and delta (hpx - std), delta % (delta / std median * 100) on the median.

Writes:
  conditions/<cell_id>/condition_comparison.txt

condition_comparison.txt opens with a machine-readable `KEY: VALUE`
block (terminated by `# END HEADER`) so the matrix summarizer can parse
state without re-aggregating. Timing numbers are descriptive and gate
nothing; the only OVERALL flag is correctness.
"""
import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean, median, pstdev

ROOT     = Path("/Users/Ashk/Desktop/HPX/llama-hpx/local/baselines/perf_hpx_vs_std_matrix")
SCHEDULE = ROOT / "_schedule.json"

LAYER_DIR = {
    "correctness_trace_on": "correctness",
    "timing_trace_off":     "timing",
}


def parse_args():
    p = argparse.ArgumentParser(description="summarize one matrix cell")
    p.add_argument("--cell-id", required=True,
                   choices=["A_1x1", "B_2x2", "C_2x4", "D_4x4"])
    return p.parse_args()


def parse_header(path: Path) -> dict:
    """Parse the leading 'KEY: VALUE' block (terminated by '# END HEADER')."""
    if not path.exists():
        return {}
    out = {}
    for raw in path.read_text().splitlines():
        if raw.strip() == "# END HEADER":
            break
        if raw.startswith("#") or not raw.strip():
            continue
        if ":" not in raw:
            continue
        key, _, val = raw.partition(":")
        out[key.strip()] = val.strip()
    return out


def stats(xs):
    if not xs:
        return None
    n = len(xs)
    s = sorted(xs)
    def percentile(p):
        if n == 1:
            return s[0]
        k = (n - 1) * (p / 100.0)
        lo = int(k); hi = min(lo + 1, n - 1); frac = k - lo
        return s[lo] * (1.0 - frac) + s[hi] * frac
    pop = pstdev(xs) if n > 1 else 0.0
    mn = mean(xs)
    cv = (pop / mn) if mn != 0 else 0.0
    return {
        "n":         n,
        "min":       min(xs),
        "max":       max(xs),
        "mean":      mn,
        "median":    median(xs),
        "p95":       percentile(95),
        "p99":       percentile(99),
        "stdev_pop": pop,
        "cv":        cv,
    }


def load_pool(path: Path):
    """Return measured-only pool dict of {metric_name: list[float]}."""
    out = {"total_ms": [], "ttft_ms": [], "tokens_per_second": [], "total_minus_ttft_ms": []}
    if not path.exists():
        return out
    with path.open() as f:
        for r in csv.DictReader(f):
            if r.get("measured_for_timing", "").lower() != "true":
                continue
            if r.get("trial_pass", "").lower() != "true":
                continue
            try:
                out["total_ms"].append(float(r["total_ms"]))
                out["ttft_ms"].append(float(r["ttft_ms"]))
                out["tokens_per_second"].append(float(r["tokens_per_second"]))
                out["total_minus_ttft_ms"].append(float(r["total_minus_ttft_ms"]))
            except (KeyError, ValueError):
                continue
    return out


def load_per_trial(path: Path):
    """Return measured-only per-trial dict of {metric: list[float]}."""
    out = {
        "process_wall_ms":            [],
        "agg_tok_s":                  [],
        "total_median_ms":            [],
        "total_minus_ttft_median_ms": [],
    }
    if not path.exists():
        return out
    with path.open() as f:
        for r in csv.DictReader(f):
            if r.get("measured_for_timing", "").lower() != "true":
                continue
            if r.get("trial_pass", "").lower() != "true":
                continue
            for k_csv, k_out in [
                ("process_wall_ms",            "process_wall_ms"),
                ("agg_tok_s",                  "agg_tok_s"),
                ("total_median_ms",            "total_median_ms"),
                ("total_minus_ttft_median_ms", "total_minus_ttft_median_ms"),
            ]:
                v = r.get(k_csv, "")
                if v == "":
                    continue
                try:
                    out[k_out].append(float(v))
                except ValueError:
                    pass
    return out


def main():
    args = parse_args()
    cell_dir = ROOT / "conditions" / args.cell_id

    schedule = json.loads(SCHEDULE.read_text())
    sample = next(
        (e for e in schedule["schedule"] if e["cell_id"] == args.cell_id),
        None,
    )
    if sample is None:
        sys.exit(f"no schedule entries for cell_id={args.cell_id}")
    n_contexts   = sample["n_contexts"]
    n_concurrent = sample["n_concurrent"]
    n_requests   = sample["n_requests"]

    summaries = {}
    for layer, ldir in LAYER_DIR.items():
        for backend in ("std", "hpx"):
            path = cell_dir / ldir / backend / "condition_summary.txt"
            summaries[(layer, backend)] = {
                "path":   path,
                "header": parse_header(path),
            }

    def _layer_pass(layer, backend):
        h = summaries[(layer, backend)]["header"]
        return h.get("LAYER_OVERALL") == "PASS"

    correctness_std_pass = _layer_pass("correctness_trace_on", "std")
    correctness_hpx_pass = _layer_pass("correctness_trace_on", "hpx")
    timing_std_pass      = _layer_pass("timing_trace_off",     "std")
    timing_hpx_pass      = _layer_pass("timing_trace_off",     "hpx")
    correctness_overall = (
        correctness_std_pass and correctness_hpx_pass
        and timing_std_pass and timing_hpx_pass
    )

    timing_dir_std = cell_dir / "timing" / "std"
    timing_dir_hpx = cell_dir / "timing" / "hpx"

    pool_std = load_pool(timing_dir_std / "all_requests.csv")
    pool_hpx = load_pool(timing_dir_hpx / "all_requests.csv")
    pt_std   = load_per_trial(timing_dir_std / "per_trial_summary.csv")
    pt_hpx   = load_per_trial(timing_dir_hpx / "per_trial_summary.csv")

    # ---- per-metric stats ----
    metrics = []  # list of dicts: {name, view, std_stats, hpx_stats}
    def _add(name, view, std_xs, hpx_xs):
        metrics.append({
            "name": name,
            "view": view,
            "std":  stats(std_xs),
            "hpx":  stats(hpx_xs),
        })

    _add("total_ms",             "pool",      pool_std["total_ms"],             pool_hpx["total_ms"])
    _add("ttft_ms",              "pool",      pool_std["ttft_ms"],              pool_hpx["ttft_ms"])
    _add("total_minus_ttft_ms",  "pool",      pool_std["total_minus_ttft_ms"],  pool_hpx["total_minus_ttft_ms"])
    _add("tokens_per_second",    "pool",      pool_std["tokens_per_second"],    pool_hpx["tokens_per_second"])
    _add("process_wall_ms",      "per_trial", pt_std["process_wall_ms"],        pt_hpx["process_wall_ms"])
    _add("agg_tok_s",            "per_trial", pt_std["agg_tok_s"],              pt_hpx["agg_tok_s"])
    _add("median_total_ms",      "per_trial", pt_std["total_median_ms"],        pt_hpx["total_median_ms"])
    _add("median_total_minus_ttft_ms", "per_trial",
         pt_std["total_minus_ttft_median_ms"],
         pt_hpx["total_minus_ttft_median_ms"])

    # ---- write condition_comparison.txt ----
    out = []
    out.append(f"# condition comparison: cell {args.cell_id}")
    out.append("# machine-readable header (KEY: VALUE); ends at '# END HEADER'")
    out.append(f"cell_id: {args.cell_id}")
    out.append(f"n_contexts: {n_contexts}")
    out.append(f"n_concurrent: {n_concurrent}")
    out.append(f"n_requests: {n_requests}")
    out.append(f"correctness_trace_on.std: {'PASS' if correctness_std_pass else 'FAIL'}")
    out.append(f"correctness_trace_on.hpx: {'PASS' if correctness_hpx_pass else 'FAIL'}")
    out.append(f"timing_trace_off.std: {'PASS' if timing_std_pass else 'FAIL'}")
    out.append(f"timing_trace_off.hpx: {'PASS' if timing_hpx_pass else 'FAIL'}")
    for m in metrics:
        s_std = m["std"]; s_hpx = m["hpx"]
        if s_std is not None:
            out.append(f"metric.{m['name']}.{m['view']}.std.n: {s_std['n']}")
            out.append(f"metric.{m['name']}.{m['view']}.std.median: {s_std['median']:.6f}")
            out.append(f"metric.{m['name']}.{m['view']}.std.mean: {s_std['mean']:.6f}")
            out.append(f"metric.{m['name']}.{m['view']}.std.p95: {s_std['p95']:.6f}")
            out.append(f"metric.{m['name']}.{m['view']}.std.p99: {s_std['p99']:.6f}")
            out.append(f"metric.{m['name']}.{m['view']}.std.stdev_pop: {s_std['stdev_pop']:.6f}")
            out.append(f"metric.{m['name']}.{m['view']}.std.cv: {s_std['cv']:.6f}")
        else:
            out.append(f"metric.{m['name']}.{m['view']}.std.n: 0")
        if s_hpx is not None:
            out.append(f"metric.{m['name']}.{m['view']}.hpx.n: {s_hpx['n']}")
            out.append(f"metric.{m['name']}.{m['view']}.hpx.median: {s_hpx['median']:.6f}")
            out.append(f"metric.{m['name']}.{m['view']}.hpx.mean: {s_hpx['mean']:.6f}")
            out.append(f"metric.{m['name']}.{m['view']}.hpx.p95: {s_hpx['p95']:.6f}")
            out.append(f"metric.{m['name']}.{m['view']}.hpx.p99: {s_hpx['p99']:.6f}")
            out.append(f"metric.{m['name']}.{m['view']}.hpx.stdev_pop: {s_hpx['stdev_pop']:.6f}")
            out.append(f"metric.{m['name']}.{m['view']}.hpx.cv: {s_hpx['cv']:.6f}")
        else:
            out.append(f"metric.{m['name']}.{m['view']}.hpx.n: 0")
        if s_std is not None and s_hpx is not None and s_std["median"] != 0:
            d   = s_hpx["median"] - s_std["median"]
            dpc = (d / s_std["median"]) * 100.0
            out.append(f"metric.{m['name']}.{m['view']}.delta_median: {d:.6f}")
            out.append(f"metric.{m['name']}.{m['view']}.delta_pct: {dpc:.4f}")

    out.append(f"CONDITION_OVERALL_CORRECTNESS: {'PASS' if correctness_overall else 'FAIL'}")
    out.append("# END HEADER")
    out.append("")

    out.append(f"=== condition comparison: cell {args.cell_id} ===")
    out.append(f"n_contexts={n_contexts}  n_concurrent={n_concurrent}  n_requests={n_requests}")
    out.append("")
    out.append("--- correctness state by layer/backend ---")
    out.append(f"  correctness_trace_on / std : {'PASS' if correctness_std_pass else 'FAIL'}")
    out.append(f"  correctness_trace_on / hpx : {'PASS' if correctness_hpx_pass else 'FAIL'}")
    out.append(f"  timing_trace_off     / std : {'PASS' if timing_std_pass else 'FAIL'}")
    out.append(f"  timing_trace_off     / hpx : {'PASS' if timing_hpx_pass else 'FAIL'}")
    out.append("")

    if not correctness_overall:
        out.append("CONDITION_OVERALL_CORRECTNESS: FAIL")
        out.append("Timing comparison suppressed because correctness state is uncertain.")
        out.append("See per-layer condition_summary.txt for first-fail labels.")
        out.append("")
    else:
        out.append("CONDITION_OVERALL_CORRECTNESS: PASS")
        out.append("Timing tables below are descriptive only and gate nothing.")
        out.append("")

    def _row(name, view, s_std, s_hpx, unit):
        if s_std is None or s_hpx is None:
            return f"| {name:38s} | {view:9s} | n=0          | n=0          |"
        d = s_hpx["median"] - s_std["median"]
        dpc = (d / s_std["median"]) * 100.0 if s_std["median"] != 0 else 0.0
        return (
            f"| {name:38s} | {view:9s} | "
            f"std n={s_std['n']:4d} median={s_std['median']:8.3f}{unit} "
            f"p95={s_std['p95']:8.3f}{unit} cv={s_std['cv']:.3f} | "
            f"hpx n={s_hpx['n']:4d} median={s_hpx['median']:8.3f}{unit} "
            f"p95={s_hpx['p95']:8.3f}{unit} cv={s_hpx['cv']:.3f} | "
            f"d={d:+8.3f}{unit} ({dpc:+5.1f}%)"
        )

    out.append("--- timing comparison (std vs hpx; measured_for_timing=true rows only) ---")
    units = {
        "total_ms":                       "ms",
        "ttft_ms":                        "ms",
        "total_minus_ttft_ms":            "ms",
        "tokens_per_second":              "tps",
        "process_wall_ms":                "ms",
        "agg_tok_s":                      "tps",
        "median_total_ms":                "ms",
        "median_total_minus_ttft_ms":     "ms",
    }
    for m in metrics:
        out.append(_row(m["name"], m["view"], m["std"], m["hpx"], units[m["name"]]))
    out.append("")

    out.append("--- caveats ---")
    out.append("- Pool view: samples within a trial share runtime/cache state and are NOT iid;")
    out.append("  the n is large but percentile estimates are optimistic.")
    out.append("- Per-trial view: small n (=30 measured) but iid across trials.")
    out.append("- Timing numbers gate nothing. The only OVERALL flag is correctness.")
    out.append("- Δ % is (hpx_median - std_median) / std_median * 100, on median only.")
    out.append("- Sub-1% deltas inside the CV band are not interpretable as 'different'.")
    out.append("")

    cmp_path = cell_dir / "condition_comparison.txt"
    cell_dir.mkdir(parents=True, exist_ok=True)
    cmp_path.write_text("\n".join(out))

    # ---- console echo ----
    print(f"cell_id={args.cell_id}")
    print(f"correctness_trace_on.std={correctness_std_pass}  hpx={correctness_hpx_pass}")
    print(f"timing_trace_off.std={timing_std_pass}  hpx={timing_hpx_pass}")
    print(f"CONDITION_OVERALL_CORRECTNESS: {'PASS' if correctness_overall else 'FAIL'}")
    print(f"wrote {cmp_path}")
    sys.exit(0 if correctness_overall else 1)


if __name__ == "__main__":
    main()
