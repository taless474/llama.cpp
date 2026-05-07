"""
Pair the std and hpx layer summaries for one layer and apply the
per-budget cross-backend hash-equality gate. Compute std-vs-hpx
deltas at multiple percentiles for the short-class pool and at the
median for trial-level metrics. Descriptive only — branch label is
assigned by _summarize_experiment.py.

Usage:
    python3 _summarize_condition.py --layer correctness_trace_on
    python3 _summarize_condition.py --layer timing_trace_off

Reads:
    summaries/layer_<layer_short>_std.txt
    summaries/layer_<layer_short>_hpx.txt
    summaries/layer_<layer_short>_std_all_requests.csv
    summaries/layer_<layer_short>_hpx_all_requests.csv
    summaries/layer_<layer_short>_std_per_trial.csv
    summaries/layer_<layer_short>_hpx_per_trial.csv

Writes:
    summaries/condition_<layer_short>.txt

Pass policy:
    CONDITION_OVERALL = PASS iff:
        layer_<layer_short>_std.LAYER_OVERALL    == PASS
        layer_<layer_short>_hpx.LAYER_OVERALL    == PASS
        per-budget cross-backend hash equality   PASS for every budget
            value in the plan (which here is the singleton {8})
"""
import argparse
import csv
import sys
from pathlib import Path
from statistics import mean, median, pstdev

EXP_DIR = Path(__file__).resolve().parent
SUMM    = EXP_DIR / "summaries"

LAYER_DIR = {
    "correctness_trace_on": "correctness",
    "timing_trace_off":     "timing",
}


def parse_args():
    p = argparse.ArgumentParser(description="pair std vs hpx for one layer")
    p.add_argument("--layer", required=True,
                   choices=["correctness_trace_on", "timing_trace_off"])
    return p.parse_args()


def parse_header(path: Path):
    if not path.exists():
        sys.exit(f"layer summary missing: {path}")
    out = {}
    for line in path.read_text().splitlines():
        if line.strip() == "# END HEADER":
            break
        if ":" in line:
            k, _, v = line.partition(":")
            out[k.strip()] = v.strip()
    return out


def percentile(xs_sorted, p):
    n = len(xs_sorted)
    if n == 0:
        return None
    k = (n - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, n - 1)
    frac = k - lo
    return xs_sorted[lo] * (1 - frac) + xs_sorted[hi] * frac


def stats(xs):
    if not xs:
        return None
    n = len(xs)
    s = sorted(xs)
    out = {
        "n":      n,
        "min":    s[0],
        "max":    s[-1],
        "mean":   mean(s),
        "median": median(s),
        "p50":    percentile(s, 50.0),
        "p90":    percentile(s, 90.0),
        "p95":    percentile(s, 95.0),
        "p99":    percentile(s, 99.0),
    }
    if n >= 2:
        sd = pstdev(s)
        out["stdev"] = sd
        out["cv"]    = (sd / out["mean"]) if out["mean"] else 0.0
    else:
        out["stdev"] = 0.0
        out["cv"]    = 0.0
    return out


def fmt(label, st):
    if st is None or st["n"] == 0:
        return f"  {label:36s} n=0"
    return (
        f"  {label:36s} n={st['n']:5d}  "
        f"p50={st['p50']:.3f}  p90={st['p90']:.3f}  p95={st['p95']:.3f}  "
        f"p99={st['p99']:.3f}  cv={st['cv']:.4f}"
    )


def fmt_delta_perc(label, std_st, hpx_st, percentile_keys=("p50", "p90", "p95", "p99")):
    if std_st is None or std_st["n"] == 0 or hpx_st is None or hpx_st["n"] == 0:
        return f"  {label:36s} (insufficient data)"
    parts = []
    for k in percentile_keys:
        s = std_st[k]; h = hpx_st[k]
        d = h - s
        dp = (d / s * 100.0) if s else 0.0
        parts.append(f"{k}: std={s:.3f} hpx={h:.3f} d={d:+.3f} ({dp:+.2f}%)")
    return f"  {label:36s}\n      " + "\n      ".join(parts)


def fmt_delta_med(label, std_st, hpx_st):
    if std_st is None or std_st["n"] == 0 or hpx_st is None or hpx_st["n"] == 0:
        return f"  {label:36s} (insufficient data)"
    s = std_st["median"]; h = hpx_st["median"]
    d = h - s
    dp = (d / s * 100.0) if s else 0.0
    return f"  {label:36s} std.med={s:.3f} hpx.med={h:.3f}  delta={d:+.3f} ({dp:+.2f}%)"


def read_all_requests(path: Path):
    if not path.exists():
        sys.exit(f"all-requests CSV missing: {path}")
    with path.open() as f:
        return list(csv.DictReader(f))


def read_per_trial(path: Path):
    if not path.exists():
        sys.exit(f"per-trial CSV missing: {path}")
    with path.open() as f:
        return list(csv.DictReader(f))


def per_budget_hashes(rows):
    out = {}
    for r in rows:
        b = int(r["n_tokens_generated"])
        out.setdefault(b, set()).add(r["generated_token_hash"])
    return out


def main():
    args        = parse_args()
    layer       = args.layer
    layer_short = LAYER_DIR[layer]

    SUMM.mkdir(parents=True, exist_ok=True)

    std_hdr_path = SUMM / f"layer_{layer_short}_std.txt"
    hpx_hdr_path = SUMM / f"layer_{layer_short}_hpx.txt"
    std_all_path = SUMM / f"layer_{layer_short}_std_all_requests.csv"
    hpx_all_path = SUMM / f"layer_{layer_short}_hpx_all_requests.csv"
    std_pt_path  = SUMM / f"layer_{layer_short}_std_per_trial.csv"
    hpx_pt_path  = SUMM / f"layer_{layer_short}_hpx_per_trial.csv"

    std_hdr = parse_header(std_hdr_path)
    hpx_hdr = parse_header(hpx_hdr_path)

    std_layer_pass = std_hdr.get("LAYER_OVERALL") == "PASS"
    hpx_layer_pass = hpx_hdr.get("LAYER_OVERALL") == "PASS"

    std_rows = read_all_requests(std_all_path)
    hpx_rows = read_all_requests(hpx_all_path)

    plan_unique = sorted(
        int(x) for x in (std_hdr.get("PLAN_UNIQUE", "")
                          .split(",")) if x
    )

    # ---- per-budget cross-backend hash equality ----
    std_hashes = per_budget_hashes(std_rows)
    hpx_hashes = per_budget_hashes(hpx_rows)

    n_budgets_total      = len(plan_unique)
    n_budgets_equal      = 0
    n_budgets_unequal    = 0
    n_budgets_missing    = 0
    n_budgets_inconsist  = 0
    per_budget_report    = []

    for b in plan_unique:
        std_set = std_hashes.get(b, set())
        hpx_set = hpx_hashes.get(b, set())
        if not std_set or not hpx_set:
            n_budgets_missing += 1
            per_budget_report.append((
                b, "MISSING", sorted(std_set), sorted(hpx_set),
            ))
            continue
        if len(std_set) != 1 or len(hpx_set) != 1:
            n_budgets_inconsist += 1
            per_budget_report.append((
                b, "INCONSISTENT", sorted(std_set), sorted(hpx_set),
            ))
            continue
        s = next(iter(std_set))
        h = next(iter(hpx_set))
        if s == h:
            n_budgets_equal += 1
            per_budget_report.append((b, "EQUAL", [s], [h]))
        else:
            n_budgets_unequal += 1
            per_budget_report.append((b, "UNEQUAL", [s], [h]))

    hash_gate_pass = (
        n_budgets_equal == n_budgets_total
        and n_budgets_unequal == 0
        and n_budgets_missing == 0
        and n_budgets_inconsist == 0
    )
    condition_pass = std_layer_pass and hpx_layer_pass and hash_gate_pass

    # ---- short-class pool stats (measured rows only) ----
    def pool_stats(rows, metric):
        xs = [
            float(r[metric]) for r in rows
            if r.get("measured_for_timing", "False") == "True"
        ]
        return stats(xs)

    std_total = pool_stats(std_rows, "total_ms")
    hpx_total = pool_stats(hpx_rows, "total_ms")
    std_ttft  = pool_stats(std_rows, "ttft_ms")
    hpx_ttft  = pool_stats(hpx_rows, "ttft_ms")
    std_dec   = pool_stats(std_rows, "total_minus_ttft_ms")
    hpx_dec   = pool_stats(hpx_rows, "total_minus_ttft_ms")

    def trial_metric(rows, key):
        xs = []
        for r in rows:
            if r.get("measured_for_timing", "False") != "True":
                continue
            v = r.get(key, "")
            if v == "" or v is None:
                continue
            try:
                xs.append(float(v))
            except ValueError:
                continue
        return stats(xs)

    std_pt = read_per_trial(std_pt_path)
    hpx_pt = read_per_trial(hpx_pt_path)

    std_makespan = trial_metric(std_pt, "makespan_ms")
    hpx_makespan = trial_metric(hpx_pt, "makespan_ms")
    std_pwall    = trial_metric(std_pt, "process_wall_ms")
    hpx_pwall    = trial_metric(hpx_pt, "process_wall_ms")
    std_atps     = trial_metric(std_pt, "agg_tok_per_s")
    hpx_atps     = trial_metric(hpx_pt, "agg_tok_per_s")

    def delta_pct_at(std_st, hpx_st, key):
        if not std_st or not hpx_st or not std_st["n"] or not hpx_st["n"]:
            return None
        s = std_st[key]
        if s == 0:
            return None
        return (hpx_st[key] - s) / s * 100.0

    d_total_med  = delta_pct_at(std_total, hpx_total, "median")
    d_total_p99  = delta_pct_at(std_total, hpx_total, "p99")
    d_total_p95  = delta_pct_at(std_total, hpx_total, "p95")
    d_total_p90  = delta_pct_at(std_total, hpx_total, "p90")
    d_mksp_med   = delta_pct_at(std_makespan, hpx_makespan, "median")

    out = []
    out.append(f"LAYER: {layer}")
    out.append(f"STD_LAYER_OVERALL: {std_hdr.get('LAYER_OVERALL', 'UNKNOWN')}")
    out.append(f"HPX_LAYER_OVERALL: {hpx_hdr.get('LAYER_OVERALL', 'UNKNOWN')}")
    out.append(f"PLAN_UNIQUE: {','.join(str(b) for b in plan_unique)}")
    out.append(f"N_BUDGETS_TOTAL: {n_budgets_total}")
    out.append(f"N_BUDGETS_EQUAL: {n_budgets_equal}")
    out.append(f"N_BUDGETS_UNEQUAL: {n_budgets_unequal}")
    out.append(f"N_BUDGETS_MISSING: {n_budgets_missing}")
    out.append(f"N_BUDGETS_INCONSISTENT: {n_budgets_inconsist}")
    out.append(f"HASH_EQUALITY: {'PASS' if hash_gate_pass else 'FAIL'}")
    out.append(f"CONDITION_OVERALL: {'PASS' if condition_pass else 'FAIL'}")
    out.append(f"DELTA_PCT_SHORT_TOTAL_MED: "
               f"{f'{d_total_med:+.4f}' if d_total_med is not None else 'NA'}")
    out.append(f"DELTA_PCT_SHORT_TOTAL_P90: "
               f"{f'{d_total_p90:+.4f}' if d_total_p90 is not None else 'NA'}")
    out.append(f"DELTA_PCT_SHORT_TOTAL_P95: "
               f"{f'{d_total_p95:+.4f}' if d_total_p95 is not None else 'NA'}")
    out.append(f"DELTA_PCT_SHORT_TOTAL_P99: "
               f"{f'{d_total_p99:+.4f}' if d_total_p99 is not None else 'NA'}")
    out.append(f"DELTA_PCT_MAKESPAN_MED: "
               f"{f'{d_mksp_med:+.4f}' if d_mksp_med is not None else 'NA'}")
    out.append("# END HEADER")
    out.append("")

    out.append(f"layer={layer}")
    out.append(
        f"std layer overall: {std_hdr.get('LAYER_OVERALL', 'UNKNOWN')}  "
        f"hpx layer overall: {hpx_hdr.get('LAYER_OVERALL', 'UNKNOWN')}"
    )
    out.append("")

    out.append("Per-budget cross-backend hash equality:")
    out.append("  budget  status        std_hash(es)                    hpx_hash(es)")
    for (b, st, sh, hh) in per_budget_report:
        sh_disp = sh[0] if len(sh) == 1 else (
            f"INCONSISTENT[{len(sh)}]" if sh else "(none)"
        )
        hh_disp = hh[0] if len(hh) == 1 else (
            f"INCONSISTENT[{len(hh)}]" if hh else "(none)"
        )
        out.append(f"  {b:6d}  {st:13s}  {sh_disp:34s}  {hh_disp}")
    out.append(f"  HASH_EQUALITY summary: {n_budgets_equal}/{n_budgets_total} EQUAL  "
               f"unequal={n_budgets_unequal}  missing={n_budgets_missing}  "
               f"inconsistent={n_budgets_inconsist}")
    out.append("")

    out.append("Short-class request pool (measured trials only):")
    out.append("  std view:")
    out.append(fmt("short  total_ms",            std_total))
    out.append(fmt("short  ttft_ms",             std_ttft))
    out.append(fmt("short  total_minus_ttft_ms", std_dec))
    out.append("  hpx view:")
    out.append(fmt("short  total_ms",            hpx_total))
    out.append(fmt("short  ttft_ms",             hpx_ttft))
    out.append(fmt("short  total_minus_ttft_ms", hpx_dec))
    out.append("")

    out.append("HPX-vs-std deltas at p50, p90, p95, p99:")
    out.append(fmt_delta_perc("short  total_ms",            std_total, hpx_total))
    out.append(fmt_delta_perc("short  ttft_ms",             std_ttft,  hpx_ttft))
    out.append(fmt_delta_perc("short  total_minus_ttft_ms", std_dec,   hpx_dec))
    out.append("")

    out.append("Per-trial trial-level metrics (measured trials only):")
    out.append("  std view:")
    out.append(fmt("makespan_ms",     std_makespan))
    out.append(fmt("process_wall_ms", std_pwall))
    out.append(fmt("agg_tok_per_s",   std_atps))
    out.append("  hpx view:")
    out.append(fmt("makespan_ms",     hpx_makespan))
    out.append(fmt("process_wall_ms", hpx_pwall))
    out.append(fmt("agg_tok_per_s",   hpx_atps))
    out.append("  delta (hpx - std):")
    out.append(fmt_delta_med("makespan_ms",     std_makespan, hpx_makespan))
    out.append(fmt_delta_med("process_wall_ms", std_pwall,    hpx_pwall))
    out.append(fmt_delta_med("agg_tok_per_s",   std_atps,     hpx_atps))
    out.append("")

    summary_path = SUMM / f"condition_{layer_short}.txt"
    summary_path.write_text("\n".join(out) + "\n")

    print(f"wrote {summary_path}")
    print(f"CONDITION_OVERALL: {'PASS' if condition_pass else 'FAIL'}")
    print(
        f"per-budget hash equality: {n_budgets_equal}/{n_budgets_total} EQUAL  "
        f"unequal={n_budgets_unequal}  missing={n_budgets_missing}  "
        f"inconsistent={n_budgets_inconsist}"
    )
    sys.exit(0 if condition_pass else 2)


if __name__ == "__main__":
    main()
