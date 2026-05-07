"""
Pair the std and hpx layer summaries for one layer and apply the
per-budget cross-backend hash-equality gate. Compute std-vs-hpx
deltas per class and per trial-level metric. Descriptive only —
branch label is assigned by _summarize_experiment.py.

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
            value in the plan

Hash-equality scope:
    For each budget b in the plan, compare the per-budget hash observed
    across all PASS trials of std with the per-budget hash observed
    across all PASS trials of hpx. The within-/across-trial uniqueness
    of the per-budget hash is gated upstream by _summarize_layer.py.

    Because the harness output is in completion order rather than
    submission order, per-(trial_index, req_index) matching is not a
    valid gate without harness changes — see facts.md "Harness output
    convention".

Timing aggregation:
    Restricted to rows with measured_for_timing == True. For
    correctness_trace_on, no rows are measured; only the hash-equality
    gate is reported.
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
        "p95":    percentile(s, 95.0),
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
        f"  {label:36s} n={st['n']:3d}  "
        f"median={st['median']:.3f}  p95={st['p95']:.3f}  "
        f"mean={st['mean']:.3f}  cv={st['cv']:.4f}"
    )


def fmt_delta(label, std_st, hpx_st):
    if std_st is None or std_st["n"] == 0 or hpx_st is None or hpx_st["n"] == 0:
        return f"  {label:36s} (insufficient data)"
    dmed = hpx_st["median"] - std_st["median"]
    dpct = (dmed / std_st["median"] * 100.0) if std_st["median"] else 0.0
    dp95 = hpx_st["p95"] - std_st["p95"]
    dp95p = (dp95 / std_st["p95"] * 100.0) if std_st["p95"] else 0.0
    return (
        f"  {label:36s} "
        f"std.med={std_st['median']:.3f} hpx.med={hpx_st['median']:.3f}  "
        f"delta_med={dmed:+.3f} ({dpct:+.2f}%)  "
        f"std.p95={std_st['p95']:.3f} hpx.p95={hpx_st['p95']:.3f}  "
        f"delta_p95={dp95:+.3f} ({dp95p:+.2f}%)"
    )


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
    """Return {budget_value: set(hashes)} across all rows of a layer.
    The within-trial / across-trial uniqueness is gated upstream;
    here we just report what was observed."""
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

    plan_budgets = sorted(
        int(x) for x in (std_hdr.get("PLAN_BUDGETS_SORTED_UNIQUE", "")
                          .split(",")) if x
    )

    # ---- per-budget cross-backend hash equality ----
    std_hashes = per_budget_hashes(std_rows)
    hpx_hashes = per_budget_hashes(hpx_rows)

    n_budgets_total      = len(plan_budgets)
    n_budgets_equal      = 0
    n_budgets_unequal    = 0
    n_budgets_missing    = 0
    n_budgets_inconsist  = 0
    per_budget_report    = []  # (b, status, std_hash, hpx_hash)

    for b in plan_budgets:
        std_set = std_hashes.get(b, set())
        hpx_set = hpx_hashes.get(b, set())
        if not std_set or not hpx_set:
            n_budgets_missing += 1
            per_budget_report.append((
                b, "MISSING",
                sorted(std_set), sorted(hpx_set),
            ))
            continue
        if len(std_set) != 1 or len(hpx_set) != 1:
            n_budgets_inconsist += 1
            per_budget_report.append((
                b, "INCONSISTENT",
                sorted(std_set), sorted(hpx_set),
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

    # ---- per-class timing aggregates (measured rows only) ----
    def class_stats(rows, cls, metric):
        xs = [
            float(r[metric]) for r in rows
            if r.get("measured_for_timing", "False") == "True"
            and r["class"] == cls
        ]
        return stats(xs)

    std_short_total = class_stats(std_rows, "short",  "total_ms")
    hpx_short_total = class_stats(hpx_rows, "short",  "total_ms")
    std_short_ttft  = class_stats(std_rows, "short",  "ttft_ms")
    hpx_short_ttft  = class_stats(hpx_rows, "short",  "ttft_ms")
    std_short_dec   = class_stats(std_rows, "short",  "total_minus_ttft_ms")
    hpx_short_dec   = class_stats(hpx_rows, "short",  "total_minus_ttft_ms")

    std_med_total   = class_stats(std_rows, "medium", "total_ms")
    hpx_med_total   = class_stats(hpx_rows, "medium", "total_ms")
    std_med_ttft    = class_stats(std_rows, "medium", "ttft_ms")
    hpx_med_ttft    = class_stats(hpx_rows, "medium", "ttft_ms")
    std_med_dec     = class_stats(std_rows, "medium", "total_minus_ttft_ms")
    hpx_med_dec     = class_stats(hpx_rows, "medium", "total_minus_ttft_ms")

    std_long_total  = class_stats(std_rows, "long",   "total_ms")
    hpx_long_total  = class_stats(hpx_rows, "long",   "total_ms")
    std_long_ttft   = class_stats(std_rows, "long",   "ttft_ms")
    hpx_long_ttft   = class_stats(hpx_rows, "long",   "ttft_ms")
    std_long_dec    = class_stats(std_rows, "long",   "total_minus_ttft_ms")
    hpx_long_dec    = class_stats(hpx_rows, "long",   "total_minus_ttft_ms")

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
    std_qd       = trial_metric(std_pt, "queue_drain_ms")
    hpx_qd       = trial_metric(hpx_pt, "queue_drain_ms")
    std_pwall    = trial_metric(std_pt, "process_wall_ms")
    hpx_pwall    = trial_metric(hpx_pt, "process_wall_ms")
    std_atps     = trial_metric(std_pt, "agg_tok_per_s")
    hpx_atps     = trial_metric(hpx_pt, "agg_tok_per_s")

    # ---- compose summary text ----
    def delta_pct(std_st, hpx_st):
        if not std_st or not hpx_st or not std_st["n"] or not hpx_st["n"]:
            return None
        if std_st["median"] == 0:
            return None
        return (hpx_st["median"] - std_st["median"]) / std_st["median"] * 100.0

    d_short = delta_pct(std_short_total, hpx_short_total)
    d_long  = delta_pct(std_long_total,  hpx_long_total)
    d_med   = delta_pct(std_med_total,   hpx_med_total)
    d_mksp  = delta_pct(std_makespan,    hpx_makespan)

    out = []
    out.append(f"LAYER: {layer}")
    out.append(f"STD_LAYER_OVERALL: {std_hdr.get('LAYER_OVERALL', 'UNKNOWN')}")
    out.append(f"HPX_LAYER_OVERALL: {hpx_hdr.get('LAYER_OVERALL', 'UNKNOWN')}")
    out.append(f"PLAN_BUDGETS: {','.join(str(b) for b in plan_budgets)}")
    out.append(f"N_BUDGETS_TOTAL: {n_budgets_total}")
    out.append(f"N_BUDGETS_EQUAL: {n_budgets_equal}")
    out.append(f"N_BUDGETS_UNEQUAL: {n_budgets_unequal}")
    out.append(f"N_BUDGETS_MISSING: {n_budgets_missing}")
    out.append(f"N_BUDGETS_INCONSISTENT: {n_budgets_inconsist}")
    out.append(f"HASH_EQUALITY: {'PASS' if hash_gate_pass else 'FAIL'}")
    out.append(f"CONDITION_OVERALL: {'PASS' if condition_pass else 'FAIL'}")
    out.append(f"DELTA_PCT_SHORT_TOTAL_MED: "
               f"{f'{d_short:+.4f}' if d_short is not None else 'NA'}")
    out.append(f"DELTA_PCT_LONG_TOTAL_MED: "
               f"{f'{d_long:+.4f}' if d_long is not None else 'NA'}")
    out.append(f"DELTA_PCT_MED_TOTAL_MED: "
               f"{f'{d_med:+.4f}' if d_med is not None else 'NA'}")
    out.append(f"DELTA_PCT_MAKESPAN_MED: "
               f"{f'{d_mksp:+.4f}' if d_mksp is not None else 'NA'}")
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
        sh_disp = sh[0] if len(sh) == 1 else f"INCONSISTENT[{len(sh)}]"
        hh_disp = hh[0] if len(hh) == 1 else (
            f"INCONSISTENT[{len(hh)}]" if len(hh) > 1 else "(none)"
        )
        if not sh:
            sh_disp = "(none)"
        out.append(f"  {b:6d}  {st:13s}  {sh_disp:34s}  {hh_disp}")
    out.append(f"  HASH_EQUALITY summary: {n_budgets_equal}/{n_budgets_total} EQUAL  "
               f"unequal={n_budgets_unequal}  missing={n_budgets_missing}  "
               f"inconsistent={n_budgets_inconsist}")
    out.append("")

    out.append("Per-class request pools (measured trials only):")
    out.append("  std view:")
    out.append(fmt("short  total_ms",            std_short_total))
    out.append(fmt("short  ttft_ms",             std_short_ttft))
    out.append(fmt("short  total_minus_ttft_ms", std_short_dec))
    out.append(fmt("medium total_ms",            std_med_total))
    out.append(fmt("medium ttft_ms",             std_med_ttft))
    out.append(fmt("medium total_minus_ttft_ms", std_med_dec))
    out.append(fmt("long   total_ms",            std_long_total))
    out.append(fmt("long   ttft_ms",             std_long_ttft))
    out.append(fmt("long   total_minus_ttft_ms", std_long_dec))
    out.append("  hpx view:")
    out.append(fmt("short  total_ms",            hpx_short_total))
    out.append(fmt("short  ttft_ms",             hpx_short_ttft))
    out.append(fmt("short  total_minus_ttft_ms", hpx_short_dec))
    out.append(fmt("medium total_ms",            hpx_med_total))
    out.append(fmt("medium ttft_ms",             hpx_med_ttft))
    out.append(fmt("medium total_minus_ttft_ms", hpx_med_dec))
    out.append(fmt("long   total_ms",            hpx_long_total))
    out.append(fmt("long   ttft_ms",             hpx_long_ttft))
    out.append(fmt("long   total_minus_ttft_ms", hpx_long_dec))
    out.append("")

    out.append("HPX-vs-std deltas (median, p95):")
    out.append(fmt_delta("short  total_ms",            std_short_total, hpx_short_total))
    out.append(fmt_delta("short  ttft_ms",             std_short_ttft,  hpx_short_ttft))
    out.append(fmt_delta("short  total_minus_ttft_ms", std_short_dec,   hpx_short_dec))
    out.append(fmt_delta("medium total_ms",            std_med_total,   hpx_med_total))
    out.append(fmt_delta("medium ttft_ms",             std_med_ttft,    hpx_med_ttft))
    out.append(fmt_delta("medium total_minus_ttft_ms", std_med_dec,     hpx_med_dec))
    out.append(fmt_delta("long   total_ms",            std_long_total,  hpx_long_total))
    out.append(fmt_delta("long   ttft_ms",             std_long_ttft,   hpx_long_ttft))
    out.append(fmt_delta("long   total_minus_ttft_ms", std_long_dec,    hpx_long_dec))
    out.append("")

    out.append("Per-trial trial-level metrics (measured trials only):")
    out.append("  std view:")
    out.append(fmt("makespan_ms",     std_makespan))
    out.append(fmt("queue_drain_ms",  std_qd))
    out.append(fmt("process_wall_ms", std_pwall))
    out.append(fmt("agg_tok_per_s",   std_atps))
    out.append("  hpx view:")
    out.append(fmt("makespan_ms",     hpx_makespan))
    out.append(fmt("queue_drain_ms",  hpx_qd))
    out.append(fmt("process_wall_ms", hpx_pwall))
    out.append(fmt("agg_tok_per_s",   hpx_atps))
    out.append("  delta (hpx - std):")
    out.append(fmt_delta("makespan_ms",     std_makespan, hpx_makespan))
    out.append(fmt_delta("queue_drain_ms",  std_qd,       hpx_qd))
    out.append(fmt_delta("process_wall_ms", std_pwall,    hpx_pwall))
    out.append(fmt_delta("agg_tok_per_s",   std_atps,     hpx_atps))
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
