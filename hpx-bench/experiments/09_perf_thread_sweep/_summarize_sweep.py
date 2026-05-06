"""
Phase 2 helper — write the cross-n_threads sweep summary for the
C_2x4 n_threads sensitivity protocol.

Usage:
  python3 _summarize_sweep.py

Reads (one per n_threads):
  thread_settings/n1/condition_comparison.txt
  thread_settings/n2/condition_comparison.txt
  thread_settings/n4/condition_comparison.txt

Each file begins with a machine-readable `KEY: VALUE` block produced by
_summarize_thread_setting.py and terminated by `# END HEADER`. We parse:

  cell_id, n_threads, n_contexts, n_concurrent, n_requests
  correctness_trace_on.std / .hpx, timing_trace_off.std / .hpx
  THREAD_OVERALL_CORRECTNESS
  metric.<name>.<view>.std.median / mean / p95 / p99 / stdev_pop / cv
  metric.<name>.<view>.hpx.median / mean / p95 / p99 / stdev_pop / cv
  metric.<name>.<view>.delta_median / delta_pct

If a thread setting's condition_comparison.txt is missing, that setting
is reported MISSING. If correctness for any setting is FAIL, that
setting's timing rows are suppressed (replaced with '<suppressed>').

The sweep-level OVERALL gate is correctness for ALL three thread
settings. Timing deltas are descriptive and do not gate.

Trend classification (a-priori) on metric.total_ms.pool.delta_pct:

  oversubscription:
    delta_pct(n=1) < delta_pct(n=2) < delta_pct(n=4)  (strict monotonic)
    AND  max(deltas) - min(deltas) >= FLAT_THRESHOLD_PCT

  orchestration_or_waiter_path:
    max(deltas) - min(deltas) < FLAT_THRESHOLD_PCT  (across all three)

  mixed:
    otherwise

  incomplete:
    any thread setting missing or correctness FAIL

FLAT_THRESHOLD_PCT is 1.0 by convention; this matches the README's
"sub-1% deltas inside the CV band are not interpretable as 'different'".

Writes:
  sweep_summary.txt
"""
import sys
from pathlib import Path

ROOT = Path("/Users/Ashk/Desktop/HPX/llama-hpx/local/baselines/perf_thread_sweep")
THREAD_SETTINGS = [1, 2, 4]
FLAT_THRESHOLD_PCT = 1.0


def parse_header(path: Path) -> dict:
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


def fnum(d: dict, key: str):
    if key not in d:
        return None
    try:
        return float(d[key])
    except ValueError:
        return None


def inum(d: dict, key: str):
    if key not in d:
        return None
    try:
        return int(d[key])
    except ValueError:
        return None


def classify_trend(settings_data, key):
    """
    settings_data: list of per-setting dicts, sorted by n_threads ascending.
    key: metric.delta_pct field, e.g. 'metric.total_ms.pool.delta_pct'.
    Returns ('branch_label', list_of_(n_threads, delta_pct)).
    """
    pairs = []
    for s in settings_data:
        if not s["present"] or s["overall"] != "PASS":
            return ("incomplete", pairs)
        v = fnum(s["header"], key)
        if v is None:
            return ("incomplete", pairs)
        pairs.append((s["n_threads"], v))
    if len(pairs) != len(THREAD_SETTINGS):
        return ("incomplete", pairs)
    vals = [v for _, v in pairs]  # ascending by n_threads: [d_n1, d_n2, d_n4]
    rng = max(vals) - min(vals)
    if rng < FLAT_THRESHOLD_PCT:
        return ("orchestration_or_waiter_path", pairs)
    # strict-monotonic-shrink as n_threads decreases means d_n1 < d_n2 < d_n4
    if vals[0] < vals[1] < vals[2]:
        return ("oversubscription", pairs)
    return ("mixed", pairs)


def main():
    settings_data = []
    for n in THREAD_SETTINGS:
        cmp_path = ROOT / "thread_settings" / f"n{n}" / "condition_comparison.txt"
        h = parse_header(cmp_path)
        settings_data.append({
            "n_threads": n,
            "path":      cmp_path,
            "header":    h,
            "present":   bool(h),
            "overall":   h.get("THREAD_OVERALL_CORRECTNESS"),
            "cell_id":   h.get("cell_id"),
            "n_ctx":     inum(h, "n_contexts"),
            "n_conc":    inum(h, "n_concurrent"),
            "n_req":     inum(h, "n_requests"),
            "c_std":     h.get("correctness_trace_on.std"),
            "c_hpx":     h.get("correctness_trace_on.hpx"),
            "t_std":     h.get("timing_trace_off.std"),
            "t_hpx":     h.get("timing_trace_off.hpx"),
        })

    sweep_overall_pass = all(s["overall"] == "PASS" for s in settings_data)

    # ---- trend classifications ----
    trend_total,  pairs_total  = classify_trend(settings_data, "metric.total_ms.pool.delta_pct")
    trend_tmt,    pairs_tmt    = classify_trend(settings_data, "metric.total_minus_ttft_ms.pool.delta_pct")
    trend_ttft,   pairs_ttft   = classify_trend(settings_data, "metric.ttft_ms.pool.delta_pct")
    trend_pwall,  pairs_pwall  = classify_trend(settings_data, "metric.process_wall_ms.per_trial.delta_pct")

    out = []
    out.append("# sweep summary (cross-n_threads HPX-vs-std at C_2x4)")
    out.append("# machine-readable header (KEY: VALUE); ends at '# END HEADER'")
    cell_ids = {s["cell_id"] for s in settings_data if s["cell_id"]}
    if len(cell_ids) == 1:
        out.append(f"cell_id: {next(iter(cell_ids))}")
    else:
        out.append(f"cell_id: {sorted(cell_ids)}")
    for s in settings_data:
        out.append(f"thread_setting.n{s['n_threads']}.present: {str(s['present']).lower()}")
        out.append(f"thread_setting.n{s['n_threads']}.correctness_overall: "
                   f"{s['overall'] or 'MISSING'}")
        out.append(f"thread_setting.n{s['n_threads']}.correctness_trace_on.std: "
                   f"{s['c_std'] or 'MISSING'}")
        out.append(f"thread_setting.n{s['n_threads']}.correctness_trace_on.hpx: "
                   f"{s['c_hpx'] or 'MISSING'}")
        out.append(f"thread_setting.n{s['n_threads']}.timing_trace_off.std: "
                   f"{s['t_std'] or 'MISSING'}")
        out.append(f"thread_setting.n{s['n_threads']}.timing_trace_off.hpx: "
                   f"{s['t_hpx'] or 'MISSING'}")
    out.append(f"flat_threshold_pct: {FLAT_THRESHOLD_PCT}")
    out.append(f"trend.total_ms.pool: {trend_total}")
    out.append(f"trend.total_minus_ttft_ms.pool: {trend_tmt}")
    out.append(f"trend.ttft_ms.pool: {trend_ttft}")
    out.append(f"trend.process_wall_ms.per_trial: {trend_pwall}")
    for n, v in pairs_total:
        out.append(f"delta_pct.total_ms.pool.n{n}: {v:.4f}")
    for n, v in pairs_tmt:
        out.append(f"delta_pct.total_minus_ttft_ms.pool.n{n}: {v:.4f}")
    for n, v in pairs_ttft:
        out.append(f"delta_pct.ttft_ms.pool.n{n}: {v:.4f}")
    for n, v in pairs_pwall:
        out.append(f"delta_pct.process_wall_ms.per_trial.n{n}: {v:.4f}")
    out.append(f"SWEEP_OVERALL_CORRECTNESS: {'PASS' if sweep_overall_pass else 'FAIL'}")
    out.append("# END HEADER")
    out.append("")

    out.append("=== C_2x4 n_threads sensitivity sweep — cross-setting summary ===")
    out.append("")
    out.append("--- correctness gate by thread setting ---")
    out.append("| n_thr | n_ctx | n_conc | n_req | c_std | c_hpx | t_std | t_hpx | OVERALL |")
    for s in settings_data:
        if not s["present"]:
            out.append(
                f"| {s['n_threads']:5d} | -     | -      | -     | -     | -     | -     | -     | MISSING |"
            )
            continue
        out.append(
            f"| {s['n_threads']:5d} | "
            f"{s['n_ctx']:5d} | {s['n_conc']:6d} | {s['n_req']:5d} | "
            f"{s['c_std']:5s} | {s['c_hpx']:5s} | "
            f"{s['t_std']:5s} | {s['t_hpx']:5s} | "
            f"{s['overall']:7s} |"
        )
    out.append("")
    out.append(f"SWEEP_OVERALL_CORRECTNESS: {'PASS' if sweep_overall_pass else 'FAIL'}")
    out.append("")

    def _metric_table(title, key, view, unit):
        out.append(f"--- {title} (view: {view}; std vs hpx; Δ on median) ---")
        out.append(
            "| n_thr | OK?   | "
            "std median       | hpx median       | "
            "Δ                | Δ %     | std CV  | hpx CV  |"
        )
        for s in settings_data:
            if not s["present"]:
                out.append(
                    f"| {s['n_threads']:5d} | -     | "
                    f"-                | -                | "
                    f"-                | -       | -       | -       |"
                )
                continue
            ok = s["overall"] == "PASS"
            ok_str = "yes" if ok else "no"
            std_med = fnum(s["header"], f"metric.{key}.{view}.std.median")
            hpx_med = fnum(s["header"], f"metric.{key}.{view}.hpx.median")
            std_cv  = fnum(s["header"], f"metric.{key}.{view}.std.cv")
            hpx_cv  = fnum(s["header"], f"metric.{key}.{view}.hpx.cv")
            d       = fnum(s["header"], f"metric.{key}.{view}.delta_median")
            dpc     = fnum(s["header"], f"metric.{key}.{view}.delta_pct")
            if not ok or std_med is None or hpx_med is None:
                out.append(
                    f"| {s['n_threads']:5d} | {ok_str:5s} | "
                    f"{'<suppressed>':16s} | {'<suppressed>':16s} | "
                    f"{'<suppressed>':16s} | {'-':7s} | "
                    f"{'-':7s} | {'-':7s} |"
                )
                continue
            out.append(
                f"| {s['n_threads']:5d} | {ok_str:5s} | "
                f"{std_med:8.3f}{unit:8s} | {hpx_med:8.3f}{unit:8s} | "
                f"{d:+8.3f}{unit:8s} | "
                f"{dpc:+6.2f}% | "
                f"{std_cv:7.4f} | {hpx_cv:7.4f} |"
            )
        out.append("")

    _metric_table("total_ms",                  "total_ms",                  "pool",      "ms")
    _metric_table("ttft_ms",                   "ttft_ms",                   "pool",      "ms")
    _metric_table("total_minus_ttft_ms",       "total_minus_ttft_ms",       "pool",      "ms")
    _metric_table("tokens_per_second",         "tokens_per_second",         "pool",      "tps")
    _metric_table("process_wall_ms",           "process_wall_ms",           "per_trial", "ms")
    _metric_table("agg_tok_s",                 "agg_tok_s",                 "per_trial", "tps")
    _metric_table("per-trial median total_ms", "median_total_ms",           "per_trial", "ms")
    _metric_table("per-trial median tmt_ms",   "median_total_minus_ttft_ms","per_trial", "ms")

    # ---- trend section ----
    out.append("--- HPX-vs-std delta % across n_threads (primary trend metric: total_ms median) ---")
    out.append(f"flat_threshold_pct: {FLAT_THRESHOLD_PCT}")
    out.append("")

    def _trend_block(title, key, label, pairs_local, trend_local):
        out.append(f"  {title}  ({key})")
        if not pairs_local:
            out.append(f"    delta_pct values:  <unavailable>")
        else:
            for n, v in pairs_local:
                out.append(f"    n_threads={n}: {v:+7.4f}%")
            rng = max(v for _, v in pairs_local) - min(v for _, v in pairs_local)
            out.append(f"    range:             {rng:7.4f}%")
        out.append(f"    trend:             {trend_local}")
        out.append("")

    _trend_block("primary",
                 "metric.total_ms.pool.delta_pct",
                 "total_ms (pool, median)", pairs_total, trend_total)
    _trend_block("secondary",
                 "metric.total_minus_ttft_ms.pool.delta_pct",
                 "total_minus_ttft_ms (pool, median)", pairs_tmt, trend_tmt)
    _trend_block("secondary",
                 "metric.ttft_ms.pool.delta_pct",
                 "ttft_ms (pool, median)", pairs_ttft, trend_ttft)
    _trend_block("secondary",
                 "metric.process_wall_ms.per_trial.delta_pct",
                 "process_wall_ms (per_trial, median)", pairs_pwall, trend_pwall)

    out.append("--- trend interpretation rules (a-priori) ---")
    out.append("- oversubscription:")
    out.append("    delta_pct(n=1) < delta_pct(n=2) < delta_pct(n=4) (strict monotonic)")
    out.append(f"    AND max - min >= {FLAT_THRESHOLD_PCT}%")
    out.append("    Reading: prior C_2x4 HPX overhead was amplified by kernel-thread")
    out.append("    oversubscription. Orchestration overhead alone is small.")
    out.append("- orchestration_or_waiter_path:")
    out.append(f"    max - min < {FLAT_THRESHOLD_PCT}%")
    out.append("    Reading: HPX overhead is intrinsic to serving-level orchestration")
    out.append("    or waiter path, not to oversubscription. C_2x4 in the matrix was")
    out.append("    not confounded by thread pressure.")
    out.append("- mixed:")
    out.append("    delta_pct trend is non-monotonic across n_threads = {1, 2, 4}.")
    out.append("    Reading: both effects contribute; cannot cleanly attribute.")
    out.append("- incomplete:")
    out.append("    any thread setting missing or its correctness state is FAIL.")
    out.append("    Reading: do not interpret timing.")
    out.append("")

    out.append("--- caveats ---")
    out.append("- The sweep OVERALL gate is correctness for all three thread settings.")
    out.append("  Timing numbers above are descriptive and do not gate anything.")
    out.append("- Δ % is (hpx_median - std_median) / std_median * 100. Sub-1% deltas")
    out.append("  inside the CV band are not interpretable as 'different'.")
    out.append("- Timing rows for thread settings whose correctness is FAIL or MISSING")
    out.append("  are shown as '<suppressed>'.")
    out.append("- This is still C_2x4: waiter pressure is held constant. The sweep")
    out.append("  reduces kernel-thread oversubscription, not queueing pressure.")
    out.append("- n_threads=1 may use a different ggml/single-thread path. If the")
    out.append("  shrink is concentrated at n=1 only, prefer the 'mixed' or")
    out.append("  'incomplete' reading over a clean 'oversubscription' label.")
    out.append("- If the trend label is 'orchestration_or_waiter_path', the likely")
    out.append("  source is serving / HPX orchestration or waiter-path overhead.")
    out.append("- If the trend label is 'oversubscription', the prior C_2x4 overhead")
    out.append("  was likely amplified by oversubscription.")
    out.append("- This protocol measures THIS serving-bench harness on THIS machine,")
    out.append("  with THIS build, THIS model, THIS prompt, THIS generation length.")
    out.append("  It is not a measurement of llama.cpp performance, not a measurement")
    out.append("  of HPX scheduler quality in general, and not a recommendation for")
    out.append("  or against either backend.")
    out.append("")

    sweep_path = ROOT / "sweep_summary.txt"
    sweep_path.write_text("\n".join(out))

    for s in settings_data:
        print(f"n_threads={s['n_threads']}  present={s['present']}  overall={s['overall']}")
    print(f"trend.total_ms.pool: {trend_total}")
    print(f"trend.total_minus_ttft_ms.pool: {trend_tmt}")
    print(f"trend.ttft_ms.pool: {trend_ttft}")
    print(f"trend.process_wall_ms.per_trial: {trend_pwall}")
    print(f"SWEEP_OVERALL_CORRECTNESS: {'PASS' if sweep_overall_pass else 'FAIL'}")
    print(f"wrote {sweep_path}")
    sys.exit(0 if sweep_overall_pass else 1)


if __name__ == "__main__":
    main()
