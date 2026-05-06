"""
Phase 2 helper — write the cross-cell matrix summary.

Usage:
  python3 _summarize_matrix.py

Reads (one per cell):
  conditions/A_1x1/condition_comparison.txt
  conditions/B_2x2/condition_comparison.txt
  conditions/C_2x4/condition_comparison.txt
  conditions/D_4x4/condition_comparison.txt

Each file begins with a machine-readable `KEY: VALUE` block produced by
_summarize_condition.py and terminated by `# END HEADER`. We parse:

  cell_id, n_contexts, n_concurrent, n_requests
  correctness_trace_on.std / .hpx, timing_trace_off.std / .hpx
  CONDITION_OVERALL_CORRECTNESS
  metric.<name>.<view>.std.median / mean / p95 / p99 / stdev_pop / cv
  metric.<name>.<view>.hpx.median / mean / p95 / p99 / stdev_pop / cv
  metric.<name>.<view>.delta_median / delta_pct

If a cell's condition_comparison.txt is missing, that cell is reported
MISSING. If correctness for any cell is FAIL, that cell's timing rows
are suppressed (replaced with '<suppressed>') in the cross-cell tables.

The matrix-level OVERALL gate is correctness for ALL four cells. Timing
deltas are descriptive and do not gate.

Writes:
  matrix_summary.txt
"""
import sys
from pathlib import Path

ROOT = Path("/Users/Ashk/Desktop/HPX/llama-hpx/local/baselines/perf_hpx_vs_std_matrix")
CELLS = ["A_1x1", "B_2x2", "C_2x4", "D_4x4"]


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


def main():
    cells_data = []
    for cid in CELLS:
        cmp_path = ROOT / "conditions" / cid / "condition_comparison.txt"
        h = parse_header(cmp_path)
        cells_data.append({
            "cell_id":   cid,
            "path":      cmp_path,
            "header":    h,
            "present":   bool(h),
            "overall":   h.get("CONDITION_OVERALL_CORRECTNESS"),
            "n_ctx":     inum(h, "n_contexts"),
            "n_conc":    inum(h, "n_concurrent"),
            "n_req":     inum(h, "n_requests"),
            "c_std":     h.get("correctness_trace_on.std"),
            "c_hpx":     h.get("correctness_trace_on.hpx"),
            "t_std":     h.get("timing_trace_off.std"),
            "t_hpx":     h.get("timing_trace_off.hpx"),
        })

    matrix_overall_pass = all(c["overall"] == "PASS" for c in cells_data)

    out = []
    out.append("# matrix summary (cross-cell HPX-vs-std performance matrix)")
    out.append("# machine-readable header (KEY: VALUE); ends at '# END HEADER'")
    for c in cells_data:
        out.append(f"cell.{c['cell_id']}.present: {str(c['present']).lower()}")
        out.append(f"cell.{c['cell_id']}.correctness_overall: {c['overall'] or 'MISSING'}")
        out.append(f"cell.{c['cell_id']}.correctness_trace_on.std: {c['c_std'] or 'MISSING'}")
        out.append(f"cell.{c['cell_id']}.correctness_trace_on.hpx: {c['c_hpx'] or 'MISSING'}")
        out.append(f"cell.{c['cell_id']}.timing_trace_off.std: {c['t_std'] or 'MISSING'}")
        out.append(f"cell.{c['cell_id']}.timing_trace_off.hpx: {c['t_hpx'] or 'MISSING'}")
    out.append(f"MATRIX_OVERALL_CORRECTNESS: {'PASS' if matrix_overall_pass else 'FAIL'}")
    out.append("# END HEADER")
    out.append("")

    out.append("=== HPX-vs-std performance matrix — cross-cell summary ===")
    out.append("")
    out.append("--- correctness gate by cell ---")
    out.append("| cell  | n_ctx | n_conc | n_req | c_std | c_hpx | t_std | t_hpx | OVERALL |")
    for c in cells_data:
        if not c["present"]:
            out.append(
                f"| {c['cell_id']} | -     | -      | -     | -     | -     | -     | -     | MISSING |"
            )
            continue
        out.append(
            f"| {c['cell_id']} | "
            f"{c['n_ctx']:5d} | {c['n_conc']:6d} | {c['n_req']:5d} | "
            f"{c['c_std']:5s} | {c['c_hpx']:5s} | "
            f"{c['t_std']:5s} | {c['t_hpx']:5s} | "
            f"{c['overall']:7s} |"
        )
    out.append("")
    out.append(
        f"MATRIX_OVERALL_CORRECTNESS: {'PASS' if matrix_overall_pass else 'FAIL'}"
    )
    out.append("")

    def _metric_table(title, key, view, unit):
        out.append(f"--- {title} (view: {view}; std vs hpx; Δ on median) ---")
        out.append(
            "| cell  | n_ctx | n_conc | n_req | OK?   | "
            "std median       | hpx median       | "
            "Δ                | Δ %     | std CV  | hpx CV  |"
        )
        for c in cells_data:
            if not c["present"]:
                out.append(
                    f"| {c['cell_id']} | -     | -      | -     | -     | "
                    f"-                | -                | "
                    f"-                | -       | -       | -       |"
                )
                continue
            ok = c["overall"] == "PASS"
            ok_str = "yes" if ok else "no"
            std_med = fnum(c["header"], f"metric.{key}.{view}.std.median")
            hpx_med = fnum(c["header"], f"metric.{key}.{view}.hpx.median")
            std_cv  = fnum(c["header"], f"metric.{key}.{view}.std.cv")
            hpx_cv  = fnum(c["header"], f"metric.{key}.{view}.hpx.cv")
            d       = fnum(c["header"], f"metric.{key}.{view}.delta_median")
            dpc     = fnum(c["header"], f"metric.{key}.{view}.delta_pct")
            if not ok or std_med is None or hpx_med is None:
                out.append(
                    f"| {c['cell_id']} | "
                    f"{c['n_ctx']:5d} | {c['n_conc']:6d} | {c['n_req']:5d} | "
                    f"{ok_str:5s} | "
                    f"{'<suppressed>':16s} | {'<suppressed>':16s} | "
                    f"{'<suppressed>':16s} | {'-':7s} | "
                    f"{'-':7s} | {'-':7s} |"
                )
                continue
            out.append(
                f"| {c['cell_id']} | "
                f"{c['n_ctx']:5d} | {c['n_conc']:6d} | {c['n_req']:5d} | "
                f"{ok_str:5s} | "
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

    out.append("--- caveats ---")
    out.append("- The matrix OVERALL gate is correctness for all four cells. Timing")
    out.append("  numbers above are descriptive and do not gate anything.")
    out.append("- Cross-cell deltas (e.g. C is slower than B) reflect oversubscription")
    out.append("  shape, not anything HPX-vs-std. Only within-cell std-vs-hpx deltas")
    out.append("  speak to the protocol's question.")
    out.append("- Δ % is (hpx_median - std_median) / std_median * 100. Sub-1% deltas")
    out.append("  inside the CV band are not interpretable as 'different'.")
    out.append("- Timing rows for cells whose correctness is FAIL or MISSING are")
    out.append("  shown as '<suppressed>' to keep readers from interpreting timing")
    out.append("  over an uncertain correctness state.")
    out.append("- This protocol measures the HPX-vs-std overhead of THIS serving-bench")
    out.append("  harness on THIS machine, with THIS build, THIS model, THIS prompt,")
    out.append("  THIS generation length. It is not a measurement of llama.cpp")
    out.append("  performance, not a measurement of HPX scheduler quality in general,")
    out.append("  and not a recommendation for or against either backend.")
    out.append("")

    matrix_path = ROOT / "matrix_summary.txt"
    matrix_path.write_text("\n".join(out))

    # ---- console echo ----
    for c in cells_data:
        print(f"cell={c['cell_id']}  present={c['present']}  overall={c['overall']}")
    print(f"MATRIX_OVERALL_CORRECTNESS: {'PASS' if matrix_overall_pass else 'FAIL'}")
    print(f"wrote {matrix_path}")
    sys.exit(0 if matrix_overall_pass else 1)


if __name__ == "__main__":
    main()
