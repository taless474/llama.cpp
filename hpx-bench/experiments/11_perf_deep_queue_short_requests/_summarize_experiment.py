"""
Top-level rollup of the deep-queue short-request stress experiment.
Reads both condition summaries, decides EXPERIMENT_OVERALL, and — only
if EXPERIMENT_OVERALL is PASS — assigns a branch label per the
interpretation rules in facts.md "Interpretation rules".

Reads:
    summaries/condition_correctness.txt
    summaries/condition_timing.txt

Writes:
    summaries/experiment_summary.txt

Branch labels (timing layer only; correctness layer must also PASS):
    hpx_better:
        DELTA_PCT_SHORT_TOTAL_MED  ≤ −1.0
        DELTA_PCT_MAKESPAN_MED     ≤ +0.5
        DELTA_PCT_SHORT_TOTAL_P99  ≤ −1.0

    equivalent:
        |DELTA_PCT_SHORT_TOTAL_MED| ≤ 1.0
        |DELTA_PCT_MAKESPAN_MED|    ≤ 1.0
        |DELTA_PCT_SHORT_TOTAL_P99| ≤ 2.0

    hpx_worse:
        DELTA_PCT_SHORT_TOTAL_MED  ≥ +1.0  OR
        DELTA_PCT_SHORT_TOTAL_P99  ≥ +2.0

    mixed:
        sign of DELTA_PCT disagrees between median and p99, OR
        DELTA_PCT_SHORT_TOTAL_MED and DELTA_PCT_MAKESPAN_MED disagree

    incomplete:
        EXPERIMENT_OVERALL is not PASS, OR any required delta is NA.
"""
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
SUMM    = EXP_DIR / "summaries"


def parse_header(path: Path):
    if not path.exists():
        return None
    out = {}
    for line in path.read_text().splitlines():
        if line.strip() == "# END HEADER":
            break
        if ":" in line:
            k, _, v = line.partition(":")
            out[k.strip()] = v.strip()
    return out


def to_float_or_none(s):
    if s is None or s == "" or s == "NA":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def assign_branch(d_med, d_p99, d_mksp):
    if d_med is None or d_p99 is None or d_mksp is None:
        return "incomplete"

    def sign(x):
        if x > 0: return +1
        if x < 0: return -1
        return 0

    # rule precedence: hpx_better > equivalent > hpx_worse > mixed
    if d_med <= -1.0 and d_mksp <= +0.5 and d_p99 <= -1.0:
        return "hpx_better"
    if abs(d_med) <= 1.0 and abs(d_mksp) <= 1.0 and abs(d_p99) <= 2.0:
        return "equivalent"
    if d_med >= +1.0 or d_p99 >= +2.0:
        return "hpx_worse"

    med_s, p99_s, mksp_s = sign(d_med), sign(d_p99), sign(d_mksp)
    if med_s != 0 and p99_s != 0 and med_s != p99_s:
        return "mixed"
    if med_s != 0 and mksp_s != 0 and med_s != mksp_s:
        return "mixed"
    return "mixed"


def main():
    cor_path = SUMM / "condition_correctness.txt"
    tim_path = SUMM / "condition_timing.txt"

    cor_hdr = parse_header(cor_path)
    tim_hdr = parse_header(tim_path)

    if cor_hdr is None:
        sys.exit(f"missing condition summary: {cor_path}")
    if tim_hdr is None:
        sys.exit(f"missing condition summary: {tim_path}")

    cor_pass = cor_hdr.get("CONDITION_OVERALL") == "PASS"
    tim_pass = tim_hdr.get("CONDITION_OVERALL") == "PASS"
    experiment_pass = cor_pass and tim_pass

    d_med  = to_float_or_none(tim_hdr.get("DELTA_PCT_SHORT_TOTAL_MED"))
    d_p99  = to_float_or_none(tim_hdr.get("DELTA_PCT_SHORT_TOTAL_P99"))
    d_mksp = to_float_or_none(tim_hdr.get("DELTA_PCT_MAKESPAN_MED"))

    if experiment_pass:
        branch = assign_branch(d_med, d_p99, d_mksp)
    else:
        branch = "incomplete"

    out = []
    out.append(f"EXPERIMENT_OVERALL: {'PASS' if experiment_pass else 'FAIL'}")
    out.append(f"CORRECTNESS_CONDITION: {cor_hdr.get('CONDITION_OVERALL', 'UNKNOWN')}")
    out.append(f"TIMING_CONDITION: {tim_hdr.get('CONDITION_OVERALL', 'UNKNOWN')}")
    out.append(f"DELTA_PCT_SHORT_TOTAL_MED: "
               f"{tim_hdr.get('DELTA_PCT_SHORT_TOTAL_MED', 'NA')}")
    out.append(f"DELTA_PCT_SHORT_TOTAL_P90: "
               f"{tim_hdr.get('DELTA_PCT_SHORT_TOTAL_P90', 'NA')}")
    out.append(f"DELTA_PCT_SHORT_TOTAL_P95: "
               f"{tim_hdr.get('DELTA_PCT_SHORT_TOTAL_P95', 'NA')}")
    out.append(f"DELTA_PCT_SHORT_TOTAL_P99: "
               f"{tim_hdr.get('DELTA_PCT_SHORT_TOTAL_P99', 'NA')}")
    out.append(f"DELTA_PCT_MAKESPAN_MED: "
               f"{tim_hdr.get('DELTA_PCT_MAKESPAN_MED', 'NA')}")
    out.append(f"BRANCH_LABEL: {branch}")
    out.append("# END HEADER")
    out.append("")

    out.append("Condition summaries:")
    out.append(
        f"  correctness_trace_on : {cor_hdr.get('CONDITION_OVERALL', 'UNKNOWN')}  "
        f"(std layer: {cor_hdr.get('STD_LAYER_OVERALL', '?')}, "
        f"hpx layer: {cor_hdr.get('HPX_LAYER_OVERALL', '?')}, "
        f"hash equality: {cor_hdr.get('HASH_EQUALITY', '?')})"
    )
    out.append(
        f"  timing_trace_off     : {tim_hdr.get('CONDITION_OVERALL', 'UNKNOWN')}  "
        f"(std layer: {tim_hdr.get('STD_LAYER_OVERALL', '?')}, "
        f"hpx layer: {tim_hdr.get('HPX_LAYER_OVERALL', '?')}, "
        f"hash equality: {tim_hdr.get('HASH_EQUALITY', '?')})"
    )
    out.append("")

    out.append("Headline deltas (timing layer):")
    out.append(
        f"  short.total_ms.delta_pct (median) : "
        f"{tim_hdr.get('DELTA_PCT_SHORT_TOTAL_MED', 'NA')}"
    )
    out.append(
        f"  short.total_ms.delta_pct (p90)    : "
        f"{tim_hdr.get('DELTA_PCT_SHORT_TOTAL_P90', 'NA')}"
    )
    out.append(
        f"  short.total_ms.delta_pct (p95)    : "
        f"{tim_hdr.get('DELTA_PCT_SHORT_TOTAL_P95', 'NA')}"
    )
    out.append(
        f"  short.total_ms.delta_pct (p99)    : "
        f"{tim_hdr.get('DELTA_PCT_SHORT_TOTAL_P99', 'NA')}"
    )
    out.append(
        f"  makespan.delta_pct       (median) : "
        f"{tim_hdr.get('DELTA_PCT_MAKESPAN_MED', 'NA')}"
    )
    out.append("")

    out.append(f"Branch label: {branch}")
    out.append("")
    out.append("Branch rules (see facts.md \"Interpretation rules\"):")
    out.append("  hpx_better : short.med ≤ −1.0%, makespan.med ≤ +0.5%, short.p99 ≤ −1.0%")
    out.append("  equivalent : |short.med| ≤ 1.0%, |makespan.med| ≤ 1.0%, |short.p99| ≤ 2.0%")
    out.append("  hpx_worse  : short.med ≥ +1.0% OR short.p99 ≥ +2.0%")
    out.append("  mixed      : sign disagrees (short.med vs short.p99) or (short.med vs makespan)")
    out.append("  incomplete : EXPERIMENT_OVERALL ≠ PASS or any required delta is NA")

    summary_path = SUMM / "experiment_summary.txt"
    summary_path.write_text("\n".join(out) + "\n")

    print(f"wrote {summary_path}")
    print(f"EXPERIMENT_OVERALL: {'PASS' if experiment_pass else 'FAIL'}")
    print(f"BRANCH_LABEL: {branch}")
    sys.exit(0 if experiment_pass else 2)


if __name__ == "__main__":
    main()
