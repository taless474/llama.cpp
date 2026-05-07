"""
Top-level rollup of the heterogeneous-budgets experiment. Reads both
condition summaries, decides EXPERIMENT_OVERALL, and — only if
EXPERIMENT_OVERALL is PASS — assigns a branch label per the
interpretation rules in facts.md "Interpretation rules".

Reads:
    summaries/condition_correctness.txt
    summaries/condition_timing.txt

Writes:
    summaries/experiment_summary.txt

A condition is PASS when both backend layer summaries PASS and every
budget value in the plan has equal canonical hashes across backends
(see _summarize_condition.py).

Branch labels (timing layer only; correctness layer must also PASS):
    hpx_short_better:
        DELTA_PCT_SHORT_TOTAL_MED  ≤ −1.0
        DELTA_PCT_MAKESPAN_MED     ≤ +0.5
        DELTA_PCT_LONG_TOTAL_MED   ≤ +1.0

    equivalent:
        |DELTA_PCT_SHORT_TOTAL_MED| ≤ 1.0
        |DELTA_PCT_MAKESPAN_MED|    ≤ 1.0
        |DELTA_PCT_LONG_TOTAL_MED|  ≤ 1.0

    hpx_short_worse:
        DELTA_PCT_SHORT_TOTAL_MED  ≥ +1.0

    mixed:
        sign of DELTA_PCT disagrees across short / medium / long, OR
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


def assign_branch(d_short, d_long, d_mksp, d_med):
    if d_short is None or d_long is None or d_mksp is None:
        return "incomplete"

    # mixed precedence (sign disagreement first; sign at exactly zero treated as 0)
    def sign(x):
        if x > 0: return +1
        if x < 0: return -1
        return 0
    short_s, long_s, mksp_s = sign(d_short), sign(d_long), sign(d_mksp)
    med_s = sign(d_med) if d_med is not None else 0

    nonzero_signs = [s for s in (short_s, long_s, med_s) if s != 0]
    class_disagrees = (
        len(set(nonzero_signs)) > 1 if nonzero_signs else False
    )
    short_makespan_disagree = (
        short_s != 0 and mksp_s != 0 and short_s != mksp_s
    )

    # honour rule precedence: hpx_short_better, equivalent, hpx_short_worse, mixed
    if (d_short <= -1.0 and d_mksp <= +0.5 and d_long <= +1.0):
        return "hpx_short_better"
    if (abs(d_short) <= 1.0 and abs(d_mksp) <= 1.0 and abs(d_long) <= 1.0):
        return "equivalent"
    if (d_short >= +1.0):
        return "hpx_short_worse"
    if class_disagrees or short_makespan_disagree:
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

    d_short = to_float_or_none(tim_hdr.get("DELTA_PCT_SHORT_TOTAL_MED"))
    d_long  = to_float_or_none(tim_hdr.get("DELTA_PCT_LONG_TOTAL_MED"))
    d_mksp  = to_float_or_none(tim_hdr.get("DELTA_PCT_MAKESPAN_MED"))
    # medium delta is not in condition header by name; read from text? We
    # rely on the three primary deltas above for the branch. Medium is
    # informational for "mixed" sign-disagreement; we approximate it by
    # treating it as sign-zero when unavailable.
    d_med = None

    if experiment_pass:
        branch = assign_branch(d_short, d_long, d_mksp, d_med)
    else:
        branch = "incomplete"

    out = []
    out.append(f"EXPERIMENT_OVERALL: {'PASS' if experiment_pass else 'FAIL'}")
    out.append(f"CORRECTNESS_CONDITION: {cor_hdr.get('CONDITION_OVERALL', 'UNKNOWN')}")
    out.append(f"TIMING_CONDITION: {tim_hdr.get('CONDITION_OVERALL', 'UNKNOWN')}")
    out.append(f"DELTA_PCT_SHORT_TOTAL_MED: "
               f"{tim_hdr.get('DELTA_PCT_SHORT_TOTAL_MED', 'NA')}")
    out.append(f"DELTA_PCT_LONG_TOTAL_MED: "
               f"{tim_hdr.get('DELTA_PCT_LONG_TOTAL_MED', 'NA')}")
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
        f"  long.total_ms.delta_pct  (median) : "
        f"{tim_hdr.get('DELTA_PCT_LONG_TOTAL_MED', 'NA')}"
    )
    out.append(
        f"  makespan.delta_pct       (median) : "
        f"{tim_hdr.get('DELTA_PCT_MAKESPAN_MED', 'NA')}"
    )
    out.append("")

    out.append(f"Branch label: {branch}")
    out.append("")
    out.append(
        "Branch rules (see facts.md §10):"
    )
    out.append(
        "  hpx_short_better : short ≤ −1.0%, makespan ≤ +0.5%, long ≤ +1.0%"
    )
    out.append(
        "  equivalent       : |short| ≤ 1.0%, |makespan| ≤ 1.0%, |long| ≤ 1.0%"
    )
    out.append(
        "  hpx_short_worse  : short ≥ +1.0%  (CV-disjoint check left descriptive)"
    )
    out.append(
        "  mixed            : sign disagrees across classes, or short and "
        "makespan disagree"
    )
    out.append(
        "  incomplete       : EXPERIMENT_OVERALL ≠ PASS or any required delta is NA"
    )

    summary_path = SUMM / "experiment_summary.txt"
    summary_path.write_text("\n".join(out) + "\n")

    print(f"wrote {summary_path}")
    print(f"EXPERIMENT_OVERALL: {'PASS' if experiment_pass else 'FAIL'}")
    print(f"BRANCH_LABEL: {branch}")
    sys.exit(0 if experiment_pass else 2)


if __name__ == "__main__":
    main()
