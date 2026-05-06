"""
Phase 2 helper — summarize one (n_threads, layer, backend) group of trials.

Usage:
  python3 _summarize_layer.py \
    --n-threads 1 \
    --layer correctness_trace_on \
    --backend hpx

Reads:
  local/baselines/perf_thread_sweep/_schedule.json
  thread_settings/n<N>/<layer_short>/<backend>/trial_NN/{
    bench.stdout, bench.stderr, bench.exit_code.txt,
    per_repeat.csv, hpx_trace.txt, process_wall_ms.txt
  }   for every scheduled trial in the group (expected 11 trials)

Writes (under the same directory):
  thread_settings/n<N>/<layer_short>/<backend>/all_requests.csv
  thread_settings/n<N>/<layer_short>/<backend>/per_trial_summary.csv
  thread_settings/n<N>/<layer_short>/<backend>/condition_summary.txt

The condition_summary.txt begins with a machine-readable `KEY: VALUE`
block (terminated by `# END HEADER`) so downstream summarizers can parse
PASS/FAIL state without re-applying gates.

Per-trial gates (every layer/backend combo):
  - bench.exit_code.txt exists and == 0
  - per_repeat.csv has n_requests data rows
  - bench.stdout contains aggregate line: 'n_ok=N n_cancelled=0 n_error=0'
  - all rows status == ok
  - all rows n_tokens_generated == 16
  - all rows generated_token_hash == 0x833045f1e2ebf49f
  - bench.stderr contains 'prompt fits: 6 prompt tokens + 16 max_tokens <= 2048 ctx_size'
  - bench.stderr has no '[serving-bench] error'
  - bench.stderr has no 'failed'

Layer/backend-specific gates:

  layer=correctness_trace_on, backend=hpx:
    - exactly one start line  '[serving-bench] hpx_runtime_start_once: starting (os_threads=2)'
    - exactly one ready line  '[serving-bench] engine_hpx ready: n_contexts=2 pool_size=2'
    - exactly one stop line   '[serving-bench] hpx_runtime_stop: stopping'
    - for every i in 0..n_requests-1: exactly one acquire and one release
    - acquire ctx id and release ctx id both in {0..n_contexts-1}
    - release ctx id == acquire ctx id (same-ctx pairing)
    - acquire line precedes release line
    - set of acquire ctx ids equals {0, 1}
    - total filtered HPX trace lines == 27

  layer=correctness_trace_on, backend=std:
    - hpx_trace.txt has zero matching HPX lifecycle/pool lines

  layer=timing_trace_off (both backends):
    - hpx_trace.txt has zero matching HPX lifecycle/pool lines
      (LLAMA_SERVING_BENCH_HPX_TRACE was unset; no events should be emitted)
    - lifecycle counts are NOT required

LAYER_OVERALL is PASS iff every scheduled trial is present and every
trial's per-trial gates pass.

This summarizer does NOT gate timing. Timing aggregates are descriptive
only and are computed from measured_for_timing=true rows so the
thread-setting summarizer can reuse them.
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path
from statistics import mean, median, pstdev

ROOT     = Path("/Users/Ashk/Desktop/HPX/llama-hpx/local/baselines/perf_thread_sweep")
SCHEDULE = ROOT / "_schedule.json"

CANONICAL_HASH            = "0x833045f1e2ebf49f"
EXPECTED_N_TOKENS         = 16
EXPECTED_PROMPT_FITS_LINE = "prompt fits: 6 prompt tokens + 16 max_tokens <= 2048 ctx_size"

LAYER_DIR = {
    "correctness_trace_on": "correctness",
    "timing_trace_off":     "timing",
}

ACQUIRE_RE = re.compile(r"^\[serving-bench\] req\[(\d+)\] acquire ctx=(\d+)$")
RELEASE_RE = re.compile(r"^\[serving-bench\] req\[(\d+)\] release ctx=(\d+)$")
ANY_TRACE_RE = re.compile(
    r"^\[serving-bench\] (?:"
    r"hpx_runtime_start_once|"
    r"engine_hpx ready|"
    r"hpx_runtime_stop|"
    r"req\[\d+\] acquire|"
    r"req\[\d+\] release"
    r")"
)
AGG_TPS_RE = re.compile(r"^\[serving-bench\] wall=([0-9.]+) s\s+agg_tok/s=([0-9.]+)\s*$")


def parse_args():
    p = argparse.ArgumentParser(description="summarize one (n_threads, layer, backend) group")
    p.add_argument("--n-threads", type=int, required=True, choices=[1, 2, 4])
    p.add_argument("--layer", required=True,
                   choices=["correctness_trace_on", "timing_trace_off"])
    p.add_argument("--backend", required=True, choices=["std", "hpx"])
    return p.parse_args()


def load_schedule():
    if not SCHEDULE.exists():
        sys.exit(f"schedule not found: {SCHEDULE}; run _make_schedule.py first")
    return json.loads(SCHEDULE.read_text())


def stats(xs):
    if not xs:
        return None
    n = len(xs)
    s = sorted(xs)
    def percentile(p):
        if n == 1:
            return s[0]
        k = (n - 1) * (p / 100.0)
        lo = int(k)
        hi = min(lo + 1, n - 1)
        frac = k - lo
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


def read_text_safe(p: Path) -> str:
    return p.read_text() if p.exists() else ""


def gate_trial(entry, out_dir):
    """Apply per-trial gates. Returns (checks, parsed_rows, agg_tok_s, process_wall_ms, trial_pass)."""
    checks = []
    rows = []
    agg_tok_s = None
    process_wall_ms = None

    n_req = entry["n_requests"]
    layer = entry["layer"]
    backend = entry["backend"]
    expected_agg_line = f"n_ok={n_req} n_cancelled=0 n_error=0"

    ec_path = out_dir / "bench.exit_code.txt"
    if not ec_path.exists():
        checks.append((f"trial directory present and bench.exit_code.txt exists "
                       f"({out_dir})", False))
        return checks, rows, agg_tok_s, process_wall_ms, False

    try:
        ec = int(ec_path.read_text().strip())
    except ValueError:
        ec = -1
    checks.append(("bench exit code == 0", ec == 0))

    stdout = read_text_safe(out_dir / "bench.stdout")
    stderr = read_text_safe(out_dir / "bench.stderr")

    csv_path = out_dir / "per_repeat.csv"
    if csv_path.exists():
        with csv_path.open() as f:
            for r in csv.DictReader(f):
                rows.append(r)
        rows.sort(key=lambda r: int(r["req_index"]))

    checks.append((f"per_repeat.csv has {n_req} data rows", len(rows) == n_req))
    checks.append((f"stdout contains aggregate line: {expected_agg_line!r}",
                   expected_agg_line in stdout))
    checks.append((f"all {n_req} status == ok",
                   len(rows) == n_req and all(r["status"] == "ok" for r in rows)))
    checks.append((f"all {n_req} n_tokens_generated == {EXPECTED_N_TOKENS}",
                   len(rows) == n_req and all(
                       int(r["n_tokens_generated"]) == EXPECTED_N_TOKENS for r in rows
                   )))
    checks.append((f"all {n_req} generated_token_hash == {CANONICAL_HASH}",
                   len(rows) == n_req and all(
                       r["generated_token_hash"] == CANONICAL_HASH for r in rows
                   )))
    checks.append((f"stderr contains: {EXPECTED_PROMPT_FITS_LINE!r}",
                   EXPECTED_PROMPT_FITS_LINE in stderr))
    checks.append(("stderr has no '[serving-bench] error'",
                   "[serving-bench] error" not in stderr))
    checks.append(("stderr has no 'failed'", "failed" not in stderr))

    for ln in stdout.splitlines():
        m = AGG_TPS_RE.match(ln)
        if m:
            agg_tok_s = float(m.group(2))
            break

    pw_path = out_dir / "process_wall_ms.txt"
    if pw_path.exists():
        first = pw_path.read_text().splitlines()[0] if pw_path.read_text().splitlines() else ""
        try:
            process_wall_ms = float(first.strip())
        except ValueError:
            process_wall_ms = None

    trace_text = read_text_safe(out_dir / "hpx_trace.txt")
    trace_lines = [ln for ln in trace_text.splitlines() if ln.strip()]

    if layer == "correctness_trace_on" and backend == "hpx":
        os_threads  = entry["expected_os_threads"]
        n_contexts  = entry["n_contexts"]
        pool_size   = entry["expected_pool_size"]
        exp_traces  = entry["expected_trace_lines"]

        start_line = f"[serving-bench] hpx_runtime_start_once: starting (os_threads={os_threads})"
        ready_line = f"[serving-bench] engine_hpx ready: n_contexts={n_contexts} pool_size={pool_size}"
        stop_line  = "[serving-bench] hpx_runtime_stop: stopping"

        n_start = sum(1 for ln in trace_lines if ln == start_line)
        n_ready = sum(1 for ln in trace_lines if ln == ready_line)
        n_stop  = sum(1 for ln in trace_lines if ln == stop_line)

        checks.append((f"exactly one start (os_threads={os_threads})", n_start == 1))
        checks.append((
            f"exactly one ready (n_contexts={n_contexts} pool_size={pool_size})",
            n_ready == 1,
        ))
        checks.append(("exactly one stop", n_stop == 1))

        acquire_idx = {}
        release_idx = {}
        for pos, ln in enumerate(trace_lines):
            m_a = ACQUIRE_RE.match(ln)
            m_r = RELEASE_RE.match(ln)
            if m_a:
                acquire_idx.setdefault(int(m_a.group(1)), []).append(
                    (pos, int(m_a.group(2)))
                )
            elif m_r:
                release_idx.setdefault(int(m_r.group(1)), []).append(
                    (pos, int(m_r.group(2)))
                )

        valid_ctx = set(range(n_contexts))
        ctx_seen = set()
        pairing_ok = True
        for i in range(n_req):
            a = acquire_idx.get(i, [])
            r = release_idx.get(i, [])
            if len(a) != 1 or len(r) != 1:
                pairing_ok = False
                continue
            if a[0][1] not in valid_ctx or r[0][1] not in valid_ctx:
                pairing_ok = False
                continue
            if a[0][1] != r[0][1]:
                pairing_ok = False
                continue
            if a[0][0] >= r[0][0]:
                pairing_ok = False
                continue
            ctx_seen.add(a[0][1])

        checks.append((
            f"acquire/release pairing valid for all {n_req} req indices "
            f"(same-ctx, ordered, ctx in {{0..{n_contexts - 1}}})",
            pairing_ok,
        ))
        checks.append((
            f"set of acquire ctx ids equals {{0..{n_contexts - 1}}}",
            ctx_seen == valid_ctx,
        ))
        checks.append((
            f"total filtered HPX trace lines == {exp_traces}",
            len(trace_lines) == exp_traces,
        ))
    elif layer == "correctness_trace_on" and backend == "std":
        n_match = sum(1 for ln in trace_lines if ANY_TRACE_RE.match(ln))
        checks.append((
            "std hpx_trace.txt has zero HPX lifecycle/pool lines",
            n_match == 0,
        ))
    elif layer == "timing_trace_off":
        n_match = sum(1 for ln in trace_lines if ANY_TRACE_RE.match(ln))
        checks.append((
            "trace-off layer hpx_trace.txt has zero HPX lifecycle/pool lines",
            n_match == 0,
        ))

    trial_pass = all(ok for _, ok in checks)
    return checks, rows, agg_tok_s, process_wall_ms, trial_pass


def main():
    args = parse_args()
    schedule = load_schedule()
    entries = [
        e for e in schedule["schedule"]
        if e["n_threads"] == args.n_threads
        and e["layer"]   == args.layer
        and e["backend"] == args.backend
    ]
    if not entries:
        sys.exit(
            f"no schedule entries match n_threads={args.n_threads} "
            f"layer={args.layer} backend={args.backend}"
        )
    entries.sort(key=lambda e: e["trial_index"])

    expected_trials = schedule["trials_per_backend"]

    layer_short = LAYER_DIR[args.layer]
    group_dir = ROOT / "thread_settings" / f"n{args.n_threads}" / layer_short / args.backend
    group_dir.mkdir(parents=True, exist_ok=True)

    sample = entries[0]
    cell_id = sample["cell_id"]
    n_req = sample["n_requests"]
    n_contexts = sample["n_contexts"]
    n_concurrent = sample["n_concurrent"]
    expected_os_threads = sample["expected_os_threads"]
    expected_pool_size = sample["expected_pool_size"]
    expected_trace_lines = sample["expected_trace_lines"]

    per_trial = []
    n_present = 0
    n_pass = 0

    for entry in entries:
        out_dir = ROOT / entry["output_dir"]
        ec_path = out_dir / "bench.exit_code.txt"
        present = ec_path.exists()
        if present:
            n_present += 1

        checks, rows, agg_tok_s, process_wall_ms, trial_pass = gate_trial(
            entry, out_dir
        )
        if trial_pass:
            n_pass += 1

        ttft   = [float(r["ttft_ms"])             for r in rows] if rows else []
        total  = [float(r["total_ms"])            for r in rows] if rows else []
        tps    = [float(r["tokens_per_second"])   for r in rows] if rows else []
        tmt    = [float(r["total_minus_ttft_ms"]) for r in rows] if rows else []

        per_trial.append({
            "entry":      entry,
            "out_dir":    out_dir,
            "present":    present,
            "checks":     checks,
            "rows":       rows,
            "trial_pass": trial_pass,
            "agg_tok_s":  agg_tok_s,
            "process_wall_ms": process_wall_ms,
            "ttft":  ttft,
            "total": total,
            "tps":   tps,
            "tmt":   tmt,
        })

    layer_overall = (n_present == len(entries)) and (n_pass == len(entries))

    all_req_path = group_dir / "all_requests.csv"
    with all_req_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "global_index", "trial_index", "measured_for_timing", "trial_pass",
            "req_index", "status", "n_tokens_generated", "generated_token_hash",
            "ttft_ms", "total_ms", "tokens_per_second", "total_minus_ttft_ms",
        ])
        for pt in per_trial:
            e = pt["entry"]
            for r in pt["rows"]:
                w.writerow([
                    e["global_index"], e["trial_index"],
                    str(e["measured_for_timing"]).lower(),
                    str(pt["trial_pass"]).lower(),
                    r["req_index"], r["status"],
                    r["n_tokens_generated"], r["generated_token_hash"],
                    r["ttft_ms"], r["total_ms"],
                    r["tokens_per_second"], r["total_minus_ttft_ms"],
                ])

    pts_path = group_dir / "per_trial_summary.csv"
    with pts_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "global_index", "trial_index", "measured_for_timing", "trial_pass",
            "n_rows_parsed", "process_wall_ms", "agg_tok_s",
            "ttft_median_ms", "ttft_p95_ms",
            "total_median_ms", "total_p95_ms",
            "total_minus_ttft_median_ms", "total_minus_ttft_p95_ms",
            "tps_median",
        ])
        def _med(xs):
            return f"{median(xs):.6f}" if xs else ""
        def _p95(xs):
            if not xs:
                return ""
            s = sorted(xs)
            if len(s) == 1:
                return f"{s[0]:.6f}"
            k = (len(s) - 1) * 0.95
            lo = int(k); hi = min(lo + 1, len(s) - 1); frac = k - lo
            return f"{s[lo] * (1.0 - frac) + s[hi] * frac:.6f}"
        for pt in per_trial:
            e = pt["entry"]
            w.writerow([
                e["global_index"], e["trial_index"],
                str(e["measured_for_timing"]).lower(),
                str(pt["trial_pass"]).lower(),
                len(pt["rows"]),
                f"{pt['process_wall_ms']:.6f}" if pt["process_wall_ms"] is not None else "",
                f"{pt['agg_tok_s']:.6f}" if pt["agg_tok_s"] is not None else "",
                _med(pt["ttft"]),  _p95(pt["ttft"]),
                _med(pt["total"]), _p95(pt["total"]),
                _med(pt["tmt"]),   _p95(pt["tmt"]),
                _med(pt["tps"]),
            ])

    measured_trials = [pt for pt in per_trial if pt["entry"]["measured_for_timing"]]
    pool_total = []
    pool_ttft  = []
    pool_tmt   = []
    pool_tps   = []
    per_trial_wall = []
    per_trial_agg  = []
    per_trial_total_med = []
    per_trial_tmt_med   = []
    for pt in measured_trials:
        pool_total.extend(pt["total"])
        pool_ttft.extend(pt["ttft"])
        pool_tmt.extend(pt["tmt"])
        pool_tps.extend(pt["tps"])
        if pt["process_wall_ms"] is not None:
            per_trial_wall.append(pt["process_wall_ms"])
        if pt["agg_tok_s"] is not None:
            per_trial_agg.append(pt["agg_tok_s"])
        if pt["total"]:
            per_trial_total_med.append(median(pt["total"]))
        if pt["tmt"]:
            per_trial_tmt_med.append(median(pt["tmt"]))

    s_total = stats(pool_total)
    s_ttft  = stats(pool_ttft)
    s_tmt   = stats(pool_tmt)
    s_tps   = stats(pool_tps)
    s_wall  = stats(per_trial_wall)
    s_agg   = stats(per_trial_agg)
    s_pt_total_med = stats(per_trial_total_med)
    s_pt_tmt_med   = stats(per_trial_tmt_med)

    out = []
    out.append(f"# layer summary: n_threads={args.n_threads} / {args.layer} / {args.backend}")
    out.append("# machine-readable header (KEY: VALUE); ends at '# END HEADER'")
    out.append(f"cell_id: {cell_id}")
    out.append(f"n_threads: {args.n_threads}")
    out.append(f"layer: {args.layer}")
    out.append(f"backend: {args.backend}")
    out.append(f"n_contexts: {n_contexts}")
    out.append(f"n_concurrent: {n_concurrent}")
    out.append(f"n_requests: {n_req}")
    out.append(f"expected_os_threads: {expected_os_threads}")
    out.append(f"expected_pool_size: {expected_pool_size}")
    out.append(f"expected_trace_lines: {expected_trace_lines}")
    out.append(f"trials_expected: {expected_trials}")
    out.append(f"trials_present: {n_present}")
    out.append(f"trials_pass: {n_pass}")
    out.append(f"measured_trials_present: {len(measured_trials)}")
    out.append(f"measured_trials_pass: "
               f"{sum(1 for pt in measured_trials if pt['trial_pass'])}")

    def _emit_stats(prefix, s):
        if s is None:
            out.append(f"{prefix}.n: 0")
            return
        out.append(f"{prefix}.n: {s['n']}")
        out.append(f"{prefix}.min: {s['min']:.6f}")
        out.append(f"{prefix}.max: {s['max']:.6f}")
        out.append(f"{prefix}.mean: {s['mean']:.6f}")
        out.append(f"{prefix}.median: {s['median']:.6f}")
        out.append(f"{prefix}.p95: {s['p95']:.6f}")
        out.append(f"{prefix}.p99: {s['p99']:.6f}")
        out.append(f"{prefix}.stdev_pop: {s['stdev_pop']:.6f}")
        out.append(f"{prefix}.cv: {s['cv']:.6f}")

    _emit_stats("pool.total_ms",            s_total)
    _emit_stats("pool.ttft_ms",             s_ttft)
    _emit_stats("pool.total_minus_ttft_ms", s_tmt)
    _emit_stats("pool.tokens_per_second",   s_tps)
    _emit_stats("per_trial.process_wall_ms",     s_wall)
    _emit_stats("per_trial.agg_tok_s",           s_agg)
    _emit_stats("per_trial.median_total_ms",     s_pt_total_med)
    _emit_stats("per_trial.median_total_minus_ttft_ms", s_pt_tmt_med)

    out.append(f"LAYER_OVERALL: {'PASS' if layer_overall else 'FAIL'}")
    out.append("# END HEADER")
    out.append("")

    out.append(f"=== layer summary: n_threads={args.n_threads} / "
               f"{args.layer} / {args.backend} ===")
    out.append(f"cell_id={cell_id}")
    out.append("")
    out.append(f"trials_expected:        {expected_trials}")
    out.append(f"trials_present:         {n_present}")
    out.append(f"trials_pass:            {n_pass}")
    out.append(f"measured_trials_present:{len(measured_trials)}")
    out.append(f"measured_trials_pass:   "
               f"{sum(1 for pt in measured_trials if pt['trial_pass'])}")
    out.append("")
    out.append("--- per-trial gate state ---")
    for pt in per_trial:
        e = pt["entry"]
        flag = "PASS" if pt["trial_pass"] else "FAIL"
        first_fail = next(
            (label for label, ok in pt["checks"] if not ok),
            "",
        )
        out.append(
            f"[{flag}] gidx={e['global_index']:4d} "
            f"trial={e['trial_index']:2d} "
            f"measured={str(e['measured_for_timing']):5s} "
            f"rows={len(pt['rows']):2d} "
            f"first_fail={first_fail!r}"
        )
    out.append("")

    def _fmt_stats(label, s, unit):
        if s is None:
            return f"{label}: n=0"
        return (
            f"{label}: n={s['n']} min={s['min']:.3f}{unit} "
            f"max={s['max']:.3f}{unit} mean={s['mean']:.3f}{unit} "
            f"median={s['median']:.3f}{unit} p95={s['p95']:.3f}{unit} "
            f"p99={s['p99']:.3f}{unit} stdev_pop={s['stdev_pop']:.3f}{unit} "
            f"cv={s['cv']:.4f}"
        )

    out.append("--- descriptive timing (measured_for_timing=true rows only) ---")
    out.append(_fmt_stats("pool total_ms             ", s_total, "ms"))
    out.append(_fmt_stats("pool ttft_ms              ", s_ttft,  "ms"))
    out.append(_fmt_stats("pool total_minus_ttft_ms  ", s_tmt,   "ms"))
    out.append(_fmt_stats("pool tokens_per_second    ", s_tps,   "tps"))
    out.append(_fmt_stats("per_trial process_wall_ms ", s_wall,  "ms"))
    out.append(_fmt_stats("per_trial agg_tok_s       ", s_agg,   "tps"))
    out.append(_fmt_stats("per_trial median total_ms ", s_pt_total_med, "ms"))
    out.append(_fmt_stats("per_trial median tmt_ms   ", s_pt_tmt_med,   "ms"))
    out.append("")

    out.append(f"LAYER_OVERALL: {'PASS' if layer_overall else 'FAIL'}")
    out.append("")
    out.append("--- caveats ---")
    out.append("- Timing aggregates are descriptive; gates do not depend on them.")
    out.append("- Pool stats include all rows from measured_for_timing=true trials")
    out.append("  (samples within a trial share runtime/cache state and are NOT iid).")
    out.append("- Per-trial stats use one number per trial (small-n; iid across trials).")
    out.append("- HPX expected_os_threads/pool_size are HPX runtime carriers and are")
    out.append("  independent of the n_threads sweep variable (which controls llama")
    out.append("  kernel threads).")
    out.append("- The thread-setting summarizer (_summarize_thread_setting.py) reads")
    out.append("  this file's machine-readable header and gates the n_threads correctness.")
    out.append("")

    summary_path = group_dir / "condition_summary.txt"
    summary_path.write_text("\n".join(out))

    print(f"n_threads={args.n_threads}  layer={args.layer}  backend={args.backend}")
    print(f"trials_expected={expected_trials}  trials_present={n_present}  trials_pass={n_pass}")
    print(f"LAYER_OVERALL: {'PASS' if layer_overall else 'FAIL'}")
    print(f"wrote {all_req_path}")
    print(f"wrote {pts_path}")
    print(f"wrote {summary_path}")
    sys.exit(0 if layer_overall else 1)


if __name__ == "__main__":
    main()
