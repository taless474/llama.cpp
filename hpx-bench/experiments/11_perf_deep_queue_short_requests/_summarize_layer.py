"""
Summarize one (layer, backend) group of trials for the deep-queue
short-request stress experiment.

Usage:
    python3 _summarize_layer.py --layer correctness_trace_on --backend std
    python3 _summarize_layer.py --layer correctness_trace_on --backend hpx
    python3 _summarize_layer.py --layer timing_trace_off    --backend std
    python3 _summarize_layer.py --layer timing_trace_off    --backend hpx

Reads (relative to the experiment dir):
    runs/_schedule.json
    runs/<layer_short>/<backend>/trial_NN/{
        bench.stdout, bench.stderr, bench.exit_code.txt,
        per_repeat.csv, hpx_trace.txt, process_wall_ms.txt
    }

Writes:
    summaries/layer_<layer_short>_<backend>.txt
    summaries/layer_<layer_short>_<backend>_per_trial.csv
    summaries/layer_<layer_short>_<backend>_all_requests.csv

The summary file begins with a machine-readable KEY: VALUE block
terminated by `# END HEADER`.

Per-trial gates (completion-order-safe):
    bench.exit_code == 0
    per_repeat.csv has 200 data rows
    bench.stdout aggregate line: "n_ok=200 n_cancelled=0 n_error=0"
    all rows status == ok
    bench.stderr contains  "prompt fits: 6 prompt tokens + 8 max_tokens <= 2048 ctx_size"
    bench.stderr contains no "[serving-bench] error"
    bench.stderr contains no "failed"

    multiset(n_tokens_generated over the 200 rows) == {8: 200}
        A multiset mismatch fails the trial. The harness output line
        does not print the original request_index, so a shortened row
        cannot be mapped back to a specific plan slot.

    per-budget hash consistency within a trial:
        all 200 rows share one generated_token_hash. Restricted to
        budgets in set(plan) = {8}; rows whose n_tokens_generated is
        not 8 only appear when the multiset gate above already fails.

Layer-specific gates: see facts.md "Layer-specific gates".

LAYER_OVERALL is PASS iff every scheduled trial is present and every
trial passes its applicable gates.
"""
import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median, pstdev

EXP_DIR  = Path(__file__).resolve().parent
RUNS     = EXP_DIR / "runs"
SUMM     = EXP_DIR / "summaries"
SCHEDULE = RUNS / "_schedule.json"

LAYER_DIR = {
    "correctness_trace_on": "correctness",
    "timing_trace_off":     "timing",
}

START_RE = re.compile(
    r"^\[serving-bench\] hpx_runtime_start_once: starting "
    r"\(os_threads=(\d+)\)\s*$"
)
READY_RE = re.compile(
    r"^\[serving-bench\] engine_hpx ready: "
    r"n_contexts=(\d+) pool_size=(\d+)\s*$"
)
STOP_RE  = re.compile(r"^\[serving-bench\] hpx_runtime_stop: stopping")
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
AGG_RE        = re.compile(
    r"^\[serving-bench\] n_ok=(\d+) n_cancelled=(\d+) n_error=(\d+)\s*$"
)
PROMPT_FIT_RE = re.compile(
    r"^\[serving-bench\] prompt fits: 6 prompt tokens \+ 8 max_tokens "
    r"<= 2048 ctx_size\s*$"
)
WALL_RE = re.compile(
    r"^\[serving-bench\] wall=([0-9.]+) s\s+agg_tok/s=([0-9.]+)\s*$"
)


def parse_args():
    p = argparse.ArgumentParser(
        description="summarize one (layer, backend) group of trials"
    )
    p.add_argument("--layer",   required=True,
                   choices=["correctness_trace_on", "timing_trace_off"])
    p.add_argument("--backend", required=True, choices=["std", "hpx"])
    return p.parse_args()


def load_schedule():
    if not SCHEDULE.exists():
        sys.exit(f"schedule not found: {SCHEDULE}; run _make_schedule.py first")
    return json.loads(SCHEDULE.read_text())


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


def fmt_stats(label, st):
    if st is None or st["n"] == 0:
        return f"  {label:36s} n=0"
    return (
        f"  {label:36s} n={st['n']:5d}  "
        f"p50={st['p50']:.3f}  p90={st['p90']:.3f}  p95={st['p95']:.3f}  "
        f"p99={st['p99']:.3f}  cv={st['cv']:.4f}"
    )


def gate_trial(entry, trial_dir, plan):
    layer    = entry["layer"]
    backend  = entry["backend"]
    n_req    = entry["n_requests"]
    expected_trace_lines = entry["expected_trace_lines"]
    plan_counter = Counter(plan)

    out = {
        "present":   trial_dir.exists(),
        "gates":     [],
        "rows":      [],
        "wall_s":    None,
        "agg_tps":   None,
        "process_wall_ms": None,
        "n_tok_counter":         Counter(),
        "hash_by_budget":        {},
        "n_tok_multiset_delta":  {},
    }

    if not out["present"]:
        out["gates"].append(("trial_dir_present", False, str(trial_dir)))
        return out

    ec_path  = trial_dir / "bench.exit_code.txt"
    out_path = trial_dir / "bench.stdout"
    err_path = trial_dir / "bench.stderr"
    csv_path = trial_dir / "per_repeat.csv"
    trc_path = trial_dir / "hpx_trace.txt"
    wm_path  = trial_dir / "process_wall_ms.txt"

    for p in (ec_path, out_path, err_path, csv_path, trc_path):
        if not p.exists():
            out["gates"].append((f"{p.name}_exists", False, str(p)))
            return out

    exit_code = int(ec_path.read_text().strip())
    out["gates"].append(("exit_code_zero", exit_code == 0, f"exit_code={exit_code}"))

    stdout = out_path.read_text()
    stderr = err_path.read_text()
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    out["rows"] = rows

    out["gates"].append((
        "per_repeat_row_count_eq_n_requests",
        len(rows) == n_req, f"rows={len(rows)} expected={n_req}",
    ))

    agg_match = None
    for line in stdout.splitlines():
        m = AGG_RE.match(line)
        if m:
            agg_match = m
            break
    if agg_match is None:
        out["gates"].append((
            "aggregate_line_present", False,
            "no n_ok=... n_cancelled=... n_error=... line"
        ))
    else:
        n_ok, n_cancelled, n_error = (int(g) for g in agg_match.groups())
        out["gates"].append((
            "aggregate_n_ok_n_canc_n_err",
            (n_ok == n_req and n_cancelled == 0 and n_error == 0),
            f"n_ok={n_ok} n_cancelled={n_cancelled} n_error={n_error}",
        ))

    statuses = [r["status"] for r in rows]
    out["gates"].append((
        "all_rows_status_ok",
        all(s == "ok" for s in statuses),
        f"distinct={sorted(set(statuses))}",
    ))

    n_tok_list   = [int(r["n_tokens_generated"]) for r in rows]
    n_tok_count  = Counter(n_tok_list)
    out["n_tok_counter"] = n_tok_count
    multiset_ok  = (n_tok_count == plan_counter)
    out["gates"].append((
        "n_tokens_multiset_eq_plan",
        multiset_ok,
        f"observed={dict(sorted(n_tok_count.items()))}  "
        f"plan={dict(sorted(plan_counter.items()))}",
    ))
    multiset_delta = {}
    all_keys = set(n_tok_count.keys()) | set(plan_counter.keys())
    for k in sorted(all_keys):
        d = n_tok_count.get(k, 0) - plan_counter.get(k, 0)
        if d != 0:
            multiset_delta[k] = d
    out["n_tok_multiset_delta"] = multiset_delta

    plan_set = set(plan)
    hash_by_b = {}
    for r in rows:
        k = int(r["n_tokens_generated"])
        if k not in plan_set:
            continue
        h = r["generated_token_hash"]
        hash_by_b.setdefault(k, set()).add(h)
    out["hash_by_budget"] = {k: set(v) for k, v in hash_by_b.items()}
    inconsistent_budgets = sorted(
        k for k, hs in hash_by_b.items() if len(hs) > 1
    )
    out["gates"].append((
        "per_budget_hash_consistency_within_trial",
        len(inconsistent_budgets) == 0,
        f"inconsistent_planned_budgets={inconsistent_budgets}",
    ))

    has_prompt_fit = any(PROMPT_FIT_RE.match(line) for line in stderr.splitlines())
    out["gates"].append((
        "prompt_fits_line_with_max_plan",
        has_prompt_fit, f"present={has_prompt_fit}",
    ))

    out["gates"].append((
        "no_serving_bench_error_in_stderr",
        "[serving-bench] error" not in stderr,
        "string='[serving-bench] error' must be absent",
    ))
    out["gates"].append((
        "no_failed_in_stderr",
        "failed" not in stderr,
        "string='failed' must be absent",
    ))

    for line in stdout.splitlines():
        m = WALL_RE.match(line)
        if m:
            out["wall_s"]  = float(m.group(1))
            out["agg_tps"] = float(m.group(2))
            break

    if wm_path.exists():
        try:
            first = wm_path.read_text().splitlines()[0]
            out["process_wall_ms"] = float(first)
        except (ValueError, IndexError):
            out["process_wall_ms"] = None

    trace_text = trc_path.read_text()
    trace_lines = [l for l in trace_text.splitlines() if ANY_TRACE_RE.match(l)]

    if layer == "correctness_trace_on" and backend == "hpx":
        starts   = [m for l in trace_lines if (m := START_RE.match(l))]
        readies  = [m for l in trace_lines if (m := READY_RE.match(l))]
        stops    = [l for l in trace_lines if STOP_RE.match(l)]
        acquires = [m for l in trace_lines if (m := ACQUIRE_RE.match(l))]
        releases = [m for l in trace_lines if (m := RELEASE_RE.match(l))]

        out["gates"].append((
            "hpx_start_line_count_eq_1", len(starts) == 1, f"count={len(starts)}",
        ))
        out["gates"].append((
            "hpx_ready_line_count_eq_1", len(readies) == 1, f"count={len(readies)}",
        ))
        if readies:
            ready_n_ctx, ready_pool = readies[0].groups()
            out["gates"].append((
                "hpx_ready_n_contexts_pool_size_match",
                int(ready_n_ctx) == entry["n_contexts"]
                    and int(ready_pool) == entry["expected_pool_size"],
                f"ready_n_contexts={ready_n_ctx} ready_pool={ready_pool}",
            ))
        out["gates"].append((
            "hpx_stop_line_count_eq_1", len(stops) == 1, f"count={len(stops)}",
        ))
        out["gates"].append((
            "hpx_acquire_count_eq_n_requests",
            len(acquires) == n_req, f"count={len(acquires)}",
        ))
        out["gates"].append((
            "hpx_release_count_eq_n_requests",
            len(releases) == n_req, f"count={len(releases)}",
        ))

        ctx_lo, ctx_hi = 0, entry["n_contexts"] - 1
        all_acq_ctx = [int(m.group(2)) for m in acquires]
        all_rel_ctx = [int(m.group(2)) for m in releases]
        out["gates"].append((
            "hpx_acquire_ctx_ids_in_range",
            all(ctx_lo <= c <= ctx_hi for c in all_acq_ctx),
            f"ids={sorted(set(all_acq_ctx))}",
        ))
        out["gates"].append((
            "hpx_release_ctx_ids_in_range",
            all(ctx_lo <= c <= ctx_hi for c in all_rel_ctx),
            f"ids={sorted(set(all_rel_ctx))}",
        ))
        out["gates"].append((
            "hpx_acquire_ctx_set_covers_all_contexts",
            set(all_acq_ctx) == set(range(entry["n_contexts"])),
            f"set={sorted(set(all_acq_ctx))} expected={list(range(entry['n_contexts']))}",
        ))

        acq_pos, rel_pos = {}, {}
        for pos, line in enumerate(trace_lines):
            m_a = ACQUIRE_RE.match(line)
            m_r = RELEASE_RE.match(line)
            if m_a:
                acq_pos[int(m_a.group(1))] = pos
            if m_r:
                rel_pos[int(m_r.group(1))] = pos

        same_ctx_ok = True
        order_ok    = True
        for i in range(n_req):
            a = next((m for m in acquires if int(m.group(1)) == i), None)
            r = next((m for m in releases if int(m.group(1)) == i), None)
            if a is None or r is None:
                same_ctx_ok = False
                order_ok    = False
                continue
            if int(a.group(2)) != int(r.group(2)):
                same_ctx_ok = False
            if not (i in acq_pos and i in rel_pos and acq_pos[i] < rel_pos[i]):
                order_ok = False
        out["gates"].append((
            "hpx_release_ctx_eq_acquire_ctx_per_req",
            same_ctx_ok, f"ok={same_ctx_ok}",
        ))
        out["gates"].append((
            "hpx_acquire_precedes_release_per_req",
            order_ok, f"ok={order_ok}",
        ))

        out["gates"].append((
            "hpx_total_filtered_trace_lines_eq_expected",
            len(trace_lines) == expected_trace_lines,
            f"count={len(trace_lines)} expected={expected_trace_lines}",
        ))
    else:
        out["gates"].append((
            "no_hpx_trace_lines",
            len(trace_lines) == 0, f"count={len(trace_lines)}",
        ))

    return out


def trial_pass(parsed):
    if not parsed["present"]:
        return False
    return all(ok for (_, ok, _) in parsed["gates"])


def main():
    args     = parse_args()
    schedule = load_schedule()
    plan     = (
        schedule.get("max_tokens_plan")
        or [e["max_tokens_plan"] for e in schedule["schedule"]][0]
    )

    layer        = args.layer
    backend      = args.backend
    layer_short  = LAYER_DIR[layer]

    entries = [
        e for e in schedule["schedule"]
        if e["layer"] == layer and e["backend"] == backend
    ]
    entries.sort(key=lambda e: e["trial_index"])
    if not entries:
        sys.exit(f"no schedule entries for layer={layer} backend={backend}")

    SUMM.mkdir(parents=True, exist_ok=True)

    parsed_per_trial = []
    n_trials_pass = 0
    multiset_mismatch_records = []
    for entry in entries:
        td = RUNS / entry["output_dir"]
        parsed = gate_trial(entry, td, plan)
        ok = trial_pass(parsed)
        parsed_per_trial.append((entry, parsed, ok))
        if ok:
            n_trials_pass += 1
        if parsed["n_tok_multiset_delta"]:
            multiset_mismatch_records.append(
                (entry["trial_index"], dict(parsed["n_tok_multiset_delta"]))
            )

    layer_overall_pass = (
        n_trials_pass == len(entries)
        and all(p["present"] for (_, p, _) in parsed_per_trial)
    )

    # ---- per-budget canonical hashes (std correctness layer only) ----
    canonical_by_budget = {}
    canonical_inconsistent = []
    if layer == "correctness_trace_on" and backend == "std":
        per_budget_hashes = {}
        for entry, parsed, ok in parsed_per_trial:
            if not ok:
                continue
            for b, hs in parsed["hash_by_budget"].items():
                if len(hs) != 1:
                    canonical_inconsistent.append(("within_trial", entry["trial_index"], b))
                    continue
                only = next(iter(hs))
                per_budget_hashes.setdefault(b, set()).add(only)
        for b, hs in per_budget_hashes.items():
            if len(hs) == 1:
                canonical_by_budget[str(b)] = next(iter(hs))
            else:
                canonical_by_budget[str(b)] = "INCONSISTENT_ACROSS_TRIALS"
                canonical_inconsistent.append(("across_trials", None, b))
        (SUMM / "canonical_hashes_by_budget.json").write_text(
            json.dumps(
                {"plan_budgets_sorted_unique":
                    sorted(set(plan)),
                 "canonical_hash_by_budget": canonical_by_budget,
                 "comment":
                    "Per-budget canonical hash from the std correctness "
                    "layer. At greedy/argmax with a fixed prompt, every "
                    "row of a given budget value shares one hash. This "
                    "file is a regression aid; the primary gate is the "
                    "per-budget cross-backend hash equality enforced by "
                    "_summarize_condition.py."},
                indent=2, sort_keys=True
            ) + "\n"
        )

    # ---- per-trial timing aggregates (descriptive) ----
    measured_trials = [
        (e, p) for (e, p, ok) in parsed_per_trial
        if ok and e["measured_for_timing"]
    ]
    pool_total_ms = []
    pool_ttft_ms  = []
    pool_dec_ms   = []
    makespans     = []
    process_walls = []
    agg_tps_list  = []

    for entry, parsed in measured_trials:
        per_trial_total_ms = []
        for r in parsed["rows"]:
            total_ms = float(r["total_ms"])
            ttft_ms  = float(r["ttft_ms"])
            tnt_ms   = float(r["total_minus_ttft_ms"])
            per_trial_total_ms.append(total_ms)
            pool_total_ms.append(total_ms)
            pool_ttft_ms.append(ttft_ms)
            pool_dec_ms.append(tnt_ms)
        if per_trial_total_ms:
            makespans.append(max(per_trial_total_ms))
        if parsed["process_wall_ms"] is not None:
            process_walls.append(parsed["process_wall_ms"])
        if parsed["agg_tps"] is not None:
            agg_tps_list.append(parsed["agg_tps"])

    # ---- per-trial CSV (every PASS trial) ----
    per_trial_csv = SUMM / f"layer_{layer_short}_{backend}_per_trial.csv"
    with per_trial_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "trial_index", "measured_for_timing", "trial_pass",
            "n_rows", "n_tok_multiset_delta_json",
            "makespan_ms",
            "process_wall_ms", "agg_tok_per_s",
        ])
        for entry, parsed, ok in parsed_per_trial:
            row_total_ms = [float(r["total_ms"]) for r in parsed["rows"]]
            mksp = max(row_total_ms) if row_total_ms else None
            w.writerow([
                entry["trial_index"], entry["measured_for_timing"], ok,
                len(parsed["rows"]),
                json.dumps(parsed["n_tok_multiset_delta"], sort_keys=True),
                f"{mksp:.6f}" if mksp is not None else "",
                f"{parsed['process_wall_ms']:.3f}" if parsed["process_wall_ms"] is not None else "",
                f"{parsed['agg_tps']:.3f}" if parsed["agg_tps"] is not None else "",
            ])

    # ---- all-requests CSV (every PASS trial; measured filter applied later) ----
    all_req_csv = SUMM / f"layer_{layer_short}_{backend}_all_requests.csv"
    with all_req_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "trial_index", "measured_for_timing",
            "completion_index", "n_tokens_generated",
            "generated_token_hash",
            "ttft_ms", "total_ms", "total_minus_ttft_ms", "tokens_per_second",
            "status",
        ])
        for entry, parsed, ok in parsed_per_trial:
            if not ok:
                continue
            for r in parsed["rows"]:
                n_tok = int(r["n_tokens_generated"])
                w.writerow([
                    entry["trial_index"], entry["measured_for_timing"],
                    r["completion_index"], n_tok,
                    r["generated_token_hash"],
                    r["ttft_ms"], r["total_ms"], r["total_minus_ttft_ms"],
                    r["tokens_per_second"],
                    r["status"],
                ])

    # ---- per-budget hash consistency *across* PASS trials ----
    cross_trial_hashes = {}
    for entry, parsed, ok in parsed_per_trial:
        if not ok:
            continue
        for b, hs in parsed["hash_by_budget"].items():
            if len(hs) != 1:
                continue
            only = next(iter(hs))
            cross_trial_hashes.setdefault(b, set()).add(only)
    cross_trial_inconsistent = [
        (b, sorted(hs)) for b, hs in sorted(cross_trial_hashes.items())
        if len(hs) > 1
    ]

    # ---- compose summary text ----
    out = []
    out.append(f"LAYER: {layer}")
    out.append(f"BACKEND: {backend}")
    out.append(f"N_TRIALS_SCHEDULED: {len(entries)}")
    out.append(f"N_TRIALS_PASS: {n_trials_pass}")
    out.append(f"N_TRIALS_PRESENT: {sum(1 for (_,p,_) in parsed_per_trial if p['present'])}")
    out.append(f"LAYER_OVERALL: {'PASS' if layer_overall_pass else 'FAIL'}")
    out.append(f"N_TRIALS_WITH_MULTISET_MISMATCH: {len(multiset_mismatch_records)}")
    out.append(f"N_MEASURED_TRIALS: {len(measured_trials)}")
    out.append(f"PLAN_LEN: {len(plan)}")
    out.append(f"PLAN_UNIQUE: {','.join(str(x) for x in sorted(set(plan)))}")
    out.append(f"PLAN_MULTISET: "
               f"{json.dumps({str(k): plan.count(k) for k in sorted(set(plan))}, sort_keys=True)}")
    out.append(f"CROSS_TRIAL_HASH_INCONSISTENT_BUDGETS: "
               f"{','.join(str(b) for (b,_) in cross_trial_inconsistent) or 'none'}")
    out.append("# END HEADER")
    out.append("")
    out.append(
        f"layer={layer}  backend={backend}  "
        f"n_trials_scheduled={len(entries)}  n_trials_pass={n_trials_pass}"
    )
    out.append("")

    out.append("Per-trial gate results:")
    out.append(
        "  trial  present  pass  n_rows  multiset_delta            failed_gates"
    )
    for entry, parsed, ok in parsed_per_trial:
        failed = [name for (name, gok, _) in parsed["gates"] if not gok]
        delta_str = ",".join(
            f"{k}:{v:+d}" for k, v in sorted(parsed["n_tok_multiset_delta"].items())
        )
        out.append(
            f"  {entry['trial_index']:5d}  "
            f"{str(parsed['present']):7s}  "
            f"{str(ok):4s}  "
            f"{len(parsed['rows']):6d}  "
            f"{delta_str if delta_str else '-':25s}  "
            f"{','.join(failed) if failed else '-'}"
        )
    out.append("")

    if multiset_mismatch_records:
        out.append(
            "Trials with n_tok multiset mismatch from plan (these trials FAIL):"
        )
        out.append(
            f"  plan multiset (sorted): "
            f"{dict(sorted(Counter(plan).items()))}"
        )
        out.append("  trial_index  observed-delta(value:diff)")
        for tidx, delta in multiset_mismatch_records:
            out.append(
                f"  {tidx:11d}  "
                f"{','.join(f'{k}:{v:+d}' for k, v in sorted(delta.items()))}"
            )
        out.append("")

    out.append("Failed-gate details (per trial):")
    any_failed = False
    for entry, parsed, _ in parsed_per_trial:
        for (name, gok, detail) in parsed["gates"]:
            if not gok:
                any_failed = True
                out.append(
                    f"  trial={entry['trial_index']}  gate={name}  detail={detail}"
                )
    if not any_failed:
        out.append("  (none)")
    out.append("")

    out.append("Per-budget hash observed in this (layer, backend):")
    if cross_trial_hashes:
        for b in sorted(cross_trial_hashes.keys()):
            hs = sorted(cross_trial_hashes[b])
            if len(hs) == 1:
                out.append(f"  budget={b:3d}  hash={hs[0]}")
            else:
                out.append(f"  budget={b:3d}  INCONSISTENT  hashes={hs}")
    else:
        out.append("  (no PASS trials)")
    out.append("")

    out.append("Descriptive timing aggregates (measured trials only):")
    out.append(f"  measured trials   : {len(measured_trials)}")
    out.append("  ----- per-request short-class pool (deep queue) -----")
    out.append(fmt_stats("short  total_ms",            stats(pool_total_ms)))
    out.append(fmt_stats("short  ttft_ms",             stats(pool_ttft_ms)))
    out.append(fmt_stats("short  total_minus_ttft_ms", stats(pool_dec_ms)))
    out.append("  ----- per-trial trial-level -----")
    out.append(fmt_stats("makespan_ms",     stats(makespans)))
    out.append(fmt_stats("process_wall_ms", stats(process_walls)))
    out.append(fmt_stats("agg_tok_per_s",   stats(agg_tps_list)))
    out.append("")

    if canonical_by_budget:
        out.append(
            "Canonical per-budget hashes recorded to "
            "summaries/canonical_hashes_by_budget.json"
        )
        if canonical_inconsistent:
            out.append(f"  inconsistencies: {canonical_inconsistent}")
        out.append("")

    out.append("Outputs:")
    out.append(f"  per-trial table : {per_trial_csv.name}")
    out.append(f"  request rows    : {all_req_csv.name}")
    out.append("")

    summary_path = SUMM / f"layer_{layer_short}_{backend}.txt"
    summary_path.write_text("\n".join(out) + "\n")

    print(f"wrote {summary_path}")
    print(f"LAYER_OVERALL: {'PASS' if layer_overall_pass else 'FAIL'}")
    print(f"n_trials_pass: {n_trials_pass}/{len(entries)}")
    print(f"trials with n_tok multiset mismatch: {len(multiset_mismatch_records)}")
    sys.exit(0 if layer_overall_pass else 2)


if __name__ == "__main__":
    main()
