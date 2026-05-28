"""Exp15 Phase 2 concurrent driver — non-streaming and streaming.

Purpose
-------
Validate the schema, process lifecycle, and gates of a *concurrent*
client harness against hpx-server and llama-server, then drive the
Phase 2 timing slices (Phase 2a admission/rejection, Phase 2b
admitted parallelism, both non-streaming and SSE streaming).

Scope
-----
- non-streaming and streaming (per cfg["streaming"])
- p0 only at decode_budget=8 (canonical anchor)
- concurrency levels read from config (S0 uses [1, 2])
- iterations_per_client read from config (S0 uses 6)
- warm-up: drop iteration == 1 per client from timing aggregates;
  every row is kept for correctness/stability gates

Constraints
-----------
- Python 3 stdlib only.
- Reuses Exp12 adapters (`adapters.hpx_server`, `adapters.llama_server`)
  via sys.path so request/response semantics are not duplicated.
- Reuses Exp12 server-lifecycle helpers (`bench.tcp_wait`,
  `bench.http_post_json`, `bench.launch_server`, `bench.stop_server`,
  `bench.write_args_file`, `bench.server_pid_running`) for the same
  reason. Exp12 is not modified.
- The optional config block `server_args` (Phase 2b on) overrides the
  Exp12 adapter's hard-coded `--n-seq-max` / `--max-concurrent` /
  `--parallel` defaults via a harness-side argv builder. When
  `server_args` is absent, the harness falls back to
  `adapter.build_args(cfg, port)` so Phase 2a-S0 configs run
  unchanged.
- One server process at a time. hpx-server first, then llama-server.
  Each server boots exactly once per run; all concurrency cells for
  that server share the boot. This sidesteps port reuse / TIME_WAIT
  issues by construction.
- Threaded clients synchronize on a start barrier at the start of each
  concurrency cell so requests are submitted as close together as
  practical. Within a client, iterations run back-to-back with no
  inter-iteration delay.

Outputs (per run)
-----------------
results/<run-id>-c-smoke/
  config.json                  — echo of input config
  client_results.jsonl         — one JSON object per request row
  summary.csv                  — flattened columns of the same rows
  run_notes.txt                — human-readable run log + gates
  hpx_server.{stdout,stderr,args}
  llama_server.{stdout,stderr,args}
"""

import argparse
import csv
import datetime
import hashlib
import json
import os
import socket
import sys
import threading
import time
from pathlib import Path

# ----- Import Exp12 adapters and helpers without modifying them. -----

_HERE   = Path(__file__).resolve().parent
_EXP12  = (_HERE / ".." / "12_hpx_vs_llama_server_pair").resolve()
if str(_EXP12) not in sys.path:
    sys.path.insert(0, str(_EXP12))

from adapters import hpx_server as hpx_adapter           # noqa: E402
from adapters import llama_server as llama_adapter       # noqa: E402
import bench as exp12_bench                              # noqa: E402

tcp_wait            = exp12_bench.tcp_wait
http_post_json      = exp12_bench.http_post_json
http_post_streaming = exp12_bench.http_post_streaming
launch_server       = exp12_bench.launch_server
stop_server         = exp12_bench.stop_server
write_args_file     = exp12_bench.write_args_file
server_pid_running  = exp12_bench.server_pid_running


# ----- Run identification --------------------------------------------

def utc_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def make_run_id():
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return "{}-c-smoke".format(ts)


# ----- Config loading -------------------------------------------------

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    # Minimal shape validation. No defaulting.
    required = ("model_path", "binaries", "ports", "ctx_size", "threads",
                "workload", "concurrency_levels", "iterations_per_client",
                "readiness", "shutdown", "request_timeout_seconds")
    for k in required:
        if k not in cfg:
            raise ValueError(
                "config missing required key: {!r}".format(k))
    if not isinstance(cfg["concurrency_levels"], list) \
            or not cfg["concurrency_levels"]:
        raise ValueError("concurrency_levels must be a non-empty list")
    for c in cfg["concurrency_levels"]:
        if not isinstance(c, int) or c < 1:
            raise ValueError(
                "concurrency_levels entries must be positive ints; "
                "got {!r}".format(c))
    if not isinstance(cfg["iterations_per_client"], int) \
            or cfg["iterations_per_client"] < 1:
        raise ValueError("iterations_per_client must be a positive int")
    w = cfg["workload"]
    for k in ("workload_id", "prompt_id", "prompt", "decode_budget",
              "is_canonical_anchor"):
        if k not in w:
            raise ValueError(
                "workload missing required key: {!r}".format(k))
    if "streaming" in cfg:
        if not isinstance(cfg["streaming"], bool):
            raise ValueError("streaming must be a bool")
    for k in ("stream_line_timeout_seconds",
              "stream_overall_timeout_seconds"):
        if k in cfg and not isinstance(cfg[k], (int, float)):
            raise ValueError("{} must be a number".format(k))
    if "server_args" in cfg:
        sa = cfg["server_args"]
        if not isinstance(sa, dict):
            raise ValueError("server_args must be a dict")
        for label, block in sa.items():
            if label not in ("hpx_server", "llama_server"):
                raise ValueError(
                    "server_args label must be 'hpx_server' or "
                    "'llama_server'; got {!r}".format(label))
            if not isinstance(block, dict):
                raise ValueError(
                    "server_args[{}] must be a dict".format(label))
        if "hpx_server" in sa:
            for k in ("n_seq_max", "max_concurrent"):
                if k not in sa["hpx_server"]:
                    raise ValueError(
                        "server_args.hpx_server missing required "
                        "key: {!r}".format(k))
        if "llama_server" in sa:
            if "parallel" not in sa["llama_server"]:
                raise ValueError(
                    "server_args.llama_server missing required "
                    "key: 'parallel'")
    return cfg


# ----- Per-request runner -------------------------------------------

def _now_ms_from(t0):
    return int((time.monotonic() - t0) * 1000.0)


def _request_once(adapter, host, port, prompt, decode_budget,
                  timeout_s, t_cell_start):
    """Issue one non-streaming request. Return a partial row dict.

    Caller fills in workload/server/concurrency/client_id/iteration.
    """
    body = adapter.request_body(prompt, decode_budget)
    submit_ms = _now_ms_from(t_cell_start)
    t_send = time.monotonic()
    failed_reason = ""
    status   = -1
    headers  = {}
    raw_body = ""
    try:
        status, headers, raw_body = http_post_json(
            host, port, "/completion", body, timeout_s=timeout_s)
    except socket.timeout:
        failed_reason = "client_timeout_after_{}s".format(timeout_s)
    except TimeoutError as e:
        failed_reason = "client_timeout: {}".format(e)
    except OSError as e:
        failed_reason = "transport_error: {}".format(e)
    except Exception as e:
        failed_reason = "client_exception: {}".format(e)
    complete_ms = _now_ms_from(t_cell_start)
    latency_ms  = int((time.monotonic() - t_send) * 1000.0)

    parsed = adapter.parse_response(status, headers, raw_body)
    parse_error = parsed["parse_error"]
    if failed_reason == "" and status != 200:
        failed_reason = "non_200_status_{}".format(status)
    if failed_reason == "" and parse_error != "":
        failed_reason = "parse_error: {}".format(parse_error)

    return {
        "submit_monotonic_ms":    submit_ms,
        "complete_monotonic_ms":  complete_ms,
        "latency_ms":             latency_ms,
        "http_status":            status,
        "parse_error":            parse_error,
        "n_decoded":              parsed["n_decoded"],
        "hash":                   parsed["hash"],
        "text_normalized_sha256": parsed["text_normalized_sha256"],
        "text_len_chars":         parsed["text_len_chars"],
        "server_status":          parsed["server_status"],
        "failed_reason":          failed_reason,
        "raw_json":               parsed["raw_json"],
        "request_body":           body,
    }


def _request_once_streaming(adapter, host, port, prompt, decode_budget,
                            line_timeout_s, overall_timeout_s,
                            t_cell_start):
    """Issue one streaming (SSE) request. Return a partial row dict.

    Reuses Exp12's http_post_streaming and the adapter's
    stream_request_body / parse_stream_events; no SSE parser logic
    is duplicated here. The per-token timestamp comprehension below
    operates on the already-classified public `events` list and is
    the same shape as Exp12 bench.py uses (event_type=="token" for
    hpx-server, or event_type=="" with stop=False and non-empty
    content for llama-server).
    """
    body = adapter.stream_request_body(prompt, decode_budget)
    submit_ms = _now_ms_from(t_cell_start)
    t_send = time.monotonic()
    failed_reason   = ""
    status          = -1
    headers         = {}
    events          = []
    stream_err      = ""
    t_request_sent  = t_send
    try:
        (t_request_sent, events, status, headers,
         stream_err) = http_post_streaming(
            host, port, "/completion", body,
            line_timeout_s=line_timeout_s,
            overall_timeout_s=overall_timeout_s)
    except socket.timeout:
        failed_reason = "client_timeout_after_{}s".format(overall_timeout_s)
    except TimeoutError as e:
        failed_reason = "client_timeout: {}".format(e)
    except OSError as e:
        failed_reason = "transport_error: {}".format(e)
    except Exception as e:
        failed_reason = "client_exception: {}".format(e)
    complete_ms = _now_ms_from(t_cell_start)
    total_latency_ms = int((time.monotonic() - t_send) * 1000.0)

    if status == 200:
        parsed = adapter.parse_stream_events(events)
        if stream_err and not parsed["parse_error"]:
            parsed["parse_error"] = stream_err
    else:
        parsed = {
            "ok":                       False,
            "http_status":              status,
            "text":                     "",
            "text_sha256":              "",
            "text_normalized_sha256":   "",
            "text_len_chars":           0,
            "n_decoded":                -1,
            "hash":                     "",
            "prompt_tokens":            -1,
            "server_request_id":        -1,
            "server_status":            "",
            "raw_json":                 {},
            "parse_error":              stream_err or (
                "non_200_status_{}".format(status) if status != -1 else ""),
            "token_event_count":        0,
            "done_seen":                False,
            "done_status":              "",
            "first_content_event_ts_ms": -1,
            "first_event_ts_ms":        -1,
            "last_event_ts_ms":         -1,
        }

    stream_parse_error = parsed["parse_error"]
    if failed_reason == "" and status != 200:
        failed_reason = "non_200_status_{}".format(status)
    if failed_reason == "" and stream_parse_error != "":
        failed_reason = "stream_parse_error: {}".format(stream_parse_error)
    if failed_reason == "" and not parsed["done_seen"]:
        failed_reason = "no_done_seen"
    if failed_reason == "" and parsed["token_event_count"] <= 0:
        failed_reason = "token_event_count_zero"

    token_event_times_ms = [
        e["ts_ms_from_request"] for e in events
        if (e["event_type"] == "token") or
           (e["event_type"] == "" and
            isinstance(e["data_json"], dict) and
            e["data_json"].get("stop") is False and
            bool(e["data_json"].get("content")))
    ]

    return {
        "stream_mode":            "streaming",
        "submit_monotonic_ms":    submit_ms,
        "complete_monotonic_ms":  complete_ms,
        "latency_ms":             total_latency_ms,
        "total_latency_ms":       total_latency_ms,
        "first_event_latency_ms": parsed["first_event_ts_ms"],
        "first_token_latency_ms": parsed["first_content_event_ts_ms"],
        "token_event_count":      parsed["token_event_count"],
        "token_event_times_ms":   token_event_times_ms,
        "done_seen":              parsed["done_seen"],
        "done_status":            parsed["done_status"],
        "stream_parse_error":     stream_parse_error,
        "http_status":            status,
        "parse_error":            "",
        "n_decoded":              parsed["n_decoded"],
        "hash":                   parsed["hash"],
        "text_normalized_sha256": parsed["text_normalized_sha256"],
        "text_len_chars":         parsed["text_len_chars"],
        "server_status":          parsed["server_status"],
        "failed_reason":          failed_reason,
        "raw_json":               parsed["raw_json"],
        "request_body":           body,
    }


def _client_worker(client_id, server_label, workload, concurrency,
                   adapter, host, port, iters_per_client, timeout_s,
                   barrier, t_cell_start_holder, rows_lock, rows_out,
                   exceptions_out, streaming, line_timeout_s,
                   overall_timeout_s):
    """Single client thread. Issues `iters_per_client` requests."""
    try:
        barrier.wait()
        # All clients past the barrier; the first wake snapshots cell start.
        with rows_lock:
            if t_cell_start_holder["t"] is None:
                t_cell_start_holder["t"] = time.monotonic()
            t_cell_start = t_cell_start_holder["t"]
        for it in range(1, iters_per_client + 1):
            if streaming:
                partial = _request_once_streaming(
                    adapter, host, port,
                    workload["prompt"], workload["decode_budget"],
                    line_timeout_s=line_timeout_s,
                    overall_timeout_s=overall_timeout_s,
                    t_cell_start=t_cell_start)
            else:
                partial = _request_once(
                    adapter, host, port,
                    workload["prompt"], workload["decode_budget"],
                    timeout_s=timeout_s,
                    t_cell_start=t_cell_start)
                partial["stream_mode"] = "non_streaming"
            row = {
                "server":                  server_label,
                "workload_id":             workload["workload_id"],
                "prompt_id":               workload["prompt_id"],
                "is_canonical_anchor":     workload["is_canonical_anchor"],
                "concurrency":             concurrency,
                "client_id":               client_id,
                "iteration":               it,
                "is_warmup":               (it == 1),
                "decode_budget_requested": workload["decode_budget"],
                "ts_utc":                  utc_iso(),
            }
            row.update(partial)
            with rows_lock:
                rows_out.append(row)
    except Exception as e:
        exceptions_out.append((client_id, repr(e)))


def run_cell(server_label, adapter, host, port, workload, concurrency,
             iters_per_client, timeout_s, notes,
             streaming=False,
             line_timeout_s=30.0,
             overall_timeout_s=60.0):
    """Run one (server, concurrency) cell. Returns (rows, agg_dict).

    Server is assumed already booted, ready, and warmed up by the caller.
    """
    notes.append(
        "[{}] cell concurrency={} workload={} iters_per_client={} "
        "streaming={}".format(
            server_label, concurrency, workload["workload_id"],
            iters_per_client, streaming))
    barrier = threading.Barrier(concurrency)
    rows_lock = threading.Lock()
    rows = []
    exceptions = []
    t_cell_start_holder = {"t": None}

    threads = []
    for cid in range(concurrency):
        t = threading.Thread(
            target=_client_worker,
            args=(cid, server_label, workload, concurrency, adapter,
                  host, port, iters_per_client, timeout_s,
                  barrier, t_cell_start_holder, rows_lock, rows,
                  exceptions, streaming, line_timeout_s,
                  overall_timeout_s),
            name="client-{}-{}".format(server_label, cid),
            daemon=False)
        threads.append(t)

    t_dispatch = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    t_cell_start = t_cell_start_holder["t"] or t_dispatch
    t_cell_end = time.monotonic()
    batch_wall_clock_ms = int((t_cell_end - t_cell_start) * 1000.0)

    if exceptions:
        notes.append(
            "[{}] cell c={} thread exceptions: {}".format(
                server_label, concurrency, exceptions))

    completed = sum(
        1 for r in rows
        if r["http_status"] == 200 and r["parse_error"] == ""
        and r["failed_reason"] == "")
    failed = sum(
        1 for r in rows
        if r["failed_reason"] and "timeout" not in r["failed_reason"])
    timeouts = sum(
        1 for r in rows
        if r["failed_reason"] and "timeout" in r["failed_reason"])

    rps = 0.0
    if batch_wall_clock_ms > 0 and completed > 0:
        rps = round(completed * 1000.0 / batch_wall_clock_ms, 3)

    agg = {
        "server":                   server_label,
        "workload_id":              workload["workload_id"],
        "concurrency":              concurrency,
        "rows":                     len(rows),
        "completed_count":          completed,
        "failed_count":             failed,
        "timeout_count":            timeouts,
        "batch_wall_clock_ms":      batch_wall_clock_ms,
        "client_observed_rps":      rps,
        "client_observed_rps_note": ("approximate; client-observed; "
                                     "includes loopback; not a server "
                                     "throughput claim"),
    }
    notes.append(
        "[{}] cell c={} done rows={} completed={} failed={} "
        "timeouts={} batch_wall_clock_ms={} client_observed_rps={}".format(
            server_label, concurrency, len(rows), completed, failed,
            timeouts, batch_wall_clock_ms, rps))

    # Sort rows for stable downstream artifacts.
    rows.sort(key=lambda r: (r["client_id"], r["iteration"]))
    return rows, agg


# ----- Server lifecycle (per server, once) ---------------------------

def warm_up_server(adapter, host, port, readiness_cfg, notes, label):
    """tcp_wait then warm-up /completion POST until 200, with timeout."""
    try:
        elapsed = tcp_wait(
            host, port,
            interval_s=readiness_cfg["tcp_poll_interval_seconds"],
            timeout_s=readiness_cfg["tcp_poll_timeout_seconds"])
        notes.append("[{}] tcp ready in {:.3f}s".format(label, elapsed))
    except TimeoutError as e:
        return False, "tcp_wait_timeout: {}".format(e)

    deadline = (time.monotonic()
                + readiness_cfg["tcp_poll_timeout_seconds"])
    attempts = 0
    last_err = ""
    while time.monotonic() < deadline:
        attempts += 1
        try:
            wstatus, _wh, _wb = http_post_json(
                host, port, "/completion",
                adapter.warmup_body(),
                timeout_s=readiness_cfg["warmup_request_timeout_seconds"])
            if wstatus == 200:
                notes.append(
                    "[{}] warmup POST status=200 attempts={}".format(
                        label, attempts))
                return True, ""
            last_err = "warmup_status_{}".format(wstatus)
        except Exception as e:
            last_err = "warmup_exception: {}".format(e)
        time.sleep(readiness_cfg["tcp_poll_interval_seconds"])
    return False, (last_err
                   or "warmup_timeout_after_{}_attempts".format(attempts))


def _hpx_argv_from_server_args(cfg, port):
    """Phase 2b argv builder for hpx-server.

    Uses cfg['server_args']['hpx_server']. Required: n_seq_max,
    max_concurrent. Optional: max_prompt_tokens (default 512).
    The shared knobs (model_path, threads, ctx_size) come from cfg
    top-level. `--engine-pool` is deliberately omitted (default OFF).
    """
    sa = cfg["server_args"]["hpx_server"]
    max_prompt_tokens = sa.get("max_prompt_tokens", 512)
    return [
        cfg["binaries"]["hpx_server"],
        "--model",             cfg["model_path"],
        "--host",              "127.0.0.1",
        "--port",              str(port),
        "--n-seq-max",         str(sa["n_seq_max"]),
        "--max-prompt-tokens", str(max_prompt_tokens),
        "--n-threads",         str(cfg["threads"]),
        "--max-concurrent",    str(sa["max_concurrent"]),
        "--ctx-size",          str(cfg["ctx_size"]),
    ]


def _llama_argv_from_server_args(cfg, port):
    """Phase 2b argv builder for llama-server.

    Uses cfg['server_args']['llama_server']. Required: parallel.
    Optional: no_context_shift (bool), seed (int).
    """
    sa = cfg["server_args"]["llama_server"]
    argv = [
        cfg["binaries"]["llama_server"],
        "--model",     cfg["model_path"],
        "--host",      "127.0.0.1",
        "--port",      str(port),
        "--ctx-size",  str(cfg["ctx_size"]),
        "--parallel",  str(sa["parallel"]),
        "--threads",   str(cfg["threads"]),
    ]
    if sa.get("no_context_shift", False):
        argv.append("--no-context-shift")
    if "seed" in sa:
        argv.extend(["--seed", str(sa["seed"])])
    return argv


def _build_argv(server_label, adapter, cfg, port):
    """Return argv for `server_label`.

    If cfg['server_args'][server_label] is present, build argv from the
    harness-side builder (Phase 2b on). Otherwise, fall back to
    `adapter.build_args(cfg, port)` so Phase 2a-S0 configs run
    unchanged.
    """
    sa_block = cfg.get("server_args")
    if (isinstance(sa_block, dict)
            and isinstance(sa_block.get(server_label), dict)):
        if server_label == "hpx_server":
            return _hpx_argv_from_server_args(cfg, port)
        if server_label == "llama_server":
            return _llama_argv_from_server_args(cfg, port)
    return adapter.build_args(cfg, port)


def run_one_server(server_label, adapter, cfg, run_dir, notes):
    """Boot the server once and run all concurrency cells against it."""
    port = cfg["ports"][server_label]
    host = "127.0.0.1"
    workload = cfg["workload"]

    argv = _build_argv(server_label, adapter, cfg, port)
    stdout_path = run_dir / "{}.stdout".format(server_label)
    stderr_path = run_dir / "{}.stderr".format(server_label)
    args_path   = run_dir / "{}.args".format(server_label)
    write_args_file(args_path, argv)

    notes.append("[{}] launch argv:".format(server_label))
    for a in argv:
        notes.append("    {}".format(a))

    proc = launch_server(server_label, argv, stdout_path, stderr_path)
    notes.append("[{}] launched pid={} at {}".format(
        server_label, proc.pid, utc_iso()))

    rows_all = []
    cell_aggs = []
    cell_failed = False
    try:
        ok, err = warm_up_server(
            adapter, host, port, cfg["readiness"], notes, server_label)
        if not ok:
            notes.append("[{}] readiness failed: {}".format(
                server_label, err))
            return rows_all, cell_aggs, False

        streaming = bool(cfg.get("streaming", False))
        line_timeout_s    = float(cfg.get("stream_line_timeout_seconds", 30.0))
        overall_timeout_s = float(cfg.get("stream_overall_timeout_seconds", 60.0))
        for c in cfg["concurrency_levels"]:
            rows, agg = run_cell(
                server_label, adapter, host, port,
                workload, c,
                cfg["iterations_per_client"],
                cfg["request_timeout_seconds"],
                notes,
                streaming=streaming,
                line_timeout_s=line_timeout_s,
                overall_timeout_s=overall_timeout_s)
            rows_all.extend(rows)
            cell_aggs.append(agg)
            if agg["failed_count"] or agg["timeout_count"]:
                cell_failed = True

    finally:
        exit_code, term_method = stop_server(
            proc, grace_seconds=cfg["shutdown"]["sigterm_grace_seconds"])
        notes.append(
            "[{}] stopped exit={} term_method={} "
            "pid_still_running={}".format(
                server_label, exit_code, term_method,
                server_pid_running(proc.pid)))
    return rows_all, cell_aggs, (not cell_failed)


# ----- Gate evaluation ----------------------------------------------

CANONICAL_HPX_HASH = {
    ("p0_b8", 8):  "0x0619d4d1900c2365",
    ("p0_b32", 32): "0x6794e47fe0f84af1",
}

FORBIDDEN_WORDS = ("faster", "slower", "speedup", "regression",
                   "wins", "beats", "outperforms", "better", "worse")


def evaluate_gates(cfg, hpx_rows, llama_rows, notes):
    """Apply S0 gates. Append gate lines to notes. Return True on PASS."""
    workload = cfg["workload"]
    wid = workload["workload_id"]
    budget = workload["decode_budget"]
    expected_hpx_hash = CANONICAL_HPX_HASH.get((wid, budget))

    passed = True
    notes.append("--- gates ---")

    def fail(msg):
        nonlocal passed
        passed = False
        notes.append("FAIL: {}".format(msg))

    def ok_line(msg):
        notes.append("gate: {}".format(msg))

    streaming = bool(cfg.get("streaming", False))

    # Row-level: HTTP 200, no parse errors, no failed_reason.
    for label, rows in (("hpx_server", hpx_rows),
                        ("llama_server", llama_rows)):
        for r in rows:
            if r["http_status"] != 200:
                fail("{} c={} client_id={} iter={} http_status={}".format(
                    label, r["concurrency"], r["client_id"],
                    r["iteration"], r["http_status"]))
            if r["parse_error"]:
                fail("{} c={} client_id={} iter={} parse_error={!r}".format(
                    label, r["concurrency"], r["client_id"],
                    r["iteration"], r["parse_error"]))
            if r["failed_reason"]:
                fail("{} c={} client_id={} iter={} failed_reason={!r}".format(
                    label, r["concurrency"], r["client_id"],
                    r["iteration"], r["failed_reason"]))
            if r["n_decoded"] != budget:
                fail("{} c={} client_id={} iter={} n_decoded={} "
                     "expected={}".format(
                         label, r["concurrency"], r["client_id"],
                         r["iteration"], r["n_decoded"], budget))
            if streaming:
                if r.get("stream_parse_error"):
                    fail("{} c={} client_id={} iter={} stream_parse_error="
                         "{!r}".format(label, r["concurrency"],
                                       r["client_id"], r["iteration"],
                                       r.get("stream_parse_error")))
                if not r.get("done_seen"):
                    fail("{} c={} client_id={} iter={} done_seen=False".format(
                        label, r["concurrency"], r["client_id"],
                        r["iteration"]))
                if int(r.get("token_event_count", 0)) <= 0:
                    fail("{} c={} client_id={} iter={} token_event_count="
                         "{} <= 0".format(label, r["concurrency"],
                                          r["client_id"], r["iteration"],
                                          r.get("token_event_count")))
    if passed:
        if streaming:
            ok_line("all rows http=200, no parse errors, no failed reasons, "
                    "n_decoded == {}, stream_parse_error empty, done_seen "
                    "true, token_event_count > 0".format(budget))
        else:
            ok_line("all rows http=200, no parse errors, no failed reasons, "
                    "n_decoded == {}".format(budget))

    # Canonical hpx hash on every hpx row.
    if expected_hpx_hash is not None:
        for r in hpx_rows:
            if r["hash"] != expected_hpx_hash:
                fail("hpx_server c={} client_id={} iter={} hash={} "
                     "expected={}".format(
                         r["concurrency"], r["client_id"], r["iteration"],
                         r["hash"], expected_hpx_hash))
        if passed:
            ok_line("hpx_server hash == {} across all hpx rows".format(
                expected_hpx_hash))
    else:
        notes.append(
            "note: no canonical hash pinned for workload={} budget={}; "
            "stability-only gate".format(wid, budget))

    # Per-(server, concurrency) text_normalized_sha256 stable.
    def stable_by_cell(label, rows):
        by_cell = {}
        for r in rows:
            key = (r["concurrency"],)
            by_cell.setdefault(key, set()).add(r["text_normalized_sha256"])
        for key, vals in sorted(by_cell.items()):
            if len(vals) != 1:
                fail("{} c={} text_normalized_sha256 not stable: {}".format(
                    label, key[0], sorted(vals)))
            else:
                ok_line(
                    "{} c={} text_normalized_sha256 stable: {}".format(
                        label, key[0], next(iter(vals))))

    stable_by_cell("hpx_server", hpx_rows)
    stable_by_cell("llama_server", llama_rows)

    notes.append("--- end ---")
    return passed


def check_forbidden_words(notes_text):
    hits = []
    low = notes_text.lower()
    for word in FORBIDDEN_WORDS:
        idx = 0
        while True:
            j = low.find(word, idx)
            if j < 0:
                break
            hits.append((word, j))
            idx = j + 1
    return hits


# ----- Output writers ----------------------------------------------

CSV_COLUMNS = [
    "server", "workload_id", "prompt_id", "is_canonical_anchor",
    "concurrency", "client_id", "iteration", "is_warmup",
    "decode_budget_requested",
    "submit_monotonic_ms", "complete_monotonic_ms", "latency_ms",
    "http_status", "n_decoded", "hash", "text_normalized_sha256",
    "text_len_chars", "server_status", "parse_error", "failed_reason",
    "stream_mode", "total_latency_ms", "first_event_latency_ms",
    "first_token_latency_ms", "token_event_count", "done_seen",
    "done_status", "stream_parse_error",
    "ts_utc",
]


def write_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True))
            f.write("\n")


def write_summary_csv(rows, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        for r in rows:
            w.writerow([r.get(k, "") for k in CSV_COLUMNS])


def write_run_notes(path, notes, cell_aggs, gates_passed, forbidden_hits):
    with open(path, "w", encoding="utf-8") as f:
        for n in notes:
            f.write(n)
            f.write("\n")
        f.write("--- cell aggregates ---\n")
        for agg in cell_aggs:
            f.write(json.dumps(agg, sort_keys=True))
            f.write("\n")
        f.write("--- forbidden-word audit ---\n")
        if forbidden_hits:
            f.write("HITS: {}\n".format(forbidden_hits))
        else:
            f.write("no forbidden comparative wording detected\n")
        f.write("--- overall ---\n")
        f.write("S0_GATES: {}\n".format("PASS" if gates_passed else "FAIL"))


def write_config_echo(cfg, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)
        f.write("\n")


# ----- Main ---------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Exp15 Phase 2a-S0 concurrent harness smoke")
    ap.add_argument("--config", required=True, help="Path to S0 config JSON")
    ap.add_argument("--results-root", required=True,
                    help="Directory under which a results/<run-id>/ subtree "
                         "will be created.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    results_root = Path(args.results_root).resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    run_id = make_run_id()
    run_dir = results_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    notes = []
    notes.append("run_id={} started={}".format(run_id, utc_iso()))
    notes.append("config_path={}".format(os.path.abspath(args.config)))
    notes.append("results_dir={}".format(run_dir))

    write_config_echo(cfg, run_dir / "config.json")

    # hpx-server first, then llama-server.
    hpx_rows, hpx_aggs, hpx_ok = run_one_server(
        "hpx_server", hpx_adapter, cfg, run_dir, notes)
    llama_rows, llama_aggs, llama_ok = run_one_server(
        "llama_server", llama_adapter, cfg, run_dir, notes)

    all_rows = hpx_rows + llama_rows
    cell_aggs = hpx_aggs + llama_aggs

    write_jsonl(all_rows, run_dir / "client_results.jsonl")
    write_summary_csv(all_rows, run_dir / "summary.csv")

    gates_passed = evaluate_gates(cfg, hpx_rows, llama_rows, notes)
    notes_text = "\n".join(notes)
    forbidden_hits = check_forbidden_words(notes_text)

    write_run_notes(
        run_dir / "run_notes.txt",
        notes, cell_aggs, gates_passed, forbidden_hits)

    print("run_id={}".format(run_id))
    print("results_dir={}".format(run_dir))
    print("hpx_rows={} llama_rows={}".format(len(hpx_rows), len(llama_rows)))
    print("hpx_server_cells_ok={} llama_server_cells_ok={}".format(
        hpx_ok, llama_ok))
    for agg in cell_aggs:
        print(json.dumps(agg, sort_keys=True))
    print("forbidden_word_hits={}".format(forbidden_hits))
    print("S0_GATES={}".format("PASS" if gates_passed else "FAIL"))

    # Exit code: 0 only if both sides ran and gates pass.
    return 0 if (hpx_ok and llama_ok and gates_passed
                 and not forbidden_hits) else 1


if __name__ == "__main__":
    sys.exit(main())
