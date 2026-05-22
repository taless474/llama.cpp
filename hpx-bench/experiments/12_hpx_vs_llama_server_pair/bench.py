#!/usr/bin/env python3
"""M8b matched pair-run harness — hpx-server vs llama-server.

Python 3 stdlib only. No third-party deps. No HPX/llama.cpp build
coupling. The harness:

  1. reads a config JSON
  2. creates results/<run-id>/
  3. launches hpx-server, waits readiness, runs N requests, stops it
  4. launches llama-server, waits readiness, runs N requests, stops it
  5. writes config.json, per-server stdout/stderr/args, client_results.jsonl,
     summary.csv, run_notes.txt under results/<run-id>/

Never runs both servers concurrently.

By design this harness writes NO comparative performance claim.
run_notes.txt is factual only.

Usage:
    python3 bench.py --config config.example.json
"""

import argparse
import csv
import datetime
import http.client
import json
import os
import pathlib
import re
import signal
import socket
import subprocess
import sys
import time

from adapters import hpx_server as hpx_adapter
from adapters import llama_server as llama_adapter


HARNESS_VERSION = "m8e.0"

CANONICAL_HPX_HASH = "0x0619d4d1900c2365"

FORBIDDEN_WORDS = (
    "faster", "slower", "speedup", "regression",
    "wins", "beats", "outperforms", "better", "worse",
)
FORBIDDEN_RE = re.compile(
    r"\b(" + "|".join(FORBIDDEN_WORDS) + r")\b", re.IGNORECASE)


def utc_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def make_run_id():
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return "{}-pair".format(ts)


def tcp_wait(host, port, interval_s, timeout_s):
    """Poll a TCP connect until success or timeout. Returns elapsed seconds.

    Raises TimeoutError on timeout. No std::this_thread::sleep_for analogue
    is needed here — this is the external client, not HPX-owned code.
    """
    deadline = time.monotonic() + timeout_s
    started = time.monotonic()
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return time.monotonic() - started
        except OSError:
            time.sleep(interval_s)
    raise TimeoutError(
        "tcp_wait({}:{}) timed out after {}s".format(host, port, timeout_s))


def http_post_json(host, port, path, body_dict, timeout_s):
    """Issue one HTTP POST with a JSON body. Returns (status, headers, body).

    body is returned as bytes-decoded str.
    """
    conn = http.client.HTTPConnection(host, port, timeout=timeout_s)
    try:
        payload = json.dumps(body_dict).encode("utf-8")
        conn.request(
            "POST", path, body=payload,
            headers={"Content-Type": "application/json",
                     "Content-Length": str(len(payload))})
        resp = conn.getresponse()
        status = resp.status
        headers = dict(resp.getheaders())
        raw = resp.read().decode("utf-8", errors="replace")
        return status, headers, raw
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _iter_sse_records(resp, t_request_sent, line_timeout_s,
                      overall_timeout_s):
    """Yield (ts_ms_from_request, event_type, data_raw, data_json_or_None)
    tuples from a streaming HTTPResponse.

    Implements a minimal SSE parser:
      - reads line by line via resp.readline()
      - accepts CRLF or LF
      - ignores comment lines (start with ':')
      - accumulates `data:` fields with '\\n' separators per SSE spec
      - tracks the `event:` field; default event_type when none provided
        is the empty string (we treat that as the llama-server convention)
      - emits one tuple per blank line (record terminator)
      - stops on EOF or overall_timeout_s elapsed

    `t_request_sent` is monotonic seconds at the moment the request was
    fully written. ts_ms_from_request is the monotonic delta in ms at
    the moment the *terminating blank line* of the record is observed.
    """
    deadline_overall = t_request_sent + overall_timeout_s
    try:
        resp.fp.raw._sock.settimeout(line_timeout_s)  # type: ignore[attr-defined]
    except Exception:
        pass

    event_type = ""
    data_lines = []

    while True:
        if time.monotonic() > deadline_overall:
            raise TimeoutError("sse_overall_timeout")
        try:
            raw = resp.readline()
        except socket.timeout:
            raise TimeoutError("sse_line_timeout")
        except Exception as e:
            raise IOError("sse_read_error: {}".format(e))
        if not raw:
            # EOF mid-stream is treated as end-of-stream; do NOT
            # synthesize a final record. Caller handles done_seen.
            if data_lines or event_type:
                # An unterminated dangling record; surface it so the
                # adapter can decide whether to flag a parse error.
                yield (int((time.monotonic() - t_request_sent) * 1000.0),
                       event_type, "\n".join(data_lines), None)
            return
        # Normalize line terminator
        if raw.endswith(b"\r\n"):
            line = raw[:-2]
        elif raw.endswith(b"\n"):
            line = raw[:-1]
        elif raw.endswith(b"\r"):
            line = raw[:-1]
        else:
            line = raw
        try:
            decoded = line.decode("utf-8")
        except UnicodeDecodeError:
            decoded = line.decode("utf-8", errors="replace")

        if decoded == "":
            # Record terminator — emit accumulated event.
            if data_lines or event_type:
                data_raw = "\n".join(data_lines)
                data_json = None
                if data_raw:
                    try:
                        data_json = json.loads(data_raw)
                    except Exception:
                        data_json = None
                ts_ms = int(
                    (time.monotonic() - t_request_sent) * 1000.0)
                yield (ts_ms, event_type, data_raw, data_json)
            event_type = ""
            data_lines = []
            continue

        if decoded.startswith(":"):
            # Comment / keepalive — ignore.
            continue

        if decoded.startswith("event:"):
            event_type = decoded[len("event:"):].lstrip()
        elif decoded.startswith("data:"):
            data_lines.append(decoded[len("data:"):].lstrip(" "))
        # Other SSE fields (id:, retry:) are silently ignored.


def http_post_streaming(host, port, path, body_dict,
                        line_timeout_s, overall_timeout_s):
    """Issue one streaming POST. Returns:
        t_request_sent_monotonic, events_list, response_status,
        headers_dict, stream_parse_error_or_empty_str

    `events_list` is a list of dicts with keys:
        ts_ms_from_request, event_type, data_raw, data_json

    The function fully consumes the stream before returning so the
    caller can run gate evaluations synchronously. Per-record arrival
    timestamps are captured inside _iter_sse_records.
    """
    conn = http.client.HTTPConnection(host, port, timeout=line_timeout_s)
    events = []
    parse_error = ""
    try:
        payload = json.dumps(body_dict).encode("utf-8")
        conn.putrequest("POST", path)
        conn.putheader("Content-Type", "application/json")
        conn.putheader("Content-Length", str(len(payload)))
        conn.endheaders()
        conn.send(payload)
        t_request_sent = time.monotonic()

        resp = conn.getresponse()
        status = resp.status
        headers = dict(resp.getheaders())
        if status != 200:
            try:
                body_bytes = resp.read()
            except Exception:
                body_bytes = b""
            parse_error = "non_200_status_{}; body={!r}".format(
                status, body_bytes[:512])
            return t_request_sent, events, status, headers, parse_error

        try:
            for ts_ms, evt, data_raw, data_json in _iter_sse_records(
                    resp, t_request_sent,
                    line_timeout_s=line_timeout_s,
                    overall_timeout_s=overall_timeout_s):
                events.append({
                    "ts_ms_from_request": ts_ms,
                    "event_type":         evt,
                    "data_raw":           data_raw,
                    "data_json":          data_json,
                })
        except TimeoutError as e:
            parse_error = "stream_timeout: {}".format(e)
        except Exception as e:
            parse_error = "stream_read_error: {}".format(e)
        return t_request_sent, events, status, headers, parse_error
    finally:
        try:
            conn.close()
        except Exception:
            pass


def write_args_file(path, argv):
    with open(path, "w", encoding="utf-8") as f:
        for a in argv:
            f.write(a)
            f.write("\n")


def launch_server(label, argv, stdout_path, stderr_path):
    """Start a server with stdout/stderr redirected to files. Returns Popen."""
    fout = open(stdout_path, "wb")
    ferr = open(stderr_path, "wb")
    proc = subprocess.Popen(
        argv,
        stdout=fout,
        stderr=ferr,
        stdin=subprocess.DEVNULL,
        start_new_session=True,  # so SIGTERM hits this process group cleanly
    )
    # Attach the file handles to the proc so callers can close them.
    proc._fout = fout  # type: ignore[attr-defined]
    proc._ferr = ferr  # type: ignore[attr-defined]
    proc._label = label  # type: ignore[attr-defined]
    return proc


def stop_server(proc, grace_seconds):
    """SIGTERM, wait grace, SIGKILL on timeout. Closes redirected file handles.

    Returns (exit_code, term_method) where term_method is "sigterm" or "sigkill"
    or "already_exited".
    """
    if proc.poll() is not None:
        method = "already_exited"
    else:
        try:
            proc.send_signal(signal.SIGTERM)
        except ProcessLookupError:
            method = "already_exited"
        else:
            method = "sigterm"
        try:
            proc.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            proc.kill()
            method = "sigkill"
            try:
                proc.wait(timeout=grace_seconds)
            except subprocess.TimeoutExpired:
                pass
    exit_code = proc.returncode
    for fh_name in ("_fout", "_ferr"):
        fh = getattr(proc, fh_name, None)
        if fh is not None:
            try:
                fh.close()
            except Exception:
                pass
    return exit_code, method


def server_pid_running(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _empty_row(label, workload, iteration, body, exception_msg, elapsed_ms):
    return {
        "server":                  label,
        "workload_id":             workload["workload_id"],
        "prompt_id":               workload["prompt_id"],
        "is_canonical_anchor":     workload["is_canonical_anchor"],
        "iteration":               iteration,
        "ts_utc":                  utc_iso(),
        "http_status":             -1,
        "ok":                      False,
        "latency_ms":              elapsed_ms,
        "stream_mode":             "non_streaming",
        "total_latency_ms":        elapsed_ms,
        "first_token_latency_ms":  -1,
        "first_event_latency_ms":  -1,
        "token_event_count":       0,
        "done_seen":               False,
        "done_status":             "",
        "stream_parse_error":      "",
        "text":                    "",
        "text_sha256":             "",
        "text_normalized_sha256":  "",
        "text_len_chars":          0,
        "n_decoded":               -1,
        "hash":                    "",
        "prompt_tokens":           -1,
        "decode_budget_requested": workload["decode_budget"],
        "server_request_id":       -1,
        "server_status":           "",
        "parse_error":             exception_msg,
        "raw_json":                {},
        "request_body":            body,
    }


def run_one_server(label, adapter, cfg, port, run_dir, notes):
    """Drive one server across the full workload matrix.

    Server is launched once per leg, reused across all workloads and
    iterations. Per-seq KV cleanup between requests is the engine
    invariant (M6/M7) on hpx-server and llama-server's --parallel 1
    slot handling.

    Returns a list of per-iteration rows across all workloads.
    """
    argv = adapter.build_args(cfg, port)
    stdout_path = run_dir / "{}.stdout".format(label)
    stderr_path = run_dir / "{}.stderr".format(label)
    args_path   = run_dir / "{}.args".format(label)
    write_args_file(args_path, argv)

    notes.append("[{}] launch argv:".format(label))
    for a in argv:
        notes.append("    {}".format(a))

    t_launch = time.monotonic()
    proc = launch_server(label, argv, stdout_path, stderr_path)
    notes.append("[{}] launched pid={} at {}".format(
        label, proc.pid, utc_iso()))

    rows = []
    readiness_ms    = None
    readiness_error = ""
    warmup_status   = None

    workloads = cfg["workload_matrix"]
    iters_per = cfg["iterations_per_workload"]

    try:
        # Readiness: TCP connect poll then warm-up POST poll.
        try:
            elapsed = tcp_wait(
                "127.0.0.1", port,
                interval_s=cfg["readiness"]["tcp_poll_interval_seconds"],
                timeout_s=cfg["readiness"]["tcp_poll_timeout_seconds"])
            notes.append("[{}] tcp ready in {:.3f}s".format(label, elapsed))
        except TimeoutError as e:
            readiness_error = "tcp_wait_timeout: {}".format(e)
            notes.append("[{}] {}".format(label, readiness_error))
            return rows

        warmup_deadline = (time.monotonic()
                           + cfg["readiness"]["tcp_poll_timeout_seconds"])
        warmup_attempts = 0
        last_warmup_error = ""
        while time.monotonic() < warmup_deadline:
            warmup_attempts += 1
            try:
                wstatus, _wh, _wb = http_post_json(
                    "127.0.0.1", port, "/completion",
                    adapter.warmup_body(),
                    timeout_s=cfg["readiness"]
                                 ["warmup_request_timeout_seconds"])
                warmup_status = wstatus
                if wstatus == 200:
                    break
                last_warmup_error = "warmup_status_{}".format(wstatus)
            except Exception as e:
                last_warmup_error = "warmup_exception: {}".format(e)
            time.sleep(cfg["readiness"]["tcp_poll_interval_seconds"])
        notes.append(
            "[{}] warmup POST final status={} attempts={}".format(
                label, warmup_status, warmup_attempts))
        if warmup_status != 200:
            readiness_error = (last_warmup_error
                               or "warmup_timeout_after_{}_attempts".format(
                                   warmup_attempts))
            notes.append("[{}] {}".format(label, readiness_error))
            return rows

        readiness_ms = int((time.monotonic() - t_launch) * 1000.0)

        # Workload loop. Iteration counter resets per shape so per-shape
        # gates can index rows naturally.
        streaming = bool(cfg.get("streaming", False))
        for w in workloads:
            notes.append("[{}] === workload {} (prompt_id={} budget={}) "
                         "===".format(label, w["workload_id"],
                                      w["prompt_id"], w["decode_budget"]))
            for i in range(1, iters_per + 1):
                if streaming:
                    body = adapter.stream_request_body(
                        w["prompt"], w["decode_budget"])
                    try:
                        (t_sent, events, http_status, headers,
                         stream_err) = http_post_streaming(
                            "127.0.0.1", port, "/completion", body,
                            line_timeout_s=30.0,
                            overall_timeout_s=60.0)
                        t_finished = time.monotonic()
                        total_ms = int((t_finished - t_sent) * 1000.0)
                        if http_status != 200:
                            parsed = {
                                "ok": False, "http_status": http_status,
                                "text": "", "text_sha256": "",
                                "text_normalized_sha256": "",
                                "text_len_chars": 0,
                                "n_decoded": -1, "hash": "",
                                "prompt_tokens": -1,
                                "server_request_id": -1,
                                "server_status": "",
                                "raw_json": {},
                                "parse_error": stream_err or
                                    "non_200_status_{}".format(http_status),
                                "token_event_count": 0,
                                "done_seen": False,
                                "done_status": "",
                                "first_content_event_ts_ms": -1,
                                "first_event_ts_ms": -1,
                                "last_event_ts_ms": -1,
                            }
                        else:
                            parsed = adapter.parse_stream_events(events)
                            if stream_err and not parsed["parse_error"]:
                                parsed["parse_error"] = stream_err
                        row = {
                            "server":                  label,
                            "workload_id":             w["workload_id"],
                            "prompt_id":               w["prompt_id"],
                            "is_canonical_anchor":
                                w["is_canonical_anchor"],
                            "iteration":               i,
                            "ts_utc":                  utc_iso(),
                            "http_status":             parsed["http_status"],
                            "ok":                      parsed["ok"],
                            "latency_ms":              total_ms,
                            "total_latency_ms":        total_ms,
                            "first_token_latency_ms":
                                parsed["first_content_event_ts_ms"],
                            "first_event_latency_ms":
                                parsed["first_event_ts_ms"],
                            "token_event_count":
                                parsed["token_event_count"],
                            "done_seen":               parsed["done_seen"],
                            "done_status":             parsed["done_status"],
                            "stream_parse_error":
                                parsed["parse_error"],
                            "stream_mode":             "streaming",
                            "text":                    parsed["text"],
                            "text_sha256":             parsed["text_sha256"],
                            "text_normalized_sha256":
                                parsed["text_normalized_sha256"],
                            "text_len_chars":
                                parsed["text_len_chars"],
                            "n_decoded":               parsed["n_decoded"],
                            "hash":                    parsed["hash"],
                            "prompt_tokens":
                                parsed["prompt_tokens"],
                            "decode_budget_requested":
                                w["decode_budget"],
                            "server_request_id":
                                parsed["server_request_id"],
                            "server_status":
                                parsed["server_status"],
                            "parse_error":             parsed["parse_error"],
                            "raw_json":                parsed["raw_json"],
                            "request_body":            body,
                            "streaming_events":        events,
                            "token_event_times_ms": [
                                e["ts_ms_from_request"] for e in events
                                if (e["event_type"] == "token") or
                                   (e["event_type"] == "" and
                                    isinstance(e["data_json"], dict) and
                                    e["data_json"].get("stop") is False and
                                    bool(e["data_json"].get("content")))
                            ],
                            "streaming_events_count":  len(events),
                            "raw_stream_excerpt":
                                ("\n\n".join(
                                    e["data_raw"] for e in events
                                )[:2048]),
                        }
                    except Exception as e:
                        total_ms = 0
                        row = _empty_row(
                            label, w, i, body,
                            "request_exception: {}".format(e), total_ms)
                        row["stream_mode"]            = "streaming"
                        row["total_latency_ms"]       = total_ms
                        row["first_token_latency_ms"] = -1
                        row["first_event_latency_ms"] = -1
                        row["token_event_count"]      = 0
                        row["done_seen"]              = False
                        row["done_status"]            = ""
                        row["stream_parse_error"]     = row["parse_error"]
                        row["streaming_events"]       = []
                        row["token_event_times_ms"]   = []
                        row["streaming_events_count"] = 0
                        row["raw_stream_excerpt"]     = ""
                    rows.append(row)
                    notes.append(
                        ("[{}] workload={} iter={} stream "
                         "status={} ok={} n_decoded={} hash={} "
                         "token_events={} done_seen={} "
                         "first_token_ms={} first_event_ms={} "
                         "total_ms={} parse_error={!r}").format(
                            label, w["workload_id"], i,
                            row["http_status"], row["ok"],
                            row["n_decoded"], row["hash"],
                            row["token_event_count"], row["done_seen"],
                            row["first_token_latency_ms"],
                            row["first_event_latency_ms"],
                            row["total_latency_ms"],
                            row["stream_parse_error"]))
                    continue

                # ---- non-streaming path (unchanged from M8d-fix) ----
                body = adapter.request_body(w["prompt"], w["decode_budget"])
                t0 = time.monotonic()
                try:
                    status, headers, raw = http_post_json(
                        "127.0.0.1", port, "/completion", body,
                        timeout_s=60.0)
                    elapsed_ms = int((time.monotonic() - t0) * 1000.0)
                    parsed = adapter.parse_response(status, headers, raw)
                    row = {
                        "server":                  label,
                        "workload_id":             w["workload_id"],
                        "prompt_id":               w["prompt_id"],
                        "is_canonical_anchor":     w["is_canonical_anchor"],
                        "iteration":               i,
                        "ts_utc":                  utc_iso(),
                        "http_status":             parsed["http_status"],
                        "ok":                      parsed["ok"],
                        "latency_ms":              elapsed_ms,
                        "stream_mode":             "non_streaming",
                        "text":                    parsed["text"],
                        "text_sha256":             parsed["text_sha256"],
                        "text_normalized_sha256":
                            parsed["text_normalized_sha256"],
                        "text_len_chars":          parsed["text_len_chars"],
                        "n_decoded":               parsed["n_decoded"],
                        "hash":                    parsed["hash"],
                        "prompt_tokens":           parsed["prompt_tokens"],
                        "decode_budget_requested": w["decode_budget"],
                        "server_request_id":
                            parsed["server_request_id"],
                        "server_status":           parsed["server_status"],
                        "parse_error":             parsed["parse_error"],
                        "raw_json":                parsed["raw_json"],
                        "request_body":            body,
                    }
                except Exception as e:
                    elapsed_ms = int((time.monotonic() - t0) * 1000.0)
                    row = _empty_row(
                        label, w, i, body,
                        "request_exception: {}".format(e), elapsed_ms)
                    row["stream_mode"] = "non_streaming"
                rows.append(row)
                notes.append(
                    ("[{}] workload={} iter={} status={} ok={} "
                     "n_decoded={} hash={} latency_ms={} "
                     "parse_error={!r}").format(
                        label, w["workload_id"], i, row["http_status"],
                        row["ok"], row["n_decoded"], row["hash"],
                        row["latency_ms"], row["parse_error"]))
    finally:
        exit_code, term_method = stop_server(
            proc, grace_seconds=cfg["shutdown"]["sigterm_grace_seconds"])
        notes.append(
            "[{}] stopped exit={} term_method={} pid_still_running={}".format(
                label, exit_code, term_method,
                server_pid_running(proc.pid)))

    if readiness_ms is not None:
        notes.append("[{}] readiness_ms={}".format(label, readiness_ms))
    if readiness_error:
        notes.append("[{}] readiness_error={}".format(label, readiness_error))
    if warmup_status is not None:
        notes.append("[{}] warmup_status={}".format(label, warmup_status))

    return rows


def write_summary_csv(rows, path):
    fields = [
        "server", "iteration",
        "workload_id", "prompt_id", "is_canonical_anchor",
        "ts_utc", "stream_mode",
        "http_status", "ok", "latency_ms",
        "total_latency_ms", "first_token_latency_ms",
        "first_event_latency_ms",
        "token_event_count", "done_seen", "done_status",
        "stream_parse_error",
        "n_decoded", "decode_budget_requested", "hash",
        "text_sha256", "text_normalized_sha256", "text_len_chars",
        "prompt_tokens", "server_request_id", "server_status",
        "parse_error", "text",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def write_stream_events_jsonl(rows, path):
    """Flat per-record evidence trail for streaming runs.

    One JSONL row per parsed SSE record across all (server, workload,
    iteration) combinations. Empty file if no streaming row exists.
    """
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            events = r.get("streaming_events") or []
            for ev in events:
                out = {
                    "server":             r["server"],
                    "workload_id":        r["workload_id"],
                    "iteration":          r["iteration"],
                    "ts_ms_from_request": ev["ts_ms_from_request"],
                    "event_type":         ev["event_type"],
                    "data_json":          ev["data_json"],
                }
                f.write(json.dumps(out, ensure_ascii=False))
                f.write("\n")


def write_jsonl(rows, path):
    """Write per-iteration rows. `streaming_events` is pruned because
    the per-record evidence trail lives in stream_events.jsonl; keeping
    it here would duplicate that data per row."""
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            slim = {k: v for k, v in r.items() if k != "streaming_events"}
            f.write(json.dumps(slim, ensure_ascii=False))
            f.write("\n")


def port_in_use(host, port):
    """Return True iff the given host:port currently accepts a TCP connect."""
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


CANONICAL_ANCHOR_PROMPT = "Hello, my name is"
CANONICAL_ANCHOR_BUDGET = 8


def normalize_config(cfg):
    """Resolve `cfg` into the M8d canonical shape.

    Returns (normalized_cfg, warning_notes, fatal_error_msg_or_None).

    Behavior:
      - If `workload_matrix` is present, use it. If old single-shape fields
        (`prompt`, `decode_budget`, `iterations`) are also present, emit a
        factual warning into run_notes.txt and prefer the matrix.
      - If `workload_matrix` is absent and old single-shape fields exist,
        synthesize a 1-shape matrix and mark `_backward_compat=True`.
      - Reject multiple canonical anchors at load time (fatal).
      - Zero canonical anchors is a warning, not fatal.
    """
    warnings = []
    has_matrix = "workload_matrix" in cfg
    has_legacy = any(k in cfg for k in
                     ("prompt", "decode_budget", "iterations"))

    backward_compat = False

    if has_matrix and has_legacy:
        warnings.append(
            "config_warn: both 'workload_matrix' and legacy "
            "single-shape fields present; using workload_matrix and "
            "ignoring 'prompt'/'decode_budget'/'iterations'")

    if has_matrix:
        wm = cfg["workload_matrix"]
        if "iterations_per_workload" not in cfg:
            return None, warnings, (
                "iterations_per_workload required when "
                "workload_matrix is set")
        iters = cfg["iterations_per_workload"]
    elif has_legacy:
        prompt = cfg.get("prompt")
        budget = cfg.get("decode_budget")
        iters  = cfg.get("iterations")
        if prompt is None or budget is None or iters is None:
            return None, warnings, (
                "legacy single-shape config missing one of "
                "prompt / decode_budget / iterations")
        wm = [{
            "workload_id":         "p0_b{}".format(budget),
            "prompt_id":           "p0",
            "prompt":              prompt,
            "decode_budget":       budget,
            "is_canonical_anchor": (
                prompt == CANONICAL_ANCHOR_PROMPT
                and budget == CANONICAL_ANCHOR_BUDGET),
        }]
        backward_compat = True
        warnings.append(
            "config_info: backward-compat mode "
            "(no workload_matrix; synthesized 1-shape matrix)")
    else:
        return None, warnings, (
            "config requires 'workload_matrix' or legacy "
            "single-shape fields (prompt/decode_budget/iterations)")

    if not isinstance(wm, list) or not wm:
        return None, warnings, "workload_matrix must be a non-empty list"

    seen_ids = set()
    anchors  = 0
    for idx, w in enumerate(wm):
        for k in ("workload_id", "prompt_id", "prompt",
                  "decode_budget", "is_canonical_anchor"):
            if k not in w:
                return None, warnings, (
                    "workload_matrix[{}] missing required key '{}'".format(
                        idx, k))
        if not isinstance(w["workload_id"], str) or not w["workload_id"]:
            return None, warnings, (
                "workload_matrix[{}].workload_id must be non-empty "
                "string".format(idx))
        if w["workload_id"] in seen_ids:
            return None, warnings, (
                "duplicate workload_id '{}'".format(w["workload_id"]))
        seen_ids.add(w["workload_id"])
        if not isinstance(w["decode_budget"], int) or w["decode_budget"] < 1:
            return None, warnings, (
                "workload_matrix[{}].decode_budget must be positive "
                "int".format(idx))
        if not isinstance(w["is_canonical_anchor"], bool):
            return None, warnings, (
                "workload_matrix[{}].is_canonical_anchor must be "
                "bool".format(idx))
        if w["is_canonical_anchor"]:
            anchors += 1

    if anchors > 1:
        return None, warnings, (
            "multiple canonical anchors in workload_matrix "
            "({}); reject".format(anchors))
    if anchors == 0:
        warnings.append(
            "config_warn: zero canonical anchors; HPX hash equality "
            "gate is not exercised")

    if not isinstance(iters, int) or iters < 1:
        return None, warnings, (
            "iterations_per_workload must be a positive int")

    normalized = dict(cfg)
    normalized["workload_matrix"]         = wm
    normalized["iterations_per_workload"] = iters
    normalized["_backward_compat"]        = backward_compat
    return normalized, warnings, None


def write_match_conditions(cfg, path):
    """Write the *intended* matched-condition decisions for the run.

    Smaller and more audit-friendly than config.json. The values here
    describe what the harness believes is being held equal between the
    two servers; they are descriptive, not enforcement.

    Expects `cfg` already normalized by `normalize_config` so that
    `workload_matrix` and `iterations_per_workload` are present.
    """
    md = {
        "model_path":              cfg["model_path"],
        "ctx_size":                cfg["ctx_size"],
        "threads":                 cfg["threads"],
        "concurrency":             1,
        "streaming":               bool(cfg.get("streaming", False)),
        "iterations_per_workload": cfg["iterations_per_workload"],
        "workloads": [
            {
                "workload_id":         w["workload_id"],
                "prompt_id":           w["prompt_id"],
                "prompt":              w["prompt"],
                "decode_budget":       w["decode_budget"],
                "is_canonical_anchor": w["is_canonical_anchor"],
            }
            for w in cfg["workload_matrix"]
        ],
        "hpx_server": {
            "sampling": "greedy (argmax) — no sampler chain exposed",
            "request_body_pins": {
                "prompt":        "<per workload>",
                "decode_budget": "<per workload>",
            },
        },
        "llama_server": {
            "sampling": "pinned chain ['top_k','temperature']",
            "request_body_pins": {
                "temperature":       0,
                "top_k":             1,
                "top_p":             1.0,
                "min_p":             1.0,
                "typical_p":         1.0,
                "repeat_penalty":    1.0,
                "presence_penalty":  0.0,
                "frequency_penalty": 0.0,
                "dry_multiplier":    0.0,
                "xtc_probability":   0.0,
                "mirostat":          0,
                "n_probs":           0,
                "samplers":          ["top_k", "temperature"],
                "stop":              [],
                "seed":              0,
                "stream":            False,
                "cache_prompt":      False,
            },
        },
    }
    if cfg.get("_backward_compat"):
        md["legacy_single_shape"] = {
            "note":          ("backward-compat: synthesized 1-shape matrix "
                              "from legacy top-level prompt/decode_budget/"
                              "iterations"),
            "prompt":        cfg.get("prompt"),
            "decode_budget": cfg.get("decode_budget"),
            "iterations":    cfg.get("iterations"),
        }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(md, f, indent=2)
        f.write("\n")


def write_known_mismatches(path):
    """Static text listing acknowledged mismatches between the two servers.

    Reviewer-facing. The line at the bottom is the operational rule that
    governs how this run's output may be cited.
    """
    lines = [
        "Known mismatches between hpx-server and llama-server "
        "(M8c harness {}):".format(HARNESS_VERSION),
        "",
        "- llama-server's /completion string-prompt path adds an implicit",
        "  BOS token (`<s>`) via the model tokenizer. hpx-server's",
        "  internal tokenization is not exposed via the HTTP response, so",
        "  BOS handling on the hpx-server side cannot be directly",
        "  observed from the harness.",
        "",
        "- llama-server's `content` field includes a leading space for",
        "  the first generated token (tokenizer convention). hpx-server's",
        "  `text` field does not. Byte-equality across servers is",
        "  therefore not meaningful even when underlying tokens agree.",
        "",
        "- llama-server uses a pinned sampler chain ['top_k','temperature']",
        "  reachable through HTTP. hpx-server's default greedy path is a",
        "  local argmax with no exposed sampler chain. Even with",
        "  top_k=1 + temperature=0 on the llama-server side, the two",
        "  paths are not equivalent implementations.",
        "",
        "- hpx-server does not expose a prompt-token-count in its",
        "  /completion response. The `prompt_tokens` field in",
        "  summary.csv is -1 for hpx-server and is taken from",
        "  llama-server's `tokens_evaluated` for llama-server. The",
        "  values are not comparable.",
        "",
        "- Within-shape stability is asserted per (prompt, decode_budget)",
        "  pair. Across shapes, hashes and texts are expected to differ.",
        "  Cross-shape comparison (e.g. budget=8 vs budget=32 on the same",
        "  prompt) is not sanctioned: budget changes batch shape and",
        "  per-iteration cost on both servers.",
        "",
        "- decode_budget / n_predict is treated as an upper bound.",
        "  Either server may stop earlier on EOG (end-of-generation).",
        "  Per-shape n_decoded stability is asserted within each",
        "  server; cross-server equality is recorded but not required.",
        "  For prompts that end in sentence-final punctuation (e.g. '.'),",
        "  greedy argmax on TinyLlama often selects EOG immediately at",
        "  the post-prefill site, yielding n_decoded=0 on hpx-server.",
        "  This is the documented engine behavior (engine.cpp EOG branch",
        "  finalizes as completed without appending the EOG token).",
        "  llama-server may stop at a different position depending on",
        "  its sampler chain and tokenizer-detokenizer conventions.",
        "",
        "- (M8e streaming) Streaming wire formats differ between the",
        "  two servers. hpx-server M7c emits SSE records with explicit",
        "  `event: token` / `event: done` headers. llama-server emits",
        "  bare `data: {json}` records with no `event:` field and",
        "  signals termination via `stop: true` plus `stop_type` in",
        "  the terminal chunk. The harness records both shapes through",
        "  a single SSE parser; cross-server token-event-count equality",
        "  is not gated. TTFT (first_token_latency_ms) and",
        "  first_event_latency_ms are recorded per iteration as raw",
        "  monotonic deltas from the moment the request payload",
        "  finished writing. They are client-observed (include",
        "  loopback) and are not aggregated, averaged, or compared",
        "  across servers.",
        "",
        "- (M8e streaming detokenization) hpx-server streams one token",
        "  at a time and emits the raw single-token detokenization.",
        "  SentencePiece word-boundary spacing is reconstructed only",
        "  when detokenizing a multi-token run, so per-token text can",
        "  lose the leading-space glue between words. llama-server",
        "  streams `content` chunks produced by its own incremental",
        "  detokenizer with spacing preserved. Cross-server streaming",
        "  text equality is therefore not meaningful and is not gated.",
        "",
        "- (ok-field semantics) The `ok` column has divergent meaning",
        "  between modes. Non-streaming adapters set `ok` based on",
        "  non-empty `text`. Streaming adapters set `ok` based on a",
        "  clean terminal stream record (hpx-server: done_status ==",
        "  'completed'; llama-server: terminal `stop: true` chunk).",
        "  For EOG-stop rows, empty streaming text can still be a",
        "  successful streaming completion, so a False `ok` in",
        "  non-streaming mode and a True `ok` in streaming mode are",
        "  not contradictory and do not indicate a regression.",
        "",
        "Descriptive raw values only. No cross-server comparison is",
        "sanctioned without explicit alignment of sampler chain, prompt",
        "tokenization, and detokenization conventions.",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")


def evaluate_gates(cfg, hpx_rows, llama_rows):
    """Return (passed: bool, gate_notes: list[str]).

    Hard-fail semantics per M8c/M8d-fix approval. Per-row and per-shape
    failures are collected; the function does not short-circuit so that
    all evidence is recorded.

    Gates (M8d-fix):
      - row count == len(workload_matrix) * iterations_per_workload
        per server
      - per row: http_status == 200, parse_error == "",
        0 <= n_decoded <= decode_budget_requested
      - per (server, workload_id): text_normalized_sha256 stable AND
        n_decoded stable across all iterations
      - per (hpx_server, workload_id): `hash` stable across all
        iterations
      - for the canonical anchor shape only: every hpx_server row has
        `hash == CANONICAL_HPX_HASH` AND `n_decoded == decode_budget`
      - cross-server n_decoded equality is *recorded* per shape, NOT
        gated. EOG-stop semantics differ across servers and the M8d-fix
        treats decode_budget as an upper bound.
    """
    notes  = []
    passed = True

    def fail(msg):
        nonlocal passed
        passed = False
        notes.append("GATE FAIL: " + msg)

    def info(msg):
        notes.append("gate: " + msg)

    workloads     = cfg["workload_matrix"]
    iters_per     = cfg["iterations_per_workload"]
    expected_rows = len(workloads) * iters_per

    if len(hpx_rows) != expected_rows:
        fail("hpx_server total rows {} != expected {}".format(
            len(hpx_rows), expected_rows))
    if len(llama_rows) != expected_rows:
        fail("llama_server total rows {} != expected {}".format(
            len(llama_rows), expected_rows))

    # Per-row checks. `n_decoded` is bounded by budget (upper) and 0
    # (lower); equality to budget is no longer required at the row
    # level. EOG-stop is a legal early termination.
    for r in hpx_rows + llama_rows:
        if r["http_status"] != 200:
            fail("{} workload={} iter={} http_status={}".format(
                r["server"], r["workload_id"], r["iteration"],
                r["http_status"]))
        if r["parse_error"]:
            fail("{} workload={} iter={} parse_error={!r}".format(
                r["server"], r["workload_id"], r["iteration"],
                r["parse_error"]))
        nd  = r["n_decoded"]
        bud = r["decode_budget_requested"]
        if not (0 <= nd <= bud):
            fail(("{} workload={} iter={} n_decoded={} out of "
                  "range [0, {}]").format(
                r["server"], r["workload_id"], r["iteration"],
                nd, bud))
        # Streaming-specific row gates (no-op for non_streaming rows).
        if r.get("stream_mode") == "streaming":
            if r.get("stream_parse_error"):
                fail(("{} workload={} iter={} stream_parse_error="
                      "{!r}").format(
                    r["server"], r["workload_id"], r["iteration"],
                    r["stream_parse_error"]))
            if not r.get("done_seen"):
                fail("{} workload={} iter={} done_seen=False".format(
                    r["server"], r["workload_id"], r["iteration"]))
            # TTFT and first-event positive only when content arrived.
            if r.get("token_event_count", 0) > 0:
                if r.get("first_token_latency_ms", -1) <= 0:
                    fail(("{} workload={} iter={} "
                          "first_token_latency_ms={} <= 0 "
                          "(token_event_count>0)").format(
                        r["server"], r["workload_id"], r["iteration"],
                        r.get("first_token_latency_ms", -1)))
            if r.get("first_event_latency_ms", -1) <= 0:
                fail(("{} workload={} iter={} "
                      "first_event_latency_ms={} <= 0").format(
                    r["server"], r["workload_id"], r["iteration"],
                    r.get("first_event_latency_ms", -1)))
            if r.get("total_latency_ms", 0) <= 0:
                fail(("{} workload={} iter={} "
                      "total_latency_ms={} <= 0").format(
                    r["server"], r["workload_id"], r["iteration"],
                    r.get("total_latency_ms", 0)))

    def group_by_workload(rows):
        groups = {}
        for r in rows:
            groups.setdefault(r["workload_id"], []).append(r)
        return groups

    hpx_groups   = group_by_workload(hpx_rows)
    llama_groups = group_by_workload(llama_rows)

    for w in workloads:
        wid       = w["workload_id"]
        hpx_grp   = hpx_groups.get(wid, [])
        llama_grp = llama_groups.get(wid, [])

        if len(hpx_grp) != iters_per:
            fail("shape {} hpx_server iterations {} != expected {}".format(
                wid, len(hpx_grp), iters_per))
        if len(llama_grp) != iters_per:
            fail("shape {} llama_server iterations {} != expected {}".format(
                wid, len(llama_grp), iters_per))

        def check_text_stab(label, grp):
            sigs = {r["text_normalized_sha256"] for r in grp}
            if len(sigs) == 0:
                fail("shape {} {}: no rows to check stability".format(
                    wid, label))
            elif len(sigs) == 1:
                info(("shape {} {} text_normalized_sha256 stable: "
                      "{}").format(wid, label, next(iter(sigs))))
            else:
                fail(("shape {} {} text_normalized_sha256 drifted "
                      "across iterations: {} distinct values").format(
                    wid, label, len(sigs)))

        def check_n_decoded_stab(label, grp):
            vals = sorted({r["n_decoded"] for r in grp})
            if not vals:
                fail("shape {} {}: no rows for n_decoded stability".format(
                    wid, label))
            elif len(vals) == 1:
                info("shape {} {} n_decoded stable: {}".format(
                    wid, label, vals[0]))
            else:
                fail(("shape {} {} n_decoded drifted across iterations: "
                      "{} distinct values {}").format(
                    wid, label, len(vals), vals))

        check_text_stab("hpx_server",   hpx_grp)
        check_text_stab("llama_server", llama_grp)
        check_n_decoded_stab("hpx_server",   hpx_grp)
        check_n_decoded_stab("llama_server", llama_grp)

        # hpx-server hash stability within shape
        hpx_hashes = {r["hash"] for r in hpx_grp}
        if len(hpx_hashes) == 1:
            info("shape {} hpx_server hash stable: {}".format(
                wid, next(iter(hpx_hashes))))
        elif len(hpx_hashes) > 1:
            fail(("shape {} hpx_server hash drifted across iterations: "
                  "{} distinct values").format(wid, len(hpx_hashes)))

        # Per-shape streaming stability checks. token_event_count must
        # be stable within (server, shape) when in streaming mode. The
        # check is a no-op for non_streaming rows because every row
        # records token_event_count=0 in that mode (consistent value).
        def check_token_count_stab(label, grp):
            if not grp:
                return
            if grp[0].get("stream_mode") != "streaming":
                return
            vals = sorted({r.get("token_event_count", 0) for r in grp})
            if len(vals) == 1:
                info(("shape {} {} token_event_count stable: "
                      "{}").format(wid, label, vals[0]))
            else:
                fail(("shape {} {} token_event_count drifted across "
                      "iterations: {} distinct values {}").format(
                    wid, label, len(vals), vals))

        check_token_count_stab("hpx_server",   hpx_grp)
        check_token_count_stab("llama_server", llama_grp)

        # Canonical anchor: hash equality AND exact-budget retained.
        if w["is_canonical_anchor"]:
            for r in hpx_grp:
                if r["hash"] != CANONICAL_HPX_HASH:
                    fail(("shape {} (canonical anchor) hpx_server "
                          "iter={} hash {} != canonical {}").format(
                        wid, r["iteration"], r["hash"],
                        CANONICAL_HPX_HASH))
                if r["n_decoded"] != r["decode_budget_requested"]:
                    fail(("shape {} (canonical anchor) hpx_server "
                          "iter={} n_decoded={} != decode_budget {} "
                          "(anchor must decode to exact budget)").format(
                        wid, r["iteration"], r["n_decoded"],
                        r["decode_budget_requested"]))
                if r.get("stream_mode") == "streaming":
                    if r.get("done_status") != "completed":
                        fail(("shape {} (canonical anchor) hpx_server "
                              "iter={} done_status={!r} != "
                              "'completed'").format(
                            wid, r["iteration"],
                            r.get("done_status")))
                    if r.get("token_event_count") != \
                            r["decode_budget_requested"]:
                        fail(("shape {} (canonical anchor) hpx_server "
                              "iter={} token_event_count={} != "
                              "decode_budget {}").format(
                            wid, r["iteration"],
                            r.get("token_event_count"),
                            r["decode_budget_requested"]))

        # Cross-server n_decoded recorded but NOT gated. EOG-stop
        # semantics differ; record the comparison for the operator.
        hpx_nd_set   = sorted({r["n_decoded"] for r in hpx_grp})
        llama_nd_set = sorted({r["n_decoded"] for r in llama_grp})
        if hpx_nd_set == llama_nd_set:
            info("shape {} n_decoded matches across servers: {}".format(
                wid, hpx_nd_set))
        else:
            info(("shape {} n_decoded differs across servers "
                  "(recorded, not gated): hpx={} llama={}").format(
                wid, hpx_nd_set, llama_nd_set))

        # Per-shape EOG-stop fact: if any row in either group stopped
        # before budget, note it factually. Helps the operator see
        # which shapes are EOG-sensitive without inferring from rows.
        for label, grp in (("hpx_server", hpx_grp),
                           ("llama_server", llama_grp)):
            if grp:
                rep = grp[0]
                if rep["n_decoded"] < rep["decode_budget_requested"]:
                    info(("shape {} {} stopped early: n_decoded={} "
                          "of budget={} (EOG-stop accepted)").format(
                        wid, label, rep["n_decoded"],
                        rep["decode_budget_requested"]))

    return passed, notes


def check_forbidden_words(notes_text):
    """Return list of forbidden tokens found in notes_text. Empty list = OK.

    Case-insensitive, word-boundary anchored.
    """
    hits = set()
    for m in FORBIDDEN_RE.finditer(notes_text):
        hits.add(m.group(1).lower())
    return sorted(hits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="path to a JSON config file")
    ap.add_argument("--results-root", default=None,
                    help="override results root directory")
    args = ap.parse_args()

    cfg_path = pathlib.Path(args.config).resolve()
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw_cfg = json.load(f)

    cfg, cfg_warnings, cfg_fatal = normalize_config(raw_cfg)
    if cfg_fatal is not None:
        print("FATAL: config error: {}".format(cfg_fatal), file=sys.stderr)
        for w in cfg_warnings:
            print(w, file=sys.stderr)
        print("gates=FAIL")
        return 2

    here = pathlib.Path(__file__).resolve().parent
    results_root = (pathlib.Path(args.results_root).resolve()
                    if args.results_root
                    else here / "results")
    results_root.mkdir(parents=True, exist_ok=True)
    run_id = make_run_id()
    run_dir = results_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    # Echo the merged config for the run.
    out_cfg = {
        "harness_version": HARNESS_VERSION,
        "run_id": run_id,
        "ts_utc": utc_iso(),
        "config_source": str(cfg_path),
        "config": cfg,
    }
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(out_cfg, f, indent=2)
        f.write("\n")

    notes = []
    notes.append("run_id={}".format(run_id))
    notes.append("ts_utc={}".format(utc_iso()))
    notes.append("harness_version={}".format(HARNESS_VERSION))
    notes.append("config_source={}".format(cfg_path))
    notes.append("python={}".format(sys.version.split()[0]))
    notes.append("platform={} {}".format(os.uname().sysname,
                                         os.uname().release))
    notes.append("workloads={}".format(
        ",".join(w["workload_id"] for w in cfg["workload_matrix"])))
    notes.append("iterations_per_workload={}".format(
        cfg["iterations_per_workload"]))
    notes.append("streaming={}".format(bool(cfg.get("streaming", False))))
    notes.append("backward_compat_synthesis={}".format(
        cfg.get("_backward_compat", False)))
    for w in cfg_warnings:
        notes.append(w)

    # Write the M8c-specific descriptive artifacts up front so they
    # exist even if a later step bails.
    write_match_conditions(cfg, run_dir / "match_conditions.json")
    write_known_mismatches(run_dir / "known_mismatches.txt")
    notes.append("wrote match_conditions.json + known_mismatches.txt")

    # Validate binaries and model exist.
    for label, path in (
        ("hpx_server",   cfg["binaries"]["hpx_server"]),
        ("llama_server", cfg["binaries"]["llama_server"]),
        ("model",        cfg["model_path"]),
    ):
        if not os.path.exists(path):
            notes.append("FATAL: {} not found at {}".format(label, path))
            with open(run_dir / "run_notes.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(notes))
                f.write("\n")
            print("FATAL: {} not found at {}".format(label, path),
                  file=sys.stderr)
            print("gates=FAIL")
            return 2

    # Pre-flight: neither configured port may already be in use. A
    # leftover server from a previous run must be cleaned up before M8c
    # can start. Failing closed here is intentional — silently
    # bypassing the cap would mask the leak.
    for label, port in (
        ("hpx_server",   cfg["ports"]["hpx_server"]),
        ("llama_server", cfg["ports"]["llama_server"]),
    ):
        if port_in_use("127.0.0.1", port):
            msg = ("FATAL: {} port {} already in use; refusing to launch. "
                   "Clean up the leftover process and retry.").format(
                       label, port)
            notes.append(msg)
            with open(run_dir / "run_notes.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(notes))
                f.write("\n")
            print(msg, file=sys.stderr)
            print("gates=FAIL")
            return 2

    all_rows = []

    # hpx-server leg
    notes.append("--- hpx-server leg ---")
    hpx_rows = run_one_server(
        label="hpx_server",
        adapter=hpx_adapter,
        cfg=cfg,
        port=cfg["ports"]["hpx_server"],
        run_dir=run_dir,
        notes=notes,
    )
    all_rows.extend(hpx_rows)

    # llama-server leg
    notes.append("--- llama-server leg ---")
    llama_rows = run_one_server(
        label="llama_server",
        adapter=llama_adapter,
        cfg=cfg,
        port=cfg["ports"]["llama_server"],
        run_dir=run_dir,
        notes=notes,
    )
    all_rows.extend(llama_rows)

    # Write artifacts.
    write_jsonl(all_rows, run_dir / "client_results.jsonl")
    write_summary_csv(all_rows, run_dir / "summary.csv")
    if cfg.get("streaming"):
        write_stream_events_jsonl(
            all_rows, run_dir / "stream_events.jsonl")

    # Evaluate determinism / matching gates.
    gates_passed, gate_notes = evaluate_gates(cfg, hpx_rows, llama_rows)
    notes.append("--- gates ---")
    notes.extend(gate_notes)

    # Factual notes only — no comparative performance language.
    # `run_notes.txt` is excluded from the artifact-existence check below
    # because it is the file being written *now*; an inline check of its
    # own existence would always report exists=False / size=-1.
    notes.append("--- end ---")
    notes.append("artifacts:")
    artifact_names = ["config.json", "match_conditions.json",
                      "known_mismatches.txt",
                      "hpx_server.stdout", "hpx_server.stderr",
                      "hpx_server.args",
                      "llama_server.stdout", "llama_server.stderr",
                      "llama_server.args",
                      "client_results.jsonl", "summary.csv"]
    if cfg.get("streaming"):
        artifact_names.append("stream_events.jsonl")
    for name in artifact_names:
        p = run_dir / name
        notes.append("    {} exists={} size={}".format(
            name, p.exists(),
            p.stat().st_size if p.exists() else -1))

    notes_blob = "\n".join(notes) + "\n"

    # Forbidden-words check: enforce that nothing the harness assembled
    # contains comparative-performance wording. If it does, fail closed
    # so the operator sees the violation rather than the file shipping
    # with the offending text.
    forbidden_hits = check_forbidden_words(notes_blob)
    notes_path = run_dir / "run_notes.txt"
    if forbidden_hits:
        gates_passed = False
        violation = (
            "\nGATE FAIL: run_notes.txt contains forbidden wording: {}\n"
        ).format(", ".join(forbidden_hits))
        with open(notes_path, "w", encoding="utf-8") as f:
            f.write(notes_blob)
            f.write(violation)
        print("FATAL: forbidden wording in run_notes.txt: {}".format(
            ", ".join(forbidden_hits)), file=sys.stderr)
    else:
        with open(notes_path, "w", encoding="utf-8") as f:
            f.write(notes_blob)

    # Driver-side console output: run_id, run_dir, single gates line.
    print("run_id={}".format(run_id))
    print("run_dir={}".format(run_dir))
    print("gates={}".format("PASS" if gates_passed else "FAIL"))
    return 0 if gates_passed else 1


if __name__ == "__main__":
    sys.exit(main())
