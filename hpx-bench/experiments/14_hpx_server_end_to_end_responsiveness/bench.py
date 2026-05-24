#!/usr/bin/env python3
"""Experiment 14 Phase 1 driver — hpx-server end-to-end client-visible
responsiveness across placement modes.

Python 3 stdlib only. Owns the cell loop, the SSE client, the
disconnect harness, the JSONL writer, and the summary aggregator.
Does not modify hpx-server, the engine, or any smoke.

One hpx-server process per (mode, workload, decode_budget) cell. Cells
run strictly sequentially.

Usage:
    python3 bench.py \\
      --binary /Users/unick/Desktop/hpx/builds/llama-hpx-hpx-on/bin/llama-hpx-server \\
      --model  /Users/unick/Desktop/hpx/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \\
      --modes default_os1,default_os2,engine_pool_os2 \\
      --workloads w1,w2,w3 \\
      --trials 30 --warmup-trials 1 \\
      --decode-budgets 8,64 \\
      --w3-decode-budget 128 \\
      --label phase1
"""

import argparse
import csv
import datetime
import hashlib
import http.client
import json
import os
import pathlib
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time

from adapters import hpx_server as hpx_adapter


HARNESS_VERSION = "exp14.phase1.v1"

DEFAULT_BINARY = "/Users/unick/Desktop/hpx/builds/llama-hpx-hpx-on/bin/llama-hpx-server"
DEFAULT_MODEL  = "/Users/unick/Desktop/hpx/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

CANONICAL_PROMPT = "Hello, my name is"
CANONICAL_HASH_B8 = "0x0619d4d1900c2365"

ALL_MODES     = list(hpx_adapter.MODES)
ALL_WORKLOADS = ("w1", "w2", "w3")

W3_ABORT_AFTER_BYTES = 64


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def utc_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def make_run_id(label):
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return "{}-{}".format(ts, label)


def pick_free_port():
    """Bind to (127.0.0.1, 0) to let the OS hand us a free port,
    close immediately, and return the number. There is a small TOCTOU
    window before hpx-server binds; with serial single-machine runs
    this has been reliable across Exp 12/13."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def sha256_of_file(path, max_bytes=5 * 1024 * 1024 * 1024):
    """Streaming SHA256 of `path`. Returns "" if the file is larger
    than `max_bytes` (avoid pegging the disk for a 4 GB model on every
    run; the size + name in the manifest is enough fingerprint)."""
    try:
        st = os.stat(path)
    except OSError:
        return ""
    if st.st_size > max_bytes:
        return ""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
    except OSError:
        return ""
    return h.hexdigest()


def git_info(repo_root):
    """Capture git SHA/branch/dirty for the manifest. Best effort; missing
    git returns blanks."""
    out = {"git_sha": "", "git_branch": "", "git_dirty": False}
    git = shutil.which("git")
    if not git:
        return out
    try:
        out["git_sha"] = subprocess.run(
            [git, "rev-parse", "--short", "HEAD"],
            cwd=repo_root, capture_output=True, text=True,
            timeout=5).stdout.strip()
    except Exception:
        pass
    try:
        out["git_branch"] = subprocess.run(
            [git, "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_root, capture_output=True, text=True,
            timeout=5).stdout.strip()
    except Exception:
        pass
    try:
        status = subprocess.run(
            [git, "status", "--porcelain"],
            cwd=repo_root, capture_output=True, text=True,
            timeout=5).stdout
        out["git_dirty"] = bool(status.strip())
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

def launch_server(argv, env_extra, stdout_path, stderr_path):
    """Start one hpx-server. Returns a Popen with attached file handles
    and label fields the stopper needs."""
    fout = open(stdout_path, "wb")
    ferr = open(stderr_path, "wb")
    env = os.environ.copy()
    env.update(env_extra)
    proc = subprocess.Popen(
        argv,
        stdout=fout,
        stderr=ferr,
        stdin=subprocess.DEVNULL,
        env=env,
        start_new_session=True,
    )
    proc._fout = fout    # type: ignore[attr-defined]
    proc._ferr = ferr    # type: ignore[attr-defined]
    return proc


def stop_server(proc, grace_seconds=15.0):
    """SIGTERM → wait grace → SIGKILL on timeout. Closes stdout/stderr
    file handles. Returns (exit_code, term_method)."""
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
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            method = "sigkill"
            try:
                proc.wait(timeout=grace_seconds)
            except subprocess.TimeoutExpired:
                pass
    for fh_name in ("_fout", "_ferr"):
        fh = getattr(proc, fh_name, None)
        if fh is not None:
            try:
                fh.close()
            except Exception:
                pass
    return proc.returncode, method


def tcp_wait(host, port, interval_s, timeout_s):
    """Poll a TCP connect until success or timeout. Returns elapsed seconds.
    Raises TimeoutError on timeout."""
    deadline = time.monotonic() + timeout_s
    started  = time.monotonic()
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return time.monotonic() - started
        except OSError:
            time.sleep(interval_s)
    raise TimeoutError(
        "tcp_wait({}:{}) timed out after {}s".format(host, port, timeout_s))


# ---------------------------------------------------------------------------
# HTTP client (non-streaming, streaming, streaming+abort)
# ---------------------------------------------------------------------------

def http_post_json(host, port, path, body_dict, timeout_s):
    """One POST with a JSON body. Returns (status, headers, body_str,
    response_complete_ms, request_start_monotonic)."""
    conn = http.client.HTTPConnection(host, port, timeout=timeout_s)
    try:
        payload = json.dumps(body_dict).encode("utf-8")
        conn.putrequest("POST", path)
        conn.putheader("Content-Type", "application/json")
        conn.putheader("Content-Length", str(len(payload)))
        conn.endheaders()
        conn.send(payload)
        t_sent = time.monotonic()
        resp = conn.getresponse()
        status  = resp.status
        headers = dict(resp.getheaders())
        raw     = resp.read().decode("utf-8", errors="replace")
        t_done  = time.monotonic()
        return (status, headers, raw,
                (t_done - t_sent) * 1000.0, t_sent)
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _iter_sse_records(resp, t_request_sent,
                      line_timeout_s, overall_timeout_s):
    """Yield SSE records as (ts_ms_from_request, event_type, data_raw,
    data_json). Minimal SSE parser; same shape as Exp 12."""
    deadline = t_request_sent + overall_timeout_s
    try:
        resp.fp.raw._sock.settimeout(line_timeout_s)  # type: ignore[attr-defined]
    except Exception:
        pass

    event_type = ""
    data_lines = []

    while True:
        if time.monotonic() > deadline:
            raise TimeoutError("sse_overall_timeout")
        try:
            raw = resp.readline()
        except socket.timeout:
            raise TimeoutError("sse_line_timeout")
        except Exception as e:
            raise IOError("sse_read_error: {}".format(e))
        if not raw:
            if data_lines or event_type:
                yield ((time.monotonic() - t_request_sent) * 1000.0,
                       event_type, "\n".join(data_lines), None)
            return
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
            if data_lines or event_type:
                data_raw = "\n".join(data_lines)
                data_json = None
                if data_raw:
                    try:
                        data_json = json.loads(data_raw)
                    except Exception:
                        data_json = None
                ts_ms = (time.monotonic() - t_request_sent) * 1000.0
                yield (ts_ms, event_type, data_raw, data_json)
            event_type = ""
            data_lines = []
            continue
        if decoded.startswith(":"):
            continue
        if decoded.startswith("event:"):
            event_type = decoded[len("event:"):].lstrip()
        elif decoded.startswith("data:"):
            data_lines.append(decoded[len("data:"):].lstrip(" "))


def http_post_streaming(host, port, path, body_dict,
                        line_timeout_s=30.0, overall_timeout_s=120.0):
    """One streaming POST drained to EOF or terminal `done`. Returns
    (events_list, http_status, headers, stream_parse_error,
    response_complete_ms, request_start_monotonic, bytes_received_estimate)."""
    conn = http.client.HTTPConnection(host, port, timeout=line_timeout_s)
    events = []
    parse_error = ""
    bytes_recv_est = 0
    try:
        payload = json.dumps(body_dict).encode("utf-8")
        conn.putrequest("POST", path)
        conn.putheader("Content-Type", "application/json")
        conn.putheader("Content-Length", str(len(payload)))
        conn.endheaders()
        conn.send(payload)
        t_sent = time.monotonic()
        resp = conn.getresponse()
        status  = resp.status
        headers = dict(resp.getheaders())
        if status != 200:
            try:
                body_bytes = resp.read()
            except Exception:
                body_bytes = b""
            parse_error = "non_200_status_{}; body={!r}".format(
                status, body_bytes[:512])
            t_done = time.monotonic()
            return (events, status, headers, parse_error,
                    (t_done - t_sent) * 1000.0, t_sent, len(body_bytes))
        try:
            for ts_ms, evt, data_raw, data_json in _iter_sse_records(
                    resp, t_sent,
                    line_timeout_s=line_timeout_s,
                    overall_timeout_s=overall_timeout_s):
                events.append({
                    "ts_ms_from_request": ts_ms,
                    "event_type":         evt,
                    "data_raw":           data_raw,
                    "data_json":          data_json,
                })
                # bytes_received_est: rough lower bound. We don't have
                # exact wire byte count from the SSE iterator; estimate
                # as data_raw length + "event:<et>\ndata:<dr>\n\n"
                # framing overhead.
                bytes_recv_est += len(data_raw) + len(evt) + 16
        except TimeoutError as e:
            parse_error = "stream_timeout: {}".format(e)
        except Exception as e:
            parse_error = "stream_read_error: {}".format(e)
        t_done = time.monotonic()
        return (events, status, headers, parse_error,
                (t_done - t_sent) * 1000.0, t_sent, bytes_recv_est)
    finally:
        try:
            conn.close()
        except Exception:
            pass


def http_post_streaming_disconnect(host, port, path, body_dict,
                                   abort_after_bytes,
                                   line_timeout_s=30.0,
                                   overall_timeout_s=30.0):
    """Streaming POST that the client aborts after the first cumulative
    >= abort_after_bytes received over the socket. Returns a dict with:
        request_start_monotonic
        first_byte_ms          (recv of any byte)
        first_event_ms         (first complete SSE record terminator)
        disconnect_ms          (socket close completed)
        bytes_received
        events_collected       partial event list
        http_status
        stream_parse_error
    The connection is closed unilaterally — no Connection: close header,
    no clean SSE done. Mirrors the mid-stream client teardown the M7d
    smoke models."""
    conn = http.client.HTTPConnection(host, port, timeout=line_timeout_s)
    events = []
    parse_error = ""
    first_byte_ms = None
    first_event_ms = None
    bytes_received = 0
    status = -1
    t_sent = None
    try:
        payload = json.dumps(body_dict).encode("utf-8")
        conn.putrequest("POST", path)
        conn.putheader("Content-Type", "application/json")
        conn.putheader("Content-Length", str(len(payload)))
        conn.endheaders()
        conn.send(payload)
        t_sent = time.monotonic()
        resp = conn.getresponse()
        status = resp.status
        if status != 200:
            try:
                body_bytes = resp.read()
            except Exception:
                body_bytes = b""
            parse_error = "non_200_status_{}; body={!r}".format(
                status, body_bytes[:512])
            t_close = time.monotonic()
            return {
                "request_start_monotonic": t_sent,
                "first_byte_ms":           None,
                "first_event_ms":          None,
                "disconnect_ms":           (t_close - t_sent) * 1000.0,
                "bytes_received":          len(body_bytes),
                "events_collected":        [],
                "http_status":             status,
                "stream_parse_error":      parse_error,
            }

        # Set a tight socket read timeout so we don't block forever
        # between records.
        try:
            resp.fp.raw._sock.settimeout(line_timeout_s)  # type: ignore[attr-defined]
        except Exception:
            pass
        deadline = t_sent + overall_timeout_s
        event_type = ""
        data_lines = []
        try:
            while True:
                if time.monotonic() > deadline:
                    parse_error = "abort_overall_timeout"
                    break
                raw = resp.readline()
                if not raw:
                    parse_error = "eof_before_abort"
                    break
                if first_byte_ms is None:
                    first_byte_ms = (time.monotonic() - t_sent) * 1000.0
                bytes_received += len(raw)
                # Parse line into SSE accumulator
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
                    if data_lines or event_type:
                        data_raw = "\n".join(data_lines)
                        data_json = None
                        if data_raw:
                            try:
                                data_json = json.loads(data_raw)
                            except Exception:
                                data_json = None
                        ts_ms = (time.monotonic() - t_sent) * 1000.0
                        events.append({
                            "ts_ms_from_request": ts_ms,
                            "event_type":         event_type,
                            "data_raw":           data_raw,
                            "data_json":          data_json,
                        })
                        if first_event_ms is None:
                            first_event_ms = ts_ms
                    event_type = ""
                    data_lines = []
                    # After we've seen at least one complete event AND
                    # received >= abort_after_bytes, abort. The two
                    # gates are AND'd so we always emit at least one
                    # `first_event_ms` reading.
                    if (first_event_ms is not None
                        and bytes_received >= abort_after_bytes):
                        break
                    continue
                if decoded.startswith(":"):
                    continue
                if decoded.startswith("event:"):
                    event_type = decoded[len("event:"):].lstrip()
                elif decoded.startswith("data:"):
                    data_lines.append(decoded[len("data:"):].lstrip(" "))
        except socket.timeout:
            parse_error = "sse_line_timeout_before_abort"
        except Exception as e:
            parse_error = "abort_read_error: {}".format(e)
    finally:
        try:
            conn.close()
        except Exception:
            pass
    t_close = time.monotonic()
    return {
        "request_start_monotonic": t_sent,
        "first_byte_ms":           first_byte_ms,
        "first_event_ms":          first_event_ms,
        "disconnect_ms":           ((t_close - t_sent) * 1000.0
                                    if t_sent is not None else 0.0),
        "bytes_received":          bytes_received,
        "events_collected":        events,
        "http_status":             status,
        "stream_parse_error":      parse_error,
    }


# ---------------------------------------------------------------------------
# Readiness
# ---------------------------------------------------------------------------

def wait_for_ready(host, port, readiness_timeout_s, interval_s=0.1):
    """Two-stage readiness: TCP, then warmup POST. Returns (ok, elapsed_s,
    error_message)."""
    t0 = time.monotonic()
    try:
        tcp_wait(host, port, interval_s=interval_s,
                 timeout_s=readiness_timeout_s)
    except TimeoutError as e:
        return False, time.monotonic() - t0, "tcp_wait_timeout: {}".format(e)

    deadline = t0 + readiness_timeout_s
    last_err = ""
    while time.monotonic() < deadline:
        try:
            status, _, _, _, _ = http_post_json(
                host, port, "/completion",
                hpx_adapter.warmup_body(),
                timeout_s=15.0)
            if status == 200:
                return True, time.monotonic() - t0, ""
            last_err = "warmup_status_{}".format(status)
        except Exception as e:
            last_err = "warmup_exception: {}".format(e)
        time.sleep(interval_s)
    return False, time.monotonic() - t0, last_err or "warmup_timeout"


def wait_for_cap_free(host, port, timeout_s=30.0, interval_s=0.05):
    """After a W3 disconnect, poll the server until a warmup POST
    succeeds (HTTP 200). The HTTP-layer capacity lease (M7e) is held
    by a cpp-httplib stream_state shared_ptr that only drops after
    the engine observes the cancel, finalizes the request, and the
    chunked-provider + resource-releaser lambdas release their
    captures. Until then, any new POST gets HTTP 503 server_busy.
    Returns True if a 200 was observed within the timeout, False
    otherwise.

    Single warmup POST decodes budget=1 (~5-15ms). This intentionally
    runs ONE extra decode between W3 trials; the cost is the price of
    keeping per-trial disconnect_ms measurements free of queue-wait
    contamination."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            status, _, _, _, _ = http_post_json(
                host, port, "/completion",
                hpx_adapter.warmup_body(),
                timeout_s=15.0)
            if status == 200:
                return True
        except Exception:
            pass
        time.sleep(interval_s)
    return False


# ---------------------------------------------------------------------------
# Per-cell execution
# ---------------------------------------------------------------------------

def run_w1_trial(host, port, prompt, decode_budget, timeout_s=120.0):
    """One W1 trial. Returns a row dict."""
    body = hpx_adapter.round_trip_body(prompt, decode_budget)
    row = {
        "request_body":            body,
        "request_start_monotonic": None,
        "http_status":             -1,
        "response_bytes":          0,
        "response_complete_ms":    None,
        "first_byte_ms":           None,
        "first_event_ms":          None,
        "first_token_event_ms":    None,
        "token_event_count":       0,
        "done_seen":               False,
        "done_status":             "",
        "inter_event_gaps_ms":     [],
        "disconnect_ms":           None,
        "bytes_received":          0,
        "n_decoded":               -1,
        "hash":                    "",
        "server_request_id":       -1,
        "server_status":           "",
        "parse_error":             "",
        "stream_parse_error":      "",
        "text_len_chars":          0,
        "text_sha256":             "",
    }
    try:
        status, _, raw, complete_ms, t_sent = http_post_json(
            host, port, "/completion", body, timeout_s=timeout_s)
    except Exception as e:
        row["parse_error"] = "request_exception: {}".format(e)
        return row
    row["request_start_monotonic"] = t_sent
    row["http_status"]             = status
    row["response_bytes"]          = len(raw.encode("utf-8"))
    row["response_complete_ms"]    = complete_ms
    parsed = hpx_adapter.parse_round_trip_response(status, raw)
    row["n_decoded"]         = parsed["n_decoded"]
    row["hash"]              = parsed["hash"]
    row["server_request_id"] = parsed["server_request_id"]
    row["server_status"]     = parsed["server_status"]
    row["parse_error"]       = parsed["parse_error"]
    row["text_len_chars"]    = parsed["text_len_chars"]
    row["text_sha256"]       = parsed["text_sha256"]
    return row


def run_w2_trial(host, port, prompt, decode_budget, timeout_s=180.0):
    """One W2 streaming trial drained to terminal done. Returns a row."""
    body = hpx_adapter.stream_body(prompt, decode_budget)
    row = {
        "request_body":            body,
        "request_start_monotonic": None,
        "http_status":             -1,
        "response_bytes":          0,
        "response_complete_ms":    None,
        "first_byte_ms":           None,
        "first_event_ms":          None,
        "first_token_event_ms":    None,
        "token_event_count":       0,
        "done_seen":               False,
        "done_status":             "",
        "inter_event_gaps_ms":     [],
        "disconnect_ms":           None,
        "bytes_received":          0,
        "n_decoded":               -1,
        "hash":                    "",
        "server_request_id":       -1,
        "server_status":           "",
        "parse_error":             "",
        "stream_parse_error":      "",
        "text_len_chars":          0,
        "text_sha256":             "",
    }
    try:
        (events, status, _, stream_err, complete_ms,
         t_sent, bytes_est) = http_post_streaming(
            host, port, "/completion", body,
            line_timeout_s=30.0,
            overall_timeout_s=timeout_s)
    except Exception as e:
        row["stream_parse_error"] = "stream_exception: {}".format(e)
        return row
    row["request_start_monotonic"] = t_sent
    row["http_status"]             = status
    row["response_bytes"]          = bytes_est
    row["response_complete_ms"]    = complete_ms
    if stream_err:
        row["stream_parse_error"] = stream_err
    parsed = hpx_adapter.parse_stream_events(events)
    row["first_event_ms"]       = parsed["first_event_ms"]
    row["first_token_event_ms"] = parsed["first_token_event_ms"]
    row["token_event_count"]    = parsed["token_event_count"]
    row["done_seen"]            = parsed["done_seen"]
    row["done_status"]          = parsed["done_status"]
    row["inter_event_gaps_ms"]  = parsed["inter_event_gaps_ms"]
    row["n_decoded"]            = parsed["n_decoded"]
    row["hash"]                 = parsed["hash"]
    row["server_request_id"]    = parsed["server_request_id"]
    row["server_status"]        = parsed["server_status"]
    row["text_len_chars"]       = parsed["text_len_chars"]
    row["text_sha256"]          = parsed["text_sha256"]
    if parsed["stream_parse_error"] and not row["stream_parse_error"]:
        row["stream_parse_error"] = parsed["stream_parse_error"]
    return row


def run_w3_trial(host, port, prompt, decode_budget, timeout_s=30.0):
    """One W3 disconnect trial. Returns a row."""
    body = hpx_adapter.stream_body(prompt, decode_budget)
    row = {
        "request_body":            body,
        "request_start_monotonic": None,
        "http_status":             -1,
        "response_bytes":          0,
        "response_complete_ms":    None,
        "first_byte_ms":           None,
        "first_event_ms":          None,
        "first_token_event_ms":    None,
        "token_event_count":       0,
        "done_seen":               False,
        "done_status":             "",
        "inter_event_gaps_ms":     [],
        "disconnect_ms":           None,
        "bytes_received":          0,
        "n_decoded":               -1,
        "hash":                    "",
        "server_request_id":       -1,
        "server_status":           "",
        "parse_error":             "",
        "stream_parse_error":      "",
        "text_len_chars":          0,
        "text_sha256":             "",
    }
    try:
        result = http_post_streaming_disconnect(
            host, port, "/completion", body,
            abort_after_bytes=W3_ABORT_AFTER_BYTES,
            line_timeout_s=30.0,
            overall_timeout_s=timeout_s)
    except Exception as e:
        row["stream_parse_error"] = "abort_exception: {}".format(e)
        return row
    row["request_start_monotonic"] = result["request_start_monotonic"]
    row["http_status"]             = result["http_status"]
    row["response_bytes"]          = result["bytes_received"]
    row["first_byte_ms"]           = result["first_byte_ms"]
    row["first_event_ms"]          = result["first_event_ms"]
    row["disconnect_ms"]           = result["disconnect_ms"]
    row["bytes_received"]          = result["bytes_received"]
    if result["stream_parse_error"]:
        row["stream_parse_error"]  = result["stream_parse_error"]
    return row


def run_cell(cfg, mode, workload, decode_budget,
             cell_index, total_cells,
             run_dir, manifest_cells, jsonl_paths_seen):
    """Run one (mode, workload, decode_budget) cell. Returns the
    manifest entry."""
    host = "127.0.0.1"
    port = pick_free_port()
    argv = hpx_adapter.build_args(
        binary=cfg["binary"], model_path=cfg["model"], port=port,
        mode=mode,
        n_seq_max=cfg["n_seq_max"],
        max_prompt_tokens=cfg["max_prompt_tokens"],
        n_threads=cfg["n_threads"],
        max_concurrent=cfg["max_concurrent"],
        ctx_size=cfg["ctx_size"])
    env_extra = {}
    if (cfg["placement_trace_engine_pool"]
        and mode == "engine_pool_os2"):
        env_extra["LLAMA_HPX_PLACEMENT_TRACE"] = "1"

    cell_tag = "{}_{}_b{}".format(mode, workload, decode_budget)
    stdout_path = run_dir / "logs" / (cell_tag + ".stdout")
    stderr_path = run_dir / "logs" / (cell_tag + ".stderr")
    args_path   = run_dir / "logs" / (cell_tag + ".args")
    with open(args_path, "w", encoding="utf-8") as f:
        for a in argv:
            f.write(a + "\n")

    jsonl_path = run_dir / "raw" / (cell_tag + ".jsonl")
    cmd_line = " ".join(["env"] + [
        "{}={}".format(k, v) for k, v in env_extra.items()
    ] + argv)

    with open(run_dir / "commands.txt", "a", encoding="utf-8") as f:
        f.write("# cell {}/{}: mode={} workload={} budget={}\n".format(
            cell_index + 1, total_cells, mode, workload, decode_budget))
        f.write(cmd_line + "\n\n")

    entry = {
        "mode":               mode,
        "workload":           workload,
        "decode_budget":      decode_budget,
        "port":               port,
        "argv":               argv,
        "env_extra":          env_extra,
        "started_utc":        utc_iso(),
        "ended_utc":          "",
        "readiness_ok":       False,
        "readiness_error":    "",
        "trials_attempted":   0,
        "trials_ok":          0,
        "trials_hash_ok":     0,
        "trials_done_seen":   0,
        "w3_sanity_ok":       None,
        "w3_sanity_hash_ok":  None,
        "exit_code":          None,
        "term_method":        "",
        "placement_trace_seen": False,
    }

    print("[cell {}/{}] launching {} workload={} budget={} on port {} ..."
          .format(cell_index + 1, total_cells, mode, workload,
                  decode_budget, port),
          flush=True)

    proc = launch_server(argv, env_extra, stdout_path, stderr_path)

    try:
        ok, elapsed_s, err = wait_for_ready(
            host, port,
            readiness_timeout_s=cfg["readiness_timeout_s"])
        entry["readiness_ok"]    = ok
        entry["readiness_error"] = err
        if not ok:
            print("[cell {}/{}]   readiness FAILED: {}"
                  .format(cell_index + 1, total_cells, err), flush=True)
            return entry
        print("[cell {}/{}]   ready in {:.2f}s; running {} warmup + {} trials"
              .format(cell_index + 1, total_cells, elapsed_s,
                      cfg["warmup_trials"], cfg["trials"]), flush=True)

        with open(jsonl_path, "w", encoding="utf-8") as jf:
            jsonl_paths_seen.add(str(jsonl_path))
            total_iter = cfg["warmup_trials"] + cfg["trials"]
            for i in range(total_iter):
                is_warmup = i < cfg["warmup_trials"]
                if workload == "w1":
                    row = run_w1_trial(host, port,
                                       cfg["prompt"], decode_budget)
                elif workload == "w2":
                    row = run_w2_trial(host, port,
                                       cfg["prompt"], decode_budget)
                elif workload == "w3":
                    # W3 only: between iterations, sleep for a fixed
                    # configurable delay instead of probing with a
                    # warmup POST. The warmup probe was a budget=1
                    # natural-completion request, which exposed a
                    # separate admission issue (cancelled -> completed
                    # -> later request queued-not-admitted) and made
                    # the multi-disconnect cell hang. Phase 1 measures
                    # client-visible disconnect behavior only, so we
                    # bypass that probe; the admission bug is left for
                    # a future dedicated smoke. Skipped on iteration 0
                    # because cell readiness already proved capacity
                    # was free.
                    if i > 0 and cfg["w3_post_disconnect_sleep_s"] > 0:
                        time.sleep(cfg["w3_post_disconnect_sleep_s"])
                    row = run_w3_trial(host, port,
                                       cfg["prompt"], decode_budget)
                else:
                    raise ValueError("unknown workload " + repr(workload))
                full = {
                    "mode":              mode,
                    "workload":          workload,
                    "decode_budget":     decode_budget,
                    "iteration":         i,
                    "is_warmup":         is_warmup,
                    "ts_utc":            utc_iso(),
                }
                full.update(row)
                jf.write(json.dumps(full) + "\n")
                if not is_warmup:
                    entry["trials_attempted"] += 1
                    # PASS counting
                    if workload == "w1":
                        if (row["http_status"] == 200
                            and not row["parse_error"]):
                            entry["trials_ok"] += 1
                            if (decode_budget == 8
                                and row["hash"] == CANONICAL_HASH_B8):
                                entry["trials_hash_ok"] += 1
                    elif workload == "w2":
                        if (row["done_seen"]
                            and row["done_status"] == "completed"):
                            entry["trials_done_seen"] += 1
                            entry["trials_ok"] += 1
                            if (decode_budget == 8
                                and row["hash"] == CANONICAL_HASH_B8):
                                entry["trials_hash_ok"] += 1
                    elif workload == "w3":
                        # W3 is OK if we tore down the socket and got
                        # at least one event back (or saw http_status
                        # == 200 with a clean break).
                        if (row["disconnect_ms"] is not None
                            and row["http_status"] == 200):
                            entry["trials_ok"] += 1

            # W3 sanity request (one per W3 cell, runs after trial loop)
            if workload == "w3":
                # Sleep before the sanity POST instead of probing with
                # wait_for_cap_free; the probe would itself be a
                # natural-completion request and could expose the
                # cancelled -> completed admission issue. See known
                # issue note in facts.md / results.md.
                if cfg["w3_post_disconnect_sleep_s"] > 0:
                    time.sleep(cfg["w3_post_disconnect_sleep_s"])
                try:
                    status, _, raw, complete_ms, t_sent = http_post_json(
                        host, port, "/completion",
                        hpx_adapter.round_trip_body(
                            cfg["prompt"], 8),
                        timeout_s=60.0)
                    parsed = hpx_adapter.parse_round_trip_response(
                        status, raw)
                    entry["w3_sanity_ok"] = (
                        status == 200 and not parsed["parse_error"])
                    entry["w3_sanity_hash_ok"] = (
                        parsed["hash"] == CANONICAL_HASH_B8)
                    sanity_row = {
                        "mode":                    mode,
                        "workload":                "w3_sanity",
                        "decode_budget":           8,
                        "iteration":               0,
                        "is_warmup":               False,
                        "ts_utc":                  utc_iso(),
                        "request_body":            hpx_adapter.round_trip_body(
                            cfg["prompt"], 8),
                        "request_start_monotonic": t_sent,
                        "http_status":             status,
                        "response_bytes":          len(
                            raw.encode("utf-8")),
                        "response_complete_ms":    complete_ms,
                        "first_byte_ms":           None,
                        "first_event_ms":          None,
                        "first_token_event_ms":    None,
                        "token_event_count":       0,
                        "done_seen":               False,
                        "done_status":             "",
                        "inter_event_gaps_ms":     [],
                        "disconnect_ms":           None,
                        "bytes_received":          0,
                        "n_decoded":               parsed["n_decoded"],
                        "hash":                    parsed["hash"],
                        "server_request_id":
                            parsed["server_request_id"],
                        "server_status":           parsed["server_status"],
                        "parse_error":             parsed["parse_error"],
                        "stream_parse_error":      "",
                        "text_len_chars":          parsed["text_len_chars"],
                        "text_sha256":             parsed["text_sha256"],
                    }
                    jf.write(json.dumps(sanity_row) + "\n")
                except Exception as e:
                    entry["w3_sanity_ok"] = False
                    entry["w3_sanity_hash_ok"] = False
                    entry["readiness_error"] = (
                        entry["readiness_error"]
                        + " | w3_sanity_exception: {}".format(e))

    finally:
        exit_code, term_method = stop_server(proc, grace_seconds=15.0)
        entry["exit_code"]   = exit_code
        entry["term_method"] = term_method
        entry["ended_utc"]   = utc_iso()
        # Check for placement-trace evidence in stderr if engine_pool
        # mode was traced.
        if env_extra.get("LLAMA_HPX_PLACEMENT_TRACE") == "1":
            try:
                with open(stderr_path, "r",
                          encoding="utf-8", errors="replace") as f:
                    txt = f.read()
                entry["placement_trace_seen"] = (
                    "engine_task_placement pool=engine" in txt)
            except Exception:
                pass

    print("[cell {}/{}]   done. trials_ok={}/{} term={}"
          .format(cell_index + 1, total_cells,
                  entry["trials_ok"], entry["trials_attempted"],
                  entry["term_method"]),
          flush=True)
    return entry


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def percentile(sorted_values, p):
    """Linear-interpolated percentile on a pre-sorted list."""
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return (sorted_values[f] * (c - k)
            + sorted_values[c] * (k - f))


def aggregate_metric(values):
    if not values:
        return None
    s = sorted(values)
    return {
        "n":       len(s),
        "min_ms":  s[0],
        "p50_ms":  percentile(s, 50),
        "p95_ms":  percentile(s, 95),
        "p99_ms":  percentile(s, 99),
        "max_ms":  s[-1],
        "mean_ms": statistics.fmean(s),
    }


W1_METRICS = ("response_complete_ms",)
W2_METRICS = ("first_event_ms", "first_token_event_ms",
              "response_complete_ms", "inter_event_gap_ms")
W3_METRICS = ("first_event_ms", "disconnect_ms", "bytes_received")


def metric_value(row, metric):
    """Extract a metric value (or list of values for inter_event_gap)
    from a row. Returns None or list[float]."""
    if metric == "inter_event_gap_ms":
        gaps = row.get("inter_event_gaps_ms") or []
        return [g for g in gaps if g is not None]
    if metric == "bytes_received":
        v = row.get("bytes_received")
        return float(v) if v is not None else None
    v = row.get(metric)
    if v is None:
        return None
    return float(v)


def metrics_for(workload):
    if workload == "w1":
        return W1_METRICS
    if workload == "w2":
        return W2_METRICS
    if workload == "w3":
        return W3_METRICS
    return ()


def write_summary_csv(run_dir, jsonl_paths):
    """Read every raw JSONL file and emit summary.csv with one row per
    (mode, workload, decode_budget, metric)."""
    summary_rows = []
    for p in sorted(jsonl_paths):
        rows = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        if not rows:
            continue
        # Skip w3_sanity records when aggregating; they belong to the
        # manifest counters, not the latency tables.
        meta = rows[0]
        workload = meta["workload"]
        if workload == "w3_sanity":
            continue
        for metric in metrics_for(workload):
            values = []
            for r in rows:
                if r.get("is_warmup"):
                    continue
                if r.get("workload") == "w3_sanity":
                    continue
                # W1/W2 latency rows excluded if request did not complete
                if workload in ("w1", "w2"):
                    if r.get("http_status") != 200:
                        continue
                    if (workload == "w2"
                        and not r.get("done_seen", False)):
                        continue
                v = metric_value(r, metric)
                if v is None:
                    continue
                if isinstance(v, list):
                    values.extend(v)
                else:
                    values.append(v)
            agg = aggregate_metric(values)
            if agg is None:
                summary_rows.append({
                    "mode":          meta["mode"],
                    "workload":      meta["workload"],
                    "decode_budget": meta["decode_budget"],
                    "metric":        metric,
                    "n":             0,
                    "min_ms":        "",
                    "p50_ms":        "",
                    "p95_ms":        "",
                    "p99_ms":        "",
                    "max_ms":        "",
                    "mean_ms":       "",
                })
            else:
                summary_rows.append({
                    "mode":          meta["mode"],
                    "workload":      meta["workload"],
                    "decode_budget": meta["decode_budget"],
                    "metric":        metric,
                    "n":             agg["n"],
                    "min_ms":        agg["min_ms"],
                    "p50_ms":        agg["p50_ms"],
                    "p95_ms":        agg["p95_ms"],
                    "p99_ms":        agg["p99_ms"],
                    "max_ms":        agg["max_ms"],
                    "mean_ms":       agg["mean_ms"],
                })

    out_path = run_dir / "summary.csv"
    fieldnames = ["mode", "workload", "decode_budget", "metric",
                  "n", "min_ms", "p50_ms", "p95_ms", "p99_ms",
                  "max_ms", "mean_ms"]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_csv_list(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_int_csv_list(s):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main(argv):
    p = argparse.ArgumentParser(
        description="Experiment 14 Phase 1 driver (hpx-server placement, "
                    "end-to-end client-visible).")
    p.add_argument("--binary", default=DEFAULT_BINARY)
    p.add_argument("--model",  default=DEFAULT_MODEL)
    p.add_argument("--modes",
                   default=",".join(ALL_MODES),
                   help="comma-separated subset of {}".format(
                       ",".join(ALL_MODES)))
    p.add_argument("--workloads",
                   default=",".join(ALL_WORKLOADS),
                   help="comma-separated subset of {}".format(
                       ",".join(ALL_WORKLOADS)))
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--warmup-trials", type=int, default=1)
    p.add_argument("--decode-budgets",
                   default="8,64",
                   help="comma-separated budgets for W1/W2")
    p.add_argument("--w3-decode-budget", type=int, default=128)
    p.add_argument("--w3-post-disconnect-sleep", type=float, default=2.0,
                   help="seconds to sleep between W3 disconnect trials "
                        "(and before the W3 post-cell sanity POST). "
                        "Replaces the older wait_for_cap_free probe; "
                        "see facts.md known-issue note.")
    p.add_argument("--prompt", default=CANONICAL_PROMPT)
    p.add_argument("--n-seq-max", type=int, default=1)
    p.add_argument("--max-prompt-tokens", type=int, default=512)
    p.add_argument("--n-threads", type=int, default=2,
                   help="libllama compute threads (separate from "
                        "--hpx-os-threads)")
    p.add_argument("--max-concurrent", type=int, default=1)
    p.add_argument("--ctx-size", type=int, default=2048)
    p.add_argument("--readiness-timeout-seconds", type=float, default=30.0)
    p.add_argument("--placement-trace-engine-pool", action="store_true",
                   help="set LLAMA_HPX_PLACEMENT_TRACE=1 only for "
                        "engine_pool_os2 cells; use for validation runs only.")
    p.add_argument("--label", default="run")
    p.add_argument("--results-root",
                   default=str(pathlib.Path(__file__).parent / "results"))
    args = p.parse_args(argv)

    modes = parse_csv_list(args.modes)
    for m in modes:
        if m not in ALL_MODES:
            print("error: unknown mode {!r}".format(m), file=sys.stderr)
            return 2
    workloads = parse_csv_list(args.workloads)
    for w in workloads:
        if w not in ALL_WORKLOADS:
            print("error: unknown workload {!r}".format(w),
                  file=sys.stderr)
            return 2
    budgets = parse_int_csv_list(args.decode_budgets)
    for b in budgets:
        if b < 1:
            print("error: decode_budgets must be >= 1", file=sys.stderr)
            return 2
    if args.w3_decode_budget < 1:
        print("error: --w3-decode-budget must be >= 1", file=sys.stderr)
        return 2

    binary = os.path.abspath(args.binary)
    model  = os.path.abspath(args.model)
    if not os.path.isfile(binary):
        print("error: binary not found: {}".format(binary),
              file=sys.stderr)
        return 1
    if not os.path.isfile(model):
        print("error: model not found: {}".format(model),
              file=sys.stderr)
        return 1

    repo_root = pathlib.Path(__file__).resolve().parents[3]
    run_id = make_run_id(args.label)
    run_dir = pathlib.Path(args.results_root) / run_id
    (run_dir / "raw").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    with open(run_dir / "commands.txt", "w", encoding="utf-8") as f:
        f.write("# harness_version={}\n".format(HARNESS_VERSION))
        f.write("# w3_post_disconnect_sleep_s={}\n\n".format(
            float(args.w3_post_disconnect_sleep)))

    cfg = {
        "binary":             binary,
        "model":              model,
        "prompt":             args.prompt,
        "trials":             args.trials,
        "warmup_trials":      args.warmup_trials,
        "n_seq_max":          args.n_seq_max,
        "max_prompt_tokens":  args.max_prompt_tokens,
        "n_threads":          args.n_threads,
        "max_concurrent":     args.max_concurrent,
        "ctx_size":           args.ctx_size,
        "readiness_timeout_s": args.readiness_timeout_seconds,
        "placement_trace_engine_pool":
            bool(args.placement_trace_engine_pool),
        "w3_post_disconnect_sleep_s":
            float(args.w3_post_disconnect_sleep),
    }

    # Build cell list: 3 modes x (budgets for W1) + (budgets for W2)
    #                          + (one budget for W3)
    cells = []
    for mode in modes:
        for w in workloads:
            if w == "w3":
                cells.append((mode, w, args.w3_decode_budget))
            else:
                for b in budgets:
                    cells.append((mode, w, b))

    git = git_info(str(repo_root))
    manifest = {
        "harness_version":    HARNESS_VERSION,
        "run_id":             run_id,
        "label":              args.label,
        "started_utc":        utc_iso(),
        "ended_utc":          "",
        "git_sha":            git["git_sha"],
        "git_branch":         git["git_branch"],
        "git_dirty":          git["git_dirty"],
        "machine": {
            "platform": sys.platform,
            "uname":    list(os.uname()),
            "python":   sys.version,
        },
        "binary":             binary,
        "binary_sha256":      sha256_of_file(binary),
        "model":              model,
        "model_sha256":       sha256_of_file(model),
        "modes":              modes,
        "workloads":          workloads,
        "decode_budgets":     budgets,
        "w3_decode_budget":   args.w3_decode_budget,
        "w3_post_disconnect_sleep_s":
            float(args.w3_post_disconnect_sleep),
        "trials":             args.trials,
        "warmup_trials":      args.warmup_trials,
        "placement_trace_engine_pool":
            bool(args.placement_trace_engine_pool),
        "cells":              [],
    }

    print("[run] run_id={} binary={} model={}"
          .format(run_id, binary, model), flush=True)
    print("[run] modes={} workloads={} budgets={} w3_budget={} "
          "trials={} warmup={} cells={}"
          .format(modes, workloads, budgets, args.w3_decode_budget,
                  args.trials, args.warmup_trials, len(cells)),
          flush=True)
    print("[run] results dir: {}".format(run_dir), flush=True)

    jsonl_paths_seen = set()
    for i, (mode, w, b) in enumerate(cells):
        entry = run_cell(cfg, mode, w, b,
                         cell_index=i, total_cells=len(cells),
                         run_dir=run_dir,
                         manifest_cells=manifest["cells"],
                         jsonl_paths_seen=jsonl_paths_seen)
        manifest["cells"].append(entry)
        # Write manifest after every cell so a Ctrl-C keeps partial
        # progress queryable.
        manifest["ended_utc"] = utc_iso()
        with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    write_summary_csv(run_dir, jsonl_paths_seen)

    # Print a one-line PASS/FAIL gate to stdout.
    ok = True
    failures = []
    for cell in manifest["cells"]:
        if not cell["readiness_ok"]:
            ok = False
            failures.append("{}_{}_b{}:readiness".format(
                cell["mode"], cell["workload"], cell["decode_budget"]))
        if cell["workload"] in ("w1", "w2"):
            if cell["trials_ok"] != cell["trials_attempted"]:
                ok = False
                failures.append("{}_{}_b{}:trials_ok={}/{}".format(
                    cell["mode"], cell["workload"], cell["decode_budget"],
                    cell["trials_ok"], cell["trials_attempted"]))
            if cell["decode_budget"] == 8:
                if cell["trials_hash_ok"] != cell["trials_attempted"]:
                    ok = False
                    failures.append("{}_{}_b{}:hash_ok={}/{}".format(
                        cell["mode"], cell["workload"], cell["decode_budget"],
                        cell["trials_hash_ok"],
                        cell["trials_attempted"]))
        if cell["workload"] == "w3":
            if cell["trials_ok"] != cell["trials_attempted"]:
                ok = False
                failures.append("{}_{}_b{}:trials_ok={}/{}".format(
                    cell["mode"], cell["workload"], cell["decode_budget"],
                    cell["trials_ok"], cell["trials_attempted"]))
            if not cell["w3_sanity_ok"]:
                ok = False
                failures.append("{}_{}_b{}:w3_sanity_ok=false".format(
                    cell["mode"], cell["workload"], cell["decode_budget"]))
            if not cell["w3_sanity_hash_ok"]:
                ok = False
                failures.append(
                    "{}_{}_b{}:w3_sanity_hash_ok=false".format(
                        cell["mode"], cell["workload"],
                        cell["decode_budget"]))

    if ok:
        print("EXP14_PHASE1: PASS run_id={}".format(run_id))
    else:
        print("EXP14_PHASE1: FAIL run_id={} failures={}".format(
            run_id, ",".join(failures)))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
