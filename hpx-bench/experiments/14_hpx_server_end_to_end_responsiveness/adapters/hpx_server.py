"""hpx-server adapter for Experiment 14 Phase 1.

Pure stdlib. Owns:
  - mode_to_extra_args: maps a Phase 1 mode name to the extra CLI args
    that enable that placement.
  - build_args: builds the full argv for launching llama-hpx-server in
    a given mode on a given port.
  - warmup_body / round_trip_body / stream_body: request bodies the
    driver POSTs to /completion.
  - parse_round_trip_response: normalize a W1 response.
  - parse_stream_events: normalize a list of SSE records into the
    streaming metrics the driver records (W2).

Phase 1 deliberately keeps the adapter narrow. It does NOT import the
Exp 12 adapter; experiment directories are self-contained.

The mode contract:

    default_os1       --hpx-os-threads 1
    default_os2       --hpx-os-threads 2
    engine_pool_os2   --hpx-os-threads 2 --engine-pool

The engine-pool path requires --hpx-os-threads >= 2; the hpx-server
preflight (N4) fails closed otherwise. The driver does not retry
preflight failures.
"""

import hashlib


NAME = "hpx_server"

MODES = ("default_os1", "default_os2", "engine_pool_os2")


def mode_to_extra_args(mode):
    """Return the extra CLI args for a Phase 1 mode."""
    if mode == "default_os1":
        return ["--hpx-os-threads", "1"]
    if mode == "default_os2":
        return ["--hpx-os-threads", "2"]
    if mode == "engine_pool_os2":
        return ["--hpx-os-threads", "2", "--engine-pool"]
    raise ValueError("unknown mode: {!r}".format(mode))


def build_args(binary, model_path, port, mode,
               n_seq_max=1, max_prompt_tokens=512,
               n_threads=2, max_concurrent=1, ctx_size=2048):
    """Return argv for launching llama-hpx-server in the given mode.

    Single-client serving config (n_seq_max=1, max_concurrent=1) matches
    the Phase 1 single-client scope; revisit for any concurrent-client
    extension.
    """
    argv = [
        binary,
        "--model",             model_path,
        "--host",              "127.0.0.1",
        "--port",              str(port),
        "--n-seq-max",         str(n_seq_max),
        "--max-prompt-tokens", str(max_prompt_tokens),
        "--n-threads",         str(n_threads),
        "--max-concurrent",    str(max_concurrent),
        "--ctx-size",          str(ctx_size),
    ]
    argv.extend(mode_to_extra_args(mode))
    return argv


def warmup_body():
    """Tiny POST used during readiness probing."""
    return {"prompt": "hi", "decode_budget": 1}


def round_trip_body(prompt, decode_budget):
    """W1 non-streaming body."""
    return {"prompt": prompt, "decode_budget": decode_budget}


def stream_body(prompt, decode_budget):
    """W2/W3 streaming body."""
    return {"prompt": prompt, "decode_budget": decode_budget,
            "stream": True}


def parse_round_trip_response(status, raw_body):
    """Normalize an hpx-server non-streaming /completion response.

    Returns a dict suitable for the W1 JSONL row.
    """
    out = {
        "ok":                False,
        "http_status":       status,
        "text":              "",
        "text_sha256":       "",
        "text_len_chars":    0,
        "n_decoded":         -1,
        "hash":              "",
        "server_request_id": -1,
        "server_status":     "",
        "parse_error":       "",
    }
    if status != 200:
        return out
    try:
        import json
        body = json.loads(raw_body)
    except Exception as e:
        out["parse_error"] = "json_parse_failed: {}".format(e)
        return out
    if isinstance(body, dict):
        if isinstance(body.get("text"), str):
            text = body["text"]
            out["text"]            = text
            out["text_len_chars"]  = len(text)
            out["text_sha256"]     = hashlib.sha256(
                text.encode("utf-8")).hexdigest()
        if isinstance(body.get("n_decoded"), int):
            out["n_decoded"] = body["n_decoded"]
        if isinstance(body.get("hash"), str):
            out["hash"] = body["hash"]
        if isinstance(body.get("request_id"), int):
            out["server_request_id"] = body["request_id"]
        if isinstance(body.get("status"), str):
            out["server_status"] = body["status"]
    out["ok"] = bool(out["text"])
    return out


def parse_stream_events(events):
    """Aggregate a list of SSE records into a W2-shaped dict.

    `events` is a list of dicts with keys:
        ts_ms_from_request:  float
        event_type:          str   ("token", "done", or "")
        data_raw:            str
        data_json:           dict | None

    Returns the streaming row fields the driver writes to JSONL.
    """
    out = {
        "ok":                     False,
        "text":                   "",
        "text_sha256":            "",
        "text_len_chars":         0,
        "n_decoded":              -1,
        "hash":                   "",
        "server_request_id":      -1,
        "server_status":          "",
        "stream_parse_error":     "",
        "token_event_count":      0,
        "done_seen":              False,
        "done_status":            "",
        "first_event_ms":         None,
        "first_token_event_ms":   None,
        "last_event_ms":          None,
        "inter_event_gaps_ms":    [],
    }
    if not events:
        out["stream_parse_error"] = "no_events"
        return out

    out["first_event_ms"] = events[0]["ts_ms_from_request"]
    out["last_event_ms"]  = events[-1]["ts_ms_from_request"]

    prev_ts = None
    gaps = []
    text_parts = []
    done_data = None
    for ev in events:
        ts = ev["ts_ms_from_request"]
        if prev_ts is not None:
            gaps.append(ts - prev_ts)
        prev_ts = ts
        et   = ev["event_type"]
        data = ev["data_json"]
        if et == "token":
            if not isinstance(data, dict):
                out["stream_parse_error"] = "token_event_data_not_object"
                return out
            tok_text = data.get("token", "")
            if not isinstance(tok_text, str):
                out["stream_parse_error"] = "token_event_token_not_string"
                return out
            text_parts.append(tok_text)
            out["token_event_count"] += 1
            if out["first_token_event_ms"] is None:
                out["first_token_event_ms"] = ts
        elif et == "done":
            if not isinstance(data, dict):
                out["stream_parse_error"] = "done_event_data_not_object"
                return out
            done_data = data
            out["done_seen"] = True
        else:
            out["stream_parse_error"] = (
                "unexpected_event_type: {!r}".format(et))
            return out

    out["inter_event_gaps_ms"] = gaps

    if done_data is None:
        out["stream_parse_error"] = "no_done_event"
        return out

    out["done_status"]   = done_data.get("status", "") or ""
    out["server_status"] = out["done_status"]
    if isinstance(done_data.get("request_id"), int):
        out["server_request_id"] = done_data["request_id"]
    if isinstance(done_data.get("n_decoded"), int):
        out["n_decoded"] = done_data["n_decoded"]
    if isinstance(done_data.get("hash"), str):
        out["hash"] = done_data["hash"]

    text = "".join(text_parts)
    out["text"]           = text
    out["text_len_chars"] = len(text)
    out["text_sha256"]    = hashlib.sha256(
        text.encode("utf-8")).hexdigest()
    out["ok"] = (out["done_status"] == "completed")
    return out
