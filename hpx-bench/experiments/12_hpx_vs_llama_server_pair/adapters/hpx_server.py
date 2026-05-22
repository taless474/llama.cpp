"""hpx-server adapter for the M8b/M8c pair-run harness.

Pure stdlib. No HPX coupling. Owns:
  - argv list for launching the binary
  - request body shape
  - response parser that normalizes the JSON into the fields the
    harness writes to client_results.jsonl / summary.csv

The hpx-server response is the canonical greedy path defined in M7a.
Per M6/M7a, the default greedy hash for the canonical prompt is
0x0619d4d1900c2365. The adapter does not assert this; the driver does.

M8c additions:
  - parse_response records text_sha256, text_normalized_sha256, and
    text_len_chars for descriptive cross-iteration comparison.
  - prompt_tokens is recorded as -1 because hpx-server does not expose
    a post-tokenize prompt-token-count field today; the asymmetry with
    llama-server is documented in known_mismatches.txt.
"""

import hashlib

NAME = "hpx_server"


def build_args(cfg, port):
    """Return argv for launching llama-hpx-server.

    cfg keys consulted:
        binaries.hpx_server, model_path, threads, ctx_size
    """
    return [
        cfg["binaries"]["hpx_server"],
        "--model", cfg["model_path"],
        "--host", "127.0.0.1",
        "--port", str(port),
        "--n-seq-max", "1",
        "--max-prompt-tokens", "512",
        "--n-threads", str(cfg["threads"]),
        "--max-concurrent", "1",
        "--ctx-size", str(cfg["ctx_size"]),
    ]


def warmup_body():
    """Smallest valid request used only for readiness probing."""
    return {"prompt": "hi", "decode_budget": 1}


def request_body(prompt, decode_budget):
    return {"prompt": prompt, "decode_budget": decode_budget}


def parse_response(status, headers, raw_body):
    """Normalize an hpx-server /completion response.

    Returns a dict with keys:
        ok                  bool   HTTP 200 + parseable JSON
        http_status         int
        text                str    response text or ""
        n_decoded           int    -1 if absent
        hash                str    "0x..." or ""
        server_request_id   int    -1 if absent
        server_status       str    e.g. "completed", or ""
        raw_json            dict   parsed JSON (or {} on parse failure)
        parse_error         str    "" if JSON parsed cleanly
    """
    out = {
        "ok": False,
        "http_status": status,
        "text": "",
        "text_sha256": "",
        "text_normalized_sha256": "",
        "text_len_chars": 0,
        "n_decoded": -1,
        "hash": "",
        "prompt_tokens": -1,  # hpx-server does not expose this today
        "server_request_id": -1,
        "server_status": "",
        "raw_json": {},
        "parse_error": "",
    }
    if status != 200:
        return out
    try:
        import json
        body = json.loads(raw_body)
    except Exception as e:
        out["parse_error"] = "json_parse_failed: {}".format(e)
        return out
    out["raw_json"] = body
    if isinstance(body, dict):
        if isinstance(body.get("text"), str):
            text = body["text"]
            out["text"]                   = text
            out["text_len_chars"]         = len(text)
            out["text_sha256"]            = hashlib.sha256(
                text.encode("utf-8")).hexdigest()
            out["text_normalized_sha256"] = hashlib.sha256(
                text.strip().encode("utf-8")).hexdigest()
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


# ---------- M8e streaming additions ------------------------------------
#
# hpx-server M7c SSE wire format:
#   event: token
#   data: {"token":"<utf8>","token_id":<int>}
#
#   event: token
#   ...
#
#   event: done
#   data: {"request_id":<int>,"status":"completed|cancelled|failed_reserved",
#          "n_decoded":<int>,"hash":"0x...."}
#
# Records are separated by a blank line. The parser handles CRLF or LF.
# The terminal `event: done` carries the same correctness fingerprint as
# the non-streaming response (n_decoded, hash, status).


def stream_request_body(prompt, decode_budget):
    """hpx-server streaming request body. Adds `stream: true` to the
    canonical greedy POST. No sampler block — hpx-server defaults to
    local argmax."""
    return {
        "prompt":        prompt,
        "decode_budget": decode_budget,
        "stream":        True,
    }


def parse_stream_events(events):
    """Aggregate a list of SSE records into a row-shaped dict.

    `events` is a list of dicts with keys:
        ts_ms_from_request:  int
        event_type:          str   (\"token\", \"done\", or \"\" if no event line)
        data_json:           dict | None
        data_raw:            str

    Returns the same key set as `parse_response`, extended with the
    streaming fields the driver writes to summary.csv:
        token_event_count, done_seen, done_status,
        first_content_event_ts_ms, first_event_ts_ms, last_event_ts_ms

    Mirrors the non-streaming contract (text/hash/n_decoded/etc.) by
    aggregating per-token text and reading the terminal `done` record
    for engine-side correctness fingerprints.
    """
    out = {
        "ok":                     False,
        "http_status":            200,
        "text":                   "",
        "text_sha256":            "",
        "text_normalized_sha256": "",
        "text_len_chars":         0,
        "n_decoded":              -1,
        "hash":                   "",
        "prompt_tokens":          -1,
        "server_request_id":      -1,
        "server_status":          "",
        "raw_json":               {},
        "parse_error":            "",
        # streaming extras
        "token_event_count":      0,
        "done_seen":              False,
        "done_status":            "",
        "first_content_event_ts_ms": -1,
        "first_event_ts_ms":      -1,
        "last_event_ts_ms":       -1,
    }
    if not events:
        out["parse_error"] = "no_events"
        return out

    out["first_event_ts_ms"] = events[0]["ts_ms_from_request"]
    out["last_event_ts_ms"]  = events[-1]["ts_ms_from_request"]

    text_parts = []
    done_data  = None
    for ev in events:
        et   = ev["event_type"]
        data = ev["data_json"]
        if et == "token":
            if not isinstance(data, dict):
                out["parse_error"] = "token_event_data_not_object"
                return out
            tok_text = data.get("token", "")
            if not isinstance(tok_text, str):
                out["parse_error"] = "token_event_token_not_string"
                return out
            text_parts.append(tok_text)
            out["token_event_count"] += 1
            if out["first_content_event_ts_ms"] == -1 and tok_text:
                out["first_content_event_ts_ms"] = ev["ts_ms_from_request"]
        elif et == "done":
            if not isinstance(data, dict):
                out["parse_error"] = "done_event_data_not_object"
                return out
            done_data = data
            out["done_seen"] = True
        else:
            out["parse_error"] = (
                "unexpected_event_type: {!r}".format(et))
            return out

    if done_data is None:
        out["parse_error"] = "no_done_event"
        return out

    out["raw_json"]      = done_data
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
    out["text_normalized_sha256"] = hashlib.sha256(
        text.strip().encode("utf-8")).hexdigest()
    out["ok"] = (out["done_status"] == "completed")
    return out

