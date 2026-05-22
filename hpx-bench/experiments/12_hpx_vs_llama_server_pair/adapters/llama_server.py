"""llama-server adapter for the M8b/M8c pair-run harness.

Pure stdlib. The llama.cpp upstream `/completion` response shape is
documented at:
  https://github.com/ggml-org/llama.cpp/tree/master/tools/server
We do not assert hash/content equality with hpx-server. The harness
records whatever llama-server actually returns; field-name mismatches
are surfaced via `parse_error` and `raw_json`.

M8c additions:
  - request_body / warmup_body pin every sampler knob observed in M8b's
    raw_json, so widening sampling later is an explicit edit rather than
    an implicit drift.
  - parse_response records text_sha256, text_normalized_sha256,
    text_len_chars, and prompt_tokens (from tokens_evaluated).
"""

import hashlib

NAME = "llama_server"


def build_args(cfg, port):
    """Return argv for launching llama-server.

    cfg keys consulted:
        binaries.llama_server, model_path, threads, ctx_size
    """
    return [
        cfg["binaries"]["llama_server"],
        "--model", cfg["model_path"],
        "--host", "127.0.0.1",
        "--port", str(port),
        "--ctx-size", str(cfg["ctx_size"]),
        "--parallel", "1",
        "--threads", str(cfg["threads"]),
        "--no-context-shift",
        "--seed", "0",
    ]


def _pinned_sampler_body():
    """M8d-fix sampler pins. Used by both warmup_body and request_body so
    the warm-up POST exercises the exact body shape measured iterations
    will use. Any sampler-field rejection therefore surfaces at readiness,
    not measurement.

    Note: `ignore_eos` is intentionally omitted (was True in M8c). The
    M8d investigation showed that pinning ignore_eos on llama-server
    while hpx-server has no equivalent created a behavioral asymmetry
    on EOG-sensitive prompts. Both servers now use normal EOG-stop
    semantics; `decode_budget` / `n_predict` is treated as an upper
    bound rather than an exact target.
    """
    return {
        "temperature": 0,
        "top_k": 1,
        "top_p": 1.0,
        "min_p": 1.0,
        "typical_p": 1.0,
        "repeat_penalty": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "dry_multiplier": 0.0,
        "xtc_probability": 0.0,
        "mirostat": 0,
        "n_probs": 0,
        "samplers": ["top_k", "temperature"],
        "stop": [],
        "seed": 0,
        "stream": False,
        "cache_prompt": False,
    }


def warmup_body():
    body = _pinned_sampler_body()
    body["prompt"]    = "hi"
    body["n_predict"] = 1
    return body


def request_body(prompt, decode_budget):
    body = _pinned_sampler_body()
    body["prompt"]    = prompt
    body["n_predict"] = decode_budget
    return body


def _first_present(d, keys):
    for k in keys:
        if k in d:
            return d[k]
    return None


def parse_response(status, headers, raw_body):
    """Normalize a llama-server /completion response.

    llama-server typically returns:
        content              str   generated text
        tokens_predicted     int   tokens generated
        tokens_evaluated     int   prompt tokens
        stop                 bool
        stopped_eos          bool
        stopped_limit        bool
        generation_settings  dict
    Field names may drift across upstream versions; the adapter
    therefore records `raw_json` so the operator can inspect it
    directly. `parse_error` is non-empty when the expected key
    cannot be located.
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
        "prompt_tokens": -1,
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
    if not isinstance(body, dict):
        out["parse_error"] = "response_root_not_object"
        return out

    text = _first_present(body, ["content", "text", "response"])
    if isinstance(text, str):
        out["text"] = text
        out["text_len_chars"]         = len(text)
        out["text_sha256"]            = hashlib.sha256(
            text.encode("utf-8")).hexdigest()
        out["text_normalized_sha256"] = hashlib.sha256(
            text.strip().encode("utf-8")).hexdigest()
    else:
        out["parse_error"] = "no_text_field"

    n_tok = _first_present(body, ["tokens_predicted", "n_decoded",
                                  "completion_tokens"])
    if isinstance(n_tok, int):
        out["n_decoded"] = n_tok

    prompt_tok = _first_present(body, ["tokens_evaluated", "prompt_tokens",
                                       "prompt_n"])
    if isinstance(prompt_tok, int):
        out["prompt_tokens"] = prompt_tok

    if isinstance(body.get("id_slot"), int):
        out["server_request_id"] = body["id_slot"]

    if body.get("stopped_eos"):
        out["server_status"] = "stopped_eos"
    elif body.get("stopped_limit"):
        out["server_status"] = "stopped_limit"
    elif body.get("stop"):
        out["server_status"] = "stop"

    out["ok"] = bool(out["text"])
    return out


# ---------- M8e streaming additions ------------------------------------
#
# llama-server /completion streaming wire format (verified via M8e probe
# 2026-05-21 against /Users/unick/Desktop/hpx/builds/llama-base/bin/llama-server,
# build b8995-523480de2):
#
#   Content-Type: text/event-stream
#
#   data: {"index":0,"content":" John","tokens":[2259],"stop":false,
#          "id_slot":-1,"tokens_predicted":1,"tokens_evaluated":6}
#
#   data: {"index":0,"content":" Smith","tokens":[7075],"stop":false,...}
#   ...
#   data: {"index":0,"content":"","tokens":[],"id_slot":0,"stop":true,
#          "model":"...","tokens_predicted":8,"tokens_evaluated":6,
#          "generation_settings":{...},"prompt":"<s> Hello, my name is",
#          "stop_type":"limit","timings":{...}}
#
# Records are separated by a blank line. Each record carries a single
# `data:` field (no `event:` header). The terminal record has
# `stop: true` and includes the full bookkeeping that the M8d non-
# streaming `raw_json` carried. Per-token records have `stop: false`
# and accumulate `tokens_predicted` from 1..N.


def stream_request_body(prompt, decode_budget):
    """llama-server streaming request body. Same pinned sampler set as
    the M8d-fix non-streaming body, with `stream: true`. `ignore_eos`
    intentionally absent — see _pinned_sampler_body docstring."""
    body = _pinned_sampler_body()
    body["prompt"]    = prompt
    body["n_predict"] = decode_budget
    body["stream"]    = True
    return body


def parse_stream_events(events):
    """Aggregate a list of SSE records into a row-shaped dict.

    `events` is a list of dicts with keys:
        ts_ms_from_request:  int
        event_type:          str   (always \"\" for llama-server records)
        data_json:           dict | None
        data_raw:            str

    Mirrors the non-streaming `parse_response` contract and adds the
    streaming-specific fields (token_event_count, done_seen, etc.).
    Terminal-record fields (`tokens_predicted`, `stop_type`, etc.) are
    surfaced through the same field names used in M8d.
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

    content_parts = []
    terminal      = None
    for ev in events:
        et   = ev["event_type"]
        data = ev["data_json"]
        # llama-server uses `data:` only — no `event:` field.
        if et != "":
            out["parse_error"] = (
                "unexpected_event_type: {!r}".format(et))
            return out
        if not isinstance(data, dict):
            out["parse_error"] = "data_not_object"
            return out
        stop = data.get("stop")
        if stop is True:
            terminal = data
            out["done_seen"] = True
            stop_type = data.get("stop_type")
            if isinstance(stop_type, str):
                out["done_status"] = stop_type
        else:
            content = data.get("content", "")
            if isinstance(content, str):
                content_parts.append(content)
                if content:
                    out["token_event_count"] += 1
                    if out["first_content_event_ts_ms"] == -1:
                        out["first_content_event_ts_ms"] = (
                            ev["ts_ms_from_request"])
            else:
                out["parse_error"] = "content_not_string"
                return out

    if terminal is None:
        out["parse_error"] = "no_terminal_event"
        return out

    out["raw_json"]      = terminal
    out["server_status"] = out["done_status"] or "stop"
    if isinstance(terminal.get("id_slot"), int):
        out["server_request_id"] = terminal["id_slot"]
    n_tok = _first_present(terminal, ["tokens_predicted", "n_decoded",
                                      "completion_tokens"])
    if isinstance(n_tok, int):
        out["n_decoded"] = n_tok
    prompt_tok = _first_present(terminal, ["tokens_evaluated",
                                           "prompt_tokens", "prompt_n"])
    if isinstance(prompt_tok, int):
        out["prompt_tokens"] = prompt_tok

    text = "".join(content_parts)
    out["text"]           = text
    out["text_len_chars"] = len(text)
    out["text_sha256"]    = hashlib.sha256(
        text.encode("utf-8")).hexdigest()
    out["text_normalized_sha256"] = hashlib.sha256(
        text.strip().encode("utf-8")).hexdigest()
    out["ok"] = bool(terminal.get("stop") is True)
    return out

