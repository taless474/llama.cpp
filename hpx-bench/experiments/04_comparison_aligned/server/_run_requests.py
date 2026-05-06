"""
Aligned llama-server timing helper for the std-vs-server comparison.

Copy of local/baselines/server_timing/_run_requests.py with two changes:
  - PAYLOAD adds "return_tokens": true so each response carries the predicted
    token-ID array.
  - per_repeat.csv adds token_id_count and token_id_hash columns. The hash is
    a FNV-1a 64-bit fold over response.tokens that bit-for-bit mirrors the
    fold in tools/serving-bench/harness.h (constants k_token_hash_init and
    0x100000001b3, k_token_hash_empty = 0 for empty token lists). It is what
    the cross-binary summarizer compares against the bench's
    generated_token_hash.

Inputs:
  - server already running on http://127.0.0.1:18083 with -t 4
Outputs (under local/baselines/comparison_aligned/server/):
  - request_warmup.headers, request_warmup.json, request_warmup.curl_meta.txt
  - request_measured_{1..5}.headers, request_measured_{1..5}.json,
    request_measured_{1..5}.curl_meta.txt
  - per_repeat.csv (header + 6 data rows: warmup + 5 measured)
  - request_summary.txt
"""
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path("/Users/Ashk/Desktop/HPX/llama-hpx/local/baselines/comparison_aligned/server")
URL = "http://127.0.0.1:18083/completion"

PAYLOAD = {
    "prompt": "Hello, my name is",
    "n_predict": 16,
    "temperature": 0.0,
    "top_k": 1,
    "seed": 1234,
    "cache_prompt": False,
    "stream": False,
    "return_tokens": True,
}

REQUESTS = [("warmup", "warmup")] + [(f"measured_{i}", f"measured {i}") for i in range(1, 6)]

CSV_HEADER = (
    "label,kind,http_status,wall_ms,client_tokens_per_sec,"
    "tokens_predicted,tokens_evaluated,prompt_ms,predicted_ms,"
    "prompt_per_second,predicted_per_second,content_len_chars,"
    "token_id_count,token_id_hash"
)

# FNV-1a 64-bit fold over generated token IDs.
# Mirrors tools/serving-bench/harness.h:23-30 exactly:
#   state = 0xcbf29ce484222325 (offset basis)
#   for each token id: state ^= uint32(token_id); state *= 0x100000001b3 (mod 2^64)
# Empty token list returns 0 (k_token_hash_empty), matching backend_std.cpp:280.
FNV1A_OFFSET = 0xcbf29ce484222325
FNV1A_PRIME  = 0x100000001b3
MASK64       = 0xFFFFFFFFFFFFFFFF


def fnv1a_fold_token_ids(token_ids):
    if not token_ids:
        return 0
    state = FNV1A_OFFSET
    for tid in token_ids:
        tid_u32 = int(tid) & 0xFFFFFFFF
        state = (state ^ tid_u32) & MASK64
        state = (state * FNV1A_PRIME) & MASK64
    return state


def post_once(label):
    body = json.dumps(PAYLOAD).encode("utf-8")
    req = urllib.request.Request(
        URL,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            status = resp.status
            headers_text = f"HTTP/1.1 {status} {resp.reason}\n" + "".join(
                f"{k}: {v}\n" for k, v in resp.getheaders()
            )
            raw = resp.read()
        t1 = time.monotonic()
    except Exception as e:
        t1 = time.monotonic()
        wall_ms = (t1 - t0) * 1000.0
        (ROOT / f"request_{label}.curl_meta.txt").write_text(
            f"label={label}\nerror={e!r}\nwall_ms={wall_ms:.3f}\n"
        )
        raise

    wall_ms = (t1 - t0) * 1000.0

    (ROOT / f"request_{label}.headers").write_text(headers_text)
    (ROOT / f"request_{label}.json").write_bytes(raw)

    body_obj = json.loads(raw.decode("utf-8"))
    timings = body_obj.get("timings", {}) or {}
    tokens_predicted = body_obj.get("tokens_predicted")
    tokens_evaluated = body_obj.get("tokens_evaluated")
    content = body_obj.get("content", "") or ""

    token_ids = body_obj.get("tokens", []) or []
    if not isinstance(token_ids, list):
        token_ids = []
    token_id_count = len(token_ids)
    token_id_hash  = fnv1a_fold_token_ids(token_ids)

    if isinstance(tokens_predicted, int) and tokens_predicted > 0 and wall_ms > 0:
        client_tps = tokens_predicted / (wall_ms / 1000.0)
    else:
        client_tps = 0.0

    (ROOT / f"request_{label}.curl_meta.txt").write_text(
        f"label={label}\nstatus={status}\nwall_ms={wall_ms:.3f}\n"
        f"tokens_predicted={tokens_predicted}\ntokens_evaluated={tokens_evaluated}\n"
        f"token_id_count={token_id_count}\ntoken_id_hash=0x{token_id_hash:016x}\n"
    )

    return {
        "label": label,
        "kind": "warmup" if label == "warmup" else "measured",
        "status": status,
        "wall_ms": wall_ms,
        "client_tps": client_tps,
        "tokens_predicted": tokens_predicted,
        "tokens_evaluated": tokens_evaluated,
        "prompt_ms": timings.get("prompt_ms"),
        "predicted_ms": timings.get("predicted_ms"),
        "prompt_per_second": timings.get("prompt_per_second"),
        "predicted_per_second": timings.get("predicted_per_second"),
        "content": content,
        "token_id_count": token_id_count,
        "token_id_hash": token_id_hash,
    }


def fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


rows = []
summary_lines = []
for label, _desc in REQUESTS:
    r = post_once(label)
    rows.append(r)
    summary_lines.append(
        f"{r['label']:<12} status={r['status']} wall_ms={r['wall_ms']:.3f} "
        f"tp={r['tokens_predicted']} te={r['tokens_evaluated']} "
        f"client_tps={r['client_tps']:.3f} "
        f"token_id_count={r['token_id_count']} "
        f"token_id_hash=0x{r['token_id_hash']:016x}"
    )

csv_path = ROOT / "per_repeat.csv"
with csv_path.open("w") as f:
    f.write(CSV_HEADER + "\n")
    for r in rows:
        f.write(
            ",".join(
                [
                    r["label"],
                    r["kind"],
                    fmt(r["status"]),
                    fmt(r["wall_ms"]),
                    fmt(r["client_tps"]),
                    fmt(r["tokens_predicted"]),
                    fmt(r["tokens_evaluated"]),
                    fmt(r["prompt_ms"]),
                    fmt(r["predicted_ms"]),
                    fmt(r["prompt_per_second"]),
                    fmt(r["predicted_per_second"]),
                    fmt(len(r["content"])),
                    fmt(r["token_id_count"]),
                    f"0x{r['token_id_hash']:016x}",
                ]
            )
            + "\n"
        )

(ROOT / "request_summary.txt").write_text("\n".join(summary_lines) + "\n")

print("=== request_summary ===")
print("\n".join(summary_lines))
print()
print(f"per_repeat.csv: {csv_path}")
