import json
from pathlib import Path

ROOT = Path("/Users/Ashk/Desktop/HPX/llama-hpx/local/baselines/server_repeat")

EXPECTED_GEN_SETTINGS = {
    "seed": 1234,
    "temperature": 0.0,
    "top_k": 1,
    "stream": False,
    "n_predict": 16,
}
EXPECTED_PROMPT_TOKENS = 6
EXPECTED_PRED_TOKENS = 16


def load_session(name):
    base = ROOT / name
    headers = (base / "single_request.headers").read_text(errors="replace")
    body = json.loads((base / "single_request.json").read_text())
    shutdown = (base / "server_shutdown.txt").read_text(errors="replace")
    return headers, body, shutdown


def is_clean_shutdown(ps_text):
    lines = [ln for ln in ps_text.splitlines() if ln.strip()]
    if not lines:
        return False
    if "PID" not in lines[0]:
        return False
    return len(lines) == 1


h1, b1, s1 = load_session("session1")
h2, b2, s2 = load_session("session2")

c1 = b1.get("content", "")
c2 = b2.get("content", "")

results = []

results.append(("session1 HTTP 200", "HTTP/1.1 200 OK" in h1))
results.append(("session2 HTTP 200", "HTTP/1.1 200 OK" in h2))

results.append(("session1 JSON parse", True))
results.append(("session2 JSON parse", True))

results.append(("content byte-equal across sessions", c1 == c2))

results.append(("session1 tokens_predicted == 16", b1.get("tokens_predicted") == EXPECTED_PRED_TOKENS))
results.append(("session2 tokens_predicted == 16", b2.get("tokens_predicted") == EXPECTED_PRED_TOKENS))
results.append(("tokens_predicted parity", b1.get("tokens_predicted") == b2.get("tokens_predicted")))

results.append(("session1 tokens_evaluated == 6", b1.get("tokens_evaluated") == EXPECTED_PROMPT_TOKENS))
results.append(("session2 tokens_evaluated == 6", b2.get("tokens_evaluated") == EXPECTED_PROMPT_TOKENS))
results.append(("tokens_evaluated parity", b1.get("tokens_evaluated") == b2.get("tokens_evaluated")))

for label, body in [("session1", b1), ("session2", b2)]:
    gs = body.get("generation_settings", {})
    for key, expected in EXPECTED_GEN_SETTINGS.items():
        actual = gs.get(key)
        results.append((f"{label} generation_settings.{key} == {expected!r}", actual == expected))

results.append(("session1 clean shutdown", is_clean_shutdown(s1)))
results.append(("session2 clean shutdown", is_clean_shutdown(s2)))

print("=== contents ===")
print(f"session1 content: {c1!r}")
print(f"session2 content: {c2!r}")
print()
print("=== token counts ===")
print(f"session1 tokens_predicted={b1.get('tokens_predicted')} tokens_evaluated={b1.get('tokens_evaluated')}")
print(f"session2 tokens_predicted={b2.get('tokens_predicted')} tokens_evaluated={b2.get('tokens_evaluated')}")
print()
print("=== checks ===")
overall = True
for label, ok in results:
    flag = "PASS" if ok else "FAIL"
    if not ok:
        overall = False
    print(f"[{flag}] {label}")
print()
print("OVERALL:", "PASS" if overall else "FAIL")
