# HPX serving experiments — index

Current serving-layer responsiveness experiments:

- **`13_control_plane_responsiveness/`** — internal (HPX-owned)
  control-plane responsiveness. Engine-task placement effect measured
  from timestamps stamped *inside* the engine task (W2 queued-cancel,
  W3 multi-stream). Latency-of-orchestration only; no throughput or
  token-hash claim.

- **`14_hpx_server_end_to_end_responsiveness/`** — end-to-end
  client-visible responsiveness of `hpx-server` through the HTTP/SSE
  adapter (W1 round-trip, W2 stream-full, W3 stream-disconnect). The
  client-facing counterpart to Exp 13. Phase 1 makes no llama-server
  comparison.

Each experiment directory has its own `readme.md` (design + how to
run), `facts.md` (stable design-time facts), and `results.md` (curated
recorded numbers + interpretation).

`01_*`–`12_*` are earlier serving-bench evidence packages and the
Exp 12 server-vs-server comparison; see each directory for details.
