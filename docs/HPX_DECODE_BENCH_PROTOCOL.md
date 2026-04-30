## Fair comparison protocol for HPX decode benchmarks

Any PACKET=0 vs PACKET=1 (or similar A/B) decode-speed comparison MUST satisfy:

1. **Quiet machine check** before each run:
   - `ps -A -o %cpu,comm | awk 'NR==1 || $1+0 > 5.0'` shows nothing but
     Claude/editor processes.
   - `pmset -g therm` reports no warnings.
   - Kill any stray `llama-bench` / `llama-simple` / `mds_stores` at >20% CPU
     and wait for spotlight indexing to settle after rebuilds.
2. **Single rebuild per batch.** Rebuild once; use the same binary for every
   condition in a comparison. Do not rebuild between arms.
3. **Back-to-back runs, alternating order across batches.** Run (packet,
   baseline) in batch 1, (baseline, packet) in batch 2. If both batches agree
   on the speedup direction, drift is not confounding you.
4. **Same -n across conditions.** Default `-n 16` on M4 for F32 TinyLlama.
   Extend only if the plateau is unclear, never past the point where thermal
   drift reappears.
5. **Report per-bucket mean ± stdev**, not just eval tok/s:
   `lowered_ms`, `fallback_ms`, `packet_ms`. `packet_ms` should be
   low-variance (stdev < 10% of mean); `lowered_ms` normally shows
   ~25% stdev — that is scheduler noise on the non-packet path, not a
   defect in the packet work.
6. **Sanity-check the math.** `(lowered_0 - lowered_1) + (packet_0 - packet_1)`
   should equal the observed `total` delta within rounding. If it doesn't,
   one of the buckets is mis-accounted.
7. **Never compare runs across machines or CPU-contention states.** A run on
   a contended machine is not comparable to a quiet-machine run, even on the
   same hardware — the contested path's node mix amplifies contention
   differently.

Reference worked example: `hpx-bench/results/2026-04-20-packet-baseline-quiet/README.md`.