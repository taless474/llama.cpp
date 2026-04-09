# llama-simple smoke results — 02-release-post-sched-reset-fix

## Run
- date: 2026-04-08
- commit: e387827e4dafa06ce3b3e159d83da216434ebd3e
- build: Release
- model: TinyLlama-1.1B-Chat-v1.0.Q4_K_M
- ngl: 0
- fix: removed sched_reset + sched_alloc_graph from inside graph_compute (HPX path)
  - src/llama-context.cpp lines 2210-2213 deleted
  - root cause: HPX path was resetting and reallocating after set_inputs had already
    written input data, clobbering it before compute ran
- note: raw logs not saved; data reconstructed from session output

## Correctness
- S1 "Hello my name is" -n 32:  match
- S2 "Hello "*200      -n 1:    match  (was diverging <unk> before fix)
- S3 "Hello"           -n 128:  match

## Timing

### S1 — "Hello my name is", -n 32
| metric                | baseline        | hpx (LLAMA_USE_HPX=1) | delta    |
|-----------------------|-----------------|-----------------------|----------|
| prompt eval (5 tok)   | 35 ms → 142 t/s | 101 ms → 49 t/s       | −2.9×    |
| decode (31 runs)      | 301 ms → 103 t/s | 307 ms → 101 t/s     | noise    |

### S2 — "Hello "*200, -n 1 (prefill-heavy)
| metric                | baseline         | hpx (LLAMA_USE_HPX=1) | delta    |
|-----------------------|------------------|-----------------------|----------|
| prompt eval (202 tok) | 620 ms → 326 t/s | 628 ms → 322 t/s      | −1.0×    |

### S3 — "Hello", -n 128 (decode-heavy)
| metric                | baseline        | hpx (LLAMA_USE_HPX=1) | delta    |
|-----------------------|-----------------|-----------------------|----------|
| prompt eval (2 tok)   | 24 ms → 84 t/s  | 112 ms → 18 t/s       | −4.7×    |
| decode (82 runs)      | 817 ms → 100 t/s | 864 ms → 95 t/s      | noise    |

## Notes
- S2 divergence (the <unk> token) was the correctness bug fixed in this run
- HPX prompt eval overhead still large for small batches (2-5 tokens)
- HPX decode parity with baseline — no regression
- Prefill overhead amortizes with batch size: 4.7× at 2 tok, 1.0× at 202 tok
