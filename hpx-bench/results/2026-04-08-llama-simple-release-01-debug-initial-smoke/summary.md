# llama-simple smoke results — 01-debug-initial-smoke

## Run
- date: 2026-04-08
- commit: e387827e4dafa06ce3b3e159d83da216434ebd3e
- build: Debug
- model: TinyLlama-1.1B-Chat-v1.0.Q4_K_M
- prompt: "Hello my name is"
- n_predict: 32
- ngl: 0
- note: raw logs not saved; data reconstructed from session output

## Correctness
- output text identical: YES
- no NaNs: YES
- no crash: YES
- no divergence: YES

## Timing

| metric                    | baseline        | hpx (LLAMA_USE_HPX=1) | delta      |
|---------------------------|-----------------|------------------------|------------|
| load time                 | 1436 ms         | 1697 ms                | +1.2×      |
| prompt eval (5 tok)       | 484 ms → 10.3 t/s | 822 ms → 6.1 t/s     | −1.7×      |
| decode (31 runs)          | 5391 ms → 5.75 t/s | 6055 ms → 5.12 t/s  | −1.1×      |
| total                     | 6833 ms         | 7759 ms                | −1.1×      |

## Notes
- Debug build; numbers not representative of optimized performance
- HPX path slower across the board — expected overhead without optimization
- Same binary, HPX toggled via LLAMA_USE_HPX=1
- graphs reused = 30 in both runs
