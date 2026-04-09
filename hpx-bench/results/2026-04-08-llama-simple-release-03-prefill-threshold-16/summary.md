# llama-simple smoke results

## Run
- date: 2026-04-08
- commit: e387827e4dafa06ce3b3e159d83da216434ebd3e
- branch: hpx-prefill-orchestrator
- build: Release
- model: TinyLlama-1.1B-Chat-v1.0.Q4_K_M
- prompt: "Hello my name is"
- n_predict: 32
- ngl: 0
- note: llama-simple takes prompt as positional arg, not -p; -p is not a valid flag

## Correctness
- output text identical: YES
  - baseline: "Hello my name is John Smith and I am a software engineer. I have been working on a project for the past few months and I am excited to share it with you."
  - hpx:      same
- no NaNs: YES
- no crash: YES
- no divergence: YES

## Timing

| metric                    | baseline       | hpx (LLAMA_USE_HPX=1) | delta        |
|---------------------------|----------------|-----------------------|--------------|
| load time                 | 463 ms         | 337 ms                | −1.4×        |
| prompt eval (5 tok)       | 35 ms → 141.8 t/s | 101 ms → 49.3 t/s  | −2.9× slower |
| decode (31 runs)          | 301 ms → 102.8 t/s | 307 ms → 100.9 t/s | −1.02× (noise) |
| total                     | 766 ms         | 645 ms                | 1.2× faster  |

## Notes
- same binary used for both runs
- HPX toggled only through LLAMA_USE_HPX=1
- HPX prompt eval overhead is real and large for short prompts (5 tokens)
- HPX decode throughput is within noise of baseline at this batch size
- raw logs: baseline.txt, hpx.txt
- environment: env.txt
