# 01 — llama-server repeatability smoke

This directory packages the first repeatability check for the local llama-server baseline.

## Purpose

If we run the same simple llama-server request in two local sessions, do the outputs remain stable enough to use llama-server as a baseline reference?


## What this proved

This step showed that the local llama-server baseline was repeatable enough to continue using it as a reference.

It checked the two captured sessions and produced a compact comparison report:

```text
compare_report.txt
```

The helper script used for the comparison is included:

```text
_compare_sessions.py
```

## What this did not prove

This step did not prove:

- server-vs-harness timing equivalence
- std-vs-HPX correctness
- HPX lifecycle behavior
- performance under concurrency
- benchmark-grade latency or throughput


It only reduced uncertainty around the initial llama-server baseline.

## Why this matters

Before comparing `llama-serving-bench` against llama-server, we needed to know that the llama-server reference path was not obviously unstable across repeated local runs.

This repeatability smoke helped justify the later aligned comparison work.
