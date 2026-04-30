---
name: write-handoff
description: Write or update local/HANDOFF.md for the next fresh Claude Code session.
disable-model-invocation: true
allowed-tools: Read, Write, Edit, Bash(git status*), Bash(git branch*), Bash(git rev-parse*), Bash(git diff*), Bash(ls *), Bash(find *)
---

# Write Handoff

Write or update:

`local/HANDOFF.md`

This handoff is for the next fresh Claude Code session. Keep it short, current, and useful.

## Rules

- Do not include long history.
- Do not include old failed paths unless they directly affect the next step.
- Do not use provenance as design truth.
- Prefer current code, current git state, and current result files.
- If exact values are unknown, write `unknown`, do not invent them.
- Keep the file practical enough that the next session can start from it.

## Include

- current branch
- latest commit hash if known
- active goal
- files changed or most relevant files
- commands/tests/benchmarks run
- exact result directories
- current known facts
- current blocker
- safest next step
- things not to do
- old assumptions that were invalidated

