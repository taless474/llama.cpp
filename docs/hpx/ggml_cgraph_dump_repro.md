# Reproducing a tiny ggml cgraph excerpt from llama.cpp

This note documents how to dump a real `ggml_cgraph` from upstream `llama.cpp`, extract a small readable subgraph, and render it as SVG for architecture slides.

The goal is to produce a small, evidence-backed visual for the `ggml graph IR` box in the llama.cpp architecture diagram.

## Example summary

We temporarily patched `src/llama-context.cpp` to dump the internal ggml graph during `llama_decode`.

In the TinyLlama run used here, the dumped DOT had hundreds of labeled tensor nodes and over a thousand dependency edges. We therefore extract a smaller subgraph, such as nodes `0..24`, and render only that subset.

Optional matching DOT source and graph:

```text
docs/hpx/figures/ggml_cgraph_tinyllama_layer0_nodes_0_24.svg
docs/hpx/figures/ggml_cgraph_tinyllama_layer0_nodes_0_24.dot
```

The tiny graph is a real excerpt from a dumped TinyLlama `ggml_cgraph`.

- Each box is a `ggml_tensor`.
- The operation is stored on the result tensor.
- Edges are `src[]` dependencies.
- `ggml_cgraph` is the graph container / ordered compute graph.

For example:

```text
token_embd.weight --src 0--> embd :: get_rows(x)
inp_tokens        --src 1--> embd :: get_rows(x)
```

means:

```text
embd = get_rows(token_embd.weight, inp_tokens)
```

and:

```text
norm-0            --src 0--> attn_norm-0 :: x*y
attn_norm.weight  --src 1--> attn_norm-0 :: x*y
```

means:

```text
attn_norm_0 = rms_norm(embd) ⊙ blk.0.attn_norm.weight
```


## Temporary graph-dump patch

In `src/llama-context.cpp`, add:

```cpp
#include <cstdio>
#include <cstdlib>
```

near the other includes.

Then, near the existing commented graph dump block inside `llama_context::decode`, add:

```cpp
// plot the computation graph in dot format (for debugging purposes)
if (std::getenv("LLAMA_DUMP_GRAPH_ONCE")) {
    static bool dumped_graph = false;
    if (!dumped_graph) {
        dumped_graph = true;

        const char * path = std::getenv("LLAMA_DUMP_GRAPH_PATH");
        if (path == nullptr) {
            path = "llama_graph.dot";
        }

        ggml_graph_print(res->get_gf());
        ggml_graph_dump_dot(res->get_gf(), nullptr, path);

        fprintf(stderr, "[llama_graph_dump] wrote %s\n", path);
    }
}
```

This is env-gated, so normal runs are unchanged unless `LLAMA_DUMP_GRAPH_ONCE=1` is set.

## Build

From repo root, using the existing baseline build directory:

```bash
cmake --build ../builds/llama-base --target llama-cli -j
```

## Dump a graph

Example model:

```bash
MODEL="../models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
```

Run:

```bash
mkdir -p local/graphdump

LLAMA_DUMP_GRAPH_ONCE=1 \
LLAMA_DUMP_GRAPH_PATH="local/graphdump/llama_graph.dot" \
../builds/llama-base/bin/llama-cli \
  -m "$MODEL" \
  -p "Hi" \
  -n 1 \
  -ngl 0 \
  --ctx-size 64 \
  --batch-size 1 \
  --ubatch-size 1 \
  --threads 1 \
  > local/graphdump/run.stdout \
  2> local/graphdump/run.stderr
```

In the run used to produce this sample, the program aborted after writing the DOT due to an internal batch-size assertion, but the DOT file was successfully written first. The successful signal was:

```text
[llama_graph_dump] wrote local/graphdump/llama_graph.dot
```

If you want to avoid the abort later, try a less constrained batch setting. The graph artifact was already emitted before the assertion in the documented run.

## Inspect the dump

```bash
head -40 local/graphdump/llama_graph.dot
```

Useful grep:

```bash
grep -E "get_rows|rms_norm|X\*Y|rope|flash_attn_ext|x\+y|glu|set_rows|view|reshape|permute" \
  local/graphdump/llama_graph.dot | head -80
```

Count nodes and edges:

```bash
grep -c "label=" local/graphdump/llama_graph.dot
grep -c -- "->" local/graphdump/llama_graph.dot
```

## Convert DOT addresses to readable node names

Use this helper to print nodes `0..32` and their incoming `src[]` edges:

```bash
python3 - <<'PY'
from pathlib import Path
import re

dot = Path("local/graphdump/llama_graph.dot").read_text().splitlines()

nodes = {}
edges = []

node_re = re.compile(r'^\s*"([^"]+)" \[ .*label="([^"]+)"; \]$')
edge_re = re.compile(r'^\s*"([^"]+)" -> "([^"]+)" .*label = "(src \d+)"; \]$')

for line in dot:
    m = node_re.match(line)
    if m:
        addr, label = m.groups()
        parts = label.split("|")
        name = parts[0].strip()
        mid  = parts[1].strip() if len(parts) > 1 else ""
        op   = parts[2].replace("<x>", "").strip() if len(parts) > 2 else ""
        idx_m = re.match(r"(\d+)\s+\[", mid)
        idx = int(idx_m.group(1)) if idx_m else None
        nodes[addr] = {"idx": idx, "name": name, "mid": mid, "op": op, "label": label}
        continue

    m = edge_re.match(line)
    if m:
        src, dst, srcn = m.groups()
        edges.append((src, dst, srcn))

def pretty(addr):
    n = nodes.get(addr)
    if not n:
        return addr
    idx = "leaf" if n["idx"] is None else f'{n["idx"]:03d}'
    op = f' :: {n["op"]}' if n["op"] else ""
    return f'{idx} {n["name"]}{op}'

print("=== Nodes 0..32 ===")
for addr, n in sorted(nodes.items(), key=lambda kv: (999999 if kv[1]["idx"] is None else kv[1]["idx"])):
    if n["idx"] is not None and n["idx"] <= 32:
        print(f'{n["idx"]:03d}: {n["name"]:<32} {n["op"]}')

print()
print("=== Edges into nodes 0..32 ===")
for src, dst, srcn in edges:
    nd = nodes.get(dst)
    if nd and nd["idx"] is not None and nd["idx"] <= 32:
        print(f'{pretty(src)}  --{srcn}-->  {pretty(dst)}')
PY
```

The real extracted nodes began with:

```text
000: embd (f32)                       get_rows(x)
001: norm-0 (f32)                     rms_norm(x)
002: attn_norm-0 (f32)                x*y
003: Qcur-0 (f32)                     X*Y
004: Qcur-0 (f32)                     reshape(x)
005: Qcur-0 (f32)                     rope(x)
```

The real edges showed:

```text
token_embd.weight --src 0--> embd :: get_rows(x)
inp_tokens        --src 1--> embd :: get_rows(x)

embd              --src 0--> norm-0 :: rms_norm(x)

norm-0            --src 0--> attn_norm-0 :: x*y
attn_norm.weight  --src 1--> attn_norm-0 :: x*y

attn_q.weight     --src 0--> Qcur-0 :: X*Y
attn_norm-0       --src 1--> Qcur-0 :: X*Y
```

So the equations for the first part of the excerpt are:

```text
embd        = get_rows(token_embd.weight, inp_tokens)
norm_0      = rms_norm(embd)
attn_norm_0 = norm_0 ⊙ blk.0.attn_norm.weight
Qcur_0      = blk.0.attn_q.weight × attn_norm_0
Qcur_0_r    = reshape(Qcur_0)
Qcur_0_rope = rope(Qcur_0_r, pos)
```

## Extract the slide subgraph: nodes 0..24

This extracts computed nodes `0..24` and the leaf/input nodes that feed them:

```bash
python3 - <<'PY'
from pathlib import Path
import re

src = Path("local/graphdump/llama_graph.dot")
dst = Path("local/graphdump/tiny_0_24.dot")

lines = src.read_text().splitlines()

node_re = re.compile(r'^\s*"([^"]+)" \[ .*label="([^"]+)"; \]$')
edge_re = re.compile(r'^\s*"([^"]+)" -> "([^"]+)" .*label = "(src \d+)"; \]$')

nodes = {}
edges = []

for line in lines:
    m = node_re.match(line)
    if m:
        addr, label = m.groups()
        parts = label.split("|")
        mid = parts[1].strip() if len(parts) > 1 else ""
        idx_m = re.match(r"(\d+)\s+\[", mid)
        idx = int(idx_m.group(1)) if idx_m else None
        nodes[addr] = {"line": line, "label": label, "idx": idx}
        continue

    m = edge_re.match(line)
    if m:
        src_addr, dst_addr, srcn = m.groups()
        edges.append((src_addr, dst_addr, srcn, line))

# Keep computed nodes 0..24.
keep = {
    addr for addr, n in nodes.items()
    if n["idx"] is not None and 0 <= n["idx"] <= 24
}

# Also keep all leaf/input nodes that connect into kept nodes.
changed = True
while changed:
    changed = False
    for s, d, srcn, line in edges:
        if d in keep and s not in keep:
            keep.add(s)
            changed = True

with dst.open("w") as f:
    f.write("digraph G {\n")
    f.write("  rankdir = TB;\n")
    f.write("  newrank = true;\n")

    for addr, n in nodes.items():
        if addr in keep:
            f.write(n["line"] + "\n")

    for s, d, srcn, line in edges:
        if s in keep and d in keep:
            f.write(line + "\n")

    f.write("}\n")

print(f"wrote {dst}")
print(f"kept {sum(1 for a in keep if nodes[a]['idx'] is not None)} indexed nodes")
print(f"kept {sum(1 for a in keep if nodes[a]['idx'] is None)} leaf/input nodes")
PY
```

## Render the tiny graph

Install Graphviz if needed. Render SVG:

```bash
dot -Tsvg local/graphdump/tiny_0_24.dot -o local/graphdump/tiny_0_24.svg
open local/graphdump/tiny_0_24.svg
```

## Clean up the temporary source patch

After generating the graph artifacts, restore the debug patch:

```bash
git restore src/llama-context.cpp
```

## Notes

- The DOT labels use lowercase/symbolic op labels such as `get_rows(x)`, `rms_norm(x)`, `x*y`, `X*Y`, `reshape(x)`, and `rope(x)`.
- The tiny SVG is intended as an illustrative excerpt from a real dumped graph, not the full tinlyllama graph.
- Find the example here in `docs/hpx/figures/` as `ggml_cgraph_tinyllama_layer0_nodes_0_24.dot` and `ggml_cgraph_tinyllama_layer0_nodes_0_24.svg`.
