#!/usr/bin/env python3
"""The longest frames in a trace from `run.py --trace`: for each, the events with the most self time on the renderer
and GPU process threads, and the panes that its Layout and Paint events touched.

  ./trace.py out/latest/traces/layout-drag.json [--top 10] [--frames 1]
"""
import argparse
import itertools
import json
import re
from collections import defaultdict
from pathlib import Path

PANE = re.compile(r"\b([\w-]+-pane|pane-body|topbar|playbar)\b")


def load(path):
  d = json.loads(Path(path).read_text())
  events = d["traceEvents"] if isinstance(d, dict) else d
  names = {(e["pid"], e["tid"]): e["args"]["name"] for e in events if e.get("ph") == "M" and e["name"] == "thread_name"}
  procs = {e["pid"]: e["args"]["name"] for e in events if e.get("ph") == "M" and e["name"] == "process_name"}
  return events, names, procs


def spans(events):
  """(pid, tid) → [(start, end, event)], complete events and B/E pairs alike"""
  out, open_ = defaultdict(list), defaultdict(list)
  for e in sorted((e for e in events if e.get("ph") in "XBE" and "ts" in e), key=lambda e: e["ts"]):
    key = (e["pid"], e["tid"])
    if e["ph"] == "X":
      out[key].append((e["ts"], e["ts"] + e.get("dur", 0), e))
    elif e["ph"] == "B":
      open_[key].append(e)
    elif open_[key]:
      b = open_[key].pop()
      out[key].append((b["ts"], e["ts"], {**b, "args": {**b.get("args", {}), **e.get("args", {})}}))
  return out


def self_times(thread_spans, a, b):
  """event name → self time (µs) inside [a, b]: each span's clipped duration minus its children's"""
  clip = lambda s, e: max(0, min(e, b) - max(s, a))  # noqa: E731
  total, stack = defaultdict(float), []
  for s, e, ev in sorted(thread_spans, key=lambda x: (x[0], -x[1])):
    while stack and stack[-1][1] <= s:
      stack.pop()
    if stack:
      total[stack[-1][2]["name"]] -= clip(s, min(e, stack[-1][1]))
    total[ev["name"]] += clip(s, e)
    stack.append((s, e, ev))
  return total


def dom_index(dom_path):
  """backendNodeId → (description, the pane it sits in)"""
  if not Path(dom_path).exists():
    return {}
  index = {}

  def walk(node, pane):
    attrs = dict(itertools.batched(node.get("attributes", []), 2))
    desc = node.get("localName", "") + "".join(f".{c}" for c in attrs.get("class", "").split()[:2])
    m = PANE.search(attrs.get("class", ""))
    pane = m.group(1) if m else pane
    index[node["backendNodeId"]] = (desc, pane)
    for child in node.get("children", []) + node.get("shadowRoots", []) + ([node["contentDocument"]] if "contentDocument" in node else []):
      walk(child, pane)

  walk(json.loads(Path(dom_path).read_text())["root"], None)
  return index


def targets(ev):
  args = ev.get("args", {})
  data = args.get("data", {})
  ids = [data.get("nodeId")] + [r.get("nodeId") for r in args.get("endData", {}).get("layoutRoots", [])]
  return [i for i in ids if i]


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("trace")
  ap.add_argument("--top", type=int, default=10)
  ap.add_argument("--frames", type=int, default=1, help="how many of the longest frames to break down")
  args = ap.parse_args()

  events, names, procs = load(args.trace)
  by_thread = spans(events)
  renderer = max((k for k, n in names.items() if n == "CrRendererMain"),
                 key=lambda k: sum(e["name"] == "FireAnimationFrame" for _, _, e in by_thread[k]))
  # a frame is the gap between two animation frames on the renderer's main thread
  raf = sorted(s for s, _, e in by_thread[renderer] if e["name"] == "FireAnimationFrame")
  gaps = sorted(itertools.pairwise(raf), key=lambda g: g[0] - g[1])[:args.frames]
  threads = [k for k in by_thread if k[0] == renderer[0] or procs.get(k[0]) == "GPU Process"]
  nodes = dom_index(re.sub(r"\.json$", ".dom.json", args.trace))

  for a, b in gaps:
    print(f"frame of {(b - a) / 1000:.0f} ms, {(a - raf[0]) / 1e6:.2f} s after the first animation frame")
    rows = []
    for k in threads:
      for name, us in self_times(by_thread[k], a, b).items():
        rows.append((us, f"{procs.get(k[0], k[0])} {names.get(k, k[1])}", name))
    for us, thread, name in sorted(rows, reverse=True)[:args.top]:
      print(f"  {us / 1000:7.1f} ms  {thread:<40} {name}")
    touched = defaultdict(float)
    for s, e, ev in by_thread[renderer]:
      if ev["name"] in ("Layout", "Paint", "PrePaint") and s < b and e > a:
        for i in targets(ev) or [None]:
          desc, pane = nodes.get(i, (f"node {i}" if i else "document", None))
          touched[f"{ev['name']} {pane or '-'} ({desc})"] += (min(e, b) - max(s, a)) / 1000
    for what, ms in sorted(touched.items(), key=lambda x: -x[1])[:args.top]:
      print(f"  {ms:7.1f} ms  {what}")


if __name__ == "__main__":
  main()
