#!/usr/bin/env python3
# Usage: DEBUG=5 python -m tinygrad.viz.cli --json | ./extra/viz/kernel_graph.py E_8_8_16_4
import argparse, json, sys
from tinygrad.helpers import ansistrip

def get_node(graph:dict, key): return graph[str(key)]

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="print CALL graph from DEBUG=5 tinygrad.viz.cli --json output")
  parser.add_argument("kernel", type=str, nargs="?", default="ALL", metavar="NAME", help="Kernel name to stop at (default: print all kernels)")
  args = parser.parse_args()
  ref:int|None = None
  for line in sys.stdin:
    if not line.strip(): continue
    graph = json.loads(line)
    if graph.get("ref") is not None and (args.kernel == "ALL" or graph["ref"] == ref):
      print(graph)
      if (v:=json.loads(next(sys.stdin, "{}")).get("value")): print(v)
    if ref is not None or not isinstance(rec:=next(iter(graph.values()), {}), dict) or "label" not in rec: continue
    for v in graph.values():
      if not v["label"].startswith("CALL"): continue
      lines = v["label"].splitlines()
      # print the CALL and its kernel name from codegen
      print(f"{lines[0]:<12} {lines[-1]}")
      # print sources (buffer, param, multi)
      unique:dict[str, int] = {}
      for i,(_,s) in enumerate(v["src"][1:]):
        while get_node(graph, s)["label"].startswith("AFTER"): s = get_node(graph, s)["src"][0][1]
        if (num:=unique.get(str(s))) is None: unique[str(s)] = num = len(unique)
        print(f"SRC {i} {' '.join(get_node(graph, s)['label'].splitlines())} g{num}")
      # print access patterns
      ss = [v["src"][0][1]]
      seen:set[str] = set()
      while ss:
        if (s:=str(ss.pop())) in seen: continue
        seen.add(s)
        if get_node(graph, s)["label"].startswith("INDEX"):
          idx_str = get_node(graph, s)["label"].splitlines()
          src_str = ["SRC"]+get_node(graph, get_node(graph, s)["src"][0][1])["label"].splitlines()[1:]
          print(" ".join(idx_str+src_str))
        ss += [x[1] for x in get_node(graph, s)["src"]]
      if args.kernel != "ALL" and args.kernel in ansistrip(v["label"]):
        ref = v["ref"]
        break
