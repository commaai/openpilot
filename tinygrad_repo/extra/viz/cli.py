#!/usr/bin/env python3
import argparse, pathlib
from typing import Iterator
from tinygrad.viz import serve as viz
from tinygrad.uop.ops import RewriteTrace
from tinygrad.helpers import temp, ansistrip, colored

def optional_eq(val:dict, arg:str|None) -> bool: return arg is None or ansistrip(val["name"]) == arg

def print_data(data:dict) -> None:
  if isinstance(data.get("value"), Iterator):
    for m in data["value"]:
      if not m["diff"]: continue
      fp = pathlib.Path(m["upat"][0][0])
      print(f"{fp.parent.name}/{fp.name}:{m['upat'][0][1]}")
      print(m["upat"][1])
      for line in m["diff"]:
        color = "red" if line.startswith("-") else "green" if line.startswith("+") else None
        print(colored(line, color))
  if data.get("src") is not None: print(data["src"])

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--kernel', type=str, default=None, metavar="NAME", help='Select a kernel by name (optional name, default: only list names)')
  parser.add_argument('--select', type=str, default=None, metavar="NAME",
                      help='Select an item within the chosen kernel (optional name, default: only list names)')
  args = parser.parse_args()

  viz.trace = viz.load_pickle(pathlib.Path(temp("rewrites.pkl", append_user=True)), default=RewriteTrace([], [], {}))
  viz.ctxs = viz.get_rewrites(viz.trace)
  for k in viz.ctxs:
    if not optional_eq(k, args.kernel): continue
    print(k["name"])
    if args.kernel is None: continue
    for s in k["steps"]:
      if not optional_eq(s, args.select): continue
      print(" "*s["depth"]+s['name']+(f" - {s['match_count']}" if s.get('match_count') is not None else ''))
      if args.select is not None: print_data(viz.get_render(s['query']))
