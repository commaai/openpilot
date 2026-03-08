#!/usr/bin/env python3
import argparse, pathlib
from typing import Iterator
from tinygrad.viz import serve as viz
from tinygrad.uop.ops import RewriteTrace
from tinygrad.helpers import temp, ansistrip, colored, time_to_str, ansilen
from test.null.test_viz import load_profile

def optional_eq(val:dict, arg:str|None) -> bool: return arg is None or ansistrip(val["name"]) == arg

def print_data(data:dict) -> None:
  if isinstance(data.get("value"), Iterator):
    for m in data["value"]:
      if m.get("uop"):
        print("Input UOp:")
        print(m["uop"])
      if not m["diff"]: continue
      print("Rewrites:")
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
                      help='Rewrites: Select an item within the chosen kernel (optional name, default: only list names)')
  parser.add_argument('--profile', action="store_true", help="View profiling trace (default: views rewrites)")
  parser.add_argument('--device', type=str, default=None, metavar="NAME", help="Profile only: Select a device (default: prints all devices)")
  args = parser.parse_args()

  viz.trace = viz.load_pickle(pathlib.Path(temp("rewrites.pkl", append_user=True)), default=RewriteTrace([], [], {}))
  viz.ctxs = viz.get_rewrites(viz.trace)

  if args.profile:
    from tabulate import tabulate
    profile = load_profile(viz.load_pickle(pathlib.Path(temp("profile.pkl", append_user=True)), default=[]))
    agg, total, n = {}, 0, 0
    for k,v in profile["layout"].items():
      if not optional_eq({"name":k}, args.device): continue
      print(k)
      if args.device is None: continue
      for e in v.get("events", []):
        et = e["dur"]*1e-6
        if args.kernel is not None:
          if ansistrip(e["name"]) == args.kernel and n < 10:
            ptm = colored(time_to_str(et, w=9), "yellow" if et > 0.01 else None) if et is not None else ""
            name = e["name"]+(" " * (46 - ansilen(e["name"])))
            print(f"{name} {ptm}/{(et or 0)*1e3:9.2f}ms  "+e['fmt'].replace('\n', ' | ')+"  ")
            n += 1
        else:
          a = agg.setdefault(e["name"], [0.0, 0])
          a[0] += et
          a[1] += 1
          total += et

    if agg:
      rows = [[n, t, time_to_str(t, w=9), t / c if c else 0.0, c, (t / total * 100.0) if total else 0.0] for n, (t, c) in agg.items()]
      rows.sort(key=lambda r: r[1], reverse=True)
      print(tabulate([[r[0], r[2], r[4], f"{r[5]:.2f}%"] for r in rows[:30]], headers=["name", "total", "count", "pct"], tablefmt="github"))
    exit(0)

  for k in viz.ctxs:
    if not optional_eq(k, args.kernel): continue
    print(k["name"])
    if args.kernel is None: continue
    for s in k["steps"]:
      if not optional_eq(s, args.select): continue
      print(" "*s["depth"]+s['name']+(f" - {s['match_count']}" if s.get('match_count') is not None else ''))
      if args.select is not None: print_data(viz.get_render(s['query']))
