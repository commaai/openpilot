#!/usr/bin/env python3
"""Loads the demo route's plot view in the real app (cold, then warm from OPFS) and prints what __perf saw while the
logs loaded. Needs ./serve.py running: data/bench/app_load.py --port 8110 --repeat 10"""
import argparse
import tempfile
import time

import psutil
from playwright.sync_api import sync_playwright

ROUTE = "5beb9b58bd12b691/0000010a--a51155e496"


def renderer_rss(profile):
  # the largest renderer of this browser: renderers carry the profile dir on their command line
  rss = 0
  for p in psutil.process_iter(["cmdline", "memory_info"]):
    cmd = " ".join(p.info["cmdline"] or [])
    if profile in cmd and "--type=renderer" in cmd and p.info["memory_info"]:
      rss = max(rss, p.info["memory_info"].rss)
  return rss


def run(ctx, url, profile, timeout):
  page = ctx.new_page()
  page.on("pageerror", lambda e: print("pageerror:", e))
  start = time.monotonic()
  page.goto(url)
  first = peak = None
  while time.monotonic() - start < timeout:
    peak = max(peak or 0, renderer_rss(profile))
    seg = page.evaluate("window.__perf?.marks.segments")
    if seg:
      done, total = map(int, seg.split("/"))
      if done and first is None:
        first = time.monotonic() - start
      if done == total:
        break
    time.sleep(0.2)
  elapsed = time.monotonic() - start
  time.sleep(1)
  r = page.evaluate("""({drops: __perf.drops, marks: __perf.marks, heap: performance.memory.usedJSHeapSize / 1e6,
    long: __perf.longFrames.filter((f) => f.duration > 50).map((f) => `${f.duration.toFixed(0)}ms/${f.blocking.toFixed(0)} ${f.scripts}`)})""")
  stats = [f"first segment {first:.1f} s", f"complete {elapsed:.1f} s", f"dropped frames {r['drops']}", f"heap {r['heap']:.0f} MB",
           f"peak renderer RSS {peak / 1e6:.0f} MB"]
  print(" · ".join(stats))
  print(f"  {r['marks']}")
  for f in r["long"]:
    print(f"  long frame {f}")
  page.close()


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--port", type=int, default=8110)
  ap.add_argument("--repeat", type=int, default=1)
  ap.add_argument("--view", default="plot")
  ap.add_argument("--warm", type=int, default=1, help="warm runs after the cold one")
  ap.add_argument("--timeout", type=float, default=900)
  args = ap.parse_args()
  url = f"http://localhost:{args.port}/{ROUTE}/{args.view}?repeat={args.repeat}"
  with tempfile.TemporaryDirectory(prefix="ycpe-bench-") as profile, sync_playwright() as p:
    ctx = p.chromium.launch_persistent_context(profile, executable_path="/usr/bin/google-chrome", viewport={"width": 1600, "height": 900})
    for i in range(1 + args.warm):
      print(f"--- {'cold' if i == 0 else 'warm'} ×{args.repeat}")
      run(ctx, url, profile, args.timeout)
    ctx.close()
