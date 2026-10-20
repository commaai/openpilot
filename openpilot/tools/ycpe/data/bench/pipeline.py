#!/usr/bin/env python3
"""Loads dev/pipeline.html in Chrome (cold, then warm from OPFS) and prints load timings, dropped frames and peak memory.
Needs ./serve.py running: data/bench/pipeline.py --port 8110 --repeat 10"""
import argparse
import tempfile
import time

import psutil
from playwright.sync_api import sync_playwright


def rss(profile):
  # every process of this browser instance carries the profile dir on its command line (renderers via the zygote parent)
  procs = [p for p in psutil.process_iter(["cmdline"]) if profile in " ".join(p.info["cmdline"] or [])]
  procs += [c for p in procs for c in p.children(recursive=True)]
  return sum(p.memory_info().rss for p in {p.pid: p for p in procs}.values() if p.is_running())


# long animation frames with their script attribution, from page start
LOAF = """
window.loafs = [];
new PerformanceObserver((l) => {
  for (const e of l.getEntries()) {
    window.loafs.push({start: Math.round(e.startTime), ms: Math.round(e.duration), block: Math.round(e.blockingDuration),
      scripts: e.scripts.map((s) => `${s.invoker || s.invokerType} ${s.sourceURL.split("/").pop()}:${s.sourceFunctionName} ${Math.round(s.duration)}ms`)});
  }
}).observe({type: "long-animation-frame", buffered: true});
"""


def run(ctx, url, profile, timeout):
  page = ctx.new_page()
  page.add_init_script(LOAF)
  page.on("pageerror", lambda e: print("pageerror:", e))
  page.goto(url)
  peak, start = 0, time.monotonic()
  while time.monotonic() - start < timeout:
    peak = max(peak, rss(profile))
    if page.evaluate("window.bench?.done != null"):
      break
    time.sleep(0.25)
  time.sleep(0.5)
  print(page.inner_text("#stats"))
  print(f"peak browser RSS {peak / 1e6:.0f} MB")
  print("  frame gaps over 50 ms [at, ms]:", page.evaluate("bench.gaps"))
  for f in page.evaluate("loafs"):
    if f["ms"] > 50:
      print(f"  long frame at {f['start'] - page.evaluate('bench.start'):.0f} ms: {f['ms']} ms (blocking {f['block']}) {f['scripts']}")
  page.screenshot(path=f"/tmp/ycpe-pipeline-{int(start)}.png")
  page.close()


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--port", type=int, default=8110)
  ap.add_argument("--repeat", type=int, default=1)
  ap.add_argument("--t", type=float, default=0)
  ap.add_argument("--warm", type=int, default=1, help="warm runs after the cold one")
  ap.add_argument("--timeout", type=float, default=600)
  args = ap.parse_args()
  url = f"http://localhost:{args.port}/dev/pipeline.html?repeat={args.repeat}&t={args.t}"
  with tempfile.TemporaryDirectory(prefix="ycpe-bench-") as profile, sync_playwright() as p:
    ctx = p.chromium.launch_persistent_context(profile, executable_path="/usr/bin/google-chrome", viewport={"width": 1200, "height": 600})
    for i in range(1 + args.warm):
      print(f"--- {'cold' if i == 0 else 'warm'} ×{args.repeat}")
      run(ctx, url, profile, args.timeout)
    ctx.close()
