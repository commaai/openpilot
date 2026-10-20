#!/usr/bin/env python3
"""YCPE QA harness: drives scenarios/*.json in Chrome, records frame timing and UI lint, writes out/latest/report.{html,md}."""
import argparse
import json
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

import report

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
PORTS = range(8170, 8180)
COMMA_API = re.compile(r"^https://(api\.commadotai\.com|api\.comma\.ai|athena\.comma\.ai)/")
READ_METHODS = ("GET", "HEAD", "OPTIONS")
DEFAULT_BUDGETS = {"loaf_ms": 50, "work_p99_ms": 8}
DEFAULT_VIEWPORT = {"width": 1440, "height": 900}
# gpu: headless Chrome on the NVIDIA GPU through ANGLE/EGL; hardware raster and compositing, 60 Hz begin-frames
# soft: headless SwiftShader; main-thread numbers are fine, raster/composite cost is inflated
# xvfb: headful window on a private Xvfb (llvmpipe GL); real window/input path, software GPU
MODES = {
  "gpu": ["--enable-gpu", "--use-gl=angle", "--use-angle=gl-egl"],
  "soft": [],
  "xvfb": [],
}
PROBE_JS = (HERE / "probe.js").read_text()
LINT_JS = (HERE / "lint.js").read_text()
SETTLE_JS = "document.fonts.ready.then(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))))"
# noise that never points at a bug: cancelled loads, and Chrome's echo of an HTTP error we already record with its URL
IGNORED_EVENTS = re.compile(r"net::ERR_ABORTED|^Failed to load resource")


def free_port():
  for port in PORTS:
    with socket.socket() as s:
      if s.connect_ex(("127.0.0.1", port)) != 0:
        return port
  raise RuntimeError(f"no free port in {PORTS}")


def start_server():
  port = free_port()
  proc = subprocess.Popen([sys.executable, str(HERE.parent / "serve.py"), str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
  for _ in range(100):
    with socket.socket() as s:
      if s.connect_ex(("127.0.0.1", port)) == 0:
        return proc, f"http://localhost:{port}"
    time.sleep(0.05)
  proc.kill()
  raise RuntimeError("serve.py did not start")


def start_xvfb():
  display = next(f":{n}" for n in range(170, 180) if not Path(f"/tmp/.X11-unix/X{n}").exists())
  proc = subprocess.Popen(["Xvfb", display, "-screen", "0", "1920x1080x24", "-nolisten", "tcp"], stderr=subprocess.DEVNULL)
  time.sleep(0.5)
  return proc, display


def point(page, target, timeout):
  if isinstance(target, list):
    return target
  b = page.locator(target).first.bounding_box(timeout=timeout)
  if b is None:
    raise RuntimeError(f"{target} has no box")
  return [b["x"] + b["width"] / 2, b["y"] + b["height"] / 2]


def do_step(page, step, timeout, take_shot):
  op = next(k for k in step if k != "timeout")
  arg = step[op]
  timeout = step.get("timeout", timeout)
  if op == "click" and isinstance(arg, str):
    page.locator(arg).first.click(timeout=timeout)
  elif op == "click":
    page.mouse.click(*arg)
  elif op == "hover":
    page.mouse.move(*point(page, arg, timeout))
  elif op == "drag":
    x, y = point(page, arg["from"], timeout)
    tx, ty = point(page, arg["to"], timeout) if "to" in arg else (x + arg["by"][0], y + arg["by"][1])
    page.mouse.move(x, y)
    page.mouse.down()
    page.mouse.move(tx, ty, steps=arg.get("steps", 10))
    page.mouse.up()
  elif op == "wheel":
    page.mouse.move(*point(page, arg.get("at", [DEFAULT_VIEWPORT["width"] / 2, 300]), timeout))
    page.mouse.wheel(arg.get("dx", 0), arg.get("dy", 0))
  elif op == "key":
    for _ in range(step.get("repeat", 1)):
      page.keyboard.press(arg)
  elif op == "wait":
    page.wait_for_timeout(arg)
  elif op == "waitFor" and isinstance(arg, str):
    page.locator(arg).first.wait_for(state="visible", timeout=timeout)
  elif op == "waitFor":
    page.wait_for_function(arg["fn"], timeout=timeout)
  elif op == "eval":
    page.evaluate(arg)
  elif op == "shot":
    take_shot(arg)
  else:
    raise ValueError(f"unknown step {op}")


def run_scenario(browser, base, sc, out, token):
  res = {"name": sc["name"], "url": sc["url"], "failures": [], "events": [], "shots": [], "reference": sc.get("reference", [])}
  vp = {**DEFAULT_VIEWPORT, **sc.get("viewport", {})}
  ctx = browser.new_context(viewport={"width": vp["width"], "height": vp["height"]}, is_mobile=vp.get("mobile", False),
                            has_touch=vp.get("mobile", False), device_scale_factor=1)
  allow = [re.compile(p) for p in sc.get("allow", [])]

  def event(kind, text):
    if not IGNORED_EVENTS.search(text) and not any(p.search(text) for p in allow):
      res["events"].append({"kind": kind, "text": text[:400], "shot": len(res["shots"])})

  def guard(route):
    req = route.request
    if req.method in READ_METHODS:
      return route.continue_()
    event("blocked", f"{req.method} {req.url}")
    route.abort("blockedbyclient")

  ctx.route(COMMA_API, guard)
  ctx.add_init_script(PROBE_JS)
  if sc.get("auth"):
    if not token:
      res["failures"].append("auth: no access_token in ~/.comma/auth.json")
    ctx.add_init_script(f"localStorage.setItem('comma_jwt', {json.dumps(token or '')})")
  page = ctx.new_page()
  page.on("console", lambda m: m.type in ("error", "warning") and event(f"console-{m.type}", m.text))
  page.on("pageerror", lambda e: event("page-error", f"{e.name}: {e.message}"))
  page.on("requestfailed", lambda r: r.failure != "net::ERR_BLOCKED_BY_CLIENT" and event("request-failed", f"{r.failure} {r.method} {r.url}"))
  page.on("response", lambda r: r.status >= 400 and event("http-error", f"{r.status} {r.request.method} {r.url}"))
  if sc.get("cpu_throttle"):
    ctx.new_cdp_session(page).send("Emulation.setCPUThrottlingRate", {"rate": sc["cpu_throttle"]})

  shifts_seen = 0

  def take_shot(name):
    nonlocal shifts_seen
    page.evaluate(SETTLE_JS)
    path = out / "shots" / sc["name"] / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(path), animations="disabled", caret="hide")
    lint = page.evaluate(LINT_JS, {"ignore": ", ".join(sc.get("lint_ignore", [])), "shiftsFrom": shifts_seen})
    shifts_seen = lint["shifts"]
    res["shots"].append({"name": name, "path": str(path.relative_to(out)), "issues": lint["issues"], "cls": lint["cls"]})

  t0 = time.monotonic()
  try:
    page.goto(base + sc["url"], wait_until="load")
    for step in sc.get("steps", []):
      do_step(page, step, sc.get("timeout", 10000), take_shot)
  except Exception as e:
    res["failures"].append(f"step failed: {str(e).splitlines()[0]}")
    take_shot("failure")
  res["perf"] = collect_perf(page)
  res["duration"] = time.monotonic() - t0
  ctx.close()
  check(res, {**DEFAULT_BUDGETS, **sc.get("budgets", {})})
  return res


def collect_perf(page):
  probe = page.evaluate("({raf: __probe.raf, loaf: __probe.loaf})")
  app = page.evaluate("window.__perf ? JSON.parse(JSON.stringify({frames: __perf.frames, longFrames: __perf.longFrames, drops: __perf.drops})) : null")
  raf = probe["raf"]
  return {"t": raf[1:], "interval": [b - a for a, b in zip(raf, raf[1:], strict=False)], "loaf": probe["loaf"], "app": app}


def percentile(xs, p):
  xs = sorted(xs)
  return xs[min(len(xs) - 1, int(len(xs) * p))] if xs else 0


def check(res, budgets):
  perf = res["perf"]
  iv = perf["interval"]
  work = [f.get("work", 0) for f in (perf["app"] or {}).get("frames") or []]
  res["stats"] = {
    "frames": len(iv),
    "worst_ms": max(iv, default=0),
    "p99_ms": percentile(iv, 0.99),
    "loaf": len(perf["loaf"]),
    "loaf_worst_ms": max((x["duration"] for x in perf["loaf"]), default=0),
    "work_p99_ms": percentile(work, 0.99) if work else None,
    "work_max_ms": max(work, default=None),
  }
  s = res["stats"]
  if s["loaf_worst_ms"] > budgets["loaf_ms"]:
    res["failures"].append(f"budget: long animation frame {s['loaf_worst_ms']:.0f} ms > {budgets['loaf_ms']} ms")
  if s["work_p99_ms"] is not None and s["work_p99_ms"] > budgets["work_p99_ms"]:
    res["failures"].append(f"budget: perf.js work p99 {s['work_p99_ms']:.1f} ms > {budgets['work_p99_ms']} ms")
  for kind in ("blocked", "page-error", "console-error"):
    if n := sum(e["kind"] == kind for e in res["events"]):
      res["failures"].append(f"{n} {kind}")
  res["pass"] = not res["failures"]


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("-k", help="only scenarios whose name contains this")
  ap.add_argument("--mode", choices=MODES, default="gpu")
  ap.add_argument("--bless", action="store_true", help="store this run's shots as the goldens")
  args = ap.parse_args()

  scenarios = [json.loads(p.read_text()) for p in sorted((HERE / "scenarios").glob("*.json"))]
  scenarios = [s for s in scenarios if not args.k or args.k in s["name"]]
  auth_file = Path.home() / ".comma" / "auth.json"
  token = json.loads(auth_file.read_text()).get("access_token") if auth_file.exists() else None

  latest, prev = OUT / "latest", OUT / "prev"
  shutil.rmtree(prev, ignore_errors=True)
  if latest.exists():
    latest.rename(prev)
  latest.mkdir(parents=True)

  t0 = time.monotonic()
  server, base = start_server()
  xvfb, env = None, None
  if args.mode == "xvfb":
    xvfb, display = start_xvfb()
    env = {"DISPLAY": display}
  results = []
  try:
    with sync_playwright() as p:
      browser = p.chromium.launch(executable_path="/usr/bin/google-chrome", headless=args.mode != "xvfb", args=MODES[args.mode], env=env)
      for sc in scenarios:
        r = run_scenario(browser, base, sc, latest, token)
        print(f"{'PASS' if r['pass'] else 'FAIL'} {r['name']:<24} {r['duration']:5.1f}s  {'; '.join(r['failures'])}")
        results.append(r)
      browser.close()
  finally:
    server.kill()
    if xvfb:
      xvfb.kill()

  (latest / "results.json").write_text(json.dumps({"mode": args.mode, "time": time.time(), "duration": time.monotonic() - t0, "scenarios": results}))
  if args.bless:
    report.bless(latest)
  report.build(latest, prev)
  print(f"{sum(r['pass'] for r in results)}/{len(results)} passed in {time.monotonic() - t0:.1f}s")
  print(f"report: {latest / 'report.html'}\n        {latest / 'report.md'}")
  sys.exit(0 if all(r["pass"] for r in results) else 1)


if __name__ == "__main__":
  main()
