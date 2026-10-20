#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["playwright", "pillow", "numpy"]
# ///
"""YCPE QA harness: drives scenarios/*.json in Chrome, records frame timing and UI lint, writes out/latest/report.{html,md}."""
import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

from playwright.sync_api import sync_playwright

import report

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
PORTS = range(8170, 8180)
COMMA_API = re.compile(r"^https://(api\.commadotai\.com|api\.comma\.ai|athena\.comma\.ai)/")
READ_METHODS = ("GET", "HEAD", "OPTIONS")
NETWORK_FLAKE = re.compile(r"ERR_NETWORK_CHANGED|ERR_INTERNET_DISCONNECTED|ERR_NAME_NOT_RESOLVED")
FATAL = ("blocked", "page-error", "console-error")  # events that fail a scenario
DEFAULT_BUDGETS = {"loaf_ms": 50, "work_p99_ms": 8, "drops": None}
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
# (temporary) panes that aren't built yet: serve.py answers their module URL with index.html
IGNORED_EVENTS = re.compile(r"net::ERR_ABORTED|^Failed to load resource|^Failed to load module script")


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


def do_step(page, step, timeout, take_shot, failures):
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
    vp = page.viewport_size
    page.mouse.move(*point(page, arg.get("at", [vp["width"] / 2, vp["height"] / 2]), timeout))
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
  elif op == "select":
    page.locator(arg[0]).first.select_option(arg[1], timeout=timeout)
  elif op == "goto":
    page.goto(urljoin(page.url, arg), wait_until="load")
  elif op == "eval":
    page.evaluate(arg)
  elif op == "assert":
    if not page.evaluate(arg):
      failures.append({"rule": "assert", "detail": f"false: {arg}", "key": arg})
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
      res["failures"].append({"rule": "auth", "detail": "no access_token in ~/.comma/auth.json", "key": "auth"})
    ctx.add_init_script(f"localStorage.setItem('comma_jwt', {json.dumps(token or '')})")
  page = ctx.new_page()
  page.on("console", lambda m: m.type in ("error", "warning") and event(f"console-{m.type}", m.text))
  page.on("pageerror", lambda e: event("page-error", f"{e.name}: {e.message}"))
  page.on("requestfailed", lambda r: not r.failure.startswith("net::ERR_BLOCKED_BY_CLIENT") and event("request-failed", f"{r.failure} {r.method} {r.url}"))
  page.on("response", lambda r: r.status >= 400 and event("http-error", f"{r.status} {r.request.method} {r.url}"))
  if sc.get("cpu_throttle"):
    ctx.new_cdp_session(page).send("Emulation.setCPUThrottlingRate", {"rate": sc["cpu_throttle"]})

  shifts_seen = 0
  paused = []  # [start, end] page times spent screenshotting and linting, excluded from frame stats

  def take_shot(name):
    nonlocal shifts_seen
    start = page.evaluate("performance.now()")
    page.evaluate(SETTLE_JS)
    path = out / "shots" / sc["name"] / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(path), animations="disabled", caret="hide")
    lint = page.evaluate(LINT_JS, {"ignore": ", ".join(sc.get("lint_ignore", [])), "shiftsFrom": shifts_seen})
    shifts_seen = lint["shifts"]
    paused.append([start, page.evaluate("performance.now()")])
    res["shots"].append({"name": name, "path": str(path.relative_to(out)), "issues": lint["issues"], "cls": lint["cls"]})

  t0 = time.monotonic()
  step = {"goto": sc["url"]}
  try:
    page.goto(base + sc["url"], wait_until="load")
    for step in sc.get("steps", []):
      do_step(page, step, sc.get("timeout", 10000), take_shot, res["failures"])
  except Exception as e:
    res["failures"].append({"rule": "step", "detail": f"{json.dumps(step)}: {str(e).splitlines()[0]}", "key": json.dumps(step)})
    try:
      take_shot("failure")
    except Exception:
      pass
  res["perf"] = collect_perf(page, paused)
  res["duration"] = time.monotonic() - t0
  ctx.close()
  check(res, {**DEFAULT_BUDGETS, **sc.get("budgets", {})})
  return res


def collect_perf(page, paused):
  probe = page.evaluate("({raf: __probe.raf, loaf: __probe.loaf})")
  app = page.evaluate("window.__perf ? JSON.parse(JSON.stringify({frames: __perf.frames, longFrames: __perf.longFrames, drops: __perf.drops})) : null")
  harness = lambda a, b: any(a < pe and b > ps for ps, pe in paused)  # noqa: E731
  raf = probe["raf"]
  frames = [(b, b - a) for a, b in zip(raf, raf[1:], strict=False) if not harness(a, b)]
  return {"t": [f[0] for f in frames], "interval": [f[1] for f in frames], "paused": paused, "app": app,
          "loaf": [e for e in probe["loaf"] if not harness(e["start"], e["start"] + e["duration"])]}


def percentile(xs, p):
  xs = sorted(xs)
  return xs[min(len(xs) - 1, int(len(xs) * p))] if xs else 0


def check(res, budgets):
  perf = res["perf"]
  iv = perf["interval"]
  work = [f.get("work", 0) for f in (perf["app"] or {}).get("frames") or []]
  refresh = percentile(iv, 0.5) or 1000 / 60
  s = res["stats"] = {
    "frames": len(iv),
    "worst_ms": max(iv, default=0),
    "p99_ms": percentile(iv, 0.99),
    "loaf": len(perf["loaf"]),
    "loaf_worst_ms": max((x["duration"] for x in perf["loaf"]), default=0),
    "work_p99_ms": percentile(work, 0.99) if work else None,
    "work_max_ms": max(work, default=None),
    "drops": sum(round(v / refresh) - 1 for v in iv if v > refresh * 1.5),
    "perf_js_drops": (perf["app"] or {}).get("drops"),
  }
  fail = lambda rule, detail, key=None: res["failures"].append({"rule": rule, "detail": detail, "key": key or detail})  # noqa: E731
  for e in perf["loaf"]:
    if e["duration"] > budgets["loaf_ms"]:
      top = sorted(e["scripts"], key=lambda x: -x["duration"])[:3]
      who = ", ".join(f"{x['fn'] or x['invoker']}{'' if x['url'] == x['invoker'] else ' ' + x['url']}:{x['pos']} {x['duration']:.0f} ms" for x in top)
      fail("long-frame", f"{e['duration']:.0f} ms frame at {e['start'] / 1000:.1f}s (blocking {e['blocking']:.0f} ms): {who or 'no script, so style/layout/paint'}",
           key=" ".join(f"{x['fn'] or x['invoker']} {x['url']}" for x in top) or "no script")
  if s["work_p99_ms"] is not None and s["work_p99_ms"] > budgets["work_p99_ms"]:
    fail("frame-work", f"perf.js work p99 {s['work_p99_ms']:.1f} ms > {budgets['work_p99_ms']} ms (max {s['work_max_ms']:.1f})", key="p99")
  if budgets["drops"] is not None and (s["drops"] or 0) > budgets["drops"]:
    fail("drops", f"{s['drops']} dropped frames (perf.js counter: {s['perf_js_drops']}), budget {budgets['drops']}", key="drops")
  res["pass"] = not res["failures"] and not any(e["kind"] in FATAL for e in res["events"])


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("-k", help="only scenarios whose name contains this")
  ap.add_argument("--mode", choices=MODES, default="gpu")
  ap.add_argument("--slow", action="store_true", help="include scenarios marked slow")
  ap.add_argument("--bless", action="store_true", help="store this run's shots as the goldens")
  args = ap.parse_args()

  scenarios = [json.loads(p.read_text()) for p in sorted((HERE / "scenarios").glob("*.json"))]
  scenarios = [s for s in scenarios if (args.k in s["name"] if args.k else args.slow or not s.get("slow"))]
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
        # this machine's network drops out now and then; one retry keeps that from reading as an app bug
        if not r["pass"] and any(NETWORK_FLAKE.search(e["text"]) for e in r["events"]):
          print(f"RETRY {r['name']} (network flake)")
          r = run_scenario(browser, base, sc, latest, token)
          r["retried"] = True
        errors = [f"{f['rule']}: {f['detail'][:70]}" for f in r["failures"]] + [f"{e['kind']}: {e['text'][:70]}" for e in r["events"] if e["kind"] in FATAL]
        print(f"{'PASS' if r['pass'] else 'FAIL'} {r['name']:<16} {r['duration']:5.1f}s  {' | '.join(errors[:2])}{f' (+{len(errors) - 2})' if len(errors) > 2 else ''}")
        results.append(r)
      browser.close()
  finally:
    server.kill()
    if xvfb:
      xvfb.kill()

  (latest / "results.json").write_text(json.dumps({"mode": args.mode, "time": time.time(), "duration": time.monotonic() - t0,
                                                   "load": os.getloadavg()[0], "cpus": os.cpu_count(), "scenarios": results}))
  if args.bless:
    report.bless(latest)
  report.build(latest, prev)
  print(f"{sum(r['pass'] for r in results)}/{len(results)} passed in {time.monotonic() - t0:.1f}s")
  print(f"report: {latest / 'report.html'}\n        {latest / 'report.md'}")
  sys.exit(0 if all(r["pass"] for r in results) else 1)


if __name__ == "__main__":
  main()
