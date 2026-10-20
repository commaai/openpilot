#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["playwright", "pillow", "numpy"]
# ///
"""YCPE QA harness: drives scenarios/*.json in Chrome, records frame timing and UI lint, writes out/latest/report.{html,md}."""
import argparse
import functools
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

from playwright.sync_api import sync_playwright

import report
from procs import free_port, openpilot_venv, wait_port

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
PORTS = range(8170, 8180)
COMMA_API = re.compile(r"^https://(api\.commadotai\.com|api\.comma\.ai|athena\.comma\.ai)/")
# athena, directly or through the daemon's relay (daemon/athena.py)
ATHENA = re.compile(r"^(https://athena\.comma\.ai/|http://localhost:\d+/local/athena/)")
READ_METHODS = ("GET", "HEAD", "OPTIONS")
# athena calls are JSON-RPC POSTs; these only read the device's state, so they pass
ATHENA_READS = {"getNetworkMetered", "getNetworkType", "getNetworks", "getVersion", "getSimInfo", "listUploadQueue", "listDataDirectory"}
# USB ethernet links (devices on this box) come and go, and Chrome fails in-flight requests whenever they do
NETWORK_FLAKE = re.compile(r"ERR_NETWORK_CHANGED|ERR_INTERNET_DISCONNECTED|ERR_NAME_NOT_RESOLVED")
FATAL = ("blocked", "page-error", "console-error")  # events that fail a scenario
RAF_SAMPLE_JS = """(n) => new Promise((r) => {
  const t = [], f = (now) => { t.push(now); t.length > n ? r(t) : requestAnimationFrame(f); };
  requestAnimationFrame(f);
})"""
DEFAULT_BUDGETS = {"loaf_ms": 50, "work_p99_ms": 8, "drops": None, "first_paint_ms": 150}
TRACE_CATEGORIES = ["devtools.timeline", "disabled-by-default-devtools.timeline.frame", "blink", "cc", "gpu", "v8.gc",
                    "disabled-by-default-v8.gc", "viz"]
DEFAULT_VIEWPORT = {"width": 1440, "height": 900}
# gpu: headless Chrome on the NVIDIA GPU through ANGLE/EGL; hardware raster and compositing, 60 Hz begin-frames
# soft: headless SwiftShader; main-thread numbers are fine, raster/composite cost is inflated
# xvfb: headful window on a private Xvfb (llvmpipe GL); real window/input path, software GPU
MODES = {
  "gpu": ["--enable-gpu", "--use-gl=angle", "--use-angle=gl-egl"],
  "soft": [],
  "xvfb": [],
}
# every run has a fresh profile, so the GPU process writes each new shader to its disk cache; on this busy disk a write
# held a frame for 680 ms in a trace. Real users have that cache warm.
CHROME_ARGS = ["--disable-gpu-shader-disk-cache"]
PROBE_JS = (HERE / "probe.js").read_text()
LINT_JS = (HERE / "lint.js").read_text()
SETTLE_JS = "document.fonts.ready.then(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))))"
# noise that never points at a bug: cancelled loads, and Chrome's echo of an HTTP error we already record with its URL
IGNORED_EVENTS = re.compile(r"net::ERR_ABORTED|^Failed to load resource")


def start_server(static=False):
  """serve.py on a free port; with the openpilot venv it also runs the local daemon (/local/*)"""
  venv = None if static else openpilot_venv()
  python = str(venv / "bin" / "python") if venv else sys.executable
  args = [python, str(HERE.parent / "serve.py"), str(port := free_port(PORTS))] + (["--static"] if static else [])
  proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
  try:
    wait_port(port, proc, "serve.py")
  except RuntimeError:
    proc.kill()
    raise
  return proc, f"http://localhost:{port}"


def start_xvfb():
  display = next(f":{n}" for n in range(170, 180) if not Path(f"/tmp/.X11-unix/X{n}").exists())
  proc = subprocess.Popen(["Xvfb", display, "-screen", "0", "1920x1080x24", "-nolisten", "tcp"], stderr=subprocess.DEVNULL)
  time.sleep(0.5)
  return proc, display


def athena_rpc(req):
  """the method of a single JSON-RPC call to athena, else None"""
  if not ATHENA.match(req.url):
    return None
  try:
    body = json.loads(req.post_data or "")
  except ValueError:
    return None
  return body.get("method") if isinstance(body, dict) else None


def top_frame(stack):
  """the innermost local frame of a JS stack, as " at /app.js:115\""""
  m = re.search(r"https?://localhost:\d+(/[^\s)]+?\.js):(\d+)", stack or "")
  return f" at {m.group(1)}:{m.group(2)}" if m else ""


class Scenario:
  """one scenario in a fresh browser context (so storage, OPFS included, starts empty)"""

  def __init__(self, browser, base, sc, out, token, trace=False):
    self.sc, self.out, self.browser, self.trace = sc, out, browser, trace
    self.timeout = sc.get("timeout", 10000)
    self.res = {"name": sc["name"], "url": sc["url"], "failures": [], "events": [], "shots": [], "films": [], "reference": sc.get("reference", [])}
    self.paused = []  # [start, end] page times spent screenshotting and linting, excluded from frame stats
    self.steps = [(0, "load")]  # [page time, step] so a long frame can say what it happened during
    self.shifts_seen = 0
    self.allow = [re.compile(p) for p in sc.get("allow", [])]
    vp = {**DEFAULT_VIEWPORT, **sc.get("viewport", {})}
    self.ctx = browser.new_context(viewport={"width": vp["width"], "height": vp["height"]}, is_mobile=vp.get("mobile", False),
                                   has_touch=vp.get("mobile", False), device_scale_factor=1)
    self.ctx.route(COMMA_API, self.guard)
    self.ctx.route(ATHENA, self.guard)
    for r in sc.get("routes", []):  # registered later, so these answer before the guard
      self.ctx.route(re.compile(r["url"]), functools.partial(self.fulfill, r))
    self.ctx.add_init_script(PROBE_JS)
    if sc.get("auth"):
      if not token:
        self.fail("auth", "no access_token in ~/.comma/auth.json")
      self.ctx.add_init_script(f"localStorage.setItem('comma_jwt', {json.dumps(token or '')})")
    else:  # signed out, even though the daemon offers ~/.comma/auth.json's JWT
      self.ctx.add_init_script("localStorage.setItem('ycpe_daemon_auth_off', '1')")
    page = self.page = self.ctx.new_page()
    page.on("console", lambda m: m.type in ("error", "warning") and self.event(f"console-{m.type}", m.text))
    page.on("pageerror", lambda e: self.event("page-error", f"{e.name}: {e.message}{top_frame(e.stack)}"))
    # the browser's own confirm/alert/prompt ignore the theme: the app asks with dom.js ask() (beforeunload can't)
    page.on("dialog", self.dialog)
    page.on("requestfailed", lambda r: "BLOCKED_BY_CLIENT" not in r.failure and self.event("request-failed", f"{r.failure} {r.method} {r.url}"))
    page.on("response", lambda r: r.status >= 400 and self.event("http-error", f"{r.status} {r.request.method} {r.url}"))
    self.cdp = self.ctx.new_cdp_session(page)
    if sc.get("cpu_throttle"):
      self.cdp.send("Emulation.setCPUThrottlingRate", {"rate": sc["cpu_throttle"]})

  def dialog(self, d):
    """the browser's own confirm/alert/prompt ignore the theme: the app asks with dom.js ask() (beforeunload can't)"""
    if d.type != "beforeunload":
      self.fail("native-dialog", f"{d.type}: {d.message}")
    d.dismiss()

  def fail(self, rule, detail, key=None):
    self.res["failures"].append({"rule": rule, "detail": detail, "key": key or detail})

  def event(self, kind, text):
    # a blocked call always counts: "allow" can't hide one
    if kind == "blocked" or not IGNORED_EVENTS.search(text) and not any(p.search(text) for p in self.allow):
      self.res["events"].append({"kind": "network-flake" if NETWORK_FLAKE.search(text) else kind, "text": text[:400], "shot": len(self.res["shots"])})

  def guard(self, route):
    req = route.request
    rpc = athena_rpc(req)
    if req.method in READ_METHODS or rpc in ATHENA_READS:
      return route.continue_()
    self.event("blocked", f"{req.method} {req.url}" + (f" {rpc}" if rpc else ""))
    route.abort("blockedbyclient")

  def fulfill(self, r, route):
    # the page's origin is localhost, so every answer allows it; a fetch with headers asks with a preflight first
    headers = {"access-control-allow-origin": "*", "access-control-allow-headers": "*"}
    if route.request.method == "OPTIONS":
      return route.fulfill(status=204, headers=headers)
    # a route may answer only one method, or one athena RPC; the rest go on (to the guard)
    if r.get("method", route.request.method) != route.request.method or r.get("rpc", athena_rpc(route.request)) != athena_rpc(route.request):
      return route.fallback()
    body ={"json": r["json"]} if "json" in r else {"path": HERE / r["file"]} if "file" in r else {"body": r.get("body", "")}
    route.fulfill(status=r.get("status", 200), headers=headers, **body)

  def run(self, base):
    t0 = time.monotonic()
    step = {"goto": self.sc["url"]}
    try:
      if self.trace:
        self.browser.start_tracing(page=self.page, categories=TRACE_CATEGORIES)
      self.page.goto(base + self.sc["url"], wait_until="load")
      for step in self.sc.get("steps", []):
        self.steps.append((self.page.evaluate("performance.now()"), json.dumps(step)[:80]))
        self.step(step)
    except Exception as e:
      self.fail("step", f"{json.dumps(step)}: {str(e).splitlines()[0]}", json.dumps(step))
      try:
        self.shot("failure")
      except Exception:
        pass
    self.res["perf"] = self.collect_perf()
    self.res["duration"] = time.monotonic() - t0
    if self.trace:
      self.save_trace()
    self.ctx.close()
    check(self.res, {**DEFAULT_BUDGETS, **self.sc.get("budgets", {})})
    return self.res

  def save_trace(self):
    """the Chrome trace, plus the DOM so trace.py can name the nodes that Layout and Paint events point at"""
    folder = self.out / "traces"
    folder.mkdir(exist_ok=True)
    (folder / f"{self.sc['name']}.dom.json").write_text(json.dumps(self.cdp.send("DOM.getDocument", {"depth": -1, "pierce": True})))
    (folder / f"{self.sc['name']}.json").write_bytes(self.browser.stop_tracing())
    self.res["trace"] = f"traces/{self.sc['name']}.json"

  def point(self, target, timeout):
    if isinstance(target, list):
      return target
    b = self.page.locator(target).first.bounding_box(timeout=timeout)
    if b is None:
      raise RuntimeError(f"{target} has no box")
    return [b["x"] + b["width"] / 2, b["y"] + b["height"] / 2]

  def step(self, step):
    page = self.page
    op = next(k for k in step if k not in ("timeout", "repeat", "rule"))
    arg, timeout = step[op], step.get("timeout", self.timeout)
    if op == "click" and isinstance(arg, str):
      page.locator(arg).first.click(timeout=timeout)
    elif op == "click":
      page.mouse.click(*arg)
    elif op == "hover":
      page.mouse.move(*self.point(arg, timeout))
    elif op == "drag":
      x, y = self.point(arg["from"], timeout)
      tx, ty = self.point(arg["to"], timeout) if "to" in arg else (x + arg["by"][0], y + arg["by"][1])
      page.mouse.move(x, y)
      page.mouse.down()
      page.mouse.move(tx, ty, steps=arg.get("steps", 10))
      page.mouse.up()
    elif op == "dragTo":  # HTML drag and drop, element onto element
      page.locator(arg[0]).first.drag_to(page.locator(arg[1]).first, timeout=timeout)
    elif op == "wheel":
      vp = page.viewport_size
      page.mouse.move(*self.point(arg.get("at", [vp["width"] / 2, vp["height"] / 2]), timeout))
      for _ in range(step.get("repeat", 1)):  # a run of notches, `every` ms apart
        page.mouse.wheel(arg.get("dx", 0), arg.get("dy", 0))
        page.wait_for_timeout(arg.get("every", 0))
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
    elif op == "fill":
      page.locator(arg[0]).first.fill(arg[1], timeout=timeout)
    elif op == "viewport":
      page.set_viewport_size({"width": arg[0], "height": arg[1]})
    elif op == "goto":
      page.goto(urljoin(page.url, arg), wait_until="load")
      self.steps = [(0, f"goto {arg}")]
    elif op == "eval":
      page.evaluate(arg)
    elif op == "note":  # a value worth keeping in the report
      self.res.setdefault("notes", []).append(str(page.evaluate(arg)))
    elif op == "assert":
      # a string result fails and says why; under its rule, if it has one (a perf rule only warns on a busy machine)
      if not (v := page.evaluate(arg)) or isinstance(v, str):
        self.fail(step.get("rule", "assert"), v if isinstance(v, str) else f"false: {arg}", arg)
    elif op == "network":
      # throttles the page and its workers: {down_kbps, up_kbps?, latency_ms?}, or "online"
      on = arg != "online"
      self.cdp.send("Network.emulateNetworkConditions", {"offline": False, "latency": arg.get("latency_ms", 0) if on else 0,
                    "downloadThroughput": arg["down_kbps"] * 125 if on else -1, "uploadThroughput": arg.get("up_kbps", 1000) * 125 if on else -1})
    elif op == "shot":
      self.shot(arg)
    elif op == "filmstrip":
      self.filmstrip(arg, timeout)
    else:
      raise ValueError(f"unknown step {op}")

  def shot(self, name):
    page = self.page
    start = page.evaluate("performance.now()")
    page.evaluate(SETTLE_JS)
    path = self.out / "shots" / self.sc["name"] / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(path), animations="disabled", caret="hide")
    lint = page.evaluate(LINT_JS, {"ignore": ", ".join(self.sc.get("lint_ignore", [])), "shiftsFrom": self.shifts_seen})
    self.shifts_seen = lint["shifts"]
    self.paused.append([start, page.evaluate("performance.now()")])
    self.res["shots"].append({"name": name, "path": str(path.relative_to(self.out)), "issues": lint["issues"], "cls": lint["cls"]})
    # the lint findings that fail the scenario (report.SEVERITY), not just its shot
    for i in lint["issues"]:
      if report.SEVERITY.get(i["rule"]) == report.ERROR:
        self.fail(i["rule"], f"{i['selector']}: {i['detail']}", f"{i['rule']} {i['selector']}")

  def filmstrip(self, arg, timeout):
    """a screenshot every `every` ms until `until` (js) holds, with the boxes of the `crop` selectors for the strip"""
    page, name, frames = self.page, arg.get("name", "film"), []
    folder = self.out / "shots" / self.sc["name"]
    folder.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    while True:
      start = page.evaluate("performance.now()")
      path = folder / f"{name}-{len(frames):03d}.png"
      page.screenshot(path=str(path), caret="hide")
      boxes = [page.locator(sel).first.bounding_box() for sel in arg.get("crop", [])]
      done = page.evaluate(arg["until"])
      self.paused.append([start, page.evaluate("performance.now()")])
      frames.append({"t": time.monotonic() - t0, "path": str(path.relative_to(self.out)), "boxes": [b for b in boxes if b]})
      if done or time.monotonic() - t0 > timeout / 1000:
        break
      time.sleep(max(0, arg.get("every", 250) / 1000 - (time.monotonic() - t0 - frames[-1]["t"])))
    if not done:
      self.fail("filmstrip", f"{name}: {arg['until']} still false after {timeout / 1000:.0f}s", name)
    self.res["films"].append({"name": name, "frames": frames})

  def collect_perf(self):
    page = self.page
    # the page counts as loading until its first contentful paint (or perf.js's first frame, if it never paints)
    probe = page.evaluate("""({raf: __probe.raf, loaf: __probe.loaf,
      paint: performance.getEntriesByName('first-contentful-paint')[0]?.startTime ?? window.__perf?.frames[0]?.t ?? null})""")
    app = page.evaluate("window.__perf ? JSON.parse(JSON.stringify({frames: __perf.frames, longFrames: __perf.longFrames, drops: __perf.drops})) : null")
    harness = lambda a, b: any(a < pe and b > ps for ps, pe in self.paused)  # noqa: E731
    raf = probe["raf"]
    paint = probe["paint"] or 0
    frames = [(b, b - a) for a, b in itertools.pairwise(raf) if a >= paint and not harness(a, b)]
    loaf = [e for e in probe["loaf"] if not harness(e["start"], e["start"] + e["duration"])]
    return {"t": [f[0] for f in frames], "interval": [f[1] for f in frames], "paused": self.paused, "app": app, "steps": self.steps,
            "paint": probe["paint"], "loaf": [e for e in loaf if e["start"] >= paint], "load_loaf": [e for e in loaf if e["start"] < paint]}


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
    "first_paint_ms": perf.get("paint"),
  }
  fail = lambda rule, detail, key: res["failures"].append({"rule": rule, "detail": detail, "key": key})  # noqa: E731
  for e in perf.get("load_loaf", []):
    if e["duration"] > budgets["loaf_ms"]:
      fail("load-frame", f"{e['duration']:.0f} ms frame at {e['start'] / 1000:.1f}s, before the first paint", "load")
  if s["first_paint_ms"] is not None and s["first_paint_ms"] > budgets["first_paint_ms"]:
    fail("first-paint", f"first contentful paint at {s['first_paint_ms']:.0f} ms > {budgets['first_paint_ms']} ms", "first paint")
  for e in perf["loaf"]:
    if e["duration"] > budgets["loaf_ms"]:
      top = sorted(e["scripts"], key=lambda x: -x["duration"])[:3]
      who = ", ".join(f"{x['fn'] or x['invoker']}{'' if x['url'] == x['invoker'] else ' ' + x['url']}:{x['pos']} {x['duration']:.0f} ms" for x in top)
      who = who or "no script, so style/layout/paint"
      during = max((st for st in perf["steps"] if st[0] <= e["start"]), default=(0, "load"))[1]
      fail("long-frame", f"{e['duration']:.0f} ms frame at {e['start'] / 1000:.1f}s (blocking {e['blocking']:.0f} ms) during {during}: {who}",
           " ".join(f"{x['fn'] or x['invoker']} {x['url']}" for x in top) or "no script")
  if s["work_p99_ms"] is not None and s["work_p99_ms"] > budgets["work_p99_ms"]:
    fail("frame-work", f"perf.js work p99 {s['work_p99_ms']:.1f} ms > {budgets['work_p99_ms']} ms (max {s['work_max_ms']:.1f})", "p99")
  if budgets["drops"] is not None and (s["drops"] or 0) > budgets["drops"]:
    fail("drops", f"{s['drops']} dropped frames (perf.js counter: {s['perf_js_drops']}), budget {budgets['drops']}", "drops")
  res["pass"] = passed(res)


def passed(res, perf=True):
  """no failure counts (perf ones only when perf is on; info ones never) and no fatal event"""
  counts = lambda f: (perf or f["rule"] not in report.PERF_RULES) and report.SEVERITY.get(f["rule"]) != report.INFO  # noqa: E731
  return not any(map(counts, res["failures"])) and not any(e["kind"] in FATAL for e in res["events"])


def baseline_jank(browser):
  """frames over 25 ms on an empty page in 1.5 s: a machine that drops frames here can't measure the app"""
  page = browser.new_page()
  t = page.evaluate(RAF_SAMPLE_JS, 90)
  page.close()
  return sum(b - a > 25 for a, b in itertools.pairwise(t))


def run_all(browser, base, scenarios, mode, token, trace=False):
  latest, prev = OUT / "latest", OUT / "prev"
  shutil.rmtree(prev, ignore_errors=True)
  if latest.exists():
    latest.rename(prev)
  latest.mkdir(parents=True)

  t0 = time.monotonic()
  jank, load, results = baseline_jank(browser), [os.getloadavg()[0]], []
  for sc in scenarios:
    r = Scenario(browser, base, sc, latest, token, trace).run(base)
    # this machine's network drops out now and then; one retry keeps that from reading as an app bug
    if not r["pass"] and any(e["kind"] == "network-flake" for e in r["events"]):
      print(f"RETRY {r['name']} (network flake)")
      r = Scenario(browser, base, sc, latest, token, trace).run(base)
      r["retried"] = True
    errors = [f"{f['rule']}: {f['detail'][:70]}" for f in r["failures"]] + [f"{e['kind']}: {e['text'][:70]}" for e in r["events"] if e["kind"] in FATAL]
    more = f" (+{len(errors) - 2})" if len(errors) > 2 else ""
    print(f"{'PASS' if r['pass'] else 'FAIL'} {r['name']:<16} {r['duration']:5.1f}s  {' | '.join(errors[:2])}{more}")
    results.append(r)
    load.append(os.getloadavg()[0])
  jank += baseline_jank(browser)

  # on a saturated machine every page drops frames; then the perf budgets only warn
  noisy = ""
  if jank >= 3:
    noisy = f"an empty page dropped {jank} frames"
  elif max(load) > os.cpu_count():
    noisy = f"load average reached {max(load):.0f} on {os.cpu_count()} cpus"
  if noisy:
    print(f"NOISY: {noisy}, so perf budgets only warn")
    for r in results:
      r["pass"] = passed(r, perf=False)
  (latest / "results.json").write_text(json.dumps({"mode": mode, "duration": time.monotonic() - t0, "noisy": noisy,
                                                   "load": max(load), "cpus": os.cpu_count(), "scenarios": results}))
  return latest, prev, results


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("-k", action="append", help="only scenarios whose name contains this (repeatable)")
  ap.add_argument("--mode", choices=MODES, default="gpu")
  ap.add_argument("--slow", action="store_true", help="include scenarios marked slow")
  ap.add_argument("--bless", action="store_true", help="store this run's shots as the goldens")
  ap.add_argument("--loop", type=int, metavar="N", help="run N times (0: until interrupted), each diffed against the one before")
  ap.add_argument("--trace", action="store_true", help="record a Chrome trace per scenario in out/latest/traces (see trace.py)")
  ap.add_argument("--static", action="store_true", help="serve static files only, without the local daemon")
  args = ap.parse_args()

  scenarios = [json.loads(p.read_text()) for p in sorted((HERE / "scenarios").glob("*.json"))]
  scenarios = [s for s in scenarios if (any(k in s["name"] for k in args.k) if args.k else args.slow or not s.get("slow"))]
  auth_file = Path.home() / ".comma" / "auth.json"
  token = json.loads(auth_file.read_text()).get("access_token") if auth_file.exists() else None

  server, base = start_server(args.static)
  xvfb, env = None, None
  if args.mode == "xvfb":
    xvfb, display = start_xvfb()
    env = {"DISPLAY": display}
  ok = True
  try:
    with sync_playwright() as p:
      browser = p.chromium.launch(executable_path="/usr/bin/google-chrome", headless=args.mode != "xvfb", args=MODES[args.mode] + CHROME_ARGS, env=env)
      runs = itertools.count(1) if args.loop == 0 else range(1, (args.loop or 1) + 1)
      for n in runs:
        t0 = time.monotonic()
        latest, prev, results = run_all(browser, base, scenarios, args.mode, token, args.trace)
        if args.bless:
          report.bless(latest)
        diff = report.build(latest, prev)
        ok = all(r["pass"] for r in results)
        found = len(diff["new"]) + len(diff["still"])
        print(f"run {n}: {sum(r['pass'] for r in results)}/{len(results)} passed in {time.monotonic() - t0:.1f}s · " +
              f"{found} findings: {len(diff['new'])} new, {len(diff['fixed'])} fixed")
        print(f"report: {latest / 'report.html'}\n        {latest / 'report.md'}")
  except KeyboardInterrupt:
    pass
  finally:
    server.kill()
    if xvfb:
      xvfb.kill()
  sys.exit(0 if ok else 1)


if __name__ == "__main__":
  main()
