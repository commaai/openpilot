#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["playwright"]
# ///
"""Live data and device logs over WebRTC, end to end (SPEC.md Live): this checkout's webrtcd on port+1000 with an
OPENPILOT_PREFIX of its own, dev/webrtc_device.py publishing into it and serving a synthetic route from a log root under
/tmp, a serve.py of its own, and dev/webrtc.html in Chrome. Athena is intercepted, never reached: the page's
startStream goes to the local webrtcd. Prints latency (publish → on screen), frame times and read throughput."""
import argparse
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
DONGLE, LOG_ID = "0123456789abcdef", "00000001--0123456789"
ROUTE = f"{DONGLE}|{LOG_ID}"
ROUTE_URL = f"{DONGLE}(?:\\||%7C){LOG_ID}"  # as the page puts it in API URLs


def venv_python():
  common = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--path-format=absolute", "--git-common-dir"], capture_output=True, text=True)
  venv = next(v for v in (REPO / ".venv", Path(common.stdout.strip()).parent / ".venv") if v.exists())
  return str(venv / "bin" / "python")


def free_port(ports):
  for port in ports:
    with socket.socket() as a, socket.socket() as b:
      if a.connect_ex(("127.0.0.1", port)) and b.connect_ex(("127.0.0.1", port + 1000)):
        return port
  raise RuntimeError("no free port")


def wait_port(port, proc, what):
  for _ in range(100):
    if proc.poll() is not None:
      raise RuntimeError(f"{what} exited")
    with socket.socket() as s:
      if not s.connect_ex(("127.0.0.1", port)):
        return
    time.sleep(0.2)
  raise RuntimeError(f"{what} isn't listening on {port}")


def stats(xs):
  xs = sorted(xs)
  return f"n={len(xs)} p50={xs[len(xs) // 2]:.1f} p90={xs[int(len(xs) * .9)]:.1f} max={xs[-1]:.1f}" if xs else "none"


class Run:
  def __init__(self, port, webrtcd):
    self.port, self.webrtcd, self.athena_calls, self.stock = port, webrtcd, [], False

  # athenad's startStream on a device: the page's params as webrtcd's request, or a stock device's answer
  def athena(self, route):
    call = json.loads(route.request.post_data)
    self.athena_calls.append(call["method"])
    if call["method"] != "startStream":
      return route.fulfill(status=403, json={"error": "only startStream is stood in for"})
    p = call["params"]
    if self.stock and set(p) - {"sdp", "enabled"}:
      return route.fulfill(json={"id": call["id"], "jsonrpc": "2.0", "error": {"code": -32602, "message": "Invalid params",
                                 "data": {"message": "startStream() got an unexpected keyword argument 'services'"}}})
    body = ({"sdp": p["sdp"], "cameras": ["wideRoad"], "enabled": p["enabled"], "bridge_services_out": ["carState", "deviceState"]} if self.stock
            else {"sdp": p["sdp"], "cameras": p["cameras"], "enabled": p["enabled"], "services": p["services"], "files": p["files"]})
    req = urllib.request.Request(f"http://127.0.0.1:{self.webrtcd}/stream", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=40) as r:
      route.fulfill(json={"id": call["id"], "jsonrpc": "2.0", "result": json.load(r)})

  def page(self, browser):
    page = browser.new_page()
    page.on("pageerror", lambda e: print("  page error:", e))
    page.on("console", lambda m: m.type in ("error", "warning") and print("  console:", m.text))
    # dev pages don't carry the daemon's meta tag: give this one the app's
    meta = next(line for line in page.request.get(f"http://localhost:{self.port}/").text().splitlines() if "ycpe-daemon" in line)
    page.route(re.compile(r"/dev/webrtc\.html"), lambda r: r.fulfill(response=(f := r.fetch()), body=f.text().replace("<head>", "<head>" + meta, 1)))
    page.route(re.compile(r"^https://athena\.comma\.ai/|/local/athena/"), self.athena)
    # the synthetic route's API records, and its device online
    now = int(time.time() * 1000)  # noqa: TID251  # API records are wall times
    page.route(f"**/v1.1/devices/{DONGLE}/", lambda r: r.fulfill(json={"dongle_id": DONGLE, "last_athena_ping": now // 1000}))
    page.route(f"**/v1/devices/{DONGLE}/routes_segments*", lambda r: r.fulfill(json=[]))
    page.route(re.compile(f"/v1/route/{ROUTE_URL}/files"), lambda r: r.fulfill(json={}))
    page.route(re.compile(f"/v1/route/{ROUTE_URL}/$"), lambda r: r.fulfill(json={
      "fullname": ROUTE, "dongle_id": DONGLE, "maxqlog": 1, "start_time_utc_millis": now, "end_time_utc_millis": now + 120000}))
    return page

  def live(self, browser, wall, query, secs=8):
    page = self.page(browser)
    page.goto(f"http://localhost:{self.port}/dev/webrtc.html?{query}&wall={wall}")
    page.wait_for_function("live.states.some(([, s]) => s === 'live')", timeout=60000)
    page.wait_for_timeout(secs * 1000)
    live = page.evaluate("({...live, newest: String(live.newest)})")
    work = page.evaluate("__perf.frames.map((f) => f.work)")
    page.close()
    return live, work


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--port", type=int, help="serve.py's port (webrtcd's is 1000 more); default: a free one from 8560")
  ap.add_argument("--webrtcd", help="a webrtcd.py to run instead of this checkout's")
  args = ap.parse_args()
  sys.stdout.reconfigure(line_buffering=True)
  signal.signal(signal.SIGTERM, lambda *_: sys.exit("terminated"))  # still stops what it started
  port = args.port or free_port(range(8560, 8570))
  webrtcd, prefix, python = port + 1000, f"ycpe-webrtc-test-{port}", venv_python()
  tmp = Path(tempfile.mkdtemp(prefix="ycpe-webrtc-"))
  env = {**os.environ, "OPENPILOT_PREFIX": prefix, "LOG_ROOT": str(tmp / "realdata"),
         "PYTHONPATH": os.pathsep.join(str(REPO / d) for d in ("", "msgq_repo", "opendbc_repo"))}
  Path(f"/dev/shm/msgq_{prefix}").mkdir(exist_ok=True)
  procs = []
  try:
    webrtcd_cmd = [args.webrtcd] if args.webrtcd else ["-m", "openpilot.system.webrtc.webrtcd"]
    procs.append(subprocess.Popen([python, *webrtcd_cmd, "--port", str(webrtcd)], env=env, cwd=REPO,
                                  stdout=subprocess.DEVNULL, stderr=open(tmp / "webrtcd.log", "w")))
    procs.append(subprocess.Popen([python, str(HERE.parent / "serve.py"), str(port), "--webrtcd-port", str(webrtcd)], cwd=REPO,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    device = subprocess.Popen([python, str(HERE / "webrtc_device.py"), "--log-root", env["LOG_ROOT"], "--log-id", LOG_ID, "--seconds", "300"],
                              env=env, stdout=subprocess.PIPE, text=True)
    procs.append(device)
    wait_port(webrtcd, procs[0], "webrtcd")
    wait_port(port, procs[1], "serve.py")
    video = int(device.stdout.readline().split()[1])
    wall = device.stdout.readline().split()[1]
    run, failed = Run(port, webrtcd), []

    with sync_playwright() as p:
      browser = p.chromium.launch(executable_path="/usr/bin/google-chrome", args=["--autoplay-policy=no-user-gesture-required"])

      def check(name, query, stock=False):
        run.stock = stock
        live, work = run.live(browser, wall, query)
        got = live["json"] if stock else live["events"]
        print(f"{name}: live after {next(t for t, s, _ in live['states'] if s == 'live')} ms, {got} {'JSON messages' if stock else 'events'}, " +
              f"video frames {({k: v['n'] for k, v in live['frames'].items()})}")
        print(f"  publish → on screen ms: {stats(live['lat'])}; frame work ms: {stats(work)}")
        if not got or (video and not any(v["n"] for v in live["frames"].values())):
          failed.append(name)

      check("this PC", "local=1&cameras=road")
      check("a device through athena", f"dongle={DONGLE}&cameras=road")

      # a route's rlogs from its device: resolved peer to peer, fetched by the store's workers through the service worker
      page = run.page(browser)
      calls = len(run.athena_calls)
      page.goto(f"http://localhost:{port}/dev/webrtc.html?files={ROUTE.replace('|', '%7C')}")
      page.wait_for_function("live.files?.segments.length >= 2", timeout=60000)
      print(f"device logs: segments loaded at {page.evaluate('live.files.segments')} ms, {len(run.athena_calls) - calls} startStream")
      raw = page.evaluate(f"""(async () => {{
        const {{peer}} = await import("/data/live/peer.js"), ch = await peer("{DONGLE}").filesChannel(), t = performance.now();
        const n = (await ch.bytes("{LOG_ID}--0/fcamera.hevc")).length;
        return [n, performance.now() - t]; }})()""")
      print(f"  files channel read: {raw[0] / 1e6:.1f} MB in {raw[1]:.0f} ms ({raw[0] / raw[1] / 1e3:.1f} MB/s)")
      calls = len(run.athena_calls)
      page.reload()
      page.wait_for_function("live.files?.segments.length >= 2", timeout=60000)
      print(f"  revisit: {len(run.athena_calls) - calls} startStream (from OPFS)")
      if len(run.athena_calls) != calls:
        failed.append("revisit")
      page.close()

      # last: a stock device's webrtcd answers with the wide road camera and carState/deviceState as JSON
      check("a stock device", f"dongle={DONGLE}&cameras=road", stock=True)
      browser.close()
  except Exception:
    print((tmp / "webrtcd.log").read_text()[-3000:])
    raise
  finally:
    for proc in procs:
      proc.terminate()
    for proc in procs:
      try:
        proc.wait(5)
      except subprocess.TimeoutExpired:  # a deadlocked webrtcd doesn't get to its SIGTERM handler
        proc.kill()
        proc.wait()
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(f"/dev/shm/msgq_{prefix}", ignore_errors=True)
  if failed:
    sys.exit(f"failed: {', '.join(failed)}")


if __name__ == "__main__":
  main()
