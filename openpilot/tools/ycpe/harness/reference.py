#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["playwright", "pillow", "numpy"]
# ///
"""Reference screenshots shown next to the matching scenario shots: connect.comma.ai (read-only), and native jotpluggler and
cabana on a private Xvfb. Writes out/reference/<name>.png; a scenario lists the names it compares against in "reference".

The native tools come from a built checkout (default: this repo). To build them, from the repo root:
  scons -j$(nproc) openpilot/tools/cabana/cabana openpilot/tools/jotpluggler/jotpluggler
They need the openpilot venv (found at <root>/.venv or the main checkout's .venv) to download route files.
Neither tool takes a start time: jotpluggler renders the route start, cabana is captured after --wait seconds of its demo.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

from procs import REPO, openpilot_venv
from run import COMMA_API, READ_METHODS, start_xvfb

HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / "reference"
DEMO = "5beb9b58bd12b691/0000010a--a51155e496"
W, H = 1440, 900
REFS = {
  "connect-home": ("connect", {"url": "/"}),
  "connect-route-demo": ("connect", {"url": f"/{DEMO}"}),
  "connect-route-demo-mobile": ("connect", {"url": f"/{DEMO}", "viewport": {"width": 390, "height": 844}}),
  "jotpluggler-demo": ("jotpluggler", {"args": ["--demo"]}),
  "jotpluggler-longitudinal": ("jotpluggler", {"args": ["--demo", "--layout", "longitudinal"]}),
  "cabana-demo": ("cabana", {"args": ["--demo"]}),
}


def connect(name, url, viewport=None):
  with sync_playwright() as p:
    browser = p.chromium.launch(executable_path="/usr/bin/google-chrome")
    ctx = browser.new_context(viewport=viewport or {"width": W, "height": H})
    ctx.route(COMMA_API, lambda r: r.continue_() if r.request.method in READ_METHODS else r.abort("blockedbyclient"))
    page = ctx.new_page()
    page.goto(f"https://connect.comma.ai{url}", wait_until="networkidle", timeout=60000)
    page.wait_for_timeout(2000)
    page.screenshot(path=str(OUT / f"{name}.png"))
    browser.close()


def native_env(root, display):
  venv = openpilot_venv(root) or openpilot_venv()
  path = f"{venv / 'bin'}:{os.environ['PATH']}" if venv else os.environ["PATH"]
  return {**os.environ, "DISPLAY": display, "PYTHONPATH": str(root), "PATH": path, "PWD": str(root)}


def jotpluggler(name, root, display, args):
  exe = root / "openpilot/tools/jotpluggler/jotpluggler"
  subprocess.run([str(exe), *args, "--width", str(W), "--height", str(H), "--output", str(OUT / f"{name}.png")],
                 cwd=root, env=native_env(root, display), check=True, timeout=600)


def cabana(name, root, display, args, wait):
  exe = root / "openpilot/tools/cabana/cabana"
  env = native_env(root, display)
  proc = subprocess.Popen([str(exe), *args], cwd=root, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
  try:
    win = subprocess.run(["xdotool", "search", "--sync", "--onlyvisible", "--pid", str(proc.pid)], env=env, capture_output=True,
                         text=True, timeout=60).stdout.split()[0]
    subprocess.run(["xdotool", "windowmove", win, "0", "0", "windowsize", win, str(W), str(H)], env=env, check=True)
    time.sleep(wait)
    subprocess.run(["import", "-display", display, "-window", "root", "-crop", f"{W}x{H}+0+0", str(OUT / f"{name}.png")], check=True)
  finally:
    proc.kill()


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("names", nargs="*", help=f"default: all of {', '.join(REFS)}")
  ap.add_argument("--native-root", type=Path, default=REPO, help="checkout with built cabana and jotpluggler")
  ap.add_argument("--wait", type=float, default=30, help="seconds of cabana demo before the capture")
  args = ap.parse_args()
  OUT.mkdir(parents=True, exist_ok=True)

  xvfb, display = start_xvfb()
  ok = True
  try:
    for name in args.names or REFS:
      kind, opts = REFS[name]
      t0 = time.monotonic()
      try:
        if kind == "connect":
          connect(name, **opts)
        elif kind == "jotpluggler":
          jotpluggler(name, args.native_root.resolve(), display, opts["args"])
        else:
          cabana(name, args.native_root.resolve(), display, opts["args"], args.wait)
        print(f"ok   {name:<28} {time.monotonic() - t0:5.1f}s  {OUT / name}.png")
      except Exception as e:
        ok = False
        print(f"FAIL {name:<28} {str(e).splitlines()[0]}")
  finally:
    xvfb.kill()
  sys.exit(0 if ok else 1)


if __name__ == "__main__":
  main()
