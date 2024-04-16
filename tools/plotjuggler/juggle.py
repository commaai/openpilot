#!/usr/bin/env python3
import os
import sys
import platform
import shutil
import subprocess
import tarfile
import tempfile
import requests
import argparse
from functools import partial

from openpilot.common.basedir import BASEDIR
from openpilot.selfdrive.car.fingerprints import MIGRATION
from openpilot.tools.lib.helpers import save_log
from openpilot.tools.lib.logreader import LogReader, ReadMode

juggle_dir = os.path.dirname(os.path.realpath(__file__))

DEMO_ROUTE = "a2a0ccea32023010|2023-07-27--13-01-19"
RELEASES_URL = "https://github.com/commaai/PlotJuggler/releases/download/latest"
INSTALL_DIR = os.path.join(juggle_dir, "bin")
PLOTJUGGLER_BIN = os.path.join(juggle_dir, "bin/plotjuggler")
MINIMUM_PLOTJUGGLER_VERSION = (3, 5, 2)
MAX_STREAMING_BUFFER_SIZE = 1000


def install():
  m = f"{platform.system()}-{platform.machine()}"
  supported = ("Linux-x86_64", "Linux-aarch64", "Darwin-arm64", "Darwin-x86_64")
  if m not in supported:
    raise Exception(f"Unsupported platform: '{m}'. Supported platforms: {supported}")

  if os.path.exists(INSTALL_DIR):
    shutil.rmtree(INSTALL_DIR)
  os.mkdir(INSTALL_DIR)

  url = os.path.join(RELEASES_URL, m + ".tar.gz")
  with requests.get(url, stream=True, timeout=10) as r, tempfile.NamedTemporaryFile() as tmp:
    r.raise_for_status()
    with open(tmp.name, 'wb') as tmpf:
      for chunk in r.iter_content(chunk_size=1024 * 1024):
        tmpf.write(chunk)

    with tarfile.open(tmp.name) as tar:
      tar.extractall(path=INSTALL_DIR)


def get_plotjuggler_version():
  out = subprocess.check_output([PLOTJUGGLER_BIN, "-v"], encoding="utf-8").strip()
  version = out.split(" ")[1]
  return tuple(map(int, version.split(".")))


def start_juggler(fn=None, dbc=None, layout=None, route_or_segment_name=None):
  env = os.environ.copy()
  env["BASEDIR"] = BASEDIR
  env["PATH"] = f"{INSTALL_DIR}:{os.getenv('PATH', '')}"
  if dbc:
    env["DBC_NAME"] = dbc

  extra_args = ""
  if fn is not None:
    extra_args += f" -d {fn}"
  if layout is not None:
    extra_args += f" -l {layout}"
  if route_or_segment_name is not None:
    extra_args += f" --window_title \"{route_or_segment_name}\""

  cmd = f'{PLOTJUGGLER_BIN} --buffer_size {MAX_STREAMING_BUFFER_SIZE} --plugin_folders {INSTALL_DIR}{extra_args}'
  subprocess.call(cmd, shell=True, env=env, cwd=juggle_dir)


def process(can, lr):
  return [d for d in lr if can or d.which() not in ['can', 'sendcan']]


def juggle_route(route_or_segment_name, can, layout, dbc=None):
  sr = LogReader(route_or_segment_name, default_mode=ReadMode.AUTO_INTERACTIVE)

  all_data = sr.run_across_segments(24, partial(process, can))

  # Infer DBC name from logs
  if dbc is None:
    for cp in [m for m in all_data if m.which() == 'carParams']:
      try:
        DBC = __import__(f"openpilot.selfdrive.car.{cp.carParams.carName}.values", fromlist=['DBC']).DBC
        fingerprint = cp.carParams.carFingerprint
        dbc = DBC[MIGRATION.get(fingerprint, fingerprint)]['pt']
      except Exception:
        pass
      break

  with tempfile.NamedTemporaryFile(suffix='.rlog', dir=juggle_dir) as tmp:
    save_log(tmp.name, all_data, compress=False)
    del all_data
    start_juggler(tmp.name, dbc, layout, route_or_segment_name)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="A helper to run PlotJuggler on openpilot routes",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("--demo", action="store_true", help="Use the demo route instead of providing one")
  parser.add_argument("--can", action="store_true", help="Parse CAN data")
  parser.add_argument("--stream", action="store_true", help="Start PlotJuggler in streaming mode")
  parser.add_argument("--layout", nargs='?', help="Run PlotJuggler with a pre-defined layout")
  parser.add_argument("--install", action="store_true", help="Install or update PlotJuggler + plugins")
  parser.add_argument("--dbc", help="Set the DBC name to load for parsing CAN data. If not set, the DBC will be automatically inferred from the logs.")
  parser.add_argument("route_or_segment_name", nargs='?', help="The route or segment name to plot (cabana share URL accepted)")

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit()
  args = parser.parse_args()

  if args.install:
    install()
    sys.exit()

  if not os.path.exists(PLOTJUGGLER_BIN):
    print("PlotJuggler is missing. Downloading...")
    install()

  if get_plotjuggler_version() < MINIMUM_PLOTJUGGLER_VERSION:
    print("PlotJuggler is out of date. Installing update...")
    install()

  if args.stream:
    start_juggler(layout=args.layout)
  else:
    route_or_segment_name = DEMO_ROUTE if args.demo else args.route_or_segment_name.strip()
    juggle_route(route_or_segment_name, args.can, args.layout, args.dbc)
