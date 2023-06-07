#!/usr/bin/env python3
import os
import sys
import multiprocessing
import platform
import shutil
import subprocess
import tarfile
import tempfile
import requests
import argparse

from common.basedir import BASEDIR
from selfdrive.test.openpilotci import get_url
from tools.lib.logreader import LogReader
from tools.lib.route import Route, SegmentName
from tools.lib.helpers import save_log
from urllib.parse import urlparse, parse_qs

juggle_dir = os.path.dirname(os.path.realpath(__file__))

DEMO_ROUTE = "4cf7a6ad03080c90|2021-09-29--13-46-36"
RELEASES_URL="https://github.com/commaai/PlotJuggler/releases/download/latest"
INSTALL_DIR = os.path.join(juggle_dir, "bin")
PLOTJUGGLER_BIN = os.path.join(juggle_dir, "bin/plotjuggler")
MINIMUM_PLOTJUGGLER_VERSION = (3, 5, 2)
MAX_STREAMING_BUFFER_SIZE = 1000

def install():
  m = f"{platform.system()}-{platform.machine()}"
  supported = ("Linux-x86_64", "Darwin-arm64", "Darwin-x86_64")
  if m not in supported:
    raise Exception(f"Unsupported platform: '{m}'. Supported platforms: {supported}")

  if os.path.exists(INSTALL_DIR):
    shutil.rmtree(INSTALL_DIR)
  os.mkdir(INSTALL_DIR)

  url = os.path.join(RELEASES_URL, m + ".tar.gz")
  with requests.get(url, stream=True, timeout=10) as r, tempfile.NamedTemporaryFile() as tmp:
    r.raise_for_status()
    with open(tmp.name, 'wb') as tmpf:
      for chunk in r.iter_content(chunk_size=1024*1024):
        tmpf.write(chunk)

    with tarfile.open(tmp.name) as tar:
      tar.extractall(path=INSTALL_DIR)


def get_plotjuggler_version():
  out = subprocess.check_output([PLOTJUGGLER_BIN, "-v"], encoding="utf-8").strip()
  version = out.split(" ")[1]
  return tuple(map(int, version.split(".")))


def load_segment(segment_name):
  if segment_name is None:
    return []

  try:
    return list(LogReader(segment_name))
  except (AssertionError, ValueError) as e:
    print(f"Error parsing {segment_name}: {e}")
    return []


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


def juggle_route(route_or_segment_name, segment_count, qlog, can, layout, dbc=None, ci=False):
  segment_start = 0
  if 'cabana' in route_or_segment_name:
    query = parse_qs(urlparse(route_or_segment_name).query)
    route_or_segment_name = query["route"][0]

  if route_or_segment_name.startswith(("http://", "https://", "cd:/")) or os.path.isfile(route_or_segment_name):
    logs = [route_or_segment_name]
  elif ci:
    route_or_segment_name = SegmentName(route_or_segment_name, allow_route_name=True)
    route = route_or_segment_name.route_name.canonical_name
    segment_start = max(route_or_segment_name.segment_num, 0)
    logs = [get_url(route, i) for i in range(100)]  # Assume there not more than 100 segments
  else:
    route_or_segment_name = SegmentName(route_or_segment_name, allow_route_name=True)
    segment_start = max(route_or_segment_name.segment_num, 0)

    if route_or_segment_name.segment_num != -1 and segment_count is None:
      segment_count = 1

    r = Route(route_or_segment_name.route_name.canonical_name, route_or_segment_name.data_dir)
    logs = r.qlog_paths() if qlog else r.log_paths()

  segment_end = segment_start + segment_count if segment_count else None
  logs = logs[segment_start:segment_end]

  if None in logs:
    resp = input(f"{logs.count(None)}/{len(logs)} of the rlogs in this segment are missing, would you like to fall back to the qlogs? (y/n) ")
    if resp == 'y':
      logs = r.qlog_paths()[segment_start:segment_end]
    else:
      print("Please try a different route or segment")
      return

  all_data = []
  with multiprocessing.Pool(24) as pool:
    for d in pool.map(load_segment, logs):
      all_data += d

  if not can:
    all_data = [d for d in all_data if d.which() not in ['can', 'sendcan']]

  # Infer DBC name from logs
  if dbc is None:
    for cp in [m for m in all_data if m.which() == 'carParams']:
      try:
        DBC = __import__(f"selfdrive.car.{cp.carParams.carName}.values", fromlist=['DBC']).DBC
        dbc = DBC[cp.carParams.carFingerprint]['pt']
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
  parser.add_argument("--qlog", action="store_true", help="Use qlogs")
  parser.add_argument("--ci", action="store_true", help="Download data from openpilot CI bucket")
  parser.add_argument("--can", action="store_true", help="Parse CAN data")
  parser.add_argument("--stream", action="store_true", help="Start PlotJuggler in streaming mode")
  parser.add_argument("--layout", nargs='?', help="Run PlotJuggler with a pre-defined layout")
  parser.add_argument("--install", action="store_true", help="Install or update PlotJuggler + plugins")
  parser.add_argument("--dbc", help="Set the DBC name to load for parsing CAN data. If not set, the DBC will be automatically inferred from the logs.")
  parser.add_argument("route_or_segment_name", nargs='?', help="The route or segment name to plot (cabana share URL accepted)")
  parser.add_argument("segment_count", type=int, nargs='?', help="The number of segments to plot")

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
    juggle_route(route_or_segment_name, args.segment_count, args.qlog, args.can, args.layout, args.dbc, args.ci)
