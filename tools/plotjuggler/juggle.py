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

from opendbc.car.fingerprints import MIGRATION
from openpilot.common.basedir import BASEDIR
from openpilot.common.swaglog import cloudlog
from openpilot.tools.cabana.dbc.generate_dbc_json import generate_dbc_dict
from openpilot.tools.lib.logreader import LogReader, ReadMode, save_log
from openpilot.selfdrive.test.process_replay.migration import migrate_all

juggle_dir = os.path.dirname(os.path.realpath(__file__))

os.environ['LD_LIBRARY_PATH'] = os.environ.get('LD_LIBRARY_PATH', '') + f":{juggle_dir}/bin/"

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


def start_juggler(fn=None, dbc=None, layout=None, route_or_segment_name=None, platform=None):
  env = os.environ.copy()
  env["BASEDIR"] = BASEDIR
  env["PATH"] = f"{INSTALL_DIR}:{os.getenv('PATH', '')}"
  if dbc:
    if os.path.exists(dbc):
      dbc = os.path.abspath(dbc)
    env["DBC_NAME"] = dbc

  extra_args = ""
  if fn is not None:
    extra_args += f" -d {os.path.abspath(fn)}"
  if layout is not None:
    extra_args += f" -l {os.path.abspath(layout)}"
  if route_or_segment_name is not None:
    extra_args += f" --window_title \"{route_or_segment_name}{f' ({platform})' if platform is not None else ''}\""

  cmd = f'{PLOTJUGGLER_BIN} --buffer_size {MAX_STREAMING_BUFFER_SIZE} --plugin_folders {INSTALL_DIR}{extra_args}'
  subprocess.call(cmd, shell=True, env=env, cwd=juggle_dir)


def process(can, lr):
  return [d for d in lr if can or d.which() not in ['can', 'sendcan'] and not d.which().startswith('customReserved')]


def juggle_route(route_or_segment_name, can, layout, dbc, should_migrate):
  lr = LogReader(route_or_segment_name, default_mode=ReadMode.AUTO_INTERACTIVE)

  all_data = lr.run_across_segments(24, partial(process, can))
  if should_migrate:
    all_data = migrate_all(all_data)

  # Infer DBC name from logs
  platform = None
  if dbc is None:
    try:
      CP = lr.first('carParams')
      platform = MIGRATION.get(CP.carFingerprint, CP.carFingerprint)
      dbc = generate_dbc_dict()[platform]
    except Exception:
      cloudlog.exception("Failed to get DBC name from logs!")

  with tempfile.NamedTemporaryFile(suffix='.rlog', dir=juggle_dir) as tmp:
    save_log(tmp.name, all_data, compress=False)
    del all_data
    start_juggler(tmp.name, dbc, layout, route_or_segment_name, platform)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="A helper to run PlotJuggler on openpilot routes",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("--demo", action="store_true", help="Use the demo route instead of providing one")
  parser.add_argument("--can", action="store_true", help="Parse CAN data")
  parser.add_argument("--stream", action="store_true", help="Start PlotJuggler in streaming mode")
  parser.add_argument("--no-migration", action="store_true", help="Do not perform log migration")
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
    juggle_route(route_or_segment_name, args.can, args.layout, args.dbc, not args.no_migration)
