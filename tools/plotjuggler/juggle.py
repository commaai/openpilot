#!/usr/bin/env python3
import os
import sys
import multiprocessing
import subprocess
import argparse
from tempfile import NamedTemporaryFile

from common.basedir import BASEDIR
from selfdrive.test.process_replay.compare_logs import save_log
from tools.lib.route import Route
from tools.lib.logreader import LogReader

juggle_dir = os.path.dirname(os.path.realpath(__file__))

def load_segment(segment_name):
  print(f"Loading {segment_name}")
  if segment_name is None:
    return []

  try:
    return list(LogReader(segment_name))
  except ValueError as e:
    print(f"Error parsing {segment_name}: {e}")
    return []

def start_juggler(fn=None, dbc=None, layout=None):
  env = os.environ.copy()
  env["BASEDIR"] = BASEDIR
  pj = os.getenv("PLOTJUGGLER_PATH", os.path.join(juggle_dir, "bin/plotjuggler"))

  if dbc:
    env["DBC_NAME"] = dbc

  extra_args = []
  if fn is not None:
    extra_args.append(f'-d {fn}')

  if layout is not None:
    extra_args.append(f'-l {layout}')

  extra_args = " ".join(extra_args)
  subprocess.call(f'{pj} --plugin_folders {os.path.join(juggle_dir, "bin")} {extra_args}', shell=True, env=env, cwd=juggle_dir)

def juggle_route(route_name, segment_number, qlog, can, layout):
  if route_name.startswith("http://") or route_name.startswith("https://") or os.path.isfile(route_name):
    logs = [route_name]
  else:
    r = Route(route_name)
    logs = r.qlog_paths() if qlog else r.log_paths()

  if segment_number is not None:
    logs = logs[segment_number:segment_number+1]

  if None in logs:
    fallback_answer = input("At least one of the rlogs in this segment does not exist, would you like to use the qlogs? (y/n) : ")
    if fallback_answer == 'y':
      logs = r.qlog_paths()
      if segment_number is not None:
        logs = logs[segment_number:segment_number+1]
    else:
      print(f"Please try a different {'segment' if segment_number is not None else 'route'}")
      return

  all_data = []
  pool = multiprocessing.Pool(24)
  for d in pool.map(load_segment, logs):
    all_data += d

  if not can:
    all_data = [d for d in all_data if d.which() not in ['can', 'sendcan']]

  # Infer DBC name from logs
  dbc = None
  for cp in [m for m in all_data if m.which() == 'carParams']:
    try:
      DBC = __import__(f"selfdrive.car.{cp.carParams.carName}.values", fromlist=['DBC']).DBC
      dbc = DBC[cp.carParams.carFingerprint]['pt']
    except (ImportError, KeyError, AttributeError):
      pass
    break

  tempfile = NamedTemporaryFile(suffix='.rlog', dir=juggle_dir)
  save_log(tempfile.name, all_data, compress=False)
  del all_data

  start_juggler(tempfile.name, dbc, layout)

def get_arg_parser():
  parser = argparse.ArgumentParser(description="PlotJuggler plugin for reading rlogs",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("--qlog", action="store_true", help="Use qlogs")
  parser.add_argument("--can", action="store_true", help="Parse CAN data")
  parser.add_argument("--stream", action="store_true", help="Start PlotJuggler without a route to stream data using Cereal")
  parser.add_argument("--layout", nargs='?', help="Run PlotJuggler with a pre-defined layout")
  parser.add_argument("route_name", nargs='?', help="The name of the route that will be plotted.")
  parser.add_argument("segment_number", type=int, nargs='?', help="The index of the segment that will be plotted")
  return parser

if __name__ == "__main__":
  arg_parser = get_arg_parser()
  if len(sys.argv) == 1:
    arg_parser.print_help()
    sys.exit()
  args = arg_parser.parse_args(sys.argv[1:])

  if args.stream:
    start_juggler()
  else:
    juggle_route(args.route_name, args.segment_number, args.qlog, args.can, args.layout)
