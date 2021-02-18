#!/usr/bin/env python3

import sys
import multiprocessing
import subprocess
from tempfile import NamedTemporaryFile
from tools.lib.route import Route
from tools.lib.logreader import LogReader
from selfdrive.test.process_replay.compare_logs import save_log
from tools.lib.url_file import URLFile



def load_segment(segment_name):
  print(f"Loading {segment_name}")
  lr = LogReader(segment_name)
  r = [d for d in lr if d.which() not in ['can', 'sendcan']]
  print(f"done {segment_name}")
  return r

def juggle_file(fn):
  subprocess.call(f"bin/plotjuggler -d {fn}", shell=True)
  # subprocess.call(f"/home/batman/PlotJuggler/build/bin/plotjuggler -d {fn}", shell=True)

def juggle_route(route_name):
  r = Route(route_name)
  all_data = []

  pool = multiprocessing.Pool(24)

  all_data = []
  for d in pool.map(load_segment, r.log_paths()):
    all_data += d

  tempfile = NamedTemporaryFile(suffix='.rlog')
  save_log(tempfile.name, all_data, compress=False)
  del all_data

  juggle_file(tempfile.name)

def juggle_segment(route_name, segment_nr):
  r = Route(route_name)
  lp = r.log_paths()[segment_nr]

  if lp is None:
    print("This segment does not exist, please try a different one")
    return

  uf = URLFile(lp)
  juggle_file(uf.name)


if __name__ == "__main__":
  if len(sys.argv) == 2:
    juggle_route(sys.argv[1])
  elif len(sys.argv) == 3:
    juggle_segment(sys.argv[1], int(sys.argv[2]))
