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
from tools.lib.url_file import URLFile


def load_segment(segment_name):
  print(f"Loading {segment_name}")
  lr = LogReader(segment_name)
  r = [d for d in lr if d.which() not in ['can', 'sendcan']]
  print(f"done {segment_name}")
  return r

def juggle_file(fn):
  env = os.environ.copy()
  env["BASEDIR"] = BASEDIR
  juggle_dir = os.path.dirname(os.path.realpath(__file__))
  subprocess.call(f"bin/plotjuggler -d {fn}", shell=True, env=env, cwd=juggle_dir)

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

def get_arg_parser():
  parser = argparse.ArgumentParser(description="PlotJuggler plugin for reading rlogs",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  
  parser.add_argument("route_name", nargs='?', help="The name of the route that will be plotted.") 
  parser.add_argument("segment_number", type=int, nargs='?', help="The index of the segment that will be plotted")
  return parser

if __name__ == "__main__":
  
  args = get_arg_parser().parse_args(sys.argv[1:])    
  if args.segment_number is None:
    juggle_route(args.route_name)
  else:
    juggle_segment(args.route_name, args.segment_number)
