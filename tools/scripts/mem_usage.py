#!/usr/bin/env python3
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from openpilot.selfdrive.test.mem_usage import DEMO_ROUTE, print_report
from openpilot.tools.lib.logreader import LogReader


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Analyze memory usage from route logs")
  parser.add_argument("route", nargs="?", default=None, help="route ID or local rlog path")
  parser.add_argument("--demo", action="store_true", help=f"use demo route ({DEMO_ROUTE})")
  args = parser.parse_args()

  if args.demo:
    route = DEMO_ROUTE
  elif args.route:
    route = args.route
  else:
    parser.error("provide a route or use --demo")

  print(f"Reading logs from: {route}")

  proc_logs = []
  device_states = []
  for msg in LogReader(route):
    if msg.which() == 'procLog':
      proc_logs.append(msg)
    elif msg.which() == 'deviceState':
      device_states.append(msg)

  print_report(proc_logs, device_states)
