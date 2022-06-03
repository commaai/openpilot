#!/usr/bin/env python3
import argparse
import sys

from tools.lib.logreader import logreader_from_route_or_segment

DEMO_ROUTE = "9f583b1d93915c31|2022-05-18--10-49-51--0"

def read_logs(lr):
  for msg in lr:
    if msg.which() == 'sendcan':
      pass

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="A tool for graphing openpilot's message dependencies",
                                   formatter_class = argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--demo", action="store_true", help="Use the demo route instead of providing one")
  parser.add_argument("route_or_segment_name", nargs='?', help="The route to print")

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit()
  args = parser.parse_args()

  r = DEMO_ROUTE if args.demo else args.route_or_segment_name.strip()
  lr = logreader_from_route_or_segment(r, sort_by_time=True)
  read_logs(lr)

