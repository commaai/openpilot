#!/usr/bin/env python3

import argparse
from collections import deque

from openpilot.selfdrive.test.process_replay.process_replay import MultiProcessReplaySession
from openpilot.tools.lib.logreader import save_log

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Run process on route and create new logs",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--fingerprint", help="The fingerprint to use. (e.g. --fingerprint HONDA_CIVIC_2022)")
  parser.add_argument("route", help="The route name to use. (e.g. 'a2a0ccea32023010|2023-07-27--13-01-19')")
  parser.add_argument("process", nargs='+', help="The process(s) to run. (e.g.  card controlsd torqued)")
  parser.add_argument("--no-log", action="store_true", help="Don't save the log")
  parser.add_argument("--no-compress", action="store_true", help="Don't compress the log")
  args = parser.parse_args()

  with MultiProcessReplaySession(
    args.process, args.route, fingerprint=args.fingerprint, return_all_logs=not args.no_log
  ) as replay_stream:
    if args.no_log:
      deque(replay_stream, maxlen=0)
    else:
      save_log(f"{args.route.replace('/', '_')}_{'_'.join(args.process)}.zst", replay_stream, compress=not args.no_compress)
