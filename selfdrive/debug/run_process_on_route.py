#!/usr/bin/env python3

import argparse

from openpilot.selfdrive.test.process_replay.process_replay import CONFIGS, MultiProcessReplaySession
from openpilot.tools.lib.logreader import LogReader, save_log

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Run process on route and create new logs",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--fingerprint", help="The fingerprint to use")
  parser.add_argument("route", help="The route name to use")
  parser.add_argument("process", nargs='+', help="The process(s) to run")
  args = parser.parse_args()

  cfgs = [c for c in CONFIGS if c.proc_name in args.process]

  lr = LogReader(args.route)#, prefetch_all=True, max_download_workers=32)
  fn = f"{args.route.replace('/', '_')}_{'_'.join(args.process)}.zst",
  replay_stream = MultiProcessReplaySession(cfgs, lr, fingerprint=args.fingerprint, return_all_logs=True),
  save_log(fn, replay_stream)
