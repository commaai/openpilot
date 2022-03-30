#!/usr/bin/env python3
import argparse
import sys

from common.realtime import DT_CTRL
from tools.lib.route import Route
from tools.lib.logreader import MultiLogIterator


# Verifies car ports and new fingerprints before they're merged
# - [ ] Fingerprints correctly and fully (i.e. not fuzzy)
# - [x] Checks engaged for longer than a minute without user input
# - [ ] Gas cancellation is immediate and reliable
# - [ ] Runs through selfdrive/test/test_models.py

def check_engagement(lr):
  carState = None
  engaged_frames = 0  # counts frames openpilot was engaged with no user inputs

  for msg in lr:
    if msg.which() == "carState":
      carState = msg.carState

    elif msg.which() == "controlsState":
      controlsState = msg.controlsState
      if carState is None:
        continue
      if carState.cruiseState.enabled and controlsState.active and \
        not (carState.gasPressed or carState.brakePressed):
        engaged_frames += 1
      else:
        engaged_frames = 0

  assert engaged_frames >= 60 / DT_CTRL, \
    "Engaged time without user input was only {} seconds".format(round(engaged_frames * DT_CTRL, 1))


def get_log_reader(route_name):
  route = Route(route_name)
  log_paths = [log for log in route.log_paths() if log is not None]
  assert len(log_paths) > 0, "No uploaded rlogs in this route"

  return MultiLogIterator(log_paths, sort_by_time=True)


def verify_route(route_name):
  lr = get_log_reader(route_name)
  # check_fingerprint(lr)
  lr.reset()
  check_engagement(lr)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Verifies new car ports and fingerprints",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("route_name", nargs='?', help="")

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit()
  args = parser.parse_args()

  verify_route(args.route_name)
