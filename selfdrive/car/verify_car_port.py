#!/usr/bin/env python3
import argparse
import sys

from cereal import car
from common.realtime import DT_CTRL
from selfdrive.car.car_helpers import get_interface_attr
from tools.lib.logreader import MultiLogIterator
from tools.lib.route import Route

MIN_ENGAGED_FRAMES = 10 / DT_CTRL


# Verifies car ports and new fingerprints before they're merged
# - [x] Fingerprints correctly and fully (i.e. not fuzzy)
# - [x] Checks engaged for longer than a minute without user input
# - [ ] Gas cancellation is immediate and reliable
# - [x] No steering faults (temporary or permanent)
# - [ ] Runs through selfdrive/test/test_models.py

def get_log_reader(route_name):
  route = Route(route_name)
  log_paths = [log for log in route.log_paths() if log is not None]
  assert len(log_paths) > 0, "No uploaded rlogs in this route"

  return MultiLogIterator(log_paths, sort_by_time=True)


def verify_route(route_name):
  lr = get_log_reader(route_name)
  test_fingerprint(lr)
  lr.reset()
  test_engagement(lr)
  lr.reset()
  test_steering_faults(lr)


def test_fingerprint(lr):
  for msg in lr:
    if msg.which() == "carParams":
      CP = msg.carParams
      fw_versions = get_interface_attr("FW_VERSIONS")[CP.carName]
      has_fw_versions = fw_versions is not None and len(fw_versions)

      assert CP.carName != "mock"
      assert len(CP.carFingerprint) > 0
      if has_fw_versions:
        assert CP.fingerprintSource == car.CarParams.FingerprintSource.fw
      else:
        assert CP.fingerprintSource != car.CarParams.FingerprintSource.fixed
      assert not CP.fuzzyFingerprint
      print('Fingerprinted: {}'.format(CP.carFingerprint))
      return True

  assert False, "No carParams packets in logs"


def test_engagement(lr):
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
        not (carState.gasPressed or carState.brakePressed or carState.steeringPressed):
        engaged_frames += 1
        if engaged_frames >= MIN_ENGAGED_FRAMES:
          print('Route had had at least {} seconds of being engaged'.format(round(MIN_ENGAGED_FRAMES * DT_CTRL, 2)))
          return True
      else:
        engaged_frames = 0

  assert False, "Route was not engaged for longer than {} seconds".format(MIN_ENGAGED_FRAMES * DT_CTRL)


def test_steering_faults(lr):
  CS = None
  for msg in lr:
    if msg.which() == "carState":
      CS = msg.carState
      assert not (CS.steeringFaultTemporary or CS.steeringFaultPermanent), "Route had at least one steering fault event"

  assert CS is not None, "No carState packets in logs"


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Verifies new car ports and fingerprints",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("route_name", nargs='?', help="")

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit()
  args = parser.parse_args()

  verify_route(args.route_name)
