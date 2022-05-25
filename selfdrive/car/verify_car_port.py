#!/usr/bin/env python3
import argparse
import os
import sys

os.environ['FILEREADER_CACHE'] = '1'

from common.realtime import DT_CTRL
from selfdrive.car.fw_versions import build_fw_dict, match_fw_to_car_exact
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


def format_exception(test_name, exception_text):
  exception = '=' * 70
  exception += f'\nERROR: {test_name}\n'
  exception += '-' * 70
  exception += f'\n{exception_text}\n'

  return exception


def verify_route(route_name):
  all_msgs = list(get_log_reader(route_name))
  tests = [
    # test_car_params,
    # test_blocked_msgs,
    test_engagement,
    # test_steering_faults
  ]
  exceptions = {}
  for test in tests:
    success, text = test(all_msgs)
    if not success:
      print("TEST FAILED: {}\n".format(test.__name__))
      exceptions[test.__name__] = format_exception(test.__name__, text)

  if len(exceptions):
    for exc in exceptions.values():
      print(exc)
    print("---\nFAILED: Some tests failed")
  else:
    print("---\nSUCCESS: All tests passed")


def _test_fingerprint(CP):
  """
  Runs fingerprinting on fw versions in log to ensure we're not fuzzy fingerprinting
  """

  fw_versions_dict = build_fw_dict(CP.carFw)
  matches = match_fw_to_car_exact(fw_versions_dict)
  assert len(matches) == 1, f"got more than one candidate: {matches}"
  assert list(matches)[0] == CP.carFingerprint  # TODO: support specifying with argparse

  print('SUCCESS: Fingerprinted: {}'.format(CP.carFingerprint))


def test_blocked_msgs(lr):
  for msg in list(lr)[::-1]:
    if msg.which() == "pandaStates":
      if msg.pandaStates[0].ignitionCan or msg.pandaStates[0].ignitionLine:
        pandaStates = msg.pandaStates
        break
  else:
    assert False, "No pandaStates packets"

  assert pandaStates[0].blockedCnt < 10, "Blocked messages {} is not less than 10".format(pandaStates[0].blockedCnt)
  print('SUCCESS: Blocked messages under threshold: {} < 10'.format(pandaStates[0].blockedCnt))


def test_car_params(lr):
  """
  Common carParams tests
  """

  for msg in lr:
    if msg.which() == "carParams":
      CP = msg.carParams
      assert not CP.dashcamOnly, "openpilot not active, dashcamOnly is True"
      _test_fingerprint(CP)
      return True

  assert False, "No carParams packets in logs"


def test_engagement(lr):
  carState = None
  engaged_frames = 0
  for msg in lr:
    if msg.which() == "carState":
      carState = msg.carState

    elif msg.which() == "controlsState":
      if carState is None:
        continue

      controlsState = msg.controlsState
      if carState.cruiseState.enabled and controlsState.active and \
        not (carState.gasPressed or carState.brakePressed or carState.steeringPressed):
        engaged_frames += 1
        if engaged_frames >= MIN_ENGAGED_FRAMES:
          print("SUCCESS: Route was engaged for at least {} seconds".format(round(MIN_ENGAGED_FRAMES * DT_CTRL, 2)))
          return True, ""
      else:
        engaged_frames = 0

  return False, "Route was not engaged for longer than {} seconds".format(MIN_ENGAGED_FRAMES * DT_CTRL)


def test_steering_faults(lr):
  CS = None
  CC = None
  for msg in lr:
    if msg.which() == "carControl":
      CC = msg.carControl

    elif msg.which() == "carState":
      if CC is None:
        continue

      CS = msg.carState
      steering_fault = (CS.steerFaultTemporary or CS.steerFaultPermanent) and CC.latActive
      assert not steering_fault, "Route had at least one steering fault event"

  assert CS is not None, "No carState packets in logs"
  print("SUCCESS: Route had no steering faults")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Verifies new car ports and fingerprints",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("route_name", nargs='?',
                      help="Pass a route name with uploaded rlogs to verify against common issues")

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit()
  args = parser.parse_args()

  verify_route(args.route_name)
