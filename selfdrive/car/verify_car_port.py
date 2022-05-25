#!/usr/bin/env python3
import argparse
import os
import sys
import unittest

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


class TestCarPort(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    route_name = os.getenv("ROUTE_NAME", None)
    cls.assertIsNotNone(route_name, "Set ROUTE_NAME environment variable")

    route = Route(route_name)
    log_paths = [log for log in route.log_paths() if log is not None]
    cls.assertTrue(len(log_paths) > 0, "No uploaded rlogs in this route")

    cls.all_msgs = list(MultiLogIterator(log_paths, sort_by_time=True))
    cls.CP = None
    for msg in cls.all_msgs:
      if msg.which() == "carParams":
        cls.CP = msg.carParams
    cls.assertIsNotNone(cls.CP, "No CarParams packets in logs")

  def test_fingerprint(self):
    """
    Runs fingerprinting on fw versions in log to ensure we're not fuzzy fingerprinting
    """

    fw_versions_dict = build_fw_dict(self.CP.carFw)
    matches = match_fw_to_car_exact(fw_versions_dict)
    self.assertNotEqual(len(matches), 0, "Car failed to exact fingerprint")
    self.assertEqual(len(matches), 1, f"Got more than one candidate: {matches}")
    # TODO: support specifying expected fingerprint with argparse
    self.assertEqual(list(matches)[0], self.CP.carFingerprint,
                     f"Car mis-fingerprinted as {list(matches)[0]}, expecting {self.CP.carFingerprint}")

  def test_blocked_msgs(self):
    for msg in self.all_msgs[::-1]:
      if msg.which() == "pandaStates":
        if msg.pandaStates[0].ignitionCan or msg.pandaStates[0].ignitionLine:
          pandaStates = msg.pandaStates
          break
    else:
      raise Exception("No pandaStates packets")

    self.assertLess(pandaStates[0].blockedCnt, 10, "Blocked messages {} is not less than 10".format(pandaStates[0].blockedCnt))

  def test_car_params(self):
    """
    Common carParams tests
    """

    self.assertFalse(self.CP.dashcamOnly, "openpilot not active, dashcamOnly is True")

  def test_engagement(self):
    carState = None
    engaged_frames = 0
    max_engaged_frames = 0
    for msg in self.all_msgs:
      if msg.which() == "carState":
        carState = msg.carState

      elif msg.which() == "controlsState":
        if carState is None:
          continue

        controlsState = msg.controlsState
        if carState.cruiseState.enabled and controlsState.active and \
          not (carState.gasPressed or carState.brakePressed or carState.steeringPressed):
          engaged_frames += 1
          max_engaged_frames = max(engaged_frames, max_engaged_frames)
        else:
          engaged_frames = 0

    self.assertGreaterEqual(engaged_frames, MIN_ENGAGED_FRAMES,
                            "Route was not engaged for longer than {} seconds".format(MIN_ENGAGED_FRAMES * DT_CTRL))

  def test_steering_faults(self):
    CC = None
    steering_faults = 0
    for msg in self.all_msgs:
      if msg.which() == "carControl":
        CC = msg.carControl

      elif msg.which() == "carState":
        if CC is None:
          continue

        CS = msg.carState
        steering_faults += (CS.steerFaultTemporary or CS.steerFaultPermanent) and CC.latActive

    self.assertLess(steering_faults, 10, f"Route had {steering_faults} steering fault frames")

  # def test_cancellation(self):
  #   """
  #   Tests that when openpilot wants to cancel, the car cancels relatively quickly
  #   and doesn't re-engage within a few frames after cancellation
  #   """
  #
  #   controlsState = None
  #   prev_controlsState = None
  #   CS = None
  #   prev_cancel = False
  #   cancel_t = None
  #
  #   for msg in self.all_msgs:
  #     if msg.which() == 'controlsState':
  #       prev_controlsState = controlsState
  #       if msg.controlsState.cancel and not prev_controlsState.cancel:
  #         cancel_t =
  #       controlsState = msg.controlsState
  #
  #     elif msg.which() == 'carState':
  #       if controlsState is None or prev_controlsState is None:
  #         continue
  #
  #       CS = msg.carState
  #       if controlsState.cancel and not prev_controlsState.cancel:
  #         pass
  #
  #       prev_cancel =
  #
  #
  #     elif msg.which() == 'can':
  #       # Just for easy finding in PJ, can delete
  #       if start_log_mono_time is None:
  #         start_log_mono_time = msg.logMonoTime
  #
  #       cp.update_string(msg.as_builder().to_bytes())
  #       if not cp.can_valid or CP is None or controlsState is None:
  #         continue
  #
  #       enabled = cp.vl["SCC12"]["ACCMode"] != 0
  #       gas_pressed = (cp.vl["E_EMS11"]["CR_Vcu_AccPedDep_Pos"] / 254.) > 0
  #       cruise_buttons.append(cp.vl["CLU11"]["CF_Clu_CruiseSwState"])
  #       cruise_main.append(cp.vl["CLU11"]["CF_Clu_CruiseSwMain"] != 0)
  #
  #       if prev_enabled is not None:
  #         if not enabled and prev_enabled:
  #           cancel_t = msg.logMonoTime
  #           total_disengagements += 1
  #           if gas_pressed:
  #             total_gas_disengagements += 1
  #         # On enabling rising edge and no buttons pressed recently
  #         elif enabled and not prev_enabled:
  #           #             print(cruise_buttons)
  #           total_engagements += 1
  #           if not any(cruise_main) and not any(cruise_buttons):
  #             time_disengaged = (msg.logMonoTime - cancel_t) / 1e7  # converts to frames
  #             reengage_times.append(time_disengaged)
  #             if time_disengaged < 20:  # frames (*10 ms)
  #               print('Possible re-engage: {} at {} s'.format(time_disengaged,
  #                                                             (msg.logMonoTime - start_log_mono_time) / 1e+9))
  #               print('Route: {}'.format(route))
  #       prev_enabled = enabled


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Verifies new car ports and fingerprints",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("route_name", nargs='?',
                      help="Pass a route name with uploaded rlogs to verify against common issues")

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit()
  args = parser.parse_args()

  os.environ["ROUTE_NAME"] = args.route_name
  unittest.main(argv=[''])
  # verify_route(args.route_name)
