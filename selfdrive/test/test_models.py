#!/usr/bin/env python3
# pylint: disable=E1101
import os
import importlib
import unittest
from collections import defaultdict, Counter
from parameterized import parameterized_class

from cereal import log, car
from common.params import Params
from selfdrive.car.fingerprints import all_known_cars
from selfdrive.car.car_helpers import interfaces
from selfdrive.car.gm.values import CAR as GM
from selfdrive.car.honda.values import HONDA_BOSCH, CAR as HONDA
from selfdrive.car.chrysler.values import CAR as CHRYSLER
from selfdrive.car.hyundai.values import CAR as HYUNDAI
from selfdrive.test.test_routes import routes, non_tested_cars
from selfdrive.test.openpilotci import get_url
from tools.lib.logreader import LogReader

from panda.tests.safety import libpandasafety_py
from panda.tests.safety.common import package_can_msg

PandaType = log.PandaState.PandaType

NUM_JOBS = int(os.environ.get("NUM_JOBS", "1"))
JOB_ID = int(os.environ.get("JOB_ID", "0"))

ROUTES = {rt.car_fingerprint: rt.route for rt in routes}

# TODO: get updated routes for these cars
ignore_can_valid = [
  HYUNDAI.SANTA_FE,
]

ignore_carstate_check = [
  # TODO: chrysler gas state in panda also checks wheel speed, refactor so it's only gas
  CHRYSLER.PACIFICA_2017_HYBRID,
]

ignore_addr_checks_valid = [
  GM.BUICK_REGAL,
  HYUNDAI.GENESIS_G70_2020,
]

@parameterized_class(('car_model'), [(car,) for i, car in enumerate(sorted(all_known_cars())) if i % NUM_JOBS == JOB_ID])
class TestCarModel(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    if cls.car_model not in ROUTES:
      # TODO: get routes for missing cars and remove this
      if cls.car_model in non_tested_cars:
        print(f"Skipping tests for {cls.car_model}: missing route")
        raise unittest.SkipTest
      else:
        raise Exception(f"missing test route for car {cls.car_model}")

    for seg in [2, 1, 0]:
      try:
        lr = LogReader(get_url(ROUTES[cls.car_model], seg))
        break
      except Exception:
        lr = None

    if lr is None:
      raise Exception("Route not found. Is it uploaded?")

    params = Params()
    params.clear_all()

    can_msgs = []
    fingerprint = {i: dict() for i in range(3)}
    for msg in lr:
      if msg.which() == "can":
        for m in msg.can:
          if m.src < 64:
            fingerprint[m.src][m.address] = len(m.dat)
        can_msgs.append(msg)
      elif msg.which() == "carParams":
        if msg.carParams.openpilotLongitudinalControl:
          params.put_bool("DisableRadar", True)

    assert len(can_msgs) > 0, "No CAN msgs in test segment"
    cls.can_msgs = sorted(can_msgs, key=lambda msg: msg.logMonoTime)

    cls.CarInterface, cls.CarController, cls.CarState = interfaces[cls.car_model]

    cls.CP = cls.CarInterface.get_params(cls.car_model, fingerprint, [])
    assert cls.CP

  def setUp(self):
    self.CI = self.CarInterface(self.CP, self.CarController, self.CarState)
    assert self.CI

    # TODO: check safetyModel is in release panda build
    self.safety = libpandasafety_py.libpandasafety
    set_status = self.safety.set_safety_hooks(self.CP.safetyConfigs[0].safetyModel.raw, self.CP.safetyConfigs[0].safetyParam)
    self.assertEqual(0, set_status, f"failed to set safetyModel {self.CP.safetyConfigs[0].safetyModel}")
    self.safety.init_tests()

  def test_car_params(self):
    if self.CP.dashcamOnly:
      self.skipTest("no need to check carParams for dashcamOnly")

    # make sure car params are within a valid range
    self.assertGreater(self.CP.mass, 1)
    self.assertGreater(self.CP.steerRateCost, 1e-3)

    if self.CP.steerControlType != car.CarParams.SteerControlType.angle:
      tuning = self.CP.lateralTuning.which()
      if tuning == 'pid':
        self.assertTrue(len(self.CP.lateralTuning.pid.kpV))
      elif tuning == 'lqr':
        self.assertTrue(len(self.CP.lateralTuning.lqr.a))
      elif tuning == 'indi':
        self.assertTrue(len(self.CP.lateralTuning.indi.outerLoopGainV))
      else:
        raise Exception("unkown tuning")

  def test_car_interface(self):
    # TODO: also check for checkusm and counter violations from can parser
    can_invalid_cnt = 0
    CC = car.CarControl.new_message()
    for i, msg in enumerate(self.can_msgs):
      CS = self.CI.update(CC, (msg.as_builder().to_bytes(),))
      self.CI.apply(CC)

      # wait 2s for low frequency msgs to be seen
      if i > 200:
        can_invalid_cnt += not CS.canValid

    if self.car_model not in ignore_can_valid:
      self.assertLess(can_invalid_cnt, 50)

  def test_radar_interface(self):
    os.environ['NO_RADAR_SLEEP'] = "1"
    RadarInterface = importlib.import_module('selfdrive.car.%s.radar_interface' % self.CP.carName).RadarInterface
    RI = RadarInterface(self.CP)
    assert RI

    error_cnt = 0
    for msg in self.can_msgs:
      radar_data = RI.update((msg.as_builder().to_bytes(),))
      if radar_data is not None:
        error_cnt += car.RadarData.Error.canError in radar_data.errors
    self.assertLess(error_cnt, 20)

  def test_panda_safety_rx_valid(self):
    if self.CP.dashcamOnly:
      self.skipTest("no need to check panda safety for dashcamOnly")

    start_ts = self.can_msgs[0].logMonoTime

    failed_addrs = Counter()
    for can in self.can_msgs:
      # update panda timer
      t = (can.logMonoTime - start_ts) / 1e3
      self.safety.set_timer(int(t))

      # run all msgs through the safety RX hook
      for msg in can.can:
        if msg.src >= 64:
          continue

        to_send = package_can_msg([msg.address, 0, msg.dat, msg.src])
        if self.safety.safety_rx_hook(to_send) != 1:
          failed_addrs[hex(msg.address)] += 1

      # ensure all msgs defined in the addr checks are valid
      if self.car_model not in ignore_addr_checks_valid:
        self.safety.safety_tick_current_rx_checks()
        if t > 1e6:
          self.assertTrue(self.safety.addr_checks_valid())
    self.assertFalse(len(failed_addrs), f"panda safety RX check failed: {failed_addrs}")

  def test_panda_safety_carstate(self):
    """
      Assert that panda safety matches openpilot's carState
    """
    if self.CP.dashcamOnly:
      self.skipTest("no need to check panda safety for dashcamOnly")
    if self.car_model in ignore_carstate_check:
      self.skipTest("see comments in test_models.py")

    checks = defaultdict(lambda: 0)
    CC = car.CarControl.new_message()
    for can in self.can_msgs:
      for msg in can.can:
        if msg.src >= 64:
          continue
        to_send = package_can_msg([msg.address, 0, msg.dat, msg.src])
        ret = self.safety.safety_rx_hook(to_send)
        self.assertEqual(1, ret, f"safety rx failed ({ret=}): {to_send}")
      CS = self.CI.update(CC, (can.as_builder().to_bytes(),))

      # TODO: check steering state
      # check that openpilot and panda safety agree on the car's state
      checks['gasPressed'] += CS.gasPressed != self.safety.get_gas_pressed_prev()
      checks['brakePressed'] += CS.brakePressed != self.safety.get_brake_pressed_prev()
      if self.CP.pcmCruise:
        checks['controlsAllowed'] += not CS.cruiseState.enabled and self.safety.get_controls_allowed()

      # TODO: extend this to all cars
      if self.CP.carName == "honda":
        checks['mainOn'] += CS.cruiseState.available != self.safety.get_acc_main_on()

    # TODO: reduce tolerance to 0
    failed_checks = {k: v for k, v in checks.items() if v > 25}

    # TODO: the panda and openpilot interceptor thresholds should match
    skip_gas_check = self.CP.carName == 'chrysler'
    if "gasPressed" in failed_checks and (self.CP.enableGasInterceptor or skip_gas_check):
      if failed_checks['gasPressed'] < 150 or skip_gas_check:
        del failed_checks['gasPressed']

    # TODO: honda nidec: do same checks in carState and panda
    if "brakePressed" in failed_checks and self.CP.carName == 'honda' and \
      (self.car_model not in HONDA_BOSCH or self.car_model in [HONDA.CRV_HYBRID, HONDA.HONDA_E]):
      if failed_checks['brakePressed'] < 150:
        del failed_checks['brakePressed']

    # TODO: use the same signal in panda and carState
    # tolerate a small delay between the button press and PCM entering a cruise state
    if self.car_model == HONDA.ACCORD_2021:
      if failed_checks['controlsAllowed'] < 500:
        del failed_checks['controlsAllowed']

    self.assertFalse(len(failed_checks), f"panda safety doesn't agree with CarState: {failed_checks}")

if __name__ == "__main__":
  unittest.main()
