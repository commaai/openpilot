#!/usr/bin/env python3
# pylint: disable=E1101
import os
import importlib
import unittest
from collections import defaultdict, Counter
from parameterized import parameterized_class

from cereal import log, car
from selfdrive.car.fingerprints import all_known_cars
from selfdrive.car.car_helpers import interfaces
from selfdrive.car.honda.values import HONDA_BOSCH
from selfdrive.car.honda.values import CAR as HONDA
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

@parameterized_class(('car_model'), [(car,) for i, car in enumerate(all_known_cars()) if i % NUM_JOBS == JOB_ID])
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

    can_msgs = []
    fingerprint = {i: dict() for i in range(3)}
    for msg in lr:
      if msg.which() == "can":
        for m in msg.can:
          if m.src < 128:
            fingerprint[m.src][m.address] = len(m.dat)
        can_msgs.append(msg)
    cls.can_msgs = sorted(can_msgs, key=lambda msg: msg.logMonoTime)

    CarInterface, CarController, CarState = interfaces[cls.car_model]

    cls.CP = CarInterface.get_params(cls.car_model, fingerprint, [])
    assert cls.CP

    cls.CI = CarInterface(cls.CP, CarController, CarState)
    assert cls.CI

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

    # TODO: check safetyModel is in release panda build
    safety = libpandasafety_py.libpandasafety
    set_status = safety.set_safety_hooks(self.CP.safetyConfigs[0].safetyModel.raw, self.CP.safetyConfigs[0].safetyParam)
    self.assertEqual(0, set_status, f"failed to set safetyModel {self.CP.safetyConfigs[0].safetyModel}")

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

    safety = libpandasafety_py.libpandasafety
    set_status = safety.set_safety_hooks(self.CP.safetyConfigs[0].safetyModel.raw, self.CP.safetyConfigs[0].safetyParam)
    self.assertEqual(0, set_status)

    failed_addrs = Counter()
    for can in self.can_msgs:
      for msg in can.can:
        if msg.src >= 128:
          continue
        to_send = package_can_msg([msg.address, 0, msg.dat, msg.src])
        if not safety.safety_rx_hook(to_send):
          failed_addrs[hex(msg.address)] += 1
    self.assertFalse(len(failed_addrs), f"panda safety RX check failed: {failed_addrs}")

  def test_panda_safety_carstate(self):
    if self.CP.dashcamOnly:
      self.skipTest("no need to check panda safety for dashcamOnly")
    if self.car_model in ignore_carstate_check:
      self.skipTest("see comments in test_models.py")

    safety = libpandasafety_py.libpandasafety
    set_status = safety.set_safety_hooks(self.CP.safetyConfigs[0].safetyModel.raw, self.CP.safetyConfigs[0].safetyParam)
    self.assertEqual(0, set_status)

    checks = defaultdict(lambda: 0)
    CC = car.CarControl.new_message()
    for can in self.can_msgs:
      for msg in can.can:
        if msg.src >= 128:
          continue
        to_send = package_can_msg([msg.address, 0, msg.dat, msg.src])
        safety.safety_rx_hook(to_send)
      CS = self.CI.update(CC, (can.as_builder().to_bytes(),))

      # TODO: check steering state
      # check that openpilot and panda safety agree on the car's state
      checks['gasPressed'] += CS.gasPressed != safety.get_gas_pressed_prev()
      checks['brakePressed'] += CS.brakePressed != safety.get_brake_pressed_prev()
      checks['controlsAllowed'] += not CS.cruiseState.enabled and safety.get_controls_allowed()

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

    self.assertFalse(len(failed_checks), f"panda safety doesn't agree with CarState: {failed_checks}")

if __name__ == "__main__":
  unittest.main()
