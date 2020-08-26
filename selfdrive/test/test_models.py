#!/usr/bin/env python3
# pylint: disable=E1101
import os
import importlib
import unittest
from collections import Counter
from parameterized import parameterized_class

from cereal import log, car
from selfdrive.car.fingerprints import all_known_cars
from selfdrive.car.car_helpers import interfaces
from selfdrive.test.test_car_models import routes, non_tested_cars
from selfdrive.test.openpilotci import get_url
from tools.lib.logreader import LogReader

from panda.tests.safety import libpandasafety_py
from panda.tests.safety.common import package_can_msg

HwType = log.HealthData.HwType

ROUTES = {v['carFingerprint']: k for k, v in routes.items() if v['enableCamera']}

# TODO: get updated routes for these cars
ignore_can_valid = [
  "VOLKSWAGEN GOLF",
  "ACURA ILX 2016 ACURAWATCH PLUS",
  "TOYOTA PRIUS 2017",
  "TOYOTA COROLLA 2017",
  "LEXUS RX HYBRID 2017",
  "TOYOTA AVALON 2016",
  "HONDA PILOT 2019 ELITE",
  "HYUNDAI SANTA FE LIMITED 2019",
]

@parameterized_class(('car_model'), [(car,) for car in all_known_cars()])
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
        if seg == 0:
          raise

    has_relay = False
    can_msgs = []
    fingerprint = {i: dict() for i in range(3)}
    for msg in lr:
      if msg.which() == "can":
        for m in msg.can:
          if m.src < 128:
            fingerprint[m.src][m.address] = len(m.dat)
        can_msgs.append(msg)
      elif msg.which() == "health":
        has_relay = msg.health.hwType in [HwType.blackPanda, HwType.uno, HwType.dos]
    cls.can_msgs = sorted(can_msgs, key=lambda msg: msg.logMonoTime)

    CarInterface, CarController, CarState = interfaces[cls.car_model]

    # TODO: test with relay and without
    cls.car_params = CarInterface.get_params(cls.car_model, fingerprint, has_relay, [])
    assert cls.car_params

    cls.CI = CarInterface(cls.car_params, CarController, CarState)
    assert cls.CI

  def test_car_params(self):
    if self.car_params.dashcamOnly:
      self.skipTest("no need to check carParams for dashcamOnly")

    # make sure car params are within a valid range
    self.assertGreater(self.car_params.mass, 1)
    self.assertGreater(self.car_params.steerRateCost, 1e-3)

    tuning = self.car_params.lateralTuning.which()
    if tuning == 'pid':
      self.assertTrue(len(self.car_params.lateralTuning.pid.kpV))
    elif tuning == 'lqr':
      self.assertTrue(len(self.car_params.lateralTuning.lqr.a))
    elif tuning == 'indi':
      self.assertGreater(self.car_params.lateralTuning.indi.outerLoopGain, 1e-3)

    self.assertTrue(self.car_params.enableCamera)

    # TODO: check safetyModel is in release panda build
    safety = libpandasafety_py.libpandasafety
    set_status = safety.set_safety_hooks(self.car_params.safetyModel.raw, self.car_params.safetyParam)
    self.assertEqual(0, set_status, f"failed to set safetyModel {self.car_params.safetyModel}")

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
    RadarInterface = importlib.import_module('selfdrive.car.%s.radar_interface' % self.car_params.carName).RadarInterface
    RI = RadarInterface(self.car_params)
    assert RI

    error_cnt = 0
    for msg in self.can_msgs:
      radar_data = RI.update((msg.as_builder().to_bytes(),))
      if radar_data is not None:
        error_cnt += car.RadarData.Error.canError in radar_data.errors
    self.assertLess(error_cnt, 20)

  def test_panda_safety_rx(self):
    if self.car_params.dashcamOnly:
      self.skipTest("no need to check panda safety for dashcamOnly")

    safety = libpandasafety_py.libpandasafety
    set_status = safety.set_safety_hooks(self.car_params.safetyModel.raw, self.car_params.safetyParam)
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


if __name__ == "__main__":
  unittest.main()
