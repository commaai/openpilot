#!/usr/bin/env python3
# type: ignore
# pylint: skip-file
import importlib
import unittest
from parameterized import parameterized_class

from cereal import car
from selfdrive.car.fingerprints import all_known_cars, _FINGERPRINTS
from selfdrive.car.car_helpers import interfaces
from selfdrive.test.test_car_models import routes
from selfdrive.test.openpilotci import get_url
from tools.lib.logreader import LogReader

routes = {v['carFingerprint']: k for k, v in routes.items()}


@parameterized_class(('car_model'), [(car,) for car in all_known_cars()][:2])
class TestCarModel(unittest.TestCase):

  def setUp(self):
    if self.car_model not in routes:
      self.skipTest(f"skipping {self.car_model} due to missing route")
    route_url = get_url(routes[self.car_model], 0)
    self.lr = list(LogReader(route_url))

    fingerprint = _FINGERPRINTS[self.car_model][0]
    fingerprints = {
      0: fingerprint,
      1: fingerprint,
      2: fingerprint,
    }
    CarInterface, CarController, CarState = interfaces[self.car_model]
    has_relay = False
    self.car_params = CarInterface.get_params(self.car_model, fingerprints, has_relay, [])
    assert self.car_params

    if self.car_params.dashcamOnly:
      self.skipTest(f"skipping {self.car_model}, dashcam only")

    self.CI = CarInterface(self.car_params, CarController, CarState)
    assert self.CI

  def test_car_params(self):
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

  def test_car_interface(self):
    can_invalid_cnt = 0
    CC = car.CarControl.new_message()
    for msg in self.lr:
      if msg.which() != "can":
        continue
      CS = self.CI.update(CC, (msg.as_builder().to_bytes(),))
      self.CI.apply(CC)
      can_invalid_cnt += CS.canValid
    self.assertLess(can_invalid_cnt, 20)

  def test_radar_interface(self):
    RadarInterface = importlib.import_module('selfdrive.car.%s.radar_interface' % self.car_params.carName).RadarInterface
    RI = RadarInterface(self.car_params)
    assert RI

    # Run radar interface once
    RI.update([])
    if hasattr(RI, '_update') and hasattr(RI, 'trigger_msg'):
      RI._update([RI.trigger_msg])

  def test_can_valid(self):
    pass

  def test_panda_safety(self):
    pass


if __name__ == "__main__":
  unittest.main()
