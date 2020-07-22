#!/usr/bin/env python3
# pylint: disable=E1101

import importlib
import unittest
from parameterized import parameterized_class

from cereal import car
from selfdrive.car.fingerprints import all_known_cars, _FINGERPRINTS
from selfdrive.car.car_helpers import interfaces
from selfdrive.test.test_car_models import routes
from selfdrive.test.openpilotci import get_url
from tools.lib.logreader import LogReader

from panda.tests.safety import libpandasafety_py
from panda.tests.safety.common import package_can_msg

ROUTES = {v['carFingerprint']: k for k, v in routes.items() if v['enableCamera']}

@parameterized_class(('car_model'), [(car,) for car in all_known_cars()][:2])
class TestCarModel(unittest.TestCase):

  def setUp(self):
    if self.car_model not in ROUTES:
      self.skipTest(f"skipping {self.car_model} due to missing route")

    fingerprints = {i: _FINGERPRINTS[self.car_model][0] for i in range(3)}

    CarInterface, CarController, CarState = interfaces[self.car_model]
    has_relay = False
    self.car_params = CarInterface.get_params(self.car_model, fingerprints, has_relay, [])
    assert self.car_params

    self.CI = CarInterface(self.car_params, CarController, CarState)
    assert self.CI

    route_url = get_url(ROUTES[self.car_model], 0)
    self.can_msgs = [msg for msg in LogReader(route_url) if msg.which() == "can"]

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

    # TODO: check safetyModel is in release panda build
    safety = libpandasafety_py.libpandasafety
    set_status = safety.set_safety_hooks(self.car_params.safetyModel.raw, self.car_params.safetyParam)
    self.assertEqual(0, set_status, f"failed to set safetyModel {self.car_params.safetyModel}")

  def test_car_interface(self):
    can_invalid_cnt = 0
    CC = car.CarControl.new_message()
    for msg in self.can_msgs:
      CS = self.CI.update(CC, (msg.as_builder().to_bytes(),))
      self.CI.apply(CC)
      can_invalid_cnt += CS.canValid
      # TODO: add this back
      #self.assertLess(can_invalid_cnt, 10)

  def test_radar_interface(self):
    RadarInterface = importlib.import_module('selfdrive.car.%s.radar_interface' % self.car_params.carName).RadarInterface
    RI = RadarInterface(self.car_params)
    assert RI
    for msg in self.can_msgs:
      RI.update((msg.as_builder().to_bytes(),))

  def test_panda_safety_rx(self):
    if self.car_params.dashcamOnly:
      self.skipTest("no need to check panda safety for dashcamOnly")

    safety = libpandasafety_py.libpandasafety
    safety.set_safety_hooks(self.car_params.safetyModel.raw, self.car_params.safetyParam)

    failed_addrs = set()
    for can in self.can_msgs:
      for msg in can.can:
        to_send = package_can_msg([msg.address, 0, msg.dat, msg.src])
        if not safety.safety_rx_hook(to_send):
          failed_addrs.add(msg.address)
    self.assertFalse(len(failed_addrs), f"panda safety RX check failed: {failed_addrs}")


if __name__ == "__main__":
  unittest.main()
