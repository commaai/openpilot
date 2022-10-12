#!/usr/bin/env python3
import math
import unittest
import importlib
from parameterized import parameterized

from cereal import car
from selfdrive.car import gen_empty_fingerprint
from selfdrive.car.fingerprints import all_known_cars
from selfdrive.car.car_helpers import interfaces
from selfdrive.car.fingerprints import _FINGERPRINTS as FINGERPRINTS

class TestCarInterfaces(unittest.TestCase):

  @parameterized.expand([(car,) for car in all_known_cars()])
  def test_car_interfaces(self, car_name):
    if car_name in FINGERPRINTS:
      fingerprint = FINGERPRINTS[car_name][0]
    else:
      fingerprint = {}

    CarInterface, CarController, CarState = interfaces[car_name]
    fingerprints = gen_empty_fingerprint()
    fingerprints.update({k: fingerprint for k in fingerprints.keys()})

    car_fw = []

    car_params = CarInterface.get_params(car_name, fingerprints, car_fw)
    car_interface = CarInterface(car_params, CarController, CarState)
    assert car_params
    assert car_interface

    self.assertGreater(car_params.mass, 1)
    self.assertGreater(car_params.maxLateralAccel, 0)

    if car_params.steerControlType != car.CarParams.SteerControlType.angle:
      tuning = car_params.lateralTuning.which()
      if tuning == 'pid':
        self.assertTrue(len(car_params.lateralTuning.pid.kpV))
      elif tuning == 'torque':
        kf = car_params.lateralTuning.torque.kf
        self.assertTrue(not math.isnan(kf) and kf > 0)
        self.assertTrue(not math.isnan(car_params.lateralTuning.torque.friction))
      elif tuning == 'indi':
        self.assertTrue(len(car_params.lateralTuning.indi.outerLoopGainV))

    # Run car interface
    CC = car.CarControl.new_message()
    for _ in range(10):
      car_interface.update(CC, [])
      car_interface.apply(CC)
      car_interface.apply(CC)

    CC = car.CarControl.new_message()
    CC.enabled = True
    for _ in range(10):
      car_interface.update(CC, [])
      car_interface.apply(CC)
      car_interface.apply(CC)

    # Test radar interface
    RadarInterface = importlib.import_module(f'selfdrive.car.{car_params.carName}.radar_interface').RadarInterface
    radar_interface = RadarInterface(car_params)
    assert radar_interface

    # Run radar interface once
    radar_interface.update([])
    if not car_params.radarOffCan and radar_interface.rcp is not None and \
       hasattr(radar_interface, '_update') and hasattr(radar_interface, 'trigger_msg'):
      radar_interface._update([radar_interface.trigger_msg])

if __name__ == "__main__":
  unittest.main()
