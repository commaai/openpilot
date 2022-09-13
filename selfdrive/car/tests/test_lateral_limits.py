#!/usr/bin/env python3
import importlib
from parameterized import parameterized_class
import unittest

from cereal import car
from selfdrive.car.car_helpers import interfaces
from selfdrive.car.fingerprints import all_known_cars
from selfdrive.car.interfaces import get_torque_params


MAX_LAT_UP_JERK = 2.5  # m/s^3
MAX_LAT_UP_JERK_TOLERANCE = 0.5  # m/s^3
MIN_LAT_DOWN_JERK = 2.0  # m/s^3


@parameterized_class('car_model', [(c,) for c in all_known_cars()])
class TestLateralLimits(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    CarInterface, _, _ = interfaces[cls.car_model]
    CP = CarInterface.get_params(cls.car_model)

    if CP.dashcamOnly:
      raise unittest.SkipTest("Platform is behind dashcamOnly")

    # TODO: test angle
    if CP.steerControlType == car.CarParams.SteerControlType.angle:
      raise unittest.SkipTest

    if CP.notCar:
      raise unittest.SkipTest

    # TODO: test Honda
    if CP.carName in ("honda",):
      raise unittest.SkipTest("No steering safety")

    CarControllerParams = importlib.import_module(f'selfdrive.car.{CP.carName}.values').CarControllerParams
    cls.control_params = CarControllerParams(CP)
    cls.torque_params = get_torque_params(cls.car_model)

  def _calc_jerk(self):
    # TODO: some cars don't send at 100hz, put steer rate/step into CCP to calculate this properly
    steer_step = self.control_params.STEER_STEP
    time_to_max = self.control_params.STEER_MAX / self.control_params.STEER_DELTA_UP / 100. * steer_step
    time_to_min = self.control_params.STEER_MAX / self.control_params.STEER_DELTA_DOWN / 100. * steer_step
    max_lat_accel = self.torque_params['LAT_ACCEL_FACTOR']

    return max_lat_accel / time_to_max, max_lat_accel / time_to_min

  def test_jerk_limits(self):
    up_jerk, down_jerk = self._calc_jerk()
    self.assertLess(up_jerk, MAX_LAT_UP_JERK + MAX_LAT_UP_JERK_TOLERANCE)
    self.assertGreater(down_jerk, MIN_LAT_DOWN_JERK)


if __name__ == "__main__":
  unittest.main()
