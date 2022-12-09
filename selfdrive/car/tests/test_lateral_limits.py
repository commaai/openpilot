#!/usr/bin/env python3
from collections import defaultdict
import importlib
from parameterized import parameterized_class
import sys
import unittest

from selfdrive.car.car_helpers import interfaces
from selfdrive.car.fingerprints import all_known_cars
from selfdrive.car.interfaces import get_torque_params
from selfdrive.car.gm.values import CAR as GM
from selfdrive.car.mazda.values import CAR as MAZDA
from selfdrive.car.toyota.values import CAR as TOYOTA

CAR_MODELS = all_known_cars()

# ISO 11270
MAX_LAT_JERK = 5.0   # m/s^3
MAX_LAT_ACCEL = 3.0  # m/s^2

# TODO: fix the lateral limits on this cars and remove from list
ABOVE_LIMITS_CARS = [
  GM.SILVERADO,     # 5.4 m/s^3 down
  GM.BOLT_EUV,      # 5.7 m/s^3 down
  MAZDA.CX5_2022,   # 5.5 m/s^3 down
  MAZDA.CX9_2021,   # 5.5 m/s^3 down
  TOYOTA.COROLLA,   # 5.2 m/s^3 down
]

car_model_jerks = defaultdict(dict)


@parameterized_class('car_model', [(c,) for c in CAR_MODELS])
class TestLateralLimits(unittest.TestCase):
  car_model: str

  @classmethod
  def setUpClass(cls):
    CarInterface, _, _ = interfaces[cls.car_model]
    CP = CarInterface.get_params(cls.car_model)

    if CP.dashcamOnly:
      raise unittest.SkipTest("Platform is behind dashcamOnly")

    # TODO: test all platforms
    if CP.lateralTuning.which() != 'torque':
      raise unittest.SkipTest

    if CP.notCar:
      raise unittest.SkipTest

    if CP.carName == "hyundai" or CP.carFingerprint in ABOVE_LIMITS_CARS:
      raise unittest.SkipTest

    CarControllerParams = importlib.import_module(f'selfdrive.car.{CP.carName}.values').CarControllerParams
    cls.control_params = CarControllerParams(CP)
    cls.torque_params = get_torque_params(cls.car_model)

  @staticmethod
  def calculate_jerk(control_params, torque_params):
    steer_step = control_params.STEER_STEP
    time_to_max = control_params.STEER_MAX / control_params.STEER_DELTA_UP / 100. * steer_step
    time_to_min = control_params.STEER_MAX / control_params.STEER_DELTA_DOWN / 100. * steer_step
    max_lat_accel = torque_params['LAT_ACCEL_FACTOR']

    return max_lat_accel / time_to_max, max_lat_accel / time_to_min

  def test_jerk_limits(self):
    up_jerk, down_jerk = self.calculate_jerk(self.control_params, self.torque_params)
    car_model_jerks[self.car_model] = {"up_jerk": up_jerk, "down_jerk": down_jerk}
    self.assertLessEqual(up_jerk, MAX_LAT_JERK)
    self.assertLessEqual(down_jerk, MAX_LAT_JERK)

  def test_max_lateral_accel(self):
    self.assertLessEqual(self.torque_params["LAT_ACCEL_FACTOR"], MAX_LAT_ACCEL)


if __name__ == "__main__":
  result = unittest.main(exit=False)

  print(f"\n\n---- Lateral limit report ({len(CAR_MODELS)} cars) ----\n")

  max_car_model_len = max([len(car_model) for car_model in car_model_jerks])
  for car_model, _jerks in sorted(car_model_jerks.items(), key=lambda i: i[1]['up_jerk'], reverse=True):
    violation = any([_jerk >= MAX_LAT_JERK for _jerk in _jerks.values()])
    violation_str = " - VIOLATION" if violation else ""

    print(f"{car_model:{max_car_model_len}} - up jerk: {round(_jerks['up_jerk'], 2):5} m/s^3, down jerk: {round(_jerks['down_jerk'], 2):5} m/s^3{violation_str}")

  # exit with test result
  sys.exit(not result.result.wasSuccessful())
