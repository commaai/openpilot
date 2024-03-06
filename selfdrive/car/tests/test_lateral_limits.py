#!/usr/bin/env python3
from collections import defaultdict
import importlib
from parameterized import parameterized_class
import sys
import unittest

from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.car.car_helpers import interfaces
from openpilot.selfdrive.car.fingerprints import all_known_cars
from openpilot.selfdrive.car.interfaces import get_torque_params
from openpilot.selfdrive.car.subaru.values import CAR as SUBARU

CAR_MODELS = all_known_cars()

# ISO 11270 - allowed up jerk is strictly lower than recommended limits
MAX_LAT_ACCEL = 3.0              # m/s^2
MAX_LAT_JERK_UP = 2.5            # m/s^3
MAX_LAT_JERK_DOWN = 5.0          # m/s^3
MAX_LAT_JERK_UP_TOLERANCE = 0.5  # m/s^3

# jerk is measured over half a second
JERK_MEAS_T = 0.5

# TODO: put these cars within limits
ABOVE_LIMITS_CARS = [
  SUBARU.LEGACY,
  SUBARU.OUTBACK,
]

car_model_jerks: defaultdict[str, dict[str, float]] = defaultdict(dict)


@parameterized_class('car_model', [(c,) for c in sorted(CAR_MODELS)])
class TestLateralLimits(unittest.TestCase):
  car_model: str

  @classmethod
  def setUpClass(cls):
    CarInterface, _, _ = interfaces[cls.car_model]
    CP = CarInterface.get_non_essential_params(cls.car_model)

    if CP.dashcamOnly:
      raise unittest.SkipTest("Platform is behind dashcamOnly")

    # TODO: test all platforms
    if CP.lateralTuning.which() != 'torque':
      raise unittest.SkipTest

    if CP.notCar:
      raise unittest.SkipTest

    if CP.carFingerprint in ABOVE_LIMITS_CARS:
      raise unittest.SkipTest

    CarControllerParams = importlib.import_module(f'selfdrive.car.{CP.carName}.values').CarControllerParams
    cls.control_params = CarControllerParams(CP)
    cls.torque_params = get_torque_params(cls.car_model)

  @staticmethod
  def calculate_0_5s_jerk(control_params, torque_params):
    steer_step = control_params.STEER_STEP
    max_lat_accel = torque_params['MAX_LAT_ACCEL_MEASURED']

    # Steer up/down delta per 10ms frame, in percentage of max torque
    steer_up_per_frame = control_params.STEER_DELTA_UP / control_params.STEER_MAX / steer_step
    steer_down_per_frame = control_params.STEER_DELTA_DOWN / control_params.STEER_MAX / steer_step

    # Lateral acceleration reached in 0.5 seconds, clipping to max torque
    accel_up_0_5_sec = min(steer_up_per_frame * JERK_MEAS_T / DT_CTRL, 1.0) * max_lat_accel
    accel_down_0_5_sec = min(steer_down_per_frame * JERK_MEAS_T / DT_CTRL, 1.0) * max_lat_accel

    # Convert to m/s^3
    return accel_up_0_5_sec / JERK_MEAS_T, accel_down_0_5_sec / JERK_MEAS_T

  def test_jerk_limits(self):
    up_jerk, down_jerk = self.calculate_0_5s_jerk(self.control_params, self.torque_params)
    car_model_jerks[self.car_model] = {"up_jerk": up_jerk, "down_jerk": down_jerk}
    self.assertLessEqual(up_jerk, MAX_LAT_JERK_UP + MAX_LAT_JERK_UP_TOLERANCE)
    self.assertLessEqual(down_jerk, MAX_LAT_JERK_DOWN)

  def test_max_lateral_accel(self):
    self.assertLessEqual(self.torque_params["MAX_LAT_ACCEL_MEASURED"], MAX_LAT_ACCEL)


if __name__ == "__main__":
  result = unittest.main(exit=False)

  print(f"\n\n---- Lateral limit report ({len(CAR_MODELS)} cars) ----\n")

  max_car_model_len = max([len(car_model) for car_model in car_model_jerks])
  for car_model, _jerks in sorted(car_model_jerks.items(), key=lambda i: i[1]['up_jerk'], reverse=True):
    violation = _jerks["up_jerk"] > MAX_LAT_JERK_UP + MAX_LAT_JERK_UP_TOLERANCE or \
                _jerks["down_jerk"] > MAX_LAT_JERK_DOWN
    violation_str = " - VIOLATION" if violation else ""

    print(f"{car_model:{max_car_model_len}} - up jerk: {round(_jerks['up_jerk'], 2):5} " +
          f"m/s^3, down jerk: {round(_jerks['down_jerk'], 2):5} m/s^3{violation_str}")

  # exit with test result
  sys.exit(not result.result.wasSuccessful())
