#!/usr/bin/env python3
import argparse
import random
import unittest
from parameterized import parameterized, parameterized_class
# from selfdrive.car.hyundai.values import CarControllerParams
import importlib


from cereal import car
from selfdrive.car.car_helpers import get_interface_attr, interfaces
from selfdrive.car.fingerprints import FW_VERSIONS
from selfdrive.car.fw_versions import FW_QUERY_CONFIGS, match_fw_to_car
from selfdrive.car.fingerprints import all_known_cars
from selfdrive.car.interfaces import get_torque_params
from collections import defaultdict


MAX_LAT_ACCEL = 3  # m/s^2
MAX_LAT_UP_JERK = 3  # m/s^2
MAX_LAT_DOWN_JERK = 3  # m/s^2

jerks = defaultdict(dict)


@parameterized_class('car_model', [(c,) for c in all_known_cars()])
class TestLateralLimits(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    CarInterface, _, _ = interfaces[cls.car_model]
    CP = CarInterface.get_params(cls.car_model)

    if CP.dashcamOnly:
      raise unittest.SkipTest("Platform is behind dashcamOnly")

    # TODO: test these
    if CP.carName in ("honda", "nissan", "body"):
      raise unittest.SkipTest("No steering safety")

    CarControllerParams = importlib.import_module(f'selfdrive.car.{CP.carName}.values').CarControllerParams
    cls.control_params = CarControllerParams(CP)

    cls.torque_params = get_torque_params(cls.car_model)

  @classmethod
  def tearDownClass(cls):
    for car, _jerks in jerks.items():
      print(f'{car:37} - up jerk: {round(_jerks["up_jerk"], 2)} m/s^3, down jerk: {round(_jerks["down_jerk"], 2)} m/s^3')
    print()
    # print(dict(jerks))

  def _calc_jerk(self):
    # TODO: some cars don't send at 100hz, put steer rate/step into CCP to calculate this properly
    steer_step = self.control_params.STEER_STEP
    time_to_max = self.control_params.STEER_MAX / self.control_params.STEER_DELTA_UP / 100. * steer_step
    time_to_min = self.control_params.STEER_MAX / self.control_params.STEER_DELTA_DOWN / 100. * steer_step
    max_lat_accel = self.torque_params['LAT_ACCEL_FACTOR']

    return max_lat_accel / time_to_max, max_lat_accel / time_to_min

  def test_jerk_limits(self):
    up_jerk, down_jerk = self._calc_jerk()
    jerks[self.car_model] = {"up_jerk": up_jerk, "down_jerk": down_jerk}
    self.assertLess(up_jerk, 2.5)
    # self.assertGreater(down_jerk, 2.0)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--print-jerks", action="store_true", help="Print theoretical max lateral jerk values for all platforms")

  args = parser.parse_args()

  if args.print_jerks:
    # TODO: would be nice to support this
    pass
  else:
    unittest.main()
