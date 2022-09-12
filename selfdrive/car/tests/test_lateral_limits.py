#!/usr/bin/env python3
import random
import unittest
from parameterized import parameterized, parameterized_class
from selfdrive.car.hyundai.values import CarControllerParams


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
    # TODO: remove these once it works with all platforms
    if not cls.car_model.startswith(('KIA', 'HYUNDAI', 'GENESIS')):
      raise unittest.SkipTest

    # if not cls.car_model.startswith('KIA EV6'):
    #   raise unittest.SkipTest

    CarInterface, _, _ = interfaces[cls.car_model]
    CP = CarInterface.get_params(cls.car_model)
    # TODO: just make all CCPs take CarParams
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
    time_to_max = self.control_params.STEER_MAX / self.control_params.STEER_DELTA_UP / 100.
    time_to_min = self.control_params.STEER_MAX / self.control_params.STEER_DELTA_DOWN / 100.
    # TODO: fix this
    max_lat_accel = (self.torque_params['LAT_ACCEL_FACTOR'] + self.torque_params['MAX_LAT_ACCEL_MEASURED']) / 2.

    return max_lat_accel / time_to_max, max_lat_accel / time_to_min

  def test_something(self):
    up_jerk, down_jerk = self._calc_jerk()
    jerks[self.car_model] = {"up_jerk": up_jerk, "down_jerk": down_jerk}
    # self.assertLess(up_jerk, 2.0)
    # self.assertLess(down_jerk, 2.5)


if __name__ == "__main__":
  unittest.main()
