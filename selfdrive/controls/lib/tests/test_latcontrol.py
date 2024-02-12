#!/usr/bin/env python3
import unittest

from parameterized import parameterized

from cereal import car, log
from openpilot.selfdrive.car.car_helpers import interfaces
from openpilot.selfdrive.car.honda.values import CAR as HONDA
from openpilot.selfdrive.car.toyota.values import CAR as TOYOTA
from openpilot.selfdrive.car.nissan.values import CAR as NISSAN
from openpilot.selfdrive.controls.lib.latcontrol_pid import LatControlPID
from openpilot.selfdrive.controls.lib.latcontrol_torque import LatControlTorque
from openpilot.selfdrive.controls.lib.latcontrol_angle import LatControlAngle
from openpilot.selfdrive.controls.lib.vehicle_model import VehicleModel
from openpilot.common.mock.generators import generate_liveLocationKalman


class TestLatControl(unittest.TestCase):

  @parameterized.expand([(HONDA.CIVIC, LatControlPID), (TOYOTA.RAV4, LatControlTorque),  (NISSAN.LEAF, LatControlAngle)])
  def test_saturation(self, car_name, controller):
    CarInterface, CarController, CarState = interfaces[car_name]
    CP = CarInterface.get_non_essential_params(car_name)
    CI = CarInterface(CP, CarController, CarState)
    VM = VehicleModel(CP)

    controller = controller(CP, CI)

    CS = car.CarState.new_message()
    CS.vEgo = 30
    CS.steeringPressed = False

    params = log.LiveParametersData.new_message()

    llk = generate_liveLocationKalman()
    for _ in range(1000):
      _, _, lac_log = controller.update(True, CS, VM, params, False, 1, llk)

    self.assertTrue(lac_log.saturated)


if __name__ == "__main__":
  unittest.main()
