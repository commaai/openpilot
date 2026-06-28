import numpy as np
import pytest

from openpilot.common.parameterized import parameterized

from openpilot.cereal import log
from opendbc.car.structs import car
from opendbc.car.car_helpers import interfaces
from opendbc.car.toyota.values import CAR as TOYOTA
from opendbc.car.vehicle_model import VehicleModel
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.latcontrol_torque import KP, KP_INTERP, INTERP_SPEEDS, LatControlTorque, LAT_ACCEL_REQUEST_BUFFER_SECONDS

def get_controller(car_name):
  CarInterface = interfaces[car_name]
  CP = CarInterface.get_non_essential_params(car_name)
  CI = CarInterface(CP)
  VM = VehicleModel(CP)
  controller = LatControlTorque(CP.as_reader(), CI, DT_CTRL)
  return controller, VM

class TestLatControlTorqueBuffer:

  @parameterized.expand([(TOYOTA.TOYOTA_COROLLA_TSS2,)])
  def test_request_buffer_consistency(self, car_name):
    buffer_steps = int(LAT_ACCEL_REQUEST_BUFFER_SECONDS / DT_CTRL)
    controller, VM = get_controller(car_name)

    CS = car.CarState.new_message()
    CS.vEgo = 30
    CS.steeringPressed = False
    params = log.LiveParametersData.new_message()

    for _ in range(buffer_steps):
      controller.update(True, CS, VM, params, False, 0.001, False, 0.2)
    assert all(val != 0 for val in controller.lat_accel_request_buffer)

    for _ in range(buffer_steps):
      controller.update(False, CS, VM, params, False, 0.0, False, 0.2)
    assert all(val == 0 for val in controller.lat_accel_request_buffer)

  @parameterized.expand([(TOYOTA.TOYOTA_COROLLA_TSS2,)])
  def test_low_speed_kp_logged(self, car_name):
    controller, VM = get_controller(car_name)

    CS = car.CarState.new_message()
    CS.vEgo = 2.0
    CS.steeringPressed = False
    params = log.LiveParametersData.new_message()

    _, _, pid_log = controller.update(True, CS, VM, params, False, 0.001, False, 0.2)
    assert pid_log.kp == pytest.approx(np.interp(CS.vEgo, INTERP_SPEEDS, KP_INTERP))

  @parameterized.expand([(TOYOTA.TOYOTA_COROLLA_TSS2,)])
  def test_high_speed_kp_logged(self, car_name):
    controller, VM = get_controller(car_name)

    CS = car.CarState.new_message()
    CS.vEgo = 30.0
    CS.steeringPressed = False
    params = log.LiveParametersData.new_message()

    _, _, pid_log = controller.update(True, CS, VM, params, False, 0.001, False, 0.2)
    assert pid_log.kp == pytest.approx(KP)

  @parameterized.expand([(TOYOTA.TOYOTA_COROLLA_TSS2,)])
  def test_inactive_kp_default_serializes(self, car_name):
    controller, VM = get_controller(car_name)

    CS = car.CarState.new_message()
    CS.vEgo = 2.0
    CS.steeringPressed = False
    params = log.LiveParametersData.new_message()

    _, _, pid_log = controller.update(False, CS, VM, params, False, 0.001, False, 0.2)
    assert pid_log.kp == 0.0

    serialized = pid_log.to_bytes()
    with log.ControlsState.LateralTorqueState.from_bytes(serialized) as deserialized:
      assert deserialized.kp == 0.0
