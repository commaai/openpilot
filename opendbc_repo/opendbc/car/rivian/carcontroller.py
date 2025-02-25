from opendbc.can.packer import CANPacker
from opendbc.car import Bus, apply_driver_steer_torque_limits
from opendbc.car.interfaces import CarControllerBase
from opendbc.car.rivian.riviancan import create_lka_steering, create_longitudinal, create_wheel_touch
from opendbc.car.rivian.values import CarControllerParams


class CarController(CarControllerBase):
  def __init__(self, dbc_names, CP):
    super().__init__(dbc_names, CP)
    self.apply_steer_last = 0
    self.packer = CANPacker(dbc_names[Bus.pt])

  def update(self, CC, CS, now_nanos):
    actuators = CC.actuators
    can_sends = []

    apply_steer = 0
    if CC.latActive:
      new_steer = int(round(CC.actuators.steer * CarControllerParams.STEER_MAX))
      apply_steer = apply_driver_steer_torque_limits(new_steer, self.apply_steer_last,
                                                     CS.out.steeringTorque, CarControllerParams)

    # send steering command
    self.apply_steer_last = apply_steer
    can_sends.append(create_lka_steering(self.packer, CS.acm_lka_hba_cmd, apply_steer, CC.latActive))

    if self.frame % 5 == 0:
      can_sends.append(create_wheel_touch(self.packer, CS.sccm_wheel_touch, CC.enabled))

    # Longitudinal control
    if self.CP.openpilotLongitudinalControl:
      can_sends.append(create_longitudinal(self.packer, self.frame % 15, actuators.accel, CC.enabled))

    new_actuators = actuators.as_builder()
    new_actuators.steer = apply_steer / CarControllerParams.STEER_MAX
    new_actuators.steerOutputCan = apply_steer

    self.frame += 1
    return new_actuators, can_sends
