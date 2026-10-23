from opendbc.can.packer import CANPacker
from opendbc.car import Bus
from opendbc.car.lateral import apply_steer_angle_limits_vm
from opendbc.car.interfaces import CarControllerBase
from opendbc.car.byd import bydcan
from opendbc.car.byd.values import CarControllerParams
from opendbc.car.vehicle_model import VehicleModel


def get_safety_CP():
  from opendbc.car.byd.interface import CarInterface
  return CarInterface.get_non_essential_params("BYD_ATTO_3")


class CarController(CarControllerBase):
  def __init__(self, dbc_names, CP):
    super().__init__(dbc_names, CP)
    self.packer = CANPacker(dbc_names[Bus.pt])
    self.apply_angle_last = 0.0

    # Vehicle model used for lateral limiting
    self.VM = VehicleModel(get_safety_CP())

  def update(self, CC, CS, now_nanos):
    can_sends = []
    actuators = CC.actuators
    hud_control = CC.hudControl

    if self.frame % 2:
      self.apply_angle_last = apply_steer_angle_limits_vm(actuators.steeringAngleDeg, self.apply_angle_last, CS.out.vEgoRaw,
                                                          CS.out.steeringAngleDeg, CC.latActive, CarControllerParams, self.VM)

      cntr = (self.frame // 2) % 16
      can_sends.append(bydcan.create_steering_control(self.packer, self.apply_angle_last, CC.latActive, cntr))
      can_sends.append(bydcan.create_lkas_hud(self.packer, CC.latActive, cntr, CS.lkas_hud, hud_control))

    if CC.cruiseControl.cancel and self.frame % 10 == 0:
      can_sends.append(bydcan.create_buttons(self.packer, cancel=True))

    new_actuators = actuators.as_builder()
    new_actuators.steeringAngleDeg = float(self.apply_angle_last)

    self.frame += 1
    return new_actuators, can_sends
