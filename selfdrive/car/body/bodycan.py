import copy
from selfdrive.car.body.values import CAR


def create_control(packer, torque_l, torque_r):
  can_bus = 1

  values = {
    "TORQUE_L": torque_l,
    "TORQUE_R": torque_r,
  }
  return packer.make_can_msg("BODY_COMMAND", can_bus, values)
