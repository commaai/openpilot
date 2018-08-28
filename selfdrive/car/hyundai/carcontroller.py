from common.numpy_fast import interp
from selfdrive.car import apply_std_steer_torque_limits
from selfdrive.boardd.boardd import can_list_to_can_capnp
from selfdrive.car.hyundai.hyundaican import create_lkas11, create_lkas12, create_1191, create_1156
from selfdrive.car.hyundai.values import CAR
from selfdrive.can.packer import CANPacker


# Steer torque limits

class SteerLimitParams:
  STEER_MAX = 250   # 409 is the max
  STEER_DELTA_UP = 4
  STEER_DELTA_DOWN = 10
  STEER_DRIVER_ALLOWANCE = 50
  STEER_DRIVER_MULTIPLIER = 4
  STEER_DRIVER_FACTOR = 1

class CarController(object):
  def __init__(self, dbc_name, car_fingerprint, enable_camera):
    self.braking = False
    self.controls_allowed = True
    self.apply_steer_last = 0
    self.car_fingerprint = car_fingerprint
    self.angle_control = False
    self.lkas11_cnt = 0
    self.cnt = 0
    self.enable_camera = enable_camera

    self.packer = CANPacker(dbc_name)

  def update(self, sendcan, enabled, CS, actuators):

    if not self.enable_camera:
      return

    ### Steering Torque
    apply_steer = actuators.steer * SteerLimitParams.STEER_MAX

    apply_std_steer_torque_limits(apply_steer, self.apply_steer_last, CS.steer_torque_driver, SteerLimitParams)

    if not enabled:
      apply_steer = 0

    steer_req = 1 if enabled else 0

    self.apply_steer_last = apply_steer

    can_sends = []

    self.lkas11_cnt = self.cnt % 0x10

    can_sends.append(create_lkas11(self.packer, apply_steer, steer_req, self.lkas11_cnt, enabled))
    if (self.cnt % 10) == 0:
      can_sends.append(create_lkas12())
    if (self.cnt % 50) == 0:
      can_sends.append(create_1191())
    if (self.cnt % 7) == 0:
      can_sends.append(create_1156())

    ### Send messages to canbus
    sendcan.send(can_list_to_can_capnp(can_sends, msgtype='sendcan').to_bytes())

    self.cnt += 1
