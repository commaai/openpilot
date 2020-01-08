from selfdrive.car import apply_std_steer_torque_limits
from selfdrive.car.hyundai.hyundaican import create_lkas11, create_clu11
from selfdrive.car.hyundai.values import Buttons, SteerLimitParams
from opendbc.can.packer import CANPacker


class CarController():
  def __init__(self, dbc_name, car_fingerprint):
    self.apply_steer_last = 0
    self.car_fingerprint = car_fingerprint
    self.lkas11_cnt = 0
    self.cnt = 0
    self.last_resume_cnt = 0
    self.packer = CANPacker(dbc_name)
    self.steer_rate_limited = False

  def update(self, enabled, CS, actuators, pcm_cancel_cmd, hud_alert):

    ### Steering Torque
    new_steer = actuators.steer * SteerLimitParams.STEER_MAX
    apply_steer = apply_std_steer_torque_limits(new_steer, self.apply_steer_last, CS.steer_torque_driver, SteerLimitParams)
    self.steer_rate_limited = new_steer != apply_steer

    if not enabled:
      apply_steer = 0

    steer_req = 1 if enabled else 0

    self.apply_steer_last = apply_steer

    can_sends = []

    self.lkas11_cnt = self.cnt % 0x10
    self.clu11_cnt = self.cnt % 0x10

    can_sends.append(create_lkas11(self.packer, self.car_fingerprint, apply_steer, steer_req, self.lkas11_cnt,
                                   enabled, CS.lkas11, hud_alert, keep_stock=True))

    if pcm_cancel_cmd:
      can_sends.append(create_clu11(self.packer, CS.clu11, Buttons.CANCEL))
    elif CS.stopped and (self.cnt - self.last_resume_cnt) > 5:
      self.last_resume_cnt = self.cnt
      can_sends.append(create_clu11(self.packer, CS.clu11, Buttons.RES_ACCEL))

    self.cnt += 1

    return can_sends
