from selfdrive.car import apply_std_steer_torque_limits
from selfdrive.car.hyundai.hyundaican import create_lkas11, create_lkas12, \
                                             create_1191, create_1156, \
                                             create_clu11
from selfdrive.car.hyundai.values import Buttons
from selfdrive.can.packer import CANPacker


# Steer torque limits

class SteerLimitParams:
  STEER_MAX = 255   # 409 is the max, 255 is stock
  STEER_DELTA_UP = 3
  STEER_DELTA_DOWN = 7
  STEER_DRIVER_ALLOWANCE = 50
  STEER_DRIVER_MULTIPLIER = 2
  STEER_DRIVER_FACTOR = 1

class CarController(object):
  def __init__(self, dbc_name, car_fingerprint):
    self.apply_steer_last = 0
    self.car_fingerprint = car_fingerprint
    self.lkas11_cnt = 0
    self.clu11_cnt = 0
    self.cnt = 0
    self.last_resume_cnt = 0
    self.last_lead_distance = 0
    # True when giraffe switch 2 is low and we need to replace all the camera messages
    # otherwise we forward the camera msgs and we just replace the lkas cmd signals
    self.camera_disconnected = False

    self.packer = CANPacker(dbc_name)

  def update(self, enabled, CS, actuators, pcm_cancel_cmd, hud_alert):

    ### Steering Torque
    apply_steer = actuators.steer * SteerLimitParams.STEER_MAX

    apply_steer = apply_std_steer_torque_limits(apply_steer, self.apply_steer_last, CS.steer_torque_driver, SteerLimitParams)

    if not enabled:
      apply_steer = 0

    steer_req = 1 if enabled else 0

    self.apply_steer_last = apply_steer

    can_sends = []

    self.lkas11_cnt = self.cnt % 0x10

    if self.camera_disconnected:
      if (self.cnt % 10) == 0:
        can_sends.append(create_lkas12())
      if (self.cnt % 50) == 0:
        can_sends.append(create_1191())
      if (self.cnt % 7) == 0:
        can_sends.append(create_1156())

    can_sends.append(create_lkas11(self.packer, self.car_fingerprint, apply_steer, steer_req, self.lkas11_cnt,
                                   enabled, CS.lkas11, hud_alert, keep_stock=(not self.camera_disconnected)))

    if pcm_cancel_cmd:
      self.clu11_cnt = self.cnt % 0x10
      can_sends.append(create_clu11(self.packer, CS.clu11, Buttons.CANCEL, self.clu11_cnt))

    if CS.stopped:
      # run only first time when the car stopped
      if self.last_lead_distance == 0:
        # get the lead distance from the Radar
        self.last_lead_distance = CS.lead_distance
        self.clu11_cnt = 0
      # when lead car starts moving, create 6 RES msgs
      elif CS.lead_distance > self.last_lead_distance and (self.cnt - self.last_resume_cnt) > 5:
        can_sends.append(create_clu11(self.packer, CS.clu11, Buttons.RES_ACCEL, self.clu11_cnt))
        self.clu11_cnt += 1
        # interval after 6 msgs
        if self.clu11_cnt > 5:
          self.last_resume_cnt = self.cnt
          self.clu11_cnt = 0
    elif self.last_lead_distance != 0:
      self.last_lead_distance = 0  


    self.cnt += 1

    return can_sends
