from cereal import car
from selfdrive.car import apply_std_steer_torque_limits
from selfdrive.car.hyundai.hyundaican import create_lkas11, create_lkas12, \
                                             create_1191, create_1156, \
                                             create_clu11
from selfdrive.car.hyundai.values import CAR, Buttons
from selfdrive.can.packer import CANPacker


# Steer torque limits

class SteerLimitParams:
  STEER_MAX = 255   # 409 is the max, 255 is stock
  STEER_DELTA_UP = 3
  STEER_DELTA_DOWN = 7
  STEER_DRIVER_ALLOWANCE = 50
  STEER_DRIVER_MULTIPLIER = 2
  STEER_DRIVER_FACTOR = 1

VisualAlert = car.CarControl.HUDControl.VisualAlert

def process_hud_alert(visual_alert, car_fingerprint):
  if visual_alert == VisualAlert.steerRequired:
    return 5 if car_fingerprint in [CAR.SANTA_FE, CAR.SANTA_FE_1] else 4
  else:
    return 0

def process_lane_visible(enabled, left_line, right_line, hud_alert):
  # initialize to no line visible
  lane_visible = 1
  if left_line and right_line or hud_alert:
    if enabled or hud_alert:
      lane_visible = 3
    else:
      lane_visible = 4
  elif left_line:
    lane_visible = 5
  elif right_line:
    lane_visible = 6

  return lane_visible

class CarController(object):
  def __init__(self, dbc_name, car_fingerprint):
    self.apply_steer_last = 0
    self.car_fingerprint = car_fingerprint
    self.lkas11_cnt = 0
    self.clu11_cnt = 0
    self.last_resume_frame = 0
    self.last_lead_distance = 0
    # True when giraffe switch 2 is low and we need to replace all the camera messages
    # otherwise we forward the camera msgs and we just replace the lkas cmd signals
    self.camera_disconnected = False

    self.packer = CANPacker(dbc_name)

  def update(self, enabled, CS, frame, actuators, pcm_cancel_cmd, visual_alert,
              left_line, right_line, left_lane_depart, right_lane_depart):

    ### Steering Torque
    apply_steer = actuators.steer * SteerLimitParams.STEER_MAX

    apply_steer = apply_std_steer_torque_limits(apply_steer, self.apply_steer_last, CS.steer_torque_driver, SteerLimitParams)

    if not enabled:
      apply_steer = 0

    steer_req = 1 if enabled else 0

    self.apply_steer_last = apply_steer

    hud_alert = process_hud_alert(visual_alert, self.car_fingerprint)
    lane_visible = process_lane_visible(enabled, left_line, right_line, hud_alert)

    can_sends = []

    self.lkas11_cnt = frame % 0x10

    if self.camera_disconnected:
      if (frame % 10) == 0:
        can_sends.append(create_lkas12())
      if (frame % 50) == 0:
        can_sends.append(create_1191())
      if (frame % 7) == 0:
        can_sends.append(create_1156())

    can_sends.append(create_lkas11(self.packer, self.car_fingerprint, apply_steer, steer_req, self.lkas11_cnt,
                                   enabled, CS.lkas11, hud_alert, lane_visible, left_lane_depart, right_lane_depart,
                                   keep_stock=(not self.camera_disconnected)))

    if pcm_cancel_cmd:
      self.clu11_cnt = frame % 0x10
      can_sends.append(create_clu11(self.packer, CS.clu11, Buttons.CANCEL, self.clu11_cnt))

    if CS.stopped:
      # run only first time when the car stopped
      if self.last_lead_distance == 0:
        # get the lead distance from the Radar
        self.last_lead_distance = CS.lead_distance
        self.clu11_cnt = 0
      # when lead car starts moving, create 6 RES msgs
      elif CS.lead_distance > self.last_lead_distance and (frame - self.last_resume_frame) > 5:
        can_sends.append(create_clu11(self.packer, CS.clu11, Buttons.RES_ACCEL, self.clu11_cnt))
        self.clu11_cnt += 1
        # interval after 6 msgs
        if self.clu11_cnt > 5:
          self.last_resume_frame = frame
          self.clu11_cnt = 0
    # reset lead distnce after the car starts moving
    elif self.last_lead_distance != 0:
      self.last_lead_distance = 0  


    return can_sends
