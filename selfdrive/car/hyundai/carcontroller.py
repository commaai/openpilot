from cereal import car
from common.realtime import DT_CTRL
from selfdrive.car import apply_std_steer_torque_limits
from selfdrive.car.hyundai.hyundaican import create_lkas11, create_clu11, create_lfahda_mfc
from selfdrive.car.hyundai.values import Buttons, CarControllerParams, CAR
from opendbc.can.packer import CANPacker
from common.dp_common import common_controller_ctrl
from common.params import Params

VisualAlert = car.CarControl.HUDControl.VisualAlert


def process_hud_alert(enabled, fingerprint, visual_alert, left_lane,
                      right_lane, left_lane_depart, right_lane_depart):
  sys_warning = (visual_alert == VisualAlert.steerRequired)

  # initialize to no line visible
  sys_state = 1
  if left_lane and right_lane or sys_warning:  # HUD alert only display when LKAS status is active
    sys_state = 3 if enabled or sys_warning else 4
  elif left_lane:
    sys_state = 5
  elif right_lane:
    sys_state = 6

  # initialize to no warnings
  left_lane_warning = 0
  right_lane_warning = 0
  if left_lane_depart:
    left_lane_warning = 1 if fingerprint in [CAR.GENESIS_G90, CAR.GENESIS_G80] else 2
  if right_lane_depart:
    right_lane_warning = 1 if fingerprint in [CAR.GENESIS_G90, CAR.GENESIS_G80] else 2

  return sys_warning, sys_state, left_lane_warning, right_lane_warning


class CarController():
  def __init__(self, dbc_name, CP, VM):
    # dp
    self.last_blinker_on = False
    self.blinker_end_frame = 0.
    self.dp_hkg_smart_mdps = Params().get('dp_hkg_smart_mdps') == b'1'

    self.p = CarControllerParams(CP)
    self.packer = CANPacker(dbc_name)

    self.apply_steer_last = 0
    self.car_fingerprint = CP.carFingerprint
    self.steer_rate_limited = False
    self.last_resume_frame = 0

  def update(self, enabled, CS, frame, actuators, pcm_cancel_cmd, visual_alert,
             left_lane, right_lane, left_lane_depart, right_lane_depart, dragonconf):
    # Steering Torque
    new_steer = int(round(actuators.steer * self.p.STEER_MAX))
    apply_steer = apply_std_steer_torque_limits(new_steer, self.apply_steer_last, CS.out.steeringTorque, self.p)
    self.steer_rate_limited = new_steer != apply_steer

    # disable when temp fault is active
    lkas_active = enabled and not CS.out.steerWarning

    # fix for Genesis hard fault at low speed
    if not self.dp_hkg_smart_mdps and CS.out.vEgo < 16.7 and self.car_fingerprint == CAR.HYUNDAI_GENESIS:
      lkas_active = False

    if not lkas_active:
      apply_steer = 0

    # dp
    blinker_on = CS.out.leftBlinker or CS.out.rightBlinker
    if not enabled:
      self.blinker_end_frame = 0
    if self.last_blinker_on and not blinker_on:
      self.blinker_end_frame = frame + dragonconf.dpSignalOffDelay
    apply_steer = common_controller_ctrl(enabled,
                                         dragonconf,
                                         blinker_on or frame < self.blinker_end_frame,
                                         apply_steer, CS.out.vEgo)
    self.last_blinker_on = blinker_on

    self.apply_steer_last = apply_steer

    sys_warning, sys_state, left_lane_warning, right_lane_warning = \
      process_hud_alert(enabled, self.car_fingerprint, visual_alert,
                        left_lane, right_lane, left_lane_depart, right_lane_depart)

    can_sends = []
    can_sends.append(create_lkas11(self.packer, frame, self.car_fingerprint, apply_steer, lkas_active,
                                   CS.lkas11, sys_warning, sys_state, enabled,
                                   left_lane, right_lane,
                                   left_lane_warning, right_lane_warning))

    if pcm_cancel_cmd:
      can_sends.append(create_clu11(self.packer, frame, CS.clu11, Buttons.CANCEL))
    elif CS.out.cruiseState.standstill:
      # send resume at a max freq of 10Hz
      if (frame - self.last_resume_frame) * DT_CTRL > 0.1:
        # send 25 messages at a time to increases the likelihood of resume being accepted
        can_sends.extend([create_clu11(self.packer, frame, CS.clu11, Buttons.RES_ACCEL)] * 25)
        self.last_resume_frame = frame

    # 20 Hz LFA MFA message
    if frame % 5 == 0 and self.car_fingerprint in [CAR.SONATA, CAR.PALISADE, CAR.IONIQ, CAR.KIA_NIRO_EV, CAR.KONA_EV,
                                                   CAR.IONIQ_EV_2020, CAR.KIA_CEED, CAR.KIA_SELTOS, CAR.ELANTRA_2021]:
      can_sends.append(create_lfahda_mfc(self.packer, enabled))

    return can_sends
