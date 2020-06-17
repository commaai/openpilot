from cereal import car
from common.numpy_fast import clip
from selfdrive.car import apply_std_steer_torque_limits
from selfdrive.car.hyundai.hyundaican import create_lkas11, create_clu11, \
                                             create_scc12, create_mdps12
from selfdrive.car.hyundai.values import Buttons, SteerLimitParams, CAR
from opendbc.can.packer import CANPacker

VisualAlert = car.CarControl.HUDControl.VisualAlert

# Accel limits
ACCEL_HYST_GAP = 0.02  # don't change accel command for small oscilalitons within this value
ACCEL_MAX = 1.5  # 1.5 m/s2
ACCEL_MIN = -3.0 # 3   m/s2
ACCEL_SCALE = max(ACCEL_MAX, -ACCEL_MIN)

def accel_hysteresis(accel, accel_steady):

  # for small accel oscillations within ACCEL_HYST_GAP, don't change the accel command
  if accel > accel_steady + ACCEL_HYST_GAP:
    accel_steady = accel - ACCEL_HYST_GAP
  elif accel < accel_steady - ACCEL_HYST_GAP:
    accel_steady = accel + ACCEL_HYST_GAP
  accel = accel_steady

  return accel, accel_steady

def process_hud_alert(enabled, button_on, fingerprint, visual_alert, left_line,
                       right_line, left_lane_depart, right_lane_depart):
  hud_alert = 0
  if visual_alert == VisualAlert.steerRequired:
    hud_alert = 3

  # initialize to no line visible
  
  lane_visible = 1
  if not button_on:
    lane_visible = 0
  elif left_line and right_line or hud_alert: #HUD alert only display when LKAS status is active
    if enabled or hud_alert:
      lane_visible = 3
    else:
      lane_visible = 4
  elif left_line:
    lane_visible = 5
  elif right_line:
    lane_visible = 6

  # initialize to no warnings
  left_lane_warning = 0
  right_lane_warning = 0
  if left_lane_depart:
    left_lane_warning = 1 if fingerprint in [CAR.GENESIS , CAR.GENESIS_G90, CAR.GENESIS_G80] else 2
  if right_lane_depart:
    right_lane_warning = 1 if fingerprint in [CAR.GENESIS , CAR.GENESIS_G90, CAR.GENESIS_G80] else 2

  return hud_alert, lane_visible, left_lane_warning, right_lane_warning

class CarController():
  def __init__(self, dbc_name, car_fingerprint):
    self.packer = CANPacker(dbc_name)
    self.car_fingerprint = car_fingerprint
    self.accel_steady = 0
    self.apply_steer_last = 0
    self.steer_rate_limited = False
    self.lkas11_cnt = 0
    self.scc12_cnt = 0
    self.resume_cnt = 0
    self.last_resume_frame = 0
    self.last_lead_distance = 0
    self.turning_signal_timer = 0
    self.lkas_button = 1
    self.lkas_button_last = 0
    self.longcontrol = 1 #TODO: make auto

  def update(self, enabled, CS, frame, actuators, pcm_cancel_cmd, visual_alert,
              left_line, right_line, left_lane_depart, right_lane_depart):

    # *** compute control surfaces ***

    # gas and brake
    apply_accel = actuators.gas - actuators.brake

    apply_accel, self.accel_steady = accel_hysteresis(apply_accel, self.accel_steady)
    apply_accel = clip(apply_accel * ACCEL_SCALE, ACCEL_MIN, ACCEL_MAX)

    ### Steering Torque
    new_steer = actuators.steer * SteerLimitParams.STEER_MAX
    apply_steer = apply_std_steer_torque_limits(new_steer, self.apply_steer_last, CS.steer_torque_driver, SteerLimitParams)
    self.steer_rate_limited = new_steer != apply_steer

    ### LKAS button to temporarily disable steering
    if not CS.lkas_error:
      if CS.lkas_button_on != self.lkas_button_last:
        self.lkas_button = not self.lkas_button
      self.lkas_button_last = CS.lkas_button_on

    # disable if steer angle reach 90 deg, otherwise mdps fault in some models
    lkas_active = enabled and abs(CS.angle_steers) < 90. and self.lkas_button

    # fix for Genesis hard fault at low speed
    if CS.v_ego < 16.7 and self.car_fingerprint == CAR.GENESIS and not CS.mdps_bus:
      lkas_active = 0

    # Disable steering while turning blinker on and speed below 60 kph
    if CS.left_blinker_on or CS.right_blinker_on:
      if self.car_fingerprint not in [CAR.KIA_OPTIMA, CAR.KIA_OPTIMA_H]:
        self.turning_signal_timer = 100  # Disable for 1.0 Seconds after blinker turned off
      elif CS.left_blinker_flash or CS.right_blinker_flash: # Optima has blinker flash signal only
        self.turning_signal_timer = 100
    if self.turning_signal_timer and CS.v_ego < 16.7:
      lkas_active = 0
    if self.turning_signal_timer:
      self.turning_signal_timer -= 1
    if not lkas_active:
      apply_steer = 0

    steer_req = 1 if apply_steer else 0

    self.apply_accel_last = apply_accel
    self.apply_steer_last = apply_steer

    hud_alert, lane_visible, left_lane_warning, right_lane_warning =\
            process_hud_alert(lkas_active, self.lkas_button, self.car_fingerprint, visual_alert,
            left_line, right_line,left_lane_depart, right_lane_depart)

    clu11_speed = CS.clu11["CF_Clu_Vanz"]
    enabled_speed = 38 if CS.is_set_speed_in_mph  else 60
    if clu11_speed > enabled_speed or not lkas_active:
      enabled_speed = clu11_speed

    can_sends = []

    if frame == 0: # initialize counts from last received count signals
      self.lkas11_cnt = CS.lkas11["CF_Lkas_MsgCount"] + 1
      self.scc12_cnt = CS.scc12["CR_VSM_Alive"] + 1 if not CS.no_radar else 0

    self.lkas11_cnt %= 0x10
    self.scc12_cnt %= 0xF
    self.clu11_cnt = frame % 0x10
    self.mdps12_cnt = frame % 0x100

    can_sends.append(create_lkas11(self.packer, self.car_fingerprint, 0, apply_steer, steer_req, self.lkas11_cnt, lkas_active,
                                   CS.lkas11, hud_alert, lane_visible, left_lane_depart, right_lane_depart, keep_stock=True))
    if CS.mdps_bus or CS.scc_bus == 1: # send lkas12 bus 1 if mdps or scc is on bus 1
      can_sends.append(create_lkas11(self.packer, self.car_fingerprint, 1, apply_steer, steer_req, self.lkas11_cnt, lkas_active,
                                   CS.lkas11, hud_alert, lane_visible, left_lane_depart, right_lane_depart, keep_stock=True))
    if CS.mdps_bus: # send clu11 to mdps if it is not on bus 0
      can_sends.append(create_clu11(self.packer, CS.mdps_bus, CS.clu11, Buttons.NONE, enabled_speed, self.clu11_cnt))

    if pcm_cancel_cmd and self.longcontrol:
      can_sends.append(create_clu11(self.packer, CS.scc_bus, CS.clu11, Buttons.CANCEL, clu11_speed, self.clu11_cnt))
    else: # send mdps12 to LKAS to prevent LKAS error if no cancel cmd
      can_sends.append(create_mdps12(self.packer, self.car_fingerprint, self.mdps12_cnt, CS.mdps12))

    if CS.scc_bus and self.longcontrol and frame % 2: # send scc12 to car if SCC not on bus 0 and longcontrol enabled
      can_sends.append(create_scc12(self.packer, apply_accel, enabled, self.scc12_cnt, CS.scc12))
      self.scc12_cnt += 1

    if CS.stopped:
      # run only first time when the car stopped
      if self.last_lead_distance == 0:
        # get the lead distance from the Radar
        self.last_lead_distance = CS.lead_distance
        self.resume_cnt = 0
      # when lead car starts moving, create 6 RES msgs
      elif CS.lead_distance > self.last_lead_distance and (frame - self.last_resume_frame) > 5:
        can_sends.append(create_clu11(self.packer, CS.scc_bus, CS.clu11, Buttons.RES_ACCEL, clu11_speed, self.clu11_cnt))
        self.resume_cnt += 1
        # interval after 6 msgs
        if self.resume_cnt > 5:
          self.last_resume_frame = frame
          self.clu11_cnt = 0
    # reset lead distnce after the car starts moving
    elif self.last_lead_distance != 0:
      self.last_lead_distance = 0  

    self.lkas11_cnt += 1

    return can_sends
