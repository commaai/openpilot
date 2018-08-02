import os
import subprocess
from  threading import Thread
import traceback
import shlex
from collections import namedtuple
from selfdrive.boardd.boardd import can_list_to_can_capnp
from selfdrive.controls.lib.drive_helpers import rate_limit
from common.numpy_fast import clip, interp
from selfdrive.car.tesla import teslacan
from selfdrive.car.tesla.values import AH, CruiseButtons, CAR
from selfdrive.can.packer import CANPacker
from selfdrive.config import Conversions as CV

import time


# Steer angle limits
ANGLE_MAX_BP = [0., 27., 36.]
ANGLE_MAX_V = [410., 92., 36.]

ANGLE_DELTA_BP = [0., 5., 15.]
ANGLE_DELTA_V = [5., .8, .15]     # windup limit
ANGLE_DELTA_VU = [5., 3.5, 0.4]   # unwind limit

def actuator_hystereses(brake, braking, brake_steady, v_ego, car_fingerprint):
  # hyst params... TODO: move these to VehicleParams
  brake_hyst_on = 0.02     # to activate brakes exceed this value
  brake_hyst_off = 0.005   # to deactivate brakes below this value
  brake_hyst_gap = 0.01    # don't change brake command for small ocilalitons within this value

  #*** histeresys logic to avoid brake blinking. go above 0.1 to trigger
  if (brake < brake_hyst_on and not braking) or brake < brake_hyst_off:
    brake = 0.
  braking = brake > 0.

  # for small brake oscillations within brake_hyst_gap, don't change the brake command
  if brake == 0.:
    brake_steady = 0.
  elif brake > brake_steady + brake_hyst_gap:
    brake_steady = brake - brake_hyst_gap
  elif brake < brake_steady - brake_hyst_gap:
    brake_steady = brake + brake_hyst_gap

  return brake, braking, brake_steady


def process_hud_alert(hud_alert):
  # initialize to no alert
  fcw_display = 0
  steer_required = 0
  acc_alert = 0
  if hud_alert == AH.NONE:          # no alert
    pass
  elif hud_alert == AH.FCW:         # FCW
    fcw_display = hud_alert[1]
  elif hud_alert == AH.STEER:       # STEER
    steer_required = hud_alert[1]
  else:                             # any other ACC alert
    acc_alert = hud_alert[1]

  return fcw_display, steer_required, acc_alert


HUDData = namedtuple("HUDData",
                     ["pcm_accel", "v_cruise", "mini_car", "car", "X4",
                      "lanes", "beep", "chime", "fcw", "acc_alert", "steer_required"])


class CarController(object):
  def __init__(self, dbc_name, enable_camera=True):
    self.braking = False
    self.brake_steady = 0.
    self.brake_last = 0.
    self.accelerating = False
    self.accel_steady = 0.
    self.accel_last = 0.
    self.enable_camera = enable_camera
    self.packer = CANPacker(dbc_name)
    self.epas_disabled = True
    self.human_cruise_action_time = 0
    self.automated_cruise_action_time = 0
    self.last_angle = 0.

  def update(self, sendcan, enabled, CS, frame, actuators, \
             pcm_speed, pcm_override, pcm_cancel_cmd, pcm_accel, \
             hud_v_cruise, hud_show_lanes, hud_show_car, hud_alert, \
             snd_beep, snd_chime):

    """ Controls thread """

    ## Todo add code to detect Tesla DAS (camera) and go into listen and record mode only (for AP1 / AP2 cars)
    if not self.enable_camera:
      return

    # *** apply brake hysteresis ***
    brake, self.braking, self.brake_steady = actuator_hystereses(actuators.brake, self.braking, self.brake_steady, CS.v_ego, CS.CP.carFingerprint)
    accel, self.accelerating, self.accel_steady =  actuator_hystereses(actuators.gas, self.accelerating, self.accel_steady, CS.v_ego, CS.CP.carFingerprint)
    # *** no output if not enabled ***
    if not enabled and CS.pcm_acc_status:
      # send pcm acc cancel cmd if drive is disabled but pcm is still on, or if the system can't be activated
      pcm_cancel_cmd = True

    # *** rate limit after the enable check ***
    self.brake_last = rate_limit(brake, self.brake_last, -2., 1./100)

    # vehicle hud display, wait for one update from 10Hz 0x304 msg
    if hud_show_lanes:
      hud_lanes = 1
    else:
      hud_lanes = 0

    # TODO: factor this out better
    if enabled:
      if hud_show_car:
        hud_car = 2
      else:
        hud_car = 1
    else:
      hud_car = 0
    
    # For lateral control-only, send chimes as a beep since we don't send 0x1fa
    #if CS.CP.radarOffCan:

    #print chime, alert_id, hud_alert
    fcw_display, steer_required, acc_alert = process_hud_alert(hud_alert)

    hud = HUDData(int(pcm_accel), int(round(hud_v_cruise)), 1, hud_car,
                  0xc1, hud_lanes, int(snd_beep), snd_chime, fcw_display, acc_alert, steer_required)
 
    if not all(isinstance(x, int) and 0 <= x < 256 for x in hud):
      print "INVALID HUD", hud
      hud = HUDData(0xc6, 255, 64, 0xc0, 209, 0x40, 0, 0, 0, 0)

    # **** process the car messages ****

    # *** compute control surfaces ***

    STEER_MAX = 420
    # Prevent steering while stopped
    MIN_STEERING_VEHICLE_VELOCITY = 0.05 # m/s
    vehicle_moving = (CS.v_ego >= MIN_STEERING_VEHICLE_VELOCITY)
    
    # Basic highway lane change logic
    changing_lanes = CS.right_blinker_on or CS.left_blinker_on

    enable_steer_control = (enabled and not changing_lanes)
    
    # Angle
    apply_angle = -actuators.steerAngle
    angle_lim = interp(CS.v_ego, ANGLE_MAX_BP, ANGLE_MAX_V)
    apply_angle = clip(apply_angle, -angle_lim, angle_lim)
    # windup slower
    if self.last_angle * apply_angle > 0. and abs(apply_angle) > abs(self.last_angle):
      angle_rate_lim = interp(CS.v_ego, ANGLE_DELTA_BP, ANGLE_DELTA_V)
    else:
      angle_rate_lim = interp(CS.v_ego, ANGLE_DELTA_BP, ANGLE_DELTA_VU)

    apply_angle = clip(apply_angle, self.last_angle - angle_rate_lim, self.last_angle + angle_rate_lim)
    #if blinker is on send the actual angle
    if (changing_lanes):
      apply_angle = CS.angle_steers
    # Send CAN commands.
    can_sends = []
    send_step = 5

    if  (True): #(frame % send_step) == 0:
      idx = frame % 16 #(frame/send_step) % 16 
      can_sends.append(teslacan.create_steering_control(enable_steer_control, apply_angle, idx))
      can_sends.append(teslacan.create_epb_enable_signal(idx))
      self.last_angle = apply_angle
      
      # Adaptive cruise control
      current_time_ms = int(round(time.time() * 1000))
      if CS.cruise_buttons not in [CruiseButtons.IDLE, CruiseButtons.MAIN]:
        self.human_cruise_action_time = current_time_ms
      cruise_msg = None
      # The difference between OP's target speed and the current cruise
      # control speed, in KPH.
      speed_offset = (pcm_speed * CV.MS_TO_KPH - CS.v_cruise_actual)
      # If ACC was just enabled, also enable traditional cruise control if
      # possible.
      if (CS.enable_adaptive_cruise and CS.pcm_acc_status == 1
          and CS.v_ego > 18 * CV.MPH_TO_MS and speed_offset >= 0):
        self.automated_cruise_action_time = current_time_ms
        cruise_msg = teslacan.create_cruise_adjust_msg(CruiseButtons.DECEL_2ND, CS.steering_wheel_stalk)
      elif (# Only do adaptive cruise control while cruise control is enabled.
          CS.enable_adaptive_cruise and CS.pcm_acc_status == 2
          # And OpenPilot is steering.
          and enable_steer_control
          # And adjust infrequently, since sending repeated adjustments makes
          # the car think we're doing a 'long press' on the cruise stalk,
          # resulting in small, jerky speed adjustments.
          and current_time_ms > self.automated_cruise_action_time + 1000
          # And don't make adjustments if a human has manually done so in the
          # last 2 seconds. Human intention should not be overridden.
          and current_time_ms > self.human_cruise_action_time + 2000):
        # Metric cars adjust cruise in units of 1 and 5 kph.
        half_press_kph = 1
        full_press_kph = 5
        # Imperial unit cars adjust cruise in units of 1 and 5 mph.
        if CS.imperial_speed_units:
          half_press_kph = 1 * CV.MPH_TO_KPH
          full_press_kph = 5 * CV.MPH_TO_KPH
        
        # Reduce cruise speed significantly if necessary.
        if speed_offset < (-1 * full_press_kph):
          # Send cruise stalk dn_2nd.
          cruise_msg = teslacan.create_cruise_adjust_msg(CruiseButtons.DECEL_2ND, CS.steering_wheel_stalk)
        # Reduce speed slightly if necessary.
        elif speed_offset < (-1 * half_press_kph):
          # Send cruise stalk dn_1st.
          cruise_msg = teslacan.create_cruise_adjust_msg(CruiseButtons.DECEL_SET, CS.steering_wheel_stalk)
        # Increase cruise speed if possible.
        elif CS.v_ego > 18 * CV.MPH_TO_MS:  # cruise only works >18mph.
          # How much we can accelerate without exceeding the max allowed speed.
          available_speed = CS.v_cruise_pcm - CS.v_cruise_actual
          if full_press_kph < speed_offset and full_press_kph < available_speed:
            # Send cruise stalk up_2nd.
            cruise_msg = teslacan.create_cruise_adjust_msg(CruiseButtons.RES_ACCEL_2ND, CS.steering_wheel_stalk)
          elif half_press_kph < speed_offset and half_press_kph < available_speed:
            # Send cruise stalk up_1st.
            cruise_msg = teslacan.create_cruise_adjust_msg(CruiseButtons.RES_ACCEL, CS.steering_wheel_stalk)
      if cruise_msg:
        self.automated_cruise_action_time = current_time_ms
        can_sends.insert(0, cruise_msg)

      sendcan.send(can_list_to_can_capnp(can_sends, msgtype='sendcan').to_bytes())
