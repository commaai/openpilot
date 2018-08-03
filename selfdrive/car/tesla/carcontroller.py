import os
import subprocess
from  threading import Thread
import traceback
import shlex
from collections import namedtuple
from selfdrive.boardd.boardd import can_list_to_can_capnp
from selfdrive.controls.lib.drive_helpers import rate_limit
from common.numpy_fast import clip, interp
import numpy as np
import math as mth
from common.realtime import sec_since_boot
from selfdrive.car.tesla import teslacan
from selfdrive.car.tesla.values import AH, CruiseButtons, CAR
from selfdrive.can.packer import CANPacker
from selfdrive.config import Conversions as CV
import custom_alert as tcm

# Steer angle limits
ANGLE_MAX_BP = [0., 27., 36.]
ANGLE_MAX_V = [410., 92., 36.]

ANGLE_DELTA_BP = [0., 5., 15.]
ANGLE_DELTA_V = [5., .8, .15]     # windup limit
ANGLE_DELTA_VU = [5., 3.5, 0.4]   # unwind limit

#change lane delta angles and other params
CL_MAXD_BP = [1., 32, 44.]
CL_MAXD_A = [.5, 0.22, 0.08] #delta angle based on speed; needs fine tune
CL_MIN_V = 8.9 # do not turn if speed less than x m/2; 20 mph = 8.9 m/s
CL_MAX_A = 10. # do not turn if actuator wants more than x deg for going straight; this should be interp based on speed

def actuator_hystereses(brake, braking, brake_steady, v_ego, car_fingerprint):
  # hyst params... TODO: move these to VehicleParams
  brake_hyst_on = 0.02     # to activate brakes exceed this value
  brake_hyst_off = 0.005                     # to deactivate brakes below this value
  brake_hyst_gap = 0.01                      # don't change brake command for small ocilalitons within this value

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
  brake = brake_steady

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
    self.enable_camera = enable_camera
    self.packer = CANPacker(dbc_name)
    self.epas_disabled = True
    self.last_angle = 0.
    self.laneChange_avg_angle = 0.
    self.laneChange_avg_count = 0.

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

    #countdown for custom message timer
    if (CS.custom_alert_counter > 0):
      CS.custom_alert_counter -= 1
      if (CS.custom_alert_counter ==0):
        tcm.custom_alert_message("")
        CS.custom_alert_counter = -1

    enable_steer_control = (enabled and not changing_lanes)
    laneChange_angle = 0.
    if (CS.prev_right_blinker_on and (not CS.right_blinker_on)) or ( CS.prev_left_blinker_on and (not CS.left_blinker_on)):
      if CS.laneChange_enabled ==7:
        CS.laneChange_enabled =1
        CS.laneChange_counter =0
        tcm.custom_alert_message("")
    if (((not CS.prev_right_blinker_on) and CS.right_blinker_on) or ((not     CS.prev_left_blinker_on) and CS.left_blinker_on)) and ((CS.v_ego < CL_MIN_V) or (abs(actuators.steerAngle) >= CL_MAX_A)):
      #something is not right; signal in oposite direction; cancel
      tcm.custom_alert_message("Auto Lane Change Unavailable!")
      CS.custom_alert_counter = 500
    if (((not CS.prev_right_blinker_on) and CS.right_blinker_on) or ((not CS.prev_left_blinker_on) and CS.left_blinker_on)) and (CS.v_ego >= CL_MIN_V) and (abs(actuators.steerAngle) < CL_MAX_A):
      laneChange_direction = 1.2 # for some reason right turns for me nees more angke
      #changing lanes
      if CS.left_blinker_on:
        laneChange_direction = -1.2
      if (CS.laneChange_enabled > 1) and (CS.laneChange_direction <> laneChange_direction):
        #something is not right; signal in oposite direction; cancel
        tcm.custom_alert_message("Auto Lane Change Canceled! (s)")
        CS.custom_alert_counter = 200
        CS.laneChange_enabled = 1
        CS.laneChange_counter = 0
        CS.laneChange_direction = 0
      elif (CS.laneChange_enabled == 1) : 
        #compute angle delta for lane change
        tcm.custom_alert_message("Auto Lane Change Engaged! (1)")
        CS.custom_alert_counter = 100
        CS.laneChange_enabled = 5 
        CS.laneChange_counter = 1
        CS.laneChange_direction = laneChange_direction
        self.laneChange_avg_angle = 0.
        self.laneChange_avg_count = 0.
    #lane change in process
    if CS.laneChange_enabled > 1:
      CS.laneChange_angled = CS.laneChange_direction * CS.laneChange_steerr *  interp(CS.v_ego, CL_MAXD_BP, CL_MAXD_A)
      if (CS.steer_override or (CS.v_ego < CL_MIN_V)):
        tcm.custom_alert_message("Auto Lane Change Canceled! (u)")
        CS.custom_alert_counter = 200
        #if any steer override cancel process or if speed less than min speed
        CS.laneChange_counter = 0
        CS.laneChange_enabled = 1
        CS.laneChange_direction = 0
      if CS.laneChange_enabled == 2:
        if CS.laneChange_counter == 1:
          tcm.custom_alert_message("Auto Lane Change Engaged! (5)")
          CS.custom_alert_counter = 100
        laneChange_angle = CS.laneChange_angled * (50 - CS.laneChange_counter )/ 50
        CS.laneChange_counter += 1
        if CS.laneChange_counter == 2: #this stage can be removed once we are all done with dev
          CS.laneChange_enabled = 7
          CS.laneChange_counter = 1 
          CS.laneChange_direction = 0
      if CS.laneChange_enabled ==3:
        if CS.laneChange_counter == 1:
          tcm.custom_alert_message("Auto Lane Change Engaged! (4)")
          CS.custom_alert_counter = 800
        CS.laneChange_counter += 1
        laneChange_angle = CS.laneChange_angled
        if CS.laneChange_counter > CS.laneChange_duration * 100:
          angle_limit = CS.laneChange_angled if CS.laneChange_angled > 2. else 3.
          if (abs(actuators.steerAngle - CS.laneChange_angle - laneChange_angle) < angle_limit):
            #we're close to the angle, so we should release
            #need to add outer limits based on delta angle sign change
            #between steerAngle and what we compute as well as an upper limit
            CS.laneChange_enabled = 2
            CS.laneChange_counter = 1
        if CS.laneChange_counter >  (CS.laneChange_duration - 1) * 100:
          CS.laneChange_enabled = 2
          CS.laneChange_counter = 1

      if CS.laneChange_enabled == 4:
        if CS.laneChange_counter == 1:
          CS.laneChange_angle = self.laneChange_avg_angle / self.laneChange_avg_count
          #-actuators.steerAngle
          tcm.custom_alert_message("Auto Lane Change Engaged! (3)")
          CS.custom_alert_counter=100
          #if angle more than max angle allowed cancel; last chance to cancel on road curvature
          if (abs(CS.laneChange_angle) > CL_MAX_A):
            tcm.custom_alert_message("Auto Lane Change Canceled! (a)")
            CS.custom_alert_counter = 200
            CS.laneChange_enabled = 1
            CS.laneChange_counter = 0
            CS.laneChange_direction = 0
        laneChange_angle = CS.laneChange_angled *  CS.laneChange_counter / 50
        CS.laneChange_counter += 1
        if CS.laneChange_counter == 50:
          CS.laneChange_enabled = 3
          CS.laneChange_counter = 1
      if CS.laneChange_enabled == 5:
        if CS.laneChange_counter == 1:
          tcm.custom_alert_message("Auto Lane Change Engaged! (2)")
          CS.custom_alert_counter = CS.laneChange_wait * 100
        CS.laneChange_counter += 1
        if CS.laneChange_counter > (CS.laneChange_wait -1) *100:
          self.laneChange_avg_angle +=  -actuators.steerAngle
          self.laneChange_avg_count += 1
        if CS.laneChange_counter == CS.laneChange_wait * 100:
          CS.laneChange_enabled = 4
          CS.laneChange_counter = 1
      if CS.laneChange_enabled == 7:
        if CS.laneChange_counter ==1:
          tcm.custom_alert_message("Auto Lane Change Complete!")
          CS.custom_alert_counter = 200
        CS.laneChange_counter +=1
    enable_steer_control = (enabled and ((not changing_lanes) or (CS.laneChange_enabled > 1)))
    apply_angle = 0.
    # Angle
    if (CS.laneChange_enabled > 1) and (CS.laneChange_enabled < 5): 
      apply_angle = CS.laneChange_angle + laneChange_angle
    else:
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
    #if (changing_lanes and (CS.laneChange_enabled < 2)):
    #  apply_angle = CS.angle_steers
    # Send CAN commands.
    can_sends = []
    send_step = 5

    if  (True): #(frame % send_step) == 0:
      """#first we emulate DAS
      if (CS.DAS_info_frm == -1):
        #initialize all frames
        CS.DAS_info_frm = frame # 1.00 s interval
        CS.DAS_status_frm = (frame + 10) % 100 # 0.50 s interval
        CS.DAS_status2_frm = (frame + 35) % 100 # 0.50 s interval in between DAS_status
        CS.DAS_bodyControls_frm = (frame + 40) % 100 # 0.50 s interval
        CS.DAS_lanes_frm = (frame + 5) % 100 # 0.10 s interval 
        CS.DAS_objects_frm = (frame + 2) % 100 # 0.03 s interval
        CS.DAS_pscControl_frm = (frame + 3) % 100 # 0.04 s interval
      if (CS.DAS_info_frm == frame):
        can_sends.append(teslacan.create_DAS_info_msg(CS.DAS_info_msg))
        CS.DAS_info_msg += 1
        CS.DAS_info_msg = CS.DAS_info_msg % 10
      if (CS.DAS_status_frm == frame):
        can_sends.append(teslacan.create_DAS_status_msg(CS.DAS_status_idx))
        CS.DAS_status_idx += 1
        CS.DAS_status_idx = CS.DAS_status_idx % 16
        CS.DAS_status_frm = (CS.DAS_status_frm + 50) % 100
      if (CS.DAS_status2_frm == frame):
        can_sends.append(teslacan.create_DAS_status2_msg(CS.DAS_status2_idx))
        CS.DAS_status2_idx += 1
        CS.DAS_status2_idx = CS.DAS_status2_idx % 16
        CS.DAS_status2_frm = (CS.DAS_status2_frm + 50) % 100
      if (CS.DAS_bodyControls_frm == frame):
        can_sends.append(teslacan.create_DAS_bodyControls_msg(CS.DAS_bodyControls_idx))
        CS.DAS_bodyControls_idx += 1
        CS.DAS_bodyControls_idx = CS.DAS_bodyControls_idx % 16
        CS.DAS_bodyControls_frm = (CS.DAS_bodyControls_frm + 50) % 100
      if (CS.DAS_lanes_frm == frame):
        can_sends.append(teslacan.create_DAS_lanes_msg(CS.DAS_lanes_idx))
        CS.DAS_lanes_idx += 1
        CS.DAS_lanes_idx = CS.DAS_lanes_idx % 16
        CS.DAS_lanes_frm = (CS.DAS_lanes_frm + 10) % 100
      if (CS.DAS_pscControl_frm == frame):
        can_sends.append(teslacan.create_DAS_pscControl_msg(CS.DAS_pscControl_idx))
        CS.DAS_pscControl_idx += 1
        CS.DAS_pscControl_idx = CS.DAS_pscControl_idx % 16
        CS.DAS_pscControl_frm = (CS.DAS_pscControl_frm + 4) % 100
      if (CS.DAS_objects_frm == frame):
        can_sends.append(teslacan.create_DAS_objects_msg(CS.DAS_objects_idx))
        CS.DAS_objects_idx += 1
        CS.DAS_objects_idx = CS.DAS_objects_idx % 16
        CS.DAS_objects_frm = (CS.DAS_objects_frm + 3) % 100
      # end of DAS emulation """
      idx = frame % 16 #(frame/send_step) % 16 
      can_sends.append(teslacan.create_steering_control(enable_steer_control, apply_angle, idx))
      can_sends.append(teslacan.create_epb_enable_signal(idx))
      self.last_angle = apply_angle
      sendcan.send(can_list_to_can_capnp(can_sends, msgtype='sendcan').to_bytes())
