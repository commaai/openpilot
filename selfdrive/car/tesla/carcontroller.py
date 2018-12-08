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
from selfdrive.car.tesla.values import AH, CruiseButtons, CAR, CM
from selfdrive.can.packer import CANPacker
from selfdrive.config import Conversions as CV
from selfdrive.car.modules.ALCA_module import ALCAController
from selfdrive.car.tesla.ACC_module import ACCController
from selfdrive.car.tesla.PCC_module import PCCController
from selfdrive.car.tesla.HSO_module import HSOController
import zmq
import selfdrive.messaging as messaging
from selfdrive.services import service_list
from cereal import ui

# Steer angle limits
ANGLE_MAX_BP = [0., 27., 36.]
ANGLE_MAX_V = [410., 92., 36.]

ANGLE_DELTA_BP = [0., 5., 15.]
ANGLE_DELTA_V = [5., .8, .25]     # windup limit
ANGLE_DELTA_VU = [5., 3.5, 0.8]   # unwind limit


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
    self.last_accel = 0.
    self.ALCA = ALCAController(self,True,True)  # Enabled and SteerByAngle both True
    self.ACC = ACCController()
    self.PCC = PCCController(self)
    self.HSO = HSOController(self)
    self.sent_DAS_bootID = False
    context = zmq.Context()
    self.poller = zmq.Poller()
    self.speedlimit = messaging.sub_sock(context, service_list['speedLimit'].port, conflate=True, poller=self.poller)
    self.speedlimit_mph = 0


  def update(self, sendcan, enabled, CS, frame, actuators, \
             pcm_speed, pcm_override, pcm_cancel_cmd, pcm_accel, \
             hud_v_cruise, hud_show_lanes, hud_show_car, hud_alert, \
             snd_beep, snd_chime):

    """ Controls thread """

    ## Todo add code to detect Tesla DAS (camera) and go into listen and record mode only (for AP1 / AP2 cars)
    if not self.enable_camera:
      return

    # *** no output if not enabled ***
    if not enabled and CS.pcm_acc_status:
      # send pcm acc cancel cmd if drive is disabled but pcm is still on, or if the system can't be activated
      pcm_cancel_cmd = True

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

    #upodate custom UI buttons and alerts
    CS.UE.update_custom_ui()
      
    if (frame % 1000 == 0):
      CS.cstm_btns.send_button_info()

    # Update statuses for custom buttons every 0.1 sec.
    if self.ALCA.pid == None:
      self.ALCA.set_pid(CS)
    if (frame % 10 == 0):
      self.ALCA.update_status(CS.cstm_btns.get_button_status("alca") > 0)
      #print CS.cstm_btns.get_button_status("alca")

    
    if CS.pedal_interceptor_available:
      #update PCC module info
      self.PCC.update_stat(CS, True, sendcan)
      self.ACC.enable_adaptive_cruise = False
    else:
      # Update ACC module info.
      self.ACC.update_stat(CS, True)
      self.PCC.enable_pedal_cruise = False
    
    # Update HSO module info.
    human_control = False

    # update CS.v_cruise_pcm based on module selected.
    if self.ACC.enable_adaptive_cruise:
      CS.v_cruise_pcm = self.ACC.acc_speed_kph
    elif self.PCC.enable_pedal_cruise:
      CS.v_cruise_pcm = self.PCC.pedal_speed_kph
    else:
      CS.v_cruise_pcm = CS.v_cruise_actual
    # Get the angle from ALCA.
    alca_enabled = False
    turn_signal_needed = 0
    alca_steer = 0.
    apply_angle, alca_steer,alca_enabled, turn_signal_needed = self.ALCA.update(enabled, CS, frame, actuators)
    apply_angle = -apply_angle  # Tesla is reversed vs OP.
    human_control = self.HSO.update_stat(CS, enabled, actuators, frame)
    human_lane_changing = changing_lanes and not alca_enabled
    enable_steer_control = (enabled
                            and not human_lane_changing
                            and not human_control)
    
    angle_lim = interp(CS.v_ego, ANGLE_MAX_BP, ANGLE_MAX_V)
    apply_angle = clip(apply_angle, -angle_lim, angle_lim)
    # Windup slower.
    if self.last_angle * apply_angle > 0. and abs(apply_angle) > abs(self.last_angle):
      angle_rate_lim = interp(CS.v_ego, ANGLE_DELTA_BP, ANGLE_DELTA_V)
    else:
      angle_rate_lim = interp(CS.v_ego, ANGLE_DELTA_BP, ANGLE_DELTA_VU)

    apply_angle = clip(apply_angle, self.last_angle - angle_rate_lim, self.last_angle + angle_rate_lim)
    # If human control, send the steering angle as read at steering wheel.
    if human_control:
      apply_angle = CS.angle_steers
    # If blinker is on send the actual angle.
    #if (changing_lanes and (CS.laneChange_enabled < 2)):
    #  apply_angle = CS.angle_steers
    # Send CAN commands.
    can_sends = []
    send_step = 5

    if  (True):
      #First we emulate DAS.
      #send DAS_bootID
      if not self.sent_DAS_bootID:
        can_sends.append(teslacan.create_DAS_bootID_msg())
        self.sent_DAS_bootID = True
      else:
        #get speed limit
        for socket, _ in self.poller.poll(0):
            if socket is self.speedlimit:
              self.speedlimit_mph = ui.SpeedLimitData.from_bytes(socket.recv()).speed * CV.MS_TO_MPH
        #send DAS_info
        if frame % 100 == 0: 
          can_sends.append(teslacan.create_DAS_info_msg(CS.DAS_info_msg))
          CS.DAS_info_msg += 1
          CS.DAS_info_msg = CS.DAS_info_msg % 10
        #send DAS_status
        if frame % 50 == 0: 
          op_status = 0x02
          hands_on_state = 0x00
          speed_limit_kph = int(self.speedlimit_mph)
          alca_state = 0x08 
          if enabled:
            op_status = 0x03
            alca_state = 0x08 + turn_signal_needed
            #if not enable_steer_control:
              #op_status = 0x04
              #hands_on_state = 0x03
            if hud_alert == AH.STEER:
              if snd_chime == CM.MUTE:
                hands_on_state = 0x03
              else:
                hands_on_state = 0x05
          can_sends.append(teslacan.create_DAS_status_msg(CS.DAS_status_idx,op_status,speed_limit_kph,alca_state,hands_on_state))
          CS.DAS_status_idx += 1
          CS.DAS_status_idx = CS.DAS_status_idx % 16
        #send DAS_status2
        if frame % 50 == 0: 
          collision_warning = 0x00
          acc_speed_limit_mph = CS.v_cruise_pcm * CV.KPH_TO_MPH
          if hud_alert == AH.FCW:
            collision_warning = 0x01
          can_sends.append(teslacan.create_DAS_status2_msg(CS.DAS_status2_idx,acc_speed_limit_mph,collision_warning))
          CS.DAS_status2_idx += 1
          CS.DAS_status2_idx = CS.DAS_status2_idx % 16
        #send DAS_bodyControl
        if frame % 50 == 0: 
          can_sends.append(teslacan.create_DAS_bodyControls_msg(CS.DAS_bodyControls_idx,turn_signal_needed))
          CS.DAS_bodyControls_idx += 1
          CS.DAS_bodyControls_idx = CS.DAS_bodyControls_idx % 16
        #send DAS_control
        if frame % 4 == 0:
          acc_speed_limit_kph = self.ACC.new_speed #pcm_speed * CV.MS_TO_KPH
          accel_min = -15
          accel_max = 5
          speed_control_enabled = enabled and (acc_speed_limit_kph > 0) 
          can_sends.append(teslacan.create_DAS_control(CS.DAS_control_idx,speed_control_enabled,acc_speed_limit_kph,accel_min,accel_max))
          CS.DAS_control_idx += 1
          CS.DAS_control_idx = CS.DAS_control_idx % 8 
        #send DAS_lanes
        if frame % 10 == 0: 
          can_sends.append(teslacan.create_DAS_lanes_msg(CS.DAS_lanes_idx))
          CS.DAS_lanes_idx += 1
          CS.DAS_lanes_idx = CS.DAS_lanes_idx % 16
        #send DAS_pscControl
        if frame % 4 == 0: 
          can_sends.append(teslacan.create_DAS_pscControl_msg(CS.DAS_pscControl_idx))
          CS.DAS_pscControl_idx += 1
          CS.DAS_pscControl_idx = CS.DAS_pscControl_idx % 16
        #send DAS_telemetryPeriodic
        if frame % 4 == 0:
          can_sends.append(teslacan.create_DAS_telemetryPeriodic(CS.DAS_telemetryPeriodic1_idx,CS.DAS_telemetryPeriodic2_idx))
          CS.DAS_telemetryPeriodic2_idx += 1
          CS.DAS_telemetryPeriodic2_idx = CS.DAS_telemetryPeriodic2_idx % 10
          if CS.DAS_telemetryPeriodic2_idx == 0:
            CS.DAS_telemetryPeriodic1_idx += 2
            CS.DAS_telemetryPeriodic1_idx = CS.DAS_telemetryPeriodic1_idx % 16
        #send DAS_telemetryEvent
        if frame % 10 == 0:
          #can_sends.append(teslacan.create_DAS_telemetryEvent(CS.DAS_telemetryEvent1_idx,CS.DAS_telemetryEvent2_idx))
          CS.DAS_telemetryEvent2_idx += 1
          CS.DAS_telemetryEvent2_idx = CS.DAS_telemetryEvent2_idx % 10
          if CS.DAS_telemetryEvent2_idx == 0:
            CS.DAS_telemetryEvent1_idx += 2
            CS.DAS_telemetryEvent1_idx = CS.DAS_telemetryEvent1_idx % 16
        #send DAS_visualDebug
        if (frame + 1) % 10 == 0:
          can_sends.append(teslacan.create_DAS_visualDebug_msg())
        #send DAS_chNm
        if (frame + 2) % 10 == 0:
          can_sends.append(teslacan.create_DAS_chNm())
        #send DAS_objects
        if frame % 3 == 0: 
          can_sends.append(teslacan.create_DAS_objects_msg(CS.DAS_objects_idx))
          CS.DAS_objects_idx += 1
          CS.DAS_objects_idx = CS.DAS_objects_idx % 16
        #send DAS_warningMatrix0
        if frame % 6 == 0: 
          can_sends.append(teslacan.create_DAS_warningMatrix0(CS.DAS_warningMatrix0_idx))
          CS.DAS_warningMatrix0_idx += 1
          CS.DAS_warningMatrix0_idx = CS.DAS_warningMatrix0_idx % 16
        #send DAS_warningMatrix3
        if (frame + 3) % 6 == 0: 
          driverResumeRequired = 0
          if enabled and not enable_steer_control:
            driverResumeRequired = 1
          can_sends.append(teslacan.create_DAS_warningMatrix3(CS.DAS_warningMatrix3_idx,driverResumeRequired))
          CS.DAS_warningMatrix3_idx += 1
          CS.DAS_warningMatrix3_idx = CS.DAS_warningMatrix3_idx % 16
        #send DAS_warningMatrix1
        if frame  % 100 == 0: 
          can_sends.append(teslacan.create_DAS_warningMatrix1(CS.DAS_warningMatrix1_idx))
          CS.DAS_warningMatrix1_idx += 1
          CS.DAS_warningMatrix1_idx = CS.DAS_warningMatrix1_idx % 16
      # end of DAS emulation """
      idx = frame % 16
      can_sends.append(teslacan.create_steering_control(enable_steer_control, apply_angle, idx))
      can_sends.append(teslacan.create_epb_enable_signal(idx))
      cruise_btn = None
      if self.ACC.enable_adaptive_cruise and not CS.pedal_interceptor_available:
        cruise_btn = self.ACC.update_acc(enabled, CS, frame, actuators, pcm_speed)

      #add fake carConfig to trigger IC to display AP
      if frame % 2 == 0:
        carConfig_msg = teslacan.create_GTW_carConfig_msg(
          real_carConfig_data = CS.real_carConfig,
          dasHw = 1,
          autoPilot = 1,
          fRadarHw = 1)
        #can_sends.append(carConfig_msg)
      
      if cruise_btn or (turn_signal_needed > 0 and frame % 2 == 0):
          cruise_msg = teslacan.create_cruise_adjust_msg(
            spdCtrlLvr_stat=cruise_btn,
            turnIndLvr_Stat= 0, #turn_signal_needed,
            real_steering_wheel_stalk=CS.steering_wheel_stalk)
          # Send this CAN msg first because it is racing against the real stalk.
          can_sends.insert(0, cruise_msg)
      apply_accel = 0.
      if CS.pedal_interceptor_available and frame % 5 == 0: # pedal processed at 20Hz
        apply_accel, accel_needed, accel_idx = self.PCC.update_pdl(enabled, CS, frame, actuators, pcm_speed)
        can_sends.append(teslacan.create_pedal_command_msg(apply_accel, int(accel_needed), accel_idx))
      self.last_angle = apply_angle
      self.last_accel = apply_accel
      sendcan.send(can_list_to_can_capnp(can_sends, msgtype='sendcan').to_bytes())
