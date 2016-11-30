#!/usr/bin/env python
import zmq
import numpy as np

from common.services import service_list
from common.realtime import sec_since_boot, set_realtime_priority, Ratekeeper

from selfdrive.config import CruiseButtons
from selfdrive.config import Conversions as CV

from selfdrive.controls.lib.drive_helpers import learn_angle_offset
from selfdrive.controls.lib.alert_database import process_alert, AI

import selfdrive.messaging as messaging

from selfdrive.controls.lib.carstate import CarState
from selfdrive.controls.lib.carcontroller import CarController
from selfdrive.controls.lib.longcontrol import LongControl
from selfdrive.controls.lib.latcontrol import LatControl

from selfdrive.controls.lib.pathplanner import PathPlanner
from selfdrive.controls.lib.adaptivecruise import AdaptiveCruise

def controlsd_thread(gctx, rate=100):  #rate in Hz
  # *** log ***
  context = zmq.Context()
  live100 = messaging.pub_sock(context, service_list['live100'].port)
  thermal = messaging.sub_sock(context, service_list['thermal'].port)
  live20 = messaging.sub_sock(context, service_list['live20'].port)
  model = messaging.sub_sock(context, service_list['model'].port)

  logcan = messaging.sub_sock(context, service_list['can'].port)
  sendcan = messaging.pub_sock(context, service_list['sendcan'].port)

  # *** init the major players ***
  CS = CarState(logcan)
  CC = CarController()

  PP = PathPlanner(model)
  AC = AdaptiveCruise(live20)

  LoC = LongControl()
  LaC = LatControl()

  # *** control initial values ***
  apply_brake = 0
  enabled = False

  # *** time values ***
  last_enable_pressed = 0

  # *** controls initial values ***
  # *** display stuff
  soft_disable_start = 0
  sounding = False
  no_mismatch_pcm_last, no_mismatch_ctrl_last = 0, 0

  # car state
  alert, sound_exp, hud_exp, text_exp, alert_p = None, 0, 0, 0, 0
  rear_view_cam, rear_view_toggle = False, False

  v_cruise = 255           # this means no display
  v_cruise_max = 144
  v_cruise_min = 8
  v_cruise_delta = 8

  # on activation target at least 25mph. With 5mph you need too much tapping
  v_cruise_enable_min = 40

  hud_v_cruise = 255

  angle_offset = 0

  max_enable_speed = 57.   # ~91 mph

  pcm_threshold = 25.*CV.MPH_TO_MS    # below this speed pcm cancels

  overtemp = True

  # 0.0 - 1.0
  awareness_status = 0.0

  # start the loop
  set_realtime_priority(2)

  rk = Ratekeeper(rate)
  while 1:
    cur_time = sec_since_boot()

    # read CAN
    canMonoTimes = CS.update(logcan)

    # **** rearview mirror management ***
    if CS.cruise_setting == 1 and CS.prev_cruise_setting == 0:
      rear_view_toggle = not rear_view_toggle

    # show rear view camera on phone if in reverse gear or when lkas button is pressed
    rear_view_cam = (CS.gear_shifter == 2) or rear_view_toggle or CS.blinker_on

    # *** thermal checking logic ***

    # thermal data, checked every second
    td = messaging.recv_sock(thermal)
    if td is not None:
      cpu_temps = [td.thermal.cpu0, td.thermal.cpu1, td.thermal.cpu2,
                   td.thermal.cpu3, td.thermal.mem, td.thermal.gpu]
      # check overtemp
      overtemp = any(t > 950 for t in cpu_temps)

    # *** getting model logic ***
    PP.update(cur_time, CS.v_ego)

    if rk.frame % 5 == 2:
      # *** run this at 20hz again ***
      angle_offset = learn_angle_offset(enabled, CS.v_ego, angle_offset, np.asarray(PP.d_poly), LaC.y_des, CS.steer_override)

    # to avoid race conditions, check if control has been disabled for at least 0.2s
    mismatch_ctrl = not CC.controls_allowed and enabled
    mismatch_pcm = (not CS.pcm_acc_status and (not apply_brake or CS.v_ego < 0.1)) and enabled

    # keep resetting start timer if mismatch isn't true
    if not mismatch_ctrl:
      no_mismatch_ctrl_last = cur_time
    if not mismatch_pcm or not CS.brake_only:
      no_mismatch_pcm_last = cur_time

    #*** v_cruise logic ***
    if CS.brake_only:
      v_cruise = int(CS.v_cruise_pcm) # TODO: why sometimes v_cruise_pcm is long type?
    else:
      if CS.prev_cruise_buttons == 0 and CS.cruise_buttons == CruiseButtons.RES_ACCEL and enabled:
        v_cruise = v_cruise - (v_cruise % v_cruise_delta) + v_cruise_delta
      elif CS.prev_cruise_buttons == 0 and CS.cruise_buttons == CruiseButtons.DECEL_SET and enabled:
        v_cruise = v_cruise + (v_cruise % v_cruise_delta) - v_cruise_delta

    # *** enabling/disabling logic ***
    enable_pressed = (CS.prev_cruise_buttons == CruiseButtons.DECEL_SET or CS.prev_cruise_buttons == CruiseButtons.RES_ACCEL) \
                     and CS.cruise_buttons == 0

    if enable_pressed:
      print "enabled pressed at", cur_time
      last_enable_pressed = cur_time

    # if pcm does speed control than we need to wait on pcm to enable
    if CS.brake_only:
      enable_condition = (cur_time - last_enable_pressed) < 0.2 and CS.pcm_acc_status
    else:
      enable_condition = enable_pressed

    # always clear the alert at every cycle
    alert_id = []

    # check for PCM not enabling
    if CS.brake_only and (cur_time - last_enable_pressed) < 0.2 and not CS.pcm_acc_status:
      print "waiting for PCM to enable"

    # check for denied enabling
    if enable_pressed and not enabled:
      deny_enable = \
        [(AI.SEATBELT, not CS.seatbelt),
         (AI.DOOR_OPEN, not CS.door_all_closed),
         (AI.ESP_OFF, CS.esp_disabled),
         (AI.STEER_ERROR, CS.steer_error),
         (AI.BRAKE_ERROR, CS.brake_error),
         (AI.GEAR_NOT_D, not CS.gear_shifter_valid),
         (AI.MAIN_OFF, not CS.main_on),
         (AI.PEDAL_PRESSED, CS.user_gas_pressed or CS.brake_pressed or (CS.pedal_gas > 0 and CS.brake_only)),
         (AI.HIGH_SPEED, CS.v_ego > max_enable_speed),
         (AI.OVERHEAT, overtemp),
         (AI.COMM_ISSUE, PP.dead or AC.dead),
         (AI.CONTROLSD_LAG, rk.remaining < -0.2)]
      for alertn, cond in deny_enable:
        if cond:
          alert_id += [alertn]

    # check for soft disables
    if enabled:
      soft_disable = \
        [(AI.SEATBELT_SD, not CS.seatbelt),
         (AI.DOOR_OPEN_SD, not CS.door_all_closed),
         (AI.ESP_OFF_SD, CS.esp_disabled),
         (AI.OVERHEAT_SD, overtemp),
         (AI.COMM_ISSUE_SD, PP.dead or AC.dead),
         (AI.CONTROLSD_LAG_SD, rk.remaining < -0.2)]
      sounding = False
      for alertn, cond in soft_disable:
        if cond:
          alert_id += [alertn]
          sounding = True
          # soft disengagement expired, user need to take control
          if (cur_time - soft_disable_start) > 3.:
            enabled = False
            v_cruise = 255
      if not sounding:
        soft_disable_start = cur_time

    # check for immediate disables
    if enabled:
      immediate_disable = \
        [(AI.PCM_LOW_SPEED, (cur_time > no_mismatch_pcm_last > 0.2) and CS.v_ego < pcm_threshold),
         (AI.STEER_ERROR_ID, CS.steer_error),
         (AI.BRAKE_ERROR_ID, CS.brake_error),
         (AI.CTRL_MISMATCH_ID, (cur_time - no_mismatch_ctrl_last) > 0.2),
         (AI.PCM_MISMATCH_ID, (cur_time - no_mismatch_pcm_last) > 0.2)]
      for alertn, cond in immediate_disable:
        if cond:
          alert_id += [alertn]
          # immediate turn off control
          enabled = False
          v_cruise = 255

    # user disabling
    if enabled and (CS.user_gas_pressed or CS.brake_pressed or not CS.gear_shifter_valid or \
         (CS.cruise_buttons == CruiseButtons.CANCEL and CS.prev_cruise_buttons == 0) or \
         not CS.main_on or (CS.pedal_gas > 0 and CS.brake_only)):
      enabled = False
      v_cruise = 255
      alert_id += [AI.DISABLE]

    # enabling
    if enable_condition and not enabled and len(alert_id) == 0:
      print "*** enabling controls"

      #enable both lateral and longitudinal controls
      enabled = True
      counter_pcm_enabled = CS.counter_pcm
      # on activation, let's always set v_cruise from where we are, even if PCM ACC is active
      # what we want to be displayed in mph
      v_cruise_mph = round(CS.v_ego * CV.MS_TO_MPH * CS.ui_speed_fudge)
      # what we need to send to have that displayed
      v_cruise = int(round(np.maximum(v_cruise_mph * CV.MPH_TO_KPH, v_cruise_enable_min)))

      # 6 minutes driver you're on
      awareness_status = 1.0

      # reset the PID loops
      LaC.reset()
      # start long control at actual speed
      LoC.reset(v_pid = CS.v_ego)

      alert_id += [AI.ENABLE]

    if v_cruise != 255 and not CS.brake_only:
      v_cruise = np.clip(v_cruise, v_cruise_min, v_cruise_max)

    # **** awareness status manager ****
    if enabled:
      # gives the user 6 minutes
      awareness_status -= 1.0/(100*60*6)
      # reset on steering, blinker, or cruise buttons
      if CS.steer_override or CS.blinker_on or CS.cruise_buttons or CS.cruise_setting:
        awareness_status = 1.0
      if awareness_status <= 0.:
        alert_id += [AI.DRIVER_DISTRACTED]

    # ****** initial actuators commands ***
    # *** gas/brake PID loop ***
    AC.update(cur_time, CS.v_ego, CS.angle_steers, LoC.v_pid, awareness_status, CS.VP)
    final_gas, final_brake = LoC.update(enabled, CS, v_cruise, AC.v_target_lead, AC.a_target, AC.jerk_factor)
    pcm_accel = int(np.clip(AC.a_pcm/1.4,0,1)*0xc6)   # TODO: perc of max accel in ACC?

    # *** steering PID loop ***
    final_steer, sat_flag = LaC.update(enabled, CS, PP.d_poly, angle_offset)

    # this needs to stay before hysteresis logic to avoid pcm staying on control during brake hysteresis
    pcm_override = True   # this is always True
    pcm_cancel_cmd = False
    if CS.brake_only and final_brake == 0.:
      pcm_speed = LoC.v_pid - .3  # FIXME: just for exp
    else:
      pcm_speed = 0

    # ***** handle alerts ****
    # send a "steering required alert" if saturation count has reached the limit
    if sat_flag:
      alert_id += [AI.STEER_SATURATED]

    # process the alert, based on id
    alert, chime, beep, hud_alert, alert_text, sound_exp, hud_exp, text_exp, alert_p = \
      process_alert(alert_id, alert, cur_time, sound_exp, hud_exp, text_exp, alert_p)

    # alerts pub
    if len(alert_id) != 0:
      print alert_id, alert_text

    # *** process for hud display ***
    if not enabled or (hud_v_cruise == 255 and CS.counter_pcm == counter_pcm_enabled):
      hud_v_cruise = 255
    else:
      hud_v_cruise = v_cruise

    # *** actually do can sends ***
    CC.update(sendcan, enabled, CS, rk.frame, \
      final_gas, final_brake, final_steer, \
      pcm_speed, pcm_override, pcm_cancel_cmd, pcm_accel, \
      hud_v_cruise, hud_show_lanes = enabled, \
      hud_show_car = AC.has_lead, \
      hud_alert = hud_alert, \
      snd_beep = beep, snd_chime = chime)

    # ***** publish state to logger *****

    # publish controls state at 100Hz
    dat = messaging.new_message()
    dat.init('live100')

    # move liveUI into live100
    dat.live100.rearViewCam = bool(rear_view_cam)
    dat.live100.alertText1 = alert_text[0]
    dat.live100.alertText2 = alert_text[1]
    dat.live100.awarenessStatus = max(awareness_status, 0.0) if enabled else 0.0

    # what packets were used to process
    dat.live100.canMonoTimes = canMonoTimes
    dat.live100.mdMonoTime = PP.logMonoTime
    dat.live100.l20MonoTime = AC.logMonoTime

    # if controls is enabled
    dat.live100.enabled = enabled

    # car state
    dat.live100.vEgo = float(CS.v_ego)
    dat.live100.aEgo = float(CS.a_ego)
    dat.live100.angleSteers = float(CS.angle_steers)
    dat.live100.hudLead = CS.hud_lead
    dat.live100.steerOverride = CS.steer_override

    # longitudinal control state
    dat.live100.vPid = float(LoC.v_pid)
    dat.live100.vCruise = float(v_cruise)
    dat.live100.upAccelCmd = float(LoC.Up_accel_cmd)
    dat.live100.uiAccelCmd = float(LoC.Ui_accel_cmd)

    # lateral control state
    dat.live100.yActual = float(LaC.y_actual)
    dat.live100.yDes = float(LaC.y_des)
    dat.live100.upSteer = float(LaC.Up_steer)
    dat.live100.uiSteer = float(LaC.Ui_steer)

    # processed radar state, should add a_pcm?
    dat.live100.vTargetLead = float(AC.v_target_lead)
    dat.live100.aTargetMin = float(AC.a_target[0])
    dat.live100.aTargetMax = float(AC.a_target[1])
    dat.live100.jerkFactor = float(AC.jerk_factor)

    # lag
    dat.live100.cumLagMs = -rk.remaining*1000.

    live100.send(dat.to_bytes())

    # *** run loop at fixed rate ***
    rk.keep_time()

def main(gctx=None):
  controlsd_thread(gctx, 100)

if __name__ == "__main__":
  main()
