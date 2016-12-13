#!/usr/bin/env python
import os
import zmq
import numpy as np
import selfdrive.messaging as messaging

from cereal import car

from selfdrive.config import Conversions as CV
from common.services import service_list
from common.realtime import sec_since_boot, set_realtime_priority, Ratekeeper

from selfdrive.controls.lib.drive_helpers import learn_angle_offset

from selfdrive.controls.lib.longcontrol import LongControl
from selfdrive.controls.lib.latcontrol import LatControl

from selfdrive.controls.lib.pathplanner import PathPlanner
from selfdrive.controls.lib.adaptivecruise import AdaptiveCruise

from selfdrive.controls.lib.alertmanager import AlertManager

car_type = os.getenv("CAR")
if car_type is not None:
  exec('from selfdrive.car.'+car_type+'.interface import CarInterface')
else:
  from selfdrive.car.honda.interface import CarInterface

V_CRUISE_MAX = 144
V_CRUISE_MIN = 8
V_CRUISE_DELTA = 8
V_CRUISE_ENABLE_MIN = 40

def controlsd_thread(gctx, rate=100):  #rate in Hz
  # *** log ***
  context = zmq.Context()
  live100 = messaging.pub_sock(context, service_list['live100'].port)

  thermal = messaging.sub_sock(context, service_list['thermal'].port)
  live20 = messaging.sub_sock(context, service_list['live20'].port)
  model = messaging.sub_sock(context, service_list['model'].port)
  health = messaging.sub_sock(context, service_list['health'].port)

  # connects to can and sendcan
  CI = CarInterface()
  VP = CI.getVehicleParams()

  PP = PathPlanner(model)
  AC = AdaptiveCruise(live20)

  AM = AlertManager()

  LoC = LongControl()
  LaC = LatControl()

  # controls enabled state
  enabled = False
  last_enable_request = 0

  # learned angle offset
  angle_offset = 0

  # rear view camera state
  rear_view_toggle = False

  v_cruise_kph = 255

  # 0.0 - 1.0
  awareness_status = 0.0

  soft_disable_timer = None

  # start the loop
  set_realtime_priority(2)

  rk = Ratekeeper(rate)
  while 1:
    cur_time = sec_since_boot()

    # read CAN
    CS = CI.update()

    # did it request to enable?
    enable_request, enable_condition = False, False

    if enabled:
      # gives the user 6 minutes
      awareness_status -= 1.0/(100*60*6)
      if awareness_status <= 0.:
        AM.add("driverDistracted", enabled)

    # handle button presses
    for b in CS.buttonEvents:
      print b

      # reset awareness on any user action
      awareness_status = 1.0

      # button presses for rear view
      if b.type == "leftBlinker" or b.type == "rightBlinker":
        if b.pressed:
          rear_view_toggle = True
        else:
          rear_view_toggle = False

      if b.type == "altButton1" and b.pressed:
        rear_view_toggle = not rear_view_toggle

      if not VP.brake_only and enabled and not b.pressed:
        if b.type == "accelCruise":
          v_cruise_kph = v_cruise_kph - (v_cruise_kph % V_CRUISE_DELTA) + V_CRUISE_DELTA
        elif b.type == "decelCruise":
          v_cruise_kph = v_cruise_kph - (v_cruise_kph % V_CRUISE_DELTA) - V_CRUISE_DELTA
        v_cruise_kph = np.clip(v_cruise_kph, V_CRUISE_MIN, V_CRUISE_MAX)

      if not enabled and b.type in ["accelCruise", "decelCruise"] and not b.pressed:
        enable_request = True 

      # do disable on button down
      if b.type == "cancel" and b.pressed:
        AM.add("disable", enabled)

    # *** health checking logic ***
    hh = messaging.recv_sock(health)
    if hh is not None:
      # if the board isn't allowing controls but somehow we are enabled!
      if not hh.health.controlsAllowed and enabled:
        AM.add("controlsMismatch", enabled)

    # *** thermal checking logic ***

    # thermal data, checked every second
    td = messaging.recv_sock(thermal)
    if td is not None:
      cpu_temps = [td.thermal.cpu0, td.thermal.cpu1, td.thermal.cpu2,
                   td.thermal.cpu3, td.thermal.mem, td.thermal.gpu]
      # check overtemp
      if any(t > 950 for t in cpu_temps):
        AM.add("overheat", enabled)

    # *** getting model logic ***
    PP.update(cur_time, CS.vEgo)

    if rk.frame % 5 == 2:
      # *** run this at 20hz again ***
      angle_offset = learn_angle_offset(enabled, CS.vEgo, angle_offset, np.asarray(PP.d_poly), LaC.y_des, CS.steeringPressed)

    # disable if the pedals are pressed while engaged, this is a user disable
    if enabled:
      if CS.gasPressed or CS.brakePressed:
        AM.add("disable", enabled)

    if enable_request:
      # check for pressed pedals
      if CS.gasPressed or CS.brakePressed:
        AM.add("pedalPressed", enabled)
        enable_request = False
      else:
        print "enabled pressed at", cur_time
        last_enable_request = cur_time

    if VP.brake_only:
      enable_condition = ((cur_time - last_enable_request) < 0.2) and CS.cruiseState.enabled
    else:
      enable_condition = enable_request

    if enable_request or enable_condition or enabled:
      # add all alerts from car
      for alert in CS.errors:
        AM.add(alert, enabled)

      if AC.dead:
        AM.add("radarCommIssue", enabled)

      if PP.dead:
        AM.add("modelCommIssue", enabled)

    if enable_condition and not enabled and not AM.alertPresent():
      print "*** enabling controls"

      # beep for enabling
      AM.add("enable", enabled)

      # enable both lateral and longitudinal controls
      enabled = True

      # on activation, let's always set v_cruise from where we are, even if PCM ACC is active
      v_cruise_kph = int(round(np.maximum(CS.vEgo * CV.MS_TO_KPH * VP.ui_speed_fudge, V_CRUISE_ENABLE_MIN)))

      # 6 minutes driver you're on
      awareness_status = 1.0

      # reset the PID loops
      LaC.reset()
      # start long control at actual speed
      LoC.reset(v_pid = CS.vEgo)

    if VP.brake_only and CS.cruiseState.enabled:
      v_cruise_kph = CS.cruiseState.speed * CV.MS_TO_KPH

    # *** put the adaptive in adaptive cruise control ***
    AC.update(cur_time, CS.vEgo, CS.steeringAngle, LoC.v_pid, awareness_status, VP)

    # *** gas/brake PID loop ***
    final_gas, final_brake = LoC.update(enabled, CS.vEgo, v_cruise_kph, AC.v_target_lead, AC.a_target, AC.jerk_factor, VP)

    # *** steering PID loop ***
    final_steer, sat_flag = LaC.update(enabled, CS.vEgo, CS.steeringAngle, CS.steeringPressed, PP.d_poly, angle_offset, VP)

    # ***** handle alerts ****
    # send a "steering required alert" if saturation count has reached the limit
    if sat_flag:
      AM.add("steerSaturated", enabled)

    if enabled and AM.alertShouldDisable():
      print "DISABLING IMMEDIATELY ON ALERT"
      enabled = False

    if enabled and AM.alertShouldSoftDisable():
      if soft_disable_timer is None:
        soft_disable_timer = 3 * rate
      elif soft_disable_timer == 0:
        print "SOFT DISABLING ON ALERT"
        enabled = False
      else:
        soft_disable_timer -= 1
    else:
      soft_disable_timer = None
      

    # *** push the alerts to current ***
    alert_text_1, alert_text_2, visual_alert, audible_alert = AM.process_alerts(cur_time)

    # ***** control the car *****
    CC = car.CarControl.new_message()

    CC.enabled = enabled

    CC.gas = float(final_gas)
    CC.brake = float(final_brake)
    CC.steeringTorque = float(final_steer)

    CC.cruiseControl.override = True
    CC.cruiseControl.cancel = bool((not VP.brake_only) or (not enabled and CS.cruiseState.enabled))    # always cancel if we have an interceptor
    CC.cruiseControl.speedOverride = float((LoC.v_pid - .3) if (VP.brake_only and final_brake == 0.) else 0.0)
    CC.cruiseControl.accelOverride = float(AC.a_pcm)

    CC.hudControl.setSpeed = float(v_cruise_kph * CV.KPH_TO_MS)
    CC.hudControl.speedVisible = enabled
    CC.hudControl.lanesVisible = enabled
    CC.hudControl.leadVisible = bool(AC.has_lead)

    CC.hudControl.visualAlert = visual_alert
    CC.hudControl.audibleAlert = audible_alert

    # this alert will apply next controls cycle
    if not CI.apply(CC):
      AM.add("controlsFailed", enabled)

    # ***** publish state to logger *****

    # publish controls state at 100Hz
    dat = messaging.new_message()
    dat.init('live100')

    # show rear view camera on phone if in reverse gear or when button is pressed
    dat.live100.rearViewCam = ('reverseGear' in CS.errors) or rear_view_toggle
    dat.live100.alertText1 = alert_text_1
    dat.live100.alertText2 = alert_text_2
    dat.live100.awarenessStatus = max(awareness_status, 0.0) if enabled else 0.0

    # what packets were used to process
    dat.live100.canMonoTimes = list(CS.canMonoTimes)
    dat.live100.mdMonoTime = PP.logMonoTime
    dat.live100.l20MonoTime = AC.logMonoTime

    # if controls is enabled
    dat.live100.enabled = enabled

    # car state
    dat.live100.vEgo = CS.vEgo
    dat.live100.angleSteers = CS.steeringAngle
    dat.live100.steerOverride = CS.steeringPressed

    # longitudinal control state
    dat.live100.vPid = float(LoC.v_pid)
    dat.live100.vCruise = float(v_cruise_kph)
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

