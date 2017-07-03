#!/usr/bin/env python
import os
import zmq
import selfdrive.messaging as messaging

from cereal import car, log

from selfdrive.swaglog import cloudlog
from common.numpy_fast import clip
from common.fingerprints import fingerprint

from selfdrive.config import Conversions as CV
from selfdrive.services import service_list
from common.realtime import sec_since_boot, set_realtime_priority, Ratekeeper
from common.profiler import Profiler
from common.params import Params

from selfdrive.controls.lib.drive_helpers import learn_angle_offset

from selfdrive.controls.lib.longcontrol import LongControl
from selfdrive.controls.lib.latcontrol import LatControl

from selfdrive.controls.lib.alertmanager import AlertManager

V_CRUISE_MAX = 144
V_CRUISE_MIN = 8
V_CRUISE_DELTA = 8
V_CRUISE_ENABLE_MIN = 40

def controlsd_thread(gctx, rate=100):  #rate in Hz
  # *** log ***
  context = zmq.Context()
  live100 = messaging.pub_sock(context, service_list['live100'].port)
  carstate = messaging.pub_sock(context, service_list['carState'].port)
  carcontrol = messaging.pub_sock(context, service_list['carControl'].port)

  thermal = messaging.sub_sock(context, service_list['thermal'].port)
  health = messaging.sub_sock(context, service_list['health'].port)
  plan_sock = messaging.sub_sock(context, service_list['plan'].port)

  logcan = messaging.sub_sock(context, service_list['can'].port)

  # connects to can
  CP = fingerprint(logcan)

  # import the car from the fingerprint
  cloudlog.info("controlsd is importing %s", CP.carName)
  exec('from selfdrive.car.'+CP.carName+'.interface import CarInterface')

  sendcan = messaging.pub_sock(context, service_list['sendcan'].port)
  CI = CarInterface(CP, logcan, sendcan)

  # write CarParams
  params = Params()
  params.put("CarParams", CP.to_bytes())

  AM = AlertManager()

  LoC = LongControl()
  LaC = LatControl()

  # fake plan
  plan = log.Plan.new_message()
  plan.lateralValid = False
  plan.longitudinalValid = False
  last_plan_time = 0

  # controls enabled state
  enabled = False
  last_enable_request = 0

  # learned angle offset
  angle_offset = 0

  # rear view camera state
  rear_view_toggle = False
  rear_view_allowed = bool(params.get("IsRearViewMirror"))

  v_cruise_kph = 255

  # 0.0 - 1.0
  awareness_status = 0.0

  soft_disable_timer = None

  # Is cpu temp too high to enable?
  overtemp = False
  free_space = 1.0

  # start the loop
  set_realtime_priority(2)

  rk = Ratekeeper(rate, print_delay_threshold=2./1000)
  while 1:
    cur_time = sec_since_boot()
    prof = Profiler()

    # read CAN
    CS = CI.update()

    prof.checkpoint("CarInterface")

    # broadcast carState
    cs_send = messaging.new_message()
    cs_send.init('carState')
    cs_send.carState = CS    # copy?
    carstate.send(cs_send.to_bytes())

    prof.checkpoint("CarState")

    # did it request to enable?
    enable_request, enable_condition = False, False

    if enabled:
      # gives the user 6 minutes
      awareness_status -= 1.0/(100*60*6)
      if awareness_status <= 0.:
        AM.add("driverDistracted", enabled)

    # reset awareness status on steering
    if CS.steeringPressed:
      awareness_status = 1.0

    # handle button presses
    for b in CS.buttonEvents:
      print b

      # reset awareness on any user action
      awareness_status = 1.0

      # button presses for rear view
      if b.type == "leftBlinker" or b.type == "rightBlinker":
        if b.pressed and rear_view_allowed:
          rear_view_toggle = True
        else:
          rear_view_toggle = False

      if b.type == "altButton1" and b.pressed:
        rear_view_toggle = not rear_view_toggle

      if not CP.enableCruise and enabled and not b.pressed:
        if b.type == "accelCruise":
          v_cruise_kph = v_cruise_kph - (v_cruise_kph % V_CRUISE_DELTA) + V_CRUISE_DELTA
        elif b.type == "decelCruise":
          v_cruise_kph = v_cruise_kph - (v_cruise_kph % V_CRUISE_DELTA) - V_CRUISE_DELTA
        v_cruise_kph = clip(v_cruise_kph, V_CRUISE_MIN, V_CRUISE_MAX)

      if not enabled and b.type in ["accelCruise", "decelCruise"] and not b.pressed:
        enable_request = True

      # do disable on button down
      if b.type == "cancel" and b.pressed:
        AM.add("disable", enabled)

    prof.checkpoint("Buttons")

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
      # Check temperature.
      overtemp = any(
          t > 950
          for t in (td.thermal.cpu0, td.thermal.cpu1, td.thermal.cpu2,
                    td.thermal.cpu3, td.thermal.mem, td.thermal.gpu))
      # under 15% of space free
      free_space = td.thermal.freeSpace

    prof.checkpoint("Health")

    # disable if the pedals are pressed while engaged, this is a user disable
    if enabled:
      if CS.gasPressed or CS.brakePressed:
        AM.add("disable", enabled)

      # how did we get into this state?
      if CP.enableCruise and not CS.cruiseState.enabled:
        AM.add("cruiseDisabled", enabled)

    if enable_request:
      # check for pressed pedals
      if CS.gasPressed or CS.brakePressed:
        AM.add("pedalPressed", enabled)
        enable_request = False
      else:
        print "enabled pressed at", cur_time
        last_enable_request = cur_time

      # don't engage with less than 15% free
      if free_space < 0.15:
        AM.add("outOfSpace", enabled)
        enable_request = False

    if CP.enableCruise:
      enable_condition = ((cur_time - last_enable_request) < 0.2) and CS.cruiseState.enabled
    else:
      enable_condition = enable_request

    if CP.enableCruise and CS.cruiseState.enabled:
      v_cruise_kph = CS.cruiseState.speed * CV.MS_TO_KPH

    prof.checkpoint("AdaptiveCruise")

    # *** what's the plan ***
    new_plan = messaging.recv_sock(plan_sock)
    if new_plan is not None:
      plan = new_plan.plan
      plan = plan.as_builder()  # plan can change in controls
      last_plan_time = cur_time

    # check plan for timeout
    if cur_time - last_plan_time > 0.5:
      plan.lateralValid = False
      plan.longitudinalValid = False

    # gives 18 seconds before decel begins (w 6 minute timeout)
    if awareness_status < -0.05:
      plan.aTargetMax = min(plan.aTargetMax, -0.2)
      plan.aTargetMin = min(plan.aTargetMin, plan.aTargetMax)

    if enable_request or enable_condition or enabled:
      # add all alerts from car
      for alert in CS.errors:
        AM.add(alert, enabled)

      if not plan.longitudinalValid:
        AM.add("radarCommIssue", enabled)

      if not plan.lateralValid:
        # If the model is not broadcasting, assume that it is because
        # the user has uploaded insufficient data for calibration.
        # Other cases that would trigger this are rare and unactionable by the user.
        AM.add("dataNeeded", enabled)

      if overtemp:
        AM.add("overheat", enabled)

    # *** angle offset learning ***
    if rk.frame % 5 == 2 and plan.lateralValid:
      # *** run this at 20hz again ***
      angle_offset = learn_angle_offset(enabled, CS.vEgo, angle_offset, plan.dPoly, LaC.y_des, CS.steeringPressed)

    # *** gas/brake PID loop ***
    final_gas, final_brake = LoC.update(enabled, CS.vEgo, v_cruise_kph,
                                        plan.vTarget,
                                        [plan.aTargetMin, plan.aTargetMax],
                                        plan.jerkFactor, CP)

    # *** steering PID loop ***
    final_steer, sat_flag = LaC.update(enabled, CS.vEgo, CS.steeringAngle, CS.steeringPressed, plan.dPoly, angle_offset, CP)

    prof.checkpoint("PID")

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

    if enable_condition and not enabled and not AM.alertPresent():
      print "*** enabling controls"

      # beep for enabling
      AM.add("enable", enabled)

      # enable both lateral and longitudinal controls
      enabled = True

      # on activation, let's always set v_cruise from where we are, even if PCM ACC is active
      v_cruise_kph = int(round(max(CS.vEgo * CV.MS_TO_KPH, V_CRUISE_ENABLE_MIN)))

      # 6 minutes driver you're on
      awareness_status = 1.0

      # reset the PID loops
      LaC.reset()
      # start long control at actual speed
      LoC.reset(v_pid = CS.vEgo)

    # *** push the alerts to current ***
    alert_text_1, alert_text_2, visual_alert, audible_alert = AM.process_alerts(cur_time)

    # ***** control the car *****
    CC = car.CarControl.new_message()

    CC.enabled = enabled

    CC.gas = float(final_gas)
    CC.brake = float(final_brake)
    CC.steeringTorque = float(final_steer)

    CC.cruiseControl.override = True
    CC.cruiseControl.cancel = bool((not CP.enableCruise) or (not enabled and CS.cruiseState.enabled))    # always cancel if we have an interceptor

    # brake discount removes a sharp nonlinearity
    brake_discount = (1.0 - clip(final_brake*3., 0.0, 1.0))
    CC.cruiseControl.speedOverride = float(max(0.0, ((LoC.v_pid - .5) * brake_discount)) if CP.enableCruise else 0.0)

    #CC.cruiseControl.accelOverride = float(AC.a_pcm)
    # TODO: fix this
    CC.cruiseControl.accelOverride = float(1.0)

    CC.hudControl.setSpeed = float(v_cruise_kph * CV.KPH_TO_MS)
    CC.hudControl.speedVisible = enabled
    CC.hudControl.lanesVisible = enabled
    CC.hudControl.leadVisible = plan.hasLead

    CC.hudControl.visualAlert = visual_alert
    CC.hudControl.audibleAlert = audible_alert

    # this alert will apply next controls cycle
    if not CI.apply(CC):
      AM.add("controlsFailed", enabled)

    # broadcast carControl
    cc_send = messaging.new_message()
    cc_send.init('carControl')
    cc_send.carControl = CC    # copy?
    carcontrol.send(cc_send.to_bytes())

    prof.checkpoint("CarControl")

    # ***** publish state to logger *****

    # publish controls state at 100Hz
    dat = messaging.new_message()
    dat.init('live100')

    # show rear view camera on phone if in reverse gear or when button is pressed
    dat.live100.rearViewCam = ('reverseGear' in CS.errors and rear_view_allowed) or rear_view_toggle
    dat.live100.alertText1 = alert_text_1
    dat.live100.alertText2 = alert_text_2
    dat.live100.awarenessStatus = max(awareness_status, 0.0) if enabled else 0.0

    # what packets were used to process
    dat.live100.canMonoTimes = list(CS.canMonoTimes)

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
    dat.live100.vTargetLead = float(plan.vTarget)
    dat.live100.aTargetMin = float(plan.aTargetMin)
    dat.live100.aTargetMax = float(plan.aTargetMax)
    dat.live100.jerkFactor = float(plan.jerkFactor)

    # log learned angle offset
    dat.live100.angleOffset = float(angle_offset)

    # lag
    dat.live100.cumLagMs = -rk.remaining*1000.

    live100.send(dat.to_bytes())

    prof.checkpoint("Live100")

    # *** run loop at fixed rate ***
    if rk.keep_time():
      prof.display()

def main(gctx=None):
  controlsd_thread(gctx, 100)

if __name__ == "__main__":
  main()
