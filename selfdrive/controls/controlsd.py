#!/usr/bin/env python
import os
import json
from copy import copy

import zmq

from cereal import car, log
from common.numpy_fast import clip
from common.fingerprints import fingerprint
from common.realtime import sec_since_boot, set_realtime_priority, Ratekeeper
from common.profiler import Profiler
from common.params import Params

import selfdrive.messaging as messaging
from selfdrive.swaglog import cloudlog
from selfdrive.config import Conversions as CV
from selfdrive.services import service_list
from selfdrive.car import get_car
from selfdrive.controls.lib.planner import Planner
from selfdrive.controls.lib.drive_helpers import learn_angle_offset
from selfdrive.controls.lib.longcontrol import LongControl
from selfdrive.controls.lib.latcontrol import LatControl
from selfdrive.controls.lib.alertmanager import AlertManager
from selfdrive.controls.lib.vehicle_model import VehicleModel
from selfdrive.controls.lib.adaptivecruise import A_ACC_MAX

V_CRUISE_MAX = 144
V_CRUISE_MIN = 8
V_CRUISE_DELTA = 8
V_CRUISE_ENABLE_MIN = 40

AWARENESS_TIME = 360.      # 6 minutes limit without user touching steering wheels
AWARENESS_PRE_TIME = 20.   # a first alert is issued 20s before start decelerating the car
AWARENESS_DECEL = -0.2     # car smoothly decel at .2m/s^2 when user is distracted

# class Cal
class Calibration:
  UNCALIBRATED = 0
  CALIBRATED = 1
  INVALID = 2

# to be used
class State():
  DISABLED = 0
  ENABLED = 1
  SOFT_DISABLE = 2

class Controls(object):
  def __init__(self, gctx, rate=100):
    self.rate = rate

    # *** log ***
    context = zmq.Context()

    # pub
    self.live100 = messaging.pub_sock(context, service_list['live100'].port)
    self.carstate = messaging.pub_sock(context, service_list['carState'].port)
    self.carcontrol = messaging.pub_sock(context, service_list['carControl'].port)
    sendcan = messaging.pub_sock(context, service_list['sendcan'].port)
 
    # sub
    self.thermal = messaging.sub_sock(context, service_list['thermal'].port)
    self.health = messaging.sub_sock(context, service_list['health'].port)
    logcan = messaging.sub_sock(context, service_list['can'].port)
    self.cal = messaging.sub_sock(context, service_list['liveCalibration'].port)
    
    self.CC = car.CarControl.new_message()
    self.CI, self.CP = get_car(logcan, sendcan)
    self.PL = Planner(self.CP)
    self.AM = AlertManager()
    self.LoC = LongControl()
    self.LaC = LatControl()
    self.VM = VehicleModel(self.CP)
  
    # write CarParams
    params = Params()
    params.put("CarParams", self.CP.to_bytes())
  
    # fake plan
    self.plan_ts = 0
    self.plan = log.Plan.new_message()
    self.plan.lateralValid = False
    self.plan.longitudinalValid = False
  
    # controls enabled state
    self.enabled = False
    self.last_enable_request = 0
  
    # learned angle offset
    self.angle_offset = 0
    calibration_params = params.get("CalibrationParams")
    if calibration_params:
      try:
        calibration_params = json.loads(calibration_params)
        self.angle_offset = calibration_params["angle_offset"]
      except (ValueError, KeyError):
        pass
  
    # rear view camera state
    self.rear_view_toggle = False
    self.rear_view_allowed = (params.get("IsRearViewMirror") == "1")
  
    self.v_cruise_kph = 255
  
    # 0.0 - 1.0
    self.awareness_status = 1.0
  
    self.soft_disable_timer = None
  
    self.overtemp = False
    self.free_space = 1.0
    self.cal_status = Calibration.UNCALIBRATED
    self.cal_perc = 0
  
    self.rk = Ratekeeper(self.rate, print_delay_threshold=2./1000)
 
  def data_sample(self):
    self.prof = Profiler()
    self.cur_time = sec_since_boot()
    # first read can and compute car states
    self.CS = self.CI.update()

    self.prof.checkpoint("CarInterface")

    # *** thermal checking logic ***
    # thermal data, checked every second
    td = messaging.recv_sock(self.thermal)
    if td is not None:
      # Check temperature.
      self.overtemp = any(
          t > 950
          for t in (td.thermal.cpu0, td.thermal.cpu1, td.thermal.cpu2,
                    td.thermal.cpu3, td.thermal.mem, td.thermal.gpu))
      # under 15% of space free
      self.free_space = td.thermal.freeSpace

    # read calibration status
    cal = messaging.recv_sock(self.cal)
    if cal is not None:
      self.cal_status = cal.liveCalibration.calStatus
      self.cal_perc = cal.liveCalibration.calPerc
    

  def state_transition(self):
    pass # for now

  def state_control(self):
    
    # did it request to enable?
    enable_request, enable_condition = False, False

    # reset awareness status on steering
    if self.CS.steeringPressed or not self.enabled:
      self.awareness_status = 1.0
    elif self.enabled:
      # gives the user 6 minutes
      self.awareness_status -= 1.0/(self.rate * AWARENESS_TIME)
      if self.awareness_status <= 0.:
        self.AM.add("driverDistracted", self.enabled)
      elif self.awareness_status <= AWARENESS_PRE_TIME / AWARENESS_TIME and \
           self.awareness_status >= (AWARENESS_PRE_TIME - 0.1) / AWARENESS_TIME:
        self.AM.add("preDriverDistracted", self.enabled)

    # handle button presses
    for b in self.CS.buttonEvents:
      print b

      # button presses for rear view
      if b.type == "leftBlinker" or b.type == "rightBlinker":
        if b.pressed and self.rear_view_allowed:
          self.rear_view_toggle = True
        else:
          self.rear_view_toggle = False

      if b.type == "altButton1" and b.pressed:
        self.rear_view_toggle = not self.rear_view_toggle

      if not self.CP.enableCruise and self.enabled and not b.pressed:
        if b.type == "accelCruise":
          self.v_cruise_kph -= (self.v_cruise_kph % V_CRUISE_DELTA) - V_CRUISE_DELTA
        elif b.type == "decelCruise":
          self.v_cruise_kph -= (self.v_cruise_kph % V_CRUISE_DELTA) + V_CRUISE_DELTA
        self.v_cruise_kph = clip(self.v_cruise_kph, V_CRUISE_MIN, V_CRUISE_MAX)

      if not self.enabled and b.type in ["accelCruise", "decelCruise"] and not b.pressed:
        enable_request = True

      # do disable on button down
      if b.type == "cancel" and b.pressed:
        self.AM.add("disable", self.enabled)

    self.prof.checkpoint("Buttons")
    
    # *** health checking logic ***
    hh = messaging.recv_sock(self.health)
    if hh is not None:
      # if the board isn't allowing controls but somehow we are enabled!
      # TODO: this should be in state transition with a function follower logic
      if not hh.health.controlsAllowed and self.enabled:
        self.AM.add("controlsMismatch", self.enabled)

    # disable if the pedals are pressed while engaged, this is a user disable
    if self.enabled:
      if self.CS.gasPressed or self.CS.brakePressed or not self.CS.cruiseState.available:
        self.AM.add("disable", self.enabled)

      # it can happen that car cruise disables while comma system is enabled: need to
      # keep braking if needed or if the speed is very low
      # TODO: for the Acura, cancellation below 25mph is normal. Issue a non loud alert
      if self.CP.enableCruise and not self.CS.cruiseState.enabled and \
         (self.CC.brake <= 0. or self.CS.vEgo < 0.3):
        self.AM.add("cruiseDisabled", self.enabled)

    if enable_request:
      # check for pressed pedals
      if self.CS.gasPressed or self.CS.brakePressed:
        self.AM.add("pedalPressed", self.enabled)
        enable_request = False
      else:
        print "enabled pressed at", self.cur_time
        self.last_enable_request = self.cur_time

      # don't engage with less than 15% free
      if self.free_space < 0.15:
        self.AM.add("outOfSpace", self.enabled)
        enable_request = False

    if self.CP.enableCruise:
      enable_condition = ((self.cur_time - self.last_enable_request) < 0.2) and self.CS.cruiseState.enabled
    else:
      enable_condition = enable_request

    if self.CP.enableCruise and self.CS.cruiseState.enabled:
      self.v_cruise_kph = self.CS.cruiseState.speed * CV.MS_TO_KPH

    self.prof.checkpoint("AdaptiveCruise")

    # *** what's the plan ***
    plan_packet = self.PL.update(self.CS, self.LoC)
    self.plan = plan_packet.plan
    self.plan_ts = plan_packet.logMonoTime

    # if user is not responsive to awareness alerts, then start a smooth deceleration
    if self.awareness_status < -0.:
      self.plan.aTargetMax = min(self.plan.aTargetMax, AWARENESS_DECEL)
      self.plan.aTargetMin = min(self.plan.aTargetMin, self.plan.aTargetMax)

    if enable_request or enable_condition or self.enabled:
      # add all alerts from car
      for alert in self.CS.errors:
        self.AM.add(alert, self.enabled)

      if not self.plan.longitudinalValid:
        self.AM.add("radarCommIssue", self.enabled)

      if self.cal_status != Calibration.CALIBRATED:
        if self.cal_status == Calibration.UNCALIBRATED:
          self.AM.add("calibrationInProgress", self.enabled, str(self.cal_perc) + '%')
        else:
          self.AM.add("calibrationInvalid", self.enabled)

      if not self.plan.lateralValid:
        # If the model is not broadcasting, assume that it is because
        # the user has uploaded insufficient data for calibration.
        # Other cases that would trigger this are rare and unactionable by the user.
        self.AM.add("dataNeeded", self.enabled)

      if self.overtemp:
        self.AM.add("overheat", self.enabled)

    
    # *** angle offset learning *** 
    if self.rk.frame % 5 == 2 and self.plan.lateralValid: 
      # *** run this at 20hz again *** 
      self.angle_offset = learn_angle_offset(self.enabled, self.CS.vEgo, self.angle_offset, 
                                             self.PL.PP.c_poly, self.PL.PP.c_prob, self.LaC.y_des,
                                             self.CS.steeringPressed) 

    # *** gas/brake PID loop *** 
    final_gas, final_brake = self.LoC.update(self.enabled, self.CS.vEgo, self.v_cruise_kph, 
                                        self.plan.vTarget, 
                                        [self.plan.aTargetMin, self.plan.aTargetMax], 
                                        self.plan.jerkFactor, self.CP) 

    # *** steering PID loop *** 
    final_steer, sat_flag = self.LaC.update(self.enabled, self.CS.vEgo, self.CS.steeringAngle, 
                                            self.CS.steeringPressed, self.plan.dPoly, self.angle_offset, self.VM) 
 
    self.prof.checkpoint("PID") 
    
        # ***** handle alerts ****
    # send FCW alert if triggered by planner
    if self.plan.fcw:
      self.AM.add("fcw", self.enabled)

    # send a "steering required alert" if saturation count has reached the limit
    if sat_flag:
      self.AM.add("steerSaturated", self.enabled)

    if self.enabled and self.AM.alertShouldDisable():
      print "DISABLING IMMEDIATELY ON ALERT"
      self.enabled = False

    if self.enabled and self.AM.alertShouldSoftDisable():
      if self.soft_disable_timer is None:
        self.soft_disable_timer = 3 * self.rate
      elif self.soft_disable_timer == 0:
        print "SOFT DISABLING ON ALERT"
        self.enabled = False
      else:
        self.soft_disable_timer -= 1
    else:
      self.soft_disable_timer = None

    if enable_condition and not self.enabled and not self.AM.alertPresent():
      print "*** enabling controls"

      # beep for enabling
      self.AM.add("enable", self.enabled)

      # enable both lateral and longitudinal controls
      self.enabled = True

      # on activation, let's always set v_cruise from where we are, even if PCM ACC is active
      self.v_cruise_kph = int(round(max(self.CS.vEgo * CV.MS_TO_KPH, V_CRUISE_ENABLE_MIN)))

      # 6 minutes driver you're on
      self.awareness_status = 1.0

      # reset the PID loops
      self.LaC.reset()
      # start long control at actual speed
      self.LoC.reset(v_pid = self.CS.vEgo)

    # *** push the alerts to current ***
    # TODO: remove output, store them inside AM class instead
    self.alert_text_1, self.alert_text_2, self.visual_alert, self.audible_alert = self.AM.process_alerts(self.cur_time)
 
    # ***** control the car *****
    self.CC.enabled = self.enabled

    self.CC.gas = float(final_gas)
    self.CC.brake = float(final_brake)
    self.CC.steeringTorque = float(final_steer)

    self.CC.cruiseControl.override = True
    # always cancel if we have an interceptor
    self.CC.cruiseControl.cancel = bool(not self.CP.enableCruise or 
                                        (not self.enabled and self.CS.cruiseState.enabled))

    # brake discount removes a sharp nonlinearity
    brake_discount = (1.0 - clip(final_brake*3., 0.0, 1.0))
    self.CC.cruiseControl.speedOverride = float(max(0.0, ((self.LoC.v_pid - .5) * brake_discount)) if self.CP.enableCruise else 0.0)

    #CC.cruiseControl.accelOverride = float(AC.a_pcm)
    # TODO: parametrize 0.714 in interface?
    # accelOverride is more or less the max throttle allowed to pcm: usually set to a constant
    # unless aTargetMax is very high and then we scale with it; this helpw in quicker restart
    self.CC.cruiseControl.accelOverride = float(max(0.714, self.plan.aTargetMax/A_ACC_MAX))

    self.CC.hudControl.setSpeed = float(self.v_cruise_kph * CV.KPH_TO_MS)
    self.CC.hudControl.speedVisible = self.enabled
    self.CC.hudControl.lanesVisible = self.enabled
    self.CC.hudControl.leadVisible = self.plan.hasLead
    self.CC.hudControl.visualAlert = self.visual_alert
    self.CC.hudControl.audibleAlert = self.audible_alert

    # TODO: remove it from here and put it in state_transition
    # this alert will apply next controls cycle
    if not self.CI.apply(self.CC):
      self.AM.add("controlsFailed", self.enabled)

  def data_send(self):
    
    # broadcast carControl first
    cc_send = messaging.new_message()
    cc_send.init('carControl')
    cc_send.carControl = copy(self.CC)
    self.carcontrol.send(cc_send.to_bytes())

    self.prof.checkpoint("CarControl")

    # broadcast carState
    cs_send = messaging.new_message()
    cs_send.init('carState')
    cs_send.carState = copy(self.CS)
    self.carstate.send(cs_send.to_bytes())
    
    # ***** publish state to logger *****

    # publish controls state at 100Hz
    dat = messaging.new_message()
    dat.init('live100')

    # show rear view camera on phone if in reverse gear or when button is pressed
    dat.live100.rearViewCam = ('reverseGear' in self.CS.errors and self.rear_view_allowed) or self.rear_view_toggle
    dat.live100.alertText1 = self.alert_text_1
    dat.live100.alertText2 = self.alert_text_2
    dat.live100.awarenessStatus = max(self.awareness_status, 0.0) if self.enabled else 0.0

    # what packets were used to process
    dat.live100.canMonoTimes = list(self.CS.canMonoTimes)
    dat.live100.planMonoTime = self.plan_ts

    # if controls is enabled
    dat.live100.enabled = self.enabled

    # car state
    dat.live100.vEgo = self.CS.vEgo
    dat.live100.angleSteers = self.CS.steeringAngle
    dat.live100.steerOverride = self.CS.steeringPressed

    # longitudinal control state
    dat.live100.vPid = float(self.LoC.v_pid)
    dat.live100.vCruise = float(self.v_cruise_kph)
    dat.live100.upAccelCmd = float(self.LoC.Up_accel_cmd)
    dat.live100.uiAccelCmd = float(self.LoC.Ui_accel_cmd)

    # lateral control state
    dat.live100.yDes = float(self.LaC.y_des)
    dat.live100.angleSteersDes = float(self.LaC.angle_steers_des)
    dat.live100.upSteer = float(self.LaC.Up_steer)
    dat.live100.uiSteer = float(self.LaC.Ui_steer)

    # processed radar state, should add a_pcm?
    dat.live100.vTargetLead = float(self.plan.vTarget)
    dat.live100.aTargetMin = float(self.plan.aTargetMin)
    dat.live100.aTargetMax = float(self.plan.aTargetMax)
    dat.live100.jerkFactor = float(self.plan.jerkFactor)

    # log learned angle offset
    dat.live100.angleOffset = float(self.angle_offset)

    # lag
    dat.live100.cumLagMs = -self.rk.remaining*1000.

    self.live100.send(dat.to_bytes())

    self.prof.checkpoint("Live100")

  def wait(self):
    # *** run loop at fixed rate ***
    if self.rk.keep_time():
      self.prof.display()


def controlsd_thread(gctx, rate=100):
  # start the loop
  set_realtime_priority(2)
  CTRLS = Controls(gctx, rate)
  while 1:
    CTRLS.data_sample()
    CTRLS.state_transition()
    CTRLS.state_control()
    CTRLS.data_send()
    CTRLS.wait()


def main(gctx=None):
  controlsd_thread(gctx, 100)

if __name__ == "__main__":
  main()
 
