#!/usr/bin/env python
import os
import time
import common.numpy_fast as np

from selfdrive.config import Conversions as CV
from .carstate import CarState
from .carcontroller import CarController, AH
from selfdrive.boardd.boardd import can_capnp_to_can_list

from cereal import car

from selfdrive.services import service_list
import selfdrive.messaging as messaging

NEW_CAN = os.getenv("OLD_CAN") is None
NEW_CAN = False

# Car button codes
#JCT from dbc: VAL_ 69 SpdCtrlLvr_Stat 32 "DN_1ST" 16 "UP_1ST" 8 "DN_2ND" 4 "UP_2ND" 2 "RWD" 1 "FWD" 0 "IDLE" ;
class CruiseButtons:
  DN_1ST = 32 # Set / remove 1 km/h if already set
  UP_1ST = 16 # Set / add 1 km/h if already set
  DN_2ND = 8 # Remove 10 km/h if already set
  UP_2ND = 4 # Add 10 km/h if already set
  RWD    = 2 # Resume (twice would eventually be enable openpilot)
  FWD    = 1 # Cancel
  IDLE   = 0

#car chimes: enumeration from dbc file. Chimes are for alerts and warnings
class CM:
  MUTE = 0
  SINGLE = 3
  DOUBLE = 4
  REPEATED = 1
  CONTINUOUS = 2

#car beepss: enumeration from dbc file. Beeps are for activ and deactiv
class BP:
  MUTE = 0
  SINGLE = 3
  TRIPLE = 2
  REPEATED = 1

class AH: 
  #[alert_idx, value]
  # See dbc files for info on values"
  NONE           = [0, 0]
  FCW            = [1, 0x8]
  STEER          = [2, 1]
  BRAKE_PRESSED  = [3, 10]
  GEAR_NOT_D     = [4, 6]
  SEATBELT       = [5, 5]
  SPEED_TOO_HIGH = [6, 8]

class CarInterface(object):
  def __init__(self, CP, logcan, sendcan=None):
    self.logcan = logcan
    self.CP = CP

    self.frame = 0
    self.can_invalid_count = 0

    # *** init the major players ***
    self.CS = CarState(CP, self.logcan)

    # sending if read only is False
    if sendcan is not None:
      self.sendcan = sendcan
      self.CC = CarController()

  @staticmethod
  def get_params(candidate, fingerprint):

    # pedal
    brake_only = True

    ret = car.CarParams.new_message()

    ret.carName = "tesla"
    ret.radarName = "bosch"
    ret.carFingerprint = candidate

    ret.enableSteer = True
    ret.enableBrake = True
    ret.enableGas = not brake_only
    ret.enableCruise = brake_only
    
    if candidate == "TESLA CLASSIC MODEL S":
        ret.wheelBase = 2.959
        ret.slipFactor = 0.000713841
        ret.steerRatio = 12
        #ret.steerKp, ret.steerKi = 2.5, 0.15 # 2017-07-11am was at 2.5/0.25
        ret.steerKp, ret.steerKi = 1.25, 0.2 # 2017-07-11am was at 2.5/0.25
    else:
        raise ValueError("unsupported car %s" % candidate)

    return ret

  # returns a car.CarState
  def update(self):
    # ******************* do can recv *******************
    can_pub_main = []
    canMonoTimes = []

    if NEW_CAN:
      self.CS.update(can_pub_main)
    else:
      for a in messaging.drain_sock(self.logcan):
        canMonoTimes.append(a.logMonoTime)
        can_pub_main.extend(can_capnp_to_can_list(a.can, [0,2]))
      self.CS.update(can_pub_main)

    # create message
    ret = car.CarState.new_message()

    # speeds
    ret.vEgo = self.CS.v_ego
    ret.wheelSpeeds.fl = self.CS.v_ego #JCT TODO while we find these self.CS.cp.vl[0x1D0]['WHEEL_SPEED_FL']
    ret.wheelSpeeds.fr = self.CS.v_ego #JCT TODO while we find these self.CS.cp.vl[0x1D0]['WHEEL_SPEED_FR']
    ret.wheelSpeeds.rl = self.CS.v_ego #JCT TODO while we find these self.CS.cp.vl[0x1D0]['WHEEL_SPEED_RL']
    ret.wheelSpeeds.rr = self.CS.v_ego #JCT TODO while we find these self.CS.cp.vl[0x1D0]['WHEEL_SPEED_RR']

    # gas pedal
    ret.gas = 0 #self.CS.car_gas / 100.0 #JCT Tesla scaling is 0-100, Honda was 0-255
    if not self.CP.enableGas:
      ret.gasPressed = self.CS.pedal_gas > 0
    else:
      ret.gasPressed = self.CS.user_gas_pressed

    # brake pedal
    ret.brake = self.CS.user_brake
    ret.brakePressed = self.CS.brake_pressed != 0

    # steering wheel
    # TODO: units
    ret.steeringAngle = self.CS.angle_steers #JCT degree range -819.2|819
    ret.steeringTorque = self.CS.cp.vl[880]['EPAS_torsionBarTorque']
    ret.steeringPressed = self.CS.steer_override

    # cruise state
    #ret.cruiseState.enabled = self.CS.pcm_acc_status == 2
    #ret.cruiseState.enabled = self.CS.cruise_buttons == 2 #CruiseButtons.RWD
    ret.cruiseState.enabled = True #(self.CS.cp.vl[69]['VSL_Enbl_Rq'] == 0) #cruise light off on stalk
    #print "VSL_Enbl_Rq: " + str(self.CS.cp.vl[69]['VSL_Enbl_Rq'])
    #print "cruiseState.enabled: " + str(ret.cruiseState.enabled)
    ret.cruiseState.speed = self.CS.v_cruise_pcm * CV.KPH_TO_MS
    #ret.right_blinker = self.CS.right_blinker_on
    #ret.left_blinker = self.CS.left_blinker_on
    # TODO: for now, set to same as cruise light, check if car is in D?
    ret.cruiseState.available = True #(self.CS.cp.vl[69]['VSL_Enbl_Rq'] == 0)

    # TODO: button presses
    buttonEvents = []

    if self.CS.left_blinker_on != self.CS.prev_left_blinker_on:
      be = car.CarState.ButtonEvent.new_message()
      be.type = 'leftBlinker'
      be.pressed = self.CS.left_blinker_on != 0
      buttonEvents.append(be)

    if self.CS.right_blinker_on != self.CS.prev_right_blinker_on:
      be = car.CarState.ButtonEvent.new_message()
      be.type = 'rightBlinker'
      be.pressed = self.CS.right_blinker_on != 0
      buttonEvents.append(be)

    if self.CS.cruise_buttons != self.CS.prev_cruise_buttons:
      #print "Cruise Buttons Debug: " + str(self.CS.cruise_buttons) + " Prev: " + str(self.CS.prev_cruise_buttons)
      be = car.CarState.ButtonEvent.new_message()
      be.type = 'unknown'
      if self.CS.cruise_buttons != 0:
        be.pressed = True
        but = self.CS.cruise_buttons
      else:
        be.pressed = False
        but = self.CS.prev_cruise_buttons
      if but == CruiseButtons.UP_1ST:
        be.type = 'accelCruise'
      elif but == CruiseButtons.DN_1ST:
        be.type = 'decelCruise'
      elif but == CruiseButtons.FWD:
        be.type = 'cancel'
      elif but == 2:
        be.type = 'altButton1'
        #print "altButton1"
      elif but == CruiseButtons.IDLE:
        be.type = 'altButton3'
      buttonEvents.append(be)

    # JCT TODO This will be used to select openpilot or just regular cruise control via cruise ON/OFF button
    # on cruise OFF (yellow led off) openpilot will be used if we can make it work even with led off, we'll see...
    #if self.CS.cruise_setting != self.CS.prev_cruise_setting:
    #  be = car.CarState.ButtonEvent.new_message()
    #  be.type = 'unknown'
    #  if self.CS.cruise_setting != 0:
    #    be.pressed = True
    #    but = self.CS.cruise_setting
    #  else:
    #    be.pressed = False
    #    but = self.CS.prev_cruise_setting
    #  if but == 1:
    #    be.type = 'altButton1'
    #  # TODO: more buttons?
    #  buttonEvents.append(be)
    ret.buttonEvents = buttonEvents

    # errors
    # TODO: I don't like the way capnp does enums
    # These strings aren't checked at compile time
    errors = []
    if not self.CS.can_valid:
      self.can_invalid_count =0#+= 1
      #print "interface.py line 165, can_invalid_count = " + str(self.can_invalid_count)
	  #if self.can_invalid_count >= 5:
      #  errors.append('commIssue')
    else:
      self.can_invalid_count = 0
    if self.CS.steer_error:
      errors.append('steerUnavailable')
    elif self.CS.steer_not_allowed:
      errors.append('steerTempUnavailable')
    if self.CS.brake_error:
      errors.append('brakeUnavailable')
    if not self.CS.gear_shifter_valid:
      errors.append('wrongGear')
    if not self.CS.door_all_closed:
      errors.append('doorOpen')
    if not self.CS.seatbelt:
      errors.append('seatbeltNotLatched')
    if self.CS.esp_disabled:
      errors.append('espDisabled')
    if not self.CS.main_on:
      errors.append('wrongCarMode')
    if self.CS.gear_shifter == 2:
      errors.append('reverseGear')

    ret.errors = errors
    ret.canMonoTimes = canMonoTimes

    # cast to reader so it can't be modified
    #print ret
    return ret.as_reader()

  # pass in a car.CarControl
  # to be called @ 100hz
  def apply(self, c):
    #print c

    if c.hudControl.speedVisible:
      hud_v_cruise = c.hudControl.setSpeed * CV.MS_TO_KPH
    else:
      hud_v_cruise = 255

    hud_alert = {
      "none": AH.NONE, 
      "fcw": AH.FCW,
      "steerRequired": AH.STEER,
      "brakePressed": AH.BRAKE_PRESSED,
      "wrongGear": AH.GEAR_NOT_D,
      "seatbeltUnbuckled": AH.SEATBELT,
      "speedTooHigh": AH.SPEED_TOO_HIGH}[str(c.hudControl.visualAlert)]

    snd_beep, snd_chime = {
      "none": (BP.MUTE, CM.MUTE),
      "beepSingle": (BP.SINGLE, CM.MUTE),
      "beepTriple": (BP.TRIPLE, CM.MUTE),
      "beepRepeated": (BP.REPEATED, CM.MUTE),
      "chimeSingle": (BP.MUTE, CM.SINGLE),
      "chimeDouble": (BP.MUTE, CM.DOUBLE),
      "chimeRepeated": (BP.MUTE, CM.REPEATED),
      "chimeContinuous": (BP.MUTE, CM.CONTINUOUS)}[str(c.hudControl.audibleAlert)]

    pcm_accel = int(np.clip(c.cruiseControl.accelOverride/1.4,0,1)*0xc6)

    read_only = False
    # sending if read only is False
    if not read_only:
        self.CC.update(self.sendcan, c.enabled, self.CS, self.frame, \
          c.gas, c.brake, c.steeringTorque, \
          c.cruiseControl.speedOverride, \
          c.cruiseControl.override, \
          c.cruiseControl.cancel, \
          pcm_accel, \
          hud_v_cruise, c.hudControl.lanesVisible, \
          hud_show_car = c.hudControl.leadVisible, \
          hud_alert = hud_alert, \
          snd_beep = snd_beep, \
          snd_chime = snd_chime)
        
    self.frame += 1
    return not (c.enabled and not self.CC.controls_allowed)
