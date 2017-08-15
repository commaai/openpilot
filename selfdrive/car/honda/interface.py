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

# Car button codes
class CruiseButtons:
  RES_ACCEL   = 4
  DECEL_SET   = 3
  CANCEL      = 2
  MAIN        = 1

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

    if self.CS.accord:
      self.accord_msg = []

  @staticmethod
  def get_params(candidate, fingerprint):

    # kg of standard extra cargo to count for drive, gas, etc...
    std_cargo = 136

    ret = car.CarParams.new_message()

    ret.carName = "honda"
    ret.radarName = "nidec"
    ret.carFingerprint = candidate

    ret.enableSteer = True
    ret.enableBrake = True

    # pedal
    ret.enableGas = 0x201 in fingerprint
    ret.enableCruise = not ret.enableGas

    # FIXME: hardcoding honda civic 2016 touring wight so it can be used to 
    # scale unknown params for other cars
    m_civic = 2923./2.205 + std_cargo
    l_civic = 2.70
    aF_civic = l_civic * 0.4
    aR_civic = l_civic - aF_civic
    j_civic = 2500
    cF_civic = 85400
    cR_civic = 90000

    if candidate == "HONDA CIVIC 2016 TOURING":
      ret.m = m_civic
      ret.l = l_civic
      ret.aF = aF_civic
      ret.sR = 13.0
      ret.steerKp, ret.steerKi = 0.8, 0.24
    elif candidate == "ACURA ILX 2016 ACURAWATCH PLUS":
      ret.m = 3095./2.205 + std_cargo
      ret.l = 2.67
      ret.aF = ret.l * 0.37
      ret.sR = 15.3
      # Acura at comma has modified steering FW, so different tuning for the Neo in that car
      # FIXME: using dongleId isn't great, better to identify the car than the Neo
      is_fw_modified = os.getenv("DONGLE_ID") == 'cb38263377b873ee'
      ret.steerKp, ret.steerKi = [0.4, 0.12] if is_fw_modified else [0.8, 0.24]
    elif candidate == "HONDA ACCORD 2016 TOURING":
      ret.m = 3580./2.205 + std_cargo
      ret.l = 2.74
      ret.aF = ret.l * 0.38
      ret.sR = 15.3
      ret.steerKp, ret.steerKi = 0.8, 0.24
    elif candidate == "HONDA CR-V 2016 TOURING":
      ret.m = 3572./2.205 + std_cargo
      ret.l = 2.62
      ret.aF = ret.l * 0.41
      ret.sR = 15.3
      ret.steerKp, ret.steerKi = 0.8, 0.24
    else:
      raise ValueError("unsupported car %s" % candidate)
 
    ret.aR = ret.l - ret.aF
    # TODO: get actual value, for now starting with reasonable value for 
    # civic and scaling by mass and wheelbase
    ret.j = j_civic * ret.m * ret.l**2 / (m_civic * l_civic**2)

    # TODO: start from empirically derived lateral slip stiffness for the civic and scale by
    # mass and CG position... all cars will have approximately similar dyn behaviors
    ret.cF = cF_civic * ret.m / m_civic * (ret.aR / ret.l) / (aR_civic / l_civic)
    ret.cR = cR_civic * ret.m / m_civic * (ret.aF / ret.l) / (aF_civic / l_civic)

    # no rear steering, at least on the listed cars above
    ret.chi = 0.

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
        can_pub_main.extend(can_capnp_to_can_list(a.can, [0,0x80]))
        if self.CS.accord:
          self.accord_msg.extend(can_capnp_to_can_list(a.can, [9]))
          self.accord_msg = self.accord_msg[-1:]
      self.CS.update(can_pub_main)

    # create message
    ret = car.CarState.new_message()

    # speeds
    ret.vEgo = self.CS.v_ego
    ret.wheelSpeeds.fl = self.CS.cp.vl[0x1D0]['WHEEL_SPEED_FL']
    ret.wheelSpeeds.fr = self.CS.cp.vl[0x1D0]['WHEEL_SPEED_FR']
    ret.wheelSpeeds.rl = self.CS.cp.vl[0x1D0]['WHEEL_SPEED_RL']
    ret.wheelSpeeds.rr = self.CS.cp.vl[0x1D0]['WHEEL_SPEED_RR']

    # gas pedal
    ret.gas = self.CS.car_gas / 256.0
    if not self.CP.enableGas:
      ret.gasPressed = self.CS.pedal_gas > 0
    else:
      ret.gasPressed = self.CS.user_gas_pressed

    # brake pedal
    ret.brake = self.CS.user_brake
    ret.brakePressed = self.CS.brake_pressed != 0

    # steering wheel
    # TODO: units
    ret.steeringAngle = self.CS.angle_steers

    if self.CS.accord:
      # TODO: move this into the CAN parser
      ret.steeringTorque = 0
      if len(self.accord_msg) > 0:
        aa = map(lambda x: ord(x)&0x7f, self.accord_msg[0][2])
        if len(aa) != 5 or (-(aa[0]+aa[1]+aa[2]+aa[3]))&0x7f != aa[4]:
          print "ACCORD MSG BAD LEN OR CHECKSUM!"
          # TODO: throw an error here?
        else:
          st = ((aa[0]&0xF) << 5) + (aa[1]&0x1F)
          if st >= 256:
            st = -(512-st)
          ret.steeringTorque = st
      ret.steeringPressed = abs(ret.steeringTorque) > 20
    else:
      ret.steeringTorque = self.CS.cp.vl[0x18F]['STEER_TORQUE_SENSOR']
      ret.steeringPressed = self.CS.steer_override

    # cruise state
    ret.cruiseState.enabled = self.CS.pcm_acc_status != 0
    ret.cruiseState.speed = self.CS.v_cruise_pcm * CV.KPH_TO_MS
    ret.cruiseState.available = bool(self.CS.main_on)

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
      be = car.CarState.ButtonEvent.new_message()
      be.type = 'unknown'
      if self.CS.cruise_buttons != 0:
        be.pressed = True
        but = self.CS.cruise_buttons
      else:
        be.pressed = False
        but = self.CS.prev_cruise_buttons
      if but == CruiseButtons.RES_ACCEL:
        be.type = 'accelCruise'
      elif but == CruiseButtons.DECEL_SET:
        be.type = 'decelCruise'
      elif but == CruiseButtons.CANCEL:
        be.type = 'cancel'
      elif but == CruiseButtons.MAIN:
        be.type = 'altButton3'
      buttonEvents.append(be)

    if self.CS.cruise_setting != self.CS.prev_cruise_setting:
      be = car.CarState.ButtonEvent.new_message()
      be.type = 'unknown'
      if self.CS.cruise_setting != 0:
        be.pressed = True
        but = self.CS.cruise_setting
      else:
        be.pressed = False
        but = self.CS.prev_cruise_setting
      if but == 1:
        be.type = 'altButton1'
      # TODO: more buttons?
      buttonEvents.append(be)
    ret.buttonEvents = buttonEvents

    # errors
    # TODO: I don't like the way capnp does enums
    # These strings aren't checked at compile time
    errors = []
    if not self.CS.can_valid:
      self.can_invalid_count += 1
      if self.can_invalid_count >= 5:
        errors.append('commIssue')
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

    pcm_accel = int(np.clip(c.cruiseControl.accelOverride,0,1)*0xc6)

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
