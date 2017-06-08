#!/usr/bin/env python
import time
import numpy as np

from selfdrive.config import Conversions as CV
from selfdrive.car.honda.carstate import CarState
from selfdrive.car.honda.carcontroller import CarController, AH
from selfdrive.boardd.boardd import can_capnp_to_can_list

from cereal import car

import zmq
from selfdrive.services import service_list
import selfdrive.messaging as messaging

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

  # returns a car.CarState
  def update(self):
    # ******************* do can recv *******************
    can_pub_main = []
    canMonoTimes = []

    for a in messaging.drain_sock(self.logcan):
      canMonoTimes.append(a.logMonoTime)
      can_pub_main.extend(can_capnp_to_can_list(a.can, [0,2]))
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
      errors.append('steerTemporarilyUnavailable')
    if self.CS.brake_error:
      errors.append('brakeUnavailable')
    # crvtodo: fix gearbox read.
    if not self.CS.gear_shifter_valid and not self.CS.crv:
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
