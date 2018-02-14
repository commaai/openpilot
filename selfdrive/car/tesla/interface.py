#!/usr/bin/env python
import os
import time
import common.numpy_fast as np

from selfdrive.config import Conversions as CV
from .carstate import CarState, get_can_parser
from .carcontroller import CarController, AH
from selfdrive.boardd.boardd import can_capnp_to_can_list

from selfdrive.controls.lib.drive_helpers import EventTypes as ET, create_event
from selfdrive.controls.lib.vehicle_model import VehicleModel

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
  def __init__(self, CP, sendcan=None):
    self.CP = CP

    self.frame = 0
    self.can_invalid_count = 0

    # *** init the major players ***
    self.CS = CarState(CP)
    self.cp = get_can_parser(CP)

    # sending if read only is False
    if sendcan is not None:
      self.sendcan = sendcan
      self.CC = CarController()

  @staticmethod
  def compute_gb(accel, speed):
    return float(accel) / 3.0

  @staticmethod
  def calc_accel_override(a_ego, a_target, v_ego, v_target):
    return 1.0

  @staticmethod
  def get_params(candidate, fingerprint):

    # kg of standard extra cargo to count for drive, gas, etc...
    std_cargo = 136
    # pedal
    brake_only = True

    ret = car.CarParams.new_message()

    ret.carName = "tesla"
    # This causes an error in OP 0.4
    # ret.radarName = "bosch"
    ret.radarName = "mock"
    ret.carFingerprint = candidate
    ret.safetyModel = car.CarParams.SafetyModels.noOutput

    ret.enableSteer = True
    ret.enableBrake = True
    ret.enableGas = not brake_only
    ret.enableCruise = brake_only # Shouldn't this be True (like Toyota)
    
    # FIXME: hardcoding honda civic 2016 touring params so they can be used to
    # scale unknown params for other cars
    mass_civic = 2923./2.205 + std_cargo
    wheelbase_civic = 2.70
    centerToFront_civic = wheelbase_civic * 0.4
    centerToRear_civic = wheelbase_civic - centerToFront_civic
    rotationalInertia_civic = 2500
    tireStiffnessFront_civic = 85400
    tireStiffnessRear_civic = 90000

    ret.mass = 3045./2.205 + std_cargo
    ret.wheelbase = 2.70
    ret.centerToFront = ret.wheelbase * 0.44
    ret.steerRatio = 14.5 #Rav4 2017, TODO: find exact value for Prius
    ret.steerKp, ret.steerKi = 0.6, 0.05
    ret.steerKf = 0.00006   # full torque for 10 deg at 80mph means 0.00007818594

    ret.longPidDeadzoneBP = [0., 9.]
    ret.longPidDeadzoneV = [0., .15]

    # min speed to enable ACC. if car can do stop and go, then set enabling speed
    # to a negative value, so it won't matter.
    ret.minEnableSpeed = 17. * CV.MPH_TO_MS

    centerToRear = ret.wheelbase - ret.centerToFront
    # TODO: get actual value, for now starting with reasonable value for
    # civic and scaling by mass and wheelbase
    ret.rotationalInertia = rotationalInertia_civic * \
                            ret.mass * ret.wheelbase**2 / (mass_civic * wheelbase_civic**2)

    # TODO: start from empirically derived lateral slip stiffness for the civic and scale by
    # mass and CG position, so all cars will have approximately similar dyn behaviors
    ret.tireStiffnessFront = tireStiffnessFront_civic * \
                             ret.mass / mass_civic * \
                             (centerToRear / ret.wheelbase) / (centerToRear_civic / wheelbase_civic)
    ret.tireStiffnessRear = tireStiffnessRear_civic * \
                            ret.mass / mass_civic * \
                            (ret.centerToFront / ret.wheelbase) / (centerToFront_civic / wheelbase_civic)

    # no rear steering, at least on the listed cars above
    ret.steerRatioRear = 0.

    # steer, gas, brake limitations VS speed
    ret.steerMaxBP = [16. * CV.KPH_TO_MS, 45. * CV.KPH_TO_MS]  # breakpoints at 1 and 40 kph
    ret.steerMaxV = [1., 1.]  # 2/3rd torque allowed above 45 kph
    ret.gasMaxBP = [0.]
    ret.gasMaxV = [0.5]
    ret.brakeMaxBP = [5., 20.]
    ret.brakeMaxV = [1., 0.8]

    if candidate == "TESLA CLASSIC MODEL S":
        ret.wheelbase = 2.959
        # This isn't in OP 0.4 (?)
        # ret.slipFactor = 0.000713841
        ret.steerRatio = 12
        #ret.steerKp, ret.steerKi = 2.5, 0.15 # 2017-07-11am was at 2.5/0.25
        ret.steerKp, ret.steerKi = 1.25, 0.2 # 2017-07-11am was at 2.5/0.25
    else:
        raise ValueError("unsupported car %s" % candidate)

    ret.steerLimitAlert = False
    ret.stoppingControl = False
    ret.startAccel = 0.0

    ret.longitudinalKpBP = [0., 5., 35.]
    ret.longitudinalKpV = [3.6, 2.4, 1.5]
    ret.longitudinalKiBP = [0., 35.]
    ret.longitudinalKiV = [0.54, 0.36]

    ret.steerRateCost = 1.

    return ret

  # returns a car.CarState
  def update(self, c):
    # ******************* do can recv *******************
    can_pub_main = []
    canMonoTimes = []

    # This causes error - THIS SEEMS NECESSARY!
    # self.cp.update(int(sec_since_boot() * 1e9), False)

    self.CS.update(self.cp)

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

    # errors - replaced with events
    events = []
    if not self.CS.can_valid:
      self.can_invalid_count =0#+= 1
      #print "interface.py line 165, can_invalid_count = " + str(self.can_invalid_count)
	  #if self.can_invalid_count >= 5:
      #  errors.append('commIssue')
    else:
      self.can_invalid_count = 0
    if self.CS.steer_error:
      events.append(create_event('steerTempUnavailable', [ET.NO_ENTRY, ET.WARNING]))
    elif self.CS.steer_not_allowed:
      events.append(create_event('steerTempUnavailable', [ET.NO_ENTRY, ET.WARNING]))
    if self.CS.brake_error:
      events.append(create_event('brakeUnavailable', [ET.NO_ENTRY, ET.WARNING]))
      # errors.append('brakeUnavailable') - Maybe wrong new message? 
    if not self.CS.gear_shifter_valid:
      events.append(create_event('wrongGear', [ET.NO_ENTRY, ET.SOFT_DISABLE]))
    if not self.CS.door_all_closed:
      events.append(create_event('doorOpen', [ET.NO_ENTRY, ET.SOFT_DISABLE]))
    if not self.CS.seatbelt:
      events.append(create_event('seatbeltNotLatched', [ET.NO_ENTRY, ET.SOFT_DISABLE]))
    if self.CS.esp_disabled:
      events.append(create_event('espDisabled', [ET.NO_ENTRY, ET.SOFT_DISABLE]))
    if not self.CS.main_on:
      events.append(create_event('wrongCarMode', [ET.NO_ENTRY, ET.USER_DISABLE]))
    if self.CS.gear_shifter == 2:
      events.append(create_event('reverseGear', [ET.NO_ENTRY, ET.IMMEDIATE_DISABLE]))

    ret.events = events
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
