#!/usr/bin/env python
import numpy as np
from common.kalman.simple_kalman import KF1D
from cereal import car, log
from common.numpy_fast import clip, interp
from common.realtime import sec_since_boot
from selfdrive.config import Conversions as CV
from selfdrive.controls.lib.drive_helpers import create_event, EventTypes as ET, get_events
from selfdrive.controls.lib.vehicle_model import VehicleModel
from selfdrive.car.tesla.carstate import CarState, get_can_parser, get_epas_parser
from selfdrive.car.tesla.values import CruiseButtons, CM, BP, AH, CAR
from selfdrive.controls.lib.planner import _A_CRUISE_MAX_V_FOLLOWING

try:
  from selfdrive.car.tesla.carcontroller import CarController
except ImportError:
  CarController = None

AudibleAlert = car.CarControl.HUDControl.AudibleAlert
VisualAlert = car.CarControl.HUDControl.VisualAlert

K_MULT = 0.8 
K_MULTi = 280000.

def tesla_compute_gb(accel, speed):
  return float(accel) / 3.


class CarInterface(object):
  def __init__(self, CP, sendcan=None):
    self.CP = CP

    self.frame = 0
    self.last_enable_pressed = 0
    self.last_enable_sent = 0
    self.gas_pressed_prev = False
    self.brake_pressed_prev = False
    self.can_invalid_count = 0

    self.cp = get_can_parser(CP)
    self.epas_cp = get_epas_parser(CP)

    # *** init the major players ***
    self.CS = CarState(CP)
    self.VM = VehicleModel(CP)

    # sending if read only is False
    if sendcan is not None:
      self.sendcan = sendcan
      self.CC = CarController(self.cp.dbc_name, CP.enableCamera)

    self.compute_gb = tesla_compute_gb

  @staticmethod
  def calc_accel_override(a_ego, a_target, v_ego, v_target):
    # limit the pcm accel cmd if:
    # - v_ego exceeds v_target, or
    # - a_ego exceeds a_target and v_ego is close to v_target

    eA = a_ego - a_target
    valuesA = [1.0, 0.1]
    bpA = [0.3, 1.1]

    eV = v_ego - v_target
    valuesV = [1.0, 0.1]
    bpV = [0.0, 0.5]

    valuesRangeV = [1., 0.]
    bpRangeV = [-1., 0.]

    # only limit if v_ego is close to v_target
    speedLimiter = interp(eV, bpV, valuesV)
    accelLimiter = max(interp(eA, bpA, valuesA), interp(eV, bpRangeV, valuesRangeV))

    # accelOverride is more or less the max throttle allowed to pcm: usually set to a constant
    # unless aTargetMax is very high and then we scale with it; this help in quicker restart

    return float(max(0.714, a_target / max(_A_CRUISE_MAX_V_FOLLOWING))) * min(speedLimiter, accelLimiter)

  @staticmethod
  def get_params(candidate, fingerprint):

    # kg of standard extra cargo to count for drive, gas, etc...
    std_cargo = 136

    # Scaled tire stiffness
    ts_factor = 8 

    ret = car.CarParams.new_message()

    ret.carName = "tesla"
    ret.carFingerprint = candidate

    ret.safetyModel = car.CarParams.SafetyModels.tesla

    ret.enableCamera = True
    ret.enableGasInterceptor = False #keep this False for now
    print "ECU Camera Simulated: ", ret.enableCamera
    print "ECU Gas Interceptor: ", ret.enableGasInterceptor

    ret.enableCruise = not ret.enableGasInterceptor

    mass_models = 4722./2.205 + std_cargo
    wheelbase_models = 2.959
    # RC: I'm assuming center means center of mass, and I think Model S is pretty even between two axles
    centerToFront_models = wheelbase_models * 0.48
    centerToRear_models = wheelbase_models - centerToFront_models
    rotationalInertia_models = 2500
    tireStiffnessFront_models = 85400
    tireStiffnessRear_models = 90000
    # will create Kp and Ki for 0, 20, 40, 60 mph
    ret.steerKiBP, ret.steerKpBP = [[0., 8.94, 17.88, 26.82 ], [0., 8.94, 17.88, 26.82]]
    if candidate == CAR.MODELS:
      stop_and_go = True
      ret.mass = mass_models
      ret.wheelbase = wheelbase_models
      ret.centerToFront = centerToFront_models
      ret.steerRatio = 15.75
      # Kp and Ki for the lateral control for 0, 20, 40, 60 mph
      ret.steerKpV, ret.steerKiV = [[1.20, 0.80, 0.60, 0.30], [0.16, 0.12, 0.08, 0.04]]
      ret.steerKf = 0.00006 # Initial test value TODO: investigate FF steer control for Model S?
      ret.steerActuatorDelay = 0.09
      
      # Kp and Ki for the longitudinal control
      ret.longitudinalKpBP = [0., 5., 35.]
      ret.longitudinalKpV = [1.27/K_MULT , 1.05/K_MULT, 0.85/K_MULT]
      ret.longitudinalKiBP = [0., 5., 35.]
      ret.longitudinalKiV = [0.11/K_MULTi, 0.09/K_MULTi, 0.06/K_MULTi]
      
      #from honda
      #ret.longitudinalKpBP = [0., 5., 35.]
      #ret.longitudinalKpV = [1.2, 0.8, 0.5]
      #ret.longitudinalKiBP = [0., 35.]
      #ret.longitudinalKiV = [0.18, 0.12]
      # from toyota
      #ret.longitudinalKpBP = [0., 5., 35.]
      #ret.longitudinalKpV = [3.6, 2.4, 1.5]
      #ret.longitudinalKiBP = [0., 35.]
      #ret.longitudinalKiV = [0.54, 0.36]
    else:
      raise ValueError("unsupported car %s" % candidate)

    ret.steerControlType = car.CarParams.SteerControlType.angle

    # min speed to enable ACC. if car can do stop and go, then set enabling speed
    # to a negative value, so it won't matter. Otherwise, add 0.5 mph margin to not
    # conflict with PCM acc
    ret.minEnableSpeed = -1. if (stop_and_go or ret.enableGasInterceptor) else 25.5 * CV.MPH_TO_MS

    centerToRear = ret.wheelbase - ret.centerToFront
    # TODO: get actual value, for now starting with reasonable value for Model S
    ret.rotationalInertia = rotationalInertia_models * \
                            ret.mass * ret.wheelbase**2 / (mass_models * wheelbase_models**2)

    # TODO: start from empirically derived lateral slip stiffness and scale by
    # mass and CG position, so all cars will have approximately similar dyn behaviors
    ret.tireStiffnessFront = (tireStiffnessFront_models * ts_factor) * \
                             ret.mass / mass_models * \
                             (centerToRear / ret.wheelbase) / (centerToRear_models / wheelbase_models)
    ret.tireStiffnessRear = (tireStiffnessRear_models * ts_factor) * \
                            ret.mass / mass_models * \
                            (ret.centerToFront / ret.wheelbase) / (centerToFront_models / wheelbase_models)

    # no rear steering, at least on the listed cars above
    ret.steerRatioRear = 0.

    # no max steer limit VS speed
    ret.steerMaxBP = [0.,15.]  # m/s
    ret.steerMaxV = [420.,420.]   # max steer allowed

    ret.gasMaxBP = [0.]  # m/s
    ret.gasMaxV = [0.3] #if ret.enableGasInterceptor else [0.] # max gas allowed
    ret.brakeMaxBP = [0., 20.]  # m/s
    ret.brakeMaxV = [1., 1.]   # max brake allowed - BB: since we are using regen, make this even

    ret.longPidDeadzoneBP = [0., 9.] #BB: added from Toyota to start pedal work; need to tune
    ret.longPidDeadzoneV = [0., 0.] #BB: added from Toyota to start pedal work; need to tune; changed to 0 for now

    ret.stoppingControl = True
    ret.steerLimitAlert = False
    ret.startAccel = 0.5
    ret.steerRateCost = 1.

    return ret

  # returns a car.CarState
  def update(self, c):
    # ******************* do can recv *******************
    canMonoTimes = []

    self.cp.update(int(sec_since_boot() * 1e9), False)
    self.epas_cp.update(int(sec_since_boot() * 1e9), False)

    self.CS.update(self.cp, self.epas_cp)

    # create message
    ret = car.CarState.new_message()

    # speeds
    ret.vEgo = self.CS.v_ego
    ret.aEgo = self.CS.a_ego
    ret.vEgoRaw = self.CS.v_ego_raw
    ret.yawRate = self.VM.yaw_rate(self.CS.angle_steers * CV.DEG_TO_RAD, self.CS.v_ego)
    ret.standstill = self.CS.standstill
    ret.wheelSpeeds.fl = self.CS.v_wheel_fl
    ret.wheelSpeeds.fr = self.CS.v_wheel_fr
    ret.wheelSpeeds.rl = self.CS.v_wheel_rl
    ret.wheelSpeeds.rr = self.CS.v_wheel_rr

    # gas pedal, we don't use with with interceptor so it's always 0/False
    ret.gas = self.CS.user_gas 
    if not self.CP.enableGasInterceptor:
      ret.gasPressed = self.CS.user_gas_pressed
    else:
      ret.gasPressed = self.CS.user_gas_pressed

    

    # brake pedal
    ret.brakePressed = (self.CS.brake_pressed != 0) and (self.CS.cstm_btns.get_button_status("brake") == 0)
    # FIXME: read sendcan for brakelights
    brakelights_threshold = 0.1
    ret.brakeLights = bool(self.CS.brake_switch or
                           c.actuators.brake > brakelights_threshold)

    # steering wheel
    ret.steeringAngle = self.CS.angle_steers
    ret.steeringRate = self.CS.angle_steers_rate

    # gear shifter lever
    ret.gearShifter = self.CS.gear_shifter

    ret.steeringTorque = self.CS.steer_torque_driver
    ret.steeringPressed = self.CS.steer_override

    # cruise state
    ret.cruiseState.enabled = True #self.CS.pcm_acc_status != 0
    ret.cruiseState.speed = self.CS.v_cruise_pcm * CV.KPH_TO_MS
    ret.cruiseState.available = bool(self.CS.main_on)
    ret.cruiseState.speedOffset = self.CS.cruise_speed_offset
    ret.cruiseState.standstill = False

    # TODO: button presses
    buttonEvents = []
    ret.leftBlinker = bool(self.CS.left_blinker_on)
    ret.rightBlinker = bool(self.CS.right_blinker_on)


    ret.doorOpen = not self.CS.door_all_closed
    ret.seatbeltUnlatched = not self.CS.seatbelt

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

    if self.CS.cruise_buttons != self.CS.prev_cruise_buttons:
      be = car.CarState.ButtonEvent.new_message()
      be.type = 'unknown'
      be.pressed = bool(self.CS.cruise_buttons)
      buttonEvents.append(be)
    ret.buttonEvents = buttonEvents

    # events
    # TODO: I don't like the way capnp does enums
    # These strings aren't checked at compile time
    events = []
    if not self.CS.can_valid:
      self.can_invalid_count += 1
      if self.can_invalid_count >= 5:
        events.append(create_event('commIssue', [ET.NO_ENTRY, ET.IMMEDIATE_DISABLE]))
    else:
      self.can_invalid_count = 0
    if self.CS.steer_error:
      if self.CS.cstm_btns.get_button_status("steer") == 0:
        events.append(create_event('steerUnavailable', [ET.NO_ENTRY, ET.WARNING]))
    elif self.CS.steer_warning:
      if self.CS.cstm_btns.get_button_status("steer") == 0:
         events.append(create_event('steerTempUnavailable', [ET.NO_ENTRY, ET.IMMEDIATE_DISABLE]))
    if self.CS.brake_error:
      events.append(create_event('brakeUnavailable', [ET.NO_ENTRY, ET.IMMEDIATE_DISABLE, ET.PERMANENT]))
    if not ret.gearShifter == 'drive':
      events.append(create_event('wrongGear', [ET.NO_ENTRY, ET.SOFT_DISABLE]))
    if ret.doorOpen:
      events.append(create_event('doorOpen', [ET.NO_ENTRY, ET.SOFT_DISABLE]))
    if ret.seatbeltUnlatched:
      events.append(create_event('seatbeltNotLatched', [ET.NO_ENTRY, ET.SOFT_DISABLE]))
    if self.CS.esp_disabled:
      events.append(create_event('espDisabled', [ET.NO_ENTRY, ET.SOFT_DISABLE]))
    if not self.CS.main_on:
      events.append(create_event('wrongCarMode', [ET.NO_ENTRY, ET.USER_DISABLE]))
    if ret.gearShifter == 'reverse':
      events.append(create_event('reverseGear', [ET.NO_ENTRY, ET.IMMEDIATE_DISABLE]))
    if self.CS.brake_hold:
      events.append(create_event('brakeHold', [ET.NO_ENTRY, ET.USER_DISABLE]))
    if self.CS.park_brake:
      events.append(create_event('parkBrake', [ET.NO_ENTRY, ET.USER_DISABLE]))


    if self.CP.enableCruise and ret.vEgo < self.CP.minEnableSpeed:
      events.append(create_event('speedTooLow', [ET.NO_ENTRY]))

# Standard OP method to disengage:
# disable on pedals rising edge or when brake is pressed and speed isn't zero
#    if (ret.gasPressed and not self.gas_pressed_prev) or \
#       (ret.brakePressed and (not self.brake_pressed_prev or ret.vEgo > 0.001)):
#      events.append(create_event('steerTempUnavailable', [ET.NO_ENTRY, ET.IMMEDIATE_DISABLE]))

    if (self.CS.cstm_btns.get_button_status("brake")>0):
      if ((self.CS.brake_pressed !=0) != self.brake_pressed_prev): #break not canceling when pressed
        self.CS.cstm_btns.set_button_status("brake", 2 if self.CS.brake_pressed != 0 else 1)
    else:
      if ret.brakePressed:
        events.append(create_event('pedalPressed', [ET.NO_ENTRY, ET.USER_DISABLE]))
    if ret.gasPressed:
      events.append(create_event('pedalPressed', [ET.PRE_ENABLE]))

    # it can happen that car cruise disables while comma system is enabled: need to
    # keep braking if needed or if the speed is very low
    if self.CP.enableCruise and not ret.cruiseState.enabled and c.actuators.brake <= 0.:
      # non loud alert if cruise disbales below 25mph as expected (+ a little margin)
      if ret.vEgo < self.CP.minEnableSpeed + 2.:
        events.append(create_event('speedTooLow', [ET.IMMEDIATE_DISABLE]))
      else:
        events.append(create_event("cruiseDisabled", [ET.IMMEDIATE_DISABLE]))
    if self.CS.CP.minEnableSpeed > 0 and ret.vEgo < 0.001:
      events.append(create_event('manualRestart', [ET.WARNING]))

    cur_time = sec_since_boot()
    enable_pressed = False
    # handle button presses
    for b in ret.buttonEvents:

      # do enable on both accel and decel buttons
      if b.type == "altButton3" and not b.pressed:
        print "enabled pressed at", cur_time
        self.last_enable_pressed = cur_time
        enable_pressed = True

      # do disable on button down
      if b.type == "cancel" and b.pressed:
        events.append(create_event('buttonCancel', [ET.USER_DISABLE]))

    if self.CP.enableCruise:
      # KEEP THIS EVENT LAST! send enable event if button is pressed and there are
      # NO_ENTRY events, so controlsd will display alerts. Also not send enable events
      # too close in time, so a no_entry will not be followed by another one.
      # TODO: button press should be the only thing that triggers enble
      if ((cur_time - self.last_enable_pressed) < 0.2 and
          (cur_time - self.last_enable_sent) > 0.2 and
          ret.cruiseState.enabled) or \
         (enable_pressed and get_events(events, [ET.NO_ENTRY])):
        events.append(create_event('buttonEnable', [ET.ENABLE]))
        self.last_enable_sent = cur_time
    elif enable_pressed:
      events.append(create_event('buttonEnable', [ET.ENABLE]))

    ret.events = events
    ret.canMonoTimes = canMonoTimes

    # update previous brake/gas pressed
    self.gas_pressed_prev = ret.gasPressed
    self.brake_pressed_prev = self.CS.brake_pressed != 0

    # cast to reader so it can't be modified
    return ret.as_reader()

  # pass in a car.CarControl
  # to be called @ 100hz
  def apply(self, c, perception_state=log.Live20Data.new_message()):
    if c.hudControl.speedVisible:
      hud_v_cruise = c.hudControl.setSpeed * CV.MS_TO_KPH
    else:
      hud_v_cruise = 255

    VISUAL_HUD = {
      VisualAlert.none: AH.NONE,
      VisualAlert.fcw: AH.FCW,
      VisualAlert.steerRequired: AH.STEER,
      VisualAlert.brakePressed: AH.BRAKE_PRESSED,
      VisualAlert.wrongGear: AH.GEAR_NOT_D,
      VisualAlert.seatbeltUnbuckled: AH.SEATBELT,
      VisualAlert.speedTooHigh: AH.SPEED_TOO_HIGH}

    AUDIO_HUD = {
      AudibleAlert.none: (BP.MUTE, CM.MUTE),
      AudibleAlert.chimeEngage: (BP.SINGLE, CM.MUTE),
      AudibleAlert.chimeDisengage: (BP.SINGLE, CM.MUTE),
      AudibleAlert.chimeError: (BP.MUTE, CM.DOUBLE),
      AudibleAlert.chimePrompt: (BP.MUTE, CM.SINGLE),
      AudibleAlert.chimeWarning1: (BP.MUTE, CM.DOUBLE),
      AudibleAlert.chimeWarning2: (BP.MUTE, CM.REPEATED),
      AudibleAlert.chimeWarningRepeat: (BP.MUTE, CM.REPEATED)}

    hud_alert = VISUAL_HUD[c.hudControl.visualAlert.raw]
    snd_beep, snd_chime = AUDIO_HUD[c.hudControl.audibleAlert.raw]

    pcm_accel = int(clip(c.cruiseControl.accelOverride,0,1)*0xc6)

    self.CC.update(self.sendcan, c.enabled, self.CS, self.frame, \
      c.actuators, \
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
