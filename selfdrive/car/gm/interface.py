#!/usr/bin/env python3
from cereal import car
from selfdrive.config import Conversions as CV
from selfdrive.controls.lib.drive_helpers import create_event, EventTypes as ET
from selfdrive.controls.lib.vehicle_model import VehicleModel
from selfdrive.car.gm.values import DBC, CAR, Ecu, ECU_FINGERPRINT, \
                                    SUPERCRUISE_CARS, AccState, FINGERPRINTS
from selfdrive.car.gm.carstate import CarState, CruiseButtons, get_powertrain_can_parser
from selfdrive.car import STD_CARGO_KG, scale_rot_inertia, scale_tire_stiffness, is_ecu_disconnected, gen_empty_fingerprint
from selfdrive.car.interfaces import CarInterfaceBase

ButtonType = car.CarState.ButtonEvent.Type

class CanBus(CarInterfaceBase):
  def __init__(self):
    self.powertrain = 0
    self.obstacle = 1
    self.chassis = 2
    self.sw_gmlan = 3

class CarInterface(CarInterfaceBase):
  def __init__(self, CP, CarController):
    self.CP = CP

    self.frame = 0
    self.gas_pressed_prev = False
    self.brake_pressed_prev = False
    self.acc_active_prev = 0

    # *** init the major players ***
    canbus = CanBus()
    self.CS = CarState(CP, canbus)
    self.VM = VehicleModel(CP)
    self.pt_cp = get_powertrain_can_parser(CP, canbus)
    self.ch_cp_dbc_name = DBC[CP.carFingerprint]['chassis']

    self.CC = None
    if CarController is not None:
      self.CC = CarController(canbus, CP.carFingerprint)

  @staticmethod
  def compute_gb(accel, speed):
    return float(accel) / 4.0

  @staticmethod
  def get_params(candidate, fingerprint=gen_empty_fingerprint(), has_relay=False, car_fw=[]):
    ret = car.CarParams.new_message()

    ret.carName = "gm"
    ret.carFingerprint = candidate
    ret.isPandaBlack = has_relay

    ret.enableCruise = False
    # GM port is considered a community feature, since it disables AEB;
    # TODO: make a port that uses a car harness and it only intercepts the camera
    ret.communityFeature = True

    # Presence of a camera on the object bus is ok.
    # Have to go to read_only if ASCM is online (ACC-enabled cars),
    # or camera is on powertrain bus (LKA cars without ACC).
    ret.enableCamera = is_ecu_disconnected(fingerprint[0], FINGERPRINTS, ECU_FINGERPRINT, candidate, Ecu.fwdCamera) or \
                       has_relay or \
                       candidate == CAR.CADILLAC_CT6
    ret.openpilotLongitudinalControl = ret.enableCamera
    tire_stiffness_factor = 0.444  # not optimized yet

    if candidate == CAR.VOLT:
      # supports stop and go, but initial engage must be above 18mph (which include conservatism)
      ret.minEnableSpeed = 18 * CV.MPH_TO_MS
      ret.mass = 1607. + STD_CARGO_KG
      ret.safetyModel = car.CarParams.SafetyModel.gm
      ret.wheelbase = 2.69
      ret.steerRatio = 15.7
      ret.steerRatioRear = 0.
      ret.centerToFront = ret.wheelbase * 0.4 # wild guess

    elif candidate == CAR.MALIBU:
      # supports stop and go, but initial engage must be above 18mph (which include conservatism)
      ret.minEnableSpeed = 18 * CV.MPH_TO_MS
      ret.mass = 1496. + STD_CARGO_KG
      ret.safetyModel = car.CarParams.SafetyModel.gm
      ret.wheelbase = 2.83
      ret.steerRatio = 15.8
      ret.steerRatioRear = 0.
      ret.centerToFront = ret.wheelbase * 0.4 # wild guess

    elif candidate == CAR.HOLDEN_ASTRA:
      ret.mass = 1363. + STD_CARGO_KG
      ret.wheelbase = 2.662
      # Remaining parameters copied from Volt for now
      ret.centerToFront = ret.wheelbase * 0.4
      ret.minEnableSpeed = 18 * CV.MPH_TO_MS
      ret.safetyModel = car.CarParams.SafetyModel.gm
      ret.steerRatio = 15.7
      ret.steerRatioRear = 0.

    elif candidate == CAR.ACADIA:
      ret.minEnableSpeed = -1. # engage speed is decided by pcm
      ret.mass = 4353. * CV.LB_TO_KG + STD_CARGO_KG
      ret.safetyModel = car.CarParams.SafetyModel.gm
      ret.wheelbase = 2.86
      ret.steerRatio = 14.4  #end to end is 13.46
      ret.steerRatioRear = 0.
      ret.centerToFront = ret.wheelbase * 0.4

    elif candidate == CAR.BUICK_REGAL:
      ret.minEnableSpeed = 18 * CV.MPH_TO_MS
      ret.mass = 3779. * CV.LB_TO_KG + STD_CARGO_KG # (3849+3708)/2
      ret.safetyModel = car.CarParams.SafetyModel.gm
      ret.wheelbase = 2.83 #111.4 inches in meters
      ret.steerRatio = 14.4 # guess for tourx
      ret.steerRatioRear = 0.
      ret.centerToFront = ret.wheelbase * 0.4 # guess for tourx

    elif candidate == CAR.CADILLAC_ATS:
      ret.minEnableSpeed = 18 * CV.MPH_TO_MS
      ret.mass = 1601. + STD_CARGO_KG
      ret.safetyModel = car.CarParams.SafetyModel.gm
      ret.wheelbase = 2.78
      ret.steerRatio = 15.3
      ret.steerRatioRear = 0.
      ret.centerToFront = ret.wheelbase * 0.49

    elif candidate == CAR.CADILLAC_CT6:
      # engage speed is decided by pcm
      ret.minEnableSpeed = -1.
      ret.mass = 4016. * CV.LB_TO_KG + STD_CARGO_KG
      ret.safetyModel = car.CarParams.SafetyModel.cadillac
      ret.wheelbase = 3.11
      ret.steerRatio = 14.6   # it's 16.3 without rear active steering
      ret.steerRatioRear = 0. # TODO: there is RAS on this car!
      ret.centerToFront = ret.wheelbase * 0.465


    # TODO: get actual value, for now starting with reasonable value for
    # civic and scaling by mass and wheelbase
    ret.rotationalInertia = scale_rot_inertia(ret.mass, ret.wheelbase)

    # TODO: start from empirically derived lateral slip stiffness for the civic and scale by
    # mass and CG position, so all cars will have approximately similar dyn behaviors
    ret.tireStiffnessFront, ret.tireStiffnessRear = scale_tire_stiffness(ret.mass, ret.wheelbase, ret.centerToFront,
                                                                         tire_stiffness_factor=tire_stiffness_factor)

    # same tuning for Volt and CT6 for now
    ret.lateralTuning.pid.kiBP, ret.lateralTuning.pid.kpBP = [[0.], [0.]]
    ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.2], [0.00]]
    ret.lateralTuning.pid.kf = 0.00004   # full torque for 20 deg at 80mph means 0.00007818594

    ret.steerMaxBP = [0.]  # m/s
    ret.steerMaxV = [1.]
    ret.gasMaxBP = [0.]
    ret.gasMaxV = [.5]
    ret.brakeMaxBP = [0.]
    ret.brakeMaxV = [1.]

    ret.longitudinalTuning.kpBP = [5., 35.]
    ret.longitudinalTuning.kpV = [2.4, 1.5]
    ret.longitudinalTuning.kiBP = [0.]
    ret.longitudinalTuning.kiV = [0.36]
    ret.longitudinalTuning.deadzoneBP = [0.]
    ret.longitudinalTuning.deadzoneV = [0.]

    ret.stoppingControl = True
    ret.startAccel = 0.8

    ret.steerActuatorDelay = 0.1  # Default delay, not measured yet
    ret.steerRateCost = 1.0
    ret.steerLimitTimer = 0.4
    ret.radarTimeStep = 0.0667  # GM radar runs at 15Hz instead of standard 20Hz
    ret.steerControlType = car.CarParams.SteerControlType.torque

    return ret

  # returns a car.CarState
  def update(self, c, can_strings):
    self.pt_cp.update_strings(can_strings)

    self.CS.update(self.pt_cp)

    # create message
    ret = car.CarState.new_message()

    ret.canValid = self.pt_cp.can_valid

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

    # gas pedal information.
    ret.gas = self.CS.pedal_gas / 254.0
    ret.gasPressed = self.CS.user_gas_pressed

    # brake pedal
    ret.brake = self.CS.user_brake / 0xd0
    ret.brakePressed = self.CS.brake_pressed

    # steering wheel
    ret.steeringAngle = self.CS.angle_steers

    # torque and user override. Driver awareness
    # timer resets when the user uses the steering wheel.
    ret.steeringPressed = self.CS.steer_override
    ret.steeringTorque = self.CS.steer_torque_driver
    ret.steeringRateLimited = self.CC.steer_rate_limited if self.CC is not None else False

    # cruise state
    ret.cruiseState.available = bool(self.CS.main_on)
    cruiseEnabled = self.CS.pcm_acc_status != AccState.OFF
    ret.cruiseState.enabled = cruiseEnabled
    ret.cruiseState.standstill = self.CS.pcm_acc_status == 4

    ret.leftBlinker = self.CS.left_blinker_on
    ret.rightBlinker = self.CS.right_blinker_on
    ret.doorOpen = not self.CS.door_all_closed
    ret.seatbeltUnlatched = not self.CS.seatbelt
    ret.gearShifter = self.CS.gear_shifter

    buttonEvents = []

    # blinkers
    if self.CS.left_blinker_on != self.CS.prev_left_blinker_on:
      be = car.CarState.ButtonEvent.new_message()
      be.type = ButtonType.leftBlinker
      be.pressed = self.CS.left_blinker_on
      buttonEvents.append(be)

    if self.CS.right_blinker_on != self.CS.prev_right_blinker_on:
      be = car.CarState.ButtonEvent.new_message()
      be.type = ButtonType.rightBlinker
      be.pressed = self.CS.right_blinker_on
      buttonEvents.append(be)

    if self.CS.cruise_buttons != self.CS.prev_cruise_buttons:
      be = car.CarState.ButtonEvent.new_message()
      be.type = ButtonType.unknown
      if self.CS.cruise_buttons != CruiseButtons.UNPRESS:
        be.pressed = True
        but = self.CS.cruise_buttons
      else:
        be.pressed = False
        but = self.CS.prev_cruise_buttons
      if but == CruiseButtons.RES_ACCEL:
        if not (cruiseEnabled and self.CS.standstill):
          be.type = ButtonType.accelCruise # Suppress resume button if we're resuming from stop so we don't adjust speed.
      elif but == CruiseButtons.DECEL_SET:
        be.type = ButtonType.decelCruise
      elif but == CruiseButtons.CANCEL:
        be.type = ButtonType.cancel
      elif but == CruiseButtons.MAIN:
        be.type = ButtonType.altButton3
      buttonEvents.append(be)

    ret.buttonEvents = buttonEvents

    events = []
    if self.CS.steer_error:
      events.append(create_event('steerUnavailable', [ET.NO_ENTRY, ET.IMMEDIATE_DISABLE, ET.PERMANENT]))
    if self.CS.steer_not_allowed:
      events.append(create_event('steerTempUnavailable', [ET.NO_ENTRY, ET.WARNING]))
    if ret.doorOpen:
      events.append(create_event('doorOpen', [ET.NO_ENTRY, ET.SOFT_DISABLE]))
    if ret.seatbeltUnlatched:
      events.append(create_event('seatbeltNotLatched', [ET.NO_ENTRY, ET.SOFT_DISABLE]))

    if self.CS.car_fingerprint in SUPERCRUISE_CARS:
      if self.CS.acc_active and not self.acc_active_prev:
        events.append(create_event('pcmEnable', [ET.ENABLE]))
      if not self.CS.acc_active:
        events.append(create_event('pcmDisable', [ET.USER_DISABLE]))

    else:
      if self.CS.brake_error:
        events.append(create_event('brakeUnavailable', [ET.NO_ENTRY, ET.IMMEDIATE_DISABLE, ET.PERMANENT]))
      if not self.CS.gear_shifter_valid:
        events.append(create_event('wrongGear', [ET.NO_ENTRY, ET.SOFT_DISABLE]))
      if self.CS.esp_disabled:
        events.append(create_event('espDisabled', [ET.NO_ENTRY, ET.SOFT_DISABLE]))
      if not self.CS.main_on:
        events.append(create_event('wrongCarMode', [ET.NO_ENTRY, ET.USER_DISABLE]))
      if self.CS.gear_shifter == 3:
        events.append(create_event('reverseGear', [ET.NO_ENTRY, ET.IMMEDIATE_DISABLE]))
      if ret.vEgo < self.CP.minEnableSpeed:
        events.append(create_event('speedTooLow', [ET.NO_ENTRY]))
      if self.CS.park_brake:
        events.append(create_event('parkBrake', [ET.NO_ENTRY, ET.USER_DISABLE]))
      # disable on pedals rising edge or when brake is pressed and speed isn't zero
      if (ret.gasPressed and not self.gas_pressed_prev) or \
        (ret.brakePressed): # and (not self.brake_pressed_prev or ret.vEgo > 0.001)):
        events.append(create_event('pedalPressed', [ET.NO_ENTRY, ET.USER_DISABLE]))
      if ret.gasPressed:
        events.append(create_event('pedalPressed', [ET.PRE_ENABLE]))
      if ret.cruiseState.standstill:
        events.append(create_event('resumeRequired', [ET.WARNING]))
      if self.CS.pcm_acc_status == AccState.FAULTED:
        events.append(create_event('controlsFailed', [ET.NO_ENTRY, ET.IMMEDIATE_DISABLE]))

      # handle button presses
      for b in ret.buttonEvents:
        # do enable on both accel and decel buttons
        if b.type in [ButtonType.accelCruise, ButtonType.decelCruise] and not b.pressed:
          events.append(create_event('buttonEnable', [ET.ENABLE]))
        # do disable on button down
        if b.type == ButtonType.cancel and b.pressed:
          events.append(create_event('buttonCancel', [ET.USER_DISABLE]))

    ret.events = events

    # update previous brake/gas pressed
    self.acc_active_prev = self.CS.acc_active
    self.gas_pressed_prev = ret.gasPressed
    self.brake_pressed_prev = ret.brakePressed

    # cast to reader so it can't be modified
    return ret.as_reader()

  # pass in a car.CarControl
  # to be called @ 100hz
  def apply(self, c):
    hud_v_cruise = c.hudControl.setSpeed
    if hud_v_cruise > 70:
      hud_v_cruise = 0

    # For Openpilot, "enabled" includes pre-enable.
    # In GM, PCM faults out if ACC command overlaps user gas.
    enabled = c.enabled and not self.CS.user_gas_pressed

    can_sends = self.CC.update(enabled, self.CS, self.frame, \
                               c.actuators,
                               hud_v_cruise, c.hudControl.lanesVisible, \
                               c.hudControl.leadVisible, c.hudControl.visualAlert)

    self.frame += 1
    return can_sends
