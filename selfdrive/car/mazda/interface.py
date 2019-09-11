#!/usr/bin/env python3
from cereal import car
from selfdrive.config import Conversions as CV
from selfdrive.controls.lib.drive_helpers import create_event, EventTypes as ET
from selfdrive.controls.lib.vehicle_model import VehicleModel
from selfdrive.car.mazda.values import CAR
from selfdrive.car.mazda.carstate import CarState, get_powertrain_can_parser, get_cam_can_parser
from selfdrive.car import STD_CARGO_KG, scale_rot_inertia, scale_tire_stiffness, gen_empty_fingerprint
from selfdrive.car.interfaces import CarInterfaceBase

class CanBus():
  def __init__(self):
    self.powertrain = 0
    self.obstacle = 1
    self.cam = 2

ButtonType = car.CarState.ButtonEvent.Type

class CarInterface(CarInterfaceBase):
  def __init__(self, CP, CarController):
    self.CP = CP

    self.frame = 0
    self.acc_active_prev = 0

    # *** init the major players ***
    canbus = CanBus()
    self.CS = CarState(CP, canbus)
    self.VM = VehicleModel(CP)
    self.pt_cp = get_powertrain_can_parser(CP, canbus)
    self.cam_cp = get_cam_can_parser(CP, canbus)

    self.CC = None
    if CarController is not None:
      self.CC = CarController(canbus, CP.carFingerprint)

  @staticmethod
  def compute_gb(accel, speed):
    return float(accel) / 4.0

  @staticmethod
  def get_params(candidate, fingerprint=gen_empty_fingerprint(), vin="", has_relay=False):
    ret = car.CarParams.new_message()

    ret.carName = "mazda"
    ret.radarOffCan = True
    ret.carFingerprint = candidate
    ret.carVin = vin
    ret.isPandaBlack = has_relay

    ret.safetyModel = car.CarParams.SafetyModel.mazda

    ret.enableCruise = True

    # force openpilot to fake the stock camera, since car harness is not supported yet and old style
    # giraffe (with switches) was never released
    ret.enableCamera = True
    
    tire_stiffness_factor = 0.70   # not optimized yet

    if candidate in [CAR.CX5]:
      ret.mass =  3655 * CV.LB_TO_KG + STD_CARGO_KG
      ret.wheelbase = 2.7
      ret.centerToFront = ret.wheelbase * 0.41
      ret.steerRatio = 15.5

      ret.lateralTuning.pid.kiBP, ret.lateralTuning.pid.kpBP = [[0.], [0.]]
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.2], [0.18]]

      ret.lateralTuning.pid.kf = 0.00004


    ret.steerActuatorDelay = 0.1
    ret.steerRateCost = 1.0
    ret.steerRatioRear = 0.
    ret.steerControlType = car.CarParams.SteerControlType.torque
    ret.steerLimitAlert = True

    # steer limitations VS speed
    ret.steerMaxBP = [0.]  # m/s
    ret.steerMaxV = [1.]


    # No long control in Mazda
    ret.gasMaxBP = [0.]
    ret.gasMaxV = [0.]
    ret.brakeMaxBP = [0.]
    ret.brakeMaxV = [0.]
    ret.longitudinalTuning.deadzoneBP = [0.]
    ret.longitudinalTuning.deadzoneV = [0.]
    ret.longitudinalTuning.kpBP = [0.]
    ret.longitudinalTuning.kpV = [0.]
    ret.longitudinalTuning.kiBP = [0.]
    ret.longitudinalTuning.kiV = [0.]
    
    
    ret.openpilotLongitudinalControl = False
    ret.stoppingControl = False
    ret.startAccel = 0.0

    ret.minEnableSpeed = -1

    # end from gm

    # TODO: get actual value, for now starting with reasonable value for
    # civic and scaling by mass and wheelbase

    ret.rotationalInertia = scale_rot_inertia(ret.mass, ret.wheelbase)

    # TODO: start from empirically derived lateral slip stiffness for the civic and scale by
    # mass and CG position, so all cars will have approximately similar dyn behaviors
    ret.tireStiffnessFront, ret.tireStiffnessRear = scale_tire_stiffness(ret.mass, ret.wheelbase, ret.centerToFront,
                                                                         tire_stiffness_factor=tire_stiffness_factor)

    return ret

  # returns a car.CarState
  def update(self, c, can_strings):

    self.pt_cp.update_strings(can_strings)
    self.cam_cp.update_strings(can_strings)
    
    self.CS.update(self.pt_cp, self.cam_cp)

    # create message
    ret = car.CarState.new_message()

    ret.canValid = self.pt_cp.can_valid and self.cam_cp.can_valid

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

    # steering wheel
    ret.steeringAngle = self.CS.angle_steers
    ret.steeringRate = self.CS.angle_steers_rate

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

    #be = car.CarState.ButtonEvent.new_message()
    #be.type = ButtonType.accelCruise
    #buttonEvents.append(be)

    ret.buttonEvents = buttonEvents

    # torque and user override. Driver awareness
    # timer resets when the user uses the steering wheel.
    ret.steeringPressed = self.CS.steer_override
    ret.steeringTorque = self.CS.steer_torque_driver

    # cruise state
    ret.cruiseState.enabled = bool(self.CS.acc_active)
    ret.cruiseState.speedOffset = 0.

    ret.cruiseState.available = bool(self.CS.main_on)
    ret.leftBlinker = bool(self.CS.left_blinker_on)
    ret.rightBlinker = bool(self.CS.right_blinker_on)

    ret.doorOpen = self.CS.door_open
    ret.seatbeltUnlatched = self.CS.seatbelt_unlatched


    events = []
    if self.CS.acc_active and not self.acc_active_prev:
      events.append(create_event('pcmEnable', [ET.ENABLE]))
    if not self.CS.acc_active:
      events.append(create_event('pcmDisable', [ET.USER_DISABLE]))

    if ret.doorOpen:
      events.append(create_event('doorOpen', [ET.NO_ENTRY, ET.SOFT_DISABLE]))
    if ret.seatbeltUnlatched:
      events.append(create_event('seatbeltNotLatched', [ET.NO_ENTRY, ET.SOFT_DISABLE]))

    if self.CS.low_speed_lockout:
      events.append(create_event('speedTooLow', [ET.NO_ENTRY]))

    # handle button presses
    for b in ret.buttonEvents:
      # do enable on both accel and decel buttons
      if b.type in ["accelCruise", "decelCruise"] and not b.pressed:
        events.append(create_event('buttonEnable', [ET.ENABLE]))
      # do disable on button down
      if b.type == "cancel" and b.pressed:
        events.append(create_event('buttonCancel', [ET.USER_DISABLE]))

    ret.events = events

    self.acc_active_prev = self.CS.acc_active


    # cast to reader so it can't be modified
    return ret.as_reader()

  def apply(self, c):
    can_sends = self.CC.update(c.enabled, self.CS, self.frame, c.actuators)
    self.frame += 1
    return can_sends
