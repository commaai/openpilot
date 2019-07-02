#!/usr/bin/env python
from cereal import car
from common.realtime import sec_since_boot
from selfdrive.config import Conversions as CV
from selfdrive.controls.lib.drive_helpers import create_event, EventTypes as ET
from selfdrive.controls.lib.vehicle_model import VehicleModel
from selfdrive.car.subaru.values import CAR
from selfdrive.car.subaru.carstate import CarState, get_powertrain_can_parser, get_camera_can_parser
from selfdrive.car import STD_CARGO_KG


class CarInterface(object):
  def __init__(self, CP, CarController):
    self.CP = CP

    self.frame = 0
    self.acc_active_prev = 0
    self.gas_pressed_prev = False

    # *** init the major players ***
    self.CS = CarState(CP)
    self.VM = VehicleModel(CP)
    self.pt_cp = get_powertrain_can_parser(CP)
    self.cam_cp = get_camera_can_parser(CP)

    self.gas_pressed_prev = False

    self.CC = None
    if CarController is not None:
      self.CC = CarController(CP.carFingerprint)

  @staticmethod
  def compute_gb(accel, speed):
    return float(accel) / 4.0

  @staticmethod
  def calc_accel_override(a_ego, a_target, v_ego, v_target):
    return 1.0

  @staticmethod
  def get_params(candidate, fingerprint, vin=""):
    ret = car.CarParams.new_message()

    ret.carName = "subaru"
    ret.carFingerprint = candidate
    ret.carVin = vin
    ret.safetyModel = car.CarParams.SafetyModel.subaru

    ret.enableCruise = True
    ret.steerLimitAlert = True

    ret.enableCamera = True

    ret.steerRateCost = 0.5
    ret.lateralTuning.pid.dampTime = 0.1
    ret.lateralTuning.pid.reactMPC = 0.0
    ret.lateralTuning.pid.rateFFGain = 0.4

    if candidate in [CAR.IMPREZA]:
      ret.mass = 1568. + STD_CARGO_KG
      ret.wheelbase = 2.67
      ret.centerToFront = ret.wheelbase * 0.5
      ret.steerRatio = 15
      tire_stiffness_factor = 1.0
      ret.steerActuatorDelay = 0.4   # end-to-end angle controller
      ret.lateralTuning.pid.kf = 0.00005
      ret.lateralTuning.pid.kiBP, ret.lateralTuning.pid.kpBP = [[0., 20.], [0., 20.]]
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.2, 0.3], [0.02, 0.03]]
      ret.steerMaxBP = [0.] # m/s
      ret.steerMaxV = [1.]

    ret.steerControlType = car.CarParams.SteerControlType.torque
    ret.steerRatioRear = 0.
    # testing tuning

    # No long control in subaru
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

    # end from gm

    # hardcoding honda civic 2016 touring params so they can be used to
    # scale unknown params for other cars
    mass_civic = 2923. * CV.LB_TO_KG + STD_CARGO_KG
    wheelbase_civic = 2.70
    centerToFront_civic = wheelbase_civic * 0.4
    centerToRear_civic = wheelbase_civic - centerToFront_civic
    rotationalInertia_civic = 2500
    tireStiffnessFront_civic = 192150
    tireStiffnessRear_civic = 202500
    centerToRear = ret.wheelbase - ret.centerToFront

    # TODO: get actual value, for now starting with reasonable value for
    # civic and scaling by mass and wheelbase
    ret.rotationalInertia = rotationalInertia_civic * \
                            ret.mass * ret.wheelbase**2 / (mass_civic * wheelbase_civic**2)

    # TODO: start from empirically derived lateral slip stiffness for the civic and scale by
    # mass and CG position, so all cars will have approximately similar dyn behaviors
    ret.tireStiffnessFront = (tireStiffnessFront_civic * tire_stiffness_factor) * \
                             ret.mass / mass_civic * \
                             (centerToRear / ret.wheelbase) / (centerToRear_civic / wheelbase_civic)
    ret.tireStiffnessRear = (tireStiffnessRear_civic * tire_stiffness_factor) * \
                            ret.mass / mass_civic * \
                            (ret.centerToFront / ret.wheelbase) / (centerToFront_civic / wheelbase_civic)

    return ret

  # returns a car.CarState
  def update(self, c):
    can_rcv_valid, _ = self.pt_cp.update(int(sec_since_boot() * 1e9), True)
    cam_rcv_valid, _ = self.cam_cp.update(int(sec_since_boot() * 1e9), False)

    self.CS.update(self.pt_cp, self.cam_cp)

    # create message
    ret = car.CarState.new_message()

    ret.canValid = can_rcv_valid and cam_rcv_valid and self.pt_cp.can_valid and self.cam_cp.can_valid

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

    # torque and user override. Driver awareness
    # timer resets when the user uses the steering wheel.
    ret.steeringPressed = self.CS.steer_override
    ret.steeringTorque = self.CS.steer_torque_driver

    ret.gas = self.CS.pedal_gas / 255.
    ret.gasPressed = self.CS.user_gas_pressed

    # cruise state
    ret.cruiseState.enabled = bool(self.CS.acc_active)
    ret.cruiseState.speed = self.CS.v_cruise_pcm * CV.KPH_TO_MS
    ret.cruiseState.available = bool(self.CS.main_on)
    ret.cruiseState.speedOffset = 0.

    ret.leftBlinker = self.CS.left_blinker_on
    ret.rightBlinker = self.CS.right_blinker_on
    ret.seatbeltUnlatched = self.CS.seatbelt_unlatched
    ret.doorOpen = self.CS.door_open

    buttonEvents = []

    # blinkers
    if self.CS.left_blinker_on != self.CS.prev_left_blinker_on:
      be = car.CarState.ButtonEvent.new_message()
      be.type = 'leftBlinker'
      be.pressed = self.CS.left_blinker_on
      buttonEvents.append(be)

    if self.CS.right_blinker_on != self.CS.prev_right_blinker_on:
      be = car.CarState.ButtonEvent.new_message()
      be.type = 'rightBlinker'
      be.pressed = self.CS.right_blinker_on
      buttonEvents.append(be)

    be = car.CarState.ButtonEvent.new_message()
    be.type = 'accelCruise'
    buttonEvents.append(be)


    events = []
    if ret.seatbeltUnlatched:
      events.append(create_event('seatbeltNotLatched', [ET.NO_ENTRY, ET.SOFT_DISABLE]))

    if ret.doorOpen:
      events.append(create_event('doorOpen', [ET.NO_ENTRY, ET.SOFT_DISABLE]))

    if self.CS.acc_active and not self.acc_active_prev:
      events.append(create_event('pcmEnable', [ET.ENABLE]))
    if not self.CS.acc_active:
      events.append(create_event('pcmDisable', [ET.USER_DISABLE]))

    # disable on gas pedal rising edge
    if (ret.gasPressed and not self.gas_pressed_prev):
      events.append(create_event('pedalPressed', [ET.NO_ENTRY, ET.USER_DISABLE]))

    if ret.gasPressed:
      events.append(create_event('pedalPressed', [ET.PRE_ENABLE]))

    ret.events = events

    # update previous brake/gas pressed
    self.gas_pressed_prev = ret.gasPressed
    self.acc_active_prev = self.CS.acc_active

    # cast to reader so it can't be modified
    return ret.as_reader()

  def apply(self, c):
    can_sends = self.CC.update(c.enabled, self.CS, self.frame, c.actuators,
                               c.cruiseControl.cancel, c.hudControl.visualAlert,
                               c.hudControl.leftLaneVisible, c.hudControl.rightLaneVisible)
    self.frame += 1
    return can_sends
