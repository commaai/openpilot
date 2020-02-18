#!/usr/bin/env python3
from cereal import car
from selfdrive.config import Conversions as CV
from selfdrive.controls.lib.drive_helpers import create_event, EventTypes as ET
from selfdrive.controls.lib.vehicle_model import VehicleModel
from selfdrive.car.subaru.values import CAR
from selfdrive.car.subaru.carstate import CarState, get_powertrain_can_parser, get_camera_can_parser
from selfdrive.car import STD_CARGO_KG, scale_rot_inertia, scale_tire_stiffness, gen_empty_fingerprint
from selfdrive.car.interfaces import CarInterfaceBase

ButtonType = car.CarState.ButtonEvent.Type

class CarInterface(CarInterfaceBase):
  def __init__(self, CP, CarController):
    self.CP = CP

    self.frame = 0
    self.enabled_prev = 0
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
  def get_params(candidate, fingerprint=gen_empty_fingerprint(), has_relay=False, car_fw=[]):
    ret = car.CarParams.new_message()

    ret.carName = "subaru"
    ret.radarOffCan = True
    ret.carFingerprint = candidate
    ret.isPandaBlack = has_relay
    ret.safetyModel = car.CarParams.SafetyModel.subaru

    ret.enableCruise = True

    # force openpilot to fake the stock camera, since car harness is not supported yet and old style giraffe (with switches)
    # was never released
    ret.enableCamera = True

    ret.steerRateCost = 0.7
    ret.steerLimitTimer = 0.4

    if candidate in [CAR.IMPREZA]:
      ret.mass = 1568. + STD_CARGO_KG
      ret.wheelbase = 2.67
      ret.centerToFront = ret.wheelbase * 0.5
      ret.steerRatio = 15
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

    # TODO: get actual value, for now starting with reasonable value for
    # civic and scaling by mass and wheelbase
    ret.rotationalInertia = scale_rot_inertia(ret.mass, ret.wheelbase)

    # TODO: start from empirically derived lateral slip stiffness for the civic and scale by
    # mass and CG position, so all cars will have approximately similar dyn behaviors
    ret.tireStiffnessFront, ret.tireStiffnessRear = scale_tire_stiffness(ret.mass, ret.wheelbase, ret.centerToFront)

    return ret

  # returns a car.CarState
  def update(self, c, can_strings):
    self.pt_cp.update_strings(can_strings)
    self.cam_cp.update_strings(can_strings)

    ret = self.CS.update(self.pt_cp, self.cam_cp)

    ret.canValid = self.pt_cp.can_valid and self.cam_cp.can_valid
    ret.steeringRateLimited = self.CC.steer_rate_limited if self.CC is not None else False
    ret.yawRate = self.VM.yaw_rate(ret.steeringAngle * CV.DEG_TO_RAD, ret.vEgo)

    buttonEvents = []
    be = car.CarState.ButtonEvent.new_message()
    be.type = ButtonType.accelCruise
    buttonEvents.append(be)

    events = []
    if ret.seatbeltUnlatched:
      events.append(create_event('seatbeltNotLatched', [ET.NO_ENTRY, ET.SOFT_DISABLE]))

    if ret.doorOpen:
      events.append(create_event('doorOpen', [ET.NO_ENTRY, ET.SOFT_DISABLE]))

    if ret.cruiseState.enabled and not self.enabled_prev:
      events.append(create_event('pcmEnable', [ET.ENABLE]))
    if not ret.cruiseState.enabled:
      events.append(create_event('pcmDisable', [ET.USER_DISABLE]))

    # disable on gas pedal rising edge
    if (ret.gasPressed and not self.gas_pressed_prev):
      events.append(create_event('pedalPressed', [ET.NO_ENTRY, ET.USER_DISABLE]))

    if ret.gasPressed:
      events.append(create_event('pedalPressed', [ET.PRE_ENABLE]))

    ret.events = events

    self.gas_pressed_prev = ret.gasPressed
    self.enabled_prev = ret.cruiseState.enabled

    self.CS.out = ret.as_reader()
    return self.CS.out

  def apply(self, c):
    can_sends = self.CC.update(c.enabled, self.CS, self.frame, c.actuators,
                               c.cruiseControl.cancel, c.hudControl.visualAlert,
                               c.hudControl.leftLaneVisible, c.hudControl.rightLaneVisible)
    self.frame += 1
    return can_sends
