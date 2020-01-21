#!/usr/bin/env python3
from cereal import car
from selfdrive.config import Conversions as CV
from selfdrive.controls.lib.drive_helpers import create_event, EventTypes as ET
from selfdrive.car.mazda.values import CAR,  FINGERPRINTS, ECU_FINGERPRINT, ECU
from selfdrive.car.mazda.carstate import CarState
from selfdrive.car import STD_CARGO_KG, scale_rot_inertia, scale_tire_stiffness, gen_empty_fingerprint, is_ecu_disconnected
from selfdrive.car.interfaces import CarInterfaceBase

class CanBus():
  def __init__(self):
    self.powertrain = 0
    self.obstacle = 1
    self.cam = 2

ButtonType = car.CarState.ButtonEvent.Type

class CarInterface(CarInterfaceBase):
  def __init__(self, CP, CarController, CarState):
    super().__init__(CP, CarController, CarState)

    self.gas_pressed_prev = False
    self.low_speed_alert = False

  @staticmethod
  def compute_gb(accel, speed):
    return float(accel) / 4.0

  @staticmethod
  def get_params(candidate, fingerprint=gen_empty_fingerprint(), has_relay=False, car_fw=[]):
    ret = car.CarParams.new_message()

    ret.carName = "mazda"
    ret.radarOffCan = True
    ret.carFingerprint = candidate

    ret.isPandaBlack = has_relay

    ret.safetyModel = car.CarParams.SafetyModel.mazda

    ret.enableCruise = True

    ret.enableCamera = is_ecu_disconnected(fingerprint[0], FINGERPRINTS, ECU_FINGERPRINT, candidate, ECU.CAM) or has_relay

    tire_stiffness_factor = 0.70   # not optimized yet

    if candidate in [CAR.CX5]:
      ret.mass =  3655 * CV.LB_TO_KG + STD_CARGO_KG
      ret.wheelbase = 2.7
      ret.centerToFront = ret.wheelbase * 0.41
      ret.steerRatio = 15.5

      ret.lateralTuning.pid.kiBP, ret.lateralTuning.pid.kpBP = [[0.], [0.]]
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.2], [0.2]]

      ret.lateralTuning.pid.kf = 0.00006


    ret.steerLimitTimer = 0.8
    ret.steerActuatorDelay = 0.1
    ret.steerRateCost = 1.0
    ret.steerRatioRear = 0.
    ret.steerControlType = car.CarParams.SteerControlType.torque

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

    ret.minEnableSpeed = -1.   # enable is done by stock ACC, so ignore this

    # no steer below 45kph
    ret.minSteerSpeed = 45 * CV.KPH_TO_MS

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

    self.cp.update_strings(can_strings)
    self.cp_cam.update_strings(can_strings)

    ret = self.CS.update(self.cp, self.cp_cam)
    ret.canValid = self.cp.can_valid and self.cp_cam.can_valid

    # TODO: button presses
    ret.buttonEvents = []

    events = self.create_common_events(ret)

    if ret.cruiseState.enabled and self.CS.lkas_speed_lock:
      self.low_speed_alert = True
    else:
      self.low_speed_alert = False

    # events
    events = self.create_common_events(ret)

    if self.CS.low_speed_lockout:
      events.append(create_event('speedTooLow', [ET.NO_ENTRY]))
      if ret.cruiseState.enabled  and not self.cruise_enabled_prev:
        ret.cruiseState.enabled = False

    if self.low_speed_alert:
      events.append(create_event('belowSteerSpeed', [ET.WARNING]))

    if self.CS.steer_lkas.handsoff:
      events.append(create_event('steerTempUnavailable', [ET.NO_ENTRY, ET.WARNING]))

    if (ret.gasPressed and not self.gas_pressed_prev):
      ret.cruiseState.enabled = False

    ret.events = events

    self.CS.out = ret.as_reader()
    return self.CS.out

  def apply(self, c):
    can_sends = self.CC.update(c.enabled, self.CS, self.frame, c.actuators)
    self.frame += 1
    return can_sends
