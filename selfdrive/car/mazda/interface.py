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

    self.low_speed_alert = False

  @staticmethod
  def compute_gb(accel, speed):
    return float(accel) / 4.0

  @staticmethod
  def get_params(candidate, fingerprint=gen_empty_fingerprint(), has_relay=False, car_fw=[]):
    ret = CarInterfaceBase.get_std_params(candidate, fingerprint, has_relay)

    ret.carName = "mazda"
    ret.safetyModel = car.CarParams.SafetyModel.mazda
    ret.radarOffCan = True

    # Mazda port is a community feature for now
    ret.communityFeature = True

    ret.steerActuatorDelay = 0.1
    ret.steerRateCost = 1.0
    ret.steerLimitTimer = 0.8
    tire_stiffness_factor = 0.70   # not optimized yet

    if candidate in [CAR.CX5]:
      ret.mass =  3655 * CV.LB_TO_KG + STD_CARGO_KG
      ret.wheelbase = 2.7
      ret.steerRatio = 15.5

      ret.lateralTuning.pid.kiBP, ret.lateralTuning.pid.kpBP = [[0.], [0.]]
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.2], [0.2]]

      ret.lateralTuning.pid.kf = 0.00006

      # No steer below 45kph
      ret.minSteerSpeed = 45 * CV.KPH_TO_MS


    ret.centerToFront = ret.wheelbase * 0.41

    # TODO: get actual value, for now starting with reasonable value for
    # civic and scaling by mass and wheelbase
    ret.rotationalInertia = scale_rot_inertia(ret.mass, ret.wheelbase)

    # TODO: start from empirically derived lateral slip stiffness for the civic and scale by
    # mass and CG position, so all cars will have approximately similar dyn behaviors
    ret.tireStiffnessFront, ret.tireStiffnessRear = scale_tire_stiffness(ret.mass, ret.wheelbase, ret.centerToFront,
                                                                         tire_stiffness_factor=tire_stiffness_factor)

    ret.enableCamera = is_ecu_disconnected(fingerprint[0], FINGERPRINTS, ECU_FINGERPRINT, candidate, ECU.CAM) or has_relay

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
      if ret.cruiseState.enabled:
        ret.cruiseState.enabled = False

    if self.low_speed_alert:
      events.append(create_event('belowSteerSpeed', [ET.WARNING]))

    if self.CS.steer_lkas.handsoff:
      events.append(create_event('steerTempUnavailable', [ET.NO_ENTRY, ET.WARNING]))

    ret.events = events

    self.CS.out = ret.as_reader()
    return self.CS.out

  def apply(self, c):
    can_sends = self.CC.update(c.enabled, self.CS, self.frame, c.actuators)
    self.frame += 1
    return can_sends
