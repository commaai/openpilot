#!/usr/bin/env python3
from cereal import car
from selfdrive.car.stellantis.values import CAR
from selfdrive.car import STD_CARGO_KG, scale_rot_inertia, scale_tire_stiffness, gen_empty_fingerprint
from selfdrive.car.interfaces import CarInterfaceBase


class CarInterface(CarInterfaceBase):
  @staticmethod
  def get_params(candidate, fingerprint=gen_empty_fingerprint(), car_fw=None):
    ret = CarInterfaceBase.get_std_params(candidate, fingerprint)
    ret.carName = "stellantis"
    ret.safetyModel = car.CarParams.SafetyModel.stellantis
    ret.radarOffCan = True

    # RAM port is a community feature, since it was made by Tunder using magic
    ret.communityFeature = True

    # Global default tuning params

    ret.lateralTuning.pid.kpBP, ret.lateralTuning.pid.kiBP = [[0.], [0.,]]
    ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.15], [0.015]]
    ret.lateralTuning.pid.kf = 0.00006
    ret.steerActuatorDelay = 0.1  # may need tuning
    ret.steerRateCost = 1.0  # may need tuning
    ret.steerLimitTimer = 0.4
    ret.centerToFront = ret.wheelbase * 0.4 # just a guess
    ret.minSteerSpeed = 14.0  # m/s

    # Per-vehicle tuning params

    if candidate == CAR.RAM_1500_DT:
      ret.wheelbase = 3.88  # 2021 Ram 1500
      ret.steerRatio = 15.  # just a guess
      ret.mass = 2493. + STD_CARGO_KG  # kg curb weight 2021 Ram 1500

    # starting with reasonable value for civic and scaling by mass and wheelbase
    ret.rotationalInertia = scale_rot_inertia(ret.mass, ret.wheelbase)

    # TODO: start from empirically derived lateral slip stiffness for the civic and scale by
    # mass and CG position, so all cars will have approximately similar dyn behaviors
    ret.tireStiffnessFront, ret.tireStiffnessRear = scale_tire_stiffness(ret.mass, ret.wheelbase, ret.centerToFront)

    return ret

  # returns a car.CarState
  def update(self, c, can_strings):

    # Process the most recent CAN message traffic, and check for validity
    self.cp.update_strings(can_strings)
    self.cp_cam.update_strings(can_strings)

    ret = self.CS.update(self.cp, self.cp_cam)
    ret.canValid = self.cp.can_valid and self.cp_cam.can_valid
    ret.steeringRateLimited = self.CC.steer_rate_limited if self.CC is not None else False

    events = self.create_common_events(ret)
    if ret.vEgo < self.CP.minSteerSpeed:
      events.add(car.CarEvent.EventName.belowSteerSpeed)

    ret.events = events.to_msg()
    self.CS.out = ret.as_reader()
    return self.CS.out

  # pass in a car.CarControl
  # to be called @ 100hz
  def apply(self, c):

    can_sends = self.CC.update(c.enabled, self.CS, self.frame, c.actuators, c.cruiseControl.cancel)
    self.frame += 1
    return can_sends
