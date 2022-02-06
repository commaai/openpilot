#!/usr/bin/env python3
from cereal import car
from selfdrive.config import Conversions as CV
from selfdrive.car.ford.values import MAX_ANGLE
from selfdrive.car import STD_CARGO_KG, scale_rot_inertia, scale_tire_stiffness, gen_empty_fingerprint, get_safety_config
from selfdrive.car.interfaces import CarInterfaceBase


class CarInterface(CarInterfaceBase):
  @staticmethod
  def get_params(candidate, fingerprint=gen_empty_fingerprint(), car_fw=None):
    ret = CarInterfaceBase.get_std_params(candidate, fingerprint)
    ret.carName = "ford"
    ret.safetyConfigs = [get_safety_config(car.CarParams.SafetyModel.ford)]
    ret.dashcamOnly = True

    ret.wheelbase = 2.85
    ret.steerRatio = 14.8
    ret.mass = 3045. * CV.LB_TO_KG + STD_CARGO_KG
    ret.lateralTuning.pid.kiBP, ret.lateralTuning.pid.kpBP = [[0.], [0.]]
    ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.01], [0.005]]     # TODO: tune this
    ret.lateralTuning.pid.kf = 1. / MAX_ANGLE   # MAX Steer angle to normalize FF
    ret.steerActuatorDelay = 0.1  # Default delay, not measured yet
    ret.steerLimitTimer = 1.0
    ret.steerRateCost = 1.0
    ret.centerToFront = ret.wheelbase * 0.44
    tire_stiffness_factor = 0.5328

    # TODO: get actual value, for now starting with reasonable value for
    # civic and scaling by mass and wheelbase
    ret.rotationalInertia = scale_rot_inertia(ret.mass, ret.wheelbase)

    # TODO: start from empirically derived lateral slip stiffness for the civic and scale by
    # mass and CG position, so all cars will have approximately similar dyn behaviors
    ret.tireStiffnessFront, ret.tireStiffnessRear = scale_tire_stiffness(ret.mass, ret.wheelbase, ret.centerToFront,
                                                                         tire_stiffness_factor=tire_stiffness_factor)

    ret.steerControlType = car.CarParams.SteerControlType.angle

    return ret

  # returns a car.CarState
  def update(self, c, can_strings):
    # ******************* do can recv *******************
    self.cp.update_strings(can_strings)

    ret = self.CS.update(self.cp)

    ret.canValid = self.cp.can_valid

    # events
    events = self.create_common_events(ret)

    if self.CS.lkas_state not in (2, 3) and ret.vEgo > 13. * CV.MPH_TO_MS and ret.cruiseState.enabled:
      events.add(car.CarEvent.EventName.steerTempUnavailable)

    ret.events = events.to_msg()

    self.CS.out = ret.as_reader()
    return self.CS.out

  # pass in a car.CarControl
  # to be called @ 100hz
  def apply(self, c):

    ret = self.CC.update(c.enabled, self.CS, self.frame, c.actuators,
                               c.hudControl.visualAlert, c.cruiseControl.cancel)

    self.frame += 1
    return ret
