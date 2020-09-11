#!/usr/bin/env python3
from cereal import car
from selfdrive.car.nissan.values import CAR
from selfdrive.car import STD_CARGO_KG, scale_rot_inertia, scale_tire_stiffness, gen_empty_fingerprint
from selfdrive.car.interfaces import CarInterfaceBase

class CarInterface(CarInterfaceBase):
  def __init__(self, CP, CarController, CarState):
    super().__init__(CP, CarController, CarState)
    self.cp_adas = self.CS.get_adas_can_parser(CP)

  @staticmethod
  def compute_gb(accel, speed):
    return float(accel) / 4.0

  @staticmethod
  def get_params(candidate, fingerprint=gen_empty_fingerprint(), has_relay=False, car_fw=None):

    ret = CarInterfaceBase.get_std_params(candidate, fingerprint, has_relay)
    ret.carName = "nissan"
    ret.safetyModel = car.CarParams.SafetyModel.nissan

    # Nissan port is a community feature, since we don't own one to test
    ret.communityFeature = True

    ret.steerLimitAlert = False
    ret.enableCamera = True
    ret.steerRateCost = 0.5

    ret.steerActuatorDelay = 0.1
    ret.lateralTuning.pid.kf = 0.00006
    ret.lateralTuning.pid.kiBP, ret.lateralTuning.pid.kpBP = [[0.0], [0.0]]
    ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.01], [0.005]]
    ret.steerMaxBP = [0.]  # m/s
    ret.steerMaxV = [1.]

    if candidate in [CAR.ROGUE, CAR.XTRAIL]:
      ret.mass = 1610 + STD_CARGO_KG
      ret.wheelbase = 2.705
      ret.centerToFront = ret.wheelbase * 0.44
      ret.steerRatio = 17
    elif candidate == CAR.LEAF:
      ret.mass = 1610 + STD_CARGO_KG
      ret.wheelbase = 2.705
      ret.centerToFront = ret.wheelbase * 0.44
      ret.steerRatio = 17

    ret.steerControlType = car.CarParams.SteerControlType.angle
    ret.radarOffCan = True

    # TODO: get actual value, for now starting with reasonable value for
    # civic and scaling by mass and wheelbase
    ret.rotationalInertia = scale_rot_inertia(ret.mass, ret.wheelbase)

    # TODO: start from empirically derived lateral slip stiffness for the civic and scale by
    # mass and CG position, so all cars will have approximately similar dyn behaviors
    ret.tireStiffnessFront, ret.tireStiffnessRear = scale_tire_stiffness(ret.mass, ret.wheelbase, ret.centerToFront)

    return ret

  # returns a car.CarState
  def update(self, c, can_strings):
    self.cp.update_strings(can_strings)
    self.cp_cam.update_strings(can_strings)
    self.cp_adas.update_strings(can_strings)

    ret = self.CS.update(self.cp, self.cp_adas, self.cp_cam)

    ret.canValid = self.cp.can_valid and self.cp_adas.can_valid and self.cp_cam.can_valid

    buttonEvents = []
    be = car.CarState.ButtonEvent.new_message()
    be.type = car.CarState.ButtonEvent.Type.accelCruise
    buttonEvents.append(be)

    events = self.create_common_events(ret)

    if self.CS.lkas_enabled:
      events.add(car.CarEvent.EventName.invalidLkasSetting)

    ret.events = events.to_msg()

    self.CS.out = ret.as_reader()
    return self.CS.out

  def apply(self, c):
    can_sends = self.CC.update(c.enabled, self.CS, self.frame, c.actuators,
                               c.cruiseControl.cancel, c.hudControl.visualAlert,
                               c.hudControl.leftLaneVisible, c.hudControl.rightLaneVisible,
                               c.hudControl.leftLaneDepart, c.hudControl.rightLaneDepart)
    self.frame += 1
    return can_sends
