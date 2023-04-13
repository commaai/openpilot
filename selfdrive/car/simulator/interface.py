#!/usr/bin/env python3
from cereal import car
from selfdrive.car.simulator.values import CAR
from selfdrive.car import scale_tire_stiffness, get_safety_config
from selfdrive.car.interfaces import CarInterfaceBase

ButtonType = car.CarState.ButtonEvent.Type
EventName = car.CarEvent.EventName

class CarInterface(CarInterfaceBase):

  @staticmethod
  def _get_params(ret, candidate, fingerprint, car_fw, experimental_long):
    ret.carName = "simulator"
    ret.safetyConfigs = [get_safety_config(car.CarParams.SafetyModel.simulator)]
    ret.radarUnavailable = True

    ret.dashcamOnly = False

    ret.steerActuatorDelay = 0.05
    ret.steerLimitTimer = 0.8
    tire_stiffness_factor = 0.70   # not optimized yet

    ret.openpilotLongitudinalControl = True

    CarInterfaceBase.configure_torque_tune(candidate, ret.lateralTuning, use_steering_angle=False)

    if candidate in (CAR.SIMULATOR):
      ret.mass = 1100
      ret.wheelbase = 3.0
      ret.steerRatio = 1
      #ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.01], [0.01]]
      #ret.lateralTuning.pid.kpBP, ret.lateralTuning.pid.kiBP = [[0.,30.], [0.,30.]]
      #ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.07, 0.01], [0.00,0.00]]
      #ret.lateralTuning.pid.kf = 0.000055   # full torque for 20 deg at 80mph means 0.00007818594
      ret.steerActuatorDelay = 0.1  # Default delay, not measured yet
      ret.longitudinalActuatorDelayLowerBound = 0.05
      ret.longitudinalActuatorDelayUpperBound = 0.05
      ret.stopAccel = 0.0
      ret.longitudinalTuning.kf = 0.05
      ret.longitudinalTuning.kpBP = [0.]
      ret.longitudinalTuning.kpV = [0.]
      ret.longitudinalTuning.kiBP = [0.]
      ret.longitudinalTuning.kiV = [0.]

    ret.centerToFront = ret.wheelbase * 0.41

    # TODO: start from empirically derived lateral slip stiffness for the civic and scale by
    # mass and CG position, so all cars will have approximately similar dyn behaviors
    ret.tireStiffnessFront, ret.tireStiffnessRear = scale_tire_stiffness(ret.mass, ret.wheelbase, ret.centerToFront,
                                                                         tire_stiffness_factor=tire_stiffness_factor)

    return ret

  # returns a car.CarState
  def _update(self, c):
    ret = self.CS.update(self.cp)

    # events
    events = self.create_common_events(ret)

    ret.events = events.to_msg()

    return ret

  def apply(self, c, now_nanos):
    return self.CC.update(c, self.CS, now_nanos)
