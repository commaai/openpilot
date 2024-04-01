#!/usr/bin/env python3
from cereal import car
from panda import Panda
from openpilot.selfdrive.car.tesla.values import CANBUS, CAR
from openpilot.selfdrive.car import get_safety_config
from openpilot.selfdrive.car.interfaces import CarInterfaceBase


class CarInterface(CarInterfaceBase):
  @staticmethod
  def _get_params(ret, candidate, fingerprint, car_fw, experimental_long, docs):
    ret.carName = "tesla"

    # There is no safe way to do steer blending with user torque,
    # so the steering behaves like autopilot. This is not
    # how openpilot should be, hence dashcamOnly
    ret.dashcamOnly = True

    ret.steerControlType = car.CarParams.SteerControlType.angle

    # Set kP and kI to 0 over the whole speed range to have the planner accel as actuator command
    ret.longitudinalTuning.kpBP = [0]
    ret.longitudinalTuning.kpV = [0]
    ret.longitudinalTuning.kiBP = [0]
    ret.longitudinalTuning.kiV = [0]
    ret.longitudinalActuatorDelayUpperBound = 0.5 # s
    ret.radarTimeStep = (1.0 / 8) # 8Hz

    # Check if we have messages on an auxiliary panda, and that 0x2bf (DAS_control) is present on the AP powertrain bus
    # If so, we assume that it is connected to the longitudinal harness.
    flags = (Panda.FLAG_TESLA_RAVEN if candidate == CAR.TESLA_MODELS_RAVEN else 0)
    if (CANBUS.autopilot_powertrain in fingerprint.keys()) and (0x2bf in fingerprint[CANBUS.autopilot_powertrain].keys()):
      ret.openpilotLongitudinalControl = True
      flags |= Panda.FLAG_TESLA_LONG_CONTROL
      ret.safetyConfigs = [
        get_safety_config(car.CarParams.SafetyModel.tesla, flags),
        get_safety_config(car.CarParams.SafetyModel.tesla, flags | Panda.FLAG_TESLA_POWERTRAIN),
      ]
    else:
      ret.openpilotLongitudinalControl = False
      ret.safetyConfigs = [get_safety_config(car.CarParams.SafetyModel.tesla, flags)]

    ret.steerLimitTimer = 1.0
    ret.steerActuatorDelay = 0.25
    return ret

  def _update(self, c):
    ret = self.CS.update(self.cp, self.cp_cam)

    ret.events = self.create_common_events(ret).to_msg()

    return ret
