#!/usr/bin/env python3
from cereal import car
from common.conversions import Conversions as CV
from selfdrive.car import STD_CARGO_KG, scale_rot_inertia, scale_tire_stiffness, gen_empty_fingerprint
from selfdrive.car.ford.values import TransmissionType, CAR
from selfdrive.car.interfaces import CarInterfaceBase


EventName = car.CarEvent.EventName


class CarInterface(CarInterfaceBase):
  @staticmethod
  def get_params(candidate, fingerprint=gen_empty_fingerprint(), car_fw=None, disable_radar=False):
    ret = CarInterfaceBase.get_std_params(candidate, fingerprint)

    ret.carName = "ford"
    #ret.safetyConfigs = [get_safety_config(car.CarParams.SafetyModel.ford)]
    ret.dashcamOnly = True

    # Angle-based steering
    # TODO: use curvature control when ready
    ret.steerControlType = car.CarParams.SteerControlType.angle
    ret.steerActuatorDelay = 0.1
    ret.steerLimitTimer = 1.0

    # TODO: detect stop-and-go vehicles
    stop_and_go = False

    if candidate == CAR.ESCAPE_MK4:
      ret.wheelbase = 2.71
      ret.steerRatio = 14.3  # Copied from Focus
      tire_stiffness_factor = 0.5328  # Copied from Focus
      ret.mass = 1750 + STD_CARGO_KG

    elif candidate == CAR.FOCUS_MK4:
      ret.wheelbase = 2.7
      ret.steerRatio = 14.3
      tire_stiffness_factor = 0.5328
      ret.mass = 1350 + STD_CARGO_KG

    else:
      raise ValueError(f"Unsupported car: ${candidate}")

    # Auto Transmission: Gear_Shift_by_Wire_FD1
    # TODO: detect transmission in car_fw?
    if 0x5A in fingerprint[0]:
      ret.transmissionType = TransmissionType.automatic
    else:
      ret.transmissionType = TransmissionType.manual

    # BSM: Side_Detect_L_Stat, Side_Detect_R_Stat
    # TODO: detect bsm in car_fw?
    ret.enableBsm = 0x3A6 in fingerprint[0] and 0x3A7 in fingerprint[0]

    # min speed to enable ACC. if car can do stop and go, then set enabling speed
    # to a negative value, so it won't matter.
    ret.minEnableSpeed = -1. if (stop_and_go) else 20. * CV.MPH_TO_MS
    # LCA can steer down to zero
    ret.minSteerSpeed = 0.

    ret.steerRateCost = 1.0
    ret.centerToFront = ret.wheelbase * 0.44

    ret.rotationalInertia = scale_rot_inertia(ret.mass, ret.wheelbase)
    ret.tireStiffnessFront, ret.tireStiffnessRear = scale_tire_stiffness(ret.mass, ret.wheelbase, ret.centerToFront,
                                                                         tire_stiffness_factor=tire_stiffness_factor)

    return ret

  def _update(self, c):
    ret = self.CS.update(self.cp, self.cp_cam)

    ret.steeringRateLimited = self.CC.steer_rate_limited if self.CC is not None else False

    events = self.create_common_events(ret)
    ret.events = events.to_msg()

    return ret

  def apply(self, c):
    ret = self.CC.update(c, self.CS, self.frame)
    self.frame += 1
    return ret
