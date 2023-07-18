#!/usr/bin/env python3
from cereal import car
from panda import Panda
from common.conversions import Conversions as CV
from selfdrive.car import STD_CARGO_KG, get_safety_config
from selfdrive.car.ford.fordcan import CanBus
from selfdrive.car.ford.values import CANFD_CARS, CAR, Ecu
from selfdrive.car.interfaces import CarInterfaceBase

TransmissionType = car.CarParams.TransmissionType
GearShifter = car.CarState.GearShifter


class CarInterface(CarInterfaceBase):
  @staticmethod
  def _get_params(ret, candidate, fingerprint, car_fw, experimental_long, docs):
    ret.carName = "ford"
    ret.dashcamOnly = candidate in {CAR.F_150_MK14}

    ret.radarUnavailable = True
    ret.steerControlType = car.CarParams.SteerControlType.angle
    ret.steerActuatorDelay = 0.2
    ret.steerLimitTimer = 1.0

    CAN = CanBus(fingerprint=fingerprint)
    cfgs = [get_safety_config(car.CarParams.SafetyModel.ford)]
    if CAN.main >= 4:
      cfgs.insert(0, get_safety_config(car.CarParams.SafetyModel.noOutput))
    ret.safetyConfigs = cfgs

    ret.experimentalLongitudinalAvailable = True
    if experimental_long:
      ret.safetyConfigs[-1].safetyParam |= Panda.FLAG_FORD_LONG_CONTROL
      ret.openpilotLongitudinalControl = True

    if candidate in CANFD_CARS:
      ret.safetyConfigs[-1].safetyParam |= Panda.FLAG_FORD_CANFD

    if candidate == CAR.BRONCO_SPORT_MK1:
      ret.wheelbase = 2.67
      ret.steerRatio = 17.7
      ret.mass = 1625 + STD_CARGO_KG

    elif candidate == CAR.ESCAPE_MK4:
      ret.wheelbase = 2.71
      ret.steerRatio = 16.7
      ret.mass = 1750 + STD_CARGO_KG

    elif candidate == CAR.EXPLORER_MK6:
      ret.wheelbase = 3.025
      ret.steerRatio = 16.8
      ret.mass = 2050 + STD_CARGO_KG

    elif candidate == CAR.F_150_MK14:
      # required trim only on SuperCrew
      ret.wheelbase = 3.69
      ret.steerRatio = 17.0
      ret.mass = 2000 + STD_CARGO_KG

    elif candidate == CAR.FOCUS_MK4:
      ret.wheelbase = 2.7
      ret.steerRatio = 15.0
      ret.mass = 1350 + STD_CARGO_KG

    elif candidate == CAR.MAVERICK_MK1:
      ret.wheelbase = 3.076
      ret.steerRatio = 17.0
      ret.mass = 1650 + STD_CARGO_KG

    else:
      raise ValueError(f"Unsupported car: {candidate}")

    # Auto Transmission: 0x732 ECU or Gear_Shift_by_Wire_FD1
    found_ecus = [fw.ecu for fw in car_fw]
    if Ecu.shiftByWire in found_ecus or 0x5A in fingerprint[CAN.main] or docs:
      ret.transmissionType = TransmissionType.automatic
    else:
      ret.transmissionType = TransmissionType.manual
      ret.minEnableSpeed = 20.0 * CV.MPH_TO_MS

    # BSM: Side_Detect_L_Stat, Side_Detect_R_Stat
    # TODO: detect bsm in car_fw?
    ret.enableBsm = 0x3A6 in fingerprint[CAN.main] and 0x3A7 in fingerprint[CAN.main]

    # LCA can steer down to zero
    ret.minSteerSpeed = 0.

    ret.autoResumeSng = ret.minEnableSpeed == -1.
    ret.centerToFront = ret.wheelbase * 0.44
    return ret

  def _update(self, c):
    ret = self.CS.update(self.cp, self.cp_cam)

    events = self.create_common_events(ret, extra_gears=[GearShifter.manumatic])
    if not self.CS.vehicle_sensors_valid:
      events.add(car.CarEvent.EventName.vehicleSensorsInvalid)
    if self.CS.hybrid_platform:
      events.add(car.CarEvent.EventName.startupNoControl)

    ret.events = events.to_msg()

    return ret

  def apply(self, c, now_nanos):
    return self.CC.update(c, self.CS, now_nanos)
