from panda import Panda
from openpilot.selfdrive.car import get_safety_config, structs
from openpilot.selfdrive.car.common.conversions import Conversions as CV
from openpilot.selfdrive.car.ford.fordcan import CanBus
from openpilot.selfdrive.car.ford.values import Ecu, FordFlags
from openpilot.selfdrive.car.interfaces import CarInterfaceBase

TransmissionType = structs.CarParams.TransmissionType


class CarInterface(CarInterfaceBase):
  @staticmethod
  def _get_params(ret: structs.CarParams, candidate, fingerprint, car_fw, experimental_long, docs) -> structs.CarParams:
    ret.carName = "ford"
    ret.dashcamOnly = bool(ret.flags & FordFlags.CANFD)

    ret.radarUnavailable = True
    ret.steerControlType = structs.CarParams.SteerControlType.angle
    ret.steerActuatorDelay = 0.2
    ret.steerLimitTimer = 1.0

    CAN = CanBus(fingerprint=fingerprint)
    cfgs = [get_safety_config(structs.CarParams.SafetyModel.ford)]
    if CAN.main >= 4:
      cfgs.insert(0, get_safety_config(structs.CarParams.SafetyModel.noOutput))
    ret.safetyConfigs = cfgs

    ret.experimentalLongitudinalAvailable = True
    if experimental_long:
      ret.safetyConfigs[-1].safetyParam |= Panda.FLAG_FORD_LONG_CONTROL
      ret.openpilotLongitudinalControl = True

    if ret.flags & FordFlags.CANFD:
      ret.safetyConfigs[-1].safetyParam |= Panda.FLAG_FORD_CANFD
    else:
      # Lock out if the car does not have needed lateral and longitudinal control APIs.
      # Note that we also check CAN for adaptive cruise, but no known signal for LCA exists
      pscm_config = next((fw for fw in car_fw if fw.ecu == Ecu.eps and b'\x22\xDE\x01' in fw.request), None)
      if pscm_config:
        if len(pscm_config.fwVersion) != 24:
          ret.dashcamOnly = True
        else:
          config_tja = pscm_config.fwVersion[7]  # Traffic Jam Assist
          config_lca = pscm_config.fwVersion[8]  # Lane Centering Assist
          if config_tja != 0xFF or config_lca != 0xFF:
            ret.dashcamOnly = True

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
