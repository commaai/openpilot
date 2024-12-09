from panda import Panda
from opendbc.car import get_safety_config, structs
from opendbc.car.interfaces import CarInterfaceBase
from opendbc.car.volkswagen.values import CAR, NetworkLocation, TransmissionType, VolkswagenFlags


class CarInterface(CarInterfaceBase):
  @staticmethod
  def _get_params(ret: structs.CarParams, candidate: CAR, fingerprint, car_fw, experimental_long, docs) -> structs.CarParams:
    ret.carName = "volkswagen"
    ret.radarUnavailable = True

    if ret.flags & VolkswagenFlags.PQ:
      # Set global PQ35/PQ46/NMS parameters
      ret.safetyConfigs = [get_safety_config(structs.CarParams.SafetyModel.volkswagenPq)]
      ret.enableBsm = 0x3BA in fingerprint[0]  # SWA_1

      if 0x440 in fingerprint[0] or docs:  # Getriebe_1
        ret.transmissionType = TransmissionType.automatic
      else:
        ret.transmissionType = TransmissionType.manual

      if any(msg in fingerprint[1] for msg in (0x1A0, 0xC2)):  # Bremse_1, Lenkwinkel_1
        ret.networkLocation = NetworkLocation.gateway
      else:
        ret.networkLocation = NetworkLocation.fwdCamera

      # The PQ port is in dashcam-only mode due to a fixed six-minute maximum timer on HCA steering. An unsupported
      # EPS flash update to work around this timer, and enable steering down to zero, is available from:
      #   https://github.com/pd0wm/pq-flasher
      # It is documented in a four-part blog series:
      #   https://blog.willemmelching.nl/carhacking/2022/01/02/vw-part1/
      # Panda ALLOW_DEBUG firmware required.
      ret.dashcamOnly = True

    else:
      # Set global MQB parameters
      ret.safetyConfigs = [get_safety_config(structs.CarParams.SafetyModel.volkswagen)]
      ret.enableBsm = 0x30F in fingerprint[0]  # SWA_01

      if 0xAD in fingerprint[0] or docs:  # Getriebe_11
        ret.transmissionType = TransmissionType.automatic
      elif 0x187 in fingerprint[0]:  # Motor_EV_01
        ret.transmissionType = TransmissionType.direct
      else:
        ret.transmissionType = TransmissionType.manual

      if any(msg in fingerprint[1] for msg in (0x40, 0x86, 0xB2, 0xFD)):  # Airbag_01, LWI_01, ESP_19, ESP_21
        ret.networkLocation = NetworkLocation.gateway
      else:
        ret.networkLocation = NetworkLocation.fwdCamera

      if 0x126 in fingerprint[2]:  # HCA_01
        ret.flags |= VolkswagenFlags.STOCK_HCA_PRESENT.value

    # Global lateral tuning defaults, can be overridden per-vehicle

    ret.steerLimitTimer = 0.4
    if ret.flags & VolkswagenFlags.PQ:
      ret.steerActuatorDelay = 0.2
      CarInterfaceBase.configure_torque_tune(candidate, ret.lateralTuning)
    else:
      ret.steerActuatorDelay = 0.1
      ret.lateralTuning.pid.kpBP = [0.]
      ret.lateralTuning.pid.kiBP = [0.]
      ret.lateralTuning.pid.kf = 0.00006
      ret.lateralTuning.pid.kpV = [0.6]
      ret.lateralTuning.pid.kiV = [0.2]

    # Global longitudinal tuning defaults, can be overridden per-vehicle

    ret.experimentalLongitudinalAvailable = ret.networkLocation == NetworkLocation.gateway or docs
    if experimental_long:
      # Proof-of-concept, prep for E2E only. No radar points available. Panda ALLOW_DEBUG firmware required.
      ret.openpilotLongitudinalControl = True
      ret.safetyConfigs[0].safetyParam |= Panda.FLAG_VOLKSWAGEN_LONG_CONTROL
      if ret.transmissionType == TransmissionType.manual:
        ret.minEnableSpeed = 4.5

    ret.pcmCruise = not ret.openpilotLongitudinalControl
    ret.stopAccel = -0.55
    ret.vEgoStarting = 0.1
    ret.vEgoStopping = 0.5
    ret.autoResumeSng = ret.minEnableSpeed == -1

    return ret
