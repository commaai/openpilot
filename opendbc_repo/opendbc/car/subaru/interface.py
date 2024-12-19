from panda import Panda
from opendbc.car import get_safety_config, structs
from opendbc.car.disable_ecu import disable_ecu
from opendbc.car.interfaces import CarInterfaceBase
from opendbc.car.subaru.values import CAR, GLOBAL_ES_ADDR, SubaruFlags


class CarInterface(CarInterfaceBase):

  @staticmethod
  def _get_params(ret: structs.CarParams, candidate: CAR, fingerprint, car_fw, experimental_long, docs) -> structs.CarParams:
    ret.carName = "subaru"
    ret.radarUnavailable = True
    # for HYBRID CARS to be upstreamed, we need:
    # - replacement for ES_Distance so we can cancel the cruise control
    # - to find the Cruise_Activated bit from the car
    # - proper panda safety setup (use the correct cruise_activated bit, throttle from Throttle_Hybrid, etc)
    ret.dashcamOnly = bool(ret.flags & (SubaruFlags.PREGLOBAL | SubaruFlags.LKAS_ANGLE | SubaruFlags.HYBRID))
    ret.autoResumeSng = False

    # Detect infotainment message sent from the camera
    if not (ret.flags & SubaruFlags.PREGLOBAL) and 0x323 in fingerprint[2]:
      ret.flags |= SubaruFlags.SEND_INFOTAINMENT.value

    if ret.flags & SubaruFlags.PREGLOBAL:
      ret.enableBsm = 0x25c in fingerprint[0]
      ret.safetyConfigs = [get_safety_config(structs.CarParams.SafetyModel.subaruPreglobal)]
    else:
      ret.enableBsm = 0x228 in fingerprint[0]
      ret.safetyConfigs = [get_safety_config(structs.CarParams.SafetyModel.subaru)]
      if ret.flags & SubaruFlags.GLOBAL_GEN2:
        ret.safetyConfigs[0].safetyParam |= Panda.FLAG_SUBARU_GEN2

    ret.steerLimitTimer = 0.4
    ret.steerActuatorDelay = 0.1

    if ret.flags & SubaruFlags.LKAS_ANGLE:
      ret.steerControlType = structs.CarParams.SteerControlType.angle
    else:
      CarInterfaceBase.configure_torque_tune(candidate, ret.lateralTuning)

    if candidate in (CAR.SUBARU_ASCENT, CAR.SUBARU_ASCENT_2023):
      ret.steerActuatorDelay = 0.3  # end-to-end angle controller
      ret.lateralTuning.init('pid')
      ret.lateralTuning.pid.kf = 0.00003
      ret.lateralTuning.pid.kiBP, ret.lateralTuning.pid.kpBP = [[0., 20.], [0., 20.]]
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.0025, 0.1], [0.00025, 0.01]]

    elif candidate == CAR.SUBARU_IMPREZA:
      ret.steerActuatorDelay = 0.4  # end-to-end angle controller
      ret.lateralTuning.init('pid')
      ret.lateralTuning.pid.kf = 0.00005
      ret.lateralTuning.pid.kiBP, ret.lateralTuning.pid.kpBP = [[0., 20.], [0., 20.]]
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.2, 0.3], [0.02, 0.03]]

    elif candidate == CAR.SUBARU_IMPREZA_2020:
      ret.lateralTuning.init('pid')
      ret.lateralTuning.pid.kf = 0.00005
      ret.lateralTuning.pid.kiBP, ret.lateralTuning.pid.kpBP = [[0., 14., 23.], [0., 14., 23.]]
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.045, 0.042, 0.20], [0.04, 0.035, 0.045]]

    elif candidate == CAR.SUBARU_CROSSTREK_HYBRID:
      ret.steerActuatorDelay = 0.1

    elif candidate in (CAR.SUBARU_FORESTER, CAR.SUBARU_FORESTER_2022, CAR.SUBARU_FORESTER_HYBRID):
      ret.lateralTuning.init('pid')
      ret.lateralTuning.pid.kf = 0.000038
      ret.lateralTuning.pid.kiBP, ret.lateralTuning.pid.kpBP = [[0., 14., 23.], [0., 14., 23.]]
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.01, 0.065, 0.2], [0.001, 0.015, 0.025]]

    elif candidate in (CAR.SUBARU_OUTBACK, CAR.SUBARU_LEGACY, CAR.SUBARU_OUTBACK_2023):
      ret.steerActuatorDelay = 0.1

    elif candidate in (CAR.SUBARU_FORESTER_PREGLOBAL, CAR.SUBARU_OUTBACK_PREGLOBAL_2018):
      ret.safetyConfigs[0].safetyParam = Panda.FLAG_SUBARU_PREGLOBAL_REVERSED_DRIVER_TORQUE  # Outback 2018-2019 and Forester have reversed driver torque signal

    elif candidate == CAR.SUBARU_LEGACY_PREGLOBAL:
      ret.steerActuatorDelay = 0.15

    elif candidate == CAR.SUBARU_OUTBACK_PREGLOBAL:
      pass
    else:
      raise ValueError(f"unknown car: {candidate}")

    ret.experimentalLongitudinalAvailable = not (ret.flags & (SubaruFlags.GLOBAL_GEN2 | SubaruFlags.PREGLOBAL |
                                                              SubaruFlags.LKAS_ANGLE | SubaruFlags.HYBRID))
    ret.openpilotLongitudinalControl = experimental_long and ret.experimentalLongitudinalAvailable

    if ret.flags & SubaruFlags.GLOBAL_GEN2 and ret.openpilotLongitudinalControl:
      ret.flags |= SubaruFlags.DISABLE_EYESIGHT.value

    if ret.openpilotLongitudinalControl:
      ret.safetyConfigs[0].safetyParam |= Panda.FLAG_SUBARU_LONG

    return ret

  @staticmethod
  def init(CP, can_recv, can_send):
    if CP.flags & SubaruFlags.DISABLE_EYESIGHT:
      disable_ecu(can_recv, can_send, bus=2, addr=GLOBAL_ES_ADDR, com_cont_req=b'\x28\x03\x01')
