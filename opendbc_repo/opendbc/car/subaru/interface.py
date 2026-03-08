from opendbc.car import get_safety_config, structs, uds
from opendbc.car.disable_ecu import disable_ecu
from opendbc.car.interfaces import CarInterfaceBase
from opendbc.car.subaru.carcontroller import CarController
from opendbc.car.subaru.carstate import CarState
from opendbc.car.subaru.values import CAR, GLOBAL_ES_ADDR, SubaruFlags, SubaruSafetyFlags


class CarInterface(CarInterfaceBase):
  CarState = CarState
  CarController = CarController

  @staticmethod
  def _get_params(ret: structs.CarParams, candidate: CAR, fingerprint, car_fw, alpha_long, is_release, docs) -> structs.CarParams:
    ret.brand = "subaru"
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
        ret.safetyConfigs[0].safetyParam |= SubaruSafetyFlags.GEN2.value

    ret.steerLimitTimer = 0.4
    ret.steerActuatorDelay = 0.1

    if not (ret.flags & SubaruFlags.LKAS_ANGLE):
      CarInterfaceBase.configure_torque_tune(candidate, ret.lateralTuning)

    if ret.flags & SubaruFlags.LKAS_ANGLE:
      ret.steerControlType = structs.CarParams.SteerControlType.angle

    elif candidate == CAR.SUBARU_ASCENT:
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

    elif candidate in (CAR.SUBARU_FORESTER, CAR.SUBARU_FORESTER_HYBRID):
      ret.lateralTuning.init('pid')
      ret.lateralTuning.pid.kf = 0.000038
      ret.lateralTuning.pid.kiBP, ret.lateralTuning.pid.kpBP = [[0., 14., 23.], [0., 14., 23.]]
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.01, 0.065, 0.2], [0.001, 0.015, 0.025]]

    elif candidate in (CAR.SUBARU_OUTBACK, CAR.SUBARU_LEGACY):
      ret.steerActuatorDelay = 0.1

    elif candidate in (CAR.SUBARU_FORESTER_PREGLOBAL, CAR.SUBARU_OUTBACK_PREGLOBAL_2018):
      # Outback 2018-2019 and Forester have reversed driver torque signal
      ret.safetyConfigs[0].safetyParam = SubaruSafetyFlags.PREGLOBAL_REVERSED_DRIVER_TORQUE.value

    elif candidate == CAR.SUBARU_LEGACY_PREGLOBAL:
      ret.steerActuatorDelay = 0.15

    elif candidate == CAR.SUBARU_OUTBACK_PREGLOBAL:
      pass
    else:
      raise ValueError(f"unknown car: {candidate}")

    ret.alphaLongitudinalAvailable = not (ret.flags & (SubaruFlags.GLOBAL_GEN2 | SubaruFlags.PREGLOBAL |
                                                       SubaruFlags.LKAS_ANGLE | SubaruFlags.HYBRID))
    ret.openpilotLongitudinalControl = alpha_long and ret.alphaLongitudinalAvailable

    if ret.flags & SubaruFlags.GLOBAL_GEN2 and ret.openpilotLongitudinalControl:
      ret.flags |= SubaruFlags.DISABLE_EYESIGHT.value

    if ret.openpilotLongitudinalControl:
      ret.safetyConfigs[0].safetyParam |= SubaruSafetyFlags.LONG.value

    return ret

  @staticmethod
  def init(CP, can_recv, can_send, communication_control=None):
    if CP.flags & SubaruFlags.DISABLE_EYESIGHT:
      if communication_control is None:
        communication_control = bytes([uds.SERVICE_TYPE.COMMUNICATION_CONTROL, uds.CONTROL_TYPE.DISABLE_RX_DISABLE_TX, uds.MESSAGE_TYPE.NORMAL])
      disable_ecu(can_recv, can_send, bus=2, addr=GLOBAL_ES_ADDR, com_cont_req=communication_control)

  @staticmethod
  def deinit(CP, can_recv, can_send):
    communication_control = bytes([uds.SERVICE_TYPE.COMMUNICATION_CONTROL, uds.CONTROL_TYPE.ENABLE_RX_ENABLE_TX, uds.MESSAGE_TYPE.NORMAL])
    CarInterface.init(CP, can_recv, can_send, communication_control)
