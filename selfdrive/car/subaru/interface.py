#!/usr/bin/env python3
from cereal import car
from opendbc.can.tests.test_packer_parser import can_list_to_can_capnp
from panda import Panda
from panda.python.uds import CONTROL_TYPE, MESSAGE_TYPE
from selfdrive.car import STD_CARGO_KG, get_safety_config, create_button_event
from selfdrive.car.interfaces import CarInterfaceBase
from selfdrive.car.isotp_parallel_query import IsoTpParallelQuery
from selfdrive.car.subaru.values import CAR, GEN2_ES_BUTTONS_DID, GEN2_ES_BUTTONS_MEMORY_ADDRESS, GEN2_ES_SECRET_KEY, GLOBAL_GEN2, PREGLOBAL_CARS, SubaruFlags, Buttons
from selfdrive.car.disable_ecu import EXT_DIAG_REQUEST, EXT_DIAG_RESPONSE, disable_ecu
from Crypto.Cipher import AES
from system.swaglog import cloudlog

ButtonType = car.CarState.ButtonEvent.Type

BUTTONS_DICT = {Buttons.RES_INC: ButtonType.accelCruise, Buttons.SET_DEC: ButtonType.decelCruise,
                Buttons.GAP_DIST_INC: ButtonType.gapAdjustCruise, Buttons.GAP_DIST_DEC: ButtonType.gapAdjustCruise,
                Buttons.LKAS_TOGGLE: ButtonType.cancel, Buttons.ACC_TOGGLE: ButtonType.cancel}


class CarInterface(CarInterfaceBase):

  @staticmethod
  def _get_params(ret, candidate, fingerprint, car_fw, experimental_long, docs):
    ret.carName = "subaru"
    ret.radarUnavailable = True
    ret.dashcamOnly = candidate in PREGLOBAL_CARS
    ret.autoResumeSng = False

    # Detect infotainment message sent from the camera
    if candidate not in PREGLOBAL_CARS and 0x323 in fingerprint[2]:
      ret.flags |= SubaruFlags.SEND_INFOTAINMENT.value
    
    if candidate in GLOBAL_GEN2 and experimental_long:
      ret.flags |= SubaruFlags.GEN2_DISABLE_FWD_CAMERA.value

    if candidate in PREGLOBAL_CARS:
      ret.enableBsm = 0x25c in fingerprint[0]
      ret.safetyConfigs = [get_safety_config(car.CarParams.SafetyModel.subaruPreglobal)]
    else:
      ret.enableBsm = 0x228 in fingerprint[0]
      ret.safetyConfigs = [get_safety_config(car.CarParams.SafetyModel.subaru)]
      if candidate in GLOBAL_GEN2:
        ret.safetyConfigs[0].safetyParam |= Panda.FLAG_SUBARU_GEN2

    ret.steerLimitTimer = 0.4
    ret.steerActuatorDelay = 0.1
    CarInterfaceBase.configure_torque_tune(candidate, ret.lateralTuning)

    if candidate == CAR.ASCENT:
      ret.mass = 2031. + STD_CARGO_KG
      ret.wheelbase = 2.89
      ret.centerToFront = ret.wheelbase * 0.5
      ret.steerRatio = 13.5
      ret.steerActuatorDelay = 0.3   # end-to-end angle controller
      ret.lateralTuning.init('pid')
      ret.lateralTuning.pid.kf = 0.00003
      ret.lateralTuning.pid.kiBP, ret.lateralTuning.pid.kpBP = [[0., 20.], [0., 20.]]
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.0025, 0.1], [0.00025, 0.01]]

    elif candidate == CAR.IMPREZA:
      ret.mass = 1568. + STD_CARGO_KG
      ret.wheelbase = 2.67
      ret.centerToFront = ret.wheelbase * 0.5
      ret.steerRatio = 15
      ret.steerActuatorDelay = 0.4   # end-to-end angle controller
      ret.lateralTuning.init('pid')
      ret.lateralTuning.pid.kf = 0.00005
      ret.lateralTuning.pid.kiBP, ret.lateralTuning.pid.kpBP = [[0., 20.], [0., 20.]]
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.2, 0.3], [0.02, 0.03]]

    elif candidate == CAR.IMPREZA_2020:
      ret.mass = 1480. + STD_CARGO_KG
      ret.wheelbase = 2.67
      ret.centerToFront = ret.wheelbase * 0.5
      ret.steerRatio = 17           # learned, 14 stock
      ret.lateralTuning.init('pid')
      ret.lateralTuning.pid.kf = 0.00005
      ret.lateralTuning.pid.kiBP, ret.lateralTuning.pid.kpBP = [[0., 14., 23.], [0., 14., 23.]]
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.045, 0.042, 0.20], [0.04, 0.035, 0.045]]

    elif candidate == CAR.FORESTER:
      ret.mass = 1568. + STD_CARGO_KG
      ret.wheelbase = 2.67
      ret.centerToFront = ret.wheelbase * 0.5
      ret.steerRatio = 17           # learned, 14 stock
      ret.lateralTuning.init('pid')
      ret.lateralTuning.pid.kf = 0.000038
      ret.lateralTuning.pid.kiBP, ret.lateralTuning.pid.kpBP = [[0., 14., 23.], [0., 14., 23.]]
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.01, 0.065, 0.2], [0.001, 0.015, 0.025]]

    elif candidate in (CAR.OUTBACK, CAR.LEGACY):
      ret.mass = 1568. + STD_CARGO_KG
      ret.wheelbase = 2.67
      ret.centerToFront = ret.wheelbase * 0.5
      ret.steerRatio = 17
      ret.steerActuatorDelay = 0.1
      CarInterfaceBase.configure_torque_tune(candidate, ret.lateralTuning)

    elif candidate in (CAR.FORESTER_PREGLOBAL, CAR.OUTBACK_PREGLOBAL_2018):
      ret.safetyConfigs[0].safetyParam = 1  # Outback 2018-2019 and Forester have reversed driver torque signal
      ret.mass = 1568 + STD_CARGO_KG
      ret.wheelbase = 2.67
      ret.centerToFront = ret.wheelbase * 0.5
      ret.steerRatio = 20           # learned, 14 stock

    elif candidate == CAR.LEGACY_PREGLOBAL:
      ret.mass = 1568 + STD_CARGO_KG
      ret.wheelbase = 2.67
      ret.centerToFront = ret.wheelbase * 0.5
      ret.steerRatio = 12.5   # 14.5 stock
      ret.steerActuatorDelay = 0.15

    elif candidate == CAR.OUTBACK_PREGLOBAL:
      ret.mass = 1568 + STD_CARGO_KG
      ret.wheelbase = 2.67
      ret.centerToFront = ret.wheelbase * 0.5
      ret.steerRatio = 20           # learned, 14 stock

    else:
      raise ValueError(f"unknown car: {candidate}")

    # longitudinal
    ret.experimentalLongitudinalAvailable = candidate not in PREGLOBAL_CARS

    if experimental_long and ret.experimentalLongitudinalAvailable:
      ret.longitudinalTuning.kpBP = [0., 5., 35.]
      ret.longitudinalTuning.kpV = [0.8, 1.0, 1.5]
      ret.longitudinalTuning.kiBP = [0., 35.]
      ret.longitudinalTuning.kiV = [0.54, 0.36]

      ret.stoppingControl = True
      ret.openpilotLongitudinalControl = experimental_long
      if ret.openpilotLongitudinalControl:
        ret.safetyConfigs[0].safetyParam |= Panda.FLAG_SUBARU_LONG
    
    ret.pcmCruise = True

    return ret

  # returns a car.CarState
  def _update(self, c):

    ret = self.CS.update(self.cp, self.cp_cam, self.cp_body)

    ret.buttonEvents = []

    if self.CS.cruise_buttons[-1] != self.CS.prev_cruise_buttons:
      buttonEvents = [create_button_event(self.CS.cruise_buttons[-1], self.CS.prev_cruise_buttons, BUTTONS_DICT)]
      ret.buttonEvents = buttonEvents

    self.CS.buttonEvents = ret.buttonEvents

    ret.events = self.create_common_events(ret, pcm_enable=self.CS.CP.pcmCruise).to_msg()

    return ret

  def apply(self, c, now_nanos):
    return self.CC.update(c, self.CS, now_nanos)

  @staticmethod
  def gen2_security_access(seed):
    cipher = AES.new(GEN2_ES_SECRET_KEY, AES.MODE_ECB)
    key = cipher.encrypt(seed)
    return key

  @staticmethod
  def init(CP, logcan, sendcan):
    if CP.flags & SubaruFlags.GEN2_DISABLE_FWD_CAMERA.value:
      # Disable FWD Camera
      bus = 2
      addr = 0x787

      EXT_DIAG_REQUEST = b'\x10\x03'
      EXT_DIAG_RESPONSE = b'\x50\x03'
      query = IsoTpParallelQuery(sendcan, logcan, bus, [addr], [EXT_DIAG_REQUEST], [EXT_DIAG_RESPONSE], debug=False)
      resp = query.get_data(2)

      if not len(resp):
        cloudlog.warning("failed to enter diagnostic session...")
        return

      sub_function = CONTROL_TYPE.DISABLE_RX_DISABLE_TX
      communication_type = MESSAGE_TYPE.NORMAL
      COMM_CONT_REQUEST = b'\x28' + int.to_bytes(sub_function, 1, byteorder="big") + int.to_bytes(communication_type, 1, byteorder="big")
      COM_CONT_RESPONSE = b'\x68'
      query = IsoTpParallelQuery(sendcan, logcan, bus, [addr], [COMM_CONT_REQUEST], [COM_CONT_RESPONSE], debug=False)
      resp = query.get_data(2)

      if not len(resp):
        cloudlog.warning("failed to disable ecu...")
        return

      # Unlock ECU
      ES_SEED_REQUEST = b'\x27\x03'
      ES_SEED_RESPONSE = b'\x67\x03'
      query = IsoTpParallelQuery(sendcan, logcan, bus, [addr], [ES_SEED_REQUEST], [ES_SEED_RESPONSE], debug=False)
      resp = query.get_data(2)

      if not len(resp):
        cloudlog.warning("failed to request seed...")
        return

      seed = resp[(addr, None)]
      key = CarInterface.gen2_security_access(seed)
      ES_KEY_REQUEST = b'\x27\x04' + key
      ES_KEY_RESPONSE = b'\x67\x04'
      query = IsoTpParallelQuery(sendcan, logcan, bus, [addr], [ES_KEY_REQUEST], [ES_KEY_RESPONSE], debug=False)
      resp = query.get_data(2)

      if not len(resp):
        cloudlog.warning("failed to unlock ecu with key...")
        return

      # Setup Button DID
      BUTTON_DID_REQUEST = b'\x2c\x01' + GEN2_ES_BUTTONS_DID + GEN2_ES_BUTTONS_MEMORY_ADDRESS
      BUTTON_DID_RESPONSE = b'\x6c\x01' + GEN2_ES_BUTTONS_DID

      query = IsoTpParallelQuery(sendcan, logcan, bus, [addr], [BUTTON_DID_REQUEST], [BUTTON_DID_RESPONSE], debug=False)
      resp = query.get_data(2)

      if not len(resp):
        cloudlog.warning("failed to setup button DID...")
        return