import copy
from collections import deque
from cereal import car
from opendbc.can.can_define import CANDefine
from common.conversions import Conversions as CV
from selfdrive.car.interfaces import CarStateBase
from opendbc.can.parser import CANParser
from selfdrive.car.subaru.values import DBC, CAR, GLOBAL_GEN1, GLOBAL_GEN2, PREGLOBAL_CARS, SubaruFlags, Buttons


PREV_BUTTON_SAMPLES = 8

def cruise_buttons_conversion(cruise_buttons):
  if "Main" in cruise_buttons:
    if cruise_buttons["Main"]:
      return Buttons.ACC_TOGGLE
    if cruise_buttons["Resume"]:
      return Buttons.RES_INC
    if cruise_buttons["Set"]:
      return Buttons.SET_DEC

  return Buttons.NONE

def cruise_buttons_conversion_gen2_uds(eyesight_uds):
  if "PCI" in eyesight_uds and eyesight_uds["SID"] == 0x62 and eyesight_uds["DID"] == 0x00f3:
    data = eyesight_uds["DATA"]

    if data == 4:
      return Buttons.RES_INC
    elif data == 2:
      return Buttons.SET_DEC
    elif data == 1:
      return Buttons.ACC_TOGGLE
    
    return Buttons.NONE

  return None

class CarState(CarStateBase):
  def __init__(self, CP):
    super().__init__(CP)
    can_define = CANDefine(DBC[CP.carFingerprint]["pt"])
    self.shifter_values = can_define.dv["Transmission"]["Gear"]
    self.cruise_buttons = deque([Buttons.NONE] * PREV_BUTTON_SAMPLES, maxlen=PREV_BUTTON_SAMPLES)

  def update(self, cp, cp_cam, cp_body):
    ret = car.CarState.new_message()

    ret.gas = cp.vl["Throttle"]["Throttle_Pedal"] / 255.
    ret.gasPressed = ret.gas > 1e-5
    if self.car_fingerprint in PREGLOBAL_CARS:
      ret.brakePressed = cp.vl["Brake_Pedal"]["Brake_Pedal"] > 2
    else:
      cp_brakes = cp_body if self.car_fingerprint in GLOBAL_GEN2 else cp
      ret.brakePressed = cp_brakes.vl["Brake_Status"]["Brake"] == 1

    cp_wheels = cp_body if self.car_fingerprint in GLOBAL_GEN2 else cp
    ret.wheelSpeeds = self.get_wheel_speeds(
      cp_wheels.vl["Wheel_Speeds"]["FL"],
      cp_wheels.vl["Wheel_Speeds"]["FR"],
      cp_wheels.vl["Wheel_Speeds"]["RL"],
      cp_wheels.vl["Wheel_Speeds"]["RR"],
    )
    ret.vEgoRaw = (ret.wheelSpeeds.fl + ret.wheelSpeeds.fr + ret.wheelSpeeds.rl + ret.wheelSpeeds.rr) / 4.
    ret.vEgo, ret.aEgo = self.update_speed_kf(ret.vEgoRaw)
    ret.standstill = ret.vEgoRaw == 0

    # continuous blinker signals for assisted lane change
    ret.leftBlinker, ret.rightBlinker = self.update_blinker_from_lamp(50, cp.vl["Dashlights"]["LEFT_BLINKER"],
                                                                          cp.vl["Dashlights"]["RIGHT_BLINKER"])

    if self.CP.enableBsm:
      ret.leftBlindspot = (cp.vl["BSD_RCTA"]["L_ADJACENT"] == 1) or (cp.vl["BSD_RCTA"]["L_APPROACHING"] == 1)
      ret.rightBlindspot = (cp.vl["BSD_RCTA"]["R_ADJACENT"] == 1) or (cp.vl["BSD_RCTA"]["R_APPROACHING"] == 1)

    can_gear = int(cp.vl["Transmission"]["Gear"])
    ret.gearShifter = self.parse_gear_shifter(self.shifter_values.get(can_gear, None))

    ret.steeringAngleDeg = cp.vl["Steering_Torque"]["Steering_Angle"]
    ret.steeringTorque = cp.vl["Steering_Torque"]["Steer_Torque_Sensor"]
    ret.steeringTorqueEps = cp.vl["Steering_Torque"]["Steer_Torque_Output"]

    steer_threshold = 75 if self.CP.carFingerprint in PREGLOBAL_CARS else 80
    ret.steeringPressed = abs(ret.steeringTorque) > steer_threshold

    cp_cruise = cp_body if self.car_fingerprint in GLOBAL_GEN2 else cp

    ret.cruiseState.enabled = cp_cruise.vl["CruiseControl"]["Cruise_Activated"] != 0
    ret.cruiseState.available = cp_cruise.vl["CruiseControl"]["Cruise_On"] != 0
    ret.cruiseState.speed = cp_cruise.vl["Cruise_Status"]["Cruise_Set_Speed"] * CV.KPH_TO_MS

    if (self.car_fingerprint in PREGLOBAL_CARS and cp.vl["Dash_State2"]["UNITS"] == 1) or \
      (self.car_fingerprint not in PREGLOBAL_CARS and cp.vl["Dashlights"]["UNITS"] == 1):
      ret.cruiseState.speed *= CV.MPH_TO_KPH
    
    ret.seatbeltUnlatched = cp.vl["Dashlights"]["SEATBELT_FL"] == 1
    ret.doorOpen = any([cp.vl["BodyInfo"]["DOOR_OPEN_RR"],
                        cp.vl["BodyInfo"]["DOOR_OPEN_RL"],
                        cp.vl["BodyInfo"]["DOOR_OPEN_FR"],
                        cp.vl["BodyInfo"]["DOOR_OPEN_FL"]])
    ret.steerFaultPermanent = cp.vl["Steering_Torque"]["Steer_Error_1"] == 1

    if self.car_fingerprint in PREGLOBAL_CARS:
      self.cruise_button = cp_cam.vl["ES_Distance"]["Cruise_Button"]
      self.ready = not cp_cam.vl["ES_DashStatus"]["Not_Ready_Startup"]
    else:
      ret.steerFaultTemporary = cp.vl["Steering_Torque"]["Steer_Warning"] == 1
      if not (self.CP.flags & SubaruFlags.GEN2_DISABLE_FWD_CAMERA.value):
        ret.cruiseState.nonAdaptive = cp_cam.vl["ES_DashStatus"]["Conventional_Cruise"] == 1
        ret.cruiseState.standstill = cp_cam.vl["ES_DashStatus"]["Cruise_State"] == 3
      else:
        ret.cruiseState.nonAdaptive = False
        ret.cruiseState.standstill = False
      
      if not (self.CP.flags & SubaruFlags.GEN2_DISABLE_FWD_CAMERA.value):
        ret.stockFcw = (cp_cam.vl["ES_LKAS_State"]["LKAS_Alert"] == 1) or \
                      (cp_cam.vl["ES_LKAS_State"]["LKAS_Alert"] == 2)
        ret.stockAeb = (cp_cam.vl["ES_LKAS_State"]["LKAS_Alert"] == 5) or \
                      (cp_cam.vl["ES_LKAS_State"]["LKAS_Alert_Msg"] == 6)

        self.es_lkas_state_msg = copy.copy(cp_cam.vl["ES_LKAS_State"])

        cp_es_brake = cp_body if self.car_fingerprint in GLOBAL_GEN2 else cp_cam
        self.es_brake_msg = copy.copy(cp_es_brake.vl["ES_Brake"])
        cp_es_status = cp_body if self.car_fingerprint in GLOBAL_GEN2 else cp_cam
        self.es_status_msg = copy.copy(cp_es_status.vl["ES_Status"])

      else:
        ret.stockFcw = False
        ret.stockAeb = False
        self.es_lkas_state_msg = None
        self.es_brake_msg = None
        self.es_status_msg = None
      
      self.cruise_control_msg = copy.copy(cp_cruise.vl["CruiseControl"])
      self.brake_status_msg = copy.copy(cp_brakes.vl["Brake_Status"])
    
    cp_es_distance = cp_body if self.car_fingerprint in GLOBAL_GEN2 else cp_cam
    cp_buttons = cp_cam if self.car_fingerprint in GLOBAL_GEN2 else cp_cam

    self.prev_cruise_buttons = self.cruise_buttons[-1]

    if not (self.CP.flags & SubaruFlags.GEN2_DISABLE_FWD_CAMERA.value):
      self.es_distance_msg = copy.copy(cp_es_distance.vl["ES_Distance"])
      self.es_dashstatus_msg = copy.copy(cp_cam.vl["ES_DashStatus"])

      if self.CP.flags & SubaruFlags.SEND_INFOTAINMENT:
        self.es_infotainmentstatus_msg = copy.copy(cp_cam.vl["INFOTAINMENT_STATUS"])
      
      self.cruise_buttons.append(cruise_buttons_conversion(cp_buttons.vl["Cruise_Buttons"]))

    else:
      self.es_distance_msg = None
      self.es_dashstatus_msg = None
      self.es_infotainmentstatus_msg = None

      button_data = cruise_buttons_conversion_gen2_uds(cp_buttons.vl["EYESIGHT_UDS_RESPONSE"])
      if button_data is not None:
        self.cruise_buttons.append(button_data)

    return ret

  @staticmethod
  def get_common_global_signals():
    signals = [
      ("COUNTER", "CruiseControl"),
      ("Signal1", "CruiseControl"),
      ("Cruise_On", "CruiseControl"),
      ("Cruise_Activated", "CruiseControl"),
      ("Signal2", "CruiseControl"),
      ("Main", "Cruise_Buttons"),
      ("Set", "Cruise_Buttons"),
      ("Resume", "Cruise_Buttons"),
      ("FL", "Wheel_Speeds"),
      ("FR", "Wheel_Speeds"),
      ("RL", "Wheel_Speeds"),
      ("RR", "Wheel_Speeds"),
      ("COUNTER", "Brake_Status"),
      ("Signal1", "Brake_Status"),
      ("ES_Brake", "Brake_Status"),
      ("Signal2", "Brake_Status"),
      ("Brake", "Brake_Status"),
      ("Signal3", "Brake_Status"),
      ("Cruise_Set_Speed", "Cruise_Status"),
    ]
    checks = [
      ("Cruise_Buttons", 50),
      ("Cruise_Status", 20),
      ("CruiseControl", 20),
      ("Wheel_Speeds", 50),
      ("Brake_Status", 50),
    ]

    return signals, checks

  @staticmethod
  def get_global_es_signals():
    signals = [
      ("COUNTER", "ES_Distance"),
      ("CHECKSUM", "ES_Distance"),
      ("Signal1", "ES_Distance"),
      ("Cruise_Fault", "ES_Distance"),
      ("Cruise_Throttle", "ES_Distance"),
      ("Signal2", "ES_Distance"),
      ("Car_Follow", "ES_Distance"),
      ("Signal3", "ES_Distance"),
      ("Cruise_Soft_Disable", "ES_Distance"),
      ("Signal7", "ES_Distance"),
      ("Cruise_Brake_Active", "ES_Distance"),
      ("Distance_Swap", "ES_Distance"),
      ("Cruise_EPB", "ES_Distance"),
      ("Signal4", "ES_Distance"),
      ("Close_Distance", "ES_Distance"),
      ("Signal5", "ES_Distance"),
      ("Cruise_Cancel", "ES_Distance"),
      ("Cruise_Set", "ES_Distance"),
      ("Cruise_Resume", "ES_Distance"),
      ("Signal6", "ES_Distance"),

      ("COUNTER", "ES_Status"),
      ("Signal1", "ES_Status"),
      ("Cruise_Fault", "ES_Status"),
      ("Cruise_RPM", "ES_Status"),
      ("Signal2", "ES_Status"),
      ("Cruise_Activated", "ES_Status"),
      ("Brake_Lights", "ES_Status"),
      ("Cruise_Hold", "ES_Status"),
      ("Signal3", "ES_Status"),

      ("COUNTER", "ES_Brake"),
      ("Signal1", "ES_Brake"),
      ("Brake_Pressure", "ES_Brake"),
      ("Signal2", "ES_Brake"),
      ("Cruise_Brake_Lights", "ES_Brake"),
      ("Cruise_Brake_Fault", "ES_Brake"),
      ("Cruise_Brake_Active", "ES_Brake"),
      ("Cruise_Activated", "ES_Brake"),
      ("Signal3", "ES_Brake"),

    ]
    checks = [
      ("ES_Distance", 20),
      ("ES_Status", 20),
      ("ES_Brake", 20),
    ]

    return signals, checks

  @staticmethod
  def get_can_parser(CP):
    signals = [
      # sig_name, sig_address
      ("Steer_Torque_Sensor", "Steering_Torque"),
      ("Steer_Torque_Output", "Steering_Torque"),
      ("Steering_Angle", "Steering_Torque"),
      ("Steer_Error_1", "Steering_Torque"),
      ("Brake_Pedal", "Brake_Pedal"),
      ("Throttle_Pedal", "Throttle"),
      ("Throttle_Cruise", "Throttle"),
      ("LEFT_BLINKER", "Dashlights"),
      ("RIGHT_BLINKER", "Dashlights"),
      ("SEATBELT_FL", "Dashlights"),
      ("DOOR_OPEN_FR", "BodyInfo"),
      ("DOOR_OPEN_FL", "BodyInfo"),
      ("DOOR_OPEN_RR", "BodyInfo"),
      ("DOOR_OPEN_RL", "BodyInfo"),
      ("Gear", "Transmission"),
    ]

    checks = [
      # sig_address, frequency
      ("Throttle", 100),
      ("Dashlights", 10),
      ("Brake_Pedal", 50),
      ("Transmission", 100),
      ("Steering_Torque", 50),
      ("BodyInfo", 1),
    ]

    if CP.enableBsm:
      signals += [
        ("L_ADJACENT", "BSD_RCTA"),
        ("R_ADJACENT", "BSD_RCTA"),
        ("L_APPROACHING", "BSD_RCTA"),
        ("R_APPROACHING", "BSD_RCTA"),
      ]
      checks.append(("BSD_RCTA", 17))

    if CP.carFingerprint not in PREGLOBAL_CARS:
      if CP.carFingerprint not in GLOBAL_GEN2:
        signals += CarState.get_common_global_signals()[0]
        checks += CarState.get_common_global_signals()[1]

      signals += [
        ("Steer_Warning", "Steering_Torque"),
        ("RPM", "Transmission"),
        ("UNITS", "Dashlights"),
      ]

      checks += [
        ("Dashlights", 10),
        ("BodyInfo", 10),
      ]
    else:
      signals += [
        ("FL", "Wheel_Speeds"),
        ("FR", "Wheel_Speeds"),
        ("RL", "Wheel_Speeds"),
        ("RR", "Wheel_Speeds"),
        ("UNITS", "Dash_State2"),
        ("Cruise_On", "CruiseControl"),
        ("Cruise_Activated", "CruiseControl"),
      ]
      checks += [
        ("Wheel_Speeds", 50),
        ("Dash_State2", 1),
      ]

      if CP.carFingerprint == CAR.FORESTER_PREGLOBAL:
        checks += [
          ("Dashlights", 20),
          ("BodyInfo", 1),
          ("CruiseControl", 50),
        ]

      if CP.carFingerprint in (CAR.LEGACY_PREGLOBAL, CAR.OUTBACK_PREGLOBAL, CAR.OUTBACK_PREGLOBAL_2018):
        checks += [
          ("Dashlights", 10),
          ("CruiseControl", 50),
        ]

    return CANParser(DBC[CP.carFingerprint]["pt"], signals, checks, 0)

  @staticmethod
  def get_global_cam_es_signals(CP):
    if CP.carFingerprint in PREGLOBAL_CARS:
      signals = [
        ("Cruise_Set_Speed", "ES_DashStatus"),
        ("Not_Ready_Startup", "ES_DashStatus"),

        ("Cruise_Throttle", "ES_Distance"),
        ("Signal1", "ES_Distance"),
        ("Car_Follow", "ES_Distance"),
        ("Signal2", "ES_Distance"),
        ("Brake_On", "ES_Distance"),
        ("Distance_Swap", "ES_Distance"),
        ("Standstill", "ES_Distance"),
        ("Signal3", "ES_Distance"),
        ("Close_Distance", "ES_Distance"),
        ("Signal4", "ES_Distance"),
        ("Standstill_2", "ES_Distance"),
        ("Cruise_Fault", "ES_Distance"),
        ("Signal5", "ES_Distance"),
        ("COUNTER", "ES_Distance"),
        ("Signal6", "ES_Distance"),
        ("Cruise_Button", "ES_Distance"),
        ("Signal7", "ES_Distance"),
      ]

      checks = [
        ("ES_DashStatus", 20),
        ("ES_Distance", 20),
      ]
    else:
      signals = [
        ("COUNTER", "ES_DashStatus"),
        ("CHECKSUM", "ES_DashStatus"),
        ("PCB_Off", "ES_DashStatus"),
        ("LDW_Off", "ES_DashStatus"),
        ("Signal1", "ES_DashStatus"),
        ("Cruise_State_Msg", "ES_DashStatus"),
        ("LKAS_State_Msg", "ES_DashStatus"),
        ("Signal2", "ES_DashStatus"),
        ("Cruise_Soft_Disable", "ES_DashStatus"),
        ("Cruise_Status_Msg", "ES_DashStatus"),
        ("Signal3", "ES_DashStatus"),
        ("Cruise_Distance", "ES_DashStatus"),
        ("Signal4", "ES_DashStatus"),
        ("Conventional_Cruise", "ES_DashStatus"),
        ("Signal5", "ES_DashStatus"),
        ("Cruise_Disengaged", "ES_DashStatus"),
        ("Cruise_Activated", "ES_DashStatus"),
        ("Signal6", "ES_DashStatus"),
        ("Cruise_Set_Speed", "ES_DashStatus"),
        ("Cruise_Fault", "ES_DashStatus"),
        ("Cruise_On", "ES_DashStatus"),
        ("Display_Own_Car", "ES_DashStatus"),
        ("Brake_Lights", "ES_DashStatus"),
        ("Car_Follow", "ES_DashStatus"),
        ("Signal7", "ES_DashStatus"),
        ("Far_Distance", "ES_DashStatus"),
        ("Cruise_State", "ES_DashStatus"),

        ("COUNTER", "ES_LKAS_State"),
        ("CHECKSUM", "ES_LKAS_State"),
        ("LKAS_Alert_Msg", "ES_LKAS_State"),
        ("Signal1", "ES_LKAS_State"),
        ("LKAS_ACTIVE", "ES_LKAS_State"),
        ("LKAS_Dash_State", "ES_LKAS_State"),
        ("Signal2", "ES_LKAS_State"),
        ("Backward_Speed_Limit_Menu", "ES_LKAS_State"),
        ("LKAS_Left_Line_Enable", "ES_LKAS_State"),
        ("LKAS_Left_Line_Light_Blink", "ES_LKAS_State"),
        ("LKAS_Right_Line_Enable", "ES_LKAS_State"),
        ("LKAS_Right_Line_Light_Blink", "ES_LKAS_State"),
        ("LKAS_Left_Line_Visible", "ES_LKAS_State"),
        ("LKAS_Right_Line_Visible", "ES_LKAS_State"),
        ("LKAS_Alert", "ES_LKAS_State"),
        ("Signal3", "ES_LKAS_State"),
      ]
    
      checks = [
        ("ES_DashStatus", 10),
        ("ES_LKAS_State", 10),
      ]

    return signals, checks

  @staticmethod
  def get_cam_can_parser(CP):
    if not (CP.flags & SubaruFlags.GEN2_DISABLE_FWD_CAMERA.value):
      signals, checks = CarState.get_global_cam_es_signals(CP)
    else:
      signals, checks = [], []
    
    if CP.carFingerprint in GLOBAL_GEN1:
      signals += CarState.get_global_es_signals()[0]
      checks += CarState.get_global_es_signals()[1]

    if CP.flags & SubaruFlags.SEND_INFOTAINMENT and not (CP.flags & SubaruFlags.GEN2_DISABLE_FWD_CAMERA.value):
      signals += [
        ("LKAS_State_Infotainment", "INFOTAINMENT_STATUS"),
        ("LKAS_Blue_Lines", "INFOTAINMENT_STATUS"),
        ("Signal1", "INFOTAINMENT_STATUS"),
        ("Signal2", "INFOTAINMENT_STATUS"),
      ]
      checks.append(("INFOTAINMENT_STATUS", 10))
  
    if (CP.flags & SubaruFlags.GEN2_DISABLE_FWD_CAMERA.value):
      signals += [
        ("PCI", "EYESIGHT_UDS_RESPONSE"),
        ("SID", "EYESIGHT_UDS_RESPONSE"),
        ("DID", "EYESIGHT_UDS_RESPONSE"),
        ("DATA", "EYESIGHT_UDS_RESPONSE")
      ]

      checks += [
        ("EYESIGHT_UDS_RESPONSE", 5) # just because eyesight is sending UDS messages doesn't mean its the correct messages, need dbc multiplexing
      ]
    
    return CANParser(DBC[CP.carFingerprint]["pt"], signals, checks, 2)

  @staticmethod
  def get_body_can_parser(CP):
    if CP.carFingerprint in GLOBAL_GEN2:
      signals, checks = CarState.get_common_global_signals()

      if CP.carFingerprint in GLOBAL_GEN2 and not (CP.flags & SubaruFlags.GEN2_DISABLE_FWD_CAMERA.value):
        signals += CarState.get_global_es_signals()[0]
        checks += CarState.get_global_es_signals()[1]
      
      return CANParser(DBC[CP.carFingerprint]["pt"], signals, checks, 1)

    return None
