import copy
from cereal import car
from opendbc.can.can_define import CANDefine
from common.conversions import Conversions as CV
from selfdrive.car.interfaces import CarStateBase
from opendbc.can.parser import CANParser
from selfdrive.car.subaru.values import DBC, CAR, GLOBAL_GEN2, PREGLOBAL_CARS


class CarState(CarStateBase):
  def __init__(self, CP):
    super().__init__(CP)
    can_define = CANDefine(DBC[CP.carFingerprint]["pt"])
    self.shifter_values = can_define.dv["Transmission"]["Gear"]

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
    # Kalman filter, even though Subaru raw wheel speed is heaviliy filtered by default
    ret.vEgo, ret.aEgo = self.update_speed_kf(ret.vEgoRaw)
    ret.standstill = ret.vEgoRaw < 0.01

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
    ret.cruiseState.speed = cp_cam.vl["ES_DashStatus"]["Cruise_Set_Speed"] * CV.KPH_TO_MS

    if self.car_fingerprint not in PREGLOBAL_CARS:
      ret.cruiseState.standstill = cp_cam.vl["ES_DashStatus"]["Cruise_State"] == 3

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
      ret.cruiseState.nonAdaptive = cp_cam.vl["ES_DashStatus"]["Conventional_Cruise"] == 1
      self.es_lkas_msg = copy.copy(cp_cam.vl["ES_LKAS_State"])

    cp_es_distance = cp_body if self.car_fingerprint in GLOBAL_GEN2 else cp_cam
    self.es_distance_msg = copy.copy(cp_es_distance.vl["ES_Distance"])
    self.es_dashstatus_msg = copy.copy(cp_cam.vl["ES_DashStatus"])

    return ret

  @staticmethod
  def get_common_global_signals():
    signals = [
      ("Cruise_On", "CruiseControl"),
      ("Cruise_Activated", "CruiseControl"),
      ("FL", "Wheel_Speeds"),
      ("FR", "Wheel_Speeds"),
      ("RL", "Wheel_Speeds"),
      ("RR", "Wheel_Speeds"),
      ("Brake", "Brake_Status"),
    ]
    checks = [
      ("CruiseControl", 20),
      ("Wheel_Speeds", 50),
      ("Brake_Status", 50),
    ]

    return signals, checks

  @staticmethod
  def get_global_es_distance_signals():
    signals = [
      ("COUNTER", "ES_Distance"),
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
    ]
    checks = [
      ("ES_Distance", 20),
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
  def get_cam_can_parser(CP):
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
        ("Counter", "ES_DashStatus"),
        ("PCB_Off", "ES_DashStatus"),
        ("LDW_Off", "ES_DashStatus"),
        ("Signal1", "ES_DashStatus"),
        ("Cruise_State_Msg", "ES_DashStatus"),
        ("LKAS_State_Msg", "ES_DashStatus"),
        ("Signal2", "ES_DashStatus"),
        ("Cruise_Soft_Disable", "ES_DashStatus"),
        ("EyeSight_Status_Msg", "ES_DashStatus"),
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

      if CP.carFingerprint not in GLOBAL_GEN2:
        signals += CarState.get_global_es_distance_signals()[0]
        checks += CarState.get_global_es_distance_signals()[1]

    return CANParser(DBC[CP.carFingerprint]["pt"], signals, checks, 2)

  @staticmethod
  def get_body_can_parser(CP):
    if CP.carFingerprint in GLOBAL_GEN2:
      signals, checks = CarState.get_common_global_signals()
      signals += CarState.get_global_es_distance_signals()[0]
      checks += CarState.get_global_es_distance_signals()[1]
      return CANParser(DBC[CP.carFingerprint]["pt"], signals, checks, 1)

    return None