import copy
from cereal import car
from opendbc.can.can_define import CANDefine
from selfdrive.config import Conversions as CV
from selfdrive.car.interfaces import CarStateBase
from opendbc.can.parser import CANParser
from selfdrive.car.subaru.values import DBC, STEER_THRESHOLD, CAR, PREGLOBAL_CARS


class CarState(CarStateBase):
  def __init__(self, CP):
    super().__init__(CP)
    can_define = CANDefine(DBC[CP.carFingerprint]["pt"])
    self.shifter_values = can_define.dv["Transmission"]["Gear"]

  def update(self, cp, cp_cam):
    ret = car.CarState.new_message()

    ret.gas = cp.vl["Throttle"]["Throttle_Pedal"] / 255.
    ret.gasPressed = ret.gas > 1e-5
    if self.car_fingerprint in PREGLOBAL_CARS:
      ret.brakePressed = cp.vl["Brake_Pedal"]["Brake_Pedal"] > 2
    else:
      ret.brakePressed = cp.vl["Brake_Status"]["Brake"] == 1

    ret.wheelSpeeds = self.get_wheel_speeds(
      cp.vl["Wheel_Speeds"]["FL"],
      cp.vl["Wheel_Speeds"]["FR"],
      cp.vl["Wheel_Speeds"]["RL"],
      cp.vl["Wheel_Speeds"]["RR"],
    )
    ret.vEgoRaw = (ret.wheelSpeeds.fl + ret.wheelSpeeds.fr + ret.wheelSpeeds.rl + ret.wheelSpeeds.rr) / 4.
    # Kalman filter, even though Subaru raw wheel speed is heaviliy filtered by default
    ret.vEgo, ret.aEgo = self.update_speed_kf(ret.vEgoRaw)
    ret.standstill = ret.vEgoRaw < 0.01

    # continuous blinker signals for assisted lane change
    ret.leftBlinker, ret.rightBlinker = self.update_blinker_from_lamp(
      50, cp.vl["Dashlights"]["LEFT_BLINKER"], cp.vl["Dashlights"]["RIGHT_BLINKER"])

    if self.CP.enableBsm:
      ret.leftBlindspot = (cp.vl["BSD_RCTA"]["L_ADJACENT"] == 1) or (cp.vl["BSD_RCTA"]["L_APPROACHING"] == 1)
      ret.rightBlindspot = (cp.vl["BSD_RCTA"]["R_ADJACENT"] == 1) or (cp.vl["BSD_RCTA"]["R_APPROACHING"] == 1)

    can_gear = int(cp.vl["Transmission"]["Gear"])
    ret.gearShifter = self.parse_gear_shifter(self.shifter_values.get(can_gear, None))

    ret.steeringAngleDeg = cp.vl["Steering_Torque"]["Steering_Angle"]
    ret.steeringTorque = cp.vl["Steering_Torque"]["Steer_Torque_Sensor"]
    ret.steeringPressed = abs(ret.steeringTorque) > STEER_THRESHOLD[self.car_fingerprint]

    ret.cruiseState.enabled = cp.vl["CruiseControl"]["Cruise_Activated"] != 0
    ret.cruiseState.available = cp.vl["CruiseControl"]["Cruise_On"] != 0
    ret.cruiseState.speed = cp_cam.vl["ES_DashStatus"]["Cruise_Set_Speed"] * CV.KPH_TO_MS

    if (self.car_fingerprint in PREGLOBAL_CARS and cp.vl["Dash_State2"]["UNITS"] == 1) or \
       (self.car_fingerprint not in PREGLOBAL_CARS and cp.vl["Dashlights"]["UNITS"] == 1):
      ret.cruiseState.speed *= CV.MPH_TO_KPH

    ret.seatbeltUnlatched = cp.vl["Dashlights"]["SEATBELT_FL"] == 1
    ret.doorOpen = any([cp.vl["BodyInfo"]["DOOR_OPEN_RR"],
                        cp.vl["BodyInfo"]["DOOR_OPEN_RL"],
                        cp.vl["BodyInfo"]["DOOR_OPEN_FR"],
                        cp.vl["BodyInfo"]["DOOR_OPEN_FL"]])
    ret.steerError = cp.vl["Steering_Torque"]["Steer_Error_1"] == 1

    if self.car_fingerprint in PREGLOBAL_CARS:
      self.cruise_button = cp_cam.vl["ES_Distance"]["Cruise_Button"]
      self.ready = not cp_cam.vl["ES_DashStatus"]["Not_Ready_Startup"]
    else:
      ret.steerWarning = cp.vl["Steering_Torque"]["Steer_Warning"] == 1
      ret.cruiseState.nonAdaptive = cp_cam.vl["ES_DashStatus"]["Conventional_Cruise"] == 1
      self.es_lkas_msg = copy.copy(cp_cam.vl["ES_LKAS_State"])
    self.es_distance_msg = copy.copy(cp_cam.vl["ES_Distance"])

    return ret

  @staticmethod
  def get_can_parser(CP):
    signals = [
      # sig_name, sig_address
      ("Steer_Torque_Sensor", "Steering_Torque"),
      ("Steering_Angle", "Steering_Torque"),
      ("Steer_Error_1", "Steering_Torque"),
      ("Cruise_On", "CruiseControl"),
      ("Cruise_Activated", "CruiseControl"),
      ("Brake_Pedal", "Brake_Pedal"),
      ("Throttle_Pedal", "Throttle"),
      ("LEFT_BLINKER", "Dashlights"),
      ("RIGHT_BLINKER", "Dashlights"),
      ("SEATBELT_FL", "Dashlights"),
      ("FL", "Wheel_Speeds"),
      ("FR", "Wheel_Speeds"),
      ("RL", "Wheel_Speeds"),
      ("RR", "Wheel_Speeds"),
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
      ("Wheel_Speeds", 50),
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
      signals += [
        ("Steer_Warning", "Steering_Torque"),
        ("Brake", "Brake_Status"),
        ("UNITS", "Dashlights"),
      ]

      checks += [
        ("Dashlights", 10),
        ("BodyInfo", 10),
        ("Brake_Status", 50),
        ("CruiseControl", 20),
      ]
    else:
      signals.append(("UNITS", "Dash_State2"))

      checks.append(("Dash_State2", 1))

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
        ("Counter", "ES_Distance"),
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
        ("Cruise_Set_Speed", "ES_DashStatus"),
        ("Conventional_Cruise", "ES_DashStatus"),

        ("Counter", "ES_Distance"),
        ("Signal1", "ES_Distance"),
        ("Cruise_Fault", "ES_Distance"),
        ("Cruise_Throttle", "ES_Distance"),
        ("Signal2", "ES_Distance"),
        ("Car_Follow", "ES_Distance"),
        ("Signal3", "ES_Distance"),
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

        ("Counter", "ES_LKAS_State"),
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
        ("ES_Distance", 20),
        ("ES_LKAS_State", 10),
      ]

    return CANParser(DBC[CP.carFingerprint]["pt"], signals, checks, 2)
