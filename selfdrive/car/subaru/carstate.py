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
    can_define = CANDefine(DBC[CP.carFingerprint]['pt'])
    self.shifter_values = can_define.dv["Transmission"]['Gear']

  def update(self, cp, cp_cam):
    ret = car.CarState.new_message()

    ret.gas = cp.vl["Throttle"]['Throttle_Pedal'] / 255.
    ret.gasPressed = ret.gas > 1e-5
    if self.car_fingerprint in PREGLOBAL_CARS:
      ret.brakePressed = cp.vl["Brake_Pedal"]['Brake_Pedal'] > 2
    else:
      ret.brakePressed = cp.vl["Brake_Pedal"]['Brake_Pedal'] > 1e-5
    ret.brakeLights = ret.brakePressed

    ret.wheelSpeeds.fl = cp.vl["Wheel_Speeds"]['FL'] * CV.KPH_TO_MS
    ret.wheelSpeeds.fr = cp.vl["Wheel_Speeds"]['FR'] * CV.KPH_TO_MS
    ret.wheelSpeeds.rl = cp.vl["Wheel_Speeds"]['RL'] * CV.KPH_TO_MS
    ret.wheelSpeeds.rr = cp.vl["Wheel_Speeds"]['RR'] * CV.KPH_TO_MS
    ret.vEgoRaw = (ret.wheelSpeeds.fl + ret.wheelSpeeds.fr + ret.wheelSpeeds.rl + ret.wheelSpeeds.rr) / 4.
    # Kalman filter, even though Subaru raw wheel speed is heaviliy filtered by default
    ret.vEgo, ret.aEgo = self.update_speed_kf(ret.vEgoRaw)
    ret.standstill = ret.vEgoRaw < 0.01

    # continuous blinker signals for assisted lane change
    ret.leftBlinker, ret.rightBlinker = self.update_blinker(50, cp.vl["Dashlights"]['LEFT_BLINKER'],
                                                            cp.vl["Dashlights"]['RIGHT_BLINKER'])

    ret.leftBlindspot = (cp.vl["BSD_RCTA"]['L_ADJACENT'] == 1) or (cp.vl["BSD_RCTA"]['L_APPROACHING'] == 1)
    ret.rightBlindspot = (cp.vl["BSD_RCTA"]['R_ADJACENT'] == 1) or (cp.vl["BSD_RCTA"]['R_APPROACHING'] == 1)

    can_gear = int(cp.vl["Transmission"]['Gear'])
    ret.gearShifter = self.parse_gear_shifter(self.shifter_values.get(can_gear, None))

    ret.steeringAngle = cp.vl["Steering_Torque"]['Steering_Angle']
    ret.steeringTorque = cp.vl["Steering_Torque"]['Steer_Torque_Sensor']
    ret.steeringPressed = abs(ret.steeringTorque) > STEER_THRESHOLD[self.car_fingerprint]

    ret.cruiseState.enabled = cp.vl["CruiseControl"]['Cruise_Activated'] != 0
    ret.cruiseState.available = cp.vl["CruiseControl"]['Cruise_On'] != 0
    ret.cruiseState.speed = cp_cam.vl["ES_DashStatus"]['Cruise_Set_Speed'] * CV.KPH_TO_MS

    # UDM Forester, Legacy: mph = 0
    if self.car_fingerprint in [CAR.FORESTER_PREGLOBAL, CAR.LEGACY_PREGLOBAL] and cp.vl["Dash_State"]['Units'] == 0:
      ret.cruiseState.speed *= CV.MPH_TO_KPH
    # EDM Global: mph = 1, 2; All Outback: mph = 1, UDM Forester: mph = 7
    elif self.car_fingerprint not in [CAR.FORESTER_PREGLOBAL, CAR.LEGACY_PREGLOBAL] and cp.vl["Dash_State"]['Units'] in [1, 2, 7]:
      ret.cruiseState.speed *= CV.MPH_TO_KPH

    ret.seatbeltUnlatched = cp.vl["Dashlights"]['SEATBELT_FL'] == 1
    ret.doorOpen = any([cp.vl["BodyInfo"]['DOOR_OPEN_RR'],
                        cp.vl["BodyInfo"]['DOOR_OPEN_RL'],
                        cp.vl["BodyInfo"]['DOOR_OPEN_FR'],
                        cp.vl["BodyInfo"]['DOOR_OPEN_FL']])

    if self.car_fingerprint in PREGLOBAL_CARS:
      ret.steerError = cp.vl["Steering_Torque"]["LKA_Lockout"] == 1
      self.button = cp_cam.vl["ES_CruiseThrottle"]["Button"]
      self.ready = not cp_cam.vl["ES_DashStatus"]["Not_Ready_Startup"]
      self.es_accel_msg = copy.copy(cp_cam.vl["ES_CruiseThrottle"])
    else:
      ret.steerError = cp.vl["Steering_Torque"]['Steer_Error_1'] == 1
      ret.steerWarning = cp.vl["Steering_Torque"]['Steer_Warning'] == 1
      ret.cruiseState.nonAdaptive = cp_cam.vl["ES_DashStatus"]['Conventional_Cruise'] == 1
      self.es_distance_msg = copy.copy(cp_cam.vl["ES_Distance"])
      self.es_lkas_msg = copy.copy(cp_cam.vl["ES_LKAS_State"])

    return ret

  @staticmethod
  def get_can_parser(CP):
    # this function generates lists for signal, messages and initial values
    signals = [
      # sig_name, sig_address, default
      ("Steer_Torque_Sensor", "Steering_Torque", 0),
      ("Steering_Angle", "Steering_Torque", 0),
      ("Cruise_On", "CruiseControl", 0),
      ("Cruise_Activated", "CruiseControl", 0),
      ("Brake_Pedal", "Brake_Pedal", 0),
      ("Throttle_Pedal", "Throttle", 0),
      ("LEFT_BLINKER", "Dashlights", 0),
      ("RIGHT_BLINKER", "Dashlights", 0),
      ("SEATBELT_FL", "Dashlights", 0),
      ("FL", "Wheel_Speeds", 0),
      ("FR", "Wheel_Speeds", 0),
      ("RL", "Wheel_Speeds", 0),
      ("RR", "Wheel_Speeds", 0),
      ("DOOR_OPEN_FR", "BodyInfo", 1),
      ("DOOR_OPEN_FL", "BodyInfo", 1),
      ("DOOR_OPEN_RR", "BodyInfo", 1),
      ("DOOR_OPEN_RL", "BodyInfo", 1),
      ("Units", "Dash_State", 1),
      ("Gear", "Transmission", 0),
      ("L_ADJACENT", "BSD_RCTA", 0),
      ("R_ADJACENT", "BSD_RCTA", 0),
      ("L_APPROACHING", "BSD_RCTA", 0),
      ("R_APPROACHING", "BSD_RCTA", 0),
    ]

    checks = [
      # sig_address, frequency
      ("Throttle", 100),
      ("Dashlights", 10),
      ("Brake_Pedal", 50),
      ("Wheel_Speeds", 50),
      ("Transmission", 100),
      ("Steering_Torque", 50),
    ]

    if CP.carFingerprint in PREGLOBAL_CARS:
      signals += [
        ("LKA_Lockout", "Steering_Torque", 0),
      ]
    else:
      signals += [
        ("Steer_Error_1", "Steering_Torque", 0),
        ("Steer_Warning", "Steering_Torque", 0),
      ]

      checks += [
        ("Dashlights", 10),
        ("BodyInfo", 10),
        ("CruiseControl", 20),
      ]

    if CP.carFingerprint == CAR.FORESTER_PREGLOBAL:
      checks += [
        ("Dashlights", 20),
        ("BodyInfo", 1),
        ("CruiseControl", 50),
      ]

    if CP.carFingerprint in [CAR.LEGACY_PREGLOBAL, CAR.OUTBACK_PREGLOBAL, CAR.OUTBACK_PREGLOBAL_2018]:
      checks += [
        ("Dashlights", 10),
        ("CruiseControl", 50),
      ]

    return CANParser(DBC[CP.carFingerprint]['pt'], signals, checks, 0)

  @staticmethod
  def get_cam_can_parser(CP):
    if CP.carFingerprint in PREGLOBAL_CARS:
      signals = [
        ("Cruise_Set_Speed", "ES_DashStatus", 0),
        ("Not_Ready_Startup", "ES_DashStatus", 0),

        ("Throttle_Cruise", "ES_CruiseThrottle", 0),
        ("Signal1", "ES_CruiseThrottle", 0),
        ("Cruise_Activated", "ES_CruiseThrottle", 0),
        ("Signal2", "ES_CruiseThrottle", 0),
        ("Brake_On", "ES_CruiseThrottle", 0),
        ("DistanceSwap", "ES_CruiseThrottle", 0),
        ("Standstill", "ES_CruiseThrottle", 0),
        ("Signal3", "ES_CruiseThrottle", 0),
        ("CloseDistance", "ES_CruiseThrottle", 0),
        ("Signal4", "ES_CruiseThrottle", 0),
        ("Standstill_2", "ES_CruiseThrottle", 0),
        ("ES_Error", "ES_CruiseThrottle", 0),
        ("Signal5", "ES_CruiseThrottle", 0),
        ("Counter", "ES_CruiseThrottle", 0),
        ("Signal6", "ES_CruiseThrottle", 0),
        ("Button", "ES_CruiseThrottle", 0),
        ("Signal7", "ES_CruiseThrottle", 0),
      ]

      checks = [
        ("ES_DashStatus", 20),
        ("ES_CruiseThrottle", 20),
      ]
    else:
      signals = [
        ("Cruise_Set_Speed", "ES_DashStatus", 0),
        ("Conventional_Cruise", "ES_DashStatus", 0),

        ("Counter", "ES_Distance", 0),
        ("Signal1", "ES_Distance", 0),
        ("Cruise_Fault", "ES_Distance", 0),
        ("Cruise_Throttle", "ES_Distance", 0),
        ("Signal2", "ES_Distance", 0),
        ("Car_Follow", "ES_Distance", 0),
        ("Signal3", "ES_Distance", 0),
        ("Cruise_Brake_Active", "ES_Distance", 0),
        ("Distance_Swap", "ES_Distance", 0),
        ("Cruise_EPB", "ES_Distance", 0),
        ("Signal4", "ES_Distance", 0),
        ("Close_Distance", "ES_Distance", 0),
        ("Signal5", "ES_Distance", 0),
        ("Cruise_Cancel", "ES_Distance", 0),
        ("Cruise_Set", "ES_Distance", 0),
        ("Cruise_Resume", "ES_Distance", 0),
        ("Signal6", "ES_Distance", 0),

        ("Counter", "ES_LKAS_State", 0),
        ("Keep_Hands_On_Wheel", "ES_LKAS_State", 0),
        ("Empty_Box", "ES_LKAS_State", 0),
        ("Signal1", "ES_LKAS_State", 0),
        ("LKAS_ACTIVE", "ES_LKAS_State", 0),
        ("Signal2", "ES_LKAS_State", 0),
        ("Backward_Speed_Limit_Menu", "ES_LKAS_State", 0),
        ("LKAS_ENABLE_3", "ES_LKAS_State", 0),
        ("LKAS_Left_Line_Light_Blink", "ES_LKAS_State", 0),
        ("LKAS_ENABLE_2", "ES_LKAS_State", 0),
        ("LKAS_Right_Line_Light_Blink", "ES_LKAS_State", 0),
        ("LKAS_Left_Line_Visible", "ES_LKAS_State", 0),
        ("LKAS_Left_Line_Green", "ES_LKAS_State", 0),
        ("LKAS_Right_Line_Visible", "ES_LKAS_State", 0),
        ("LKAS_Right_Line_Green", "ES_LKAS_State", 0),
        ("LKAS_Alert", "ES_LKAS_State", 0),
        ("Signal3", "ES_LKAS_State", 0),
      ]

      checks = [
        ("ES_DashStatus", 10),
        ("ES_Distance", 20),
        ("ES_LKAS_State", 10),
      ]

    return CANParser(DBC[CP.carFingerprint]['pt'], signals, checks, 2)
