from cereal import car
from selfdrive.config import Conversions as CV
from opendbc.can.can_define import CANDefine
from opendbc.can.parser import CANParser
from selfdrive.car.interfaces import CarStateBase
from selfdrive.car.mazda.values import DBC, LKAS_LIMITS, GEN1

class CarState(CarStateBase):
  def __init__(self, CP):
    super().__init__(CP)

    can_define = CANDefine(DBC[CP.carFingerprint]["pt"])
    self.shifter_values = can_define.dv["GEAR"]["GEAR"]

    self.cruise_speed = 0
    self.acc_active_last = False
    self.low_speed_lockout = True
    self.low_speed_alert = False
    self.lkas_allowed = False

  def update(self, cp, cp_cam):

    ret = car.CarState.new_message()
    ret.wheelSpeeds.fl = cp.vl["WHEEL_SPEEDS"]["FL"] * CV.KPH_TO_MS
    ret.wheelSpeeds.fr = cp.vl["WHEEL_SPEEDS"]["FR"] * CV.KPH_TO_MS
    ret.wheelSpeeds.rl = cp.vl["WHEEL_SPEEDS"]["RL"] * CV.KPH_TO_MS
    ret.wheelSpeeds.rr = cp.vl["WHEEL_SPEEDS"]["RR"] * CV.KPH_TO_MS
    ret.vEgoRaw = (ret.wheelSpeeds.fl + ret.wheelSpeeds.fr + ret.wheelSpeeds.rl + ret.wheelSpeeds.rr) / 4.
    ret.vEgo, ret.aEgo = self.update_speed_kf(ret.vEgoRaw)

    # Match panda speed reading
    speed_kph = cp.vl["ENGINE_DATA"]["SPEED"]
    ret.standstill = speed_kph < .1

    can_gear = int(cp.vl["GEAR"]["GEAR"])
    ret.gearShifter = self.parse_gear_shifter(self.shifter_values.get(can_gear, None))

    ret.leftBlinker = cp.vl["BLINK_INFO"]["LEFT_BLINK"] == 1
    ret.rightBlinker = cp.vl["BLINK_INFO"]["RIGHT_BLINK"] == 1

    ret.steeringAngleDeg = cp.vl["STEER"]["STEER_ANGLE"]
    ret.steeringTorque = cp.vl["STEER_TORQUE"]["STEER_TORQUE_SENSOR"]
    ret.steeringPressed = abs(ret.steeringTorque) > LKAS_LIMITS.STEER_THRESHOLD

    ret.steeringTorqueEps = cp.vl["STEER_TORQUE"]["STEER_TORQUE_MOTOR"]
    ret.steeringRateDeg = cp.vl["STEER_RATE"]["STEER_ANGLE_RATE"]

    ret.brakePressed = cp.vl["PEDALS"]["BRAKE_ON"] == 1
    ret.brake = cp.vl["BRAKE"]["BRAKE_PRESSURE"]

    ret.seatbeltUnlatched = cp.vl["SEATBELT"]["DRIVER_SEATBELT"] == 0
    ret.doorOpen = any([cp.vl["DOORS"]["FL"], cp.vl["DOORS"]["FR"],
                        cp.vl["DOORS"]["BL"], cp.vl["DOORS"]["BR"]])

    ret.gas = cp.vl["ENGINE_DATA"]["PEDAL_GAS"]
    ret.gasPressed = ret.gas > 0

    ret.leftBlindspot = cp.vl["BSM"]["LEFT_BS1"] == 1
    ret.rightBlindspot = cp.vl["BSM"]["RIGHT_BS1"] == 1

    # LKAS is enabled at 52kph going up and disabled at 45kph going down
    if speed_kph > LKAS_LIMITS.ENABLE_SPEED:
      self.lkas_allowed = True
    elif speed_kph < LKAS_LIMITS.DISABLE_SPEED:
      self.lkas_allowed = False

    ret.cruiseState.available = cp.vl["CRZ_CTRL"]["CRZ_AVAILABLE"] == 1
    ret.cruiseState.enabled = cp.vl["CRZ_CTRL"]["CRZ_ACTIVE"] == 1
    ret.cruiseState.speed = cp.vl["CRZ_EVENTS"]["CRZ_SPEED"] * CV.KPH_TO_MS

    if ret.cruiseState.enabled:
      if not self.lkas_allowed:
        if not self.acc_active_last:
          self.low_speed_lockout = True
        else:
          self.low_speed_alert = True
      else:
        self.low_speed_lockout = False
        self.low_speed_alert = False

    # Check if LKAS is disabled due to lack of driver torque when all other states indicate
    # it should be enabled (steer lockout)
    ret.steerWarning = self.lkas_allowed and (cp.vl["STEER_RATE"]["LKAS_BLOCK"] == 1)

    self.acc_active_last = ret.cruiseState.enabled

    self.cam_lkas = cp_cam.vl["CAM_LKAS"]
    ret.steerError = cp_cam.vl["CAM_LKAS"]["ERR_BIT_1"] == 1

    return ret

  @staticmethod
  def get_can_parser(CP):
    # this function generates lists for signal, messages and initial values
    signals = [
      # sig_name, sig_address, default
      ("LEFT_BLINK", "BLINK_INFO", 0),
      ("RIGHT_BLINK", "BLINK_INFO", 0),
      ("STEER_ANGLE", "STEER", 0),
      ("STEER_ANGLE_RATE", "STEER_RATE", 0),
      ("STEER_TORQUE_SENSOR", "STEER_TORQUE", 0),
      ("STEER_TORQUE_MOTOR", "STEER_TORQUE", 0),
      ("FL", "WHEEL_SPEEDS", 0),
      ("FR", "WHEEL_SPEEDS", 0),
      ("RL", "WHEEL_SPEEDS", 0),
      ("RR", "WHEEL_SPEEDS", 0),
    ]

    checks = [
      # sig_address, frequency
      ("BLINK_INFO", 10),
      ("STEER", 67),
      ("STEER_RATE", 83),
      ("STEER_TORQUE", 83),
      ("WHEEL_SPEEDS", 100),
    ]

    if CP.carFingerprint in GEN1:
      signals += [
        ("LKAS_BLOCK", "STEER_RATE", 0),
        ("LKAS_TRACK_STATE", "STEER_RATE", 0),
        ("HANDS_OFF_5_SECONDS", "STEER_RATE", 0),
        ("CRZ_ACTIVE", "CRZ_CTRL", 0),
        ("CRZ_AVAILABLE", "CRZ_CTRL", 0),
        ("CRZ_SPEED", "CRZ_EVENTS", 0),
        ("STANDSTILL", "PEDALS", 0),
        ("BRAKE_ON", "PEDALS", 0),
        ("BRAKE_PRESSURE", "BRAKE", 0),
        ("GEAR", "GEAR", 0),
        ("DRIVER_SEATBELT", "SEATBELT", 0),
        ("FL", "DOORS", 0),
        ("FR", "DOORS", 0),
        ("BL", "DOORS", 0),
        ("BR", "DOORS", 0),
        ("PEDAL_GAS", "ENGINE_DATA", 0),
        ("SPEED", "ENGINE_DATA", 0),
        ("RES", "CRZ_BTNS", 0),
        ("SET_P", "CRZ_BTNS", 0),
        ("SET_M", "CRZ_BTNS", 0),
        ("CTR", "CRZ_BTNS", 0),
        ("LEFT_BS1", "BSM", 0),
        ("RIGHT_BS1", "BSM", 0),
      ]

      checks += [
        ("ENGINE_DATA", 100),
        ("CRZ_CTRL", 50),
        ("CRZ_EVENTS", 50),
        ("CRZ_BTNS", 10),
        ("PEDALS", 50),
        ("BRAKE", 50),
        ("SEATBELT", 10),
        ("DOORS", 10),
        ("GEAR", 20),
        ("BSM", 10),
      ]

    return CANParser(DBC[CP.carFingerprint]["pt"], signals, checks, 0)

  @staticmethod
  def get_cam_can_parser(CP):
    signals = []
    checks = []

    if CP.carFingerprint in GEN1:
      signals += [
        # sig_name, sig_address, default
        ("LKAS_REQUEST",     "CAM_LKAS", 0),
        ("CTR",              "CAM_LKAS", 0),
        ("ERR_BIT_1",        "CAM_LKAS", 0),
        ("LINE_NOT_VISIBLE", "CAM_LKAS", 0),
        ("LDW",              "CAM_LKAS", 0),
        ("BIT_1",            "CAM_LKAS", 1),
        ("ERR_BIT_2",        "CAM_LKAS", 0),
        ("STEERING_ANGLE",   "CAM_LKAS", 0),
        ("ANGLE_ENABLED",    "CAM_LKAS", 0),
        ("CHKSUM",           "CAM_LKAS", 0),
      ]

      checks += [
        # sig_address, frequency
        ("CAM_LKAS",      16),
      ]

    return CANParser(DBC[CP.carFingerprint]["pt"], signals, checks, 2)
