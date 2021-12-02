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

    self.crz_btns_counter = 0
    self.acc_active_last = False
    self.low_speed_alert = False
    self.lkas_allowed_speed = False

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

    ret.genericToggle = bool(cp.vl["BLINK_INFO"]["HIGH_BEAMS"])
    ret.leftBlindspot = cp.vl["BSM"]["LEFT_BS1"] == 1
    ret.rightBlindspot = cp.vl["BSM"]["RIGHT_BS1"] == 1
    ret.leftBlinker, ret.rightBlinker = self.update_blinker_from_lamp(40, cp.vl["BLINK_INFO"]["LEFT_BLINK"] == 1,
                                                                      cp.vl["BLINK_INFO"]["RIGHT_BLINK"] == 1)

    ret.steeringAngleDeg = cp.vl["STEER"]["STEER_ANGLE"]
    ret.steeringTorque = cp.vl["STEER_TORQUE"]["STEER_TORQUE_SENSOR"]
    ret.steeringPressed = abs(ret.steeringTorque) > LKAS_LIMITS.STEER_THRESHOLD

    ret.steeringTorqueEps = cp.vl["STEER_TORQUE"]["STEER_TORQUE_MOTOR"]
    ret.steeringRateDeg = cp.vl["STEER_RATE"]["STEER_ANGLE_RATE"]

    # TODO: this should be from 0 - 1.
    ret.brakePressed = cp.vl["PEDALS"]["BRAKE_ON"] == 1
    ret.brake = cp.vl["BRAKE"]["BRAKE_PRESSURE"]

    ret.seatbeltUnlatched = cp.vl["SEATBELT"]["DRIVER_SEATBELT"] == 0
    ret.doorOpen = any([cp.vl["DOORS"]["FL"], cp.vl["DOORS"]["FR"],
                        cp.vl["DOORS"]["BL"], cp.vl["DOORS"]["BR"]])

    # TODO: this should be from 0 - 1.
    ret.gas = cp.vl["ENGINE_DATA"]["PEDAL_GAS"]
    ret.gasPressed = ret.gas > 0

    # Either due to low speed or hands off
    lkas_blocked = cp.vl["STEER_RATE"]["LKAS_BLOCK"] == 1

    # LKAS is enabled at 52kph going up and disabled at 45kph going down
    # wait for LKAS_BLOCK signal to clear when going up since it lags behind the speed sometimes
    if speed_kph > LKAS_LIMITS.ENABLE_SPEED and not lkas_blocked:
      self.lkas_allowed_speed = True
    elif speed_kph < LKAS_LIMITS.DISABLE_SPEED:
      self.lkas_allowed_speed = False

    # TODO: the signal used for available seems to be the adaptive cruise signal, instead of the main on
    #       it should be used for carState.cruiseState.nonAdaptive instead
    ret.cruiseState.available = cp.vl["CRZ_CTRL"]["CRZ_AVAILABLE"] == 1
    ret.cruiseState.enabled = cp.vl["CRZ_CTRL"]["CRZ_ACTIVE"] == 1
    ret.cruiseState.speed = cp.vl["CRZ_EVENTS"]["CRZ_SPEED"] * CV.KPH_TO_MS

    if ret.cruiseState.enabled:
      if not self.lkas_allowed_speed and self.acc_active_last:
        self.low_speed_alert = True
      else:
        self.low_speed_alert = False

    # Check if LKAS is disabled due to lack of driver torque when all other states indicate
    # it should be enabled (steer lockout). Don't warn until we actually get lkas active
    # and lose it again, i.e, after initial lkas activation

    ret.steerWarning = self.lkas_allowed_speed and lkas_blocked

    self.acc_active_last = ret.cruiseState.enabled

    self.cam_lkas = cp_cam.vl["CAM_LKAS"]
    self.cam_laneinfo = cp_cam.vl["CAM_LANEINFO"]
    self.crz_btns_counter = cp.vl["CRZ_BTNS"]["CTR"]
    ret.steerError = cp_cam.vl["CAM_LKAS"]["ERR_BIT_1"] == 1

    return ret

  @staticmethod
  def get_can_parser(CP):
    # this function generates lists for signal, messages and initial values
    signals = [
      # sig_name, sig_address, default
      ("LEFT_BLINK", "BLINK_INFO", 0),
      ("RIGHT_BLINK", "BLINK_INFO", 0),
      ("HIGH_BEAMS", "BLINK_INFO", 0),
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
        ("LKAS_REQUEST", "CAM_LKAS", 0),
        ("CTR", "CAM_LKAS", 0),
        ("ERR_BIT_1", "CAM_LKAS", 0),
        ("LINE_NOT_VISIBLE", "CAM_LKAS", 0),
        ("BIT_1", "CAM_LKAS", 1),
        ("ERR_BIT_2", "CAM_LKAS", 0),
        ("STEERING_ANGLE", "CAM_LKAS", 0),
        ("ANGLE_ENABLED", "CAM_LKAS", 0),
        ("CHKSUM", "CAM_LKAS", 0),

        ("LINE_VISIBLE", "CAM_LANEINFO", 0),
        ("LINE_NOT_VISIBLE", "CAM_LANEINFO", 1),
        ("LANE_LINES", "CAM_LANEINFO", 0),
        ("BIT1", "CAM_LANEINFO", 0),
        ("BIT2", "CAM_LANEINFO", 0),
        ("BIT3", "CAM_LANEINFO", 0),
        ("NO_ERR_BIT", "CAM_LANEINFO", 1),
        ("S1", "CAM_LANEINFO", 0),
        ("S1_HBEAM", "CAM_LANEINFO", 0),
      ]

      checks += [
        # sig_address, frequency
        ("CAM_LANEINFO", 2),
        ("CAM_LKAS", 16),
      ]

    return CANParser(DBC[CP.carFingerprint]["pt"], signals, checks, 2)
