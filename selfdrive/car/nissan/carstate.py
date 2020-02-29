import copy
from cereal import car
from selfdrive.car.interfaces import CarStateBase
from selfdrive.config import Conversions as CV
from opendbc.can.parser import CANParser
from selfdrive.car.nissan.values import DBC


class CarState(CarStateBase):
  def __init__(self, CP):
    # initialize can parser
    super().__init__(CP)

    self.left_blinker_on = False
    self.prev_left_blinker_on = False
    self.right_blinker_on = False
    self.prev_right_blinker_on = False
    self.steer_torque_driver = 0
    self.steer_not_allowed = False
    self.main_on = False

    self.v_ego = 0.

  def update(self, cp, cp_adas, cp_cam):
    ret = car.CarState.new_message()

    ret.gas = cp.vl["Throttle"]["ThrottlePedal"]
    ret.gasPressed = bool(ret.gas)
    ret.brakePressed = bool(cp.vl["DoorsLights"]["USER_BRAKE_PRESSED"])
    ret.brakeLights = bool(cp.vl["DoorsLights"]["BRAKE_LIGHT"])

    ret.wheelSpeeds.fl = cp.vl["WheelspeedFront"]["FL"] * CV.KPH_TO_MS
    ret.wheelSpeeds.fr = cp.vl["WheelspeedFront"]["FR"] * CV.KPH_TO_MS
    ret.wheelSpeeds.rl = cp.vl["WheelspeedRear"]["RL"] * CV.KPH_TO_MS
    ret.wheelSpeeds.rr = cp.vl["WheelspeedRear"]["RR"] * CV.KPH_TO_MS

    ret.vEgoRaw = (ret.wheelSpeeds.fl + ret.wheelSpeeds.fr + ret.wheelSpeeds.rl + ret.wheelSpeeds.rr) / 4.

    # Kalman filter, even though Subaru raw wheel speed is heaviliy filtered by default
    ret.vEgo, ret.aEgo = self.update_speed_kf(ret.vEgoRaw)
    ret.standstill = ret.vEgoRaw < 0.01

    # TODO: Work out gear shifter imessage
    ret.gearShifter = self.parse_gear_shifter("D")

    ret.leftBlinker = bool(cp.vl["Lights"]["LEFT_BLINKER"])
    ret.rightBlinker = bool(cp.vl["Lights"]["RIGHT_BLINKER"])
    # TODO: Work out Seatbelt message
    ret.seatbeltUnlatched = False
    ret.cruiseState.enabled = bool(cp_cam.vl["ProPilot"]["CRUISE_ACTIVATED"])
    ret.cruiseState.available = bool(cp_cam.vl["ProPilot"]["CRUISE_ON"])

    ret.doorOpen = any([cp.vl["DoorsLights"]["DOOR_OPEN_RR"],
      cp.vl["DoorsLights"]["DOOR_OPEN_RL"],
      cp.vl["DoorsLights"]["DOOR_OPEN_FR"],
      cp.vl["DoorsLights"]["DOOR_OPEN_FL"]])

    ret.steeringPressed = bool(cp.vl["STEER_TORQUE"]["DriverTouchingWheel"])
    ret.steeringTorque = cp.vl["Steering"]["DriverTorque"]
    ret.steeringAngle = cp.vl["SteeringWheel"]["Steering_Angle"]

    self.cruise_throttle_msg = copy.copy(cp.vl["CruiseThrottle"])

    return ret

  @staticmethod
  def get_can_parser(CP):
    # this function generates lists for signal, messages and initial values
    signals = [
      # sig_name, sig_address, default
      ("FL", "WheelspeedFront", 0),
      ("FR", "WheelspeedFront", 0),
      ("RL", "WheelspeedRear", 0),
      ("RR", "WheelspeedRear", 0),
      ("DOOR_OPEN_FR", "DoorsLights", 1),
      ("DOOR_OPEN_FL", "DoorsLights", 1),
      ("DOOR_OPEN_RR", "DoorsLights", 1),
      ("DOOR_OPEN_RL", "DoorsLights", 1),
      ("USER_BRAKE_PRESSED", "DoorsLights", 1),
      ("BRAKE_LIGHT", "DoorsLights", 1),
      ("DriverTorque", "Steering", 0),
      ("DriverTouchingWheel", "STEER_TORQUE", 0),
      ("ThrottlePedal", "Throttle", 0),
      ("Steering_Angle", "SteeringWheel", 0),
      ("RIGHT_BLINKER", "Lights", 0),
      ("LEFT_BLINKER", "Lights", 0),
      ("PROPILOT_BUTTON", "CruiseThrottle", 0),
      ("CANCEL_BUTTON", "CruiseThrottle", 0),
      ("GAS_PEDAL_INVERTED", "CruiseThrottle", 0),
      ("unsure2", "CruiseThrottle", 0),
      ("SET_BUTTON", "CruiseThrottle", 0),
      ("RES_BUTTON", "CruiseThrottle", 0),
      ("FOLLOW_DISTANCE_BUTTON", "CruiseThrottle", 0),
      ("NO_BUTTON_PRESSED", "CruiseThrottle", 0),
      ("GAS_PEDAL", "CruiseThrottle", 0),
      ("unsure3", "CruiseThrottle", 0),
      ("unsure", "CruiseThrottle", 0),
    ]

    checks = [
      # sig_address, frequency
      ("WheelspeedRear", 50),
      ("WheelspeedFront", 50),
      ("DoorsLights", 10),
    ]

    return CANParser(DBC[CP.carFingerprint]['pt'], signals, checks, 0)

  @staticmethod
  def get_adas_can_parser(CP):
    # this function generates lists for signal, messages and initial values
    signals = [
      # sig_name, sig_address, default
      ("Des_Angle", "LKAS", 0),
      ("SET_0x80_2", "LKAS", 0),
      ("NEW_SIGNAL_4", "LKAS", 0),
      ("SET_X80", "LKAS", 0),
      ("Counter", "LKAS", 0),
      ("LKA_Active", "LKAS", 0),
    ]

    checks = [
      # sig_address, frequency
    ]

    return CANParser(DBC[CP.carFingerprint]['pt'], signals, checks, 2)

  @staticmethod
  def get_cam_can_parser(CP):
    signals = [
      ("CRUISE_ON", "ProPilot", 0),
      ("CRUISE_ACTIVATED", "ProPilot", 0),
      ("STEER_STATUS", "ProPilot", 0),
    ]

    checks = [
    ]

    return CANParser(DBC[CP.carFingerprint]['pt'], signals, checks, 1)
