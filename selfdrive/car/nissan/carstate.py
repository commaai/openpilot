import copy
from collections import deque
from cereal import car
from opendbc.can.can_define import CANDefine
from openpilot.selfdrive.car.interfaces import CarStateBase
from openpilot.common.conversions import Conversions as CV
from opendbc.can.parser import CANParser
from openpilot.selfdrive.car.nissan.values import CAR, DBC, CarControllerParams

TORQUE_SAMPLES = 12

class CarState(CarStateBase):
  def __init__(self, CP):
    super().__init__(CP)
    can_define = CANDefine(DBC[CP.carFingerprint]["pt"])

    self.lkas_hud_msg = {}
    self.lkas_hud_info_msg = {}

    self.steeringTorqueSamples = deque(TORQUE_SAMPLES*[0], TORQUE_SAMPLES)
    self.shifter_values = can_define.dv["GEARBOX"]["GEAR_SHIFTER"]

    self.prev_distance_button = 0
    self.distance_button = 0

  def update(self, cp, cp_adas, cp_cam):
    ret = car.CarState.new_message()

    self.prev_distance_button = self.distance_button
    self.distance_button = cp.vl["CRUISE_THROTTLE"]["FOLLOW_DISTANCE_BUTTON"]

    if self.CP.carFingerprint in (CAR.ROGUE, CAR.XTRAIL, CAR.ALTIMA):
      ret.gas = cp.vl["GAS_PEDAL"]["GAS_PEDAL"]
    elif self.CP.carFingerprint in (CAR.LEAF, CAR.LEAF_IC):
      ret.gas = cp.vl["CRUISE_THROTTLE"]["GAS_PEDAL"]

    ret.gasPressed = bool(ret.gas > 3)

    if self.CP.carFingerprint in (CAR.ROGUE, CAR.XTRAIL, CAR.ALTIMA):
      ret.brakePressed = bool(cp.vl["DOORS_LIGHTS"]["USER_BRAKE_PRESSED"])
    elif self.CP.carFingerprint in (CAR.LEAF, CAR.LEAF_IC):
      ret.brakePressed = bool(cp.vl["CRUISE_THROTTLE"]["USER_BRAKE_PRESSED"])

    ret.wheelSpeeds = self.get_wheel_speeds(
      cp.vl["WHEEL_SPEEDS_FRONT"]["WHEEL_SPEED_FL"],
      cp.vl["WHEEL_SPEEDS_FRONT"]["WHEEL_SPEED_FR"],
      cp.vl["WHEEL_SPEEDS_REAR"]["WHEEL_SPEED_RL"],
      cp.vl["WHEEL_SPEEDS_REAR"]["WHEEL_SPEED_RR"],
    )
    ret.vEgoRaw = (ret.wheelSpeeds.fl + ret.wheelSpeeds.fr + ret.wheelSpeeds.rl + ret.wheelSpeeds.rr) / 4.

    ret.vEgo, ret.aEgo = self.update_speed_kf(ret.vEgoRaw)
    ret.standstill = cp.vl["WHEEL_SPEEDS_REAR"]["WHEEL_SPEED_RL"] == 0.0 and cp.vl["WHEEL_SPEEDS_REAR"]["WHEEL_SPEED_RR"] == 0.0

    if self.CP.carFingerprint == CAR.ALTIMA:
      ret.cruiseState.enabled = bool(cp.vl["CRUISE_STATE"]["CRUISE_ENABLED"])
    else:
      ret.cruiseState.enabled = bool(cp_adas.vl["CRUISE_STATE"]["CRUISE_ENABLED"])

    if self.CP.carFingerprint in (CAR.ROGUE, CAR.XTRAIL):
      ret.seatbeltUnlatched = cp.vl["HUD"]["SEATBELT_DRIVER_LATCHED"] == 0
      ret.cruiseState.available = bool(cp_cam.vl["PRO_PILOT"]["CRUISE_ON"])
    elif self.CP.carFingerprint in (CAR.LEAF, CAR.LEAF_IC):
      if self.CP.carFingerprint == CAR.LEAF:
        ret.seatbeltUnlatched = cp.vl["SEATBELT"]["SEATBELT_DRIVER_LATCHED"] == 0
      elif self.CP.carFingerprint == CAR.LEAF_IC:
        ret.seatbeltUnlatched = cp.vl["CANCEL_MSG"]["CANCEL_SEATBELT"] == 1
      ret.cruiseState.available = bool(cp.vl["CRUISE_THROTTLE"]["CRUISE_AVAILABLE"])
    elif self.CP.carFingerprint == CAR.ALTIMA:
      ret.seatbeltUnlatched = cp.vl["HUD"]["SEATBELT_DRIVER_LATCHED"] == 0
      ret.cruiseState.available = bool(cp_adas.vl["PRO_PILOT"]["CRUISE_ON"])

    if self.CP.carFingerprint == CAR.ALTIMA:
      speed = cp.vl["PROPILOT_HUD"]["SET_SPEED"]
    else:
      speed = cp_adas.vl["PROPILOT_HUD"]["SET_SPEED"]

    if speed != 255:
      if self.CP.carFingerprint in (CAR.LEAF, CAR.LEAF_IC):
        conversion = CV.MPH_TO_MS if cp.vl["HUD_SETTINGS"]["SPEED_MPH"] else CV.KPH_TO_MS
      else:
        conversion = CV.MPH_TO_MS if cp.vl["HUD"]["SPEED_MPH"] else CV.KPH_TO_MS
      ret.cruiseState.speed = speed * conversion
      ret.cruiseState.speedCluster = (speed - 1) * conversion  # Speed on HUD is always 1 lower than actually sent on can bus

    if self.CP.carFingerprint == CAR.ALTIMA:
      ret.steeringTorque = cp_cam.vl["STEER_TORQUE_SENSOR"]["STEER_TORQUE_DRIVER"]
    else:
      ret.steeringTorque = cp.vl["STEER_TORQUE_SENSOR"]["STEER_TORQUE_DRIVER"]

    self.steeringTorqueSamples.append(ret.steeringTorque)
    # Filtering driver torque to prevent steeringPressed false positives
    ret.steeringPressed = bool(abs(sum(self.steeringTorqueSamples) / TORQUE_SAMPLES) > CarControllerParams.STEER_THRESHOLD)

    ret.steeringAngleDeg = cp.vl["STEER_ANGLE_SENSOR"]["STEER_ANGLE"]

    ret.leftBlinker = bool(cp.vl["LIGHTS"]["LEFT_BLINKER"])
    ret.rightBlinker = bool(cp.vl["LIGHTS"]["RIGHT_BLINKER"])

    ret.doorOpen = any([cp.vl["DOORS_LIGHTS"]["DOOR_OPEN_RR"],
                        cp.vl["DOORS_LIGHTS"]["DOOR_OPEN_RL"],
                        cp.vl["DOORS_LIGHTS"]["DOOR_OPEN_FR"],
                        cp.vl["DOORS_LIGHTS"]["DOOR_OPEN_FL"]])

    ret.espDisabled = bool(cp.vl["ESP"]["ESP_DISABLED"])

    can_gear = int(cp.vl["GEARBOX"]["GEAR_SHIFTER"])
    ret.gearShifter = self.parse_gear_shifter(self.shifter_values.get(can_gear, None))

    if self.CP.carFingerprint == CAR.ALTIMA:
      self.lkas_enabled = bool(cp.vl["LKAS_SETTINGS"]["LKAS_ENABLED"])
    else:
      self.lkas_enabled = bool(cp_adas.vl["LKAS_SETTINGS"]["LKAS_ENABLED"])

    self.cruise_throttle_msg = copy.copy(cp.vl["CRUISE_THROTTLE"])

    if self.CP.carFingerprint in (CAR.LEAF, CAR.LEAF_IC):
      self.cancel_msg = copy.copy(cp.vl["CANCEL_MSG"])

    if self.CP.carFingerprint != CAR.ALTIMA:
      self.lkas_hud_msg = copy.copy(cp_adas.vl["PROPILOT_HUD"])
      self.lkas_hud_info_msg = copy.copy(cp_adas.vl["PROPILOT_HUD_INFO_MSG"])

    return ret

  @staticmethod
  def get_can_parser(CP):
    messages = [
      # sig_address, frequency
      ("STEER_ANGLE_SENSOR", 100),
      ("WHEEL_SPEEDS_REAR", 50),
      ("WHEEL_SPEEDS_FRONT", 50),
      ("ESP", 25),
      ("GEARBOX", 25),
      ("DOORS_LIGHTS", 10),
      ("LIGHTS", 10),
    ]

    if CP.carFingerprint in (CAR.ROGUE, CAR.XTRAIL, CAR.ALTIMA):
      messages += [
        ("GAS_PEDAL", 100),
        ("CRUISE_THROTTLE", 50),
        ("HUD", 25),
      ]

    elif CP.carFingerprint in (CAR.LEAF, CAR.LEAF_IC):
      messages += [
        ("BRAKE_PEDAL", 100),
        ("CRUISE_THROTTLE", 50),
        ("CANCEL_MSG", 50),
        ("HUD_SETTINGS", 25),
        ("SEATBELT", 10),
      ]

    if CP.carFingerprint == CAR.ALTIMA:
      messages += [
        ("CRUISE_STATE", 10),
        ("LKAS_SETTINGS", 10),
        ("PROPILOT_HUD", 50),
      ]
      return CANParser(DBC[CP.carFingerprint]["pt"], messages, 1)

    messages.append(("STEER_TORQUE_SENSOR", 100))

    return CANParser(DBC[CP.carFingerprint]["pt"], messages, 0)

  @staticmethod
  def get_adas_can_parser(CP):
    # this function generates lists for signal, messages and initial values

    if CP.carFingerprint == CAR.ALTIMA:
      messages = [
        ("LKAS", 100),
        ("PRO_PILOT", 100),
      ]
    else:
      messages = [
        ("PROPILOT_HUD_INFO_MSG", 2),
        ("LKAS_SETTINGS", 10),
        ("CRUISE_STATE", 50),
        ("PROPILOT_HUD", 50),
        ("LKAS", 100),
      ]

    return CANParser(DBC[CP.carFingerprint]["pt"], messages, 2)

  @staticmethod
  def get_cam_can_parser(CP):
    messages = []

    if CP.carFingerprint in (CAR.ROGUE, CAR.XTRAIL):
      messages.append(("PRO_PILOT", 100))
    elif CP.carFingerprint == CAR.ALTIMA:
      messages.append(("STEER_TORQUE_SENSOR", 100))
      return CANParser(DBC[CP.carFingerprint]["pt"], messages, 0)

    return CANParser(DBC[CP.carFingerprint]["pt"], messages, 1)
