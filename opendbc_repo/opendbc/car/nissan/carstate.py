import copy
from collections import deque
from opendbc.can import CANDefine, CANParser
from opendbc.car import Bus, create_button_events, structs
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.interfaces import CarStateBase
from opendbc.car.nissan.values import CAR, DBC, CarControllerParams

ButtonType = structs.CarState.ButtonEvent.Type

TORQUE_SAMPLES = 12


class CarState(CarStateBase):
  def __init__(self, CP):
    super().__init__(CP)
    can_define = CANDefine(DBC[CP.carFingerprint][Bus.pt])

    self.lkas_hud_msg = {}
    self.lkas_hud_info_msg = {}

    self.steeringTorqueSamples = deque(TORQUE_SAMPLES*[0], TORQUE_SAMPLES)
    self.shifter_values = can_define.dv["GEARBOX"]["GEAR_SHIFTER"]

    self.distance_button = 0

  def update(self, can_parsers) -> structs.CarState:
    cp = can_parsers[Bus.pt]
    cp_cam = can_parsers[Bus.cam]
    cp_adas = can_parsers[Bus.adas]

    ret = structs.CarState()

    prev_distance_button = self.distance_button
    self.distance_button = cp.vl["CRUISE_THROTTLE"]["FOLLOW_DISTANCE_BUTTON"]

    if self.CP.carFingerprint in (CAR.NISSAN_ROGUE, CAR.NISSAN_XTRAIL, CAR.NISSAN_ALTIMA):
      ret.gasPressed = bool(cp.vl["GAS_PEDAL"]["GAS_PEDAL"] > 3)
    elif self.CP.carFingerprint in (CAR.NISSAN_LEAF, CAR.NISSAN_LEAF_IC):
      ret.gasPressed = bool(cp.vl["CRUISE_THROTTLE"]["GAS_PEDAL"] > 3)

    if self.CP.carFingerprint in (CAR.NISSAN_ROGUE, CAR.NISSAN_XTRAIL, CAR.NISSAN_ALTIMA):
      ret.brakePressed = bool(cp.vl["DOORS_LIGHTS"]["USER_BRAKE_PRESSED"])
    elif self.CP.carFingerprint in (CAR.NISSAN_LEAF, CAR.NISSAN_LEAF_IC):
      ret.brakePressed = bool(cp.vl["CRUISE_THROTTLE"]["USER_BRAKE_PRESSED"])

    fl = cp.vl["WHEEL_SPEEDS_FRONT"]["WHEEL_SPEED_FL"] * CV.KPH_TO_MS
    fr = cp.vl["WHEEL_SPEEDS_FRONT"]["WHEEL_SPEED_FR"] * CV.KPH_TO_MS
    rl = cp.vl["WHEEL_SPEEDS_REAR"]["WHEEL_SPEED_RL"] * CV.KPH_TO_MS
    rr = cp.vl["WHEEL_SPEEDS_REAR"]["WHEEL_SPEED_RR"] * CV.KPH_TO_MS
    # safety uses the rear wheel speeds for the speed measurement and angle limiting
    ret.vEgoRaw = (rl + rr) / 2.0

    v_ego_raw_full = (fl + fr + rl + rr) / 4.0
    ret.vEgo, ret.aEgo = self.update_speed_kf(v_ego_raw_full)
    ret.standstill = cp.vl["WHEEL_SPEEDS_REAR"]["WHEEL_SPEED_RL"] == 0.0 and cp.vl["WHEEL_SPEEDS_REAR"]["WHEEL_SPEED_RR"] == 0.0

    if self.CP.carFingerprint == CAR.NISSAN_ALTIMA:
      ret.cruiseState.enabled = bool(cp.vl["CRUISE_STATE"]["CRUISE_ENABLED"])
    else:
      ret.cruiseState.enabled = bool(cp_adas.vl["CRUISE_STATE"]["CRUISE_ENABLED"])

    if self.CP.carFingerprint in (CAR.NISSAN_ROGUE, CAR.NISSAN_XTRAIL):
      ret.seatbeltUnlatched = cp.vl["HUD"]["SEATBELT_DRIVER_LATCHED"] == 0
      ret.cruiseState.available = bool(cp_cam.vl["PRO_PILOT"]["CRUISE_ON"])
    elif self.CP.carFingerprint in (CAR.NISSAN_LEAF, CAR.NISSAN_LEAF_IC):
      if self.CP.carFingerprint == CAR.NISSAN_LEAF:
        ret.seatbeltUnlatched = cp.vl["SEATBELT"]["SEATBELT_DRIVER_LATCHED"] == 0
      elif self.CP.carFingerprint == CAR.NISSAN_LEAF_IC:
        ret.seatbeltUnlatched = cp.vl["CANCEL_MSG"]["CANCEL_SEATBELT"] == 1
      ret.cruiseState.available = bool(cp.vl["CRUISE_THROTTLE"]["CRUISE_AVAILABLE"])
    elif self.CP.carFingerprint == CAR.NISSAN_ALTIMA:
      ret.seatbeltUnlatched = cp.vl["HUD"]["SEATBELT_DRIVER_LATCHED"] == 0
      ret.cruiseState.available = bool(cp_adas.vl["PRO_PILOT"]["CRUISE_ON"])

    if self.CP.carFingerprint == CAR.NISSAN_ALTIMA:
      speed = cp.vl["PROPILOT_HUD"]["SET_SPEED"]
    else:
      speed = cp_adas.vl["PROPILOT_HUD"]["SET_SPEED"]

    if speed != 255:
      if self.CP.carFingerprint in (CAR.NISSAN_LEAF, CAR.NISSAN_LEAF_IC):
        conversion = CV.MPH_TO_MS if cp.vl["HUD_SETTINGS"]["SPEED_MPH"] else CV.KPH_TO_MS
      else:
        conversion = CV.MPH_TO_MS if cp.vl["HUD"]["SPEED_MPH"] else CV.KPH_TO_MS
      ret.cruiseState.speed = speed * conversion
      ret.cruiseState.speedCluster = (speed - 1) * conversion  # Speed on HUD is always 1 lower than actually sent on can bus

    if self.CP.carFingerprint == CAR.NISSAN_ALTIMA:
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

    # stock lkas should be off
    # TODO: is this needed?
    if self.CP.carFingerprint == CAR.NISSAN_ALTIMA:
      ret.invalidLkasSetting = bool(cp.vl["LKAS_SETTINGS"]["LKAS_ENABLED"])
    else:
      ret.invalidLkasSetting = bool(cp_adas.vl["LKAS_SETTINGS"]["LKAS_ENABLED"])

    self.cruise_throttle_msg = copy.copy(cp.vl["CRUISE_THROTTLE"])

    if self.CP.carFingerprint in (CAR.NISSAN_LEAF, CAR.NISSAN_LEAF_IC):
      self.cancel_msg = copy.copy(cp.vl["CANCEL_MSG"])

    if self.CP.carFingerprint != CAR.NISSAN_ALTIMA:
      self.lkas_hud_msg = copy.copy(cp_adas.vl["PROPILOT_HUD"])
      self.lkas_hud_info_msg = copy.copy(cp_adas.vl["PROPILOT_HUD_INFO_MSG"])

    ret.buttonEvents = create_button_events(self.distance_button, prev_distance_button, {1: ButtonType.gapAdjustCruise})

    return ret

  @staticmethod
  def get_can_parsers(CP):
    return {
      Bus.pt: CANParser(DBC[CP.carFingerprint][Bus.pt], [], 1 if CP.carFingerprint == CAR.NISSAN_ALTIMA else 0),
      Bus.cam: CANParser(DBC[CP.carFingerprint][Bus.pt], [], 0 if CP.carFingerprint == CAR.NISSAN_ALTIMA else 1),
      Bus.adas: CANParser(DBC[CP.carFingerprint][Bus.pt], [], 2),
    }
