import copy
from cereal import car
from opendbc.can.can_define import CANDefine
from selfdrive.car.interfaces import CarStateBase
from selfdrive.config import Conversions as CV
from opendbc.can.parser import CANParser
from selfdrive.car.nissan.values import DBC


class CarState(CarStateBase):
  def __init__(self, CP):
    super().__init__(CP)
    can_define = CANDefine(DBC[CP.carFingerprint]['pt'])
    self.shifter_values = can_define.dv["GEARBOX"]["GEAR_SHIFTER"]

  def update(self, cp, cp_adas, cp_cam):
    ret = car.CarState.new_message()

    ret.gas = cp.vl["GAS_PEDAL"]["GAS_PEDAL"]
    ret.gasPressed = bool(ret.gas)
    ret.brakePressed = bool(cp.vl["DOORS_LIGHTS"]["USER_BRAKE_PRESSED"])
    ret.brakeLights = bool(cp.vl["DOORS_LIGHTS"]["BRAKE_LIGHT"])

    ret.wheelSpeeds.fl = cp.vl["WHEEL_SPEEDS_FRONT"]["WHEEL_SPEED_FL"] * CV.KPH_TO_MS
    ret.wheelSpeeds.fr = cp.vl["WHEEL_SPEEDS_FRONT"]["WHEEL_SPEED_FR"] * CV.KPH_TO_MS
    ret.wheelSpeeds.rl = cp.vl["WHEEL_SPEEDS_REAR"]["WHEEL_SPEED_RL"] * CV.KPH_TO_MS
    ret.wheelSpeeds.rr = cp.vl["WHEEL_SPEEDS_REAR"]["WHEEL_SPEED_RR"] * CV.KPH_TO_MS

    ret.vEgoRaw = (ret.wheelSpeeds.fl + ret.wheelSpeeds.fr + ret.wheelSpeeds.rl + ret.wheelSpeeds.rr) / 4.

    # Kalman filter, even though Subaru raw wheel speed is heaviliy filtered by default
    ret.vEgo, ret.aEgo = self.update_speed_kf(ret.vEgoRaw)
    ret.standstill = ret.vEgoRaw < 0.01

    can_gear = int(cp.vl["GEARBOX"]["GEAR_SHIFTER"])
    ret.gearShifter = self.parse_gear_shifter(self.shifter_values.get(can_gear, None))

    # Don't see this in the drive
    ret.leftBlinker = bool(cp.vl["LIGHTS"]["LEFT_BLINKER"])
    ret.rightBlinker = bool(cp.vl["LIGHTS"]["RIGHT_BLINKER"])

    ret.seatbeltUnlatched = cp.vl["SEATBELT"]["SEATBELT_DRIVER_UNLATCHED"] == 0
    ret.cruiseState.enabled = bool(cp_cam.vl["PRO_PILOT"]["CRUISE_ACTIVATED"])
    ret.cruiseState.available = bool(cp_cam.vl["PRO_PILOT"]["CRUISE_ON"])

    ret.doorOpen = any([cp.vl["DOORS_LIGHTS"]["DOOR_OPEN_RR"],
                        cp.vl["DOORS_LIGHTS"]["DOOR_OPEN_RL"],
                        cp.vl["DOORS_LIGHTS"]["DOOR_OPEN_FR"],
                        cp.vl["DOORS_LIGHTS"]["DOOR_OPEN_FL"]])

    ret.steeringPressed = bool(cp.vl["STEER_TORQUE_SENSOR2"]["STEERING_PRESSED"])
    ret.steeringTorque = cp.vl["STEER_TORQUE_SENSOR"]["STEER_TORQUE_DRIVER"]
    ret.steeringAngle = cp.vl["STEER_ANGLE_SENSOR"]["STEER_ANGLE"]

    ret.espDisabled = bool(cp.vl["ESP"]["ESP_DISABLED"])

    self.cruise_throttle_msg = copy.copy(cp.vl["CRUISE_THROTTLE"])
    self.lkas_hud_msg = copy.copy(cp_adas.vl["PROPILOT_HUD"])
    self.lkas_hud_info_msg = copy.copy(cp_adas.vl["PROPILOT_HUD_INFO_MSG"])

    return ret

  @staticmethod
  def get_can_parser(CP):
    # this function generates lists for signal, messages and initial values
    signals = [
      # sig_name, sig_address, default
      ("WHEEL_SPEED_FL", "WHEEL_SPEEDS_FRONT", 0),
      ("WHEEL_SPEED_FR", "WHEEL_SPEEDS_FRONT", 0),
      ("WHEEL_SPEED_RL", "WHEEL_SPEEDS_REAR", 0),
      ("WHEEL_SPEED_RR", "WHEEL_SPEEDS_REAR", 0),

      ("SEATBELT_DRIVER_UNLATCHED", "SEATBELT", 0),

      ("DOOR_OPEN_FR", "DOORS_LIGHTS", 1),
      ("DOOR_OPEN_FL", "DOORS_LIGHTS", 1),
      ("DOOR_OPEN_RR", "DOORS_LIGHTS", 1),
      ("DOOR_OPEN_RL", "DOORS_LIGHTS", 1),

      ("USER_BRAKE_PRESSED", "DOORS_LIGHTS", 1),
      ("BRAKE_LIGHT", "DOORS_LIGHTS", 1),

      ("GAS_PEDAL", "GAS_PEDAL", 0),

      ("STEER_TORQUE_DRIVER", "STEER_TORQUE_SENSOR", 0),
      ("STEERING_PRESSED", "STEER_TORQUE_SENSOR2", 0),

      ("STEER_ANGLE", "STEER_ANGLE_SENSOR", 0),

      ("RIGHT_BLINKER", "LIGHTS", 0),
      ("LEFT_BLINKER", "LIGHTS", 0),

      ("PROPILOT_BUTTON", "CRUISE_THROTTLE", 0),
      ("CANCEL_BUTTON", "CRUISE_THROTTLE", 0),
      ("GAS_PEDAL_INVERTED", "CRUISE_THROTTLE", 0),
      ("SET_BUTTON", "CRUISE_THROTTLE", 0),
      ("RES_BUTTON", "CRUISE_THROTTLE", 0),
      ("FOLLOW_DISTANCE_BUTTON", "CRUISE_THROTTLE", 0),
      ("NO_BUTTON_PRESSED", "CRUISE_THROTTLE", 0),
      ("GAS_PEDAL", "CRUISE_THROTTLE", 0),
      ("unsure1", "CRUISE_THROTTLE", 0),
      ("unsure2", "CRUISE_THROTTLE", 0),
      ("unsure3", "CRUISE_THROTTLE", 0),
      # TODO: why are USER_BRAKE_PRESSED, NEW_SIGNAL_2 and GAS_PRESSED_INVERTED  not forwarded

      ("ESP_DISABLED", "ESP", 0),
      ("GEAR_SHIFTER", "GEARBOX", 0),
    ]

    checks = [
      # sig_address, frequency
      ("WHEEL_SPEEDS_REAR", 50),
      ("WHEEL_SPEEDS_FRONT", 50),
      ("DOORS_LIGHTS", 10),
    ]

    return CANParser(DBC[CP.carFingerprint]['pt'], signals, checks, 0)

  @staticmethod
  def get_adas_can_parser(CP):
    # this function generates lists for signal, messages and initial values
    signals = [
      # sig_name, sig_address, default
      ("DESIRED_ANGLE", "LKAS", 0),
      ("SET_0x80_2", "LKAS", 0),
      ("MAX_TORQUE", "LKAS", 0),
      ("SET_0x80", "LKAS", 0),
      ("COUNTER", "LKAS", 0),
      ("LKA_ACTIVE", "LKAS", 0),

      # Below are the HUD messages. We copy the stock message and modify
      ("LARGE_WARNING_FLASHING", "PROPILOT_HUD", 0),
      ("SIDE_RADAR_ERROR_FLASHING1", "PROPILOT_HUD", 0),
      ("SIDE_RADAR_ERROR_FLASHING2", "PROPILOT_HUD", 0),
      ("LEAD_CAR", "PROPILOT_HUD", 0),
      ("LEAD_CAR_ERROR", "PROPILOT_HUD", 0),
      ("FRONT_RADAR_ERROR", "PROPILOT_HUD", 0),
      ("FRONT_RADAR_ERROR_FLASHING", "PROPILOT_HUD", 0),
      ("SIDE_RADAR_ERROR_FLASHING3", "PROPILOT_HUD", 0),
      ("LKAS_ERROR_FLASHING", "PROPILOT_HUD", 0),
      ("SAFETY_SHIELD_ACTIVE", "PROPILOT_HUD", 0),
      ("RIGHT_LANE_GREEN_FLASH", "PROPILOT_HUD", 0),
      ("LEFT_LANE_GREEN_FLASH", "PROPILOT_HUD", 0),
      ("FOLLOW_DISTANCE", "PROPILOT_HUD", 0),
      ("AUDIBLE_TONE", "PROPILOT_HUD", 0),
      ("SPEED_SET_ICON", "PROPILOT_HUD", 0),
      ("SMALL_STEERING_WHEEL_ICON", "PROPILOT_HUD", 0),
      ("unknown59", "PROPILOT_HUD", 0),
      ("unknown55", "PROPILOT_HUD", 0),
      ("unknown26", "PROPILOT_HUD", 0),
      ("unknown28", "PROPILOT_HUD", 0),
      ("unknown31", "PROPILOT_HUD", 0),
      ("unknown39", "PROPILOT_HUD", 0),
      ("unknown43", "PROPILOT_HUD", 0),
      ("unknown08", "PROPILOT_HUD", 0),
      ("unknown05", "PROPILOT_HUD", 0),
      ("unknown02", "PROPILOT_HUD", 0),

      ("NA_HIGH_ACCEL_TEMP", "PROPILOT_HUD_INFO_MSG", 0),
      ("SIDE_RADAR_NA_HIGH_CABIN_TEMP", "PROPILOT_HUD_INFO_MSG", 0),
      ("SIDE_RADAR_MALFUNCTION", "PROPILOT_HUD_INFO_MSG", 0),
      ("LKAS_MALFUNCTION", "PROPILOT_HUD_INFO_MSG", 0),
      ("FRONT_RADAR_MALFUNCTION", "PROPILOT_HUD_INFO_MSG", 0),
      ("SIDE_RADAR_NA_CLEAN_REAR_CAMERA", "PROPILOT_HUD_INFO_MSG", 0),
      ("NA_POOR_ROAD_CONDITIONS", "PROPILOT_HUD_INFO_MSG", 0),
      ("CURRENTLY_UNAVAILABLE", "PROPILOT_HUD_INFO_MSG", 0),
      ("SAFETY_SHIELD_OFF", "PROPILOT_HUD_INFO_MSG", 0),
      ("FRONT_COLLISION_NA_FRONT_RADAR_OBSTRUCTION", "PROPILOT_HUD_INFO_MSG", 0),
      ("PEDAL_MISSAPPLICATION_SYSTEM_ACTIVATED", "PROPILOT_HUD_INFO_MSG", 0),
      ("SIDE_IMPACT_NA_RADAR_OBSTRUCTION", "PROPILOT_HUD_INFO_MSG", 0),
      ("WARNING_DO_NOT_ENTER", "PROPILOT_HUD_INFO_MSG", 0),
      ("SIDE_IMPACT_SYSTEM_OFF", "PROPILOT_HUD_INFO_MSG", 0),
      ("SIDE_IMPACT_MALFUNCTION", "PROPILOT_HUD_INFO_MSG", 0),
      ("FRONT_COLLISION_MALFUNCTION", "PROPILOT_HUD_INFO_MSG", 0),
      ("SIDE_RADAR_MALFUNCTION2", "PROPILOT_HUD_INFO_MSG", 0),
      ("LKAS_MALFUNCTION2", "PROPILOT_HUD_INFO_MSG", 0),
      ("FRONT_RADAR_MALFUNCTION2", "PROPILOT_HUD_INFO_MSG", 0),
      ("PROPILOT_NA_MSGS", "PROPILOT_HUD_INFO_MSG", 0),
      ("BOTTOM_MSG", "PROPILOT_HUD_INFO_MSG", 0),
      ("HANDS_ON_WHEEL_WARNING", "PROPILOT_HUD_INFO_MSG", 0),
      ("WARNING_STEP_ON_BRAKE_NOW", "PROPILOT_HUD_INFO_MSG", 0),
      ("PROPILOT_NA_FRONT_CAMERA_OBSTRUCTED", "PROPILOT_HUD_INFO_MSG", 0),
      ("PROPILOT_NA_HIGH_CABIN_TEMP", "PROPILOT_HUD_INFO_MSG", 0),
      ("WARNING_PROPILOT_MALFUNCTION", "PROPILOT_HUD_INFO_MSG", 0),
      ("ACC_UNAVAILABLE_HIGH_CABIN_TEMP", "PROPILOT_HUD_INFO_MSG", 0),
      ("ACC_NA_FRONT_CAMERA_IMPARED", "PROPILOT_HUD_INFO_MSG", 0),
      ("unknown07", "PROPILOT_HUD_INFO_MSG", 0),
      ("unknown10", "PROPILOT_HUD_INFO_MSG", 0),
      ("unknown15", "PROPILOT_HUD_INFO_MSG", 0),
      ("unknown23", "PROPILOT_HUD_INFO_MSG", 0),
      ("unknown19", "PROPILOT_HUD_INFO_MSG", 0),
      ("unknown31", "PROPILOT_HUD_INFO_MSG", 0),
      ("unknown32", "PROPILOT_HUD_INFO_MSG", 0),
      ("unknown46", "PROPILOT_HUD_INFO_MSG", 0),
      ("unknown61", "PROPILOT_HUD_INFO_MSG", 0),
      ("unknown55", "PROPILOT_HUD_INFO_MSG", 0),
      ("unknown50", "PROPILOT_HUD_INFO_MSG", 0),

    ]

    checks = [
      # sig_address, frequency
    ]

    return CANParser(DBC[CP.carFingerprint]['pt'], signals, checks, 2)

  @staticmethod
  def get_cam_can_parser(CP):
    signals = [
      ("CRUISE_ON", "PRO_PILOT", 0),
      ("CRUISE_ACTIVATED", "PRO_PILOT", 0),
      ("STEER_STATUS", "PRO_PILOT", 0),
    ]

    checks = [
    ]

    return CANParser(DBC[CP.carFingerprint]['pt'], signals, checks, 1)
