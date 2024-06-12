import crcmod
from openpilot.selfdrive.car.nissan.values import CAR

# TODO: add this checksum to the CANPacker
nissan_checksum = crcmod.mkCrcFun(0x11d, initCrc=0x00, rev=False, xorOut=0xff)


def create_steering_control(packer, apply_steer, frame, steer_on, lkas_max_torque):
  values = {
    "COUNTER": frame % 0x10,
    "DESIRED_ANGLE": apply_steer,
    "SET_0x80_2": 0x80,
    "SET_0x80": 0x80,
    "MAX_TORQUE": lkas_max_torque if steer_on else 0,
    "LKA_ACTIVE": steer_on,
  }

  dat = packer.make_can_msg("LKAS", 0, values)[2]

  values["CHECKSUM"] = nissan_checksum(dat[:7])
  return packer.make_can_msg("LKAS", 0, values)


def create_acc_cancel_cmd(packer, car_fingerprint, cruise_throttle_msg):
  values = {s: cruise_throttle_msg[s] for s in [
    "COUNTER",
    "PROPILOT_BUTTON",
    "CANCEL_BUTTON",
    "GAS_PEDAL_INVERTED",
    "SET_BUTTON",
    "RES_BUTTON",
    "FOLLOW_DISTANCE_BUTTON",
    "NO_BUTTON_PRESSED",
    "GAS_PEDAL",
    "USER_BRAKE_PRESSED",
    "NEW_SIGNAL_2",
    "GAS_PRESSED_INVERTED",
    "unsure1",
    "unsure2",
    "unsure3",
  ]}
  can_bus = 1 if car_fingerprint == CAR.NISSAN_ALTIMA else 2

  values["CANCEL_BUTTON"] = 1
  values["NO_BUTTON_PRESSED"] = 0
  values["PROPILOT_BUTTON"] = 0
  values["SET_BUTTON"] = 0
  values["RES_BUTTON"] = 0
  values["FOLLOW_DISTANCE_BUTTON"] = 0

  return packer.make_can_msg("CRUISE_THROTTLE", can_bus, values)


def create_cancel_msg(packer, cancel_msg, cruise_cancel):
  values = {s: cancel_msg[s] for s in [
    "CANCEL_SEATBELT",
    "NEW_SIGNAL_1",
    "NEW_SIGNAL_2",
    "NEW_SIGNAL_3",
  ]}

  if cruise_cancel:
    values["CANCEL_SEATBELT"] = 1

  return packer.make_can_msg("CANCEL_MSG", 2, values)


def create_lkas_hud_msg(packer, lkas_hud_msg, enabled, left_line, right_line, left_lane_depart, right_lane_depart):
  values = {s: lkas_hud_msg[s] for s in [
    "LARGE_WARNING_FLASHING",
    "SIDE_RADAR_ERROR_FLASHING1",
    "SIDE_RADAR_ERROR_FLASHING2",
    "LEAD_CAR",
    "LEAD_CAR_ERROR",
    "FRONT_RADAR_ERROR",
    "FRONT_RADAR_ERROR_FLASHING",
    "SIDE_RADAR_ERROR_FLASHING3",
    "LKAS_ERROR_FLASHING",
    "SAFETY_SHIELD_ACTIVE",
    "RIGHT_LANE_GREEN_FLASH",
    "LEFT_LANE_GREEN_FLASH",
    "FOLLOW_DISTANCE",
    "AUDIBLE_TONE",
    "SPEED_SET_ICON",
    "SMALL_STEERING_WHEEL_ICON",
    "unknown59",
    "unknown55",
    "unknown26",
    "unknown28",
    "unknown31",
    "SET_SPEED",
    "unknown43",
    "unknown08",
    "unknown05",
    "unknown02",
  ]}

  values["RIGHT_LANE_YELLOW_FLASH"] = 1 if right_lane_depart else 0
  values["LEFT_LANE_YELLOW_FLASH"] = 1 if left_lane_depart else 0

  values["LARGE_STEERING_WHEEL_ICON"] = 2 if enabled else 0
  values["RIGHT_LANE_GREEN"] = 1 if right_line and enabled else 0
  values["LEFT_LANE_GREEN"] = 1 if left_line and enabled else 0

  return packer.make_can_msg("PROPILOT_HUD", 0, values)


def create_lkas_hud_info_msg(packer, lkas_hud_info_msg, steer_hud_alert):
  values = {s: lkas_hud_info_msg[s] for s in [
    "NA_HIGH_ACCEL_TEMP",
    "SIDE_RADAR_NA_HIGH_CABIN_TEMP",
    "SIDE_RADAR_MALFUNCTION",
    "LKAS_MALFUNCTION",
    "FRONT_RADAR_MALFUNCTION",
    "SIDE_RADAR_NA_CLEAN_REAR_CAMERA",
    "NA_POOR_ROAD_CONDITIONS",
    "CURRENTLY_UNAVAILABLE",
    "SAFETY_SHIELD_OFF",
    "FRONT_COLLISION_NA_FRONT_RADAR_OBSTRUCTION",
    "PEDAL_MISSAPPLICATION_SYSTEM_ACTIVATED",
    "SIDE_IMPACT_NA_RADAR_OBSTRUCTION",
    "WARNING_DO_NOT_ENTER",
    "SIDE_IMPACT_SYSTEM_OFF",
    "SIDE_IMPACT_MALFUNCTION",
    "FRONT_COLLISION_MALFUNCTION",
    "SIDE_RADAR_MALFUNCTION2",
    "LKAS_MALFUNCTION2",
    "FRONT_RADAR_MALFUNCTION2",
    "PROPILOT_NA_MSGS",
    "BOTTOM_MSG",
    "HANDS_ON_WHEEL_WARNING",
    "WARNING_STEP_ON_BRAKE_NOW",
    "PROPILOT_NA_FRONT_CAMERA_OBSTRUCTED",
    "PROPILOT_NA_HIGH_CABIN_TEMP",
    "WARNING_PROPILOT_MALFUNCTION",
    "ACC_UNAVAILABLE_HIGH_CABIN_TEMP",
    "ACC_NA_FRONT_CAMERA_IMPARED",
    "unknown07",
    "unknown10",
    "unknown15",
    "unknown23",
    "unknown19",
    "unknown31",
    "unknown32",
    "unknown46",
    "unknown61",
    "unknown55",
    "unknown50",
  ]}

  if steer_hud_alert:
    values["HANDS_ON_WHEEL_WARNING"] = 1

  return packer.make_can_msg("PROPILOT_HUD_INFO_MSG", 0, values)
