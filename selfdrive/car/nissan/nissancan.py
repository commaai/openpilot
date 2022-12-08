import copy
import crcmod
from selfdrive.car.nissan.values import CAR

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
  values = copy.copy(cruise_throttle_msg)
  can_bus = 1 if car_fingerprint == CAR.ALTIMA else 2

  values["CANCEL_BUTTON"] = 1
  values["NO_BUTTON_PRESSED"] = 0
  values["PROPILOT_BUTTON"] = 0
  values["SET_BUTTON"] = 0
  values["RES_BUTTON"] = 0
  values["FOLLOW_DISTANCE_BUTTON"] = 0

  return packer.make_can_msg("CRUISE_THROTTLE", can_bus, values)


def create_cancel_msg(packer, cancel_msg, cruise_cancel):
  values = copy.copy(cancel_msg)

  if cruise_cancel:
    values["CANCEL_SEATBELT"] = 1

  return packer.make_can_msg("CANCEL_MSG", 2, values)


def create_lkas_hud_msg(packer, lkas_hud_msg, enabled, left_line, right_line, left_lane_depart, right_lane_depart):
  values = lkas_hud_msg

  values["RIGHT_LANE_YELLOW_FLASH"] = 1 if right_lane_depart else 0
  values["LEFT_LANE_YELLOW_FLASH"] = 1 if left_lane_depart else 0

  values["LARGE_STEERING_WHEEL_ICON"] = 2 if enabled else 0
  values["RIGHT_LANE_GREEN"] = 1 if right_line and enabled else 0
  values["LEFT_LANE_GREEN"] = 1 if left_line and enabled else 0

  return packer.make_can_msg("PROPILOT_HUD", 0, values)


def create_lkas_hud_info_msg(packer, lkas_hud_info_msg, steer_hud_alert):
  values = lkas_hud_info_msg

  if steer_hud_alert:
    values["HANDS_ON_WHEEL_WARNING"] = 1

  return packer.make_can_msg("PROPILOT_HUD_INFO_MSG", 0, values)
