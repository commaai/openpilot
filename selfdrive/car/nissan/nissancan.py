import copy
import crcmod
from selfdrive.car.nissan.values import CAR

nissan_checksum = crcmod.mkCrcFun(0x11d, initCrc=0x00, rev=False, xorOut=0xff)


def create_steering_control(packer, car_fingerprint, apply_steer, frame, steer_on, lkas_max_torque):
  if car_fingerprint == CAR.XTRAIL:
    idx = (frame % 16)
    values = {
      "DESIRED_ANGLE": apply_steer,
      "SET_0x80_2": 0x80,
      "SET_0x80": 0x80,
      "MAX_TORQUE": lkas_max_torque if steer_on else 0,
      "COUNTER": idx,
      "LKA_ACTIVE": steer_on,
    }

    dat = packer.make_can_msg("LKAS", 0, values)[2]

    values["CRC"] = nissan_checksum(dat[:7])

  return packer.make_can_msg("LKAS", 0, values)


def create_acc_cancel_cmd(packer, cruise_throttle_msg, frame):
  values = copy.copy(cruise_throttle_msg)

  values["CANCEL_BUTTON"] = 1
  values["NO_BUTTON_PRESSED"] = 0
  values["PROPILOT_BUTTON"] = 0
  values["SET_BUTTON"] = 0
  values["RES_BUTTON"] = 0
  values["FOLLOW_DISTANCE_BUTTON"] = 0
  values["COUNTER"] = (frame % 4)
  
  return packer.make_can_msg("CruiseThrottle", 2, values)
