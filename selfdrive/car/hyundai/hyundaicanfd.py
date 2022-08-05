from selfdrive.car.hyundai.values import HyundaiFlags


def create_lkas(packer, CP, enabled, lat_active, apply_steer):
  values = {
    "LKA_MODE": 2,
    "LKA_ICON": 2 if enabled else 1,
    "TORQUE_REQUEST": apply_steer,
    "LKA_ASSIST": 0,
    "STEER_REQ": 1 if lat_active else 0,
    "STEER_MODE": 0,
    "SET_ME_1": 0,
    "NEW_SIGNAL_1": 0,
    "NEW_SIGNAL_2": 0,
  }

  msg = "LKAS" if CP.flags & HyundaiFlags.CANFD_HDA2 else "LFA"
  return packer.make_can_msg(msg, 4, values)

def create_cam_0x2a4(packer, camera_values):
  camera_values.update({
    "BYTE7": 0,
  })
  return packer.make_can_msg("CAM_0x2a4", 4, camera_values)

def create_buttons(packer, CP, cruise_buttons_copy, cnt, btn):
  values = cruise_buttons_copy
  values["COUNTER"] = cnt
  values["CRUISE_BUTTONS"] = btn
  if CP.flags & HyundaiFlags.CANFD_HDA2:
    bus = 5
    values["SET_ME_1"] = 1
  else:
    bus = 4

  return packer.make_can_msg("CRUISE_BUTTONS", bus, values)

def create_cruise_info(packer, cruise_info_copy, cruise_main, pcm_cancel_cmd, standstill_req, cnt):
  values = cruise_info_copy
  values["COUNTER"] = cnt
  values["CRUISE_STANDSTILL"] = not standstill_req
  if pcm_cancel_cmd and cruise_main:
    values["CRUISE_STATUS"] = 0
    values["CRUISE_INACTIVE"] = 1

  return packer.make_can_msg("CRUISE_INFO", 6, values)
