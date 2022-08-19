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

def create_buttons(packer, cnt, btn):
  values = {
    "COUNTER": cnt,
    "SET_ME_1": 1,
    "CRUISE_BUTTONS": btn,
  }
  return packer.make_can_msg("CRUISE_BUTTONS", 5, values)

def create_cruise_info(packer, cruise_info_copy, cancel):
  values = cruise_info_copy
  if cancel:
    values["CRUISE_STATUS"] = 0
    values["CRUISE_INACTIVE"] = 1
  return packer.make_can_msg("CRUISE_INFO", 4, values)

def create_lfahda_cluster(packer, enabled):
  values = {
    "HDA_ICON": 1 if enabled else 0,
    "LFA_ICON": 2 if enabled else 0,
  }
  return packer.make_can_msg("LFAHDA_CLUSTER", 4, values)
