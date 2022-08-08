def create_lkas(packer, enabled, lat_active, apply_steer):
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
  return packer.make_can_msg("LKAS", 4, values)

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
