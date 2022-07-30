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
    "BYTE3": 0,
    "BYTE4": 0,
    "BYTE5": 0,
    "BYTE6": 0,
    "BYTE7": 0,
    "BYTE8": 0,
    "BYTE9": 0,
    "BYTE10": 0,
    "BYTE11": 0,
    "BYTE12": 0,
    "BYTE13": 0,
    "BYTE14": 0,
    "BYTE15": 0,
    "BYTE16": 0,
    "BYTE17": 0,
    "BYTE18": 0,
    "BYTE19": 0,
    "BYTE20": 0,
    "BYTE21": 0,
    "BYTE22": 0,
    "BYTE23": 0,
  })
  return packer.make_can_msg("CAM_0x2a4", 4, camera_values)

def create_buttons(packer, cnt, btn):
  values = {
    "COUNTER": cnt,
    "SET_ME_1": 1,
    "CRUISE_BUTTONS": btn,
  }
  return packer.make_can_msg("CRUISE_BUTTONS", 5, values)
