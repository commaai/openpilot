def create_lkas(packer, enabled, frame, lat_active, apply_steer):
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
  return packer.make_can_msg("LKAS", 4, values, frame % 255)


def create_buttons(packer, cnt, cancel, resume):
  values = {
    "_COUNTER": cnt % 0xf,
    "SET_ME_1": 1,
    "DISTANCE_BTN": 1 if resume else 0,
    "PAUSE_RESUME_BTN": 1 if cancel else 0,
  }
  return packer.make_can_msg("CRUISE_BUTTONS", 5, values)
