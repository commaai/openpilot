from selfdrive.car.hyundai.values import CAR

def get_btn_bus(car_fingerprint):
  return 4 if car_fingerprint in (CAR.GENESIS_GV70) else 5

def create_lkas(packer, enabled, lat_active, apply_steer):
  values = create_lkas_values(enabled, lat_active, apply_steer)
  return packer.make_can_msg("LKAS", 4, values)

def create_lfa(packer, enabled, lat_active, apply_steer):
  values = create_lkas_values(enabled, lat_active, apply_steer)
  return packer.make_can_msg("LFA", 4, values)

def create_lkas_values(enabled, lat_active, apply_steer):
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
  return values

def create_lfa_icon(packer, enabled):
  values = {
    "HDA": 0,
    "LFA_GREY": 0,
    "LFA_YELLOW": 0,
    #Will turn on HDA when cruise enabled
    #"HDA": 1 if enabled else 0,
    #"LFA_GREY": 0 if enabled else 1,
    #"LFA_YELLOW": 1 if enabled else 0,
  }
  return packer.make_can_msg("LFA_ICONS", 4, values)

def create_cam_0x2a4(packer, camera_values):
  camera_values.update({
    "BYTE7": 0,
  })
  return packer.make_can_msg("CAM_0x2a4", 4, camera_values)

def create_buttons(packer, cnt, btn, car_fingerprint):
  counter = "COUNTER"
  if car_fingerprint in (CAR.GENESIS_GV70):
    counter="_COUNTER"

  values = {
    counter: cnt,
    "SET_ME_1": 1,
    "CRUISE_BUTTONS": btn,
  }
  return packer.make_can_msg("CRUISE_BUTTONS", get_btn_bus(car_fingerprint), values)