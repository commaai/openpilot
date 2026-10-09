def byd_checksum(address: int, sig, d: bytearray) -> int:
  return (~sum(d[:7])) & 0xFF


def create_steering_control(packer, apply_angle: float, lat_active: bool, counter: int):
  # Stock saturates the rate limits at ±299 when engaged, 0 when disengaged
  rate_limit = 299 if lat_active else 0
  values = {
    "STEER_REQ": 1 if lat_active else 0,
    "STEER_REQ_ACTIVE_LOW": 0 if lat_active else 1,
    "STEER_ANGLE": apply_angle,
    "ANGLE_RATE_LIMIT_UPPER": rate_limit,
    "ANGLE_RATE_LIMIT_LOWER": -rate_limit,
    "E2E_ALIVE_1": 1,
    "E2E_ALIVE_2": 1,
    "SET_ME_FF": 0xFF,
    "SET_ME_F": 0xF,
    "COUNTER": counter,
  }
  return packer.make_can_msg("STEERING_MODULE_ADAS", 0, values)


def create_buttons(packer, cancel: bool):
  values = {
    "SET_ME_1_1": 1,
    "SET_ME_1_2": 1,
    "ACC_ON_BTN": 1 if cancel else 0,
  }
  return packer.make_can_msg("PCM_BUTTONS", 0, values)


def create_lkas_hud(packer, lat_active: bool, counter: int, stock_lkas_hud: dict, hud_control):
  values = {**stock_lkas_hud, "COUNTER": counter, "HANDS_ON_WHEEL_REQ": 0}
  if lat_active:
    values["LKS_MODE"] = 2 # green lane line icon
    values["LKAS_STATE"] = 2 # green steering wheel icon
    # LANE_STATE: 0=Grey, 1=Green, 2=Orange
    values["LEFT_LANE_STATE"] = 1 if hud_control.leftLaneVisible else 0
    values["RIGHT_LANE_STATE"] = 1 if hud_control.rightLaneVisible else 0

  if hud_control.leftLaneDepart:
    values["LEFT_LANE_STATE"] = 2
  if hud_control.rightLaneDepart:
    values["RIGHT_LANE_STATE"] = 2

  return packer.make_can_msg("LKAS_HUD_ADAS", 0, values)
