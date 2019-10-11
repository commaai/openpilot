import struct


# *** Toyota specific ***

def fix(msg, addr):
  checksum = 0
  idh = (addr & 0xff00) >> 8
  idl = (addr & 0xff)

  checksum = idh + idl + len(msg) + 1
  for d_byte in msg:
    checksum += d_byte

  #return msg + chr(checksum & 0xFF)
  return msg + struct.pack("B", checksum & 0xFF)


def make_can_msg(addr, dat, alt, cks=False):
  if cks:
    dat = fix(dat, addr)
  return [addr, 0, dat, alt]


def create_video_target(frame, addr):
  counter = frame & 0xff
  msg = struct.pack("!BBBBBBB", counter, 0x03, 0xff, 0x00, 0x00, 0x00, 0x00)
  return make_can_msg(addr, msg, 1, True)


def create_ipas_steer_command(packer, steer, enabled, apgs_enabled):
  """Creates a CAN message for the Toyota Steer Command."""
  if steer < 0:
    direction = 3
  elif steer > 0:
    direction = 1
  else:
    direction = 2

  mode = 3 if enabled else 1

  values = {
    "STATE": mode,
    "DIRECTION_CMD": direction,
    "ANGLE": steer,
    "SET_ME_X10": 0x10,
    "SET_ME_X40": 0x40
  }
  if apgs_enabled:
    return packer.make_can_msg("STEERING_IPAS", 0, values)
  else:
    return packer.make_can_msg("STEERING_IPAS_COMMA", 0, values)


def create_steer_command(packer, steer, steer_req, raw_cnt):
  """Creates a CAN message for the Toyota Steer Command."""

  values = {
    "STEER_REQUEST": steer_req,
    "STEER_TORQUE_CMD": steer,
    "COUNTER": raw_cnt,
    "SET_ME_1": 1,
  }
  return packer.make_can_msg("STEERING_LKA", 0, values)


def create_lta_steer_command(packer, steer, steer_req, raw_cnt, angle):
  """Creates a CAN message for the Toyota LTA Steer Command."""

  values = {
    "COUNTER": raw_cnt,
    "SETME_X3": 3,
    "PERCENTAGE" : 100,
    "SETME_X64": 0x64,
    "ANGLE": angle,
    "STEER_ANGLE_CMD": steer,
    "STEER_REQUEST": steer_req,
    "BIT": 0,
  }
  return packer.make_can_msg("STEERING_LTA", 0, values)


def create_accel_command(packer, accel, pcm_cancel, standstill_req, lead):
  # TODO: find the exact canceling bit that does not create a chime
  values = {
    "ACCEL_CMD": accel,
    "SET_ME_X01": 1,
    "DISTANCE": 0,
    "MINI_CAR": lead,
    "SET_ME_X3": 3,
    "SET_ME_1": 1,
    "RELEASE_STANDSTILL": not standstill_req,
    "CANCEL_REQ": pcm_cancel,
  }
  return packer.make_can_msg("ACC_CONTROL", 0, values)


def create_acc_cancel_command(packer):
  values = {
    "GAS_RELEASED": 0,
    "CRUISE_ACTIVE": 0,
    "STANDSTILL_ON": 0,
    "ACCEL_NET": 0,
    "CRUISE_STATE": 0,
    "CANCEL_REQ": 1,
  }
  return packer.make_can_msg("PCM_CRUISE", 0, values)


def create_fcw_command(packer, fcw):
  values = {
    "FCW": fcw,
    "SET_ME_X20": 0x20,
    "SET_ME_X10": 0x10,
    "SET_ME_X80": 0x80,
  }
  return packer.make_can_msg("ACC_HUD", 0, values)


def create_ui_command(packer, steer, chime, left_line, right_line, left_lane_depart, right_lane_depart):
  values = {
    "RIGHT_LINE": 3 if right_lane_depart else 1 if right_line else 2,
    "LEFT_LINE": 3 if left_lane_depart else 1 if left_line else 2,
    "BARRIERS" : 3 if left_lane_depart or right_lane_depart else 0,
    "SET_ME_X0C": 0x0c,
    "SET_ME_X2C": 0x2c,
    "SET_ME_X38": 0x38,
    "SET_ME_X02": 0x02,
    "SET_ME_X01": 1,
    "SET_ME_X01_2": 1,
    "REPEATED_BEEPS": 0,
    "TWO_BEEPS": chime,
    "LDA_ALERT": steer,
  }
  return packer.make_can_msg("LKAS_HUD", 0, values)
