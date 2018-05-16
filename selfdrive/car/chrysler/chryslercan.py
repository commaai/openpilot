import struct


# *** Chrysler specific ***

def calc_checksum(data):
  """This version does not want checksum byte in input data.

  jeep chrysler canbus checksum from http://illmatics.com/Remote%20Car%20Hacking.pdf
  """
  end_index = len(data)
  index = 0
  checksum = 0xFF
  temp_chk = 0;
  bit_sum = 0;
  if(end_index <= index):
    return False
  for index in range(0, end_index):
    shift = 0x80
    curr = data[index]
    iterate = 8
    while(iterate > 0):
      iterate -= 1
      bit_sum = curr & shift;
      temp_chk = checksum & 0x80
      if (bit_sum != 0):
        bit_sum = 0x1C
        if (temp_chk != 0):
          bit_sum = 1
        checksum = checksum << 1
        temp_chk = checksum | 1
        bit_sum ^= temp_chk
      else:
        if (temp_chk != 0):
          bit_sum = 0x1D
        checksum = checksum << 1
        bit_sum ^= checksum
      checksum = bit_sum
      shift = shift >> 1
  return ~checksum & 0xFF


def make_can_msg(addr, dat, alt, cks=False):
  if cks:
    dat = dat + struct.pack("B", calc_checksum(dat))
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


def create_steer_command(packer, steer, raw_cnt):
  """Creates a CAN message for the LKAS Steer Command."""

  values = {
    # TODO add the high nibble set to one and anything else.
    "LKAS_STEERING_TORQUE_MAYBE": steer,  # TODO verify units
    "LKAS_INCREMENTING": raw_cnt,
  }
  return packer.make_can_msg("LKAS_INDICATOR_2", 0, values)
