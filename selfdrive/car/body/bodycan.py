from selfdrive.car import crc8_pedal


def create_control(packer, torque_l, torque_r, idx):
  can_bus = 0

  values = {
    "TORQUE_L": torque_l,
    "TORQUE_R": torque_r,
    "COUNTER": idx & 0xF,
  }

  dat = packer.make_can_msg("TORQUE_CMD", 0, values)[2]

  checksum = crc8_pedal(dat[:-1])
  values["CHECKSUM"] = checksum

  return packer.make_can_msg("TORQUE_CMD", can_bus, values)
