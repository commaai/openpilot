def create_control(packer, torque_l, torque_r):
  values = {
    "TORQUE_L": torque_l,
    "TORQUE_R": torque_r,
  }

  return packer.make_can_msg("TORQUE_CMD", 0, values)


def body_checksum(address: int, sig, d: bytearray) -> int:
  crc = 0xFF
  poly = 0xD5
  for i in range(len(d) - 2, -1, -1):
    crc ^= d[i]
    for _ in range(8):
      if crc & 0x80:
        crc = ((crc << 1) ^ poly) & 0xFF
      else:
        crc = (crc << 1) & 0xFF
  return crc
