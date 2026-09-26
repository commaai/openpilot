from opendbc.car.crc import CRC8BODY


def create_control(packer, torque_l, torque_r):
  values = {
    "TORQUE_L": torque_l,
    "TORQUE_R": torque_r,
  }

  return packer.make_can_msg("TORQUE_CMD", 0, values)


def body_checksum(address: int, sig, d: bytearray) -> int:
  crc = 0xFF
  for i in range(len(d) - 2, -1, -1):
    crc = CRC8BODY[crc ^ d[i]]
  return crc
