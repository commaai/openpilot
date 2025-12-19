
def _gen_crc8_table(poly: int) -> list[int]:
  table = []
  for i in range(256):
    crc = i
    for _ in range(8):
      if crc & 0x80:
        crc = ((crc << 1) ^ poly) & 0xFF
      else:
        crc = (crc << 1) & 0xFF
    table.append(crc)
  return table


def _gen_crc16_table(poly: int) -> list[int]:
  table = []
  for i in range(256):
    crc = i << 8
    for _ in range(8):
      if crc & 0x8000:
        crc = ((crc << 1) ^ poly) & 0xFFFF
      else:
        crc = (crc << 1) & 0xFFFF
    table.append(crc)
  return table


CRC8H2F = _gen_crc8_table(0x2F)
CRC8J1850 = _gen_crc8_table(0x1D)
CRC16_XMODEM = _gen_crc16_table(0x1021)
