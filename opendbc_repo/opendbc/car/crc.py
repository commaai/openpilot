
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
CRC8BODY = _gen_crc8_table(0xD5)
CRC16_XMODEM = _gen_crc16_table(0x1021)


def mk_crc8_fun(table: list[int], init_crc: int = 0x00, xor_out: int = 0x00):
  init_reg = init_crc ^ xor_out

  def crc(data: bytes) -> int:
    crc = init_reg
    for b in data:
      crc = table[crc ^ b]
    return crc ^ xor_out
  return crc
