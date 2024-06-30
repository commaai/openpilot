from openpilot.selfdrive.car import CanBusBase

def crc8(data: bytes, poly: int = 0x1D, init_val: int = 0x00, xor_out: int = 0x2D):
    crc = init_val
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = (crc << 1) ^ poly
            else:
                crc <<= 1
            crc &= 0xFF
    return crc ^ xor_out


def gwm_crc(message):
    output_int = crc8(message)
    return output_int

class GwmCan(CanBusBase):
  def __init__(self, CP=None, fingerprint=None) -> None:
    # use fingerprint if specified
    super().__init__(CP if fingerprint is None else None, fingerprint)
