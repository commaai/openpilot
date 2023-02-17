import time
import struct
from functools import reduce

from .constants import McuType
from .spi import SpiDevice

SYNC = 0x5A
ACK = 0x79
NACK = 0x1F

# https://www.st.com/resource/en/application_note/an4286-spi-protocol-used-in-the-stm32-bootloader-stmicroelectronics.pdf
class PandaSpiDFU:
  def __init__(self, dfu_serial):
    self.dev = SpiDevice(speed=1000000)

    # say hello
    with self.dev.acquire() as spi:
      try:
        spi.xfer([SYNC, ])
        self._get_ack(spi)
      except Exception:
        raise Exception("failed to connect to panda")  # pylint: disable=W0707

    self._mcu_type = self.get_mcu_type()

  def _get_ack(self, spi, timeout=1.0):
    data = 0x00
    start_time = time.monotonic()
    while data not in (ACK, NACK) and (time.monotonic() - start_time < timeout):
      data = spi.xfer([0x00, ])[0]
      time.sleep(0.001)
    spi.xfer([ACK, ])

    if data == NACK:
      raise Exception("Got NACK response")
    elif data != ACK:
      raise Exception("Missing ACK")

  def _cmd(self, cmd, data=None, read_bytes=0) -> bytes:
    ret = b""
    with self.dev.acquire() as spi:
      # sync
      spi.xfer([SYNC, ])

      # send command
      spi.xfer([cmd, cmd ^ 0xFF])
      self._get_ack(spi)

      # send data
      if data is not None:
        for d in data:
          spi.xfer(self.add_checksum(d))
          self._get_ack(spi, timeout=20)

      # receive
      if read_bytes > 0:
        # send busy byte
        ret = spi.xfer([0x00, ]*(read_bytes + 1))[1:]
        self._get_ack(spi)

    return ret

  def add_checksum(self, data):
    return data + bytes([reduce(lambda a, b: a ^ b, data)])

  # ***** ST Bootloader functions *****

  def get_bootloader_version(self) -> int:
    return self._cmd(0x01, read_bytes=1)[0]

  def get_id(self) -> int:
    ret = self._cmd(0x02, read_bytes=3)
    assert ret[0] == 1
    return ((ret[1] << 8) + ret[2])

  def go_cmd(self, address: int) -> None:
    self._cmd(0x21, data=[struct.pack('>I', address), ])

  def erase(self, address: int) -> None:
    d = struct.pack('>H', address)
    self._cmd(0x44, data=[d, ])

  # ***** panda api *****

  def get_mcu_type(self) -> McuType:
    mcu_by_id = {mcu.config.mcu_idcode: mcu for mcu in McuType}
    return mcu_by_id[self.get_id()]

  def global_erase(self):
    self.erase(0xFFFF)

  def program_file(self, address, fn):
    with open(fn, 'rb') as f:
      code = f.read()

    i = 0
    while i < len(code):
      #print(i, len(code))
      block = code[i:i+256]
      if len(block) < 256:
        block += b'\xFF' * (256 - len(block))

      self._cmd(0x31, data=[
        struct.pack('>I', address + i),
        bytes([len(block) - 1]) + block,
      ])
      #print(f"Written {len(block)} bytes to {hex(address + i)}")
      i += 256

  def program_bootstub(self):
    self.program_file(self._mcu_type.config.bootstub_address, self._mcu_type.config.bootstub_path)

  def program_app(self):
    self.program_file(self._mcu_type.config.app_address, self._mcu_type.config.app_path)

  def reset(self):
    self.go_cmd(self._mcu_type.config.bootstub_address)
