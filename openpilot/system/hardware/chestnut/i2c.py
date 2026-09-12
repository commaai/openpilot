import struct
import time


class ChestnutI2C:
  """INA reads through the ASM register interface; requires exclusive I2C ownership."""

  def __init__(self, handle):
    self.handle = handle
    self.register = None
    self.status = None
    self.expected = None

  def read(self, address, length=1):
    data = bytes(self.handle.controlRead(0xC0, 0xE4, address, 0, length, timeout=100))
    if len(data) != length:
      raise OSError('short chestnut register read')
    return data

  def write(self, address, value):
    self.handle.controlWrite(0x40, 0xE5, address, value, b'', timeout=100)

  def transfer(self, expected):
    self.expected = expected
    self.status = None
    self.write(0xC875, 0xFF)
    self.write(0xC875, 1)
    deadline = time.monotonic() + 0.05
    try:
      while True:
        status = self.read(0xC875)[0]
        self.status = status
        if status & 0x40:
          if status != expected:
            raise OSError(f'chestnut I2C status {status:#x}, expected {expected:#x}')
          return
        if time.monotonic() >= deadline:
          raise TimeoutError('chestnut I2C transfer timed out')
        time.sleep(0.001)
    finally:
      self.write(0xC875, 0x40)

  def read_u16(self, register, address=0x45):
    self.register = register
    for reg, value in ((0xC870, address << 1), (0xC871, register), (0xC873, 0), (0xC874, 1),
                       (0xC878, 0), (0xC879, 0), (0xC87C, 0), (0xC87D, 0)):
      self.write(reg, value)
    self.transfer(0x42)
    for reg, value in ((0xC870, address << 1 | 1), (0xC871, 0), (0xC874, 2)):
      self.write(reg, value)
    dma_enable = self.read(0xC805)[0]
    try:
      self.write(0xC805, dma_enable | 0x10)
      self.transfer(0x48)
      return struct.unpack('>H', self.read(0xE800, 2))[0]
    finally:
      self.write(0xC805, dma_enable & ~0x10)
