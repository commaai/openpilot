import os
import fcntl
import ctypes

# I2C constants from /usr/include/linux/i2c-dev.h
I2C_SLAVE = 0x0703
I2C_SLAVE_FORCE = 0x0706
I2C_SMBUS = 0x0720

# SMBus transfer types
I2C_SMBUS_READ = 1
I2C_SMBUS_WRITE = 0
I2C_SMBUS_BYTE_DATA = 2
I2C_SMBUS_I2C_BLOCK_DATA = 8

I2C_SMBUS_BLOCK_MAX = 32


class _I2cSmbusData(ctypes.Union):
  _fields_ = [
    ("byte", ctypes.c_uint8),
    ("word", ctypes.c_uint16),
    ("block", ctypes.c_uint8 * (I2C_SMBUS_BLOCK_MAX + 2)),
  ]


class _I2cSmbusIoctlData(ctypes.Structure):
  _fields_ = [
    ("read_write", ctypes.c_uint8),
    ("command", ctypes.c_uint8),
    ("size", ctypes.c_uint32),
    ("data", ctypes.POINTER(_I2cSmbusData)),
  ]


class SMBus:
  def __init__(self, bus: int):
    self._fd = os.open(f'/dev/i2c-{bus}', os.O_RDWR)

  def __enter__(self) -> 'SMBus':
    return self

  def __exit__(self, *args) -> None:
    self.close()

  def close(self) -> None:
    if hasattr(self, '_fd') and self._fd >= 0:
      os.close(self._fd)
      self._fd = -1

  def _set_address(self, addr: int, force: bool = False) -> None:
    ioctl_arg = I2C_SLAVE_FORCE if force else I2C_SLAVE
    fcntl.ioctl(self._fd, ioctl_arg, addr)

  def _smbus_access(self, read_write: int, command: int, size: int, data: _I2cSmbusData) -> None:
    ioctl_data = _I2cSmbusIoctlData(read_write, command, size, ctypes.pointer(data))
    fcntl.ioctl(self._fd, I2C_SMBUS, ioctl_data)

  def read_byte_data(self, addr: int, register: int, force: bool = False) -> int:
    self._set_address(addr, force)
    data = _I2cSmbusData()
    self._smbus_access(I2C_SMBUS_READ, register, I2C_SMBUS_BYTE_DATA, data)
    return int(data.byte)

  def write_byte_data(self, addr: int, register: int, value: int, force: bool = False) -> None:
    self._set_address(addr, force)
    data = _I2cSmbusData()
    data.byte = value & 0xFF
    self._smbus_access(I2C_SMBUS_WRITE, register, I2C_SMBUS_BYTE_DATA, data)

  def read_i2c_block_data(self, addr: int, register: int, length: int, force: bool = False) -> list[int]:
    self._set_address(addr, force)
    if not (0 <= length <= I2C_SMBUS_BLOCK_MAX):
      raise ValueError(f"length must be 0..{I2C_SMBUS_BLOCK_MAX}")

    data = _I2cSmbusData()
    data.block[0] = length
    self._smbus_access(I2C_SMBUS_READ, register, I2C_SMBUS_I2C_BLOCK_DATA, data)
    read_len = int(data.block[0]) or length
    read_len = min(read_len, length)
    return [int(b) for b in data.block[1 : read_len + 1]]
