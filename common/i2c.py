import os
import fcntl
import struct

# I2C constants from /usr/include/linux/i2c-dev.h
I2C_SLAVE = 0x0703
I2C_SLAVE_FORCE = 0x0706
I2C_SMBUS = 0x0720

# SMBus transfer types
I2C_SMBUS_READ = 1
I2C_SMBUS_WRITE = 0
I2C_SMBUS_BYTE_DATA = 2
I2C_SMBUS_I2C_BLOCK_DATA = 8

# Structure for SMBus ioctl
SMBUS_MSG_SIZE = 34


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

  def read_byte_data(self, addr: int, register: int, force: bool = False) -> int:
    self._set_address(addr, force)
    msg = struct.pack('BBBB', I2C_SMBUS_READ, register, I2C_SMBUS_BYTE_DATA, 0)
    buf = struct.pack('34B', *msg, *([0] * 30))
    fcntl.ioctl(self._fd, I2C_SMBUS, buf)
    return struct.unpack('34B', buf)[0]

  def write_byte_data(self, addr: int, register: int, value: int, force: bool = False) -> None:
    self._set_address(addr, force)
    msg = struct.pack('BBBB', I2C_SMBUS_WRITE, register, I2C_SMBUS_BYTE_DATA, value)
    buf = struct.pack('34B', *msg, *([0] * 30))
    fcntl.ioctl(self._fd, I2C_SMBUS, buf)

  def read_i2c_block_data(self, addr: int, register: int, length: int, force: bool = False) -> list[int]:
    self._set_address(addr, force)
    msg = struct.pack('BBBB', I2C_SMBUS_READ, register, I2C_SMBUS_I2C_BLOCK_DATA, length)
    buf = struct.pack('34B', *msg, *([0] * 30))
    fcntl.ioctl(self._fd, I2C_SMBUS, buf)
    return list(struct.unpack('34B', buf)[1 : length + 1])
