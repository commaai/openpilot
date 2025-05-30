import time
import smbus2
import ctypes
from collections.abc import Iterable

from cereal import log

class Sensor:
  class SensorException(Exception):
    pass

  class DataNotReady(SensorException):
    pass

  def __init__(self, bus: int) -> None:
    self.bus = smbus2.SMBus(bus)
    self.source = log.SensorEventData.SensorSource.velodyne  # unknown
    self.start_ts = 0.

  def __del__(self):
    self.bus.close()

  def read(self, addr: int, length: int) -> bytes:
    return bytes(self.bus.read_i2c_block_data(self.device_address, addr, length))

  def write(self, addr: int, data: int) -> None:
    self.bus.write_byte_data(self.device_address, addr, data)

  def writes(self, writes: Iterable[tuple[int, int]]) -> None:
    for addr, data in writes:
      self.write(addr, data)

  def verify_chip_id(self, address: int, expected_ids: list[int]) -> int:
    chip_id = self.read(address, 1)[0]
    assert chip_id in expected_ids
    return chip_id

  # Abstract methods that must be implemented by subclasses
  @property
  def device_address(self) -> int:
    raise NotImplementedError

  def init(self) -> None:
    raise NotImplementedError

  def get_event(self, ts: int | None = None) -> log.SensorEventData:
    raise NotImplementedError

  def shutdown(self) -> None:
    raise NotImplementedError

  def is_data_valid(self) -> bool:
    if self.start_ts == 0:
      self.start_ts = time.monotonic()

    # unclear whether we need this...
    return (time.monotonic() - self.start_ts) > 0.5

  # *** helpers ***
  @staticmethod
  def wait():
    # a standard small sleep
    time.sleep(0.005)

  @staticmethod
  def parse_16bit(lsb: int, msb: int) -> int:
    return ctypes.c_int16((msb << 8) | lsb).value

  @staticmethod
  def parse_20bit(b2: int, b1: int, b0: int) -> int:
    combined = ctypes.c_uint32((b0 << 16) | (b1 << 8) | b2).value
    return ctypes.c_int32(combined).value // (1 << 4)
