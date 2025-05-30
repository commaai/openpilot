import time
import smbus2

from cereal import log

class I2CSensor:
  def __init__(self, bus: int) -> None:
    self.bus = smbus2.SMBus(bus)
    self.source = log.SensorEventData.SensorSource.velodyne  # unknown
    self.start_ts = 0

  def __del__(self):
    self.bus.close()

  def read(self, register_address: int, length: int) -> bytes:
    return bytes(self.bus.read_i2c_block_data(self.device_address, register_address, length))

  def write(self, register_address: int, data: int) -> None:
    self.bus.write_byte_data(self.device_address, register_address, data)

  def writes(self, writes: tuple[int, int]) -> None:
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
    return (msb << 8) | lsb

  @staticmethod
  def parse_20bit(b2: int, b1: int, b0: int) -> int:
    combined = (b0 << 16) | (b1 << 8) | b2
    return combined >> 4
