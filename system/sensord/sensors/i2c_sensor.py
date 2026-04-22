import os
import time
import ctypes
from collections.abc import Iterable

from cereal import log
from openpilot.common.gpio import get_irqs_for_action, gpiochip_get_ro_value_fd
from openpilot.common.i2c import SMBus
from openpilot.common.swaglog import cloudlog
from openpilot.common.utils import sudo_write


class Sensor:
  class SensorException(Exception):
    pass

  class DataNotReady(SensorException):
    pass

  def __init__(self, bus: int) -> None:
    self.bus = SMBus(bus)
    self._irq_fd = -1
    self.source = log.SensorEventData.SensorSource.velodyne  # unknown
    self.start_ts = 0.

  def __del__(self):
    self.close()

  def close(self) -> None:
    self._close_irq_fd()
    self.bus.close()

  def read(self, addr: int, length: int) -> bytes:
    return bytes(self.bus.read_i2c_block_data(self.device_address, addr, length))

  def write(self, addr: int, data: int) -> None:
    self.bus.write_byte_data(self.device_address, addr, data)

  def writes(self, writes: Iterable[tuple[int, int]]) -> None:
    for addr, data in writes:
      self.bus.write_byte_data(self.device_address, addr, data)

  def verify_chip_id(self, address: int, expected_ids: list[int]) -> int:
    chip_id = self.read(address, 1)[0]
    assert chip_id in expected_ids
    return chip_id

  # Abstract methods that must be implemented by subclasses
  @property
  def device_address(self) -> int:
    raise NotImplementedError

  @property
  def service(self) -> str:
    raise NotImplementedError

  @property
  def irq_pin(self) -> int | None:
    return None

  @property
  def irq_gpiochip(self) -> int:
    return 0

  @property
  def irq_action(self) -> str:
    return "sensord"

  @property
  def irq_affinity(self) -> str:
    return "1"

  def reset(self) -> None:
    # optional.
    # not part of init due to shared registers
    pass

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

  def open_irq_fd(self) -> int:
    if self._irq_fd < 0:
      assert self.irq_pin is not None
      # Request both edges as the data ready pulse is short and may only be
      # detected as one edge depending on timing.
      self._irq_fd = gpiochip_get_ro_value_fd(self.irq_action, self.irq_gpiochip, self.irq_pin)
      self._configure_irq_affinity()
    return self._irq_fd

  def _close_irq_fd(self) -> None:
    if self._irq_fd >= 0:
      os.close(self._irq_fd)
      self._irq_fd = -1

  def _configure_irq_affinity(self) -> None:
    irqs = get_irqs_for_action(self.irq_action)
    if len(irqs) == 0:
      cloudlog.warning(f"No IRQs found for '{self.irq_action}'")
      return

    for irq in irqs:
      try:
        sudo_write(self.irq_affinity, f"/proc/irq/{irq}/smp_affinity_list")
      except Exception:
        cloudlog.exception(f"Error setting affinity for IRQ {irq}")

  @staticmethod
  def parse_16bit(lsb: int, msb: int) -> int:
    return ctypes.c_int16((msb << 8) | lsb).value

  @staticmethod
  def parse_20bit(b2: int, b1: int, b0: int) -> int:
    combined = ctypes.c_uint32((b0 << 16) | (b1 << 8) | b2).value
    return ctypes.c_int32(combined).value // (1 << 4)
