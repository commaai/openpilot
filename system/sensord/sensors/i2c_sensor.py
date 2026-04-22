import os
import time
import ctypes
import select
import threading
from collections.abc import Iterable

import cereal.messaging as messaging
from cereal import log
from cereal.services import SERVICE_LIST
from openpilot.common.gpio import get_irqs_for_action, gpiochip_get_ro_value_fd, gpioevent_data
from openpilot.common.i2c import SMBus
from openpilot.common.realtime import Ratekeeper
from openpilot.common.swaglog import cloudlog
from openpilot.common.utils import sudo_write


class Sensor:
  _bus_locks: dict[int, threading.RLock] = {}
  _bus_locks_lock = threading.Lock()

  class SensorException(Exception):
    pass

  class DataNotReady(SensorException):
    pass

  def __init__(self, bus: int) -> None:
    self.bus_num = bus
    self.bus = SMBus(bus)
    self._bus_lock = self._get_bus_lock(bus)
    self._irq_fd = -1
    self.source = log.SensorEventData.SensorSource.velodyne  # unknown
    self.start_ts = 0.

  def __del__(self):
    self.close()

  @classmethod
  def _get_bus_lock(cls, bus: int) -> threading.RLock:
    with cls._bus_locks_lock:
      lock = cls._bus_locks.get(bus)
      if lock is None:
        lock = threading.RLock()
        cls._bus_locks[bus] = lock
      return lock

  def close(self) -> None:
    self._close_irq_fd()
    self.bus.close()

  def read(self, addr: int, length: int) -> bytes:
    with self._bus_lock:
      return bytes(self.bus.read_i2c_block_data(self.device_address, addr, length))

  def write(self, addr: int, data: int) -> None:
    with self._bus_lock:
      self.bus.write_byte_data(self.device_address, addr, data)

  def writes(self, writes: Iterable[tuple[int, int]]) -> None:
    with self._bus_lock:
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

  def run(self, event: threading.Event) -> None:
    try:
      if self.irq_pin is None:
        self._polling_loop(event)
      else:
        self._interrupt_loop(event)
    except Exception:
      cloudlog.exception(f"Error in {self.service} worker")
    finally:
      self._close_irq_fd()

  def _polling_loop(self, event: threading.Event) -> None:
    pm = messaging.PubMaster([self.service])
    rk = Ratekeeper(SERVICE_LIST[self.service].frequency, print_delay_threshold=None)
    while not event.is_set():
      try:
        self._publish_event(pm)
      except Exception:
        cloudlog.exception(f"Error in {self.service} polling loop")
      rk.keep_time()

  def _interrupt_loop(self, event: threading.Event) -> None:
    assert self.irq_pin is not None

    pm = messaging.PubMaster([self.service])
    fd = self._open_irq_fd()
    offset = time.time_ns() - time.monotonic_ns()

    poller = select.poll()
    poller.register(fd, select.POLLIN | select.POLLPRI)
    while not event.is_set():
      events = poller.poll(100)
      if not events:
        cloudlog.error(f"{self.service} poll timed out")
        continue
      if not (events[0][1] & (select.POLLIN | select.POLLPRI)):
        cloudlog.error(f"{self.service} no poll events set")
        continue

      dat = os.read(fd, ctypes.sizeof(gpioevent_data) * 16)
      evd = gpioevent_data.from_buffer_copy(dat)

      cur_offset = time.time_ns() - time.monotonic_ns()
      if abs(cur_offset - offset) > 10 * 1e6:  # ms
        cloudlog.warning(f"{self.service} time jumped: {cur_offset} {offset}")
        offset = cur_offset
        continue

      try:
        self._publish_event(pm, evd.timestamp - cur_offset)
      except self.DataNotReady:
        pass
      except Exception:
        cloudlog.exception(f"Error processing {self.service}")

  def _publish_event(self, pm: messaging.PubMaster, ts: int | None = None) -> None:
    evt = self.get_event(ts)
    if not self.is_data_valid():
      return
    msg = messaging.new_message(self.service, valid=True)
    setattr(msg, self.service, evt)
    pm.send(self.service, msg)

  def _open_irq_fd(self) -> int:
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
