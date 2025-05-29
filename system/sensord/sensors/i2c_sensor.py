import smbus2

from cereal import log
from openpilot.system.sensord.sensors.gpio_ctypes import GPIOHandler, export_gpio, set_gpio_direction

class I2CSensor:
  def __init__(self, bus: int, gpio_nr: int = 0, shared_gpio: bool = False) -> None:
    self.bus = smbus2.SMBus(bus)
    self.gpio_nr = gpio_nr
    self.shared_gpio = shared_gpio
    self.gpio_handler: GPIOHandler | None = None

  def __del__(self):
    if self.gpio_handler is not None:
      self.gpio_handler.__exit__(None, None, None)
      self.gpio_handler = None
    self.bus.close()

  def twos_complement(self, val: int, bits: int) -> int:
    if (val & (1 << (bits - 1))) != 0:
        val = val - (1 << bits)
    return val

  @property
  def device_address(self) -> int:
    """Abstract property that must be implemented by subclasses"""
    raise NotImplementedError

  def read_register(self, register_address: int, length: int) -> bytes:
    return bytes(self.bus.read_i2c_block_data(self.device_address, register_address, length))

  def write_register(self, register_address: int, data: int) -> None:
    self.bus.write_byte_data(self.device_address, register_address, data)

  def init_gpio(self) -> None:
    """Initialize GPIO for sensor interrupt if needed"""
    if self.gpio_nr <= 0:
      return

    try:
      # Export and configure the GPIO
      export_gpio(self.gpio_nr)
      set_gpio_direction(self.gpio_nr, "in")

      # Create GPIO handler
      if self.gpio_handler is None:
        self.gpio_handler = GPIOHandler(self.gpio_nr)
    except Exception as e:
      print(f"Failed to initialize GPIO {self.gpio_nr}: {e}")
      self.gpio_handler = None

  def has_interrupt_occurred(self) -> bool:
    if self.gpio_nr <= 0 or self.gpio_handler is None:
      return False
    try:
      return bool(self.gpio_handler.get_value())
    except Exception as e:
      print(f"Error reading GPIO {self.gpio_nr}: {e}")
      return False

  def verify_chip_id(self, address: int, expected_ids: list[int]) -> int:
    chip_id = self.read_register(address, 1)[0]
    assert chip_id in expected_ids
    return chip_id

  # Abstract methods that must be implemented by subclasses
  def init(self) -> bool:
    raise NotImplementedError

  def get_event(self, ts: int | None = None) -> log.SensorEventData:
    raise NotImplementedError

  def shutdown(self) -> None:
    raise NotImplementedError

  def is_data_valid(self, ts: int) -> bool:
    return True
