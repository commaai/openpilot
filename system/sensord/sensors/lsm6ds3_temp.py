import time
from cereal import log
from openpilot.common.swaglog import cloudlog
from openpilot.system.sensord.sensors.i2c_sensor import I2CSensor

# Register addresses
LSM6DS3_TEMP_I2C_REG_ID = 0x0F
LSM6DS3_TEMP_I2C_REG_CTRL1_XL = 0x10
LSM6DS3_TEMP_I2C_REG_CTRL3_C = 0x12
LSM6DS3_TEMP_I2C_REG_OUT_TEMP_L = 0x20

# CTRL1_XL configuration
LSM6DS3_TEMP_ODR_52HZ = 0x40  # 52Hz output data rate

# CTRL3_C configuration
LSM6DS3_TEMP_IF_INC = 0x04  # Enable register address auto-increment

class LSM6DS3_Temp(I2CSensor):
  def __init__(self, bus: int, gpio_nr: int = 0, shared_gpio: bool = False):
    super().__init__(bus, gpio_nr, shared_gpio)
    self.source = log.SensorEventData.SensorSource.lsm6ds3
    self.scale = 256.0  # 16-bit resolution, 256 LSB/°C
    self.offset = 25.0  # 0°C = 25.0°C

  @property
  def device_address(self) -> int:
    return 0x6A  # Default I2C address for LSM6DS3

  def _read_temperature(self) -> float:
    data = self.read(LSM6DS3_TEMP_I2C_REG_OUT_TEMP_L, 2)
    temp_raw = self.twos_complement((data[1] << 8) | data[0], 16)
    return temp_raw / self.scale + self.offset

  def init(self) -> bool:
    # Verify chip ID
    chip_id = self.verify_chip_id(LSM6DS3_TEMP_I2C_REG_ID, [0x69, 0x6A])
    if chip_id < 0:
      return False

    # Configure sensor
    try:
      # Enable auto-increment
      self.write(LSM6DS3_TEMP_I2C_REG_CTRL3_C, LSM6DS3_TEMP_IF_INC)

      # Configure accelerometer (required for temperature sensor)
      self.write(LSM6DS3_TEMP_I2C_REG_CTRL1_XL, LSM6DS3_TEMP_ODR_52HZ)

      # Wait for first sample
      time.sleep(0.1)

      # Read temperature to initialize
      _ = self._read_temperature()

      return True

    except Exception:
      cloudlog.exception("Error initializing LSM6DS3 temperature sensor")
      return False

  def get_event(self, ts: int | None = None) -> log.SensorEventData:
    if ts is None:
      ts = int(time.monotonic() * 1e9)

    event = log.SensorEventData.new_message()
    event.timestamp = ts
    event.source = self.source
    event.temperature = self._read_temperature()

    return event

  def shutdown(self) -> None:
    try:
      self.write(LSM6DS3_TEMP_I2C_REG_CTRL1_XL, 0x00)
    except Exception:
      cloudlog.exception("Error shutting down LSM6DS3 temperature sensor")

if __name__ == "__main__":
  s = LSM6DS3_Temp(1)
  s.init()
  print(s.get_event())
  s.shutdown()
