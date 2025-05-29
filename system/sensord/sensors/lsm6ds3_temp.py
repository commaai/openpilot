import time

from cereal import log
from openpilot.system.sensord.sensors.i2c_sensor import I2CSensor

# https://content.arduino.cc/assets/st_imu_lsm6ds3_datasheet.pdf
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
    scale = {
      log.SensorEventData.SensorSource.lsm6ds3: 16.0,
      log.SensorEventData.SensorSource.lsm6ds3trc: 256.0,
    }.get(self.source)
    data = self.read(0x20, 2)
    return 25 + (self.parse_16bit(data[0], data[1]) / scale)

  def init(self):
    chip_id = self.verify_chip_id(0x0F, [0x69, 0x6A])
    if chip_id == 0x6A:
      self.source = log.SensorEventData.SensorSource.lsm6ds3trc

  def get_event(self, ts: int | None = None) -> log.SensorEventData:
    event = log.SensorEventData.new_message()
    event.version = 1
    event.timestamp = int(time.monotonic() * 1e9)
    event.source = self.source
    event.temperature = self._read_temperature()
    return event

  def shutdown(self) -> None:
    pass

if __name__ == "__main__":
  s = LSM6DS3_Temp(1)
  s.init()
  print(s.get_event())
  s.shutdown()
