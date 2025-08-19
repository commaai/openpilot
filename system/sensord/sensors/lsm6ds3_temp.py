import time

from cereal import log
from openpilot.system.sensord.sensors.i2c_sensor import Sensor

# https://content.arduino.cc/assets/st_imu_lsm6ds3_datasheet.pdf
class LSM6DS3_Temp(Sensor):
  @property
  def device_address(self) -> int:
    return 0x6A

  def _read_temperature(self) -> float:
    scale = 16.0 if self.source == log.SensorEventData.SensorSource.lsm6ds3 else 256.0
    data = self.read(0x20, 2)
    return 25 + (self.parse_16bit(data[0], data[1]) / scale)

  def init(self):
    chip_id = self.verify_chip_id(0x0F, [0x69, 0x6A])
    if chip_id == 0x6A:
      self.source = log.SensorEventData.SensorSource.lsm6ds3trc
    else:
      self.source = log.SensorEventData.SensorSource.lsm6ds3

  def get_event(self, ts: int | None = None) -> log.SensorEventData:
    event = log.SensorEventData.new_message()
    event.version = 1
    event.timestamp = int(time.monotonic() * 1e9)
    event.source = self.source
    event.temperature = self._read_temperature()
    return event

  def shutdown(self) -> None:
    pass
