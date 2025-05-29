import time
import numpy as np
from cereal import log
from openpilot.common.swaglog import cloudlog
from openpilot.system.sensord.sensors.i2c_sensor import I2CSensor

# Register addresses
LSM6DS3_GYRO_I2C_REG_ID = 0x0F
LSM6DS3_GYRO_I2C_REG_CTRL2_G = 0x11
LSM6DS3_GYRO_I2C_REG_CTRL3_C = 0x12
LSM6DS3_GYRO_I2C_REG_CTRL5_C = 0x14
LSM6DS3_GYRO_I2C_REG_STAT_REG = 0x1E
LSM6DS3_GYRO_I2C_REG_OUTX_L_G = 0x22

# CTRL2_G configuration
LSM6DS3_GYRO_ODR_52HZ = 0x40  # 52Hz output data rate
LSM6DS3_GYRO_FS_2000DPS = 0x00  # ±2000 dps full scale

# CTRL3_C configuration
LSM6DS3_GYRO_IF_INC = 0x04  # Enable register address auto-increment
LSM6DS3_GYRO_BDU = 0x40     # Block data update

# STAT_REG bits
LSM6DS3_GYRO_DRDY_G = 0x02  # Gyroscope data ready

class LSM6DS3_Gyro(I2CSensor):
  def __init__(self, bus: int, gpio_nr: int = 0, shared_gpio: bool = False):
    super().__init__(bus, gpio_nr, shared_gpio)
    self.source = log.SensorEventData.SensorSource.lsm6ds3
    self.scaling = 70.0  # Default scaling for 2000dps (mdps/LSB)

  @property
  def device_address(self) -> int:
    return 0x6A  # Default I2C address for LSM6DS3

  def _wait_for_data_ready(self) -> None:
    """Wait for gyroscope data to be ready"""
    while True:
      status = self.read(LSM6DS3_GYRO_I2C_REG_STAT_REG, 1)[0]
      if status & LSM6DS3_GYRO_DRDY_G:
        break

  def _read_and_avg_data(self) -> list[float]:
    samples = []
    for _ in range(5):
      self._wait_for_data_ready()
      data = self.read(LSM6DS3_GYRO_I2C_REG_OUTX_L_G, 6)
      x = self.twos_complement((data[1] << 8) | data[0], 16)
      y = self.twos_complement((data[3] << 8) | data[2], 16)
      z = self.twos_complement((data[5] << 8) | data[4], 16)
      samples.append([x, y, z])

    avg = np.mean(samples, axis=0) * (self.scaling / 1000.0)
    return [float(x) for x in avg.tolist()]

  def init(self) -> bool:
    # Verify chip ID
    chip_id = self.verify_chip_id(LSM6DS3_GYRO_I2C_REG_ID, [0x69, 0x6A])
    if chip_id < 0:
      return False
    # Configure sensor
    try:
      # Enable block data update and auto-increment
      self.write(LSM6DS3_GYRO_I2C_REG_CTRL3_C,
                        LSM6DS3_GYRO_IF_INC | LSM6DS3_GYRO_BDU)

      # Configure gyroscope (52Hz, 2000 dps)
      self.write(LSM6DS3_GYRO_I2C_REG_CTRL2_G,
                        LSM6DS3_GYRO_ODR_52HZ | LSM6DS3_GYRO_FS_2000DPS)

      # Set scaling factor for 2000 dps (70 mdps/LSB)
      self.scaling = 70.0

      # Wait for first sample
      time.sleep(0.1)
      self._wait_for_data_ready()

      return True

    except Exception as e:
      print(f"Error initializing LSM6DS3 gyroscope: {e}")
      return False

  def get_event(self, ts: int | None = None) -> log.SensorEventData:
    if ts is None:
      ts = int(time.monotonic() * 1e9)

    event = log.SensorEventData.new_message()
    event.timestamp = ts
    event.source = self.source
    g = event.init('gyroUncalibrated')
    g.v = self._read_and_avg_data()
    g.status = 1

    return event

  def shutdown(self) -> None:
    try:
      self.write(LSM6DS3_GYRO_I2C_REG_CTRL2_G, 0x00)
    except Exception:
      cloudlog.exception("Error shutting down LSM6DS3 gyroscope")

if __name__ == "__main__":
  s = LSM6DS3_Gyro(1)
  s.init()
  print(s.get_event())
  s.shutdown()
