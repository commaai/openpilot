import os
import time
import numpy as np
from cereal import log
from openpilot.system.sensord.sensors.i2c_sensor import I2CSensor

# Register addresses
LSM6DS3_ACCEL_I2C_REG_ID = 0x0F
LSM6DS3_ACCEL_I2C_REG_CTRL1_XL = 0x10
LSM6DS3_ACCEL_I2C_REG_CTRL3_C = 0x12
LSM6DS3_ACCEL_I2C_REG_CTRL5_C = 0x14
LSM6DS3_ACCEL_I2C_REG_STAT_REG = 0x1E
LSM6DS3_ACCEL_I2C_REG_OUTX_L_XL = 0x28

# CTRL1_XL configuration
LSM6DS3_ACCEL_ODR_52HZ = 0x50  # 52Hz output data rate
LSM6DS3_ACCEL_ODR_104HZ = (0b0100 << 4) # 52Hz output data rate
LSM6DS3_ACCEL_FS_4G = 0x08     # ±4g full scale

# CTRL3_C configuration
LSM6DS3_ACCEL_IF_INC = 0x04    # Enable register address auto-increment
LSM6DS3_ACCEL_BDU = 0x40       # Block data update

# STAT_REG bits
LSM6DS3_ACCEL_DRDY_XLDA = 0x01  # Accelerometer data ready

# CTRL5_C configuration
LSM6DS3_ACCEL_ST_XL_POS = 0x04  # Accelerometer self-test positive
LSM6DS3_ACCEL_ST_XL_NEG = 0x0C  # Accelerometer self-test negative

class LSM6DS3_Accel(I2CSensor):
  def __init__(self, bus: int, gpio_nr: int = 0, shared_gpio: bool = False):
    super().__init__(bus, gpio_nr, shared_gpio)
    self.source = log.SensorEventData.SensorSource.lsm6ds3
    self.scaling = 0.061  # Default scaling for 2g full scale (mg/LSB)

  @property
  def device_address(self) -> int:
    return 0x6A  # Default I2C address for LSM6DS3

  def _wait_for_data_ready(self) -> None:
    while True:
      status = self.read(LSM6DS3_ACCEL_I2C_REG_STAT_REG, 1)[0]
      if status & LSM6DS3_ACCEL_DRDY_XLDA:
        break

  def _read_and_avg_data(self) -> list[float]:
    samples = []
    for _ in range(5):
      self._wait_for_data_ready()
      data = self.read(LSM6DS3_ACCEL_I2C_REG_OUTX_L_XL, 6)

      x = self.twos_complement((data[1] << 8) | data[0], 16)
      y = self.twos_complement((data[3] << 8) | data[1], 16)
      z = self.twos_complement((data[5] << 8) | data[4], 16)
      samples.append([x, y, z])

    avg: np.ndarray = np.mean(samples, axis=0) * self.scaling
    return [float(x) for x in avg.tolist()]

  def init(self) -> None:
    chip_id = self.verify_chip_id(LSM6DS3_ACCEL_I2C_REG_ID, [0x69, 0x6A])
    if chip_id == 0x6A:
      self.source = log.SensorEventData.SensorSource.lsm6ds3trc

    if "LSM_SELF_TEST" in os.environ:
      self.self_test()

    #self.init_gpio()

    # enable continuous update, and automatic increase
    self.write(LSM6DS3_ACCEL_I2C_REG_CTRL3_C, LSM6DS3_ACCEL_IF_INC)

    # TODO: set scale and bandwidth. Default is +- 2G, 50 Hz
    self.write(LSM6DS3_ACCEL_I2C_REG_CTRL1_XL, LSM6DS3_ACCEL_ODR_104HZ)

    # Set scaling factor for 4g full scale (2mg/LSB)
    self.scaling = 0.122

    # Wait for first sample
    time.sleep(0.1)
    self._wait_for_data_ready()

  def get_event(self, ts: int | None = None) -> log.SensorEventData:
    if ts is None:
      ts = int(time.monotonic() * 1e9)

    event = log.SensorEventData.new_message()
    event.timestamp = ts
    event.source = self.source

    a = event.init('acceleration')
    a.v = self._read_and_avg_data()
    a.status = 1

    return event

  def shutdown(self) -> None:
    self.write(LSM6DS3_ACCEL_I2C_REG_CTRL1_XL, 0x00)

  def self_test(self, test_type: int = LSM6DS3_ACCEL_ST_XL_POS) -> bool:
    # TODO: implement this
    return True


if __name__ == "__main__":
  s = LSM6DS3_Accel(1)
  s.init()
  print(s.get_event())
  s.shutdown()
