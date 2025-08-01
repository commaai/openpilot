import os
import time

from cereal import log
from openpilot.system.sensord.sensors.i2c_sensor import Sensor

class LSM6DS3_Accel(Sensor):
  LSM6DS3_ACCEL_I2C_REG_DRDY_CFG  = 0x0B
  LSM6DS3_ACCEL_I2C_REG_INT1_CTRL = 0x0D
  LSM6DS3_ACCEL_I2C_REG_CTRL1_XL  = 0x10
  LSM6DS3_ACCEL_I2C_REG_CTRL3_C   = 0x12
  LSM6DS3_ACCEL_I2C_REG_CTRL5_C   = 0x14
  LSM6DS3_ACCEL_I2C_REG_STAT_REG  = 0x1E
  LSM6DS3_ACCEL_I2C_REG_OUTX_L_XL = 0x28

  LSM6DS3_ACCEL_ODR_104HZ       = (0b0100 << 4)
  LSM6DS3_ACCEL_INT1_DRDY_XL    = 0b1
  LSM6DS3_ACCEL_DRDY_XLDA       = 0b1
  LSM6DS3_ACCEL_DRDY_PULSE_MODE = (1 << 7)
  LSM6DS3_ACCEL_IF_INC          = 0b00000100

  LSM6DS3_ACCEL_ODR_52HZ        = (0b0011 << 4)
  LSM6DS3_ACCEL_FS_4G           = (0b10 << 2)
  LSM6DS3_ACCEL_IF_INC_BDU      = 0b01000100
  LSM6DS3_ACCEL_POSITIVE_TEST   = 0b01
  LSM6DS3_ACCEL_NEGATIVE_TEST   = 0b10
  LSM6DS3_ACCEL_MIN_ST_LIMIT_mg = 90.0
  LSM6DS3_ACCEL_MAX_ST_LIMIT_mg = 1700.0

  @property
  def device_address(self) -> int:
    return 0x6A

  def reset(self):
    self.write(0x12, 0x1)
    time.sleep(0.1)

  def init(self):
    chip_id = self.verify_chip_id(0x0F, [0x69, 0x6A])
    if chip_id == 0x6A:
      self.source = log.SensorEventData.SensorSource.lsm6ds3trc
    else:
      self.source = log.SensorEventData.SensorSource.lsm6ds3

    # self-test
    if os.getenv("LSM_SELF_TEST") == "1":
      self.self_test(self.LSM6DS3_ACCEL_POSITIVE_TEST)
      self.self_test(self.LSM6DS3_ACCEL_NEGATIVE_TEST)

    # actual init
    int1 = self.read(self.LSM6DS3_ACCEL_I2C_REG_INT1_CTRL, 1)[0]
    int1 |= self.LSM6DS3_ACCEL_INT1_DRDY_XL
    self.writes((
      # Enable continuous update and automatic address increment
      (self.LSM6DS3_ACCEL_I2C_REG_CTRL3_C, self.LSM6DS3_ACCEL_IF_INC),
      # Set ODR to 104 Hz, FS to ±2g (default)
      (self.LSM6DS3_ACCEL_I2C_REG_CTRL1_XL, self.LSM6DS3_ACCEL_ODR_104HZ),
      # Configure data ready signal to pulse mode
      (self.LSM6DS3_ACCEL_I2C_REG_DRDY_CFG, self.LSM6DS3_ACCEL_DRDY_PULSE_MODE),
      # Enable data ready interrupt on INT1 without resetting existing interrupts
      (self.LSM6DS3_ACCEL_I2C_REG_INT1_CTRL, int1),
    ))

  def get_event(self, ts: int | None = None) -> log.SensorEventData:
    assert ts is not None  # must come from the IRQ event

    # Check if data is ready since IRQ is shared with gyro
    status_reg = self.read(self.LSM6DS3_ACCEL_I2C_REG_STAT_REG, 1)[0]
    if (status_reg & self.LSM6DS3_ACCEL_DRDY_XLDA) == 0:
      raise self.DataNotReady

    scale = 9.81 * 2.0 / (1 << 15)
    b = self.read(self.LSM6DS3_ACCEL_I2C_REG_OUTX_L_XL, 6)
    x = self.parse_16bit(b[0], b[1]) * scale
    y = self.parse_16bit(b[2], b[3]) * scale
    z = self.parse_16bit(b[4], b[5]) * scale

    event = log.SensorEventData.new_message()
    event.timestamp = ts
    event.version = 1
    event.sensor = 1  # SENSOR_ACCELEROMETER
    event.type = 1    # SENSOR_TYPE_ACCELEROMETER
    event.source = self.source
    a = event.init('acceleration')
    a.v = [y, -x, z]
    a.status = 1
    return event

  def shutdown(self) -> None:
    # Disable data ready interrupt on INT1
    value = self.read(self.LSM6DS3_ACCEL_I2C_REG_INT1_CTRL, 1)[0]
    value &= ~self.LSM6DS3_ACCEL_INT1_DRDY_XL
    self.write(self.LSM6DS3_ACCEL_I2C_REG_INT1_CTRL, value)

    # Power down by clearing ODR bits
    value = self.read(self.LSM6DS3_ACCEL_I2C_REG_CTRL1_XL, 1)[0]
    value &= 0x0F
    self.write(self.LSM6DS3_ACCEL_I2C_REG_CTRL1_XL, value)

  # *** self-test stuff ***
  def _wait_for_data_ready(self):
    while True:
      drdy = self.read(self.LSM6DS3_ACCEL_I2C_REG_STAT_REG, 1)[0]
      if drdy & self.LSM6DS3_ACCEL_DRDY_XLDA:
        break

  def _read_and_avg_data(self, scaling: float) -> list[float]:
    out_buf = [0.0, 0.0, 0.0]
    for _ in range(5):
      self._wait_for_data_ready()
      b = self.read(self.LSM6DS3_ACCEL_I2C_REG_OUTX_L_XL, 6)
      for j in range(3):
        val = self.parse_16bit(b[j*2], b[j*2+1]) * scaling
        out_buf[j] += val
    return [x / 5.0 for x in out_buf]

  def self_test(self, test_type: int) -> None:
    # Prepare sensor for self-test
    self.write(self.LSM6DS3_ACCEL_I2C_REG_CTRL3_C, self.LSM6DS3_ACCEL_IF_INC_BDU)

    # Configure ODR and full scale based on sensor type
    if self.source == log.SensorEventData.SensorSource.lsm6ds3trc:
      odr_fs = self.LSM6DS3_ACCEL_FS_4G | self.LSM6DS3_ACCEL_ODR_52HZ
      scaling = 0.122  # mg/LSB for ±4g
    else:
      odr_fs = self.LSM6DS3_ACCEL_ODR_52HZ
      scaling = 0.061  # mg/LSB for ±2g
    self.write(self.LSM6DS3_ACCEL_I2C_REG_CTRL1_XL, odr_fs)

    # Wait for stable output
    time.sleep(0.1)
    self._wait_for_data_ready()
    val_st_off = self._read_and_avg_data(scaling)

    # Enable self-test
    self.write(self.LSM6DS3_ACCEL_I2C_REG_CTRL5_C, test_type)

    # Wait for stable output
    time.sleep(0.1)
    self._wait_for_data_ready()
    val_st_on = self._read_and_avg_data(scaling)

    # Disable sensor and self-test
    self.write(self.LSM6DS3_ACCEL_I2C_REG_CTRL1_XL, 0)
    self.write(self.LSM6DS3_ACCEL_I2C_REG_CTRL5_C, 0)

    # Calculate differences and check limits
    test_val = [abs(on - off) for on, off in zip(val_st_on, val_st_off, strict=False)]
    for val in test_val:
      if val < self.LSM6DS3_ACCEL_MIN_ST_LIMIT_mg or val > self.LSM6DS3_ACCEL_MAX_ST_LIMIT_mg:
        raise self.SensorException(f"Accelerometer self-test failed for test type {test_type}")

if __name__ == "__main__":
  import numpy as np
  s = LSM6DS3_Accel(1)
  s.init()
  time.sleep(0.2)
  e = s.get_event(0)
  print(e)
  print(np.linalg.norm(e.acceleration.v))
  s.shutdown()
