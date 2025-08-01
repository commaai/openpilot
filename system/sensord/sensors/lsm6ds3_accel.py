import os
import time

from cereal import log
from openpilot.system.sensord.sensors.i2c_sensor import Sensor

class LSM6DS3_Accel(Sensor):
  LSM6DS3_ACCEL_I2C_REG_FIFO_CTRL1 = 0x06
  LSM6DS3_ACCEL_I2C_REG_FIFO_CTRL2 = 0x07
  LSM6DS3_ACCEL_I2C_REG_FIFO_CTRL3 = 0x08
  LSM6DS3_ACCEL_I2C_REG_FIFO_CTRL5 = 0x0A
  LSM6DS3_ACCEL_I2C_REG_DRDY_CFG   = 0x0B
  LSM6DS3_ACCEL_I2C_REG_INT1_CTRL  = 0x0D
  LSM6DS3_ACCEL_I2C_REG_CTRL1_XL   = 0x10
  LSM6DS3_ACCEL_I2C_REG_CTRL3_C    = 0x12
  LSM6DS3_ACCEL_I2C_REG_CTRL5_C    = 0x14
  LSM6DS3_ACCEL_I2C_REG_CTRL8_XL   = 0x17
  LSM6DS3_ACCEL_I2C_REG_STAT_REG   = 0x1E
  LSM6DS3_ACCEL_I2C_REG_OUTX_L_XL  = 0x28
  LSM6DS3_ACCEL_I2C_REG_FIFO_OUT_L = 0x3E

  LSM6DS3_ACCEL_FIFO_DEC_8      = 0b011
  LSM6DS3_ACCEL_FIFO_MODE_CONT  = 0b110
  LSM6DS3_ACCEL_FIFO_ODR_833Hz  = (0b0111 << 3)
  LSM6DS3_ACCEL_ODR_104HZ       = (0b0100 << 4)
  LSM6DS3_ACCEL_ODR_833HZ       = (0b0111 << 4)
  LSM6DS3_ACCEL_LPF1_BW_SEL     = (1 << 1)
  LSM6DS3_ACCEL_INT1_DRDY_XL    = 0b1
  LSM6DS3_ACCEL_INT1_FTH        = (1 << 3)
  LSM6DS3_ACCEL_DRDY_XLDA       = 0b1
  LSM6DS3_ACCEL_DRDY_PULSE_MODE = (1 << 7)
  LSM6DS3_ACCEL_IF_INC          = 0b00000100
  LSM6DS3_ACCEL_LPF2_XL_EN      = (1 << 7)
  LSM6DS3_ACCEL_HPCF_XL_ODRDIV9 = (0b10 << 5)
  LSM6DS3_ACCEL_INPUT_COMPOSITE = (1 << 3)

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

  def init(self):
    chip_id = self.verify_chip_id(0x0F, [0x69, 0x6A])
    if chip_id == 0x6A:
      self.source = log.SensorEventData.SensorSource.lsm6ds3trc
    else:
      self.source = log.SensorEventData.SensorSource.lsm6ds3

    # reset chip before init
    self.write(self.LSM6DS3_ACCEL_I2C_REG_CTRL3_C, 0b1)
    time.sleep(0.1)

    # self-test
    if os.getenv("LSM_SELF_TEST") == "1":
      self.self_test(self.LSM6DS3_ACCEL_POSITIVE_TEST)
      self.self_test(self.LSM6DS3_ACCEL_NEGATIVE_TEST)

    # actual init
    int1 = self.read(self.LSM6DS3_ACCEL_I2C_REG_INT1_CTRL, 1)[0]
    if self.source == log.SensorEventData.SensorSource.lsm6ds3trc:
      int1 |= self.LSM6DS3_ACCEL_INT1_FTH
      self.writes((
        # Enable continuous update and automatic address increment
        (self.LSM6DS3_ACCEL_I2C_REG_CTRL3_C, self.LSM6DS3_ACCEL_IF_INC),
        # Set ODR to 833 Hz, FS to ±2g (default)
        (self.LSM6DS3_ACCEL_I2C_REG_CTRL1_XL, self.LSM6DS3_ACCEL_ODR_833HZ | self.LSM6DS3_ACCEL_LPF1_BW_SEL),
        # Enable LPF2
        (self.LSM6DS3_ACCEL_I2C_REG_CTRL8_XL, self.LSM6DS3_ACCEL_LPF2_XL_EN | self.LSM6DS3_ACCEL_HPCF_XL_ODRDIV9 | self.LSM6DS3_ACCEL_INPUT_COMPOSITE),
        # Watermark: 3 words + 1
        (self.LSM6DS3_ACCEL_I2C_REG_FIFO_CTRL1, 7),
        (self.LSM6DS3_ACCEL_I2C_REG_FIFO_CTRL2, 0),
        # Decimate in FIFO
        (self.LSM6DS3_ACCEL_I2C_REG_FIFO_CTRL3, self.LSM6DS3_ACCEL_FIFO_DEC_8),
        # Other FIFO settings
        (self.LSM6DS3_ACCEL_I2C_REG_FIFO_CTRL5, self.LSM6DS3_ACCEL_FIFO_ODR_833Hz | self.LSM6DS3_ACCEL_FIFO_MODE_CONT),
        # Enable data ready interrupt on INT1 without resetting existing interrupts
        (self.LSM6DS3_ACCEL_I2C_REG_INT1_CTRL, int1),
      ))
    else:
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
    DATA_ADDR = self.LSM6DS3_ACCEL_I2C_REG_FIFO_OUT_L if self.source == log.SensorEventData.SensorSource.lsm6ds3trc else self.LSM6DS3_ACCEL_I2C_REG_OUTX_L_XL
    b = self.read(DATA_ADDR, 6)
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

  def interrupt_recovery(self) -> None:
    if self.source == log.SensorEventData.SensorSource.lsm6ds3trc:
      # Blink FIFO into bypass to clear contents
      self.write(self.LSM6DS3_ACCEL_I2C_REG_FIFO_CTRL5, self.LSM6DS3_ACCEL_FIFO_ODR_833Hz)
      time.sleep(0.01)
      self.write(self.LSM6DS3_ACCEL_I2C_REG_FIFO_CTRL5, self.LSM6DS3_ACCEL_FIFO_ODR_833Hz | self.LSM6DS3_ACCEL_FIFO_MODE_CONT)

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
