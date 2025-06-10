import os
import math
import time

from cereal import log
from openpilot.system.sensord.sensors.i2c_sensor import Sensor

class LSM6DS3_Gyro(Sensor):
  LSM6DS3_GYRO_I2C_REG_DRDY_CFG  = 0x0B
  LSM6DS3_GYRO_I2C_REG_INT1_CTRL = 0x0D
  LSM6DS3_GYRO_I2C_REG_CTRL2_G   = 0x11
  LSM6DS3_GYRO_I2C_REG_CTRL5_C   = 0x14
  LSM6DS3_GYRO_I2C_REG_STAT_REG  = 0x1E
  LSM6DS3_GYRO_I2C_REG_OUTX_L_G  = 0x22

  LSM6DS3_GYRO_ODR_104HZ       = (0b0100 << 4)
  LSM6DS3_GYRO_INT1_DRDY_G     = 0b10
  LSM6DS3_GYRO_DRDY_GDA        = 0b10
  LSM6DS3_GYRO_DRDY_PULSE_MODE = (1 << 7)

  LSM6DS3_GYRO_ODR_208HZ       = (0b0101 << 4)
  LSM6DS3_GYRO_FS_2000dps      = (0b11 << 2)
  LSM6DS3_GYRO_POSITIVE_TEST   = (0b01 << 2)
  LSM6DS3_GYRO_NEGATIVE_TEST   = (0b11 << 2)
  LSM6DS3_GYRO_MIN_ST_LIMIT_mdps = 150000.0
  LSM6DS3_GYRO_MAX_ST_LIMIT_mdps = 700000.0

  @property
  def device_address(self) -> int:
    return 0x6A

  def init(self):
    chip_id = self.verify_chip_id(0x0F, [0x69, 0x6A])
    if chip_id == 0x6A:
      self.source = log.SensorEventData.SensorSource.lsm6ds3trc
    else:
      self.source = log.SensorEventData.SensorSource.lsm6ds3

    # self-test
    if "LSM_SELF_TEST" in os.environ:
      self.self_test(self.LSM6DS3_GYRO_POSITIVE_TEST)
      self.self_test(self.LSM6DS3_GYRO_NEGATIVE_TEST)

    # actual init
    self.writes((
      # TODO: set scale. Default is +- 250 deg/s
      (self.LSM6DS3_GYRO_I2C_REG_CTRL2_G, self.LSM6DS3_GYRO_ODR_104HZ),
      # Configure data ready signal to pulse mode
      (self.LSM6DS3_GYRO_I2C_REG_DRDY_CFG, self.LSM6DS3_GYRO_DRDY_PULSE_MODE),
    ))
    value = self.read(self.LSM6DS3_GYRO_I2C_REG_INT1_CTRL, 1)[0]
    value |= self.LSM6DS3_GYRO_INT1_DRDY_G
    self.write(self.LSM6DS3_GYRO_I2C_REG_INT1_CTRL, value)

  def get_event(self, ts: int | None = None) -> log.SensorEventData:
    assert ts is not None  # must come from the IRQ event

    # Check if gyroscope data is ready, since it's shared with accelerometer
    status_reg = self.read(self.LSM6DS3_GYRO_I2C_REG_STAT_REG, 1)[0]
    if not (status_reg & self.LSM6DS3_GYRO_DRDY_GDA):
      raise self.DataNotReady

    b = self.read(self.LSM6DS3_GYRO_I2C_REG_OUTX_L_G, 6)
    x = self.parse_16bit(b[0], b[1])
    y = self.parse_16bit(b[2], b[3])
    z = self.parse_16bit(b[4], b[5])
    scale = (8.75 / 1000.0) * (math.pi / 180.0)
    xyz = [y * scale, -x * scale, z * scale]

    event = log.SensorEventData.new_message()
    event.timestamp = ts
    event.version = 2
    event.sensor = 5  # SENSOR_GYRO_UNCALIBRATED
    event.type = 16   # SENSOR_TYPE_GYROSCOPE_UNCALIBRATED
    event.source = self.source
    g = event.init('gyroUncalibrated')
    g.v = xyz
    g.status = 1
    return event

  def shutdown(self) -> None:
    # Disable data ready interrupt on INT1
    value = self.read(self.LSM6DS3_GYRO_I2C_REG_INT1_CTRL, 1)[0]
    value &= ~self.LSM6DS3_GYRO_INT1_DRDY_G
    self.write(self.LSM6DS3_GYRO_I2C_REG_INT1_CTRL, value)

    # Power down by clearing ODR bits
    value = self.read(self.LSM6DS3_GYRO_I2C_REG_CTRL2_G, 1)[0]
    value &= 0x0F
    self.write(self.LSM6DS3_GYRO_I2C_REG_CTRL2_G, value)

  # *** self-test stuff ***
  def _wait_for_data_ready(self):
    while True:
      drdy = self.read(self.LSM6DS3_GYRO_I2C_REG_STAT_REG, 1)[0]
      if drdy & self.LSM6DS3_GYRO_DRDY_GDA:
        break

  def _read_and_avg_data(self) -> list[float]:
    out_buf = [0.0, 0.0, 0.0]
    for _ in range(5):
      self._wait_for_data_ready()
      b = self.read(self.LSM6DS3_GYRO_I2C_REG_OUTX_L_G, 6)
      for j in range(3):
        val = self.parse_16bit(b[j*2], b[j*2+1]) * 70.0  # mdps/LSB for 2000 dps
        out_buf[j] += val
    return [x / 5.0 for x in out_buf]

  def self_test(self, test_type: int):
    # Set ODR to 208Hz, FS to 2000dps
    self.write(self.LSM6DS3_GYRO_I2C_REG_CTRL2_G, self.LSM6DS3_GYRO_ODR_208HZ | self.LSM6DS3_GYRO_FS_2000dps)

    # Wait for stable output
    time.sleep(0.15)
    self._wait_for_data_ready()
    val_st_off = self._read_and_avg_data()

    # Enable self-test
    self.write(self.LSM6DS3_GYRO_I2C_REG_CTRL5_C, test_type)

    # Wait for stable output
    time.sleep(0.05)
    self._wait_for_data_ready()
    val_st_on = self._read_and_avg_data()

    # Disable sensor and self-test
    self.write(self.LSM6DS3_GYRO_I2C_REG_CTRL2_G, 0)
    self.write(self.LSM6DS3_GYRO_I2C_REG_CTRL5_C, 0)

    # Calculate differences and check limits
    test_val = [abs(on - off) for on, off in zip(val_st_on, val_st_off, strict=False)]
    for val in test_val:
      if val < self.LSM6DS3_GYRO_MIN_ST_LIMIT_mdps or val > self.LSM6DS3_GYRO_MAX_ST_LIMIT_mdps:
        raise Exception(f"Gyroscope self-test failed for test type {test_type}")

if __name__ == "__main__":
  s = LSM6DS3_Gyro(1)
  s.init()
  time.sleep(0.1)
  print(s.get_event(0))
  s.shutdown()
