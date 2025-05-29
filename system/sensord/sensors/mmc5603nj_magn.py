import time

from cereal import log
from openpilot.system.sensord.sensors.i2c_sensor import I2CSensor

# https://www.mouser.com/datasheet/2/821/Memsic_09102019_MMC5603NJ_Datasheet_Rev.B-1635324.pdf

# Register addresses
MMC5603NJ_I2C_REG_ODR = 0x1A
MMC5603NJ_I2C_REG_INTERNAL_0 = 0x1B
MMC5603NJ_I2C_REG_INTERNAL_1 = 0x1C

# Control register settings
MMC5603NJ_CMM_FREQ_EN = (1 << 7)
MMC5603NJ_AUTO_SR_EN  = (1 << 5)
MMC5603NJ_SET         = (1 << 3)
MMC5603NJ_RESET       = (1 << 4)

# Status register bits
MMC5603NJ_STATUS_MEAS_M_DONE = 0x01

class MMC5603NJ_Magn(I2CSensor):
  @property
  def device_address(self) -> int:
    return 0x30

  def init(self):
    self.verify_chip_id(0x39, [0x10, ])
    self.writes((
      (MMC5603NJ_I2C_REG_ODR, 0),

      # Set BW to 0b01 for 1-150 Hz operation
      (MMC5603NJ_I2C_REG_INTERNAL_1, 0b01),
    ))

  def _read_data(self) -> list[float]:
    # start measurement
    self.write(MMC5603NJ_I2C_REG_INTERNAL_0, 0b01)
    self.wait()

    # read out XYZ
    scale = 1.0 / 16384.0
    b = self.read(0x00, 9)
    return [
      (self.parse_20bit(b[6], b[1], b[0]) * scale) - 32.0,
      (self.parse_20bit(b[7], b[3], b[2]) * scale) - 32.0,
      (self.parse_20bit(b[8], b[5], b[4]) * scale) - 32.0,
    ]

  def get_event(self, ts: int | None = None) -> log.SensorEventData:
    st = int(time.monotonic() * 1e9)

    # SET - RESET cycle
    self.write(MMC5603NJ_I2C_REG_INTERNAL_0, MMC5603NJ_SET)
    self.wait()
    xyz = self._read_data()

    self.write(MMC5603NJ_I2C_REG_INTERNAL_0, MMC5603NJ_RESET)
    self.wait()
    reset_xyz = self._read_data()

    vals = [*xyz, *reset_xyz]
    assert not any(int(v) == -32 for v in vals)

    event = log.SensorEventData.new_message()
    event.timestamp = st
    event.version = 1
    event.sensor = 3 # SENSOR_MAGNETOMETER_UNCALIBRATED
    event.type = 14  # SENSOR_TYPE_MAGNETIC_FIELD_UNCALIBRATED
    event.source = log.SensorEventData.SensorSource.mmc5603nj

    m = event.init('magneticUncalibrated')
    m.v = vals
    m.status = 1

    return event

  def shutdown(self) -> None:
    v = self.read(MMC5603NJ_I2C_REG_INTERNAL_0, 1)[0]
    self.writes((
      # disable auto-reset of measurements
      (MMC5603NJ_I2C_REG_INTERNAL_0, (v & (~(MMC5603NJ_CMM_FREQ_EN | MMC5603NJ_AUTO_SR_EN)))),

      # disable continuous mode
      (MMC5603NJ_I2C_REG_ODR, 0),
    ))


if __name__ == "__main__":
  s = MMC5603NJ_Magn(1)
  s.init()
  print(s.get_event())
  s.shutdown()
