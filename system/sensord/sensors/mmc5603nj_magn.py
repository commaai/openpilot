import time

from cereal import log
from openpilot.system.sensord.sensors.i2c_sensor import Sensor

# https://www.mouser.com/datasheet/2/821/Memsic_09102019_Datasheet_Rev.B-1635324.pdf

# Register addresses
REG_ODR = 0x1A
REG_INTERNAL_0 = 0x1B
REG_INTERNAL_1 = 0x1C

# Control register settings
CMM_FREQ_EN = (1 << 7)
AUTO_SR_EN  = (1 << 5)
SET         = (1 << 3)
RESET       = (1 << 4)

class MMC5603NJ_Magn(Sensor):
  @property
  def device_address(self) -> int:
    return 0x30

  def init(self):
    self.verify_chip_id(0x39, [0x10, ])
    self.writes((
      (REG_ODR, 0),

      # Set BW to 0b01 for 1-150 Hz operation
      (REG_INTERNAL_1, 0b01),
    ))

  def _read_data(self, cycle) -> list[float]:
    # start measurement
    self.write(REG_INTERNAL_0, cycle)
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
    ts = time.monotonic_ns()

    # SET - RESET cycle
    xyz = self._read_data(SET)
    reset_xyz = self._read_data(RESET)
    vals = [*xyz, *reset_xyz]

    event = log.SensorEventData.new_message()
    event.timestamp = ts
    event.version = 1
    event.sensor = 3 # SENSOR_MAGNETOMETER_UNCALIBRATED
    event.type = 14  # SENSOR_TYPE_MAGNETIC_FIELD_UNCALIBRATED
    event.source = log.SensorEventData.SensorSource.mmc5603nj

    m = event.init('magneticUncalibrated')
    m.v = vals
    m.status = int(all(int(v) != -32 for v in vals))

    return event

  def shutdown(self) -> None:
    v = self.read(REG_INTERNAL_0, 1)[0]
    self.writes((
      # disable auto-reset of measurements
      (REG_INTERNAL_0, (v & (~(CMM_FREQ_EN | AUTO_SR_EN)))),

      # disable continuous mode
      (REG_ODR, 0),
    ))
