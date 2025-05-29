import time
from cereal import log
from openpilot.system.sensord.sensors.i2c_sensor import I2CSensor

# Register addresses
MMC5603_MAGN_I2C_REG_ID = 0x39
MMC5603_MAGN_I2C_REG_ODR = 0x1A
MMC5603_MAGN_I2C_REG_CTRL0 = 0x1B
MMC5603_MAGN_I2C_REG_CTRL1 = 0x1C
MMC5603_MAGN_I2C_REG_CTRL2 = 0x1D
MMC5603_MAGN_I2C_REG_STATUS = 0x18
MMC5603_MAGN_I2C_REG_OUT_X_L = 0x00

# Control register settings
MMC5603_MAGN_ODR_50HZ = 0x0A  # 50Hz output data rate
MMC5603_MAGN_CONTINUOUS_MODE = 0x80
MMC5603_MAGN_AUTO_SR_EN = 0x10  # Auto Set/Reset enable
MMC5603_MAGN_HIGH_ACCURACY = 0x80  # High accuracy mode
MMC5603_MAGN_CMD_TM_M = 0x01  # Trigger measurement
MMC5603_MAGN_CMD_RESET = 0x10  # Software reset

# Status register bits
MMC5603_MAGN_STATUS_MEAS_M_DONE = 0x01

class MMC5603NJ_Magn(I2CSensor):
  def __init__(self, bus: int, gpio_nr: int = 0, shared_gpio: bool = False):
    super().__init__(bus, gpio_nr, shared_gpio)
    self.source = log.SensorEventData.SensorSource.mmc5603nj
    self.scaling = 0.0625  # 0.0625 mG/LSB

  @property
  def device_address(self) -> int:
    return 0x30  # Default I2C address for MMC5603NJ

  def _wait_for_measurement(self, timeout_ms: int = 100) -> bool:
    """Wait for measurement to complete"""
    start_time = time.time()
    while (time.time() - start_time) * 1000 < timeout_ms:
      status = self.read_register(MMC5603_MAGN_I2C_REG_STATUS, 1)[0]
      if status & MMC5603_MAGN_STATUS_MEAS_M_DONE:
        return True
      time.sleep(0.001)
    return False

  def _read_data(self) -> list[float]:
    status = self.read_register(MMC5603_MAGN_I2C_REG_STATUS, 1)[0]
    assert (status & 0x01) != 0

    data = self.read_register(MMC5603_MAGN_I2C_REG_OUT_X_L, 8)
    x = self.twos_complement((data[0] << 12) | (data[1] << 4) | (data[6] >> 4), 20)
    y = self.twos_complement((data[2] << 12) | (data[3] << 4) | (data[7] >> 4), 20)
    z = self.twos_complement((data[4] << 12) | (data[5] << 4) | (data[6] & 0x0F), 20)
    return [x * 0.0000625, y * 0.0000625, z * 0.0000625]

  def init(self):
    self.write_register(MMC5603_MAGN_I2C_REG_CTRL1, MMC5603_MAGN_CMD_RESET)
    time.sleep(0.01)  # Wait for reset to complete

    # Configure ODR and continuous measurement mode
    self.write_register(MMC5603_MAGN_I2C_REG_ODR, MMC5603_MAGN_ODR_50HZ)
    # Enable auto Set/Reset and high accuracy mode
    self.write_register(MMC5603_MAGN_I2C_REG_CTRL0, MMC5603_MAGN_AUTO_SR_EN)
    self.write_register(MMC5603_MAGN_I2C_REG_CTRL1, MMC5603_MAGN_HIGH_ACCURACY)
    self.write_register(MMC5603_MAGN_I2C_REG_CTRL2, 0x00)  # No interrupt

    # Wait for first measurement
    time.sleep(0.02)  # 20ms for first measurement

    # Read once to clear any stale data
    try:
      self._read_data()
    except Exception:
      pass

  def get_event(self, ts: int | None = None) -> log.SensorEventData:
    if ts is None:
      ts = int(time.monotonic() * 1e9)

    event = log.SensorEventData.new_message()
    event.timestamp = ts
    event.source = self.source

    m = event.init('magnetic')
    m.v = self._read_data()
    m.status = 1

    return event

  def shutdown(self) -> None:
    # Put sensor in idle mode
    self.write_register(MMC5603_MAGN_I2C_REG_CTRL0, 0x00)


if __name__ == "__main__":
  s = MMC5603NJ_Magn(1)
  s.init()
  print(s.get_event())
  s.shutdown()
