import ft4222
import ft4222.I2CMaster

DEBUG = False

INA231_ADDR = 0x40
INA231_REG_CONFIG = 0x00
INA231_REG_SHUNT_VOLTAGE = 0x01
INA231_REG_BUS_VOLTAGE = 0x02
INA231_REG_POWER = 0x03
INA231_REG_CURRENT = 0x04
INA231_REG_CALIBRATION = 0x05

INA231_BUS_LSB = 1.25e-3
INA231_SHUNT_LSB = 2.5e-6
SHUNT_RESISTOR = 30e-3
CURRENT_LSB = 1e-5

class Zookeeper:
  def __init__(self):
    if ft4222.createDeviceInfoList() < 2:
      raise Exception("No connected zookeeper found!")
    self.dev_a = ft4222.openByDescription("FT4222 A")
    self.dev_b = ft4222.openByDescription("FT4222 B")

    if DEBUG:
      for i in range(ft4222.createDeviceInfoList()):
        print(f"Device {i}: {ft4222.getDeviceInfoDetail(i, False)}")

    # Setup GPIO
    self.dev_b.gpio_Init(gpio2=ft4222.Dir.OUTPUT, gpio3=ft4222.Dir.OUTPUT)
    self.dev_b.setSuspendOut(False)
    self.dev_b.setWakeUpInterrut(False)

    # Setup I2C
    self.dev_a.i2cMaster_Init(kbps=400)
    self._initialize_ina()

  # Helper functions
  def _read_ina_register(self, register, length):
    self.dev_a.i2cMaster_WriteEx(INA231_ADDR, data=register, flag=ft4222.I2CMaster.Flag.REPEATED_START)
    return self.dev_a.i2cMaster_Read(INA231_ADDR, bytesToRead=length)

  def _write_ina_register(self, register, data):
    msg = register.to_bytes(1, byteorder="big") + data.to_bytes(2, byteorder="big")
    self.dev_a.i2cMaster_Write(INA231_ADDR, data=msg)

  def _initialize_ina(self):
    # Config
    self._write_ina_register(INA231_REG_CONFIG, 0x4127)

    # Calibration
    CAL_VALUE = int(0.00512 / (CURRENT_LSB * SHUNT_RESISTOR))
    if DEBUG:
      print(f"Calibration value: {hex(CAL_VALUE)}")
    self._write_ina_register(INA231_REG_CALIBRATION, CAL_VALUE)

  def _set_gpio(self, number, enabled):
    self.dev_b.gpio_Write(portNum=number, value=enabled)

  # Public API functions
  def set_device_power(self, enabled):
    self._set_gpio(2, enabled)

  def set_device_ignition(self, enabled):
    self._set_gpio(3, enabled)

  def read_current(self):
    # Returns in A
    return int.from_bytes(self._read_ina_register(INA231_REG_CURRENT, 2), byteorder="big") * CURRENT_LSB

  def read_power(self):
    # Returns in W
    return int.from_bytes(self._read_ina_register(INA231_REG_POWER, 2), byteorder="big") * CURRENT_LSB * 25

  def read_voltage(self):
    # Returns in V
    return int.from_bytes(self._read_ina_register(INA231_REG_BUS_VOLTAGE, 2), byteorder="big") * INA231_BUS_LSB
