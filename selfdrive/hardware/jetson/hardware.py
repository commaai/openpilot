import random
import os

from cereal import log
from selfdrive.hardware.base import HardwareBase, ThermalConfig

NetworkType = log.DeviceState.NetworkType
NetworkStrength = log.DeviceState.NetworkStrength


class Jetson(HardwareBase):


  def get_os_version(self):
    return None

  def get_device_type(self):
    return "jetson"

  def get_sound_card_online(self):
    return True

  def reboot(self, reason=None):
    os.system("sudo reboot")

  def uninstall(self):
    print("uninstall")

  def get_imei(self, slot):
    return "%015d" % random.randint(0, 1 << 32)

  def get_serial(self):
    return "cccccccc"

  def get_subscriber_info(self):
    return ""

  def get_network_type(self):
    return NetworkType.wifi

  def get_sim_info(self):
    return {
      'sim_id': '',
      'mcc_mnc': None,
      'network_type': ["Unknown"],
      'sim_state': ["ABSENT"],
      'data_connected': False
    }

  def get_network_strength(self, network_type):
    return NetworkStrength.unknown

  def get_battery_capacity(self):
    return 100

  def get_battery_status(self):
    return ""

  def get_battery_current(self):
    return 0

  def get_battery_voltage(self):
    return 0

  def get_battery_charging(self):
    return True

  def set_battery_charging(self, on):
    pass

  def get_usb_present(self):
    return False

  def get_current_power_draw(self):
    return 0

  def shutdown(self):
    os.system("sudo poweroff")

  def get_thermal_config(self):
    return ThermalConfig(cpu=((None,), 1), gpu=((None,), 1), mem=(None, 1), bat=(None, 1), ambient=(None, 1))

  def set_screen_brightness(self, percentage):
    pass

  def set_power_save(self, enabled):
    if enabled:
      os.system("sudo nvpmodel -m 3")
    else:
      os.system("sudo echo 5000 > /sys/devices/c250000.i2c/i2c-7/7-0040/iio:device0/crit_current_limit_0")
      os.system("sudo nvpmodel -m 2 && sudo jetson_clocks")
