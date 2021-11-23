import random

from cereal import log
from selfdrive.hardware.base import HardwareBase, ThermalConfig

NetworkType = log.DeviceState.NetworkType
NetworkStrength = log.DeviceState.NetworkStrength


class Pc(HardwareBase):
  def get_os_version(self):
    return None

  def get_device_type(self):
    return "pc"

  def get_sound_card_online(self):
    return True

  def reboot(self, reason=None):
    print("REBOOT!")

  def uninstall(self):
    print("uninstall")

  def get_imei(self, slot):
    return "%015d" % random.randint(0, 1 << 32)

  def get_serial(self):
    return "cccccccc"

  def get_subscriber_info(self):
    return ""

  def get_network_info(self):
    return None

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
    print("SHUTDOWN!")

  def get_thermal_config(self):
    return ThermalConfig(cpu=((None,), 1), gpu=((None,), 1), mem=(None, 1), bat=(None, 1), ambient=(None, 1), pmic=(None, 1))

  def set_screen_brightness(self, percentage):
    pass

  def get_screen_brightness(self):
    return 0

  def set_power_save(self, powersave_enabled):
    pass

  def get_gpu_usage_percent(self):
    return 0

  def get_modem_version(self):
    return None

  def get_modem_temperatures(self):
    return []

  def get_nvme_temperatures(self):
    return []

  def initialize_hardware(self):
    pass

  def get_networks(self):
    return None
