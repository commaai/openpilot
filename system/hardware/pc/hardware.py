import random

from cereal import log
from openpilot.system.hardware.base import HardwareBase, LPABase

NetworkType = log.DeviceState.NetworkType
NetworkStrength = log.DeviceState.NetworkStrength

class Pc(HardwareBase):
  def get_os_version(self):
    return None

  def get_device_type(self):
    return "pc"

  def reboot(self, reason=None):
    print("REBOOT!")

  def uninstall(self):
    print("uninstall")

  def get_imei(self, slot):
    return f"{random.randint(0, 1 << 32):015d}"

  def get_serial(self):
    return "cccccccc"

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

  def get_sim_lpa(self) -> LPABase:
    raise NotImplementedError("SIM LPA not implemented for PC")

  def get_network_strength(self, network_type):
    return NetworkStrength.unknown

  def get_current_power_draw(self):
    return 0

  def get_som_power_draw(self):
    return 0

  def shutdown(self):
    print("SHUTDOWN!")

  def set_screen_brightness(self, percentage):
    pass

  def get_screen_brightness(self):
    return 0

  def set_power_save(self, powersave_enabled):
    pass

  def get_gpu_usage_percent(self):
    return 0

  def get_modem_temperatures(self):
    return []

  def get_nvme_temperatures(self):
    return []

  def initialize_hardware(self):
    pass

  def get_networks(self):
    return None
