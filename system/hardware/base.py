from abc import abstractmethod, ABC
from collections import namedtuple
from typing import Dict

from cereal import log

ThermalConfig = namedtuple('ThermalConfig', ['cpu', 'gpu', 'mem', 'bat', 'ambient', 'pmic'])
NetworkType = log.DeviceState.NetworkType


class HardwareBase(ABC):
  @staticmethod
  def get_cmdline() -> Dict[str, str]:
    with open('/proc/cmdline') as f:
      cmdline = f.read()
    return {kv[0]: kv[1] for kv in [s.split('=') for s in cmdline.split(' ')] if len(kv) == 2}

  @staticmethod
  def read_param_file(path, parser, default=0):
    try:
      with open(path) as f:
        return parser(f.read())
    except Exception:
      return default

  @abstractmethod
  def reboot(self, reason=None):
    pass

  @abstractmethod
  def uninstall(self):
    pass

  @abstractmethod
  def get_os_version(self):
    pass

  @abstractmethod
  def get_device_type(self):
    pass

  @abstractmethod
  def get_sound_card_online(self):
    pass

  @abstractmethod
  def get_imei(self, slot) -> str:
    pass

  @abstractmethod
  def get_serial(self):
    pass

  @abstractmethod
  def get_subscriber_info(self):
    pass

  @abstractmethod
  def get_network_info(self):
    pass

  @abstractmethod
  def get_network_type(self):
    pass

  @abstractmethod
  def get_sim_info(self):
    pass

  @abstractmethod
  def get_network_strength(self, network_type):
    pass

  def get_network_metered(self, network_type) -> bool:
    return network_type not in (NetworkType.none, NetworkType.wifi, NetworkType.ethernet)

  @staticmethod
  def set_bandwidth_limit(upload_speed_kbps: int, download_speed_kbps: int) -> None:
    pass

  @abstractmethod
  def get_usb_present(self):
    pass

  @abstractmethod
  def get_current_power_draw(self):
    pass

  @abstractmethod
  def get_som_power_draw(self):
    pass

  @abstractmethod
  def shutdown(self):
    pass

  @abstractmethod
  def get_thermal_config(self):
    pass

  @abstractmethod
  def set_screen_brightness(self, percentage):
    pass

  @abstractmethod
  def get_screen_brightness(self):
    pass

  @abstractmethod
  def set_power_save(self, powersave_enabled):
    pass

  @abstractmethod
  def get_gpu_usage_percent(self):
    pass

  def get_modem_version(self):
    return None

  def get_modem_nv(self):
    return None

  @abstractmethod
  def get_modem_temperatures(self):
    pass

  @abstractmethod
  def get_nvme_temperatures(self):
    pass

  @abstractmethod
  def initialize_hardware(self):
    pass

  def configure_modem(self):
    pass

  @abstractmethod
  def get_networks(self):
    pass

  def reset_internal_panda(self):
    pass

  def recover_internal_panda(self):
    pass
