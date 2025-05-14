import os
from abc import abstractmethod, ABC
from dataclasses import dataclass, fields

from cereal import log

NetworkType = log.DeviceState.NetworkType

@dataclass
class ThermalZone:
  # a zone from /sys/class/thermal/thermal_zone*
  name: str             # a.k.a type
  scale: float = 1000.  # scale to get degrees in C
  zone_number = -1

  def read(self) -> float:
    if self.zone_number < 0:
      for n in os.listdir("/sys/devices/virtual/thermal"):
        if not n.startswith("thermal_zone"):
          continue
        with open(os.path.join("/sys/devices/virtual/thermal", n, "type")) as f:
          if f.read().strip() == self.name:
            self.zone_number = int(n.removeprefix("thermal_zone"))
            break

    try:
      with open(f"/sys/devices/virtual/thermal/thermal_zone{self.zone_number}/temp") as f:
        return int(f.read()) / self.scale
    except FileNotFoundError:
      return 0

@dataclass
class ThermalConfig:
  cpu: list[ThermalZone] | None = None
  gpu: list[ThermalZone] | None = None
  dsp: ThermalZone | None = None
  pmic: list[ThermalZone] | None = None
  memory: ThermalZone | None = None
  intake: ThermalZone | None = None
  exhaust: ThermalZone | None = None
  case: ThermalZone | None = None

  def get_msg(self):
    ret = {}
    for f in fields(ThermalConfig):
      v = getattr(self, f.name)
      if v is not None:
        if isinstance(v, list):
          ret[f.name + "TempC"] = [x.read() for x in v]
        else:
          ret[f.name + "TempC"] = v.read()
    return ret

class HardwareBase(ABC):
  @staticmethod
  def get_cmdline() -> dict[str, str]:
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

  def booted(self) -> bool:
    return True

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
  def get_imei(self, slot) -> str:
    pass

  @abstractmethod
  def get_serial(self):
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
  def get_current_power_draw(self):
    pass

  @abstractmethod
  def get_som_power_draw(self):
    pass

  @abstractmethod
  def shutdown(self):
    pass

  def get_thermal_config(self):
    return ThermalConfig()

  def set_display_power(self, on: bool):
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

  def has_internal_panda(self) -> bool:
    return False

  def reset_internal_panda(self):
    pass

  def recover_internal_panda(self):
    pass

  def get_modem_data_usage(self):
    return -1, -1
