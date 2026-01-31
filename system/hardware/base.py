import os
from abc import abstractmethod, ABC
from dataclasses import dataclass, fields

from cereal import log

NetworkType = log.DeviceState.NetworkType
NetworkStrength = log.DeviceState.NetworkStrength

class LPAError(RuntimeError):
  pass

class LPAProfileNotFoundError(LPAError):
  pass

@dataclass
class Profile:
  iccid: str
  nickname: str
  enabled: bool
  provider: str

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

class LPABase(ABC):
  @abstractmethod
  def list_profiles(self) -> list[Profile]:
    pass

  @abstractmethod
  def get_active_profile(self) -> Profile | None:
    pass

  @abstractmethod
  def delete_profile(self, iccid: str) -> None:
    pass

  @abstractmethod
  def download_profile(self, qr: str, nickname: str | None = None) -> None:
    pass

  @abstractmethod
  def nickname_profile(self, iccid: str, nickname: str) -> None:
    pass

  @abstractmethod
  def switch_profile(self, iccid: str) -> None:
    pass

  def is_comma_profile(self, iccid: str) -> bool:
    return any(iccid.startswith(prefix) for prefix in ('8985235',))

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

  def reboot(self, reason=None):
    print("REBOOT!")

  def uninstall(self):
    print("uninstall")

  def get_os_version(self):
    return None

  @abstractmethod
  def get_device_type(self):
    pass

  def get_imei(self, slot) -> str:
    return ""

  def get_serial(self):
    return ""

  def get_network_info(self):
    return None

  def get_network_type(self):
    return NetworkType.none

  def get_sim_info(self):
    return {
      'sim_id': '',
      'mcc_mnc': None,
      'network_type': ["Unknown"],
      'sim_state': ["ABSENT"],
      'data_connected': False
    }

  def get_sim_lpa(self) -> LPABase:
    raise NotImplementedError("SIM LPA not available")

  def get_network_strength(self, network_type):
    return NetworkStrength.unknown

  def get_network_metered(self, network_type) -> bool:
    return network_type not in (NetworkType.none, NetworkType.wifi, NetworkType.ethernet)

  def get_current_power_draw(self):
    return 0

  def get_som_power_draw(self):
    return 0

  def shutdown(self):
    print("SHUTDOWN!")

  def get_thermal_config(self):
    return ThermalConfig()

  def set_display_power(self, on: bool):
    pass

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

  def initialize_hardware(self):
    pass

  def configure_modem(self):
    pass

  def reboot_modem(self):
    pass

  def get_networks(self):
    return None

  def has_internal_panda(self) -> bool:
    return False

  def reset_internal_panda(self):
    pass

  def recover_internal_panda(self):
    pass

  def get_modem_data_usage(self):
    return -1, -1

  def get_voltage(self) -> float:
    return 0.

  def get_current(self) -> float:
    return 0.

  def set_ir_power(self, percent: int):
    pass
