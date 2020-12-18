from abc import abstractmethod


class HardwareBase:
  @staticmethod
  def get_cmdline():
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
  def get_sound_card_online(self):
    pass

  @abstractmethod
  def get_imei(self, slot):
    pass

  @abstractmethod
  def get_serial(self):
    pass

  @abstractmethod
  def get_subscriber_info(self):
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

  @abstractmethod
  def get_battery_capacity(self):
    pass

  @abstractmethod
  def get_battery_status(self):
    pass

  @abstractmethod
  def get_battery_current(self):
    pass

  @abstractmethod
  def get_battery_voltage(self):
    pass

  @abstractmethod
  def get_battery_charging(self):
    pass

  @abstractmethod
  def set_battery_charging(self, on):
    pass

  @abstractmethod
  def get_usb_present(self):
    pass

  @abstractmethod
  def get_current_power_draw(self):
    pass
