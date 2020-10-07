from abc import abstractmethod


class HardwareBase:
  @staticmethod
  def get_cmdline():
    with open('/proc/cmdline') as f:
      cmdline = f.read()
    return {kv[0]: kv[1] for kv in [s.split('=') for s in cmdline.split(' ')] if len(kv) == 2}

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
  def reboot(self, reason=None):
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
