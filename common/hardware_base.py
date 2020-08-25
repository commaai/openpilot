from abc import abstractmethod

class HardwareBase:
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
