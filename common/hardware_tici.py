import random
from common.hardware_base import HardwareBase
from cereal import log


NetworkType = log.ThermalData.NetworkType
NetworkStrength = log.ThermalData.NetworkStrength


class Tici(HardwareBase):
  @staticmethod
  def get_cmdline():
    with open('/proc/cmdline') as f:
      cmdline = f.read()

    return {kv[0]: kv[1] for kv in [s.split('=') for s in cmdline.split(' ')] if len(kv) == 2}

  def get_sound_card_online(self):
    return True

  def get_imei(self, slot):
    return "%015d" % random.randint(0, 1 << 32)

  def get_serial(self):
    return self.get_cmdline()['androidboot.serialno']

  def get_subscriber_info(self):
    return ""

  def reboot(self, reason=None):
    print("REBOOT!")

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


if __name__ == "__main__":
  t = Tici()
  print(t.get_serial())
