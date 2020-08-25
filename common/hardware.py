import os
from typing import cast

from cereal import log
from common.android import Android
from common.hardware_base import HardwareBase

EON = os.path.isfile('/EON')
TICI = os.path.isfile('/TICI')
PC = (not EON) and (not TICI)
ANDROID = EON


NetworkType = log.ThermalData.NetworkType
NetworkStrength = log.ThermalData.NetworkStrength


class Linux(HardwareBase):
  def get_sound_card_online(self):
    return True

  def get_imei(self, slot):
    return ""

  def get_serial(self):
    return "cccccccc"

  def get_subscriber_info(self):
    return ""

  def reboot(self, reason=None):
    print("REBOOT!")

  def get_network_type(self):
    return NetworkType.none

  def get_network_strength(self, network_type):
    pass


if EON:
  HARDWARE = cast(HardwareBase, Android())
else:
  HARDWARE = cast(HardwareBase, Linux())
