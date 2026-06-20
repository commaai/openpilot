from cereal import log
from openpilot.common.hardware.base import HardwareBase

class Pc(HardwareBase):
  def get_device_type(self):
    return "pc"

  def get_network_type(self):
    # some stuff is gated on wifi, so just assume for now
    return log.DeviceState.NetworkType.wifi
