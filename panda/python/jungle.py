# Python interface for the preflashed Jungle fixtures used by HITL tests.
import struct
from functools import wraps

from panda import Panda


def ensure_jungle_health_packet_version(fn):
  @wraps(fn)
  def wrapper(self, *args, **kwargs):
    if self.health_version != self.HEALTH_PACKET_VERSION:
      raise RuntimeError(f"Jungle firmware ({self.health_version}) doesn't match the \
                           library's health packet version ({self.HEALTH_PACKET_VERSION}).")
    return fn(self, *args, **kwargs)
  return wrapper


class PandaJungle(Panda):
  USB_PIDS = (0xddef, 0xddcf)

  HW_TYPE_UNKNOWN = b'\x00'
  HW_TYPE_V2 = b'\x02'

  H7_DEVICES = [HW_TYPE_V2, ]
  SUPPORTED_DEVICES = H7_DEVICES

  # Frozen protocol hash of the health packet in the installed Jungle firmware.
  HEALTH_PACKET_VERSION = 0xDD322770
  HEALTH_STRUCT = struct.Struct("<IffffffHHHHHHHHHHHH")

  HARNESS_ORIENTATION_NONE = 0
  HARNESS_ORIENTATION_1 = 1
  HARNESS_ORIENTATION_2 = 2

  @classmethod
  def spi_connect(cls, serial, ignore_version=False):
    return None, None, None, None

  def flash(self, fn=None, code=None, reconnect=True):
    raise NotImplementedError("Jungle firmware flashing is no longer supported")

  def recover(self, timeout: int | None = 60, reset: bool = True) -> bool:
    raise NotImplementedError("Jungle firmware recovery is no longer supported")

  def up_to_date(self, fn=None) -> bool:
    raise NotImplementedError("Jungle firmware is no longer distributed")

  # ******************* health *******************

  @ensure_jungle_health_packet_version
  def health(self):
    dat = self._handle.controlRead(PandaJungle.REQUEST_IN, 0xd2, 0, 0, self.HEALTH_STRUCT.size)
    a = self.HEALTH_STRUCT.unpack(dat)
    return {
      "uptime": a[0],
      "ch1_power": a[1],
      "ch2_power": a[2],
      "ch3_power": a[3],
      "ch4_power": a[4],
      "ch5_power": a[5],
      "ch6_power": a[6],
      "ch1_sbu1_voltage": a[7] / 1000.0,
      "ch1_sbu2_voltage": a[8] / 1000.0,
      "ch2_sbu1_voltage": a[9] / 1000.0,
      "ch2_sbu2_voltage": a[10] / 1000.0,
      "ch3_sbu1_voltage": a[11] / 1000.0,
      "ch3_sbu2_voltage": a[12] / 1000.0,
      "ch4_sbu1_voltage": a[13] / 1000.0,
      "ch4_sbu2_voltage": a[14] / 1000.0,
      "ch5_sbu1_voltage": a[15] / 1000.0,
      "ch5_sbu2_voltage": a[16] / 1000.0,
      "ch6_sbu1_voltage": a[17] / 1000.0,
      "ch6_sbu2_voltage": a[18] / 1000.0,
    }

  # ******************* control *******************

  def get_packets_versions(self):
    dat = self._handle.controlRead(PandaJungle.REQUEST_IN, 0xdd, 0, 0, 8)
    if dat and len(dat) == 8:
      return struct.unpack("<II", dat)
    return (0, 0)

  # ******************* jungle stuff *******************

  def set_panda_power(self, enabled):
    self._handle.controlWrite(PandaJungle.REQUEST_OUT, 0xa0, int(enabled), 0, b'')

  def set_harness_orientation(self, mode):
    self._handle.controlWrite(PandaJungle.REQUEST_OUT, 0xa1, int(mode), 0, b'')

  def set_ignition(self, enabled):
    self._handle.controlWrite(PandaJungle.REQUEST_OUT, 0xa2, int(enabled), 0, b'')

  def set_can_silent(self, silent):
    self._handle.controlWrite(PandaJungle.REQUEST_OUT, 0xf5, int(silent), 0, b'')
