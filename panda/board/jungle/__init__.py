# python library to interface with panda
import os
import struct
from functools import wraps

from panda import Panda, PandaDFU
from panda.python.constants import McuType

BASEDIR = os.path.dirname(os.path.realpath(__file__))
FW_PATH = os.path.join(BASEDIR, "obj/")


def ensure_jungle_health_packet_version(fn):
  @wraps(fn)
  def wrapper(self, *args, **kwargs):
    if self.health_version != self.HEALTH_PACKET_VERSION:
      raise RuntimeError(f"Jungle firmware ({self.health_version}) doesn't match the \
                           library's health packet version ({self.HEALTH_PACKET_VERSION}). \
                           Reflash jungle.")
    return fn(self, *args, **kwargs)
  return wrapper


class PandaJungleDFU(PandaDFU):
  def recover(self):
    fn = os.path.join(FW_PATH, self._mcu_type.config.bootstub_fn.replace("panda", "panda_jungle"))
    with open(fn, "rb") as f:
      code = f.read()
    self.program_bootstub(code)
    self.reset()


class PandaJungle(Panda):
  USB_PIDS = (0xddef, 0xddcf)

  HW_TYPE_UNKNOWN = b'\x00'
  HW_TYPE_V1 = b'\x01'
  HW_TYPE_V2 = b'\x02'

  F4_DEVICES = [HW_TYPE_V1, ]
  H7_DEVICES = [HW_TYPE_V2, ]

  HEALTH_PACKET_VERSION = 1
  HEALTH_STRUCT = struct.Struct("<IffffffHHHHHHHHHHHH")

  HARNESS_ORIENTATION_NONE = 0
  HARNESS_ORIENTATION_1 = 1
  HARNESS_ORIENTATION_2 = 2

  @classmethod
  def spi_connect(cls, serial, ignore_version=False):
    return None, None, None, None, None

  def flash(self, fn=None, code=None, reconnect=True):
    if not fn:
      fn = os.path.join(FW_PATH, self._mcu_type.config.app_fn.replace("panda", "panda_jungle"))
    super().flash(fn=fn, code=code, reconnect=reconnect)

  def recover(self, timeout: int | None = 60, reset: bool = True) -> bool:
    dfu_serial = self.get_dfu_serial()

    if reset:
      self.reset(enter_bootstub=True)
      self.reset(enter_bootloader=True)

    if not self.wait_for_dfu(dfu_serial, timeout=timeout):
      return False

    dfu = PandaJungleDFU(dfu_serial)
    dfu.recover()

    # reflash after recover
    self.connect(True, True)
    self.flash()
    return True

  def get_mcu_type(self) -> McuType:
    hw_type = self.get_type()
    if hw_type in PandaJungle.F4_DEVICES:
      return McuType.F4
    elif hw_type in PandaJungle.H7_DEVICES:
      return McuType.H7
    else:
      # have to assume F4, see comment in Panda.connect
      # initially Jungle V1 has HW type: bytearray(b'')
      if hw_type == b'' or self._assume_f4_mcu:
        return McuType.F4

    raise ValueError(f"unknown HW type: {hw_type}")

  def up_to_date(self, fn=None) -> bool:
    if fn is None:
      fn = os.path.join(FW_PATH, self.get_mcu_type().config.app_fn.replace("panda", "panda_jungle"))
    return super().up_to_date(fn=fn)

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

  # Returns tuple with health packet version and CAN packet/USB packet version
  def get_packets_versions(self):
    dat = self._handle.controlRead(PandaJungle.REQUEST_IN, 0xdd, 0, 0, 3)
    if dat and len(dat) == 3:
      a = struct.unpack("BBB", dat)
      return (a[0], a[1], a[2])
    return (-1, -1, -1)

  # ******************* jungle stuff *******************

  def set_panda_power(self, enabled):
    self._handle.controlWrite(PandaJungle.REQUEST_OUT, 0xa0, int(enabled), 0, b'')

  def set_panda_individual_power(self, port, enabled):
    self._handle.controlWrite(PandaJungle.REQUEST_OUT, 0xa3, int(port), int(enabled), b'')

  def set_harness_orientation(self, mode):
    self._handle.controlWrite(PandaJungle.REQUEST_OUT, 0xa1, int(mode), 0, b'')

  def set_ignition(self, enabled):
    self._handle.controlWrite(PandaJungle.REQUEST_OUT, 0xa2, int(enabled), 0, b'')

  def set_can_silent(self, silent):
    self._handle.controlWrite(PandaJungle.REQUEST_OUT, 0xf5, int(silent), 0, b'')

  # ******************* serial *******************

  def debug_read(self):
    ret = []
    while 1:
      lret = bytes(self._handle.controlRead(PandaJungle.REQUEST_IN, 0xe0, 0, 0, 0x40))
      if len(lret) == 0:
        break
      ret.append(lret)
    return b''.join(ret)

  # ******************* header pins *******************

  def set_header_pin(self, pin_num, enabled):
    self._handle.controlWrite(Panda.REQUEST_OUT, 0xf7, int(pin_num), int(enabled), b'')
