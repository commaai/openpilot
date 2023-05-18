import usb1
import struct
import binascii
from typing import List, Optional

from .base import BaseSTBootloaderHandle
from .spi import STBootloaderSPIHandle, PandaSpiException
from .usb import STBootloaderUSBHandle
from .constants import McuType


class PandaDFU:
  def __init__(self, dfu_serial: Optional[str]):
    # try USB, then SPI
    handle: Optional[BaseSTBootloaderHandle]
    handle = PandaDFU.usb_connect(dfu_serial)
    if handle is None:
      handle = PandaDFU.spi_connect(dfu_serial)

    if handle is None:
      raise Exception(f"failed to open DFU device {dfu_serial}")

    self._handle: BaseSTBootloaderHandle = handle
    self._mcu_type: McuType = self._handle.get_mcu_type()

  @staticmethod
  def usb_connect(dfu_serial: Optional[str]) -> Optional[STBootloaderUSBHandle]:
    handle = None
    context = usb1.USBContext()
    context.open()
    for device in context.getDeviceList(skip_on_error=True):
      if device.getVendorID() == 0x0483 and device.getProductID() == 0xdf11:
        try:
          this_dfu_serial = device.open().getASCIIStringDescriptor(3)
        except Exception:
          continue

        if this_dfu_serial == dfu_serial or dfu_serial is None:
          handle = STBootloaderUSBHandle(device, device.open())
          break

    return handle

  @staticmethod
  def spi_connect(dfu_serial: Optional[str]) -> Optional[STBootloaderSPIHandle]:
    handle = None
    this_dfu_serial = None

    try:
      handle = STBootloaderSPIHandle()
      this_dfu_serial = PandaDFU.st_serial_to_dfu_serial(handle.get_uid(), handle.get_mcu_type())
    except PandaSpiException:
      handle = None

    if dfu_serial is not None and dfu_serial != this_dfu_serial:
      handle = None

    return handle

  @staticmethod
  def list() -> List[str]:
    ret = PandaDFU.usb_list()
    ret += PandaDFU.spi_list()
    return list(set(ret))

  @staticmethod
  def usb_list() -> List[str]:
    dfu_serials = []
    try:
      with usb1.USBContext() as context:
        for device in context.getDeviceList(skip_on_error=True):
          if device.getVendorID() == 0x0483 and device.getProductID() == 0xdf11:
            try:
              dfu_serials.append(device.open().getASCIIStringDescriptor(3))
            except Exception:
              pass
    except Exception:
      pass
    return dfu_serials

  @staticmethod
  def spi_list() -> List[str]:
    try:
      h = PandaDFU.spi_connect(None)
      if h is not None:
        dfu_serial = PandaDFU.st_serial_to_dfu_serial(h.get_uid(), h.get_mcu_type())
        return [dfu_serial, ]
    except PandaSpiException:
      pass
    return []

  @staticmethod
  def st_serial_to_dfu_serial(st: str, mcu_type: McuType = McuType.F4):
    if st is None or st == "none":
      return None
    uid_base = struct.unpack("H" * 6, bytes.fromhex(st))
    if mcu_type == McuType.H7:
      return binascii.hexlify(struct.pack("!HHH", uid_base[1] + uid_base[5], uid_base[0] + uid_base[4], uid_base[3])).upper().decode("utf-8")
    else:
      return binascii.hexlify(struct.pack("!HHH", uid_base[1] + uid_base[5], uid_base[0] + uid_base[4] + 0xA, uid_base[3])).upper().decode("utf-8")

  def get_mcu_type(self) -> McuType:
    return self._mcu_type

  def reset(self):
    self._handle.jump(self._mcu_type.config.bootstub_address)

  def program_bootstub(self, code_bootstub):
    self._handle.clear_status()
    self._handle.erase_bootstub()
    self._handle.erase_app()
    self._handle.program(self._mcu_type.config.bootstub_address, code_bootstub)
    self.reset()

  def recover(self):
    with open(self._mcu_type.config.bootstub_path, "rb") as f:
      code = f.read()
    self.program_bootstub(code)

