import struct
from typing import List

from .base import BaseHandle, BaseSTBootloaderHandle, TIMEOUT
from .constants import McuType

class PandaUsbHandle(BaseHandle):
  def __init__(self, libusb_handle):
    self._libusb_handle = libusb_handle

  def close(self):
    self._libusb_handle.close()

  def controlWrite(self, request_type: int, request: int, value: int, index: int, data, timeout: int = TIMEOUT, expect_disconnect: bool = False):
    return self._libusb_handle.controlWrite(request_type, request, value, index, data, timeout)

  def controlRead(self, request_type: int, request: int, value: int, index: int, length: int, timeout: int = TIMEOUT):
    return self._libusb_handle.controlRead(request_type, request, value, index, length, timeout)

  def bulkWrite(self, endpoint: int, data: List[int], timeout: int = TIMEOUT) -> int:
    return self._libusb_handle.bulkWrite(endpoint, data, timeout)  # type: ignore

  def bulkRead(self, endpoint: int, length: int, timeout: int = TIMEOUT) -> bytes:
    return self._libusb_handle.bulkRead(endpoint, length, timeout)  # type: ignore



class STBootloaderUSBHandle(BaseSTBootloaderHandle):
  DFU_DNLOAD = 1
  DFU_UPLOAD = 2
  DFU_GETSTATUS = 3
  DFU_CLRSTATUS = 4
  DFU_ABORT = 6

  def __init__(self, libusb_device, libusb_handle):
    self._libusb_handle = libusb_handle

    # TODO: Find a way to detect F4 vs F2
    # TODO: also check F4 BCD, don't assume in else
    self._mcu_type = McuType.H7 if libusb_device.getbcdDevice() == 512 else McuType.F4

  def _status(self) -> None:
    while 1:
      dat = self._libusb_handle.controlRead(0x21, self.DFU_GETSTATUS, 0, 0, 6)
      if dat[1] == 0:
        break

  def _erase_page_address(self, address: int) -> None:
    self._libusb_handle.controlWrite(0x21, self.DFU_DNLOAD, 0, 0, b"\x41" + struct.pack("I", address))
    self._status()

  def get_mcu_type(self):
    return self._mcu_type

  def erase_app(self):
    self._erase_page_address(self._mcu_type.config.app_address)

  def erase_bootstub(self):
    self._erase_page_address(self._mcu_type.config.bootstub_address)

  def clear_status(self):
    # Clear status
    stat = self._libusb_handle.controlRead(0x21, self.DFU_GETSTATUS, 0, 0, 6)
    if stat[4] == 0xa:
      self._libusb_handle.controlRead(0x21, self.DFU_CLRSTATUS, 0, 0, 0)
    elif stat[4] == 0x9:
      self._libusb_handle.controlWrite(0x21, self.DFU_ABORT, 0, 0, b"")
      self._status()
    stat = str(self._libusb_handle.controlRead(0x21, self.DFU_GETSTATUS, 0, 0, 6))

  def close(self):
    self._libusb_handle.close()

  def program(self, address, dat):
    # Set Address Pointer
    self._libusb_handle.controlWrite(0x21, self.DFU_DNLOAD, 0, 0, b"\x21" + struct.pack("I", address))
    self._status()

    # Program
    bs = min(len(dat), self._mcu_type.config.block_size)
    dat += b"\xFF" * ((bs - len(dat)) % bs)
    for i in range(0, len(dat) // bs):
      ldat = dat[i * bs:(i + 1) * bs]
      print("programming %d with length %d" % (i, len(ldat)))
      self._libusb_handle.controlWrite(0x21, self.DFU_DNLOAD, 2 + i, 0, ldat)
      self._status()

  def jump(self, address):
    self._libusb_handle.controlWrite(0x21, self.DFU_DNLOAD, 0, 0, b"\x21" + struct.pack("I", address))
    self._status()
    try:
      self._libusb_handle.controlWrite(0x21, self.DFU_DNLOAD, 2, 0, b"")
      _ = str(self._libusb_handle.controlRead(0x21, self.DFU_GETSTATUS, 0, 0, 6))
    except Exception:
      pass
