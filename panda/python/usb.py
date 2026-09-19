import struct

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

  def bulkWrite(self, endpoint: int, data: bytes, timeout: int = TIMEOUT) -> int:
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

    # example from F4: lsusb -v | grep Flash
    # iInterface  4 @Internal Flash  /0x08000000/04*016Kg,01*064Kg,011*128Kg
    for i in range(20):
      desc = libusb_handle.getStringDescriptor(i, 0)
      if desc is not None and desc.startswith("@Internal Flash"):
        sector_count = sum([int(s.split('*')[0]) for s in desc.split('/')[-1].split(',')])
        break
    mcu_by_sector_count = {m.config.sector_count: m for m in McuType}
    assert sector_count in mcu_by_sector_count, f"Unkown MCU: {sector_count=}"
    self._mcu_type = mcu_by_sector_count[sector_count]

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

  def erase_sector(self, sector: int):
    self._erase_page_address(self._mcu_type.config.sector_address(sector))

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
    for i in range(len(dat) // bs):
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
