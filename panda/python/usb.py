from typing import List

from .base import BaseHandle

class PandaUsbHandle(BaseHandle):
  def __init__(self, libusb_handle):
    self._libusb_handle = libusb_handle

  def close(self):
    self._libusb_handle.close()

  def controlWrite(self, request_type: int, request: int, value: int, index: int, data, timeout: int = 0):
    return self._libusb_handle.controlWrite(request_type, request, value, index, data, timeout)

  def controlRead(self, request_type: int, request: int, value: int, index: int, length: int, timeout: int = 0):
    return self._libusb_handle.controlRead(request_type, request, value, index, length, timeout)

  def bulkWrite(self, endpoint: int, data: List[int], timeout: int = 0) -> int:
    return self._libusb_handle.bulkWrite(endpoint, data, timeout)  # type: ignore

  def bulkRead(self, endpoint: int, length: int, timeout: int = 0) -> bytes:
    return self._libusb_handle.bulkRead(endpoint, length, timeout)  # type: ignore

