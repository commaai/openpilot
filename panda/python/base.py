from abc import ABC, abstractmethod

from .constants import McuType

TIMEOUT = int(15 * 1e3)  # default timeout, in milliseconds

class BaseHandle(ABC):
  """
    A handle to talk to a panda.
    Borrows heavily from the libusb1 handle API.
  """

  @abstractmethod
  def close(self) -> None:
    ...

  @abstractmethod
  def controlWrite(self, request_type: int, request: int, value: int, index: int, data, timeout: int = TIMEOUT, expect_disconnect: bool = False):
    ...

  @abstractmethod
  def controlRead(self, request_type: int, request: int, value: int, index: int, length: int, timeout: int = TIMEOUT) -> bytes:
    ...

  @abstractmethod
  def bulkWrite(self, endpoint: int, data: bytes, timeout: int = TIMEOUT) -> int:
    ...

  @abstractmethod
  def bulkRead(self, endpoint: int, length: int, timeout: int = TIMEOUT) -> bytes:
    ...


class BaseSTBootloaderHandle(ABC):
  """
    A handle to talk to a panda while it's in the STM32 bootloader.
  """

  @abstractmethod
  def get_mcu_type(self) -> McuType:
    ...

  @abstractmethod
  def close(self) -> None:
    ...

  @abstractmethod
  def clear_status(self) -> None:
    ...

  @abstractmethod
  def program(self, address: int, dat: bytes) -> None:
    ...

  @abstractmethod
  def erase_sector(self, sector: int) -> None:
    ...

  @abstractmethod
  def jump(self, address: int) -> None:
    ...
