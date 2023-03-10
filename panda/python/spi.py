import binascii
import os
import fcntl
import math
import time
import struct
import logging
import threading
from contextlib import contextmanager
from functools import reduce
from typing import List, Optional

from .base import BaseHandle, BaseSTBootloaderHandle, TIMEOUT
from .constants import McuType, MCU_TYPE_BY_IDCODE

try:
  import spidev
except ImportError:
  spidev = None

# Constants
SYNC = 0x5A
HACK = 0x79
DACK = 0x85
NACK = 0x1F
CHECKSUM_START = 0xAB

MIN_ACK_TIMEOUT_MS = 100
MAX_XFER_RETRY_COUNT = 5

USB_MAX_SIZE = 0x40

DEV_PATH = "/dev/spidev0.0"


class PandaSpiException(Exception):
  pass

class PandaSpiUnavailable(PandaSpiException):
  pass

class PandaSpiNackResponse(PandaSpiException):
  pass

class PandaSpiMissingAck(PandaSpiException):
  pass

class PandaSpiBadChecksum(PandaSpiException):
  pass

class PandaSpiTransferFailed(PandaSpiException):
  pass


SPI_LOCK = threading.Lock()

class SpiDevice:
  """
  Provides locked, thread-safe access to a panda's SPI interface.
  """
  def __init__(self, speed=30000000):
    if not os.path.exists(DEV_PATH):
      raise PandaSpiUnavailable(f"SPI device not found: {DEV_PATH}")
    if spidev is None:
      raise PandaSpiUnavailable("spidev is not installed")

    self._spidev = spidev.SpiDev()  # pylint: disable=c-extension-no-member
    self._spidev.open(0, 0)
    self._spidev.max_speed_hz = speed

  @contextmanager
  def acquire(self):
    try:
      SPI_LOCK.acquire()
      fcntl.flock(self._spidev, fcntl.LOCK_EX)
      yield self._spidev
    finally:
      fcntl.flock(self._spidev, fcntl.LOCK_UN)
      SPI_LOCK.release()

  def close(self):
    self._spidev.close()


class PandaSpiHandle(BaseHandle):
  """
  A class that mimics a libusb1 handle for panda SPI communications.
  """
  def __init__(self):
    self.dev = SpiDevice()

  # helpers
  def _calc_checksum(self, data: List[int]) -> int:
    cksum = CHECKSUM_START
    for b in data:
      cksum ^= b
    return cksum

  def _wait_for_ack(self, spi, ack_val: int, timeout: int) -> None:
    timeout_s = max(MIN_ACK_TIMEOUT_MS, timeout) * 1e-3

    start = time.monotonic()
    while (timeout == 0) or ((time.monotonic() - start) < timeout_s):
      dat = spi.xfer2(b"\x12")[0]
      if dat == NACK:
        raise PandaSpiNackResponse
      elif dat == ack_val:
        return

    raise PandaSpiMissingAck

  def _transfer(self, spi, endpoint: int, data, timeout: int, max_rx_len: int = 1000) -> bytes:
    logging.debug("starting transfer: endpoint=%d, max_rx_len=%d", endpoint, max_rx_len)
    logging.debug("==============================================")

    exc = PandaSpiException()
    for n in range(MAX_XFER_RETRY_COUNT):
      logging.debug("\ntry #%d", n+1)
      try:
        logging.debug("- send header")
        packet = struct.pack("<BBHH", SYNC, endpoint, len(data), max_rx_len)
        packet += bytes([reduce(lambda x, y: x^y, packet) ^ CHECKSUM_START])
        spi.xfer2(packet)

        logging.debug("- waiting for header ACK")
        self._wait_for_ack(spi, HACK, timeout)

        # send data
        logging.debug("- sending data")
        packet = bytes([*data, self._calc_checksum(data)])
        spi.xfer2(packet)

        logging.debug("- waiting for data ACK")
        self._wait_for_ack(spi, DACK, timeout)

        # get response length, then response
        response_len_bytes = bytes(spi.xfer2(b"\x00" * 2))
        response_len = struct.unpack("<H", response_len_bytes)[0]
        if response_len > max_rx_len:
          raise PandaSpiException("response length greater than max")

        logging.debug("- receiving response")
        dat = bytes(spi.xfer2(b"\x00" * (response_len + 1)))
        if self._calc_checksum([DACK, *response_len_bytes, *dat]) != 0:
          raise PandaSpiBadChecksum

        return dat[:-1]
      except PandaSpiException as e:
        exc = e
        logging.debug("SPI transfer failed, %d retries left", n, exc_info=True)
    raise exc

  # libusb1 functions
  def close(self):
    self.dev.close()

  def controlWrite(self, request_type: int, request: int, value: int, index: int, data, timeout: int = TIMEOUT):
    with self.dev.acquire() as spi:
      return self._transfer(spi, 0, struct.pack("<BHHH", request, value, index, 0), timeout)

  def controlRead(self, request_type: int, request: int, value: int, index: int, length: int, timeout: int = TIMEOUT):
    with self.dev.acquire() as spi:
      return self._transfer(spi, 0, struct.pack("<BHHH", request, value, index, length), timeout)

  # TODO: implement these properly
  def bulkWrite(self, endpoint: int, data: List[int], timeout: int = TIMEOUT) -> int:
    with self.dev.acquire() as spi:
      for x in range(math.ceil(len(data) / USB_MAX_SIZE)):
        self._transfer(spi, endpoint, data[USB_MAX_SIZE*x:USB_MAX_SIZE*(x+1)], timeout)
      return len(data)

  def bulkRead(self, endpoint: int, length: int, timeout: int = TIMEOUT) -> bytes:
    ret: List[int] = []
    with self.dev.acquire() as spi:
      for _ in range(math.ceil(length / USB_MAX_SIZE)):
        d = self._transfer(spi, endpoint, [], timeout, max_rx_len=USB_MAX_SIZE)
        ret += d
        if len(d) < USB_MAX_SIZE:
          break
    return bytes(ret)


class STBootloaderSPIHandle(BaseSTBootloaderHandle):
  """
    Implementation of the STM32 SPI bootloader protocol described in:
    https://www.st.com/resource/en/application_note/an4286-spi-protocol-used-in-the-stm32-bootloader-stmicroelectronics.pdf
  """

  SYNC = 0x5A
  ACK = 0x79
  NACK = 0x1F

  def __init__(self):
    self.dev = SpiDevice(speed=1000000)

    # say hello
    try:
      with self.dev.acquire() as spi:
        spi.xfer([self.SYNC, ])
        try:
          self._get_ack(spi)
        except PandaSpiNackResponse:
          # NACK ok here, will only ACK the first time
          pass

      self._mcu_type = MCU_TYPE_BY_IDCODE[self.get_chip_id()]
    except PandaSpiException:
      raise PandaSpiException("failed to connect to panda")  # pylint: disable=W0707

  def _get_ack(self, spi, timeout=1.0):
    data = 0x00
    start_time = time.monotonic()
    while data not in (self.ACK, self.NACK) and (time.monotonic() - start_time < timeout):
      data = spi.xfer([0x00, ])[0]
      time.sleep(0.001)
    spi.xfer([self.ACK, ])

    if data == self.NACK:
      raise PandaSpiNackResponse
    elif data != self.ACK:
      raise PandaSpiMissingAck

  def _cmd(self, cmd: int, data: Optional[List[bytes]] = None, read_bytes: int = 0, predata=None) -> bytes:
    ret = b""
    with self.dev.acquire() as spi:
      # sync + command
      spi.xfer([self.SYNC, ])
      spi.xfer([cmd, cmd ^ 0xFF])
      self._get_ack(spi)

      # "predata" - for commands that send the first data without a checksum
      if predata is not None:
        spi.xfer(predata)
        self._get_ack(spi)

      # send data
      if data is not None:
        for d in data:
          if predata is not None:
            spi.xfer(d + self._checksum(predata + d))
          else:
            spi.xfer(d + self._checksum(d))
          self._get_ack(spi, timeout=20)

      # receive
      if read_bytes > 0:
        ret = spi.xfer([0x00, ]*(read_bytes + 1))[1:]
        if data is None or len(data) == 0:
          self._get_ack(spi)

    return bytes(ret)

  def _checksum(self, data: bytes) -> bytes:
    if len(data) == 1:
      ret = data[0] ^ 0xFF
    else:
      ret = reduce(lambda a, b: a ^ b, data)
    return bytes([ret, ])

  # *** Bootloader commands ***

  def read(self, address: int, length: int):
    data = [struct.pack('>I', address), struct.pack('B', length - 1)]
    return self._cmd(0x11, data=data, read_bytes=length)

  def get_chip_id(self) -> int:
    r = self._cmd(0x02, read_bytes=3)
    assert r[0] == 1  # response length - 1
    return ((r[1] << 8) + r[2])

  def go_cmd(self, address: int) -> None:
    self._cmd(0x21, data=[struct.pack('>I', address), ])

  # *** helpers ***

  def get_uid(self):
    dat = self.read(McuType.H7.config.uid_address, 12)
    return binascii.hexlify(dat).decode()

  def erase_sector(self, sector: int):
    p = struct.pack('>H', 0)  # number of sectors to erase
    d = struct.pack('>H', sector)
    self._cmd(0x44, data=[d, ], predata=p)

  # *** PandaDFU API ***

  def erase_app(self):
    self.erase_sector(1)

  def erase_bootstub(self):
    self.erase_sector(0)

  def get_mcu_type(self):
    return self._mcu_type

  def clear_status(self):
    pass

  def close(self):
    self.dev.close()

  def program(self, address, dat):
    bs = 256  # max block size for writing to flash over SPI
    dat += b"\xFF" * ((bs - len(dat)) % bs)
    for i in range(0, len(dat) // bs):
      block = dat[i * bs:(i + 1) * bs]
      self._cmd(0x31, data=[
        struct.pack('>I', address + i*bs),
        bytes([len(block) - 1]) + block,
      ])

  def jump(self, address):
    self.go_cmd(self._mcu_type.config.bootstub_address)
