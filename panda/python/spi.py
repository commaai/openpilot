import binascii
import ctypes
import os
import math
import time
import struct
import threading
from contextlib import contextmanager
from functools import reduce

from .base import BaseHandle, BaseSTBootloaderHandle, TIMEOUT
from .constants import McuType, MCU_TYPE_BY_IDCODE, USBPACKET_MAX_SIZE
from .utils import logger

# Constants
SYNC = 0x5A
HACK = 0x79
DACK = 0x85
NACK = 0x1F
CHECKSUM_START = 0xAB

MIN_ACK_TIMEOUT_MS = 100
MAX_XFER_RETRY_COUNT = 5

SPI_BUF_SIZE = 4096  # from panda/board/drivers/spi.h
XFER_SIZE = SPI_BUF_SIZE - 0x40 # give some room for SPI protocol overhead

DEV_PATH = "/dev/spidev0.0"


def crc8(data):
  crc = 0xFF    # standard init value
  poly = 0xD5   # standard crc8: x8+x7+x6+x4+x2+1
  size = len(data)
  for i in range(size - 1, -1, -1):
    crc ^= data[i]
    for _ in range(8):
      if ((crc & 0x80) != 0):
        crc = ((crc << 1) ^ poly) & 0xFF
      else:
        crc <<= 1
  return crc


class PandaSpiException(Exception):
  pass

class PandaProtocolMismatch(PandaSpiException):
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
SPI_DEVICES = {}
class SpiDevice:
  """Provides locked, thread-safe access to a panda's SPI interface.

  xfer2 returns a view valid until the next transfer; xfer returns an owned copy.
  """

  MAX_SPEED = 50000000  # max of the SDM845

  # Linux asm-generic ioctl ABI (including aarch64 and x86_64), linux/spi/spidev.h.
  SPI_IOC_RD_BITS_PER_WORD = 0x80016B03
  SPI_IOC_WR_MAX_SPEED_HZ = 0x40046B04
  SPI_IOC_MESSAGE_1 = 0x40206B00
  SPI_IOC_TRANSFER = struct.Struct('=QQIIHBBBBBB')
  MAX_TRANSFER_SIZE = 4096

  def __init__(self, speed=MAX_SPEED):
    try:
      import fcntl
    except ImportError as e:
      raise PandaSpiUnavailable("SPI requires Linux") from e

    assert speed <= self.MAX_SPEED
    if not os.path.exists(DEV_PATH):
      raise PandaSpiUnavailable(f"SPI device not found: {DEV_PATH}")

    self._ioctl = fcntl.ioctl
    self._flock = fcntl.flock
    self._lock_ex, self._lock_un = fcntl.LOCK_EX, fcntl.LOCK_UN
    with SPI_LOCK:
      if speed not in SPI_DEVICES:
        file = open(DEV_PATH, 'r+b', buffering=0)
        try:
          self._ioctl(file.fileno(), self.SPI_IOC_WR_MAX_SPEED_HZ, struct.pack('=I', speed))
          bits = self._ioctl(file.fileno(), self.SPI_IOC_RD_BITS_PER_WORD, b'\x00')[0]
        except BaseException:
          file.close()
          raise
        SPI_DEVICES[speed] = (file, bits)
      self._file, bits = SPI_DEVICES[speed]

    self._fd = self._file.fileno()
    self._buffer = bytearray(self.MAX_TRANSFER_SIZE)
    self._view = memoryview(self._buffer)
    address = ctypes.addressof(ctypes.c_char.from_buffer(self._buffer))
    # The kernel supports using the same buffer for transmit and receive.
    self._transfer = bytearray(self.SPI_IOC_TRANSFER.pack(address, address, 0, speed, 0, bits, 0, 0, 0, 0, 0))
    self._transfer_words = memoryview(self._transfer).cast('I')

  @contextmanager
  def acquire(self):
    with SPI_LOCK:
      self._flock(self._fd, self._lock_ex)
      try:
        yield self
      finally:
        self._flock(self._fd, self._lock_un)

  def fileno(self) -> int:
    return self._fd

  def close(self):
    pass  # Connections are shared and cached by speed.

  @classmethod
  def _check_length(cls, length: int):
    if not 0 < length <= cls.MAX_TRANSFER_SIZE:
      raise ValueError(f"SPI transfer length must be between 1 and {cls.MAX_TRANSFER_SIZE}")

  def xfer2(self, data) -> memoryview:
    length = len(data)
    if not 0 < length <= self.MAX_TRANSFER_SIZE:
      raise ValueError(f"SPI transfer length must be between 1 and {self.MAX_TRANSFER_SIZE}")
    self._view[:length] = bytes(data)
    self._transfer_words[4] = length  # len field at byte offset 16
    if self._ioctl(self._fd, self.SPI_IOC_MESSAGE_1, self._transfer) != length:
      raise OSError("Short SPI transfer")
    return self._view[:length]

  # Both operations use one SPI_IOC_MESSAGE(1) for the single-block calls in panda.
  def xfer(self, data) -> bytes:
    return bytes(self.xfer2(data))

  def readbytes(self, length: int) -> bytes:
    self._check_length(length)
    data = os.read(self.fileno(), length)
    if len(data) != length:
      raise OSError("Short SPI read")
    return data

  def writebytes(self, data):
    data = bytes(data)
    self._check_length(len(data))
    if os.write(self.fileno(), data) != len(data):
      raise OSError("Short SPI write")

class PandaSpiHandle(BaseHandle):
  """
  A class that mimics a libusb1 handle for panda SPI communications.
  """

  PROTOCOL_VERSION = 2
  HEADER = struct.Struct("<BBHH")

  def __init__(self) -> None:
    self.dev = SpiDevice()
    self.no_retry = "NO_RETRY" in os.environ

  # helpers
  def _calc_checksum(self, data: bytes) -> int:
    cksum = CHECKSUM_START
    for b in data:
      cksum ^= b
    return cksum

  def _wait_for_ack(self, spi, ack_val: int, timeout: int, tx: int, length: int = 1) -> bytes:
    timeout_s = max(MIN_ACK_TIMEOUT_MS, timeout) * 1e-3

    start = time.monotonic()
    while (timeout == 0) or ((time.monotonic() - start) < timeout_s):
      dat = spi.xfer2([tx, ] * length)
      if dat[0] == ack_val:
        return bytes(dat)
      elif dat[0] == NACK:
        raise PandaSpiNackResponse

    raise PandaSpiMissingAck

  def _transfer_spidev(self, spi, endpoint: int, data, timeout: int, max_rx_len: int = 1000, expect_disconnect: bool = False) -> bytes:
    max_rx_len = max(USBPACKET_MAX_SIZE, max_rx_len)

    logger.debug("- send header")
    packet = self.HEADER.pack(SYNC, endpoint, len(data), max_rx_len)
    packet += bytes([self._calc_checksum(packet), ])
    spi.xfer2(packet)

    logger.debug("- waiting for header ACK")
    self._wait_for_ack(spi, HACK, MIN_ACK_TIMEOUT_MS, 0x11)

    logger.debug("- sending data")
    packet = bytes([*data, self._calc_checksum(data)])
    spi.xfer2(packet)

    if expect_disconnect:
      logger.debug("- expecting disconnect, returning")
      return b""
    else:
      logger.debug("- waiting for data ACK")
      preread_len = USBPACKET_MAX_SIZE + 1  # read enough for a controlRead
      dat = self._wait_for_ack(spi, DACK, timeout, 0x13, length=3 + preread_len)

      # get response length, then response
      response_len = struct.unpack("<H", dat[1:3])[0]
      if response_len > max_rx_len:
        raise PandaSpiException(f"response length greater than max ({max_rx_len} {response_len})")

      # read rest
      remaining = (response_len + 1) - preread_len
      if remaining > 0:
        dat += bytes(spi.readbytes(remaining))

      dat = dat[:3 + response_len + 1]
      if self._calc_checksum(dat) != 0:
        raise PandaSpiBadChecksum

      return dat[3:-1]

  def _transfer(self, endpoint: int, data, timeout: int, max_rx_len: int = 1000, expect_disconnect: bool = False) -> bytes:
    logger.debug("starting transfer: endpoint=%d, max_rx_len=%d", endpoint, max_rx_len)
    logger.debug("==============================================")

    n = 0
    start_time = time.monotonic()
    exc = PandaSpiException()
    while (timeout == 0) or (time.monotonic() - start_time) < timeout*1e-3:
      n += 1
      logger.debug("\ntry #%d", n)
      with self.dev.acquire() as spi:
        try:
          return self._transfer_spidev(spi, endpoint, data, timeout, max_rx_len, expect_disconnect)
        except PandaSpiException as e:
          exc = e
          logger.debug("SPI transfer failed, retrying", exc_info=True)
          if self.no_retry:
            break

          # ensure slave is in a consistent state and ready for the next transfer
          # (e.g. slave TX buffer isn't stuck full)
          nack_cnt = 0
          attempts = 5
          while (nack_cnt <= 3) and (attempts > 0):
            attempts -= 1
            try:
              self._wait_for_ack(spi, NACK, MIN_ACK_TIMEOUT_MS, 0x11, length=XFER_SIZE//2)
              nack_cnt += 1
            except PandaSpiException:
              nack_cnt = 0

    raise exc

  def get_protocol_version(self) -> bytes:
    vers_str = b"VERSION"
    def _get_version(spi) -> bytes:
      spi.writebytes(vers_str)

      logger.debug("- waiting for echo")
      start = time.monotonic()
      while True:
        version_bytes = spi.readbytes(len(vers_str) + 2)
        if bytes(version_bytes).startswith(vers_str):
          break
        if (time.monotonic() - start) > 0.001:
          raise PandaSpiMissingAck

      rlen = struct.unpack("<H", bytes(version_bytes[-2:]))[0]
      if rlen > 1000:
        raise PandaSpiException("response length greater than max")

      # get response
      dat = spi.readbytes(rlen + 1)
      resp = dat[:-1]
      calculated_crc = crc8(bytes(version_bytes + resp))
      if calculated_crc != dat[-1]:
        raise PandaSpiBadChecksum
      return bytes(resp)

    exc = PandaSpiException()
    with self.dev.acquire() as spi:
      for _ in range(10):
        try:
          return _get_version(spi)
        except PandaSpiException as e:
          exc = e
          logger.debug("SPI get protocol version failed, retrying", exc_info=True)
    raise exc

  # libusb1 functions
  def close(self):
    self.dev.close()

  def controlWrite(self, request_type: int, request: int, value: int, index: int, data, timeout: int = TIMEOUT, expect_disconnect: bool = False):
    return self._transfer(0, struct.pack("<BHHH", request, value, index, 0), timeout, expect_disconnect=expect_disconnect)

  def controlRead(self, request_type: int, request: int, value: int, index: int, length: int, timeout: int = TIMEOUT):
    return self._transfer(0, struct.pack("<BHHH", request, value, index, length), timeout, max_rx_len=length)

  def bulkWrite(self, endpoint: int, data: bytes, timeout: int = TIMEOUT) -> int:
    mv = memoryview(data)
    for x in range(math.ceil(len(data) / XFER_SIZE)):
      self._transfer(endpoint, mv[XFER_SIZE*x:XFER_SIZE*(x+1)], timeout)
    return len(data)

  def bulkRead(self, endpoint: int, length: int, timeout: int = TIMEOUT) -> bytes:
    ret = b""
    for _ in range(math.ceil(length / XFER_SIZE)):
      d = self._transfer(endpoint, [], timeout, max_rx_len=XFER_SIZE)
      ret += d
      if len(d) < XFER_SIZE:
        break
    return ret


class STBootloaderSPIHandle(BaseSTBootloaderHandle):
  """
    Implementation of the STM32 SPI bootloader protocol described in:
    https://www.st.com/resource/en/application_note/an4286-spi-protocol-used-in-the-stm32-bootloader-stmicroelectronics.pdf

    NOTE: the bootloader's state machine is fragile and immediately gets into a bad state when
          sending any junk, e.g. when using the panda SPI protocol.
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
          self._get_ack(spi, 0.1)
        except (PandaSpiNackResponse, PandaSpiMissingAck):
          # NACK ok here, will only ACK the first time
          pass

      self._mcu_type = MCU_TYPE_BY_IDCODE[self.get_chip_id()]
    except PandaSpiException:
      raise PandaSpiException("failed to connect to panda") from None

  def _get_ack(self, spi, timeout=1.0):
    data = 0x00
    start_time = time.monotonic()
    while data not in (self.ACK, self.NACK) and (time.monotonic() - start_time < timeout):
      data = spi.xfer([0x00, ])[0]
      time.sleep(0)
    spi.xfer([self.ACK, ])

    if data == self.NACK:
      raise PandaSpiNackResponse
    elif data != self.ACK:
      raise PandaSpiMissingAck

  def _cmd_no_retry(self, cmd: int, data: list[bytes] | None = None, read_bytes: int = 0, predata=None) -> bytes:
    ret = b""
    with self.dev.acquire() as spi:
      # sync + command
      spi.xfer([self.SYNC, ])
      spi.xfer([cmd, cmd ^ 0xFF])
      self._get_ack(spi, timeout=0.01)

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

  def _cmd(self, cmd: int, data: list[bytes] | None = None, read_bytes: int = 0, predata=None) -> bytes:
    exc = PandaSpiException()
    for n in range(MAX_XFER_RETRY_COUNT):
      try:
        return self._cmd_no_retry(cmd, data, read_bytes, predata)
      except PandaSpiException as e:
        exc = e
        logger.debug("SPI transfer failed, %d retries left", MAX_XFER_RETRY_COUNT - n - 1, exc_info=True)
    raise exc

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

  def get_bootloader_id(self):
    return self.read(0x1FF1E7FE, 1)

  def get_chip_id(self) -> int:
    r = self._cmd(0x02, read_bytes=3)
    if r[0] != 1: # response length - 1
      raise PandaSpiException("incorrect response length")
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

  def get_mcu_type(self):
    return self._mcu_type

  def clear_status(self):
    pass

  def close(self):
    self.dev.close()

  def program(self, address, dat):
    bs = 256  # max block size for writing to flash over SPI
    dat += b"\xFF" * ((bs - len(dat)) % bs)
    for i in range(len(dat) // bs):
      block = dat[i * bs:(i + 1) * bs]
      self._cmd(0x31, data=[
        struct.pack('>I', address + i*bs),
        bytes([len(block) - 1]) + block,
      ])

  def jump(self, address):
    self.go_cmd(self._mcu_type.config.bootstub_address)
