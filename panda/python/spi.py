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
from .constants import McuType, MCU_TYPE_BY_IDCODE
from .utils import logger

# Protocol, see board/drivers/spi.h
SLAVE_READY = 0x3C
ACK = 0xC3
NACK = 0x5A
INFO_ENDPOINT = 0xF0

REQ_HEADER = struct.Struct("<BIHH")  # endpoint, id, tx_len, rx_len
RESP_HEADER = struct.Struct("<BIH")  # status, id, len
CRC = struct.Struct("<I")
RESP_OVERHEAD = RESP_HEADER.size + CRC.size
POLL = 0xFF  # filler while reading the response, never a valid request

MIN_TIMEOUT_MS = 100
SILENT_TIMEOUT = 1.0  # seconds, longer than the slowest request takes on the panda
MAX_XFER_RETRY_COUNT = 5  # ST bootloader

SPI_BUF_SIZE = 4096  # from panda/board/drivers/drivers.h
XFER_SIZE = SPI_BUF_SIZE - 0x40 # give some room for SPI protocol overhead
FIRST_READ_SIZE = 1024 - RESP_OVERHEAD  # reads are usually small, start small

DEV_PATH = "/dev/spidev0.0"


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


  def __init__(self) -> None:
    self.dev = SpiDevice()

  def _transfer_spidev(self, spi, endpoint: int, req_id: int, data, deadline: float, max_rx_len: int, expect_disconnect: bool) -> bytes | None:
    """One request and its response. Retries until the panda got the request and we got its response, or the deadline.
    Returns None if the panda went quiet, so the caller can let go of the bus for a bit."""
    # the response is always read in one transfer of rx_len
    rx_len = max_rx_len + RESP_OVERHEAD
    packet = REQ_HEADER.pack(endpoint, req_id, len(data), rx_len) + bytes(data)
    packet += CRC.pack(binascii.crc32(packet))
    poll = bytes([POLL, ]) * rx_len

    # when the panda last showed it's there
    alive = time.monotonic()
    while True:
      # send the request, until the panda was ready for it
      if spi.xfer2(packet)[0] == SLAVE_READY:
        alive = time.monotonic()
        if expect_disconnect:
          return b""

        # read until the response. if the panda is ready again, it never got the request
        while True:
          dat = spi.xfer(poll)
          if dat[0] == SLAVE_READY:
            alive = time.monotonic()
            break

          # a response that got cut off or corrupted isn't one, keep reading
          status, resp_id, response_len = RESP_HEADER.unpack_from(dat)
          end = RESP_HEADER.size + response_len
          if status in (ACK, NACK) and response_len <= max_rx_len and CRC.unpack_from(dat, end)[0] == binascii.crc32(dat[:end]):
            # otherwise it's the response to an old request, and we send ours again
            if resp_id == req_id:
              if status == NACK:
                raise PandaSpiNackResponse
              return dat[RESP_HEADER.size:end]
            alive = time.monotonic()
            break

          if time.monotonic() > min(deadline, alive + SILENT_TIMEOUT):
            break

      if time.monotonic() > deadline:
        raise PandaSpiMissingAck
      if time.monotonic() > alive + SILENT_TIMEOUT:
        return None

  def _transfer(self, endpoint: int, data, timeout: int, max_rx_len: int = 0, expect_disconnect: bool = False) -> bytes:
    logger.debug("transfer: endpoint=%d, max_rx_len=%d", endpoint, max_rx_len)

    # timeout of 0 means forever
    deadline = time.monotonic() + (max(MIN_TIMEOUT_MS, timeout) * 1e-3 if timeout != 0 else math.inf)
    nack = None
    nack_count = 0
    # random, so it's never the same as another request the panda just handled, even from another client.
    # it stays the same when we send the request again, so the panda never runs it twice
    req_id = int.from_bytes(os.urandom(4), "little")
    while True:
      try:
        with self.dev.acquire() as spi:
          dat = self._transfer_spidev(spi, endpoint, req_id, data, deadline, max_rx_len, expect_disconnect)
        if dat is not None:
          return dat
        # the panda went quiet. let go of the bus for a bit, someone else might need it
        logger.warning("SPI: panda isn't responding")
        time.sleep(0.01)
      except PandaSpiNackResponse as e:
        # the panda NACKs while its CAN TX buffers are full, don't busy wait on it
        if time.monotonic() > deadline:
          raise
        # a new id, since the panda would answer the same id with the same NACK
        req_id = int.from_bytes(os.urandom(4), "little")
        nack = e
        nack_count += 1
        if nack_count > 3:
          time.sleep(min(max(nack_count * 10, 200), 2000) * 1e-6)
      except PandaSpiMissingAck:
        # only raised once the deadline passed. if the panda was refusing, that's why
        if nack is not None:
          raise nack from None
        raise

  def get_info(self) -> bytes:
    # serial, HW type, bootstub
    return self._transfer(INFO_ENDPOINT, b"", MIN_TIMEOUT_MS, max_rx_len=14)

  # libusb1 functions
  def close(self):
    self.dev.close()

  def controlWrite(self, request_type: int, request: int, value: int, index: int, data, timeout: int = TIMEOUT, expect_disconnect: bool = False):
    return self._transfer(0, struct.pack("<BHHH", request, value, index, 0), timeout, expect_disconnect=expect_disconnect)

  def controlRead(self, request_type: int, request: int, value: int, index: int, length: int, timeout: int = TIMEOUT):
    return self._transfer(0, struct.pack("<BHHH", request, value, index, length), timeout, max_rx_len=length)

  def bulkWrite(self, endpoint: int, data: bytes, timeout: int = TIMEOUT) -> int:
    mv = memoryview(data)
    for x in range(0, len(data), XFER_SIZE):
      self._transfer(endpoint, mv[x:x+XFER_SIZE], timeout)
    return len(data)

  def bulkRead(self, endpoint: int, length: int, timeout: int = TIMEOUT) -> bytes:
    ret = b""
    while len(ret) < length:
      to_read = min(XFER_SIZE if len(ret) else FIRST_READ_SIZE, length - len(ret))
      d = self._transfer(endpoint, b"", timeout, max_rx_len=to_read)
      ret += d
      if len(d) < to_read:
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
      self._get_ack(spi, timeout=0.1)

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
