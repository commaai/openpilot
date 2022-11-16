import math
import struct
import spidev
import logging
from functools import reduce
from typing import List

# Constants
SYNC = 0x5A
HACK = 0x79
DACK = 0x85
NACK = 0x1F
CHECKSUM_START = 0xAB

MAX_RETRY_COUNT = 5

USB_MAX_SIZE = 0x40

# This mimics the handle given by libusb1 for easy interoperability
class SpiHandle:
  def __init__(self):
    self.spi = spidev.SpiDev()  # pylint: disable=c-extension-no-member
    self.spi.open(0, 0)

    self.spi.max_speed_hz = 30000000

  # helpers
  def _calc_checksum(self, data: List[int]) -> int:
    cksum = CHECKSUM_START
    for b in data:
      cksum ^= b
    return cksum

  def _transfer(self, endpoint: int, data, max_rx_len: int = 1000) -> bytes:
    logging.debug("starting transfer: endpoint=%d, max_rx_len=%d", endpoint, max_rx_len)
    logging.debug("==============================================")

    for n in range(MAX_RETRY_COUNT):
      logging.debug("\ntry #%d", n+1)
      try:
        logging.debug("- send header")
        packet = struct.pack("<BBHH", SYNC, endpoint, len(data), max_rx_len)
        packet += bytes([reduce(lambda x, y: x^y, packet) ^ CHECKSUM_START])
        self.spi.xfer2(packet)

        logging.debug("- waiting for ACK")
        # TODO: add timeout?
        dat = b"\x00"
        while dat[0] not in [HACK, NACK]:
          dat = self.spi.xfer2(b"\x12")

        if dat[0] == NACK:
          raise Exception("Got NACK response for header")

        # send data
        logging.debug("- sending data")
        packet = bytes([*data, self._calc_checksum(data)])
        self.spi.xfer2(packet)

        logging.debug("- waiting for ACK")
        dat = b"\x00"
        while dat[0] not in [DACK, NACK]:
          dat = self.spi.xfer2(b"\xab")

        if dat[0] == NACK:
          raise Exception("Got NACK response for data")

        # get response length, then response
        response_len_bytes = bytes(self.spi.xfer2(b"\x00" * 2))
        response_len = struct.unpack("<H", response_len_bytes)[0]

        logging.debug("- receiving response")
        dat = bytes(self.spi.xfer2(b"\x00" * (response_len + 1)))
        if self._calc_checksum([DACK, *response_len_bytes, *dat]) != 0:
          raise Exception("SPI got bad checksum")

        return dat[:-1]
      except Exception:
        logging.exception("SPI transfer failed, %d retries left", n)
    raise Exception(f"SPI transaction failed {MAX_RETRY_COUNT} times")

  # libusb1 functions
  def close(self):
    self.spi.close()

  def controlWrite(self, request_type: int, request: int, value: int, index: int, data, timeout: int = 0):
    return self._transfer(0, struct.pack("<BHHH", request, value, index, 0))

  def controlRead(self, request_type: int, request: int, value: int, index: int, length: int, timeout: int = 0):
    return self._transfer(0, struct.pack("<BHHH", request, value, index, length))

  # TODO: implement these properly
  def bulkWrite(self, endpoint: int, data: List[int], timeout: int = 0) -> int:
    for x in range(math.ceil(len(data) / USB_MAX_SIZE)):
      self._transfer(endpoint, data[USB_MAX_SIZE*x:USB_MAX_SIZE*(x+1)])
    return len(data)

  def bulkRead(self, endpoint: int, length: int, timeout: int = 0) -> bytes:
    ret: List[int] = []
    for _ in range(math.ceil(length / USB_MAX_SIZE)):
      d = self._transfer(endpoint, [], max_rx_len=USB_MAX_SIZE)
      ret += d
      if len(d) < USB_MAX_SIZE:
        break
    return bytes(ret)
