#!/usr/bin/env python3
import sys
import time
import signal
import serial
import struct
import requests
import urllib.parse
from datetime import datetime
from typing import List, Optional, Tuple

from cereal import messaging
from common.params import Params
from system.swaglog import cloudlog
from system.hardware import TICI
from common.gpio import gpio_init, gpio_set
from system.hardware.tici.pins import GPIO

UBLOX_TTY = "/dev/ttyHS0"

UBLOX_ACK = b"\xb5\x62\x05\x01\x02\x00"
UBLOX_NACK = b"\xb5\x62\x05\x00\x02\x00"
UBLOX_SOS_ACK = b"\xb5\x62\x09\x14\x08\x00\x02\x00\x00\x00\x01\x00\x00\x00"
UBLOX_SOS_NACK = b"\xb5\x62\x09\x14\x08\x00\x02\x00\x00\x00\x00\x00\x00\x00"
UBLOX_BACKUP_RESTORE_MSG = b"\xb5\x62\x09\x14\x08\x00\x03"
UBLOX_ASSIST_ACK = b"\xb5\x62\x13\x60\x08\x00"

GPS_RES_TRACK = 0x10
GPS_MAX_TRACK = 0x12
GLONASS_RES_TRACK = 0x04
GLONASS_MAX_TRACK = 0x08
QZSS_RES_TRACK = 0x00
QZSS_MAX_TRACK = 0x02

def set_power(enabled: bool) -> None:
  gpio_init(GPIO.UBLOX_SAFEBOOT_N, True)
  gpio_init(GPIO.UBLOX_PWR_EN, True)
  gpio_init(GPIO.UBLOX_RST_N, True)

  gpio_set(GPIO.UBLOX_SAFEBOOT_N, True)
  gpio_set(GPIO.UBLOX_PWR_EN, enabled)
  gpio_set(GPIO.UBLOX_RST_N, enabled)

def add_ubx_checksum(msg: bytes) -> bytes:
  A = B = 0
  for b in msg[2:]:
    A = (A + b) % 256
    B = (B + A) % 256
  return msg + bytes([A, B])

def get_assistnow_messages(token: bytes) -> List[bytes]:
  # make request
  # TODO: implement adding the last known location
  r = requests.get("https://online-live2.services.u-blox.com/GetOnlineData.ashx", params=urllib.parse.urlencode({
    'token': token,
    'gnss': 'gps,glo',
    'datatype': 'eph,alm,aux',
  }, safe=':,'), timeout=5)
  assert r.status_code == 200, "Got invalid status code"
  dat = r.content

  # split up messages
  msgs = []
  while len(dat) > 0:
    assert dat[:2] == b"\xB5\x62"
    msg_len = 6 + (dat[5] << 8 | dat[4]) + 2
    msgs.append(dat[:msg_len])
    dat = dat[msg_len:]
  return msgs


class TTYPigeon():
  def __init__(self):
    self.tty = serial.VTIMESerial(UBLOX_TTY, baudrate=9600, timeout=0)

  def send(self, dat: bytes) -> None:
    self.tty.write(dat)

  def receive(self) -> bytes:
    dat = b''
    while len(dat) < 0x1000:
      d = self.tty.read(0x40)
      dat += d
      if len(d) == 0:
        break
    return dat

  def set_baud(self, baud: int) -> None:
    self.tty.baudrate = baud

  def wait_for_ack(self, ack: bytes = UBLOX_ACK, nack: bytes = UBLOX_NACK, timeout: float = 0.5) -> bool:
    dat = b''
    st = time.monotonic()
    while True:
      dat += self.receive()
      if ack in dat:
        cloudlog.debug("Received ACK from ublox")
        return True
      elif nack in dat:
        cloudlog.error("Received NACK from ublox")
        return False
      elif time.monotonic() - st > timeout:
        cloudlog.error("No response from ublox")
        raise TimeoutError('No response from ublox')
      time.sleep(0.001)

  def send_with_ack(self, dat: bytes, ack: bytes = UBLOX_ACK, nack: bytes = UBLOX_NACK) -> None:
    self.send(dat)
    self.wait_for_ack(ack, nack)

  def wait_for_backup_restore_status(self, timeout: float = 1.) -> int:
    dat = b''
    st = time.monotonic()
    while True:
      dat += self.receive()
      position = dat.find(UBLOX_BACKUP_RESTORE_MSG)
      if position >= 0 and len(dat) >= position + 11:
        return dat[position + 10]
      elif time.monotonic() - st > timeout:
        cloudlog.error("No backup restore response from ublox")
        raise TimeoutError('No response from ublox')
      time.sleep(0.001)

  def reset_device(self) -> bool:
    # deleting the backup does not always work on first try (mostly on second try)
    for _ in range(5):
      # device cold start
      self.send(b"\xb5\x62\x06\x04\x04\x00\xff\xff\x00\x00\x0c\x5d")
      time.sleep(1) # wait for cold start
      init_baudrate(self)

      # clear configuration
      self.send_with_ack(b"\xb5\x62\x06\x09\x0d\x00\x00\x00\x1f\x1f\x00\x00\x00\x00\x00\x00\x00\x00\x17\x71\x5b")

      # clear flash memory (almanac backup)
      self.send_with_ack(b"\xB5\x62\x09\x14\x04\x00\x01\x00\x00\x00\x22\xf0")

      # try restoring backup to verify it got deleted
      self.send(b"\xB5\x62\x09\x14\x00\x00\x1D\x60")
      # 1: failed to restore, 2: could restore, 3: no backup
      status = self.wait_for_backup_restore_status()
      if status == 1 or status == 3:
        return True
    return False

  def get_gnss_config(self):
    # UBX-CFG-GNSS pull request
    hdr = b'\xb5\x62\x06\x3E'
    self.send(hdr + b'\x00\x00\x44\xD2')
    while True:
      recv = self.receive()
      if hdr in recv:
        return hdr + recv.split(hdr)[1]

  def set_gnss_config(self):
    # UBX-CFG-GNSS
    hdr  = b'\xB5\x62\x06\x3E'
    c    = b'\x00\x1c\x1c'
    msg  = struct.pack("I", GPS_MAX_TRACK<<16 | GPS_RES_TRACK<<8) # GPS GNSS CFG
    msg += b'\x01\x00\x01\x01'
    msg += b'\x01\x00\x00\x00' # GPS SBAS (disable)
    msg += b'\x00\x00\x01\x01'
    msg += b'\x02\x00\x00\x00' # GALILEO (disable)
    msg += b'\x00\x20\x00\x01'
    msg += b'\x03\x00\x00\x00' # BEIDOU (disable)
    msg += b'\x00\x00\x01\x01'
    msg += b'\x04\x00\x00\x00' # IMES (disable)
    msg += b'\x00\x00\x01\x01'
    msg += struct.pack("I", QZSS_MAX_TRACK<<16 | QZSS_RES_TRACK<<8 | 5) # QZSS (keep enabled with GPS)
    msg += b'\x01\x00\x01\x01'
    msg += struct.pack("I", GLONASS_MAX_TRACK<<16 | GLONASS_RES_TRACK<<8 | 6) # GLONASS
    msg += b'\x01\x00\x01\x01'
    gnss_cfg = hdr + struct.pack("H", len(msg) + 4) + c + struct.pack("b", int(len(msg)/8)) + msg
    self.send_with_ack(add_ubx_checksum(gnss_cfg))

    # UBX-CFG-CFG (saveMask)
    self.send_with_ack(b'\xB5\x62\x06\x09\x0C\x00\x00\x00\x00\x00\x1A\x00\x00\x00\x00\x00\x00\x00\x35\x5F')

  def verify_gnss_config(self):
    def check_data(data, resTrkCh, maxTrkCh, flags):
      return data[1] == resTrkCh and data[2] == maxTrkCh and data[4:8]  == flags

    def check_block_config(block):
      if block[0] == 0: # GPS
        return check_data(block, GPS_RES_TRACK, GPS_MAX_TRACK, b"\x01\x00\x01\x01")
      elif block[0] in [1,2,3,4]: # SBAS, GALILEO, BEIDOU, IMES
        return block[4] == 0 # disabled
      elif block[0] == 5: # QZSS
        return check_data(block, 0x00, 0x02, b"\x01\x00\x01\x01")
      elif block[0] == 6: # GLONASS
        return check_data(block, GLONASS_RES_TRACK, GLONASS_MAX_TRACK, b"\x01\x00\x01\x01")

    data = self.get_gnss_config()
    length = struct.unpack("H", data[4:6])[0]
    numConfigBlocks = data[9]
    configs = data[10:10+length-4]
    return all(check_block_config(configs[idx:idx+8]) for idx in range(0, numConfigBlocks*8, 8))

  def apply_gnss_config(self):
    if not self.verify_gnss_config():
      cloudlog.warning("set GNSS config")
      self.set_gnss_config()
      return True
    return False


def init_baudrate(pigeon: TTYPigeon):
  # ublox default setting on startup is 9600 baudrate
  pigeon.set_baud(9600)

  # $PUBX,41,1,0007,0003,460800,0*15\r\n
  pigeon.send(b"\x24\x50\x55\x42\x58\x2C\x34\x31\x2C\x31\x2C\x30\x30\x30\x37\x2C\x30\x30\x30\x33\x2C\x34\x36\x30\x38\x30\x30\x2C\x30\x2A\x31\x35\x0D\x0A")
  time.sleep(0.1)
  pigeon.set_baud(460800)


def initialize_pigeon(pigeon: TTYPigeon) -> bool:
  gnss_config_setting = False

  # try initializing a few times
  for _ in range(10):
    try:

      # setup port config
      pigeon.send_with_ack(b"\xb5\x62\x06\x00\x14\x00\x03\xFF\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x01\x00\x00\x00\x00\x00\x1E\x7F")
      pigeon.send_with_ack(b"\xb5\x62\x06\x00\x14\x00\x00\xFF\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x19\x35")
      pigeon.send_with_ack(b"\xb5\x62\x06\x00\x14\x00\x01\x00\x00\x00\xC0\x08\x00\x00\x00\x08\x07\x00\x01\x00\x01\x00\x00\x00\x00\x00\xF4\x80")
      pigeon.send_with_ack(b"\xb5\x62\x06\x00\x14\x00\x04\xFF\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1D\x85")
      pigeon.send_with_ack(b"\xb5\x62\x06\x00\x00\x00\x06\x18")
      pigeon.send_with_ack(b"\xb5\x62\x06\x00\x01\x00\x01\x08\x22")
      pigeon.send_with_ack(b"\xb5\x62\x06\x00\x01\x00\x03\x0A\x24")

      # UBX-CFG-RATE
      pigeon.send_with_ack(b"\xB5\x62\x06\x08\x06\x00\x64\x00\x01\x00\x00\x00\x79\x10")

      # UBX-CFG-NAV5
      pigeon.send_with_ack(b"\xB5\x62\x06\x24\x24\x00\x05\x00\x04\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x5A\x63")

      # UBX-CFG-ODO
      pigeon.send_with_ack(b"\xB5\x62\x06\x1E\x14\x00\x00\x00\x00\x00\x01\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x3C\x37")
      pigeon.send_with_ack(b"\xB5\x62\x06\x39\x08\x00\xFF\xAD\x62\xAD\x1E\x63\x00\x00\x83\x0C")
      pigeon.send_with_ack(b"\xB5\x62\x06\x23\x28\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x56\x24")

      # UBX-CFG-NAV5
      pigeon.send_with_ack(b"\xB5\x62\x06\x24\x00\x00\x2A\x84")
      pigeon.send_with_ack(b"\xB5\x62\x06\x23\x00\x00\x29\x81")
      pigeon.send_with_ack(b"\xB5\x62\x06\x1E\x00\x00\x24\x72")
      pigeon.send_with_ack(b"\xB5\x62\x06\x39\x00\x00\x3F\xC3")

      # UBX-CFG-MSG (set message rate)
      pigeon.send_with_ack(b"\xB5\x62\x06\x01\x03\x00\x01\x07\x01\x13\x51")
      pigeon.send_with_ack(b"\xB5\x62\x06\x01\x03\x00\x02\x15\x01\x22\x70")
      pigeon.send_with_ack(b"\xB5\x62\x06\x01\x03\x00\x02\x13\x01\x20\x6C")
      pigeon.send_with_ack(b"\xB5\x62\x06\x01\x03\x00\x0A\x09\x01\x1E\x70")
      pigeon.send_with_ack(b"\xB5\x62\x06\x01\x03\x00\x0A\x0B\x01\x20\x74")

      if pigeon.apply_gnss_config():
        if gnss_config_setting:
          cloudlog.error("Could not set GNSS config")
        if not gnss_config_setting:
          gnss_config_setting = True

          # GNSS config change needs a device reset, Software Reset
          pigeon.send(b"\xB5\x62\x06\x04\x04\x00\xFF\xFF\x01\x00\x0D\x5F")
          time.sleep(0.5)
          init_baudrate(pigeon)
          continue

      cloudlog.debug("pigeon configured")

      # try restoring almanac backup
      pigeon.send(b"\xB5\x62\x09\x14\x00\x00\x1D\x60")
      restore_status = pigeon.wait_for_backup_restore_status()
      if restore_status == 2:
        cloudlog.warning("almanac backup restored")
      elif restore_status == 3:
        cloudlog.warning("no almanac backup found")
      else:
        cloudlog.error(f"failed to restore almanac backup, status: {restore_status}")

      # sending time to ublox
      t_now = datetime.utcnow()
      if t_now >= datetime(2021, 6, 1):
        cloudlog.warning("Sending current time to ublox")

        # UBX-MGA-INI-TIME_UTC
        msg = add_ubx_checksum(b"\xB5\x62\x13\x40\x18\x00" + struct.pack("<BBBBHBBBBBxIHxxI",
          0x10,
          0x00,
          0x00,
          0x80,
          t_now.year,
          t_now.month,
          t_now.day,
          t_now.hour,
          t_now.minute,
          t_now.second,
          0,
          30,
          0
        ))
        pigeon.send_with_ack(msg, ack=UBLOX_ASSIST_ACK)

      # try getting AssistNow if we have a token
      token = Params().get('AssistNowToken')
      if token is not None:
        try:
          for msg in get_assistnow_messages(token):
            pigeon.send_with_ack(msg, ack=UBLOX_ASSIST_ACK)
          cloudlog.warning("AssistNow messages sent")
        except Exception:
          cloudlog.warning("failed to get AssistNow messages")

      cloudlog.warning("Pigeon GPS on!")
      break
    except TimeoutError:
      cloudlog.warning("Initialization failed, trying again!")
  else:
    cloudlog.warning("Failed to initialize pigeon")
    return False
  return True

def deinitialize_and_exit(pigeon: Optional[TTYPigeon]):
  cloudlog.warning("Storing almanac in ublox flash")

  if pigeon is not None:
    # controlled GNSS stop
    pigeon.send(b"\xB5\x62\x06\x04\x04\x00\x00\x00\x08\x00\x16\x74")

    # store almanac in flash
    pigeon.send(b"\xB5\x62\x09\x14\x04\x00\x00\x00\x00\x00\x21\xEC")
    try:
      if pigeon.wait_for_ack(ack=UBLOX_SOS_ACK, nack=UBLOX_SOS_NACK):
        cloudlog.warning("Done storing almanac")
      else:
        cloudlog.error("Error storing almanac")
    except TimeoutError:
      pass

  # turn off power and exit cleanly
  set_power(False)
  sys.exit(0)

def create_pigeon() -> Tuple[TTYPigeon, messaging.PubMaster]:
  pigeon = None

  # register exit handler
  signal.signal(signal.SIGINT, lambda sig, frame: deinitialize_and_exit(pigeon))
  pm = messaging.PubMaster(['ubloxRaw'])

  # power cycle ublox
  set_power(False)
  time.sleep(0.1)
  set_power(True)
  time.sleep(0.5)

  pigeon = TTYPigeon()
  return pigeon, pm

def run_receiving(pigeon: TTYPigeon, pm: messaging.PubMaster, duration: int = 0):

  start_time = time.monotonic()
  def end_condition():
    return True if duration == 0 else time.monotonic() - start_time < duration

  while end_condition():
    dat = pigeon.receive()
    if len(dat) > 0:
      if dat[0] == 0x00:
        cloudlog.warning("received invalid data from ublox, re-initing!")
        init_baudrate(pigeon)
        initialize_pigeon(pigeon)
        continue

      # send out to socket
      msg = messaging.new_message('ubloxRaw', len(dat))
      msg.ubloxRaw = dat[:]
      pm.send('ubloxRaw', msg)
    else:
      # prevent locking up a CPU core if ublox disconnects
      time.sleep(0.001)


def main():
  assert TICI, "unsupported hardware for pigeond"

  pigeon, pm = create_pigeon()
  init_baudrate(pigeon)
  initialize_pigeon(pigeon)

  # start receiving data
  run_receiving(pigeon, pm)

if __name__ == "__main__":
  main()
