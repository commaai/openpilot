# python library to interface with panda
import os
import sys
import time
import usb1
import struct
import hashlib
import datetime
import traceback
import warnings
import logging
from functools import wraps
from typing import Optional
from itertools import accumulate

from .config import DEFAULT_FW_FN, DEFAULT_H7_FW_FN, SECTOR_SIZES_FX, SECTOR_SIZES_H7
from .dfu import PandaDFU, MCU_TYPE_F2, MCU_TYPE_F4, MCU_TYPE_H7
from .isotp import isotp_send, isotp_recv
from .spi import SpiHandle

__version__ = '0.0.10'

# setup logging
LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
logging.basicConfig(level=LOGLEVEL, format='%(message)s')


BASEDIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../")

DEBUG = os.getenv("PANDADEBUG") is not None

CAN_TRANSACTION_MAGIC = struct.pack("<I", 0x43414E2F)
USBPACKET_MAX_SIZE = 0x40
CANPACKET_HEAD_SIZE = 0x5
DLC_TO_LEN = [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 32, 48, 64]
LEN_TO_DLC = {length: dlc for (dlc, length) in enumerate(DLC_TO_LEN)}

def pack_can_buffer(arr):
  snds = [CAN_TRANSACTION_MAGIC]
  for address, _, dat, bus in arr:
    assert len(dat) in LEN_TO_DLC
    #logging.debug("  W 0x%x: 0x%s", address, dat.hex())

    extended = 1 if address >= 0x800 else 0
    data_len_code = LEN_TO_DLC[len(dat)]
    header = bytearray(5)
    word_4b = address << 3 | extended << 2
    header[0] = (data_len_code << 4) | (bus << 1)
    header[1] = word_4b & 0xFF
    header[2] = (word_4b >> 8) & 0xFF
    header[3] = (word_4b >> 16) & 0xFF
    header[4] = (word_4b >> 24) & 0xFF

    snds[-1] += header + dat
    if len(snds[-1]) > 256: # Limit chunks to 256 bytes
      snds.append(CAN_TRANSACTION_MAGIC)

  return snds

def unpack_can_buffer(dat):
  ret = []
  if len(dat) < len(CAN_TRANSACTION_MAGIC):
    return ret

  if dat[:len(CAN_TRANSACTION_MAGIC)] != CAN_TRANSACTION_MAGIC:
    logging.error("CAN: recv didn't start with magic")
    return ret

  dat = dat[len(CAN_TRANSACTION_MAGIC):]

  while len(dat) >= CANPACKET_HEAD_SIZE:
    data_len = DLC_TO_LEN[(dat[0]>>4)]

    header = dat[:CANPACKET_HEAD_SIZE]
    dat = dat[CANPACKET_HEAD_SIZE:]

    bus = (header[0] >> 1) & 0x7
    address = (header[4] << 24 | header[3] << 16 | header[2] << 8 | header[1]) >> 3

    if (header[1] >> 1) & 0x1:
      # returned
      bus += 128
    if header[1] & 0x1:
      # rejected
      bus += 192

    data = dat[:data_len]
    dat = dat[data_len:]

    #logging.debug("  R 0x%x: 0x%s", address, data.hex())

    ret.append((address, 0, data, bus))

  if len(dat) > 0:
    logging.error("CAN: malformed packet. leftover data")

  return ret

def ensure_health_packet_version(fn):
  @wraps(fn)
  def wrapper(self, *args, **kwargs):
    if self.health_version < self.HEALTH_PACKET_VERSION:
      raise RuntimeError("Panda firmware has outdated health packet definition. Reflash panda firmware.")
    elif self.health_version > self.HEALTH_PACKET_VERSION:
      raise RuntimeError("Panda python library has outdated health packet definition. Update panda python library.")
    return fn(self, *args, **kwargs)
  return wrapper

def ensure_can_packet_version(fn):
  @wraps(fn)
  def wrapper(self, *args, **kwargs):
    if self.can_version < self.CAN_PACKET_VERSION:
      raise RuntimeError("Panda firmware has outdated CAN packet definition. Reflash panda firmware.")
    elif self.can_version > self.CAN_PACKET_VERSION:
      raise RuntimeError("Panda python library has outdated CAN packet definition. Update panda python library.")
    return fn(self, *args, **kwargs)
  return wrapper

def ensure_can_health_packet_version(fn):
  @wraps(fn)
  def wrapper(self, *args, **kwargs):
    if self.can_health_version < self.CAN_HEALTH_PACKET_VERSION:
      raise RuntimeError("Panda firmware has outdated CAN health packet definition. Reflash panda firmware.")
    elif self.can_health_version > self.CAN_HEALTH_PACKET_VERSION:
      raise RuntimeError("Panda python library has outdated CAN health packet definition. Update panda python library.")
    return fn(self, *args, **kwargs)
  return wrapper

class ALTERNATIVE_EXPERIENCE:
  DEFAULT = 0
  DISABLE_DISENGAGE_ON_GAS = 1
  DISABLE_STOCK_AEB = 2
  RAISE_LONGITUDINAL_LIMITS_TO_ISO_MAX = 8

class Panda:

  # matches cereal.car.CarParams.SafetyModel
  SAFETY_SILENT = 0
  SAFETY_HONDA_NIDEC = 1
  SAFETY_TOYOTA = 2
  SAFETY_ELM327 = 3
  SAFETY_GM = 4
  SAFETY_HONDA_BOSCH_GIRAFFE = 5
  SAFETY_FORD = 6
  SAFETY_HYUNDAI = 8
  SAFETY_CHRYSLER = 9
  SAFETY_TESLA = 10
  SAFETY_SUBARU = 11
  SAFETY_MAZDA = 13
  SAFETY_NISSAN = 14
  SAFETY_VOLKSWAGEN_MQB = 15
  SAFETY_ALLOUTPUT = 17
  SAFETY_GM_ASCM = 18
  SAFETY_NOOUTPUT = 19
  SAFETY_HONDA_BOSCH = 20
  SAFETY_VOLKSWAGEN_PQ = 21
  SAFETY_SUBARU_LEGACY = 22
  SAFETY_HYUNDAI_LEGACY = 23
  SAFETY_HYUNDAI_COMMUNITY = 24
  SAFETY_STELLANTIS = 25
  SAFETY_FAW = 26
  SAFETY_BODY = 27
  SAFETY_HYUNDAI_CANFD = 28

  SERIAL_DEBUG = 0
  SERIAL_ESP = 1
  SERIAL_LIN1 = 2
  SERIAL_LIN2 = 3
  SERIAL_SOM_DEBUG = 4

  GMLAN_CAN2 = 1
  GMLAN_CAN3 = 2

  REQUEST_IN = usb1.ENDPOINT_IN | usb1.TYPE_VENDOR | usb1.RECIPIENT_DEVICE
  REQUEST_OUT = usb1.ENDPOINT_OUT | usb1.TYPE_VENDOR | usb1.RECIPIENT_DEVICE

  HW_TYPE_UNKNOWN = b'\x00'
  HW_TYPE_WHITE_PANDA = b'\x01'
  HW_TYPE_GREY_PANDA = b'\x02'
  HW_TYPE_BLACK_PANDA = b'\x03'
  HW_TYPE_PEDAL = b'\x04'
  HW_TYPE_UNO = b'\x05'
  HW_TYPE_DOS = b'\x06'
  HW_TYPE_RED_PANDA = b'\x07'
  HW_TYPE_RED_PANDA_V2 = b'\x08'
  HW_TYPE_TRES = b'\x09'

  CAN_PACKET_VERSION = 3
  HEALTH_PACKET_VERSION = 11
  CAN_HEALTH_PACKET_VERSION = 3
  HEALTH_STRUCT = struct.Struct("<IIIIIIIIIBBBBBBHBBBHfBB")
  CAN_HEALTH_STRUCT = struct.Struct("<BIBBBBBBBBIIIIIIHHBBB")

  F2_DEVICES = (HW_TYPE_PEDAL, )
  F4_DEVICES = (HW_TYPE_WHITE_PANDA, HW_TYPE_GREY_PANDA, HW_TYPE_BLACK_PANDA, HW_TYPE_UNO, HW_TYPE_DOS)
  H7_DEVICES = (HW_TYPE_RED_PANDA, HW_TYPE_RED_PANDA_V2, HW_TYPE_TRES)

  INTERNAL_DEVICES = (HW_TYPE_UNO, HW_TYPE_DOS)
  HAS_OBD = (HW_TYPE_BLACK_PANDA, HW_TYPE_UNO, HW_TYPE_DOS, HW_TYPE_RED_PANDA, HW_TYPE_RED_PANDA_V2, HW_TYPE_TRES)

  # first byte is for EPS scaling factor
  FLAG_TOYOTA_ALT_BRAKE = (1 << 8)
  FLAG_TOYOTA_STOCK_LONGITUDINAL = (2 << 8)

  FLAG_HONDA_ALT_BRAKE = 1
  FLAG_HONDA_BOSCH_LONG = 2
  FLAG_HONDA_NIDEC_ALT = 4
  FLAG_HONDA_RADARLESS = 8

  FLAG_HYUNDAI_EV_GAS = 1
  FLAG_HYUNDAI_HYBRID_GAS = 2
  FLAG_HYUNDAI_LONG = 4
  FLAG_HYUNDAI_CAMERA_SCC = 8
  FLAG_HYUNDAI_CANFD_HDA2 = 16
  FLAG_HYUNDAI_CANFD_ALT_BUTTONS = 32
  FLAG_HYUNDAI_ALT_LIMITS = 64

  FLAG_TESLA_POWERTRAIN = 1
  FLAG_TESLA_LONG_CONTROL = 2

  FLAG_VOLKSWAGEN_LONG_CONTROL = 1

  FLAG_CHRYSLER_RAM_DT = 1
  FLAG_CHRYSLER_RAM_HD = 2

  FLAG_SUBARU_GEN2 = 1

  FLAG_GM_HW_CAM = 1
  FLAG_GM_HW_CAM_LONG = 2

  def __init__(self, serial: Optional[str] = None, claim: bool = True, spi: bool = False, disable_checks: bool = True):
    self._serial = serial
    self._disable_checks = disable_checks

    self._handle = None
    self._bcd_device = None

    # connect and set mcu type
    self._spi = spi
    self.connect(claim)

  def close(self):
    self._handle.close()
    self._handle = None

  def connect(self, claim=True, wait=False):
    if self._handle is not None:
      self.close()
    self._handle = None

    if self._spi:
      self._handle = SpiHandle()

      # TODO implement
      self._serial = "SPIDEV"
      self.bootstub = False

    else:
      self.usb_connect(claim=claim, wait=wait)

    assert self._handle is not None
    self._mcu_type = self.get_mcu_type()
    self.health_version, self.can_version, self.can_health_version = self.get_packets_versions()
    print("connected")

    # disable openpilot's heartbeat checks
    if self._disable_checks:
      self.set_heartbeat_disabled()
      self.set_power_save(0)

  def usb_connect(self, claim=True, wait=False):
    context = usb1.USBContext()
    while 1:
      try:
        for device in context.getDeviceList(skip_on_error=True):
          if device.getVendorID() == 0xbbaa and device.getProductID() in (0xddcc, 0xddee):
            try:
              this_serial = device.getSerialNumber()
            except Exception:
              continue
            if self._serial is None or this_serial == self._serial:
              self._serial = this_serial
              print("opening device", self._serial, hex(device.getProductID()))
              self.bootstub = device.getProductID() == 0xddee
              self._handle = device.open()
              if sys.platform not in ("win32", "cygwin", "msys", "darwin"):
                self._handle.setAutoDetachKernelDriver(True)
              if claim:
                self._handle.claimInterface(0)
                # self._handle.setInterfaceAltSetting(0, 0)  # Issue in USB stack

              # bcdDevice wasn't always set to the hw type, ignore if it's the old constant
              bcd = device.getbcdDevice()
              if bcd is not None and bcd != 0x2300:
                self._bcd_device = bytearray([bcd >> 8, ])

              break
      except Exception as e:
        print("exception", e)
        traceback.print_exc()
      if not wait or self._handle is not None:
        break
      context = usb1.USBContext()  # New context needed so new devices show up

  def reset(self, enter_bootstub=False, enter_bootloader=False, reconnect=True):
    try:
      if enter_bootloader:
        self._handle.controlWrite(Panda.REQUEST_IN, 0xd1, 0, 0, b'')
      else:
        if enter_bootstub:
          self._handle.controlWrite(Panda.REQUEST_IN, 0xd1, 1, 0, b'')
        else:
          self._handle.controlWrite(Panda.REQUEST_IN, 0xd8, 0, 0, b'')
    except Exception:
      pass
    if not enter_bootloader and reconnect:
      self.reconnect()

  def reconnect(self):
    if self._handle is not None:
      self.close()
      time.sleep(1.0)

    success = False
    # wait up to 15 seconds
    for i in range(0, 15):
      try:
        self.connect()
        success = True
        break
      except Exception:
        print("reconnecting is taking %d seconds..." % (i + 1))
        try:
          dfu = PandaDFU(PandaDFU.st_serial_to_dfu_serial(self._serial, self._mcu_type))
          dfu.recover()
        except Exception:
          pass
        time.sleep(1.0)
    if not success:
      raise Exception("reconnect failed")



  @staticmethod
  def flash_static(handle, code, mcu_type):
    assert mcu_type is not None, "must set valid mcu_type to flash"

    # confirm flasher is present
    fr = handle.controlRead(Panda.REQUEST_IN, 0xb0, 0, 0, 0xc)
    assert fr[4:8] == b"\xde\xad\xd0\x0d"

    # determine sectors to erase
    apps_sectors_cumsum = accumulate(SECTOR_SIZES_H7[1:] if mcu_type == MCU_TYPE_H7 else SECTOR_SIZES_FX[1:])
    last_sector = next((i + 1 for i, v in enumerate(apps_sectors_cumsum) if v > len(code)), -1)
    assert last_sector >= 1, "Binary too small? No sector to erase."
    assert last_sector < 7, "Binary too large! Risk of overwriting provisioning chunk."

    # unlock flash
    print("flash: unlocking")
    handle.controlWrite(Panda.REQUEST_IN, 0xb1, 0, 0, b'')

    # erase sectors
    print(f"flash: erasing sectors 1 - {last_sector}")
    for i in range(1, last_sector + 1):
      handle.controlWrite(Panda.REQUEST_IN, 0xb2, i, 0, b'')

    # flash over EP2
    STEP = 0x10
    print("flash: flashing")
    for i in range(0, len(code), STEP):
      handle.bulkWrite(2, code[i:i + STEP])

    # reset
    print("flash: resetting")
    try:
      handle.controlWrite(Panda.REQUEST_IN, 0xd8, 0, 0, b'')
    except Exception:
      pass

  def flash(self, fn=None, code=None, reconnect=True):
    if not fn:
      fn = DEFAULT_H7_FW_FN if self._mcu_type == MCU_TYPE_H7 else DEFAULT_FW_FN
    assert os.path.isfile(fn)
    print("flash: main version is " + self.get_version())
    if not self.bootstub:
      self.reset(enter_bootstub=True)
    assert(self.bootstub)

    if code is None:
      with open(fn, "rb") as f:
        code = f.read()

    # get version
    print("flash: bootstub version is " + self.get_version())

    # do flash
    Panda.flash_static(self._handle, code, mcu_type=self._mcu_type)

    # reconnect
    if reconnect:
      self.reconnect()

  def recover(self, timeout: Optional[int] = None, reset: bool = True) -> bool:
    dfu_serial = PandaDFU.st_serial_to_dfu_serial(self._serial, self._mcu_type)

    if reset:
      self.reset(enter_bootstub=True)
      self.reset(enter_bootloader=True)

    if not self.wait_for_dfu(dfu_serial, timeout=timeout):
      return False

    dfu = PandaDFU(dfu_serial)
    dfu.recover()

    # reflash after recover
    self.connect(True, True)
    self.flash()
    return True

  @staticmethod
  def wait_for_dfu(dfu_serial: str, timeout: Optional[int] = None) -> bool:
    t_start = time.monotonic()
    while dfu_serial not in PandaDFU.list():
      print("waiting for DFU...")
      time.sleep(0.1)
      if timeout is not None and (time.monotonic() - t_start) > timeout:
        return False
    return True

  @staticmethod
  def list():
    context = usb1.USBContext()
    ret = []
    try:
      for device in context.getDeviceList(skip_on_error=True):
        if device.getVendorID() == 0xbbaa and device.getProductID() in (0xddcc, 0xddee):
          try:
            serial = device.getSerialNumber()
            if len(serial) == 24:
              ret.append(serial)
            else:
              warnings.warn(f"found device with panda descriptors but invalid serial: {serial}", RuntimeWarning)
          except Exception:
            continue
    except Exception:
      pass
    return ret

  def call_control_api(self, msg):
    self._handle.controlWrite(Panda.REQUEST_OUT, msg, 0, 0, b'')

  # ******************* health *******************

  @ensure_health_packet_version
  def health(self):
    dat = self._handle.controlRead(Panda.REQUEST_IN, 0xd2, 0, 0, self.HEALTH_STRUCT.size)
    a = self.HEALTH_STRUCT.unpack(dat)
    return {
      "uptime": a[0],
      "voltage": a[1],
      "current": a[2],
      "safety_tx_blocked": a[3],
      "safety_rx_invalid": a[4],
      "tx_buffer_overflow": a[5],
      "rx_buffer_overflow": a[6],
      "gmlan_send_errs": a[7],
      "faults": a[8],
      "ignition_line": a[9],
      "ignition_can": a[10],
      "controls_allowed": a[11],
      "gas_interceptor_detected": a[12],
      "car_harness_status": a[13],
      "safety_mode": a[14],
      "safety_param": a[15],
      "fault_status": a[16],
      "power_save_enabled": a[17],
      "heartbeat_lost": a[18],
      "alternative_experience": a[19],
      "interrupt_load": a[20],
      "fan_power": a[21],
      "safety_rx_checks_invalid": a[22],
    }

  @ensure_can_health_packet_version
  def can_health(self, can_number):
    LEC_ERROR_CODE = {
      0: "No error",
      1: "Stuff error",
      2: "Form error",
      3: "AckError",
      4: "Bit1Error",
      5: "Bit0Error",
      6: "CRCError",
      7: "NoChange",
    }
    dat = self._handle.controlRead(Panda.REQUEST_IN, 0xc2, int(can_number), 0, self.CAN_HEALTH_STRUCT.size)
    a = self.CAN_HEALTH_STRUCT.unpack(dat)
    return {
      "bus_off": a[0],
      "bus_off_cnt": a[1],
      "error_warning": a[2],
      "error_passive": a[3],
      "last_error": LEC_ERROR_CODE[a[4]],
      "last_stored_error": LEC_ERROR_CODE[a[5]],
      "last_data_error": LEC_ERROR_CODE[a[6]],
      "last_data_stored_error": LEC_ERROR_CODE[a[7]],
      "receive_error_cnt": a[8],
      "transmit_error_cnt": a[9],
      "total_error_cnt": a[10],
      "total_tx_lost_cnt": a[11],
      "total_rx_lost_cnt": a[12],
      "total_tx_cnt": a[13],
      "total_rx_cnt": a[14],
      "total_fwd_cnt": a[15],
      "can_speed": a[16],
      "can_data_speed": a[17],
      "canfd_enabled": a[18],
      "brs_enabled": a[19],
      "canfd_non_iso": a[20],
    }

  # ******************* control *******************

  def enter_bootloader(self):
    try:
      self._handle.controlWrite(Panda.REQUEST_OUT, 0xd1, 0, 0, b'')
    except Exception as e:
      print(e)

  def get_version(self):
    return self._handle.controlRead(Panda.REQUEST_IN, 0xd6, 0, 0, 0x40).decode('utf8')

  @staticmethod
  def get_signature_from_firmware(fn) -> bytes:
    f = open(fn, 'rb')
    f.seek(-128, 2)  # Seek from end of file
    return f.read(128)

  def get_signature(self):
    part_1 = self._handle.controlRead(Panda.REQUEST_IN, 0xd3, 0, 0, 0x40)
    part_2 = self._handle.controlRead(Panda.REQUEST_IN, 0xd4, 0, 0, 0x40)
    return bytes(part_1 + part_2)

  def get_type(self):
    ret = self._handle.controlRead(Panda.REQUEST_IN, 0xc1, 0, 0, 0x40)

    # bootstub doesn't implement this call, so fallback to bcdDevice
    invalid_type = self.bootstub and (ret is None or len(ret) != 1)
    if invalid_type and self._bcd_device is not None:
      ret = self._bcd_device

    return ret

  # Returns tuple with health packet version and CAN packet/USB packet version
  def get_packets_versions(self):
    dat = self._handle.controlRead(Panda.REQUEST_IN, 0xdd, 0, 0, 3)
    if dat and len(dat) == 3:
      a = struct.unpack("BBB", dat)
      return (a[0], a[1], a[2])
    else:
      return (0, 0, 0)

  def get_mcu_type(self):
    hw_type = self.get_type()
    if hw_type in Panda.F2_DEVICES:
      return MCU_TYPE_F2
    elif hw_type in Panda.F4_DEVICES:
      return MCU_TYPE_F4
    elif hw_type in Panda.H7_DEVICES:
      return MCU_TYPE_H7
    return None

  def has_obd(self):
    return self.get_type() in Panda.HAS_OBD

  def is_internal(self):
    return self.get_type() in Panda.INTERNAL_DEVICES

  def get_serial(self):
    dat = self._handle.controlRead(Panda.REQUEST_IN, 0xd0, 0, 0, 0x20)
    hashsig, calc_hash = dat[0x1c:], hashlib.sha1(dat[0:0x1c]).digest()[0:4]
    assert(hashsig == calc_hash)
    return [dat[0:0x10].decode("utf8"), dat[0x10:0x10 + 10].decode("utf8")]

  def get_usb_serial(self):
    return self._serial

  def get_secret(self):
    return self._handle.controlRead(Panda.REQUEST_IN, 0xd0, 1, 0, 0x10)

  # ******************* configuration *******************

  def set_power_save(self, power_save_enabled=0):
    self._handle.controlWrite(Panda.REQUEST_OUT, 0xe7, int(power_save_enabled), 0, b'')

  def enable_deepsleep(self):
    self._handle.controlWrite(Panda.REQUEST_OUT, 0xfb, 0, 0, b'')

  def set_esp_power(self, on):
    self._handle.controlWrite(Panda.REQUEST_OUT, 0xd9, int(on), 0, b'')

  def esp_reset(self, bootmode=0):
    self._handle.controlWrite(Panda.REQUEST_OUT, 0xda, int(bootmode), 0, b'')
    time.sleep(0.2)

  def set_safety_mode(self, mode=SAFETY_SILENT, param=0):
    self._handle.controlWrite(Panda.REQUEST_OUT, 0xdc, mode, param, b'')

  def set_gmlan(self, bus=2):
    # TODO: check panda type
    if bus is None:
      self._handle.controlWrite(Panda.REQUEST_OUT, 0xdb, 0, 0, b'')
    elif bus in (Panda.GMLAN_CAN2, Panda.GMLAN_CAN3):
      self._handle.controlWrite(Panda.REQUEST_OUT, 0xdb, 1, bus, b'')

  def set_obd(self, obd):
    # TODO: check panda type
    self._handle.controlWrite(Panda.REQUEST_OUT, 0xdb, int(obd), 0, b'')

  def set_can_loopback(self, enable):
    # set can loopback mode for all buses
    self._handle.controlWrite(Panda.REQUEST_OUT, 0xe5, int(enable), 0, b'')

  def set_can_enable(self, bus_num, enable):
    # sets the can transceiver enable pin
    self._handle.controlWrite(Panda.REQUEST_OUT, 0xf4, int(bus_num), int(enable), b'')

  def set_can_speed_kbps(self, bus, speed):
    self._handle.controlWrite(Panda.REQUEST_OUT, 0xde, bus, int(speed * 10), b'')

  def set_can_data_speed_kbps(self, bus, speed):
    self._handle.controlWrite(Panda.REQUEST_OUT, 0xf9, bus, int(speed * 10), b'')

  def set_canfd_non_iso(self, bus, non_iso):
    self._handle.controlWrite(Panda.REQUEST_OUT, 0xfc, bus, int(non_iso), b'')

  def set_uart_baud(self, uart, rate):
    self._handle.controlWrite(Panda.REQUEST_OUT, 0xe4, uart, int(rate / 300), b'')

  def set_uart_parity(self, uart, parity):
    # parity, 0=off, 1=even, 2=odd
    self._handle.controlWrite(Panda.REQUEST_OUT, 0xe2, uart, parity, b'')

  def set_uart_callback(self, uart, install):
    self._handle.controlWrite(Panda.REQUEST_OUT, 0xe3, uart, int(install), b'')

  # ******************* can *******************

  # The panda will NAK CAN writes when there is CAN congestion.
  # libusb will try to send it again, with a max timeout.
  # Timeout is in ms. If set to 0, the timeout is infinite.
  CAN_SEND_TIMEOUT_MS = 10

  @ensure_can_packet_version
  def can_send_many(self, arr, timeout=CAN_SEND_TIMEOUT_MS):
    snds = pack_can_buffer(arr)
    while True:
      try:
        for tx in snds:
          while True:
            bs = self._handle.bulkWrite(3, tx, timeout=timeout)
            tx = tx[bs:]
            if len(tx) == 0:
              break
            print("CAN: PARTIAL SEND MANY, RETRYING")
        break
      except (usb1.USBErrorIO, usb1.USBErrorOverflow):
        print("CAN: BAD SEND MANY, RETRYING")

  def can_send(self, addr, dat, bus, timeout=CAN_SEND_TIMEOUT_MS):
    self.can_send_many([[addr, None, dat, bus]], timeout=timeout)

  @ensure_can_packet_version
  def can_recv(self):
    dat = bytearray()
    while True:
      try:
        dat = self._handle.bulkRead(1, 16384) # Max receive batch size + 2 extra reserve frames
        break
      except (usb1.USBErrorIO, usb1.USBErrorOverflow):
        print("CAN: BAD RECV, RETRYING")
        time.sleep(0.1)
    return unpack_can_buffer(dat)

  def can_clear(self, bus):
    """Clears all messages from the specified internal CAN ringbuffer as
    though it were drained.

    Args:
      bus (int): can bus number to clear a tx queue, or 0xFFFF to clear the
        global can rx queue.

    """
    self._handle.controlWrite(Panda.REQUEST_OUT, 0xf1, bus, 0, b'')

  # ******************* isotp *******************

  def isotp_send(self, addr, dat, bus, recvaddr=None, subaddr=None):
    return isotp_send(self, dat, addr, bus, recvaddr, subaddr)

  def isotp_recv(self, addr, bus=0, sendaddr=None, subaddr=None):
    return isotp_recv(self, addr, bus, sendaddr, subaddr)

  # ******************* serial *******************

  def serial_read(self, port_number):
    ret = []
    while 1:
      lret = bytes(self._handle.controlRead(Panda.REQUEST_IN, 0xe0, port_number, 0, 0x40))
      if len(lret) == 0:
        break
      ret.append(lret)
    return b''.join(ret)

  def serial_write(self, port_number, ln):
    ret = 0
    for i in range(0, len(ln), 0x20):
      ret += self._handle.bulkWrite(2, struct.pack("B", port_number) + bytes(ln[i:i + 0x20], 'utf-8'))
    return ret

  def serial_clear(self, port_number):
    """Clears all messages (tx and rx) from the specified internal uart
    ringbuffer as though it were drained.

    Args:
      port_number (int): port number of the uart to clear.

    """
    self._handle.controlWrite(Panda.REQUEST_OUT, 0xf2, port_number, 0, b'')

  # ******************* kline *******************

  # pulse low for wakeup
  def kline_wakeup(self, k=True, l=True):
    assert k or l, "must specify k-line, l-line, or both"
    if DEBUG:
      print("kline wakeup...")
    self._handle.controlWrite(Panda.REQUEST_OUT, 0xf0, 2 if k and l else int(l), 0, b'')
    if DEBUG:
      print("kline wakeup done")

  def kline_5baud(self, addr, k=True, l=True):
    assert k or l, "must specify k-line, l-line, or both"
    if DEBUG:
      print("kline 5 baud...")
    self._handle.controlWrite(Panda.REQUEST_OUT, 0xf4, 2 if k and l else int(l), addr, b'')
    if DEBUG:
      print("kline 5 baud done")

  def kline_drain(self, bus=2):
    # drain buffer
    bret = bytearray()
    while True:
      ret = self._handle.controlRead(Panda.REQUEST_IN, 0xe0, bus, 0, 0x40)
      if len(ret) == 0:
        break
      elif DEBUG:
        print(f"kline drain: 0x{ret.hex()}")
      bret += ret
    return bytes(bret)

  def kline_ll_recv(self, cnt, bus=2):
    echo = bytearray()
    while len(echo) != cnt:
      ret = self._handle.controlRead(Panda.REQUEST_OUT, 0xe0, bus, 0, cnt - len(echo))
      if DEBUG and len(ret) > 0:
        print(f"kline recv: 0x{ret.hex()}")
      echo += ret
    return bytes(echo)

  def kline_send(self, x, bus=2, checksum=True):
    self.kline_drain(bus=bus)
    if checksum:
      x += bytes([sum(x) % 0x100])
    for i in range(0, len(x), 0xf):
      ts = x[i:i + 0xf]
      if DEBUG:
        print(f"kline send: 0x{ts.hex()}")
      self._handle.bulkWrite(2, bytes([bus]) + ts)
      echo = self.kline_ll_recv(len(ts), bus=bus)
      if echo != ts:
        print(f"**** ECHO ERROR {i} ****")
        print(f"0x{echo.hex()}")
        print(f"0x{ts.hex()}")
    assert echo == ts

  def kline_recv(self, bus=2, header_len=4):
    # read header (last byte is length)
    msg = self.kline_ll_recv(header_len, bus=bus)
    # read data (add one byte to length for checksum)
    msg += self.kline_ll_recv(msg[-1]+1, bus=bus)
    return msg

  def send_heartbeat(self, engaged=True):
    self._handle.controlWrite(Panda.REQUEST_OUT, 0xf3, engaged, 0, b'')

  # disable heartbeat checks for use outside of openpilot
  # sending a heartbeat will reenable the checks
  def set_heartbeat_disabled(self):
    self._handle.controlWrite(Panda.REQUEST_OUT, 0xf8, 0, 0, b'')

  # ******************* RTC *******************
  def set_datetime(self, dt):
    self._handle.controlWrite(Panda.REQUEST_OUT, 0xa1, int(dt.year), 0, b'')
    self._handle.controlWrite(Panda.REQUEST_OUT, 0xa2, int(dt.month), 0, b'')
    self._handle.controlWrite(Panda.REQUEST_OUT, 0xa3, int(dt.day), 0, b'')
    self._handle.controlWrite(Panda.REQUEST_OUT, 0xa4, int(dt.isoweekday()), 0, b'')
    self._handle.controlWrite(Panda.REQUEST_OUT, 0xa5, int(dt.hour), 0, b'')
    self._handle.controlWrite(Panda.REQUEST_OUT, 0xa6, int(dt.minute), 0, b'')
    self._handle.controlWrite(Panda.REQUEST_OUT, 0xa7, int(dt.second), 0, b'')

  def get_datetime(self):
    dat = self._handle.controlRead(Panda.REQUEST_IN, 0xa0, 0, 0, 8)
    a = struct.unpack("HBBBBBB", dat)
    return datetime.datetime(a[0], a[1], a[2], a[4], a[5], a[6])

  # ******************* IR *******************
  def set_ir_power(self, percentage):
    self._handle.controlWrite(Panda.REQUEST_OUT, 0xb0, int(percentage), 0, b'')

  # ******************* Fan ******************
  def set_fan_power(self, percentage):
    self._handle.controlWrite(Panda.REQUEST_OUT, 0xb1, int(percentage), 0, b'')

  def get_fan_rpm(self):
    dat = self._handle.controlRead(Panda.REQUEST_IN, 0xb2, 0, 0, 2)
    a = struct.unpack("H", dat)
    return a[0]

  # ****************** Phone *****************
  def set_phone_power(self, enabled):
    self._handle.controlWrite(Panda.REQUEST_OUT, 0xb3, int(enabled), 0, b'')

  # ****************** Siren *****************
  def set_siren(self, enabled):
    self._handle.controlWrite(Panda.REQUEST_OUT, 0xf6, int(enabled), 0, b'')
