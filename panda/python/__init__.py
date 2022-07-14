# python library to interface with panda
import datetime
import struct
import hashlib
import usb1
import os
import time
import traceback
import sys
from functools import wraps
from typing import Optional
from .dfu import PandaDFU, MCU_TYPE_F2, MCU_TYPE_F4, MCU_TYPE_H7  # pylint: disable=import-error
from .flash_release import flash_release  # noqa pylint: disable=import-error
from .update import ensure_st_up_to_date  # noqa pylint: disable=import-error
from .serial import PandaSerial  # noqa pylint: disable=import-error
from .isotp import isotp_send, isotp_recv  # pylint: disable=import-error
from .config import DEFAULT_FW_FN, DEFAULT_H7_FW_FN  # noqa pylint: disable=import-error

__version__ = '0.0.10'

BASEDIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../")

DEBUG = os.getenv("PANDADEBUG") is not None

CANPACKET_HEAD_SIZE = 0x5
DLC_TO_LEN = [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 32, 48, 64]
LEN_TO_DLC = {length: dlc for (dlc, length) in enumerate(DLC_TO_LEN)}

def pack_can_buffer(arr):
  snds = [b'']
  idx = 0
  for address, _, dat, bus in arr:
    assert len(dat) in LEN_TO_DLC
    if DEBUG:
      print(f"  W 0x{address:x}: 0x{dat.hex()}")
    extended = 1 if address >= 0x800 else 0
    data_len_code = LEN_TO_DLC[len(dat)]
    header = bytearray(5)
    word_4b = address << 3 | extended << 2
    header[0] = (data_len_code << 4) | (bus << 1)
    header[1] = word_4b & 0xFF
    header[2] = (word_4b >> 8) & 0xFF
    header[3] = (word_4b >> 16) & 0xFF
    header[4] = (word_4b >> 24) & 0xFF
    snds[idx] += header + dat
    if len(snds[idx]) > 256: # Limit chunks to 256 bytes
      snds.append(b'')
      idx += 1

  #Apply counter to each 64 byte packet
  for idx in range(len(snds)):
    tx = b''
    counter = 0
    for i in range (0, len(snds[idx]), 63):
      tx += bytes([counter]) + snds[idx][i:i+63]
      counter += 1
    snds[idx] = tx
  return snds

def unpack_can_buffer(dat):
  ret = []
  counter = 0
  tail = bytearray()
  for i in range(0, len(dat), 64):
    if counter != dat[i]:
      print("CAN: LOST RECV PACKET COUNTER")
      break
    counter+=1
    chunk = tail + dat[i+1:i+64]
    tail = bytearray()
    pos = 0
    while pos<len(chunk):
      data_len = DLC_TO_LEN[(chunk[pos]>>4)]
      pckt_len = CANPACKET_HEAD_SIZE + data_len
      if pckt_len <= len(chunk[pos:]):
        header = chunk[pos:pos+CANPACKET_HEAD_SIZE]
        if len(header) < 5:
          print("CAN: MALFORMED USB RECV PACKET")
          break
        bus = (header[0] >> 1) & 0x7
        address = (header[4] << 24 | header[3] << 16 | header[2] << 8 | header[1]) >> 3
        returned = (header[1] >> 1) & 0x1
        rejected = header[1] & 0x1
        data = chunk[pos + CANPACKET_HEAD_SIZE:pos + CANPACKET_HEAD_SIZE + data_len]
        if returned:
          bus += 128
        if rejected:
          bus += 192
        if DEBUG:
          print(f"  R 0x{address:x}: 0x{data.hex()}")
        ret.append((address, 0, data, bus))
        pos += pckt_len
      else:
        tail = chunk[pos:]
        break
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
  SAFETY_HYUNDAI_HDA2 = 28

  SERIAL_DEBUG = 0
  SERIAL_ESP = 1
  SERIAL_LIN1 = 2
  SERIAL_LIN2 = 3

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

  CAN_PACKET_VERSION = 2
  HEALTH_PACKET_VERSION = 7
  HEALTH_STRUCT = struct.Struct("<IIIIIIIIBBBBBBBHBBBHIf")

  F2_DEVICES = (HW_TYPE_PEDAL, )
  F4_DEVICES = (HW_TYPE_WHITE_PANDA, HW_TYPE_GREY_PANDA, HW_TYPE_BLACK_PANDA, HW_TYPE_UNO, HW_TYPE_DOS)
  H7_DEVICES = (HW_TYPE_RED_PANDA, )

  CLOCK_SOURCE_MODE_DISABLED = 0
  CLOCK_SOURCE_MODE_FREE_RUNNING = 1

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

  FLAG_TESLA_POWERTRAIN = 1
  FLAG_TESLA_LONG_CONTROL = 2

  FLAG_CHRYSLER_RAM_DT = 1

  def __init__(self, serial: Optional[str] = None, claim: bool = True):
    self._serial = serial
    self._handle = None
    self.connect(claim)
    self._mcu_type = self.get_mcu_type()

  def close(self):
    self._handle.close()
    self._handle = None

  def connect(self, claim=True, wait=False):
    if self._handle is not None:
      self.close()

    context = usb1.USBContext()
    self._handle = None

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
              break
      except Exception as e:
        print("exception", e)
        traceback.print_exc()
      if not wait or self._handle is not None:
        break
      context = usb1.USBContext()  # New context needed so new devices show up
    assert(self._handle is not None)
    self.health_version, self.can_version = self.get_packets_versions()
    print("connected")

  def reset(self, enter_bootstub=False, enter_bootloader=False):
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
    if not enter_bootloader:
      self.reconnect()

  def reconnect(self):
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
  def flash_static(handle, code):
    # confirm flasher is present
    fr = handle.controlRead(Panda.REQUEST_IN, 0xb0, 0, 0, 0xc)
    assert fr[4:8] == b"\xde\xad\xd0\x0d"

    # unlock flash
    print("flash: unlocking")
    handle.controlWrite(Panda.REQUEST_IN, 0xb1, 0, 0, b'')

    # erase sectors 1 through 3
    print("flash: erasing")
    for i in range(1, 4):
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

  def flash(self, fn=DEFAULT_FW_FN, code=None, reconnect=True):
    if self._mcu_type == MCU_TYPE_H7 and fn == DEFAULT_FW_FN:
      fn = DEFAULT_H7_FW_FN
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
    Panda.flash_static(self._handle, code)

    # reconnect
    if reconnect:
      self.reconnect()

  def recover(self, timeout: Optional[int] = None, reset: bool = True) -> bool:
    if reset:
      self.reset(enter_bootstub=True)
      self.reset(enter_bootloader=True)

    t_start = time.time()
    while len(PandaDFU.list()) == 0:
      print("waiting for DFU...")
      time.sleep(0.1)
      if timeout is not None and (time.time() - t_start) > timeout:
        return False

    dfu = PandaDFU(PandaDFU.st_serial_to_dfu_serial(self._serial, self._mcu_type))
    dfu.recover()

    # reflash after recover
    self.connect(True, True)
    self.flash()
    return True

  @staticmethod
  def list():
    context = usb1.USBContext()
    ret = []
    try:
      for device in context.getDeviceList(skip_on_error=True):
        if device.getVendorID() == 0xbbaa and device.getProductID() in (0xddcc, 0xddee):
          try:
            ret.append(device.getSerialNumber())
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
      "can_rx_errs": a[3],
      "can_send_errs": a[4],
      "can_fwd_errs": a[5],
      "gmlan_send_errs": a[6],
      "faults": a[7],
      "ignition_line": a[8],
      "ignition_can": a[9],
      "controls_allowed": a[10],
      "gas_interceptor_detected": a[11],
      "car_harness_status": a[12],
      "usb_power_mode": a[13],
      "safety_mode": a[14],
      "safety_param": a[15],
      "fault_status": a[16],
      "power_save_enabled": a[17],
      "heartbeat_lost": a[18],
      "alternative_experience": a[19],
      "blocked_msg_cnt": a[20],
      "interrupt_load": a[21],
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
    return self._handle.controlRead(Panda.REQUEST_IN, 0xc1, 0, 0, 0x40)

  # Returns tuple with health packet version and CAN packet/USB packet version
  def get_packets_versions(self):
    dat = self._handle.controlRead(Panda.REQUEST_IN, 0xdd, 0, 0, 2)
    if dat:
      a = struct.unpack("BB", dat)
      return (a[0], a[1])
    else:
      return (0, 0)

  def is_white(self):
    return self.get_type() == Panda.HW_TYPE_WHITE_PANDA

  def is_grey(self):
    return self.get_type() == Panda.HW_TYPE_GREY_PANDA

  def is_black(self):
    return self.get_type() == Panda.HW_TYPE_BLACK_PANDA

  def is_pedal(self):
    return self.get_type() == Panda.HW_TYPE_PEDAL

  def is_uno(self):
    return self.get_type() == Panda.HW_TYPE_UNO

  def is_dos(self):
    return self.get_type() == Panda.HW_TYPE_DOS

  def is_red(self):
    return self.get_type() == Panda.HW_TYPE_RED_PANDA

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
    return (self.is_uno() or self.is_dos() or self.is_black() or self.is_red())

  def has_canfd(self) -> bool:
    return self.get_type() in Panda.H7_DEVICES

  def is_internal(self) -> bool:
    return self.get_type() in (Panda.HW_TYPE_UNO, Panda.HW_TYPE_DOS)

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

  def set_usb_power(self, on):
    self._handle.controlWrite(Panda.REQUEST_OUT, 0xe6, int(on), 0, b'')

  def set_power_save(self, power_save_enabled=0):
    self._handle.controlWrite(Panda.REQUEST_OUT, 0xe7, int(power_save_enabled), 0, b'')

  def enable_deepsleep(self):
    self._handle.controlWrite(Panda.REQUEST_OUT, 0xfb, 0, 0, b'')

  def set_esp_power(self, on):
    self._handle.controlWrite(Panda.REQUEST_OUT, 0xd9, int(on), 0, b'')

  def esp_reset(self, bootmode=0):
    self._handle.controlWrite(Panda.REQUEST_OUT, 0xda, int(bootmode), 0, b'')
    time.sleep(0.2)

  def set_safety_mode(self, mode=SAFETY_SILENT, param=0, disable_checks=True):
    self._handle.controlWrite(Panda.REQUEST_OUT, 0xdc, mode, param, b'')
    if disable_checks:
      self.set_heartbeat_disabled()
      self.set_power_save(0)

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

  # CAN FD and BRS status
  def get_canfd_status(self, bus):
    dat = self._handle.controlRead(Panda.REQUEST_IN, 0xfa, bus, 0, 2)
    if dat:
      a = struct.unpack("BB", dat)
      return (a[0], a[1])
    else:
      return (None, None)

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
      ret += self._handle.bulkWrite(2, struct.pack("B", port_number) + ln[i:i + 0x20])
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

  # ************** Clock Source **************
  def set_clock_source_mode(self, mode):
    self._handle.controlWrite(Panda.REQUEST_OUT, 0xf5, int(mode), 0, b'')

  # ****************** Siren *****************
  def set_siren(self, enabled):
    self._handle.controlWrite(Panda.REQUEST_OUT, 0xf6, int(enabled), 0, b'')

  # ****************** Debug *****************
  def set_green_led(self, enabled):
    self._handle.controlWrite(Panda.REQUEST_OUT, 0xf7, int(enabled), 0, b'')
