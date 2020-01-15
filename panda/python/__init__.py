# python library to interface with panda
import datetime
import binascii
import struct
import hashlib
import socket
import usb1
import os
import time
import traceback
import subprocess
from .dfu import PandaDFU
from .esptool import ESPROM, CesantaFlasher  # noqa: F401
from .flash_release import flash_release  # noqa: F401
from .update import ensure_st_up_to_date  # noqa: F401
from .serial import PandaSerial  # noqa: F401
from .isotp import isotp_send, isotp_recv

__version__ = '0.0.9'

BASEDIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../")

DEBUG = os.getenv("PANDADEBUG") is not None

# *** wifi mode ***
def build_st(target, mkfile="Makefile", clean=True):
  from panda import BASEDIR

  clean_cmd = "make -f %s clean" % mkfile if clean else ":"
  cmd = 'cd %s && %s && make -f %s %s' % (os.path.join(BASEDIR, "board"), clean_cmd, mkfile, target)
  try:
    _ = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
  except subprocess.CalledProcessError:
    raise

def parse_can_buffer(dat):
  ret = []
  for j in range(0, len(dat), 0x10):
    ddat = dat[j:j+0x10]
    f1, f2 = struct.unpack("II", ddat[0:8])
    extended = 4
    if f1 & extended:
      address = f1 >> 3
    else:
      address = f1 >> 21
    dddat = ddat[8:8+(f2&0xF)]
    if DEBUG:
      print("  R %x: %s" % (address, binascii.hexlify(dddat)))
    ret.append((address, f2>>16, dddat, (f2>>4)&0xFF))
  return ret

class PandaWifiStreaming(object):
  def __init__(self, ip="192.168.0.10", port=1338):
    self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    self.sock.setblocking(0)
    self.ip = ip
    self.port = port
    self.kick()

  def kick(self):
    # must be called at least every 5 seconds
    self.sock.sendto("hello", (self.ip, self.port))

  def can_recv(self):
    ret = []
    while True:
      try:
        dat, addr = self.sock.recvfrom(0x200*0x10)
        if addr == (self.ip, self.port):
          ret += parse_can_buffer(dat)
      except socket.error as e:
        if e.errno != 35 and e.errno != 11:
          traceback.print_exc()
        break
    return ret

# stupid tunneling of USB over wifi and SPI
class WifiHandle(object):
  def __init__(self, ip="192.168.0.10", port=1337):
    self.sock = socket.create_connection((ip, port))

  def __recv(self):
    ret = self.sock.recv(0x44)
    length = struct.unpack("I", ret[0:4])[0]
    return ret[4:4+length]

  def controlWrite(self, request_type, request, value, index, data, timeout=0):
    # ignore data in reply, panda doesn't use it
    return self.controlRead(request_type, request, value, index, 0, timeout)

  def controlRead(self, request_type, request, value, index, length, timeout=0):
    self.sock.send(struct.pack("HHBBHHH", 0, 0, request_type, request, value, index, length))
    return self.__recv()

  def bulkWrite(self, endpoint, data, timeout=0):
    if len(data) > 0x10:
      raise ValueError("Data must not be longer than 0x10")
    self.sock.send(struct.pack("HH", endpoint, len(data))+data)
    self.__recv()  # to /dev/null

  def bulkRead(self, endpoint, length, timeout=0):
    self.sock.send(struct.pack("HH", endpoint, 0))
    return self.__recv()

  def close(self):
    self.sock.close()

# *** normal mode ***

class Panda(object):

  # matches cereal.car.CarParams.SafetyModel
  SAFETY_SILENT = 0
  SAFETY_HONDA_NIDEC = 1
  SAFETY_TOYOTA = 2
  SAFETY_ELM327 = 3
  SAFETY_GM = 4
  SAFETY_HONDA_BOSCH_GIRAFFE = 5
  SAFETY_FORD = 6
  SAFETY_CADILLAC = 7
  SAFETY_HYUNDAI = 8
  SAFETY_CHRYSLER = 9
  SAFETY_TESLA = 10
  SAFETY_SUBARU = 11
  SAFETY_MAZDA = 13
  SAFETY_VOLKSWAGEN = 15
  SAFETY_TOYOTA_IPAS = 16
  SAFETY_ALLOUTPUT = 17
  SAFETY_GM_ASCM = 18
  SAFETY_NOOUTPUT = 19
  SAFETY_HONDA_BOSCH_HARNESS = 20

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

  def __init__(self, serial=None, claim=True):
    self._serial = serial
    self._handle = None
    self.connect(claim)

  def close(self):
    self._handle.close()
    self._handle = None

  def connect(self, claim=True, wait=False):
    if self._handle != None:
      self.close()

    if self._serial == "WIFI":
      self._handle = WifiHandle()
      print("opening WIFI device")
      self.wifi = True
    else:
      context = usb1.USBContext()
      self._handle = None
      self.wifi = False

      while 1:
        try:
          for device in context.getDeviceList(skip_on_error=True):
            #print(device)
            if device.getVendorID() == 0xbbaa and device.getProductID() in [0xddcc, 0xddee]:
              try:
                this_serial = device.getSerialNumber()
              except Exception:
                continue
              if self._serial is None or this_serial == self._serial:
                self._serial = this_serial
                print("opening device", self._serial, hex(device.getProductID()))
                time.sleep(1)
                self.bootstub = device.getProductID() == 0xddee
                self.legacy = (device.getbcdDevice() != 0x2300)
                self._handle = device.open()
                if claim:
                  self._handle.claimInterface(0)
                  #self._handle.setInterfaceAltSetting(0, 0) #Issue in USB stack
                break
        except Exception as e:
          print("exception", e)
          traceback.print_exc()
        if wait == False or self._handle != None:
          break
        context = usb1.USBContext() #New context needed so new devices show up
    assert(self._handle != None)
    print("connected")

  def reset(self, enter_bootstub=False, enter_bootloader=False):
    # reset
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
        print("reconnecting is taking %d seconds..." % (i+1))
        try:
          dfu = PandaDFU(PandaDFU.st_serial_to_dfu_serial(self._serial))
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
      handle.bulkWrite(2, code[i:i+STEP])

    # reset
    print("flash: resetting")
    try:
      handle.controlWrite(Panda.REQUEST_IN, 0xd8, 0, 0, b'')
    except Exception:
      pass

  def flash(self, fn=None, code=None, reconnect=True):
    print("flash: main version is " + self.get_version())
    if not self.bootstub:
      self.reset(enter_bootstub=True)
    assert(self.bootstub)

    if fn is None and code is None:
      if self.legacy:
        fn = "obj/comma.bin"
        print("building legacy st code")
        build_st(fn, "Makefile.legacy")
      else:
        fn = "obj/panda.bin"
        print("building panda st code")
        build_st(fn)
      fn = os.path.join(BASEDIR, "board", fn)

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

  def recover(self, timeout=None):
    self.reset(enter_bootloader=True)
    t_start = time.time()
    while len(PandaDFU.list()) == 0:
      print("waiting for DFU...")
      time.sleep(0.1)
      if timeout is not None and (time.time() - t_start) > timeout:
        return False

    dfu = PandaDFU(PandaDFU.st_serial_to_dfu_serial(self._serial))
    dfu.recover()

    # reflash after recover
    self.connect(True, True)
    self.flash()
    return True

  @staticmethod
  def flash_ota_st():
    ret = os.system("cd %s && make clean && make ota" % (os.path.join(BASEDIR, "board")))
    time.sleep(1)
    return ret==0

  @staticmethod
  def flash_ota_wifi(release=False):
    release_str = "RELEASE=1" if release else ""
    ret = os.system("cd {} && make clean && {} make ota".format(os.path.join(BASEDIR, "boardesp"),release_str))
    time.sleep(1)
    return ret==0

  @staticmethod
  def list():
    context = usb1.USBContext()
    ret = []
    try:
      for device in context.getDeviceList(skip_on_error=True):
        if device.getVendorID() == 0xbbaa and device.getProductID() in [0xddcc, 0xddee]:
          try:
            ret.append(device.getSerialNumber())
          except Exception:
            continue
    except Exception:
      pass
    # TODO: detect if this is real
    #ret += ["WIFI"]
    return ret

  def call_control_api(self, msg):
    self._handle.controlWrite(Panda.REQUEST_OUT, msg, 0, 0, b'')

  # ******************* health *******************

  def health(self):
    dat = self._handle.controlRead(Panda.REQUEST_IN, 0xd2, 0, 0, 41)
    a = struct.unpack("IIIIIIIIBBBBBBBBB", dat)
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
      "fault_status": a[15],
      "power_save_enabled": a[16]
    }

  # ******************* control *******************

  def enter_bootloader(self):
    try:
      self._handle.controlWrite(Panda.REQUEST_OUT, 0xd1, 0, 0, b'')
    except Exception as e:
      print(e)
      pass

  def get_version(self):
    return self._handle.controlRead(Panda.REQUEST_IN, 0xd6, 0, 0, 0x40).decode('utf8')

  @staticmethod
  def get_signature_from_firmware(fn):
    f = open(fn, 'rb')
    f.seek(-128, 2)  # Seek from end of file
    return f.read(128)

  def get_signature(self):
    part_1 = self._handle.controlRead(Panda.REQUEST_IN, 0xd3, 0, 0, 0x40)
    part_2 = self._handle.controlRead(Panda.REQUEST_IN, 0xd4, 0, 0, 0x40)
    return bytes(part_1 + part_2)

  def get_type(self):
    return self._handle.controlRead(Panda.REQUEST_IN, 0xc1, 0, 0, 0x40)

  def is_white(self):
    return self.get_type() == Panda.HW_TYPE_WHITE_PANDA

  def is_grey(self):
    return self.get_type() == Panda.HW_TYPE_GREY_PANDA

  def is_black(self):
    return self.get_type() == Panda.HW_TYPE_BLACK_PANDA

  def is_uno(self):
    return self.get_type() == Panda.HW_TYPE_UNO

  def has_obd(self):
    return (self.is_uno() or self.is_black())

  def get_serial(self):
    dat = self._handle.controlRead(Panda.REQUEST_IN, 0xd0, 0, 0, 0x20)
    hashsig, calc_hash = dat[0x1c:], hashlib.sha1(dat[0:0x1c]).digest()[0:4]
    assert(hashsig == calc_hash)
    return [dat[0:0x10].decode("utf8"), dat[0x10:0x10+10].decode("utf8")]

  def get_secret(self):
    return self._handle.controlRead(Panda.REQUEST_IN, 0xd0, 1, 0, 0x10)

  # ******************* configuration *******************

  def set_usb_power(self, on):
    self._handle.controlWrite(Panda.REQUEST_OUT, 0xe6, int(on), 0, b'')

  def set_power_save(self, power_save_enabled=0):
    self._handle.controlWrite(Panda.REQUEST_OUT, 0xe7, int(power_save_enabled), 0, b'')

  def set_esp_power(self, on):
    self._handle.controlWrite(Panda.REQUEST_OUT, 0xd9, int(on), 0, b'')

  def esp_reset(self, bootmode=0):
    self._handle.controlWrite(Panda.REQUEST_OUT, 0xda, int(bootmode), 0, b'')
    time.sleep(0.2)

  def set_safety_mode(self, mode=SAFETY_SILENT):
    self._handle.controlWrite(Panda.REQUEST_OUT, 0xdc, mode, 0, b'')

  def set_can_forwarding(self, from_bus, to_bus):
    # TODO: This feature may not work correctly with saturated buses
    self._handle.controlWrite(Panda.REQUEST_OUT, 0xdd, from_bus, to_bus, b'')

  def set_gmlan(self, bus=2):
    # TODO: check panda type
    if bus is None:
      self._handle.controlWrite(Panda.REQUEST_OUT, 0xdb, 0, 0, b'')
    elif bus in [Panda.GMLAN_CAN2, Panda.GMLAN_CAN3]:
      self._handle.controlWrite(Panda.REQUEST_OUT, 0xdb, 1, bus, b'')

  def set_obd(self, obd):
    # TODO: check panda type
    self._handle.controlWrite(Panda.REQUEST_OUT, 0xdb, int(obd), 0, b'')

  def set_can_loopback(self, enable):
    # set can loopback mode for all buses
    self._handle.controlWrite(Panda.REQUEST_OUT, 0xe5, int(enable), 0, b'')

  def set_can_enable(self, bus_num, enable):
    # sets the can transciever enable pin
    self._handle.controlWrite(Panda.REQUEST_OUT, 0xf4, int(bus_num), int(enable), b'')

  def set_can_speed_kbps(self, bus, speed):
    self._handle.controlWrite(Panda.REQUEST_OUT, 0xde, bus, int(speed*10), b'')

  def set_uart_baud(self, uart, rate):
    self._handle.controlWrite(Panda.REQUEST_OUT, 0xe4, uart, int(rate/300), b'')

  def set_uart_parity(self, uart, parity):
    # parity, 0=off, 1=even, 2=odd
    self._handle.controlWrite(Panda.REQUEST_OUT, 0xe2, uart, parity, b'')

  def set_uart_callback(self, uart, install):
    self._handle.controlWrite(Panda.REQUEST_OUT, 0xe3, uart, int(install), b'')

  # ******************* can *******************

  def can_send_many(self, arr):
    snds = []
    transmit = 1
    extended = 4
    for addr, _, dat, bus in arr:
      assert len(dat) <= 8
      if DEBUG:
        print("  W %x: %s" % (addr, binascii.hexlify(dat)))
      if addr >= 0x800:
        rir = (addr << 3) | transmit | extended
      else:
        rir = (addr << 21) | transmit
      snd = struct.pack("II", rir, len(dat) | (bus << 4)) + dat
      snd = snd.ljust(0x10, b'\x00')
      snds.append(snd)

    while True:
      try:
        #print("DAT: %s"%b''.join(snds).__repr__())
        if self.wifi:
          for s in snds:
            self._handle.bulkWrite(3, s)
        else:
          self._handle.bulkWrite(3, b''.join(snds))
        break
      except (usb1.USBErrorIO, usb1.USBErrorOverflow):
        print("CAN: BAD SEND MANY, RETRYING")

  def can_send(self, addr, dat, bus):
    self.can_send_many([[addr, None, dat, bus]])

  def can_recv(self):
    dat = bytearray()
    while True:
      try:
        dat = self._handle.bulkRead(1, 0x10*256)
        break
      except (usb1.USBErrorIO, usb1.USBErrorOverflow):
        print("CAN: BAD RECV, RETRYING")
        time.sleep(0.1)
    return parse_can_buffer(dat)

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
      ret += self._handle.bulkWrite(2, struct.pack("B", port_number) + ln[i:i+0x20])
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
  def kline_wakeup(self):
    if DEBUG:
      print("kline wakeup...")
    self._handle.controlWrite(Panda.REQUEST_OUT, 0xf0, 0, 0, b'')
    if DEBUG:
      print("kline wakeup done")

  def kline_drain(self, bus=2):
    # drain buffer
    bret = bytearray()
    while True:
      ret = self._handle.controlRead(Panda.REQUEST_IN, 0xe0, bus, 0, 0x40)
      if len(ret) == 0:
        break
      elif DEBUG:
        print("kline drain: " + binascii.hexlify(ret))
      bret += ret
    return bytes(bret)

  def kline_ll_recv(self, cnt, bus=2):
    echo = bytearray()
    while len(echo) != cnt:
      ret = self._handle.controlRead(Panda.REQUEST_OUT, 0xe0, bus, 0, cnt-len(echo))
      if DEBUG and len(ret) > 0:
        print("kline recv: " + binascii.hexlify(ret))
      echo += ret
    return str(echo)

  def kline_send(self, x, bus=2, checksum=True):
    def get_checksum(dat):
      result = 0
      result += sum(map(ord, dat)) if isinstance(b'dat', str) else sum(dat)
      result = -result
      return struct.pack("B", result % 0x100)

    self.kline_drain(bus=bus)
    if checksum:
      x += get_checksum(x)
    for i in range(0, len(x), 0xf):
      ts = x[i:i+0xf]
      if DEBUG:
        print("kline send: " + binascii.hexlify(ts))
      self._handle.bulkWrite(2, bytes([bus]) + ts)
      echo = self.kline_ll_recv(len(ts), bus=bus)
      if echo != ts:
        print("**** ECHO ERROR %d ****" % i)
        print(binascii.hexlify(echo))
        print(binascii.hexlify(ts))
    assert echo == ts

  def kline_recv(self, bus=2):
    msg = self.kline_ll_recv(2, bus=bus)
    msg += self.kline_ll_recv(ord(msg[1])-2, bus=bus)
    return msg

  def send_heartbeat(self):
    self._handle.controlWrite(Panda.REQUEST_OUT, 0xf3, 0, 0, b'')

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
