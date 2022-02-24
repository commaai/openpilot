import usb1
import struct
import binascii
from .config import BOOTSTUB_ADDRESS, APP_ADDRESS_H7, APP_ADDRESS_FX, BLOCK_SIZE_H7, BLOCK_SIZE_FX, DEFAULT_H7_BOOTSTUB_FN, DEFAULT_BOOTSTUB_FN


MCU_TYPE_F2 = 0
MCU_TYPE_F4 = 1
MCU_TYPE_H7 = 2

# *** DFU mode ***
DFU_DNLOAD = 1
DFU_UPLOAD = 2
DFU_GETSTATUS = 3
DFU_CLRSTATUS = 4
DFU_ABORT = 6

class PandaDFU(object):
  def __init__(self, dfu_serial):
    context = usb1.USBContext()
    for device in context.getDeviceList(skip_on_error=True):
      if device.getVendorID() == 0x0483 and device.getProductID() == 0xdf11:
        try:
          this_dfu_serial = device.open().getASCIIStringDescriptor(3)
        except Exception:
          continue
        if this_dfu_serial == dfu_serial or dfu_serial is None:
          self._mcu_type = self.get_mcu_type(device)
          self._handle = device.open()
          return
    raise Exception("failed to open " + dfu_serial if dfu_serial is not None else "DFU device")

  @staticmethod
  def list():
    context = usb1.USBContext()
    dfu_serials = []
    try:
      for device in context.getDeviceList(skip_on_error=True):
        if device.getVendorID() == 0x0483 and device.getProductID() == 0xdf11:
          try:
            dfu_serials.append(device.open().getASCIIStringDescriptor(3))
          except Exception:
            pass
    except Exception:
      pass
    return dfu_serials

  @staticmethod
  def st_serial_to_dfu_serial(st, mcu_type=MCU_TYPE_F4):
    if st is None or st == "none":
      return None
    uid_base = struct.unpack("H" * 6, bytes.fromhex(st))
    if mcu_type == MCU_TYPE_H7:
      return binascii.hexlify(struct.pack("!HHH", uid_base[1] + uid_base[5], uid_base[0] + uid_base[4], uid_base[3])).upper().decode("utf-8")
    else:
      return binascii.hexlify(struct.pack("!HHH", uid_base[1] + uid_base[5], uid_base[0] + uid_base[4] + 0xA, uid_base[3])).upper().decode("utf-8")

  # TODO: Find a way to detect F4 vs F2
  def get_mcu_type(self, dev):
    return MCU_TYPE_H7 if dev.getbcdDevice() == 512 else MCU_TYPE_F4

  def status(self):
    while 1:
      dat = self._handle.controlRead(0x21, DFU_GETSTATUS, 0, 0, 6)
      if dat[1] == 0:
        break

  def clear_status(self):
    # Clear status
    stat = self._handle.controlRead(0x21, DFU_GETSTATUS, 0, 0, 6)
    if stat[4] == 0xa:
      self._handle.controlRead(0x21, DFU_CLRSTATUS, 0, 0, 0)
    elif stat[4] == 0x9:
      self._handle.controlWrite(0x21, DFU_ABORT, 0, 0, b"")
      self.status()
    stat = str(self._handle.controlRead(0x21, DFU_GETSTATUS, 0, 0, 6))

  def erase(self, address):
    self._handle.controlWrite(0x21, DFU_DNLOAD, 0, 0, b"\x41" + struct.pack("I", address))
    self.status()

  def program(self, address, dat, block_size=None):
    if block_size is None:
      block_size = len(dat)

    # Set Address Pointer
    self._handle.controlWrite(0x21, DFU_DNLOAD, 0, 0, b"\x21" + struct.pack("I", address))
    self.status()

    # Program
    dat += b"\xFF" * ((block_size - len(dat)) % block_size)
    for i in range(0, len(dat) // block_size):
      ldat = dat[i * block_size:(i + 1) * block_size]
      print("programming %d with length %d" % (i, len(ldat)))
      self._handle.controlWrite(0x21, DFU_DNLOAD, 2 + i, 0, ldat)
      self.status()

  def program_bootstub(self, code_bootstub):
    self.clear_status()
    self.erase(BOOTSTUB_ADDRESS)
    if self._mcu_type == MCU_TYPE_H7:
      self.erase(APP_ADDRESS_H7)
      self.program(BOOTSTUB_ADDRESS, code_bootstub, BLOCK_SIZE_H7)
    else:
      self.erase(APP_ADDRESS_FX)
      self.program(BOOTSTUB_ADDRESS, code_bootstub, BLOCK_SIZE_FX)
    self.reset()

  def recover(self):
    fn = DEFAULT_H7_BOOTSTUB_FN if self._mcu_type == MCU_TYPE_H7 else DEFAULT_BOOTSTUB_FN

    with open(fn, "rb") as f:
      code = f.read()

    self.program_bootstub(code)

  def reset(self):
    # **** Reset ****
    self._handle.controlWrite(0x21, DFU_DNLOAD, 0, 0, b"\x21" + struct.pack("I", BOOTSTUB_ADDRESS))
    self.status()
    try:
      self._handle.controlWrite(0x21, DFU_DNLOAD, 2, 0, b"")
      _ = str(self._handle.controlRead(0x21, DFU_GETSTATUS, 0, 0, 6))
    except Exception:
      pass
