import usb1
import struct
import binascii

from .constants import McuType


# *** DFU mode ***
DFU_DNLOAD = 1
DFU_UPLOAD = 2
DFU_GETSTATUS = 3
DFU_CLRSTATUS = 4
DFU_ABORT = 6

class PandaDFU:
  def __init__(self, dfu_serial):
    self._handle = None
    context = usb1.USBContext()
    for device in context.getDeviceList(skip_on_error=True):
      if device.getVendorID() == 0x0483 and device.getProductID() == 0xdf11:
        try:
          this_dfu_serial = device.open().getASCIIStringDescriptor(3)
        except Exception:
          continue
        if this_dfu_serial == dfu_serial or dfu_serial is None:
          self._handle = device.open()
          self._mcu_type = self.get_mcu_type(device)
          break

    if self._handle is None:
      raise Exception(f"failed to open DFU device {dfu_serial}")

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
  def st_serial_to_dfu_serial(st, mcu_type=McuType.F4):
    if st is None or st == "none":
      return None
    uid_base = struct.unpack("H" * 6, bytes.fromhex(st))
    if mcu_type == McuType.H7:
      return binascii.hexlify(struct.pack("!HHH", uid_base[1] + uid_base[5], uid_base[0] + uid_base[4], uid_base[3])).upper().decode("utf-8")
    else:
      return binascii.hexlify(struct.pack("!HHH", uid_base[1] + uid_base[5], uid_base[0] + uid_base[4] + 0xA, uid_base[3])).upper().decode("utf-8")

  def get_mcu_type(self, dev) -> McuType:
    # TODO: Find a way to detect F4 vs F2
    # TODO: also check F4 BCD, don't assume in else
    return McuType.H7 if dev.getbcdDevice() == 512 else McuType.F4

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
    self.erase(self._mcu_type.config.bootstub_address)
    self.erase(self._mcu_type.config.app_address)
    self.program(self._mcu_type.config.bootstub_address, code_bootstub, self._mcu_type.config.block_size)
    self.reset()

  def recover(self):
    with open(self._mcu_type.config.bootstub_path, "rb") as f:
      code = f.read()
    self.program_bootstub(code)

  def reset(self):
    self._handle.controlWrite(0x21, DFU_DNLOAD, 0, 0, b"\x21" + struct.pack("I", self._mcu_type.config.bootstub_address))
    self.status()
    try:
      self._handle.controlWrite(0x21, DFU_DNLOAD, 2, 0, b"")
      _ = str(self._handle.controlRead(0x21, DFU_GETSTATUS, 0, 0, 6))
    except Exception:
      pass
