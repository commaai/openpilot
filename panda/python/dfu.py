import os
import usb1
import struct
import binascii

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
          self._handle = device.open()
          self.legacy = "07*128Kg" in self._handle.getASCIIStringDescriptor(4)
          return
    raise Exception("failed to open " + dfu_serial)

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
  def st_serial_to_dfu_serial(st):
    if st is None or st == "none":
      return None
    uid_base = struct.unpack("H" * 6, bytes.fromhex(st))
    return binascii.hexlify(struct.pack("!HHH", uid_base[1] + uid_base[5], uid_base[0] + uid_base[4] + 0xA, uid_base[3])).upper().decode("utf-8")

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
    self.erase(0x8004000)
    self.erase(0x8000000)
    self.program(0x8000000, code_bootstub, 0x800)
    self.reset()

  def recover(self):
    from panda import BASEDIR, build_st
    if self.legacy:
      fn = "obj/bootstub.comma.bin"
      print("building legacy bootstub")
      build_st(fn, "Makefile.legacy")
    else:
      fn = "obj/bootstub.panda.bin"
      print("building panda bootstub")
      build_st(fn)
    fn = os.path.join(BASEDIR, "board", fn)

    with open(fn, "rb") as f:
      code = f.read()

    self.program_bootstub(code)

  def reset(self):
    # **** Reset ****
    self._handle.controlWrite(0x21, DFU_DNLOAD, 0, 0, b"\x21" + struct.pack("I", 0x8000000))
    self.status()
    try:
      self._handle.controlWrite(0x21, DFU_DNLOAD, 2, 0, b"")
      _ = str(self._handle.controlRead(0x21, DFU_GETSTATUS, 0, 0, 6))
    except Exception:
      pass
