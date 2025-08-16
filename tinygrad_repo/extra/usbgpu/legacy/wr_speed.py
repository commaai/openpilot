import array, time, ctypes, struct, random
from hexdump import hexdump
from tinygrad.runtime.support.usb import ASMController, WriteOp
from tinygrad.runtime.autogen import pci
from tinygrad.helpers import Timing
from tinygrad import Device

usb = ASMController()

xxx = (ctypes.c_uint8 * 4096)()
dfg = random.randint(0, 255)
for i in range(len(xxx)): xxx[i] = dfg

print(dfg, usb.read(0xf000, 0x10))

with Timing():
  for i in range(64): usb.scsi_write(xxx)

with Timing():
  for i in range(64): usb.read(0xf000, 0x1000)

exit(0)
