import array, time, ctypes, struct, random
from hexdump import hexdump
from tinygrad.runtime.support.usb import ASM24Controller, WriteOp, ScsiWriteOp
from tinygrad.runtime.autogen import pci
from tinygrad.helpers import Timing
from tinygrad import Device

usb = ASM24Controller()

def real_scsi_write():
  self.exec_ops([ScsiWriteOp(buf, lba)])

for i in range(256):
  xxx = (ctypes.c_uint8 * 4096)()
  dfg = random.randint(0, 255)
  for i in range(len(xxx)): xxx[i] = dfg
  # print(dfg, usb.read(0xf000, 0x10))
  st = time.perf_counter_ns()
  usb.scsi_write(bytes(xxx), lba=0x1000 + i)
  en = time.perf_counter_ns()
  print("mb/s is ", (0x1000) / (en - st) * 1e9 / 1024 / 1024)

exit(0)
