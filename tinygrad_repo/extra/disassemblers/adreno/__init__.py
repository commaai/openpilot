import ctypes
import os
import pathlib
import struct
from hexdump import hexdump

fxn = None
def disasm_raw(buf):
  global fxn
  if fxn is None:
    shared = pathlib.Path(__file__).parent / "disasm.so"
    if not shared.is_file():
      os.system(f'cd {pathlib.Path(__file__).parent} && gcc -shared disasm-a3xx.c -o disasm.so')
    fxn = ctypes.CDLL(shared.as_posix())['disasm']
  fxn(buf, len(buf))

def disasm(buf):
  def _read_lib(off): return struct.unpack("I", buf[off:off+4])[0]

  image_offset = _read_lib(0xc0)
  image_size = _read_lib(0x100)
  disasm_raw(buf[image_offset:image_offset+image_size])
