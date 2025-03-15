#!/usr/bin/env python3
from pathlib import Path
from ctypes import *
import json
import collections
import numpy as np
import faulthandler
import struct
faulthandler.enable()

basedir = Path(__file__).resolve().parent

libane = None
aneregs = None
def init_libane():
  global libane, aneregs
  libane = cdll.LoadLibrary((basedir / "libane.dylib").as_posix())

  libane.ANE_Compile.argtypes = [c_char_p, c_int]
  libane.ANE_Compile.restype = c_void_p

  libane.ANE_TensorCreate.restype = c_void_p

  libane.ANE_TensorData.argtypes = [c_void_p]
  libane.ANE_TensorData.restype = POINTER(c_uint16)

  libane.ANE_Run.argtypes = [c_void_p]*4
  libane.ANE_Run.restype = c_int

  #libane.ANE_RegDebug.restype = c_char_p

  with open(basedir / "aneregs.json") as f:
    aneregs = json.load(f)

ANE_Struct = [
# aneTD.Header
  ("u32", 0x1C, "NextCommandOffset"),

# KernelDMASrc @ section @ 0x2C len 0xF4
  # reloc 0x2c-0x34?? = weights
  # u32[16] 0x34-0x74 = 0x80 | 1 if used
  # u32[16] 0x74-0xB4 = <channel data offset>
  # u32[16] 0xB4-0xF4 = <channel data length>

# Common @ section @ 0x128 len 0x3C (conv)
  ("u16", 0x128, "InputWidth"),
  ("u16", 0x12A, "InputHeight"),
  ("u16", 0x12C, "InputDepth"),

  ("u32", 0x130, "InputOutputType"),   # (OutputType * 0x10) | InputType
                                       # UInt8 = 0, Int8 = 1, Float16 = 2

  ("u32", 0x134, "InputChannels"),
  ("u32", 0x138, "OutputChannels"),

  ("u16", 0x13C, "OutputWidth"),
  ("u16", 0x13E, "OutputHeight"),
  ("u16", 0x140, "OutputDepth"),

  ("u16", 0x144, "KernelSize"),        # 0xa000 | (KernelHeight * 0x20) | KernelWidth
  ("u16", 0x146, "Padding"),           # 0x5000 | (PadTop * 0x40) | (PadLeft * 2)

  ("u16", 0x14C, "BatchSize"),

# TileDMASrc @ section @ 0x16C len 0x6C (input)
  # reloc 0x16c-0x174 = image
  ("u32", 0x178, "InputRowStride"),
  ("u32", 0x17C, "InputPlaneStride"),
  ("u32", 0x180, "InputDepthStride"),
  ("u32", 0x184, "InputBatchStride"),

  ("u8",  0x1A7, "InputInterleave"),

# L2 @ section @ 0x1E0 len 0x44
  # [0x1ec, 0x1f0, 0x1f4, 0x1f8, 0x214] = number of engines
  # [0x1f0, 0x1f4, 0x1f8, 0x214] = engines for inconv?
  # [0x21c, 0x220, 0x224] = engines for outconv?

# NE @ section @ 0x22c len 0xC (scaling)
  ("u16", 0x230, "BiasScalar"),
  ("u16", 0x232, "ScaleScalar"),

# section @ 0x240 len 0x10
  ("u16", 0x246, "NeuronType"),  # 0x10 = copy, 0x11 = ReLU, 0x12 = custom
  ("u32", 0x250, "PostScale"),

# TileDMADst @ section @ 0x258 len 0x18

# HandleTileDmaDstConfig
  # 0x258 -- *(uint *)(this + 0x334) = *(uint *)(this + 0x334) & 0xfffffc3f | 0xc0;
  #          (GetCacheHintRegisterValue & 0xf) << 6;
  ("u32", 0x25C, "OutputOffset"),  # offset into output buffer to write at?

  # 0x260 -- *(uint *)(this + 0x33c) = *(uint *)(this + 0x33c) & 0x3f | (int)uVar10 << 6;
  ("u32", 0x260, "OutputRowStride"),
  ("u32", 0x264, "OutputPlaneStride"),
  ("u32", 0x268, "OutputDepthStride"),
  ("u32", 0x26C, "OutputBatchStride"),

  # 0x270 -- *(uint *)(this + 0x34c) = *(uint *)(this + 0x34c) & 0xf0ffffff | 0x1000000;
  #  uVar6 = *(uint *)(this + 0x34c) & 0xffffcfcc | 0x2031;
  #          (ZinTensorDescriptorDmaInterleave & 0xf) << 0x18;
  ("u8",  0x273, "OutputInterleave"),    # i also have this at 0x211?
]

ANE_Struct_Dict = {}
for typ, num, nam in ANE_Struct:
  styp = {"u32": "I", "u16": "H", "u8": "B"}[typ]
  ANE_Struct_Dict[nam] = (styp, num)

class ANETensor:
  def __init__(self, *shape):
    self.shape = shape
    self.dtype = np.float16
    self.sz = int(np.prod(shape))
    assert(self.sz <= 0x4000)
    self.tt = libane.ANE_TensorCreate(self.sz, 1)
    assert(self.tt is not None)

  def data(self):
    data = libane.ANE_TensorData(self.tt)
    assert(data is not None)
    #print(hex(addressof(data.contents)))
    buf = np.ctypeslib.as_array(data, shape=(self.sz,))
    ret = np.frombuffer(buf, dtype=self.dtype)
    #print(ret.data)
    return ret

class ANE:
  def __init__(self):
    init_libane()
    libane.ANE_Open()

  def compile(self, dat):
    ret = libane.ANE_Compile(create_string_buffer(dat), len(dat))
    assert(ret is not None)
    return ret

  def run(self, prog, tin, tout, tweights=None):
    libane.ANE_Run(prog, tin.tt, tout.tt, tweights.tt if tweights is not None else 0)

  def tensor(self, shape):
    return ANETensor(shape)

  def unpack(self, dat):
    dat = struct.unpack("Q"*(len(dat)//8), dat)
    ret = {}
    for k,v in aneregs:
      by,bi,sz = v
      bi += (by%8)*8
      by //= 8
      rv = (dat[by] >> bi) & ((1 << sz)-1)
      ret[k] = rv
    return ret

  def pack(self, pk, dat):
    dat = list(struct.unpack("Q"*(len(dat)//8), dat))
    for k,v in aneregs:
      by,bi,sz = v
      bi += (by%8)*8
      by //= 8
      dat[by] &= ~(((1 << sz)-1) << bi)
      dat[by] |= pk[k] << bi
    dat = struct.pack("Q"*len(dat), *dat)
    return dat

  def debug(self, dat, mems=0):
    add = [0x30, 0x1d4, 0x220, 0x29c, 0x2f0, 0x30c, 0x32c]
    lens = [244, 60, 108, 68, 12, 16, 24]
    ptr = 0x2b
    ddat = dat[0:0x28]
    for a, pm in zip(add, lens):
      #assert pm == dat[ptr]
      ddat += b"\x00" * (a-len(ddat))
      ddat += dat[ptr+1:ptr+1+pm+4]
      ptr += pm+8
    ddat += b"\x00" * 0x100
    ret = collections.OrderedDict()
    for ln in libane.ANE_RegDebug(0, create_string_buffer(ddat), mems).decode('utf-8').strip().split("\n"):
      lnn = ln.split(" = ")
      if len(lnn) == 2:
        ret[lnn[0]] = int(lnn[1])
    return ret

  def filln(self, dat, nvdict, base=0x4000):
    for n,v in nvdict.items():
      styp, num = ANE_Struct_Dict[n]
      dat = self.fill(dat, [num], styp, v)
    return dat

  def fill(self, dat, addrs, type, val, base=0x4000):
    x = struct.pack(type, val)
    for a in addrs:
      dat[base+a:base+a+len(x)] = x
    return dat

if __name__ == "__main__":
  ane = ANE()

  tin = ANETensor(16)
  tout = ANETensor(16)

  tind = tin.data()
  toutd = tout.data()

  tind[0:4] = [-1,1,-2,2]
  print("** before **")
  print(tind)
  print(toutd)

  dat = open("../ops/relu.hwx", "rb").read()
  md = dat[0x4000:0x4300]
  dd = ane.unpack(md)
  mdf = ane.pack(dd, md)
  assert(md == mdf)

  comp = ane.compile(dat)
  ret = ane.run(comp, tin, tout)
  print("** after **")
  print(tind)
  print(toutd)

