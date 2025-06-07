# type: ignore
import ctypes, ctypes.util, struct, platform, pathlib, re, time, os
start = time.perf_counter()

# *** ioctl lib ***
libc = ctypes.CDLL(ctypes.util.find_library("c"))
# platform.processor calls `uname -p` which can return `unknown` on some systems
processor = os.getenv("IOCTL_PROCESSOR") or platform.processor()
IOCTL_SYSCALL = {"aarch64": 0x1d, "x86_64":16}[processor]

def get_struct(argp, stype):
  return ctypes.cast(ctypes.c_void_p(argp), ctypes.POINTER(stype)).contents

def format_struct(s):
  sdats = []
  for field_name, field_type in s._fields_:
    dat = getattr(s, field_name)
    if isinstance(dat, int): sdats.append(f"{field_name}:0x{dat:X}")
    else: sdats.append(f"{field_name}:{dat}")
  return sdats

def install_hook(c_function, python_function):
  python_function_addr = ctypes.cast(ctypes.byref(python_function), ctypes.POINTER(ctypes.c_ulong)).contents.value
  # AARCH64 trampoline to ioctl
  if processor == "aarch64":
    # 0x0000000000000000:  70 00 00 10    adr x16, #0xc
    # 0x0000000000000004:  10 02 40 F9    ldr x16, [x16]
    # 0x0000000000000008:  00 02 1F D6    br  x16
    tramp = b"\x70\x00\x00\x10\x10\x02\x40\xf9\x00\x02\x1f\xd6"
    tramp += struct.pack("Q", python_function_addr)
  elif processor == "x86_64":
    # 0x0000000000000000:  49 B8 aa aa aa aa aa aa aa aa    movabs r8, <address>
    # 0x000000000000000a:  41 FF E0                         jmp    r8
    tramp = b"\x49\xB8" + struct.pack("Q", python_function_addr) + b"\x41\xFF\xE0"
  else:
    raise Exception(f"processor {processor} not supported")

  # get real ioctl address
  ioctl_address = ctypes.cast(ctypes.byref(c_function), ctypes.POINTER(ctypes.c_ulong))

  # hook ioctl
  ret = libc.mprotect(ctypes.c_ulong((ioctl_address.contents.value//0x1000)*0x1000), 0x2000, 7)
  assert ret == 0
  libc.memcpy(ioctl_address.contents, ctypes.create_string_buffer(tramp), len(tramp))

# *** ioctl lib end ***

import tinygrad.runtime.autogen.kfd as kfd_ioctl
def ioctls_from_header():
  hdr = (pathlib.Path(__file__).parent / "kfd_ioctl.h").read_text().replace("\\\n", "")
  pattern = r'#define\s+(AMDKFD_IOC_[A-Z0-9_]+)\s+AMDKFD_IOW?R?\((0x[0-9a-fA-F]+),\s+struct\s([A-Za-z0-9_]+)\)'
  matches = re.findall(pattern, hdr, re.MULTILINE)
  return {int(nr, 0x10):(name, getattr(kfd_ioctl, "struct_"+sname)) for name, nr, sname in matches}
nrs = ioctls_from_header()

@ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_ulong, ctypes.c_void_p)
def ioctl(fd, request, argp):
  st = time.perf_counter()
  ret = libc.syscall(IOCTL_SYSCALL, ctypes.c_int(fd), ctypes.c_ulong(request), ctypes.c_void_p(argp))
  et = time.perf_counter()-st
  idir, size, itype, nr = (request>>30), (request>>16)&0x3FFF, (request>>8)&0xFF, request&0xFF
  if nr in nrs and itype == 75:
    # /dev/kfd
    name, stype = nrs[nr]
    s = get_struct(argp, stype)
    print(f"{(st-start)*1000:7.2f} ms +{et*1000.:7.2f} ms : {ret:2d} = {name:40s}", ' '.join(format_struct(s)))
    if name == "AMDKFD_IOC_SVM":
      out = ctypes.cast(s.attrs, ctypes.POINTER(kfd_ioctl.struct_kfd_ioctl_svm_attribute))
      for i in range(s.nattr): print(f"{i}: {kfd_ioctl.kfd_ioctl_svm_attr_type__enumvalues[out[i].type]:40s}: {out[i].value:#x}")
  else:
    print(f"{(st-start)*1000:7.2f} ms +{et*1000.:7.2f} ms : ioctl",
          f"{idir=} {size=} {itype=} {nr=} {fd=} {ret=}", os.readlink(f"/proc/self/fd/{fd}") if fd >= 0 else "")
  return ret

install_hook(libc.ioctl, ioctl)

# AMD_LOG_LEVEL=4 HSAKMT_DEBUG_LEVEL=7
if __name__ == "__main__":
  print("***** import tinygrad")
  from tinygrad import Tensor, Device, TinyJit
  print("***** access HIP")
  dev = Device["HIP"]
  print("***** create tensor a")
  a = Tensor([1.,2.]*1024*1024, device="HIP").realize()
  print("***** create tensor b")
  b = Tensor([3.,4.]*1024*1024, device="HIP").realize()
  @TinyJit
  def add(a, b): return (a+b).realize()
  for i in range(4):
    print(f"***** add tensors {i}")
    c = add(a, b)
    #dev.synchronize()
    c = add(b, a)
    dev.synchronize()
  print(f"***** copyout")
  nc = c.numpy()
  print(f"***** delete")
  del add, a, b, c, dev
  print(f"***** done")
  os._exit(0)
