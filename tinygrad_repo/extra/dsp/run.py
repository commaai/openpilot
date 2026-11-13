# mypy: ignore-errors
#!/usr/bin/env python3
import os, ctypes, ctypes.util, struct, platform, time
from tinygrad.runtime.autogen import libc, qcom_dsp
def to_mv(ptr, sz) -> memoryview: return memoryview(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint8 * sz)).contents).cast("B")
from hexdump import hexdump

def get_struct(argp, stype):
  return ctypes.cast(ctypes.c_void_p(argp), ctypes.POINTER(stype)).contents

def format_struct(s):
  sdats = []
  for field in s._fields_:
    dat = getattr(s, field[0])
    if isinstance(dat, int): sdats.append(f"{field[0]}:0x{dat:X}")
    elif hasattr(dat, "_fields_"): sdats.append((field[0], format_struct(dat)))
    elif field[0] == "PADDING_0": pass
    else: sdats.append(f"{field[0]}:{dat}")
  return sdats

@ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_ulong, ctypes.c_void_p)
def ioctl(fd, request, argp):
  fn = os.readlink(f"/proc/self/fd/{fd}")
  idir, size, itype, nr = (request>>30), (request>>16)&0x3FFF, (request>>8)&0xFF, request&0xFF

  if fn == "/dev/adsprpc-smd":
    if nr == 1:
      st = get_struct(argp, qcom_dsp.struct_fastrpc_ioctl_invoke)
      method = (st.sc>>24) & 0xFF
      in_args = (st.sc>>16) & 0xFF
      out_args = (st.sc>>8) & 0xFF
      if out_args:
        for arg in range(in_args, in_args+out_args):
          ctypes.memset(st.pra[arg].buf.pv, 0, st.pra[arg].buf.len)

  # print("enter", libc.gettid())
  ret = libc.syscall(0x1d, ctypes.c_int(fd), ctypes.c_ulong(request), ctypes.c_void_p(argp))
  # print("done", libc.gettid())
  if fn == "/dev/ion":
    if nr == 0:
      st = get_struct(argp, qcom_dsp.struct_ion_allocation_data)
      print(ret, "ION_IOC_ALLOC", format_struct(st))
    elif nr == 1:
      st = get_struct(argp, qcom_dsp.struct_ion_handle_data)
      print(ret, "ION_IOC_FREE", format_struct(st))
    elif nr == 2:
      st = get_struct(argp, qcom_dsp.struct_ion_fd_data)
      print(ret, "ION_IOC_MAP", format_struct(st))
  elif fn == "/dev/adsprpc-smd":
    assert chr(itype) == 'R'
    if nr == 8:
      st = ctypes.c_uint32.from_address(argp)
      print(ret, "FASTRPC_IOCTL_GETINFO", st.value)
    elif nr == 2:
      st = get_struct(argp, qcom_dsp.struct_fastrpc_ioctl_mmap)
      print(ret, "FASTRPC_IOCTL_MMAP", format_struct(st))
    elif nr == 1:
      # https://research.checkpoint.com/2021/pwn2own-qualcomm-dsp/
      st = get_struct(argp, qcom_dsp.struct_fastrpc_ioctl_invoke)
      print(ret, "FASTRPC_IOCTL_INVOKE", format_struct(st))
      # 0xFF000000 = Method index and attribute (the highest byte)
      # 0x00FF0000 = Number of input arguments
      # 0x0000FF00 = Number of output arguments
      # 0x000000F0 = Number of input handles
      # 0x0000000F = Number of output handles

      method = (st.sc>>24) & 0xFF
      in_args = (st.sc>>16) & 0xFF
      out_args = (st.sc>>8) & 0xFF
      in_h = (st.sc>>4) & 0xF
      out_h = (st.sc>>0) & 0xF
      print(f"\tm:{method} ia:{in_args} oa:{out_args} ih:{in_h} oh:{out_h}")
      if in_args or out_args:
        for arg in range(in_args+out_args):
          print(arg, format_struct(st.pra[arg]))
          if st.pra[arg].buf.pv is not None:
            ww = to_mv(st.pra[arg].buf.pv, st.pra[arg].buf.len)
            hexdump(to_mv(st.pra[arg].buf.pv, st.pra[arg].buf.len)[:0x40])
    elif nr == 6:
      print(ret, "FASTRPC_IOCTL_INIT", format_struct(ini:=get_struct(argp, qcom_dsp.struct_fastrpc_ioctl_init)))
      print(os.readlink(f"/proc/self/fd/{ini.filefd}"))
      # print(bytearray(to_mv(ini.file, ini.filelen)))
    elif nr == 7:
      print(ret, "FASTRPC_IOCTL_INVOKE_ATTRS", format_struct(ini:=get_struct(argp, qcom_dsp.struct_fastrpc_ioctl_invoke_attrs)))
    elif nr == 12: print(ret, "FASTRPC_IOCTL_CONTROL", format_struct(get_struct(argp, qcom_dsp.struct_fastrpc_ioctl_control)))
    else:
      print(f"{ret} UNPARSED {nr}")
  else:
    print("ioctl", f"{idir=} {size=} {itype=} {nr=} {fd=} {ret=}", fn)
  return ret

def install_hook(c_function, python_function):
  orig_func = (ctypes.c_char*4096)()
  python_function_addr = ctypes.cast(ctypes.byref(python_function), ctypes.POINTER(ctypes.c_ulong)).contents.value
  # AARCH64 trampoline to ioctl
  # 0x0000000000000000:  70 00 00 10    adr x16, #0xc
  # 0x0000000000000004:  10 02 40 F9    ldr x16, [x16]
  # 0x0000000000000008:  00 02 1F D6    br  x16
  tramp = b"\x70\x00\x00\x10\x10\x02\x40\xf9\x00\x02\x1f\xd6"
  tramp += struct.pack("Q", python_function_addr)

  # get real ioctl address
  ioctl_address = ctypes.cast(ctypes.byref(c_function), ctypes.POINTER(ctypes.c_ulong))

  # hook ioctl
  ret = libc.mprotect(ctypes.c_ulong((ioctl_address.contents.value//0x1000)*0x1000), 0x2000, 7)
  assert ret == 0
  ret = libc.mprotect(ctypes.c_ulong((ctypes.addressof(orig_func)//0x1000)*0x1000), 0x3000, 7)
  assert ret == 0
  libc.memcpy(orig_func, ioctl_address.contents, 0x1000)
  libc.memcpy(ioctl_address.contents, ctypes.create_string_buffer(tramp), len(tramp))
  return orig_func

libc = ctypes.CDLL(ctypes.util.find_library("libc"))
#install_hook(libc.ioctl, ioctl)
adsp = ctypes.CDLL(ctypes.util.find_library("adsprpc"))

def send_rpc_invoke(filename):
  pass

if __name__ == "__main__":
  print("calculator_open")
  # /dsp/cdsp/fastrpc_shell_3

  handle = ctypes.c_int64(-1)
  z = adsp.remote_handle64_open(ctypes.create_string_buffer(b"file:///libcalculator_skel.so?calculator_skel_handle_invoke&_modver=1.0&_dom=cdsp"),
                            ctypes.byref(handle))

  print("handle", z, hex(handle.value))
  assert handle.value != -1
  test = (ctypes.c_int32 * 100)()
  for i in range(100): test[i] = i
  print("calculator_sum")
  pra = (qcom_dsp.union_remote_arg64 * 3)()
  #arg_0 = ctypes.c_int32(100)
  arg_0 = ctypes.c_int32(100)
  arg_2 = ctypes.c_int64(-1)
  pra[0].buf.pv = ctypes.addressof(arg_0)
  pra[0].buf.len = 4
  pra[1].buf.pv = ctypes.addressof(test)
  pra[1].buf.len = 0x190
  pra[2].buf.pv = ctypes.addressof(arg_2)
  pra[2].buf.len = 8
  adsp.remote_handle64_invoke(handle, (2<<24) | (2<<16) | (1<<8), pra)
  print(arg_2.value)
  print("done")

  print("closing")
  x = adsp.remote_handle64_close(handle)
  print(x)
  print("dun")
  os._exit(0)
