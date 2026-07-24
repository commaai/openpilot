#!/usr/bin/env python3
import os, ctypes, ctypes.util, struct, platform, pathlib, contextlib, mmap, array
from threading import Thread
from tinygrad.runtime.autogen import qcom_dsp
from tinygrad.helpers import round_up, mv_address, to_mv
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
          # print(arg, f"arg (0x{st.pra[arg].buf.pv:X} len=0x{st.pra[arg].buf.len:X})")
          # print("input" if arg < in_args else "output", f"arg (0x{st.pra[arg].buf.pv:X} len=0x{st.pra[arg].buf.len:X})")
          if st.pra[arg].buf.pv is not None:
            # if st.pra[arg].buf.len == 0x258:
            #   print(bytearray(to_mv(st.pra[arg].buf.pv, st.pra[arg].buf.len)))
            if st.pra[arg].buf.len == 0x68:
              print(bytearray(to_mv(st.pra[arg].buf.pv, st.pra[arg].buf.len)))

            cut = 0x2000 if st.pra[arg].buf.len == 0x2000 or st.pra[arg].buf.len == 0x258 else 0x100
            ww = to_mv(st.pra[arg].buf.pv, st.pra[arg].buf.len)
            hexdump(to_mv(st.pra[arg].buf.pv, st.pra[arg].buf.len)[:cut])

            # if st.pra[arg].buf.len == 0x1000 and ww[0x30] == 0x6e:
            #   z = ww.cast('Q')[1] + 0x7F00000000
            #   print("DOO")
            #   hexdump(to_mv(z, 0x200))
      #print(format_struct(st.pra)))
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
install_hook(libc.ioctl, ioctl)
from tinygrad.runtime.autogen import libc

# adsp = ctypes.CDLL(ctypes.util.find_library("adsprpc"))
# print(adsp)

def rpc_invoke(rpcfd, handle, method, ins=None, outs=None):
  if ins or outs:
    ins = ins or list()
    outs = outs or list()
    pra = (qcom_dsp.union_remote_arg * (len(ins) + len(outs)))()
    for i,mv in enumerate(ins + outs):
      if isinstance(mv, memoryview):
        pra[i].buf.pv = mv_address(mv) if mv.nbytes > 0 else 0
        pra[i].buf.len = mv.nbytes
      else: assert False, "not supported"
    # pra = (qcom_dsp.union_remote_arg * (len(ins) + len(outs))).from_address(ctypes.addressof(pra))
  else:
    pra = None
    ins = ins or list()
    outs = outs or list()

  sc = (method << 24) | (len(ins) << 16) | (len(outs) << 8)
  return qcom_dsp.FASTRPC_IOCTL_INVOKE(rpcfd, handle=handle, sc=sc, pra=pra)

def listner_worker():
  context = 0
  handle = 0xffffffff
  msg_send = memoryview(bytearray(0x10)).cast('I')
  msg_recv = memoryview(bytearray(0x10)).cast('I')
  out_buf = memoryview(bytearray(0x1000)).cast('I')
  in_buf = memoryview(bytearray(0x1000)).cast('I')

  prev_res = 0xffffffff
  out_buf_size = 0

  req_args = (qcom_dsp.union_remote_arg * 4)()
  req_args[0].buf = qcom_dsp.struct_remote_buf(pv=mv_address(msg_send), len=0x10)
  req_args[1].buf = qcom_dsp.struct_remote_buf(pv=mv_address(out_buf), len=0x1000)
  req_args[2].buf = qcom_dsp.struct_remote_buf(pv=mv_address(msg_recv), len=0x10)
  req_args[3].buf = qcom_dsp.struct_remote_buf(pv=mv_address(in_buf), len=0x1000)

  while True:
    msg_send[0] = context
    msg_send[1] = prev_res
    msg_send[2] = out_buf_size
    msg_send[3] = 0x1000

    req_args[1].buf.len = out_buf_size
    qcom_dsp.FASTRPC_IOCTL_INVOKE(rpcfd, handle=0x3, sc=0x04020200, pra=req_args) # listener

    context = msg_recv[0]
    handle = msg_recv[1]
    sc = msg_recv[2]
    inbufs = (sc >> 16) & 0xff
    outbufs = (sc >> 8) & 0xff

    in_args, out_args = [], []
    ptr = mv_address(in_buf)
    for i in range(inbufs):
      sz = to_mv(ptr, 4).cast('I')[0]
      obj_ptr = round_up(ptr + 4, 8)
      in_args.append(to_mv(obj_ptr, sz))
      ptr = obj_ptr + sz

    ctypes.memset(mv_address(out_buf), 0, 0x1000)
    ptr_out = mv_address(out_buf)
    for i in range(outbufs):
      sz = to_mv(ptr, 4).cast('I')[0]
      ptr += 4

      to_mv(ptr_out, 4).cast('I')[0] = sz
      obj_ptr = round_up(ptr_out + 4, 8)

      out_args.append(to_mv(obj_ptr, sz))
      ptr_out = obj_ptr + sz

    out_buf_size = ptr_out - mv_address(out_buf)

    if sc == 0x20200: # greating?
      prev_res = 0
    elif sc == 0x13050100: # open
      # for a in in_args: hexdump(a)
      try:
        fd = os.open(in_args[3].tobytes()[:-1].decode(), os.O_RDONLY)
        out_args[0].cast('I')[0] = fd
        prev_res = 0
      except: prev_res = 2
    elif sc == 0x9010000: # seek
      res = os.lseek(in_args[0].cast('I')[0], in_args[0].cast('I')[1], in_args[0].cast('I')[2])
      prev_res = 0 if res >= 0 else res
    elif sc == 0x4010200: # read
      buf = os.read(in_args[0].cast('I')[0], in_args[0].cast('I')[1])
      out_args[1][:len(buf)] = buf
      out_args[0].cast('I')[0] = len(buf)
      out_args[0].cast('I')[1] = int(len(buf) == 0)
      prev_res = 0
    elif sc == 0x3010000: # close
      os.close(in_args[0].cast('I')[0])
      prev_res = 0
    elif sc == 0x1f020100: # stat
      # try:
      stat = os.stat(in_args[1].tobytes()[:-1].decode())
      out_stat = out_args[0].cast('Q')
      out_stat[1] = stat.st_dev
      out_stat[2] = stat.st_ino
      out_stat[3] = stat.st_mode | (stat.st_nlink << 32)
      out_stat[4] = stat.st_rdev
      out_stat[5] = stat.st_size
      # print(stat, stat.st_rdev)
      # assert False
      prev_res = 0
      # except: prev_res = 2
    elif sc == 0x2010100:
      heapid = in_args[0].cast('I')[0]
      lflags = in_args[0].cast('I')[1]
      rflags = in_args[0].cast('I')[2]
      assert rflags == 0x1000

      # print(in_args[0])

      # print("WOOW", in_args[0].cast('Q')[2])
      # print("WOOW2", in_args[0].cast('Q')[2])
      # print("WOOW3", in_args[0].cast('Q')[3])
      # print("WOOW3", in_args[0].cast('Q')[3])

      vin = in_args[0].cast('Q')[2]
      sz = in_args[0].cast('Q')[3]
      # vin = to_mv(in_args[0].cast('Q')[2], 8).cast('Q')[0]
      # sz = to_mv(in_args[0].cast('Q')[3], 8).cast('Q')[0]

      st = qcom_dsp.FASTRPC_IOCTL_MMAP(rpcfd, fd=-1, flags=rflags, vaddrin=0, size=sz)
      out_args[0].cast('Q')[0] = 0
      out_args[0].cast('Q')[1] = st.vaddrout
      prev_res = 0
    else: raise RuntimeError(f"Unknown {sc=:X}")

if __name__ == "__main__":
  ionfd = os.open('/dev/ion', os.O_RDONLY)
  rpcfd = os.open('/dev/adsprpc-smd', os.O_RDONLY | os.O_NONBLOCK)

  with contextlib.suppress(RuntimeError, OSError): qcom_dsp.ION_IOC_FREE(ionfd, handle=0)
  info = qcom_dsp.FASTRPC_IOCTL_GETINFO(rpcfd, 3)
  # x = qcom_dsp.FASTRPC_IOCTL_SETMODE(rpcfd, 0, __force_as_val=True)

  # init shell?
  fastrpc_shell = memoryview(bytearray(pathlib.Path('/vendor/dsp/cdsp/fastrpc_shell_3').read_bytes()))
  shell_mem = qcom_dsp.ION_IOC_ALLOC(ionfd, len=round_up(fastrpc_shell.nbytes, 0x1000), align=0x1000, heap_id_mask=0x2000000, flags=0x1)
  shell_mapped = qcom_dsp.ION_IOC_MAP(ionfd, handle=shell_mem.handle)
  fastrpc_shell_addr = libc.mmap(0, shell_mem.len, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED, shell_mapped.fd, 0)

  ctypes.memmove(fastrpc_shell_addr, mv_address(fastrpc_shell), fastrpc_shell.nbytes)
  # ctypes.memset(fastrpc_shell_addr, 0x0, 0xd6000)
  # print(hex(fastrpc_shell_addr))

  ctrls = qcom_dsp.FASTRPC_IOCTL_CONTROL(rpcfd, req=0x3)

  init = qcom_dsp.FASTRPC_IOCTL_INIT(rpcfd, flags=0x1, file=fastrpc_shell_addr, filelen=fastrpc_shell.nbytes, filefd=shell_mapped.fd)
  print("init shell done", shell_mapped.fd)

  # TODO: unmap here
  # qcom_dsp.ION_IOC_FREE(ionfd, handle=shell_mem.handle)

  rpc_invoke(rpcfd, handle=3, method=3)

  thread = Thread(target=listner_worker)
  thread.start()

  a1 = memoryview(bytearray(b'\x52\x00\x00\x00\xFF\x00\x00\x00'))
  a2 = memoryview(bytearray(b"file:///libcalculator_skel.so?calculator_skel_handle_invoke&_modver=1.0&_dom=cdsp\0"))
  o1 = memoryview(bytearray(0x8))
  o2 = memoryview(bytearray(0xff))
  z = rpc_invoke(rpcfd, handle=0, method=0, ins=[a1, a2], outs=[o1, o2])
  prg_handle = o1.cast('I')[0]

  # test
  test = (ctypes.c_int32 * 100)()
  for i in range(100): test[i] = i
  print("calculator_sum")
  pra = (qcom_dsp.union_remote_arg * 3)()
  #arg_0 = ctypes.c_int32(100)
  arg_0 = ctypes.c_int32(100)
  arg_2 = ctypes.c_int64(-1)
  pra[0].buf.pv = ctypes.addressof(arg_0)
  pra[0].buf.len = 4
  pra[1].buf.pv = ctypes.addressof(test)
  pra[1].buf.len = 0x190
  pra[2].buf.pv = ctypes.addressof(arg_2)
  pra[2].buf.len = 8
  qcom_dsp.FASTRPC_IOCTL_INVOKE(rpcfd, handle=prg_handle, sc=(2<<24) | (2<<16) | (1<<8), pra=pra)

  print(arg_2.value)
  print("done")
  os._exit(0)
