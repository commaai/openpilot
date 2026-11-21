# type: ignore
import ctypes, ctypes.util, struct, fcntl, re
from hexdump import hexdump
from copy import deepcopy
import pathlib, sys
from tinygrad.helpers import to_mv, getenv
from tinygrad.runtime.autogen import mesa
sys.path.append(pathlib.Path(__file__).parent.parent.parent.as_posix())

IOCTL = getenv("IOCTL", 0)

ops = {}
import xml.etree.ElementTree as ET
xml = ET.parse(pathlib.Path(__file__).parent / "adreno_pm4.xml")
for child in xml.getroot():
  if 'name' in child.attrib and child.attrib['name'] == "adreno_pm4_type3_packets":
    for sc in child:
      if 'name' in sc.attrib and ('variants' not in sc.attrib or sc.attrib['variants'] != "A2XX"):
        ops[int(sc.attrib['value'], 0x10)] = sc.attrib['name']
#print(ops)
#exit(0)

CAPTURED_STATE = {}

REGS = {}
for k, v in mesa.__dict__.items():
  if k.startswith("REG_") and isinstance(v, int) and v > 1024: REGS[v] = k

from extra.qcom_gpu_driver import msm_kgsl
def ioctls_from_header():
  hdr = (pathlib.Path(__file__).parent.parent.parent / "extra/qcom_gpu_driver/msm_kgsl.h").read_text().replace("\\\n", "")
  pattern = r'#define\s+(IOCTL_KGSL_[A-Z0-9_]+)\s+_IOWR?\(KGSL_IOC_TYPE,\s+(0x[0-9a-fA-F]+),\s+struct\s([A-Za-z0-9_]+)\)'
  matches = re.findall(pattern, hdr, re.MULTILINE)
  return {int(nr, 0x10):(name, getattr(msm_kgsl, "struct_"+sname)) for name, nr, sname in matches}

nrs = ioctls_from_header()

# https://github.com/ensc/dietlibc/blob/master/include/sys/aarch64-ioctl.h

def get_struct(argp, stype):
  return ctypes.cast(ctypes.c_void_p(argp), ctypes.POINTER(stype)).contents

def format_struct(s):
  sdats = []
  for field_name, *_ in s._fields_:
    if field_name in {"__pad", "PADDING_0"}: continue
    dat = getattr(s, field_name)
    if isinstance(dat, int): sdats.append(f"{field_name}:0x{dat:X}")
    else: sdats.append(f"{field_name}:{dat}")
  return sdats

import mmap
mmaped = {}
def get_mem(addr, vlen):
  for k,v in mmaped.items():
    if k <= addr and addr < k+len(v):
      return v[addr-k:addr-k+vlen]
  # hope it was mmapped by someone else
  return bytes(to_mv(addr, vlen))

def hprint(vals):
  ret = []
  for v in vals:
    if v > 31: ret.append(f"{v:#x}")
    else: ret.append(f"{v}")
  return f"({','.join(ret)})"

ST6_SHADER = 0
ST6_CONSTANTS = 1
ST6_UBO = 2
ST6_IBO = 3

SB6_CS_TEX = 5
SB6_CS_SHADER = 13

def parse_cmd_buf(dat):
  global CAPTURED_STATE

  ptr = 0
  while ptr < len(dat):
    cmd = struct.unpack("I", dat[ptr:ptr+4])[0]
    if (cmd>>24) == 0x70:
      # packet with opcode and opcode specific payload (replace pkt3 starting with a5xx)
      opcode, size = ((cmd>>16)&0x7F), cmd&0x3FFF
      vals = struct.unpack("I"*size, dat[ptr+4:ptr+4+4*size])
      if IOCTL > 0: print(f"{ptr:3X} -- typ 7: {size=:3d}, {opcode=:#x} {ops[opcode]}", hprint(vals))
      if ops[opcode] == "CP_LOAD_STATE6_FRAG": # for compute shaders CP_LOAD_STATE6_FRAG is used
        dst_off = vals[0] & 0x3FFF
        state_type = (vals[0]>>14) & 0x3
        state_src = (vals[0]>>16) & 0x3
        state_block = (vals[0]>>18) & 0xF  # 13 = SB4_CS_SHADER
        num_unit = vals[0]>>22
        if IOCTL > 0: print(f"{num_unit=} {state_block=} {state_src=} {state_type=} {dst_off=}")

        if "LOAD_FRAGS" not in CAPTURED_STATE: CAPTURED_STATE['LOAD_FRAGS'] = []
        CAPTURED_STATE['LOAD_FRAGS'].append((state_block, state_type, num_unit, dst_off))

        if state_block == SB6_CS_SHADER:
          from tinygrad.runtime.support.compiler_mesa import disas_adreno
          if state_type == ST6_SHADER and IOCTL > 3:
            disas_adreno(get_mem(((vals[2] << 32) | vals[1]), num_unit * 128))
          if state_type == ST6_CONSTANTS:
            x = get_mem(((vals[2] << 32) | vals[1]), num_unit*4)
            CAPTURED_STATE['constants'] = x[:]
            if IOCTL > 2:
              print('constants')
              hexdump(x)
          if state_type == ST6_IBO:
            if state_src == 0x1:
              ibos_bytes = get_mem(CAPTURED_STATE['bindless_base'] + ((vals[2] << 32) | vals[1]) * 4, num_unit * 64)
            else: ibos_bytes = get_mem((vals[2] << 32) | vals[1], num_unit * 16 * 4)
            CAPTURED_STATE['ibos'] = ibos_bytes[:]
            if IOCTL > 1:
              print('texture ibos')
              hexdump(ibos_bytes)
        elif state_block == SB6_CS_TEX:
          if state_type == ST6_SHADER:
            if state_src == 0x1:
              samplers_bytes = get_mem(CAPTURED_STATE['bindless_base'] + ((vals[2] << 32) | vals[1]) * 4, num_unit * 64)
            else: samplers_bytes = get_mem((vals[2] << 32) | vals[1], num_unit * 4 * 4)
            CAPTURED_STATE['samplers'] = samplers_bytes[:]
            if IOCTL > 1:
              print('texture samplers')
              hexdump(samplers_bytes)
          if state_type == ST6_CONSTANTS:
            if state_src == 0x1:
              descriptors_bytes = get_mem(CAPTURED_STATE['bindless_base'] + ((vals[2] << 32) | vals[1]) * 4, num_unit * 64)
            else: descriptors_bytes = get_mem((vals[2] << 32) | vals[1], 1600)
            CAPTURED_STATE['descriptors'] = descriptors_bytes[:]
            if IOCTL > 1:
              print('texture descriptors')
              hexdump(descriptors_bytes)
      elif ops[opcode] == "CP_REG_TO_MEM":
        reg, cnt, b64, accum = vals[0] & 0x3FFFF, (vals[0] >> 18) & 0xFFF, (vals[0] >> 30) & 0x1, (vals[0] >> 31) & 0x1
        dest = vals[1] | (vals[2] << 32)
        if IOCTL > 0: print(f"{reg=} {cnt=} {b64=} {accum=} {dest=:#x}")
      ptr += 4*size
    elif (cmd>>28) == 0x4:
      # write one or more registers (replace pkt0 starting with a5xx)
      offset, size = ((cmd>>8)&0x7FFFF), cmd&0x7F
      reg_name = REGS.get(offset, f"reg {offset=:#x}")
      vals = struct.unpack("I"*size, dat[ptr+4:ptr+4+4*size])
      if IOCTL > 0: print(f"{ptr:3X} -- typ 4: {size=:3d}, {reg_name}", hprint(vals))
      for vi,v in enumerate(vals): CAPTURED_STATE[offset+vi] = v
      if offset == mesa.REG_A6XX_SP_CS_CONFIG:
        val = vals[0]
        if IOCTL > 0:
          print(f"\tBINDLESS_TEX={(val >> 0) & 0b1}")
          print(f"\tBINDLESS_SAMP={(val >> 1) & 0b1}")
          print(f"\tBINDLESS_IBO={(val >> 2) & 0b1}")
          print(f"\tBINDLESS_UBO={(val >> 3) & 0b1}")
          print(f"\tEN={(val >> 8) & 0b1}")
          print(f"\tNTEX={(val >> 9) & 0b11111111}")
          print(f"\tNSAMP={(val >> 17) & 0b11111}")
          print(f"\tNIBO={(val >> 22) & 0b1111111}")
      if offset == 0xa9b0:
        if IOCTL > 0:
          print(f'THREADSIZE-{(vals[0] >> 20)&0x1}\nEARLYPREAMBLE-{(vals[0] >> 23) & 0x1}\nMERGEDREGS-{(vals[0] >> 3) & 0x1}\nTHREADMODE-{vals[0] & 0x1}\nHALFREGFOOTPRINT-{(vals[0] >> 1) & 0x3f}\nFULLREGFOOTPRINT-{(vals[0] >> 7) & 0x3f}\nBRANCHSTACK-{(vals[0] >> 14) & 0x3f}\n')
          print(f'SP_CS_UNKNOWN_A9B1-{vals[1]}\nSP_CS_BRANCH_COND-{vals[2]}\nSP_CS_OBJ_FIRST_EXEC_OFFSET-{vals[3]}\nSP_CS_OBJ_START-{vals[4] | (vals[5] << 32)}\nSP_CS_PVT_MEM_PARAM-{vals[6]}\nSP_CS_PVT_MEM_ADDR-{vals[7] | (vals[8] << 32)}\nSP_CS_PVT_MEM_SIZE-{vals[9]}')
      if offset == 0xa9e8:
        CAPTURED_STATE['bindless_base'] = (vals[0] | (vals[1] << 32)) & ~0b11
        # print(hex(CAPTURED_STATE['bindless_base']))
        # hexdump(get_mem(CAPTURED_STATE['bindless_base'], 0x200))
      if offset == 0xb180:
        if IOCTL > 0:
          print('border color offset', hex(vals[1] << 32 | vals[0]))
          hexdump(get_mem(vals[1] << 32 | vals[0], 0x200))
      ptr += 4*size
    else:
      if IOCTL > 0:
        print("unk", hex(cmd))
    ptr += 4

@ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_ulong, ctypes.c_void_p)
def ioctl(fd, request, argp):
  ret = libc.syscall(0x1d, ctypes.c_int(fd), ctypes.c_ulong(request), ctypes.c_void_p(argp))

  idir, size, itype, nr = (request>>30), (request>>16)&0x3FFF, (request>>8)&0xFF, request&0xFF
  if nr in nrs and itype == 9:
    name, stype = nrs[nr]
    s = get_struct(argp, stype)
    if IOCTL > 0: print(f"{ret:2d} = {name:40s}", ' '.join(format_struct(s)))
    if name == "IOCTL_KGSL_GPUOBJ_INFO":
      mmaped[s.gpuaddr] = mmap.mmap(fd, s.size, offset=s.id*0x1000)
    if name == "IOCTL_KGSL_GPU_COMMAND":
      for i in range(s.numcmds):
        cmd = get_struct(s.cmdlist+ctypes.sizeof(msm_kgsl.struct_kgsl_command_object)*i, msm_kgsl.struct_kgsl_command_object)
        if IOCTL > 0: print(f"cmd {i}:", format_struct(cmd))
        parse_cmd_buf(get_mem(cmd.gpuaddr, cmd.size))
      for i in range(s.numobjs):
        obj = get_struct(s.objlist+s.objsize*i, msm_kgsl.struct_kgsl_command_object)
        if IOCTL > 0:
          print(f"obj {i}:", format_struct(obj))
          print(format_struct(msm_kgsl.struct_kgsl_cmdbatch_profiling_buffer.from_buffer_copy(get_mem(obj.gpuaddr, obj.size))))
        #hexdump(get_mem(obj.gpuaddr, obj.size))
  else:
    #print(f"ioctl({fd=}, (dir:{idir}, size:0x{size:3X}, type:{itype:d}, nr:0x{nr:2X}), {argp=:X}) = {ret=}")
    pass

  return ret

def install_hook(c_function, python_function):
  # AARCH64 trampoline to ioctl
  tramp = b"\x70\x00\x00\x10\x10\x02\x40\xf9\x00\x02\x1f\xd6"
  tramp += struct.pack("Q", ctypes.cast(ctypes.byref(python_function), ctypes.POINTER(ctypes.c_ulong)).contents.value)

  # get real ioctl address
  ioctl_address = ctypes.cast(ctypes.byref(c_function), ctypes.POINTER(ctypes.c_ulong))

  # hook ioctl
  libc = ctypes.CDLL(ctypes.util.find_library("libc"))
  ret = libc.mprotect(ctypes.c_ulong((ioctl_address.contents.value//0x1000)*0x1000), 0x2000, 7)
  assert ret == 0
  libc.memcpy(ioctl_address.contents, ctypes.create_string_buffer(tramp), len(tramp))

libc = ctypes.CDLL(ctypes.util.find_library("libc"))
install_hook(libc.ioctl, ioctl)
