from __future__ import annotations
import os, ctypes, functools, mmap, struct, array, math, sys, weakref
assert sys.platform != 'win32'
from types import SimpleNamespace
from typing import Any, cast
from tinygrad.device import BufferSpec
from tinygrad.runtime.support.hcq import HCQBuffer, HWQueue, HCQProgram, HCQCompiled, HCQAllocatorBase, HCQSignal, HCQArgsState, BumpAllocator
from tinygrad.runtime.support.hcq import FileIOInterface, MMIOInterface
from tinygrad.runtime.autogen import kgsl, adreno
from tinygrad.runtime.ops_gpu import CLCompiler, CLDevice
from tinygrad.renderer.cstyle import QCOMRenderer
from tinygrad.helpers import getenv, mv_address, to_mv, round_up, data64_le, prod, fromimport
if getenv("IOCTL"): import extra.qcom_gpu_driver.opencl_ioctl  # noqa: F401  # pylint: disable=unused-import

BUFTYPE_BUF, BUFTYPE_TEX, BUFTYPE_IBO = 0, 1, 2

#Parse C-style defines: <regname>_<field_x>__SHIFT and <regname>_<field_y>__MASK from the adreno module into the following format:
# qreg.<regname>(<field_x>=..., <field_y>=..., ..., <field_n>=...)
def _qreg_exec(reg, __val=0, **kwargs):
  for k, v in kwargs.items():
    __val |= (getattr(adreno, f'{reg[4:]}_{k.upper()}') if v else 0) if type(v) is bool else (v << getattr(adreno, f'{reg[4:]}_{k.upper()}__SHIFT'))
  return __val
qreg: Any = type("QREG", (object,), {name[4:].lower(): functools.partial(_qreg_exec, name) for name in adreno.__dict__.keys() if name[:4] == 'REG_'})

def next_power2(x): return 1 if x == 0 else 1 << (x - 1).bit_length()

def parity(val: int):
  for i in range(4,1,-1): val ^= val >> (1 << i)
  return (~0x6996 >> (val & 0xf)) & 1

def pkt7_hdr(opcode: int, cnt: int): return adreno.CP_TYPE7_PKT | cnt & 0x3FFF | parity(cnt) << 15 | (opcode & 0x7F) << 16 | parity(opcode) << 23

def pkt4_hdr(reg: int, cnt: int): return adreno.CP_TYPE4_PKT | cnt & 0x7F | parity(cnt) << 7 | (reg & 0x3FFFF) << 8 | parity(reg) << 27

class QCOMCompiler(CLCompiler):
  def __init__(self, device:str=""): super().__init__(CLDevice(device), 'compile_qcom')
  def disassemble(self, lib:bytes): fromimport('extra.disassemblers.adreno', 'disasm')(lib)

class QCOMSignal(HCQSignal):
  def __init__(self, *args, **kwargs): super().__init__(*args, **{**kwargs, 'timestamp_divider': 19.2})

  def _sleep(self, time_spent_waiting_ms:int):
    # Sleep only for timeline signals. Do it immediately to free cpu.
    if self.is_timeline and self.owner is not None:
      kgsl.IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID(self.owner.fd, context_id=self.owner.ctx, timestamp=self.owner.last_cmd, timeout=0xffffffff)

class QCOMComputeQueue(HWQueue):
  def __del__(self):
    if self.binded_device is not None: self.binded_device.allocator.free(self.hw_page, self.hw_page.size, BufferSpec(cpu_access=True, nolru=True))

  def cmd(self, opcode: int, *vals: int): self.q(pkt7_hdr(opcode, len(vals)), *vals)

  def reg(self, reg: int, *vals: int): self.q(pkt4_hdr(reg, len(vals)), *vals)

  def _cache_flush(self, write_back=True, invalidate=False, sync=True, memsync=False):
    # TODO: 7xx support.
    if write_back: self.cmd(adreno.CP_EVENT_WRITE, adreno.CACHE_FLUSH_TS, *data64_le(QCOMDevice.dummy_addr), 0) # dirty cache write-back.
    if invalidate: self.cmd(adreno.CP_EVENT_WRITE, adreno.CACHE_INVALIDATE) # invalidate cache lines (following reads from RAM).
    if memsync: self.cmd(adreno.CP_WAIT_MEM_WRITES)
    if sync: self.cmd(adreno.CP_WAIT_FOR_IDLE)

  def memory_barrier(self):
    self._cache_flush(write_back=True, invalidate=True, sync=True, memsync=True)
    return self

  def signal(self, signal:QCOMSignal, value=0, ts=False):
    self.cmd(adreno.CP_WAIT_FOR_IDLE)
    if QCOMDevice.gpu_id < 700:
      self.cmd(adreno.CP_EVENT_WRITE, qreg.cp_event_write_0(event=adreno.CACHE_FLUSH_TS, timestamp=ts),
               *data64_le(signal.timestamp_addr if ts else signal.value_addr), qreg.cp_event_write_3(value & 0xFFFFFFFF))
      self._cache_flush(write_back=True, invalidate=False, sync=False, memsync=False)
    else:
      # TODO: support devices starting with 8 Gen 1. Also, 700th series have convenient CP_GLOBAL_TIMESTAMP and CP_LOCAL_TIMESTAMP
      raise RuntimeError('CP_EVENT_WRITE7 is not supported')
    return self

  def timestamp(self, signal:QCOMSignal): return self.signal(signal, 0, ts=True)

  def wait(self, signal:QCOMSignal, value=0):
    self.cmd(adreno.CP_WAIT_REG_MEM, qreg.cp_wait_reg_mem_0(function=adreno.WRITE_GE, poll=adreno.POLL_MEMORY),*data64_le(signal.value_addr),
             qreg.cp_wait_reg_mem_3(ref=value&0xFFFFFFFF), qreg.cp_wait_reg_mem_4(mask=0xFFFFFFFF), qreg.cp_wait_reg_mem_5(delay_loop_cycles=32))
    return self

  def _build_gpu_command(self, dev:QCOMDevice, hw_addr=None):
    to_mv((hw_page_addr:=hw_addr or dev.cmd_buf_allocator.alloc(len(self._q) * 4)), len(self._q) * 4).cast('I')[:] = array.array('I', self._q)
    obj = kgsl.struct_kgsl_command_object(gpuaddr=hw_page_addr, size=len(self._q) * 4, flags=kgsl.KGSL_CMDLIST_IB)
    submit_req = kgsl.struct_kgsl_gpu_command(cmdlist=ctypes.addressof(obj), numcmds=1, context_id=dev.ctx,
                                              cmdsize=ctypes.sizeof(kgsl.struct_kgsl_command_object))
    return submit_req, obj

  def bind(self, dev:QCOMDevice):
    self.binded_device = dev
    self.hw_page = dev.allocator.alloc(len(self._q) * 4, BufferSpec(cpu_access=True, nolru=True))
    self.submit_req, self.obj = self._build_gpu_command(self.binded_device, self.hw_page.va_addr)
    # From now on, the queue is on the device for faster submission.
    self._q = to_mv(self.obj.gpuaddr, len(self._q) * 4).cast("I")

  def _submit(self, dev:QCOMDevice):
    if self.binded_device == dev: submit_req = self.submit_req
    else: submit_req, _ = self._build_gpu_command(dev)
    dev.last_cmd = kgsl.IOCTL_KGSL_GPU_COMMAND(dev.fd, __payload=submit_req).timestamp

  def exec(self, prg:QCOMProgram, args_state:QCOMArgsState, global_size, local_size):
    self.bind_args_state(args_state)

    def cast_int(x, ceil=False): return (math.ceil(x) if ceil else int(x)) if isinstance(x, float) else x
    global_size_mp = [cast_int(g*l) for g,l in zip(global_size, local_size)]

    self.cmd(adreno.CP_SET_MARKER, qreg.a6xx_cp_set_marker_0(mode=adreno.RM6_COMPUTE))
    self.reg(adreno.REG_A6XX_HLSQ_INVALIDATE_CMD, qreg.a6xx_hlsq_invalidate_cmd(cs_state=True, cs_ibo=True))
    self.reg(adreno.REG_A6XX_HLSQ_INVALIDATE_CMD, 0x0)
    self.reg(adreno.REG_A6XX_SP_CS_TEX_COUNT, qreg.a6xx_sp_cs_tex_count(0x80))
    self.reg(adreno.REG_A6XX_SP_CS_IBO_COUNT, qreg.a6xx_sp_cs_ibo_count(0x40))
    self.reg(adreno.REG_A6XX_SP_MODE_CONTROL, qreg.a6xx_sp_mode_control(isammode=adreno.ISAMMODE_CL))
    self.reg(adreno.REG_A6XX_SP_PERFCTR_ENABLE, qreg.a6xx_sp_perfctr_enable(cs=True))
    self.reg(adreno.REG_A6XX_SP_TP_MODE_CNTL, qreg.a6xx_sp_tp_mode_cntl(isammode=adreno.ISAMMODE_CL, unk3=2))
    self.reg(adreno.REG_A6XX_TPL1_DBG_ECO_CNTL, 0)
    self.cmd(adreno.CP_WAIT_FOR_IDLE)

    self.reg(adreno.REG_A6XX_HLSQ_CS_NDRANGE_0,
             qreg.a6xx_hlsq_cs_ndrange_0(kerneldim=3, localsizex=local_size[0] - 1, localsizey=local_size[1] - 1, localsizez=local_size[2] - 1),
             global_size_mp[0], 0, global_size_mp[1], 0, global_size_mp[2], 0, 0xccc0cf, 0xfc | qreg.a6xx_hlsq_cs_cntl_1(threadsize=adreno.THREAD64),
             cast_int(global_size[0], ceil=True), cast_int(global_size[1], ceil=True), cast_int(global_size[2], ceil=True))

    self.reg(adreno.REG_A6XX_SP_CS_CTRL_REG0,
             qreg.a6xx_sp_cs_ctrl_reg0(threadsize=adreno.THREAD64, halfregfootprint=prg.hregs, fullregfootprint=prg.fregs, branchstack=prg.brnchstck),
             qreg.a6xx_sp_cs_unknown_a9b1(unk6=True, shared_size=prg.shared_size), 0, prg.prg_offset, *data64_le(prg.lib_gpu.va_addr),
             qreg.a6xx_sp_cs_pvt_mem_param(memsizeperitem=prg.pvtmem_size_per_item), *data64_le(prg.dev._stack.va_addr),
             qreg.a6xx_sp_cs_pvt_mem_size(totalpvtmemsize=prg.pvtmem_size_total))

    self.cmd(adreno.CP_LOAD_STATE6_FRAG, qreg.cp_load_state6_0(state_type=adreno.ST_CONSTANTS, state_src=adreno.SS6_INDIRECT,
                                                               state_block=adreno.SB6_CS_SHADER, num_unit=1024 // 4),
             *data64_le(args_state.buf.va_addr))
    self.cmd(adreno.CP_LOAD_STATE6_FRAG, qreg.cp_load_state6_0(state_type=adreno.ST_SHADER, state_src=adreno.SS6_INDIRECT,
                                                               state_block=adreno.SB6_CS_SHADER, num_unit=round_up(prg.image_size, 128) // 128),
             *data64_le(prg.lib_gpu.va_addr))

    self.reg(adreno.REG_A6XX_HLSQ_CONTROL_2_REG, 0xfcfcfcfc, 0xfcfcfcfc, 0xfcfcfcfc, 0xfc, qreg.a6xx_hlsq_cs_cntl(constlen=1024 // 4, enabled=True))

    self.reg(adreno.REG_A6XX_SP_CS_PVT_MEM_HW_STACK_OFFSET, qreg.a6xx_sp_cs_pvt_mem_hw_stack_offset(prg.hw_stack_offset))
    self.reg(adreno.REG_A6XX_SP_CS_INSTRLEN, qreg.a6xx_sp_cs_instrlen(prg.image_size // 4))

    if args_state.prg.samp_cnt > 0:
      self.cmd(adreno.CP_LOAD_STATE6_FRAG, qreg.cp_load_state6_0(state_type=adreno.ST_SHADER, state_src=adreno.SS6_INDIRECT,
                                                                 state_block=adreno.SB6_CS_TEX, num_unit=args_state.prg.samp_cnt),
               *data64_le(args_state.buf.va_addr + args_state.prg.samp_off))
      self.reg(adreno.REG_A6XX_SP_CS_TEX_SAMP, *data64_le(args_state.buf.va_addr + args_state.prg.samp_off))
      self.reg(adreno.REG_A6XX_SP_PS_TP_BORDER_COLOR_BASE_ADDR, *data64_le(prg.dev.border_color_buf.va_addr))

    if args_state.prg.tex_cnt > 0:
      self.cmd(adreno.CP_LOAD_STATE6_FRAG, qreg.cp_load_state6_0(state_type=adreno.ST_CONSTANTS, state_src=adreno.SS6_INDIRECT,
                                                                 state_block=adreno.SB6_CS_TEX, num_unit=min(16, args_state.prg.tex_cnt)),
               *data64_le(args_state.buf.va_addr + args_state.prg.tex_off))
      self.reg(adreno.REG_A6XX_SP_CS_TEX_CONST, *data64_le(args_state.buf.va_addr + args_state.prg.tex_off))

    if args_state.prg.ibo_cnt > 0:
      self.cmd(adreno.CP_LOAD_STATE6_FRAG, qreg.cp_load_state6_0(state_type=adreno.ST6_IBO, state_src=adreno.SS6_INDIRECT,
                                                                 state_block=adreno.SB6_CS_SHADER, num_unit=args_state.prg.ibo_cnt),
               *data64_le(args_state.buf.va_addr + args_state.prg.ibo_off))
      self.reg(adreno.REG_A6XX_SP_CS_IBO, *data64_le(args_state.buf.va_addr + args_state.prg.ibo_off))

    self.reg(adreno.REG_A6XX_SP_CS_CONFIG,
             qreg.a6xx_sp_cs_config(enabled=True, nsamp=args_state.prg.samp_cnt, ntex=args_state.prg.tex_cnt, nibo=args_state.prg.ibo_cnt))
    self.cmd(adreno.CP_RUN_OPENCL, 0)
    self._cache_flush(write_back=True, invalidate=False, sync=False, memsync=False)
    return self

class QCOMArgsState(HCQArgsState):
  def __init__(self, buf:HCQBuffer, prg:QCOMProgram, bufs:tuple[HCQBuffer, ...], vals:tuple[int, ...]=()):
    super().__init__(buf, prg, bufs, vals=vals)

    if len(bufs) + len(vals) != len(prg.buf_info): raise RuntimeError(f'incorrect args size given={len(bufs)+len(vals)} != want={len(prg.buf_info)}')

    self.buf_info, self.args_info = prg.buf_info[:len(bufs)], prg.buf_info[len(bufs):]

    ctypes.memset(cast(int, self.buf.va_addr), 0, prg.kernargs_alloc_size)
    for cnst_val,cnst_off,cnst_sz in prg.consts_info: to_mv(self.buf.va_addr + cnst_off, cnst_sz)[:] = cnst_val.to_bytes(cnst_sz, byteorder='little')

    if prg.samp_cnt > 0: to_mv(self.buf.va_addr + prg.samp_off, len(prg.samplers) * 4).cast('I')[:] = array.array('I', prg.samplers)
    for i, b in enumerate(bufs):
      if prg.buf_info[i].type in {BUFTYPE_TEX, BUFTYPE_IBO}:
        obj = b.texture_info.desc if prg.buf_info[i].type is BUFTYPE_TEX else b.texture_info.ibo
        to_mv(self.buf.va_addr + prg.buf_info[i].offset, len(obj) * 4).cast('I')[:] = array.array('I', obj)
      self.bind_sints_to_buf(b.va_addr, buf=self.buf, fmt='Q', offset=self.buf_info[i].offset+(0 if self.buf_info[i].type is BUFTYPE_BUF else 16))

    for i, v in enumerate(vals): self.bind_sints_to_buf(v, buf=self.buf, fmt='I', offset=self.args_info[i].offset)

class QCOMProgram(HCQProgram):
  def __init__(self, dev: QCOMDevice, name: str, lib: bytes):
    self.dev: QCOMDevice = dev
    self.name, self.lib = name, lib
    self._parse_lib()

    self.lib_gpu: HCQBuffer = self.dev.allocator.alloc(self.image_size, buf_spec:=BufferSpec(cpu_access=True, nolru=True))
    to_mv(cast(int, self.lib_gpu.va_addr), self.image_size)[:] = self.image

    self.pvtmem_size_per_item: int = round_up(self.pvtmem, 512) >> 9
    self.pvtmem_size_total: int = self.pvtmem_size_per_item * 128 * 2
    self.hw_stack_offset: int = round_up(next_power2(round_up(self.pvtmem, 512)) * 128 * 16, 0x1000)
    self.shared_size: int = max(1, (self.shmem - 1) // 1024)
    self.max_threads = min(1024, ((384 * 32) // (max(1, (self.fregs + round_up(self.hregs, 2) // 2)) * 128)) * 128)
    dev._ensure_stack_size(self.hw_stack_offset * 4)

    kernargs_alloc_size = round_up(2048 + (self.tex_cnt + self.ibo_cnt) * 0x40 + self.samp_cnt * 0x10, 0x100)
    super().__init__(QCOMArgsState, self.dev, self.name, kernargs_alloc_size=kernargs_alloc_size)
    weakref.finalize(self, self._fini, self.dev, self.lib_gpu, buf_spec)

  def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False):
    if self.max_threads < prod(local_size): raise RuntimeError("Too many resources requested for launch")
    if any(g*l>mx for g,l,mx in zip(global_size, local_size, [65536, 65536, 65536])) and any(l>mx for l,mx in zip(local_size, [1024, 1024, 1024])):
      raise RuntimeError(f"Invalid global/local dims {global_size=}, {local_size=}")
    return super().__call__(*bufs, global_size=global_size, local_size=local_size, vals=vals, wait=wait)

  def _parse_lib(self):
    def _read_lib(off) -> int: return struct.unpack("I", self.lib[off:off+4])[0]

    # Extract image binary
    self.image_size = _read_lib(0x100)
    self.image = bytearray(self.lib[(image_offset:=_read_lib(0xc0)):image_offset+self.image_size])

    # Parse image descriptors
    image_desc_off = _read_lib(0x110)
    self.prg_offset, self.brnchstck = _read_lib(image_desc_off+0xc4), _read_lib(image_desc_off+0x108) // 2
    self.pvtmem, self.shmem = _read_lib(image_desc_off+0xc8), _read_lib(image_desc_off+0xd8)

    # Fill up constants and buffers info
    self.buf_info, self.consts_info = [], []

    # Collect sampler info.
    self.samp_cnt = samp_cnt_in_file = _read_lib(image_desc_off + 0xdc)
    assert self.samp_cnt <= 1, "Up to one sampler supported"
    if self.samp_cnt:
      self.samp_cnt += 1
      self.samplers = [qreg.a6xx_tex_samp_0(wrap_s=(clamp_mode:=adreno.A6XX_TEX_CLAMP_TO_BORDER), wrap_t=clamp_mode, wrap_r=clamp_mode),
                       qreg.a6xx_tex_samp_1(unnorm_coords=True, cubemapseamlessfiltoff=True), 0, 0, 0, 0, 0, 0]

    # Collect kernel arguments (buffers) info.
    bdoff = round_up(image_desc_off + 0x158 + len(self.name), 4) + 8 * samp_cnt_in_file
    while bdoff + 32 <= len(self.lib):
      length, _, _, offset_words, _, _, _, typ = struct.unpack("IIIIIIII", self.lib[bdoff:bdoff+32])
      if length == 0: break
      self.buf_info.append(SimpleNamespace(offset=offset_words * 4, type=typ))
      bdoff += length

    # Setting correct offsets to textures/ibos.
    self.tex_cnt, self.ibo_cnt = sum(x.type is BUFTYPE_TEX for x in self.buf_info), sum(x.type is BUFTYPE_IBO for x in self.buf_info)
    self.ibo_off, self.tex_off, self.samp_off = 2048, 2048 + 0x40 * self.ibo_cnt, 2048 + 0x40 * self.tex_cnt + 0x40 * self.ibo_cnt
    cur_ibo_off, cur_tex_off = self.ibo_off, self.tex_off
    for x in self.buf_info:
      if x.type is BUFTYPE_IBO: x.offset, cur_ibo_off = cur_ibo_off, cur_ibo_off + 0x40
      elif x.type is BUFTYPE_TEX: x.offset, cur_tex_off = cur_tex_off, cur_tex_off + 0x40

    if _read_lib(0xb0) != 0: # check if we have constants.
      cdoff = _read_lib(0xac)
      while cdoff + 40 <= image_offset:
        cnst, offset_words, _, is32 = struct.unpack("I", self.lib[cdoff:cdoff+4])[0], *struct.unpack("III", self.lib[cdoff+16:cdoff+28])
        self.consts_info.append((cnst, offset_words * (sz_bytes:=(2 << is32)), sz_bytes))
        cdoff += 40

    # Registers info
    reg_desc_off = _read_lib(0x34)
    self.fregs, self.hregs = _read_lib(reg_desc_off + 0x14), _read_lib(reg_desc_off + 0x18)

class QCOMTextureInfo:
  def __init__(self, pitch:int, real_stride:int, desc:list[int], ibo:list[int]):
    self.pitch, self.real_stride, self.desc, self.ibo = pitch, real_stride, desc, ibo

class QCOMAllocator(HCQAllocatorBase):
  def _alloc(self, size:int, options:BufferSpec) -> HCQBuffer:
    # Recalculate real size for texture
    if options.image is not None:
      imgw, imgh, itemsize_log = options.image.shape[1], options.image.shape[0], int(math.log2(options.image.itemsize))
      pitchalign = max(6, 11 - int(math.log2(imgh))) if imgh > 1 else 6
      align_up = max(1, (8 // itemsize_log + 1) - imgh // 32) if pitchalign == 6 else (2 ** (pitchalign - itemsize_log - 2))

      granularity = 128 if options.image.itemsize == 4 else 256
      pitch_add = (1 << pitchalign) if min(next_power2(imgw), round_up(imgw, granularity)) - align_up + 1 <= imgw and imgw > granularity//2 else 0
      pitch = round_up((real_stride:=imgw * 4 * options.image.itemsize), 1 << pitchalign) + pitch_add
      size = pitch * imgh

    buf = HCQBuffer(options.external_ptr, size, owner=self.dev) if options.external_ptr else self.dev._gpu_alloc(size)

    if options.image is not None:
      tex_fmt = adreno.FMT6_32_32_32_32_FLOAT if options.image.itemsize == 4 else adreno.FMT6_16_16_16_16_FLOAT
      desc = [qreg.a6xx_tex_const_0(0x8, swiz_x=0, swiz_y=1, swiz_z=2, swiz_w=3, fmt=tex_fmt), qreg.a6xx_tex_const_1(width=imgw, height=imgh),
              qreg.a6xx_tex_const_2(type=adreno.A6XX_TEX_2D, pitch=pitch, pitchalign=pitchalign-6), 0,
              *data64_le(buf.va_addr), qreg.a6xx_tex_const_6(plane_pitch=0x400000), qreg.a6xx_tex_const_7(13)]

      buf.texture_info = QCOMTextureInfo(pitch, real_stride, desc, [desc[0] & (~0xffff), *desc[1:len(desc)]])
    return buf

  def _do_copy(self, src_addr, dest_addr, src_size, real_size, src_stride, dest_stride, dest_off=0, src_off=0):
    while src_off < src_size:
      ctypes.memmove(dest_addr+dest_off, src_addr+src_off, real_size)
      src_off, dest_off = src_off+src_stride, dest_off+dest_stride

  def _copyin(self, dest:HCQBuffer, src:memoryview):
    stride, pitch = (src.nbytes, src.nbytes) if (ti:=cast(QCOMTextureInfo, dest.texture_info)) is None else (ti.real_stride, ti.pitch)
    self._do_copy(mv_address(src), dest.va_addr, src.nbytes, stride, stride, pitch)

  def _copyout(self, dest:memoryview, src:HCQBuffer):
    self.dev.synchronize()

    stride, pitch = (src.size, src.size) if (ti:=cast(QCOMTextureInfo, src.texture_info)) is None else (ti.real_stride, ti.pitch)
    self._do_copy(src.va_addr, mv_address(dest), src.size, stride, pitch, stride)

  def _as_buffer(self, src:HCQBuffer) -> memoryview:
    self.dev.synchronize()
    return to_mv(cast(int, src.va_addr), src.size)

  def _free(self, opaque, options:BufferSpec):
    self.dev.synchronize()
    self.dev._gpu_free(opaque)

class QCOMDevice(HCQCompiled):
  gpu_id: int = 0
  dummy_addr: int = 0

  def __init__(self, device:str=""):
    self.fd = FileIOInterface('/dev/kgsl-3d0', os.O_RDWR)
    QCOMDevice.dummy_addr = cast(int, self._gpu_alloc(0x1000).va_addr)

    flags = kgsl.KGSL_CONTEXT_PREAMBLE | kgsl.KGSL_CONTEXT_PWR_CONSTRAINT | kgsl.KGSL_CONTEXT_NO_FAULT_TOLERANCE | kgsl.KGSL_CONTEXT_NO_GMEM_ALLOC \
              | kgsl.KGSL_CONTEXT_PRIORITY(8) | kgsl.KGSL_CONTEXT_PREEMPT_STYLE(kgsl.KGSL_CONTEXT_PREEMPT_STYLE_FINEGRAIN)
    self.ctx = kgsl.IOCTL_KGSL_DRAWCTXT_CREATE(self.fd, flags=flags).drawctxt_id

    self.cmd_buf = self._gpu_alloc(16 << 20)
    self.cmd_buf_allocator = BumpAllocator(size=self.cmd_buf.size, base=cast(int, self.cmd_buf.va_addr), wrap=True)

    self.border_color_buf = self._gpu_alloc(0x1000, fill_zeroes=True)

    self.last_cmd:int = 0

    # Set max power
    struct.pack_into('IIQQ', pwr:=memoryview(bytearray(0x18)), 0, 1, self.ctx, mv_address(_:=memoryview(array.array('I', [1]))), 4)
    kgsl.IOCTL_KGSL_SETPROPERTY(self.fd, type=kgsl.KGSL_PROP_PWR_CONSTRAINT, value=mv_address(pwr), sizebytes=pwr.nbytes)

    # Load info about qcom device
    info = kgsl.struct_kgsl_devinfo()
    kgsl.IOCTL_KGSL_DEVICE_GETPROPERTY(self.fd, type=kgsl.KGSL_PROP_DEVICE_INFO, value=ctypes.addressof(info), sizebytes=ctypes.sizeof(info))
    QCOMDevice.gpu_id = ((info.chip_id >> 24) & 0xFF) * 100 + ((info.chip_id >> 16) & 0xFF) * 10 + ((info.chip_id >>  8) & 0xFF)
    if QCOMDevice.gpu_id >= 700: raise RuntimeError(f"Unsupported GPU: {QCOMDevice.gpu_id}")

    super().__init__(device, QCOMAllocator(self), QCOMRenderer(), QCOMCompiler(device), functools.partial(QCOMProgram, self),
                     QCOMSignal, QCOMComputeQueue, None)

  def _gpu_alloc(self, size:int, flags:int=0, uncached=False, fill_zeroes=False) -> HCQBuffer:
    flags |= kgsl.KGSL_MEMALIGN(alignment_hint:=12) | kgsl.KGSL_MEMFLAGS_USE_CPU_MAP
    if uncached: flags |= kgsl.KGSL_CACHEMODE(kgsl.KGSL_CACHEMODE_UNCACHED)

    alloc = kgsl.IOCTL_KGSL_GPUOBJ_ALLOC(self.fd, size=(bosz:=round_up(size, 1<<alignment_hint)), flags=flags, mmapsize=bosz)
    va_addr = self.fd.mmap(0, bosz, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, alloc.id * 0x1000)

    if fill_zeroes: ctypes.memset(va_addr, 0, size)
    return HCQBuffer(va_addr=va_addr, size=size, meta=alloc, view=MMIOInterface(va_addr, size, fmt='B'), owner=self)

  def _gpu_free(self, mem:HCQBuffer):
    kgsl.IOCTL_KGSL_GPUOBJ_FREE(self.fd, id=mem.meta.id)
    FileIOInterface.munmap(mem.va_addr, mem.meta.mmapsize)

  def _ensure_stack_size(self, sz):
    if not hasattr(self, '_stack'): self._stack = self._gpu_alloc(sz)
    elif self._stack.size < sz:
      self.synchronize()
      self._gpu_free(self._stack)
      self._stack = self._gpu_alloc(sz)
