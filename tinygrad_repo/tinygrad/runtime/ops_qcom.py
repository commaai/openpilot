from __future__ import annotations
import os, ctypes, functools, mmap, struct, array, math, sys, contextlib
assert sys.platform != 'win32'
from typing import Any
from tinygrad.device import Compiled, BufferStorage, BufferSpec, Buffer, Device, Allocator, TinyELF
from tinygrad.runtime.support.hcq2 import HWQueue, HCQ_RUNTIME_DEV, encode_submit, ccall, cstruct, patch, unwrap_view, layout_args, pack_args
from tinygrad.runtime.support.hcq import FileIOInterface, MMIOInterface
from tinygrad.runtime.autogen import kgsl, mesa, libc
from tinygrad.renderer.cstyle import QCOMCLRenderer
from tinygrad.renderer.nir import IR3Renderer
from tinygrad.helpers import getenv, mv_address, round_up, ceildiv, prod, is_image_shape
from tinygrad.helpers import next_power2, flatten, PROFILE, IMAGE
from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.uop.ops import Ops, UOp, UPat, PatternMatcher
from tinygrad.engine.realize import get_call_arg_uops, get_call_var_uops
from tinygrad.runtime.support.system import System
if getenv("IOCTL"): import extra.qcom_gpu_driver.opencl_ioctl  # noqa: F401  # pylint: disable=unused-import

BUFTYPE_BUF, BUFTYPE_TEX, BUFTYPE_IBO = 0, 1, 2

@functools.cache
def dcache_flush():
  from tinygrad.uop.ops import KernelInfo
  from tinygrad.codegen import to_program
  buf, n = UOp.param(0, dtypes.uint8, 1), UOp.param(1, dtypes.int, shape=(), name="n", addrspace=AddrSpace.ALU)
  i = UOp.range(n, 0, dtype=dtypes.int)
  flush = UOp(Ops.CUSTOM, src=(buf.index(i * 64),), arg=('__asm__ volatile("dc cvac, %0" :: "r"({0}) : "memory");', dtypes.void))
  sink = UOp.sink(flush.end(i), UOp(Ops.CUSTOM, arg=('__asm__ volatile("dsb sy" ::: "memory");', dtypes.void)),
                  arg=KernelInfo(name="dcache_flush"), tag=1)
  prg = to_program(sink, Device["CPU"].renderer)
  return Device["CPU"].runtime(prg.to_elf())

#Parse C-style defines: <regname>_<field_x>__SHIFT and <regname>_<field_y>__MASK from the adreno module into the following format:
# qreg.<regname>(<field_x>=..., <field_y>=..., ..., <field_n>=...)
def _qreg_exec(__reg, __val=0, **kwargs):
  for k, v in kwargs.items():
    reg_name = f"{__reg[4:]}_{k.removeprefix('_').upper()}"
    __val |= (getattr(mesa, reg_name) if v else 0) if type(v) is bool else (v << getattr(mesa, f'{reg_name}__SHIFT'))
  return __val
qreg: Any = type("QREG", (object,), {name[4:].lower(): functools.partial(_qreg_exec, name) for name in mesa.__dict__.keys() if name[:4] == 'REG_'})

def ctz(v): return (v & -v).bit_length() - 1

def parity(val: int):
  for i in range(4,1,-1): val ^= val >> (1 << i)
  return (~0x6996 >> (val & 0xf)) & 1

def pkt7_hdr(opcode: int, cnt: int): return mesa.CP_TYPE7_PKT | cnt & 0x3FFF | parity(cnt) << 15 | (opcode & 0x7F) << 16 | parity(opcode) << 23

def pkt4_hdr(reg: int, cnt: int): return mesa.CP_TYPE4_PKT | cnt & 0x7F | parity(cnt) << 7 | (reg & 0x3FFFF) << 8 | parity(reg) << 27

def _read_lib(lib, off) -> int: return struct.unpack("I", lib[off:off+4])[0]

class QCOMComputeQueue(HWQueue):
  dev:QCOMDevice
  def cmd(self, opcode:int, *vals): self.q(pkt7_hdr(opcode, sum(x.dtype.itemsize // 4 if isinstance(x, UOp) else 1 for x in vals)), *vals)

  def reg(self, reg:int, *vals): self.q(pkt4_hdr(reg, sum(x.dtype.itemsize // 4 if isinstance(x, UOp) else 1 for x in vals)), *vals)

  def _cache_flush(self, write_back=True, invalidate=False, sync=True, memsync=False):
    # TODO: 7xx support.
    if write_back: # dirty cache write-back, into the device's dummy buffer
      dummy = UOp.placeholder((0x1000,), dtypes.uint8, 0, device=self.devs, tag="dummy")
      self.cmd(mesa.CP_EVENT_WRITE, mesa.CACHE_FLUSH_TS, dummy.getaddr(self.devs), 0)
    if invalidate: self.cmd(mesa.CP_EVENT_WRITE, mesa.CACHE_INVALIDATE) # invalidate cache lines (following reads from RAM).
    if memsync: self.cmd(mesa.CP_WAIT_MEM_WRITES)
    if sync: self.cmd(mesa.CP_WAIT_FOR_IDLE)

  def memory_barrier(self): self._cache_flush(write_back=True, invalidate=True, sync=True, memsync=True)

  def signal(self, signal:UOp, value:UOp):
    self.cmd(mesa.CP_WAIT_FOR_IDLE)
    if self.dev.gpu_id[:2] < (7, 3):
      self.cmd(mesa.CP_EVENT_WRITE, qreg.cp_event_write_0(event=mesa.CACHE_FLUSH_TS), signal.getaddr(self.devs), value.cast(dtypes.uint32))
      self._cache_flush(write_back=True, invalidate=False, sync=False, memsync=False)
    else:
      # TODO: support devices starting with 8 Gen 1. Also, 700th series have convenient CP_GLOBAL_TIMESTAMP and CP_LOCAL_TIMESTAMP
      raise RuntimeError('CP_EVENT_WRITE7 is not supported')

  def timestamp(self, signal:UOp):
    self.cmd(mesa.CP_WAIT_FOR_IDLE)
    self.cmd(mesa.CP_REG_TO_MEM, qreg.cp_reg_to_mem_0(reg=mesa.REG_A6XX_CP_ALWAYS_ON_COUNTER, cnt=2, _64b=True), signal.getaddr(self.devs))

  def wait(self, signal:UOp, value:UOp):
    self.cmd(mesa.CP_WAIT_REG_MEM, qreg.cp_wait_reg_mem_0(function=mesa.WRITE_GE, poll=mesa.POLL_MEMORY), signal.getaddr(self.devs),
             value.cast(dtypes.uint32), qreg.cp_wait_reg_mem_4(mask=0xFFFFFFFF), qreg.cp_wait_reg_mem_5(delay_loop_cycles=32))

  def kernargs(self, call:UOp, prg:UOp, data:QCOMProgramData) -> UOp:
    bufs, vals = get_call_arg_uops(call), get_call_var_uops(call, prg)
    ubos = [bufs[slot] for _,slot,_,shape in data.signature if slot < len(bufs) and not is_image_shape(shape)]
    uavs = [(dt,shape,bufs[slot]) for _,slot,dt,shape in data.signature if slot < len(bufs) and is_image_shape(shape)]
    # NIR can reorder images to different texture slots
    ibos, texs = uavs[:data.ibo_cnt], [uavs[data.ibo_cnt + (data.tex_to_image[i] if data.NIR else i)] for i in range(data.tex_cnt)]

    args = [(off, UOp.const(val, dtypes.uint32 if sz == 4 else dtypes.uint16)) for val,off,sz in data.consts_info]
    args += layout_args(data.samplers, data.samp_off)
    vals = [v.ccast(dt) for v,(_,_,dt,_) in zip(vals, data.signature[len(bufs):])]
    if data.NIR:
      args += layout_args([b.getaddr(self.devs) for b in ubos] + vals, data.buf_off)
      if data.wgsz != 0xfc: args += layout_args(list(prg.arg.local_size), data.wgsz * 4)
    else: args += list(zip(data.buf_offs, [b.getaddr(self.devs) for b in ubos] + vals))

    def _tex(b, ibo=False):
      imgdt, shape, buf = b
      pitch = shape[1] * 4 * imgdt.itemsize
      fmt = mesa.FMT6_32_32_32_32_FLOAT if imgdt.itemsize == 4 else mesa.FMT6_16_16_16_16_FLOAT
      return [qreg.a6xx_tex_const_0(fmt=fmt) if ibo else qreg.a6xx_tex_const_0(0x8, swiz_x=0, swiz_y=1, swiz_z=2, swiz_w=3, fmt=fmt),
              qreg.a6xx_tex_const_1(width=shape[1], height=shape[0]),
              qreg.a6xx_tex_const_2(type=mesa.A6XX_TEX_2D, pitch=pitch, pitchalign=ctz(pitch)-6), 0, buf.getaddr(self.devs),
              qreg.a6xx_tex_const_6(plane_pitch=0x400000), qreg.a6xx_tex_const_7(13), 0, 0, 0, 0, 0, 0, 0, 0]
    args += layout_args(flatten(map(_tex, texs)), data.tex_off) + layout_args(flatten(map(functools.partial(_tex, ibo=True), ibos)), data.ibo_off)
    return UOp(Ops.LINEAR, src=tuple(pack_args(args, data.kernargs_alloc_size)), arg="kernargs")

  def exec(self, call:UOp, prg:UOp):
    data, lib = qcom_build_program(self.dev, prg, self.devs)
    global_size, local_size = prg.arg.global_size, prg.arg.local_size
    if data.max_threads < prod(local_size): raise RuntimeError("Too many resources requested for launch")
    if any(g*l>mx for g,l,mx in zip(global_size, local_size, [65536, 65536, 65536])) and any(l>mx for l,mx in zip(local_size, [1024, 1024, 1024])):
      raise RuntimeError(f"Invalid global/local dims {global_size=}, {local_size=}")

    def cast_int(x, ceil=False): return (math.ceil(x) if ceil else int(x)) if isinstance(x, float) else x
    global_size_mp = [cast_int(g*l) for g,l in zip(global_size, local_size)]

    args_addr, lib_addr = self.kernargs(call, prg, data).getaddr(self.devs), lib.getaddr(self.devs)
    stack_addr = UOp.placeholder((data.hw_stack_offset * 4,), dtypes.uint8, 0, device=self.devs).rtag("stack").getaddr(self.devs)

    self.cmd(mesa.CP_SET_MARKER, qreg.a6xx_cp_set_marker_0(mode=mesa.RM6_COMPUTE))
    self.reg(mesa.REG_A6XX_SP_UPDATE_CNTL, qreg.a6xx_sp_update_cntl(cs_state=True, cs_uav=True))
    self.reg(mesa.REG_A6XX_SP_UPDATE_CNTL, 0x0)
    self.reg(mesa.REG_A6XX_SP_CS_TSIZE, qreg.a6xx_sp_cs_tsize(0x80)) # is this right? mesa uses 1
    self.reg(mesa.REG_A6XX_SP_CS_USIZE, qreg.a6xx_sp_cs_usize(0x40)) # mesa also uses 1
    self.reg(mesa.REG_A6XX_SP_MODE_CNTL, qreg.a6xx_sp_mode_cntl(isammode=mesa.ISAMMODE_GL if data.NIR else mesa.ISAMMODE_CL,
                                                                constant_demotion_enable=data.NIR))
    self.reg(mesa.REG_A6XX_SP_PERFCTR_SHADER_MASK, qreg.a6xx_sp_perfctr_shader_mask(cs=True))
    self.reg(mesa.REG_A6XX_TPL1_MODE_CNTL, qreg.a6xx_tpl1_mode_cntl(isammode=mesa.ISAMMODE_GL if data.NIR else mesa.ISAMMODE_CL))
    self.reg(mesa.REG_A6XX_TPL1_DBG_ECO_CNTL, 0)
    self.cmd(mesa.CP_WAIT_FOR_IDLE)

    self.reg(mesa.REG_A6XX_SP_CS_NDRANGE_0,
             qreg.a6xx_sp_cs_ndrange_0(kerneldim=3, localsizex=local_size[0] - 1, localsizey=local_size[1] - 1, localsizez=local_size[2] - 1),
             global_size_mp[0], 0, global_size_mp[1], 0, global_size_mp[2], 0, 0xccc0cf, 0xfc | qreg.a6xx_sp_cs_wge_cntl(threadsize=mesa.THREAD64),
             cast_int(global_size[0], ceil=True), cast_int(global_size[1], ceil=True), cast_int(global_size[2], ceil=True))

    self.reg(mesa.REG_A6XX_SP_CS_CNTL_0,
             qreg.a6xx_sp_cs_cntl_0(threadsize=mesa.THREAD64, halfregfootprint=data.hregs, fullregfootprint=data.fregs, branchstack=data.brnchstck),
             qreg.a6xx_sp_cs_cntl_1(constantrammode=mesa.CONSTLEN_256, shared_size=data.shared_size), # should this be CONSTLEN_512?
             0, data.prg_offset, lib_addr,
             qreg.a6xx_sp_cs_pvt_mem_param(memsizeperitem=data.pvtmem_size_per_item), stack_addr,
             qreg.a6xx_sp_cs_pvt_mem_size(totalpvtmemsize=data.pvtmem_size_total))

    # the kernargs sit in the cmdbuf, so the const upload is sized to them (in vec4s) rather than to the whole constlen: it must not read past
    self.cmd(mesa.CP_LOAD_STATE6_FRAG, qreg.cp_load_state6_0(state_type=mesa.ST_CONSTANTS, state_src=mesa.SS6_INDIRECT,
                                                             state_block=mesa.SB6_CS_SHADER, num_unit=data.kernargs_alloc_size // 16), args_addr)
    self.cmd(mesa.CP_LOAD_STATE6_FRAG, qreg.cp_load_state6_0(state_type=mesa.ST_SHADER, state_src=mesa.SS6_INDIRECT,
                                                             state_block=mesa.SB6_CS_SHADER, num_unit=ceildiv(data.image_size, 128)), lib_addr)

    self.reg(mesa.REG_A6XX_SP_REG_PROG_ID_0, 0xfcfcfcfc, 0xfcfcfcfc, 0xfcfcfcfc, 0xfc, qreg.a6xx_sp_cs_const_config(constlen=1024 // 4, enabled=True))

    self.reg(mesa.REG_A6XX_SP_CS_PVT_MEM_STACK_OFFSET, qreg.a6xx_sp_cs_pvt_mem_stack_offset(data.hw_stack_offset))
    # image_size is in bytes, but INSTR_SIZE is measured in units of instruction groups (16 instructions, 8 bytes each)
    # https://elixir.bootlin.com/mesa/mesa-26.1.5/source/src/freedreno/ir3/ir3_shader.h#L719-L723
    self.reg(mesa.REG_A6XX_SP_CS_INSTR_SIZE, qreg.a6xx_sp_cs_instr_size(ceildiv(data.image_size, 128)))

    if data.samp_cnt > 0:
      self.cmd(mesa.CP_LOAD_STATE6_FRAG, qreg.cp_load_state6_0(state_type=mesa.ST_SHADER, state_src=mesa.SS6_INDIRECT,
                                                               state_block=mesa.SB6_CS_TEX, num_unit=data.samp_cnt), args_addr + data.samp_off)
      self.reg(mesa.REG_A6XX_SP_CS_SAMPLER_BASE, args_addr + data.samp_off)
      self.reg(mesa.REG_A6XX_TPL1_CS_BORDER_COLOR_BASE,
               UOp.placeholder((0x1000,), dtypes.uint8, 0, device=self.devs, tag="border_color").getaddr(self.devs))

    if data.tex_cnt > 0:
      self.cmd(mesa.CP_LOAD_STATE6_FRAG, qreg.cp_load_state6_0(state_type=mesa.ST_CONSTANTS, state_src=mesa.SS6_INDIRECT,
                                                               state_block=mesa.SB6_CS_TEX, num_unit=min(16, data.tex_cnt)), args_addr + data.tex_off)
      self.reg(mesa.REG_A6XX_SP_CS_TEXMEMOBJ_BASE, args_addr + data.tex_off)

    if data.ibo_cnt > 0:
      self.cmd(mesa.CP_LOAD_STATE6_FRAG, qreg.cp_load_state6_0(state_type=mesa.ST6_UAV, state_src=mesa.SS6_INDIRECT,
                                                               state_block=mesa.SB6_CS_SHADER, num_unit=data.ibo_cnt), args_addr + data.ibo_off)
      self.reg(mesa.REG_A6XX_SP_CS_UAV_BASE, args_addr + data.ibo_off)

    self.reg(mesa.REG_A6XX_SP_CS_CONFIG, qreg.a6xx_sp_cs_config(enabled=True, nsamp=data.samp_cnt, ntex=data.tex_cnt, nuav=data.ibo_cnt))

    if data.NIR:
      self.reg(mesa.REG_A6XX_SP_CS_CONST_CONFIG_0,
               qreg.a6xx_sp_cs_const_config_0(wgidconstid=data.wgid, wgsizeconstid=data.wgsz, wgoffsetconstid=0xfc, localidregid=data.lid),
               qreg.a6xx_sp_cs_wge_cntl(linearlocalidregid=0xfc, threadsize=mesa.THREAD64))
      self.cmd(mesa.CP_EXEC_CS, 0,
               qreg.cp_exec_cs_1(ngroups_x=global_size[0]), qreg.cp_exec_cs_2(ngroups_y=global_size[1]), qreg.cp_exec_cs_3(_ngroups_z=global_size[2]))
    else: self.cmd(mesa.CP_RUN_OPENCL, 0)

    self._cache_flush(write_back=True, invalidate=False, sync=False, memsync=False)

  def submit(self, cmdbuf:UOp) -> UOp:
    ib, ib_off = unwrap_view(cmdbuf)
    fd, ctxid = [UOp.variable(n, 0, 2**31 - 1, dtypes.int32, param=True) for n in ("kgsl_fd", "kgsl_ctx")]
    obj = cstruct(kgsl.struct_kgsl_command_object, gpuaddr=ib.getaddr(self.devs) + ib_off, size=cmdbuf.max_numel(), flags=kgsl.KGSL_CMDLIST_IB)
    req = cstruct(kgsl.struct_kgsl_gpu_command, cmdlist=obj.getaddr(HCQ_RUNTIME_DEV.value), cmdsize=ctypes.sizeof(kgsl.struct_kgsl_command_object),
                  numcmds=1, context_id=ctxid)
    ret = UOp.placeholder((1,), dtypes.int32, device=self.devs, volatile=True, tag="submit_ret")

    idir, base, nr, struct_t = kgsl.IOCTL_KGSL_GPU_COMMAND.args
    ioctl_cmd = (idir << 30) | (ctypes.sizeof(struct_t) << 16) | (base << 8) | nr
    return ret.index(0).store(ccall(libc.dll.ioctl, fd, UOp.const(ioctl_cmd, dtypes.uint32), req.after(cmdbuf).index(0)))

class QCOMProgramData:
  def __init__(self, dev:QCOMDevice, obj:TinyELF):
    self.signature, self.name, self.NIR = obj.signature, obj.name, isinstance(dev.renderer, IR3Renderer)

    if self.NIR:
      from tinygrad.runtime.support.compiler_mesa import IR3Compiler
      v, cs, imm_vals, self.image = IR3Compiler.unpack_lib(obj.lib)
      self.prg_offset, self.brnchstck, self.image_size, self.pvtmem, self.shmem = 0, v.branchstack, v.info.size, v.pvtmem_size, v.shared_size
      self.wgsz = alloc.offset_vec4 * 4 + 8 if (alloc:=cs.allocs.consts[mesa.IR3_CONST_ALLOC_DRIVER_PARAMS]).size_vec4 else 0xfc

      self.wgid, self.lid = v.cs.work_group_id, v.cs.local_invocation_id # register ids
      self.buf_off, imm_off = cs.ubo_state.range[0].offset, cs.allocs.max_const_offset_vec4 * 16
      self.consts_info = [(struct.unpack_from("<I", imm_vals, i)[0], imm_off + i, 4) for i in range(0, len(imm_vals), 4)]

      # see https://elixir.bootlin.com/mesa/mesa-25.3.0/source/src/freedreno/ir3/ir3_shader.h#L525
      # and https://elixir.bootlin.com/mesa/mesa-25.3.0/source/src/freedreno/ir3/ir3_compiler_nir.c#L5389
      self.samp_cnt, self.tex_cnt, self.ibo_cnt = (nt:=v.image_mapping.num_tex), nt, v.num_uavs - nt
      self.tex_to_image = v.image_mapping.tex_to_image[:]
      # IR3 outputs a sampler for every texture (https://elixir.bootlin.com/mesa/mesa-25.3.0/source/src/freedreno/ir3/ir3_compiler_nir.c#L1714)
      self.samplers = [qreg.a6xx_tex_samp_0(wrap_s=(clamp_mode:=mesa.A6XX_TEX_CLAMP_TO_BORDER), wrap_t=clamp_mode, wrap_r=clamp_mode),
                       qreg.a6xx_tex_samp_1(unnorm_coords=True, cubemapseamlessfiltoff=True), 0, 0] * self.samp_cnt

      self.tex_off, self.ibo_off, self.samp_off = 2048, 2048 + 0x40 * self.tex_cnt, 2048 + 0x40 * (self.tex_cnt + self.ibo_cnt)
      self.fregs, self.hregs = v.info.max_reg + 1, v.info.max_half_reg + 1
    else: self._parse_lib(obj.lib)

    self.pvtmem_size_per_item: int = round_up(self.pvtmem, 512) >> 9
    self.pvtmem_size_total: int = self.pvtmem_size_per_item * 128 * 2
    self.hw_stack_offset: int = round_up(next_power2(round_up(self.pvtmem, 512)) * 128 * 16, 0x1000)
    self.shared_size: int = max(1, (self.shmem - 1) // 1024)
    self.max_threads = min(1024, ((384 * 32) // (max(1, (self.fregs + round_up(self.hregs, 2) // 2)) * 128)) * 128)
    self.kernargs_alloc_size = round_up(2048 + (self.tex_cnt + self.ibo_cnt) * 0x40 + len(self.samplers) * 4, 0x100)

  def _parse_lib(self, lib):
    # Extract image binary
    self.image_size = _read_lib(lib, 0x100)
    self.image = lib[(image_offset:=_read_lib(lib, 0xc0)):image_offset+self.image_size]

    # Parse image descriptors
    image_desc_off = _read_lib(lib, 0x110)
    self.prg_offset, self.brnchstck = _read_lib(lib, image_desc_off+0xc4), _read_lib(lib, image_desc_off+0x108) // 2
    self.pvtmem, self.shmem = _read_lib(lib, image_desc_off+0xc8), _read_lib(lib, image_desc_off+0xd8)

    # Fill up constants and buffers info
    self.consts_info = []

    # Collect sampler info.
    self.samp_cnt = samp_cnt_in_file = _read_lib(lib, image_desc_off + 0xdc)
    assert self.samp_cnt <= 1, "Up to one sampler supported"
    if self.samp_cnt:
      self.samp_cnt += 1
      self.samplers = [qreg.a6xx_tex_samp_0(wrap_s=(clamp_mode:=mesa.A6XX_TEX_CLAMP_TO_BORDER), wrap_t=clamp_mode, wrap_r=clamp_mode),
                       qreg.a6xx_tex_samp_1(unnorm_coords=True, cubemapseamlessfiltoff=True), 0, 0, 0, 0, 0, 0]
    else: self.samplers = []

    # Collect kernel arguments (buffers) info.
    bdoff, binfos = round_up(image_desc_off + 0x158 + len(self.name), 4) + 8 * samp_cnt_in_file, []
    while bdoff + 32 <= len(lib):
      length, _, _, offset_words, _, _, _, typ = struct.unpack("8I", lib[bdoff:bdoff+32])
      if length == 0: break
      binfos.append((offset_words * 4, typ))
      bdoff += length
    self.buf_offs = [off for off,typ in binfos if typ not in {BUFTYPE_TEX, BUFTYPE_IBO}]

    # Setting correct offsets to textures/ibos.
    self.tex_cnt, self.ibo_cnt = sum(typ is BUFTYPE_TEX for _,typ in binfos), sum(typ is BUFTYPE_IBO for _,typ in binfos)
    self.ibo_off, self.tex_off, self.samp_off = 2048, 2048 + 0x40 * self.ibo_cnt, 2048 + 0x40 * self.tex_cnt + 0x40 * self.ibo_cnt

    if _read_lib(lib, 0xb0) != 0: # check if we have constants.
      cdoff = _read_lib(lib, 0xac)
      while cdoff + 40 <= image_offset:
        cnst, offset_words, _, is32 = struct.unpack("I", lib[cdoff:cdoff+4])[0], *struct.unpack("III", lib[cdoff+16:cdoff+28])
        self.consts_info.append((cnst, offset_words * (sz_bytes:=(2 << is32)), sz_bytes))
        cdoff += 40

    # Registers info
    reg_desc_off = _read_lib(lib, 0x34)
    self.fregs, self.hregs = _read_lib(lib, reg_desc_off + 0x14), _read_lib(lib, reg_desc_off + 0x18)

_qcom_program_cache:dict[tuple[bytes, tuple[str, ...]], tuple[QCOMProgramData, UOp]] = {}
def qcom_build_program(dev:QCOMDevice, prg:UOp, devs:tuple[str, ...]) -> tuple[QCOMProgramData, UOp]:
  if (cached:=_qcom_program_cache.get(key:=(prg.src[3].arg, devs))) is None:
    data = QCOMProgramData(dev, prg.to_elf())
    image = bytes(data.image).ljust(round_up(len(data.image), 4), b"\x00")
    buf = UOp.placeholder((len(image),), dtypes.uint8, next(UOp.unique_num), device=devs).rtag("program")
    cached = _qcom_program_cache[key] = (data, patch(buf, [], image))
  return cached

class QCOMAllocator(Allocator['QCOMDevice']):
  def _alloc(self, size:int, options:BufferSpec) -> BufferStorage:
    return self.dev._gpu_map(options.external_ptr, size) if options.external_ptr else self.dev._gpu_alloc(size)

  def _free(self, storage:BufferStorage, options:BufferSpec):
    self.dev.synchronize()
    self.dev._gpu_free(storage)
  def _offset(self, buf:int, size:int, offset:int) -> int: return buf + offset

def flag(nm, val): return (val << getattr(kgsl, f"{nm}_SHIFT")) & getattr(kgsl, f"{nm}_MASK")

class QCOMDevice(Compiled):
  timestamp_divider = 19.2
  pm_encode = PatternMatcher([
    (UPat(Ops.CUSTOM_FUNCTION, arg="submit_qcom_compute", name="submit"), lambda ctx, submit: encode_submit(QCOMComputeQueue(ctx, submit))),
  ])

  @property
  def has_copy_queue(self) -> bool: return False

  def __init__(self, device:str=""):
    self.fd = FileIOInterface('/dev/kgsl-3d0', os.O_RDWR)

    flags = kgsl.KGSL_CONTEXT_PREAMBLE | kgsl.KGSL_CONTEXT_PWR_CONSTRAINT | kgsl.KGSL_CONTEXT_NO_FAULT_TOLERANCE | kgsl.KGSL_CONTEXT_NO_GMEM_ALLOC \
      | flag("KGSL_CONTEXT_PRIORITY", getenv("QCOM_PRIORITY", 8)) | flag("KGSL_CONTEXT_PREEMPT_STYLE", kgsl.KGSL_CONTEXT_PREEMPT_STYLE_FINEGRAIN)
    self.ctx = kgsl.IOCTL_KGSL_DRAWCTXT_CREATE(self.fd, flags=flags).drawctxt_id
    self._stack:Buffer|None = None # private-memory stack

    # Set max power
    struct.pack_into('IIQQ', pwr:=memoryview(bytearray(0x18)), 0, 1, self.ctx, mv_address(_:=memoryview(array.array('I', [1]))), 4)
    kgsl.IOCTL_KGSL_SETPROPERTY(self.fd, type=kgsl.KGSL_PROP_PWR_CONSTRAINT, value=mv_address(pwr), sizebytes=pwr.nbytes)

    # Load info about qcom device
    info = kgsl.struct_kgsl_devinfo()
    kgsl.IOCTL_KGSL_DEVICE_GETPROPERTY(self.fd, type=kgsl.KGSL_PROP_DEVICE_INFO, value=ctypes.addressof(info), sizebytes=ctypes.sizeof(info))
    self.gpu_id = (info.chip_id >> 24, (info.chip_id >> 16) & 0xFF, (info.chip_id >> 8) & 0xFF)

    # a7xx start with 730x or 'Cxxx', a8xx starts 'Exxx'
    if self.gpu_id[:2] >= (7, 3): raise RuntimeError(f"Unsupported GPU: chip_id={info.chip_id:#x}")

    if PROFILE and self.gpu_id[:2] < (7, 3):
      System.write_sysfs("/sys/class/kgsl/kgsl-3d0/idle_timer", value="4000000000", msg="Failed to disable suspend mode", expected="4294967276")

    super().__init__(device, QCOMAllocator(self), [QCOMCLRenderer, IR3Renderer], None,
                     arch=("a%d%d%d" + (",IMAGE_PITCH_ALIGNMENT=64" if IMAGE else "")) % self.gpu_id)

    self.var_vals = {"kgsl_fd": self.fd.fd, "kgsl_ctx": self.ctx}
    self.pm_bufferize = PatternMatcher([
      (UPat(Ops.PARAM, tag="stack", name="b"), lambda ctx, b: ctx._ensure_stack_size(b.max_numel())),
      (UPat(Ops.PARAM, tag="dummy"), lambda ctx: ctx.dummy),
      (UPat(Ops.PARAM, tag="border_color"), lambda ctx: ctx.border_color),
    ]) + self.pm_bufferize

  @functools.cached_property
  def dummy(self) -> Buffer: return Buffer(self.device, 0x1000, dtypes.uint8, options=BufferSpec(nolru=True), preallocate=True) # cache flush target

  @functools.cached_property
  def border_color(self) -> Buffer: # zeros: the samplers clamp to a black border
    return Buffer(self.device, 0x1000, dtypes.uint8, options=BufferSpec(nolru=True), initial_value=bytes(0x1000))

  def _gpu_alloc(self, size:int, flags:int=0, uncached=False, fill_zeroes=False) -> BufferStorage:
    flags |= flag("KGSL_MEMALIGN", alignment_hint:=12) | kgsl.KGSL_MEMFLAGS_USE_CPU_MAP
    if uncached: flags |= flag("KGSL_CACHEMODE", kgsl.KGSL_CACHEMODE_UNCACHED)

    alloc = kgsl.IOCTL_KGSL_GPUOBJ_ALLOC(self.fd, size=(bosz:=round_up(size, 1<<alignment_hint)), flags=flags, mmapsize=bosz)
    va_addr = self.fd.mmap(0, bosz, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, alloc.id * 0x1000)

    if fill_zeroes: ctypes.memset(va_addr, 0, size)
    return BufferStorage(va_addr, (alloc, True), MMIOInterface(va_addr, size, fmt='B'))

  def _gpu_map(self, ptr:int, size:int) -> BufferStorage:
    ptr_aligned, size_aligned = (ptr & ~0xfff), round_up(size + (ptr & 0xfff), 0x1000)
    dcache_flush().fxn(ctypes.c_uint64(ptr_line_aligned:=ptr & ~63), ceildiv(ptr + size - ptr_line_aligned, 64))
    try:
      mi = kgsl.IOCTL_KGSL_MAP_USER_MEM(self.fd, hostptr=ptr_aligned, len=size_aligned, memtype=kgsl.KGSL_USER_MEM_TYPE_ADDR)
      return BufferStorage(mi.gpuaddr + (ptr - ptr_aligned), (mi, False), MMIOInterface(ptr, size, fmt='B'))
    except OSError as e:
      if e.errno == 14: return BufferStorage(ptr, (None, False), MMIOInterface(ptr, size, fmt='B'))
      raise RuntimeError("Failed to map external pointer to GPU memory") from e

  def _gpu_free(self, storage:BufferStorage):
    if storage.meta[0] is None: return # external (gpu) ptr
    if not storage.meta[1]: kgsl.IOCTL_KGSL_SHAREDMEM_FREE(self.fd, gpuaddr=storage.meta[0].gpuaddr) # external (cpu) ptr
    else:
      kgsl.IOCTL_KGSL_GPUOBJ_FREE(self.fd, id=storage.meta[0].id)
      FileIOInterface.munmap(storage.buf, storage.meta[0].mmapsize)

  def _wait_signal(self, sig:MMIOInterface|memoryview, value:int, timeout:int|None=None):
    if sig[0] < value:
      ts = kgsl.IOCTL_KGSL_CMDSTREAM_READTIMESTAMP_CTXTID(self.fd, context_id=self.ctx, type=kgsl.KGSL_TIMESTAMP_QUEUED).timestamp
      with contextlib.suppress(OSError, RuntimeError):
        kgsl.IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID(self.fd, context_id=self.ctx, timestamp=ts, timeout=int(timeout or self.wait_timeout_ms))
    super()._wait_signal(sig, value, timeout)

  def _ensure_stack_size(self, sz:int) -> Buffer: # one stack for the device, grown to the deepest program's private memory
    if self._stack is None or self._stack.nbytes < sz:
      if self._stack is not None: self.synchronize()
      self._stack = Buffer(self.device, sz, dtypes.uint8, options=BufferSpec(nolru=True), preallocate=True)
    return self._stack

  def _at_profile_finalize(self):
    super()._at_profile_finalize()
    with contextlib.suppress(RuntimeError): System.write_sysfs("/sys/class/kgsl/kgsl-3d0/idle_timer", "10", "Failed to reenable suspend mode")
