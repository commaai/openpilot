from __future__ import annotations
from typing import cast, Any, Callable
import os, ctypes, struct, hashlib, functools, importlib, mmap, errno, array, contextlib, sys, weakref, itertools, collections, atexit
assert sys.platform != 'win32'
from dataclasses import dataclass
from extra.hcq2.hcq2 import HCQ2Compiled, HCQAllocator, HCQ2Buffer, encode_kernargs_clike, make_getaddr, make_ins, make_cmdbuf, make_placeholder
from extra.hcq2.hcq2 import make_binary_patch
from tinygrad.uop.ops import sint, UOp
from tinygrad.device import Compiled, BufferSpec, Buffer, Device
from tinygrad.dtype import dtypes
from tinygrad.helpers import getenv, round_up, data64_le, DEBUG, PROFILE, ProfileEvent, lo32, hi32, colored, prod, ContextVar, TracingKey
from tinygrad.helpers import VIZ, ceildiv, unwrap, pluralize, to_tuple
from tinygrad.renderer.cstyle import HIPRenderer, HIPCCRenderer
from tinygrad.renderer.llvmir import AMDLLVMRenderer
from tinygrad.runtime.autogen import kfd, hsa, sqtt, amdgpu_kd, amdgpu_drm
from tinygrad.runtime.autogen.am import am
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.runtime.support.hcq import FileIOInterface, HCQBuffer, MMIOInterface, hcq_filter_visible_devices
from tinygrad.runtime.support.am.amdev import AMDev, AMMemoryManager
from tinygrad.runtime.support.amd import AMDReg, AMDIP, import_module, import_soc, import_pmc
from tinygrad.runtime.support.system import PCIIfaceBase, PCIAllocationMeta, USBPCIDevice, MAP_FIXED, MAP_NORESERVE
from tinygrad.runtime.support.usb import USB3
from tinygrad.runtime.support.memory import AddrSpace, BumpAllocator
from tinygrad.runtime.ops_amd import SQTT, SQTT_ITRACE_SE_MASK, SQTT_LIMIT_SE, SQTT_SIMD_SEL, SQTT_TOKEN_EXCLUDE, PMC
from tinygrad.runtime.ops_amd import EVENT_INDEX_PARTIAL_FLUSH, WAIT_REG_MEM_FUNCTION_EQ, WAIT_REG_MEM_FUNCTION_NEQ, WAIT_REG_MEM_FUNCTION_GEQ
if getenv("IOCTL"): import extra.hip_gpu_driver.hip_ioctl  # noqa: F401 # pylint: disable=unused-import

from tinygrad.engine.realize import get_runtime, pm_flatten_linear
from tinygrad.uop import FastEnum, auto
from tinygrad.uop.ops import Ops, UPat, PatternMatcher, graph_rewrite

# *****************
# PM4

class PM4Ops(FastEnum):
  SET_SH_REG = auto(); SET_UCONFIG_REG = auto(); WAIT_REG_MEM = auto(); ACQUIRE_MEM = auto()  # noqa: E702
  RELEASE_MEM = auto(); DISPATCH_DIRECT = auto(); EVENT_WRITE = auto()  # noqa: E702

def pkt3(ctx, op:PM4Ops, *vals): return make_ins(op, ctx.pm4.PACKET3(getattr(ctx.pm4, f"PACKET3_{op.name}"), len(vals) - 1), *vals)

def wreg(ctx, reg:AMDReg, *args:sint, **kwargs:int):
  if bool(args) == bool(kwargs): raise RuntimeError('One (and only one) of *args or **kwargs must be specified')
  if ctx.pm4.PACKET3_SET_SH_REG_START <= reg.addr[0] < ctx.pm4.PACKET3_SET_SH_REG_END:
    op, set_packet_start = PM4Ops.SET_SH_REG, ctx.pm4.PACKET3_SET_SH_REG_START
  elif ctx.pm4.PACKET3_SET_UCONFIG_REG_START <= reg.addr[0] < ctx.pm4.PACKET3_SET_UCONFIG_REG_START + 2**16-1:
    op, set_packet_start = PM4Ops.SET_UCONFIG_REG, ctx.pm4.PACKET3_SET_UCONFIG_REG_START
  else: raise RuntimeError(f'Cannot set {reg.name} ({reg.addr[0]}) via pm4 packet')
  return pkt3(ctx, op, reg.addr[0] - set_packet_start, *(args or (reg.encode(**kwargs),)))

def wait_reg_mem(ctx, value, mask=0xffffffff, mem=None, reg=None, reg_done=0, op=WAIT_REG_MEM_FUNCTION_GEQ):
  wrm_info_dw = ctx.pm4.WAIT_REG_MEM_MEM_SPACE(int(mem is not None)) | ctx.pm4.WAIT_REG_MEM_OPERATION(int(mem is None and reg_done > 0)) \
              | ctx.pm4.WAIT_REG_MEM_FUNCTION(op) | ctx.pm4.WAIT_REG_MEM_ENGINE(0)
  return pkt3(ctx, PM4Ops.WAIT_REG_MEM, wrm_info_dw, *(data64_le(mem) if mem is not None else (reg, reg_done)), value, mask, 4)

def acquire_mem(ctx, addr=0x0, sz=(1 << 64)-1, gli=1, glm=1, glk=1, glv=1, gl1=1, gl2=1):
  if ctx.target[0] != 9:
    cache_flags_dw = ctx.pm4.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLI_INV(gli) \
                   | ctx.pm4.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLM_INV(glm) | ctx.pm4.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLM_WB(glm) \
                   | ctx.pm4.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLK_INV(glk) | ctx.pm4.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLK_WB(glk) \
                   | ctx.pm4.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLV_INV(glv) | ctx.pm4.PACKET3_ACQUIRE_MEM_GCR_CNTL_GL1_INV(gl1) \
                   | ctx.pm4.PACKET3_ACQUIRE_MEM_GCR_CNTL_GL2_INV(gl2) | ctx.pm4.PACKET3_ACQUIRE_MEM_GCR_CNTL_GL2_WB(gl2)
    return pkt3(ctx, PM4Ops.ACQUIRE_MEM, 0, *data64_le(sz), *data64_le(addr), 0, cache_flags_dw)
  cp_coher_cntl = ctx.pm4.PACKET3_ACQUIRE_MEM_CP_COHER_CNTL_SH_ICACHE_ACTION_ENA(gli) | \
                  ctx.pm4.PACKET3_ACQUIRE_MEM_CP_COHER_CNTL_SH_KCACHE_ACTION_ENA(glk) | \
                  ctx.pm4.PACKET3_ACQUIRE_MEM_CP_COHER_CNTL_TC_ACTION_ENA(gl2) | \
                  ctx.pm4.PACKET3_ACQUIRE_MEM_CP_COHER_CNTL_TCL1_ACTION_ENA(gl1) | \
                  ctx.pm4.PACKET3_ACQUIRE_MEM_CP_COHER_CNTL_TC_WB_ACTION_ENA(gl2)
  return pkt3(ctx, PM4Ops.ACQUIRE_MEM, cp_coher_cntl, *data64_le(sz), *data64_le(addr), 0x0000000A)

def release_mem(ctx, address=0x0, value=0, data_sel=0, int_sel=2, ctxid=0, cache_flush=False):
  if ctx.target[0] != 9:
    cache_flags_dw = 0 if not cache_flush else (ctx.pm4.PACKET3_RELEASE_MEM_GCR_GLV_INV | ctx.pm4.PACKET3_RELEASE_MEM_GCR_GL1_INV \
                   | ctx.pm4.PACKET3_RELEASE_MEM_GCR_GL2_INV | ctx.pm4.PACKET3_RELEASE_MEM_GCR_GLM_WB \
                   | ctx.pm4.PACKET3_RELEASE_MEM_GCR_GLM_INV | ctx.pm4.PACKET3_RELEASE_MEM_GCR_GL2_WB | ctx.pm4.PACKET3_RELEASE_MEM_GCR_SEQ)
    event_dw = ctx.pm4.PACKET3_RELEASE_MEM_EVENT_TYPE(ctx.pm4.CACHE_FLUSH_AND_INV_TS_EVENT) \
             | ctx.pm4.PACKET3_RELEASE_MEM_EVENT_INDEX(ctx.pm4.event_index__mec_release_mem__end_of_pipe)
    memsel_dw = ctx.pm4.PACKET3_RELEASE_MEM_DATA_SEL(data_sel) | ctx.pm4.PACKET3_RELEASE_MEM_INT_SEL(int_sel) \
              | ctx.pm4.PACKET3_RELEASE_MEM_DST_SEL(0)
  else:
    cache_flags_dw = 0 if not cache_flush else (ctx.pm4.EOP_TC_WB_ACTION_EN | ctx.pm4.EOP_TC_NC_ACTION_EN)
    event_dw = ctx.pm4.EVENT_TYPE(ctx.pm4.CACHE_FLUSH_AND_INV_TS_EVENT) | ctx.pm4.EVENT_INDEX(ctx.pm4.event_index__mec_release_mem__end_of_pipe)
    memsel_dw = ctx.pm4.DATA_SEL(data_sel) | ctx.pm4.INT_SEL(int_sel)
    ctxid = 0
  return pkt3(ctx, PM4Ops.RELEASE_MEM, event_dw | cache_flags_dw, memsel_dw, *data64_le(address), *data64_le(value), ctxid)

def memory_barrier(ctx):
  pf = '' if ctx.nbio.version[0] == 2 else '0' if ctx.nbio.version[:2] != (7, 11) else '1'
  return UOp(Ops.LINEAR, dtypes.void, (
    wait_reg_mem(ctx, reg=getattr(ctx.nbio, f'regBIF_BX_PF{pf}_GPU_HDP_FLUSH_REQ').addr[0],
                 reg_done=getattr(ctx.nbio, f'regBIF_BX_PF{pf}_GPU_HDP_FLUSH_DONE').addr[0], value=0xffffffff),
    acquire_mem(ctx)))

def pm4_wait(ctx, x, y): return wait_reg_mem(ctx, y, mem=make_getaddr(x.buf_uop, ctx.devs))

def pm4_barrier(ctx): return memory_barrier(ctx)

def pm4_store(ctx, dst, val):
  if val.op is Ops.BINARY: return None
  return release_mem(ctx, make_getaddr(dst, ctx.devs), val, ctx.pm4.data_sel__mec_release_mem__send_32_bit_low,
                     ctx.pm4.int_sel__mec_release_mem__send_interrupt_after_write_confirm, cache_flush=True)

def pm4_timestamp(ctx, dst):
  return release_mem(ctx, make_getaddr(dst, ctx.devs), 0, ctx.pm4.data_sel__mec_release_mem__send_gpu_clock_counter,
                     ctx.pm4.int_sel__mec_release_mem__none)

def pm4_program(ctx, call, prg):
  data, info = prg.arg
  lib_gpu = prg.src[0]
  args = encode_kernargs_clike(call, prg, ctx.devs)
  prog_addr = make_getaddr(lib_gpu, ctx.devs) + data.entry_point_offset
  scratch_addr = make_getaddr(make_placeholder(ctx.devs, data.private_segment_size, dtypes.uint8, "scratch", unique=False), ctx.devs)
  args_addr = make_getaddr(args, ctx.devs)

  user_regs = []
  if data.enable_private_segment_sgpr:
    scratch_hilo = data64_le(scratch_addr)
    user_regs = [scratch_hilo[0], scratch_hilo[1] | 1 << 31, 0xffffffff, 0x20c14000]
  if data.enable_dispatch_ptr: user_regs += [*data64_le(args_addr + data.kernargs_segment_size)]
  user_regs += [*data64_le(args_addr)]

  dispatch_init = ctx.gc.regCOMPUTE_DISPATCH_INITIATOR.encode(
    **({'cs_w32_en': int(data.wave32)} if ctx.target[0] != 9 else {}), force_start_at_000=1, compute_shader_en=1)
  ins = [acquire_mem(ctx, gli=0, gl2=0),
    wreg(ctx, ctx.gc.regCOMPUTE_PGM_LO, *data64_le(prog_addr >> 8)),
    wreg(ctx, ctx.gc.regCOMPUTE_PGM_RSRC1, data.rsrc1, data.rsrc2),
    wreg(ctx, ctx.gc.regCOMPUTE_PGM_RSRC3, data.rsrc3),
    wreg(ctx, ctx.gc.regCOMPUTE_TMPRING_SIZE, ctx.tmpring_size(data.private_segment_size))]
  ins += [wreg(ctx, ctx.gc.regCOMPUTE_DISPATCH_SCRATCH_BASE_LO, *data64_le((scratch_addr + data.private_segment_size // ctx.xccs * xcc_id) >> 8))
          for xcc_id in range(ctx.xccs)]
  ins += [wreg(ctx, ctx.gc.regCOMPUTE_RESTART_X, 0, 0, 0),
    wreg(ctx, ctx.gc.regCOMPUTE_USER_DATA_0, *user_regs),
    wreg(ctx, ctx.gc.regCOMPUTE_RESOURCE_LIMITS, ctx.gc.regCOMPUTE_RESOURCE_LIMITS.encode(waves_per_sh=getenv("WAVES_PER_SH"))),
    wreg(ctx, ctx.gc.regCOMPUTE_START_X, 0, 0, 0, *(info.local_size or (1, 1, 1)), 0, 0),
    pkt3(ctx, PM4Ops.DISPATCH_DIRECT, *info.global_size, dispatch_init),
    pkt3(ctx, PM4Ops.EVENT_WRITE, ctx.pm4.EVENT_TYPE(ctx.soc.CS_PARTIAL_FLUSH) | ctx.pm4.EVENT_INDEX(EVENT_INDEX_PARTIAL_FLUSH))]
  return UOp(Ops.LINEAR, dtypes.void, tuple(ins))

pm_pm4_opsel = PatternMatcher([
  (UPat(Ops.CALL, src=(UPat(Ops.PROGRAM, name="prg"),), name="call", allow_any_len=True), pm4_program),

  (UPat(Ops.WAIT, src=(UPat.var("x") >= UPat.var("y"),)), pm4_wait),
  (UPat(Ops.BARRIER), pm4_barrier),
  (UPat(Ops.CUSTOM_FUNCTION, arg="timestamp", src=(UPat(name="dst"),)), pm4_timestamp),
  (UPat(Ops.STORE, src=(UPat((Ops.BUFFER, Ops.PARAM), name="dst"), UPat(name="val"))), pm4_store),
])

def pm4_submit(cmdbuf, devs):
  size, zero = UOp.const(dtypes.uint32, cmdbuf.nbytes() // dtypes.uint32.itemsize), UOp.const(dtypes.int, 0)

  # the compute queue's ring and its host-side ring/write/put pointers (placeholders, resolved in pm_bufferize)
  for d in devs: q = Device[d].compute_queue
  ring, wptr, doorbell, put_ptr = (make_placeholder(devs, b.size, b.dtype, ("COMPUTE:0", name), unique=False)
    for name, b in (("ring", q.ring), ("write_ptr", q.write_ptr), ("doorbell", q.doorbell), ("put_value", q.put_value)))

  # place the cmdbuf at the ring's write offset, wrapping the ring
  put = put_ptr.index(zero)
  next_put = put + size.cast(put.dtype)
  i = UOp.range(size, 0, dtype=dtypes.int, src=(cmdbuf,))
  ring_idx = ((put + i.cast(put.dtype)) % q.ring.size).cast(dtypes.int)

  # copy the cmdbuf into the ring and advance the put/write pointers
  copy_to_ring = ring.index(ring_idx).store(cmdbuf.index(i).load()).end(i)
  bump_put_ptr = put_ptr.index(zero).store(next_put)
  bump_wptr = wptr.index(zero).store(next_put)

  # ring the doorbell once the copy and pointer bumps have landed
  flush = UOp.barrier(copy_to_ring, bump_put_ptr, bump_wptr)
  return doorbell.after(flush).index(zero).store(next_put)

pm_pm4_submit = PatternMatcher([(UPat(Ops.LINEAR, name="lin"),
  lambda lin: pm4_submit(make_cmdbuf(lin, to_tuple(lin.arg[0])), to_tuple(lin.arg[0])))])

# *****************
# SDMA

class SDMAOps(FastEnum): COPY = auto(); POLL_REGMEM = auto(); FENCE = auto(); TRAP = auto(); TIMESTAMP = auto()  # noqa: E702

def sdma_copy(ctx, call):
  dst, src = call.src[1], call.src[2]
  sz = src.max_numel() * src.dtype.itemsize
  src_addr, dst_addr = make_getaddr(src, ctx.devs), make_getaddr(dst, ctx.devs)
  return UOp(Ops.LINEAR, dtypes.void, tuple([make_ins(SDMAOps.COPY,
     ctx.sdma.SDMA_OP_COPY | ctx.sdma.SDMA_PKT_COPY_LINEAR_HEADER_SUB_OP(ctx.sdma.SDMA_SUBOP_COPY_LINEAR),
     ctx.sdma.SDMA_PKT_COPY_LINEAR_COUNT_COUNT(min(sz - off, ctx.max_copy_size) - 1), 0,
     *data64_le(src_addr + off), *data64_le(dst_addr + off)) for off in range(0, sz, ctx.max_copy_size)]))

def sdma_wait(ctx, x, y):
  op = ctx.sdma.SDMA_OP_POLL_REGMEM | ctx.sdma.SDMA_PKT_POLL_REGMEM_HEADER_FUNC(WAIT_REG_MEM_FUNCTION_GEQ) \
     | ctx.sdma.SDMA_PKT_POLL_REGMEM_HEADER_MEM_POLL(1)
  return make_ins(SDMAOps.POLL_REGMEM, op, *data64_le(make_getaddr(x.buf_uop, ctx.devs)), y, 0xffffffff,
    ctx.sdma.SDMA_PKT_POLL_REGMEM_DW5_INTERVAL(0x04) | ctx.sdma.SDMA_PKT_POLL_REGMEM_DW5_RETRY_COUNT(0xfff))

def sdma_store(ctx, dst, val):
  op = ctx.sdma.SDMA_OP_FENCE | (ctx.sdma.SDMA_PKT_FENCE_HEADER_MTYPE(3) if ctx.target[0] != 9 else 0)
  return UOp(Ops.LINEAR, dtypes.void, (
    make_ins(SDMAOps.FENCE, op, *data64_le(make_getaddr(dst, ctx.devs)), val), make_ins(SDMAOps.TRAP, ctx.sdma.SDMA_OP_TRAP, 0)))

def sdma_timestamp(ctx, dst):
  op = ctx.sdma.SDMA_OP_TIMESTAMP | ctx.sdma.SDMA_PKT_TIMESTAMP_GET_HEADER_SUB_OP(ctx.sdma.SDMA_SUBOP_TIMESTAMP_GET_GLOBAL)
  return make_ins(SDMAOps.TIMESTAMP, op, *data64_le(make_getaddr(dst, ctx.devs)))

pm_sdma_opsel = PatternMatcher([
  (UPat(Ops.CALL, src=(UPat(Ops.COPY),), name="call", allow_any_len=True), sdma_copy),

  (UPat(Ops.BARRIER), lambda: UOp(Ops.NOOP, dtypes.void, ())),
  (UPat(Ops.WAIT, src=(UPat.var("x") >= UPat.var("y"),)), sdma_wait),
  (UPat(Ops.CUSTOM_FUNCTION, arg="timestamp", src=(UPat(name="dst"),)), sdma_timestamp),
  (UPat(Ops.STORE, src=(UPat((Ops.BUFFER, Ops.PARAM), name="dst"), UPat(name="val"))), sdma_store),
])

def sdma_submit(cmdbuf, devs):
  # the cmdbuf to submit + the patch writes that fill it
  size_dw, zero = cmdbuf.nbytes() // dtypes.uint32.itemsize, UOp.const(dtypes.int, 0)

  # the sdma queue's ring and its host-side ring/write/put pointers
  for d in devs: q = Device[d].sdma_queue(0)
  ring, wptr, doorbell, put_ptr = (make_placeholder(devs, b.size, b.dtype, ("COPY:0", name), unique=False)
    for name, b in (("ring", q.ring), ("write_ptr", q.write_ptr), ("doorbell", q.doorbell), ("put_value", q.put_value)))

  # sdma needs the cmdbuf contiguous: if it won't fit before the ring end, restart at 0 and zero the tail
  put_b = put_ptr.index(zero)
  tail_off_dw = ((put_b % (q.ring.size * 4)) // 4).cast(dtypes.int)
  fits = (size_dw <= q.ring.size - tail_off_dw).cast(dtypes.int)
  start_dw = fits * tail_off_dw
  zero_amt_dw = (1 - fits) * (q.ring.size - tail_off_dw)

  # zero the wrapped tail, then copy the cmdbuf into the ring
  zi = UOp.range(zero_amt_dw, 0, dtype=dtypes.int, src=(cmdbuf,))
  zero_tail = ring.index(tail_off_dw + zi).store(UOp.const(dtypes.uint32, 0)).end(zi)
  i = UOp.range(UOp.const(dtypes.int, size_dw), 0, dtype=dtypes.int, src=(cmdbuf,))
  copy_to_ring = ring.index(start_dw + i).store(cmdbuf.index(i).load()).end(i)

  # advance the put/write pointers past the zeroed tail and the cmdbuf
  next_put_b = put_b + ((zero_amt_dw + size_dw) * 4).cast(put_b.dtype)
  bump_put_ptr = put_ptr.index(zero).store(next_put_b)
  bump_wptr = wptr.index(zero).store(next_put_b)

  # ring the doorbell once the writes have landed
  flush = UOp.barrier(zero_tail, copy_to_ring, bump_put_ptr, bump_wptr)
  return doorbell.after(flush).index(zero).store(next_put_b)

pm_sdma_submit = PatternMatcher([(UPat(Ops.LINEAR, name="lin"),
  lambda lin: sdma_submit(make_cmdbuf(lin, to_tuple(lin.arg[0])), to_tuple(lin.arg[0])))])

@dataclass(frozen=True)
class AMDEncodeCtx:  # encode-time constants for one queue: devs (every cmdbuf address resolves into these) + gfx version + packet/ip modules
  devs: tuple[str, ...]; target: tuple[int, ...]; pm4: Any; sdma: Any; soc: Any  # noqa: E702
  gc: AMDIP; nbio: AMDIP; xccs: int; max_copy_size: int; tmpring_size: Callable  # noqa: E702

def encode_queue(q:UOp) -> UOp|None:
  d = Device[(devs:=to_tuple(q.arg[0]))[0]]
  ctx = AMDEncodeCtx(devs, d.target, d.pm4, d.sdma, d.soc, d.gc, d.nbio, d.xccs, d.max_copy_size, d.tmpring_size)
  opsel, submit = (pm_pm4_opsel, pm_pm4_submit) if q.arg[1].startswith("COMPUTE") else (pm_sdma_opsel, pm_sdma_submit)
  return submit.rewrite(graph_rewrite(q, opsel + pm_flatten_linear, walk=True, ctx=ctx, name=f"{q.arg[1]} opsel"))

@dataclass(frozen=True)
class AMDProgramData:
  entry_point_offset:int; rsrc1:int; rsrc2:int; rsrc3:int; wave32:bool
  private_segment_size:int; kernargs_segment_size:int; kernargs_alloc_size:int
  enable_dispatch_ptr:int; enable_private_segment_sgpr:int

_amd_program_cache:dict[tuple[bytes,str], tuple[AMDProgramData,bytes]] = {}
def amd_build_program(prg:UOp) -> UOp:
  dev = Device[to_tuple(prg.device)[0]] # TODO: rm this
  if (cached:=_amd_program_cache.get(key:=(lib:=prg.src[3].arg, dev.device))) is None:
    image, sections, relocs = elf_loader(lib)
    rodata = next(sh.header.sh_addr for sh in sections if sh.name == ".rodata")
    for off, sym, typ, addent in relocs:
      assert typ == 5, f"unknown AMD reloc {typ}"  # R_AMDGPU_REL64
      image[off:off+8] = struct.pack('<q', sym - off + addent)
    desc = amdgpu_kd.llvm_amdhsa_kernel_descriptor_t.from_buffer_copy(bytes(image[rodata:rodata+ctypes.sizeof(amdgpu_kd.llvm_amdhsa_kernel_descriptor_t)]))
    if (lds:=((desc.group_segment_fixed_size+511)//512)&0x1FF) > (dev.iface.props['lds_size_in_kb']*1024)//512:
      raise RuntimeError("Too many resources requested: group_segment_size")
    edp = desc.kernel_code_properties & hsa.AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_PTR

    data = AMDProgramData(entry_point_offset=rodata + desc.kernel_code_entry_byte_offset,
      rsrc1=desc.compute_pgm_rsrc1 | ((1<<20) if dev.target[0]==11 else 0),  # priv=1 on gfx11 for cwsr
      rsrc2=desc.compute_pgm_rsrc2 | (lds<<15), rsrc3=desc.compute_pgm_rsrc3,
      wave32=bool(desc.kernel_code_properties & 0x400), private_segment_size=desc.private_segment_fixed_size, kernargs_segment_size=desc.kernarg_size,
      kernargs_alloc_size=desc.kernarg_size + (ctypes.sizeof(hsa.hsa_kernel_dispatch_packet_t) if edp else 0), enable_dispatch_ptr=edp,
      enable_private_segment_sgpr=desc.kernel_code_properties & hsa.AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER)
    buf = make_placeholder(prg.device, len(image), dtypes.uint8, "program")
    cached = _amd_program_cache[key] = prg.replace(src=(buf.after(make_binary_patch(buf, bytes(image))),), arg=(data, prg.arg))
  return cached

class AMDAllocator(HCQAllocator['AMDDevice']):
  def __init__(self, dev:AMDDevice):
    super().__init__(dev, supports_copy_from_disk=dev.has_sdma_queue, supports_transfer=dev.has_sdma_queue and not dev.is_usb())

  def _alloc(self, size:int, options:BufferSpec) -> HCQ2Buffer:
    return self.dev.iface.alloc(size, host=options.host, uncached=options.uncached, cpu_access=options.cpu_access or not self.dev.has_sdma_queue)

  def _do_free(self, opaque, options:BufferSpec): self.dev.iface.free(opaque)

  def _do_map(self, buf:HCQ2Buffer): return self.dev.iface.map(buf._base if buf._base is not None else buf)

@dataclass
class AMDQueueDesc:
  ring: Buffer; read_ptr: Buffer; write_ptr: Buffer; doorbell: Buffer; put_value: Buffer  # noqa: E702
  eop_buffer: Buffer|None = None; cwsr_buffer: Buffer|None = None; params: tuple|None = None  # noqa: E702

class KFDIface:
  kfd:FileIOInterface|None = None
  event_page:HCQBuffer|None = None
  gpus:list[FileIOInterface] = []
  count:int = 0

  def _is_usable_gpu(self, gpu_id):
    with contextlib.suppress(OSError): return int(gpu_id.read()) != 0
    return False

  def __init__(self, dev, device_id):
    self.dev = dev

    kfd_topo_path = "/sys/devices/virtual/kfd/kfd/topology/nodes"

    # Initialize KFD interface during first run
    if KFDIface.kfd is None:
      KFDIface.kfd = FileIOInterface("/dev/kfd", os.O_RDWR)
      gpus = [g for g in FileIOInterface(kfd_topo_path).listdir() if self._is_usable_gpu(FileIOInterface(f"{kfd_topo_path}/{g}/gpu_id"))]
      KFDIface.gpus = hcq_filter_visible_devices(sorted(gpus, key=lambda x: int(x.split('/')[-1])), "AMD")
      KFDIface.count = len(KFDIface.gpus)

    if device_id >= len(KFDIface.gpus): raise RuntimeError(f"No device found for {device_id}. Requesting more devices than the system has?")

    self.gpu_id = int(FileIOInterface(f"{kfd_topo_path}/{KFDIface.gpus[device_id]}/gpu_id").read())
    self.props = {(p:=l.split())[0]: int(p[1]) for l in FileIOInterface(f"{kfd_topo_path}/{KFDIface.gpus[device_id]}/properties").read().splitlines()}
    self.dev_sysfs_path = f"/sys/class/drm/renderD{self.props['drm_render_minor']}/device"
    ip_base = f"{self.dev_sysfs_path}/ip_discovery/die/0"
    id2ip = {am.GC_HWID: am.GC_HWIP, am.SDMA0_HWID: am.SDMA0_HWIP, am.NBIF_HWID: am.NBIF_HWIP}
    ip_hw = [(id2ip[int(hwid)], int(hwid)) for hwid in FileIOInterface(ip_base).listdir() if hwid.isnumeric() and int(hwid) in id2ip]
    self.ip_versions = {ip:tuple(int(FileIOInterface(f'{ip_base}/{hw}/0/{part}').read()) for part in ['major','minor','revision']) for ip,hw in ip_hw}
    self.drm_fd = FileIOInterface(f"/dev/dri/renderD{self.props['drm_render_minor']}", os.O_RDWR)

    self.kfd_ver = ((ver_st:=kfd.AMDKFD_IOC_GET_VERSION(KFDIface.kfd)).major_version, ver_st.minor_version)
    kfd.AMDKFD_IOC_ACQUIRE_VM(KFDIface.kfd, drm_fd=self.drm_fd.fd, gpu_id=self.gpu_id)
    if self.kfd_ver >= (1,14): kfd.AMDKFD_IOC_RUNTIME_ENABLE(KFDIface.kfd, mode_mask=0)

    # Set these for our device.
    if KFDIface.event_page is None:
      KFDIface.event_page = self.alloc(0x8000, uncached=True)
      kfd.AMDKFD_IOC_CREATE_EVENT(KFDIface.kfd, event_page_offset=KFDIface.event_page.meta.handle)
    else: self.map(KFDIface.event_page)

    # Event to wait for queues completion
    self.dev.queue_event = kfd.AMDKFD_IOC_CREATE_EVENT(KFDIface.kfd, event_type=kfd.KFD_IOC_EVENT_SIGNAL, auto_reset=1)
    self.dev.queue_event_mailbox_ptr = KFDIface.event_page.va_addr + self.dev.queue_event.event_slot_index * 8

    # OS events to collect memory and hardware faults
    self.mem_fault_event = kfd.AMDKFD_IOC_CREATE_EVENT(KFDIface.kfd, event_type=kfd.KFD_IOC_EVENT_MEMORY)
    self.hw_fault_event = kfd.AMDKFD_IOC_CREATE_EVENT(KFDIface.kfd, event_type=kfd.KFD_IOC_EVENT_HW_EXCEPTION)

    self.queue_event_arr = (kfd.struct_kfd_event_data * 3)(kfd.struct_kfd_event_data(event_id=self.dev.queue_event.event_id),
      kfd.struct_kfd_event_data(event_id=self.mem_fault_event.event_id), kfd.struct_kfd_event_data(event_id=self.hw_fault_event.event_id))
    self.queue_event_arr_ptr = ctypes.addressof(self.queue_event_arr)

  def alloc(self, size:int, host=False, uncached=False, cpu_access=False, contiguous=False, cpu_addr=None) -> HCQBuffer:
    flags = kfd.KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE | kfd.KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE | kfd.KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE

    if uncached: flags |= kfd.KFD_IOC_ALLOC_MEM_FLAGS_COHERENT | kfd.KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED | kfd.KFD_IOC_ALLOC_MEM_FLAGS_GTT
    else: flags |= (kfd.KFD_IOC_ALLOC_MEM_FLAGS_USERPTR if host else kfd.KFD_IOC_ALLOC_MEM_FLAGS_VRAM)

    # Make mapped cpu address to be uncachable
    if cpu_addr is not None: flags |= kfd.KFD_IOC_ALLOC_MEM_FLAGS_COHERENT | kfd.KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED

    if cpu_access or host: flags |= kfd.KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC

    if flags & kfd.KFD_IOC_ALLOC_MEM_FLAGS_USERPTR:
      buf = addr = cpu_addr or FileIOInterface.anon_mmap(0, size, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED | mmap.MAP_ANONYMOUS, 0)
    else: buf, addr = 0, FileIOInterface.anon_mmap(0, size, 0, mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS | MAP_NORESERVE, 0)

    try: mem = kfd.AMDKFD_IOC_ALLOC_MEMORY_OF_GPU(self.kfd, va_addr=addr, size=size, gpu_id=self.gpu_id, flags=flags, mmap_offset=buf)
    except OSError as e:
      if e.errno == errno.EINVAL and (flags & kfd.KFD_IOC_ALLOC_MEM_FLAGS_VRAM) and cpu_access:
        raise MemoryError("Cannot allocate host-visible VRAM. Ensure the resizable BAR option is enabled on your system.") from e
      if e.errno == errno.ENOMEM: raise MemoryError(f"Cannot allocate {size} bytes: no memory is available.") from e
      raise

    if not (flags & kfd.KFD_IOC_ALLOC_MEM_FLAGS_USERPTR):
      buf = self.drm_fd.mmap(mem.va_addr, mem.size, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED | MAP_FIXED, mem.mmap_offset)
      assert addr == buf == mem.va_addr

    view = MMIOInterface(mem.va_addr, mem.size, fmt='B') if cpu_access or host else None
    self.map(hcqbuf:=HCQBuffer(mem.va_addr, mem.size, meta=mem, view=view, owner=self.dev))
    return hcqbuf

  def free(self, mem):
    gpus = (ctypes.c_int32 * 1)(self.gpu_id)
    stm = kfd.AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU(self.kfd, handle=mem.meta.handle, device_ids_array_ptr=ctypes.addressof(gpus), n_devices=1)
    assert stm.n_success == 1
    if mem.owner == self.dev:
      if mem.va_addr: FileIOInterface.munmap(mem.va_addr, mem.size)
      kfd.AMDKFD_IOC_FREE_MEMORY_OF_GPU(self.kfd, handle=mem.meta.handle)

  def map(self, mem):
    if mem.owner is not None and mem.owner._is_cpu(): return self.alloc(mem.size, host=True, cpu_addr=mem.va_addr)

    c_gpus = (ctypes.c_int32 * 1)(self.gpu_id)
    stm = kfd.AMDKFD_IOC_MAP_MEMORY_TO_GPU(self.kfd, handle=mem.meta.handle, device_ids_array_ptr=ctypes.addressof(c_gpus), n_devices=1)
    assert stm.n_success == 1
    return HCQBuffer(mem.va_addr, mem.size, meta=mem.meta, owner=mem.owner)

  def create_queue(self, queue_type, ring, gart, rptr, wptr, eop_buffer=None, cwsr_buffer=None, ctl_stack_size=0, ctx_save_restore_size=0,
                   xcc_id=0, idx=0):
    queue = kfd.AMDKFD_IOC_CREATE_QUEUE(KFDIface.kfd, ring_base_address=ring._buf.va_addr, ring_size=ring._buf.size, gpu_id=self.gpu_id,
      queue_type=queue_type, queue_percentage=kfd.KFD_MAX_QUEUE_PERCENTAGE|(xcc_id<<8), queue_priority=getenv("AMD_KFD_QUEUE_PRIORITY", 7),
      eop_buffer_address=eop_buffer._buf.va_addr if eop_buffer else 0, eop_buffer_size=eop_buffer._buf.size if eop_buffer else 0,
      ctl_stack_size=ctl_stack_size, ctx_save_restore_address=cwsr_buffer._buf.va_addr if cwsr_buffer else 0, ctx_save_restore_size=ctx_save_restore_size,
      write_pointer_address=gart._buf.va_addr+wptr, read_pointer_address=gart._buf.va_addr+rptr+8*xcc_id)

    if not hasattr(self, 'doorbells'):
      self.doorbells_base = queue.doorbell_offset & (~0x1fff) # doorbell is two pages
      self.doorbells = cast(FileIOInterface, KFDIface.kfd).mmap(0, 0x2000, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED, self.doorbells_base)

    (put_value := Buffer("CPU", 1, dtypes.uint64, preallocate=True))._buf.view.view(fmt='Q')[0] = 0
    doorbell = Buffer("CPU", 1, dtypes.uint64,
      options=BufferSpec(external_ptr=self.doorbells + queue.doorbell_offset - self.doorbells_base), preallocate=True)
    return AMDQueueDesc(ring=ring, doorbell=doorbell, read_ptr=gart.view(1, dtypes.uint64, rptr+8*xcc_id).ensure_allocated(),
      write_ptr=gart.view(1, dtypes.uint64, wptr).ensure_allocated(), put_value=put_value, eop_buffer=eop_buffer, cwsr_buffer=cwsr_buffer)

  def sleep(self, tm:int):
    kfd.AMDKFD_IOC_WAIT_EVENTS(KFDIface.kfd, events_ptr=self.queue_event_arr_ptr, num_events=3, wait_for_all=0, timeout=tm)
    if self.queue_event_arr[1].memory_exception_data.gpu_id or self.queue_event_arr[2].hw_exception_data.gpu_id: self.on_device_hang()

  def on_device_hang(self):
    def _str(st): return ' '.join(f'{k[0]}={getattr(st, k[0])}' for k in st._real_fields_)

    # try to collect fault info if not already set from sleep().
    if not self.queue_event_arr[1].memory_exception_data.gpu_id and not self.queue_event_arr[2].hw_exception_data.gpu_id:
      with contextlib.suppress(RuntimeError): self.sleep(tm=1)

    report = []
    if self.queue_event_arr[1].memory_exception_data.gpu_id:
      report += [f"MMU fault: 0x{self.queue_event_arr[1].memory_exception_data.va:X} | {_str(self.queue_event_arr[1].memory_exception_data.failure)}"]
    if self.queue_event_arr[2].hw_exception_data.gpu_id: report += [f"HW fault: {_str(self.queue_event_arr[2].hw_exception_data)}"]

    raise RuntimeError("\n".join(report))

  def require_profile_mode(self, can_set_mode=True):
    if self.dev.target[0] == 9: return
    fn = f'{self.dev_sysfs_path}/power_dpm_force_performance_level'
    if (perflevel:=FileIOInterface(fn).read().strip()) != 'profile_standard':
      if can_set_mode:
        atexit.register(lambda: os.system(f"echo '{perflevel}' | sudo tee {fn} > /dev/null"))
        os.system(f"echo 'profile_standard' | sudo tee {fn} > /dev/null")
        self.require_profile_mode(can_set_mode=False)
      else:
        raise RuntimeError("PMC/SQTT requires stable power state: run `amd-smi set -l stable_std` for KFD iface")

  @functools.cached_property
  def drm_dev_info(self) -> amdgpu_drm.struct_drm_amdgpu_info_device:
    amdgpu_drm.DRM_IOCTL_AMDGPU_INFO(self.drm_fd, query=amdgpu_drm.AMDGPU_INFO_DEV_INFO,
      return_pointer=ctypes.addressof(inf:=amdgpu_drm.struct_drm_amdgpu_info_device()), return_size=ctypes.sizeof(inf))
    return inf
  def is_wgp_active(self, xcc, se, sa, wgp) -> bool: return ((self.drm_dev_info.cu_bitmap[se % 4][sa + (se // 4) * 2] >> (2 * wgp)) & 0x3) == 0x3

class PCIIface(PCIIfaceBase):
  def __init__(self, dev, dev_id):
    super().__init__(dev, dev_id, vendor=0x1002, devices=((0xffff, (0x74a1,0x744c,0x7480,0x7550,0x7551,0x7590,0x75a0)),), vram_bar=0,
      va_start=AMMemoryManager.va_allocator.base, va_size=AMMemoryManager.va_allocator.size, dev_impl_t=AMDev)
    self._compute_props()

  def p2p_paddrs(self, paddrs:list[tuple[int,int]]) -> tuple[list[tuple[int,int]], AddrSpace]:
    return ([(self.dev_impl.paddr2xgmi(p), sz) for p, sz in paddrs], AddrSpace.PEER) if self.dev_impl.is_hive() else super().p2p_paddrs(paddrs)

  def require_profile_mode(self): return True
  def is_wgp_active(self, xcc, se, sa, wgp) -> bool: return True # TODO: account for WGP disablement on some asics.

  def _compute_props(self):
    self.ip_versions = self.dev_impl.ip_ver

    gfxver = int(f"{self.dev_impl.ip_ver[am.GC_HWIP][0]:02d}{self.dev_impl.ip_ver[am.GC_HWIP][1]:02d}{self.dev_impl.ip_ver[am.GC_HWIP][2]:02d}")
    if self.dev_impl.gc_info.header.version_major == 2:
      cu_per_sa = self.dev_impl.gc_info.gc_num_cu_per_sh
      max_sh_per_se = self.dev_impl.gc_info.gc_num_sh_per_se
    else:
      cu_per_sa = 2 * (self.dev_impl.gc_info.gc_num_wgp0_per_sa + self.dev_impl.gc_info.gc_num_wgp1_per_sa)
      max_sh_per_se = self.dev_impl.gc_info.gc_num_sa_per_se

    array_count = max_sh_per_se * self.dev_impl.gc_info.gc_num_se * self.dev_impl.gfx.xccs
    self.props = {'cu_per_simd_array': cu_per_sa, 'simd_count': 2 * cu_per_sa * array_count, 'simd_per_cu': 2, 'array_count': array_count,
      'max_slots_scratch_cu': self.dev_impl.gc_info.gc_max_scratch_slots_per_cu, 'max_waves_per_simd': self.dev_impl.gc_info.gc_max_waves_per_simd,
      'simd_arrays_per_engine': max_sh_per_se, 'lds_size_in_kb': self.dev_impl.gc_info.gc_lds_size, 'num_xcc': self.dev_impl.gfx.xccs,
      'gfx_target_version': {90403: 90402}.get(gfxver, gfxver)}

  def create_queue(self, queue_type, ring, gart, rptr, wptr, eop_buffer=None, cwsr_buffer=None, ctl_stack_size=0, ctx_save_restore_size=0,
                   xcc_id=0, idx=0):
    assert cwsr_buffer is None, "no cwsr buffer for am"

    rcvr_params: tuple
    if queue_type == kfd.KFD_IOC_QUEUE_TYPE_SDMA:
      doorbell_index = self.dev_impl.sdma.setup_ring(*(rcvr_params:=(ring._buf.va_addr, ring._buf.size, gart._buf.va_addr+rptr,
        gart._buf.va_addr+wptr, idx)))
    else:
      doorbell_index = self.dev_impl.gfx.setup_ring(*(rcvr_params:=(ring._buf.va_addr, ring._buf.size, gart._buf.va_addr+rptr,
        gart._buf.va_addr+wptr, eop_buffer._buf.va_addr, eop_buffer._buf.size, is_aql:=(queue_type==kfd.KFD_IOC_QUEUE_TYPE_COMPUTE_AQL), is_aql)))

    (put_value := Buffer("CPU", 1, dtypes.uint64, preallocate=True))._buf.view.view(fmt='Q')[0] = 0
    doorbell = Buffer("CPU", 1, dtypes.uint64, options=BufferSpec(external_ptr=self.dev_impl.doorbell64.addr + doorbell_index*8), preallocate=True)
    return AMDQueueDesc(ring=ring, doorbell=doorbell, read_ptr=gart.view(1, dtypes.uint64, rptr).ensure_allocated(),
      write_ptr=gart.view(1, dtypes.uint64, wptr).ensure_allocated(), put_value=put_value, eop_buffer=eop_buffer, params=rcvr_params)

  def _collect_interrupts(self, reset=False, drain_only=False):
    d = self.dev
    if drain_only: d.iface.dev_impl.ih.drain()
    else: d.iface.dev_impl.ih.interrupt_handler()

    if reset and d.iface.dev_impl.recover():
      cq = d.compute_queue
      for b in (cq.put_value, cq.read_ptr, cq.write_ptr): b._buf.view.view(fmt='Q')[0] = 0
      d.iface.dev_impl.gfx.setup_ring(*cq.params)
      d.timeline_signal()._buf.cpu_view().mv.cast('Q')[0] = d.timeline_value().as_memoryview(force_zero_copy=True).cast('Q')[0] - 1

  def sleep(self, timeout):
    if hasattr(self.pci_dev, 'irq_poller') and self.pci_dev.irq_poller is not None and (events_cnt:=len(self.pci_dev.irq_poller.poll(timeout))):
      self.pci_dev.irq_fd.read(8 * events_cnt)
    self._collect_interrupts()
    if self.dev_impl.is_err_state: raise RuntimeError("Device is in error state")

  def on_device_hang(self):
    self._collect_interrupts(reset=True)
    raise RuntimeError("Device hang detected")

  def device_fini(self): self.dev_impl.fini()

def _mock(iface, name=None): return type(name or f"MOCK{iface.__name__}", (iface,), {})

class AMDDevice(HCQ2Compiled):
  pm_lower = PatternMatcher([
    # prep program
    (UPat(Ops.PROGRAM, src=(UPat(), UPat(), UPat(), UPat(Ops.BINARY)), name="prg"), amd_build_program),

    # encoding of cmdbuf
    (UPat(Ops.CUSTOM_FUNCTION, arg="submit_cmdbuf", src=(UPat(Ops.LINEAR, name="q"),)), encode_queue),
  ])

  timestamp_divider = 100.0  # AMD GPU clock: ticks/us

  ifaces = [KFDIface, PCIIface, _mock(KFDIface, "MOCKIface"), _mock(KFDIface), _mock(PCIIface)]

  def is_am(self) -> bool: return isinstance(self.iface, (PCIIface,))
  def is_usb(self) -> bool: return False

  def __init__(self, device:str=""):
    self.device_id = int(device.split(":")[1]) if ":" in device else 0

    self.iface = self._select_iface()

    self.target:tuple[int, ...] = ((trgt:=self.iface.props['gfx_target_version']) // 10000, (trgt // 100) % 100, trgt % 100)
    self.arch = "gfx%d%x%x" % self.target
    assert (self.target in ((9,4,2),(9,5,0))) or self.target[0] in (11, 12), f"Unsupported arch: {self.arch}"
    if DEBUG >= 1: print(f"AMDDevice: opening {self.device_id} with target {self.target} arch {self.arch}")

    self.xccs = self.iface.props.get('num_xcc', 1)
    self.se_cnt = self.iface.props['array_count'] // self.iface.props['simd_arrays_per_engine'] // self.xccs
    self.cu_cnt = self.iface.props['simd_count'] // self.iface.props['simd_per_cu'] // self.xccs
    self.waves_per_cu = self.iface.props['max_waves_per_simd'] * self.iface.props['simd_per_cu']
    self.wave_cnt = (self.cu_cnt * self.waves_per_cu) if self.target[0] != 9 else min(self.cu_cnt * 40, self.se_cnt * self.xccs * 512)

    self.ip_off = importlib.import_module(f"tinygrad.runtime.autogen.am.{'vega' if self.target[0] == 9 else 'navi'}_offsets")
    self.soc = import_soc(self.target)
    self.pm4 = importlib.import_module(f"tinygrad.runtime.autogen.am.pm4_{'soc15' if self.target[0] == 9 else 'nv'}")
    self.sdma = import_module('sdma', min(self.iface.ip_versions[am.SDMA0_HWIP], (6, 0, 0)))
    self.gc = AMDIP('gc', self.iface.ip_versions[am.GC_HWIP],
                    bases={i: tuple(getattr(self.ip_off, f'GC_BASE__INST{i}_SEG{s}', 0) for s in range(6)) for i in range(6)})

    self.nbio = AMDIP('nbio' if self.target[0] < 12 else 'nbif', self.iface.ip_versions[am.NBIF_HWIP],
                      bases={i: tuple(getattr(self.ip_off, f'NBIO_BASE__INST{i}_SEG{s}', 0) for s in range(9)) for i in range(6)})

    self.is_aql = getenv("AMD_AQL", int(self.xccs > 1))
    if self.is_aql:
      self.pm4_ibs = self.iface.alloc(0x2000 if self.is_usb() else (16 << 20), uncached=True, cpu_access=True)
      self.pm4_ib_alloc = BumpAllocator(self.pm4_ibs.size, wrap=True)

    self.max_copy_size = 0x40000000 if self.iface.ip_versions[am.SDMA0_HWIP][0] >= 5 else 0x400000
    self.sdma_queues:dict = {}
    self.has_sdma_queue = True # self.sdma_queue(0) is not None, TODO: think of this

    super().__init__(device, AMDAllocator(self), [HIPRenderer, AMDLLVMRenderer, HIPCCRenderer], None, can_recover=self.is_am(), arch=self.arch)

    # Scratch setup
    self.max_private_segment_size = 0
    self.pm_bufferize = PatternMatcher([(UPat(Ops.PARAM, tag="scratch", name="b"), lambda ctx, b: ctx[0].scratch_buffer(b.max_numel()))]) + self.pm_bufferize

    self.pmc_enabled:bool = PROFILE > 0 and PMC > 0
    if self.pmc_enabled:
      self.iface.require_profile_mode()

      self.pmc_sched:list[PMCSample] = []
      self.pmc_counters = import_pmc(self.target)

      # validate counters: SQ for SIMD busy/instruction counts, LDS stats, GRBM for GPU cycles, L2 cache hits/misses
      l2, lds = ("TCC", "SQ") if self.target[0] == 9 else ("GL2C", "SQC")
      pmc_default = f"SQ_BUSY_CYCLES,SQ_INSTS_VALU,SQ_INSTS_SALU,{lds}_LDS_IDX_ACTIVE,{lds}_LDS_BANK_CONFLICT,GRBM_GUI_ACTIVE,{l2}_HIT,{l2}_MISS"
      for k in (PMC_COUNTERS:=getenv("PMC_COUNTERS", pmc_default).split(",")):
        if k not in self.pmc_counters: raise RuntimeError(f"PMC counter {k} is not supported. Available: {','.join(self.pmc_counters.keys())}")

      raise NotImplementedError("PMC start not migrated to hcq2 yet")

    # SQTT is disabled by default because of runtime overhead and big file sizes (~200mb to Tensor.full() two 4096x4096 tensors and matmul them)
    self.sqtt_enabled:bool = PROFILE > 0 and SQTT > 0
    if self.sqtt_enabled:
      self.iface.require_profile_mode()

      SQTT_BUFFER_SIZE = getenv("SQTT_BUFFER_SIZE", 256) # in mb, per shader engine
      self.sqtt_buffers = [self.allocator.alloc(SQTT_BUFFER_SIZE<<20, BufferSpec(nolru=True, uncached=True)) for _ in range(self.se_cnt * self.xccs)]
      self.sqtt_wptrs = self.allocator.alloc(round_up(self.se_cnt * self.xccs * 4, 0x1000), BufferSpec(cpu_access=True, nolru=True))
      self.sqtt_next_cmd_id = itertools.count(0)

  def create_queue(self, queue_type, ring_size, ctx_save_restore_size=0, eop_buffer_size=0, ctl_stack_size=0, debug_memory_size=0, idx=0):
    ring = Buffer(self.device, ring_size // 4, dtypes.uint32, options=BufferSpec(uncached=True, cpu_access=True), preallocate=True)
    gart = Buffer(self.device, 0x100, dtypes.uint8, options=BufferSpec(uncached=True, cpu_access=True), preallocate=True)

    if queue_type == kfd.KFD_IOC_QUEUE_TYPE_COMPUTE_AQL:
      self.aql_gart = gart
      self.aql_desc = hsa.amd_queue_t(queue_properties=hsa.AMD_QUEUE_PROPERTIES_IS_PTR64 | hsa.AMD_QUEUE_PROPERTIES_ENABLE_PROFILING,
        read_dispatch_id_field_base_byte_offset=getattr(hsa.amd_queue_t, 'read_dispatch_id').offset,
        max_cu_id=(self.cu_cnt * self.xccs) - 1, max_wave_id=self.waves_per_cu - 1)
      self.aql_gart._buf.cpu_view().view(fmt='B')[:ctypes.sizeof(self.aql_desc)] = bytes(self.aql_desc)

    cwsr_buffer_size = round_up((ctx_save_restore_size + debug_memory_size) * self.xccs, mmap.PAGESIZE)
    cwsr_buffer = Buffer(self.device, cwsr_buffer_size, dtypes.uint8, preallocate=True) if ctx_save_restore_size else None
    eop_buffer = Buffer(self.device, eop_buffer_size, dtypes.uint8, preallocate=True) if eop_buffer_size else None

    queue = (self.iface.create_queue(queue_type, ring, gart, rptr=getattr(hsa.amd_queue_t, 'read_dispatch_id').offset,
             wptr=getattr(hsa.amd_queue_t, 'write_dispatch_id').offset, eop_buffer=eop_buffer, cwsr_buffer=cwsr_buffer,
             ctx_save_restore_size=ctx_save_restore_size, ctl_stack_size=ctl_stack_size, idx=idx))

    qname = f"{'COPY' if queue_type == kfd.KFD_IOC_QUEUE_TYPE_SDMA else 'COMPUTE'}:{idx}"
    self.pm_bufferize = PatternMatcher([
      (UPat(Ops.PARAM, tag={(qname, name)}), lambda ctx, b=getattr(queue, name): b) for name in ["ring", "write_ptr", "doorbell", "put_value"]
    ] + [
      (UPat(Ops.PARAM, tag={(qname, "timeline_signal")}), lambda ctx, q=qname: ctx[0].timeline_signal(q)),
      (UPat(Ops.PARAM, tag={(qname, "timeline_value")}), lambda ctx, q=qname: ctx[0].timeline_value(q)),
    ]) + self.pm_bufferize

    return queue

  @functools.cached_property
  def compute_queue(self) -> AMDQueueDesc:
    # https://gitlab.freedesktop.org/agd5f/linux/-/blob/a1fc9f584c4aaf8bc1ebfa459fc57a3f26a290d8/drivers/gpu/drm/amd/amdkfd/kfd_queue.c#L391
    sgrp_size_per_cu, hwreg_size_per_cu = 0x4000, 0x1000
    lds_size_per_cu = self.iface.props["lds_size_in_kb"] << 10 if self.target[:2] == (9,5) else 0x10000
    vgpr_size_per_cu = 0x60000 if self.target in {(11,0,0), (11,0,1), (11,5,1), (12,0,0), (12,0,1)} else 0x80000 if self.target[0] == 9 else 0x40000
    wg_data_size = round_up((vgpr_size_per_cu + sgrp_size_per_cu + lds_size_per_cu + hwreg_size_per_cu) * self.cu_cnt, mmap.PAGESIZE)
    ctl_stack_size = round_up((12 if self.target[0] != 9 else 8) * self.wave_cnt + 8 + 40, mmap.PAGESIZE)
    return self.create_queue(kfd.KFD_IOC_QUEUE_TYPE_COMPUTE_AQL if self.is_aql else kfd.KFD_IOC_QUEUE_TYPE_COMPUTE,
      0x2000 if self.is_usb() else (16 << 20), eop_buffer_size=0x1000,
      ctx_save_restore_size=0 if self.is_am() else wg_data_size + ctl_stack_size, ctl_stack_size=ctl_stack_size,
      debug_memory_size=round_up(self.wave_cnt * 32, 64))

  def sdma_queue(self, idx:int):
    if getenv("AMD_DISABLE_SDMA"): return None
    if idx in self.sdma_queues: return self.sdma_queues[idx]
    with contextlib.suppress(OSError):
      self.sdma_queues[idx] = self.create_queue(kfd.KFD_IOC_QUEUE_TYPE_SDMA, 0x200 if self.is_usb() else (16 << 20), idx=idx)
    return self.sdma_queues.get(idx, None)

  def tmpring_size(self, private_segment_size):
    private_segment_size = max(private_segment_size, 128)

    lanes_per_wave = 64 # wave64
    mem_alignment_size = 256 if self.target[0] != 9 else 1024
    size_per_thread = round_up(private_segment_size, mem_alignment_size // lanes_per_wave)
    size_per_xcc = size_per_thread * lanes_per_wave * self.iface.props['max_slots_scratch_cu'] * self.cu_cnt

    # NOTE: xcc logic is correct only for GFX9.
    max_scratch_waves = self.cu_cnt * self.iface.props['max_slots_scratch_cu'] * self.xccs
    wave_scratch = ceildiv(lanes_per_wave * size_per_thread, mem_alignment_size)
    num_waves = (size_per_xcc // (wave_scratch * mem_alignment_size)) // (self.se_cnt if self.target[0] != 9 else 1)

    tmpring_t = getattr(hsa, f'union_COMPUTE_TMPRING_SIZE{"_GFX"+str(self.target[0]) if self.target[0] != 9 else ""}_bitfields')
    tmpring = int.from_bytes(tmpring_t(WAVES=min(num_waves, max_scratch_waves), WAVESIZE=wave_scratch), 'little')

    if hasattr(self, 'aql_desc'):
      gfx9_rsrc = {'NUM_FORMAT':hsa.BUF_NUM_FORMAT_UINT, 'DATA_FORMAT':hsa.BUF_DATA_FORMAT_32, 'ELEMENT_SIZE':1, 'INDEX_STRIDE':3}
      rsrc = {'DST_SEL_X':hsa.SQ_SEL_X, 'DST_SEL_Y':hsa.SQ_SEL_Y, 'DST_SEL_Z':hsa.SQ_SEL_Z, 'DST_SEL_W':hsa.SQ_SEL_W, 'ADD_TID_ENABLE':1,
              'TYPE':hsa.SQ_RSRC_BUF, **(gfx9_rsrc if self.target[0] == 9 else {'FORMAT':hsa.BUF_FORMAT_32_UINT, 'OOB_SELECT':2})}
      rsrc1_t = getattr(hsa, f'union_SQ_BUF_RSRC_WORD1{"_GFX11" if self.target[0] != 9 else ""}_bitfields')
      rsrc3_t = getattr(hsa, f'union_SQ_BUF_RSRC_WORD3{"_GFX"+str(self.target[0]) if self.target[0] != 9 else ""}_bitfields')

      self.aql_desc.scratch_backing_memory_location = int(self.scratch.get_buf().va_addr)
      self.aql_desc.scratch_wave64_lane_byte_size = self.max_private_segment_size * lanes_per_wave // 64
      self.aql_desc.scratch_resource_descriptor[:] = [lo32(self.scratch.get_buf().va_addr),
        int.from_bytes(rsrc1_t(BASE_ADDRESS_HI=hi32(self.scratch.get_buf().va_addr), SWIZZLE_ENABLE=1), 'little'),
        lo32(size_per_xcc), int.from_bytes(bytes(rsrc3_t(**rsrc)), 'little')]
      self.aql_desc.compute_tmpring_size = tmpring
      self.aql_gart._buf.cpu_view()[:ctypes.sizeof(self.aql_desc)] = bytes(self.aql_desc)

    return tmpring

  def scratch_buffer(self, private_segment_size):
    private_segment_size = max(private_segment_size, 128)
    if self.max_private_segment_size < private_segment_size:
      lanes_per_wave = 64 # wave64
      mem_alignment_size = 256 if self.target[0] != 9 else 1024
      size_per_thread = round_up(private_segment_size, mem_alignment_size // lanes_per_wave)
      size_per_xcc = size_per_thread * lanes_per_wave * self.iface.props['max_slots_scratch_cu'] * self.cu_cnt
      self.scratch = Buffer(self.device, size_per_xcc * self.xccs, dtypes.uint8, options=BufferSpec(nolru=True), preallocate=True)
      self.max_private_segment_size = private_segment_size
    return self.scratch

  def on_device_hang(self): self.iface.on_device_hang()

  def device_props(self): return self.iface.props
