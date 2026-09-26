from __future__ import annotations
from typing import cast, Any
import os, ctypes, struct, functools, importlib, mmap, errno, contextlib, sys, hashlib, itertools, collections, atexit
assert sys.platform != 'win32'
from dataclasses import dataclass, replace
from tinygrad.runtime.support.hcq2 import HWQueue, encode_submit, bufferize_linear, to_name, patch, unwrap_view, rt_addr, layout_args
from tinygrad.runtime.support.hcq2 import pack_args
from tinygrad.uop.ops import sint, UOp, ProgramInfo
from tinygrad.device import BufferStorage, BufferSpec, Buffer, Device, Allocator, Compiled, ProfileProgramEvent
from tinygrad.dtype import dtypes
from tinygrad.helpers import getenv, round_up, data64_le, DEBUG, PROFILE, ProfileEvent, lo32, hi32, prod, colored
from tinygrad.helpers import ceildiv, unwrap, pluralize, HCQ2, ContextVar, VIZ
from tinygrad.renderer.cstyle import HIPRenderer, HIPCCRenderer
from tinygrad.renderer.llvmir import AMDLLVMRenderer
from tinygrad.runtime.autogen import kfd, hsa, sqtt, amdgpu_kd, amdgpu_drm
from tinygrad.runtime.autogen.am import am
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.runtime.support.hcq import FileIOInterface, MMIOInterface, hcq_filter_visible_devices
from tinygrad.runtime.support.am.amdev import AMDev, AMMemoryManager
from tinygrad.runtime.support.amd import AMDReg, AMDIP, import_module, import_soc, import_pmc
from tinygrad.runtime.support.system import PCIIfaceBase, USBPCIDevice, MAP_FIXED, MAP_NORESERVE
from tinygrad.runtime.support.usb import USB3, pm_usb_batch, pm_usb_lower, pm_usb_bufferize
from tinygrad.runtime.support.memory import AddrSpace
if getenv("IOCTL"): import extra.hip_gpu_driver.hip_ioctl  # noqa: F401 # pylint: disable=unused-import

from tinygrad.engine.realize import get_call_arg_uops, get_call_var_uops
from tinygrad.uop.ops import Ops, UPat, PatternMatcher

SQTT = ContextVar("SQTT", abs(VIZ.value)>=2)
SQTT_ITRACE_SE_MASK, SQTT_LIMIT_SE, SQTT_SIMD_SEL, SQTT_TOKEN_EXCLUDE = \
  ContextVar("SQTT_ITRACE_SE_MASK", 0b11), ContextVar("SQTT_LIMIT_SE", 0), ContextVar("SQTT_SIMD_SEL", 0), ContextVar("SQTT_TOKEN_EXCLUDE", 0)
PMC = ContextVar("PMC", abs(VIZ.value)>=2)

EVENT_INDEX_PARTIAL_FLUSH = 4 # based on a comment in nvd.h
WAIT_REG_MEM_FUNCTION_EQ  = 3 # ==
WAIT_REG_MEM_FUNCTION_NEQ = 4 # !=
WAIT_REG_MEM_FUNCTION_GEQ = 5 # >=
AQL_HDR = (1 << hsa.HSA_PACKET_HEADER_BARRIER) | (hsa.HSA_FENCE_SCOPE_SYSTEM << hsa.HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) \
        | (hsa.HSA_FENCE_SCOPE_SYSTEM << hsa.HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE)

@dataclass(frozen=True)
class ProfileSQTTEvent(ProfileEvent): device:str; kern:int; se:int; blob:bytes; itrace:bool; exec_tag:int # noqa: E702

@dataclass(frozen=True)
class PMCSample: name:str; block:str; xcc:int; inst:int; se:int; sa:int; wgp:int; off:int; size:int; regsample:str # noqa: E702

@dataclass(frozen=True)
class ProfilePMCEvent(ProfileEvent): device:str; kern:int; sched:list[PMCSample]; blob:bytes; exec_tag:int # noqa: E702

# *****************
# PM4

def _queue_args(hq:HWQueue, q) -> list[UOp]: # the ring and its pointers, tagged {name}_{queue} like the device's bufferize rules
  shapes = [("ring", (q.ring.size,), q.ring.dtype)] + [(n, (1,), dtypes.uint64) for n in ("write_ptr", "doorbell", "put_value")]
  return [UOp.placeholder(s, d, 0, device=hq.devs, volatile=True, tag=to_name(n, hq.queue)) for n, s, d in shapes]

def _dw(vals) -> int: return sum(2 if isinstance(x, UOp) and x.dtype.itemsize == 8 else 1 for x in vals)

def dispatch_packet(data:AMDProgramData, info:ProgramInfo, kernel_object:UOp=UOp.const(0, dtypes.uint64),
                    kernarg_address:UOp=UOp.const(0, dtypes.uint64)) -> list: # as words: the grid may be symbolic
  pkt = bytes(hsa.hsa_kernel_dispatch_packet_t(header=AQL_HDR | (hsa.HSA_PACKET_TYPE_KERNEL_DISPATCH << hsa.HSA_PACKET_HEADER_TYPE),
    setup=3 << hsa.HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS, private_segment_size=data.private_segment_size,
    group_segment_size=data.group_segment_size, **{f"workgroup_size_{d}": l for d, l in zip("xyz", info.local_size)}))
  grid = [(g * l).cast(dtypes.uint32) if isinstance(g, UOp) else g * l for g, l in zip(cast(tuple[Any, ...], info.global_size), info.local_size)]
  return [UOp(Ops.BINARY, arg=pkt[:12]), *grid, UOp(Ops.BINARY, arg=pkt[24:32]), kernel_object, kernarg_address, UOp(Ops.BINARY, arg=pkt[48:])]

class AMDComputeQueue(HWQueue):
  dev:AMDDevice

  def __init__(self, ctx, submit):
    super().__init__(ctx, submit)
    self.pm4, self.gc, self.soc, self.nbio, self.target = self.dev.pm4, self.dev.gc, self.dev.soc, self.dev.nbio, self.dev.target
    self.profiled:list[UOp] = []
    if self.dev.pmc_enabled: self.pmc_start()

  def pkt3(self, cmd, *vals): self.q(self.pm4.PACKET3(cmd, _dw(vals) - 1), *vals)

  def wreg(self, reg:AMDReg, *args:sint, **kwargs:int):
    if bool(args) == bool(kwargs): raise RuntimeError('One (and only one) of *args or **kwargs must be specified')
    if self.pm4.PACKET3_SET_SH_REG_START <= reg.addr[0] < self.pm4.PACKET3_SET_SH_REG_END:
      set_packet, set_packet_start = self.pm4.PACKET3_SET_SH_REG, self.pm4.PACKET3_SET_SH_REG_START
    elif self.pm4.PACKET3_SET_UCONFIG_REG_START <= reg.addr[0] < self.pm4.PACKET3_SET_UCONFIG_REG_START + 2**16-1:
      set_packet, set_packet_start = self.pm4.PACKET3_SET_UCONFIG_REG, self.pm4.PACKET3_SET_UCONFIG_REG_START
    else: raise RuntimeError(f'Cannot set {reg.name} ({reg.addr[0]}) via pm4 packet')
    self.pkt3(set_packet, reg.addr[0] - set_packet_start, *(args or (reg.encode(**kwargs),)))

  @contextlib.contextmanager
  def pred_exec(self, xcc_mask:int): # the count fills in when the block closes
    if self.dev.xccs > 1: self.pkt3(self.pm4.PACKET3_PRED_EXEC, xcc_mask << 24)
    start = len(self.blob)
    yield
    if self.dev.xccs > 1:
      cnt, = struct.unpack("I", self.blob[start-4:start])
      self.blob[start-4:start] = struct.pack("I", cnt | (len(self.blob) - start) // 4)

  def set_grbm(self, instance=None, se=None, sh=None, wgp=None):
    instance_val = (wgp << 2 | (instance or 0)) if wgp is not None else instance
    self.wreg(self.gc.regGRBM_GFX_INDEX, **{(f'{key}_broadcast_writes' if val is None else f'{key}_index'): (1 if val is None else val)
      for key, val in [('instance', instance_val), ('se', se), ('sh' if self.target[0] == 9 else 'sa', sh)]})

  def wait_reg_mem(self, value, mask=0xffffffff, mem=None, reg=None, reg_done=0, op=WAIT_REG_MEM_FUNCTION_GEQ):
    wrm_info_dw = self.pm4.WAIT_REG_MEM_MEM_SPACE(int(mem is not None)) | self.pm4.WAIT_REG_MEM_OPERATION(int(mem is None and reg_done > 0)) \
                | self.pm4.WAIT_REG_MEM_FUNCTION(op) | self.pm4.WAIT_REG_MEM_ENGINE(0)
    self.pkt3(self.pm4.PACKET3_WAIT_REG_MEM, wrm_info_dw, *((mem,) if mem is not None else (reg, reg_done)), value, mask, 4)

  def acquire_mem(self, addr=0x0, sz=(1 << 64)-1, gli=1, glm=1, glk=1, glv=1, gl1=1, gl2=1):
    if self.target[0] != 9:
      cache_flags_dw = self.pm4.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLI_INV(gli) \
                     | self.pm4.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLM_INV(glm) | self.pm4.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLM_WB(glm) \
                     | self.pm4.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLK_INV(glk) | self.pm4.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLK_WB(glk) \
                     | self.pm4.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLV_INV(glv) | self.pm4.PACKET3_ACQUIRE_MEM_GCR_CNTL_GL1_INV(gl1) \
                     | self.pm4.PACKET3_ACQUIRE_MEM_GCR_CNTL_GL2_INV(gl2) | self.pm4.PACKET3_ACQUIRE_MEM_GCR_CNTL_GL2_WB(gl2)
      return self.pkt3(self.pm4.PACKET3_ACQUIRE_MEM, 0, *data64_le(sz), *data64_le(addr), 0, cache_flags_dw)
    cp_coher_cntl = self.pm4.PACKET3_ACQUIRE_MEM_CP_COHER_CNTL_SH_ICACHE_ACTION_ENA(gli) | \
                    self.pm4.PACKET3_ACQUIRE_MEM_CP_COHER_CNTL_SH_KCACHE_ACTION_ENA(glk) | \
                    self.pm4.PACKET3_ACQUIRE_MEM_CP_COHER_CNTL_TC_ACTION_ENA(gl2) | \
                    self.pm4.PACKET3_ACQUIRE_MEM_CP_COHER_CNTL_TCL1_ACTION_ENA(gl1) | \
                    self.pm4.PACKET3_ACQUIRE_MEM_CP_COHER_CNTL_TC_WB_ACTION_ENA(gl2)
    return self.pkt3(self.pm4.PACKET3_ACQUIRE_MEM, cp_coher_cntl, *data64_le(sz), *data64_le(addr), 0x0000000A)

  def release_mem(self, address=0x0, value=0, data_sel=0, int_sel=2, ctxid=0, cache_flush=False):
    if self.target[0] != 9:
      cache_flags_dw = 0 if not cache_flush else (self.pm4.PACKET3_RELEASE_MEM_GCR_GLV_INV | self.pm4.PACKET3_RELEASE_MEM_GCR_GL1_INV \
                     | self.pm4.PACKET3_RELEASE_MEM_GCR_GL2_INV | self.pm4.PACKET3_RELEASE_MEM_GCR_GLM_WB \
                     | self.pm4.PACKET3_RELEASE_MEM_GCR_GLM_INV | self.pm4.PACKET3_RELEASE_MEM_GCR_GL2_WB | self.pm4.PACKET3_RELEASE_MEM_GCR_SEQ)
      event_dw = self.pm4.PACKET3_RELEASE_MEM_EVENT_TYPE(self.pm4.CACHE_FLUSH_AND_INV_TS_EVENT) \
               | self.pm4.PACKET3_RELEASE_MEM_EVENT_INDEX(self.pm4.event_index__mec_release_mem__end_of_pipe)
      memsel_dw = self.pm4.PACKET3_RELEASE_MEM_DATA_SEL(data_sel) | self.pm4.PACKET3_RELEASE_MEM_INT_SEL(int_sel) \
                | self.pm4.PACKET3_RELEASE_MEM_DST_SEL(0)
    else:
      cache_flags_dw = 0 if not cache_flush else (self.pm4.EOP_TC_WB_ACTION_EN | self.pm4.EOP_TC_NC_ACTION_EN)
      event_dw = self.pm4.EVENT_TYPE(self.pm4.CACHE_FLUSH_AND_INV_TS_EVENT) | \
                 self.pm4.EVENT_INDEX(self.pm4.event_index__mec_release_mem__end_of_pipe)
      memsel_dw = self.pm4.DATA_SEL(data_sel) | self.pm4.INT_SEL(int_sel)
      ctxid = 0
    addr_w = address if isinstance(address, UOp) else UOp.const(address, dtypes.uint64)
    val_w = value.cast(dtypes.uint64) if isinstance(value, UOp) else UOp.const(value, dtypes.uint64)
    self.pkt3(self.pm4.PACKET3_RELEASE_MEM, event_dw | cache_flags_dw, memsel_dw, addr_w, val_w, ctxid)

  def memory_barrier(self):
    pf = '' if self.nbio.version[0] == 2 else '0' if self.nbio.version[:2] != (7, 11) else '1'
    self.wait_reg_mem(reg=getattr(self.nbio, f'regBIF_BX_PF{pf}_GPU_HDP_FLUSH_REQ').addr[0],
                      reg_done=getattr(self.nbio, f'regBIF_BX_PF{pf}_GPU_HDP_FLUSH_DONE').addr[0], value=0xffffffff)
    self.acquire_mem()

  def spi_config(self, tracing:bool):
    self.wreg(self.gc.regSPI_CONFIG_CNTL, ps_pkr_priority_cntl=3, exp_priority_order=3, gpr_write_priority=0x2c688,
              enable_sqg_bop_events=int(tracing), enable_sqg_top_events=int(tracing))

  ### profiling: a kernel's slot holds its counters and trace until a synchronize reads them back

  def prof_buf(self, name:str) -> UOp:
    return UOp.placeholder((getattr(self.dev, name).size,), getattr(self.dev, name).dtype, 0, device=self.devs, tag=name)

  def prof_start(self, data:AMDProgramData, info:ProgramInfo, lib:UOp) -> UOp|None:
    if not (self.dev.pmc_enabled or self.dev.sqtt_enabled): return None
    slot = (self.prof_buf("prof_log").index(0).load() + len(self.profiled)) % self.dev.prof_slots
    tag = UOp.const(unwrap_view(lib)[0].arg.slot, dtypes.uint64)
    self.profiled.append(self.prof_buf("prof_log").index(1 + slot.cast(dtypes.int)).store(tag))
    if self.dev.sqtt_enabled:
      self.sqtt_start(slot)
      self.sqtt_setup_exec(data, info)
    return slot

  def prof_stop(self, slot:UOp|None):
    if slot is None: return
    if self.dev.pmc_enabled: self.pmc_read(slot)
    if self.dev.sqtt_enabled: self.sqtt_stop(slot)

  def prof_bump(self, cmdbuf:UOp) -> UOp:
    if not self.profiled: return cmdbuf
    log = self.prof_buf("prof_log")
    return cmdbuf.after(log.after(cmdbuf, *self.profiled).index(0).store(log.index(0).load() + len(self.profiled)))

  ### PMC

  def pmc_reset_counters(self, en=True):
    self.set_grbm()
    self.wreg(self.gc.regCP_PERFMON_CNTL if self.target[0] <= 11 else self.gc.regCP_PERFMON_CNTL_1, perfmon_state=0)
    if en: self.wreg(self.gc.regCP_PERFMON_CNTL if self.target[0] <= 11 else self.gc.regCP_PERFMON_CNTL_1, perfmon_state=1)

  def pmc_start(self): # every submit
    self.pmc_reset_counters(en=False)
    self.wreg(self.gc.regSQ_PERFCOUNTER_CTRL, cs_en=1, ps_en=1, gs_en=1, hs_en=1, **({'vmid_mask':0xffff} if (gfx9:=self.target[0] == 9) else {}))
    if not gfx9: self.wreg(self.gc.regSQ_PERFCOUNTER_CTRL2, force_en=1, vmid_en=0xffff)

    end_off, sched = 0, []
    block2pid:dict[str, itertools.count] = collections.defaultdict(lambda: itertools.count())
    for name in self.dev.pmc_names:
      block, idx = self.dev.pmc_counters[name]
      # sq block on gfx11+ goes down to wgps
      inst_cnt, se_cnt, sa_cnt, wgp_cnt = {"GRBM": (1, 1, 1, 1), "GL2C": (32, 1, 1, 1), "TCC": (16, 1, 1, 1),
        "SQ": (1, self.dev.se_cnt) + ((1, 1) if gfx9 else (2, self.dev.iface.props['cu_per_simd_array'] // 2))}[block]
      end_off += (rec_size:=prod((self.dev.xccs, inst_cnt, se_cnt, sa_cnt, wgp_cnt)) * 8)

      # gfx11+ and later require even-numbered SQ *_SELECT registers
      regsample = f'reg{block}_PERFCOUNTER{(pcid:=next(block2pid[block]))}'
      if (regsel:=getattr(self.gc, (f'reg{block}_PERFCOUNTER{(pcid*2) if not gfx9 and block=="SQ" else pcid}_SELECT'), None)) is None:
        raise RuntimeError(f'{block} is out of perfcounter registers: ({regsample} is not found)')

      self.wreg(regsel, perf_sel=idx, **({'simd_mask':0xf, 'sqc_bank_mask':0xf, 'sqc_client_mask':0xf} if gfx9 and block == "SQ" else {}))
      sched.append(PMCSample(name, block, self.dev.xccs, inst_cnt, se_cnt, sa_cnt, wgp_cnt, end_off-rec_size, rec_size, regsample))
    self.dev.pmc_sched = sched

    if gfx9: self.wreg(self.gc.regSQ_PERFCOUNTER_MASK, sh0_mask=0xffff, sh1_mask=0xffff)
    self.wreg(self.gc.regCOMPUTE_PERFCOUNT_ENABLE, 1)
    self.pmc_reset_counters(en=True)

  def pmc_read(self, slot:UOp):
    buf = rt_addr(self.prof_buf("pmc_buf"), self.devs) + slot * self.dev.pmc_size
    self.set_grbm()
    self.wreg(self.gc.regCP_PERFMON_CNTL if self.target[0] <= 11 else self.gc.regCP_PERFMON_CNTL_1, perfmon_state=1, perfmon_sample_enable=1)

    for smp in self.dev.pmc_sched:
      offset = itertools.count(smp.off, step=8)

      for xcc in range(smp.xcc):
        with self.pred_exec(xcc_mask=1 << xcc):
          for inst, se_idx, sa_idx, wgp_idx in itertools.product(range(smp.inst), range(smp.se), range(smp.sa), range(smp.wgp)):
            loff = next(offset)
            if smp.wgp > 1 and not self.dev.iface.is_wgp_active(xcc, se_idx, sa_idx, wgp_idx): continue
            self.set_grbm(**({'instance':inst} if smp.inst > 1 else ({'se':se_idx}|({'sh':sa_idx, 'wgp':wgp_idx} if self.target[0] != 9 else {}))))

            # Copy counter to memory (src_sel = perf, dst_sel = tc_l2)
            lo, hi = getattr(self.gc, f'{smp.regsample}_LO'), getattr(self.gc, f'{smp.regsample}_HI', None)
            self.pkt3(self.pm4.PACKET3_COPY_DATA, (2 << 8) | 4, lo.addr[0], 0, buf + loff)
            if hi is not None: self.pkt3(self.pm4.PACKET3_COPY_DATA, (2 << 8) | 4, hi.addr[0], 0, buf + (loff + 4))

    self.pmc_reset_counters(en=True)

  ### SQTT

  def sqtt_userdata(self, data, *extra_dwords):
    data_ints = [x[0] for x in struct.iter_unpack('<I', bytes(data))] + list(extra_dwords)
    for i in range(0, len(data_ints), 2):
      self.wreg(self.gc.regSQ_THREAD_TRACE_USERDATA_2, *data_ints[i:i+2])

  def sqtt_config(self, tracing:bool):
    trace_ctrl = {'rt_freq': self.soc.SQ_TT_RT_FREQ_4096_CLK} if self.target < (12,0,0) else {}
    self.wreg(self.gc.regSQ_THREAD_TRACE_CTRL, draw_event_en=1, spi_stall_en=1, sq_stall_en=1, reg_at_hwm=2, hiwater=1, util_timer=1,
      mode=int(tracing), **trace_ctrl)

  def sqtt_setup_exec(self, data:AMDProgramData, info:ProgramInfo):
    self.sqtt_userdata(sqtt.struct_rgp_sqtt_marker_pipeline_bind(identifier=sqtt.RGP_SQTT_MARKER_IDENTIFIER_BIND_PIPELINE,
                                                                 bind_point=(__BIND_POINT_COMPUTE:=1), api_pso_hash=data64_le(data.libhash)))
    self.sqtt_userdata(sqtt.struct_rgp_sqtt_marker_event(has_thread_dims=1, cmd_id=next(self.dev.sqtt_next_cmd_id)), *info.global_size)

    if SQTT_LIMIT_SE:
      # Calculate number of CUs per SE to enable based on blocks count. 4 is maximum simd per CU, but on rdna we can trace only 1.
      cu_per_se = prod([x if isinstance(x, int) else 1 for x in info.global_size]) // ((self.dev.cu_cnt // self.dev.se_cnt) * 4)
      for xcc in range(self.dev.xccs):
        with self.pred_exec(xcc_mask=1 << xcc):
          for i in range(8 if self.target[0] != 9 else 4):
            if SQTT_LIMIT_SE > 1: mask = 1 if SQTT_ITRACE_SE_MASK.value & (1 << i) else 0 # only run unmasked shader engines
            else:
              sa_mask = (1 << (self.dev.iface.props['cu_per_simd_array'] // 2)) - 1
              cu_mask = (1 << (cu_per_se + (1 if i == 0 else 0))) - 1
              mask = lo32((cu_mask & sa_mask) | (cu_mask & (sa_mask << 16)) << 16)
            self.wreg(getattr(self.gc, f'regCOMPUTE_STATIC_THREAD_MGMT_SE{i}'), mask)

  def sqtt_start(self, slot:UOp):
    self.memory_barrier()
    win, ses = self.dev.sqtt_win, self.dev.sqtt_ses
    base = rt_addr(self.prof_buf("sqtt_buf"), self.devs) + slot * win
    if self.target[0] == 9:
      self.set_grbm()
      self.wreg(self.gc.regSQ_THREAD_TRACE_MASK, simd_en=0xf, cu_sel=0, sq_stall_en=1, spi_stall_en=1, reg_stall_en=1, vm_id_mask=0)
      for se in range(ses):
        mask = (__SQTT_MISC:=1<<0) | (__SQTT_TIME:=1<<1) | (__SQTT_REG:=1<<2) | (__SQTT_WAVE_START:=1<<3) | (__SQTT_WAVE_END:=1<<6) \
             | (__SQTT_USERDATA:=1<<12) | (__SQTT_REG_CS:=1<<5) | (__SQTT_REG_CS_PRIV:=1<<15)
        if (SQTT_ITRACE_SE_MASK.value >> se) & 0b1: mask |= (__SQTTINST:=1<<10) | (__SQTT_INST_PC:=1<<11) | (__SQTT_ISSUE:=1<<13)

        buf0_lo, buf0_hi = [((base + se * self.dev.prof_slots * win) >> sh).cast(dtypes.uint32) for sh in (12, 44)]
        with self.pred_exec(xcc_mask=1<<(se // self.dev.se_cnt)):
          self.set_grbm(se=se % self.dev.se_cnt, sh=0)
          self.wreg(self.gc.regSQ_THREAD_TRACE_TOKEN_MASK, reg_mask=0xf, token_mask=mask)
          self.wreg(self.gc.regSQ_THREAD_TRACE_TOKEN_MASK2, inst_mask=0xffffffff)
          self.wreg(self.gc.regSQ_THREAD_TRACE_BASE, buf0_lo)
          self.wreg(self.gc.regSQ_THREAD_TRACE_BASE2, buf0_hi)
          self.wreg(self.gc.regSQ_THREAD_TRACE_SIZE, size=win >> 12)
          self.wreg(self.gc.regSQ_THREAD_TRACE_CTRL, reset_buffer=1)
          self.wreg(self.gc.regSQ_THREAD_TRACE_MODE, mask_cs=1, autoflush_en=1, mode=1)
    else:
      self.spi_config(tracing=True)
      # One buffer for one SE, mesa does it with a single buffer and ac_sqtt_get_data_offset, but this is simpler and should work just as well
      for se in range(ses):
        self.set_grbm(se=se, sh=0)

        buf0_lo, buf0_hi = [((base + se * self.dev.prof_slots * win) >> sh).cast(dtypes.uint32) for sh in (12, 44)]
        if self.target >= (12,0,0):
          self.wreg(self.gc.regSQ_THREAD_TRACE_BUF0_SIZE, size=win >> 12)
          self.wreg(self.gc.regSQ_THREAD_TRACE_BUF0_BASE_LO, buf0_lo)
          self.wreg(self.gc.regSQ_THREAD_TRACE_BUF0_BASE_HI, buf0_hi)
        else:
          self.wreg(self.gc.regSQ_THREAD_TRACE_BUF0_SIZE, self.gc.regSQ_THREAD_TRACE_BUF0_SIZE.encode(size=win >> 12) | buf0_hi)
          self.wreg(self.gc.regSQ_THREAD_TRACE_BUF0_BASE, buf0_lo)
        # NOTE: SQTT can only trace instructions on one simd per se, this selects the simd in first wgp in first sa.
        # For RGP to display instruction trace it has to see it on first SE. Howerver ACE/MEC/whatever does the dispatching starting with second se,
        # and on amdgpu/non-AM it also does weird things with dispatch order inside se: around 7 times out of 10 it starts from the last cu, but
        # sometimes not, especially if the kernel has more than one wavefront which means that kernels with small global size might get unlucky and
        # be dispatched on something else and not be seen in instruction tracing tab. You can force the wavefronts of a kernel to be dispatched on the
        # CUs you want to by disabling other CUs via bits in regCOMPUTE_STATIC_THREAD_MGMT_SE<x> and trace even kernels that only have one wavefront.
        # Use SQTT_SIMD_SEL to select which SIMD to trace (0-3). Memory ops show different InstOp values (0x2x vs 0x5x) based on SIMD.
        cs_wtype = (1 << 6) if self.target >= (12,0,0) else self.soc.SQ_TT_WTYPE_INCLUDE_CS_BIT
        self.wreg(self.gc.regSQ_THREAD_TRACE_MASK, wtype_include=cs_wtype, simd_sel=SQTT_SIMD_SEL.value, wgp_sel=0, sa_sel=0)
        reg_include = self.soc.SQ_TT_TOKEN_MASK_SQDEC_BIT | self.soc.SQ_TT_TOKEN_MASK_SHDEC_BIT | self.soc.SQ_TT_TOKEN_MASK_GFXUDEC_BIT | \
                      self.soc.SQ_TT_TOKEN_MASK_COMP_BIT | self.soc.SQ_TT_TOKEN_MASK_CONTEXT_BIT
        token_exclude = SQTT_TOKEN_EXCLUDE.value | ((1 << self.soc.SQ_TT_TOKEN_EXCLUDE_PERF_SHIFT) if self.target < (12,0,0) else 0)

        # disable instr tracing
        if not (SQTT_ITRACE_SE_MASK.value >> se) & 0b1:
          # gfx12 doesn't have enums with all fields, so it's hardcoded, but it's the same as gfx11.
          token_exclude |= (1 << self.soc.SQ_TT_TOKEN_EXCLUDE_VMEMEXEC_SHIFT | 1 << self.soc.SQ_TT_TOKEN_EXCLUDE_ALUEXEC_SHIFT | \
                            1 << self.soc.SQ_TT_TOKEN_EXCLUDE_VALUINST_SHIFT | 1 << self.soc.SQ_TT_TOKEN_EXCLUDE_IMMEDIATE_SHIFT | \
                            1 << self.soc.SQ_TT_TOKEN_EXCLUDE_INST_SHIFT) if self.target < (12,0,0) else 0x927

        self.wreg(self.gc.regSQ_THREAD_TRACE_TOKEN_MASK, reg_include=reg_include, token_exclude=token_exclude, bop_events_token_include=1,
                  **({} if self.target < (12,0,0) else {'exclude_barrier_wait': 1}))
        self.sqtt_config(tracing=True)

    self.set_grbm()
    if self.target[0] != 9: self.wreg(self.gc.regCOMPUTE_THREAD_TRACE_ENABLE, 1)
    self.memory_barrier()

  # Magic values from src/amd/common/ac_sqtt.c:ac_sqtt_emit_stop and src/amd/common/ac_sqtt.c:ac_sqtt_emit_wait
  def sqtt_stop(self, slot:UOp):
    self.memory_barrier()
    self.set_grbm()
    ses = self.dev.sqtt_ses
    wptrs = rt_addr(self.prof_buf("sqtt_wptrs"), self.devs) + slot * (ses * 4)

    # Start shutting everything down
    if self.target[0] == 9: self.wreg(self.gc.regSQ_THREAD_TRACE_MODE, mask_cs=1, autoflush_en=1, mode=0)
    else:
      self.wreg(self.gc.regCOMPUTE_THREAD_TRACE_ENABLE, 0)
      self.pkt3(self.pm4.PACKET3_EVENT_WRITE, self.pm4.EVENT_TYPE(self.soc.THREAD_TRACE_FINISH) | self.pm4.EVENT_INDEX(0))

    # For each SE wait for finish to complete and copy regSQ_THREAD_TRACE_WPTR to know where in the buffer trace data ends
    for se in range(ses):
      with self.pred_exec(xcc_mask=1<<(se // self.dev.se_cnt)):
        self.set_grbm(se=se % self.dev.se_cnt, sh=0)

        regstatus = self.gc.regSQ_THREAD_TRACE_STATUS.addr[0] - (self.pm4.PACKET3_SET_UCONFIG_REG_START if self.target[0] == 9 else 0)
        if self.target[0] != 9:
          self.wait_reg_mem(reg=regstatus, mask=self.gc.regSQ_THREAD_TRACE_STATUS.fields_mask('finish_pending'), op=WAIT_REG_MEM_FUNCTION_EQ, value=0)
          self.sqtt_config(tracing=False)
        self.wait_reg_mem(reg=regstatus, mask=self.gc.regSQ_THREAD_TRACE_STATUS.fields_mask('busy'), op=WAIT_REG_MEM_FUNCTION_EQ, value=0)
        self.pkt3(self.pm4.PACKET3_EVENT_WRITE, self.pm4.EVENT_TYPE(self.soc.CS_PARTIAL_FLUSH) | self.pm4.EVENT_INDEX(EVENT_INDEX_PARTIAL_FLUSH))

        # Copy WPTR to memory (src_sel = perf, dst_sel = tc_l2, wr_confirm = True)
        self.pkt3(self.pm4.PACKET3_COPY_DATA, 1 << 20 | 2 << 8 | 4, self.gc.regSQ_THREAD_TRACE_WPTR.addr[0], 0, wptrs + se * 4)

    self.set_grbm()
    if self.target[0] != 9: self.spi_config(tracing=False)
    self.memory_barrier()

  ### exec

  def kernargs(self, call:UOp, prg:UOp, data:AMDProgramData) -> list[UOp]:
    args = [get_call_arg_uops(call)[gi].getaddr(self.devs) for gi in prg.arg.globals] + \
            [b.ccast(v.dtype) for v, b in zip(prg.arg.vars, get_call_var_uops(call, prg))] # a bound value is a bare const, the var has the width
    return pack_args(layout_args(args), data.kernargs_segment_size) + (dispatch_packet(data, prg.arg) if data.enable_dispatch_ptr else [])

  def exec(self, call:UOp, prg:UOp):
    data, lib = amd_build_program(self.dev, prg, self.devs)
    info = prg.arg

    # kernargs: a nested blob linear inside a getaddr, merged into the kernargs buffer
    ka = UOp(Ops.LINEAR, src=tuple(self.kernargs(call, prg, data)), arg="kernargs")

    prog_addr = lib.getaddr(self.devs) + data.entry_point_offset
    scratch_addr = UOp.placeholder((data.private_segment_size,), dtypes.uint8, 0, device=self.devs).rtag("scratch").getaddr(self.devs)
    args_addr = ka.getaddr(self.devs)

    user_regs:list = []
    if data.enable_private_segment_sgpr: user_regs = [scratch_addr | (1 << 63), 0xffffffff, 0x20c14000]
    if data.enable_dispatch_ptr: user_regs += [args_addr + data.kernargs_segment_size]
    user_regs += [args_addr]

    dispatch_init = self.gc.regCOMPUTE_DISPATCH_INITIATOR.encode(
      **({'cs_w32_en': int(data.wave32)} if self.target[0] != 9 else {}), force_start_at_000=1, compute_shader_en=1)
    self.acquire_mem(gli=0, gl2=0)
    slot = self.prof_start(data, info, lib)
    self.wreg(self.gc.regCOMPUTE_PGM_LO, prog_addr >> 8)
    self.wreg(self.gc.regCOMPUTE_PGM_RSRC1, data.rsrc1, data.rsrc2)
    self.wreg(self.gc.regCOMPUTE_PGM_RSRC3, data.rsrc3)
    self.wreg(self.gc.regCOMPUTE_TMPRING_SIZE, self.dev.tmpring_size(data.private_segment_size))
    for xcc_id in range(self.dev.xccs): # architected flat scratch: each xcc gets its part
      with self.pred_exec(xcc_mask=1 << xcc_id):
        self.wreg(self.gc.regCOMPUTE_DISPATCH_SCRATCH_BASE_LO, (scratch_addr + data.private_segment_size // self.dev.xccs * xcc_id) >> 8)
    self.wreg(self.gc.regCOMPUTE_RESTART_X, 0, 0, 0)
    self.wreg(self.gc.regCOMPUTE_USER_DATA_0, *user_regs)
    self.wreg(self.gc.regCOMPUTE_RESOURCE_LIMITS, self.gc.regCOMPUTE_RESOURCE_LIMITS.encode(waves_per_sh=getenv("WAVES_PER_SH")))
    self.wreg(self.gc.regCOMPUTE_START_X, 0, 0, 0, *info.local_size, 0, 0)
    self.pkt3(self.pm4.PACKET3_DISPATCH_DIRECT, *info.global_size, dispatch_init)
    if self.dev.sqtt_enabled: self.pkt3(self.pm4.PACKET3_EVENT_WRITE, self.pm4.EVENT_TYPE(self.soc.THREAD_TRACE_MARKER) | self.pm4.EVENT_INDEX(0))
    self.pkt3(self.pm4.PACKET3_EVENT_WRITE, self.pm4.EVENT_TYPE(self.soc.CS_PARTIAL_FLUSH) | self.pm4.EVENT_INDEX(EVENT_INDEX_PARTIAL_FLUSH))
    self.prof_stop(slot)

  def wait(self, signal:UOp, value:UOp): self.wait_reg_mem(value.cast(dtypes.uint32), mem=signal.getaddr(self.devs))

  def timestamp(self, signal:UOp):
    with self.pred_exec(xcc_mask=0b1):
      self.release_mem(signal.getaddr(self.devs) + UOp.const(8, dtypes.uint64), 0, self.pm4.data_sel__mec_release_mem__send_gpu_clock_counter,
                       self.pm4.int_sel__mec_release_mem__none)

  def signal(self, signal:UOp, value:UOp):
    with self.pred_exec(xcc_mask=0b1):
      self.release_mem(signal.getaddr(self.devs), value, self.pm4.data_sel__mec_release_mem__send_32_bit_low,
                       self.pm4.int_sel__mec_release_mem__send_interrupt_after_write_confirm, cache_flush=True)

  def submit(self, cmdbuf:UOp) -> UOp: # the ring gets an indirect buffer packet: 4 dwords, put stays aligned so it never wraps mid packet
    base, off = unwrap_view(cmdbuf)
    blob = struct.pack("IIII", self.pm4.PACKET3(self.pm4.PACKET3_INDIRECT_BUFFER, 2), 0, 0, cmdbuf.max_numel() // 4 | self.pm4.INDIRECT_BUFFER_VALID)
    ib = UOp.placeholder((16,), dtypes.uint8, device=self.dev.host, tag=to_name("ib", self.queue))
    return self.push(self.prof_bump(cmdbuf), patch(ib, [(4, base.getaddr(self.devs) + off)], blob), self.dev.compute_queue)

  def push(self, cmdbuf:UOp, words:UOp, q, unit:int=4, doorbell_lag:int=0) -> UOp:
    ring, wptr, doorbell, put = _queue_args(self, q)
    rs, n, p = q.ring.size, words.max_numel() // 4, put.index(0).load() # put counts units, the ring dwords
    first = (rs - (tail:=((p * (unit // 4)) % rs).cast(dtypes.int))).minimum(n)
    for rid, (dst, src, count) in enumerate(((tail, 0, first), (0, first, n - first)), 10):
      i = UOp.range(count, rid, dtype=dtypes.int, src=(cmdbuf,))
      cmdbuf = ring.after(cmdbuf).index(dst + i).store(words.bitcast(dtypes.uint32).index(src + i).load()).end(i)
    w = wptr.after(cmdbuf).index(0).store(nxt:=p + words.max_numel() // unit)
    return doorbell.after(put.after(w).index(0).store(nxt)).index(0).store(nxt - doorbell_lag)

class AMDComputeAQLQueue(AMDComputeQueue): # the ring holds 64 byte aql packets: a dispatch per kernel, the pm4 between them wrapped as an ib
  def __init__(self, ctx, submit):
    super().__init__(ctx, submit)
    self.cmd_addr = UOp.variable("cmdbuf", 0, 2**48, dtypes.uint64) # the packets point into the cmdbuf, its address binds at submit
    self.pkts:list[UOp] = []
    self.run_start = 0

  def close_run(self, end:int):
    if end > self.run_start:
      hdr = AQL_HDR | (hsa.HSA_PACKET_TYPE_VENDOR_SPECIFIC << hsa.HSA_PACKET_HEADER_TYPE) | (1 << 16)
      ib = [self.pm4.PACKET3(self.pm4.PACKET3_INDIRECT_BUFFER, 2), self.cmd_addr + self.run_start,
            (end - self.run_start) // 4 | self.pm4.INDIRECT_BUFFER_VALID]
      self.pkts += [UOp.const(w, dtypes.uint32) if isinstance(w, int) else w for w in [hdr, *ib, 10, *[0] * 10]]
    self.run_start = end

  def signal(self, signal:UOp, value:UOp):
    self.close_run(len(self.blob)) # close pm4, so we can sync xccs. without that wait-signal can race
    super().signal(signal, value)

  def exec(self, call:UOp, prg:UOp):
    data, lib = amd_build_program(self.dev, prg, self.devs)
    self.dev.scratch_buffer(data.private_segment_size) # the queue descriptor holds the scratch
    slot = self.prof_start(data, prg.arg, lib)
    self.close_run(len(self.blob))
    kernarg_address = UOp(Ops.LINEAR, src=tuple(self.kernargs(call, prg, data)), arg="kernargs").getaddr(self.devs)
    self.pkts += [UOp.const(w, dtypes.uint32) if isinstance(w, int) else w
                  for w in dispatch_packet(data, prg.arg, lib.getaddr(self.devs) + data.desc_offset, kernarg_address)]
    self.run_start = len(self.blob)
    self.prof_stop(slot)

  def submit(self, cmdbuf:UOp) -> UOp: # the doorbell is the last packet's index
    self.close_run(cmdbuf.max_numel())
    base, off = unwrap_view(cmdbuf)
    self.blob, self.patches = bytearray(), [] # q again, for the aql stream
    self.q(*UOp.sink(*self.pkts).substitute({self.cmd_addr: base.getaddr(self.devs) + off}).src)
    return self.push(self.prof_bump(cmdbuf), bufferize_linear(self, "aql", self.dev.host), self.dev.compute_queue, unit=64, doorbell_lag=1)

# *****************
# SDMA

class AMDSDMAQueue(HWQueue):
  dev:AMDDevice

  def __init__(self, ctx, submit):
    super().__init__(ctx, submit)
    self.sdma, self.target, self.max_copy_size = self.dev.sdma, self.dev.target, self.dev.max_copy_size

  def copy(self, call:UOp):
    sz = call.src[2].max_numel() * call.src[2].dtype.itemsize
    hdr = self.sdma.SDMA_OP_COPY | self.sdma.SDMA_PKT_COPY_LINEAR_HEADER_SUB_OP(self.sdma.SDMA_SUBOP_COPY_LINEAR)
    for off in range(0, sz, self.max_copy_size):
      self.q(hdr, min(sz-off, self.max_copy_size)-1, 0,
             *(a + UOp.const(off, dtypes.uint64) if off else a for a in (call.src[2].getaddr(self.devs), call.src[1].getaddr(self.devs))))

  def wait(self, signal:UOp, value:UOp, eq:bool=False):
    func = WAIT_REG_MEM_FUNCTION_EQ if eq else WAIT_REG_MEM_FUNCTION_GEQ
    op = self.sdma.SDMA_OP_POLL_REGMEM | self.sdma.SDMA_PKT_POLL_REGMEM_HEADER_FUNC(func) | self.sdma.SDMA_PKT_POLL_REGMEM_HEADER_MEM_POLL(1)
    self.q(op, signal.getaddr(self.devs), value.cast(dtypes.uint32), (1 << 8 * min(value.dtype.itemsize, 4)) - 1,
           self.sdma.SDMA_PKT_POLL_REGMEM_DW5_INTERVAL(0x04) | self.sdma.SDMA_PKT_POLL_REGMEM_DW5_RETRY_COUNT(0xfff))

  def timestamp(self, signal:UOp):
    self.q(self.sdma.SDMA_OP_TIMESTAMP | self.sdma.SDMA_PKT_TIMESTAMP_GET_HEADER_SUB_OP(self.sdma.SDMA_SUBOP_TIMESTAMP_GET_GLOBAL),
           signal.getaddr(self.devs) + UOp.const(8, dtypes.uint64))

  def write(self, dst:UOp, *words):
    self.q(self.sdma.SDMA_OP_WRITE | self.sdma.SDMA_PKT_WRITE_UNTILED_HEADER_SUB_OP(self.sdma.SDMA_SUBOP_WRITE_LINEAR), dst.getaddr(self.devs),
           _dw(words) - 1, *words)

  def signal(self, signal:UOp, value:UOp):
    op = self.sdma.SDMA_OP_FENCE | (self.sdma.SDMA_PKT_FENCE_HEADER_MTYPE(3) if self.target[0] != 9 else 0)
    self.q(op, signal.getaddr(self.devs), value.cast(dtypes.uint32), self.sdma.SDMA_OP_TRAP, 0)

  def submit(self, cmdbuf:UOp) -> UOp:
    # sdma needs the cmdbuf contiguous in the ring: if it won't fit before the ring end, restart at 0 and zero the tail
    q = unwrap(self.dev.sdma_queue(int(self.queue.split(":")[1])))

    ring, wptr, doorbell, put = _queue_args(self, q)
    base = unwrap_view(cmdbuf)[0] # in host memory: streamed into the ring, the device never reads it
    cmdbuf = cmdbuf.substitute({base: base.replace(arg=replace(base.arg, device=self.dev.host))})

    rs, size_dw = q.ring.size, cmdbuf.max_numel() // 4
    put_b = put.index(0).load()
    tail = ((put_b % (rs * 4)) // 4).cast(dtypes.int)
    fits = (size_dw <= rs - tail).cast(dtypes.int)
    start_dw, zero_amt = fits * tail, (1 - fits) * (rs - tail)
    zi = UOp.range(zero_amt, 10, dtype=dtypes.int, src=(cmdbuf,))
    zero_tail = ring.index(tail + zi).store(UOp.const(0, dtypes.uint32)).end(zi)
    i = UOp.range(size_dw, 11, dtype=dtypes.int, src=(cmdbuf,))
    copy = ring.after(zero_tail).index(start_dw + i).store(cmdbuf.bitcast(dtypes.uint32).index(i).load()).end(i)
    next_put = put_b + ((zero_amt + size_dw) * 4).cast(put_b.dtype)
    w = wptr.after(copy).index(0).store(next_put)
    return doorbell.after(put.after(w).index(0).store(next_put)).index(0).store(next_put)

def amd_compute_queue(ctx, submit:UOp) -> HWQueue:
  return (AMDComputeAQLQueue if cast(AMDDevice, Device[submit.src[0].arg[0][0]]).is_aql else AMDComputeQueue)(ctx, submit)

@dataclass(frozen=True)
class AMDProgramData:
  desc_offset:int; entry_point_offset:int; rsrc1:int; rsrc2:int; rsrc3:int; wave32:bool; libhash:int # noqa: E702
  private_segment_size:int; group_segment_size:int; kernargs_segment_size:int # noqa: E702
  enable_dispatch_ptr:int; enable_private_segment_sgpr:int # noqa: E702

_amd_program_cache:dict[tuple[bytes, tuple[str, ...]], tuple[AMDProgramData, UOp]] = {}
_amd_program_prof:dict[UOp, tuple[str, bytes, bytes]] = {} # placeholder -> (name, lib, key) for its profile event
def amd_build_program(dev, prg:UOp, devs:tuple[str, ...]) -> tuple[AMDProgramData, UOp]:
  # the image parses once per lib, each device set gets its own program buffer of it
  if (cached:=_amd_program_cache.get(key:=(lib:=prg.src[3].arg, devs))) is None:
    data, image = _amd_program_image(dev, lib)
    buf = UOp.placeholder((len(image),), dtypes.uint8, next(UOp.unique_num), device=devs).rtag("program")
    cached = _amd_program_cache[key] = (data, buf.after(buf.store(UOp(Ops.BINARY, src=(), arg=image).bitcast(buf.dtype))))
    if PROFILE: _amd_program_prof[buf] = (prg.src[0].arg.function_name, lib, prg.key)
  return cached

@functools.cache
def _amd_program_image(dev, lib:bytes) -> tuple[AMDProgramData, bytes]:
  image, sections, relocs = elf_loader(lib)
  rodata = next(sh.header.sh_addr for sh in sections if sh.name == ".rodata")
  for off, sym, typ, addent in relocs:
    assert typ == 5, f"unknown AMD reloc {typ}"  # R_AMDGPU_REL64
    image[off:off+8] = struct.pack('<q', sym - off + addent)
  desc = amdgpu_kd.llvm_amdhsa_kernel_descriptor_t.from_buffer_copy(
    bytes(image[rodata:rodata+ctypes.sizeof(amdgpu_kd.llvm_amdhsa_kernel_descriptor_t)]))
  if (lds:=((desc.group_segment_fixed_size+511)//512)&0x1FF) > (dev.iface.props['lds_size_in_kb']*1024)//512:
    raise RuntimeError("Too many resources requested: group_segment_size")
  edp = desc.kernel_code_properties & hsa.AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_PTR

  data = AMDProgramData(desc_offset=rodata, entry_point_offset=rodata + desc.kernel_code_entry_byte_offset,
    rsrc1=desc.compute_pgm_rsrc1 | ((1<<20) if dev.target[0]==11 else 0),  # priv=1 on gfx11 for cwsr
    rsrc2=desc.compute_pgm_rsrc2 | (lds<<15), rsrc3=desc.compute_pgm_rsrc3, wave32=bool(desc.kernel_code_properties & 0x400),
    libhash=struct.unpack('<Q', hashlib.md5(lib).digest()[:8])[0], private_segment_size=desc.private_segment_fixed_size,
    group_segment_size=desc.group_segment_fixed_size, kernargs_segment_size=desc.kernarg_size, enable_dispatch_ptr=edp,
    enable_private_segment_sgpr=desc.kernel_code_properties & hsa.AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER)
  return data, bytes(image).ljust(round_up(len(image), 4), b"\x00") # the program is uploaded as whole dwords

class AMDAllocator(Allocator['AMDDevice']):
  def _alloc(self, size:int, options:BufferSpec) -> BufferStorage:
    return self.dev.iface.alloc(size, host=options.host, uncached=options.uncached, cpu_access=options.cpu_access or not self.dev.has_copy_queue)

  def _free(self, storage:BufferStorage, options:BufferSpec):
    self.dev.synchronize()
    self.dev.iface.free(storage)
  def _map(self, buf:Buffer) -> BufferStorage: return self.dev.iface.map(buf)
  def _unmap(self, mapping:BufferStorage): self.dev.iface.unmap(mapping)
  def _offset(self, buf:int, size:int, offset:int) -> int: return buf + offset

@dataclass
class AMDQueueDesc:
  ring: Buffer; read_ptr: Buffer; write_ptr: Buffer; doorbell: Buffer; put_value: Buffer  # noqa: E702
  eop_buffer: Buffer|None = None; cwsr_buffer: Buffer|None = None; params: tuple|None = None  # noqa: E702

class KFDIface:
  kfd:FileIOInterface|None = None
  event_page:Buffer
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

  def alloc(self, size:int, host=False, uncached=False, cpu_access=False, contiguous=False, cpu_addr=None) -> BufferStorage:
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

    self._map_handle(mem.handle)
    return BufferStorage(mem.va_addr, mem, MMIOInterface(mem.va_addr, mem.size, fmt='B') if cpu_access or host else None)

  def free(self, storage:BufferStorage):
    self._unmap_handle(storage.meta.handle)
    if storage.buf: FileIOInterface.munmap(storage.buf, storage.meta.size)
    kfd.AMDKFD_IOC_FREE_MEMORY_OF_GPU(self.kfd, handle=storage.meta.handle)

  def unmap(self, mapping:BufferStorage):
    handle, owned = mapping.meta
    self._unmap_handle(handle)
    if owned: kfd.AMDKFD_IOC_FREE_MEMORY_OF_GPU(self.kfd, handle=handle)

  def map(self, buf:Buffer) -> BufferStorage:
    if buf.device.split(":")[0] in {"CPU", "PYTHON", "NPY"}:
      if buf._buf % 0x1000: raise RuntimeError("Host mapping requires a page-aligned address")
      return replace(mem:=self.alloc(buf.nbytes, host=True, cpu_addr=buf._buf), meta=(mem.meta.handle, True))
    if buf.device.split(":")[0] != "AMD": raise RuntimeError(f"Cannot map {buf.device} on {self.dev.device}")
    self._map_handle(buf.meta.handle)
    return BufferStorage(buf._buf, (buf.meta.handle, False))

  def _map_handle(self, handle):
    gpus = (ctypes.c_int32 * 1)(self.gpu_id)
    stm = kfd.AMDKFD_IOC_MAP_MEMORY_TO_GPU(self.kfd, handle=handle, device_ids_array_ptr=ctypes.addressof(gpus), n_devices=1)
    assert stm.n_success == 1

  def _unmap_handle(self, handle):
    gpus = (ctypes.c_int32 * 1)(self.gpu_id)
    stm = kfd.AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU(self.kfd, handle=handle, device_ids_array_ptr=ctypes.addressof(gpus), n_devices=1)
    assert stm.n_success == 1

  def create_queue(self, queue_type, ring, gart, rptr, wptr, eop_buffer=None, cwsr_buffer=None, ctl_stack_size=0, ctx_save_restore_size=0,
                   xcc_id=0, idx=0):
    if not hasattr(self, 'queue_event_arr'):
      if not hasattr(KFDIface, 'event_page'):
        KFDIface.event_page = Buffer(self.dev.device, 0x8000, dtypes.uint8, options=BufferSpec(uncached=True), preallocate=True)
        kfd.AMDKFD_IOC_CREATE_EVENT(KFDIface.kfd, event_page_offset=KFDIface.event_page.meta.handle)

      KFDIface.event_page.get_buf(self.dev.device)
      self.queue_event_arr = (kfd.struct_kfd_event_data * 3)(*[kfd.struct_kfd_event_data(event_id=kfd.AMDKFD_IOC_CREATE_EVENT(
        KFDIface.kfd, event_type=t, auto_reset=int(t == kfd.KFD_IOC_EVENT_SIGNAL)).event_id)
        for t in (kfd.KFD_IOC_EVENT_SIGNAL, kfd.KFD_IOC_EVENT_MEMORY, kfd.KFD_IOC_EVENT_HW_EXCEPTION)])

    queue = kfd.AMDKFD_IOC_CREATE_QUEUE(KFDIface.kfd, ring_base_address=ring._buf, ring_size=ring.nbytes, gpu_id=self.gpu_id,
      queue_type=queue_type, queue_percentage=kfd.KFD_MAX_QUEUE_PERCENTAGE|(xcc_id<<8), queue_priority=getenv("AMD_KFD_QUEUE_PRIORITY", 7),
      eop_buffer_address=eop_buffer._buf if eop_buffer else 0, eop_buffer_size=eop_buffer.nbytes if eop_buffer else 0,
      ctl_stack_size=ctl_stack_size, ctx_save_restore_address=cwsr_buffer._buf if cwsr_buffer else 0, ctx_save_restore_size=ctx_save_restore_size,
      write_pointer_address=gart._buf+wptr, read_pointer_address=gart._buf+rptr+8*xcc_id)

    if not hasattr(self, 'doorbells'):
      self.doorbells_base = queue.doorbell_offset & (~0x1fff) # doorbell is two pages
      self.doorbells = cast(FileIOInterface, KFDIface.kfd).mmap(0, 0x2000, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED, self.doorbells_base)

    (put_value := Buffer("CPU", 1, dtypes.uint64, preallocate=True)).host.view(fmt='Q')[0] = 0
    doorbell = Buffer("CPU", 1, dtypes.uint64,
      options=BufferSpec(external_ptr=self.doorbells + queue.doorbell_offset - self.doorbells_base), preallocate=True)
    return AMDQueueDesc(ring=ring, doorbell=doorbell, read_ptr=gart.view(1, dtypes.uint64, rptr+8*xcc_id).ensure_allocated(),
      write_ptr=gart.view(1, dtypes.uint64, wptr).ensure_allocated(), put_value=put_value, eop_buffer=eop_buffer, cwsr_buffer=cwsr_buffer)

  def sleep(self, tm:int):
    kfd.AMDKFD_IOC_WAIT_EVENTS(KFDIface.kfd, events_ptr=ctypes.addressof(self.queue_event_arr), num_events=3, wait_for_all=0, timeout=tm)
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
    super().__init__(dev, dev_id, vendor=0x1002, devices=((0xffff, (0x74a1,0x74b5,0x744c,0x7480,0x7550,0x7551,0x7590,0x75a0,0x75a8,0x75b0,0x75b3)),),
      vram_bar=0, va_start=AMMemoryManager.va_allocator.base, va_size=AMMemoryManager.va_allocator.size, dev_impl_t=AMDev)
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
      doorbell_index = self.dev_impl.sdma.setup_ring(*(rcvr_params:=(ring._buf, ring.nbytes, gart._buf+rptr, gart._buf+wptr, idx)))
    else:
      doorbell_index = self.dev_impl.gfx.setup_ring(*(rcvr_params:=(ring._buf, ring.nbytes, gart._buf+rptr,
        gart._buf+wptr, eop_buffer._buf, eop_buffer.nbytes, is_aql:=(queue_type==kfd.KFD_IOC_QUEUE_TYPE_COMPUTE_AQL), is_aql)))

    put_value = Buffer(host:=self.dev.host, 1, dtypes.uint64, initial_value=bytes(8))
    doorbell = Buffer(host, 1, dtypes.uint64, options=BufferSpec(external_ptr=self.dev_impl.doorbell64.addr + doorbell_index*8), preallocate=True)
    return AMDQueueDesc(ring=ring, doorbell=doorbell, read_ptr=gart.view(1, dtypes.uint64, rptr).ensure_allocated(),
      write_ptr=gart.view(1, dtypes.uint64, wptr).ensure_allocated(), put_value=put_value, eop_buffer=eop_buffer, params=rcvr_params)

  def _collect_interrupts(self, reset=False, drain_only=False):
    d = self.dev
    if drain_only: d.iface.dev_impl.ih.drain()
    else: d.iface.dev_impl.ih.interrupt_handler()

    if reset and d.iface.dev_impl.recover(force=True):
      cq = d.compute_queue
      for b in (cq.put_value, cq.read_ptr, cq.write_ptr): b.host.view(fmt='Q')[0] = 0
      d.iface.dev_impl.gfx.setup_ring(*cq.params)
      tl = d.timeline.host.view(fmt='Q')
      tl[0] = tl[1]

  def sleep(self, timeout):
    if hasattr(self.pci_dev, 'irq_poller') and self.pci_dev.irq_poller is not None and (events_cnt:=len(self.pci_dev.irq_poller.poll(timeout))):
      self.pci_dev.irq_fd.read(8 * events_cnt)
    self._collect_interrupts()
    if self.dev_impl.is_err_state: raise RuntimeError("Device is in error state")

  def on_device_hang(self):
    self._collect_interrupts(reset=self.dev.can_recover)
    raise RuntimeError("Device hang detected")

  def device_fini(self): self.dev_impl.fini()

class USBAllocator(AMDAllocator): # the host program reads another device's memory in place: its bytes are the mapping
  def map(self, buf:Buffer) -> BufferStorage: return BufferStorage(buf.host.addr, buf.host.mv)
  def _unmap(self, mapping:BufferStorage): pass

class USBIface(PCIIface):
  def __init__(self, dev, dev_id): # pylint: disable=super-init-not-called
    if dev_id >= len(visible:=hcq_filter_visible_devices(USB3.list_devices(0xADD1, 0x0001) + USB3.list_devices(0x3801, 0x0001), "AMD")):
      raise RuntimeError(f"AMD:{dev_id} does not exist ({pluralize('device', len(visible))} available)")
    self.dev, self.pci_dev, self.vram_bar, self.count = dev, USBPCIDevice("AM", *visible[dev_id]), 0, len(visible)
    self.dev_impl = AMDev(self.pci_dev)
    self._compute_props()

  @functools.cached_property
  def ctrl(self) -> Buffer:
    # the controller's memory the queue and the host share, one range (usb.py slices it): the sys page at 0, the cq page at 0x1000, the sram at
    # 0x5000. the host's view starts at the sys page's controller address 0xa000, which puts the sram on its scsi window 0xf000
    vaddr, pieces = self.dev_impl.mm.alloc_vaddr(size=0x85000), [(0x0, 0x820000, 0x1000), (0x1000, 0x822000, 0x1000), (0x5000, 0x200000, 0x80000)]
    for off, paddr, n in pieces: self.dev_impl.mm.map_range(vaddr + off, n, [(paddr, n)], aspace=AddrSpace.SYS, uncached=True)
    view = self.pci_dev.dma_view(0xa000, 0x85000)
    for off, n in ((0x800, 4), (0x5000, 0x80000)): view.view(off, n)[:] = bytes(n) # no stale fence or sentinel
    return Buffer(self.dev.device, 0x85000, dtypes.uint8, options=BufferSpec(external_ptr=vaddr), opaque=BufferStorage(vaddr, host=view))

  def alloc(self, size:int, host=False, uncached=False, cpu_access=False, contiguous=False, force_devmem=False, zero=False,
            **kwargs) -> BufferStorage:
    # everything, even host-style signals, lives in vram: gpu writes into the bridge's own memory collide with an armed 0xF2 read stream
    return super().alloc(size, host=False, uncached=uncached, cpu_access=cpu_access or host, contiguous=contiguous, force_devmem=True, **kwargs)

  def sleep(self, timeout): pass

def _mock(iface, name=None): return type(name or f"MOCK{iface.__name__}", (iface,), {})

class AMDDevice(Compiled):
  timestamp_divider = 100.0  # AMD GPU clock: ticks/us
  sleep_timeout_ms = 200
  max_scratch_psize = 0
  pm_encode = PatternMatcher([
    (UPat(Ops.CUSTOM_FUNCTION, arg="submit_amd_compute", name="submit"), lambda ctx, submit: encode_submit(amd_compute_queue(ctx, submit))),
    (UPat(Ops.CUSTOM_FUNCTION, arg="submit_amd_copy", name="submit"), lambda ctx, submit: encode_submit(AMDSDMAQueue(ctx, submit))),
  ])

  ifaces = [KFDIface, PCIIface, USBIface, _mock(KFDIface, "MOCKIface"), _mock(KFDIface), _mock(PCIIface), _mock(USBIface)]

  def device_props(self): return self.iface.props

  def is_am(self) -> bool: return isinstance(self.iface, (PCIIface,))

  def __init__(self, device:str=""):
    self.iface = self._select_iface(device)
    self.is_usb = isinstance(self.iface, USBIface)
    self.is_vf = self.is_am() and self.iface.dev_impl.is_vf
    self.can_recover, self.rtalloc_size = self.is_am() and not self.is_vf, (4 if self.is_usb else 64)<<20

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
    self.max_copy_size = 0x40000000 if (4, 4, 2) <= (v:=self.iface.ip_versions[am.SDMA0_HWIP]) < (5, 0, 0) or v >= (5, 2, 0) else 0x400000
    self.sdma_queues:dict = {}

    allocator = USBAllocator(self) if self.is_usb else AMDAllocator(self)
    super().__init__(device, allocator, [HIPRenderer, AMDLLVMRenderer, HIPCCRenderer], None, arch=self.arch)

    # Scratch setup
    self.max_private_segment_size = 0
    self.pm_bufferize = PatternMatcher([
      (UPat(Ops.PARAM, tag="scratch", name="b"), lambda ctx, b: ctx.scratch_buffer(b.max_numel())),
      (UPat(Ops.PARAM, tag="program", name="b"), lambda ctx, b: ctx.program_buffer(b)),
      (UPat(Ops.PARAM, name="b"), lambda ctx, b: ctx.queue_buffer(b.tag)),
    ]) + self.pm_bufferize

    if self.is_usb: # the submits write the rings over the link, the copies go through the controller's sram (usb.py)
      self.pm_batch, self.pm_lower = pm_usb_batch, pm_usb_lower
      self.pm_bufferize = pm_usb_bufferize + self.pm_bufferize

    # SQTT is disabled by default because of runtime overhead and big file sizes (~200mb to Tensor.full() two 4096x4096 tensors and matmul them)
    self.pmc_enabled, self.sqtt_enabled = PROFILE > 0 and PMC > 0, PROFILE > 0 and SQTT > 0
    if self.pmc_enabled or self.sqtt_enabled:
      self.iface.require_profile_mode()
      self.prof_slots, self.prof_read, self.sqtt_next_cmd_id = getenv("PROF_SLOTS", 32), 0, itertools.count(0)
      self.pmc_sched:list[PMCSample] = []
      self.sqtt_ses, self.sqtt_win = self.se_cnt * self.xccs, (getenv("SQTT_BUFFER_SIZE", 256) << 20) // self.prof_slots # mb, per shader engine
      self.pm_bufferize = PatternMatcher([(UPat(Ops.PARAM, tag=n), lambda ctx, n=n: getattr(ctx, n))
                                          for n in ("prof_log", "pmc_buf", "sqtt_buf", "sqtt_wptrs")]) + self.pm_bufferize
    if self.pmc_enabled:
      self.pmc_counters = import_pmc(self.target)
      # validate counters: SQ for SIMD busy/instruction counts, LDS stats, GRBM for GPU cycles, L2 cache hits/misses
      l2, lds = ("TCC", "SQ") if self.target[0] == 9 else ("GL2C", "SQC")
      pmc_default = f"SQ_BUSY_CYCLES,SQ_INSTS_VALU,SQ_INSTS_SALU,{lds}_LDS_IDX_ACTIVE,{lds}_LDS_BANK_CONFLICT,GRBM_GUI_ACTIVE,{l2}_HIT,{l2}_MISS"
      self.pmc_names = getenv("PMC_COUNTERS", pmc_default).split(",")
      for k in self.pmc_names:
        if k not in self.pmc_counters: raise RuntimeError(f"PMC counter {k} is not supported. Available: {','.join(self.pmc_counters.keys())}")

    # the pf resets a vf that holds its init access for 20s: set up every queue now and hand it back
    if self.is_vf:
      self.compute_queue # the property builds it
      for idx in range(min(self.iface.count, 8)): self.sdma_queue(idx)
      self.iface.dev_impl.release_vf_access()

  def create_queue(self, queue_type, ring_size, ctx_save_restore_size=0, eop_buffer_size=0, ctl_stack_size=0, debug_memory_size=0, idx=0):
    spec = BufferSpec(host=True, uncached=True, cpu_access=True)
    ring = Buffer(self.device, ring_size // 4, dtypes.uint32, options=spec, preallocate=True, allocator=self.allocator)
    gart = Buffer(self.device, 0x100, dtypes.uint8, options=spec, initial_value=bytes(0x100), allocator=self.allocator)

    if queue_type == kfd.KFD_IOC_QUEUE_TYPE_COMPUTE_AQL:
      self.aql_gart = gart
      self.aql_desc = hsa.amd_queue_t(queue_properties=hsa.AMD_QUEUE_PROPERTIES_IS_PTR64 | hsa.AMD_QUEUE_PROPERTIES_ENABLE_PROFILING,
        read_dispatch_id_field_base_byte_offset=getattr(hsa.amd_queue_t, 'read_dispatch_id').offset,
        max_cu_id=(self.cu_cnt * self.xccs) - 1, max_wave_id=self.waves_per_cu - 1)
      if hasattr(self, 'scratch'): self.aql_scratch()
      else: self.aql_gart.host.view(fmt='B')[:ctypes.sizeof(self.aql_desc)] = bytes(self.aql_desc)

    cwsr_buffer_size = round_up((ctx_save_restore_size + debug_memory_size) * self.xccs, mmap.PAGESIZE)
    cwsr_buffer = Buffer(self.device, cwsr_buffer_size, dtypes.uint8, preallocate=True, allocator=self.allocator) if ctx_save_restore_size else None
    eop_buffer = Buffer(self.device, eop_buffer_size, dtypes.uint8, preallocate=True, allocator=self.allocator) if eop_buffer_size else None

    return self.iface.create_queue(queue_type, ring, gart, rptr=getattr(hsa.amd_queue_t, 'read_dispatch_id').offset,
             wptr=getattr(hsa.amd_queue_t, 'write_dispatch_id').offset, eop_buffer=eop_buffer, cwsr_buffer=cwsr_buffer,
             ctx_save_restore_size=ctx_save_restore_size, ctl_stack_size=ctl_stack_size, idx=idx)

  def queue_buffer(self, tag):
    if not isinstance(tag, str) or not tag.startswith(("ring_", "write_ptr_", "doorbell_", "put_value_")): return None
    name, queue, idx = tag.rsplit('_', 2)
    return getattr(self.compute_queue if queue == 'compute' else self.sdma_queue(int(idx)), name)

  @functools.cached_property
  def compute_queue(self) -> AMDQueueDesc:
    # https://gitlab.freedesktop.org/agd5f/linux/-/blob/a1fc9f584c4aaf8bc1ebfa459fc57a3f26a290d8/drivers/gpu/drm/amd/amdkfd/kfd_queue.c#L391
    sgrp_size_per_cu, hwreg_size_per_cu = 0x4000, 0x1000
    lds_size_per_cu = self.iface.props["lds_size_in_kb"] << 10 if self.target[:2] == (9,5) else 0x10000
    vgpr_size_per_cu = 0x60000 if self.target in {(11,0,0), (11,0,1), (11,5,1), (12,0,0), (12,0,1)} else 0x80000 if self.target[0] == 9 else 0x40000
    wg_data_size = round_up((vgpr_size_per_cu + sgrp_size_per_cu + lds_size_per_cu + hwreg_size_per_cu) * self.cu_cnt, mmap.PAGESIZE)
    ctl_stack_size = round_up((12 if self.target[0] != 9 else 8) * self.wave_cnt + 8 + 40, mmap.PAGESIZE)
    return self.create_queue(kfd.KFD_IOC_QUEUE_TYPE_COMPUTE_AQL if self.is_aql else kfd.KFD_IOC_QUEUE_TYPE_COMPUTE,
      (1 << 20) if self.is_usb else (16 << 20), eop_buffer_size=0x1000,
      ctx_save_restore_size=0 if self.is_am() else wg_data_size + ctl_stack_size, ctl_stack_size=ctl_stack_size,
      debug_memory_size=round_up(self.wave_cnt * 32, 64))

  @functools.cached_property
  def has_copy_queue(self) -> bool:
    self.has_copy_queue = False # queue setup can allocate buffers that check this property
    return self.sdma_queue(0) is not None

  def sdma_queue(self, idx:int):
    if getenv("AMD_DISABLE_SDMA"): return None
    if idx in self.sdma_queues: return self.sdma_queues[idx]
    with contextlib.suppress(OSError):
      self.sdma_queues[idx] = self.create_queue(kfd.KFD_IOC_QUEUE_TYPE_SDMA, (1 << 20) if self.is_usb else (16 << 20), idx=idx)
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
    return int.from_bytes(tmpring_t(WAVES=min(num_waves, max_scratch_waves), WAVESIZE=wave_scratch), 'little')

  def scratch_buffer(self, private_segment_size):
    AMDDevice.max_scratch_psize = private_segment_size = max(private_segment_size, 128, AMDDevice.max_scratch_psize)
    if self.max_private_segment_size < private_segment_size:
      lanes_per_wave = 64 # wave64
      mem_alignment_size = 256 if self.target[0] != 9 else 1024
      size_per_thread = round_up(private_segment_size, mem_alignment_size // lanes_per_wave)
      size_per_xcc = size_per_thread * lanes_per_wave * self.iface.props['max_slots_scratch_cu'] * self.cu_cnt
      self.scratch = Buffer(self.device, size_per_xcc * self.xccs, dtypes.uint8, options=BufferSpec(nolru=True), preallocate=True)
      self.max_private_segment_size = private_segment_size
      if hasattr(self, 'aql_desc'): self.aql_scratch()
    return self.scratch

  def aql_scratch(self):
    gfx9_rsrc = {'NUM_FORMAT':hsa.BUF_NUM_FORMAT_UINT, 'DATA_FORMAT':hsa.BUF_DATA_FORMAT_32, 'ELEMENT_SIZE':1, 'INDEX_STRIDE':3}
    rsrc = {'DST_SEL_X':hsa.SQ_SEL_X, 'DST_SEL_Y':hsa.SQ_SEL_Y, 'DST_SEL_Z':hsa.SQ_SEL_Z, 'DST_SEL_W':hsa.SQ_SEL_W, 'ADD_TID_ENABLE':1,
            'TYPE':hsa.SQ_RSRC_BUF, **(gfx9_rsrc if self.target[0] == 9 else {'FORMAT':hsa.BUF_FORMAT_32_UINT, 'OOB_SELECT':2})}
    rsrc1_t = getattr(hsa, f'union_SQ_BUF_RSRC_WORD1{"_GFX11" if self.target[0] != 9 else ""}_bitfields')
    rsrc3_t = getattr(hsa, f'union_SQ_BUF_RSRC_WORD3{"_GFX"+str(self.target[0]) if self.target[0] != 9 else ""}_bitfields')

    base = self.scratch._buf
    self.aql_desc.scratch_backing_memory_location = base
    self.aql_desc.scratch_wave64_lane_byte_size = self.max_private_segment_size
    self.aql_desc.scratch_resource_descriptor[:] = [lo32(base), int.from_bytes(rsrc1_t(BASE_ADDRESS_HI=hi32(base), SWIZZLE_ENABLE=1), 'little'),
                                                    lo32(self.scratch.nbytes // self.xccs), int.from_bytes(bytes(rsrc3_t(**rsrc)), 'little')]
    self.aql_desc.compute_tmpring_size = self.tmpring_size(self.max_private_segment_size)
    self.aql_gart.host.view(fmt='B')[:ctypes.sizeof(self.aql_desc)] = bytes(self.aql_desc)

  def _prof_buffer(self, size:int, dtype, host:bool=True) -> Buffer:
    buf = Buffer(self.device, size, dtype, options=BufferSpec(host=host, nolru=True, uncached=host, cpu_access=True), preallocate=True)
    buf.host.view(fmt='B')[:buf.nbytes] = bytes(buf.nbytes)
    return buf

  @functools.cached_property
  def prof_log(self) -> Buffer: return self._prof_buffer(1 + self.prof_slots, dtypes.uint64)
  @property
  def pmc_size(self) -> int: return self.pmc_sched[-1].off + self.pmc_sched[-1].size
  @functools.cached_property
  def pmc_buf(self) -> Buffer: return self._prof_buffer(self.pmc_size * self.prof_slots, dtypes.uint8)
  @functools.cached_property
  def sqtt_buf(self) -> Buffer: return self._prof_buffer(self.sqtt_win * self.prof_slots * self.sqtt_ses, dtypes.uint8, host=False)
  @functools.cached_property
  def sqtt_wptrs(self) -> Buffer: return self._prof_buffer(self.prof_slots * self.sqtt_ses, dtypes.uint32)

  def program_buffer(self, b:UOp) -> Buffer:
    if b not in self.prog_bufs:
      buf = self.prog_bufs[b] = Buffer(self.device, b.max_numel(), b.dtype, options=BufferSpec(cpu_access=True, nolru=True)).ensure_allocated()
      if PROFILE:
        name, lib, key = _amd_program_prof[b]
        Compiled.profile_events.append(ProfileProgramEvent(self.device, name, lib, buf._buf, b.arg.slot, key))
    return self.prog_bufs[b]

  def sqtt_trace(self, slot:int, se:int) -> bytes:
    off = (se * self.prof_slots + slot) * self.sqtt_win
    wptr = (self.sqtt_wptrs.host.view(fmt='I')[slot * self.sqtt_ses + se] & 0x1FFFFFFF) * 32
    if self.target[:2] == (11, 0): wptr -= (((self.sqtt_buf._buf + off) // 32) & 0x1FFFFFFF) * 32
    assert 0 <= wptr <= self.sqtt_win, f"{wptr} > {self.sqtt_win}, should never happen"
    if wptr >= self.sqtt_win - 32: # the wptr stops at the last dword when the window overflows
      print(colored(f"{self.device}: Warning: SQTT buffer is full (SE {se})! Increase SQTT buffer with SQTT_BUFFER_SIZE=X (in MB)", "yellow"))
    blob = bytes(self.sqtt_buf.host.view(fmt='B')[off:off + wptr])
    return (struct.pack('<Q', 0x11 | (4 << 13) | (0xf << 16) | (se << 24)) + blob) if self.target[0] == 9 else blob

  def _at_profile_finalize(self): # the calibration kernels aren't profiles
    self.synchronize()
    super()._at_profile_finalize()
    if self.pmc_enabled or self.sqtt_enabled: self.prof_read = self.prof_log.host.view(fmt='Q')[0]

  def collect_prof(self):
    if self.pmc_enabled or self.sqtt_enabled:
      log = self.prof_log.host.view(fmt='Q')
      if (lost:=log[0] - self.prof_read - self.prof_slots) > 0:
        print(colored(f"{self.device}: Warning: {lost} kernel profiles were overwritten: synchronize more often or raise PROF_SLOTS", "yellow"))
      for k in range(max(self.prof_read, log[0] - self.prof_slots), log[0]):
        slot, tag = k % self.prof_slots, log[1 + k % self.prof_slots]
        if self.pmc_enabled:
          blob = bytes(self.pmc_buf.host.view(fmt='B')[slot * self.pmc_size:(slot + 1) * self.pmc_size])
          Compiled.profile_events.append(ProfilePMCEvent(self.device, tag, self.pmc_sched, blob, k))
        for se in range(self.sqtt_ses if self.sqtt_enabled else 0):
          itrace = bool((SQTT_ITRACE_SE_MASK.value >> se) & 1)
          Compiled.profile_events.append(ProfileSQTTEvent(self.device, tag, se, self.sqtt_trace(slot, se), itrace, k))
      self.prof_read = log[0]
    super().collect_prof()

  def on_device_hang(self): self.iface.on_device_hang()

if not HCQ2: from extra.hcq1.ops_amd_old import * # noqa: F401, F403 # pylint: disable=unused-import
