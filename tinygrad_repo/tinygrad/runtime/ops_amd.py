from __future__ import annotations
from typing import cast, ClassVar
import os, ctypes, struct, hashlib, functools, importlib, mmap, errno, array, contextlib, sys, weakref, itertools, collections, atexit
assert sys.platform != 'win32'
from dataclasses import dataclass
from tinygrad.runtime.support.hcq import HCQCompiled, HCQAllocator, HCQBuffer, HWQueue, CLikeArgsState, HCQSignal, HCQProgram, FileIOInterface
from tinygrad.runtime.support.hcq import MMIOInterface, BumpAllocator, hcq_filter_visible_devices
from tinygrad.uop.ops import sint
from tinygrad.device import Compiled, DMAFdRef, BufferSpec, CompilerSet, CompilerPair
from tinygrad.helpers import getenv, round_up, data64_le, DEBUG, PROFILE, ProfileEvent, lo32, hi32, colored, prod, ContextVar
from tinygrad.helpers import VIZ, AMD_CC, AMD_LLVM, ceildiv
from tinygrad.renderer.cstyle import AMDHIPRenderer, AMDHIPCCRenderer
from tinygrad.renderer.llvmir import AMDLLVMRenderer
from tinygrad.runtime.autogen import kfd, hsa, pci, sqtt, amdgpu_kd, amdgpu_drm
from tinygrad.runtime.autogen.am import am
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.runtime.support.am.amdev import AMDev, AMMemoryManager
from tinygrad.runtime.support.amd import AMDReg, AMDIP, import_module, import_soc, import_ip_offsets, import_pmc
from tinygrad.runtime.support.system import System, PCIIfaceBase, PCIAllocationMeta, PCIDevice, USBPCIDevice, MAP_FIXED, MAP_NORESERVE
from tinygrad.runtime.support.memory import AddrSpace
if getenv("IOCTL"): import extra.hip_gpu_driver.hip_ioctl  # noqa: F401 # pylint: disable=unused-import

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

class AMDSignal(HCQSignal):
  def __init__(self, *args, **kwargs): super().__init__(*args, **{**kwargs, 'timestamp_divider': 100})

  def _sleep(self, time_spent_waiting_ms:int) -> bool:
    # Resonable to sleep for long workloads (which take more than 2s) and only timeline signals.
    if time_spent_waiting_ms > 2000 and self.is_timeline and self.owner is not None: return self.owner.iface.sleep(200)
    return False

class AMDComputeQueue(HWQueue):
  def __init__(self, dev:AMDDevice):
    self.dev, self.soc, self.pm4, self.gc, self.nbio = dev, dev.soc, dev.pm4, dev.gc, dev.nbio
    super().__init__()

  def __del__(self):
    if self.binded_device is not None:
      self.binded_device.allocator.free(self.hw_page, self.hw_page.size, BufferSpec(cpu_access=True, nolru=True, uncached=True))

  def pkt3(self, cmd, *vals): self.q(self.pm4.PACKET3(cmd, len(vals) - 1), *vals)

  def wreg(self, reg:AMDReg, *args:sint, **kwargs:int):
    if bool(args) == bool(kwargs): raise RuntimeError('One (and only one) of *args or **kwargs must be specified')
    if self.pm4.PACKET3_SET_SH_REG_START <= reg.addr[0] < self.pm4.PACKET3_SET_SH_REG_END:
      set_packet, set_packet_start = self.pm4.PACKET3_SET_SH_REG, self.pm4.PACKET3_SET_SH_REG_START
    elif self.pm4.PACKET3_SET_UCONFIG_REG_START <= reg.addr[0] < self.pm4.PACKET3_SET_UCONFIG_REG_START + 2**16-1:
      set_packet, set_packet_start = self.pm4.PACKET3_SET_UCONFIG_REG, self.pm4.PACKET3_SET_UCONFIG_REG_START
    else: raise RuntimeError(f'Cannot set {reg.name} ({reg.addr[0]}) via pm4 packet')
    self.pkt3(set_packet, reg.addr[0] - set_packet_start, *(args or (reg.encode(**kwargs),)))

  @contextlib.contextmanager
  def pred_exec(self, xcc_mask:int):
    if self.dev.xccs > 1:
      self.pkt3(self.pm4.PACKET3_PRED_EXEC, xcc_mask << 24)
      prev_len = len(self._q)
    yield
    if self.dev.xccs > 1:
      self._q[prev_len-1] |= (len(self._q) - prev_len)

  def set_grbm(self, instance=None, se=None, sh=None, wgp=None):
    instance_val = (wgp << 2 | (instance or 0)) if wgp is not None else instance
    self.wreg(self.gc.regGRBM_GFX_INDEX, **{(f'{key}_broadcast_writes' if val is None else f'{key}_index'): (1 if val is None else val)
      for key, val in [('instance', instance_val), ('se', se), ('sh' if self.dev.target[0] == 9 else 'sa', sh)]})

  def wait_reg_mem(self, value, mask=0xffffffff, mem=None, reg=None, reg_done=0, op=WAIT_REG_MEM_FUNCTION_GEQ):
    wrm_info_dw = self.pm4.WAIT_REG_MEM_MEM_SPACE(int(mem is not None)) | self.pm4.WAIT_REG_MEM_OPERATION(int(mem is None and reg_done > 0)) \
                | self.pm4.WAIT_REG_MEM_FUNCTION(op) | self.pm4.WAIT_REG_MEM_ENGINE(0)

    self.pkt3(self.pm4.PACKET3_WAIT_REG_MEM, wrm_info_dw, *(data64_le(mem) if mem is not None else (reg, reg_done)), value, mask, 4)

  def acquire_mem(self, addr=0x0, sz=(1 << 64)-1, gli=1, glm=1, glk=1, glv=1, gl1=1, gl2=1):
    if self.dev.target >= (10,0,0):
      cache_flags_dw = self.pm4.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLI_INV(gli) \
                     | self.pm4.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLM_INV(glm) | self.pm4.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLM_WB(glm) \
                     | self.pm4.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLK_INV(glk) | self.pm4.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLK_WB(glk) \
                     | self.pm4.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLV_INV(glv) | self.pm4.PACKET3_ACQUIRE_MEM_GCR_CNTL_GL1_INV(gl1) \
                     | self.pm4.PACKET3_ACQUIRE_MEM_GCR_CNTL_GL2_INV(gl2) | self.pm4.PACKET3_ACQUIRE_MEM_GCR_CNTL_GL2_WB(gl2)

      self.pkt3(self.pm4.PACKET3_ACQUIRE_MEM, 0, *data64_le(sz), *data64_le(addr), 0, cache_flags_dw)
    else:
      cp_coher_cntl = self.pm4.PACKET3_ACQUIRE_MEM_CP_COHER_CNTL_SH_ICACHE_ACTION_ENA(gli) | \
                      self.pm4.PACKET3_ACQUIRE_MEM_CP_COHER_CNTL_SH_KCACHE_ACTION_ENA(glk) | \
                      self.pm4.PACKET3_ACQUIRE_MEM_CP_COHER_CNTL_TC_ACTION_ENA(gl2) | \
                      self.pm4.PACKET3_ACQUIRE_MEM_CP_COHER_CNTL_TCL1_ACTION_ENA(gl1) | \
                      self.pm4.PACKET3_ACQUIRE_MEM_CP_COHER_CNTL_TC_WB_ACTION_ENA(gl2)
      self.pkt3(self.pm4.PACKET3_ACQUIRE_MEM, cp_coher_cntl, *data64_le(sz), *data64_le(addr), 0x0000000A)

  def release_mem(self, address=0x0, value=0, data_sel=0, int_sel=2, ctxid=0, cache_flush=False):
    if self.dev.target >= (10,0,0):
      cache_flags_dw = 0 if not cache_flush else (self.pm4.PACKET3_RELEASE_MEM_GCR_GLV_INV | self.pm4.PACKET3_RELEASE_MEM_GCR_GL1_INV \
                     | self.pm4.PACKET3_RELEASE_MEM_GCR_GL2_INV | self.pm4.PACKET3_RELEASE_MEM_GCR_GLM_WB \
                     | self.pm4.PACKET3_RELEASE_MEM_GCR_GLM_INV | self.pm4.PACKET3_RELEASE_MEM_GCR_GL2_WB | self.pm4.PACKET3_RELEASE_MEM_GCR_SEQ)

      event_dw = self.pm4.PACKET3_RELEASE_MEM_EVENT_TYPE(self.pm4.CACHE_FLUSH_AND_INV_TS_EVENT) \
               | self.pm4.PACKET3_RELEASE_MEM_EVENT_INDEX(self.pm4.event_index__mec_release_mem__end_of_pipe)

      memsel_dw = self.pm4.PACKET3_RELEASE_MEM_DATA_SEL(data_sel) | self.pm4.PACKET3_RELEASE_MEM_INT_SEL(int_sel) \
                | self.pm4.PACKET3_RELEASE_MEM_DST_SEL(0)
    else:
      cache_flags_dw = 0 if not cache_flush else (self.pm4.EOP_TC_WB_ACTION_EN | self.pm4.EOP_TC_NC_ACTION_EN)

      event_dw = self.pm4.EVENT_TYPE(self.pm4.CACHE_FLUSH_AND_INV_TS_EVENT) | self.pm4.EVENT_INDEX(self.pm4.event_index__mec_release_mem__end_of_pipe)

      memsel_dw = self.pm4.DATA_SEL(data_sel) | self.pm4.INT_SEL(int_sel)

      ctxid = 0

    self.pkt3(self.pm4.PACKET3_RELEASE_MEM, event_dw | cache_flags_dw, memsel_dw, *data64_le(address), *data64_le(value), ctxid)

  def memory_barrier(self):
    pf = '' if self.nbio.version[0] == 2 else '0' if self.nbio.version[:2] != (7, 11) else '1'
    self.wait_reg_mem(reg=getattr(self.nbio, f'regBIF_BX_PF{pf}_GPU_HDP_FLUSH_REQ').addr[0],
                      reg_done=getattr(self.nbio, f'regBIF_BX_PF{pf}_GPU_HDP_FLUSH_DONE').addr[0], value=0xffffffff)
    self.acquire_mem()
    return self

  def spi_config(self, tracing:bool):
    self.wreg(self.gc.regSPI_CONFIG_CNTL, ps_pkr_priority_cntl=3, exp_priority_order=3, gpr_write_priority=0x2c688,
              enable_sqg_bop_events=int(tracing), enable_sqg_top_events=int(tracing))

  ### PMC ###

  def pmc_reset_counters(self, en=True):
    self.set_grbm()
    self.wreg(self.gc.regCP_PERFMON_CNTL if self.dev.target[0] <= 11 else self.gc.regCP_PERFMON_CNTL_1, perfmon_state=0)
    if en: self.wreg(self.gc.regCP_PERFMON_CNTL if self.dev.target[0] <= 11 else self.gc.regCP_PERFMON_CNTL_1, perfmon_state=1)
    return self

  def pmc_start(self, counters):
    self.pmc_reset_counters(en=False)
    self.wreg(self.gc.regSQ_PERFCOUNTER_CTRL, cs_en=1, ps_en=1, gs_en=1, hs_en=1, **({'vmid_mask':0xffff} if (gfx9:=self.dev.target[0] == 9) else {}))
    if self.dev.target[0] >= 11: self.wreg(self.gc.regSQ_PERFCOUNTER_CTRL2, force_en=1, vmid_en=0xffff)

    end_off = 0
    block2pid:dict[str, itertools.count] = collections.defaultdict(lambda: itertools.count())
    for name,block,idx in counters:
      # sq block on gfx11+ goes down to wgps
      inst_cnt, se_cnt, sa_cnt, wgp_cnt = {"GRBM": (1, 1, 1, 1), "GL2C": (32, 1, 1, 1), "TCC": (16, 1, 1, 1),
        "SQ": (1, self.dev.se_cnt) + ((1, 1) if gfx9 else (2, self.dev.iface.props['cu_per_simd_array'] // 2))}[block]
      end_off += (rec_size:=prod((self.dev.xccs, inst_cnt, se_cnt, sa_cnt, wgp_cnt)) * 8)

      # gfx11+ and later require even-numbered SQ *_SELECT registers
      regsample = f'reg{block}_PERFCOUNTER{(pcid:=next(block2pid[block]))}'
      if (regsel:=getattr(self.gc, (f'reg{block}_PERFCOUNTER{(pcid*2) if self.dev.target[0]>=11 and block=="SQ" else pcid}_SELECT'), None)) is None:
        raise RuntimeError(f'{block} is out of perfcounter registers: ({regsample} is not found)')

      self.wreg(regsel, perf_sel=idx, **({'simd_mask':0xf, 'sqc_bank_mask':0xf, 'sqc_client_mask':0xf} if gfx9 and block == "SQ" else {}))
      self.dev.pmc_sched.append(PMCSample(name, block, self.dev.xccs, inst_cnt, se_cnt, sa_cnt, wgp_cnt, end_off-rec_size, rec_size, regsample))

    if gfx9: self.wreg(self.gc.regSQ_PERFCOUNTER_MASK, sh0_mask=0xffff, sh1_mask=0xffff)
    self.wreg(self.gc.regCOMPUTE_PERFCOUNT_ENABLE, 1)
    return self.pmc_reset_counters(en=True)

  def pmc_read(self, buf, sched):
    self.set_grbm()
    self.wreg(self.gc.regCP_PERFMON_CNTL if self.dev.target[0] <= 11 else self.gc.regCP_PERFMON_CNTL_1, perfmon_state=1, perfmon_sample_enable=1)

    for s in sched:
      offset = itertools.count(s.off, step=8)

      for xcc in range(s.xcc):
        with self.pred_exec(xcc_mask=1 << xcc):
          for inst, se_idx, sa_idx, wgp_idx in itertools.product(range(s.inst), range(s.se), range(s.sa), range(s.wgp)):
            loff = next(offset)
            if s.wgp > 1 and not self.dev.iface.is_wgp_active(xcc, se_idx, sa_idx, wgp_idx): continue
            self.set_grbm(**({'instance':inst} if s.inst > 1 else ({'se':se_idx}|({'sh':sa_idx, 'wgp':wgp_idx} if self.dev.target[0] != 9 else {}))))

            # Copy counter to memory (src_sel = perf, dst_sel = tc_l2)
            lo, hi = getattr(self.gc, f'{s.regsample}_LO'), getattr(self.gc, f'{s.regsample}_HI', None)
            self.pkt3(self.pm4.PACKET3_COPY_DATA, (2 << 8) | 4, lo.addr[0], 0, *data64_le(buf.va_addr+loff))
            if hi is not None: self.pkt3(self.pm4.PACKET3_COPY_DATA, (2 << 8) | 4, hi.addr[0], 0, *data64_le(buf.va_addr+loff+4))

    return self.pmc_reset_counters(en=True)

  ### SQTT ###

  def sqtt_setup_exec(self, prg, global_size):
    self.sqtt_userdata(sqtt.struct_rgp_sqtt_marker_pipeline_bind(identifier=sqtt.RGP_SQTT_MARKER_IDENTIFIER_BIND_PIPELINE,
                                                                 bind_point=(__BIND_POINT_COMPUTE:=1), api_pso_hash=data64_le(prg.libhash[0])))
    self.sqtt_userdata(sqtt.struct_rgp_sqtt_marker_event(has_thread_dims=1, cmd_id=next(prg.dev.sqtt_next_cmd_id)), *global_size)

    if SQTT_LIMIT_SE:
      # Calculate number of CUs per SE to enable based on blocks count. 4 is maximum simd per CU, but on rdna we can trace only 1.
      cu_per_se = prod([x if isinstance(x, int) else 1 for x in global_size]) // ((self.dev.cu_cnt // self.dev.se_cnt) * 4)
      for xcc in range(self.dev.xccs):
        with self.pred_exec(xcc_mask=1 << xcc):
          for i in range(8 if prg.dev.target >= (11,0,0) else 4):
            if SQTT_LIMIT_SE > 1: mask = 1 if SQTT_ITRACE_SE_MASK.value & (1 << i) else 0 # only run unmasked shader engines
            else:
              sa_mask = (1 << (self.dev.iface.props['cu_per_simd_array'] // 2)) - 1
              cu_mask = (1 << (cu_per_se + (1 if i == 0 else 0))) - 1
              mask = lo32((cu_mask & sa_mask) | (cu_mask & (sa_mask << 16)) << 16)
            self.wreg(getattr(self.gc, f'regCOMPUTE_STATIC_THREAD_MGMT_SE{i}'), mask)

  def sqtt_userdata(self, data, *extra_dwords):
    data_ints = [x[0] for x in struct.iter_unpack('<I', bytes(data))] + list(extra_dwords)
    for i in range(0, len(data_ints), 2):
      self.wreg(self.gc.regSQ_THREAD_TRACE_USERDATA_2, *data_ints[i:i+2])

  def sqtt_config(self, tracing:bool):
    trace_ctrl = {'rt_freq': self.soc.SQ_TT_RT_FREQ_4096_CLK} if self.dev.target < (12,0,0) else {}
    self.wreg(self.gc.regSQ_THREAD_TRACE_CTRL, draw_event_en=1, spi_stall_en=1, sq_stall_en=1, reg_at_hwm=2, hiwater=1, util_timer=1,
      mode=int(tracing), **trace_ctrl)

  def sqtt_start(self, buf0s:list[HCQBuffer]):
    self.memory_barrier()
    if self.dev.target[0] == 9:
      self.set_grbm()
      self.wreg(self.gc.regSQ_THREAD_TRACE_MASK, simd_en=0xf, cu_sel=0, sq_stall_en=1, spi_stall_en=1, reg_stall_en=1, vm_id_mask=0)
      for se in range(len(buf0s)):
        mask = (__SQTT_MISC:=1<<0) | (__SQTT_TIME:=1<<1) | (__SQTT_REG:=1<<2) | (__SQTT_WAVE_START:=1<<3) | (__SQTT_WAVE_END:=1<<6) \
             | (__SQTT_USERDATA:=1<<12) | (__SQTT_REG_CS:=1<<5) | (__SQTT_REG_CS_PRIV:=1<<15)
        if (SQTT_ITRACE_SE_MASK.value >> se) & 0b1: mask |= (__SQTTINST:=1<<10) | (__SQTT_INST_PC:=1<<11) | (__SQTT_ISSUE:=1<<13)

        with self.pred_exec(xcc_mask=1<<(se // self.dev.se_cnt)):
          self.set_grbm(se=se % self.dev.se_cnt, sh=0)
          self.wreg(self.gc.regSQ_THREAD_TRACE_TOKEN_MASK, reg_mask=0xf, token_mask=mask)
          self.wreg(self.gc.regSQ_THREAD_TRACE_TOKEN_MASK2, inst_mask=0xffffffff)
          self.wreg(self.gc.regSQ_THREAD_TRACE_BASE, addr=lo32(buf0s[se].va_addr >> 12))
          self.wreg(self.gc.regSQ_THREAD_TRACE_BASE2, addr_hi=hi32(buf0s[se].va_addr >> 12))
          self.wreg(self.gc.regSQ_THREAD_TRACE_SIZE, size=buf0s[se].size >> 12)
          self.wreg(self.gc.regSQ_THREAD_TRACE_CTRL, reset_buffer=1)
          self.wreg(self.gc.regSQ_THREAD_TRACE_MODE, mask_cs=1, autoflush_en=1, mode=1)
    else:
      self.spi_config(tracing=True)
      # One buffer for one SE, mesa does it with a single buffer and ac_sqtt_get_data_offset, but this is simpler and should work just as well
      for se in range(len(buf0s)):
        self.set_grbm(se=se, sh=0)

        buf0_lo, buf0_hi = data64_le(buf0s[se].va_addr >> 12)
        if self.dev.target >= (12,0,0):
          self.wreg(self.gc.regSQ_THREAD_TRACE_BUF0_SIZE, size=buf0s[se].size >> 12)
          self.wreg(self.gc.regSQ_THREAD_TRACE_BUF0_BASE_LO, base_lo=buf0_lo)
          self.wreg(self.gc.regSQ_THREAD_TRACE_BUF0_BASE_HI, base_hi=buf0_hi)
        else:
          self.wreg(self.gc.regSQ_THREAD_TRACE_BUF0_SIZE, base_hi=buf0_hi, size=buf0s[se].size >> 12)
          self.wreg(self.gc.regSQ_THREAD_TRACE_BUF0_BASE, base_lo=buf0_lo)
        # NOTE: SQTT can only trace instructions on one simd per se, this selects the simd in first wgp in first sa.
        # For RGP to display instruction trace it has to see it on first SE. Howerver ACE/MEC/whatever does the dispatching starting with second se,
        # and on amdgpu/non-AM it also does weird things with dispatch order inside se: around 7 times out of 10 it starts from the last cu, but
        # sometimes not, especially if the kernel has more than one wavefront which means that kernels with small global size might get unlucky and
        # be dispatched on something else and not be seen in instruction tracing tab. You can force the wavefronts of a kernel to be dispatched on the
        # CUs you want to by disabling other CUs via bits in regCOMPUTE_STATIC_THREAD_MGMT_SE<x> and trace even kernels that only have one wavefront.
        # Use SQTT_SIMD_SEL to select which SIMD to trace (0-3). Memory ops show different InstOp values (0x2x vs 0x5x) based on SIMD.
        cs_wtype = (1 << 6) if self.dev.target >= (12,0,0) else self.soc.SQ_TT_WTYPE_INCLUDE_CS_BIT
        self.wreg(self.gc.regSQ_THREAD_TRACE_MASK, wtype_include=cs_wtype, simd_sel=SQTT_SIMD_SEL.value, wgp_sel=0, sa_sel=0)
        reg_include = self.soc.SQ_TT_TOKEN_MASK_SQDEC_BIT | self.soc.SQ_TT_TOKEN_MASK_SHDEC_BIT | self.soc.SQ_TT_TOKEN_MASK_GFXUDEC_BIT | \
                      self.soc.SQ_TT_TOKEN_MASK_COMP_BIT | self.soc.SQ_TT_TOKEN_MASK_CONTEXT_BIT
        token_exclude = SQTT_TOKEN_EXCLUDE.value | ((1 << self.soc.SQ_TT_TOKEN_EXCLUDE_PERF_SHIFT) if self.dev.target < (12,0,0) else 0)

        # disable instr tracing
        if not (SQTT_ITRACE_SE_MASK.value >> se) & 0b1:
          # gfx12 doesn't have enums with all fields, so it's hardcoded, but it's the same as gfx11.
          token_exclude |= (1 << self.soc.SQ_TT_TOKEN_EXCLUDE_VMEMEXEC_SHIFT | 1 << self.soc.SQ_TT_TOKEN_EXCLUDE_ALUEXEC_SHIFT | \
                            1 << self.soc.SQ_TT_TOKEN_EXCLUDE_VALUINST_SHIFT | 1 << self.soc.SQ_TT_TOKEN_EXCLUDE_IMMEDIATE_SHIFT | \
                            1 << self.soc.SQ_TT_TOKEN_EXCLUDE_INST_SHIFT) if self.dev.target < (12,0,0) else 0x927

        self.wreg(self.gc.regSQ_THREAD_TRACE_TOKEN_MASK, reg_include=reg_include, token_exclude=token_exclude, bop_events_token_include=1,
                  **({} if self.dev.target < (12,0,0) else {'exclude_barrier_wait': 1}))
        self.sqtt_config(tracing=True)

    self.set_grbm()
    if self.dev.target[0] > 9: self.wreg(self.gc.regCOMPUTE_THREAD_TRACE_ENABLE, 1)
    self.memory_barrier()
    return self

  # Magic values from src/amd/common/ac_sqtt.c:ac_sqtt_emit_stop and src/amd/common/ac_sqtt.c:ac_sqtt_emit_wait
  def sqtt_stop(self, wptrs:HCQBuffer):
    self.memory_barrier()
    self.set_grbm()

    # Start shutting everything down
    if self.dev.target[0] == 9: self.wreg(self.gc.regSQ_THREAD_TRACE_MODE, mask_cs=1, autoflush_en=1, mode=0)
    else:
      self.wreg(self.gc.regCOMPUTE_THREAD_TRACE_ENABLE, 0)
      self.pkt3(self.pm4.PACKET3_EVENT_WRITE, self.pm4.EVENT_TYPE(self.soc.THREAD_TRACE_FINISH) | self.pm4.EVENT_INDEX(0))

    # For each SE wait for finish to complete and copy regSQ_THREAD_TRACE_WPTR to know where in the buffer trace data ends
    for se in range(self.dev.se_cnt * self.dev.xccs):
      with self.pred_exec(xcc_mask=1<<(se // self.dev.se_cnt)):
        self.set_grbm(se=se % self.dev.se_cnt, sh=0)

        regstatus = self.gc.regSQ_THREAD_TRACE_STATUS.addr[0] - (self.pm4.PACKET3_SET_UCONFIG_REG_START if self.dev.target[0] == 9 else 0)
        if self.dev.target >= (10,0,0):
          self.wait_reg_mem(reg=regstatus, mask=self.gc.regSQ_THREAD_TRACE_STATUS.fields_mask('finish_pending'), op=WAIT_REG_MEM_FUNCTION_EQ, value=0)
          self.sqtt_config(tracing=False)
        self.wait_reg_mem(reg=regstatus, mask=self.gc.regSQ_THREAD_TRACE_STATUS.fields_mask('busy'), op=WAIT_REG_MEM_FUNCTION_EQ, value=0)
        self.pkt3(self.pm4.PACKET3_EVENT_WRITE, self.pm4.EVENT_TYPE(self.soc.CS_PARTIAL_FLUSH) | self.pm4.EVENT_INDEX(EVENT_INDEX_PARTIAL_FLUSH))

        # Copy WPTR to memory (src_sel = perf, dst_sel = tc_l2, wr_confirm = True)
        self.pkt3(self.pm4.PACKET3_COPY_DATA, 1 << 20 | 2 << 8 | 4, self.gc.regSQ_THREAD_TRACE_WPTR.addr[0], 0, *data64_le(wptrs.va_addr+(se*4)))

    self.set_grbm()
    if self.dev.target[0] > 9: self.spi_config(tracing=False)
    self.memory_barrier()
    return self

  def exec(self, prg:AMDProgram, args_state:CLikeArgsState, global_size:tuple[sint, ...], local_size:tuple[sint, ...]):
    self.bind_args_state(args_state)

    self.acquire_mem(gli=0, gl2=0)

    user_regs = []
    if prg.enable_private_segment_sgpr:
      assert self.dev.xccs == 1, "Only architected flat scratch is supported on multi-xcc"
      scratch_hilo = data64_le(prg.dev.scratch.va_addr)
      # sgpr word1 bit31 enables swizzle
      # sgpr word3 = 0x14 << 12 | 2 << 28 | 2 << 21 | 1 << 23
      user_regs = [scratch_hilo[0], scratch_hilo[1] | 1 << 31, 0xffffffff, 0x20c14000]

    if prg.enable_dispatch_ptr:
      dp = (dp_t:=hsa.hsa_kernel_dispatch_packet_t).from_address(int((disp_buf:=args_state.buf.offset(prg.kernargs_segment_size)).va_addr))

      self.bind_sints(*local_size, mem=disp_buf.cpu_view(), struct_t=dp_t, start_field='workgroup_size_x', fmt='H')
      self.bind_sints(*[g*l for g,l in zip(global_size, local_size)], mem=disp_buf.cpu_view(), struct_t=dp_t, start_field='grid_size_x', fmt='I')
      dp.group_segment_size, dp.private_segment_size  = prg.group_segment_size, prg.private_segment_size
      dp.kernarg_address = cast(ctypes.c_void_p, args_state.buf.va_addr)
      user_regs += [*data64_le(disp_buf.va_addr)]

    user_regs += [*data64_le(args_state.buf.va_addr)]

    if prg.dev.sqtt_enabled: self.sqtt_setup_exec(prg, global_size)

    self.wreg(self.gc.regCOMPUTE_PGM_LO, *data64_le(prg.prog_addr >> 8))
    self.wreg(self.gc.regCOMPUTE_PGM_RSRC1, prg.rsrc1, prg.rsrc2)
    self.wreg(self.gc.regCOMPUTE_PGM_RSRC3, prg.rsrc3)
    self.wreg(self.gc.regCOMPUTE_TMPRING_SIZE, prg.dev.tmpring_size)

    if prg.dev.has_scratch_base_registers:
      for xcc_id in range(self.dev.xccs):
        with self.pred_exec(xcc_mask=1<<xcc_id):
          scratch_base = prg.dev.scratch.va_addr + (prg.dev.scratch.size // self.dev.xccs * xcc_id)
          self.wreg(self.gc.regCOMPUTE_DISPATCH_SCRATCH_BASE_LO, *data64_le(scratch_base >> 8))

    if (10,0,0) <= prg.dev.target < (11,0,0): self.wreg(self.gc.mmCP_COHER_START_DELAY, 0x20)

    self.wreg(self.gc.regCOMPUTE_RESTART_X, 0, 0, 0)
    self.wreg(self.gc.regCOMPUTE_USER_DATA_0, *user_regs)
    self.wreg(self.gc.regCOMPUTE_RESOURCE_LIMITS, 0)
    self.wreg(self.gc.regCOMPUTE_START_X, 0, 0, 0, *local_size, 0, 0)

    gfx10p = {'cs_w32_en': int(prg.wave32)} if prg.dev.target >= (10,0,0) else {}
    self.pkt3(self.pm4.PACKET3_DISPATCH_DIRECT, *global_size,
              self.gc.regCOMPUTE_DISPATCH_INITIATOR.encode(**gfx10p, force_start_at_000=1, compute_shader_en=1))

    if prg.dev.sqtt_enabled: self.pkt3(self.pm4.PACKET3_EVENT_WRITE, self.pm4.EVENT_TYPE(self.soc.THREAD_TRACE_MARKER) | self.pm4.EVENT_INDEX(0))
    self.pkt3(self.pm4.PACKET3_EVENT_WRITE, self.pm4.EVENT_TYPE(self.soc.CS_PARTIAL_FLUSH) | self.pm4.EVENT_INDEX(EVENT_INDEX_PARTIAL_FLUSH))
    return self

  def wait(self, signal:AMDSignal, value:sint=0):
    self.wait_reg_mem(mem=signal.value_addr, value=value, mask=0xffffffff)
    return self

  def timestamp(self, signal:AMDSignal):
    with self.pred_exec(xcc_mask=0b1):
      self.release_mem(cache_flush=False) # ensure all prior writes are done
      self.release_mem(signal.timestamp_addr, 0, self.pm4.data_sel__mec_release_mem__send_gpu_clock_counter, self.pm4.int_sel__mec_release_mem__none)
      self.acquire_mem() # ensure timestamp is written
    return self

  def signal(self, signal:AMDSignal, value:sint=0):
    with self.pred_exec(xcc_mask=0b1):
      # NOTE: this needs an EOP buffer on the queue or it will NULL pointer
      self.release_mem(signal.value_addr, value, self.pm4.data_sel__mec_release_mem__send_32_bit_low,
                       self.pm4.int_sel__mec_release_mem__send_interrupt_after_write_confirm, cache_flush=True)

      if (dev:=signal.owner) is not None and signal.is_timeline and not dev.is_am():
        self.release_mem(dev.queue_event_mailbox_ptr, dev.queue_event.event_id, self.pm4.data_sel__mec_release_mem__send_32_bit_low,
                         self.pm4.int_sel__mec_release_mem__send_interrupt_after_write_confirm, ctxid=dev.queue_event.event_id)
    return self

  def bind(self, dev:AMDDevice):
    self.binded_device = dev
    self.hw_page = dev.allocator.alloc(len(self._q) * 4, BufferSpec(cpu_access=True, nolru=True, uncached=True))
    hw_view = self.hw_page.cpu_view().view(fmt='I')
    for i, value in enumerate(self._q): hw_view[i] = value

    self.indirect_cmd = [self.pm4.PACKET3(self.pm4.PACKET3_INDIRECT_BUFFER, 2), *data64_le(self.hw_page.va_addr),
                         len(self._q) | self.pm4.INDIRECT_BUFFER_VALID]
    self._q = hw_view
    return self

  def _submit(self, dev:AMDDevice):
    cmds = self.indirect_cmd if dev == self.binded_device else self._q
    # WORKAROUND: PACKET3_PRED_EXEC doesn't work in rings, only in IBs, create a fake IB inside a ring to work around that
    if self.dev.xccs > 1 and dev != self.binded_device:
      ib_end = ((dev.compute_queue.put_value + 5) % len(dev.compute_queue.ring)) + len(cmds)
      ib_pad = len(dev.compute_queue.ring) - (ib_end - len(cmds)) if ib_end > len(dev.compute_queue.ring) else 0
      ib_ptr = dev.compute_queue.ring.addr + ((dev.compute_queue.put_value + 5 + ib_pad) % len(dev.compute_queue.ring)) * 4
      cmds = [self.pm4.PACKET3(self.pm4.PACKET3_INDIRECT_BUFFER, 2), *data64_le(ib_ptr), len(cmds) | self.pm4.INDIRECT_BUFFER_VALID,
              self.pm4.PACKET3(self.pm4.PACKET3_NOP, ib_pad + len(cmds) - 1), *((0,) * ib_pad), *cmds]

    for i, value in enumerate(cmds): dev.compute_queue.ring[(dev.compute_queue.put_value + i) % len(dev.compute_queue.ring)] = value

    dev.compute_queue.put_value += len(cmds)
    dev.compute_queue.signal_doorbell(dev)

class AMDComputeAQLQueue(AMDComputeQueue):
  def exec(self, prg:AMDProgram, args_state:CLikeArgsState, global_size:tuple[sint, ...], local_size:tuple[sint, ...]):
    self.bind_args_state(args_state)
    if prg.dev.sqtt_enabled: self.sqtt_setup_exec(prg, global_size)
    self._q.append(pkt:=hsa.hsa_kernel_dispatch_packet_t(header=AQL_HDR | (hsa.HSA_PACKET_TYPE_KERNEL_DISPATCH << hsa.HSA_PACKET_HEADER_TYPE),
      setup=3<<hsa.HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS, private_segment_size=prg.private_segment_size,
      group_segment_size=prg.group_segment_size, kernel_object=prg.aql_prog_addr, kernarg_address=args_state.buf.va_addr))
    self.bind_sints_to_mem(*local_size, mem=(pkt_view:=MMIOInterface(addr=ctypes.addressof(pkt), nbytes=ctypes.sizeof(pkt))), fmt='H', offset=4)
    self.bind_sints_to_mem(*[l * g for l,g in zip(local_size, global_size)], mem=pkt_view, fmt='I', offset=12)
    return self

  def bind(self, dev:AMDDevice): pass # not supported
  def _submit(self, dev:AMDDevice):
    pm4_batch:list[int] = []
    aql_bytes = bytes()

    def flush_pm4_batch():
      nonlocal pm4_batch
      if not pm4_batch: return bytes()
      dev.pm4_ibs.cpu_view().view(off:=dev.pm4_ib_alloc.alloc(len(pm4_batch) * 4, 16), fmt='I')[:len(pm4_batch)] = array.array('I', pm4_batch)
      pkt = [AQL_HDR | (hsa.HSA_PACKET_TYPE_VENDOR_SPECIFIC << hsa.HSA_PACKET_HEADER_TYPE) | (1 << 16),
        self.pm4.PACKET3(self.pm4.PACKET3_INDIRECT_BUFFER, 2), *data64_le(dev.pm4_ibs.va_addr+off), len(pm4_batch)|self.pm4.INDIRECT_BUFFER_VALID, 10]
      pm4_batch.clear()
      return bytes(array.array('I', pkt + [0] * 10))

    for cmd in self._q:
      if isinstance(cmd, hsa.hsa_kernel_dispatch_packet_t): aql_bytes += flush_pm4_batch() + bytes(cmd)
      else: pm4_batch.append(cmd)
    aql_bytes += flush_pm4_batch()

    assert len(aql_bytes) < dev.compute_queue.ring.nbytes, "submit is too large for the queue"
    cp_bytes = min(len(aql_bytes), (dev.compute_queue.ring.nbytes - (dev.compute_queue.put_value * 64) % dev.compute_queue.ring.nbytes))
    dev.compute_queue.ring.view(offset=(dev.compute_queue.put_value * 64) % dev.compute_queue.ring.nbytes, fmt='B')[:cp_bytes] = aql_bytes[:cp_bytes]
    if (tail_bytes:=(len(aql_bytes) - cp_bytes)) > 0: dev.compute_queue.ring.view(offset=0, fmt='B')[:tail_bytes] = aql_bytes[cp_bytes:]
    dev.compute_queue.put_value += len(aql_bytes) // 64
    dev.compute_queue.signal_doorbell(dev, doorbell_value=dev.compute_queue.put_value-1)

class AMDCopyQueue(HWQueue):
  def __init__(self, dev, max_copy_size=0x40000000, queue_idx=0):
    self.dev, self.sdma, self.internal_cmd_sizes, self.max_copy_size, self.queue_idx = dev, dev.sdma, [], max_copy_size, queue_idx
    super().__init__()

  def q(self, *arr):
    super().q(*arr)
    self.internal_cmd_sizes.append(len(arr))

  def copy(self, dest:sint, src:sint, copy_size:int):
    copied, copy_commands = 0, (copy_size + self.max_copy_size - 1) // self.max_copy_size

    for _ in range(copy_commands):
      step_copy_size = min(copy_size - copied, self.max_copy_size)

      self.q(self.sdma.SDMA_OP_COPY | self.sdma.SDMA_PKT_COPY_LINEAR_HEADER_SUB_OP(self.sdma.SDMA_SUBOP_COPY_LINEAR),
        self.sdma.SDMA_PKT_COPY_LINEAR_COUNT_COUNT(step_copy_size - 1), 0, *data64_le(src + copied), *data64_le(dest + copied))

      copied += step_copy_size
    return self

  def signal(self, signal:AMDSignal, value:sint=0):
    fence_flags = self.sdma.SDMA_PKT_FENCE_HEADER_MTYPE(3) if self.dev.target >= (10,0,0) else 0
    self.q(self.sdma.SDMA_OP_FENCE | fence_flags, *data64_le(signal.value_addr), value)

    if (dev:=signal.owner) is not None and signal.is_timeline and not dev.is_am():
      self.q(self.sdma.SDMA_OP_FENCE | fence_flags, *data64_le(dev.queue_event_mailbox_ptr), dev.queue_event.event_id)
      self.q(self.sdma.SDMA_OP_TRAP, self.sdma.SDMA_PKT_TRAP_INT_CONTEXT_INT_CONTEXT(dev.queue_event.event_id))

    return self

  def wait(self, signal:AMDSignal, value:sint=0):
    self.q(self.sdma.SDMA_OP_POLL_REGMEM | self.sdma.SDMA_PKT_POLL_REGMEM_HEADER_FUNC(WAIT_REG_MEM_FUNCTION_GEQ) | \
           self.sdma.SDMA_PKT_POLL_REGMEM_HEADER_MEM_POLL(1), *data64_le(signal.value_addr), value, 0xffffffff,
           self.sdma.SDMA_PKT_POLL_REGMEM_DW5_INTERVAL(0x04) | self.sdma.SDMA_PKT_POLL_REGMEM_DW5_RETRY_COUNT(0xfff))
    return self

  def timestamp(self, signal:AMDSignal):
    self.q(self.sdma.SDMA_OP_TIMESTAMP | self.sdma.SDMA_PKT_TIMESTAMP_GET_HEADER_SUB_OP(self.sdma.SDMA_SUBOP_TIMESTAMP_GET_GLOBAL),
           *data64_le(signal.timestamp_addr))
    return self

  def bind(self, dev:AMDDevice):
    if not getenv("AMD_SDMA_BIND", 0) or not dev.is_am(): return

    self.binded_device = dev
    self.hw_page = dev.allocator.alloc((qsz:=round_up(len(self._q), 8)) * 4, BufferSpec(cpu_access=True, nolru=True, uncached=True))
    hw_view = self.hw_page.cpu_view().view(fmt='I')
    for i in range(qsz): hw_view[i] = self._q[i] if i < len(self._q) else 0

    self.indirect_cmd = [self.sdma.SDMA_OP_INDIRECT | self.sdma.SDMA_PKT_INDIRECT_HEADER_VMID(0), *data64_le(self.hw_page.va_addr), qsz,
                         *data64_le(0)]
    self._q, self.cmd_sizes = hw_view, [len(self.indirect_cmd)]

  def _submit(self, dev:AMDDevice):
    sdma_queue = dev.sdma_queue(self.queue_idx)
    if self.binded_device == dev:
      # An IB packet must end on a 8 DW boundary.
      add = (8 - (((sdma_queue.put_value % 32) // 4) + len(self.indirect_cmd) % 8)) % 8
      cmds, cmd_sizes = ([0] * add) + self.indirect_cmd, [len(self.indirect_cmd) + add]

      if len(cmds) * 4 >= (sdma_queue.ring.nbytes - sdma_queue.put_value % sdma_queue.ring.nbytes):
        cmds, cmd_sizes = [0, 0] + self.indirect_cmd, [8]
    else: cmds, cmd_sizes = self._q, self.internal_cmd_sizes

    tail_blit_dword = 0
    for cmdsz in cmd_sizes:
      if (tail_blit_dword + cmdsz) * 4 >= sdma_queue.ring.nbytes - sdma_queue.put_value % sdma_queue.ring.nbytes: break
      tail_blit_dword += cmdsz

    # Force align of submits to hit our usb layer write cache.
    if (rem_packet_cnt := len(cmds) - tail_blit_dword) > 0 and dev.is_usb(): tail_blit_dword = 0

    # USB devices run in single-step mode, so they can't overrun the queue.
    total_bytes = (tail_blit_dword * 4 if rem_packet_cnt == 0 else -sdma_queue.put_value % sdma_queue.ring.nbytes) + rem_packet_cnt * 4
    assert total_bytes < sdma_queue.ring.nbytes, "SDMA queue overrun"
    while not dev.is_usb() and sdma_queue.put_value + total_bytes - sdma_queue.read_ptr[0] > sdma_queue.ring.nbytes: pass

    start_idx = (sdma_queue.put_value % sdma_queue.ring.nbytes) // 4
    sdma_queue.ring[start_idx : start_idx + tail_blit_dword] = array.array('I', cmds[:tail_blit_dword])
    sdma_queue.put_value += tail_blit_dword * 4

    if (rem_packet_cnt := len(cmds) - tail_blit_dword) > 0:
      zero_fill = sdma_queue.ring.nbytes - sdma_queue.put_value % sdma_queue.ring.nbytes
      sdma_queue.ring.view(sdma_queue.put_value % sdma_queue.ring.nbytes, zero_fill, fmt='B')[:] = bytes(zero_fill)
      sdma_queue.put_value += zero_fill

      sdma_queue.ring[0:rem_packet_cnt] = array.array('I', cmds[tail_blit_dword:])
      sdma_queue.put_value += rem_packet_cnt * 4

    sdma_queue.signal_doorbell(dev)

class AMDProgram(HCQProgram):
  def __init__(self, dev:AMDDevice, name:str, lib:bytes, **kwargs):
    # TODO; this API needs the type signature of the function and global_size/local_size
    self.dev, self.name, self.lib = dev, name, lib

    image, sections, relocs = elf_loader(self.lib)

    rodata_entry = next((sh.header.sh_addr for sh in sections if sh.name == ".rodata"), -1)
    assert rodata_entry >= 0, ".rodata section not found"

    for apply_image_offset, rel_sym_offset, typ, addent in relocs:
      if typ == 5: image[apply_image_offset:apply_image_offset+8] = struct.pack('<q', rel_sym_offset - apply_image_offset + addent) # R_AMDGPU_REL64
      else: raise RuntimeError(f"unknown AMD reloc {typ}")

    self.lib_gpu = self.dev.allocator.alloc(round_up(image.nbytes, 0x1000), buf_spec:=BufferSpec(nolru=True))
    self.dev.allocator._copyin(self.lib_gpu, image)
    self.dev.synchronize()

    desc_sz = ctypes.sizeof(amdgpu_kd.llvm_amdhsa_kernel_descriptor_t)
    desc = amdgpu_kd.llvm_amdhsa_kernel_descriptor_t.from_buffer_copy(bytes(image[rodata_entry:rodata_entry+desc_sz]))
    self.group_segment_size = desc.group_segment_fixed_size
    self.private_segment_size = desc.private_segment_fixed_size
    self.kernargs_segment_size = desc.kernarg_size
    lds_size = ((self.group_segment_size + 511) // 512) & 0x1FF
    if lds_size > (self.dev.iface.props['lds_size_in_kb'] * 1024) // 512: raise RuntimeError("Too many resources requested: group_segment_size")

    # Ensure scratch size
    self.dev._ensure_has_local_memory(self.private_segment_size)

    self.wave32: bool = desc.kernel_code_properties & 0x400 == 0x400

    # Set rsrc1.priv=1 on gfx11 to workaround cwsr.
    self.rsrc1: int = desc.compute_pgm_rsrc1 | ((1 << 20) if (11,0,0) <= self.dev.target < (12,0,0) else 0)
    self.rsrc2: int = desc.compute_pgm_rsrc2 | (lds_size << 15)
    self.rsrc3: int = desc.compute_pgm_rsrc3
    self.aql_prog_addr: int = self.lib_gpu.va_addr + rodata_entry
    self.prog_addr: int = self.lib_gpu.va_addr + rodata_entry + desc.kernel_code_entry_byte_offset
    # Some programs use hsa_kernel_dispatch_packet_t to read workgroup sizes during execution.
    # The packet is represented as a pointer and set up in SGPRs. Space for the packet is allocated as part of the kernel arguments.
    self.enable_dispatch_ptr: int = desc.kernel_code_properties & hsa.AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_PTR
    self.enable_private_segment_sgpr: int = desc.kernel_code_properties & hsa.AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER
    additional_alloc_sz = ctypes.sizeof(hsa.hsa_kernel_dispatch_packet_t) if self.enable_dispatch_ptr else 0

    if dev.sqtt_enabled: self.libhash: tuple[int, int] = struct.unpack('<Q', hashlib.md5(self.lib).digest()[:8])*2

    super().__init__(CLikeArgsState, self.dev, self.name, kernargs_alloc_size=self.kernargs_segment_size+additional_alloc_sz, lib=self.lib,
                     base=self.lib_gpu.va_addr)
    weakref.finalize(self, self._fini, self.dev, self.lib_gpu, buf_spec)

  def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int|None, ...]=(), wait=False):
    if self.dev.sqtt_enabled: cast(AMDComputeQueue, self.dev.hw_compute_queue_t()).sqtt_start(self.dev.sqtt_buffers).submit(self.dev)
    res = super().__call__(*bufs, global_size=global_size, local_size=local_size, vals=vals, wait=wait)
    if self.dev.pmc_enabled:
      cast(AMDComputeQueue, self.dev.hw_compute_queue_t()).pmc_read(self.dev.pmc_buffer, self.dev.pmc_sched) \
                                                          .signal(self.dev.timeline_signal, self.dev.next_timeline()).submit(self.dev)
      self.dev.allocator._copyout(pmc_buf:=memoryview(bytearray(self.dev.pmc_buffer.size)), self.dev.pmc_buffer)
      Compiled.profile_events += [ProfilePMCEvent(self.dev.device, self.dev.prof_prg_counter, self.dev.pmc_sched, bytes(pmc_buf),
                                                  self.dev.prof_exec_counter)]
    if self.dev.sqtt_enabled:
      cast(AMDComputeQueue, self.dev.hw_compute_queue_t()).sqtt_stop(self.dev.sqtt_wptrs) \
                                                          .signal(self.dev.timeline_signal, self.dev.next_timeline()).submit(self.dev)
      self.dev.synchronize()

      for se, buf in enumerate(self.dev.sqtt_buffers):
        wptr = (self.dev.sqtt_wptrs.cpu_view().view(fmt='I')[se] & 0x1FFFFFFF) * 32
        if self.dev.target[:2] == (11, 0): wptr -= ((buf.va_addr // 32) & 0x1FFFFFFF) * 32

        if DEBUG >= 5: print(f'\t{self.dev.device}: SE {se} blob size {wptr:#x}')
        assert wptr >= 0 and wptr <= buf.size, f"{wptr} > {buf.size}, should never happen"

        # When sqtt buffer overflows, wptr stops at the last dword
        if wptr >= buf.size - 32:
          print(colored(f"{self.dev.device}: Warning: SQTT buffer is full (SE {se})! Increase SQTT buffer with SQTT_BUFFER_SIZE=X (in MB)", "yellow"))

        self.dev.allocator._copyout(sqtt_mv:=memoryview(bytearray(wptr)), buf)
        resbuf = (struct.pack('<Q', 0x11 | (4 << 13) | (0xf << 16) | (se << 24)) + bytes(sqtt_mv)) if self.dev.target[0] == 9 else bytes(sqtt_mv)
        Compiled.profile_events += [ProfileSQTTEvent(self.dev.device, self.dev.prof_prg_counter, se, resbuf,
                                                     bool((SQTT_ITRACE_SE_MASK.value >> se) & 1), self.dev.prof_exec_counter)]
    return res

class AMDAllocator(HCQAllocator['AMDDevice']):
  def __init__(self, dev:AMDDevice):
    super().__init__(dev, copy_bufs=getattr(dev.iface, 'copy_bufs', None), max_copyout_size=0x1000 if dev.is_usb() else None,
                     supports_copy_from_disk=not dev.is_usb() and dev.has_sdma_queue, supports_transfer=dev.has_sdma_queue)
    if hasattr(dev.iface, "as_dmaref"): self._as_dmaref = dev.iface.as_dmaref

  def _alloc(self, size:int, options:BufferSpec) -> HCQBuffer:
    return self.dev.iface.alloc(size, host=options.host, uncached=options.uncached, cpu_access=options.cpu_access or not self.dev.has_sdma_queue)

  def _do_free(self, opaque, options:BufferSpec): self.dev.iface.free(opaque)

  def _map(self, buf:HCQBuffer): return self.dev.iface.map(buf._base if buf._base is not None else buf)

@dataclass
class AMDQueueDesc:
  ring: MMIOInterface
  read_ptr: MMIOInterface
  write_ptr: MMIOInterface
  doorbell: MMIOInterface
  put_value: int = 0
  params: tuple|None = None  # setup_ring params for recovery

  def signal_doorbell(self, dev, doorbell_value:int|None=None):
    try:
      self.write_ptr[0] = self.put_value

      # Ensure all prior writes are visible to the GPU.
      System.memory_barrier()

      # Flush hdp if queue is in dev mem.
      if dev.is_am() and not dev.is_usb(): dev.iface.dev_impl.gmc.flush_hdp()
      self.doorbell[0] = self.put_value if doorbell_value is None else doorbell_value
    except Exception as e:
      dev.error_state = e
      raise

class KFDIface:
  kfd:FileIOInterface|None = None
  event_page:HCQBuffer|None = None
  gpus:list[FileIOInterface] = []

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
      KFDIface.gpus = hcq_filter_visible_devices(sorted(gpus, key=lambda x: int(x.split('/')[-1])))

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
    self.queue_event_arr = (kfd.struct_kfd_event_data)(event_id=self.dev.queue_event.event_id)
    self.queue_event_arr_ptr = ctypes.addressof(self.queue_event_arr)

    # OS events to collect memory and hardware faults
    self.mem_fault_event = kfd.AMDKFD_IOC_CREATE_EVENT(KFDIface.kfd, event_type=kfd.KFD_IOC_EVENT_MEMORY)
    self.hw_fault_event = kfd.AMDKFD_IOC_CREATE_EVENT(KFDIface.kfd, event_type=kfd.KFD_IOC_EVENT_HW_EXCEPTION)

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
    if len(mem.mapped_devs) > 0:
      gpus = (ctypes.c_int32 * len(mem.mapped_devs))(*[x.iface.gpu_id for x in mem.mapped_devs])
      stm = kfd.AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU(self.kfd, handle=mem.meta.handle, device_ids_array_ptr=ctypes.addressof(gpus), n_devices=len(gpus))
      assert stm.n_success == len(gpus)
    if mem.va_addr: FileIOInterface.munmap(mem.va_addr, mem.size)
    kfd.AMDKFD_IOC_FREE_MEMORY_OF_GPU(self.kfd, handle=mem.meta.handle)

  def as_dmaref(self, mem:HCQBuffer) -> DMAFdRef:
    base = mem._base if mem._base is not None else mem
    dmaref = DMAFdRef(kfd.AMDKFD_IOC_EXPORT_DMABUF(KFDIface.kfd, handle=base.meta.handle, flags=0).dmabuf_fd, int(mem.va_addr-base.va_addr), mem.size)
    weakref.finalize(dmaref, os.close, dmaref.fd)
    return dmaref

  def map(self, mem):
    if mem.owner is not None and mem.owner._is_cpu(): return self.alloc(mem.size, host=True, cpu_addr=mem.va_addr)

    c_gpus = (ctypes.c_int32 * 1)(self.gpu_id)
    stm = kfd.AMDKFD_IOC_MAP_MEMORY_TO_GPU(self.kfd, handle=mem.meta.handle, device_ids_array_ptr=ctypes.addressof(c_gpus), n_devices=1)
    assert stm.n_success == 1

  def create_queue(self, queue_type, ring, gart, rptr, wptr, eop_buffer=None, cwsr_buffer=None, ctl_stack_size=0, ctx_save_restore_size=0,
                   xcc_id=0, idx=0):
    queue = kfd.AMDKFD_IOC_CREATE_QUEUE(KFDIface.kfd, ring_base_address=ring.va_addr, ring_size=ring.size, gpu_id=self.gpu_id,
      queue_type=queue_type, queue_percentage=kfd.KFD_MAX_QUEUE_PERCENTAGE|(xcc_id<<8), queue_priority=getenv("AMD_KFD_QUEUE_PRIORITY", 7),
      eop_buffer_address=eop_buffer.va_addr if eop_buffer else 0, eop_buffer_size=eop_buffer.size if eop_buffer else 0, ctl_stack_size=ctl_stack_size,
      ctx_save_restore_address=cwsr_buffer.va_addr if cwsr_buffer else 0, ctx_save_restore_size=ctx_save_restore_size,
      write_pointer_address=gart.va_addr+wptr, read_pointer_address=gart.va_addr+rptr+8*xcc_id)

    if not hasattr(self, 'doorbells'):
      self.doorbells_base = queue.doorbell_offset & (~0x1fff) # doorbell is two pages
      self.doorbells = cast(FileIOInterface, KFDIface.kfd).mmap(0, 0x2000, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED, self.doorbells_base)

    return AMDQueueDesc(ring=MMIOInterface(ring.va_addr, ring.size, fmt='I'), read_ptr=MMIOInterface(queue.read_pointer_address, 8, fmt='Q'),
                        write_ptr=MMIOInterface(queue.write_pointer_address, 8, fmt='Q'),
                        doorbell=MMIOInterface(self.doorbells + queue.doorbell_offset - self.doorbells_base, 8, fmt='Q'))

  def sleep(self, tm:int) -> bool:
    kfd.AMDKFD_IOC_WAIT_EVENTS(KFDIface.kfd, events_ptr=self.queue_event_arr_ptr, num_events=1, wait_for_all=1, timeout=tm)
    return False

  def on_device_hang(self):
    def _collect_str(st): return ' '.join(f'{k[0]}={getattr(st, k[0])}' for k in st._real_fields_)

    report = []
    for evnt in [self.mem_fault_event, self.hw_fault_event]:
      ev = (kfd.struct_kfd_event_data)(event_id=evnt.event_id)
      kfd.AMDKFD_IOC_WAIT_EVENTS(KFDIface.kfd, events_ptr=ctypes.addressof(ev), num_events=1, wait_for_all=1)
      if evnt == self.mem_fault_event and ev.memory_exception_data.gpu_id:
        report += [f"MMU fault: 0x{ev.memory_exception_data.va:X} | {_collect_str(ev.memory_exception_data.failure)}"]
      if evnt == self.hw_fault_event and ev.hw_exception_data.gpu_id: report += [f"HW fault: {_collect_str(ev.hw_exception_data)}"]

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
  gpus:ClassVar[list[str]] = []

  def __init__(self, dev, dev_id):
    super().__init__(dev, dev_id, vendor=0x1002, devices=[(0xffff, [0x74a1, 0x744c, 0x7480, 0x7550, 0x7590, 0x75a0])], bars=[0, 2, 5], vram_bar=0,
      va_start=AMMemoryManager.va_allocator.base, va_size=AMMemoryManager.va_allocator.size)
    self._setup_adev(self.pci_dev)
    self.pci_dev.write_config(pci.PCI_COMMAND, self.pci_dev.read_config(pci.PCI_COMMAND, 2) | pci.PCI_COMMAND_MASTER, 2)

  def require_profile_mode(self): return True
  def is_wgp_active(self, xcc, se, sa, wgp) -> bool: return True # TODO: account for WGP disablement on some asics.

  def _setup_adev(self, pci_dev:PCIDevice, dma_regions:list[tuple[int, MMIOInterface]]|None=None):
    self.dev_impl:AMDev = AMDev(pci_dev, dma_regions)
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
      pv, doorbell_index = self.dev_impl.sdma.setup_ring(*(rcvr_params:=(ring.va_addr, ring.size, gart.va_addr+rptr, gart.va_addr+wptr, idx)))
    else:
      pv, doorbell_index = self.dev_impl.gfx.setup_ring(*(rcvr_params:=(ring.va_addr, ring.size, gart.va_addr+rptr, gart.va_addr+wptr,
        eop_buffer.va_addr, eop_buffer.size, is_aql:=(queue_type==kfd.KFD_IOC_QUEUE_TYPE_COMPUTE_AQL), is_aql)))

    return AMDQueueDesc(ring=ring.cpu_view().view(fmt='I'), doorbell=self.dev_impl.doorbell64.view(doorbell_index * 8, 8, fmt='Q'), put_value=pv,
      read_ptr=gart.cpu_view().view(offset=rptr, size=8, fmt='Q'), write_ptr=gart.cpu_view().view(offset=wptr, size=8, fmt='Q'), params=rcvr_params)

  def sleep(self, timeout) -> bool:
    if hasattr(self.pci_dev, 'irq_poller') and self.pci_dev.irq_poller is not None and (events_cnt:=len(self.pci_dev.irq_poller.poll(timeout))):
      self.pci_dev.irq_fd.read(8 * events_cnt)
    self.dev_impl.ih.interrupt_handler()
    return self.dev_impl.is_err_state

  def on_device_hang(self):
    devs:list[AMDDevice] = [d for pg in HCQCompiled.peer_groups.values() for d in pg if isinstance(d, AMDDevice) and d.is_am()]
    for d in devs: d.iface.dev_impl.ih.interrupt_handler()
    faults = [f for d in devs if (f:=d.iface.dev_impl.gmc.check_fault())]
    for d in devs:
      if d.iface.dev_impl.recover():
        d.compute_queue.put_value, _ = d.iface.dev_impl.gfx.setup_ring(*d.compute_queue.params)
        d.compute_queue.read_ptr[0] = d.compute_queue.write_ptr[0] = d.compute_queue.put_value
        d.timeline_signal.value = d.timeline_value - 1
        d.error_state = None
    raise RuntimeError(f"Device hang detected: {'; '.join(faults)}" if faults else "Device hang detected")

  def device_fini(self): self.dev_impl.fini()

class USBIface(PCIIface):
  def __init__(self, dev, dev_id): # pylint: disable=super-init-not-called
    self.dev, self.pci_dev = dev, USBPCIDevice(dev.__class__.__name__[:2], f"usb:{dev_id}", bars=[0, 2, 5])
    self._setup_adev(self.pci_dev, dma_regions=[(0x200000, self.pci_dev.dma_view(0xf000, 0x80000))])
    self.pci_dev.usb._pci_cacheable += [(self.pci_dev.bar_info[2].addr, self.pci_dev.bar_info[2].size)] # doorbell region is cacheable

    # special regions
    self.copy_bufs = [self._dma_region(ctrl_addr=0xf000, sys_addr=0x200000, size=0x80000)]
    self.sys_buf, self.sys_next_off = self._dma_region(ctrl_addr=0xa000, sys_addr=0x820000, size=0x1000), 0x800

  def _dma_region(self, ctrl_addr, sys_addr, size):
    region = self.dev_impl.mm.map_range(vaddr:=self.dev_impl.mm.alloc_vaddr(size=size), size, [(sys_addr, size)], aspace=AddrSpace.SYS, uncached=True)
    return HCQBuffer(vaddr, size, meta=PCIAllocationMeta(region, has_cpu_mapping=False), view=self.pci_dev.dma_view(ctrl_addr, size), owner=self.dev)

  def alloc(self, size:int, host=False, uncached=False, cpu_access=False, contiguous=False, **kwargs) -> HCQBuffer:
    if (host or (uncached and cpu_access)) and self.sys_next_off + size < self.sys_buf.size:
      self.sys_next_off += size
      return self.sys_buf.offset(self.sys_next_off - size, size)

    mapping = self.dev_impl.mm.valloc(size:=round_up(size, 4 << 10), uncached=uncached, contiguous=cpu_access)
    barview = self.pci_dev.map_bar(bar=0, off=mapping.paddrs[0][0], size=mapping.size) if cpu_access else None
    return HCQBuffer(mapping.va_addr, size, meta=PCIAllocationMeta(mapping, has_cpu_mapping=False), view=barview, owner=self.dev)

  def create_queue(self, queue_type, ring, gart, rptr, wptr, eop_buffer=None, cwsr_buffer=None, ctl_stack_size=0, ctx_save_restore_size=0,
                   xcc_id=0, idx=0):
    if queue_type == kfd.KFD_IOC_QUEUE_TYPE_COMPUTE: self.pci_dev.usb._pci_cacheable += [(ring.cpu_view().addr, ring.size)]
    return super().create_queue(queue_type, ring, gart, rptr, wptr, eop_buffer, cwsr_buffer, ctl_stack_size, ctx_save_restore_size, xcc_id, idx)

  def sleep(self, timeout) -> bool: return False

class AMDDevice(HCQCompiled):
  def is_am(self) -> bool: return isinstance(self.iface, (PCIIface, USBIface))
  def is_usb(self) -> bool: return isinstance(self.iface, USBIface)

  def __init__(self, device:str=""):
    self.device_id = int(device.split(":")[1]) if ":" in device else 0
    self.iface = self._select_iface(KFDIface, PCIIface, USBIface)
    self.target:tuple[int, ...] = ((trgt:=self.iface.props['gfx_target_version']) // 10000, (trgt // 100) % 100, trgt % 100)
    self.arch = "gfx%d%x%x" % self.target
    if self.target < (9,4,2) or self.target >= (13,0,0): raise RuntimeError(f"Unsupported arch: {self.arch}")
    if DEBUG >= 1: print(f"AMDDevice: opening {self.device_id} with target {self.target} arch {self.arch}")

    self.xccs = self.iface.props.get('num_xcc', 1)
    self.se_cnt = self.iface.props['array_count'] // self.iface.props['simd_arrays_per_engine'] // self.xccs
    self.cu_cnt = self.iface.props['simd_count'] // self.iface.props['simd_per_cu'] // self.xccs
    self.waves_per_cu = self.iface.props['max_waves_per_simd'] * self.iface.props['simd_per_cu']
    self.wave_cnt = (self.cu_cnt * self.waves_per_cu) if self.target >= (10,1,0) else min(self.cu_cnt * 40, self.se_cnt * self.xccs * 512)
    # this is what llvm refers to as "architected flat scratch"
    self.has_scratch_base_registers = self.target >= (11,0,0) or self.target in {(9,4,2), (9,5,0)}

    # https://gitlab.freedesktop.org/agd5f/linux/-/blob/a1fc9f584c4aaf8bc1ebfa459fc57a3f26a290d8/drivers/gpu/drm/amd/amdkfd/kfd_queue.c#L391
    sgrp_size_per_cu, lds_size_per_cu, hwreg_size_per_cu = 0x4000, 0x10000, 0x1000
    if self.target[:2] == (9,5): lds_size_per_cu = self.iface.props["lds_size_in_kb"] << 10
    vgpr_size_per_cu = 0x60000 if self.target in {(11,0,0), (11,0,1), (11,5,1), (12,0,0), (12,0,1)} else \
                       0x80000 if (self.target[:2]) in {(9,4), (9,5)} or self.target in {(9,0,8), (9,0,10)} else 0x40000
    wg_data_size = round_up((vgpr_size_per_cu + sgrp_size_per_cu + lds_size_per_cu + hwreg_size_per_cu) * self.cu_cnt, mmap.PAGESIZE)
    ctl_stack_size = round_up((12 if self.target >= (10,1,0) else 8) * self.wave_cnt + 8 + 40, mmap.PAGESIZE)
    if self.target[0] == 10: ctl_stack_size = min(ctl_stack_size, 0x7000)
    debug_memory_size = round_up(self.wave_cnt * 32, 64)

    self.ip_off = import_ip_offsets(self.target)
    self.soc = import_soc(self.target)
    self.pm4 = importlib.import_module(f"tinygrad.runtime.autogen.am.pm4_{'nv' if self.target[0] >= 10 else 'soc15'}")
    self.sdma = import_module('sdma', min(self.iface.ip_versions[am.SDMA0_HWIP], (6, 0, 0)))
    self.gc = AMDIP('gc', self.iface.ip_versions[am.GC_HWIP],
                    bases={i: tuple(getattr(self.ip_off, f'GC_BASE__INST{i}_SEG{s}', 0) for s in range(6)) for i in range(6)})

    self.nbio = AMDIP('nbio' if self.target[0] < 12 else 'nbif', self.iface.ip_versions[am.NBIF_HWIP],
                      bases={i: tuple(getattr(self.ip_off, f'NBIO_BASE__INST{i}_SEG{s}', 0) for s in range(9)) for i in range(6)})

    self.is_aql = getenv("AMD_AQL", int(self.xccs > 1))
    if self.is_aql:
      self.pm4_ibs = self.iface.alloc(0x2000 if self.is_usb() else (16 << 20), uncached=True, cpu_access=True)
      self.pm4_ib_alloc = BumpAllocator(self.pm4_ibs.size, wrap=True)

    self.compute_queue = self.create_queue(kfd.KFD_IOC_QUEUE_TYPE_COMPUTE_AQL if self.is_aql else kfd.KFD_IOC_QUEUE_TYPE_COMPUTE,
      0x2000 if self.is_usb() else (16 << 20), eop_buffer_size=0x1000,
      ctx_save_restore_size=0 if self.is_am() else wg_data_size + ctl_stack_size, ctl_stack_size=ctl_stack_size, debug_memory_size=debug_memory_size)

    self.max_copy_size = 0x40000000 if self.iface.ip_versions[am.SDMA0_HWIP][0] >= 5 else 0x400000
    self.has_sdma_queue = self.sdma_queue(0) is not None

    compilers = CompilerSet([CompilerPair(functools.partial(AMDHIPRenderer, self.arch), None),
                             CompilerPair(functools.partial(AMDLLVMRenderer, self.arch), None, AMD_LLVM),
                             CompilerPair(functools.partial(AMDHIPCCRenderer, self.arch), None)], ctrl_var=AMD_CC)

    super().__init__(device, AMDAllocator(self), compilers, functools.partial(AMDProgram, self), AMDSignal,
                     functools.partial(AMDComputeAQLQueue if self.is_aql else AMDComputeQueue, self),
                     functools.partial(AMDCopyQueue, self, max_copy_size=self.max_copy_size) if self.has_sdma_queue else None,
                     kernargs_size=(8 << 10) if self.is_usb() else (16 << 20), sigalloc_size=0x100 if self.is_usb() else 0x1000)

    # Scratch setup
    self.max_private_segment_size = 0
    self._ensure_has_local_memory(128) # set default scratch size to 128 bytes per thread

    self.pmc_enabled:bool = PROFILE > 0 and PMC > 0
    if self.pmc_enabled:
      if self.target[0] not in {9, 11, 12}: raise RuntimeError(f'PMC are not supported on gc:{self.target}')
      self.iface.require_profile_mode()

      self.pmc_sched:list[PMCSample] = []
      self.pmc_counters = import_pmc(self.target)

      # validate counters: SQ for SIMD busy/instruction counts, LDS stats, GRBM for GPU cycles, L2 cache hits/misses
      l2, lds = ("TCC", "SQ") if self.target[0] == 9 else ("GL2C", "SQC")
      pmc_default = f"SQ_BUSY_CYCLES,SQ_INSTS_VALU,SQ_INSTS_SALU,{lds}_LDS_IDX_ACTIVE,{lds}_LDS_BANK_CONFLICT,GRBM_GUI_ACTIVE,{l2}_HIT,{l2}_MISS"
      for k in (PMC_COUNTERS:=getenv("PMC_COUNTERS", pmc_default).split(",")):
        if k not in self.pmc_counters: raise RuntimeError(f"PMC counter {k} is not supported. Available: {','.join(self.pmc_counters.keys())}")

      cast(AMDComputeQueue, self.hw_compute_queue_t()).pmc_start([(k, *self.pmc_counters[k]) for k in PMC_COUNTERS]).submit(self)
      self.pmc_buffer = self.allocator.alloc(self.pmc_sched[-1].off + self.pmc_sched[-1].size, BufferSpec(nolru=True, uncached=True))
      self.allocator._copyin(self.pmc_buffer, memoryview(bytearray(self.pmc_buffer.size))) # zero pmc buffers, some counters have only lo part.

    # SQTT is disabled by default because of runtime overhead and big file sizes (~200mb to Tensor.full() two 4096x4096 tensors and matmul them)
    self.sqtt_enabled:bool = PROFILE > 0 and SQTT > 0
    if self.sqtt_enabled:
      if self.target[0] not in {9, 11, 12}: raise RuntimeError(f'SQ Thread Tracing is not supported on gc:{self.target}')
      self.iface.require_profile_mode()

      SQTT_BUFFER_SIZE = getenv("SQTT_BUFFER_SIZE", 256) # in mb, per shader engine
      self.sqtt_buffers = [self.allocator.alloc(SQTT_BUFFER_SIZE<<20, BufferSpec(nolru=True, uncached=True)) for _ in range(self.se_cnt * self.xccs)]
      self.sqtt_wptrs = self.allocator.alloc(round_up(self.se_cnt * self.xccs * 4, 0x1000), BufferSpec(cpu_access=True, nolru=True))
      self.sqtt_next_cmd_id = itertools.count(0)

  def create_queue(self, queue_type, ring_size, ctx_save_restore_size=0, eop_buffer_size=0, ctl_stack_size=0, debug_memory_size=0, idx=0):
    ring = self.iface.alloc(ring_size, uncached=True, cpu_access=True)
    gart = self.iface.alloc(0x100, uncached=True, cpu_access=True)

    if queue_type == kfd.KFD_IOC_QUEUE_TYPE_COMPUTE_AQL:
      aql_desc = hsa.amd_queue_t(queue_properties=hsa.AMD_QUEUE_PROPERTIES_IS_PTR64 | hsa.AMD_QUEUE_PROPERTIES_ENABLE_PROFILING,
        read_dispatch_id_field_base_byte_offset=getattr(hsa.amd_queue_t, 'read_dispatch_id').offset,
        max_cu_id=(self.cu_cnt * self.xccs) - 1, max_wave_id=self.waves_per_cu - 1)
      gart.cpu_view().view(fmt='B')[:ctypes.sizeof(aql_desc)] = bytes(aql_desc)
      self.aql_desc = hsa.amd_queue_t.from_address(gart.cpu_view().addr)

    cwsr_buffer_size = round_up((ctx_save_restore_size + debug_memory_size) * self.xccs, mmap.PAGESIZE)
    cwsr_buffer = self.iface.alloc(cwsr_buffer_size) if ctx_save_restore_size else None
    eop_buffer = self.iface.alloc(eop_buffer_size) if eop_buffer_size else None

    return (self.iface.create_queue(queue_type, ring, gart, rptr=getattr(hsa.amd_queue_t, 'read_dispatch_id').offset,
            wptr=getattr(hsa.amd_queue_t, 'write_dispatch_id').offset, eop_buffer=eop_buffer, cwsr_buffer=cwsr_buffer,
            ctx_save_restore_size=ctx_save_restore_size, ctl_stack_size=ctl_stack_size, idx=idx))

  @functools.lru_cache(None)
  def sdma_queue(self, idx:int):
    if getenv("AMD_DISABLE_SDMA"): return None
    with contextlib.suppress(OSError): return self.create_queue(kfd.KFD_IOC_QUEUE_TYPE_SDMA, 0x200 if self.is_usb() else (16 << 20), idx=idx)
    return None

  def _ensure_has_local_memory(self, private_segment_size):
    if self.max_private_segment_size >= private_segment_size: return

    lanes_per_wave = 64 # wave64
    mem_alignment_size = 256 if self.target >= (11,0,0) else 1024
    size_per_thread = round_up(private_segment_size, mem_alignment_size // lanes_per_wave)
    size_per_xcc = size_per_thread * lanes_per_wave * self.iface.props['max_slots_scratch_cu'] * self.cu_cnt
    self.scratch, ok = self._realloc(getattr(self, 'scratch', None), size_per_xcc * self.xccs)
    if ok:
      # NOTE: xcc logic is correct only for GFX9.
      max_scratch_waves = self.cu_cnt * self.iface.props['max_slots_scratch_cu'] * self.xccs
      wave_scratch = ceildiv(lanes_per_wave * size_per_thread, mem_alignment_size)
      num_waves = (size_per_xcc // (wave_scratch * mem_alignment_size)) // (self.se_cnt if self.target >= (11,0,0) else 1)

      tmpring_t = getattr(hsa, f'union_COMPUTE_TMPRING_SIZE{"_GFX"+str(self.target[0]) if self.target[0] >= 11 else ""}_bitfields')
      self.tmpring_size = int.from_bytes(tmpring_t(WAVES=min(num_waves, max_scratch_waves), WAVESIZE=wave_scratch), 'little')
      self.max_private_segment_size = private_segment_size

      if hasattr(self, 'aql_desc'):
        gfx9_rsrc = {'NUM_FORMAT':hsa.BUF_NUM_FORMAT_UINT, 'DATA_FORMAT':hsa.BUF_DATA_FORMAT_32, 'ELEMENT_SIZE':1, 'INDEX_STRIDE':3}
        rsrc = {'DST_SEL_X':hsa.SQ_SEL_X, 'DST_SEL_Y':hsa.SQ_SEL_Y, 'DST_SEL_Z':hsa.SQ_SEL_Z, 'DST_SEL_W':hsa.SQ_SEL_W, 'ADD_TID_ENABLE':1,
                'TYPE':hsa.SQ_RSRC_BUF, **(gfx9_rsrc if self.target[0] < 10 else {'FORMAT':hsa.BUF_FORMAT_32_UINT, 'OOB_SELECT':2})}
        rsrc1_t = getattr(hsa, f'union_SQ_BUF_RSRC_WORD1{"_GFX11" if self.target[0] >= 11 else ""}_bitfields')
        rsrc3_t = getattr(hsa, f'union_SQ_BUF_RSRC_WORD3{"_GFX"+str(self.target[0]) if self.target[0] >= 10 else ""}_bitfields')

        self.aql_desc.scratch_backing_memory_location = int(self.scratch.va_addr)
        self.aql_desc.scratch_wave64_lane_byte_size = self.max_private_segment_size * lanes_per_wave // 64
        self.aql_desc.scratch_resource_descriptor[:] = [lo32(self.scratch.va_addr),
          int.from_bytes(rsrc1_t(BASE_ADDRESS_HI=hi32(self.scratch.va_addr), SWIZZLE_ENABLE=1), 'little'),
          lo32(size_per_xcc), int.from_bytes(bytes(rsrc3_t(**rsrc)), 'little')]
        self.aql_desc.compute_tmpring_size = self.tmpring_size

  def invalidate_caches(self):
    self.hw_compute_queue_t().memory_barrier().signal(self.timeline_signal, self.next_timeline()).submit(self)
    self.synchronize()

  def on_device_hang(self): self.iface.on_device_hang()

  def device_props(self): return self.iface.props
