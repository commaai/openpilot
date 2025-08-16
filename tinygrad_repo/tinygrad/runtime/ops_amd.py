from __future__ import annotations
from typing import cast, ClassVar
import os, ctypes, ctypes.util, struct, hashlib, functools, importlib, mmap, errno, array, contextlib, sys, weakref
assert sys.platform != 'win32'
from dataclasses import dataclass
from tinygrad.runtime.support.hcq import HCQCompiled, HCQAllocator, HCQBuffer, HWQueue, CLikeArgsState, HCQSignal, HCQProgram, FileIOInterface
from tinygrad.runtime.support.hcq import MMIOInterface
from tinygrad.uop.ops import sint
from tinygrad.device import Compiled, DMAFdRef, BufferSpec
from tinygrad.helpers import getenv, to_mv, round_up, data64_le, all_same, flatten, DEBUG, AMD_LLVM, PROFILE, ProfileEvent, suppress_finalizing
from tinygrad.renderer.cstyle import AMDRenderer
from tinygrad.renderer.llvmir import AMDLLVMRenderer
from tinygrad.runtime.autogen import kfd, hsa, pci, sqtt
from tinygrad.runtime.autogen.am import am
from tinygrad.runtime.support.compiler_amd import HIPCompiler, AMDLLVMCompiler
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.runtime.support.am.amdev import AMDev, AMMemoryManager
from tinygrad.runtime.support.amd import AMDReg, AMDIP, import_module, setup_pci_bars
from tinygrad.runtime.support.system import System, PCIIfaceBase, PCIAllocationMeta, MAP_FIXED, MAP_NORESERVE
from tinygrad.runtime.support.usb import ASM24Controller, USBMMIOInterface
if getenv("IOCTL"): import extra.hip_gpu_driver.hip_ioctl  # noqa: F401 # pylint: disable=unused-import

EVENT_INDEX_PARTIAL_FLUSH = 4 # based on a comment in nvd.h
WAIT_REG_MEM_FUNCTION_EQ  = 3 # ==
WAIT_REG_MEM_FUNCTION_NEQ = 4 # !=
WAIT_REG_MEM_FUNCTION_GEQ = 5 # >=

class AMDSignal(HCQSignal):
  def __init__(self, *args, **kwargs): super().__init__(*args, **{**kwargs, 'timestamp_divider': 100})

  def _sleep(self, time_spent_waiting_ms:int):
    # Resonable to sleep for long workloads (which take more than 2s) and only timeline signals.
    if time_spent_waiting_ms > 2000 and self.is_timeline and self.owner is not None: self.owner.iface.sleep(200)

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
    if self.pm4.PACKET3_SET_SH_REG_START <= reg.addr < self.pm4.PACKET3_SET_SH_REG_END:
      set_packet, set_packet_start = self.pm4.PACKET3_SET_SH_REG, self.pm4.PACKET3_SET_SH_REG_START
    elif self.pm4.PACKET3_SET_UCONFIG_REG_START <= reg.addr < self.pm4.PACKET3_SET_UCONFIG_REG_START + 2**16-1:
      set_packet, set_packet_start = self.pm4.PACKET3_SET_UCONFIG_REG, self.pm4.PACKET3_SET_UCONFIG_REG_START
    else: raise RuntimeError(f'Cannot set {reg.name} ({reg.addr}) via pm4 packet')
    self.pkt3(set_packet, reg.addr - set_packet_start, *(args or (reg.encode(**kwargs),)))

  @contextlib.contextmanager
  def pred_exec(self, xcc_mask:int):
    if self.dev.xccs > 1:
      self.pkt3(self.pm4.PACKET3_PRED_EXEC, xcc_mask << 24)
      prev_len = len(self._q)
    yield
    if self.dev.xccs > 1:
      self._q[prev_len-1] |= (len(self._q) - prev_len)

  def wait_reg_mem(self, value, mask=0xffffffff, mem=None, reg_req=None, reg_done=None):
    wrm_info_dw = self.pm4.WAIT_REG_MEM_MEM_SPACE(int(mem is not None)) | self.pm4.WAIT_REG_MEM_OPERATION(int(mem is None)) \
                | self.pm4.WAIT_REG_MEM_FUNCTION(WAIT_REG_MEM_FUNCTION_GEQ) | self.pm4.WAIT_REG_MEM_ENGINE(0)

    self.pkt3(self.pm4.PACKET3_WAIT_REG_MEM, wrm_info_dw, *(data64_le(mem) if mem is not None else (reg_req, reg_done)), value, mask, 4)

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

  def xcc_barrier(self):
    if self.dev.xcc_sync is None: return self
    assert self.dev.xccs == 8, 'only 8 XCCs supported'
    a, b = self.dev.xcc_sync
    mem_eq = self.pm4.WAIT_REG_MEM_FUNCTION(WAIT_REG_MEM_FUNCTION_EQ) | self.pm4.WAIT_REG_MEM_MEM_SPACE(1)
    self.pkt3(self.pm4.PACKET3_ATOMIC_MEM, self.soc.TC_OP_ATOMIC_ADD_RTN_32, *data64_le(a.value_addr), *data64_le(1), *data64_le(0), 0x10) # a += 1
    self.pkt3(self.pm4.PACKET3_WAIT_REG_MEM, mem_eq, *data64_le(a.value_addr), 0, 0b111, 0x80) # a == 0 (mod 8) via bitmask
    self.pkt3(self.pm4.PACKET3_ATOMIC_MEM, self.soc.TC_OP_ATOMIC_ADD_RTN_32, *data64_le(b.value_addr), *data64_le(1), *data64_le(0), 0x10) # b += 1
    self.pkt3(self.pm4.PACKET3_WAIT_REG_MEM, mem_eq, *data64_le(b.value_addr), 0, 0b111, 0x80) # b == 0 (mod 8) via bitmask
    return self

  def memory_barrier(self):
    pf = '' if self.nbio.version[0] == 2 else '0' if self.nbio.version[:2] != (7, 11) else '1'
    self.wait_reg_mem(reg_req=getattr(self.nbio, f'regBIF_BX_PF{pf}_GPU_HDP_FLUSH_REQ').addr,
                      reg_done=getattr(self.nbio, f'regBIF_BX_PF{pf}_GPU_HDP_FLUSH_DONE').addr, value=0xffffffff)
    self.acquire_mem()
    return self

  def xcc_config(self):
    self.wreg(self.gc.regCOMPUTE_TG_CHUNK_SIZE, 1)
    for xcc_id in range(self.dev.xccs):
      with self.pred_exec(xcc_mask=1 << xcc_id):
        self.wreg(self.dev.regCOMPUTE_CURRENT_LOGIC_XCC_ID, xcc_id)
    return self

  def spi_config(self, tracing:bool):
    self.wreg(self.gc.regSPI_CONFIG_CNTL, ps_pkr_priority_cntl=3, exp_priority_order=3, gpr_write_priority=0x2c688,
              enable_sqg_bop_events=int(tracing), enable_sqg_top_events=int(tracing))

  ### SQTT ###

  def sqtt_userdata(self, data, *extra_dwords):
    data_ints = [x[0] for x in struct.iter_unpack('<I', bytes(data))] + list(extra_dwords)
    for i in range(0, len(data_ints), 2):
      self.wreg(self.gc.regSQ_THREAD_TRACE_USERDATA_2, *data_ints[i:i+2])

  def sqtt_config(self, tracing:bool):
    self.wreg(self.gc.regSQ_THREAD_TRACE_CTRL, draw_event_en=1, spi_stall_en=1, sq_stall_en=1, reg_at_hwm=2, hiwater=1,
              rt_freq=self.soc.SQ_TT_RT_FREQ_4096_CLK, util_timer=self.soc.SQ_TT_UTIL_TIMER_250_CLK, mode=int(tracing))

  # Magic values from mesa/src/amd/vulkan/radv_sqtt.c:radv_emit_spi_config_cntl and src/amd/common/ac_sqtt.c:ac_sqtt_emit_start
  def sqtt_start(self, buf0s:list[HCQBuffer], se_mask:int):
    self.memory_barrier()
    self.spi_config(tracing=True)
    # One buffer for one SE, mesa does it with a single buffer and ac_sqtt_get_data_offset, but this is simpler and should work just as well
    for se in range(len(buf0s)):
      self.wreg(self.gc.regGRBM_GFX_INDEX, se_index=se, instance_broadcast_writes=1)
      buf0_lo, buf0_hi = data64_le(buf0s[se].va_addr>>12)
      self.wreg(self.gc.regSQ_THREAD_TRACE_BUF0_SIZE, base_hi=buf0_hi, size=buf0s[se].size>>12)
      self.wreg(self.gc.regSQ_THREAD_TRACE_BUF0_BASE, base_lo=buf0_lo)
      # NOTE: SQTT can only trace instructions on one simd per se, this selects first simd in first wgp in first sa.
      # For RGP to display instruction trace it has to see it on first SE. Howerver ACE/MEC/whatever does the dispatching starting with second se,
      # and on amdgpu/non-AM it also does weird things with dispatch order inside se: around 7 times out of 10 it starts from the last cu, but
      # sometimes not, especially if the kernel has more than one wavefront which means that kernels with small global size might get unlucky and
      # be dispatched on something else and not be seen in instruction tracing tab. You can force the wavefronts of a kernel to be dispatched on the
      # CUs you want to by disabling other CUs via bits in regCOMPUTE_STATIC_THREAD_MGMT_SE<x> and trace even kernels that only have one wavefront.
      self.wreg(self.gc.regSQ_THREAD_TRACE_MASK, wtype_include=self.soc.SQ_TT_WTYPE_INCLUDE_CS_BIT, simd_sel=0, wgp_sel=0, sa_sel=0)
      REG_INCLUDE = self.soc.SQ_TT_TOKEN_MASK_SQDEC_BIT | self.soc.SQ_TT_TOKEN_MASK_SHDEC_BIT | self.soc.SQ_TT_TOKEN_MASK_GFXUDEC_BIT | \
                    self.soc.SQ_TT_TOKEN_MASK_COMP_BIT | self.soc.SQ_TT_TOKEN_MASK_CONTEXT_BIT | self.soc.SQ_TT_TOKEN_MASK_CONTEXT_BIT
      TOKEN_EXCLUDE = 1 << self.soc.SQ_TT_TOKEN_EXCLUDE_PERF_SHIFT
      if not (se_mask >> se) & 0b1:
        TOKEN_EXCLUDE |= 1 << self.soc.SQ_TT_TOKEN_EXCLUDE_VMEMEXEC_SHIFT | 1 << self.soc.SQ_TT_TOKEN_EXCLUDE_ALUEXEC_SHIFT | \
                         1 << self.soc.SQ_TT_TOKEN_EXCLUDE_VALUINST_SHIFT | 1 << self.soc.SQ_TT_TOKEN_EXCLUDE_IMMEDIATE_SHIFT | \
                         1 << self.soc.SQ_TT_TOKEN_EXCLUDE_INST_SHIFT
      self.wreg(self.gc.regSQ_THREAD_TRACE_TOKEN_MASK, reg_include=REG_INCLUDE, token_exclude=TOKEN_EXCLUDE, bop_events_token_include=1)
      # Enable SQTT
      self.sqtt_config(tracing=True)
    # Restore global broadcasting
    self.wreg(self.gc.regGRBM_GFX_INDEX, se_broadcast_writes=1, sa_broadcast_writes=1, instance_broadcast_writes=1)
    self.wreg(self.gc.regCOMPUTE_THREAD_TRACE_ENABLE, 1)
    self.memory_barrier()
    return self

  # Magic values from src/amd/common/ac_sqtt.c:ac_sqtt_emit_stop and src/amd/common/ac_sqtt.c:ac_sqtt_emit_wait
  def sqtt_stop(self, ses: int, wptrs: HCQBuffer):
    self.memory_barrier()
    # Start shutting everything down
    self.wreg(self.gc.regCOMPUTE_THREAD_TRACE_ENABLE, 0)
    self.pkt3(self.pm4.PACKET3_EVENT_WRITE, self.pm4.EVENT_TYPE(self.soc.THREAD_TRACE_FINISH) | self.pm4.EVENT_INDEX(0))
    # For each SE wait for finish to complete and copy regSQ_THREAD_TRACE_WPTR to know where in the buffer trace data ends
    for se in range(ses):
      self.wreg(self.gc.regGRBM_GFX_INDEX, se_index=se, instance_broadcast_writes=1)
      # Wait for FINISH_PENDING==0
      self.pkt3(self.pm4.PACKET3_WAIT_REG_MEM, self.pm4.WAIT_REG_MEM_FUNCTION(WAIT_REG_MEM_FUNCTION_EQ),
                self.gc.regSQ_THREAD_TRACE_STATUS.addr, 0, 0, self.gc.regSQ_THREAD_TRACE_STATUS.fields_mask('finish_pending'), 4)
      # Wait for FINISH_DONE!=0
      self.pkt3(self.pm4.PACKET3_WAIT_REG_MEM, self.pm4.WAIT_REG_MEM_FUNCTION(WAIT_REG_MEM_FUNCTION_NEQ),
                self.gc.regSQ_THREAD_TRACE_STATUS.addr, 0, 0, self.gc.regSQ_THREAD_TRACE_STATUS.fields_mask('finish_done'), 4)
      # Disable SQTT
      self.sqtt_config(tracing=False)
      # Wait for BUSY==0
      self.pkt3(self.pm4.PACKET3_WAIT_REG_MEM, self.pm4.WAIT_REG_MEM_FUNCTION(WAIT_REG_MEM_FUNCTION_EQ),
                self.gc.regSQ_THREAD_TRACE_STATUS.addr, 0, 0, self.gc.regSQ_THREAD_TRACE_STATUS.fields_mask('busy'), 4)
      # Copy WPTR to memory (src_sel = perf, dst_sel = tc_l2, wr_confirm = True)
      self.pkt3(self.pm4.PACKET3_COPY_DATA, 1 << 20 | 2 << 8 | 4, self.gc.regSQ_THREAD_TRACE_WPTR.addr, 0, *data64_le(wptrs.va_addr+(se*4)))
    # Restore global broadcasting
    self.wreg(self.gc.regGRBM_GFX_INDEX, se_broadcast_writes=1, sa_broadcast_writes=1, instance_broadcast_writes=1)
    self.spi_config(tracing=False)
    self.memory_barrier()
    return self

  def sqtt_prg_marker(self, prg:AMDProgram, global_size:tuple[sint, ...]):
    BIND_POINT_COMPUTE = 1

    self.sqtt_userdata(sqtt.struct_rgp_sqtt_marker_pipeline_bind(
      _0=sqtt.union_rgp_sqtt_marker_pipeline_bind_0(_0=sqtt.struct_rgp_sqtt_marker_pipeline_bind_0_0(
        identifier=sqtt.RGP_SQTT_MARKER_IDENTIFIER_BIND_PIPELINE, bind_point=BIND_POINT_COMPUTE)),
      _1=sqtt.union_rgp_sqtt_marker_pipeline_bind_1(api_pso_hash=data64_le(prg.libhash[0]))))

    self.sqtt_userdata(sqtt.struct_rgp_sqtt_marker_event(
      _0=sqtt.union_rgp_sqtt_marker_event_0(_0=sqtt.struct_rgp_sqtt_marker_event_0_0(has_thread_dims=1)),
      _2=sqtt.union_rgp_sqtt_marker_event_2(cmd_id=prg.dev.cmd_id)), *global_size)

    prg.dev.cmd_id += 1

  def exec(self, prg:AMDProgram, args_state:CLikeArgsState, global_size:tuple[sint, ...], local_size:tuple[sint, ...]):
    self.bind_args_state(args_state)

    self.acquire_mem(gli=0, gl2=0)

    user_regs = []
    if prg.enable_private_segment_sgpr:
      assert self.dev.xccs == 1, "Only architected flat scratch is suppored on multi-xcc"
      scratch_hilo = data64_le(prg.dev.scratch.va_addr)
      # sgpr word1 bit31 enables swizzle
      # sgpr word3 = 0x14 << 12 | 2 << 28 | 2 << 21 | 1 << 23
      user_regs = [scratch_hilo[0], scratch_hilo[1] | 1 << 31, 0xffffffff, 0x20c14000]

    if prg.enable_dispatch_ptr:
      dp = (dp_t:=hsa.hsa_kernel_dispatch_packet_t).from_address(cast(int, (disp_buf:=args_state.buf.offset(prg.kernargs_segment_size)).va_addr))

      self.bind_sints(*local_size, mem=disp_buf.cpu_view(), struct_t=dp_t, start_field='workgroup_size_x', fmt='H')
      self.bind_sints(*[g*l for g,l in zip(global_size, local_size)], mem=disp_buf.cpu_view(), struct_t=dp_t, start_field='grid_size_x', fmt='I')
      dp.group_segment_size, dp.private_segment_size, dp.kernarg_address = prg.group_segment_size, prg.private_segment_size, args_state.buf.va_addr
      user_regs += [*data64_le(disp_buf.va_addr)]

    user_regs += [*data64_le(args_state.buf.va_addr)]

    if prg.dev.sqtt_enabled: self.sqtt_prg_marker(prg, global_size)

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
    self.wreg(self.gc.regCOMPUTE_STATIC_THREAD_MGMT_SE0, 0xFFFFFFFF, 0xFFFFFFFF)
    self.wreg(self.gc.regCOMPUTE_STATIC_THREAD_MGMT_SE2, 0xFFFFFFFF, 0xFFFFFFFF)
    if prg.dev.target >= (11,0,0): self.wreg(self.gc.regCOMPUTE_STATIC_THREAD_MGMT_SE4, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF)

    self.wreg(self.gc.regCOMPUTE_USER_DATA_0, *user_regs)
    self.wreg(self.gc.regCOMPUTE_RESOURCE_LIMITS, 0)

    self.wreg(self.gc.regCOMPUTE_START_X, 0, 0, 0, *local_size, 0, 0)

    gfx10p = {'cs_w32_en': int(prg.wave32)} if prg.dev.target >= (10,0,0) else {}
    DISPATCH_INITIATOR = self.gc.regCOMPUTE_DISPATCH_INITIATOR.encode(**gfx10p, force_start_at_000=1, compute_shader_en=1)
    self.pkt3(self.pm4.PACKET3_DISPATCH_DIRECT, *global_size, DISPATCH_INITIATOR)

    if prg.dev.sqtt_enabled: self.pkt3(self.pm4.PACKET3_EVENT_WRITE, self.pm4.EVENT_TYPE(self.soc.THREAD_TRACE_MARKER) | self.pm4.EVENT_INDEX(0))
    self.pkt3(self.pm4.PACKET3_EVENT_WRITE, self.pm4.EVENT_TYPE(self.soc.CS_PARTIAL_FLUSH) | self.pm4.EVENT_INDEX(EVENT_INDEX_PARTIAL_FLUSH))

    if self.dev.xccs > 1:
      self.release_mem(cache_flush=True)
      self.acquire_mem(gli=0)
      self.xcc_barrier()
    return self

  def wait(self, signal:AMDSignal, value:sint=0):
    self.wait_reg_mem(mem=signal.value_addr, value=value, mask=0xffffffff)
    if self.dev.xccs > 1: self.xcc_barrier()
    return self

  def timestamp(self, signal:AMDSignal):
    with self.pred_exec(xcc_mask=0b1):
      self.release_mem(signal.timestamp_addr, 0, self.pm4.data_sel__mec_release_mem__send_gpu_clock_counter, self.pm4.int_sel__mec_release_mem__none)
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

class AMDCopyQueue(HWQueue):
  def __init__(self, dev, max_copy_size=0x40000000):
    self.dev, self.sdma, self.internal_cmd_sizes, self.max_copy_size = dev, dev.sdma, [], max_copy_size
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
    elif dev is not None and dev.is_am(): self.q(self.sdma.SDMA_OP_TRAP, self.sdma.SDMA_PKT_TRAP_INT_CONTEXT_INT_CONTEXT(0))

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
    if self.binded_device == dev:
      # An IB packet must end on a 8 DW boundary.
      add = (8 - (((dev.sdma_queue.put_value % 32) // 4) + len(self.indirect_cmd) % 8)) % 8
      cmds, cmd_sizes = ([0] * add) + self.indirect_cmd, [len(self.indirect_cmd) + add]

      if len(cmds) * 4 >= (dev.sdma_queue.ring.nbytes - dev.sdma_queue.put_value % dev.sdma_queue.ring.nbytes):
        cmds, cmd_sizes = [0, 0] + self.indirect_cmd, [8]
    else: cmds, cmd_sizes = self._q, self.internal_cmd_sizes

    tail_blit_dword = 0
    for cmdsz in cmd_sizes:
      if (tail_blit_dword + cmdsz) * 4 >= dev.sdma_queue.ring.nbytes - dev.sdma_queue.put_value % dev.sdma_queue.ring.nbytes: break
      tail_blit_dword += cmdsz

    # Force align of submits to hit our usb layer write cache.
    if (rem_packet_cnt := len(cmds) - tail_blit_dword) > 0 and dev.is_usb(): tail_blit_dword = 0

    # USB devices run in single-step mode, so they can't overrun the queue.
    total_bytes = (tail_blit_dword * 4 if rem_packet_cnt == 0 else -dev.sdma_queue.put_value % dev.sdma_queue.ring.nbytes) + rem_packet_cnt * 4
    assert total_bytes < dev.sdma_queue.ring.nbytes, "SDMA queue overrun"
    while not dev.is_usb() and dev.sdma_queue.put_value + total_bytes - dev.sdma_queue.read_ptr > dev.sdma_queue.ring.nbytes: pass

    start_idx = (dev.sdma_queue.put_value % dev.sdma_queue.ring.nbytes) // 4
    dev.sdma_queue.ring[start_idx : start_idx + tail_blit_dword] = array.array('I', cmds[:tail_blit_dword])
    dev.sdma_queue.put_value += tail_blit_dword * 4

    if (rem_packet_cnt := len(cmds) - tail_blit_dword) > 0:
      zero_fill = dev.sdma_queue.ring.nbytes - dev.sdma_queue.put_value % dev.sdma_queue.ring.nbytes
      dev.sdma_queue.ring.view(dev.sdma_queue.put_value % dev.sdma_queue.ring.nbytes, zero_fill, fmt='B')[:] = bytes(zero_fill)
      dev.sdma_queue.put_value += zero_fill

      dev.sdma_queue.ring[0:rem_packet_cnt] = array.array('I', cmds[tail_blit_dword:])
      dev.sdma_queue.put_value += rem_packet_cnt * 4

    dev.sdma_queue.signal_doorbell(dev)

class AMDProgram(HCQProgram):
  def __init__(self, dev:AMDDevice, name:str, lib:bytes):
    # TODO; this API needs the type signature of the function and global_size/local_size
    self.dev, self.name, self.lib = dev, name, lib

    image, sections, _ = elf_loader(self.lib)
    self.lib_gpu = self.dev.allocator.alloc(round_up(image.nbytes, 0x1000), buf_spec:=BufferSpec(cpu_access=True, nolru=True))
    self.dev.allocator._copyin(self.lib_gpu, image)
    self.dev.synchronize()

    rodata_entry = next((sh.header.sh_addr for sh in sections if sh.name == ".rodata"), -1)
    text_entry = next((sh.header.sh_addr for sh in sections if sh.name == ".text"), -1)
    assert rodata_entry >= 0 and text_entry >= 0, ".text or .rodata section not found"
    self.group_segment_size = image[rodata_entry:rodata_entry+4].cast("I")[0]
    self.private_segment_size = image[rodata_entry+4:rodata_entry+8].cast("I")[0]
    self.kernargs_segment_size = image[rodata_entry+8:rodata_entry+12].cast("I")[0]
    lds_size = ((self.group_segment_size + 511) // 512) & 0x1FF
    if lds_size > (self.dev.iface.props['lds_size_in_kb'] * 1024) // 512: raise RuntimeError("Too many resources requested: group_segment_size")

    # Ensure scratch size
    self.dev._ensure_has_local_memory(self.private_segment_size)

    # NOTE: this is wrong, it's not this object. pad it, since it might be smaller than the struct
    code = hsa.amd_kernel_code_t.from_buffer_copy(bytes(image[rodata_entry:rodata_entry+256]) + b'\x00'*256)
    self.wave32: bool = code.kernel_code_properties & 0x400 == 0x400

    # Set rsrc1.priv=1 on gfx11 to workaround cwsr.
    self.rsrc1: int = code.compute_pgm_rsrc1 | ((1 << 20) if (11,0,0) <= self.dev.target < (12,0,0) else 0)
    self.rsrc2: int = code.compute_pgm_rsrc2 | (lds_size << 15)
    self.rsrc3: int = image[rodata_entry+44:rodata_entry+48].cast("I")[0] # NOTE: kernel descriptor, not in amd_kernel_code_t struct
    self.prog_addr: int = self.lib_gpu.va_addr + rodata_entry + code.kernel_code_entry_byte_offset
    if code.kernel_code_entry_byte_offset == 0: self.prog_addr = self.lib_gpu.va_addr + text_entry
    # Some programs use hsa_kernel_dispatch_packet_t to read workgroup sizes during execution.
    # The packet is represented as a pointer and set up in SGPRs. Space for the packet is allocated as part of the kernel arguments.
    self.enable_dispatch_ptr: int = code.kernel_code_properties & hsa.AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_PTR
    self.enable_private_segment_sgpr: int = code.kernel_code_properties & hsa.AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER
    additional_alloc_sz = ctypes.sizeof(hsa.hsa_kernel_dispatch_packet_t) if self.enable_dispatch_ptr else 0

    if dev.sqtt_enabled: self.libhash: tuple[int, int] = struct.unpack('<Q', hashlib.md5(self.lib).digest()[:8])*2

    super().__init__(CLikeArgsState, self.dev, self.name, kernargs_alloc_size=self.kernargs_segment_size+additional_alloc_sz, lib=self.lib,
                     base=self.lib_gpu.va_addr)
    weakref.finalize(self, self._fini, self.dev, self.lib_gpu, buf_spec)

class AMDAllocator(HCQAllocator['AMDDevice']):
  def __init__(self, dev:AMDDevice):
    super().__init__(dev, copy_bufs=getattr(dev.iface, 'copy_bufs', None), max_copyout_size=0x1000 if dev.is_usb() else None)
    if hasattr(dev.iface, "as_dmaref"): self._as_dmaref = dev.iface.as_dmaref
    self.supports_copy_from_disk = not dev.is_usb()

  def _alloc(self, size:int, options:BufferSpec) -> HCQBuffer:
    return self.dev.iface.alloc(size, host=options.host, uncached=options.uncached, cpu_access=options.cpu_access)

  @suppress_finalizing
  def _free(self, opaque, options:BufferSpec):
    self.dev.synchronize()
    self.dev.iface.free(opaque)

  def _map(self, buf:HCQBuffer): return self.dev.iface.map(buf._base if buf._base is not None else buf)

@dataclass(frozen=True)
class ProfileSQTTEvent(ProfileEvent): device:str; se:int; blob:bytes; itrace:bool # noqa: E702

@dataclass
class AMDQueueDesc:
  ring: MMIOInterface
  read_ptrs: list[MMIOInterface]
  write_ptrs: list[MMIOInterface]
  doorbells: list[MMIOInterface]
  put_value: int = 0

  @property
  def read_ptr(self): return min(p[0] for p in self.read_ptrs)

  @classmethod
  def multi(cls, *queues: AMDQueueDesc):
    assert all_same([(q.ring.addr, q.put_value) for q in queues]), f"All queues must have the same ring and put_value: {queues}"
    return cls(ring=queues[0].ring, put_value=queues[0].put_value, doorbells=flatten(q.doorbells for q in queues),
               read_ptrs=flatten(q.read_ptrs for q in queues), write_ptrs=flatten(q.write_ptrs for q in queues))

  def signal_doorbell(self, dev):
    for write_ptr in self.write_ptrs: write_ptr[0] = self.put_value

    # Ensure all prior writes are visible to the GPU.
    System.memory_barrier()

    # Flush hdp if queue is in dev mem.
    if dev.is_am() and not dev.is_usb(): dev.iface.dev_impl.gmc.flush_hdp()
    for doorbell in self.doorbells: doorbell[0] = self.put_value

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
      gpus = sorted(gpus, key=lambda x: int(x.split('/')[-1]))
      visible_devices = [int(x) for x in (getenv('VISIBLE_DEVICES', getenv('HIP_VISIBLE_DEVICES', ''))).split(',') if x.strip()]
      KFDIface.gpus = [gpus[x] for x in visible_devices] if visible_devices else gpus

    if device_id >= len(KFDIface.gpus): raise RuntimeError(f"No device found for {device_id}. Requesting more devices than the system has?")

    self.gpu_id = int(FileIOInterface(f"{kfd_topo_path}/{KFDIface.gpus[device_id]}/gpu_id").read())
    self.props = {(p:=l.split())[0]: int(p[1]) for l in FileIOInterface(f"{kfd_topo_path}/{KFDIface.gpus[device_id]}/properties").read().splitlines()}
    ip_base = f"/sys/class/drm/renderD{self.props['drm_render_minor']}/device/ip_discovery/die/0"
    id2ip = {am.GC_HWID: am.GC_HWIP, am.SDMA0_HWID: am.SDMA0_HWIP, am.NBIF_HWID: am.NBIF_HWIP}
    self.ip_versions = {id2ip[int(hwid)]:tuple(int(FileIOInterface(f'{ip_base}/{hwid}/0/{part}').read()) for part in ['major', 'minor', 'revision'])
                        for hwid in FileIOInterface(ip_base).listdir() if hwid.isnumeric() and int(hwid) in id2ip}
    self.ip_offsets = {id2ip[int(hwid)]:tuple(int(x, 16) for x in FileIOInterface(f'{ip_base}/{hwid}/0/base_addr').read().splitlines())
                        for hwid in FileIOInterface(ip_base).listdir() if hwid.isnumeric() and int(hwid) in id2ip}
    self.drm_fd = FileIOInterface(f"/dev/dri/renderD{self.props['drm_render_minor']}", os.O_RDWR)

    kfd.AMDKFD_IOC_ACQUIRE_VM(KFDIface.kfd, drm_fd=self.drm_fd.fd, gpu_id=self.gpu_id)

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

    if cpu_access or host: flags |= kfd.KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC

    if flags & kfd.KFD_IOC_ALLOC_MEM_FLAGS_USERPTR:
      buf = addr = cpu_addr or FileIOInterface.anon_mmap(0, size, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED | mmap.MAP_ANONYMOUS, 0)
    else: buf, addr = 0, FileIOInterface.anon_mmap(0, size, 0, mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS | MAP_NORESERVE, 0)

    try: mem = kfd.AMDKFD_IOC_ALLOC_MEMORY_OF_GPU(self.kfd, va_addr=addr, size=size, base=addr, length=size, gpu_id=self.gpu_id,
                                                  flags=flags, mmap_offset=buf)
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
    dmaref = DMAFdRef(kfd.AMDKFD_IOC_EXPORT_DMABUF(KFDIface.kfd, handle=base.meta.handle, flags=0).dmabuf_fd, mem.va_addr-base.va_addr, mem.size)
    weakref.finalize(dmaref, os.close, dmaref.fd)
    return dmaref

  def map(self, mem):
    if mem.owner is not None and mem.owner._is_cpu(): return self.alloc(mem.size, host=True, cpu_addr=mem.va_addr)

    c_gpus = (ctypes.c_int32 * 1)(self.gpu_id)
    stm = kfd.AMDKFD_IOC_MAP_MEMORY_TO_GPU(self.kfd, handle=mem.meta.handle, device_ids_array_ptr=ctypes.addressof(c_gpus), n_devices=1)
    assert stm.n_success == 1

  def create_queue(self, queue_type, ring, gart, eop_buffer=None, cwsr_buffer=None, ctl_stack_size=0, ctx_save_restore_size=0, xcc_id=0):
    queue = kfd.AMDKFD_IOC_CREATE_QUEUE(KFDIface.kfd, ring_base_address=ring.va_addr, ring_size=ring.size, gpu_id=self.gpu_id,
      queue_type=queue_type, queue_percentage=kfd.KFD_MAX_QUEUE_PERCENTAGE|(xcc_id<<8), queue_priority=kfd.KFD_MAX_QUEUE_PRIORITY,
      eop_buffer_address=eop_buffer.va_addr if eop_buffer else 0, eop_buffer_size=eop_buffer.size if eop_buffer else 0, ctl_stack_size=ctl_stack_size,
      ctx_save_restore_address=cwsr_buffer.va_addr if cwsr_buffer else 0, ctx_save_restore_size=ctx_save_restore_size,
      write_pointer_address=gart.va_addr, read_pointer_address=gart.va_addr + 8 * (xcc_id + 1))

    if not hasattr(self, 'doorbells'):
      self.doorbells_base = queue.doorbell_offset & (~0x1fff) # doorbell is two pages
      self.doorbells = cast(FileIOInterface, KFDIface.kfd).mmap(0, 0x2000, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED, self.doorbells_base)

    return AMDQueueDesc(ring=MMIOInterface(ring.va_addr, ring.size, fmt='I'), read_ptrs=[MMIOInterface(queue.read_pointer_address, 8, fmt='Q')],
                        write_ptrs=[MMIOInterface(queue.write_pointer_address, 8, fmt='Q')],
                        doorbells=[MMIOInterface(self.doorbells + queue.doorbell_offset - self.doorbells_base, 8, fmt='Q')])

  def sleep(self, tm:int): kfd.AMDKFD_IOC_WAIT_EVENTS(KFDIface.kfd, events_ptr=self.queue_event_arr_ptr, num_events=1, wait_for_all=1, timeout=tm)

  def on_device_hang(self):
    def _collect_str(st): return ' '.join(f'{k[0]}={getattr(st, k[0])}' for k in st._fields_)

    report = []
    for evnt in [self.mem_fault_event, self.hw_fault_event]:
      ev = (kfd.struct_kfd_event_data)(event_id=evnt.event_id)
      kfd.AMDKFD_IOC_WAIT_EVENTS(KFDIface.kfd, events_ptr=ctypes.addressof(ev), num_events=1, wait_for_all=1)
      if evnt == self.mem_fault_event and ev.memory_exception_data.gpu_id:
        report += [f"MMU fault: 0x{ev.memory_exception_data.va:X} | {_collect_str(ev.memory_exception_data.failure)}"]
      if evnt == self.hw_fault_event and ev.hw_exception_data.gpu_id: report += [f"HW fault: {_collect_str(ev.hw_exception_data)}"]

    raise RuntimeError("\n".join(report))

class PCIIface(PCIIfaceBase):
  gpus:ClassVar[list[str]] = []

  def __init__(self, dev, dev_id):
    super().__init__(dev, dev_id, vendor=0x1002, devices=[0x744c, 0x7480, 0x7550], bars=[0, 2, 5], vram_bar=0,
      va_start=AMMemoryManager.va_allocator.base, va_size=AMMemoryManager.va_allocator.size)
    self._setup_adev(self.pci_dev.pcibus, self.pci_dev.map_bar(0), self.pci_dev.map_bar(2, fmt='Q'), self.pci_dev.map_bar(5, fmt='I'))
    self.pci_dev.write_config(pci.PCI_COMMAND, self.pci_dev.read_config(pci.PCI_COMMAND, 2) | pci.PCI_COMMAND_MASTER, 2)

  def _setup_adev(self, name, vram:MMIOInterface, doorbell:MMIOInterface, mmio:MMIOInterface, dma_regions:list[tuple[int, MMIOInterface]]|None=None):
    self.dev_impl:AMDev = AMDev(name, vram, doorbell, mmio, dma_regions)
    self.ip_versions = self.dev_impl.ip_ver
    self.ip_offsets = {hwip: tuple(instances[0]) for hwip,instances in self.dev_impl.regs_offset.items()}

    gfxver = int(f"{self.dev_impl.ip_ver[am.GC_HWIP][0]:02d}{self.dev_impl.ip_ver[am.GC_HWIP][1]:02d}{self.dev_impl.ip_ver[am.GC_HWIP][2]:02d}")
    array_count = self.dev_impl.gc_info.gc_num_sa_per_se * self.dev_impl.gc_info.gc_num_se
    simd_count = 2 * array_count * (self.dev_impl.gc_info.gc_num_wgp0_per_sa + self.dev_impl.gc_info.gc_num_wgp1_per_sa)
    self.props = {'simd_count': 2 * simd_count, 'simd_per_cu': 2, 'array_count': array_count, 'gfx_target_version': gfxver,
      'max_slots_scratch_cu': self.dev_impl.gc_info.gc_max_scratch_slots_per_cu, 'max_waves_per_simd': self.dev_impl.gc_info.gc_max_waves_per_simd,
      'simd_arrays_per_engine': self.dev_impl.gc_info.gc_num_sa_per_se, 'lds_size_in_kb': self.dev_impl.gc_info.gc_lds_size}

  def create_queue(self, queue_type, ring, gart, eop_buffer=None, cwsr_buffer=None, ctl_stack_size=0, ctx_save_restore_size=0, xcc_id=0):
    assert cwsr_buffer is None, "no cwsr buffer for am"

    if queue_type == kfd.KFD_IOC_QUEUE_TYPE_SDMA:
      self.dev_impl.sdma.setup_ring(ring_addr=ring.va_addr, ring_size=ring.size, rptr_addr=gart.va_addr, wptr_addr=gart.va_addr+0x10,
                                    doorbell=(doorbell_index:=am.AMDGPU_NAVI10_DOORBELL_sDMA_ENGINE0), pipe=0, queue=0)
    else:
      self.dev_impl.gfx.setup_ring(ring_addr=ring.va_addr, ring_size=ring.size, rptr_addr=gart.va_addr, wptr_addr=gart.va_addr+0x10,
        eop_addr=eop_buffer.va_addr, eop_size=eop_buffer.size, doorbell=(doorbell_index:=am.AMDGPU_NAVI10_DOORBELL_MEC_RING0), pipe=0, queue=0)

    return AMDQueueDesc(ring=ring.cpu_view().view(fmt='I'), doorbells=[self.dev_impl.doorbell64.view(doorbell_index * 8, 8, fmt='Q')],
      read_ptrs=[gart.cpu_view().view(size=8, fmt='Q')], write_ptrs=[gart.cpu_view().view(offset=0x10, size=8, fmt='Q')])

  def sleep(self, timeout):
    if self.pci_dev.irq_poller is not None and (events_cnt:=len(self.pci_dev.irq_poller.poll(timeout))):
      self.pci_dev.irq_fd.read(8 * events_cnt)
      self.dev_impl.ih.interrupt_handler()

  def on_device_hang(self):
    devs:list[AMDDevice] = [d for pg in HCQCompiled.peer_groups.values() for d in pg if isinstance(d, AMDDevice) and d.is_am()]
    for d in devs: d.iface.dev_impl.gmc.on_interrupt()
    raise RuntimeError("Device hang detected")

  def device_fini(self): self.dev_impl.fini()

class USBIface(PCIIface):
  def __init__(self, dev, dev_id):
    self.dev = dev
    self.usb = ASM24Controller()
    self.bars = setup_pci_bars(self.usb, gpu_bus=4, mem_base=0x10000000, pref_mem_base=(32 << 30))

    self._setup_adev(f"usb:{dev_id}", USBMMIOInterface(self.usb, *self.bars[0], fmt='B'), USBMMIOInterface(self.usb, *self.bars[2], fmt='Q'),
      USBMMIOInterface(self.usb, *self.bars[5], fmt='I'), dma_regions=[(0x200000, self._dma_view(0xf000, 0x80000))])
    self.usb._pci_cacheable += [self.bars[2]] # doorbell region is cacheable

    # special regions
    self.copy_bufs = [self._dma_region(ctrl_addr=0xf000, sys_addr=0x200000, size=0x80000)]
    self.sys_buf, self.sys_next_off = self._dma_region(ctrl_addr=0xa000, sys_addr=0x820000, size=0x1000), 0x800

  def _dma_view(self, ctrl_addr, size): return USBMMIOInterface(self.usb, ctrl_addr, size, fmt='B', pcimem=False)
  def _dma_region(self, ctrl_addr, sys_addr, size):
    region = self.dev_impl.mm.map_range(vaddr:=self.dev_impl.mm.alloc_vaddr(size=size), size, [(sys_addr, size)], system=True, uncached=True)
    return HCQBuffer(vaddr, size, meta=PCIAllocationMeta(region, has_cpu_mapping=False), view=self._dma_view(ctrl_addr, size), owner=self.dev)

  def alloc(self, size:int, host=False, uncached=False, cpu_access=False, contiguous=False, **kwargs) -> HCQBuffer:
    if (host or (uncached and cpu_access)) and self.sys_next_off + size < self.sys_buf.size:
      self.sys_next_off += size
      return self.sys_buf.offset(self.sys_next_off - size, size)

    am_mapping = self.dev_impl.mm.valloc(size:=round_up(size, 4 << 10), uncached=uncached, contiguous=cpu_access)
    return HCQBuffer(am_mapping.va_addr, size, meta=PCIAllocationMeta(am_mapping, has_cpu_mapping=False),
      view=USBMMIOInterface(self.usb, self.bars[0][0] + am_mapping.paddrs[0][0], size, fmt='B') if cpu_access else None, owner=self.dev)

  def create_queue(self, queue_type, ring, gart, eop_buffer=None, cwsr_buffer=None, ctl_stack_size=0, ctx_save_restore_size=0, xcc_id=0):
    if queue_type == kfd.KFD_IOC_QUEUE_TYPE_COMPUTE: self.usb._pci_cacheable += [(ring.cpu_view().addr, ring.size)]
    return super().create_queue(queue_type, ring, gart, eop_buffer, cwsr_buffer, ctl_stack_size, ctx_save_restore_size, xcc_id)

  def sleep(self, timeout): pass

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

    self.max_cu_id = self.iface.props['simd_count'] // self.iface.props['simd_per_cu'] // self.iface.props.get('num_xcc', 1) - 1
    self.max_wave_id = (self.iface.props['max_waves_per_simd'] * self.iface.props['simd_per_cu'] - 1) if self.target >= (10,1,0) else \
                       (min((self.max_cu_id+1)*40, self.iface.props['array_count'] // self.iface.props['simd_arrays_per_engine'] * 512) - 1)
    self.xccs = self.iface.props.get('num_xcc', 1) if getenv("XCCS", 1) else 1
    # this is what llvm refers to as "architected flat scratch"
    self.has_scratch_base_registers = self.target >= (11,0,0) or self.target in {(9,4,2), (9,5,0)}

    # https://gitlab.freedesktop.org/agd5f/linux/-/blob/a1fc9f584c4aaf8bc1ebfa459fc57a3f26a290d8/drivers/gpu/drm/amd/amdkfd/kfd_queue.c#L391
    sgrp_size_per_cu, lds_size_per_cu, hwreg_size_per_cu = 0x4000, 0x10000, 0x1000
    if self.target[:2] == (9,5): lds_size_per_cu = self.iface.props["lds_size_in_kb"] << 10
    vgpr_size_per_cu = 0x60000 if self.target in {(11,0,0), (11,0,1), (12,0,0), (12,0,1)} else \
                       0x80000 if (self.target[:2]) in {(9,4), (9,5)} or self.target in {(9,0,8), (9,0,10)} else 0x40000
    wg_data_size = round_up((vgpr_size_per_cu + sgrp_size_per_cu + lds_size_per_cu + hwreg_size_per_cu) * (self.max_cu_id + 1), mmap.PAGESIZE)
    ctl_stack_size = round_up(12 * (self.max_cu_id + 1) * (self.max_wave_id + 1) + 8 + 40, mmap.PAGESIZE) if self.target >= (10,1,0) else \
                     round_up((self.max_wave_id + 1) * 8 + 8 + 40, mmap.PAGESIZE)
    debug_memory_size = round_up((self.max_cu_id + 1 if self.target >= (10,1,0) else 1) * (self.max_wave_id + 1) * 32, 64)
    if self.target[0] == 10: ctl_stack_size = min(ctl_stack_size, 0x7000)

    self.soc = importlib.import_module(f"tinygrad.runtime.autogen.am.{({9: 'vega10', 10: 'navi10', 11: 'soc21', 12: 'soc24'}[self.target[0]])}")
    self.pm4 = importlib.import_module(f"tinygrad.runtime.autogen.am.pm4_{'nv' if self.target[0] >= 10 else 'soc15'}")
    self.sdma = import_module('sdma', min(self.iface.ip_versions[am.SDMA0_HWIP], (6, 0, 0)))
    self.gc = AMDIP('gc', self.iface.ip_versions[am.GC_HWIP], self.iface.ip_offsets[am.GC_HWIP])

    # Define the regCOMPUTE_CURRENT_LOGIC_XCC_ID register, which is missing from the asic_regs files.
    if self.target[:2] in {(9,4),(9,5)}: self.regCOMPUTE_CURRENT_LOGIC_XCC_ID = AMDReg("regCOMPUTE_CURRENT_LOGIC_XCC_ID", 0xe25, 0, {}, self.gc.bases)

    nbio_name = 'nbio' if self.target[0] < 12 else 'nbif'
    nbio_pad = (0,) if self.target[0] == 9 else ()
    self.nbio = AMDIP(nbio_name, self.iface.ip_versions[am.NBIF_HWIP], nbio_pad+self.iface.ip_offsets[am.NBIF_HWIP])

    self.compute_queue = self.create_queue(kfd.KFD_IOC_QUEUE_TYPE_COMPUTE, 0x2000 if self.is_usb() else (16 << 20), eop_buffer_size=0x1000,
      ctx_save_restore_size=0 if self.is_am() else wg_data_size + ctl_stack_size, ctl_stack_size=ctl_stack_size, debug_memory_size=debug_memory_size)

    max_copy_size = 0x40000000 if self.iface.ip_versions[am.SDMA0_HWIP][0] >= 5 else 0x400000
    self.sdma_queue = self.create_queue(kfd.KFD_IOC_QUEUE_TYPE_SDMA, 0x200 if self.is_usb() else (16 << 20))

    super().__init__(device, AMDAllocator(self), AMDLLVMRenderer(self.arch) if AMD_LLVM else AMDRenderer(self.arch),
                     AMDLLVMCompiler(self.arch) if AMD_LLVM else HIPCompiler(self.arch), functools.partial(AMDProgram, self),
                     AMDSignal, functools.partial(AMDComputeQueue, self), functools.partial(AMDCopyQueue, self, max_copy_size=max_copy_size),
                     kernargs_size=(8 << 10) if self.is_usb() else (16 << 20), sigalloc_size=0x100 if self.is_usb() else 0x1000)

    # Scratch setup
    self.max_private_segment_size = 0
    self._ensure_has_local_memory(128) # set default scratch size to 128 bytes per thread

    # XCC setup
    self.xcc_sync: tuple[AMDSignal, AMDSignal]|None = None
    if self.xccs > 1:
      self.xcc_sync_area = self.allocator.alloc(0x1000, BufferSpec(nolru=True, cpu_access=True))
      self.xcc_sync = (AMDSignal(base_buf=self.xcc_sync_area), AMDSignal(base_buf=self.xcc_sync_area.offset(256)))
      AMDComputeQueue(self).xcc_config().submit(self)

    # SQTT is disabled by default because of runtime overhead and big file sizes (~200mb to Tensor.full() two 4096x4096 tensors and matmul them)
    self.sqtt_enabled = PROFILE and bool(getenv("SQTT", 0))
    if self.sqtt_enabled:
      if self.arch != 'gfx1100': raise RuntimeError('SQ Thread Tracing is only supported on 7900XTX')
      if not self.is_am() and (ppfeaturemask:=int(FileIOInterface('/sys/module/amdgpu/parameters/ppfeaturemask', os.O_RDONLY).read(), 16))&0x8000:
        raise RuntimeError("SQTT can't be enabled because of hardware bug, to workaround either use AMD_IFACE=PCI or add "
                           f"ppfeaturemask={(ppfeaturemask&~0x8000):#x} (current {ppfeaturemask=:#x} & ~PP_GFXOFF_MASK) to amdgpu module parameters\n"
                           "For more information read https://github.com/tinygrad/tinygrad/blob/master/extra/sqtt/README.md")
      SQTT_BUFFER_SIZE = getenv("SQTT_BUFFER_SIZE", 256) # in mb, per shader engine
      SQTT_NUM = self.iface.props['array_count'] // self.iface.props['simd_arrays_per_engine']
      self.sqtt_buffers = [self.allocator.alloc(SQTT_BUFFER_SIZE*1024*1024, BufferSpec(cpu_access=True, nolru=True)) for _ in range(SQTT_NUM)]
      self.sqtt_itrace_se_mask = getenv("SQTT_ITRACE_SE_MASK", 2) # -1 enable all, 0 disable all, >0 bitmask for where to enable instruction tracing
      self.cmd_id = 0
      AMDComputeQueue(self).sqtt_start(self.sqtt_buffers, self.sqtt_itrace_se_mask).submit(self)

  def create_queue(self, queue_type, ring_size, ctx_save_restore_size=0, eop_buffer_size=0, ctl_stack_size=0, debug_memory_size=0):
    ring = self.iface.alloc(ring_size, uncached=True, cpu_access=True)
    gart = self.iface.alloc(0x100, uncached=True, cpu_access=True)

    cwsr_buffer_size = round_up((ctx_save_restore_size + debug_memory_size) * self.iface.props.get('num_xcc', 1), mmap.PAGESIZE)
    cwsr_buffer = self.iface.alloc(cwsr_buffer_size) if ctx_save_restore_size else None
    eop_buffer = self.iface.alloc(eop_buffer_size) if eop_buffer_size else None

    return AMDQueueDesc.multi(*(self.iface.create_queue(queue_type, ring, gart, eop_buffer=eop_buffer, cwsr_buffer=cwsr_buffer, xcc_id=xcc_id,
                                                            ctx_save_restore_size=ctx_save_restore_size, ctl_stack_size=ctl_stack_size)
                                for xcc_id in range(self.xccs if queue_type == kfd.KFD_IOC_QUEUE_TYPE_COMPUTE else 1)))

  def _ensure_has_local_memory(self, required):
    if self.max_private_segment_size >= required: return

    # <gfx103 requires alignment of 1024, >=gfx11 requires 256
    wave_scratch_len = round_up(((self.max_wave_id + 1) * required), 256 if self.target >= (11,0,0) else 1024)

    scratch_size = (self.max_cu_id+1)*self.iface.props['max_slots_scratch_cu']*wave_scratch_len # per xcc
    self.scratch, ok = self._realloc(getattr(self, 'scratch', None), scratch_size*self.xccs)
    if ok:
      engines = self.iface.props['array_count'] // self.iface.props['simd_arrays_per_engine']
      waves = wave_scratch_len // (256 if self.target >= (11,0,0) else 1024)
      # >=gfx11 wavesize is per SE
      wavesize = scratch_size // ((wave_scratch_len * engines) if self.target >= (11,0,0) else wave_scratch_len)
      self.tmpring_size = waves << 12 | wavesize
      self.max_private_segment_size = required

  def invalidate_caches(self):
    AMDComputeQueue(self).memory_barrier().signal(self.timeline_signal, self.next_timeline()).submit(self)
    self.synchronize()

  def on_device_hang(self): self.iface.on_device_hang()

  def _at_profile_finalize(self):
    if self.sqtt_enabled:
      wptrs_buf = self.allocator.alloc(round_up(len(self.sqtt_buffers), 0x1000), BufferSpec(cpu_access=True, nolru=True))
      wptrs = to_mv(wptrs_buf.va_addr, wptrs_buf.size)
      AMDComputeQueue(self).sqtt_stop(len(self.sqtt_buffers), wptrs_buf).signal(self.timeline_signal, self.next_timeline()).submit(self)
      self.synchronize()
      if DEBUG>=2: print('Saving SQTT in profile...')
      for i,buf0 in enumerate(self.sqtt_buffers):
        wptr = ((struct.unpack('<I', wptrs[i*4:i*4+4])[0] & 0x1FFFFFFF) - ((buf0.va_addr//32) & 0x1FFFFFFF)) * 32
        if DEBUG>=2: print(f'Se {i} blob size {wptr:#x}')
        assert wptr >= 0 and wptr <= buf0.size, f"{wptr} > {buf0.size}, should never happen"
        # When sqtt buffer overflows, wptr stops at the last dword
        if wptr >= buf0.size-32: print(f"WARNING: SQTT BUFFER IS FULL (SE {i})! INCREASE SQTT BUFFER SIZE WITH SQTT_BUFFER_SIZE=X (in MB)")
        self.allocator._copyout(sqtt_buf:=memoryview(bytearray(wptr)), buf0)
        Compiled.profile_events += [ProfileSQTTEvent(self.device, i, bytes(sqtt_buf), bool((self.sqtt_itrace_se_mask >> i) & 0b1))]
    super()._at_profile_finalize()
