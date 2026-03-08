import ctypes, time
from test.mockgpu.gpu import VirtGPU
from test.mockgpu.helpers import _try_dlopen_remu
from tinygrad.helpers import getbits, to_mv, getenv
from tinygrad.runtime.support import c

MOCKGPU_ARCH = getenv("MOCKGPU_ARCH", "rdna3")
GFX_TARGET_VERSION = {"rdna3": 110000, "rdna4": 120000}[MOCKGPU_ARCH]
import tinygrad.runtime.autogen.amd_gpu as amd_gpu, tinygrad.runtime.autogen.am.pm4_nv as pm4

SDMA_MAX_COPY_SIZE = 0x400000

regCOMPUTE_PGM_LO = 0x1bac + amd_gpu.GC_BASE__INST0_SEG0
regCOMPUTE_PGM_RSRC2 = 0x1bb3 + amd_gpu.GC_BASE__INST0_SEG0
regCOMPUTE_TMPRING_SIZE = 0x1bb8 + amd_gpu.GC_BASE__INST0_SEG0
regCOMPUTE_USER_DATA_0 = 0x1be0 + amd_gpu.GC_BASE__INST0_SEG0
regCOMPUTE_NUM_THREAD_X = 0x1ba7 + amd_gpu.GC_BASE__INST0_SEG0
regGRBM_GFX_INDEX = 0x2200 + amd_gpu.GC_BASE__INST0_SEG1
regSQ_THREAD_TRACE_BUF0_BASE = 0x39e8 + amd_gpu.GC_BASE__INST0_SEG1
regSQ_THREAD_TRACE_BUF0_SIZE = 0x39e9 + amd_gpu.GC_BASE__INST0_SEG1
regSQ_THREAD_TRACE_WPTR = 0x39ef + amd_gpu.GC_BASE__INST0_SEG1
regSQ_THREAD_TRACE_STATUS = 0x39f4 + amd_gpu.GC_BASE__INST0_SEG1
regCP_PERFMON_CNTL = 0x3808 + amd_gpu.GC_BASE__INST0_SEG1
regCPG_PERFCOUNTER1_LO = 0x3000 + amd_gpu.GC_BASE__INST0_SEG1
regGUS_PERFCOUNTER_HI = 0x3643 + amd_gpu.GC_BASE__INST0_SEG1

class SQTT_EVENTS:
  THREAD_TRACE_FINISH = 0x00000037

CACHE_FLUSH_AND_INV_TS_EVENT = 0x14

WAIT_REG_MEM_FUNCTION_ALWAYS = 0
WAIT_REG_MEM_FUNCTION_EQ  = 3 # ==
WAIT_REG_MEM_FUNCTION_NEQ = 4 # !=
WAIT_REG_MEM_FUNCTION_GEQ = 5 # >=

remu = _try_dlopen_remu()

def create_sdma_packets():
  # TODO: clean up this, if we want to keep it
  structs = {}
  for name,pkt in [(name,s) for name,s in amd_gpu.__dict__.items() if name.startswith("rocr_AMD_SDMA_PKT_") and name.endswith("_TAG")]:
    names = set()
    fields = []
    for pkt_fields in pkt._real_fields_:
      if not pkt_fields[0].endswith("_UNION"): fields.append(pkt_fields)
      else:
        for union_fields in pkt_fields[1]._real_fields_[:-1]:
          fname = union_fields[0]
          if fname in names: fname = pkt_fields[0]+fname
          names.add(fname)
          # merge together 64-bit fields, otherwise just append them
          if fname.endswith("_63_32") and fields[-1][0].endswith("_31_0"): fields[-1] = (fname[:-6], ctypes.c_ulong, fields[-1][2], 64, 0)
          else: fields.append((fname, union_fields[1], union_fields[2] + pkt_fields[2], *union_fields[3:]))
    new_name = name[18:-4].lower()
    structs[new_name] = c.init_c_struct_t(ctypes.sizeof(pkt), tuple(fields))
  return type("SDMA_PKTS", (object, ), structs)
sdma_pkts = create_sdma_packets()

class AMDQueue:
  def __init__(self, base, size, rptr, wptr):
    self.queue, self.size = to_mv(base, size).cast("I"), size
    self.rptr = to_mv(rptr, 8).cast("Q") if isinstance(rptr, int) else rptr
    self.wptr = to_mv(wptr, 8).cast("Q") if isinstance(wptr, int) else wptr

  @property
  def executing(self): return self.rptr[0] < self.wptr[0]

class PM4Executor(AMDQueue):
  def __init__(self, gpu, base, size, rptr, wptr):
    self.gpu = gpu
    self.ib_executor: PM4Executor|None = None
    super().__init__(base, size, rptr, wptr)

  def _next_dword(self):
    x = self.queue[self.rptr[0] % (self.size // 4)]
    self.rptr[0] += 1
    return x

  @property
  def executing(self): return self.rptr[0] < self.wptr[0] or self.ib_executor is not None

  def execute(self):
    prev_rptr, executed_in_ib, cont = self.rptr[0], 0, True
    while self.executing and cont:
      if self.ib_executor is not None:
        executed_in_ib += self.ib_executor.execute()
        if self.ib_executor.executing: break
        self.ib_executor = None
        continue # this continue is needed if PACKET3_INDIRECT_BUFFER is the last packet and rptr == wptr
      header = self._next_dword()
      packet_type = header >> 30
      op = (header >> 8) & 0xFF
      n = (header >> 16) & 0x3FFF
      assert packet_type == 3, "Can parse only packet3"
      if op == amd_gpu.PACKET3_SET_SH_REG: self._exec_set_reg(n, pm4.PACKET3_SET_SH_REG_START)
      elif op == amd_gpu.PACKET3_SET_UCONFIG_REG: self._exec_set_reg(n, pm4.PACKET3_SET_UCONFIG_REG_START)
      elif op == amd_gpu.PACKET3_ACQUIRE_MEM: self._exec_acquire_mem(n)
      elif op == amd_gpu.PACKET3_RELEASE_MEM: self._exec_release_mem(n)
      elif op == amd_gpu.PACKET3_COPY_DATA: self._exec_copy_data(n)
      elif op == amd_gpu.PACKET3_WAIT_REG_MEM: cont = self._exec_wait_reg_mem(n)
      elif op == amd_gpu.PACKET3_DISPATCH_DIRECT: self._exec_dispatch_direct(n)
      elif op == amd_gpu.PACKET3_INDIRECT_BUFFER: self._exec_indirect_buffer(n)
      elif op == amd_gpu.PACKET3_EVENT_WRITE: self._exec_event_write(n)
      else: raise RuntimeError(f"PM4: Unknown opcode: {op}")
    return (self.rptr[0] - prev_rptr) + executed_in_ib

  def _exec_acquire_mem(self, n):
    assert n == 6
    for _ in range(7): self._next_dword() # TODO: implement

  def _exec_release_mem(self, n):
    assert n == 6
    mem_event_type = (self._next_dword() >> 0) & 0xff
    selectors = self._next_dword()
    mem_data_sel = (selectors >> 29) & 0b111
    # int_sel = (selectors >> 24) & 0b11
    # mem_dst_sel = (selectors >> 16) & 0b1
    addr_lo = self._next_dword()
    addr_hi = self._next_dword()
    val_lo = self._next_dword()
    val_hi = self._next_dword()
    val = val_lo + (val_hi << 32)
    _ = self._next_dword() # ev

    ptr = to_mv(addr_lo + (addr_hi << 32), 8)
    if mem_data_sel == 1 or mem_data_sel == 2: ptr.cast('Q')[0] = val
    elif mem_data_sel == 3:
      if mem_event_type == CACHE_FLUSH_AND_INV_TS_EVENT: ptr.cast('Q')[0] = int(time.perf_counter() * 1e8)
      else: raise RuntimeError(f"Unknown {mem_data_sel=} {mem_event_type=}")
    elif mem_data_sel == 0: pass # no write
    else: raise RuntimeError(f"Unknown {mem_data_sel=}")

  def _exec_copy_data(self, n):
    assert n == 4
    copy_data_flags = self._next_dword()
    src_addr_lo = self._next_dword()
    _src_addr_hi = self._next_dword()
    dst_addr_lo = self._next_dword()
    dst_addr_hi = self._next_dword()
    assert copy_data_flags in {0x100204, 0x000204}, hex(copy_data_flags) # better fail than silently do the wrong thing
    to_mv(dst_addr_hi<<32|dst_addr_lo, 4).cast('I')[0] = self.gpu.regs[src_addr_lo]

  def _exec_wait_reg_mem(self, n):
    assert n == 5
    info = self._next_dword()
    addr_lo = self._next_dword()
    addr_hi = self._next_dword()
    val = self._next_dword()
    mask = self._next_dword()
    _ = self._next_dword() # timeout

    mem_function = (info >> 0) & 0b111
    mem_space = (info >> 4) & 0b1
    mem_op = (info >> 6) & 0b1
    _ = (info >> 8) & 0b1 # mem_engine

    if mem_space == 0 and mem_op == 1: mval = val # hack for memory barrier, should properly handle (req_req, reg_done)
    elif mem_space == 0: mval = self.gpu.regs[addr_hi<<32|addr_lo]
    elif mem_space == 1: mval = to_mv(addr_lo + (addr_hi << 32), 4).cast('I')[0]

    mval &= mask

    if mem_function == WAIT_REG_MEM_FUNCTION_GEQ: can_cont = bool(mval >= val)
    elif mem_function == WAIT_REG_MEM_FUNCTION_NEQ: can_cont = bool(mval != val)
    elif mem_function == WAIT_REG_MEM_FUNCTION_EQ: can_cont = bool(mval == val)
    else: raise RuntimeError(f"Do not support {mem_function=}")

    if not can_cont: self.rptr[0] = self.rptr[0] - 7 # revert this packet, need to wait again
    return can_cont

  def _exec_set_reg(self, n, off):
    reg = off + self._next_dword()
    for i in range(n):
      self.gpu.regs[reg] = self._next_dword()
      reg += 1

  def _exec_dispatch_direct(self, n):
    assert n == 3
    gl = [self._next_dword() for _ in range(3)]
    _ = self._next_dword() # flags

    prg_addr = (self.gpu.regs[regCOMPUTE_PGM_LO] + (self.gpu.regs[regCOMPUTE_PGM_LO + 1] << 32)) << 8
    args_addr = self.gpu.regs[regCOMPUTE_USER_DATA_0] + (self.gpu.regs[regCOMPUTE_USER_DATA_0 + 1] << 32)
    lc = [self.gpu.regs[i] for i in range(regCOMPUTE_NUM_THREAD_X, regCOMPUTE_NUM_THREAD_X+3)]
    rsrc2 = self.gpu.regs[regCOMPUTE_PGM_RSRC2]

    prg_sz = 0
    for st,sz in self.gpu.mapped_ranges:
      if st <= prg_addr < st+sz: prg_sz = sz - (prg_addr - st)

    # Get scratch size from COMPUTE_TMPRING_SIZE register
    # For gfx11: WAVESIZE = ceildiv(64 * size_per_thread, 256), so size_per_thread ≈ WAVESIZE * 256 / 64 = WAVESIZE * 4
    try: tmpring_size = self.gpu.regs[regCOMPUTE_TMPRING_SIZE]
    except KeyError: tmpring_size = 0
    wavesize = (tmpring_size >> 12) & 0x3FFF  # WAVESIZE field is bits 12:25 for gfx11
    scratch_size = wavesize * 4  # This gives the scratch size per thread (lane)

    assert prg_sz > 0, "Invalid prg ptr (not found in mapped ranges)"
    # Pass valid memory ranges, rsrc2, scratch_size and arch to Python emulator
    if hasattr(remu, 'valid_mem_ranges'): remu.valid_mem_ranges = self.gpu.mapped_ranges
    if hasattr(remu, 'rsrc2'): remu.rsrc2 = rsrc2
    if hasattr(remu, 'scratch_size'): remu.scratch_size = scratch_size
    if hasattr(remu, 'arch'): remu.arch = self.gpu.arch
    err = remu.run_asm(prg_addr, prg_sz, *gl, *lc, args_addr)
    if err != 0: raise RuntimeError("remu does not support the new instruction introduced in this kernel")

  def _exec_indirect_buffer(self, n):
    addr_lo = self._next_dword()
    addr_hi = self._next_dword()
    buf_sz = self._next_dword() & (0x7fffff)

    rptr = memoryview(bytearray(8)).cast('Q')
    wptr = memoryview(bytearray(8)).cast('Q')
    rptr[0] = 0
    wptr[0] = buf_sz
    self.ib_executor = PM4Executor(self.gpu, (addr_hi << 32) | addr_lo, buf_sz * 4, rptr, wptr)

  def _exec_event_write(self, n):
    assert n == 0
    event_dw = self._next_dword()
    match (event_dw & 0xFF): # event type
      case SQTT_EVENTS.THREAD_TRACE_FINISH:
        old_idx = self.gpu.regs.grbm_index
        for se in range(self.gpu.regs.n_se):
          self.gpu.regs.grbm_index = 0b011 << 29 | se << 16 # select se, broadcast sa and instance
          self.gpu.regs[regSQ_THREAD_TRACE_STATUS] = 1 << 12 # FINISH_PENDING==0 FINISH_DONE==1 BUSY==0
          buf = ((self.gpu.regs[regSQ_THREAD_TRACE_BUF0_SIZE]&0xf)<<32|self.gpu.regs[regSQ_THREAD_TRACE_BUF0_BASE])<<12 # per page addressing
          fake_used = 0x1000 # fake one page long trace
          self.gpu.regs[regSQ_THREAD_TRACE_WPTR] = ((buf+fake_used)//32) & 0x1FFFFFFF
        self.gpu.regs.grbm_index = old_idx
      case _: pass # NOTE: for now most events aren't emulated

class SDMAExecutor(AMDQueue):
  def __init__(self, gpu, base, size, rptr, wptr):
    self.gpu, self.base = gpu, base
    super().__init__(base, size, rptr, wptr)

  def execute(self):
    prev_rptr, cont = self.rptr[0], True
    while self.executing and cont:
      header = self.queue[(self.rptr[0] // 4) % (self.size // 4)]
      op = (header >> 0) & 0xff
      if op == 0: self.rptr[0] += 4
      elif op == amd_gpu.SDMA_OP_FENCE: self._execute_fence()
      elif op == amd_gpu.SDMA_OP_TRAP: self._execute_trap()
      elif op == amd_gpu.SDMA_OP_POLL_REGMEM: cont = self._execute_poll_regmem()
      elif op == amd_gpu.SDMA_OP_GCR: self._execute_gcr()
      elif op == amd_gpu.SDMA_OP_COPY: self._execute_copy()
      elif op == amd_gpu.SDMA_OP_TIMESTAMP: self._execute_timestamp()
      else: raise RuntimeError(f"Unknown SDMA op {op}")
    return self.rptr[0] - prev_rptr

  def _execute_fence(self):
    struct = sdma_pkts.fence.from_address(self.base + self.rptr[0] % self.size)
    to_mv(struct.addr, 8).cast('Q')[0] = struct.data
    self.rptr[0] += ctypes.sizeof(struct)

  def _execute_trap(self):
    struct = sdma_pkts.trap.from_address(self.base + self.rptr[0] % self.size)
    self.rptr[0] += ctypes.sizeof(struct)

  def _execute_poll_regmem(self):
    struct = sdma_pkts.poll_regmem.from_address(self.base + self.rptr[0] % self.size)

    if struct.mem_poll == 0: mval = struct.value & struct.mask
    elif struct.mem_poll == 1: mval = to_mv(struct.addr, 4).cast('I')[0] & struct.mask

    if struct.func == WAIT_REG_MEM_FUNCTION_GEQ: can_cont = bool(mval >= struct.value)
    elif struct.func == WAIT_REG_MEM_FUNCTION_EQ: can_cont = bool(mval == struct.value)
    elif struct.func == WAIT_REG_MEM_FUNCTION_ALWAYS: can_cont = True
    else: raise RuntimeError(f"Do not support {struct.func=}")

    if not can_cont: return False

    self.rptr[0] += ctypes.sizeof(struct)
    return True

  def _execute_timestamp(self):
    struct = sdma_pkts.timestamp.from_address(self.base + self.rptr[0] % self.size)

    mem = to_mv(struct.addr, 8).cast('Q')
    mem[0] = int(time.perf_counter() * 1e8)

    self.rptr[0] += ctypes.sizeof(struct)

  def _execute_gcr(self):
    struct = sdma_pkts.gcr.from_address(self.base + self.rptr[0] % self.size)
    self.rptr[0] += ctypes.sizeof(struct)

  def _execute_copy(self):
    struct = sdma_pkts.copy_linear.from_address(self.base + self.rptr[0] % self.size)
    count_cnt = to_mv(self.base + self.rptr[0] + 4, 4).cast('I')[0] & 0x3FFFFFFF
    ctypes.memmove(struct.dst_addr, struct.src_addr, count_cnt + 1)
    self.rptr[0] += ctypes.sizeof(struct)

class AMDGPURegisters:
  def __init__(self, n_se:int=6):
    self.n_se = n_se
    self.grbm_index = 0b111 << 0x1d # all broadcast. NOTE: only per-se register emulation is currently supported
    self.regs: dict[tuple[int, int], int] = {}
  def __getitem__(self, addr:int) -> int:
    if addr == regGRBM_GFX_INDEX: return self.grbm_index
    if regCPG_PERFCOUNTER1_LO < addr < regGUS_PERFCOUNTER_HI:
      assert self.regs[(regCP_PERFMON_CNTL, 0)] == 0x401, "read mode should be enabled"
      return addr << 16 | self.grbm_index
    return self.regs[(addr, getbits(self.grbm_index, 16, 23))]
  def __setitem__(self, addr:int, val:int):
    if addr == regGRBM_GFX_INDEX: self.grbm_index = val
    if getbits(self.grbm_index, 31, 31):
      for se in range(self.n_se): self.regs[(addr, se)] = val
    else:
      self.regs[(addr, getbits(self.grbm_index, 16, 23))] = val

class AMDGPU(VirtGPU):
  def __init__(self, gpuid):
    super().__init__(gpuid)
    self.regs = AMDGPURegisters()
    self.mapped_ranges = set()
    self.queues = []
    self.arch = MOCKGPU_ARCH

  def map_range(self, vaddr, size): self.mapped_ranges.add((vaddr, size))
  def unmap_range(self, vaddr, size): self.mapped_ranges.remove((vaddr, size))
  def add_pm4_queue(self, base, size, rptr, wptr):
    self.queues.append(PM4Executor(self, base, size, rptr, wptr))
    return len(self.queues) - 1
  def add_sdma_queue(self, base, size, rptr, wptr):
    self.queues.append(SDMAExecutor(self, base, size, rptr, wptr))
    return len(self.queues) - 1

gpu_props = """cpu_cores_count 0
simd_count 192
mem_banks_count 1
caches_count 206
io_links_count 1
p2p_links_count 5
cpu_core_id_base 0
simd_id_base 2147488032
max_waves_per_simd 16
lds_size_in_kb 64
gds_size_in_kb 0
num_gws 64
wave_front_size 32
array_count 12
simd_arrays_per_engine 2
cu_per_simd_array 8
simd_per_cu 2
max_slots_scratch_cu 32
gfx_target_version {gfx_target_version}
vendor_id 4098
device_id 29772
location_id 34304
domain 0
drm_render_minor {drm_render_minor}
hive_id 0
num_sdma_engines 2
num_sdma_xgmi_engines 0
num_sdma_queues_per_engine 6
num_cp_queues 8
max_engine_clk_fcompute 2482
local_mem_size 0
fw_version 2140
capability 671588992
debug_prop 1495
sdma_fw_version 20
unique_id 11673270660693242239
num_xcc 1
max_engine_clk_ccompute 2400"""
