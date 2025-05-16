import ctypes, time
from test.mockgpu.gpu import VirtGPU
from tinygrad.helpers import to_mv, init_c_struct_t, mv_address
import tinygrad.runtime.autogen.amd_gpu as amd_gpu

SDMA_MAX_COPY_SIZE = 0x400000

PACKET3_SET_SH_REG_START = 0x2c00
SUB = PACKET3_SET_SH_REG_START - amd_gpu.GC_BASE__INST0_SEG0

regCOMPUTE_PGM_LO = 0x1bac - SUB
regCOMPUTE_USER_DATA_0 = 0x1be0 - SUB
regCOMPUTE_NUM_THREAD_X = 0x1ba7 - SUB

CACHE_FLUSH_AND_INV_TS_EVENT = 0x14

WAIT_REG_MEM_FUNCTION_ALWAYS = 0
WAIT_REG_MEM_FUNCTION_EQ = 3 # ==
WAIT_REG_MEM_FUNCTION_GEQ = 5 # >=

REMU_PATHS = ["extra/remu/target/release/libremu.so", "libremu.so", "/usr/local/lib/libremu.so",
              "extra/remu/target/release/libremu.dylib", "libremu.dylib", "/usr/local/lib/libremu.dylib", "/opt/homebrew/lib/libremu.dylib"]
def _try_dlopen_remu():
  for path in REMU_PATHS:
    try:
      remu = ctypes.CDLL(path)
      remu.run_asm.restype = ctypes.c_int32
      remu.run_asm.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
        ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
    except OSError: pass
    else: return remu
  print("Could not find libremu.so")
  return None
remu = _try_dlopen_remu()

def create_sdma_packets():
  # TODO: clean up this, if we want to keep it
  structs = {}
  for name,pkt in [(name,s) for name,s in amd_gpu.__dict__.items() if name.startswith("struct_SDMA_PKT_") and name.endswith("_TAG")]:
    names = set()
    fields = []
    for pkt_fields in pkt._fields_:
      if not pkt_fields[0].endswith("_UNION"): fields.append(pkt_fields)
      else:
        assert pkt_fields[1]._fields_[0][0] == '_0'
        for union_fields in pkt_fields[1]._fields_[0][1]._fields_:
          fname = union_fields[0]
          if fname in names: fname = pkt_fields[0]+fname
          names.add(fname)
          # merge together 64-bit fields, otherwise just append them
          if fname.endswith("_63_32") and fields[-1][0].endswith("_31_0"): fields[-1] = tuple([fname[:-6], ctypes.c_ulong, 64])
          else: fields.append(tuple([fname, *union_fields[1:]]))
    new_name = name[16:-4].lower()
    structs[new_name] = init_c_struct_t(tuple(fields))
    assert ctypes.sizeof(structs[new_name]) == ctypes.sizeof(pkt), f"{ctypes.sizeof(structs[new_name])} != {ctypes.sizeof(pkt)}"
  return type("SDMA_PKTS", (object, ), structs)
sdma_pkts = create_sdma_packets()

class AMDQueue:
  def __init__(self, base, size, rptr, wptr):
    self.queue, self.size = to_mv(base, size).cast("I"), size
    self.rptr = to_mv(rptr, 8).cast("Q")
    self.wptr = to_mv(wptr, 8).cast("Q")

class PM4Executor(AMDQueue):
  def __init__(self, gpu, base, size, rptr, wptr):
    self.gpu = gpu
    super().__init__(base, size, rptr, wptr)

  def _next_dword(self):
    x = self.queue[self.rptr[0] % (self.size // 4)]
    self.rptr[0] += 1
    return x

  def execute(self):
    while self.rptr[0] < self.wptr[0]:
      cont = True
      header = self._next_dword()
      packet_type = header >> 30
      op = (header >> 8) & 0xFF
      n = (header >> 16) & 0x3FFF
      assert packet_type == 3, "Can parse only packet3"
      if op == amd_gpu.PACKET3_SET_SH_REG: self._exec_set_sh_reg(n)
      elif op == amd_gpu.PACKET3_ACQUIRE_MEM: self._exec_acquire_mem(n)
      elif op == amd_gpu.PACKET3_RELEASE_MEM: self._exec_release_mem(n)
      elif op == amd_gpu.PACKET3_WAIT_REG_MEM: cont = self._exec_wait_reg_mem(n)
      elif op == amd_gpu.PACKET3_DISPATCH_DIRECT: self._exec_dispatch_direct(n)
      elif op == amd_gpu.PACKET3_INDIRECT_BUFFER: self._exec_indirect_buffer(n)
      elif op == amd_gpu.PACKET3_EVENT_WRITE: self._exec_event_write(n)
      else: raise RuntimeError(f"PM4: Unknown opcode: {op}")
      if not cont: return

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
    else: raise RuntimeError(f"Unknown {mem_data_sel=}")

  def _exec_wait_reg_mem(self, n):
    assert n == 5
    info = self._next_dword()
    addr_lo = self._next_dword()
    addr_hi = self._next_dword()
    val = self._next_dword()
    _ = self._next_dword() # mask
    _ = self._next_dword() # timeout

    mem_function = (info >> 0) & 0b111
    mem_space = (info >> 4) & 0b1
    _ = (info >> 6) & 0b1 # memop
    _ = (info >> 8) & 0b1 # mem_engine

    if mem_space == 0: mval = val
    elif mem_space == 1: mval = to_mv(addr_lo + (addr_hi << 32), 4).cast('I')[0]

    if mem_function == WAIT_REG_MEM_FUNCTION_GEQ: can_cont = bool(mval >= val)
    elif mem_function == WAIT_REG_MEM_FUNCTION_EQ: can_cont = bool(mval == val)
    else: raise RuntimeError(f"Do not support {mem_function=}")

    if not can_cont: self.rptr[0] = self.rptr[0] - 7 # revert this packet, need to wait again
    return can_cont

  def _exec_set_sh_reg(self, n):
    reg = self._next_dword()
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

    prg_sz = 0
    for st,sz in self.gpu.mapped_ranges:
      if st <= prg_addr < st+sz: prg_sz = sz - (prg_addr - st)

    assert prg_sz > 0, "Invalid prg ptr (not found in mapped ranges)"
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
    PM4Executor(self.gpu, (addr_hi << 32) | addr_lo, buf_sz * 4, mv_address(rptr), mv_address(wptr)).execute()
    assert rptr[0] == wptr[0], "not everything executed in amdgpu"

  def _exec_event_write(self, n):
    assert n == 0
    _ = self._next_dword() # do not emulate events for now

class SDMAExecutor(AMDQueue):
  def __init__(self, gpu, base, size, rptr, wptr):
    self.gpu, self.base = gpu, base
    super().__init__(base, size, rptr, wptr)

  def execute(self):
    while self.rptr[0] < self.wptr[0]:
      cont = True
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
      if not cont: return

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

class AMDGPU(VirtGPU):
  def __init__(self, gpuid):
    super().__init__(gpuid)
    self.mapped_ranges = set()
    self.queues = []

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
gfx_target_version 110000
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
