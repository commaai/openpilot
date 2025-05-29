import ctypes, ctypes.util, time
import tinygrad.runtime.autogen.nv_gpu as nv_gpu
from enum import Enum, auto
from test.mockgpu.gpu import VirtGPU
from tinygrad.helpers import to_mv, init_c_struct_t

def make_qmd_struct_type():
  fields = []
  bits = [(name,dt) for name,dt in nv_gpu.__dict__.items() if name.startswith("NVC6C0_QMDV03_00") and isinstance(dt, tuple)]
  bits += [(name+f"_{i}",dt(i)) for name,dt in nv_gpu.__dict__.items() for i in range(8) if name.startswith("NVC6C0_QMDV03_00") and callable(dt)]
  bits = sorted(bits, key=lambda x: x[1][1])
  for i,(name, data) in enumerate(bits):
    if i > 0 and (gap:=(data[1] - bits[i-1][1][0] - 1)) != 0:  fields.append((f"_reserved{i}", ctypes.c_uint32, gap))
    fields.append((name.replace("NVC6C0_QMDV03_00_", "").lower(), ctypes.c_uint32, data[0]-data[1]+1))
  return init_c_struct_t(tuple(fields))
qmd_struct_t = make_qmd_struct_type()
assert ctypes.sizeof(qmd_struct_t) == 0x40 * 4

try:
  gpuocelot_lib = ctypes.CDLL(ctypes.util.find_library("gpuocelot"))
  gpuocelot_lib.ptx_run.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]  # noqa: E501
except Exception: pass

class SchedResult(Enum): CONT = auto(); YIELD = auto() # noqa: E702

class GPFIFO:
  def __init__(self, token, base, entries_cnt):
    self.token, self.base, self.entries_cnt = token, base, entries_cnt
    self.gpfifo = to_mv(self.base, self.entries_cnt * 8).cast("Q")
    self.ctrl = nv_gpu.AmpereAControlGPFifo.from_address(self.base + self.entries_cnt * 8)
    self.state = {}

    # Buf exec state
    self.buf = None
    self.buf_sz = 0
    self.buf_ptr = 0

  def _next_dword(self):
    assert self.buf is not None
    x = self.buf[self.buf_ptr]
    self.buf_ptr += 1
    return x

  def _next_header(self):
    header = self._next_dword()
    typ = (header >> 28) & 0b111
    size = (header >> 16) & 0xFFF
    subc = (header >> 13) & 0x7
    mthd = (header & 0x1FFF) << 2
    return typ, size, subc, mthd

  def _state(self, reg): return self.state[reg]
  def _state64(self, reg): return (self.state[reg] << 32) + self.state[reg + 4]
  def _state64_le(self, reg): return (self.state[reg + 4] << 32) + self.state[reg]

  def _reset_buf_state(self): self.buf, self.buf_ptr = None, 0
  def _set_buf_state(self, gpfifo_entry):
    ptr = ((gpfifo_entry >> 2) & 0x3fffffffff) << 2
    sz = ((gpfifo_entry >> 42) & 0x1fffff) << 2
    self.buf = to_mv(ptr, sz).cast("I")
    self.buf_sz = sz // 4

  def execute(self) -> bool:
    initial_off = self.buf_ptr
    while self.ctrl.GPGet != self.ctrl.GPPut:
      self._set_buf_state(self.gpfifo[self.ctrl.GPGet])

      if not self.execute_buf():
        # Buffer isn't executed fully, check if any progress and report.
        # Do not move GPGet in this case, will continue from the same state next time.
        return self.buf_ptr != initial_off

      self.ctrl.GPGet = (self.ctrl.GPGet + 1) % self.entries_cnt
      self._reset_buf_state()
    return True

  def execute_buf(self) -> bool:
    while self.buf_ptr < self.buf_sz:
      init_off = self.buf_ptr
      _, size, _, mthd = self._next_header()
      cmd_end_off = self.buf_ptr + size

      while self.buf_ptr < cmd_end_off:
        res = self.execute_cmd(mthd)
        if res == SchedResult.YIELD:
          self.buf_ptr = init_off # just revert to the header
          return False
        mthd += 4
    return True

  def execute_qmd(self, qmd_addr):
    qmd = qmd_struct_t.from_address(qmd_addr)
    prg_addr = qmd.program_address_lower + (qmd.program_address_upper << 32)
    const0 = to_mv(qmd.constant_buffer_addr_lower_0 + (qmd.constant_buffer_addr_upper_0 << 32), 0x160).cast('I')
    args_cnt, vals_cnt = const0[80], const0[81]
    args_addr = qmd.constant_buffer_addr_lower_0 + (qmd.constant_buffer_addr_upper_0 << 32) + 0x160
    args = to_mv(args_addr, args_cnt*8).cast('Q')
    vals = to_mv(args_addr + args_cnt*8, vals_cnt*4).cast('I')
    cargs = [ctypes.cast(args[i], ctypes.c_void_p) for i in range(args_cnt)] + [ctypes.cast(vals[i], ctypes.c_void_p) for i in range(vals_cnt)]
    gx, gy, gz = qmd.cta_raster_width, qmd.cta_raster_height, qmd.cta_raster_depth
    lx, ly, lz = qmd.cta_thread_dimension0, qmd.cta_thread_dimension1, qmd.cta_thread_dimension2
    gpuocelot_lib.ptx_run(ctypes.cast(prg_addr, ctypes.c_char_p), args_cnt+vals_cnt, (ctypes.c_void_p*len(cargs))(*cargs), lx, ly, lz, gx, gy, gz, 0)
    if qmd.release0_enable:
      rel0 = to_mv(qmd.release0_address_lower + (qmd.release0_address_upper << 32), 0x10).cast('Q')
      rel0[0] = qmd.release0_payload_lower + (qmd.release0_payload_upper << 32)
      rel0[1] = int(time.perf_counter() * 1e9)
    if qmd.release1_enable:
      rel1 = to_mv(qmd.release1_address_lower + (qmd.release1_address_upper << 32), 0x10).cast('Q')
      rel1[0] = qmd.release1_payload_lower + (qmd.release1_payload_upper << 32)
      rel1[1] = int(time.perf_counter() * 1e9)
    if qmd.dependent_qmd0_enable:
      if qmd.dependent_qmd0_action == 1: self.execute_qmd(qmd.dependent_qmd0_pointer << 8)
      else: raise RuntimeError("unsupported dependent qmd action")

  def execute_cmd(self, cmd) -> SchedResult:
    if cmd == nv_gpu.NVC56F_SEM_EXECUTE: return self._exec_signal()
    elif cmd == nv_gpu.NVC6C0_LAUNCH_DMA: return self._exec_nvc6c0_dma()
    elif cmd == nv_gpu.NVC6B5_LAUNCH_DMA: return self._exec_nvc6b5_dma()
    elif cmd == nv_gpu.NVC6C0_SEND_SIGNALING_PCAS2_B: return self._exec_pcas2()
    elif cmd == 0x0320: return self._exec_load_inline_qmd() # NVC6C0_LOAD_INLINE_QMD_DATA
    else: self.state[cmd] = self._next_dword() # just state update
    return SchedResult.CONT

  def _exec_signal(self) -> SchedResult:
    signal = self._state64_le(nv_gpu.NVC56F_SEM_ADDR_LO)
    val = self._state64_le(nv_gpu.NVC56F_SEM_PAYLOAD_LO)
    flags = self._next_dword()
    typ = (flags >> 0) & 0b111
    timestamp = (flags & (1 << 25)) == (1 << 25)
    if typ == 1:
      to_mv(signal, 8).cast('Q')[0] = val
      if timestamp: to_mv(signal + 8, 8).cast('Q')[0] = int(time.perf_counter() * 1e9)
    elif typ == 3:
      mval = to_mv(signal, 8).cast('Q')[0]
      return SchedResult.CONT if mval >= val else SchedResult.YIELD
    else: raise RuntimeError(f"Unsupported type={typ} in exec wait/signal")
    return SchedResult.CONT

  def _exec_load_inline_qmd(self):
    qmd_addr = self._state64(nv_gpu.NVC6C0_SET_INLINE_QMD_ADDRESS_A) << 8
    assert qmd_addr != 0x0, f"invalid qmd address {qmd_addr}"
    qmd_data = [self._next_dword() for _ in range(0x40)]
    cdata = (ctypes.c_uint32 * len(qmd_data))(*qmd_data)
    ctypes.memmove(qmd_addr, cdata, 0x40 * 4)
    self.execute_qmd(qmd_addr)

  def _exec_nvc6c0_dma(self):
    addr = self._state64(nv_gpu.NVC6C0_OFFSET_OUT_UPPER)
    sz = self._state(nv_gpu.NVC6C0_LINE_LENGTH_IN)
    lanes = self._state(nv_gpu.NVC6C0_LINE_COUNT)
    assert lanes == 1, f"unsupported lanes > 1 in _exec_nvc6c0_dma: {lanes}"
    flags = self._next_dword()
    assert flags == 0x41, f"unsupported flags in _exec_nvc6c0_dma: {flags}"
    typ, dsize, _, mthd = self._next_header()
    assert typ == 6 and mthd == nv_gpu.NVC6C0_LOAD_INLINE_DATA, f"Expected inline data not found after nvc6c0_dma, {typ=} {mthd=}"
    copy_data = [self._next_dword() for _ in range(dsize)]
    assert len(copy_data) * 4 == sz, f"different copy sizes in _exec_nvc6c0_dma: {len(copy_data) * 4} != {sz}"
    cdata = (ctypes.c_uint32 * len(copy_data))(*copy_data)
    ctypes.memmove(addr, cdata, sz)

  def _exec_nvc6b5_dma(self):
    flags = self._next_dword()
    if (flags & 0b11) != 0:
      src = self._state64(nv_gpu.NVC6B5_OFFSET_IN_UPPER)
      dst = self._state64(nv_gpu.NVC6B5_OFFSET_OUT_UPPER)
      sz = self._state(nv_gpu.NVC6B5_LINE_LENGTH_IN)
      assert flags == 0x182, f"unsupported flags in _exec_nvc6b5_dma: {flags}"
      ctypes.memmove(dst, src, sz)
    elif ((flags >> 3) & 0b11) != 0:
      src = to_mv(self._state64(nv_gpu.NVC6B5_SET_SEMAPHORE_A), 0x10).cast('Q')
      val = self._state(nv_gpu.NVC6B5_SET_SEMAPHORE_PAYLOAD)
      src[0] = val
      src[1] = int(time.perf_counter() * 1e9)
    else: raise RuntimeError("unknown nvc6b5_dma flags")

  def _exec_pcas2(self):
    qmd_addr = self._state(nv_gpu.NVC6C0_SEND_PCAS_A) << 8
    typ = self._next_dword()
    if typ == 2 or typ == 9: # schedule
      self.execute_qmd(qmd_addr)

class NVGPU(VirtGPU):
  def __init__(self, gpuid):
    super().__init__(gpuid)
    self.regs = {}
    self.mapped_ranges = set()
    self.queues = []

  def map_range(self, vaddr, size): self.mapped_ranges.add((vaddr, size))
  def unmap_range(self, vaddr, size): self.mapped_ranges.remove((vaddr, size))
  def add_gpfifo(self, base, entries_count):
    self.queues.append(GPFIFO(token:=len(self.queues), base, entries_count))
    return token
  def gpu_uuid(self, sz=16): return self.gpuid.to_bytes(sz, byteorder='big', signed=False)
