# Test to compare Python and Rust RDNA3 emulators by running real tinygrad kernels
import unittest, ctypes
from dataclasses import dataclass

from extra.assembly.amd.emu import WaveState, decode_program, WAVE_SIZE, VCC_LO, EXEC_LO, SCC
from extra.assembly.amd import decode_inst
from extra.assembly.amd.test.helpers import KernelInfo
from extra.assembly.amd.test.bench_emu import REMU_PATH

def set_valid_mem_ranges(ranges): pass  # emu2 doesn't need this

def _is_f32_nan(bits: int) -> bool:
  """Check if 32-bit value is a NaN (exponent all 1s, mantissa non-zero)."""
  return (bits & 0x7f800000) == 0x7f800000 and (bits & 0x007fffff) != 0

def _vals_equal(a: int, b: int) -> bool:
  """Compare two 32-bit values, treating all NaN bit patterns as equal."""
  if a == b: return True
  return _is_f32_nan(a) and _is_f32_nan(b)

@dataclass
class StateSnapshot:
  pc: int
  scc: int
  vcc: int
  exec_mask: int
  sgpr: list[int]
  vgpr: list[list[int]]

  def diff(self, other: 'StateSnapshot', n_lanes: int, arrow: str = " vs ") -> list[str]:
    """Return list of differences between two states."""
    diffs = []
    if self.pc != other.pc: diffs.append(f"pc: {self.pc}{arrow}{other.pc}")
    if self.scc != other.scc: diffs.append(f"scc: {self.scc}{arrow}{other.scc}")
    if self.vcc != other.vcc: diffs.append(f"vcc: 0x{self.vcc:08x}{arrow}0x{other.vcc:08x}")
    if self.exec_mask != other.exec_mask: diffs.append(f"exec: 0x{self.exec_mask:08x}{arrow}0x{other.exec_mask:08x}")
    for i, (a, b) in enumerate(zip(self.sgpr, other.sgpr)):
      # Skip VCC_LO/HI (106/107) and EXEC_LO/HI (126/127) as they alias vcc/exec_mask which are compared separately
      if i in (106, 107, 126, 127): continue
      if not _vals_equal(a, b): diffs.append(f"sgpr[{i}]: 0x{a:08x}{arrow}0x{b:08x}")
    for lane in range(n_lanes):
      for i, (a, b) in enumerate(zip(self.vgpr[lane], other.vgpr[lane])):
        if not _vals_equal(a, b): diffs.append(f"vgpr[{lane}][{i}]: 0x{a:08x}{arrow}0x{b:08x}")
    return diffs

class CStateSnapshot(ctypes.Structure):
  _fields_ = [("pc", ctypes.c_uint32), ("scc", ctypes.c_uint32), ("vcc", ctypes.c_uint32), ("exec_mask", ctypes.c_uint32),
              ("sgpr", ctypes.c_uint32 * 128), ("vgpr", (ctypes.c_uint32 * 256) * 32)]

  def to_snapshot(self) -> StateSnapshot:
    return StateSnapshot(pc=self.pc, scc=self.scc, vcc=self.vcc, exec_mask=self.exec_mask,
                         sgpr=list(self.sgpr), vgpr=[list(self.vgpr[i]) for i in range(32)])

class RustEmulator:
  def __init__(self):
    self.lib = ctypes.CDLL(str(REMU_PATH))
    self.lib.wave_create.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32]
    self.lib.wave_create.restype = ctypes.c_void_p
    self.lib.wave_step.argtypes = [ctypes.c_void_p]
    self.lib.wave_step.restype = ctypes.c_int32
    self.lib.wave_get_snapshot.argtypes = [ctypes.c_void_p, ctypes.POINTER(CStateSnapshot)]
    self.lib.wave_set_sgpr.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32]
    self.lib.wave_set_vgpr.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
    self.lib.wave_init_lds.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    self.lib.wave_free.argtypes = [ctypes.c_void_p]
    self.ctx = None

  def create(self, kernel: bytes, n_lanes: int):
    kernel_buf = (ctypes.c_char * len(kernel)).from_buffer_copy(kernel)
    self.ctx = self.lib.wave_create(ctypes.addressof(kernel_buf), len(kernel), n_lanes)
    self._kernel_buf = kernel_buf

  def step(self) -> int: return self.lib.wave_step(self.ctx)
  def set_sgpr(self, idx: int, val: int): self.lib.wave_set_sgpr(self.ctx, idx, val)
  def set_vgpr(self, lane: int, idx: int, val: int): self.lib.wave_set_vgpr(self.ctx, lane, idx, val)
  def init_lds(self, size: int): self.lib.wave_init_lds(self.ctx, size)

  def get_snapshot(self) -> StateSnapshot:
    snap = CStateSnapshot()
    self.lib.wave_get_snapshot(self.ctx, ctypes.byref(snap))
    return snap.to_snapshot()

  def free(self):
    if self.ctx: self.lib.wave_free(self.ctx); self.ctx = None

class PythonEmulator:
  def __init__(self):
    self.state: WaveState | None = None
    self.program: dict | None = None
    self.vmem_buf = None
    self.lds_buf = None
    self.kernel_buf = None  # Keep kernel bytes alive
    self.lib_addr = 0  # Base address of kernel code

  def create(self, kernel: bytes, n_lanes: int):
    import ctypes
    from tinygrad.device import Buffer, BufferSpec
    from tinygrad.dtype import dtypes
    # Store kernel in a ctypes buffer so generic instructions can read from vmem at actual PC address
    self.kernel_buf = (ctypes.c_char * len(kernel)).from_buffer_copy(kernel)
    self.lib_addr = ctypes.addressof(self.kernel_buf)
    # Remap program dict to use actual addresses (like run_asm does)
    program_raw = decode_program(kernel)
    self.program = {self.lib_addr + offset: val for offset, val in program_raw.items()}
    self.state = WaveState(n_lanes)
    self.state.pc = self.lib_addr  # Set PC to code base address
    self.vmem_buf = Buffer('CPU', 1 << 40, dtypes.uint32, options=BufferSpec(external_ptr=0)).ensure_allocated()
    self.lds_buf = Buffer('CPU', 65536 // 4, dtypes.uint32).ensure_allocated()

  def step(self) -> int:
    import ctypes
    assert self.program is not None and self.state is not None
    pc = self.state.pc
    if pc == 0xFFFFFFFFFFFFFFFF or pc not in self.program: return -1
    name, fxn, globals_list, _runner = self.program[pc]
    if fxn is None: return 1  # unsupported instruction
    buf_addrs = {0: self.state.sgpr_buf._buf.va_addr, 1: self.state.vgpr_buf._buf.va_addr,
                 2: self.vmem_buf._buf.va_addr, 3: self.lds_buf._buf.va_addr}
    # Direct ctypes call - bypasses HCQ overhead
    fxn(*[ctypes.c_uint64(buf_addrs[g]) for g in globals_list], ctypes.c_int32(0))
    return -1 if self.state.pc == 0xFFFFFFFFFFFFFFFF else 0

  def set_sgpr(self, idx: int, val: int):
    assert self.state is not None
    self.state._write_sgpr(idx, val)
  def set_vgpr(self, lane: int, idx: int, val: int):
    assert self.state is not None
    self.state._write_vgpr(idx, lane, val)

  def get_snapshot(self) -> StateSnapshot:
    assert self.state is not None
    sgpr = [self.state._read_sgpr(i) for i in range(128)]
    vgpr = [[self.state._read_vgpr(reg, lane) for reg in range(256)] for lane in range(WAVE_SIZE)]
    # Convert actual PC address to word offset for comparison with Rust emulator
    pc_offset = (self.state.pc - self.lib_addr) // 4 if self.state.pc != 0xFFFFFFFFFFFFFFFF else 0xFFFFFFFFFFFFFFFF
    return StateSnapshot(pc=pc_offset, scc=self.state._read_sgpr(SCC.offset), vcc=sgpr[VCC_LO.offset],
                         exec_mask=sgpr[EXEC_LO.offset], sgpr=sgpr, vgpr=vgpr)

def run_single_kernel(kernel: bytes, n_lanes: int, args_ptr: int, global_size: tuple[int, int, int],
                      local_size: tuple[int, int, int], program, max_steps: int, debug: bool, trace_len: int,
                      kernel_idx: int = 0, max_workgroups: int = 8) -> tuple[bool, str, int]:
  """Run a single kernel through both emulators. Returns (success, message, total_steps)."""
  gx, gy, gz = global_size
  lx, ly, lz = local_size
  total_steps = 0
  wg_count = 0

  for gidz in range(gz):
    for gidy in range(gy):
      for gidx in range(gx):
        if wg_count >= max_workgroups: return True, f"Completed {wg_count} workgroups (limit reached)", total_steps
        wg_count += 1
        rust = RustEmulator()
        python = PythonEmulator()
        rust.create(kernel, n_lanes)
        python.create(kernel, n_lanes)

        # Initialize LDS (64KB, standard size for AMD GPUs)
        rust.init_lds(65536)

        for emu in (rust, python):
          emu.set_sgpr(0, args_ptr & 0xffffffff)
          emu.set_sgpr(1, (args_ptr >> 32) & 0xffffffff)
          emu.set_sgpr(13, gidx)
          emu.set_sgpr(14, gidy)
          emu.set_sgpr(15, gidz)
          # Initialize v[0] with packed workitem IDs for each lane
          for lane in range(n_lanes):
            tid = lane
            z, y, x = tid // (lx * ly), (tid // lx) % ly, tid % lx
            emu.set_vgpr(lane, 0, (z << 20) | (y << 10) | x)

        step = 0
        trace: list[tuple[int, int, str, StateSnapshot, StateSnapshot]] = []
        prev_sync_after = False  # Track if previous instruction had known Rust bugs
        try:
          while step < max_steps:
            rust_before = rust.get_snapshot()
            python_before = python.get_snapshot()

            inst_info = python.program.get(python.lib_addr + python_before.pc * 4)  # Convert word offset to actual address
            inst_hex_name = inst_info[0] if inst_info else f"unknown at PC={python_before.pc}"
            # Decode the instruction to get mnemonic for sync_after checks
            try:
              # Format is mnemonic_hexbytes, e.g. v_exp_f32_e32_014b027e -> hex is 014b027e
              parts = inst_hex_name.rsplit('_', 1)
              inst_bytes_hex = parts[1] if len(parts) == 2 else ""
              inst_bytes = bytes.fromhex(inst_bytes_hex) if inst_bytes_hex else b''
              decoded = decode_inst(inst_bytes) if inst_bytes else None
              inst_mnemonic = repr(decoded).split('(')[0] if decoded else ""
            except:
              inst_mnemonic = ""
            # For generic instructions, use function name for sync_after check
            if not inst_mnemonic: inst_mnemonic = inst_hex_name
            inst_str = inst_hex_name
            trace.append((step, python_before.pc, inst_str, rust_before, python_before))
            if len(trace) > trace_len: trace.pop(0)

            if debug: print(f"K{kernel_idx} WG({gidx},{gidy},{gidz}) Step {step}: PC={python_before.pc}, inst={inst_str}")

            # Instructions with known Rust emulator bugs or precision differences - sync Python to Rust after execution
            # v_div_scale/v_div_fixup: Rust has different VCC handling
            # v_cvt_f16_f32: Rust clears high 16 bits, but hardware (and Python) preserves them
            # s_add_i32/s_sub_i32: Rust has incorrect SCC overflow detection
            # v_exp_f32/v_log_f32/v_ldexp_f32: precision differences in transcendental functions
            # s_delay_alu: Rust handles differently
            # v_add_co_ci_u32/v_sub_co_ci_u32/v_subrev_co_ci_u32: Rust preserves inactive VCC bits, but hardware clears all bits
            sync_after = any(x in inst_mnemonic.lower() for x in ('v_div_scale', 'v_div_fixup', 'v_cvt_f16_f32', 's_add_i32', 's_sub_i32',
                                                                   'v_exp_f32', 'v_log_f32', 'v_ldexp_f32', 's_delay_alu',
                                                                   'v_add_co_ci_u32', 'v_sub_co_ci_u32', 'v_subrev_co_ci_u32'))
            # Skip comparison if previous instruction had known Rust bugs (states were synced but may still differ slightly)
            diffs = rust_before.diff(python_before, n_lanes) if not prev_sync_after else []
            if diffs:
              trace_lines = []
              for idx, (s, pc, d, rb, pb) in enumerate(trace):
                trace_lines.append(f"    step {s}: PC={pc:3d} {d}")
                if idx < len(trace) - 1:
                  next_rb, next_pb = trace[idx + 1][3:5]
                  rust_diffs = rb.diff(next_rb, n_lanes, "->")
                  python_diffs = pb.diff(next_pb, n_lanes, "->")
                  if rust_diffs: trace_lines.append(f"             rust:   {', '.join(rust_diffs[:5])}")
                  if python_diffs: trace_lines.append(f"             python: {', '.join(python_diffs[:5])}")
                  elif rust_diffs: trace_lines.append(f"             python: (no changes)")
                else:
                  # Last traced instruction - compare with current state
                  rust_diffs = rb.diff(rust_before, n_lanes, "->")
                  python_diffs = pb.diff(python_before, n_lanes, "->")
                  if rust_diffs: trace_lines.append(f"             rust:   {', '.join(rust_diffs[:5])}")
                  if python_diffs: trace_lines.append(f"             python: {', '.join(python_diffs[:5])}")
                  elif rust_diffs: trace_lines.append(f"             python: (no changes)")
              trace_str = "\n".join(trace_lines)
              return False, f"K{kernel_idx} WG({gidx},{gidy},{gidz}) Step {step} before inst '{inst_str}': states differ (rust vs python):\n  " + "\n  ".join(diffs[:10]) + f"\n  Recent instructions:\n{trace_str}", total_steps

            rust_result = rust.step()
            python_result = python.step()

            if rust_result != python_result:
              # Rust returns 1 for unsupported instructions - skip test
              if rust_result == 1 and python_result == 0:
                raise unittest.SkipTest(f"Rust emulator doesn't support instruction: {inst_str}")
              trace_str = "\n".join(f"    step {s}: PC={pc:3d} {d}" for s, pc, d, _, _ in trace)
              return False, f"K{kernel_idx} WG({gidx},{gidy},{gidz}) Step {step}: different return codes: rust={rust_result}, python={python_result}, inst={inst_str}\n  Recent instructions:\n{trace_str}", total_steps

            # Sync Python state to Rust after instructions with known Rust emulator differences
            if sync_after:
              rust_after = rust.get_snapshot()
              for i in range(128): python.set_sgpr(i, rust_after.sgpr[i])
              for lane in range(n_lanes):
                for i in range(256): python.set_vgpr(lane, i, rust_after.vgpr[lane][i])
              assert python.state is not None
              # Convert Rust's word-based PC to Python's actual address
              python.state.pc = python.lib_addr + rust_after.pc * 4
              python.state._write_sgpr(SCC.offset, rust_after.scc)
              python.state._write_sgpr(VCC_LO.offset, rust_after.vcc)
              python.state._write_sgpr(EXEC_LO.offset, rust_after.exec_mask)
            prev_sync_after = sync_after

            if rust_result == -1:
              total_steps += step + 1
              break
            if rust_result == 1:
              total_steps += step + 1
              break
            if rust_result < 0 and rust_result != -2:
              return False, f"K{kernel_idx} WG({gidx},{gidy},{gidz}) Step {step}: error code {rust_result}", total_steps

            step += 1
          else:
            return False, f"K{kernel_idx} WG({gidx},{gidy},{gidz}) Max steps ({max_steps}) reached", total_steps
        finally:
          rust.free()

  return True, f"Completed {gx*gy*gz} workgroups", total_steps

def compare_emulators_multi_kernel(kernels: list[KernelInfo], buf_pool: dict[int, int], max_steps: int = 1000,
                                    debug: bool = False, trace_len: int = 10, buf_data: dict[int, bytes] | None = None) -> tuple[bool, str]:
  """Run all kernels through both emulators with shared buffer pool."""
  if buf_data is None: buf_data = {}

  # Allocate shared buffer pool with padding for over-reads (GPU loads up to 16 bytes at once)
  buf_id_to_ptr: dict[int, int] = {}
  buffers = []
  for buf_id, size in buf_pool.items():
    padded_size = ((size + 15) // 16) * 16 + 16  # round up to 16 bytes + extra padding
    # Initialize with data from COPY if available
    init_data = buf_data.get(buf_id, b'\x00' * padded_size)
    init_list = list(init_data) + [0] * (padded_size - len(init_data))
    buf = (ctypes.c_uint8 * padded_size)(*init_list[:padded_size])
    buffers.append((buf, padded_size))
    buf_id_to_ptr[buf_id] = ctypes.addressof(buf)

  # Set up valid memory ranges
  ranges = {(ctypes.addressof(b), size) for b, size in buffers}

  total_steps = 0
  for ki, kernel in enumerate(kernels):
    # Create args array for this kernel's buffers
    args = (ctypes.c_uint64 * len(kernel.buf_idxs))(*[buf_id_to_ptr[bid] for bid in kernel.buf_idxs])
    args_ptr = ctypes.addressof(args)

    # Update valid ranges to include this args array
    kernel_ranges = ranges | {(args_ptr, ctypes.sizeof(args))}
    set_valid_mem_ranges(kernel_ranges)

    program = decode_program(kernel.code)
    n_lanes = kernel.local_size[0] * kernel.local_size[1] * kernel.local_size[2]

    ok, msg, steps = run_single_kernel(
      kernel.code, min(n_lanes, 32), args_ptr, kernel.global_size,
      kernel.local_size, program, max_steps, debug, trace_len, ki
    )
    total_steps += steps
    if not ok:
      return False, msg

  return True, f"Completed {len(kernels)} kernels, {total_steps} total steps"

def compare_emulators_with_memory(kernel: bytes, n_lanes: int, buf_sizes: list, max_steps: int = 1000, debug: bool = False,
                                   global_size: tuple[int, int, int] = (1, 1, 1), trace_len: int = 10) -> tuple[bool, str]:
  """Run both emulators with memory set up for tinygrad kernels, executing all workgroups. Legacy wrapper."""
  # Allocate buffers
  buffers = []
  for size in buf_sizes:
    buf = (ctypes.c_uint8 * size)(*[0] * size)
    buffers.append(buf)

  # Create args array with buffer pointers
  args = (ctypes.c_uint64 * len(buffers))(*[ctypes.addressof(b) for b in buffers])
  args_ptr = ctypes.addressof(args)

  # Set up valid memory ranges for Python emulator
  ranges = {(ctypes.addressof(b), len(b)) for b in buffers}
  ranges.add((args_ptr, ctypes.sizeof(args)))
  set_valid_mem_ranges(ranges)

  program = decode_program(kernel)
  # Legacy wrapper assumes local_size = (n_lanes, 1, 1)
  ok, msg, _ = run_single_kernel(kernel, n_lanes, args_ptr, global_size, (n_lanes, 1, 1), program, max_steps, debug, trace_len)
  return ok, msg

def get_kernels_from_tinygrad(op_fn) -> tuple[list[KernelInfo], dict[int, int], dict[int, bytes]]:
  """Compile a tinygrad operation and extract all kernels with their buffer mappings."""
  from tinygrad import Tensor
  from tinygrad.runtime.support.elf import elf_loader

  out = op_fn(Tensor)
  sched = out.schedule()
  kernels = []
  buf_pool: dict[int, int] = {}  # buffer id -> size
  buf_data: dict[int, bytes] = {}  # buffer id -> initial data from COPY

  for ei in sched:
    lowered = ei.lower()
    if ei.ast.op.name == 'COPY':
      # Handle COPY: extract source data to initialize destination buffer
      if len(lowered.bufs) >= 2:
        dst_buf, src_buf = lowered.bufs[0], lowered.bufs[1]
        dst_id = id(dst_buf)
        if dst_id not in buf_pool:
          buf_pool[dst_id] = dst_buf.nbytes
        # Get source data if it's from numpy/CPU
        if hasattr(src_buf, 'base') and src_buf.base is not None and hasattr(src_buf.base, '_buf'):
          src_data = bytes(src_buf.base._buf)
          buf_data[dst_id] = src_data
    elif ei.ast.op.name == 'SINK':
      if lowered.prg and lowered.prg.p.lib:
        lib = bytes(lowered.prg.p.lib)
        _, sections, _ = elf_loader(lib)
        for sec in sections:
          if sec.name == '.text':
            buf_idxs = []
            buf_sizes = []
            for b in lowered.bufs:
              buf_id = id(b)
              if buf_id not in buf_pool:
                buf_pool[buf_id] = b.nbytes
              buf_idxs.append(buf_id)
              buf_sizes.append(b.nbytes)
            kernels.append(KernelInfo(
              code=bytes(sec.content),
              src=lowered.prg.p.src,
              global_size=tuple(lowered.prg.p.global_size),
              local_size=tuple(lowered.prg.p.local_size),
              buf_idxs=buf_idxs,
              buf_sizes=buf_sizes
            ))
  if not kernels: raise RuntimeError("No kernel found")
  return kernels, buf_pool, buf_data

def get_kernel_from_tinygrad(op_fn) -> tuple[bytes, tuple[int, int, int], tuple[int, int, int], list]:
  """Compile a tinygrad operation and extract the last (main) kernel binary. Legacy wrapper."""
  kernels, _, _ = get_kernels_from_tinygrad(op_fn)
  k = kernels[-1]
  return k.code, k.global_size, k.local_size, k.buf_sizes

class TestTinygradKernels(unittest.TestCase):
  """Compare emulators on real tinygrad-compiled kernels."""

  def _test_kernel(self, op_fn, max_steps=10000):
    kernels, buf_pool, buf_data = get_kernels_from_tinygrad(op_fn)
    ok, msg = compare_emulators_multi_kernel(kernels, buf_pool, max_steps=max_steps, buf_data=buf_data)
    self.assertTrue(ok, msg)

  # Basic ops - consolidated tests covering key instruction patterns
  def test_unary_ops(self): self._test_kernel(lambda T: T([-1.0, 0.0, 1.0, 2.0]).relu().exp().log().sqrt().reciprocal())
  def test_binary_ops(self): self._test_kernel(lambda T: (T([1.0, 2.0]) + T([3.0, 4.0])) * T([0.5, 0.5]) - T([1.0, 1.0]))
  def test_trig(self): self._test_kernel(lambda T: T([0.1, 1.0, 3.14, -1.0]*8).sin() + T([0.1, 1.0, 3.14, -1.0]*8).cos())
  def test_compare(self): self._test_kernel(lambda T: (T.empty(64) < T.empty(64)).where(T.empty(64), T.empty(64)))
  def test_bitwise(self): self._test_kernel(lambda T: (T([0xF0, 0x0F, 0xFF]*11).int() & T([0x0F, 0x0F, 0x00]*11).int()) | T([1]*33).int())
  def test_int_ops(self): self._test_kernel(lambda T: ((T.empty(64).int() + T.empty(64).int()) * T.empty(64).int()).float())

  # Reductions
  def test_reduce(self): self._test_kernel(lambda T: T.empty(64).sum() + T.empty(64).max())
  def test_argmax(self): self._test_kernel(lambda T: T.empty(64).argmax())

  # Matmul
  def test_gemm(self): self._test_kernel(lambda T: T.empty(8, 8) @ T.empty(8, 8), max_steps=100000)
  @unittest.skip("Rust emulator crashes on this kernel (assertion failure in thread.rs)")
  def test_gemm_fp16(self): self._test_kernel(lambda T: T.empty(16, 16).half() @ T.empty(16, 16).half(), max_steps=100000)

  # Complex ops
  def test_softmax(self): self._test_kernel(lambda T: T.empty(16).softmax())
  def test_layernorm(self): self._test_kernel(lambda T: T.empty(8, 8).layernorm())

  # Memory patterns
  def test_memory(self): self._test_kernel(lambda T: T.empty(4, 4).permute(1, 0).contiguous() + T.empty(4, 1).expand(4, 4))

  # Cast ops
  def test_cast(self): self._test_kernel(lambda T: T.empty(32).half().float() + T.empty(32).int().float())

  # Pooling - regression for VCC wave32 mode
  def test_pool2d(self): self._test_kernel(lambda T: T.empty(1, 1, 8, 8).avg_pool2d(kernel_size=(4,4)) + T.empty(1, 1, 8, 8).max_pool2d(kernel_size=(4,4)))

  # Convolution
  def test_conv2d(self): self._test_kernel(lambda T: T.empty(1, 2, 8, 8).conv2d(T.empty(2, 2, 3, 3)), max_steps=50000)

  # Regression tests
  def test_topk(self): self._test_kernel(lambda T: T.empty(64).topk(3)[0])
  def test_interpolate(self): self._test_kernel(lambda T: T.empty(1,2,16,16).relu().cast('uint8').interpolate((8,8), mode="linear"))
  def test_index_int64(self):
    from tinygrad import dtypes
    self._test_kernel(lambda T: T.empty(4, 4)[T.arange(4).cast(dtypes.int64), :])
  def test_gelu(self): self._test_kernel(lambda T: T.empty(32, 32).gelu())
  def test_exp(self): self._test_kernel(lambda T: T.empty(1024).exp())
  def test_cross_entropy(self):
    import numpy as np
    np.random.seed(0)
    classes = np.random.randint(0, 10, (16,), dtype=np.int32).tolist()
    x_np = np.random.randn(16, 10).astype(np.float32)
    self._test_kernel(lambda T: (T(x_np.tolist()).reshape(16,10) + 0).cross_entropy((T(classes).int().reshape(16) + 0)))
  def test_isinf(self): self._test_kernel(lambda T: T([float('-inf'), 0., float('inf'), 1.1]*8).isinf())
  def test_sin_f64(self):
    from tinygrad import dtypes
    self._test_kernel(lambda T: T([2.0], dtype=dtypes.float64).sin())

  def test_sin_large_f32(self):
    """Test sin with large values that trigger Payne-Hanek range reduction."""
    # Values around 859240 trigger the Payne-Hanek algorithm
    # This tests the integer multiply-high instructions used in range reduction
    self._test_kernel(lambda T: T([859240.0, 1000000.0, 100594688.0]).sin())

  def test_clip_zero_one(self):
    """Test clip(0, 1) - regression for binary_crossentropy failure."""
    import numpy as np
    np.random.seed(0)
    x_np = np.random.uniform(-2, 2, (32, 10)).astype(np.float32).tolist()
    self._test_kernel(lambda T: T(x_np).clip(0, 1))

  def test_mod_int64(self):
    """Test int64 modulo, especially edge cases like 1 % -1."""
    from tinygrad import dtypes
    self._test_kernel(lambda T: T([1, 10, -10, 7], dtype=dtypes.int64) % T([-1, 3, 3, -3], dtype=dtypes.int64))

  def test_expand_flatten_sum(self):
    """Test flatten of expanded tensor followed by sum.

    Bug: flatten() of an expanded tensor produces wrong results for certain sizes.
    Sizes that are multiples of 32 work (32, 48, 64), but sizes like 33, 49, 50 fail.
    This breaks masked_select and nonzero operations.
    """
    import numpy as np
    np.random.seed(0)
    x_np = np.random.uniform(-2, 2, (33,)).astype(np.float32)
    self._test_kernel(lambda T: (T(x_np.tolist()) > 0.5).unsqueeze(-1).expand(33, 3).flatten().sum())

  @unittest.skip("slow and broken with AMD_LLVM=1")
  def test_nonzero(self):
    """Test nonzero operation - counts and gathers indices of non-zero elements."""
    import numpy as np
    np.random.seed(42)
    x_np = np.random.rand(10, 5, 3).astype(np.float32)
    self._test_kernel(lambda T: (T(x_np.tolist()) > 0.5).nonzero())

  @unittest.skip("Precision differences in v_exp/v_log accumulate across kernels, causing memory divergence")
  def test_softmax_argmax_fused(self):
    """Test fused softmax+argmax - tracks exp2 precision issue.

    The fused kernel recomputes softmax inline and Python emulator's exp2 polynomial
    has up to 1 ULP error vs native exp2f, causing accumulated differences.
    """
    import torch
    torch.manual_seed(0)
    x_np = torch.rand(4, 10).numpy()
    self._test_kernel(lambda T: T(x_np.tolist()).softmax(1).argmax())

if __name__ == "__main__":
  unittest.main()
