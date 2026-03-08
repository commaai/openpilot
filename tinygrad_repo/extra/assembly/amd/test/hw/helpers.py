"""Test infrastructure for hardware-validated RDNA3 emulator tests.

Uses run_asm() with memory output, so tests can run on both emulator and real hardware.
Set USE_HW=1 to run on both emulator and hardware, comparing results.
"""
import ctypes, math, os, struct
from extra.assembly.amd.autogen.rdna3.ins import *

from extra.assembly.amd.emu import run_asm
from extra.assembly.amd.dsl import NULL, SCC, VCC_LO, VCC_HI, EXEC_LO, EXEC_HI, M0

def _i32(f: float) -> int: return struct.unpack('<I', struct.pack('<f', f))[0]
def _f32(i: int) -> float: return struct.unpack('<f', struct.pack('<I', i & 0xFFFFFFFF))[0]

# f16 conversion helpers
def f16(i: int) -> float: return struct.unpack('<e', struct.pack('<H', i & 0xFFFF))[0]
def f32_to_f16(f: float) -> int:
  f = float(f)
  if math.isnan(f): return 0x7e00
  if math.isinf(f): return 0x7c00 if f > 0 else 0xfc00
  try: return struct.unpack('<H', struct.pack('<e', f))[0]
  except OverflowError: return 0x7c00 if f > 0 else 0xfc00

# For backwards compatibility with tests using SrcEnum.NULL etc.
class SrcEnum:
  NULL = NULL
  VCC_LO = VCC_LO
  VCC_HI = VCC_HI
  EXEC_LO = EXEC_LO
  EXEC_HI = EXEC_HI
  SCC = SCC
  M0 = M0
  POS_HALF = 0.5
  NEG_HALF = -0.5
  POS_ONE = 1.0
  NEG_ONE = -1.0
  POS_TWO = 2.0
  NEG_TWO = -2.0
  POS_FOUR = 4.0
  NEG_FOUR = -4.0

VCC = VCC_LO  # For VOP3SD sdst field (VCC_LO is exported from dsl)
USE_HW = os.environ.get("USE_HW", "0") == "1"
FLOAT_TOLERANCE = 1e-5

def get_gpu_target() -> tuple[int, int, int]:
  """Get the GPU target as (major, minor, stepping) tuple."""
  if not USE_HW: return (0, 0, 0)
  from tinygrad.device import Device
  return Device["AMD"].target

def skip_unless_gfx(min_major: int, min_minor: int = 0, reason: str = ""):
  """Skip test if GPU target is below the minimum required version."""
  import unittest
  def decorator(test_func):
    if not USE_HW: return test_func
    target = get_gpu_target()
    if target[0] < min_major or (target[0] == min_major and target[1] < min_minor):
      return unittest.skip(reason or f"requires gfx{min_major}{min_minor}0+")(test_func)
    return test_func
  return decorator

# Output buffer layout: vgpr[16][32], sgpr[16], vcc, scc, exec
N_VGPRS, N_SGPRS, WAVE_SIZE = 16, 16, 32
VGPR_BYTES = N_VGPRS * WAVE_SIZE * 4  # 16 regs * 32 lanes * 4 bytes = 2048
SGPR_BYTES = N_SGPRS * 4  # 16 regs * 4 bytes = 64
OUT_BYTES = VGPR_BYTES + SGPR_BYTES + 12  # + vcc + scc + exec

# Float conversion helpers
def f2i(f: float) -> int: return _i32(f)
def i2f(i: int) -> float: return _f32(i)
def f2i64(f: float) -> int: return struct.unpack('<Q', struct.pack('<d', f))[0]
def i642f(i: int) -> float: return struct.unpack('<d', struct.pack('<Q', i))[0]

def assemble(instructions: list) -> bytes:
  return b''.join(inst.to_bytes() for inst in instructions)

# Simple WaveState class for test output parsing (mirrors emu.py interface for tests)
class WaveState:
  def __init__(self):
    self.vgpr = [[0] * 256 for _ in range(32)]  # vgpr[lane][reg]
    self.sgpr = [0] * 128
    self.vcc = 0
    self.scc = 0

def get_prologue_epilogue(n_lanes: int) -> tuple[list, list]:
  """Generate prologue and epilogue instructions for state capture."""
  prologue = [
    s_mov_b32(s[80], s[0]),
    s_mov_b32(s[81], s[1]),
    v_mov_b32_e32(v[255], v[0]),
  ]
  for i in range(N_VGPRS):
    prologue.append(v_mov_b32_e32(v[i], 0))
  for i in range(N_SGPRS):
    prologue.append(s_mov_b32(s[i], 0))
  prologue.append(s_mov_b32(VCC_LO, 0))

  epilogue = [
    s_mov_b32(s[90], VCC_LO),
    s_cselect_b32(s[91], 1, 0),
    # Save EXEC early (before we modify it for VGPR stores)
    s_mov_b32(s[95], EXEC_LO),
    # Restore EXEC to all active lanes for VGPR stores (test may have modified EXEC)
    s_mov_b32(EXEC_LO, (1 << n_lanes) - 1),
    s_load_b64(s[92:93], s[80:81], 0, soffset=NULL),
    s_waitcnt(0),  # simm16=0 waits for all
    v_lshlrev_b32_e32(v[240], 2, v[255]),
  ]
  for i in range(N_VGPRS):
    epilogue.append(global_store_b32(addr=v[240], data=v[i], saddr=s[92:93], offset=i * WAVE_SIZE * 4))
  epilogue.append(v_mov_b32_e32(v[241], 0))
  epilogue.append(v_cmp_eq_u32_e32(v[255], v[241]))
  epilogue.append(s_and_saveexec_b32(s[94], VCC_LO))
  epilogue.append(v_mov_b32_e32(v[240], 0))
  for i in range(N_SGPRS):
    epilogue.append(v_mov_b32_e32(v[243], s[i]))
    epilogue.append(global_store_b32(addr=v[240], data=v[243], saddr=s[92:93], offset=VGPR_BYTES + i * 4))
  epilogue.append(v_mov_b32_e32(v[243], s[90]))
  epilogue.append(global_store_b32(addr=v[240], data=v[243], saddr=s[92:93], offset=VGPR_BYTES + SGPR_BYTES))
  epilogue.append(v_mov_b32_e32(v[243], s[91]))
  epilogue.append(global_store_b32(addr=v[240], data=v[243], saddr=s[92:93], offset=VGPR_BYTES + SGPR_BYTES + 4))
  # Store EXEC (saved earlier in s[95])
  epilogue.append(v_mov_b32_e32(v[243], s[95]))
  epilogue.append(global_store_b32(addr=v[240], data=v[243], saddr=s[92:93], offset=VGPR_BYTES + SGPR_BYTES + 8))
  epilogue.append(s_mov_b32(EXEC_LO, s[94]))
  epilogue.append(s_endpgm())
  return prologue, epilogue

def parse_output(out_buf: bytes, n_lanes: int) -> WaveState:
  """Parse output buffer into WaveState."""
  st = WaveState()
  for i in range(N_VGPRS):
    for lane in range(n_lanes):
      off = i * WAVE_SIZE * 4 + lane * 4
      st.vgpr[lane][i] = struct.unpack_from('<I', out_buf, off)[0]
  for i in range(N_SGPRS):
    st.sgpr[i] = struct.unpack_from('<I', out_buf, VGPR_BYTES + i * 4)[0]
  st.vcc = struct.unpack_from('<I', out_buf, VGPR_BYTES + SGPR_BYTES)[0]
  st.scc = struct.unpack_from('<I', out_buf, VGPR_BYTES + SGPR_BYTES + 4)[0]
  # Store EXEC in its proper location (index 126)
  st.sgpr[EXEC_LO.offset] = struct.unpack_from('<I', out_buf, VGPR_BYTES + SGPR_BYTES + 8)[0]
  return st

def run_program_emu(instructions: list, n_lanes: int = 1) -> WaveState:
  """Run instructions via emulator run_asm, dump state to memory, return WaveState."""
  out_buf = (ctypes.c_uint8 * OUT_BYTES)(*([0] * OUT_BYTES))
  out_addr = ctypes.addressof(out_buf)

  prologue, epilogue = get_prologue_epilogue(n_lanes)
  code = assemble(prologue + instructions + epilogue)

  args = (ctypes.c_uint64 * 1)(out_addr)
  args_ptr = ctypes.addressof(args)
  kernel_buf = (ctypes.c_char * len(code)).from_buffer_copy(code)
  lib_ptr = ctypes.addressof(kernel_buf)

  # rsrc2: USER_SGPR_COUNT=2, ENABLE_SGPR_WORKGROUP_ID_X/Y/Z=1, LDS_SIZE=128 (64KB)
  rsrc2 = 0x19c | (128 << 15)
  scratch_size = 0x10000  # 64KB per lane, matches .amdhsa_private_segment_fixed_size in run_program_hw
  result = run_asm(lib_ptr, len(code), 1, 1, 1, n_lanes, 1, 1, args_ptr, rsrc2, scratch_size)
  assert result == 0, f"run_asm failed with {result}"

  return parse_output(bytes(out_buf), n_lanes)

def run_program_hw(instructions: list, n_lanes: int = 1) -> WaveState:
  """Run instructions on real AMD hardware via HIPCompiler and AMDProgram."""
  from tinygrad.device import Device
  from tinygrad.runtime.ops_amd import AMDProgram
  from tinygrad.runtime.support.compiler_amd import HIPCompiler
  from tinygrad.helpers import flat_mv

  dev = Device["AMD"]
  compiler = HIPCompiler(dev.arch)

  prologue, epilogue = get_prologue_epilogue(n_lanes)
  code = assemble(prologue + instructions + epilogue)

  byte_str = ', '.join(f'0x{b:02x}' for b in code)
  asm_src = f""".text
.globl test
.p2align 8
.type test,@function
test:
.byte {byte_str}

.rodata
.p2align 6
.amdhsa_kernel test
  .amdhsa_next_free_vgpr 256
  .amdhsa_next_free_sgpr 96
  .amdhsa_wavefront_size32 1
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_kernarg_size 8
  .amdhsa_group_segment_fixed_size 65536
  .amdhsa_private_segment_fixed_size 65536
  .amdhsa_enable_private_segment 1
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version:
  - 1
  - 0
amdhsa.kernels:
  - .name: test
    .symbol: test.kd
    .kernarg_segment_size: 8
    .group_segment_fixed_size: 65536
    .private_segment_fixed_size: 65536
    .kernarg_segment_align: 8
    .wavefront_size: 32
    .sgpr_count: 96
    .vgpr_count: 256
    .max_flat_workgroup_size: 1024
...
.end_amdgpu_metadata
"""

  lib = compiler.compile(asm_src)
  prg = AMDProgram(dev, "test", lib)

  out_gpu = dev.allocator.alloc(OUT_BYTES)
  assert out_gpu.va_addr % 16 == 0, f"buffer not 16-byte aligned: 0x{out_gpu.va_addr:x}"
  prg(out_gpu, global_size=(1, 1, 1), local_size=(n_lanes, 1, 1), wait=True)

  out_buf = bytearray(OUT_BYTES)
  dev.allocator._copyout(flat_mv(memoryview(out_buf)), out_gpu)

  return parse_output(bytes(out_buf), n_lanes)

def compare_wave_states(emu_st: WaveState, hw_st: WaveState, n_lanes: int, n_vgprs: int = N_VGPRS, ulp_tolerance: int = 0) -> list[str]:
  """Compare two WaveStates and return list of differences.

  Args:
    ulp_tolerance: Allow up to this many ULPs difference for float comparisons (0 = exact match required)
  """
  import math
  diffs = []
  for i in range(n_vgprs):
    for lane in range(n_lanes):
      emu_val = emu_st.vgpr[lane][i]
      hw_val = hw_st.vgpr[lane][i]
      if emu_val != hw_val:
        emu_f, hw_f = _f32(emu_val), _f32(hw_val)
        if math.isnan(emu_f) and math.isnan(hw_f):
          continue
        # Check ULP difference for floats (only for same-sign values)
        if ulp_tolerance > 0 and (emu_val < 0x80000000) == (hw_val < 0x80000000):
          ulp_diff = abs(int(emu_val) - int(hw_val))
          if ulp_diff <= ulp_tolerance:
            continue
        diffs.append(f"v[{i}] lane {lane}: emu=0x{emu_val:08x} ({emu_f:.6g}) hw=0x{hw_val:08x} ({hw_f:.6g})")
  for i in range(N_SGPRS):
    emu_val = emu_st.sgpr[i]
    hw_val = hw_st.sgpr[i]
    if emu_val != hw_val:
      diffs.append(f"s[{i}]: emu=0x{emu_val:08x} hw=0x{hw_val:08x}")
  if emu_st.vcc != hw_st.vcc:
    diffs.append(f"vcc: emu=0x{emu_st.vcc:08x} hw=0x{hw_st.vcc:08x}")
  if emu_st.scc != hw_st.scc:
    diffs.append(f"scc: emu={emu_st.scc} hw={hw_st.scc}")
  return diffs

def run_program(instructions: list, n_lanes: int = 1, ulp_tolerance: int = 0) -> WaveState:
  """Run instructions and return WaveState.

  If USE_HW=1, runs on both emulator and hardware, compares results, and raises if they differ.
  Otherwise, runs only on emulator.

  Args:
    ulp_tolerance: Allow up to this many ULPs difference for float comparisons (0 = exact match required)
  """
  emu_st = run_program_emu(instructions, n_lanes)
  if USE_HW:
    hw_st = run_program_hw(instructions, n_lanes)
    diffs = compare_wave_states(emu_st, hw_st, n_lanes, ulp_tolerance=ulp_tolerance)
    if diffs:
      raise AssertionError(f"Emulator vs Hardware mismatch:\n" + "\n".join(diffs))
    return hw_st
  return emu_st
