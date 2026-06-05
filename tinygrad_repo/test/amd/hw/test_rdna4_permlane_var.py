"""RDNA4 V_PERMLANE16_VAR_B32 / V_PERMLANEX16_VAR_B32 coverage.

Exercises the generated pcode path end-to-end in the emulator and compares against
real RDNA4 hardware when USE_HW=1.
"""
import ctypes, unittest
import tinygrad.runtime.autogen.amd.rdna4.ins as r4
from tinygrad.helpers import flat_mv
from tinygrad.renderer.amd.dsl import NULL
from test.amd.hw.helpers import USE_HW, assemble
from test.mockgpu.amd.emu import run_asm

LANES = 32

def _code(instructions: list, out_reg: int = 2) -> bytes:
  return assemble([
    r4.s_mov_b32(r4.s[80], r4.s[0]),
    r4.s_mov_b32(r4.s[81], r4.s[1]),
    r4.v_mov_b32_e32(r4.v[255], r4.v[0]),
    *instructions,
    r4.s_load_b64(r4.s[92:93], r4.s[80:81], soffset=NULL),
    r4.s_wait_kmcnt(simm16=0),
    r4.v_lshlrev_b32_e32(r4.v[240], 2, r4.v[255]),
    r4.v_mov_b32_e32(r4.v[241], 0),
    r4.global_store_b32(vaddr=r4.v[240:241], saddr=r4.s[92:93], vsrc=r4.v[out_reg]),
    r4.s_endpgm(),
  ])

def _run_emu(instructions: list, out_reg: int = 2) -> list[int]:
  out_buf = (ctypes.c_uint32 * LANES)(*([0] * LANES))
  args = (ctypes.c_uint64 * 1)(ctypes.addressof(out_buf))
  code = _code(instructions, out_reg)
  kernel_buf = (ctypes.c_char * len(code)).from_buffer_copy(code)
  result = run_asm(ctypes.addressof(kernel_buf), len(code), 1, 1, 1, LANES, 1, 1, ctypes.addressof(args), arch='rdna4')
  assert result == 0, f"run_asm failed with {result}"
  return list(out_buf)

def _run_hw(instructions: list, out_reg: int = 2) -> list[int]:
  from tinygrad.device import Device
  from tinygrad.runtime.ops_amd import AMDProgram
  from tinygrad.runtime.support.compiler_amd import HIPCompiler

  dev = Device['AMD']
  if not dev.arch.startswith('gfx12'): raise unittest.SkipTest('requires RDNA4 hardware')
  compiler = HIPCompiler(dev.arch)
  code = _code(instructions, out_reg)
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
  prg = AMDProgram(dev, 'test', lib)
  out_gpu = dev.allocator.alloc(LANES * 4)
  prg(out_gpu, global_size=(1, 1, 1), local_size=(LANES, 1, 1), wait=True)
  out = bytearray(LANES * 4)
  dev.allocator._copyout(flat_mv(memoryview(out)), out_gpu)
  return [int.from_bytes(out[i*4:(i+1)*4], 'little') for i in range(LANES)]

def run_rdna4(instructions: list, out_reg: int = 2) -> list[int]:
  emu = _run_emu(instructions, out_reg)
  if not USE_HW: return emu
  hw = _run_hw(instructions, out_reg)
  if emu != hw:
    diffs = [f"lane {i}: emu=0x{e:08x} hw=0x{h:08x}" for i, (e, h) in enumerate(zip(emu, hw)) if e != h]
    raise AssertionError("Emulator vs Hardware mismatch:\n" + '\n'.join(diffs[:16]))
  return hw

class TestPermlaneVarRDNA4(unittest.TestCase):
  def test_v_permlane16_var_b32_reverse(self):
    out = run_rdna4([
      r4.v_mov_b32_e32(r4.v[0], r4.v[255]),
      r4.v_xor_b32_e32(r4.v[1], 15, r4.v[255]),
      r4.v_permlane16_var_b32(r4.v[2], r4.v[0], r4.v[1]),
    ])
    self.assertEqual(out[0], 15)
    self.assertEqual(out[5], 10)
    self.assertEqual(out[15], 0)
    self.assertEqual(out[16], 31)
    self.assertEqual(out[21], 26)
    self.assertEqual(out[31], 16)

  def test_v_permlanex16_var_b32_cross_row(self):
    out = run_rdna4([
      r4.v_mov_b32_e32(r4.v[0], r4.v[255]),
      r4.v_mov_b32_e32(r4.v[1], r4.v[255]),
      r4.v_permlanex16_var_b32(r4.v[2], r4.v[0], r4.v[1]),
    ])
    self.assertEqual(out[0], 16)
    self.assertEqual(out[5], 21)
    self.assertEqual(out[15], 31)
    self.assertEqual(out[16], 0)
    self.assertEqual(out[21], 5)
    self.assertEqual(out[31], 15)
