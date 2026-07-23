"""Tests for DPP16 source swizzles.

These instructions trap in the default wave32 hw helper, so this file uses a
minimal wave64 lane-store harness and compares emulator vs hardware directly
when USE_HW=1.
"""
import ctypes, unittest
from tinygrad.runtime.autogen.amd.rdna3.ins import *
from tinygrad.helpers import flat_mv
from test.amd.hw.helpers import USE_HW, assemble
from test.mockgpu.amd.emu import run_asm

WAVE64 = 64

def _wave64_code(instructions: list, out_reg: int = 1) -> bytes:
  return assemble([
    s_mov_b32(s[80], s[0]),
    s_mov_b32(s[81], s[1]),
    v_mov_b32_e32(v[255], v[0]),
    *instructions,
    s_load_b64(s[92:93], s[80:81], 0, soffset=NULL),
    s_waitcnt(0),
    v_lshlrev_b32_e32(v[240], 2, v[255]),
    global_store_b32(addr=v[240], data=v[out_reg], saddr=s[92:93], offset=0),
    s_endpgm(),
  ])

def _run_wave64_emu(instructions: list, out_reg: int = 1) -> list[int]:
  out_buf = (ctypes.c_uint32 * WAVE64)(*([0] * WAVE64))
  args = (ctypes.c_uint64 * 1)(ctypes.addressof(out_buf))
  code = _wave64_code(instructions, out_reg)
  kernel_buf = (ctypes.c_char * len(code)).from_buffer_copy(code)
  rsrc2 = 0x19c | (128 << 15)
  scratch_size = 0x10000
  result = run_asm(ctypes.addressof(kernel_buf), len(code), 1, 1, 1, WAVE64, 1, 1, ctypes.addressof(args), rsrc2, scratch_size)
  assert result == 0, f"run_asm failed with {result}"
  return list(out_buf)

def _run_wave64_hw(instructions: list, out_reg: int = 1) -> list[int]:
  from tinygrad.device import Device
  from tinygrad.runtime.ops_amd import AMDProgram
  from tinygrad.runtime.support.compiler_amd import HIPCompiler

  dev = Device["AMD"]
  compiler = HIPCompiler(dev.arch)  # type: ignore[attr-defined]
  code = _wave64_code(instructions, out_reg)
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
    .wavefront_size: 64
    .sgpr_count: 96
    .vgpr_count: 256
    .max_flat_workgroup_size: 1024
...
.end_amdgpu_metadata
"""
  lib = compiler.compile(asm_src)
  prg = AMDProgram(dev, "test", lib)  # type: ignore[arg-type]
  out_gpu = dev.allocator.alloc(WAVE64 * 4)
  prg(out_gpu, global_size=(1, 1, 1), local_size=(WAVE64, 1, 1), wait=True)
  out = bytearray(WAVE64 * 4)
  dev.allocator._copyout(flat_mv(memoryview(out)), out_gpu)
  return [int.from_bytes(out[i*4:(i+1)*4], 'little') for i in range(WAVE64)]

def run_wave64(instructions: list, out_reg: int = 1) -> list[int]:
  emu = _run_wave64_emu(instructions, out_reg)
  if not USE_HW: return emu
  hw = _run_wave64_hw(instructions, out_reg)
  if emu != hw:
    diffs = [f"lane {i}: emu=0x{e:08x} hw=0x{h:08x}" for i, (e, h) in enumerate(zip(emu, hw)) if e != h]
    raise AssertionError("Emulator vs Hardware mismatch:\n" + '\n'.join(diffs[:16]))
  return hw

class TestDPP16(unittest.TestCase):
  def _run_copy(self, dpp: int, *, row_mask: int = 0xf, bank_mask: int = 0xf, bc: int = 1, dst_seed: int | None = None) -> list[int]:
    instructions = [
      v_mul_u32_u24_e32(v[0], 10, v[255]),
      v_add_nc_u32_e32(v[0], 3, v[0]),
    ]
    if dst_seed is not None: instructions.append(v_mov_b32_e32(v[1], dst_seed))
    instructions += [v_mov_b32_e32(v[2], 0), v_or_b32_e32(v[1], DPP, v[2], vsrc0=v[0], dpp=dpp, row_mask=row_mask, bank_mask=bank_mask, bc=bc)]
    return run_wave64(instructions)

  def test_quad_perm_reverse(self):
    out = self._run_copy(0x1b)
    self.assertEqual(out[0], 33)
    self.assertEqual(out[1], 23)
    self.assertEqual(out[2], 13)
    self.assertEqual(out[3], 3)
    self.assertEqual(out[4], 73)

  def test_row_shl(self):
    out = self._run_copy(0x101)
    self.assertEqual(out[0], 13)
    self.assertEqual(out[7], 83)
    self.assertEqual(out[14], 153)
    self.assertEqual(out[15], 0)
    self.assertEqual(out[16], 173)

  def test_row_shr(self):
    out = self._run_copy(0x111)
    self.assertEqual(out[0], 0)
    self.assertEqual(out[1], 3)
    self.assertEqual(out[8], 73)
    self.assertEqual(out[15], 143)
    self.assertEqual(out[16], 0)
    self.assertEqual(out[17], 163)

  def test_row_ror(self):
    out = self._run_copy(0x121)
    self.assertEqual(out[0], 153)
    self.assertEqual(out[1], 3)
    self.assertEqual(out[15], 143)
    self.assertEqual(out[16], 313)

  def test_row_mirror(self):
    out = self._run_copy(0x140)
    self.assertEqual(out[0], 153)
    self.assertEqual(out[5], 103)
    self.assertEqual(out[8], 73)
    self.assertEqual(out[16], 313)

  def test_row_half_mirror(self):
    out = self._run_copy(0x141)
    self.assertEqual(out[0], 73)
    self.assertEqual(out[7], 3)
    self.assertEqual(out[8], 153)
    self.assertEqual(out[15], 83)
    self.assertEqual(out[16], 233)

  def test_row_mask(self):
    out = self._run_copy(0x101, row_mask=0x5, dst_seed=0xDEADBEEF)
    self.assertEqual(out[0], 13)
    self.assertEqual(out[15], 0)
    self.assertEqual(out[16], 0xDEADBEEF)
    self.assertEqual(out[32], 333)
    self.assertEqual(out[47], 0)
    self.assertEqual(out[48], 0xDEADBEEF)

  def test_bank_mask(self):
    out = self._run_copy(0x101, bank_mask=0x5, dst_seed=0xDEADBEEF)
    self.assertEqual(out[0], 13)
    self.assertEqual(out[3], 43)
    self.assertEqual(out[4], 0xDEADBEEF)
    self.assertEqual(out[8], 93)
    self.assertEqual(out[12], 0xDEADBEEF)

class TestVOPCDPP16(unittest.TestCase):
  def test_row_bcast15_materializes_vcc(self):
    out = run_wave64([
      v_mov_b32_e32(v[0], v[255]),
      v_cmp_eq_u32_e32(DPP, v[0], vsrc0=v[0], dpp=0x142, row_mask=0xf, bank_mask=0xf, bc=1),
      v_mov_b32_e32(v[2], 0),
      v_mov_b32_e32(v[3], 1),
      v_cndmask_b32_e32(v[1], v[2], v[3]),
    ])
    for lane in (0, 16, 32, 48): self.assertEqual(out[lane], 1)
    for lane in (1, 15, 31, 47, 63): self.assertEqual(out[lane], 0)
