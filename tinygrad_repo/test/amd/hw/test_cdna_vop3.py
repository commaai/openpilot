"""CDNA VOP3 instruction coverage.

Exercises generated CDNA pcode end-to-end in the emulator and compares against
gfx950 hardware when USE_HW=1.
"""
import ctypes, struct, unittest
import tinygrad.runtime.autogen.amd.cdna.ins as cdna
from tinygrad.helpers import flat_mv
from tinygrad.renderer.amd.dsl import NULL
from test.amd.hw.helpers import USE_HW, assemble
from test.mockgpu.amd.emu import run_asm

LANES = 1

def _code(instructions: list, out_reg: int = 2, out_addr: int | None = None) -> bytes:
  load_out_addr = [
    cdna.s_mov_b32(cdna.s[92], out_addr & 0xffffffff),
    cdna.s_mov_b32(cdna.s[93], out_addr >> 32),
  ] if out_addr is not None else [
    cdna.s_load_dwordx2(cdna.s[92:93], cdna.s[80:81], 0, soffset=NULL),
    cdna.s_waitcnt(0),
  ]
  return assemble([
    cdna.s_mov_b32(cdna.s[80], cdna.s[0]),
    cdna.s_mov_b32(cdna.s[81], cdna.s[1]),
    cdna.v_mov_b32_e32(cdna.v[255], cdna.v[0]),
    *instructions,
    *load_out_addr,
    cdna.v_lshlrev_b32_e32(cdna.v[240], 2, cdna.v[255]),
    cdna.global_store_dword(addr=cdna.v[240], data=cdna.v[out_reg], saddr=cdna.s[92:93], offset=0),
    cdna.s_endpgm(),
  ])

def _run_emu(instructions: list, out_reg: int = 2) -> int:
  out_buf = (ctypes.c_uint32 * LANES)(*([0] * LANES))
  args = (ctypes.c_uint64 * 1)(ctypes.addressof(out_buf))
  code = _code(instructions, out_reg)
  kernel_buf = (ctypes.c_char * len(code)).from_buffer_copy(code)
  result = run_asm(ctypes.addressof(kernel_buf), len(code), 1, 1, 1, LANES, 1, 1, ctypes.addressof(args),
                   0x19c | (128 << 15), 0x10000, arch="cdna")
  assert result == 0, f"run_asm failed with {result}"
  return out_buf[0]

def _run_hw(instructions: list, out_reg: int = 2) -> int:
  from tinygrad.device import Device
  from tinygrad.runtime.ops_amd import AMDProgram
  from tinygrad.runtime.support.compiler_amd import HIPCompiler

  dev = Device["AMD"]
  if dev.arch != "gfx950": raise unittest.SkipTest("requires gfx950 hardware")
  out_gpu = dev.allocator.alloc(LANES * 4)
  code = _code(instructions, out_reg, out_gpu.va_addr)
  byte_str = ", ".join(f"0x{b:02x}" for b in code)
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
  .amdhsa_accum_offset 256
  .amdhsa_kernarg_size 0
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version:
  - 1
  - 0
amdhsa.kernels:
  - .name: test
    .symbol: test.kd
    .kernarg_segment_size: 0
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 96
    .vgpr_count: 256
    .max_flat_workgroup_size: 1024
...
.end_amdgpu_metadata
"""
  prg = AMDProgram(dev, "test", HIPCompiler(dev.arch).compile(asm_src))
  prg(global_size=(1, 1, 1), local_size=(LANES, 1, 1), wait=True)
  out = bytearray(LANES * 4)
  dev.allocator._copyout(flat_mv(memoryview(out)), out_gpu)
  return struct.unpack("<I", out)[0]

def run_cdna(instructions: list, out_reg: int = 2) -> int:
  emu = _run_emu(instructions, out_reg)
  if not USE_HW: return emu
  hw = _run_hw(instructions, out_reg)
  if emu != hw: raise AssertionError(f"Emulator vs Hardware mismatch: emu=0x{emu:08x} hw=0x{hw:08x}")
  return hw

class TestCDNAVOP3(unittest.TestCase):
  def test_cvt_pk_fp8_f32_preserves_upper_half(self):
    """V_CVT_PK_FP8_F32 with OPSEL[3]=0 writes only D[15:0]."""
    out = run_cdna([
      cdna.s_mov_b32(cdna.s[0], 0xdeadbeef),
      cdna.v_mov_b32_e32(cdna.v[2], cdna.s[0]),
      cdna.v_mov_b32_e32(cdna.v[0], 1.0),
      cdna.v_mov_b32_e32(cdna.v[1], 2.0),
      cdna.v_cvt_pk_fp8_f32(cdna.v[2], cdna.v[0], cdna.v[1]),
    ])
    self.assertEqual(out, 0xdead4038)

  def test_cvt_pk_bf8_f32_overflow_and_inf(self):
    """V_CVT_PK_BF8_F32 converts finite overflow and infinities to E5M2 infinities."""
    for name, bits, expected in [
      ("finite_overflow", 0x47700000, 0x7c),
      ("pos_inf", 0x7f800000, 0x7c),
      ("neg_inf", 0xff800000, 0xfc),
    ]:
      with self.subTest(name=name):
        out = run_cdna([
          cdna.s_mov_b32(cdna.s[0], 0xdeadbeef),
          cdna.v_mov_b32_e32(cdna.v[2], cdna.s[0]),
          cdna.s_mov_b32(cdna.s[0], bits),
          cdna.v_mov_b32_e32(cdna.v[0], cdna.s[0]),
          cdna.v_mov_b32_e32(cdna.v[1], 1.0),
          cdna.v_cvt_pk_bf8_f32(cdna.v[2], cdna.v[0], cdna.v[1]),
        ])
        self.assertEqual(out, 0xdead3c00 | expected)
