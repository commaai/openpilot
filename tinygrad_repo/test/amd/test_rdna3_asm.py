#!/usr/bin/env python3
import unittest
from tinygrad.runtime.autogen.amd.rdna3.ins import *
from test.amd.helpers import llvm_assemble
from test.amd.disasm import disasm

def _asm(asm: str) -> bytes: return llvm_assemble([asm], 'gfx1100', '+real-true16,+wavefrontsize32')[0]

class TestRDNA3Asm(unittest.TestCase):
  def test_full_program(self):
    """Test the full program from rdna3fun.py matches LLVM output."""
    program = [
      v_bfe_u32(v[1], v[0], 10, 10),
      s_load_b128(s[4:7], s[0:1], NULL),
      v_and_b32_e32(v[0], 0x3FF, v[0]),
      s_mulk_i32(s[3], 0x87),
      v_mad_u64_u32(v[1:2], NULL, s[2], 3, v[1:2]),
      v_mul_u32_u24_e32(v[0], 45, v[0]),
      v_ashrrev_i32_e32(v[2], 31, v[1]),
      v_add3_u32(v[0], v[0], s[3], v[1]),
      v_lshlrev_b64(v[2:3], 2, v[1:2]),
      v_ashrrev_i32_e32(v[1], 31, v[0]),
      v_lshlrev_b64(v[0:1], 2, v[0:1]),
      s_waitcnt(0xfc07),  # lgkmcnt(0)
      v_add_co_u32(v[2], VCC_LO, s[6], v[2]),
      v_add_co_ci_u32_e32(v[3], s[7], v[3]),
      v_add_co_u32(v[0], VCC_LO, s[4], v[0]),
      global_load_b32(vdst=v[2], addr=v[2:3], saddr=OFF),
      v_add_co_ci_u32_e32(v[1], s[5], v[1]),
      s_waitcnt(0x03f7),  # vmcnt(0)
      global_store_b32(addr=v[0:1], data=v[2], saddr=OFF),
      s_endpgm(),
    ]

    asm_lines = [
      "v_bfe_u32 v1, v0, 10, 10", "s_load_b128 s[4:7], s[0:1], null", "v_and_b32_e32 v0, 0x3FF, v0",
      "s_mulk_i32 s3, 0x87", "v_mad_u64_u32 v[1:2], null, s2, 3, v[1:2]", "v_mul_u32_u24_e32 v0, 45, v0",
      "v_ashrrev_i32_e32 v2, 31, v1", "v_add3_u32 v0, v0, s3, v1", "v_lshlrev_b64 v[2:3], 2, v[1:2]",
      "v_ashrrev_i32_e32 v1, 31, v0", "v_lshlrev_b64 v[0:1], 2, v[0:1]", "s_waitcnt lgkmcnt(0)",
      "v_add_co_u32 v2, vcc_lo, s6, v2", "v_add_co_ci_u32_e32 v3, vcc_lo, s7, v3, vcc_lo",
      "v_add_co_u32 v0, vcc_lo, s4, v0", "global_load_b32 v2, v[2:3], off",
      "v_add_co_ci_u32_e32 v1, vcc_lo, s5, v1, vcc_lo", "s_waitcnt vmcnt(0)",
      "global_store_b32 v[0:1], v2, off", "s_endpgm",
    ]
    expected = llvm_assemble(asm_lines, 'gfx1100', '+real-true16,+wavefrontsize32')
    for inst, rt in zip(program, asm_lines): print(f"{disasm(inst):50s} {rt}")
    for inst, exp in zip(program, expected): self.assertEqual(inst.to_bytes(), exp)

  def test_sop2_s_add_u32(self):
    inst = SOP2(SOP2Op.S_ADD_U32, s[3], s[0], s[1])
    self.assertEqual(inst.to_bytes(), _asm("s_add_u32 s3, s0, s1"))

  def test_vop2_v_and_b32_inline_const(self):
    inst = v_and_b32_e32(v[0], 10, v[0])
    self.assertEqual(inst.to_bytes(), _asm("v_and_b32_e32 v0, 10, v0"))

  def test_sopp_s_endpgm(self):
    inst = s_endpgm()
    self.assertEqual(inst.to_bytes(), _asm("s_endpgm"))

  def test_sop1_s_mov_b32(self):
    inst = s_mov_b32(s[0], s[1])
    self.assertEqual(inst.to_bytes(), _asm("s_mov_b32 s0, s1"))

if __name__ == "__main__":
  unittest.main()
