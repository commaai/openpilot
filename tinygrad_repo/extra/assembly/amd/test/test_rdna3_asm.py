#!/usr/bin/env python3
import unittest, subprocess
from extra.assembly.amd.autogen.rdna3.ins import *
from extra.assembly.amd.test.helpers import get_llvm_mc

def llvm_assemble(asm: str) -> bytes:
  """Assemble using llvm-mc and return bytes."""
  result = subprocess.run(
    [get_llvm_mc(), "-triple=amdgcn", "-mcpu=gfx1100", "-show-encoding"],
    input=asm, capture_output=True, text=True
  )
  out = b''
  for line in result.stdout.split('\n'):
    if 'encoding:' in line:
      enc = line.split('encoding:')[1].strip()
      enc = enc.strip('[]').replace('0x', '').replace(',', '')
      out += bytes.fromhex(enc)
  if not out: raise ValueError(f"no encoding found: {result.stdout} {result.stderr}")
  return out

class TestRDNA3Asm(unittest.TestCase):
  def test_full_program(self):
    """Test the full program from rdna3fun.py matches llvm-mc output."""
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

    asm = """
v_bfe_u32 v1, v0, 10, 10
s_load_b128 s[4:7], s[0:1], null
v_and_b32_e32 v0, 0x3FF, v0
s_mulk_i32 s3, 0x87
v_mad_u64_u32 v[1:2], null, s2, 3, v[1:2]
v_mul_u32_u24_e32 v0, 45, v0
v_ashrrev_i32_e32 v2, 31, v1
v_add3_u32 v0, v0, s3, v1
v_lshlrev_b64 v[2:3], 2, v[1:2]
v_ashrrev_i32_e32 v1, 31, v0
v_lshlrev_b64 v[0:1], 2, v[0:1]
s_waitcnt lgkmcnt(0)
v_add_co_u32 v2, vcc_lo, s6, v2
v_add_co_ci_u32_e32 v3, vcc_lo, s7, v3, vcc_lo
v_add_co_u32 v0, vcc_lo, s4, v0
global_load_b32 v2, v[2:3], off
v_add_co_ci_u32_e32 v1, vcc_lo, s5, v1, vcc_lo
s_waitcnt vmcnt(0)
global_store_b32 v[0:1], v2, off
s_endpgm
"""
    expected = llvm_assemble(asm)
    for inst,rt in zip(program, asm.strip().split("\n")): print(f"{inst.disasm():50s} {rt}")
    actual = b''.join(inst.to_bytes() for inst in program)
    self.assertEqual(actual, expected)

  def test_sop2_s_add_u32(self):
    inst = SOP2(SOP2Op.S_ADD_U32, s[3], s[0], s[1])
    expected = llvm_assemble("s_add_u32 s3, s0, s1")
    self.assertEqual(inst.to_bytes(), expected)

  def test_vop2_v_and_b32_inline_const(self):
    inst = v_and_b32_e32(v[0], 10, v[0])
    expected = llvm_assemble("v_and_b32_e32 v0, 10, v0")
    self.assertEqual(inst.to_bytes(), expected)

  def test_sopp_s_endpgm(self):
    inst = s_endpgm()
    expected = llvm_assemble("s_endpgm")
    self.assertEqual(inst.to_bytes(), expected)

  def test_sop1_s_mov_b32(self):
    inst = s_mov_b32(s[0], s[1])
    expected = llvm_assemble("s_mov_b32 s0, s1")
    self.assertEqual(inst.to_bytes(), expected)

if __name__ == "__main__":
  unittest.main()
