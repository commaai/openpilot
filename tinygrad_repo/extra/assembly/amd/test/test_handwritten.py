# do not change these tests. we need to fix bugs to make them pass
# the Inst constructor should be looking at the types of the fields to correctly set the value

import unittest, struct
from extra.assembly.amd.autogen.rdna3.ins import *
from extra.assembly.amd.dsl import Inst
from extra.assembly.amd.test.test_roundtrip import compile_asm

class IntegrationTestBase(unittest.TestCase):
  inst: Inst
  arch: str
  def tearDown(self):
    if not hasattr(self, 'inst'): return
    b = self.inst.to_bytes()
    st = self.inst.disasm()
    # Test that the instruction can be compiled by LLVM and produces the same bytes
    desc = f"{st:25s} {self.inst} {b!r}"
    self.assertEqual(b, compile_asm(st, arch=self.arch), desc)
    print(desc)

class TestIntegration(IntegrationTestBase):
  arch: str = "rdna3"

  def test_wmma(self):
    self.inst = v_wmma_f32_16x16x16_f16(v[0:7], v[184:191], v[136:143], v[0:7])

  def test_load_b128(self):
    self.inst = s_load_b128(s[4:7], s[0:1], NULL, 0)

  def test_load_b128_wrong_size(self):
    # this should have to be 4 regs on the loaded to
    with self.assertRaises(TypeError):
      self.inst = s_load_b128(s[4:6], s[0:1], NULL, 0)

  def test_mov_b32(self):
    self.inst = s_mov_b32(s[80], s[0])

  def test_mov_b64(self):
    self.inst = s_mov_b64(s[80:81], s[0:1])

  def test_mov_b32_wrong(self):
    with self.assertRaises(Exception):
      self.inst = s_mov_b32(s[80:81], s[0:1])
    with self.assertRaises(Exception):
      self.inst = s_mov_b32(s[80:81], s[0])
    with self.assertRaises(Exception):
      self.inst = s_mov_b32(s[80], s[0:1])

  def test_mov_b64_wrong(self):
    with self.assertRaises(Exception):
      self.inst = s_mov_b64(s[80], s[0])
    with self.assertRaises(Exception):
      self.inst = s_mov_b64(s[80], s[0:1])
    with self.assertRaises(Exception):
      self.inst = s_mov_b64(s[80:81], s[0])

  def test_load_b128_no_0(self):
    self.inst = s_load_b128(s[4:7], s[0:1], NULL)

  def test_load_b128_s(self):
    self.inst = s_load_b128(s[4:7], s[0:1], s[8], 0)

  def test_load_b128_v(self):
    with self.assertRaises(TypeError):
      self.inst = s_load_b128(s[4:7], s[0:1], v[8], 0)

  def test_load_b128_off(self):
    self.inst = s_load_b128(s[4:7], s[0:1], NULL, 3)

  def test_simple_stos(self):
    self.inst = s_mov_b32(s[0], s[1])

  def test_simple_wrong(self):
    with self.assertRaises(TypeError):
      self.inst = s_mov_b32(v[0], s[1])

  def test_simple_vtov(self):
    self.inst = v_mov_b32_e32(v[0], v[1])

  def test_simple_stov(self):
    self.inst = v_mov_b32_e32(v[0], s[2])

  def test_simple_float_to_v(self):
    self.inst = v_mov_b32_e32(v[0], 1.0)

  def test_simple_v_to_float(self):
    with self.assertRaises(TypeError):
      self.inst = v_mov_b32_e32(1, v[0])

  def test_simple_int_to_v(self):
    self.inst = v_mov_b32_e32(v[0], 1)

  def test_three_add(self):
    self.inst = v_add_co_ci_u32_e32(v[3], s[7], v[3])

  def test_three_add_v(self):
    self.inst = v_add_co_ci_u32_e32(v[3], v[7], v[3])

  def test_three_add_const(self):
    self.inst = v_add_co_ci_u32_e32(v[3], 2.0, v[3])

  def test_swaitcnt_lgkm(self): self.inst = s_waitcnt(0xfc07)
  def test_swaitcnt_vm(self): self.inst = s_waitcnt(0x03f7)

  def test_vmad(self):
    self.inst = v_mad_u64_u32(v[1:2], NULL, s[2], 3, v[1:2])

  def test_large_imm(self):
    self.inst = v_mov_b32_e32(v[0], 0x1234)

  def test_dual_mov(self):
    self.inst = VOPD(VOPDOp.V_DUAL_MOV_B32, VOPDOp.V_DUAL_MOV_B32, vdstx=v[0], vdsty=v[1], srcx0=v[2], srcy0=v[4])

  def test_dual_mul(self):
    self.inst = v_dual_mul_f32(VOPDOp.V_DUAL_MUL_F32, vdstx=v[0], vdsty=v[1], srcx0=v[2], vsrcx1=v[3], srcy0=v[4], vsrcy1=v[5])

  def test_simple_int_to_s(self):
    self.inst = s_mov_b32(s[0], 3)

  def test_complex_int_to_s(self):
    self.inst = s_mov_b32(s[0], 0x235646)

  def test_simple_float_to_s(self):
    self.inst = s_mov_b32(s[0], 1.0)

  def test_complex_float_to_s(self):
    self.inst = s_mov_b32(s[0], 1337.0)
    int_inst = s_mov_b32(s[0], struct.unpack("I", struct.pack("f", 1337.0))[0])
    self.assertEqual(self.inst, int_inst)

class TestIntegrationCDNA(IntegrationTestBase):
  arch = "cdna"

  def test_mfma(self):
    from extra.assembly.amd.autogen.cdna.ins import v_mfma_f32_16x16x16_f16
    self.inst = v_mfma_f32_16x16x16_f16(v[0:3], v[0:1], v[0:1], 0)

  def test_mfma_fp8(self):
    from extra.assembly.amd.autogen.cdna.ins import v_mfma_f32_16x16x128_f8f6f4
    self.inst = v_mfma_f32_16x16x128_f8f6f4(v[0:3], v[0:5], v[0:5], 1, cbsz=2, blgp=2)

class TestRegisterSliceSyntax(unittest.TestCase):
  """
  Issue: Register slice syntax should use AMD assembly convention (inclusive end).

  In AMD assembly, s[4:7] means registers s4, s5, s6, s7 (4 registers, inclusive).
  The DSL should match this convention so that:
  - s[4:7] gives 4 registers
  - Disassembler output can be copied directly back into DSL code

  Fix: Change _RegFactory.__getitem__ to use inclusive end:
    key.stop - key.start + 1  (instead of key.stop - key.start)
  """
  def test_register_slice_count(self):
    # s[4:7] should give 4 registers: s4, s5, s6, s7 (AMD convention, inclusive)
    reg = s[4:7]
    self.assertEqual(reg.sz, 4, "s[4:7] should give 4 registers (s4, s5, s6, s7)")

  def test_register_slice_roundtrip(self):
    # Round-trip: DSL -> disasm -> DSL should preserve register count
    reg = s[4:7]  # 4 registers in AMD convention
    inst = s_load_b128(reg, s[0:1], NULL, 0)
    disasm = inst.disasm()
    # Disasm shows s[4:7] - user should be able to copy this back
    self.assertIn("s[4:7]", disasm)
    # And s[4:7] in DSL should give the same 4 registers
    reg_from_disasm = s[4:7]
    self.assertEqual(reg_from_disasm.sz, 4, "s[4:7] from disasm should give 4 registers")

class TestInstructionEquality(unittest.TestCase):
  """
  Issue: No __eq__ method - instruction comparison requires repr() workaround.

  Two identical instructions should compare equal with ==, but currently:
    inst1 == inst2 returns False

  The test_handwritten.py works around this with:
    self.assertEqual(repr(self.inst), repr(reasm))
  """
  def test_identical_instructions_equal(self):
    inst1 = v_mov_b32_e32(v[0], v[1])
    inst2 = v_mov_b32_e32(v[0], v[1])
    self.assertEqual(inst1, inst2, "identical instructions should be equal")

  def test_different_instructions_not_equal(self):
    inst1 = v_mov_b32_e32(v[0], v[1])
    inst2 = v_mov_b32_e32(v[0], v[2])
    self.assertNotEqual(inst1, inst2, "different instructions should not be equal")


if __name__ == "__main__":
  unittest.main()
