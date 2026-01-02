# do not change these tests. we need to fix bugs to make them pass
# the Inst constructor should be looking at the types of the fields to correctly set the value

import unittest
from extra.assembly.rdna3.autogen import *
from extra.assembly.rdna3.lib import Inst
from extra.assembly.rdna3.asm import asm
from extra.assembly.rdna3.test.test_roundtrip import compile_asm

class TestIntegration(unittest.TestCase):
  inst: Inst
  def tearDown(self):
    if not hasattr(self, 'inst'): return
    b = self.inst.to_bytes()
    st = self.inst.disasm()
    reasm = asm(st)
    desc = f"{st:25s} {self.inst} {b!r} {reasm}"
    self.assertEqual(b, compile_asm(st), desc)
    # TODO: this compare should work for valid things
    #self.assertEqual(self.inst, reasm)
    self.assertEqual(repr(self.inst), repr(reasm))
    print(desc)

  def test_load_b128(self):
    self.inst = s_load_b128(s[4:7], s[0:1], NULL, 0)

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

if __name__ == "__main__":
  unittest.main()
