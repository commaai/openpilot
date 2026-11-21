"""RDNA4 V_PERMLANE16_VAR_B32 / V_PERMLANEX16_VAR_B32 coverage.

Exercises the generated pcode path end-to-end in the emulator and compares against
real RDNA4 hardware when USE_HW=1.
"""
import unittest
import tinygrad.runtime.autogen.amd.rdna4.ins as r4
from test.amd.hw.helpers import run_rdna4

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
