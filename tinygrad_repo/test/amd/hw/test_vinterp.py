"""Tests for VINTERP instructions."""
import unittest
from test.amd.hw.helpers import *

class TestVInterp(unittest.TestCase):
  def test_v_interp_p10_f32(self):
    instructions = [
      v_mov_b32_e32(v[10], v[255]),
      v_cvt_f32_u32_e32(v[1], v[10]),
      s_mov_b32(s[0], f2i(100.0)),
      v_add_f32_e32(v[1], s[0], v[1]),
      v_cvt_f32_u32_e32(v[3], v[10]),
      s_mov_b32(s[1], f2i(10.0)),
      v_add_f32_e32(v[3], s[1], v[3]),
      s_mov_b32(s[2], f2i(2.0)),
      v_interp_p10_f32(v[4], v[1], s[2], v[3]),
    ]
    st = run_program(instructions, n_lanes=8)
    for lane in range(4): self.assertAlmostEqual(i2f(st.vgpr[lane][4]), 212.0, places=5)
    for lane in range(4, 8): self.assertAlmostEqual(i2f(st.vgpr[lane][4]), 224.0, places=5)

  def test_v_interp_p10_f16_f32(self):
    instructions = [
      v_mov_b32_e32(v[10], v[255]),
      v_cvt_f32_u32_e32(v[11], v[10]),
      v_cvt_f16_f32_e32(v[1], v[11]),
      s_mov_b32(s[0], f2i(10.0)),
      v_add_f32_e32(v[12], s[0], v[11]),
      v_cvt_f16_f32_e32(v[3], v[12]),
      s_mov_b32(s[1], f2i(2.0)),
      v_interp_p10_f16_f32(v[4], v[1], s[1], v[3]),
    ]
    st = run_program(instructions, n_lanes=8)
    for lane in range(4): self.assertAlmostEqual(i2f(st.vgpr[lane][4]), 12.0, places=5)
    for lane in range(4, 8): self.assertAlmostEqual(i2f(st.vgpr[lane][4]), 24.0, places=5)
