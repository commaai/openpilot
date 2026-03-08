"""Tests for VOP3 instructions - three operand vector operations.

Includes: v_fma_f32, v_div_scale_f32, v_div_fmas_f32, v_div_fixup_f32,
          v_alignbit_b32, v_bfe_i32, v_mad_u64_u32, v_readlane_b32, v_writelane_b32
"""
import unittest
from extra.assembly.amd.test.hw.helpers import *

class TestFMA(unittest.TestCase):
  """Tests for FMA instructions."""

  def test_v_fma_f32_basic(self):
    """V_FMA_F32: a*b+c basic case."""
    instructions = [
      v_mov_b32_e32(v[0], 2.0),
      v_mov_b32_e32(v[1], 4.0),
      v_mov_b32_e32(v[2], 1.0),
      v_fma_f32(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][3]), 9.0, places=5)

  def test_v_fma_f32_negative(self):
    """V_FMA_F32 with negative multiplier."""
    instructions = [
      v_mov_b32_e32(v[0], -2.0),
      v_mov_b32_e32(v[1], 4.0),
      v_mov_b32_e32(v[2], 1.0),
      v_fma_f32(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][3]), -7.0, places=5)

  def test_v_fma_f32_with_sgpr(self):
    """V_FMA_F32: using SGPR for non-inline constant."""
    instructions = [
      s_mov_b32(s[0], f2i(3.0)),
      v_mov_b32_e32(v[0], 2.0),
      v_mov_b32_e32(v[1], s[0]),
      v_mov_b32_e32(v[2], 4.0),
      v_fma_f32(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][3]), 10.0, places=5)

  def test_v_fma_f32_with_inf(self):
    """V_FMA_F32: 1.0 * inf + 0 = inf."""
    import math
    instructions = [
      v_mov_b32_e32(v[0], 1.0),
      s_mov_b32(s[0], 0x7f800000),
      v_mov_b32_e32(v[1], s[0]),
      v_mov_b32_e32(v[2], 0),
      v_fma_f32(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i2f(st.vgpr[0][3])
    self.assertTrue(math.isinf(result) and result > 0)


class TestFmacE64(unittest.TestCase):
  """Regression tests for V_FMAC_F32 VOP3 encoding (e64).

  V_FMAC_F32: D0 = D0 + S0 * S1 (fused multiply-add with accumulator)

  The VOP3 encoding needs to read D0 from the destination register as the
  accumulator input, not just write to it.

  Regression test for: VOP3 FMAC missing D0 accumulator bug.
  """

  def test_v_fmac_f32_e64_basic(self):
    """V_FMAC_F32_E64: basic accumulate test."""
    instructions = [
      v_mov_b32_e32(v[0], 2.0),  # S0 = 2.0
      v_mov_b32_e32(v[1], 3.0),  # S1 = 3.0
      v_mov_b32_e32(v[2], 1.0),  # D0 (accumulator) = 1.0
      # v_fmac_f32_e64 v[2], v[0], v[1]
      # D0 = D0 + S0 * S1 = 1.0 + 2.0 * 3.0 = 7.0
      v_fmac_f32_e64(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][2]), 7.0, places=5)

  def test_v_fmac_f32_e64_with_sgpr_sources(self):
    """V_FMAC_F32_E64 with SGPR sources (common in AMD_LLVM output).

    This tests the exact pattern that was failing: v_fmac_f32_e64(v[0], s[4], 0)
    where src0 is SGPR and src1 is inline constant 0.

    Regression test for: VOP3 FMAC missing D0 accumulator bug.
    """
    instructions = [
      s_mov_b32(s[4], f2i(2.0)),  # S0 = 2.0 in SGPR
      v_mov_b32_e32(v[0], 5.0),   # D0 (accumulator) = 5.0
      # v_fmac_f32_e64 v[0], s[4], 0
      # D0 = D0 + S0 * S1 = 5.0 + 2.0 * 0.0 = 5.0
      v_fmac_f32_e64(v[0], s[4], 0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][0]), 5.0, places=5)

  def test_v_fmac_f32_e64_with_two_sgprs(self):
    """V_FMAC_F32_E64 with two SGPR sources.

    Tests pattern: v_fmac_f32_e64(v[0], s[a], s[b])

    Regression test for: VOP3 FMAC missing D0 accumulator bug.
    """
    instructions = [
      s_mov_b32(s[10], f2i(3.0)),  # S0 = 3.0
      s_mov_b32(s[12], f2i(4.0)),  # S1 = 4.0
      v_mov_b32_e32(v[9], 2.0),    # D0 (accumulator) = 2.0
      # v_fmac_f32_e64 v[9], s[10], s[12]
      # D0 = D0 + S0 * S1 = 2.0 + 3.0 * 4.0 = 14.0
      v_fmac_f32_e64(v[9], s[10], s[12]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][9]), 14.0, places=5)

  def test_v_fmac_f32_e64_accumulates_correctly(self):
    """V_FMAC_F32_E64 accumulates multiple times."""
    instructions = [
      v_mov_b32_e32(v[0], 0.0),   # D0 = 0.0
      v_mov_b32_e32(v[1], 1.0),   # S0 = 1.0
      v_mov_b32_e32(v[2], 2.0),   # S1 = 2.0
      # First: D0 = 0.0 + 1.0 * 2.0 = 2.0
      v_fmac_f32_e64(v[0], v[1], v[2]),
      # Second: D0 = 2.0 + 1.0 * 2.0 = 4.0
      v_fmac_f32_e64(v[0], v[1], v[2]),
      # Third: D0 = 4.0 + 1.0 * 2.0 = 6.0
      v_fmac_f32_e64(v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][0]), 6.0, places=5)

  def test_v_fmac_f32_e64_negative_accumulator(self):
    """V_FMAC_F32_E64 with negative accumulator."""
    instructions = [
      v_mov_b32_e32(v[0], 2.0),   # S0 = 2.0
      v_mov_b32_e32(v[1], 3.0),   # S1 = 3.0
      v_mov_b32_e32(v[2], -10.0), # D0 (accumulator) = -10.0
      # D0 = -10.0 + 2.0 * 3.0 = -4.0
      v_fmac_f32_e64(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][2]), -4.0, places=5)


class TestDivScale(unittest.TestCase):
  """Tests for V_DIV_SCALE_F32."""

  def test_div_scale_f32_vcc_zero_single_lane(self):
    """V_DIV_SCALE_F32 sets VCC=0 when no scaling needed."""
    instructions = [
      v_mov_b32_e32(v[0], 1.0),
      v_mov_b32_e32(v[1], 4.0),
      v_div_scale_f32(v[2], VCC, v[0], v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc, 0, "VCC should be 0 when no scaling needed")

  def test_div_scale_f32_vcc_zero_multiple_lanes(self):
    """V_DIV_SCALE_F32 sets VCC=0 for all lanes when no scaling needed."""
    instructions = [
      v_mov_b32_e32(v[0], 1.0),
      v_mov_b32_e32(v[1], 4.0),
      v_div_scale_f32(v[2], VCC, v[0], v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=4)
    self.assertEqual(st.vcc & 0xf, 0, "VCC should be 0 for all lanes")

  def test_div_scale_f32_preserves_input(self):
    """V_DIV_SCALE_F32 outputs S0 when no scaling needed."""
    instructions = [
      v_mov_b32_e32(v[0], 2.0),
      v_mov_b32_e32(v[1], 4.0),
      v_div_scale_f32(v[2], VCC, v[0], v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][2]), 2.0, places=5)

  def test_div_scale_f32_zero_denom_gives_nan(self):
    """V_DIV_SCALE_F32: zero denominator -> NaN, VCC=1."""
    import math
    instructions = [
      v_mov_b32_e32(v[0], 1.0),
      v_mov_b32_e32(v[1], 0.0),
      v_div_scale_f32(v[2], VCC, v[0], v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertTrue(math.isnan(i2f(st.vgpr[0][2])), "Should be NaN for zero denom")
    self.assertEqual(st.vcc & 1, 1, "VCC should be 1 for zero denom")

  def test_div_scale_f32_zero_numer_gives_nan(self):
    """V_DIV_SCALE_F32: zero numerator -> NaN, VCC=1."""
    import math
    instructions = [
      v_mov_b32_e32(v[0], 0.0),
      v_mov_b32_e32(v[1], 1.0),
      v_div_scale_f32(v[2], VCC, v[0], v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertTrue(math.isnan(i2f(st.vgpr[0][2])), "Should be NaN for zero numer")
    self.assertEqual(st.vcc & 1, 1, "VCC should be 1 for zero numer")

  def test_div_scale_f32_large_exp_diff_scales_denom(self):
    """V_DIV_SCALE_F32: exp(numer) - exp(denom) >= 96 -> scale denom, VCC=1."""
    max_float = 0x7f7fffff  # 3.4028235e+38, exp=254
    instructions = [
      s_mov_b32(s[0], max_float),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], 1.0),
      v_div_scale_f32(v[2], VCC, v[1], v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "VCC should be 1 when scaling denom for large exp diff")
    expected = 1.0 * (2.0 ** 64)
    self.assertAlmostEqual(i2f(st.vgpr[0][2]), expected, delta=expected * 1e-6)

  def test_div_scale_f32_denorm_denom(self):
    """V_DIV_SCALE_F32: denormalized denominator -> NaN, VCC=1."""
    import math
    denorm = 0x00000001
    instructions = [
      s_mov_b32(s[0], denorm),
      v_mov_b32_e32(v[0], 1.0),
      v_mov_b32_e32(v[1], s[0]),
      v_div_scale_f32(v[2], VCC, v[1], v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertTrue(math.isnan(i2f(st.vgpr[0][2])), "Hardware returns NaN for denorm denom")
    self.assertEqual(st.vcc & 1, 1, "VCC should be 1 for denorm denom")

  def test_div_scale_f32_tiny_numer_exp_le_23(self):
    """V_DIV_SCALE_F32: exponent(numer) <= 23 -> scale by 2^64, VCC=1."""
    smallest_normal = 0x00800000
    instructions = [
      s_mov_b32(s[0], smallest_normal),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], 1.0),
      v_div_scale_f32(v[2], VCC, v[0], v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    numer_f = i2f(smallest_normal)
    expected = numer_f * (2.0 ** 64)
    self.assertAlmostEqual(i2f(st.vgpr[0][2]), expected, delta=abs(expected) * 1e-5)
    self.assertEqual(st.vcc & 1, 1, "VCC should be 1 when scaling tiny numer")

  def test_div_scale_f32_result_would_be_denorm(self):
    """V_DIV_SCALE_F32: result would be denorm -> no scaling, VCC=1."""
    large_denom = 0x7f000000  # 2^127
    instructions = [
      s_mov_b32(s[0], large_denom),
      v_mov_b32_e32(v[0], 1.0),   # numer = 1.0 (S2)
      v_mov_b32_e32(v[1], s[0]),  # denom = 2^127 (S1)
      v_div_scale_f32(v[2], VCC, v[0], v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][2]), 1.0, places=5)
    self.assertEqual(st.vcc & 1, 1, "VCC should be 1 when result would be denorm")


class TestDivFmas(unittest.TestCase):
  """Tests for V_DIV_FMAS_F32."""

  def test_div_fmas_f32_no_scale(self):
    """V_DIV_FMAS_F32: VCC=0 -> normal FMA."""
    instructions = [
      s_mov_b32(VCC_LO, 0),
      v_mov_b32_e32(v[0], 2.0),
      v_mov_b32_e32(v[1], 3.0),
      v_mov_b32_e32(v[2], 1.0),
      v_div_fmas_f32(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][3]), 7.0, places=5)

  def test_div_fmas_f32_scale_up(self):
    """V_DIV_FMAS_F32: VCC=1 with S2 >= 2.0 -> scale by 2^+64."""
    instructions = [
      s_mov_b32(VCC_LO, 1),
      v_mov_b32_e32(v[0], 1.0),
      v_mov_b32_e32(v[1], 1.0),
      v_mov_b32_e32(v[2], 2.0),
      v_div_fmas_f32(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    expected = 3.0 * (2.0 ** 64)
    self.assertAlmostEqual(i2f(st.vgpr[0][3]), expected, delta=abs(expected) * 1e-6)

  def test_div_fmas_f32_scale_down(self):
    """V_DIV_FMAS_F32: VCC=1 with S2 < 2.0 -> scale by 2^-64."""
    instructions = [
      s_mov_b32(VCC_LO, 1),
      v_mov_b32_e32(v[0], 2.0),
      v_mov_b32_e32(v[1], 3.0),
      v_mov_b32_e32(v[2], 1.0),
      v_div_fmas_f32(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    expected = 7.0 * (2.0 ** -64)
    self.assertAlmostEqual(i2f(st.vgpr[0][3]), expected, delta=abs(expected) * 1e-6)

  def test_div_fmas_f32_per_lane_vcc(self):
    """V_DIV_FMAS_F32: different VCC per lane with S2 < 2.0."""
    instructions = [
      s_mov_b32(VCC_LO, 0b0101),
      v_mov_b32_e32(v[0], 1.0),
      v_mov_b32_e32(v[1], 1.0),
      v_mov_b32_e32(v[2], 1.0),
      v_div_fmas_f32(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=4)
    scaled = 2.0 * (2.0 ** -64)
    unscaled = 2.0
    self.assertAlmostEqual(i2f(st.vgpr[0][3]), scaled, delta=abs(scaled) * 1e-6)
    self.assertAlmostEqual(i2f(st.vgpr[1][3]), unscaled, places=5)
    self.assertAlmostEqual(i2f(st.vgpr[2][3]), scaled, delta=abs(scaled) * 1e-6)
    self.assertAlmostEqual(i2f(st.vgpr[3][3]), unscaled, places=5)


class TestDivFixup(unittest.TestCase):
  """Tests for V_DIV_FIXUP_F32."""

  def test_div_fixup_f32_normal(self):
    """V_DIV_FIXUP_F32: normal division passes through quotient."""
    instructions = [
      v_mov_b32_e32(v[0], 3.0),
      v_mov_b32_e32(v[1], 2.0),
      v_mov_b32_e32(v[2], 6.0),
      v_div_fixup_f32(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][3]), 3.0, places=5)

  def test_div_fixup_f32_zero_div_zero(self):
    """V_DIV_FIXUP_F32: 0/0 -> NaN."""
    import math
    instructions = [
      v_mov_b32_e32(v[0], 1.0),
      v_mov_b32_e32(v[1], 0.0),
      v_mov_b32_e32(v[2], 0.0),
      v_div_fixup_f32(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertTrue(math.isnan(i2f(st.vgpr[0][3])), "0/0 should be NaN")

  def test_div_fixup_f32_x_div_zero(self):
    """V_DIV_FIXUP_F32: x/0 -> +/-inf based on sign."""
    import math
    instructions = [
      v_mov_b32_e32(v[0], 1.0),
      v_mov_b32_e32(v[1], 0.0),
      v_mov_b32_e32(v[2], 1.0),
      v_div_fixup_f32(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertTrue(math.isinf(i2f(st.vgpr[0][3])), "x/0 should be inf")

  def test_div_fixup_f32_one_div_inf(self):
    """V_DIV_FIXUP_F32: 1.0 / +inf = 0."""
    instructions = [
      s_mov_b32(s[0], 0),           # approximation (rcp of inf = 0)
      s_mov_b32(s[1], 0x7f800000),  # denominator = +inf
      s_mov_b32(s[2], f2i(1.0)),    # numerator = 1.0
      v_mov_b32_e32(v[0], s[0]),
      v_div_fixup_f32(v[1], v[0], s[1], s[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(i2f(st.vgpr[0][1]), 0.0)

  def test_div_fixup_f32_inf_div_inf(self):
    """V_DIV_FIXUP_F32: inf / inf = NaN."""
    import math
    instructions = [
      s_mov_b32(s[0], 0),           # approximation
      s_mov_b32(s[1], 0x7f800000),  # denominator = +inf
      s_mov_b32(s[2], 0x7f800000),  # numerator = +inf
      v_mov_b32_e32(v[0], s[0]),
      v_div_fixup_f32(v[1], v[0], s[1], s[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertTrue(math.isnan(i2f(st.vgpr[0][1])))

  def test_div_fixup_f32_nan_numer(self):
    """V_DIV_FIXUP_F32: NaN numerator -> quiet NaN."""
    import math
    nan = 0x7fc00000
    instructions = [
      s_mov_b32(s[0], nan),
      v_mov_b32_e32(v[0], 1.0),
      v_mov_b32_e32(v[1], 1.0),
      v_mov_b32_e32(v[2], s[0]),
      v_div_fixup_f32(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertTrue(math.isnan(i2f(st.vgpr[0][3])))

  def test_div_fixup_f32_nan_denom(self):
    """V_DIV_FIXUP_F32: NaN denominator -> quiet NaN."""
    import math
    nan = 0x7fc00000
    instructions = [
      s_mov_b32(s[0], nan),
      v_mov_b32_e32(v[0], 1.0),
      v_mov_b32_e32(v[1], s[0]),
      v_mov_b32_e32(v[2], 1.0),
      v_div_fixup_f32(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertTrue(math.isnan(i2f(st.vgpr[0][3])))

  def test_div_fixup_f32_neg_x_div_zero(self):
    """V_DIV_FIXUP_F32: -x/0 -> -inf."""
    import math
    instructions = [
      v_mov_b32_e32(v[0], 1.0),
      v_mov_b32_e32(v[1], 0.0),
      v_mov_b32_e32(v[2], -1.0),
      v_div_fixup_f32(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertTrue(math.isinf(i2f(st.vgpr[0][3])))
    self.assertLess(i2f(st.vgpr[0][3]), 0, "-1/0 should be -inf")

  def test_div_fixup_f32_zero_div_x(self):
    """V_DIV_FIXUP_F32: 0/x -> 0."""
    instructions = [
      v_mov_b32_e32(v[0], 1.0),
      v_mov_b32_e32(v[1], 2.0),
      v_mov_b32_e32(v[2], 0.0),
      v_div_fixup_f32(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(i2f(st.vgpr[0][3]), 0.0)

  def test_div_fixup_f32_x_div_inf(self):
    """V_DIV_FIXUP_F32: x/inf -> 0."""
    pos_inf = 0x7f800000
    instructions = [
      s_mov_b32(s[0], pos_inf),
      v_mov_b32_e32(v[0], 1.0),
      v_mov_b32_e32(v[1], s[0]),
      v_mov_b32_e32(v[2], 1.0),
      v_div_fixup_f32(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(i2f(st.vgpr[0][3]), 0.0)

  def test_div_fixup_f32_inf_div_x(self):
    """V_DIV_FIXUP_F32: inf/x -> inf."""
    import math
    pos_inf = 0x7f800000
    instructions = [
      s_mov_b32(s[0], pos_inf),
      v_mov_b32_e32(v[0], 1.0),
      v_mov_b32_e32(v[1], 1.0),
      v_mov_b32_e32(v[2], s[0]),
      v_div_fixup_f32(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertTrue(math.isinf(i2f(st.vgpr[0][3])))

  def test_div_fixup_f32_sign_propagation(self):
    """V_DIV_FIXUP_F32: sign is XOR of numer and denom signs."""
    instructions = [
      v_mov_b32_e32(v[0], 3.0),
      v_mov_b32_e32(v[1], -2.0),
      v_mov_b32_e32(v[2], 6.0),
      v_div_fixup_f32(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][3]), -3.0, places=5)

  def test_div_fixup_f32_neg_neg(self):
    """V_DIV_FIXUP_F32: neg/neg -> positive."""
    instructions = [
      v_mov_b32_e32(v[0], 3.0),
      v_mov_b32_e32(v[1], -2.0),
      v_mov_b32_e32(v[2], -6.0),
      v_div_fixup_f32(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][3]), 3.0, places=5)

  def test_div_fixup_f32_nan_estimate_overflow(self):
    """V_DIV_FIXUP_F32: NaN estimate returns overflow (inf)."""
    import math
    quiet_nan = 0x7fc00000
    instructions = [
      s_mov_b32(s[0], quiet_nan),
      v_mov_b32_e32(v[0], s[0]),  # S0 = NaN (failed estimate)
      v_mov_b32_e32(v[1], 1.0),   # S1 = denominator = 1.0
      v_mov_b32_e32(v[2], 1.0),   # S2 = numerator = 1.0
      v_div_fixup_f32(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertTrue(math.isinf(i2f(st.vgpr[0][3])), "NaN estimate should return inf")
    self.assertEqual(st.vgpr[0][3], 0x7f800000, "Should be +inf (pos/pos)")

  def test_div_fixup_f32_nan_estimate_sign(self):
    """V_DIV_FIXUP_F32: NaN estimate with negative sign returns -inf."""
    import math
    quiet_nan = 0x7fc00000
    instructions = [
      s_mov_b32(s[0], quiet_nan),
      v_mov_b32_e32(v[0], s[0]),  # S0 = NaN (failed estimate)
      v_mov_b32_e32(v[1], -1.0),  # S1 = denominator = -1.0
      v_mov_b32_e32(v[2], 1.0),   # S2 = numerator = 1.0
      v_div_fixup_f32(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertTrue(math.isinf(i2f(st.vgpr[0][3])), "NaN estimate should return inf")
    self.assertEqual(st.vgpr[0][3], 0xff800000, "Should be -inf (pos/neg)")

  def test_v_div_fixup_f32_one_div_neg_inf(self):
    """V_DIV_FIXUP_F32: 1/-inf = -0."""
    neg_inf = 0xff800000
    instructions = [
      v_mov_b32_e32(v[0], 0.0),   # estimate (doesn't matter, will be overridden)
      s_mov_b32(s[0], neg_inf),
      v_mov_b32_e32(v[1], s[0]),  # denom = -inf
      v_mov_b32_e32(v[2], 1.0),   # numer = 1.0
      v_div_fixup_f32(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][3], 0x80000000, "1/-inf should be -0")


class TestAlignbit(unittest.TestCase):
  """Tests for V_ALIGNBIT_B32."""

  def test_v_alignbit_b32(self):
    """V_ALIGNBIT_B32 extracts bits from concatenated sources."""
    instructions = [
      s_mov_b32(s[0], 0x12),
      s_mov_b32(s[1], 0x34),
      s_mov_b32(s[2], 4),
      v_mov_b32_e32(v[0], s[2]),
      v_alignbit_b32(v[1], s[0], s[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    expected = ((0x12 << 32) | 0x34) >> 4
    self.assertEqual(st.vgpr[0][1], expected & 0xffffffff)


class TestBfe(unittest.TestCase):
  """Tests for V_BFE_I32."""

  def test_v_bfe_i32_sign_extend(self):
    """V_BFE_I32 sign extends based on MSB of extracted field."""
    instructions = [
      s_mov_b32(s[0], 0x0000007F),  # 0x7F = 0b1111111
      v_mov_b32_e32(v[0], s[0]),
      v_bfe_i32(v[1], v[0], 0, 7),  # Extract 7 bits from offset 0
    ]
    st = run_program(instructions, n_lanes=1)
    # 0x7F in 7 bits has bit 6 = 1 (the sign bit in 7-bit signed)
    # So it represents -1 in 7-bit signed, sign-extended to 32 bits = 0xFFFFFFFF
    self.assertEqual(st.vgpr[0][1], 0xFFFFFFFF)

  def test_v_bfe_i32_sign_extend_negative(self):
    """V_BFE_I32 sign extends negative."""
    instructions = [
      s_mov_b32(s[0], 0x000000FF),  # -1 in 8 bits
      v_mov_b32_e32(v[0], s[0]),
      v_bfe_i32(v[1], v[0], 0, 8),  # Extract 8 bits from offset 0
    ]
    st = run_program(instructions, n_lanes=1)
    # 0xFF in 8 bits is -1, sign-extended to 32 bits = 0xFFFFFFFF
    self.assertEqual(st.vgpr[0][1], 0xFFFFFFFF)


class TestMad64(unittest.TestCase):
  """Tests for V_MAD_U64_U32."""

  def test_v_mad_u64_u32_simple(self):
    """V_MAD_U64_U32: D = S0 * S1 + S2 (64-bit result)."""
    instructions = [
      s_mov_b32(s[0], 3),
      s_mov_b32(s[1], 4),
      v_mov_b32_e32(v[2], 5),
      v_mov_b32_e32(v[3], 0),
      v_mad_u64_u32(v[4:5], SrcEnum.NULL, s[0], s[1], v[2:3]),
    ]
    st = run_program(instructions, n_lanes=1)
    result_lo = st.vgpr[0][4]
    result_hi = st.vgpr[0][5]
    result = result_lo | (result_hi << 32)
    self.assertEqual(result, 17)

  def test_v_mad_u64_u32_large_mult(self):
    """V_MAD_U64_U32 with large values that overflow 32 bits."""
    instructions = [
      s_mov_b32(s[0], 0x80000000),
      s_mov_b32(s[1], 2),
      v_mov_b32_e32(v[2], 0),
      v_mov_b32_e32(v[3], 0),
      v_mad_u64_u32(v[4:5], SrcEnum.NULL, s[0], s[1], v[2:3]),
    ]
    st = run_program(instructions, n_lanes=1)
    result_lo = st.vgpr[0][4]
    result_hi = st.vgpr[0][5]
    result = result_lo | (result_hi << 32)
    self.assertEqual(result, 0x100000000)


class TestLaneOps(unittest.TestCase):
  """Tests for lane operations (readlane, writelane)."""

  def _readlane(self, sdst, vsrc, lane_idx):
    return v_readlane_b32(sdst, vsrc, lane_idx)

  def test_v_readlane_b32_basic(self):
    """V_READLANE_B32 reads a value from a specific lane's VGPR."""
    instructions = [
      v_lshlrev_b32_e32(v[0], 1, v[255]),
      v_lshlrev_b32_e32(v[1], 3, v[255]),
      v_add_nc_u32_e32(v[0], v[0], v[1]),
      self._readlane(s[0], v[0], 2),
      v_mov_b32_e32(v[2], s[0]),
    ]
    st = run_program(instructions, n_lanes=4)
    for lane in range(4):
      self.assertEqual(st.vgpr[lane][2], 20)

  def test_v_readlane_b32_lane_0(self):
    """V_READLANE_B32 reading from lane 0."""
    instructions = [
      v_lshlrev_b32_e32(v[0], 2, v[255]),  # v0 = lane_id * 4
      v_add_nc_u32_e32(v[0], 100, v[0]),   # v0 = 100 + lane_id * 4
      self._readlane(s[0], v[0], 0),       # s0 = lane 0's v0 = 100
      v_mov_b32_e32(v[1], s[0]),
    ]
    st = run_program(instructions, n_lanes=4)
    for lane in range(4):
      self.assertEqual(st.vgpr[lane][1], 100)

  def test_v_readlane_b32_last_lane(self):
    """V_READLANE_B32 reading from the last active lane (lane 3)."""
    instructions = [
      v_lshlrev_b32_e32(v[0], 2, v[255]),  # v0 = lane_id * 4
      v_add_nc_u32_e32(v[0], 100, v[0]),   # v0 = 100 + lane_id * 4
      self._readlane(s[0], v[0], 3),       # s0 = lane 3's v0 = 112
      v_mov_b32_e32(v[1], s[0]),
    ]
    st = run_program(instructions, n_lanes=4)
    for lane in range(4):
      self.assertEqual(st.vgpr[lane][1], 112)

  def test_v_readlane_b32_different_vgpr(self):
    """V_READLANE_B32 reading from different VGPR indices."""
    instructions = [
      v_lshlrev_b32_e32(v[5], 3, v[255]),  # v5 = lane_id * 8
      v_add_nc_u32_e32(v[5], 50, v[5]),    # v5 = 50 + lane_id * 8
      self._readlane(s[0], v[5], 1),       # s0 = lane 1's v5 = 58
      v_mov_b32_e32(v[6], s[0]),
    ]
    st = run_program(instructions, n_lanes=4)
    for lane in range(4):
      self.assertEqual(st.vgpr[lane][6], 58)

  def test_v_writelane_b32_basic(self):
    """V_WRITELANE_B32 writes a scalar to a specific lane's VGPR."""
    instructions = [
      v_mov_b32_e32(v[0], 0),
      s_mov_b32(s[0], 999),
      v_writelane_b32(v[0], s[0], 2),
    ]
    st = run_program(instructions, n_lanes=4)
    for lane in range(4):
      if lane == 2:
        self.assertEqual(st.vgpr[lane][0], 999)
      else:
        self.assertEqual(st.vgpr[lane][0], 0)

  def test_v_writelane_then_readlane(self):
    """V_WRITELANE followed by V_READLANE to verify round-trip."""
    instructions = [
      v_mov_b32_e32(v[0], 0),
      s_mov_b32(s[0], 0xdeadbeef),
      v_writelane_b32(v[0], s[0], 1),      # Write to lane 1
      self._readlane(s[1], v[0], 1),       # Read back from lane 1 into s1
      v_mov_b32_e32(v[1], s[1]),
    ]
    st = run_program(instructions, n_lanes=4)
    for lane in range(4):
      self.assertEqual(st.vgpr[lane][1], 0xdeadbeef)

  def test_v_readlane_for_reduction(self):
    """Simulate a wave reduction using readlane - common WMMA/reduction pattern."""
    instructions = [
      v_add_nc_u32_e32(v[0], 1, v[255]),   # v0 = lane_id + 1 (1, 2, 3, 4)
      self._readlane(s[0], v[0], 0),       # s0 = 1
      self._readlane(s[1], v[0], 1),       # s1 = 2
      s_add_u32(s[0], s[0], s[1]),         # s0 = 3
      self._readlane(s[1], v[0], 2),       # s1 = 3
      s_add_u32(s[0], s[0], s[1]),         # s0 = 6
      self._readlane(s[1], v[0], 3),       # s1 = 4
      s_add_u32(s[0], s[0], s[1]),         # s0 = 10
      v_mov_b32_e32(v[1], s[0]),           # Broadcast sum to all lanes
    ]
    st = run_program(instructions, n_lanes=4)
    for lane in range(4):
      self.assertEqual(st.vgpr[lane][1], 10, "Sum 1+2+3+4 should be 10")

  def test_v_writelane_b32_different_vgpr(self):
    """V_WRITELANE_B32 writes to a non-zero VGPR index.

    Regression test for bug where vdst_idx was always 0 due to function signature
    mismatch (_vars parameter shifted all arguments). This caused all WRITELANE
    operations to write to v[0] regardless of the actual destination register.
    """
    instructions = [
      v_mov_b32_e32(v[0], 0),              # Initialize v0 = 0
      v_mov_b32_e32(v[5], 0),              # Initialize v5 = 0
      s_mov_b32(s[0], 0x12345678),         # Value to write
      v_writelane_b32(v[5], s[0], 1),      # Write to lane 1's v5 (NOT v0!)
    ]
    st = run_program(instructions, n_lanes=4)
    # v[0] should remain 0 for all lanes (bug would have written here)
    for lane in range(4):
      self.assertEqual(st.vgpr[lane][0], 0, f"v[0] lane {lane} should be 0 (untouched)")
    # v[5] should have the value only in lane 1
    for lane in range(4):
      if lane == 1:
        self.assertEqual(st.vgpr[lane][5], 0x12345678, f"v[5] lane 1 should have 0x12345678")
      else:
        self.assertEqual(st.vgpr[lane][5], 0, f"v[5] lane {lane} should be 0")

  def test_v_writelane_b32_high_vgpr_index(self):
    """V_WRITELANE_B32 writes to a high VGPR index (v[15]).

    Tests that the vdst_idx is correctly passed through for larger register indices.
    """
    instructions = [
      v_mov_b32_e32(v[0], 0),              # Initialize v0 = 0
      v_mov_b32_e32(v[15], 0),             # Initialize v15 = 0
      s_mov_b32(s[0], 0xCAFEBABE),         # Value to write
      v_writelane_b32(v[15], s[0], 0),     # Write to lane 0's v15
    ]
    st = run_program(instructions, n_lanes=4)
    # v[0] should remain 0 for all lanes
    for lane in range(4):
      self.assertEqual(st.vgpr[lane][0], 0, f"v[0] lane {lane} should be 0")
    # v[15] should have the value only in lane 0
    self.assertEqual(st.vgpr[0][15], 0xCAFEBABE, "v[15] lane 0 should have 0xCAFEBABE")
    for lane in range(1, 4):
      self.assertEqual(st.vgpr[lane][15], 0, f"v[15] lane {lane} should be 0")

  def test_v_writelane_b32_multiple_writes_different_vgprs(self):
    """V_WRITELANE_B32 writes to multiple different VGPRs.

    This is the pattern used in sparse_categorical_crossentropy where values
    are written to different VGPR indices via writelane, then read back.
    """
    instructions = [
      # Initialize all target VGPRs to 0
      v_mov_b32_e32(v[0], 0),
      v_mov_b32_e32(v[3], 0),
      v_mov_b32_e32(v[7], 0),
      v_mov_b32_e32(v[10], 0),
      # Write different values to different VGPRs at different lanes
      s_mov_b32(s[0], 100),
      v_writelane_b32(v[3], s[0], 0),      # v[3] lane 0 = 100
      s_mov_b32(s[0], 200),
      v_writelane_b32(v[7], s[0], 1),      # v[7] lane 1 = 200
      s_mov_b32(s[0], 300),
      v_writelane_b32(v[10], s[0], 2),     # v[10] lane 2 = 300
    ]
    st = run_program(instructions, n_lanes=4)

    # v[0] should remain 0 everywhere
    for lane in range(4):
      self.assertEqual(st.vgpr[lane][0], 0, f"v[0] lane {lane} should be 0")

    # Check each target VGPR
    self.assertEqual(st.vgpr[0][3], 100, "v[3] lane 0 should be 100")
    for lane in range(1, 4):
      self.assertEqual(st.vgpr[lane][3], 0, f"v[3] lane {lane} should be 0")

    self.assertEqual(st.vgpr[1][7], 200, "v[7] lane 1 should be 200")
    for lane in [0, 2, 3]:
      self.assertEqual(st.vgpr[lane][7], 0, f"v[7] lane {lane} should be 0")

    self.assertEqual(st.vgpr[2][10], 300, "v[10] lane 2 should be 300")
    for lane in [0, 1, 3]:
      self.assertEqual(st.vgpr[lane][10], 0, f"v[10] lane {lane} should be 0")

  def test_v_writelane_then_readlane_different_vgpr(self):
    """V_WRITELANE followed by V_READLANE on a non-zero VGPR.

    Regression test: the original bug caused writelane to always write to v[0],
    so reading back from the intended VGPR would return 0 instead of the written value.
    This is the exact pattern that failed in sparse_categorical_crossentropy.
    """
    instructions = [
      v_mov_b32_e32(v[0], 0),              # Initialize v0 = 0
      v_mov_b32_e32(v[8], 0),              # Initialize v8 = 0
      s_mov_b32(s[0], 0xABCD1234),
      v_writelane_b32(v[8], s[0], 2),      # Write to lane 2's v8
      self._readlane(s[1], v[8], 2),       # Read back from lane 2's v8 into s1
      v_mov_b32_e32(v[1], s[1]),           # Broadcast to all lanes
    ]
    st = run_program(instructions, n_lanes=4)
    # The read value should be what we wrote
    for lane in range(4):
      self.assertEqual(st.vgpr[lane][1], 0xABCD1234,
                       f"Lane {lane}: readlane should return 0xABCD1234, got 0x{st.vgpr[lane][1]:08x}")
    # v[0] should still be 0 (bug would have written here instead of v[8])
    for lane in range(4):
      self.assertEqual(st.vgpr[lane][0], 0, f"v[0] lane {lane} should be 0 (untouched)")

  def test_v_writelane_b32_accumulate_pattern(self):
    """V_WRITELANE_B32 used to accumulate values across lanes into a single VGPR.

    This pattern is used in reductions where each lane writes its result to
    a different lane of the same VGPR, then the results are read back.
    """
    instructions = [
      v_mov_b32_e32(v[6], 0),              # Initialize accumulator v6 = 0
      # Each "iteration" writes to a different lane
      s_mov_b32(s[0], 10),
      v_writelane_b32(v[6], s[0], 0),      # lane 0 gets 10
      s_mov_b32(s[0], 20),
      v_writelane_b32(v[6], s[0], 1),      # lane 1 gets 20
      s_mov_b32(s[0], 30),
      v_writelane_b32(v[6], s[0], 2),      # lane 2 gets 30
      s_mov_b32(s[0], 40),
      v_writelane_b32(v[6], s[0], 3),      # lane 3 gets 40
      # Now read them all back and sum
      self._readlane(s[0], v[6], 0),       # s0 = 10
      self._readlane(s[1], v[6], 1),       # s1 = 20
      s_add_u32(s[0], s[0], s[1]),         # s0 = 30
      self._readlane(s[1], v[6], 2),       # s1 = 30
      s_add_u32(s[0], s[0], s[1]),         # s0 = 60
      self._readlane(s[1], v[6], 3),       # s1 = 40
      s_add_u32(s[0], s[0], s[1]),         # s0 = 100
      v_mov_b32_e32(v[7], s[0]),           # Broadcast sum to all lanes
    ]
    st = run_program(instructions, n_lanes=4)

    # Check that each lane of v[6] has the correct value
    self.assertEqual(st.vgpr[0][6], 10, "v[6] lane 0 should be 10")
    self.assertEqual(st.vgpr[1][6], 20, "v[6] lane 1 should be 20")
    self.assertEqual(st.vgpr[2][6], 30, "v[6] lane 2 should be 30")
    self.assertEqual(st.vgpr[3][6], 40, "v[6] lane 3 should be 40")

    # Check the sum
    for lane in range(4):
      self.assertEqual(st.vgpr[lane][7], 100, f"Sum should be 100, got {st.vgpr[lane][7]}")


class TestF16Modifiers(unittest.TestCase):
  """Tests for F16 operations with abs/neg modifiers and inline constants."""

  def test_v_fma_f16_inline_const_1_0(self):
    """V_FMA_F16: a*b + 1.0 should use f16 inline constant."""
    f16_a = f32_to_f16(0.325928)  # ~0x3537
    f16_b = f32_to_f16(-0.486572)  # ~0xb7c9
    instructions = [
      s_mov_b32(s[0], f16_a),
      v_mov_b32_e32(v[4], s[0]),
      s_mov_b32(s[1], f16_b),
      v_mov_b32_e32(v[6], s[1]),
      v_fma_f16(v[4], v[4], v[6], 1.0),  # 1.0 is inline constant
    ]
    st = run_program(instructions, n_lanes=1)
    result = f16(st.vgpr[0][4] & 0xffff)
    expected = 0.325928 * (-0.486572) + 1.0
    self.assertAlmostEqual(result, expected, delta=0.01)

  def test_v_fma_f16_inline_const_0_5(self):
    """V_FMA_F16: a*b + 0.5 should use f16 inline constant."""
    f16_a = f32_to_f16(2.0)
    f16_b = f32_to_f16(3.0)
    instructions = [
      s_mov_b32(s[0], f16_a),
      v_mov_b32_e32(v[0], s[0]),
      s_mov_b32(s[1], f16_b),
      v_mov_b32_e32(v[1], s[1]),
      v_fma_f16(v[2], v[0], v[1], 0.5),  # 0.5 is inline constant
    ]
    st = run_program(instructions, n_lanes=1)
    result = f16(st.vgpr[0][2] & 0xffff)
    expected = 2.0 * 3.0 + 0.5
    self.assertAlmostEqual(result, expected, delta=0.01)

  def test_v_fma_f16_inline_const_neg_1_0(self):
    """V_FMA_F16: a*b + (-1.0) should use f16 inline constant."""
    f16_a = f32_to_f16(2.0)
    f16_b = f32_to_f16(3.0)
    instructions = [
      s_mov_b32(s[0], f16_a),
      v_mov_b32_e32(v[0], s[0]),
      s_mov_b32(s[1], f16_b),
      v_mov_b32_e32(v[1], s[1]),
      v_fma_f16(v[2], v[0], v[1], -1.0),  # -1.0 is inline constant
    ]
    st = run_program(instructions, n_lanes=1)
    result = f16(st.vgpr[0][2] & 0xffff)
    expected = 2.0 * 3.0 + (-1.0)
    self.assertAlmostEqual(result, expected, delta=0.01)

  def test_v_add_f16_abs_both(self):
    """V_ADD_F16 with abs on both operands."""
    f16_neg2 = f32_to_f16(-2.0)
    f16_neg3 = f32_to_f16(-3.0)
    instructions = [
      s_mov_b32(s[0], f16_neg2),
      v_mov_b32_e32(v[0], s[0]),
      s_mov_b32(s[1], f16_neg3),
      v_mov_b32_e32(v[1], s[1]),
      v_add_f16_e64(v[2], abs(v[0]), abs(v[1])),  # |-2| + |-3| = 5
    ]
    st = run_program(instructions, n_lanes=1)
    result = f16(st.vgpr[0][2] & 0xffff)
    self.assertAlmostEqual(result, 5.0, delta=0.01)

  def test_v_mul_f16_neg_abs(self):
    """V_MUL_F16 with neg on one operand and abs on another."""
    f16_2 = f32_to_f16(2.0)
    f16_neg3 = f32_to_f16(-3.0)
    instructions = [
      s_mov_b32(s[0], f16_2),
      v_mov_b32_e32(v[0], s[0]),
      s_mov_b32(s[1], f16_neg3),
      v_mov_b32_e32(v[1], s[1]),
      v_mul_f16_e64(v[2], -v[0], abs(v[1])),  # -(2) * |-3| = -6
    ]
    st = run_program(instructions, n_lanes=1)
    result = f16(st.vgpr[0][2] & 0xffff)
    self.assertAlmostEqual(result, -6.0, delta=0.01)

  def test_v_fmac_f16_hi_dest(self):
    """v_fmac_f16 with .h destination: dst.h = src0 * src1 + dst.h.

    This tests the case from AMD_LLVM sin(0) where V_FMAC_F16 writes to v0.h.
    """
    instructions = [
      s_mov_b32(s[0], 0x38003c00),  # v0 = {hi=0.5, lo=1.0}
      v_mov_b32_e32(v[0], s[0]),
      s_mov_b32(s[1], 0x38000000),  # v1 = {hi=0.5, lo=0.0}
      v_mov_b32_e32(v[1], s[1]),
      # v_fmac_f16 v0.h, literal(0.318...), v1.l: D.h = D.h + S0 * S1 = 0.5 + 0.318 * 0.0 = 0.5
      v_fmac_f16_e32(v[0].h, 0x3518, v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    v0 = st.vgpr[0][0]
    result_hi = f16((v0 >> 16) & 0xffff)
    result_lo = f16(v0 & 0xffff)
    self.assertAlmostEqual(result_hi, 0.5, delta=0.01, msg=f"Expected hi=0.5, got {result_hi}")
    self.assertAlmostEqual(result_lo, 1.0, delta=0.01, msg=f"Expected lo=1.0, got {result_lo}")


class TestF16FmaMix(unittest.TestCase):
  """Tests for V_FMA_MIX_F32/F16."""

  def test_v_fma_mix_f32_all_f32(self):
    """V_FMA_MIX_F32 with all f32 sources."""
    instructions = [
      s_mov_b32(s[0], f2i(2.0)),
      v_mov_b32_e32(v[0], s[0]),
      s_mov_b32(s[1], f2i(3.0)),
      v_mov_b32_e32(v[1], s[1]),
      s_mov_b32(s[2], f2i(1.0)),
      v_mov_b32_e32(v[2], s[2]),
      VOP3P(VOP3POp.V_FMA_MIX_F32, vdst=v[3], src0=v[0], src1=v[1], src2=v[2], opsel=0, opsel_hi=0, opsel_hi2=0),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i2f(st.vgpr[0][3])
    self.assertAlmostEqual(result, 7.0, places=5)


class TestF64Ops(unittest.TestCase):
  """Tests for 64-bit float operations."""

  def test_v_add_f64_inline_constant(self):
    """V_ADD_F64 with inline constant POS_ONE (1.0) as f64."""
    one_f64 = f2i64(1.0)
    instructions = [
      s_mov_b32(s[0], one_f64 & 0xffffffff),
      s_mov_b32(s[1], one_f64 >> 32),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_add_f64(v[2:3], v[0:1], SrcEnum.POS_ONE),  # 1.0 + 1.0 = 2.0
    ]
    st = run_program(instructions, n_lanes=1)
    result = i642f(st.vgpr[0][2] | (st.vgpr[0][3] << 32))
    self.assertAlmostEqual(result, 2.0, places=5)

  def test_v_mul_f64_basic(self):
    """V_MUL_F64: 2.0 * 3.0 = 6.0."""
    two_f64 = f2i64(2.0)
    three_f64 = f2i64(3.0)
    instructions = [
      s_mov_b32(s[0], two_f64 & 0xffffffff),
      s_mov_b32(s[1], two_f64 >> 32),
      s_mov_b32(s[2], three_f64 & 0xffffffff),
      s_mov_b32(s[3], three_f64 >> 32),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_mov_b32_e32(v[2], s[2]),
      v_mov_b32_e32(v[3], s[3]),
      v_mul_f64(v[4:5], v[0:1], v[2:3]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i642f(st.vgpr[0][4] | (st.vgpr[0][5] << 32))
    self.assertAlmostEqual(result, 6.0, places=10)

  def test_v_cvt_i32_f64_writes_32bit_only(self):
    """V_CVT_I32_F64 should only write 32 bits, not clobber vdst+1."""
    val_bits = f2i64(-1.0)
    instructions = [
      s_mov_b32(s[0], val_bits & 0xffffffff),
      s_mov_b32(s[1], val_bits >> 32),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      s_mov_b32(s[2], 0xDEADBEEF),
      v_mov_b32_e32(v[3], s[2]),     # Canary in v3
      v_cvt_i32_f64_e32(v[2], v[0:1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][2], 0xffffffff, "-1.0 converts to -1")
    self.assertEqual(st.vgpr[0][3], 0xDEADBEEF, "v3 canary should not be clobbered")

  def test_v_ldexp_f64_negative_exponent(self):
    """V_LDEXP_F64 with negative exponent (-32)."""
    val = -8.0
    val_bits = f2i64(val)
    expected = -8.0 * (2.0 ** -32)
    instructions = [
      s_mov_b32(s[0], val_bits & 0xffffffff),
      s_mov_b32(s[1], val_bits >> 32),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_ldexp_f64(v[2:3], v[0:1], 0xffffffe0),  # -32
    ]
    st = run_program(instructions, n_lanes=1)
    result = i642f(st.vgpr[0][2] | (st.vgpr[0][3] << 32))
    self.assertAlmostEqual(result, expected, places=15)

  def test_v_frexp_mant_f64_range(self):
    """V_FREXP_MANT_F64 should return mantissa in [0.5, 1.0) range."""
    two_f64 = f2i64(2.0)
    instructions = [
      s_mov_b32(s[0], two_f64 & 0xffffffff),
      s_mov_b32(s[1], two_f64 >> 32),
      v_frexp_mant_f64_e32(v[0:1], s[0:1]),
      v_frexp_exp_i32_f64_e32(v[2], s[0:1]),
    ]
    st = run_program(instructions, n_lanes=1)
    mant = i642f(st.vgpr[0][0] | (st.vgpr[0][1] << 32))
    exp = st.vgpr[0][2]
    if exp >= 0x80000000: exp -= 0x100000000  # sign extend
    self.assertAlmostEqual(mant, 0.5, places=10)
    self.assertEqual(exp, 2)

  def test_v_div_scale_f64_reads_64bit_sources(self):
    """V_DIV_SCALE_F64 must read all sources as 64-bit values."""
    import math
    sqrt2_f64 = f2i64(1.4142135623730951)
    one_f64 = f2i64(1.0)
    instructions = [
      s_mov_b32(s[0], sqrt2_f64 & 0xffffffff),
      s_mov_b32(s[1], sqrt2_f64 >> 32),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      s_mov_b32(s[2], one_f64 & 0xffffffff),
      s_mov_b32(s[3], one_f64 >> 32),
      v_mov_b32_e32(v[2], s[2]),
      v_mov_b32_e32(v[3], s[3]),
      VOP3SD(VOP3SDOp.V_DIV_SCALE_F64, vdst=v[4:5], sdst=s[10], src0=v[0:1], src1=v[0:1], src2=v[2:3]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i642f(st.vgpr[0][4] | (st.vgpr[0][5] << 32))
    self.assertFalse(math.isnan(result), "Result should not be NaN")
    self.assertAlmostEqual(result, 1.4142135623730951, places=10)

  def test_f64_to_i64_conversion_sequence(self):
    """Full f64->i64 conversion sequence with negative value."""
    import struct
    val = f2i64(-8.0)
    lit = 0xC1F00000  # high 32 bits of f64 -2^32
    instructions = [
      s_mov_b32(s[0], val & 0xffffffff),
      s_mov_b32(s[1], (val >> 32) & 0xffffffff),
      v_trunc_f64_e32(v[0:1], s[0:1]),
      v_ldexp_f64(v[2:3], v[0:1], 0xffffffe0),  # -32
      v_floor_f64_e32(v[2:3], v[2:3]),
      s_mov_b32(s[2], f2i64(-4294967296.0) & 0xffffffff),
      s_mov_b32(s[3], f2i64(-4294967296.0) >> 32),
      v_fma_f64(v[0:1], s[2:3], v[2:3], v[0:1]),
      v_cvt_u32_f64_e32(v[4], v[0:1]),
      v_cvt_i32_f64_e32(v[5], v[2:3]),
    ]
    st = run_program(instructions, n_lanes=1)
    lo = st.vgpr[0][4]
    hi = st.vgpr[0][5]
    result = struct.unpack('<q', struct.pack('<II', lo, hi))[0]
    self.assertEqual(result, -8)

  def test_v_trig_preop_f64_index0(self):
    """V_TRIG_PREOP_F64 index=0: primary chunk of 2/PI."""
    import math
    two_over_pi = 2.0 / math.pi
    instructions = [
      s_mov_b32(s[0], 0x00000000),  # low bits of 1.0
      s_mov_b32(s[1], 0x3ff00000),  # high bits of 1.0
      v_trig_preop_f64(v[0:1], abs(s[0:1]), 0),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i642f(st.vgpr[0][0] | (st.vgpr[0][1] << 32))
    self.assertAlmostEqual(result, two_over_pi, places=10)

  def test_v_trig_preop_f64_sum_equals_two_over_pi(self):
    """V_TRIG_PREOP_F64: sum of chunks 0,1,2 should equal 2/PI."""
    import math
    two_over_pi = 2.0 / math.pi
    instructions = [
      s_mov_b32(s[0], 0x00000000),  # low bits of 1.0
      s_mov_b32(s[1], 0x3ff00000),  # high bits of 1.0
      v_trig_preop_f64(v[0:1], abs(s[0:1]), 0),
      v_trig_preop_f64(v[2:3], abs(s[0:1]), 1),
      v_trig_preop_f64(v[4:5], abs(s[0:1]), 2),
    ]
    st = run_program(instructions, n_lanes=1)
    p0 = i642f(st.vgpr[0][0] | (st.vgpr[0][1] << 32))
    p1 = i642f(st.vgpr[0][2] | (st.vgpr[0][3] << 32))
    p2 = i642f(st.vgpr[0][4] | (st.vgpr[0][5] << 32))
    total = p0 + p1 + p2
    self.assertAlmostEqual(total, two_over_pi, places=14)

  def test_v_fma_f64_sin_kernel_step84(self):
    """V_FMA_F64: exact values from sin(2.0) kernel step 84 that shows 1-bit difference."""
    # From test_sin_f64 failure trace at step 84:
    # v_fma_f64 v[7:8], v[17:18], v[7:8], v[15:16]
    # We need to capture the exact input values and verify output matches hardware
    # v[7:8] before = 0x3f80fdf3_d69db28f (0.008296875941334462)
    v78 = 0x3f80fdf3d69db28f
    # For the FMA to produce 0xbf457ef0_ab8c254d, we need v[17:18] and v[15:16]
    # Let's test with known precision-sensitive values
    a = 1.0000000001
    b = 1.0000000002
    c = -1.0000000003
    a_bits, b_bits, c_bits = f2i64(a), f2i64(b), f2i64(c)
    instructions = [
      s_mov_b32(s[0], a_bits & 0xffffffff),
      s_mov_b32(s[1], a_bits >> 32),
      s_mov_b32(s[2], b_bits & 0xffffffff),
      s_mov_b32(s[3], b_bits >> 32),
      s_mov_b32(s[4], c_bits & 0xffffffff),
      s_mov_b32(s[5], c_bits >> 32),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_mov_b32_e32(v[2], s[2]),
      v_mov_b32_e32(v[3], s[3]),
      v_mov_b32_e32(v[4], s[4]),
      v_mov_b32_e32(v[5], s[5]),
      v_fma_f64(v[6:7], v[0:1], v[2:3], v[4:5]),
    ]
    # run_program with USE_HW=1 will verify exact bit match with hardware
    st = run_program(instructions, n_lanes=1)
    result_bits = st.vgpr[0][6] | (st.vgpr[0][7] << 32)
    self.assertNotEqual(result_bits, 0, "Result should not be zero")


class TestMad64More(unittest.TestCase):
  """More tests for V_MAD_U64_U32."""

  def test_v_mad_u64_u32_with_add(self):
    """V_MAD_U64_U32 with 64-bit addend."""
    instructions = [
      s_mov_b32(s[0], 1000),
      s_mov_b32(s[1], 1000),
      v_mov_b32_e32(v[2], 0),  # S2 lo
      v_mov_b32_e32(v[3], 1),  # S2 hi = 0x100000000
      v_mad_u64_u32(v[4:5], SrcEnum.NULL, s[0], s[1], v[2:3]),
    ]
    st = run_program(instructions, n_lanes=1)
    result_lo = st.vgpr[0][4]
    result_hi = st.vgpr[0][5]
    result = result_lo | (result_hi << 32)
    expected = 1000 * 1000 + 0x100000000
    self.assertEqual(result, expected)

  def test_v_mad_u64_u32_max_values(self):
    """V_MAD_U64_U32 with max u32 values."""
    instructions = [
      s_mov_b32(s[0], 0xFFFFFFFF),
      s_mov_b32(s[1], 0xFFFFFFFF),
      v_mov_b32_e32(v[2], 0),
      v_mov_b32_e32(v[3], 0),
      v_mad_u64_u32(v[4:5], SrcEnum.NULL, s[0], s[1], v[2:3]),
    ]
    st = run_program(instructions, n_lanes=1)
    result_lo = st.vgpr[0][4]
    result_hi = st.vgpr[0][5]
    result = result_lo | (result_hi << 32)
    expected = 0xFFFFFFFF * 0xFFFFFFFF
    self.assertEqual(result, expected)


class TestPermMore(unittest.TestCase):
  """More tests for V_PERM_B32."""

  def test_v_perm_b32_select_high_bytes(self):
    """V_PERM_B32: Select bytes from high word (s0)."""
    instructions = [
      s_mov_b32(s[0], 0x03020100),
      s_mov_b32(s[1], 0x07060504),
      s_mov_b32(s[2], 0x04050607),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_mov_b32_e32(v[2], s[2]),
      v_perm_b32(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][3]
    self.assertEqual(result, 0x00010203)

  def test_v_perm_b32_constant_values(self):
    """V_PERM_B32: Test constant 0x00 (sel=12) and 0xFF (sel>=13)."""
    instructions = [
      s_mov_b32(s[0], 0x12345678),
      s_mov_b32(s[1], 0xABCDEF01),
      s_mov_b32(s[2], 0x0C0D0E0F),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_mov_b32_e32(v[2], s[2]),
      v_perm_b32(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][3]
    self.assertEqual(result, 0x00FFFFFF)

  def test_v_perm_b32_sign_extend(self):
    """V_PERM_B32: Test sign extension selectors 8-11."""
    instructions = [
      s_mov_b32(s[0], 0x00008000),
      s_mov_b32(s[1], 0x80000080),
      s_mov_b32(s[2], 0x08090A0B),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_mov_b32_e32(v[2], s[2]),
      v_perm_b32(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][3]
    self.assertEqual(result, 0x00FFFF00)


class TestF64LiteralOps(unittest.TestCase):
  """Tests for 64-bit operations with literal encoding."""

  def test_v_fma_f64_literal_neg_2pow32(self):
    """V_FMA_F64 with literal encoding of -2^32."""
    val_41 = f2i64(-41.0)
    val_m1 = f2i64(-1.0)
    lit = 0xC1F00000  # high 32 bits of f64 -2^32
    instructions = [
      s_mov_b32(s[0], val_41 & 0xffffffff),
      s_mov_b32(s[1], (val_41 >> 32) & 0xffffffff),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      s_mov_b32(s[2], val_m1 & 0xffffffff),
      s_mov_b32(s[3], (val_m1 >> 32) & 0xffffffff),
      v_mov_b32_e32(v[2], s[2]),
      v_mov_b32_e32(v[3], s[3]),
      VOP3(VOP3Op.V_FMA_F64, vdst=v[4:5], src0=lit, src1=v[2:3], src2=v[0:1]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i642f(st.vgpr[0][4] | (st.vgpr[0][5] << 32))
    expected = 4294967255.0  # 2^32 - 41
    self.assertAlmostEqual(result, expected, places=0)

  def test_v_ldexp_f64_literal_neg32(self):
    """V_LDEXP_F64 with literal -32 for exponent."""
    val = f2i64(-41.0)
    expected = -41.0 * (2.0 ** -32)
    instructions = [
      s_mov_b32(s[0], val & 0xffffffff),
      s_mov_b32(s[1], (val >> 32) & 0xffffffff),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_ldexp_f64(v[2:3], v[0:1], 0xFFFFFFE0),  # -32
    ]
    st = run_program(instructions, n_lanes=1)
    result = i642f(st.vgpr[0][2] | (st.vgpr[0][3] << 32))
    self.assertAlmostEqual(result, expected, places=15)


class TestF64ToI64Conversion(unittest.TestCase):
  """Tests for f64 to i64 conversion sequence."""

  def _convert_f64_to_i64(self, val_f64):
    """Helper to create f64->i64 conversion sequence."""
    val = f2i64(val_f64)
    lit = 0xC1F00000
    instructions = [
      s_mov_b32(s[0], val & 0xffffffff),
      s_mov_b32(s[1], (val >> 32) & 0xffffffff),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_trunc_f64_e32(v[0:1], v[0:1]),
      v_ldexp_f64(v[2:3], v[0:1], 0xFFFFFFE0),
      v_floor_f64_e32(v[2:3], v[2:3]),
      VOP3(VOP3Op.V_FMA_F64, vdst=v[0:1], src0=lit, src1=v[2:3], src2=v[0:1]),
      v_cvt_u32_f64_e32(v[4], v[0:1]),
      v_cvt_i32_f64_e32(v[5], v[2:3]),
    ]
    return instructions

  def test_f64_to_i64_full_sequence(self):
    """Full f64->i64 conversion sequence with negative value."""
    import struct
    instructions = self._convert_f64_to_i64(-41.0)
    st = run_program(instructions, n_lanes=1)
    lo = st.vgpr[0][4]
    hi = st.vgpr[0][5]
    result = struct.unpack('<q', struct.pack('<II', lo, hi))[0]
    self.assertEqual(result, -41)

  def test_f64_to_i64_large_negative(self):
    """f64->i64 conversion with larger negative value (-1000000)."""
    import struct
    instructions = self._convert_f64_to_i64(-1000000.0)
    st = run_program(instructions, n_lanes=1)
    lo = st.vgpr[0][4]
    hi = st.vgpr[0][5]
    result = struct.unpack('<q', struct.pack('<II', lo, hi))[0]
    self.assertEqual(result, -1000000)

  def test_f64_to_i64_positive(self):
    """f64->i64 conversion with positive value (1000000)."""
    import struct
    instructions = self._convert_f64_to_i64(1000000.0)
    st = run_program(instructions, n_lanes=1)
    lo = st.vgpr[0][4]
    hi = st.vgpr[0][5]
    result = struct.unpack('<q', struct.pack('<II', lo, hi))[0]
    self.assertEqual(result, 1000000)

  def test_f64_to_i64_large_positive(self):
    """f64->i64 conversion with value > 2^32."""
    import struct
    instructions = self._convert_f64_to_i64(5000000000.0)
    st = run_program(instructions, n_lanes=1)
    lo = st.vgpr[0][4]
    hi = st.vgpr[0][5]
    result = struct.unpack('<q', struct.pack('<II', lo, hi))[0]
    self.assertEqual(result, 5000000000)


class TestB64VOPLiteral(unittest.TestCase):
  """Tests for B64 VOP operations with literal encoding.

  B64 operations (like V_LSHLREV_B64) should zero-extend the literal to 64 bits,
  NOT put it in the high 32 bits like F64 operations do.
  """

  def test_v_lshlrev_b64_literal_shift_amount(self):
    """V_LSHLREV_B64 with literal shift amount (src0 is 32-bit)."""
    # Shift 1 left by 100 (0x64) - uses literal encoding for src0
    # Shift amount is 100 & 63 = 36, so 1 << 36 = 0x1000000000
    instructions = [
      s_mov_b32(s[0], 1),
      s_mov_b32(s[1], 0),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_lshlrev_b64(v[2:3], 100, v[0:1]),  # 100 > 64, uses literal encoding
    ]
    st = run_program(instructions, n_lanes=1)
    # lo = 0x00000000, hi = 0x00000010 = 1 << (36-32)
    self.assertEqual(st.vgpr[0][2], 0x00000000)
    self.assertEqual(st.vgpr[0][3], 0x00000010)

  def test_v_lshlrev_b64_literal_value(self):
    """V_LSHLREV_B64 with literal as the 64-bit value being shifted (src1).

    B64 literals are zero-extended (not shifted to high bits like F64).
    0xDEADBEEF << 4 = 0xDEADBEEF0 = lo=0xEADBEEF0, hi=0x0000000D
    """
    instructions = [
      v_lshlrev_b64(v[0:1], 4, 0xDEADBEEF),  # shift literal left by 4
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][0], 0xEADBEEF0)  # lo
    self.assertEqual(st.vgpr[0][1], 0x0000000D)  # hi


class TestWMMAMore(unittest.TestCase):
  """More WMMA tests."""

  def test_v_wmma_f32_16x16x16_f16_basic(self):
    """V_WMMA_F32_16X16X16_F16 basic test - verify output is non-zero."""
    instructions = []
    instructions.append(s_mov_b32(s[0], 0x3c003c00))
    for i in range(16, 32):
      instructions.append(v_mov_b32_e32(v[i], s[0]))
    for i in range(8):
      instructions.append(v_mov_b32_e32(v[i], 0))
    instructions.append(v_wmma_f32_16x16x16_f16(v[0:7], v[16:23], v[24:31], v[0:7]))
    st = run_program(instructions, n_lanes=32)
    any_nonzero = any(st.vgpr[lane][0] != 0 for lane in range(32))
    self.assertTrue(any_nonzero, "WMMA should produce non-zero output")


class TestSinReduction(unittest.TestCase):
  """Tests for sin argument reduction steps."""

  def test_sin_reduction_step1_mul(self):
    """First step: v1 = |x| * (1/2pi)."""
    import math
    one_over_2pi = 1.0 / (2.0 * math.pi)
    x = 100000.0
    instructions = [
      s_mov_b32(s[0], f2i(x)),
      s_mov_b32(s[1], f2i(one_over_2pi)),
      v_mov_b32_e32(v[0], s[0]),
      v_mul_f32_e32(v[1], s[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i2f(st.vgpr[0][1])
    expected = x * one_over_2pi
    self.assertAlmostEqual(result, expected, places=0)

  def test_sin_reduction_step2_round(self):
    """Second step: round to nearest integer."""
    import math
    one_over_2pi = 1.0 / (2.0 * math.pi)
    x = 100000.0
    val = x * one_over_2pi  # ~15915.49
    instructions = [
      s_mov_b32(s[0], f2i(val)),
      v_mov_b32_e32(v[0], s[0]),
      v_rndne_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i2f(st.vgpr[0][1])
    expected = round(val)
    self.assertAlmostEqual(result, expected, places=0)

  def test_sin_reduction_step3_fma(self):
    """Third step: x - n * (pi/2) via FMA."""
    import math
    neg_half_pi = -math.pi / 2.0
    x = 100000.0
    n = 15915.0
    instructions = [
      s_mov_b32(s[0], f2i(neg_half_pi)),
      s_mov_b32(s[1], f2i(n)),
      s_mov_b32(s[2], f2i(x)),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_mov_b32_e32(v[2], s[2]),
      v_fma_f32(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i2f(st.vgpr[0][3])
    expected = x + neg_half_pi * n
    self.assertAlmostEqual(result, expected, places=2)

  def test_sin_1e5_full_reduction(self):
    """Full reduction sequence for sin(1e5)."""
    import math
    x = 100000.0
    one_over_2pi = 1.0 / (2.0 * math.pi)
    neg_half_pi = -math.pi / 2.0

    instructions = [
      s_mov_b32(s[0], f2i(x)),
      s_mov_b32(s[1], f2i(one_over_2pi)),
      s_mov_b32(s[2], f2i(neg_half_pi)),
      v_mov_b32_e32(v[0], s[0]),
      v_mul_f32_e32(v[1], s[1], v[0]),
      v_rndne_f32_e32(v[2], v[1]),
      v_fma_f32(v[3], s[2], v[2], v[0]),
      v_cvt_i32_f32_e32(v[4], v[2]),
      v_and_b32_e32(v[5], 3, v[4]),
    ]
    st = run_program(instructions, n_lanes=1)

    mul_result = i2f(st.vgpr[0][1])
    round_result = i2f(st.vgpr[0][2])
    quadrant = st.vgpr[0][5]

    expected_mul = x * one_over_2pi
    expected_round = round(expected_mul)
    expected_quadrant = int(expected_round) & 3

    self.assertAlmostEqual(mul_result, expected_mul, places=0)
    self.assertAlmostEqual(round_result, expected_round, places=0)
    self.assertEqual(quadrant, expected_quadrant)


class TestTrigPreop(unittest.TestCase):
  """Tests for V_TRIG_PREOP_F64 - chunks of 2/PI for argument reduction."""

  def test_trig_preop_f64_index0(self):
    """V_TRIG_PREOP_F64 index=0: primary chunk of 2/PI."""
    import math
    two_over_pi = 2.0 / math.pi
    instructions = [
      s_mov_b32(s[0], 0x00000000),  # low bits of 1.0
      s_mov_b32(s[1], 0x3ff00000),  # high bits of 1.0
      v_trig_preop_f64(v[0:1], abs(s[0:1]), 0),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i642f(st.vgpr[0][0] | (st.vgpr[0][1] << 32))
    self.assertAlmostEqual(result, two_over_pi, places=10)

  def test_trig_preop_f64_index1(self):
    """V_TRIG_PREOP_F64 index=1: secondary chunk (extended precision bits)."""
    instructions = [
      s_mov_b32(s[0], 0x00000000),
      s_mov_b32(s[1], 0x3ff00000),
      v_trig_preop_f64(v[0:1], abs(s[0:1]), 1),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i642f(st.vgpr[0][0] | (st.vgpr[0][1] << 32))
    self.assertLess(abs(result), 1e-15)
    self.assertGreater(abs(result), 0)

  def test_trig_preop_f64_index2(self):
    """V_TRIG_PREOP_F64 index=2: tertiary chunk (more extended precision bits)."""
    instructions = [
      s_mov_b32(s[0], 0x00000000),
      s_mov_b32(s[1], 0x3ff00000),
      v_trig_preop_f64(v[0:1], abs(s[0:1]), 2),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i642f(st.vgpr[0][0] | (st.vgpr[0][1] << 32))
    self.assertLess(abs(result), 1e-30)

  def test_trig_preop_f64_sum_equals_two_over_pi(self):
    """V_TRIG_PREOP_F64: sum of chunks 0,1,2 should equal 2/PI."""
    import math
    two_over_pi = 2.0 / math.pi
    instructions = [
      s_mov_b32(s[0], 0x00000000),
      s_mov_b32(s[1], 0x3ff00000),
      v_trig_preop_f64(v[0:1], abs(s[0:1]), 0),
      v_trig_preop_f64(v[2:3], abs(s[0:1]), 1),
      v_trig_preop_f64(v[4:5], abs(s[0:1]), 2),
    ]
    st = run_program(instructions, n_lanes=1)
    p0 = i642f(st.vgpr[0][0] | (st.vgpr[0][1] << 32))
    p1 = i642f(st.vgpr[0][2] | (st.vgpr[0][3] << 32))
    p2 = i642f(st.vgpr[0][4] | (st.vgpr[0][5] << 32))
    total = p0 + p1 + p2
    self.assertAlmostEqual(total, two_over_pi, places=14)

  def test_trig_preop_f64_large_input(self):
    """V_TRIG_PREOP_F64 with larger input should adjust shift based on exponent."""
    import math
    large_val = 2.0 ** 60
    large_bits = f2i64(large_val)
    instructions = [
      s_mov_b32(s[0], large_bits & 0xffffffff),
      s_mov_b32(s[1], (large_bits >> 32) & 0xffffffff),
      v_trig_preop_f64(v[0:1], abs(s[0:1]), 0),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i642f(st.vgpr[0][0] | (st.vgpr[0][1] << 32))
    self.assertFalse(math.isnan(result))
    self.assertFalse(math.isinf(result))


class TestModifierInteractions(unittest.TestCase):
  """Tests for abs/neg/clamp/omod modifier interactions."""

  def test_neg_abs_combination(self):
    """-|x| should negate the absolute value."""
    instructions = [
      v_mov_b32_e32(v[0], -5.0),
      VOP3(VOP3Op.V_MUL_F32, vdst=v[1], src0=1.0, src1=v[0], neg=0b10, abs=0b10),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), -5.0, places=5)

  def test_abs_neg_on_neg_zero(self):
    """|(-0.0)| = +0.0, -|(-0.0)| = -0.0."""
    neg_zero = 0x80000000
    instructions = [
      s_mov_b32(s[0], neg_zero),
      v_mov_b32_e32(v[0], s[0]),
      VOP3(VOP3Op.V_MUL_F32, vdst=v[1], src0=1.0, src1=v[0], abs=0b10),
      VOP3(VOP3Op.V_MUL_F32, vdst=v[2], src0=1.0, src1=v[0], neg=0b10, abs=0b10),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 0x00000000, "|(-0.0)| = +0.0")
    self.assertEqual(st.vgpr[0][2], 0x80000000, "-|(-0.0)| = -0.0")

  def test_clamp_with_nan(self):
    """Clamp with NaN input should still produce NaN."""
    import math
    quiet_nan = 0x7fc00000
    instructions = [
      s_mov_b32(s[0], quiet_nan),
      v_mov_b32_e32(v[0], s[0]),
      VOP3(VOP3Op.V_ADD_F32, vdst=v[1], src0=v[0], src1=0.0, clamp=1),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertTrue(math.isnan(i2f(st.vgpr[0][1])))

  def test_omod_ignored(self):
    """OMOD field is ignored on RDNA3 hardware."""
    instructions = [
      v_mov_b32_e32(v[0], 3.0),
      VOP3(VOP3Op.V_ADD_F32, vdst=v[1], src0=v[0], src1=1.0, omod=1),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), 4.0, places=5)

  def test_nan_propagation(self):
    """NaN should propagate through FMA operations."""
    import math
    quiet_nan = 0x7fc00000
    instructions = [
      s_mov_b32(s[0], quiet_nan),
      v_mov_b32_e32(v[0], s[0]),
      v_fma_f32(v[1], v[0], 1.0, 0.0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertTrue(math.isnan(i2f(st.vgpr[0][1])), "fma(NaN, 1, 0) = NaN")


class TestBitfieldEdges(unittest.TestCase):
  """Tests for bitfield operation edge cases."""

  def test_bfe_u32_max_width(self):
    """V_BFE_U32 extracting max 31 bits (width field is 5 bits)."""
    instructions = [
      s_mov_b32(s[0], 0xDEADBEEF),
      v_mov_b32_e32(v[0], s[0]),
      v_bfe_u32(v[1], v[0], 0, 31),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 0x5EADBEEF)

  def test_bfe_u32_zero_width(self):
    """V_BFE_U32 with zero width should return 0."""
    instructions = [
      s_mov_b32(s[0], 0xFFFFFFFF),
      v_mov_b32_e32(v[0], s[0]),
      v_bfe_u32(v[1], v[0], 16, 0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 0)

  def test_bfe_i32_sign_extend(self):
    """V_BFE_I32 should sign extend."""
    instructions = [
      s_mov_b32(s[0], 0x000000F0),
      v_mov_b32_e32(v[0], s[0]),
      v_bfe_i32(v[1], v[0], 4, 4),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 0xFFFFFFFF)

  def test_bfi_b32_basic(self):
    """V_BFI_B32 bit field insert."""
    instructions = [
      s_mov_b32(s[0], 0x0000FFFF),
      s_mov_b32(s[1], 0xAAAAAAAA),
      s_mov_b32(s[2], 0x55555555),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_mov_b32_e32(v[2], s[2]),
      v_bfi_b32(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][3], 0x5555AAAA)


class TestCarryBorrow(unittest.TestCase):
  """Tests for carry/borrow operations (VOP3SD)."""

  def test_add_co_u32_no_carry(self):
    """V_ADD_CO_U32 without carry."""
    instructions = [
      v_mov_b32_e32(v[0], 100),
      v_mov_b32_e32(v[1], 50),
      v_add_co_u32(v[2], VCC, v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][2], 150)
    self.assertEqual(st.vcc & 1, 0, "No carry")

  def test_add_co_u32_with_carry(self):
    """V_ADD_CO_U32 with carry."""
    instructions = [
      s_mov_b32(s[0], 0xFFFFFFFF),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], 2),
      v_add_co_u32(v[2], VCC, v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][2], 1)
    self.assertEqual(st.vcc & 1, 1, "Should have carry")

  def test_sub_co_u32_no_borrow(self):
    """V_SUB_CO_U32 without borrow."""
    instructions = [
      v_mov_b32_e32(v[0], 100),
      v_mov_b32_e32(v[1], 50),
      v_sub_co_u32(v[2], VCC, v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][2], 50)
    self.assertEqual(st.vcc & 1, 0, "No borrow")

  def test_sub_co_u32_with_borrow(self):
    """V_SUB_CO_U32 with borrow."""
    instructions = [
      v_mov_b32_e32(v[0], 50),
      v_mov_b32_e32(v[1], 100),
      v_sub_co_u32(v[2], VCC, v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][2], 0xFFFFFFCE)
    self.assertEqual(st.vcc & 1, 1, "Should have borrow")

  def test_addc_co_u32_chain(self):
    """V_ADD_CO_CI_U32 chained addition (64-bit add via two 32-bit adds)."""
    instructions = [
      s_mov_b32(s[0], 0xFFFFFFFF),
      s_mov_b32(s[1], 0x00000001),
      s_mov_b32(s[2], 0x00000001),
      s_mov_b32(s[3], 0x00000001),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_mov_b32_e32(v[2], s[2]),
      v_mov_b32_e32(v[3], s[3]),
      v_add_co_u32(v[4], VCC, v[0], v[2]),
      v_add_co_ci_u32_e32(v[5], v[1], v[3]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][4], 0x00000000, "lo result")
    self.assertEqual(st.vgpr[0][5], 0x00000003, "hi result")

  def test_add_co_u32_same_dst_src(self):
    """V_ADD_CO_U32 where dst is same as src - VCC must use original src value."""
    instructions = [
      s_mov_b32(s[0], 0xFFFFFFFF),
      v_mov_b32_e32(v[0], s[0]),
      v_add_co_u32(v[0], VCC, v[0], 1),  # v[0] = v[0] + 1, VCC should be set from overflow
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][0], 0, "0xFFFFFFFF + 1 = 0")
    self.assertEqual(st.vcc & 1, 1, "Should have carry from 0xFFFFFFFF + 1")

  def test_add_co_u32_same_dst_src_no_carry(self):
    """V_ADD_CO_U32 where dst is same as src - no carry case."""
    instructions = [
      v_mov_b32_e32(v[0], 100),
      v_add_co_u32(v[0], VCC, v[0], 1),  # v[0] = v[0] + 1
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][0], 101, "100 + 1 = 101")
    self.assertEqual(st.vcc & 1, 0, "No carry from 100 + 1")


class TestReadlane(unittest.TestCase):
  """Tests for V_READLANE_B32 and related cross-lane operations."""

  def test_lane_id_distinct(self):
    """Each lane should have distinct lane_id in v255."""
    instructions = [
      v_mov_b32_e32(v[0], v[255]),
    ]
    st = run_program(instructions, n_lanes=32)
    for lane in range(32):
      self.assertEqual(st.vgpr[lane][0], lane)

  def test_reduction_pattern(self):
    """Test reduction using readlane."""
    instructions = [
      v_mov_b32_e32(v[0], v[255]),
      v_readlane_b32(s[0], v[0], 0),
      v_readlane_b32(s[1], v[0], 1),
      v_readlane_b32(s[2], v[0], 2),
      v_readlane_b32(s[3], v[0], 3),
      s_add_u32(s[4], s[0], s[1]),
      s_add_u32(s[4], s[4], s[2]),
      s_add_u32(s[4], s[4], s[3]),
    ]
    st = run_program(instructions, n_lanes=4)
    self.assertEqual(st.sgpr[4], 6)


class TestMed3(unittest.TestCase):
  """Tests for V_MED3 - median of 3 values."""

  def test_v_med3_f32_basic(self):
    """V_MED3_F32: median of 1.0, 2.0, 3.0 is 2.0."""
    instructions = [
      v_mov_b32_e32(v[0], 1.0),
      v_mov_b32_e32(v[1], 2.0),
      v_mov_b32_e32(v[2], 3.0),
      v_med3_f32(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][3]), 2.0, places=5)

  def test_v_med3_f32_reversed(self):
    """V_MED3_F32: median of 3.0, 2.0, 1.0 is still 2.0."""
    instructions = [
      v_mov_b32_e32(v[0], 3.0),
      v_mov_b32_e32(v[1], 2.0),
      v_mov_b32_e32(v[2], 1.0),
      v_med3_f32(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][3]), 2.0, places=5)

  def test_v_med3_f32_two_equal(self):
    """V_MED3_F32: median of 1.0, 3.0, 3.0 is 3.0."""
    instructions = [
      v_mov_b32_e32(v[0], 1.0),
      v_mov_b32_e32(v[1], 3.0),
      v_mov_b32_e32(v[2], 3.0),
      v_med3_f32(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][3]), 3.0, places=5)

  def test_v_med3_f32_all_equal(self):
    """V_MED3_F32: median of 5.0, 5.0, 5.0 is 5.0."""
    instructions = [
      v_mov_b32_e32(v[0], 5.0),
      v_mov_b32_e32(v[1], 5.0),
      v_mov_b32_e32(v[2], 5.0),
      v_med3_f32(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][3]), 5.0, places=5)

  def test_v_med3_f32_negative(self):
    """V_MED3_F32: median of -1.0, 0.0, 1.0 is 0.0."""
    instructions = [
      v_mov_b32_e32(v[0], -1.0),
      v_mov_b32_e32(v[1], 0.0),
      v_mov_b32_e32(v[2], 1.0),
      v_med3_f32(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][3]), 0.0, places=5)

  def test_v_med3_f32_with_nan(self):
    """V_MED3_F32: NaN handling - returns min of non-NaN values."""
    import math
    instructions = [
      s_mov_b32(s[0], 0x7fc00000),  # NaN
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], 1.0),
      v_mov_b32_e32(v[2], 2.0),
      v_med3_f32(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i2f(st.vgpr[0][3])
    # With one NaN, result should be min of non-NaN values
    self.assertAlmostEqual(result, 1.0, places=5)

  def test_v_med3_i32_basic(self):
    """V_MED3_I32: median of signed integers."""
    instructions = [
      s_mov_b32(s[0], (-5) & 0xFFFFFFFF),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], 0),
      v_mov_b32_e32(v[2], 10),
      v_med3_i32(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][3], 0)

  def test_v_med3_i32_all_negative(self):
    """V_MED3_I32: median of -10, -5, -1 is -5."""
    instructions = [
      s_mov_b32(s[0], (-10) & 0xFFFFFFFF),
      s_mov_b32(s[1], (-5) & 0xFFFFFFFF),
      s_mov_b32(s[2], (-1) & 0xFFFFFFFF),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_mov_b32_e32(v[2], s[2]),
      v_med3_i32(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][3], (-5) & 0xFFFFFFFF)

  def test_v_med3_u32_basic(self):
    """V_MED3_U32: median of unsigned integers."""
    instructions = [
      v_mov_b32_e32(v[0], 100),
      v_mov_b32_e32(v[1], 200),
      v_mov_b32_e32(v[2], 150),
      v_med3_u32(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][3], 150)

  def test_v_med3_u32_large(self):
    """V_MED3_U32: median with large unsigned values."""
    instructions = [
      s_mov_b32(s[0], 0xFFFFFFFF),
      s_mov_b32(s[1], 0x80000000),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_mov_b32_e32(v[2], 0),
      v_med3_u32(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][3], 0x80000000)


class TestMinMax(unittest.TestCase):
  """Tests for V_MIN/V_MAX with edge cases including NaN."""

  def test_v_min_f32_basic(self):
    """V_MIN_F32: min of 1.0 and 2.0 is 1.0."""
    instructions = [
      v_mov_b32_e32(v[0], 1.0),
      v_mov_b32_e32(v[1], 2.0),
      v_min_f32_e32(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][2]), 1.0, places=5)

  def test_v_max_f32_basic(self):
    """V_MAX_F32: max of 1.0 and 2.0 is 2.0."""
    instructions = [
      v_mov_b32_e32(v[0], 1.0),
      v_mov_b32_e32(v[1], 2.0),
      v_max_f32_e32(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][2]), 2.0, places=5)

  def test_v_min_f32_with_nan_first(self):
    """V_MIN_F32: min(NaN, 1.0) returns 1.0 (IEEE 754-2008)."""
    instructions = [
      s_mov_b32(s[0], 0x7fc00000),  # NaN
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], 1.0),
      v_min_f32_e32(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][2]), 1.0, places=5)

  def test_v_min_f32_with_nan_second(self):
    """V_MIN_F32: min(1.0, NaN) returns 1.0."""
    instructions = [
      s_mov_b32(s[0], 0x7fc00000),  # NaN
      v_mov_b32_e32(v[0], 1.0),
      v_mov_b32_e32(v[1], s[0]),
      v_min_f32_e32(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][2]), 1.0, places=5)

  def test_v_max_f32_with_nan(self):
    """V_MAX_F32: max(NaN, 1.0) returns 1.0."""
    instructions = [
      s_mov_b32(s[0], 0x7fc00000),  # NaN
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], 1.0),
      v_max_f32_e32(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][2]), 1.0, places=5)

  def test_v_min_f32_neg_zero(self):
    """V_MIN_F32: min(+0, -0) should return -0."""
    instructions = [
      v_mov_b32_e32(v[0], 0),          # +0
      s_mov_b32(s[0], 0x80000000),     # -0
      v_mov_b32_e32(v[1], s[0]),
      v_min_f32_e32(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    # -0 < +0 according to IEEE 754 totalOrder
    self.assertEqual(st.vgpr[0][2], 0x80000000)

  def test_v_max_f32_neg_zero(self):
    """V_MAX_F32: max(+0, -0) should return +0."""
    instructions = [
      v_mov_b32_e32(v[0], 0),          # +0
      s_mov_b32(s[0], 0x80000000),     # -0
      v_mov_b32_e32(v[1], s[0]),
      v_max_f32_e32(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][2], 0)

  def test_v_min_i32_signed(self):
    """V_MIN_I32: handles signed comparison correctly."""
    instructions = [
      s_mov_b32(s[0], (-5) & 0xFFFFFFFF),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], 5),
      v_min_i32_e32(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][2], (-5) & 0xFFFFFFFF)

  def test_v_max_u32_large(self):
    """V_MAX_U32: handles large unsigned values."""
    instructions = [
      s_mov_b32(s[0], 0xFFFFFFFF),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], 100),
      v_max_u32_e32(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][2], 0xFFFFFFFF)


class TestCeil(unittest.TestCase):
  """Tests for V_CEIL_F32."""

  def test_v_ceil_f32_positive_frac(self):
    """V_CEIL_F32: ceil(2.3) = 3.0."""
    instructions = [
      s_mov_b32(s[0], f2i(2.3)),
      v_mov_b32_e32(v[0], s[0]),
      v_ceil_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), 3.0, places=5)

  def test_v_ceil_f32_negative_frac(self):
    """V_CEIL_F32: ceil(-2.3) = -2.0."""
    instructions = [
      s_mov_b32(s[0], f2i(-2.3)),
      v_mov_b32_e32(v[0], s[0]),
      v_ceil_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), -2.0, places=5)

  def test_v_ceil_f32_whole(self):
    """V_CEIL_F32: ceil(5.0) = 5.0."""
    instructions = [
      v_mov_b32_e32(v[0], 5.0),
      v_ceil_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), 5.0, places=5)

  def test_v_ceil_f32_zero(self):
    """V_CEIL_F32: ceil(0.0) = 0.0."""
    instructions = [
      v_mov_b32_e32(v[0], 0),
      v_ceil_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(i2f(st.vgpr[0][1]), 0.0)

  def test_v_ceil_f32_neg_zero(self):
    """V_CEIL_F32: ceil(-0.0) = -0.0."""
    instructions = [
      s_mov_b32(s[0], 0x80000000),
      v_mov_b32_e32(v[0], s[0]),
      v_ceil_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 0x80000000)

  def test_v_ceil_f32_small_positive(self):
    """V_CEIL_F32: ceil(0.1) = 1.0."""
    instructions = [
      s_mov_b32(s[0], f2i(0.1)),
      v_mov_b32_e32(v[0], s[0]),
      v_ceil_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), 1.0, places=5)

  def test_v_ceil_f32_small_negative(self):
    """V_CEIL_F32: ceil(-0.1) = -0.0."""
    instructions = [
      s_mov_b32(s[0], f2i(-0.1)),
      v_mov_b32_e32(v[0], s[0]),
      v_ceil_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i2f(st.vgpr[0][1])
    self.assertEqual(result, 0.0)


class TestAlignBit(unittest.TestCase):
  """Tests for V_ALIGNBIT_B32 and V_ALIGNBYTE_B32."""

  def test_v_alignbit_b32_zero_shift(self):
    """V_ALIGNBIT_B32: shift by 0 returns src1."""
    instructions = [
      s_mov_b32(s[0], 0x12345678),
      s_mov_b32(s[1], 0xAABBCCDD),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_mov_b32_e32(v[2], 0),
      v_alignbit_b32(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][3], 0xAABBCCDD)

  def test_v_alignbit_b32_shift_8(self):
    """V_ALIGNBIT_B32: shift by 8 bits."""
    instructions = [
      s_mov_b32(s[0], 0x12345678),
      s_mov_b32(s[1], 0xAABBCCDD),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_mov_b32_e32(v[2], 8),
      v_alignbit_b32(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    # (0x12345678 << 24) | (0xAABBCCDD >> 8) = 0x78AABBCC
    self.assertEqual(st.vgpr[0][3], 0x78AABBCC)

  def test_v_alignbit_b32_shift_16(self):
    """V_ALIGNBIT_B32: shift by 16 bits."""
    instructions = [
      s_mov_b32(s[0], 0x12345678),
      s_mov_b32(s[1], 0xAABBCCDD),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_mov_b32_e32(v[2], 16),
      v_alignbit_b32(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    # (0x12345678 << 16) | (0xAABBCCDD >> 16) = 0x5678AABB
    self.assertEqual(st.vgpr[0][3], 0x5678AABB)

  def test_v_alignbit_b32_shift_32(self):
    """V_ALIGNBIT_B32: shift by 32 returns src0."""
    instructions = [
      s_mov_b32(s[0], 0x12345678),
      s_mov_b32(s[1], 0xAABBCCDD),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_mov_b32_e32(v[2], 32),
      v_alignbit_b32(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    # Hardware only uses low 5 bits of shift, so shift 32 = shift 0
    self.assertEqual(st.vgpr[0][3], 0xAABBCCDD)

  def test_v_alignbyte_b32_shift_1(self):
    """V_ALIGNBYTE_B32: shift by 1 byte."""
    instructions = [
      s_mov_b32(s[0], 0x12345678),
      s_mov_b32(s[1], 0xAABBCCDD),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_mov_b32_e32(v[2], 1),
      v_alignbyte_b32(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    # (0x12345678 << 24) | (0xAABBCCDD >> 8) = 0x78AABBCC
    self.assertEqual(st.vgpr[0][3], 0x78AABBCC)

  def test_v_alignbyte_b32_shift_3(self):
    """V_ALIGNBYTE_B32: shift by 3 bytes."""
    instructions = [
      s_mov_b32(s[0], 0x12345678),
      s_mov_b32(s[1], 0xAABBCCDD),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_mov_b32_e32(v[2], 3),
      v_alignbyte_b32(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    # (0x12345678 << 8) | (0xAABBCCDD >> 24) = 0x345678AA
    self.assertEqual(st.vgpr[0][3], 0x345678AA)


class TestShiftEdgeCases(unittest.TestCase):
  """Tests for shift operations with edge cases."""

  def test_v_lshlrev_b32_by_0(self):
    """V_LSHLREV_B32: shift by 0 returns original."""
    instructions = [
      s_mov_b32(s[0], 0x12345678),
      v_mov_b32_e32(v[0], s[0]),
      v_lshlrev_b32_e32(v[1], 0, v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 0x12345678)

  def test_v_lshlrev_b32_by_31(self):
    """V_LSHLREV_B32: shift by 31 bits."""
    instructions = [
      v_mov_b32_e32(v[0], 1),
      v_lshlrev_b32_e32(v[1], 31, v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 0x80000000)

  def test_v_lshlrev_b32_by_32(self):
    """V_LSHLREV_B32: shift by 32 - only low 5 bits used."""
    instructions = [
      v_mov_b32_e32(v[0], 1),
      v_lshlrev_b32_e32(v[1], 32, v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    # 32 & 0x1f = 0, so no shift
    self.assertEqual(st.vgpr[0][1], 1)

  def test_v_lshrrev_b32_by_32(self):
    """V_LSHRREV_B32: shift by 32 - only low 5 bits used."""
    instructions = [
      s_mov_b32(s[0], 0x80000000),
      v_mov_b32_e32(v[0], s[0]),
      v_lshrrev_b32_e32(v[1], 32, v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    # 32 & 0x1f = 0, so no shift
    self.assertEqual(st.vgpr[0][1], 0x80000000)

  def test_v_ashrrev_i32_negative(self):
    """V_ASHRREV_I32: arithmetic shift preserves sign."""
    instructions = [
      s_mov_b32(s[0], 0x80000000),  # -2147483648
      v_mov_b32_e32(v[0], s[0]),
      v_ashrrev_i32_e32(v[1], 4, v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    # Arithmetic right shift fills with sign bit
    self.assertEqual(st.vgpr[0][1], 0xF8000000)

  def test_v_ashrrev_i32_by_31(self):
    """V_ASHRREV_I32: shift by 31 gives all 1s for negative."""
    instructions = [
      s_mov_b32(s[0], 0x80000000),
      v_mov_b32_e32(v[0], s[0]),
      v_ashrrev_i32_e32(v[1], 31, v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 0xFFFFFFFF)

  def test_v_lshrrev_b32_by_31(self):
    """V_LSHRREV_B32: logical shift by 31 gives 0 or 1."""
    instructions = [
      s_mov_b32(s[0], 0x80000000),
      v_mov_b32_e32(v[0], s[0]),
      v_lshrrev_b32_e32(v[1], 31, v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 1)


class TestMulHiLo(unittest.TestCase):
  """Tests for V_MUL_HI/V_MUL_LO operations."""

  def test_v_mul_lo_u32_basic(self):
    """V_MUL_LO_U32: low 32 bits of 32x32 multiply."""
    instructions = [
      v_mov_b32_e32(v[0], 100),
      v_mov_b32_e32(v[1], 200),
      v_mul_lo_u32(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][2], 20000)

  def test_v_mul_lo_u32_overflow(self):
    """V_MUL_LO_U32: result wraps on overflow."""
    instructions = [
      s_mov_b32(s[0], 0x10000),
      s_mov_b32(s[1], 0x10000),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_mul_lo_u32(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    # 0x10000 * 0x10000 = 0x100000000, low 32 bits = 0
    self.assertEqual(st.vgpr[0][2], 0)

  def test_v_mul_hi_u32_basic(self):
    """V_MUL_HI_U32: high 32 bits of 32x32 multiply."""
    instructions = [
      s_mov_b32(s[0], 0x10000),
      s_mov_b32(s[1], 0x10000),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_mul_hi_u32(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    # 0x10000 * 0x10000 = 0x100000000, high 32 bits = 1
    self.assertEqual(st.vgpr[0][2], 1)

  def test_v_mul_hi_u32_large(self):
    """V_MUL_HI_U32: large values."""
    instructions = [
      s_mov_b32(s[0], 0xFFFFFFFF),
      s_mov_b32(s[1], 0xFFFFFFFF),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_mul_hi_u32(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    # 0xFFFFFFFF * 0xFFFFFFFF = 0xFFFFFFFE00000001, high = 0xFFFFFFFE
    self.assertEqual(st.vgpr[0][2], 0xFFFFFFFE)

  def test_v_mul_hi_i32_positive(self):
    """V_MUL_HI_I32: signed multiply with positive values."""
    instructions = [
      s_mov_b32(s[0], 0x10000),
      s_mov_b32(s[1], 0x10000),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_mul_hi_i32(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][2], 1)

  def test_v_mul_hi_i32_negative(self):
    """V_MUL_HI_I32: signed multiply with negative value."""
    instructions = [
      s_mov_b32(s[0], (-10000) & 0xFFFFFFFF),
      s_mov_b32(s[1], 100000),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_mul_hi_i32(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    # -10000 * 100000 = -1000000000, which fits in 32 bits
    # high 32 bits should be -1 (0xFFFFFFFF) for negative numbers that fit
    self.assertEqual(st.vgpr[0][2], 0xFFFFFFFF)

  def test_v_mul_hi_i32_both_negative(self):
    """V_MUL_HI_I32: both values negative."""
    instructions = [
      s_mov_b32(s[0], (-0x10000) & 0xFFFFFFFF),
      s_mov_b32(s[1], (-0x10000) & 0xFFFFFFFF),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_mul_hi_i32(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    # -0x10000 * -0x10000 = 0x100000000, high = 1
    self.assertEqual(st.vgpr[0][2], 1)


class TestMulF32EdgeCases(unittest.TestCase):
  """Edge cases for V_MUL_F32."""

  def test_v_mul_f32_inf_by_zero(self):
    """V_MUL_F32: inf * 0 = NaN."""
    import math
    instructions = [
      s_mov_b32(s[0], 0x7f800000),  # +inf
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], 0),
      v_mul_f32_e32(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertTrue(math.isnan(i2f(st.vgpr[0][2])))

  def test_v_mul_f32_inf_by_inf(self):
    """V_MUL_F32: inf * inf = inf."""
    import math
    instructions = [
      s_mov_b32(s[0], 0x7f800000),  # +inf
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[0]),
      v_mul_f32_e32(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertTrue(math.isinf(i2f(st.vgpr[0][2])))

  def test_v_mul_f32_neg_zero_by_pos(self):
    """V_MUL_F32: -0 * positive = -0."""
    instructions = [
      s_mov_b32(s[0], 0x80000000),  # -0.0
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], 1.0),
      v_mul_f32_e32(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][2], 0x80000000)

  def test_v_mul_f32_neg_zero_by_neg(self):
    """V_MUL_F32: -0 * negative = +0."""
    instructions = [
      s_mov_b32(s[0], 0x80000000),  # -0.0
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], -1.0),
      v_mul_f32_e32(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][2], 0)  # +0


class TestAddF32EdgeCases(unittest.TestCase):
  """Edge cases for V_ADD_F32."""

  def test_v_add_f32_inf_minus_inf(self):
    """V_ADD_F32: inf + (-inf) = NaN."""
    import math
    instructions = [
      s_mov_b32(s[0], 0x7f800000),  # +inf
      s_mov_b32(s[1], 0xff800000),  # -inf
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_add_f32_e32(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertTrue(math.isnan(i2f(st.vgpr[0][2])))

  def test_v_add_f32_pos_neg_zero(self):
    """V_ADD_F32: +0 + (-0) = +0."""
    instructions = [
      v_mov_b32_e32(v[0], 0),
      s_mov_b32(s[0], 0x80000000),  # -0.0
      v_mov_b32_e32(v[1], s[0]),
      v_add_f32_e32(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][2], 0)  # +0

  def test_v_add_f32_neg_neg_zero(self):
    """V_ADD_F32: -0 + (-0) = -0."""
    instructions = [
      s_mov_b32(s[0], 0x80000000),  # -0.0
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[0]),
      v_add_f32_e32(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][2], 0x80000000)  # -0


class TestDivScaleF64(unittest.TestCase):
  """Tests for V_DIV_SCALE_F64 - critical for tan() and division.

  These tests verify that VCC bits are set independently per lane,
  which is essential for correct multi-lane f64 division operations.
  """

  def test_div_scale_f64_basic_no_scaling(self):
    """V_DIV_SCALE_F64: normal values with no scaling needed."""
    sqrt2 = f2i64(1.4142135623730951)
    one = f2i64(1.0)
    instructions = [
      s_mov_b32(s[0], sqrt2 & 0xffffffff),
      s_mov_b32(s[1], sqrt2 >> 32),
      s_mov_b32(s[2], one & 0xffffffff),
      s_mov_b32(s[3], one >> 32),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_mov_b32_e32(v[2], s[2]),
      v_mov_b32_e32(v[3], s[3]),
      VOP3SD(VOP3SDOp.V_DIV_SCALE_F64, vdst=v[4:5], sdst=VCC, src0=v[0:1], src1=v[0:1], src2=v[2:3]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i642f(st.vgpr[0][4] | (st.vgpr[0][5] << 32))
    self.assertAlmostEqual(result, 1.4142135623730951, places=10)
    self.assertEqual(st.vcc & 1, 0, "VCC should be 0 when no scaling needed")

  def test_div_scale_f64_vcc_per_lane_uniform_input(self):
    """V_DIV_SCALE_F64: VCC bits should be set independently per lane (uniform input).

    This is a regression test for the bug where VCC = 0x0LL was setting the whole
    64-bit VCC register instead of just the current lane's bit. With uniform input
    all lanes should get VCC=0.
    """
    val = f2i64(2.0)
    instructions = [
      s_mov_b32(s[0], val & 0xffffffff),
      s_mov_b32(s[1], val >> 32),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      VOP3SD(VOP3SDOp.V_DIV_SCALE_F64, vdst=v[2:3], sdst=VCC, src0=v[0:1], src1=v[0:1], src2=v[0:1]),
    ]
    st = run_program(instructions, n_lanes=4)
    # All lanes should have VCC=0 for normal values
    self.assertEqual(st.vcc & 0xf, 0, "All lanes should have VCC=0 for normal values")
    # All lanes should have same result
    for lane in range(4):
      result = i642f(st.vgpr[lane][2] | (st.vgpr[lane][3] << 32))
      self.assertAlmostEqual(result, 2.0, places=10, msg=f"Lane {lane} result mismatch")

  def test_div_scale_f64_vcc_per_lane_varying_input(self):
    """V_DIV_SCALE_F64: VCC bits set per-lane with different inputs per lane.

    This test uses different inputs per lane to verify that VCC is tracked
    independently. This catches the bug where the emulator was setting VCC
    for all lanes to the same value.
    """
    import math
    # Use lane-varying input: lane 0 gets 2.0, lane 1 gets 3.0, etc.
    # All normal values should result in VCC=0 for each lane
    instructions = [
      # Set up per-lane values using lane_id
      v_cvt_f64_i32_e32(v[0:1], v[255]),  # v0:1 = f64(lane_id)
      v_add_f64(v[0:1], v[0:1], SrcEnum.POS_TWO),  # v0:1 = lane_id + 2.0
      VOP3SD(VOP3SDOp.V_DIV_SCALE_F64, vdst=v[2:3], sdst=VCC, src0=v[0:1], src1=v[0:1], src2=v[0:1]),
    ]
    st = run_program(instructions, n_lanes=4)
    # All lanes should have VCC=0 (no scaling needed for 2.0, 3.0, 4.0, 5.0)
    self.assertEqual(st.vcc & 0xf, 0, "All lanes should have VCC=0 for normal values")
    # Verify each lane has correct result
    for lane in range(4):
      expected = float(lane) + 2.0
      result = i642f(st.vgpr[lane][2] | (st.vgpr[lane][3] << 32))
      self.assertAlmostEqual(result, expected, places=10, msg=f"Lane {lane}: expected {expected}, got {result}")

  def test_div_scale_f64_zero_denom_sets_vcc(self):
    """V_DIV_SCALE_F64: zero denominator -> NaN, VCC=1."""
    import math
    one = f2i64(1.0)
    zero = f2i64(0.0)
    instructions = [
      s_mov_b32(s[0], one & 0xffffffff),
      s_mov_b32(s[1], one >> 32),
      s_mov_b32(s[2], zero & 0xffffffff),
      s_mov_b32(s[3], zero >> 32),
      v_mov_b32_e32(v[0], s[0]),  # numer = 1.0
      v_mov_b32_e32(v[1], s[1]),
      v_mov_b32_e32(v[2], s[2]),  # denom = 0.0
      v_mov_b32_e32(v[3], s[3]),
      VOP3SD(VOP3SDOp.V_DIV_SCALE_F64, vdst=v[4:5], sdst=VCC, src0=v[0:1], src1=v[2:3], src2=v[0:1]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i642f(st.vgpr[0][4] | (st.vgpr[0][5] << 32))
    self.assertTrue(math.isnan(result), "Should be NaN for zero denom")
    self.assertEqual(st.vcc & 1, 1, "VCC should be 1 for zero denom")

  def test_div_scale_f64_mixed_vcc_per_lane(self):
    """V_DIV_SCALE_F64: some lanes need scaling, others don't.

    This is the key test for the tan() bug - it verifies that VCC is set
    correctly for each lane independently when some lanes need scaling and
    others don't.
    """
    import math
    # Lane 0: normal value (VCC=0), Lane 1: zero denom (VCC=1)
    # Lane 2: normal value (VCC=0), Lane 3: zero denom (VCC=1)
    normal = f2i64(2.0)
    zero = f2i64(0.0)
    instructions = [
      # Set up numer = 2.0 for all lanes
      s_mov_b32(s[0], normal & 0xffffffff),
      s_mov_b32(s[1], normal >> 32),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      # Set up denom: lane 0,2 get 2.0, lane 1,3 get 0.0
      s_mov_b32(s[2], zero & 0xffffffff),
      s_mov_b32(s[3], zero >> 32),
      v_mov_b32_e32(v[2], s[0]),  # default to 2.0
      v_mov_b32_e32(v[3], s[1]),
      # Override lanes 1 and 3 with 0.0 using writelane
      v_writelane_b32(v[2], s[2], 1),
      v_writelane_b32(v[3], s[3], 1),
      v_writelane_b32(v[2], s[2], 3),
      v_writelane_b32(v[3], s[3], 3),
      VOP3SD(VOP3SDOp.V_DIV_SCALE_F64, vdst=v[4:5], sdst=VCC, src0=v[0:1], src1=v[2:3], src2=v[0:1]),
    ]
    st = run_program(instructions, n_lanes=4)
    # Lanes 0,2 should have VCC=0 (normal), lanes 1,3 should have VCC=1 (zero denom)
    self.assertEqual(st.vcc & 0b0001, 0, "Lane 0 VCC should be 0")
    self.assertEqual(st.vcc & 0b0010, 0b0010, "Lane 1 VCC should be 1")
    self.assertEqual(st.vcc & 0b0100, 0, "Lane 2 VCC should be 0")
    self.assertEqual(st.vcc & 0b1000, 0b1000, "Lane 3 VCC should be 1")

    # Check results
    for lane in [0, 2]:
      result = i642f(st.vgpr[lane][4] | (st.vgpr[lane][5] << 32))
      self.assertAlmostEqual(result, 2.0, places=10, msg=f"Lane {lane} should be 2.0")
    for lane in [1, 3]:
      result = i642f(st.vgpr[lane][4] | (st.vgpr[lane][5] << 32))
      self.assertTrue(math.isnan(result), f"Lane {lane} should be NaN")


class TestDivFmasF64(unittest.TestCase):
  """Tests for V_DIV_FMAS_F64 - scaling FMA for f64 division.

  These tests verify that V_DIV_FMAS applies the correct scaling
  based on VCC per lane, which is essential for correct tan() results.
  """

  def test_div_fmas_f64_no_scale_vcc0(self):
    """V_DIV_FMAS_F64: VCC=0 -> normal FMA, no scaling."""
    a = f2i64(2.0)
    b = f2i64(3.0)
    c = f2i64(1.0)
    instructions = [
      s_mov_b32(VCC_LO, 0),
      s_mov_b32(s[0], a & 0xffffffff),
      s_mov_b32(s[1], a >> 32),
      s_mov_b32(s[2], b & 0xffffffff),
      s_mov_b32(s[3], b >> 32),
      s_mov_b32(s[4], c & 0xffffffff),
      s_mov_b32(s[5], c >> 32),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_mov_b32_e32(v[2], s[2]),
      v_mov_b32_e32(v[3], s[3]),
      v_mov_b32_e32(v[4], s[4]),
      v_mov_b32_e32(v[5], s[5]),
      v_div_fmas_f64(v[6:7], v[0:1], v[2:3], v[4:5]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i642f(st.vgpr[0][6] | (st.vgpr[0][7] << 32))
    expected = 2.0 * 3.0 + 1.0  # = 7.0
    self.assertAlmostEqual(result, expected, places=10)

  def test_div_fmas_f64_scale_up_vcc1_large_s2(self):
    """V_DIV_FMAS_F64: VCC=1 with S2 exponent > 1023 -> scale by 2^+128."""
    a = f2i64(1.0)
    b = f2i64(1.0)
    c = f2i64(2.0)  # exponent = 1024 > 1023, so scale UP
    instructions = [
      s_mov_b32(VCC_LO, 1),
      s_mov_b32(s[0], a & 0xffffffff),
      s_mov_b32(s[1], a >> 32),
      s_mov_b32(s[2], b & 0xffffffff),
      s_mov_b32(s[3], b >> 32),
      s_mov_b32(s[4], c & 0xffffffff),
      s_mov_b32(s[5], c >> 32),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_mov_b32_e32(v[2], s[2]),
      v_mov_b32_e32(v[3], s[3]),
      v_mov_b32_e32(v[4], s[4]),
      v_mov_b32_e32(v[5], s[5]),
      v_div_fmas_f64(v[6:7], v[0:1], v[2:3], v[4:5]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i642f(st.vgpr[0][6] | (st.vgpr[0][7] << 32))
    expected = (1.0 * 1.0 + 2.0) * (2.0 ** 128)  # = 3.0 * 2^128
    self.assertAlmostEqual(result, expected, delta=abs(expected) * 1e-10)

  def test_div_fmas_f64_scale_down_vcc1_small_s2(self):
    """V_DIV_FMAS_F64: VCC=1 with S2 exponent <= 1023 -> scale by 2^-128."""
    a = f2i64(2.0)
    b = f2i64(3.0)
    c = f2i64(1.0)  # exponent = 1023, so scale DOWN
    instructions = [
      s_mov_b32(VCC_LO, 1),
      s_mov_b32(s[0], a & 0xffffffff),
      s_mov_b32(s[1], a >> 32),
      s_mov_b32(s[2], b & 0xffffffff),
      s_mov_b32(s[3], b >> 32),
      s_mov_b32(s[4], c & 0xffffffff),
      s_mov_b32(s[5], c >> 32),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_mov_b32_e32(v[2], s[2]),
      v_mov_b32_e32(v[3], s[3]),
      v_mov_b32_e32(v[4], s[4]),
      v_mov_b32_e32(v[5], s[5]),
      v_div_fmas_f64(v[6:7], v[0:1], v[2:3], v[4:5]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i642f(st.vgpr[0][6] | (st.vgpr[0][7] << 32))
    expected = (2.0 * 3.0 + 1.0) * (2.0 ** -128)  # = 7.0 * 2^-128
    self.assertAlmostEqual(result, expected, delta=abs(expected) * 1e-10)

  def test_div_fmas_f64_per_lane_vcc_varying(self):
    """V_DIV_FMAS_F64: different VCC per lane applies different scaling.

    This is the key test for the tan() bug - verifies that scaling is
    applied per-lane based on VCC bits, not uniformly.
    """
    a = f2i64(1.0)
    b = f2i64(1.0)
    c = f2i64(1.0)  # exponent = 1023, so when VCC=1 it scales DOWN
    instructions = [
      # VCC = 0b0101: lanes 0,2 scale, lanes 1,3 don't
      s_mov_b32(VCC_LO, 0b0101),
      s_mov_b32(s[0], a & 0xffffffff),
      s_mov_b32(s[1], a >> 32),
      s_mov_b32(s[2], b & 0xffffffff),
      s_mov_b32(s[3], b >> 32),
      s_mov_b32(s[4], c & 0xffffffff),
      s_mov_b32(s[5], c >> 32),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_mov_b32_e32(v[2], s[2]),
      v_mov_b32_e32(v[3], s[3]),
      v_mov_b32_e32(v[4], s[4]),
      v_mov_b32_e32(v[5], s[5]),
      v_div_fmas_f64(v[6:7], v[0:1], v[2:3], v[4:5]),
    ]
    st = run_program(instructions, n_lanes=4)

    scaled = (1.0 * 1.0 + 1.0) * (2.0 ** -128)  # = 2.0 * 2^-128
    unscaled = 1.0 * 1.0 + 1.0  # = 2.0

    # Lane 0: VCC=1, scale
    result0 = i642f(st.vgpr[0][6] | (st.vgpr[0][7] << 32))
    self.assertAlmostEqual(result0, scaled, delta=abs(scaled) * 1e-10, msg="Lane 0 should be scaled")

    # Lane 1: VCC=0, no scale
    result1 = i642f(st.vgpr[1][6] | (st.vgpr[1][7] << 32))
    self.assertAlmostEqual(result1, unscaled, places=10, msg="Lane 1 should be unscaled")

    # Lane 2: VCC=1, scale
    result2 = i642f(st.vgpr[2][6] | (st.vgpr[2][7] << 32))
    self.assertAlmostEqual(result2, scaled, delta=abs(scaled) * 1e-10, msg="Lane 2 should be scaled")

    # Lane 3: VCC=0, no scale
    result3 = i642f(st.vgpr[3][6] | (st.vgpr[3][7] << 32))
    self.assertAlmostEqual(result3, unscaled, places=10, msg="Lane 3 should be unscaled")


class TestDivScaleFmasF64Integration(unittest.TestCase):
  """Integration tests for V_DIV_SCALE_F64 + V_DIV_FMAS_F64.

  These tests verify the full division sequence used by tan() works
  correctly with multiple lanes having different values.
  """

  def test_div_scale_then_fmas_multi_lane_tan_pattern(self):
    """Test the pattern used by tan(): DIV_SCALE sets VCC, DIV_FMAS uses it.

    This is the exact bug scenario: tan([2.0, 3.0, 4.0]) was failing because
    VCC from DIV_SCALE was being set incorrectly for all lanes.
    """
    import math
    # Set up values like tan() would: different values per lane
    instructions = [
      # Create per-lane values: 2.0, 3.0, 4.0, 5.0
      v_cvt_f64_i32_e32(v[0:1], v[255]),  # v0:1 = f64(lane_id)
      v_add_f64(v[0:1], v[0:1], SrcEnum.POS_TWO),  # numer = lane_id + 2.0
      # denom = 1.0 for all lanes (uniform)
      v_mov_b32_e32(v[2], f2i64(1.0) & 0xffffffff),
      v_mov_b32_e32(v[3], f2i64(1.0) >> 32),
      # V_DIV_SCALE_F64: sets VCC per lane
      VOP3SD(VOP3SDOp.V_DIV_SCALE_F64, vdst=v[4:5], sdst=VCC, src0=v[0:1], src1=v[2:3], src2=v[0:1]),
      # Copy scaled numer for FMA
      v_mov_b32_e32(v[6], v[4]),
      v_mov_b32_e32(v[7], v[5]),
      # V_DIV_FMAS_F64: uses VCC to apply scaling
      v_div_fmas_f64(v[8:9], v[6:7], v[2:3], v[4:5]),
    ]
    st = run_program(instructions, n_lanes=4)

    # All lanes should have VCC=0 (no scaling needed for normal values)
    self.assertEqual(st.vcc & 0xf, 0, "All lanes should have VCC=0 for normal values")

    # Verify each lane has correct intermediate value
    for lane in range(4):
      expected_numer = float(lane) + 2.0
      # With VCC=0, DIV_FMAS should just do FMA with no scaling
      result = i642f(st.vgpr[lane][8] | (st.vgpr[lane][9] << 32))
      # The FMA result should be: scaled_numer * denom + scaled_numer = 2*scaled_numer
      expected = expected_numer * 1.0 + expected_numer  # Simple FMA for this test setup
      self.assertAlmostEqual(result, expected, places=8,
        msg=f"Lane {lane}: expected {expected}, got {result}")


class TestVOP3VOPC(unittest.TestCase):
  """Tests for VOP3-encoded VOPC instructions (comparisons with scalar dest)."""

  def test_v_cmp_ge_f32_e64_nan(self):
    """V_CMP_GE_F32_E64: |NaN| >= |0.0| should be FALSE (NaN comparisons always false)."""
    from extra.assembly.amd.autogen.rdna3.ins import VOP3_SDST
    instructions = [
      s_mov_b32(s[0], 0xffc00000),  # NaN
      s_mov_b32(s[1], 0x00000000),  # 0.0
      v_mov_b32_e32(v[5], s[0]),
      v_mov_b32_e32(v[3], s[1]),
      VOP3_SDST(VOP3Op.V_CMP_GE_F32, vdst=s[5], src0=v[5], src1=v[3], abs_=3),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[5], 0)  # NaN comparison is always FALSE


class TestMin3Max3Unsigned(unittest.TestCase):
  """Regression tests for V_MIN3/V_MAX3 with unsigned integer types.

  The emulator's _minmax_reduce used UOp.minimum() which implements min(a,b) as
  -max(-a,-b). This is broken for unsigned types because negation (mul by -1)
  doesn't preserve ordering: for uint16, -0 = 0 but -5 = 65531, so
  max(-0, -5) = max(0, 65531) = 65531, and -65531 = 5, giving min(0,5) = 5 (wrong!).

  Fix: use comparison-based min/max for unsigned types: min(a,b) = (a<b)?a:b
  """

  def test_v_min3_u16_with_zero(self):
    """V_MIN3_U16: min3(0, 3, 5) should return 0, not a wrong value."""
    instructions = [
      s_mov_b32(s[0], 0),   # 0
      s_mov_b32(s[1], 3),   # 3
      s_mov_b32(s[2], 5),   # 5
      v_mov_b32_e32(v[0], s[0]),
      v_min3_u16(v[1], v[0], s[1], s[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1] & 0xFFFF, 0)

  def test_v_min3_u16_all_nonzero(self):
    """V_MIN3_U16: min3(2, 5, 3) should return 2."""
    instructions = [
      s_mov_b32(s[0], 2),
      s_mov_b32(s[1], 5),
      s_mov_b32(s[2], 3),
      v_mov_b32_e32(v[0], s[0]),
      v_min3_u16(v[1], v[0], s[1], s[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1] & 0xFFFF, 2)

  def test_v_min3_u32_with_zero(self):
    """V_MIN3_U32: min3(0, 100, 50) should return 0."""
    instructions = [
      s_mov_b32(s[0], 0),
      s_mov_b32(s[1], 100),
      s_mov_b32(s[2], 50),
      v_mov_b32_e32(v[0], s[0]),
      v_min3_u32(v[1], v[0], s[1], s[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 0)

  def test_v_max3_u16_basic(self):
    """V_MAX3_U16: max3(0, 3, 5) should return 5."""
    instructions = [
      s_mov_b32(s[0], 0),
      s_mov_b32(s[1], 3),
      s_mov_b32(s[2], 5),
      v_mov_b32_e32(v[0], s[0]),
      v_max3_u16(v[1], v[0], s[1], s[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1] & 0xFFFF, 5)

  def test_v_min_u16_two_operand(self):
    """V_MIN_U16 (two operand): min(0, 5) should return 0."""
    instructions = [
      s_mov_b32(s[0], 0),
      s_mov_b32(s[1], 5),
      v_mov_b32_e32(v[0], s[0]),
      v_min_u16(v[1], v[0], s[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1] & 0xFFFF, 0)


class TestVOP3Clamp(unittest.TestCase):
  """Tests for VOP3 clamp modifier (clmp=1).

  The clamp modifier restricts float outputs to [0.0, 1.0] range.
  This is used by operations like clip(0, 1) which AMD LLVM compiles to
  v_max_f32_e64 with clmp=1.

  Regression test for: clip(0, 1) bug where emulator ignored clmp field.
  """

  def test_v_max_f32_e64_clamp_positive(self):
    """V_MAX_F32_E64 with clamp: value > 1.0 should be clamped to 1.0."""
    instructions = [
      v_mov_b32_e32(v[0], 2.5),
      VOP3(VOP3Op.V_MAX_F32_E64, vdst=v[1], src0=v[0], src1=v[0], clmp=1),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), 1.0, places=5)

  def test_v_max_f32_e64_clamp_negative(self):
    """V_MAX_F32_E64 with clamp: value < 0.0 should be clamped to 0.0."""
    instructions = [
      v_mov_b32_e32(v[0], -1.5),
      VOP3(VOP3Op.V_MAX_F32_E64, vdst=v[1], src0=v[0], src1=v[0], clmp=1),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), 0.0, places=5)

  def test_v_max_f32_e64_clamp_in_range(self):
    """V_MAX_F32_E64 with clamp: value in [0,1] should pass through."""
    instructions = [
      v_mov_b32_e32(v[0], 0.5),
      VOP3(VOP3Op.V_MAX_F32_E64, vdst=v[1], src0=v[0], src1=v[0], clmp=1),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), 0.5, places=5)

  def test_v_max_f32_e64_no_clamp(self):
    """V_MAX_F32_E64 without clamp: value > 1.0 should pass through."""
    instructions = [
      v_mov_b32_e32(v[0], 2.5),
      VOP3(VOP3Op.V_MAX_F32_E64, vdst=v[1], src0=v[0], src1=v[0], clmp=0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), 2.5, places=5)

  def test_v_min_f32_e64_clamp_negative(self):
    """V_MIN_F32_E64 with clamp: value < 0.0 should be clamped to 0.0."""
    instructions = [
      v_mov_b32_e32(v[0], -2.0),
      VOP3(VOP3Op.V_MIN_F32_E64, vdst=v[1], src0=v[0], src1=v[0], clmp=1),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), 0.0, places=5)

  def test_v_add_f32_e64_clamp(self):
    """V_ADD_F32_E64 with clamp: 0.7 + 0.8 = 1.5 -> 1.0."""
    instructions = [
      v_mov_b32_e32(v[0], 0.7),
      v_mov_b32_e32(v[1], 0.8),
      VOP3(VOP3Op.V_ADD_F32_E64, vdst=v[2], src0=v[0], src1=v[1], clmp=1),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][2]), 1.0, places=5)

  def test_v_mul_f32_e64_clamp_underflow(self):
    """V_MUL_F32_E64 with clamp: 0.5 * -2.0 = -1.0 -> 0.0."""
    instructions = [
      v_mov_b32_e32(v[0], 0.5),
      v_mov_b32_e32(v[1], -2.0),
      VOP3(VOP3Op.V_MUL_F32_E64, vdst=v[2], src0=v[0], src1=v[1], clmp=1),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][2]), 0.0, places=5)

  def test_v_fma_f32_clamp(self):
    """V_FMA_F32 with clamp: 2*2+1 = 5 -> 1.0."""
    instructions = [
      v_mov_b32_e32(v[0], 2.0),
      v_mov_b32_e32(v[1], 2.0),
      v_mov_b32_e32(v[2], 1.0),
      VOP3(VOP3Op.V_FMA_F32, vdst=v[3], src0=v[0], src1=v[1], src2=v[2], clmp=1),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][3]), 1.0, places=5)

  def test_v_max_f32_e64_clamp_multilane(self):
    """V_MAX_F32_E64 with clamp: test multiple lanes with different values."""
    # lane 0: -0.5 -> 0.0
    # lane 1: 0.5 -> 0.5
    # lane 2: 1.5 -> 1.0
    # lane 3: 2.5 -> 1.0
    instructions = [
      # Setup different values per lane using lane_id
      s_mov_b32(s[0], f2i(0.5)),
      v_cvt_f32_i32_e32(v[0], v[255]),  # Convert lane_id to float
      v_mov_b32_e32(v[2], s[0]),        # v2 = 0.5
      v_sub_f32_e32(v[0], v[0], v[2]),  # Subtract 0.5: lane0=-0.5, lane1=0.5, lane2=1.5, lane3=2.5
      VOP3(VOP3Op.V_MAX_F32_E64, vdst=v[1], src0=v[0], src1=v[0], clmp=1),
    ]
    st = run_program(instructions, n_lanes=4)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), 0.0, places=5, msg="lane 0: -0.5 should clamp to 0.0")
    self.assertAlmostEqual(i2f(st.vgpr[1][1]), 0.5, places=5, msg="lane 1: 0.5 should pass through")
    self.assertAlmostEqual(i2f(st.vgpr[2][1]), 1.0, places=5, msg="lane 2: 1.5 should clamp to 1.0")
    self.assertAlmostEqual(i2f(st.vgpr[3][1]), 1.0, places=5, msg="lane 3: 2.5 should clamp to 1.0")


class TestCvtPkF16(unittest.TestCase):
  """Tests for V_CVT_PK_RTZ_F16_F32 - pack two f32 to f16 with round toward zero."""

  def test_cvt_pk_rtz_f16_f32_basic(self):
    """V_CVT_PK_RTZ_F16_F32: basic pack of two f32 values."""
    instructions = [
      v_mov_b32_e32(v[0], 1.0),
      v_mov_b32_e32(v[1], 2.0),
      v_cvt_pk_rtz_f16_f32_e64(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][2]
    lo_f16 = f16(result & 0xffff)
    hi_f16 = f16((result >> 16) & 0xffff)
    self.assertAlmostEqual(lo_f16, 1.0, delta=0.01)
    self.assertAlmostEqual(hi_f16, 2.0, delta=0.01)


class TestCvtPkNorm(unittest.TestCase):
  """Tests for V_CVT_PK_NORM_I16_F32 and V_CVT_PK_NORM_U16_F32."""

  def test_cvt_pk_norm_i16_f32_basic(self):
    """V_CVT_PK_NORM_I16_F32: pack two f32 to normalized i16."""
    instructions = [
      v_mov_b32_e32(v[0], 1.0),
      v_mov_b32_e32(v[1], -1.0),
      v_cvt_pk_norm_i16_f32(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][2]
    lo = result & 0xffff
    hi = (result >> 16) & 0xffff
    self.assertEqual(lo, 32767)
    self.assertEqual(hi, 0x8001)  # -32767, hardware uses symmetric range

  def test_cvt_pk_norm_u16_f32_basic(self):
    """V_CVT_PK_NORM_U16_F32: pack two f32 to normalized u16."""
    instructions = [
      v_mov_b32_e32(v[0], 1.0),
      v_mov_b32_e32(v[1], 0.5),
      v_cvt_pk_norm_u16_f32(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][2]
    lo = result & 0xffff
    hi = (result >> 16) & 0xffff
    self.assertEqual(lo, 65535)
    self.assertAlmostEqual(hi, 32768, delta=1)


class TestCvtPkInt(unittest.TestCase):
  """Tests for V_CVT_PK_I16_I32, V_CVT_PK_U16_U32, V_CVT_PK_I16_F32, V_CVT_PK_U16_F32."""

  def test_cvt_pk_i16_i32_basic(self):
    """V_CVT_PK_I16_I32: pack two i32 to i16."""
    instructions = [
      s_mov_b32(s[0], 100),
      s_mov_b32(s[1], -100 & 0xffffffff),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_cvt_pk_i16_i32(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][2]
    lo = result & 0xffff
    hi = (result >> 16) & 0xffff
    lo_signed = lo if lo < 32768 else lo - 65536
    hi_signed = hi if hi < 32768 else hi - 65536
    self.assertEqual(lo_signed, 100)
    self.assertEqual(hi_signed, -100)

  def test_cvt_pk_u16_u32_basic(self):
    """V_CVT_PK_U16_U32: pack two u32 to u16."""
    instructions = [
      s_mov_b32(s[0], 1000),
      s_mov_b32(s[1], 2000),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_cvt_pk_u16_u32(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][2]
    lo = result & 0xffff
    hi = (result >> 16) & 0xffff
    self.assertEqual(lo, 1000)
    self.assertEqual(hi, 2000)

  def test_cvt_pk_i16_f32_basic(self):
    """V_CVT_PK_I16_F32: convert two f32 to packed i16."""
    instructions = [
      v_mov_b32_e32(v[0], 100.5),
      v_mov_b32_e32(v[1], -50.7),
      v_cvt_pk_i16_f32(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][2]
    lo = result & 0xffff
    hi = (result >> 16) & 0xffff
    lo_signed = lo if lo < 32768 else lo - 65536
    hi_signed = hi if hi < 32768 else hi - 65536
    self.assertEqual(lo_signed, 100)
    self.assertEqual(hi_signed, -50)

  def test_cvt_pk_u16_f32_basic(self):
    """V_CVT_PK_U16_F32: convert two f32 to packed u16."""
    instructions = [
      v_mov_b32_e32(v[0], 100.9),
      v_mov_b32_e32(v[1], 200.1),
      v_cvt_pk_u16_f32(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][2]
    lo = result & 0xffff
    hi = (result >> 16) & 0xffff
    self.assertEqual(lo, 100)
    self.assertEqual(hi, 200)

  def test_cvt_pk_u8_f32_basic(self):
    """V_CVT_PK_U8_F32: convert f32 to u8 and pack at byte position."""
    instructions = [
      v_mov_b32_e32(v[0], 128.5),
      v_mov_b32_e32(v[1], 0),
      v_mov_b32_e32(v[2], 0),
      v_cvt_pk_u8_f32(v[2], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][2]
    byte0 = result & 0xff
    self.assertEqual(byte0, 128)


class TestDotProduct(unittest.TestCase):
  """Tests for dot product instructions V_DOT4_U32_U8, V_DOT8_U32_U4."""

  def test_v_dot4_u32_u8_basic(self):
    """V_DOT4_U32_U8: 4-element dot product of u8 vectors."""
    src0 = 0x04030201  # {4, 3, 2, 1}
    src1 = 0x01010101  # {1, 1, 1, 1}
    instructions = [
      s_mov_b32(s[0], src0),
      s_mov_b32(s[1], src1),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_mov_b32_e32(v[2], 0),
      v_dot4_u32_u8(v[2], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][2]
    self.assertEqual(result, 10)

  def test_v_dot4_u32_u8_with_accumulator(self):
    """V_DOT4_U32_U8 with non-zero accumulator."""
    src0 = 0x02020202  # {2, 2, 2, 2}
    src1 = 0x03030303  # {3, 3, 3, 3}
    instructions = [
      s_mov_b32(s[0], src0),
      s_mov_b32(s[1], src1),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_mov_b32_e32(v[2], 100),
      v_dot4_u32_u8(v[2], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][2]
    self.assertEqual(result, 124)

  def test_v_dot8_u32_u4_basic(self):
    """V_DOT8_U32_U4: 8-element dot product of u4 vectors."""
    # src0 = 8 nibbles: {1,2,3,4,5,6,7,8} packed as 0x87654321
    # src1 = 8 nibbles: {1,1,1,1,1,1,1,1} packed as 0x11111111
    # result = 1+2+3+4+5+6+7+8 = 36
    src0 = 0x87654321
    src1 = 0x11111111
    instructions = [
      s_mov_b32(s[0], src0),
      s_mov_b32(s[1], src1),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_mov_b32_e32(v[2], 0),
      v_dot8_u32_u4(v[2], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][2]
    self.assertEqual(result, 36)


class TestMinMaxF16Vop3(unittest.TestCase):
  """Tests for V_MIN3_F16, V_MAX3_F16, V_MED3_F16, V_MINMAX_F16, V_MAXMIN_F16."""

  def test_v_min3_f16_basic(self):
    """V_MIN3_F16: minimum of three f16 values."""
    instructions = [
      s_mov_b32(s[0], f32_to_f16(3.0)),
      s_mov_b32(s[1], f32_to_f16(1.0)),
      s_mov_b32(s[2], f32_to_f16(2.0)),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_mov_b32_e32(v[2], s[2]),
      v_min3_f16(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = f16(st.vgpr[0][3] & 0xffff)
    self.assertAlmostEqual(result, 1.0, delta=0.01)

  def test_v_max3_f16_basic(self):
    """V_MAX3_F16: maximum of three f16 values."""
    instructions = [
      s_mov_b32(s[0], f32_to_f16(1.0)),
      s_mov_b32(s[1], f32_to_f16(3.0)),
      s_mov_b32(s[2], f32_to_f16(2.0)),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_mov_b32_e32(v[2], s[2]),
      v_max3_f16(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = f16(st.vgpr[0][3] & 0xffff)
    self.assertAlmostEqual(result, 3.0, delta=0.01)

  def test_v_med3_f16_basic(self):
    """V_MED3_F16: median of three f16 values."""
    instructions = [
      s_mov_b32(s[0], f32_to_f16(3.0)),
      s_mov_b32(s[1], f32_to_f16(1.0)),
      s_mov_b32(s[2], f32_to_f16(2.0)),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_mov_b32_e32(v[2], s[2]),
      v_med3_f16(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = f16(st.vgpr[0][3] & 0xffff)
    self.assertAlmostEqual(result, 2.0, delta=0.01)

  def test_v_minmax_f16_basic(self):
    """V_MINMAX_F16: clamp(src0, min=src1, max=src2)."""
    instructions = [
      s_mov_b32(s[0], f32_to_f16(2.5)),
      s_mov_b32(s[1], f32_to_f16(1.0)),
      s_mov_b32(s[2], f32_to_f16(2.0)),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_mov_b32_e32(v[2], s[2]),
      v_minmax_f16(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = f16(st.vgpr[0][3] & 0xffff)
    self.assertAlmostEqual(result, 2.0, delta=0.01)

  def test_v_maxmin_f16_basic(self):
    """V_MAXMIN_F16: clamp(src0, min=src2, max=src1)."""
    instructions = [
      s_mov_b32(s[0], f32_to_f16(0.5)),
      s_mov_b32(s[1], f32_to_f16(2.0)),
      s_mov_b32(s[2], f32_to_f16(1.0)),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_mov_b32_e32(v[2], s[2]),
      v_maxmin_f16(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = f16(st.vgpr[0][3] & 0xffff)
    self.assertAlmostEqual(result, 1.0, delta=0.01)

  def test_v_min3_f16_with_neg(self):
    """V_MIN3_F16 with neg modifier: min(-3, 1, 2) = -3."""
    instructions = [
      s_mov_b32(s[0], f32_to_f16(3.0)),
      s_mov_b32(s[1], f32_to_f16(1.0)),
      s_mov_b32(s[2], f32_to_f16(2.0)),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_mov_b32_e32(v[2], s[2]),
      v_min3_f16(v[3], -v[0], v[1], v[2]),  # neg on first operand
    ]
    st = run_program(instructions, n_lanes=1)
    result = f16(st.vgpr[0][3] & 0xffff)
    self.assertAlmostEqual(result, -3.0, delta=0.01)

  def test_v_max3_f16_with_abs(self):
    """V_MAX3_F16 with abs modifier: max(|-3|, 1, 2) = 3."""
    instructions = [
      s_mov_b32(s[0], f32_to_f16(-3.0)),
      s_mov_b32(s[1], f32_to_f16(1.0)),
      s_mov_b32(s[2], f32_to_f16(2.0)),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_mov_b32_e32(v[2], s[2]),
      v_max3_f16(v[3], abs(v[0]), v[1], v[2]),  # abs on first operand
    ]
    st = run_program(instructions, n_lanes=1)
    result = f16(st.vgpr[0][3] & 0xffff)
    self.assertAlmostEqual(result, 3.0, delta=0.01)

  def test_v_med3_f16_opsel_hi(self):
    """V_MED3_F16 with opsel reading from hi half."""
    # Pack two f16 values: hi=5.0, lo=1.0
    packed = (f32_to_f16(5.0) << 16) | f32_to_f16(1.0)
    instructions = [
      s_mov_b32(s[0], packed),
      s_mov_b32(s[1], f32_to_f16(3.0)),
      s_mov_b32(s[2], f32_to_f16(4.0)),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_mov_b32_e32(v[2], s[2]),
      # Read hi half of v[0] (5.0), med3(5, 3, 4) = 4
      v_med3_f16(v[3], v[0].h, v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = f16(st.vgpr[0][3] & 0xffff)
    self.assertAlmostEqual(result, 4.0, delta=0.01)


class TestSadHi(unittest.TestCase):
  """Tests for V_SAD_HI_U8 instruction."""

  def test_v_sad_hi_u8_basic(self):
    """V_SAD_HI_U8: (sad << 16) + acc."""
    # |1-5| + |2-6| + |3-7| + |4-8| = 16, << 16 = 0x100000, + 100 = 0x100064
    instructions = [
      v_mov_b32_e32(v[0], 0x04030201),
      v_mov_b32_e32(v[1], 0x08070605),
      v_mov_b32_e32(v[2], 100),
      v_sad_hi_u8(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][3], (16 << 16) + 100)

  def test_v_sad_hi_u8_zero_diff(self):
    """V_SAD_HI_U8: identical inputs gives acc only."""
    instructions = [
      v_mov_b32_e32(v[0], 0x12345678),
      v_mov_b32_e32(v[2], 50),
      v_sad_hi_u8(v[3], v[0], v[0], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][3], 50)


class TestPermlane(unittest.TestCase):
  """Tests for V_PERMLANE16_B32 and V_PERMLANEX16_B32 instructions."""

  def test_v_permlane16_b32_identity(self):
    """V_PERMLANE16_B32 with identity permutation (lane i reads from lane i within row)."""
    # lanesel encodes 4 bits per position: position i gets lanesel[i*4+3:i*4]
    # Identity: position 0->0, 1->1, ..., 15->15
    # lanesel = 0xFEDCBA9876543210 (positions 15-0 in nibbles)
    instructions = [
      v_mov_b32_e32(v[0], 0xDEADBEEF),  # source data
      s_mov_b32(s[0], 0x76543210),       # lanesel low (positions 0-7)
      s_mov_b32(s[1], 0xFEDCBA98),       # lanesel high (positions 8-15)
      v_permlane16_b32(v[1], v[0], s[0], s[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    # Lane 0 reads from lane 0 (position 0 -> lanesel[3:0] = 0)
    self.assertEqual(st.vgpr[0][1], 0xDEADBEEF)

  def test_v_permlane16_b32_broadcast(self):
    """V_PERMLANE16_B32 broadcast lane 0 to all lanes in row."""
    # lanesel = all zeros -> all positions read from lane 0 within row
    instructions = [
      v_mov_b32_e32(v[0], 0xCAFEBABE),  # source data
      s_mov_b32(s[0], 0),                # lanesel low = 0 (all read lane 0)
      s_mov_b32(s[1], 0),                # lanesel high = 0
      v_permlane16_b32(v[1], v[0], s[0], s[1]),
    ]
    st = run_program(instructions, n_lanes=4)
    # All lanes read from lane 0 of their row
    for lane in range(4):
      self.assertEqual(st.vgpr[lane][1], 0xCAFEBABE)

  def test_v_permlanex16_b32_identity(self):
    """V_PERMLANEX16_B32 cross-row read with identity selection."""
    # In wave32: row 0 (lanes 0-15) reads from row 1 (lanes 16-31) and vice versa
    # With single lane in row 0, it reads from lane 0 of row 1 (lane 16)
    # But lane 16 doesn't exist in 1-lane test, so use 32 lanes
    instructions = [
      v_mov_b32_e32(v[0], 0x11111111),  # All lanes have this initially
      s_mov_b32(s[0], 0x76543210),       # lanesel low
      s_mov_b32(s[1], 0xFEDCBA98),       # lanesel high
      v_permlanex16_b32(v[1], v[0], s[0], s[1]),
    ]
    st = run_program(instructions, n_lanes=32)
    # Lane 0 in row 0 reads from lane 0 of row 1 (lane 16)
    self.assertEqual(st.vgpr[0][1], 0x11111111)
    # Lane 16 in row 1 reads from lane 0 of row 0 (lane 0)
    self.assertEqual(st.vgpr[16][1], 0x11111111)


if __name__ == '__main__':
  unittest.main()
