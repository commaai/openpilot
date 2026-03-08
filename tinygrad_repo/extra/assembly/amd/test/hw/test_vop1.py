"""Tests for VOP1 instructions - single operand vector operations.

Includes: v_mov_b32, v_cvt_*, v_sin_f32, v_rcp_f32, v_exp_f32, v_rndne_f32,
          v_floor_f32, v_trunc_f32, v_fract_f32, v_clz_i32_u32, v_ctz_i32_b32,
          v_readfirstlane_b32
"""
import unittest
from extra.assembly.amd.test.hw.helpers import *

class TestMov(unittest.TestCase):
  """Tests for V_MOV_B32."""

  def test_v_mov_b32(self):
    """V_MOV_B32 moves a value."""
    instructions = [
      s_mov_b32(s[0], 42),
      v_mov_b32_e32(v[0], s[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][0], 42)

  def test_v_mov_all_lanes(self):
    """V_MOV_B32 sets all lanes to the same value."""
    instructions = [
      s_mov_b32(s[0], 42),
      v_mov_b32_e32(v[0], s[0]),
    ]
    st = run_program(instructions, n_lanes=4)
    for lane in range(4):
      self.assertEqual(st.vgpr[lane][0], 42)

  def test_v_mov_b16_to_hi(self):
    """V_MOV_B16 can write to high 16 bits with .h suffix."""
    instructions = [
      s_mov_b32(s[0], 0x0000DEAD),  # lo=0xDEAD, hi=0
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b16_e32(v[0].h, 0x5678),  # Move 0x5678 to high half
    ]
    st = run_program(instructions, n_lanes=1)
    result_hi = (st.vgpr[0][0] >> 16) & 0xFFFF
    result_lo = st.vgpr[0][0] & 0xFFFF
    self.assertEqual(result_hi, 0x5678, f"Expected hi=0x5678, got 0x{result_hi:04x}")
    self.assertEqual(result_lo, 0xDEAD, f"Expected lo=0xDEAD (preserved), got 0x{result_lo:04x}")

  def test_v_mov_b16_to_lo(self):
    """V_MOV_B16 writes to low 16 bits by default."""
    instructions = [
      s_mov_b32(s[0], 0xBEEF0000),  # hi=0xBEEF, lo=0
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b16_e32(v[0], 0x1234),  # Move to low half
    ]
    st = run_program(instructions, n_lanes=1)
    result_hi = (st.vgpr[0][0] >> 16) & 0xFFFF
    result_lo = st.vgpr[0][0] & 0xFFFF
    self.assertEqual(result_lo, 0x1234, f"Expected lo=0x1234, got 0x{result_lo:04x}")
    self.assertEqual(result_hi, 0xBEEF, f"Expected hi=0xBEEF (preserved), got 0x{result_hi:04x}")


class TestTrigonometry(unittest.TestCase):
  """Tests for trigonometric instructions."""

  def test_v_sin_f32_small(self):
    """V_SIN_F32 computes sin for small values."""
    import math
    instructions = [
      v_mov_b32_e32(v[0], 1.0),
      v_sin_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i2f(st.vgpr[0][1])
    expected = math.sin(1.0 * 2 * math.pi)
    self.assertAlmostEqual(result, expected, places=4)

  def test_v_sin_f32_quarter(self):
    """V_SIN_F32 at 0.25 cycles = sin(pi/2) = 1.0."""
    instructions = [
      s_mov_b32(s[0], f2i(0.25)),
      v_mov_b32_e32(v[0], s[0]),
      v_sin_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i2f(st.vgpr[0][1])
    self.assertAlmostEqual(result, 1.0, places=4)

  def test_v_sin_f32_large(self):
    """V_SIN_F32 for large input value (132000.0)."""
    import math
    instructions = [
      s_mov_b32(s[0], f2i(132000.0)),
      v_mov_b32_e32(v[0], s[0]),
      v_sin_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i2f(st.vgpr[0][1])
    expected = math.sin(132000.0 * 2 * math.pi)
    self.assertAlmostEqual(result, expected, places=2, msg=f"sin(132000) got {result}, expected ~{expected}")


class TestRounding(unittest.TestCase):
  """Tests for rounding instructions."""

  def test_v_rndne_f32_half_even(self):
    """V_RNDNE_F32 rounds to nearest even."""
    instructions = [
      s_mov_b32(s[0], f2i(2.5)),
      v_mov_b32_e32(v[0], s[0]),
      v_rndne_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), 2.0, places=5)

  def test_v_rndne_f32_half_odd(self):
    """V_RNDNE_F32 rounds 3.5 to 4 (nearest even)."""
    instructions = [
      s_mov_b32(s[0], f2i(3.5)),
      v_mov_b32_e32(v[0], s[0]),
      v_rndne_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), 4.0, places=5)

  def test_v_rndne_f32_large(self):
    """V_RNDNE_F32 with large value (like sin reduction uses)."""
    val = 100000.0 * 0.15915494309189535
    instructions = [
      s_mov_b32(s[0], f2i(val)),
      v_mov_b32_e32(v[0], s[0]),
      v_rndne_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    expected = round(val)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), expected, places=0)

  def test_v_floor_f32(self):
    """V_FLOOR_F32 floors to integer."""
    instructions = [
      s_mov_b32(s[0], f2i(3.7)),
      v_mov_b32_e32(v[0], s[0]),
      v_floor_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), 3.0, places=5)

  def test_v_trunc_f32(self):
    """V_TRUNC_F32 truncates toward zero."""
    instructions = [
      s_mov_b32(s[0], f2i(-3.7)),
      v_mov_b32_e32(v[0], s[0]),
      v_trunc_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), -3.0, places=5)

  def test_v_fract_f32(self):
    """V_FRACT_F32 returns fractional part."""
    instructions = [
      s_mov_b32(s[0], f2i(3.75)),
      v_mov_b32_e32(v[0], s[0]),
      v_fract_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), 0.75, places=5)

  def test_v_fract_f32_large(self):
    """V_FRACT_F32 with large value - precision matters here."""
    instructions = [
      s_mov_b32(s[0], f2i(132000.25)),
      v_mov_b32_e32(v[0], s[0]),
      v_fract_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i2f(st.vgpr[0][1])
    self.assertGreaterEqual(result, 0.0)
    self.assertLess(result, 1.0)


class TestConversion(unittest.TestCase):
  """Tests for conversion instructions."""

  def test_v_cvt_i32_f32_positive(self):
    """V_CVT_I32_F32 converts float to signed int."""
    instructions = [
      s_mov_b32(s[0], f2i(42.7)),
      v_mov_b32_e32(v[0], s[0]),
      v_cvt_i32_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 42)

  def test_v_cvt_i32_f32_negative(self):
    """V_CVT_I32_F32 converts negative float to signed int."""
    instructions = [
      s_mov_b32(s[0], f2i(-42.7)),
      v_mov_b32_e32(v[0], s[0]),
      v_cvt_i32_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1] & 0xffffffff, (-42) & 0xffffffff)

  def test_v_cvt_i32_f32_large(self):
    """V_CVT_I32_F32 with large float (used in sin for quadrant)."""
    instructions = [
      s_mov_b32(s[0], f2i(15915.0)),
      v_mov_b32_e32(v[0], s[0]),
      v_cvt_i32_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 15915)

  def test_v_cvt_f32_i32(self):
    """V_CVT_F32_I32 converts signed int to float."""
    instructions = [
      s_mov_b32(s[0], 42),
      v_mov_b32_e32(v[0], s[0]),
      v_cvt_f32_i32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), 42.0, places=5)

  def test_v_cvt_f32_u32(self):
    """V_CVT_F32_U32 converts unsigned int to float."""
    instructions = [
      s_mov_b32(s[0], 0xffffffff),
      v_mov_b32_e32(v[0], s[0]),
      v_cvt_f32_u32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), 4294967296.0, places=-5)


class TestF16Conversions(unittest.TestCase):
  """Tests for f16 conversion instructions."""

  def test_v_cvt_f16_f32_basic(self):
    """V_CVT_F16_F32 converts f32 to f16 in low 16 bits."""
    instructions = [
      v_mov_b32_e32(v[0], 1.0),
      v_cvt_f16_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][1]
    lo_bits = result & 0xffff
    self.assertEqual(lo_bits, 0x3c00, f"Expected 0x3c00, got 0x{lo_bits:04x}")

  def test_v_cvt_f16_f32_negative(self):
    """V_CVT_F16_F32 converts negative f32 to f16."""
    instructions = [
      v_mov_b32_e32(v[0], -2.0),
      v_cvt_f16_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][1]
    lo_bits = result & 0xffff
    self.assertEqual(lo_bits, 0xc000, f"Expected 0xc000, got 0x{lo_bits:04x}")

  def test_v_cvt_f16_f32_small(self):
    """V_CVT_F16_F32 converts small f32 value."""
    instructions = [
      v_mov_b32_e32(v[0], 0.5),
      v_cvt_f16_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][1]
    lo_bits = result & 0xffff
    expected = f32_to_f16(0.5)
    self.assertEqual(lo_bits, expected, f"Expected 0x{expected:04x}, got 0x{lo_bits:04x}")

  def test_v_cvt_f16_f32_preserves_high_bits(self):
    """V_CVT_F16_F32 preserves high 16 bits of destination."""
    instructions = [
      s_mov_b32(s[0], 0xdead0000),
      v_mov_b32_e32(v[1], s[0]),
      v_mov_b32_e32(v[0], 1.0),
      v_cvt_f16_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][1]
    hi_bits = (result >> 16) & 0xffff
    lo_bits = result & 0xffff
    self.assertEqual(lo_bits, 0x3c00, f"Low bits should be 0x3c00, got 0x{lo_bits:04x}")
    self.assertEqual(hi_bits, 0xdead, f"High bits should be preserved as 0xdead, got 0x{hi_bits:04x}")

  def test_v_cvt_f16_f32_same_src_dst_preserves_high_bits(self):
    """V_CVT_F16_F32 with same src/dst preserves high bits of source."""
    instructions = [
      v_mov_b32_e32(v[0], 1.0),
      v_cvt_f16_f32_e32(v[0], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][0]
    self.assertEqual(result, 0x3f803c00, f"Expected 0x3f803c00, got 0x{result:08x}")

  def test_v_cvt_f16_f32_reads_full_32bit_source(self):
    """V_CVT_F16_F32 must read full 32-bit f32 source."""
    instructions = [
      s_mov_b32(s[0], 0x3fc00000),  # f32 1.5
      v_mov_b32_e32(v[0], s[0]),
      v_cvt_f16_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][1]
    lo_bits = result & 0xffff
    self.assertEqual(lo_bits, 0x3e00, f"Expected f16(1.5)=0x3e00, got 0x{lo_bits:04x} ({f16(lo_bits)})")

  def test_v_cvt_i16_f16_zero(self):
    """V_CVT_I16_F16 converts f16 zero to i16 zero."""
    instructions = [
      v_mov_b32_e32(v[0], 0),
      v_cvt_i16_f16_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][1] & 0xffff
    self.assertEqual(result, 0, f"Expected 0, got {result}")

  def test_v_cvt_i16_f16_one(self):
    """V_CVT_I16_F16 converts f16 1.0 to i16 1."""
    instructions = [
      s_mov_b32(s[0], 0x3c00),  # f16 1.0 in low bits
      v_mov_b32_e32(v[0], s[0]),
      v_cvt_i16_f16_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][1] & 0xffff
    self.assertEqual(result, 1, f"Expected 1, got {result}")

  def test_v_cvt_i16_f16_negative(self):
    """V_CVT_I16_F16 converts f16 -2.0 to i16 -2."""
    instructions = [
      s_mov_b32(s[0], 0xc000),  # f16 -2.0 in low bits
      v_mov_b32_e32(v[0], s[0]),
      v_cvt_i16_f16_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][1] & 0xffff
    self.assertEqual(result, (-2) & 0xffff, f"Expected 0xfffe (-2), got 0x{result:04x}")

  def test_v_cvt_i16_f16_from_hi(self):
    """V_CVT_I16_F16 can read from high 16 bits with opsel."""
    instructions = [
      s_mov_b32(s[0], 0x3c000000),  # f16 1.0 in HIGH bits, 0 in low
      v_mov_b32_e32(v[0], s[0]),
      VOP3(VOP3Op.V_CVT_I16_F16, vdst=v[1], src0=v[0], opsel=0b0001),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][1] & 0xffff
    self.assertEqual(result, 1, f"Expected 1 from high bits, got {result}")


class TestF64Conversions(unittest.TestCase):
  """Tests for f64 conversion instructions. Regression tests for f32_to_f64/f64_to_f32."""

  def test_v_cvt_f64_f32_one(self):
    """V_CVT_F64_F32 converts f32 1.0 to f64."""
    instructions = [
      s_mov_b32(s[0], f2i(1.0)),
      v_mov_b32_e32(v[0], s[0]),
      v_cvt_f64_f32_e32(v[2:3], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i642f((st.vgpr[0][3] << 32) | st.vgpr[0][2])
    self.assertAlmostEqual(result, 1.0, places=10)

  def test_v_cvt_f64_f32_negative(self):
    """V_CVT_F64_F32 converts f32 -2.5 to f64."""
    instructions = [
      s_mov_b32(s[0], f2i(-2.5)),
      v_mov_b32_e32(v[0], s[0]),
      v_cvt_f64_f32_e32(v[2:3], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i642f((st.vgpr[0][3] << 32) | st.vgpr[0][2])
    self.assertAlmostEqual(result, -2.5, places=10)

  def test_v_cvt_f64_f32_pi(self):
    """V_CVT_F64_F32 converts f32 pi to f64."""
    import math
    instructions = [
      s_mov_b32(s[0], f2i(3.14159265)),
      v_mov_b32_e32(v[0], s[0]),
      v_cvt_f64_f32_e32(v[2:3], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i642f((st.vgpr[0][3] << 32) | st.vgpr[0][2])
    self.assertAlmostEqual(result, 3.14159265, places=5)

  def test_v_cvt_f64_f32_zero(self):
    """V_CVT_F64_F32 converts f32 0.0 to f64."""
    instructions = [
      v_mov_b32_e32(v[0], 0),
      v_cvt_f64_f32_e32(v[2:3], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i642f((st.vgpr[0][3] << 32) | st.vgpr[0][2])
    self.assertEqual(result, 0.0)

  def test_v_cvt_f32_f64_one(self):
    """V_CVT_F32_F64 converts f64 1.0 to f32."""
    f64_bits = f2i64(1.0)
    lo, hi = f64_bits & 0xFFFFFFFF, (f64_bits >> 32) & 0xFFFFFFFF
    instructions = [
      s_mov_b32(s[0], lo),
      s_mov_b32(s[1], hi),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_cvt_f32_f64_e32(v[2], v[0:1]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i2f(st.vgpr[0][2])
    self.assertAlmostEqual(result, 1.0, places=5)

  def test_v_cvt_f32_f64_negative(self):
    """V_CVT_F32_F64 converts f64 -3.5 to f32."""
    f64_bits = f2i64(-3.5)
    lo, hi = f64_bits & 0xFFFFFFFF, (f64_bits >> 32) & 0xFFFFFFFF
    instructions = [
      s_mov_b32(s[0], lo),
      s_mov_b32(s[1], hi),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_cvt_f32_f64_e32(v[2], v[0:1]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i2f(st.vgpr[0][2])
    self.assertAlmostEqual(result, -3.5, places=5)

  def test_v_cvt_f32_f64_large(self):
    """V_CVT_F32_F64 converts large f64 to f32."""
    f64_bits = f2i64(123456.789)
    lo, hi = f64_bits & 0xFFFFFFFF, (f64_bits >> 32) & 0xFFFFFFFF
    instructions = [
      s_mov_b32(s[0], lo),
      s_mov_b32(s[1], hi),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_cvt_f32_f64_e32(v[2], v[0:1]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i2f(st.vgpr[0][2])
    self.assertAlmostEqual(result, 123456.789, places=0)

  def test_v_cvt_f64_i32_positive(self):
    """V_CVT_F64_I32 converts positive i32 to f64."""
    instructions = [
      s_mov_b32(s[0], 42),
      v_mov_b32_e32(v[0], s[0]),
      v_cvt_f64_i32_e32(v[2:3], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i642f((st.vgpr[0][3] << 32) | st.vgpr[0][2])
    self.assertAlmostEqual(result, 42.0, places=10)

  def test_v_cvt_f64_i32_negative(self):
    """V_CVT_F64_I32 converts negative i32 to f64."""
    instructions = [
      s_mov_b32(s[0], 0xFFFFFFFF),  # -1 as i32
      v_mov_b32_e32(v[0], s[0]),
      v_cvt_f64_i32_e32(v[2:3], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i642f((st.vgpr[0][3] << 32) | st.vgpr[0][2])
    self.assertAlmostEqual(result, -1.0, places=10)

  def test_v_cvt_f64_u32_large(self):
    """V_CVT_F64_U32 converts large u32 to f64."""
    instructions = [
      s_mov_b32(s[0], 0xFFFFFFFF),  # max u32
      v_mov_b32_e32(v[0], s[0]),
      v_cvt_f64_u32_e32(v[2:3], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i642f((st.vgpr[0][3] << 32) | st.vgpr[0][2])
    self.assertAlmostEqual(result, 4294967295.0, places=0)

  def test_v_cvt_f64_u32_zero(self):
    """V_CVT_F64_U32 converts 0 to f64."""
    instructions = [
      v_mov_b32_e32(v[0], 0),
      v_cvt_f64_u32_e32(v[2:3], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i642f((st.vgpr[0][3] << 32) | st.vgpr[0][2])
    self.assertEqual(result, 0.0)


class TestClz(unittest.TestCase):
  """Tests for V_CLZ_I32_U32 - count leading zeros."""

  def test_v_clz_i32_u32_zero(self):
    """V_CLZ_I32_U32 of 0 returns -1 (all bits are 0)."""
    instructions = [
      v_mov_b32_e32(v[0], 0),
      v_clz_i32_u32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 0xFFFFFFFF)

  def test_v_clz_i32_u32_one(self):
    """V_CLZ_I32_U32 of 1 returns 31 (31 leading zeros)."""
    instructions = [
      v_mov_b32_e32(v[0], 1),
      v_clz_i32_u32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 31)

  def test_v_clz_i32_u32_msb_set(self):
    """V_CLZ_I32_U32 of 0x80000000 returns 0 (no leading zeros)."""
    instructions = [
      s_mov_b32(s[0], 0x80000000),
      v_mov_b32_e32(v[0], s[0]),
      v_clz_i32_u32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 0)

  def test_v_clz_i32_u32_half(self):
    """V_CLZ_I32_U32 of 0x8000 (bit 15) returns 16."""
    instructions = [
      s_mov_b32(s[0], 0x8000),
      v_mov_b32_e32(v[0], s[0]),
      v_clz_i32_u32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 16)

  def test_v_clz_i32_u32_all_ones(self):
    """V_CLZ_I32_U32 of 0xFFFFFFFF returns 0."""
    instructions = [
      s_mov_b32(s[0], 0xFFFFFFFF),
      v_mov_b32_e32(v[0], s[0]),
      v_clz_i32_u32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 0)


class TestCtz(unittest.TestCase):
  """Tests for V_CTZ_I32_B32 - count trailing zeros."""

  def test_v_ctz_i32_b32_zero(self):
    """V_CTZ_I32_B32 of 0 returns -1 (all bits are 0)."""
    instructions = [
      v_mov_b32_e32(v[0], 0),
      v_ctz_i32_b32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 0xFFFFFFFF)

  def test_v_ctz_i32_b32_one(self):
    """V_CTZ_I32_B32 of 1 returns 0 (no trailing zeros)."""
    instructions = [
      v_mov_b32_e32(v[0], 1),
      v_ctz_i32_b32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 0)

  def test_v_ctz_i32_b32_msb_set(self):
    """V_CTZ_I32_B32 of 0x80000000 returns 31."""
    instructions = [
      s_mov_b32(s[0], 0x80000000),
      v_mov_b32_e32(v[0], s[0]),
      v_ctz_i32_b32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 31)

  def test_v_ctz_i32_b32_half(self):
    """V_CTZ_I32_B32 of 0x8000 (bit 15) returns 15."""
    instructions = [
      s_mov_b32(s[0], 0x8000),
      v_mov_b32_e32(v[0], s[0]),
      v_ctz_i32_b32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 15)

  def test_v_ctz_i32_b32_all_ones(self):
    """V_CTZ_I32_B32 of 0xFFFFFFFF returns 0."""
    instructions = [
      s_mov_b32(s[0], 0xFFFFFFFF),
      v_mov_b32_e32(v[0], s[0]),
      v_ctz_i32_b32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 0)


class TestRcp(unittest.TestCase):
  """Tests for V_RCP_F32 - reciprocal."""

  def test_v_rcp_f32_normal(self):
    """V_RCP_F32 of 2.0 returns 0.5."""
    instructions = [
      v_mov_b32_e32(v[0], 2.0),
      v_rcp_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), 0.5, places=5)

  def test_v_rcp_f32_inf(self):
    """V_RCP_F32 of +inf returns 0."""
    instructions = [
      s_mov_b32(s[0], 0x7f800000),
      v_mov_b32_e32(v[0], s[0]),
      v_rcp_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(i2f(st.vgpr[0][1]), 0.0)

  def test_v_rcp_f32_neg_inf(self):
    """V_RCP_F32 of -inf returns -0."""
    instructions = [
      s_mov_b32(s[0], 0xff800000),
      v_mov_b32_e32(v[0], s[0]),
      v_rcp_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i2f(st.vgpr[0][1])
    self.assertEqual(result, 0.0)
    self.assertEqual(st.vgpr[0][1], 0x80000000)

  def test_v_rcp_f32_zero(self):
    """V_RCP_F32 of 0 returns +inf."""
    import math
    instructions = [
      v_mov_b32_e32(v[0], 0),
      v_rcp_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertTrue(math.isinf(i2f(st.vgpr[0][1])))


class TestExp(unittest.TestCase):
  """Tests for V_EXP_F32 - base-2 exponential."""

  def test_v_exp_f32_large_negative(self):
    """V_EXP_F32 of large negative value (2^-100) returns very small number."""
    instructions = [
      s_mov_b32(s[0], f2i(-100.0)),
      v_mov_b32_e32(v[0], s[0]),
      v_exp_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i2f(st.vgpr[0][1])
    self.assertLess(result, 1e-20)

  def test_v_exp_f32_large_positive(self):
    """V_EXP_F32 of large positive value (2^100) returns very large number."""
    instructions = [
      s_mov_b32(s[0], f2i(100.0)),
      v_mov_b32_e32(v[0], s[0]),
      v_exp_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i2f(st.vgpr[0][1])
    self.assertGreater(result, 1e20)


class TestReadFirstLane(unittest.TestCase):
  """Tests for V_READFIRSTLANE_B32."""

  def _readfirstlane(self, sdst, vsrc):
    """Helper to create V_READFIRSTLANE_B32 with SGPR destination."""
    return v_readfirstlane_b32_e32(sdst, vsrc)

  def test_v_readfirstlane_b32_basic(self):
    """V_READFIRSTLANE_B32 reads from the first active lane."""
    instructions = [
      v_lshlrev_b32_e32(v[0], 2, v[255]),
      v_add_nc_u32_e32(v[0], 1000, v[0]),
      self._readfirstlane(s[0], v[0]),
      v_mov_b32_e32(v[1], s[0]),
    ]
    st = run_program(instructions, n_lanes=4)
    for lane in range(4):
      self.assertEqual(st.vgpr[lane][1], 1000)

  def test_v_readfirstlane_b32_different_vgpr(self):
    """V_READFIRSTLANE_B32 reading from different VGPR index."""
    instructions = [
      v_lshlrev_b32_e32(v[7], 5, v[255]),
      v_add_nc_u32_e32(v[7], 200, v[7]),
      self._readfirstlane(s[0], v[7]),
      v_mov_b32_e32(v[8], s[0]),
    ]
    st = run_program(instructions, n_lanes=4)
    for lane in range(4):
      self.assertEqual(st.vgpr[lane][8], 200)


class TestCvtF16Modifiers(unittest.TestCase):
  """Tests for V_CVT_F32_F16 with VOP3 abs/neg modifiers."""

  def test_v_cvt_f32_f16_abs_negative(self):
    """V_CVT_F32_F16 with |abs| on negative value."""
    f16_neg1 = f32_to_f16(-1.0)  # 0xbc00
    instructions = [
      s_mov_b32(s[0], f16_neg1),
      v_mov_b32_e32(v[1], s[0]),
      v_cvt_f32_f16_e64(v[0], abs(v[1])),  # |(-1.0)| = 1.0
    ]
    st = run_program(instructions, n_lanes=1)
    result = i2f(st.vgpr[0][0])
    self.assertAlmostEqual(result, 1.0, places=5)

  def test_v_cvt_f32_f16_abs_positive(self):
    """V_CVT_F32_F16 with |abs| on positive value (should stay positive)."""
    f16_2 = f32_to_f16(2.0)  # 0x4000
    instructions = [
      s_mov_b32(s[0], f16_2),
      v_mov_b32_e32(v[1], s[0]),
      v_cvt_f32_f16_e64(v[0], abs(v[1])),  # |2.0| = 2.0
    ]
    st = run_program(instructions, n_lanes=1)
    result = i2f(st.vgpr[0][0])
    self.assertAlmostEqual(result, 2.0, places=5)

  def test_v_cvt_f32_f16_neg_positive(self):
    """V_CVT_F32_F16 with neg on positive value."""
    f16_2 = f32_to_f16(2.0)  # 0x4000
    instructions = [
      s_mov_b32(s[0], f16_2),
      v_mov_b32_e32(v[1], s[0]),
      v_cvt_f32_f16_e64(v[0], -v[1]),  # -(2.0) = -2.0
    ]
    st = run_program(instructions, n_lanes=1)
    result = i2f(st.vgpr[0][0])
    self.assertAlmostEqual(result, -2.0, places=5)

  def test_v_cvt_f32_f16_neg_negative(self):
    """V_CVT_F32_F16 with neg on negative value (double negative)."""
    f16_neg2 = f32_to_f16(-2.0)  # 0xc000
    instructions = [
      s_mov_b32(s[0], f16_neg2),
      v_mov_b32_e32(v[1], s[0]),
      v_cvt_f32_f16_e64(v[0], -v[1]),  # -(-2.0) = 2.0
    ]
    st = run_program(instructions, n_lanes=1)
    result = i2f(st.vgpr[0][0])
    self.assertAlmostEqual(result, 2.0, places=5)

  def test_v_cvt_f16_f32_then_pack_for_wmma(self):
    """CVT F32->F16 followed by pack (common WMMA pattern)."""
    f32_val = 3.5
    instructions = [
      s_mov_b32(s[0], f2i(f32_val)),
      v_mov_b32_e32(v[0], s[0]),
      v_cvt_f16_f32_e32(v[1], v[0]),
      v_pack_b32_f16(v[2], v[1], v[1]),  # Pack same value
    ]
    st = run_program(instructions, n_lanes=1)
    lo = f16(st.vgpr[0][2] & 0xffff)
    hi = f16((st.vgpr[0][2] >> 16) & 0xffff)
    self.assertAlmostEqual(lo, f32_val, places=1)
    self.assertAlmostEqual(hi, f32_val, places=1)


class TestConversionRounding(unittest.TestCase):
  """Tests for conversion rounding behavior."""

  def test_cvt_f32_to_i32_round_toward_zero(self):
    """F32 to I32 should truncate (round toward zero)."""
    instructions = [
      v_mov_b32_e32(v[0], 2.9),
      v_mov_b32_e32(v[1], -2.9),
      v_cvt_i32_f32_e32(v[2], v[0]),
      v_cvt_i32_f32_e32(v[3], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][2], 2, "2.9 -> 2")
    self.assertEqual(st.vgpr[0][3] & 0xFFFFFFFF, 0xFFFFFFFE, "-2.9 -> -2")

  def test_cvt_f32_to_u32_negative(self):
    """F32 to U32 with negative input should clamp to 0."""
    instructions = [
      v_mov_b32_e32(v[0], -1.0),
      v_cvt_u32_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 0)

  def test_rndne_f32_half_even(self):
    """V_RNDNE_F32 should round to nearest even."""
    instructions = [
      v_mov_b32_e32(v[0], 2.5),
      v_mov_b32_e32(v[1], 3.5),
      v_mov_b32_e32(v[2], 4.5),
      v_rndne_f32_e32(v[3], v[0]),
      v_rndne_f32_e32(v[4], v[1]),
      v_rndne_f32_e32(v[5], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][3]), 2.0, places=5)  # 2.5 -> 2 (even)
    self.assertAlmostEqual(i2f(st.vgpr[0][4]), 4.0, places=5)  # 3.5 -> 4 (even)
    self.assertAlmostEqual(i2f(st.vgpr[0][5]), 4.0, places=5)  # 4.5 -> 4 (even)

  def test_f16_to_f32_precision(self):
    """F16 to F32 conversion precision."""
    f16_val = f32_to_f16(1.5)
    instructions = [
      s_mov_b32(s[0], f16_val),
      v_mov_b32_e32(v[0], s[0]),
      v_cvt_f32_f16_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), 1.5, places=5)

  def test_f16_denormal_to_f32(self):
    """F16 denormal converts to small positive f32."""
    f16_denorm = 0x0001  # Smallest positive f16 denormal
    instructions = [
      v_mov_b32_e32(v[0], f16_denorm),
      v_cvt_f32_f16_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i2f(st.vgpr[0][1])
    self.assertGreater(result, 0)
    self.assertLess(result, 1e-6)


class TestSqrt(unittest.TestCase):
  """Tests for V_SQRT_F32 - square root."""

  def test_v_sqrt_f32_normal(self):
    """V_SQRT_F32 of 4.0 returns 2.0."""
    instructions = [
      v_mov_b32_e32(v[0], 4.0),
      v_sqrt_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), 2.0, places=5)

  def test_v_sqrt_f32_one(self):
    """V_SQRT_F32 of 1.0 returns 1.0."""
    instructions = [
      v_mov_b32_e32(v[0], 1.0),
      v_sqrt_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), 1.0, places=5)

  def test_v_sqrt_f32_zero(self):
    """V_SQRT_F32 of 0.0 returns 0.0."""
    instructions = [
      v_mov_b32_e32(v[0], 0),
      v_sqrt_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(i2f(st.vgpr[0][1]), 0.0)

  def test_v_sqrt_f32_neg_zero(self):
    """V_SQRT_F32 of -0.0 returns -0.0."""
    instructions = [
      s_mov_b32(s[0], 0x80000000),  # -0.0
      v_mov_b32_e32(v[0], s[0]),
      v_sqrt_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 0x80000000)  # -0.0

  def test_v_sqrt_f32_inf(self):
    """V_SQRT_F32 of +inf returns +inf."""
    import math
    instructions = [
      s_mov_b32(s[0], 0x7f800000),  # +inf
      v_mov_b32_e32(v[0], s[0]),
      v_sqrt_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertTrue(math.isinf(i2f(st.vgpr[0][1])))
    self.assertGreater(i2f(st.vgpr[0][1]), 0)

  def test_v_sqrt_f32_negative(self):
    """V_SQRT_F32 of negative value returns NaN."""
    import math
    instructions = [
      v_mov_b32_e32(v[0], -1.0),
      v_sqrt_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertTrue(math.isnan(i2f(st.vgpr[0][1])))

  def test_v_sqrt_f32_nan(self):
    """V_SQRT_F32 of NaN returns NaN."""
    import math
    instructions = [
      s_mov_b32(s[0], 0x7fc00000),  # quiet NaN
      v_mov_b32_e32(v[0], s[0]),
      v_sqrt_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertTrue(math.isnan(i2f(st.vgpr[0][1])))

  def test_v_sqrt_f32_small(self):
    """V_SQRT_F32 of small value (0.25) returns 0.5."""
    instructions = [
      v_mov_b32_e32(v[0], 0.25),
      v_sqrt_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), 0.5, places=5)


class TestRsq(unittest.TestCase):
  """Tests for V_RSQ_F32 - reciprocal square root (1/sqrt(x))."""

  def test_v_rsq_f32_normal(self):
    """V_RSQ_F32 of 4.0 returns 0.5."""
    instructions = [
      v_mov_b32_e32(v[0], 4.0),
      v_rsq_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), 0.5, places=5)

  def test_v_rsq_f32_one(self):
    """V_RSQ_F32 of 1.0 returns 1.0."""
    instructions = [
      v_mov_b32_e32(v[0], 1.0),
      v_rsq_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), 1.0, places=5)

  def test_v_rsq_f32_zero(self):
    """V_RSQ_F32 of 0 returns +inf."""
    import math
    instructions = [
      v_mov_b32_e32(v[0], 0),
      v_rsq_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertTrue(math.isinf(i2f(st.vgpr[0][1])))
    self.assertGreater(i2f(st.vgpr[0][1]), 0)

  def test_v_rsq_f32_neg_zero(self):
    """V_RSQ_F32 of -0.0 returns -inf."""
    import math
    instructions = [
      s_mov_b32(s[0], 0x80000000),  # -0.0
      v_mov_b32_e32(v[0], s[0]),
      v_rsq_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertTrue(math.isinf(i2f(st.vgpr[0][1])))
    self.assertLess(i2f(st.vgpr[0][1]), 0)

  def test_v_rsq_f32_inf(self):
    """V_RSQ_F32 of +inf returns 0."""
    instructions = [
      s_mov_b32(s[0], 0x7f800000),  # +inf
      v_mov_b32_e32(v[0], s[0]),
      v_rsq_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(i2f(st.vgpr[0][1]), 0.0)

  def test_v_rsq_f32_negative(self):
    """V_RSQ_F32 of negative value returns NaN."""
    import math
    instructions = [
      v_mov_b32_e32(v[0], -1.0),
      v_rsq_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertTrue(math.isnan(i2f(st.vgpr[0][1])))

  def test_v_rsq_f32_large(self):
    """V_RSQ_F32 of large value."""
    instructions = [
      s_mov_b32(s[0], f2i(1e10)),
      v_mov_b32_e32(v[0], s[0]),
      v_rsq_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i2f(st.vgpr[0][1])
    # 1/sqrt(1e10) ~= 1e-5
    self.assertAlmostEqual(result, 1e-5, places=8)


class TestLog(unittest.TestCase):
  """Tests for V_LOG_F32 - base-2 logarithm."""

  def test_v_log_f32_one(self):
    """V_LOG_F32 of 1.0 returns 0.0."""
    instructions = [
      v_mov_b32_e32(v[0], 1.0),
      v_log_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), 0.0, places=4)

  def test_v_log_f32_two(self):
    """V_LOG_F32 of 2.0 returns 1.0."""
    instructions = [
      v_mov_b32_e32(v[0], 2.0),
      v_log_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), 1.0, places=4)

  def test_v_log_f32_four(self):
    """V_LOG_F32 of 4.0 returns 2.0."""
    instructions = [
      v_mov_b32_e32(v[0], 4.0),
      v_log_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), 2.0, places=4)

  def test_v_log_f32_half(self):
    """V_LOG_F32 of 0.5 returns -1.0."""
    instructions = [
      v_mov_b32_e32(v[0], 0.5),
      v_log_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), -1.0, places=4)

  def test_v_log_f32_zero(self):
    """V_LOG_F32 of 0 returns -inf."""
    import math
    instructions = [
      v_mov_b32_e32(v[0], 0),
      v_log_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertTrue(math.isinf(i2f(st.vgpr[0][1])))
    self.assertLess(i2f(st.vgpr[0][1]), 0)

  def test_v_log_f32_inf(self):
    """V_LOG_F32 of +inf returns +inf."""
    import math
    instructions = [
      s_mov_b32(s[0], 0x7f800000),  # +inf
      v_mov_b32_e32(v[0], s[0]),
      v_log_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertTrue(math.isinf(i2f(st.vgpr[0][1])))
    self.assertGreater(i2f(st.vgpr[0][1]), 0)

  def test_v_log_f32_negative(self):
    """V_LOG_F32 of negative value returns NaN."""
    import math
    instructions = [
      v_mov_b32_e32(v[0], -1.0),
      v_log_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertTrue(math.isnan(i2f(st.vgpr[0][1])))


class TestCos(unittest.TestCase):
  """Tests for V_COS_F32 - cosine (input in cycles, not radians)."""

  def test_v_cos_f32_zero(self):
    """V_COS_F32 at 0 cycles = cos(0) = 1.0."""
    instructions = [
      v_mov_b32_e32(v[0], 0),
      v_cos_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), 1.0, places=4)

  def test_v_cos_f32_quarter(self):
    """V_COS_F32 at 0.25 cycles = cos(pi/2) = 0.0."""
    instructions = [
      s_mov_b32(s[0], f2i(0.25)),
      v_mov_b32_e32(v[0], s[0]),
      v_cos_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), 0.0, places=4)

  def test_v_cos_f32_half(self):
    """V_COS_F32 at 0.5 cycles = cos(pi) = -1.0."""
    instructions = [
      s_mov_b32(s[0], f2i(0.5)),
      v_mov_b32_e32(v[0], s[0]),
      v_cos_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), -1.0, places=4)

  def test_v_cos_f32_full(self):
    """V_COS_F32 at 1.0 cycles = cos(2*pi) = 1.0."""
    instructions = [
      v_mov_b32_e32(v[0], 1.0),
      v_cos_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), 1.0, places=4)

  def test_v_cos_f32_large(self):
    """V_COS_F32 for large input value."""
    import math
    val = 132000.0
    instructions = [
      s_mov_b32(s[0], f2i(val)),
      v_mov_b32_e32(v[0], s[0]),
      v_cos_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i2f(st.vgpr[0][1])
    expected = math.cos(val * 2 * math.pi)
    self.assertAlmostEqual(result, expected, places=2)


class TestFractEdgeCases(unittest.TestCase):
  """Additional edge case tests for V_FRACT_F32."""

  def test_v_fract_f32_negative(self):
    """V_FRACT_F32 of -1.25 should return 0.75 (fract is always positive)."""
    instructions = [
      s_mov_b32(s[0], f2i(-1.25)),
      v_mov_b32_e32(v[0], s[0]),
      v_fract_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i2f(st.vgpr[0][1])
    self.assertAlmostEqual(result, 0.75, places=5)

  def test_v_fract_f32_negative_small(self):
    """V_FRACT_F32 of -0.25 should return 0.75."""
    instructions = [
      s_mov_b32(s[0], f2i(-0.25)),
      v_mov_b32_e32(v[0], s[0]),
      v_fract_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i2f(st.vgpr[0][1])
    self.assertAlmostEqual(result, 0.75, places=5)

  def test_v_fract_f32_whole_number(self):
    """V_FRACT_F32 of 5.0 should return 0.0."""
    instructions = [
      v_mov_b32_e32(v[0], 5.0),
      v_fract_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i2f(st.vgpr[0][1])
    self.assertAlmostEqual(result, 0.0, places=5)

  def test_v_fract_f32_negative_whole(self):
    """V_FRACT_F32 of -5.0 should return 0.0."""
    instructions = [
      v_mov_b32_e32(v[0], -5.0),
      v_fract_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i2f(st.vgpr[0][1])
    self.assertAlmostEqual(result, 0.0, places=5)

  def test_v_fract_f32_zero(self):
    """V_FRACT_F32 of 0.0 returns 0.0."""
    instructions = [
      v_mov_b32_e32(v[0], 0),
      v_fract_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(i2f(st.vgpr[0][1]), 0.0)

  def test_v_fract_f32_inf(self):
    """V_FRACT_F32 of +inf returns NaN."""
    import math
    instructions = [
      s_mov_b32(s[0], 0x7f800000),  # +inf
      v_mov_b32_e32(v[0], s[0]),
      v_fract_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertTrue(math.isnan(i2f(st.vgpr[0][1])))

  def test_v_fract_f32_nan(self):
    """V_FRACT_F32 of NaN returns NaN."""
    import math
    instructions = [
      s_mov_b32(s[0], 0x7fc00000),  # quiet NaN
      v_mov_b32_e32(v[0], s[0]),
      v_fract_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertTrue(math.isnan(i2f(st.vgpr[0][1])))


class TestF16EdgeCases(unittest.TestCase):
  """Additional F16 conversion edge cases."""

  def test_v_cvt_f32_f16_inf(self):
    """V_CVT_F32_F16 converts f16 infinity to f32 infinity."""
    import math
    instructions = [
      s_mov_b32(s[0], 0x7c00),  # f16 +inf
      v_mov_b32_e32(v[0], s[0]),
      v_cvt_f32_f16_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertTrue(math.isinf(i2f(st.vgpr[0][1])))
    self.assertGreater(i2f(st.vgpr[0][1]), 0)

  def test_v_cvt_f32_f16_neg_inf(self):
    """V_CVT_F32_F16 converts f16 -inf to f32 -inf."""
    import math
    instructions = [
      s_mov_b32(s[0], 0xfc00),  # f16 -inf
      v_mov_b32_e32(v[0], s[0]),
      v_cvt_f32_f16_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertTrue(math.isinf(i2f(st.vgpr[0][1])))
    self.assertLess(i2f(st.vgpr[0][1]), 0)

  def test_v_cvt_f32_f16_nan(self):
    """V_CVT_F32_F16 converts f16 NaN to f32 NaN."""
    import math
    instructions = [
      s_mov_b32(s[0], 0x7e00),  # f16 quiet NaN
      v_mov_b32_e32(v[0], s[0]),
      v_cvt_f32_f16_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertTrue(math.isnan(i2f(st.vgpr[0][1])))

  def test_v_cvt_f32_f16_neg_zero(self):
    """V_CVT_F32_F16 preserves negative zero."""
    instructions = [
      s_mov_b32(s[0], 0x8000),  # f16 -0.0
      v_mov_b32_e32(v[0], s[0]),
      v_cvt_f32_f16_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 0x80000000)

  def test_v_cvt_f16_f32_overflow(self):
    """V_CVT_F16_F32 converts large f32 to f16 infinity."""
    instructions = [
      s_mov_b32(s[0], f2i(100000.0)),  # too large for f16
      v_mov_b32_e32(v[0], s[0]),
      v_cvt_f16_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    lo_bits = st.vgpr[0][1] & 0xffff
    self.assertEqual(lo_bits, 0x7c00)  # f16 +inf

  def test_v_cvt_f16_f32_underflow(self):
    """V_CVT_F16_F32 converts very small f32 to f16 zero or denormal."""
    instructions = [
      s_mov_b32(s[0], f2i(1e-10)),  # very small, below f16 range
      v_mov_b32_e32(v[0], s[0]),
      v_cvt_f16_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    lo_bits = st.vgpr[0][1] & 0xffff
    # Should be zero or very small denormal
    self.assertLess(lo_bits, 0x0400)  # Less than smallest normal f16


class TestExpEdgeCases(unittest.TestCase):
  """Additional edge cases for V_EXP_F32."""

  def test_v_exp_f32_zero(self):
    """V_EXP_F32 of 0.0 returns 1.0 (2^0 = 1)."""
    instructions = [
      v_mov_b32_e32(v[0], 0),
      v_exp_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), 1.0, places=5)

  def test_v_exp_f32_one(self):
    """V_EXP_F32 of 1.0 returns 2.0 (2^1 = 2)."""
    instructions = [
      v_mov_b32_e32(v[0], 1.0),
      v_exp_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), 2.0, places=5)

  def test_v_exp_f32_neg_one(self):
    """V_EXP_F32 of -1.0 returns 0.5 (2^-1 = 0.5)."""
    instructions = [
      v_mov_b32_e32(v[0], -1.0),
      v_exp_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), 0.5, places=5)

  def test_v_exp_f32_inf(self):
    """V_EXP_F32 of +inf returns +inf."""
    import math
    instructions = [
      s_mov_b32(s[0], 0x7f800000),  # +inf
      v_mov_b32_e32(v[0], s[0]),
      v_exp_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertTrue(math.isinf(i2f(st.vgpr[0][1])))
    self.assertGreater(i2f(st.vgpr[0][1]), 0)

  def test_v_exp_f32_neg_inf(self):
    """V_EXP_F32 of -inf returns 0."""
    instructions = [
      s_mov_b32(s[0], 0xff800000),  # -inf
      v_mov_b32_e32(v[0], s[0]),
      v_exp_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(i2f(st.vgpr[0][1]), 0.0)

  def test_v_exp_f32_nan(self):
    """V_EXP_F32 of NaN returns NaN."""
    import math
    instructions = [
      s_mov_b32(s[0], 0x7fc00000),  # quiet NaN
      v_mov_b32_e32(v[0], s[0]),
      v_exp_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertTrue(math.isnan(i2f(st.vgpr[0][1])))


class TestFloorEdgeCases(unittest.TestCase):
  """Additional edge cases for V_FLOOR_F32."""

  def test_v_floor_f32_negative(self):
    """V_FLOOR_F32 of -2.3 returns -3.0."""
    instructions = [
      s_mov_b32(s[0], f2i(-2.3)),
      v_mov_b32_e32(v[0], s[0]),
      v_floor_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), -3.0, places=5)

  def test_v_floor_f32_neg_zero(self):
    """V_FLOOR_F32 of -0.0 returns -0.0."""
    instructions = [
      s_mov_b32(s[0], 0x80000000),  # -0.0
      v_mov_b32_e32(v[0], s[0]),
      v_floor_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 0x80000000)

  def test_v_floor_f32_small_positive(self):
    """V_FLOOR_F32 of 0.9 returns 0.0."""
    instructions = [
      s_mov_b32(s[0], f2i(0.9)),
      v_mov_b32_e32(v[0], s[0]),
      v_floor_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(i2f(st.vgpr[0][1]), 0.0)

  def test_v_floor_f32_small_negative(self):
    """V_FLOOR_F32 of -0.9 returns -1.0."""
    instructions = [
      s_mov_b32(s[0], f2i(-0.9)),
      v_mov_b32_e32(v[0], s[0]),
      v_floor_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), -1.0, places=5)


class TestVop1F16HiHalf(unittest.TestCase):
  """Regression tests for VOP1 f16 hi-half source operand handling.

  For 16-bit VOP1 operations, when src0 is in the range v[128]+ (offset >= 384),
  the hardware reads from the high 16 bits of v[src0-128]. The emulator must
  extract bits [31:16] from the actual VGPR.
  """

  def test_v_cvt_f32_f16_src_hi_half(self):
    """V_CVT_F32_F16 with source from hi-half (v[128]+).

    When src0 >= v[128], it reads from the high 16 bits of v[src0-128].
    This is critical for global_load_d16_hi_b16 + v_cvt_f32_f16 patterns.

    Regression test for: VOP1 f16 src0 hi-half extraction bug.
    """
    instructions = [
      # v[0] = 0x4000_3c00: hi=f16(2.0), lo=f16(1.0)
      s_mov_b32(s[0], 0x40003c00),
      v_mov_b32_e32(v[0], s[0]),
      # v_cvt_f32_f16 v[1], v[128] (reads hi half of v[0])
      # Should convert f16(2.0) to f32(2.0)
      v_cvt_f32_f16_e32(v[1], v[128]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i2f(st.vgpr[0][1])
    self.assertAlmostEqual(result, 2.0, places=5, msg=f"Expected f32(2.0), got {result}")

  def test_v_cvt_f32_f16_src_lo_vs_hi(self):
    """V_CVT_F32_F16 comparing lo and hi half reads.

    v[0] has different values in lo and hi halves.
    v_cvt_f32_f16 v[1], v[0] should read lo (1.0)
    v_cvt_f32_f16 v[2], v[128] should read hi (2.0)

    Regression test for: VOP1 f16 src0 hi-half extraction bug.
    """
    instructions = [
      # v[0] = 0x4000_3c00: hi=f16(2.0), lo=f16(1.0)
      s_mov_b32(s[0], 0x40003c00),
      v_mov_b32_e32(v[0], s[0]),
      # Read from lo half
      v_cvt_f32_f16_e32(v[1], v[0]),
      # Read from hi half
      v_cvt_f32_f16_e32(v[2], v[128]),
    ]
    st = run_program(instructions, n_lanes=1)
    result_lo = i2f(st.vgpr[0][1])
    result_hi = i2f(st.vgpr[0][2])
    self.assertAlmostEqual(result_lo, 1.0, places=5, msg=f"Expected f32(1.0) from lo, got {result_lo}")
    self.assertAlmostEqual(result_hi, 2.0, places=5, msg=f"Expected f32(2.0) from hi, got {result_hi}")

  def test_v_cvt_i16_f16_src_hi_half(self):
    """V_CVT_I16_F16 with source from hi-half.

    Regression test for: VOP1 f16 src0 hi-half extraction bug.
    """
    instructions = [
      # v[0] = 0xc000_3c00: hi=f16(-2.0), lo=f16(1.0)
      s_mov_b32(s[0], 0xc0003c00),
      v_mov_b32_e32(v[0], s[0]),
      # v_cvt_i16_f16 v[1], v[128] (reads hi half of v[0])
      # Should convert f16(-2.0) to i16(-2)
      v_cvt_i16_f16_e32(v[1], v[128]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][1] & 0xffff
    expected = (-2) & 0xffff
    self.assertEqual(result, expected, f"Expected i16(-2)=0x{expected:04x}, got 0x{result:04x}")

  def test_v_mov_b16_src_hi_half(self):
    """V_MOV_B16 with source from hi-half.

    Regression test for: VOP1 f16 src0 hi-half extraction bug.
    """
    instructions = [
      # v[0] = 0xBEEF_DEAD: hi=0xBEEF, lo=0xDEAD
      s_mov_b32(s[0], 0xBEEFDEAD),
      v_mov_b32_e32(v[0], s[0]),
      # v[1] = 0x0000_0000 initially
      v_mov_b32_e32(v[1], 0),
      # v_mov_b16 v[1], v[128] (reads hi half of v[0])
      # Should move 0xBEEF to v[1].lo
      v_mov_b16_e32(v[1], v[128]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][1] & 0xffff
    self.assertEqual(result, 0xBEEF, f"Expected 0xBEEF from hi half, got 0x{result:04x}")


class TestReciprocalF16(unittest.TestCase):
  """Tests for V_RCP_F16 - reciprocal in half precision.

  The pcode uses a 16-bit float literal: D0.f16 = 16'1.0 / S0.f16
  This tests that the sized float literal (16'1.0) is correctly parsed.
  """

  def test_v_rcp_f16_one(self):
    """V_RCP_F16: 1/1.0 = 1.0"""
    import struct
    def f16_to_bits(f): return struct.unpack('<H', struct.pack('<e', f))[0]
    def bits_to_f16(b): return struct.unpack('<e', struct.pack('<H', b))[0]
    instructions = [
      # Load f16 1.0 into low 16 bits of v[0]
      v_mov_b32_e32(v[0], f16_to_bits(1.0)),
      v_rcp_f16_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = bits_to_f16(st.vgpr[0][1] & 0xFFFF)
    self.assertAlmostEqual(result, 1.0, places=2, msg="1/1.0 should be 1.0")

  def test_v_rcp_f16_two(self):
    """V_RCP_F16: 1/2.0 = 0.5"""
    import struct
    def f16_to_bits(f): return struct.unpack('<H', struct.pack('<e', f))[0]
    def bits_to_f16(b): return struct.unpack('<e', struct.pack('<H', b))[0]
    instructions = [
      v_mov_b32_e32(v[0], f16_to_bits(2.0)),
      v_rcp_f16_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = bits_to_f16(st.vgpr[0][1] & 0xFFFF)
    self.assertAlmostEqual(result, 0.5, places=2, msg="1/2.0 should be 0.5")

  def test_v_rcp_f16_four(self):
    """V_RCP_F16: 1/4.0 = 0.25"""
    import struct
    def f16_to_bits(f): return struct.unpack('<H', struct.pack('<e', f))[0]
    def bits_to_f16(b): return struct.unpack('<e', struct.pack('<H', b))[0]
    instructions = [
      v_mov_b32_e32(v[0], f16_to_bits(4.0)),
      v_rcp_f16_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = bits_to_f16(st.vgpr[0][1] & 0xFFFF)
    self.assertAlmostEqual(result, 0.25, places=2, msg="1/4.0 should be 0.25")


class TestCvtNormF16(unittest.TestCase):
  """Tests for V_CVT_NORM_I16_F16 and V_CVT_NORM_U16_F16."""

  def test_cvt_norm_i16_f16_positive(self):
    """V_CVT_NORM_I16_F16: f16 1.0 -> i16 max (32767)."""
    instructions = [
      s_mov_b32(s[0], f32_to_f16(1.0)),
      v_mov_b32_e32(v[0], s[0]),
      v_cvt_norm_i16_f16_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][1] & 0xffff
    self.assertEqual(result, 32767)

  def test_cvt_norm_i16_f16_negative(self):
    """V_CVT_NORM_I16_F16: f16 -1.0 -> i16 -32767 (0x8001)."""
    instructions = [
      s_mov_b32(s[0], f32_to_f16(-1.0)),
      v_mov_b32_e32(v[0], s[0]),
      v_cvt_norm_i16_f16_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][1] & 0xffff
    self.assertEqual(result, 0x8001)  # -32767, hardware uses symmetric range

  def test_cvt_norm_i16_f16_zero(self):
    """V_CVT_NORM_I16_F16: f16 0.0 -> i16 0."""
    instructions = [
      v_mov_b32_e32(v[0], 0),
      v_cvt_norm_i16_f16_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][1] & 0xffff
    self.assertEqual(result, 0)

  def test_cvt_norm_u16_f16_one(self):
    """V_CVT_NORM_U16_F16: f16 1.0 -> u16 max (65535)."""
    instructions = [
      s_mov_b32(s[0], f32_to_f16(1.0)),
      v_mov_b32_e32(v[0], s[0]),
      v_cvt_norm_u16_f16_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][1] & 0xffff
    self.assertEqual(result, 65535)

  def test_cvt_norm_u16_f16_half(self):
    """V_CVT_NORM_U16_F16: f16 0.5 -> u16 ~32768."""
    instructions = [
      s_mov_b32(s[0], f32_to_f16(0.5)),
      v_mov_b32_e32(v[0], s[0]),
      v_cvt_norm_u16_f16_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][1] & 0xffff
    self.assertAlmostEqual(result, 32768, delta=1)


class TestPermlane64(unittest.TestCase):
  """Tests for V_PERMLANE64_B32 instruction (wave64 cross-half swap)."""

  def test_v_permlane64_b32_is_nop_in_wave32(self):
    """V_PERMLANE64_B32 is a NOP in wave32 mode.

    Per AMD pcode: "if WAVE32 then s_nop(...) else ... endif"
    The emulator runs in wave32 mode, so this instruction should not modify registers.
    """
    instructions = [
      v_mov_b32_e32(v[0], 0xCAFEBABE),  # source
      v_mov_b32_e32(v[1], 0x12345678),  # dest (should be preserved)
      v_permlane64_b32_e32(v[1], v[0]),  # NOP in wave32
    ]
    st = run_program(instructions, n_lanes=1)
    # Dest register should be unchanged (NOP behavior in wave32)
    self.assertEqual(st.vgpr[0][1], 0x12345678)


if __name__ == '__main__':
  unittest.main()
