"""Tests for VOP2 instructions - two operand vector operations.

Includes: v_add_f32, v_mul_f32, v_and_b32, v_or_b32, v_xor_b32,
          v_lshrrev_b32, v_lshlrev_b32, v_fmac_f32, v_fmaak_f32, v_fmamk_f32,
          v_add_nc_u32, v_cndmask_b32, v_add_f16, v_mul_f16
"""
import unittest
from extra.assembly.amd.test.hw.helpers import *

class TestBasicArithmetic(unittest.TestCase):
  """Tests for basic arithmetic VOP2 instructions."""

  def test_v_add_f32(self):
    """V_ADD_F32 adds two floats."""
    instructions = [
      v_mov_b32_e32(v[0], 1.0),
      v_mov_b32_e32(v[1], 2.0),
      v_add_f32_e32(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][2]), 3.0, places=5)

  def test_v_mul_f32(self):
    """V_MUL_F32 multiplies two floats."""
    instructions = [
      v_mov_b32_e32(v[0], 2.0),
      v_mov_b32_e32(v[1], 4.0),
      v_mul_f32_e32(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][2]), 8.0, places=5)

  def test_v_fmac_f32(self):
    """V_FMAC_F32: d = d + a*b using inline constants."""
    instructions = [
      v_mov_b32_e32(v[0], 2.0),
      v_mov_b32_e32(v[1], 4.0),
      v_mov_b32_e32(v[2], 1.0),
      v_fmac_f32_e32(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][2]), 9.0, places=5)

  def test_v_fmaak_f32(self):
    """V_FMAAK_F32: d = a * b + K using inline constants."""
    instructions = [
      v_mov_b32_e32(v[0], 2.0),
      v_mov_b32_e32(v[1], 4.0),
      v_fmaak_f32_e32(v[2], v[0], v[1], literal=0x3f800000),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][2]), 9.0, places=5)

  def test_v_fmamk_f32_basic(self):
    """V_FMAMK_F32: d = a * K + b."""
    instructions = [
      v_mov_b32_e32(v[0], 2.0),
      v_mov_b32_e32(v[1], 1.0),
      v_fmamk_f32_e32(v[2], v[0], v[1], literal=0x40800000),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][2]), 9.0, places=5)

  def test_v_fmamk_f32_small_constant(self):
    """V_FMAMK_F32 with small constant."""
    instructions = [
      v_mov_b32_e32(v[0], 4.0),
      v_mov_b32_e32(v[1], 1.0),
      v_fmamk_f32_e32(v[2], v[0], v[1], literal=f2i(0.5)),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][2]), 3.0, places=5)


class TestBitManipulation(unittest.TestCase):
  """Tests for bit manipulation VOP2 instructions."""

  def test_v_and_b32(self):
    """V_AND_B32 bitwise and."""
    instructions = [
      s_mov_b32(s[0], 0xff),
      s_mov_b32(s[1], 0x0f),
      v_mov_b32_e32(v[0], s[0]),
      v_and_b32_e32(v[1], s[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 0x0f)

  def test_v_and_b32_quadrant(self):
    """V_AND_B32 for quadrant extraction (n & 3)."""
    instructions = [
      s_mov_b32(s[0], 15915),
      v_mov_b32_e32(v[0], s[0]),
      v_and_b32_e32(v[1], 3, v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 15915 & 3)

  def test_v_lshrrev_b32(self):
    """V_LSHRREV_B32 logical shift right."""
    instructions = [
      s_mov_b32(s[0], 0xff00),
      v_mov_b32_e32(v[0], s[0]),
      v_lshrrev_b32_e32(v[1], 8, v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 0xff)

  def test_v_lshlrev_b32(self):
    """V_LSHLREV_B32 logical shift left."""
    instructions = [
      s_mov_b32(s[0], 0xff),
      v_mov_b32_e32(v[0], s[0]),
      v_lshlrev_b32_e32(v[1], 8, v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 0xff00)

  def test_v_xor_b32(self):
    """V_XOR_B32 bitwise xor (used in sin for sign)."""
    instructions = [
      s_mov_b32(s[0], 0x80000000),
      s_mov_b32(s[1], f2i(1.0)),
      v_mov_b32_e32(v[0], s[1]),
      v_xor_b32_e32(v[1], s[0], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), -1.0, places=5)

  def test_v_xor_b32_sign_flip(self):
    """V_XOR_B32 for sign flip pattern."""
    instructions = [
      s_mov_b32(s[0], 0x80000000),
      v_mov_b32_e32(v[0], -2.0),
      v_xor_b32_e32(v[1], s[0], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), 2.0, places=5)


class TestSpecialValues(unittest.TestCase):
  """Tests for special float values - inf, nan, zero handling."""

  def test_v_mul_f32_zero_times_inf(self):
    """V_MUL_F32: 0 * inf = NaN."""
    import math
    instructions = [
      v_mov_b32_e32(v[0], 0),
      s_mov_b32(s[0], 0x7f800000),
      v_mov_b32_e32(v[1], s[0]),
      v_mul_f32_e32(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertTrue(math.isnan(i2f(st.vgpr[0][2])))

  def test_v_add_f32_inf_minus_inf(self):
    """V_ADD_F32: inf + (-inf) = NaN."""
    import math
    instructions = [
      s_mov_b32(s[0], 0x7f800000),
      s_mov_b32(s[1], 0xff800000),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_add_f32_e32(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertTrue(math.isnan(i2f(st.vgpr[0][2])))


class TestF16Ops(unittest.TestCase):
  """Tests for 16-bit VOP2 operations."""

  def test_v_add_f16_basic(self):
    """V_ADD_F16 adds two f16 values."""
    instructions = [
      s_mov_b32(s[0], 0x3c00),  # f16 1.0
      s_mov_b32(s[1], 0x4000),  # f16 2.0
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_add_f16_e32(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][2] & 0xffff
    self.assertEqual(result, 0x4200, f"Expected 0x4200 (f16 3.0), got 0x{result:04x}")

  def test_v_add_f16_negative(self):
    """V_ADD_F16 with negative values."""
    instructions = [
      s_mov_b32(s[0], 0x3c00),  # f16 1.0
      s_mov_b32(s[1], 0xc000),  # f16 -2.0
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_add_f16_e32(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][2] & 0xffff
    self.assertEqual(result, 0xbc00, f"Expected 0xbc00 (f16 -1.0), got 0x{result:04x}")

  def test_v_mul_f16_basic(self):
    """V_MUL_F16 multiplies two f16 values."""
    instructions = [
      s_mov_b32(s[0], 0x4000),  # f16 2.0
      s_mov_b32(s[1], 0x4200),  # f16 3.0
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_mul_f16_e32(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][2] & 0xffff
    self.assertEqual(result, 0x4600, f"Expected 0x4600 (f16 6.0), got 0x{result:04x}")

  def test_v_mul_f16_by_zero(self):
    """V_MUL_F16 by zero."""
    instructions = [
      s_mov_b32(s[0], 0x4000),  # f16 2.0
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], 0),
      v_mul_f16_e32(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][2] & 0xffff
    self.assertEqual(result, 0x0000, f"Expected 0x0000 (f16 0.0), got 0x{result:04x}")

  def test_v_fmac_f16_basic(self):
    """V_FMAC_F16: d = d + a*b."""
    instructions = [
      s_mov_b32(s[0], 0x4000),  # f16 2.0
      s_mov_b32(s[1], 0x4200),  # f16 3.0
      s_mov_b32(s[2], 0x3c00),  # f16 1.0
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_mov_b32_e32(v[2], s[2]),
      v_fmac_f16_e32(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][2] & 0xffff
    # 2.0 * 3.0 + 1.0 = 7.0, f16 7.0 = 0x4700
    self.assertEqual(result, 0x4700, f"Expected 0x4700 (f16 7.0), got 0x{result:04x}")

  def test_v_max_f16_basic(self):
    """V_MAX_F16 returns the maximum of two f16 values."""
    instructions = [
      s_mov_b32(s[0], 0x3c00),  # f16 1.0
      s_mov_b32(s[1], 0x4000),  # f16 2.0
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_max_f16_e32(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][2] & 0xffff
    self.assertEqual(result, 0x4000, f"Expected 0x4000 (f16 2.0), got 0x{result:04x}")

  def test_v_min_f16_basic(self):
    """V_MIN_F16 returns the minimum of two f16 values."""
    instructions = [
      s_mov_b32(s[0], 0x3c00),  # f16 1.0
      s_mov_b32(s[1], 0x4000),  # f16 2.0
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_min_f16_e32(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][2] & 0xffff
    self.assertEqual(result, 0x3c00, f"Expected 0x3c00 (f16 1.0), got 0x{result:04x}")

  def test_v_fmaak_f16_basic(self):
    """V_FMAAK_F16: d = a * b + K."""
    instructions = [
      s_mov_b32(s[0], 0x4000),  # f16 2.0
      s_mov_b32(s[1], 0x4200),  # f16 3.0
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_fmaak_f16_e32(v[2], v[0], v[1], literal=0x3c00),  # + f16 1.0
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][2] & 0xffff
    # 2.0 * 3.0 + 1.0 = 7.0, f16 7.0 = 0x4700
    self.assertEqual(result, 0x4700, f"Expected 0x4700 (f16 7.0), got 0x{result:04x}")


class TestHiHalfOps(unittest.TestCase):
  """Tests for VOP2 16-bit operations with hi-half operands."""

  def test_v_add_f16_src0_hi_fold(self):
    """V_ADD_F16 with src0 hi-half fold (same register, different halves)."""
    instructions = [
      s_mov_b32(s[0], 0x40003c00),  # lo=f16(1.0), hi=f16(2.0)
      v_mov_b32_e32(v[0], s[0]),
      VOP3(VOP3Op.V_ADD_F16, vdst=v[1], src0=v[0], src1=v[0], opsel=0b0001),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][1] & 0xffff
    self.assertEqual(result, 0x4200, f"Expected f16(3.0)=0x4200, got 0x{result:04x}")

  def test_v_add_f16_src0_hi_different_reg(self):
    """V_ADD_F16 with src0 hi-half from different register."""
    instructions = [
      s_mov_b32(s[0], 0x40000000),  # hi=f16(2.0), lo=0
      s_mov_b32(s[1], 0x00003c00),  # hi=0, lo=f16(1.0)
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      VOP3(VOP3Op.V_ADD_F16, vdst=v[2], src0=v[0], src1=v[1], opsel=0b0001),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][2] & 0xffff
    self.assertEqual(result, 0x4200, f"Expected f16(3.0)=0x4200, got 0x{result:04x}")

  def test_v_mul_f16_src0_hi(self):
    """V_MUL_F16 with src0 from high half."""
    instructions = [
      s_mov_b32(s[0], 0x40000000),  # hi=f16(2.0), lo=0
      s_mov_b32(s[1], 0x00004200),  # hi=0, lo=f16(3.0)
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      VOP3(VOP3Op.V_MUL_F16, vdst=v[2], src0=v[0], src1=v[1], opsel=0b0001),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][2] & 0xffff
    self.assertEqual(result, 0x4600, f"Expected f16(6.0)=0x4600, got 0x{result:04x}")

  def test_v_mul_f16_hi_half(self):
    """V_MUL_F16 reading from high half."""
    instructions = [
      s_mov_b32(s[0], 0x40003c00),  # lo=1.0, hi=2.0
      v_mov_b32_e32(v[0], s[0]),
      VOP3(VOP3Op.V_MUL_F16, vdst=v[1], src0=v[0], src1=v[0], opsel=0b0011),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][1] & 0xffff
    self.assertEqual(result, 0x4400, f"Expected f16(4.0)=0x4400, got 0x{result:04x}")

  def test_v_fma_f16_hi_dest(self):
    """V_FMA_F16 writing to high half with opsel.

    Uses V_FMA_F16 (not V_FMAC_F16) because it has explicit src2 operand
    which makes opsel handling clearer.
    """
    instructions = [
      s_mov_b32(s[0], 0x3c000000),  # hi=f16(1.0), lo=0
      s_mov_b32(s[1], 0x4000),      # f16(2.0) in lo
      s_mov_b32(s[2], 0x4200),      # f16(3.0) in lo
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_mov_b32_e32(v[2], s[2]),
      # V_FMA_F16: dst = src0 * src1 + src2
      # opsel=0b1100: bit2=src2 hi, bit3=dst hi
      # So: v[0].hi = v[1].lo * v[2].lo + v[0].hi = 2.0 * 3.0 + 1.0 = 7.0
      VOP3(VOP3Op.V_FMA_F16, vdst=v[0], src0=v[1], src1=v[2], src2=v[0], opsel=0b1100),
    ]
    st = run_program(instructions, n_lanes=1)
    hi = (st.vgpr[0][0] >> 16) & 0xffff
    # 2.0 * 3.0 + 1.0 = 7.0, f16 7.0 = 0x4700
    self.assertEqual(hi, 0x4700, f"Expected f16(7.0)=0x4700 in hi, got 0x{hi:04x}")

  def test_v_add_f16_multilane(self):
    """V_ADD_F16 with multiple lanes."""
    instructions = [
      s_mov_b32(s[0], 0x3c00),  # f16 1.0
      s_mov_b32(s[1], 0x4000),  # f16 2.0
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_add_f16_e32(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=4)
    for lane in range(4):
      result = st.vgpr[lane][2] & 0xffff
      self.assertEqual(result, 0x4200, f"Lane {lane}: expected 0x4200, got 0x{result:04x}")


class TestVop2F16HiHalf(unittest.TestCase):
  """Regression tests for VOP2 f16 hi-half operand handling.

  These test the bugs where:
  1. VOP2 vsrc1 >= 384 (v[128]+) wasn't extracting hi 16 bits
  2. VOP2 vdst >= 384 (v[128]+) wasn't preserving lo 16 bits
  """

  def test_v_add_f16_e32_vsrc1_hi_half(self):
    """V_ADD_F16_E32 with vsrc1 from hi-half (v[128]+).

    When vsrc1 >= 384 (representing v[128]+), the hardware reads from the hi 16 bits
    of v[vsrc1-128]. The emulator must extract bits [31:16] from the actual VGPR.

    Regression test for: VOP2 f16 vsrc1 hi-half extraction bug.
    """
    instructions = [
      # v[0] = 0x4000_3c00: hi=f16(2.0), lo=f16(1.0)
      s_mov_b32(s[0], 0x40003c00),
      v_mov_b32_e32(v[0], s[0]),
      # v_add_f16_e32 v[1], v[0], v[128]  (vsrc1=v[128] reads hi of v[0])
      # In VOP2 encoding, vsrc1=384 means v[128], which maps to v[0].hi
      # v[1] = v[0].lo + v[0].hi = 1.0 + 2.0 = 3.0
      VOP2(VOP2Op.V_ADD_F16, vdst=v[1], src0=v[0], vsrc1=v[128]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][1] & 0xffff
    # 1.0 + 2.0 = 3.0, f16 3.0 = 0x4200
    self.assertEqual(result, 0x4200, f"Expected f16(3.0)=0x4200, got 0x{result:04x}")

  def test_v_mul_f16_e32_vsrc1_hi_half(self):
    """V_MUL_F16_E32 with vsrc1 from hi-half.

    Regression test for: VOP2 f16 vsrc1 hi-half extraction bug.
    """
    instructions = [
      # v[0] = 0x4200_4000: hi=f16(3.0), lo=f16(2.0)
      s_mov_b32(s[0], 0x42004000),
      v_mov_b32_e32(v[0], s[0]),
      # v_mul_f16_e32 v[1], v[0], v[128]  (vsrc1=v[128] reads hi of v[0])
      # v[1] = v[0].lo * v[0].hi = 2.0 * 3.0 = 6.0
      VOP2(VOP2Op.V_MUL_F16, vdst=v[1], src0=v[0], vsrc1=v[128]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][1] & 0xffff
    # 2.0 * 3.0 = 6.0, f16 6.0 = 0x4600
    self.assertEqual(result, 0x4600, f"Expected f16(6.0)=0x4600, got 0x{result:04x}")

  def test_v_add_f16_e32_vdst_hi_half(self):
    """V_ADD_F16_E32 writing to hi-half destination (v[128]+).

    When vdst >= 384 (representing v[128]+), the hardware writes to bits [31:16]
    of v[vdst-128] while preserving bits [15:0]. The emulator must merge the result.

    Regression test for: VOP2 f16 vdst hi-half write bug.
    """
    instructions = [
      # v[0] = 0x0000_BEEF: lo has marker value
      s_mov_b32(s[0], 0x0000BEEF),
      v_mov_b32_e32(v[0], s[0]),
      # v[1] = f16(1.0), v[2] = f16(2.0)
      s_mov_b32(s[1], 0x3c00),
      s_mov_b32(s[2], 0x4000),
      v_mov_b32_e32(v[1], s[1]),
      v_mov_b32_e32(v[2], s[2]),
      # v_add_f16_e32 v[128], v[1], v[2]  (vdst=v[128] writes hi of v[0])
      # v[0].hi = 1.0 + 2.0 = 3.0, v[0].lo preserved = 0xBEEF
      VOP2(VOP2Op.V_ADD_F16, vdst=v[128], src0=v[1], vsrc1=v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    hi = (st.vgpr[0][0] >> 16) & 0xffff
    lo = st.vgpr[0][0] & 0xffff
    # hi = 3.0 = 0x4200, lo preserved = 0xBEEF
    self.assertEqual(hi, 0x4200, f"Expected hi=f16(3.0)=0x4200, got 0x{hi:04x}")
    self.assertEqual(lo, 0xBEEF, f"Expected lo preserved=0xBEEF, got 0x{lo:04x}")

  def test_v_mul_f16_e32_vdst_hi_half(self):
    """V_MUL_F16_E32 writing to hi-half destination.

    Regression test for: VOP2 f16 vdst hi-half write bug.
    """
    instructions = [
      # v[0] = 0x0000_DEAD: lo has marker value
      s_mov_b32(s[0], 0x0000DEAD),
      v_mov_b32_e32(v[0], s[0]),
      # v[1] = f16(2.0), v[2] = f16(4.0)
      s_mov_b32(s[1], 0x4000),
      s_mov_b32(s[2], 0x4400),
      v_mov_b32_e32(v[1], s[1]),
      v_mov_b32_e32(v[2], s[2]),
      # v_mul_f16_e32 v[128], v[1], v[2]  (vdst=v[128] writes hi of v[0])
      # v[0].hi = 2.0 * 4.0 = 8.0, v[0].lo preserved = 0xDEAD
      VOP2(VOP2Op.V_MUL_F16, vdst=v[128], src0=v[1], vsrc1=v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    hi = (st.vgpr[0][0] >> 16) & 0xffff
    lo = st.vgpr[0][0] & 0xffff
    # hi = 8.0 = 0x4800, lo preserved = 0xDEAD
    self.assertEqual(hi, 0x4800, f"Expected hi=f16(8.0)=0x4800, got 0x{hi:04x}")
    self.assertEqual(lo, 0xDEAD, f"Expected lo preserved=0xDEAD, got 0x{lo:04x}")

  def test_v_add_f16_e32_both_hi_half(self):
    """V_ADD_F16_E32 with both vsrc1 and vdst as hi-half (different underlying regs).

    Tests the combination of both fixes: reading vsrc1 from hi-half AND
    writing result to hi-half destination, using different underlying VGPRs.

    Regression test for: VOP2 f16 hi-half bugs (combined).
    """
    instructions = [
      # v[0] = 0x4000_xxxx: hi=f16(2.0) for vsrc1
      s_mov_b32(s[0], 0x40000000),
      v_mov_b32_e32(v[0], s[0]),
      # v[1] = 0x0000_3c00: lo=f16(1.0) for src0
      s_mov_b32(s[1], 0x00003c00),
      v_mov_b32_e32(v[1], s[1]),
      # v[2] = 0x0000_CAFE: lo=marker for vdst preservation
      s_mov_b32(s[2], 0x0000CAFE),
      v_mov_b32_e32(v[2], s[2]),
      # v_add_f16_e32 v[130], v[1], v[128]
      # src0 = v[1].lo = 1.0
      # vsrc1 = v[128] reads v[0].hi = 2.0
      # result = 1.0 + 2.0 = 3.0
      # vdst = v[130] writes to v[2].hi, preserving v[2].lo
      VOP2(VOP2Op.V_ADD_F16, vdst=v[130], src0=v[1], vsrc1=v[128]),
    ]
    st = run_program(instructions, n_lanes=1)
    hi = (st.vgpr[0][2] >> 16) & 0xffff
    lo = st.vgpr[0][2] & 0xffff
    # hi = 3.0 = 0x4200, lo preserved = 0xCAFE
    self.assertEqual(hi, 0x4200, f"Expected hi=f16(3.0)=0x4200, got 0x{hi:04x}")
    self.assertEqual(lo, 0xCAFE, f"Expected lo preserved=0xCAFE, got 0x{lo:04x}")

  def test_v_fmac_f16_e32_vsrc1_hi_half(self):
    """V_FMAC_F16_E32 with vsrc1 from hi-half.

    V_FMAC_F16: vdst = vdst + src0 * vsrc1

    Regression test for: VOP2 f16 vsrc1 hi-half extraction bug.
    """
    instructions = [
      # v[0] = 0x4000_3c00: hi=f16(2.0), lo=f16(1.0)
      s_mov_b32(s[0], 0x40003c00),
      v_mov_b32_e32(v[0], s[0]),
      # v[1] = f16(3.0) = 0x4200
      s_mov_b32(s[1], 0x4200),
      v_mov_b32_e32(v[1], s[1]),
      # v_fmac_f16_e32 v[1], v[0], v[128]
      # vdst = v[1] = 3.0 + v[0].lo * v[0].hi = 3.0 + 1.0 * 2.0 = 5.0
      VOP2(VOP2Op.V_FMAC_F16, vdst=v[1], src0=v[0], vsrc1=v[128]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][1] & 0xffff
    # 3.0 + 1.0 * 2.0 = 5.0, f16 5.0 = 0x4500
    self.assertEqual(result, 0x4500, f"Expected f16(5.0)=0x4500, got 0x{result:04x}")

  def test_v_fmac_f16_e32_vdst_hi_half(self):
    """V_FMAC_F16_E32 writing to hi-half destination.

    V_FMAC_F16: vdst.h = vdst.h + src0 * vsrc1

    When vdst is v[128]+, the accumulator D0 must also read from the hi-half.
    This tests the bug where D0 was read from lo-half instead of hi-half.

    Regression test for: VOP2 FMAC hi-half D0 accumulator read bug.
    """
    instructions = [
      # v[0] = 0x3800_DEAD: hi=f16(0.5), lo=marker (0xDEAD)
      s_mov_b32(s[0], 0x3800DEAD),
      v_mov_b32_e32(v[0], s[0]),
      # v[1] = f16(2.0) = 0x4000
      s_mov_b32(s[1], 0x4000),
      v_mov_b32_e32(v[1], s[1]),
      # v[2] = f16(3.0) = 0x4200
      s_mov_b32(s[2], 0x4200),
      v_mov_b32_e32(v[2], s[2]),
      # v_fmac_f16_e32 v[128], v[1], v[2]
      # vdst = v[128] means v[0].hi
      # D0 = v[0].hi = 0.5
      # result = D0 + src0 * vsrc1 = 0.5 + 2.0 * 3.0 = 6.5
      # v[0].hi = 6.5, v[0].lo preserved = 0xDEAD
      VOP2(VOP2Op.V_FMAC_F16, vdst=v[128], src0=v[1], vsrc1=v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    hi = (st.vgpr[0][0] >> 16) & 0xffff
    lo = st.vgpr[0][0] & 0xffff
    # hi = 6.5 = 0x4680, lo preserved = 0xDEAD
    self.assertEqual(hi, 0x4680, f"Expected hi=f16(6.5)=0x4680, got 0x{hi:04x}")
    self.assertEqual(lo, 0xDEAD, f"Expected lo preserved=0xDEAD, got 0x{lo:04x}")

  def test_v_mul_f16_e32_src0_hi_half(self):
    """V_MUL_F16_E32 with src0 from hi-half (src0 >= v[128]).

    When src0 >= 384 (representing v[128]+), the hardware reads from the hi 16 bits
    of v[src0-128]. The emulator must extract bits [31:16] from the actual VGPR.

    Regression test for: VOP2 f16 src0 hi-half extraction bug.
    """
    instructions = [
      # v[0] = 0x4000_3c00: hi=f16(2.0), lo=f16(1.0)
      s_mov_b32(s[0], 0x40003c00),
      v_mov_b32_e32(v[0], s[0]),
      # v[1] = f16(3.0) = 0x4200
      s_mov_b32(s[1], 0x4200),
      v_mov_b32_e32(v[1], s[1]),
      # v_mul_f16_e32 v[2], v[128], v[1]
      # src0 = v[128] reads from v[0].hi = 2.0
      # result = 2.0 * 3.0 = 6.0
      VOP2(VOP2Op.V_MUL_F16, vdst=v[2], src0=v[128], vsrc1=v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][2] & 0xffff
    # 2.0 * 3.0 = 6.0, f16 6.0 = 0x4600
    self.assertEqual(result, 0x4600, f"Expected f16(6.0)=0x4600, got 0x{result:04x}")

  def test_v_add_f16_e32_src0_hi_half(self):
    """V_ADD_F16_E32 with src0 from hi-half (src0 >= v[128]).

    Regression test for: VOP2 f16 src0 hi-half extraction bug.
    """
    instructions = [
      # v[0] = 0x4000_3c00: hi=f16(2.0), lo=f16(1.0)
      s_mov_b32(s[0], 0x40003c00),
      v_mov_b32_e32(v[0], s[0]),
      # v[1] = f16(5.0) = 0x4500
      s_mov_b32(s[1], 0x4500),
      v_mov_b32_e32(v[1], s[1]),
      # v_add_f16_e32 v[2], v[128], v[1]
      # src0 = v[128] reads from v[0].hi = 2.0
      # result = 2.0 + 5.0 = 7.0
      VOP2(VOP2Op.V_ADD_F16, vdst=v[2], src0=v[128], vsrc1=v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][2] & 0xffff
    # 2.0 + 5.0 = 7.0, f16 7.0 = 0x4700
    self.assertEqual(result, 0x4700, f"Expected f16(7.0)=0x4700, got 0x{result:04x}")


class TestF16InlineConstants(unittest.TestCase):
  """Regression tests for VOP2 F16 inline float constants.

  For 16-bit VOP2 operations (v_add_f16, v_mul_f16, etc.), inline float constants
  like 1.0, 2.0 must use F16 encoding (0x3c00, 0x4000) not F32 encoding (0x3f800000).

  The emulator's rsrc() function needs bits=16 to select F16_INLINE constants.

  Regression test for: VOP2 16-bit inline constant using F32 instead of F16.
  """

  def test_v_add_f16_inline_constant_1_0(self):
    """V_ADD_F16_E32 with inline constant 1.0 should use F16 encoding."""
    instructions = [
      s_mov_b32(s[0], 0x3c00),  # f16 1.0
      v_mov_b32_e32(v[0], s[0]),
      # v_add_f16_e32 v[1], 1.0, v[0]  -- 1.0 must be F16 0x3c00, not F32 0x3f800000
      v_add_f16_e32(v[1], 1.0, v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][1] & 0xFFFF
    # 1.0 + 1.0 = 2.0, f16 2.0 = 0x4000
    self.assertEqual(result, 0x4000, f"Expected f16(2.0)=0x4000, got 0x{result:04x}")

  def test_v_add_f16_inline_constant_2_0(self):
    """V_ADD_F16_E32 with inline constant 2.0."""
    instructions = [
      s_mov_b32(s[0], 0x4200),  # f16 3.0
      v_mov_b32_e32(v[0], s[0]),
      v_add_f16_e32(v[1], 2.0, v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][1] & 0xFFFF
    # 2.0 + 3.0 = 5.0, f16 5.0 = 0x4500
    self.assertEqual(result, 0x4500, f"Expected f16(5.0)=0x4500, got 0x{result:04x}")

  def test_v_mul_f16_inline_constant(self):
    """V_MUL_F16_E32 with inline constant 2.0."""
    instructions = [
      s_mov_b32(s[0], 0x4200),  # f16 3.0
      v_mov_b32_e32(v[0], s[0]),
      v_mul_f16_e32(v[1], 2.0, v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][1] & 0xFFFF
    # 2.0 * 3.0 = 6.0, f16 6.0 = 0x4600
    self.assertEqual(result, 0x4600, f"Expected f16(6.0)=0x4600, got 0x{result:04x}")


class TestCndmask(unittest.TestCase):
  """Tests for V_CNDMASK_B32 and V_CNDMASK_B16."""

  def test_v_cndmask_b16_select_src0(self):
    """V_CNDMASK_B16 selects src0 when VCC bit is 0."""
    instructions = [
      s_mov_b32(VCC_LO, 0),  # VCC = 0
      s_mov_b32(s[0], 0x3c00),  # f16 1.0
      s_mov_b32(s[1], 0x4000),  # f16 2.0
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_cndmask_b16(v[2], v[0], v[1], VCC),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][2] & 0xffff
    self.assertEqual(result, 0x3c00, f"Expected src0=0x3c00, got 0x{result:04x}")

  def test_v_cndmask_b16_select_src1(self):
    """V_CNDMASK_B16 selects src1 when VCC bit is 1."""
    instructions = [
      s_mov_b32(VCC_LO, 1),  # VCC = 1
      s_mov_b32(s[0], 0x3c00),  # f16 1.0
      s_mov_b32(s[1], 0x4000),  # f16 2.0
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_cndmask_b16(v[2], v[0], v[1], VCC),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][2] & 0xffff
    self.assertEqual(result, 0x4000, f"Expected src1=0x4000, got 0x{result:04x}")

  def test_v_cndmask_b16_write_hi(self):
    """V_CNDMASK_B16 can write to high 16 bits with opsel."""
    instructions = [
      s_mov_b32(s[0], 0x3c003800),  # src0: hi=1.0, lo=0.5
      v_mov_b32_e32(v[0], s[0]),
      s_mov_b32(s[1], 0x4000c000),  # src1: hi=2.0, lo=-2.0
      v_mov_b32_e32(v[1], s[1]),
      s_mov_b32(s[2], 0xDEAD0000),  # v2 initial: hi=0xDEAD, lo=0
      v_mov_b32_e32(v[2], s[2]),
      s_mov_b32(VCC_LO, 0),  # vcc = 0, select src0
      # opsel=0b1011: bit0=src0 hi, bit1=src1 hi, bit3=dst hi
      VOP3(VOP3Op.V_CNDMASK_B16, vdst=v[2], src0=v[0], src1=v[1], src2=SrcEnum.VCC_LO, opsel=0b1011),
    ]
    st = run_program(instructions, n_lanes=1)
    hi = (st.vgpr[0][2] >> 16) & 0xffff
    lo = st.vgpr[0][2] & 0xffff
    # vcc=0 selects src0.h = 1.0 = 0x3c00, writes to hi
    self.assertEqual(hi, 0x3c00, f"Expected hi=0x3c00 (1.0), got 0x{hi:04x}")
    self.assertEqual(lo, 0x0000, f"Expected lo preserved as 0, got 0x{lo:04x}")


class TestSpecialFloatValues(unittest.TestCase):
  """Tests for special float value handling in VOP2 instructions."""

  def test_neg_zero_add(self):
    """-0.0 + 0.0 = +0.0 (IEEE 754)."""
    neg_zero = 0x80000000
    instructions = [
      s_mov_b32(s[0], neg_zero),
      v_mov_b32_e32(v[0], s[0]),
      v_add_f32_e32(v[1], 0.0, v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 0x00000000, "Should be +0.0")

  def test_neg_zero_mul(self):
    """-0.0 * -1.0 = +0.0."""
    neg_zero = 0x80000000
    instructions = [
      s_mov_b32(s[0], neg_zero),
      v_mov_b32_e32(v[0], s[0]),
      v_mul_f32_e32(v[1], -1.0, v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 0x00000000, "Should be +0.0")

  def test_inf_minus_inf(self):
    """+inf - inf = NaN."""
    import math
    pos_inf = 0x7f800000
    neg_inf = 0xff800000
    instructions = [
      s_mov_b32(s[0], pos_inf),
      s_mov_b32(s[1], neg_inf),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_sub_f32_e32(v[2], v[0], v[1]),  # inf - (-inf) = inf
      v_add_f32_e32(v[3], v[0], v[1]),  # inf + (-inf) = NaN
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][2], pos_inf, "inf - (-inf) = inf")
    self.assertTrue(math.isnan(i2f(st.vgpr[0][3])), "inf + (-inf) = NaN")

  def test_denormal_f32_mul_ftz(self):
    """Denormal * normal - RDNA3 flushes denormals to zero (FTZ mode)."""
    smallest_denorm = 0x00000001  # Smallest positive denormal
    instructions = [
      s_mov_b32(s[0], smallest_denorm),
      v_mov_b32_e32(v[0], s[0]),
      v_mul_f32_e32(v[1], 2.0, v[0]),  # Denormal input gets flushed to 0
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 0x00000000)


class TestCarryOps(unittest.TestCase):
  """Tests for VOP2 carry instructions (v_add_co_ci_u32, v_sub_co_ci_u32, v_subrev_co_ci_u32)."""

  def test_v_subrev_co_ci_u32_no_borrow(self):
    """V_SUBREV_CO_CI_U32: D0 = S1 - S0 - VCC_IN, when VCC_IN=0."""
    instructions = [
      s_mov_b32(VCC_LO, 0),  # VCC = 0 (no borrow in)
      v_mov_b32_e32(v[0], 5),  # S0 = 5
      v_mov_b32_e32(v[1], 10),  # S1 = 10
      v_subrev_co_ci_u32_e32(v[2], v[0], v[1]),  # D0 = 10 - 5 - 0 = 5
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][2], 5)
    self.assertEqual(st.vcc, 0)  # No borrow out

  def test_v_subrev_co_ci_u32_with_borrow(self):
    """V_SUBREV_CO_CI_U32: D0 = S1 - S0 - VCC_IN, when VCC_IN=1."""
    instructions = [
      s_mov_b32(VCC_LO, 1),  # VCC = 1 (borrow in)
      v_mov_b32_e32(v[0], 5),  # S0 = 5
      v_mov_b32_e32(v[1], 10),  # S1 = 10
      v_subrev_co_ci_u32_e32(v[2], v[0], v[1]),  # D0 = 10 - 5 - 1 = 4
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][2], 4)
    self.assertEqual(st.vcc, 0)  # No borrow out

  def test_v_subrev_co_ci_u32_generates_borrow(self):
    """V_SUBREV_CO_CI_U32: generates borrow when S0 + VCC_IN > S1."""
    instructions = [
      s_mov_b32(VCC_LO, 0),  # VCC = 0
      v_mov_b32_e32(v[0], 10),  # S0 = 10
      v_mov_b32_e32(v[1], 5),  # S1 = 5
      v_subrev_co_ci_u32_e32(v[2], v[0], v[1]),  # D0 = 5 - 10 - 0 = -5 (underflow)
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][2], 0xFFFFFFFB)  # -5 as unsigned
    self.assertEqual(st.vcc, 1)  # Borrow out

  def test_v_add_co_ci_u32_no_carry(self):
    """V_ADD_CO_CI_U32: D0 = S0 + S1 + VCC_IN, when VCC_IN=0."""
    instructions = [
      s_mov_b32(VCC_LO, 0),  # VCC = 0 (no carry in)
      v_mov_b32_e32(v[0], 5),  # S0 = 5
      v_mov_b32_e32(v[1], 10),  # S1 = 10
      v_add_co_ci_u32_e32(v[2], v[0], v[1]),  # D0 = 5 + 10 + 0 = 15
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][2], 15)
    self.assertEqual(st.vcc, 0)  # No carry out

  def test_v_add_co_ci_u32_with_carry(self):
    """V_ADD_CO_CI_U32: D0 = S0 + S1 + VCC_IN, when VCC_IN=1."""
    instructions = [
      s_mov_b32(VCC_LO, 1),  # VCC = 1 (carry in)
      v_mov_b32_e32(v[0], 5),  # S0 = 5
      v_mov_b32_e32(v[1], 10),  # S1 = 10
      v_add_co_ci_u32_e32(v[2], v[0], v[1]),  # D0 = 5 + 10 + 1 = 16
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][2], 16)
    self.assertEqual(st.vcc, 0)  # No carry out

  def test_v_add_co_ci_u32_generates_carry(self):
    """V_ADD_CO_CI_U32: generates carry when overflow occurs."""
    instructions = [
      s_mov_b32(VCC_LO, 1),  # VCC = 1 (carry in)
      s_mov_b32(s[0], 0xFFFFFFFF),  # max u32
      v_mov_b32_e32(v[0], s[0]),  # S0 = 0xFFFFFFFF
      v_mov_b32_e32(v[1], 0),  # S1 = 0
      v_add_co_ci_u32_e32(v[2], v[0], v[1]),  # D0 = 0xFFFFFFFF + 0 + 1 = 0 (overflow)
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][2], 0)  # Overflowed to 0
    self.assertEqual(st.vcc, 1)  # Carry out

  def test_v_add_co_ci_u32_clears_carry(self):
    """V_ADD_CO_CI_U32: VCC must be updated even when no carry is generated.

    This tests the case where VCC=1 going in (carry-in consumed) but the addition
    does not overflow, so VCC must be cleared to 0.

    Regression test for: VCC not being written by v_add_co_ci_u32_e32.
    """
    instructions = [
      s_mov_b32(VCC_LO, 1),  # VCC = 1 (carry in)
      v_mov_b32_e32(v[0], 1),  # S0 = 1
      v_mov_b32_e32(v[1], 1),  # S1 = 1
      v_add_co_ci_u32_e32(v[2], v[0], v[1]),  # D0 = 1 + 1 + 1 = 3 (no overflow)
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][2], 3)  # 1 + 1 + 1 = 3
    self.assertEqual(st.vcc, 0)  # No carry out - VCC must be cleared

  def test_v_add_co_ci_u32_multilane_clears_vcc(self):
    """V_ADD_CO_CI_U32 with multiple lanes: VCC bits must be updated per-lane.

    When VCC has multiple bits set (one per active lane), and the addition doesn't
    overflow for any lane, all VCC bits must be cleared.

    Regression test for: VCC not being written by v_add_co_ci_u32_e32 in multi-lane case.
    """
    instructions = [
      s_mov_b32(VCC_LO, 0b11),  # VCC = 0b11 (lanes 0,1 have carry-in)
      v_mov_b32_e32(v[0], 1),  # S0 = 1 for all lanes
      v_mov_b32_e32(v[1], 1),  # S1 = 1 for all lanes
      v_add_co_ci_u32_e32(v[2], v[0], v[1]),  # D0 = 1 + 1 + 1 = 3 (no overflow)
    ]
    st = run_program(instructions, n_lanes=2)
    self.assertEqual(st.vgpr[0][2], 3)  # lane 0: 1 + 1 + 1 = 3
    self.assertEqual(st.vgpr[1][2], 3)  # lane 1: 1 + 1 + 1 = 3
    self.assertEqual(st.vcc, 0)  # No carry out for any lane - all VCC bits must be cleared

  def test_v_add_co_ci_u32_preserves_inactive_vcc_bits(self):
    """V_ADD_CO_CI_U32: VCC carry-out overwrites entire VCC register.

    VOP2 carry instructions write ALL VCC bits based on carry-out, clearing
    bits for lanes that don't overflow regardless of EXEC mask.

    Note: This differs from VOPC which only writes active lane bits.
    """
    instructions = [
      s_mov_b32(VCC_LO, 0x00010000),  # VCC bit 16 set
      v_mov_b32_e32(v[0], 1),  # S0 = 1
      v_mov_b32_e32(v[1], 1),  # S1 = 1
      v_add_co_ci_u32_e32(v[2], v[0], v[1]),  # D0 = 1 + 1 + 0 = 2 (no carry)
    ]
    st = run_program(instructions, n_lanes=4)
    self.assertEqual(st.vgpr[0][2], 2)  # lane 0: 1 + 1 + 0 = 2
    # VCC should be completely cleared (all lanes have no carry-out)
    self.assertEqual(st.vcc, 0)

  def test_v_add_co_ci_u32_all_lanes_same_result(self):
    """V_ADD_CO_CI_U32: all active lanes should produce the same result.

    When the same constant inputs are used across all lanes, each lane should
    compute the same result and write to its own VGPR slot.

    Regression test for: VGPR writes not happening for all lanes.
    """
    instructions = [
      s_mov_b32(VCC_LO, 0),  # No carry-in
      v_mov_b32_e32(v[0], 3),  # inline constant 3
      v_mov_b32_e32(v[1], 5),  # value 5
      v_add_co_ci_u32_e32(v[1], 3, v[1]),  # v[1] = 3 + v[1] + 0 = 3 + 5 = 8
    ]
    st = run_program(instructions, n_lanes=4)
    # All 4 lanes should have v[1] = 8
    for lane in range(4):
      self.assertEqual(st.vgpr[lane][1], 8, f"lane {lane} should have v[1]=8")

  def test_v_sub_co_ci_u32_no_borrow(self):
    """V_SUB_CO_CI_U32: D0 = S0 - S1 - VCC_IN, when VCC_IN=0."""
    instructions = [
      s_mov_b32(VCC_LO, 0),  # VCC = 0 (no borrow in)
      v_mov_b32_e32(v[0], 10),  # S0 = 10
      v_mov_b32_e32(v[1], 5),  # S1 = 5
      v_sub_co_ci_u32_e32(v[2], v[0], v[1]),  # D0 = 10 - 5 - 0 = 5
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][2], 5)
    self.assertEqual(st.vcc, 0)  # No borrow out

  def test_v_sub_co_ci_u32_vop3sd_separate_carry_regs(self):
    """VOP3SD V_SUB_CO_CI_U32: carry-in from src2, carry-out to sdst (separate registers).

    This tests the VOP3SD encoding where src2 specifies the carry-in register
    independently from sdst (carry-out). The bug was reading carry-in from sdst
    instead of src2.

    Computation: D0 = S0 - S1 - carry_in = 0 - 0 - 1 = -1 = 0xFFFFFFFF
    """
    instructions = [
      s_mov_b32(s[6], 1),  # carry-in = 1 (in s[6])
      s_mov_b32(s[10], 0),  # carry-out dest = 0 initially (in s[10])
      # VOP3SD: v_sub_co_ci_u32(vdst, sdst, src0, src1, src2)
      # src2 is carry-in (s[6]=1), sdst is carry-out (s[10])
      v_sub_co_ci_u32(v[0], s[10], 0, 0, s[6]),  # D0 = 0 - 0 - 1 = -1
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][0], 0xFFFFFFFF)  # -1 as unsigned
    self.assertEqual(st.sgpr[10], 1)  # Borrow out to s[10]

  def test_v_add_co_ci_u32_vop3sd_separate_carry_regs(self):
    """VOP3SD V_ADD_CO_CI_U32: carry-in from src2, carry-out to sdst (separate registers).

    This tests the VOP3SD encoding where src2 specifies the carry-in register
    independently from sdst (carry-out).

    Computation: D0 = S0 + S1 + carry_in = 5 + 10 + 1 = 16
    """
    instructions = [
      s_mov_b32(s[6], 1),  # carry-in = 1 (in s[6])
      s_mov_b32(s[10], 0),  # carry-out dest = 0 initially (in s[10])
      # VOP3SD: v_add_co_ci_u32(vdst, sdst, src0, src1, src2)
      v_add_co_ci_u32(v[0], s[10], 5, 10, s[6]),  # D0 = 5 + 10 + 1 = 16
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][0], 16)
    self.assertEqual(st.sgpr[10], 0)  # No carry out

  def test_v_add_co_ci_u32_vop3sd_null_sdst(self):
    """VOP3SD V_ADD_CO_CI_U32 with sdst=NULL: carry output is discarded.

    When sdst=NULL (register 124), the carry-out should NOT be written anywhere.
    We verify this by checking that VCC (which we set to a sentinel value) is unchanged.
    """
    instructions = [
      s_mov_b32(VCC_LO, 0xDEADBEEF),  # Sentinel value in VCC
      s_mov_b32(s[6], 0),  # carry-in = 0
      # VOP3SD with NULL sdst: carry-out should be discarded
      # Uses 0xFFFFFFFF + 1 + 0 = 0 with carry-out=1, but carry should not be written
      v_add_co_ci_u32(v[0], NULL, 0xFFFFFFFF, 1, s[6]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][0], 0)  # 0xFFFFFFFF + 1 + 0 = 0 (overflow)
    self.assertEqual(st.vcc, 0xDEADBEEF)  # VCC unchanged - carry was discarded


if __name__ == '__main__':
  unittest.main()
