"""Tests for VOPC instructions - vector compare operations.

Includes: v_cmp_class_f32, v_cmp_class_f16, v_cmp_eq_*, v_cmp_lt_*, v_cmp_gt_*
"""
import unittest
from extra.assembly.amd.test.hw.helpers import *

VCC = 106  # SGPR index for VCC_LO

class TestCmpClass(unittest.TestCase):
  """Tests for V_CMP_CLASS_F32 float classification."""

  def test_cmp_class_quiet_nan(self):
    """V_CMP_CLASS_F32 detects quiet NaN."""
    quiet_nan = 0x7fc00000
    instructions = [
      s_mov_b32(s[0], quiet_nan),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], 0b0000000010),  # bit 1 = quiet NaN
      v_cmp_class_f32_e32(v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "Should detect quiet NaN")

  def test_cmp_class_signaling_nan(self):
    """V_CMP_CLASS_F32 detects signaling NaN."""
    signal_nan = 0x7f800001
    instructions = [
      s_mov_b32(s[0], signal_nan),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], 0b0000000001),  # bit 0 = signaling NaN
      v_cmp_class_f32_e32(v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "Should detect signaling NaN")

  def test_cmp_class_positive_inf(self):
    """V_CMP_CLASS_F32 detects +inf."""
    pos_inf = 0x7f800000
    instructions = [
      s_mov_b32(s[0], pos_inf),
      s_mov_b32(s[1], 0b1000000000),  # bit 9 = +inf
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_cmp_class_f32_e32(v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "Should detect +inf")

  def test_cmp_class_negative_inf(self):
    """V_CMP_CLASS_F32 detects -inf."""
    neg_inf = 0xff800000
    instructions = [
      s_mov_b32(s[0], neg_inf),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], 0b0000000100),  # bit 2 = -inf
      v_cmp_class_f32_e32(v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "Should detect -inf")

  def test_cmp_class_normal_positive(self):
    """V_CMP_CLASS_F32 detects positive normal."""
    instructions = [
      v_mov_b32_e32(v[0], 1.0),
      s_mov_b32(s[1], 0b0100000000),  # bit 8 = positive normal
      v_mov_b32_e32(v[1], s[1]),
      v_cmp_class_f32_e32(v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "Should detect positive normal")

  def test_cmp_class_normal_negative(self):
    """V_CMP_CLASS_F32 detects negative normal."""
    instructions = [
      v_mov_b32_e32(v[0], -1.0),
      v_mov_b32_e32(v[1], 0b0000001000),  # bit 3 = negative normal
      v_cmp_class_f32_e32(v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "Should detect negative normal")

  def test_cmp_class_quiet_nan_not_signaling(self):
    """Quiet NaN does not match signaling NaN mask."""
    quiet_nan = 0x7fc00000
    instructions = [
      s_mov_b32(s[0], quiet_nan),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], 0b0000000001),  # bit 0 = signaling NaN only
      v_cmp_class_f32_e32(v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 0, "Quiet NaN should not match signaling mask")

  def test_cmp_class_signaling_nan_not_quiet(self):
    """Signaling NaN does not match quiet NaN mask."""
    signal_nan = 0x7f800001
    instructions = [
      s_mov_b32(s[0], signal_nan),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], 0b0000000010),  # bit 1 = quiet NaN only
      v_cmp_class_f32_e32(v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 0, "Signaling NaN should not match quiet mask")

  def test_v_cmp_sets_vcc_bits(self):
    """V_CMP_EQ sets VCC bits based on per-lane comparison."""
    instructions = [
      s_mov_b32(s[0], 5),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[0]),
      v_cmp_eq_u32_e32(v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=4)
    self.assertEqual(st.vcc & 0xf, 0xf, "All lanes should match")


class TestCmpClassF16(unittest.TestCase):
  """Tests for V_CMP_CLASS_F16 float classification.

  Class bit mapping:
    bit 0 = signaling NaN
    bit 1 = quiet NaN
    bit 2 = -infinity
    bit 3 = -normal
    bit 4 = -denormal
    bit 5 = -zero
    bit 6 = +zero
    bit 7 = +denormal
    bit 8 = +normal
    bit 9 = +infinity
  """

  def test_cmp_class_f16_positive_zero(self):
    """V_CMP_CLASS_F16: +zero matches bit 6."""
    instructions = [
      v_mov_b32_e32(v[0], 0x0000),  # f16 +0.0
      v_mov_b32_e32(v[1], 0x40),     # bit 6 = +zero
      v_cmp_class_f16_e32(v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "Should detect positive zero")

  def test_cmp_class_f16_negative_zero(self):
    """V_CMP_CLASS_F16: -zero matches bit 5."""
    instructions = [
      s_mov_b32(s[0], 0x8000),       # f16 -0.0
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], 0x20),     # bit 5 = -zero
      v_cmp_class_f16_e32(v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "Should detect negative zero")

  def test_cmp_class_f16_positive_normal(self):
    """V_CMP_CLASS_F16: +1.0 (normal) matches bit 8."""
    instructions = [
      s_mov_b32(s[0], 0x3c00),       # f16 +1.0
      s_mov_b32(s[1], 0x100),        # bit 8 = +normal
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_cmp_class_f16_e32(v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "Should detect positive normal")

  def test_cmp_class_f16_negative_normal(self):
    """V_CMP_CLASS_F16: -1.0 (normal) matches bit 3."""
    instructions = [
      s_mov_b32(s[0], 0xbc00),       # f16 -1.0
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], 0x08),     # bit 3 = -normal
      v_cmp_class_f16_e32(v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "Should detect negative normal")

  def test_cmp_class_f16_positive_infinity(self):
    """V_CMP_CLASS_F16: +inf matches bit 9."""
    instructions = [
      s_mov_b32(s[0], 0x7c00),       # f16 +inf
      s_mov_b32(s[1], 0x200),        # bit 9 = +inf
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_cmp_class_f16_e32(v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "Should detect positive infinity")

  def test_cmp_class_f16_negative_infinity(self):
    """V_CMP_CLASS_F16: -inf matches bit 2."""
    instructions = [
      s_mov_b32(s[0], 0xfc00),       # f16 -inf
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], 0x04),     # bit 2 = -inf
      v_cmp_class_f16_e32(v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "Should detect negative infinity")

  def test_cmp_class_f16_quiet_nan(self):
    """V_CMP_CLASS_F16: quiet NaN matches bit 1."""
    instructions = [
      s_mov_b32(s[0], 0x7e00),       # f16 quiet NaN
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], 0x02),     # bit 1 = quiet NaN
      v_cmp_class_f16_e32(v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "Should detect quiet NaN")

  def test_cmp_class_f16_signaling_nan(self):
    """V_CMP_CLASS_F16: signaling NaN matches bit 0."""
    instructions = [
      s_mov_b32(s[0], 0x7c01),       # f16 signaling NaN
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], 0x01),     # bit 0 = signaling NaN
      v_cmp_class_f16_e32(v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "Should detect signaling NaN")

  def test_cmp_class_f16_positive_denormal(self):
    """V_CMP_CLASS_F16: positive denormal matches bit 7."""
    instructions = [
      v_mov_b32_e32(v[0], 1),        # f16 +denormal (0x0001)
      v_mov_b32_e32(v[1], 0x80),     # bit 7 = +denormal
      v_cmp_class_f16_e32(v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "Should detect positive denormal")

  def test_cmp_class_f16_negative_denormal(self):
    """V_CMP_CLASS_F16: negative denormal matches bit 4."""
    instructions = [
      s_mov_b32(s[0], 0x8001),       # f16 -denormal
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], 0x10),     # bit 4 = -denormal
      v_cmp_class_f16_e32(v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "Should detect negative denormal")

  def test_cmp_class_f16_combined_mask_zeros(self):
    """V_CMP_CLASS_F16: mask 0x60 covers both +zero and -zero."""
    instructions = [
      v_mov_b32_e32(v[0], 0),         # f16 +0.0
      v_mov_b32_e32(v[1], 0x60),      # bits 5 and 6 (+-zero)
      v_cmp_class_f16_e32(v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "VCC should be 1 for +zero with mask 0x60")

  def test_cmp_class_f16_combined_mask_1f8(self):
    """V_CMP_CLASS_F16: mask 0x1f8 covers -normal,-denorm,-zero,+zero,+denorm,+normal.

    This is the exact mask used in the f16 sin kernel at PC=46.
    """
    instructions = [
      v_mov_b32_e32(v[0], 0),         # f16 +0.0
      s_mov_b32(s[0], 0x1f8),
      v_mov_b32_e32(v[1], s[0]),      # mask 0x1f8
      v_cmp_class_f16_e32(v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "VCC should be 1 for +zero with mask 0x1f8")

  def test_cmp_class_f16_vop3_encoding(self):
    """V_CMP_CLASS_F16 in VOP3 encoding (v_cmp_class_f16_e64)."""
    instructions = [
      v_mov_b32_e32(v[0], 0),         # f16 +0.0
      s_mov_b32(s[0], 0x1f8),         # class mask
      v_cmp_class_f16_e64(VCC_LO, v[0], s[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "VCC should be 1 for +zero with VOP3 encoding")

  def test_cmp_class_f16_vop3_normal_positive(self):
    """V_CMP_CLASS_F16 VOP3 encoding with +1.0 (normal)."""
    instructions = [
      s_mov_b32(s[0], 0x3c00),        # f16 +1.0
      v_mov_b32_e32(v[0], s[0]),
      s_mov_b32(s[1], 0x1f8),         # class mask
      v_cmp_class_f16_e64(VCC_LO, v[0], s[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "VCC should be 1 for +1.0 (normal) with mask 0x1f8")

  def test_cmp_class_f16_vop3_nan_fails_mask(self):
    """V_CMP_CLASS_F16 VOP3: NaN should NOT match mask 0x1f8 (no NaN bits set)."""
    instructions = [
      s_mov_b32(s[0], 0x7e00),        # f16 quiet NaN
      v_mov_b32_e32(v[0], s[0]),
      s_mov_b32(s[1], 0x1f8),         # class mask
      v_cmp_class_f16_e64(VCC_LO, v[0], s[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 0, "VCC should be 0 for NaN with mask 0x1f8 (no NaN bits)")

  def test_cmp_class_f16_vop3_inf_fails_mask(self):
    """V_CMP_CLASS_F16 VOP3: +inf should NOT match mask 0x1f8 (no inf bits set)."""
    instructions = [
      s_mov_b32(s[0], 0x7c00),        # f16 +inf
      v_mov_b32_e32(v[0], s[0]),
      s_mov_b32(s[1], 0x1f8),         # class mask
      v_cmp_class_f16_e64(VCC_LO, v[0], s[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 0, "VCC should be 0 for +inf with mask 0x1f8 (no inf bits)")


class TestCmpInt(unittest.TestCase):
  """Tests for integer comparison operations."""

  def test_v_cmp_eq_u32(self):
    """V_CMP_EQ_U32 sets VCC bits based on per-lane comparison."""
    instructions = [
      s_mov_b32(s[0], 5),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[0]),
      v_cmp_eq_u32_e32(v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=4)
    self.assertEqual(st.vcc & 0xf, 0xf, "All lanes should match")

  def test_v_cmp_ne_u32_with_zero(self):
    """V_CMP_NE_U32: compare with zero, used for int->bool cast."""
    instructions = [
      v_mov_b32_e32(v[1], 0),
      v_cmp_eq_u32_e32(1, v[255]),  # vcc = (lane == 1)
      v_cndmask_b32_e64(v[1], v[1], 1, VCC_LO),  # v1[lane1] = 1
      v_cmp_ne_u32_e32(0, v[1]),  # vcc = (0 != v1)
      v_cndmask_b32_e64(v[0], 0, 1, VCC_LO),  # v0 = vcc ? 1 : 0
    ]
    st = run_program(instructions, n_lanes=2)
    self.assertEqual(st.vgpr[0][0], 0, "lane 0: 0 != 0 should be false")
    self.assertEqual(st.vgpr[1][0], 1, "lane 1: 0 != 1 should be true")
    self.assertEqual(st.vcc & 0x3, 0x2, "VCC should be 0b10")

  def test_v_cmp_ne_u32_all_nonzero(self):
    """V_CMP_NE_U32: all lanes have nonzero values."""
    instructions = [
      v_mov_b32_e32(v[1], 5),
      v_cmp_ne_u32_e32(0, v[1]),
    ]
    st = run_program(instructions, n_lanes=4)
    self.assertEqual(st.vcc & 0xf, 0xf, "All lanes should be != 0")

  def test_cmp_eq_u16_opsel_lo_lo(self):
    """V_CMP_EQ_U16 comparing lo halves."""
    instructions = [
      s_mov_b32(s[0], 0x12340005),  # lo=5, hi=0x1234
      s_mov_b32(s[1], 0xABCD0005),  # lo=5, hi=0xABCD
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_cmp_eq_u16_e32(v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "Lo halves should be equal")

  def test_cmp_eq_u16_opsel_hi_hi(self):
    """V_CMP_EQ_U16 comparing hi halves with VOP3 opsel."""
    instructions = [
      s_mov_b32(s[2], 0x00051234),  # hi=5, lo=0x1234
      v_mov_b32_e32(v[0], s[2]),
      s_mov_b32(s[2], 0x0005ABCD),  # hi=5, lo=0xABCD
      v_mov_b32_e32(v[1], s[2]),
      v_cmp_eq_u16_e64(vdst=s[0], src0=v[0], src1=v[1], opsel=3),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[0] & 1, 1, "Hi halves should be equal: 5==5")

  def test_cmp_eq_u16_opsel_hi_hi_equal(self):
    """V_CMP_EQ_U16 VOP3 with opsel=3 compares hi halves (equal case)."""
    instructions = [
      s_mov_b32(s[2], 0x12340005),  # lo=5, hi=0x1234
      v_mov_b32_e32(v[0], s[2]),
      s_mov_b32(s[2], 0x12340009),  # lo=9, hi=0x1234
      v_mov_b32_e32(v[1], s[2]),
      v_cmp_eq_u16_e64(vdst=s[0], src0=v[0], src1=v[1], opsel=3),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[0] & 1, 1, "hi==hi should be true: 0x1234==0x1234")

  def test_cmp_gt_u16_opsel_hi(self):
    """V_CMP_GT_U16 VOP3 with opsel=3 compares hi halves."""
    instructions = [
      s_mov_b32(s[2], 0x99990005),  # lo=5, hi=0x9999
      v_mov_b32_e32(v[0], s[2]),
      s_mov_b32(s[2], 0x12340005),  # lo=5, hi=0x1234
      v_mov_b32_e32(v[1], s[2]),
      v_cmp_gt_u16_e64(vdst=s[0], src0=v[0], src1=v[1], opsel=3),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[0] & 1, 1, "hi>hi should be true: 0x9999>0x1234")


class TestCmpFloat(unittest.TestCase):
  """Tests for float comparison operations."""

  def test_v_cmp_lt_f16_vsrc1_hi(self):
    """V_CMP_LT_F16 with both operands from high half using VOP3 opsel."""
    instructions = [
      s_mov_b32(s[2], 0x3c000000),  # hi=1.0 (f16), lo=0
      v_mov_b32_e32(v[0], s[2]),
      s_mov_b32(s[2], 0x40000000),  # hi=2.0 (f16), lo=0
      v_mov_b32_e32(v[1], s[2]),
      v_cmp_lt_f16_e64(vdst=s[0], src0=v[0], src1=v[1], opsel=3),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[0] & 1, 1, "1.0 < 2.0 should be true")

  def test_v_cmp_gt_f16_vsrc1_hi(self):
    """V_CMP_GT_F16 with both operands from high half using VOP3 opsel."""
    instructions = [
      s_mov_b32(s[2], 0x40000000),  # hi=2.0 (f16), lo=0
      v_mov_b32_e32(v[0], s[2]),
      s_mov_b32(s[2], 0x3c000000),  # hi=1.0 (f16), lo=0
      v_mov_b32_e32(v[1], s[2]),
      v_cmp_gt_f16_e64(vdst=s[0], src0=v[0], src1=v[1], opsel=3),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[0] & 1, 1, "2.0 > 1.0 should be true")

  def test_v_cmp_eq_f16_vsrc1_hi_equal(self):
    """v_cmp_eq_f16 with equal low and high halves."""
    instructions = [
      s_mov_b32(s[0], 0x42004200),  # hi=3.0 (0x4200), lo=3.0 (0x4200)
      v_mov_b32_e32(v[0], s[0]),
      v_cmp_eq_f16_e32(v[0], v[0].h),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "Expected vcc=1 (3.0 == 3.0)")

  def test_v_cmp_neq_f16_vsrc1_hi(self):
    """v_cmp_neq_f16 with different low and high halves."""
    instructions = [
      s_mov_b32(s[0], 0x40003c00),  # hi=2.0 (0x4000), lo=1.0 (0x3c00)
      v_mov_b32_e32(v[0], s[0]),
      v_cmp_lg_f16_e32(v[0], v[0].h),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "Expected vcc=1 (1.0 != 2.0)")

  def test_v_cmp_nge_f16_inf_self(self):
    """v_cmp_nge_f16 comparing -inf with itself (unordered less than).

    Regression test: -inf < -inf should be false (IEEE 754).
    """
    instructions = [
      s_mov_b32(s[0], 0xFC00FC00),  # both halves = -inf (0xFC00)
      v_mov_b32_e32(v[0], s[0]),
      v_cmp_nge_f16_e32(v[0], v[0].h),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 0, "Expected vcc=0 (-inf >= -inf)")

  def test_v_cmp_f16_multilane(self):
    """v_cmp_lt_f16 with vsrc1=v128 across multiple lanes."""
    instructions = [
      # Lane 0: v0 = 0x40003c00 (hi=2.0, lo=1.0) -> 1.0 < 2.0 = true
      # Lane 1: v0 = 0x3c004000 (hi=1.0, lo=2.0) -> 2.0 < 1.0 = false
      v_mov_b32_e32(v[0], 0x40003c00),  # default
      v_cmp_eq_u32_e32(1, v[255]),  # vcc = (lane == 1)
      v_cndmask_b32_e64(v[0], v[0], 0x3c004000, SrcEnum.VCC_LO),
      v_cmp_lt_f16_e32(v[0], v[0].h),
    ]
    st = run_program(instructions, n_lanes=2)
    self.assertEqual(st.vcc & 1, 1, "Lane 0: expected vcc=1 (1.0 < 2.0)")
    self.assertEqual((st.vcc >> 1) & 1, 0, "Lane 1: expected vcc=0 (2.0 < 1.0)")


class TestVOP3VOPCModifiers(unittest.TestCase):
  """Tests for VOP3 VOPC with abs/neg modifiers."""

  def test_v_cmp_ge_f32_abs_both(self):
    """v_cmp_ge_f32 with abs on both sources: abs(0.0) >= abs(-1.0) = false.

    Regression test: int16 mod operation uses v_cmp_ge_f32 with abs modifiers.
    """
    instructions = [
      v_mov_b32_e32(v[0], 0.0),
      v_mov_b32_e32(v[1], -1.0),
      # abs=0b11 means abs(src0) and abs(src1)
      v_cmp_ge_f32_e64(VCC_LO, v[0], v[1], abs=0b11),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 0, "abs(0.0) >= abs(-1.0) should be false")

  def test_v_cmp_ge_f32_abs_negative_divisor(self):
    """v_cmp_ge_f32 with abs: remainder check for negative divisor.

    Tests the exact comparison used in int16 mod: abs(rem_f) >= abs(div_f).
    For 1 % -1: rem_f = 0.0, div_f = -1.0, so abs(0.0) >= abs(-1.0) = false.
    """
    instructions = [
      v_mov_b32_e32(v[0], 0.0),    # remainder as float
      v_mov_b32_e32(v[1], -1.0),   # divisor as float
      v_cmp_ge_f32_e64(VCC_LO, v[0], v[1], abs=0b11),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 0, "abs(0.0) >= abs(-1.0) should be false")

  def test_v_cmp_ge_f32_abs_small_remainder(self):
    """v_cmp_ge_f32 with abs: abs(-0.5) >= abs(-3.0) = false."""
    instructions = [
      v_mov_b32_e32(v[0], -0.5),
      v_mov_b32_e32(v[1], -3.0),
      v_cmp_ge_f32_e64(VCC_LO, v[0], v[1], abs=0b11),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 0, "abs(-0.5) >= abs(-3.0) should be false")

  def test_v_cmp_ge_f32_abs_equal(self):
    """v_cmp_ge_f32 with abs: abs(-1.0) >= abs(1.0) = true."""
    instructions = [
      v_mov_b32_e32(v[0], -1.0),
      v_mov_b32_e32(v[1], 1.0),
      v_cmp_ge_f32_e64(VCC_LO, v[0], v[1], abs=0b11),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "abs(-1.0) >= abs(1.0) should be true")


class TestVOP3VOPC64Bit(unittest.TestCase):
  """Tests for VOP3 VOPC with 64-bit operands."""

  def test_v_cmp_lt_f64_basic(self):
    """v_cmp_lt_f64: 0.0 < 1.0 = true."""
    zero_f64 = f2i64(0.0)
    one_f64 = f2i64(1.0)
    instructions = [
      s_mov_b32(s[0], zero_f64 & 0xffffffff),
      s_mov_b32(s[1], zero_f64 >> 32),
      s_mov_b32(s[2], one_f64 & 0xffffffff),
      s_mov_b32(s[3], one_f64 >> 32),
      v_cmp_lt_f64_e64(VCC_LO, s[0:1], s[2:3]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "0.0 < 1.0 should be true")

  def test_v_cmp_lt_f64_negative(self):
    """v_cmp_lt_f64: -1.0 < 0.0 = true."""
    neg_one_f64 = f2i64(-1.0)
    zero_f64 = f2i64(0.0)
    instructions = [
      s_mov_b32(s[0], neg_one_f64 & 0xffffffff),
      s_mov_b32(s[1], neg_one_f64 >> 32),
      s_mov_b32(s[2], zero_f64 & 0xffffffff),
      s_mov_b32(s[3], zero_f64 >> 32),
      v_cmp_lt_f64_e64(VCC_LO, s[0:1], s[2:3]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "-1.0 < 0.0 should be true")

  def test_v_cmp_lt_i64_signed(self):
    """v_cmp_lt_i64: 0 < -1 (signed) = false."""
    instructions = [
      s_mov_b32(s[0], 0),
      s_mov_b32(s[1], 0),              # s[0:1] = 0
      s_mov_b32(s[2], 0xffffffff),
      s_mov_b32(s[3], 0xffffffff),     # s[2:3] = -1
      v_cmp_lt_i64_e64(VCC_LO, s[0:1], s[2:3]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 0, "0 < -1 (signed) should be false")

  def test_v_cmp_lt_u64_unsigned(self):
    """v_cmp_lt_u64: 0 < 0xFFFFFFFFFFFFFFFF (unsigned) = true."""
    instructions = [
      s_mov_b32(s[0], 0),
      s_mov_b32(s[1], 0),              # s[0:1] = 0
      s_mov_b32(s[2], 0xffffffff),
      s_mov_b32(s[3], 0xffffffff),     # s[2:3] = max uint64
      v_cmp_lt_u64_e64(VCC_LO, s[0:1], s[2:3]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "0 < max_uint64 should be true")


class TestVOPCF64(unittest.TestCase):
  """Tests for VOPC (E32 encoding) with 64-bit float operands. Regression test for f64 compare bug."""

  def test_v_cmp_lt_f64_e32_true(self):
    """v_cmp_lt_f64_e32: 2.0 < 3.0 = true."""
    lo0, hi0 = f2i64(2.0) & 0xffffffff, f2i64(2.0) >> 32
    lo1, hi1 = f2i64(3.0) & 0xffffffff, f2i64(3.0) >> 32
    instructions = [
      s_mov_b32(s[0], lo0), s_mov_b32(s[1], hi0),
      s_mov_b32(s[2], lo1), s_mov_b32(s[3], hi1),
      v_mov_b32_e32(v[0], s[0]), v_mov_b32_e32(v[1], s[1]),
      v_mov_b32_e32(v[2], s[2]), v_mov_b32_e32(v[3], s[3]),
      v_cmp_lt_f64_e32(v[0:1], v[2:3]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "2.0 < 3.0 should be true")

  def test_v_cmp_lt_f64_e32_false(self):
    """v_cmp_lt_f64_e32: 3.0 < 2.0 = false."""
    lo0, hi0 = f2i64(3.0) & 0xffffffff, f2i64(3.0) >> 32
    lo1, hi1 = f2i64(2.0) & 0xffffffff, f2i64(2.0) >> 32
    instructions = [
      s_mov_b32(s[0], lo0), s_mov_b32(s[1], hi0),
      s_mov_b32(s[2], lo1), s_mov_b32(s[3], hi1),
      v_mov_b32_e32(v[0], s[0]), v_mov_b32_e32(v[1], s[1]),
      v_mov_b32_e32(v[2], s[2]), v_mov_b32_e32(v[3], s[3]),
      v_cmp_lt_f64_e32(v[0:1], v[2:3]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 0, "3.0 < 2.0 should be false")

  def test_v_cmp_nlt_f64_e32_true(self):
    """v_cmp_nlt_f64_e32: !(3.0 < 2.0) = true."""
    lo0, hi0 = f2i64(3.0) & 0xffffffff, f2i64(3.0) >> 32
    lo1, hi1 = f2i64(2.0) & 0xffffffff, f2i64(2.0) >> 32
    instructions = [
      s_mov_b32(s[0], lo0), s_mov_b32(s[1], hi0),
      s_mov_b32(s[2], lo1), s_mov_b32(s[3], hi1),
      v_mov_b32_e32(v[0], s[0]), v_mov_b32_e32(v[1], s[1]),
      v_mov_b32_e32(v[2], s[2]), v_mov_b32_e32(v[3], s[3]),
      v_cmp_nlt_f64_e32(v[0:1], v[2:3]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "!(3.0 < 2.0) should be true")

  def test_v_cmp_nlt_f64_e32_false(self):
    """v_cmp_nlt_f64_e32: !(2.0 < 3.0) = false."""
    lo0, hi0 = f2i64(2.0) & 0xffffffff, f2i64(2.0) >> 32
    lo1, hi1 = f2i64(3.0) & 0xffffffff, f2i64(3.0) >> 32
    instructions = [
      s_mov_b32(s[0], lo0), s_mov_b32(s[1], hi0),
      s_mov_b32(s[2], lo1), s_mov_b32(s[3], hi1),
      v_mov_b32_e32(v[0], s[0]), v_mov_b32_e32(v[1], s[1]),
      v_mov_b32_e32(v[2], s[2]), v_mov_b32_e32(v[3], s[3]),
      v_cmp_nlt_f64_e32(v[0:1], v[2:3]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 0, "!(2.0 < 3.0) should be false")


class TestCmpxExec(unittest.TestCase):
  """Tests for V_CMPX instructions that modify EXEC mask."""

  def test_v_cmpx_ngt_f32_e64_all_true(self):
    """V_CMPX_NGT_F32_E64: all lanes pass (literal <= all values)."""
    # 131072.0 = 0x48000000
    # All values > 131072, so !(131072 > val) = true for all
    instructions = [
      s_mov_b32(EXEC_LO, 0x7),  # 3 lanes active
      v_mov_b32_e32(v[0], f2i(200000.0)),  # lane 0
      v_cmp_eq_u32_e32(1, v[255]),
      v_cndmask_b32_e64(v[1], v[0], f2i(300000.0), VCC_LO),  # lane 1
      v_cmp_eq_u32_e32(2, v[255]),
      v_cndmask_b32_e64(v[1], v[1], f2i(400000.0), VCC_LO),  # lane 2
      # Now v[1] has: lane0=200000, lane1=300000, lane2=400000
      # Compare: !(131072.0 > v[1]) i.e., 131072.0 <= v[1]
      v_cmpx_ngt_f32_e64(EXEC_LO, f2i(131072.0), v[1]),
    ]
    st = run_program(instructions, n_lanes=3)
    # All values > 131072, so all lanes should remain active
    self.assertEqual(st.sgpr[EXEC_LO.offset] & 0x7, 0x7, "All 3 lanes should remain active")

  def test_v_cmpx_ngt_f32_e64_some_false(self):
    """V_CMPX_NGT_F32_E64: some lanes fail (literal > some values)."""
    instructions = [
      s_mov_b32(EXEC_LO, 0x7),  # 3 lanes active
      v_mov_b32_e32(v[0], f2i(100000.0)),  # lane 0: 131072 > 100000 = true, so !(true) = false
      v_cmp_eq_u32_e32(1, v[255]),
      v_cndmask_b32_e64(v[1], v[0], f2i(200000.0), VCC_LO),  # lane 1: 131072 > 200000 = false, so !(false) = true
      v_cmp_eq_u32_e32(2, v[255]),
      v_cndmask_b32_e64(v[1], v[1], f2i(150000.0), VCC_LO),  # lane 2: 131072 > 150000 = false, so !(false) = true
      v_cmpx_ngt_f32_e64(EXEC_LO, f2i(131072.0), v[1]),
    ]
    st = run_program(instructions, n_lanes=3)
    # lane 0: fail (100000 < 131072), lanes 1,2: pass
    self.assertEqual(st.sgpr[EXEC_LO.offset] & 0x7, 0x6, "Lanes 1,2 should be active, lane 0 inactive")

  def test_v_cmpx_ngt_f32_e64_all_false(self):
    """V_CMPX_NGT_F32_E64: all lanes fail (literal > all values)."""
    instructions = [
      s_mov_b32(EXEC_LO, 0x7),  # 3 lanes active
      v_mov_b32_e32(v[0], f2i(100.0)),  # all lanes have 100.0
      # 131072 > 100 = true, so !(true) = false for all
      v_cmpx_ngt_f32_e64(EXEC_LO, f2i(131072.0), v[0]),
    ]
    st = run_program(instructions, n_lanes=3)
    self.assertEqual(st.sgpr[EXEC_LO.offset] & 0x7, 0x0, "All lanes should be inactive")

  def test_v_cmpx_ngt_f32_e64_large_values(self):
    """V_CMPX_NGT_F32_E64: test with values that trigger Payne-Hanek in sin().

    This is a regression test for the sin(859240.0) bug.
    Values 859240, 1000000, 100594688 should all pass !(131072 > val).
    """
    instructions = [
      s_mov_b32(EXEC_LO, 0x7),  # 3 lanes active
      v_mov_b32_e32(v[0], f2i(859240.0)),   # lane 0
      v_cmp_eq_u32_e32(1, v[255]),
      v_cndmask_b32_e64(v[1], v[0], f2i(1000000.0), VCC_LO),   # lane 1
      v_cmp_eq_u32_e32(2, v[255]),
      v_cndmask_b32_e64(v[1], v[1], f2i(100594688.0), VCC_LO), # lane 2
      v_cmpx_ngt_f32_e64(EXEC_LO, f2i(131072.0), v[1]),
    ]
    st = run_program(instructions, n_lanes=3)
    # All values > 131072, so !(131072 > val) = true for all
    self.assertEqual(st.sgpr[EXEC_LO.offset] & 0x7, 0x7, "All 3 lanes should remain active")


class TestVCCBehavior(unittest.TestCase):
  """Tests for VCC condition code behavior."""

  def test_vcc_all_lanes_true(self):
    """VCC should have all bits set when all lanes compare true."""
    instructions = [
      v_mov_b32_e32(v[0], 5),
      v_mov_b32_e32(v[1], 5),
      v_cmp_eq_u32_e32(v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=32)
    self.assertEqual(st.vcc, 0xFFFFFFFF, "All 32 lanes should be true")

  def test_vcc_lane_dependent(self):
    """VCC should differ per lane based on lane_id comparison."""
    instructions = [
      v_mov_b32_e32(v[0], 16),
      v_cmp_lt_u32_e32(v[255], v[0]),  # lanes 0-15 are < 16
    ]
    st = run_program(instructions, n_lanes=32)
    self.assertEqual(st.vcc & 0xFFFF, 0xFFFF, "Lanes 0-15 should be true")
    self.assertEqual(st.vcc >> 16, 0x0000, "Lanes 16-31 should be false")


class TestCmpNge(unittest.TestCase):
  """Tests for V_CMP_NGE (not-greater-or-equal) with NaN semantics.

  NGE = !(a >= b). With NaN inputs:
  - If either input is NaN, a >= b is false, so !(false) = true
  - This differs from a < b which returns false for NaN inputs
  """

  def test_v_cmp_nge_f32_normal_values(self):
    """v_cmp_nge_f32: basic comparison with normal floats."""
    instructions = [
      v_mov_b32_e32(v[0], f2i(1.0)),
      v_mov_b32_e32(v[1], f2i(2.0)),
      v_cmp_nge_f32_e32(v[0], v[1]),  # !(1.0 >= 2.0) = !(false) = true
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "!(1.0 >= 2.0) should be true")

  def test_v_cmp_nge_f32_equal_values(self):
    """v_cmp_nge_f32: equal values should return false."""
    instructions = [
      v_mov_b32_e32(v[0], f2i(1.0)),
      v_mov_b32_e32(v[1], f2i(1.0)),
      v_cmp_nge_f32_e32(v[0], v[1]),  # !(1.0 >= 1.0) = !(true) = false
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 0, "!(1.0 >= 1.0) should be false")

  def test_v_cmp_nge_f32_greater_value(self):
    """v_cmp_nge_f32: greater value should return false."""
    instructions = [
      v_mov_b32_e32(v[0], f2i(2.0)),
      v_mov_b32_e32(v[1], f2i(1.0)),
      v_cmp_nge_f32_e32(v[0], v[1]),  # !(2.0 >= 1.0) = !(true) = false
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 0, "!(2.0 >= 1.0) should be false")

  def test_v_cmp_nge_f32_neg_inf(self):
    """v_cmp_nge_f32: -inf compared to normal value."""
    neg_inf = 0xff800000  # -inf
    instructions = [
      s_mov_b32(s[0], neg_inf),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], f2i(1.0)),
      v_cmp_nge_f32_e32(v[0], v[1]),  # !(-inf >= 1.0) = !(false) = true
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "!(-inf >= 1.0) should be true")

  def test_v_cmp_nge_f32_clears_inactive_vcc_bits(self):
    """v_cmp_nge_f32 with partial EXEC clears inactive VCC bits (hardware behavior)."""
    neg_inf = 0xff800000  # -inf
    instructions = [
      # Set VCC to all 1s first
      s_mov_b32(VCC_LO, 0xFFFFFFFF),
      # Set EXEC to only lane 0
      s_mov_b32(EXEC_LO, 0x00000001),
      # v0 = 1.0 for lane 0
      v_mov_b32_e32(v[0], f2i(1.0)),
      # Compare: !(-inf >= 1.0) = true for lane 0
      v_cmp_nge_f32_e32(neg_inf, v[0]),
    ]
    st = run_program(instructions, n_lanes=16)
    # Hardware clears inactive lane bits, only active lane results remain
    # Lane 0 result = 1 (true), lanes 1-15 = 0 (cleared)
    self.assertEqual(st.vcc, 0x00000001, "VCC should only have active lane results")

  def test_v_cmp_nge_f32_nan_src0(self):
    """v_cmp_nge_f32: NaN in src0 should return true (NaN >= x is false)."""
    quiet_nan = 0x7fc00000
    instructions = [
      s_mov_b32(s[0], quiet_nan),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], f2i(1.0)),
      v_cmp_nge_f32_e32(v[0], v[1]),  # !(NaN >= 1.0) = !(false) = true
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "!(NaN >= 1.0) should be true")

  def test_v_cmp_nge_f32_nan_src1(self):
    """v_cmp_nge_f32: NaN in src1 should return true (x >= NaN is false)."""
    quiet_nan = 0x7fc00000
    instructions = [
      s_mov_b32(s[0], quiet_nan),
      v_mov_b32_e32(v[0], f2i(1.0)),
      v_mov_b32_e32(v[1], s[0]),
      v_cmp_nge_f32_e32(v[0], v[1]),  # !(1.0 >= NaN) = !(false) = true
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "!(1.0 >= NaN) should be true")

  def test_v_cmp_nge_f32_both_nan(self):
    """v_cmp_nge_f32: both NaN should return true."""
    quiet_nan = 0x7fc00000
    instructions = [
      s_mov_b32(s[0], quiet_nan),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[0]),
      v_cmp_nge_f32_e32(v[0], v[1]),  # !(NaN >= NaN) = !(false) = true
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "!(NaN >= NaN) should be true")


class TestCmpxPartialWavefront(unittest.TestCase):
  """Tests for V_CMPX with partial wavefronts (fewer than 32 active lanes).

  Regression tests for bug where v_cmpx incorrectly set EXEC bits for inactive
  lanes when the wavefront had fewer than 32 lanes. This caused garbage data
  from uninitialized lanes to corrupt memory writes.
  """

  def test_v_cmpx_eq_u32_partial_wave_3_lanes(self):
    """V_CMPX_EQ_U32 with 3 active lanes should only affect those 3 lanes.

    With n_lanes=3, initial EXEC=0x7. After v_cmpx comparing lane_id == 1,
    only lane 1 should pass, so EXEC should become 0x2 (not have bits 3-31 set).
    """
    instructions = [
      v_cmpx_eq_u32_e32(1, v[255]),  # EXEC = lanes where lane_id == 1
    ]
    st = run_program(instructions, n_lanes=3)
    # Only lane 1 should be active (bit 1 set)
    self.assertEqual(st.sgpr[EXEC_LO.offset] & 0xFFFFFFFF, 0x2,
                     "Only lane 1 should be active after v_cmpx_eq_u32 with 3 lanes")

  def test_v_cmpx_eq_u32_partial_wave_5_lanes(self):
    """V_CMPX_EQ_U32 with 5 active lanes."""
    instructions = [
      v_cmpx_eq_u32_e32(3, v[255]),  # EXEC = lanes where lane_id == 3
    ]
    st = run_program(instructions, n_lanes=5)
    self.assertEqual(st.sgpr[EXEC_LO.offset] & 0xFFFFFFFF, 0x8,
                     "Only lane 3 should be active after v_cmpx_eq_u32 with 5 lanes")

  def test_v_cmpx_lt_u32_partial_wave(self):
    """V_CMPX_LT_U32 with partial wavefront."""
    # VOPC: src0 < vsrc1, so we need v_cmpx_gt_u32 to get lane_id < 2
    instructions = [
      v_cmpx_gt_u32_e32(2, v[255]),  # EXEC = lanes where 2 > lane_id (i.e., lane_id < 2)
    ]
    st = run_program(instructions, n_lanes=4)
    # Lanes 0,1 should be active (bits 0,1 set = 0x3)
    self.assertEqual(st.sgpr[EXEC_LO.offset] & 0xFFFFFFFF, 0x3,
                     "Only lanes 0,1 should be active after v_cmpx_gt_u32(2, lane_id) with 4 lanes")

  def test_v_cmpx_ge_u32_partial_wave(self):
    """V_CMPX_GE_U32 with partial wavefront."""
    # VOPC: src0 >= vsrc1, so v_cmpx_le_u32(1, lane_id) gives lane_id >= 2? No.
    # v_cmpx_le_u32(src0, vsrc1) = src0 <= vsrc1 = 1 <= lane_id
    instructions = [
      v_cmpx_le_u32_e32(2, v[255]),  # EXEC = lanes where 2 <= lane_id (i.e., lane_id >= 2)
    ]
    st = run_program(instructions, n_lanes=4)
    # Lanes 2,3 should be active (bits 2,3 set = 0xC)
    self.assertEqual(st.sgpr[EXEC_LO.offset] & 0xFFFFFFFF, 0xC,
                     "Only lanes 2,3 should be active after v_cmpx_le_u32(2, lane_id) with 4 lanes")

  def test_v_cmpx_ne_u32_partial_wave_all_pass(self):
    """V_CMPX_NE_U32 where all active lanes pass."""
    instructions = [
      v_cmpx_ne_u32_e32(99, v[255]),  # EXEC = lanes where lane_id != 99
    ]
    st = run_program(instructions, n_lanes=3)
    # All 3 lanes should remain active (bits 0,1,2 set = 0x7)
    self.assertEqual(st.sgpr[EXEC_LO.offset] & 0xFFFFFFFF, 0x7,
                     "All 3 lanes should remain active when all pass")

  def test_v_cmpx_eq_u32_partial_wave_none_pass(self):
    """V_CMPX_EQ_U32 where no active lanes pass."""
    instructions = [
      v_cmpx_eq_u32_e32(99, v[255]),  # EXEC = lanes where lane_id == 99
    ]
    st = run_program(instructions, n_lanes=3)
    # No lanes should be active
    self.assertEqual(st.sgpr[EXEC_LO.offset] & 0xFFFFFFFF, 0x0,
                     "No lanes should be active when none pass")

  def test_v_cmpx_f32_partial_wave(self):
    """V_CMPX_GT_F32 with partial wavefront - float comparison."""
    instructions = [
      v_cvt_f32_u32_e32(v[0], v[255]),  # v[0] = float(lane_id)
      v_mov_b32_e32(v[1], f2i(0.5)),    # v[1] = 0.5
      v_cmpx_gt_f32_e32(v[0], v[1]),    # EXEC = lanes where v[0] > 0.5
    ]
    st = run_program(instructions, n_lanes=4)
    # Lanes 1,2,3 have values > 0.5, lane 0 has 0.0
    self.assertEqual(st.sgpr[EXEC_LO.offset] & 0xFFFFFFFF, 0xE,
                     "Lanes 1,2,3 should be active (float > 0.5)")

  def test_v_cmpx_e64_partial_wave(self):
    """V_CMPX_EQ_U32_E64 (VOP3 encoding) with partial wavefront."""
    instructions = [
      v_cmpx_eq_u32_e64(EXEC_LO, v[255], 2),  # EXEC = lanes where lane_id == 2
    ]
    st = run_program(instructions, n_lanes=4)
    self.assertEqual(st.sgpr[EXEC_LO.offset] & 0xFFFFFFFF, 0x4,
                     "Only lane 2 should be active after v_cmpx_eq_u32_e64")


if __name__ == '__main__':
  unittest.main()
