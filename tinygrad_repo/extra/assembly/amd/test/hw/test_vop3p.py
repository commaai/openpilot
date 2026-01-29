"""Tests for VOP3P instructions - packed 16-bit vector operations.

Includes: v_pk_add_f16, v_pk_mul_f16, v_pk_fma_f16, v_pack_b32_f16, v_wmma_*, v_dot2_*
"""
import unittest
from extra.assembly.amd.test.hw.helpers import *

class TestPackInstructions(unittest.TestCase):
  """Tests for pack instructions."""

  def test_v_pack_b32_f16(self):
    """V_PACK_B32_F16 packs two f16 values into one 32-bit register."""
    instructions = [
      s_mov_b32(s[0], 0x3c00),  # f16 1.0
      s_mov_b32(s[1], 0x4000),  # f16 2.0
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_pack_b32_f16(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][2]
    self.assertEqual(result, 0x40003c00, f"Expected 0x40003c00, got 0x{result:08x}")

  def test_v_pack_b32_f16_opsel_hi_hi(self):
    """V_PACK_B32_F16 with opsel to read high halves."""
    instructions = [
      s_mov_b32(s[0], 0x40003c00),  # hi=2.0, lo=1.0
      s_mov_b32(s[1], 0x44004200),  # hi=4.0, lo=3.0
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_pack_b32_f16(v[2], v[0], v[1], opsel=0b0011),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][2]
    self.assertEqual(result, 0x44004000, f"Expected 0x44004000, got 0x{result:08x}")


class TestPackMore(unittest.TestCase):
  """Additional pack instruction tests."""

  def test_v_pack_b32_f16_basic(self):
    """V_PACK_B32_F16 packs two f16 values."""
    instructions = [
      s_mov_b32(s[0], 0x3c00),  # f16 1.0
      s_mov_b32(s[1], 0x4000),  # f16 2.0
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_pack_b32_f16(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][2]
    self.assertEqual(result, 0x40003c00, f"Expected 0x40003c00, got 0x{result:08x}")

  def test_v_pack_b32_f16_with_cvt(self):
    """V_PACK_B32_F16 after V_CVT_F16_F32 conversions."""
    instructions = [
      s_mov_b32(s[0], 0x3f800000),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[0]),
      v_cvt_f16_f32_e32(v[2], v[0]),
      v_cvt_f16_f32_e32(v[3], v[1]),
      v_pack_b32_f16(v[4], v[2], v[3]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][4]
    self.assertEqual(result, 0x3c003c00, f"Expected 0x3c003c00, got 0x{result:08x}")

  def test_v_pack_b32_f16_packed_sources(self):
    """V_PACK_B32_F16 with packed f16 sources (reads lo halves)."""
    instructions = [
      s_mov_b32(s[0], 0x40003c00),  # hi=2.0, lo=1.0
      s_mov_b32(s[1], 0x44004200),  # hi=4.0, lo=3.0
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_pack_b32_f16(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][2]
    # Expected: hi=v1.lo=0x4200 (3.0), lo=v0.lo=0x3c00 (1.0) -> 0x42003c00
    self.assertEqual(result, 0x42003c00, f"Expected 0x42003c00, got 0x{result:08x}")

  def test_v_pack_b32_f16_opsel_lo_hi(self):
    """V_PACK_B32_F16 with opsel=0b0010 to read lo from src0, hi from src1."""
    instructions = [
      s_mov_b32(s[0], 0x40003c00),
      s_mov_b32(s[1], 0x44004200),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_pack_b32_f16(v[2], v[0], v[1], opsel=0b0010),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][2]
    self.assertEqual(result, 0x44003c00, f"Expected 0x44003c00, got 0x{result:08x}")

  def test_v_pack_b32_f16_opsel_hi_lo(self):
    """V_PACK_B32_F16 with opsel=0b0001 to read hi from src0, lo from src1."""
    instructions = [
      s_mov_b32(s[0], 0x40003c00),
      s_mov_b32(s[1], 0x44004200),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_pack_b32_f16(v[2], v[0], v[1], opsel=0b0001),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][2]
    self.assertEqual(result, 0x42004000, f"Expected 0x42004000, got 0x{result:08x}")

  def test_v_pack_b32_f16_zeros(self):
    """V_PACK_B32_F16 with zero values."""
    instructions = [
      v_mov_b32_e32(v[0], 0),
      v_mov_b32_e32(v[1], 0),
      v_pack_b32_f16(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][2], 0)

  def test_v_pack_b32_f16_both_positive(self):
    """V_PACK_B32_F16 with positive f16 values."""
    instructions = [
      s_mov_b32(s[0], 0x4200),  # f16 3.0
      s_mov_b32(s[1], 0x4400),  # f16 4.0
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_pack_b32_f16(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][2]
    self.assertEqual(result, 0x44004200, f"Expected 0x44004200, got 0x{result:08x}")


class TestFmaMix(unittest.TestCase):
  """Tests for V_FMA_MIX_F32 and V_FMA_MIXLO_F16."""

  def test_v_fma_mix_f32_all_f32_sources(self):
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

  def test_v_fma_mix_f32_src2_f16_lo(self):
    """V_FMA_MIX_F32 with src2 as f16 from lo bits."""
    from extra.assembly.amd.test.hw.helpers import f32_to_f16
    f16_2 = f32_to_f16(2.0)
    instructions = [
      s_mov_b32(s[0], f2i(1.0)),
      v_mov_b32_e32(v[0], s[0]),
      s_mov_b32(s[1], f2i(3.0)),
      v_mov_b32_e32(v[1], s[1]),
      s_mov_b32(s[2], f16_2),
      v_mov_b32_e32(v[2], s[2]),
      VOP3P(VOP3POp.V_FMA_MIX_F32, vdst=v[3], src0=v[0], src1=v[1], src2=v[2], opsel=0, opsel_hi=0, opsel_hi2=1),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i2f(st.vgpr[0][3])
    self.assertAlmostEqual(result, 5.0, places=5)

  def test_v_fma_mix_f32_src2_f16_hi(self):
    """V_FMA_MIX_F32 with src2 as f16 from hi bits."""
    from extra.assembly.amd.test.hw.helpers import f32_to_f16
    f16_2 = f32_to_f16(2.0)
    val = (f16_2 << 16) | 0
    instructions = [
      s_mov_b32(s[0], f2i(1.0)),
      v_mov_b32_e32(v[0], s[0]),
      s_mov_b32(s[1], f2i(3.0)),
      v_mov_b32_e32(v[1], s[1]),
      s_mov_b32(s[2], val),
      v_mov_b32_e32(v[2], s[2]),
      VOP3P(VOP3POp.V_FMA_MIX_F32, vdst=v[3], src0=v[0], src1=v[1], src2=v[2], opsel=4, opsel_hi=0, opsel_hi2=1),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i2f(st.vgpr[0][3])
    self.assertAlmostEqual(result, 5.0, places=5)

  def test_v_fma_mix_f32_with_abs(self):
    """V_FMA_MIX_F32 with abs modifier on src2."""
    instructions = [
      s_mov_b32(s[0], f2i(2.0)),
      v_mov_b32_e32(v[0], s[0]),
      s_mov_b32(s[1], f2i(3.0)),
      v_mov_b32_e32(v[1], s[1]),
      s_mov_b32(s[2], f2i(-1.0)),
      v_mov_b32_e32(v[2], s[2]),
      VOP3P(VOP3POp.V_FMA_MIX_F32, vdst=v[3], src0=v[0], src1=v[1], src2=v[2], opsel=0, opsel_hi=0, opsel_hi2=0, neg_hi=4),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i2f(st.vgpr[0][3])
    self.assertAlmostEqual(result, 7.0, places=5)

  def test_v_fma_mix_f32_with_abs_f16_src2_lo(self):
    """V_FMA_MIX_F32 with abs modifier on f16 src2 (lo half). Regression test for sin(1.0) bug."""
    from extra.assembly.amd.test.hw.helpers import f32_to_f16
    f16_neg1 = f32_to_f16(-1.0)  # 0xbc00
    instructions = [
      s_mov_b32(s[0], f2i(0.0)),  # src0 = 0.0 (f32)
      v_mov_b32_e32(v[0], s[0]),
      s_mov_b32(s[1], f2i(1.0)),  # src1 = 1.0 (f32)
      v_mov_b32_e32(v[1], s[1]),
      s_mov_b32(s[2], f16_neg1),  # src2 = -1.0 (f16 in lo)
      v_mov_b32_e32(v[2], s[2]),
      # 0*1 + abs(-1.0) = 1.0; neg_hi=4 means abs on src2, opsel_hi2=1 means src2 is f16
      VOP3P(VOP3POp.V_FMA_MIX_F32, vdst=v[3], src0=v[0], src1=v[1], src2=v[2], opsel=0, opsel_hi=0, opsel_hi2=1, neg_hi=4),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i2f(st.vgpr[0][3])
    self.assertAlmostEqual(result, 1.0, places=5)

  def test_v_fma_mix_f32_with_neg_f16_src2_lo(self):
    """V_FMA_MIX_F32 with neg modifier on f16 src2 (lo half)."""
    from extra.assembly.amd.test.hw.helpers import f32_to_f16
    f16_1 = f32_to_f16(1.0)  # 0x3c00
    instructions = [
      s_mov_b32(s[0], f2i(0.0)),  # src0 = 0.0 (f32)
      v_mov_b32_e32(v[0], s[0]),
      s_mov_b32(s[1], f2i(1.0)),  # src1 = 1.0 (f32)
      v_mov_b32_e32(v[1], s[1]),
      s_mov_b32(s[2], f16_1),  # src2 = 1.0 (f16 in lo)
      v_mov_b32_e32(v[2], s[2]),
      # 0*1 + neg(1.0) = -1.0; neg=4 means neg on src2, opsel_hi2=1 means src2 is f16
      VOP3P(VOP3POp.V_FMA_MIX_F32, vdst=v[3], src0=v[0], src1=v[1], src2=v[2], opsel=0, opsel_hi=0, opsel_hi2=1, neg=4),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i2f(st.vgpr[0][3])
    self.assertAlmostEqual(result, -1.0, places=5)

  def test_v_fma_mix_f32_with_abs_f16_src2_hi(self):
    """V_FMA_MIX_F32 with abs modifier on f16 src2 (hi half)."""
    from extra.assembly.amd.test.hw.helpers import f32_to_f16
    f16_neg1 = f32_to_f16(-1.0)  # 0xbc00
    val = (f16_neg1 << 16) | 0  # -1.0 in hi, 0 in lo
    instructions = [
      s_mov_b32(s[0], f2i(0.0)),
      v_mov_b32_e32(v[0], s[0]),
      s_mov_b32(s[1], f2i(1.0)),
      v_mov_b32_e32(v[1], s[1]),
      s_mov_b32(s[2], val),
      v_mov_b32_e32(v[2], s[2]),
      # opsel=4 selects hi half of src2; neg_hi=4 means abs on src2
      VOP3P(VOP3POp.V_FMA_MIX_F32, vdst=v[3], src0=v[0], src1=v[1], src2=v[2], opsel=4, opsel_hi=0, opsel_hi2=1, neg_hi=4),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i2f(st.vgpr[0][3])
    self.assertAlmostEqual(result, 1.0, places=5)

  def test_v_fma_mixlo_f16(self):
    """V_FMA_MIXLO_F16 writes to low 16 bits of destination."""
    from extra.assembly.amd.test.hw.helpers import _f16
    instructions = [
      s_mov_b32(s[0], f2i(2.0)),
      v_mov_b32_e32(v[0], s[0]),
      s_mov_b32(s[1], f2i(3.0)),
      v_mov_b32_e32(v[1], s[1]),
      s_mov_b32(s[2], f2i(1.0)),
      v_mov_b32_e32(v[2], s[2]),
      s_mov_b32(s[3], 0xdead0000),
      v_mov_b32_e32(v[3], s[3]),
      VOP3P(VOP3POp.V_FMA_MIXLO_F16, vdst=v[3], src0=v[0], src1=v[1], src2=v[2], opsel=0, opsel_hi=0, opsel_hi2=0),
    ]
    st = run_program(instructions, n_lanes=1)
    lo = _f16(st.vgpr[0][3] & 0xffff)
    hi = (st.vgpr[0][3] >> 16) & 0xffff
    self.assertAlmostEqual(lo, 7.0, places=1)
    self.assertEqual(hi, 0xdead, f"hi should be preserved, got 0x{hi:04x}")

  def test_v_fma_mixlo_f16_all_f32_sources(self):
    """V_FMA_MIXLO_F16 with all f32 sources."""
    from extra.assembly.amd.test.hw.helpers import _f16
    instructions = [
      s_mov_b32(s[0], f2i(1.0)),
      v_mov_b32_e32(v[0], s[0]),
      s_mov_b32(s[1], f2i(2.0)),
      v_mov_b32_e32(v[1], s[1]),
      s_mov_b32(s[2], f2i(3.0)),
      v_mov_b32_e32(v[2], s[2]),
      v_mov_b32_e32(v[3], 0),
      VOP3P(VOP3POp.V_FMA_MIXLO_F16, vdst=v[3], src0=v[0], src1=v[1], src2=v[2], opsel=0, opsel_hi=0, opsel_hi2=0),
    ]
    st = run_program(instructions, n_lanes=1)
    lo = _f16(st.vgpr[0][3] & 0xffff)
    # 1*2+3 = 5
    self.assertAlmostEqual(lo, 5.0, places=1)

  def test_v_fma_mixlo_f16_sin_case(self):
    """V_FMA_MIXLO_F16 case from sin kernel."""
    from extra.assembly.amd.test.hw.helpers import _f16
    instructions = [
      s_mov_b32(s[0], 0x3f800000),  # f32 1.0
      v_mov_b32_e32(v[3], s[0]),
      s_mov_b32(s[1], 0xaf05a309),  # f32 tiny negative
      s_mov_b32(s[6], s[1]),
      s_mov_b32(s[2], 0xc0490fdb),  # f32 -π
      v_mov_b32_e32(v[5], s[2]),
      s_mov_b32(s[3], 0x3f800000),
      v_mov_b32_e32(v[3], s[3]),
      VOP3P(VOP3POp.V_FMA_MIXLO_F16, vdst=v[3], src0=v[3], src1=s[6], src2=v[5], opsel=0, opsel_hi=0, opsel_hi2=0),
    ]
    st = run_program(instructions, n_lanes=1)
    lo = _f16(st.vgpr[0][3] & 0xffff)
    self.assertAlmostEqual(lo, -3.14159, delta=0.01)


class TestVOP3P(unittest.TestCase):
  """Tests for VOP3P packed 16-bit operations."""

  def test_v_pk_add_f16_basic(self):
    """V_PK_ADD_F16 adds two packed f16 values."""
    from extra.assembly.amd.test.hw.helpers import _f16
    instructions = [
      s_mov_b32(s[0], 0x40003c00),  # hi=2.0, lo=1.0
      s_mov_b32(s[1], 0x44004200),  # hi=4.0, lo=3.0
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_pk_add_f16(v[2], v[0], v[1], opsel_hi=3, opsel_hi2=1),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][2]
    lo = _f16(result & 0xffff)
    hi = _f16((result >> 16) & 0xffff)
    self.assertAlmostEqual(lo, 4.0, places=2)
    self.assertAlmostEqual(hi, 6.0, places=2)

  def test_v_pk_mul_f16_basic(self):
    """V_PK_MUL_F16 multiplies two packed f16 values."""
    from extra.assembly.amd.test.hw.helpers import _f16
    instructions = [
      s_mov_b32(s[0], 0x42004000),  # hi=3.0, lo=2.0
      s_mov_b32(s[1], 0x45004400),  # hi=5.0, lo=4.0
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_pk_mul_f16(v[2], v[0], v[1], opsel_hi=3, opsel_hi2=1),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][2]
    lo = _f16(result & 0xffff)
    hi = _f16((result >> 16) & 0xffff)
    self.assertAlmostEqual(lo, 8.0, places=1)
    self.assertAlmostEqual(hi, 15.0, places=1)

  def test_v_pk_fma_f16_basic(self):
    """V_PK_FMA_F16: D = A * B + C for packed f16."""
    from extra.assembly.amd.test.hw.helpers import _f16
    instructions = [
      s_mov_b32(s[0], 0x42004000),  # A: hi=3.0, lo=2.0
      s_mov_b32(s[1], 0x45004400),  # B: hi=5.0, lo=4.0
      s_mov_b32(s[2], 0x3c003c00),  # C: hi=1.0, lo=1.0
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_mov_b32_e32(v[2], s[2]),
      v_pk_fma_f16(v[3], v[0], v[1], v[2], opsel_hi=3, opsel_hi2=1),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][3]
    lo = _f16(result & 0xffff)
    hi = _f16((result >> 16) & 0xffff)
    self.assertAlmostEqual(lo, 9.0, places=1)   # 2*4+1
    self.assertAlmostEqual(hi, 16.0, places=0)  # 3*5+1

  def test_v_pk_add_f16_with_inline_constant(self):
    """V_PK_ADD_F16 with inline constant POS_ONE (1.0).
    Inline constants for VOP3P are f16 values in the low 16 bits only.
    hi half of inline constant is 0, so hi result = v0.hi + 0 = 1.0.
    """
    from extra.assembly.amd.test.hw.helpers import _f16
    instructions = [
      s_mov_b32(s[0], 0x3c003c00),  # packed f16: hi=1.0, lo=1.0
      v_mov_b32_e32(v[0], s[0]),
      v_pk_add_f16(v[1], v[0], SrcEnum.POS_ONE, opsel_hi=3, opsel_hi2=1),  # Add inline constant 1.0
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][1]
    lo = _f16(result & 0xffff)
    hi = _f16((result >> 16) & 0xffff)
    # lo = 1.0 + 1.0 = 2.0, hi = 1.0 + 0.0 = 1.0 (inline const hi half is 0)
    self.assertAlmostEqual(lo, 2.0, places=2)
    self.assertAlmostEqual(hi, 1.0, places=2)

  def test_v_pk_mul_f16_with_inline_constant(self):
    """V_PK_MUL_F16 with inline constant POS_TWO (2.0).
    Inline constant has value only in low 16 bits, hi is 0.
    """
    from extra.assembly.amd.test.hw.helpers import _f16
    # v0 = packed (3.0, 4.0), multiply by POS_TWO
    # lo = 3.0 * 2.0 = 6.0, hi = 4.0 * 0.0 = 0.0 (inline const hi is 0)
    instructions = [
      s_mov_b32(s[0], 0x44004200),  # packed f16: hi=4.0, lo=3.0
      v_mov_b32_e32(v[0], s[0]),
      v_pk_mul_f16(v[1], v[0], SrcEnum.POS_TWO, opsel_hi=3, opsel_hi2=1),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][1]
    lo = _f16(result & 0xffff)
    hi = _f16((result >> 16) & 0xffff)
    self.assertAlmostEqual(lo, 6.0, places=1)
    self.assertAlmostEqual(hi, 0.0, places=1)


class TestWMMAF16(unittest.TestCase):
  """Tests for WMMA F16 output variant (V_WMMA_F16_16X16X16_F16).

  Note: RDNA3 WMMA F16 uses 8 VGPRs for accumulator/output (same as F32 variant),
  but values are packed as f16. This differs from RDNA4 which uses 4 VGPRs.
  """

  def test_v_wmma_f16_16x16x16_f16_all_ones(self):
    """V_WMMA_F16_16X16X16_F16 with all ones produces 16.0 in f16."""
    from extra.assembly.amd.test.hw.helpers import _f16
    instructions = []
    instructions.append(s_mov_b32(s[0], 0x3c003c00))  # packed f16 1.0
    # Initialize A matrix in v[16:23] (8 regs)
    for i in range(16, 24):
      instructions.append(v_mov_b32_e32(v[i], s[0]))
    # Initialize B matrix in v[24:31] (8 regs)
    for i in range(24, 32):
      instructions.append(v_mov_b32_e32(v[i], s[0]))
    # Initialize C (accumulator) in v[0:7] to zero (8 regs for RDNA3 WMMA F16)
    for i in range(8):
      instructions.append(v_mov_b32_e32(v[i], 0))
    # WMMA F16: D = A @ B + C
    instructions.append(v_wmma_f16_16x16x16_f16(v[0:7], v[16:23], v[24:31], v[0:7]))
    st = run_program(instructions, n_lanes=32)
    # Result should be 16.0 in f16, stored in lo 16 bits of each VGPR (hi bits are 0)
    for lane in range(32):
      for reg in range(8):
        result = st.vgpr[lane][reg]
        lo = _f16(result & 0xffff)
        self.assertAlmostEqual(lo, 16.0, places=1, msg=f"v[{reg}] lane {lane}: expected 16.0, got {lo}")
        self.assertEqual(result >> 16, 0, msg=f"v[{reg}] lane {lane}: hi bits should be 0")

  def test_v_wmma_f16_16x16x16_f16_with_accumulator(self):
    """V_WMMA_F16_16X16X16_F16 with non-zero accumulator."""
    from extra.assembly.amd.test.hw.helpers import _f16
    instructions = []
    instructions.append(s_mov_b32(s[0], 0x3c003c00))  # packed f16 1.0
    instructions.append(s_mov_b32(s[1], 0x4500))  # f16 5.0 in lo bits only
    # Initialize A matrix in v[16:23] (8 regs)
    for i in range(16, 24):
      instructions.append(v_mov_b32_e32(v[i], s[0]))
    # Initialize B matrix in v[24:31] (8 regs)
    for i in range(24, 32):
      instructions.append(v_mov_b32_e32(v[i], s[0]))
    # Initialize C (accumulator) in v[0:7] to 5.0 in lo bits (8 regs for RDNA3 WMMA F16)
    for i in range(8):
      instructions.append(v_mov_b32_e32(v[i], s[1]))
    # WMMA F16: D = A @ B + C
    instructions.append(v_wmma_f16_16x16x16_f16(v[0:7], v[16:23], v[24:31], v[0:7]))
    st = run_program(instructions, n_lanes=32)
    # Result should be 16.0 + 5.0 = 21.0 in f16, stored in lo 16 bits (hi bits are 0)
    for lane in range(32):
      for reg in range(8):
        result = st.vgpr[lane][reg]
        lo = _f16(result & 0xffff)
        self.assertAlmostEqual(lo, 21.0, places=0, msg=f"v[{reg}] lane {lane}: expected 21.0, got {lo}")
        self.assertEqual(result >> 16, 0, msg=f"v[{reg}] lane {lane}: hi bits should be 0")

  def test_v_wmma_f16_16x16x16_f16_high_registers(self):
    """V_WMMA_F16_16X16X16_F16 with high register indices.

    Regression test: WMMA was using static register indices instead of dynamic.
    This test uses v[64:71] for A, v[80:87] for B, v[96:103] for C/D.
    """
    from extra.assembly.amd.test.hw.helpers import _f16
    instructions = []
    instructions.append(s_mov_b32(s[0], 0x3c003c00))  # packed f16 1.0
    # Initialize A matrix in v[64:71] (8 regs)
    for i in range(64, 72):
      instructions.append(v_mov_b32_e32(v[i], s[0]))
    # Initialize B matrix in v[80:87] (8 regs)
    for i in range(80, 88):
      instructions.append(v_mov_b32_e32(v[i], s[0]))
    # Initialize C (accumulator) in v[96:103] to zero (8 regs for RDNA3 WMMA F16)
    for i in range(96, 104):
      instructions.append(v_mov_b32_e32(v[i], 0))
    # WMMA F16: D = A @ B + C, result in v[96:103]
    instructions.append(v_wmma_f16_16x16x16_f16(v[96:103], v[64:71], v[80:87], v[96:103]))
    # Copy results to v[0:7] for checking
    for i in range(8):
      instructions.append(v_mov_b32_e32(v[i], v[96+i]))
    st = run_program(instructions, n_lanes=32)
    # Result should be 16.0 in f16, stored in lo 16 bits (hi bits are 0)
    for lane in range(32):
      for reg in range(8):
        result = st.vgpr[lane][reg]
        lo = _f16(result & 0xffff)
        self.assertAlmostEqual(lo, 16.0, places=1, msg=f"v[{reg}] lane {lane}: expected 16.0, got {lo}")
        self.assertEqual(result >> 16, 0, msg=f"v[{reg}] lane {lane}: hi bits should be 0")


class TestWMMA(unittest.TestCase):
  """Tests for WMMA (Wave Matrix Multiply-Accumulate) instructions with F32 output."""

  def test_v_wmma_f32_16x16x16_f16_all_ones(self):
    """V_WMMA_F32_16X16X16_F16 with all ones produces 16.0."""
    instructions = []
    instructions.append(s_mov_b32(s[0], 0x3c003c00))  # packed f16 1.0
    for i in range(16, 32):
      instructions.append(v_mov_b32_e32(v[i], s[0]))
    for i in range(8):
      instructions.append(v_mov_b32_e32(v[i], 0))
    instructions.append(v_wmma_f32_16x16x16_f16(v[0:7], v[16:23], v[24:31], v[0:7]))
    st = run_program(instructions, n_lanes=32)
    expected = f2i(16.0)
    for lane in range(32):
      for reg in range(8):
        result = st.vgpr[lane][reg]
        self.assertEqual(result, expected, f"v[{reg}] lane {lane}: expected 16.0, got {i2f(result)}")

  def test_v_wmma_f32_16x16x16_f16_with_accumulator(self):
    """V_WMMA_F32_16X16X16_F16 with non-zero accumulator."""
    instructions = []
    instructions.append(s_mov_b32(s[0], 0x3c003c00))
    instructions.append(s_mov_b32(s[1], f2i(5.0)))
    for i in range(16, 32):
      instructions.append(v_mov_b32_e32(v[i], s[0]))
    for i in range(8):
      instructions.append(v_mov_b32_e32(v[i], s[1]))
    instructions.append(v_wmma_f32_16x16x16_f16(v[0:7], v[16:23], v[24:31], v[0:7]))
    st = run_program(instructions, n_lanes=32)
    expected = f2i(21.0)  # 16 + 5
    for lane in range(32):
      for reg in range(8):
        result = st.vgpr[lane][reg]
        self.assertEqual(result, expected, f"v[{reg}] lane {lane}: expected 21.0, got {i2f(result)}")

  def test_v_wmma_f32_16x16x16_f16_high_registers(self):
    """V_WMMA_F32_16X16X16_F16 with high register indices.

    Regression test: WMMA was using static register indices instead of dynamic,
    causing incorrect results when registers weren't at the default positions.
    This test uses v[64:71] for A, v[80:87] for B, v[96:103] for C/D.
    """
    instructions = []
    instructions.append(s_mov_b32(s[0], 0x3c003c00))  # packed f16 1.0
    # Initialize A matrix in v[64:71]
    for i in range(64, 72):
      instructions.append(v_mov_b32_e32(v[i], s[0]))
    # Initialize B matrix in v[80:87]
    for i in range(80, 88):
      instructions.append(v_mov_b32_e32(v[i], s[0]))
    # Initialize C (accumulator) in v[96:103] to zero
    for i in range(96, 104):
      instructions.append(v_mov_b32_e32(v[i], 0))
    # WMMA: D = A @ B + C, result in v[96:103]
    instructions.append(v_wmma_f32_16x16x16_f16(v[96:103], v[64:71], v[80:87], v[96:103]))
    # Copy results to v[0:7] for checking
    for i in range(8):
      instructions.append(v_mov_b32_e32(v[i], v[96+i]))
    st = run_program(instructions, n_lanes=32)
    expected = f2i(16.0)
    for lane in range(32):
      for reg in range(8):
        result = st.vgpr[lane][reg]
        self.assertEqual(result, expected, f"v[{reg}] lane {lane}: expected 16.0, got {i2f(result)}")


class TestWMMABF16(unittest.TestCase):
  """Tests for WMMA BF16 instructions."""

  def test_v_wmma_f32_16x16x16_bf16_all_ones(self):
    """V_WMMA_F32_16X16X16_BF16 with all ones produces 16.0."""
    instructions = []
    # BF16 1.0 = 0x3f80, packed = 0x3f803f80
    instructions.append(s_mov_b32(s[0], 0x3f803f80))
    for i in range(16, 32):
      instructions.append(v_mov_b32_e32(v[i], s[0]))
    for i in range(8):
      instructions.append(v_mov_b32_e32(v[i], 0))
    instructions.append(v_wmma_f32_16x16x16_bf16(v[0:7], v[16:23], v[24:31], v[0:7]))
    st = run_program(instructions, n_lanes=32)
    expected = f2i(16.0)
    for lane in range(32):
      for reg in range(8):
        result = st.vgpr[lane][reg]
        self.assertEqual(result, expected, f"v[{reg}] lane {lane}: expected 16.0, got {i2f(result)}")

  def test_v_wmma_f32_16x16x16_bf16_with_accumulator(self):
    """V_WMMA_F32_16X16X16_BF16 with non-zero accumulator."""
    instructions = []
    # BF16 1.0 = 0x3f80, packed = 0x3f803f80
    instructions.append(s_mov_b32(s[0], 0x3f803f80))
    instructions.append(s_mov_b32(s[1], f2i(5.0)))
    for i in range(16, 32):
      instructions.append(v_mov_b32_e32(v[i], s[0]))
    for i in range(8):
      instructions.append(v_mov_b32_e32(v[i], s[1]))
    instructions.append(v_wmma_f32_16x16x16_bf16(v[0:7], v[16:23], v[24:31], v[0:7]))
    st = run_program(instructions, n_lanes=32)
    expected = f2i(21.0)  # 16 + 5
    for lane in range(32):
      for reg in range(8):
        result = st.vgpr[lane][reg]
        self.assertEqual(result, expected, f"v[{reg}] lane {lane}: expected 21.0, got {i2f(result)}")


class TestSpecialOps(unittest.TestCase):
  """Tests for special operations (SAD, PERM, DOT2)."""

  def test_v_sad_u8_basic(self):
    """V_SAD_U8 computes sum of absolute differences."""
    instructions = [
      s_mov_b32(s[0], 0x04030201),  # bytes: 1, 2, 3, 4
      s_mov_b32(s[1], 0x05040302),  # bytes: 2, 3, 4, 5
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_mov_b32_e32(v[2], 0),
      v_sad_u8(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    # |1-2| + |2-3| + |3-4| + |4-5| = 1 + 1 + 1 + 1 = 4
    self.assertEqual(st.vgpr[0][3], 4)

  def test_v_sad_u8_identical_bytes(self):
    """V_SAD_U8 with identical inputs returns accumulator."""
    instructions = [
      s_mov_b32(s[0], 0x04030201),
      v_mov_b32_e32(v[0], s[0]),
      s_mov_b32(s[1], 10),
      v_mov_b32_e32(v[2], s[1]),
      v_sad_u8(v[3], v[0], v[0], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    # Same inputs -> SAD = 0, result = accumulator = 10
    self.assertEqual(st.vgpr[0][3], 10)

  def test_v_sad_u16_basic(self):
    """V_SAD_U16 computes sum of absolute differences of u16 pairs."""
    instructions = [
      s_mov_b32(s[0], 0x00030001),  # hi=3, lo=1
      s_mov_b32(s[1], 0x00050002),  # hi=5, lo=2
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_mov_b32_e32(v[2], 0),
      v_sad_u16(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    # |1-2| + |3-5| = 1 + 2 = 3
    self.assertEqual(st.vgpr[0][3], 3)

  def test_v_sad_u32_basic(self):
    """V_SAD_U32 computes absolute difference of u32 values."""
    instructions = [
      s_mov_b32(s[0], 100),
      s_mov_b32(s[1], 70),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_mov_b32_e32(v[2], 0),
      v_sad_u32(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    # |100-70| = 30
    self.assertEqual(st.vgpr[0][3], 30)

  def test_v_msad_u8_masked(self):
    """V_MSAD_U8 masked SAD operation."""
    instructions = [
      s_mov_b32(s[0], 0x04030201),
      s_mov_b32(s[1], 0x05040302),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_mov_b32_e32(v[2], 0),
      v_msad_u8(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    # V_MSAD_U8 skips bytes where src0 is 0
    # Since no bytes are 0, result same as V_SAD_U8 = 4
    self.assertEqual(st.vgpr[0][3], 4)

  def test_v_perm_b32_select_bytes(self):
    """V_PERM_B32 selects bytes from two sources.

    V_PERM_B32 concatenates {S1, S0} as a 64-bit value with S1 in low 32 bits.
    Selector byte values 0-3 select from S1, values 4-7 select from S0.
    """
    instructions = [
      s_mov_b32(s[0], 0x44332211),  # src0: bytes 4-7 in 64-bit view
      s_mov_b32(s[1], 0x88776655),  # src1: bytes 0-3 in 64-bit view
      s_mov_b32(s[2], 0x07060504),  # select bytes 4,5,6,7 (from src0)
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_perm_b32(v[2], v[0], v[1], s[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][2], 0x44332211)

  def test_v_dot2_f32_bf16_basic(self):
    """V_DOT2_F32_BF16 computes dot product of bf16 pairs."""
    # bf16 1.0 = 0x3f80, bf16 2.0 = 0x4000
    instructions = [
      s_mov_b32(s[0], 0x3f803f80),  # packed bf16: lo=1.0, hi=1.0
      s_mov_b32(s[1], 0x40003f80),  # packed bf16: lo=1.0, hi=2.0
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_mov_b32_e32(v[2], 0),
      v_dot2_f32_bf16(v[3], v[0], v[1], v[2], opsel_hi=3, opsel_hi2=1),
    ]
    st = run_program(instructions, n_lanes=1)
    # 1.0*1.0 + 1.0*2.0 + 0 = 3.0
    result = i2f(st.vgpr[0][3])
    self.assertAlmostEqual(result, 3.0, places=4)


class TestPackedMixedSigns(unittest.TestCase):
  """Tests for packed operations with mixed sign values."""

  def test_pk_add_f16_mixed_signs(self):
    """V_PK_ADD_F16 with mixed positive/negative values."""
    from extra.assembly.amd.test.hw.helpers import _f16
    instructions = [
      s_mov_b32(s[0], 0xc0003c00),  # packed: hi=-2.0, lo=1.0
      s_mov_b32(s[1], 0x3c003c00),  # packed: hi=1.0, lo=1.0
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_pk_add_f16(v[2], v[0], v[1], opsel_hi=3, opsel_hi2=1),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][2]
    lo = _f16(result & 0xffff)
    hi = _f16((result >> 16) & 0xffff)
    self.assertAlmostEqual(lo, 2.0, places=2)   # 1.0 + 1.0
    self.assertAlmostEqual(hi, -1.0, places=2)  # -2.0 + 1.0

  def test_pk_mul_f16_zero(self):
    """V_PK_MUL_F16 with zero."""
    from extra.assembly.amd.test.hw.helpers import _f16
    instructions = [
      s_mov_b32(s[0], 0x40004000),  # packed: 2.0, 2.0
      s_mov_b32(s[1], 0x00000000),  # packed: 0.0, 0.0
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_pk_mul_f16(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][2]
    self.assertEqual(result, 0x00000000, "2.0 * 0.0 should be 0.0")


if __name__ == '__main__':
  unittest.main()
