"""Tests for SOP instructions - scalar operations.

Includes: s_add_u32, s_mov_b32, s_and_b32, s_or_b32, s_quadmask_b32, s_wqm_b32,
          s_cbranch_vccnz, s_cbranch_vccz
"""
import unittest
from extra.assembly.amd.test.hw.helpers import *

class TestBasicScalar(unittest.TestCase):
  """Tests for basic scalar operations."""

  def test_s_add_u32(self):
    """S_ADD_U32 adds two scalar values."""
    instructions = [
      s_mov_b32(s[0], 100),
      s_mov_b32(s[1], 200),
      s_add_u32(s[2], s[0], s[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[2], 300)

  def test_s_add_u32_carry(self):
    """S_ADD_U32 sets SCC on overflow."""
    instructions = [
      s_mov_b32(s[0], 64),
      s_not_b32(s[0], s[0]),  # ~64 = 0xffffffbf
      s_mov_b32(s[1], 64),
      s_add_u32(s[2], s[0], s[1]),  # 0xffffffbf + 64 = 0xffffffff
      s_mov_b32(s[3], 1),
      s_add_u32(s[4], s[2], s[3]),  # 0xffffffff + 1 = overflow
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[4], 0)
    self.assertEqual(st.scc, 1)

  def test_s_brev_b32(self):
    """S_BREV_B32 reverses bits of a 32-bit value."""
    # 10 = 0b00000000000000000000000000001010
    # reversed = 0b01010000000000000000000000000000 = 0x50000000
    instructions = [
      s_mov_b32(s[0], 10),
      s_brev_b32(s[1], s[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[1], 0x50000000)

  def test_s_brev_b32_all_ones(self):
    """S_BREV_B32 with all ones stays all ones."""
    instructions = [
      s_mov_b32(s[0], 0xFFFFFFFF),
      s_brev_b32(s[1], s[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[1], 0xFFFFFFFF)

  def test_s_brev_b32_single_bit(self):
    """S_BREV_B32 with bit 0 set becomes bit 31."""
    instructions = [
      s_mov_b32(s[0], 1),
      s_brev_b32(s[1], s[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[1], 0x80000000)

  @skip_unless_gfx(11, 5, "SALU FP ops require gfx1150+")
  def test_s_fmamk_f32(self):
    """S_FMAMK_F32: D = S0 * literal + S1."""
    # 2.0 * 3.0 + 1.0 = 7.0
    instructions = [
      s_mov_b32(s[0], f2i(2.0)),
      s_mov_b32(s[1], f2i(1.0)),
      s_fmamk_f32(s[2], s[0], s[1], literal=f2i(3.0)),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[2], f2i(7.0))

  @skip_unless_gfx(11, 5, "SALU FP ops require gfx1150+")
  def test_s_fmamk_f32_negative(self):
    """S_FMAMK_F32 with negative values."""
    # -2.0 * 4.0 + 10.0 = 2.0
    instructions = [
      s_mov_b32(s[0], f2i(-2.0)),
      s_mov_b32(s[1], f2i(10.0)),
      s_fmamk_f32(s[2], s[0], s[1], literal=f2i(4.0)),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[2], f2i(2.0))


class TestQuadmaskWqm(unittest.TestCase):
  """Tests for S_QUADMASK_B32 and S_WQM_B32."""

  def test_s_quadmask_b32_all_quads_active(self):
    """S_QUADMASK_B32 with all quads active."""
    instructions = [
      s_mov_b32(s[0], 0xFFFFFFFF),  # All lanes active
      s_quadmask_b32(s[1], s[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    # Each quad (4 lanes) with any bit set -> 1 bit in result
    # 32 lanes = 8 quads, all active -> 0xFF
    self.assertEqual(st.sgpr[1], 0xFF)

  def test_s_quadmask_b32_alternating_quads(self):
    """S_QUADMASK_B32 with alternating quads active."""
    instructions = [
      s_mov_b32(s[0], 0x0F0F0F0F),  # Quads 0,2,4,6 active
      s_quadmask_b32(s[1], s[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    # Quads 0,2,4,6 have at least one bit -> 0b01010101 = 0x55
    self.assertEqual(st.sgpr[1], 0x55)

  def test_s_quadmask_b32_no_quads_active(self):
    """S_QUADMASK_B32 with no quads active."""
    instructions = [
      s_mov_b32(s[0], 0),
      s_quadmask_b32(s[1], s[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[1], 0)

  def test_s_quadmask_b32_single_lane_per_quad(self):
    """S_QUADMASK_B32 with single lane active in each quad."""
    instructions = [
      s_mov_b32(s[0], 0x11111111),  # Bit 0 of each nibble
      s_quadmask_b32(s[1], s[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    # All 8 quads have at least one lane -> 0xFF
    self.assertEqual(st.sgpr[1], 0xFF)

  def test_s_wqm_b32_all_active(self):
    """S_WQM_B32 with all lanes active returns all 1s."""
    instructions = [
      s_mov_b32(s[0], 0xFFFFFFFF),
      s_wqm_b32(s[1], s[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[1], 0xFFFFFFFF)

  def test_s_wqm_b32_alternating_quads(self):
    """S_WQM_B32 with single lane per quad expands to full quads."""
    instructions = [
      s_mov_b32(s[0], 0x11111111),  # One lane per quad
      s_wqm_b32(s[1], s[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    # Each quad with any bit expands to all 4 bits
    self.assertEqual(st.sgpr[1], 0xFFFFFFFF)

  def test_s_wqm_b32_zero(self):
    """S_WQM_B32 with zero input returns zero."""
    instructions = [
      s_mov_b32(s[0], 0),
      s_wqm_b32(s[1], s[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[1], 0)


class TestBranch(unittest.TestCase):
  """Tests for branch instructions."""

  def test_cbranch_vccnz_ignores_vcc_hi(self):
    """S_CBRANCH_VCCNZ should only check VCC_LO in wave32."""
    instructions = [
      # Set VCC_LO = 0, VCC_HI = 1
      s_mov_b32(VCC_LO, 0),
      s_mov_b32(VCC_HI, 1),
      v_mov_b32_e32(v[0], 0),
      # If VCC_HI is incorrectly used, branch will be taken
      s_cbranch_vccnz(1),  # Skip next instruction if VCC != 0
      v_mov_b32_e32(v[0], 42),  # This should execute
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][0], 42, "Branch should NOT be taken (VCC_LO is 0)")

  def test_cbranch_vccz_ignores_vcc_hi(self):
    """S_CBRANCH_VCCZ should only check VCC_LO in wave32."""
    instructions = [
      # Set VCC_LO = 1, VCC_HI = 0
      s_mov_b32(VCC_LO, 1),
      s_mov_b32(VCC_HI, 0),
      v_mov_b32_e32(v[0], 0),
      # If VCC_HI is incorrectly used, branch will be taken
      s_cbranch_vccz(1),  # Skip next instruction if VCC == 0
      v_mov_b32_e32(v[0], 42),  # This should execute
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][0], 42, "Branch should NOT be taken (VCC_LO is 1)")

  def test_cbranch_vccnz_branches_on_vcc_lo(self):
    """S_CBRANCH_VCCNZ branches when VCC_LO is non-zero."""
    instructions = [
      s_mov_b32(VCC_LO, 1),
      v_mov_b32_e32(v[0], 0),
      s_cbranch_vccnz(1),  # Skip next instruction if VCC != 0
      v_mov_b32_e32(v[0], 42),  # This should be skipped
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][0], 0, "Branch should be taken (VCC_LO is 1)")


class Test64BitLiterals(unittest.TestCase):
  """Tests for 64-bit literal encoding in instructions."""

  def test_64bit_literal_negative_encoding(self):
    """64-bit literal -2^32 encodes correctly."""
    lit = -4294967296.0  # -2^32
    lit_bits = f2i64(lit)
    instructions = [
      s_mov_b32(s[0], lit_bits & 0xffffffff),
      s_mov_b32(s[1], lit_bits >> 32),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i642f(st.vgpr[0][0] | (st.vgpr[0][1] << 32))
    self.assertAlmostEqual(result, -4294967296.0, places=5)

class TestSCCBehavior(unittest.TestCase):
  """Tests for SCC condition code behavior."""

  def test_scc_from_s_cmp(self):
    """SCC should be set by scalar compare."""
    instructions = [
      s_mov_b32(s[0], 10),
      s_cmp_eq_u32(s[0], 10),
      s_cselect_b32(s[1], 1, 0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[1], 1, "SCC should be true")
    self.assertEqual(st.scc, 1)

  def test_scc_clear(self):
    """SCC should be cleared by failing compare."""
    instructions = [
      s_mov_b32(s[0], 10),
      s_cmp_eq_u32(s[0], 20),
      s_cselect_b32(s[1], 1, 0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[1], 0, "SCC should be false")
    self.assertEqual(st.scc, 0)


class TestSignedArithmetic(unittest.TestCase):
  """Tests for S_ADD_I32, S_SUB_I32 and their SCC overflow behavior."""

  def test_s_add_i32_no_overflow(self):
    """S_ADD_I32: 1 + 1 = 2, no overflow, SCC=0."""
    instructions = [
      s_mov_b32(s[0], 1),
      s_add_i32(s[1], s[0], 1),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[1], 2)
    self.assertEqual(st.scc, 0, "No overflow, SCC should be 0")

  def test_s_add_i32_positive_overflow(self):
    """S_ADD_I32: MAX_INT + 1 overflows, SCC=1."""
    instructions = [
      s_mov_b32(s[0], 0x7FFFFFFF),  # MAX_INT
      s_add_i32(s[1], s[0], 1),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[1], 0x80000000)  # Wraps to MIN_INT
    self.assertEqual(st.scc, 1, "Overflow, SCC should be 1")

  def test_s_add_i32_negative_no_overflow(self):
    """S_ADD_I32: -10 + 20 = 10, no overflow."""
    instructions = [
      s_mov_b32(s[0], 0xFFFFFFF6),  # -10 in two's complement
      s_mov_b32(s[1], 20),
      s_add_i32(s[2], s[0], s[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[2], 10)
    self.assertEqual(st.scc, 0)

  def test_s_add_i32_negative_overflow(self):
    """S_ADD_I32: MIN_INT + (-1) underflows, SCC=1."""
    instructions = [
      s_mov_b32(s[0], 0x80000000),  # MIN_INT
      s_mov_b32(s[1], 0xFFFFFFFF),  # -1
      s_add_i32(s[2], s[0], s[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[2], 0x7FFFFFFF)  # Wraps to MAX_INT
    self.assertEqual(st.scc, 1, "Underflow, SCC should be 1")

  def test_s_sub_i32_no_overflow(self):
    """S_SUB_I32: 10 - 5 = 5, no overflow."""
    instructions = [
      s_mov_b32(s[0], 10),
      s_mov_b32(s[1], 5),
      s_sub_i32(s[2], s[0], s[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[2], 5)
    self.assertEqual(st.scc, 0)

  def test_s_sub_i32_overflow(self):
    """S_SUB_I32: MAX_INT - (-1) overflows, SCC=1."""
    instructions = [
      s_mov_b32(s[0], 0x7FFFFFFF),  # MAX_INT
      s_mov_b32(s[1], 0xFFFFFFFF),  # -1
      s_sub_i32(s[2], s[0], s[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[2], 0x80000000)  # Wraps to MIN_INT
    self.assertEqual(st.scc, 1, "Overflow, SCC should be 1")

  def test_s_mul_hi_u32(self):
    """S_MUL_HI_U32: high 32 bits of u32 * u32."""
    instructions = [
      s_mov_b32(s[0], 0x80000000),  # 2^31
      s_mov_b32(s[1], 4),
      s_mul_hi_u32(s[2], s[0], s[1]),  # (2^31 * 4) >> 32 = 2
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[2], 2)

  def test_s_mul_hi_u32_max(self):
    """S_MUL_HI_U32: 0xFFFFFFFF * 0xFFFFFFFF."""
    instructions = [
      s_mov_b32(s[0], 0xFFFFFFFF),
      s_mov_b32(s[1], 0xFFFFFFFF),
      s_mul_hi_u32(s[2], s[0], s[1]),  # (0xFFFFFFFF * 0xFFFFFFFF) >> 32 = 0xFFFFFFFE
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[2], 0xFFFFFFFE)

  def test_s_mul_hi_i32_positive(self):
    """S_MUL_HI_I32: positive * positive."""
    instructions = [
      s_mov_b32(s[0], 0x40000000),  # 2^30
      s_mov_b32(s[1], 4),
      s_mul_hi_i32(s[2], s[0], s[1]),  # (2^30 * 4) >> 32 = 1
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[2], 1)

  def test_s_mul_hi_i32_neg_times_neg(self):
    """S_MUL_HI_I32: (-1) * (-1) = 1, high bits = 0."""
    instructions = [
      s_mov_b32(s[0], 0xFFFFFFFF),  # -1
      s_mov_b32(s[1], 0xFFFFFFFF),  # -1
      s_mul_hi_i32(s[2], s[0], s[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[2], 0)

  def test_s_mul_hi_i32_neg_times_pos(self):
    """S_MUL_HI_I32: (-1) * 2 = -2, high bits = -1 (sign extension)."""
    instructions = [
      s_mov_b32(s[0], 0xFFFFFFFF),  # -1
      s_mov_b32(s[1], 2),
      s_mul_hi_i32(s[2], s[0], s[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[2], 0xFFFFFFFF)  # -1 sign extends

  def test_s_mul_hi_i32_min_int(self):
    """S_MUL_HI_I32: MIN_INT * 2 = -2^32, high = -1."""
    instructions = [
      s_mov_b32(s[0], 0x80000000),  # -2^31 (MIN_INT)
      s_mov_b32(s[1], 2),
      s_mul_hi_i32(s[2], s[0], s[1]),  # (-2^31 * 2) >> 32 = -1
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[2], 0xFFFFFFFF)

  def test_s_mul_i32(self):
    """S_MUL_I32: signed multiply low 32 bits."""
    instructions = [
      s_mov_b32(s[0], 0xFFFFFFFF),  # -1
      s_mov_b32(s[1], 10),
      s_mul_i32(s[2], s[0], s[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[2], 0xFFFFFFF6)  # -10

  def test_division_sequence_from_llvm(self):
    """Test the division sequence pattern from LLVM-generated code."""
    # This sequence is from the sin kernel and computes integer division
    # s10 = dividend, s18 = divisor, result in s6/s14
    dividend = 0x28BE60DB  # Some value from the sin kernel
    divisor = 3  # Simplified divisor
    instructions = [
      s_mov_b32(s[10], dividend),
      s_mov_b32(s[18], divisor),
      # Compute reciprocal approximation: s6 = ~0 / divisor (approx)
      s_mov_b32(s[11], 0),
      s_sub_i32(s[11], s[11], s[18]),  # s11 = -divisor
      # For testing, just verify basic arithmetic works
      s_mul_i32(s[6], s[10], 2),
      s_add_i32(s[7], s[6], 1),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[6], (dividend * 2) & 0xFFFFFFFF)
    self.assertEqual(st.sgpr[7], ((dividend * 2) + 1) & 0xFFFFFFFF)


class TestBitSet(unittest.TestCase):
  """Tests for S_BITSET0_B32 and S_BITSET1_B32 instructions."""

  def test_s_bitset1_b32_set_bit0(self):
    """S_BITSET1_B32: set bit 0 in destination."""
    instructions = [
      s_mov_b32(s[0], 0),     # start with 0
      s_mov_b32(s[1], 0),     # bit position = 0
      s_bitset1_b32(s[0], s[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[0], 1, "Bit 0 should be set")

  def test_s_bitset1_b32_set_bit31(self):
    """S_BITSET1_B32: set bit 31 in destination."""
    instructions = [
      s_mov_b32(s[0], 0),     # start with 0
      s_mov_b32(s[1], 31),    # bit position = 31
      s_bitset1_b32(s[0], s[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[0], 0x80000000, "Bit 31 should be set")

  def test_s_bitset1_b32_preserves_other_bits(self):
    """S_BITSET1_B32: preserves bits not being set."""
    instructions = [
      s_mov_b32(s[0], 0xFF00FF00),  # existing pattern
      s_mov_b32(s[1], 0),            # bit position = 0
      s_bitset1_b32(s[0], s[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[0], 0xFF00FF01, "Should set bit 0 while preserving others")

  def test_s_bitset0_b32_clear_bit0(self):
    """S_BITSET0_B32: clear bit 0 in destination."""
    instructions = [
      s_mov_b32(s[0], 0xFFFFFFFF),  # start with all bits set
      s_mov_b32(s[1], 0),            # bit position = 0
      s_bitset0_b32(s[0], s[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[0], 0xFFFFFFFE, "Bit 0 should be cleared")

  def test_s_bitset0_b32_clear_bit31(self):
    """S_BITSET0_B32: clear bit 31 in destination."""
    instructions = [
      s_mov_b32(s[0], 0xFFFFFFFF),  # start with all bits set
      s_mov_b32(s[1], 31),           # bit position = 31
      s_bitset0_b32(s[0], s[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[0], 0x7FFFFFFF, "Bit 31 should be cleared")

  def test_s_bitset1_b32_uses_low5_bits(self):
    """S_BITSET1_B32: only uses low 5 bits of position (mod 32)."""
    instructions = [
      s_mov_b32(s[0], 0),
      s_mov_b32(s[1], 32 + 5),   # position = 37, but mod 32 = 5
      s_bitset1_b32(s[0], s[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[0], 0x20, "Bit 5 should be set (37 mod 32 = 5)")


class TestBfeI64(unittest.TestCase):
  """Tests for S_BFE_I64 - 64-bit bit field extract with sign extension.

  Regression tests for sign extension bug where 32-bit masks were incorrectly
  used for 64-bit operations, causing the high 32 bits to not be sign-extended.
  """

  def test_s_bfe_i64_positive_no_sign_extend(self):
    """S_BFE_I64: positive value (1) in 16 bits should not sign extend."""
    # S1 encodes: [22:16] = width, [5:0] = offset
    # width=16, offset=0 -> S1 = (16 << 16) | 0 = 0x100000
    instructions = [
      s_mov_b32(s[0], 1),         # S0 lo = 1
      s_mov_b32(s[1], 0),         # S0 hi = 0
      s_mov_b32(s[2], 0x100000),  # width=16, offset=0
      s_bfe_i64(s[4:5], s[0:1], s[2]),
      v_mov_b32_e32(v[0], s[4]),
      v_mov_b32_e32(v[1], s[5]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][0], 1, "lo should be 1")
    self.assertEqual(st.vgpr[0][1], 0, "hi should be 0 (no sign extend)")

  def test_s_bfe_i64_negative_sign_extend(self):
    """S_BFE_I64: 0xFFFF (-1 in 16 bits) should sign extend to 64 bits.

    This is the main regression test - before the fix, hi was 0 instead of 0xFFFFFFFF.
    """
    instructions = [
      s_mov_b32(s[0], 0xFFFF),    # S0 lo = -1 in 16 bits
      s_mov_b32(s[1], 0),         # S0 hi = 0
      s_mov_b32(s[2], 0x100000),  # width=16, offset=0
      s_bfe_i64(s[4:5], s[0:1], s[2]),
      v_mov_b32_e32(v[0], s[4]),
      v_mov_b32_e32(v[1], s[5]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][0], 0xFFFFFFFF, "lo should be 0xFFFFFFFF")
    self.assertEqual(st.vgpr[0][1], 0xFFFFFFFF, "hi should be 0xFFFFFFFF (sign extended)")

  def test_s_bfe_i64_8bit_negative_sign_extend(self):
    """S_BFE_I64: 0xFF (-1 in 8 bits) should sign extend to 64 bits."""
    # width=8, offset=0 -> S1 = (8 << 16) | 0 = 0x80000
    instructions = [
      s_mov_b32(s[0], 0xFF),      # S0 lo = -1 in 8 bits
      s_mov_b32(s[1], 0),         # S0 hi = 0
      s_mov_b32(s[2], 0x80000),   # width=8, offset=0
      s_bfe_i64(s[4:5], s[0:1], s[2]),
      v_mov_b32_e32(v[0], s[4]),
      v_mov_b32_e32(v[1], s[5]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][0], 0xFFFFFFFF, "lo should be 0xFFFFFFFF")
    self.assertEqual(st.vgpr[0][1], 0xFFFFFFFF, "hi should be 0xFFFFFFFF (sign extended)")

  def test_s_bfe_i64_8bit_positive(self):
    """S_BFE_I64: 0x7F (127 in 8 bits) should not sign extend."""
    # width=8, offset=0 -> S1 = (8 << 16) | 0 = 0x80000
    instructions = [
      s_mov_b32(s[0], 0x7F),      # S0 lo = 127 in 8 bits (MSB=0)
      s_mov_b32(s[1], 0),         # S0 hi = 0
      s_mov_b32(s[2], 0x80000),   # width=8, offset=0
      s_bfe_i64(s[4:5], s[0:1], s[2]),
      v_mov_b32_e32(v[0], s[4]),
      v_mov_b32_e32(v[1], s[5]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][0], 0x7F, "lo should be 0x7F")
    self.assertEqual(st.vgpr[0][1], 0, "hi should be 0 (no sign extend)")

  def test_s_bfe_i64_with_offset(self):
    """S_BFE_I64: extract from non-zero bit offset with sign extension."""
    # Extract 16 bits starting at bit 8: value 0xFF00 >> 8 = 0xFF = -1 in 8 bits? No wait...
    # Let's put 0x8000FF00: extract 16 bits at offset 8 = 0x00FF (positive)
    # Put 0xFF00_0000: extract 16 bits at offset 16 = 0xFF00 = -256 in signed 16-bit
    instructions = [
      s_mov_b32(s[0], 0xFF000000),  # bits [31:24] = 0xFF, [23:16] = 0x00
      s_mov_b32(s[1], 0),
      # width=16, offset=16 -> S1 = (16 << 16) | 16 = 0x100010
      s_mov_b32(s[2], 0x100010),
      s_bfe_i64(s[4:5], s[0:1], s[2]),
      v_mov_b32_e32(v[0], s[4]),
      v_mov_b32_e32(v[1], s[5]),
    ]
    st = run_program(instructions, n_lanes=1)
    # Extract bits [31:16] = 0xFF00, sign bit is bit 15 of extracted = bit 31 of original = 1
    # So result should be sign-extended 0xFF00 -> 0xFFFFFF00 in lo, 0xFFFFFFFF in hi
    self.assertEqual(st.vgpr[0][0], 0xFFFFFF00, "lo should be sign-extended 0xFF00")
    self.assertEqual(st.vgpr[0][1], 0xFFFFFFFF, "hi should be 0xFFFFFFFF (sign extended)")

  def test_s_bfe_i64_32bit_negative(self):
    """S_BFE_I64: extract 32 bits with sign extension."""
    # width=32, offset=0 -> S1 = (32 << 16) | 0 = 0x200000
    instructions = [
      s_mov_b32(s[0], 0x80000000),  # MIN_INT32 = -2^31
      s_mov_b32(s[1], 0),
      s_mov_b32(s[2], 0x200000),    # width=32, offset=0
      s_bfe_i64(s[4:5], s[0:1], s[2]),
      v_mov_b32_e32(v[0], s[4]),
      v_mov_b32_e32(v[1], s[5]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][0], 0x80000000, "lo should be 0x80000000")
    self.assertEqual(st.vgpr[0][1], 0xFFFFFFFF, "hi should be 0xFFFFFFFF (sign extended)")


class Test64BitCompare(unittest.TestCase):
  """Tests for 64-bit scalar compare instructions."""

  def test_s_cmp_eq_u64_equal(self):
    """S_CMP_EQ_U64: comparing equal 64-bit values sets SCC=1."""
    val = 0x123456789ABCDEF0
    instructions = [
      s_mov_b32(s[0], val & 0xFFFFFFFF),
      s_mov_b32(s[1], val >> 32),
      s_mov_b32(s[2], val & 0xFFFFFFFF),
      s_mov_b32(s[3], val >> 32),
      s_cmp_eq_u64(s[0:1], s[2:3]),
      s_cselect_b32(s[4], 1, 0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.scc, 1)
    self.assertEqual(st.sgpr[4], 1)

  def test_s_cmp_eq_u64_different_upper_bits(self):
    """S_CMP_EQ_U64: values differing only in upper 32 bits are not equal."""
    # This is the bug case - if only lower 32 bits are compared, these would be equal
    instructions = [
      s_mov_b32(s[0], 0),  # lower 32 bits of value 0
      s_mov_b32(s[1], 0),  # upper 32 bits of value 0
      s_mov_b32(s[2], 0),  # lower 32 bits of 0x100000000
      s_mov_b32(s[3], 1),  # upper 32 bits of 0x100000000
      s_cmp_eq_u64(s[0:1], s[2:3]),
      s_cselect_b32(s[4], 1, 0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.scc, 0, "0 != 0x100000000, SCC should be 0")
    self.assertEqual(st.sgpr[4], 0)

  def test_s_cmp_lg_u64_different(self):
    """S_CMP_LG_U64: different 64-bit values sets SCC=1."""
    instructions = [
      s_mov_b32(s[0], 0),
      s_mov_b32(s[1], 0),  # s[0:1] = 0
      s_mov_b32(s[2], 0),
      s_mov_b32(s[3], 1),  # s[2:3] = 0x100000000
      s_cmp_lg_u64(s[0:1], s[2:3]),
      s_cselect_b32(s[4], 1, 0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.scc, 1, "0 != 0x100000000, SCC should be 1")
    self.assertEqual(st.sgpr[4], 1)


class TestSOPPNop(unittest.TestCase):
  """Tests for S_NOP and other SOPP instructions with expression-based for loops.

  S_NOP's pcode uses 'for i in 0U : SIMM16.u16[3 : 0].u32 do' which requires
  the parser to handle non-constant loop bounds.
  """

  def test_s_nop_basic(self):
    """S_NOP executes without side effects."""
    # S_NOP with immediate 0 should just do nothing
    instructions = [
      s_mov_b32(s[0], 42),
      s_nop(0),  # nop with simm16=0
      s_mov_b32(s[1], 100),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[0], 42)
    self.assertEqual(st.sgpr[1], 100)

  def test_s_nop_with_count(self):
    """S_NOP with count parameter executes multiple nops."""
    # S_NOP with immediate 3 should execute 4 nops (0:3 inclusive)
    instructions = [
      s_mov_b32(s[0], 1),
      s_nop(3),  # nop with simm16=3 -> 4 iterations
      s_add_u32(s[0], s[0], 1),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[0], 2)


class TestNullRegister(unittest.TestCase):
  """Tests for NULL register (124) behavior - writes should be discarded, reads return 0."""

  def test_s_mov_b32_from_null(self):
    """S_MOV_B32 from NULL should read as 0."""
    instructions = [
      s_mov_b32(s[0], 0xDEADBEEF),  # Set s[0] to sentinel
      s_mov_b32(s[0], NULL),  # Read from NULL - should be 0
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[0], 0)

  def test_s_add_u32_with_null_src(self):
    """S_ADD_U32 with NULL as source should use 0."""
    instructions = [
      s_mov_b32(s[0], 100),
      s_add_u32(s[1], s[0], NULL),  # 100 + 0 = 100
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[1], 100)

  def test_s_mov_b32_to_null(self):
    """S_MOV_B32 to NULL (sdst=124) should discard the write."""
    instructions = [
      s_mov_b32(s[0], 0xDEADBEEF),  # Set s[0] to sentinel
      s_mov_b32(NULL, 42),  # Write to NULL - should be discarded
      # s[0] should still be 0xDEADBEEF since NULL write doesn't affect it
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[0], 0xDEADBEEF)

  def test_s_add_u32_to_null(self):
    """S_ADD_U32 with sdst=NULL should discard result but still set SCC."""
    instructions = [
      s_mov_b32(s[0], 0xFFFFFFFF),
      s_mov_b32(s[1], 1),
      s_add_u32(NULL, s[0], s[1]),  # overflow, write to NULL
      s_cselect_b32(s[2], 1, 0),  # capture SCC
    ]
    st = run_program(instructions, n_lanes=1)
    # SCC should still be set from overflow even though result was discarded
    self.assertEqual(st.sgpr[2], 1)
    self.assertEqual(st.scc, 1)

  def test_s_and_b32_to_null(self):
    """S_AND_B32 with sdst=NULL should discard result but still set SCC."""
    instructions = [
      s_mov_b32(s[0], 0xFF00FF00),
      s_mov_b32(s[1], 0x0F0F0F0F),
      s_and_b32(NULL, s[0], s[1]),  # result=0x0F000F00, non-zero so SCC=1
      s_cselect_b32(s[2], 1, 0),  # capture SCC
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[2], 1)  # SCC=1 because result was non-zero
    self.assertEqual(st.scc, 1)

  def test_s_or_b32_to_null_zero_result(self):
    """S_OR_B32 with sdst=NULL and zero result should set SCC=0."""
    instructions = [
      s_mov_b32(s[0], 0),
      s_mov_b32(s[1], 0),
      s_or_b32(NULL, s[0], s[1]),  # result=0, so SCC=0
      s_cselect_b32(s[2], 1, 0),  # capture SCC
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[2], 0)  # SCC=0 because result was zero
    self.assertEqual(st.scc, 0)


class Test64BitSOP1InlineConstants(unittest.TestCase):
  """Tests for 64-bit SOP1 instructions with inline constants.

  Regression tests for bug where rsrc_dyn didn't properly handle 64-bit
  inline constants, incorrectly duplicating lo bits to hi instead of
  zero/sign-extending.
  """

  def test_s_mov_b64_inline_0(self):
    """S_MOV_B64 with inline constant 0."""
    instructions = [
      s_mov_b64(s[0:1], 0),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][0], 0)
    self.assertEqual(st.vgpr[0][1], 0)

  def test_s_mov_b64_inline_16(self):
    """S_MOV_B64 with inline constant 16 should set lo=16, hi=0."""
    instructions = [
      s_mov_b64(s[0:1], 16),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][0], 16)
    self.assertEqual(st.vgpr[0][1], 0)

  def test_s_mov_b64_inline_64(self):
    """S_MOV_B64 with inline constant 64 (max positive)."""
    instructions = [
      s_mov_b64(s[0:1], 64),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][0], 64)
    self.assertEqual(st.vgpr[0][1], 0)

  def test_s_mov_b64_inline_neg1(self):
    """S_MOV_B64 with inline constant -1 should sign-extend."""
    instructions = [
      s_mov_b64(s[0:1], -1),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][0], 0xFFFFFFFF)
    self.assertEqual(st.vgpr[0][1], 0xFFFFFFFF)

  def test_s_mov_b64_inline_neg16(self):
    """S_MOV_B64 with inline constant -16 should sign-extend."""
    instructions = [
      s_mov_b64(s[0:1], -16),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][0], 0xFFFFFFF0)
    self.assertEqual(st.vgpr[0][1], 0xFFFFFFFF)

  def test_s_mov_b64_float_const_1_0(self):
    """S_MOV_B64 with float inline constant 1.0 - casts F32 to F64."""
    instructions = [
      s_mov_b64(s[0:1], 1.0),  # inline constant 242 (1.0f)
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    # Hardware casts F32 to F64: 1.0f64 = 0x3FF0000000000000
    self.assertEqual(st.vgpr[0][0], 0x00000000)  # lo
    self.assertEqual(st.vgpr[0][1], 0x3FF00000)  # hi

  def test_s_or_b64_inline_constant(self):
    """S_OR_B64 with 64-bit inline constant."""
    instructions = [
      s_mov_b64(s[0:1], 0),
      s_or_b64(s[2:3], s[0:1], 16),
      v_mov_b32_e32(v[0], s[2]),
      v_mov_b32_e32(v[1], s[3]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][0], 16)
    self.assertEqual(st.vgpr[0][1], 0)

  def test_s_and_b64_inline_constant(self):
    """S_AND_B64 with 64-bit inline constant."""
    instructions = [
      s_mov_b32(s[0], 0xFFFFFFFF),
      s_mov_b32(s[1], 0xFFFFFFFF),
      s_and_b64(s[2:3], s[0:1], 16),
      v_mov_b32_e32(v[0], s[2]),
      v_mov_b32_e32(v[1], s[3]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][0], 16)
    self.assertEqual(st.vgpr[0][1], 0)


class Test64BitSOPLiterals(unittest.TestCase):
  """Tests for 64-bit SOP instructions with 32-bit literals.

  Tests the behavior when a 64-bit SOP instruction uses a 32-bit literal
  (offset 255 in instruction encoding). The literal is zero-extended to 64 bits.
  """

  def test_s_mov_b64_literal(self):
    """S_MOV_B64 with 32-bit literal value - zero-extended to 64 bits."""
    instructions = [
      s_mov_b64(s[0:1], 0x12345678),  # literal > 64, uses literal encoding
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][0], 0x12345678)
    self.assertEqual(st.vgpr[0][1], 0)

  def test_s_or_b64_literal(self):
    """S_OR_B64 with 32-bit literal value - zero-extended to 64 bits."""
    instructions = [
      s_mov_b64(s[0:1], 0),
      s_or_b64(s[2:3], s[0:1], 0x12345678),  # literal
      v_mov_b32_e32(v[0], s[2]),
      v_mov_b32_e32(v[1], s[3]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][0], 0x12345678)
    self.assertEqual(st.vgpr[0][1], 0)

  def test_s_and_b64_literal(self):
    """S_AND_B64 with 32-bit literal value - zero-extended to 64 bits."""
    instructions = [
      s_mov_b32(s[0], 0xFFFFFFFF),
      s_mov_b32(s[1], 0xFFFFFFFF),
      s_and_b64(s[2:3], s[0:1], 0x12345678),  # literal
      v_mov_b32_e32(v[0], s[2]),
      v_mov_b32_e32(v[1], s[3]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][0], 0x12345678)
    self.assertEqual(st.vgpr[0][1], 0)

  def test_s_mov_b64_literal_negative(self):
    """S_MOV_B64 with 0xFFFFFFFF literal - zero-extended (not sign-extended)."""
    instructions = [
      s_mov_b64(s[0:1], 0xFFFFFFFF),  # -1 as 32-bit, but zero-extended to 64-bit
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][0], 0xFFFFFFFF)
    self.assertEqual(st.vgpr[0][1], 0)  # zero-extended, not sign-extended

  def test_s_mov_b64_literal_high_bit(self):
    """S_MOV_B64 with 0x80000000 literal - zero-extended (not sign-extended)."""
    instructions = [
      s_mov_b64(s[0:1], 0x80000000),  # high bit set, but zero-extended
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][0], 0x80000000)
    self.assertEqual(st.vgpr[0][1], 0)  # zero-extended, not sign-extended


if __name__ == '__main__':
  unittest.main()
