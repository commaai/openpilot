"""Tests for VOPD instructions - dual-issue vector operations.

VOPD executes two operations simultaneously. Key behavior:
- Both ops read their sources BEFORE either writes (dual-issue semantics)
- This means if X writes to a register that Y reads, Y sees the OLD value
- Op X can use ops 0-15 (FMAC, MUL, ADD, MOV, etc.)
- Op Y can use ops 0-18 (includes ADD_NC_U32, LSHLREV, AND)
"""
import unittest
from extra.assembly.amd.test.hw.helpers import run_program, run_program_emu, run_program_hw, compare_wave_states, \
  v, s, v_mov_b32_e32, s_mov_b32
from extra.assembly.amd.autogen.rdna3.ins import VOPD, VOPD_LIT, VOPDOp

class TestVOPDBasic(unittest.TestCase):
  """Basic VOPD functionality tests."""

  def test_vopd_dual_mov(self):
    """VOPD with two MOV operations to different registers."""
    instructions = [
      v_mov_b32_e32(v[0], 0x12345678),
      v_mov_b32_e32(v[1], 0xDEADBEEF),
      # X: v[2] = v[0], Y: v[3] = v[1]
      VOPD(VOPDOp.V_DUAL_MOV_B32, VOPDOp.V_DUAL_MOV_B32, v[2], v[3], v[0], v[1], v[0], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][2], 0x12345678)
    self.assertEqual(st.vgpr[0][3], 0xDEADBEEF)

  def test_vopd_mov_and_add(self):
    """VOPD with MOV (X) and ADD_NC_U32 (Y) - ADD_NC_U32 can only be Y op."""
    instructions = [
      v_mov_b32_e32(v[0], 10),
      v_mov_b32_e32(v[1], 5),
      # X: v[2] = 100 (literal), Y: v[3] = v[0] + v[1] = 15
      VOPD(VOPDOp.V_DUAL_MOV_B32, VOPDOp.V_DUAL_ADD_NC_U32, v[2], v[3], 100, v[0], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][2], 100)
    self.assertEqual(st.vgpr[0][3], 15)


class TestVOPDReadBeforeWrite(unittest.TestCase):
  """Tests for VOPD dual-issue read-before-write semantics.

  In VOPD, both X and Y operations read their sources BEFORE either writes.
  This is critical when X's destination is Y's source.
  """

  def test_vopd_x_writes_y_reads_same_reg(self):
    """VOPD where X writes to a register that Y reads.

    X: v[2] = 0 (overwrites v[2])
    Y: v[1] = v[2] + v[0]  (srcy0=v[2], vsrcy1=v[0])

    If reads happen before writes: v[1] = OLD_v[2] + v[0] = 0xFFFFFFFF + 1 = 0
    If writes happen before reads: v[1] = 0 + v[0] = 0 + 1 = 1

    Hardware does reads-before-writes, so v[1] should be 0.
    """
    instructions = [
      v_mov_b32_e32(v[0], 1),          # v[0] = 1
      v_mov_b32_e32(v[1], 0x99999999), # v[1] = placeholder (will be overwritten)
      v_mov_b32_e32(v[2], 0xFFFFFFFF), # v[2] = 0xFFFFFFFF
      # X: v[2] = 0 (literal), srcx0=0, vsrcx1=v[0] (unused for MOV)
      # Y: v[1] = srcy0 + vsrcy1 = v[2] + v[0] (should read OLD v[2] = 0xFFFFFFFF)
      # vdsty encoding: (vdsty << 1) | ((vdstx & 1) ^ 1) where vdsty field = 0, vdstx = v[2]
      # So vdsty_reg = (0 << 1) | ((2 & 1) ^ 1) = 0 | 1 = 1 = v[1]
      VOPD(VOPDOp.V_DUAL_MOV_B32, VOPDOp.V_DUAL_ADD_NC_U32, v[2], v[0], 0, v[2], v[0], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    # X should have written 0 to v[2]
    self.assertEqual(st.vgpr[0][2], 0, "X should write 0 to v[2]")
    # Y should have read OLD v[2] (0xFFFFFFFF) and added v[0] (1)
    # 0xFFFFFFFF + 1 = 0 (wrap around)
    self.assertEqual(st.vgpr[0][1], 0, "Y should read OLD v[2]=0xFFFFFFFF, compute 0xFFFFFFFF+1=0")

  def test_vopd_x_writes_y_reads_same_reg_v2(self):
    """VOPD where X writes to a register that Y reads - cleaner test case.

    X: v[2] = 0 (MOV)
    Y: v[1] = v[2] + v[2] (ADD_NC_U32 with both sources from v[2])

    If reads happen before writes: v[1] = OLD_v[2] + OLD_v[2] = 100 + 100 = 200
    If writes happen before reads: v[1] = 0 + 0 = 0

    Hardware does reads-before-writes, so v[1] should be 200.
    """
    instructions = [
      v_mov_b32_e32(v[0], 0x88888888), # v[0] = unused placeholder
      v_mov_b32_e32(v[1], 0x99999999), # v[1] = placeholder (will be overwritten)
      v_mov_b32_e32(v[2], 100),        # v[2] = 100
      # X: v[2] = 0 (literal)
      # Y: v[1] = srcy0 + vsrcy1 = v[2] + v[2] (should read OLD v[2] = 100)
      VOPD(VOPDOp.V_DUAL_MOV_B32, VOPDOp.V_DUAL_ADD_NC_U32, v[2], v[0], 0, v[2], v[0], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    # X should have written 0 to v[2]
    self.assertEqual(st.vgpr[0][2], 0, "X should write 0 to v[2]")
    # Y should have read OLD v[2] (100) twice and added them
    self.assertEqual(st.vgpr[0][1], 200, "Y should read OLD v[2]=100 twice, compute 100+100=200")


class TestVOPDLiterals(unittest.TestCase):
  """Tests for VOPD instructions that use SIMM32 literals (FMAAK, FMAMK)."""

  def test_vopd_fmaak_f32(self):
    """VOPD V_DUAL_FMAAK_F32: D = S0 * S1 + SIMM32 (literal addend).

    Tests that the 32-bit literal (SIMM32) is correctly passed to the instruction.
    fma(2.0, 3.0, 10.0) = 2*3 + 10 = 16.0
    """
    from extra.assembly.amd.test.hw.helpers import f2i, i2f
    instructions = [
      v_mov_b32_e32(v[0], f2i(2.0)),  # v[0] = 2.0
      v_mov_b32_e32(v[1], f2i(3.0)),  # v[1] = 3.0
      # VOPD args: opx, opy, vdstx, vdsty, srcx0, srcy0, vsrcx1, vsrcy1
      # X: v[2] = fma(srcx0, vsrcx1, SIMM32) = v[0]*v[1]+10.0 = 2*3+10 = 16
      # Y: v[3] = srcy0 (MOV) = v[0] = 2.0
      VOPD_LIT(VOPDOp.V_DUAL_FMAAK_F32, VOPDOp.V_DUAL_MOV_B32, v[2], v[3], v[0], v[0], v[1], v[0], literal=f2i(10.0)),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][2]), 16.0, places=5, msg="fma(2.0, 3.0, 10.0) should be 16.0")

  def test_vopd_fmamk_f32(self):
    """VOPD V_DUAL_FMAMK_F32: D = S0 * SIMM32 + S1 (literal multiplier).

    Tests that the 32-bit literal (SIMM32) is correctly used as the multiplier.
    fma(2.0, 5.0, 3.0) = 2*5 + 3 = 13.0
    """
    from extra.assembly.amd.test.hw.helpers import f2i, i2f
    instructions = [
      v_mov_b32_e32(v[0], f2i(2.0)),  # v[0] = 2.0
      v_mov_b32_e32(v[1], f2i(3.0)),  # v[1] = 3.0
      # X: v[2] = fma(srcx0, SIMM32, vsrcx1) = v[0]*5.0+v[1] = 2*5+3 = 13
      # Y: v[3] = srcy0 (MOV) = v[0] = 2.0
      VOPD_LIT(VOPDOp.V_DUAL_FMAMK_F32, VOPDOp.V_DUAL_MOV_B32, v[2], v[3], v[0], v[0], v[1], v[0], literal=f2i(5.0)),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][2]), 13.0, places=5, msg="fma(2.0, 5.0, 3.0) should be 13.0")


class TestVOPDMultilane(unittest.TestCase):
  """Tests for VOPD with multiple lanes."""

  def test_vopd_multilane_mov_add(self):
    """VOPD MOV and ADD with multiple active lanes - no register conflict."""
    instructions = [
      v_mov_b32_e32(v[0], 5),
      v_mov_b32_e32(v[1], 10),
      # X: v[2] = 100 (constant), Y: v[1] = v[0] + v[1] = 5 + 10 = 15
      # vdsty_reg = (vdsty << 1) | ((vdstx.offset & 1) ^ 1) = (0 << 1) | ((258 & 1) ^ 1) = 0 | 1 = 1
      VOPD(VOPDOp.V_DUAL_MOV_B32, VOPDOp.V_DUAL_ADD_NC_U32, v[2], v[0], 100, v[0], v[2], v[1]),
    ]
    st = run_program(instructions, n_lanes=4)
    for lane in range(4):
      self.assertEqual(st.vgpr[lane][2], 100, f"Lane {lane}: v[2] should be 100")
      self.assertEqual(st.vgpr[lane][1], 15, f"Lane {lane}: v[1] should be 15 (5+10)")


if __name__ == '__main__':
  unittest.main()
