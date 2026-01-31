"""Tests for SCRATCH instructions - scratch (private) memory operations.

Includes: scratch_load_*, scratch_store_*
"""
import unittest
from extra.assembly.amd.test.hw.helpers import *

class TestScratchStore(unittest.TestCase):
  """Tests for SCRATCH store instructions."""

  def test_scratch_store_b32_basic(self):
    """SCRATCH_STORE_B32 stores 32-bit value to scratch memory."""
    TEST_OFFSET = 256
    instructions = [
      s_load_b64(s[2:3], s[80:81], 0, soffset=SrcEnum.NULL),
      s_waitcnt(lgkmcnt=0),
      s_mov_b32(s[4], 0xDEADBEEF),
      v_mov_b32_e32(v[2], s[4]),
      v_mov_b32_e32(v[0], 0),
      # Store via scratch
      scratch_store_b32(addr=v[0], data=v[2], saddr=SrcEnum.NULL, offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      # Load back via scratch
      scratch_load_b32(addr=v[0], vdst=v[3], saddr=SrcEnum.NULL, offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      v_mov_b32_e32(v[0], v[3]),
      s_mov_b32(s[2], 0),
      s_mov_b32(s[3], 0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][0], 0xDEADBEEF)

  def test_scratch_store_b64_basic(self):
    """SCRATCH_STORE_B64 stores 64-bit value to scratch memory."""
    TEST_OFFSET = 256
    instructions = [
      s_load_b64(s[2:3], s[80:81], 0, soffset=SrcEnum.NULL),
      s_waitcnt(lgkmcnt=0),
      s_mov_b32(s[4], 0xDEADBEEF),
      s_mov_b32(s[5], 0xCAFEBABE),
      v_mov_b32_e32(v[2], s[4]),
      v_mov_b32_e32(v[3], s[5]),
      v_mov_b32_e32(v[0], 0),
      scratch_store_b64(addr=v[0], data=v[2:3], saddr=SrcEnum.NULL, offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      scratch_load_b64(addr=v[0], vdst=v[4:5], saddr=SrcEnum.NULL, offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      v_mov_b32_e32(v[0], v[4]),
      v_mov_b32_e32(v[1], v[5]),
      s_mov_b32(s[2], 0),
      s_mov_b32(s[3], 0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][0], 0xDEADBEEF)
    self.assertEqual(st.vgpr[0][1], 0xCAFEBABE)

  def test_scratch_store_b8_basic(self):
    """SCRATCH_STORE_B8 stores single byte to scratch memory."""
    TEST_OFFSET = 256
    instructions = [
      s_load_b64(s[2:3], s[80:81], 0, soffset=SrcEnum.NULL),
      s_waitcnt(lgkmcnt=0),
      # First store full word
      s_mov_b32(s[4], 0xDEADBEEF),
      v_mov_b32_e32(v[2], s[4]),
      v_mov_b32_e32(v[0], 0),
      scratch_store_b32(addr=v[0], data=v[2], saddr=SrcEnum.NULL, offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      # Store single byte
      v_mov_b32_e32(v[2], 0x42),
      scratch_store_b8(addr=v[0], data=v[2], saddr=SrcEnum.NULL, offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      # Load back
      scratch_load_b32(addr=v[0], vdst=v[3], saddr=SrcEnum.NULL, offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      v_mov_b32_e32(v[0], v[3]),
      s_mov_b32(s[2], 0),
      s_mov_b32(s[3], 0),
    ]
    st = run_program(instructions, n_lanes=1)
    # Only byte 0 should change from 0xEF to 0x42
    self.assertEqual(st.vgpr[0][0], 0xDEADBE42)

  def test_scratch_store_b16_basic(self):
    """SCRATCH_STORE_B16 stores 16-bit value to scratch memory."""
    TEST_OFFSET = 256
    instructions = [
      s_load_b64(s[2:3], s[80:81], 0, soffset=SrcEnum.NULL),
      s_waitcnt(lgkmcnt=0),
      s_mov_b32(s[4], 0xDEADBEEF),
      v_mov_b32_e32(v[2], s[4]),
      v_mov_b32_e32(v[0], 0),
      scratch_store_b32(addr=v[0], data=v[2], saddr=SrcEnum.NULL, offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      s_mov_b32(s[4], 0xCAFE),
      v_mov_b32_e32(v[2], s[4]),
      scratch_store_b16(addr=v[0], data=v[2], saddr=SrcEnum.NULL, offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      scratch_load_b32(addr=v[0], vdst=v[3], saddr=SrcEnum.NULL, offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      v_mov_b32_e32(v[0], v[3]),
      s_mov_b32(s[2], 0),
      s_mov_b32(s[3], 0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][0], 0xDEADCAFE)


class TestScratchLoad(unittest.TestCase):
  """Tests for SCRATCH load instructions."""

  def test_scratch_load_b96(self):
    """SCRATCH_LOAD_B96 loads 96-bit value correctly."""
    TEST_OFFSET = 256
    instructions = [
      s_load_b64(s[2:3], s[80:81], 0, soffset=SrcEnum.NULL),
      s_waitcnt(lgkmcnt=0),
      v_mov_b32_e32(v[0], 0),
      s_mov_b32(s[4], 0xAAAAAAAA),
      v_mov_b32_e32(v[2], s[4]),
      s_mov_b32(s[4], 0xBBBBBBBB),
      v_mov_b32_e32(v[3], s[4]),
      s_mov_b32(s[4], 0xCCCCCCCC),
      v_mov_b32_e32(v[4], s[4]),
      scratch_store_b96(addr=v[0], data=v[2:4], saddr=SrcEnum.NULL, offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      scratch_load_b96(addr=v[0], vdst=v[5:7], saddr=SrcEnum.NULL, offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      v_mov_b32_e32(v[0], v[5]),
      v_mov_b32_e32(v[1], v[6]),
      v_mov_b32_e32(v[2], v[7]),
      s_mov_b32(s[2], 0),
      s_mov_b32(s[3], 0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][0], 0xAAAAAAAA)
    self.assertEqual(st.vgpr[0][1], 0xBBBBBBBB)
    self.assertEqual(st.vgpr[0][2], 0xCCCCCCCC)

  def test_scratch_load_b128(self):
    """SCRATCH_LOAD_B128 loads 128-bit value correctly."""
    TEST_OFFSET = 256
    instructions = [
      s_load_b64(s[2:3], s[80:81], 0, soffset=SrcEnum.NULL),
      s_waitcnt(lgkmcnt=0),
      v_mov_b32_e32(v[0], 0),
      s_mov_b32(s[4], 0xDEADBEEF),
      v_mov_b32_e32(v[2], s[4]),
      s_mov_b32(s[4], 0xCAFEBABE),
      v_mov_b32_e32(v[3], s[4]),
      s_mov_b32(s[4], 0x12345678),
      v_mov_b32_e32(v[4], s[4]),
      s_mov_b32(s[4], 0x9ABCDEF0),
      v_mov_b32_e32(v[5], s[4]),
      scratch_store_b128(addr=v[0], data=v[2:5], saddr=SrcEnum.NULL, offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      scratch_load_b128(addr=v[0], vdst=v[6:9], saddr=SrcEnum.NULL, offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      v_mov_b32_e32(v[0], v[6]),
      v_mov_b32_e32(v[1], v[7]),
      v_mov_b32_e32(v[2], v[8]),
      v_mov_b32_e32(v[3], v[9]),
      s_mov_b32(s[2], 0),
      s_mov_b32(s[3], 0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][0], 0xDEADBEEF)
    self.assertEqual(st.vgpr[0][1], 0xCAFEBABE)
    self.assertEqual(st.vgpr[0][2], 0x12345678)
    self.assertEqual(st.vgpr[0][3], 0x9ABCDEF0)

  def test_scratch_load_u8(self):
    """SCRATCH_LOAD_U8 loads unsigned byte with zero extension."""
    TEST_OFFSET = 256
    instructions = [
      s_load_b64(s[2:3], s[80:81], 0, soffset=SrcEnum.NULL),
      s_waitcnt(lgkmcnt=0),
      v_mov_b32_e32(v[0], 0),
      s_mov_b32(s[4], 0xDEADBEAB),
      v_mov_b32_e32(v[2], s[4]),
      scratch_store_b32(addr=v[0], data=v[2], saddr=SrcEnum.NULL, offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      scratch_load_u8(addr=v[0], vdst=v[3], saddr=SrcEnum.NULL, offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      v_mov_b32_e32(v[0], v[3]),
      s_mov_b32(s[2], 0),
      s_mov_b32(s[3], 0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][0], 0xAB)

  def test_scratch_load_i8(self):
    """SCRATCH_LOAD_I8 loads signed byte with sign extension."""
    TEST_OFFSET = 256
    instructions = [
      s_load_b64(s[2:3], s[80:81], 0, soffset=SrcEnum.NULL),
      s_waitcnt(lgkmcnt=0),
      v_mov_b32_e32(v[0], 0),
      s_mov_b32(s[4], 0x80),  # -128 as signed byte
      v_mov_b32_e32(v[2], s[4]),
      scratch_store_b8(addr=v[0], data=v[2], saddr=SrcEnum.NULL, offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      scratch_load_i8(addr=v[0], vdst=v[3], saddr=SrcEnum.NULL, offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      v_mov_b32_e32(v[0], v[3]),
      s_mov_b32(s[2], 0),
      s_mov_b32(s[3], 0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][0], 0xFFFFFF80)

  def test_scratch_load_u16(self):
    """SCRATCH_LOAD_U16 loads unsigned 16-bit with zero extension."""
    TEST_OFFSET = 256
    instructions = [
      s_load_b64(s[2:3], s[80:81], 0, soffset=SrcEnum.NULL),
      s_waitcnt(lgkmcnt=0),
      v_mov_b32_e32(v[0], 0),
      s_mov_b32(s[4], 0xDEADCAFE),
      v_mov_b32_e32(v[2], s[4]),
      scratch_store_b32(addr=v[0], data=v[2], saddr=SrcEnum.NULL, offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      scratch_load_u16(addr=v[0], vdst=v[3], saddr=SrcEnum.NULL, offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      v_mov_b32_e32(v[0], v[3]),
      s_mov_b32(s[2], 0),
      s_mov_b32(s[3], 0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][0], 0xCAFE)

  def test_scratch_load_i16(self):
    """SCRATCH_LOAD_I16 loads signed 16-bit with sign extension."""
    TEST_OFFSET = 256
    instructions = [
      s_load_b64(s[2:3], s[80:81], 0, soffset=SrcEnum.NULL),
      s_waitcnt(lgkmcnt=0),
      v_mov_b32_e32(v[0], 0),
      s_mov_b32(s[4], 0x8000),  # -32768 as signed 16-bit
      v_mov_b32_e32(v[2], s[4]),
      scratch_store_b16(addr=v[0], data=v[2], saddr=SrcEnum.NULL, offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      scratch_load_i16(addr=v[0], vdst=v[3], saddr=SrcEnum.NULL, offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      v_mov_b32_e32(v[0], v[3]),
      s_mov_b32(s[2], 0),
      s_mov_b32(s[3], 0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][0], 0xFFFF8000)


class TestScratchSVE(unittest.TestCase):
  """Tests for SCRATCH SVE (Scratch VGPR Enable) bit behavior."""

  def test_scratch_sve_zero_ignores_vaddr(self):
    """With SVE=0, VADDR should be ignored in address calculation."""
    TEST_OFFSET = 256
    # Store a marker value at offset 256 (where SVE=0 should go)
    # Then set v[0] to a non-zero value (100) and store via scratch with SVE=0
    # If SVE=0 is handled correctly, the VADDR (100) should be IGNORED,
    # and the store should go to offset 256, not 256+100=356
    instructions = [
      s_load_b64(s[2:3], s[80:81], 0, soffset=SrcEnum.NULL),
      s_waitcnt(lgkmcnt=0),
      # First, store 0xAAAAAAAA at offset 256 with v[0]=0
      v_mov_b32_e32(v[0], 0),
      s_mov_b32(s[4], 0xAAAAAAAA),
      v_mov_b32_e32(v[2], s[4]),
      scratch_store_b32(addr=v[0], data=v[2], saddr=SrcEnum.NULL, offset=TEST_OFFSET, sve=0),
      s_waitcnt(vmcnt=0),
      # Now set v[0] to 100 (non-zero) and store 0xBBBBBBBB with SVE=0
      # With SVE=0, v[0] should be IGNORED, so this should overwrite offset 256
      v_mov_b32_e32(v[0], 100),
      s_mov_b32(s[4], 0xBBBBBBBB),
      v_mov_b32_e32(v[2], s[4]),
      scratch_store_b32(addr=v[0], data=v[2], saddr=SrcEnum.NULL, offset=TEST_OFFSET, sve=0),
      s_waitcnt(vmcnt=0),
      # Load back from offset 256 (with v[0]=0) - should get 0xBBBBBBBB
      v_mov_b32_e32(v[0], 0),
      scratch_load_b32(addr=v[0], vdst=v[3], saddr=SrcEnum.NULL, offset=TEST_OFFSET, sve=0),
      s_waitcnt(vmcnt=0),
      v_mov_b32_e32(v[0], v[3]),
      s_mov_b32(s[2], 0),
      s_mov_b32(s[3], 0),
    ]
    st = run_program(instructions, n_lanes=1)
    # If SVE=0 works correctly, v[0] should be 0xBBBBBBBB (the second store overwrote the first)
    # If SVE=0 is wrong (VADDR used), v[0] would be 0xAAAAAAAA (stores went to different locations)
    self.assertEqual(st.vgpr[0][0], 0xBBBBBBBB, "SVE=0 should ignore VADDR, both stores should go to same location")

  def test_scratch_sve_one_uses_vaddr(self):
    """With SVE=1, VADDR should be used as offset in address calculation."""
    TEST_OFFSET = 256
    # Store at offset 256 with v[0]=0, then store at offset 256 with v[0]=100 and SVE=1
    # With SVE=1, the second store should go to 256+100=356, not 256
    instructions = [
      s_load_b64(s[2:3], s[80:81], 0, soffset=SrcEnum.NULL),
      s_waitcnt(lgkmcnt=0),
      # First, store 0xAAAAAAAA at offset 256 with v[0]=0
      v_mov_b32_e32(v[0], 0),
      s_mov_b32(s[4], 0xAAAAAAAA),
      v_mov_b32_e32(v[2], s[4]),
      scratch_store_b32(addr=v[0], data=v[2], saddr=SrcEnum.NULL, offset=TEST_OFFSET, sve=1),
      s_waitcnt(vmcnt=0),
      # Now set v[0] to 100 and store 0xBBBBBBBB with SVE=1
      # With SVE=1, v[0] IS used, so this should go to offset 256+100=356
      v_mov_b32_e32(v[0], 100),
      s_mov_b32(s[4], 0xBBBBBBBB),
      v_mov_b32_e32(v[2], s[4]),
      scratch_store_b32(addr=v[0], data=v[2], saddr=SrcEnum.NULL, offset=TEST_OFFSET, sve=1),
      s_waitcnt(vmcnt=0),
      # Load back from offset 256 (with v[0]=0) - should still be 0xAAAAAAAA
      v_mov_b32_e32(v[0], 0),
      scratch_load_b32(addr=v[0], vdst=v[3], saddr=SrcEnum.NULL, offset=TEST_OFFSET, sve=1),
      s_waitcnt(vmcnt=0),
      v_mov_b32_e32(v[0], v[3]),
      s_mov_b32(s[2], 0),
      s_mov_b32(s[3], 0),
    ]
    st = run_program(instructions, n_lanes=1)
    # If SVE=1 works correctly, v[0] should be 0xAAAAAAAA (stores went to different locations)
    self.assertEqual(st.vgpr[0][0], 0xAAAAAAAA, "SVE=1 should use VADDR, stores should go to different locations")


class TestScratchMultiLane(unittest.TestCase):
  """Tests for SCRATCH operations with multiple lanes."""

  def test_scratch_store_load_multi_lane(self):
    """SCRATCH store/load works correctly with multiple lanes (private per-lane memory)."""
    TEST_OFFSET = 256
    instructions = [
      s_load_b64(s[2:3], s[80:81], 0, soffset=SrcEnum.NULL),
      s_waitcnt(lgkmcnt=0),
      # Each lane stores its lane ID
      v_mov_b32_e32(v[0], 0),
      v_mov_b32_e32(v[2], v[255]),  # v[255] has packed workitem IDs, low 10 bits = x
      v_and_b32_e32(v[2], 0x3FF, v[2]),  # extract lane ID
      scratch_store_b32(addr=v[0], data=v[2], saddr=SrcEnum.NULL, offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      # Load back
      scratch_load_b32(addr=v[0], vdst=v[3], saddr=SrcEnum.NULL, offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      v_mov_b32_e32(v[0], v[3]),
      s_mov_b32(s[2], 0),
      s_mov_b32(s[3], 0),
    ]
    st = run_program(instructions, n_lanes=4)
    # Each lane should have loaded its own lane ID
    for lane in range(4):
      self.assertEqual(st.vgpr[lane][0], lane, f"Lane {lane} should have value {lane}")


if __name__ == '__main__':
  unittest.main()
