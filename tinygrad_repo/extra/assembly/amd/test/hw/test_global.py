"""Tests for GLOBAL instructions - global memory operations.

Includes: global_load_*, global_store_*, global_atomic_*, global_load_d16_*
"""
import unittest
from extra.assembly.amd.test.hw.helpers import *

class TestGlobalAtomic(unittest.TestCase):
  """Tests for GLOBAL atomic instructions."""

  def _make_test(self, setup_instrs, atomic_instr, check_fn, test_offset=2000):
    """Helper to create atomic test instructions."""
    instructions = [
      s_load_b64(s[2:3], s[80:81], 0, soffset=SrcEnum.NULL),
      s_waitcnt(lgkmcnt=0),
      v_mov_b32_e32(v[0], s[2]),
      v_mov_b32_e32(v[1], s[3]),
    ] + setup_instrs + [atomic_instr, s_waitcnt(vmcnt=0),
      v_mov_b32_e32(v[0], 0),
      v_mov_b32_e32(v[1], 0),
      s_mov_b32(s[2], 0),
      s_mov_b32(s[3], 0),
    ]
    st = run_program(instructions, n_lanes=1)
    check_fn(st)

  def test_global_atomic_add_u32(self):
    """GLOBAL_ATOMIC_ADD_U32 adds to memory and returns old value."""
    TEST_OFFSET = 2000
    setup = [
      s_mov_b32(s[0], 100),
      v_mov_b32_e32(v[2], s[0]),
      global_store_b32(addr=v[0:1], data=v[2], saddr=SrcEnum.NULL, offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      s_mov_b32(s[0], 50),
      v_mov_b32_e32(v[3], s[0]),
    ]
    atomic = GLOBAL(GLOBALOp.GLOBAL_ATOMIC_ADD_U32, addr=v[0:1], data=v[3], vdst=v[4], saddr=SrcEnum.NULL, offset=TEST_OFFSET, glc=1)
    def check(st):
      self.assertEqual(st.vgpr[0][4], 100)
    self._make_test(setup, atomic, check, TEST_OFFSET)

  def test_global_atomic_add_u64(self):
    """GLOBAL_ATOMIC_ADD_U64 adds 64-bit value and returns old value."""
    TEST_OFFSET = 2000
    setup = [
      s_mov_b32(s[0], 0xFFFFFFFF),
      v_mov_b32_e32(v[2], s[0]),
      s_mov_b32(s[0], 0x00000000),
      v_mov_b32_e32(v[3], s[0]),
      global_store_b64(addr=v[0:1], data=v[2:3], saddr=SrcEnum.NULL, offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      s_mov_b32(s[0], 0x00000001),
      v_mov_b32_e32(v[4], s[0]),
      s_mov_b32(s[0], 0x00000000),
      v_mov_b32_e32(v[5], s[0]),
    ]
    atomic = GLOBAL(GLOBALOp.GLOBAL_ATOMIC_ADD_U64, addr=v[0:1], data=v[4:5], vdst=v[6:7], saddr=SrcEnum.NULL, offset=TEST_OFFSET, glc=1)
    def check(st):
      self.assertEqual(st.vgpr[0][6], 0xFFFFFFFF)
      self.assertEqual(st.vgpr[0][7], 0x00000000)
    self._make_test(setup, atomic, check, TEST_OFFSET)


class TestGlobalLoad(unittest.TestCase):
  """Tests for GLOBAL load instructions."""

  def test_global_load_b96(self):
    """GLOBAL_LOAD_B96 loads 96-bit value correctly."""
    TEST_OFFSET = 2000
    instructions = [
      s_load_b64(s[2:3], s[80:81], 0, soffset=SrcEnum.NULL),
      s_waitcnt(lgkmcnt=0),
      v_mov_b32_e32(v[0], s[2]),
      v_mov_b32_e32(v[1], s[3]),
      s_mov_b32(s[0], 0xAAAAAAAA),
      v_mov_b32_e32(v[2], s[0]),
      s_mov_b32(s[0], 0xBBBBBBBB),
      v_mov_b32_e32(v[3], s[0]),
      s_mov_b32(s[0], 0xCCCCCCCC),
      v_mov_b32_e32(v[4], s[0]),
      global_store_b96(addr=v[0:1], data=v[2:4], saddr=SrcEnum.NULL, offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      GLOBAL(GLOBALOp.GLOBAL_LOAD_B96, addr=v[0:1], vdst=v[5:7], saddr=SrcEnum.NULL, offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      v_mov_b32_e32(v[0], 0),
      v_mov_b32_e32(v[1], 0),
      s_mov_b32(s[2], 0),
      s_mov_b32(s[3], 0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][5], 0xAAAAAAAA)
    self.assertEqual(st.vgpr[0][6], 0xBBBBBBBB)
    self.assertEqual(st.vgpr[0][7], 0xCCCCCCCC)

  def test_global_load_b128(self):
    """GLOBAL_LOAD_B128 loads 128-bit value correctly."""
    TEST_OFFSET = 2000
    instructions = [
      s_load_b64(s[2:3], s[80:81], 0, soffset=SrcEnum.NULL),
      s_waitcnt(lgkmcnt=0),
      v_mov_b32_e32(v[0], s[2]),
      v_mov_b32_e32(v[1], s[3]),
      s_mov_b32(s[0], 0xDEADBEEF),
      v_mov_b32_e32(v[2], s[0]),
      s_mov_b32(s[0], 0xCAFEBABE),
      v_mov_b32_e32(v[3], s[0]),
      s_mov_b32(s[0], 0x12345678),
      v_mov_b32_e32(v[4], s[0]),
      s_mov_b32(s[0], 0x9ABCDEF0),
      v_mov_b32_e32(v[5], s[0]),
      global_store_b128(addr=v[0:1], data=v[2:5], saddr=SrcEnum.NULL, offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      GLOBAL(GLOBALOp.GLOBAL_LOAD_B128, addr=v[0:1], vdst=v[6:9], saddr=SrcEnum.NULL, offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      v_mov_b32_e32(v[0], 0),
      v_mov_b32_e32(v[1], 0),
      s_mov_b32(s[2], 0),
      s_mov_b32(s[3], 0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][6], 0xDEADBEEF)
    self.assertEqual(st.vgpr[0][7], 0xCAFEBABE)
    self.assertEqual(st.vgpr[0][8], 0x12345678)
    self.assertEqual(st.vgpr[0][9], 0x9ABCDEF0)


class TestGlobalStore(unittest.TestCase):
  """Tests for GLOBAL store instructions."""

  def test_global_store_b8_basic(self):
    """GLOBAL_STORE_B8 stores a single byte from VDATA[7:0]."""
    TEST_OFFSET = 256
    instructions = [
      s_load_b64(s[2:3], s[80:81], 0, soffset=SrcEnum.NULL),
      s_waitcnt(lgkmcnt=0),
      # First store 0xDEADBEEF to memory
      s_mov_b32(s[4], 0xDEADBEEF),
      v_mov_b32_e32(v[2], s[4]),
      v_mov_b32_e32(v[0], 0),
      global_store_b32(addr=v[0], data=v[2], saddr=s[2:3], offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      # Now store single byte 0x42 to same address (should only change byte 0)
      v_mov_b32_e32(v[2], 0x42),
      global_store_b8(addr=v[0], data=v[2], saddr=s[2:3], offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      # Read back and check
      GLOBAL(GLOBALOp.GLOBAL_LOAD_B32, addr=v[0], vdst=v[3], data=v[3], saddr=s[2:3], offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      v_mov_b32_e32(v[0], v[3]),
      s_mov_b32(s[2], 0),
      s_mov_b32(s[3], 0),
    ]
    st = run_program(instructions, n_lanes=1)
    # Only byte 0 should change from 0xEF to 0x42
    self.assertEqual(st.vgpr[0][0], 0xDEADBE42, "Only byte 0 should be modified")

  def test_global_store_b8_byte1(self):
    """GLOBAL_STORE_B8 at offset+1 stores to byte 1."""
    TEST_OFFSET = 256
    instructions = [
      s_load_b64(s[2:3], s[80:81], 0, soffset=SrcEnum.NULL),
      s_waitcnt(lgkmcnt=0),
      s_mov_b32(s[4], 0xDEADBEEF),
      v_mov_b32_e32(v[2], s[4]),
      v_mov_b32_e32(v[0], 0),
      global_store_b32(addr=v[0], data=v[2], saddr=s[2:3], offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      v_mov_b32_e32(v[2], 0x42),
      global_store_b8(addr=v[0], data=v[2], saddr=s[2:3], offset=TEST_OFFSET+1),
      s_waitcnt(vmcnt=0),
      GLOBAL(GLOBALOp.GLOBAL_LOAD_B32, addr=v[0], vdst=v[3], data=v[3], saddr=s[2:3], offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      v_mov_b32_e32(v[0], v[3]),
      s_mov_b32(s[2], 0),
      s_mov_b32(s[3], 0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][0], 0xDEAD42EF, "Only byte 1 should be modified")

  def test_global_store_b16_basic(self):
    """GLOBAL_STORE_B16 stores a 16-bit value from VDATA[15:0]."""
    TEST_OFFSET = 256
    instructions = [
      s_load_b64(s[2:3], s[80:81], 0, soffset=SrcEnum.NULL),
      s_waitcnt(lgkmcnt=0),
      s_mov_b32(s[4], 0xDEADBEEF),
      v_mov_b32_e32(v[2], s[4]),
      v_mov_b32_e32(v[0], 0),
      global_store_b32(addr=v[0], data=v[2], saddr=s[2:3], offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      s_mov_b32(s[4], 0xCAFE),
      v_mov_b32_e32(v[2], s[4]),
      global_store_b16(addr=v[0], data=v[2], saddr=s[2:3], offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      GLOBAL(GLOBALOp.GLOBAL_LOAD_B32, addr=v[0], vdst=v[3], data=v[3], saddr=s[2:3], offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      v_mov_b32_e32(v[0], v[3]),
      s_mov_b32(s[2], 0),
      s_mov_b32(s[3], 0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][0], 0xDEADCAFE, "Only lower 16 bits should be modified")

  def test_global_store_b16_high_half(self):
    """GLOBAL_STORE_B16 at offset+2 stores to high 16 bits."""
    TEST_OFFSET = 256
    instructions = [
      s_load_b64(s[2:3], s[80:81], 0, soffset=SrcEnum.NULL),
      s_waitcnt(lgkmcnt=0),
      s_mov_b32(s[4], 0xDEADBEEF),
      v_mov_b32_e32(v[2], s[4]),
      v_mov_b32_e32(v[0], 0),
      global_store_b32(addr=v[0], data=v[2], saddr=s[2:3], offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      s_mov_b32(s[4], 0xCAFE),
      v_mov_b32_e32(v[2], s[4]),
      global_store_b16(addr=v[0], data=v[2], saddr=s[2:3], offset=TEST_OFFSET+2),
      s_waitcnt(vmcnt=0),
      GLOBAL(GLOBALOp.GLOBAL_LOAD_B32, addr=v[0], vdst=v[3], data=v[3], saddr=s[2:3], offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      v_mov_b32_e32(v[0], v[3]),
      s_mov_b32(s[2], 0),
      s_mov_b32(s[3], 0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][0], 0xCAFEBEEF, "Only upper 16 bits should be modified")

  def test_global_store_b16_byte_offset_1(self):
    """GLOBAL_STORE_B16 at byte offset 1 stores bytes 1-2 within the same word."""
    TEST_OFFSET = 256
    instructions = [
      s_load_b64(s[2:3], s[80:81], 0, soffset=SrcEnum.NULL),
      s_waitcnt(lgkmcnt=0),
      s_mov_b32(s[4], 0xDDCCBBAA),
      v_mov_b32_e32(v[2], s[4]),
      v_mov_b32_e32(v[0], 0),
      global_store_b32(addr=v[0], data=v[2], saddr=s[2:3], offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      # Store 0xBEEF at byte offset 1 (bytes 1-2)
      s_mov_b32(s[4], 0xBEEF),
      v_mov_b32_e32(v[2], s[4]),
      global_store_b16(addr=v[0], data=v[2], saddr=s[2:3], offset=TEST_OFFSET+1),
      s_waitcnt(vmcnt=0),
      GLOBAL(GLOBALOp.GLOBAL_LOAD_B32, addr=v[0], vdst=v[3], data=v[3], saddr=s[2:3], offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      v_mov_b32_e32(v[0], v[3]),
      s_mov_b32(s[2], 0),
      s_mov_b32(s[3], 0),
    ]
    st = run_program(instructions, n_lanes=1)
    # Bytes 1-2 should be 0xBEEF (0xEF at byte 1, 0xBE at byte 2)
    # Original: 0xDDCCBBAA -> bytes [AA, BB, CC, DD]
    # After:    0xDDBEEFAA -> bytes [AA, EF, BE, DD]
    self.assertEqual(st.vgpr[0][0], 0xDDBEEFAA, "Bytes 1-2 should be 0xBEEF")

  def test_global_store_b16_cross_word_boundary(self):
    """GLOBAL_STORE_B16 at byte offset 3 crosses word boundary (byte 3 of word N, byte 0 of word N+1)."""
    TEST_OFFSET = 256
    instructions = [
      s_load_b64(s[2:3], s[80:81], 0, soffset=SrcEnum.NULL),
      s_waitcnt(lgkmcnt=0),
      # Initialize two consecutive words
      s_mov_b32(s[4], 0xDDCCBBAA),
      v_mov_b32_e32(v[2], s[4]),
      v_mov_b32_e32(v[0], 0),
      global_store_b32(addr=v[0], data=v[2], saddr=s[2:3], offset=TEST_OFFSET),
      s_mov_b32(s[4], 0x44332211),
      v_mov_b32_e32(v[2], s[4]),
      global_store_b32(addr=v[0], data=v[2], saddr=s[2:3], offset=TEST_OFFSET+4),
      s_waitcnt(vmcnt=0),
      # Store 0xBEEF at byte offset 3 (crosses word boundary)
      # Low byte (0xEF) goes to byte 3 of first word
      # High byte (0xBE) goes to byte 0 of second word
      s_mov_b32(s[4], 0xBEEF),
      v_mov_b32_e32(v[2], s[4]),
      global_store_b16(addr=v[0], data=v[2], saddr=s[2:3], offset=TEST_OFFSET+3),
      s_waitcnt(vmcnt=0),
      # Load back both words
      GLOBAL(GLOBALOp.GLOBAL_LOAD_B32, addr=v[0], vdst=v[3], data=v[3], saddr=s[2:3], offset=TEST_OFFSET),
      GLOBAL(GLOBALOp.GLOBAL_LOAD_B32, addr=v[0], vdst=v[4], data=v[4], saddr=s[2:3], offset=TEST_OFFSET+4),
      s_waitcnt(vmcnt=0),
      v_mov_b32_e32(v[0], v[3]),
      v_mov_b32_e32(v[1], v[4]),
      s_mov_b32(s[2], 0),
      s_mov_b32(s[3], 0),
    ]
    st = run_program(instructions, n_lanes=1)
    # First word: 0xDDCCBBAA -> 0xEFCCBBAA (byte 3 becomes 0xEF)
    # Second word: 0x44332211 -> 0x443322BE (byte 0 becomes 0xBE)
    self.assertEqual(st.vgpr[0][0], 0xEFCCBBAA, "Byte 3 of first word should be 0xEF")
    self.assertEqual(st.vgpr[0][1], 0x443322BE, "Byte 0 of second word should be 0xBE")

  def test_global_store_b64_basic(self):
    """GLOBAL_STORE_B64 stores 8 bytes from v[n:n+1] to memory."""
    TEST_OFFSET = 256
    instructions = [
      s_load_b64(s[2:3], s[80:81], 0, soffset=SrcEnum.NULL),
      s_waitcnt(lgkmcnt=0),
      s_mov_b32(s[4], 0xDEADBEEF),
      s_mov_b32(s[5], 0xCAFEBABE),
      v_mov_b32_e32(v[2], s[4]),
      v_mov_b32_e32(v[3], s[5]),
      v_mov_b32_e32(v[0], 0),
      global_store_b64(addr=v[0], data=v[2:3], saddr=s[2:3], offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      GLOBAL(GLOBALOp.GLOBAL_LOAD_B64, addr=v[0], vdst=v[4:5], data=v[4:5], saddr=s[2:3], offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      v_mov_b32_e32(v[0], v[4]),
      v_mov_b32_e32(v[1], v[5]),
      s_mov_b32(s[2], 0),
      s_mov_b32(s[3], 0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][0], 0xDEADBEEF)
    self.assertEqual(st.vgpr[0][1], 0xCAFEBABE)


class TestD16HiLoads(unittest.TestCase):
  """Tests for D16_HI load instructions that load into high 16 bits."""

  def test_global_load_d16_hi_b16_preserves_low_bits(self):
    """GLOBAL_LOAD_D16_HI_B16 must preserve low 16 bits of destination."""
    TEST_OFFSET = 256
    instructions = [
      s_load_b64(s[2:3], s[80:81], 0, soffset=SrcEnum.NULL),
      s_waitcnt(lgkmcnt=0),
      v_mov_b32_e32(v[0], s[2]),
      v_mov_b32_e32(v[1], s[3]),
      s_mov_b32(s[4], 0xCAFE),
      v_mov_b32_e32(v[2], s[4]),
      global_store_b16(addr=v[0:1], data=v[2], saddr=SrcEnum.NULL, offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      s_mov_b32(s[4], 0x0000BEEF),
      v_mov_b32_e32(v[3], s[4]),
      GLOBAL(GLOBALOp.GLOBAL_LOAD_D16_HI_B16, addr=v[0:1], vdst=v[3], data=v[3], saddr=SrcEnum.NULL, offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      v_mov_b32_e32(v[0], v[3]),
      v_mov_b32_e32(v[1], 0),
      s_mov_b32(s[2], 0),
      s_mov_b32(s[3], 0),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][0]
    self.assertEqual(result, 0xCAFEBEEF, f"Expected 0xCAFEBEEF, got 0x{result:08x}")

  def test_global_load_d16_hi_b16_data_differs_from_vdst(self):
    """GLOBAL_LOAD_D16_HI_B16 where data field differs from vdst."""
    TEST_OFFSET = 256
    instructions = [
      s_load_b64(s[2:3], s[80:81], 0, soffset=SrcEnum.NULL),
      s_waitcnt(lgkmcnt=0),
      s_mov_b32(s[4], 0xCAFE),
      v_mov_b32_e32(v[2], s[4]),
      v_mov_b32_e32(v[3], 0),
      global_store_b16(addr=v[3], data=v[2], saddr=s[2:3], offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      s_mov_b32(s[4], 0x0000DEAD),
      v_mov_b32_e32(v[0], s[4]),  # data field - should NOT affect result
      v_mov_b32_e32(v[1], 0),     # vdst - low bits should be preserved
      GLOBAL(GLOBALOp.GLOBAL_LOAD_D16_HI_B16, addr=v[1], vdst=v[1], data=v[0], saddr=s[2:3], offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      v_mov_b32_e32(v[0], v[1]),
      s_mov_b32(s[2], 0),
      s_mov_b32(s[3], 0),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][0]
    self.assertEqual(result, 0xCAFE0000, f"Expected 0xCAFE0000, got 0x{result:08x}")

  def test_global_load_d16_hi_u8_data_differs_from_vdst(self):
    """GLOBAL_LOAD_D16_HI_U8 where data field differs from vdst."""
    TEST_OFFSET = 256
    instructions = [
      s_load_b64(s[2:3], s[80:81], 0, soffset=SrcEnum.NULL),
      s_waitcnt(lgkmcnt=0),
      s_mov_b32(s[4], 0xAB),
      v_mov_b32_e32(v[2], s[4]),
      v_mov_b32_e32(v[3], 0),
      global_store_b8(addr=v[3], data=v[2], saddr=s[2:3], offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      s_mov_b32(s[4], 0x0000DEAD),
      v_mov_b32_e32(v[4], s[4]),  # data field
      s_mov_b32(s[4], 0x0000BEEF),
      v_mov_b32_e32(v[5], s[4]),  # vdst
      v_mov_b32_e32(v[3], 0),
      GLOBAL(GLOBALOp.GLOBAL_LOAD_D16_HI_U8, addr=v[3], vdst=v[5], data=v[4], saddr=s[2:3], offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      v_mov_b32_e32(v[0], v[5]),
      s_mov_b32(s[2], 0),
      s_mov_b32(s[3], 0),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][0]
    self.assertEqual(result, 0x00ABBEEF, f"Expected 0x00ABBEEF, got 0x{result:08x}")

  def test_global_load_d16_hi_b16_same_addr_and_dst_zero_addr(self):
    """GLOBAL_LOAD_D16_HI_B16 with same register for addr and vdst, addr value=0."""
    TEST_OFFSET = 256
    instructions = [
      s_load_b64(s[2:3], s[80:81], 0, soffset=SrcEnum.NULL),
      s_waitcnt(lgkmcnt=0),
      s_mov_b32(s[4], 0xCAFE),
      v_mov_b32_e32(v[2], s[4]),
      v_mov_b32_e32(v[3], 0),
      global_store_b16(addr=v[3], data=v[2], saddr=s[2:3], offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      v_mov_b32_e32(v[1], 0),
      GLOBAL(GLOBALOp.GLOBAL_LOAD_D16_HI_B16, addr=v[1], vdst=v[1], data=v[1], saddr=s[2:3], offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      v_mov_b32_e32(v[0], v[1]),
      s_mov_b32(s[2], 0),
      s_mov_b32(s[3], 0),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][0]
    self.assertEqual(result, 0xCAFE0000, f"Expected 0xCAFE0000, got 0x{result:08x}")

  def test_global_load_d16_hi_b16_tril_exact_pattern(self):
    """Exact pattern from tril() failure: data=v0 differs from vdst=v1."""
    TEST_OFFSET = 256
    instructions = [
      s_load_b64(s[2:3], s[80:81], 0, soffset=SrcEnum.NULL),
      s_waitcnt(lgkmcnt=0),
      s_mov_b32(s[4], 0x01010101),
      v_mov_b32_e32(v[10], s[4]),
      v_mov_b32_e32(v[3], 0),
      global_store_b32(addr=v[3], data=v[10], saddr=s[2:3], offset=TEST_OFFSET),
      global_store_b32(addr=v[3], data=v[10], saddr=s[2:3], offset=TEST_OFFSET+4),
      s_waitcnt(vmcnt=0),
      # Set v[0] to 0x0101 (simulating prior u16 load result)
      s_mov_b32(s[4], 0x0101),
      v_mov_b32_e32(v[0], s[4]),
      # Set v[1] to 0
      v_mov_b32_e32(v[1], 0),
      # Load using v[1] as addr AND vdst, but v[0] as data
      GLOBAL(GLOBALOp.GLOBAL_LOAD_D16_HI_B16, addr=v[1], vdst=v[1], data=v[0], saddr=s[2:3], offset=TEST_OFFSET+6),
      s_waitcnt(vmcnt=0),
      v_mov_b32_e32(v[0], v[1]),
      s_mov_b32(s[2], 0),
      s_mov_b32(s[3], 0),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][0]
    # Expected: hi=0x0101 (loaded), lo=0x0000 (from v1) -> 0x01010000
    self.assertEqual(result, 0x01010000, f"Expected 0x01010000, got 0x{result:08x}")

  def test_global_load_d16_hi_i8_data_differs_from_vdst(self):
    """GLOBAL_LOAD_D16_HI_I8 where data field differs from vdst."""
    TEST_OFFSET = 256
    instructions = [
      s_load_b64(s[2:3], s[80:81], 0, soffset=SrcEnum.NULL),
      s_waitcnt(lgkmcnt=0),
      s_mov_b32(s[4], 0x80),  # negative signed byte = -128
      v_mov_b32_e32(v[2], s[4]),
      v_mov_b32_e32(v[3], 0),
      global_store_b8(addr=v[3], data=v[2], saddr=s[2:3], offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      s_mov_b32(s[4], 0x0000DEAD),
      v_mov_b32_e32(v[4], s[4]),  # data field
      s_mov_b32(s[4], 0x0000BEEF),
      v_mov_b32_e32(v[5], s[4]),  # vdst
      v_mov_b32_e32(v[3], 0),
      GLOBAL(GLOBALOp.GLOBAL_LOAD_D16_HI_I8, addr=v[3], vdst=v[5], data=v[4], saddr=s[2:3], offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      v_mov_b32_e32(v[0], v[5]),
      s_mov_b32(s[2], 0),
      s_mov_b32(s[3], 0),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][0]
    # 0x80 sign-extended = 0xFF80, lo=0xBEEF -> 0xFF80BEEF
    self.assertEqual(result, 0xFF80BEEF, f"Expected 0xFF80BEEF, got 0x{result:08x}")

  def test_global_store_b64_tril_pattern(self):
    """Test the exact pattern from tril() kernel that was failing."""
    TEST_OFFSET = 256
    instructions = [
      s_load_b64(s[2:3], s[80:81], 0, soffset=SrcEnum.NULL),
      s_waitcnt(lgkmcnt=0),
      s_mov_b32(s[4], 0x01010101),
      v_mov_b32_e32(v[10], s[4]),
      v_mov_b32_e32(v[11], s[4]),
      s_mov_b32(s[4], 0x01),
      v_mov_b32_e32(v[12], s[4]),
      v_mov_b32_e32(v[0], 0),
      global_store_b64(addr=v[0], data=v[10:11], saddr=s[2:3], offset=TEST_OFFSET),
      global_store_b8(addr=v[0], data=v[12], saddr=s[2:3], offset=TEST_OFFSET+8),
      s_waitcnt(vmcnt=0),

      v_mov_b32_e32(v[2], 0),
      v_mov_b32_e32(v[1], 0),
      GLOBAL(GLOBALOp.GLOBAL_LOAD_U16, addr=v[2], vdst=v[0], data=v[0], saddr=s[2:3], offset=TEST_OFFSET+3),
      GLOBAL(GLOBALOp.GLOBAL_LOAD_D16_HI_B16, addr=v[1], vdst=v[1], data=v[1], saddr=s[2:3], offset=TEST_OFFSET+6),
      GLOBAL(GLOBALOp.GLOBAL_LOAD_U8, addr=v[2], vdst=v[3], data=v[3], saddr=s[2:3], offset=TEST_OFFSET),
      GLOBAL(GLOBALOp.GLOBAL_LOAD_U8, addr=v[2], vdst=v[4], data=v[4], saddr=s[2:3], offset=TEST_OFFSET+8),
      s_waitcnt(vmcnt=0),

      v_and_b32_e32(v[5], 0xffff, v[0]),
      v_lshlrev_b32_e32(v[0], 24, v[0]),
      v_lshrrev_b32_e32(v[5], 8, v[5]),
      v_or_b32_e32(v[0], v[3], v[0]),
      v_or_b32_e32(v[1], v[5], v[1]),

      global_store_b64(addr=v[2], data=v[0:1], saddr=s[2:3], offset=TEST_OFFSET+16),
      s_waitcnt(vmcnt=0),

      GLOBAL(GLOBALOp.GLOBAL_LOAD_B64, addr=v[2], vdst=v[6:7], data=v[6:7], saddr=s[2:3], offset=TEST_OFFSET+16),
      s_waitcnt(vmcnt=0),
      v_mov_b32_e32(v[0], v[6]),
      v_mov_b32_e32(v[1], v[7]),
      s_mov_b32(s[2], 0),
      s_mov_b32(s[3], 0),
    ]
    st = run_program(instructions, n_lanes=1)

    v0 = st.vgpr[0][0]
    v1 = st.vgpr[0][1]
    self.assertEqual(v0, 0x01000001, f"v0: expected 0x01000001, got 0x{v0:08x}")
    self.assertEqual(v1, 0x01010001, f"v1: expected 0x01010001, got 0x{v1:08x}")

    byte5 = (v1 >> 8) & 0xff
    self.assertEqual(byte5, 0x00, f"byte5: expected 0x00, got 0x{byte5:02x}")


class TestGlobalOffset(unittest.TestCase):
  """Tests for GLOBAL instructions with different offsets.

  These tests verify that instruction deduplication correctly handles different offset values.
  If offset is made dynamic incorrectly, instructions with different offsets may load/store wrong data.
  """

  def test_global_load_different_offsets(self):
    """Load from two different offsets and verify correct values."""
    instructions = [
      s_load_b64(s[2:3], s[80:81], 0, soffset=SrcEnum.NULL),
      s_waitcnt(lgkmcnt=0),
      v_mov_b32_e32(v[0], s[2]),
      v_mov_b32_e32(v[1], s[3]),
      # Store 0xAAAAAAAA at offset 100
      s_mov_b32(s[0], 0xAAAAAAAA),
      v_mov_b32_e32(v[2], s[0]),
      global_store_b32(addr=v[0:1], data=v[2], saddr=SrcEnum.NULL, offset=100),
      # Store 0xBBBBBBBB at offset 200
      s_mov_b32(s[0], 0xBBBBBBBB),
      v_mov_b32_e32(v[2], s[0]),
      global_store_b32(addr=v[0:1], data=v[2], saddr=SrcEnum.NULL, offset=200),
      s_waitcnt(vmcnt=0),
      # Load from offset 100 -> should get 0xAAAAAAAA
      GLOBAL(GLOBALOp.GLOBAL_LOAD_B32, addr=v[0:1], vdst=v[3], saddr=SrcEnum.NULL, offset=100),
      # Load from offset 200 -> should get 0xBBBBBBBB
      GLOBAL(GLOBALOp.GLOBAL_LOAD_B32, addr=v[0:1], vdst=v[4], saddr=SrcEnum.NULL, offset=200),
      s_waitcnt(vmcnt=0),
      v_mov_b32_e32(v[0], v[3]),
      v_mov_b32_e32(v[1], v[4]),
      s_mov_b32(s[2], 0),
      s_mov_b32(s[3], 0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][0], 0xAAAAAAAA, f"offset 100: expected 0xAAAAAAAA, got 0x{st.vgpr[0][0]:08x}")
    self.assertEqual(st.vgpr[0][1], 0xBBBBBBBB, f"offset 200: expected 0xBBBBBBBB, got 0x{st.vgpr[0][1]:08x}")

  def test_global_store_different_offsets(self):
    """Store to two different offsets and verify correct values."""
    instructions = [
      s_load_b64(s[2:3], s[80:81], 0, soffset=SrcEnum.NULL),
      s_waitcnt(lgkmcnt=0),
      v_mov_b32_e32(v[0], s[2]),
      v_mov_b32_e32(v[1], s[3]),
      # Store 0x11111111 at offset 300
      s_mov_b32(s[0], 0x11111111),
      v_mov_b32_e32(v[2], s[0]),
      global_store_b32(addr=v[0:1], data=v[2], saddr=SrcEnum.NULL, offset=300),
      # Store 0x22222222 at offset 400
      s_mov_b32(s[0], 0x22222222),
      v_mov_b32_e32(v[3], s[0]),
      global_store_b32(addr=v[0:1], data=v[3], saddr=SrcEnum.NULL, offset=400),
      s_waitcnt(vmcnt=0),
      # Load back to verify
      GLOBAL(GLOBALOp.GLOBAL_LOAD_B32, addr=v[0:1], vdst=v[4], saddr=SrcEnum.NULL, offset=300),
      GLOBAL(GLOBALOp.GLOBAL_LOAD_B32, addr=v[0:1], vdst=v[5], saddr=SrcEnum.NULL, offset=400),
      s_waitcnt(vmcnt=0),
      v_mov_b32_e32(v[0], v[4]),
      v_mov_b32_e32(v[1], v[5]),
      s_mov_b32(s[2], 0),
      s_mov_b32(s[3], 0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][0], 0x11111111, f"offset 300: expected 0x11111111, got 0x{st.vgpr[0][0]:08x}")
    self.assertEqual(st.vgpr[0][1], 0x22222222, f"offset 400: expected 0x22222222, got 0x{st.vgpr[0][1]:08x}")

  def test_global_negative_offset_no_saddr(self):
    """Test negative offset without saddr (VGPR pair for address).
    Store 0xAAAA at offset 100, 0xBBBB at offset 200.
    Load with offset -100 from vaddr pointing to base+200 -> should get 0xAAAA (at 100).
    Load with offset -100 from vaddr pointing to base+300 -> should get 0xBBBB (at 200)."""
    instructions = [
      s_load_b64(s[2:3], s[80:81], 0, soffset=SrcEnum.NULL),
      s_waitcnt(lgkmcnt=0),
      v_mov_b32_e32(v[0], s[2]),
      v_mov_b32_e32(v[1], s[3]),
      # Store 0xAAAAAAAA at offset 100, 0xBBBBBBBB at offset 200
      s_mov_b32(s[0], 0xAAAAAAAA),
      v_mov_b32_e32(v[2], s[0]),
      global_store_b32(addr=v[0:1], data=v[2], saddr=SrcEnum.NULL, offset=100),
      s_mov_b32(s[0], 0xBBBBBBBB),
      v_mov_b32_e32(v[2], s[0]),
      global_store_b32(addr=v[0:1], data=v[2], saddr=SrcEnum.NULL, offset=200),
      s_waitcnt(vmcnt=0),
      # vaddr = base+200, load with offset -100 -> should get value at 100
      s_add_u32(s[4], s[2], 200),
      s_addc_u32(s[5], s[3], 0),
      v_mov_b32_e32(v[4], s[4]),
      v_mov_b32_e32(v[5], s[5]),
      GLOBAL(GLOBALOp.GLOBAL_LOAD_B32, addr=v[4:5], vdst=v[6], saddr=SrcEnum.NULL, offset=-100),
      # vaddr = base+300, load with offset -100 -> should get value at 200
      s_add_u32(s[4], s[2], 300),
      s_addc_u32(s[5], s[3], 0),
      v_mov_b32_e32(v[4], s[4]),
      v_mov_b32_e32(v[5], s[5]),
      GLOBAL(GLOBALOp.GLOBAL_LOAD_B32, addr=v[4:5], vdst=v[7], saddr=SrcEnum.NULL, offset=-100),
      s_waitcnt(vmcnt=0),
      v_mov_b32_e32(v[0], v[6]),
      v_mov_b32_e32(v[1], v[7]),
      v_mov_b32_e32(v[4], 0),
      v_mov_b32_e32(v[5], 0),
      v_mov_b32_e32(v[6], 0),
      v_mov_b32_e32(v[7], 0),
      s_mov_b32(s[2], 0),
      s_mov_b32(s[3], 0),
      s_mov_b32(s[4], 0),
      s_mov_b32(s[5], 0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][0], 0xAAAAAAAA, f"offset 200-100=100: expected 0xAAAAAAAA, got 0x{st.vgpr[0][0]:08x}")
    self.assertEqual(st.vgpr[0][1], 0xBBBBBBBB, f"offset 300-100=200: expected 0xBBBBBBBB, got 0x{st.vgpr[0][1]:08x}")

  def test_global_negative_offset_with_saddr(self):
    """Test negative offset with saddr (SGPR pair for base address).
    Store 0xAAAA at offset 100, 0xBBBB at offset 200.
    Load with offset -100 from saddr pointing to base+200 -> should get 0xAAAA (at 100).
    Load with offset -100 from saddr pointing to base+300 -> should get 0xBBBB (at 200)."""
    instructions = [
      s_load_b64(s[2:3], s[80:81], 0, soffset=SrcEnum.NULL),
      s_waitcnt(lgkmcnt=0),
      v_mov_b32_e32(v[0], 0),
      # Store 0xAAAAAAAA at offset 100, 0xBBBBBBBB at offset 200
      s_mov_b32(s[0], 0xAAAAAAAA),
      v_mov_b32_e32(v[2], s[0]),
      global_store_b32(addr=v[0], data=v[2], saddr=s[2:3], offset=100),
      s_mov_b32(s[0], 0xBBBBBBBB),
      v_mov_b32_e32(v[2], s[0]),
      global_store_b32(addr=v[0], data=v[2], saddr=s[2:3], offset=200),
      s_waitcnt(vmcnt=0),
      # saddr = base+200, load with offset -100 -> should get value at 100
      s_add_u32(s[4], s[2], 200),
      s_addc_u32(s[5], s[3], 0),
      GLOBAL(GLOBALOp.GLOBAL_LOAD_B32, addr=v[0], vdst=v[6], saddr=s[4:5], offset=-100),
      # saddr = base+300, load with offset -100 -> should get value at 200
      s_add_u32(s[4], s[2], 300),
      s_addc_u32(s[5], s[3], 0),
      GLOBAL(GLOBALOp.GLOBAL_LOAD_B32, addr=v[0], vdst=v[7], saddr=s[4:5], offset=-100),
      s_waitcnt(vmcnt=0),
      v_mov_b32_e32(v[0], v[6]),
      v_mov_b32_e32(v[1], v[7]),
      v_mov_b32_e32(v[6], 0),
      v_mov_b32_e32(v[7], 0),
      s_mov_b32(s[2], 0),
      s_mov_b32(s[3], 0),
      s_mov_b32(s[4], 0),
      s_mov_b32(s[5], 0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][0], 0xAAAAAAAA, f"offset 200-100=100: expected 0xAAAAAAAA, got 0x{st.vgpr[0][0]:08x}")
    self.assertEqual(st.vgpr[0][1], 0xBBBBBBBB, f"offset 300-100=200: expected 0xBBBBBBBB, got 0x{st.vgpr[0][1]:08x}")


if __name__ == '__main__':
  unittest.main()
