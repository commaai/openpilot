"""Tests for SMEM instructions - scalar memory operations.

Includes: s_load_b32, s_load_b64, s_load_b128, s_load_b256, s_load_b512
Tests both immediate and register offset addressing modes.
"""
import unittest
from extra.assembly.amd.test.hw.helpers import *

# Use offset into output buffer for test data (output buffer is 2124 bytes)
TEST_OFFSET = 2000

# Cache invalidation sequence for scalar loads after vector stores
# s_wait_idle waits for all outstanding memory operations including cache flushes
CACHE_INV = [s_gl1_inv(), s_dcache_inv(), s_wait_idle()]

class TestSLoadRegisterOffset(unittest.TestCase):
  """Tests for s_load with register offset (soffset field).

  Bug: s_load_b32(s[dst], s[base:base+1], s[off]) ignores the register offset
  and only uses the immediate offset field. This causes incorrect memory loads
  when the offset comes from a register.
  """

  def test_s_load_b32_register_offset_basic(self):
    """s_load_b32 with register offset should load from base + reg_offset."""
    instructions = [
      # Load output buffer pointer from args
      s_load_b64(s[2:3], s[80:81], 0, soffset=NULL),
      s_waitcnt(lgkmcnt=0),
      # Store test values to output buffer: 0xAAAAAAAA at offset, 0xBBBBBBBB at offset+4
      s_mov_b32(s[4], 0xAAAAAAAA),
      s_mov_b32(s[5], 0xBBBBBBBB),
      v_mov_b32_e32(v[2], s[4]),
      v_mov_b32_e32(v[3], s[5]),
      v_mov_b32_e32(v[0], 0),
      global_store_b32(addr=v[0], data=v[2], saddr=s[2:3], offset=TEST_OFFSET),
      global_store_b32(addr=v[0], data=v[3], saddr=s[2:3], offset=TEST_OFFSET+4),
      s_waitcnt(vmcnt=0),
      *CACHE_INV,
      # Now test s_load with register offset
      # Put offset value in s[4]: offset = 4 bytes (1 dword)
      s_mov_b32(s[4], 4),
      # Load from out_ptr + TEST_OFFSET + s[4] (should load 0xBBBBBBBB)
      s_load_b32(s[5], s[2:3], s[4], offset=TEST_OFFSET),
      s_waitcnt(0),
      # Zero out pointer regs (different addresses in emu vs hw)
      s_mov_b32(s[2], 0), s_mov_b32(s[3], 0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[5], 0xBBBBBBBB,
      f"s_load with reg offset 4 should load 0xBBBBBBBB: s[5]=0x{st.sgpr[5]:08x}")

  def test_s_load_b32_register_offset_different_from_immediate(self):
    """s_load_b32 with register offset loads different data than immediate offset 0."""
    instructions = [
      # Load output buffer pointer from args
      s_load_b64(s[2:3], s[80:81], 0, soffset=NULL),
      s_waitcnt(lgkmcnt=0),
      # Store test values: 0xAAAAAAAA at offset, 0xBBBBBBBB at offset+4
      s_mov_b32(s[4], 0xAAAAAAAA),
      s_mov_b32(s[5], 0xBBBBBBBB),
      v_mov_b32_e32(v[2], s[4]),
      v_mov_b32_e32(v[3], s[5]),
      v_mov_b32_e32(v[0], 0),
      global_store_b32(addr=v[0], data=v[2], saddr=s[2:3], offset=TEST_OFFSET),
      global_store_b32(addr=v[0], data=v[3], saddr=s[2:3], offset=TEST_OFFSET+4),
      s_waitcnt(vmcnt=0),
      *CACHE_INV,
      # Load with immediate offset 0
      s_load_b32(s[5], s[2:3], NULL, offset=TEST_OFFSET),
      s_waitcnt(0),
      # Load with register offset 4
      s_mov_b32(s[4], 4),
      s_load_b32(s[6], s[2:3], s[4], offset=TEST_OFFSET),
      s_waitcnt(0),
      # Zero out pointer regs (different addresses in emu vs hw)
      s_mov_b32(s[2], 0), s_mov_b32(s[3], 0),
    ]
    st = run_program(instructions, n_lanes=1)
    # s[5] has dword at offset 0 (0xAAAAAAAA), s[6] has dword at offset 4 (0xBBBBBBBB)
    self.assertEqual(st.sgpr[5], 0xAAAAAAAA)
    self.assertEqual(st.sgpr[6], 0xBBBBBBBB)
    self.assertNotEqual(st.sgpr[5], st.sgpr[6],
      f"s_load with reg offset 4 should load different value than offset 0: "
      f"s[5]=0x{st.sgpr[5]:08x}, s[6]=0x{st.sgpr[6]:08x}")

  def test_s_load_b32_register_offset_same_as_dst(self):
    """s_load_b32 where soffset register is same as destination.

    This is the exact pattern that exposes the bug:
    s_load_b32(s[8], s[2:3], s[8])
    The offset should be read BEFORE the destination is overwritten.
    """
    instructions = [
      # Load output buffer pointer from args
      s_load_b64(s[2:3], s[80:81], 0, soffset=NULL),
      s_waitcnt(lgkmcnt=0),
      # Store test values: 0xAAAAAAAA at offset, 0xBBBBBBBB at offset+4
      s_mov_b32(s[6], 0xAAAAAAAA),
      s_mov_b32(s[7], 0xBBBBBBBB),
      v_mov_b32_e32(v[2], s[6]),
      v_mov_b32_e32(v[3], s[7]),
      v_mov_b32_e32(v[0], 0),
      global_store_b32(addr=v[0], data=v[2], saddr=s[2:3], offset=TEST_OFFSET),
      global_store_b32(addr=v[0], data=v[3], saddr=s[2:3], offset=TEST_OFFSET+4),
      s_waitcnt(vmcnt=0),
      *CACHE_INV,
      # Set up s[4] = 4 (offset in bytes)
      s_mov_b32(s[4], 4),
      # Load using s[4] as both offset and destination
      # Should load from base + 4, then store result in s[4]
      s_load_b32(s[4], s[2:3], s[4], offset=TEST_OFFSET),
      s_waitcnt(0),
      # Also load with immediate offset 4 for comparison
      s_load_b32(s[5], s[2:3], NULL, offset=TEST_OFFSET+4),
      s_waitcnt(0),
      # Zero out pointer regs (different addresses in emu vs hw)
      s_mov_b32(s[2], 0), s_mov_b32(s[3], 0),
    ]
    st = run_program(instructions, n_lanes=1)
    # s[4] and s[5] should have the same value (both loaded from offset 4 = 0xBBBBBBBB)
    self.assertEqual(st.sgpr[4], 0xBBBBBBBB)
    self.assertEqual(st.sgpr[4], st.sgpr[5],
      f"s_load with reg offset s[4]=4 should match immediate offset=4: "
      f"s[4]=0x{st.sgpr[4]:08x}, s[5]=0x{st.sgpr[5]:08x}")

  def test_s_load_b32_register_offset_zero(self):
    """s_load_b32 with register offset = 0 should be same as immediate offset 0."""
    instructions = [
      # Load output buffer pointer from args
      s_load_b64(s[2:3], s[80:81], 0, soffset=NULL),
      s_waitcnt(lgkmcnt=0),
      # Store test value: 0xDEADBEEF at offset
      s_mov_b32(s[7], 0xDEADBEEF),
      v_mov_b32_e32(v[2], s[7]),
      v_mov_b32_e32(v[0], 0),
      global_store_b32(addr=v[0], data=v[2], saddr=s[2:3], offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      *CACHE_INV,
      # Load with register offset 0
      s_mov_b32(s[4], 0),
      s_load_b32(s[5], s[2:3], s[4], offset=TEST_OFFSET),
      s_waitcnt(0),
      # Load with immediate offset 0
      s_load_b32(s[6], s[2:3], NULL, offset=TEST_OFFSET),
      s_waitcnt(0),
      # Zero out pointer regs (different addresses in emu vs hw)
      s_mov_b32(s[2], 0), s_mov_b32(s[3], 0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[5], 0xDEADBEEF)
    self.assertEqual(st.sgpr[5], st.sgpr[6],
      f"s_load with reg offset 0 should match immediate offset 0: "
      f"s[5]=0x{st.sgpr[5]:08x}, s[6]=0x{st.sgpr[6]:08x}")

  def test_s_load_b32_register_plus_immediate_offset(self):
    """s_load_b32 with both register and immediate offset should add them."""
    instructions = [
      # Load output buffer pointer from args
      s_load_b64(s[2:3], s[80:81], 0, soffset=NULL),
      s_waitcnt(lgkmcnt=0),
      # Store test values: 0xAAAAAAAA at offset, 0xBBBBBBBB at offset+4
      s_mov_b32(s[8], 0xAAAAAAAA),
      s_mov_b32(s[9], 0xBBBBBBBB),
      v_mov_b32_e32(v[2], s[8]),
      v_mov_b32_e32(v[3], s[9]),
      v_mov_b32_e32(v[0], 0),
      global_store_b32(addr=v[0], data=v[2], saddr=s[2:3], offset=TEST_OFFSET),
      global_store_b32(addr=v[0], data=v[3], saddr=s[2:3], offset=TEST_OFFSET+4),
      s_waitcnt(vmcnt=0),
      *CACHE_INV,
      # reg offset = 4, imm offset = 0 -> total offset = 4
      s_mov_b32(s[4], 4),
      s_load_b32(s[5], s[2:3], s[4], offset=TEST_OFFSET),
      s_waitcnt(0),
      # reg offset = 0, imm offset = 4 -> total offset = 4
      s_mov_b32(s[6], 0),
      s_load_b32(s[7], s[2:3], s[6], offset=TEST_OFFSET+4),
      s_waitcnt(0),
      # Zero out pointer regs (different addresses in emu vs hw)
      s_mov_b32(s[2], 0), s_mov_b32(s[3], 0),
    ]
    st = run_program(instructions, n_lanes=1)
    # Both should load from offset 4 (0xBBBBBBBB)
    self.assertEqual(st.sgpr[5], 0xBBBBBBBB)
    self.assertEqual(st.sgpr[7], 0xBBBBBBBB)
    self.assertEqual(st.sgpr[5], st.sgpr[7],
      f"reg_off=4 + imm_off=0 should equal reg_off=0 + imm_off=4: "
      f"s[5]=0x{st.sgpr[5]:08x}, s[7]=0x{st.sgpr[7]:08x}")


class TestSLoadMultiDword(unittest.TestCase):
  """Tests for multi-dword s_load with register offset."""

  def test_s_load_b64_register_offset(self):
    """s_load_b64 with register offset should load 2 dwords from base + reg_offset."""
    instructions = [
      # Load output buffer pointer from args
      s_load_b64(s[2:3], s[80:81], 0, soffset=NULL),
      s_waitcnt(lgkmcnt=0),
      # Store test values: 0xAAAAAAAA, 0xBBBBBBBB at offset
      s_mov_b32(s[10], 0xAAAAAAAA),
      s_mov_b32(s[11], 0xBBBBBBBB),
      v_mov_b32_e32(v[2], s[10]),
      v_mov_b32_e32(v[3], s[11]),
      v_mov_b32_e32(v[0], 0),
      global_store_b32(addr=v[0], data=v[2], saddr=s[2:3], offset=TEST_OFFSET),
      global_store_b32(addr=v[0], data=v[3], saddr=s[2:3], offset=TEST_OFFSET+4),
      s_waitcnt(vmcnt=0),
      *CACHE_INV,
      # Load with register offset 0
      s_mov_b32(s[4], 0),
      s_load_b64(s[6:7], s[2:3], s[4], offset=TEST_OFFSET),
      s_waitcnt(0),
      # Compare with immediate offset
      s_load_b64(s[8:9], s[2:3], NULL, offset=TEST_OFFSET),
      s_waitcnt(0),
      # Zero out pointer regs (different addresses in emu vs hw)
      s_mov_b32(s[2], 0), s_mov_b32(s[3], 0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[6], 0xAAAAAAAA)
    self.assertEqual(st.sgpr[7], 0xBBBBBBBB)
    self.assertEqual(st.sgpr[6], st.sgpr[8])
    self.assertEqual(st.sgpr[7], st.sgpr[9])

  def test_s_load_b128_register_offset(self):
    """s_load_b128 with register offset should load 4 dwords from base + reg_offset."""
    instructions = [
      # Load output buffer pointer from args
      s_load_b64(s[2:3], s[80:81], 0, soffset=NULL),
      s_waitcnt(lgkmcnt=0),
      # Store test values: 0xAAAAAAAA, 0xBBBBBBBB, 0xCCCCCCCC, 0xDDDDDDDD at offset
      v_mov_b32_e32(v[0], 0),
      s_mov_b32(s[14], 0xAAAAAAAA),
      v_mov_b32_e32(v[2], s[14]),
      global_store_b32(addr=v[0], data=v[2], saddr=s[2:3], offset=TEST_OFFSET),
      s_mov_b32(s[14], 0xBBBBBBBB),
      v_mov_b32_e32(v[2], s[14]),
      global_store_b32(addr=v[0], data=v[2], saddr=s[2:3], offset=TEST_OFFSET+4),
      s_mov_b32(s[14], 0xCCCCCCCC),
      v_mov_b32_e32(v[2], s[14]),
      global_store_b32(addr=v[0], data=v[2], saddr=s[2:3], offset=TEST_OFFSET+8),
      s_mov_b32(s[14], 0xDDDDDDDD),
      v_mov_b32_e32(v[2], s[14]),
      global_store_b32(addr=v[0], data=v[2], saddr=s[2:3], offset=TEST_OFFSET+12),
      s_waitcnt(vmcnt=0),
      *CACHE_INV,
      # Load with register offset 0 (s_load_b128 requires 4-aligned dest: s[4], s[8], s[12], ...)
      s_mov_b32(s[15], 0),
      s_load_b128(s[4:7], s[2:3], s[15], offset=TEST_OFFSET),
      s_waitcnt(0),
      # Compare with immediate offset
      s_load_b128(s[8:11], s[2:3], NULL, offset=TEST_OFFSET),
      s_waitcnt(0),
      # Zero out pointer regs (different addresses in emu vs hw)
      s_mov_b32(s[2], 0), s_mov_b32(s[3], 0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[4], 0xAAAAAAAA)
    self.assertEqual(st.sgpr[5], 0xBBBBBBBB)
    self.assertEqual(st.sgpr[6], 0xCCCCCCCC)
    self.assertEqual(st.sgpr[7], 0xDDDDDDDD)
    self.assertEqual(st.sgpr[4], st.sgpr[8])
    self.assertEqual(st.sgpr[5], st.sgpr[9])


class TestSLoadLarge(unittest.TestCase):
  """Tests for large s_load operations (s_load_b256, s_load_b512)."""

  def test_s_load_b256_basic(self):
    """s_load_b256 loads 8 consecutive dwords."""
    instructions = [
      s_load_b64(s[2:3], s[80:81], 0, soffset=NULL),
      s_waitcnt(lgkmcnt=0),
      v_mov_b32_e32(v[0], 0),
      # Store 8 test values
      s_mov_b32(s[20], 0x11111111),
      v_mov_b32_e32(v[2], s[20]),
      global_store_b32(addr=v[0], data=v[2], saddr=s[2:3], offset=TEST_OFFSET),
      s_mov_b32(s[20], 0x22222222),
      v_mov_b32_e32(v[2], s[20]),
      global_store_b32(addr=v[0], data=v[2], saddr=s[2:3], offset=TEST_OFFSET+4),
      s_mov_b32(s[20], 0x33333333),
      v_mov_b32_e32(v[2], s[20]),
      global_store_b32(addr=v[0], data=v[2], saddr=s[2:3], offset=TEST_OFFSET+8),
      s_mov_b32(s[20], 0x44444444),
      v_mov_b32_e32(v[2], s[20]),
      global_store_b32(addr=v[0], data=v[2], saddr=s[2:3], offset=TEST_OFFSET+12),
      s_mov_b32(s[20], 0x55555555),
      v_mov_b32_e32(v[2], s[20]),
      global_store_b32(addr=v[0], data=v[2], saddr=s[2:3], offset=TEST_OFFSET+16),
      s_mov_b32(s[20], 0x66666666),
      v_mov_b32_e32(v[2], s[20]),
      global_store_b32(addr=v[0], data=v[2], saddr=s[2:3], offset=TEST_OFFSET+20),
      s_mov_b32(s[20], 0x77777777),
      v_mov_b32_e32(v[2], s[20]),
      global_store_b32(addr=v[0], data=v[2], saddr=s[2:3], offset=TEST_OFFSET+24),
      s_mov_b32(s[20], 0x88888888),
      v_mov_b32_e32(v[2], s[20]),
      global_store_b32(addr=v[0], data=v[2], saddr=s[2:3], offset=TEST_OFFSET+28),
      s_waitcnt(vmcnt=0),
      *CACHE_INV,
      # Load all 8 dwords with s_load_b256
      s_load_b256(s[4:11], s[2:3], NULL, offset=TEST_OFFSET),
      s_waitcnt(lgkmcnt=0),
      s_mov_b32(s[2], 0), s_mov_b32(s[3], 0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[4], 0x11111111)
    self.assertEqual(st.sgpr[5], 0x22222222)
    self.assertEqual(st.sgpr[6], 0x33333333)
    self.assertEqual(st.sgpr[7], 0x44444444)
    self.assertEqual(st.sgpr[8], 0x55555555)
    self.assertEqual(st.sgpr[9], 0x66666666)
    self.assertEqual(st.sgpr[10], 0x77777777)
    self.assertEqual(st.sgpr[11], 0x88888888)

  def test_s_load_b512_basic(self):
    """s_load_b512 loads 16 consecutive dwords."""
    instructions = [
      s_load_b64(s[2:3], s[80:81], 0, soffset=NULL),
      s_waitcnt(lgkmcnt=0),
      v_mov_b32_e32(v[0], 0),
      # Store 16 test values (use a pattern: 0x10, 0x20, ..., 0x100)
      *[instr for i in range(16) for instr in [
        s_mov_b32(s[20], (i + 1) * 0x11111111),
        v_mov_b32_e32(v[2], s[20]),
        global_store_b32(addr=v[0], data=v[2], saddr=s[2:3], offset=TEST_OFFSET + i * 4),
      ]],
      s_waitcnt(vmcnt=0),
      *CACHE_INV,
      # Load all 16 dwords with s_load_b512
      s_load_b512(s[64:79], s[2:3], NULL, offset=TEST_OFFSET),
      s_waitcnt(lgkmcnt=0),
      # Copy results to lower regs for verification (since st.sgpr only has 16 regs in test)
      s_mov_b32(s[4], s[64]),
      s_mov_b32(s[5], s[65]),
      s_mov_b32(s[6], s[78]),
      s_mov_b32(s[7], s[79]),
      s_mov_b32(s[2], 0), s_mov_b32(s[3], 0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[4], 0x11111111, "first dword")
    self.assertEqual(st.sgpr[5], 0x22222222, "second dword")
    self.assertEqual(st.sgpr[6], 0xFFFFFFFF & (15 * 0x11111111), "15th dword")
    self.assertEqual(st.sgpr[7], 0xFFFFFFFF & (16 * 0x11111111), "16th dword")

  def test_s_load_b256_with_register_offset(self):
    """s_load_b256 with register offset should add reg offset to address."""
    instructions = [
      s_load_b64(s[2:3], s[80:81], 0, soffset=NULL),
      s_waitcnt(lgkmcnt=0),
      v_mov_b32_e32(v[0], 0),
      # Store pattern at TEST_OFFSET+8: skip first 2 dwords
      *[instr for i in range(8) for instr in [
        s_mov_b32(s[20], (i + 1) * 0x11111111),
        v_mov_b32_e32(v[2], s[20]),
        global_store_b32(addr=v[0], data=v[2], saddr=s[2:3], offset=TEST_OFFSET + 8 + i * 4),
      ]],
      s_waitcnt(vmcnt=0),
      *CACHE_INV,
      # Load with register offset 8
      s_mov_b32(s[20], 8),
      s_load_b256(s[4:11], s[2:3], s[20], offset=TEST_OFFSET),
      s_waitcnt(lgkmcnt=0),
      s_mov_b32(s[2], 0), s_mov_b32(s[3], 0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[4], 0x11111111, "first dword at offset+8")
    self.assertEqual(st.sgpr[5], 0x22222222, "second dword at offset+8")
    self.assertEqual(st.sgpr[11], 0x88888888, "last dword at offset+8")


class TestSLoadOffset(unittest.TestCase):
  """Tests for s_load with different immediate offsets.

  These tests verify that instruction deduplication correctly handles different offset values.
  If offset is made dynamic incorrectly, instructions with different offsets may load wrong data.
  """

  def test_s_load_different_offsets(self):
    """Load from two different offsets and verify correct values."""
    instructions = [
      s_load_b64(s[2:3], s[80:81], 0, soffset=NULL),
      s_waitcnt(lgkmcnt=0),
      v_mov_b32_e32(v[0], 0),
      # Store 0xAAAAAAAA at offset 100
      s_mov_b32(s[4], 0xAAAAAAAA),
      v_mov_b32_e32(v[2], s[4]),
      global_store_b32(addr=v[0], data=v[2], saddr=s[2:3], offset=100),
      # Store 0xBBBBBBBB at offset 200
      s_mov_b32(s[4], 0xBBBBBBBB),
      v_mov_b32_e32(v[2], s[4]),
      global_store_b32(addr=v[0], data=v[2], saddr=s[2:3], offset=200),
      s_waitcnt(vmcnt=0),
      *CACHE_INV,
      # Load from offset 100 -> should get 0xAAAAAAAA
      s_load_b32(s[4], s[2:3], NULL, offset=100),
      # Load from offset 200 -> should get 0xBBBBBBBB
      s_load_b32(s[5], s[2:3], NULL, offset=200),
      s_waitcnt(lgkmcnt=0),
      s_mov_b32(s[2], 0), s_mov_b32(s[3], 0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[4], 0xAAAAAAAA, f"offset 100: expected 0xAAAAAAAA, got 0x{st.sgpr[4]:08x}")
    self.assertEqual(st.sgpr[5], 0xBBBBBBBB, f"offset 200: expected 0xBBBBBBBB, got 0x{st.sgpr[5]:08x}")

  def test_s_load_negative_offset(self):
    """Test negative offset (21-bit signed).
    Store 0xAAAA at offset 100, 0xBBBB at offset 200.
    Load with offset -100 from base+200 -> should get 0xAAAA.
    Load with offset -100 from base+300 -> should get 0xBBBB."""
    instructions = [
      s_load_b64(s[2:3], s[80:81], 0, soffset=NULL),
      s_waitcnt(lgkmcnt=0),
      v_mov_b32_e32(v[0], 0),
      # Store 0xAAAAAAAA at offset 100, 0xBBBBBBBB at offset 200
      s_mov_b32(s[8], 0xAAAAAAAA),
      v_mov_b32_e32(v[2], s[8]),
      global_store_b32(addr=v[0], data=v[2], saddr=s[2:3], offset=100),
      s_mov_b32(s[8], 0xBBBBBBBB),
      v_mov_b32_e32(v[2], s[8]),
      global_store_b32(addr=v[0], data=v[2], saddr=s[2:3], offset=200),
      s_waitcnt(vmcnt=0),
      *CACHE_INV,
      # base+200, load with offset -100 -> should get value at 100
      s_add_u32(s[6], s[2], 200),
      s_addc_u32(s[7], s[3], 0),
      s_load_b32(s[4], s[6:7], NULL, offset=-100),
      # base+300, load with offset -100 -> should get value at 200
      s_add_u32(s[6], s[2], 300),
      s_addc_u32(s[7], s[3], 0),
      s_load_b32(s[5], s[6:7], NULL, offset=-100),
      s_waitcnt(lgkmcnt=0),
      s_mov_b32(s[2], 0),
      s_mov_b32(s[3], 0),
      s_mov_b32(s[6], 0),
      s_mov_b32(s[7], 0),
      s_mov_b32(s[8], 0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[4], 0xAAAAAAAA, f"offset 200-100=100: expected 0xAAAAAAAA, got 0x{st.sgpr[4]:08x}")
    self.assertEqual(st.sgpr[5], 0xBBBBBBBB, f"offset 300-100=200: expected 0xBBBBBBBB, got 0x{st.sgpr[5]:08x}")


if __name__ == '__main__':
  unittest.main()
