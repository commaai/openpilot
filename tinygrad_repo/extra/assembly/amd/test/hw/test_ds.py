"""Tests for DS instructions - data share (LDS) operations.

Includes: ds_store_b32, ds_load_b32, ds_store_2addr_*, ds_load_2addr_*,
          ds_add_*, ds_max_*, ds_min_*, ds_and_*, ds_or_*, ds_xor_*,
          ds_inc_*, ds_dec_*, ds_cmpstore_*, ds_storexchg_*
"""
import unittest
from extra.assembly.amd.test.hw.helpers import *

class TestDS2Addr(unittest.TestCase):
  """Tests for DS_*_2ADDR instructions."""

  def test_ds_store_load_2addr_b32(self):
    """DS_STORE_2ADDR_B32 and DS_LOAD_2ADDR_B32 with offset * 4."""
    instructions = [
      v_mov_b32_e32(v[10], 0),
      s_mov_b32(s[0], 0xAAAAAAAA),
      v_mov_b32_e32(v[0], s[0]),
      s_mov_b32(s[0], 0xBBBBBBBB),
      v_mov_b32_e32(v[1], s[0]),
      DS(DSOp.DS_STORE_2ADDR_B32, addr=v[10], data0=v[0], data1=v[1], vdst=v[0], offset0=0, offset1=1),
      s_waitcnt(lgkmcnt=0),
      DS(DSOp.DS_LOAD_2ADDR_B32, addr=v[10], vdst=v[2:3], offset0=0, offset1=1),
      s_waitcnt(lgkmcnt=0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][2], 0xAAAAAAAA)
    self.assertEqual(st.vgpr[0][3], 0xBBBBBBBB)

  def test_ds_store_load_2addr_b64(self):
    """DS_STORE_2ADDR_B64 and DS_LOAD_2ADDR_B64."""
    instructions = [
      v_mov_b32_e32(v[10], 0),
      s_mov_b32(s[0], 0xDEADBEEF),
      v_mov_b32_e32(v[0], s[0]),
      s_mov_b32(s[0], 0xCAFEBABE),
      v_mov_b32_e32(v[1], s[0]),
      s_mov_b32(s[0], 0x12345678),
      v_mov_b32_e32(v[2], s[0]),
      s_mov_b32(s[0], 0x9ABCDEF0),
      v_mov_b32_e32(v[3], s[0]),
      DS(DSOp.DS_STORE_2ADDR_B64, addr=v[10], data0=v[0:1], data1=v[2:3], vdst=v[0], offset0=0, offset1=2),
      s_waitcnt(lgkmcnt=0),
      DS(DSOp.DS_LOAD_2ADDR_B64, addr=v[10], vdst=v[4:7], offset0=0, offset1=2),
      s_waitcnt(lgkmcnt=0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][4], 0xDEADBEEF)
    self.assertEqual(st.vgpr[0][5], 0xCAFEBABE)
    self.assertEqual(st.vgpr[0][6], 0x12345678)
    self.assertEqual(st.vgpr[0][7], 0x9ABCDEF0)


class TestDS2AddrMore(unittest.TestCase):
  """Additional DS_*_2ADDR tests."""

  def test_ds_store_load_2addr_b32_nonzero_offsets(self):
    """DS_STORE_2ADDR_B32 with non-zero offsets (offset*4 scaling)."""
    instructions = [
      v_mov_b32_e32(v[10], 0),
      s_mov_b32(s[2], 0x11111111),
      v_mov_b32_e32(v[0], s[2]),
      s_mov_b32(s[2], 0x22222222),
      v_mov_b32_e32(v[1], s[2]),
      DS(DSOp.DS_STORE_2ADDR_B32, addr=v[10], data0=v[0], data1=v[1], vdst=v[0], offset0=2, offset1=5),
      s_waitcnt(lgkmcnt=0),
      DS(DSOp.DS_LOAD_2ADDR_B32, addr=v[10], vdst=v[2:3], offset0=2, offset1=5),
      s_waitcnt(lgkmcnt=0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][2], 0x11111111, "v2 should have value from offset 8 (2*4)")
    self.assertEqual(st.vgpr[0][3], 0x22222222, "v3 should have value from offset 20 (5*4)")

  def test_ds_2addr_b64_no_overlap(self):
    """DS_LOAD_2ADDR_B64 with adjacent offsets should not overlap."""
    instructions = [
      v_mov_b32_e32(v[10], 0),
      s_mov_b32(s[2], 0x11111111),
      v_mov_b32_e32(v[0], s[2]),
      ds_store_b32(addr=v[10], data0=v[0], offset0=0),
      s_mov_b32(s[2], 0x22222222),
      v_mov_b32_e32(v[0], s[2]),
      ds_store_b32(addr=v[10], data0=v[0], offset0=4),
      s_mov_b32(s[2], 0x33333333),
      v_mov_b32_e32(v[0], s[2]),
      ds_store_b32(addr=v[10], data0=v[0], offset0=8),
      s_mov_b32(s[2], 0x44444444),
      v_mov_b32_e32(v[0], s[2]),
      ds_store_b32(addr=v[10], data0=v[0], offset0=12),
      s_waitcnt(lgkmcnt=0),
      DS(DSOp.DS_LOAD_2ADDR_B64, addr=v[10], vdst=v[4:7], offset0=0, offset1=1),
      s_waitcnt(lgkmcnt=0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][4], 0x11111111, "v4 should be 0x11111111")
    self.assertEqual(st.vgpr[0][5], 0x22222222, "v5 should be 0x22222222")
    self.assertEqual(st.vgpr[0][6], 0x33333333, "v6 should be 0x33333333")
    self.assertEqual(st.vgpr[0][7], 0x44444444, "v7 should be 0x44444444")

  def test_ds_load_2addr_b32_no_overwrite(self):
    """DS_LOAD_2ADDR_B32 should only write 2 VGPRs."""
    instructions = [
      v_mov_b32_e32(v[10], 0),
      s_mov_b32(s[2], 0xAAAAAAAA),
      v_mov_b32_e32(v[0], s[2]),
      s_mov_b32(s[2], 0xBBBBBBBB),
      v_mov_b32_e32(v[1], s[2]),
      DS(DSOp.DS_STORE_2ADDR_B32, addr=v[10], data0=v[0], data1=v[1], vdst=v[0], offset0=0, offset1=1),
      s_waitcnt(lgkmcnt=0),
      s_mov_b32(s[2], 0xDEADBEEF),
      v_mov_b32_e32(v[4], s[2]),  # Sentinel
      DS(DSOp.DS_LOAD_2ADDR_B32, addr=v[10], vdst=v[2:3], offset0=0, offset1=1),
      s_waitcnt(lgkmcnt=0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][2], 0xAAAAAAAA)
    self.assertEqual(st.vgpr[0][3], 0xBBBBBBBB)
    self.assertEqual(st.vgpr[0][4], 0xDEADBEEF, "v4 should be untouched")

  def test_ds_load_b64_no_overwrite(self):
    """DS_LOAD_B64 should only write 2 VGPRs."""
    instructions = [
      v_mov_b32_e32(v[10], 0),
      s_mov_b32(s[2], 0xDEADBEEF),
      v_mov_b32_e32(v[0], s[2]),
      s_mov_b32(s[2], 0xCAFEBABE),
      v_mov_b32_e32(v[1], s[2]),
      ds_store_b64(addr=v[10], data0=v[0:1], offset0=0),
      s_waitcnt(lgkmcnt=0),
      s_mov_b32(s[2], 0x12345678),
      v_mov_b32_e32(v[4], s[2]),  # Sentinel
      ds_load_b64(addr=v[10], vdst=v[2:3], offset0=0),
      s_waitcnt(lgkmcnt=0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][2], 0xDEADBEEF)
    self.assertEqual(st.vgpr[0][3], 0xCAFEBABE)
    self.assertEqual(st.vgpr[0][4], 0x12345678, "v4 should be untouched")


class TestDSB96(unittest.TestCase):
  """Tests for DS_STORE_B96 and DS_LOAD_B96 (96-bit / 3 dwords)."""

  def test_ds_store_load_b96(self):
    """DS_STORE_B96 stores 3 VGPRs, DS_LOAD_B96 loads them back."""
    instructions = [
      v_mov_b32_e32(v[10], 0),
      s_mov_b32(s[0], 0x11111111),
      v_mov_b32_e32(v[0], s[0]),
      s_mov_b32(s[0], 0x22222222),
      v_mov_b32_e32(v[1], s[0]),
      s_mov_b32(s[0], 0x33333333),
      v_mov_b32_e32(v[2], s[0]),
      ds_store_b96(addr=v[10], data0=v[0:2]),
      s_waitcnt(lgkmcnt=0),
      ds_load_b96(addr=v[10], vdst=v[4:6]),
      s_waitcnt(lgkmcnt=0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][4], 0x11111111, "v4 should have first dword")
    self.assertEqual(st.vgpr[0][5], 0x22222222, "v5 should have second dword")
    self.assertEqual(st.vgpr[0][6], 0x33333333, "v6 should have third dword")

  def test_ds_store_b96_with_offset(self):
    """DS_STORE_B96 with non-zero offset."""
    instructions = [
      v_mov_b32_e32(v[10], 0),
      s_mov_b32(s[0], 0xAAAAAAAA),
      v_mov_b32_e32(v[0], s[0]),
      s_mov_b32(s[0], 0xBBBBBBBB),
      v_mov_b32_e32(v[1], s[0]),
      s_mov_b32(s[0], 0xCCCCCCCC),
      v_mov_b32_e32(v[2], s[0]),
      DS(DSOp.DS_STORE_B96, addr=v[10], data0=v[0:2], offset0=12),
      s_waitcnt(lgkmcnt=0),
      DS(DSOp.DS_LOAD_B96, addr=v[10], vdst=v[4:6], offset0=12),
      s_waitcnt(lgkmcnt=0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][4], 0xAAAAAAAA)
    self.assertEqual(st.vgpr[0][5], 0xBBBBBBBB)
    self.assertEqual(st.vgpr[0][6], 0xCCCCCCCC)


class TestDSB128(unittest.TestCase):
  """Tests for DS_STORE_B128 and DS_LOAD_B128 (128-bit / 4 dwords)."""

  def test_ds_store_load_b128(self):
    """DS_STORE_B128 stores 4 VGPRs, DS_LOAD_B128 loads them back."""
    instructions = [
      v_mov_b32_e32(v[10], 0),
      s_mov_b32(s[0], 0x11111111),
      v_mov_b32_e32(v[0], s[0]),
      s_mov_b32(s[0], 0x22222222),
      v_mov_b32_e32(v[1], s[0]),
      s_mov_b32(s[0], 0x33333333),
      v_mov_b32_e32(v[2], s[0]),
      s_mov_b32(s[0], 0x44444444),
      v_mov_b32_e32(v[3], s[0]),
      ds_store_b128(addr=v[10], data0=v[0:3]),
      s_waitcnt(lgkmcnt=0),
      ds_load_b128(addr=v[10], vdst=v[4:7]),
      s_waitcnt(lgkmcnt=0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][4], 0x11111111, "v4 should have first dword")
    self.assertEqual(st.vgpr[0][5], 0x22222222, "v5 should have second dword")
    self.assertEqual(st.vgpr[0][6], 0x33333333, "v6 should have third dword")
    self.assertEqual(st.vgpr[0][7], 0x44444444, "v7 should have fourth dword")

  def test_ds_store_b128_with_offset(self):
    """DS_STORE_B128 with non-zero offset."""
    instructions = [
      v_mov_b32_e32(v[10], 0),
      s_mov_b32(s[0], 0xAAAAAAAA),
      v_mov_b32_e32(v[0], s[0]),
      s_mov_b32(s[0], 0xBBBBBBBB),
      v_mov_b32_e32(v[1], s[0]),
      s_mov_b32(s[0], 0xCCCCCCCC),
      v_mov_b32_e32(v[2], s[0]),
      s_mov_b32(s[0], 0xDDDDDDDD),
      v_mov_b32_e32(v[3], s[0]),
      DS(DSOp.DS_STORE_B128, addr=v[10], data0=v[0:3], offset0=16),
      s_waitcnt(lgkmcnt=0),
      DS(DSOp.DS_LOAD_B128, addr=v[10], vdst=v[4:7], offset0=16),
      s_waitcnt(lgkmcnt=0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][4], 0xAAAAAAAA)
    self.assertEqual(st.vgpr[0][5], 0xBBBBBBBB)
    self.assertEqual(st.vgpr[0][6], 0xCCCCCCCC)
    self.assertEqual(st.vgpr[0][7], 0xDDDDDDDD)


class TestDSAtomic(unittest.TestCase):
  """Tests for DS atomic operations."""

  def test_ds_max_rtn_u32(self):
    """DS_MAX_RTN_U32: atomically store max and return old value."""
    instructions = [
      v_mov_b32_e32(v[10], 0),
      s_mov_b32(s[2], 100),
      v_mov_b32_e32(v[0], s[2]),
      ds_store_b32(addr=v[10], data0=v[0], offset0=0),
      s_waitcnt(lgkmcnt=0),
      s_mov_b32(s[2], 200),
      v_mov_b32_e32(v[1], s[2]),
      ds_max_rtn_u32(addr=v[10], data0=v[1], vdst=v[2], offset0=0),
      s_waitcnt(lgkmcnt=0),
      ds_load_b32(addr=v[10], vdst=v[3], offset0=0),
      s_waitcnt(lgkmcnt=0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][2], 100, "v2 should have old value (100)")
    self.assertEqual(st.vgpr[0][3], 200, "v3 should have max(100, 200) = 200")

  def test_ds_min_rtn_u32(self):
    """DS_MIN_RTN_U32: atomically store min and return old value."""
    instructions = [
      v_mov_b32_e32(v[10], 0),
      s_mov_b32(s[2], 200),
      v_mov_b32_e32(v[0], s[2]),
      ds_store_b32(addr=v[10], data0=v[0], offset0=0),
      s_waitcnt(lgkmcnt=0),
      s_mov_b32(s[2], 100),
      v_mov_b32_e32(v[1], s[2]),
      ds_min_rtn_u32(addr=v[10], data0=v[1], vdst=v[2], offset0=0),
      s_waitcnt(lgkmcnt=0),
      ds_load_b32(addr=v[10], vdst=v[3], offset0=0),
      s_waitcnt(lgkmcnt=0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][2], 200)
    self.assertEqual(st.vgpr[0][3], 100)

  def test_ds_and_rtn_b32(self):
    """DS_AND_RTN_B32: atomically AND and return old value."""
    instructions = [
      v_mov_b32_e32(v[10], 0),
      s_mov_b32(s[2], 0xFF00FF00),
      v_mov_b32_e32(v[0], s[2]),
      ds_store_b32(addr=v[10], data0=v[0], offset0=0),
      s_waitcnt(lgkmcnt=0),
      s_mov_b32(s[2], 0xFFFF0000),
      v_mov_b32_e32(v[1], s[2]),
      ds_and_rtn_b32(addr=v[10], data0=v[1], vdst=v[2], offset0=0),
      s_waitcnt(lgkmcnt=0),
      ds_load_b32(addr=v[10], vdst=v[3], offset0=0),
      s_waitcnt(lgkmcnt=0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][2], 0xFF00FF00)
    self.assertEqual(st.vgpr[0][3], 0xFF000000)

  def test_ds_or_rtn_b32(self):
    """DS_OR_RTN_B32: atomically OR and return old value."""
    instructions = [
      v_mov_b32_e32(v[10], 0),
      s_mov_b32(s[2], 0x00FF0000),
      v_mov_b32_e32(v[0], s[2]),
      ds_store_b32(addr=v[10], data0=v[0], offset0=0),
      s_waitcnt(lgkmcnt=0),
      s_mov_b32(s[2], 0x000000FF),
      v_mov_b32_e32(v[1], s[2]),
      ds_or_rtn_b32(addr=v[10], data0=v[1], vdst=v[2], offset0=0),
      s_waitcnt(lgkmcnt=0),
      ds_load_b32(addr=v[10], vdst=v[3], offset0=0),
      s_waitcnt(lgkmcnt=0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][2], 0x00FF0000)
    self.assertEqual(st.vgpr[0][3], 0x00FF00FF)

  def test_ds_xor_rtn_b32(self):
    """DS_XOR_RTN_B32: atomically XOR and return old value."""
    instructions = [
      v_mov_b32_e32(v[10], 0),
      s_mov_b32(s[2], 0xAAAAAAAA),
      v_mov_b32_e32(v[0], s[2]),
      ds_store_b32(addr=v[10], data0=v[0], offset0=0),
      s_waitcnt(lgkmcnt=0),
      s_mov_b32(s[2], 0xFFFFFFFF),
      v_mov_b32_e32(v[1], s[2]),
      ds_xor_rtn_b32(addr=v[10], data0=v[1], vdst=v[2], offset0=0),
      s_waitcnt(lgkmcnt=0),
      ds_load_b32(addr=v[10], vdst=v[3], offset0=0),
      s_waitcnt(lgkmcnt=0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][2], 0xAAAAAAAA)
    self.assertEqual(st.vgpr[0][3], 0x55555555)

  def test_ds_inc_rtn_u32(self):
    """DS_INC_RTN_U32: increment with wrap."""
    instructions = [
      v_mov_b32_e32(v[10], 0),
      s_mov_b32(s[2], 5),
      v_mov_b32_e32(v[0], s[2]),
      ds_store_b32(addr=v[10], data0=v[0], offset0=0),
      s_waitcnt(lgkmcnt=0),
      s_mov_b32(s[2], 10),  # limit
      v_mov_b32_e32(v[1], s[2]),
      ds_inc_rtn_u32(addr=v[10], data0=v[1], vdst=v[2], offset0=0),
      s_waitcnt(lgkmcnt=0),
      ds_load_b32(addr=v[10], vdst=v[3], offset0=0),
      s_waitcnt(lgkmcnt=0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][2], 5)
    self.assertEqual(st.vgpr[0][3], 6)

  def test_ds_dec_rtn_u32(self):
    """DS_DEC_RTN_U32: decrement with wrap."""
    instructions = [
      v_mov_b32_e32(v[10], 0),
      s_mov_b32(s[2], 5),
      v_mov_b32_e32(v[0], s[2]),
      ds_store_b32(addr=v[10], data0=v[0], offset0=0),
      s_waitcnt(lgkmcnt=0),
      s_mov_b32(s[2], 10),  # limit
      v_mov_b32_e32(v[1], s[2]),
      ds_dec_rtn_u32(addr=v[10], data0=v[1], vdst=v[2], offset0=0),
      s_waitcnt(lgkmcnt=0),
      ds_load_b32(addr=v[10], vdst=v[3], offset0=0),
      s_waitcnt(lgkmcnt=0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][2], 5)
    self.assertEqual(st.vgpr[0][3], 4)

  def test_ds_cmpstore_b32_match(self):
    """DS_CMPSTORE_B32: conditional store when compare matches."""
    instructions = [
      v_mov_b32_e32(v[10], 0),
      s_mov_b32(s[2], 100),
      v_mov_b32_e32(v[0], s[2]),
      ds_store_b32(addr=v[10], data0=v[0], offset0=0),
      s_waitcnt(lgkmcnt=0),
      s_mov_b32(s[2], 200),
      v_mov_b32_e32(v[1], s[2]),  # new value
      s_mov_b32(s[2], 100),
      v_mov_b32_e32(v[2], s[2]),  # compare = 100 (matches)
      ds_cmpstore_b32(addr=v[10], data0=v[1], data1=v[2], vdst=v[3], offset0=0),
      s_waitcnt(lgkmcnt=0),
      ds_load_b32(addr=v[10], vdst=v[4], offset0=0),
      s_waitcnt(lgkmcnt=0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][4], 200)

  def test_ds_cmpstore_b32_no_match(self):
    """DS_CMPSTORE_B32: no store when compare doesn't match."""
    instructions = [
      v_mov_b32_e32(v[10], 0),
      s_mov_b32(s[2], 100),
      v_mov_b32_e32(v[0], s[2]),
      ds_store_b32(addr=v[10], data0=v[0], offset0=0),
      s_waitcnt(lgkmcnt=0),
      s_mov_b32(s[2], 200),
      v_mov_b32_e32(v[1], s[2]),  # new value
      s_mov_b32(s[2], 50),
      v_mov_b32_e32(v[2], s[2]),  # compare = 50 (doesn't match)
      ds_cmpstore_b32(addr=v[10], data0=v[1], data1=v[2], vdst=v[3], offset0=0),
      s_waitcnt(lgkmcnt=0),
      ds_load_b32(addr=v[10], vdst=v[4], offset0=0),
      s_waitcnt(lgkmcnt=0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][4], 100)

  def test_ds_max_u32_no_rtn(self):
    """DS_MAX_U32 (no RTN): atomically store max, no return value."""
    instructions = [
      v_mov_b32_e32(v[10], 0),
      s_mov_b32(s[2], 100),
      v_mov_b32_e32(v[0], s[2]),
      ds_store_b32(addr=v[10], data0=v[0], offset0=0),
      s_waitcnt(lgkmcnt=0),
      s_mov_b32(s[2], 200),
      v_mov_b32_e32(v[1], s[2]),
      ds_max_u32(addr=v[10], data0=v[1], vdst=v[2], offset0=0),
      s_waitcnt(lgkmcnt=0),
      ds_load_b32(addr=v[10], vdst=v[3], offset0=0),
      s_waitcnt(lgkmcnt=0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][3], 200, "v3 should have max(100, 200) = 200")

  def test_ds_add_u32_no_rtn_preserves_vdst(self):
    """DS_ADD_U32 (no RTN) should NOT write to vdst."""
    instructions = [
      v_mov_b32_e32(v[10], 0),
      s_mov_b32(s[2], 0xDEADBEEF),
      v_mov_b32_e32(v[2], s[2]),  # sentinel
      s_mov_b32(s[2], 100),
      v_mov_b32_e32(v[0], s[2]),
      ds_store_b32(addr=v[10], data0=v[0], offset0=0),
      s_waitcnt(lgkmcnt=0),
      s_mov_b32(s[2], 50),
      v_mov_b32_e32(v[1], s[2]),
      ds_add_u32(addr=v[10], data0=v[1], vdst=v[2], offset0=0),
      s_waitcnt(lgkmcnt=0),
      ds_load_b32(addr=v[10], vdst=v[3], offset0=0),
      s_waitcnt(lgkmcnt=0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][2], 0xDEADBEEF, "v2 should preserve sentinel")
    self.assertEqual(st.vgpr[0][3], 150, "v3 should have 100 + 50 = 150")

  def test_ds_add_rtn_u32_writes_vdst(self):
    """DS_ADD_RTN_U32 should write old value to vdst."""
    instructions = [
      v_mov_b32_e32(v[10], 0),
      s_mov_b32(s[2], 0xDEADBEEF),
      v_mov_b32_e32(v[2], s[2]),  # sentinel
      s_mov_b32(s[2], 100),
      v_mov_b32_e32(v[0], s[2]),
      ds_store_b32(addr=v[10], data0=v[0], offset0=0),
      s_waitcnt(lgkmcnt=0),
      s_mov_b32(s[2], 50),
      v_mov_b32_e32(v[1], s[2]),
      ds_add_rtn_u32(addr=v[10], data0=v[1], vdst=v[2], offset0=0),
      s_waitcnt(lgkmcnt=0),
      ds_load_b32(addr=v[10], vdst=v[3], offset0=0),
      s_waitcnt(lgkmcnt=0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][2], 100, "v2 should have old value (100)")
    self.assertEqual(st.vgpr[0][3], 150, "v3 should have 100 + 50 = 150")

  def test_ds_dec_rtn_u32_wrap(self):
    """DS_DEC_RTN_U32: decrement wraps when value is 0 or > limit."""
    instructions = [
      v_mov_b32_e32(v[10], 0),
      s_mov_b32(s[2], 0),  # Start at 0
      v_mov_b32_e32(v[0], s[2]),
      ds_store_b32(addr=v[10], data0=v[0], offset0=0),
      s_waitcnt(lgkmcnt=0),
      s_mov_b32(s[2], 10),  # limit
      v_mov_b32_e32(v[1], s[2]),
      ds_dec_rtn_u32(addr=v[10], data0=v[1], vdst=v[2], offset0=0),
      s_waitcnt(lgkmcnt=0),
      ds_load_b32(addr=v[10], vdst=v[3], offset0=0),
      s_waitcnt(lgkmcnt=0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][2], 0, "v2 should have old value (0)")
    # When mem == 0 or mem > limit, result = limit
    self.assertEqual(st.vgpr[0][3], 10, "v3 should wrap to limit (10)")


class TestDSStorexchg(unittest.TestCase):
  """Tests for DS_STOREXCHG instructions."""

  def test_ds_storexchg_rtn_b32(self):
    """DS_STOREXCHG_RTN_B32: exchange value and return old."""
    instructions = [
      v_mov_b32_e32(v[10], 0),
      s_mov_b32(s[0], 0xAAAAAAAA),
      v_mov_b32_e32(v[0], s[0]),
      ds_store_b32(addr=v[10], data0=v[0], offset0=0),
      s_waitcnt(lgkmcnt=0),
      s_mov_b32(s[0], 0xBBBBBBBB),
      v_mov_b32_e32(v[1], s[0]),
      DS(DSOp.DS_STOREXCHG_RTN_B32, addr=v[10], data0=v[1], vdst=v[2], offset0=0),
      s_waitcnt(lgkmcnt=0),
      ds_load_b32(addr=v[10], vdst=v[3], offset0=0),
      s_waitcnt(lgkmcnt=0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][2], 0xAAAAAAAA)
    self.assertEqual(st.vgpr[0][3], 0xBBBBBBBB)


class TestDSRegisterWidth(unittest.TestCase):
  """Regression tests: DS loads should only write correct number of VGPRs."""

  def test_ds_load_b32_no_overwrite(self):
    """DS_LOAD_B32 should only write 1 VGPR."""
    instructions = [
      v_mov_b32_e32(v[0], 0),
      s_mov_b32(s[0], 0xDEADBEEF),
      v_mov_b32_e32(v[1], s[0]),
      s_mov_b32(s[0], 0x11111111),
      v_mov_b32_e32(v[2], s[0]),  # sentinel
      ds_store_b32(addr=v[0], data0=v[1], offset0=0),
      s_waitcnt(lgkmcnt=0),
      ds_load_b32(addr=v[0], vdst=v[1], offset0=0),
      s_waitcnt(lgkmcnt=0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 0xDEADBEEF)
    self.assertEqual(st.vgpr[0][2], 0x11111111, "v2 should be untouched")


class TestDS2AddrStride64(unittest.TestCase):
  """Tests for DS_*_2ADDR_STRIDE64 (offset * 256 for B32, offset * 512 for B64)."""

  def test_ds_store_load_2addr_stride64_b32(self):
    """DS_STORE_2ADDR_STRIDE64_B32: stores at ADDR + offset*256."""
    instructions = [
      v_mov_b32_e32(v[10], 0),
      s_mov_b32(s[0], 0xAAAAAAAA),
      v_mov_b32_e32(v[0], s[0]),
      s_mov_b32(s[0], 0xBBBBBBBB),
      v_mov_b32_e32(v[1], s[0]),
      DS(DSOp.DS_STORE_2ADDR_STRIDE64_B32, addr=v[10], data0=v[0], data1=v[1], vdst=v[0], offset0=1, offset1=2),
      s_waitcnt(lgkmcnt=0),
      DS(DSOp.DS_LOAD_2ADDR_STRIDE64_B32, addr=v[10], vdst=v[2:3], offset0=1, offset1=2),
      s_waitcnt(lgkmcnt=0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][2], 0xAAAAAAAA, "v2 from addr 256")
    self.assertEqual(st.vgpr[0][3], 0xBBBBBBBB, "v3 from addr 512")

  def test_ds_store_load_2addr_stride64_b64(self):
    """DS_STORE_2ADDR_STRIDE64_B64: stores at ADDR + offset*512."""
    instructions = [
      v_mov_b32_e32(v[10], 0),
      s_mov_b32(s[0], 0xDEADBEEF),
      v_mov_b32_e32(v[0], s[0]),
      s_mov_b32(s[0], 0xCAFEBABE),
      v_mov_b32_e32(v[1], s[0]),
      s_mov_b32(s[0], 0x12345678),
      v_mov_b32_e32(v[2], s[0]),
      s_mov_b32(s[0], 0x9ABCDEF0),
      v_mov_b32_e32(v[3], s[0]),
      DS(DSOp.DS_STORE_2ADDR_STRIDE64_B64, addr=v[10], data0=v[0:1], data1=v[2:3], vdst=v[0], offset0=1, offset1=2),
      s_waitcnt(lgkmcnt=0),
      DS(DSOp.DS_LOAD_2ADDR_STRIDE64_B64, addr=v[10], vdst=v[4:7], offset0=1, offset1=2),
      s_waitcnt(lgkmcnt=0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][4], 0xDEADBEEF)
    self.assertEqual(st.vgpr[0][5], 0xCAFEBABE)
    self.assertEqual(st.vgpr[0][6], 0x12345678)
    self.assertEqual(st.vgpr[0][7], 0x9ABCDEF0)

  def test_ds_storexchg_2addr_rtn_b32(self):
    """DS_STOREXCHG_2ADDR_RTN_B32: exchange at two addresses."""
    instructions = [
      v_mov_b32_e32(v[10], 0),
      s_mov_b32(s[0], 0x11111111),
      v_mov_b32_e32(v[0], s[0]),
      s_mov_b32(s[0], 0x22222222),
      v_mov_b32_e32(v[1], s[0]),
      DS(DSOp.DS_STORE_2ADDR_B32, addr=v[10], data0=v[0], data1=v[1], vdst=v[0], offset0=0, offset1=1),
      s_waitcnt(lgkmcnt=0),
      s_mov_b32(s[0], 0xAAAAAAAA),
      v_mov_b32_e32(v[2], s[0]),
      s_mov_b32(s[0], 0xBBBBBBBB),
      v_mov_b32_e32(v[3], s[0]),
      DS(DSOp.DS_STOREXCHG_2ADDR_RTN_B32, addr=v[10], data0=v[2], data1=v[3], vdst=v[4:5], offset0=0, offset1=1),
      s_waitcnt(lgkmcnt=0),
      DS(DSOp.DS_LOAD_2ADDR_B32, addr=v[10], vdst=v[6:7], offset0=0, offset1=1),
      s_waitcnt(lgkmcnt=0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][4], 0x11111111, "old val 0")
    self.assertEqual(st.vgpr[0][5], 0x22222222, "old val 1")
    self.assertEqual(st.vgpr[0][6], 0xAAAAAAAA, "new val 0")
    self.assertEqual(st.vgpr[0][7], 0xBBBBBBBB, "new val 1")


  def test_ds_storexchg_rtn_b64(self):
    """DS_STOREXCHG_RTN_B64: exchange 64-bit value and return old."""
    instructions = [
      v_mov_b32_e32(v[10], 0),
      s_mov_b32(s[0], 0xDEADBEEF),
      v_mov_b32_e32(v[0], s[0]),   # initial low
      s_mov_b32(s[0], 0xCAFEBABE),
      v_mov_b32_e32(v[1], s[0]),   # initial high
      DS(DSOp.DS_STORE_B64, addr=v[10], data0=v[0:1], vdst=v[0], offset0=0),
      s_waitcnt(lgkmcnt=0),
      s_mov_b32(s[0], 0x12345678),
      v_mov_b32_e32(v[2], s[0]),   # new low
      s_mov_b32(s[0], 0x9ABCDEF0),
      v_mov_b32_e32(v[3], s[0]),   # new high
      DS(DSOp.DS_STOREXCHG_RTN_B64, addr=v[10], data0=v[2:3], vdst=v[4:5], offset0=0),
      s_waitcnt(lgkmcnt=0),
      DS(DSOp.DS_LOAD_B64, addr=v[10], vdst=v[6:7], offset0=0),
      s_waitcnt(lgkmcnt=0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][4], 0xDEADBEEF, "v4 should have old low dword")
    self.assertEqual(st.vgpr[0][5], 0xCAFEBABE, "v5 should have old high dword")
    self.assertEqual(st.vgpr[0][6], 0x12345678, "v6 should have new low dword")
    self.assertEqual(st.vgpr[0][7], 0x9ABCDEF0, "v7 should have new high dword")

  def test_ds_store_load_2addr_stride64_b64_roundtrip(self):
    """DS_STORE_2ADDR_STRIDE64_B64 followed by DS_LOAD_2ADDR_STRIDE64_B64 works correctly."""
    instructions = [
      v_mov_b32_e32(v[10], 0),
      s_mov_b32(s[0], 0x11111111),
      v_mov_b32_e32(v[0], s[0]),
      s_mov_b32(s[0], 0x22222222),
      v_mov_b32_e32(v[1], s[0]),
      DS(DSOp.DS_STORE_2ADDR_STRIDE64_B64, addr=v[10], data0=v[0:1], data1=v[0:1], vdst=v[0], offset0=1, offset1=2),
      s_waitcnt(lgkmcnt=0),
      DS(DSOp.DS_LOAD_2ADDR_STRIDE64_B64, addr=v[10], vdst=v[2:5], offset0=1, offset1=2),
      s_waitcnt(lgkmcnt=0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][2], 0x11111111, "v2 should have val1 low")
    self.assertEqual(st.vgpr[0][3], 0x22222222, "v3 should have val1 high")
    self.assertEqual(st.vgpr[0][4], 0x11111111, "v4 should have val2 low")
    self.assertEqual(st.vgpr[0][5], 0x22222222, "v5 should have val2 high")

  def test_ds_storexchg_2addr_stride64_rtn_b32(self):
    """DS_STOREXCHG_2ADDR_STRIDE64_RTN_B32: exchange at two addresses (offset*256)."""
    instructions = [
      v_mov_b32_e32(v[10], 0),
      s_mov_b32(s[0], 0x11111111),
      v_mov_b32_e32(v[0], s[0]),
      s_mov_b32(s[0], 0x22222222),
      v_mov_b32_e32(v[1], s[0]),
      DS(DSOp.DS_STORE_2ADDR_STRIDE64_B32, addr=v[10], data0=v[0], data1=v[1], vdst=v[0], offset0=1, offset1=2),
      s_waitcnt(lgkmcnt=0),
      s_mov_b32(s[0], 0xAAAAAAAA),
      v_mov_b32_e32(v[2], s[0]),
      s_mov_b32(s[0], 0xBBBBBBBB),
      v_mov_b32_e32(v[3], s[0]),
      DS(DSOp.DS_STOREXCHG_2ADDR_STRIDE64_RTN_B32, addr=v[10], data0=v[2], data1=v[3], vdst=v[4:5], offset0=1, offset1=2),
      s_waitcnt(lgkmcnt=0),
      DS(DSOp.DS_LOAD_2ADDR_STRIDE64_B32, addr=v[10], vdst=v[6:7], offset0=1, offset1=2),
      s_waitcnt(lgkmcnt=0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][4], 0x11111111, "v4 should have old value")
    self.assertEqual(st.vgpr[0][5], 0x22222222, "v5 should have old value")
    self.assertEqual(st.vgpr[0][6], 0xAAAAAAAA, "v6 should have new value")
    self.assertEqual(st.vgpr[0][7], 0xBBBBBBBB, "v7 should have new value")

  def test_ds_storexchg_2addr_stride64_rtn_b64_returns_old(self):
    """DS_STOREXCHG_2ADDR_STRIDE64_RTN_B64: returns old values correctly."""
    instructions = [
      v_mov_b32_e32(v[10], 0),
      s_mov_b32(s[0], 0x11111111),
      v_mov_b32_e32(v[0], s[0]),
      s_mov_b32(s[0], 0x22222222),
      v_mov_b32_e32(v[1], s[0]),
      DS(DSOp.DS_STORE_2ADDR_STRIDE64_B64, addr=v[10], data0=v[0:1], data1=v[0:1], vdst=v[0], offset0=1, offset1=2),
      s_waitcnt(lgkmcnt=0),
      s_mov_b32(s[0], 0xAAAAAAAA),
      v_mov_b32_e32(v[6], s[0]),
      s_mov_b32(s[0], 0xBBBBBBBB),
      v_mov_b32_e32(v[7], s[0]),
      DS(DSOp.DS_STOREXCHG_2ADDR_STRIDE64_RTN_B64, addr=v[10], data0=v[6:7], data1=v[6:7], vdst=v[8:11], offset0=1, offset1=2),
      s_waitcnt(lgkmcnt=0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][8], 0x11111111, "v8 should have old val1 low")
    self.assertEqual(st.vgpr[0][9], 0x22222222, "v9 should have old val1 high")
    self.assertEqual(st.vgpr[0][10], 0x11111111, "v10 should have old val2 low")
    self.assertEqual(st.vgpr[0][11], 0x22222222, "v11 should have old val2 high")


class TestAtomicOrdering(unittest.TestCase):
  """Tests for atomic operation return values and ordering."""

  def test_ds_add_rtn_sequence(self):
    """DS_ADD_RTN returns correct old values in sequence."""
    instructions = [
      v_mov_b32_e32(v[10], 0),
      v_mov_b32_e32(v[0], 100),
      DS(DSOp.DS_STORE_B32, addr=v[10], data0=v[0], vdst=v[0], offset0=0),
      s_waitcnt(lgkmcnt=0),
      v_mov_b32_e32(v[1], 25),
      DS(DSOp.DS_ADD_RTN_U32, addr=v[10], data0=v[1], vdst=v[2], offset0=0),
      s_waitcnt(lgkmcnt=0),
      DS(DSOp.DS_ADD_RTN_U32, addr=v[10], data0=v[1], vdst=v[3], offset0=0),
      s_waitcnt(lgkmcnt=0),
      DS(DSOp.DS_LOAD_B32, addr=v[10], vdst=v[4], offset0=0),
      s_waitcnt(lgkmcnt=0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][2], 100, "First add should return 100")
    self.assertEqual(st.vgpr[0][3], 125, "Second add should return 125")
    self.assertEqual(st.vgpr[0][4], 150, "Final value should be 150")


class TestDsPermute(unittest.TestCase):
  """Tests for DS_PERMUTE_B32 and DS_BPERMUTE_B32 instructions."""

  def test_ds_permute_b32_identity(self):
    """DS_PERMUTE_B32 with identity permutation (lane 0 sends to lane 0)."""
    # For simplicity, test with single lane
    instructions = [
      v_mov_b32_e32(v[0], 0),  # addr = 0 (lane 0)
      v_mov_b32_e32(v[1], 0xDEADBEEF),  # data
      ds_permute_b32(v[2], v[0], v[1]),
      s_waitcnt(lgkmcnt=0),
    ]
    st = run_program(instructions, n_lanes=1)
    # Lane 0 sends to lane 0, so lane 0 gets 0xDEADBEEF
    self.assertEqual(st.vgpr[0][2], 0xDEADBEEF)

  def test_ds_bpermute_b32_identity(self):
    """DS_BPERMUTE_B32 with identity permutation (each lane reads from itself)."""
    instructions = [
      v_mov_b32_e32(v[0], 0),  # addr = 0 (read from lane 0)
      v_mov_b32_e32(v[1], 0xCAFEBABE),  # data in lane 0
      ds_bpermute_b32(v[2], v[0], v[1]),
      s_waitcnt(lgkmcnt=0),
    ]
    st = run_program(instructions, n_lanes=1)
    # Lane 0 reads from lane 0's v[1]
    self.assertEqual(st.vgpr[0][2], 0xCAFEBABE)

  def test_ds_permute_b32_broadcast(self):
    """DS_PERMUTE_B32 broadcast - all lanes send to lane 0."""
    # With 4 lanes, all sending to lane 0, highest lane wins
    instructions = [
      v_mov_b32_e32(v[0], 0),  # All lanes send to addr 0 (lane 0)
      v_mov_b32_e32(v[1], 0x11111111),  # All lanes send same data
      ds_permute_b32(v[2], v[0], v[1]),
      s_waitcnt(lgkmcnt=0),
    ]
    st = run_program(instructions, n_lanes=4)
    # Lane 0 receives data (highest numbered active lane wins)
    self.assertEqual(st.vgpr[0][2], 0x11111111)


if __name__ == '__main__':
  unittest.main()
