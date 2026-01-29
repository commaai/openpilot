"""Tests for FLAT instructions - flat memory operations.

Includes: flat_load_*, flat_store_*, flat_atomic_*
"""
import unittest
from extra.assembly.amd.test.hw.helpers import *

class TestFlatAtomic(unittest.TestCase):
  """Tests for FLAT atomic instructions."""

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

  def test_flat_atomic_add_u32(self):
    """FLAT_ATOMIC_ADD_U32 adds to memory and returns old value."""
    TEST_OFFSET = 2000
    setup = [
      s_mov_b32(s[0], 100),
      v_mov_b32_e32(v[2], s[0]),
      global_store_b32(addr=v[0:1], data=v[2], saddr=SrcEnum.NULL, offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      s_mov_b32(s[0], 50),
      v_mov_b32_e32(v[3], s[0]),
    ]
    atomic = FLAT(FLATOp.FLAT_ATOMIC_ADD_U32, addr=v[0:1], data=v[3], vdst=v[4], saddr=SrcEnum.NULL, offset=TEST_OFFSET, glc=1)
    def check(st):
      self.assertEqual(st.vgpr[0][4], 100)
    self._make_test(setup, atomic, check, TEST_OFFSET)

  def test_flat_atomic_swap_b32(self):
    """FLAT_ATOMIC_SWAP_B32 swaps memory value and returns old value."""
    TEST_OFFSET = 2000
    setup = [
      s_mov_b32(s[0], 0xAAAAAAAA),
      v_mov_b32_e32(v[2], s[0]),
      global_store_b32(addr=v[0:1], data=v[2], saddr=SrcEnum.NULL, offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      s_mov_b32(s[0], 0xBBBBBBBB),
      v_mov_b32_e32(v[3], s[0]),
    ]
    atomic = FLAT(FLATOp.FLAT_ATOMIC_SWAP_B32, addr=v[0:1], data=v[3], vdst=v[4], saddr=SrcEnum.NULL, offset=TEST_OFFSET, glc=1)
    def check(st):
      self.assertEqual(st.vgpr[0][4], 0xAAAAAAAA)
    self._make_test(setup, atomic, check, TEST_OFFSET)

  def test_flat_atomic_and_b32(self):
    """FLAT_ATOMIC_AND_B32 ANDs with memory and returns old value."""
    TEST_OFFSET = 2000
    setup = [
      s_mov_b32(s[0], 0xFF00FF00),
      v_mov_b32_e32(v[2], s[0]),
      global_store_b32(addr=v[0:1], data=v[2], saddr=SrcEnum.NULL, offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      s_mov_b32(s[0], 0xFFFF0000),
      v_mov_b32_e32(v[3], s[0]),
    ]
    atomic = FLAT(FLATOp.FLAT_ATOMIC_AND_B32, addr=v[0:1], data=v[3], vdst=v[4], saddr=SrcEnum.NULL, offset=TEST_OFFSET, glc=1)
    def check(st):
      self.assertEqual(st.vgpr[0][4], 0xFF00FF00)
    self._make_test(setup, atomic, check, TEST_OFFSET)

  def test_flat_atomic_or_b32(self):
    """FLAT_ATOMIC_OR_B32 ORs with memory and returns old value."""
    TEST_OFFSET = 2000
    setup = [
      s_mov_b32(s[0], 0x00FF0000),
      v_mov_b32_e32(v[2], s[0]),
      global_store_b32(addr=v[0:1], data=v[2], saddr=SrcEnum.NULL, offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      s_mov_b32(s[0], 0x0000FF00),
      v_mov_b32_e32(v[3], s[0]),
    ]
    atomic = FLAT(FLATOp.FLAT_ATOMIC_OR_B32, addr=v[0:1], data=v[3], vdst=v[4], saddr=SrcEnum.NULL, offset=TEST_OFFSET, glc=1)
    def check(st):
      self.assertEqual(st.vgpr[0][4], 0x00FF0000)
    self._make_test(setup, atomic, check, TEST_OFFSET)

  def test_flat_atomic_inc_u32(self):
    """FLAT_ATOMIC_INC_U32 increments and returns old value."""
    TEST_OFFSET = 2000
    setup = [
      s_mov_b32(s[0], 10),
      v_mov_b32_e32(v[2], s[0]),
      global_store_b32(addr=v[0:1], data=v[2], saddr=SrcEnum.NULL, offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      s_mov_b32(s[0], 100),  # threshold
      v_mov_b32_e32(v[3], s[0]),
    ]
    atomic = FLAT(FLATOp.FLAT_ATOMIC_INC_U32, addr=v[0:1], data=v[3], vdst=v[4], saddr=SrcEnum.NULL, offset=TEST_OFFSET, glc=1)
    def check(st):
      self.assertEqual(st.vgpr[0][4], 10)
    self._make_test(setup, atomic, check, TEST_OFFSET)

  def test_flat_atomic_dec_u32(self):
    """FLAT_ATOMIC_DEC_U32 decrements and returns old value."""
    TEST_OFFSET = 2000
    setup = [
      s_mov_b32(s[0], 10),
      v_mov_b32_e32(v[2], s[0]),
      global_store_b32(addr=v[0:1], data=v[2], saddr=SrcEnum.NULL, offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      s_mov_b32(s[0], 100),
      v_mov_b32_e32(v[3], s[0]),
    ]
    atomic = FLAT(FLATOp.FLAT_ATOMIC_DEC_U32, addr=v[0:1], data=v[3], vdst=v[4], saddr=SrcEnum.NULL, offset=TEST_OFFSET, glc=1)
    def check(st):
      self.assertEqual(st.vgpr[0][4], 10)
    self._make_test(setup, atomic, check, TEST_OFFSET)

  def test_flat_atomic_sub_u32(self):
    """FLAT_ATOMIC_SUB_U32 subtracts from memory and returns old value."""
    TEST_OFFSET = 2000
    setup = [
      s_mov_b32(s[0], 100),
      v_mov_b32_e32(v[2], s[0]),
      global_store_b32(addr=v[0:1], data=v[2], saddr=SrcEnum.NULL, offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      s_mov_b32(s[0], 30),
      v_mov_b32_e32(v[3], s[0]),  # sub 30
    ]
    atomic = FLAT(FLATOp.FLAT_ATOMIC_SUB_U32, addr=v[0:1], data=v[3], vdst=v[4], saddr=SrcEnum.NULL, offset=TEST_OFFSET, glc=1)
    def check(st):
      self.assertEqual(st.vgpr[0][4], 100, "v4 should have old value (100)")
    self._make_test(setup, atomic, check, TEST_OFFSET)

  def test_flat_atomic_xor_b32(self):
    """FLAT_ATOMIC_XOR_B32 XORs with memory and returns old value."""
    TEST_OFFSET = 2000
    setup = [
      s_mov_b32(s[0], 0xAAAAAAAA),
      v_mov_b32_e32(v[2], s[0]),
      global_store_b32(addr=v[0:1], data=v[2], saddr=SrcEnum.NULL, offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      s_mov_b32(s[0], 0xFFFFFFFF),
      v_mov_b32_e32(v[3], s[0]),  # XOR mask
    ]
    atomic = FLAT(FLATOp.FLAT_ATOMIC_XOR_B32, addr=v[0:1], data=v[3], vdst=v[4], saddr=SrcEnum.NULL, offset=TEST_OFFSET, glc=1)
    def check(st):
      self.assertEqual(st.vgpr[0][4], 0xAAAAAAAA, "v4 should have old value")
    self._make_test(setup, atomic, check, TEST_OFFSET)

  def test_flat_atomic_min_u32(self):
    """FLAT_ATOMIC_MIN_U32 stores min and returns old value."""
    TEST_OFFSET = 2000
    setup = [
      s_mov_b32(s[0], 100),
      v_mov_b32_e32(v[2], s[0]),
      global_store_b32(addr=v[0:1], data=v[2], saddr=SrcEnum.NULL, offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      s_mov_b32(s[0], 50),
      v_mov_b32_e32(v[3], s[0]),  # compare value (smaller)
    ]
    atomic = FLAT(FLATOp.FLAT_ATOMIC_MIN_U32, addr=v[0:1], data=v[3], vdst=v[4], saddr=SrcEnum.NULL, offset=TEST_OFFSET, glc=1)
    def check(st):
      self.assertEqual(st.vgpr[0][4], 100, "v4 should have old value (100)")
    self._make_test(setup, atomic, check, TEST_OFFSET)

  def test_flat_atomic_max_u32(self):
    """FLAT_ATOMIC_MAX_U32 stores max and returns old value."""
    TEST_OFFSET = 2000
    setup = [
      s_mov_b32(s[0], 50),
      v_mov_b32_e32(v[2], s[0]),
      global_store_b32(addr=v[0:1], data=v[2], saddr=SrcEnum.NULL, offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      s_mov_b32(s[0], 100),
      v_mov_b32_e32(v[3], s[0]),  # compare value (larger)
    ]
    atomic = FLAT(FLATOp.FLAT_ATOMIC_MAX_U32, addr=v[0:1], data=v[3], vdst=v[4], saddr=SrcEnum.NULL, offset=TEST_OFFSET, glc=1)
    def check(st):
      self.assertEqual(st.vgpr[0][4], 50, "v4 should have old value (50)")
    self._make_test(setup, atomic, check, TEST_OFFSET)

  def test_flat_atomic_inc_u64_returns_old_value(self):
    """FLAT_ATOMIC_INC_U64 should return full 64-bit old value."""
    TEST_OFFSET = 2000
    setup = [
      # Store initial 64-bit value: 0xCAFEBABE_DEADBEEF
      s_mov_b32(s[0], 0xDEADBEEF),
      v_mov_b32_e32(v[2], s[0]),
      s_mov_b32(s[0], 0xCAFEBABE),
      v_mov_b32_e32(v[3], s[0]),
      global_store_b64(addr=v[0:1], data=v[2:3], saddr=SrcEnum.NULL, offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      # Threshold: 0xFFFFFFFF_FFFFFFFF
      s_mov_b32(s[0], 0xFFFFFFFF),
      v_mov_b32_e32(v[4], s[0]),
      v_mov_b32_e32(v[5], s[0]),
    ]
    atomic = FLAT(FLATOp.FLAT_ATOMIC_INC_U64, addr=v[0:1], data=v[4:5], vdst=v[6:7], saddr=SrcEnum.NULL, offset=TEST_OFFSET, glc=1)
    def check(st):
      self.assertEqual(st.vgpr[0][6], 0xDEADBEEF, "v6 should have old value low dword")
      self.assertEqual(st.vgpr[0][7], 0xCAFEBABE, "v7 should have old value high dword")
    self._make_test(setup, atomic, check, TEST_OFFSET)

  def test_flat_atomic_add_u64(self):
    """FLAT_ATOMIC_ADD_U64 adds 64-bit value and returns old value."""
    TEST_OFFSET = 2000
    setup = [
      s_mov_b32(s[0], 0x11111111),
      v_mov_b32_e32(v[2], s[0]),
      s_mov_b32(s[0], 0x22222222),
      v_mov_b32_e32(v[3], s[0]),
      global_store_b64(addr=v[0:1], data=v[2:3], saddr=SrcEnum.NULL, offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      s_mov_b32(s[0], 0x00000001),  # add 1
      v_mov_b32_e32(v[4], s[0]),
      s_mov_b32(s[0], 0x00000000),
      v_mov_b32_e32(v[5], s[0]),
    ]
    atomic = FLAT(FLATOp.FLAT_ATOMIC_ADD_U64, addr=v[0:1], data=v[4:5], vdst=v[6:7], saddr=SrcEnum.NULL, offset=TEST_OFFSET, glc=1)
    def check(st):
      self.assertEqual(st.vgpr[0][6], 0x11111111, "v6 should have old value low")
      self.assertEqual(st.vgpr[0][7], 0x22222222, "v7 should have old value high")
    self._make_test(setup, atomic, check, TEST_OFFSET)

  def test_flat_atomic_swap_b64(self):
    """FLAT_ATOMIC_SWAP_B64 swaps 64-bit value and returns old value."""
    TEST_OFFSET = 2000
    setup = [
      s_mov_b32(s[0], 0xAAAAAAAA),
      v_mov_b32_e32(v[2], s[0]),
      s_mov_b32(s[0], 0xBBBBBBBB),
      v_mov_b32_e32(v[3], s[0]),
      global_store_b64(addr=v[0:1], data=v[2:3], saddr=SrcEnum.NULL, offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      s_mov_b32(s[0], 0xCCCCCCCC),
      v_mov_b32_e32(v[4], s[0]),
      s_mov_b32(s[0], 0xDDDDDDDD),
      v_mov_b32_e32(v[5], s[0]),
    ]
    atomic = FLAT(FLATOp.FLAT_ATOMIC_SWAP_B64, addr=v[0:1], data=v[4:5], vdst=v[6:7], saddr=SrcEnum.NULL, offset=TEST_OFFSET, glc=1)
    def check(st):
      self.assertEqual(st.vgpr[0][6], 0xAAAAAAAA, "v6 should have old value low")
      self.assertEqual(st.vgpr[0][7], 0xBBBBBBBB, "v7 should have old value high")
    self._make_test(setup, atomic, check, TEST_OFFSET)


class TestFlatLoad(unittest.TestCase):
  """Tests for FLAT load instructions."""

  def test_flat_load_b32(self):
    """FLAT_LOAD_B32 loads 32-bit value correctly."""
    TEST_OFFSET = 2000
    instructions = [
      s_load_b64(s[2:3], s[80:81], 0, soffset=SrcEnum.NULL),
      s_waitcnt(lgkmcnt=0),
      v_mov_b32_e32(v[0], s[2]),
      v_mov_b32_e32(v[1], s[3]),
      s_mov_b32(s[0], 0xDEADBEEF),
      v_mov_b32_e32(v[2], s[0]),
      global_store_b32(addr=v[0:1], data=v[2], saddr=SrcEnum.NULL, offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      FLAT(FLATOp.FLAT_LOAD_B32, addr=v[0:1], vdst=v[4], saddr=SrcEnum.NULL, offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      v_mov_b32_e32(v[0], 0),
      v_mov_b32_e32(v[1], 0),
      s_mov_b32(s[2], 0),
      s_mov_b32(s[3], 0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][4], 0xDEADBEEF)

  def test_flat_load_b64(self):
    """FLAT_LOAD_B64 loads 64-bit value correctly."""
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
      global_store_b64(addr=v[0:1], data=v[2:3], saddr=SrcEnum.NULL, offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      FLAT(FLATOp.FLAT_LOAD_B64, addr=v[0:1], vdst=v[4:5], saddr=SrcEnum.NULL, offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      v_mov_b32_e32(v[0], 0),
      v_mov_b32_e32(v[1], 0),
      s_mov_b32(s[2], 0),
      s_mov_b32(s[3], 0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][4], 0xDEADBEEF)
    self.assertEqual(st.vgpr[0][5], 0xCAFEBABE)

  def test_flat_load_b96(self):
    """FLAT_LOAD_B96 loads 96-bit (3 dword) value correctly."""
    TEST_OFFSET = 2000
    instructions = [
      s_load_b64(s[2:3], s[80:81], 0, soffset=SrcEnum.NULL),
      s_waitcnt(lgkmcnt=0),
      v_mov_b32_e32(v[0], s[2]),
      v_mov_b32_e32(v[1], s[3]),
      s_mov_b32(s[0], 0x11111111),
      v_mov_b32_e32(v[2], s[0]),
      s_mov_b32(s[0], 0x22222222),
      v_mov_b32_e32(v[3], s[0]),
      s_mov_b32(s[0], 0x33333333),
      v_mov_b32_e32(v[4], s[0]),
      global_store_b96(addr=v[0:1], data=v[2:4], saddr=SrcEnum.NULL, offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      FLAT(FLATOp.FLAT_LOAD_B96, addr=v[0:1], vdst=v[5:7], saddr=SrcEnum.NULL, offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      v_mov_b32_e32(v[0], 0),
      v_mov_b32_e32(v[1], 0),
      s_mov_b32(s[2], 0),
      s_mov_b32(s[3], 0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][5], 0x11111111)
    self.assertEqual(st.vgpr[0][6], 0x22222222)
    self.assertEqual(st.vgpr[0][7], 0x33333333)

  def test_flat_load_b128(self):
    """FLAT_LOAD_B128 loads 128-bit value correctly."""
    TEST_OFFSET = 2000
    instructions = [
      s_load_b64(s[2:3], s[80:81], 0, soffset=SrcEnum.NULL),
      s_waitcnt(lgkmcnt=0),
      v_mov_b32_e32(v[0], s[2]),
      v_mov_b32_e32(v[1], s[3]),
      s_mov_b32(s[0], 0x11111111),
      v_mov_b32_e32(v[2], s[0]),
      s_mov_b32(s[0], 0x22222222),
      v_mov_b32_e32(v[3], s[0]),
      s_mov_b32(s[0], 0x33333333),
      v_mov_b32_e32(v[4], s[0]),
      s_mov_b32(s[0], 0x44444444),
      v_mov_b32_e32(v[5], s[0]),
      global_store_b128(addr=v[0:1], data=v[2:5], saddr=SrcEnum.NULL, offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      FLAT(FLATOp.FLAT_LOAD_B128, addr=v[0:1], vdst=v[6:9], saddr=SrcEnum.NULL, offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      v_mov_b32_e32(v[0], 0),
      v_mov_b32_e32(v[1], 0),
      s_mov_b32(s[2], 0),
      s_mov_b32(s[3], 0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][6], 0x11111111)
    self.assertEqual(st.vgpr[0][7], 0x22222222)
    self.assertEqual(st.vgpr[0][8], 0x33333333)
    self.assertEqual(st.vgpr[0][9], 0x44444444)


if __name__ == '__main__':
  unittest.main()
