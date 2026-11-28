"""RDNA4 GLOBAL_LOAD_TR_B128 / GLOBAL_LOAD_TR_B64 coverage.

Exercises transposed global loads end-to-end in the emulator (saddr, 64-bit vaddr, signed ioffset,
vdst/vaddr aliasing, and whole-wave EXEC-ignoring behavior) and compares against real RDNA4 hardware when USE_HW=1.
"""
import unittest
import tinygrad.runtime.autogen.amd.rdna4.ins as r4
from tinygrad.renderer.amd.dsl import EXEC_LO, NULL
from test.amd.hw.helpers import run_rdna4

def _fill_u16_seq() -> list:
  # Populate 128-byte output buffer (s[4:5]) with uint16[i] = i (i = 0..63).
  return [
    r4.s_load_b64(r4.s[4:5], r4.s[80:81], soffset=NULL),
    r4.s_wait_kmcnt(simm16=0),
    r4.v_lshlrev_b32_e32(r4.v[10], 1, r4.v[255]),
    r4.v_add_nc_u32_e32(r4.v[11], 1, r4.v[10]),
    r4.v_lshlrev_b32_e32(r4.v[11], 16, r4.v[11]),
    r4.v_or_b32_e32(r4.v[10], r4.v[10], r4.v[11]),
    r4.v_lshlrev_b32_e32(r4.v[12], 2, r4.v[255]),
    r4.v_mov_b32_e32(r4.v[13], 0),
    r4.global_store_b32(vaddr=r4.v[12:13], saddr=r4.s[4:5], vsrc=r4.v[10]),
    r4.s_wait_storecnt(simm16=0),
  ]

def _fill_u8_seq() -> list:
  # Populate 128-byte output buffer (s[4:5]) with uint8[i] = i (i = 0..127).
  return [
    r4.s_load_b64(r4.s[4:5], r4.s[80:81], soffset=NULL),
    r4.s_wait_kmcnt(simm16=0),
    r4.v_lshlrev_b32_e32(r4.v[10], 2, r4.v[255]),
    r4.v_mov_b32_e32(r4.v[11], 0x03020100),
    r4.v_mul_lo_u32(r4.v[13], r4.v[10], 0x01010101),
    r4.v_add_nc_u32_e32(r4.v[10], r4.v[13], r4.v[11]),
    r4.v_lshlrev_b32_e32(r4.v[12], 2, r4.v[255]),
    r4.v_mov_b32_e32(r4.v[13], 0),
    r4.global_store_b32(vaddr=r4.v[12:13], saddr=r4.s[4:5], vsrc=r4.v[10]),
    r4.s_wait_storecnt(simm16=0),
  ]

class TestGlobalLoadTrRDNA4(unittest.TestCase):
  def test_global_load_tr_b128(self):
    # Each 8-lane group reads the 8x8 u16 tile; check all 4 dwords, signed ioffset, and partial EXEC (whole-wave mode).
    for d in range(4):
      with self.subTest(dword=d):
        out = run_rdna4([
          *_fill_u16_seq(),
          r4.v_and_b32_e32(r4.v[0], 7, r4.v[255]),
          r4.v_lshlrev_b32_e32(r4.v[0], 4, r4.v[0]),
          r4.v_add_nc_u32_e32(r4.v[0], 16, r4.v[0]),
          r4.v_mov_b32_e32(r4.v[1], 0),
          *[r4.v_mov_b32_e32(r4.v[4 + k], 0xdeadbeef) for k in range(4)],
          r4.s_mov_b32(EXEC_LO, 0xffffffdb),
          r4.global_load_tr_b128(vdst=r4.v[4:7], vaddr=r4.v[0:1], saddr=r4.s[4:5], ioffset=-16),
          r4.s_wait_loadcnt(simm16=0),
          r4.s_mov_b32(EXEC_LO, 0xffffffff),
        ], out_reg=4 + d)
        expected = [(((2 * d + 1) * 8 + (i & 7)) << 16) | ((2 * d) * 8 + (i & 7)) for i in range(32)]
        self.assertEqual(out, expected)

  def test_global_load_tr_b128_vaddr64_alias(self):
    # 64-bit vaddr (saddr=NULL) with vdst aliasing vaddr (v[0:3] overwrites v[0:1]).
    out = run_rdna4([
      *_fill_u16_seq(),
      r4.v_and_b32_e32(r4.v[2], 7, r4.v[255]),
      r4.v_lshlrev_b32_e32(r4.v[2], 4, r4.v[2]),
      r4.v_mov_b32_e32(r4.v[3], 0),
      r4.v_add_co_u32(r4.v[0], r4.VCC_LO, r4.s[4], r4.v[2]),
      r4.v_add_co_ci_u32_e32(r4.v[1], r4.s[5], r4.v[3]),
      r4.global_load_tr_b128(vdst=r4.v[0:3], vaddr=r4.v[0:1], saddr=NULL),
      r4.s_wait_loadcnt(simm16=0),
    ], out_reg=3)
    expected = [((7 * 8 + (i & 7)) << 16) | (6 * 8 + (i & 7)) for i in range(32)]
    self.assertEqual(out, expected)

  def test_global_load_tr_b64_vaddr64_alias(self):
    # 64-bit vaddr (saddr=NULL) with vdst aliasing vaddr (v[0:1] overwrites v[0:1]).
    out = run_rdna4([
      *_fill_u8_seq(),
      r4.v_and_b32_e32(r4.v[2], 15, r4.v[255]),
      r4.v_lshlrev_b32_e32(r4.v[2], 3, r4.v[2]),
      r4.v_mov_b32_e32(r4.v[3], 0),
      r4.v_add_co_u32(r4.v[0], r4.VCC_LO, r4.s[4], r4.v[2]),
      r4.v_add_co_ci_u32_e32(r4.v[1], r4.s[5], r4.v[3]),
      r4.global_load_tr_b64(vdst=r4.v[0:1], vaddr=r4.v[0:1], saddr=NULL),
      r4.s_wait_loadcnt(simm16=0),
    ], out_reg=1)
    expected = [sum(((8 + ((i >> 3) & 1) * 4 + k) * 8 + (i & 7)) << (k * 8) for k in range(4)) for i in range(32)]
    self.assertEqual(out, expected)

  def test_global_load_tr_b64(self):
    # Each 16-lane group reads the 16x8 u8 tile; check both dwords, signed ioffset, and partial EXEC (whole-wave mode).
    for d in range(2):
      with self.subTest(dword=d):
        out = run_rdna4([
          *_fill_u8_seq(),
          r4.v_and_b32_e32(r4.v[0], 15, r4.v[255]),
          r4.v_lshlrev_b32_e32(r4.v[0], 3, r4.v[0]),
          r4.v_add_nc_u32_e32(r4.v[0], 8, r4.v[0]),
          r4.v_mov_b32_e32(r4.v[1], 0),
          *[r4.v_mov_b32_e32(r4.v[4 + k], 0xdeadbeef) for k in range(2)],
          r4.s_mov_b32(EXEC_LO, 0xfffffdfb),
          r4.global_load_tr_b64(vdst=r4.v[4:5], vaddr=r4.v[0:1], saddr=r4.s[4:5], ioffset=-8),
          r4.s_wait_loadcnt(simm16=0),
          r4.s_mov_b32(EXEC_LO, 0xffffffff),
        ], out_reg=4 + d)
        expected = []
        for i in range(32):
          base, col = d * 8 + ((i >> 3) & 1) * 4, i & 7
          expected.append(sum(((base + k) * 8 + col) << (k * 8) for k in range(4)))
        self.assertEqual(out, expected)

if __name__ == '__main__': unittest.main()
