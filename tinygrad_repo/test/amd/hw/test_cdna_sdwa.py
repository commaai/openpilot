#!/usr/bin/env python3
import unittest
import tinygrad.runtime.autogen.amd.cdna.ins as cdna
from test.amd.hw.test_cdna_vop3 import run_cdna

class TestCDNASDWA(unittest.TestCase):
  def test_v_add_co_u32_e32_writes_vcc(self):
    out = run_cdna([
      cdna.s_mov_b32(cdna.s[0], 0xffffffff),
      cdna.v_mov_b32_e32(cdna.v[0], cdna.s[0]),
      cdna.v_mov_b32_e32(cdna.v[13], 1),
      cdna.v_add_co_u32_e32(cdna.v[0], cdna.SDWA, cdna.v[13], vsrc0=cdna.v[0], dst_sel=6, src0_sel=6),
      cdna.v_mov_b32_e32(cdna.v[2], cdna.VCC_LO),
      cdna.v_lshlrev_b32_e32(cdna.v[2], 31, cdna.v[2]),
      cdna.v_or_b32_e32(cdna.v[2], cdna.v[2], cdna.v[0]),
    ])
    self.assertEqual(out, 0x80000000)

if __name__ == "__main__":
  unittest.main()
