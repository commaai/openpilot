import unittest
import tinygrad.runtime.autogen.amd.cdna.ins as cdna
from test.amd.hw.test_cdna_vop3 import run_cdna_lanes

class TestCDNADSSwizzle(unittest.TestCase):
  def test_modes(self):
    # the source lane of lanes 0-7 for each mode of the ISA pseudocode
    for offset, src in {0x041f: [1, 0, 3, 2, 5, 4, 7, 6],                  # bitmask: xor 1
                        0x00a0: [5]*8,                                     # bitmask: or 5, broadcast
                        0x801b: [3, 2, 1, 0, 7, 6, 5, 4],                  # quad permute: reversed
                        0xc020: [1, 2, 3, 4, 5, 6, 7, 8],                  # rotate by 1
                        0xe000: [0, 16, 8, 24, 4, 20, 12, 28]}.items():    # fft: bit reversed
      with self.subTest(offset=hex(offset)):
        out = run_cdna_lanes([cdna.ds_swizzle_b32(vdst=cdna.v[2], addr=cdna.v[255], offset0=offset & 255, offset1=offset >> 8), cdna.s_waitcnt(0)])
        self.assertEqual(out[:8], src)

if __name__ == "__main__":
  unittest.main()
