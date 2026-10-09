import itertools, unittest
import tinygrad.runtime.autogen.amd.rdna4.ins as r4
from tinygrad.dtype import dtypes, fp8_to_float
from test.amd.hw.helpers import f2i, run_rdna4

class TestFP8RDNA4(unittest.TestCase):
  def test_cvt_f32_fp8(self):
    packed = 0x813F7807
    for name, dt in [('fp8', dtypes.fp8e4m3), ('bf8', dtypes.fp8e5m2)]:
      for encoding, opsel in [('e32', 0), *[('e64', i) for i in range(4)]]:
        with self.subTest(dtype=dt, encoding=encoding, opsel=opsel):
          out = run_rdna4([
            r4.v_mov_b32_e32(r4.v[3], packed),
            getattr(r4, f'v_cvt_f32_{name}_{encoding}')(r4.v[2], r4.v[3], **({'opsel': opsel} if encoding == 'e64' else {})),
          ])
          byte = ((opsel & 1) << 1) | (opsel >> 1)
          self.assertEqual(out, [f2i(fp8_to_float((packed >> (8 * byte)) & 0xFF, dt))] * 32)

  def test_wmma_fp8(self):
    formats = [('fp8', dtypes.fp8e4m3), ('bf8', dtypes.fp8e5m2)]
    for (a_fmt, a_dt), (b_fmt, b_dt) in itertools.product(formats, repeat=2):
      with self.subTest(a=a_fmt, b=b_fmt):
        out = run_rdna4([
          *[r4.v_mov_b32_e32(r4.v[i], 0xB8B8B8B8) for i in (0, 1)],
          *[r4.v_mov_b32_e32(r4.v[i], 0x3C3C3C3C) for i in (4, 5)],
          getattr(r4, f'v_wmma_f32_16x16x16_{a_fmt}_{b_fmt}')(r4.v[8:15], r4.v[0:1], r4.v[4:5], 0),
        ], out_reg=8)
        self.assertEqual(out, [f2i(16 * fp8_to_float(0xB8, a_dt) * fp8_to_float(0x3C, b_dt))] * 32)

if __name__ == '__main__': unittest.main()
