import unittest
from tinygrad import Tensor, dtypes, Device
from tinygrad.helpers import Context
from tinygrad.engine.realize import estimate_uop, compile_linear
from tinygrad.renderer.ptx import PTXRenderer

class TestArange(unittest.TestCase):
  def test_cat_complexity(self):
    x = Tensor.arange(2**10) + Tensor.empty((), dtype=dtypes.uint32)
    out = x.cat(x).cat(Tensor.empty(1, dtype=dtypes.uint32))
    linear = compile_linear(out.schedule_linear())
    self.assertLessEqual(estimate_uop(linear.src[-1]).ops, out.numel()*20)

  @unittest.skipIf(isinstance(Device[Device.DEFAULT].renderer, PTXRenderer), "PTX indexing is weird")
  def test_tri_complexity(self):
    with Context(NOOPT=1):
      t = Tensor.ones(256, 256).contiguous().realize()
      linear = compile_linear(t.triu().schedule_linear())
      self.assertLessEqual(estimate_uop(linear.src[-1]).ops, 4 * 256 * 256)

if __name__ == '__main__':
  unittest.main()
