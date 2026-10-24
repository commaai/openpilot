import unittest
import numpy as np
from tinygrad import Tensor, Device, dtypes, UOp
from tinygrad.renderer.wgsl import WGSLRenderer
from tinygrad.uop.ops import Ops, KernelInfo
from test.runtime.test_renderer_failures import _test_uop_result, _setup_and_test_alu

@unittest.skipUnless(isinstance(Device[Device.DEFAULT].renderer, WGSLRenderer), "tests for wgsl renderer")
class TestWGSLFailures(unittest.TestCase):
  def test_multiply_infinity(self):
    # multiplying a positive constant by infinity should return infinity
    # WGSL pipelines do not handle this reliably, some of which return zero, unless infinity always comes from a read on a dynamic buffer
    ret = _setup_and_test_alu(Ops.MUL, 5.0, UOp.const(float("inf")).cast(dtypes.float32))
    self.assertEqual(ret[0], float("inf"))

  # WGSL has a specific select(alt, val, gate) ternary operator instead of gate?val:alt
  def test_gated_load(self):
    a = UOp.param(0, dtypes.int, 4)
    b = UOp.param(1, dtypes.int, 4)
    c = UOp.param(2, dtypes.int, 4)
    lidx0 = UOp.special(4, "lidx0")
    gate = lidx0.ne(0)
    alt = c.index(lidx0).load()
    ld = UOp.load(b.index(lidx0.valid(gate)))
    alt_load = gate.where(ld, alt)
    store = UOp.store(a.index(lidx0), alt_load)
    sink = UOp(Ops.SINK, src=(store,), arg=KernelInfo())
    ret = _test_uop_result([Tensor([0,1,2,3], dtype=dtypes.int), Tensor([4,5,6,7], dtype=dtypes.int)], sink, local_size=[4])[0]
    np.testing.assert_equal(ret, [4,1,2,3])

if __name__ == "__main__": unittest.main()
