import unittest, sys
from tinygrad import Tensor, GlobalCounters, dtypes, Context
from tinygrad.helpers import WINO

@unittest.skipIf(sys.platform.startswith("win"), "flaky on Windows")
class TestWinograd(unittest.TestCase):
  def setUp(self):
    self.old = WINO.value
    WINO.value = 1
  def tearDown(self):
    WINO.value = self.old

  def test_forward_kernels(self):
    x,w = Tensor.rand(1,4,9,9).realize(), Tensor.rand(4,4,3,3).realize()
    out = Tensor.conv2d(x,w)
    self.assertEqual(len(out.schedule_linear().src), 4)

  def test_backward_kernels(self):
    x,w = Tensor.empty(1,4,9,9).realize(), Tensor.empty(4,4,3,3).realize()
    out = Tensor.conv2d(x,w, padding=1)
    out.mean().backward()
    backward_schedule = x.grad.schedule_linear(w.grad)
    self.assertEqual(len(backward_schedule.src), 4)

  def test_counters(self):
    IC, OC, H = 64, 64, 28
    x,w = Tensor.empty(1,IC,H,H,device="NULL").realize(), Tensor.empty(OC,IC,3,3,device="NULL").realize()
    GlobalCounters.reset()
    with Context(NOOPT=0, WINO=1): Tensor.conv2d(x,w).realize()
    ops_wino = GlobalCounters.global_ops
    GlobalCounters.reset()
    with Context(NOOPT=0, WINO=0): Tensor.conv2d(x,w).realize()
    ops_normal = GlobalCounters.global_ops
    print(f"ops: normal {ops_normal} wino {ops_wino} ratio {ops_wino/ops_normal:.2f}")
    self.assertLess(ops_wino/ops_normal, 0.6)

  def test_dtype(self):
    IC, OC, X, Y = 4,4,9,9
    x,w = Tensor.empty(1,IC,Y,X), Tensor.empty(OC,IC,3,3)
    self.assertEqual(Tensor.conv2d(x,w).dtype, dtypes.default_float)

    x,w = Tensor.empty(1,IC,Y,X,dtype=dtypes.half), Tensor.empty(OC,IC,3,3,dtype=dtypes.half)
    self.assertEqual(Tensor.conv2d(x,w).dtype, dtypes.half)

if __name__ == '__main__':
  unittest.main(verbosity=2)
