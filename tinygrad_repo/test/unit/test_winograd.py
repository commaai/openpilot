import unittest, sys
import numpy as np
from tinygrad import Tensor, GlobalCounters, Context, nn
from tinygrad.helpers import WINO

@unittest.skipIf(sys.platform.startswith("win"), "flaky on Windows")
class TestWinogradClose(unittest.TestCase):
  def test_close(self):
    inp = Tensor.rand(1, 16, 16, 16)
    conv = nn.Conv2d(16, 16, 3)
    conv(inp).realize() # warmup
    GlobalCounters.reset()
    print("non winograd")
    with Context(WINO=0):
      cmp = conv(inp).realize() # warmup
    GlobalCounters.reset()
    print("winograd")
    with Context(WINO=1):
      test = conv(inp).realize()
    np.testing.assert_allclose(cmp.numpy(), test.numpy(), atol=1e-5)

@unittest.skipIf(sys.platform.startswith("win"), "flaky on Windows")
class TestWinograd(unittest.TestCase):
  def setUp(self):
    self.old = WINO.value
    WINO.value = 1
  def tearDown(self):
    WINO.value = self.old

  def test_padded_conv2d(self):
    # tests padding order in winograd
    x,w = Tensor.rand(1,3,11,28).realize(), Tensor.rand(4,3,3,3).realize()
    with Context(WINO=0): expected = Tensor.conv2d(x,w,padding=1).realize()
    with Context(WINO=1): result = Tensor.conv2d(x,w,padding=1).realize()
    np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=1e-4)

if __name__ == '__main__':
  unittest.main(verbosity=2)
