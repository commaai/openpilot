import unittest, sys
import numpy as np
from tinygrad import Tensor, Context

@unittest.skipIf(sys.platform.startswith("win"), "flaky on Windows")
class TestWinograd(unittest.TestCase):
  def test_padded_conv2d(self):
    # tests padding order in winograd
    Tensor.manual_seed(0)
    x,w = Tensor.rand(1,3,11,28).realize(), Tensor.rand(4,3,3,3).realize()
    with Context(WINO=0): expected = Tensor.conv2d(x,w,padding=1).realize()
    with Context(WINO=1): result = Tensor.conv2d(x,w,padding=1).realize()
    np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=1e-4)

if __name__ == '__main__':
  unittest.main(verbosity=2)
