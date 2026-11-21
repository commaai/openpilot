import unittest
from tinygrad import Device
from tinygrad.helpers import DEV
from test.runtime import test_ops

class TestOps(unittest.TestCase):
  @unittest.skipUnless(Device.DEFAULT == "CPU" and DEV.renderer == "LLVM", "DEVECTORIZE=0 only for LLVM")
  def test_strided_conv2d_simple_vec(self):
    test_ops.TestOps.test_strided_conv2d_simple(self)

if __name__ == "__main__": unittest.main()
