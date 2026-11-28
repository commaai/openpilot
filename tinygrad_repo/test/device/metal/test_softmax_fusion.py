import unittest
from tinygrad import Tensor, Device

class TestFuse(unittest.TestCase):
  @unittest.skipUnless(Device.DEFAULT == "METAL", "METAL TC")
  def test_fuse_and_tc_opt(self):
    A = Tensor.randn(8, 8).realize()
    B = Tensor.randn(8, 8).realize()
    C = Tensor.ones(1, 8, 8).pad(((1,1), None, None),).sum(0)
    out = (C + (A @ B))
    out.realize()

if __name__ == "__main__": unittest.main()
