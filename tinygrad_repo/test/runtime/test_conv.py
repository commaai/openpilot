import unittest
import numpy as np
from tinygrad.tensor import Tensor

class TestConv(unittest.TestCase):
  def test_simple(self):
    x = Tensor.ones(1,12,16,32).contiguous().realize()
    w = Tensor.ones(32,12,3,3).contiguous().realize()
    ret = x.conv2d(w, stride=(2,2), padding=(1,1)).numpy()
    # it's not 108 around the padding
    assert (ret[:, :, 1:-1, 1:-1] == 108).all()
    assert ret[0,0,0,0] == 48
    assert ret[0,0,0,1] == 72

  def test_two_overlapping_binops_no_rerun(self):
    x = Tensor.randn(1,3,8,16)
    w = Tensor.randn(4,3,3,3)
    out = x.conv2d(w, stride=(2,2), padding=(1,1))
    r1, r2 = out.relu(), out.elu()
    np.testing.assert_allclose(r1.numpy(), np.maximum(out.numpy(), 0), atol=1e-5)
    np.testing.assert_allclose(r2.numpy(), np.where(out.numpy() > 0, out.numpy(), (np.exp(out.numpy()) - 1)), atol=1e-5)

if __name__ == '__main__':
  unittest.main()
