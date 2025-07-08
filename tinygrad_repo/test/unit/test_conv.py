import unittest
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.helpers import Context

class TestConv(unittest.TestCase):
  def test_simple(self):
    x = Tensor.ones(1,12,16,32).contiguous().realize()
    w = Tensor.ones(32,12,3,3).contiguous().realize()
    ret = x.conv2d(w, stride=(2,2), padding=(1,1)).numpy()
    # it's not 108 around the padding
    assert (ret[:, :, 1:-1, 1:-1] == 108).all()
    assert ret[0,0,0,0] == 48
    assert ret[0,0,0,1] == 72

  def test_simple_rand(self):
    x = Tensor.rand(1,12,16,32)
    w = Tensor.rand(32,12,3,3)
    x.conv2d(w, stride=(2,2), padding=(1,1)).numpy()

  def test_many_simple(self):
    x = Tensor(np.arange(8*2*8).reshape(1,8,2,8).astype(np.float32))
    #w = Tensor(np.arange(8*8*1*1).reshape(8,8,1,1).astype(np.float32))
    w = Tensor.eye(8).reshape((8,8,1,1))
    ret = x.conv2d(w, stride=(1,2), padding=(0,0)).numpy()
    print(ret)

  def test_lazycache(self):
    x = Tensor.rand(1, 32)
    y = Tensor.rand(32)
    out = x + y.reshape((1,32,1)).reshape((1,32)) + y.reshape((1,32,1)).reshape((1,32))
    out.numpy()

  def test_simple_biased(self):
    C = 8
    x = Tensor.rand(1,C,5,5)
    w = Tensor.eye(C).reshape((C,C,1,1))
    b = Tensor(np.arange(C).astype(np.float32))
    ret = Tensor.conv2d(x,w,b).relu().conv2d(w,b)

    print(ret.numpy())

  def test_two_binops_no_rerun_small(self):
    x = Tensor.rand(1,1,32,32)
    w = Tensor.rand(1,1,3,3)
    out = x.conv2d(w, padding=(1,1))
    np.testing.assert_allclose(out.relu().numpy(), np.maximum(out.numpy(), 0))

  def test_two_binops_no_rerun(self):
    x = Tensor.randn(1,12,16,32)
    w = Tensor.randn(32,12,3,3)
    out = x.conv2d(w, stride=(2,2), padding=(1,1))
    r1, r2 = out.relu(), (out-1)
    np.testing.assert_allclose(r1.numpy(), np.maximum(out.numpy(), 0))
    np.testing.assert_allclose(r2.numpy(), out.numpy() - 1)

  def test_two_overlapping_binops_no_rerun(self):
    x = Tensor.randn(1,12,16,32)
    w = Tensor.randn(32,12,3,3)
    out = x.conv2d(w, stride=(2,2), padding=(1,1))
    r1, r2 = out.relu(), out.elu()
    np.testing.assert_allclose(r1.numpy(), np.maximum(out.numpy(), 0))
    np.testing.assert_allclose(r2.numpy(), np.where(out.numpy() > 0, out.numpy(), (np.exp(out.numpy()) - 1)), atol=1e-5)

  def test_two_overlapping_binops_no_rerun_wino(self):
    with Context(WINO=1):
      x = Tensor.randn(1,4,16,16)
      w = Tensor.randn(6,4,3,3)
      out = x.conv2d(w, padding=(1,1))
      r1, r2 = out.relu(), out.elu()
      np.testing.assert_allclose(r1.numpy(), np.maximum(out.numpy(), 0))
      np.testing.assert_allclose(r2.numpy(), np.where(out.numpy() > 0, out.numpy(), (np.exp(out.numpy()) - 1)), atol=1e-5)

  def test_first_three(self):
    x = Tensor.rand(1,12,16,32)

    w = Tensor.rand(32,12,3,3)
    x = x.conv2d(w, stride=(2,2), padding=(1,1)).elu()

    w = Tensor.rand(32,1,3,3)
    x = x.conv2d(w, padding=(1,1), groups=32).elu()

    w = Tensor.rand(16,32,1,1)
    x = x.conv2d(w).elu()

    x = x.numpy()
    print(x.shape)

  def test_elu(self):
    x = Tensor.rand(1,12,16,32)

    w = Tensor.rand(32,12,3,3)
    x = x.conv2d(w, stride=(2,2), padding=(1,1))

    x = x.elu()

    w = Tensor.rand(32,1,3,3)
    x = x.conv2d(w, padding=(1,1), groups=32)
    x.numpy()

  def test_reduce_relu(self):
    x = Tensor.rand(1,12,16,32)
    x = x.sum(keepdim=True).relu()
    x.numpy()

  def test_bias(self):
    from tinygrad.nn import Conv2d
    x = Tensor.rand(1,12,16,32)
    c = Conv2d(12, 32, 3)
    x = c(x).relu()
    w = Tensor.uniform(32, 1, 3, 3)
    x = x.conv2d(w, groups=32)
    x.numpy()

  def test_multiadd(self):
    w = Tensor.rand(32)
    x = Tensor.rand(32).relu()
    (w+x).numpy()

  def test_reorder(self):
    x = Tensor.rand(1,12,16,32)
    w = Tensor.rand(12,12,3,3)
    x = x.conv2d(w, padding=(1,1))
    print(x.shape)
    x = x.reshape((1, 12, 32, 16))
    x += 1
    x += 1
    x = x.reshape((1, 12, 16, 32))
    x.numpy()

if __name__ == '__main__':
  unittest.main()
