import unittest
from tinygrad import Variable
from tinygrad.tensor import Tensor
from examples.gpt2 import Attention
import numpy as np

class TestSymbolicOps(unittest.TestCase):
  def test_plus1(self):
    def f(a): return (a+1).realize()
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      a = Tensor.rand(3, i)
      symbolic = f(a.reshape(3, vi)).reshape(3, i).numpy()
      expected = f(a).numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

  def test_add(self):
    def f(a, b): return (a+b).realize()
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      a = Tensor.rand(3, i)
      b = Tensor.rand(3, i)
      symbolic = f(a.reshape(3, vi), b.reshape(3, vi)).reshape(3, i).numpy()
      expected = f(a, b).numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

  def test_matmul(self):
    def f(a, b): return (a@b).realize()
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      a = Tensor.rand(3, i)
      b = Tensor.rand(i, 5)
      symbolic = f(a.reshape(3, vi), b.reshape(vi, 5)).numpy()
      expected = f(a, b).numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

  def test_attention(self, dropout_p=0.0):
    def f(q, k, v): return Tensor.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=dropout_p).realize()
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      q = Tensor.rand(2, 1, 4, 8)
      k = Tensor.rand(2, i, 4, 8)
      v = Tensor.rand(2, i, 4, 8)
      symbolic = f(q, k.reshape(2, vi, 4, 8), v.reshape(2, vi, 4, 8)).reshape(2, 4, 1, 8).numpy()
      expected = f(q, k, v).numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

  def test_attention_training(self):
    with Tensor.train():
      self.test_attention(dropout_p=0.0)
      with self.assertRaises(ValueError):
        # symbolic shape dropout is not supported
        self.test_attention(dropout_p=0.5)

  def test_attention_pos_0_sz_0(self):
    Attention(128, 8)(Tensor.ones(1, 0, 128), Variable("start_pos", 0, 128).bind(0), None)

  def test_attention_pos_0_sz_1(self):
    Attention(128, 8)(Tensor.ones(1, 1, 128), Variable("start_pos", 0, 128).bind(0), None)

  def test_attention_pos_0_sz_2(self):
    Attention(128, 8)(Tensor.ones(1, 2, 128), Variable("start_pos", 0, 128).bind(0), None)

  def test_cat_dim0(self):
    def f(a, b): return a.cat(b, dim=0).realize()
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      a = Tensor.rand(i, 3)
      b = Tensor.rand(2, 3)
      symbolic = f(a.reshape(vi, 3), b).reshape(i+2, 3).numpy()
      expected = f(a, b).numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

  def test_cat_dim1(self):
    def f(a, b): return a.cat(b, dim=1).realize()
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      a = Tensor.rand(3, i)
      b = Tensor.rand(3, 2)
      symbolic = f(a.reshape(3, vi), b).reshape(3, i+2).numpy()
      expected = f(a, b).numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

  def test_cat_dim0_two_vars(self):
    def f(a, b): return a.cat(b, dim=0).realize()
    for i in range(1, 5):
      for j in range(1, 5):
        vi = Variable("i", 1, 10).bind(i)
        vj = Variable("j", 1, 10).bind(j)
        a = Tensor.rand(i, 3)
        b = Tensor.rand(j, 3)
        symbolic = f(a.reshape(vi, 3), b.reshape(vj, 3)).reshape(i+j, 3).numpy()
        expected = f(a, b).numpy()
        np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

  def test_cat_dim1_two_vars(self):
    def f(a, b): return a.cat(b, dim=1).realize()
    for i in range(1, 5):
      for j in range(1, 5):
        vi = Variable("i", 1, 10).bind(i)
        vj = Variable("j", 1, 10).bind(j)
        a = Tensor.rand(3, i)
        b = Tensor.rand(3, j)
        symbolic = f(a.reshape(3, vi), b.reshape(3, vj)).reshape(3, i+j).numpy()
        expected = f(a, b).numpy()
        np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

  def test_two_vars_plus1_ij(self):
    def f(a, b): return (a@b+1).realize()
    for i in range(1, 5):
      for j in range(1, 5):
        vi = Variable("i", 1, 10).bind(i)
        vj = Variable("j", 1, 10).bind(j)
        a = Tensor.rand(i, 3)
        b = Tensor.rand(3, j)
        symbolic = f(a.reshape(vi, 3), b.reshape(3, vj)).reshape(i, j).numpy()
        expected = f(a, b).numpy()
        np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

  def test_two_vars_plus1_ji(self):
    # reverse the order of variables
    def f(a, b): return (a@b+1).realize()
    for i in range(1, 5):
      for j in range(1, 5):
        vi = Variable("i", 1, 10).bind(i)
        vj = Variable("j", 1, 10).bind(j)
        a = Tensor.rand(j, 3)
        b = Tensor.rand(3, i)
        symbolic = f(a.reshape(vj, 3), b.reshape(3, vi)).reshape(j, i).numpy()
        expected = f(a, b).numpy()
        np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

  def test_shrink(self):
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      a = Tensor.rand(7, 11)
      symbolic = a.shrink(((3,5),(vi,vi+2)))
      symbolic = symbolic.numpy()
      expected = a.shrink(((3,5),(i,i+2))).numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

  def test_ones_sum(self):
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      t = Tensor.ones(i)
      symbolic = t.reshape(vi).sum().item()
      expected = t.sum().item()
      np.testing.assert_equal(symbolic, expected)

  def test_mean(self):
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      for axis in [None, 0, 1]:
        a = Tensor.rand(i, 3)
        expected = a.mean(axis).numpy()
        symbolic = a.reshape(vi, 3).mean(axis).reshape(expected.shape).numpy()
        np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

  def test_mean_2d(self):
    for i in range(1, 5):
      for j in range(1, 5):
        vi = Variable("i", 1, 10).bind(i)
        vj = Variable("j", 1, 10).bind(j)
        for axis in [None, 0, 1]:
          a = Tensor.rand(i, j)
          expected = a.mean(axis).numpy()
          symbolic = a.reshape(vi, vj).mean(axis).reshape(expected.shape).numpy()
          np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

  def test_var(self):
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      for axis in [None, 0, 1]:
        a = Tensor.rand(i, 3)
        expected = a.var(axis).numpy()
        symbolic = a.reshape(vi, 3).var(axis).reshape(expected.shape).numpy()
        np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

  def test_var_2d(self):
    for i in range(1, 5):
      for j in range(1, 5):
        vi = Variable("i", 1, 10).bind(i)
        vj = Variable("j", 1, 10).bind(j)
        for axis in [None, 0, 1]:
          a = Tensor.rand(i, j)
          expected = a.var(axis).numpy()
          symbolic = a.reshape(vi, vj).var(axis).reshape(expected.shape).numpy()
          np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

if __name__ == '__main__':
  unittest.main()