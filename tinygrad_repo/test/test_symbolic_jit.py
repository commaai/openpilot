import unittest

from test.helpers import assert_jit_cache_len
from tinygrad import Variable, Tensor, TinyJit
import numpy as np

class TestSymbolicJit(unittest.TestCase):
  def test_plus1(self):
    def f(a): return (a+1).realize()
    jf = TinyJit(f)
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      a = Tensor.rand(3, i)
      symbolic = jf(a.reshape(3, vi)).reshape(3, i).numpy()
      expected = f(a).numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)
    assert_jit_cache_len(jf, 1)

  def test_add(self):
    def f(a, b): return (a+b).realize()
    jf = TinyJit(f)
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      a = Tensor.rand(3, i)
      b = Tensor.rand(3, i)
      symbolic = jf(a.reshape(3, vi), b.reshape(3, vi)).reshape(3, i).numpy()
      expected = f(a, b).numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)
    assert_jit_cache_len(jf, 1)

  def test_matmul(self):
    def f(a, b): return (a@b).realize()
    jf = TinyJit(f)
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      a = Tensor.rand(3, i)
      b = Tensor.rand(i, 5)
      symbolic = jf(a.reshape(3, vi), b.reshape(vi, 5)).numpy()
      expected = f(a, b).numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)
    assert_jit_cache_len(jf, 1)

  def test_mixed_with_no_symbol_kernel(self):
    def f(a, b):
      s = (a@b).realize()
      s = (s+s).realize() # this one does not have symbols in input
      return s
    jf = TinyJit(f)
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      a = Tensor.rand(3, i)
      b = Tensor.rand(i, 5)
      symbolic = jf(a.reshape(3, vi), b.reshape(vi, 5)).numpy()
      expected = f(a, b).numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)
    assert_jit_cache_len(jf, 2)

  def test_attention(self):
    def f(q, k, v): return Tensor.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).realize()
    jf = TinyJit(f)
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      q = Tensor.rand(2, 1, 4, 8)
      k = Tensor.rand(2, i, 4, 8)
      v = Tensor.rand(2, i, 4, 8)
      symbolic = jf(q, k.reshape(2, vi, 4, 8), v.reshape(2, vi, 4, 8)).reshape(2, 4, 1, 8).numpy()
      expected = f(q, k, v).numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)
    assert_jit_cache_len(jf, 5)

  def test_cat_dim0(self):
    def f(a, b): return a.cat(b, dim=0).realize()
    jf = TinyJit(f)
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      a = Tensor.rand(i, 3)
      b = Tensor.rand(2, 3)
      symbolic = jf(a.reshape(vi, 3), b).reshape(i+2, 3).numpy()
      expected = f(a, b).numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)
    assert_jit_cache_len(jf, 1)

  def test_cat_dim1(self):
    def f(a, b): return a.cat(b, dim=1).realize()
    jf = TinyJit(f)
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      a = Tensor.rand(3, i)
      b = Tensor.rand(3, 2)
      symbolic = jf(a.reshape(3, vi), b).reshape(3, i+2).numpy()
      expected = f(a, b).numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)
    assert_jit_cache_len(jf, 1)

  def test_cat_dim0_two_vars(self):
    def f(a, b): return a.cat(b, dim=0).realize()
    jf = TinyJit(f)
    for i in range(1, 5):
      for j in range(1, 5):
        vi = Variable("i", 1, 10).bind(i)
        vj = Variable("j", 1, 10).bind(j)
        a = Tensor.rand(i, 3)
        b = Tensor.rand(j, 3)
        symbolic = jf(a.reshape(vi, 3), b.reshape(vj, 3)).reshape(i+j, 3).numpy()
        expected = f(a, b).numpy()
        np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)
    assert_jit_cache_len(jf, 1)

  def test_cat_dim1_two_vars(self):
    def f(a, b): return a.cat(b, dim=1).realize()
    jf = TinyJit(f)
    for i in range(1, 5):
      for j in range(1, 5):
        vi = Variable("i", 1, 10).bind(i)
        vj = Variable("j", 1, 10).bind(j)
        a = Tensor.rand(3, i)
        b = Tensor.rand(3, j)
        symbolic = jf(a.reshape(3, vi), b.reshape(3, vj)).reshape(3, i+j).numpy()
        expected = f(a, b).numpy()
        np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)
    assert_jit_cache_len(jf, 1)

  def test_two_vars_plus1_ij(self):
    def f(a, b): return (a@b+1).realize()
    jf = TinyJit(f)
    for i in range(1, 5):
      for j in range(1, 5):
        vi = Variable("i", 1, 10).bind(i)
        vj = Variable("j", 1, 10).bind(j)
        a = Tensor.rand(i, 3)
        b = Tensor.rand(3, j)
        symbolic = jf(a.reshape(vi, 3), b.reshape(3, vj)).reshape(i, j).numpy()
        expected = f(a, b).numpy()
        np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)
    assert_jit_cache_len(jf, 1)

  def test_two_vars_plus1_ji(self):
    def f(a, b): return (a@b+1).realize()
    jf = TinyJit(f)
    for i in range(1, 5):
      for j in range(1, 5):
        vi = Variable("i", 1, 10).bind(i)
        vj = Variable("j", 1, 10).bind(j)
        a = Tensor.rand(j, 3)
        b = Tensor.rand(3, i)
        symbolic = jf(a.reshape(vj, 3), b.reshape(3, vi)).reshape(j, i).numpy()
        expected = f(a, b).numpy()
        np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)
    assert_jit_cache_len(jf, 1)

  def test_jit_symbolic_shape_mismatch(self):
    @TinyJit
    def add(a, b): return (a+b).realize()
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      a = Tensor.rand(3, i).reshape(3, vi)
      b = Tensor.rand(3, i).reshape(3, vi)
      add(a, b)
    vi2 = Variable("i", 1, 10).bind(7)
    a = Tensor.rand(3, 7).reshape(3, vi2)
    bad = Tensor.rand(4, 7).reshape(4, vi2)
    with self.assertRaises(AssertionError):
      add(a, bad)

  def test_shrink(self):
    # shrink is a movement, so we pair it with a simple function to test the JIT interaction
    def f(a): return (a+1).realize()
    jf = TinyJit(f)
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      a = Tensor.rand(7, 11)
      symbolic = a.shrink(((3,5),(vi,vi+2)))
      symbolic = jf(symbolic).numpy()
      expected = f(a.shrink(((3,5),(i,i+2)))).numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)
    assert_jit_cache_len(jf, 1)

  def test_ones_sum(self):
    def f(a): return a.sum().realize()
    jf = TinyJit(f)
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      t = Tensor.ones(i)
      symbolic = jf(t.reshape(vi)).item()
      expected = f(t).item()
      np.testing.assert_equal(symbolic, expected)

  def test_mean(self):
    def f(a): return a.mean().realize()
    def f0(a): return a.mean(0).realize()
    def f1(a): return a.mean(1).realize()
    jf = TinyJit(f)
    jf0 = TinyJit(f0)
    jf1 = TinyJit(f1)
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      # aixs = None
      a = Tensor.rand(i, 3)
      symbolic = jf(a.reshape(vi, 3)).numpy()
      expected = a.mean().numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)
      # aixs = 0
      a = Tensor.rand(i, 3)
      symbolic = jf0(a.reshape(vi, 3)).numpy()
      expected = a.mean(0).numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)
      # aixs = 1
      a = Tensor.rand(i, 3)
      symbolic = jf1(a.reshape(vi, 3)).reshape(i).numpy()
      expected = a.mean(1).numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

  def test_mean_2d(self):
    def f(a): return a.mean().realize()
    def f0(a): return a.mean(0).realize()
    def f1(a): return a.mean(1).realize()
    jf = TinyJit(f)
    jf0 = TinyJit(f0)
    jf1 = TinyJit(f1)
    for i in range(1, 5):
      for j in range(1, 5):
        vi = Variable("i", 1, 10).bind(i)
        vj = Variable("j", 1, 10).bind(j)
        # aixs = None
        a = Tensor.rand(i, j)
        symbolic = jf(a.reshape(vi, vj)).numpy()
        expected = a.mean().numpy()
        np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)
        # aixs = 0
        a = Tensor.rand(i, j)
        symbolic = jf0(a.reshape(vi, vj)).reshape(j).numpy()
        expected = a.mean(0).numpy()
        np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)
        # aixs = 1
        a = Tensor.rand(i, j)
        symbolic = jf1(a.reshape(vi, vj)).reshape(i).numpy()
        expected = a.mean(1).numpy()
        np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

  def test_var(self):
    def f(a): return a.var().realize()
    def f0(a): return a.var(0).realize()
    def f1(a): return a.var(1).realize()
    jf = TinyJit(f)
    jf0 = TinyJit(f0)
    jf1 = TinyJit(f1)
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      # aixs = None
      a = Tensor.rand(i, 3)
      symbolic = jf(a.reshape(vi, 3)).numpy()
      expected = a.var().numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)
      # aixs = 0
      a = Tensor.rand(i, 3)
      symbolic = jf0(a.reshape(vi, 3)).numpy()
      expected = a.var(0).numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)
      # aixs = 1
      a = Tensor.rand(i, 3)
      symbolic = jf1(a.reshape(vi, 3)).reshape(i).numpy()
      expected = a.var(1).numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

  def test_var_2d(self):
    def f(a): return a.var().realize()
    def f0(a): return a.var(0).realize()
    def f1(a): return a.var(1).realize()
    jf = TinyJit(f)
    jf0 = TinyJit(f0)
    jf1 = TinyJit(f1)
    for i in range(1, 5):
      for j in range(1, 5):
        vi = Variable("i", 1, 10).bind(i)
        vj = Variable("j", 1, 10).bind(j)
        # aixs = None
        a = Tensor.rand(i, j)
        symbolic = jf(a.reshape(vi, vj)).numpy()
        expected = a.var().numpy()
        np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)
        # aixs = 0
        a = Tensor.rand(i, j)
        symbolic = jf0(a.reshape(vi, vj)).reshape(j).numpy()
        expected = a.var(0).numpy()
        np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)
        # aixs = 1
        a = Tensor.rand(i, j)
        symbolic = jf1(a.reshape(vi, vj)).reshape(i).numpy()
        expected = a.var(1).numpy()
        np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

if __name__ == '__main__':
  unittest.main()