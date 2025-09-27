import unittest

from test.helpers import assert_jit_cache_len
from tinygrad import Variable, Tensor, TinyJit
from tinygrad.helpers import RANGEIFY
import numpy as np

class TestSymbolicJit(unittest.TestCase):
  def test_plus1(self):
    def f(a): return (a+1).realize()
    jf = TinyJit(f)
    a = Tensor.rand(3, 10)
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      symbolic = jf(a[:, :vi])[:3, :i].numpy()
      expected = f(a[:, :i]).numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)
    assert_jit_cache_len(jf, 1)

  def test_plus1_pad(self):
    # TODO: without contiguous, the pad is not captured in jit
    def f(a): return (a+1).pad((None, (0, 10-a.shape[1]))).contiguous().realize()
    jf = TinyJit(f)
    a = Tensor.rand(3, 10)
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      symbolic = jf(a[:, :vi]).numpy()
      expected = f(a[:, :i]).numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)
    assert_jit_cache_len(jf, 1 if RANGEIFY else 2) # one add and one pad, can be one kernel?

  def test_add(self):
    def f(a, b): return (a+b).realize()
    jf = TinyJit(f)
    a = Tensor.rand(3, 10)
    b = Tensor.rand(3, 10)
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      symbolic = jf(a[:, :vi], b[:, :vi])
      symbolic = symbolic[:3, :i].numpy()
      expected = f(a[:, :i], b[:, :i]).numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)
    assert_jit_cache_len(jf, 1)

  def test_matmul(self):
    def f(a, b): return (a@b).realize()
    jf = TinyJit(f)
    a = Tensor.rand(3, 10)
    b = Tensor.rand(10, 5)
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      symbolic = jf(a[:, :vi], b[:vi, :]).numpy()
      expected = f(a[:, :i], b[:i, :]).numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)
    assert_jit_cache_len(jf, 1)

  def test_mixed_with_no_symbol_kernel(self):
    def f(a, b):
      s = (a@b).realize()
      s = (s+s).realize() # this one does not have symbols in input
      return s
    jf = TinyJit(f)
    a = Tensor.rand(3, 10)
    b = Tensor.rand(10, 5)
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      symbolic = jf(a[:, :vi], b[:vi, :]).numpy()
      expected = f(a[:, :i], b[:i, :]).numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)
    assert_jit_cache_len(jf, 2)

  def test_attention(self):
    def f(q, k, v): return Tensor.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).realize()
    jf = TinyJit(f)
    q = Tensor.rand(2, 1, 4, 8)
    k = Tensor.rand(2, 10, 4, 8)
    v = Tensor.rand(2, 10, 4, 8)
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      symbolic = jf(q, k[:, :vi], v[:, :vi])[:2, :4, :1, :8].numpy()
      expected = f(q, k[:, :i], v[:, :i]).numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)
    assert_jit_cache_len(jf, 4 if RANGEIFY else 5)

  def test_cat_dim0(self):
    def f(a, b): return a.cat(b, dim=0).realize()
    jf = TinyJit(f)
    a = Tensor.rand(10, 3)
    b = Tensor.rand(2, 3)
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      symbolic = jf(a[:vi], b)[:i+2, :3].numpy()
      expected = f(a[:i], b).numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)
    assert_jit_cache_len(jf, 1)

  def test_cat_dim1(self):
    def f(a, b): return a.cat(b, dim=1).realize()
    jf = TinyJit(f)
    a = Tensor.rand(3, 10)
    b = Tensor.rand(3, 2)
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      symbolic = jf(a[:, :vi], b)[:3, :i+2].numpy()
      expected = f(a[:, :i], b).numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)
    assert_jit_cache_len(jf, 1)

  def test_cat_dim0_two_vars(self):
    def f(a, b): return a.cat(b, dim=0).realize()
    jf = TinyJit(f)
    a = Tensor.rand(10, 3)
    b = Tensor.rand(10, 3)
    for i in range(2, 5):
      for j in range(2, 5):
        vi = Variable("i", 1, 10).bind(i)
        vj = Variable("j", 1, 10).bind(j)
        symbolic = jf(a[:vi], b[:vj])[:i+j, :3].numpy()
        expected = f(a[:i], b[:j]).numpy()
        np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)
    assert_jit_cache_len(jf, 1)

  def test_cat_dim1_two_vars(self):
    def f(a, b): return a.cat(b, dim=1).realize()
    jf = TinyJit(f)
    a = Tensor.rand(3, 10)
    b = Tensor.rand(3, 10)
    for i in range(2, 5):
      for j in range(2, 5):
        vi = Variable("i", 1, 10).bind(i)
        vj = Variable("j", 1, 10).bind(j)
        symbolic = jf(a[:, :vi], b[:, :vj])[:3, :i+j].numpy()
        expected = f(a[:, :i], b[:, :j]).numpy()
        np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)
    assert_jit_cache_len(jf, 1)

  def test_two_vars_plus1_ij(self):
    def f(a, b): return (a@b+1).realize()
    jf = TinyJit(f)
    a = Tensor.rand(10, 3)
    b = Tensor.rand(3, 10)
    for i in range(2, 5):
      for j in range(2, 5):
        vi = Variable("i", 1, 10).bind(i)
        vj = Variable("j", 1, 10).bind(j)
        symbolic = jf(a[:vi, :], b[:, :vj])[:i, :j].numpy()
        expected = f(a[:i, :], b[:, :j]).numpy()
        np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)
    assert_jit_cache_len(jf, 1)

  def test_two_vars_plus1_ji(self):
    def f(a, b): return (a@b+1).realize()
    jf = TinyJit(f)
    a = Tensor.rand(10, 3)
    b = Tensor.rand(3, 10)
    for i in range(2, 5):
      for j in range(2, 5):
        vi = Variable("i", 1, 10).bind(i)
        vj = Variable("j", 1, 10).bind(j)
        symbolic = jf(a[:vj, :], b[:, :vi])[:j, :i].numpy()
        expected = f(a[:j, :], b[:, :i]).numpy()
        np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)
    assert_jit_cache_len(jf, 1)

  def test_jit_symbolic_shape_mismatch(self):
    @TinyJit
    def add(a, b): return (a+b).realize()
    a = Tensor.rand(3, 10)
    b = Tensor.rand(3, 10)
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      add(a[:, :vi], b[:, :vi])
    vi2 = Variable("i", 1, 10).bind(7)
    a = Tensor.rand(3, 7)[:, :vi2]
    bad = Tensor.rand(4, 7)[:, :vi2]
    with self.assertRaises(AssertionError):
      add(a, bad)

  def test_shrink(self):
    # shrink is a movement, so we pair it with a simple function to test the JIT interaction
    def f(a): return (a+1).realize()
    jf = TinyJit(f)
    a = Tensor.rand(7, 11)
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      symbolic = a.shrink(((3,5),(vi,vi+2)))
      symbolic = jf(symbolic).numpy()
      expected = f(a.shrink(((3,5),(i,i+2)))).numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)
    assert_jit_cache_len(jf, 1)

  def test_slice(self):
    # slice is a movement, so we pair it with a simple function to test the JIT interaction
    def f(a): return (a+1).realize()
    jf = TinyJit(f)
    a = Tensor.rand(7, 11)
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      symbolic = a[3:5, vi:vi+2]
      symbolic = jf(symbolic).numpy()
      expected = f(a[3:5, i:i+2]).numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)
    assert_jit_cache_len(jf, 1)

  def test_slice_var_shape(self):
    def f(a): return (a+1).realize()
    jf = TinyJit(f)
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      a = Tensor.ones(vi, 11).contiguous()
      symbolic = a[:, 1:2]
      symbolic = jf(symbolic)[:i, :1].numpy()
      expected = f(a[:i, :][:, 1:2]).numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)
    assert_jit_cache_len(jf, 1)

  def test_ones_sum(self):
    def f(a): return a.sum().realize()
    jf = TinyJit(f)
    t = Tensor.ones(10)
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      symbolic = jf(t[:vi]).item()
      expected = f(t[:i]).item()
      np.testing.assert_equal(symbolic, expected)

  def test_mean(self):
    def f(a): return a.mean().realize()
    def f0(a): return a.mean(0).realize()
    def f1(a): return a.mean(1).realize()
    jf = TinyJit(f)
    jf0 = TinyJit(f0)
    jf1 = TinyJit(f1)
    a = Tensor.rand(10, 3)
    b = Tensor.rand(10, 3)
    c = Tensor.rand(10, 3)
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      # axis = None
      symbolic = jf(a[:vi]).numpy()
      expected = a[:i].mean().numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)
      # axis = 0
      symbolic = jf0(b[:vi]).numpy()
      expected = b[:i].mean(0).numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)
      # axis = 1
      symbolic = jf1(c[:vi])[:i].numpy()
      expected = c[:i].mean(1).numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

  def test_mean_2d(self):
    def f(a): return a.mean().realize()
    def f0(a): return a.mean(0).realize()
    def f1(a): return a.mean(1).realize()
    jf = TinyJit(f)
    jf0 = TinyJit(f0)
    jf1 = TinyJit(f1)
    a = Tensor.rand(10, 10)
    b = Tensor.rand(10, 10)
    c = Tensor.rand(10, 10)
    for i in range(2, 5):
      for j in range(2, 5):
        vi = Variable("i", 1, 10).bind(i)
        vj = Variable("j", 1, 10).bind(j)
        # axis = None
        symbolic = jf(a[:vi, :vj]).numpy()
        expected = a[:i, :j].mean().numpy()
        np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)
        # axis = 0
        symbolic = jf0(b[:vi, :vj])[:j].numpy()
        expected = b[:i, :j].mean(0).numpy()
        np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)
        # axis = 1
        symbolic = jf1(c[:vi, :vj])[:i].numpy()
        expected = c[:i, :j].mean(1).numpy()
        np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

  def test_var(self):
    def f(a): return a.var().realize()
    def f0(a): return a.var(0).realize()
    def f1(a): return a.var(1).realize()
    jf = TinyJit(f)
    jf0 = TinyJit(f0)
    jf1 = TinyJit(f1)
    a = Tensor.rand(10, 3)
    b = Tensor.rand(10, 3)
    c = Tensor.rand(10, 3)
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      # axis = None
      symbolic = jf(a[:vi]).numpy()
      expected = a[:i].var().numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)
      # axis = 0
      symbolic = jf0(b[:vi]).numpy()
      expected = b[:i].var(0).numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)
      # axis = 1
      symbolic = jf1(c[:vi])[:i].numpy()
      expected = c[:i].var(1).numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

  def test_var_2d(self):
    def f(a): return a.var().realize()
    def f0(a): return a.var(0).realize()
    def f1(a): return a.var(1).realize()
    jf = TinyJit(f)
    jf0 = TinyJit(f0)
    jf1 = TinyJit(f1)
    a = Tensor.rand(10, 10)
    b = Tensor.rand(10, 10)
    c = Tensor.rand(10, 10)
    for i in range(2, 5):
      for j in range(2, 5):
        vi = Variable("i", 1, 10).bind(i)
        vj = Variable("j", 1, 10).bind(j)
        # axis = None
        symbolic = jf(a[:vi, :vj]).numpy()
        expected = a[:i, :j].var().numpy()
        np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)
        # axis = 0
        symbolic = jf0(b[:vi, :vj])[:j].numpy()
        expected = b[:i, :j].var(0).numpy()
        np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)
        # axis = 1
        symbolic = jf1(c[:vi, :vj])[:i].numpy()
        expected = c[:i, :j].var(1).numpy()
        np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

if __name__ == '__main__':
  unittest.main()
