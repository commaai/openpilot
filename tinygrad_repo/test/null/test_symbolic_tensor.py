import unittest
from tinygrad import Variable
from tinygrad.tensor import Tensor

class TestSymbolic(unittest.TestCase):
  def assert_tuple_equal(self, x, y):
    for a,b in zip(x,y): self.assertFalse(a != b)

  def test_cat_dim0_is_expanded(self):
    i = Variable("i", 1, 5).bind(3)
    j = Variable("j", 1, 5).bind(3)
    k = Variable("k", 1, 5).bind(3)
    t = Tensor.rand(5, 4)[:i].cat(Tensor.rand(5, 4)[:j], dim=0).cat(Tensor.rand(5, 4)[:k], dim=0)
    self.assert_tuple_equal(t.shape, (i+j+k, 4))
    t = Tensor.rand(5, 3)[:i].cat(Tensor.rand(5, 3)[:i], dim=0).cat(Tensor.rand(3, 3), dim=0)
    self.assert_tuple_equal(t.shape, (2*i+3, 3))

  def test_cat_dim1_strides(self):
    i = Variable("i", 1, 5).bind(4)
    j = Variable("j", 1, 5).bind(4)
    k = Variable("k", 1, 5).bind(4)
    t = Tensor.rand(3, 5)[:, :i].cat(Tensor.rand(3, 5)[:, :j], dim=1).cat(Tensor.rand(3, 5)[:, :k], dim=1)
    self.assert_tuple_equal(t.shape, (3, i+j+k))

class TestSymbolicVarVals(unittest.TestCase):
  def assert_equal(self, x, y): self.assertFalse(x != y)

  def test_shrink_unbind(self):
    v = Variable("v", 1, 100)
    bv = Variable("v", 1, 100).bind(2)
    t = Tensor.rand(3, 4).shrink(((0,bv),(0,4)))
    unbound_st, var_val = t.uop.unbind_all()
    assert var_val == {v: 2}
    t = Tensor.rand(3, 4).shrink(((bv, bv+1), (0, 4)))
    unbound_st, var_val = t.uop.unbind_all()
    assert var_val == {v: 2}

class TestSymbolicReshape(unittest.TestCase):
  def test_reshape(self):
    a = Tensor.rand(5, 4)
    b = Tensor.rand(5, 6)
    for i in range(1, 6):
      vi = Variable("i", 1, 5).bind(i)
      ret = a[:vi]
      ret = ret.reshape((vi, 4))
      assert ret.shape == (vi, 4)
      ret = b[:vi]
      ret = ret.reshape((vi, 2, 3))
      assert ret.shape == (vi, 2, 3)

  def test_two_symbol_reshape(self):
    t = Tensor.rand(5, 5)
    for i in range(1, 6):
      for j in range(1, 6):
        vi = Variable("i", 1, 5).bind(i)
        vj = Variable("j", 1, 5).bind(j)
        ret = t[:vi, :vj]
        ret = ret.reshape(vj, vi)
        assert ret.shape == (vj, vi)
        ret = ret.reshape(vi, vj)
        assert ret.shape == (vi, vj)
        ret = ret.reshape(1, vi*vj)
        assert ret.shape == (1, vi*vj)

class TestSymbolicExpand(unittest.TestCase):
  def test_expand_into_symbols(self):
    vi = Variable("i", 1, 5).bind(3)
    vj = Variable("j", 1, 5).bind(3)
    a = Tensor([[1], [2], [3]]).expand((3, vi))
    assert a.shape == (3, vi)
    a = a.reshape(3, vi, 1).expand((3, vi, vj))
    assert a.shape == (3, vi, vj)

  def test_plus_expands_constant(self):
    a = Tensor.rand(3, 5)
    for i in range(1, 6):
      vi = Variable("i", 1, 5).bind(i)
      ret = a[:, :vi]
      ret = ret + 1
      self.assertTupleEqual(ret.shape, (3, vi))

  def test_pad_then_expand_into_symbols(self):
    vi = Variable("i", 1, 10).bind(3)
    a = Tensor(1).unsqueeze(0).pad((0, 24)).unsqueeze(0).expand((vi, 25))
    self.assertEqual(a.shape, (vi, 25))
    self.assertEqual(a.reshape(25*vi).shape, (vi*25,))
    self.assertEqual(a.reshape(vi*25).shape, (vi*25,))

class TestSymbolicShrink(unittest.TestCase):
  def test_shrink_symbols_simple(self):
    vi = Variable("i", 1, 5)
    t = Tensor.rand(5, 5).shrink(((0, 5),(0,vi)))
    assert t.shape == (5, vi)

  def test_shrink_symbols(self):
    vi = Variable("i", 1, 5)
    t = Tensor.rand(3, 5).shrink(((0, 2), (vi, vi+1)))
    assert t.shape == (2, 1)

if __name__ == '__main__':
  unittest.main()
