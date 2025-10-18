import unittest
from tinygrad.shape.shapetracker import ShapeTracker, View
from tinygrad import Variable
from tinygrad.tensor import Tensor

class TestSymbolic(unittest.TestCase):
  def assert_tuple_equal(self, x, y):
    for a,b in zip(x,y): self.assertFalse(a != b)

  def test_symbolic_st(self):
    x = Variable("x", 1, 100)
    st = ShapeTracker.from_shape((x, 3))
    assert st.shape == (x, 3)
    assert st.real_strides() == (3, 1)

  def test_real_strides_0(self):
    st = ShapeTracker(views=(View(shape=(2, (Variable('start_pos', 1, 8)+1), 1, 1), strides=(8, 1, 0, 0), offset=0, mask=((0, 2), (0, Variable('start_pos', 1, 8)), (0, 1), (0, 1)), contiguous=False), View(shape=(2, (Variable('start_pos', 1, 8)+1)), strides=((Variable('start_pos', 1, 8)+1), 1), offset=0, mask=None, contiguous=True)))   # noqa: E501
    self.assertEqual(st.real_strides(), (8, None))

  @unittest.expectedFailure
  def test_real_strides_1(self):
    st = ShapeTracker(views=(View(shape=(3, (Variable('i', 1, 10)+2)), strides=(Variable('i', 1, 10), 1), offset=0, mask=((0, 3), (0, Variable('i', 1, 10))), contiguous=False),))   # noqa: E501
    self.assertEqual(st.real_strides(), (Variable('i', 1, 10), None))

  @unittest.expectedFailure
  def test_real_strides_2(self):
    st = ShapeTracker(views=(View(shape=(3, (Variable('i', 1, 10)+Variable('j', 1, 10))), strides=(Variable('i', 1, 10), 1), offset=0, mask=((0, 3), (0, Variable('i', 1, 10))), contiguous=False),))   # noqa: E501
    self.assertEqual(st.real_strides(), (Variable('i', 1, 10), None))

  def test_merge_view_recursion_err(self):
    vm2 = View(shape=(Variable('j', 1, 10),), strides=(0,), offset=0, mask=None, contiguous=False)
    vm1 = View(shape=(1,), strides=(0,), offset=0, mask=None, contiguous=True)
    self.assertEqual(vm2+vm1, vm1)

  def test_merge_view_recursion_err2(self):
    vm2 = View(shape=(Variable('a', 1, 10).bind(4),), strides=(0,), offset=0, mask=None, contiguous=False)
    # NOTE: vm1 is different from what create function would give, and this test vm2+vm1 halts
    vm1 = View(shape=(Variable('a', 1, 10).bind(4),), strides=(1,), offset=0, mask=((0, Variable('a', 1, 10).bind(4)),), contiguous=False)
    self.assertEqual(vm2+vm1, None)

    vm3 = View.create(shape=(Variable('a', 1, 10).bind(4),))
    self.assertEqual(vm3.shape, vm1.shape)
    self.assertEqual(vm3.strides, vm1.strides)
    self.assertEqual(vm2+vm3, vm2)

  def test_cat_dim0_strides(self):
    i = Variable("i", 1, 5).bind(3)
    j = Variable("j", 1, 5).bind(3)
    k = Variable("k", 1, 5).bind(3)
    t = Tensor.rand(5, 4)[:i].cat(Tensor.rand(5, 4)[:j], dim=0).cat(Tensor.rand(5, 4)[:k], dim=0)
    st = t.uop.st
    self.assert_tuple_equal(st.shape, (i+j+k, 4))
    assert st.real_strides() == (4, 1)
    t = Tensor.rand(5, 3)[:i].cat(Tensor.rand(5, 3)[:i], dim=0).cat(Tensor.rand(3, 3), dim=0)
    st = t.uop.st
    self.assert_tuple_equal(st.shape, (2*i+3, 3))
    assert st.real_strides() == (3, 1)

  def test_cat_dim1_strides(self):
    i = Variable("i", 1, 5).bind(4)
    j = Variable("j", 1, 5).bind(4)
    k = Variable("k", 1, 5).bind(4)
    t = Tensor.rand(3, 5)[:, :i].cat(Tensor.rand(3, 5)[:, :j], dim=1).cat(Tensor.rand(3, 5)[:, :k], dim=1)
    st = t.uop.st
    self.assert_tuple_equal(st.shape, (3, i+j+k))
    self.assert_tuple_equal(st.real_strides(), (i+j+k, 1))

class TestSymbolicVarVals(unittest.TestCase):
  def assert_equal(self, x, y): self.assertFalse(x != y)
  def test_var_vals_empty(self):
    assert ShapeTracker.from_shape((3, 4, 5)).var_vals == {}

  def test_var_vals_shape(self):
    x = Variable("x", 1, 100).bind(3)
    assert ShapeTracker.from_shape((x, 3)).var_vals == {"x": 3}

  def test_var_vals_offset(self):
    x = Variable("x", 1, 100).bind(3)
    st = ShapeTracker.from_shape((4, 3)).shrink(((x, x+1), (0, 3)))
    self.assert_equal(st.views[-1].offset, x * 3)
    assert st.var_vals == {"x": 3}

  def test_var_vals_mask(self):
    x = Variable("x", 1, 100).bind(3)
    view = View.create(shape=(3,4), strides=(4,1), offset=0, mask=((0, x), (0, 4)))
    st = ShapeTracker(views=(view,))
    assert st.var_vals == {"x": 3}

  def test_var_vals_complex(self):
    x = Variable("x", 1, 100).bind(3)
    y = Variable("y", 1, 100).bind(4)
    z = Variable("z", 1, 100).bind(5)
    st = ShapeTracker.from_shape((x, 5, y)).shrink(((0, x), (z, z+1), (0, 3)))
    self.assert_equal(st.views[-1].offset, y * z)
    assert st.var_vals == {"x": 3, "y": 4, "z": 5}

  def test_shrink_reshape(self):
    x = Variable("x", 1, 100).bind(3)
    st = ShapeTracker.from_shape((10, 10, 10)).shrink(((x, x+3), (3, 7), (2, 5)))
    st = st.reshape((3*4*3,))
    assert st.var_vals == {"x": 3}

class TestShapeTrackerUnbind(unittest.TestCase):
  def test_view_unbind(self):
    v = Variable("v", 1, 100)
    bv = Variable("v", 1, 100).bind(3)
    unbound_view, var_val = View.create(shape=(bv, 4)).unbind()
    assert unbound_view == View.create(shape=(v, 4))
    assert var_val == {v: 3}

  def test_shrink_unbind(self):
    v = Variable("v", 1, 100)
    bv = Variable("v", 1, 100).bind(2)
    t = Tensor.rand(3, 4).shrink(((0,bv),(0,4)))
    unbound_st, var_val = t.uop.st.unbind()
    assert unbound_st == ShapeTracker((View.create(shape=(v, 4)),))
    assert var_val == {v: 2}
    t = Tensor.rand(3, 4).shrink(((bv, bv+1), (0, 4)))
    unbound_st, var_val = t.uop.st.unbind()
    assert unbound_st == ShapeTracker((View.create(shape=(1, 4), offset=4*v),))
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

  def test_symbolic_mask(self):
    # taken from gpt2 single kvcache
    # these two caused problems in gpt2 if reshape merged views
    view = View(shape=(1, (Variable('start_pos', 1, 128).bind(2)+1), 16, 64), strides=(0, 0, 64, 1), offset=1024, mask=((0, 1), (Variable('start_pos', 1, 128).bind(2), (Variable('start_pos', 1, 128).bind(2)+1)), (0, 16), (0, 64)), contiguous=False)   # noqa: E501
    new_shape = (1, 1, (Variable('start_pos', 1, 128).bind(2)+1), 16, 64)
    assert view.reshape(new_shape) is None

    view = View(shape=(2, 1, (Variable('start_pos', 1, 128)+1), 16, 64), strides=(0, 0, 1024, 64, 1), offset=131072, mask=((1, 2), (0, 1), (0, (Variable('start_pos', 1, 128)+1)), (0, 16), (0, 64)), contiguous=False)   # noqa: E501
    new_shape = (2, (Variable('start_pos', 1, 128)+1), 16, 64)
    assert view.reshape(new_shape) is None

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

class TestSymbolicPad(unittest.TestCase):
  def test_pad(self):
    v = Variable("v", 1, 100).bind(5)
    t = Tensor.ones(100)[:v].pad(((4, 0),))
    t = t[:9]
    assert t.tolist() == [0,0,0,0,1,1,1,1,1]


if __name__ == '__main__':
  unittest.main()
