import unittest
from tinygrad.helpers import prod
from tinygrad.shape.view import View
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad import Variable
from test.unit.test_shapetracker import shapetracker_getitem

class MultiShapeTracker:
  def __init__(self, sts:list[ShapeTracker]): self.sts = sts
  @property
  def shape(self): return self.sts[0].shape
  def reshape(self, arg): self.sts = [x.reshape(arg) for x in self.sts]
  def permute(self, arg): self.sts = [x.permute(arg) for x in self.sts]
  def expand(self, arg): self.sts = [x.expand(arg) for x in self.sts]
  def shrink(self, arg): self.sts = [x.shrink(arg) for x in self.sts]
  def flip(self, arg): self.sts = [x.flip(arg) for x in self.sts]
  def pad(self, arg): self.sts = [x.pad(arg) for x in self.sts]

def st_equal(st1:ShapeTracker, st2:ShapeTracker) -> bool:
  if st1.shape != st2.shape: return False
  if st1 == st2: return True
  for i in range(0, prod(st1.shape)):
    st1_off, st1_v = shapetracker_getitem(st1, i)
    st2_off, st2_v = shapetracker_getitem(st2, i)
    if st1_v != st2_v or (st1_off != st2_off and st1_v):
      print(f"ST MISMATCH @ {i}, {st1_v=} != {st2_v=}, {st1_off=} != {st2_off=}")
      print(st1)
      print(st2)
      return False
  return True

class TestShapeTrackerBasics(unittest.TestCase):
  def test_pad_shrink_removes_mask(self):
    a = ShapeTracker.from_shape((10, 10))
    a = a.pad(((0,2), (0,2)))
    a = a.shrink(((0,10), (0,10)))
    assert len(a.views) == 1 and a.views[-1].mask is None

  def test_pad_shrink_leaves_mask(self):
    a = ShapeTracker.from_shape((10, 10))
    a = a.pad(((0,2), (0,2)))
    a = a.shrink(((0,10), (0,11)))
    assert len(a.views) == 1 and a.views[-1].mask is not None

  def test_reshape_makes_same(self):
    a = ShapeTracker.from_shape((2, 5))
    x = a.pad( ((2, 0), (0, 0)) )
    x = x.reshape( (2, 2, 5) )
    x1 = x.reshape( (4, 5) )
    x1 = x1.reshape( (2, 2, 5) )
    assert x == x1.simplify()

  def test_simplify_is_correct(self):
    multiv = ShapeTracker(views=(View(shape=(15, 3), strides=(9, 1), offset=6, mask=None, contiguous=False),
                                 View(shape=(4, 3), strides=(12, 4), offset=0, mask=None, contiguous=False)))
    assert st_equal(multiv, multiv.simplify())

class TestShapeTrackerAdd(unittest.TestCase):
  def test_simple_add_reshape(self):
    a = ShapeTracker.from_shape((10, 10))
    a = a.reshape((100,))
    b = ShapeTracker.from_shape((100,))
    assert a+b == b

  def test_simple_add_permute(self):
    a = ShapeTracker.from_shape((10, 10))
    a = a.permute((1,0))
    b = ShapeTracker.from_shape((10, 10))
    b = b.permute((1,0))
    assert a+b == ShapeTracker.from_shape((10, 10))

  def test_plus_real1(self):
    st = MultiShapeTracker([ShapeTracker.from_shape((15, 9))])
    st.shrink( ((0, 15), (6, 9)) )
    backup = st.sts[0]
    st.sts.append(ShapeTracker.from_shape(backup.shape))
    st.reshape( (45,) )
    st.flip( (True,) )
    st.reshape( (15, 3) )
    assert st_equal(backup + st.sts[1], st.sts[0])

  def test_off_by_one(self):
    st1 = ShapeTracker(views=(View(shape=(5,), strides=(1,), offset=0, mask=None, contiguous=True),
                              View(shape=(5,), strides=(1,), offset=0, mask=None, contiguous=True)))
    st2 = ShapeTracker(views=(View(shape=(4,), strides=(1,), offset=0, mask=None, contiguous=True),
                              View(shape=(5,), strides=(1,), offset=0, mask=None, contiguous=True)))
    assert not (st_equal(st1, st2))

class TestShapeTrackerAddVariable(unittest.TestCase):
  def test_self_add(self):
    j = Variable("j", 0, 20).bind(10)
    a = ShapeTracker.from_shape((10,10))
    x = a.reshape((10, j))
    out = x + x
    assert out == x

  def test_self_add_reshape(self):
    j = Variable("j", 0, 20).bind(10)
    a = ShapeTracker.from_shape((10,10))
    x = a.reshape((10, j))
    out = x.reshape((5, 2, j)) + x
    assert out == x

  def test_merge_symbolic_views(self):
    var_i = Variable('i', 1, 10)
    var_j = Variable('i', 1, 10)
    vm1 = View(shape=(var_i, var_j, 3), strides=(3, 0, 1), offset=0, mask=None, contiguous=False)
    vm2 = View(shape=(var_i, var_j, 3), strides=(var_j*3, 3, 1), offset=0, mask=None, contiguous=True)
    ShapeTracker((vm1,)) + ShapeTracker((vm2,))

  def test_merge_symbolic_views_2(self):
    var_i = Variable('i', 1, 10)
    var_j = Variable('j', 1, 10)
    vm1 = View(shape=(var_i, var_j), strides=(0, 0), offset=0, mask=None, contiguous=False)
    vm2 = View(shape=(var_i, var_j), strides=(var_j, 1), offset=0, mask=None, contiguous=True)
    ret = (ShapeTracker((vm1,)) + ShapeTracker((vm2,))).reshape((var_i, var_j, 1))
    ret_2 = ShapeTracker((vm1,)) + ShapeTracker((vm2,)).reshape((var_i, var_j, 1))
    assert ret == ret_2

class TestShapeTrackerInvert(unittest.TestCase):
  def test_invert_reshape(self):
    a = ShapeTracker.from_shape((10, 10))
    x = a.reshape((5, 20))
    ap = ShapeTracker.from_shape(x.shape) + x.invert(a.shape)
    assert ap == a, f"{ap} != {a}"

  def test_invert_permute(self):
    a = ShapeTracker.from_shape((5, 20))
    x = a.permute((1,0))
    ap = x + x.invert(a.shape)
    assert ap == a, f"{ap} != {a}"

  def test_invert_permute_3(self):
    a = ShapeTracker.from_shape((8, 4, 5))
    x = a.permute((1,2,0))
    ap = x + x.invert(a.shape)
    assert ap == a, f"{ap} != {a}"

  def test_invert_real1(self):
    a = ShapeTracker.from_shape((3, 6, 10))
    x = a.reshape( (3, 3, 2, 10) )
    x = x.permute( (2, 1, 3, 0) )
    ap = x + x.invert(a.shape)
    assert ap == a, f"{ap} != {a}"

  def test_cant_invert_expand(self):
    a = ShapeTracker.from_shape((10, 1))
    x = a.expand((10,10))
    assert x.invert(a.shape) is None

  def test_cant_invert_shrink(self):
    a = ShapeTracker.from_shape((10, 10))
    x = a.shrink(((0,10),(2,8)))
    assert x.invert(a.shape) is None

  def test_can_invert_flip(self):
    a = ShapeTracker.from_shape((20, 10))
    x = a.flip((True,False))
    ap = x + x.invert(a.shape)
    assert st_equal(ap, a)

  def test_can_invert_flip_permute(self):
    a = ShapeTracker.from_shape((20, 10))
    x = a.permute((1,0))
    x = x.flip((True,False))
    ap = x + x.invert(a.shape)
    assert st_equal(ap, a)

  def test_invert_failure(self):
    a = ShapeTracker.from_shape((2, 5))
    x = a.pad( ((2, 0), (0, 0)) )
    x = x.reshape( (2, 2, 5) )
    x = x.reshape( (4, 5) )
    ap = x + x.invert(a.shape)
    assert st_equal(ap, a)

if __name__ == '__main__':
  unittest.main()
