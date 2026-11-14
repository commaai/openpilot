import unittest
from tinygrad import Tensor, dtypes

class TestSetitem(unittest.TestCase):
  def test_setitem_dtype(self):
    for dt in (dtypes.int, dtypes.float, dtypes.bool):
      for v in (5., 5, True):
        t = Tensor.ones(6,6, dtype=dt).contiguous()
        t[1] = v
        self.assertEqual(t.dtype, dt)

  def test_setitem_dtype_mismatch(self):
    t = Tensor.zeros(6, dtype=dtypes.float).contiguous().realize()
    with self.assertRaises(RuntimeError): t[2:4] = Tensor([1, 2], dtype=dtypes.int)

class TestWithGrad(unittest.TestCase):
  def test_basic_setitem_works(self):
    z = Tensor.rand(8, 8)
    x = Tensor.rand(8)
    z[:3] = x

  def test_set_used_before_setitem(self):
    z = Tensor([1.0, 2.0, 3.0, 4.0])
    _ = z.sum()
    with self.assertRaises(RuntimeError):
      z[:2] = Tensor([0.0, 0.0])

  def test_setitem_raises_with_unrealized_downstream(self):
    x = Tensor([1.0, 2.0, 3.0, 4.0]).realize()
    _y = x * 2.0
    with self.assertRaises(RuntimeError):
      x[0] = 99.0

  def test_setitem_raises_on_unrealized_compute_base(self):
    # y has a compute (unrealized) base; tmp is a view of y. eager: tmp would follow y's mutation. lazy: tmp keeps the old MUL graph.
    x = Tensor([1.0, 2.0, 3.0, 4.0]).realize()
    y = x * 2.0
    _tmp = y[:1]
    with self.assertRaises(RuntimeError):
      y[0] = 99.0

  def test_setitem_raises_on_aliased_uop(self):
    # two Tensor objects sharing the exact same unrealized uop. setitem on one updates its uop, the other keeps the stale graph reference.
    x = Tensor([1.0, 2.0, 3.0, 4.0]).realize()
    y = x * 2.0
    _z = Tensor(y.uop)
    with self.assertRaises(RuntimeError):
      y[0] = 99.0

if __name__ == '__main__':
  unittest.main()
