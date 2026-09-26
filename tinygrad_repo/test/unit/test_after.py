import unittest
from tinygrad import Tensor

class TestAfterCounterexamples(unittest.TestCase):
  def test_ordered_writes_allowed(self):
    x = Tensor([0.]).realize().uop
    a = x.after(x.store(1))
    b = a.after(a.store(2))
    self.assertEqual(Tensor(b).tolist(), [2.])

  def test_disjoint_writes_allowed(self):
    x = Tensor([0., 0.]).realize().uop
    y = Tensor(x.after(x[:1].store(1), x[1:].store(2)))
    self.assertEqual(y.tolist(), [1., 2.])

  def test_read_modify_write_chain(self):
    x = Tensor([2.]).clone()
    x.assign(x + 1)
    x.assign(x * 2)
    self.assertEqual(x.tolist(), [6.])

  def test_overwrite_cuts_gradient(self):
    x = Tensor([2.])
    y = x.clone()
    y.assign(3)  # overwriting with a constant makes y independent of x
    self.assertEqual(y.sum().gradient(x)[0].tolist(), [0.])

  def test_shared_state_readers(self):
    x = Tensor([2.]).clone()
    x.assign(x + 1)
    a, b = x + 1, x * 2
    Tensor.realize(a, b)
    self.assertEqual(a.tolist(), [4.])
    self.assertEqual(b.tolist(), [6.])

  @unittest.expectedFailure
  def test_chained_square_assign_gradient(self):
    x = Tensor([2.0])
    y = x.clone()
    y.assign(y*y)
    y.assign(y*y)
    # y = x**4, so dy/dx = 4*x**3. Currently raises "cycle detected while indexing".
    self.assertEqual(y.sum().gradient(x)[0].tolist(), [32.])

  def test_partial_store_gradient(self):
    x = Tensor([2., 3.]).realize()
    y = Tensor(x.uop.after(x[:1].uop.store(4)))
    # y = [4, x[1]]; only the untouched element depends on x.
    self.assertEqual(y.sum().gradient(x)[0].tolist(), [0., 1.])

  def test_partial_store_source_gradient(self):
    x = Tensor([4.])
    y = Tensor([2., 3.]).realize()
    z = Tensor(y.uop.after(y[:1].uop.store(x.uop)))
    # x contributes once, not twice.
    self.assertEqual(z.sum().gradient(x)[0].tolist(), [1.])

  def test_unrelated_store_gradient(self):
    x = Tensor([2.]).realize()
    y = x.clone()
    z = Tensor(x.uop.after(y.uop.store(0)))
    # Zeroing y does not change x.
    self.assertEqual(z.sum().gradient(x)[0].tolist(), [1.])

  def test_after_dependency_gradient(self):
    x = Tensor([2., 3.])
    y = x.clone()
    y[:1].assign(0)
    # View assign is an AFTER on a partial STORE; only the untouched element depends on x.
    self.assertEqual(y.sum().gradient(x)[0].tolist(), [0., 1.])

  def test_view_assign_gradient(self):
    for view, expected in ((lambda t: t.reshape(3, 2)[1:], [[1., 1., 0.], [0., 0., 0.]]),
                           (lambda t: t.permute(1, 0)[1:], [[1., 0., 0.], [1., 0., 0.]]),
                           (lambda t: t.flip((0, 1))[:1], [[1., 1., 1.], [0., 0., 0.]])):
      with self.subTest(expected=expected):
        x = Tensor([[1., 2., 3.], [4., 5., 6.]])
        y = x.clone()
        v = Tensor.full(view(y).shape, 7.)
        view(y).assign(v)
        gx, gv = y.sum().gradient(x, v)
        self.assertEqual(gx.tolist(), expected)
        self.assertEqual(gv.tolist(), Tensor.ones(v.shape).tolist())

  @unittest.expectedFailure
  def test_unordered_overlapping_stores_rejected(self):
    x = Tensor([0.]).realize().uop
    # No ordering between the writes. Currently succeeds with [2.].
    with self.assertRaises(RuntimeError):
      Tensor(x.after(x.store(1), x.store(2))).realize()

  @unittest.expectedFailure
  def test_gradient_after_callify(self):
    x = Tensor([2.]).realize()
    y = x * 2
    y.callify()
    # Currently raises: "expected a CALL with unbound BUFFER outputs or a grad_fxn".
    self.assertEqual(y.sum().gradient(x)[0].tolist(), [2.])

if __name__ == "__main__":
  unittest.main()
