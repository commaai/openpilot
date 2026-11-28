import unittest
from tinygrad import Tensor
from tinygrad.dtype import dtypes, Invalid

class TestAfterCounterexamples(unittest.TestCase):
  def test_ordered_writes_allowed(self):
    x = Tensor([0.]).realize().uop
    a = x.after(x.store(1))
    b = a.after(a.store(2))
    self.assertEqual(Tensor(b).tolist(), [2.])

  def test_older_dependency_preserves_selected_state(self):
    x = Tensor([0.]).realize().uop
    first = x.store(1)
    a = x.after(first)
    second = a.store((Tensor(a) + 1).contiguous().uop)
    self.assertEqual((Tensor(a.after(second).after(first)) + 0).tolist(), [2.])

  def test_sibling_afters_rejected(self):
    x = Tensor([0.]).realize().uop
    with self.assertRaises(RuntimeError):
      Tensor.realize(Tensor(x.after(x.store(1))) + 0, Tensor(x.after(x.store(2))) + 0)

  def test_write_into_newer_state_allowed(self):
    x = Tensor([0.]).realize().uop
    first = x.store(1)
    a = x.after(first)
    p, q = Tensor(x.after(a.store(2))) + 0, Tensor(a) + 0
    Tensor.realize(p, q)
    self.assertEqual((p.tolist(), q.tolist()), ([2.], [1.]))

  def test_shared_readers_after_older_dependency(self):
    x = Tensor([0.]).realize().uop
    first = x.store(1)
    a = x.after(first)
    second = a.store(2)
    y = a.after(second)
    p, q = Tensor(y) + 0, Tensor(y.after(first)) + 0
    Tensor.realize(p, q)
    self.assertEqual((p.tolist(), q.tolist()), ([2.], [2.]))

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
                           (lambda t: t.flip((0, 1))[:1], [[1., 1., 1.], [0., 0., 0.]]),
                           (lambda t: t[:, ::2], [[0., 1., 0.], [0., 1., 0.]])):
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

  def test_store_broadcasts_value_to_target(self):
    def stored(shape:tuple[int, ...], y:Tensor) -> list:
      x = Tensor.zeros(*shape, dtype=dtypes.int).contiguous().realize().uop
      return Tensor(x.after(x.store(y.contiguous().realize().uop))).tolist()
    self.assertEqual(stored((2, 3), Tensor([[3], [5]])), [[3, 3, 3], [5, 5, 5]])
    self.assertEqual(stored((2, 3), Tensor([1, 2, 3])), [[1, 2, 3], [1, 2, 3]])
    self.assertEqual(stored((3,), Tensor([[7, 8, 9]])), [7, 8, 9])
    with self.assertRaises(ValueError): stored((2, 1), Tensor([[1, 2, 3], [4, 5, 6]]))

class TestAfterInvalidStore(unittest.TestCase):
  def test_partial_invalid_store_keeps_uncovered_reads(self):
    x = Tensor([10., 20., 30., 40.]).realize()
    after = x.uop.after(x.uop.shrink(((0, 2),)).store(Invalid))
    self.assertEqual(Tensor(after).contiguous().tolist(), [10., 20., 30., 40.])

  def test_multi_after_invalid_store_dep_removed(self):
    x = Tensor.empty(4).uop
    self.assertEqual(Tensor(x.after(x.store(5), x.store(Invalid))).tolist(), [5]*4)

  def test_expand_view_invalid_assign_keeps_uncovered_reads(self):
    x = Tensor([[10., 11., 12., 13.], [20., 21., 22., 23.], [30., 31., 32., 33.], [40., 41., 42., 43.]]).realize()
    v = x[:1, :].expand(4, 4)
    v.assign(Tensor.invalids(4, 4, dtype=dtypes.float))
    self.assertEqual(v.contiguous().tolist(), [[10., 11., 12., 13.]]*4)

if __name__ == "__main__":
  unittest.main()
