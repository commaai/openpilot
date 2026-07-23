import unittest
from tinygrad import Tensor, TinyJit, Variable, dtypes, Device
from tinygrad.helpers import Context
import numpy as np

class TestSetitem(unittest.TestCase):
  def test_simple_setitem(self):
    cases = (
      ((6,6), (slice(2,4), slice(3,5)), Tensor.ones(2,2)),
      ((6,6), (slice(2,4), slice(3,5)), Tensor([1.,2.])),
      ((6,6), (slice(2,4), slice(3,5)), 1.0),
      ((6,6), (3, 4), 1.0),
      ((6,6), (3, None, 4, None), 1.0),
      ((4,4,4,4), (Ellipsis, slice(1,3), slice(None)), Tensor(4.0)),
      ((4,4,4,4), (Ellipsis, slice(1,3)), 4),
      ((4,4,4,4), (2, slice(1,3), None, 1), 4),
      ((4,4,4,4), (slice(1,3), slice(None), slice(0,4,2)), 4),
      ((4,4,4,4), (slice(1,3), slice(None), slice(None), slice(0,3)), 4),
      ((6,6), (slice(1,5,2), slice(0,5,3)), 1.0),
      ((6,6), (slice(5,1,-2), slice(5,0,-3)), 1.0),
    )
    for shp, slc, val in cases:
      t = Tensor.zeros(shp).contiguous()
      t[slc] = val
      n = np.zeros(shp)
      n[slc] = val.numpy() if isinstance(val, Tensor) else val
      np.testing.assert_allclose(t.numpy(), n)

  def test_padded_setitem(self):
    t = Tensor.arange(10)
    t[4:1:-2] = 11
    self.assertListEqual(t.tolist(), [0, 1, 11, 3, 11, 5, 6, 7, 8, 9])

  def test_setitem_inplace_mul(self):
    t = Tensor.arange(10).clone().realize()
    t[:3] *= 10
    self.assertListEqual(t.tolist(), [0, 10, 20, 3, 4, 5, 6, 7, 8, 9])

  @unittest.skip("crashed in LLVM CI")
  def test_setitem_fancy_on_unrealized_view(self):
    # fancy indexing setitem on unrealized SHRINK view (triggered infinite loop in graph_rewrite)
    base = Tensor.arange(20, dtype=dtypes.float).reshape(4, 5).clone().realize()
    sub = base[1:3]
    flat = sub.reshape(sub.numel()).contiguous()
    idx = Tensor([0, 3, 7, 9])
    flat[idx] = Tensor([99, 98, 97, 96], dtype=dtypes.float)
    sub.assign(flat.reshape(2, 5))
    np.testing.assert_allclose(sub.numpy(), [[99, 6, 7, 98, 9], [10, 11, 97, 13, 96]])

  def test_setitem_dtype(self):
    for dt in (dtypes.int, dtypes.float, dtypes.bool):
      for v in (5., 5, True):
        t = Tensor.ones(6,6, dtype=dt).contiguous()
        t[1] = v
        self.assertEqual(t.dtype, dt)

  def test_setitem_dtype_mismatch(self):
    t = Tensor.zeros(6, dtype=dtypes.float).contiguous().realize()
    with self.assertRaises(RuntimeError): t[2:4] = Tensor([1, 2], dtype=dtypes.int)

  def test_setitem_chained_indexing(self):
    # N[i][j] must work the same as N[i, j]
    N1 = Tensor.zeros((3, 3)).contiguous().realize()
    N1[1, 2] = 5
    N2 = Tensor.zeros((3, 3)).contiguous().realize()
    N2[1][2] = 5
    np.testing.assert_equal(N1.numpy(), N2.numpy())

  def test_setitem_detach(self):
    # setitem on detached tensor should work
    t = Tensor.zeros((3, 3)).contiguous().realize()
    t.detach()[1, 2] = 5
    self.assertEqual(t[1, 2].item(), 5.0)

  def test_setitem_permute(self):
    # setitem on permuted tensor should modify original
    t = Tensor.zeros((2, 3)).contiguous().realize()
    t.T[1, 0] = 5  # t.T is (3, 2), so [1, 0] maps to t[0, 1]
    self.assertEqual(t[0, 1].item(), 5.0)

  def test_setitem_flip(self):
    # setitem on flipped tensor should modify original
    t = Tensor.zeros((3,)).contiguous().realize()
    t[::-1][0] = 5  # flip, then set first element (which is last in original)
    self.assertEqual(t[2].item(), 5.0)

  def test_setitem_inplace_operator(self):
    t = Tensor.arange(4).reshape(2, 2).contiguous()
    t[1] += 2
    np.testing.assert_allclose(t.numpy(), [[0, 1], [4, 5]])

    t = Tensor.arange(4).reshape(2, 2).contiguous()
    t[1] -= 1
    np.testing.assert_allclose(t.numpy(), [[0, 1], [1, 2]])

    t = Tensor.arange(4).reshape(2, 2).contiguous()
    t[1] *= 2
    np.testing.assert_allclose(t.numpy(), [[0, 1], [4, 6]])

    # NOTE: have to manually cast setitem target to least_upper_float for div
    t = Tensor.arange(4, dtype=dtypes.float).reshape(2, 2).contiguous()
    t[1] /= 2
    np.testing.assert_allclose(t.numpy(), [[0, 1], [1, 1.5]])

    t = Tensor.arange(4).reshape(2, 2).contiguous()
    t[1] **= 2
    np.testing.assert_allclose(t.numpy(), [[0, 1], [4, 9]])

    t = Tensor.arange(4).reshape(2, 2).contiguous()
    t[1] ^= 5
    np.testing.assert_allclose(t.numpy(), [[0, 1], [7, 6]])

  def test_setitem_consecutive_inplace_operator(self):
    t = Tensor.arange(4).reshape(2, 2).contiguous()
    t[1] += 2
    t[1] -= 1
    np.testing.assert_allclose(t.numpy(), [[0, 1], [3, 4]])

  def test_setitem_overlapping_indices(self):
    t = Tensor([1,2,3,4])
    # regular overlapping indices
    t[[1,1]] = Tensor([5,6])
    np.testing.assert_allclose(t.numpy(), [1,6,3,4])

    # overlapping indices with zero value overlapped
    t[[1,1]] = Tensor([0,1])
    np.testing.assert_allclose(t.numpy(), [1,1,3,4])

  def test_setitem_overlapping_indices_with_0(self):
    t = Tensor([1,2,3,4])
    t[[1,1]] = Tensor([1,0])
    np.testing.assert_allclose(t.numpy(), [1,0,3,4])

  def test_setitem_with_1_in_shape(self):
    t = Tensor([[1],[2],[3]])
    t[[0,0]] = Tensor([[1],[2]])
    np.testing.assert_allclose(t.numpy(), [[2],[2],[3]])

  def test_fancy_setitem(self):
    t = Tensor.zeros(6,6).contiguous()
    t[[1,2], [3,2]] = 3
    n = np.zeros((6,6))
    n[[1,2], [3,2]] = 3
    np.testing.assert_allclose(t.numpy(), n)

  def test_simple_jit_setitem(self):
    @TinyJit
    def f(t:Tensor, a:Tensor):
      t[2:4, 3:5] = a
      # NOTE: without return t or an explicit realize, it's lazy and not captured
      return t

    for i in range(1, 6):
      t = Tensor.zeros(6, 6).contiguous().realize()
      a = Tensor.full((2, 2), fill_value=i, dtype=dtypes.float).contiguous()
      f(t, a)

      n = np.zeros((6, 6))
      n[2:4, 3:5] = np.full((2, 2), i)
      np.testing.assert_allclose(t.numpy(), n)

  def test_jit_setitem_variable_offset(self):
    with Context(CHECK_OOB=0):
      @TinyJit
      def f(t:Tensor, a:Tensor, v:Variable):
        t.shrink(((v,v+1), None)).assign(a).realize()

      t = Tensor.zeros(6, 6).contiguous().realize()
      n = np.zeros((6, 6))

      for i in range(6):
        v = Variable("v", 0, 6).bind(i)
        a = Tensor.full((1, 6), fill_value=i+1, dtype=dtypes.float).contiguous()
        n[i, :] = i+1
        f(t, a, v)
        np.testing.assert_allclose(t.numpy(), n)
      np.testing.assert_allclose(t.numpy(), [[1,1,1,1,1,1],[2,2,2,2,2,2],[3,3,3,3,3,3],[4,4,4,4,4,4],[5,5,5,5,5,5],[6,6,6,6,6,6]])

  def test_setitem_overlapping_inplace1(self):
    t = Tensor([[3.0], [2.0], [1.0]]).contiguous()
    t[1:] = t[:-1]
    self.assertEqual(t.tolist(), [[3.0], [3.0], [2.0]])

  def test_setitem_overlapping_inplace2(self):
    t = Tensor([[3.0], [2.0], [1.0]]).contiguous()
    t[:-1] = t[1:]
    self.assertEqual(t.tolist(), [[2.0], [1.0], [1.0]])

  # TODO: WEBGPU pipeline validation error. this generates (1==gidx0)|(2==gidx0)|(3==gidx0)|(4==gidx0)|(5==gidx0) ...
  @unittest.skipIf(Device.DEFAULT == "WEBGPU", "WEBGPU pipeline validation error")
  def test_setitem_big(self):
    idx_size, val = 256, 4
    t = Tensor.arange(0, idx_size+1)
    idx = Tensor.arange(0, idx_size)
    t[idx] = val
    self.assertEqual(t.tolist(), [val]*idx_size+[idx_size])

  def test_setitem_advanced_indexing(self):
    # Example from https://numpy.org/doc/stable/user/basics.indexing.html#combining-advanced-and-basic-indexing
    t = Tensor.zeros(10,20,30,40,50, dtype=dtypes.int).contiguous()
    ind_1 = Tensor([5,3,7,8])
    ind_2 = Tensor([[[0],[1],[2]],[[3],[4],[5]]])
    v = Tensor.arange(2*3*4*10*30*50).reshape(2,3,4,10,30,50)
    t[:, ind_1, :, ind_2, :] = v
    n = np.zeros((10,20,30,40,50), dtype=np.int32)
    n[:, ind_1.numpy(), :, ind_2.numpy(), :] = v.numpy()
    np.testing.assert_equal(t.numpy(), n)

  def test_setitem_tensor_int_indexing(self):
    t = Tensor.zeros(4, 3, dtype=dtypes.int).contiguous()
    t[Tensor([0, 2]), 0] = Tensor([99, 88], dtype=dtypes.int)
    n = np.zeros((4, 3), dtype=np.int32)
    n[[0, 2], 0] = [99, 88]
    np.testing.assert_equal(t.numpy(), n)

  def test_setitem_tensor_slice_indexing(self):
    t = Tensor.zeros(4, 3, dtype=dtypes.int).contiguous()
    t[Tensor([0, 2]), :2] = Tensor([[10, 20], [30, 40]], dtype=dtypes.int)
    n = np.zeros((4, 3), dtype=np.int32)
    n[[0, 2], :2] = [[10, 20], [30, 40]]
    np.testing.assert_equal(t.numpy(), n)

  def test_setitem_2d_tensor_indexing(self):
    t = Tensor.zeros(2, dtype=dtypes.int).contiguous()
    index = Tensor([[0, 1], [1,0]])
    v = Tensor.arange(2*2).reshape(2, 2).contiguous()
    t[index] = v
    n = np.zeros((2,), dtype=np.int32)
    n[index.numpy()] = v.numpy()
    np.testing.assert_equal(t.numpy(), n)

  def test_setitem_swap_rows(self):
    t = Tensor.arange(6, dtype=dtypes.float).reshape(3, 2).clone().realize()
    tmp = t[0]
    t[0] = t[1]
    t[2] = tmp
    # NOTE: not [[2, 3], [2, 3], [0, 1]], same with eager
    np.testing.assert_allclose(t.numpy(), [[2, 3], [2, 3], [2, 3]])

    # eager version
    t = Tensor.arange(6, dtype=dtypes.float).reshape(3, 2).clone().realize()
    tmp = t[0].realize()
    t[0] = t[1].realize()
    t[2] = tmp.realize()
    np.testing.assert_allclose(t.numpy(), [[2, 3], [2, 3], [2, 3]])

  def test_lazy_sum_between_writes(self):
    # lazy sums should capture buffer state at the time they were created
    t = Tensor.zeros(6).contiguous().realize()
    s0 = t.sum()
    t[:3].assign(1.0)
    s1 = t.sum()
    t[3:].assign(2.0)
    s2 = t.sum()
    try:
      np.testing.assert_allclose([s0.item(), s1.item(), s2.item()], [0.0, 3.0, 9.0])
    except AssertionError:
      # TODO: broken now, lazy sums all see final buffer state
      np.testing.assert_allclose([s0.item(), s1.item(), s2.item()], [9.0, 9.0, 9.0])

    # eager version
    t = Tensor.zeros(6).contiguous().realize()
    s0 = t.sum().realize()
    t[:3].assign(1.0).realize()
    s1 = t.sum().realize()
    t[3:].assign(2.0).realize()
    s2 = t.sum().realize()
    np.testing.assert_allclose([s0.item(), s1.item(), s2.item()], [0.0, 3.0, 9.0])

  def test_cross_assign_independence(self):
    # when assigning to two tensors using computations from both,
    # both assigns should see the OLD values of both tensors
    a = Tensor.arange(4, dtype=dtypes.float).clone().realize()
    b = Tensor.arange(4, 8, dtype=dtypes.float).clone().realize()
    new_a = a + b    # [4, 6, 8, 10]
    new_b = a * 2    # [0, 2, 4, 6] -- should use OLD a
    a.assign(new_a)
    b.assign(new_b)
    np.testing.assert_allclose(a.numpy(), [4, 6, 8, 10])
    try:
      np.testing.assert_allclose(b.numpy(), [0, 2, 4, 6])
    except AssertionError:
      # TODO: broken now, new_b sees mutated a
      np.testing.assert_allclose(b.numpy(), [8, 12, 16, 20])

    # eager version
    a = Tensor.arange(4, dtype=dtypes.float).clone().realize()
    b = Tensor.arange(4, 8, dtype=dtypes.float).clone().realize()
    new_a = (a + b).realize()
    new_b = (a * 2).realize()
    a.assign(new_a).realize()
    b.assign(new_b).realize()
    np.testing.assert_allclose(a.numpy(), [4, 6, 8, 10])
    np.testing.assert_allclose(b.numpy(), [0, 2, 4, 6])

  def test_setitem_multiple_disjoint_on_invalid(self):
    z = Tensor.invalids(10, dtype="int").realize()
    z[2:5] = 2
    z[6:7] = 3
    z.realize()
    self.assertListEqual(z[2:5].tolist(), [2, 2, 2])
    self.assertListEqual(z[6:7].tolist(), [3])

class TestWithGrad(unittest.TestCase):
  def test_basic_setitem_works(self):
    z = Tensor.rand(8, 8)
    x = Tensor.rand(8)
    z[:3] = x

  def test_set_backward(self):
    z = Tensor.ones(8, 8)
    x = Tensor.rand(8, 8)
    z[:] = x
    z.sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), np.ones((8, 8)))

  def test_set_nonleaf_backward(self):
    x = Tensor([1.0, 2.0, 3.0, 4.0])
    z = x * 2
    z[:2] = Tensor([10.0, 20.0])
    z.sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), [0, 0, 2, 2])

  def test_set_overlapping_backward(self):
    z = Tensor.zeros(6)
    x = Tensor.ones(4).contiguous()
    y = Tensor.ones(4) * 2
    z[:4] = x
    z[2:] = y
    z.sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), [1, 1, 0, 0])
    np.testing.assert_allclose(y.grad.numpy(), np.ones(4))

  def test_set_iadd_backward(self):
    z = Tensor([1.0, 2.0, 3.0, 4.0])
    x = Tensor([10.0, 20.0])
    z[:2] += x
    z.sum().backward()
    np.testing.assert_allclose(z.grad.numpy(), np.ones(4))
    np.testing.assert_allclose(x.grad.numpy(), np.ones(2))

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

class TestSetitemLoop(unittest.TestCase):
  def test_arange(self):
    N = 10
    cmp = Tensor.empty(N)
    for i in range(N): cmp[i] = i
    self.assertListEqual(Tensor.arange(N).tolist(), cmp.tolist())

if __name__ == '__main__':
  unittest.main()
