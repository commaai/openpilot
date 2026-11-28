import unittest, operator
from tinygrad import Tensor, TinyJit, Variable, dtypes, Device, GlobalCounters
from test.helpers import assert_kernel_count
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
      ((6,6), (slice(None), slice(0,6,2)), 1.0),
      ((3,3), (slice(0,2,2), slice(0,3,3)), 1.0),
      ((4,), slice(2,0,-4), 1.0),
    )
    for shp, slc, val in cases:
      for realize in (False, True):
        t = Tensor.zeros(shp).contiguous()
        if realize: t.realize()
        t[slc] = val
        n = np.zeros(shp)
        n[slc] = val.numpy() if isinstance(val, Tensor) else val
        np.testing.assert_allclose(t.numpy(), n)

  def test_padded_setitem(self):
    t = Tensor.arange(10)
    t[4:1:-2] = 11
    self.assertListEqual(t.tolist(), [0, 1, 11, 3, 11, 5, 6, 7, 8, 9])

  def test_setitem_reshape_only_view(self):
    t = Tensor([[1.,2,3,4]]).realize()
    t[0] = 7
    self.assertListEqual(t.tolist(), [[7.,7,7,7]])

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

  def test_jit_setitem_after_assign(self):
    @TinyJit
    def f(t:Tensor):
      t += 1
      t[0] = 5
      return t

    for i in range(3):
      t = Tensor.full((3,), float(i)).contiguous().realize()
      f(t)
      self.assertEqual(t.tolist(), [5, i+1, i+1])

  def test_jit_setitem_variable_offset(self):
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

  def test_setitem_view_after_assign(self):
    t = Tensor([0.0, 0.0, 0.0, 0.0]).contiguous().realize()
    t += 1
    v = t[1:3]
    v[0] = 5
    self.assertEqual(t.tolist(), [1, 5, 1, 1])

  def test_setitem_overlapping_inplace1(self):
    t = Tensor([[3.0], [2.0], [1.0]]).contiguous()
    t[1:] = t[:-1]
    self.assertEqual(t.tolist(), [[3.0], [3.0], [2.0]])

  def test_setitem_overlapping_inplace2(self):
    t = Tensor([[3.0], [2.0], [1.0]]).contiguous()
    t[:-1] = t[1:]
    self.assertEqual(t.tolist(), [[2.0], [1.0], [1.0]])

  def test_setitem_older_state(self):
    t = Tensor([1, 2, 3, 4]).realize()
    old = Tensor(t.assign(t + 1).uop)
    t.assign(t + 10)
    t[:] = old
    # TODO: broken now, raises: should be [2, 3, 4, 5]
    with self.assertRaises(RuntimeError): t.tolist()

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
    t = Tensor.zeros(2,9,3,6,5, dtype=dtypes.int).contiguous()
    ind_1 = Tensor([5,3,7,8])
    ind_2 = Tensor([[[0],[1],[2]],[[3],[4],[5]]])
    v = Tensor.arange(2*3*4*2*3*5).reshape(2,3,4,2,3,5)
    t[:, ind_1, :, ind_2, :] = v
    n = np.zeros((2,9,3,6,5), dtype=np.int32)
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

  def test_setitem_tensor_indexing_broadcast(self):
    t = Tensor.zeros(3, dtype=dtypes.int).contiguous()
    t[Tensor([0, 1, 2])] = Tensor([[7, 8, 9]], dtype=dtypes.int)
    n = np.zeros((3,), dtype=np.int32)
    n[[0, 1, 2]] = [[7, 8, 9]]
    np.testing.assert_equal(t.numpy(), n)
    # v broadcasts to the indexed shape, not the other way
    with self.assertRaises(ValueError): t[Tensor([0])] = Tensor([1, 2], dtype=dtypes.int)

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
    np.testing.assert_allclose([s0.item(), s1.item(), s2.item()], [0.0, 3.0, 9.0])

  def test_setitem_multiple_disjoint_on_invalid(self):
    z = Tensor.invalids(10, dtype="int").realize()
    z[2:5] = 2
    z[6:7] = 3
    z.realize()
    self.assertListEqual(z[2:5].tolist(), [2, 2, 2])
    self.assertListEqual(z[6:7].tolist(), [3])

class TestAssignBitcast(unittest.TestCase):
  def test_assign_through_bitcast(self):
    # the dest is unrealized, so the call transform cannot fold the BITCAST into a buffer view and the STORE keeps a
    # BITCAST dest; the bitcast has to move to the value side or the store never reaches the buffer
    a = Tensor.full((4,), 1.0, dtype=dtypes.float32).contiguous()
    a.bitcast(dtypes.uint32).assign(Tensor([0x40800000, 0x40400000, 0x40000000, 0x3f800000], dtype=dtypes.uint32)).realize()
    np.testing.assert_allclose(a.numpy(), [4.0, 3.0, 2.0, 1.0])

  @unittest.expectedFailure
  def test_setitem_through_bitcast(self):
    # TODO: the view of a bitcast is not storage, so the functional path replaces the temporary and a is never written
    a = Tensor.full((4,), 1.0, dtype=dtypes.float32).contiguous().realize()
    a.bitcast(dtypes.uint32)[1] = 0x40000000
    np.testing.assert_allclose(a.numpy(), [1.0, 2.0, 1.0, 1.0])

class TestWithGrad(unittest.TestCase):
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

  def test_unrealized_inplace_keeps_storage(self):
    x = Tensor([1., 2.]).clone()
    view = x[:1]
    x += 3
    x.realize()
    self.assertEqual(x.tolist(), [4., 5.])
    self.assertEqual(view.tolist(), [4.])

  def test_unrealized_view_inplace_keeps_storage(self):
    x = Tensor([1., 2.]).clone()
    view = x[:1]
    view += 3
    view.realize()
    self.assertEqual(x.tolist(), [4., 2.])
    self.assertEqual(view.tolist(), [4.])

  def test_set_augmented_backward(self):
    for op, expected in ((operator.isub, [-1., -1.]), (operator.imul, [1., 2.]), (operator.itruediv, [-0.01, -0.005])):
      with self.subTest(op=op.__name__):
        z = Tensor([1.0, 2.0, 3.0, 4.0])
        x = Tensor([10.0, 20.0])
        z[:2] = op(z[:2], x)
        z.sum().backward()
        np.testing.assert_allclose(z.grad.numpy(), np.ones(4))
        np.testing.assert_allclose(x.grad.numpy(), expected)

class TestSetitemLoop(unittest.TestCase):
  def test_arange(self):
    N = 10
    cmp = Tensor.empty(N)
    for i in range(N): cmp[i] = i
    self.assertListEqual(Tensor.arange(N).tolist(), cmp.tolist())

class TestSetitemInto(unittest.TestCase):
  def test_setitem_into_unrealized(self):
    GlobalCounters.reset()
    t = Tensor.arange(4, dtype=dtypes.int32).reshape(2, 2)
    assert_kernel_count(0)
    t[1] = 5
    assert_kernel_count(0)
    t.realize()
    assert_kernel_count(0)
    self.assertEqual(GlobalCounters.global_mem, 0)
    self.assertListEqual(t.tolist(), [[0, 1], [5, 5]])

  def test_setitem_into_unrealized_sliced_compute(self):
    # base computation contains SHRINK from prior slicing (like QR decomposition pattern)
    GlobalCounters.reset()
    a = Tensor.arange(8, dtype=dtypes.int32).reshape(2, 4)
    w = a[0] + a[1]  # unrealized ADD with SHRINK in graph: [4, 6, 8, 10]
    assert_kernel_count(0)
    w[1] = 99
    assert_kernel_count(0)
    w.realize()
    assert_kernel_count(0)
    self.assertEqual(GlobalCounters.global_mem, 0)
    self.assertListEqual(w.tolist(), [4, 99, 8, 10])

  def test_setitem_into_empty(self):
    GlobalCounters.reset()
    t = Tensor.empty(4, dtype=dtypes.int32)
    t[1] = 5
    assert_kernel_count(0)
    t.realize()
    assert_kernel_count(1)
    self.assertEqual(GlobalCounters.global_mem, 4)
    t[1].realize()
    t.realize()
    assert_kernel_count(1)
    self.assertEqual(t[1].item(), 5)

  def test_setitem_into_tensor(self):
    t = Tensor([1, 2, 3, 4], dtype=dtypes.int32).realize()
    GlobalCounters.reset()
    t[1] = 5
    assert_kernel_count(0)
    t[1].realize()
    assert_kernel_count(1)
    self.assertEqual(GlobalCounters.global_mem, 4)
    t.realize()
    assert_kernel_count(1)
    self.assertListEqual(t.tolist(), [1, 5, 3, 4])

  def test_setitem_into_const(self):
    GlobalCounters.reset()
    t = Tensor.ones(4, dtype=dtypes.int32, buffer=False)
    t[1] = 5
    assert_kernel_count(0)
    t.realize()
    assert_kernel_count(0)
    self.assertEqual(GlobalCounters.global_mem, 0)
    self.assertListEqual(t.tolist(), [1, 5, 1, 1])

  @unittest.skipUnless(Device.DEFAULT != "CPU", "source must be on another device")
  def test_setitem_slice_assign_from_other_device(self):
    # Copy and assign; CL and WEBGPU also need a kernel to materialize the source slice (no buffer views).
    a = Tensor.ones(20, device="CPU")
    b = Tensor.arange(20).float().clone()
    Tensor.realize(a, b)
    GlobalCounters.reset()
    a[10:12].assign(b[13:15].to(a.device)).realize()
    assert_kernel_count(3 if Device.DEFAULT in {"CL", "WEBGPU"} else 2)
    self.assertListEqual(a.tolist(), [1.0]*10 + [13.0, 14.0] + [1.0]*8)

if __name__ == '__main__':
  unittest.main()
