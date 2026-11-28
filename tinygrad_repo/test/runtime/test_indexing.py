# test cases are modified from pytorch test_indexing.py https://github.com/pytorch/pytorch/blob/597d3fb86a2f3b8d6d8ee067e769624dcca31cdb/test/test_indexing.py

import unittest, random
import numpy as np

from tinygrad import Tensor, dtypes, Device, Variable, GlobalCounters
from tinygrad.helpers import all_same, prod

random.seed(42)

def numpy_testing_assert_equal_helper(a, b):
  if isinstance(a, Tensor): a = a.numpy()
  if isinstance(b, Tensor): b = b.numpy()
  np.testing.assert_equal(a, b)

def consec(shape, start=1):
  return Tensor.arange(prod(shape)).reshape(shape)+start

# creates strided tensor with base set to reference tensor's base, equivalent to torch.set_()
def set_(reference: Tensor, shape, strides, offset):
  raise NotImplementedError("need to implement without calling uop.view")
  """
  if reference.uop.base.realized is None: reference.realize()
  assert reference.uop.base.realized, "base has to be realized before setting it to strided's base"
  strided = Tensor(reference.uop.view(ShapeTracker((View.create(shape=shape, strides=strides, offset=offset),))))
  assert strided.uop.st.real_strides() == strides, "real_strides should equal strides for strided"
  return strided
  """

def clone(original:Tensor): return original.clone()
def copy_(src:Tensor, other:Tensor) -> Tensor: return src.clone()
def argsort(tensor:Tensor) -> Tensor:
  pass

# https://pytorch.org/docs/stable/generated/torch.all.html
def all_(tensor:Tensor) -> Tensor:
  return tensor != 0

# https://pytorch.org/docs/stable/generated/torch.diagonal.html
def diagonal(tensor:Tensor) -> Tensor:
  assert tensor.ndim == 2 and all_same(tensor.shape), 'only support 2 ndim square tensors'
  return (Tensor.eye(tensor.shape[0]) * tensor).sum(0)

# https://numpy.org/doc/stable/reference/generated/numpy.unravel_index.html
def unravel_index(tensor, shape):
  pass

class TestIndexing(unittest.TestCase):
  def test_index(self):

    reference = consec((3, 3, 3))

    numpy_testing_assert_equal_helper(reference[0], consec((3, 3)))
    numpy_testing_assert_equal_helper(reference[1], consec((3, 3), 10))
    numpy_testing_assert_equal_helper(reference[2], consec((3, 3), 19))
    numpy_testing_assert_equal_helper(reference[0, 1], consec((3,), 4))
    numpy_testing_assert_equal_helper(reference[0:2], consec((2, 3, 3)))
    numpy_testing_assert_equal_helper(reference[2, 2, 2], 27)
    numpy_testing_assert_equal_helper(reference[:], consec((3, 3, 3)))

    # indexing with Ellipsis
    numpy_testing_assert_equal_helper(reference[..., 2], np.array([[3., 6., 9.],[12., 15., 18.],[21., 24., 27.]]))
    numpy_testing_assert_equal_helper(reference[0, ..., 2], np.array([3., 6., 9.]))
    numpy_testing_assert_equal_helper(reference[..., 2], reference[:, :, 2])
    numpy_testing_assert_equal_helper(reference[0, ..., 2], reference[0, :, 2])
    numpy_testing_assert_equal_helper(reference[0, 2, ...], reference[0, 2])
    numpy_testing_assert_equal_helper(reference[..., 2, 2, 2], 27)
    numpy_testing_assert_equal_helper(reference[2, ..., 2, 2], 27)
    numpy_testing_assert_equal_helper(reference[2, 2, ..., 2], 27)
    numpy_testing_assert_equal_helper(reference[2, 2, 2, ...], 27)
    numpy_testing_assert_equal_helper(reference[...], reference)

    reference_5d = consec((3, 3, 3, 3, 3))
    numpy_testing_assert_equal_helper(reference_5d[..., 1, 0], reference_5d[:, :, :, 1, 0])
    numpy_testing_assert_equal_helper(reference_5d[2, ..., 1, 0], reference_5d[2, :, :, 1, 0])
    numpy_testing_assert_equal_helper(reference_5d[2, 1, 0, ..., 1], reference_5d[2, 1, 0, :, 1])
    numpy_testing_assert_equal_helper(reference_5d[...], reference_5d)

    # None indexing
    numpy_testing_assert_equal_helper(reference[2, None], reference[2].unsqueeze(0))
    numpy_testing_assert_equal_helper(reference[2, None, None], reference[2].unsqueeze(0).unsqueeze(0))
    numpy_testing_assert_equal_helper(reference[2:4, None], reference[2:4].unsqueeze(1))
    numpy_testing_assert_equal_helper(reference[None, 2, None, None], reference.unsqueeze(0)[:, 2].unsqueeze(0).unsqueeze(0))
    numpy_testing_assert_equal_helper(reference[None, 2:5, None, None], reference.unsqueeze(0)[:, 2:5].unsqueeze(2).unsqueeze(2))

    # indexing 0-length slice
    numpy_testing_assert_equal_helper(np.empty((0, 3, 3)), reference[slice(0)])
    numpy_testing_assert_equal_helper(np.empty((0, 3)), reference[slice(0), 2])
    numpy_testing_assert_equal_helper(np.empty((0, 3)), reference[2, slice(0)])
    numpy_testing_assert_equal_helper(np.empty([]), reference[2, 1:1, 2])

    # indexing with step
    reference = consec((10, 10, 10))
    numpy_testing_assert_equal_helper(reference[1:5:2], Tensor.stack(reference[1], reference[3], dim=0))
    numpy_testing_assert_equal_helper(reference[1:6:2], Tensor.stack(reference[1], reference[3], reference[5], dim=0))
    numpy_testing_assert_equal_helper(reference[1:9:4], Tensor.stack(reference[1], reference[5], dim=0))
    numpy_testing_assert_equal_helper(reference[2:4, 1:5:2], Tensor.stack(reference[2:4, 1], reference[2:4, 3], dim=1))
    numpy_testing_assert_equal_helper(reference[3, 1:6:2], Tensor.stack(reference[3, 1], reference[3, 3], reference[3, 5], dim=0))
    numpy_testing_assert_equal_helper(reference[None, 2, 1:9:4], Tensor.stack(reference[2, 1], reference[2, 5], dim=0).unsqueeze(0))
    numpy_testing_assert_equal_helper(reference[:, 2, 1:6:2], Tensor.stack(reference[:, 2, 1], reference[:, 2, 3], reference[:, 2, 5], dim=1))

    lst = [list(range(i, i+10)) for i in range(0, 100, 10)]
    tensor = Tensor(lst)
    for _ in range(100):
      idx1_start = random.randrange(10)
      idx1_end = idx1_start + random.randrange(1, 10 - idx1_start + 1)
      idx1_step = random.randrange(1, 8)
      idx1 = slice(idx1_start, idx1_end, idx1_step)
      if random.randrange(2) == 0:
        idx2_start = random.randrange(10)
        idx2_end = idx2_start + random.randrange(1, 10 - idx2_start + 1)
        idx2_step = random.randrange(1, 8)
        idx2 = slice(idx2_start, idx2_end, idx2_step)
        lst_indexed = [l[idx2] for l in lst[idx1]]
        tensor_indexed = tensor[idx1, idx2]
      else:
        lst_indexed = lst[idx1]
        tensor_indexed = tensor[idx1]
      numpy_testing_assert_equal_helper(tensor_indexed, np.array(lst_indexed))

    self.assertRaises(ValueError, lambda: reference[1:9:0])
    # NOTE torch doesn't support this but numpy does so we should too. Torch raises ValueError
    # see test_slice_negative_strides in test_ops.py
    # self.assertRaises(ValueError, lambda: reference[1:9:-1])

    self.assertRaises(IndexError, lambda: reference[1, 1, 1, 1])
    self.assertRaises(IndexError, lambda: reference[1, 1, 1, 1:1])
    self.assertRaises(IndexError, lambda: reference[3, 3, 3, 3, 3, 3, 3, 3])

    self.assertRaises(IndexError, lambda: reference[0.0])
    self.assertRaises(TypeError, lambda: reference[0.0:2.0])
    self.assertRaises(IndexError, lambda: reference[0.0, 0.0:2.0])
    self.assertRaises(IndexError, lambda: reference[0.0, :, 0.0:2.0])
    self.assertRaises(IndexError, lambda: reference[0.0, ..., 0.0:2.0])
    self.assertRaises(IndexError, lambda: reference[0.0, :, 0.0])

    def delitem(): del reference[0]
    self.assertRaises(TypeError, delitem)

  def test_set_item_to_scalar_tensor(self):
    m = random.randint(1, 10)
    n = random.randint(1, 10)
    z = Tensor.randn([m, n])
    a = 1.0
    w = Tensor([a])
    z[:, 0] = w
    z.sum().backward()
    numpy_testing_assert_equal_helper(w.grad, m * a)

  def test_step(self):
    v = Tensor.arange(10)
    numpy_testing_assert_equal_helper(v[::1], v)
    numpy_testing_assert_equal_helper(v[::2], [0, 2, 4, 6, 8])
    numpy_testing_assert_equal_helper(v[::3], [0, 3, 6, 9])
    numpy_testing_assert_equal_helper(v[::11], [0])
    numpy_testing_assert_equal_helper(v[1:6:2], [1, 3, 5])

  # TODO setitem
  # NOTE: tinygrad doesn't support idx.max that big
  '''
  def test_index_put_accumulate_large_tensor(self):
    # This test is for tensors with number of elements >= INT_MAX (2^31 - 1).
    N = (1 << 31) + 5
    dt = dtypes.int8
    a = Tensor.ones(N, dtype=dt).contiguous()
    indices = Tensor([-2, 0, -2, -1, 0, -1, 1], dtype=dtypes.int64)
    values = Tensor([6, 5, 6, 6, 5, 7, 11], dtype=dt)

    index_put_(a, (indices, ), values, accumulate=True)

    numpy_testing_assert_equal_helper(a[0], 11)
    numpy_testing_assert_equal_helper(a[1], 12)
    numpy_testing_assert_equal_helper(a[2], 1)
    numpy_testing_assert_equal_helper(a[-3], 1)
    numpy_testing_assert_equal_helper(a[-2], 13)
    numpy_testing_assert_equal_helper(a[-1], 14)

    a = Tensor.ones((2, N), dtype=dt).contiguous()
    indices0 = np.array([0, -1, 0, 1], dtype=dtypes.int64)
    indices1 = np.array([-2, -1, 0, 1], dtype=dtypes.int64)
    values = np.array([12, 13, 10, 11], dtype=dt)

    index_put_(a, (indices0, indices1), values, accumulate=True)

    numpy_testing_assert_equal_helper(a[0, 0], 11)
    numpy_testing_assert_equal_helper(a[0, 1], 1)
    numpy_testing_assert_equal_helper(a[1, 0], 1)
    numpy_testing_assert_equal_helper(a[1, 1], 12)
    numpy_testing_assert_equal_helper(a[:, 2], Tensor.ones(2, dtype=dtypes.int8))
    numpy_testing_assert_equal_helper(a[:, -3], Tensor.ones(2, dtype=dtypes.int8))
    numpy_testing_assert_equal_helper(a[0, -2], 13)
    numpy_testing_assert_equal_helper(a[1, -2], 1)
    numpy_testing_assert_equal_helper(a[-1, -1], 14)
    numpy_testing_assert_equal_helper(a[0, -1], 1)
  '''

  # TODO fancy setitem
  '''
  def test_index_put_accumulate_duplicate_indices(self):
    for i in range(1, 512):
      # generate indices by random walk, this will create indices with
      # lots of duplicates interleaved with each other
      delta = Tensor.uniform(low=-1, high=1, dtype=dtypes.double)
      indices = delta.cumsum(0).cast(dtypes.int64)

      # input = torch.randn(indices.abs().max() + 1)
      input = Tensor.randn(indices.abs().max().item() + 1)
      # values = torch.randn(indices.size(0))
      values = Tensor.randn(indices.shape(0))
      output = index_put_(input, (indices,), values, accumulate=True)

      input_list = input.numpy().tolist()
      indices_list = indices.numpy().tolist()
      values_list = values.numpy().tolist()
      for i, v in zip(indices_list, values_list):
        input_list[i] += v

      numpy_testing_assert_equal_helper(output, input_list)
  '''

  # TODO empty setitem
  '''
  def test_index_put_accumulate_empty(self):
    # Regression test for https://github.com/pytorch/pytorch/issues/94667
    input = Tensor.rand([], dtype=dtypes.float32)
    with self.assertRaises(RuntimeError):
      index_put_(input, [], np.array([1.0]), True)
  '''

  @unittest.skipIf(Device.DEFAULT == "WEBGPU", "WEBGPU doesn't support long indexing: #13624")
  def test_index_ind_dtype(self):
    x = Tensor.randn(4, 4)
    # ind_long = torch.randint(4, (4,), dtype=torch.long)
    # TODO should we spend an extra line to allow for randint other dtypes?
    # copied from randint
    ind_long = (Tensor.rand((4,),)*(4-0)+0).cast(dtypes.int64)
    # ind_int = ind_long.int()
    ind_int = (ind_long).cast(dtypes.int32)
    ref = x[ind_long, ind_long]
    res = x[ind_int, ind_int]
    numpy_testing_assert_equal_helper(ref, res)
    ref = x[ind_long, :]
    res = x[ind_int, :]
    numpy_testing_assert_equal_helper(ref, res)
    ref = x[:, ind_long]
    res = x[:, ind_int]
    numpy_testing_assert_equal_helper(ref, res)
    # no repeating indices for index_put
    # TODO fancy setitem
    '''
    src = Tensor.randn(4)
    ind_long = Tensor.arange(4, dtype=dtypes.int64)
    ind_int = ind_long.cast(dtypes.int32)
    for accum in (True, False):
      inp_ref = clone(x)
      inp_res = clone(x)
      index_put_(inp_ref, (ind_long, ind_long), src, accum)
      index_put_(inp_res, (ind_int, ind_int), src, accum)
      numpy_testing_assert_equal_helper(inp_ref, inp_res)
    '''

  def test_int_indices2d(self):
    # From the NumPy indexing example
    x = Tensor.arange(0, 12).reshape(4, 3)
    rows = Tensor([[0, 0], [3, 3]])
    columns = Tensor([[0, 2], [0, 2]])
    numpy_testing_assert_equal_helper(x[rows, columns].numpy().tolist(), [[0, 2], [9, 11]])

  def test_getitem_scalars(self):
    zero = Tensor(0, dtype=dtypes.int64)
    one = Tensor(1, dtype=dtypes.int64)

    # non-scalar indexed with scalars
    a = Tensor.randn(2, 3).realize()
    numpy_testing_assert_equal_helper(a[0], a[zero])
    numpy_testing_assert_equal_helper(a[0][1], a[zero][one])
    numpy_testing_assert_equal_helper(a[0, 1], a[zero, one])
    numpy_testing_assert_equal_helper(a[0, one], a[zero, 1])

    # indexing by a scalar should slice (not copy)
    numpy_testing_assert_equal_helper(a[0, 1], a[zero, one])
    numpy_testing_assert_equal_helper(a[1], a[one.cast(dtypes.int32)])
    numpy_testing_assert_equal_helper(a[1], a[one.cast(dtypes.int16)])

    # scalar indexed with scalar
    r = Tensor.randn().realize()
    with self.assertRaises(IndexError):
      r[:]
    with self.assertRaises(IndexError):
      r[zero]
    numpy_testing_assert_equal_helper(r, r[...])

  # TODO fancy setitem
  '''
  def test_setitem_scalars(self):
    zero = Tensor(0, dtype=dtypes.int64)

    # non-scalar indexed with scalars
    a = Tensor.randn(2, 3).contiguous()
    a_set_with_number = clone(a).contiguous()
    a_set_with_scalar = clone(a).contiguous()
    b = Tensor.randn(3)

    a_set_with_number[0] = b
    a_set_with_scalar[zero] = b
    numpy_testing_assert_equal_helper(a_set_with_number, a_set_with_scalar)
    a[1, zero] = 7.7
    # TODO: weird inaccuracy Max relative difference: 2.47707621e-08
    # numpy_testing_assert_equal_helper(7.7, a[1, 0])
    np.testing.assert_allclose(7.7, a[1, 0].numpy(), rtol=1e-7)

    # scalar indexed with scalars
    r = Tensor.randn().contiguous()
    with self.assertRaises(IndexError):
      r[:] = 8.8
    with self.assertRaises(IndexError):
      r[zero] = 8.8
    r[...] = 9.9
    # TODO: weird inaccuracy Max relative difference: 3.85322971e-08
    # numpy_testing_assert_equal_helper(9.9, r)
    np.testing.assert_allclose(9.9, r, rtol=1e-7)
  '''

  # TODO fancy setitem
  '''
  def test_byte_tensor_assignment(self):
    x = Tensor.arange(0., 16).reshape(4, 4)
    b = Tensor([True, False, True, False], dtype=dtypes.uint8)
    value = Tensor([3., 4., 5., 6.])

    with warnings.catch_warnings(record=True) as w:
      x[b] = value
      numpy_testing_assert_equal_helper(len(w), 1)

    numpy_testing_assert_equal_helper(x[0], value)
    numpy_testing_assert_equal_helper(x[1], Tensor.arange(4., 8))
    numpy_testing_assert_equal_helper(x[2], value)
    numpy_testing_assert_equal_helper(x[3], Tensor.arange(12., 16))
  '''

  def test_variable_with_tensor_index(self):
    t = Tensor.arange(12).reshape(3, 4)
    v = Variable("v", 0, 2).bind(1)
    numpy_testing_assert_equal_helper(t[v, Tensor([0, 1, 2])], t[1, Tensor([0, 1, 2])])

  def test_variable_negative_index(self):
    self.assertEqual([Tensor([9, 8, 7])[Variable("i", -1, 1).bind(i)].item() for i in (-1, 0, 1)], [7, 9, 8])

  # TODO unravel_index
  '''
  def test_unravel_index_errors(self):
    with self.assertRaises(TypeError):
      unravel_index(
        Tensor(0.5),
        (2, 2))

    with self.assertRaises(TypeError):
      unravel_index(
        Tensor([]),
        (10, 3, 5))

    with self.assertRaises(TypeError):
      unravel_index(
        Tensor([1], dtype=dtypes.int64),
        Tensor([1, 2, 3]))

    with self.assertRaises(TypeError):
      unravel_index(
        Tensor([1], dtype=dtypes.int64),
        (1, 2, 2.0))

    with self.assertRaises(ValueError):
      unravel_index(
        Tensor(0),
        (2, -3))
  '''

  def test_zero_dim_index(self):
    x = Tensor(10)
    numpy_testing_assert_equal_helper(x, x.item())

    def runner():
      print(x[0])
      return x[0]

    self.assertRaises(IndexError, runner)

  # TODO fancy setitem
  '''
  def test_cpu_indices(self):
    idx = Tensor([0, 1])
    b = Tensor.zeros(2)
    x = Tensor.ones(10).contiguous()
    x[idx] = b  # index_put_
    ref = Tensor.ones(10).contiguous()
    ref[:2] = 0
    numpy_testing_assert_equal_helper(x, ref)
    out = x[idx]  # index
    numpy_testing_assert_equal_helper(out, Tensor.zeros(2))
  '''

class TestNumpy(unittest.TestCase):
  def test_empty_tuple_index(self):
    # Empty tuple index creates a view
    a = Tensor([1, 2, 3])
    numpy_testing_assert_equal_helper(a[()], a)
    #self.assertEqual(data_ptr(a[()]), data_ptr(a))

  def test_ellipsis_index(self):
    a = Tensor([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
    self.assertIs(a[...], a)
    numpy_testing_assert_equal_helper(a[...], a)
    # `a[...]` was `a` in numpy <1.9.
    #numpy_testing_assert_equal_helper(data_ptr(a[...]), data_ptr(a))

    # Slicing with ellipsis can skip an
    # arbitrary number of dimensions
    numpy_testing_assert_equal_helper(a[0, ...], a[0])
    numpy_testing_assert_equal_helper(a[0, ...], a[0, :])
    numpy_testing_assert_equal_helper(a[..., 0], a[:, 0])

    # In NumPy, slicing with ellipsis results in a 0-dim array. In PyTorch
    # we don't have separate 0-dim arrays and scalars.
    numpy_testing_assert_equal_helper(a[0, ..., 1], np.array(2))

    # Assignment with `(Ellipsis,)` on 0-d arrays
    b = np.array(1)
    b[(Ellipsis,)] = 2
    numpy_testing_assert_equal_helper(b, 2)

  def test_single_int_index(self):
    # Single integer index selects one row
    a = Tensor([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

    numpy_testing_assert_equal_helper(a[0], [1, 2, 3])
    numpy_testing_assert_equal_helper(a[-1], [7, 8, 9])

    self.assertRaises(IndexError, a.__getitem__, 1 << 30)
    self.assertRaises(IndexError, a.__getitem__, 1 << 64)

  # TODO out of bound getitem does not raise error
  '''
  def test_trivial_fancy_out_of_bounds(self):
    a = Tensor.zeros(5)
    ind = Tensor.ones(20, dtype=dtypes.int64)
    ind[-1] = 10
    self.assertRaises(IndexError, a.__getitem__, ind)
    self.assertRaises(IndexError, a.__setitem__, ind, 0)
    ind = Tensor.ones(20, dtype=dtypes.int64)
    ind[0] = 11
    self.assertRaises(IndexError, a.__getitem__, ind)
    self.assertRaises(IndexError, a.__setitem__, ind, 0)
  '''

  # TODO fancy setitem
  '''
  def test_index_is_larger(self):
    # Simple case of fancy index broadcasting of the index.
    a = Tensor.zeros((5, 5))
    a[[[0], [1], [2]], [0, 1, 2]] = Tensor([2., 3., 4.])

    self.assertTrue((a[:3, :3] == all_(Tensor([2., 3., 4.]))))
  '''

  # TODO fancy setitem
  '''
  def test_broadcast_subspace(self):
    a = Tensor.zeros((100, 100))
    v = Tensor.arange(0., 100)[:, None]
    b = Tensor.arange(99, -1, -1).cast(dtypes.int64)
    a[b] = v
    expected = b.float().unsqueeze(1).expand(100, 100)
    numpy_testing_assert_equal_helper(a, expected)
  '''

  # TODO fancy setitem
  '''
  def test_truncate_leading_1s(self):
    col_max = Tensor.randn(1, 4)
    kernel = col_max.T * col_max  # [4, 4] tensor
    kernel2 = clone(kernel)
    # Set the diagonal
    # len(torch.tensor) is just tensor.shape[0]
    kernel[range(kernel.shape[0]), range(kernel.shape[0])] = col_max.square()
    kernel2 = diagonal(kernel2)
    # torch.diagonal(kernel2).copy_(torch.square(col_max.view(4)))
    kernel2 = copy_(kernel2, col_max.reshape(4).square())
    numpy_testing_assert_equal_helper(kernel, kernel2)
  '''

class TestAdvancedIndexing(unittest.TestCase):
  def test_integer_array_indexing(self):
    x = Tensor.arange(5)
    x[[0, 2, 4]] = Tensor([5, 4, 3])
    numpy_testing_assert_equal_helper(x, [5, 1, 4, 3, 3])

  def test_negative_list_setitem(self):
    x = Tensor.arange(6).reshape(2, 3).clone()
    expected = np.arange(6).reshape(2, 3)
    x[[0, 1], [-1]] = Tensor([7, 8])
    expected[[0, 1], [-1]] = [7, 8]
    numpy_testing_assert_equal_helper(x, expected)

class TestGetitemOps(unittest.TestCase):
  def test_two_tensor_indices(self):
    # linear indexing is O(idx_size), one-hot masks is O(idx_size * src_size)
    src_np = np.random.rand(10, 100, 200).astype(np.float32)
    idx1_np, idx2_np = np.random.randint(0, 100, (50, 60), dtype=np.int32), np.random.randint(0, 200, (50, 60), dtype=np.int32)
    src, idx1, idx2 = Tensor(src_np), Tensor(idx1_np), Tensor(idx2_np)
    # O(50*60) = 3K vs O(50*60*100*200) = 60M
    GlobalCounters.reset()
    np.testing.assert_allclose(src_np[0, idx1_np, idx2_np], src[0, idx1, idx2].numpy())
    self.assertLess(GlobalCounters.global_ops, 100_000)
    # consecutive indices not starting from dim 0: O(10*50*60) = 30K vs O(10*50*60*100*200) = 600M
    GlobalCounters.reset()
    np.testing.assert_allclose(src_np[:, idx1_np, idx2_np], src[:, idx1, idx2].numpy())
    self.assertLess(GlobalCounters.global_ops, 1_000_000)

if __name__ == '__main__':
  unittest.main()
