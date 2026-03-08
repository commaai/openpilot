# test cases are modified from pytorch test_indexing.py

import unittest
import numpy as np

from tinygrad import Tensor, dtypes

def numpy_testing_assert_equal_helper(a, b):
  if isinstance(a, Tensor): a = a.numpy()
  if isinstance(b, Tensor): b = b.numpy()
  np.testing.assert_equal(a, b)

class TestIndexing(unittest.TestCase):
  def test_single_int(self):
    v = Tensor.randn(5, 7, 3)
    numpy_testing_assert_equal_helper(v[4].shape, (7, 3))

  def test_multiple_int(self):
    v = Tensor.randn(5, 7, 3)
    numpy_testing_assert_equal_helper(v[4].shape, (7, 3))
    numpy_testing_assert_equal_helper(v[4, :, 1].shape, (7,))

  def test_none(self):
    v = Tensor.randn(5, 7, 3)
    numpy_testing_assert_equal_helper(v[None].shape, (1, 5, 7, 3))
    numpy_testing_assert_equal_helper(v[:, None].shape, (5, 1, 7, 3))
    numpy_testing_assert_equal_helper(v[:, None, None].shape, (5, 1, 1, 7, 3))
    numpy_testing_assert_equal_helper(v[..., None].shape, (5, 7, 3, 1))

  def test_int_indices(self):
    v = Tensor.randn(5, 7, 3)
    numpy_testing_assert_equal_helper(v[[0, 4, 2]].shape, (3, 7, 3))
    numpy_testing_assert_equal_helper(v[:, [0, 4, 2]].shape, (5, 3, 3))
    numpy_testing_assert_equal_helper(v[:, [[0, 1], [4, 3]]].shape, (5, 2, 2, 3))

  def test_index_src_datatype(self):
    src = Tensor.ones(3, 2, 4)
    # test index
    res = src[[0, 2, 1], :, :]
    numpy_testing_assert_equal_helper(res.shape, src.shape)

  def test_empty_slice(self):
    x = Tensor.randn(2, 3, 4, 5)
    y = x[:, :, :, 1]
    z = y[:, 1:1, :]
    numpy_testing_assert_equal_helper((2, 0, 4), z.shape)

  def test_invalid_index(self):
    x = Tensor.arange(0, 16).reshape(4, 4)
    self.assertRaises(TypeError, lambda: x["0":"1"])

  def test_out_of_bound_index(self):
    x = Tensor.arange(0, 100).reshape(2, 5, 10)
    self.assertRaises(IndexError, lambda: x[0, 5])
    self.assertRaises(IndexError, lambda: x[4, 5])
    self.assertRaises(IndexError, lambda: x[0, 1, 15])
    self.assertRaises(IndexError, lambda: x[:, :, 12])

  def test_take_along_dim(self):
    # NOTE: the actual test logic is inside _test_against_numpy which is never called
    # This test effectively does nothing but defines a function
    def _test_against_numpy(t: Tensor, indices: Tensor, dim):
      actual = t.gather(dim, indices)
      t_np = t.numpy()
      indices_np = indices.numpy()
      expected = np.take_along_axis(t_np, indices_np, axis=dim)
      numpy_testing_assert_equal_helper(actual, expected)

      # TODO argsort
      '''
      for shape in [(3, 2), (2, 3, 5), (2, 4, 0), (2, 3, 1, 4)]:
        for noncontiguous in [True, False]:
          for dtype in (dtypes.float32, dtypes.int64):
            t = make_tensor(shape, dtype=dtype, noncontiguous=noncontiguous)
            for dim in list(range(t.ndim)) + [None]:
              if dim is None:
                indices = argsort(t.reshape(-1))
              else:
                indices = argsort(t, dim=dim)

          _test_against_numpy(t, indices, dim)
      '''

      # test broadcasting
      t = Tensor.ones((3, 4, 1))
      indices = Tensor.ones((1, 2, 5), dtype=dtypes.int64)

      _test_against_numpy(t, indices, 1)

      # test empty indices
      t = Tensor.ones((3, 4, 5))
      indices = Tensor.ones((3, 0, 5), dtype=dtypes.int64)

      _test_against_numpy(t, indices, 1)

class TestNumpy(unittest.TestCase):
  def test_index_no_floats(self):
    a = Tensor([[[5.]]])

    self.assertRaises(IndexError, lambda: a[0.0])
    self.assertRaises(IndexError, lambda: a[0, 0.0])
    self.assertRaises(IndexError, lambda: a[0.0, 0])
    self.assertRaises(IndexError, lambda: a[0.0, :])
    self.assertRaises(IndexError, lambda: a[:, 0.0])
    self.assertRaises(IndexError, lambda: a[:, 0.0, :])
    self.assertRaises(IndexError, lambda: a[0.0, :, :])
    self.assertRaises(IndexError, lambda: a[0, 0, 0.0])
    self.assertRaises(IndexError, lambda: a[0.0, 0, 0])
    self.assertRaises(IndexError, lambda: a[0, 0.0, 0])
    self.assertRaises(IndexError, lambda: a[-1.4])
    self.assertRaises(IndexError, lambda: a[0, -1.4])
    self.assertRaises(IndexError, lambda: a[-1.4, 0])
    self.assertRaises(IndexError, lambda: a[-1.4, :])
    self.assertRaises(IndexError, lambda: a[:, -1.4])
    self.assertRaises(IndexError, lambda: a[:, -1.4, :])
    self.assertRaises(IndexError, lambda: a[-1.4, :, :])
    self.assertRaises(IndexError, lambda: a[0, 0, -1.4])
    self.assertRaises(IndexError, lambda: a[-1.4, 0, 0])
    self.assertRaises(IndexError, lambda: a[0, -1.4, 0])
    # these two trigger slice internal type verification first
    self.assertRaises(TypeError, lambda: a[0.0:, 0.0])
    self.assertRaises(TypeError, lambda: a[0.0:, 0.0,:])

  def test_none_index(self):
    # `None` index adds newaxis
    a = Tensor([1, 2, 3])
    numpy_testing_assert_equal_helper(a[None].ndim, a.ndim+1)

  def test_everything_returns_views(self):
    # Before `...` would return a itself.
    a = Tensor([5])

    self.assertIs(a, a[()])
    self.assertIs(a, a[...])
    self.assertIs(a, a[:])

  def test_broaderrors_indexing(self):
    a = Tensor.zeros(5, 5)
    self.assertRaises(IndexError, a.__getitem__, ([0, 1], [0, 1, 2]))
    self.assertRaises(IndexError, a.contiguous().__setitem__, ([0, 1], [0, 1, 2]), 0)

if __name__ == '__main__':
  unittest.main()
