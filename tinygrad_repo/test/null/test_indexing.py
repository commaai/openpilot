# test cases are modified from pytorch test_indexing.py

import unittest

from tinygrad import Tensor

class TestIndexing(unittest.TestCase):
  def test_single_int(self):
    v = Tensor.randn(5, 7, 3)
    self.assertEqual(v[4].shape, (7, 3))

  def test_multiple_int(self):
    v = Tensor.randn(5, 7, 3)
    self.assertEqual(v[4].shape, (7, 3))
    self.assertEqual(v[4, :, 1].shape, (7,))

  def test_none(self):
    v = Tensor.randn(5, 7, 3)
    self.assertEqual(v[None].shape, (1, 5, 7, 3))
    self.assertEqual(v[:, None].shape, (5, 1, 7, 3))
    self.assertEqual(v[:, None, None].shape, (5, 1, 1, 7, 3))
    self.assertEqual(v[..., None].shape, (5, 7, 3, 1))

  def test_int_indices(self):
    v = Tensor.randn(5, 7, 3)
    self.assertEqual(v[[0, 4, 2]].shape, (3, 7, 3))
    self.assertEqual(v[:, [0, 4, 2]].shape, (5, 3, 3))
    self.assertEqual(v[:, [[0, 1], [4, 3]]].shape, (5, 2, 2, 3))

  def test_index_src_datatype(self):
    src = Tensor.ones(3, 2, 4)
    # test index
    res = src[[0, 2, 1], :, :]
    self.assertEqual(res.shape, src.shape)

  def test_empty_slice(self):
    x = Tensor.randn(2, 3, 4, 5)
    y = x[:, :, :, 1]
    z = y[:, 1:1, :]
    self.assertEqual((2, 0, 4), z.shape)

  def test_invalid_index(self):
    x = Tensor.arange(0, 16).reshape(4, 4)
    self.assertRaises(TypeError, lambda: x["0":"1"])

  def test_out_of_bound_index(self):
    x = Tensor.arange(0, 100).reshape(2, 5, 10)
    self.assertRaises(IndexError, lambda: x[0, 5])
    self.assertRaises(IndexError, lambda: x[4, 5])
    self.assertRaises(IndexError, lambda: x[0, 1, 15])
    self.assertRaises(IndexError, lambda: x[:, :, 12])

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
    self.assertEqual(a[None].ndim, a.ndim+1)

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
