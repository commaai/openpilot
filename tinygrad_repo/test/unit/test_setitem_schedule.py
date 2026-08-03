import unittest
from tinygrad import Tensor, dtypes, GlobalCounters

class TestSetitemInto(unittest.TestCase):
  def test_setitem_into_unrealized(self):
    GlobalCounters.reset()
    t = Tensor.arange(4, dtype=dtypes.int32).reshape(2, 2)
    self.assertEqual(GlobalCounters.kernel_count, 0)
    t[1] = 5
    self.assertEqual(GlobalCounters.kernel_count, 0)
    t.realize()
    self.assertEqual(GlobalCounters.kernel_count, 0)
    self.assertEqual(GlobalCounters.global_mem, 0)
    self.assertListEqual(t.tolist(), [[0, 1], [5, 5]])

  def test_setitem_into_unrealized_sliced_compute(self):
    # base computation contains SHRINK from prior slicing (like QR decomposition pattern)
    GlobalCounters.reset()
    a = Tensor.arange(8, dtype=dtypes.int32).reshape(2, 4)
    w = a[0] + a[1]  # unrealized ADD with SHRINK in graph: [4, 6, 8, 10]
    self.assertEqual(GlobalCounters.kernel_count, 0)
    w[1] = 99
    self.assertEqual(GlobalCounters.kernel_count, 0)
    w.realize()
    self.assertEqual(GlobalCounters.kernel_count, 0)
    self.assertEqual(GlobalCounters.global_mem, 0)
    self.assertListEqual(w.tolist(), [4, 99, 8, 10])

  def test_setitem_into_empty(self):
    GlobalCounters.reset()
    t = Tensor.empty(4, dtype=dtypes.int32)
    t[1] = 5
    self.assertEqual(GlobalCounters.kernel_count, 0)
    t.realize()
    self.assertEqual(GlobalCounters.kernel_count, 1)
    self.assertEqual(GlobalCounters.global_mem, 4)
    t[1].realize()
    t.realize()
    self.assertEqual(GlobalCounters.kernel_count, 1)
    self.assertEqual(t[1].item(), 5)

  def test_setitem_into_empty_alu(self):
    GlobalCounters.reset()
    t = Tensor.empty(4, dtype=dtypes.int32) + 1
    self.assertEqual(GlobalCounters.kernel_count, 0)
    t[1] = 5
    self.assertEqual(GlobalCounters.kernel_count, 0)
    t.realize()
    self.assertEqual(GlobalCounters.kernel_count, 1)
    self.assertLessEqual(GlobalCounters.global_mem, 32)
    t[1].realize()
    t.realize()
    self.assertEqual(GlobalCounters.kernel_count, 1)
    self.assertEqual(t[1].item(), 5)

  def test_setitem_into_tensor(self):
    t = Tensor([1, 2, 3, 4], dtype=dtypes.int32).realize()
    GlobalCounters.reset()
    t[1] = 5
    self.assertEqual(GlobalCounters.kernel_count, 0)
    t[1].realize()
    self.assertEqual(GlobalCounters.kernel_count, 1)
    self.assertEqual(GlobalCounters.global_mem, 4)
    t.realize()
    self.assertEqual(GlobalCounters.kernel_count, 1)
    self.assertListEqual(t.tolist(), [1, 5, 3, 4])

  def test_setitem_into_tensor_alu(self):
    t = Tensor([1, 2, 3, 4], dtype=dtypes.int32).realize() + 1
    GlobalCounters.reset()
    t[1] = 5
    self.assertEqual(GlobalCounters.kernel_count, 0)
    t[1].realize()
    self.assertEqual(GlobalCounters.kernel_count, 1)
    self.assertLessEqual(GlobalCounters.global_mem, 32)
    t[1].realize()
    t.realize()
    self.assertEqual(GlobalCounters.kernel_count, 1)
    self.assertListEqual(t.tolist(), [2, 5, 4, 5])

  def test_setitem_into_const(self):
    GlobalCounters.reset()
    t = Tensor.ones(4, dtype=dtypes.int32, buffer=False)
    t[1] = 5
    self.assertEqual(GlobalCounters.kernel_count, 0)
    t.realize()
    self.assertEqual(GlobalCounters.kernel_count, 0)
    self.assertEqual(GlobalCounters.global_mem, 0)
    self.assertListEqual(t.tolist(), [1, 5, 1, 1])

  def test_setitem_into_const_alu(self):
    GlobalCounters.reset()
    t = Tensor.ones(4, dtype=dtypes.int32, buffer=False) + 1
    t[1] = 5
    self.assertEqual(GlobalCounters.kernel_count, 0)
    t.realize()
    self.assertEqual(GlobalCounters.kernel_count, 0)
    self.assertEqual(GlobalCounters.global_mem, 0)
    self.assertListEqual(t.tolist(), [2, 5, 2, 2])

  def test_setitem_into_arange(self):
    # NOTE: arange has no real buffer, but assigning to it is fine
    GlobalCounters.reset()
    other = Tensor.arange(4, dtype=dtypes.int32)
    t = Tensor.arange(4, dtype=dtypes.int32)
    self.assertIs(other.uop, t.uop)
    t[1] = 5
    self.assertEqual(GlobalCounters.kernel_count, 0)
    t.realize()
    self.assertEqual(GlobalCounters.kernel_count, 0)
    self.assertListEqual(t.tolist(), [0, 5, 2, 3])

  def test_setitem_slice_const(self):
    t = Tensor.zeros(100, dtype=dtypes.int32).contiguous().realize()
    GlobalCounters.reset()
    t[20:50] = 3
    self.assertEqual(GlobalCounters.kernel_count, 0)
    t.realize()
    self.assertEqual(GlobalCounters.kernel_count, 1)
    self.assertEqual(GlobalCounters.global_mem, 30*4)  # 30 elements written

  def test_setitem_slice_tensor(self):
    t = Tensor.zeros(100, dtype=dtypes.int32).contiguous().realize()
    v = Tensor.zeros(30, dtype=dtypes.int32).contiguous().realize()
    GlobalCounters.reset()
    t[20:50] = v
    self.assertEqual(GlobalCounters.kernel_count, 0)
    t.realize()
    self.assertEqual(GlobalCounters.kernel_count, 1)
    self.assertEqual(GlobalCounters.global_mem, 30*4*2)  # 30 read + 30 written

  def test_setitem_full(self):
    t = Tensor.zeros(100, dtype=dtypes.int32).contiguous().realize()
    GlobalCounters.reset()
    t[:] = 3
    self.assertEqual(GlobalCounters.kernel_count, 0)
    t.realize()
    self.assertEqual(GlobalCounters.kernel_count, 1)
    self.assertEqual(GlobalCounters.global_mem, 100*4)  # full buffer written

if __name__ == '__main__':
  unittest.main()
