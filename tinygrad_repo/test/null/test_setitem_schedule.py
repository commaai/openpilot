import unittest
from tinygrad import Tensor, dtypes, GlobalCounters
from test.helpers import assert_kernel_count

class TestSetitemInto(unittest.TestCase):
  def test_setitem_slice_const(self):
    t = Tensor.zeros(100, dtype=dtypes.int32).contiguous().realize()
    GlobalCounters.reset()
    t[20:50] = 3
    assert_kernel_count(0)
    t.realize()
    assert_kernel_count(1)
    self.assertEqual(GlobalCounters.global_mem, 30*4)  # 30 elements written

  def test_setitem_slice_tensor(self):
    t = Tensor.zeros(100, dtype=dtypes.int32).contiguous().realize()
    v = Tensor.zeros(30, dtype=dtypes.int32).contiguous().realize()
    GlobalCounters.reset()
    t[20:50] = v
    assert_kernel_count(0)
    t.realize()
    assert_kernel_count(1)
    self.assertEqual(GlobalCounters.global_mem, 30*4*2)  # 30 read + 30 written

  def test_setitem_full(self):
    t = Tensor.zeros(100, dtype=dtypes.int32).contiguous().realize()
    GlobalCounters.reset()
    t[:] = 3
    assert_kernel_count(0)
    t.realize()
    assert_kernel_count(1)
    self.assertEqual(GlobalCounters.global_mem, 100*4)  # full buffer written

if __name__ == '__main__':
  unittest.main()
