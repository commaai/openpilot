import unittest
from tinygrad import Tensor, dtypes

class TestCallify(unittest.TestCase):
  def test_basic(self):
    a = Tensor([1.,2,3])
    b = Tensor([4.,5,6])
    out = a + b
    out.callify()
    self.assertListEqual(out.tolist(), [5.0, 7.0, 9.0])

  def test_const(self):
    out = Tensor(2.0) + Tensor(3.0)
    out.callify()
    self.assertEqual(out.item(), 5.0)

  def test_sum(self):
    out = Tensor.ones(16).contiguous().sum()
    out.callify()
    self.assertEqual(out.item(), 16.0)

  def test_multi_output(self):
    a = Tensor([1.,2,3])
    b = Tensor([4.,5,6])
    c = a + b
    d = a * b
    c.callify(d)
    self.assertListEqual(c.tolist(), [5.0, 7.0, 9.0])
    self.assertListEqual(d.tolist(), [4.0, 10.0, 18.0])

  def test_two_callify_independent(self):
    a = Tensor([1.,2,3])
    b = Tensor([4.,5,6])
    c = a + b
    c.callify()

    d = Tensor([10.,20,30])
    e = Tensor([1.,1,1])
    f = d - e
    f.callify()

    self.assertListEqual(c.tolist(), [5.0, 7.0, 9.0])
    self.assertListEqual(f.tolist(), [9.0, 19.0, 29.0])

  def test_two_callify_shared_input(self):
    a = Tensor([1.,2,3]).contiguous().realize()
    b = a + 1
    b.callify()
    c = a * 2
    c.callify()
    self.assertListEqual(b.tolist(), [2.0, 3.0, 4.0])
    self.assertListEqual(c.tolist(), [2.0, 4.0, 6.0])

  def test_chained_callify(self):
    a = Tensor([1.,2,3])
    b = a + 1
    b.callify()
    b.realize()
    c = b + 1
    c.callify()
    self.assertListEqual(c.tolist(), [3.0, 4.0, 5.0])

  def test_gemm(self):
    a = Tensor.ones(8, 8).contiguous()
    b = Tensor.eye(8).contiguous()
    out = a @ b
    out.callify()
    lst = out.tolist()
    for y in range(8):
      for x in range(8):
        self.assertEqual(lst[y][x], 1.0)

  def test_int_dtype(self):
    a = Tensor([1,2,3], dtype=dtypes.int)
    b = Tensor([4,5,6], dtype=dtypes.int)
    out = a + b
    out.callify()
    self.assertListEqual(out.tolist(), [5, 7, 9])

  def test_reduce(self):
    out = Tensor([1.,2,3,4]).sum()
    out.callify()
    self.assertEqual(out.item(), 10.0)

  def test_multiple_ops(self):
    a = Tensor([1.,2,3])
    b = Tensor([4.,5,6])
    out = (a + b) * (a - b)
    out.callify()
    self.assertListEqual(out.tolist(), [-15.0, -21.0, -27.0])

  def test_double_callify(self):
    a = Tensor([1.,2,3])
    b = Tensor([4.,5,6])
    out = a + b
    out.callify()
    out.callify()
    self.assertListEqual(out.tolist(), [5.0, 7.0, 9.0])

  def test_double_callify_multi_output(self):
    a = Tensor([1.,2,3])
    b = Tensor([4.,5,6])
    c = a + b
    d = a * b
    c.callify(d)
    c.callify(d)
    self.assertListEqual(c.tolist(), [5.0, 7.0, 9.0])
    self.assertListEqual(d.tolist(), [4.0, 10.0, 18.0])

if __name__ == "__main__":
  unittest.main()
