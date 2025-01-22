import unittest
from tinygrad.dtype import dtypes
from tinygrad.ops import UOp, resolve

class TestUOpResolve(unittest.TestCase):
  def test_simple_int(self):
    u = UOp.const(dtypes.int, 4)
    self.assertEqual(int(u), 4)

  def test_int_add(self):
    u = UOp.const(dtypes.int, 4) + 7
    self.assertEqual(int(u), 11)

  def test_lt(self):
    u = UOp.const(dtypes.int, 4) < 7
    self.assertTrue(u)

  def test_rfloordiv(self):
    u = 8 // UOp.const(dtypes.int, 4)
    self.assertEqual(int(u), 2)

  def test_rtruediv(self):
    u = 9 / UOp.const(dtypes.float, 4)
    self.assertEqual(float(u), 2.25)

  def test_leq(self):
    u = UOp.const(dtypes.int, 4) <= 4
    self.assertTrue(u)

  def test_ne(self):
    u = UOp.const(dtypes.int, 4) != 7
    self.assertTrue(u)

  def test_ne_f(self):
    u = UOp.const(dtypes.int, 4) != 4
    self.assertFalse(u)

  def test_ngt(self):
    u = UOp.const(dtypes.int, 4) > 7
    self.assertFalse(u)

  def test_ssimplify(self):
    self.assertEqual((8 % UOp.const(dtypes.int, 4)).ssimplify(), 0)
    self.assertEqual((8 * UOp.const(dtypes.int, 4)).ssimplify(), 32)

  def test_ambiguous_less_than(self):
    u = UOp.variable("i", 1, 10)
    self.assertTrue(resolve(u < 4))
    self.assertFalse(resolve(u < 4, False))
    self.assertTrue(resolve(u < 11, False))
    self.assertFalse(resolve(u < -1, False))
    self.assertFalse(resolve(u < -1, True))

  def test_float_direct(self):
    u = UOp.const(dtypes.float, 4.5) + 7
    self.assertEqual(float(u), 11.5)

  def test_var_cmp_t(self):
    u = UOp.variable("i", 1, 10) < 20
    self.assertTrue(u)

  def test_var_cmp_t2(self):
    u = UOp.variable("i", 1, 10)//2 < 20
    self.assertTrue(u)

  def test_var_cmp_f(self):
    u = UOp.variable("i", 1, 10) < 1
    self.assertFalse(u)

  def test_var_cmp_f2(self):
    u = UOp.variable("i", 1, 10) > 11
    self.assertFalse(u)

  def test_or_true(self):
    u = UOp.variable("b", False, True, dtypes.bool) | True
    self.assertTrue(u)

  def test_or_false(self):
    with self.assertRaises(ValueError):
      u = UOp.variable("b", False, True, dtypes.bool) | False
      self.assertTrue(u)

  def test_and_false(self):
    u = UOp.variable("b", False, True, dtypes.bool) & False
    self.assertFalse(u)

  def test_max(self):
    x = UOp.variable("x", 1, 10)
    y = UOp.variable("y", 5, 10)
    u = x.maximum(y)
    self.assertTrue(u < 20)
    self.assertFalse(u < 3)

  def test_x_lt_x(self):
    x = UOp.variable("i", 1, 10)
    self.assertFalse(x < x)

  @unittest.expectedFailure
  def test_x_lt_xp1(self):
    x = UOp.variable("i", 1, 10)
    self.assertTrue(x < (x+1))

  def test_and_true(self):
    with self.assertRaises(ValueError):
      u = UOp.variable("b", False, True, dtypes.bool) & True
      self.assertFalse(u)

  @unittest.expectedFailure
  def test_var_cmp_range(self):
    v = UOp.variable("i", 1, 10)
    u = (v > 4) | (v < 6)
    self.assertTrue(u)

  def test_var_cmp_assert(self):
    with self.assertRaises(ValueError):
      u = UOp.variable("i", 1, 10) < 5
      self.assertFalse(u)

  def test_plus_ordering_lt(self):
    i = UOp.variable("i", 1, 10)
    j = UOp.variable("j", 1, 10)
    self.assertFalse((i+j) < (j+i))

if __name__ == '__main__':
  unittest.main()