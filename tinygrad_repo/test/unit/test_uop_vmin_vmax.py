import unittest, math
from tinygrad.ops import UOp, Ops
from tinygrad.dtype import dtypes

class TestVminVmaxProperties(unittest.TestCase):
  def test_vmin_vmax_constant(self):
    # vmin and vmax for a constant
    uop = UOp.const(dtypes.int32, 42)
    self.assertEqual(uop.vmin, 42)
    self.assertEqual(uop.vmax, 42)

  def test_vmin_vmax_cmpne(self):
    uop = UOp.const(dtypes.int32, 42)
    def test_bool(u, x):
      self.assertEqual(u.vmin, x)
      self.assertEqual(u.vmax, x)
    test_bool(uop != 42, False)
    test_bool(uop != 43, True)
    test_bool(uop != 41, True)

  def test_vmin_vmax_addition_with_variable(self):
    # vmin and vmax for addition with a variable
    x = UOp.variable('x', 10, 20)
    uop = x + 5
    self.assertEqual(uop.vmin, 15)
    self.assertEqual(uop.vmax, 25)

  def test_vmin_vmax_multiplication_with_variable(self):
    # vmin and vmax for multiplication with a variable
    x = UOp.variable('x', -3, 4)
    uop = x * 2
    self.assertEqual(uop.vmin, -6)
    self.assertEqual(uop.vmax, 8)

  def test_vmin_vmax_variable_inside_special(self):
    uop = UOp(Ops.SPECIAL, dtypes.int, arg=('gidx0', UOp(Ops.DEFINE_VAR, dtypes.int, arg=('i', 1, 10))))
    self.assertEqual(uop.vmin, 0)
    self.assertEqual(uop.vmax, 10)

  def test_vmin_vmax_multiplication_0_inf(self):
    # vmin and vmax for multiplication with a variable
    x = UOp.const(dtypes.float, 0.0)
    y = UOp.load(UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), (), 0), UOp.const(dtypes.int, 0), dtype=dtypes.float)
    uop = x * y
    # TODO: these should be 0, but definitely should not be nan
    self.assertEqual(uop.vmin, -math.inf)
    self.assertEqual(uop.vmax, math.inf)

  def test_vmin_vmax_with_negative_multiplication(self):
    # vmin and vmax when multiplying by a negative number
    x = UOp.variable('x', 2, 5)
    uop = x * -3
    self.assertEqual(uop.vmin, -15)
    self.assertEqual(uop.vmax, -6)

  def test_vmin_vmax_with_negative_multiplication2(self):
    # vmin and vmax when multiplying by a negative number
    x = UOp.variable('x', -2, 5)
    uop = x * -3
    self.assertEqual(uop.vmin, -15)
    self.assertEqual(uop.vmax, 6)

  def test_vmin_vmax_nested_min_max(self):
    # vmin and vmax with nested min/max operations
    x = UOp.variable('x', 0, 10)
    uop = x.maximum(5).minimum(8)
    self.assertEqual(uop.vmin, 5)
    self.assertEqual(uop.vmax, 8)

  def test_vmin_vmax_where(self):
    x = UOp.variable('x', 0, 10)
    y = UOp.variable('y', 1, 11)
    z = UOp.variable('z', 2, 12)
    uop = (x<5).where(y, z)
    self.assertEqual(uop.vmin, 1)
    self.assertEqual(uop.vmax, 12)

  def test_vmin_vmax_shl(self):
    x = UOp.variable('x', 0, 10) << 5
    self.assertEqual(x.vmin, 0)
    self.assertEqual(x.vmax, 10 << 5)

  def test_vmin_vmax_shr(self):
    x = UOp.variable('x', 0, 10) >> 2
    self.assertEqual(x.vmin, 0)
    self.assertEqual(x.vmax, 10 >> 2)

class TestVminVmaxDivMod(unittest.TestCase):
  def test_vmin_vmax_division_positive(self):
    # vmin and vmax for division of a variable by a positive constant
    x = UOp.variable('x', 10, 20)
    uop = x // 2
    self.assertEqual(uop.vmin, 5)
    self.assertEqual(uop.vmax, 10)

  def test_vmin_vmax_division_negative(self):
    # vmin and vmax for division of a variable by a negative constant
    x = UOp.variable('x', 10, 20)
    uop = x // -2
    self.assertEqual(uop.vmin, -10)
    self.assertEqual(uop.vmax, -5)

  def test_vmin_vmax_mod_positive(self):
    # vmin and vmax for modulo of a variable by a positive constant
    x = UOp.variable('x', 10, 20)
    uop = x % 3
    self.assertEqual(uop.vmin, 0)
    self.assertEqual(uop.vmax, 2)

  @unittest.skip("broken")
  def test_vmin_vmax_mod_negative(self):
    # vmin and vmax for modulo of a variable by a negative constant
    x = UOp.variable('x', 10, 20)
    uop = x % -3
    self.assertEqual(uop.vmin, -2)
    self.assertEqual(uop.vmax, 0)

  def test_vmin_vmax_division_with_mixed_range(self):
    # vmin and vmax for division of a variable with a range crossing zero
    x = UOp.variable('x', -10, 10)
    uop = x // 3
    self.assertEqual(uop.vmin, -4)  # -10//3 = -4
    self.assertEqual(uop.vmax, 3)   # 10//3 = 3

  def test_vmin_vmax_mod_with_mixed_range(self):
    # vmin and vmax for modulo of a variable with a range crossing zero
    x = UOp.variable('x', -10, 10)
    uop = x % 4
    self.assertEqual(uop.vmin, -3)
    self.assertEqual(uop.vmax, 3)

class TestVminVmaxVConst(unittest.TestCase):
  def test_vmin_vmax_vconst_single_element(self):
    # vmin and vmax for a single-element vector constant
    uop = UOp.const(dtypes.int32.vec(1), (42,))
    self.assertEqual(uop.vmin, 42)
    self.assertEqual(uop.vmax, 42)

  def test_vmin_vmax_vconst_multiple_elements(self):
    # vmin and vmax for a multi-element vector constant
    uop = UOp.const(dtypes.int32.vec(4), (10, 20, -5, 7))
    self.assertEqual(uop.vmin, -5)
    self.assertEqual(uop.vmax, 20)

  def test_vmin_vmax_vconst_all_equal(self):
    # vmin and vmax for a vector where all elements are equal
    uop = UOp.const(dtypes.int32.vec(3), (7, 7, 7))
    self.assertEqual(uop.vmin, 7)
    self.assertEqual(uop.vmax, 7)

  def test_vmin_vmax_vconst_with_negative_values(self):
    # vmin and vmax for a vector constant containing negative values
    uop = UOp.const(dtypes.int32.vec(4), (-10, -20, -5, -15))
    self.assertEqual(uop.vmin, -20)
    self.assertEqual(uop.vmax, -5)

  def test_vmin_vmax_vconst_with_floats(self):
    # vmin and vmax for a vector constant of float values
    uop = UOp.const(dtypes.float32.vec(3), (1.5, -3.2, 0.0))
    self.assertEqual(uop.vmin, -3.2)
    self.assertEqual(uop.vmax, 1.5)

  def test_vmin_vmax_vconst_with_bools(self):
    # vmin and vmax for a vector constant of bool values
    uop = UOp.const(dtypes.float32.vec(3), (True, False, False))
    # TODO: these return floats, not bool
    self.assertEqual(uop.vmin, 0.0)
    self.assertEqual(uop.vmax, 1.0)

class TestConstFactor(unittest.TestCase):
  def test_const_factor_constant(self):
    # const_factor for a constant
    uop = UOp.const(dtypes.int32, 42)
    self.assertEqual(uop.const_factor(), 42)

  def test_const_factor_addition(self):
    # const_factor for an addition of constants
    uop = UOp.const(dtypes.int32, 30) + UOp.const(dtypes.int32, 12)
    self.assertEqual(uop.const_factor(), 6)  # GCD(30, 12) = 6

  def test_const_factor_multiplication(self):
    # const_factor for a multiplication of constants
    uop = UOp.const(dtypes.int32, 5) * UOp.const(dtypes.int32, 7)
    self.assertEqual(uop.const_factor(), 5)  # For multiplication, it's one of the factors

  def test_const_factor_with_variable(self):
    # const_factor for an expression involving a variable
    x = UOp.variable('x', 10, 20)
    uop = x * 3
    self.assertEqual(uop.const_factor(), 3)

  def test_const_factor_division(self):
    # const_factor for an expression with division
    x = UOp.variable('x', 10, 20)
    uop = x // 4
    self.assertEqual(uop.const_factor(), 1)  # Division reduces the const_factor to 1

  def test_const_factor_multiplication_of_var_and_const(self):
    # const_factor for multiplication of a variable and a constant
    x = UOp.variable('x', 6, 18)
    uop = x * 4
    self.assertEqual(uop.const_factor(), 4)  # Constant factor 4

  @unittest.skip("broken")
  def test_const_factor_multiplication_of_consts_and_vars(self):
    # Multiplying constants and variables
    x = UOp.variable('x', 10, 20)
    uop = (x * 3) * 5
    self.assertEqual(uop.const_factor(), 15)  # Constant multipliers are combined (3 * 5 = 15)

class TestDivides(unittest.TestCase):
  def test_divides_constant_exact(self):
    # Divides a constant by an exact divisor
    uop = UOp.const(dtypes.int32, 42)
    result = uop.divides(7)
    self.assertIsNotNone(result)
    self.assertEqual(result.const_factor(), 6)  # 42 / 7 = 6

  def test_divides_constant_inexact(self):
    # Try to divide a constant by a non-exact divisor
    uop = UOp.const(dtypes.int32, 42)
    result = uop.divides(5)
    self.assertIsNone(result)  # 42 is not divisible by 5

  @unittest.skip("broken")
  def test_divides_variable_and_constant(self):
    # Multiplying a variable by a constant, then dividing by the same constant
    x = UOp.variable('x', 10, 20)
    uop = x * 6
    result = uop.divides(6)
    self.assertIsNotNone(result)
    self.assertEqual(result, x)  # (x * 6) / 6 = x

  def test_divides_complex_expression(self):
    # Dividing a more complex expression
    x = UOp.variable('x', 10, 20)
    uop = (x * 6) + 18
    result = uop.divides(6)
    self.assertIsNotNone(result)
    self.assertEqual(result.const_factor(), 1)  # (x + 3), const_factor is 1

  def test_divides_with_inexact_factors(self):
    # Multiplying by a constant but dividing by a non-exact divisor
    x = UOp.variable('x', 15, 45)
    uop = x * 4
    result = uop.divides(3)
    self.assertIsNone(result)  # Cannot divide by 3, since 4 is not divisible by 3

if __name__ == '__main__':
  unittest.main()
