import unittest
from tinygrad import UOp, dtypes

class TestUOpRepr(unittest.TestCase):
  def test_simple_const(self):
    a = UOp.const(dtypes.int, 42)
    self.assertEqual(repr(a), "UOp(Ops.CONST, dtypes.int, arg=42, src=())")
  def test_different_consts(self):
    a, b = UOp.const(dtypes.int, 42), UOp.const(dtypes.int, 3)
    expected = (
      "UOp(Ops.ADD, dtypes.int, arg=None, src=(\n" +
      "  UOp(Ops.CONST, dtypes.int, arg=42, src=()),\n" +
      "  UOp(Ops.CONST, dtypes.int, arg=3, src=()),))"
    )
    self.assertEqual(repr(a+b), expected)
  def test_walrus_operator_indentation(self):
    # The reference should have the same indentation as the definition
    a = UOp.const(dtypes.int, 42)
    expected = (
      "UOp(Ops.ADD, dtypes.int, arg=None, src=(\n" +
      "  x0:=UOp(Ops.CONST, dtypes.int, arg=42, src=()),\n" +
      "  x0,))"
    )
    self.assertEqual(repr(a+a), expected)
  def test_nested_walrus_indentation(self):
    # Ensure indentation is consistent at multiple levels
    b = (a:=UOp.const(dtypes.int, 1)) + a
    expected = (
      "UOp(Ops.MUL, dtypes.int, arg=None, src=(\n" +
      "  x0:=UOp(Ops.ADD, dtypes.int, arg=None, src=(\n" +
      "    x1:=UOp(Ops.CONST, dtypes.int, arg=1, src=()),\n" +
      "    x1,)),\n" +
      "  x0,))"
    )
    self.assertEqual(repr(b*b), expected)

if __name__ == '__main__':
  unittest.main()
