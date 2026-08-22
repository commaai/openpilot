import unittest
from tinygrad import UOp

class TestUOpRepr(unittest.TestCase):
  def test_simple_const(self):
    a = UOp.const(42)
    self.assertEqual(repr(a), "UOp(Ops.CONST, dtypes.weakint, arg=42, src=())")
  def test_different_consts(self):
    a, b = UOp.const(42), UOp.const(3)
    expected = (
      "UOp(Ops.ADD, dtypes.weakint, arg=None, src=(\n" +
      "  UOp(Ops.CONST, dtypes.weakint, arg=42, src=()),\n" +
      "  UOp(Ops.CONST, dtypes.weakint, arg=3, src=()),))"
    )
    self.assertEqual(repr(a+b), expected)
  def test_walrus_operator_indentation(self):
    # The reference should have the same indentation as the definition
    a = UOp.const(42)
    expected = (
      "UOp(Ops.ADD, dtypes.weakint, arg=None, src=(\n" +
      "  x0:=UOp(Ops.CONST, dtypes.weakint, arg=42, src=()),\n" +
      "  x0,))"
    )
    self.assertEqual(repr(a+a), expected)
  def test_nested_walrus_indentation(self):
    # Ensure indentation is consistent at multiple levels
    b = (a:=UOp.const(1)) + a
    expected = (
      "UOp(Ops.MUL, dtypes.weakint, arg=None, src=(\n" +
      "  x0:=UOp(Ops.ADD, dtypes.weakint, arg=None, src=(\n" +
      "    x1:=UOp(Ops.CONST, dtypes.weakint, arg=1, src=()),\n" +
      "    x1,)),\n" +
      "  x0,))"
    )
    self.assertEqual(repr(b*b), expected)

if __name__ == '__main__':
  unittest.main()
