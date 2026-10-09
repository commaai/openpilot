import unittest
from tinygrad.uop.ops import Ops, UOp

class TestInvalidTensor(unittest.TestCase):
  def test_uop_where_keeps_invalid_bare(self):
    cond = UOp.const(0) < UOp.const(1)
    idx = UOp(Ops.STACK, src=tuple(UOp.const(x) for x in range(3)))
    out = cond.where(idx, UOp.invalid())
    self.assertIs(cond.op, Ops.CMPLT)
    self.assertIs(idx.op, Ops.STACK)
    self.assertIs(out.op, Ops.WHERE)
    self.assertIs(out.src[2].op, Ops.CONST)
    self.assertTrue(out.src[2].is_invalid)

if __name__ == '__main__':
  unittest.main()
