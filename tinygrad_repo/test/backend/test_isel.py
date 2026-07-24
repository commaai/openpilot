import unittest
from typing import cast
from tinygrad import Device
from tinygrad.uop import Ops
from tinygrad.uop.ops import UOp, dtypes, graph_rewrite
from tinygrad.renderer.isa.x86 import X86Renderer, X86Ops
from tinygrad.renderer.isa import IselContext

# INDEX on a register value with a constant index extracts a single element (the old GEP)
def lane(y:UOp, i:int) -> UOp: return y.index(UOp.const(dtypes.int, i), dtype=y.dtype.scalar())

@unittest.skipUnless(isinstance(Device[Device.DEFAULT].renderer, X86Renderer), "only x86")
class TestIselX86(unittest.TestCase):
  def isel_rewrite(self, x:UOp):
    return graph_rewrite(x, cast(X86Renderer, Device[Device.DEFAULT].renderer).isel_matcher, IselContext(x), bottom_up=True)

  def _check_op(self, dt_op, expr):
    nargs = expr.__code__.co_argcount
    for dt,op in dt_op:
      with self.subTest(dtype=dt):
        v = [UOp.variable(str(i), 0, 0, dt) for i in range(nargs)]
        n = self.isel_rewrite(expr(*v))
        self.assertIs(n.arg, op)

  def test_cmove(self):
    a = UOp.variable("a", 0, 0, dtypes.int32)
    b = UOp.variable("b", 0, 0, dtypes.int32)
    c = (a < b).where(a, b)
    d = (a != b).where(a, b)
    f = c + d
    n = self.isel_rewrite(f)
    self.assertTrue(n.src[0].arg is X86Ops.CMOVL and n.src[1].arg is X86Ops.CMOVNE)
    # both comparisons become the same instruction
    self.assertTrue(n.src[0].src[2] == n.src[1].src[2] and n.src[0].src[2].arg is X86Ops.CMP)

  def test_vinsertps(self):
    a = UOp.variable("a", 0, 0, dtypes.float32)
    b = UOp.variable("b", 0, 0, dtypes.float32)
    c = UOp.variable("c", 0, 0, dtypes.float32)
    d = UOp.variable("e", 0, 0, dtypes.float32)

    valid = [UOp.stack(lane(a, 0), lane(b, 1), lane(a, 2), lane(b, 3)),
             UOp.stack(lane(a, 3), lane(b, 2), lane(c, 1), d)]
    for shuf in valid: self.assertIs(self.isel_rewrite(shuf).arg, X86Ops.VINSERTPS)

  # complex address is [base + index*scale + displacement]
  def test_complex_address(self):
    a = UOp.variable("a", 0, 0, dtypes.int32)
    load = UOp.param(0, dtypes.int32, (16,)).index(a + 1).load()
    n = self.isel_rewrite(load)
    # displacement is the constant in "a" scaled to the buffer element size, dtype is int8 when the value fits otherwise int32
    self.assertTrue(n.src[2].op is Ops.CONST and n.src[2].dtype is dtypes.int8 and n.src[2].arg == 4)

if __name__ == "__main__":
  unittest.main()
