# ruff: noqa: E501
import unittest
from tinygrad.uop.ops import UOp, Ops, AxisType, KernelInfo
from tinygrad.dtype import dtypes
from tinygrad.device import Device
from tinygrad.codegen import to_program

class TestLinearizerFailures(unittest.TestCase):
  def test_fail_1(self):
    c0 = UOp.param(0, dtypes.float, 64)
    c1 = UOp.range(UOp.const(2), 1, AxisType.WEAK)
    c2 = UOp.range(UOp.const(32), 2, AxisType.WEAK)
    c3 = ((c1*UOp.const(32))+c2)
    c4 = UOp.param(1, dtypes.float, 163840)
    c5 = UOp.range(UOp.const(2560), 0, AxisType.REDUCE)
    c6 = c4.index(((((((c5//UOp.const(8))%UOp.const(8))*UOp.const(8))+(c5%UOp.const(8)))+(((c2*UOp.const(40))+(c5//UOp.const(64)))*UOp.const(64)))+(c1*UOp.const(81920))))
    c7 = UOp.param(2, dtypes.float, 64)
    c8 = c7.index(c3)
    c9 = ((((c6+(c8*UOp.const(-1.0)))*(c6+(c8*UOp.const(-1.0)))).reduce(c5, arg=Ops.ADD)*UOp.const(0.000390625))+UOp.const(1e-05)).sqrt().reciprocal()
    c10 = c0.index(c3).store(c9).end(c1, c2)
    ast = c10.sink(arg=KernelInfo())
    to_program(ast, renderer=Device[Device.DEFAULT].renderer)

if __name__ == '__main__':
  unittest.main()
