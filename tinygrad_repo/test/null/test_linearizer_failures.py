# ruff: noqa: E501
import unittest
from tinygrad.uop.ops import UOp, Ops, AxisType, KernelInfo
from tinygrad.dtype import dtypes
from tinygrad.device import Device
from tinygrad.codegen import to_program

class TestLinearizerFailures(unittest.TestCase):
  def test_fail_1(self):
    c0 = UOp(Ops.PARAM, dtypes.float.ptr(64), arg=0, src=())
    c1 = UOp.range(UOp.const(dtypes.weakint, 2), 1, AxisType.LOOP)
    c2 = UOp.range(UOp.const(dtypes.weakint, 32), 2, AxisType.LOOP)
    c3 = ((c1*UOp.const(dtypes.weakint, 32))+c2)
    c4 = UOp(Ops.PARAM, dtypes.float.ptr(163840), arg=1, src=())
    c5 = UOp.range(UOp.const(dtypes.weakint, 2560), 0, AxisType.REDUCE)
    c6 = c4.index(((((((c5//UOp.const(dtypes.weakint, 8))%UOp.const(dtypes.weakint, 8))*UOp.const(dtypes.weakint, 8))+(c5%UOp.const(dtypes.weakint, 8)))+(((c2*UOp.const(dtypes.weakint, 40))+(c5//UOp.const(dtypes.weakint, 64)))*UOp.const(dtypes.weakint, 64)))+(c1*UOp.const(dtypes.weakint, 81920))))
    c7 = UOp(Ops.PARAM, dtypes.float.ptr(64), arg=2, src=())
    c8 = c7.index(c3)
    c9 = ((((c6+(c8*UOp.const(dtypes.float, -1.0)))*(c6+(c8*UOp.const(dtypes.float, -1.0)))).reduce(c5, arg=Ops.ADD)*UOp.const(dtypes.float, 0.000390625))+UOp.const(dtypes.float, 1e-05)).sqrt().reciprocal()
    c10 = c0.index(c3).store(c9).end(c1, c2)
    ast = c10.sink(arg=KernelInfo())
    to_program(ast, renderer=Device[Device.DEFAULT].renderer)

if __name__ == '__main__':
  unittest.main()
