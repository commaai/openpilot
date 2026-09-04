# ruff: noqa: E501
# tests where the Linearizer is doing something dumb
# like test_linearizer_failures, but they don't have to fail

import unittest
from tinygrad import Device, dtypes
from tinygrad.uop.ops import UOp, Ops, AxisType, KernelInfo
from tinygrad.codegen.opt.search import Opt, OptOps
from tinygrad.codegen import to_program

class TestLinearizerFailure(unittest.TestCase):
  @unittest.skipUnless(Device.DEFAULT == "METAL", "only tested on METAL")
  def test_failure_beam_mnist(self):
    c0 = UOp.param(0, dtypes.uchar, 4014080)
    c1 = UOp.range(UOp.const(512), 0, AxisType.GLOBAL)
    c2 = UOp.range(UOp.const(784), 1, AxisType.GLOBAL)
    c3 = UOp.range(UOp.const(10), 3, AxisType.GLOBAL)
    c4 = UOp.param(1, dtypes.int, 512)
    c5 = c4.index(c1.valid(UOp.const(True)))
    c6 = UOp.range(UOp.const(6000), 1004, AxisType.REDUCE)
    c7 = UOp.range(UOp.const(3750), 2006, AxisType.REDUCE)
    c8 = UOp.range(UOp.const(16), 2007, AxisType.GROUP_REDUCE)
    c9 = UOp.param(2, dtypes.uchar, 47040000)
    c10 = c9.index((((c3*UOp.const(4704000))+c2)+(c6*UOp.const(784))).valid(UOp.const(True)))
    c11 = c5.alu(Ops.CMPNE, ((((c3*UOp.const(6000))+c6)+((c7*UOp.const(16))+c8)).alu(Ops.CMPLT, UOp.const(59999)).where(UOp.const(0).cast(dtypes.int), UOp.const(1).cast(dtypes.int)).reduce(c7, c8, arg=Ops.ADD)+UOp.const(-1).cast(dtypes.int))).where(UOp.const(0).cast(dtypes.uchar), c10).reduce(c6, arg=Ops.ADD)
    c12 = c0.index((((c1*UOp.const(7840))+(c2*UOp.const(10)))+c3).valid(UOp.const(True))).store(c11).end(c1, c2, c3)
    ast = c12.sink(arg=KernelInfo(name='test', applied_opts=(Opt(op=OptOps.SPLIT, axis=4, arg=(16, AxisType.GROUP_REDUCE)),), opts_to_apply=None))
    _ = to_program(ast, Device["METAL"].renderer)

if __name__ == '__main__':
  unittest.main()
