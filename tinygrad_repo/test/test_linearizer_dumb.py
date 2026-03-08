# ruff: noqa: E501
# tests where the Linearizer is doing something dumb
# like test_linearizer_failures, but they don't have to fail

import unittest
from tinygrad import Device, dtypes
from tinygrad.uop.ops import UOp, Ops, AxisType, KernelInfo
from tinygrad.codegen.opt.search import Opt, OptOps
from tinygrad.engine.realize import get_program

class TestLinearizerFailure(unittest.TestCase):
  @unittest.skipUnless(Device.DEFAULT == "METAL", "only tested on METAL")
  def test_failure_beam_mnist(self):
    c0 = UOp(Ops.PARAM, dtypes.uchar.ptr(4014080), arg=0, src=())
    c1 = UOp.range(UOp.const(dtypes.index, 512), 0, AxisType.GLOBAL)
    c2 = UOp.range(UOp.const(dtypes.index, 784), 1, AxisType.GLOBAL)
    c3 = UOp.range(UOp.const(dtypes.index, 10), 3, AxisType.GLOBAL)
    c4 = UOp(Ops.PARAM, dtypes.int.ptr(512), arg=1, src=())
    c5 = c4.index(c1.valid(UOp.const(dtypes.bool, True)))
    c6 = UOp.range(UOp.const(dtypes.index, 6000), 1004, AxisType.REDUCE)
    c7 = UOp.range(UOp.const(dtypes.index, 3750), 2006, AxisType.REDUCE)
    c8 = UOp.range(UOp.const(dtypes.index, 16), 2007, AxisType.GROUP_REDUCE)
    c9 = UOp(Ops.PARAM, dtypes.uchar.ptr(47040000), arg=2, src=())
    c10 = c9.index((((c3*UOp.const(dtypes.index, 4704000))+c2)+(c6*UOp.const(dtypes.index, 784))).valid(UOp.const(dtypes.bool, True)))
    c11 = c5.alu(Ops.CMPNE, ((((c3*UOp.const(dtypes.index, 6000))+c6)+((c7*UOp.const(dtypes.index, 16))+c8)).alu(Ops.CMPLT, UOp.const(dtypes.index, 59999)).where(UOp.const(dtypes.int, 0), UOp.const(dtypes.int, 1)).reduce(c7, c8, arg=Ops.ADD)+UOp.const(dtypes.int, -1))).where(UOp.const(dtypes.uchar, 0), c10).reduce(c6, arg=Ops.ADD)
    c12 = c0.index((((c1*UOp.const(dtypes.index, 7840))+(c2*UOp.const(dtypes.index, 10)))+c3).valid(UOp.const(dtypes.bool, True))).store(c11).end(c1, c2, c3)
    ast = c12.sink(arg=KernelInfo(name='test', axis_types=(), dont_use_locals=False, applied_opts=(Opt(op=OptOps.GROUP, axis=1, arg=16),), opts_to_apply=None))
    _ = get_program(ast, Device["METAL"].renderer)

if __name__ == '__main__':
  unittest.main()
