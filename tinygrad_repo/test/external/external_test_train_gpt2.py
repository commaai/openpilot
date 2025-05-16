# ruff: noqa: E501
import unittest

from tinygrad.ops import UOp, Ops
from tinygrad.engine.search import Opt, OptOps
from tinygrad.dtype import dtypes
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View
from tinygrad.codegen.kernel import Kernel

from test.external.fuzz_linearizer import run_linearizer

class TestTrainGpt2Kernel(unittest.TestCase):
  def test_1(self):
    # kernel 244
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(206045184), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(4, 1024, 50304, 1), strides=(51511296, 50304, 1, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (3,)), src=(
        UOp(Ops.MUL, dtypes.float, arg=None, src=(
          UOp(Ops.LOAD, dtypes.float, arg=None, src=(
          UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(3145728), arg=1, src=()),
          UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(4, 1024, 50304, 768), strides=(786432, 768, 0, 1), offset=0, mask=None, contiguous=False),)), src=()),)),
          UOp(Ops.LOAD, dtypes.float, arg=None, src=(
          UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(38633472), arg=2, src=()),
          UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(4, 1024, 50304, 768), strides=(0, 0, 768, 1), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),)),))

    opts = [Opt(op=OptOps.LOCAL, axis=0, arg=16), Opt(op=OptOps.UPCAST, axis=1, arg=3), Opt(op=OptOps.LOCAL, axis=0, arg=2)]
    kernel = Kernel(ast)
    kernel.apply_opts(opts)
    run_linearizer(kernel)

  def test_2(self):
    # kernel 254
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(3145728), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(4, 1024, 1, 768), strides=(786432, 768, 0, 1), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (2,)), src=(
        UOp(Ops.MUL, dtypes.float, arg=None, src=(
          UOp(Ops.LOAD, dtypes.float, arg=None, src=(
          UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(38633472), arg=1, src=()),
          UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(4, 1024, 50304, 768), strides=(0, 0, 768, 1), offset=0, mask=None, contiguous=False),)), src=()),)),
          UOp(Ops.LOAD, dtypes.float, arg=None, src=(
          UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(205852672), arg=2, src=()),
          UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(4, 1024, 50304, 768), strides=(51463168, 50257, 1, 0), offset=0, mask=((0, 4), (0, 1024), (0, 50257), (0, 768)), contiguous=False),)), src=()),)),)),)),)),))

    opts = [Opt(op=OptOps.LOCAL, axis=1, arg=16), Opt(op=OptOps.LOCAL, axis=0, arg=8), Opt(op=OptOps.UPCAST, axis=2, arg=4), Opt(op=OptOps.UPCAST, axis=1, arg=4), Opt(op=OptOps.LOCAL, axis=1, arg=4), Opt(op=OptOps.UPCAST, axis=3, arg=4)]
    kernel = Kernel(ast)
    kernel.apply_opts(opts)
    run_linearizer(kernel)

if __name__ == "__main__":
  unittest.main()