# ruff: noqa: E501
import unittest

from tinygrad import Device
from tinygrad.ops import UOp, Ops
from tinygrad.engine.search import Opt, OptOps
from tinygrad.dtype import dtypes
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View
from tinygrad.codegen.kernel import Kernel

class TestOpenpilotValidhack(unittest.TestCase):
  def test_valid_removal(self):
    Device.DEFAULT = "GPU"

    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.imagef((64, 1024, 4)), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 64, 128, 1, 1, 8, 4, 1, 1, 1, 1), strides=(0, 4096, 32, 0, 0, 4, 1, 0, 0, 0, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.ADD, dtypes.float, arg=None, src=(
          UOp(Ops.MAX, dtypes.float, arg=None, src=(
            x5:=UOp(Ops.ADD, dtypes.float, arg=None, src=(
              UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (7, 8, 9, 10)), src=(
                UOp(Ops.CAST, dtypes.float, arg=None, src=(
                  UOp(Ops.MUL, dtypes.float, arg=None, src=(
                    UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                      UOp(Ops.DEFINE_GLOBAL, dtypes.imagef((128, 768, 4)), arg=1, src=()),
                      UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 1, 1, 1, 1, 3, 1, 4, 4, 130, 4, 258), strides=(0, 0, 0, 0, 0, 4, 0, 1, 0, 3072, 0, 12), offset=-3084, mask=((0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 3), (0, 1), (0, 4), (0, 4), (1, 129), (0, 4), (1, 257)), contiguous=False), View(shape=(1, 64, 128, 1, 1, 8, 4, 3, 4, 3, 3), strides=(0, 2064, 2, 0, 0, 0, 0, 2146560, 536640, 135192, 259), offset=0, mask=None, contiguous=False))), src=()),)),
                    UOp(Ops.CAST, dtypes.float, arg=None, src=(
                      UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                        UOp(Ops.DEFINE_GLOBAL, dtypes.imagef((8, 108, 4)), arg=2, src=()),
                        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 64, 128, 1, 1, 8, 4, 3, 4, 3, 3), strides=(0, 0, 0, 0, 0, 432, 1, 48, 4, 144, 16), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),)),)),
              UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=3, src=()),
                UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 64, 128, 1, 1, 8, 4, 1, 1, 1, 1), strides=(0, 0, 0, 0, 0, 4, 1, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),
            x19:=UOp(Ops.CONST, dtypes.float, arg=0.0, src=(
              x20:=UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 64, 128, 1, 1, 8, 4, 1, 1, 1, 1), strides=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),
          UOp(Ops.MUL, dtypes.float, arg=None, src=(
            UOp(Ops.MAX, dtypes.float, arg=None, src=(
              UOp(Ops.ADD, dtypes.float, arg=None, src=(
                UOp(Ops.CONST, dtypes.float, arg=1.0, src=(
                  x20,)),
                UOp(Ops.MUL, dtypes.float, arg=None, src=(
                  UOp(Ops.EXP2, dtypes.float, arg=None, src=(
                    UOp(Ops.MUL, dtypes.float, arg=None, src=(
                      x5,
                      UOp(Ops.CONST, dtypes.float, arg=1.4426950408889634, src=(
                        x20,)),)),)),
                  x29:=UOp(Ops.CONST, dtypes.float, arg=-1.0, src=(
                    x20,)),)),)),
              x19,)),
            x29,)),)),)),))

    opts = [Opt(op=OptOps.UPCAST, axis=3, arg=4), Opt(op=OptOps.UNROLL, axis=1, arg=4), Opt(op=OptOps.UPCAST, axis=1, arg=4), Opt(op=OptOps.NOLOCALS, axis=None, arg=None)]
    kernel = Kernel(ast)

    kernel.apply_opts(opts)

    p = kernel.to_program()
    print(p.src)

  def test_const_idx(self):
    Device.DEFAULT = "GPU"

    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.imagef((10, 128, 4)), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 10, 512), strides=(0, 512, 1), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.CAST, dtypes.float, arg=None, src=(
          UOp(Ops.ADD, dtypes.float, arg=None, src=(
            UOp(Ops.ADD, dtypes.float, arg=None, src=(
              UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                UOp(Ops.DEFINE_GLOBAL, dtypes.imagef((1, 128, 4)), arg=1, src=()),
                UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 10, 512), strides=(0, 0, 1), offset=0, mask=((0, 1), (0, 1), (0, 512)), contiguous=False),)), src=()),)),
              UOp(Ops.CAST, dtypes.float, arg=None, src=(
                UOp(Ops.ADD, dtypes.float, arg=None, src=(
                  UOp(Ops.ADD, dtypes.float, arg=None, src=(
                    UOp(Ops.ADD, dtypes.float, arg=None, src=(
                      UOp(Ops.ADD, dtypes.float, arg=None, src=(
                        UOp(Ops.ADD, dtypes.float, arg=None, src=(
                          UOp(Ops.ADD, dtypes.float, arg=None, src=(
                            UOp(Ops.ADD, dtypes.float, arg=None, src=(
                              UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                                x18:=UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=2, src=()),
                                UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 10, 512), strides=(0, 0, 1), offset=48128, mask=((0, 1), (1, 2), (0, 512)), contiguous=False),)), src=()),)),
                              UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                                x18,
                                UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 10, 512), strides=(0, 0, 1), offset=45568, mask=((0, 1), (2, 3), (0, 512)), contiguous=False),)), src=()),)),)),
                            UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                              x18,
                              UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 10, 512), strides=(0, 0, 1), offset=43008, mask=((0, 1), (3, 4), (0, 512)), contiguous=False),)), src=()),)),)),
                          UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                            x18,
                            UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 10, 512), strides=(0, 0, 1), offset=40448, mask=((0, 1), (4, 5), (0, 512)), contiguous=False),)), src=()),)),)),
                        UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                          x18,
                          UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 10, 512), strides=(0, 0, 1), offset=37888, mask=((0, 1), (5, 6), (0, 512)), contiguous=False),)), src=()),)),)),
                      UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                        x18,
                        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 10, 512), strides=(0, 0, 1), offset=35328, mask=((0, 1), (6, 7), (0, 512)), contiguous=False),)), src=()),)),)),
                    UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                      x18,
                      UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 10, 512), strides=(0, 0, 1), offset=32768, mask=((0, 1), (7, 8), (0, 512)), contiguous=False),)), src=()),)),)),
                  UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                    x18,
                    UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 10, 512), strides=(0, 0, 1), offset=30208, mask=((0, 1), (8, 9), (0, 512)), contiguous=False),)), src=()),)),)),)),)),
            UOp(Ops.LOAD, dtypes.float, arg=None, src=(
              UOp(Ops.DEFINE_GLOBAL, dtypes.imagef((1, 128, 4)), arg=3, src=()),
              UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 10, 512), strides=(0, 0, 1), offset=0, mask=((0, 1), (9, 10), (0, 512)), contiguous=False),)), src=()),)),)),)),)),))

    opts = [Opt(op=OptOps.UPCAST, axis=1, arg=4), Opt(op=OptOps.NOLOCALS, axis=None, arg=None)]
    kernel = Kernel(ast)

    kernel.apply_opts(opts)

    p = kernel.to_program()
    # ((idx1<1)?read_imagef(data1, smp, (int2)(idx0,0)):(float4)(0.0f,0.0f,0.0f,0.0f))
    print(p.src)

if __name__ == '__main__':
  unittest.main()
