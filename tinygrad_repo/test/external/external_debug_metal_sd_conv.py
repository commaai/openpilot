# ruff: noqa: E501
from tinygrad.codegen.kernel import Kernel, Opt, OptOps
from tinygrad.dtype import dtypes
from tinygrad.engine.realize import CompiledRunner
from tinygrad.engine.search import bufs_from_lin
from tinygrad.ops import UOp, Ops
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View

ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
  UOp(Ops.STORE, dtypes.void, arg=None, src=(
    UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(), arg=0, src=()),
    UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(2, 1, 1280, 8, 8, 1, 1, 1), strides=(81920, 0, 64, 8, 1, 0, 0, 0), offset=0, mask=None, contiguous=True),)), src=()),
    UOp(Ops.ADD, dtypes.half, arg=None, src=(
      UOp(Ops.ADD, dtypes.half, arg=None, src=(
        UOp(Ops.CAST, dtypes.half, arg=None, src=(
          UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (5, 6, 7)), src=(
            UOp(Ops.CAST, dtypes.float, arg=None, src=(
              UOp(Ops.MUL, dtypes.half, arg=None, src=(
                UOp(Ops.LOAD, dtypes.half, arg=None, src=(
                  UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(), arg=1, src=()),
                  UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 2, 1, 2560, 4, 10, 4, 10), strides=(0, 163840, 0, 64, 0, 8, 0, 1), offset=-9, mask=((0, 1), (0, 2), (0, 1), (0, 2560), (0, 4), (1, 9), (0, 4), (1, 9)), contiguous=False), View(shape=(2, 1, 1280, 8, 8, 2560, 3, 3), strides=(4096000, 0, 0, 40, 1, 1600, 440, 11), offset=0, mask=None, contiguous=False))), src=()),)),
                UOp(Ops.LOAD, dtypes.half, arg=None, src=(
                  UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(), arg=2, src=()),
                  UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(2, 1, 1280, 8, 8, 2560, 3, 3), strides=(0, 0, 23040, 0, 0, 9, 3, 1), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),)),)),
        UOp(Ops.LOAD, dtypes.half, arg=None, src=(
          UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(), arg=3, src=()),
          x17:=UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(2, 1, 1280, 8, 8, 1, 1, 1), strides=(0, 0, 1, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),
      UOp(Ops.LOAD, dtypes.half, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(), arg=4, src=()),
         x17,)),)),)),))
opts = [Opt(op=OptOps.UPCAST, axis=3, arg=4), Opt(op=OptOps.UPCAST, axis=1, arg=4), Opt(op=OptOps.UNROLL, axis=2, arg=0), Opt(op=OptOps.UNROLL, axis=1, arg=0), Opt(op=OptOps.LOCAL, axis=1, arg=8), Opt(op=OptOps.LOCAL, axis=2, arg=8), Opt(op=OptOps.LOCAL, axis=2, arg=2)]

k = Kernel(ast)
for opt in opts: k.apply_opt(opt)
bufs = bufs_from_lin(k)

prg = CompiledRunner(k.to_program())

for i in range(10):
  speed = prg(bufs, var_vals={}, wait=True)
  print(f"kernel time: {speed*1e3:.2f} ms")

# on M1 Max
# 11ms before block 9b0859d71780fef5cf3831e317f74e53f2483229
# 15ms after block cbcc1c20eb09a1342f6581cfbb99632bade982a8