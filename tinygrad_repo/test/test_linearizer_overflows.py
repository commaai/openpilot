# ruff: noqa: E501
import unittest
from tinygrad import dtypes, Device
from tinygrad.helpers import CI
from tinygrad.codegen.kernel import Kernel
from tinygrad.engine.search import Opt, OptOps, bufs_from_lin
from extra.optimization.helpers import time_linearizer

# stuff needed to unpack a kernel
from tinygrad.uop.ops import UOp, Ops
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View

def _test_overflow(ast, opts):
  lin = Kernel(ast)
  lin.apply_opts(opts)
  lin.linearize()
  bufs = bufs_from_lin(lin)
  print(bufs)
  time_linearizer(lin, bufs)

# NOTE: if you want these to trigger, set launch bounds on HIP kernels
@unittest.skip("unneeded without launch bounds")
class TestLinearizerOverflow(unittest.TestCase):
  def test_overflow_1(self):
    ast = UOp(Ops.SINK, None, arg=None, src=(
      UOp(Ops.STORE, None, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(51380224), arg=0, src=()),
        UOp(Ops.VIEW, None, arg=ShapeTracker(views=(View(shape=(64, 1, 64, 112, 112, 1, 1, 1), strides=(802816, 0, 12544, 112, 1, 0, 0, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.MAX, dtypes.float, arg=None, src=(
          UOp(Ops.ADD, dtypes.float, arg=None, src=(
            UOp(Ops.MUL, dtypes.float, arg=None, src=(
              UOp(Ops.MUL, dtypes.float, arg=None, src=(
                UOp(Ops.ADD, dtypes.float, arg=None, src=(
                  UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (7, 6, 5)), src=(
                    UOp(Ops.MUL, dtypes.float, arg=None, src=(
                      UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                        UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(9633792), arg=1, src=()),
                        UOp(Ops.VIEW, None, arg=ShapeTracker(views=(View(shape=(1, 64, 1, 3, 8, 230, 8, 230), strides=(0, 150528, 0, 50176, 0, 224, 0, 1), offset=-675, mask=((0, 1), (0, 64), (0, 1), (0, 3), (0, 8), (3, 227), (0, 8), (3, 227)), contiguous=False), View(shape=(64, 1, 64, 112, 112, 3, 7, 7), strides=(10156800, 0, 0, 3680, 2, 3385600, 425040, 231), offset=0, mask=None, contiguous=False))), src=()),)),
                      UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                        UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(9408), arg=2, src=()),
                        UOp(Ops.VIEW, None, arg=ShapeTracker(views=(View(shape=(64, 1, 64, 112, 112, 3, 7, 7), strides=(0, 0, 147, 0, 0, 49, 7, 1), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),
                  x16:=UOp(Ops.CONST, dtypes.float, arg=0.0, src=(
                    x17:=UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(64, 1, 64, 112, 112, 1, 1, 1), strides=(0, 0, 0, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),
                UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                  UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(64), arg=3, src=()),
                  x20:=UOp(Ops.VIEW, None, arg=ShapeTracker(views=(View(shape=(64, 1, 64, 112, 112, 1, 1, 1), strides=(0, 0, 1, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),
              UOp(Ops.SQRT, dtypes.float, arg=None, src=(
                UOp(Ops.MUL, dtypes.float, arg=None, src=(
                  x23:=UOp(Ops.CONST, dtypes.float, arg=1.0, src=(
                     x17,)),
                  UOp(Ops.RECIP, dtypes.float, arg=None, src=(
                    UOp(Ops.ADD, dtypes.float, arg=None, src=(
                       x23,
                      UOp(Ops.CONST, dtypes.float, arg=1e-05, src=(
                         x17,)),)),)),)),)),)),
            UOp(Ops.LOAD, dtypes.float, arg=None, src=(
              UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(64), arg=4, src=()),
               x20,)),)),
           x16,)),)),))
    opts = [Opt(op=OptOps.LOCAL, axis=3, arg=16), Opt(op=OptOps.LOCAL, axis=2, arg=16), Opt(op=OptOps.UPCAST, axis=1, arg=4), Opt(op=OptOps.UPCAST, axis=0, arg=4), Opt(op=OptOps.UPCAST, axis=2, arg=0)]
    _test_overflow(ast, opts)

  # From BEAM on hlb_cifar.py
  def test_overflow_2(self):
    ast = UOp(Ops.SINK, None, arg=None, src=(
      UOp(Ops.STORE, None, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(33554432), arg=0, src=()),
        UOp(Ops.VIEW, None, arg=ShapeTracker(views=(View(shape=(512, 1, 64, 32, 32, 1, 1, 1), strides=(65536, 0, 1024, 32, 1, 0, 0, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (7, 6, 5)), src=(
          UOp(Ops.MUL, dtypes.float, arg=None, src=(
            UOp(Ops.LOAD, dtypes.float, arg=None, src=(
              UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(16777216), arg=1, src=()),
              UOp(Ops.VIEW, None, arg=ShapeTracker(views=(View(shape=(1, 512, 1, 32, 4, 34, 4, 34), strides=(0, 32768, 0, 1024, 0, 32, 0, 1), offset=-33, mask=((0, 1), (0, 512), (0, 1), (0, 32), (0, 4), (1, 33), (0, 4), (1, 33)), contiguous=False), View(shape=(512, 1, 64, 32, 32, 32, 3, 3), strides=(591872, 0, 0, 136, 1, 18496, 4760, 35), offset=0, mask=None, contiguous=False))), src=()),)),
            UOp(Ops.LOAD, dtypes.float, arg=None, src=(
              UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(18432), arg=2, src=()),
              UOp(Ops.VIEW, None, arg=ShapeTracker(views=(View(shape=(512, 1, 64, 32, 32, 32, 3, 3), strides=(0, 0, 288, 0, 0, 9, 3, 1), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),)),))
    opts = [Opt(op=OptOps.LOCAL, axis=3, arg=16), Opt(op=OptOps.LOCAL, axis=2, arg=4), Opt(op=OptOps.UPCAST, axis=1, arg=4), Opt(op=OptOps.UPCAST, axis=2, arg=0), Opt(op=OptOps.UPCAST, axis=1, arg=4), Opt(op=OptOps.UNROLL, axis=0, arg=0)]
    _test_overflow(ast, opts)

  # from BEAM on default simple_conv.py (which is quite large):
  def test_overflow_3(self):
    ast = UOp(Ops.SINK, None, arg=None, src=(
      UOp(Ops.STORE, None, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(33554432), arg=0, src=()),
        UOp(Ops.VIEW, None, arg=ShapeTracker(views=(View(shape=(16, 1, 128, 128, 128, 1, 1, 1), strides=(2097152, 0, 16384, 128, 1, 0, 0, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (7, 6, 5)), src=(
          UOp(Ops.MUL, dtypes.float, arg=None, src=(
            UOp(Ops.LOAD, dtypes.float, arg=None, src=(
              UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(33554432), arg=1, src=()),
              UOp(Ops.VIEW, None, arg=ShapeTracker(views=(View(shape=(1, 16, 1, 128, 4, 130, 4, 130), strides=(0, 2097152, 0, 16384, 0, 128, 0, 1), offset=-129, mask=((0, 1), (0, 16), (0, 1), (0, 128), (0, 4), (1, 129), (0, 4), (1, 129)), contiguous=False), View(shape=(16, 1, 128, 128, 128, 128, 3, 3), strides=(34611200, 0, 0, 520, 1, 270400, 68120, 131), offset=0, mask=None, contiguous=False))), src=()),)),
            UOp(Ops.LOAD, dtypes.float, arg=None, src=(
              UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(147456), arg=2, src=()),
              UOp(Ops.VIEW, None, arg=ShapeTracker(views=(View(shape=(16, 1, 128, 128, 128, 128, 3, 3), strides=(0, 0, 1152, 0, 0, 9, 3, 1), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),)),))
    opts = [Opt(op=OptOps.LOCAL, axis=3, arg=16), Opt(op=OptOps.LOCAL, axis=2, arg=8), Opt(op=OptOps.UPCAST, axis=1, arg=4), Opt(op=OptOps.UPCAST, axis=3, arg=0), Opt(op=OptOps.UPCAST, axis=1, arg=2), Opt(op=OptOps.UPCAST, axis=2, arg=2)]
    _test_overflow(ast, opts)

  # from BEAM on BS=4 simple_conv.py:
  def test_overflow_4(self):
    ast = UOp(Ops.SINK, None, arg=None, src=(
      UOp(Ops.STORE, None, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(8388608), arg=0, src=()),
        UOp(Ops.VIEW, None, arg=ShapeTracker(views=(View(shape=(4, 1, 128, 128, 128, 1, 1, 1), strides=(2097152, 0, 16384, 128, 1, 0, 0, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (7, 6, 5)), src=(
          UOp(Ops.MUL, dtypes.float, arg=None, src=(
            UOp(Ops.LOAD, dtypes.float, arg=None, src=(
              UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(8388608), arg=1, src=()),
              UOp(Ops.VIEW, None, arg=ShapeTracker(views=(View(shape=(1, 4, 1, 128, 4, 130, 4, 130), strides=(0, 2097152, 0, 16384, 0, 128, 0, 1), offset=-129, mask=((0, 1), (0, 4), (0, 1), (0, 128), (0, 4), (1, 129), (0, 4), (1, 129)), contiguous=False), View(shape=(4, 1, 128, 128, 128, 128, 3, 3), strides=(34611200, 0, 0, 520, 1, 270400, 68120, 131), offset=0, mask=None, contiguous=False))), src=()),)),
            UOp(Ops.LOAD, dtypes.float, arg=None, src=(
              UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(147456), arg=2, src=()),
              UOp(Ops.VIEW, None, arg=ShapeTracker(views=(View(shape=(4, 1, 128, 128, 128, 128, 3, 3), strides=(0, 0, 1152, 0, 0, 9, 3, 1), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),)),))
    opts = [Opt(op=OptOps.UPCAST, axis=3, arg=4), Opt(op=OptOps.LOCAL, axis=3, arg=16), Opt(op=OptOps.UPCAST, axis=1, arg=4), Opt(op=OptOps.LOCAL, axis=2, arg=4), Opt(op=OptOps.UPCAST, axis=1, arg=2), Opt(op=OptOps.UPCAST, axis=2, arg=4)]
    _test_overflow(ast, opts)

  # from BEAM on BS=2 simple_conv.py:
  def test_overflow_5(self):
    ast = UOp(Ops.SINK, None, arg=None, src=(
      UOp(Ops.STORE, None, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(4194304), arg=0, src=()),
        UOp(Ops.VIEW, None, arg=ShapeTracker(views=(View(shape=(2, 1, 128, 128, 128, 1, 1, 1), strides=(2097152, 0, 16384, 128, 1, 0, 0, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (7, 6, 5)), src=(
          UOp(Ops.MUL, dtypes.float, arg=None, src=(
            UOp(Ops.LOAD, dtypes.float, arg=None, src=(
              UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(4194304), arg=1, src=()),
              UOp(Ops.VIEW, None, arg=ShapeTracker(views=(View(shape=(1, 2, 1, 128, 4, 130, 4, 130), strides=(0, 2097152, 0, 16384, 0, 128, 0, 1), offset=-129, mask=((0, 1), (0, 2), (0, 1), (0, 128), (0, 4), (1, 129), (0, 4), (1, 129)), contiguous=False), View(shape=(2, 1, 128, 128, 128, 128, 3, 3), strides=(34611200, 0, 0, 520, 1, 270400, 68120, 131), offset=0, mask=None, contiguous=False))), src=()),)),
            UOp(Ops.LOAD, dtypes.float, arg=None, src=(
              UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(147456), arg=2, src=()),
              UOp(Ops.VIEW, None, arg=ShapeTracker(views=(View(shape=(2, 1, 128, 128, 128, 128, 3, 3), strides=(0, 0, 1152, 0, 0, 9, 3, 1), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),)),))
    opts = [Opt(op=OptOps.LOCAL, axis=3, arg=16), Opt(op=OptOps.UPCAST, axis=1, arg=4), Opt(op=OptOps.UPCAST, axis=3, arg=0), Opt(op=OptOps.LOCAL, axis=2, arg=2), Opt(op=OptOps.UPCAST, axis=1, arg=2), Opt(op=OptOps.UPCAST, axis=2, arg=2)]
    _test_overflow(ast, opts)

  # from BEAM on BS=3 simple_conv.py:
  def test_overflow_6(self):
    ast = UOp(Ops.SINK, None, arg=None, src=(
      UOp(Ops.STORE, None, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(6291456), arg=0, src=()),
        UOp(Ops.VIEW, None, arg=ShapeTracker(views=(View(shape=(3, 1, 128, 128, 128, 1, 1, 1), strides=(2097152, 0, 16384, 128, 1, 0, 0, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (7, 6, 5)), src=(
          UOp(Ops.MUL, dtypes.float, arg=None, src=(
            UOp(Ops.LOAD, dtypes.float, arg=None, src=(
              UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(6291456), arg=1, src=()),
              UOp(Ops.VIEW, None, arg=ShapeTracker(views=(View(shape=(1, 3, 1, 128, 4, 130, 4, 130), strides=(0, 2097152, 0, 16384, 0, 128, 0, 1), offset=-129, mask=((0, 1), (0, 3), (0, 1), (0, 128), (0, 4), (1, 129), (0, 4), (1, 129)), contiguous=False), View(shape=(3, 1, 128, 128, 128, 128, 3, 3), strides=(34611200, 0, 0, 520, 1, 270400, 68120, 131), offset=0, mask=None, contiguous=False))), src=()),)),
            UOp(Ops.LOAD, dtypes.float, arg=None, src=(
              UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(147456), arg=2, src=()),
              UOp(Ops.VIEW, None, arg=ShapeTracker(views=(View(shape=(3, 1, 128, 128, 128, 128, 3, 3), strides=(0, 0, 1152, 0, 0, 9, 3, 1), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),)),))
    opts = [Opt(op=OptOps.LOCAL, axis=3, arg=16), Opt(op=OptOps.UPCAST, axis=3, arg=0), Opt(op=OptOps.UPCAST, axis=1, arg=4), Opt(op=OptOps.LOCAL, axis=2, arg=8), Opt(op=OptOps.UPCAST, axis=1, arg=2), Opt(op=OptOps.UPCAST, axis=3, arg=2)]
    _test_overflow(ast, opts)

  # from BEAM on BS=3 simple_conv.py: (alt)
  def test_overflow_7(self):
    ast = UOp(Ops.SINK, None, arg=None, src=(
      UOp(Ops.STORE, None, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(6291456), arg=0, src=()),
        UOp(Ops.VIEW, None, arg=ShapeTracker(views=(View(shape=(3, 1, 128, 128, 128, 1, 1, 1), strides=(2097152, 0, 16384, 128, 1, 0, 0, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (7, 6, 5)), src=(
          UOp(Ops.MUL, dtypes.float, arg=None, src=(
            UOp(Ops.LOAD, dtypes.float, arg=None, src=(
              UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(6291456), arg=1, src=()),
              UOp(Ops.VIEW, None, arg=ShapeTracker(views=(View(shape=(1, 3, 1, 128, 4, 130, 4, 130), strides=(0, 2097152, 0, 16384, 0, 128, 0, 1), offset=-129, mask=((0, 1), (0, 3), (0, 1), (0, 128), (0, 4), (1, 129), (0, 4), (1, 129)), contiguous=False), View(shape=(3, 1, 128, 128, 128, 128, 3, 3), strides=(34611200, 0, 0, 520, 1, 270400, 68120, 131), offset=0, mask=None, contiguous=False))), src=()),)),
            UOp(Ops.LOAD, dtypes.float, arg=None, src=(
              UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(147456), arg=2, src=()),
              UOp(Ops.VIEW, None, arg=ShapeTracker(views=(View(shape=(3, 1, 128, 128, 128, 128, 3, 3), strides=(0, 0, 1152, 0, 0, 9, 3, 1), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),)),))
    opts = [Opt(op=OptOps.UPCAST, axis=3, arg=4), Opt(op=OptOps.LOCAL, axis=3, arg=16), Opt(op=OptOps.UPCAST, axis=1, arg=4), Opt(op=OptOps.LOCAL, axis=2, arg=8), Opt(op=OptOps.UPCAST, axis=1, arg=2), Opt(op=OptOps.UPCAST, axis=2, arg=4)]
    _test_overflow(ast, opts)

@unittest.skipIf(Device.DEFAULT not in {"GPU", "HSA", "CUDA", "METAL"}, "only backends with locals")
@unittest.skipIf(CI, "slow")
class TestLinearizerOverflowAlt(unittest.TestCase):
  def test_overflow_1(self):
    BS = 2
    g0, g1, g2 = [UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=i) for i in range(3)]
    in_st_1 = ShapeTracker(views=(View(shape=(1, BS, 1, 3, 8, 230, 8, 230), strides=(0, 150528, 0, 50176, 0, 224, 0, 1), offset=-675, mask=((0, 1), (0, BS), (0, 1), (0, 3), (0, 8), (3, 227), (0, 8), (3, 227)), contiguous=False),
                                  View(shape=(BS, 1, 64, 112, 112, 3, 7, 7), strides=(10156800, 0, 0, 3680, 2, 3385600, 425040, 231), offset=0, mask=None, contiguous=False))).to_uop()
    in_st_2 = ShapeTracker(views=(View(shape=(BS, 1, 64, 112, 112, 3, 7, 7), strides=(0, 0, 147, 0, 0, 49, 7, 1), offset=0, mask=None, contiguous=False),)).to_uop()
    ot_st = ShapeTracker(views=(View(shape=(BS, 1, 64, 112, 112, 1, 1, 1), strides=(802816, 0, 12544, 112, 1, 0, 0, 0), offset=0, mask=None, contiguous=True),)).to_uop()
    prod = UOp(Ops.LOAD, dtypes.float, (g1.view(in_st_1.arg),)) * UOp(Ops.LOAD, dtypes.float, (g2.view(in_st_2.arg),))
    store = UOp(Ops.STORE, src=(g0.view(ot_st.arg), UOp(Ops.REDUCE_AXIS, dtypes.float, (prod,), (Ops.ADD, (7, 6, 5)))))
    ast = UOp(Ops.SINK, src=(store,))
    opts = [Opt(op=OptOps.LOCAL, axis=3, arg=16), Opt(op=OptOps.LOCAL, axis=2, arg=2), Opt(op=OptOps.UPCAST, axis=0, arg=2)]
    _test_overflow(ast, opts)
  def test_overflow_2(self):
    BS = 2
    g0, g1, g2 = [UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=i) for i in range(3)]
    in_st_1 = ShapeTracker(views=(View(shape=(1, BS, 1, 3, 8, 230, 8, 230), strides=(0, 150528, 0, 50176, 0, 224, 0, 1), offset=-675, mask=((0, 1), (0, BS), (0, 1), (0, 3), (0, 8), (3, 227), (0, 8), (3, 227)), contiguous=False),
                                  View(shape=(BS, 1, 64, 112, 112, 3, 7, 7), strides=(10156800, 0, 0, 3680, 2, 3385600, 425040, 231), offset=0, mask=None, contiguous=False))).to_uop()
    in_st_2 = ShapeTracker(views=(View(shape=(BS, 1, 64, 112, 112, 3, 7, 7), strides=(0, 0, 147, 0, 0, 49, 7, 1), offset=0, mask=None, contiguous=False),)).to_uop()
    ot_st = ShapeTracker(views=(View(shape=(BS, 1, 64, 112, 112, 1, 1, 1), strides=(802816, 0, 12544, 112, 1, 0, 0, 0), offset=0, mask=None, contiguous=True),)).to_uop()
    prod = UOp(Ops.LOAD, dtypes.float, (g1.view(in_st_1.arg),)) * UOp(Ops.LOAD, dtypes.float, (g2.view(in_st_2.arg),))
    store = UOp(Ops.STORE, src=(g0.view(ot_st.arg), UOp(Ops.REDUCE_AXIS, dtypes.float, (prod,), (Ops.ADD, (7, 6, 5)))))
    ast = UOp(Ops.SINK, src=(store,))
    opts = [Opt(op=OptOps.LOCAL, axis=3, arg=16), Opt(op=OptOps.UPCAST, axis=1, arg=4), Opt(op=OptOps.LOCAL, axis=2, arg=16), Opt(op=OptOps.UPCAST, axis=4, arg=4), Opt(op=OptOps.UPCAST, axis=1, arg=2), Opt(op=OptOps.UPCAST, axis=5, arg=2)]
    _test_overflow(ast, opts)

if __name__ == '__main__':
  unittest.main()
