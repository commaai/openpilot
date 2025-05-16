# ruff: noqa: E501
import unittest, random
import numpy as np
from tinygrad.codegen.kernel import Kernel, KernelOptError
from tinygrad.device import is_dtype_supported
from tinygrad.ops import UOp, Ops
from tinygrad.engine.search import Opt, OptOps
from tinygrad import Device, dtypes, Tensor
from tinygrad.helpers import CI, Context
from test.external.fuzz_linearizer import compare_linearizer
from test.helpers import ast_const

from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View

def helper_test_lin(lin: Kernel, opts, failed_platforms, rtol=1e-2, atol=1e-2):
  if any(b.dtype.base == dtypes.half for b in lin.membufs) and not is_dtype_supported(dtypes.half): return
  if any(b.dtype.base == dtypes.bfloat16 for b in lin.membufs) and not is_dtype_supported(dtypes.bfloat16): return

  try:
    lin.apply_opts(opts)
  except KernelOptError:
    # it's considered fixed if we invalidated the opts
    assert Device.DEFAULT not in failed_platforms, f"unexpected success on {Device.DEFAULT}"
    return

  compare_result = compare_linearizer(lin, rtol=rtol, atol=atol)
  if compare_result[0] in ["PASS", "KernelOptError"]:
    # it's considered fixed if we invalidated the opts
    assert Device.DEFAULT not in failed_platforms, f"unexpected success on {Device.DEFAULT}"
  else:
    assert Device.DEFAULT in failed_platforms, f"failed on {Device.DEFAULT} with {compare_result[0]}"
  return lin

@unittest.skipIf(CI and Device.DEFAULT in {"CUDA", "NV"}, "failed on CUDA CI")
class TestLinearizerFailures(unittest.TestCase):
  def setUp(self):
    random.seed(42)
    np.random.seed(42)
    Tensor.manual_seed(42)

  def test_failure_1(self):
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(32, 16, 1), strides=(16, 1, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.ADD, dtypes.float, arg=None, src=(
          UOp(Ops.ADD, dtypes.float, arg=None, src=(
            UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (2,)), src=(
              UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=1, src=()),
                UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(32, 16, 16), strides=(16, 1, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),
            UOp(Ops.LOAD, dtypes.float, arg=None, src=(
              UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=2, src=()),
              UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(32, 16, 1), strides=(0, 1, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),
          UOp(Ops.LOAD, dtypes.float, arg=None, src=(
            UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=1, src=()),
            UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(32, 16, 1), strides=(16, 1, 0), offset=0, mask=None, contiguous=True),)), src=()),)),)),)),))
    helper_test_lin(Kernel(ast), [], failed_platforms=[])

  def test_failure_2(self):
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(32, 2, 37, 9, 1, 1), strides=(666, 333, 9, 1, 0, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.MAX, (4, 5)), src=(
          UOp(Ops.LOAD, dtypes.float, arg=None, src=(
            UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=1, src=()),
            UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(32, 2, 111, 27), strides=(6160, 3080, 28, 1), offset=0, mask=((0, 32), (0, 2), (0, 110), (0, 27)), contiguous=False), View(shape=(32, 2, 37, 9, 2, 2), strides=(5994, 2997, 81, 3, 27, 1), offset=0, mask=None, contiguous=False))), src=()),)),)),)),))
    opts = [Opt(op=OptOps.LOCAL, axis=0, arg=32)]
    helper_test_lin(Kernel(ast), opts, failed_platforms=[])

  def test_failure_3(self):
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(32, 8, 16, 1), strides=(128, 16, 1, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (3,)), src=(
          UOp(Ops.LOAD, dtypes.float, arg=None, src=(
            UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=1, src=()),
            UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(32, 8, 16, 16), strides=(2048, 256, 16, 1), offset=0, mask=None, contiguous=True),)), src=()),)),)),)),))
    opts = [Opt(op=OptOps.GROUP, axis=0, arg=4), Opt(op=OptOps.UPCAST, axis=0, arg=4), Opt(op=OptOps.UPCAST, axis=0, arg=2), Opt(op=OptOps.UNROLL, axis=1, arg=0), Opt(op=OptOps.UPCAST, axis=0, arg=4), Opt(op=OptOps.LOCAL, axis=0, arg=2), Opt(op=OptOps.LOCAL, axis=0, arg=2), Opt(op=OptOps.UPCAST, axis=1, arg=0), Opt(op=OptOps.LOCAL, axis=0, arg=32)]
    # METAL: AssertionError: Error Domain=AGXMetalG13X Code=3 "Threadgroup memory size (65536) exceeds the maximum threadgroup memory allowed (32768)" UserInfo={NSLocalizedDescription=Threadgroup memory size (65536) exceeds the maximum threadgroup memory allowed (32768)}
    helper_test_lin(Kernel(ast), opts, failed_platforms=[])

  def test_failure_5(self):
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 1, 1, 1, 1, 1, 1, 1), strides=(0, 0, 0, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (0, 2, 4, 6)), src=(
          UOp(Ops.ADD, dtypes.float, arg=None, src=(
            x5:=UOp(Ops.MUL, dtypes.float, arg=None, src=(
              UOp(Ops.ADD, dtypes.float, arg=None, src=(
                ast_const(dtypes.float, 0.1464405059814453, st_src=(
                  UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(2, 1, 4, 1, 3, 1, 4, 1), strides=(0, 0, 0, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),
                ast_const(dtypes.float, 1.0, st_src=(
                  UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(2, 1, 4, 1, 3, 1, 4, 1), strides=(0, 0, 0, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),
              UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=1, src=()),
                UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(2, 1, 4, 1, 3, 1, 4, 1), strides=(0, 0, 0, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),
             x5,)),)),)),))
    opts = [Opt(op=OptOps.UNROLL, axis=0, arg=4), Opt(op=OptOps.UNROLL, axis=0, arg=0)]
    # EXEC_ERROR, it has no global_size
    helper_test_lin(Kernel(ast), opts, failed_platforms=[])

  def test_failure_6(self):
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(10, 1), strides=(1, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.ADD, dtypes.int, arg=None, src=(
          UOp(Ops.REDUCE_AXIS, dtypes.int, arg=(Ops.ADD, (1,)), src=(
            ast_const(dtypes.int, -1, st_src=(
              UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(11, 19), strides=(0, 0), offset=0, mask=((0, 11), (9, 19)), contiguous=False), View(shape=(10, 10), strides=(1, 20), offset=0, mask=None, contiguous=False))), src=()),)),)),
          ast_const(dtypes.int, 10, st_src=(
            UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(10, 1), strides=(0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),))
    opts = [Opt(op=OptOps.UPCAST, axis=0, arg=2), Opt(op=OptOps.UPCAST, axis=0, arg=0)]
    # COMPILE FAILED, KeyError: Ops.CONST
    helper_test_lin(Kernel(ast), opts, failed_platforms=[])

  def test_failure_7(self):
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(512, 32, 1, 34, 1, 34), strides=(36992, 1156, 0, 34, 0, 1), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (2, 4)), src=(
          UOp(Ops.LOAD, dtypes.float, arg=None, src=(
            UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=1, src=()),
            UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(512, 32, 6, 8, 4, 6, 8, 4), strides=(2048, 64, 6291456, 8, 0, 1048576, 1, 0), offset=0, mask=((0, 512), (0, 32), (0, 6), (0, 8), (0, 1), (0, 6), (0, 8), (0, 1)), contiguous=False), View(shape=(512, 32, 6, 35, 6, 35), strides=(1179648, 36864, 6144, 192, 32, 1), offset=0, mask=((0, 512), (0, 32), (0, 6), (0, 32), (0, 6), (0, 32)), contiguous=False), View(shape=(512, 32, 238, 238), strides=(1411200, 44100, 210, 1), offset=0, mask=((0, 512), (0, 32), (0, 210), (0, 210)), contiguous=False), View(shape=(512, 32, 7, 34, 7, 34), strides=(1812608, 56644, 8092, 238, 34, 1), offset=0, mask=None, contiguous=True))), src=()),)),)),)),))
    opts = [Opt(op=OptOps.UPCAST, axis=0, arg=4)]
    # test/test_linearizer_failures.py Fatal Python error: Segmentation fault
    helper_test_lin(Kernel(ast), opts, failed_platforms=[])

  def test_failure_8(self):
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 1, 1), strides=(0, 0, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.SQRT, dtypes.float, arg=None, src=(
          UOp(Ops.RECIP, dtypes.float, arg=None, src=(
            UOp(Ops.ADD, dtypes.float, arg=None, src=(
              UOp(Ops.MUL, dtypes.float, arg=None, src=(
                UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (2,)), src=(
                  UOp(Ops.MUL, dtypes.float, arg=None, src=(
                    x9:=UOp(Ops.ADD, dtypes.float, arg=None, src=(
                      UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                        UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=1, src=()),
                        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 1, 4096), strides=(0, 0, 1), offset=0, mask=None, contiguous=True),)), src=()),)),
                      UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                        UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=2, src=()),
                        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 1, 4096), strides=(0, 0, 1), offset=0, mask=None, contiguous=True),)), src=()),)),)),
                     x9,)),)),
                ast_const(dtypes.float, 0.000244140625, st_src=(
                  UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 1, 1), strides=(0, 0, 0), offset=0, mask=None, contiguous=True),)), src=()),)),)),
              ast_const(dtypes.float, 1e-06, st_src=(
                UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 1, 1), strides=(0, 0, 0), offset=0, mask=None, contiguous=True),)), src=()),)),)),)),)),)),))
    opts = [Opt(op=OptOps.UNROLL, axis=0, arg=4), Opt(op=OptOps.UNROLL, axis=0, arg=4), Opt(op=OptOps.UNROLL, axis=0, arg=4), Opt(op=OptOps.UNROLL, axis=0, arg=4)]
    # fatal error: bracket nesting level exceeded maximum of 256
    # note: use -fbracket-depth=N to increase maximum nesting level
    helper_test_lin(Kernel(ast), opts, failed_platforms=[])

  def test_failure_9(self):
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 1, 1, 3, 1, 1, 1, 1, 5, 15, 5, 3, 4), strides=(0, 0, 0, 4500, 0, 0, 0, 0, 900, 60, 12, 4, 1), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (1,)), src=(
          UOp(Ops.MUL, dtypes.float, arg=None, src=(
            UOp(Ops.LOAD, dtypes.float, arg=None, src=(
              UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=1, src=()),
              UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 2, 1, 3, 1, 1, 1, 1, 5, 15, 5, 3, 4), strides=(0, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),
            UOp(Ops.LOAD, dtypes.float, arg=None, src=(
              UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=2, src=()),
              UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 2, 1, 3, 1, 1, 1, 1, 5, 15, 5, 3, 4), strides=(0, 4500, 0, 0, 0, 0, 0, 0, 900, 60, 12, 4, 1), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),)),))
    opts = [Opt(op=OptOps.UPCAST, axis=1, arg=2), Opt(op=OptOps.UPCAST, axis=0, arg=0), Opt(op=OptOps.PADTO, axis=0, arg=32)]
    helper_test_lin(Kernel(ast), opts, failed_platforms=[])

  def test_failure_10(self):
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 1, 1024, 1), strides=(0, 0, 1, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.ADD, dtypes.half, arg=None, src=(
          UOp(Ops.REDUCE_AXIS, dtypes.half, arg=(Ops.ADD, (3,)), src=(
            UOp(Ops.MUL, dtypes.half, arg=None, src=(
              UOp(Ops.LOAD, dtypes.half, arg=None, src=(
                UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(), arg=1, src=()),
                UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 1, 1024, 50257), strides=(0, 0, 0, 1), offset=0, mask=None, contiguous=False),)), src=()),)),
              UOp(Ops.LOAD, dtypes.half, arg=None, src=(
                UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(), arg=2, src=()),
                UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 1, 1024, 50257), strides=(0, 0, 1, 1024), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),
          UOp(Ops.LOAD, dtypes.half, arg=None, src=(
            UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(), arg=3, src=()),
            UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 1, 1024, 1), strides=(0, 0, 1, 0), offset=0, mask=None, contiguous=True),)), src=()),)),)),)),))
    helper_test_lin(Kernel(ast), [], failed_platforms=[])

  def test_failure_11(self):
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 64, 1, 1), strides=(0, 1, 0, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.RECIP, dtypes.float, arg=None, src=(
          UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (0, 2, 3)), src=(
            UOp(Ops.MUL, dtypes.float, arg=None, src=(
              UOp(Ops.MUL, dtypes.float, arg=None, src=(
                UOp(Ops.ADD, dtypes.float, arg=None, src=(
                  UOp(Ops.MAX, dtypes.float, arg=None, src=(
                    UOp(Ops.ADD, dtypes.float, arg=None, src=(
                      UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                        UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=1, src=()),
                        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(512, 64, 6, 6), strides=(2304, 36, 6, 1), offset=0, mask=None, contiguous=True),)), src=()),)),
                      UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                        UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=2, src=()),
                        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(512, 64, 6, 6), strides=(0, 1, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),
                    ast_const(dtypes.float, 0.0, st_src=(
                      UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(512, 64, 6, 6), strides=(0, 0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),
                  UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                    UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=3, src=()),
                    UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(512, 64, 6, 6), strides=(0, 1, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),
                ast_const(dtypes.float, 1.0, st_src=(
                  UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(512, 64, 6, 6), strides=(0, 0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),
              UOp(Ops.MUL, dtypes.float, arg=None, src=(
                UOp(Ops.MUL, dtypes.float, arg=None, src=(
                  UOp(Ops.ADD, dtypes.float, arg=None, src=(
                    ast_const(dtypes.float, 1.0, st_src=(
                      UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(512, 64, 3, 3, 2, 2), strides=(0, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False), View(shape=(512, 64, 3, 2, 3, 2), strides=(2304, 36, 12, 2, 4, 1), offset=0, mask=None, contiguous=False), View(shape=(512, 64, 6, 6), strides=(2304, 36, 6, 1), offset=0, mask=None, contiguous=True))), src=()),)),
                    UOp(Ops.CAST, dtypes.float, arg=None, src=(
                      UOp(Ops.CMPLT, dtypes.bool, arg=None, src=(
                        UOp(Ops.ADD, dtypes.float, arg=None, src=(
                          UOp(Ops.MUL, dtypes.float, arg=None, src=(
                            UOp(Ops.MUL, dtypes.float, arg=None, src=(
                              UOp(Ops.ADD, dtypes.float, arg=None, src=(
                                UOp(Ops.MAX, dtypes.float, arg=None, src=(
                                  UOp(Ops.ADD, dtypes.float, arg=None, src=(
                                    UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                                      UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=1, src=()),
                                      UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(512, 64, 6, 6), strides=(2304, 36, 6, 1), offset=0, mask=None, contiguous=True), View(shape=(512, 64, 3, 3, 2, 2), strides=(2304, 36, 12, 2, 6, 1), offset=0, mask=None, contiguous=False), View(shape=(512, 64, 3, 2, 3, 2), strides=(2304, 36, 12, 2, 4, 1), offset=0, mask=None, contiguous=False), View(shape=(512, 64, 6, 6), strides=(2304, 36, 6, 1), offset=0, mask=None, contiguous=True))), src=()),)),
                                    UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                                      UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=2, src=()),
                                      UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(512, 64, 6, 6), strides=(0, 1, 0, 0), offset=0, mask=None, contiguous=False), View(shape=(512, 64, 3, 3, 2, 2), strides=(2304, 36, 12, 2, 6, 1), offset=0, mask=None, contiguous=False), View(shape=(512, 64, 3, 2, 3, 2), strides=(2304, 36, 12, 2, 4, 1), offset=0, mask=None, contiguous=False), View(shape=(512, 64, 6, 6), strides=(2304, 36, 6, 1), offset=0, mask=None, contiguous=True))), src=()),)),)),
                                  x42:=ast_const(dtypes.float, 0.0, st_src=(
                                    UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(512, 64, 6, 6), strides=(0, 0, 0, 0), offset=0, mask=None, contiguous=False), View(shape=(512, 64, 3, 3, 2, 2), strides=(2304, 36, 12, 2, 6, 1), offset=0, mask=None, contiguous=False), View(shape=(512, 64, 3, 2, 3, 2), strides=(2304, 36, 12, 2, 4, 1), offset=0, mask=None, contiguous=False), View(shape=(512, 64, 6, 6), strides=(2304, 36, 6, 1), offset=0, mask=None, contiguous=True))), src=()),)),)),
                                UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                                  UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=3, src=()),
                                  UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(512, 64, 6, 6), strides=(0, 1, 0, 0), offset=0, mask=None, contiguous=False), View(shape=(512, 64, 3, 3, 2, 2), strides=(2304, 36, 12, 2, 6, 1), offset=0, mask=None, contiguous=False), View(shape=(512, 64, 3, 2, 3, 2), strides=(2304, 36, 12, 2, 4, 1), offset=0, mask=None, contiguous=False), View(shape=(512, 64, 6, 6), strides=(2304, 36, 6, 1), offset=0, mask=None, contiguous=True))), src=()),)),)),
                              ast_const(dtypes.float, 1.0, st_src=(
                                UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(512, 64, 6, 6), strides=(0, 0, 0, 0), offset=0, mask=None, contiguous=False), View(shape=(512, 64, 3, 3, 2, 2), strides=(2304, 36, 12, 2, 6, 1), offset=0, mask=None, contiguous=False), View(shape=(512, 64, 3, 2, 3, 2), strides=(2304, 36, 12, 2, 4, 1), offset=0, mask=None, contiguous=False), View(shape=(512, 64, 6, 6), strides=(2304, 36, 6, 1), offset=0, mask=None, contiguous=True))), src=()),)),)),
                            UOp(Ops.SQRT, dtypes.float, arg=None, src=(
                              UOp(Ops.CAST, dtypes.float, arg=None, src=(
                                UOp(Ops.RECIP, dtypes.float, arg=None, src=(
                                  UOp(Ops.ADD, dtypes.float, arg=None, src=(
                                    UOp(Ops.MUL, dtypes.float, arg=None, src=(
                                      UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                                        UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=4, src=()),
                                        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(64,), strides=(1,), offset=0, mask=None, contiguous=True), View(shape=(512, 64, 6, 6), strides=(0, 1, 0, 0), offset=0, mask=None, contiguous=False), View(shape=(512, 64, 3, 3, 2, 2), strides=(2304, 36, 12, 2, 6, 1), offset=0, mask=None, contiguous=False), View(shape=(512, 64, 3, 2, 3, 2), strides=(2304, 36, 12, 2, 4, 1), offset=0, mask=None, contiguous=False), View(shape=(512, 64, 6, 6), strides=(2304, 36, 6, 1), offset=0, mask=None, contiguous=True))), src=()),)),
                                      ast_const(dtypes.float, 5.425347222222222e-05, st_src=(
                                        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(64,), strides=(0,), offset=0, mask=None, contiguous=False), View(shape=(512, 64, 6, 6), strides=(0, 1, 0, 0), offset=0, mask=None, contiguous=False), View(shape=(512, 64, 3, 3, 2, 2), strides=(2304, 36, 12, 2, 6, 1), offset=0, mask=None, contiguous=False), View(shape=(512, 64, 3, 2, 3, 2), strides=(2304, 36, 12, 2, 4, 1), offset=0, mask=None, contiguous=False), View(shape=(512, 64, 6, 6), strides=(2304, 36, 6, 1), offset=0, mask=None, contiguous=True))), src=()),)),)),
                                    ast_const(dtypes.float, 1e-05, st_src=(
                                      UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(64,), strides=(0,), offset=0, mask=None, contiguous=False), View(shape=(512, 64, 6, 6), strides=(0, 1, 0, 0), offset=0, mask=None, contiguous=False), View(shape=(512, 64, 3, 3, 2, 2), strides=(2304, 36, 12, 2, 6, 1), offset=0, mask=None, contiguous=False), View(shape=(512, 64, 3, 2, 3, 2), strides=(2304, 36, 12, 2, 4, 1), offset=0, mask=None, contiguous=False), View(shape=(512, 64, 6, 6), strides=(2304, 36, 6, 1), offset=0, mask=None, contiguous=True))), src=()),)),)),)),)),)),)),
                           x42,)),
                        UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                          UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=5, src=()),
                          UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(512, 64, 3, 3, 2, 2), strides=(576, 9, 3, 1, 0, 0), offset=0, mask=None, contiguous=False), View(shape=(512, 64, 3, 2, 3, 2), strides=(2304, 36, 12, 2, 4, 1), offset=0, mask=None, contiguous=False), View(shape=(512, 64, 6, 6), strides=(2304, 36, 6, 1), offset=0, mask=None, contiguous=True))), src=()),)),)),)),)),
                  UOp(Ops.RECIP, dtypes.float, arg=None, src=(
                    UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                      UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=6, src=()),
                      UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(512, 64, 3, 3, 2, 2), strides=(576, 9, 3, 1, 0, 0), offset=0, mask=None, contiguous=False), View(shape=(512, 64, 3, 2, 3, 2), strides=(2304, 36, 12, 2, 4, 1), offset=0, mask=None, contiguous=False), View(shape=(512, 64, 6, 6), strides=(2304, 36, 6, 1), offset=0, mask=None, contiguous=True))), src=()),)),)),)),
                UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                  UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=7, src=()),
                  UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(512, 64, 3, 3, 2, 2), strides=(576, 9, 3, 1, 0, 0), offset=0, mask=None, contiguous=False), View(shape=(512, 64, 3, 2, 3, 2), strides=(2304, 36, 12, 2, 4, 1), offset=0, mask=None, contiguous=False), View(shape=(512, 64, 6, 6), strides=(2304, 36, 6, 1), offset=0, mask=None, contiguous=True))), src=()),)),)),)),)),)),)),))
    helper_test_lin(Kernel(ast), [], failed_platforms=[])

  def test_failure_12(self):
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 1, 1, 1, 1, 4, 1, 6, 1, 3), strides=(0, 0, 0, 0, 0, 18, 0, 3, 0, 1), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (0, 2, 4, 6)), src=(
          UOp(Ops.ADD, dtypes.float, arg=None, src=(
            x5:=UOp(Ops.MUL, dtypes.float, arg=None, src=(
              UOp(Ops.ADD, dtypes.float, arg=None, src=(
                UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                  UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=1, src=()),
                  UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(2, 1, 4, 1, 3, 4, 2, 6, 1, 3), strides=(0, 0, 0, 0, 0, 18, 0, 3, 0, 1), offset=0, mask=None, contiguous=False),)), src=()),)),
                ast_const(dtypes.float, 1.0, st_src=(
                  UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(2, 1, 4, 1, 3, 4, 2, 6, 1, 3), strides=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),
              UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=2, src=()),
                UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(2, 1, 4, 1, 3, 4, 2, 6, 1, 3), strides=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),
             x5,)),)),)),))
    opts = [Opt(op=OptOps.PADTO, axis=0, arg=32), Opt(op=OptOps.GROUP, axis=0, arg=4)]
    helper_test_lin(Kernel(ast), opts, failed_platforms=[])

  @unittest.skip("found implicit expand")
  def test_failure_12_multireduce(self):
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 1, 1, 1, 1, 4, 1, 6, 1, 3), strides=(0, 0, 0, 0, 0, 18, 0, 3, 0, 1), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (0, 2, 4, 8)), src=(
          UOp(Ops.ADD, dtypes.float, arg=None, src=(
            x5:=UOp(Ops.ADD, dtypes.float, arg=None, src=(
              x6:=UOp(Ops.MUL, dtypes.float, arg=None, src=(
                UOp(Ops.ADD, dtypes.float, arg=None, src=(
                  UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                    UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=1, src=()),
                    UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(2, 1, 4, 1, 3, 4, 2, 6, 1, 3), strides=(0, 0, 0, 0, 0, 18, 0, 3, 0, 1), offset=0, mask=None, contiguous=False),)), src=()),)),
                  ast_const(dtypes.float, 1.0, st_src=(
                    UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(2, 1, 4, 1, 3, 4, 2, 6, 1, 3), strides=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),
                UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                  UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=2, src=()),
                  UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(2, 1, 4, 1, 3, 4, 2, 6, 1, 3), strides=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),
               x6,)),
            UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (0, 2, 4, 8)), src=(
               x5,)),)),)),)),))
    opts = [Opt(op=OptOps.PADTO, axis=0, arg=32), Opt(op=OptOps.GROUP, axis=0, arg=4)]
    helper_test_lin(Kernel(ast), opts, failed_platforms=[])

  # both kernels are correct from a code standpoint, but generate different results due to precision errors (switching to float results in output matches)
  def test_failure_13(self):
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(2, 1, 384, 1), strides=(384, 0, 1, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.ADD, dtypes.half, arg=None, src=(
          UOp(Ops.REDUCE_AXIS, dtypes.half, arg=(Ops.ADD, (3,)), src=(
            UOp(Ops.MUL, dtypes.half, arg=None, src=(
              UOp(Ops.LOAD, dtypes.half, arg=None, src=(
                UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(), arg=1, src=()),
                UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(2, 1, 384, 51864), strides=(51864, 0, 0, 1), offset=0, mask=None, contiguous=False),)), src=()),)),
              UOp(Ops.LOAD, dtypes.half, arg=None, src=(
                UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(), arg=2, src=()),
                UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(2, 1, 384, 51864), strides=(0, 0, 1, 384), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),
          UOp(Ops.LOAD, dtypes.half, arg=None, src=(
            UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(), arg=3, src=()),
            UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(2, 1, 384, 1), strides=(0, 0, 1, 0), offset=19584, mask=None, contiguous=False),)), src=()),)),)),)),))
    opts = [Opt(op=OptOps.GROUP, axis=0, arg=4)]
    helper_test_lin(Kernel(ast), opts, failed_platforms=["METAL", "GPU", "CUDA"])

  def test_failure_14(self):
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 1, 1, 1, 1, 4, 1, 6, 1, 3), strides=(0, 0, 0, 0, 0, 18, 0, 3, 0, 1), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (0, 2, 4, 6)), src=(
          UOp(Ops.ADD, dtypes.float, arg=None, src=(
            x5:=UOp(Ops.MUL, dtypes.float, arg=None, src=(
              UOp(Ops.ADD, dtypes.float, arg=None, src=(
                UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                  UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=1, src=()),
                  UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(2, 1, 4, 1, 3, 4, 2, 6, 1, 3), strides=(0, 0, 0, 0, 0, 18, 0, 3, 0, 1), offset=0, mask=None, contiguous=False),)), src=()),)),
                ast_const(dtypes.float, 1.0, st_src=(
                  UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(2, 1, 4, 1, 3, 4, 2, 6, 1, 3), strides=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),
              UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=2, src=()),
                UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(2, 1, 4, 1, 3, 4, 2, 6, 1, 3), strides=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),
             x5,)),)),)),))
    opts = [Opt(op=OptOps.PADTO, axis=0, arg=32), Opt(op=OptOps.UPCAST, axis=0, arg=4), Opt(op=OptOps.UPCAST, axis=0, arg=4)]
    # COMPILE_ERROR on METAL in fuzz_linearizer: unused variables and undeclared variables
    helper_test_lin(Kernel(ast), opts, failed_platforms=[])

  def test_failure_15(self):
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 1, 112, 14, 14, 1, 1, 1), strides=(0, 0, 196, 14, 1, 0, 0, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.ADD, dtypes.float, arg=None, src=(
          UOp(Ops.MUL, dtypes.float, arg=None, src=(
            UOp(Ops.MUL, dtypes.float, arg=None, src=(
              UOp(Ops.ADD, dtypes.float, arg=None, src=(
                UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (5,)), src=(
                  UOp(Ops.MUL, dtypes.float, arg=None, src=(
                    UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                      UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=1, src=()),
                      UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 1, 112, 14, 14, 480, 1, 1), strides=(0, 0, 0, 14, 1, 196, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),
                    UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                      UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=2, src=()),
                      UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 1, 112, 14, 14, 480, 1, 1), strides=(0, 0, 480, 0, 0, 1, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),
                UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                  UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=3, src=()),
                  UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 1, 112, 14, 14, 1, 1, 1), strides=(0, 0, 1, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),
              UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=4, src=()),
                UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 1, 112, 14, 14, 1, 1, 1), strides=(0, 0, 1, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),
            UOp(Ops.SQRT, dtypes.float, arg=None, src=(
              UOp(Ops.RECIP, dtypes.float, arg=None, src=(
                UOp(Ops.ADD, dtypes.float, arg=None, src=(
                  UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                    UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=5, src=()),
                    UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 1, 112, 14, 14, 1, 1, 1), strides=(0, 0, 1, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),
                  ast_const(dtypes.float, 1e-05, st_src=(
                    UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 1, 112, 14, 14, 1, 1, 1), strides=(0, 0, 0, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),)),)),
          UOp(Ops.LOAD, dtypes.float, arg=None, src=(
            UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=6, src=()),
            UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 1, 112, 14, 14, 1, 1, 1), strides=(0, 0, 1, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),))
    opts = [Opt(op=OptOps.UPCAST, axis=1, arg=4), Opt(op=OptOps.UPCAST, axis=0, arg=2), Opt(op=OptOps.PADTO, axis=1, arg=32), Opt(op=OptOps.LOCAL, axis=0, arg=4), Opt(op=OptOps.LOCAL, axis=0, arg=2), Opt(op=OptOps.UPCAST, axis=1, arg=2), Opt(op=OptOps.UPCAST, axis=3, arg=0), Opt(op=OptOps.GROUP, axis=0, arg=8), Opt(op=OptOps.UPCAST, axis=1, arg=2), Opt(op=OptOps.LOCAL, axis=1, arg=16)]
    # COMPILE_ERROR on METAL in fuzz_linearizer ast 115: Error Domain=AGXMetalG14X Code=3 "Compiler encountered an internal error"
    helper_test_lin(Kernel(ast), opts, failed_platforms=[])

  def test_failure_16(self):
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 13, 1), strides=(0, 1, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.MUL, dtypes.float, arg=None, src=(
          UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (2,)), src=(
            UOp(Ops.LOAD, dtypes.float, arg=None, src=(
              UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=1, src=()),
              UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 13, 1024), strides=(0, 1024, 1), offset=0, mask=None, contiguous=True),)), src=()),)),)),
          ast_const(dtypes.float, 0.0009765625, st_src=(
            UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 13, 1), strides=(0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),))
    opts = [Opt(op=OptOps.GROUP, axis=0, arg=4), Opt(op=OptOps.UNROLL, axis=0, arg=0), Opt(op=OptOps.UNROLL, axis=0, arg=4), Opt(op=OptOps.GROUP, axis=0, arg=8), Opt(op=OptOps.UNROLL, axis=0, arg=4), Opt(op=OptOps.UNROLL, axis=1, arg=4)]
    # COMPILE_ERROR on METAL/GPU (probably HIP/CUDA too) in fuzz_linearizer ast 154: bracket nesting level exceeded maximum of 256
    helper_test_lin(Kernel(ast), opts, failed_platforms=[])

  def test_failure_17(self):
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(2, 1, 40, 1, 28, 28, 1, 1), strides=(31360, 0, 784, 0, 28, 1, 0, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (3,)), src=(
          UOp(Ops.MUL, dtypes.float, arg=None, src=(
            UOp(Ops.LOAD, dtypes.float, arg=None, src=(
              UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=1, src=()),
              UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(2, 1, 40, 240, 28, 28, 1, 1), strides=(0, 0, 1, 40, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),
            UOp(Ops.LOAD, dtypes.float, arg=None, src=(
              UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=2, src=()),
              UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(2, 1, 40, 240, 28, 28, 1, 1), strides=(188160, 0, 0, 784, 28, 1, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),)),))
    opts = [Opt(op=OptOps.UPCAST, axis=1, arg=4), Opt(op=OptOps.UPCAST, axis=0, arg=0), Opt(op=OptOps.PADTO, axis=1, arg=32), Opt(op=OptOps.LOCAL, axis=0, arg=2), Opt(op=OptOps.UPCAST, axis=1, arg=4), Opt(op=OptOps.UPCAST, axis=1, arg=4), Opt(op=OptOps.UPCAST, axis=1, arg=2), Opt(op=OptOps.GROUPTOP, axis=0, arg=16), Opt(op=OptOps.PADTO, axis=1, arg=32), Opt(op=OptOps.LOCAL, axis=1, arg=4)]
    # COMPILE_ERROR on METAL in fuzz_linearizer ast 178: Error Domain=AGXMetalG14X Code=3 "Compiler encountered an internal error"
    helper_test_lin(Kernel(ast), opts, failed_platforms=[])

  def test_failure_18(self):
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(2, 1, 384, 1), strides=(384, 0, 1, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.ADD, dtypes.float, arg=None, src=(
          UOp(Ops.LOAD, dtypes.float, arg=None, src=(
            UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=1, src=()),
            UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(2, 1, 384, 1), strides=(384, 0, 1, 0), offset=0, mask=None, contiguous=True),)), src=()),)),
          UOp(Ops.ADD, dtypes.float, arg=None, src=(
            UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (3,)), src=(
              UOp(Ops.MUL, dtypes.float, arg=None, src=(
                UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                  UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=2, src=()),
                  UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(2, 1, 384, 1536), strides=(1536, 0, 0, 1), offset=0, mask=None, contiguous=False),)), src=()),)),
                UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                  UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=3, src=()),
                  UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(2, 1, 384, 1536), strides=(0, 0, 1536, 1), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),
            UOp(Ops.LOAD, dtypes.float, arg=None, src=(
              UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=4, src=()),
              UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(2, 1, 384, 1), strides=(0, 0, 1, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),)),))
    opts = [Opt(op=OptOps.UPCAST, axis=1, arg=4), Opt(op=OptOps.UPCAST, axis=0, arg=0), Opt(op=OptOps.GROUPTOP, axis=0, arg=256), Opt(op=OptOps.UPCAST, axis=0, arg=4), Opt(op=OptOps.UPCAST, axis=0, arg=3)]
    # COMPILE_ERROR on METAL in fuzz_linearizer ast 239: Error Domain=AGXMetalG14X Code=3 "Compiler encountered an internal error"
    helper_test_lin(Kernel(ast), opts, failed_platforms=[])

  def test_failure_19(self):
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(2, 1, 4, 1, 9, 7, 3, 3), strides=(2268, 0, 567, 0, 63, 9, 3, 1), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (3,)), src=(
          UOp(Ops.MUL, dtypes.float, arg=None, src=(
            UOp(Ops.LOAD, dtypes.float, arg=None, src=(
              UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=1, src=()),
              UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(2, 1, 4, 4, 9, 7, 3, 3), strides=(0, 0, 36, 9, 0, 0, -3, -1), offset=8, mask=None, contiguous=False),)), src=()),)),
            UOp(Ops.LOAD, dtypes.float, arg=None, src=(
              UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=2, src=()),
              UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(2, 1, 4, 4, 9, 7, 3, 3), strides=(252, 0, 0, 63, 7, 1, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),)),))
    opts = [Opt(op=OptOps.LOCAL, axis=2, arg=3), Opt(op=OptOps.UPCAST, axis=1, arg=2), Opt(op=OptOps.UPCAST, axis=0, arg=0), Opt(op=OptOps.GROUP, axis=0, arg=4), Opt(op=OptOps.UPCAST, axis=1, arg=7), Opt(op=OptOps.UPCAST, axis=2, arg=3), Opt(op=OptOps.UPCAST, axis=1, arg=0), Opt(op=OptOps.LOCAL, axis=0, arg=2), Opt(op=OptOps.LOCAL, axis=0, arg=3)]
    # COMPILE_ERROR on METAL in fuzz_linearizer ast 379: Error Domain=AGXMetalG14X Code=3 "Compiler encountered an internal error"
    helper_test_lin(Kernel(ast), opts, failed_platforms=[])

  def test_failure_20(self):
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(4, 4), strides=(4, 1), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.MUL, dtypes.float, arg=None, src=(
          UOp(Ops.LOAD, dtypes.float, arg=None, src=(
            UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=1, src=()),
            UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(4, 4), strides=(0, 1), offset=0, mask=None, contiguous=False),)), src=()),)),
          ast_const(dtypes.float, 1.0, st_src=(
            UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(4, 4), strides=(0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),))
    opts = [Opt(op=OptOps.UPCAST, axis=1, arg=0), Opt(op=OptOps.UPCAST, axis=0, arg=0)]
    helper_test_lin(Kernel(ast), opts, failed_platforms=[])

  def test_failure_21(self):
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(45, 65), strides=(65, 1), offset=0, mask=None, contiguous=True),)), src=()),
        ast_const(dtypes.float, 1.0, st_src=(
          UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(45, 65), strides=(0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),))
    opts = [Opt(op=OptOps.PADTO, axis=0, arg=32)]
    helper_test_lin(Kernel(ast), opts, failed_platforms=[])

  #@unittest.skipIf(Device.DEFAULT in ("LLVM", "METAL", "CPU"), "flaky")
  @unittest.skip("flaky everywhere")
  def test_failure_22(self):
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 96, 1, 1), strides=(0, 1, 0, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.MUL, dtypes.float, arg=None, src=(
          x4:=ast_const(dtypes.float, 0.000244140625, st_src=(
            UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 96, 1, 1), strides=(0, 0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),
          UOp(Ops.MUL, dtypes.float, arg=None, src=(
            UOp(Ops.MUL, dtypes.float, arg=None, src=(
              UOp(Ops.MUL, dtypes.float, arg=None, src=(
                UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (0, 2, 3)), src=(
                  UOp(Ops.MUL, dtypes.float, arg=None, src=(
                    UOp(Ops.MUL, dtypes.float, arg=None, src=(
                      UOp(Ops.ADD, dtypes.float, arg=None, src=(
                        UOp(Ops.ADD, dtypes.float, arg=None, src=(
                          UOp(Ops.MUL, dtypes.float, arg=None, src=(
                            UOp(Ops.MUL, dtypes.float, arg=None, src=(
                              UOp(Ops.ADD, dtypes.float, arg=None, src=(
                                UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                                  UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=1, src=()),
                                  UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(32, 96, 8, 16), strides=(12288, 128, 16, 1), offset=0, mask=None, contiguous=True),)), src=()),)),
                                UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                                  UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=2, src=()),
                                  UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(32, 96, 8, 16), strides=(0, 1, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),
                              UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                                UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=3, src=()),
                                UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(32, 96, 8, 16), strides=(0, 1, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),
                            UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                              UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=4, src=()),
                              UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(32, 96, 8, 16), strides=(0, 1, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),
                          UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                            UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=5, src=()),
                            UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(32, 96, 8, 16), strides=(0, 1, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),
                        UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                          UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=6, src=()),
                          UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(32, 96, 8, 16), strides=(0, 1, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),
                      UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                        UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=7, src=()),
                        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(32, 96, 8, 16), strides=(0, 1, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),
                    UOp(Ops.ADD, dtypes.float, arg=None, src=(
                      UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                        UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=8, src=()),
                        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(2, 32, 48, 8, 16), strides=(0, 8640, 180, 18, 1), offset=19, mask=((1, 2), (0, 32), (0, 48), (0, 8), (0, 16)), contiguous=False), View(shape=(1536, 2, 128), strides=(128, 196608, 1), offset=0, mask=None, contiguous=False), View(shape=(32, 96, 8, 16), strides=(12288, 128, 16, 1), offset=0, mask=None, contiguous=True))), src=()),)),
                      UOp(Ops.ADD, dtypes.float, arg=None, src=(
                        UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                          UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=9, src=()),
                          UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(2, 32, 48, 8, 16), strides=(0, 8640, 180, 18, 1), offset=19, mask=((1, 2), (0, 32), (0, 48), (0, 8), (0, 16)), contiguous=False), View(shape=(1536, 2, 128), strides=(128, 196608, 1), offset=0, mask=None, contiguous=False), View(shape=(2, 32, 48, 8, 16), strides=(0, 12288, 128, 16, 1), offset=0, mask=((0, 1), (0, 32), (0, 48), (0, 8), (0, 16)), contiguous=False), View(shape=(1536, 2, 128), strides=(128, 196608, 1), offset=0, mask=None, contiguous=False), View(shape=(32, 96, 8, 16), strides=(12288, 128, 16, 1), offset=0, mask=None, contiguous=True))), src=()),)),
                        UOp(Ops.ADD, dtypes.float, arg=None, src=(
                          UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                            UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=10, src=()),
                            UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(2, 32, 48, 8, 16), strides=(0, 8640, 180, 18, 1), offset=19, mask=((1, 2), (0, 32), (0, 48), (0, 8), (0, 16)), contiguous=False), View(shape=(1536, 2, 128), strides=(128, 196608, 1), offset=0, mask=None, contiguous=False), View(shape=(2, 32, 48, 8, 16), strides=(0, 12288, 128, 16, 1), offset=0, mask=((0, 1), (0, 32), (0, 48), (0, 8), (0, 16)), contiguous=False), View(shape=(1536, 2, 128), strides=(128, 196608, 1), offset=0, mask=None, contiguous=False), View(shape=(2, 32, 48, 8, 16), strides=(0, 12288, 128, 16, 1), offset=0, mask=((0, 1), (0, 32), (0, 48), (0, 8), (0, 16)), contiguous=False), View(shape=(1536, 2, 128), strides=(128, 196608, 1), offset=0, mask=None, contiguous=False), View(shape=(32, 96, 8, 16), strides=(12288, 128, 16, 1), offset=0, mask=None, contiguous=True))), src=()),)),
                          UOp(Ops.ADD, dtypes.float, arg=None, src=(
                            UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                              UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=11, src=()),
                              UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(2, 32, 48, 8, 16), strides=(0, 8640, 180, 18, 1), offset=19, mask=((1, 2), (0, 32), (0, 48), (0, 8), (0, 16)), contiguous=False), View(shape=(1536, 2, 128), strides=(128, 196608, 1), offset=0, mask=None, contiguous=False), View(shape=(2, 32, 48, 8, 16), strides=(0, 12288, 128, 16, 1), offset=0, mask=((0, 1), (0, 32), (0, 48), (0, 8), (0, 16)), contiguous=False), View(shape=(1536, 2, 128), strides=(128, 196608, 1), offset=0, mask=None, contiguous=False), View(shape=(2, 32, 48, 8, 16), strides=(0, 12288, 128, 16, 1), offset=0, mask=((0, 1), (0, 32), (0, 48), (0, 8), (0, 16)), contiguous=False), View(shape=(1536, 2, 128), strides=(128, 196608, 1), offset=0, mask=None, contiguous=False), View(shape=(2, 32, 48, 8, 16), strides=(0, 12288, 128, 16, 1), offset=0, mask=((0, 1), (0, 32), (0, 48), (0, 8), (0, 16)), contiguous=False), View(shape=(1536, 2, 128), strides=(128, 196608, 1), offset=0, mask=None, contiguous=False), View(shape=(32, 96, 8, 16), strides=(12288, 128, 16, 1), offset=0, mask=None, contiguous=True))), src=()),)),
                            UOp(Ops.ADD, dtypes.float, arg=None, src=(
                              UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                                UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=12, src=()),
                                UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(2, 32, 48, 8, 16), strides=(0, 8640, 180, 18, 1), offset=19, mask=((1, 2), (0, 32), (0, 48), (0, 8), (0, 16)), contiguous=False), View(shape=(1536, 2, 128), strides=(128, 196608, 1), offset=0, mask=None, contiguous=False), View(shape=(2, 32, 48, 8, 16), strides=(0, 12288, 128, 16, 1), offset=0, mask=((0, 1), (0, 32), (0, 48), (0, 8), (0, 16)), contiguous=False), View(shape=(1536, 2, 128), strides=(128, 196608, 1), offset=0, mask=None, contiguous=False), View(shape=(2, 32, 48, 8, 16), strides=(0, 12288, 128, 16, 1), offset=0, mask=((0, 1), (0, 32), (0, 48), (0, 8), (0, 16)), contiguous=False), View(shape=(1536, 2, 128), strides=(128, 196608, 1), offset=0, mask=None, contiguous=False), View(shape=(2, 32, 48, 8, 16), strides=(0, 12288, 128, 16, 1), offset=0, mask=((0, 1), (0, 32), (0, 48), (0, 8), (0, 16)), contiguous=False), View(shape=(1536, 2, 128), strides=(128, 196608, 1), offset=0, mask=None, contiguous=False), View(shape=(2, 32, 48, 8, 16), strides=(0, 12288, 128, 16, 1), offset=0, mask=((0, 1), (0, 32), (0, 48), (0, 8), (0, 16)), contiguous=False), View(shape=(1536, 2, 128), strides=(128, 196608, 1), offset=0, mask=None, contiguous=False), View(shape=(32, 96, 8, 16), strides=(12288, 128, 16, 1), offset=0, mask=None, contiguous=True))), src=()),)),
                              UOp(Ops.ADD, dtypes.float, arg=None, src=(
                                UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                                  UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=13, src=()),
                                  UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(2, 32, 48, 8, 16), strides=(0, 8640, 180, 18, 1), offset=19, mask=((1, 2), (0, 32), (0, 48), (0, 8), (0, 16)), contiguous=False), View(shape=(1536, 2, 128), strides=(128, 196608, 1), offset=0, mask=None, contiguous=False), View(shape=(2, 32, 48, 8, 16), strides=(0, 12288, 128, 16, 1), offset=0, mask=((0, 1), (0, 32), (0, 48), (0, 8), (0, 16)), contiguous=False), View(shape=(1536, 2, 128), strides=(128, 196608, 1), offset=0, mask=None, contiguous=False), View(shape=(2, 32, 48, 8, 16), strides=(0, 12288, 128, 16, 1), offset=0, mask=((0, 1), (0, 32), (0, 48), (0, 8), (0, 16)), contiguous=False), View(shape=(1536, 2, 128), strides=(128, 196608, 1), offset=0, mask=None, contiguous=False), View(shape=(2, 32, 48, 8, 16), strides=(0, 12288, 128, 16, 1), offset=0, mask=((0, 1), (0, 32), (0, 48), (0, 8), (0, 16)), contiguous=False), View(shape=(1536, 2, 128), strides=(128, 196608, 1), offset=0, mask=None, contiguous=False), View(shape=(2, 32, 48, 8, 16), strides=(0, 12288, 128, 16, 1), offset=0, mask=((0, 1), (0, 32), (0, 48), (0, 8), (0, 16)), contiguous=False), View(shape=(1536, 2, 128), strides=(128, 196608, 1), offset=0, mask=None, contiguous=False), View(shape=(2, 32, 48, 8, 16), strides=(0, 12288, 128, 16, 1), offset=0, mask=((0, 1), (0, 32), (0, 48), (0, 8), (0, 16)), contiguous=False), View(shape=(1536, 2, 128), strides=(128, 196608, 1), offset=0, mask=None, contiguous=False), View(shape=(32, 96, 8, 16), strides=(12288, 128, 16, 1), offset=0, mask=None, contiguous=True))), src=()),)),
                                UOp(Ops.ADD, dtypes.float, arg=None, src=(
                                  UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                                    UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=14, src=()),
                                    UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(2, 32, 48, 8, 16), strides=(0, 8640, 180, 18, 1), offset=19, mask=((1, 2), (0, 32), (0, 48), (0, 8), (0, 16)), contiguous=False), View(shape=(1536, 2, 128), strides=(128, 196608, 1), offset=0, mask=None, contiguous=False), View(shape=(2, 32, 48, 8, 16), strides=(0, 12288, 128, 16, 1), offset=0, mask=((0, 1), (0, 32), (0, 48), (0, 8), (0, 16)), contiguous=False), View(shape=(1536, 2, 128), strides=(128, 196608, 1), offset=0, mask=None, contiguous=False), View(shape=(2, 32, 48, 8, 16), strides=(0, 12288, 128, 16, 1), offset=0, mask=((0, 1), (0, 32), (0, 48), (0, 8), (0, 16)), contiguous=False), View(shape=(1536, 2, 128), strides=(128, 196608, 1), offset=0, mask=None, contiguous=False), View(shape=(2, 32, 48, 8, 16), strides=(0, 12288, 128, 16, 1), offset=0, mask=((0, 1), (0, 32), (0, 48), (0, 8), (0, 16)), contiguous=False), View(shape=(1536, 2, 128), strides=(128, 196608, 1), offset=0, mask=None, contiguous=False), View(shape=(2, 32, 48, 8, 16), strides=(0, 12288, 128, 16, 1), offset=0, mask=((0, 1), (0, 32), (0, 48), (0, 8), (0, 16)), contiguous=False), View(shape=(1536, 2, 128), strides=(128, 196608, 1), offset=0, mask=None, contiguous=False), View(shape=(2, 32, 48, 8, 16), strides=(0, 12288, 128, 16, 1), offset=0, mask=((0, 1), (0, 32), (0, 48), (0, 8), (0, 16)), contiguous=False), View(shape=(1536, 2, 128), strides=(128, 196608, 1), offset=0, mask=None, contiguous=False), View(shape=(2, 32, 48, 8, 16), strides=(0, 12288, 128, 16, 1), offset=0, mask=((0, 1), (0, 32), (0, 48), (0, 8), (0, 16)), contiguous=False), View(shape=(1536, 2, 128), strides=(128, 196608, 1), offset=0, mask=None, contiguous=False), View(shape=(32, 96, 8, 16), strides=(12288, 128, 16, 1), offset=0, mask=None, contiguous=True))), src=()),)),
                                  UOp(Ops.ADD, dtypes.float, arg=None, src=(
                                    UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                                      UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=15, src=()),
                                      UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(2, 32, 48, 8, 16), strides=(0, 8640, 180, 18, 1), offset=19, mask=((1, 2), (0, 32), (0, 48), (0, 8), (0, 16)), contiguous=False), View(shape=(1536, 2, 128), strides=(128, 196608, 1), offset=0, mask=None, contiguous=False), View(shape=(2, 32, 48, 8, 16), strides=(0, 12288, 128, 16, 1), offset=0, mask=((0, 1), (0, 32), (0, 48), (0, 8), (0, 16)), contiguous=False), View(shape=(1536, 2, 128), strides=(128, 196608, 1), offset=0, mask=None, contiguous=False), View(shape=(2, 32, 48, 8, 16), strides=(0, 12288, 128, 16, 1), offset=0, mask=((0, 1), (0, 32), (0, 48), (0, 8), (0, 16)), contiguous=False), View(shape=(1536, 2, 128), strides=(128, 196608, 1), offset=0, mask=None, contiguous=False), View(shape=(2, 32, 48, 8, 16), strides=(0, 12288, 128, 16, 1), offset=0, mask=((0, 1), (0, 32), (0, 48), (0, 8), (0, 16)), contiguous=False), View(shape=(1536, 2, 128), strides=(128, 196608, 1), offset=0, mask=None, contiguous=False), View(shape=(2, 32, 48, 8, 16), strides=(0, 12288, 128, 16, 1), offset=0, mask=((0, 1), (0, 32), (0, 48), (0, 8), (0, 16)), contiguous=False), View(shape=(1536, 2, 128), strides=(128, 196608, 1), offset=0, mask=None, contiguous=False), View(shape=(2, 32, 48, 8, 16), strides=(0, 12288, 128, 16, 1), offset=0, mask=((0, 1), (0, 32), (0, 48), (0, 8), (0, 16)), contiguous=False), View(shape=(1536, 2, 128), strides=(128, 196608, 1), offset=0, mask=None, contiguous=False), View(shape=(2, 32, 48, 8, 16), strides=(0, 12288, 128, 16, 1), offset=0, mask=((0, 1), (0, 32), (0, 48), (0, 8), (0, 16)), contiguous=False), View(shape=(1536, 2, 128), strides=(128, 196608, 1), offset=0, mask=None, contiguous=False), View(shape=(2, 32, 48, 8, 16), strides=(0, 12288, 128, 16, 1), offset=0, mask=((0, 1), (0, 32), (0, 48), (0, 8), (0, 16)), contiguous=False), View(shape=(1536, 2, 128), strides=(128, 196608, 1), offset=0, mask=None, contiguous=False), View(shape=(32, 96, 8, 16), strides=(12288, 128, 16, 1), offset=0, mask=None, contiguous=True))), src=()),)),
                                    UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                                      UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=16, src=()),
                                      UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(2, 32, 48, 8, 16), strides=(0, 17280, 180, 18, 1), offset=19, mask=((0, 1), (0, 32), (0, 48), (0, 8), (0, 16)), contiguous=False), View(shape=(1536, 2, 128), strides=(128, 196608, 1), offset=0, mask=None, contiguous=False), View(shape=(2, 32, 48, 8, 16), strides=(0, 12288, 128, 16, 1), offset=0, mask=((0, 1), (0, 32), (0, 48), (0, 8), (0, 16)), contiguous=False), View(shape=(1536, 2, 128), strides=(128, 196608, 1), offset=0, mask=None, contiguous=False), View(shape=(2, 32, 48, 8, 16), strides=(0, 12288, 128, 16, 1), offset=0, mask=((0, 1), (0, 32), (0, 48), (0, 8), (0, 16)), contiguous=False), View(shape=(1536, 2, 128), strides=(128, 196608, 1), offset=0, mask=None, contiguous=False), View(shape=(2, 32, 48, 8, 16), strides=(0, 12288, 128, 16, 1), offset=0, mask=((0, 1), (0, 32), (0, 48), (0, 8), (0, 16)), contiguous=False), View(shape=(1536, 2, 128), strides=(128, 196608, 1), offset=0, mask=None, contiguous=False), View(shape=(2, 32, 48, 8, 16), strides=(0, 12288, 128, 16, 1), offset=0, mask=((0, 1), (0, 32), (0, 48), (0, 8), (0, 16)), contiguous=False), View(shape=(1536, 2, 128), strides=(128, 196608, 1), offset=0, mask=None, contiguous=False), View(shape=(2, 32, 48, 8, 16), strides=(0, 12288, 128, 16, 1), offset=0, mask=((0, 1), (0, 32), (0, 48), (0, 8), (0, 16)), contiguous=False), View(shape=(1536, 2, 128), strides=(128, 196608, 1), offset=0, mask=None, contiguous=False), View(shape=(2, 32, 48, 8, 16), strides=(0, 12288, 128, 16, 1), offset=0, mask=((0, 1), (0, 32), (0, 48), (0, 8), (0, 16)), contiguous=False), View(shape=(1536, 2, 128), strides=(128, 196608, 1), offset=0, mask=None, contiguous=False), View(shape=(2, 32, 48, 8, 16), strides=(0, 12288, 128, 16, 1), offset=0, mask=((0, 1), (0, 32), (0, 48), (0, 8), (0, 16)), contiguous=False), View(shape=(1536, 2, 128), strides=(128, 196608, 1), offset=0, mask=None, contiguous=False), View(shape=(32, 96, 8, 16), strides=(12288, 128, 16, 1), offset=0, mask=None, contiguous=True))), src=()),)),)),)),)),)),)),)),)),)),)),)),
                UOp(Ops.RECIP, dtypes.float, arg=None, src=(
                  UOp(Ops.MUL, dtypes.float, arg=None, src=(
                    UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                      UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=17, src=()),
                      UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 96, 1, 1), strides=(0, 1, 0, 0), offset=0, mask=None, contiguous=True),)), src=()),)),
                    ast_const(dtypes.float, 2.0, st_src=(
                      UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 96, 1, 1), strides=(0, 0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),)),
              x80:=UOp(Ops.RECIP, dtypes.float, arg=None, src=(
                UOp(Ops.ADD, dtypes.float, arg=None, src=(
                  UOp(Ops.MUL, dtypes.float, arg=None, src=(
                    UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                      UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=18, src=()),
                      UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 96, 1, 1), strides=(0, 1, 0, 0), offset=0, mask=None, contiguous=True),)), src=()),)),
                     x4,)),
                  ast_const(dtypes.float, 1e-05, st_src=(
                    UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 96, 1, 1), strides=(0, 0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),)),
             x80,)),)),)),))
    opts = []
    helper_test_lin(Kernel(ast), opts, failed_platforms=["METAL", "CUDA"])

  def test_failure_23(self):
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(240, 40, 1, 1), strides=(40, 1, 0, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.LOAD, dtypes.float, arg=None, src=(
          UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=1, src=()),
          UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(240, 40, 1, 1), strides=(1, 240, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),))
    opts = [Opt(op=OptOps.UPCAST, axis=1, arg=4), Opt(op=OptOps.LOCAL, axis=0, arg=16), Opt(op=OptOps.LOCAL, axis=1, arg=2), Opt(op=OptOps.UPCAST, axis=3, arg=2)]
    helper_test_lin(Kernel(ast), opts, failed_platforms=[])

  def test_failure_24(self):
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(8, 32, 1, 1), strides=(32, 1, 0, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.LOAD, dtypes.float, arg=None, src=(
          UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=1, src=()),
          UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(8, 32, 1, 1), strides=(1, 8, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),))
    opts = [Opt(op=OptOps.LOCAL, axis=1, arg=4), Opt(op=OptOps.UPCAST, axis=2, arg=2), Opt(op=OptOps.LOCAL, axis=1, arg=8), Opt(op=OptOps.UPCAST, axis=2, arg=0), Opt(op=OptOps.UPCAST, axis=1, arg=4), Opt(op=OptOps.LOCAL, axis=0, arg=8), Opt(op=OptOps.UPCAST, axis=1, arg=0), Opt(op=OptOps.UPCAST, axis=0, arg=2)]
    helper_test_lin(Kernel(ast), opts, failed_platforms=[])

  # this is the cause of the GPT2 BEAM instability. bisects to PR#3530 O(n) arange attempt
  def test_failure_25(self):
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1024, 1), strides=(1, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.ADD, dtypes.int, arg=None, src=(
          UOp(Ops.REDUCE_AXIS, dtypes.int, arg=(Ops.ADD, (1,)), src=(
            ast_const(dtypes.int, 1, st_src=(
              UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1025, 2047), strides=(0, 0), offset=0, mask=((0, 1025), (1023, 2047)), contiguous=False), View(shape=(1024, 1024), strides=(1, 2048), offset=0, mask=None, contiguous=False))), src=()),)),)),
          ast_const(dtypes.int, -1, st_src=(
            UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1024, 1), strides=(0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),))
    opts = [Opt(op=OptOps.GROUP, axis=0, arg=16), Opt(op=OptOps.UNROLL, axis=0, arg=4)]
    helper_test_lin(Kernel(ast), opts, failed_platforms=[])

  # COMPARE_ERROR from GPT2 kernel - stems from uops.py self.simplify_phi_loops
  def test_failure_26(self):
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(128, 1), strides=(1, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.ADD, dtypes.int, arg=None, src=(
          UOp(Ops.REDUCE_AXIS, dtypes.int, arg=(Ops.ADD, (1,)), src=(
            ast_const(dtypes.int, 1, st_src=(
              UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(129, 255), strides=(0, 0), offset=0, mask=((0, 129), (127, 255)), contiguous=False), View(shape=(128, 128), strides=(1, 256), offset=0, mask=None, contiguous=False))), src=()),)),)),
          ast_const(dtypes.int, -1, st_src=(
            UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(128, 1), strides=(0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),))
    all_failing_opts = [
      [Opt(op=OptOps.UPCAST, axis=0, arg=4), Opt(op=OptOps.GROUPTOP, axis=0, arg=32), Opt(op=OptOps.UNROLL, axis=0, arg=0)],
      [Opt(op=OptOps.GROUPTOP, axis=0, arg=32), Opt(op=OptOps.UNROLL, axis=0, arg=0), Opt(op=OptOps.UPCAST, axis=0, arg=4)],
      [Opt(op=OptOps.UNROLL, axis=0, arg=4), Opt(op=OptOps.UNROLL, axis=0, arg=4), Opt(op=OptOps.LOCAL, axis=0, arg=16), Opt(op=OptOps.UPCAST, axis=0, arg=0)],
      [Opt(op=OptOps.UNROLL, axis=0, arg=4), Opt(op=OptOps.LOCAL, axis=0, arg=4), Opt(op=OptOps.UPCAST, axis=0, arg=4), Opt(op=OptOps.UPCAST, axis=0, arg=0)],
      [Opt(op=OptOps.UNROLL, axis=0, arg=4), Opt(op=OptOps.LOCAL, axis=0, arg=16), Opt(op=OptOps.UPCAST, axis=0, arg=0), Opt(op=OptOps.UNROLL, axis=0, arg=4)],
      [Opt(op=OptOps.LOCAL, axis=0, arg=4), Opt(op=OptOps.UPCAST, axis=0, arg=4), Opt(op=OptOps.UNROLL, axis=0, arg=4), Opt(op=OptOps.UPCAST, axis=0, arg=0)],
      [Opt(op=OptOps.LOCAL, axis=0, arg=16), Opt(op=OptOps.UPCAST, axis=0, arg=0), Opt(op=OptOps.UNROLL, axis=0, arg=4), Opt(op=OptOps.UNROLL, axis=0, arg=4)],
      [Opt(op=OptOps.LOCAL, axis=0, arg=16), Opt(op=OptOps.UPCAST, axis=0, arg=0), Opt(op=OptOps.GROUP, axis=0, arg=8), Opt(op=OptOps.UNROLL, axis=1, arg=4)],
      [Opt(op=OptOps.LOCAL, axis=0, arg=16), Opt(op=OptOps.GROUP, axis=0, arg=16), Opt(op=OptOps.UPCAST, axis=0, arg=0), Opt(op=OptOps.UNROLL, axis=1, arg=4)],
      [Opt(op=OptOps.LOCAL, axis=0, arg=16), Opt(op=OptOps.GROUP, axis=0, arg=16), Opt(op=OptOps.UNROLL, axis=1, arg=4), Opt(op=OptOps.UPCAST, axis=0, arg=0)],
      [Opt(op=OptOps.GROUP, axis=0, arg=8), Opt(op=OptOps.UNROLL, axis=1, arg=4), Opt(op=OptOps.LOCAL, axis=0, arg=16), Opt(op=OptOps.UPCAST, axis=0, arg=0)],
    ]
    for opts in all_failing_opts:
      helper_test_lin(Kernel(ast), opts, failed_platforms=[])

  # COMPARE_ERROR from GPT2 kernel - just the first element off
  # testing ast 41
  # 0 ━┳ STORE MemBuffer(idx=0, dtype=dtypes.half, st=ShapeTracker(views=(View(shape=(1, 16, 13, 1), strides=(0, 13, 1, 0), offset=0, mask=None, contiguous=True),)))
  # 1  ┗━┳ MAX (3,)
  # 2    ┗━━ LOAD MemBuffer(idx=1, dtype=dtypes.half, st=ShapeTracker(views=(View(shape=(1, 16, 13, 13), strides=(0, 169, 13, 1), offset=0, mask=None, contiguous=True),)))
  # 208   13
  # ...
  # Mismatched elements: 1 / 1232 (0.0812%)
  # Max absolute difference: 0.8687
  # Max relative difference: 1.
  #  x: array([0.   , 0.996, 0.829, ..., 0.   , 0.   , 0.   ], dtype=float16)
  #  y: array([0.8687, 0.996 , 0.829 , ..., 0.    , 0.    , 0.    ], dtype=float16)
  # COMPARE FAILED!!
  def test_failure_27(self):
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 16, 13, 1), strides=(0, 13, 1, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.REDUCE_AXIS, dtypes.half, arg=(Ops.MAX, (3,)), src=(
          UOp(Ops.LOAD, dtypes.half, arg=None, src=(
            UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(), arg=1, src=()),
            UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 16, 13, 13), strides=(0, 169, 13, 1), offset=0, mask=None, contiguous=True),)), src=()),)),)),)),))
    all_failing_opts = [
      [Opt(op=OptOps.PADTO, axis=0, arg=32), Opt(op=OptOps.UPCAST, axis=0, arg=4), Opt(op=OptOps.UPCAST, axis=0, arg=7), Opt(op=OptOps.UPCAST, axis=0, arg=0)],
    ]
    for opts in all_failing_opts:
      helper_test_lin(Kernel(ast), opts, failed_platforms=[])

  def test_failure_28(self):
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.bfloat16.ptr(), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1,), strides=(0,), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.WHERE, dtypes.bfloat16, arg=None, src=(
          UOp(Ops.CMPLT, dtypes.bool, arg=None, src=(
            x5:=UOp(Ops.CAST, dtypes.bfloat16, arg=None, src=(
              UOp(Ops.LOAD, dtypes.int, arg=None, src=(
                UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), arg=1, src=()),
                UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1,), strides=(0,), offset=0, mask=None, contiguous=True),)), src=()),)),)),
            x9:=ast_const(dtypes.bfloat16, 230.0, st_src=(
              UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1,), strides=(0,), offset=0, mask=None, contiguous=True),)), src=()),)),)),
          UOp(Ops.ADD, dtypes.bfloat16, arg=None, src=(
            UOp(Ops.MUL, dtypes.bfloat16, arg=None, src=(
              UOp(Ops.MUL, dtypes.bfloat16, arg=None, src=(
                 x5,
                ast_const(dtypes.bfloat16, 0.004347826086956522, st_src=(
                  UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1,), strides=(0,), offset=0, mask=None, contiguous=True),)), src=()),)),)),
              ast_const(dtypes.bfloat16, 0.199374800625, st_src=(
                UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1,), strides=(0,), offset=0, mask=None, contiguous=True),)), src=()),)),)),
            ast_const(dtypes.bfloat16, 1.99375e-07, st_src=(
              UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1,), strides=(0,), offset=0, mask=None, contiguous=True),)), src=()),)),)),
          UOp(Ops.ADD, dtypes.bfloat16, arg=None, src=(
            UOp(Ops.MUL, dtypes.bfloat16, arg=None, src=(
              UOp(Ops.MUL, dtypes.bfloat16, arg=None, src=(
                UOp(Ops.ADD, dtypes.bfloat16, arg=None, src=(
                   x5,
                   x9,)),
                ast_const(dtypes.bfloat16, 0.0012987012987012987, st_src=(
                  UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1,), strides=(0,), offset=0, mask=None, contiguous=True),)), src=()),)),)),
              ast_const(dtypes.bfloat16, -0.19439062499999998, st_src=(
                UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1,), strides=(0,), offset=0, mask=None, contiguous=True),)), src=()),)),)),
            ast_const(dtypes.bfloat16, 0.199375, st_src=(
              UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1,), strides=(0,), offset=0, mask=None, contiguous=True),)), src=()),)),)),)),)),))
    helper_test_lin(Kernel(ast), opts=[], failed_platforms=[])

  def test_failure_29(self):
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(128, 1, 64, 56, 56, 1, 1, 1), strides=(200704, 0, 3136, 56, 1, 0, 0, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.CAST, dtypes.half, arg=None, src=(
          UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (7, 6, 5)), src=(
            UOp(Ops.CAST, dtypes.float, arg=None, src=(
              UOp(Ops.MUL, dtypes.half, arg=None, src=(
                UOp(Ops.LOAD, dtypes.half, arg=None, src=(
                  UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(), arg=1, src=()),
                  UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 128, 1, 64, 4, 58, 4, 58), strides=(0, 200704, 0, 3136, 0, 56, 0, 1), offset=-57, mask=((0, 1), (0, 128), (0, 1), (0, 64), (0, 4), (1, 57), (0, 4), (1, 57)), contiguous=False), View(shape=(128, 1, 64, 56, 56, 64, 3, 3), strides=(3444736, 0, 0, 232, 1, 53824, 13688, 59), offset=0, mask=None, contiguous=False))), src=()),)),
                UOp(Ops.LOAD, dtypes.half, arg=None, src=(
                  UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(), arg=2, src=()),
                  UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(128, 1, 64, 56, 56, 64, 3, 3), strides=(0, 0, 576, 0, 0, 9, 3, 1), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),)),)),)),))
    opts = [Opt(op=OptOps.TC, axis=0, arg=(-1, 1, 1)), Opt(op=OptOps.PADTO, axis=2, arg=32)]
    helper_test_lin(Kernel(ast), opts, failed_platforms=[], atol=1.0)

  def test_failure_30(self):
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(256, 1, 12, 31, 31, 1, 1, 1), strides=(11532, 0, 961, 31, 1, 0, 0, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.CAST, dtypes.half, arg=None, src=(
          UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (7, 6, 5)), src=(
            UOp(Ops.CAST, dtypes.float, arg=None, src=(
              UOp(Ops.MUL, dtypes.half, arg=None, src=(
                UOp(Ops.LOAD, dtypes.half, arg=None, src=(
                  UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(), arg=1, src=()),
                  UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(256, 1, 12, 31, 31, 3, 2, 2), strides=(3072, 0, 0, 32, 1, 1024, 32, 1), offset=0, mask=None, contiguous=False),)), src=()),)),
                UOp(Ops.LOAD, dtypes.half, arg=None, src=(
                  UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(), arg=2, src=()),
                  UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(256, 1, 12, 31, 31, 3, 2, 2), strides=(0, 0, 12, 0, 0, 4, 2, 1), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),)),)),)),))
    opts = [Opt(op=OptOps.PADTO, axis=3, arg=32), Opt(op=OptOps.LOCAL, axis=3, arg=32), Opt(op=OptOps.UPCAST, axis=3, arg=4), Opt(op=OptOps.UPCAST, axis=3, arg=0)]
    helper_test_lin(Kernel(ast), opts=opts, failed_platforms=[])

  # from METAL=1 fuzz_linearizer command in test.yml
  def test_failure_31(self):
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 16, 13, 1), strides=(0, 13, 1, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (3,)), src=(
          UOp(Ops.EXP2, dtypes.float, arg=None, src=(
            UOp(Ops.MUL, dtypes.float, arg=None, src=(
              UOp(Ops.ADD, dtypes.float, arg=None, src=(
                UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                  UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=1, src=()),
                  UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 16, 13, 13), strides=(0, 169, 13, 1), offset=0, mask=None, contiguous=True),)), src=()),)),
                UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                  UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=2, src=()),
                  UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 16, 13, 13), strides=(0, 13, 1, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),
              ast_const(dtypes.float, 1.4426950408889634, st_src=(
                UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 16, 13, 13), strides=(0, 0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),)),)),))
    opts = [Opt(op=OptOps.UNROLL, axis=0, arg=0), Opt(op=OptOps.PADTO, axis=1, arg=32)]
    helper_test_lin(Kernel(ast), opts=opts, failed_platforms=[])

  @unittest.skipIf(CI, "for real AMD GPU")
  def test_failure_32(self):
    # kernel from beaming resnet
    # Memory access fault on tinybox red
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(256, 1, 256, 14, 14, 1, 1, 1), strides=(50176, 0, 196, 14, 1, 0, 0, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.CAST, dtypes.half, arg=None, src=(
          UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (7, 6, 5)), src=(
            UOp(Ops.CAST, dtypes.float, arg=None, src=(
              UOp(Ops.MUL, dtypes.half, arg=None, src=(
                UOp(Ops.LOAD, dtypes.half, arg=None, src=(
                  UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(), arg=1, src=()),
                  UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 256, 1, 256, 4, 16, 4, 16), strides=(0, 50176, 0, 196, 0, 14, 0, 1), offset=-15, mask=((0, 1), (0, 256), (0, 1), (0, 256), (0, 4), (1, 15), (0, 4), (1, 15)), contiguous=False), View(shape=(256, 1, 256, 14, 14, 256, 3, 3), strides=(1048576, 0, 0, 64, 1, 4096, 1088, 17), offset=0, mask=None, contiguous=False))), src=()),)),
                UOp(Ops.LOAD, dtypes.half, arg=None, src=(
                  UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(), arg=2, src=()),
                  UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(256, 1, 256, 14, 14, 256, 3, 3), strides=(0, 0, 2304, 0, 0, 9, 3, 1), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),)),)),)),))
    opts = [Opt(op=OptOps.TC, axis=2, arg=(-1, 2, 1)), Opt(op=OptOps.UPCAST, axis=2, arg=7), Opt(op=OptOps.UNROLL, axis=1, arg=0), Opt(op=OptOps.LOCAL, axis=1, arg=16)]
    helper_test_lin(Kernel(ast), opts=opts, failed_platforms=[], atol=0.1, rtol=0.05)

  def test_failure_33(self):
    # Ops.UNMUL left after linearize
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1,), strides=(0,), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (0,)), src=(
          UOp(Ops.MUL, dtypes.float, arg=None, src=(
            x5:=UOp(Ops.LOAD, dtypes.float, arg=None, src=(
              UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=1, src=()),
              UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(32640,), strides=(1,), offset=0, mask=((0, 26040),), contiguous=False),)), src=()),)),
            UOp(Ops.WHERE, dtypes.float, arg=None, src=(
              UOp(Ops.CMPNE, dtypes.bool, arg=None, src=(
                 x5,
                x10:=ast_const(dtypes.float, 0.0, st_src=(
                  UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(32640,), strides=(0,), offset=0, mask=None, contiguous=False),)), src=()),)),)),
              UOp(Ops.WHERE, dtypes.float, arg=None, src=(
                UOp(Ops.CMPLT, dtypes.bool, arg=None, src=(
                  UOp(Ops.ADD, dtypes.float, arg=None, src=(
                    UOp(Ops.ADD, dtypes.float, arg=None, src=(
                      UOp(Ops.MUL, dtypes.float, arg=None, src=(
                        ast_const(dtypes.float, 0.06788442333021306, st_src=(
                          UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(32640,), strides=(0,), offset=0, mask=((0, 26040),), contiguous=False),)), src=()),)),
                         x5,)),
                      ast_const(dtypes.float, -0.03394221166510653, st_src=(
                        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(32640,), strides=(0,), offset=0, mask=((0, 26040),), contiguous=False),)), src=()),)),)),
                    UOp(Ops.ADD, dtypes.float, arg=None, src=(
                      UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                        UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=2, src=()),
                        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(32640,), strides=(1,), offset=-26040, mask=((26040, 32640),), contiguous=False),)), src=()),)),
                      ast_const(dtypes.float, -0.18257418583505536, st_src=(
                        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(32640,), strides=(0,), offset=0, mask=((26040, 32640),), contiguous=False),)), src=()),)),)),)),
                   x10,)),
                ast_const(dtypes.float, -1.0, st_src=(
                  UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(32640,), strides=(0,), offset=0, mask=None, contiguous=False),)), src=()),)),
                ast_const(dtypes.float, 1.0, st_src=(
                  UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(32640,), strides=(0,), offset=0, mask=None, contiguous=False),)), src=()),)),)),
               x10,)),)),)),)),))
    opts = [Opt(op=OptOps.GROUPTOP, axis=0, arg=16)]
    helper_test_lin(Kernel(ast), opts=opts, failed_platforms=[])

  # from fuzzing on metal
  def test_failure_34(self, unroll=False):
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(4, 1, 6, 10, 3, 1, 1, 1), strides=(180, 0, 30, 3, 1, 0, 0, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.MAX, dtypes.float, arg=None, src=(
          UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (6, 7)), src=(
            UOp(Ops.MUL, dtypes.float, arg=None, src=(
              UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=1, src=()),
                UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(4, 1, 6, 10, 3, 1, 2, 5), strides=(77, 0, 0, 7, 1, 0, 7, 1), offset=0, mask=None, contiguous=False),)), src=()),)),
              UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=2, src=()),
                UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(4, 1, 6, 10, 3, 1, 2, 5), strides=(0, 0, 10, 0, 0, 0, 5, 1), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),
          ast_const(dtypes.float, 0.0, st_src=(
            UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(4, 1, 6, 10, 3, 1, 1, 1), strides=(0, 0, 0, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),))
    opts = [Opt(op=OptOps.TC, axis=0, arg=(-1, 2, 1)), Opt(op=OptOps.UNROLL, axis=0, arg=0)] if unroll else [Opt(op=OptOps.TC, axis=0, arg=(-1, 2, 1))]
    helper_test_lin(Kernel(ast), opts=opts, failed_platforms=[])

  def test_failure_35(self): self.test_failure_34(True)

  # from world fuzz_linearizer: PYTHONPATH=. METAL=1 FUZZ_ALL_ACTIONS=1 DEPTH=1 FUZZ_N=100 FUZZ_NTH=84 python3 ./test/external/fuzz_linearizer.py
  def test_failure_36(self):
    # Ops.UNMUL left after linearize
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.uchar.ptr(), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(5, 1), strides=(1, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.CAST, dtypes.uchar, arg=None, src=(
          UOp(Ops.ADD, dtypes.uint, arg=None, src=(
            UOp(Ops.REDUCE_AXIS, dtypes.uint, arg=(Ops.ADD, (1,)), src=(
              UOp(Ops.CAST, dtypes.uint, arg=None, src=(
                ast_const(dtypes.uchar, 1, st_src=(
                  UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(6, 9), strides=(0, 0), offset=0, mask=((0, 6), (4, 9)), contiguous=False), View(shape=(5, 5), strides=(1, 10), offset=0, mask=None, contiguous=False))), src=()),)),)),)),
            ast_const(dtypes.uint, -1, st_src=(
              UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(5, 1), strides=(0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),)),))
    opts = [Opt(op=OptOps.UPCAST, axis=0, arg=0)]
    helper_test_lin(Kernel(ast), opts=opts, failed_platforms=[])

  # BEGIN METAL=1 ./examples/beautiful_mnist.py failures
  # log : PYTHONPATH=. LOGKERNS=/tmp/beautiful_mnist.kernels.txt METAL=1 python3 ./examples/beautiful_mnist.py
  def test_failure_37(self):
    # beautiful mnist kernel number 28: 6 possible TC axis_choices (3 for axis_buf1 and 2 reduce) and all fail
    # fuzz: PYTHONPATH=. METAL=1 FUZZ_ALL_ACTIONS=1 DEPTH=1 FUZZ_NTH=28 DEBUG=2 python3 ./test/external/fuzz_linearizer.py --logfile /tmp/beautiful_mnist.kernels.txt
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(512, 1, 32, 24, 24, 1, 1, 1), strides=(18432, 0, 576, 24, 1, 0, 0, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.MAX, dtypes.float, arg=None, src=(
          UOp(Ops.ADD, dtypes.float, arg=None, src=(
            UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (6, 7)), src=(
              UOp(Ops.MUL, dtypes.float, arg=None, src=(
                UOp(Ops.CAST, dtypes.float, arg=None, src=(
                  UOp(Ops.LOAD, dtypes.uchar, arg=None, src=(
                    UOp(Ops.DEFINE_GLOBAL, dtypes.uchar.ptr(), arg=1, src=()),
                    UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(512, 1, 32, 24, 24, 1, 5, 5), strides=(784, 0, 0, 28, 1, 0, 28, 1), offset=0, mask=None, contiguous=False),)), src=()),)),)),
                UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                  UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=2, src=()),
                  UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(512, 1, 32, 24, 24, 1, 5, 5), strides=(0, 0, 25, 0, 0, 0, 5, 1), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),
            UOp(Ops.LOAD, dtypes.float, arg=None, src=(
              UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=3, src=()),
              UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(512, 1, 32, 24, 24, 1, 1, 1), strides=(0, 0, 1, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),
          ast_const(dtypes.float, 0.0, st_src=(
            UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(512, 1, 32, 24, 24, 1, 1, 1), strides=(0, 0, 0, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),))
    for axis in [0,1,2,3,4,5]:
      opts = [Opt(op=OptOps.TC, axis=axis, arg=(-1, 2, 1))]
      helper_test_lin(Kernel(ast), opts=opts, failed_platforms=[])

  def test_failure_38(self):
    # beautiful mnist kernel number 87: 6 possible TC axis_choices (2 for axis_buf1 and 3 reduce) and first/second reduce axis fail for both axis_buf1 choices
    # fuzz: PYTHONPATH=. METAL=1 FUZZ_ALL_ACTIONS=1 DEPTH=1 FUZZ_NTH=87 DEBUG=2 python3 ./test/external/fuzz_linearizer.py --logfile /tmp/beautiful_mnist.kernels.txt
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 1, 32, 1, 1, 1, 5, 5, 256), strides=(0, 0, 6400, 0, 0, 0, 1280, 256, 1), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (0, 3, 4)), src=(
          UOp(Ops.MUL, dtypes.float, arg=None, src=(
            UOp(Ops.CAST, dtypes.float, arg=None, src=(
              UOp(Ops.LOAD, dtypes.uchar, arg=None, src=(
                UOp(Ops.DEFINE_GLOBAL, dtypes.uchar.ptr(), arg=1, src=()),
                UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(2, 1, 32, 24, 24, 1, 5, 5, 256), strides=(784, 0, 0, 28, 1, 0, 28, 1, 1568), offset=0, mask=None, contiguous=False),)), src=()),)),)),
            UOp(Ops.LOAD, dtypes.float, arg=None, src=(
              UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=2, src=()),
              UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(2, 1, 32, 24, 24, 1, 5, 5, 256), strides=(18432, 0, 576, 24, 1, 0, 0, 0, 36864), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),)),))
    for axis in [0,1,3,4]:
      opts = [Opt(op=OptOps.TC, axis=axis, arg=(-1, 2, 1))]
      helper_test_lin(Kernel(ast), opts=opts, failed_platforms=[])

  @unittest.skip("very slow, similar to test_failure_37")
  def test_failure_39(self):
    # beautiful mnist kernel number 127: 6 possible TC axis_choices (3 for axis_buf1 and 2 reduce) and all fail
    # fuzz: PYTHONPATH=. METAL=1 FUZZ_ALL_ACTIONS=1 DEPTH=1 FUZZ_NTH=127 DEBUG=2 python3 ./test/external/fuzz_linearizer.py --logfile /tmp/beautiful_mnist.kernels.txt
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(10000, 1, 32, 24, 24, 1, 1, 1), strides=(18432, 0, 576, 24, 1, 0, 0, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.MAX, dtypes.float, arg=None, src=(
          UOp(Ops.ADD, dtypes.float, arg=None, src=(
            UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (6, 7)), src=(
              UOp(Ops.MUL, dtypes.float, arg=None, src=(
                UOp(Ops.CAST, dtypes.float, arg=None, src=(
                  UOp(Ops.LOAD, dtypes.uchar, arg=None, src=(
                    UOp(Ops.DEFINE_GLOBAL, dtypes.uchar.ptr(), arg=1, src=()),
                    UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(10000, 1, 32, 24, 24, 1, 5, 5), strides=(784, 0, 0, 28, 1, 0, 28, 1), offset=0, mask=None, contiguous=False),)), src=()),)),)),
                UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                  UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=2, src=()),
                  UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(10000, 1, 32, 24, 24, 1, 5, 5), strides=(0, 0, 25, 0, 0, 0, 5, 1), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),
            UOp(Ops.LOAD, dtypes.float, arg=None, src=(
              UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=3, src=()),
              UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(10000, 1, 32, 24, 24, 1, 1, 1), strides=(0, 0, 1, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),
          ast_const(dtypes.float, 0.0, st_src=(
            UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(10000, 1, 32, 24, 24, 1, 1, 1), strides=(0, 0, 0, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),))
    for axis in [0,1,2,3,4,5]:
      opts = [Opt(op=OptOps.TC, axis=axis, arg=(-1, 2, 1))]
      helper_test_lin(Kernel(ast), opts=opts, failed_platforms=[])

  def test_failure_40(self):
    # beautiful mnist kernel number 3:
    # fuzz: PYTHONPATH=. METAL=1 FUZZ_ALL_ACTIONS=1 DEPTH=2 DEBUG=2 FUZZ_NTH=3 python3 ./test/external/fuzz_linearizer.py --logfile /tmp/beautiful_mnist.kernels.txt
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(60000, 1), strides=(1, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.ADD, dtypes.int, arg=None, src=(
          UOp(Ops.REDUCE_AXIS, dtypes.int, arg=(Ops.ADD, (1,)), src=(
            ast_const(dtypes.int, 1, st_src=(
              UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(60001, 119999), strides=(0, 0), offset=0, mask=((0, 60001), (59999, 119999)), contiguous=False), View(shape=(60000, 60000), strides=(1, 120000), offset=0, mask=None, contiguous=False))), src=()),)),)),
          ast_const(dtypes.int, -1, st_src=(
            UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(60000, 1), strides=(0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),))
    for amt in [16,32]:
      opts = [Opt(op=OptOps.GROUPTOP, axis=0, arg=amt), Opt(op=OptOps.UNROLL, axis=0, arg=0)]
      helper_test_lin(Kernel(ast), opts=opts, failed_platforms=[])
  # END METAL=1 ./examples/beautiful_mnist.py failures

  @unittest.skipIf(CI, "for real AMD GPU")
  def test_failure_41(self):
    # One more resnet crash with a page fault on AMD. Checked on rocm6.1.3, -O1 works, -O2 fails
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(256, 1, 128, 28, 28, 1, 1, 1), strides=(100352, 0, 784, 28, 1, 0, 0, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.CAST, dtypes.half, arg=None, src=(
          UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (5, 6, 7)), src=(
            UOp(Ops.CAST, dtypes.float, arg=None, src=(
              UOp(Ops.MUL, dtypes.half, arg=None, src=(
                UOp(Ops.LOAD, dtypes.half, arg=None, src=(
                  UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(), arg=1, src=()),
                  UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 256, 1, 128, 4, 58, 4, 58), strides=(0, 401408, 0, 3136, 0, 56, 0, 1), offset=-57, mask=((0, 1), (0, 256), (0, 1), (0, 128), (0, 4), (1, 57), (0, 4), (1, 57)), contiguous=False), View(shape=(256, 1, 128, 28, 28, 128, 3, 3), strides=(6889472, 0, 0, 464, 2, 53824, 13688, 59), offset=0, mask=None, contiguous=False))), src=()),)),
                UOp(Ops.LOAD, dtypes.half, arg=None, src=(
                  UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(), arg=2, src=()),
                  UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(256, 1, 128, 28, 28, 128, 3, 3), strides=(0, 0, 1152, 0, 0, 9, 3, 1), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),)),)),)),))
    opts=[Opt(op=OptOps.TC, axis=5, arg=(-1, 2, 1)), Opt(op=OptOps.UNROLL, axis=0, arg=0)]
    helper_test_lin(Kernel(ast), opts=opts, failed_platforms=["AMD", "HIP"], atol=0.02)

  # llama3 8B failure with BEAM=2 https://github.com/tinygrad/tinygrad/actions/runs/10150118124/job/28066519425#step:14:1, these don't compile
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test needs local")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_shared, "test needs shared")
  def test_failure_42(self):
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(25, 1), strides=(1, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (1,)), src=(
          UOp(Ops.LOAD, dtypes.float, arg=None, src=(
            UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=1, src=()),
            UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(26, 49), strides=(0, -1), offset=48, mask=((0, 26), (24, 49)), contiguous=False), View(shape=(25, 25), strides=(1, 50), offset=0, mask=None, contiguous=False))), src=()),)),)),)),))
    opts = [Opt(op=OptOps.GROUP, axis=0, arg=0), Opt(op=OptOps.PADTO, axis=0, arg=32), Opt(op=OptOps.UPCAST, axis=0, arg=2), Opt(op=OptOps.PADTO, axis=0, arg=32)]
    helper_test_lin(Kernel(ast), opts=opts, failed_platforms=[])

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test needs local")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_shared, "test needs shared")
  def test_failure_43(self):
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(25, 1), strides=(1, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (1,)), src=(
          UOp(Ops.LOAD, dtypes.float, arg=None, src=(
            UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=1, src=()),
            UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(26, 49), strides=(0, -1), offset=48, mask=((0, 26), (24, 49)), contiguous=False), View(shape=(25, 25), strides=(1, 50), offset=0, mask=None, contiguous=False))), src=()),)),)),)),))
    opts = [Opt(op=OptOps.GROUP, axis=0, arg=0), Opt(op=OptOps.PADTO, axis=0, arg=32), Opt(op=OptOps.LOCAL, axis=0, arg=4), Opt(op=OptOps.UPCAST, axis=0, arg=0)]
    helper_test_lin(Kernel(ast), opts=opts, failed_platforms=[])

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test needs local")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_shared, "test needs shared")
  def test_failure_44(self):
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(25, 1), strides=(1, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (1,)), src=(
          UOp(Ops.LOAD, dtypes.float, arg=None, src=(
            UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=1, src=()),
            UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(26, 49), strides=(0, -1), offset=48, mask=((0, 26), (24, 49)), contiguous=False), View(shape=(25, 25), strides=(1, 50), offset=0, mask=None, contiguous=False))), src=()),)),)),)),))
    opts = [Opt(op=OptOps.GROUP, axis=0, arg=0), Opt(op=OptOps.PADTO, axis=0, arg=32), Opt(op=OptOps.LOCAL, axis=0, arg=4), Opt(op=OptOps.UPCAST, axis=0, arg=4)]
    k = helper_test_lin(Kernel(ast), opts=opts, failed_platforms=[])
    assert k is not None
    ifs = [u for u in k.uops if u.op is Ops.IF]
    self.assertEqual(len(ifs), 3)
    #for st in k.uops.sink.src: self.assertEqual(len(st.src), 4)
    self.assertLessEqual(len(ifs[0].src[0].toposort()), 17)

  def test_failure_45(self):
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(2, 3, 1, 1, 1), strides=(3, 1, 0, 0, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (2, 3)), src=(
          UOp(Ops.MUL, dtypes.float, arg=None, src=(
            UOp(Ops.LOAD, dtypes.float, arg=None, src=(
              UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=1, src=()),
              UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(2, 3, 2, 3, 1), strides=(0, 0, 3, 1, 0), offset=0, mask=None, contiguous=False),)), src=()),)),
            UOp(Ops.CAST, dtypes.float, arg=None, src=(
              UOp(Ops.MUL, dtypes.bool, arg=None, src=(
                UOp(Ops.CMPNE, dtypes.bool, arg=None, src=(
                  UOp(Ops.CMPNE, dtypes.bool, arg=None, src=(
                    UOp(Ops.LOAD, dtypes.int, arg=None, src=(
                      UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), arg=2, src=()),
                      UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(2, 3, 2, 3, 1), strides=(0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),
                    UOp(Ops.ADD, dtypes.int, arg=None, src=(
                      UOp(Ops.REDUCE_AXIS, dtypes.int, arg=(Ops.ADD, (4,)), src=(
                        ast_const(dtypes.int, 1, st_src=(
                          UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(3, 3), strides=(0, 0), offset=0, mask=((0, 3), (1, 3)), contiguous=False), View(shape=(2, 3, 2, 3, 3), strides=(0, 0, 1, 0, 4), offset=0, mask=((0, 2), (0, 3), (0, 2), (0, 3), (0, 2)), contiguous=False))), src=()),)),)),
                      x19:=ast_const(dtypes.int, -1, st_src=(
                        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(2, 3, 2, 3, 1), strides=(0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),
                  x21:=ast_const(dtypes.bool, True, st_src=(
                    UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(2, 3, 2, 3, 1), strides=(0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),
                UOp(Ops.CMPNE, dtypes.bool, arg=None, src=(
                  UOp(Ops.CMPNE, dtypes.bool, arg=None, src=(
                    UOp(Ops.LOAD, dtypes.int, arg=None, src=(
                      UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), arg=3, src=()),
                      UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(2, 3, 2, 3, 1), strides=(3, 1, 0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),
                    UOp(Ops.ADD, dtypes.int, arg=None, src=(
                      UOp(Ops.REDUCE_AXIS, dtypes.int, arg=(Ops.ADD, (4,)), src=(
                        ast_const(dtypes.int, 1, st_src=(
                          UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(4, 5), strides=(0, 0), offset=0, mask=((0, 4), (2, 5)), contiguous=False), View(shape=(2, 3, 2, 3, 3), strides=(0, 0, 0, 1, 6), offset=0, mask=None, contiguous=False))), src=()),)),)),
                       x19,)),)),
                   x21,)),)),)),)),)),)),))
    # ValueError: size mismatched, can't reshape self.shape=(6, 2, 3, 3) -> new_shape=(6, 2, 3, 1, 2)
    opts = [Opt(op=OptOps.UNROLL, axis=2, arg=0)]
    helper_test_lin(Kernel(ast), opts=opts, failed_platforms=[])

  def test_failure_46(self):
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(512, 1), strides=(1, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.MUL, dtypes.float, arg=None, src=(
          UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (1,)), src=(
            UOp(Ops.MUL, dtypes.float, arg=None, src=(
              UOp(Ops.CAST, dtypes.float, arg=None, src=(
                UOp(Ops.MUL, dtypes.bool, arg=None, src=(
                  UOp(Ops.CMPNE, dtypes.bool, arg=None, src=(
                    UOp(Ops.CMPNE, dtypes.bool, arg=None, src=(
                      UOp(Ops.LOAD, dtypes.int, arg=None, src=(
                        UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), arg=1, src=()),
                        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(512, 10), strides=(0, 1), offset=0, mask=None, contiguous=False),)), src=()),)),
                      UOp(Ops.LOAD, dtypes.int, arg=None, src=(
                        UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), arg=2, src=()),
                        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(512, 10), strides=(1, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),
                    ast_const(dtypes.bool, True, st_src=(
                      UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(512, 10), strides=(0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),
                  UOp(Ops.LOAD, dtypes.bool, arg=None, src=(
                    UOp(Ops.DEFINE_GLOBAL, dtypes.bool.ptr(), arg=3, src=()),
                    UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(512, 10), strides=(1, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),
              UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=4, src=()),
                UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(512, 10), strides=(0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),
          UOp(Ops.RECIP, dtypes.float, arg=None, src=(
            UOp(Ops.LOAD, dtypes.float, arg=None, src=(
              UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=5, src=()),
              UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(512, 1), strides=(1, 0), offset=0, mask=None, contiguous=True),)), src=()),)),)),)),)),))
    opts = [Opt(op=OptOps.UPCAST, axis=0, arg=2)]
    helper_test_lin(Kernel(ast), opts=opts, failed_platforms=[])

  def test_failure_47(self):
    # upcast an arange, failed with UOP_IS_SYMBOLIC=1 (fixed!)
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(60000, 1), strides=(1, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.ADD, dtypes.int, arg=None, src=(
          UOp(Ops.REDUCE_AXIS, dtypes.int, arg=(Ops.ADD, (1,)), src=(
            ast_const(dtypes.int, 1, st_src=(
              UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(60001, 119999), strides=(0, 0), offset=0, mask=((0, 60001), (59999, 119999)), contiguous=False), View(shape=(60000, 60000), strides=(1, 120000), offset=0, mask=None, contiguous=False))), src=()),)),)),
          ast_const(dtypes.int, -1, st_src=(
            UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(60000, 1), strides=(0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),))
    opts = [Opt(op=OptOps.UPCAST, axis=0, arg=3)]
    helper_test_lin(Kernel(ast), opts=opts, failed_platforms=[])

  @unittest.skipUnless(not CI and Device.DEFAULT in ("NV", "CUDA"), "for real NV")
  def test_failure_48(self):
    # with UOP_IS_SYMBOLIC=1, generates the wrong IDIV (fixed!)
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 1, 64, 1, 1, 256, 1, 1, 256), strides=(0, 0, 65536, 0, 0, 256, 0, 0, 1), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (3, 4)), src=(
          UOp(Ops.CAST, dtypes.float, arg=None, src=(
            UOp(Ops.MUL, dtypes.half, arg=None, src=(
              UOp(Ops.LOAD, dtypes.half, arg=None, src=(
                UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(), arg=1, src=()),
                UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 1, 64, 56, 56, 256, 1, 1, 256), strides=(0, 0, 0, 56, 1, 3136, 0, 0, 802816), offset=0, mask=None, contiguous=False),)), src=()),)),
              UOp(Ops.LOAD, dtypes.half, arg=None, src=(
                UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(), arg=2, src=()),
                UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 1, 64, 56, 56, 256, 1, 1, 256), strides=(0, 0, 3136, 56, 1, 0, 0, 0, 200704), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),)),)),))
    opts = [Opt(op=OptOps.TC, axis=0, arg=(-1, 0, 1)), Opt(op=OptOps.UPCAST, axis=1, arg=4), Opt(op=OptOps.UPCAST, axis=0, arg=4), Opt(op=OptOps.LOCAL, axis=0, arg=2)]
    helper_test_lin(Kernel(ast, opts=Device[Device.DEFAULT].renderer), opts=opts, failed_platforms=[])

  def test_failure_49(self):
    # with UOP_IS_SYMBOLIC=1, on METAL it breaks store fusion and has A+B and B+A being two different UOp
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(10, 6, 1), strides=(6, 1, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (2,)), src=(
          UOp(Ops.MUL, dtypes.float, arg=None, src=(
            UOp(Ops.LOAD, dtypes.float, arg=None, src=(
              UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=1, src=()),
              UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(10, 6, 10), strides=(10, 0, 1), offset=0, mask=None, contiguous=False),)), src=()),)),
            UOp(Ops.LOAD, dtypes.float, arg=None, src=(
              UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=2, src=()),
              UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(10, 6, 10), strides=(0, 1, 6), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),)),))
    opts = [Opt(op=OptOps.TC, axis=0, arg=(-1, 2, 1)), Opt(op=OptOps.UPCAST, axis=0, arg=2)]
    helper_test_lin(Kernel(ast, opts=Device[Device.DEFAULT].renderer), opts=opts, failed_platforms=[])

  def test_failure_50(self):
    # from BEAM_COMPARE=2 running tinyphysics.onnx model
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.bool.ptr(), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 1, 20, 1, 20), strides=(0, 0, 20, 0, 1), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.CMPNE, dtypes.bool, arg=None, src=(
          UOp(Ops.REDUCE_AXIS, dtypes.bool, arg=(Ops.ADD, (3,)), src=(
            UOp(Ops.MUL, dtypes.bool, arg=None, src=(
              UOp(Ops.LOAD, dtypes.bool, arg=None, src=(
                UOp(Ops.DEFINE_GLOBAL, dtypes.bool.ptr(), arg=1, src=()),
                UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 1, 20, 20, 20), strides=(0, 0, 0, 20, 1), offset=0, mask=None, contiguous=False),)), src=()),)),
              UOp(Ops.CMPNE, dtypes.bool, arg=None, src=(
                UOp(Ops.CMPNE, dtypes.bool, arg=None, src=(
                  UOp(Ops.LOAD, dtypes.int, arg=None, src=(
                    UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), arg=2, src=()),
                    UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 1, 20, 20, 20), strides=(0, 0, 1, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),
                  UOp(Ops.LOAD, dtypes.int, arg=None, src=(
                    UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), arg=3, src=()),
                    UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 1, 20, 20, 20), strides=(0, 0, 0, 1, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),
                ast_const(dtypes.bool, True, st_src=(
                  UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 1, 20, 20, 20), strides=(0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),)),
          ast_const(dtypes.bool, True, st_src=(
            UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 1, 20, 1, 20), strides=(0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),))
    opts = [Opt(op=OptOps.UPCAST, axis=1, arg=2)]
    helper_test_lin(Kernel(ast, opts=Device[Device.DEFAULT].renderer), opts=opts, failed_platforms=[])

  def test_failure_51(self):
    # regression test for #7019, training bert on tinybox red
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(12, 1024, 1), strides=(1024, 1, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.RECIP, dtypes.half, arg=None, src=(
          UOp(Ops.ADD, dtypes.half, arg=None, src=(
            UOp(Ops.CONST, dtypes.half, arg=1.0, src=(
              x6:=UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(12, 1024, 1), strides=(0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),
            UOp(Ops.EXP2, dtypes.half, arg=None, src=(
              UOp(Ops.MUL, dtypes.half, arg=None, src=(
                UOp(Ops.MUL, dtypes.half, arg=None, src=(
                  UOp(Ops.CONST, dtypes.half, arg=2.0, src=(
                     x6,)),
                  UOp(Ops.ADD, dtypes.half, arg=None, src=(
                    UOp(Ops.CAST, dtypes.half, arg=None, src=(
                      UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (2,)), src=(
                        UOp(Ops.CAST, dtypes.float, arg=None, src=(
                          UOp(Ops.MUL, dtypes.half, arg=None, src=(
                            UOp(Ops.LOAD, dtypes.half, arg=None, src=(
                              UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(), arg=1, src=()),
                              UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(12, 1024, 1024), strides=(524288, 0, 1), offset=0, mask=None, contiguous=False),)), src=()),)),
                            UOp(Ops.LOAD, dtypes.half, arg=None, src=(
                              UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(), arg=2, src=()),
                              UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(12, 1024, 1024), strides=(0, 1024, 1), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),)),)),
                    UOp(Ops.LOAD, dtypes.half, arg=None, src=(
                      UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(), arg=3, src=()),
                      UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(12, 1024, 1), strides=(0, 1, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),
                UOp(Ops.CONST, dtypes.half, arg=-1.4426950408889634, src=(
                   x6,)),)),)),)),)),)),))
    opts = [Opt(op=OptOps.TC, axis=0, arg=(-1, 2, 1))]
    helper_test_lin(Kernel(ast, opts=Device[Device.DEFAULT].renderer), opts=opts, failed_platforms=[])

  @unittest.skipIf(CI and Device.DEFAULT in {"METAL"}, "hangs metal gpu CI")
  def test_failure_52(self):
    # resnet beam.
    # NV also fails with a pf.
    # CUDA Error 700, an illegal memory access was encountered
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(256, 1, 64, 112, 112, 1, 1, 1), strides=(802816, 0, 12544, 112, 1, 0, 0, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.CAST, dtypes.half, arg=None, src=(
          UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (5, 6, 7)), src=(
            UOp(Ops.CAST, dtypes.float, arg=None, src=(
              UOp(Ops.MUL, dtypes.half, arg=None, src=(
                UOp(Ops.LOAD, dtypes.half, arg=None, src=(
                  UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(), arg=1, src=()),
                  UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 256, 1, 3, 8, 230, 8, 230), strides=(0, 150528, 0, 50176, 0, 224, 0, 1), offset=-675, mask=((0, 1), (0, 256), (0, 1), (0, 3), (0, 8), (3, 227), (0, 8), (3, 227)), contiguous=False), View(shape=(256, 1, 64, 112, 112, 3, 7, 7), strides=(10156800, 0, 0, 3680, 2, 3385600, 425040, 231), offset=0, mask=None, contiguous=False))), src=()),)),
                UOp(Ops.LOAD, dtypes.half, arg=None, src=(
                  UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(), arg=2, src=()),
                  UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(256, 1, 64, 112, 112, 3, 7, 7), strides=(0, 0, 147, 0, 0, 49, 7, 1), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),)),)),)),))
    opts = [Opt(op=OptOps.TC, axis=0, arg=(-1, 2, 1)), Opt(op=OptOps.UPCAST, axis=1, arg=4), Opt(op=OptOps.LOCAL, axis=0, arg=16)]
    helper_test_lin(Kernel(ast, opts=Device[Device.DEFAULT].renderer), opts=opts, failed_platforms=[])

  def test_failure_53(self):
    # COMPILE_ERROR, val scope issue
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.uchar.ptr(), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1024, 1, 1), strides=(1, 0, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.REDUCE_AXIS, dtypes.uchar, arg=(Ops.ADD, (1,)), src=(
          UOp(Ops.MUL, dtypes.uchar, arg=None, src=(
            UOp(Ops.LOAD, dtypes.uchar, arg=None, src=(
              UOp(Ops.DEFINE_GLOBAL, dtypes.uchar.ptr(), arg=1, src=()),
              UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1024, 50000, 1), strides=(0, 1, 0), offset=0, mask=None, contiguous=False),)), src=()),)),
            UOp(Ops.CAST, dtypes.uchar, arg=None, src=(
              UOp(Ops.CMPNE, dtypes.bool, arg=None, src=(
                UOp(Ops.CMPNE, dtypes.bool, arg=None, src=(
                  UOp(Ops.LOAD, dtypes.int, arg=None, src=(
                    UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), arg=2, src=()),
                    UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1024, 50000, 1), strides=(1, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),
                  UOp(Ops.ADD, dtypes.int, arg=None, src=(
                    UOp(Ops.REDUCE_AXIS, dtypes.int, arg=(Ops.ADD, (2,)), src=(
                      UOp(Ops.WHERE, dtypes.int, arg=None, src=(
                        UOp(Ops.VALID, dtypes.bool, arg=None, src=(
                          UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(50001, 99999), strides=(0, 0), offset=0, mask=((0, 50001), (49999, 99999)), contiguous=False), View(shape=(1024, 50000, 50000), strides=(0, 1, 100000), offset=0, mask=None, contiguous=False))), src=()),)),
                        UOp(Ops.CONST, dtypes.int, arg=1, src=(
                          x20:=UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1024, 50000, 50000), strides=(0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),
                        UOp(Ops.CONST, dtypes.int, arg=0, src=(
                           x20,)),)),)),
                    UOp(Ops.CONST, dtypes.int, arg=-1, src=(
                      x23:=UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1024, 50000, 1), strides=(0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),
                UOp(Ops.CONST, dtypes.bool, arg=True, src=(
                   x23,)),)),)),)),)),)),))
    opts = [Opt(op=OptOps.GROUPTOP, axis=1, arg=16)]
    helper_test_lin(Kernel(ast, opts=Device[Device.DEFAULT].renderer), opts=opts, failed_platforms=["AMD", "GPU", "METAL", "NV", "CUDA"])

  @unittest.skipIf(CI and Device.DEFAULT in {"METAL"}, "hangs metal gpu CI")
  def test_failure_54(self):
    # resnet beam
    # HIP: Memory access fault by GPU node-1 (Agent handle: 0x56c21f1d1480) on address 0x730cc242e000. Reason: Page not present or supervisor privilege.
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(256, 1, 64, 56, 56, 1, 1, 1), strides=(200704, 0, 3136, 56, 1, 0, 0, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.CAST, dtypes.half, arg=None, src=(
          UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (5, 6, 7)), src=(
            UOp(Ops.CAST, dtypes.float, arg=None, src=(
              UOp(Ops.MUL, dtypes.half, arg=None, src=(
                UOp(Ops.LOAD, dtypes.half, arg=None, src=(
                  UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(), arg=1, src=()),
                  UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 256, 1, 64, 4, 58, 4, 58), strides=(0, 200704, 0, 3136, 0, 56, 0, 1), offset=-57, mask=((0, 1), (0, 256), (0, 1), (0, 64), (0, 4), (1, 57), (0, 4), (1, 57)), contiguous=False), View(shape=(256, 1, 64, 56, 56, 64, 3, 3), strides=(3444736, 0, 0, 232, 1, 53824, 13688, 59), offset=0, mask=None, contiguous=False))), src=()),)),
                  UOp(Ops.LOAD, dtypes.half, arg=None, src=(
                    UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(), arg=2, src=()),
                    UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(256, 1, 64, 56, 56, 64, 3, 3), strides=(0, 0, 576, 0, 0, 9, 3, 1), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),)),)),)),))
    opts = [Opt(op=OptOps.TC, axis=2, arg=(-1, 2, 1)), Opt(op=OptOps.UPCAST, axis=2, arg=7), Opt(op=OptOps.UPCAST, axis=1, arg=2)]
    helper_test_lin(Kernel(ast, opts=Device[Device.DEFAULT].renderer), opts=opts, failed_platforms=["HIP", "AMD"])

  @unittest.skipIf(CI and Device.DEFAULT in {"METAL"}, "hangs metal gpu CI")
  def test_failure_55(self):
    W = 2
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
        UOp(Ops.STORE, dtypes.void, arg=None, src=(
              UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(), arg=0, src=()),
              UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(W, 1, 64, 56, 56, 1, 1, 1), strides=(200704, 0, 3136, 56, 1, 0, 0, 0), offset=0, mask=None, contiguous=True),)), src=()),
              UOp(Ops.CAST, dtypes.half, arg=None, src=(
                UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (5, 6, 7)), src=(
                  UOp(Ops.CAST, dtypes.float, arg=None, src=(
                    UOp(Ops.MUL, dtypes.half, arg=None, src=(
                      UOp(Ops.LOAD, dtypes.half, arg=None, src=(
                        UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(), arg=1, src=()),
                        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, W, 1, 64, 4, 58, 4, 58), strides=(0, 200704, 0, 3136, 0, 56, 0, 1), offset=-57, mask=((0, 1), (0, W), (0, 1), (0, 64), (0, 4), (1, 57), (0, 4), (1, 57)), contiguous=False),
                                                                            View(shape=(W, 1, 64, 56, 56, 64, 3, 3), strides=(3444736, 0, 0, 232, 1, 53824, 13688, 59), offset=0, mask=None, contiguous=False))), src=()),)),
                      UOp(Ops.LOAD, dtypes.half, arg=None, src=(
                        UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(), arg=2, src=()),
                        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(W, 1, 64, 56, 56, 64, 3, 3), strides=(0, 0, 576, 0, 0, 9, 3, 1), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),)),)),)),))
    opts = [Opt(op=OptOps.SWAP, axis=1, arg=2)]
    helper_test_lin(Kernel(ast, opts=Device[Device.DEFAULT].renderer), opts=opts, failed_platforms=[])

  def test_failure_56(self):
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 16, 1, 1), strides=(0, 1, 0, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (0, 2, 3)), src=(
          UOp(Ops.MUL, dtypes.float, arg=None, src=(
            UOp(Ops.CAST, dtypes.float, arg=None, src=(
              UOp(Ops.CMPLT, dtypes.bool, arg=None, src=(
                x7:=UOp(Ops.CONST, dtypes.float, arg=0.0, src=(
                  x8:=UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(128, 16, 11, 11), strides=(0, 0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),
                UOp(Ops.MAX, dtypes.float, arg=None, src=(
                  UOp(Ops.ADD, dtypes.float, arg=None, src=(
                    UOp(Ops.MUL, dtypes.float, arg=None, src=(
                      UOp(Ops.MUL, dtypes.float, arg=None, src=(
                        UOp(Ops.ADD, dtypes.float, arg=None, src=(
                          UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                            UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=1, src=()),
                            UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(128, 16, 11, 11), strides=(1936, 121, 11, 1), offset=0, mask=None, contiguous=True),)), src=()),)),
                          UOp(Ops.MUL, dtypes.float, arg=None, src=(
                            UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                              UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=2, src=()),
                              x20:=UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(128, 16, 11, 11), strides=(0, 1, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),
                            UOp(Ops.CONST, dtypes.float, arg=-1.0, src=(
                               x8,)),)),)),
                        UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                          UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=3, src=()),
                           x20,)),)),
                      UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                        UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=4, src=()),
                         x20,)),)),
                    UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                      UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=5, src=()),
                       x20,)),)),
                   x7,)),)),)),
            UOp(Ops.LOAD, dtypes.float, arg=None, src=(
              UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=6, src=()),
              UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(128, 16, 5, 2, 5, 2), strides=(1600, 100, 20, 2, 4, 1), offset=0, mask=None, contiguous=False), View(shape=(128, 16, 11, 11), strides=(1600, 100, 10, 1), offset=0, mask=((0, 128), (0, 16), (0, 10), (0, 10)), contiguous=False))), src=()),)),)),)),)),))
    opts = [Opt(op=OptOps.UPCAST, axis=0, arg=0), Opt(op=OptOps.PADTO, axis=2, arg=32)]
    helper_test_lin(Kernel(ast, opts=Device[Device.DEFAULT].renderer), opts=opts, failed_platforms=[])

  def test_failure_57(self):
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 16, 1, 1), strides=(0, 1, 0, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (0, 2, 3)), src=(
          UOp(Ops.MUL, dtypes.float, arg=None, src=(
            UOp(Ops.CAST, dtypes.float, arg=None, src=(
              UOp(Ops.CMPLT, dtypes.bool, arg=None, src=(
                x7:=UOp(Ops.CONST, dtypes.float, arg=0.0, src=(
                  x8:=UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(128, 16, 11, 11), strides=(0, 0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),
                UOp(Ops.MAX, dtypes.float, arg=None, src=(
                  UOp(Ops.ADD, dtypes.float, arg=None, src=(
                    UOp(Ops.MUL, dtypes.float, arg=None, src=(
                      UOp(Ops.MUL, dtypes.float, arg=None, src=(
                        UOp(Ops.ADD, dtypes.float, arg=None, src=(
                          UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                            UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=1, src=()),
                            UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(128, 16, 11, 11), strides=(1936, 121, 11, 1), offset=0, mask=None, contiguous=True),)), src=()),)),
                          UOp(Ops.MUL, dtypes.float, arg=None, src=(
                            UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                              UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=2, src=()),
                              x20:=UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(128, 16, 11, 11), strides=(0, 1, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),
                            UOp(Ops.CONST, dtypes.float, arg=-1.0, src=(
                               x8,)),)),)),
                        UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                          UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=3, src=()),
                           x20,)),)),
                      UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                        UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=4, src=()),
                         x20,)),)),
                    UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                      UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=5, src=()),
                       x20,)),)),
                   x7,)),)),)),
            UOp(Ops.LOAD, dtypes.float, arg=None, src=(
              UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=6, src=()),
              UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(128, 16, 5, 2, 5, 2), strides=(1600, 100, 20, 2, 4, 1), offset=0, mask=None, contiguous=False), View(shape=(128, 16, 11, 11), strides=(1600, 100, 10, 1), offset=0, mask=((0, 128), (0, 16), (0, 10), (0, 10)), contiguous=False))), src=()),)),)),)),)),))
    opts = [Opt(op=OptOps.UPCAST, axis=0, arg=0), Opt(op=OptOps.PADTO, axis=1, arg=32)]
    helper_test_lin(Kernel(ast, opts=Device[Device.DEFAULT].renderer), opts=opts, failed_platforms=[])

  def test_failure_58(self):
    # OUT OF BOUNDS ACCESS in INDEX 0 - 50271 not in 0 - 50257. idx.src[1].render()='gidx0'
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(50257), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(50257, 1), strides=(1, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.ADD, dtypes.int, arg=None, src=(
          UOp(Ops.REDUCE_AXIS, dtypes.int, arg=(Ops.ADD, (1,)), src=(
            UOp(Ops.WHERE, dtypes.int, arg=None, src=(
              UOp(Ops.VALID, dtypes.bool, arg=None, src=(
                UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(50258, 100513), strides=(0, 0), offset=0, mask=((0, 50258), (50256, 100513)), contiguous=False), View(shape=(50257, 50257), strides=(1, 100514), offset=0, mask=None, contiguous=False))), src=()),)),
              UOp(Ops.CONST, dtypes.int, arg=1, src=(
                x9:=UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(50257, 50257), strides=(0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),
              UOp(Ops.CONST, dtypes.int, arg=0, src=(
                x9,)),)),)),
          UOp(Ops.CONST, dtypes.int, arg=-1, src=(
            UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(50257, 1), strides=(0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),))
    opts = [Opt(op=OptOps.GROUPTOP, axis=0, arg=29), Opt(op=OptOps.PADTO, axis=0, arg=32)]
    with Context(IGNORE_OOB=0):
      helper_test_lin(Kernel(ast, opts=Device[Device.DEFAULT].renderer), opts=opts, failed_platforms=["METAL", "GPU", "AMD", "NV", "CUDA"])

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test needs local")
  def test_failure_59(self):
    # stable diffusion with SINGLE_KERNEL_SOFTMAX=1
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(268435456), arg=0, src=()),
        x2:=UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(2, 8, 4096, 4096, 1, 1), strides=(134217728, 16777216, 4096, 1, 0, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.MUL, dtypes.float, arg=None, src=(
          UOp(Ops.EXP2, dtypes.float, arg=None, src=(
            UOp(Ops.MUL, dtypes.float, arg=None, src=(
              UOp(Ops.ADD, dtypes.float, arg=None, src=(
                UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                  x8:=UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(268435456), arg=1, src=()),
                  x2,)),
                UOp(Ops.MUL, dtypes.float, arg=None, src=(
                  UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.MAX, (5,), True), src=(
                    UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                      x8,
                      UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(2, 8, 4096, 4096, 1, 4096), strides=(134217728, 16777216, 4096, 0, 0, 1), offset=0, mask=None, contiguous=False),)), src=()),)),)),
                  UOp(Ops.CONST, dtypes.float, arg=-1.0, src=(
                    x14:=UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(2, 8, 4096, 4096, 1, 1), strides=(0, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),
              UOp(Ops.CONST, dtypes.float, arg=1.4426950408889634, src=(
                x14,)),)),)),
          UOp(Ops.RECIP, dtypes.float, arg=None, src=(
            UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (4,)), src=(
              UOp(Ops.EXP2, dtypes.float, arg=None, src=(
                UOp(Ops.MUL, dtypes.float, arg=None, src=(
                  UOp(Ops.ADD, dtypes.float, arg=None, src=(
                    UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                      x8,
                      UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(2, 8, 4096, 4096, 4096, 1), strides=(134217728, 16777216, 4096, 0, 1, 0), offset=0, mask=None, contiguous=False),)), src=()),)),
                    UOp(Ops.MUL, dtypes.float, arg=None, src=(
                      UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.MAX, (5,), True), src=(
                        UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                          x8,
                          UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(2, 8, 4096, 4096, 4096, 4096), strides=(134217728, 16777216, 4096, 0, 0, 1), offset=0, mask=None, contiguous=False),)), src=()),)),)),
                      UOp(Ops.CONST, dtypes.float, arg=-1.0, src=(
                        x28:=UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(2, 8, 4096, 4096, 4096, 1), strides=(0, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),
                  UOp(Ops.CONST, dtypes.float, arg=1.4426950408889634, src=(
                    x28,)),)),)),)),)),)),)),))
    opts = [Opt(op=OptOps.UPCAST, axis=1, arg=4), Opt(op=OptOps.UNROLL, axis=1, arg=4), Opt(op=OptOps.LOCAL, axis=0, arg=8), Opt(op=OptOps.LOCAL, axis=1, arg=16)]
    # NOTE: this is slow to run, just confirm it can generate the program without Exception
    Kernel(ast, opts=Device[Device.DEFAULT].renderer).apply_opts(opts).to_program()

  @unittest.expectedFailure
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test needs local")
  def test_failure_60(self):
    # TestSymbolicOps.test_attention
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(80), arg=0, src=()),
        x2:=UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(2, 4, 1, UOp(Ops.DEFINE_VAR, dtypes.int, arg=('i', 1, 10), src=()), 1, 1), strides=(UOp(Ops.MUL, dtypes.int, arg=None, src=(
      UOp(Ops.MUL, dtypes.int, arg=None, src=(
        UOp(Ops.MUL, dtypes.int, arg=None, src=(
          x2:=UOp(Ops.CONST, dtypes.int, arg=1, src=()),
          UOp(Ops.DEFINE_VAR, dtypes.int, arg=('i', 1, 10), src=()),)),
        x2,)),
      UOp(Ops.CONST, dtypes.int, arg=4, src=()),)), UOp(Ops.MUL, dtypes.int, arg=None, src=(
      UOp(Ops.MUL, dtypes.int, arg=None, src=(
        x1:=UOp(Ops.CONST, dtypes.int, arg=1, src=()),
        UOp(Ops.DEFINE_VAR, dtypes.int, arg=('i', 1, 10), src=()),)),
      x1,)), 0, 1, 0, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.MUL, dtypes.float, arg=None, src=(
          UOp(Ops.EXP2, dtypes.float, arg=None, src=(
            UOp(Ops.MUL, dtypes.float, arg=None, src=(
              UOp(Ops.ADD, dtypes.float, arg=None, src=(
                UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                  UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(80), arg=1, src=()),
                  x2,)),
                UOp(Ops.MUL, dtypes.float, arg=None, src=(
                  UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.MAX, (5,), True), src=(
                    UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                      x12:=UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(80), arg=2, src=()),
                      UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(2, 4, 1, UOp(Ops.DEFINE_VAR, dtypes.int, arg=('i', 1, 10), src=()), 1, UOp(Ops.DEFINE_VAR, dtypes.int, arg=('i', 1, 10), src=())), strides=(UOp(Ops.MUL, dtypes.int, arg=None, src=(
      UOp(Ops.MUL, dtypes.int, arg=None, src=(
        UOp(Ops.CONST, dtypes.int, arg=4, src=()),
        UOp(Ops.MUL, dtypes.int, arg=None, src=(
          x3:=UOp(Ops.CONST, dtypes.int, arg=1, src=()),
          UOp(Ops.DEFINE_VAR, dtypes.int, arg=('i', 1, 10), src=()),)),)),
      x3,)), UOp(Ops.MUL, dtypes.int, arg=None, src=(
      UOp(Ops.MUL, dtypes.int, arg=None, src=(
        x1:=UOp(Ops.CONST, dtypes.int, arg=1, src=()),
        UOp(Ops.MUL, dtypes.int, arg=None, src=(
          x1,
          UOp(Ops.DEFINE_VAR, dtypes.int, arg=('i', 1, 10), src=()),)),)),
      x1,)), 0, UOp(Ops.MUL, dtypes.int, arg=None, src=(
      UOp(Ops.MUL, dtypes.int, arg=None, src=(
        UOp(Ops.CONST, dtypes.int, arg=0, src=()),
        UOp(Ops.MUL, dtypes.int, arg=None, src=(
          x3:=UOp(Ops.CONST, dtypes.int, arg=1, src=()),
          UOp(Ops.DEFINE_VAR, dtypes.int, arg=('i', 1, 10), src=()),)),)),
      x3,)), 0, 1), offset=0, mask=None, contiguous=False),)), src=()),)),)),
                  UOp(Ops.CONST, dtypes.float, arg=-1.0, src=(
                    x15:=UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(2, 4, 1, UOp(Ops.DEFINE_VAR, dtypes.int, arg=('i', 1, 10), src=()), 1, 1), strides=(0, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),
              UOp(Ops.CONST, dtypes.float, arg=1.4426950408889634, src=(
                x15,)),)),)),
          UOp(Ops.RECIP, dtypes.float, arg=None, src=(
            UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (4,)), src=(
              UOp(Ops.EXP2, dtypes.float, arg=None, src=(
                UOp(Ops.MUL, dtypes.float, arg=None, src=(
                  UOp(Ops.ADD, dtypes.float, arg=None, src=(
                    UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                      x12,
                      UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(2, 4, 1, UOp(Ops.DEFINE_VAR, dtypes.int, arg=('i', 1, 10), src=()), UOp(Ops.DEFINE_VAR, dtypes.int, arg=('i', 1, 10), src=()), 1), strides=(UOp(Ops.MUL, dtypes.int, arg=None, src=(
      UOp(Ops.MUL, dtypes.int, arg=None, src=(
        UOp(Ops.CONST, dtypes.int, arg=4, src=()),
        UOp(Ops.MUL, dtypes.int, arg=None, src=(
          x3:=UOp(Ops.CONST, dtypes.int, arg=1, src=()),
          UOp(Ops.DEFINE_VAR, dtypes.int, arg=('i', 1, 10), src=()),)),)),
      x3,)), UOp(Ops.MUL, dtypes.int, arg=None, src=(
      UOp(Ops.MUL, dtypes.int, arg=None, src=(
        x1:=UOp(Ops.CONST, dtypes.int, arg=1, src=()),
        UOp(Ops.MUL, dtypes.int, arg=None, src=(
          x1,
          UOp(Ops.DEFINE_VAR, dtypes.int, arg=('i', 1, 10), src=()),)),)),
      x1,)), 0, UOp(Ops.MUL, dtypes.int, arg=None, src=(
      UOp(Ops.MUL, dtypes.int, arg=None, src=(
        UOp(Ops.CONST, dtypes.int, arg=0, src=()),
        UOp(Ops.MUL, dtypes.int, arg=None, src=(
          x3:=UOp(Ops.CONST, dtypes.int, arg=1, src=()),
          UOp(Ops.DEFINE_VAR, dtypes.int, arg=('i', 1, 10), src=()),)),)),
      x3,)), 1, 0), offset=0, mask=None, contiguous=False),)), src=()),)),
                    UOp(Ops.MUL, dtypes.float, arg=None, src=(
                      UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.MAX, (5,), True), src=(
                        UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                          x12,
                          UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(2, 4, 1, UOp(Ops.DEFINE_VAR, dtypes.int, arg=('i', 1, 10), src=()), UOp(Ops.DEFINE_VAR, dtypes.int, arg=('i', 1, 10), src=())), strides=(UOp(Ops.MUL, dtypes.int, arg=None, src=(
      UOp(Ops.CONST, dtypes.int, arg=4, src=()),
      UOp(Ops.MUL, dtypes.int, arg=None, src=(
        UOp(Ops.CONST, dtypes.int, arg=1, src=()),
        UOp(Ops.DEFINE_VAR, dtypes.int, arg=('i', 1, 10), src=()),)),)), UOp(Ops.MUL, dtypes.int, arg=None, src=(
      x0:=UOp(Ops.CONST, dtypes.int, arg=1, src=()),
      UOp(Ops.MUL, dtypes.int, arg=None, src=(
        x0,
        UOp(Ops.DEFINE_VAR, dtypes.int, arg=('i', 1, 10), src=()),)),)), 0, UOp(Ops.MUL, dtypes.int, arg=None, src=(
      UOp(Ops.CONST, dtypes.int, arg=0, src=()),
      UOp(Ops.MUL, dtypes.int, arg=None, src=(
        UOp(Ops.CONST, dtypes.int, arg=1, src=()),
        UOp(Ops.DEFINE_VAR, dtypes.int, arg=('i', 1, 10), src=()),)),)), 1), offset=0, mask=None, contiguous=False), View(shape=(2, 4, 1, UOp(Ops.DEFINE_VAR, dtypes.int, arg=('i', 1, 10), src=()), UOp(Ops.DEFINE_VAR, dtypes.int, arg=('i', 1, 10), src=()), UOp(Ops.DEFINE_VAR, dtypes.int, arg=('i', 1, 10), src=())), strides=(UOp(Ops.
    MUL, dtypes.int, arg=None, src=(
      UOp(Ops.MUL, dtypes.int, arg=None, src=(
        UOp(Ops.CONST, dtypes.int, arg=4, src=()),
        x2:=UOp(Ops.MUL, dtypes.int, arg=None, src=(
          UOp(Ops.CONST, dtypes.int, arg=1, src=()),
          UOp(Ops.DEFINE_VAR, dtypes.int, arg=('i', 1, 10), src=()),)),)),
      x2,)), UOp(Ops.MUL, dtypes.int, arg=None, src=(
      UOp(Ops.MUL, dtypes.int, arg=None, src=(
        x1:=UOp(Ops.CONST, dtypes.int, arg=1, src=()),
        x2:=UOp(Ops.MUL, dtypes.int, arg=None, src=(
          x1,
          UOp(Ops.DEFINE_VAR, dtypes.int, arg=('i', 1, 10), src=()),)),)),
      x2,)), 0, UOp(Ops.MUL, dtypes.int, arg=None, src=(
      UOp(Ops.MUL, dtypes.int, arg=None, src=(
        UOp(Ops.CONST, dtypes.int, arg=0, src=()),
        x2:=UOp(Ops.MUL, dtypes.int, arg=None, src=(
          UOp(Ops.CONST, dtypes.int, arg=1, src=()),
          UOp(Ops.DEFINE_VAR, dtypes.int, arg=('i', 1, 10), src=()),)),)),
      x2,)), UOp(Ops.MUL, dtypes.int, arg=None, src=(
      x0:=UOp(Ops.CONST, dtypes.int, arg=1, src=()),
      UOp(Ops.MUL, dtypes.int, arg=None, src=(
        x0,
        UOp(Ops.DEFINE_VAR, dtypes.int, arg=('i', 1, 10), src=()),)),)), 1), offset=0, mask=None, contiguous=False))), src=()),)),)),
                      UOp(Ops.CONST, dtypes.float, arg=-1.0, src=(
                        x29:=UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(2, 4, 1, UOp(Ops.DEFINE_VAR, dtypes.int, arg=('i', 1, 10), src=())), strides=(0, 0, 0, 0), offset=0, mask=None, contiguous=False), View(shape=(2, 4, 1, UOp(Ops.DEFINE_VAR, dtypes.int, arg=('i', 1, 10), src=()), UOp(Ops.DEFINE_VAR, dtypes.int
    , arg=('i', 1, 10), src=()), 1), strides=(UOp(Ops.MUL, dtypes.int, arg=None, src=(
      UOp(Ops.MUL, dtypes.int, arg=None, src=(
        UOp(Ops.CONST, dtypes.int, arg=4, src=()),
        UOp(Ops.MUL, dtypes.int, arg=None, src=(
          x3:=UOp(Ops.CONST, dtypes.int, arg=1, src=()),
          UOp(Ops.DEFINE_VAR, dtypes.int, arg=('i', 1, 10), src=()),)),)),
      x3,)), UOp(Ops.MUL, dtypes.int, arg=None, src=(
      UOp(Ops.MUL, dtypes.int, arg=None, src=(
        x1:=UOp(Ops.CONST, dtypes.int, arg=1, src=()),
        UOp(Ops.MUL, dtypes.int, arg=None, src=(
          x1,
          UOp(Ops.DEFINE_VAR, dtypes.int, arg=('i', 1, 10), src=()),)),)),
      x1,)), 0, UOp(Ops.MUL, dtypes.int, arg=None, src=(
      UOp(Ops.MUL, dtypes.int, arg=None, src=(
        UOp(Ops.CONST, dtypes.int, arg=0, src=()),
        UOp(Ops.MUL, dtypes.int, arg=None, src=(
          x3:=UOp(Ops.CONST, dtypes.int, arg=1, src=()),
          UOp(Ops.DEFINE_VAR, dtypes.int, arg=('i', 1, 10), src=()),)),)),
      x3,)), 1, 0), offset=0, mask=None, contiguous=False))), src=()),)),)),)),
                  UOp(Ops.CONST, dtypes.float, arg=1.4426950408889634, src=(
                    x29,)),)),)),)),)),)),)),))
    opts = [Opt(op=OptOps.LOCAL, axis=0, arg=2), Opt(op=OptOps.LOCAL, axis=0, arg=4)]
    # NOTE: this is slow to run, just confirm it can generate the program without Exception
    Kernel(ast, opts=Device[Device.DEFAULT].renderer).apply_opts(opts).to_program()

if __name__ == '__main__':
  unittest.main()
