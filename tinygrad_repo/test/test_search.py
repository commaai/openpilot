import unittest

from tinygrad.opt.kernel import Opt, OptOps, Kernel
from tinygrad.uop.ops import UOp, Ops
from tinygrad.opt.search import bufs_from_lin, actions, beam_search
from tinygrad.device import Device
from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinygrad.helpers import Context, GlobalCounters
from tinygrad.engine.realize import capturing
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View
from extra.optimization.helpers import time_linearizer

class TestBEAM(unittest.TestCase):
  def test_dynamic_beam(self):
    # TODO: make this infra globally usable
    class Capture:
      def __init__(self): self.captured = []
      def add(self, x): self.captured.append(x)

    capturing.append(Capture())
    kernel_count = GlobalCounters.kernel_count
    with Context(BEAM=1): Tensor.zeros(16).contiguous().realize()
    assert GlobalCounters.kernel_count == kernel_count + 1
    k_beam_1 = capturing[0].captured
    capturing.clear()

    capturing.append(Capture())
    kernel_count = GlobalCounters.kernel_count
    with Context(BEAM=0): Tensor.zeros(16).contiguous().realize()
    assert GlobalCounters.kernel_count == kernel_count + 1
    k_beam_0 = capturing[0].captured
    capturing.clear()
    self.assertNotEqual(k_beam_0[-1].prg.p.src, k_beam_1[-1].prg.p.src)

  def test_get_kernel_actions(self):
    from test.test_linearizer import helper_realized_ast
    a = Tensor.rand(4, 3)
    b = Tensor.rand(3)
    realized_ast, _ = helper_realized_ast(a @ b)
    from tinygrad.opt.search import get_kernel_actions
    lins = get_kernel_actions(Kernel(realized_ast), False).values()

    # ensure amt=0 are not duplicated
    if Opt(OptOps.UPCAST, 0, 0) in actions:
      assert len([x for x in lins if x.applied_opts[0] == Opt(OptOps.UPCAST, axis=0, arg=4)]) == 0, "did not de-dup UPCAST"
    if Opt(OptOps.LOCAL, 0, 0) in actions:
      assert len([x for x in lins if x.applied_opts[0] == Opt(OptOps.LOCAL, axis=0, arg=4)]) == 0, "did not de-dup LOCAL"
    if Opt(OptOps.UNROLL, 0, 0) in actions:
      assert len([x for x in lins if x.applied_opts[0] == Opt(OptOps.UNROLL, axis=0, arg=3)]) == 0, "did not de-dup UNROLL"
    if Opt(OptOps.GROUP, 0, 0) in actions:
      assert len([x for x in lins if x.applied_opts[0] == Opt(OptOps.GROUP, axis=0, arg=3)]) == 0, "did not de-dup GROUP"
    if Opt(OptOps.GROUPTOP, 0, 0) in actions:
      assert len([x for x in lins if x.applied_opts[0] == Opt(OptOps.GROUPTOP, axis=0, arg=3)]) == 0, "did not de-dup GROUPTOP"

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.tensor_cores, "test requires tensor cores")
  def test_search_over_shape(self):
    from test.test_linearizer import helper_realized_ast
    from tinygrad.opt.search import get_kernel_actions

    dtype_pairs = [(tc.dtype_in, tc.dtype_out) for tc in Device[Device.DEFAULT].renderer.tensor_cores]
    multi_shape_dtype_pairs = [dts for dts in dtype_pairs if dtype_pairs.count(dts) > 1]

    if len(multi_shape_dtype_pairs) == 0: raise unittest.SkipTest("only one tc available per dtype pair to search over")

    for (dtype_in, dtype_out) in multi_shape_dtype_pairs:
      a = Tensor.rand(16, 16, dtype=dtype_in)
      b = Tensor.rand(16, 16, dtype=dtype_in)
      realized_ast, _ = helper_realized_ast(a.matmul(b, dtype=dtype_out))

      lins = get_kernel_actions(Kernel(realized_ast)).values()
      assert len(set(lin.tensor_core.dims for lin in lins if lin.tensor_core is not None)) > 1

  def test_get_kernel_actions_preserves_actions_state(self):
    from test.test_linearizer import helper_realized_ast
    from tinygrad.opt.search import get_kernel_actions
    a = Tensor.rand(16, 16)
    b = Tensor.rand(16, 16)
    realized_ast, _ = helper_realized_ast(a @ b)
    actions_before = actions.copy()
    get_kernel_actions(Kernel(realized_ast))
    actions_after = actions.copy()
    assert actions_after == actions_before, "actions state was not preserved"

  @unittest.skip("invalid reduce now")
  def test_filter_global_buffer(self):
    # taken from https://github.com/tinygrad/tinygrad/issues/4612
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.VIEW, dtypes.float.ptr(256), arg=ShapeTracker(views=(View(shape=(1, 1, 256), strides=(0, 0, 1), offset=0, mask=None, contiguous=True),)), src=( # noqa: E501
          UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(256), arg=0, src=()),)),
        UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.MAX, (1,)), src=(
          UOp(Ops.MUL, dtypes.float, arg=None, src=(
            UOp(Ops.ADD, dtypes.float, arg=None, src=(
              UOp(Ops.ADD, dtypes.float, arg=None, src=(
                UOp(Ops.ADD, dtypes.float, arg=None, src=(
                  UOp(Ops.ADD, dtypes.float, arg=None, src=(
                    UOp(Ops.ADD, dtypes.float, arg=None, src=(
                      UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                        UOp(Ops.VIEW, dtypes.float.ptr(64128), arg=ShapeTracker(views=(View(shape=(384768,), strides=(1,), offset=0, mask=((0, 64128),), contiguous=False), View(shape=(1, 501, 256), strides=(0, 1, 501), offset=256512, mask=None, contiguous=False))), src=( # noqa: E501
                          UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(64128), arg=1, src=()),)),)),
                      UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                        UOp(Ops.VIEW, dtypes.float.ptr(64128), arg=ShapeTracker(views=(View(shape=(384768,), strides=(1,), offset=-64128, mask=((64128, 128256),), contiguous=False), View(shape=(1, 501, 256), strides=(0, 1, 501), offset=256512, mask=None, contiguous=False))), src=( # noqa: E501
                          UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(64128), arg=2, src=()),)),)),)),
                    UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                      UOp(Ops.VIEW, dtypes.float.ptr(64128), arg=ShapeTracker(views=(View(shape=(384768,), strides=(1,), offset=-128256, mask=((128256, 192384),), contiguous=False), View(shape=(1, 501, 256), strides=(0, 1, 501), offset=256512, mask=None, contiguous=False))), src=( # noqa: E501
                        UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(64128), arg=3, src=()),)),)),)),
                  UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                    UOp(Ops.VIEW, dtypes.float.ptr(64128), arg=ShapeTracker(views=(View(shape=(384768,), strides=(1,), offset=-192384, mask=((192384, 256512),), contiguous=False), View(shape=(1, 501, 256), strides=(0, 1, 501), offset=256512, mask=None, contiguous=False))), src=( # noqa: E501
                      UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(64128), arg=4, src=()),)),)),)),
                UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                  UOp(Ops.VIEW, dtypes.float.ptr(64128), arg=ShapeTracker(views=(View(shape=(384768,), strides=(1,), offset=-256512, mask=((256512, 320640),), contiguous=False), View(shape=(1, 501, 256), strides=(0, 1, 501), offset=256512, mask=None, contiguous=False))), src=( # noqa: E501
                    UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(64128), arg=5, src=()),)),)),)),
              UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                UOp(Ops.VIEW, dtypes.float.ptr(64128), arg=ShapeTracker(views=(View(shape=(384768,), strides=(1,), offset=-320640, mask=((320640, 384768),), contiguous=False), View(shape=(1, 501, 256), strides=(0, 1, 501), offset=256512, mask=None, contiguous=False))), src=( # noqa: E501
                  UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(64128), arg=6, src=()),)),)),)),
            UOp(Ops.CONST, dtypes.float, arg=1.4285714285714286, src=(
              UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 501, 256), strides=(0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),)),)) # noqa: E501
    lin = Kernel(ast)

    bufs = bufs_from_lin(lin)
    best_lin = beam_search(lin, bufs, 2)
    assert best_lin
    # need disable_cache to trigger.
    tm = time_linearizer(best_lin, bufs, allow_test_size=False, cnt=2, disable_cache=True)
    assert tm

  def test_beam_unnamed_kernels(self):
    a = Tensor.rand(100)
    b = Tensor.rand(100)
    si = (a+b).schedule()[-1]
    lin = Kernel(si.ast)
    bufs = bufs_from_lin(lin)
    # TODO: beam should have better instrumentation so we don't have to check this indirect thing
    kcount = len(Kernel.kernel_cnt)
    beam_search(lin, bufs, 3, disable_cache=True)
    self.assertEqual(kcount, len(Kernel.kernel_cnt))

if __name__ == '__main__':
  unittest.main()
