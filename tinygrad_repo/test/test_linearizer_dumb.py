# ruff: noqa: E501
# tests where the Linearizer is doing something dumb
# like test_linearizer_failures, but they don't have to fail

import unittest
from tinygrad import Device, dtypes
from tinygrad.device import is_dtype_supported
from tinygrad.uop.ops import UOp, Ops, AxisType, KernelInfo
from tinygrad.shape.shapetracker import ShapeTracker, View
from tinygrad.codegen.opt.search import Opt, OptOps
from tinygrad.engine.realize import get_program
from tinygrad.renderer.ptx import PTXRenderer

class TestLinearizerFailure(unittest.TestCase):
  @unittest.skipUnless(Device.DEFAULT == "METAL", "only tested on METAL")
  def test_failure_beam_mnist(self):
    c0 = UOp(Ops.DEFINE_GLOBAL, dtypes.uchar.ptr(4014080), arg=0, src=())
    c1 = UOp.range(UOp.const(dtypes.int, 512), 0, AxisType.GLOBAL)
    c2 = UOp.range(UOp.const(dtypes.int, 784), 1, AxisType.GLOBAL)
    c3 = UOp.range(UOp.const(dtypes.int, 10), 3, AxisType.GLOBAL)
    c4 = UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(512), arg=1, src=())
    c5 = c4.index(c1, UOp.const(dtypes.bool, True)).load()
    c6 = UOp.range(UOp.const(dtypes.int, 6000), 1004, AxisType.REDUCE)
    c7 = UOp.range(UOp.const(dtypes.int, 3750), 2006, AxisType.REDUCE)
    c8 = UOp.range(UOp.const(dtypes.int, 16), 2007, AxisType.GROUP_REDUCE)
    c9 = UOp(Ops.DEFINE_GLOBAL, dtypes.uchar.ptr(47040000), arg=2, src=())
    c10 = c9.index((((c3*UOp.const(dtypes.int, 4704000))+c2)+(c6*UOp.const(dtypes.int, 784))), UOp.const(dtypes.bool, True)).load()
    c11 = c5.alu(Ops.CMPNE, ((((c3*UOp.const(dtypes.int, 6000))+c6)+((c7*UOp.const(dtypes.int, 16))+c8)).alu(Ops.CMPLT, UOp.const(dtypes.int, 59999)).where(UOp.const(dtypes.int, 0), UOp.const(dtypes.int, 1)).reduce(c7, c8, arg=Ops.ADD)+UOp.const(dtypes.int, -1))).where(UOp.const(dtypes.uchar, 0), c10).reduce(c6, arg=Ops.ADD)
    c12 = c0.index((((c1*UOp.const(dtypes.int, 7840))+(c2*UOp.const(dtypes.int, 10)))+c3), UOp.const(dtypes.bool, True)).store(c11, c1, c2, c3)
    ast = c12.sink(arg=KernelInfo(name='test', axis_types=(), dont_use_locals=False, applied_opts=(Opt(op=OptOps.GROUP, axis=1, arg=16),), opts_to_apply=None))
    _ = get_program(ast, Device["METAL"].renderer)

class TestLinearizerDumb(unittest.TestCase):
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "need local")
  def test_max_simplify_and_cancel(self):
    c0 = UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(1000), arg=0, src=())
    c1 = c0.view(ShapeTracker(views=(View(shape=(1000, 1), strides=(1, 0), offset=0, mask=None, contiguous=True),)))
    c2 = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(1000), arg=1, src=())
    c3 = c2.view(ShapeTracker(views=(View(shape=(1000, 1), strides=(1, 0), offset=0, mask=None, contiguous=True),)))
    c4 = c3.load()
    c5 = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(1), arg=2, src=())
    c6 = c5.view(ShapeTracker(views=(View(shape=(1000, 1), strides=(0, 0), offset=0, mask=None, contiguous=False),)))
    c7 = c6.load()
    c8 = UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1000, 1), strides=(0, 0), offset=0, mask=None, contiguous=False),)), src=())
    c9 = UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1001, 1999), strides=(0, 0), offset=0, mask=((0, 1001), (999, 1999)), contiguous=False), View(shape=(1000, 1000), strides=(1, 2000), offset=0, mask=None, contiguous=False))), src=())
    c10 = UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1000, 1000), strides=(0, 0), offset=0, mask=None, contiguous=False),)), src=())
    c11 = c1.store((c4.alu(Ops.CMPNE, c7).alu(Ops.CMPNE, UOp.const(dtypes.bool, True, src=c8)).cast(dtypes.int)*(c9.f(Ops.VALID, dtype=dtypes.bool).where(UOp.const(dtypes.int, -1, src=c10), UOp.const(dtypes.int, 0, src=c10)).f(Ops.REDUCE_AXIS, arg=(Ops.ADD, (1,)))+UOp.const(dtypes.int, 1000, src=c8))))
    ast = c11.sink()
    #opts = [Opt(op=OptOps.UNROLL, axis=0, arg=4), Opt(op=OptOps.LOCAL, axis=0, arg=8)]
    opts = [Opt(op=OptOps.LOCAL, axis=0, arg=8)]
    prg = get_program(ast, Device[Device.DEFAULT].renderer, opts)
    print(prg.src)
    assert prg.uops is not None and not any(uop.op is Ops.MAX for uop in prg.uops), "leftover MAX"

  # this was a bug in embedding, someday we should fold this anyway
  @unittest.skipUnless(is_dtype_supported(dtypes.half), f"half dtype not supported on {Device.DEFAULT}")
  def test_llama_embedding(self):
    c0 = UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(4096), arg=0, src=())
    c1 = c0.view(ShapeTracker(views=(View(shape=(4096, 1, 1), strides=(1, 0, 0), offset=0, mask=None, contiguous=True),)))
    c2 = UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(32001, 63999), strides=(0, 0), offset=0, mask=((0, 32001), (31999, 63999)), contiguous=False), View(shape=(4096, 32000, 32000), strides=(0, 1, 64000), offset=0, mask=None, contiguous=False))), src=())
    c3 = UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(4096, 32000, 32000), strides=(0, 0, 0), offset=0, mask=None, contiguous=False),)), src=())
    c4 = UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(4096, 32000, 1), strides=(0, 0, 0), offset=0, mask=None, contiguous=False),)), src=())
    c5 = UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(1), arg=1, src=())
    c6 = c5.view(ShapeTracker(views=(View(shape=(4096, 32000, 1), strides=(0, 0, 0), offset=0, mask=None, contiguous=False),)))
    c7 = c6.load()
    c8 = UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(131072000), arg=2, src=())
    c9 = c8.view(ShapeTracker(views=(View(shape=(4096, 32000, 1), strides=(1, 4096, 0), offset=0, mask=None, contiguous=False),)))
    c10 = c9.load()
    c11 = c1.store(((c2.f(Ops.VALID, dtype=dtypes.bool).where(UOp.const(dtypes.int, 1, src=c3), UOp.const(dtypes.int, 0, src=c3)).f(Ops.REDUCE_AXIS, arg=(Ops.ADD, (2,)))+UOp.const(dtypes.int, -1, src=c4)).alu(Ops.CMPNE, c7).alu(Ops.CMPNE, UOp.const(dtypes.bool, True, src=c4)).cast(dtypes.half)*c10).cast(dtypes.float).f(Ops.REDUCE_AXIS, arg=(Ops.ADD, (1,))).cast(dtypes.half))
    ast = c11.sink()
    prg = get_program(ast, Device[Device.DEFAULT].renderer)
    print(prg.src)

  @unittest.expectedFailure
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.supports_float4, "need float4")
  def test_unrolled_float4_align(self):
    c0 = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(1), arg=0, src=())
    c1 = c0.view(ShapeTracker(views=(View(shape=(1, 1), strides=(0, 0), offset=0, mask=None, contiguous=True),)))
    c2 = UOp(Ops.DEFINE_GLOBAL, dtypes.long.ptr(18), arg=1, src=())
    c3 = c2.view(ShapeTracker(views=(View(shape=(3, 6), strides=(6, 1), offset=0, mask=None, contiguous=True),)))
    c4 = c3.load()
    c5 = UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(3, 6), strides=(0, 0), offset=0, mask=None, contiguous=False),)), src=())
    c6 = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(18), arg=2, src=())
    c7 = c6.view(ShapeTracker(views=(View(shape=(3, 6), strides=(6, 1), offset=0, mask=None, contiguous=True),)))
    c8 = c7.load()
    c9 = c1.store(c4.alu(Ops.CMPNE, UOp.const(dtypes.long, -1, src=c5)).alu(Ops.CMPNE, UOp.const(dtypes.bool, True, src=c5)).where(UOp.const(dtypes.float, 0.0, src=c5), c8).f(Ops.REDUCE_AXIS, arg=(Ops.ADD, (0, 1))))
    ast = c9.sink()
    opts = [Opt(op=OptOps.UNROLL, axis=0, arg=0)]
    prg = get_program(ast, Device[Device.DEFAULT].renderer, opts)
    print(prg.src)
    load_idxs = [x.src[1] for x in prg.uops if x.op is Ops.LOAD and x.src[0].arg == 2]
    assert load_idxs[0] < load_idxs[1], f"first loaded idx {load_idxs[0].arg} then {load_idxs[1].arg}!"

  @unittest.expectedFailure
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.supports_float4, "need float4")
  @unittest.skipIf(isinstance(Device[Device.DEFAULT].renderer, PTXRenderer), "this is somehow correct in PTX")
  def test_upcasted_stores_out_of_order(self):
    c0 = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(9360), arg=0, src=())
    c1 = c0.view(ShapeTracker(views=(View(shape=(4, 5, 13, 1, 1, 1, 1, 1, 4, 3, 3), strides=(2340, 468, 36, 0, 0, 0, 0, 0, 9, 3, 1), offset=0, mask=None, contiguous=True),)))
    c2 = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(144), arg=1, src=())
    c3 = c2.view(ShapeTracker(views=(View(shape=(4, 5, 13, 1, 1, 1, 4, 1, 4, 3, 3), strides=(0, 0, 0, 0, 0, 0, 1, 0, 4, 48, 16), offset=0, mask=None, contiguous=False),)))
    c4 = c3.load()
    c5 = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(1040), arg=2, src=())
    c6 = c5.view(ShapeTracker(views=(View(shape=(4, 5, 13, 1, 1, 1, 4, 1, 4, 3, 3), strides=(260, 13, 1, 0, 0, 0, 65, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),)))
    c7 = c6.load()
    c8 = c1.store((c4*c7).f(Ops.REDUCE_AXIS, arg=(Ops.ADD, (6,))))
    ast = c8.sink()
    opts = [Opt(op=OptOps.UPCAST, axis=3, arg=0), Opt(op=OptOps.UPCAST, axis=2, arg=0)]
    prg = get_program(ast, Device[Device.DEFAULT].renderer, opts)
    print(prg.src)
    store_idxs = [x.src[1] for x in prg.uops if x.op is Ops.STORE]
    for i in range(len(store_idxs) - 1):
      first_bounds = store_idxs[i].vmin+store_idxs[i].vmax
      next_bounds = store_idxs[i+1].vmin+store_idxs[i+1].vmax
      assert first_bounds < next_bounds, f"first stored (max) idx {first_bounds} then {next_bounds}!"

if __name__ == '__main__':
  unittest.main()
