# ruff: noqa: E501
# tests where the Linearizer is doing something dumb
# like test_linearizer_failures, but they don't have to fail

import unittest
from tinygrad import Device, dtypes
from tinygrad.device import is_dtype_supported
from tinygrad.uop.ops import UOp, Ops
from tinygrad.helpers import getenv
from tinygrad.shape.shapetracker import ShapeTracker, View
from tinygrad.codegen.opt.search import Opt, OptOps
from tinygrad.codegen.opt.kernel import Kernel
from tinygrad.engine.realize import get_program

class TestLinearizerDumb(unittest.TestCase):
  @unittest.skipUnless(Device.DEFAULT == "METAL", "only tested on METAL")
  def test_unmerged_ifs(self):
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.VIEW, dtypes.half.ptr(1605632), arg=ShapeTracker(views=(View(shape=(64, 1, 512, 7, 7, 1, 1, 1), strides=(25088, 0, 49, 7, 1, 0, 0, 0), offset=0, mask=None, contiguous=True),)), src=(
          UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(1605632), arg=0, src=()),)),
        UOp(Ops.MAX, dtypes.half, arg=None, src=(
          UOp(Ops.MUL, dtypes.half, arg=None, src=(
            UOp(Ops.CAST, dtypes.half, arg=None, src=(
              UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (5, 6, 7)), src=(
                UOp(Ops.CAST, dtypes.float, arg=None, src=(
                  UOp(Ops.MUL, dtypes.half, arg=None, src=(
                    UOp(Ops.LOAD, dtypes.half, arg=None, src=(
                      UOp(Ops.VIEW, dtypes.half.ptr(1605632), arg=ShapeTracker(views=(View(shape=(1, 64, 1, 512, 4, 9, 4, 9), strides=(0, 25088, 0, 49, 0, 7, 0, 1), offset=-8, mask=((0, 1), (0, 64), (0, 1), (0, 512), (0, 4), (1, 8), (0, 4), (1, 8)), contiguous=False), View(shape=(64, 1, 512, 7, 7, 512, 3, 3), strides=(663552, 0, 0, 36, 1, 1296, 360, 10), offset=0, mask=None, contiguous=False))), src=(
                        UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(1605632), arg=1, src=()),)),)),
                    UOp(Ops.LOAD, dtypes.half, arg=None, src=(
                      UOp(Ops.VIEW, dtypes.half.ptr(2359296), arg=ShapeTracker(views=(View(shape=(64, 1, 512, 7, 7, 512, 3, 3), strides=(0, 0, 4608, 0, 0, 9, 3, 1), offset=0, mask=None, contiguous=False),)), src=(
                        UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(2359296), arg=2, src=()),)),)),)),)),)),)),
            UOp(Ops.CONST, dtypes.half, arg=0.9999950000374996, src=(
              x16:=UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(64, 1, 512, 7, 7, 1, 1, 1), strides=(0, 0, 0, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),
          UOp(Ops.CONST, dtypes.half, arg=0.0, src=(
             x16,)),)),)),))
    opts = [Opt(op=OptOps.TC, axis=2, arg=(-1, 2, 1)), Opt(op=OptOps.UPCAST, axis=2, arg=0), Opt(op=OptOps.UNROLL, axis=1, arg=0)]
    k = Kernel(ast, opts=Device["METAL"].renderer)
    k.apply_opts(opts)
    prg = get_program(k.get_optimized_ast(), k.opts)
    print(prg.src)
    Device[Device.DEFAULT].compiler.compile_cached(prg.src)
    gate_count = len([x for x in prg.src.splitlines() if "if" in x])
    assert gate_count == 1, f"must have only one gate {gate_count} != 1"
    assert len([u for u in prg.uops if u.op is Ops.IF]) == 1, "must have a single IF"

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "need local")
  def test_max_simplify_and_cancel(self):
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.VIEW, dtypes.int.ptr(1000), arg=ShapeTracker(views=(View(shape=(1000, 1), strides=(1, 0), offset=0, mask=None, contiguous=True),)), src=(
          UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(1000), arg=0, src=()),)),
        UOp(Ops.MUL, dtypes.int, arg=None, src=(
          UOp(Ops.CAST, dtypes.int, arg=None, src=(
            UOp(Ops.CMPNE, dtypes.bool, arg=None, src=(
              UOp(Ops.CMPNE, dtypes.bool, arg=None, src=(
                UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                  UOp(Ops.VIEW, dtypes.float.ptr(1000), arg=ShapeTracker(views=(View(shape=(1000, 1), strides=(1, 0), offset=0, mask=None, contiguous=True),)), src=(
                    UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(1000), arg=1, src=()),)),)),
                UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                  UOp(Ops.VIEW, dtypes.float.ptr(1), arg=ShapeTracker(views=(View(shape=(1000, 1), strides=(0, 0), offset=0, mask=None, contiguous=False),)), src=(
                    UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(1), arg=2, src=()),)),)),)),
              UOp(Ops.CONST, dtypes.bool, arg=True, src=(
                x14:=UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1000, 1), strides=(0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),
          UOp(Ops.ADD, dtypes.int, arg=None, src=(
            UOp(Ops.REDUCE_AXIS, dtypes.int, arg=(Ops.ADD, (1,)), src=(
              UOp(Ops.WHERE, dtypes.int, arg=None, src=(
                UOp(Ops.VALID, dtypes.bool, arg=None, src=(
                  UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1001, 1999), strides=(0, 0), offset=0, mask=((0, 1001), (999, 1999)), contiguous=False), View(shape=(1000, 1000), strides=(1, 2000), offset=0, mask=None, contiguous=False))), src=()),)),
                UOp(Ops.CONST, dtypes.int, arg=-1, src=(
                  x21:=UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1000, 1000), strides=(0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),
                UOp(Ops.CONST, dtypes.int, arg=0, src=(
                   x21,)),)),)),
            UOp(Ops.CONST, dtypes.int, arg=1000, src=(
               x14,)),)),)),)),))
    opts = [Opt(op=OptOps.UNROLL, axis=0, arg=4), Opt(op=OptOps.LOCAL, axis=0, arg=8)]
    k = Kernel(ast, opts=Device[Device.DEFAULT].renderer)
    k.apply_opts(opts)
    prg = get_program(k.get_optimized_ast(), k.opts)
    print(prg.src)
    assert prg.uops is not None and not any(uop.op is Ops.MAX for uop in prg.uops), "leftover MAX"

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "need local")
  @unittest.skip("not applicable")
  def test_expander_new_srcs(self):
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.VIEW, dtypes.float.ptr(25), arg=ShapeTracker(views=(View(shape=(25, 1), strides=(1, 0), offset=0, mask=None, contiguous=True),)), src=(
          UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(25), arg=0, src=()),)),
        UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (1,)), src=(
          UOp(Ops.LOAD, dtypes.float, arg=None, src=(
            UOp(Ops.VIEW, dtypes.float.ptr(25), arg=ShapeTracker(views=(View(shape=(26, 49), strides=(0, -1), offset=48, mask=((0, 26), (24, 49)), contiguous=False), View(shape=(25, 25), strides=(1, 50), offset=0, mask=None, contiguous=False))), src=(
              UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(25), arg=1, src=()),)),)),)),)),))
    opts = [Opt(op=OptOps.GROUP, axis=0, arg=0), Opt(op=OptOps.PADTO, axis=0, arg=32), Opt(op=OptOps.LOCAL, axis=0, arg=4), Opt(op=OptOps.UPCAST, axis=0, arg=0)]
    k = Kernel(ast, opts=Device[Device.DEFAULT].renderer)
    k.apply_opts(opts)
    prg = get_program(k.get_optimized_ast(), k.opts)
    print(prg.src)
    if_uops = [u for u in prg.uops if u.op is Ops.IF]
    self.assertIn(len(if_uops), {1,2,3})
    conditions = if_uops[0].src[0].toposort()
    self.assertLessEqual(len(conditions), 9)

  # this was a bug in embedding, someday we should fold this anyway
  @unittest.skipUnless(is_dtype_supported(dtypes.half), f"half dtype not supported on {Device.DEFAULT}")
  def test_llama_embedding(self):
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.VIEW, dtypes.half.ptr(4096), arg=ShapeTracker(views=(View(shape=(4096, 1, 1), strides=(1, 0, 0), offset=0, mask=None, contiguous=True),)), src=(
          UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(4096), arg=0, src=()),)),
        UOp(Ops.CAST, dtypes.half, arg=None, src=(
          UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (1,)), src=(
            UOp(Ops.CAST, dtypes.float, arg=None, src=(
              UOp(Ops.MUL, dtypes.half, arg=None, src=(
                UOp(Ops.CAST, dtypes.half, arg=None, src=(
                  UOp(Ops.CMPNE, dtypes.bool, arg=None, src=(
                    UOp(Ops.CMPNE, dtypes.bool, arg=None, src=(
                      UOp(Ops.ADD, dtypes.int, arg=None, src=(
                        UOp(Ops.REDUCE_AXIS, dtypes.int, arg=(Ops.ADD, (2,)), src=(
                          UOp(Ops.WHERE, dtypes.int, arg=None, src=(
                            UOp(Ops.VALID, dtypes.bool, arg=None, src=(
                              UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(32001, 63999), strides=(0, 0), offset=0, mask=((0, 32001), (31999, 63999)), contiguous=False), View(shape=(4096, 32000, 32000), strides=(0, 1, 64000), offset=0, mask=None, contiguous=False))), src=()),)),
                            UOp(Ops.CONST, dtypes.int, arg=1, src=(
                              x16:=UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(4096, 32000, 32000), strides=(0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),
                            UOp(Ops.CONST, dtypes.int, arg=0, src=(
                               x16,)),)),)),
                        UOp(Ops.CONST, dtypes.int, arg=-1, src=(
                          x19:=UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(4096, 32000, 1), strides=(0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),
                      UOp(Ops.LOAD, dtypes.int, arg=None, src=(
                        UOp(Ops.VIEW, dtypes.int.ptr(1), arg=ShapeTracker(views=(View(shape=(4096, 32000, 1), strides=(0, 0, 0), offset=0, mask=None, contiguous=False),)), src=(
                          UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(1), arg=1, src=()),)),)),)),
                    UOp(Ops.CONST, dtypes.bool, arg=True, src=(
                       x19,)),)),)),
                UOp(Ops.LOAD, dtypes.half, arg=None, src=(
                  UOp(Ops.VIEW, dtypes.half.ptr(131072000), arg=ShapeTracker(views=(View(shape=(4096, 32000, 1), strides=(1, 4096, 0), offset=0, mask=None, contiguous=False),)), src=(
                    UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(131072000), arg=2, src=()),)),)),)),)),)),)),)),))
    k = Kernel(ast, opts=Device[Device.DEFAULT].renderer)
    prg = get_program(k.get_optimized_ast(), k.opts)
    print(prg.src)

  @unittest.expectedFailure
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.supports_float4, "need float4")
  def test_unrolled_float4_align(self):
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.VIEW, dtypes.float.ptr(1), arg=ShapeTracker(views=(View(shape=(1, 1), strides=(0, 0), offset=0, mask=None, contiguous=True),)), src=(
          UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(1), arg=0, src=()),)),
        UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (0, 1)), src=(
          UOp(Ops.WHERE, dtypes.float, arg=None, src=(
            UOp(Ops.CMPNE, dtypes.bool, arg=None, src=(
              UOp(Ops.CMPNE, dtypes.bool, arg=None, src=(
                UOp(Ops.LOAD, dtypes.long, arg=None, src=(
                  UOp(Ops.VIEW, dtypes.long.ptr(18), arg=ShapeTracker(views=(View(shape=(3, 6), strides=(6, 1), offset=0, mask=None, contiguous=True),)), src=(
                    UOp(Ops.DEFINE_GLOBAL, dtypes.long.ptr(18), arg=1, src=()),)),)),
                UOp(Ops.CONST, dtypes.long, arg=-1, src=(
                  x11:=UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(3, 6), strides=(0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),
              UOp(Ops.CONST, dtypes.bool, arg=True, src=(
                 x11,)),)),
            UOp(Ops.CONST, dtypes.float, arg=0.0, src=(
               x11,)),
            UOp(Ops.LOAD, dtypes.float, arg=None, src=(
              UOp(Ops.VIEW, dtypes.float.ptr(18), arg=ShapeTracker(views=(View(shape=(3, 6), strides=(6, 1), offset=0, mask=None, contiguous=True),)), src=(
                UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(18), arg=2, src=()),)),)),)),)),)),))
    opts = [Opt(op=OptOps.UNROLL, axis=0, arg=0)]
    k = Kernel(ast, opts=Device[Device.DEFAULT].renderer)
    k.apply_opts(opts)
    prg = get_program(k.get_optimized_ast(), k.opts)
    print(prg.src)
    load_idxs = [x.src[1] for x in k.uops if x.op is Ops.LOAD and x.src[0].arg == 2]
    assert load_idxs[0] < load_idxs[1], f"first loaded idx {load_idxs[0].arg} then {load_idxs[1].arg}!"

  @unittest.expectedFailure
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.supports_float4, "need float4")
  @unittest.skipIf(getenv("PTX"), "this is somehow correct in PTX")
  def test_upcasted_stores_out_of_order(self):
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.VIEW, dtypes.float.ptr(9360), arg=ShapeTracker(views=(View(shape=(4, 5, 13, 1, 1, 1, 1, 1, 4, 3, 3), strides=(2340, 468, 36, 0, 0, 0, 0, 0, 9, 3, 1), offset=0, mask=None, contiguous=True),)), src=(
          UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(9360), arg=0, src=()),)),
        UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (6,)), src=(
          UOp(Ops.MUL, dtypes.float, arg=None, src=(
            UOp(Ops.LOAD, dtypes.float, arg=None, src=(
              UOp(Ops.VIEW, dtypes.float.ptr(144), arg=ShapeTracker(views=(View(shape=(4, 5, 13, 1, 1, 1, 4, 1, 4, 3, 3), strides=(0, 0, 0, 0, 0, 0, 1, 0, 4, 48, 16), offset=0, mask=None, contiguous=False),)), src=(
                UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(144), arg=1, src=()),)),)),
            UOp(Ops.LOAD, dtypes.float, arg=None, src=(
              UOp(Ops.VIEW, dtypes.float.ptr(1040), arg=ShapeTracker(views=(View(shape=(4, 5, 13, 1, 1, 1, 4, 1, 4, 3, 3), strides=(260, 13, 1, 0, 0, 0, 65, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),)), src=(
                UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(1040), arg=2, src=()),)),)),)),)),)),))
    opts = [Opt(op=OptOps.UPCAST, axis=3, arg=0), Opt(op=OptOps.UPCAST, axis=2, arg=0)]
    k = Kernel(ast, opts=Device[Device.DEFAULT].renderer)
    k.apply_opts(opts)
    prg = get_program(k.get_optimized_ast(), k.opts)
    print(prg.src)
    store_idxs = [x.src[1] for x in k.uops if x.op is Ops.STORE]
    for i in range(len(store_idxs) - 1):
      first_bounds = store_idxs[i].vmin+store_idxs[i].vmax
      next_bounds = store_idxs[i+1].vmin+store_idxs[i+1].vmax
      assert first_bounds < next_bounds, f"first stored (max) idx {first_bounds} then {next_bounds}!"

if __name__ == '__main__':
  unittest.main()
