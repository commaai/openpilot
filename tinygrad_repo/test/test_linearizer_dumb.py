# ruff: noqa: E501
# tests where the Linearizer is doing something dumb
# like test_linearizer_failures, but they don't have to fail

import unittest
from test.helpers import ast_const
from tinygrad import Device, dtypes
from tinygrad.device import is_dtype_supported
from tinygrad.ops import UOp, Ops
from tinygrad.helpers import getenv
from tinygrad.shape.shapetracker import ShapeTracker, View
from tinygrad.engine.search import Opt, OptOps
from tinygrad.codegen.kernel import Kernel

class TestLinearizerDumb(unittest.TestCase):
  @unittest.skipUnless(Device.DEFAULT == "METAL", "only tested on METAL")
  def test_unmerged_ifs(self):
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(64, 1, 512, 7, 7, 1, 1, 1), strides=(25088, 0, 49, 7, 1, 0, 0, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.MAX, dtypes.half, arg=None, src=(
          UOp(Ops.MUL, dtypes.half, arg=None, src=(
            UOp(Ops.CAST, dtypes.half, arg=None, src=(
              UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (5, 6, 7)), src=(
                UOp(Ops.CAST, dtypes.float, arg=None, src=(
                  UOp(Ops.MUL, dtypes.half, arg=None, src=(
                    UOp(Ops.LOAD, dtypes.half, arg=None, src=(
                      UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(), arg=1, src=()),
                      UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 64, 1, 512, 4, 9, 4, 9), strides=(0, 25088, 0, 49, 0, 7, 0, 1), offset=-8, mask=((0, 1), (0, 64), (0, 1), (0, 512), (0, 4), (1, 8), (0, 4), (1, 8)), contiguous=False), View(shape=(64, 1, 512, 7, 7, 512, 3, 3), strides=(663552, 0, 0, 36, 1, 1296, 360, 10), offset=0, mask=None, contiguous=False))), src=()),)),
                    UOp(Ops.LOAD, dtypes.half, arg=None, src=(
                      UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(), arg=2, src=()),
                      UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(64, 1, 512, 7, 7, 512, 3, 3), strides=(0, 0, 4608, 0, 0, 9, 3, 1), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),)),)),
            ast_const(dtypes.half, 0.9999950000374996, st_src=(
              UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(64, 1, 512, 7, 7, 1, 1, 1), strides=(0, 0, 0, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),
          ast_const(dtypes.half, 0.0, st_src=(
            UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(64, 1, 512, 7, 7, 1, 1, 1), strides=(0, 0, 0, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),))
    opts = [Opt(op=OptOps.TC, axis=2, amt=2), Opt(op=OptOps.UPCAST, axis=2, amt=0), Opt(op=OptOps.UNROLL, axis=1, amt=0)]
    k = Kernel(ast, opts=Device["METAL"].renderer)
    k.required_optimizations()
    for opt in opts: k.apply_opt(opt)
    prg = k.to_program()
    print(prg.src)
    Device[Device.DEFAULT].compiler.compile_cached(prg.src)
    gate_count = len([x for x in prg.src.splitlines() if "if" in x])
    assert gate_count == 1, f"must have only one gate {gate_count} != 1"
    assert len([u for u in k.uops if u.op is Ops.IF]) == 1, "must have a single IF"

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "need local")
  def test_max_simplify_and_cancel(self):
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1000, 1), strides=(1, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.MUL, dtypes.int, arg=None, src=(
          UOp(Ops.CAST, dtypes.int, arg=None, src=(
            UOp(Ops.CMPNE, dtypes.bool, arg=None, src=(
              UOp(Ops.CMPNE, dtypes.bool, arg=None, src=(
                UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                  UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=1, src=()),
                  UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1000, 1), strides=(1, 0), offset=0, mask=None, contiguous=True),)), src=()),)),
                UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                  UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=2, src=()),
                  UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1000, 1), strides=(0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),
              ast_const(dtypes.bool, True, st_src=(
                UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1000, 1), strides=(0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),
          UOp(Ops.ADD, dtypes.int, arg=None, src=(
            UOp(Ops.REDUCE_AXIS, dtypes.int, arg=(Ops.ADD, (1,)), src=(
              ast_const(dtypes.int, -1, st_src=(
                UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1001, 1999), strides=(0, 0), offset=0, mask=((0, 1001), (999, 1999)), contiguous=False), View(shape=(1000, 1000), strides=(1, 2000), offset=0, mask=None, contiguous=False))), src=()),)),)),
            ast_const(dtypes.int, 1000, st_src=(
              UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1000, 1), strides=(0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),)),))
    opts = [Opt(op=OptOps.UNROLL, axis=0, amt=4), Opt(op=OptOps.LOCAL, axis=0, amt=8)]
    k = Kernel(ast, opts=Device[Device.DEFAULT].renderer)
    k.required_optimizations()
    for opt in opts: k.apply_opt(opt)
    prg = k.to_program()
    print(prg.src)
    assert prg.uops is not None and not any(uop.op is Ops.MAX for uop in prg.uops), "leftover MAX"

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "need local")
  def test_expander_new_srcs(self):
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(25, 1), strides=(1, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (1,)), src=(
          UOp(Ops.LOAD, dtypes.float, arg=None, src=(
            UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=1, src=()),
            UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(26, 49), strides=(0, -1), offset=48, mask=((0, 26), (24, 49)), contiguous=False), View(shape=(25, 25), strides=(1, 50), offset=0, mask=None, contiguous=False))), src=()),)),)),)),))
    opts = [Opt(op=OptOps.GROUP, axis=0, amt=0), Opt(op=OptOps.PADTO, axis=0, amt=32), Opt(op=OptOps.LOCAL, axis=0, amt=4), Opt(op=OptOps.UPCAST, axis=0, amt=0)]
    k = Kernel(ast, opts=Device[Device.DEFAULT].renderer)
    k.required_optimizations()
    for opt in opts: k.apply_opt(opt)
    prg = k.to_program()
    print(prg.src)
    if_uops = [u for u in k.uops if u.op is Ops.IF]
    self.assertIn(len(if_uops), {1,2,3})
    conditions = if_uops[0].src[0].toposort
    self.assertLessEqual(len(conditions), 9)

  # this was a bug in embedding, someday we should fold this anyway
  @unittest.skipUnless(is_dtype_supported(dtypes.half), f"half dtype not supported on {Device.DEFAULT}")
  def test_llama_embedding(self):
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(4096, 1, 1), strides=(1, 0, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.CAST, dtypes.half, arg=None, src=(
          UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (1,)), src=(
            UOp(Ops.CAST, dtypes.float, arg=None, src=(
              UOp(Ops.MUL, dtypes.half, arg=None, src=(
                UOp(Ops.CAST, dtypes.half, arg=None, src=(
                  UOp(Ops.CMPNE, dtypes.bool, arg=None, src=(
                    UOp(Ops.CMPNE, dtypes.bool, arg=None, src=(
                      UOp(Ops.ADD, dtypes.int, arg=None, src=(
                        UOp(Ops.REDUCE_AXIS, dtypes.int, arg=(Ops.ADD, (2,)), src=(
                          ast_const(dtypes.int, 1, st_src=(
                            UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(32001, 63999), strides=(0, 0), offset=0, mask=((0, 32001), (31999, 63999)), contiguous=False), View(shape=(4096, 32000, 32000), strides=(0, 1, 64000), offset=0, mask=None, contiguous=False))), src=()),)),)),
                        ast_const(dtypes.int, -1, st_src=(
                          UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(4096, 32000, 1), strides=(0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),
                      UOp(Ops.LOAD, dtypes.int, arg=None, src=(
                        UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), arg=1, src=()),
                        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(4096, 32000, 1), strides=(0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),
                    ast_const(dtypes.bool, True, st_src=(
                      UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(4096, 32000, 1), strides=(0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),
                UOp(Ops.LOAD, dtypes.half, arg=None, src=(
                  UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(), arg=2, src=()),
                  UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(4096, 32000, 1), strides=(1, 4096, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),)),)),)),))
    k = Kernel(ast, opts=Device[Device.DEFAULT].renderer)
    prg = k.to_program()
    print(prg.src)

  # from process replay https://github.com/tinygrad/tinygrad/actions/runs/10389229290/job/28766762085#step:18:6490
  @unittest.expectedFailure
  def test_unaligns_idxs(self):
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(3, 1, 1), strides=(1, 0, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (2,)), src=(
          UOp(Ops.MUL, dtypes.float, arg=None, src=(
            UOp(Ops.CAST, dtypes.float, arg=None, src=(
              UOp(Ops.CMPNE, dtypes.bool, arg=None, src=(
                UOp(Ops.CMPNE, dtypes.bool, arg=None, src=(
                  UOp(Ops.LOAD, dtypes.long, arg=None, src=(
                    UOp(Ops.DEFINE_GLOBAL, dtypes.long.ptr(), arg=1, src=()),
                    UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(3, 1, 5), strides=(1, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),
                  UOp(Ops.CAST, dtypes.long, arg=None, src=(
                    UOp(Ops.LOAD, dtypes.int, arg=None, src=(
                      UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), arg=2, src=()),
                      UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(3, 1, 5), strides=(0, 0, 1), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),
                ast_const(dtypes.bool, True, st_src=(
                  UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(3, 1, 5), strides=(0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),
            UOp(Ops.LOAD, dtypes.float, arg=None, src=(
              UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=3, src=()),
              UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(3, 1, 5), strides=(0, 0, 1), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),)),))
    opts = [Opt(op=OptOps.UNROLL, axis=0, amt=0), Opt(op=OptOps.LOCAL, axis=0, amt=3)]
    k = Kernel(ast, opts=Device[Device.DEFAULT].renderer)
    for opt in opts: k.apply_opt(opt)
    prg = k.to_program()
    print(prg.src)
    load_idxs = [x.src[1] for x in k.uops if x.op is Ops.LOAD and x.src[0].arg == 3]
    assert load_idxs[0] < load_idxs[1], f"first loaded idx {load_idxs[0].arg} then {load_idxs[1].arg}!"

  @unittest.expectedFailure
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.supports_float4, "need float4")
  def test_unrolled_float4_align(self):
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 1), strides=(0, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (0, 1)), src=(
          UOp(Ops.WHERE, dtypes.float, arg=None, src=(
            UOp(Ops.CMPNE, dtypes.bool, arg=None, src=(
              UOp(Ops.CMPNE, dtypes.bool, arg=None, src=(
                UOp(Ops.LOAD, dtypes.long, arg=None, src=(
                  UOp(Ops.DEFINE_GLOBAL, dtypes.long.ptr(), arg=1, src=()),
                  UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(3, 6), strides=(6, 1), offset=0, mask=None, contiguous=True),)), src=()),)),
                ast_const(dtypes.long, -1, st_src=(
                  UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(3, 6), strides=(0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),
              ast_const(dtypes.bool, True, st_src=(
                UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(3, 6), strides=(0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),
            ast_const(dtypes.float, 0.0, st_src=(
              UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(3, 6), strides=(0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),
            UOp(Ops.LOAD, dtypes.float, arg=None, src=(
              UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=2, src=()),
              UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(3, 6), strides=(6, 1), offset=0, mask=None, contiguous=True),)), src=()),)),)),)),)),))
    opts = [Opt(op=OptOps.UNROLL, axis=0, amt=0)]
    k = Kernel(ast, opts=Device[Device.DEFAULT].renderer)
    for opt in opts: k.apply_opt(opt)
    prg = k.to_program()
    print(prg.src)
    load_idxs = [x.src[1] for x in k.uops if x.op is Ops.LOAD and x.src[0].arg == 2]
    assert load_idxs[0] < load_idxs[1], f"first loaded idx {load_idxs[0].arg} then {load_idxs[1].arg}!"

  @unittest.expectedFailure
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.supports_float4, "need float4")
  @unittest.skipIf(getenv("PTX"), "this is somehow correct in PTX")
  def test_upcasted_stores_out_of_order(self):
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(4, 5, 13, 1, 1, 1, 1, 1, 4, 3, 3), strides=(2340, 468, 36, 0, 0, 0, 0, 0, 9, 3, 1), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (6,)), src=(
          UOp(Ops.MUL, dtypes.float, arg=None, src=(
            UOp(Ops.LOAD, dtypes.float, arg=None, src=(
              UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=1, src=()),
              UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(4, 5, 13, 1, 1, 1, 4, 1, 4, 3, 3), strides=(0, 0, 0, 0, 0, 0, 1, 0, 4, 48, 16), offset=0, mask=None, contiguous=False),)), src=()),)),
            UOp(Ops.LOAD, dtypes.float, arg=None, src=(
              UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=2, src=()),
              UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(4, 5, 13, 1, 1, 1, 4, 1, 4, 3, 3), strides=(260, 13, 1, 0, 0, 0, 65, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),)),))
    opts = [Opt(op=OptOps.UPCAST, axis=3, amt=0), Opt(op=OptOps.UPCAST, axis=2, amt=0)]
    k = Kernel(ast, opts=Device[Device.DEFAULT].renderer)
    for opt in opts: k.apply_opt(opt)
    prg = k.to_program()
    print(prg.src)
    store_idxs = [x.src[1] for x in k.uops if x.op is Ops.STORE]
    for i in range(len(store_idxs) - 1):
      first_bounds = store_idxs[i].vmin+store_idxs[i].vmax
      next_bounds = store_idxs[i+1].vmin+store_idxs[i+1].vmax
      assert first_bounds < next_bounds, f"first stored (max) idx {first_bounds} then {next_bounds}!"

if __name__ == '__main__':
  unittest.main()
