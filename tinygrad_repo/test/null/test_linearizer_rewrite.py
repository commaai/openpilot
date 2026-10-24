import unittest
from tinygrad import Tensor, Context, Device, dtypes
from tinygrad.codegen import to_program
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.codegen.late.linearizer import do_split_ends
from tinygrad.uop.ops import KernelInfo, AxisType, UOp, Ops
from test.helpers import to_uops_list

class TestLinearizerRewrite(unittest.TestCase):
  def test_range_order(self):
    types = [AxisType.DEVICE, AxisType.GLOBAL, AxisType.LOCAL, AxisType.WARP, AxisType.WEAK, AxisType.LOOP,
             AxisType.UPCAST, AxisType.PLACEHOLDER]
    # Axis type wins over numeric ids, including multi-part ids.
    ranges = [UOp.range(4, 10-i, t).replace(arg=(t, 10-i, j)) for i,t in enumerate(types) for j in (0, 1)]
    self.assertEqual(sorted(ranges[::-1], key=lambda r: r.arg), ranges)
    self.assertEqual(ranges[0].axis_id, (10, 0))
    self.assertEqual(ranges[0].axis_type, AxisType.DEVICE)
    end = do_split_ends(UOp(Ops.NOOP).end(*ranges[::-1]))
    for r in ranges:
      self.assertIs(end.src[1], r)
      end = end.src[0]
    self.assertIs(end.op, Ops.NOOP)

  def test_reduction(self):
    t = Tensor.ones((64,64), device="NULL").contiguous().realize()
    out = (t*2).sum(axis=1)
    with Context(SPLIT_REDUCEOP=0):
      si = out.schedule_linear().src[-1]
      opts_to_apply = []
      opts_to_apply.append(Opt(OptOps.SPLIT, 0, (4, AxisType.UPCAST)))
      opts_to_apply.append(Opt(OptOps.SPLIT, 2, (4, AxisType.UPCAST)))
      ast = si.src[0].replace(arg=KernelInfo(opts_to_apply=tuple(opts_to_apply)))
      prg = to_program(ast, Device.default.renderer)
      print(prg.src[2].arg)
      self.assertNotIn(Ops.REDUCE, [u.op for u in prg.src[1].src])
      ranges = [u for u in prg.src[1].src if u.op is Ops.RANGE]
      self.assertTrue(ranges)
      self.assertTrue(all(r.axis_type is AxisType.WEAK for r in ranges))

  def test_arange(self):
    out = Tensor.arange(32).clone("NULL")
    with Context(SPLIT_REDUCEOP=0):
      si = out.schedule_linear().src[-1]
      opts_to_apply = []
      opts_to_apply.append(Opt(OptOps.SPLIT, 0, (4, AxisType.UPCAST)))
      ast = si.src[0].replace(arg=KernelInfo(opts_to_apply=tuple(opts_to_apply)))
      prg = to_program(ast, Device.default.renderer)
      print(prg.src[2].arg)

  def test_kernel_info(self):
    out = Tensor.arange(4).clone("NULL")
    si = out.schedule_linear().src[-1]

    ast = si.src[0].replace(arg=KernelInfo(opts_to_apply=()))
    prg = to_program(ast, Device.default.renderer)
    assert prg.src[0].arg.applied_opts == (), f"expected no opts, got {prg}"

    prg = to_program(ast.replace(arg=KernelInfo(name="custom")), Device.default.renderer)
    self.assertEqual(prg.src[0].arg.name, "custom")

  def test_dependent_loop_bound(self):
    buf, out, counts = UOp.param(0, dtypes.int, 16), UOp.param(1, dtypes.int, 4), UOp.param(2, dtypes.int, 4)
    outer = UOp.range(4, 0, AxisType.LOOP)
    inner = UOp.range(counts.index(outer).load().maximum(0).minimum(4), 1)
    store = buf.index(outer * 4 + inner).store(UOp.const(1, dtypes.int)).end(inner)
    uops = to_uops_list([out.after(store).index(outer).store(UOp.const(2, dtypes.int))])
    self.assertEqual([u.op for u in uops if u.op in (Ops.RANGE, Ops.STORE, Ops.END)],
                     [Ops.RANGE, Ops.RANGE, Ops.STORE, Ops.END, Ops.STORE, Ops.END])
    rngs, ends = [u for u in uops if u.op is Ops.RANGE], [u for u in uops if u.op is Ops.END]
    self.assertEqual([e.src[1] for e in ends], rngs[::-1])

if __name__ == '__main__':
  unittest.main()
