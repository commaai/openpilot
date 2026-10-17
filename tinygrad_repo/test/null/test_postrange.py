import unittest
from tinygrad import Tensor, UOp, dtypes
from tinygrad.codegen.opt import Opt, OptOps, KernelOptError
from tinygrad.codegen.opt.postrange import Scheduler
from tinygrad.renderer.llvmir import AMDLLVMRenderer
from tinygrad.helpers import Target
from tinygrad.uop.ops import AxisType, KernelInfo, Ops

class TestReductionAxes(unittest.TestCase):
  def scheduler(self, ast): return Scheduler(ast, AMDLLVMRenderer(Target("AMD", arch="gfx1100")))

  def reduction(self, size=64):
    # Reduction role, not numeric id, determines optimizer axis order.
    i, r = UOp.range(size, 1), UOp.range(size, 0)
    inp, out = UOp.param(0, dtypes.float, (size, size)), UOp.param(1, dtypes.float, (size,))
    return self.scheduler(out[i].store(inp[i, r].reduce(r, arg=Ops.ADD)).end(i).sink(arg=KernelInfo())), i, r

  def test_weak_reduction_membership(self):
    k, i, r = self.reduction()
    self.assertIs(i.axis_type, AxisType.WEAK)
    self.assertIs(r.axis_type, AxisType.WEAK)
    self.assertEqual(k.rngs, [i, r])
    self.assertEqual(k.reduce_ranges, {r})
    self.assertEqual(k.upcastable_dims(), [0])
    self.assertEqual(k.upcastable_dims(reduce=True), [1])
    self.assertEqual(k.upcastable_dims(reduce=None), [0, 1])
    k.convert_loop_to_global()
    self.assertEqual(k.axis_types, [AxisType.GLOBAL, AxisType.WEAK])
    self.assertEqual(k.reduce_ranges, {r})

  def test_reduction_bound_is_not_reduced(self):
    i = UOp.range(4, 0)
    r = UOp.range(i+1, 1)
    inp, out = UOp.param(0, dtypes.float, (4, 4)), UOp.param(1, dtypes.float, (4,))
    k = self.scheduler(out[i].store(inp[i, r].reduce(r, arg=Ops.ADD)).end(i).sink(arg=KernelInfo()))
    self.assertEqual(k.reduce_ranges, {r})
    self.assertEqual(k.axes_of(reduce=True), [1])
    k.apply_opt(Opt(OptOps.PADTO, 0, 8))
    self.assertIs(k.reduceop.src[0].op, Ops.INDEX)  # padding a bound dependency must not mask the reduction's input

  def test_upcast_both_roles(self):
    k, _, _ = self.reduction()
    k.apply_opt(Opt(OptOps.SPLIT, 1, (4, AxisType.UPCAST)))
    k.apply_opt(Opt(OptOps.SPLIT, 0, (2, AxisType.UPCAST)))
    self.assertEqual(k.axis_types, [AxisType.WEAK, AxisType.UPCAST, AxisType.WEAK, AxisType.UPCAST])
    self.assertEqual(k.axes_of(AxisType.UPCAST, reduce=False), [1])
    self.assertEqual(k.axes_of(AxisType.UPCAST, reduce=True), [3])
    self.assertEqual(k.colors()[1::2], ["yellow", "yellow"])

  def test_upcast_limits_follow_reduction_role(self):
    for axis, amount, allowed in [(0, 16, True), (0, 32, False), (1, 32, True), (1, 64, False)]:
      with self.subTest(axis=axis, amount=amount):
        k, _, _ = self.reduction()
        opt = Opt(OptOps.SPLIT, axis, (amount, AxisType.UPCAST))
        if allowed: k.apply_opt(opt)
        else:
          with self.assertRaises(KernelOptError): k.apply_opt(opt)

  def test_group_shared_memory_excludes_reduction_upcasts(self):
    k, _, _ = self.reduction()
    k.apply_opt(Opt(OptOps.SPLIT, 0, (2, AxisType.UPCAST)))
    k.apply_opt(Opt(OptOps.SPLIT, k.axes_of(AxisType.WEAK, reduce=True)[0], (32, AxisType.UPCAST)))
    # Two output lanes times two local lanes times sizeof(float), not times the 32 reduction lanes.
    k.ren.shared_max = 16
    k.apply_opt(Opt(OptOps.SPLIT, k.axes_of(AxisType.WEAK, reduce=True)[0], (2, AxisType.LOCAL)))
    self.assertEqual(k.group_for_reduces, 1)

  def test_nested_reduction_group_refused(self):
    i, r = UOp.range(8, 0), UOp.range(8, 1)
    inp, out = UOp.param(0, dtypes.float, (8, 8)), UOp.param(1, dtypes.float, (1,))
    ast = out[0].store(inp[i, r].reduce(r, arg=Ops.ADD).reduce(i, arg=Ops.ADD)).sink(arg=KernelInfo())
    for upcast_outer in (False, True):
      with self.subTest(upcast_outer=upcast_outer):
        k = self.scheduler(ast)
        if upcast_outer: k.apply_opt(Opt(OptOps.SPLIT, k.rngs.index(i), (0, AxisType.UPCAST)))
        with self.assertRaisesRegex(KernelOptError, "inside another reduce"):
          k.apply_opt(Opt(OptOps.SPLIT, k.rngs.index(r), (2, AxisType.LOCAL)))

  def test_execute_upcast_reduction(self):
    def kernel(out, inp):
      i, r = UOp.range(4, 0), UOp.range(8, 1)
      return out[i].store(inp[i, r].reduce(r, arg=Ops.ADD)).end(i).sink(arg=KernelInfo(opts_to_apply=(
        Opt(OptOps.SPLIT, 1, (4, AxisType.UPCAST)), Opt(OptOps.SPLIT, 0, (2, AxisType.UPCAST)))))
    inp = Tensor([float(i) for i in range(32)], device="PYTHON").reshape(4, 8).contiguous().realize()
    out = Tensor.empty(4, device="PYTHON").custom_kernel(inp, fxn=kernel)[0]
    self.assertEqual(out.tolist(), [float(sum(range(i*8, (i+1)*8))) for i in range(4)])

if __name__ == '__main__':
  unittest.main()
