import unittest, math
from unittest.mock import patch
import z3
from tinygrad import Tensor, Device
from tinygrad.codegen.gpudims import get_grouped_dims, group_gpudims, pm_range_to_special
from tinygrad.uop.ops import UOp, Ops, KernelInfo, AxisType, graph_rewrite
from tinygrad.uop.validate import uops_to_z3
from tinygrad.dtype import dtypes
from tinygrad.renderer import Renderer
from tinygrad.helpers import flatten, dedup, Target, Context

class TestGroupedDims(unittest.TestCase):
  def _check_grouped_dims(self, prefix, dims, max_sizes, reverse, expected_sizes, assert_same_length=True):
    grouped = UOp.sink(*get_grouped_dims(prefix, dims, max_sizes, reverse))
    idxs = graph_rewrite(grouped, pm_range_to_special).src
    loop_idxs = dedup(flatten([[y for y in x.toposort() if y.op is Ops.SPECIAL] for x in idxs]))
    loop_idxs = sorted(loop_idxs, key=lambda uop: uop.arg)
    sizes = [x.src[0].val for x in loop_idxs]
    assert len(idxs) == len(dims), f"expected idxs to have same length as dims {len(dims)}, got {len(idxs)}"
    if assert_same_length:
      assert len(loop_idxs) == min(len(sizes), len(dims)), f"expected idxs to have length {min(len(sizes), len(dims))}, got {len(loop_idxs)}"
    assert sizes == expected_sizes, f"expected sizes={expected_sizes}, got {sizes=}"
    self._verify_indices_z3(idxs, dims)

  def _verify_indices_z3(self, idxs, dims):
    """Use z3 to prove bijectivity: bounds (0 <= flat < total) + injectivity (different inputs => different flat)."""
    total = math.prod(dims)
    specials = sorted(dedup(flatten([[y for y in x.toposort() if y.op is Ops.SPECIAL] for x in idxs])), key=lambda u: u.arg)
    # build flat index and primed flat (same expression with renamed SPECIALs)
    flat = UOp.const(0)
    for i, idx in enumerate(idxs):
      flat = flat + idx * int(math.prod(dims[i+1:]))
    flat_p = flat.substitute({s: UOp(Ops.SPECIAL, src=s.src, arg=s.arg+"_p") for s in specials})
    solver = z3.Solver()
    [z3_flat, z3_flat_p] = uops_to_z3(solver, flat, flat_p)
    # bounds
    self.assertEqual(solver.check(z3_flat < 0), z3.unsat, f"flat can be negative: {dims=}")
    self.assertEqual(solver.check(z3_flat >= total), z3.unsat, f"flat can be >= {total}: {dims=}")
    # injectivity: flat == flat' but inputs differ => unsat
    inputs_differ = z3.Or(*[z3.Int(s.arg) != z3.Int(s.arg+"_p") for s in specials])
    self.assertEqual(solver.check(z3.And(z3_flat == z3_flat_p, inputs_differ)), z3.unsat, f"not injective: {dims=}")

  def test_grouped_dims(self):
    # no-op
    self._check_grouped_dims("gidx", (2,), (16,16,16), False, [2])
    self._check_grouped_dims("gidx", (2,3), (16,16,16), False, [2,3])

    # check reverse dims
    self._check_grouped_dims("gidx", (2,3), (16,16,16), True, [3,2])
    self._check_grouped_dims("gidx", (2,3,4), (16,16,16), False, [2,3,4])

    # test splitting globals:    len(dims) == len(max)
    self._check_grouped_dims("gidx", (64,3,4), (16,16,16), False, [16,12,4])
    self._check_grouped_dims("gidx", (64,3,4), (16,4,16), False, [16,3,16])
    self._check_grouped_dims("gidx", (64,3,4), (16,16,16), True, [16,3,16])
    self._check_grouped_dims("gidx", (128,3,4), (16,4,256), False, [16,3,32])
    self._check_grouped_dims("gidx", (4,4,512), (16,4,256), False, [8,4,256])
    self._check_grouped_dims("gidx", (5,12,7), (8,4,16), False, [10,3,14])

    # prefer group_dim strategy when possible
    self._check_grouped_dims("gidx", (512,4,2), (8192,2,2), False, [2048,2])

    # test splitting globals:    len(dims) < len(max)
    #                            len(dim)        ->          len(limited)
    #                              1             ->             2
    self._check_grouped_dims("gidx", (128,), (16,16,256), False, [16,8], False)
    #                              1             ->             3
    self._check_grouped_dims("gidx", (65536,), (16,16,256), False, [16,16,256], False)
    #                              2             ->             2
    self._check_grouped_dims("gidx", (65536,2), (65535,65535,65535), False, [32768,4], False)
    # test when the only divisor is the square root of dim
    self._check_grouped_dims("gidx", (121,), (12,12,12), False, [11,11], False)
    #                              2             ->             3
    self._check_grouped_dims("gidx", (128,128), (16,16,256), False, [16,16,64], False)

    # collapse on onto the left most axis
    self._check_grouped_dims("gidx", (2,3,4,5), (16,16,16), False, [6,4,5])
    self._check_grouped_dims("gidx", (2,3,4,5), (32,16,16), True, [20,3,2])

    # collapse on left-most available axis (the left most is too small)
    self._check_grouped_dims("gidx", (2,3,4,5), (4,16,16), False, [2,12,5])
    self._check_grouped_dims("gidx", (2,3,4,5), (16,16,16), True, [5,12,2])

    # dim too large and not factorable
    with self.assertRaises(RuntimeError):
      get_grouped_dims("gidx", (23,), (16,16,16), False,)
    with self.assertRaises(RuntimeError):
      get_grouped_dims("gidx", (128,3,4), (16,2,2), False,)

    # too large for sizes
    with self.assertRaises(RuntimeError):
      get_grouped_dims("gidx", (2,3,4,5,6), (16,16,16))

  def test_grouped_direct_dims_are_ranges(self):
    # when (2,3) are merged into 6, the unmerged dims (4,5) should map directly to hardware ranges (no div/mod)
    idxs = get_grouped_dims("gidx", (2,3,4,5), (16,16,16), False)
    assert idxs[2].op is Ops.RANGE, f"expected RANGE for direct-mapped dim, got {idxs[2].op}"
    assert idxs[3].op is Ops.RANGE, f"expected RANGE for direct-mapped dim, got {idxs[3].op}"

  def test_grouped_dims_high_rank(self):
    # 4D collapsed onto 2 axes
    self._check_grouped_dims("gidx", (4,4,4,4), (16,16), False, [16,16])
    # 4D untouched
    self._check_grouped_dims("gidx", (2,3,4,5), None, False, [2,3,4,5])
    idxs = get_grouped_dims("gidx", (2,3,4,5), None, False)
    assert all(u.op is Ops.RANGE for u in idxs), f"expected all-RANGE when untouched, got {[u.op for u in idxs]}"
    # 5D and 6D collapsed onto 3 axes
    self._check_grouped_dims("gidx", (2,2,2,2,2), (4,4,4), False, [4,4,2])
    self._check_grouped_dims("gidx", (2,2,2,2,2,2), (8,8,8), False, [8,4,2])

  def test_symbolic_dims_cross_launch_limit(self):
    dim = UOp.variable("dim", 1, 4)
    assert len(get_grouped_dims("gidx", (1, dim), (4, 3))) == 2

  def test_global_prod_max(self):
    g, l = UOp.range(256, 0, AxisType.GLOBAL), UOp.range(256, 1, AxisType.LOCAL)
    sink = UOp.param(0, dtypes.float, 512).index(g + l).store(UOp.const(1.0)).end(g, l).sink(arg=KernelInfo())
    class R(Renderer): global_max, local_max, global_prod_max = (256, 256, 256), (128, 128, 128), (128, 128, 128)
    grouped = group_gpudims(R(Target()), sink)
    specials = [u for u in graph_rewrite(grouped, pm_range_to_special).toposort() if u.op is Ops.SPECIAL]
    self.assertGreater(len([s for s in specials if "lidx" in s.arg]), 1)
    self.assertGreater(len([s for s in specials if "gidx" in s.arg]), 1)

  def test_execute_grouped_dims(self):
    ren = Device["PYTHON"].renderer
    # Direct mapping, merging four logical dimensions, and splitting oversized dimensions.
    for global_shape, local_shape in [((3,), (2,)), ((3,3,3), (2,)), ((2,3,2,2), (2,3,2,2)), ((32,), (16,))]:
      with self.subTest(global_shape=global_shape, local_shape=local_shape):
        dims = global_shape+local_shape+(2,)
        def kernel(out):
          # Reuse ids across axis types; global ids may also be permuted onto identical hardware ranges.
          ranges = [UOp.range(s, i, t) for t,shape in [(AxisType.GLOBAL, global_shape), (AxisType.LOCAL, local_shape),
                                                     (AxisType.LOOP, (2,))] for i,s in enumerate(shape)]
          idx = sum(r*math.prod(dims[i+1:]) for i,r in enumerate(ranges))
          return out[idx].store(idx*3+7).end(*ranges).sink(arg=KernelInfo(opts_to_apply=()))
        out = Tensor.full((math.prod(dims),), -1, device="PYTHON").contiguous().realize()
        with patch.object(ren, "global_max", (8,4,4)), patch.object(ren, "local_max", (8,4,4)), Context(CAPTURE_PROCESS_REPLAY=0):
          self.assertEqual(out.custom_kernel(fxn=kernel)[0].tolist(), [i*3+7 for i in range(math.prod(dims))])

  def test_grouped_mask_ranges(self):
    g, l = UOp.range(3, 0, AxisType.GLOBAL), UOp.range(2, 5, AxisType.LOCAL)
    sink = UOp.param(0, dtypes.int, 3)[g].store((g+l).cast(dtypes.int)).end(g, l).sink(arg=KernelInfo())
    grouped = group_gpudims(Device["PYTHON"].renderer, sink)
    self.assertEqual({r.arg for r in grouped.toposort() if r.op is Ops.RANGE}, {(AxisType.GLOBAL, 0), (AxisType.LOCAL, 0)})

  def test_upcast_same_id_as_hardware(self):
    def kernel(out):
      g, u = UOp.range(3, 0, AxisType.GLOBAL), UOp.range(2, 0, AxisType.UPCAST)
      idx = g*2+u
      return out[idx].store(idx*3+7).end(g, u).sink(arg=KernelInfo(opts_to_apply=()))
    out = Tensor.full((6,), -1, device="PYTHON").contiguous().realize()
    self.assertEqual(out.custom_kernel(fxn=kernel)[0].tolist(), [i*3+7 for i in range(6)])

  def test_max_sizes_none(self):
    self._check_grouped_dims("gidx", (2,3,4), None, False, [2,3,4])
    self._check_grouped_dims("gidx", (100,), None, False, [100])

if __name__ == '__main__':
  unittest.main()
