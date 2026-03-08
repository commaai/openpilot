import numpy as np
import unittest
from dataclasses import replace

from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.codegen.gpudims import get_grouped_dims
from tinygrad.uop.ops import UOp, Ops, GroupOp, AxisType, PatternMatcher, graph_rewrite, UPat
from tinygrad.device import Device, Buffer, is_dtype_supported
from tinygrad.tensor import Tensor, _to_np_dtype
from tinygrad.engine.realize import run_schedule, CompiledRunner, get_program
from tinygrad.helpers import Context, flatten, dedup, TC_SELECT, TC_OPT, getenv
from tinygrad.dtype import DType, dtypes, PtrDType, AddrSpace
from tinygrad.renderer.ptx import PTXRenderer
from tinygrad.renderer.cstyle import CUDARenderer
MOCKGPU = getenv("MOCKGPU")

from tinygrad.uop.ops import print_uops # noqa: F401 # pylint: disable=unused-import

class TestLinearizer(unittest.TestCase):
  def test_arg_dedup(self):
    # NOTE: this realize exists because Tensor.numpy calls .contiguous() internally
    # without contiguous folding, rand.to("CPU") and rand.contiguous().to("CPU") are different UOps.
    # this test asserts they are the identical Buffer
    # having different buffers is fine for correctness, because the outputs match.
    a, b = Tensor.randn(4).realize(), Tensor.randn(4).realize()
    np_a, np_b = a.numpy(), b.numpy()
    c = ((a.shrink(((0, 2),)) - a.shrink(((2, 4),))) - (b.shrink(((0, 2),)) - b.shrink(((2, 4),))))
    sched = c.schedule()
    for si in sched: si.run()
    rawbufs = sched[-1].bufs
    assert len(rawbufs) == 3 and set(rawbufs[1:]) == {a.uop.base.realized, b.uop.base.realized}
    np_c = (np_a[:2] - np_a[2:]) - (np_b[:2] - np_b[2:])
    np.testing.assert_allclose(np_c, c.numpy(), atol=1e-4, rtol=1e-4)

  def test_load_removed(self):
    a = Tensor.rand(1).realize()
    b = Tensor.rand(1).realize()
    ta = Tensor.where(Tensor(True), a, b).numpy()
    tb = Tensor.where(Tensor(False), a, b).numpy()
    np.testing.assert_equal(a.numpy(), ta)
    np.testing.assert_equal(b.numpy(), tb)

  @unittest.skip("TODO: some backends insert more casts")
  def test_cast_there_and_back(self):
    tst = Tensor.ones(16, dtype=dtypes.int).contiguous().realize()
    out = tst.neg().cast(dtypes.char).cast(dtypes.int).cast(dtypes.char) * 2
    ast = helper_linearizer_opt(out)
    uops = get_program(ast, renderer=Device[Device.DEFAULT].renderer, opts=[]).uops
    self.assertEqual(len([x for x in uops if x.op is Ops.CAST]), 1)

  @unittest.expectedFailure
  def test_cast_back_and_there(self):
    tst = Tensor.ones(16, dtype=dtypes.int).contiguous().realize()
    out = tst.neg().cast(dtypes.char).cast(dtypes.int) * 2
    ast = helper_linearizer_opt(out)
    uops = get_program(ast, renderer=Device[Device.DEFAULT].renderer, opts=[]).uops
    self.assertEqual(len([x for x in uops if x.op is Ops.CAST]), 0)

  @unittest.skipIf(isinstance(Device[Device.DEFAULT].renderer, PTXRenderer), "broken on ptx")
  def test_late_bias_load(self):
    img = Tensor.empty(1, 3, 16, 16)
    w = Tensor.empty(16, 3, 3, 3)
    b = Tensor.empty(16)
    out = img.conv2d(w, b)
    ast = helper_linearizer_opt(out)
    uops = get_program(ast, renderer=Device[Device.DEFAULT].renderer, opts=[]).uops
    # slice at the last loop end
    uslice = [i for i,u in enumerate(uops) if u.op == Ops.END][-1]
    # only valid test if outermost range is the reduce
    if uops[uslice].src[-1].arg[-1] == AxisType.REDUCE:
      load_types = [u.src[0].dtype for u in uops[uslice+1:] if u.op == Ops.LOAD]
      # assert that there is a global load after the reduce ends
      assert any(dt.addrspace == AddrSpace.GLOBAL for dt in load_types)

  def _test_no_nested_ranges(self, lins, skip=None):
    for l in lins:
      range_in_acc = flatten([[x for x in u.src if x.op is Ops.RANGE] for u in l.uops if u.op is Ops.DEFINE_REG])
      ranges = [u.op for u in l.uops if (u.op is Ops.RANGE and u in range_in_acc) or (u.op is Ops.END and u.src[0] in range_in_acc)]
      for i,u in enumerate(ranges):
        if skip and i in skip: continue
        assert ranges[i-1] != u, f"multireduce nested the ranges! {ranges[i-1], {u}}"

  def test_two_nested_range(self):
    a = Tensor.randn(2, ).realize()
    out = a.reshape(2, 1).expand(2, 3).sum()
    ast = helper_linearizer_opt(out, wanna_output=[np.broadcast_to(a.numpy().reshape(2, 1), (2, 3)).sum()])
    uops = get_program(ast, renderer=Device[Device.DEFAULT].renderer, opts=[]).uops
    ranges = [i for i,u in enumerate(uops) if u.op is Ops.RANGE]
    assert len(ranges) == 1 # NOTE: it collapses now

  def test_three_nested_range(self):
    a = Tensor.randn(2, ).realize()
    out = a.reshape(2, 1).expand(2, 3).expand(2, 2, 3).sum()
    ast = helper_linearizer_opt(out, wanna_output=[np.broadcast_to(np.broadcast_to(a.numpy().reshape(2, 1), (2, 3)), (2, 2, 3)).sum()])
    uops = get_program(ast, renderer=Device[Device.DEFAULT].renderer, opts=[]).uops
    ranges = [i for i,u in enumerate(uops) if u.op is Ops.RANGE]
    assert len(ranges) == 1 # NOTE: it collapses now

  def test_two_nested_range_alt_indexing(self):
    a = Tensor([2, 2]).realize()
    out = a.reshape(2, 1).pad(((1, 1), (1, 1)), value=2).sum()
    ast = helper_linearizer_opt(out, wanna_output=[24])
    uops = get_program(ast, renderer=Device[Device.DEFAULT].renderer, opts=[]).uops
    ranges = [i for i,u in enumerate(uops) if u.op is Ops.RANGE]
    # RANGE -> ALU -> RANGE -> ALU + LOAD -> STORE
    assert any(x.op in GroupOp.ALU for x in uops[ranges[0]:ranges[1]])
    # the index of the load doesnt depend on the second range
    assert any(x.op is Ops.LOAD for x in uops[ranges[0]:ranges[1]])
    assert any(x.op in {*GroupOp.ALU, Ops.LOAD} for x in uops[ranges[1]:])

  def test_range_outer_op_before_phi(self):
    a = Tensor.randn(4, 1).realize()
    b = Tensor.randn(1, 1).realize()
    out = (a + b[0]).sum() + b[0]
    ast = helper_linearizer_opt(out, wanna_output=[(a.numpy()+b.numpy()[0]).sum()+b.numpy()])
    uops = get_program(ast, renderer=Device[Device.DEFAULT].renderer, opts=[]).uops
    ranges = [i for i,u in enumerate(uops) if u.op is Ops.RANGE]
    # LOAD -> RANGE -> LOAD -> STORE
    assert len([x for x in uops[:ranges[0]] if x.op is Ops.LOAD]) == 1

  def test_range_outer_op_before_phi_nested_range(self):
    a = Tensor.randn(2, ).realize()
    b = Tensor.randn(1, 1).realize()
    out = (a.reshape(2, 1).expand(2, 3) + b[0]).sum() + b[0]
    ast = helper_linearizer_opt(out, wanna_output=[(np.broadcast_to(a.numpy().reshape(2, 1), (2, 3)) + b.numpy()[0]).sum() + b.numpy()])
    uops = get_program(ast, renderer=Device[Device.DEFAULT].renderer, opts=[]).uops
    ranges = [i for i,u in enumerate(uops) if u.op is Ops.RANGE]
    assert len(ranges) == 1 # NOTE: it collapses now

  def test_load_dedup(self):
    # for different leaves in the AST, the same loads may occur.

    a = Tensor.randn(4).realize()
    # these are of size 3 to avoid float4 coalesce
    r = a[:-1] + a[1:]

    uops = get_program(r.schedule()[-1].ast, renderer=Device[Device.DEFAULT].renderer, opts=[Opt(op=OptOps.UPCAST, axis=0, arg=0)]).uops
    num_loads = len([uop for uop in uops if uop.op is Ops.LOAD])
    assert num_loads <= 4, "more load uops than needed"
    assert num_loads >= 4, "unexpected number of uops, maybe this test needs updating?"

  @unittest.skip("this is handled at higher level now")
  def test_upcast_cse(self):
    # when upcasting, within a subtree, there may be common expressions.

    a, b = Tensor.randn(1).realize(), Tensor.randn(1).realize()
    r = a.expand([2]) + b.expand([2])

    uops = get_program(r.schedule()[-1].ast, renderer=Device[Device.DEFAULT].renderer, opts=[Opt(op=OptOps.UPCAST, axis=0, arg=0)]).uops
    num_ops = len([uop for uop in uops if uop.op in GroupOp.ALU])
    assert num_ops <= 1, "more alu uops than needed"

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.supports_float4, "test requires float4")
  def test_reduce_upcast(self):
    x, w = Tensor.randn((1,1,3)).realize(), Tensor.randn((1,1,2)).realize()
    r = Tensor.conv2d(x,w,padding=1).relu()

    uops = get_program(r.schedule()[-1].ast, renderer=Device[Device.DEFAULT].renderer,
                       opts=[Opt(op=OptOps.UPCAST, axis=0, arg=0), Opt(op=OptOps.UNROLL, axis=0, arg=0)]).uops
    accs = [u for u in uops if u.op is Ops.DEFINE_REG]
    stores = [u for u in uops if u.op is Ops.STORE]
    assert len(accs) == 0  # it's removed now
    assert len(stores) == 1
    assert stores[0].src[1].dtype == dtypes.float.vec(4)

  # NOTE: can reenable, it does work. it just makes BEAM slow
  @unittest.expectedFailure
  @unittest.skipUnless(Device.DEFAULT == "CPU", "test only for CPU")
  def test_upcast_with_locals_cpu(self):
    out = Tensor.ones(64,64).contiguous() @ Tensor.ones(64,64).contiguous()
    prg = get_program(out.schedule()[-1].ast, opts=[Opt(OptOps.LOCAL, axis=0, arg=4)]).uops
    self.assertEqual(len(prg.src.split("for")), 5)

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_shared, "test requires shared")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.supports_float4, "test requires float4")
  @unittest.skipIf(isinstance(Device[Device.DEFAULT].renderer, PTXRenderer), "broken on ptx for some reason")
  def test_upcast_with_locals(self):
    x, y = Tensor.rand(1,128), Tensor.rand(128, 128)
    r = (x@y).relu()
    opts_to_apply = [Opt(op=OptOps.GROUP, axis=0, arg=8), Opt(op=OptOps.LOCAL, axis=0, arg=4), Opt(op=OptOps.UPCAST, axis=0, arg=4)]
    program = get_program(r.schedule()[-1].ast, renderer=Device[Device.DEFAULT].renderer, opts=opts_to_apply)

    stores = [u for u in program.uops if u.op is Ops.STORE and u.src[0].dtype.addrspace != AddrSpace.REG]

    # the first store is to lds and can be upcasted
    assert stores[0].src[1].dtype == dtypes.float.vec(4)
    assert any(x.op is Ops.DEFINE_LOCAL for x in stores[0].toposort())
    # the second store is to gds with no upcasts
    assert stores[1].src[1].dtype == dtypes.float
    assert any(x.op is Ops.PARAM for x in stores[1].toposort())

  def test_zero_fold(self):
    a, b = Tensor.randn(1).realize(), Tensor.randn(1).realize()
    r = Tensor.stack(a, b)
    uops = get_program(r.schedule()[-1].ast, renderer=Device[Device.DEFAULT].renderer, opts=[Opt(op=OptOps.UPCAST, axis=0, arg=0)]).uops
    num_ops = len([uop for uop in uops if uop.op in GroupOp.ALU])
    assert num_ops == 0, "more alu uops than needed"

  def test_sum_acc_dtype(self):
    for tensor_dtype, acc_dtype in (
      (dtypes.bool, dtypes.int), (dtypes.int16, dtypes.int), (dtypes.float16, dtypes.float), (dtypes.bfloat16, dtypes.float)):
      if is_dtype_supported(tensor_dtype) and is_dtype_supported(acc_dtype):
        a = Tensor([1, 2, 3], dtype=tensor_dtype).sum()
        realized_ast = a.schedule()[-1].ast
        program = get_program(realized_ast, renderer=Device[Device.DEFAULT].renderer, opts=[])
        local = [uop for uop in program.uops if uop.op is Ops.DEFINE_REG]
        assert local[0].dtype.base == acc_dtype

  def test_arg_acc_dtype(self):
    def helper_arg_acc_dtype(c: Tensor, expected_dtype:DType):
      realized_ast = c.schedule()[-1].ast
      program = get_program(realized_ast, renderer=Device[Device.DEFAULT].renderer, opts=[])
      local = [uop for uop in program.uops if uop.op is Ops.DEFINE_REG]
      self.assertEqual(local[0].dtype.base, expected_dtype)

    tests = (
      (dtypes.float16, None, dtypes.float),
      (dtypes.bfloat16, None, dtypes.float),
      (dtypes.float, None, dtypes.float),
      (dtypes.float16, dtypes.float16, dtypes.float16),
      (dtypes.bfloat16, dtypes.bfloat16, dtypes.bfloat16),
      (dtypes.float, dtypes.float16, dtypes.float16),
    )
    for tensor_dtype, acc_dtype, expected_dtype in tests:
      if is_dtype_supported(tensor_dtype) and is_dtype_supported(acc_dtype) and is_dtype_supported(expected_dtype):
        a, b = Tensor.rand(8, 8, dtype=tensor_dtype), Tensor.rand(8, 8, dtype=tensor_dtype)
        helper_arg_acc_dtype(a.sum(dtype=acc_dtype), expected_dtype)
        helper_arg_acc_dtype(a.matmul(b, dtype=acc_dtype), expected_dtype)
        helper_arg_acc_dtype(Tensor.einsum("ki,ij->kj", a, b, dtype=acc_dtype), expected_dtype)
        d, w = Tensor.rand(4, 8, 8, 8, dtype=tensor_dtype), Tensor.rand(8, 8, 2, 2, dtype=tensor_dtype)
        helper_arg_acc_dtype(d.conv2d(w, dtype=acc_dtype), expected_dtype)

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.supports_float4, "test requires float4")
  def test_simple_unroll_no_between_phi_dependencies(self):
    x, y = Tensor.empty(64, 64), Tensor.empty(64, 64)
    r = (x@y).relu()
    opt = [Opt(OptOps.UNROLL, 0, 4), Opt(OptOps.UPCAST, 0, 4)]
    ast = helper_linearizer_opt(r, [opt])
    # the uops graph is DEFINE_REG -> 4x STORE 0.0 -> RANGE -> 4x ALU -> 4x STORE -> ENDRANGE
    uops = get_program(ast, renderer=Device[Device.DEFAULT].renderer, opts=opt).uops
    begin_range = [i for i, x in enumerate(uops) if x.op is Ops.RANGE][-1]
    end_range = [i for i, x in enumerate(uops) if x.op is Ops.END][0]
    for i,u in enumerate(uops): print(i, u.op, [uops.index(s) for s in u.src], u.arg, u.dtype)
    for u in uops:
      if u.op is Ops.STORE and isinstance(dt:=u.src[0].dtype, PtrDType) and dt.addrspace is AddrSpace.REG:
        if uops.index(u) < begin_range:
          assert u.src[1].op is Ops.CONST
        else:
          assert u.src[1].op in GroupOp.ALU
          assert begin_range < uops.index(u) < end_range
      # children of END are placed after ENDRANGE
      if any(x.op is Ops.END and x.src[1].op in GroupOp.ALU for x in u.src):
        assert end_range < uops.index(u)

  def test_grouped_dims(self):
    def _assert_grouped_dims(prefix, dims, max_sizes, reverse_dims, expected_sizes, assert_same_length = True):
      idxs = get_grouped_dims(prefix, dims, max_sizes, reverse_dims)
      loop_idxs = dedup(flatten([[y for y in x.toposort() if y.op is Ops.SPECIAL] for x in idxs]))
      loop_idxs = sorted(loop_idxs, key=lambda uop: uop.arg)
      sizes = [x.src[0].arg for x in loop_idxs]
      assert len(idxs) == len(dims), f"expected idxs to have same length as dims {len(dims)}, got {len(idxs)}"
      if assert_same_length:
        assert len(loop_idxs) == min(len(sizes), len(dims)), f"expected idxs to have length {min(len(sizes), len(dims))}, got {len(loop_idxs)}"
      assert sizes == expected_sizes, f"expected sizes={expected_sizes}, got {sizes=}"
      # TODO: add these back after uop symbolic
      # for i in range(len(dims)):
      #   assert idxs[i].max+1 == dims[i], f"idxs[{i}] should have max {dims[i]-1}"
      # for i in range(len(loop_idxs)):
      #   assert loop_idxs[i].expr.startswith(prefix), f"loop_idxs[{i}] must start with {prefix}"
      #   assert loop_idxs[i].max+1 == sizes[i], f"loop_idxs[{i}] should have max {sizes[i]-1}"

    # no-op
    _assert_grouped_dims("gidx", (2,), (16,16,16), False, [2])
    _assert_grouped_dims("gidx", (2,3), (16,16,16), False, [2,3])

    # check reverse dims
    _assert_grouped_dims("gidx", (2,3), (16,16,16), True, [3,2])
    _assert_grouped_dims("gidx", (2,3,4), (16,16,16), False, [2,3,4])

    # test splitting globals:    len(dims) == len(max)
    _assert_grouped_dims("gidx", (64,3,4), (16,16,16), False, [16,12,4])
    _assert_grouped_dims("gidx", (64,3,4), (16,4,16), False, [16,3,16])
    _assert_grouped_dims("gidx", (64,3,4), (16,16,16), True, [16,3,16])
    _assert_grouped_dims("gidx", (128,3,4), (16,4,256), False, [16,3,32])
    _assert_grouped_dims("gidx", (4,4,512), (16,4,256), False, [8,4,256])

    # prefer group_dim strategy when possible
    _assert_grouped_dims("gidx", (512,4,2), (8192,2,2), False, [2048,2])

    # test splitting globals:    len(dims) < len(max)
    #                            len(dim)        ->          len(limited)
    #                              1             ->             2
    _assert_grouped_dims("gidx", (128,), (16,16,256), False, [16,8], False)
    #                              1             ->             3
    _assert_grouped_dims("gidx", (65536,), (16,16,256), False, [16,16,256], False)
    #                              2             ->             3
    _assert_grouped_dims("gidx", (128,128), (16,16,256), False, [16,16,64], False)
    #                              2             ->             2
    _assert_grouped_dims("gidx", (65536,2), (65535,65535,65535), False, [32768,4], False)
    # test when the only divisor is the square root of dim
    _assert_grouped_dims("gidx", (121,), (12,12,12), False, [11,11], False)

    # collapse on onto the left most axis
    _assert_grouped_dims("gidx", (2,3,4,5), (16,16,16), False, [6,4,5])
    _assert_grouped_dims("gidx", (2,3,4,5), (32,16,16), True, [20,3,2])
    # _assert_grouped_dims("gidx", (Variable("start_pos",1,2),3,4,5), (32,16,16), True, [20,3,Variable("start_pos",1,2)])

    # collapse on left-most available axis (the left most is too small)
    _assert_grouped_dims("gidx", (2,3,4,5), (4,16,16), False, [2,12,5])
    _assert_grouped_dims("gidx", (2,3,4,5), (16,16,16), True, [5,12,2])

    # _assert_grouped_dims("gidx", (Variable("start_pos",1,2),3,4,5), (16,16,16), False, [Variable("start_pos",1,2)*3,4,5])

    # dim too large and not factorable
    with self.assertRaises(RuntimeError):
      get_grouped_dims("gidx", (23,), (16,16,16), False,)
    with self.assertRaises(RuntimeError):
      get_grouped_dims("gidx", (128,3,4), (16,2,2), False,)

    # too large for sizes
    with self.assertRaises(RuntimeError):
      get_grouped_dims("gidx", (2,3,4,5,6), (16,16,16))

    # TODO: In the above cases we only test if the shape after reshape is correct, never the indices.
    # We should check if the returned indices are correct, for all cases.
    # (65536, 2) -> (32768, 4)
    dims, expected_limited_dims = (65536,2), (32768, 4)
    idxs = get_grouped_dims("gidx", dims, (65535,65535,65535))
    def match_div(): raise RuntimeError("match_div")
    def match_mod(): raise RuntimeError("match_mod")
    flat_idx_pattern = UPat(Ops.SPECIAL, arg='gidx0')*expected_limited_dims[1]+UPat(Ops.SPECIAL, arg='gidx1')
    pm = PatternMatcher([
      (flat_idx_pattern//dims[1], match_div),
      (flat_idx_pattern%dims[1], match_mod)
    ])

    with self.assertRaises(RuntimeError) as error:
      graph_rewrite(idxs[0], pm)
    self.assertIn("match_div", str(error.exception))

    with self.assertRaises(RuntimeError) as error:
      graph_rewrite(idxs[1], pm)
    self.assertIn("match_mod", str(error.exception))

    # # variable too large
    # with self.assertRaises(AssertionError):
    #   get_grouped_dims("gidx", (Variable("start_pos",0,16),3,4), (16,16,16), False,)

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  def test_default_global_reversed(self):
    # shrink so that the dims do not collapse
    t = Tensor.ones(5, 6, 7).contiguous().realize().shrink(((0, 4), (0, 5), (0, 6)))
    ast = helper_linearizer_opt(t+1)
    uops = get_program(ast, renderer=Device[Device.DEFAULT].renderer, opts=[]).uops
    idxs = dedup([uop for uop in uops if uop.op is Ops.SPECIAL])
    idxs = sorted(idxs, key=lambda uop: uop.arg)
    assert (idxs[0].arg, idxs[0].src[0].arg) == ('gidx0', 6), idxs[0]
    assert (idxs[1].arg, idxs[1].src[0].arg) == ('gidx1', 5), idxs[1].arg
    assert (idxs[2].arg, idxs[2].src[0].arg) == ('gidx2', 4), idxs[2].arg

  def test_sum_collapse(self):
    t = Tensor([2]).reshape(1, 1).expand(256, 256).sum()
    sched = [si for si in t.schedule() if si.ast.op is Ops.SINK]
    # sum_collapse is a full collapse now
    assert len(sched) == 1
    assert not any(u.op is Ops.REDUCE_AXIS for u in sched[0].ast.toposort()), "found reduce in sum collapse"
    #lin = Kernel(sched[0].ast)
    #assert not any(u.op is Ops.RANGE for u in lin.linearize().uops), "found loop in sum collapse"

  def test_assign_fold(self):
    a = Tensor.ones(4, 4).contiguous().realize()
    m = Tensor.ones(4, 4).shrink(((1, 2), None)).pad(((1, 2), None))
    a.assign(a+m)
    a.realize()
    np.testing.assert_equal(a.flatten().numpy(), [1.,1.,1.,1.,2.,2.,2.,2.,1.,1.,1.,1.,1.,1.,1.,1.])

  @unittest.skipIf(MOCKGPU and isinstance(Device[Device.DEFAULT].renderer, (PTXRenderer, CUDARenderer)), "PTX indexes differently. might be ok?")
  def test_where_fold(self):
    a = Tensor.ones(4, 4).contiguous().realize()
    b = a.shrink(((1, 2), None)).pad(((1, 2), None))
    a.assign(b.where(2, a))
    sched = a.schedule()
    assert len(sched) == 1
    sched_copy = sched[:]
    run_schedule(sched)
    np.testing.assert_equal(a.flatten().numpy(), [1.,1.,1.,1.,2.,2.,2.,2.,1.,1.,1.,1.,1.,1.,1.,1.])
    program = get_program(sched_copy[-1].ast, renderer=Device[Device.DEFAULT].renderer, opts=())
    assert not any(u.op == Ops.WHERE for u in program.uops), "found where where where should be folded"

  def test_phi_simplification(self):
    def helper(t, max_ops=0):
      ast = helper_linearizer_opt(t)
      uops = get_program(ast, renderer=Device[Device.DEFAULT].renderer).uops
      # ignore kernel optimized IF statements for now
      if if_op:=next((u for u in uops if u.op is Ops.IF), None):
        uops = uops[:uops.index(if_op)]
      assert len(set([u.op for u in uops if u.op in {Ops.RANGE, Ops.SPECIAL}])) == 1, "has either specials or ranges, not both"
      reg_stores = [u for u in uops if u.op is Ops.STORE and isinstance(dt:=u.src[0].dtype, PtrDType) and dt.addrspace == AddrSpace.REG]
      assert len(reg_stores) == 0, "STORE to reg should have been simplified"
      assert len([u for u in uops if u.op is Ops.MAX]) <= max_ops, "no unnecessary MAX ops"

    helper(Tensor.arange(5.5, (3.5*300), 3.5), max_ops=2)
    helper(Tensor.arange(-1, -100, -5), max_ops=2)
    # NOTE: both of these split the reduce (this just wasn't tracked before)
    #helper(Tensor.arange(-3.2, 6.7, 0.64), max_ops=2)
    #helper(Tensor.arange(256), max_ops=2)
    helper(Tensor.arange(255), max_ops=2)

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.supports_float4, "test requires float4")
  @unittest.skipIf(isinstance(Device[Device.DEFAULT].renderer, PTXRenderer), "broken on ptx for some reason")
  def test_grouped_store_phis(self):
    """
    float4 acc0 = float4(0.0,0.0,0.0,0.0);
    {
      acc0 = // ...
    }
    *((device float4*)(data0+alu2)) = float4(acc0.x,acc0.y,acc0.z,acc0.w);
    simplifies to:
    *((device float4*)(data0+alu2)) = acc0;
    """
    x, y = Tensor.empty(64,64), Tensor.empty(64,64)
    out = x.matmul(y)
    with Context(TC=0):
      ast = helper_linearizer_opt(out)
      uops = get_program(ast, renderer=Device[Device.DEFAULT].renderer).uops
    # check that the float4 cast collapses
    store_vals = [u.src[1] for u in uops if u.op is Ops.STORE and u.src[0].dtype.addrspace != AddrSpace.REG]
    for val in store_vals:
      assert val.dtype == dtypes.float.vec(4) # and val.op is not Ops.VECTORIZE

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.supports_float4, "test requires float4")
  def test_grouped_store_values(self):
    x = Tensor.randn((4,3,6,6)).realize()
    out = x.flip((0,1)).contiguous()
    ast = helper_linearizer_opt(out)
    store_val = [u.src[1] for u in get_program(ast, renderer=Device[Device.DEFAULT].renderer).uops if u.op is Ops.STORE][0]
    assert store_val.dtype == dtypes.float.vec(4) and store_val.op is not Ops.VECTORIZE

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_shared, "test requires shared")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.supports_float4, "test requires float4")
  def test_grouped_store_locals_and_globals(self):
    x, y = Tensor.empty(64, 64), Tensor.empty(64, 64)
    out = x@y
    opt = [Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.GROUPTOP, 0, 8),
            Opt(OptOps.UNROLL, 0, 4), Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.UPCAST, 1, 2)] # upcast accs in both reduces
    ast = helper_linearizer_opt(out, opts=[opt])
    def get_recursive(uop): return set.union(set(uop.src), [uop], *[get_recursive(v) for v in uop.src])
    uops = get_program(ast, renderer=Device[Device.DEFAULT].renderer, opts=opt).uops
    local_stores = [u for u in uops if u.op is Ops.STORE and any(x.op is Ops.DEFINE_LOCAL for x in get_recursive(u.src[0]))]
    global_stores = [u for u in uops if u.op is Ops.STORE and any(x.op is Ops.PARAM for x in get_recursive(u.src[0]))]
    barrier = [u for u in uops if u.op is Ops.BARRIER]
    assert len(barrier) == 1
    # check that the float4 cast collapses for all stores
    for store in local_stores+global_stores:
      assert store.src[1].dtype.count > 1 # and store.src[2].op is not Ops.VECTORIZE
    # # check the children's vins
    # TODO: src ALU are not the same, should it?
    # assert barrier.src == tuple(local_stores)
    assert len([u for u in uops if u.op is Ops.IF])

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_shared, "test requires shared")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.supports_float4, "test requires float4")
  @unittest.skipIf(isinstance(Device[Device.DEFAULT].renderer, PTXRenderer), "broken on ptx for some reason")
  def test_grouped_store_local_only(self):
    x, y = Tensor.rand(1,128), Tensor.rand(128, 128)
    r = (x@y).relu()
    ast = helper_linearizer_opt(r)
    uops = get_program(ast, renderer=Device[Device.DEFAULT].renderer).uops
    stores = [u for u in uops if u.op is Ops.STORE and u.src[0].dtype.addrspace != AddrSpace.REG]

    # the float4 value stores directly in lds and we skip upcast
    self.assertEqual(stores[0].src[1].dtype, dtypes.float.vec(4))
    #assert stores[0].src[-1].op is not Ops.VECTORIZE

    # the global store doesn't change
    assert stores[1].src[1].dtype == dtypes.float

# *** helpers ***

def helper_realized_ast(r:Tensor|list[Tensor]) -> tuple[UOp, list[Buffer]]:
  if isinstance(r, Tensor): r = [r]
  s = Tensor.schedule(*r)
  run_schedule(s[:-1])  # run all kernels except the last one
  assert s[-1].ast.op is Ops.SINK, f"helper_realized_ast expects a SINK {s[-1]}"
  # now all input buffers in s[-1] should be realized
  # create fresh buffers for the outputs
  bufs = [Buffer(x.device, x.size, x.dtype).allocate() if i < len(s[-1].ast.src) else x for i,x in enumerate(s[-1].bufs)]
  # ensure buffers are allocated
  for b in bufs: b.ensure_allocated()
  return s[-1].ast, bufs

def helper_linearizer_ast(ast:UOp, inputs:list[Tensor], *args, **kwargs):
  assert isinstance(ast, UOp), "ast must be UOp"
  inbufs = [x.uop.base.buffer for x in inputs]
  outbufs = [Buffer(inbufs[-1].device if inbufs else Device.DEFAULT, out.size, out.src[1].dtype).allocate() for out in ast.src]
  _helper_linearizer_opt_ast(ast, outbufs+inbufs, *args, **kwargs)

def helper_linearizer_opt(r:Tensor|list[Tensor], *args, **kwargs):
  realized_ast, real_bufs = helper_realized_ast(r)
  _helper_linearizer_opt_ast(realized_ast, real_bufs, *args, **kwargs)
  return realized_ast

def copyout_outputs(outbufs:list[Buffer]) -> list[np.ndarray]:
  return [np.frombuffer(x.as_buffer(), _to_np_dtype(x.dtype)) for x in outbufs]

def reset_bufs(bufs:list[Buffer]):
  for buf in bufs: buf.copyin(np.zeros((buf.size*buf.dtype.itemsize,), dtype=np.uint8).data)

def _helper_linearizer_opt_ast(realized_ast:UOp, real_bufs:list[Buffer], opts=[],
                               apply_tc=False, atol=1e-4, rtol=1e-4, color_sizes=[], wanna_output=[]):
  outbufs = real_bufs[:len(realized_ast.src)]
  device = real_bufs[0].device
  wanna_output = [np.array(x).flatten() for x in wanna_output]

  def get_prg(opts): return CompiledRunner(replace(get_program(realized_ast, renderer=Device[Device.DEFAULT].renderer, opts=opts), device=device))

  def check_opt(opts):
    prg = get_prg(opts=opts)
    reset_bufs(outbufs)
    prg.exec(real_bufs)
    for x,want in zip(copyout_outputs(outbufs), wanna_output): np.testing.assert_allclose(x, want, atol=atol, rtol=rtol)

  # Get baseline if it is not provided, which is not optimized at all.
  prg = get_prg(opts=())
  prg.exec(real_bufs)
  if len(wanna_output) == 0: wanna_output = copyout_outputs(outbufs)
  else:
    for buf,want in zip(copyout_outputs(outbufs), wanna_output): np.testing.assert_allclose(buf, want, atol=atol, rtol=rtol)

  # Check correctness of handcoded optimiztions.
  prg = get_prg(opts=None)
  reset_bufs(outbufs)
  prg.exec(real_bufs)
  for buf,want in zip(copyout_outputs(outbufs), wanna_output): np.testing.assert_allclose(buf, want, atol=atol, rtol=rtol)
  for x in opts: # Check custom transformations if any.
    check_opt(([Opt(OptOps.TC, 0, (TC_SELECT.value, TC_OPT.value, 1))] if apply_tc else [])+x)

if __name__ == '__main__':
  unittest.main()
