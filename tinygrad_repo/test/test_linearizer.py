import numpy as np
import unittest
from dataclasses import replace

from tinygrad.codegen.opt.kernel import Opt, OptOps, KernelOptError, Kernel, AxisType
from tinygrad.codegen.gpudims import get_grouped_dims
from tinygrad.uop.ops import UOp, Ops, GroupOp, KernelInfo
from tinygrad.device import Device, Buffer, is_dtype_supported
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View
from tinygrad.tensor import Tensor, _to_np_dtype
from tinygrad.engine.realize import run_schedule, lower_schedule, CompiledRunner, get_program
from tinygrad.codegen.opt.heuristic import hand_coded_optimizations
from tinygrad.helpers import prod, Context, getenv, CI, flatten, dedup, AMX, AMD_LLVM
from tinygrad.dtype import DType, dtypes, AddrSpace
from tinygrad.codegen import apply_rewrites, rewrites_for_views

def push_views(ast): return apply_rewrites(ast, rewrites_for_views)

def helper_realized_ast(r:Tensor|list[Tensor]) -> tuple[UOp, list[Buffer]]:
  if isinstance(r, Tensor): r = [r]
  s = Tensor.schedule(*r)
  run_schedule(s[:-1])  # run all kernels except the last one
  assert s[-1].ast.op is Ops.SINK, f"helper_realized_ast expects a SINK {s[-1]}"
  # now all input buffers in s[-1] should be realized
  # create fresh buffers for the outputs
  bufs = [Buffer((x).device, x.size, x.dtype).allocate() if i < len(s[-1].ast.src) else x for i,x in enumerate(s[-1].bufs)]
  return push_views(s[-1].ast), bufs

def helper_tc_allclose(N:int, M:int, K:int, dtype_in:DType, dtype_out:DType, axis:int=0, tc_select:int=-1, tc_opt:int=0, use_tensor_cores:int=1):
  a, b = Tensor.rand(M, K, dtype=dtype_in), Tensor.rand(K, N, dtype=dtype_in)
  np_a, np_b = a.numpy(), b.numpy()
  r = a.matmul(b, dtype=dtype_out)
  if dtype_in == dtypes.bfloat16: r = r.float()
  realized_ast, bufs = helper_realized_ast(r)
  k = Kernel(realized_ast)
  k.apply_tensor_cores(use_tensor_cores, axis=axis, tc_select=tc_select, tc_opt=tc_opt)
  prg = CompiledRunner(replace(get_program(k.get_optimized_ast(), k.opts), device=Device.DEFAULT))
  if use_tensor_cores == 1: assert len([uop for uop in prg.p.uops if uop.op is Ops.WMMA]) > 0, "wmma not triggered"
  assert len([x for x in k.applied_opts if x.op is OptOps.TC]) == 1, "tensor core opt not included"
  prg.exec(bufs)
  if dtype_in == dtypes.half: tc_atol, tc_rtol = 1e-2, 1e-3
  elif dtype_in == dtypes.bfloat16: tc_atol, tc_rtol = 1e-2, 1e-2
  else: tc_atol, tc_rtol = 5e-3, 1e-4
  c = bufs[0].numpy().reshape((M,N))
  np.testing.assert_allclose(c, np_a @ np_b, atol=tc_atol, rtol=tc_rtol)

def helper_tc_ensure_uops_and_opts_count(N: int, M:int, K:int, dtype_in:DType, dtype_out:DType, axis:int=0, tc_select:int=-1, tc_opt:int=0,
                                         ensure_triggered:bool=True):
  a, b = Tensor.rand(M, K, dtype=dtype_in), Tensor.rand(K, N, dtype=dtype_in)
  r = a.matmul(b, dtype=dtype_out)
  sched = r.schedule()
  realized_ast = sched[-1].ast
  opts_to_apply = [Opt(OptOps.TC, axis, (tc_select, tc_opt, 1))]
  realized_ast = realized_ast.replace(arg=KernelInfo(opts_to_apply=tuple(opts_to_apply)))

  if ensure_triggered:
    program = get_program(realized_ast, Device[Device.DEFAULT].renderer)
    wmmas = len([uop for uop in program.uops if uop.op is Ops.WMMA])
    tcs = len([x for x in program.applied_opts if x.op is OptOps.TC])
    assert wmmas > 0, "tensor core not triggered"
    assert tcs == 1, "tensor core opt not included"
  else:
    try:
      program = get_program(realized_ast, Device[Device.DEFAULT].renderer)
      assert False, "OptOps.TC triggered, expected KernelOptError"
    except KernelOptError: pass

class TestLinearizer(unittest.TestCase):
  def test_arg_dedup(self):
    # NOTE: this realize exists because Tensor.numpy calls .contiguous() internally
    # without contiguous folding, rand.to("CPU") and rand.contiguous().to("CPU") are different UOps.
    # this test asserts they are the identical Buffer
    # having different buffers is fine for correctness, because the outputs match.
    a, b = Tensor.randn(4).realize(), Tensor.randn(4).realize()
    np_a, np_b = a.numpy(), b.numpy()
    c = ((a.shrink(((0, 2),)) - a.shrink(((2, 4),))) - (b.shrink(((0, 2),)) - b.shrink(((2, 4),))))
    lowered = [x[1] for x in lower_schedule(c.schedule())]
    for ei in lowered: ei.run()
    rawbufs = lowered[-1].bufs
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

  def test_multioutput(self):
    dtype, st = dtypes.int, ShapeTracker.from_shape((8,))
    g0, g1, g2, g3 = [UOp(Ops.DEFINE_GLOBAL, dtype.ptr(), arg=i) for i in range(4)]
    a = UOp(Ops.LOAD, dtype, src=(g2.view(st),))
    b = UOp(Ops.LOAD, dtype, src=(g3.view(st),))
    out0 = UOp(Ops.STORE, dtypes.void, src=(g0.view(st), a + b))
    out1 = UOp(Ops.STORE, dtypes.void, src=(g1.view(st), a * b))
    sink = UOp(Ops.SINK, src=(out0, out1))

    a_t = Tensor.full(st.shape, 2).contiguous().realize()
    b_t = Tensor.full(st.shape, 3).contiguous().realize()
    lin = helper_linearizer_ast(sink, [a_t, b_t], wanna_output=[a_t.numpy()+b_t.numpy(), a_t.numpy()*b_t.numpy()])[0]
    uops = get_program(lin.get_optimized_ast(), lin.opts).uops

    stores = [u for u in uops if u.op is Ops.STORE]
    mutable_bufs = dedup(flatten([[x for x in u.src[0].toposort() if x.op is Ops.DEFINE_GLOBAL] for u in stores]))
    assert len(mutable_bufs) == len(stores) == 2
    self.assertSetEqual(set([u.arg for u in mutable_bufs]), set([0,1]))

  def _test_no_nested_ranges(self, lins, skip=None):
    for l in lins:
      range_in_acc = flatten([[x for x in u.src if x.op is Ops.RANGE] for u in l.uops if u.op is Ops.DEFINE_REG])
      ranges = [u.op for u in l.uops if (u.op is Ops.RANGE and u in range_in_acc) or (u.op is Ops.ENDRANGE and u.src[0] in range_in_acc)]
      for i,u in enumerate(ranges):
        if skip and i in skip: continue
        assert ranges[i-1] != u, f"multireduce nested the ranges! {ranges[i-1], {u}}"

  @unittest.skipIf(CI and Device.DEFAULT in {"PTX", "AMD", "NV"}, "very slow")
  def test_indexing_multireduce(self):
    dataset = Tensor.rand(16384, 256).realize()
    idxs = Tensor([0,3,5,6]).realize()
    with Context(FUSE_ARANGE=1):
      sink = dataset[idxs].contiguous().kernelize().uop.base.src[1].arg.ast
    real_index = dataset.numpy()[idxs.numpy()].reshape(4, 256, 1, 1)
    helper_linearizer_ast(push_views(sink), [dataset, idxs], wanna_output=[real_index])

  def test_two_nested_range(self):
    a = Tensor.randn(2, ).realize()
    out = a.reshape(2, 1).expand(2, 3).sum()
    lin = helper_linearizer_opt(out, wanna_output=[np.broadcast_to(a.numpy().reshape(2, 1), (2, 3)).sum()])[0]
    uops = get_program(lin.get_optimized_ast(), lin.opts).uops
    ranges = [i for i,u in enumerate(uops) if u.op is Ops.RANGE]
    assert len(ranges) == 1 # NOTE: it collapses now
    # RANGE -> LOAD -> RANGE -> ASSIGN
    #assert any(x.op is Ops.LOAD for x in uops[ranges[0]:ranges[1]])

  def test_three_nested_range(self):
    a = Tensor.randn(2, ).realize()
    out = a.reshape(2, 1).expand(2, 3).expand(2, 2, 3).sum()
    lin = helper_linearizer_opt(out, wanna_output=[np.broadcast_to(np.broadcast_to(a.numpy().reshape(2, 1), (2, 3)), (2, 2, 3)).sum()])[0]
    uops = get_program(lin.get_optimized_ast(), lin.opts).uops
    ranges = [i for i,u in enumerate(uops) if u.op is Ops.RANGE]
    assert len(ranges) == 1 # NOTE: it collapses now
    # RANGE -> RANGE -> LOAD -> RANGE -> ASSIGN
    # NOTE: nothing should toposort between the first two ranges
    #assert ranges[0]+1 == ranges[1]
    #assert any(x.op is Ops.LOAD for x in uops[ranges[1]:ranges[2]])

  def test_two_nested_range_alt_indexing(self):
    a = Tensor([2, 2]).realize()
    out = a.reshape(2, 1).pad(((1, 1), (1, 1)), value=2).sum()
    lin = helper_linearizer_opt(out, wanna_output=[24])[0]
    uops = get_program(lin.get_optimized_ast(), lin.opts).uops
    ranges = [i for i,u in enumerate(uops) if u.op is Ops.RANGE]
    # RANGE -> ALU -> RANGE -> ALU + LOAD -> ASSIGN
    assert any(x.op in GroupOp.ALU for x in uops[ranges[0]:ranges[1]])
    assert not any(x.op is Ops.LOAD for x in uops[ranges[0]:ranges[1]])
    assert any(x.op in {*GroupOp.ALU, Ops.LOAD} for x in uops[ranges[1]:])

  def test_range_outer_op_before_phi(self):
    a = Tensor.randn(4, 1).realize()
    b = Tensor.randn(1, 1).realize()
    out = (a + b[0]).sum() + b[0]
    lin = helper_linearizer_opt(out, wanna_output=[(a.numpy()+b.numpy()[0]).sum()+b.numpy()])[0]
    uops = get_program(lin.get_optimized_ast(), lin.opts).uops
    ranges = [i for i,u in enumerate(uops) if u.op is Ops.RANGE]
    # LOAD -> RANGE -> LOAD -> ASSIGN
    assert len([x for x in uops[:ranges[0]] if x.op is Ops.LOAD]) == 1

  def test_range_outer_op_before_phi_nested_range(self):
    a = Tensor.randn(2, ).realize()
    b = Tensor.randn(1, 1).realize()
    out = (a.reshape(2, 1).expand(2, 3) + b[0]).sum() + b[0]
    lin = helper_linearizer_opt(out, wanna_output=[(np.broadcast_to(a.numpy().reshape(2, 1), (2, 3)) + b.numpy()[0]).sum() + b.numpy()])[0]
    uops = get_program(lin.get_optimized_ast(), lin.opts).uops
    ranges = [i for i,u in enumerate(uops) if u.op is Ops.RANGE]
    assert len(ranges) == 1 # NOTE: it collapses now
    #if getenv("PTX"):
    # LOAD -> RANGE -> CAST -> ALU -> ALU -> LOAD -> ALU -> RANGE -> ALU -> ASSIGN
    #  assert uops[ranges[0]-2].op is Ops.LOAD
    #  assert ranges[1] == ranges[0]+6
    #  assert [x.op for x in uops[ranges[1]-2:ranges[1]]] == [Ops.LOAD, Ops.ALU]
    # LOAD -> RANGE -> LOAD -> ALU -> RANGE -> ASSIGN
    #else:
    #  assert uops[ranges[0]-2].op is Ops.LOAD
    #  assert ranges[1] == ranges[0]+3
    #  assert [x.op for x in uops[ranges[1]-2:ranges[1]]] == [Ops.LOAD, Ops.ALU]

  @unittest.skip("fragile crap")
  def test_range_outer_op_after_phi(self):
    a = Tensor.randn(4, 1).realize()
    out = a.sum() * a.sum()
    lin = helper_linearizer_opt(out, wanna_output=[a.numpy().sum()*a.numpy().sum()])[0]
    uops = get_program(lin.get_optimized_ast(), lin.opts).uops
    # RANGE -> LOAD -> ASSIGN -> ALU
    end = max(i for i,u in enumerate(uops) if u.op is Ops.ENDRANGE)
    # the INDEX can be first
    assert uops[end+1].op in GroupOp.ALU or uops[end+2].op in GroupOp.ALU

  @unittest.skip("fragile crap")
  def test_range_outer_op_after_phi_nested_range(self):
    a = Tensor.randn(2, ).realize()
    out = a.reshape(2, 1).expand(2, 3).sum() + a.reshape(2, 1).expand(2, 3).sum()
    lin = helper_linearizer_opt(out, wanna_output=[(np.broadcast_to(a.numpy().reshape(2, 1), (2, 3))).sum()*2])[0]
    uops = get_program(lin.get_optimized_ast(), lin.opts).uops
    # RANGE -> LOAD -> ASSIGN -> ALU
    end = max(i for i,u in enumerate(uops) if u.op is Ops.ENDRANGE)
    # the INDEX can be first
    assert uops[end+1].op in GroupOp.ALU or uops[end+2].op in GroupOp.ALU

  def test_load_dedup(self):
    # for different leaves in the AST, the same loads may occur.

    a = Tensor.randn(4).realize()
    # these are of size 3 to avoid float4 coalesce
    r = a[:-1] + a[1:]

    uops = get_program(r.schedule()[-1].ast, opts=[Opt(op=OptOps.UPCAST, axis=0, arg=0)]).uops
    num_loads = len([uop for uop in uops if uop.op is Ops.LOAD])
    assert num_loads <= 4, "more load uops than needed"
    assert num_loads >= 4, "unexpected number of uops, maybe this test needs updating?"

  def test_upcast_cse(self):
    # when upcasting, within a subtree, there may be common expressions.

    a, b = Tensor.randn(1).realize(), Tensor.randn(1).realize()
    r = a.expand([2]) + b.expand([2])

    uops = get_program(r.schedule()[-1].ast, opts=[Opt(op=OptOps.UPCAST, axis=0, arg=0)]).uops
    num_ops = len([uop for uop in uops if uop.op in GroupOp.ALU])
    assert num_ops <= 1, "more alu uops than needed"

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.supports_float4, "test requires float4")
  def test_reduce_upcast(self):
    x, w = Tensor.randn((1,1,3)).realize(), Tensor.randn((1,1,2)).realize()
    r = Tensor.conv2d(x,w,padding=1).relu()

    uops = get_program(r.schedule()[-1].ast, opts=[Opt(op=OptOps.UPCAST, axis=0, arg=0), Opt(op=OptOps.UNROLL, axis=0, arg=0)]).uops
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
  @unittest.skipIf(getenv("PTX"), "broken on ptx for some reason")
  def test_upcast_with_locals(self):
    x, y = Tensor.rand(1,128), Tensor.rand(128, 128)
    r = (x@y).relu()
    opts_to_apply = [Opt(op=OptOps.GROUP, axis=0, arg=8), Opt(op=OptOps.LOCAL, axis=0, arg=4), Opt(op=OptOps.UPCAST, axis=0, arg=4)]
    program = get_program(r.schedule()[-1].ast, opts=opts_to_apply)

    stores = [u for u in program.uops if u.op is Ops.STORE and u.src[0].dtype.addrspace != AddrSpace.REG]

    # the first store is to lds and can be upcasted
    assert stores[0].src[1].dtype == dtypes.float.vec(4)
    assert any(x.op is Ops.DEFINE_LOCAL for x in stores[0].toposort())
    # the second store is to gds with no upcasts
    assert stores[1].src[1].dtype == dtypes.float
    assert any(x.op is Ops.DEFINE_GLOBAL for x in stores[1].toposort())

  def test_zero_fold(self):
    a, b = Tensor.randn(1).realize(), Tensor.randn(1).realize()
    r = Tensor.stack(a, b)
    uops = get_program(r.schedule()[-1].ast, opts=[Opt(op=OptOps.UPCAST, axis=0, arg=0)]).uops
    num_ops = len([uop for uop in uops if uop.op in GroupOp.ALU])
    assert num_ops == 0, "more alu uops than needed"

  def test_sum_acc_dtype(self):
    for tensor_dtype, acc_dtype in (
      (dtypes.bool, dtypes.int), (dtypes.int16, dtypes.int), (dtypes.float16, dtypes.float), (dtypes.bfloat16, dtypes.float)):
      if is_dtype_supported(tensor_dtype) and is_dtype_supported(acc_dtype):
        a = Tensor([1, 2, 3], dtype=tensor_dtype).sum()
        realized_ast = a.schedule()[-1].ast
        program = get_program(realized_ast, opts=[])
        local = [uop for uop in program.uops if uop.op is Ops.DEFINE_REG]
        assert local[0].dtype.base == acc_dtype

  def test_arg_acc_dtype(self):
    def helper_arg_acc_dtype(c: Tensor, expected_dtype:DType):
      realized_ast = c.schedule()[-1].ast
      program = get_program(realized_ast, opts=[])
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

  # TODO: don't skip bf16 for real device (METAL, AMD)
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.tensor_cores, "test requires tensor cores")
  def test_tensor_cores(self):
    for tc in Device[Device.DEFAULT].renderer.tensor_cores:
      if not is_dtype_supported(tc.dtype_in) or not is_dtype_supported(tc.dtype_out): continue
      # for AMX, tc.dims[2] == 1 so reduceop is None thus tensor_cores are not triggered
      helper_tc_allclose(tc.dims[0], tc.dims[1], 2 if AMX else tc.dims[2], tc.dtype_in, tc.dtype_out, axis=0, tc_opt=0)

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.tensor_cores, "test requires tensor cores")
  def test_tensor_cores_codegen(self):
    for tc in Device[Device.DEFAULT].renderer.tensor_cores:
      if not is_dtype_supported(tc.dtype_in) or not is_dtype_supported(tc.dtype_out): continue
      n, m, k = tc.dims[0], tc.dims[1], 2 if AMX else tc.dims[2]
      a, b = Tensor.rand(m, k, dtype=tc.dtype_in), Tensor.rand(k, n, dtype=tc.dtype_in)
      r = a.matmul(b, dtype=tc.dtype_out)
      sched = r.schedule()
      realized_ast = push_views(sched[-1].ast)
      kernel = Kernel(realized_ast)
      kernel.apply_tensor_cores(1, axis=0, tc_select=-1, tc_opt=2)
      prg = get_program(kernel.get_optimized_ast(), kernel.opts)
      if Device.DEFAULT == "LLVM":
        assert "0x201000" in prg.src
      elif Device.DEFAULT == "AMD" and AMD_LLVM:
        assert "@llvm.amdgcn.wmma" in prg.src
      elif Device[Device.DEFAULT].renderer.suffix == "PTX":
        assert "mma.sync.aligned" in prg.src
      else:
        assert "__WMMA_" in prg.src

  @unittest.skipIf((Device.DEFAULT == "AMD") or (Device.DEFAULT == "PYTHON" and getenv("EMULATE_AMD")), "broken for AMD")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.tensor_cores, "test requires tensor cores")
  def test_tensor_cores_padded(self):
    for tc in Device[Device.DEFAULT].renderer.tensor_cores:
      if not is_dtype_supported(tc.dtype_in) or not is_dtype_supported(tc.dtype_out): continue
      helper_tc_allclose(tc.dims[0]+(pad:=1), tc.dims[1]+pad, tc.dims[2]+pad, tc.dtype_in, tc.dtype_out, tc_opt=2)

  # AMD compiler bug: AMD miscompiles non-zero padded tc kernels with -O3, producing wrong results, nans or hang (see #9606)
  # Internal bug: zero-stride dimensions combined with a mask may produce wrong index/valid for pad == 1 on AMD
  @unittest.skipUnless((Device.DEFAULT == "AMD") or (Device.DEFAULT == "PYTHON" and getenv("EMULATE_AMD")), "test for AMD's tc")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.tensor_cores, "test requires tensor cores")
  @unittest.expectedFailure
  def test_tensor_cores_padded_amd(self):
    for tc in Device[Device.DEFAULT].renderer.tensor_cores:
      if not is_dtype_supported(tc.dtype_in) or not is_dtype_supported(tc.dtype_out): continue
      helper_tc_allclose(tc.dims[0]+(pad:=1), tc.dims[1]+pad, tc.dims[2]+pad, tc.dtype_in, tc.dtype_out, tc_opt=2)

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.tensor_cores, "test requires tensor cores")
  def test_tensor_cores_padded_uops(self):
    for tc in Device[Device.DEFAULT].renderer.tensor_cores:
      pad = 1

      # check that TC is triggered for TC_OPT=2
      helper_tc_ensure_uops_and_opts_count(tc.dims[0]+pad, tc.dims[1]+pad, tc.dims[2]+pad,
                                           tc.dtype_in, tc.dtype_out, tc_opt=2, ensure_triggered=True)

      # check that TC is not triggered for TC_OPT<2
      helper_tc_ensure_uops_and_opts_count(tc.dims[0]+pad, tc.dims[1]+pad, tc.dims[2]+pad,
                                           tc.dtype_in, tc.dtype_out, tc_opt=1, ensure_triggered=False)
      helper_tc_ensure_uops_and_opts_count(tc.dims[0]+pad, tc.dims[1]+pad, tc.dims[2]+pad,
                                           tc.dtype_in, tc.dtype_out, tc_opt=0, ensure_triggered=False)

      # check excessive padding doesn't trigger padded TC in TC_OPT=2
      helper_tc_ensure_uops_and_opts_count(tc.dims[0]//4, tc.dims[1], tc.dims[2], tc.dtype_in, tc.dtype_out, tc_opt=2, ensure_triggered=False)
      helper_tc_ensure_uops_and_opts_count(tc.dims[0], tc.dims[1]//4, tc.dims[2], tc.dtype_in, tc.dtype_out, tc_opt=2, ensure_triggered=False)
      if not AMX: # AMX tc.dims[2] == 1
        helper_tc_ensure_uops_and_opts_count(tc.dims[0], tc.dims[1], tc.dims[2]//4, tc.dtype_in, tc.dtype_out, tc_opt=2, ensure_triggered=False)

  @unittest.skipIf(CI and Device.DEFAULT in {"AMD"}, "AMD CI is really slow here")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.tensor_cores, "test requires tensor cores")
  def test_tensor_cores_multi_reduce(self):
    for tc in Device[Device.DEFAULT].renderer.tensor_cores:
      if not is_dtype_supported(tc.dtype_in) or not is_dtype_supported(tc.dtype_out): continue
      # this will be a M=G16, N=G32, M=G16, M=G16, K=R16, K=R16, K=R16 with 9 choices of TC MNK axes
      golden_result = None
      for axis in range(9):
        a = Tensor.rand(16, 16, 29, 29, dtype=tc.dtype_in).realize()
        b = Tensor.rand(32, 16, 16, 16, dtype=tc.dtype_in).realize()
        c = a.conv2d(b, padding=1, dtype=tc.dtype_out)
        realized_ast, real_bufs = helper_realized_ast(c)

        opts_to_apply = [Opt(OptOps.TC, axis, (-1, 2, 1))]
        realized_ast = realized_ast.replace(arg=KernelInfo(opts_to_apply=tuple(opts_to_apply)))
        program = get_program(realized_ast, Device[Device.DEFAULT].renderer)
        assert len([uop for uop in program.uops if uop.op is Ops.WMMA]) > 0, "tensor core not triggered"
        assert len([x for x in program.applied_opts if x.op is OptOps.TC]) == 1, "tensor core opt not included"

        prg = CompiledRunner(program)
        # TODO: support this even if numpy doesn't
        if _to_np_dtype(real_bufs[0].dtype) is None: continue
        real_bufs[0].copyin(np.zeros((real_bufs[0].size, ), dtype=_to_np_dtype(real_bufs[0].dtype)).data) # Zero to check that all values are filled
        prg.exec(real_bufs)
        result = np.frombuffer(real_bufs[0].as_buffer(), _to_np_dtype(real_bufs[0].dtype))

        # ensure the results for each choice of axis matches
        if golden_result is None: golden_result = np.frombuffer(real_bufs[0].as_buffer(), _to_np_dtype(real_bufs[0].dtype))
        np.testing.assert_allclose(result, golden_result, atol=0.1, rtol=0.2)

      # check that get_kernel_actions produces all 9 options
      from tinygrad.codegen.opt.search import get_kernel_actions
      tc_actions = [k for i, k in get_kernel_actions(Kernel(realized_ast), False).items() if k.applied_opts[0].op == OptOps.TC]

      available_tc = len([x for x in Device[Device.DEFAULT].renderer.tensor_cores if x.dtype_in == tc.dtype_in and x.dtype_out == tc.dtype_out])
      assert len(tc_actions) == 9 * available_tc, f"should contain 9 possible TC actions for every available TC, got {len(tc_actions)}"

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.tensor_cores, "test requires tensor cores")
  def test_tensor_cores_unroll_phi(self):
    tc = Device[Device.DEFAULT].renderer.tensor_cores[0]
    x, y = Tensor.rand(128, 128, dtype=tc.dtype_in), Tensor.rand(128, 128, dtype=tc.dtype_in)
    r = x.matmul(y, dtype=tc.dtype_out)
    k = helper_linearizer_opt(r, [[Opt(OptOps.UNROLL, 0, 4)]], apply_tc=True, atol=3e-2, rtol=1e-3)[-1]
    for u in get_program(k.get_optimized_ast(), k.opts).uops:
      if u.op is Ops.WMMA:
        assert u.src[-1].src[0].op != Ops.ASSIGN

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.tensor_cores, "test requires tensor cores")
  @unittest.skipIf(Device.DEFAULT in {"CPU", "LLVM"}, "CPU does not support using a different type for accumulation")
  def test_tensor_cores_unroll_casted_phi(self):
    tc = [tc for tc in Device[Device.DEFAULT].renderer.tensor_cores if tc.dtype_in != tc.dtype_out][0]
    x, y = Tensor.rand(128, 128, dtype=tc.dtype_in), Tensor.rand(128, 128, dtype=tc.dtype_in)
    r = x.matmul(y, dtype=tc.dtype_out)
    k = helper_linearizer_opt(r, [[Opt(OptOps.UNROLL, 0, 4)]], apply_tc=True, atol=3e-2, rtol=1e-3)[-1]
    for u in get_program(k.get_optimized_ast(), k.opts).uops:
      if u.op is Ops.WMMA:
        #assert u.src[-1].dtype == dtypes.float.vec(prod(tc.thread_local_sizes[2]))
        assert u.src[-1].src[0].op != Ops.ASSIGN

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.tensor_cores, "test requires tensor cores")
  @unittest.skipIf(Device.DEFAULT in {"CPU", "LLVM"}, "CPU does not support using a different type for accumulation")
  def test_tensor_cores_unroll_casted_phi_with_children(self):
    # all ASSIGN children are outside the loop
    tc = [tc for tc in Device[Device.DEFAULT].renderer.tensor_cores if tc.dtype_in != tc.dtype_out][0]
    x, y = Tensor.rand(128, 128, dtype=tc.dtype_in), Tensor.rand(128, 128, dtype=tc.dtype_in)
    r = x.matmul(y, dtype=tc.dtype_out).relu()
    k = helper_linearizer_opt(r, [[Opt(OptOps.UNROLL, 0, 4)]], apply_tc=True, atol=3e-2, rtol=1e-3)[-1]
    for u in get_program(k.get_optimized_ast(), k.opts).uops:
      if u.op is Ops.WMMA:
        #assert u.src[-1].dtype == dtypes.float.vec(prod(tc.thread_local_sizes[2]))
        assert u.src[-1].src[0].op != Ops.ASSIGN

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.supports_float4, "test requires float4")
  def test_simple_unroll_no_between_phi_dependencies(self):
    x, y = Tensor.rand(128, 128), Tensor.rand(128, 128)
    r = (x@y).relu()
    k = helper_linearizer_opt(r, [[Opt(OptOps.UNROLL, 0, 4), Opt(OptOps.UPCAST, 0, 4)]])[-1]
    # the uops graph is RANGE -> DEFINE_ACC -> 4x ALU -> 4x ASSIGN -> ENDRANGE
    uops = get_program(k.get_optimized_ast(), k.opts).uops
    for u in uops:
      if u.op is Ops.ASSIGN:
        assert u.src[1].op in GroupOp.ALU
      # children of ASSIGN are placed after ENDRANGE
      if any(x.op is Ops.ASSIGN for x in u.src):
        end_range = [i for i, x in enumerate(uops) if x.op is Ops.ENDRANGE][0]
        assert end_range < uops.index(u)

  def test_grouped_dims(self):
    def _assert_grouped_dims(prefix, dims, max_sizes, reverse_dims, expected_sizes, assert_same_length = True):
      idxs = get_grouped_dims(prefix, dims, max_sizes, reverse_dims)
      loop_idxs = dedup(flatten([[y for y in x.toposort() if y.op is Ops.SPECIAL] for x in idxs]))
      loop_idxs = sorted(loop_idxs, key=lambda uop: uop.arg[0])
      sizes = [x.arg[1] for x in loop_idxs]
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

    # # variable too large
    # with self.assertRaises(AssertionError):
    #   get_grouped_dims("gidx", (Variable("start_pos",0,16),3,4), (16,16,16), False,)

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  def test_default_global_reversed(self):
    # shrink so that the dims do not collapse
    t = Tensor.ones(5, 6, 7).contiguous().realize().shrink(((0, 4), (0, 5), (0, 6)))
    k = helper_linearizer_opt(t+1)[0]
    uops = get_program(k.get_optimized_ast(), k.opts).uops
    idxs = dedup([uop for uop in uops if uop.op is Ops.SPECIAL])
    idxs = sorted(idxs, key=lambda uop: uop.arg[0])
    assert idxs[0].arg == ('gidx0', 6), idxs[0].arg
    assert idxs[1].arg == ('gidx1', 5), idxs[1].arg
    assert idxs[2].arg == ('gidx2', 4), idxs[2].arg

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

  def test_where_fold(self):
    a = Tensor.ones(4, 4).contiguous().realize()
    b = a.shrink(((1, 2), None)).pad(((1, 2), None))
    a.assign(b.where(2, a))
    sched = a.schedule()
    assert len(sched) == 1
    sched_copy = sched[:]
    run_schedule(sched)
    np.testing.assert_equal(a.flatten().numpy(), [1.,1.,1.,1.,2.,2.,2.,2.,1.,1.,1.,1.,1.,1.,1.,1.])
    realized_ast = sched_copy[-1].ast
    realized_ast = realized_ast.replace(arg=KernelInfo(opts_to_apply=tuple()))
    program = get_program(realized_ast, Device[Device.DEFAULT].renderer)
    assert not any(u.op == Ops.WHERE for u in program.uops), "found where where where should be folded"

  def test_phi_simplification(self):
    def helper(t, max_ops=0):
      k = helper_linearizer_opt(t)[-1]
      uops = get_program(k.get_optimized_ast(), k.opts).uops
      # ignore kernel optimized IF statements for now
      if if_op:=next((u for u in uops if u.op is Ops.IF), None):
        uops = uops[:uops.index(if_op)]
      assert len(set([u.op for u in uops if u.op in {Ops.RANGE, Ops.SPECIAL}])) == 1, "has either specials or ranges, not both"
      assert len([u for u in uops if u.op is Ops.ASSIGN]) == 0, "ASSIGN should have been simplified"
      # TODO: once uops track min/max this will be fixed
      #assert len([u for u in uops if u.op is Ops.MAX]) <= max_ops, "no unnecessary MAX ops"

    helper(Tensor.arange(5.5, (3.5*300), 3.5), max_ops=2)
    helper(Tensor.arange(-1, -100, -5), max_ops=2)
    # NOTE: both of these split the reduce (this just wasn't tracked before)
    #helper(Tensor.arange(-3.2, 6.7, 0.64), max_ops=2)
    #helper(Tensor.arange(256), max_ops=2)
    helper(Tensor.arange(255), max_ops=2)

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.supports_float4, "test requires float4")
  @unittest.skipIf(getenv("PTX"), "broken on ptx for some reason")
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
    x, y = Tensor.randn(64,64), Tensor.randn(64,64)
    out = x.matmul(y)
    k = helper_linearizer_opt(out)[-1]
    uops = get_program(k.get_optimized_ast(), k.opts).uops
    # check that the float4 cast collapses
    store_vals = [u.src[1] for u in uops if u.op is Ops.STORE and u.src[0].dtype.addrspace != AddrSpace.REG]
    for val in store_vals:
      assert val.dtype == dtypes.float.vec(4) # and val.op is not Ops.VECTORIZE

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_shared, "test requires shared")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.supports_float4, "test requires float4")
  def test_arange_opts(self):
    a = Tensor.arange(128)
    helper_linearizer_opt(a, [
      [Opt(OptOps.GROUP, 0, 32)],
      [Opt(OptOps.GROUPTOP, 0, 32)],
      [Opt(op=OptOps.LOCAL, axis=0, arg=8)],
      [Opt(op=OptOps.LOCAL, axis=0, arg=8), Opt(op=OptOps.UPCAST, axis=0, arg=0)],
      [Opt(op=OptOps.LOCAL, axis=0, arg=8), Opt(op=OptOps.UPCAST, axis=0, arg=0), Opt(op=OptOps.GROUP, axis=0, arg=8)],
      [Opt(op=OptOps.LOCAL, axis=0, arg=8), Opt(op=OptOps.UPCAST, axis=0, arg=0), Opt(op=OptOps.GROUP, axis=0, arg=8), Opt(op=OptOps.UNROLL, axis=1, arg=4)], # noqa: E501
    ])

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.supports_float4, "test requires float4")
  def test_grouped_store_values(self):
    x = Tensor.randn((4,3,6,6)).realize()
    out = x.flip((0,1)).contiguous()
    k = helper_linearizer_opt(out)[-1]
    store_val = [u.src[1] for u in get_program(k.get_optimized_ast(), k.opts).uops if u.op is Ops.STORE][0]
    assert store_val.dtype == dtypes.float.vec(4) and store_val.op is not Ops.VECTORIZE

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_shared, "test requires shared")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.supports_float4, "test requires float4")
  def test_grouped_store_locals_and_globals(self):
    x, y = Tensor.rand(128, 128), Tensor.rand(128, 128)
    out = x@y
    opt = [Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.GROUPTOP, 0, 8),
            Opt(OptOps.UNROLL, 0, 4), Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.UPCAST, 1, 2)] # upcast accs in both reduces
    k = helper_linearizer_opt(out, opts=[opt])[-1]
    def get_recursive(uop): return set.union(set(uop.src), [uop], *[get_recursive(v) for v in uop.src])
    uops = get_program(k.get_optimized_ast(), k.opts).uops
    local_stores = [u for u in uops if u.op is Ops.STORE and any(x.op is Ops.DEFINE_LOCAL for x in get_recursive(u.src[0]))]
    global_stores = [u for u in uops if u.op is Ops.STORE and any(x.op is Ops.DEFINE_GLOBAL for x in get_recursive(u.src[0]))]
    barrier = [u for u in uops if u.op is Ops.BARRIER][0]
    # check that the float4 cast collapses for all stores
    for store in local_stores+global_stores:
      assert store.src[1].dtype.count > 1 # and store.src[2].op is not Ops.VECTORIZE
    # # check the children's vins
    # TODO: src ALU are not the same, should it?
    # assert barrier.src == tuple(local_stores)
    assert len([u for u in uops if u.op is Ops.IF and u.src[-1] == barrier]) == 1

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_shared, "test requires shared")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.supports_float4, "test requires float4")
  @unittest.skipIf(getenv("PTX"), "broken on ptx for some reason")
  def test_grouped_store_local_only(self):
    x, y = Tensor.rand(1,128), Tensor.rand(128, 128)
    r = (x@y).relu()
    k = helper_linearizer_opt(r)[-1]
    uops = get_program(k.get_optimized_ast(), k.opts).uops
    stores = [u for u in uops if u.op is Ops.STORE and u.src[0].dtype.addrspace != AddrSpace.REG]

    # the float4 value stores directly in lds and we skip upcast
    self.assertEqual(stores[0].src[1].dtype, dtypes.float.vec(4))
    #assert stores[0].src[-1].op is not Ops.VECTORIZE

    # the global store doesn't change
    assert stores[1].src[1].dtype == dtypes.float

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.supports_float4, "test requires float4")
  def test_skip_unmatching_upcasts(self):
    Tensor.manual_seed(0)
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.VIEW, dtypes.float.ptr(9600), arg=ShapeTracker(views=(View(shape=(240, 40, 1, 1), strides=(40, 1, 0, 0), offset=0, mask=None, contiguous=True),)), src=( # noqa: E501
          UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(9600), arg=0, src=()),)),
        UOp(Ops.LOAD, dtypes.float, arg=None, src=(
          UOp(Ops.VIEW, dtypes.float.ptr(9600), arg=ShapeTracker(views=(View(shape=(240, 40, 1, 1), strides=(1, 240, 0, 0), offset=0, mask=None, contiguous=False),)), src=( # noqa: E501
            UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(9600), arg=1, src=()),)),)),)),))
    opt = [
        Opt(op=OptOps.UPCAST, axis=1, arg=4), Opt(op=OptOps.LOCAL, axis=0, arg=16),
        Opt(op=OptOps.LOCAL, axis=1, arg=2), Opt(op=OptOps.UPCAST, axis=3, arg=2)
    ]
    k = helper_linearizer_ast(ast, [Tensor.randn(240*40).realize()], opts=[opt])[-1]
    out = [u for u in get_program(k.get_optimized_ast(), k.opts).uops if u.op is Ops.STORE][0]
    assert out.src[1].op is Ops.VECTORIZE and out.src[1].dtype == dtypes.float.vec(4)

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.supports_float4, "test requires float4")
  def test_skip_unmatching_upcasts_with_gep(self):
    Tensor.manual_seed(0)
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.VIEW, dtypes.float.ptr(256), arg=ShapeTracker(views=(View(shape=(8, 32, 1, 1), strides=(32, 1, 0, 0), offset=0, mask=None, contiguous=True),)), src=( # noqa: E501
          UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(256), arg=0, src=()),)),
        UOp(Ops.LOAD, dtypes.float, arg=None, src=(
          UOp(Ops.VIEW, dtypes.float.ptr(256), arg=ShapeTracker(views=(View(shape=(8, 32, 1, 1), strides=(1, 8, 0, 0), offset=0, mask=None, contiguous=False),)), src=( # noqa: E501
            UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(256), arg=1, src=()),)),)),)),))
    opt = [Opt(op=OptOps.LOCAL, axis=1, arg=4), Opt(op=OptOps.UPCAST, axis=2, arg=2), Opt(op=OptOps.LOCAL, axis=1, arg=8),
            Opt(op=OptOps.UPCAST, axis=1, arg=0), Opt(op=OptOps.UPCAST, axis=1, arg=4), Opt(op=OptOps.LOCAL, axis=0, arg=8),
            Opt(op=OptOps.UPCAST, axis=1, arg=0), Opt(op=OptOps.UPCAST, axis=0, arg=2)]
    k = helper_linearizer_ast(ast, [Tensor.randn(8*32).realize()], opts=[opt])[-1]
    out = [u for u in get_program(k.get_optimized_ast(), k.opts).uops if u.op is Ops.STORE][0]
    assert out.src[1].op is Ops.VECTORIZE and out.src[1].dtype.count != 1

@unittest.skipUnless(Device[Device.DEFAULT].renderer.supports_float4, "need backends that support float4")
class TestFloat4(unittest.TestCase):
  @staticmethod
  def count_float4(uops: list[UOp], n=4):
    return (len([uop for uop in uops if uop.op is Ops.LOAD and uop.dtype == dtypes.float.vec(n)]),
            len([uop for uop in uops if uop.op is Ops.STORE and uop.src[1].dtype == dtypes.float.vec(n)]))
  @staticmethod
  def count_half4(uops: list[UOp]):
    return (len([uop for uop in uops if uop.op is Ops.LOAD and uop.dtype == dtypes.half.vec(4)]),
            len([uop for uop in uops if uop.op is Ops.STORE and uop.src[1].dtype == dtypes.half.vec(4)]))

  def test_float4_basic(self):
    a = Tensor.empty(2, 8).realize()
    b = Tensor.empty(2, 8).realize()
    c = a + b

    s = c.schedule()[0]
    realized_ast = s.ast
    opts_to_apply = [Opt(op=OptOps.UPCAST, axis=0, arg=4)]
    realized_ast = realized_ast.replace(arg=KernelInfo(opts_to_apply=tuple(opts_to_apply)))
    program = get_program(realized_ast, Device[Device.DEFAULT].renderer)

    assert TestFloat4.count_float4(program.uops) == (2, 1)

  @unittest.skipIf(Device.DEFAULT in {"CPU", "LLVM"} and AMX, "CPU with AMX upcasts float up to size 16")
  def test_float4_multidim(self):
    a = Tensor.empty(2, 8).realize()
    b = Tensor.empty(2, 8).realize()
    c = a + b

    s = c.schedule()[0]
    uops = get_program(s.ast, opts=[Opt(op=OptOps.UPCAST, axis=0, arg=4), Opt(op=OptOps.UPCAST, axis=0, arg=2)]).uops
    assert TestFloat4.count_float4(uops) == (4, 2)

  @unittest.skipUnless(Device.DEFAULT in {"CPU", "LLVM"} and AMX, "Only CPU with AMX upcasts float up to size 16")
  def test_float4_multidim_amx(self):
    def kernel_for_shape(size, shift):
      a = Tensor.empty(2, size).realize()
      b = Tensor.empty(2, size).realize()
      c = a + b

      s = c.schedule()[0]
      return get_program(s.ast, opts=[Opt(op=OptOps.UPCAST, axis=0, arg=4), Opt(op=OptOps.UPCAST, axis=0, arg=shift)]).uops

    sizes = [12, 8, 16]
    shifts = [3, 2, 4]
    expected_upcast_size = [4, 8, 16]
    expected_output = [(6,3), (2,1), (2,1)]

    for i in range(len(sizes)):
      assert TestFloat4.count_float4(kernel_for_shape(sizes[i], shifts[i]), expected_upcast_size[i]) == expected_output[i]

  def test_float4_unaligned_load(self):
    a = Tensor.empty(9).realize().shrink(((1, 9),))
    b = Tensor.empty(9).realize().shrink(((1, 9),))
    c = a + b

    s = c.schedule()[0]
    realized_ast = s.ast
    opts_to_apply = [Opt(op=OptOps.UPCAST, axis=0, arg=4)]
    realized_ast = realized_ast.replace(arg=KernelInfo(opts_to_apply=tuple(opts_to_apply)))
    program = get_program(realized_ast, Device[Device.DEFAULT].renderer)

    assert TestFloat4.count_float4(program.uops) == (0, 1)

  @unittest.skipIf(Device.DEFAULT in {"CPU", "LLVM"} and AMX, "CPU with AMX upcasts float up to size 16")
  def test_float4_multidim_unaligned_load(self):
    a = Tensor.empty(2, 9).realize().shrink(((0, 2), (1, 9),))
    b = Tensor.empty(2, 9).realize().shrink(((0, 2), (1, 9),))
    c = a + b

    s = c.schedule()[0]
    uops = get_program(s.ast, opts=[Opt(op=OptOps.UPCAST, axis=1, arg=4), Opt(op=OptOps.UPCAST, axis=1, arg=2)]).uops

    assert TestFloat4.count_float4(uops) == (0, 2)

  @unittest.skipUnless(Device.DEFAULT in {"CPU", "LLVM"} and AMX, "Only CPU with AMX upcasts float up to size 16")
  def test_float4_multidim_unaligned_load_amx(self):
    def kernel_for_shape(size, shift):
      a = Tensor.empty(2, size).realize().shrink(((0, 2), (1, size),))
      b = Tensor.empty(2, size).realize().shrink(((0, 2), (1, size),))
      c = a + b

      s = c.schedule()[0]
      return get_program(s.ast, opts=[Opt(op=OptOps.UPCAST, axis=1, arg=4), Opt(op=OptOps.UPCAST, axis=1, arg=shift)]).uops

    sizes = [13, 9, 17]
    shifts = [3, 2, 4]
    expected_upcast_size = [4, 8, 16]
    expected_output = [(0,3), (0,1), (0,1)]

    for i in range(len(sizes)):
      assert TestFloat4.count_float4(kernel_for_shape(sizes[i], shifts[i]), expected_upcast_size[i]) == expected_output[i]

  def test_float4_sometimes_unaligned(self):
    a = Tensor.empty(1, 1, 8).realize()
    b = Tensor.empty(1, 1, 5).realize().shrink(((0, 1), (0, 1), (1, 5)))
    c = a.conv2d(b)
    # only the first and last conv dot products are aligned in a, and b is never aligned, so no
    # float4 should be emitted (the reduce axis of size 4 is the float4 axis here)

    s = c.schedule()[0]
    uops = get_program(s.ast, opts=[Opt(op=OptOps.UNROLL, axis=0, arg=4)]).uops

    assert TestFloat4.count_float4(uops) == (0, 0)

  def test_float4_multidim_sometimes_unaligned(self):
    a = Tensor.empty(1, 1, 7).realize()
    b = Tensor.empty(1, 1, 5).realize().shrink(((0, 1), (0, 1), (1, 5)))
    c = a.conv2d(b)
    # the first conv dot product is aligned in a. If we upcast the output and reduce
    # dimension, then we could do float4 for only that one set of loads, but we currently
    # don't.
    # UPDATE: now we do this fusion

    s = c.schedule()[0]
    uops = get_program(s.ast, opts=[Opt(op=OptOps.UPCAST, axis=0, arg=0), Opt(op=OptOps.UNROLL, axis=0, arg=0)]).uops

    assert TestFloat4.count_float4(uops) in {(0,1), (1,1)}

  def test_float4_expand(self):
    a = Tensor.empty(9).realize().shrink(((1, 9),))
    b = Tensor.empty(2).realize().reshape((2, 1)).expand((2,4)).reshape((8,))
    c = a + b

    # we will upcast the top axis of sz 4. they should not be coalesced into float4,
    # since the top axis is not contiguous.

    s = c.schedule()[0]
    uops = get_program(s.ast, opts=[Opt(op=OptOps.UPCAST, axis=0, arg=4)]).uops

    assert TestFloat4.count_float4(uops) == (0, 1)

  def test_float4_heterogeneous(self):
    a = Tensor.empty(8).realize()
    b = Tensor.empty(9).realize().shrink(((1, 9),))
    c = a + b

    # should float4 b but not a

    s = c.schedule()[0]
    uops = get_program(s.ast, opts=[Opt(op=OptOps.UPCAST, axis=0, arg=4)]).uops

    assert TestFloat4.count_float4(uops) == (1, 1)

  def test_half4_load_unrolled(self):
    # from llama 7B shard 4 gpus
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.VIEW, dtypes.float.ptr(96000), arg=ShapeTracker(views=(View(shape=(1, 3, 32000, 1), strides=(0, 32000, 1, 0), offset=0, mask=None, contiguous=True),)), src=( # noqa: E501
          UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(96000), arg=0, src=()),)),
        UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (3,)), src=(
          UOp(Ops.CAST, dtypes.float, arg=None, src=(
            UOp(Ops.MUL, dtypes.half, arg=None, src=(
              UOp(Ops.LOAD, dtypes.half, arg=None, src=(
                UOp(Ops.VIEW, dtypes.half.ptr(9216), arg=ShapeTracker(views=(View(shape=(1, 3, 32000, 1024), strides=(0, 4096, 0, 1), offset=0, mask=None, contiguous=False),)), src=( # noqa: E501
                  UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(9216), arg=1, src=()),)),)),
              UOp(Ops.LOAD, dtypes.half, arg=None, src=(
                UOp(Ops.VIEW, dtypes.half.ptr(32768000), arg=ShapeTracker(views=(View(shape=(1, 3, 32000, 1024), strides=(0, 0, 1024, 1), offset=0, mask=None, contiguous=False),)), src=( # noqa: E501
                  UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(32768000), arg=2, src=()),)),)),)),)),)),)),))

    # TODO: fix this, expected might change but should be positive
    for expected, opts in [
      ((7, 0), [Opt(op=OptOps.UPCAST, axis=1, arg=4), Opt(op=OptOps.UPCAST, axis=0, arg=3), Opt(op=OptOps.UNROLL, axis=0, arg=4)]),
      ((5, 0), [Opt(op=OptOps.UPCAST, axis=1, arg=4), Opt(op=OptOps.UNROLL, axis=0, arg=4)]),
      ((2, 0), [Opt(op=OptOps.UNROLL, axis=0, arg=4)]),
    ]:
      ast = ast.replace(arg=KernelInfo(opts_to_apply=tuple(opts)))
      program = get_program(ast, Device[Device.DEFAULT].renderer)

      count = TestFloat4.count_half4(program.uops)
      assert count == expected, f"{count=}, {expected=}"

  @unittest.skip("this doesn't happen anymore")
  def test_float4_acc(self):
    # from float32 stable diffusion red tinybox
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.VIEW, dtypes.float.ptr(33554432), arg=ShapeTracker(views=(View(shape=(1, 1, 128, 512, 512, 1, 1, 1), strides=(0, 0, 262144, 512, 1, 0, 0, 0), offset=0, mask=None, contiguous=True),)), src=( # noqa: E501
          UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(33554432), arg=0, src=()),)),
        UOp(Ops.ADD, dtypes.float, arg=None, src=(
          UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (5, 6, 7)), src=(
            UOp(Ops.MUL, dtypes.float, arg=None, src=(
              UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                UOp(Ops.VIEW, dtypes.float.ptr(67108864), arg=ShapeTracker(views=(View(shape=(1, 1, 1, 256, 4, 514, 4, 514), strides=(0, 0, 0, 262144, 0, 512, 0, 1), offset=-513, mask=((0, 1), (0, 1), (0, 1), (0, 256), (0, 4), (1, 513), (0, 4), (1, 513)), contiguous=False), View(shape=(1, 1, 128, 512, 512, 256, 3, 3), strides=(0, 0, 0, 2056, 1, 4227136, 1058840, 515), offset=0, mask=None, contiguous=False))), src=( # noqa: E501
                  UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(67108864), arg=1, src=()),)),)),
              UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                UOp(Ops.VIEW, dtypes.float.ptr(294912), arg=ShapeTracker(views=(View(shape=(1, 1, 128, 512, 512, 256, 3, 3), strides=(0, 0, 2304, 0, 0, 9, 3, 1), offset=0, mask=None, contiguous=False),)), src=( # noqa: E501
                  UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(294912), arg=2, src=()),)),)),)),)),
          UOp(Ops.LOAD, dtypes.float, arg=None, src=(
            UOp(Ops.VIEW, dtypes.float.ptr(128), arg=ShapeTracker(views=(View(shape=(1, 1, 128, 512, 512, 1, 1, 1), strides=(0, 0, 1, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),)), src=( # noqa: E501
              UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(128), arg=3, src=()),)),)),)),)),))

    for expected, opts in [
      (1, [Opt(op=OptOps.UPCAST, axis=2, arg=4)]),
      (4, [Opt(op=OptOps.UPCAST, axis=2, arg=4), Opt(op=OptOps.UPCAST, axis=0, arg=4)]),
    ]:
      ast = ast.replace(arg=KernelInfo(opts_to_apply=tuple(opts)))
      program = get_program(ast, Device[Device.DEFAULT].renderer)
      count = len([uop for uop in program.uops if uop.op is Ops.DEFINE_REG and uop.dtype == dtypes.float.vec(4)])
      assert count == expected, f"{count=}, {expected=}"

  @unittest.skip("this doesn't happen anymore")
  def test_float2_acc(self):
    # from resnet
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.VIEW, dtypes.half.ptr(212926464), arg=ShapeTracker(views=(View(shape=(1, 256, 1, 64, 1, 114, 1, 114), strides=(0, 831744, 0, 12996, 0, 114, 0, 1), offset=0, mask=None, contiguous=True),)), src=( # noqa: E501
          UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(212926464), arg=0, src=()),)),
        UOp(Ops.CAST, dtypes.half, arg=None, src=(
          UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (4, 6)), src=(
            UOp(Ops.CAST, dtypes.float, arg=None, src=(
              UOp(Ops.LOAD, dtypes.half, arg=None, src=(
                UOp(Ops.VIEW, dtypes.half.ptr(462422016), arg=ShapeTracker(views=(View(shape=(256, 64, 3, 56, 2, 3, 56, 2), strides=(1806336, 28224, 3, 504, 0, 1, 9, 0), offset=0, mask=((0, 256), (0, 64), (0, 3), (0, 56), (0, 1), (0, 3), (0, 56), (0, 1)), contiguous=False), View(shape=(256, 64, 3, 115, 3, 115), strides=(7225344, 112896, 37632, 336, 112, 1), offset=0, mask=((0, 256), (0, 64), (0, 3), (0, 112), (0, 3), (0, 112)), contiguous=False), View(shape=(256, 64, 456, 456), strides=(7617600, 119025, 345, 1), offset=0, mask=((0, 256), (0, 64), (0, 345), (0, 345)), contiguous=False), View(shape=(1, 256, 1, 64, 4, 114, 4, 114), strides=(0, 13307904, 0, 207936, 51984, 456, 114, 1), offset=0, mask=None, contiguous=True))), src=( # noqa: E501
                  UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(462422016), arg=1, src=()),)),)),)),)),)),)),))
    for expected, opts in [
      (16, [Opt(op=OptOps.LOCAL, axis=1, arg=16), Opt(op=OptOps.UPCAST, axis=1, arg=0), Opt(op=OptOps.UPCAST, axis=2, arg=2), Opt(op=OptOps.LOCAL, axis=2, arg=3), Opt(op=OptOps.UPCAST, axis=3, arg=4)]),  # noqa: E501
      (4, [Opt(op=OptOps.LOCAL, axis=1, arg=16), Opt(op=OptOps.UPCAST, axis=1, arg=0), Opt(op=OptOps.UPCAST, axis=2, arg=2)]),
    ]:
      ast = ast.replace(arg=KernelInfo(opts_to_apply=tuple(opts)))
      program = get_program(ast, Device[Device.DEFAULT].renderer)
      count = len([uop for uop in program.uops if uop.op is Ops.DEFINE_REG and uop.dtype == dtypes.float.vec(2)])
      assert count == expected, f"{count=}, {expected=}"

class TestHandCodedOpts(unittest.TestCase):
  def test_masked_upcast(self):
    layer_1 = Tensor.cat(*[Tensor.empty(5) for _ in range(4)])
    layer_2 = Tensor.cat(layer_1.unsqueeze(0), Tensor.empty(6, 20))

    s = layer_2.schedule()[-1]
    k = Kernel(push_views(s.ast))
    k.apply_opts(hand_coded_optimizations(k))
    assert len(k.bufs) == 6  # make sure all ops are done in one kernel
    # masked upcast should upcast masked axis of size 7
    # masked upcast should not upcast large (20) last axis
    # float4/other hcopt shouldn't upcast last axis, since we already have 7 upcast, and the last axis is not very contiguous
    assert k.upcasted == 1 and k.full_shape[-1] == 7

  @unittest.skipIf(Device.DEFAULT in {"METAL", "WEBGPU"}, "METAL/WEBGPU split this kernel since it has 37 buffers")
  def test_masked_upcast_wino(self):
    monster = Tensor.stack(*[Tensor.stack(*[Tensor.empty(16) for _ in range(6)]) for _ in range(6)])

    s = monster.schedule()[-1]
    k = Kernel(push_views(s.ast))
    k.apply_opts(hand_coded_optimizations(k))
    assert len(k.bufs) == 37  # make sure all ops are done in one kernel
    # should upcast the two Tensor.stacks
    assert k.upcasted >= 2 and k.full_shape[k.shape_len-k.upcasted:k.shape_len].count(6) == 2

  def test_masked_upcast_wino_full(self):
    with Context(WINO=1):
      x,w = Tensor.rand(1,4,8,8, requires_grad=True).realize(), Tensor.rand(4,4,3,3, requires_grad=True).realize()
      out = Tensor.conv2d(x,w, padding=1)
      out.mean().backward()

      upcasts = []
      wino_schedule = out.schedule()
      # collect upcasts of tile transform kernels
      for i, si in enumerate(wino_schedule):
        k = Kernel(push_views(si.ast))
        k.apply_opts(hand_coded_optimizations(k))
        if k.reduceop is not None: continue  # not a tile transform kernel (there is a gemm reduce kernel)
        if len(k.bufs) < 22: continue  # not a tile transform kernel (there's a permute kernel at the end)
        upcasts.append(tuple(k.full_shape[k.shape_len - k.upcasted:k.shape_len]))
      assert len(upcasts) == 3  # 3 transformation matrices
      assert len(wino_schedule) <= 4  # 4 kernels
      # this test case's inputs are too small, so one of the 4-stacks became a local, which is fine i guess
      assert upcasts.count((6, 6)) == 2 #and upcasts.count((4, 4)) == 1

      backward_schedule = Tensor.schedule(x.grad, w.grad)
      for si in backward_schedule:
        k = Kernel(push_views(si.ast))
        k.apply_opts(hand_coded_optimizations(k))
        if len(k.bufs) < 20: continue  # not a tile transform kernel
        # heuristic number to make sure that at least some upcasts but not too many upcasts are being done
        assert 6 <= prod(k.full_shape[k.shape_len - k.upcasted:k.shape_len]) <= 216
      assert len(backward_schedule) <= 13  # just the current number, but it could be better

  def test_masked_upcast_many(self):
    layer_1 = Tensor.cat(Tensor.rand(3, 4), Tensor.rand(4, 4))
    layer_2 = Tensor.cat(layer_1.unsqueeze(0), Tensor.rand(6, 7, 4))
    layer_3 = Tensor.cat(layer_2.unsqueeze(0), Tensor.rand(6, 7, 7, 4))

    k = helper_linearizer_opt(layer_3)[-1]
    assert len(k.bufs) == 5  # make sure all ops are done in one kernel
    # check that we don't do too many upcasts
    assert prod(k.full_shape[k.shape_len-k.upcasted:k.shape_len]) <= 49

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  def test_matvec(self):
    N = 128
    a = Tensor.rand(1, N).realize()
    b = Tensor.rand(N, N).realize()
    c = a @ b

    k = helper_linearizer_opt(c)[-1]

    assert k.group_for_reduces == 1
    assert k.axis_types.count(AxisType.LOCAL) == 1
    assert k.upcasted == 1

def helper_linearizer_ast(ast:UOp, inputs:list[Tensor], *args, **kwargs):
  assert isinstance(ast, UOp), "ast must be UOp"
  inbufs = [x.uop.base.buffer for x in inputs]
  outbufs = [Buffer(inbufs[-1].device if inbufs else Device.DEFAULT, out.st_arg.size, out.src[1].dtype).allocate() \
      for out in ast.src]
  return _helper_linearizer_opt_ast(ast, outbufs+inbufs, *args, **kwargs)

def helper_linearizer_opt(r:Tensor|list[Tensor], *args, **kwargs):
  realized_ast, real_bufs = helper_realized_ast(r)
  return _helper_linearizer_opt_ast(realized_ast, real_bufs, *args, **kwargs)

def copyout_outputs(lin:Kernel, outbufs:list[Buffer]) -> list[np.ndarray]:
  ret = []
  for i,x in enumerate(outbufs):
    shape: tuple[int, ...] = lin.ast.src[i].st_arg.shape
    ret.append(np.frombuffer(x.as_buffer(), _to_np_dtype(x.dtype)).reshape(shape))
  return ret

def reset_bufs(bufs:list[Buffer]):
  for buf in bufs: buf.copyin(np.zeros((buf.size, ), dtype=_to_np_dtype(buf.dtype)).data) # Zero to check that all values are filled

def _helper_linearizer_opt_ast(realized_ast:UOp, real_bufs:list[Buffer], opts=[],
                               apply_tc=False, atol=1e-4, rtol=1e-4, color_sizes=[], wanna_output=[]) -> list[Kernel]:
  lins: list[Kernel] = []
  outbufs = [real_bufs[x.src[0].base.arg] for x in realized_ast.src]
  device = real_bufs[0].device

  def get_prg(k:Kernel): return CompiledRunner(replace(get_program(k.get_optimized_ast(), k.opts), device=device))

  def check_opt(opts, create_k, expected_color_size):
    k = create_k()
    lins.append(k)
    if apply_tc:
      assert k.apply_tensor_cores(1, extra_opts=opts), "no tensor core triggered"
    else:
      k.apply_opts(opts)
    if expected_color_size is not None:
      cs = list(zip(k.colors(), k.full_shape))
      assert cs == expected_color_size, f"expected={expected_color_size} got={cs}"
    prg = get_prg(k)
    reset_bufs(outbufs)
    prg.exec(real_bufs)

    for x,want in zip(copyout_outputs(k, outbufs), wanna_output): np.testing.assert_allclose(x, want, atol=atol, rtol=rtol)

  # Get baseline if it is not provided, which is not optimized at all.
  k = Kernel(realized_ast)
  lins.append(k)
  prg = get_prg(k)
  prg.exec(real_bufs)
  if len(wanna_output) == 0: wanna_output = copyout_outputs(k, outbufs)
  else:
    for buf,want in zip(copyout_outputs(k, outbufs), wanna_output): np.testing.assert_allclose(buf, want, atol=atol, rtol=rtol)

  # Check correctness of handcoded optimiztions.
  k = Kernel(realized_ast)
  k.apply_opts(hand_coded_optimizations(k))
  lins.append(k)
  prg = get_prg(k)
  reset_bufs(outbufs)
  prg.exec(real_bufs)
  for buf,want in zip(copyout_outputs(k, outbufs), wanna_output): np.testing.assert_allclose(buf, want, atol=atol, rtol=rtol)
  for i,x in enumerate(opts): # Check custom transformations if any.
    check_opt(x, lambda: Kernel(realized_ast), color_sizes[i] if i < len(color_sizes) else None)
  return lins

class TestKernelOpts(unittest.TestCase):
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_shared, "test requires shared")
  def test_local_and_grouped_reduce(self):
    N = 128
    Tensor.manual_seed(1882)
    a = Tensor.rand(4, 4, N, N)
    b = Tensor.rand(4, 4, N)
    r = (b.sqrt() + ((a+1).sum(axis=3).exp()))
    helper_linearizer_opt(r, [
      [Opt(OptOps.LOCAL, 0, 2)],
      [Opt(OptOps.LOCAL, 0, 8)],
      [Opt(OptOps.LOCAL, 0, 16)], # Checking how it works with locals
      [Opt(OptOps.GROUPTOP, 0, 2)],
      [Opt(OptOps.GROUPTOP, 0, 32)],
      [Opt(OptOps.GROUPTOP, 0, 64)], # Checking how it works with grouped reduce
      [Opt(OptOps.LOCAL, 0, 2), Opt(OptOps.GROUPTOP, 0, 2)],
      [Opt(OptOps.LOCAL, 0, 16), Opt(OptOps.GROUPTOP, 0, 16)],
      [Opt(OptOps.LOCAL, 0, 32), Opt(OptOps.GROUPTOP, 0, 2)],
      # Checking how it works with locals + grouped reduce
      [Opt(OptOps.LOCAL, 0, 2), Opt(OptOps.GROUPTOP, 0, 64)],
      # Checking how it works with locals + grouped reduce + upcasts
      [Opt(OptOps.LOCAL, 0, 2), Opt(OptOps.GROUPTOP, 0, 2), Opt(OptOps.UPCAST, 0, 8), Opt(OptOps.UNROLL, 1, 4)],
      # many local + many group
      [Opt(OptOps.GROUP, 0, 2)] * 4,
      [Opt(OptOps.LOCAL, 0, 2)] * 4,
      [Opt(OptOps.LOCAL, 0, 2), Opt(OptOps.GROUP, 0, 2)] * 4,
    ])

  def test_upcasts(self):
    N = 16
    Tensor.manual_seed(1772)
    a = Tensor.rand(N, N)
    b = Tensor.rand(N, N)
    r = (a+b).sqrt() * ((a+1).exp())
    helper_linearizer_opt(r, [
      [Opt(OptOps.UPCAST, 0, 2)],
      [Opt(OptOps.UPCAST, 0, 4)],
      [Opt(OptOps.UPCAST, 0, 8)], # Checking how it works with upcasts
    ])

  def test_full_upcast(self):
    Tensor.manual_seed(1772)
    a = Tensor.rand(4)
    b = Tensor.rand(4)
    r = (a+b).sqrt() * ((a+1).exp())
    helper_linearizer_opt(r, [
      [Opt(OptOps.UPCAST, 0, 4)], # Checking how it works with upcasts
    ])

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_shared, "test requires shared")
  def test_matmul(self):
    N = 128
    Tensor.manual_seed(1552)
    a = Tensor.rand(N, N)
    b = Tensor.rand(N, N)
    r = a@b
    helper_linearizer_opt(r, [
      [Opt(OptOps.UPCAST, 0, 2)],
      [Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.UPCAST, 1, 4)], # Checking how it works with upcasts
      [Opt(OptOps.LOCAL, 0, 2)],
      [Opt(OptOps.LOCAL, 1, 32)],
      [Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.LOCAL, 1, 4)],
      [Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.LOCAL, 1, 32)],
      [Opt(OptOps.LOCAL, 0, 16), Opt(OptOps.LOCAL, 1, 8)], # Checking how it works with locals
      [Opt(OptOps.GROUPTOP, 0, 2)],
      [Opt(OptOps.GROUPTOP, 0, 32)],
      [Opt(OptOps.GROUPTOP, 0, 32), Opt(OptOps.UNROLL, 0, 4)], # Checking how it works with grouped_reduce
      [Opt(OptOps.LOCAL, 0, 2), Opt(OptOps.LOCAL, 1, 2), Opt(OptOps.GROUPTOP, 0, 32)],
      [Opt(OptOps.LOCAL, 0, 8), Opt(OptOps.GROUPTOP, 0, 32)],
      [Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.LOCAL, 0, 8), Opt(OptOps.GROUPTOP, 0, 4)], # Checking how it works with local+grouped_reduce
      # Checking all together
      [Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.GROUPTOP, 0, 8), Opt(OptOps.UNROLL, 0, 4), Opt(OptOps.UPCAST, 0, 4),
       Opt(OptOps.UPCAST, 1, 2)],
      # Full global upcast + local
      [Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.GROUPTOP, 0, 8), Opt(OptOps.UNROLL, 0, 4), Opt(OptOps.UPCAST, 0, 8)],
    ])

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_shared, "test requires shared")
  def test_double_reduce(self):
    N = 128
    Tensor.manual_seed(1552)
    a = Tensor.rand(8, N, 8, N)
    r = a.sum(axis=(1,3))
    helper_linearizer_opt(r, [
      # openCL / GPU=1 is 256 max threads
      [Opt(OptOps.GROUPTOP, 0, 2)], [Opt(OptOps.GROUPTOP, 0, 32)],
      [Opt(OptOps.GROUPTOP, 1, 2)], [Opt(OptOps.GROUPTOP, 1, 32)], # Checking how it works with 1 grouped_reduce.
      [Opt(OptOps.GROUPTOP, 0, 2), Opt(OptOps.GROUPTOP, 1, 2)],
      [Opt(OptOps.GROUPTOP, 0, 16), Opt(OptOps.GROUPTOP, 1, 2)],
      [Opt(OptOps.GROUPTOP, 0, 4), Opt(OptOps.GROUPTOP, 1, 64)], # Checking how it works with 2 grouped_reduces.
      [Opt(OptOps.GROUPTOP, 0, 16), Opt(OptOps.GROUPTOP, 1, 2), Opt(OptOps.UNROLL, 0, 4)],
      [Opt(OptOps.GROUPTOP, 0, 2), Opt(OptOps.GROUPTOP, 1, 32), Opt(OptOps.UNROLL, 2, 4)], # Checking how it works with 2 grouped_reduces + upcasts.
      [Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.LOCAL, 1, 4), Opt(OptOps.GROUPTOP, 0, 4), Opt(OptOps.GROUPTOP, 1, 4)],
      # Checking how it works with 2 grouped_reduces + upcasts + locals.
      [Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.LOCAL, 1, 4), Opt(OptOps.GROUPTOP, 0, 2), Opt(OptOps.GROUPTOP, 1, 32), Opt(OptOps.UNROLL, 1, 4)],
      [Opt(OptOps.LOCAL, 0, 2), Opt(OptOps.LOCAL, 1, 2), Opt(OptOps.GROUPTOP, 0, 8), Opt(OptOps.GROUPTOP, 1, 4), Opt(OptOps.UPCAST, 0, 2)],
      [Opt(OptOps.LOCAL, 0, 2), Opt(OptOps.LOCAL, 1, 2), Opt(OptOps.GROUPTOP, 0, 8), Opt(OptOps.GROUPTOP, 1, 4), Opt(OptOps.UPCAST, 0, 2),
       Opt(OptOps.UNROLL, 0, 4), Opt(OptOps.UNROLL, 1, 4)], # Checking how it works with 2 grouped_reduces + upcasts + locals.
      [Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.LOCAL, 1, 4), Opt(OptOps.GROUPTOP, 0, 4), Opt(OptOps.GROUPTOP, 1, 4), Opt(OptOps.UPCAST, 0, 2),
       Opt(OptOps.UPCAST, 0, 2)], # No globals
    ])

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.tensor_cores, "test requires tensor cores")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  def test_invalid_tensor_core_extra_opts(self):
    N = 128
    Tensor.manual_seed(1552)
    a = Tensor.rand(N, N)
    b = Tensor.rand(N, N)
    realized_ast, _ = helper_realized_ast(a@b)
    invalid_opts = [
      [Opt(OptOps.LOCAL, 2, 2)],
      [Opt(OptOps.UPCAST, 2, 2)],
      [Opt(OptOps.LOCAL, 0, 2), Opt(OptOps.LOCAL, 2, 2)],
    ]
    for x in invalid_opts:
      k = Kernel(realized_ast)
      with self.assertRaises(AssertionError):
        assert k.apply_tensor_cores(use_tensor_cores=1, extra_opts=x), "no valid tensor core" # for METAL in runners

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.tensor_cores, "test requires tensor cores")
  @unittest.skipUnless(any(tc.dtype_in == tc.dtype_out == dtypes.half for tc in Device[Device.DEFAULT].renderer.tensor_cores),
                      "test requires tensor cores with accumulation in half") # testing with half suffices.
  def test_tensor_core_opts(self):
    N = 128
    Tensor.manual_seed(1552)
    a, b = Tensor.rand(N, N, dtype=dtypes.half), Tensor.rand(N, N, dtype=dtypes.half)
    r = a.matmul(b, dtype=dtypes.half)
    atol, rtol = 0.25, 0.01
    helper_linearizer_opt(r, [
      [],
      [Opt(OptOps.UPCAST, 0, 4)],
      [Opt(OptOps.UPCAST, 1, 4)],
      [Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.UPCAST, 1, 4)], # check upcasts
      [Opt(OptOps.UNROLL, 0, 2)], # check unroll
      [Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.UNROLL, 0, 2)], # check combo of unroll and local
      [Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.UPCAST, 1, 4), Opt(OptOps.UNROLL, 0, 2)],
      [Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.UPCAST, 1, 4), Opt(OptOps.UNROLL, 0, 4)],
      [Opt(OptOps.UPCAST, 1, 4), Opt(OptOps.UPCAST, 0, 4)], # check permutations
      [Opt(OptOps.UNROLL, 0, 2), Opt(OptOps.UPCAST, 0, 4)],
      [Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.UNROLL, 0, 2), Opt(OptOps.UPCAST, 1, 4)],
      [Opt(OptOps.UNROLL, 0, 2), Opt(OptOps.UPCAST, 1, 4), Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.UNROLL, 0, 4)],
    ], apply_tc=True, atol=atol, rtol=rtol)

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.tensor_cores, "test requires tensor cores")
  @unittest.skipUnless(any(tc.dtype_in == tc.dtype_out == dtypes.half for tc in Device[Device.DEFAULT].renderer.tensor_cores),
                      "test requires tensor cores with accumulation in half") # testing with half suffices.
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  def test_tensor_core_opts_locals(self):
    N = 128
    Tensor.manual_seed(1552)
    a, b = Tensor.rand(N, N, dtype=dtypes.half), Tensor.rand(N, N, dtype=dtypes.half)
    r = a.matmul(b, dtype=dtypes.half)
    atol, rtol = 0.25, 0.01
    helper_linearizer_opt(r, [
      [Opt(OptOps.UNROLL, 0, 0)], # check full unroll of reduce with locals
      [Opt(OptOps.LOCAL, 0, 4)], # check local
      [Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.UPCAST, 1, 4), Opt(OptOps.UNROLL, 0, 4), Opt(OptOps.LOCAL, 0, 2)],
      [Opt(OptOps.LOCAL, 0, 2), Opt(OptOps.UPCAST, 1, 4), Opt(OptOps.UNROLL, 0, 2), Opt(OptOps.UPCAST, 0, 4)],
    ], apply_tc=True, atol=atol, rtol=rtol)

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.tensor_cores, "test requires tensor cores")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_shared, "test requires shared memory")
  @unittest.skipUnless(any(tc.dtype_in == tc.dtype_out == dtypes.half for tc in Device[Device.DEFAULT].renderer.tensor_cores),
                      "test requires tensor cores with accumulation in half") # testing with half suffices.
  # NOTE: the METAL test is broken, likely due to a compiler bug. passes on CI with -O0 and with default opt level locally on M3
  @unittest.skipIf(Device.DEFAULT == "METAL", "broken for METAL")
  @unittest.skip("feature was removed")
  def test_tensor_core_opts_group(self):
    N = 128
    Tensor.manual_seed(1552)
    a, b = Tensor.rand(N, N, dtype=dtypes.half), Tensor.rand(N, N, dtype=dtypes.half)
    r = a.matmul(b, dtype=dtypes.half)
    atol, rtol = 0.25, 0.01
    helper_linearizer_opt(r, [
      [Opt(OptOps.GROUP, 0, 2)],
      [Opt(OptOps.GROUPTOP, 0, 4)],
      [Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.GROUP, 0, 2)],
      [Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.GROUP, 0, 2)],
      [Opt(OptOps.UNROLL, 0, 4), Opt(OptOps.GROUP, 0, 2)],
      [Opt(OptOps.UPCAST, 0, 2), Opt(OptOps.LOCAL, 0, 2), Opt(OptOps.GROUP, 0, 2)],
      [Opt(OptOps.LOCAL, 0, 2), Opt(OptOps.GROUPTOP, 0, 8), Opt(OptOps.UNROLL, 0, 2), Opt(OptOps.UPCAST, 1, 2)],
    ], apply_tc=True, atol=atol, rtol=rtol)

  def test_padto_matmul(self):
    if (CI and Device.DEFAULT in ["AMD", "NV", "CUDA"]):
      self.skipTest("super slow on CUDA and AMD because of the big grid dims")
    N = 17 * 17
    Tensor.manual_seed(289)
    a = Tensor.rand(N, N)
    b = Tensor.rand(N, N)
    helper_linearizer_opt(a@b, [
      [Opt(OptOps.PADTO, 0, 32)],
      [Opt(OptOps.PADTO, 1, 32)],
      [Opt(OptOps.PADTO, 2, 32)],
      [Opt(OptOps.PADTO, 0, 32), Opt(OptOps.PADTO, 1, 32)],
      [Opt(OptOps.PADTO, 0, 32), Opt(OptOps.PADTO, 1, 32), Opt(OptOps.PADTO, 2, 32)],
      # can optimize further post PADTO
      [Opt(OptOps.PADTO, 0, 32), Opt(OptOps.PADTO, 1, 32), Opt(OptOps.UPCAST, 0, 2), Opt(OptOps.UPCAST, 1, 2),],
    ])

  def test_padto_upcasted_not_ok(self):
    N = 4
    a = Tensor.rand(N, N)
    b = Tensor.rand(N, N)
    helper_linearizer_opt(a@b, [
      [Opt(OptOps.UPCAST, 0, 0)],
      [Opt(OptOps.UPCAST, 1, 0)],
      [Opt(OptOps.UNROLL, 0, 0)],
      [Opt(OptOps.PADTO, 0, 8)],
      [Opt(OptOps.PADTO, 1, 8)],
      [Opt(OptOps.PADTO, 2, 8)],
    ])
    with self.assertRaises(KernelOptError):
      helper_linearizer_opt(a@b, [[Opt(OptOps.UPCAST, 0, 0), Opt(OptOps.PADTO, 1, 8)]])
    with self.assertRaises(KernelOptError):
      helper_linearizer_opt(a@b, [[Opt(OptOps.UPCAST, 1, 0), Opt(OptOps.PADTO, 1, 8)]])
    with self.assertRaises(KernelOptError):
      helper_linearizer_opt(a@b, [[Opt(OptOps.UNROLL, 0, 0), Opt(OptOps.PADTO, 2, 8)]])

  def test_padto_sum_ok(self):
    N = 18 * 18
    # NOTE: this setup prevents 17 * 17 contiguous merged into one dimension
    a = Tensor.rand(N, N).realize().shrink(((0, 17), (0, 17))) * 100
    b = (Tensor.rand(N, N) < 0.5).realize().shrink(((0, 17), (0, 17)))

    helper_linearizer_opt(a.sum(0), [
      [Opt(OptOps.PADTO, 0, 32)],
      [Opt(OptOps.PADTO, 0, 32), Opt(OptOps.UPCAST, 0, 8),],
    ])
    helper_linearizer_opt(a.sum(1), [
      [Opt(OptOps.PADTO, 0, 32)],
      [Opt(OptOps.PADTO, 0, 32), Opt(OptOps.UPCAST, 0, 8),],
    ])

    # can pad sum reduce axis if there's no unsafe ops prior to sum
    for axis in (0, 1):
      helper_linearizer_opt(a.sum(), [[Opt(OptOps.PADTO, axis, 32)],])
      helper_linearizer_opt(a.sum(0), [[Opt(OptOps.PADTO, axis, 32)],])
      helper_linearizer_opt(b.sum(), [[Opt(OptOps.PADTO, axis, 32)],])
      helper_linearizer_opt(b.sum(0), [[Opt(OptOps.PADTO, axis, 32)],])
      helper_linearizer_opt(b.sum(dtype=dtypes.bool), [[Opt(OptOps.PADTO, axis, 32)],])
      # TODO: why?
      if Device.DEFAULT != "WEBGPU":
        helper_linearizer_opt(b.sum(0, dtype=dtypes.bool), [[Opt(OptOps.PADTO, axis, 32)],])
        helper_linearizer_opt(b.sum(1, dtype=dtypes.bool), [[Opt(OptOps.PADTO, axis, 32)],])

    # having unsafe ops after sum is fine
    helper_linearizer_opt(a.sum().exp(), [[Opt(OptOps.PADTO, 0, 32)],])
    helper_linearizer_opt(a.sum(0).exp(), [[Opt(OptOps.PADTO, 1, 32)],])

  def test_padto_sum_not_ok(self):
    N = 18 * 18
    # NOTE: this setup prevents 17 * 17 contiguous merged into one dimension
    a = Tensor.rand(N, N).shrink(((0, 17), (0, 17))).exp()
    # exp is not safe to pad
    with self.assertRaises(KernelOptError):
      helper_linearizer_opt(a.exp().sum(), [[Opt(OptOps.PADTO, 0, 32)],])
    with self.assertRaises(KernelOptError):
      helper_linearizer_opt(a.exp().sum(0), [[Opt(OptOps.PADTO, 1, 32)],])

    b = a < 1
    # lt is not safe to pad
    with self.assertRaises(KernelOptError):
      helper_linearizer_opt(b.sum(), [[Opt(OptOps.PADTO, 0, 32)],])
    with self.assertRaises(KernelOptError):
      helper_linearizer_opt(b.sum(0), [[Opt(OptOps.PADTO, 1, 32)],])

  def test_padto_max(self):
    N = 18 * 18
    # NOTE: this setup prevents 17 * 17 contiguous merged into one axis
    a = -Tensor.rand(N, N).shrink(((0, 17), (0, 17))) * 100

    helper_linearizer_opt(a.max(0), [
      [Opt(OptOps.PADTO, 0, 32)],
      [Opt(OptOps.PADTO, 0, 32), Opt(OptOps.UPCAST, 0, 8),],
    ])
    helper_linearizer_opt(a.max(1), [
      [Opt(OptOps.PADTO, 0, 32)],
      [Opt(OptOps.PADTO, 0, 32), Opt(OptOps.UPCAST, 0, 8),],
    ])

    # cannot pad max kernel on reduce
    with self.assertRaises(KernelOptError):
      helper_linearizer_opt(a.max(), [[Opt(OptOps.PADTO, 0, 32)],])
    with self.assertRaises(KernelOptError):
      helper_linearizer_opt(a.max(0), [[Opt(OptOps.PADTO, 1, 32)],])

  def test_padto_where(self):
    Tensor.manual_seed(0)
    N = 17 * 17
    a = (Tensor.randn(N, N).realize().max(axis=0, keepdim=True) > 1).where(1, 0)
    helper_linearizer_opt(a.max(0), [
      [Opt(OptOps.PADTO, 0, 32)],
      [Opt(OptOps.PADTO, 0, 32), Opt(OptOps.UPCAST, 0, 8),],
    ])

  def test_padto_where_multioutput(self):
    Tensor.manual_seed(0)
    N = 17 * 17
    r = Tensor.randn(N, N).realize().max(axis=0, keepdim=True) > 1
    a0 = r.where(1, 0)
    a1 = r.where(2, 0)
    helper_linearizer_opt([a0.max(0), a1.max(0)], [
      [Opt(OptOps.PADTO, 0, 32)],
      [Opt(OptOps.PADTO, 0, 32), Opt(OptOps.UPCAST, 0, 8),],
    ])

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "test requires locals")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_shared, "test requires shared")
  def test_color_shapes_with_local(self):
    N = 32
    Tensor.manual_seed(1552)
    a = Tensor.rand(N, N)
    b = Tensor.rand(N, N)
    r = a@b
    opts_shapes = [
      ([Opt(OptOps.LOCAL, 0, 2)], [("blue",16),("blue",32),("cyan",2),("red",32)]),
      ([Opt(OptOps.LOCAL, 0, 2),Opt(OptOps.GROUP, 0, 2)], [("blue",16),("blue",32),("cyan",2),("green",2),("red",16)]),
      # check to ensure local_dims are stable for full UNROLL of the first reduce
      ([Opt(OptOps.LOCAL, 0, 2),Opt(OptOps.UNROLL, 0, 0)], [("blue",16),("blue",32),("cyan",2),("magenta",32)]),
      ([Opt(OptOps.UNROLL, 0, 0),Opt(OptOps.LOCAL, 0, 2)], [("blue",16),("blue",32),("cyan",2),("magenta",32)]),
      # check behavior for full UNROLL on an existing GROUP
      ([Opt(OptOps.LOCAL, 0, 2),Opt(OptOps.GROUP, 0, 0),Opt(OptOps.UNROLL, 0, 2)], [("blue",16),("blue",32),("cyan",2),("green",16),("magenta",2)]),
      ([Opt(OptOps.LOCAL, 0, 2),Opt(OptOps.GROUP, 0, 0),Opt(OptOps.UNROLL, 0, 0)], [("blue",16),("blue",32),("cyan",2),("magenta",32)]),
      ([Opt(OptOps.GROUP, 0, 0),Opt(OptOps.LOCAL, 0, 2),Opt(OptOps.UNROLL, 0, 0)], [("blue",16),("blue",32),("cyan",2),("magenta",32)]),
      ([Opt(OptOps.GROUP, 0, 2),Opt(OptOps.UNROLL, 0, 0)], [("blue",32),("blue",32),("red",16),("magenta",2)]),
    ]
    helper_linearizer_opt(r, [x[0] for x in opts_shapes], color_sizes=[x[1] for x in opts_shapes])

if __name__ == '__main__':
  unittest.main()
