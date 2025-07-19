# this will be the new test_ops for the next level
# schedule confirms the right things are capable of fusing
# NOTE: this has overlap with external_test_opt.py

import unittest
import numpy as np
import functools
from typing import List, Optional, Union, cast
from hypothesis import assume, given, strategies as strat

from tinygrad import nn, dtypes, Device, Tensor
from tinygrad.device import is_dtype_supported
from tinygrad.dtype import DType, ImageDType
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.uop.ops import PatternMatcher, UOp, Ops, GroupOp, UPat, graph_rewrite, track_rewrites
from tinygrad.uop.symbolic import symbolic_simple
from tinygrad.helpers import CI, DEBUG, FUSE_ARANGE, SPLIT_REDUCEOP, GlobalCounters, Context, getenv, all_same, temp
from tinygrad.kernelize.kernelize import merge_views, get_kernelize_map, Kernel
from tinygrad.engine.schedule import ScheduleItem, create_schedule_with_vars
from tinygrad.engine.realize import CompiledRunner, run_schedule, lower_schedule

class KernelCountException(Exception): pass
def check_schedule(t:Union[Tensor, List[Tensor], UOp], allowed:int, to_prerealize:Optional[List[Tensor]]=None, filter_sink=True):
  if to_prerealize:
    with Context(DEBUG=0, TRACK_MATCH_STATS=0): Tensor.realize(*to_prerealize)
  if isinstance(t, Tensor): sched = t.schedule()
  elif isinstance(t, List) and isinstance(t[0], Tensor): sched = Tensor.schedule(*t)
  else:
    assert isinstance(t, UOp), f"can't schedule {t}"
    sink = UOp.sink(t) if t.op is not Ops.SINK else t
    becomes_map = get_kernelize_map(sink)
    sched, _ = create_schedule_with_vars(sink.substitute(becomes_map))
  # test lowering all the ScheduleItems to ExecItems
  kernel_cnt = len([si for si,ei in lower_schedule(sched.copy()) if isinstance(ei.prg, CompiledRunner) or not filter_sink])
  if kernel_cnt != allowed:
    print(f"SCHEDULE ISSUE, expecting {allowed} got {len(sched)}")
    if DEBUG >= 3:
      for i,s in enumerate(sched):
        print("kernel", i+1)
        print(s.ast)
    raise KernelCountException(f"{kernel_cnt} != {allowed}")
  return sched

def _realize_weights(m):
  for p in nn.state.get_parameters(m): p.realize()

def _test_conv2d(allowed:int, dtype:DType=dtypes.float, **kwargs):
  old_default_float, dtypes.default_float = dtypes.default_float, dtype
  dtypes.default_float = dtype
  Tensor.manual_seed(0)
  BS, CIN = 2, 3
  img = Tensor.randn(BS, CIN, 64, 64, requires_grad=True).realize()
  w = Tensor.uniform(16, CIN, 3, 3, requires_grad=True).realize()
  ret = Tensor.conv2d(img, w).relu().mean().backward()
  dtypes.default_float = old_default_float
  with Context(**kwargs): s = Tensor.schedule(ret, img.grad, w.grad)
  run_schedule(s.copy())
  cnt = len([si for si in s if si.ast.op is Ops.SINK])
  assert cnt == allowed, f"expected {allowed} kernels, got {cnt}"
  if getenv("CHECK", 1):
    import torch
    ref_img = torch.tensor(img.numpy(), requires_grad=True)
    ref_w = torch.tensor(w.numpy(), requires_grad=True)
    torch.nn.functional.conv2d(ref_img, ref_w).relu().mean().backward()
    assert ref_img.grad is not None and ref_w.grad is not None and img.grad is not None and w.grad is not None
    np.testing.assert_allclose(img.grad.numpy(), ref_img.grad.detach().numpy(), atol=1e-6 if dtype == dtypes.float else 1e-2)
    np.testing.assert_allclose(w.grad.numpy(), ref_w.grad.detach().numpy(), atol=1e-6 if dtype == dtypes.float else 1e-2)

@track_rewrites(name=True)
def schedule_graph_rewrite(big_sink:UOp): return get_kernelize_map(big_sink)[big_sink]

class TestSchedule(unittest.TestCase):
  def test_arange_avgpool2d(self, kcount=2):
    x = Tensor.arange(25).reshape(1,1,5,5).cast(dtypes.float32)
    t = x.avg_pool2d(padding=1)
    sched = t.schedule()
    self.assertEqual(len(sched), kcount)
    run_schedule(sched)
    import torch
    torch_out = torch.nn.functional.avg_pool2d(torch.arange(25).reshape(1,1,5,5).float(), kernel_size=(2,2), padding=1).numpy()
    np.testing.assert_allclose(t.numpy(), torch_out)

  def test_arange_avgpool2d_fused_noopt(self):
    with Context(FUSE_ARANGE=1, NOOPT=1): self.test_arange_avgpool2d(kcount=1)

  # linearizer error
  @unittest.skip("recursion error no longer raised")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.supports_float4, "needs supports_float4 to fail")
  def test_arange_avgpool2d_fused(self):
    with self.assertRaises(RecursionError):
      with Context(FUSE_ARANGE=1, NOOPT=0): self.test_arange_avgpool2d(kcount=1)

  # when we're fusing a reduce, all ReduceOps must have the same N in the dimensions
  # all permutes, reshapes, expands and shrinks push through the reduce
  def test_arange_sum(self):
    a = Tensor.arange(6).reshape(3, 2).sum(axis=1)
    with Context(FUSE_ARANGE=1):
      run_schedule(check_schedule(a, 1))
    self.assertListEqual(a.tolist(), [1, 5, 9])

  def test_arange_sum_alt(self):
    a = (Tensor.arange(5).reshape(1,5).expand(6,5)*Tensor(2)).reshape(1,6,5).sum(axis=2)
    with Context(FUSE_ARANGE=1):
      run_schedule(check_schedule(a, 1))
    np.testing.assert_equal(a.numpy(), 20)

  def test_permute_arange(self):
    a = Tensor.arange(6).reshape(6, 1, 1).permute(2, 0, 1).sum(axis=1)
    with Context(FUSE_ARANGE=1):
      run_schedule(check_schedule(a, 1))
    self.assertListEqual(a.tolist(), [[15]])

  @unittest.skipIf(Device.DEFAULT == "CPU", "devices must mismatch")
  def test_error_on_device_mismatch(self):
    a = Tensor.empty(10)
    b = Tensor.empty(10, device="CPU")
    c = a+b
    with self.assertRaisesRegex(RuntimeError, "all buffers must be on the same device"): check_schedule(c, 1)

  @unittest.skipIf(Device.DEFAULT == "CPU", "devices must mismatch")
  def test_error_on_device_mismatch_alt(self):
    a = Tensor.empty(10)
    b = Tensor.empty((1,), device="CPU").expand(10).contiguous()
    c = a+b
    with self.assertRaisesRegex(RuntimeError, "all buffers must be on the same device"): check_schedule(c, 1)

  @unittest.skipUnless(is_dtype_supported(dtypes.half) and getenv("CAST_AFTER_EXPAND"), "need half and CAST_AFTER_EXPAND=1")
  @unittest.skip("CAST_AFTER_EXPAND is not supported")
  def test_expand_buffer_before_cast(self):
    a = Tensor.randn(4, 2, 1).realize().permute((1, 0, 2))
    b = a.cast(dtypes.half).expand((2, 4, 4))+2
    run_schedule(check_schedule(b, 1))
    np.testing.assert_allclose(b.numpy(), np.broadcast_to(a.numpy().astype(np.float16), (2, 4, 4))+2)

  def test_indexing_scalars_simple(self):
    X = Tensor.randn(2, 2).realize()
    xt = X[Tensor(1)][Tensor(0)]
    with Context(FUSE_ARANGE=1):
      run_schedule(check_schedule(xt, 2))
    np.testing.assert_equal(xt.numpy(), X.numpy()[1][0])

  @unittest.skipIf(CI and Device.DEFAULT == "NV", "crashes on NV CI")
  def test_add_chain_buffers(self):
    N = 31
    with Context(TRACK_MATCH_STATS=0, DEBUG=0):
      bufs = [Tensor(i).reshape((1,)).contiguous().realize() for i in range(N)]
    for X in range(1,N):
      root = bufs[0]
      for i in range(1,N,X):
        root = root + functools.reduce(lambda a,b:a+b, bufs[i:i+X])
      self.assertEqual(root.item(), sum(range(N)))

  @given(strat.sampled_from(range(2,4)), strat.sampled_from(range(2,4)), strat.sampled_from(range(0,4)), strat.sampled_from(range(0,4)))
  def test_indexing_scalars(self, x, y, a, b):
    assume(a<x and b<y)
    X = Tensor.randn(x, y).realize()
    xt = X[Tensor(a)][Tensor(b)]
    with Context(FUSE_ARANGE=1):
      run_schedule(check_schedule(xt, 2))
    np.testing.assert_equal(xt.numpy(), X.numpy()[a][b])

  def test_push_pads_elementwise(self):
    x = Tensor.full((4,4), 2.).contiguous().realize()
    y = Tensor.full((4,4), 4.).contiguous().realize()
    z = (x.reciprocal()*y).pad((None, (0,1),)).sum()
    run_schedule(check_schedule(z, 2))
    self.assertEqual(z.item(), 32)

  def test_push_pads_contiguous(self):
    x = Tensor.full((4,1), 2.).contiguous()
    y = Tensor.full((4,4), 4.).contiguous()
    z = (x.reciprocal().expand(4,4)*y).pad((None, (0,1),)).sum()
    run_schedule(check_schedule(z, 2, [x,y]))
    self.assertEqual(z.item(), 32)

  def test_rand(self):
    x = Tensor.rand(32)
    check_schedule(x, 4, [Tensor._device_rng_counters[x.device]])

  def test_rand_recompute_arange(self):
    x = Tensor.rand(32)
    with Context(DONT_GROUP_REDUCES=1):
      check_schedule(x, 3, [Tensor._device_rng_counters[x.device]])

  def test_empty_is_not_realized(self):
    a = Tensor.empty(10)
    child = a+2
    assert not a.uop.is_realized
    child.realize()
    assert a.uop.is_realized

  # NOTE: because empty does not have an ExecItem if realize is called on a childless empty, it never gets allocated.
  def test_childless_empty_never_allocates(self):
    a = Tensor.empty(10)
    a.realize()
    assert not a.uop.is_realized

  def test_simplify_padded_const(self):
    a = Tensor.empty(1022).cummax(axis=0)
    sched = check_schedule(a, 5)
    ast = sched[0].ast
    self.assertLessEqual(len([u for u in ast.toposort() if u.op is Ops.WHERE]), 6)

  def test_basic_binop_fusion(self):
    a = Tensor.empty(10)
    b = Tensor.empty(10)
    c = Tensor.empty(10)
    d = a+b+c
    check_schedule(d, 1)

  def test_basic_binop_fusion_deep(self):
    a = Tensor.empty(10)
    b = Tensor.empty(10)
    c = Tensor.empty(10)
    d = Tensor.empty(10)
    e = a+b+c+d
    check_schedule(e, 1)

  def test_mulacc_fusion(self):
    a = Tensor.empty(10)
    b = Tensor.empty(10)
    c = (a*b).sum()
    check_schedule(c, 1)

  def test_mulacc_relu_fusion(self):
    a = Tensor.empty(10)
    b = Tensor.empty(10)
    c = (a*b).sum().relu()
    check_schedule(c, 1)

  def test_binop_reshape_fusion(self):
    a = Tensor.empty(10)
    b = Tensor.empty(10)
    c = Tensor.empty(5,2)
    d = (a+b).reshape(5,2)+c
    check_schedule(d, 1)

  def test_binop_permute_fusion(self):
    a = Tensor.empty(2,5)
    b = Tensor.empty(2,5)
    c = Tensor.empty(5,2)
    d = (a+b).permute(1,0)+c
    check_schedule(d, 1)

  def test_constants_are_embedded(self):
    a = Tensor.empty(3,3) * 2
    check_schedule(a, 1, filter_sink=False)

  def tests_constants_are_folded(self):
    a = Tensor(2)
    check_schedule(a, 0)

  def test_constants_can_store(self):
    a = Tensor(2).contiguous()
    run_schedule(check_schedule(a, 1))
    np.testing.assert_equal(a.numpy(), 2)

  def test_binop_elu_fusion(self):
    a = Tensor.empty(10)
    b = a.elu()
    check_schedule(b, 1)

  def test_binop_reshape_reduce_fusion(self):
    a = Tensor.empty(100)
    b = Tensor.empty(100)
    c = (a+b).reshape(10, 10).sum(axis=0, keepdim=True)
    check_schedule(c, 1)

  def test_reduce_reshape_binop_fusion(self):
    a = Tensor.empty(10,10)
    b = Tensor.empty(10)
    c = a.sum(axis=0) + b
    check_schedule(c, 1)

  # not pushing permutes through reduces
  def test_reduce_permute_binop_fusion(self):
    a = Tensor.empty(10,10,10)
    b = Tensor.empty(10,10,1)
    c = a.sum(axis=0, keepdim=True).permute(2,1,0) + b
    with self.assertRaises(KernelCountException): check_schedule(c, 1)

  def test_allow_push_permutes(self):
    a = Tensor.randn(10,10,10).realize()
    b = Tensor.randn(10,10,1).realize()
    c = a.sum(axis=0, keepdim=True).permute(2,1,0) + b
    with Context(DONT_GROUP_REDUCES=1): run_schedule(check_schedule(c, 1))
    np.testing.assert_allclose(c.numpy(), np.sum(a.numpy(), axis=0, keepdims=True).transpose(2,1,0)+b.numpy())

  def test_binop_early_reshape_reduce_fusion(self):
    a = Tensor.empty(100)
    b = Tensor.empty(100)
    c = Tensor.empty(10,10)
    d = ((a+b).reshape(10,10) + c).sum(axis=0)
    check_schedule(d, 1)

  def test_diamond_folded(self):
    a = Tensor.empty(10)
    b = Tensor.empty(10)
    c = Tensor.empty(10)
    d = Tensor.empty(10)
    ab = a+b
    e = (ab+c) + (ab+d)
    check_schedule(e, 1)

  def test_cache_binaryop(self):
    a = Tensor.empty(10)
    b = Tensor.empty(10)
    c = a+b
    d = a+b
    check_schedule(d, 0, [c])

  # failing in new lazy
  def test_cache_binaryop_reshaped(self):
    a = Tensor.empty(10)
    b = Tensor.empty(10)
    c = a+b
    d = a.reshape(10,1)+b.reshape(10,1)
    with self.assertRaises(KernelCountException): check_schedule(d, 0, [c])

  # failing in new lazy
  def test_cache_binaryop_transpose(self):
    a = Tensor.empty(10,10)
    b = Tensor.empty(10,10)
    c = (a.T*b.T).T #.contiguous()
    d = a*b
    with self.assertRaises(KernelCountException): check_schedule(d, 0, [c])

  def test_cache_two_reduceops(self):
    a = Tensor.empty(10)
    b = a.sum()
    c = a.sum()
    bc = b+c
    check_schedule(bc, 1)

  def test_cache_reduce_parent(self):
    x = Tensor.empty(32)
    r0 = x.mean(axis=0, keepdim=True)
    r1 = (x - r0).sum(axis=0).div(2)
    out = r0 + r1
    schedule = check_schedule(out, 2)
    reduceops = [x for si in schedule for x in si.ast.toposort() if x.op is Ops.REDUCE_AXIS]
    assert len(reduceops) == 2

  def test_cache_reduce_multiple_children(self):
    x = Tensor.empty(32)
    y = Tensor.empty(4, 4)
    r0 = x.mean(axis=0, keepdim=True)
    r1 = (x - r0).sum(axis=0).div(2)
    out0 = r0 + y
    out1 = r1 + y
    schedule = check_schedule([out0, out1], 4)
    reduceops = [x for si in schedule for x in si.ast.toposort() if x.op is Ops.REDUCE_AXIS]
    assert len(reduceops) == 2

  def test_div_collapse_buffer(self):
    a = Tensor.full((4,), 4.0).contiguous().realize()
    b = Tensor.full((4,), 2.0).contiguous().realize()
    expr = (a*b)/b
    check_schedule(expr, 0)
    np.testing.assert_allclose(expr.numpy(), np.full((4,), 4.0))

  def test_div_collapse_const(self):
    a = Tensor.full((4,), 4.0).contiguous().realize()
    expr = a/a
    check_schedule(expr, 0)
    np.testing.assert_allclose(expr.numpy(), np.full((4,), 1.0))

  def test_div_collapse(self):
    a = Tensor.full((4,), 1.0).contiguous().realize()
    b = Tensor.full((4,), 2.0).contiguous().realize()
    c = Tensor.full((4,), 3.0).contiguous().realize()
    GlobalCounters.reset()
    expr = (a/b)/c
    expr.realize()
    self.assertEqual(GlobalCounters.kernel_count, 1)
    self.assertLessEqual(GlobalCounters.global_ops, 4*3)
    np.testing.assert_allclose(expr.numpy(), (a.numpy()/b.numpy())/c.numpy())

  def test_dedup_assign(self):
    a = Tensor.ones(4).contiguous().realize()
    b = Tensor.full((4,), 2.).contiguous()
    first = a.assign(b)
    second = a.assign(b)
    check_schedule([first, second], 1)

  # NOTE: this is causing "LAZYCACHE=1 incorrectly reuses contiguous const" #4562
  # should contiguous dedup?
  def test_dedup_contiguous(self):
    a = Tensor.ones(4).contiguous()
    b = Tensor.ones(4).contiguous()
    sched = check_schedule([a, b], 1)
    run_schedule(sched)
    # a and b share the same underlying device memory
    self.assertIs(a.uop.realized, b.uop.realized)

  def test_clone_doesnt_dedup(self):
    src = Tensor.ones(4).contiguous().realize()
    a = src.clone()
    b = src.clone()
    sched = check_schedule([a, b], 2, filter_sink=False)
    run_schedule(sched)
    # a and b are assigned to the same device Buffer
    self.assertIsNot(a.uop.realized, b.uop.realized)

  # EMPTY is assigned to a unique device Buffer

  def test_no_dedup_empty(self):
    a = Tensor.empty((4,))
    b = Tensor.empty((4,))
    # NOTE: empty does not have any schedule
    check_schedule([a, b], 0, filter_sink=False)
    self.assertIsNot(a.uop.buffer, b.uop.buffer)

  def test_dedup_outputs(self):
    a = Tensor.full((4, 4), 1.).contiguous().realize()
    b = Tensor.full((4, 4), 1.).contiguous().realize()
    check_schedule([a+b, a+b], 1)

  def test_fold_double_unary(self):
    y = Tensor.empty(2)
    out = y.sum(keepdim=True).sqrt().neg()
    check_schedule(out, 1)

  #@unittest.skip("may want to reconsider this")
  def test_fold_batchnorm(self):
    with Tensor.train():
      img = Tensor.empty(1,32,4,4)
      bn = nn.BatchNorm2d(32, track_running_stats=False)
      out = bn(img)
      check_schedule(out, 3)

  def test_fold_conv_batchnorm_notrain(self):
    with Tensor.train(False):
      img = Tensor.empty(1,3,8,8)
      c1 = nn.Conv2d(3,32,3)
      bn = nn.BatchNorm2d(32, track_running_stats=True)
      out = bn(c1(img)).relu()
      check_schedule(out, 1, [c1.weight, c1.bias])

  def test_fold_conv_batchnorm_notrain_no_running_stats(self):
    with Tensor.train(False):
      img = Tensor.empty(1,3,8,8)
      c1 = nn.Conv2d(3,32,3)
      bn = nn.BatchNorm2d(32, track_running_stats=False)
      out = bn(c1(img)).relu()
      check_schedule(out, 4, [c1.weight, c1.bias])

  def test_fold_conv_batchnorm(self):
    with Tensor.train():
      img = Tensor.empty(1,3,8,8)
      c1 = nn.Conv2d(3,32,3)
      bn = nn.BatchNorm2d(32, track_running_stats=False)
      out = bn(c1(img)).relu()
      check_schedule(out, 4, [c1.weight, c1.bias])

  @unittest.skipUnless(is_dtype_supported(dtypes.ulong), "Needs ulong")
  def test_fold_conv_batchnorm_optim(self):
    # this is too high
    for optim, cnt in [(nn.optim.Adam, 30), (nn.optim.SGD, 11)]:
      with self.subTest(optim=optim.__name__):
        with Tensor.train():
          img = Tensor.ones(1,3,4,4)
          c1 = nn.Conv2d(3,32,3)
          bn = nn.BatchNorm2d(32, track_running_stats=False)
          _realize_weights([c1, bn])
          opt = optim(nn.state.get_parameters([c1, bn]))
          img_bn = bn(c1(img)).elu().sum()
          opt.zero_grad()
          img_bn.backward()
          check_schedule(opt.schedule_step(), cnt)

  def test_fold_batchnorm_backward(self):
    with Context(FUSE_CONV_BW=1):
      with Tensor.train():
        x = Tensor.empty((2, 16, 8, 8)).contiguous()
        bn = nn.BatchNorm2d(16)
        bn.weight.requires_grad = bn.bias.requires_grad = x.requires_grad = True
        fw = bn(x).contiguous_backward().relu().contiguous()
        fw.sum().backward()
        # TODO: this is too many
        check_schedule([x.grad, bn.weight.grad, bn.bias.grad, fw], 10)

  def test_fold_conv_relu(self):
    c1 = nn.Conv2d(3,16,3)

    # run
    img = Tensor.ones(2,3,64,64)
    out = c1(img).relu()
    check_schedule(out, 1, [c1.weight, c1.bias])

  def test_fold_conv_relu_alt(self):
    img = Tensor.ones(1,4,8,8)
    c1 = nn.Conv2d(4, 4, kernel_size=3)
    c2 = nn.Conv2d(4, 4, kernel_size=3)
    img_conv = img.sequential([c1, Tensor.relu, c2, Tensor.relu])
    check_schedule(img_conv, 2, [*nn.state.get_parameters(c1), *nn.state.get_parameters(c2), img])

  def test_fold_conv_relu_nobias(self):
    img = Tensor.ones(1,4,8,8)
    c1 = nn.Conv2d(4, 4, kernel_size=3, bias=False)
    c2 = nn.Conv2d(4, 4, kernel_size=3, bias=False)
    out = img.sequential([c1, Tensor.relu, c2, Tensor.relu])
    check_schedule(out, 2, [c1.weight, c2.weight, img])

  def test_fold_conv_elu(self):
    c1 = nn.Conv2d(3,16,3)

    # run
    img = Tensor.rand(2,3,64,64)
    out = c1(img).elu()
    check_schedule(out, 1, [c1.weight, c1.bias, img])

  def test_fold_conv_elu_alt(self):
    img = Tensor.ones(1,4,8,8).contiguous()
    c1 = nn.Conv2d(4, 4, kernel_size=3)
    c2 = nn.Conv2d(4, 4, kernel_size=3)
    img_conv = img.sequential([c1, Tensor.elu, c2, Tensor.elu])
    check_schedule(img_conv, 2, [*nn.state.get_parameters(c1), *nn.state.get_parameters(c2), img])

  def test_two_sum(self):
    img = Tensor.empty(64,64)
    x = (img.sum(0) + img.sum(1))
    out = x.relu()
    check_schedule(out, 2)

  #@unittest.skip("failing in old lazy")
  def test_push_permute_through_reshape(self):
    a = Tensor.empty(16,16)
    b = Tensor.empty(16,16)
    c = (a+b).reshape(4,4,4,4).permute(2,3,0,1).contiguous()
    check_schedule(c, 1)

  #@unittest.skip("failing in old lazy")
  def test_push_permute_through_reshape_alt(self):
    a = Tensor.empty(4,4,4,4)
    b = Tensor.empty(4,4,4,4)
    c = (a+b).reshape(16,16).permute(1,0).contiguous()
    check_schedule(c, 1)

  def test_no_binop_rerun(self):
    a = Tensor.empty(16)
    b = Tensor.empty(16)
    c = a+b
    d = (a+b).reshape(16,1)
    check_schedule(d, 0, [c])

  @unittest.skipUnless(is_dtype_supported(dtypes.half), "need half")
  def test_multi_permute_should_collapse(self):
    a = Tensor.empty(4,4,4,4)
    b = Tensor.empty(16)
    c = a.sum((0,1)).cast(dtypes.float16).permute(1,0).reshape(4,4,1).permute(1,0,2).reshape(16) + b
    check_schedule(c, 1)

  def test_fancy_reshape_fusion(self):
    a = Tensor.empty(10)
    b = Tensor.empty(10)
    c = a+b
    d = a.reshape(10,1)+b.reshape(10,1)
    out = c.sum() + d.sum()
    with self.assertRaises(KernelCountException): check_schedule(out, 1)

  def test_children_dont_push(self):
    a = Tensor.empty(10, 10, 1)
    b = Tensor.empty(10, 10, 1)
    d = (a+b).expand(10, 10, 10)
    e = (a+b).permute(2,1,0)
    f = d+e
    check_schedule(f, 2)

  # failing in new lazy
  def test_dont_fuse_binops_with_children(self):
    a = Tensor.empty(10)
    b = Tensor.empty(10)
    c = Tensor.empty(10)
    keep_me = a+b
    e = keep_me.sum() # noqa: F841 give keep_me a child (NOTE: BinaryOps won't be a child since it will instant fuse)
    d = keep_me+c
    with self.assertRaises(KernelCountException): check_schedule(d, 2)
    with self.assertRaises(KernelCountException): check_schedule(keep_me, 0, [d])

  #@unittest.skip("failing in old lazy")
  def test_permute_breaks_fusion(self):
    a = Tensor.empty(10, 10, 10)
    b = Tensor.empty(10, 10)
    c = (a.sum(axis=2) + b).permute(1,0)
    d = c.permute(1,0)
    check_schedule(d, 1)

  def test_some_permute_fusion(self):
    a = Tensor.empty(8192, 16)
    b = Tensor.empty(1, 16)
    d = (a.T + b.expand(8192, 16).T)
    c = a + b.expand(8192, 16)
    e = d.T
    check_schedule(c, 1)
    check_schedule(e, 1)

  def test_shrink_fuse(self):
    a = Tensor.empty(8192, 16)
    b = Tensor.empty(8192, 16)
    c = a * b
    d = Tensor.empty(1, 16)
    e = c[0] * d
    check_schedule(e, 1)

  def test_expand_nofuse(self):
    a = Tensor.empty(1, 16)
    b = Tensor.empty(1, 16)
    c = a * b
    d = Tensor.empty(8192, 16)
    e = c * d
    check_schedule(e, 2)

  # this is the failing case in openpilot...it's very simple like this
  def test_image_conv_fusion(self):
    w1 = Tensor.empty(16, 16, 1, 1)
    b1 = Tensor.empty(16)
    w2 = Tensor.empty(16, 16, 1, 1)
    b2 = Tensor.empty(16)
    w3 = Tensor.empty(16, 16, 1, 1)
    b3 = Tensor.empty(16)

    x = Tensor.empty(1, 16, 32, 32)
    x = base = x.image_conv2d(w1, b1)
    x = x.image_conv2d(w2, b2) + base
    x = x.image_conv2d(w3, b3)

    # NOOP, 3 convs, contiguous
    with self.assertRaises(KernelCountException): check_schedule(x, 5)

  def test_image_conv_fusion_minimal(self):
    b1 = Tensor.empty(16)
    b2 = Tensor.empty(16)
    def p(x): return x.permute(1,0).contiguous().reshape(32,16,1).expand(32,16,16).sum(axis=2).permute(1,0)

    x = Tensor.empty(16, 32)
    x = base = p(x) + b1.reshape(16,1)
    x = p(x)
    x = x + b2.reshape(16,1)
    x = x + base
    del base
    x = p(x)
    check_schedule(x, 4)

  def test_image_conv_fusion_more_minimal(self):
    b1 = Tensor.empty(16)
    def p(x): return x.permute(1,0).contiguous().reshape(32,16,1).expand(32,16,16).sum(axis=2).permute(1,0)

    x = Tensor.empty(16, 32)
    x = base = p(x) + b1.reshape(16,1)
    x = p(x)
    del base
    check_schedule(x, 3)

  def test_resnet_block(self):
    with Tensor.train(False):
      in_planes, planes = 64, 64
      conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
      bn1 = nn.BatchNorm2d(planes)
      conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=1, bias=False)
      bn2 = nn.BatchNorm2d(planes)
      x = Tensor.empty(1, 64, 32, 32)
      out = bn1(conv1(x)).relu()
      out = bn2(conv2(out))
      out = (out + x).relu()
      run_schedule(check_schedule(out, 2, [conv1.weight, conv2.weight]))

  def test_contiguous_while_contiguous(self):
    x = Tensor.empty(1, 64, 32, 32)
    out = x.contiguous()
    check_schedule(out, 0, filter_sink=False)

  def test_contiguous_while_not_contiguous(self):
    x = Tensor.empty(1, 64, 32, 32)
    out = x.permute(0,2,3,1).contiguous()
    check_schedule(out, 1, filter_sink=False)

  def test_fold_with_contiguous(self):
    a = Tensor.randn(16, 16, 16).realize()
    b = Tensor.randn(16, 16).realize()
    c = (a.sum(2).contiguous() + b).contiguous()
    check_schedule(c, 2)

  def test_kernelize(self):
    a = Tensor.empty(10)
    b = Tensor.empty(10)
    c = (a+b).kernelize()
    d = c+2
    check_schedule(d, 2)

  def test_kernelize_view(self):
    a = Tensor.empty(4,1)
    b = a*2
    c = b.kernelize()+Tensor.empty(4,4)
    check_schedule(c, 2)

  def test_kernelize_diamond(self):
    a = Tensor([0]).realize()
    prev_a = (a+1).contiguous()
    a.assign(Tensor([2]))
    a.kernelize(prev_a)
    assert prev_a.uop in a.uop.src, "contiguous usage must run before assign"
    self.assertEqual((prev_a+a*3).item(), 1+2*3)

  def test_multioutput_ast(self):
    a = Tensor.zeros(1, dtype=dtypes.int).contiguous().realize().uop
    b = Tensor.zeros(1, dtype=dtypes.int).contiguous().realize().uop
    c = Tensor.arange(4).realize().uop
    kernel = UOp(Ops.KERNEL, src=(a, b, c.base), arg=Kernel(UOp.sink(c.r(Ops.ADD, (0,))+1, c.r(Ops.ADD, (0,))*2)))
    assert all(s.op is Ops.BUFFER for s in kernel.src), f"views are not allowed here {kernel}"
    run_schedule(check_schedule(UOp.sink(a.assign(kernel), b.assign(kernel)), 1))
    self.assertEqual(a.buffer.numpy(), [7])
    self.assertEqual(b.buffer.numpy(), [12])

  # unlike schedule, kernelize can be called multiple times on a Tensor
  def test_double_kerenlize(self):
    a = Tensor.empty(10)
    b = Tensor.empty(10)
    c = (a+b)
    d = c.kernelize()+2
    e = c.kernelize()+d.kernelize()
    check_schedule(e, 3)

  def test_kernelize_bw(self):
    a = Tensor.full((3,), 2.0, requires_grad=True).contiguous()
    b = Tensor.full((3,), 3.0, requires_grad=True).contiguous()
    x = (a*b).kernelize()
    y = Tensor.eye(3, requires_grad=True)
    z = y.matmul(x).sum()
    z.backward()
    self.assertEqual(z.item(), 18.0)
    self.assertEqual(z.grad.item(), 1.0)

  def test_kernelize_bw_view(self):
    a = Tensor.full((3,1), 2.0, requires_grad=True).contiguous()
    b = Tensor.full((3,1), 3.0, requires_grad=True).contiguous()
    x = (a*b).kernelize()
    y = Tensor.eye(6, requires_grad=True)
    z = y.matmul(x.expand(3,2).reshape(6)).sum()
    z.backward()
    self.assertEqual(z.item(), 36.0)
    self.assertEqual(z.grad.item(), 1.0)

  @unittest.skip("no longer supported")
  def test_double_from(self):
    x = Tensor([1,2,3,4])
    out = x.to('python')
    check_schedule(out, 0, filter_sink=False)

  def _alu_from_tensor(self, t:Tensor):
    s = [s for s in t.schedule() if s.ast.op is Ops.SINK]
    self.assertEqual(len(s), 1)
    return [u.op for u in s[0].ast.toposort() if u.op in GroupOp.ALU]

  def test_2_pow_is_exp2(self):
    t = 2.0 ** Tensor([1.0, 2.0, 3.0])
    self.assertEqual(self._alu_from_tensor(t), [Ops.EXP2])

  def test_pow_05_is_sqrt(self):
    t = Tensor([1.0, 2.0, 3.0]) ** 0.5
    self.assertEqual(self._alu_from_tensor(t), [Ops.SQRT])

  def test_pow_neg_05_is_rsqrt(self):
    t = Tensor([1.0, 2.0, 3.0]) ** -0.5
    self.assertEqual(self._alu_from_tensor(t), [Ops.RECIP, Ops.SQRT])

  def test_pow_2_has_1_mul(self):
    t = Tensor([1.0, 2.0, 3.0]) ** Tensor(2.0)
    self.assertEqual(self._alu_from_tensor(t), [Ops.MUL])

  def test_pow_8_has_3_muls(self):
    t = Tensor([1.0, 2.0, 3.0]) ** 8
    self.assertEqual(self._alu_from_tensor(t), [Ops.MUL, Ops.MUL, Ops.MUL])

  def test_pow_const_tensor_to_zero(self):
    x = Tensor([1,2,3,4])
    out = x ** Tensor(0.0)
    # NOTE: this is UOp.const(0) + UOp.const(1)
    check_schedule(out, 0)

  def test_zero_size(self):
    x = Tensor.empty(2, 3, 0)
    out = x + 1
    check_schedule(out, 0, filter_sink=False)

  def test_reduce_permute_nofuse(self):
    x = Tensor.empty(32, 32, 32)
    y = Tensor.empty(32, 32)
    out = x.sum(axis=2).T+y
    check_schedule(out, 2)

  def test_two_elus_sum(self):
    x = Tensor.empty(32, 32)
    y = Tensor.empty(32, 32)
    out = x.sum(1).relu().elu() + y.sum(1).relu().elu()
    check_schedule(out, 2)

  @unittest.skipUnless(SPLIT_REDUCEOP, "Testing split reducop requires SPLIT_REDUCEOP")
  def test_preserve_multistage_reduce(self):
    big_enough = getenv("REDUCEOP_SPLIT_THRESHOLD", 32768)
    x = Tensor.randn(big_enough).realize()
    out = (x - x.max(keepdim=True)).max()
    run_schedule(check_schedule(out, 4))
    np.testing.assert_allclose(out.numpy(), (x.numpy() - x.numpy().max(keepdims=True)).max())

  def test_multistage_reduce(self):
    x = Tensor.empty(32, 32, 32)
    out = x.sum(2).relu().sum(1)
    check_schedule(out, 2)

  def test_multistage_reduce_fork(self):
    x = Tensor.empty(32, 32, 32)
    x = x.sum(2)
    out2 = x + 1
    out = x.relu().sum(1) + out2[0]
    check_schedule(out, 2)

  @unittest.skip("these two Tensors are the same")
  def test_example_matmul(self):
    x = Tensor.eye(64, requires_grad=True)
    y = Tensor.eye(64, requires_grad=True)
    z = y.matmul(x).sum()
    z.backward()
    out = x.grad.contiguous()
    run_schedule(check_schedule(out, 2))
    np.testing.assert_allclose(out.numpy(), np.ones((64,64)))

  def test_example_matmul_contig(self):
    x = Tensor.eye(64, requires_grad=True).contiguous().realize()
    y = Tensor.eye(64, requires_grad=True).contiguous().realize()
    z = y.matmul(x).sum()
    z.backward()
    out = x.grad.contiguous()
    run_schedule(check_schedule(out, 2))
    np.testing.assert_allclose(out.numpy(), np.ones((64,64)))

  def test_example_matmul_same(self):
    x = Tensor.eye(64, requires_grad=True)
    z = x.matmul(x).sum()
    z.backward()
    out = x.grad.contiguous()
    run_schedule(check_schedule(out, 2))
    # NOTE: the gradient flows twice
    np.testing.assert_allclose(out.numpy(), 2*np.ones((64,64)))

  def test_contiguous_add(self):
    x = Tensor.empty(32)
    y = Tensor.empty(32)
    z = Tensor.empty(32)
    out = (x+y).contiguous()+z
    check_schedule(out, 2)

  def test_double_sum_ref(self):
    x = Tensor.empty(32, 32, 32)
    x = x.sum(2)
    out = x + x[:, 4]
    check_schedule(out, 2)

  def test_reduce_shrink(self):
    x = Tensor.empty(32, 32)
    y = Tensor.empty(16)
    x = x.sum(1)
    x = x[:16]
    out = x + y
    # NOTE: this could be 1 kernel if we mask the store?
    check_schedule(out, 2)

  def test_multireduce_shrink(self):
    Tensor.manual_seed(0)
    a = Tensor.randn(32, 32).realize()
    b = Tensor.randn(32, 32).realize()
    c = Tensor.randn(16).realize()
    a_out = a.sum(1)
    a_out = a_out[:16]
    b_out = b.sum(1)
    b_out = b_out[:16]
    out = a_out + b_out + c
    # run_schedule(check_schedule(out, 2))  # TODO: this should be 1 (can we make it 1 with the new linearizer?)
    run_schedule(check_schedule(out, 3))
    np.testing.assert_allclose(out.numpy(), a.numpy().sum(axis=1)[:16] + b.numpy().sum(axis=1)[:16] + c.numpy(), atol=1e-4, rtol=1e-4)

  # broken due to const folding and two contiguous are different kernels
  # NOTE: passes after delete_lazy
  def test_const_no_recompute(self):
    x = Tensor(2) + Tensor(2)
    y = Tensor(2) + Tensor(2)
    out = x.contiguous() + y.contiguous()
    check_schedule(out, 2, filter_sink=False)

  @unittest.expectedFailure
  def test_reduce_same_size(self):
    Tensor.manual_seed(0)
    a = Tensor.randn(4, 4).realize()
    out0 = a.sum() + 2
    out1 = a.sum() + 4
    out2 = out0 * out1
    run_schedule(check_schedule([out0, out1, out2], 1))
    np.testing.assert_allclose(out0.numpy(), out0_np:=a.numpy().sum()+2, atol=1e-4, rtol=1e-6)
    np.testing.assert_allclose(out1.numpy(), out1_np:=a.numpy().sum()+4, atol=1e-4, rtol=1e-6)
    np.testing.assert_allclose(out2.numpy(), out0_np*out1_np, atol=1e-4, rtol=1e-6)

  @unittest.expectedFailure
  def test_reduce_multiple_paths(self):
    Tensor.manual_seed(0)
    a = Tensor.randn(4, 4).realize()
    out0 = a.sum().exp2()
    # out1 has two paths to a.sum()
    out1 = a.sum() + out0
    run_schedule(check_schedule([out0, out1], 1))
    np.testing.assert_allclose(out0.numpy(), out0_np:=np.exp2(a.numpy().sum()), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(out1.numpy(), a.numpy().sum()+out0_np, atol=1e-4, rtol=1e-6)

  def test_multireduce_reduce_multiple_paths(self):
    Tensor.manual_seed(0)
    a = Tensor.randn(4, 4).realize()
    out0 = a.sum().exp2()
    out1 = a.sum() + out0
    b = (a + out0 + out1)
    out2 = b.sum().exp2()
    out3 = b.sum() + out2
    # run_schedule(check_schedule([out0, out1, out2, out3], 1))
    run_schedule(check_schedule([out0, out1, out2, out3], 6))
    np.testing.assert_allclose(out0.numpy(), np_out0:=np.exp2(a.numpy().sum()), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(out1.numpy(), np_out1:=a.numpy().sum()+np_out0, atol=1e-4, rtol=1e-4)
    np_b = (a.numpy() + np_out0 + np_out1)
    np.testing.assert_allclose(out2.numpy(), np_out2:=np.exp2(np_b.sum()), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(out3.numpy(), np_b.sum()+np_out2, atol=1e-4, rtol=1e-4)

  def test_reduce_ext_reduce_child(self):
    Tensor.manual_seed(0)
    a = Tensor.randn(4, 4).realize()
    b = Tensor.randn(4, 4).realize()
    # b.sum() is not a descendant of the fused nodes
    out0 = a.sum() + b.sum() + 2
    out1 = a.sum() + b.sum() + 4
    # run_schedule(check_schedule([out0, out1], 1))
    run_schedule(check_schedule([out0, out1], 4))
    np.testing.assert_allclose(out0.numpy(), a.numpy().sum()+b.numpy().sum()+2, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(out1.numpy(), a.numpy().sum()+b.numpy().sum()+4, atol=1e-4, rtol=1e-4)

  def test_reduce_multiple_paths_midreduce(self):
    Tensor.manual_seed(0)
    a = Tensor.randn(4, 4).realize()
    r = a.sum()
    out0 = r.exp2()
    # reduce node in the indirect path from r to out2
    out1 = (a - out0).max()
    out2 = r + out1
    # run_schedule(check_schedule([r, out0, out1, out2], 1))
    run_schedule(check_schedule([r, out0, out1, out2], 4))
    np.testing.assert_allclose(r.numpy(), r_np:=a.numpy().sum(), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(out0.numpy(), out0_np:=np.exp2(r_np), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(out1.numpy(), out1_np:=(a.numpy() - out0_np).max(), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(out2.numpy(), r_np + out1_np, atol=1e-4, rtol=1e-4)

  def test_reduce_multiple_paths_midreduce_fused(self):
    Tensor.manual_seed(0)
    a = Tensor.randn(4, 4).realize()
    b = Tensor.randn(4, 4).realize()
    out0 = a.sum() + 4
    out1 = b.max() + out0*2
    out2 = a.sum() + out1
    # run_schedule(check_schedule([out0, out1, out2], 1))
    run_schedule(check_schedule([out0, out1, out2], 4))
    np.testing.assert_allclose(out0.numpy(), out0_np:=a.numpy().sum()+4, atol=1e-4, rtol=1e-6)
    np.testing.assert_allclose(out1.numpy(), out1_np:=b.numpy().max() + out0_np*2, atol=1e-4, rtol=1e-6)
    np.testing.assert_allclose(out2.numpy(), a.numpy().sum() + out1_np, atol=1e-4, rtol=1e-6)

  def test_reduce_multiple_paths_midexpand(self):
    Tensor.manual_seed(0)
    a = Tensor.randn(4, 4).realize()
    b = Tensor.randn(4, 4, 4).realize()
    r = a.sum()
    out0 = r.exp2()
    # e1 is in the indirect path from a.sum() to out1
    e = b + out0
    out1 = r + e[0][0][0]
    # run_schedule(check_schedule([r, out0, out1, e], 3)) # 1 or 2 or 3? should be 1 (one reduce) but the different outputs might make it 3
    run_schedule(check_schedule([r, out0, out1, e], 4))
    np.testing.assert_allclose(r.numpy(), r_np:=a.numpy().sum(), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(out0.numpy(), out0_np:=np.exp2(r_np), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(e.numpy(), e_np:=b.numpy() + out0_np, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(out1.numpy(), r_np + e_np[0][0][0], atol=1e-4, rtol=1e-4)

  # changed by multireduce
  def test_reduce_expand_child(self):
    Tensor.manual_seed(0)
    a = Tensor.randn((32, 32, 32)).realize()
    b = Tensor.randn((1, 16)).realize()
    out0 = a.sum() + 2
    out1 = a.sum() + b
    # run_schedule(check_schedule([out0, out1], 2))
    run_schedule(check_schedule([out0, out1], 4))
    np.testing.assert_allclose(out0.numpy(), a.numpy().sum()+2, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(out1.numpy(), a.numpy().sum()+b.numpy(), atol=1e-4, rtol=1e-4)

  @unittest.expectedFailure
  def test_reduce_shrink_child(self):
    a = Tensor.empty(100, 100)
    b = Tensor.empty(10,)
    c = a.sum() + b[0]
    d = a.sum() + 2
    check_schedule([c, d], 1)

  def test_reduce_multiple_paths_midshrink(self):
    a = Tensor.empty(4, 4)
    r = a.sum(axis=1)
    out0 = r.exp2()
    out1 = out0[0] + out0
    check_schedule([r, out0, out1], 3)

  def test_reduce_shrink_output(self):
    a = Tensor.empty(4, 4)
    r = a.sum(keepdim=True)
    out0 = r.exp2()
    out1 = out0[0] + Tensor.empty(1, )
    check_schedule([r, out0, out1], 3)

  def test_std_multireduce_fusion(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(4, 32).realize()
    out = x.std(-1)
    run_schedule(check_schedule(out, 2))
    np.testing.assert_allclose(out.numpy(), x.numpy().std(axis=-1, ddof=1), atol=1e-4, rtol=1e-4)

  def test_argmin_multireduce_fusion(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(4, 32).realize()
    out = x.argmin(-1)
    run_schedule(check_schedule(out, 3))
    np.testing.assert_equal(out.numpy(), x.numpy().argmin(axis=-1))

  def test_argmax_multireduce_fusion(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(4, 32).realize()
    out = x.argmax(-1)
    run_schedule(check_schedule(out, 3))
    np.testing.assert_equal(out.numpy(), x.numpy().argmax(axis=-1))

  def test_scaled_dot_product_attention_multireduce_fusion(self):
    Tensor.manual_seed(0)
    q = Tensor.randn(32,8,16,8).realize()
    k = Tensor.randn(32,8,16,8).realize()
    v = Tensor.randn(32,8,16,8).realize()
    out = Tensor.scaled_dot_product_attention(q,k,v)
    run_schedule(check_schedule(out, 5))
    if getenv("CHECK", 1):
      import torch
      compare = torch.nn.functional.scaled_dot_product_attention(torch.tensor(q.numpy()),torch.tensor(k.numpy()),torch.tensor(v.numpy()))
      np.testing.assert_allclose(out.numpy(), compare.numpy(), atol=1e-6, rtol=1e-3)

  def test_ugly_reduceop_pairing(self):
    Tensor.manual_seed(0)
    a = Tensor.randn(4, 32).realize()
    b = Tensor.randn(4, 32).realize()
    c = Tensor.randn(4, 32).realize()
    out = (c * a.sum(-1, keepdim=True)).sum(-1) + (b * a.sum(-1, keepdim=True)).sum(-1) # a.sum has >1 children but should still fuse
    # run_schedule(check_schedule(out, 1))
    run_schedule(check_schedule(out, 3))
    np.testing.assert_allclose(out.numpy(), \
      (c.numpy()*a.numpy().sum(axis=-1,keepdims=True)).sum(-1) + (b.numpy()*a.numpy().sum(axis=-1,keepdims=True)).sum(-1), atol=1e-4, rtol=1e-4)

  def test_reduce_expand_reduce_fusion(self):
    Tensor.manual_seed(0)
    a = Tensor.randn(4, 32).realize()
    out = (a+a.sum(-1, keepdim=True)).sum(-1)
    # run_schedule(check_schedule(out, 1))
    run_schedule(check_schedule(out, 2))
    np.testing.assert_allclose(out.numpy(), (a.numpy()+a.numpy().sum(axis=-1,keepdims=True)).sum(axis=-1), atol=1e-4, rtol=1e-4)

  def test_reduce_expand_reduce_expand_fusion(self):
    Tensor.manual_seed(0)
    a = Tensor.randn(4, 32).realize()
    out = a+(a+a.sum(-1,keepdim=True)).sum(-1, keepdim=True)
    # run_schedule(check_schedule(out, 2))
    run_schedule(check_schedule(out, 3))
    np.testing.assert_allclose(out.numpy(), \
      a.numpy()+(a.numpy()+a.numpy().sum(axis=-1,keepdims=True)).sum(axis=-1,keepdims=True), atol=1e-4, rtol=1e-4)

  def test_branching_reduces_and_expands_fusion(self):
    Tensor.manual_seed(0)
    a = Tensor.randn(4, 32).realize()
    out0 = a+a.sum(-1, keepdim=True)
    out1 = out0.sum(-1)
    # run_schedule(check_schedule(out, 2))
    run_schedule(check_schedule([out0, out1], 3))
    np.testing.assert_allclose(out0.numpy(), a.numpy()+a.numpy().sum(axis=-1,keepdims=True), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(out1.numpy(), (a.numpy()+a.numpy().sum(axis=-1,keepdims=True)).sum(axis=-1), atol=1e-4, rtol=1e-4)

  def test_multireduce_fusion_simple_sequential(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(4, 32).realize()
    y = Tensor.randn(4, 32).realize()
    out = (y + x.sum(axis=-1, keepdim=True)).sum(axis=-1)
    # run_schedule(check_schedule(out, 1))
    run_schedule(check_schedule(out, 2))
    np.testing.assert_allclose(out.numpy(), (y.numpy() + x.numpy().sum(axis=-1, keepdims=True)).sum(axis=-1), atol=1e-4, rtol=1e-4)

  def test_multireduce_fusion_simple_parallel(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(4, 32).realize()
    y = Tensor.randn(4, 32).realize()
    out = y.sum(axis=-1) + x.sum(axis=-1)
    # run_schedule(check_schedule(out, 1))
    run_schedule(check_schedule(out, 2))
    np.testing.assert_allclose(out.numpy(), y.numpy().sum(axis=-1) + x.numpy().sum(axis=-1), atol=1e-4, rtol=1e-4)

  def test_multireduce_fusion_sequential(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(4, 32).realize()
    out = x.std(-1)
    # run_schedule(check_schedule(out, 1))
    run_schedule(check_schedule(out, 2))
    np.testing.assert_allclose(out.numpy(), x.numpy().std(axis=-1, ddof=1), atol=1e-4, rtol=1e-4)

  def test_multireduce_fusion_parallel(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(4, 32).realize()
    y = Tensor.randn(4, 32).realize()
    out = x.std(-1) + y.std(-1)
    # run_schedule(check_schedule(out, 1))
    run_schedule(check_schedule(out, 4))
    np.testing.assert_allclose(out.numpy(), x.numpy().std(axis=-1, ddof=1) + y.numpy().std(axis=-1, ddof=1), atol=1e-4, rtol=1e-4)

  def test_multireduce_diffops_sequential(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(4, 32).realize()
    out = (x - x.max(-1, keepdim=True)).sum(-1)
    # run_schedule(check_schedule(out, 1))
    run_schedule(check_schedule(out, 2))
    np.testing.assert_allclose(out.numpy(), (x.numpy() - x.numpy().max(axis=-1, keepdims=True)).sum(axis=-1), atol=1e-4, rtol=1e-4)

  def test_multireduce_fusion_diffops_parallel(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(4, 32).realize()
    y = Tensor.randn(4, 32).realize()
    out = x.sum(-1) + y.max(-1)
    # run_schedule(check_schedule(out, 1))
    run_schedule(check_schedule(out, 2))
    np.testing.assert_allclose(out.numpy(), x.numpy().sum(axis=-1) + y.numpy().max(axis=-1), atol=1e-4, rtol=1e-4)

  def test_multireduce_fusion_sequential_and_parallel(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(4, 32).realize()
    y = Tensor.randn(4, 32).realize()
    mu = (x - x.max(axis=-1, keepdim=True)).mean(axis=-1, keepdim=True) + (y - y.max(axis=-1, keepdim=True)).mean(axis=-1, keepdim=True)
    out = [((x - mu).square().sum(-1)/x.shape[-1]).sqrt(), ((y - mu).square().sum(-1)/y.shape[-1]).sqrt()]
    np_mu = (x.numpy() - x.numpy().max(axis=-1, keepdims=True)).mean(axis=-1, keepdims=True) + \
      (y.numpy() - y.numpy().max(axis=-1, keepdims=True)).mean(axis=-1, keepdims=True)
    # run_schedule(check_schedule(out, 1))
    run_schedule(check_schedule(out, 6))
    np.testing.assert_allclose(out[0].numpy(), np.sqrt(np.square(x.numpy() - np_mu).sum(-1)/x.shape[-1]), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(out[1].numpy(), np.sqrt(np.square(y.numpy() - np_mu).sum(-1)/y.shape[-1]), atol=1e-4, rtol=1e-4)

  def test_multimatmul_fusion(self):
    Tensor.manual_seed(0)
    a,b = Tensor.randn(4, 64).realize(), Tensor.rand(64,8).realize()
    c,d = Tensor.randn(4, 64).realize(), Tensor.rand(64,8).realize()
    out = a@b + c@d
    # run_schedule(check_schedule(out, 1))
    run_schedule(check_schedule(out, 2))
    np.testing.assert_allclose(out.numpy(), a.numpy()@b.numpy() + c.numpy()@d.numpy(), atol=1e-4, rtol=1e-4)

  def test_softmax_fusion(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(4, 12, 64, 64).realize()
    out = x.softmax()
    run_schedule(check_schedule(out, 3))
    expected = (x_exp:=np.exp(x.numpy()-x.numpy().max(-1, keepdims=True)))/x_exp.sum(-1, keepdims=True)
    np.testing.assert_allclose(out.numpy(), expected, atol=1e-4, rtol=1e-4)

  @unittest.skipUnless(is_dtype_supported(dtypes.half), "need half")
  def test_softmax_upcast(self):
    # input half, softmax in float
    Tensor.manual_seed(0)
    x = Tensor.randn(4, 12, 64, 64, dtype=dtypes.half).realize()
    out = x.softmax(dtype=dtypes.float)
    sched = out.schedule()
    self.assertEqual(len(sched), 3)
    self.assertEqual(sched[0].bufs[0].dtype, dtypes.half)

    # input float, softmax in float
    Tensor.manual_seed(0)
    x = Tensor.randn(4, 12, 64, 64, dtype=dtypes.float).realize()
    out = x.softmax(dtype=dtypes.float)
    sched = out.schedule()
    self.assertEqual(len(sched), 3)
    self.assertEqual(sched[0].bufs[0].dtype, dtypes.float)

  def test_softmax_backward(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(4, 12, 64, 64, requires_grad=True).realize()
    x.softmax().sum().backward()
    run_schedule(check_schedule(x.grad, 4))

  # changed by: multireduce spec
  def test_layernorm_onelayer_fusion(self):
    Tensor.manual_seed(0)
    layer = nn.LayerNorm([10, 10])
    layer.weight = Tensor.randn(10,10).realize()
    layer.bias = Tensor.randn(10,10).realize()
    x = Tensor.randn(20, 5, 10, 10).realize()
    out = layer(x)
    # run_schedule(check_schedule(out, 2))
    run_schedule(check_schedule(out, 3))
    y = (x.numpy() - x.numpy().mean(layer.axis, keepdims=True))
    expected = y / np.sqrt((y*y).mean(layer.axis, keepdims=True) + layer.eps)
    np.testing.assert_allclose(out.numpy(), expected * layer.weight.numpy() + layer.bias.numpy(), atol=1e-4, rtol=1e-4)

  def test_scaled_dot_product_attention_fusion(self):
    x, y, z, m = (Tensor.empty(32, 8, 16, 16) for _ in range(4))
    out = Tensor.scaled_dot_product_attention(x, y, z, attn_mask=m)
    check_schedule(out, 5)

  def test_scaled_dot_product_attention_causal_fusion(self):
    x, y, z = (Tensor.empty(32, 8, 16, 16) for _ in range(3))
    out = Tensor.scaled_dot_product_attention(x, y, z, is_causal=True)
    check_schedule(out, 5)

  def test_adam_step_fusion(self):
    with Tensor.train():
      x = Tensor.empty(4, 64, 32)
      layer = nn.Linear(32, 32*4)
      _realize_weights(layer)
      opt = nn.optim.Adam(nn.state.get_parameters(layer), lr=1e-4)
      layer(x).relu().sum().backward()
      check_schedule(opt.schedule_step(), 16)

  def test_adam_conv_fuse(self):
    with Tensor.train():
      img = Tensor.empty(2,3,4,4)
      c1 = nn.Conv2d(3,32,3)
      _realize_weights(c1)
      opt = nn.optim.Adam(nn.state.get_parameters(c1), lr=1e-4)
      opt.zero_grad()
      c1(img).relu().sum().backward()
      check_schedule(opt.schedule_step(), 16)

  def test_adam_2convs_fuse(self):
    with Tensor.train():
      img = Tensor.empty(2,3,4,4)
      c1 = nn.Conv2d(3,16,3,bias=False)
      c2 = nn.Conv2d(16,32,2,bias=False)
      _realize_weights([c1, c2])
      opt = nn.optim.Adam(nn.state.get_parameters([c1, c2]), lr=1e-4)
      opt.zero_grad()
      c2(c1(img).relu()).relu().sum().backward()
      check_schedule(opt.schedule_step(), 20)

  def test_sgd_conv_fuse(self):
    with Tensor.train():
      img = Tensor.empty(2,3,4,4)
      c1 = nn.Conv2d(3,32,3)
      _realize_weights(c1)
      opt = nn.optim.SGD(nn.state.get_parameters(c1))
      opt.zero_grad()
      c1(img).relu().sum().backward()
      check_schedule(opt.schedule_step(), 3)

  def test_sgd_2convs_fuse(self):
    with Tensor.train():
      img = Tensor.empty(2,3,4,4)
      c1 = nn.Conv2d(3,16,3,bias=False)
      c2 = nn.Conv2d(16,32,2,bias=False)
      _realize_weights([c1, c2])
      opt = nn.optim.SGD(nn.state.get_parameters([c1, c2]))
      opt.zero_grad()
      c2(c1(img).relu()).relu().sum().backward()
      check_schedule(opt.schedule_step(), 7)

  @unittest.skipUnless(is_dtype_supported(dtypes.ulong), "Needs ulong")
  def test_fold_2convs_sgd_nesterov_momentum_wd(self):
    with Tensor.train():
      img = Tensor.empty(2,3,4,4)
      c1 = nn.Conv2d(3,16,3,bias=False)
      c2 = nn.Conv2d(16,32,2,bias=False)
      _realize_weights([c1, c2])
      opt = nn.optim.SGD(nn.state.get_parameters([c1, c2]), nesterov=True, momentum=0.9, weight_decay=0.1)
      opt.zero_grad()
      c2(c1(img).relu()).relu().sum().backward()
      check_schedule(opt.schedule_step(), 13)

  def test_sgd_4convs_fuse(self):
    with Tensor.train():
      img = Tensor.empty(2,3,16,16)
      c1 = nn.Conv2d(3,4,3,bias=False)
      c2 = nn.Conv2d(4,8,3,bias=False)
      c3 = nn.Conv2d(8,16,3,bias=False)
      c4 = nn.Conv2d(16,32,3,bias=False)
      _realize_weights([c1, c2, c3, c4])
      opt = nn.optim.SGD(nn.state.get_parameters([c1, c2, c3, c4]))
      opt.zero_grad()
      c4(c3(c2(c1(img).relu()).relu()).relu()).relu().sum().backward()
      check_schedule(opt.schedule_step(), 17)

  def test_sgd_4convs_fuse_conv_bw(self):
    with Tensor.train():
      img = Tensor.empty(2,3,16,16)
      c1 = nn.Conv2d(3,4,3,bias=False)
      c2 = nn.Conv2d(4,8,3,bias=False)
      c3 = nn.Conv2d(8,16,3,bias=False)
      c4 = nn.Conv2d(16,32,3,bias=False)
      _realize_weights([c1, c2, c3, c4])
      opt = nn.optim.SGD(nn.state.get_parameters([c1, c2, c3, c4]))
      opt.zero_grad()
      c4(c3(c2(c1(img).relu()).relu()).relu()).relu().sum().backward()
      with Context(FUSE_CONV_BW=1): check_schedule(opt.schedule_step(), 14)

  @unittest.skipUnless(is_dtype_supported(dtypes.half), "need half")
  def test_prefer_half_buffer(self):
    x = Tensor.ones(4).contiguous().realize()
    # y = Tensor.ones(4).contiguous().realize()
    z = Tensor.ones(4, 4).contiguous().realize()

    # should not create extra kernel if output will be realized anyways
    dummy = x.sum().half().float()
    check_schedule(dummy, 1)
    dummy = x.sum().half().float().contiguous() + 1
    check_schedule(dummy, 2)

    # shared between two outputs
    shared = x.sum().half().float()
    a = shared * 2
    b = shared * 3
    sched = check_schedule([a, b], 3)
    # store reduceop in half
    self.assertEqual(sched[0].bufs[0].dtype, dtypes.half)
    # fuse cast with the child kernel
    self.assertEqual(sched[1].bufs[0].dtype, dtypes.float)
    self.assertEqual(sched[2].bufs[0].dtype, dtypes.float)

    # reduce
    a = z.sum(axis=0).half().float().sum(axis=0)
    sched = check_schedule(a, 2)
    self.assertEqual(sched[0].bufs[0].dtype, dtypes.half)
    self.assertEqual(sched[1].bufs[0].dtype, dtypes.float)

    # expand
    # expand will realize just after the .float(), so requires change to realize-before-expand
    # normal = (x.sum().half().float().reshape(1) * y).sum()
    # sched = check_schedule(normal, 2)
    # for si in sched[:-1]: assert all(out.dtype == dtypes.half for out in si.outputs[:-1])

    # parallel reduce
    # a = x.sum().half().float() * y.sum().half().float()
    # b = a + 1
    # c = a + 2
    # sched = check_schedule([b, c], 4)
    # doesn't store either in half because it doesn't chase

  def test_reduce_simple_chase(self):
    a = Tensor.empty(4, 4, 4)
    r = a.sum(0) + 6
    b = r.sum(0) * 4
    c = r.sum(1) * 2
    schedule = check_schedule([b, c], 3)
    self.assertIs(store_val(schedule[0]).op, Ops.ADD)

  def test_multireduce_simple_chase(self):
    Tensor.manual_seed(0)
    a = Tensor.randn(4, 4, 4).realize()
    r = (a + (a.sum(0, keepdim=True) + 6)).sum(0) * 2
    b = r.sum(0) + 8
    c = r.sum(1) + 12
    np_r = (a.numpy() + (a.numpy().sum(0) + 6)).sum(0) * 2
    # schedule = check_schedule([b,c], 3)
    # self.assertIs(schedule[0].ast[0].src[0].arg, Ops.MUL)
    schedule = check_schedule([b,c], 4)
    run_schedule(schedule)
    np.testing.assert_allclose(b.numpy(), np_r.sum(0) + 8, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(c.numpy(), np_r.sum(1) + 12, atol=1e-4, rtol=1e-4)

  def test_push_permute_chase(self):
    a = Tensor.empty(4, 4, 4)
    b = Tensor.empty(4, 4)
    r = a.sum(2) + b
    d = r.T * 4
    e = r * d
    schedule = check_schedule([d, e], 3)
    self.assertIs(store_val(schedule[0]).op, Ops.ADD)

  def test_multireduce_push_permute_chase(self):
    Tensor.manual_seed(0)
    a = Tensor.randn(4, 4, 4).realize()
    b = Tensor.randn(4, 4).realize()
    r = a.sum(2) + b
    d = r.T * 4
    e = r * (d + a).sum(2)
    schedule = check_schedule([d, e], 3) # make sure it doesn't fuse
    self.assertIs(store_val(schedule[0]).op, Ops.ADD)
    run_schedule(schedule)
    np.testing.assert_allclose(d.numpy(), (a.numpy().sum(2) + b.numpy()).T * 4, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(e.numpy(), (a.numpy().sum(2) + b.numpy()) * (d.numpy() + a.numpy()).sum(2), atol=1e-4, rtol=1e-4)

  def test_push_shrink_chase(self):
    a = Tensor.empty(16, 16)
    b = Tensor.empty(4)
    c = Tensor.empty(16, )
    r = a.sum(1) + c
    d = r[:4] * b
    schedule = check_schedule(d, 2)
    self.assertIs(store_val(schedule[0]).op, Ops.ADD)

  def test_multireduce_push_shrink_chase(self):
    Tensor.manual_seed(0)
    a = Tensor.randn(16, 16).realize()
    b = Tensor.randn(4).realize()
    c = Tensor.randn(16, ).realize()
    d = Tensor.randn(16, 16).realize()
    r = a.sum(1) + c
    out = r[:4] * b + d.sum(1)[:4]
    # schedule = check_schedule(out, 2)
    schedule = check_schedule(out, 3)
    self.assertIs(store_val(schedule[0]).op, Ops.ADD)
    run_schedule(schedule)
    np.testing.assert_allclose(out.numpy(), (a.numpy().sum(1) + c.numpy())[:4] * b.numpy() + d.numpy().sum(1)[:4], atol=1e-4, rtol=1e-4)

  def test_midreduce_nochase(self):
    a = Tensor.empty(16, 16)
    b = (a.sum(0) + a.max(1)) + 2
    schedule = check_schedule(b, 2)
    self.assertIs(store_val(schedule[0]).op, Ops.REDUCE_AXIS)

  def test_multireduce_midreduce_nochase(self):
    Tensor.manual_seed(0)
    a = Tensor.randn(16, 16).realize()
    b = (a.sum(0)+a.max(0) + a.max(1)+a.sum(1)) + 2
    # schedule = check_schedule(b, 2)
    schedule = check_schedule(b, 4)
    self.assertIs(store_val(schedule[0]).op, Ops.REDUCE_AXIS)
    run_schedule(schedule)
    np.testing.assert_allclose(b.numpy(), a.numpy().sum(0)+a.numpy().max(0) + a.numpy().max(1)+a.numpy().sum(1)+2, atol=1e-4, rtol=1e-4)

  # changed by: multireduce spec
  # pattern in test_transformer
  def test_partial_fuse1(self):
    Tensor.manual_seed(0)
    a = Tensor.randn(16, 16).realize()
    b = Tensor.randn(16, 16).realize()
    c = a.sum() + 2
    d = (a.sum() - b.sum()) * 4
    # run_schedule(check_schedule([c, d], 1))
    run_schedule(check_schedule([c, d], 3))
    np.testing.assert_allclose(c.numpy(), a.numpy().sum()+2, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(d.numpy(), (a.numpy().sum() - b.numpy().sum()) * 4, atol=1e-4, rtol=1e-4)

  # changed by: multireduce spec
  # pattern in conv
  def test_partial_fuse2(self):
    Tensor.manual_seed(0)
    a = Tensor.randn(16, 16).realize()
    b = Tensor.randn(16, 16).realize()
    c = a.sum() + 2
    d = b.sum() - c
    # run_schedule(check_schedule([c, d], 1))
    run_schedule(check_schedule([c, d], 2))
    np.testing.assert_allclose(c.numpy(), a.numpy().sum()+2, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(d.numpy(), b.numpy().sum()-(a.numpy().sum()+2), atol=1e-4, rtol=1e-4)

  # changed by: multireduce spec
  # pattern in adam
  @unittest.expectedFailure
  def test_partial_fuse3(self):
    Tensor.manual_seed(0)
    a = Tensor.randn(16, 16).realize()
    b = Tensor.randn(16, 16).realize()
    c = a.sum() + 2
    d = a.sum() * 2
    e = c * d
    f = b.sum() - e
    # run_schedule(check_schedule([c, d, e, f], 1))
    run_schedule(check_schedule([c, d, e, f], 2))
    np.testing.assert_allclose(c.numpy(), c_np:=a.numpy().sum()+2, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(d.numpy(), d_np:=a.numpy().sum()*2, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(e.numpy(), e_np:=c_np*d_np, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(f.numpy(), b.numpy().sum() - e_np, atol=1e-4, rtol=1e-4)

  # changed by: multireduce spec
  @unittest.expectedFailure
  def test_partial_fuse4(self):
    Tensor.manual_seed(0)
    a = Tensor.randn(16, 16).realize()
    b = Tensor.randn(16, 16).realize()
    c = a.sum() + 2
    d = a.sum() * 2
    e = c * d
    f = (b - d).sum() - e
    # run_schedule(check_schedule([c, d, e, f], 1))
    run_schedule(check_schedule([c, d, e, f], 3))
    np.testing.assert_allclose(c.numpy(), c_np:=a.numpy().sum()+2, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(d.numpy(), d_np:=a.numpy().sum()*2, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(e.numpy(), e_np:=c_np*d_np, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(f.numpy(), (b.numpy()-d_np).sum()-e_np, atol=1e-4, rtol=1e-4)

  def test_pad_reduce_safe(self):
    Tensor.manual_seed(0)
    a = Tensor.rand(3, 4, 5).realize()
    b = Tensor.rand(3, 4, 5).realize()
    out = (a + b).pad(((0, 1), (0, 1), (0, 1)), value=1.0).sum().contiguous()
    run_schedule(check_schedule(out, 1))
    np.testing.assert_allclose(out.numpy(), np.pad(a.numpy()+b.numpy(), ((0, 1), (0, 1), (0, 1)), constant_values=1.0).sum(), atol=1e-5, rtol=1e-6)

  def test_multireduce_pad_reduce_safe(self):
    Tensor.manual_seed(0)
    a = Tensor.randn(3, 4, 5).realize()
    b = Tensor.randn(3, 4, 5).realize()
    out = (a.pad(((0, 1), (0, 1), (0, 1)), value=1.0).sum(keepdim=True)+b.pad(((0, 1), (0, 1), (0, 1)), value=1.0).sum()).contiguous()
    # run_schedule(check_schedule(out, 1))
    run_schedule(check_schedule(out, 2))
    np.testing.assert_allclose(out.numpy(), np.pad(a.numpy(), ((0, 1), (0, 1), (0, 1)), constant_values=1.0).sum(keepdims=True) + \
                                                   np.pad(b.numpy(), ((0, 1), (0, 1), (0, 1)), constant_values=1.0).sum(), atol=1e-4, rtol=1e-4)

  def test_pad_reduce_unsafe(self):
    Tensor.manual_seed(0)
    a = Tensor.rand(3, 4, 5).realize()
    out = a.log2().pad(((0, 1), (0, 1), (0, 1)), value=1.0).sum().contiguous()
    run_schedule(check_schedule(out, 2))
    np.testing.assert_allclose(out.numpy(), np.pad(np.log2(a.numpy()), ((0, 1), (0, 1), (0, 1)), constant_values=1.0).sum(), atol=1e-5, rtol=1e-6)

  def test_multireduce_pad_reduce_unsafe(self):
    Tensor.manual_seed(0)
    a = Tensor.randn(3, 4, 5).abs().realize()
    b = Tensor.randn(3, 4, 5).abs().realize()
    out = (a.log2().pad(((0, 1), (0, 1), (0, 1)), value=1.0).sum()+b).abs().log2().pad(((0, 1), (0, 1), (0, 1)), value=1.0).sum().contiguous()
    # run_schedule(check_schedule(out, 1))
    run_schedule(check_schedule(out, 4))
    np.testing.assert_allclose(out.numpy(), np.pad(np.log2(np.abs(np.pad(np.log2(a.numpy()), ((0, 1), (0, 1), (0, 1)), constant_values=1.0).sum() + \
                                                   b.numpy())), ((0, 1), (0, 1), (0, 1)), constant_values=1.0).sum(), atol=3e-4, rtol=1e-6)

  def test_shrink_pad_safe(self):
    a = Tensor.ones((3, )).contiguous().realize()
    b = Tensor.ones((3, )).contiguous().realize()
    out = (a + b).shrink(((0, 1),)).pad(((0, 1),)).contiguous()
    run_schedule(check_schedule(out, 1))
    np.testing.assert_equal(out.numpy(), [2, 0])

  def test_shrink_pad_unsafe(self):
    a = Tensor.ones((3, )).contiguous().realize()
    out = a.exp2().shrink(((0, 1),)).pad(((0, 1),)).contiguous()
    run_schedule(check_schedule(out, 2))
    np.testing.assert_equal(out.numpy(), [2, 0])

  def test_base_change_shrink_pad(self):
    a = Tensor.ones(3, 3).contiguous().realize()
    b = a.exp2()
    c = b[:-1, :-1]
    d = c.pad(((0, 1), (0, 1))) * 2
    run_schedule(check_schedule(d, 2))
    np.testing.assert_equal(d.numpy(), np.pad(np.exp2(a.numpy())[:-1, :-1], ((0, 1), (0, 1)))*2)

  def test_base_change_expand_pad(self):
    a = Tensor.ones(3, 3).contiguous().realize()
    b = a.exp2()
    c = b[:, None, :]
    d = c.pad(((0, 0), (1, 1), (0, 0))) * 2
    run_schedule(check_schedule(d, 2))
    np.testing.assert_equal(d.numpy(), np.pad(np.exp2(a.numpy())[:, None, :], ((0, 0), (1, 1), (0, 0)))*2)

  def test_fuse_arange_pad_replicate_mode(self):
    x = Tensor.empty(3,3,3,3, requires_grad=True)
    y = x.pad((-1,2,2,-1), mode="replicate")
    dx = y.sum().gradient(x)[0]
    with Context(FUSE_ARANGE=1):
      sched = check_schedule(dx, 3)
    run_schedule(sched)
    np.testing.assert_allclose(dx.numpy(), [[[[0.,3.,9.],[0,1.,3.],[0.,0.,0.]]]*3]*3)

  # TODO like openpilot with imagef
  @unittest.skipUnless(is_dtype_supported(dtypes.half), "need half")
  def test_base_change_expand_expand(self):
    a = Tensor.ones(4, 4).contiguous().realize()
    b = a.cast(dtypes.half).expand(2, 4, 4)
    c = b.cast(dtypes.int).expand(2, 2, 4, 4)
    run_schedule(check_schedule(c, 2))
    np.testing.assert_equal(c.numpy(), np.ones(((2, 2, 4, 4)), dtype=np.int32))

  def test_base_change_pad_expand(self):
    a = Tensor.full((4, 4), 1.).contiguous().realize()
    b = Tensor.full((4, 4), 2.).contiguous().realize()
    c = (a + b).pad(((1, 1), (1, 1)))
    d = c.cast(dtypes.int).expand((2, 6, 6)) * 4
    run_schedule(check_schedule(d, 2))
    c_np = np.pad((np.full((4, 4), 2., dtype=np.float32) + np.full((4, 4), 1., dtype=np.float32)), ((1, 1), (1, 1)), constant_values=0.0)
    np.testing.assert_equal(d.numpy(), np.broadcast_to(c_np.astype(np.half), (2, *c_np.shape)) * 4)

  def test_pad_reduce_unsafe_multiview_st(self):
    P = Tensor.ones(3, 3).contiguous()
    sums = P.sum(axis=1, keepdim=True)
    P /= sums
    p = P[0]
    p = p.pad(((1, 0), ))
    p = p.repeat([2])
    run_schedule(check_schedule(p, 3))
    tiny_ret = p.numpy()

    P = np.ones((3, 3), dtype=np.float32)
    sums = P.sum(axis=1, keepdims=True)
    P /= sums
    p = P[0]
    p = np.pad(p, (1, 0), 'constant')
    p = np.tile(p, 2)
    np.testing.assert_allclose(tiny_ret, p)

  def test_bitcast_fuses(self):
    x = Tensor.empty(1, dtype=dtypes.float32)
    a = x.exp2().bitcast(dtypes.int32)
    b = x.bitcast(dtypes.int32)
    check_schedule(a+b, 1) # this should fuse when it makes sense

  @unittest.skip("disabling subbuffer manually isn't supported anymore")
  def test_bitcast_disable_subbufer(self):
    x = cast(UOp, Tensor.empty(1, dtype=dtypes.float32).realize().uop)
    a = x.alu(Ops.EXP2).cast(dtypes.int32, True, allow_buffer_view=False)
    b = x.cast(dtypes.int32, True, allow_buffer_view=False)
    b = a.alu(Ops.ADD, b)
    check_schedule(b, 1)

  def test_reduceop_reshape_dont_push(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(10, 20).realize()
    out = x.argmax(1)
    run_schedule(check_schedule(out, 3)) # TODO: push a reduceop through a reshape

  def test_conv2d(self): _test_conv2d(7)
  def test_conv2d_fused(self): _test_conv2d(5, FUSE_CONV_BW=1)

  @unittest.skipUnless(is_dtype_supported(dtypes.half) and is_dtype_supported(dtypes.ulong), "need half and ulong")
  def test_conv2d_half(self): _test_conv2d(7, dtype=dtypes.half)
  @unittest.skipUnless(is_dtype_supported(dtypes.half), "need half")
  @unittest.skipIf(Device.DEFAULT == "WEBGPU", "Causes other tests to fail")
  @unittest.expectedFailure
  def test_conv2d_fused_half(self): _test_conv2d(5, dtype=dtypes.half)

  def test_schedule_mem_used(self):
    base = GlobalCounters.mem_used
    Tensor.ones(256).contiguous().realize()
    Tensor.ones(5, 5).contiguous().schedule()
    self.assertEqual(GlobalCounters.mem_used-base, 0)

  @unittest.skip("TODO: this is consistently creating non reproducible failures")
  def test_schedule_mem_used_with_inputs(self):
    base = GlobalCounters.mem_used
    x = Tensor.ones(256).contiguous().realize()
    (x+Tensor.ones(256).contiguous()).schedule()
    self.assertEqual(GlobalCounters.mem_used-base, 1024)

  def test_const_schedule(self):
    constv = Tensor.empty(2, 2).uop.const_like(10)
    check_schedule(constv, 0)

  def test_const_schedule_contig(self):
    constv = Tensor.empty(2, 2).uop.const_like(10).contiguous()
    check_schedule(constv, 1)

  @unittest.skipIf(Device.DEFAULT != "GPU", "image only supported on GPU")
  def test_image_matmul(self):
    with Context(IMAGE=2):
      x = Tensor.randn((9, 9)).realize()
      y = Tensor.randn((9, 9)).realize()
      out = x@y
      run_schedule(check_schedule(out, 3))
      np.testing.assert_allclose(out.numpy(), x.numpy()@y.numpy(), atol=1e-4, rtol=1e-4)
      self.assertIsInstance(out.dtype, ImageDType)
      self.assertIsNotNone(out.uop.base.realized)
      self.assertIsInstance(out.uop.base.realized.dtype, ImageDType)

  def _test_fusion(self, shapes, f, cnt):
    with Context(DEBUG=0, TRACK_MATCH_STATS=0): args = [Tensor.randn(s).realize() for s in shapes]
    run_schedule(check_schedule(compare:=f(*args), cnt))
    if getenv("COMPARE", 1):
      import torch
      good = f(*[torch.tensor(x.numpy()) for x in args])
      np.testing.assert_allclose(compare.numpy(), good.numpy(), atol=1e-4, rtol=1e-4)

  def test_late_fusion_simple(self):
    self._test_fusion([(4, 4), (4, 1)], lambda a,b:a.sum(1, keepdim=True)+b, 1)

  def test_late_fusion_post_reshape(self):
    self._test_fusion([(4, 4), (1, 4)], lambda a,b:a.sum(1).reshape(b.shape)+b, 1)

  def test_late_fusion_post_permute(self):
    self._test_fusion([(4, 6, 4), (4, 4, 1)], lambda a,b:a.sum(1, keepdim=True).permute((2, 0, 1))+b, 2)

  def test_late_fusion_double_transpose(self):
    self._test_fusion([(32, 16, 1)],
                      lambda a:(a.expand(32, 16, 16).sum((2,), keepdim=True).permute((1, 0, 2))+2).permute((1, 0, 2)).contiguous(), 1)

  def test_late_fusion_post_expand(self):
    self._test_fusion([(32, 32)], lambda a:a-a.sum(1), 2)

  def test_cast_padded_view(self):
    a = Tensor.arange(4).reshape(1, 4)
    casted_view = a.pad(((0, 1), (0, 0))).cast(dtypes.float)
    casted_view.realize()
    self.assertEqual(casted_view.uop.base.realized.size, 4)
    realized_view = casted_view.contiguous().realize()
    self.assertEqual(realized_view.uop.base.realized.size, 8)
    self.assertListEqual(realized_view.tolist(), [[0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 0.0]])

  # NOTE: we only reorder CAST if it's an EXPAND
  def test_cast_after_shrink(self):
    a = Tensor.arange(4).reshape(1, 4)
    casted_view = a.shrink(((0, 1), (0, 2))).cast(dtypes.float)
    casted_view.realize()
    self.assertEqual(casted_view.uop.base.realized.size, 2)
    realized_view = casted_view.contiguous().realize()
    self.assertEqual(realized_view.uop.base.realized.size, 2)
    self.assertListEqual(realized_view.tolist(), [[0, 1]])

  def test_cast_const_view(self):
    a = Tensor.ones((4, 4), dtype=dtypes.float32)
    casted_view = a.cast(dtypes.int32)
    run_schedule(check_schedule(casted_view, 0))
    self.assertIsNone(casted_view.uop.base.realized)
    realized_const_view = casted_view.contiguous()
    run_schedule(check_schedule(realized_const_view, 1))
    self.assertListEqual(realized_const_view.tolist(), [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])

  @given(strat.sampled_from(dtypes.all), strat.sampled_from(dtypes.all))
  def test_cast_padded_const(self, dt1, dt2):
    assume(is_dtype_supported(dt1) and is_dtype_supported(dt2))
    a = Tensor(1, dtype=dt1).reshape(1, 1).pad(((1, 1), None))
    casted_view = a.cast(dt2)
    run_schedule(check_schedule(casted_view, 0))
    realized_const_view = casted_view.contiguous()
    run_schedule(check_schedule(realized_const_view, 1))
    np.testing.assert_equal(realized_const_view.numpy(), [[0], [1], [0]])

class TestIndexing(unittest.TestCase):
  def check_schedule(self, xt:Union[Tensor,List[Tensor]], cnt:int):
    with Context(FUSE_ARANGE=getenv("FUSE_ARANGE", 1)):
      lst = [xt] if isinstance(xt, Tensor) else xt
      s = Tensor.schedule(*lst)
      lowered = [x[1] for x in lower_schedule(s.copy())]
      kernels = [ei for ei in list(lowered) if isinstance(ei.prg, CompiledRunner)]
      if FUSE_ARANGE: self.assertEqual(len(kernels), cnt)
      for ei in lowered: ei.run(do_update_stats=True)
    return s

  def test_simple_indexing(self):
    X = Tensor.randn(10, 10).realize()
    idxs = Tensor([0, 2]).realize()
    xt = X[idxs]
    self.check_schedule(xt, 2)
    np.testing.assert_equal(xt.numpy(), X.numpy()[idxs.numpy()])

  def test_simple_indexing_alt(self):
    X = Tensor.arange(16).reshape(4, 4)
    xt = X[[1, 2], [1, 2]]
    self.check_schedule(xt, 5)
    np.testing.assert_equal(xt.numpy(), (np.arange(16).reshape(4, 4))[[1, 2], [1, 2]])

  def test_advanced_indexing(self):
    X = Tensor.arange(10)+1
    xt = X[[0]]
    self.check_schedule(xt, 2)
    np.testing.assert_equal(xt.numpy(), (np.arange(10)+1)[[0]])

  def test_advanced_indexing_alt(self):
    X = Tensor.arange(6).reshape(3, 2)+1
    xt = X[[Tensor([2]), Tensor([1])]]
    self.check_schedule(xt, 6)
    np.testing.assert_equal(xt.numpy(), 6)

  def test_advanced_simple_indexing_combined(self):
    X = Tensor.arange(16).reshape(4, 4)
    xt = X[1:2, [1, 2]]
    self.check_schedule(xt, 4)

  def test_push_through_reshape(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(10, 20).realize()
    out = x.argmax(1)
    self.check_schedule(out, 2)
    np.testing.assert_allclose(out.numpy(), np.argmax(x.numpy(), 1))

  def test_arange_push_through_expand(self):
    Tensor.manual_seed(0)
    a = Tensor.arange(4,)
    b = Tensor.randn(4, 4).realize()
    out = (a+b).sum()
    self.check_schedule(out, 1)
    np.testing.assert_allclose(out.numpy(), (np.arange(4)+b.numpy()).sum(), atol=1e-5)

  def test_argmin(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(4, 32).realize()
    out = x.argmin(-1)
    self.check_schedule(out, 2)
    np.testing.assert_equal(out.numpy(), x.numpy().argmin(axis=-1))

  def test_argmax(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(4, 32).realize()
    out = x.argmax(-1)
    self.check_schedule(out, 2)
    np.testing.assert_equal(out.numpy(), x.numpy().argmax(axis=-1))

  def test_arange_transposed(self):
    Tensor.manual_seed(0)
    x = Tensor.randint(4, 1).realize()
    a = ((Tensor.arange(4,)*x).T).sum()
    self.check_schedule(a, 1)
    np.testing.assert_equal(a.numpy(), (np.arange(4)*x.numpy()).T.sum())

  def test_div_padded_arange(self):
    x = Tensor.full((2,2), 16)
    y = x.idiv(Tensor.linspace(2, 8, steps=4, dtype=dtypes.int).reshape(2,2)).pad(((1,1), (1,1)))
    out = y.sum(axis=1)
    with Context(FUSE_ARANGE=1): run_schedule(check_schedule(out, 2))
    self.assertListEqual(out.tolist(), [0, 12, 4, 0])

  def test_arange_transposed_descendants(self):
    Tensor.manual_seed(0)
    x = Tensor.randint(4, 1).realize()
    a = (Tensor.arange(4,)*x).T
    b = Tensor.randint(4, 4).realize()
    out = (a+b).sum()
    self.check_schedule(out, 1)
    np.testing.assert_equal(out.numpy(), ((np.arange(4)*x.numpy()).T+b.numpy()).sum())

  def test_arange_index(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(5, 2).realize()
    a = Tensor.arange(10)
    out = (x + a[2]).sum()
    self.check_schedule(out, 1)
    np.testing.assert_allclose(out.numpy(), (x.numpy()+np.arange(10)[2]).sum(), atol=1e-5, rtol=1e-6)

  def test_arange_index_shrink(self):
    Tensor.manual_seed(0)
    with Context(TRACK_MATCH_STATS=0):
      x = Tensor.randn(11).realize()
    a = Tensor.arange(22)
    out = (x + a[:11]).sum()
    self.check_schedule(out, 1)

  def test_arange_index_contiguous(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(5, 2).realize()
    a = Tensor.arange(10).contiguous()
    out = (x + a[2]).sum()
    self.check_schedule(out, 3)
    np.testing.assert_allclose(out.numpy(), (x.numpy()+np.arange(10)[2]).sum(), atol=1e-5, rtol=1e-6)

  def test_arange_index_child(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(5, 2).realize()
    a = Tensor.arange(10)+1
    out = (x + a[2]).sum()
    self.check_schedule(out, 1)
    np.testing.assert_allclose(out.numpy(), (x.numpy()+(np.arange(10)+1)[2]).sum(), atol=1e-5, rtol=1e-6)

  def test_arange_index_contiguous_child(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(5, 2).realize()
    a = (Tensor.arange(10)+1).contiguous()
    out = (x + a[2]).sum()
    self.check_schedule(out, 3)
    np.testing.assert_allclose(out.numpy(), (x.numpy()+(np.arange(10)+1)[2]).sum(), atol=1e-5, rtol=1e-6)

  def test_arange_childless_base(self):
    a = Tensor.arange(4)
    self.check_schedule(a, 1)
    np.testing.assert_equal(a.numpy(), np.arange(4))

  def test_arange_childless_view(self):
    a = Tensor.arange(4).reshape(2, 2)
    a[0] = 4
    np.testing.assert_equal(a.numpy(), [[4, 4], [2, 3]])

  def test_arange_group_childless_base(self):
    Tensor.manual_seed(0)
    x = Tensor.randint(4).realize()
    a = Tensor.arange(4)+x
    self.check_schedule(a, 1)
    np.testing.assert_equal(a.numpy(), np.arange(4)+x.numpy())

  def test_arange_group_childless_view(self):
    Tensor.manual_seed(0)
    x = Tensor.ones(4).contiguous().realize()
    a = Tensor.arange(4)+x
    a[0] = 6
    np.testing.assert_equal(a.numpy(), [6., 2., 3., 4.])

  @unittest.skip("BUFFER_VIEW no longer supported on non-disk devices")
  def test_arange_view_op(self):
    a = Tensor.arange(12).reshape(4, 3).shrink(((1, 2), (1, 3))).contiguous()
    sched = self.check_schedule(a, 1)
    self.assertIs(sched[1].ast.op, Ops.BUFFER_VIEW)
    np.testing.assert_equal(a.numpy(), [[4, 5]])

  @unittest.skipIf(Device.DEFAULT == "CPU", "tests copy from ext device")
  def test_arange_shrink_copy(self):
    a = Tensor.arange(12).reshape(4, 3).shrink(((1, 2), (1, 3))).to("CPU")
    sched = self.check_schedule(a, 2) # NOTE: there is a contiguous between REDUCE_AXIS and COPY
    self.assertIs(sched[-1].ast.op, Ops.COPY)
    np.testing.assert_equal(a.numpy(), [[4, 5]])

  @unittest.skipIf(Device.DEFAULT == "CPU", "tests copy from ext device")
  def test_arange_expand_copy(self):
    a = Tensor.arange(4).reshape(2, 2, 1).expand(2, 2, 2).contiguous().to("CPU")
    sched = self.check_schedule(a, 2) # NOTE: there is a contiguous between REDUCE_AXIS and COPY
    self.assertIs(sched[2].ast.op, Ops.COPY)
    self.assertIs(store_val(sched[1]).op, Ops.LOAD)
    self.assertIs(store_val(sched[0]).op, Ops.ADD)
    np.testing.assert_equal(a.numpy(), [[[0, 0], [1, 1]], [[2, 2], [3, 3]]])

  @unittest.skipUnless(is_dtype_supported(dtypes.half), "need half")
  def test_precompute_freqs_cis(self):
    from extra.models.llama import precompute_freqs_cis
    args = {"dim":32 if CI else 128, "end":2048 if CI else 8192, "theta":10000}
    fused = precompute_freqs_cis(**args)
    with Context(FUSE_ARANGE=1):
      run_schedule(check_schedule(fused, 3))
    if getenv("CHECK", 1):
      ref = precompute_freqs_cis(**args)
      run_schedule(check_schedule(ref, 3))
      np.testing.assert_equal(fused.numpy(), ref.numpy())

  def test_fuse_assign_contiguous(self):
    x = Tensor.zeros(4, 4, dtype=dtypes.int).contiguous().realize()
    a = Tensor.arange(8).reshape(4, 2)
    self.check_schedule(x.shrink((None, (0, 2))).assign(a.contiguous()), 2)
    np.testing.assert_equal(x.numpy(), [[0, 1, 0, 0], [2, 3, 0, 0], [4, 5, 0, 0], [6, 7, 0, 0]])

  def test_assign_non_contiguous(self):
    x = Tensor.zeros(4, 4, dtype=dtypes.int).contiguous().realize()
    y = Tensor.randint(4, 2)
    a = Tensor.arange(8).reshape(4, 2)+y
    x.shrink((None, (0, 2))).assign(a).realize()
    xref = np.zeros((4, 4), dtype=int)
    xref[:, :2] = np.arange(8).reshape(4, 2)+y.numpy()
    np.testing.assert_equal(x.numpy(), xref)

  def test_sparse_categorical_crossentropy_simple(self):
    X = Tensor([[0, 2, 3], [1, 2, 3]]).realize()
    Y = Tensor([1, 2]).realize()
    loss = X.sparse_categorical_crossentropy(Y)
    self.check_schedule(loss, 4)
    np.testing.assert_allclose(loss.item(), 0.878309, atol=1e-5, rtol=1e-6)

  @unittest.skipIf(Device.DEFAULT == "WEBGPU", "Validation error on WebGPU")
  def test_mnist_val(self):
    from tinygrad.nn.datasets import mnist
    import torch
    _, Y_train, _, _ = mnist()
    samples = Tensor.randint(BS:=getenv("BS", 512), high=cast(int,Y_train.shape[-1])).realize()
    yt = Tensor.randn(BS, 10).realize()
    with Context(SPLIT_REDUCEOP=0):
      loss = yt.sparse_categorical_crossentropy(Y_train[samples])
      self.check_schedule(loss, 6)
      loss_fused = loss.numpy()
    loss_ref = torch.nn.CrossEntropyLoss()(torch.tensor(yt.numpy()), torch.tensor(Y_train.numpy())[torch.tensor(samples.numpy())])
    np.testing.assert_allclose(loss_fused, loss_ref.numpy(), atol=1e-6, rtol=1e-6)

  @unittest.expectedFailure
  def test_arange_fuse_grouped_children(self):
    X = Tensor.randn(4, 4).realize()
    r = (X+Tensor.arange(16).reshape(4, 4)).sum()
    out0 = r+2
    out1 = r+3
    self.check_schedule([out0, out1], 1)
    r_ref = (X.numpy()+np.arange(16).reshape(4, 4)).sum()
    np.testing.assert_allclose(out0.numpy(), r_ref+2, rtol=2e-7)
    np.testing.assert_allclose(out1.numpy(), r_ref+3, rtol=2e-7)

  def test_dont_fold_arange_contiguous_view(self):
    X = Tensor.randn(4, 4).realize()
    r = (X+Tensor.arange(16).reshape(4, 4).contiguous()).sum(1, keepdim=True)
    self.check_schedule([r], 2)
    np.testing.assert_allclose(r.numpy(), (X.numpy()+np.arange(16).reshape(4, 4)).sum(1, keepdims=True), atol=1e-5, rtol=1e-6)

  @unittest.skip("multi output isn't supported")
  def test_multiview_arange_children(self):
    X = Tensor.randn(2,3,4,4).numpy()
    with Context(FUSE_ARANGE=1):
      compare = Tensor(X).interpolate(size=(2, 2), mode="linear").numpy()
    with Context(FUSE_ARANGE=0, TRACK_MATCH_STATS=0):
      ref = Tensor(X).interpolate(size=(2, 2), mode="linear").numpy()
    np.testing.assert_allclose(ref, compare, atol=1e-5, rtol=1e-6)

  def test_recursive_swizzle(self):
    a = Tensor([1,2,3,4]).realize()
    for _ in range(24): a = a + a
    new_uop = a.reshape(4,1).realize().uop
    self.assertEqual(new_uop.st, ShapeTracker.from_shape((4,)).reshape((4, 1)))
    self.assertEqual(swizzle_cnt(new_uop), 0)

  def test_no_rewrite_elementwise(self):
    a = Tensor.empty(32, 32)
    b = Tensor.empty(32, 32)
    sink = (a+b).schedule()[0].ast
    self.assertEqual(swizzle_cnt(sink), 0)

  def test_simple_store_reshape(self):
    a = Tensor.empty(32, 32).sum(axis=1)+Tensor.empty(1,32)
    ast = a.schedule()[0].ast
    self.assertEqual(ast.shape, (32, 1))
    self.assertEqual(a.uop.shape, (1, 32))

  def test_no_reshape_reduceop(self):
    a = Tensor.empty(32, 32).sum(axis=(1,)).contiguous()
    ast = a.schedule()[0].ast
    self.assertEqual(ast.shape, (32, 1))
    self.assertEqual(a.uop.shape, (32,))

def swizzle_cnt(u:UOp) -> int:
  return len([x for x in u.toposort() if x.op is Ops.VIEW and len(x.src) != 0 and x.src[0].op not in {Ops.BUFFER, Ops.DEFINE_GLOBAL, Ops.ASSIGN}])

class TestSwizzle(unittest.TestCase):
  def test_swizzle_simple(self):
    Tensor.manual_seed(0)
    with Context(DEBUG=0, TRACK_MATCH_STATS=0):
      a = Tensor.randint(32, 32).realize()
    r = (a+a).sum(1).sum(0)
    # double reduce collapses to a single reduce
    with Context(DONT_GROUP_REDUCES=1):
      run_schedule(check_schedule(r, 1))
    self.assertEqual(r.numpy(), (a.numpy()+a.numpy()).sum(1).sum(0))

  def test_single_swizzle(self):
    Tensor.manual_seed(0)
    with Context(DEBUG=0, TRACK_MATCH_STATS=0):
      a = Tensor.randint(4, 1).realize()
      b = Tensor.ones((1, 1), dtype=a.dtype).contiguous().realize()
    # ADD(REDUCE(RESHAPE(LOAD)), LOAD) to ADD(REDUCE(RESHAPE(LOAD))), RESHAPE(LOAD)
    r = a.sum(0)+b
    run_schedule(check_schedule(r, 1))
    self.assertEqual(r.numpy(), a.numpy().sum(0)+1)

  def test_double_swizzle_possible(self):
    Tensor.manual_seed(0)
    with Context(DEBUG=0, TRACK_MATCH_STATS=0):
      a = Tensor.randint(4,).realize()
      b = Tensor.randint(4,).realize()
    # parallel reduce!
    add = a.sum(0)+b.sum(0)
    with Context(DONT_GROUP_REDUCES=1):
      run_schedule(check_schedule(add, 1))
    self.assertEqual(add.numpy(), a.numpy().sum(0)+b.numpy().sum(0))

  @unittest.skip("TODO: how do we express the norm")
  def test_softmax_one_kernel(self):
    Tensor.manual_seed(0)
    with Context(DEBUG=0, TRACK_MATCH_STATS=0):
      a = Tensor.randn(32, 32).realize()
    t = a.softmax()
    with Context(DONT_GROUP_REDUCES=1, DONT_REALIZE_EXPAND=1):
      check_schedule(t, 1)

  def test_argmax_one_kernel(self):
    Tensor.manual_seed(0)
    with Context(DEBUG=0, TRACK_MATCH_STATS=0):
      a = Tensor.randn(10, 20).realize()
    t = a.argmax(0)
    with Context(DONT_GROUP_REDUCES=1, DONT_REALIZE_EXPAND=1): t.realize()

  def test_swizzle_reduceop(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(4,4).realize()
    y = Tensor.randn(4,4,4).realize()
    out = x.reshape(4,4,1).expand(4,4,4).sum(axis=(1,))+y
    with Context(DONT_REALIZE_EXPAND=1, DONT_GROUP_REDUCES=1):
      run_schedule(check_schedule(out, 1))
    np.testing.assert_allclose(out.numpy(), np.tile(x.numpy().reshape(4,4,1), (1,1,4)).sum(axis=1)+y.numpy())

  def test_permute_rewrite(self):
    x = Tensor.randn(4, 4, 16).realize()
    y = Tensor.randn(4, 1, 16).realize()
    z = Tensor.randn(4, 4, 1).realize()
    t = (x*y).sum(axis=(0, 2)).reshape(1, 4, 1).permute(0, 2, 1)+z
    with Context(DONT_GROUP_REDUCES=1, DONT_REALIZE_EXPAND=1): run_schedule(check_schedule(t, 1))
    t_np = (x.numpy()*y.numpy()).sum(axis=(0, 2)).reshape(1, 4, 1).transpose(0, 2, 1)+z.numpy()
    np.testing.assert_allclose(t.numpy(), t_np, atol=1e-6, rtol=1e-3)

  @unittest.skip("TODO: this swizzle isn't resolvable when there's a mask")
  def test_swizzle_failure_permute(self):
    a = Tensor.empty(45,65).T.reshape(65,1,45).pad((None,None,(0,45))).expand(65,45,90)
    b = Tensor.empty(45,65)
    a_reduce = a.sum(axis=(2,), keepdim=True).sum(axis=(1,))
    b_reduce = b.sum(axis=(0,))
    t = a_reduce+b_reduce
    with Context(DONT_GROUP_REDUCES=1, DONT_REALIZE_EXPAND=1): run_schedule(check_schedule(t, 1))

  def test_parallel_reduce_possible(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(4, 2, 2).realize()
    y = Tensor.randn(4, 2, 2).realize()
    t = x.sum(axis=1)+y.sum(axis=1)
    with Context(DONT_GROUP_REDUCES=1): run_schedule(check_schedule(t, 1))
    np.testing.assert_allclose(t.numpy(), x.numpy().sum(axis=1)+y.numpy().sum(axis=1), atol=1e-6, rtol=1e-3)

  # kernels can only have 1 or n in each dim
  @unittest.expectedFailure
  def test_dont_parallelize_different_n(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(4, 2, 2).realize()
    y = Tensor.randn(4, 3, 2).realize()
    t = x.sum(axis=1)+y.sum(axis=1)
    with Context(DONT_GROUP_REDUCES=1): run_schedule(check_schedule(t, 1))
    np.testing.assert_allclose(t.numpy(), x.numpy().sum(axis=1)+y.numpy().sum(axis=1), atol=1e-6, rtol=1e-3)

  def test_unsafe_pad(self):
    x = Tensor.full((2,2), 1.0).contiguous()
    y = x*x.sum((1,)).reciprocal()
    t = y.pad(((0,1),None))
    run_schedule(check_schedule(t, 3))
    np.testing.assert_equal(t.numpy(), [[0.5, 0.5], [0.5, 0.5], [0., 0.]])

def store_val(si:ScheduleItem): return si.ast.src[0].src[1]
zero_pm = UPat(Ops.CONST, arg=0)
class TestView(unittest.TestCase):
  def test_all_masked_out(self):
    # start with non CONST Ops
    a = Tensor.rand(10, 10).realize()
    # all masked out, degrades to const 0
    b = a.pad(((0, 10), None))[10:]
    sched = check_schedule(b.contiguous(), 1)
    assert zero_pm.match(store_val(sched[-1]), {})
    run_schedule(sched)
    np.testing.assert_equal(b.numpy(), 0)

  def test_mask_dim_1(self):
    # mask out dim = 1 works too
    a = Tensor.rand(10, 10).realize()
    b = a.pad((None, (0, 10)))[:, 10:]
    assert b.shape == (10, 10)
    sched = check_schedule(b.contiguous(), 1)
    self.assertEqual(sched[-1].ast.full_shape, (10, 10))
    assert zero_pm.match(store_val(sched[-1]), {})
    run_schedule(sched)
    np.testing.assert_equal(b.numpy(), 0)

  def test_zero_size_alt(self):
    a = Tensor.empty(135, 0, 9)
    b = a.pad(((0, 0), (0, 0), (18, 0)))
    check_schedule(b, 0)

  def test_partial_mask(self):
    # partial masked out does not degrade into CONST
    a = Tensor.rand(10, 10).realize()
    b = a.pad(((0, 5), None))[5:]
    assert b.shape == (10, 10)
    sched = check_schedule(b.contiguous(), 1)
    self.assertEqual(store_val(sched[-1]).op, Ops.LOAD)
    self.assertEqual(store_val(sched[-1]).st_arg, b.uop.st)
    run_schedule(sched)
    np.testing.assert_allclose(b.numpy(), np.pad(a.numpy(), ((0, 5), (0, 0)))[5:])

  # a*VIEW(x), where VIEW(x) = 0
  # x collapses along with its children
  def test_parent_view_collapses(self):
    a = Tensor([1, 2])
    b = Tensor.arange(3).contiguous()
    bv = b.pad(((0, 2),))[-2:]
    # this becomes a late a*0
    late_mul = a*bv
    check_schedule(late_mul, 0)
    # the arange doesn't realize
    self.assertIsNone(b.uop.base.realized)
    # mul doesn't realize
    self.assertIsNone(late_mul.uop.base.realized)
    self.assertEqual(late_mul.tolist(), [0, 0])

  # SINK has two branches:
  # a*VIEW(x), where VIEW(x) = 0
  # x+2
  # as long as one child realizes, x does not collapse
  def test_parent_multiple_children_no_collapse(self):
    a = Tensor([1, 2])
    b = Tensor.arange(3).contiguous()
    bv = b.pad(((0, 2),))[-2:]
    late_mul = a*bv
    other_child = b+2
    s = check_schedule([late_mul, other_child], 2)
    # the arange becomes a BUFFER
    self.assertIs(b.uop.base.op, Ops.BUFFER)
    # mul still collapses
    self.assertIs(late_mul.uop.base.op, Ops.CONST)
    run_schedule(s)
    self.assertEqual(other_child.tolist(), [2, 3, 4])

def tensor_rewrite(t) -> UOp: return graph_rewrite(t.uop.base, merge_views+symbolic_simple)
class TestSimplifier(unittest.TestCase):
  def test_sink_childless_const(self):
    x = Tensor(0)
    check_schedule(x, 0)

  def test_sink_childless_const_alt_expanded(self):
    x = Tensor.zeros(4, 4).contiguous()
    check_schedule(x, 1)

  def test_all_const_uops(self):
    a = Tensor(4)*Tensor(2)
    sink = tensor_rewrite(a)
    assert UPat.cvar().match(sink, {})

  def test_masked_const_elementwise(self):
    a = Tensor.eye(10)@Tensor.eye(10)
    sink = tensor_rewrite(a)
    assert UPat(Ops.REDUCE_AXIS, src=(UPat.cvar().view()*UPat.cvar().view(),)).match(sink, {})

  def test_elementwise_ops(self):
    a = Tensor.empty(4, 4, dtype=dtypes.int)
    sink = tensor_rewrite(a*0)
    assert UPat(Ops.CONST, arg=0).match(sink, {})
    self.assertIs(tensor_rewrite(a*1).base, a.uop.base)
    self.assertIs(tensor_rewrite(a+0).base, a.uop.base)

  def test_cast_folding(self):
    a = Tensor(1.0).cast(dtypes.int)
    sink = tensor_rewrite(a)
    assert UPat.cvar(dtype=dtypes.int).match(sink, {})

  def test_const_folding_mul(self):
    a = Tensor([1])
    sink = tensor_rewrite(a*0)
    assert UPat(Ops.CONST, arg=0).match(sink, {}), f"expected {sink} to collapse to a const 0"
    assert sink.shape == a.shape

  def test_const_folding_ne(self):
    a = Tensor([1])
    sink = tensor_rewrite(a != a)
    assert UPat(Ops.CONST, arg=False).match(sink, {}), f"expected {sink} to collapse to a const False"
    assert sink.shape == a.shape

  def test_const_folding_lt(self):
    a = Tensor([1])
    sink = tensor_rewrite(a < a)
    assert UPat(Ops.CONST, arg=False).match(sink, {}), f"expected {sink} to collapse to a const False"
    assert sink.shape == a.shape

tensor_const_pm = PatternMatcher([
  (UPat(Ops.CONST, src=(UPat(Ops.VIEW, src=(UPat(Ops.DEVICE),)),)), lambda: True),
  (UPat(Ops.BIND, src=(UPat(Ops.DEFINE_VAR, src=(UPat(Ops.VIEW, src=(UPat(Ops.DEVICE),)))), UPat(Ops.CONST))), lambda: True),
])
class TestConst(unittest.TestCase):
  # ** part 1: basic functionality of a tensor directly created from CONST

  def test_tensor_const(self):
    a = Tensor(1)
    print(a.uop)
    self.assertTrue(tensor_const_pm.rewrite(a.uop))

  def test_tensor_variable(self):
    vv = UOp.variable("a", 0, 10).bind(1)
    a = Tensor(vv)
    print(a.uop)
    self.assertTrue(tensor_const_pm.rewrite(a.uop))

  def test_const_schedule(self):
    a = Tensor.ones((4, 4))
    sched = a.schedule()
    self.assertEqual(len(sched), 0)

  def test_const_contiguous_schedule(self):
    # this ends up in the big graph
    a = Tensor.ones((4,)).contiguous()
    sched = a.schedule()
    self.assertEqual(len(sched), 1)

  def test_const_ast(self):
    a = Tensor.ones((4,)).pad((1, 1)).contiguous()
    sched = a.schedule()
    print(sched[0].ast)
    const_ast_pattern = UPat(Ops.SINK, src=(UPat.store(UPat(), UPat.where(UPat(Ops.VALID), UPat.cvar("x"), UPat(Ops.CONST, arg=0))),))
    self.assertEqual(len(const_ast_pattern.match(sched[0].ast, {})), 1)
    run_schedule(sched)
    self.assertListEqual(a.tolist(), [0, 1, 1, 1, 1, 0])

  def test_unmasked_const_ast(self):
    a = Tensor.ones((4,)).contiguous()
    sched = a.schedule()
    print(sched[0].ast)
    const_ast_pattern = UPat(Ops.SINK, src=(UPat.store(UPat(), UPat(Ops.CONST)),))
    self.assertEqual(len(const_ast_pattern.match(sched[0].ast, {})), 1)
    run_schedule(sched)
    self.assertListEqual(a.tolist(), [1, 1, 1, 1])

  # ** part 2: scheduler behavior when const folding happens later

  def test_const_folding_no_realize(self):
    a = Tensor([1, 2, 3, 4])*0
    sched = a.schedule()
    self.assertEqual(len(sched), 0)

  def test_src_const_folding(self):
    with Context(TRACK_MATCH_STATS=0):
      a = Tensor.full((4,), 1).contiguous().realize()
      b = Tensor.full((4,), 2).contiguous().realize()
    mul0 = a*0
    add = b+mul0
    sched = add.schedule()
    self.assertEqual(len(sched), 0)
    # b+0 and b share the same underlying device memory
    self.assertIs(add.uop.buffer, b.uop.buffer)
    self.assertListEqual(add.tolist(), [2, 2, 2, 2])

  def test_src_masked_const_folding(self):
    with Context(TRACK_MATCH_STATS=0):
      a = Tensor.full((4,), 1).contiguous().realize()
      b = Tensor.full((6,), 2).contiguous().realize()
    mul0 = a*0
    add = b+mul0.pad((1, 1), value=2)
    sched = add.schedule()
    self.assertEqual(len(sched), 1)
    run_schedule(sched)
    # add gets assigned to a new buffer
    self.assertIsNot(add.uop.base.realized, b.uop.base.realized)
    self.assertListEqual(add.tolist(), [4, 2, 2, 2, 2, 4])

  # ** part 3: Tensor variable bindings

  #@unittest.expectedFailure # TODO: should schedule assert if you try to realize a Variable?
  def test_var_schedule(self):
    vv = UOp.variable("a", 0, 10).bind(1)
    a = Tensor(vv)
    sched = a.schedule()
    self.assertEqual(len(sched), 0)

  def test_add_tvar(self):
    vv = UOp.variable("a", 0, 10).bind(1)
    a = Tensor(vv)+2
    sched, var_vals = a.schedule_with_vars()
    self.assertEqual(len(sched), 1)
    run_schedule(sched, var_vals)
    self.assertEqual(a.tolist(), 3)

@unittest.skipIf(Device.DEFAULT == "CPU", "tests copy from another device to cpu")
class TestCopyFolding(unittest.TestCase):
  def test_const_copy_is_free(self):
    b = Tensor(1).to("CPU")
    check_schedule(b, 0, filter_sink=False)
    assert b.item() == 1

  def test_late_const_copy_folding(self):
    a = Tensor.arange(3).realize()
    zeros = Tensor.zeros(3).realize()
    b = (a*zeros).to("CPU")
    run_schedule(check_schedule(b, 0, filter_sink=False))
    self.assertListEqual(b.tolist(), [0, 0, 0])

  def test_alu_after_copy(self):
    a = Tensor.ones((4,)).to("CPU")
    b = Tensor.empty(4, device="CPU")
    add = a+b
    add.kernelize()
    assert all_same([x.device for x in add.uop.src]), f"ALU has different devices! {[x.device for x in add.src]}"

  def test_copy_to_same_device(self):
    a = Tensor.empty(4).uop
    b = a.copy_to_device(a.device)
    check_schedule(b, 0, filter_sink=False)
    b = schedule_graph_rewrite(b)
    # NOTE: Tensor.empty(4) always creates a VIEW(BUFFER) with ShapeTracker((4,)), we simplify this to jsut a BUFFER
    # in the scheduler because buffer already has shape (4,)
    self.assertIs(b, a.base)

  def test_copy_to_same_device_alt(self):
    a = Tensor.empty(4, 4).uop
    b = a.copy_to_device(a.device)
    check_schedule(b, 0, filter_sink=False)
    b = schedule_graph_rewrite(b)
    self.assertIs(b.base, a.base)

  def test_clone(self):
    a = Tensor.empty(4)
    check_schedule(a.clone(), 1, filter_sink=False)

  # NOTE: moving copy before view might change this
  def test_shrink_copy(self):
    a = Tensor.arange(4)
    view = a.shrink(((0, 2),))
    b = view.clone()
    # NOTE: this was sort of a bug making this 2
    run_schedule(check_schedule(b, 2, filter_sink=False))
    self.assertEqual(b.uop.base.buffer.size, 2)
    self.assertEqual(b.uop.size, 2)
    self.assertListEqual(b.tolist(), [0, 1])

  def test_expanded_copy(self):
    a = Tensor.arange(2)
    view = a.reshape(2, 1).expand(2, 2)
    b = view.clone()
    run_schedule(check_schedule(b, 2, filter_sink=False))
    self.assertEqual(b.uop.base.buffer.size, 4)
    self.assertEqual(b.uop.size, 4)
    self.assertListEqual(b.tolist(), [[0, 0], [1, 1]])

  def test_permuted_copy(self):
    a = Tensor.arange(4)
    b = a.reshape(2, 2).permute(1, 0)
    b.realize()
    self.assertListEqual(b.tolist(), [[0, 2], [1, 3]])

  def test_permute_on_disk(self):
    with open(temp('dt_arange_4_permute'), "wb") as f: f.write(Tensor.arange(4).realize().uop.base.buffer.as_buffer())
    a = Tensor.empty(4, dtype=dtypes.int32, device=f"disk:{temp('dt_arange_4_permute')}")
    b = a.reshape(2, 2).permute(1, 0).to("CPU")
    b.realize()
    self.assertListEqual(b.tolist(), [[0, 2], [1, 3]])

  def test_permute_after_shrink(self):
    a = Tensor.arange(5)
    b = a.shrink(((0, 4),)).reshape(2, 2).permute(1, 0).to("CPU")
    b.realize()
    self.assertListEqual(b.tolist(), [[0, 2], [1, 3]])

  # NOTE: disk permute must come after COPY
  # TODO: this is wrong because of the permute
  @unittest.expectedFailure
  def test_permute_after_shrink_on_disk(self):
    with open(temp('dt_arange_5_permute'), "wb") as f: f.write(Tensor.arange(5).realize().uop.base.buffer.as_buffer())
    a = Tensor.empty(5, dtype=dtypes.int32, device=f"disk:{temp('dt_arange_5_permute')}")
    b = a.shrink(((0, 4),)).reshape(2, 2).permute(1, 0).to("CPU")
    b.realize()
    self.assertListEqual(b.tolist(), [[0, 2], [1, 3]])

class TestBufferUOp(unittest.TestCase):
  # BUFFER has a ShapeTracker of shape=(n,) and stride=(1,)
  def test_buffer_has_buffer(self):
    buf = Tensor.empty(10)
    self.assertIsNotNone(buf.uop.buffer)
    self.assertEqual(buf.uop.st, ShapeTracker.from_shape((10,)))
    # the device Buffer remains unallocated until it's we run the schedule
    self.assertFalse(buf.uop.buffer.is_allocated())
    add = buf+1
    sched = add.schedule()
    self.assertFalse(buf.uop.buffer.is_allocated())
    run_schedule(sched)
    self.assertTrue(buf.uop.buffer.is_allocated())

  def test_buffer_has_unique_buffer(self):
    buf = Tensor.empty(10)
    buf1 = buf.uop.buffer
    buf2 = buf.uop.buffer
    self.assertIs(buf1, buf2)

  # we also allow VIEW(BUFFER) to access the underlying device Buffer, as long as it's contiguous
  def test_buffer_view_allowed(self):
    add = Tensor.empty(1, 1)+Tensor.empty(1, 1)
    add.realize()
    self.assertIsNotNone(add.uop.buffer)
    self.assertEqual(add.uop.shape, (1, 1))

  def test_buffer_view_not_allowed(self):
    permuted_view = Tensor.empty(1, 2, 3).permute(0, 2, 1)
    merged = graph_rewrite(permuted_view.uop, merge_views)
    with self.assertRaisesRegex(AssertionError, "VIEW only works here if it's contiguous"):
      merged.buffer # cannot access Buffer of a non contiguous VIEW

  def test_buffer_only_after_realize(self):
    a = Tensor([1])+Tensor([2])
    # accessing realized will return None
    self.assertIsNone(a.uop.realized)
    # accessing Buffer will assert
    with self.assertRaisesRegex(AssertionError, "must be BUFFER"):
      a.uop.buffer # there is no BUFFER on an unrealized ADD
    # Buffer only exists once we realize it
    a.realize()
    self.assertIsNotNone(a.uop.buffer)

  def test_const_does_not_realize(self):
    a = Tensor(1)+Tensor(2)
    run_schedule(check_schedule(a, 0))
    self.assertIsNone(a.uop.base.realized)

  def test_var_does_not_realize(self):
    a = Tensor(UOp.variable("a", 0, 10).bind(1))
    run_schedule(check_schedule(a, 0))
    self.assertIsNone(a.uop.base.realized)

  def test_view_does_not_realize(self):
    a = Tensor.randn(1, 4).expand(4, 4)
    a.realize()
    self.assertEqual(a.uop.base.realized.size, 4)
    a2 = a.contiguous().realize()
    self.assertEqual(a2.uop.base.realized.size, 16)

class TestContiguous(unittest.TestCase):
  def test_contiguous_buffer(self):
    a = Tensor.empty(4)
    b = a.contiguous()
    check_schedule(b, 0)

  def test_contiguous_buffer_view(self):
    a = Tensor.empty(4)
    b = a.reshape((2, 2)).contiguous()
    check_schedule(b, 0)

  def test_non_contiguous_buffer_view(self):
    a = Tensor.empty(4, 1)
    b = a.expand((4, 4)).contiguous()
    check_schedule(b, 1)

  def test_size_change_buffer_view(self):
    a = Tensor.empty(4)
    b = a.reshape((1, 1, 4)).shrink(((0, 1), (0, 1), (0, 3))).contiguous()
    check_schedule(b, 1)

  def test_double_contiguous_realizes_once(self):
    a = Tensor.empty(4, 1)
    b = a.expand((4, 4)).contiguous().contiguous()
    check_schedule(b, 1)

  def test_view_does_not_realize(self):
    a = Tensor.empty(4)
    b = a.expand((4, 4))
    check_schedule(b, 0)
    self.assertEqual(b.uop.base.buffer.size, 4)

  def test_contiguous_view_realizes(self):
    a = Tensor.empty(4)
    b = a.expand((4, 4)).contiguous()
    check_schedule(b, 1)
    self.assertEqual(b.uop.base.buffer.size, 16)

class TestUOpBecome(unittest.TestCase):
  # the simplest case, if we create a new BUFFER for this tensor UOp
  def test_new_buffer(self):
    a = Tensor.empty(4, 4)
    b = Tensor.empty(4, 4)
    add = a+b
    check_schedule(add, 1)
    # NOTE: realized base is always a flat buffer
    assert UPat(Ops.BUFFER).match(add.uop.base, {})
    # the Tensor UOp can optionally stack a VIEW on top of the BUFFER, in this case to preserve the (4, 4) shape of the tensor
    assert add.uop is not add.uop.base
    self.assertEqual(add.uop.size, 16)
    self.assertEqual(add.uop.shape, (4, 4))

  def test_new_buffer_view(self):
    a = Tensor.empty(4, 4)
    b = Tensor.empty(4, 4)
    add = (a+b).reshape(8, 2)
    check_schedule(add, 1)
    assert UPat(Ops.BUFFER).match(add.uop.base, {})
    # the shape is preserverd in the becomes_map.
    self.assertEqual(add.uop.shape, (8, 2))
    assert add.uop is not add.uop.base

  def test_new_flat_buffer(self):
    a = Tensor.empty(4,)
    b = Tensor.empty(4,)
    add = a+b
    check_schedule(add, 1)
    # BUFFER already has a shape (4,), this tensor just becomes a contiguous BUFFER
    assert UPat(Ops.BUFFER).match(add.uop, {})

  # sometimes we prefer to perform an op before movement ops, in this case we should stack the mops on top of the new buffer

  # NOTE: this expand is not reordered because there's before it to fuse
  def test_reorder_expand(self):
    a = Tensor.empty(4, 1)
    b = a.expand(4, 4).reciprocal()
    check_schedule(b, 1)
    self.assertEqual(b.uop.base.buffer.size, 16)
    self.assertEqual(b.uop.st, ShapeTracker.from_shape((4, 4)))

  def test_reorder_expand_alt(self):
    x = Tensor.empty(4, 1)
    y = Tensor.empty(4, 1)
    img = Tensor.empty(4, 4)
    z = (img*x) / y
    check_schedule(z, 1)

  def test_become_existing_buffer(self):
    a = Tensor.empty(4, 4)
    b = a*1
    assert UPat(Ops.MUL).match(b.uop, {}) # before scheduling it's a mul
    check_schedule(b, 0)
    assert UPat(Ops.VIEW, src=(UPat(Ops.BUFFER))).match(b.uop, {}) # scheduling merges all MovementOps into a single VIEW
    self.assertIs(a.uop.base.buffer, b.uop.base.buffer)

  def test_become_buf_with_mops(self):
    a = Tensor.empty(2, 4, 2)
    noop = a.shrink(((1, 2), (0, 4), (0, 2))).reshape(4, 2)*1+0
    # before realizing, this tensor is base
    assert noop.uop is noop.uop.base
    noop.realize()
    # it becomes a realized view after realize
    assert noop.uop is not noop.uop.base
    assert noop.uop.base.op is Ops.BUFFER
    late_add = noop+2
    late_add.realize()

  def test_become_const_in_base(self):
    a = Tensor.empty(4)
    b = a*0
    assert UPat(Ops.MUL).match(b.uop, {}) # before scheduling it's a mul
    check_schedule(b, 0)
    assert UPat(Ops.CONST, arg=0).match(b.uop.base, {}) # scheduling replaces the tensor uop with a VIEW(BUFFER)

  def test_become_const_in_view(self):
    # if we shrink the base down to a size 0, only the VIEW becomes CONST, base is unchanged.
    add = Tensor.empty(2, 2)+Tensor.empty(2, 2)
    b = add.shrink(((0, 1), (0, 0)))
    check_schedule(b, 0)
    assert UPat(Ops.CONST, arg=0).match(b.uop, {})
    self.assertEqual(b.shape, (1, 0))
    # the base is untouched.
    assert UPat(Ops.ADD).match(add.uop, {})

  def test_become_const_from_const(self):
    const_add = Tensor(1)+Tensor(2)
    assert UPat(Ops.ADD).match(const_add.uop, {})
    check_schedule(const_add, 0)
    assert UPat(Ops.CONST, arg=3).match(const_add.uop.base, {})

  # tensors can become another realized tensor source
  def test_become_existing_buf_simple(self):
    a = Tensor.empty(4, 4)
    b = a+0
    check_schedule(b, 0)
    assert b.uop.base.op is Ops.BUFFER
    self.assertIs(a.uop, b.uop)

  # they can also chain other movement ops on top of the tensor source
  def test_become_existing_buf_view(self):
    a = Tensor.empty(4, 4)
    b = a.permute((1, 0))+0
    check_schedule(b, 0)
    self.assertEqual(b.uop.st, a.uop.permute((1, 0)).st)

  def test_become_existing_buf_view_alt(self):
    a = Tensor.empty(4, 4)
    b = a.permute((1, 0)).reshape((8, 2))+0
    check_schedule(b, 0)
    self.assertEqual(b.uop.st, a.uop.permute((1, 0)).reshape((8, 2)).st)

  # they can also have other base parents that simplified, in that case we just backtrack to the chained mops
  def test_become_existing_buf_complex(self):
    a = Tensor.empty(4, 4)
    b = (a.permute((1, 0))+0).reshape((8, 2))+0
    check_schedule(b, 0)
    self.assertEqual(b.uop.st, a.uop.permute((1, 0)).reshape((8, 2)).st)
    assert b.uop.base.op is Ops.BUFFER

  def test_become_multiple_choices(self):
    a = Tensor.empty(16)
    b = (a.reshape(1, 1, 4, 1, 4)+0).reshape(1, 1, 4, 4).shrink(((0, 1), (0, 1), (0, 3), (0, 3)))+0
    c = (a.reshape(1, 1, 4, 4)+0).shrink(((0, 1), (0, 1), (0, 3), (0, 3)))+0
    check_schedule([b, c], 0)
    assert all_same([x.uop.base.realized for x in [a,b,c]])
    # these movement ops result in the same ShapeTracker
    assert b.uop.st == c.uop.st
    assert b.uop is c.uop
    assert UPat(Ops.VIEW, src=(UPat(Ops.BUFFER),)).match(c.uop, {})

  def test_setitem_becomes_subbuffer(self):
    a = Tensor.full((4,), 2.).contiguous().realize()
    b = a.shrink(((0, 2),)).assign(Tensor.full((2,), 1.0))
    b.realize()
    assert a.uop.is_realized
    assert a.uop.buffer._base is None
    # b is a subbuffer of a
    assert b.uop.op is Ops.BUFFER_VIEW
    assert b.uop.src[0] is a.uop

  def test_setitem_offset(self):
    a = Tensor.full((16,), 0.).contiguous().realize()
    b = Tensor.full((16,), 1.).contiguous().realize()
    a_view = a[4:].reshape(3, 4).shrink(((0,2),(0,2))).reshape((4,))
    b.shrink(((0,4),)).assign(a_view).realize()
    self.assertListEqual(b.tolist(), [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

if __name__ == '__main__':
  unittest.main(verbosity=2)
