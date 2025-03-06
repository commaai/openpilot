# this will be the new test_ops for the next level
# schedule confirms the right things are capable of fusing
# NOTE: this has overlap with external_test_opt.py
# ruff: noqa: E501

import unittest
import numpy as np
import functools
from typing import List, Optional, Union, cast

from tinygrad import nn, dtypes, Device, Tensor
from tinygrad.device import is_dtype_supported
from tinygrad.dtype import DType
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View
from tinygrad.ops import PatternMatcher, UOp, Ops, UPat, graph_rewrite, track_rewrites, view_supported_devices
from tinygrad.helpers import CI, DEBUG, FUSE_ARANGE, GlobalCounters, flatten, getenv, SPLIT_REDUCEOP, unwrap, prod, Context
from tinygrad.codegen.kernel import Kernel, verify_ast
from tinygrad.engine.schedule import BUF_LIMIT, ScheduleContext, ScheduleItem, create_schedule, view_right, view_left, do_realize, remove_movement_ops
from tinygrad.engine.realize import CompiledRunner, get_runner, run_schedule
from extra.models.llama import precompute_freqs_cis

class KernelCountException(Exception): pass
def check_schedule(t:Union[Tensor, List[Tensor], UOp], allowed:int, to_prerealize:Optional[List[Tensor]]=None, filter_sink=True):
  if isinstance(t, Tensor): outs = t.lazydata.lbs
  elif isinstance(t, List): outs = flatten([r.lazydata.lbs for r in t])
  else: outs = [t]
  if to_prerealize:
    for pre in to_prerealize: pre.schedule()
  sched = create_schedule(outs)
  if filter_sink: sched = [s for s in sched if s.ast.op is Ops.SINK]
  if len(sched) != allowed:
    print(f"SCHEDULE ISSUE, expecting {allowed} got {len(sched)}")
    if DEBUG >= 3:
      for i,s in enumerate(sched):
        print("kernel", i+1)
        print(s.ast)
    raise KernelCountException(f"{len(sched)=} != {allowed}")
  # test the (sink) ops linearize
  for s in sched:
    if s.ast.op is not Ops.SINK: continue
    get_runner(s.bufs[0].device, s.ast)
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
  with Context(**kwargs): s = create_schedule([ret.lazydata, img.grad.lazydata, w.grad.lazydata])
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

class TestSchedule(unittest.TestCase):
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
    check_schedule(a, 2, filter_sink=False)

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
    reduceops = [x for si in schedule for x in si.ast.toposort if x.op is Ops.REDUCE_AXIS]
    assert len(reduceops) == 2

  def test_cache_reduce_multiple_children(self):
    x = Tensor.empty(32)
    y = Tensor.empty(4, 4)
    r0 = x.mean(axis=0, keepdim=True)
    r1 = (x - r0).sum(axis=0).div(2)
    out0 = r0 + y
    out1 = r1 + y
    schedule = check_schedule([out0, out1], 4)
    reduceops = [x for si in schedule for x in si.ast.toposort if x.op is Ops.REDUCE_AXIS]
    assert len(reduceops) == 2

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
    self.assertIs(a.lazydata.realized, b.lazydata.realized)

  # EMPTY and COPY are assigned to unique device Buffers

  def test_no_dedup_copy(self):
    src = Tensor.ones(4).contiguous().realize()
    a = src.clone()
    b = src.clone()
    sched = check_schedule([a, b], 2, filter_sink=False)
    run_schedule(sched)
    # a and b are assigned to different device Buffers
    self.assertIsNot(a.lazydata.realized, b.lazydata.realized)

  def test_no_dedup_empty(self):
    a = Tensor.empty((4,))
    b = Tensor.empty((4,))
    sched = check_schedule([a, b], 2, filter_sink=False)
    run_schedule(sched)
    self.assertIsNot(a.lazydata.realized, b.lazydata.realized)

  def test_fold_double_unary(self):
    y = Tensor.empty(2)
    out = y.sum(keepdim=True).sqrt().__neg__()
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

  def test_fold_conv_batchnorm_optim(self):
    # this is too high
    for optim, cnt in [(nn.optim.Adam, 18), (nn.optim.SGD, 15)]:
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
    old_training = Tensor.training
    Tensor.training = False

    in_planes, planes = 64, 64
    conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
    bn1 = nn.BatchNorm2d(planes)
    conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=1, bias=False)
    bn2 = nn.BatchNorm2d(planes)

    x = Tensor.empty(1, 64, 32, 32)
    out = bn1(conv1(x)).relu()
    out = bn2(conv2(out))
    out = (out + x).relu()
    check_schedule(out, 2, [conv1.weight, conv2.weight])
    Tensor.training = old_training

  def test_contiguous_while_contiguous(self):
    x = Tensor.empty(1, 64, 32, 32)
    out = x.contiguous()
    check_schedule(out, 1, filter_sink=False)

  def test_contiguous_while_not_contiguous(self):
    x = Tensor.empty(1, 64, 32, 32)
    out = x.permute(0,2,3,1).contiguous()
    check_schedule(out, 2, filter_sink=False)

  def test_fold_with_contiguous(self):
    a = Tensor.randn(16, 16, 16).realize()
    b = Tensor.randn(16, 16).realize()
    c = (a.sum(2).contiguous() + b).contiguous()
    check_schedule(c, 2)

  def test_double_from(self):
    x = Tensor([1,2,3,4])
    out = x.to('python')
    check_schedule(out, 0, filter_sink=False)

  def test_pow_const_tensor_simplified(self):
    x = Tensor([1,2,3,4])
    # NOTE: this does not test ** Tensor(2) is simpler in ast than ** Tensor(2.5)
    out = x ** Tensor(2)
    check_schedule(out, 1)

  def test_pow_const_tensor_to_zero(self):
    x = Tensor([1,2,3,4])
    out = x ** Tensor(0)
    # NOTE: this is ConstBuffer 0 + ConstBuffer 1
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

  # multireduce spec
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

  # multireduce spec
  def test_example_matmul(self):
    x = Tensor.eye(64, requires_grad=True)
    y = Tensor.eye(64, requires_grad=True)
    z = y.matmul(x).sum()
    z.backward()
    out = x.grad.contiguous()
    run_schedule(check_schedule(out, 2))
    np.testing.assert_allclose(out.numpy(), np.ones((64,64)))

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
    check_schedule(out, 2)  # TODO: this should be 1

  # multireduce spec
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

  # multireduce spec
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

  # multireduce spec
  def test_reduce_multiple_paths(self):
    Tensor.manual_seed(0)
    a = Tensor.randn(4, 4).realize()
    out0 = a.sum().exp2()
    # out1 has two paths to a.sum()
    out1 = a.sum() + out0
    run_schedule(check_schedule([out0, out1], 1))
    np.testing.assert_allclose(out0.numpy(), out0_np:=np.exp2(a.numpy().sum()), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(out1.numpy(), a.numpy().sum()+out0_np, atol=1e-4, rtol=1e-6)

  # multireduce spec
  def test_multireduce_reduce_multiple_paths(self):
    Tensor.manual_seed(0)
    a = Tensor.randn(4, 4).realize()
    out0 = a.sum().exp2()
    out1 = a.sum() + out0
    b = (a + out0 + out1)
    out2 = b.sum().exp2()
    out3 = b.sum() + out2
    # run_schedule(check_schedule([out0, out1, out2, out3], 1))
    run_schedule(check_schedule([out0, out1, out2, out3], 2))
    np.testing.assert_allclose(out0.numpy(), np_out0:=np.exp2(a.numpy().sum()), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(out1.numpy(), np_out1:=a.numpy().sum()+np_out0, atol=1e-4, rtol=1e-4)
    np_b = (a.numpy() + np_out0 + np_out1)
    np.testing.assert_allclose(out2.numpy(), np_out2:=np.exp2(np_b.sum()), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(out3.numpy(), np_b.sum()+np_out2, atol=1e-4, rtol=1e-4)

  # multireduce spec
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

  # multireduce spec
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

  # multireduce spec
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

  # multireduce spec
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

  # multireduce spec
  def test_std_multireduce_fusion(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(4, 32).realize()
    out = x.std(-1)
    run_schedule(check_schedule(out, 2))
    np.testing.assert_allclose(out.numpy(), x.numpy().std(axis=-1, ddof=1), atol=1e-4, rtol=1e-4)

  # multireduce spec
  def test_argmin_multireduce_fusion(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(4, 32).realize()
    out = x.argmin(-1)
    run_schedule(check_schedule(out, 3))
    np.testing.assert_equal(out.numpy(), x.numpy().argmin(axis=-1))

  # multireduce spec
  def test_argmax_multireduce_fusion(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(4, 32).realize()
    out = x.argmax(-1)
    run_schedule(check_schedule(out, 3))
    np.testing.assert_equal(out.numpy(), x.numpy().argmax(axis=-1))

  def test_scaled_dot_product_attention_multireduce_fusion(self):
    Tensor.manual_seed(0)
    q = Tensor.randn(32,8,16,64).realize()
    k = Tensor.randn(32,8,16,64).realize()
    v = Tensor.randn(32,8,16,64).realize()
    out = Tensor.scaled_dot_product_attention(q,k,v)
    run_schedule(check_schedule(out, 5))
    if getenv("CHECK", 1):
      import torch
      compare = torch.nn.functional.scaled_dot_product_attention(torch.tensor(q.numpy()),torch.tensor(k.numpy()),torch.tensor(v.numpy()))
      np.testing.assert_allclose(out.numpy(), compare.numpy(), atol=1e-6, rtol=1e-3)

  # multireduce spec
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

  # multireduce spec
  def test_reduce_expand_reduce_fusion(self):
    Tensor.manual_seed(0)
    a = Tensor.randn(4, 32).realize()
    out = (a+a.sum(-1, keepdim=True)).sum(-1)
    # run_schedule(check_schedule(out, 1))
    run_schedule(check_schedule(out, 2))
    np.testing.assert_allclose(out.numpy(), (a.numpy()+a.numpy().sum(axis=-1,keepdims=True)).sum(axis=-1), atol=1e-4, rtol=1e-4)

  # multireduce spec
  def test_reduce_expand_reduce_expand_fusion(self):
    Tensor.manual_seed(0)
    a = Tensor.randn(4, 32).realize()
    out = a+(a+a.sum(-1,keepdim=True)).sum(-1, keepdim=True)
    # run_schedule(check_schedule(out, 2))
    run_schedule(check_schedule(out, 3))
    np.testing.assert_allclose(out.numpy(), \
      a.numpy()+(a.numpy()+a.numpy().sum(axis=-1,keepdims=True)).sum(axis=-1,keepdims=True), atol=1e-4, rtol=1e-4)

  # multireduce spec
  def test_branching_reduces_and_expands_fusion(self):
    Tensor.manual_seed(0)
    a = Tensor.randn(4, 32).realize()
    out0 = a+a.sum(-1, keepdim=True)
    out1 = out0.sum(-1)
    # run_schedule(check_schedule(out, 2))
    run_schedule(check_schedule([out0, out1], 3))
    np.testing.assert_allclose(out0.numpy(), a.numpy()+a.numpy().sum(axis=-1,keepdims=True), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(out1.numpy(), (a.numpy()+a.numpy().sum(axis=-1,keepdims=True)).sum(axis=-1), atol=1e-4, rtol=1e-4)

  # multireduce spec
  def test_multireduce_fusion_simple_sequential(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(4, 32).realize()
    y = Tensor.randn(4, 32).realize()
    out = (y + x.sum(axis=-1, keepdim=True)).sum(axis=-1)
    # run_schedule(check_schedule(out, 1))
    run_schedule(check_schedule(out, 2))
    np.testing.assert_allclose(out.numpy(), (y.numpy() + x.numpy().sum(axis=-1, keepdims=True)).sum(axis=-1), atol=1e-4, rtol=1e-4)

  # multireduce spec
  def test_multireduce_fusion_simple_parallel(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(4, 32).realize()
    y = Tensor.randn(4, 32).realize()
    out = y.sum(axis=-1) + x.sum(axis=-1)
    # run_schedule(check_schedule(out, 1))
    run_schedule(check_schedule(out, 2))
    np.testing.assert_allclose(out.numpy(), y.numpy().sum(axis=-1) + x.numpy().sum(axis=-1), atol=1e-4, rtol=1e-4)

  # multireduce spec
  def test_multireduce_fusion_sequential(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(4, 32).realize()
    out = x.std(-1)
    # run_schedule(check_schedule(out, 1))
    run_schedule(check_schedule(out, 2))
    np.testing.assert_allclose(out.numpy(), x.numpy().std(axis=-1, ddof=1), atol=1e-4, rtol=1e-4)

  # multireduce spec
  def test_multireduce_fusion_parallel(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(4, 32).realize()
    y = Tensor.randn(4, 32).realize()
    out = x.std(-1) + y.std(-1)
    # run_schedule(check_schedule(out, 1))
    run_schedule(check_schedule(out, 4))
    np.testing.assert_allclose(out.numpy(), x.numpy().std(axis=-1, ddof=1) + y.numpy().std(axis=-1, ddof=1), atol=1e-4, rtol=1e-4)

  # multireduce spec
  def test_multireduce_diffops_sequential(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(4, 32).realize()
    out = (x - x.max(-1, keepdim=True)).sum(-1)
    # run_schedule(check_schedule(out, 1))
    run_schedule(check_schedule(out, 2))
    np.testing.assert_allclose(out.numpy(), (x.numpy() - x.numpy().max(axis=-1, keepdims=True)).sum(axis=-1), atol=1e-4, rtol=1e-4)

  # multireduce spec
  def test_multireduce_fusion_diffops_parallel(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(4, 32).realize()
    y = Tensor.randn(4, 32).realize()
    out = x.sum(-1) + y.max(-1)
    # run_schedule(check_schedule(out, 1))
    run_schedule(check_schedule(out, 2))
    np.testing.assert_allclose(out.numpy(), x.numpy().sum(axis=-1) + y.numpy().max(axis=-1), atol=1e-4, rtol=1e-4)

  # multireduce spec
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

  # multireduce spec
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
    x, y, z, m = (Tensor.empty(32, 8, 16, 16) for _ in range(4))
    out = Tensor.scaled_dot_product_attention(x, y, z, attn_mask=m, is_causal=True)
    check_schedule(out, 6)

  def test_adam_step_fusion(self):
    with Tensor.train():
      x = Tensor.empty(4, 64, 768)
      layer = nn.Linear(768, 768*4)
      _realize_weights(layer)
      opt = nn.optim.Adam(nn.state.get_parameters(layer), lr=1e-4)
      layer(x).relu().sum().backward()
      check_schedule(opt.schedule_step(), 10)

  def test_adam_conv_fuse(self):
    with Tensor.train():
      img = Tensor.empty(2,3,4,4)
      c1 = nn.Conv2d(3,32,3)
      _realize_weights(c1)
      opt = nn.optim.Adam(nn.state.get_parameters(c1), lr=1e-4)
      opt.zero_grad()
      c1(img).relu().sum().backward()
      check_schedule(opt.schedule_step(), 10)

  def test_adam_2convs_fuse(self):
    with Tensor.train():
      img = Tensor.empty(2,3,4,4)
      c1 = nn.Conv2d(3,16,3,bias=False)
      c2 = nn.Conv2d(16,32,2,bias=False)
      _realize_weights([c1, c2])
      opt = nn.optim.Adam(nn.state.get_parameters([c1, c2]), lr=1e-4)
      opt.zero_grad()
      c2(c1(img).relu()).relu().sum().backward()
      check_schedule(opt.schedule_step(), 13)

  def test_sgd_conv_fuse(self):
    with Tensor.train():
      img = Tensor.empty(2,3,4,4)
      c1 = nn.Conv2d(3,32,3)
      _realize_weights(c1)
      opt = nn.optim.SGD(nn.state.get_parameters(c1))
      opt.zero_grad()
      c1(img).relu().sum().backward()
      check_schedule(opt.schedule_step(), 5)

  def test_sgd_2convs_fuse(self):
    with Tensor.train():
      img = Tensor.empty(2,3,4,4)
      c1 = nn.Conv2d(3,16,3,bias=False)
      c2 = nn.Conv2d(16,32,2,bias=False)
      _realize_weights([c1, c2])
      opt = nn.optim.SGD(nn.state.get_parameters([c1, c2]))
      opt.zero_grad()
      c2(c1(img).relu()).relu().sum().backward()
      check_schedule(opt.schedule_step(), 8)

  def test_fold_2convs_sgd_nesterov_momentum_wd(self):
    with Tensor.train():
      img = Tensor.empty(2,3,4,4)
      c1 = nn.Conv2d(3,16,3,bias=False)
      c2 = nn.Conv2d(16,32,2,bias=False)
      _realize_weights([c1, c2])
      opt = nn.optim.SGD(nn.state.get_parameters([c1, c2]), nesterov=True, momentum=0.9, weight_decay=0.1)
      opt.zero_grad()
      c2(c1(img).relu()).relu().sum().backward()
      check_schedule(opt.schedule_step(), 10)

  def test_sgd_4convs_fuse(self):
    with Tensor.train():
      img = Tensor.empty(2,3,64,64)
      c1 = nn.Conv2d(3,4,3,bias=False)
      c2 = nn.Conv2d(4,8,3,bias=False)
      c3 = nn.Conv2d(8,16,3,bias=False)
      c4 = nn.Conv2d(16,32,3,bias=False)
      _realize_weights([c1, c2, c3, c4])
      opt = nn.optim.SGD(nn.state.get_parameters([c1, c2, c3, c4]))
      opt.zero_grad()
      c4(c3(c2(c1(img).relu()).relu()).relu()).relu().sum().backward()
      check_schedule(opt.schedule_step(), 18)

  def test_sgd_4convs_fuse_conv_bw(self):
    with Tensor.train():
      img = Tensor.empty(2,3,64,64)
      c1 = nn.Conv2d(3,4,3,bias=False)
      c2 = nn.Conv2d(4,8,3,bias=False)
      c3 = nn.Conv2d(8,16,3,bias=False)
      c4 = nn.Conv2d(16,32,3,bias=False)
      _realize_weights([c1, c2, c3, c4])
      opt = nn.optim.SGD(nn.state.get_parameters([c1, c2, c3, c4]))
      opt.zero_grad()
      c4(c3(c2(c1(img).relu()).relu()).relu()).relu().sum().backward()
      with Context(FUSE_CONV_BW=1): check_schedule(opt.schedule_step(), 15)

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
    sched = check_schedule([a, b], 1)
    for si in sched[:-2]: assert all(out.dtype == dtypes.half for out in si.outputs)

    # reduce
    a = z.sum(axis=0).half().float().sum(axis=0)
    sched = check_schedule(a, 2)
    for si in sched[:-1]: assert all(out.dtype == dtypes.half for out in si.outputs)

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
    self.assertIs(schedule[0].ast.src[0].src[2].op, Ops.ADD)

  # multireduce spec
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
    self.assertIs(schedule[0].ast.src[0].src[2].op, Ops.ADD)

  # multireduce spec
  def test_multireduce_push_permute_chase(self):
    Tensor.manual_seed(0)
    a = Tensor.randn(4, 4, 4).realize()
    b = Tensor.randn(4, 4).realize()
    r = a.sum(2) + b
    d = r.T * 4
    e = r * (d + a).sum(2)
    schedule = check_schedule([d, e], 3) # make sure it doesn't fuse
    self.assertIs(schedule[0].ast.src[0].src[2].op, Ops.ADD)
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
    self.assertIs(schedule[0].ast.src[0].src[2].op, Ops.ADD)

  # multireduce spec
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
    self.assertIs(schedule[0].ast.src[0].src[2].op, Ops.ADD)
    run_schedule(schedule)
    np.testing.assert_allclose(out.numpy(), (a.numpy().sum(1) + c.numpy())[:4] * b.numpy() + d.numpy().sum(1)[:4], atol=1e-4, rtol=1e-4)

  def test_midreduce_nochase(self):
    a = Tensor.empty(16, 16)
    b = (a.sum(0) + a.max(1)) + 2
    schedule = check_schedule(b, 2)
    self.assertIs(schedule[0].ast.src[0].src[2].op, Ops.REDUCE_AXIS)

  # multireduce spec
  def test_multireduce_midreduce_nochase(self):
    Tensor.manual_seed(0)
    a = Tensor.randn(16, 16).realize()
    b = (a.sum(0)+a.max(0) + a.max(1)+a.sum(1)) + 2
    # schedule = check_schedule(b, 2)
    schedule = check_schedule(b, 4)
    self.assertIs(schedule[0].ast.src[0].src[2].op, Ops.REDUCE_AXIS)
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

  # multireduce spec
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

  # multireduce spec
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

  @unittest.skipIf(Device.DEFAULT not in view_supported_devices, "subbuffer not supported")
  def test_bitcast_subbufer(self):
    x = cast(UOp, Tensor.empty(1, dtype=dtypes.float32).realize().lazydata)
    a = x.alu(Ops.EXP2).cast(dtypes.int32, True, allow_buffer_view=True)
    b = x.cast(dtypes.int32, True, allow_buffer_view=True)
    b = a.alu(Ops.ADD, b)
    check_schedule(b, 2) # this should fuse when it makes sense

  def test_bitcast_disable_subbufer(self):
    x = cast(UOp, Tensor.empty(1, dtype=dtypes.float32).realize().lazydata)
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
  def test_conv2d_fused(self): _test_conv2d(6, FUSE_CONV_BW=1)

  @unittest.skipUnless(is_dtype_supported(dtypes.half), "need half")
  def test_conv2d_half(self): _test_conv2d(7, dtype=dtypes.half)
  @unittest.skipUnless(is_dtype_supported(dtypes.half), "need half")
  @unittest.expectedFailure
  def test_conv2d_fused_half(self): _test_conv2d(5, dtype=dtypes.half)

  def _test_buf_cnt(self, cnt:int, allowed:int):
    if (m:=BUF_LIMIT.get(Device.DEFAULT)) is None or m != 32: self.skipTest(f"test needs a buf_max of 32 {Device.DEFAULT}")
    alu = functools.reduce(lambda x,y: x+y, [Tensor.ones((1, 1)).contiguous().realize() for _ in range(cnt-1)])
    s = alu.schedule()
    assert len(s) == allowed
    run_schedule(s)
    expected = functools.reduce(lambda x,y: x+y, [np.ones((1, 1)) for _ in range(cnt-1)])
    np.testing.assert_equal(alu.numpy(), expected)

  def test_buf_cnt_at_limit(self): self._test_buf_cnt(31, allowed=1)
  @unittest.expectedFailure
  def test_buf_cnt_over_limit(self): self._test_buf_cnt(32, allowed=2)
  @unittest.expectedFailure
  def test_buf_cnt_over_limit_alt(self): self._test_buf_cnt(63, allowed=3)

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
    constv = Tensor.empty(2, 2).lazydata.const_like(10)
    self.assertEqual(len(create_schedule([constv])), 0)

  def test_const_schedule_contig(self):
    constv = Tensor.empty(2, 2).lazydata.const_like(10).contiguous()
    self.assertEqual(len(create_schedule([constv])), 1)

  @unittest.skipIf(Device.DEFAULT != "GPU", "image only supported on GPU")
  def test_image_matmul(self):
    with Context(IMAGE=2):
      x = Tensor.randn((9, 9)).realize()
      y = Tensor.randn((9, 9)).realize()
      out = x@y
      run_schedule(check_schedule(out, 4))
      np.testing.assert_allclose(out.numpy(), x.numpy()@y.numpy(), atol=1e-4, rtol=1e-4)

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

class TestIndexing(unittest.TestCase):
  def check_schedule(self, xt:Union[Tensor,List[Tensor]], cnt:int):
    with Context(FUSE_ARANGE=getenv("FUSE_ARANGE", 1)):
      lst = [xt] if isinstance(xt, Tensor) else xt
      s = Tensor.schedule(*lst)
      kernels = [si for si in s if si.ast.op is Ops.SINK]
      for si in kernels: verify_ast(si.ast)
      run_schedule(s.copy())
      if FUSE_ARANGE: self.assertEqual(len(kernels), cnt)
    return s

  def test_simple_indexing(self):
    X = Tensor.randn(10, 10).realize()
    idxs = Tensor([0, 2]).realize()
    xt = X[idxs]
    self.check_schedule(xt, 2)
    np.testing.assert_equal(xt.numpy(), X.numpy()[idxs.numpy()])

  @unittest.skip("TODO: support pads in graph_rewrite")
  def test_simple_indexing_alt(self):
    X = Tensor.arange(16).reshape(4, 4)
    xt = X[[1, 2], [1, 2]]
    self.check_schedule(xt, 3)
    np.testing.assert_equal(xt.numpy(), (np.arange(16).reshape(4, 4))[[1, 2], [1, 2]])

  def test_advanced_indexing(self):
    X = Tensor.arange(10)+1
    xt = X[[0]]
    self.check_schedule(xt, 2)
    np.testing.assert_equal(xt.numpy(), (np.arange(10)+1)[[0]])

  @unittest.expectedFailure
  def test_advanced_indexing_alt(self):
    X = Tensor.arange(6).reshape(3, 2)+1
    xt = X[[Tensor([2]), Tensor([1])]]
    self.check_schedule(xt, 6)
    np.testing.assert_equal(xt.numpy(), 6)

  @unittest.skip("TODO: support pads in graph_rewrite")
  def test_advanced_simple_indexing_combined(self):
    X = Tensor.arange(16).reshape(4, 4)
    xt = X[1:2, [1, 2]]
    self.check_schedule(xt, 2)

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
    out = a+b
    self.check_schedule(out, 1)
    np.testing.assert_allclose(out.numpy(), np.arange(4)+b.numpy())

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
    a = (Tensor.arange(4,)*x).T
    self.check_schedule(a, 1)
    np.testing.assert_equal(a.numpy(), (np.arange(4)*x.numpy()).T)

  def test_arange_transposed_descendants(self):
    Tensor.manual_seed(0)
    x = Tensor.randint(4, 1).realize()
    a = (Tensor.arange(4,)*x).T
    b = Tensor.randint(4, 4).realize()
    out = a+b
    self.check_schedule(out, 1)
    np.testing.assert_equal(out.numpy(), (np.arange(4)*x.numpy()).T+b.numpy())

  def test_arange_index(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(5, 2).realize()
    a = Tensor.arange(10)
    out = (x + a[2]).sum()
    self.check_schedule(out, 1)
    np.testing.assert_allclose(out.numpy(), (x.numpy()+np.arange(10)[2]).sum(), atol=1e-5, rtol=1e-6)

  def test_arange_index_contiguous(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(5, 2).realize()
    a = Tensor.arange(10).contiguous()
    out = (x + a[2]).sum()
    self.check_schedule(out, 2)
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
    self.check_schedule(out, 2)
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

  @unittest.skipUnless(Device.DEFAULT in view_supported_devices, "need view")
  def test_arange_view_op(self):
    a = Tensor.arange(12).reshape(4, 3).shrink(((1, 2), (1, 3))).contiguous()
    sched = self.check_schedule(a, 1)
    self.assertIs(sched[1].ast.op, Ops.BUFFER_VIEW)
    np.testing.assert_equal(a.numpy(), [[4, 5]])

  @unittest.skipIf(Device.DEFAULT == "CLANG", "tests copy from ext device")
  def test_arange_shrink_copy(self):
    a = Tensor.arange(12).reshape(4, 3).shrink(((1, 2), (1, 3))).to("CLANG")
    sched = self.check_schedule(a, 1)
    self.assertIs(sched[-1].ast.op, Ops.COPY)
    np.testing.assert_equal(a.numpy(), [[4, 5]])

  @unittest.skipIf(Device.DEFAULT == "CLANG", "tests copy from ext device")
  def test_arange_expand_copy(self):
    a = Tensor.arange(4).reshape(2, 2, 1).expand(2, 2, 2).contiguous().to("CLANG")
    sched = self.check_schedule(a, 1)
    self.assertIs(sched[1].ast.op, Ops.COPY)
    self.assertIs(sched[0].ast.src[0].src[2].op, Ops.ADD)
    np.testing.assert_equal(a.numpy(), [[[0, 0], [1, 1]], [[2, 2], [3, 3]]])

  @unittest.skip("TODO: support pads in graph_rewrite")
  #@unittest.skipUnless(is_dtype_supported(dtypes.half), "need half")
  def test_precompute_freqs_cis(self):
    args = {"dim":32 if CI else 128, "end":2048 if CI else 8192, "theta":10000, "dtype":dtypes.half}
    fused = precompute_freqs_cis(**args)
    self.check_schedule(fused, 1)
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

  def test_arange_fuse_grouped_children(self):
    X = Tensor.randn(4, 4).realize()
    r = (X+Tensor.arange(16).reshape(4, 4)).sum()
    out0 = r+2
    out1 = r+3
    self.check_schedule([out0, out1], 1)
    r_ref = (X.numpy()+np.arange(16).reshape(4, 4)).sum()
    np.testing.assert_allclose(out0.numpy(), r_ref+2, rtol=2e-7)
    np.testing.assert_allclose(out1.numpy(), r_ref+3, rtol=2e-7)

  @unittest.expectedFailure
  def test_fold_arange_view(self):
    X = Tensor.randn(4, 4).realize()
    r = (X+Tensor.arange(16).reshape(4, 4).contiguous()).sum(1, keepdim=True)
    self.check_schedule([r], 1)
    np.testing.assert_allclose(r.numpy(), (X.numpy()+np.arange(16).reshape(4, 4)).sum(1, keepdims=True))

  @unittest.expectedFailure
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
    ast = a.schedule()[0].ast
    swizzle = ast.src[0].src[2].reshape((4, 1))
    new_uop = swizzle_rewrite(swizzle)
    self.assertEqual(new_uop.st, ShapeTracker.from_shape((4,)).reshape((4, 1)))
    self.assertEqual(swizzle_cnt(new_uop), 0)

  def test_no_rewrite_elementwise(self):
    bufs = [UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), (), i) for i in range(3)]
    ld1 = UOp(Ops.LOAD, dtypes.int, (bufs[1], ShapeTracker.from_shape((32, 32)).to_uop()))
    ld2 = UOp(Ops.LOAD, dtypes.int, (bufs[2], ShapeTracker.from_shape((32, 32)).to_uop()))
    sink = UOp(Ops.SINK, dtypes.void, (UOp(Ops.STORE, dtypes.void, (bufs[0], ShapeTracker.from_shape((32, 32)).to_uop(), ld1+ld2)),))
    rsink = graph_rewrite(sink, view_right)
    self.assertEqual(rsink.key, sink.key)

  def test_simple_store_reshape(self):
    bufs = [UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), (), i) for i in range(2)]
    ld = UOp(Ops.LOAD, dtypes.int, (bufs[1], ShapeTracker.from_shape((32, 32)).to_uop()))
    r = UOp(Ops.REDUCE_AXIS, dtypes.int, (ld,), (Ops.ADD, (0, 1)))
    r = UOp(Ops.VIEW, dtypes.int, (r,), ShapeTracker.from_shape(()))
    r = r + 2
    sink = UOp(Ops.SINK, dtypes.void, (UOp(Ops.STORE, dtypes.void, (bufs[0], ShapeTracker.from_shape(()).to_uop(), r)),))
    rsink = graph_rewrite(sink, view_right)
    # this AST first needs to swizzle, but it doesn't have implicit movementops
    with self.assertRaisesRegex(AssertionError, "swizzle"): verify_ast(sink)
    verify_ast(rsink)

  def test_no_reshape_reduceop(self):
    bufs = [UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), (), i) for i in range(2)]
    ld = UOp(Ops.LOAD, dtypes.int, (bufs[1], ShapeTracker.from_shape((32, 32)).to_uop()))
    r = UOp(Ops.REDUCE_AXIS, dtypes.int, (ld,), (Ops.ADD, (0, 1)))
    sink = UOp(Ops.SINK, dtypes.void, (UOp(Ops.STORE, dtypes.void, (bufs[0], ShapeTracker.from_shape((1, 1)).to_uop(), r)),))
    rsink = graph_rewrite(sink, view_right)
    verify_ast(sink)
    self.assertEqual(sink.key, rsink.key)

@track_rewrites(named=True)
def swizzle_rewrite(u:UOp) -> UOp: return graph_rewrite(graph_rewrite(u, view_left), view_right)

def swizzle_cnt(u:UOp) -> int: return len([x for x in u.toposort if x.op is Ops.VIEW and len(x.src) != 0])

class TestSwizzle(unittest.TestCase):
  def test_swizzle_simple(self):
    sink = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 1), strides=(0, 0), offset=0, mask=None, contiguous=True),)), src=()), # noqa: E501
        UOp(Ops.REDUCE_AXIS, dtypes.int, arg=(Ops.ADD, (0, 1)), src=(
          UOp(Ops.ADD, dtypes.int, arg=None, src=(
            UOp(Ops.VIEW, dtypes.int, arg=ShapeTracker(views=(View(shape=(32, 32), strides=(0, 0), offset=0, mask=None, contiguous=False),)), src=( # noqa E501
              UOp(Ops.REDUCE_AXIS, dtypes.int, arg=(Ops.ADD, (0, 1)), src=(
                UOp(Ops.LOAD, dtypes.int, arg=None, src=(
                  x8:=UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), arg=1, src=()),
                  UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(32, 32), strides=(32, 1), offset=0, mask=None, contiguous=True),)), src=()),)),)),)), # noqa E501
            UOp(Ops.LOAD, dtypes.int, arg=None, src=(
               x8,
              UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(32, 32), strides=(32, 1), offset=0, mask=None, contiguous=True),)), src=()),)),)),)),)),)) # noqa E501
    sink = swizzle_rewrite(sink)
    k = Kernel(sink)
    p = k.to_program()
    a = Tensor.randint(32, 32).realize()
    b = Tensor.empty((), dtype=dtypes.int).realize()
    CompiledRunner(p).exec([b.lazydata.buffer, a.lazydata.buffer])
    expected_out = (a.numpy() + a.numpy().sum()).sum()
    np.testing.assert_equal(b.numpy(), expected_out)

  def test_single_swizzle(self):
    # ast in tensor style
    a = Tensor.randint(4,).realize()
    expected_out = a.numpy().sum(0)+1
    # LazyBuffer to pre-rewrite AST
    bufs = [UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), (), i) for i in range(2)]
    ld = UOp(Ops.LOAD, dtypes.int, (bufs[1], ShapeTracker.from_shape((4,)).to_uop()))
    r = UOp(Ops.REDUCE_AXIS, dtypes.int, (ld,), (Ops.ADD, (0,)))
    swizzle_r = UOp(Ops.VIEW, dtypes.int, (r,), unwrap(r.st).reshape(()))
    alu = swizzle_r+1
    sink = UOp(Ops.SINK, dtypes.void, (UOp(Ops.STORE, dtypes.void, (bufs[0], ShapeTracker.from_shape(()).to_uop(), alu,),),))
    # graph rewrite
    sink = swizzle_rewrite(sink)
    # verify output
    k = Kernel(sink)
    p = k.to_program()
    b = Tensor.empty((1,), dtype=dtypes.int).realize()
    CompiledRunner(p).exec([b.lazydata.buffer, a.lazydata.buffer])
    np.testing.assert_equal(b.numpy(), expected_out)

  def test_double_swizzle_possible(self):
    # ast in tensor style
    Tensor.manual_seed(0)
    a = Tensor.randint(4,).realize()
    b = Tensor.randint(4,).realize()
    expected_out = a.numpy().sum(0)+b.numpy().sum(0)+2
    # LazyBuffer to pre-rewrite AST
    bufs = [UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), (), i) for i in range(3)]
    ld1 = UOp(Ops.LOAD, dtypes.int, (bufs[1], ShapeTracker.from_shape((4,)).to_uop()))
    r1 = UOp(Ops.REDUCE_AXIS, dtypes.int, (ld1,), (Ops.ADD, (0,)))
    ld2 = UOp(Ops.LOAD, dtypes.int, (bufs[2], ShapeTracker.from_shape((4,)).to_uop()))
    r2 = UOp(Ops.REDUCE_AXIS, dtypes.int, (ld2,), (Ops.ADD, (0,)))
    alu = UOp(Ops.VIEW, r1.dtype, (r1,), ShapeTracker.from_shape(()))+UOp(Ops.VIEW, r2.dtype, (r2,), ShapeTracker.from_shape(()))
    sink = UOp(Ops.SINK, dtypes.void, (UOp(Ops.STORE, dtypes.void, (bufs[0], ShapeTracker.from_shape(()).to_uop(), alu+2,),),)) # noqa: E501
    # graph rewrite
    sink = swizzle_rewrite(sink)
    # verify output
    k = Kernel(sink)
    p = k.to_program()
    c = Tensor.empty((1,), dtype=dtypes.int).realize()
    CompiledRunner(p).exec([c.lazydata.buffer, a.lazydata.buffer, b.lazydata.buffer])
    np.testing.assert_equal(c.numpy(), expected_out)

  def test_swizzle_rewrite_alt(self):
    swizzle = UOp(Ops.VIEW, dtypes.float, arg=ShapeTracker(views=(View(shape=(2, 3, 3, 65, 3, 65), strides=(103788, 34596, 3, 558, 1, 9), offset=0, mask=((0, 2), (0, 3), (0, 3), (0, 62), (0, 3), (0, 62)), contiguous=False), View(shape=(2, 3, 256, 256), strides=(114075, 38025, 195, 1), offset=0, mask=((0, 2), (0, 3), (0, 195), (0, 195)), contiguous=False), View(shape=(1, 2, 1, 3, 4, 64, 4, 64), strides=(0, 196608, 0, 65536, 16384, 256, 64, 1), offset=0, mask=None, contiguous=True))), src=( # noqa: E501
  UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (3,)), src=(
    UOp(Ops.LOAD, dtypes.float, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=1, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=(ld_st:=ShapeTracker(views=(View(shape=(2, 1, 3, 16, 62, 62, 3, 3), strides=(0, 0, 9, 27, 0, 0, 3, 1), offset=0, mask=None, contiguous=False),))), src=()),)),)),)) # noqa: E501
    # there's an UNROLL pushing through the REDUCE_AXIS
    self.assertGreater(prod(swizzle.st.shape), prod(swizzle.src[0].st.shape))
    ret = swizzle_rewrite(swizzle)
    # UNROLL is rewritten
    self.assertEqual(prod(ret.st.shape), prod(ret.src[0].st.shape))
    # and pushed to the LOAD
    new_load_st = unwrap([x for x in ret.toposort if x.op is Ops.VIEW][0].st)
    self.assertGreater(prod(new_load_st.shape), prod(ld_st.shape))
    self.assertEqual(new_load_st.views[0].strides, (0, 9, 3, 0, 1, 0, 27))

  def test_permute_rewrite(self):
    sink = UOp(Ops.STORE, dtypes.void, arg=None, src=(
        x1:=UOp(Ops.BUFFER, dtypes.float, arg=(1, ('METAL', 16384, dtypes.float)), src=()),
        x2:=UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 32, 32, 16), strides=(0, 512, 16, 1), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.CONTIGUOUS, dtypes.float, arg=None, src=(
           x1,
          UOp(Ops.VIEW, dtypes.float, arg=ShapeTracker(views=(View(shape=(1, 32, 32, 16), strides=(0, 32, 1, 1024), offset=0, mask=None, contiguous=False),)), src=(
            UOp(Ops.ADD, dtypes.float, arg=None, src=(
              UOp(Ops.ADD, dtypes.float, arg=None, src=(
                UOp(Ops.VIEW, dtypes.float, arg=ShapeTracker(views=(View(shape=(1, 16, 32, 32), strides=(0, 1, 512, 16), offset=0, mask=None, contiguous=False),)), src=(
                  UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (7, 8)), src=(
                    UOp(Ops.MUL, dtypes.float, arg=None, src=(
                      UOp(Ops.VIEW, dtypes.float, arg=ShapeTracker(views=(View(shape=(1, 32, 32, 1, 1, 4, 4, 4, 4, 1, 1), strides=(0, 512, 16, 0, 0, 0, 0, 4, 1, 0, 0), offset=0, mask=None, contiguous=False),)), src=(
                        x11:=UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                          UOp(Ops.BUFFER, dtypes.float, arg=(2, ('METAL', 16384, dtypes.float)), src=()),
                           x2,)),)),
                      UOp(Ops.VIEW, dtypes.float, arg=ShapeTracker(views=(View(shape=(1, 32, 32, 1, 1, 4, 4, 4, 4, 1, 1), strides=(0, 0, 0, 0, 0, 64, 1, 16, 4, 0, 0), offset=0, mask=None, contiguous=False),)), src=(
                        UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                          UOp(Ops.BUFFER, dtypes.float, arg=(8, ('METAL', 256, dtypes.float)), src=()),
                          UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(4, 1, 4, 1, 4, 4), strides=(64, 0, 16, 0, 4, 1), offset=0, mask=None, contiguous=True),)), src=()),)),)),)),)),)),
                UOp(Ops.VIEW, dtypes.float, arg=ShapeTracker(views=(View(shape=(1, 16, 32, 32), strides=(0, 1, 0, 0), offset=0, mask=None, contiguous=False),)), src=(
                  UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                    UOp(Ops.BUFFER, dtypes.float, arg=(10, ('METAL', 16, dtypes.float)), src=()),
                    UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(16,), strides=(1,), offset=0, mask=None, contiguous=True),)), src=()),)),)),)),
              UOp(Ops.VIEW, dtypes.float, arg=ShapeTracker(views=(View(shape=(1, 16, 32, 32), strides=(0, 1, 512, 16), offset=0, mask=None, contiguous=False),)), src=(
                 x11,)),)),)),)),))
    ret = swizzle_rewrite(sink)
    self.assertEqual(swizzle_cnt(ret), 0)

  @unittest.expectedFailure
  def test_fuse_conv2_relu_bw(self):
    # fuse (relu bw, conv2d, conv2d bw, relu)
    sink = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.BUFFER, dtypes.float, arg=(10, ('METAL', 128, dtypes.float)), src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(2, 16, 2, 2), strides=(64, 4, 2, 1), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.MUL, dtypes.float, arg=None, src=(
          UOp(Ops.CAST, dtypes.float, arg=None, src=(
            UOp(Ops.CMPLT, dtypes.bool, arg=None, src=(
              x6:=UOp(Ops.WHERE, dtypes.float, arg=None, src=(
                UOp(Ops.VALID, dtypes.bool, arg=None, src=(
                  UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(2, 16, 2, 2), strides=(0, 0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),
                x9:=UOp(Ops.CONST, dtypes.float, arg=0.0, src=()),
                 x9,)),
              UOp(Ops.MAX, dtypes.float, arg=None, src=(
                UOp(Ops.VIEW, dtypes.float, arg=ShapeTracker(views=(View(shape=(2, 16, 2, 2), strides=(64, 4, 2, 1), offset=0, mask=None, contiguous=True),)), src=(
                  UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (5, 6, 7)), src=(
                    UOp(Ops.MUL, dtypes.float, arg=None, src=(
                      UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                        UOp(Ops.BUFFER, dtypes.float, arg=(9, ('METAL', 96, dtypes.float)), src=()),
                        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(2, 1, 16, 2, 2, 3, 3, 3), strides=(48, 0, 0, 4, 1, 16, 4, 1), offset=0, mask=None, contiguous=False),)), src=()),)),
                      UOp(Ops.PRELOAD, dtypes.float, arg=None, src=(
                        UOp(Ops.BUFFER, dtypes.float, arg=(16, ('METAL', 432, dtypes.float)), src=()),
                        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(2, 1, 16, 2, 2, 3, 3, 3), strides=(0, 0, 27, 0, 0, 9, 3, 1), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),)),
                 x6,)),)),)),
          UOp(Ops.VIEW, dtypes.float, arg=ShapeTracker(views=(View(shape=(2, 16, 2, 2), strides=(64, 4, 2, 1), offset=0, mask=None, contiguous=True),)), src=(
            UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (4, 6)), src=(
              UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                UOp(Ops.BUFFER, dtypes.float, arg=(18, ('METAL', 128, dtypes.float)), src=()),
                UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(2, 16, 2, 3, 2, 3), strides=(64, 4, 2, 0, 1, 0), offset=0, mask=((0, 2), (0, 16), (0, 2), (0, 1), (0, 2), (0, 1)), contiguous=False), View(shape=(1, 2, 1, 16, 3, 2, 3, 2), strides=(0, 576, 0, 36, 12, 6, 2, 1), offset=0, mask=None, contiguous=True))), src=()),)),)),)),)),)),))
    ret = swizzle_rewrite(sink)
    self.assertEqual(swizzle_cnt(ret), 0)

  @unittest.expectedFailure
  def test_swizzle_failure_permute(self):
    sink = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.BUFFER, dtypes.float, arg=(20, ('METAL', 65, dtypes.float)), src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 65), strides=(0, 1), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.ADD, dtypes.float, arg=None, src=(
          UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (0,)), src=(
            UOp(Ops.ADD, dtypes.float, arg=None, src=(
              x6:=UOp(Ops.MUL, dtypes.float, arg=None, src=(
                UOp(Ops.ADD, dtypes.float, arg=None, src=(
                  UOp(Ops.PRELOAD, dtypes.float, arg=None, src=(
                    UOp(Ops.BUFFER, dtypes.float, arg=(8, ('METAL', 2925, dtypes.float)), src=()),
                    x10:=UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(45, 65), strides=(65, 1), offset=0, mask=None, contiguous=True),)), src=()),)),
                  UOp(Ops.WHERE, dtypes.float, arg=None, src=(
                    x12:=UOp(Ops.VALID, dtypes.bool, arg=None, src=(
                      UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(45, 65), strides=(0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),
                    UOp(Ops.CONST, dtypes.float, arg=1.0, src=()),
                    x15:=UOp(Ops.CONST, dtypes.float, arg=0.0, src=()),)),)),
                UOp(Ops.WHERE, dtypes.float, arg=None, src=(
                   x12,
                  UOp(Ops.CONST, dtypes.float, arg=0.0003418803389649838, src=()),
                   x15,)),)),
               x6,)),)),
          UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (0,)), src=(
            UOp(Ops.MUL, dtypes.float, arg=None, src=(
              UOp(Ops.WHERE, dtypes.float, arg=None, src=(
                 x12,
                UOp(Ops.CONST, dtypes.float, arg=-1.0, src=()),
                 x15,)),
              UOp(Ops.MUL, dtypes.float, arg=None, src=(
                UOp(Ops.PRELOAD, dtypes.float, arg=None, src=(
                  UOp(Ops.BUFFER, dtypes.float, arg=(2, ('METAL', 2925, dtypes.float)), src=()),
                   x10,)),
                UOp(Ops.VIEW, dtypes.float, arg=ShapeTracker(views=(View(shape=(45, 65), strides=(1, 89), offset=44, mask=None, contiguous=False),)), src=(
                  UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (2,)), src=(
                    UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                      UOp(Ops.BUFFER, dtypes.float, arg=(4, ('METAL', 2925, dtypes.float)), src=()),
                      UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(65, 45, 90), strides=(1, 0, 65), offset=0, mask=((0, 65), (0, 45), (0, 45)), contiguous=False), View(shape=(65, 4094), strides=(4050, 1), offset=0, mask=((0, 65), (0, 4050)), contiguous=False), View(shape=(1, 65, 46, 89), strides=(0, 4094, 89, 1), offset=0, mask=None, contiguous=True))), src=()),)),)),)),)),)),)),)),)),))
    ret = swizzle_rewrite(sink)
    self.assertEqual(swizzle_cnt(ret), 0)

  def test_non_contiguous_view_simplify(self):
    st = ShapeTracker(views=(View(shape=(2048, 2048), strides=(1, 2048), offset=0, mask=None, contiguous=False),))
    a = UOp(Ops.LOAD, dtypes.char, (UOp.new_buffer(Device.DEFAULT, 4194304, dtypes.char), st.to_uop()))
    ret = swizzle_rewrite(a.view(st))
    self.assertEqual(ret.st_arg, st+st)

  def test_contiguous_view_simplify(self):
    base = ShapeTracker.from_shape((32, 32))
    a = UOp(Ops.LOAD, dtypes.char, (UOp.new_buffer(Device.DEFAULT, base.size, dtypes.char), base.to_uop()))
    swizzle = a.reshape((64, 16))
    swizzle = graph_rewrite(swizzle, remove_movement_ops)
    self.assertEqual(swizzle_cnt(swizzle), 1)
    ret = swizzle_rewrite(swizzle)
    self.assertEqual(ret.st_arg, base.reshape((64, 16))) # late rewrite
    reswizzle = a.reshape((64, 16)).reshape((32, 32))
    self.assertEqual(swizzle_cnt(reswizzle), 0) # instant rule
    ret = swizzle_rewrite(reswizzle)
    self.assertIs(ret, reswizzle)

  def test_late_fusion_post_permute_simpler(self):
    base = ShapeTracker.from_shape((32, 16, 1))
    start = UOp(Ops.LOAD, dtypes.char, (UOp.new_buffer(Device.DEFAULT, base.size, dtypes.char), base.to_uop()))
    r = start.expand((32, 16, 16)).r(Ops.ADD, (2,))
    add = r.reshape((16, 32, 1)) + UOp.const_with_shape(r.dtype, 0, (16, 32, 1))
    self.assertEqual(add.st, ShapeTracker.from_shape((16, 32, 1)))
    to_store = add.permute((1, 0, 2)).contiguous()
    to_store = graph_rewrite(to_store, remove_movement_ops)
    self.assertEqual(to_store.st, ShapeTracker.from_shape((32, 16, 1)))
    self.assertEqual(to_store.src[0].st, add.st.permute((1, 0, 2)))
    self.assertIs(to_store.src[0].op, Ops.VIEW)
    ret = graph_rewrite(to_store, view_left)
    self.assertEqual(swizzle_cnt(ret), 1)

def store_val(si:ScheduleItem): return si.ast.src[0].src[2]
class TestView(unittest.TestCase):
  def test_all_masked_out(self):
    # start with non CONST Ops
    a = Tensor.rand(10, 10).realize()
    # all masked out, degrades to const 0
    b = a.pad(((0, 10), None))[10:]
    sched = check_schedule(b.contiguous(), 1)
    # TODO: this VALID can clean up, where do we need st?
    self.assertIs(store_val(sched[-1]), UOp.const_with_shape(b.dtype, 0, b.lazydata.st.shape))
    run_schedule(sched)
    np.testing.assert_equal(b.numpy(), 0)

  def test_mask_dim_1(self):
    # mask out dim = 1 works too
    a = Tensor.rand(10, 10).realize()
    b = a.pad((None, (0, 10)))[:, 10:]
    assert b.shape == (10, 10)
    sched = check_schedule(b.contiguous(), 1)
    self.assertEqual(sched[-1].ast.full_shape, (10, 10))
    self.assertIs(store_val(sched[-1]), UOp.const_with_shape(b.dtype, 0, b.lazydata.st.shape))
    run_schedule(sched)
    np.testing.assert_equal(b.numpy(), 0)

  def test_zero_size_alt(self):
    st = ShapeTracker.from_shape((135, 0, 9))
    a = UOp(Ops.VIEW, dtypes.float, (UOp.new_buffer(Device.DEFAULT, 121, dtypes.float), UOp(Ops.EMPTY, dtypes.float)), st)
    b = a.pad(pad_arg:=((0, 0), (0, 0), (18, 0)))
    self.assertEqual(b.st, st.pad(pad_arg))
    # TODO: why does this help?
    b = graph_rewrite(b, remove_movement_ops)
    self.assertIs(b.base.src[1], UOp.const(dtypes.float, 0))

  def test_partial_mask(self):
    # partial masked out does not degrade into CONST
    a = Tensor.rand(10, 10).realize()
    b = a.pad(((0, 5), None))[5:]
    assert b.shape == (10, 10)
    sched = check_schedule(b.contiguous(), 1)
    self.assertEqual(store_val(sched[-1]).op, Ops.LOAD)
    self.assertEqual(store_val(sched[-1]).st_arg, b.lazydata.st)
    run_schedule(sched)
    np.testing.assert_allclose(b.numpy(), np.pad(a.numpy(), ((0, 5), (0, 0)))[5:])

@track_rewrites(named=True)
def big_graph_rewrite(big_graph:UOp, ctx) -> UOp: return graph_rewrite(big_graph, do_realize, ctx)
class TestBigGraph(unittest.TestCase):
  def test_sink_childless_const(self):
    x = UOp.const(dtypes.int, 0)
    big_graph = big_graph_rewrite(x.sink(), ctx:=ScheduleContext())
    self.assertIs(big_graph, UOp(Ops.NOOP))
    self.assertEqual(len(ctx.realizes), 0)

  def test_sink_childless_const_alt(self):
    x = UOp.const(dtypes.int, 0)
    y = UOp(Ops.VIEW, dtypes.int, (UOp(Ops.BUFFER, dtypes.int, (), 0), UOp.const(dtypes.int, 0)), ShapeTracker.from_shape(()))
    big_graph = big_graph_rewrite(UOp.sink(x, y), ctx:=ScheduleContext())
    self.assertIs(big_graph, UOp(Ops.NOOP))
    self.assertEqual(len(ctx.realizes), 0)

  def test_sink_childless_const_alt_expanded(self):
    # this is a real STORE of CONST (post expand)
    y = UOp(Ops.VIEW, dtypes.int, (UOp.new_buffer(Device.DEFAULT, 1, dtypes.int), UOp.const(dtypes.int, 0)), ShapeTracker.from_shape(()))
    out = UOp(Ops.VIEW, dtypes.int, (UOp.new_buffer(Device.DEFAULT, 2, dtypes.int), y.reshape((1,)).expand((2,)).contiguous(),), ShapeTracker.from_shape((2,)))
    big_graph = big_graph_rewrite(out.sink(), ctx:=ScheduleContext())
    self.assertIs(big_graph, out.sink())
    self.assertEqual(len(ctx.realizes), 1)

tensor_const_pm = PatternMatcher([
  (UPat(Ops.VIEW, src=(UPat(Ops.BUFFER), UPat(Ops.CONST, src=()))), lambda: True),
  (UPat(Ops.VIEW, src=(UPat(Ops.BUFFER), UPat(Ops.BIND, src=(UPat(Ops.DEFINE_VAR), UPat(Ops.CONST))))), lambda: True),
])
class TestConst(unittest.TestCase):
  # ** part 1: basic functionality of a tensor directly created from CONST

  def test_tensor_const(self):
    a = Tensor(1)
    print(a.lazydata)
    self.assertTrue(tensor_const_pm.rewrite(a.lazydata))

  def test_tensor_variable(self):
    vv = UOp.variable("a", 0, 10).bind(1)
    a = Tensor(vv)
    print(a.lazydata)
    self.assertTrue(tensor_const_pm.rewrite(a.lazydata))

  def test_uop_methods(self):
    a = Tensor(1)
    self.assertTrue(a.lazydata.is_unrealized_const())
    self.assertTrue(a.lazydata.is_unrealized_unmasked_const())

    a = Tensor.ones((4, 4))
    self.assertTrue(a.lazydata.is_unrealized_const())
    self.assertTrue(a.lazydata.is_unrealized_unmasked_const())

    a = Tensor.ones((4, 4)).pad((1, 1),)
    self.assertTrue(a.lazydata.is_unrealized_const())
    self.assertFalse(a.lazydata.is_unrealized_unmasked_const())

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
    const_ast_pattern = UPat(Ops.SINK, src=(UPat.store(UPat(), UPat(), UPat(Ops.WHERE, src=(UPat(Ops.VALID), UPat.cvar("x"), UPat(Ops.CONST, arg=0)))),))
    self.assertEqual(len(const_ast_pattern.match(sched[0].ast, {})), 1)
    run_schedule(sched)
    self.assertListEqual(a.tolist(), [0, 1, 1, 1, 1, 0])

  # TOOD: currently even unmasked constants are VALID until codegen
  def test_unmasked_const_ast(self):
    a = Tensor.ones((4,)).contiguous()
    sched = a.schedule()
    print(sched[0].ast)
    const_ast_pattern = UPat(Ops.SINK, src=(UPat.store(UPat(), UPat(), UPat(Ops.WHERE, src=(UPat(Ops.VALID), UPat.cvar("x"), UPat(Ops.CONST, arg=0)))),))
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
    self.assertIs(add.lazydata.realized, b.lazydata.realized)
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
    self.assertIsNot(add.lazydata.realized, b.lazydata.realized)
    self.assertListEqual(add.tolist(), [4, 2, 2, 2, 2, 4])

  # ** part 3: Tensor variable bindings

  @unittest.expectedFailure # TODO: should schedule assert if you try to realize a Variable?
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

if __name__ == '__main__':
  unittest.main(verbosity=2)
