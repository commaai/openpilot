# this will be the new test_ops for the next level
# schedule confirms the right things are capable of fusing
# NOTE: this has overlap with external_test_opt.py

import gc, unittest, functools
import numpy as np
from typing import cast
from hypothesis import assume, given, strategies as strat

from tinygrad import nn, dtypes, Device, Tensor, Variable
from tinygrad.dtype import DType
from tinygrad.uop.ops import UOp, Ops, UPat
from tinygrad.helpers import DEBUG, OSX, GlobalCounters, Context, getenv, all_same, temp
from tinygrad.engine.realize import compile_linear, run_linear
from test.helpers import CI

supported_dtypes = Device[Device.DEFAULT].renderer.supported_dtypes()

class KernelCountException(Exception): pass
def check_schedule(t:Tensor|list[Tensor]|UOp, allowed:int, to_prerealize:list[Tensor]|None=None, filter_sink=True):
  if to_prerealize:
    with Context(DEBUG=0, TRACK_MATCH_STATS=0): Tensor.realize(*to_prerealize)
  if isinstance(t, Tensor): linear, var_vals = t.linear_with_vars()
  elif isinstance(t, list) and isinstance(t[0], Tensor): linear, var_vals = Tensor.linear_with_vars(*t)
  else:
    assert isinstance(t, UOp), f"can't schedule {t}"
    linear, var_vals = Tensor(t).linear_with_vars()
  kernel_cnt = sum((len(call.device) if isinstance(call.device, tuple) else 1)
                   for call in linear.src if call.src[0].op is Ops.SINK or not filter_sink)
  if kernel_cnt != allowed:
    print(f"SCHEDULE ISSUE, expecting {allowed} got {kernel_cnt}")
    if DEBUG >= 3:
      for i,call in enumerate(linear.src):
        print("kernel", i+1)
        print(call.src[0])
    raise KernelCountException(f"{kernel_cnt} != {allowed}")
  # test compiling the linear
  compile_linear(linear)
  return linear, var_vals

def _realize_weights(m):
  for p in nn.state.get_parameters(m): p.realize()

def _test_conv2d(allowed:int, dtype:DType=dtypes.float):
  old_default_float, dtypes.default_float = dtypes.default_float, dtype
  dtypes.default_float = dtype
  Tensor.manual_seed(0)
  BS, CIN = 2, 3
  img = Tensor.randn(BS, CIN, 64, 64).realize()
  w = Tensor.uniform(16, CIN, 3, 3).realize()
  ret = Tensor.conv2d(img, w).relu().mean().backward()
  dtypes.default_float = old_default_float
  linear, var_vals = Tensor.linear_with_vars(ret, img.grad, w.grad)
  run_linear(linear, var_vals)
  cnt = len([call for call in linear.src if call.src[0].op is Ops.SINK])
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
  def setUp(self):
    self.ctx = Context(SPLIT_REDUCEOP=0)
    self.ctx.__enter__()
  def tearDown(self):
    self.ctx.__exit__(None, None, None)

  def test_arange_avgpool2d(self, kcount=1):
    x = Tensor.arange(25).reshape(1,1,5,5).cast(dtypes.float32)
    t = x.avg_pool2d(padding=1)
    linear, var_vals = t.linear_with_vars()
    self.assertEqual(len(linear.src), kcount)
    run_linear(linear, var_vals)
    import torch
    torch_out = torch.nn.functional.avg_pool2d(torch.arange(25).reshape(1,1,5,5).float(), kernel_size=(2,2), padding=1).numpy()
    np.testing.assert_allclose(t.numpy(), torch_out)

  def test_arange_avgpool2d_fused_noopt(self):
    with Context(NOOPT=1): self.test_arange_avgpool2d(kcount=1)

  # linearizer error
  @unittest.skip("recursion error no longer raised")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.supports_float4, "needs supports_float4 to fail")
  def test_arange_avgpool2d_fused(self):
    with self.assertRaises(RecursionError):
      with Context(NOOPT=0): self.test_arange_avgpool2d(kcount=1)

  # when we're fusing a reduce, all ReduceOps must have the same N in the dimensions
  # all permutes, reshapes, expands and shrinks push through the reduce
  def test_arange_sum(self):
    a = Tensor.arange(6).reshape(3, 2).sum(axis=1)
    run_linear(*check_schedule(a, 1))
    self.assertListEqual(a.tolist(), [1, 5, 9])

  def test_arange_sum_alt(self):
    a = (Tensor.arange(5).reshape(1,5).expand(6,5)*Tensor(2)).reshape(1,6,5).sum(axis=2)
    run_linear(*check_schedule(a, 1))
    np.testing.assert_equal(a.numpy(), 20)

  def test_permute_arange(self):
    a = Tensor.arange(6).reshape(6, 1, 1).permute(2, 0, 1).sum(axis=1)
    run_linear(*check_schedule(a, 1))
    self.assertListEqual(a.tolist(), [[15]])

  @unittest.skipUnless(dtypes.half in supported_dtypes, "need half")
  @unittest.skipIf(Device.DEFAULT == "WEBGPU" and OSX, "WEBGPU Metal backend is not accurate enough")
  def test_expand_buffer_before_cast(self):
    a = Tensor.randn(4, 2, 1).realize().permute((1, 0, 2))
    b = a.cast(dtypes.half).expand((2, 4, 4))+2
    run_linear(*check_schedule(b, 1))
    np.testing.assert_allclose(b.numpy(), np.broadcast_to(a.numpy().astype(np.float16), (2, 4, 4))+2, rtol=1e-3)

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

  def test_indexing_scalars(self):
    # cover each shape at all index corners
    for x, y in [(2,2), (2,3), (3,2), (3,3)]:
      for a, b in [(0,0), (0,y-1), (x-1,0), (x-1,y-1)]:
        X = Tensor.randn(x, y).realize()
        xt = X[Tensor(a)][Tensor(b)]
        run_linear(*check_schedule(xt, 1))
        np.testing.assert_equal(xt.numpy(), X.numpy()[a][b])

  def test_push_pads_elementwise(self):
    x = Tensor.full((4,4), 2.).contiguous().realize()
    y = Tensor.full((4,4), 4.).contiguous().realize()
    z = (x.reciprocal()*y).pad((None, (0,1),)).sum()
    run_linear(*check_schedule(z, 1))
    self.assertEqual(z.item(), 32)

  def test_push_pads_contiguous(self):
    x = Tensor.full((4,1), 2.).contiguous()
    y = Tensor.full((4,4), 4.).contiguous()
    z = (x.reciprocal().expand(4,4)*y).pad((None, (0,1),)).sum()
    run_linear(*check_schedule(z, 1, [x,y]))
    self.assertEqual(z.item(), 32)

  def test_allow_push_permutes(self):
    a = Tensor.randn(10,10,10).realize()
    b = Tensor.randn(10,10,1).realize()
    c = a.sum(axis=0, keepdim=True).permute(2,1,0) + b
    run_linear(*check_schedule(c, 1))
    np.testing.assert_allclose(c.numpy(), np.sum(a.numpy(), axis=0, keepdims=True).transpose(2,1,0)+b.numpy())

  def test_div_collapse_buffer(self):
    a = Tensor.full((4,), 4.0).contiguous().realize()
    b = Tensor.full((4,), 2.0).contiguous().realize()
    expr = (a*b)/b
    run_linear(*check_schedule(expr, 1))
    np.testing.assert_allclose(expr.numpy(), np.full((4,), 4.0))

  def test_div_collapse_const(self):
    a = Tensor.full((4,), 4.0).contiguous().realize()
    expr = a/a
    run_linear(*check_schedule(expr, 1))
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

  # NOTE: this is causing "LAZYCACHE=1 incorrectly reuses contiguous const" #4562
  # should contiguous dedup?
  @unittest.skip("we do the exact opposite now")
  def test_dedup_contiguous(self):
    a = Tensor.ones(4).contiguous()
    b = Tensor.ones(4).contiguous()
    sched = check_schedule([a, b], 1)
    run_linear(*sched)
    # a and b share the same underlying device memory
    self.assertIs(a.uop.realized, b.uop.realized)

  def test_clone_doesnt_dedup(self):
    src = Tensor.ones(4).contiguous().realize()
    a = src.clone()
    b = src.clone()
    sched = check_schedule([a, b], 2, filter_sink=False)
    run_linear(*sched)
    # a and b are assigned to the same device Buffer
    self.assertIsNot(a.uop.base.realized, b.uop.base.realized)

  @unittest.skip("no longer supported")
  def test_double_from(self):
    x = Tensor([1,2,3,4])
    out = x.to('python')
    check_schedule(out, 0, filter_sink=False)

  def test_zero_size_assign(self):
    f = Tensor.full((2,), 0.).contiguous().realize()
    a = f.shrink_to((0,))
    a.assign(Tensor.ones_like(a))
    check_schedule(a, 0)
    self.assertEqual(a.tolist(), [])

  def test_zero_size_children(self):
    r = Tensor.ones(1,2).contiguous().realize().sum(axis=(1,), keepdim=True)
    ax = r.reshape(1)*2
    ay = r.reshape(1).shrink(((1,1),))*2
    out = ax+ay.pad(((1, 0),))
    run_linear(*check_schedule(out, 1))
    self.assertEqual(out.item(), 4.)

  def test_preserve_multistage_reduce(self):
    big_enough = getenv("REDUCEOP_SPLIT_THRESHOLD", 32768)
    x = Tensor.randn(big_enough).realize()
    with Context(SPLIT_REDUCEOP=1):
      out = (x - x.max(keepdim=True)).max()
      run_linear(*check_schedule(out, 4))
    np.testing.assert_allclose(out.numpy(), (x.numpy() - x.numpy().max(keepdims=True)).max())

  def test_example_matmul_contig(self):
    x = Tensor.eye(64).contiguous().realize()
    y = Tensor.eye(64).contiguous().realize()
    z = y.matmul(x).sum()
    z.backward()
    out = x.grad.contiguous()
    run_linear(*check_schedule(out, 1))
    np.testing.assert_allclose(out.numpy(), np.ones((64,64)))

  def test_example_matmul_same(self):
    x = Tensor.eye(64)
    z = x.matmul(x).sum()
    z.backward()
    out = x.grad.contiguous()
    run_linear(*check_schedule(out, 1))
    # NOTE: the gradient flows twice
    np.testing.assert_allclose(out.numpy(), 2*np.ones((64,64)))

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
    run_linear(*check_schedule(out, 1))
    np.testing.assert_allclose(out.numpy(), a.numpy().sum(axis=1)[:16] + b.numpy().sum(axis=1)[:16] + c.numpy(), atol=1e-4, rtol=1e-4)

  def test_reduce_same_size(self):
    Tensor.manual_seed(0)
    a = Tensor.randn(4, 4).realize()
    out0 = a.sum() + 2
    out1 = a.sum() + 4
    out2 = out0 * out1
    run_linear(*check_schedule([out0, out1, out2], 3)) # TODO: 1?
    np.testing.assert_allclose(out0.numpy(), out0_np:=a.numpy().sum()+2, atol=1e-4, rtol=1e-6)
    np.testing.assert_allclose(out1.numpy(), out1_np:=a.numpy().sum()+4, atol=1e-4, rtol=1e-6)
    np.testing.assert_allclose(out2.numpy(), out0_np*out1_np, atol=1e-4, rtol=1e-6)

  def test_reduce_multiple_paths(self):
    Tensor.manual_seed(0)
    a = Tensor.randn(4, 4).realize()
    out0 = a.sum().exp2()
    # out1 has two paths to a.sum()
    out1 = a.sum() + out0
    run_linear(*check_schedule([out0, out1], 2)) # TODO: 1?
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
    # run_linear(*check_schedule([out0, out1, out2, out3], 1))
    run_linear(*check_schedule([out0, out1, out2, out3], 4))
    np.testing.assert_allclose(out0.numpy(), np_out0:=np.exp2(a.numpy().sum()), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(out1.numpy(), np_out1:=a.numpy().sum()+np_out0, atol=1e-4, rtol=1e-4)
    np_b = (a.numpy() + np_out0 + np_out1)
    with np.errstate(over='ignore'):
      np.testing.assert_allclose(out2.numpy(), np_out2:=np.exp2(np_b.sum()), atol=1e-4, rtol=1e-4)
      np.testing.assert_allclose(out3.numpy(), np_b.sum()+np_out2, atol=1e-4, rtol=1e-4)

  def test_reduce_ext_reduce_child(self):
    Tensor.manual_seed(0)
    a = Tensor.randn(4, 4).realize()
    b = Tensor.randn(4, 4).realize()
    # b.sum() is not a descendant of the fused nodes
    out0 = a.sum() + b.sum() + 2
    out1 = a.sum() + b.sum() + 4
    # run_linear(*check_schedule([out0, out1], 1))
    run_linear(*check_schedule([out0, out1], 2))
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
    # run_linear(*check_schedule([r, out0, out1, out2], 1))
    run_linear(*check_schedule([r, out0, out1, out2], 4))
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
    # run_linear(*check_schedule([out0, out1, out2], 1))
    run_linear(*check_schedule([out0, out1, out2], 3))
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
    # run_linear(*check_schedule([r, out0, out1, e], 3)) # 1 or 2 or 3? should be 1 (one reduce) but the different outputs might make it 3
    run_linear(*check_schedule([r, out0, out1, e], 4))
    np.testing.assert_allclose(r.numpy(), r_np:=a.numpy().sum(), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(out0.numpy(), out0_np:=np.exp2(r_np), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(e.numpy(), e_np:=b.numpy() + out0_np, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(out1.numpy(), r_np + e_np[0][0][0], atol=1e-4, rtol=1e-4)

  def test_reduce_expand_child(self):
    Tensor.manual_seed(0)
    a = Tensor.randn((32, 32, 32)).realize()
    b = Tensor.randn((1, 16)).realize()
    out0 = a.sum() + 2
    out1 = a.sum() + b
    # run_linear(*check_schedule([out0, out1], 2))
    run_linear(*check_schedule([out0, out1], 3))
    np.testing.assert_allclose(out0.numpy(), a.numpy().sum()+2, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(out1.numpy(), a.numpy().sum()+b.numpy(), atol=1e-4, rtol=1e-4)

  def test_std_multireduce_fusion(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(4, 32).realize()
    out = x.std(-1)
    run_linear(*check_schedule(out, 2))
    np.testing.assert_allclose(out.numpy(), x.numpy().std(axis=-1, ddof=1), atol=1e-4, rtol=1e-4)

  def test_scaled_dot_product_attention_multireduce_fusion(self):
    Tensor.manual_seed(0)
    q = Tensor.randn(32,8,16,8).realize()
    k = Tensor.randn(32,8,16,8).realize()
    v = Tensor.randn(32,8,16,8).realize()
    out = Tensor.scaled_dot_product_attention(q,k,v)
    run_linear(*check_schedule(out, 4))
    if getenv("CHECK", 1):
      import torch
      compare = torch.nn.functional.scaled_dot_product_attention(torch.tensor(q.numpy()),torch.tensor(k.numpy()),torch.tensor(v.numpy()))
      np.testing.assert_allclose(out.numpy(), compare.numpy(), atol=1e-6, rtol=1e-3)

    out = Tensor.scaled_dot_product_attention(q,k,v)
    run_linear(*check_schedule(out, 4)) # TODO: should be 1?
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
    # run_linear(*check_schedule(out, 1))
    run_linear(*check_schedule(out, 2))
    np.testing.assert_allclose(out.numpy(), \
      (c.numpy()*a.numpy().sum(axis=-1,keepdims=True)).sum(-1) + (b.numpy()*a.numpy().sum(axis=-1,keepdims=True)).sum(-1), atol=1e-4, rtol=1e-4)

  def test_reduce_expand_reduce_fusion(self):
    Tensor.manual_seed(0)
    a = Tensor.randn(4, 32).realize()
    out = (a+a.sum(-1, keepdim=True)).sum(-1)
    # run_linear(*check_schedule(out, 1))
    run_linear(*check_schedule(out, 2))
    np.testing.assert_allclose(out.numpy(), (a.numpy()+a.numpy().sum(axis=-1,keepdims=True)).sum(axis=-1), atol=1e-4, rtol=1e-4)

  def test_reduce_expand_reduce_expand_fusion(self):
    Tensor.manual_seed(0)
    a = Tensor.randn(4, 32).realize()
    out = a+(a+a.sum(-1,keepdim=True)).sum(-1, keepdim=True)
    # run_linear(*check_schedule(out, 2))
    run_linear(*check_schedule(out, 3))
    np.testing.assert_allclose(out.numpy(), \
      a.numpy()+(a.numpy()+a.numpy().sum(axis=-1,keepdims=True)).sum(axis=-1,keepdims=True), atol=1e-4, rtol=1e-4)

  def test_branching_reduces_and_expands_fusion(self):
    Tensor.manual_seed(0)
    a = Tensor.randn(4, 32).realize()
    out0 = a+a.sum(-1, keepdim=True)
    out1 = out0.sum(-1)
    # run_linear(*check_schedule(out, 2))
    run_linear(*check_schedule([out0, out1], 3))
    np.testing.assert_allclose(out0.numpy(), a.numpy()+a.numpy().sum(axis=-1,keepdims=True), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(out1.numpy(), (a.numpy()+a.numpy().sum(axis=-1,keepdims=True)).sum(axis=-1), atol=1e-4, rtol=1e-4)

  def test_multireduce_fusion_simple_sequential(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(4, 32).realize()
    y = Tensor.randn(4, 32).realize()
    out = (y + x.sum(axis=-1, keepdim=True)).sum(axis=-1)
    # run_linear(*check_schedule(out, 1))
    run_linear(*check_schedule(out, 2))
    np.testing.assert_allclose(out.numpy(), (y.numpy() + x.numpy().sum(axis=-1, keepdims=True)).sum(axis=-1), atol=1e-4, rtol=1e-4)

  def test_multireduce_fusion_simple_parallel(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(4, 32).realize()
    y = Tensor.randn(4, 32).realize()
    out = y.sum(axis=-1) + x.sum(axis=-1)
    run_linear(*check_schedule(out, 1))
    np.testing.assert_allclose(out.numpy(), y.numpy().sum(axis=-1) + x.numpy().sum(axis=-1), atol=1e-4, rtol=1e-4)

  def test_multireduce_fusion_sequential(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(4, 32).realize()
    out = x.std(-1)
    # run_linear(*check_schedule(out, 1))
    run_linear(*check_schedule(out, 2))
    np.testing.assert_allclose(out.numpy(), x.numpy().std(axis=-1, ddof=1), atol=1e-4, rtol=1e-4)

  def test_multireduce_fusion_parallel(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(4, 32).realize()
    y = Tensor.randn(4, 32).realize()
    out = x.std(-1) + y.std(-1)
    # run_linear(*check_schedule(out, 1))
    run_linear(*check_schedule(out, 3))
    np.testing.assert_allclose(out.numpy(), x.numpy().std(axis=-1, ddof=1) + y.numpy().std(axis=-1, ddof=1), atol=1e-4, rtol=1e-4)

  def test_multireduce_diffops_sequential(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(4, 32).realize()
    out = (x - x.max(-1, keepdim=True)).sum(-1)
    # run_linear(*check_schedule(out, 1))
    run_linear(*check_schedule(out, 2))
    np.testing.assert_allclose(out.numpy(), (x.numpy() - x.numpy().max(axis=-1, keepdims=True)).sum(axis=-1), atol=1e-4, rtol=1e-4)

  def test_multireduce_fusion_diffops_parallel(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(4, 32).realize()
    y = Tensor.randn(4, 32).realize()
    out = x.sum(-1) + y.max(-1)
    run_linear(*check_schedule(out, 1))
    np.testing.assert_allclose(out.numpy(), x.numpy().sum(axis=-1) + y.numpy().max(axis=-1), atol=1e-4, rtol=1e-4)

  def test_multireduce_fusion_sequential_and_parallel(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(4, 32).realize()
    y = Tensor.randn(4, 32).realize()
    mu = (x - x.max(axis=-1, keepdim=True)).mean(axis=-1, keepdim=True) + (y - y.max(axis=-1, keepdim=True)).mean(axis=-1, keepdim=True)
    out = [((x - mu).square().sum(-1)/x.shape[-1]).sqrt(), ((y - mu).square().sum(-1)/y.shape[-1]).sqrt()]
    np_mu = (x.numpy() - x.numpy().max(axis=-1, keepdims=True)).mean(axis=-1, keepdims=True) + \
      (y.numpy() - y.numpy().max(axis=-1, keepdims=True)).mean(axis=-1, keepdims=True)
    # run_linear(*check_schedule(out, 1))
    run_linear(*check_schedule(out, 5))
    np.testing.assert_allclose(out[0].numpy(), np.sqrt(np.square(x.numpy() - np_mu).sum(-1)/x.shape[-1]), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(out[1].numpy(), np.sqrt(np.square(y.numpy() - np_mu).sum(-1)/y.shape[-1]), atol=1e-4, rtol=1e-4)

  def test_cumsum_parallel_reduce_fused(self):
    # two-stage cumsum + ops triggers parallel REDUCEs in one kernel that must share an END (same nesting context = should merge)
    step, num_steps = 513, 10
    t = Tensor.arange(step).float().realize()
    phase = t.cumsum()
    tiled = phase.repeat((num_steps,)).reshape(num_steps, step)
    pattern = Tensor([1,0,0,1,0,0,0,0,1,0]).reshape(num_steps, 1)
    out = (tiled * pattern).flatten()
    expected = np.tile(np.arange(step).astype(np.float32).cumsum(), num_steps).reshape(num_steps, step)
    expected = (expected * np.array([1,0,0,1,0,0,0,0,1,0]).reshape(num_steps, 1)).flatten()
    np.testing.assert_allclose(out.numpy(), expected, atol=1e-4, rtol=1e-4)

  @unittest.skipIf(Device.DEFAULT == "CL", "TODO: fails on CI CL")
  def test_reduce_different_nesting_depth(self):
    # two REDUCEs sharing the same RANGE at different nesting depths must NOT merge
    x = Tensor.arange(768).reshape(3, 256).float()
    np.testing.assert_allclose((x.sum(axis=1) + x.sum(axis=1).sum()).numpy(), x.numpy().sum(axis=1) + x.numpy().sum(axis=1).sum())

  def test_multimatmul_fusion(self):
    Tensor.manual_seed(0)
    a,b = Tensor.randn(4, 64).realize(), Tensor.rand(64,8).realize()
    c,d = Tensor.randn(4, 64).realize(), Tensor.rand(64,8).realize()
    out = a@b + c@d
    run_linear(*check_schedule(out, 1))
    np.testing.assert_allclose(out.numpy(), a.numpy()@b.numpy() + c.numpy()@d.numpy(), atol=1e-4, rtol=1e-4)

  def test_softmax_fusion(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(4, 12, 64, 64).realize()
    out = x.softmax()
    run_linear(*check_schedule(out, 3))
    expected = (x_exp:=np.exp(x.numpy()-x.numpy().max(-1, keepdims=True)))/x_exp.sum(-1, keepdims=True)
    np.testing.assert_allclose(out.numpy(), expected, atol=1e-4, rtol=1e-4)

  def test_layernorm_onelayer_fusion(self):
    Tensor.manual_seed(0)
    layer = nn.LayerNorm([10, 10])
    layer.weight = Tensor.randn(10,10).realize()
    layer.bias = Tensor.randn(10,10).realize()
    x = Tensor.randn(20, 5, 10, 10).realize()
    out = layer(x)
    # run_linear(*check_schedule(out, 2))
    run_linear(*check_schedule(out, 3))
    y = (x.numpy() - x.numpy().mean(layer.axis, keepdims=True))
    expected = y / np.sqrt((y*y).mean(layer.axis, keepdims=True) + layer.eps)
    np.testing.assert_allclose(out.numpy(), expected * layer.weight.numpy() + layer.bias.numpy(), atol=1e-4, rtol=1e-4)

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
    run_linear(*schedule)
    np.testing.assert_allclose(b.numpy(), np_r.sum(0) + 8, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(c.numpy(), np_r.sum(1) + 12, atol=1e-4, rtol=1e-4)

  def test_multireduce_push_permute_chase(self):
    Tensor.manual_seed(0)
    a = Tensor.randn(4, 4, 4).realize()
    b = Tensor.randn(4, 4).realize()
    r = a.sum(2) + b
    d = r.T * 4
    e = r * (d + a).sum(2)
    schedule = check_schedule([d, e], 3) # make sure it doesn't fuse
    run_linear(*schedule)
    np.testing.assert_allclose(d.numpy(), (a.numpy().sum(2) + b.numpy()).T * 4, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(e.numpy(), (a.numpy().sum(2) + b.numpy()) * (d.numpy() + a.numpy()).sum(2), atol=1e-4, rtol=1e-4)

  def test_multireduce_push_shrink_chase(self):
    Tensor.manual_seed(0)
    a = Tensor.randn(16, 16).realize()
    b = Tensor.randn(4).realize()
    c = Tensor.randn(16, ).realize()
    d = Tensor.randn(16, 16).realize()
    r = a.sum(1) + c
    out = r[:4] * b + d.sum(1)[:4]
    schedule = check_schedule(out, 1)
    run_linear(*schedule)
    np.testing.assert_allclose(out.numpy(), (a.numpy().sum(1) + c.numpy())[:4] * b.numpy() + d.numpy().sum(1)[:4], atol=1e-4, rtol=1e-4)

  def test_multireduce_midreduce_nochase(self):
    Tensor.manual_seed(0)
    a = Tensor.randn(16, 16).realize()
    b = (a.sum(0)+a.max(0) + a.max(1)+a.sum(1)) + 2
    schedule = check_schedule(b, 1)
    run_linear(*schedule)
    np.testing.assert_allclose(b.numpy(), a.numpy().sum(0)+a.numpy().max(0) + a.numpy().max(1)+a.numpy().sum(1)+2, atol=1e-4, rtol=1e-4)

  # pattern in test_transformer
  def test_partial_fuse1(self):
    Tensor.manual_seed(0)
    a = Tensor.randn(16, 16).realize()
    b = Tensor.randn(16, 16).realize()
    c = a.sum() + 2
    d = (a.sum() - b.sum()) * 4
    # run_linear(*check_schedule([c, d], 1))
    run_linear(*check_schedule([c, d], 2))
    np.testing.assert_allclose(c.numpy(), a.numpy().sum()+2, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(d.numpy(), (a.numpy().sum() - b.numpy().sum()) * 4, atol=1e-4, rtol=1e-4)

  # pattern in conv
  def test_partial_fuse2(self):
    Tensor.manual_seed(0)
    a = Tensor.randn(16, 16).realize()
    b = Tensor.randn(16, 16).realize()
    c = a.sum() + 2
    d = b.sum() - c
    # run_linear(*check_schedule([c, d], 1))
    run_linear(*check_schedule([c, d], 2))
    np.testing.assert_allclose(c.numpy(), a.numpy().sum()+2, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(d.numpy(), b.numpy().sum()-(a.numpy().sum()+2), atol=1e-4, rtol=1e-4)

  # pattern in adam
  def test_partial_fuse3(self):
    Tensor.manual_seed(0)
    a = Tensor.randn(16, 16).realize()
    b = Tensor.randn(16, 16).realize()
    c = a.sum() + 2
    d = a.sum() * 2
    e = c * d
    f = b.sum() - e
    # run_linear(*check_schedule([c, d, e, f], 1))
    run_linear(*check_schedule([c, d, e, f], 4))
    np.testing.assert_allclose(c.numpy(), c_np:=a.numpy().sum()+2, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(d.numpy(), d_np:=a.numpy().sum()*2, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(e.numpy(), e_np:=c_np*d_np, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(f.numpy(), b.numpy().sum() - e_np, atol=1e-4, rtol=1e-4)

  def test_partial_fuse4(self):
    Tensor.manual_seed(0)
    a = Tensor.randn(16, 16).realize()
    b = Tensor.randn(16, 16).realize()
    c = a.sum() + 2
    d = a.sum() * 2
    e = c * d
    f = (b - d).sum() - e
    # run_linear(*check_schedule([c, d, e, f], 1))
    run_linear(*check_schedule([c, d, e, f], 4))
    np.testing.assert_allclose(c.numpy(), c_np:=a.numpy().sum()+2, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(d.numpy(), d_np:=a.numpy().sum()*2, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(e.numpy(), e_np:=c_np*d_np, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(f.numpy(), (b.numpy()-d_np).sum()-e_np, atol=1e-4, rtol=1e-4)

  def test_pad_reduce_safe(self):
    Tensor.manual_seed(0)
    a = Tensor.rand(3, 4, 5).realize()
    b = Tensor.rand(3, 4, 5).realize()
    out = (a + b).pad(((0, 1), (0, 1), (0, 1)), value=1.0).sum().contiguous()
    run_linear(*check_schedule(out, 1))
    np.testing.assert_allclose(out.numpy(), np.pad(a.numpy()+b.numpy(), ((0, 1), (0, 1), (0, 1)), constant_values=1.0).sum(), atol=1e-5, rtol=1e-6)

  def test_multireduce_pad_reduce_safe(self):
    Tensor.manual_seed(0)
    a = Tensor.randn(3, 4, 5).realize()
    b = Tensor.randn(3, 4, 5).realize()
    out = (a.pad(((0, 1), (0, 1), (0, 1)), value=1.0).sum(keepdim=True)+b.pad(((0, 1), (0, 1), (0, 1)), value=1.0).sum()).contiguous()
    run_linear(*check_schedule(out, 1))
    np.testing.assert_allclose(out.numpy(), np.pad(a.numpy(), ((0, 1), (0, 1), (0, 1)), constant_values=1.0).sum(keepdims=True) + \
                                                   np.pad(b.numpy(), ((0, 1), (0, 1), (0, 1)), constant_values=1.0).sum(), atol=1e-4, rtol=1e-4)

  def test_pad_reduce_unsafe(self):
    Tensor.manual_seed(0)
    a = Tensor.rand(3, 4, 5).realize()
    out = a.log2().pad(((0, 1), (0, 1), (0, 1)), value=1.0).sum().contiguous()
    run_linear(*check_schedule(out, 1))
    np.testing.assert_allclose(out.numpy(), np.pad(np.log2(a.numpy()), ((0, 1), (0, 1), (0, 1)), constant_values=1.0).sum(), atol=1e-5, rtol=1e-6)

  def test_multireduce_pad_reduce_unsafe(self):
    Tensor.manual_seed(0)
    a = Tensor.randn(3, 4, 5).abs().realize()
    b = Tensor.randn(3, 4, 5).abs().realize()
    out = (a.log2().pad(((0, 1), (0, 1), (0, 1)), value=1.0).sum()+b).abs().log2().pad(((0, 1), (0, 1), (0, 1)), value=1.0).sum().contiguous()
    # run_linear(*check_schedule(out, 1))
    run_linear(*check_schedule(out, 2))
    np.testing.assert_allclose(out.numpy(), np.pad(np.log2(np.abs(np.pad(np.log2(a.numpy()), ((0, 1), (0, 1), (0, 1)), constant_values=1.0).sum() + \
                                                   b.numpy())), ((0, 1), (0, 1), (0, 1)), constant_values=1.0).sum(), atol=3e-4, rtol=1e-5)

  def test_shrink_pad_safe(self):
    a = Tensor.ones((3, )).contiguous().realize()
    b = Tensor.ones((3, )).contiguous().realize()
    out = (a + b).shrink(((0, 1),)).pad(((0, 1),)).contiguous()
    run_linear(*check_schedule(out, 1))
    np.testing.assert_equal(out.numpy(), [2, 0])

  def test_shrink_pad_unsafe(self):
    a = Tensor.ones((3, )).contiguous().realize()
    out = a.exp2().shrink(((0, 1),)).pad(((0, 1),)).contiguous()
    run_linear(*check_schedule(out, 1))
    np.testing.assert_equal(out.numpy(), [2, 0])

  def test_base_change_shrink_pad(self):
    a = Tensor.ones(3, 3).contiguous().realize()
    b = a.exp2()
    c = b[:-1, :-1]
    d = c.pad(((0, 1), (0, 1))) * 2
    run_linear(*check_schedule(d, 1))
    np.testing.assert_equal(d.numpy(), np.pad(np.exp2(a.numpy())[:-1, :-1], ((0, 1), (0, 1)))*2)

  def test_base_change_expand_pad(self):
    a = Tensor.ones(3, 3).contiguous().realize()
    b = a.exp2()
    c = b[:, None, :]
    d = c.pad(((0, 0), (1, 1), (0, 0))) * 2
    run_linear(*check_schedule(d, 1))
    np.testing.assert_equal(d.numpy(), np.pad(np.exp2(a.numpy())[:, None, :], ((0, 0), (1, 1), (0, 0)))*2)

  def test_fuse_arange_pad_replicate_mode(self):
    x = Tensor.empty(3,3,3,3)
    y = x.pad((-1,2,2,-1), mode="replicate")
    dx = y.sum().gradient(x)[0]
    sched = check_schedule(dx, 0)
    run_linear(*sched)
    np.testing.assert_allclose(dx.numpy(), [[[[0.,3.,9.],[0,1.,3.],[0.,0.,0.]]]*3]*3)

  # TODO like openpilot with imagef
  @unittest.skipUnless(dtypes.half in supported_dtypes, "need half")
  def test_base_change_expand_expand(self):
    a = Tensor.ones(4, 4).contiguous().realize()
    b = a.cast(dtypes.half).expand(2, 4, 4)
    c = b.cast(dtypes.int).expand(2, 2, 4, 4)
    run_linear(*check_schedule(c, 1))
    np.testing.assert_equal(c.numpy(), np.ones(((2, 2, 4, 4)), dtype=np.int32))

  def test_base_change_pad_expand(self):
    a = Tensor.full((4, 4), 1.).contiguous().realize()
    b = Tensor.full((4, 4), 2.).contiguous().realize()
    c = (a + b).pad(((1, 1), (1, 1)))
    d = c.cast(dtypes.int).expand((2, 6, 6)) * 4
    run_linear(*check_schedule(d, 1))
    c_np = np.pad((np.full((4, 4), 2., dtype=np.float32) + np.full((4, 4), 1., dtype=np.float32)), ((1, 1), (1, 1)), constant_values=0.0)
    np.testing.assert_equal(d.numpy(), np.broadcast_to(c_np.astype(np.half), (2, *c_np.shape)) * 4)

  def test_pad_reduce_unsafe_multiview_st(self):
    P = Tensor.ones(3, 3).contiguous()
    sums = P.sum(axis=1, keepdim=True)
    P /= sums
    p = P[0]
    p = p.pad(((1, 0), ))
    p = p.repeat([2])
    run_linear(*check_schedule(p, 3))
    tiny_ret = p.numpy()

    P = np.ones((3, 3), dtype=np.float32)
    sums = P.sum(axis=1, keepdims=True)
    P /= sums
    p = P[0]
    p = np.pad(p, (1, 0), 'constant')
    p = np.tile(p, 2)
    np.testing.assert_allclose(tiny_ret, p)

  @unittest.skip("disabling subbuffer manually isn't supported anymore")
  def test_bitcast_disable_subbufer(self):
    x = cast(UOp, Tensor.empty(1, dtype=dtypes.float32).realize().uop)
    a = x.alu(Ops.EXP2).cast(dtypes.int32, True, allow_buffer_view=False)
    b = x.cast(dtypes.int32, True, allow_buffer_view=False)
    b = a.alu(Ops.ADD, b)
    check_schedule(b, 1)

  def test_conv2d(self): _test_conv2d(4)
  def test_conv2d_fused(self): _test_conv2d(4)

  @unittest.skipUnless(dtypes.half in supported_dtypes, "need half")
  def test_conv2d_half(self): _test_conv2d(4, dtype=dtypes.half)
  @unittest.skipUnless(dtypes.half in supported_dtypes, "need half")
  @unittest.skipIf(Device.DEFAULT == "WEBGPU", "Causes other tests to fail")
  def test_conv2d_fused_half(self): _test_conv2d(4, dtype=dtypes.half)

  def test_schedule_mem_used_with_inputs(self):
    gc.collect()
    base = GlobalCounters.mem_used
    x = Tensor.ones(256).contiguous().realize()
    (x+Tensor.ones(256).contiguous()).schedule_linear()
    gc.collect()
    self.assertEqual(GlobalCounters.mem_used-base, 1024)

  @unittest.skipIf(Device.DEFAULT != "CL", "image only supported on CL")
  def test_image_dot_f16_fusion(self):
    with Context(FLOAT16=1, OPENPILOT_HACKS=1):
      def cnt():
        x, y, z = Tensor.empty((64, 64), dtype='float'), Tensor.empty((64, 64), dtype='float'), Tensor.empty((64, 64), dtype='float')
        a = (x @ y).relu()
        linear = compile_linear(((a @ z).relu() + a).schedule_linear())
        return len([call for call in linear.src if call.src[0].op is Ops.PROGRAM])

      with Context(IMAGE=1):
        self.assertEqual(cnt(), 5)

  @unittest.skipIf(Device.DEFAULT != "CL", "image only supported on CL")
  def test_image_f16_residual_fusion(self):
    with Context(FLOAT16=1, OPENPILOT_HACKS=1):
      def cnt():
        inp = Tensor.empty((512,), dtype='float')
        b1, b2 = Tensor.empty((512, 1024), dtype='float'), Tensor.empty((1024, 512), dtype='float')
        c1, c2 = Tensor.empty((1024,), dtype='float'), Tensor.empty((512,), dtype='float')
        rb = (((((inp @ b1) + c1).relu() @ b2) + c2).relu() + inp).relu()
        b16, c16 = Tensor.empty((512, 16), dtype='float'), Tensor.empty((16,), dtype='float')
        b32, c32 = Tensor.empty((512, 32), dtype='float'), Tensor.empty((32,), dtype='float')
        linear = compile_linear(Tensor.schedule_linear((rb @ b16 + c16).relu(), (rb @ b32 + c32).relu()))
        return len([call for call in linear.src if call.src[0].op is Ops.PROGRAM])

      with Context(IMAGE=1):
        self.assertEqual(cnt(), 9)

  @unittest.skipIf(Device.DEFAULT != "CL", "image only supported on CL")
  def test_image_conv_fusion(self):
    with Context(OPENPILOT_HACKS=1):
      def cnt():
        x, y, z = Tensor.empty((1, 4, 3, 3)), Tensor.empty((4, 1, 3, 3)), Tensor.empty((4, 1, 7, 7))
        a = x.conv2d(y, Tensor.empty(4), groups=4, padding=1)
        b = a.conv2d(z, groups=4, padding=3)
        linear = compile_linear((a + b).schedule_linear())
        return len([call for call in linear.src if call.src[0].op is Ops.PROGRAM])

      with Context(IMAGE=1):
        self.assertEqual(cnt(), 5)

  def _test_fusion(self, shapes, f, cnt):
    with Context(DEBUG=0, TRACK_MATCH_STATS=0): args = [Tensor.randn(s).realize() for s in shapes]
    run_linear(*check_schedule(compare:=f(*args), cnt))
    if getenv("COMPARE", 1):
      import torch
      good = f(*[torch.tensor(x.numpy()) for x in args])
      np.testing.assert_allclose(compare.numpy(), good.numpy(), atol=1e-4, rtol=1e-4)

  def test_late_fusion_simple(self):
    self._test_fusion([(4, 4), (4, 1)], lambda a,b:a.sum(1, keepdim=True)+b, 1)

  def test_late_fusion_post_reshape(self):
    self._test_fusion([(4, 4), (1, 4)], lambda a,b:a.sum(1).reshape(b.shape)+b, 1)

  def test_late_fusion_post_permute(self):
    self._test_fusion([(4, 6, 4), (4, 4, 1)], lambda a,b:a.sum(1, keepdim=True).permute((2, 0, 1))+b, 1)

  def test_late_fusion_double_transpose(self):
    self._test_fusion([(32, 16, 1)],
                      lambda a:(a.expand(32, 16, 16).sum((2,), keepdim=True).permute((1, 0, 2))+2).permute((1, 0, 2)).contiguous(), 1)

  def test_late_fusion_post_expand(self):
    self._test_fusion([(32, 32)], lambda a:a-a.sum(1), 2)

  def test_cast_padded_view(self):
    a = Tensor.arange(4).reshape(1, 4)
    casted_view = a.pad(((0, 1), (0, 0))).cast(dtypes.float)
    casted_view.realize()
    self.assertEqual(casted_view.uop.base.realized.size, 8)
    contig = casted_view.contiguous().realize()
    self.assertEqual(contig.uop.base.realized.size, 8)
    self.assertListEqual(contig.tolist(), [[0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 0.0]])

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
    run_linear(*check_schedule(casted_view, 1))
    realized_const_view = casted_view.contiguous()
    run_linear(*check_schedule(realized_const_view, 0))
    self.assertListEqual(realized_const_view.tolist(), [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])

  @given(strat.sampled_from(dtypes.all), strat.sampled_from(dtypes.all))
  @unittest.skip("kernel count depends on input")
  def test_cast_padded_const(self, dt1, dt2):
    assume(dt1 in supported_dtypes and dt2 in supported_dtypes)
    a = Tensor(1, dtype=dt1).reshape(1, 1).pad(((1, 1), None))
    casted_view = a.cast(dt2)
    run_linear(*check_schedule(casted_view, 0))
    realized_const_view = casted_view.contiguous()
    run_linear(*check_schedule(realized_const_view, 1))
    np.testing.assert_equal(realized_const_view.numpy(), [[0], [1], [0]])

  def test_simple_indexing(self):
    X = Tensor.randn(10, 10).realize()
    idxs = Tensor([0, 2]).realize()
    xt = X[idxs]
    run_linear(*check_schedule(xt, 1))
    np.testing.assert_equal(xt.numpy(), X.numpy()[idxs.numpy()])

  def test_simple_indexing_alt(self):
    X = Tensor.arange(16).reshape(4, 4)
    xt = X[[1, 2], [-1, 2]]
    run_linear(*check_schedule(xt, 1))
    np.testing.assert_equal(xt.numpy(), (np.arange(16).reshape(4, 4))[[1, 2], [-1, 2]])

  def test_advanced_indexing(self):
    X = Tensor.arange(10)+1
    xt = X[[0, -1]]
    run_linear(*check_schedule(xt, 1))
    np.testing.assert_equal(xt.numpy(), (np.arange(10)+1)[[0, -1]])

  def test_advanced_indexing_alt(self):
    X = Tensor.arange(6).reshape(3, 2)+1
    xt = X[[Tensor([2]), Tensor([1])]]
    run_linear(*check_schedule(xt, 1))
    np.testing.assert_equal(xt.numpy(), 6)

  def test_push_through_reshape(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(10, 20).realize()
    out = x.argmax(1)
    run_linear(*check_schedule(out, 2))
    np.testing.assert_allclose(out.numpy(), np.argmax(x.numpy(), 1))

  def test_arange_push_through_expand(self):
    Tensor.manual_seed(0)
    a = Tensor.arange(4,)
    b = Tensor.randn(4, 4).realize()
    out = (a+b).sum()
    run_linear(*check_schedule(out, 1))
    np.testing.assert_allclose(out.numpy(), (np.arange(4)+b.numpy()).sum(), atol=1e-5)

  def test_argmin(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(4, 32).realize()
    out = x.argmin(-1)
    run_linear(*check_schedule(out, 2))
    np.testing.assert_equal(out.numpy(), x.numpy().argmin(axis=-1))

  def test_argmax(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(4, 32).realize()
    out = x.argmax(-1)
    run_linear(*check_schedule(out, 2))
    np.testing.assert_equal(out.numpy(), x.numpy().argmax(axis=-1))

  def test_arange_transposed(self):
    Tensor.manual_seed(0)
    x = Tensor.randint(4, 1).realize()
    a = ((Tensor.arange(4,)*x).T).sum()
    run_linear(*check_schedule(a, 1))
    np.testing.assert_equal(a.numpy(), (np.arange(4)*x.numpy()).T.sum())

  def test_div_padded_arange(self):
    x = Tensor.full((2,2), 16)
    y = x.div(Tensor.linspace(2, 8, steps=4, dtype=dtypes.int).reshape(2,2), rounding_mode="trunc").pad(((1,1), (1,1)))
    out = y.sum(axis=1)
    run_linear(*check_schedule(out, 1))
    self.assertListEqual(out.tolist(), [0, 12, 4, 0])

  def test_arange_transposed_descendants(self):
    Tensor.manual_seed(0)
    x = Tensor.randint(4, 1).realize()
    a = (Tensor.arange(4,)*x).T
    b = Tensor.randint(4, 4).realize()
    out = (a+b).sum()
    run_linear(*check_schedule(out, 1))
    np.testing.assert_equal(out.numpy(), ((np.arange(4)*x.numpy()).T+b.numpy()).sum())

  def test_arange_index(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(5, 2).realize()
    a = Tensor.arange(10)
    out = (x + a[2]).sum()
    run_linear(*check_schedule(out, 1))
    np.testing.assert_allclose(out.numpy(), (x.numpy()+np.arange(10)[2]).sum(), atol=1e-5, rtol=1e-6)

  def test_arange_index_contiguous(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(5, 2).realize()
    a = Tensor.arange(10).clone()
    out = (x + a[2]).sum()
    run_linear(*check_schedule(out, 2))
    np.testing.assert_allclose(out.numpy(), (x.numpy()+np.arange(10)[2]).sum(), atol=1e-5, rtol=1e-6)

  def test_arange_index_child(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(5, 2).realize()
    a = Tensor.arange(10)+1
    out = (x + a[2]).sum()
    run_linear(*check_schedule(out, 1))
    np.testing.assert_allclose(out.numpy(), (x.numpy()+(np.arange(10)+1)[2]).sum(), atol=1e-5, rtol=1e-6)

  def test_user_contiguous(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(5, 2).realize()
    a = (Tensor.arange(10)+1).clone()
    out = (x + a[2]).sum()
    run_linear(*check_schedule(out, 2))
    np.testing.assert_allclose(out.numpy(), (x.numpy()+(np.arange(10)+1)[2]).sum(), atol=1e-5, rtol=1e-6)

  @unittest.skip("BUFFER_VIEW no longer supported on non-disk devices")
  def test_arange_view_op(self):
    a = Tensor.arange(12).reshape(4, 3).shrink(((1, 2), (1, 3))).contiguous()
    sched = run_linear(*check_schedule(a, 1))
    self.assertIs(sched[1].ast.op, Ops.BUFFER_VIEW)
    np.testing.assert_equal(a.numpy(), [[4, 5]])

  @unittest.skipUnless(dtypes.half in supported_dtypes, "need half")
  def test_precompute_freqs_cis(self):
    from extra.models.llama import precompute_freqs_cis
    args = {"dim":32, "end":2048, "theta":10000}
    fused = precompute_freqs_cis(**args)
    run_linear(*check_schedule(fused, 1))
    if getenv("CHECK", 1):
      ref = precompute_freqs_cis(**args)
      run_linear(*check_schedule(ref, 1))
      np.testing.assert_equal(fused.numpy(), ref.numpy())

  def test_fuse_assign_contiguous(self):
    x = Tensor.zeros(4, 4, dtype=dtypes.int).contiguous().realize()
    a = Tensor.arange(8).reshape(4, 2)
    run_linear(*check_schedule(x.shrink((None, (0, 2))).assign(a.clone()), 2))
    np.testing.assert_equal(x.numpy(), [[0, 1, 0, 0], [2, 3, 0, 0], [4, 5, 0, 0], [6, 7, 0, 0]])

  def test_assign_non_contiguous_alt(self): self.test_assign_non_contiguous(alt=True)
  def test_assign_non_contiguous(self, alt=False):
    x = (Tensor.arange(16)-100).reshape(4,4).contiguous().realize()
    xref = x.numpy()
    if alt:
      y = Tensor.randint(2, 4).contiguous().realize()
      a = Tensor.arange(8).reshape(2, 4)+y
      tst = x.shrink(((0, 2), None)).assign(a).realize()
      xref[:2, :] = np.arange(8).reshape(2, 4)+y.numpy()
    else:
      y = Tensor.randint(4, 2).contiguous().realize()
      a = Tensor.arange(8).reshape(4, 2)+y
      tst = x.shrink((None, (0, 2))).assign(a).realize()
      xref[:, :2] = np.arange(8).reshape(4, 2)+y.numpy()
    np.testing.assert_equal(x.numpy(), xref)
    np.testing.assert_equal(tst.numpy(), a.numpy())

  def test_setitem_sched(self, mop=lambda x:x, expected_kcount=1):
    a = Tensor.arange(16, device="CPU").reshape(4, 4).contiguous().realize()
    a2 = mop(a)
    expected = (a+a2).tolist()
    a.assign(a+a2)
    linear, var_vals = a.linear_with_vars()
    kcount = len(linear.src)
    run_linear(linear, var_vals)
    self.assertListEqual(a.tolist(), expected)
    self.assertEqual(kcount, expected_kcount)
  def test_setitem_permuted_sched(self): self.test_setitem_sched(lambda x: x.T, 2)
  def test_setitem_paddded_sched(self): self.test_setitem_sched(lambda x: x.shrink_to(4, 1).pad_to(4, 4), 1)

  def test_setitem_const_fused(self):
    # https://github.com/tinygrad/tinygrad/issues/10690
    a = Tensor.arange(16).contiguous().realize()
    GlobalCounters.reset()
    a[4] = 3
    self.assertEqual(GlobalCounters.kernel_count, 0)
    a.realize()
    self.assertEqual(GlobalCounters.kernel_count, 1)
    self.assertListEqual(a.tolist(), [0, 1, 2, 3, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

  def test_no_extra_contiguous_on_setitem_assign_back(self):
    # pattern: contiguous copy, advanced setitem, assign back (e.g. torch backend _view_write)
    base = Tensor.arange(16).reshape(4, 4).clone()
    flat_base = base.reshape(16).contiguous()
    idx = Tensor([1,2,5,6], dtype=dtypes.int32)
    flat_base[idx] = Tensor([99,99,99,99])
    base.assign(flat_base.reshape(4, 4))
    sched = check_schedule(base, 4)
    run_linear(*sched)
    expected = list(range(16))
    for i, v in zip([1,2,5,6], [99,99,99,99]): expected[i] = v
    np.testing.assert_equal(base.reshape(16).numpy(), expected)

  def test_sparse_categorical_crossentropy_simple(self):
    X = Tensor([[0, 2, 3], [1, 2, 3]]).realize()
    Y = Tensor([1, 2]).realize()
    loss = X.sparse_categorical_crossentropy(Y)
    run_linear(*check_schedule(loss, 3))
    np.testing.assert_allclose(loss.item(), 0.878309, atol=1e-5, rtol=1e-6)

  def test_const_folding_alt(self):
    t = Tensor.full((2,), 1.)
    lt = (t < 0.)
    a = Tensor.empty(2).assign(t*lt.where(-1., 0.))
    b = Tensor.empty(2, dtype=dtypes.bool).assign(lt)
    Tensor.realize(a, b)
    self.assertEqual(a.tolist(), [0., 0.])
    self.assertEqual(b.tolist(), [False, False])

  @unittest.skipIf(Device.DEFAULT == "WEBGPU", "Validation error on WebGPU")
  def test_mnist_val(self):
    from tinygrad.nn.datasets import mnist
    import torch
    _, Y_train, _, _ = mnist()
    samples = Tensor.randint(BS:=getenv("BS", 512), high=cast(int,Y_train.shape[-1])).realize()
    yt = Tensor.randn(BS, 10).realize()
    loss = yt.sparse_categorical_crossentropy(Y_train[samples])
    run_linear(*check_schedule(loss, 4))
    loss_fused = loss.numpy()
    loss_ref = torch.nn.CrossEntropyLoss()(torch.tensor(yt.numpy()), torch.tensor(Y_train.numpy())[torch.tensor(samples.numpy())])
    np.testing.assert_allclose(loss_fused, loss_ref.numpy(), atol=1e-6, rtol=1e-6)

  def test_arange_fuse_grouped_children(self):
    X = Tensor.randn(4, 4).realize()
    r = (X+Tensor.arange(16).reshape(4, 4)).sum()
    out0 = r+2
    out1 = r+3
    run_linear(*check_schedule([out0, out1], 2)) # TODO: 1?
    r_ref = (X.numpy()+np.arange(16).reshape(4, 4)).sum()
    np.testing.assert_allclose(out0.numpy(), r_ref+2, rtol=2e-7)
    np.testing.assert_allclose(out1.numpy(), r_ref+3, rtol=2e-7)

  def test_recursive_swizzle(self):
    a = Tensor([1,2,3,4]).realize()
    for _ in range(24): a = a + a
    new_uop = a.reshape(4,1).realize().uop
    assert new_uop.base.op is Ops.BUFFER

  def test_self_assign_no_empty_kernel(self):
    for shape in [(3, 3), (4, 4)]:
      a = Tensor.ones(*shape).contiguous().realize()
      a.assign(a / 1)
      run_linear(*check_schedule(a, 0, filter_sink=False))
      self.assertListEqual(a.tolist(), [[1.]*shape[1]]*shape[0])

class TestLimitBufs(unittest.TestCase):
  @unittest.skipIf(CI and Device.DEFAULT == "NV", "crashes on NV CI")
  def test_limit_bufs_with_var(self):
    N = 31
    with Context(TRACK_MATCH_STATS=0, DEBUG=0):
      bufs = [Tensor([1]*10).contiguous().realize() for i in range(N)]

    vi = Variable("i", 0, 9).bind(1)
    vj = Variable("j", 0, 9).bind(2)
    root = bufs[0][vi] + bufs[0][vj]
    for X in range(1,N): root = root + bufs[X][vi] + bufs[X][vj]
    self.assertEqual(root.item(), N * 2)

  def test_limit_bufs_arange_condition(self):
    # WHERE with arange-based condition (pure index math, no device) and many buffer loads should not crash limit_bufs
    with Context(MAX_KERNEL_BUFFERS=8):
      N = 8
      idx = Tensor.arange(N)
      base = Tensor.zeros(N)
      for i in range(4):
        a, b = Tensor.rand(N).realize(), Tensor.rand(N).realize()
        base = (idx >= i).where(a + b, base)
      assert all(x > 0 for x in base.tolist())

class TestSwizzle(unittest.TestCase):
  def test_swizzle_simple(self):
    Tensor.manual_seed(0)
    with Context(DEBUG=0, TRACK_MATCH_STATS=0):
      a = Tensor.randint(32, 32).realize()
    r = (a+a).sum(1).sum(0)
    # double reduce collapses to a single reduce
    run_linear(*check_schedule(r, 1))
    self.assertEqual(r.numpy(), (a.numpy()+a.numpy()).sum(1).sum(0))

  def test_single_swizzle(self):
    Tensor.manual_seed(0)
    with Context(DEBUG=0, TRACK_MATCH_STATS=0):
      a = Tensor.randint(4, 1).realize()
      b = Tensor.ones((1, 1), dtype=a.dtype).contiguous().realize()
    # ADD(REDUCE(RESHAPE(LOAD)), LOAD) to ADD(REDUCE(RESHAPE(LOAD))), RESHAPE(LOAD)
    r = a.sum(0)+b
    run_linear(*check_schedule(r, 1))
    self.assertEqual(r.numpy(), a.numpy().sum(0)+1)

  def test_double_swizzle_possible(self):
    Tensor.manual_seed(0)
    with Context(DEBUG=0, TRACK_MATCH_STATS=0):
      a = Tensor.randint(4,).realize()
      b = Tensor.randint(4,).realize()
    # parallel reduce!
    add = a.sum(0)+b.sum(0)
    run_linear(*check_schedule(add, 1))
    self.assertEqual(add.numpy(), a.numpy().sum(0)+b.numpy().sum(0))

  def test_swizzle_reduceop(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(4,4).realize()
    y = Tensor.randn(4,4,4).realize()
    out = x.reshape(4,4,1).expand(4,4,4).sum(axis=(1,))+y
    run_linear(*check_schedule(out, 2)) # TODO: 1?
    np.testing.assert_allclose(out.numpy(), np.tile(x.numpy().reshape(4,4,1), (1,1,4)).sum(axis=1)+y.numpy())

  def test_permute_rewrite(self):
    x = Tensor.randn(4, 4, 16).realize()
    y = Tensor.randn(4, 1, 16).realize()
    z = Tensor.randn(4, 4, 1).realize()
    t = (x*y).sum(axis=(0, 2)).reshape(1, 4, 1).permute(0, 2, 1)+z
    run_linear(*check_schedule(t, 2)) # TODO: 1?
    t_np = (x.numpy()*y.numpy()).sum(axis=(0, 2)).reshape(1, 4, 1).transpose(0, 2, 1)+z.numpy()
    np.testing.assert_allclose(t.numpy(), t_np, atol=1e-6, rtol=1e-3)

  @unittest.skip("TODO: this swizzle isn't resolvable when there's a mask")
  def test_swizzle_failure_permute(self):
    a = Tensor.empty(45,65).T.reshape(65,1,45).pad((None,None,(0,45))).expand(65,45,90)
    b = Tensor.empty(45,65)
    a_reduce = a.sum(axis=(2,), keepdim=True).sum(axis=(1,))
    b_reduce = b.sum(axis=(0,))
    t = a_reduce+b_reduce
    run_linear(*check_schedule(t, 1))

  def test_parallel_reduce_possible(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(4, 2, 2).realize()
    y = Tensor.randn(4, 2, 2).realize()
    t = x.sum(axis=1)+y.sum(axis=1)
    run_linear(*check_schedule(t, 1))
    np.testing.assert_allclose(t.numpy(), x.numpy().sum(axis=1)+y.numpy().sum(axis=1), atol=1e-6, rtol=1e-3)

  # kernels can only have 1 or n in each dim
  def test_dont_parallelize_different_n(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(4, 2, 2).realize()
    y = Tensor.randn(4, 3, 2).realize()
    t = x.sum(axis=1)+y.sum(axis=1)
    run_linear(*check_schedule(t, 1))
    np.testing.assert_allclose(t.numpy(), x.numpy().sum(axis=1)+y.numpy().sum(axis=1), atol=1e-6, rtol=1e-3)

  def test_unsafe_pad(self):
    x = Tensor.full((2,2), 1.0).contiguous()
    y = x*x.sum((1,)).reciprocal()
    t = y.pad(((0,1),None))
    run_linear(*check_schedule(t, 3))
    np.testing.assert_equal(t.numpy(), [[0.5, 0.5], [0.5, 0.5], [0., 0.]])

zero_pm = UPat(Ops.CONST, arg=0)
class TestView(unittest.TestCase):
  def test_all_masked_out(self):
    # start with non CONST Ops
    a = Tensor.rand(10, 10).realize()
    # all masked out, degrades to const 0
    b = a.pad(((0, 10), None))[10:]
    sched = check_schedule(b.contiguous(), 1)
    run_linear(*sched)
    np.testing.assert_equal(b.numpy(), 0)

  def test_mask_dim_1(self):
    # mask out dim = 1 works too
    a = Tensor.rand(10, 10).realize()
    b = a.pad((None, (0, 10)))[:, 10:]
    assert b.shape == (10, 10)
    sched = check_schedule(b.contiguous(), 1)
    run_linear(*sched)
    np.testing.assert_equal(b.numpy(), 0)

  def test_partial_mask(self):
    # partial masked out does not degrade into CONST
    a = Tensor.rand(10, 10).realize()
    b = a.pad(((0, 5), None))[5:]
    assert b.shape == (10, 10)
    sched = check_schedule(b.contiguous(), 1)
    run_linear(*sched)
    np.testing.assert_allclose(b.numpy(), np.pad(a.numpy(), ((0, 5), (0, 0)))[5:])

  # a*VIEW(x), where VIEW(x) = 0
  # x collapses along with its children
  def test_parent_view_collapses(self):
    a = Tensor([1, 2])
    b = Tensor.arange(3).clone()
    bv = b.pad(((0, 2),))[-2:]
    # this becomes a late a*0
    late_mul = a*bv
    run_linear(*check_schedule(late_mul, 2))
    # the arange doesn't realize
    #self.assertIsNone(b.uop.base.realized)
    # mul doesn't realize
    #self.assertIsNone(late_mul.uop.base.realized)
    self.assertEqual(late_mul.tolist(), [0, 0])

  # SINK has two branches:
  # a*VIEW(x), where VIEW(x) = 0
  # x+2
  # as long as one child realizes, x does not collapse
  def test_parent_multiple_children_no_collapse(self):
    a = Tensor([1, 2])
    b = Tensor.arange(3).clone()
    bv = b.pad(((0, 2),))[-2:]
    late_mul = a*bv
    other_child = b+2
    s = check_schedule([late_mul, other_child], 3)
    # the arange becomes a BUFFER
    self.assertIs(b.uop.base.op, Ops.BUFFER)
    # NOTE: no longer checked
    # mul still collapses
    #self.assertIs(late_mul.uop.base.op, Ops.CONST)
    run_linear(*s)
    self.assertEqual(other_child.tolist(), [2, 3, 4])

@unittest.skipIf(Device.DEFAULT == "CPU", "tests copy from another device to cpu")
class TestCopyFolding(unittest.TestCase):
  def test_const_copy_is_free(self):
    b = Tensor(1).to("CPU") * 4
    run_linear(*check_schedule(b, 0, filter_sink=False))
    assert b.item() == 4

  def test_one_hot_with_copy(self):
    y = Tensor([1, 2, 3]).to("CPU")
    x = y.one_hot(10)
    check_schedule(x, 3, filter_sink=False)

  def test_const_copy_multi(self):
    x = Tensor.ones(1, device="CPU").to_(["CPU", "CPU:1"]) * 2
    run_linear(*check_schedule(x, 2, filter_sink=False))
    self.assertEqual(x.item(), 2.0)

  def test_late_const_copy_folding(self):
    a = Tensor.arange(3).realize()
    zeros = Tensor.zeros(3).realize()
    b = (a*zeros).to("CPU") + 1
    run_linear(*check_schedule(b, 1, filter_sink=False))
    self.assertListEqual(b.tolist(), [1, 1, 1])
    self.assertEqual(b.device, "CPU")

  def test_alu_after_copy(self):
    a = Tensor.ones((4,)).to("CPU")
    b = Tensor.empty(4, device="CPU")
    add = a+b
    assert all_same([x.device for x in add.uop.src]), f"ALU has different devices! {[x.device for x in add.src]}"
    add.schedule_linear()

  def test_alu_before_copy(self):
    buf = Tensor.ones(1).contiguous().realize()
    a = buf+1
    b = a.to("CPU")
    self.assertListEqual(b.tolist(), [2.])

  def test_copy_to_same_device(self):
    a = Tensor.empty(4).uop
    b = a.copy_to_device(a.device)
    check_schedule(b, 1, filter_sink=False) # TODO: 0?

  def test_copy_to_same_device_alt(self):
    a = Tensor.empty(4, 4).uop
    b = a.copy_to_device(a.device)
    check_schedule(b, 1, filter_sink=False) # TODO: 0?

  def test_copy_to_same_device_sched(self):
    a = Tensor.ones(4).contiguous().realize().uop.buf_uop
    t = Tensor(a.copy_to_device(a.device))
    linear, var_vals = t.linear_with_vars()
    assert len([call for call in linear.src if call.src[0].op is Ops.COPY]) == 0
    run_linear(linear, var_vals)
    assert t.uop.is_realized, f"didn't realize Tensor {t}"
    self.assertListEqual(t.tolist(), [1.,1.,1.,1.])

  def test_self_assign_same_device_copy(self):
    a = Tensor.ones(4, 4).contiguous().realize()
    # use copy_to_device to bypass Tensor.to() shortcircuit and force a real same-device COPY in the graph
    a.assign(Tensor(a.uop.copy_to_device(a.device), a.device))
    run_linear(*check_schedule(a, 2, filter_sink=False))
    self.assertListEqual(a.tolist(), [[1.]*4]*4)

  def test_clone(self):
    a = Tensor.empty(4)
    check_schedule(a.clone(), 1, filter_sink=False)

  def test_shrink_copy(self):
    a = Tensor.arange(4)
    view = a.shrink(((0, 2),))
    b = view.clone()
    run_linear(*check_schedule(b, 1, filter_sink=False))
    self.assertEqual(b.uop.base.buffer.size, 2)
    self.assertEqual(b.uop.numel(), 2)
    self.assertListEqual(b.tolist(), [0, 1])

  def test_expanded_copy(self):
    a = Tensor.arange(2)
    view = a.reshape(2, 1).expand(2, 2)
    b = view.clone()
    run_linear(*check_schedule(b, 1, filter_sink=False))
    self.assertEqual(b.uop.base.buffer.size, 4)
    self.assertEqual(b.uop.numel(), 4)
    self.assertListEqual(b.tolist(), [[0, 0], [1, 1]])

  def test_permuted_copy(self):
    a = Tensor.arange(4)
    b = a.reshape(2, 2).permute(1, 0)
    b.realize()
    self.assertListEqual(b.tolist(), [[0, 2], [1, 3]])

  def test_permute_on_disk(self):
    with open(temp('dt_arange_4_permute'), "wb") as f: f.write(Tensor.arange(4).realize().uop.base.buffer.as_memoryview())
    a = Tensor.empty(4, dtype=dtypes.int32, device=f"disk:{temp('dt_arange_4_permute')}")
    b = a.reshape(2, 2).permute(1, 0).to("CPU")
    b.realize()
    self.assertListEqual(b.tolist(), [[0, 2], [1, 3]])

  def test_permute_on_disk_contiguous(self):
    with open(temp('dt_arange_4_permute_contig'), "wb") as f: f.write(Tensor.arange(4).realize().uop.base.buffer.as_memoryview())
    a = Tensor.empty(4, dtype=dtypes.int32, device=f"disk:{temp('dt_arange_4_permute_contig')}")
    b = a.reshape(2, 2).permute(1, 0).contiguous().to("CPU")
    b.realize()
    self.assertListEqual(b.tolist(), [[0, 2], [1, 3]])

  def test_permute_after_shrink(self):
    a = Tensor.arange(5)
    b = a.shrink(((0, 4),)).reshape(2, 2).permute(1, 0).to("CPU")
    b.realize()
    self.assertListEqual(b.tolist(), [[0, 2], [1, 3]])

  # NOTE: disk permute must come after COPY
  def test_permute_after_shrink_on_disk(self):
    with open(temp('dt_arange_5_permute'), "wb") as f: f.write(Tensor.arange(5).realize().uop.base.buffer.as_memoryview())
    a = Tensor.empty(5, dtype=dtypes.int32, device=f"disk:{temp('dt_arange_5_permute')}")
    b = a.shrink(((0, 4),)).reshape(2, 2).permute(1, 0).to("CPU")
    b.realize()
    self.assertListEqual(b.tolist(), [[0, 2], [1, 3]])

class TestUOpBecome(unittest.TestCase):
  def test_setitem_offset(self):
    a = Tensor.full((16,), 0.).contiguous().realize()
    b = Tensor.full((16,), 1.).contiguous().realize()
    a_view = a[4:].reshape(3, 4).shrink(((0,2),(0,2))).reshape((4,))
    b.shrink(((0,4),)).assign(a_view).realize()
    self.assertListEqual(b.tolist(), [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

class TestFusionOp(unittest.TestCase):
  def test_contiguous_add(self):
    def test(contig=False):
      bt = Tensor(np.arange(16), dtype=dtypes.float32).reshape(4,4)
      x = bt.permute(1,0)
      if contig: x = x.contiguous()
      return (x.permute(1,0) + bt).data()
    assert test() == test(True)

  def test_expand_fuse(self):
    bt = Tensor(np.ones((10, 1)), dtype=dtypes.float32)
    out = (bt*2).expand(10,10).sum(1)
    run_linear(*out.linear_with_vars())
    outd = out.tolist()
    assert all(x == 20.0 for x in outd)

if __name__ == '__main__':
  unittest.main(verbosity=2)
