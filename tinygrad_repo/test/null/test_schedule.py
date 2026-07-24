# schedule tests that pass on NULL backend (no copyout needed)
import gc, unittest, time
from typing import cast
from tinygrad import nn, dtypes, Device, Tensor, getenv
from tinygrad.uop.ops import UOp, Ops, GroupOp, UPat, KernelInfo
from tinygrad.helpers import DEBUG, GlobalCounters, Context
from tinygrad.engine.realize import compile_linear, run_linear
from tinygrad.codegen import to_program

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

class TestBufferUOp(unittest.TestCase):
  # BUFFER has a ShapeTracker of shape=(n,) and stride=(1,)
  def test_buffer_has_buffer(self):
    buf = Tensor.empty(10)
    self.assertIsNotNone(buf.uop.buffer)
    self.assertEqual(buf.uop.shape, (10,))
    # the device Buffer remains unallocated until it's we run the schedule
    self.assertFalse(buf.uop.buffer.is_allocated())
    add = buf+1
    linear, var_vals = add.linear_with_vars()
    self.assertFalse(buf.uop.buffer.is_allocated())
    run_linear(linear, var_vals)
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
    with self.assertRaises(RuntimeError):
      permuted_view.uop.buffer # cannot access Buffer of a non contiguous VIEW

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
    a = Tensor(1)
    run_linear(*check_schedule(a, 0))
    self.assertIsNone(a.uop.base.realized)

  def test_var_does_not_realize(self):
    a = Tensor(UOp.variable("a", 0, 10).bind(1))
    run_linear(*check_schedule(a, 0))
    self.assertIsNone(a.uop.base.realized)

  def test_unused_var_not_in_var_vals(self):
    # unused variable should not appear in var_vals even when there's other work
    a = Tensor(UOp.variable("unused", 0, 10).bind(1))
    b = Tensor.empty(3) + 1
    _, var_vals = Tensor.linear_with_vars(a, b)
    self.assertEqual(var_vals, {})
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
    check_schedule(b, 0)  # contiguous shrink of a realized buffer is a zero-copy SLICE

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

class TestSimpleSchedule(unittest.TestCase):
  def test_reduce_doesnt_split(self):
    a = Tensor.empty(16,16).sum(axis=1)
    a1 = a.reshape(4,4)
    a2 = a.reshape(16,1,1)
    self.assertEqual(len(Tensor.schedule_linear(a1, a2).src), 1)

class TestSchedule(unittest.TestCase):
  def setUp(self):
    self.ctx = Context(SPLIT_REDUCEOP=0)
    self.ctx.__enter__()
  def tearDown(self):
    self.ctx.__exit__(None, None, None)

  def test_arange_avgpool2d(self, kcount=1):
    x = Tensor.arange(25).reshape(1,1,5,5).cast(dtypes.float32)
    t = x.avg_pool2d(padding=1).clone()
    linear, var_vals = t.linear_with_vars()
    self.assertEqual(len(linear.src), kcount)

  def test_arange_avgpool2d_fused_noopt(self):
    with Context(NOOPT=1): self.test_arange_avgpool2d(kcount=1)

  # when we're fusing a reduce, all ReduceOps must have the same N in the dimensions
  # all permutes, reshapes, expands and shrinks push through the reduce
  def test_arange_sum(self):
    a = Tensor.arange(6).reshape(3, 2).sum(axis=1).clone()
    check_schedule(a, 1)

  def test_arange_sum_alt(self):
    a = (Tensor.arange(5).reshape(1,5).expand(6,5)*Tensor(2)).reshape(1,6,5).sum(axis=2).clone()
    check_schedule(a, 1)

  def test_permute_arange(self):
    a = Tensor.arange(6).reshape(6, 1, 1).permute(2, 0, 1).sum(axis=1).clone()
    check_schedule(a, 1)

  def test_expand_buffer_before_cast(self):
    a = Tensor.zeros(4, 2, 1).realize().permute((1, 0, 2))
    b = a.cast(dtypes.half).expand((2, 4, 4))+2
    check_schedule(b, 1)

  def test_indexing_scalars(self):
    # cover each shape at all index corners
    for x, y in [(2,2), (2,3), (3,2), (3,3)]:
      for a, b in [(0,0), (0,y-1), (x-1,0), (x-1,y-1)]:
        X = Tensor.zeros(x, y).realize()
        xt = X[Tensor(a)][Tensor(b)]
        check_schedule(xt, 1)

  def test_push_pads_elementwise(self):
    x = Tensor.full((4,4), 2.).contiguous().realize()
    y = Tensor.full((4,4), 4.).contiguous().realize()
    z = (x.reciprocal()*y).pad((None, (0,1),)).sum()
    check_schedule(z, 1)

  def test_push_pads_contiguous(self):
    x = Tensor.full((4,1), 2.).contiguous()
    y = Tensor.full((4,4), 4.).contiguous()
    z = (x.reciprocal().expand(4,4)*y).pad((None, (0,1),)).sum()
    check_schedule(z, 1, [x,y])

  def test_allow_push_permutes(self):
    a = Tensor.empty(10,10,10).realize()
    b = Tensor.empty(10,10,1).realize()
    c = a.sum(axis=0, keepdim=True).permute(2,1,0) + b
    check_schedule(c, 1)

  def test_div_collapse_buffer(self):
    a = Tensor.full((4,), 4.0).contiguous().realize()
    b = Tensor.full((4,), 2.0).contiguous().realize()
    expr = (a*b)/b
    check_schedule(expr, 1)

  def test_div_collapse_const(self):
    a = Tensor.full((4,), 4.0).contiguous().realize()
    expr = a/a
    check_schedule(expr, 1)

  def test_div_collapse(self):
    a = Tensor.full((4,), 1.0).contiguous().realize()
    b = Tensor.full((4,), 2.0).contiguous().realize()
    c = Tensor.full((4,), 3.0).contiguous().realize()
    GlobalCounters.reset()
    expr = (a/b)/c
    expr.realize()
    self.assertEqual(GlobalCounters.kernel_count, 1)
    self.assertLessEqual(GlobalCounters.global_ops, 4*3)

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
    check_schedule(out, 1)

  def test_preserve_multistage_reduce(self):
    big_enough = getenv("REDUCEOP_SPLIT_THRESHOLD", 32768)
    x = Tensor.empty(big_enough).realize()
    with Context(SPLIT_REDUCEOP=1):
      out = (x - x.max(keepdim=True)).max()
      check_schedule(out, 3)

  def test_example_matmul_contig(self):
    x = Tensor.eye(64).clone().realize()
    y = Tensor.eye(64).clone().realize()
    z = y.matmul(x).sum()
    z.backward()
    out = x.grad.contiguous()
    check_schedule(out, 1)

  def test_multireduce_shrink(self):
    a = Tensor.empty(32, 32).realize()
    b = Tensor.empty(32, 32).realize()
    c = Tensor.empty(16).realize()
    a_out = a.sum(1)
    a_out = a_out[:16]
    b_out = b.sum(1)
    b_out = b_out[:16]
    out = a_out + b_out + c
    check_schedule(out, 1)

  def test_reduce_same_size(self):
    a = Tensor.empty(4, 4).realize()
    out0 = a.sum() + 2
    out1 = a.sum() + 4
    out2 = out0 * out1
    check_schedule([out0, out1, out2], 3) # TODO: 1?

  def test_reduce_multiple_paths(self):
    a = Tensor.empty(4, 4).realize()
    out0 = a.sum().exp2()
    # out1 has two paths to a.sum()
    out1 = a.sum() + out0
    check_schedule([out0, out1], 2) # TODO: 1?

  def test_multireduce_reduce_multiple_paths(self):
    a = Tensor.empty(4, 4).realize()
    out0 = a.sum().exp2()
    out1 = a.sum() + out0
    b = (a + out0 + out1)
    out2 = b.sum().exp2()
    out3 = b.sum() + out2
    # check_schedule([out0, out1, out2, out3], 1)
    check_schedule([out0, out1, out2, out3], 4)

  def test_reduce_ext_reduce_child(self):
    a = Tensor.empty(4, 4).realize()
    b = Tensor.empty(4, 4).realize()
    # b.sum() is not a descendant of the fused nodes
    out0 = a.sum() + b.sum() + 2
    out1 = a.sum() + b.sum() + 4
    # check_schedule([out0, out1], 1)
    check_schedule([out0, out1], 2)

  def test_reduce_multiple_paths_midreduce(self):
    a = Tensor.empty(4, 4).realize()
    r = a.sum()
    out0 = r.exp2()
    # reduce node in the indirect path from r to out2
    out1 = (a - out0).max()
    out2 = r + out1
    # check_schedule([r, out0, out1, out2], 1)
    check_schedule([r, out0, out1, out2], 4)

  def test_reduce_multiple_paths_midreduce_fused(self):
    a = Tensor.empty(4, 4).realize()
    b = Tensor.empty(4, 4).realize()
    out0 = a.sum() + 4
    out1 = b.max() + out0*2
    out2 = a.sum() + out1
    # check_schedule([out0, out1, out2], 1)
    check_schedule([out0, out1, out2], 3)

  def test_reduce_multiple_paths_midexpand(self):
    a = Tensor.empty(4, 4).realize()
    b = Tensor.empty(4, 4, 4).realize()
    r = a.sum()
    out0 = r.exp2()
    # e1 is in the indirect path from a.sum() to out1
    e = b + out0
    out1 = r + e[0][0][0]
    # check_schedule([r, out0, out1, e], 3) # 1 or 2 or 3? should be 1 (one reduce) but the different outputs might make it 3
    check_schedule([r, out0, out1, e], 4)

  def test_reduce_expand_child(self):
    a = Tensor.empty((32, 32, 32)).realize()
    b = Tensor.empty((1, 16)).realize()
    out0 = a.sum() + 2
    out1 = a.sum() + b
    check_schedule([out0, out1], 2)

  def test_scaled_dot_product_attention_multireduce_fusion(self):
    q = Tensor.empty(32,8,16,8).realize()
    k = Tensor.empty(32,8,16,8).realize()
    v = Tensor.empty(32,8,16,8).realize()
    out = Tensor.scaled_dot_product_attention(q,k,v)
    run_linear(*check_schedule(out, 4))
    out = Tensor.scaled_dot_product_attention(q,k,v)
    check_schedule(out, 4) # TODO: should be 1?

  def test_ugly_reduceop_pairing(self):
    a = Tensor.empty(4, 32).realize()
    b = Tensor.empty(4, 32).realize()
    c = Tensor.empty(4, 32).realize()
    out = (c * a.sum(-1, keepdim=True)).sum(-1) + (b * a.sum(-1, keepdim=True)).sum(-1) # a.sum has >1 children but should still fuse
    # check_schedule(out, 1)
    check_schedule(out, 2)

  def test_reduce_expand_reduce_fusion(self):
    a = Tensor.empty(4, 32).realize()
    out = (a+a.sum(-1, keepdim=True)).sum(-1)
    # check_schedule(out, 1)
    check_schedule(out, 2)

  def test_reduce_expand_reduce_expand_fusion(self):
    a = Tensor.empty(4, 32).realize()
    out = a+(a+a.sum(-1,keepdim=True)).sum(-1, keepdim=True)
    # check_schedule(out, 2)
    check_schedule(out, 3)

  def test_branching_reduces_and_expands_fusion(self):
    a = Tensor.empty(4, 32).realize()
    out0 = a+a.sum(-1, keepdim=True)
    out1 = out0.sum(-1)
    # check_schedule(out, 2)
    check_schedule([out0, out1], 3)

  def test_multireduce_fusion_simple_sequential(self):
    x = Tensor.empty(4, 32).realize()
    y = Tensor.empty(4, 32).realize()
    out = (y + x.sum(axis=-1, keepdim=True)).sum(axis=-1)
    # check_schedule(out, 1)
    check_schedule(out, 2)

  def test_multireduce_fusion_simple_parallel(self):
    x = Tensor.empty(4, 32).realize()
    y = Tensor.empty(4, 32).realize()
    out = y.sum(axis=-1) + x.sum(axis=-1)
    check_schedule(out, 1)

  def test_multireduce_fusion_sequential(self):
    x = Tensor.empty(4, 32).realize()
    out = x.std(-1)
    # check_schedule(out, 1)
    check_schedule(out, 2)

  def test_multireduce_fusion_parallel(self):
    x = Tensor.empty(4, 32).realize()
    y = Tensor.empty(4, 32).realize()
    out = x.std(-1) + y.std(-1)
    # check_schedule(out, 1)
    check_schedule(out, 3)

  def test_multireduce_diffops_sequential(self):
    x = Tensor.empty(4, 32).realize()
    out = (x - x.max(-1, keepdim=True)).sum(-1)
    # check_schedule(out, 1)
    check_schedule(out, 2)

  def test_multireduce_fusion_diffops_parallel(self):
    x = Tensor.empty(4, 32).realize()
    y = Tensor.empty(4, 32).realize()
    out = x.sum(-1) + y.max(-1)
    check_schedule(out, 1)

  def test_multireduce_fusion_sequential_and_parallel(self):
    x = Tensor.empty(4, 32).realize()
    y = Tensor.empty(4, 32).realize()
    mu = (x - x.max(axis=-1, keepdim=True)).mean(axis=-1, keepdim=True) + (y - y.max(axis=-1, keepdim=True)).mean(axis=-1, keepdim=True)
    out = [((x - mu).square().sum(-1)/x.shape[-1]).sqrt(), ((y - mu).square().sum(-1)/y.shape[-1]).sqrt()]
    # check_schedule(out, 1)
    check_schedule(out, 5)

  def test_multimatmul_fusion(self):
    a,b = Tensor.empty(4, 64).realize(), Tensor.rand(64,8).realize()
    c,d = Tensor.empty(4, 64).realize(), Tensor.rand(64,8).realize()
    out = a@b + c@d
    check_schedule(out, 1)

  def test_layernorm_onelayer_fusion(self):
    layer = nn.LayerNorm([10, 10])
    layer.weight = Tensor.empty(10,10).realize()
    layer.bias = Tensor.empty(10,10).realize()
    x = Tensor.empty(20, 5, 10, 10).realize()
    out = layer(x)
    # check_schedule(out, 2)
    check_schedule(out, 3)

  def test_multireduce_simple_chase(self):
    a = Tensor.empty(4, 4, 4).realize()
    r = (a + (a.sum(0, keepdim=True) + 6)).sum(0) * 2
    b = r.sum(0) + 8
    c = r.sum(1) + 12
    # schedule = check_schedule([b,c], 3)
    # self.assertIs(schedule[0].ast[0].src[0].arg, Ops.MUL)
    check_schedule([b,c], 4)

  def test_multireduce_push_permute_chase(self):
    a = Tensor.empty(4, 4, 4).realize()
    b = Tensor.empty(4, 4).realize()
    r = a.sum(2) + b
    d = r.T * 4
    e = r * (d + a).sum(2)
    check_schedule([d, e], 3) # make sure it doesn't fuse

  def test_multireduce_push_shrink_chase(self):
    a = Tensor.empty(16, 16).realize()
    b = Tensor.empty(4).realize()
    c = Tensor.empty(16, ).realize()
    d = Tensor.empty(16, 16).realize()
    r = a.sum(1) + c
    out = r[:4] * b + d.sum(1)[:4]
    check_schedule(out, 1)

  def test_multireduce_midreduce_nochase(self):
    a = Tensor.empty(16, 16).realize()
    b = (a.sum(0)+a.max(0) + a.max(1)+a.sum(1)) + 2
    check_schedule(b, 1)

  # pattern in test_transformer
  def test_partial_fuse1(self):
    a = Tensor.empty(16, 16).realize()
    b = Tensor.empty(16, 16).realize()
    c = a.sum() + 2
    d = (a.sum() - b.sum()) * 4
    # check_schedule([c, d], 1)
    check_schedule([c, d], 2)

  # pattern in conv
  def test_partial_fuse2(self):
    a = Tensor.empty(16, 16).realize()
    b = Tensor.empty(16, 16).realize()
    c = a.sum() + 2
    d = b.sum() - c
    # check_schedule([c, d], 1)
    check_schedule([c, d], 2)

  # pattern in adam
  def test_partial_fuse3(self):
    a = Tensor.empty(16, 16).realize()
    b = Tensor.empty(16, 16).realize()
    c = a.sum() + 2
    d = a.sum() * 2
    e = c * d
    f = b.sum() - e
    # check_schedule([c, d, e, f], 1)
    check_schedule([c, d, e, f], 4)

  def test_partial_fuse4(self):
    a = Tensor.empty(16, 16).realize()
    b = Tensor.empty(16, 16).realize()
    c = a.sum() + 2
    d = a.sum() * 2
    e = c * d
    f = (b - d).sum() - e
    # check_schedule([c, d, e, f], 1)
    check_schedule([c, d, e, f], 4)

  def test_pad_reduce_safe(self):
    a = Tensor.empty(3, 4, 5).realize()
    b = Tensor.empty(3, 4, 5).realize()
    out = (a + b).pad(((0, 1), (0, 1), (0, 1)), value=1.0).sum().contiguous()
    check_schedule(out, 1)

  def test_multireduce_pad_reduce_safe(self):
    a = Tensor.empty(3, 4, 5).realize()
    b = Tensor.empty(3, 4, 5).realize()
    out = (a.pad(((0, 1), (0, 1), (0, 1)), value=1.0).sum(keepdim=True)+b.pad(((0, 1), (0, 1), (0, 1)), value=1.0).sum()).contiguous()
    check_schedule(out, 1)

  def test_pad_reduce_unsafe(self):
    a = Tensor.empty(3, 4, 5).realize()
    out = a.log2().pad(((0, 1), (0, 1), (0, 1)), value=1.0).sum().contiguous()
    check_schedule(out, 1)

  def test_multireduce_pad_reduce_unsafe(self):
    a = Tensor.empty(3, 4, 5).abs().realize()
    b = Tensor.empty(3, 4, 5).abs().realize()
    out = (a.log2().pad(((0, 1), (0, 1), (0, 1)), value=1.0).sum()+b).abs().log2().pad(((0, 1), (0, 1), (0, 1)), value=1.0).sum().contiguous()
    check_schedule(out, 1)

  def test_shrink_pad_safe(self):
    a = Tensor.ones((3, )).contiguous().realize()
    b = Tensor.ones((3, )).contiguous().realize()
    out = (a + b).shrink(((0, 1),)).pad(((0, 1),)).contiguous()
    check_schedule(out, 1)

  def test_shrink_pad_unsafe(self):
    a = Tensor.ones((3, )).contiguous().realize()
    out = a.exp2().shrink(((0, 1),)).pad(((0, 1),)).contiguous()
    check_schedule(out, 1)

  def test_base_change_shrink_pad(self):
    a = Tensor.ones(3, 3).contiguous().realize()
    b = a.exp2()
    c = b[:-1, :-1]
    d = c.pad(((0, 1), (0, 1))) * 2
    check_schedule(d, 1)

  def test_base_change_expand_pad(self):
    a = Tensor.ones(3, 3).contiguous().realize()
    b = a.exp2()
    c = b[:, None, :]
    d = c.pad(((0, 0), (1, 1), (0, 0))) * 2
    check_schedule(d, 1)

  def test_fuse_arange_pad_replicate_mode(self):
    x = Tensor.empty(3,3,3,3)
    y = x.pad((-1,2,2,-1), mode="replicate")
    dx = y.sum().gradient(x)[0]
    check_schedule(dx, 0)

  # TODO like openpilot with imagef
  def test_base_change_expand_expand(self):
    a = Tensor.ones(4, 4).contiguous().realize()
    b = a.cast(dtypes.half).expand(2, 4, 4)
    c = b.cast(dtypes.int).expand(2, 2, 4, 4)
    check_schedule(c, 1)

  def test_base_change_pad_expand(self):
    a = Tensor.full((4, 4), 1.).contiguous().realize()
    b = Tensor.full((4, 4), 2.).contiguous().realize()
    c = (a + b).pad(((1, 1), (1, 1)))
    d = c.cast(dtypes.int).expand((2, 6, 6)) * 4
    check_schedule(d, 1)

  def test_pad_reduce_unsafe_multiview_st(self):
    P = Tensor.ones(3, 3).contiguous()
    sums = P.sum(axis=1, keepdim=True)
    P /= sums
    p = P[0]
    p = p.pad(((1, 0), ))
    p = p.repeat([2])
    # TODO: this should be 3 if fix store hazard worked correctly
    check_schedule(p, 4)

  def test_conv2d(self, allowed=4, dtype=dtypes.float):
    old_default_float, dtypes.default_float = dtypes.default_float, dtype
    dtypes.default_float = dtype
    Tensor.manual_seed(0)
    BS, CIN = 2, 3
    img = Tensor.randn(BS, CIN, 64, 64).realize()
    w = Tensor.uniform(16, CIN, 3, 3).realize()
    ret = Tensor.conv2d(img, w).relu().mean().backward()
    dtypes.default_float = old_default_float
    linear, var_vals = Tensor.linear_with_vars(ret, img.grad, w.grad)
    cnt = len([call for call in linear.src if call.src[0].op is Ops.SINK])
    assert cnt == allowed, f"expected {allowed} kernels, got {cnt}"

  def test_conv2d_half(self): self.test_conv2d(4, dtype=dtypes.half)

  def test_schedule_mem_used_with_inputs(self):
    gc.collect()
    base = GlobalCounters.mem_used
    x = Tensor.ones(256).contiguous().realize()
    (x+Tensor.ones(256).contiguous()).schedule_linear()
    gc.collect()
    self.assertEqual(GlobalCounters.mem_used-base, 1024)

  def test_image_dot_f16_fusion(self):
    with Context(FLOAT16=1, OPENPILOT_HACKS=1):
      def cnt():
        x, y, z = Tensor.empty((64, 64), dtype='float'), Tensor.empty((64, 64), dtype='float'), Tensor.empty((64, 64), dtype='float')
        a = (x @ y).relu()
        linear = compile_linear(((a @ z).relu() + a).schedule_linear())
        return len([call for call in linear.src if call.src[0].op is Ops.PROGRAM])

      with Context(IMAGE=1):
        self.assertEqual(cnt(), 5)

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

  def _test_fusion(self, shapes, f, cnt):
    with Context(DEBUG=0, TRACK_MATCH_STATS=0):
      check_schedule(f(*(Tensor.empty(s).realize() for s in shapes)), cnt)

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
    a = Tensor.arange(4).reshape(1, 4).clone().realize()
    casted_view = a.pad(((0, 1), (0, 0))).cast(dtypes.float)
    casted_view.realize()
    self.assertEqual(casted_view.uop.base.realized.size, 8)
    contig = casted_view.contiguous().realize()
    self.assertEqual(contig.uop.base.realized.size, 8)

  # NOTE: we only reorder CAST if it's an EXPAND
  def test_cast_after_shrink(self):
    a = Tensor.arange(4).reshape(1, 4).clone().realize()
    casted_view = a.shrink(((0, 1), (0, 2))).cast(dtypes.float)
    casted_view.realize()
    self.assertEqual(casted_view.uop.base.realized.size, 2)
    realized_view = casted_view.contiguous().realize()
    self.assertEqual(realized_view.uop.base.realized.size, 2)

  def test_cast_const_view(self):
    a = Tensor.ones((4, 4), dtype=dtypes.float32, buffer=False)
    casted_view = a.cast(dtypes.int32)
    run_linear(*check_schedule(casted_view, 0))
    realized_const_view = casted_view.contiguous()
    check_schedule(realized_const_view, 0)

  def test_simple_indexing(self):
    X = Tensor.empty(10, 10).realize()
    idxs = Tensor([0, 2]).realize()
    xt = X[idxs]
    check_schedule(xt, 1)

  def test_simple_indexing_alt(self):
    X = Tensor.arange(16).reshape(4, 4)
    xt = X[[1, 2], [-1, 2]]
    check_schedule(xt, 1)

  def test_advanced_indexing(self):
    X = Tensor.arange(10)+1
    xt = X[[0, -1]]
    check_schedule(xt, 1)

  def test_advanced_indexing_alt(self):
    X = Tensor.arange(6).reshape(3, 2)+1
    xt = X[[Tensor([2]), Tensor([1])]]
    check_schedule(xt, 1)

  def test_push_through_reshape(self):
    x = Tensor.empty(10, 20).realize()
    out = x.argmax(1)
    check_schedule(out, 2)

  def test_arange_push_through_expand(self):
    a = Tensor.arange(4,)
    b = Tensor.empty(4, 4).realize()
    out = (a+b).sum()
    check_schedule(out, 1)

  def test_argmin(self):
    x = Tensor.empty(4, 32).realize()
    out = x.argmin(-1)
    check_schedule(out, 2)

  def test_argmax(self):
    x = Tensor.empty(4, 32).realize()
    out = x.argmax(-1)
    check_schedule(out, 2)

  def test_arange_transposed(self):
    x = Tensor.empty(4, 1).realize()
    a = ((Tensor.arange(4,)*x).T).sum()
    check_schedule(a, 1)

  def test_div_padded_arange(self):
    x = Tensor.full((2,2), 16, buffer=False)
    y = x.div(Tensor.linspace(2, 8, steps=4, dtype=dtypes.int).reshape(2,2), rounding_mode="trunc").pad(((1,1), (1,1)))
    out = y.sum(axis=1).clone()
    check_schedule(out, 1)

  def test_arange_transposed_descendants(self):
    x = Tensor.empty(4, 1).realize()
    a = (Tensor.arange(4,)*x).T
    b = Tensor.empty(4, 4).realize()
    out = (a+b).sum()
    check_schedule(out, 1)

  def test_arange_index(self):
    x = Tensor.empty(5, 2).realize()
    a = Tensor.arange(10)
    out = (x + a[2]).sum()
    check_schedule(out, 1)

  def test_arange_index_contiguous(self):
    x = Tensor.empty(5, 2).realize()
    a = Tensor.arange(10).clone()
    out = (x + a[2]).sum()
    check_schedule(out, 2)

  def test_arange_index_child(self):
    x = Tensor.empty(5, 2).realize()
    a = Tensor.arange(10)+1
    out = (x + a[2]).sum()
    check_schedule(out, 1)

  def test_user_contiguous(self):
    x = Tensor.empty(5, 2).realize()
    a = (Tensor.arange(10)+1).clone()
    out = (x + a[2]).sum()
    check_schedule(out, 2)

  def test_sparse_categorical_crossentropy_simple(self):
    X = Tensor([[0, 2, 3], [1, 2, 3]]).realize()
    Y = Tensor([1, 2]).realize()
    loss = X.sparse_categorical_crossentropy(Y)
    check_schedule(loss, 3)

  def test_const_mul(self):
    b = Tensor(2) * 4
    self.assertIsNone(b.uop.device)
    run_linear(*check_schedule(b, 0, filter_sink=False))
    assert b.item() == 8

  def test_mnist_val(self):
    # from tinygrad.nn.datasets import mnist
    Y_train = Tensor.randint(60000, dtype='uchar').realize()
    samples = Tensor.randint(BS:=getenv("BS", 512), high=cast(int,Y_train.shape[-1])).realize()
    yt = Tensor.randn(BS, 10).realize()
    loss = yt.sparse_categorical_crossentropy(Y_train[samples])
    check_schedule(loss, 4)

  def test_arange_fuse_grouped_children(self):
    X = Tensor.empty(4, 4).realize()
    r = (X+Tensor.arange(16).reshape(4, 4)).sum()
    out0 = r+2
    out1 = r+3
    check_schedule([out0, out1], 2) # TODO: 1?

  def test_recursive_swizzle(self):
    a = Tensor([1,2,3,4]).realize()
    for _ in range(24): a = a + a
    new_uop = a.reshape(4,1).realize().uop
    assert new_uop.base.op is Ops.BUFFER

  def test_create_schedule_handles_multi_kernel_after_and_after_deps(self):
    def named_copy(name:str):
      def fxn(out:UOp, src:UOp) -> UOp:
        i = UOp.range(src.shape[0], 0)
        return out[i].store(src[i]).end(i).sink(arg=KernelInfo(name=name))
      return fxn

    src = Tensor.zeros(4, dtype=dtypes.float).contiguous().realize()
    dep = Tensor.zeros(4, dtype=dtypes.float).contiguous().realize()
    out = Tensor.zeros(4, dtype=dtypes.float).contiguous().realize()
    ones = Tensor.ones(4, dtype=dtypes.float).contiguous().realize()
    twos = Tensor.full((4,), 2.0, dtype=dtypes.float).contiguous().realize()
    threes = Tensor.full((4,), 3.0, dtype=dtypes.float).contiguous().realize()

    ka = Tensor.custom_kernel(src, ones, fxn=named_copy("ka"))[0]
    kb = Tensor.custom_kernel(src, twos, fxn=named_copy("kb"))[0]
    src_after = Tensor(src.uop.after(*ka.uop.src[1:], *kb.uop.src[1:]))

    kd = Tensor.custom_kernel(dep, threes, fxn=named_copy("kd"))[0]
    kc = Tensor.custom_kernel(out, src_after, fxn=named_copy("kc"))[0]
    out_after = Tensor(kc.uop.src[0].after(*kc.uop.src[1:], kd.uop))

    linear = out_after.schedule_linear()
    names = [call.src[0].arg.name for call in linear.src]
    self.assertEqual(set(names), {"ka", "kb", "kc", "kd"})
    self.assertEqual(names[-1], "kc")
    self.assertLess(names.index("ka"), names.index("kc"))
    self.assertLess(names.index("kb"), names.index("kc"))
    self.assertLess(names.index("kd"), names.index("kc"))

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
    with self.assertRaisesRegex(RuntimeError, "all buffers must be on the same device"): check_schedule(c, 2)

  def test_rand(self):
    x = Tensor.rand(32)
    check_schedule(x, 1, [Tensor._device_rng_counters[x.device]])

  def test_rand_recompute_arange(self):
    x = Tensor.rand(32)
    check_schedule(x, 1, [Tensor._device_rng_counters[x.device]])

  def test_empty_is_not_realized(self):
    a = Tensor.empty(10)
    child = a+2
    assert not a.uop.is_realized
    child.realize()
    assert a.uop.is_realized

  def test_realize_view_of_realized_has_empty_schedule(self):
    # views of realized buffers produce an empty schedule
    t = Tensor.zeros((3, 3)).contiguous().realize()
    v = t[1]  # view - is_realized but not has_buffer_identity
    assert v.uop.is_realized
    linear, _ = Tensor.linear_with_vars(v)
    self.assertEqual(len(linear.src), 0)

  # NOTE: because empty does not have a lowered kernel if realize is called on a childless empty, it never gets allocated.
  def test_childless_empty_never_allocates(self):
    a = Tensor.empty(10)
    a.realize()
    assert not a.uop.is_realized

  def test_simplify_padded_const(self):
    a, _ = Tensor.empty(1022).cummax(axis=0)
    check_schedule(a, 3)

  @unittest.skip("should this pass?")
  def test_contiguous_assign(self):
    a = Tensor.ones(10) * 2
    b = Tensor.empty(10)
    c = b.assign(a.contiguous())
    check_schedule(c, 1)

  def test_basic_binop_fusion(self):
    a = Tensor.empty(10)
    b = Tensor.empty(10)
    c = Tensor.empty(10)
    d = a+b+c
    check_schedule(d, 1)

  def test_basic_binop_fusion_assign(self):
    a = Tensor.empty(10)
    b = Tensor.empty(10)
    c = Tensor.empty(10)
    d = a+b+c
    e = Tensor.empty(10).assign(d)
    check_schedule(e, 1)

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

  def test_mulacc_fusion_assign(self):
    a = Tensor.empty(10)
    b = Tensor.empty(10)
    c = (a*b).sum()
    d = Tensor.empty(1).assign(c)
    check_schedule(d, 1)

  def test_detach_assign(self):
    a = Tensor.ones(4, 4).contiguous().realize()
    buf1, buf2 = Tensor.empty(4, 4).contiguous(), Tensor.empty(4, 4).contiguous()
    r = buf2.assign(buf1.assign(a + 1.0) * 2.0)
    check_schedule(r.detach().contiguous(), 2)

  def test_contiguous_backward_assign(self):
    a = Tensor.ones(4, 4).contiguous().realize()
    buf1, buf2 = Tensor.empty(4, 4).contiguous(), Tensor.empty(4, 4).contiguous()
    r = buf2.assign(buf1.assign(a + 1.0) * 2.0)
    check_schedule(r.contiguous_backward().contiguous(), 2)

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

  def test_reduce_permute_binop_fusion(self):
    a = Tensor.empty(10,10,10)
    b = Tensor.empty(10,10,1)
    c = a.sum(axis=0, keepdim=True).permute(2,1,0) + b
    check_schedule(c, 1)

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
    check_schedule(d, 1, [c])

  # failing in new lazy
  def test_cache_binaryop_transpose(self):
    a = Tensor.empty(10,10)
    b = Tensor.empty(10,10)
    c = (a.T*b.T).T #.contiguous()
    d = a*b
    check_schedule(d, 1, [c])

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
    linear, _ = check_schedule(out, 2)
    reduceops = [x for si in linear.src for x in si.src[0].toposort() if x.op is Ops.REDUCE]
    assert len(reduceops) == 2

  def test_cache_reduce_multiple_children(self):
    x = Tensor.empty(32)
    y = Tensor.empty(4, 4)
    r0 = x.mean(axis=0, keepdim=True)
    r1 = (x - r0).sum(axis=0).div(2)
    out0 = r0 + y
    out1 = r1 + y
    linear, _ = check_schedule([out0, out1], 3)
    reduceops = [x for si in linear.src for x in si.src[0].toposort() if x.op is Ops.REDUCE]
    self.assertEqual(len(reduceops), 2) # why is RANGEIFY different?

  def test_dedup_assign(self):
    a = Tensor.ones(4).contiguous().realize()
    b = Tensor.full((4,), 2.).contiguous()
    first = a.assign(b)
    second = a.assign(b)
    check_schedule([first, second], 2) # TODO: 1?

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

  def test_const_realize(self):
    t = Tensor.ones(2, buffer=False)
    check_schedule(t[0], 0)
    check_schedule(t[1], 0)

  def test_fold_double_unary(self):
    y = Tensor.empty(2)
    out = y.sum(keepdim=True).sqrt().neg()
    check_schedule(out, 1)

  #@unittest.skip("may want to reconsider this")
  def test_fold_batchnorm(self):
    with Context(TRAINING=1):
      img = Tensor.empty(1,32,4,4)
      bn = nn.BatchNorm2d(32, track_running_stats=False)
      out = bn(img)
      check_schedule(out, 3, nn.state.get_parameters(bn))

  def test_fold_conv_batchnorm_notrain(self):
    with Context(TRAINING=0):
      img = Tensor.empty(1,3,8,8)
      c1 = nn.Conv2d(3,32,3)
      bn = nn.BatchNorm2d(32, track_running_stats=True)
      out = bn(c1(img)).relu()
      check_schedule(out, 1, [c1.weight, c1.bias, *nn.state.get_parameters(bn)])

  def test_fold_conv_batchnorm_notrain_no_running_stats(self):
    with Context(TRAINING=0):
      img = Tensor.empty(1,3,8,8)
      c1 = nn.Conv2d(3,32,3)
      bn = nn.BatchNorm2d(32, track_running_stats=False)
      out = bn(c1(img)).relu()
      check_schedule(out, 4, [c1.weight, c1.bias, *nn.state.get_parameters(bn)])

  def test_fold_conv_batchnorm(self):
    with Context(TRAINING=1):
      img = Tensor.empty(1,3,8,8)
      c1 = nn.Conv2d(3,32,3)
      bn = nn.BatchNorm2d(32, track_running_stats=False)
      out = bn(c1(img)).relu()
      check_schedule(out, 4, [c1.weight, c1.bias, *nn.state.get_parameters(bn)])

  def test_fold_conv_batchnorm_optim(self, adam=False):
    optim, cnt = (nn.optim.Adam, 29) if adam else (nn.optim.SGD, 15)
    with Context(TRAINING=1):
      img = Tensor.ones(1,3,4,4)
      c1 = nn.Conv2d(3,32,3)
      bn = nn.BatchNorm2d(32, track_running_stats=False)
      _realize_weights([c1, bn])
      opt = optim(nn.state.get_parameters([c1, bn]))
      Tensor.realize(img, *nn.state.get_parameters(opt))
      img_bn = bn(c1(img)).elu().sum()
      opt.zero_grad()
      img_bn.backward()
      check_schedule(opt.schedule_step(), cnt)
  def test_fold_conv_batchnorm_optim_adam(self): self.test_fold_conv_batchnorm_optim(True)

  def test_fold_batchnorm_backward(self):
    with Context(TRAINING=1):
      x = Tensor.empty((2, 16, 8, 8)).contiguous()
      bn = nn.BatchNorm2d(16)
      fw = bn(x).contiguous_backward().relu().contiguous()
      fw.sum().backward()
      # TODO: this is too many
      check_schedule([x.grad, bn.weight.grad, bn.bias.grad, fw], 10, nn.state.get_parameters(bn))

  def test_fold_conv_relu(self):
    c1 = nn.Conv2d(3,16,3)
    # run
    img = Tensor.ones(2,3,64,64)
    out = c1(img).relu()
    check_schedule(out, 1, [c1.weight, c1.bias, img])

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
    check_schedule(out, 1)

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
    check_schedule(out, 1)

  def test_children_dont_push(self):
    a = Tensor.empty(10, 10, 1)
    b = Tensor.empty(10, 10, 1)
    d = (a+b).expand(10, 10, 10)
    e = (a+b).permute(2,1,0)
    f = d+e
    check_schedule(f, 1)

  # failing in new lazy
  @unittest.skip("always fusing elementwise")
  def test_dont_fuse_binops_with_children(self):
    a = Tensor.empty(10)
    b = Tensor.empty(10)
    c = Tensor.empty(10)
    keep_me = a+b
    e = keep_me.sum() # noqa: F841 give keep_me a child (NOTE: BinaryOps won't be a child since it will instant fuse)
    d = keep_me+c
    check_schedule(d, 2)
    check_schedule(keep_me, 0, [d])

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

  def test_expand_fuse(self):
    a = Tensor.empty(1, 16)
    b = Tensor.empty(1, 16)
    c = a * b
    d = Tensor.empty(8192, 16)
    e = c * d
    check_schedule(e, 1)

  # this is the failing case in openpilot...it's very simple like this
  def test_image_conv_fusion(self):
    with Context(OPENPILOT_HACKS=1):
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
      #check_schedule(x, 5)
      check_schedule(x, 7)

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

  def _alu_from_tensor(self, t:Tensor):
    s = [s for s in t.schedule_linear().src if s.src[0].op is Ops.SINK]
    self.assertEqual(len(s), 1)
    return [u.op for u in s[0].src[0].toposort() if u.op in GroupOp.ALU]

  def test_2_pow_is_exp2(self):
    t = 2.0 ** Tensor([1.0, 2.0, 3.0])
    self.assertEqual(self._alu_from_tensor(t), [Ops.EXP2])

  def test_pow_05_is_sqrt(self):
    t = Tensor([1.0, 2.0, 3.0]) ** 0.5
    self.assertEqual(self._alu_from_tensor(t), [Ops.SQRT])

  def test_pow_neg_05_is_rsqrt(self):
    t = Tensor([1.0, 2.0, 3.0]) ** -0.5
    self.assertEqual(self._alu_from_tensor(t), [Ops.RECIPROCAL, Ops.SQRT])

  def test_pow_2_has_1_mul(self):
    t = Tensor([1.0, 2.0, 3.0]) ** Tensor(2.0)
    self.assertEqual(self._alu_from_tensor(t), [Ops.MUL])

  def test_pow_8_has_3_muls(self):
    t = Tensor([1.0, 2.0, 3.0]) ** 8
    self.assertEqual(self._alu_from_tensor(t), [Ops.MUL, Ops.MUL, Ops.MUL])

  def test_any_has_no_alu(self):
    t = Tensor([True, False, True]).any()
    self.assertEqual(self._alu_from_tensor(t), [])

  def test_all_has_no_alu(self):
    t = Tensor([True, False, True]).all()
    self.assertEqual(self._alu_from_tensor(t), [])

  # TODO: min() should be no ALU ops, like max(). currently it's _inverse().max()._inverse() which adds two negations
  def test_min_float_has_two_mul(self):
    t = Tensor([1.0, 2.0, 3.0]).min()
    self.assertEqual(self._alu_from_tensor(t), [Ops.MUL, Ops.MUL])

  # TODO: min() should be no ALU ops, like max(). currently it's _inverse().max()._inverse() which adds two negations
  def test_min_int_has_two_xor(self):
    t = Tensor([1, 2, 3]).min()
    self.assertEqual(self._alu_from_tensor(t), [Ops.XOR, Ops.XOR])

  @unittest.skip("const folding is removed")
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
    check_schedule(out, 1)

  def test_two_elus_sum(self):
    x = Tensor.empty(32, 32)
    y = Tensor.empty(32, 32)
    out = x.sum(1).relu().elu() + y.sum(1).relu().elu()
    check_schedule(out, 1)

  def test_multistage_reduce(self):
    x = Tensor.empty(32, 32, 32)
    out = x.sum(2).relu().sum(1)
    check_schedule(out, 1)

  def test_multistage_reduce_fork(self):
    x = Tensor.empty(32, 32, 32)
    x = x.sum(2)
    out2 = x + 1
    out = x.relu().sum(1) + out2[0]
    check_schedule(out, 2)

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
    check_schedule(out, 1)

  def test_reduce_shrink_child(self):
    a = Tensor.empty(100, 100)
    b = Tensor.empty(10,)
    c = a.sum() + b[0]
    d = a.sum() + 2
    check_schedule([c, d], 2) # TODO: 1?

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

  def test_softmax_upcast(self):
    # input half, softmax in float
    Tensor.manual_seed(0)
    x = Tensor.randn(4, 12, 64, 64, dtype=dtypes.half).realize()
    out = x.softmax(dtype=dtypes.float)
    linear = out.schedule_linear()
    self.assertEqual(len(linear.src), 3)
    # max reduction stays in input dtype (no numerical loss), upcast happens after subtracting max
    self.assertEqual(linear.src[0].src[1].dtype, dtypes.half)
    self.assertEqual(linear.src[1].src[1].dtype, dtypes.float)
    self.assertEqual(linear.src[2].src[1].dtype, dtypes.float)

  def test_softmax_backward(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(4, 12, 64, 64).realize()
    x.softmax().sum().backward()
    run_linear(*check_schedule(x.grad, 4))

  def test_logsumexp_backward(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(4, 12, 64, 64).realize()
    x.logsumexp(-1).sum().backward()
    run_linear(*check_schedule(x.grad, 3))

  def test_logcumsumexp_backward(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(4, 512).realize()
    x.logcumsumexp(-1).sum().backward()
    run_linear(*check_schedule(x.grad, 3))

  def test_scaled_dot_product_attention_fusion(self):
    x, y, z, m = (Tensor.empty(32, 8, 16, 16) for _ in range(4))
    out = Tensor.scaled_dot_product_attention(x, y, z, attn_mask=m)
    check_schedule(out, 4)

  def test_scaled_dot_product_attention_causal_fusion(self):
    x, y, z = (Tensor.empty(32, 8, 16, 16) for _ in range(3))
    out = Tensor.scaled_dot_product_attention(x, y, z, is_causal=True)
    check_schedule(out, 4)

  def test_adam_step_fusion(self):
    with Context(TRAINING=1):
      x = Tensor.empty(4, 64, 32)
      layer = nn.Linear(32, 32*4)
      _realize_weights(layer)
      opt = nn.optim.Adam(nn.state.get_parameters(layer), lr=1e-4)
      Tensor.realize(*nn.state.get_parameters(opt))
      layer(x).relu().sum().backward()
      check_schedule(opt.schedule_step(), 13)

  def test_adam_conv_fuse(self):
    with Context(TRAINING=1):
      img = Tensor.empty(2,3,4,4)
      c1 = nn.Conv2d(3,32,3)
      _realize_weights(c1)
      opt = nn.optim.Adam(nn.state.get_parameters(c1), lr=1e-4)
      Tensor.realize(*nn.state.get_parameters(opt))
      opt.zero_grad()
      c1(img).relu().sum().backward()
      check_schedule(opt.schedule_step(), 13)

  def test_adam_2convs_fuse(self):
    with Context(TRAINING=1):
      img = Tensor.empty(2,3,4,4)
      c1 = nn.Conv2d(3,16,3,bias=False)
      c2 = nn.Conv2d(16,32,2,bias=False)
      _realize_weights([c1, c2])
      opt = nn.optim.Adam(nn.state.get_parameters([c1, c2]), lr=1e-4)
      Tensor.realize(*nn.state.get_parameters(opt))
      opt.zero_grad()
      c2(c1(img).relu()).relu().sum().backward()
      check_schedule(opt.schedule_step(), 15)

  def test_sgd_conv_fuse(self):
    with Context(TRAINING=1):
      img = Tensor.empty(2,3,4,4)
      c1 = nn.Conv2d(3,32,3)
      _realize_weights(c1)
      opt = nn.optim.SGD(nn.state.get_parameters(c1))
      opt.zero_grad()
      c1(img).relu().sum().backward()
      check_schedule(opt.schedule_step(), 5) # TODO: 3?

  def test_sgd_2convs_fuse(self):
    with Context(TRAINING=1):
      img = Tensor.empty(2,3,4,4)
      c1 = nn.Conv2d(3,16,3,bias=False)
      c2 = nn.Conv2d(16,32,2,bias=False)
      _realize_weights([c1, c2])
      opt = nn.optim.SGD(nn.state.get_parameters([c1, c2]))
      opt.zero_grad()
      c2(c1(img).relu()).relu().sum().backward()
      check_schedule(opt.schedule_step(), 7)

  def test_fold_2convs_sgd_nesterov_momentum_wd(self):
    with Context(TRAINING=1):
      img = Tensor.empty(2,3,4,4)
      c1 = nn.Conv2d(3,16,3,bias=False)
      c2 = nn.Conv2d(16,32,2,bias=False)
      _realize_weights([c1, c2])
      opt = nn.optim.SGD(nn.state.get_parameters([c1, c2]), nesterov=True, momentum=0.9, weight_decay=0.1)
      Tensor.realize(*nn.state.get_parameters(opt))
      opt.zero_grad()
      c2(c1(img).relu()).relu().sum().backward()
      check_schedule(opt.schedule_step(), 11)

  def test_sgd_4convs_fuse(self):
    with Context(TRAINING=1):
      img = Tensor.empty(2,3,16,16)
      c1 = nn.Conv2d(3,4,3,bias=False)
      c2 = nn.Conv2d(4,8,3,bias=False)
      c3 = nn.Conv2d(8,16,3,bias=False)
      c4 = nn.Conv2d(16,32,3,bias=False)
      _realize_weights([c1, c2, c3, c4])
      opt = nn.optim.SGD(nn.state.get_parameters([c1, c2, c3, c4]))
      opt.zero_grad()
      c4(c3(c2(c1(img).relu()).relu()).relu()).relu().sum().backward()
      check_schedule(opt.schedule_step(), 15)

  def test_sgd_4convs_fuse_conv_bw(self):
    with Context(TRAINING=1):
      img = Tensor.empty(2,3,16,16)
      c1 = nn.Conv2d(3,4,3,bias=False)
      c2 = nn.Conv2d(4,8,3,bias=False)
      c3 = nn.Conv2d(8,16,3,bias=False)
      c4 = nn.Conv2d(16,32,3,bias=False)
      _realize_weights([c1, c2, c3, c4])
      opt = nn.optim.SGD(nn.state.get_parameters([c1, c2, c3, c4]))
      opt.zero_grad()
      c4(c3(c2(c1(img).relu()).relu()).relu()).relu().sum().backward()
      check_schedule(opt.schedule_step(), 15)

  def test_reduce_simple_chase(self):
    a = Tensor.empty(4, 4, 4)
    r = a.sum(0) + 6
    b = r.sum(0) * 4
    c = r.sum(1) * 2
    check_schedule([b, c], 3)

  def test_push_permute_chase(self):
    a = Tensor.empty(4, 4, 4)
    b = Tensor.empty(4, 4)
    r = a.sum(2) + b
    d = r.T * 4
    e = r * d
    check_schedule([d, e], 3)

  def test_push_shrink_chase(self):
    a = Tensor.empty(16, 16)
    b = Tensor.empty(4)
    c = Tensor.empty(16, )
    r = a.sum(1) + c
    d = r[:4] * b
    check_schedule(d, 1)

  def test_midreduce_nochase(self):
    a = Tensor.empty(16, 16)
    b = (a.sum(0) + a.max(1)) + 2
    check_schedule(b, 1)

  def test_bitcast_fuses(self):
    x = Tensor.empty(1, dtype=dtypes.float32)
    a = x.exp2().bitcast(dtypes.int32)
    b = x.bitcast(dtypes.int32)
    check_schedule(a+b, 1) # this should fuse when it makes sense

  def test_reduceop_reshape_dont_push(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(10, 20).realize()
    out = x.argmax(1)
    run_linear(*check_schedule(out, 2))

  def test_resnet_conv2d(self):
    x = Tensor.empty(1, 8, 32, 32)
    w1 = Tensor.empty(8, 8, 3, 3)
    w2 = Tensor.empty(8, 8, 1, 1)
    out = x.conv2d(w1).conv2d(w2)
    check_schedule(out, 2)

  def test_schedule_mem_used(self):
    gc.collect()
    base = GlobalCounters.mem_used
    Tensor.ones(256).contiguous().realize()
    Tensor.ones(5, 5).contiguous().schedule_linear()
    gc.collect()
    self.assertEqual(GlobalCounters.mem_used-base, 0)

  def test_const_schedule(self):
    constv = Tensor.empty(2, 2).uop.const_like(10)
    check_schedule(constv, 0)

  def test_const_schedule_contig(self):
    constv = Tensor.empty(2, 2).uop.const_like(10).contiguous()
    check_schedule(constv, 0)

  def test_advanced_simple_indexing_combined(self):
    X = Tensor.arange(16).reshape(4, 4)
    xt = X[1:2, [-1, 2]]
    check_schedule(xt, 1)

  def test_arange_index_shrink(self):
    Tensor.manual_seed(0)
    with Context(TRACK_MATCH_STATS=0):
      x = Tensor.randn(11).realize()
    a = Tensor.arange(22)
    out = (x + a[:11]).sum()
    check_schedule(out, 1)

  def test_fuse_arange_avg_pool2d_ceil_mode(self):
    x = Tensor.avg_pool2d(Tensor.empty(1,1,6,6), kernel_size=(3,3), padding=1, stride=3, ceil_mode=True)
    linear, _ = check_schedule(x, 1)
    self.assertEqual(len([x for x in linear.src[0].src[0].backward_slice_with_self if x.op is Ops.REDUCE]), 1)

  def test_fuse_arange_pad_circular_mode_bw(self):
    x = Tensor.empty(1,1,5,5,5)
    out = x.pad((1,2,3,5,1,2), mode="circular")
    g = out.sum().gradient(x)[0].clone()
    linear, _ = check_schedule(g, 1)
    self.assertEqual(len([x for x in linear.src[0].src[0].backward_slice_with_self if x.op is Ops.REDUCE]), 0)

  def test_resnet_block(self):
    with Context(TRAINING=0):
      in_planes, planes = 64, 64
      conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
      bn1 = nn.BatchNorm2d(planes)
      conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=1, bias=False)
      bn2 = nn.BatchNorm2d(planes)
      x = Tensor.empty(1, 64, 32, 32)
      out = bn1(conv1(x)).relu()
      out = bn2(conv2(out))
      out = (out + x).relu()
      run_linear(*check_schedule(out, 2, [conv1.weight, conv2.weight, *nn.state.get_parameters(bn1), *nn.state.get_parameters(bn2)]))

class TestSwizzle(unittest.TestCase):
  def test_softmax_one_kernel(self):
    Tensor.manual_seed(0)
    with Context(DEBUG=0, TRACK_MATCH_STATS=0):
      a = Tensor.randn(32, 32).realize()
    t = a.softmax()
    check_schedule(t, 3) # TODO: 1?

  def test_argmax_one_kernel(self):
    Tensor.manual_seed(0)
    with Context(DEBUG=0, TRACK_MATCH_STATS=0):
      a = Tensor.randn(10, 20).realize()
    t = a.argmax(0)
    check_schedule(t, 2) # TODO: 1?

class TestView(unittest.TestCase):
  def test_zero_size_alt(self):
    a = Tensor.empty(135, 0, 9)
    b = a.pad(((0, 0), (0, 0), (18, 0)))
    check_schedule(b, 0)

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
    self.assertEqual(add.uop.numel(), 16)
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
    assert UPat(Ops.BUFFER).match(add.uop.base, {})

  # sometimes we prefer to perform an op before movement ops, in this case we should stack the mops on top of the new buffer

  @unittest.skip("no longer supported")
  def test_reorder_expand(self):
    a = Tensor.empty(4, 1)
    b = a.expand(4, 4).reciprocal()
    check_schedule(b, 1)
    self.assertEqual(b.uop.base.buffer.size, 4)
    self.assertEqual(b.uop.shape, (4, 4))

  def test_reorder_expand_alt(self):
    x = Tensor.empty(4, 1)
    y = Tensor.empty(4, 1)
    img = Tensor.empty(4, 4)
    z = (img*x) / y
    check_schedule(z, 1)

  # TODO: rangeify doesn't yet cleanup this kind of re-indexing
  @unittest.expectedFailure
  def test_become_existing_buffer(self):
    a = Tensor.empty(4, 4)
    b = a*1
    assert UPat(Ops.MUL).match(b.uop, {}) # before scheduling it's a mul
    check_schedule(b, 0)
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

  @unittest.skip("const folding is removed")
  def test_become_const_in_base(self):
    a = Tensor.empty(4)
    b = a*0
    assert UPat(Ops.MUL).match(b.uop, {}) # before scheduling it's a mul
    check_schedule(b, 0)
    assert UPat(Ops.CONST, arg=0).match(b.uop.base, {}) # scheduling replaces the tensor uop with a VIEW(BUFFER)

  @unittest.skip("const folding is removed")
  def test_become_const_from_const(self):
    const_add = Tensor(1)+Tensor(2)
    assert UPat(Ops.ADD).match(const_add.uop, {})
    check_schedule(const_add, 0)
    assert UPat(Ops.CONST, arg=3).match(const_add.uop.base, {})

  # tensors can become another realized tensor source
  @unittest.expectedFailure
  def test_become_existing_buf_simple(self):
    a = Tensor.empty(4, 4)
    b = a+0
    check_schedule(b, 0)
    assert b.uop.base.op is Ops.BUFFER
    self.assertIs(a.uop, b.uop)

  # they can also chain other movement ops on top of the tensor source
  @unittest.expectedFailure
  def test_become_existing_buf_view(self):
    a = Tensor.empty(4, 4)
    b = a.permute((1, 0))+0
    check_schedule(b, 0)
    self.assertEqual(b.uop.st, a.uop.permute((1, 0)).st)

  @unittest.expectedFailure
  def test_become_existing_buf_view_alt(self):
    a = Tensor.empty(4, 4)
    b = a.permute((1, 0)).reshape((8, 2))+0
    check_schedule(b, 0)
    self.assertEqual(b.uop.st, a.uop.permute((1, 0)).reshape((8, 2)).st)

  # they can also have other base parents that simplified, in that case we just backtrack to the chained mops
  @unittest.expectedFailure
  def test_become_existing_buf_complex(self):
    a = Tensor.empty(4, 4)
    b = (a.permute((1, 0))+0).reshape((8, 2))+0
    check_schedule(b, 0)
    self.assertEqual(b.uop.st, a.uop.permute((1, 0)).reshape((8, 2)).st)
    assert b.uop.base.op is Ops.BUFFER

  @unittest.expectedFailure
  def test_become_multiple_choices(self):
    a = Tensor.empty(16)
    b = (a.reshape(1, 1, 4, 1, 4)+0).reshape(1, 1, 4, 4).shrink(((0, 1), (0, 1), (0, 3), (0, 3)))+0
    c = (a.reshape(1, 1, 4, 4)+0).shrink(((0, 1), (0, 1), (0, 3), (0, 3)))+0
    check_schedule([b, c], 0)
    from tinygrad.helpers import all_same
    assert all_same([x.uop.base.realized for x in [a,b,c]])

  @unittest.skip("not clear if we want this")
  def test_setitem_becomes_subbuffer(self):
    a = Tensor.full((4,), 2.).contiguous().realize()
    b = a.shrink(((0, 2),)).assign(Tensor.full((2,), 1.0))
    b.realize()
    assert a.uop.is_realized
    assert a.uop.buffer._base is None
    assert b.uop.op_in_backward_slice_with_self(Ops.SHRINK)
    assert b.uop.base is a.uop.base

class TestFusionOp(unittest.TestCase):
  def test_recursive_add(self):
    st = time.perf_counter()
    a = Tensor([1,2,3,4])
    for _ in range(24): a = a + a
    linear = a.schedule_linear()
    prg = to_program(linear.src[-1].src[0], renderer=Device[Device.DEFAULT].renderer)
    self.assertLess(time.perf_counter()-st, 2.0)
    assert len(prg.src[2].arg.splitlines()) < 250

  def test_recursive_add_cmp(self):
    st = time.perf_counter()
    a = Tensor([1,2,3,4])
    for _ in range(24): a = a + a
    linear1 = a.schedule_linear()
    b = Tensor([1,2,3,4])
    for _ in range(24): b = b + b
    linear2 = b.schedule_linear()
    c = Tensor([1,2,3,4])
    for _ in range(23): c = c + c
    linear3 = c.schedule_linear()
    self.assertEqual(linear1.src[-1].src[0], linear2.src[-1].src[0])
    with self.assertRaises(AssertionError): self.assertEqual(linear1.src[-1].src[0], linear3.src[-1].src[0])
    self.assertLess(time.perf_counter()-st, 2.0)

  def test_recursive_pad(self):
    st = time.perf_counter()
    val = 1.0
    a = Tensor(val)
    for _ in range(24): a = Tensor.stack(a, a)[0]
    linear = a.schedule_linear()
    self.assertLessEqual(len(linear.src), 1)
    self.assertLess(time.perf_counter()-st, 2.0)

  def test_recursive_reshape(self):
    st = time.perf_counter()
    a = Tensor.empty(32, 32).realize()
    b = Tensor.empty(16, 2).realize()
    r = a.sum(1)
    for _ in range(24): r = r.reshape(16, 2) + b
    linear = r.schedule_linear()
    self.assertEqual(len(linear.src), 1)
    self.assertLess(time.perf_counter()-st, 2.0)

# NOTE: the NULL backend supports SLICE
class TestBufferView(unittest.TestCase):
  def test_shrink_contiguous_is_buffer_view(self):
    # simple 1D shrink of a realized buffer should be SLICE, not a copy kernel
    a = Tensor.arange(100).clone().realize()
    b = a.shrink(((10, 50),)).contiguous()
    run_linear(*check_schedule(b, 0))

  def test_shrink_2d_contiguous_is_buffer_view(self):
    a = Tensor.arange(100).reshape(10,10).clone().realize()
    b = a.shrink(((1, 5),None)).contiguous()
    run_linear(*check_schedule(b, 0))

  def test_chained_shrink_is_buffer_view(self):
    a = Tensor.arange(1000).clone().realize()
    b = a.shrink(((200, 800),)).shrink(((0, 300),)).reshape((30, 10)).shrink(((20, 25), (0, 10))).contiguous()
    run_linear(*check_schedule(b, 0))

  def test_shrink_non_shard_axis_is_buffer_view_multi(self):
    # indexing a non-shard axis of a realized sharded tensor should be SLICE on each device, not copy kernels
    # this is the flat_llama pattern: weight[layer_idx] where weight is (n_layers, out, dim) sharded on axis=1
    devices = ("NULL:1", "NULL:2")
    a = Tensor.arange(8*4*10).reshape(8, 4, 10).clone().shard(devices, axis=1).realize()
    run_linear(*check_schedule(a[3].contiguous(), 0))

  def test_shrink_2d_non_shard_axis_multi(self):
    devices = ("NULL:1", "NULL:2")
    a = Tensor.arange(6*4).reshape(6, 4).clone().shard(devices, axis=1).realize()
    run_linear(*check_schedule(a.shrink(((1, 4), None)).contiguous(), 0))

  def test_shrink_shard_axis_0_multi(self):
    # shrinking a middle dim is not contiguous per shard, so this needs copy kernels
    devices = ("NULL:1", "NULL:2")
    a = Tensor.arange(4*6*2).reshape(4, 6, 2).clone().shard(devices, axis=0).realize()
    run_linear(*check_schedule(a.shrink((None, (2, 5), None)).contiguous(), 2))

  def test_reshape_then_shrink_multi(self):
    devices = ("NULL:1", "NULL:2")
    a = Tensor.arange(8*6).reshape(8, 6).clone().shard(devices, axis=1).realize()
    run_linear(*check_schedule(a.reshape(4, 2, 6)[1].contiguous(), 0))

  def test_permute_then_shrink_multi(self):
    # permute makes per-shard view non-contiguous, needs copy kernels
    devices = ("NULL:1", "NULL:2")
    a = Tensor.arange(4*6*2).reshape(4, 6, 2).clone().shard(devices, axis=1).realize()
    run_linear(*check_schedule(a.permute(1, 0, 2).shrink(((0, 6), (1, 3), None)).contiguous(), 2))

  def test_multi_buffer_view_4_devices(self):
    devices = tuple(f"NULL:{i}" for i in range(4))
    a = Tensor.arange(8*12).reshape(8, 12).clone().shard(devices, axis=1).realize()
    run_linear(*check_schedule(a[5].contiguous(), 0))

  def test_chained_shrink_multi(self):
    devices = ("NULL:1", "NULL:2")
    a = Tensor.arange(10*8).reshape(10, 8).clone().shard(devices, axis=1).realize()
    run_linear(*check_schedule(a.shrink(((2, 8), None)).shrink(((1, 4), None)).contiguous(), 0))

  # negative tests: these should NOT become BUFFER_VIEW (non-contiguous per shard)
  def test_expand_multi_not_buffer_view(self):
    devices = ("NULL:1", "NULL:2")
    a = Tensor.arange(4*2).reshape(4, 1, 2).clone().shard(devices, axis=2).realize()
    run_linear(*check_schedule(a.expand(4, 3, 2).contiguous(), 2))

  def test_pad_multi_not_buffer_view(self):
    devices = ("NULL:1", "NULL:2")
    a = Tensor.arange(4*2).reshape(4, 2).clone().shard(devices, axis=1).realize()
    run_linear(*check_schedule(a.pad(((1, 1), (0, 0))).contiguous(), 2))

  def test_flip_multi_not_buffer_view(self):
    devices = ("NULL:1", "NULL:2")
    a = Tensor.arange(4*2).reshape(4, 2).clone().shard(devices, axis=1).realize()
    run_linear(*check_schedule(a.flip(0).contiguous(), 2))

  def test_replicated_reshape_is_buffer_view(self):
    devices = ("NULL:1", "NULL:2")
    a = Tensor.arange(24).clone().to(devices).realize()
    run_linear(*check_schedule(a.reshape(4, 6).contiguous(), 0))

  def test_replicated_shrink_is_buffer_view(self):
    # DP pattern: replicated weight[layer_idx]
    devices = ("NULL:1", "NULL:2")
    a = Tensor.arange(8*10).reshape(8, 10).clone().to(devices).realize()
    run_linear(*check_schedule(a[3].contiguous(), 0))

  def test_replicated_chained_mops_is_buffer_view(self):
    devices = ("NULL:1", "NULL:2")
    a = Tensor.arange(100).clone().to(devices).realize()
    run_linear(*check_schedule(a.reshape(10, 10).shrink(((2, 7), None)).contiguous(), 0))

  def test_replicated_shard_none_is_buffer_view(self):
    devices = ("NULL:1", "NULL:2")
    a = Tensor.arange(24).clone().shard(devices, axis=None).realize()
    run_linear(*check_schedule(a.reshape(4, 6).contiguous(), 0))

  def test_replicated_4_devices_is_buffer_view(self):
    devices = tuple(f"NULL:{i}" for i in range(4))
    a = Tensor.arange(8*10).reshape(8, 10).clone().to(devices).realize()
    run_linear(*check_schedule(a[3].contiguous(), 0))

  def test_replicated_expand_not_buffer_view(self):
    devices = ("NULL:1", "NULL:2")
    a = Tensor.arange(12).reshape(4, 1, 3).clone().to(devices).realize()
    run_linear(*check_schedule(a.expand(4, 3, 3).contiguous(), 2))

  def test_replicated_permute_not_buffer_view(self):
    devices = ("NULL:1", "NULL:2")
    a = Tensor.arange(24).reshape(4, 6).clone().to(devices).realize()
    run_linear(*check_schedule(a.permute(1, 0).contiguous(), 2))

  def test_replicated_flip_not_buffer_view(self):
    devices = ("NULL:1", "NULL:2")
    a = Tensor.arange(24).reshape(4, 6).clone().to(devices).realize()
    run_linear(*check_schedule(a.flip(0).contiguous(), 2))

class TestInvalidTensor(unittest.TestCase):
  def test_full_invalid_is_zero_kernels(self):
    from tinygrad.dtype import Invalid
    t = Tensor.full((4,), Invalid, dtype=dtypes.float)
    check_schedule(t, 0)

if __name__ == '__main__':
  unittest.main(verbosity=2)
