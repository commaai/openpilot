# this will be the new test_ops for the next level
# schedule confirms the right things are capable of fusing
# NOTE: this has overlap with external_test_opt.py

import unittest, time
import numpy as np

from tinygrad import nn, dtypes, Device, Tensor, Variable
from tinygrad.uop.ops import UOp, Ops, UPat
from tinygrad.helpers import DEBUG, DEV, GlobalCounters, Context, all_same, temp
from tinygrad.engine.realize import compile_linear, run_linear

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

class TestSchedule(unittest.TestCase):
  def setUp(self):
    self.ctx = Context(SPLIT_REDUCEOP=0)
    self.ctx.__enter__()
  def tearDown(self):
    self.ctx.__exit__(None, None, None)

  @unittest.skip("no longer supported")
  def test_double_from(self):
    x = Tensor([1,2,3,4])
    out = x.to('python')
    check_schedule(out, 0, filter_sink=False)

  def test_example_matmul_same(self):
    x = Tensor.eye(64).clone().realize()
    z = x.matmul(x).sum()
    z.backward()
    out = x.grad.contiguous()
    run_linear(*check_schedule(out, 1))
    # NOTE: the gradient flows twice
    np.testing.assert_allclose(out.numpy(), 2*np.ones((64,64)))

  def test_pad_reduce_scope_collision(self):
    b = Tensor.rand(4, 3).realize()
    s1 = b.pad(((1, 1), (0, 0))).sum(axis=1)
    s2 = b.pad(((1, 2), (0, 0))).shrink(((0, 6), (0, 3))).sum(axis=1)
    out = s1 + s2
    run_linear(*check_schedule(out, 1))
    np.testing.assert_allclose(out.numpy(), 2*np.pad(b.numpy(), ((1, 1), (0, 0))).sum(axis=1), rtol=1e-6)

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

  def test_fuse_assign_contiguous(self):
    x = Tensor.zeros(4, 4, dtype=dtypes.int).contiguous().realize()
    a = Tensor.arange(8).reshape(4, 2)
    run_linear(*check_schedule(x.shrink((None, (0, 2))).assign(a.clone()), 2))
    np.testing.assert_equal(x.numpy(), [[0, 1, 0, 0], [2, 3, 0, 0], [4, 5, 0, 0], [6, 7, 0, 0]])

  def test_assign_non_contiguous_alt(self): self.test_assign_non_contiguous(alt=True)
  def test_assign_non_contiguous(self, alt=False):
    x = (Tensor.arange(16)-100).reshape(4,4).clone().realize()
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
    a = Tensor.arange(16).reshape(4, 4).clone(device="CPU").realize()
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
    a = Tensor.arange(16).clone().realize()
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

  def test_const_folding_alt(self):
    t = Tensor.full((2,), 1.)
    lt = (t < 0.)
    a = Tensor.empty(2).assign(t*lt.where(-1., 0.))
    b = Tensor.empty(2, dtype=dtypes.bool).assign(lt)
    Tensor.realize(a, b)
    self.assertEqual(a.tolist(), [0., 0.])
    self.assertEqual(b.tolist(), [False, False])

  def test_self_assign_no_empty_kernel(self):
    for shape in [(3, 3), (4, 4)]:
      a = Tensor.ones(*shape).contiguous().realize()
      a.assign(a / 1)
      run_linear(*check_schedule(a, 0, filter_sink=False))
      self.assertListEqual(a.tolist(), [[1.]*shape[1]]*shape[0])

  def test_deviceless_materialize_localizes_to_target(self):
    dev = "CPU" if Device.DEFAULT != "CPU" else "CPU:1"
    t = Tensor.arange(Variable("s", 1, 128).bind(64)).cumsum().clone(dev)
    self.assertEqual(t.device, dev)
    np.testing.assert_equal(t[:64].numpy(), np.arange(64).cumsum())

  def test_copy_multi_scalar(self):
    devs = ("CPU:0", "CPU:1")
    x = Tensor.ones(2, device="CPU").shard(devs, axis=0).realize()
    out = (x.sum()*2).reshape(1).to("CPU")
    run_linear(*check_schedule(out, 5))
    np.testing.assert_equal(out.numpy(), [4.])

class TestLimitBufs(unittest.TestCase):
  @unittest.skipIf(DEV.interface.startswith("MOCK") and Device.DEFAULT == "NV", "crashes in ocelot")
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

  def test_limit_bufs_linear_scaling(self):
    def sched_time(n):
      with Context(TRACK_MATCH_STATS=0, DEBUG=0):
        bufs = [Tensor.ones(16).contiguous().realize() for _ in range(4)]
        root = bufs[0]
        for i in range(n): root = root + bufs[i % 4]
        with Context(MAX_KERNEL_BUFFERS=8, SCACHE=0):
          st = time.perf_counter()
          root.schedule_linear()
          return time.perf_counter() - st
    sched_time(400)
    t1, t2 = min(sched_time(400) for _ in range(3)), min(sched_time(1600) for _ in range(3))
    self.assertLess(t2/t1, 8, f"{t1*1e3:.1f}ms -> {t2*1e3:.1f}ms")

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

  def test_late_const_copy_folding(self):
    a = Tensor.arange(3).clone().realize()
    zeros = Tensor.zeros(3, buffer=False).realize()
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
    with open(temp('dt_arange_4_permute'), "wb") as f: f.write(Tensor.arange(4).clone().realize().uop.base.buffer.as_memoryview())
    a = Tensor.empty(4, dtype=dtypes.int32, device=f"disk:{temp('dt_arange_4_permute')}")
    b = a.reshape(2, 2).permute(1, 0).to("CPU")
    b.realize()
    self.assertListEqual(b.tolist(), [[0, 2], [1, 3]])

  def test_permute_on_disk_contiguous(self):
    with open(temp('dt_arange_4_permute_contig'), "wb") as f: f.write(Tensor.arange(4).clone().realize().uop.base.buffer.as_memoryview())
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
    with open(temp('dt_arange_5_permute'), "wb") as f: f.write(Tensor.arange(5).clone().realize().uop.base.buffer.as_memoryview())
    a = Tensor.empty(5, dtype=dtypes.int32, device=f"disk:{temp('dt_arange_5_permute')}")
    b = a.shrink(((0, 4),)).reshape(2, 2).permute(1, 0).to("CPU")
    b.realize()
    self.assertListEqual(b.tolist(), [[0, 2], [1, 3]])

  def test_permute_copy_to_device(self):
    b = Tensor([[0, 1, 2, 3], [4, 5, 6, 7]], device="CPU").permute(1, 0).to("PYTHON")
    self.assertListEqual(b.tolist(), [[0, 4], [1, 5], [2, 6], [3, 7]])

  def test_flip_copy_to_device(self):
    b = Tensor([0, 1, 2, 3], device="CPU").flip(0).to("PYTHON")
    self.assertListEqual(b.tolist(), [3, 2, 1, 0])

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
