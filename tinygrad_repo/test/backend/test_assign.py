#!/usr/bin/env python
import unittest
import numpy as np
from tinygrad import Device, dtypes, Tensor, TinyJit, GlobalCounters, Variable
from tinygrad.uop.ops import Ops, UOp
from tinygrad.helpers import temp, DEV, Context
from test.helpers import assert_kernel_count, needs_second_gpu

N = 200  # has to be bigger than the cache to fail

class TestAssign(unittest.TestCase):
  def test_simple_assignment(self):
    a = Tensor(np.arange(N*N, dtype=np.float32)).reshape(N,N)
    b = Tensor(np.arange(N*N, dtype=np.float32)).reshape(N,N)
    a.realize()
    b.realize()
    ba1 = a.uop.base.realized
    bb1 = b.uop.base.realized
    a += b
    a.realize()
    ba2 = a.uop.base.realized
    assert ba1 == ba2 and ba1 != bb1
    np.testing.assert_allclose(a.numpy(), (np.arange(N*N)*2).reshape((N,N)))

  def test_assign_keeps_identical_tensor(self):
    a = Tensor.zeros(10,10).contiguous()
    a.assign(Tensor.ones(10,10))
    b = Tensor.zeros(10,10).contiguous()
    a.realize()
    np.testing.assert_allclose(b.numpy(), 0)

  @unittest.skip("TODO: this often crashes in CI")
  def test_assign_keeps_earlier_identical_tensor(self):
    a = Tensor.zeros(10,10).contiguous()
    b = Tensor.zeros(10,10).contiguous()
    a.assign(Tensor.ones(10,10))
    a.realize()
    np.testing.assert_allclose(b.numpy(), 0)

  def test_assign_copy(self):
    a = Tensor([1.,2,3], device="PYTHON")
    c = Tensor.empty(3).assign(a.to(None))
    # it should copy into the empty buffer
    GlobalCounters.reset()
    c.realize()
    assert_kernel_count(1)

  def test_assign_slice(self):
    X = Tensor([1,2,3,4]).realize()
    xs = X[2:4]
    xs.assign(xs+1)
    GlobalCounters.reset()
    self.assertListEqual(X.tolist(), [1,2,4,5])
    assert_kernel_count(1)

  def test_assign_slice_alt(self):
    X = Tensor([1,2,3,4]).realize()
    xs1, xs2 = X[1:3], X[2:4]
    xs1.assign(xs2+1)
    GlobalCounters.reset()
    self.assertListEqual(X.tolist(), [1,4,5,4])
    assert_kernel_count(2)

  def test_assign_flip(self):
    ref = np.arange(16, dtype=np.float32)
    X = Tensor(ref, device="CPU").contiguous().realize()
    GlobalCounters.reset()
    xs = X[::-1]
    xs.assign(xs + X)
    ref = ref + ref[::-1]
    np.testing.assert_allclose(X.numpy(), ref)
    assert_kernel_count(2)

  def test_assign_add(self):
    for T in (1, 2, 10):#, 100): # this crashes in CI, not sure why
      x = Tensor([0]).realize()
      buf = x.uop.base.realized
      for _ in range(T):
        x += 1
      x.realize()
      assert x.item() == T
      assert x.uop.base.realized is buf

  def test_assign_slice_add(self):
    for T in (1, 2, 10, 100):
      x = Tensor([0, 0]).realize()
      buf = x.uop.base.realized
      for _ in range(T):
        x[0] += 1
      x.realize()
      assert x.tolist() == [T, 0]
      assert x.uop.base.realized is buf

  def test_assign_add_double(self):
    def f(x):
      x += 1
      x.realize()
    x = Tensor([0])
    f(x)
    out = x.item()
    assert out == 1, f"expected 1, got {out}"
    x = Tensor([0])
    f(x)
    out = x.item()
    assert out == 1, f"expected 1, got {out}"

  def test_pending_assign_chain_preserves_intermediate_reads(self):
    x = Tensor([0.0]).contiguous().realize()
    y0 = x + 0
    x.assign(x + 1)
    y1 = x + 0
    x.assign(x + 1)
    y2 = x + 0
    x.assign(x + 1)
    assert [y0.item(), y1.item(), y2.item(), x.item()] == [0.0, 1.0, 2.0, 3.0]

  def test_assign_add_jit_other(self):
    @TinyJit
    def f(x):
      x += 1
      x.realize()
    x = Tensor([0])
    for _ in range(5): f(x)
    assert x.item() == 5

    y = Tensor([0])
    for _ in range(4): f(y)
    assert y.item() == 4

  def test_assign_other_jit(self):
    @TinyJit
    def f(x, a):
      x.assign(a)
      x.realize()
    x = Tensor([0])
    for i in range(1, 6):
      f(x, x.full_like(i).contiguous())  # const would be implicitly folded without contiguous
      assert x.item() == i

  def test_assign_add_other_jit(self):
    @TinyJit
    def f(x, a):
      x += a
      x.realize()
    x = Tensor([0])
    a = 0
    for i in range(1, 6):
      a += i
      f(x, x.full_like(i).contiguous())
      assert x.item() == a

  def test_assign_changes(self):
    a = Tensor.ones(4).contiguous().realize()
    old_a = a
    a.assign(Tensor.full((4,), 2.).contiguous())
    # NOTE: old_a is now 2, and this would match the behavior of pytorch
    new = a + old_a
    np.testing.assert_allclose(new.numpy(), 4)

  def test_assign_changes_alt(self, realize=False):
    a = Tensor(1).contiguous()
    if realize: a.realize()
    b = a.clone()
    b.assign(2)
    b.realize()
    self.assertNotEqual(a.item(), b.item())
  def test_assign_changes_realized_alt(self): return self.test_assign_changes_alt(realize=True)

  def test_assign_changes_buffer_alt(self):
    a, b = [Tensor(Tensor([0]).realize().uop.buf_uop) for _ in range(2)]
    Tensor.realize(a.contiguous().assign(1), b.contiguous().assign(2))
    self.assertEqual((a + b).item(), 3)

  def test_assign_diamond(self):
    a = Tensor.ones(4).contiguous().realize()
    times_a = a*3
    a.assign(Tensor.full((4,), 2.).contiguous())
    new = a + (times_a-1)
    with self.assertRaisesRegex(RuntimeError, "cycle"):  # TODO: broken now, raises
      np.testing.assert_allclose(new.numpy(), 4)

  def test_assign_diamond_contiguous(self):
    a = Tensor.ones(4).contiguous().realize()
    times_a = a*3
    a.assign(Tensor.full((4,), 2.))
    new = a.contiguous() + times_a-1
    with self.assertRaisesRegex(RuntimeError, "cycle"):  # TODO: broken now, raises
      np.testing.assert_allclose(new.numpy(), 4)

  def test_assign_diamond_possible(self):
    a = Tensor.ones(4).contiguous().realize()
    times_a = a*3
    a.assign(Tensor.full((4,), 2.))
    new = a + (times_a-1).contiguous()
    np.testing.assert_allclose(new.numpy(), 4)

  def test_assign_diamond_possible_contiguous(self):
    a = Tensor.ones(4).contiguous().realize()
    times_a = a*3
    a.assign(Tensor.full((4,), 2.).contiguous())
    new = a + (times_a-1).contiguous()
    np.testing.assert_allclose(new.numpy(), 4)

  def test_assign_diamond_both_contiguous(self):
    a = Tensor.ones(4).contiguous().realize()
    times_a = a*3
    a.assign(Tensor.full((4,), 2.))
    new = a.contiguous() + (times_a-1).contiguous()
    np.testing.assert_allclose(new.numpy(), 4)

  def test_assign_diamond_alt(self):
    a = Tensor.ones(4).contiguous().realize()
    a.assign(Tensor.full((4,), 2.).contiguous())
    times_a = a*3
    new = a + times_a
    np.testing.assert_allclose(new.numpy(), 8)

  @unittest.skipIf(DEV.renderer == "LVP", "flaky in CI")
  def test_double_assign(self):
    a = Tensor.ones(4).contiguous().realize()
    a += 1
    a += 1
    np.testing.assert_allclose(a.numpy(), 3)

  def test_crossover_assign(self):
    a = Tensor.full((4,), 2).contiguous().realize()
    b = Tensor.full((4,), 3).contiguous().realize()
    a += b
    b += a
    Tensor.realize(a,b)
    np.testing.assert_allclose(a.numpy(), 5)
    np.testing.assert_allclose(b.numpy(), 8)

  def test_assign_double_diamond(self):
    a = Tensor.full((4,), 2).contiguous().realize()
    b = Tensor.full((4,), 3).contiguous().realize()
    a_prev = a*4
    b_prev = b+3
    b += a_prev.contiguous()
    a += b_prev.contiguous()
    Tensor.realize(a, b)
    np.testing.assert_equal(b.numpy(), 11)
    np.testing.assert_equal(a.numpy(), 8)

  def test_assign_double_diamond_reduce(self):
    a0 = Tensor.full((16, 16), 10).contiguous().realize()
    a1 = Tensor.full((16, 16), 20).contiguous().realize()
    b0 = Tensor.full((16, ), 1).contiguous().realize()
    b1 = Tensor.full((16, ), 2).contiguous().realize()

    r0 = (a0 - b1.contiguous()).sum(1)
    r1 = (a1 - b0.contiguous()).sum(1)
    b0.assign(r0 * b0)
    b1.assign(r1 * b1)
    Tensor.realize(b0, b1)
    np.testing.assert_equal(b0.numpy(), 128)
    np.testing.assert_equal(b1.numpy(), 608)

  def test_crossunder_assign(self):
    a = Tensor.full((4,), 2).contiguous().realize()
    b = Tensor.full((4,), 3).contiguous().realize()
    c = a+9
    a += b
    b += c
    with self.assertRaisesRegex(RuntimeError, "cycle"):  # TODO: broken now, raises
      Tensor.realize(a,b)
      np.testing.assert_allclose(a.numpy(), 2+3)
      np.testing.assert_allclose(b.numpy(), 3+2+9)

  def test_assign_kv_cache(self):
    bsz, max_context = 2, 8

    class Attn:
      @TinyJit
      def __call__(self, xk:Tensor, start_pos:Variable):
        seqlen = xk.shape[1]
        if not hasattr(self, "cache_k"):
          self.cache_k = Tensor.zeros(bsz, max_context, 1, 1).contiguous()
        keys = self.cache_k.shrink((None, (0, start_pos), None, None)).cat(xk, dim=1).contiguous() if start_pos > 0 else xk
        self.cache_k.assign(keys.pad((None,(0,max_context-start_pos-seqlen),None,None)).contiguous()).realize()

    attn = Attn()
    xk = Tensor.ones(bsz, 3, 1, 1).contiguous()
    attn(xk, 0)
    for i in range(3,6):
      # copied from LLaMA
      start_pos = Variable("start_pos", 1, max_context).bind(i)
      xk = Tensor.ones(bsz, 1, 1, 1).contiguous()
      attn(xk, start_pos)

    out = attn.cache_k.flatten().numpy()
    np.testing.assert_allclose(out, [1.,1.,1.,1.,1.,1.,0.,0.,1.,1.,1.,1.,1.,1.,0.,0.])

  def test_assign_after(self):
    t = Tensor.zeros(10).contiguous().realize()
    t.uop = t.uop.after(t.uop.store((t+1).uop))
    np.testing.assert_allclose(t.numpy(), [1.,1.,1.,1.,1.,1.,1.,1.,1.,1.])

  def test_assign_after_partial(self):
    t = Tensor.zeros(10).contiguous().realize()
    t.uop = t.uop.after(t[:5].uop.after(t[:5].uop.store(Tensor.ones(5).uop)))
    np.testing.assert_allclose(t.numpy(), [1.,1.,1.,1.,1.,0.,0.,0.,0.,0.])

  def test_assign_after_target_chain(self):
    t = Tensor.arange(16).reshape(4, 4).permute(1, 0).contiguous()
    t.assign(t + 100)
    np.testing.assert_equal(t.numpy(), [[100, 104, 108, 112], [101, 105, 109, 113], [102, 106, 110, 114], [103, 107, 111, 115]])

  def test_assign_corealize_order_independent(self):
    for order in [lambda x,y: Tensor.realize(x, y), lambda x,y: Tensor.realize(y, x)]:
      x = Tensor([1.0]).realize()
      y = x + 10
      x.assign(x*2)
      x.assign(x+3)
      order(x, y)
      self.assertEqual(y.tolist(), [11.0])
      self.assertEqual(x.tolist(), [5.0])

  def test_assign_contiguous(self):
    b = Tensor.arange(16).reshape(4,4).clone().realize()
    a = (Tensor.arange(16).reshape(4,4).clone().realize() + 1)
    GlobalCounters.reset()
    b.assign(a.contiguous()).realize()
    assert_kernel_count(2)

  def test_assign_contiguous_permute(self):
    b = Tensor.arange(16).reshape(4,4).clone().realize()
    a = (Tensor.arange(16).reshape(4,4).clone().realize() + 1).permute((1,0))
    GlobalCounters.reset()
    b.assign(a.contiguous()).realize()
    assert_kernel_count(2)

  def test_permuted_assignment(self):
    a = Tensor(np.arange(N*N, dtype=np.float32)).reshape(N,N)
    b = Tensor(np.arange(N*N, dtype=np.float32)).reshape(N,N)
    a.realize()
    b.realize()
    ba1 = a.uop.base.realized
    bb1 = b.uop.base.realized
    a = a.permute(1,0)
    a += b
    a.realize()
    ba2 = a.uop.base.realized
    np.testing.assert_allclose(a.numpy(), np.arange(N*N).reshape((N,N)) + np.arange(N*N).reshape((N,N)).transpose(1,0))
    # permute and base are the same buffer
    assert ba1 == ba2 and ba1 != bb1

  def _assign_view_of_self(self, view):
    a = Tensor.arange(N*N).reshape(N,N).clone().realize()
    b = Tensor.arange(N*N).reshape(N,N).clone().realize()
    new_a = (view(a)+b).numpy()
    a.assign(view(a)+b)
    np.testing.assert_allclose(a.numpy(), new_a)

  def test_post_permuted_assignment(self): self._assign_view_of_self(lambda a: a.T)
  def test_post_flipped_assignment(self): self._assign_view_of_self(lambda a: a.flip(0))
  def test_post_flipped_assignment_axis1(self): self._assign_view_of_self(lambda a: a.flip(1))
  def test_post_reshape_assignment(self): self._assign_view_of_self(lambda a: a.reshape(-1).reshape(N,N))

  @unittest.skip("multi output not supported anymore")
  def test_simple_assignment_multioutput(self):
    a = Tensor.arange(32*32).reshape(32, 32).clone().realize()
    b = Tensor.full((32, ), 1.).contiguous().realize()
    c = Tensor.full((32, ), 2.).contiguous().realize()
    d = Tensor.full((32, ), 3.).contiguous().realize()

    r = a.sum(axis=1)
    b.assign(r + b)
    c.assign(r + c)
    d.assign(r + d)

    GlobalCounters.reset()
    Tensor.realize(b, c, d)
    assert_kernel_count(1)
    np.testing.assert_allclose(b.numpy(), a.sum(1).numpy()+1)
    np.testing.assert_allclose(c.numpy(), a.sum(1).numpy()+2)
    np.testing.assert_allclose(d.numpy(), a.sum(1).numpy()+3)

  # NOTE: if the assign target is read/write in a single kernel, it should be contiguous

  def test_permuted_reduceop_child_dual_use(self):
    a = Tensor.arange(32*32*32).reshape(32, 32, 32).clone().realize()
    b = Tensor.ones(32, 32, dtype=dtypes.int).contiguous().realize()
    r = a.sum(axis=1)
    b.assign(r + b.permute(1, 0))
    b.realize()
    np.testing.assert_equal(b.numpy(), a.numpy().sum(axis=1)+np.ones((32, 32), dtype=np.int32).transpose(1, 0))

  @unittest.skip("multi output not supported anymore")
  def test_permuted_reduceop_multioutput_dual_use(self):
    a = Tensor.arange(32*32*32).reshape(32, 32, 32).clone().realize()
    b = Tensor.full((32, 32), 1.).contiguous().realize()
    c = Tensor.full((32, 32), 2.).contiguous().realize()

    with self.assertRaisesRegex(RuntimeError, "contiguous"):
      r = a.sum(axis=1)
      b_perm = b.permute(1, 0)
      b.assign(r + b)
      c.assign(r + b_perm)
      Tensor.realize(b, c)

  @unittest.skip("multi output not supported anymore")
  def test_permuted_reduceop_multioutput_dual_use_possible(self):
    a = Tensor.arange(32*32*32).reshape(32, 32, 32).clone().realize()
    b = Tensor.arange(32 * 32).reshape(32, 32).clone().realize()
    c = Tensor.arange(32 * 32).reshape(32, 32).clone().realize()

    GlobalCounters.reset()
    r = a.sum(axis=1)
    b_perm = b.permute(1, 0)
    b.assign(r + b)
    c.assign(r + b_perm.contiguous())
    Tensor.realize(b, c)
    assert_kernel_count(2)
    np.testing.assert_equal(b.numpy(), a.numpy().sum(1) + np.arange(32 * 32).reshape(32, 32))
    np.testing.assert_equal(c.numpy(), a.numpy().sum(1) + np.arange(32 * 32).reshape(32, 32).transpose(1, 0))

  def test_permuted_assignment_masked_view_possible(self):
    a = Tensor.ones(4, 4).contiguous().realize()
    b = a.shrink((None, (0, 2))).pad((None, (0, 2)), value=2)
    a.assign(a + b)
    GlobalCounters.reset()
    a.realize()
    assert_kernel_count(1)
    np.testing.assert_equal(a.numpy(), np.ones((4, 4))+np.pad(np.ones((4, 4))[:, 0:2], ((0, 0), (0, 2)), constant_values=2))

  def test_permuted_assignment_masked_view_not_contiguous(self):
    a = Tensor.ones(4, 4).contiguous().realize()
    b = a.shrink((None, (0, 2))).pad((None, (0, 2)), value=2).permute(1, 0)
    a.assign(a + b)
    a.realize()
    self.assertListEqual(a.tolist(), [[2.,2.,2.,2.],[2.,2.,2.,2.],[3.,3.,3.,3.], [3.,3.,3.,3.]])

  # TODO: is there a way to sneak in a permute such that it returns the wrong answer?

  def test_overlapping_shrink_assignment_forward(self):
    # Forward shift: read index > write index in overlap
    N = 100000
    shift = 1000
    a = Tensor.arange(N).float().clone().realize()
    expected = np.arange(N, dtype=np.float32)
    expected[:N-shift] = expected[shift:].copy()
    with Context(NOOPT=1): a[0:N-shift].assign(a[shift:N]).realize()
    np.testing.assert_allclose(a.numpy(), expected)

  def test_overlapping_shrink_assignment_reverse(self):
    # Reverse shift: write index > read index in overlap
    N = 100000
    shift = 1000
    a = Tensor.arange(N).float().clone().realize()
    expected = np.arange(N, dtype=np.float32)
    expected[shift:] = expected[:N-shift].copy()
    with Context(NOOPT=1): a[shift:N].assign(a[0:N-shift]).realize()
    np.testing.assert_allclose(a.numpy(), expected)

  def test_nonoverlapping_shrink_assignment(self):
    # TODO: non-overlapping shrinks don't actually need contiguous, could be 1 kernel with smarter range analysis
    a = Tensor.arange(100).float().clone().realize()
    expected = np.arange(100, dtype=np.float32)
    expected[0:10] = expected[50:60].copy()
    GlobalCounters.reset()
    a[0:10].assign(a[50:60]).realize()
    assert_kernel_count(2)  # currently conservative, forces contiguous
    np.testing.assert_allclose(a.numpy(), expected)

  def test_setitem_half(self):
    a = Tensor.full((8,), 1.0, dtype=dtypes.half).contiguous().realize()
    b = Tensor.full((4,), 2.0, dtype=dtypes.half).contiguous().realize()
    assign = a[:4].assign(b)
    assign.realize()
    np.testing.assert_allclose(a.numpy(), [2., 2., 2., 2., 1., 1., 1., 1.])

  def test_setitem_list(self):
    a = Tensor.zeros(8).contiguous().realize()
    a[2:5] = [1, 2, 3]
    np.testing.assert_allclose(a.numpy(), [0., 0., 1., 2., 3., 0., 0., 0.])

  # IEEE 754: 1.0f = 0x3f800000, 2.0f = 0x40000000, 3.0f = 0x40400000, 4.0f = 0x40800000
  REVERSED = [0x40800000, 0x40400000, 0x40000000, 0x3f800000]

  def test_assign_bitcast(self):
    a = Tensor([1.0, 2.0, 3.0, 4.0], dtype=dtypes.float32).realize()
    a.bitcast(dtypes.uint32).assign(Tensor(self.REVERSED, dtype=dtypes.uint32)).realize()
    np.testing.assert_allclose(a.numpy(), [4.0, 3.0, 2.0, 1.0])

  def test_assign_bitcast_unrealized(self):
    a = Tensor([1.0, 2.0, 3.0, 4.0], dtype=dtypes.float32).realize()
    a.bitcast(dtypes.uint32).assign(Tensor(self.REVERSED, dtype=dtypes.uint32))
    np.testing.assert_allclose(a.numpy(), [4.0, 3.0, 2.0, 1.0])

  def test_assign_double_bitcast(self):
    b = Tensor([1.0, 2.0, 3.0, 4.0], dtype=dtypes.float32).realize()
    b.bitcast(dtypes.uint32).bitcast(dtypes.int32).assign(Tensor(self.REVERSED, dtype=dtypes.int32)).realize()
    np.testing.assert_allclose(b.numpy(), [4.0, 3.0, 2.0, 1.0])

  def test_assign_shrink_then_bitcast(self):
    c = Tensor([1.0, 2.0, 3.0, 4.0], dtype=dtypes.float32).realize()
    c[0:2].bitcast(dtypes.uint32).assign(Tensor(self.REVERSED[:2], dtype=dtypes.uint32)).realize()
    np.testing.assert_allclose(c.numpy(), [4.0, 3.0, 3.0, 4.0])

  def test_assign_bitcast_different_size(self):
    # assign to a shape-changing bitcast view
    a = Tensor([0]*8, dtype=dtypes.uint8).realize()
    a.bitcast(dtypes.int64).assign(Tensor([12345], dtype=dtypes.int64)).realize()
    np.testing.assert_equal(a.numpy(), [57, 48, 0, 0, 0, 0, 0, 0])

  def test_assign_dtype_mismatch(self):
    # assign should not implicitly cast dtypes - this can lose precision
    a = Tensor.zeros(4, dtype=dtypes.float32).contiguous().realize()
    b = Tensor([1, 2, 3, 4], dtype=dtypes.int32)
    with self.assertRaisesRegex(RuntimeError, "assign dtype mismatch"):
      a.assign(b)

  def test_assign_shape_broadcast(self):
    # shape broadcasting should work when dtypes match
    a = Tensor.zeros(3, 5, dtype=dtypes.float32).contiguous().realize()
    b = Tensor([1., 2., 3., 4., 5.], dtype=dtypes.float32)
    a.assign(b)
    a.realize()
    expected = np.array([[1., 2., 3., 4., 5.]] * 3)
    np.testing.assert_allclose(a.numpy(), expected)

  def test_assign_shape_broadcast_2d(self):
    # broadcast (1, 5) to (3, 5)
    a = Tensor.zeros(3, 5, dtype=dtypes.float32).contiguous().realize()
    b = Tensor([[1., 2., 3., 4., 5.]], dtype=dtypes.float32)
    a.assign(b)
    a.realize()
    expected = np.array([[1., 2., 3., 4., 5.]] * 3)
    np.testing.assert_allclose(a.numpy(), expected)

  def test_disk_assignment(self):
    a = Tensor.empty(5, device=f"disk:{temp('disk_assignment')}").assign(Tensor.ones(5)).numpy()
    np.testing.assert_equal(a, np.ones(5))

  def test_assign_slice_then_read(self):
    """Assign to slice then read from buffer - read should see the assigned values.
    This is the KV cache pattern from llm.py.
    """
    v_pos = Variable("pos", 0, 3).bind(0)
    cache = Tensor.zeros(4, 4).contiguous().realize()
    cache[v_pos:v_pos+1, :].assign(Tensor.ones(1, 4))
    self.assertEqual(cache.sum().item(), 4.0)

  def test_chained_assign_slice_then_read(self):
    """Three caches with chained assign-then-read: each block writes to its cache and reads back,
    feeding the result to the next block's assign. Without proper dependency tracking, block N's read
    may see stale data from block N-1's cache (pre-assign zeros instead of the assigned values).
    This is the multi-layer KV cache pattern from llm.py._attention.
    """
    D, max_ctx = 4, 8
    cache1 = Tensor.zeros(max_ctx, D).contiguous().realize()
    cache2 = Tensor.zeros(max_ctx, D).contiguous().realize()
    cache3 = Tensor.zeros(max_ctx, D).contiguous().realize()
    cache1[:3].assign(Tensor.ones(3, D)).realize()
    cache2[:3].assign(Tensor.ones(3, D) * 2).realize()
    cache3[:3].assign(Tensor.ones(3, D) * 3).realize()
    # block 1: assign [10]*D at position 3, read sum -> c1=[13]*D
    cache1[3:4].assign(Tensor.ones(1, D) * 10)
    c1 = cache1[:4].sum(0, keepdim=True)
    # block 2: assign c1 at position 3, read sum -> c2=[19]*D
    cache2[3:4].assign(c1)
    c2 = cache2[:4].sum(0, keepdim=True)
    # block 3: assign c2 at position 3, read sum -> 112
    cache3[3:4].assign(c2)
    self.assertEqual(cache3[:4].sum().item(), 112.0)

  def test_chained_assign_kernel_count(self):
    """Chained pending assigns must not produce excessive kernels (tests recursive transitive processing)."""
    D, N = 4, 5
    caches = [Tensor.zeros(8, D).contiguous().realize() for _ in range(N)]
    caches[0][0:1].assign(Tensor.ones(1, D, buffer=False) * 10)
    x = caches[0][:1].sum(0, keepdim=True)
    for i in range(1, N):
      caches[i][0:1].assign(x)
      x = caches[i][:1].sum(0, keepdim=True)
    GlobalCounters.reset()
    x.realize()
    # N assigns (1 kernel each) producing N kernels total
    assert_kernel_count(N)

  def test_shared_computation_assign_kernel_count(self):
    """When a .contiguous() is shared between an assign value and the next layer's input (like QKV projection in LLM),
    substitute optimization replaces already-realized sub-graphs in remaining pending assigns, preventing kernel escalation.
    Without substitute, pending assign graphs grow linearly and produce 153 kernels instead of 48."""
    D, N = 16, 16
    caches = [Tensor.zeros(4, D).contiguous().realize() for _ in range(N)]
    W = [Tensor.full((D, D*2), 0.01).contiguous().realize() for _ in range(N)]
    x = Tensor.ones(1, D).contiguous().realize()
    for i in range(N):
      shared = (x @ W[i]).contiguous()  # .contiguous() UOp is shared between assign (k) and next layer (q)
      k, q = shared[:, :D], shared[:, D:]
      caches[i][0:1].assign(k)          # assign references the CONTIGUOUS
      x = q + caches[i][:1]             # next layer also references the same CONTIGUOUS through q
    GlobalCounters.reset()
    caches[-1][:1].contiguous().realize()
    # N matmuls + N assigns + 1 final read = 2*N+1 (AFTER embedding allows full graph scheduling with shared contiguous reuse)
    assert_kernel_count(2*N+1)

  def test_double_assign_from_const(self):
    a = Tensor.empty(2)
    a.assign(Tensor.ones(2, buffer=False))
    a.assign(Tensor.ones(2, buffer=False))
    GlobalCounters.reset()
    a.realize()
    assert_kernel_count(1)
    self.assertEqual(a.tolist(), [1.,1.])

  def test_assign_deviceless_const(self):
    s = Tensor.empty(4, device="CPU:1", dtype=dtypes.float)
    s.assign(Tensor(UOp.const(2.0).cast(dtypes.float)))
    np.testing.assert_equal(s.numpy(), [2, 2, 2, 2])

  def test_nested_after_contiguous_store(self):
    # Mirrors the nested contiguous-write-then-assign-back shape from torch backend view updates.
    base = Tensor.empty(3, dtype=dtypes.int64)
    base.assign(Tensor([1, 2, 3], dtype=dtypes.int64))
    contig = base.contiguous()
    contig.assign(Tensor([1, 4, 3], dtype=dtypes.int64))
    GlobalCounters.reset()
    base.assign(contig).realize()
    assert_kernel_count(3)  # TODO: first copy is dead, could be 2
    self.assertEqual(base.tolist(), [1,4,3])

  def test_nested_after_contiguous_store_no_init(self):
    # Same shape as test_nested_after_contiguous_store, but without the initial assign.
    base = Tensor.empty(3, dtype=dtypes.int64)
    contig = base.contiguous()
    contig.assign(Tensor([1, 4, 3], dtype=dtypes.int64))
    GlobalCounters.reset()
    base.assign(contig).realize()
    assert_kernel_count(1)
    self.assertEqual(base.tolist(), [1,4,3])

  def test_assign_temporary_copy_reshape(self):
    a = Tensor([[1., 2], [3, 4]], device="PYTHON")
    c = Tensor.empty(2, 2).assign(a.to(None))
    GlobalCounters.reset()
    c.realize()
    assert_kernel_count(1)
    self.assertEqual(c.tolist(), [[1., 2], [3, 4]])

class TestAssignOrdering(unittest.TestCase):
  """Tests for complex assign orderings that could differ between lazy and eager execution.

  The key principle: tinygrad's lazy execution with RAW/WAR dependency tracking should
  produce the same results as eager (immediate) execution for valid programs.

  These tests exercise edge cases where incorrect dependency tracking could cause:
  - Stale reads (reading before write completes)
  - Lost writes (write ordering reversed)
  - Race conditions (concurrent access to same buffer)
  """

  def test_overlapping_slice_assigns(self):
    """Overlapping slice assigns - later write should win for overlapping elements."""
    buf = Tensor.zeros(8).contiguous().realize()
    buf[0:4].assign(Tensor.ones(4))
    buf[2:6].assign(Tensor.ones(4) * 2)
    np.testing.assert_equal(buf.numpy(), [1,1,2,2,2,2,0,0])

  def test_overlapping_slice_assigns_reverse(self):
    """Overlapping slice assigns in reverse order."""
    buf = Tensor.zeros(8).contiguous().realize()
    buf[2:6].assign(Tensor.ones(4) * 2)
    buf[0:4].assign(Tensor.ones(4))
    np.testing.assert_equal(buf.numpy(), [1,1,1,1,2,2,0,0])

  def test_read_between_writes(self):
    """Read should see first write before second write happens."""
    buf = Tensor.zeros(4).contiguous().realize()
    buf.assign(Tensor.ones(4))
    r1 = buf.sum().realize()  # should see ones = 4
    buf.assign(Tensor.ones(4) * 2)
    r2 = buf.sum().realize()  # should see twos = 8
    self.assertEqual(r1.item(), 4)
    self.assertEqual(r2.item(), 8)

  def test_write_read_write_chain(self):
    """Write, read, write chain - middle read must complete before second write."""
    buf = Tensor.zeros(4).contiguous().realize()
    buf.assign(Tensor.ones(4) * 3)
    mid_sum = buf.sum()  # lazy read, should be 12
    buf.assign(Tensor.ones(4) * 5)
    final_sum = buf.sum()  # lazy read, should be 20
    # Realize in "wrong" order - final first
    self.assertEqual(final_sum.realize().item(), 20)
    try:
      self.assertEqual(mid_sum.realize().item(), 12)
    except AssertionError:
      # TODO: this is wrong
      self.assertEqual(mid_sum.realize().item(), 20)

  def test_slice_read_then_full_write(self):
    """Read from slice, then overwrite full buffer - WAR dependency works for full buffer assigns."""
    buf = Tensor([1.,2.,3.,4.]).contiguous().realize()
    partial = buf[0:2].sum()  # lazy read
    buf.assign(Tensor.ones(4) * 10)  # overwrite everything
    full = buf.sum()
    # WAR dependency correctly tracked - partial sees original data
    self.assertEqual(partial.realize().item(), 3)  # 1+2
    self.assertEqual(full.realize().item(), 40)

  def test_slice_write_then_full_read(self):
    """Write to slice, then read full buffer."""
    buf = Tensor.zeros(4, dtype=dtypes.int32).contiguous().realize()
    buf[1:3].assign(Tensor([5, 6]))
    np.testing.assert_equal(buf.numpy(), [0, 5, 6, 0])

  def test_chained_slice_copies(self):
    """Copy from one slice to another within same buffer."""
    buf = Tensor([1, 2, 3, 4, 5, 6, 7, 8]).contiguous().realize()
    buf[4:8].assign(buf[0:4].contiguous())
    np.testing.assert_equal(buf.numpy(), [1, 2, 3, 4, 1, 2, 3, 4])

  def test_swap_slices(self):
    """Swap two non-overlapping slices - requires reading both before writing."""
    # without .realize() on temps: values not captured before overwriting
    buf = Tensor([1, 2, 3, 4, 5, 6, 7, 8]).contiguous().realize()
    left = buf[0:4].clone()  # lazy - not captured yet
    right = buf[4:8].clone()  # lazy - not captured yet
    buf[0:4].assign(right).realize()  # this works
    buf[4:8].assign(left).realize()  # left now reads from modified buf!
    try:
      np.testing.assert_equal(buf.numpy(), [5, 6, 7, 8, 1, 2, 3, 4])
    except AssertionError:
      # TODO: broken now
      np.testing.assert_equal(buf.numpy(), [5, 6, 7, 8, 5, 6, 7, 8])

    # with .realize() on temps: values captured before writes
    buf = Tensor([1, 2, 3, 4, 5, 6, 7, 8]).contiguous().realize()
    left = buf[0:4].clone().realize()
    right = buf[4:8].clone().realize()
    buf[0:4].assign(right).realize()
    buf[4:8].assign(left).realize()
    np.testing.assert_equal(buf.numpy(), [5, 6, 7, 8, 1, 2, 3, 4])

  def test_reduction_after_partial_assign(self):
    """Reduction over buffer after partial assign - must see the assigned values."""
    buf = Tensor.zeros(4, 4).contiguous().realize()
    buf[0:2, :].assign(Tensor.ones(2, 4))  # top half = 1
    total = buf.sum()
    self.assertEqual(total.item(), 8)

  def test_multiple_reductions_different_views(self):
    """Multiple reductions over different views of same buffer after assign."""
    buf = Tensor.zeros(4, 4).contiguous().realize()
    buf.assign(Tensor.arange(16).reshape(4, 4).float())
    row_sums = buf.sum(axis=1)  # [6, 22, 38, 54]
    col_sums = buf.sum(axis=0)  # [24, 28, 32, 36]
    total = buf.sum()  # 120
    # All should see the assigned values
    np.testing.assert_equal(row_sums.numpy(), [6, 22, 38, 54])
    np.testing.assert_equal(col_sums.numpy(), [24, 28, 32, 36])
    self.assertEqual(total.item(), 120)

  def test_assign_from_self_transformed(self):
    """Assign to buffer from transformed view of itself."""
    buf = Tensor([1, 2, 3, 4]).contiguous().realize()
    # Read and transform, then write back (requires reading before writing)
    buf.assign((buf * 2).contiguous())
    np.testing.assert_equal(buf.numpy(), [2, 4, 6, 8])

  def test_two_buffers_cross_assign(self):
    """Two buffers each reading from the other before writing."""
    a = Tensor([1, 2, 3, 4]).contiguous().realize()
    b = Tensor([10, 20, 30, 40]).contiguous().realize()
    # Both read from each other's original values
    a_new = (a + b).contiguous()
    b_new = (a * b).contiguous()
    a.assign(a_new)
    b.assign(b_new)
    Tensor.realize(a, b)
    np.testing.assert_equal(a.numpy(), [11, 22, 33, 44])
    np.testing.assert_equal(b.numpy(), [10, 40, 90, 160])

  def test_three_buffer_chain(self):
    """Chain: A depends on B, B depends on C - ordering matters."""
    a = Tensor.zeros(4, dtype=dtypes.int32).contiguous().realize()
    b = Tensor([1, 2, 3, 4]).contiguous().realize()
    c = Tensor([10, 10, 10, 10]).contiguous().realize()
    # b reads from c, a reads from b
    b.assign((b + c).contiguous())  # b = [11, 12, 13, 14]
    a.assign((a + b).contiguous())  # a should see new b = [11, 12, 13, 14]
    Tensor.realize(a, b)
    np.testing.assert_equal(b.numpy(), [11, 12, 13, 14])
    np.testing.assert_equal(a.numpy(), [11, 12, 13, 14])

  def test_interleaved_assign_read_patterns(self):
    """Complex interleaved pattern: write A, read A into B, write B, read B."""
    a = Tensor.zeros(4, dtype=dtypes.int32).contiguous().realize()
    b = Tensor.zeros(4, dtype=dtypes.int32).contiguous().realize()

    a.assign(Tensor([1, 2, 3, 4]))
    b.assign(a.contiguous())       # b should get [1,2,3,4]
    a.assign(Tensor([5, 6, 7, 8]))
    result = b.sum()               # should be 10, not 26

    self.assertEqual(result.item(), 10)
    np.testing.assert_equal(a.numpy(), [5, 6, 7, 8])
    np.testing.assert_equal(b.numpy(), [1, 2, 3, 4])

  def test_variable_slice_ordering(self):
    """Variable-indexed slices - conflicting variable binds in same schedule are rejected."""
    v_i = Variable("i", 0, 3)
    buf = Tensor.zeros(4, 4).contiguous().realize()
    buf[v_i.bind(0):v_i.bind(0)+1, :].assign(Tensor.ones(1, 4))
    buf[v_i.bind(1):v_i.bind(1)+1, :].assign(Tensor.ones(1, 4) * 2)
    with self.assertRaises(RuntimeError): buf[0:1, :].sum().item()

  def test_multi_step_assign_read_write_same_buffer(self):
    """Assign to m and param reading b, then update b, across multiple steps.
    This is the optimizer bias-correction pattern from issue #13600: m accumulates,
    param is updated using m/(1-b), and b is updated via *= after the reads."""
    b = Tensor([0.5]).contiguous().realize()
    m = Tensor([0.0]).contiguous().realize()
    param = Tensor([1.0]).contiguous().realize()
    for _ in range(10):
      m.assign(0.9 * m + 0.1)
      param.assign(param - m / (1 - b))
      b *= 0.9
      Tensor.realize(param, m, b)
    # numpy reference
    b_np, m_np, p_np = 0.5, 0.0, 1.0
    for _ in range(10):
      m_np = 0.9 * m_np + 0.1
      p_np = p_np - m_np / (1 - b_np)
      b_np *= 0.9
    np.testing.assert_allclose(param.item(), p_np, atol=1e-5)

  def test_war_reader_already_depends_on_write(self):
    x = Tensor([1.0]).contiguous().realize()
    y = Tensor([2.0]).contiguous().realize()
    x_expr = x + 10          # 11, x is read here, before the assign
    x.assign(x * 2)
    y.assign(y + x)
    z = y + x_expr
    Tensor.realize(x, y, z)
    try:
      np.testing.assert_allclose([x.item(), y.item(), z.item()], [2.0, 4.0, 15.0])
    except AssertionError:
      # TODO: broken now, x_expr reads x after the assign
      np.testing.assert_allclose([x.item(), y.item(), z.item()], [2.0, 4.0, 16.0])

  def test_war_multi_read_then_assign(self):
    devices = ("CPU:0", "CPU:1")
    for realize_reader_first in (False, True):
      buf = Tensor([1., 2., 3., 4.], device="CPU").contiguous().realize().shard(devices, 0).realize()
      stale = buf.to("CPU")
      buf.assign(Tensor.full(buf.shape, 10.0, device="CPU").shard(devices, 0).contiguous().realize())
      Tensor.realize(stale, buf) if realize_reader_first else Tensor.realize(buf, stale)
      np.testing.assert_equal(stale.numpy(), [1., 2., 3., 4.])
      np.testing.assert_equal(buf.numpy(), [10., 10., 10., 10.])

  def test_multiple_slice_assigns_then_read(self):
    """Multiple non-overlapping slice assigns then read."""
    buf = Tensor.zeros(4).contiguous().realize()
    buf[0:1].assign(Tensor.ones(1))
    buf[1:2].assign(Tensor.full((1,), 2.0))
    buf[2:3].assign(Tensor.full((1,), 3.0))
    self.assertEqual(buf.sum().realize().item(), 6.0)

# TODO: assigns into views of unrealized non-BUFFER bases are silently dropped
  def test_read_before_two_assigns(self):
    g = Tensor.full((2,), 4.0).realize()
    before = g + 1                                     # 5
    g.assign(0.0)
    g.assign(g + 4)
    with self.assertRaisesRegex(RuntimeError, "cycle"):  # TODO: broken now, raises
      np.testing.assert_allclose((before + g).numpy(), 9)

  def test_read_between_two_assigns(self):
    a = Tensor.ones(4).realize()
    b = Tensor.full((4,), 10.).realize()
    a.assign(b + 1)                                    # a == 11
    v1 = a * 3                                         # reads 11 -> 33
    a.assign(b + 100)                                  # a == 110
    with self.assertRaisesRegex(RuntimeError, "cycle"):  # TODO: broken now, ideally v1 is realized between the assigns
      np.testing.assert_allclose((a + v1).numpy(), 143)

  def test_two_reads_between_three_assigns(self):
    a = Tensor.zeros(4).realize()
    first = a + 100
    a.assign(Tensor([1., 2., 0., 0.]))
    second = a + 0
    a.assign(a + 10)
    with self.assertRaisesRegex(RuntimeError, "cycle"):  # TODO: broken now, raises
      np.testing.assert_allclose((first + second + a).numpy(), [112, 114, 110, 110])

  def test_read_before_slice_assign(self):
    a = Tensor.ones(4).realize()
    before = a * 3
    a[0:2].assign(Tensor.full((2,), 2.))
    out = (a + (before - 1)).numpy()
    try:
      np.testing.assert_allclose(out, [4, 4, 3, 3])
    except AssertionError:
      # TODO: broken now, before reads the two assigned elements after the assign
      np.testing.assert_allclose(out, [7, 7, 3, 3])

  def test_read_before_assign_survives_a_realize(self):
    a = Tensor.ones(4).realize()
    before = a * 3
    a.assign(Tensor.full((4,), 5.))
    a.realize()
    out = before.numpy()
    try:
      np.testing.assert_allclose(out, 3)
    except AssertionError:
      # TODO: broken now, before is computed again from the assigned value
      np.testing.assert_allclose(out, 15)

  def test_loss_read_after_step_is_the_pre_step_loss(self):
    from tinygrad import nn
    w = Tensor([2.]).contiguous().realize()
    x = Tensor([3.]).realize()
    opt = nn.optim.SGD([w], lr=0.1)
    with Context(TRAINING=1):
      loss = (w*x).sum()                               # 6.0
      loss.backward()
      opt.step()                                       # w becomes 1.7
    out = loss.item()
    try:
      self.assertAlmostEqual(out, 6.0, places=5)
    except AssertionError:
      # TODO: broken now, loss is computed again from the updated weight
      self.assertAlmostEqual(out, 5.1, places=5)

  def test_rand_realized_out_of_order(self):
    Tensor.manual_seed(1)
    r = [Tensor.rand(4) for _ in range(4)]
    r[3].realize()
    out_of_order = r[0].numpy()
    Tensor.manual_seed(1)
    in_order = [Tensor.rand(4).numpy() for _ in range(4)]
    try:
      np.testing.assert_equal(out_of_order, in_order[0])
    except AssertionError:
      # TODO: broken now, r[0] returns the fourth set of numbers
      np.testing.assert_equal(out_of_order, in_order[3])

  def test_batchnorm_stats_are_realized(self):
    from tinygrad import nn
    bn, x = nn.BatchNorm(4), Tensor.randn(2, 4, 3, 3).realize()
    with Context(TRAINING=1): bn(x).realize()
    try:
      self.assertTrue(bn.running_mean.uop.base.is_realized)
    except AssertionError:
      # TODO: broken now, the stat update is never run because nothing reads it
      self.assertFalse(bn.running_mean.uop.base.is_realized)

  def test_batchnorm_under_jit_counts_every_call(self):
    from tinygrad import nn
    bn, x = nn.BatchNorm(4), Tensor.randn(8, 4, 2, 2).realize()
    @TinyJit
    def step(t):
      with Context(TRAINING=1): return bn(t).sum().realize()
    for _ in range(4): step(x)
    out = bn.num_batches_tracked.item()
    try:
      self.assertEqual(out, 4)
    except AssertionError:
      # TODO: broken now, only the calls whose stat update happened to be captured are counted
      self.assertEqual(out, 2)

  def test_assign_from_unrealized_tensor_does_not_alias(self):
    a = Tensor.full((4,), 7.).realize()
    b = Tensor.ones(4) * 1
    b.assign(a)
    b.assign(Tensor.zeros(4))
    b.realize()
    self.assertListEqual(a.tolist(), [7., 7., 7., 7.])

  def test_assign_to_function_output(self):
    from tinygrad import function
    @function
    def f(x:Tensor) -> Tensor: return x*2
    out = f(Tensor.ones(4).realize())
    out.assign(Tensor.full((4,), 9.).realize())
    self.assertListEqual(out.tolist(), [9., 9., 9., 9.])

  def test_nested_function_assign(self):
    from tinygrad import function
    @function
    def inner(x:Tensor) -> Tensor:
      x.assign(x+1)
      return x*2
    @function
    def outer(x:Tensor) -> Tensor:
      y = inner(x)
      x.assign(x+1)
      return y+x
    a = Tensor([1.]).realize()
    with self.assertRaisesRegex(RuntimeError, "cycle"):  # TODO: broken now, ideally y is realized between the assigns
      out = outer(a).item()
      self.assertEqual([out, a.item()], [7., 3.])

class TestAssignToUnrealizedView(unittest.TestCase):
  def test_copy(self):
    t = Tensor.zeros(2,2, dtype=dtypes.int).to("CPU:0").contiguous().realize()
    c = t.to("CPU:1")  # unrealized COPY
    self.assertIs(c.uop.base.op, Ops.COPY)
    c[:, 1:2].assign(Tensor.ones(2,1, dtype=dtypes.int).to("CPU:1").contiguous().realize())
    try:
      self.assertEqual(c.tolist(), [[0,1],[0,1]])
    except AssertionError:
      # TODO: broken now
      self.assertEqual(c.tolist(), [[0,0],[0,0]])

  def test_contiguous(self):
    t = Tensor([[1,2],[3,4]]).contiguous().realize()
    c = t.permute(1,0).contiguous()  # unrealized CONTIGUOUS
    self.assertIs(c.uop.base.op, Ops.COPY)
    c[:, 1:2].assign(Tensor.ones(2,1, dtype=dtypes.int).contiguous().realize())
    self.assertEqual(c.tolist(), [[1,1],[2,1]])

  def test_contiguous_partial_assign_realize(self):
    x = Tensor([1., 2.]).realize()
    y = (x + 1).contiguous()  # unrealized CONTIGUOUS
    self.assertIs(y.uop.base.op, Ops.COPY)
    # a partial write survives an explicit realize: the values are right, storage is an implementation detail
    y[:1].assign(9.)
    y.realize()
    self.assertEqual(y.tolist(), [9., 3.])
    # and it stays assigned across schedules
    y[:1].assign(7.)
    y.realize()
    self.assertEqual(y.tolist(), [7., 3.])
    # setitem syntax gives the same values, contiguous or not
    for mk in (lambda xx: xx + 1, lambda xx: (xx + 1).contiguous()):
      z = mk(Tensor([1., 2.]).realize())
      z[:1] = 9.
      self.assertEqual(z.tolist(), [9., 3.])

  def test_contiguous_backward(self):
    t = Tensor([[1,2],[3,4]]).contiguous().realize()
    cb = t.contiguous_backward()  # unrealized CONTIGUOUS_BACKWARD
    self.assertIs(cb.uop.base.op, Ops.CONTIGUOUS_BACKWARD)
    cb[:, 1:2].assign(Tensor.ones(2,1, dtype=dtypes.int).contiguous().realize())
    try:
      self.assertEqual(cb.tolist(), [[1,1],[3,1]])
    except AssertionError:
      # TODO: broken now
      self.assertEqual(cb.tolist(), [[1,2],[3,4]])

  def test_detach_copy(self):
    t = Tensor.zeros(2,2, dtype=dtypes.int).to("CPU:0").contiguous().realize()
    d = t.to("CPU:1").detach()  # DETACH(unrealized COPY)
    self.assertIs(d.uop.base.op, Ops.COPY)
    d[:, 1:2].assign(Tensor.ones(2,1, dtype=dtypes.int).to("CPU:1").contiguous().realize())
    try:
      self.assertEqual(d.tolist(), [[0,1],[0,1]])
    except AssertionError:
      # TODO: broken now
      self.assertEqual(d.tolist(), [[0,0],[0,0]])

  def test_detach_contiguous(self):
    t = Tensor([[1,2],[3,4]]).contiguous().realize()
    d = t.permute(1,0).contiguous().detach()  # DETACH(unrealized CONTIGUOUS)
    self.assertIs(d.uop.base.op, Ops.COPY)
    d[:, 1:2].assign(Tensor.ones(2,1, dtype=dtypes.int).contiguous().realize())
    self.assertEqual(d.tolist(), [[1,1],[2,1]])

  def test_alu(self):
    a = Tensor([1,2,3,4]).contiguous().realize()
    b = Tensor([5,6,7,8]).contiguous().realize()
    c = a + b  # unrealized ADD
    self.assertIs(c.uop.base.op, Ops.ADD)
    c[:2].assign(Tensor([99, 99]).realize())
    try:
      self.assertEqual(c.tolist(), [99,99,10,12])
    except AssertionError:
      # TODO: broken now, silently dropped
      self.assertEqual(c.tolist(), [6,8,10,12])

  def test_reduce(self):
    a = Tensor([[1,2],[3,4]]).contiguous().realize()
    r = a.sum(axis=0)  # unrealized REDUCE
    self.assertIs(r.uop.base.op, Ops.REDUCE)
    r[:1].assign(Tensor([99]).realize())
    try:
      self.assertEqual(r.tolist(), [99,6])
    except AssertionError:
      # TODO: broken now, silently dropped
      self.assertEqual(r.tolist(), [4,6])

  def test_cast(self):
    a = Tensor([1,2,3,4]).contiguous().realize()
    c = a.float()  # unrealized CAST
    self.assertIs(c.uop.base.op, Ops.CAST)
    c[:2].assign(Tensor([99, 99], dtype=dtypes.float).realize())
    try:
      self.assertEqual(c.tolist(), [99,99,3,4])
    except AssertionError:
      # TODO: broken now, silently dropped
      self.assertEqual(c.tolist(), [1,2,3,4])

  def test_const(self):
    c = Tensor(5).reshape(1, 1).expand(2, 2)
    self.assertIs(c.uop.base.op, Ops.CONST)
    c[:, 1:2].assign(Tensor.ones(2,1, dtype=dtypes.int).contiguous().realize())
    try:
      self.assertEqual(c.tolist(), [[5,1],[5,1]])
    except AssertionError:
      # TODO: broken now, silently dropped
      self.assertEqual(c.tolist(), [[5,5],[5,5]])

  def test_detach_assignment_preserves_earlier_update(self):
    x = Tensor([1., 2.]).detach()
    state = Tensor([0., 0.]).detach()
    state.assign(state + x * 2)
    result = state + 1
    x.assign(x + 1).realize(state, result)
    self.assertEqual(x.tolist(), [2., 3.])
    self.assertEqual(state.tolist(), [2., 4.])
    self.assertEqual(result.tolist(), [3., 5.])

class TestPartialAssignToSharedBuffer(unittest.TestCase):
  def test_five_slices(self):
    big = Tensor.zeros(50).contiguous().realize()
    views = [big[i*10:(i+1)*10].reshape(2, 5) for i in range(5)]
    for v in views: v.assign(v + 1)
    Tensor.realize(*views)
    for v in views:
      np.testing.assert_allclose(v.numpy(), np.ones((2, 5)))

  def test_many_slices(self):
    n_params = 10
    big = Tensor.zeros(n_params * 12).contiguous().realize()
    grads = [big[i*12:(i+1)*12].reshape(3, 4) for i in range(n_params)]
    for g in grads: g.assign(g + 1)
    Tensor.realize(*grads)
    for g in grads:
      np.testing.assert_allclose(g.numpy(), np.ones((3, 4)))

  def test_mixed_shapes(self):
    big = Tensor.zeros(100).contiguous().realize()
    shapes = [(3, 4), (4, 6), (6, 4), (2, 5), (4, 3)]
    pos, views = 0, []
    for s in shapes:
      n = s[0] * s[1]
      views.append(big[pos:pos+n].reshape(*s))
      pos += n
    for v in views: v.assign(v + 1)
    Tensor.realize(*views)
    for v, s in zip(views, shapes):
      np.testing.assert_allclose(v.numpy(), np.ones(s))

class TestAfterCachePatterns(unittest.TestCase):
  def test_double_store_after(self):
    a = Tensor.zeros(10).contiguous()
    b = Tensor.zeros(10).contiguous()
    c = Tensor.ones(10).contiguous()
    Tensor.realize(a, b, c)

    a_store = a.uop.store(c.uop)
    b_store = b.uop.store(c.uop)

    a = Tensor(a.uop.after(a_store, b_store))
    a.realize()
    np.testing.assert_array_equal(a.numpy(), 1)
    np.testing.assert_array_equal(b.numpy(), 1)

  def test_double_store_after_different_sizes(self):
    full = Tensor.zeros(2).contiguous()
    head = Tensor.zeros(1).contiguous()
    full_src = Tensor([1, 2], dtype=dtypes.float).contiguous()
    head_src = Tensor([3], dtype=dtypes.float).contiguous()
    Tensor.realize(full, head, full_src, head_src)

    full_store = full.uop.store(full_src.uop)
    head_store = head.uop.store(head_src.uop)

    head = Tensor(head.uop.after(head_store, full_store))
    head.realize()
    np.testing.assert_array_equal(head.numpy(), [3])
    np.testing.assert_array_equal(full.numpy(), [1, 2])

class TestMultiAssign(unittest.TestCase):
  device = tuple(f"{Device.DEFAULT}:{i}" for i in range(2))

  @needs_second_gpu
  def setUp(self): pass

  def test_multi_assign_realized(self):
    out = Tensor.zeros(4).shard(self.device, 0).contiguous().realize()
    ones = Tensor.ones(4).shard(self.device, 0).contiguous().realize()
    out.assign(ones).realize()
    self.assertListEqual(out.tolist(), [1,1,1,1])

  def test_multi_assign_unrealized(self):
    out = Tensor.zeros(4).contiguous().realize().shard(self.device, 0)
    ones = Tensor.ones(4).shard(self.device, 0).contiguous().realize()
    out.assign(ones).realize()
    self.assertListEqual(out.tolist(), [1,1,1,1])

  def test_multi_assign_both_unrealized(self):
    out = Tensor.zeros(4).contiguous().realize().shard(self.device, 0)
    ones = Tensor.ones(4).contiguous().realize().shard(self.device, 0)
    out.assign(ones).realize()
    self.assertListEqual(out.tolist(), [1,1,1,1])

  def test_multi_assign_scalar(self):
    out = Tensor.ones(4).shard(self.device, 0).contiguous().realize()
    out.assign(0).realize()
    self.assertListEqual(out.tolist(), [0,0,0,0])

  def test_multi_assign_const_like(self):
    out = Tensor.ones(4).shard(self.device, 0).contiguous().realize()
    out.assign(out.const_like(7)).realize()
    self.assertListEqual(out.tolist(), [7,7,7,7])

  def test_multi_assign_piece(self):
    out = Tensor.zeros(4,4).shard(self.device, 0).contiguous().realize()
    ones = Tensor.ones(4,1).shard(self.device, 0).contiguous().realize()
    out[:, 2:3].assign(ones).realize()
    self.assertListEqual(out.tolist(), [[0,0,1,0], [0,0,1,0], [0,0,1,0], [0,0,1,0]])

  def test_multi_assign_piece_noncontig(self):
    out = Tensor.zeros(4,4).contiguous().realize().shard(self.device, 0).realize()
    ones = Tensor.ones(4,1).shard(self.device, 0).contiguous().realize()
    out[:, 2:3].assign(ones).realize()
    self.assertListEqual(out.tolist(), [[0,0,1,0], [0,0,1,0], [0,0,1,0], [0,0,1,0]])

  def test_multi_assign_piece_unrealized(self):
    out = Tensor.zeros(4,4).contiguous().realize().shard(self.device, 0)
    ones = Tensor.ones(4,1).shard(self.device, 0).contiguous().realize()
    out[:, 2:3].assign(ones).realize()
    try:
      self.assertListEqual(out.tolist(), [[0,0,1,0], [0,0,1,0], [0,0,1,0], [0,0,1,0]])
    except AssertionError:
      # TODO: broken now, the write is dropped
      self.assertListEqual(out.tolist(), [[0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]])

  def test_multi_assign_var_offset(self):
    out = Tensor.zeros(4,4).contiguous().realize().shard(self.device, 0).realize()
    ones = Tensor.ones(4,1).shard(self.device, 0).contiguous().realize()
    vi = Variable("i", 0, 3).bind(2)
    out[:, vi:vi+1].assign(ones).realize()
    self.assertListEqual(out.tolist(), [[0,0,1,0], [0,0,1,0], [0,0,1,0], [0,0,1,0]])

  def test_multi_assign_var_offset_jit_none(self): self.test_multi_assign_var_offset_jit(None)
  def test_multi_assign_var_offset_jit(self, shard_axis=0):
    out = Tensor.zeros(4,6).contiguous().realize().shard(self.device, shard_axis).realize()
    ones = Tensor.ones(4,1).shard(self.device, shard_axis).contiguous().realize()

    @TinyJit
    def f(out:Tensor, vi):
      out[:, vi:vi+1].assign(ones).realize()
      ones.assign(ones+1).realize()

    vi = Variable("i", 0, 5)
    for i in range(1,5):
      GlobalCounters.reset()
      f(out, vi.bind(i))
    self.assertListEqual(out.tolist(), [[0,1,2,3,4,0]]*4)

if __name__ == "__main__":
  unittest.main()
