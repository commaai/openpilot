#!/usr/bin/env python
import unittest
import numpy as np
from tinygrad import dtypes, Tensor, TinyJit, GlobalCounters, Variable
from tinygrad.device import is_dtype_supported
from tinygrad.helpers import temp, CI, CPU_LVP, Context

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

  def test_assign_zeros_good(self):
    a = Tensor.zeros(10,10).contiguous()
    a.assign(Tensor.ones(10,10))
    b = Tensor.zeros(10,10).contiguous()
    a.realize()
    np.testing.assert_allclose(b.numpy(), 0)

  def test_assign_zeros(self):
    a = Tensor.zeros(10,10).contiguous()
    b = Tensor.zeros(10,10).contiguous()
    a.assign(Tensor.ones(10,10))
    a.realize()
    np.testing.assert_allclose(b.numpy(), 0)

  def test_assign_add(self):
    def f(x):
      x += 1
      x.realize()
    x = Tensor([0])
    f(x)
    assert x.item() == 1

  def test_assign_add_twice(self):
    # NOTE: this has two kernels
    def f(x):
      x += 1
      x += 1
      x.realize()
    x = Tensor([0])
    f(x)
    assert x.item() == 2

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

  def test_assign_add_jit(self):
    @TinyJit
    def f(x):
      x += 1
      x.realize()
    x = Tensor([0])
    for _ in range(5): f(x)
    assert x.item() == 5

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
    b = a.contiguous()    # b returns a new Tensor
    b.assign(2)
    b.realize()
    self.assertNotEqual(a.item(), b.item())
  # on a realized Tensor contiguous child changes the source
  @unittest.expectedFailure
  def test_assign_changes_realized_alt(self): return self.test_assign_changes_alt(realize=True)

  @unittest.skip("assign to contiguous shouldn't change the base buffer")
  def test_assign_changes_buffer_alt(self):
    a, b = [Tensor(Tensor(0).contiguous().realize().uop.as_buf()) for _ in range(2)]
    Tensor.realize(a.contiguous().assign(1), b.contiguous().assign(2))
    self.assertEqual((a + b).item(), 3)

  def test_assign_diamond_cycle(self):
    # NOTE: should *not* raise AssertionError from numpy
    with self.assertRaisesRegex(RuntimeError, "cycle"):
      a = Tensor.ones(4).contiguous().realize()
      times_a = a*3
      a.assign(Tensor.full((4,), 2.).contiguous())
      new = a + (times_a-1)
      np.testing.assert_allclose(new.numpy(), 4)

  def test_assign_diamond_contiguous_cycle(self):
    with self.assertRaisesRegex(RuntimeError, "cycle"):
      a = Tensor.ones(4).contiguous().realize()
      times_a = a*3
      a.assign(Tensor.full((4,), 2.))
      new = a.contiguous() + times_a-1
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

  @unittest.skipIf(CI and CPU_LVP, "flaky in CI")
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

  @unittest.skip("TODO: bring this assert back")
  def test_crossunder_assign(self):
    # NOTE: should *not* raise AssertionError from numpy
    with self.assertRaisesRegex(RuntimeError, "cycle"):
      a = Tensor.full((4,), 2).contiguous().realize()
      b = Tensor.full((4,), 3).contiguous().realize()
      c = a+9
      a += b
      b += c
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

  def test_assign_contiguous(self):
    b = Tensor.rand(4,4).realize()
    a = (Tensor.rand(4,4).realize() + 1)
    kc = GlobalCounters.kernel_count
    b.assign(a.contiguous()).realize()
    assert GlobalCounters.kernel_count - kc == 2

  def test_assign_contiguous_permute(self):
    b = Tensor.rand(4,4).realize()
    a = (Tensor.rand(4,4).realize() + 1).permute((1,0))
    kc = GlobalCounters.kernel_count
    b.assign(a.contiguous()).realize()
    assert GlobalCounters.kernel_count - kc == 2

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

  def test_post_permuted_assignment(self):
    a = Tensor(np.arange(N*N, dtype=np.float32)).reshape(N,N)
    b = Tensor(np.arange(N*N, dtype=np.float32)).reshape(N,N)
    a.realize()
    b.realize()
    #GlobalCounters.cache = []
    ba1 = a.uop.base.realized # noqa: F841
    bb1 = b.uop.base.realized # noqa: F841
    a.assign(a.permute(1,0) + b)   # this should not work!
    a.realize()
    ba2 = a.uop.base.realized # noqa: F841
    # NOTE: don't test that it's assigned
    #assert ba1 == ba2 and ba1 != bb1
    np.testing.assert_allclose(a.numpy(), np.arange(N*N).reshape((N,N)) + np.arange(N*N).reshape((N,N)).transpose(1,0))

  def test_post_permuted_assignment_alt(self):
    a = Tensor.arange(N*N).reshape(N,N).contiguous().realize()
    b = Tensor.arange(N*N).reshape(N,N).contiguous().realize()
    new_a = (a.T+b).numpy()
    a.assign(a.T+b)
    np.testing.assert_allclose(a.numpy(), new_a)

  def test_post_flipped_assignment(self):
    a = Tensor.arange(N*N).reshape(N,N).contiguous().realize()
    b = Tensor.arange(N*N).reshape(N,N).contiguous().realize()
    new_a = (a.flip(0)+b).numpy()
    a.assign(a.flip(0)+b)
    np.testing.assert_allclose(a.numpy(), new_a)

  def test_post_flipped_assignment_axis1(self):
    a = Tensor.arange(N*N).reshape(N,N).contiguous().realize()
    b = Tensor.arange(N*N).reshape(N,N).contiguous().realize()
    new_a = (a.flip(1)+b).numpy()
    a.assign(a.flip(1)+b)
    np.testing.assert_allclose(a.numpy(), new_a)

  def test_post_reshape_assignment_fine(self):
    a = Tensor.arange(N*N).reshape(N, N).contiguous().realize()
    b = Tensor.arange(N*N).reshape(N, N).contiguous().realize()
    rhs = a.reshape(-1).reshape(N, N)
    new_a = (rhs+b).numpy()
    a.assign(rhs+b)  # self-assign with reshape view is fine
    np.testing.assert_allclose(a.numpy(), new_a)

  @unittest.skip("multi output not supported anymore")
  def test_simple_assignment_multioutput(self):
    a = Tensor.randn(32, 32).realize()
    b = Tensor.full((32, ), 1.).contiguous().realize()
    c = Tensor.full((32, ), 2.).contiguous().realize()
    d = Tensor.full((32, ), 3.).contiguous().realize()

    r = a.sum(axis=1)
    b.assign(r + b)
    c.assign(r + c)
    d.assign(r + d)

    kc = GlobalCounters.kernel_count
    Tensor.realize(b, c, d)
    assert GlobalCounters.kernel_count - kc == 1
    np.testing.assert_allclose(b.numpy(), a.sum(1).numpy()+1)
    np.testing.assert_allclose(c.numpy(), a.sum(1).numpy()+2)
    np.testing.assert_allclose(d.numpy(), a.sum(1).numpy()+3)

  # NOTE: if the assign target is read/write in a single kernel, it should be contiguous

  def test_permuted_assignment_correct(self):
    a = Tensor.arange(4 * 4).reshape(4, 4).contiguous().realize()
    b = Tensor.arange(4 * 4).reshape(4, 4).contiguous().realize()
    a = a.permute(1, 0)
    new_val = a + b
    a.assign(new_val)
    np.testing.assert_equal(a.numpy(), np.arange(4 * 4).reshape(4, 4).transpose(1, 0) + np.arange(4 * 4).reshape(4, 4))

  def test_permuted_reduceop_child_dual_use(self):
    a = Tensor.randn(32, 32, 32).realize()
    b = Tensor.full((32, 32), 1.).contiguous().realize()
    r = a.sum(axis=1)
    b.assign(r + b.permute(1, 0))
    b.realize()
    np.testing.assert_allclose(b.numpy(), a.numpy().sum(axis=1)+np.ones((32, 32)).transpose(1, 0), atol=1e-6, rtol=1e-3)

  @unittest.skip("multi output not supported anymore")
  def test_permuted_reduceop_multioutput_dual_use(self):
    a = Tensor.randn(32, 32, 32).realize()
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
    a = Tensor.randn(32, 32, 32, dtype=dtypes.int).realize()
    b = Tensor.arange(32 * 32).reshape(32, 32).realize()
    c = Tensor.arange(32 * 32).reshape(32, 32).realize()

    kc = GlobalCounters.kernel_count
    r = a.sum(axis=1)
    b_perm = b.permute(1, 0)
    b.assign(r + b)
    c.assign(r + b_perm.contiguous())
    Tensor.realize(b, c)
    assert GlobalCounters.kernel_count - kc == 2
    np.testing.assert_equal(b.numpy(), a.numpy().sum(1) + np.arange(32 * 32).reshape(32, 32))
    np.testing.assert_equal(c.numpy(), a.numpy().sum(1) + np.arange(32 * 32).reshape(32, 32).transpose(1, 0))

  def test_permuted_assignment_masked_view_possible(self):
    a = Tensor.ones(4, 4).contiguous().realize()
    b = a.shrink((None, (0, 2))).pad((None, (0, 2)), value=2)
    a.assign(a + b)
    kc = GlobalCounters.kernel_count
    a.realize()
    assert GlobalCounters.kernel_count - kc == 1
    np.testing.assert_equal(a.numpy(), np.ones((4, 4))+np.pad(np.ones((4, 4))[:, 0:2], ((0, 0), (0, 2)), constant_values=2))

  def test_permuted_assignment_masked_view_not_contiguous(self):
    a = Tensor.ones(4, 4).contiguous().realize()
    b = a.shrink((None, (0, 2))).pad((None, (0, 2)), value=2).permute(1, 0)
    a.assign(a + b)
    a.realize()
    self.assertListEqual(a.tolist(), [[2.,2.,2.,2.],[2.,2.,2.,2.],[3.,3.,3.,3.], [3.,3.,3.,3.]])

  # TODO: is there a way to sneak in a permute such that it returns the wrong answer?

  @unittest.skip("this test is crashing!")
  def test_overlapping_shrink_assignment_forward(self):
    # Forward shift: read index > write index in overlap
    N = 100000
    shift = 1000
    a = Tensor.arange(N).float().contiguous().realize()
    expected = np.arange(N, dtype=np.float32)
    expected[:N-shift] = expected[shift:].copy()
    with Context(NOOPT=1): a[0:N-shift].assign(a[shift:N]).realize()
    np.testing.assert_allclose(a.numpy(), expected)

  @unittest.skip("this test is crashing!")
  def test_overlapping_shrink_assignment_reverse(self):
    # Reverse shift: write index > read index in overlap
    N = 100000
    shift = 1000
    a = Tensor.arange(N).float().contiguous().realize()
    expected = np.arange(N, dtype=np.float32)
    expected[shift:] = expected[:N-shift].copy()
    with Context(NOOPT=1): a[shift:N].assign(a[0:N-shift]).realize()
    np.testing.assert_allclose(a.numpy(), expected)

  @unittest.skip("this test is crashing!")
  def test_nonoverlapping_shrink_assignment(self):
    # TODO: non-overlapping shrinks don't actually need contiguous, could be 1 kernel with smarter range analysis
    a = Tensor.arange(100).float().contiguous().realize()
    expected = np.arange(100, dtype=np.float32)
    expected[0:10] = expected[50:60].copy()
    kc = GlobalCounters.kernel_count
    a[0:10].assign(a[50:60]).realize()
    assert GlobalCounters.kernel_count - kc == 2, "currently conservative, forces contiguous"
    np.testing.assert_allclose(a.numpy(), expected)

  @unittest.skipUnless(is_dtype_supported(dtypes.half), "need half")
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

  def test_assign_bitcast(self):
    # assign to a bitcast view should modify the underlying buffer
    a = Tensor([1.0, 2.0, 3.0, 4.0], dtype=dtypes.float32).realize()
    # IEEE 754: 1.0f = 0x3f800000, 2.0f = 0x40000000, 3.0f = 0x40400000, 4.0f = 0x40800000
    a.bitcast(dtypes.uint32).assign(Tensor([0x40800000, 0x40400000, 0x40000000, 0x3f800000], dtype=dtypes.uint32)).realize()
    np.testing.assert_allclose(a.numpy(), [4.0, 3.0, 2.0, 1.0])
    # double bitcast
    b = Tensor([1.0, 2.0, 3.0, 4.0], dtype=dtypes.float32).realize()
    b.bitcast(dtypes.uint32).bitcast(dtypes.int32).assign(Tensor([0x40800000, 0x40400000, 0x40000000, 0x3f800000], dtype=dtypes.int32)).realize()
    np.testing.assert_allclose(b.numpy(), [4.0, 3.0, 2.0, 1.0])
    # shrink then bitcast
    c = Tensor([1.0, 2.0, 3.0, 4.0], dtype=dtypes.float32).realize()
    c[0:2].bitcast(dtypes.uint32).assign(Tensor([0x40800000, 0x40400000], dtype=dtypes.uint32)).realize()
    np.testing.assert_allclose(c.numpy(), [4.0, 3.0, 3.0, 4.0])

  def test_assign_bitcast_different_size(self):
    # different-size bitcast creates a new tensor, not a view, so assign doesn't modify the original
    a = Tensor([0]*8, dtype=dtypes.uint8).realize()
    a.bitcast(dtypes.int64).assign(Tensor([12345], dtype=dtypes.int64)).realize()
    np.testing.assert_equal(a.numpy(), [0]*8)

  @unittest.skip("don't use output buffer, and mismatch dtype no longer supported")
  def test_cast_assignment(self):
    a = Tensor(np.arange(N*N, dtype=np.float32)).reshape(N,N)
    a.realize()
    oba1 = a.uop.base.output_buffer
    a.assign(a.cast(dtypes.int32).realize())
    a.realize()
    oba2 = a.uop.base.output_buffer
    assert oba1 is None and oba2 is None
    np.testing.assert_allclose(a.numpy(), np.arange(N*N,dtype=np.int32).reshape((N,N)))

  def test_assign_dtype_mismatch(self):
    # assign should not implicitly cast dtypes - this can lose precision
    a = Tensor.zeros(4, dtype=dtypes.float32).contiguous().realize()
    b = Tensor([1, 2, 3, 4], dtype=dtypes.int32)
    with self.assertRaisesRegex(RuntimeError, "assign dtype mismatch"):
      a.assign(b)

  def test_assign_dtype_mismatch_int64_to_float32(self):
    # int64 -> float32 loses precision for large values, should not be implicit
    a = Tensor.zeros(1, dtype=dtypes.float32).contiguous().realize()
    b = Tensor([16777217], dtype=dtypes.int64)  # 2^24 + 1, not exactly representable in float32
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

    # without .realize() after assign, the read doesn't see the assigned values
    cache = Tensor.zeros(4, 4).contiguous().realize()
    cache[v_pos:v_pos+1, :].assign(Tensor.ones(1, 4))
    self.assertEqual(cache.sum().item(), 0.0)  # should be 4.0!

    # TODO: remove .realize() workaround once assign-read dependency is fixed
    cache2 = Tensor.zeros(4, 4).contiguous().realize()
    cache2[v_pos:v_pos+1, :].assign(Tensor.ones(1, 4)).realize()
    self.assertEqual(cache2.sum().item(), 4.0)

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
    # without .realize(): assigns not executed, buffer stays zeros
    buf = Tensor.zeros(8).contiguous().realize()
    buf[0:4].assign(Tensor.ones(4))
    buf[2:6].assign(Tensor.ones(4) * 2)
    np.testing.assert_equal(buf.numpy(), [0,0,0,0,0,0,0,0])  # TODO: wrong! should be [1,1,2,2,2,2,0,0]

    # with .realize(): assigns execute in order
    buf = Tensor.zeros(8).contiguous().realize()
    buf[0:4].assign(Tensor.ones(4)).realize()
    buf[2:6].assign(Tensor.ones(4) * 2).realize()
    np.testing.assert_equal(buf.numpy(), [1,1,2,2,2,2,0,0])

  def test_overlapping_slice_assigns_reverse(self):
    """Overlapping slice assigns in reverse order."""
    # without .realize(): assigns not executed
    buf = Tensor.zeros(8).contiguous().realize()
    buf[2:6].assign(Tensor.ones(4) * 2)
    buf[0:4].assign(Tensor.ones(4))
    np.testing.assert_equal(buf.numpy(), [0,0,0,0,0,0,0,0])  # TODO: wrong! should be [1,1,1,1,2,2,0,0]

    # with .realize(): assigns execute in order
    buf = Tensor.zeros(8).contiguous().realize()
    buf[2:6].assign(Tensor.ones(4) * 2).realize()
    buf[0:4].assign(Tensor.ones(4)).realize()
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
    self.assertEqual(mid_sum.realize().item(), 12)

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
    # without .realize(): orphan slice assign not triggered by .numpy()
    buf = Tensor.zeros(4, dtype=dtypes.int32).contiguous().realize()
    buf[1:3].assign(Tensor([5, 6]))
    np.testing.assert_equal(buf.numpy(), [0, 0, 0, 0])  # TODO: wrong! should be [0, 5, 6, 0]

    # with .realize(): assign executes
    buf = Tensor.zeros(4, dtype=dtypes.int32).contiguous().realize()
    buf[1:3].assign(Tensor([5, 6])).realize()
    np.testing.assert_equal(buf.numpy(), [0, 5, 6, 0])

  def test_chained_slice_copies(self):
    """Copy from one slice to another within same buffer."""
    # without .realize(): orphan slice assign not triggered
    buf = Tensor([1, 2, 3, 4, 5, 6, 7, 8]).contiguous().realize()
    buf[4:8].assign(buf[0:4].contiguous())
    np.testing.assert_equal(buf.numpy(), [1, 2, 3, 4, 5, 6, 7, 8])  # TODO: wrong! should be [1,2,3,4,1,2,3,4]

    # with .realize(): assign executes
    buf = Tensor([1, 2, 3, 4, 5, 6, 7, 8]).contiguous().realize()
    buf[4:8].assign(buf[0:4].contiguous()).realize()
    np.testing.assert_equal(buf.numpy(), [1, 2, 3, 4, 1, 2, 3, 4])

  def test_swap_slices(self):
    """Swap two non-overlapping slices - requires reading both before writing."""
    # without .realize() on temps: values not captured before overwriting
    buf = Tensor([1, 2, 3, 4, 5, 6, 7, 8]).contiguous().realize()
    left = buf[0:4].contiguous()  # lazy - not captured yet
    right = buf[4:8].contiguous()  # lazy - not captured yet
    buf[0:4].assign(right).realize()  # this works
    buf[4:8].assign(left).realize()  # left now reads from modified buf!
    np.testing.assert_equal(buf.numpy(), [5, 6, 7, 8, 5, 6, 7, 8])  # TODO: wrong! should be [5,6,7,8,1,2,3,4]

    # with .realize() on temps: values captured before writes
    buf = Tensor([1, 2, 3, 4, 5, 6, 7, 8]).contiguous().realize()
    left = buf[0:4].contiguous().realize()
    right = buf[4:8].contiguous().realize()
    buf[0:4].assign(right).realize()
    buf[4:8].assign(left).realize()
    np.testing.assert_equal(buf.numpy(), [5, 6, 7, 8, 1, 2, 3, 4])

  def test_reduction_after_partial_assign(self):
    """Reduction over buffer after partial assign - must see the assigned values."""
    # without .realize(): orphan slice assign not triggered by reduction
    buf = Tensor.zeros(4, 4).contiguous().realize()
    buf[0:2, :].assign(Tensor.ones(2, 4))  # top half = 1
    total = buf.sum()
    self.assertEqual(total.item(), 0)  # TODO: wrong! should be 8 (2*4 ones)

    # with .realize(): assign executes before reduction
    buf = Tensor.zeros(4, 4).contiguous().realize()
    buf[0:2, :].assign(Tensor.ones(2, 4)).realize()
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
    """Variable-indexed slices - tests symbolic dependency tracking."""
    v_i = Variable("i", 0, 3)

    # without .realize(): orphan slice assigns not triggered
    buf = Tensor.zeros(4, 4).contiguous().realize()
    buf[v_i.bind(0):v_i.bind(0)+1, :].assign(Tensor.ones(1, 4))
    row0_sum = buf[0:1, :].sum()
    self.assertEqual(row0_sum.item(), 0)  # TODO: wrong! should be 4

    # with .realize(): assigns execute
    buf = Tensor.zeros(4, 4).contiguous().realize()
    buf[v_i.bind(0):v_i.bind(0)+1, :].assign(Tensor.ones(1, 4)).realize()
    row0_sum = buf[0:1, :].sum()
    buf[v_i.bind(1):v_i.bind(1)+1, :].assign(Tensor.ones(1, 4) * 2).realize()
    row1_sum = buf[1:2, :].sum()
    self.assertEqual(row0_sum.item(), 4)
    self.assertEqual(row1_sum.item(), 8)

  def test_multiple_slice_assigns_then_read(self):
    """Multiple non-overlapping slice assigns then read - RAW dependencies must ensure all writes complete before read."""
    buf = Tensor.zeros(4).contiguous().realize()
    buf[0:1].assign(Tensor.ones(1))
    buf[1:2].assign(Tensor.full((1,), 2.0))
    buf[2:3].assign(Tensor.full((1,), 3.0))
    self.assertEqual(buf.sum().realize().item(), 0.0)  # TODO: wrong! should be 1 + 2 + 3 + 0 = 6

    buf = Tensor.zeros(4).contiguous().realize()
    buf[0:1].assign(Tensor.ones(1)).realize()
    buf[1:2].assign(Tensor.full((1,), 2.0)).realize()
    buf[2:3].assign(Tensor.full((1,), 3.0)).realize()
    self.assertEqual(buf.sum().realize().item(), 6.0)

if __name__ == "__main__":
  unittest.main()
