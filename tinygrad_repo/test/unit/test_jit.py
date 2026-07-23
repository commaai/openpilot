import unittest, numpy as np
from test.helpers import assert_jit_cache_len
from tinygrad import Tensor, TinyJit, Context, UOp, dtypes
from tinygrad.engine.jit import JitError

def _simple_test(add, extract=lambda x: x, N=10):
  for _ in range(5):
    a = Tensor.randn(N, N)
    b = Tensor.randn(N, N)
    c = add(a, b)
    np.testing.assert_allclose(extract(c).numpy(), a.numpy()+b.numpy(), atol=1e-4, rtol=1e-5)
  assert_jit_cache_len(add, 1)

class TestJit(unittest.TestCase):
  def test_jitbeam_triggers_beam(self):
    from unittest.mock import patch
    from tinygrad.helpers import getenv as _getenv
    @TinyJit
    def add(a, b): return (a+b).realize()
    a, b = Tensor.ones(10, 10).contiguous().realize(), Tensor.ones(10, 10).contiguous().realize()
    with patch("tinygrad.codegen.opt.search.beam_search", wraps=lambda k,*a,**kw: k) as mock_beam:
      add(a, b)
      assert mock_beam.call_count == 0
      with patch("tinygrad.engine.jit.getenv", side_effect=lambda k, d=0: 1 if k == "JITBEAM" else _getenv(k, d)): add(a, b)
      assert mock_beam.call_count == 1

  def test_simple_jit_reset(self):
    @TinyJit
    def add(a, b): return (a+b).realize()
    _simple_test(add)
    add.reset()
    _simple_test(add, N=20)

  def test_simple_jit_norealize(self):
    @TinyJit
    def add(a, b): return (a+b)
    _simple_test(add)

  def test_simple_jit_norealize_list(self):
    @TinyJit
    def add(a, b): return [a+b]
    _simple_test(add, extract=lambda x: x[0])

  def test_simple_jit_norealize_dict(self):
    @TinyJit
    def add(a, b): return {"billy": a+b}
    _simple_test(add, extract=lambda x: x["billy"])

  def test_jit_multiple_outputs(self):
    @TinyJit
    def f(a, b): return (a+b).realize(), (a-b).realize(), (a*b).realize()
    for _ in range(5):
      a = Tensor.randn(10, 10)
      b = Tensor.randn(10, 10)
      c, d, e = f(a, b)
      np.testing.assert_allclose(c.numpy(), a.numpy()+b.numpy(), atol=1e-4, rtol=1e-5)
      np.testing.assert_allclose(d.numpy(), a.numpy()-b.numpy(), atol=1e-4, rtol=1e-5)
      np.testing.assert_allclose(e.numpy(), a.numpy()*b.numpy(), atol=1e-4, rtol=1e-5)
    assert_jit_cache_len(f, 3)

  def test_nothing_jitted(self):
    @TinyJit
    def add(a, b): return None
    with self.assertRaises(JitError):
      for _ in range(5):
        a = Tensor.randn(10, 10)
        b = Tensor.randn(10, 10)
        add(a, b)

  def test_jit_zero_does_not_jit(self):
    @TinyJit
    def add(a, b): return (a+b).realize()
    with Context(JIT=0):
      for i in range(5):
        a = Tensor([i])
        b = Tensor([i])
        c = add(a, b)
        np.testing.assert_allclose(c.numpy(), 2*i)
      assert_jit_cache_len(add, 0)

  def test_jit_not_capturing(self):
    @TinyJit
    def add(a, b):
      Tensor.zeros(4, 4).contiguous().realize()  # no-op kernel is captured
      return (a+b).realize()
    for i in range(5):
      a = Tensor([i])
      b = Tensor([i])
      c = add(a, b)
      np.testing.assert_allclose(c.numpy(), 2*i)
    assert_jit_cache_len(add, 2)

    @TinyJit
    def add2(a, b):
      with Context(CAPTURING=0):  # not captured
        Tensor.zeros(4, 4).contiguous().realize()
      return (a+b).realize()
    for i in range(5):
      a = Tensor([i])
      b = Tensor([i])
      c = add2(a, b)
      np.testing.assert_allclose(c.numpy(), 2*i)
    assert_jit_cache_len(add2, 1)

  def test_jit_shape_mismatch(self):
    @TinyJit
    def add(a, b): return (a+b).realize()
    for _ in range(5):
      a = Tensor.randn(10, 10)
      b = Tensor.randn(10, 10)
      add(a, b)
    bad = Tensor.randn(20, 20)
    with self.assertRaises(JitError):
      add(a, bad)

  def test_jit_shape_views_mismatch(self):
    @TinyJit
    def add(a): return (a+1).realize()
    with self.assertRaises(JitError):
      for i in range(1,5):
        # a has an offset that the kernel doesn't know about
        a = Tensor.randn(10, 10).realize()[:, i:i+2]
        add(a)

  def test_jit_duplicate_fail(self):
    # the jit doesn't support duplicate arguments
    @TinyJit
    def add(a, b): return (a+b).realize()
    a = Tensor.randn(10, 10)
    with self.assertRaises(JitError):
      add(a, a)

  def test_kwargs_jit(self):
    @TinyJit
    def add_kwargs(first, second): return (first+second).realize()
    for _ in range(5):
      a = Tensor.randn(10, 10)
      b = Tensor.randn(10, 10)
      c = add_kwargs(first=a, second=b)
      np.testing.assert_allclose(c.numpy(), a.numpy()+b.numpy(), atol=1e-4, rtol=1e-5)
    assert_jit_cache_len(add_kwargs, 1)

  def test_array_jit(self):
    @TinyJit
    def add_array(a, arr): return (a+arr[0]).realize()
    for _ in range(5):
      a, b = Tensor.randn(10, 10).realize(), Tensor.randn(10, 10).realize()
      np.testing.assert_allclose(add_array(a, [b]).numpy(), a.numpy()+b.numpy(), atol=1e-4, rtol=1e-5)
    assert_jit_cache_len(add_array, 1)

  def test_method_jit(self):
    class Fun:
      def __init__(self):
        self.a = Tensor.randn(10, 10)
      @TinyJit
      def __call__(self, b:Tensor) -> Tensor:
        return (self.a+b).realize()
    fun = Fun()
    for _ in range(5):
      b = Tensor.randn(10, 10)
      c = fun(b)
      np.testing.assert_allclose(c.numpy(), fun.a.numpy()+b.numpy(), atol=1e-4, rtol=1e-5)
    assert_jit_cache_len(fun.__call__.func.__self__, 1)

  def test_jit_size1_input(self):
    @TinyJit
    def f(a, b): return (a+b).realize()
    a = Tensor([1, 2, 3])
    for i in range(5):
      np.testing.assert_allclose(f(a, Tensor([i])).numpy(), (a+i).numpy(), atol=1e-4, rtol=1e-5)
    assert_jit_cache_len(f, 1)

  def test_jit_output_non_tensor_fail(self):
    @TinyJit
    def f(a, b, i): return (a+b).realize(), i
    with self.assertRaises(JitError):
      for i in range(3):
        f(Tensor.randn(10, 10), Tensor.randn(10, 10), i)

  def test_jit_random_regen(self):
    def f(a, b):
      rn = Tensor.randn(*a.shape)
      return ((a+b)*rn).realize()
    a = Tensor.randn(10, 10).realize()  # realize these before resetting the random seed
    b = Tensor.randn(10, 10).realize()

    Tensor.manual_seed(1234)
    jf = TinyJit(f)
    res = set()
    for _ in range(5):
      o1 = jf(a, b)
      res.add(o1.numpy()[0][0])
    assert len(res) == 5, "All values should be different, rand works in jit."

    Tensor.manual_seed(1234)
    jf2 = TinyJit(f)
    res2 = set()
    for _ in range(5):
      o1 = jf2(a, b)
      res2.add(o1.numpy()[0][0])
    assert len(res2) == 5, "All values should be different, rand works in jit."
    assert res == res2, "Jit rand is not reproducible with the same seed"

    Tensor.manual_seed(3421)
    jf3 = TinyJit(f)
    res3 = set()
    for _ in range(5):
      o1 = jf3(a, b)
      res3.add(o1.numpy()[0][0])
    assert len(res3) == 5, "All values should be different, rand works in jit."
    assert res3 != res2, "Jit rand is diff with diff seeds"

  def test_jit_v_nojit_random_regen(self):
    def f(a, b):
      rn = Tensor.randn(*a.shape)
      rn = rn * a
      rn2 = Tensor.randn(*a.shape)
      rn2 = rn2 * b
      rn = rn + rn2
      rn2 = rn2 + Tensor.randn(*a.shape)
      return ((a+b)*rn).realize(), ((a+b)*rn2).realize()
    Tensor.manual_seed(0)
    a = Tensor.randn(10, 10).realize()  # realize these before resetting the random seed
    b = Tensor.randn(10, 10).realize()

    Tensor.manual_seed(1234)
    without_jit = set()
    for _ in range(5):
      o1, o2 = f(a, b)
      without_jit.add(o1.numpy()[0][0])
      without_jit.add(o2.numpy()[0][0])
    assert len(without_jit) == 10, "All values should be different."

    Tensor.manual_seed(1234)
    jf = TinyJit(f)
    with_jit = set()
    for _ in range(5):
      o1, o2 = jf(a, b)
      with_jit.add(o1.numpy()[0][0])
      with_jit.add(o2.numpy()[0][0])
    assert len(with_jit) == 10, "All values should be different."
    assert with_jit == without_jit, "jit and non-jit should produce the same random values with the same seed"

  def test_jit_multiple_random_regen(self):
    def f(a, b):
      rn = Tensor.randn(*a.shape)
      rn = rn * a
      rn2 = Tensor.randn(*a.shape)
      rn2 = rn2 * b
      rn = rn + rn2
      rn2 = rn2 + Tensor.randn(*a.shape)
      return ((a+b)*rn).realize(), ((a+b)*rn2).realize()
    a = Tensor.randn(10, 10).realize()  # realize these before resetting the random seed
    b = Tensor.randn(10, 10).realize()

    Tensor.manual_seed(1234)
    jf = TinyJit(f)
    res = set()
    for _ in range(5):
      o1, o2 = jf(a, b)
      res.add(o1.numpy()[0][0])
      res.add(o2.numpy()[0][0])
    assert len(res) == 10, "All values should be different, rand works in jit."

    Tensor.manual_seed(1234)
    jf2 = TinyJit(f)
    res2 = set()
    for _ in range(5):
      o1, o2 = jf2(a, b)
      res2.add(o1.numpy()[0][0])
      res2.add(o2.numpy()[0][0])
    assert len(res2) == 10, "All values should be different, rand works in jit."
    assert res == res2, "Jit rand is not reproducible with the same seed"

    Tensor.manual_seed(3421)
    jf3 = TinyJit(f)
    res3 = set()
    for _ in range(5):
      o1, o2 = jf3(a, b)
      res3.add(o1.numpy()[0][0])
      res3.add(o2.numpy()[0][0])
    assert len(res3) == 10, "All values should be different, rand works in jit."
    assert res3 != res2, "Jit rand is diff with diff seeds"

  def test_jit_random_after_unrealized_random(self):
    @TinyJit
    def f(): return Tensor.rand()
    Tensor.manual_seed(1234)
    Tensor.rand()
    res = [f().numpy() for _ in range(3)]
    assert res[1] != res[2]

  def test_jit_realization_and_sampling(self):
    w = Tensor.eye(5)

    @TinyJit
    def foo (x): return w.dot(x).realize()

    arg  = [
        Tensor([1,2,3,4,5]),
        Tensor([1,3,3,4,6]),
        Tensor([1,2,5,4,7]),
        Tensor([0,2,3,1,0]),
    ]

    Y = [foo(e).numpy() for e in arg]

    foo(Tensor([7,7,7,7,7]))
    want = [[1., 2., 3., 4., 5.],
            [1., 3., 3., 4., 6.],
            [1., 2., 5., 4., 7.],
            [0., 2., 3., 1., 0.]]
    np.testing.assert_allclose(want, Y)

  def test_jit_buffer_behavior(self):
    @TinyJit
    def foo(x) -> Tensor: return x.sum().realize()

    result_1 = foo(Tensor([1] * 2))
    result_2 = foo(Tensor([2] * 2))
    result_3 = foo(Tensor([3] * 2))

    # expect the buffer to share underlying buffer
    np.testing.assert_allclose(result_1.numpy(), [2], atol=1e-4, rtol=1e-5)
    np.testing.assert_allclose(result_2.numpy(), [6], atol=1e-4, rtol=1e-5)
    np.testing.assert_allclose(result_3.numpy(), [6], atol=1e-4, rtol=1e-5)

  def test_jit_output_clone(self):
    @TinyJit
    def f(x:Tensor) -> Tensor: return (x + 1).realize()

    f(Tensor([0.0]))
    f(Tensor([0.0]))

    a = f(Tensor([1.0])).clone().realize()
    b = f(Tensor([2.0]))
    assert abs((a - b).item()) > 0.5

  def test_jit_init_empty(self):
    @TinyJit
    def f(x:Tensor) -> Tensor: return (x + 1).realize()

    f(Tensor.empty(1))
    f(Tensor.empty(1))
    # scalar const input is not allowed
    with self.assertRaises(JitError):
      f(Tensor(2.0)).item()
    # self.assertEqual(f(Tensor([2.0])).item(), 1.0) # TODO: wrong output, should be 3.0. currently depends on empty value

  def test_jit_const_input(self):
    @TinyJit
    def f(x:Tensor) -> Tensor: return (x + 1).realize()
    with self.assertRaises(JitError):
      f(Tensor(UOp.const(dtypes.float, 2.0))).item()

  def test_jit_deviceless_compute_input(self):
    @TinyJit
    def f(x:Tensor) -> Tensor: return (x + 1).realize()
    with self.assertRaises(JitError):
      f(Tensor(UOp.const(dtypes.float, 2.0) + UOp.const(dtypes.float, 1.0))).item()

  def test_jit_init_empty_alt(self):
    @TinyJit
    def f(a:Tensor, b:Tensor) -> Tensor: return b.assign(a+1)
    for i in range(4):
      a = Tensor([i])
      b = Tensor.empty_like(a)
      c = f(a, b)
      self.assertEqual(c.item(), i+1)

  def test_jit_lazy_grad_after_replay(self):
    # the lazy .grad created during capture is read outside the JIT, the memory planner must not suballocate its buffers (issue #16571)
    from tinygrad import nn
    def step(conv, x, y):
      out = conv(x.permute(0, 3, 1, 2).contiguous()).relu().flatten(1)
      loss = (out * y).sum(axis=1)  # per-example loss
      loss.sum().backward()
      conv.weight.grad = None
      (loss * 0.5).sum().backward()
      return loss.mean().realize()

    Tensor.manual_seed(42)
    conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
    x, y = Tensor.randn(4, 8, 8, 3).realize(), Tensor.randn(4, 4*8*8).realize()

    step(conv, x, y)
    ref = conv.weight.grad.numpy()
    jit_step = TinyJit(step)
    for _ in range(4):
      jit_step(conv, x, y)
      np.testing.assert_allclose(conv.weight.grad.numpy(), ref, atol=1e-4, rtol=1e-5)

class TestJitPrune(unittest.TestCase):
  def test_simple_prune(self):
    weights = Tensor.rand(16).realize()
    def w2(x) -> Tensor: return (weights*2).contiguous() + x
    w2_noprune = TinyJit(w2)
    w2_prune = TinyJit(w2, prune=True)

    for _ in range(3):
      a = Tensor.rand(16).realize()
      out = w2_noprune(a)
      np.testing.assert_allclose(out.tolist(), [x*2+y for x,y in zip(weights.tolist(), a.tolist())])
    assert_jit_cache_len(w2_noprune, 2)

    for _ in range(3):
      a = Tensor.rand(16).realize()
      out = w2_prune(a)
      np.testing.assert_allclose(out.tolist(), [x*2+y for x,y in zip(weights.tolist(), a.tolist())])
    assert_jit_cache_len(w2_prune, 1)


class TestJitInsideJit(unittest.TestCase):
  def test_jit_jit_error(self):
    @TinyJit
    def f(t): return t + 1

    @TinyJit
    def g(t): return f(t) * 3

    # NOTE: first does not raise
    g(Tensor([1])).realize()
    with self.assertRaisesRegex(RuntimeError, "having TinyJit inside another TinyJit is not supported"):
      g(Tensor([1])).realize()

class TestJitRandom(unittest.TestCase):
  def test_jit_rangeify(self):
    tst = {0:[], 1:[]}
    for r in [0,1]:
      Tensor.manual_seed(1337)
      with Context(JIT=r):
        _ = Tensor.randint(4, high=3)
        # this second one makes the behavior different
        _ = Tensor.randint(4, high=3)
        @TinyJit
        def f(): return Tensor.randint(20, high=5)
        for _ in range(5): tst[r].append(f().tolist())
    for i, (t0, t1) in enumerate(zip(tst[0], tst[1])):
      self.assertListEqual(t0, t1, msg=f"mismatch at list {i}")

if __name__ == '__main__':
  unittest.main()
