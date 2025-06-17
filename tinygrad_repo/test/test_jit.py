#!/usr/bin/env python
import unittest, functools
import numpy as np

from hypothesis import given, settings, strategies as strat
from test.helpers import assert_jit_cache_len, not_support_multi_device, REAL_DEV
from tinygrad.tensor import Tensor
from tinygrad.engine.jit import TinyJit
from tinygrad.device import Device
from tinygrad.helpers import Context, JIT, GlobalCounters
from tinygrad.dtype import dtypes
from extra.models.unet import ResBlock

def _simple_test(add, extract=lambda x: x, N=10):
  for _ in range(5):
    a = Tensor.randn(N, N)
    b = Tensor.randn(N, N)
    c = add(a, b)
    np.testing.assert_allclose(extract(c).numpy(), a.numpy()+b.numpy(), atol=1e-4, rtol=1e-5)
  assert_jit_cache_len(add, 1)

class TestJit(unittest.TestCase):

  @settings(deadline=2e4)
  @unittest.skipUnless(REAL_DEV in ["LLVM", "CPU"], f"no support on {REAL_DEV}")
  @given(strat.sampled_from([Tensor.exp2, Tensor.log2, Tensor.sin]))
  def test_approx_jit_timeout(self, op):
    with Context(TRANSCENDENTAL=2):
      model = [ResBlock(16, 24, 16) for _ in range(4)]
      @TinyJit
      def fw_approx(t, t2):
        for l in model: t = l(t, t2)
        return op(t).realize()
      fw_approx(Tensor.empty(4, 16, 8, 8), Tensor.empty(1, 24))

  def test_simple_jit(self):
    @TinyJit
    def add(a, b): return (a+b).realize()
    _simple_test(add)

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
    with self.assertRaises(AssertionError):
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
    with self.assertRaises(AssertionError):
      add(a, bad)

  def test_jit_shape_views_mismatch(self):
    @TinyJit
    def add(a): return (a+1).realize()
    with self.assertRaises(AssertionError):
      for i in range(1,5):
        # a has an offset that the kernel doesn't know about
        a = Tensor.randn(10, 10).realize()[:, i:i+2]
        add(a)

  def test_jit_duplicate_fail(self):
    # the jit doesn't support duplicate arguments
    @TinyJit
    def add(a, b): return (a+b).realize()
    a = Tensor.randn(10, 10)
    with self.assertRaises(AssertionError):
      add(a, a)

  def test_jit_assign(self, dtype=dtypes.float32):
    @TinyJit
    def add(a):
      a += 1
      a.realize()
    a = Tensor.zeros(1, dtype=dtype).contiguous().realize()
    for _ in range(5): add(a)
    self.assertEqual(a.item(), 5)

  def test_jit_assign_int8(self): self.test_jit_assign(dtypes.int8)

  def test_kwargs_jit(self):
    @TinyJit
    def add_kwargs(first, second): return (first+second).realize()
    for _ in range(5):
      a = Tensor.randn(10, 10)
      b = Tensor.randn(10, 10)
      c = add_kwargs(first=a, second=b)
      np.testing.assert_allclose(c.numpy(), a.numpy()+b.numpy(), atol=1e-4, rtol=1e-5)
    assert_jit_cache_len(add_kwargs, 1)

  def test_reorder_kwargs_jit(self):
    @TinyJit
    def add_kwargs(first, second): return (first/second).realize()
    for _ in range(2):
      a = Tensor.randn(10, 10)
      b = Tensor.randn(10, 10)
      c = add_kwargs(second=b, first=a)
      np.testing.assert_allclose(c.numpy(), a.numpy()/b.numpy(), atol=1e-4, rtol=1e-5)
    for _ in range(2):
      a = Tensor.randn(10, 10)
      b = Tensor.randn(10, 10)
      c = add_kwargs(first=a, second=b)
      np.testing.assert_allclose(c.numpy(), a.numpy()/b.numpy(), atol=1e-4, rtol=1e-5)
    assert_jit_cache_len(add_kwargs, 1)

  def test_array_jit(self):
    @TinyJit
    def add_array(a, arr): return (a+arr[0]).realize()
    for i in range(5):
      a = Tensor.randn(10, 10)
      b = Tensor.randn(10, 10)
      a.realize(), b.realize()
      c = add_array(a, [b])
      if i >= 2:
        # should fail once jitted since jit can't handle arrays
        np.testing.assert_allclose(np.any(np.not_equal(c.numpy(),a.numpy()+b.numpy())), True, atol=1e-4, rtol=1e-5)
      else:
        np.testing.assert_allclose(c.numpy(), a.numpy()+b.numpy(), atol=1e-4, rtol=1e-5)
    assert_jit_cache_len(add_array, 1)

  def test_jit_copyin(self):
    @TinyJit
    def f(a):
      return a + Tensor([1,2,3])
    for _ in range(5):
      b = Tensor.randn(3)
      c = f(b)
      np.testing.assert_allclose(c.numpy(), b.numpy()+[1,2,3], atol=1e-4, rtol=1e-5)

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
    output1, output2 = [], []
    expect1, expect2 = [], []
    for i in range(5):
      a = Tensor.randn(10, 10)
      b = Tensor.randn(10, 10)
      o1, o2 = f(a, b, i)
      output1.append(o1.numpy().copy())
      output2.append(o2)
      expect1.append(a.numpy().copy()+b.numpy().copy())
      expect2.append(i)
    np.testing.assert_allclose(output1, expect1, atol=1e-4, rtol=1e-5)
    # the jit only works with Tensor outputs
    assert output2 != expect2
    assert_jit_cache_len(f, 1)

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

  @unittest.expectedFailure  # TODO: fix
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
    assert with_jit == without_jit, "Jit rand produced different values from no jit."

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

  #@unittest.expectedFailure # requires contiguous folding
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

  def test_jit_batch_split(self):
    if Device[Device.DEFAULT].graph is None or JIT >= 2: raise unittest.SkipTest("only test graphs")

    # Create long jit with 83 kernels.
    def f(a, b, c, d, e):
      for _ in range(80):
        a = (a+b).realize()
      y = (a*c).realize()
      z = (y*d).realize()
      w = (z*e)
      return w.realize()

    a = Tensor.randn(10, 10).realize()
    b = Tensor.randn(10, 10).realize()
    c = Tensor.randn(10, 10).realize()
    d = Tensor.randn(10, 10).realize()
    e = Tensor.randn(10, 10).realize()

    jf = TinyJit(f)
    prev = None
    for _ in range(5):
      o = jf(a, b, c, d, e).numpy()
      if prev is not None: np.testing.assert_allclose(o, prev, atol=1e-4, rtol=1e-5)
      prev = o

    graph_t = Device[Device.DEFAULT].graph.func if isinstance(Device[Device.DEFAULT].graph, functools.partial) else Device[Device.DEFAULT].graph
    # Checking that 2 graphs are inited.
    assert isinstance(jf.jit_cache[0].prg, graph_t)
    assert isinstance(jf.jit_cache[1].prg, graph_t)

  def test_jit_const_inputs(self):
    @TinyJit
    def g(x,y,z): return (x+y+z).realize()
    for i in range(5):
      np.testing.assert_equal(g(Tensor([i]*3), Tensor.ones(3), Tensor.zeros(3)).numpy(), np.array([i+1]*3))

  def test_jitted_clone(self):
    def f(a): return a.clone().realize()
    jf = TinyJit(f)
    for _ in range(5):
      a = Tensor.randn(10, 10, device=Device.DEFAULT).realize()
      ja = jf(a)
      np.testing.assert_allclose(a.numpy(), ja.numpy(), atol=1e-4, rtol=1e-5)

  @unittest.skipIf(not_support_multi_device(), "no multi")
  def test_jitted_transfers(self):
    d0, d1 = f"{Device.DEFAULT}:0", f"{Device.DEFAULT}:1"

    def f(a, b):
      x = a.to(d1)
      y = b.to(d1)
      return x.realize(), y.realize()

    jf = TinyJit(f)
    for _ in range(5):
      a = Tensor.randn(10, 10, device=d0).realize()
      b = Tensor.randn(10, 10, device=d0).realize()
      xc, yc = jf(a, b)
      np.testing.assert_allclose(a.numpy(), xc.numpy(), atol=1e-4, rtol=1e-5)
      np.testing.assert_allclose(b.numpy(), yc.numpy(), atol=1e-4, rtol=1e-5)

  def test_jit_several_devs(self):
    d0, d1 = f"{Device.DEFAULT}:0", "CPU"

    def f(a, b):
      x = a.to(d0).realize()
      y = b.to(d0).realize()
      return x+y.realize(), x*y.realize()

    jf = TinyJit(f)
    for _ in range(5):
      a = Tensor.randn(10, 10, device=d1).realize()
      b = Tensor.randn(10, 10, device=d1).realize()
      zc, wc = jf(a, b)
      np.testing.assert_allclose((a.numpy()+b.numpy()), zc.numpy(), atol=1e-4, rtol=1e-5)
      np.testing.assert_allclose((a.numpy()*b.numpy()), wc.numpy(), atol=1e-4, rtol=1e-5)

  @unittest.skipIf(not_support_multi_device(), "no multi")
  def test_jitted_view(self):
    d0, d1 = f"{Device.DEFAULT}:0", f"{Device.DEFAULT}:1"

    def f(a):
      x1 = a.sum(axis=(1,))
      x = (x1 + 5).bitcast(dtypes.int32)
      y = x.to(d1)
      return y.realize()

    jf = TinyJit(f)
    for _ in range(5):
      a = Tensor.randn(10, 1000, device=d0).realize()
      xc = jf(a)
      np.testing.assert_allclose((a.numpy().sum(axis=(1,)) + 5).view(np.int32), xc.numpy(), atol=1e-4, rtol=5e-5)

  def test_jit_output_clone(self):
    @TinyJit
    def f(x:Tensor) -> Tensor: return (x + 1).realize()

    f(Tensor([0.0]))
    f(Tensor([0.0]))

    a = f(Tensor([1.0])).clone().realize()
    b = f(Tensor([2.0]))
    assert abs((a - b).item()) > 0.5

@unittest.skip("Pending multioutput implementation #3607")
class TestMultioutputJit(unittest.TestCase):
  def _test(self, f):
    for _ in range(5):
      a, b = Tensor.randn(10, 10), Tensor.randn(10, 10)
      out0, out1, out2 = f(a, b)
      np.testing.assert_allclose(out0.numpy(), a.numpy()+b.numpy(), atol=1e-4, rtol=1e-5)
      np.testing.assert_allclose(out1.numpy(), a.numpy()-b.numpy(), atol=1e-4, rtol=1e-5)
      np.testing.assert_allclose(out2.numpy(), a.numpy()*b.numpy(), atol=1e-4, rtol=1e-5)

  def test_jit_multioutput_realize(self):
    @TinyJit
    def fxn(a, b): return (a+b).realize(), (a-b).realize(), (a*b).realize()
    self._test(fxn)
    assert_jit_cache_len(fxn, 3)

  def test_jit_multioutput_norealize(self):
    @TinyJit
    def fxn(a, b): return a+b, a-b, a*b
    self._test(fxn)
    assert_jit_cache_len(fxn, 1)

  def test_jit_multioutput_mix(self):
    @TinyJit
    def fxn(a, b): return a+b, a-b, (a*b).realize()
    self._test(fxn)
    assert_jit_cache_len(fxn, 2)

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

class TestCopyInsideJit(unittest.TestCase):
  def test_copy_inside_jit(self):
    @TinyJit
    def add(x,y) -> Tensor: return x.to(Device.DEFAULT)+y
    for _ in range(5):
      # create a Tensor on CPU
      a = Tensor.rand(16,16,device="CPU").realize()
      b = Tensor.rand(16,16).realize()
      out = add(a,b)
      np.testing.assert_allclose(out.flatten().tolist(), [x+y for x,y in zip(a.flatten().tolist(), b.flatten().tolist())])

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
    assert len(w2_noprune.captured.jit_cache) == 2

    for _ in range(3):
      a = Tensor.rand(16).realize()
      out = w2_prune(a)
      np.testing.assert_allclose(out.tolist(), [x*2+y for x,y in zip(weights.tolist(), a.tolist())])
    assert len(w2_prune.captured.jit_cache) == 1

  def test_prune_w_copy_correct(self):
    weights = Tensor.rand(16).realize()
    def w2(x) -> Tensor: return (weights*2).contiguous() + x.to(Device.DEFAULT)
    w2_noprune = TinyJit(w2)
    w2_prune = TinyJit(w2, prune=True)

    for _ in range(3):
      a = Tensor.rand(16, device="CPU").realize()
      out = w2_noprune(a)
      np.testing.assert_allclose(out.tolist(), [x*2+y for x,y in zip(weights.tolist(), a.tolist())])

    for _ in range(3):
      a = Tensor.rand(16, device="CPU").realize()
      out = w2_prune(a)
      np.testing.assert_allclose(out.tolist(), [x*2+y for x,y in zip(weights.tolist(), a.tolist())])

  def test_prune_w_independent_copy_correct(self):
    weights = Tensor.rand(16, device="CPU").realize()
    def w2(x) -> Tensor: return (weights*2).contiguous().to(Device.DEFAULT) + x
    w2_noprune = TinyJit(w2)
    w2_prune = TinyJit(w2, prune=True)

    for _ in range(3):
      a = Tensor.rand(16).realize()
      out = w2_noprune(a)
      np.testing.assert_allclose(out.tolist(), [x*2+y for x,y in zip(weights.tolist(), a.tolist())])

    for _ in range(3):
      a = Tensor.rand(16).realize()
      out = w2_prune(a)
      np.testing.assert_allclose(out.tolist(), [x*2+y for x,y in zip(weights.tolist(), a.tolist())])

    assert len(w2_prune.captured.jit_cache) == 1, "prune should have removed the copy"

class TestJitFree(unittest.TestCase):
  def test_free_intermediates(self):
    ext_tensor = Tensor([1,24,23,45,1])
    @TinyJit
    def fxn(x:Tensor):
      out = (x*2+ext_tensor).reshape(5,1).expand(5, 100).contiguous()
      return out.sum()
    for i in range(5):
      out = fxn(Tensor([i,1,2,3,4]))
      self.assertEqual(out.item(), 11400+200*i)
    pre_free = GlobalCounters.mem_used
    fxn.captured.free_intermediates()
    savings_after_free = pre_free - GlobalCounters.mem_used

    # Different allocator implementations have different savings.
    expected_savings = 8196 if hasattr(Device[Device.DEFAULT].allocator, '_offset') else 2024

    self.assertEqual(savings_after_free, expected_savings)
    out = fxn(Tensor([11,1,2,3,4]))
    self.assertEqual(out.item(), 13600)

    # Try one more time...
    pre_free = GlobalCounters.mem_used
    fxn.captured.free_intermediates()
    fxn.captured.free_intermediates() # 2nd time to validate
    savings_after_free = pre_free - GlobalCounters.mem_used

    self.assertEqual(savings_after_free, expected_savings)
    out = fxn(Tensor([11,1,2,3,4]))
    self.assertEqual(out.item(), 13600)

  def test_updated_not_freed(self):
    x = Tensor([1]).realize()
    @TinyJit
    def fxn(y):
      nonlocal x
      x += y
      return x
    for _ in range(5): fxn(Tensor([1]))
    self.assertEqual(x.item(), 6)
    pre_free = GlobalCounters.mem_used
    fxn.captured.free_intermediates()
    savings_after_free = pre_free - GlobalCounters.mem_used
    self.assertEqual(savings_after_free, 0)
    fxn(Tensor([2]))
    self.assertEqual(x.item(), 8)

  def test_replan_buffers_memory_layout(self):
    if not hasattr(Device[Device.DEFAULT].allocator, '_offset'): raise unittest.SkipTest("replan_buffers_memory_layout useless")

    ext_tensor = Tensor([1,24,23,45,1])
    ext_tensor_2 = Tensor([2,2,2,2,2])
    @TinyJit
    def fxn(x:Tensor):
      out = (x*ext_tensor_2+ext_tensor).reshape(5,1).expand(5, 100).contiguous()
      return out.sum()
    for i in range(5):
      out = fxn(Tensor([i,1,2,3,4]))
      self.assertEqual(out.item(), 11400+200*i)
    assert len(set([b.base for item in fxn.captured.jit_cache for b in item.bufs if b is not None])) == 4
    fxn.captured.replan_buffers_memory_layout()
    assert len(set([b.base for item in fxn.captured.jit_cache for b in item.bufs if b is not None])) == 2

    out = fxn(Tensor([11,1,2,3,4]))
    self.assertEqual(out.item(), 13600)

if __name__ == '__main__':
  unittest.main()
