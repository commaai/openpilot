#!/usr/bin/env python
import unittest
import numpy as np

from test.helpers import assert_jit_cache_len, call_is_graph, not_support_multi_device, needs_second_gpu
from test.unit.test_jit import _simple_test
from tinygrad import Tensor, Variable, TinyJit, Device, dtypes
from tinygrad.engine.jit import graph_class
from tinygrad.helpers import JIT, DEV, GlobalCounters
from tinygrad.uop.ops import Ops
from tinygrad.renderer.isa.x86 import X86Renderer

class TestJit(unittest.TestCase):
  def test_simple_jit(self):
    @TinyJit
    def add(a, b): return (a+b).realize()
    _simple_test(add)

  @unittest.skipUnless(Device.DEFAULT == "CPU", "core_id is a CPU runtimevar")
  def test_hcq_core_id_runtimevar_merge(self):
    N = 262144
    @TinyJit
    def f(x, st):
      y = (x + 1).contiguous().realize()
      z = x.shrink(((st, st + N),)).contiguous().realize()
      return y, z
    x = Tensor.arange(2*N).clone().realize()
    for _ in range(3): y, z = f(x, Variable("a", 0, N).bind(0))
    self.assertEqual(y.shape, (2*N,))
    self.assertEqual(z.shape, (N,))

  def test_jit_input_view(self):
    @TinyJit
    def f(x): return (x[2:5].contiguous() + 1).realize()
    for i in range(5):
      x = (Tensor.arange(10).float() + i * 10).clone().realize()
      np.testing.assert_allclose(f(x).numpy(), x.numpy()[2:5] + 1)

  @unittest.skipIf(isinstance(Device[Device.DEFAULT].renderer, X86Renderer), "estimates are wrong for x86")
  def test_global_counters_jit(self):
    @TinyJit
    def f(a, b):
      c = (a + b).realize()
      d = (c * 2).realize()
      return (d - a).realize()
    a, b = Tensor.randn(64, 64).realize(), Tensor.randn(64, 64).realize()
    for _ in range(4):
      GlobalCounters.reset()
      f(a, b)
      Device[a.device].synchronize()
      self.assertGreater(GlobalCounters.global_mem, 0)
      self.assertGreater(GlobalCounters.global_ops, 0)

  def test_jit_assign(self, dtype=dtypes.float32):
    @TinyJit
    def add(a):
      a += 1
      a.realize()
    a = Tensor.zeros(1, dtype=dtype).contiguous().realize()
    for _ in range(5): add(a)
    self.assertEqual(a.item(), 5)

  def test_jit_assign_int8(self): self.test_jit_assign(dtypes.int8)

  def test_jit_copyin(self):
    @TinyJit
    def f(a):
      return a + Tensor([1,2,3])
    for _ in range(5):
      b = Tensor.randn(3)
      c = f(b)
      np.testing.assert_allclose(c.numpy(), b.numpy()+[1,2,3], atol=1e-4, rtol=1e-5)

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

    # Checking that 2 graphs are inited.
    assert len(jf.captured.linear.src) == 2
    for si in jf.captured.linear.src:
      assert call_is_graph(si)

  def test_jitted_clone(self):
    def f(a): return a.clone().realize()
    jf = TinyJit(f)
    for _ in range(5):
      a = Tensor.randn(10, 10, device=Device.DEFAULT).realize()
      ja = jf(a)
      np.testing.assert_allclose(a.numpy(), ja.numpy(), atol=1e-4, rtol=1e-5)

  @needs_second_gpu
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

  @needs_second_gpu
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

    assert_jit_cache_len(w2_prune, 1)

class TestJitFree(unittest.TestCase):
  def test_free_intermediates(self):
    ext_tensor = Tensor([1,24,23,45,1])
    @TinyJit
    def fxn(x:Tensor):
      t1 = (x * 2).contiguous().realize()
      t2 = (t1 + ext_tensor).contiguous().realize()
      out = (t2.sum()).contiguous().realize()
      return out
    for i in range(5):
      out = fxn(inp:=Tensor([i,1,2,3,4]))
      self.assertEqual(out.item(), 114+2*i)
    pre_free = GlobalCounters.mem_used
    fxn.captured.free_intermediates()
    savings_after_free = pre_free - GlobalCounters.mem_used

    expected_savings = (len(inp) * inp.dtype.itemsize * 2) + dtypes.float32.itemsize # (t1 and t2) + out

    self.assertGreaterEqual(savings_after_free, expected_savings)
    out = fxn(Tensor([11,1,2,3,4]))
    self.assertEqual(out.item(), 136)

    # Try one more time...
    pre_free = GlobalCounters.mem_used
    fxn.captured.free_intermediates()
    fxn.captured.free_intermediates() # 2nd time to validate
    savings_after_free = pre_free - GlobalCounters.mem_used

    self.assertGreaterEqual(savings_after_free, expected_savings)
    out = fxn(Tensor([11,1,2,3,4]))
    self.assertEqual(out.item(), 136)

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

class TestJitGraphSplit(unittest.TestCase):
  def compute(self, device, inp):
    assert inp.device == device, f"Input device {inp.device} does not match expected {device}"
    return (inp + 1.0).contiguous().realize()

  def copy(self, device, to_device, inp):
    assert inp.device == device, f"Input device {inp.device} does not match expected {device}"
    return inp.to(to_device).realize()

  def expect(self, f, *args, graph=None, multigraph=None, hcqgraph=None):
    def _numpies(tpl): return tpl.numpy() if tpl.__class__ is Tensor else tuple([t.numpy() for t in tpl])

    expected = _numpies(f(*args))
    for i in range(4):
      res = _numpies(f(*args))
      np.testing.assert_allclose(res, expected, atol=1e-4, rtol=1e-5)

    dev = Device[Device.DEFAULT]
    graph_t = graph_class(dev)
    if graph_t is None: return

    got = f.captured.linear.src
    from tinygrad.runtime.graph.hcq import HCQGraph
    from tinygrad.engine.jit import MultiGraphRunner
    if graph_t is HCQGraph:
      validate = hcqgraph
    elif issubclass(graph_t, MultiGraphRunner):
      validate = multigraph
    else:
      validate = graph

    assert len(got) == len(validate), f"Expected {len(validate)} operations, got {len(got)}"
    for expected, si in zip(validate, got):
      ast = si.src[0]
      if expected["type"] == "graph":
        assert call_is_graph(si), f"Expected graph, got {ast.op}"
        inner_cnt = len(ast.src[0].src)
        assert inner_cnt == expected["cnt"], f"Expected {expected['cnt']} operations in graph, got {inner_cnt}"
      elif expected["type"] == "comp":
        assert ast.op in (Ops.SINK, Ops.PROGRAM), f"Expected kernel, got {ast.op}"
      elif expected["type"] in ("copy", "xfer"):
        assert ast.op is Ops.COPY, f"Expected COPY, got {ast.op}"

  def ji_graph(self, cnt): return {"type": "graph", "cnt": cnt}
  def ji_comp(self): return {"type": "comp"}
  def ji_copy(self): return {"type": "copy"}
  def ji_xfer(self): return {"type": "xfer"}

  def test_jit_split_simple(self):
    @TinyJit
    def f(inp):
      op0 = self.compute(Device.DEFAULT, inp)
      op1 = self.compute(Device.DEFAULT, op0)
      op2 = self.compute(Device.DEFAULT, op1)
      return op2

    inp = Tensor.randn(10, 10, device=Device.DEFAULT).realize()
    self.expect(f, inp,
      graph=[self.ji_graph(3)],
      multigraph=[self.ji_graph(3)],
      hcqgraph=[self.ji_graph(3)])

  def test_jit_cpu_simple(self):
    if Device.DEFAULT == "CPU": raise unittest.SkipTest("CPU is not a valid default device for this test")

    @TinyJit
    def f(inp, inp_cpu):
      op0 = self.compute(Device.DEFAULT, inp)
      op1 = self.compute(Device.DEFAULT, op0)
      op2 = self.compute("CPU", inp_cpu)
      op3 = self.compute(Device.DEFAULT, op1)
      return op2, op3

    inp = Tensor.randn(10, 10, device=Device.DEFAULT).realize()
    inp_cpu = Tensor.randn(10, 10, device="CPU").realize()
    self.expect(f, inp, inp_cpu,
      graph=[self.ji_graph(2), self.ji_comp(), self.ji_comp()],
      multigraph=[self.ji_graph(2), self.ji_comp(), self.ji_comp()],
      hcqgraph=[self.ji_graph(4)])

  def test_jit_cpu_several(self):
    if Device.DEFAULT == "CPU": raise unittest.SkipTest("CPU is not a valid default device for this test")

    @TinyJit
    def f(inp, inp_cpu):
      op0 = self.compute(Device.DEFAULT, inp)
      op1 = self.compute(Device.DEFAULT, op0)
      op2 = self.compute("CPU", inp_cpu)
      op3 = self.compute("CPU", op2)
      op4 = self.compute(Device.DEFAULT, op1)
      return op3, op4

    inp = Tensor.randn(10, 10, device=Device.DEFAULT).realize()
    inp_cpu = Tensor.randn(10, 10, device="CPU").realize()
    self.expect(f, inp, inp_cpu,
      graph=[self.ji_graph(2), self.ji_graph(2), self.ji_comp()],
      multigraph=[self.ji_graph(2), self.ji_graph(2), self.ji_comp()],
      hcqgraph=[self.ji_graph(5)])

  def test_jit_multidev(self):
    if Device.DEFAULT == "CPU": raise unittest.SkipTest("CPU is not a valid default device for this test")

    try: Device[f"{Device.DEFAULT}:1"]
    except Exception: raise unittest.SkipTest("no multidevice")

    @TinyJit
    def f(inp, inp_d1):
      op0 = self.compute(Device.DEFAULT, inp)
      op1 = self.compute(Device.DEFAULT, op0)
      op2 = self.compute(f"{Device.DEFAULT}:1", inp_d1)
      op3 = self.compute(f"{Device.DEFAULT}:1", op2)
      op4 = self.compute(Device.DEFAULT, op1)
      return op3, op4

    inp = Tensor.randn(10, 10, device=Device.DEFAULT).realize()
    inp_d1 = Tensor.randn(10, 10, device=f"{Device.DEFAULT}:1").realize()
    self.expect(f, inp, inp_d1,
      graph=[self.ji_graph(2), self.ji_graph(2), self.ji_comp()],
      multigraph=[self.ji_graph(5)],
      hcqgraph=[self.ji_graph(5)])

  def test_jit_multidev_xfer(self):
    if Device.DEFAULT in {"CPU"}: raise unittest.SkipTest("CPU is not a valid default device for this test (zero-copies)")
    if Device.DEFAULT == "METAL": raise unittest.SkipTest("Metal is flaky, with multidevice (same as metal llama 4gpu?)")

    try: Device[f"{Device.DEFAULT}:1"]
    except Exception: raise unittest.SkipTest("no multidevice")

    @TinyJit
    def f(inp, inp_d1):
      op0 = self.compute(Device.DEFAULT, inp)
      op1 = self.compute(Device.DEFAULT, op0)
      op2 = self.compute(f"{Device.DEFAULT}:1", inp_d1)
      op3 = self.copy(f"{Device.DEFAULT}:1", Device.DEFAULT, op2)
      op4 = self.compute(f"{Device.DEFAULT}:1", op2)
      op5 = self.compute(Device.DEFAULT, op3)
      return op1, op4, op5

    inp = Tensor.randn(10, 10, device=Device.DEFAULT).realize()
    inp_d1 = Tensor.randn(10, 10, device=f"{Device.DEFAULT}:1").realize()
    self.expect(f, inp, inp_d1,
      graph=[self.ji_graph(2), self.ji_comp(), self.ji_xfer(), self.ji_comp(), self.ji_comp()],
      multigraph=[self.ji_graph(6)],
      hcqgraph=[self.ji_graph(6)])

  @unittest.skip("this fails if you don't have SDMA or are using AMD_DISABLE_SDMA=1")
  @unittest.skipIf(DEV.interface.startswith("MOCK"), "MockGPU does not support parallel copies")
  def test_jit_multidev_copy(self):
    if Device.DEFAULT in {"CPU"}: raise unittest.SkipTest("CPU/LLVM is not a valid default device for this test (zero-copies)")

    @TinyJit
    def f(inp):
      op0 = self.compute(Device.DEFAULT, inp)
      op1 = self.compute(Device.DEFAULT, op0)
      op2 = self.copy(Device.DEFAULT, "CPU", op1)
      op3 = self.compute("CPU", op2)
      return op3

    inp = Tensor.randn(10, 10, device=Device.DEFAULT).realize()
    self.expect(f, inp,
      graph=[self.ji_graph(2), self.ji_copy(), self.ji_comp()],
      multigraph=[self.ji_graph(2), self.ji_copy(), self.ji_comp()],
      hcqgraph=[self.ji_graph(4)])

if __name__ == '__main__':
  unittest.main()
