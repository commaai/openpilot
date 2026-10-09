import unittest
import numpy as np
from tinygrad import Tensor, function, Device
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.schedule import transform_to_call

def sched_key(t:Tensor): return transform_to_call(UOp.sink(t.uop)).body.key

class TestCall(unittest.TestCase):
  def test_call_plus(self):
    a = Tensor.randn(10, 10)
    b = Tensor.randn(10, 10)
    Tensor.realize(a,b)

    # we define a plus function
    plus_fxn = UOp.param(0, dtypes.float, (10,10)) + UOp.param(1, dtypes.float, (10,10))

    c = Tensor.call(a, b, fxn=plus_fxn)
    np.testing.assert_equal(c.numpy(), (a+b).numpy())

  def test_call_plus_backward(self):
    a = Tensor.ones(10, 10)
    b = Tensor.ones(10, 10)

    (a+b).mean().backward()
    gt_a_grad = a.grad.numpy()
    gt_b_grad = b.grad.numpy()
    a.grad, b.grad = None, None

    # this is the gradient for +
    def grad_fxn(grad:UOp, call:UOp): return (grad, grad)

    # we define a plus function
    plus_fxn = UOp.param(0, dtypes.float, (10,10)) + UOp.param(1, dtypes.float, (10,10))
    c = Tensor.call(a, b, fxn=plus_fxn, grad_fxn=grad_fxn)
    c.mean().backward()

    np.testing.assert_allclose(a.grad.numpy(), gt_a_grad, rtol=1e-5)
    np.testing.assert_allclose(b.grad.numpy(), gt_b_grad, rtol=1e-5)

  def test_call_plus_backward_auto(self):
    a = Tensor.ones(10, 10)
    b = Tensor.ones(10, 10)

    (a+b).mean().backward()
    gt_a_grad = a.grad.numpy()
    gt_b_grad = b.grad.numpy()
    a.grad, b.grad = None, None

    plus_fxn = UOp.param(0, dtypes.float, (10,10)) + UOp.param(1, dtypes.float, (10,10))
    c = Tensor.call(a, b, fxn=plus_fxn)
    c.mean().backward()

    np.testing.assert_allclose(a.grad.numpy(), gt_a_grad, rtol=1e-5)
    np.testing.assert_allclose(b.grad.numpy(), gt_b_grad, rtol=1e-5)

  def test_call_gemm(self):
    M, K, N = 4, 8, 4
    a = Tensor.randn(M, K)
    b = Tensor.randn(K, N)
    Tensor.realize(a, b)
    c = Tensor.call(a, b, fxn=a.as_param(0) @ b.as_param(1))
    np.testing.assert_allclose(c.numpy(), a.numpy() @ b.numpy(), rtol=1e-5, atol=1e-6)

  def test_call_gemm_uop(self):
    M, K, N = 4, 8, 4
    a = Tensor.randn(M, K)
    b = Tensor.randn(K, N)
    Tensor.realize(a, b)

    # we define a gemm function
    x = UOp.param(0, dtypes.float, shape=(M, K))
    y = UOp.param(1, dtypes.float, shape=(K, N))
    c = Tensor.call(a, b, fxn=x@y)

    np.testing.assert_allclose(c.numpy(), a.numpy() @ b.numpy(), rtol=1e-5, atol=1e-6)

  def test_call_complex_backward_auto(self):
    # complex chain: (a*b + a).exp2() * b.reciprocal() - tests mul, add, exp2, reciprocal, param reuse
    a = Tensor.randn(10, 10)
    b = Tensor.randn(10, 10) + 2  # avoid div by zero
    Tensor.realize(a, b)

    ((a*b + a).exp2() * b.reciprocal()).mean().backward()
    gt_a_grad, gt_b_grad = a.grad.numpy(), b.grad.numpy()
    a.grad, b.grad = None, None

    p0, p1 = UOp.param(0, dtypes.float, (10,10)), UOp.param(1, dtypes.float, (10,10))
    complex_fxn = (p0*p1 + p0).exp2() * p1.reciprocal()
    c = Tensor.call(a, b, fxn=complex_fxn)
    c.mean().backward()

    np.testing.assert_allclose(a.grad.numpy(), gt_a_grad, rtol=1e-5)
    np.testing.assert_allclose(b.grad.numpy(), gt_b_grad, rtol=1e-5)

  def test_call_plus_sharded(self):
    devs = ("CPU:0", "CPU:1")
    a = Tensor.ones(10, 10).shard(devs, axis=0)
    b = Tensor.ones(10, 10).shard(devs, axis=0)
    Tensor.realize(a, b)
    c = Tensor.call(a, b, fxn=a.as_param(0) + b.as_param(1))
    np.testing.assert_equal(c.numpy(), 2 * np.ones((10, 10)))

class TestCallSchedule(unittest.TestCase):
  def test_precompile_slice_assign(self):
    @function(precompile=True)
    def f(x:Tensor) -> Tensor: return x * 2 + 1
    a = Tensor.arange(8).float().realize()
    cache = Tensor.zeros(16)
    # the output must land at the slice offset, not at the start of the base buffer
    cache[4:12].assign(f(a)).realize()
    np.testing.assert_equal(cache.numpy(), np.concatenate([np.zeros(4), np.arange(8)*2+1, np.zeros(4)]).astype(np.float32))

  def test_precompile_slice_assign_2d(self):
    @function(precompile=True)
    def f(x:Tensor) -> Tensor: return x + 1
    a = Tensor.arange(8).reshape(2, 4).float().realize()
    big = Tensor.zeros(4, 8)
    big[1:3, 2:6].assign(f(a)).realize()
    ref = np.zeros((4, 8), dtype=np.float32)
    ref[1:3, 2:6] = np.arange(8).reshape(2, 4) + 1
    np.testing.assert_equal(big.numpy(), ref)

  def test_precompile_full_buffer_assign(self):
    @function(precompile=True)
    def f(x:Tensor) -> Tensor: return x * 2 + 1
    a = Tensor.arange(8).float().realize()
    cache = Tensor.zeros(8).realize()
    cache.assign(f(a)).realize()
    np.testing.assert_equal(cache.numpy(), np.arange(8)*2+1)

  def test_reshape_precompile(self):
    a = Tensor.empty(4, 8).realize()
    a = a.reshape(4,4,2).assign(Tensor.empty(4,4,2)).reshape(8,4)
    @function(precompile=True)
    def s(x): return x.sum(axis=0)
    (s(a)*3).realize()

  def test_call_precompiled(self):
    a = Tensor.empty(4, 8)
    @function(precompile=True)
    def s(x): return x*2
    (s(a)*3).realize()

  def test_double_call(self):
    a = Tensor.empty(4, 8)
    @function(precompile=True)
    def s(x): return x*2
    s(s(a)).realize()

  def test_double_call_contiguous(self):
    a = Tensor.empty(4, 8)
    @function(precompile=True)
    def s(x): return x*2
    s(s(a).contiguous()).realize()

  def test_contiguous_call_output_realizes_aliases(self):
    def increment(x:UOp):
      i = UOp.range(x.shape[0], 0)
      return x[i].store(x[i].load() + 1).end(i).sink(arg=KernelInfo(name="increment"))

    for precompile in (False, True):
      for reshape in (False, True):
        with self.subTest(precompile=precompile, reshape=reshape):
          @function(precompile=precompile)
          def f(x:Tensor): return x.custom_kernel(fxn=increment)[0]
          state = Tensor([1., 2.]).realize()
          a = f(state)
          alias = a.reshape(1, 2)
          b = (alias if reshape else a).contiguous().realize()
          self.assertEqual(b.flatten().tolist(), [2., 3.])
          a.realize(alias)
          self.assertEqual(state.tolist(), [2., 3.])
          self.assertIs(a.uop.buffer, b.uop.buffer)
          self.assertIs(alias.uop.buffer, b.uop.buffer)
          b.assign([9., 10.]).realize()
          self.assertEqual(a.tolist(), [9., 10.])
          self.assertEqual(alias.tolist(), [[9., 10.]])

  def test_assign_call_output_to_input(self):
    for precompile in (False, True):
      with self.subTest(precompile=precompile):
        @function(precompile=precompile)
        def f(x:Tensor): return x.flip(0).contiguous()
        a = Tensor.arange(1024).clone().realize()
        a.assign(f(a)).realize()
        self.assertEqual(a.tolist(), list(reversed(range(1024))))

  def test_call_double_gemm(self):
    a = Tensor.randn(4, 8)
    b = Tensor.randn(8, 12)
    c = Tensor.randn(12, 16)
    ref = Tensor.randn(4, 16)
    Tensor.realize(a,b,c,ref)
    @function(precompile=True)
    def gemm(a:Tensor, b:Tensor, c:Tensor) -> Tensor: return (a@b)@c
    out = gemm(a,b,c)
    (out-ref).square().mean().backward()
    out.realize(a.grad, b.grad, c.grad)

  def test_precompile_symbolic_shape(self):
    """precompile with a symbolic-shaped input produces correct values and shape"""
    @function(precompile=True)
    def f(x:Tensor) -> Tensor: return x * 2
    sz = UOp.variable("sz", 1, 8)
    a = Tensor([1., 2., 3., 4., 5., 6., 7., 8.])[:sz.bind(5)]
    out = f(a)
    self.assertIsInstance(out.shape[0], UOp)
    np.testing.assert_allclose(out[:5].numpy(), [2., 4., 6., 8., 10.])

  def test_precompile_symbolic_shape_contiguous(self):
    """precompile with a .contiguous() inside the function body on a symbolic-shaped input"""
    @function(precompile=True)
    def f(x:Tensor) -> Tensor: return (x * 2).contiguous() + 1
    sz = UOp.variable("sz", 1, 8)
    a = Tensor([1., 2., 3., 4., 5., 6., 7., 8.])[:sz.bind(3)]
    out = f(a)
    self.assertIsInstance(out.shape[0], UOp)
    np.testing.assert_allclose(out[:3].numpy(), [3., 5., 7.])

  def test_precompile_symbolic_shape_chain(self):
    """precompiled symbolic result used in downstream ops (tests AFTER has correct symbolic shape)"""
    @function(precompile=True)
    def f(x:Tensor) -> Tensor: return x * 2
    sz = UOp.variable("sz", 1, 8)
    a = Tensor([1., 2., 3., 4., 5., 6., 7., 8.])[:sz.bind(4)]
    out = f(a) + 10  # downstream op on the precompiled result
    self.assertIsInstance(out.shape[0], UOp)
    np.testing.assert_allclose(out[:4].numpy(), [12., 14., 16., 18.])

  def test_precompile_bind_arg(self):
    """precompile with a BIND (scalar variable) as a function argument"""
    @function(precompile=True)
    def f(x:Tensor, scale:UOp) -> Tensor: return x * scale
    v = UOp.variable("scale", 1, 100)
    a = Tensor([1., 2., 3.])
    out = f(a, v.bind(5))
    np.testing.assert_allclose(out.numpy(), [5., 10., 15.])

  def test_precompile_scoped_bind_arg(self):
    @function(precompile=True)
    def f(x:Tensor, scale:UOp) -> Tensor: return x * scale
    a = Tensor.ones(3)
    x = f(a, UOp.variable("scale_a", 1, 100).bind(2))
    y = f(a, UOp.variable("scale_b", 1, 100).bind(3))
    self.assertEqual(sched_key(x), sched_key(y))
    np.testing.assert_equal(x.numpy(), [2, 2, 2])
    np.testing.assert_equal(y.numpy(), [3, 3, 3])

  def test_precompile_nested_scope_collision(self):
    # a precompiled function body gets its own positional p{slot} params; they must not be renumbered when the call is
    # scheduled inside an enclosing realize with a different slot ordering. the store must use this call's Variable
    cache = Tensor.zeros(16)
    @function(precompile=True, allow_implicit=True)
    def store(x:Tensor, sp:UOp) -> Tensor:
      # update a cache at a symbolic offset, like an attention KV cache update
      return Tensor(cache.uop.after(cache[sp:sp+x.shape[0]].uop.store(x.uop)))[:sp+x.shape[0]].sum()
    sp_v, nt_v = UOp.variable("sp", 0, 8), UOp.variable("nt", 1, 8)
    t = Tensor.arange(16).float().realize()
    sp, nt = sp_v.bind(0), nt_v.bind(8)
    store(t[sp:sp+nt].clone().realize(), sp).realize()
    np.testing.assert_equal(cache.numpy()[:8], t[:8].numpy())
    np.testing.assert_equal(cache.numpy()[8:], np.zeros(8))

  def test_precompile_nested(self):
    for precompile in (False, True):
      for devices in (None, ("CPU:0", "CPU:1")):
        with self.subTest(precompile=precompile, devices=devices):
          @function(precompile=True, precompile_backward=True)
          def inner(x:Tensor): return x * 2, x + 3
          @function(precompile=precompile, precompile_backward=True)
          def outer(x:Tensor):
            a, b = inner(x)
            return a + b
          x = Tensor([1., 2., 3., 4.]).realize()
          if devices is not None: x = x.shard(devices, axis=0).realize()
          out = outer(x)
          for call in (u for u in out.uop.toposort() if u.op is Ops.CALL):
            self.assertFalse(any(b.op is Ops.BUFFER for b in call.body.toposort()))
          self.assertEqual(sched_key(out), sched_key(outer(x)))
          out.sum().backward()
          np.testing.assert_equal(out.numpy(), [6., 9., 12., 15.])
          np.testing.assert_equal(x.grad.numpy(), [3., 3., 3., 3.])

  def test_precompile_consumes_call_output(self):
    """a precompiled function consuming the output of a non-precompiled function"""
    @function
    def inner(x:Tensor) -> Tensor: return x * 2
    @function(precompile=True)
    def outer(x:Tensor) -> Tensor: return x + 1
    x = Tensor.arange(8).float().contiguous().realize()
    np.testing.assert_equal(outer(inner(x)).numpy(), np.arange(8, dtype=np.float32) * 2 + 1)

  def test_precompile_symbolic_2d(self):
    """precompile with symbolic shapes in 2D (tests debuf reshape with symbolic PARAM)"""
    @function(precompile=True)
    def f(x:Tensor) -> Tensor: return x * 2 + 1
    sz = UOp.variable("sz", 1, 16)
    a = Tensor.arange(16*4).reshape(16, 4).float().clone()[:sz.bind(5)]
    out = f(a)
    # result shape should have the symbolic dim, not the max
    self.assertIsInstance(out.shape[0], UOp)
    np.testing.assert_allclose(out[:5].numpy(), (np.arange(16*4).reshape(16, 4)[:5] * 2 + 1).astype(np.float32))

  def test_precompile_multi_sharded(self):
    @function(precompile=True)
    def f(x:Tensor) -> Tensor: return x + 1
    devs = ("CPU:0", "CPU:1")
    a = Tensor.arange(8).reshape(4, 2).float().clone().shard(devs, axis=0)
    out = f(a) + 2
    np.testing.assert_allclose(out.numpy(), np.arange(8, dtype=np.float32).reshape(4, 2) + 3)

class TestArgOrder(unittest.TestCase):
  """outputs can appear anywhere in a call's srcs (output_pos): slots are src positions, nothing reorders"""
  def _dev(self, x): return x.device if isinstance(x.device, str) else (x.device or (Device.DEFAULT,))[0]
  def make_intersperse_call(self, x, precompile=False):
    # the output is at position 0, the input (param slot 1) at position 1 in the call's args
    val = UOp.param(1, x.dtype, x.shape, self._dev(x)).reshape(x.shape) * 2
    return UOp.call_with_outputs((val,), x.uop, name='t', output_pos=(0,), precompile=precompile)

  def test_intersperse_returned(self):
    x = Tensor.arange(3, dtype=dtypes.int).realize()
    outs = self.make_intersperse_call(x)
    out = Tensor(outs[0], device=x.device) + 1
    np.testing.assert_equal(out.numpy(), [1, 3, 5])

  def test_outputs_arbitrary_order(self):
    x = Tensor([1.0, 2.0, 3.0])
    y = Tensor([4.0, 5.0, 6.0])
    x.requires_grad = True
    y.requires_grad = True
    x, y = x.realize(), y.realize()
    dev = self._dev(x)
    # args (out0, in0, out1, in1): outputs at positions 0 and 2, input params slotted at their final positions 1 and 3
    p1, p3 = UOp.param(1, x.dtype, x.shape, dev), UOp.param(3, y.dtype, y.shape, dev)
    outs = UOp.call_with_outputs((p1.reshape(x.shape) * 2, p3.reshape(y.shape) + p1.reshape(y.shape)), x.uop, y.uop,
                                 output_pos=(0, 2))
    np.testing.assert_equal(Tensor(outs[0]).numpy(), [2, 4, 6])
    np.testing.assert_equal(Tensor(outs[1]).numpy(), [5, 7, 9])
    # the auto gradient path (no grad_fxn) resolves outputs and gradients positionally at any position
    (Tensor(outs[0]).sum() + Tensor(outs[1]).sum()).backward()
    np.testing.assert_equal(x.grad.numpy(), [3, 3, 3])
    np.testing.assert_equal(y.grad.numpy(), [1, 1, 1])

  def test_intersperse_returned_gradient(self):
    x = Tensor([1.0, 2.0, 3.0]).realize()
    x.requires_grad = True
    p1 = UOp.param(1, dtypes.float, x.shape, self._dev(x))
    val = p1.reshape(x.shape) * p1.reshape(x.shape)
    outs = UOp.call_with_outputs((val,), x.uop, name='t', output_pos=(0,))
    y = Tensor(outs[0], device=x.device)
    y.sum().backward()
    np.testing.assert_equal(x.grad.numpy(), [2, 4, 6])

class TestCallMultiSharded(unittest.TestCase):
  # TODO: multi-output + sharded needs per-device CALL execution, which requires reworking how MULTI propagates through TUPLE bodies
  def test_tuple_sharded(self):
    """multi-output function with sharded input"""
    devs = ("CPU:0", "CPU:1")
    @function
    def f(x:Tensor): return (x + 1, x * 2)
    a = Tensor.arange(8).reshape(4, 2).float().clone().shard(devs, axis=0)
    t1, t2 = f(a)
    ref = np.arange(8, dtype=np.float32).reshape(4, 2)
    np.testing.assert_allclose(t1.numpy(), ref + 1)
    np.testing.assert_allclose(t2.numpy(), ref * 2)

  def test_tuple_sharded_precompile(self):
    """multi-output precompiled function with sharded input"""
    devs = ("CPU:0", "CPU:1")
    @function(precompile=True)
    def f(x:Tensor): return (x + 1, x * 2)
    a = Tensor.arange(8).reshape(4, 2).float().clone().shard(devs, axis=0)
    t1, t2 = f(a)
    ref = np.arange(8, dtype=np.float32).reshape(4, 2)
    np.testing.assert_allclose(t1.numpy(), ref + 1)
    np.testing.assert_allclose(t2.numpy(), ref * 2)

  def test_tuple_sharded_different_axis(self):
    """multi-output function where outputs have different sharding: one reduces on sharded axis, one doesn't"""
    devs = ("CPU:0", "CPU:1")
    @function
    def f(x:Tensor): return (x.sum(axis=0), x.sum(axis=1))
    a = Tensor.arange(8).reshape(4, 2).float().clone().shard(devs, axis=0)
    t1, t2 = f(a)
    ref = np.arange(8, dtype=np.float32).reshape(4, 2)
    np.testing.assert_allclose(t1.numpy(), ref.sum(axis=0))
    np.testing.assert_allclose(t2.numpy(), ref.sum(axis=1))

  def test_tuple_sharded_different_ops(self):
    """multi-output function with different operations per output"""
    devs = ("CPU:0", "CPU:1")
    @function
    def f(x:Tensor, y:Tensor): return (x + y, x * y)
    a = Tensor.arange(8).reshape(4, 2).float().clone().shard(devs, axis=0)
    b = Tensor.arange(8).reshape(4, 2).float().clone().shard(devs, axis=0) + 1
    t1, t2 = f(a, b)
    ref_a = np.arange(8, dtype=np.float32).reshape(4, 2)
    ref_b = ref_a + 1
    np.testing.assert_allclose(t1.numpy(), ref_a + ref_b)
    np.testing.assert_allclose(t2.numpy(), ref_a * ref_b)

  def test_tuple_sharded_mixed_use(self):
    """multi-output sharded results used in further computation"""
    devs = ("CPU:0", "CPU:1")
    @function
    def f(x:Tensor): return (x + 1, x * 2)
    a = Tensor.arange(8).reshape(4, 2).float().clone().shard(devs, axis=0)
    t1, t2 = f(a)
    out = (t1 + t2).sum()
    ref = np.arange(8, dtype=np.float32).reshape(4, 2)
    np.testing.assert_allclose(out.numpy(), ((ref + 1) + (ref * 2)).sum())

  def test_tuple_sharded_outputs_different_axis(self):
    """multi-output function where the two outputs are sharded on different axes"""
    devs = ("CPU:0", "CPU:1")
    @function
    def f(x:Tensor, y:Tensor): return (x + 1, y + 2)
    a = Tensor.arange(8).reshape(4, 2).float().clone().shard(devs, axis=0)
    b = Tensor.arange(8).reshape(4, 2).float().clone().shard(devs, axis=1)
    t1, t2 = f(a, b)
    ref_a = np.arange(8, dtype=np.float32).reshape(4, 2)
    ref_b = np.arange(8, dtype=np.float32).reshape(4, 2)
    np.testing.assert_allclose(t1.numpy(), ref_a + 1)
    np.testing.assert_allclose(t2.numpy(), ref_b + 2)

  def test_call_reduce_sharded(self):
    devs = ("CPU:0", "CPU:1")
    a = Tensor.ones(10, 10).shard(devs, axis=0)
    Tensor.realize(a)
    c = Tensor.call(a, fxn=a.as_param(0).sum(axis=0))
    np.testing.assert_equal(c.numpy(), 10 * np.ones(10))

  def test_call_reduce_sharded_mixed_args(self):
    devs = ("CPU:0", "CPU:1")
    a = Tensor.ones(10, 10).shard(devs, axis=0)
    b = Tensor.ones(10).shard(devs, axis=None)
    Tensor.realize(a, b)
    c = Tensor.call(a, b, fxn=a.as_param(0).sum(axis=0) + b.as_param(1))
    np.testing.assert_equal(c.numpy(), 11 * np.ones(10))

  def test_call_reduce_sharded_backward(self):
    devs = ("CPU:0", "CPU:1")
    a = Tensor.randn(10, 10).shard(devs, axis=0)
    b = Tensor.randn(10, 10).shard(devs, axis=0)
    Tensor.realize(a, b)

    def grad_fxn(grad, call):
      a_arg, b_arg = call.src[1], call.src[2]
      return (grad.expand(a_arg.shape) * b_arg, grad.expand(b_arg.shape) * a_arg)

    body = (a.as_param(0) * b.as_param(1)).sum(axis=0)
    c = Tensor.call(a, b, fxn=body, grad_fxn=grad_fxn)
    c.sum().backward()
    np.testing.assert_allclose(a.grad.numpy(), b.numpy(), rtol=1e-5)
    np.testing.assert_allclose(b.grad.numpy(), a.numpy(), rtol=1e-5)

  def test_symbolic_reshape_shard_axis(self):
    toks = UOp.variable("toks", 1, 2).bind(2)
    devs = ("CPU:0", "CPU:1")
    x = Tensor(np.arange(16, dtype=np.float32).reshape(1, 2, 8)).shard(devs, axis=2).realize()
    @function
    def f(x:Tensor) -> Tensor: return x.reshape(1, x.shape[1], 2, 4)
    out = f(x[:, :toks]).realize()
    self.assertEqual(out.uop.axis, 2)
    np.testing.assert_equal(out[:1, :2].to(devs[0]).numpy(), np.arange(16, dtype=np.float32).reshape(1, 2, 2, 4))

if __name__ == '__main__':
  unittest.main()
