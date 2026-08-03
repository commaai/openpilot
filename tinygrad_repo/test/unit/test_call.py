import unittest
import numpy as np
from tinygrad import Tensor, function
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import UOp, Ops

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

class TestCallShape(unittest.TestCase):
  def test_call_shape_int(self):
    # fixed-shape function: shape passes through unchanged
    @function
    def f(x:Tensor) -> Tensor: return x * 2
    self.assertEqual(f(Tensor.empty(4, 8)).shape, (4, 8))

  def test_call_shape_param_substitution(self):
    # symbolic shape dimension is substituted: inner PARAM replaced with the BIND arg
    @function
    def f(x:Tensor) -> Tensor: return x * 2
    sz = UOp.variable("sz", 1, 8)
    shape = f(Tensor.empty(8)[:sz.bind(5)]).shape
    # the PARAM should be gone, replaced with the BIND from the call arg
    self.assertIsInstance(shape[0], UOp)
    self.assertNotEqual(shape[0].op, Ops.PARAM)
    self.assertEqual(shape[0], sz.bind(5))

  def test_call_shape_expr_substitution(self):
    # expression containing PARAMs in shape gets fully substituted
    @function
    def f(x:Tensor) -> Tensor: return x + 1
    sz = UOp.variable("sz", 1, 10)
    shape = f(Tensor.empty(10, 4)[:sz.bind(3)]).shape
    self.assertIsInstance(shape[0], UOp)
    self.assertNotEqual(shape[0].op, Ops.PARAM)
    self.assertEqual(shape[1], 4)

  def test_call_shape_no_param_passthrough(self):
    # a non-PARAM UOp shape element passes through unchanged
    @function
    def f(x:Tensor) -> Tensor: return x * 3
    sz = UOp.variable("sz", 1, 8)
    shape = f(Tensor.empty(8)[:sz.bind(5)]).shape
    self.assertEqual(shape[0], sz.bind(5))

class TestCallSchedule(unittest.TestCase):
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

  def test_precompile_schedule_cache_hit(self):
    """two instances of the same @function should produce identical function body keys (schedule cache hit)"""
    @function(precompile=True)
    def f(x:Tensor) -> Tensor: return x + Tensor.full(x.shape, -1.0)
    a = Tensor.empty(4, 8)
    b = Tensor.empty(4, 8)
    r0, r1 = f(a), f(b)
    # find the FUNCTION nodes
    c0 = next(u for u in r0.uop.toposort() if u.op is Ops.FUNCTION)
    c1 = next(u for u in r1.uop.toposort() if u.op is Ops.FUNCTION)
    # the function bodies (src[0]) should have identical keys
    self.assertEqual(c0.src[0].key, c1.src[0].key)

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

if __name__ == '__main__':
  unittest.main()
