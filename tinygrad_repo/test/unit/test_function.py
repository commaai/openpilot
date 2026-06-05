import numpy as np
import unittest
from tinygrad.function import function
from tinygrad import Tensor, GlobalCounters
from tinygrad.uop.ops import UOp, Ops, KernelInfo

class TestFunction(unittest.TestCase):
  def test_simple(self):
    @function
    def f(a:Tensor, b:Tensor) -> Tensor: return a+b

    a = Tensor([1,2,3])
    b = Tensor([4,5,6])
    np.testing.assert_equal(f(a,b).numpy(), [5,7,9])

  def test_simple_same(self):
    @function
    def f(a:Tensor, b:Tensor) -> Tensor: return a+b

    a = Tensor([1,2,3])
    np.testing.assert_equal(f(a,a).numpy(), [2,4,6])

  def test_implicit(self):
    inp = Tensor([7,8,9])
    @function(allow_implicit=True)
    def f(a:Tensor, b:Tensor) -> Tensor: return a+b+inp

    a = Tensor([1,2,3])
    b = Tensor([4,5,6])
    np.testing.assert_equal(f(a,b).numpy(), [12,15,18])

  def test_implicit_same_as_input(self):
    inp = Tensor([7,8,9])
    @function(allow_implicit=True)
    def f(a:Tensor, b:Tensor) -> Tensor: return a+b+inp

    a = Tensor([1,2,3])
    np.testing.assert_equal(f(a, inp).numpy(), [15,18,21])

  def test_implicit_2(self):
    inp = Tensor([7,8,9])
    @function(allow_implicit=True)
    def f(a:Tensor, b:Tensor) -> Tensor:
      return a+b+inp
    inp2 = Tensor([7,8,10])
    @function(allow_implicit=True)
    def g(a:Tensor, b:Tensor) -> Tensor:
      return a+b+inp2

    a = Tensor([1,2,3])
    b = Tensor([4,5,6])
    c = f(a,b)
    d = g(a,b)
    c.realize(d)
    np.testing.assert_equal(c.numpy(), [12,15,18])
    np.testing.assert_equal(d.numpy(), [12,15,19])

  def test_implicit_unrealized(self):
    inp = Tensor([1,2,3]) + Tensor([4,5,6])
    @function(allow_implicit=True)
    def f(a:Tensor) -> Tensor: return a + inp

    np.testing.assert_equal(f(Tensor([10,20,30])).numpy(), [15,27,39])

  def test_detach(self):
    @function
    def f(a:Tensor, b:Tensor) -> Tensor: return a.detach() + b

    a = Tensor([1,2,3])
    b = Tensor([4,5,6])
    np.testing.assert_equal(f(a, b).numpy(), [5,7,9])

  def test_contiguous_backward(self):
    @function
    def f(a:Tensor, b:Tensor) -> Tensor: return (a + b).contiguous_backward()

    a = Tensor([1,2,3])
    b = Tensor([4,5,6])
    np.testing.assert_equal(f(a, b).numpy(), [5,7,9])

  def test_method(self):
    class Foo:
      def __init__(self): self.w = Tensor([10,20,30])
      @function
      def __call__(self, x:Tensor) -> Tensor: return x + self.w

    foo = Foo()
    np.testing.assert_equal(foo(Tensor([1,2,3])).numpy(), [11,22,33])

  def test_grad_gemm(self):
    @function
    def f(a:Tensor, b:Tensor) -> Tensor: return a @ b

    a = Tensor([[1.,2.],[3.,4.]])
    b = Tensor([[5.,6.],[7.,8.]])
    (f(a, b).contiguous() * b).sum().backward()
    Tensor.realize(a, b, a.grad, b.grad)
    # L = sum((a@b) * b), dL/d(a@b) = b, dL/da = b @ b^T, dL/db = a^T @ b + (a@b)
    na, nb = a.numpy(), b.numpy()
    np.testing.assert_allclose(a.grad.numpy(), nb @ nb.T)
    np.testing.assert_allclose(b.grad.numpy(), na.T @ nb + na @ nb)

  def test_grad_implicit(self):
    w = Tensor([1., 2., 3.])
    w.realize() # TODO: this is required
    @function(allow_implicit=True)
    def f(x:Tensor) -> Tensor: return x * w

    x = Tensor([4., 5., 6.])
    f(x).sum().backward()
    np.testing.assert_allclose(w.grad.numpy(), [4., 5., 6.])

  def test_symbolic_index(self):
    table = Tensor([10,20,30,40]).contiguous().realize()
    @function(allow_implicit=True)
    def f(x:Tensor, start_pos:int|UOp) -> Tensor:
      return x + table[start_pos]

    v = UOp.variable("start_pos", 0, 3)
    np.testing.assert_equal(f(Tensor([1,2,3]), v.bind(0)).numpy(), [11,12,13])

  def test_symbolic_shape_input(self):
    table = Tensor([10,20,30,40]).contiguous().realize()
    @function
    def f(x:Tensor) -> Tensor: return x * 2
    sz = UOp.variable("sz", 1, 3)
    slic = table[:sz.bind(2)]
    np.testing.assert_equal(f(slic)[:2].numpy(), [20,40])

  def test_nested_calls(self):
    w = Tensor([10., 20., 30.])
    @function(allow_implicit=True)
    def f(a:Tensor) -> Tensor: return a + w
    @function(allow_implicit=True)
    def g(a:Tensor) -> Tensor: return a * w

    a = Tensor([1., 2., 3.])
    np.testing.assert_allclose(g(f(a)).numpy(), [110., 440., 990.])

  def test_nested_calls_backward(self):
    w = Tensor([[1., 2.], [3., 4.]]).contiguous().realize()
    @function(allow_implicit=True)
    def inner(x:Tensor) -> Tensor: return x + w
    @function(allow_implicit=True)
    def outer(a:Tensor, b:Tensor) -> Tensor: return inner(a.reshape(1,2) + b.reshape(1,2))

    a = Tensor([1., 2.])
    b = Tensor([3., 4.])
    outer(a, b).sum().backward()
    np.testing.assert_allclose(a.grad.numpy(), [2., 2.])
    np.testing.assert_allclose(b.grad.numpy(), [2., 2.])

  def test_unused_param_backward(self):
    @function
    def f(a:Tensor, b:Tensor, c:Tensor) -> Tensor: return a + c  # b is unused

    a = Tensor([1., 2., 3.])
    b = Tensor([4., 5., 6.])
    c = Tensor([7., 8., 9.])
    f(a, b, c).sum().backward()
    np.testing.assert_allclose(a.grad.numpy(), [1., 1., 1.])
    np.testing.assert_allclose(b.grad.numpy(), [0., 0., 0.])
    np.testing.assert_allclose(c.grad.numpy(), [1., 1., 1.])

  def test_name(self):
    @function
    def f(a:Tensor) -> Tensor: return a + 1
    assert f(Tensor([1])).uop.src[0].arg.name.endswith("f")

  def test_method_name(self):
    class Foo:
      @function
      def __call__(self, x:Tensor) -> Tensor: return x + 1
    assert Foo()(Tensor([1])).uop.src[0].arg.name.endswith("Foo.__call__")

  def test_callable_instance(self):
    class Foo:
      def __init__(self): self.w = Tensor([10,20,30])
      def __call__(self, x:Tensor) -> Tensor: return x + self.w
    foo = Foo()
    f = function(foo, allow_implicit=True)
    np.testing.assert_equal(f(Tensor([1,2,3])).numpy(), [11,22,33])
    assert f(Tensor([1,2,3])).uop.src[0].arg.name.endswith("Foo")

  def test_iadd(self):
    @function
    def f(x:Tensor) -> Tensor:
      x += 1
      return x

    a = Tensor([1,2,3]).realize()
    np.testing.assert_equal(f(a).numpy(), [2,3,4])
    np.testing.assert_equal(a.numpy(), [3,4,5])  # TODO: should be [1,2,3]

  def test_implicit_assign(self):
    a = Tensor([1,2,3])
    a += 1
    c = Tensor([2,2,2]).contiguous()
    @function
    def f(b:Tensor) -> Tensor: return a+b+c
    b = Tensor([10,20,30]).realize()
    np.testing.assert_equal(f(b).numpy(), [14,25,36])

  def test_assign_input(self):
    @function
    def f(a:Tensor, b:Tensor) -> Tensor:
      a.assign(b+1)
      return a

    a = Tensor([1,2,3]).realize()
    b = Tensor([10,20,30]).realize()
    np.testing.assert_equal(f(a,b).numpy(), [11,21,31])
    np.testing.assert_equal(a.numpy(), [11,21,31])  # TODO: should be [1,2,3]
    np.testing.assert_equal(b.numpy(), [10,20,30])

  def test_view_assign_explicit_buffer(self):
    """view assign on an explicit param's buffer should not create implicit inputs."""
    class State:
      def __init__(self): self.buf = Tensor.zeros(2, 4).contiguous().realize()
      @function(allow_implicit=False)
      def __call__(self, x:Tensor) -> Tensor:
        self.buf[:, 0:2].assign(x)
        return self.buf[:, 0:2]
    s = State()
    np.testing.assert_equal(s(Tensor([[5., 6.], [7., 8.]])).numpy(), [[5., 6.], [7., 8.]])

  def test_single_after_store(self):
    """AFTER(buf, STORE(view, data)) should write data through the view into buf, same as the double-after pattern."""
    @function
    def f(buf:Tensor, x:Tensor, start_pos:int|UOp) -> Tensor:
      slice_uop = buf[:, start_pos:start_pos+1].uop
      assigned = Tensor(buf.uop.after(slice_uop.store(x.uop)))
      return assigned

    buf = Tensor.zeros(2, 8).contiguous().realize()
    x = Tensor([[1.], [2.]]).realize()
    v = UOp.variable("sp", 0, 7)
    r0 = f(buf, x, v.bind(0)).numpy()
    np.testing.assert_equal(r0, [[1.,0.,0.,0.,0.,0.,0.,0.], [2.,0.,0.,0.,0.,0.,0.,0.]])

  @unittest.expectedFailure
  def test_assign_slice(self):
    @function
    def f(a:Tensor, b:Tensor) -> Tensor:
      a[1:] = b[1:]+1
      return a

    a = Tensor([1,2,3]).realize()
    b = Tensor([10,20,30]).realize()
    np.testing.assert_equal(f(a,b).numpy(), [1,21,31])
    np.testing.assert_equal(a.numpy(), [1,2,3])
    np.testing.assert_equal(b.numpy(), [10,20,30])

class TestFunctionMulti(unittest.TestCase):
  devices_2 = ("CPU:0", "CPU:1")

  def test_simple_multi(self):
    @function
    def f(a:Tensor, b:Tensor) -> Tensor: return a+b

    a = Tensor([1,2,3,4]).shard(self.devices_2, axis=None)
    b = Tensor([10,20,30,40]).shard(self.devices_2, axis=None)
    np.testing.assert_equal(f(a,b).numpy(), [11,22,33,44])

  def test_simple_multi_sharded(self):
    @function
    def f(a:Tensor, b:Tensor) -> Tensor: return a+b

    a = Tensor([1,2,3,4]).shard(self.devices_2, axis=0)
    b = Tensor([10,20,30,40]).shard(self.devices_2, axis=0)
    np.testing.assert_equal(f(a,b).numpy(), [11,22,33,44])

  def test_data_parallel_multi(self):
    @function
    def f(x:Tensor, w:Tensor) -> Tensor: return x @ w

    x = Tensor([[1.,2.],[3.,4.],[5.,6.],[7.,8.]]).shard(self.devices_2, axis=0)
    w = Tensor([[1.,0.],[0.,1.]]).shard(self.devices_2, axis=None)
    np.testing.assert_allclose(f(x, w).numpy(), [[1.,2.],[3.,4.],[5.,6.],[7.,8.]])

  def test_grad_implicit_multi(self):
    w = Tensor([1., 2., 3., 4.]).shard(self.devices_2, axis=None)
    w.realize()
    @function(allow_implicit=True)
    def f(x:Tensor) -> Tensor: return x * w

    x = Tensor([4., 5., 6., 7.]).shard(self.devices_2, axis=None)
    f(x).sum().backward()
    np.testing.assert_allclose(w.grad.numpy(), [4., 5., 6., 7.])

  def test_call_axis(self):
    @function
    def f(x:Tensor, w:Tensor) -> Tensor: return x @ w

    x = Tensor([[1.,0.],[0.,1.],[1.,1.],[0.,0.]]).shard(self.devices_2, axis=0)
    w = Tensor([[1.,2.],[3.,4.]]).shard(self.devices_2, axis=None)
    result = f(x, w)
    # CALL output should inherit axis=0 from the sharded input
    self.assertEqual(result.uop.axis, 0)
    # reduce on the sharded axis should remove it
    self.assertIsNone(result.sum().uop.axis)

  def test_call_axis_shard_inside(self):
    @function
    def f(x:Tensor, w:Tensor) -> Tensor:
      return x.shard(self.devices_2, axis=0) @ w.shard(self.devices_2, axis=None)

    x = Tensor([[1.,0.],[0.,1.],[1.,1.],[0.,0.]])
    w = Tensor([[1.,2.],[3.,4.]])
    result = f(x, w)
    self.assertEqual(result.uop.axis, 0)
    np.testing.assert_allclose(result.numpy(), x.numpy() @ w.numpy())

  def test_data_parallel_backward(self):
    @function
    def f(x:Tensor, w:Tensor) -> Tensor: return x @ w

    x = Tensor([[1.,0.],[0.,1.],[1.,1.],[0.,0.]]).shard(self.devices_2, axis=0)
    w = Tensor([[1.,2.],[3.,4.]]).shard(self.devices_2, axis=None)
    w.realize()
    f(x, w).sum().backward()
    # d/dx = ones @ w^T = [[1,3],[1,3],[1,3],[1,3]], but sum so ones(4,2) @ w^T? no:
    # L = sum(x @ w), dL/dx = ones(4,2) @ w^T... actually dL/d(xw) = ones(4,2), dL/dx = ones(4,2) @ w^T
    np.testing.assert_allclose(x.grad.numpy(), np.ones((4,2)) @ np.array([[1,3],[2,4]]))

  def test_data_parallel_backward_4(self):
    devices_4 = tuple(f"CPU:{i}" for i in range(4))
    @function
    def f(x:Tensor, w:Tensor) -> Tensor: return x @ w

    x = Tensor(np.arange(16).reshape(8,2).astype(np.float32)).shard(devices_4, axis=0)
    w = Tensor([[1.,2.],[3.,4.]]).shard(devices_4, axis=None)
    w.realize()
    f(x, w).sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), np.ones((8,2)) @ np.array([[1,3],[2,4]]))

  def test_data_parallel_backward_implicit(self):
    devices_4 = tuple(f"CPU:{i}" for i in range(4))
    w = Tensor([[1.,2.],[3.,4.]]).shard(devices_4, axis=None)
    w.realize()
    @function(allow_implicit=True)
    def f(x:Tensor) -> Tensor: return x @ w

    x = Tensor(np.arange(16).reshape(8,2).astype(np.float32)).shard(devices_4, axis=0)
    f(x).sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), np.ones((8,2)) @ np.array([[1,3],[2,4]]))

  def test_data_parallel_backward_twice(self):
    devices_4 = tuple(f"CPU:{i}" for i in range(4))
    w = Tensor([[1.,2.],[3.,4.]]).shard(devices_4, axis=None)
    w.realize()
    # pre-init grads like the training loop does
    w.grad = w.zeros_like().contiguous().realize()
    @function(allow_implicit=True)
    def f(x:Tensor) -> Tensor: return x @ w

    expected = np.ones((8,2)) @ np.array([[1,3],[2,4]])
    for _ in range(2):
      x = Tensor(np.arange(16).reshape(8,2).astype(np.float32)).shard(devices_4, axis=0)
      f(x).sum().backward()
      np.testing.assert_allclose(x.grad.numpy(), expected)

class TestFunctionTuple(unittest.TestCase):
  def test_tuple(self, precompile=False):
    x = Tensor.ones(3).contiguous()
    @function(precompile=precompile)
    def f(t:Tensor): return (t+1, t+2)
    t1, t2 = f(x)
    t1.realize(t2)
    print(t1.tolist(), t2.tolist())
    assert t1.tolist() == [2,2,2]
    assert t2.tolist() == [3,3,3]
  def test_tuple_precompile(self): self.test_tuple(True)

  def test_grad_tuple(self, precompile=False):
    x = Tensor.ones(3).contiguous()
    y = Tensor.ones(3).contiguous()
    @function(precompile=precompile)
    def f(u1:Tensor, u2:Tensor): return (u1+1, u2+2)
    t1, t2 = f(x,y)
    (t1+t2).sum().backward()
    x.grad.realize(y.grad)
  def test_grad_tuple_precompile(self): self.test_grad_tuple(True)

  def test_grad_fxn_tuple(self):
    # grad_fxn for tuple: receives one gradient per output as positional args
    def grad_fxn(d_out0:UOp, d_out1:UOp, call:UOp):
      # f(u1, u2) = (u1+1, u2+2)
      # df/du1 = d_out0, df/du2 = d_out1
      return (d_out0, d_out1)

    x = Tensor.ones(3).contiguous()
    y = Tensor.ones(3).contiguous()
    @function(grad_fxn=grad_fxn)
    def f(u1:Tensor, u2:Tensor): return (u1+1, u2+2)
    t1, t2 = f(x, y)
    (t1+t2).sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), [1., 1., 1.])
    np.testing.assert_allclose(y.grad.numpy(), [1., 1., 1.])

  def test_grad_unused_tuple_output_recursive(self):
    # only one output is used
    @function(precompile=True, precompile_backward=True)
    def f(x:Tensor, w:Tensor):
      a = x @ w
      b = (x @ w) * 2  # shares x@w with a
      return (a, b)

    x = Tensor([[1., 2.], [3., 4.]]).contiguous()
    w = Tensor([[1., 0.], [0., 1.]]).contiguous()
    Tensor.realize(x, w)
    t1, _ = f(x, w)
    t1.sum().backward()
    Tensor.realize(x.grad, w.grad)
    # only t1 = x @ w flows to loss; dL/dw = x.T @ ones(2,2)
    np.testing.assert_allclose(w.grad.numpy(), np.array([[1., 2.], [3., 4.]]).T @ np.ones((2, 2)))
    np.testing.assert_allclose(x.grad.numpy(), np.ones((2, 2)) @ np.array([[1., 0.], [0., 1.]]).T)

  def test_custom_kernel_save_unused_output(self):
    def my_kernel(C:UOp, D:UOp, A:UOp) -> UOp:
      i = UOp.range(A.shape[0], 0)
      j = UOp.range(D.shape[0], 1)
      store_c = C[i].store(A[i] * 2.0).end(i)
      store_d = D[j].store(A[j]).end(j)
      return UOp.sink(store_c, store_d, arg=KernelInfo(name="my_kernel"))

    def my_grad(d_c:UOp, call:UOp):
      a_input = call.src[3]
      return (None, None, (Tensor(d_c) * 2.0 + Tensor(a_input) * 0).uop)

    @function(precompile=True, precompile_backward=True)
    def f(a:Tensor):
      c = Tensor.invalids(*a.shape, dtype=a.dtype, device=a.device)
      d = Tensor.invalids(3, dtype=a.dtype, device=a.device)
      c, d = Tensor.custom_kernel(c, d, a, fxn=my_kernel, grad_fxn=my_grad)[:2]
      return c, d

    a = Tensor([1., 2., 3., 4.]).contiguous()
    Tensor.realize(a)
    c, _ = f(a)
    c.sum().backward()
    Tensor.realize(a.grad)
    np.testing.assert_allclose(a.grad.numpy(), [2., 2., 2., 2.])

  def test_custom_kernel_both_outputs_used(self):
    def my_kernel(C:UOp, D:UOp, A:UOp) -> UOp:
      i = UOp.range(A.shape[0], 0)
      store_c = C[i].store(A[i] * 2.0)
      store_d = D[i].store(A[i] * 3.0)
      return UOp.group(store_c, store_d).end(i).sink(arg=KernelInfo(name="my_kernel"))

    def my_grad(d_c:UOp, d_d:UOp, call:UOp):
      return (None, None, (Tensor(d_c) + Tensor(d_d)).uop)

    @function(precompile=True, precompile_backward=True)
    def f(a:Tensor):
      c = Tensor.invalids(*a.shape, dtype=a.dtype, device=a.device)
      d = Tensor.invalids(*a.shape, dtype=a.dtype, device=a.device)
      c, d = Tensor.custom_kernel(c, d, a, fxn=my_kernel, grad_fxn=my_grad)[:2]
      return (c, d)

    a = Tensor([1., 2., 3., 4.]).contiguous()
    Tensor.realize(a)
    c, d = f(a)
    (c.sum() + d.sum()).backward()  # dL/da = (1 + 1) since grad_fxn passes d_combined through
    Tensor.realize(a.grad)
    np.testing.assert_allclose(a.grad.numpy(), [2., 2., 2., 2.])

  def test_custom_kernel_precompile_no_copy_kernel(self):
    def my_kernel(C:UOp, A:UOp) -> UOp:
      i = UOp.range(A.shape[0], 0)
      return C[i].store(A[i] * 2.0).end(i).sink(arg=KernelInfo(name="my_kernel"))

    def my_grad(d_c:UOp, call:UOp):
      return (None, (Tensor(d_c) * 2.0).uop)

    @function(precompile=True, precompile_backward=True)
    def f(a:Tensor):
      c = Tensor.invalids(*a.shape, dtype=a.dtype, device=a.device)
      c = Tensor.custom_kernel(c, a, fxn=my_kernel, grad_fxn=my_grad)[0]
      return c

    def count_kernels(t:Tensor):
      linear, _ = t.linear_with_vars()
      return sum((len(call.device) if isinstance(call.device, tuple) else 1)
                 for call in linear.src if call.src[0].op is Ops.SINK)

    a = Tensor([1., 2., 3., 4.]).contiguous()
    Tensor.realize(a)
    c = f(a)

    self.assertEqual(count_kernels(c), 1)

    c.sum().backward()
    Tensor.realize(a.grad)
    np.testing.assert_allclose(a.grad.numpy(), [2., 2., 2., 2.])

  def test_custom_kernel_precompile_further_compute(self):
    def my_kernel(C:UOp, A:UOp) -> UOp:
      i = UOp.range(A.shape[0], 0)
      return C[i].store(A[i] * 2.0).end(i).sink(arg=KernelInfo(name="my_kernel"))

    @function(precompile=True)
    def f(a:Tensor):
      c = Tensor.invalids(*a.shape, dtype=a.dtype, device=a.device)
      c = Tensor.custom_kernel(c, a, fxn=my_kernel)[0]
      return c + 1

    a = Tensor([1., 2., 3., 4.]).contiguous().realize()
    np.testing.assert_allclose(f(a).numpy(), [3., 5., 7., 9.])

class TestFunctionGrad(unittest.TestCase):
  def test_function_grad_ops(self, precompile=False, precompile_backward=False):
    N = 64
    x = Tensor.ones(N,N).contiguous()
    w1 = Tensor.ones(N,N).contiguous()
    w2 = Tensor.ones(N,N).contiguous()
    w3 = Tensor.ones(N,N).contiguous()
    ref = Tensor.ones(N,N).contiguous()
    Tensor.realize(x, w1, w2, w3, ref)
    @function(precompile=precompile, precompile_backward=precompile_backward)
    def f(x, w1, w2, w3) -> tuple[Tensor, ...]:
      p1 = x@w1
      p2 = p1@w2
      p3 = p2@w3
      return p1, p2, p3, p3.contiguous()
    ret = f(x, w1, w2, w3)[-1]
    loss = (ret-ref).square().mean().backward()
    print("RESET")
    GlobalCounters.reset()
    loss.realize(w1.grad, w2.grad, w3.grad)
    print(GlobalCounters.global_ops, GlobalCounters.global_mem)
    self.assertLessEqual(GlobalCounters.global_ops, 4739073)
  def test_function_grad_ops_precompile(self): self.test_function_grad_ops(precompile=True)
  def test_function_grad_ops_precompile_backward(self):
    self.test_function_grad_ops(precompile=True, precompile_backward=True)

if __name__ == '__main__':
  unittest.main()
