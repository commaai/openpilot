import unittest
from tinygrad.function import function
from tinygrad import Tensor, GlobalCounters
from tinygrad.uop.ops import UOp, KernelInfo

class TestFunction(unittest.TestCase):
  def test_depth_restored_on_exception(self):
    from tinygrad.function import _function
    @function
    def f(a:Tensor) -> Tensor: raise ValueError("error")
    with self.assertRaises(ValueError): f(Tensor([1]))
    self.assertEqual(_function.depth, 0)

  def test_name(self):
    @function
    def f(a:Tensor) -> Tensor: return a + 1
    assert f(Tensor([1])).uop.src[1].arg.name.endswith("f")

  def test_method_name(self):
    class Foo:
      @function
      def __call__(self, x:Tensor) -> Tensor: return x + 1
    assert Foo()(Tensor([1])).uop.src[1].arg.name.endswith("Foo.__call__")

class TestFunctionMulti(unittest.TestCase):
  devices_2 = ("NULL:0", "NULL:1")

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

class TestFunctionTuple(unittest.TestCase):
  def test_custom_kernel_inplace_output_is_implicit(self):
    # caller-owned storage must be captured, even before its Buffer is bound
    state = Tensor.empty(4)
    def inplace_add(C:UOp, A:UOp) -> UOp:
      i = UOp.range(A.shape[0], 0)
      return C[i].store(C[i].load() + A[i]).end(i).sink(arg=KernelInfo(name="inplace_add"))
    @function(precompile=True, allow_implicit=False)
    def f(a:Tensor): return Tensor.custom_kernel(state, a, fxn=inplace_add)[0]
    with self.assertRaisesRegex(RuntimeError, "implicit buffer"): f(Tensor([1., 2., 3., 4.]).contiguous().realize())

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
    self.assertLessEqual(GlobalCounters.global_ops, 5000000)
  def test_function_grad_ops_precompile(self): self.test_function_grad_ops(precompile=True)
  def test_function_grad_ops_precompile_backward(self):
    self.test_function_grad_ops(precompile=True, precompile_backward=True)

if __name__ == '__main__':
  unittest.main()
