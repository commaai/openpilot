import unittest
import numpy as np
from tinygrad import Tensor
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import UOp, KernelInfo

class TestTensorGradient(unittest.TestCase):
  def test_example(self):
    x = Tensor.eye(3)
    y = Tensor([[2.0,0,-2.0]])
    z = y.matmul(x).sum()
    dx, dy = z.gradient(x, y)
    self.assertListEqual(dx.tolist(), [[2.0, 2.0, 2.0], [0.0, 0.0, 0.0], [-2.0, -2.0, -2.0]])
    self.assertListEqual(dy.tolist(), [[1.0, 1.0, 1.0]])

  def test_zero_if_not_used(self):
    x = Tensor([1.0, 2.0, 3.0])
    w = Tensor.randn((3,))
    self.assertListEqual(x.sum().gradient(w)[0].tolist(), [0.0, 0.0, 0.0])

  def test_with_custom_gradient(self):
    x = Tensor([1.0, 2.0, 3.0])
    z = (x * x).sum()
    dx = z.gradient(x, gradient=Tensor([3.0]))[0]
    self.assertListEqual(dx.tolist(), [6.0, 12.0, 18.0])

  def test_broadcast_gradient(self):
    x = Tensor([[1.0], [2.0], [3.0]])
    y = Tensor([[10.0, 20.0, 30.0, 40.0]])
    z = (x + y).sum()
    dx, dy = z.gradient(x, y)
    self.assertListEqual(dx.tolist(), [[4.0], [4.0], [4.0]])
    self.assertListEqual(dy.tolist(), [[3.0, 3.0, 3.0, 3.0]])

  def test_non_scalar_output(self):
    x = Tensor([1.0, 2.0, 3.0])
    z = x * x
    with self.assertRaises(AssertionError): z.gradient(x)
    dz = Tensor([1.0, 1.0, 1.0])
    dx = z.gradient(x, gradient=dz)[0]
    self.assertListEqual(dx.tolist(), [2.0, 4.0, 6.0])

  def test_cast_before_view(self):
    x = Tensor([1.0, 1, 1, 1])
    x_reshaped = x.reshape(2,2)
    x_casted = x_reshaped.cast(dtypes.float16)
    x_casted.mean().gradient(x_reshaped)

  def test_non_float_tensor_raise(self):
    x = Tensor([1, 2, 3])
    with self.assertRaises(RuntimeError): x.sum().gradient(x)
    with self.assertRaises(RuntimeError): x.float().sum().gradient(x)

  def test_copy_to_device_gradient(self):
    t = Tensor([1.0, 2, 3]).realize()
    t.to("CPU:1").square().sum().backward()
    self.assertEqual(t.grad.device, t.device)
    self.assertListEqual(t.grad.tolist(), [2.0, 4.0, 6.0])

  def test_multiple_backward(self):
    x = Tensor([3.])
    (x*2)[0].backward()
    np.testing.assert_allclose(x.grad.numpy(), [2.0])
    old_grad = x.grad
    (x*3)[0].backward()
    np.testing.assert_allclose(x.grad.numpy(), [2.0+3.0])
    self.assertIs(x.grad, old_grad)
    (x*x)[0].backward()
    np.testing.assert_allclose(x.grad.numpy(), [2.0+3.0+2*3.0])
    self.assertIs(x.grad, old_grad)

  def test_gradient_through_clone_from_grad_src(self):
    # unlike torch, tinygrad accumulates grad on every tensor in the graph, including non-leaf x
    src = Tensor([1.0, 2.0, 3.0, 4.0])
    x = src.clone()
    (x * 2.0).sum().backward()
    np.testing.assert_allclose(src.grad.numpy(), [2.0, 2.0, 2.0, 2.0])
    np.testing.assert_allclose(x.grad.numpy(), [2.0, 2.0, 2.0, 2.0])

  def test_gradient_through_clone_from_detached_src(self):
    base = Tensor([1.0, 2.0, 3.0, 4.0])
    x = base.detach().clone()
    (x * 2.0).sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), [2.0, 2.0, 2.0, 2.0])     # gradient flows through clone
    np.testing.assert_allclose(base.grad.numpy(), [0.0, 0.0, 0.0, 0.0])  # ...but detach blocks it from base

  def test_setitem_on_grad_used_tensor_raises(self):
    x = Tensor([1.0, 2.0, 3.0, 4.0]).realize()
    _ = (x * 2.0).sum()
    with self.assertRaises(RuntimeError):
      x[0] = 99.0

  def test_gradient_through_chained_unrealized_setitem(self):
    g1 = Tensor.zeros(4).contiguous()
    g1[2] = Tensor(1.0)
    g2 = Tensor.zeros(5, 4).contiguous()
    g2[0] = g1
    x = Tensor.randn(4, 4)
    np.testing.assert_allclose(x.pad(((1,0),(0,0))).gradient(x, gradient=g2)[0].numpy(), np.zeros((4, 4)))

  def test_bare_const_skipped_by_backward(self):
    Tensor.manual_seed(0)
    w = Tensor(1.0)
    (Tensor.rand(()) + w).backward()
    self.assertIsNone(w.grad)

class TestMultiOutputGradient(unittest.TestCase):
  @staticmethod
  def addmul_kernel(C:UOp, D:UOp, A:UOp, B:UOp) -> UOp:
    C, D, A, B = C.flatten(), D.flatten(), A.flatten(), B.flatten()
    i = UOp.range(C.numel(), 0)
    store_c = C[i].store(A[i] + B[i])
    store_d = D[i].store(A[i] * B[i])
    return UOp.group(store_c, store_d).end(i).sink(arg=KernelInfo(name="addmul")).simplify()
  @staticmethod
  def backward_addmul(grad_c, grad_d, call):
    _c, _d, a, b = call.src[1:]
    grad_a = (Tensor(grad_c) + Tensor(grad_d) * Tensor(b)).uop
    grad_b = (Tensor(grad_c) + Tensor(grad_d) * Tensor(a)).uop
    return (None, None, grad_a, grad_b)

  def test_custom_kernel_multi_output_backward(self):
    a_np, b_np = np.random.randn(4, 4).astype(np.float32), np.random.randn(4, 4).astype(np.float32)
    a_ref, b_ref = Tensor(a_np), Tensor(b_np)
    ((a_ref + b_ref).sum() + (a_ref * b_ref).sum()).backward()

    a, b = Tensor(a_np), Tensor(b_np)
    Tensor.realize(a, b)
    c, d, _, _ = Tensor.custom_kernel(Tensor.empty(4, 4), Tensor.empty(4, 4), a, b, fxn=self.addmul_kernel, grad_fxn=self.backward_addmul)
    (c.sum() + d.sum()).backward()
    np.testing.assert_allclose(a.grad.numpy(), a_ref.grad.numpy(), rtol=1e-5)
    np.testing.assert_allclose(b.grad.numpy(), b_ref.grad.numpy(), rtol=1e-5)

  def test_custom_kernel_multi_output_backward_interacting(self):
    a_np, b_np = np.random.randn(4, 4).astype(np.float32), np.random.randn(4, 4).astype(np.float32)
    a_ref, b_ref = Tensor(a_np), Tensor(b_np)
    ((a_ref + b_ref) * (a_ref * b_ref)).sum().backward()

    a, b = Tensor(a_np), Tensor(b_np)
    Tensor.realize(a, b)
    c, d, _, _ = Tensor.custom_kernel(Tensor.empty(4, 4), Tensor.empty(4, 4), a, b, fxn=self.addmul_kernel, grad_fxn=self.backward_addmul)
    (c * d).sum().backward()
    np.testing.assert_allclose(a.grad.numpy(), a_ref.grad.numpy(), rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(b.grad.numpy(), b_ref.grad.numpy(), rtol=1e-5, atol=1e-7)

  def test_custom_kernel_three_output_backward(self):
    def addmulsub_kernel(C:UOp, D:UOp, E:UOp, A:UOp, B:UOp) -> UOp:
      C, D, E, A, B = C.flatten(), D.flatten(), E.flatten(), A.flatten(), B.flatten()
      i = UOp.range(C.numel(), 0)
      store_c = C[i].store(A[i] + B[i])
      store_d = D[i].store(A[i] * B[i])
      store_e = E[i].store(A[i] - B[i])
      return UOp.group(store_c, store_d, store_e).end(i).sink(arg=KernelInfo(name="addmulsub")).simplify()
    def backward_addmulsub(grad_c, grad_d, grad_e, call):
      _c, _d, _e, a, b = call.src[1:]
      grad_a = (Tensor(grad_c) + Tensor(grad_d) * Tensor(b) + Tensor(grad_e)).uop
      grad_b = (Tensor(grad_c) + Tensor(grad_d) * Tensor(a) - Tensor(grad_e)).uop
      return (None, None, None, grad_a, grad_b)

    a_np, b_np = np.random.randn(4, 4).astype(np.float32), np.random.randn(4, 4).astype(np.float32)
    a_ref, b_ref = Tensor(a_np), Tensor(b_np)
    ((a_ref + b_ref).sum() + (a_ref * b_ref).sum() + (a_ref - b_ref).sum()).backward()

    a, b = Tensor(a_np), Tensor(b_np)
    Tensor.realize(a, b)
    c, d, e, _, _ = Tensor.custom_kernel(Tensor.empty(4, 4), Tensor.empty(4, 4), Tensor.empty(4, 4), a, b,
                                          fxn=addmulsub_kernel, grad_fxn=backward_addmulsub)
    (c.sum() + d.sum() + e.sum()).backward()
    np.testing.assert_allclose(a.grad.numpy(), a_ref.grad.numpy(), rtol=1e-5)
    np.testing.assert_allclose(b.grad.numpy(), b_ref.grad.numpy(), rtol=1e-5)

class TestViewGradient(unittest.TestCase):
  def test_expand(self):
    x = Tensor.randn(5,2)
    a = Tensor([3.])
    aex = a.expand(10)
    (aex.reshape(5,2) * x).sum().backward()
    np.testing.assert_allclose(aex.grad.numpy(), x.reshape(10).numpy())
    with self.assertRaises(AssertionError):
      np.testing.assert_allclose(aex.grad.numpy(), a.grad.expand(10).numpy())

if __name__ == '__main__':
  unittest.main()
