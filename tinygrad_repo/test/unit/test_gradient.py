import unittest
import numpy as np
from tinygrad import Tensor
from tinygrad.dtype import dtypes

class TestTensorGradient(unittest.TestCase):
  def test_example(self):
    x = Tensor.eye(3)
    y = Tensor([[2.0,0,-2.0]])
    z = y.matmul(x).sum()
    dx, dy = z.gradient(x, y)
    self.assertListEqual(dx.tolist(), [[2.0, 2.0, 2.0], [0.0, 0.0, 0.0], [-2.0, -2.0, -2.0]])
    self.assertListEqual(dy.tolist(), [[1.0, 1.0, 1.0]])

  def test_raises(self):
    x = Tensor([1.0, 2.0, 3.0])
    w = Tensor.randn((3,))
    with self.assertRaises(RuntimeError): x.sum().gradient(w)

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
    t = Tensor([1.0, 2, 3], requires_grad=True).realize()
    t.to("CPU:1").square().sum().backward()
    self.assertEqual(t.grad.device, t.device)
    self.assertListEqual(t.grad.tolist(), [2.0, 4.0, 6.0])

  def test_multiple_backward(self):
    x = Tensor([3.], requires_grad=True)
    (x*2)[0].backward()
    np.testing.assert_allclose(x.grad.numpy(), [2.0])
    old_grad = x.grad
    (x*3)[0].backward()
    np.testing.assert_allclose(x.grad.numpy(), [2.0+3.0])
    self.assertIs(x.grad, old_grad)
    (x*x)[0].backward()
    np.testing.assert_allclose(x.grad.numpy(), [2.0+3.0+2*3.0])
    self.assertIs(x.grad, old_grad)

class TestViewGradient(unittest.TestCase):
  def test_expand(self):
    x = Tensor.randn(5,2)
    a = Tensor([3.], requires_grad=True)
    aex = a.expand(10)
    (aex.reshape(5,2) * x).sum().backward()
    np.testing.assert_allclose(aex.grad.numpy(), x.reshape(10).numpy())
    with self.assertRaises(AssertionError):
      np.testing.assert_allclose(aex.grad.numpy(), a.grad.expand(10).numpy())

if __name__ == '__main__':
  unittest.main()
