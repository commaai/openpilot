#!/usr/bin/env python
import unittest
import numpy as np
from tinygrad import Tensor, dtypes, Device
from tinygrad.nn import Linear
from extra.fp8.fp8_linear import FP8Linear, convert_to_float8_training
from tinygrad.device import is_dtype_supported
from test.helpers import not_support_multi_device, needs_second_gpu

BS, T, in_dim, out_dim = 16, 4, 128, 128

@unittest.skipUnless(is_dtype_supported(dtypes.fp8e4m3), f"no fp8e4m3 on {Device.DEFAULT}")
class TestFP8Linear(unittest.TestCase):
  def setUp(self):
    Tensor.manual_seed(42)

  def _test_forward(self, shape, in_features, out_features):
    fp8_layer = FP8Linear(in_features, out_features)
    normal_layer = Linear(in_features, out_features)
    weight = Tensor.randn(out_features, in_features, dtype=dtypes.float32)  * 0.2
    bias = Tensor.randn(out_features, dtype=dtypes.float32)  * 0.2
    fp8_layer.weight.assign(weight)
    normal_layer.weight.assign(weight)
    fp8_layer.bias.assign(bias)
    normal_layer.bias.assign(bias)
    x = Tensor.randn(*shape, dtype=dtypes.float32)  * 0.2
    y_fp8, y_normal = fp8_layer(x), normal_layer(x)
    np.testing.assert_allclose(y_fp8.numpy(), y_normal.numpy(), rtol=0.1, atol=0.1)

  def _test_backward(self, shape, in_features, out_features):
    fp8_layer = FP8Linear(in_features, out_features)
    normal_layer = Linear(in_features, out_features)
    weight = Tensor.randn(out_features, in_features, dtype=dtypes.float32) * 0.2
    bias = Tensor.randn(out_features, dtype=dtypes.float32) * 0.2
    fp8_layer.weight, normal_layer.weight = weight.detach(), weight.detach()
    fp8_layer.bias, normal_layer.bias = bias.detach(), bias.detach()
    fp8_layer.weight.requires_grad = normal_layer.weight.requires_grad = True
    x_fp8 = Tensor.randn(*shape, dtype=dtypes.float32, requires_grad=True)  * 0.2
    x_normal = x_fp8.detach().requires_grad_(True)
    fp8_layer(x_fp8).sum().backward()
    normal_layer(x_normal).sum().backward()
    np.testing.assert_allclose(x_fp8.grad.numpy(), x_normal.grad.numpy(), rtol=1.0, atol=0.1)
    np.testing.assert_allclose(fp8_layer.weight.grad.numpy(), normal_layer.weight.grad.numpy(), rtol=1.0, atol=0.1)

  def test_forward_2d(self): self._test_forward((BS, in_dim), in_dim, out_dim)
  def test_forward_3d(self): self._test_forward((BS, T, in_dim), in_dim, out_dim)

  def test_backward_2d(self): self._test_backward((BS, in_dim), in_dim, out_dim)
  def test_backward_3d(self): self._test_backward((BS, T, in_dim), in_dim, out_dim)

  def test_filter(self):
    class Model:
      def __init__(self):
        self.fc1 = Linear(32, 16)
        self.fc2 = Linear(16, 8)
      def __call__(self, x):
        return self.fc2(self.fc1(x).relu())
    model = Model()
    x = Tensor.randn(16, 32)
    y_before = model(x).numpy()
    convert_to_float8_training(model, module_filter_fn=lambda _, fqn: "fc1" in fqn)
    self.assertIsInstance(model.fc1, FP8Linear)
    self.assertNotIsInstance(model.fc2, FP8Linear)
    y_after = model(x).numpy()
    np.testing.assert_allclose(y_after, y_before, rtol=0.1, atol=0.1)

  @needs_second_gpu
  @unittest.skipIf(not_support_multi_device(), "no multi")
  def test_multi_gpu(self):
    GPUS = tuple(f"{Device.DEFAULT}:{i}" for i in range(2))
    fp8_layer = FP8Linear(in_dim, out_dim)
    normal_layer = Linear(in_dim, out_dim)
    weight = Tensor.randn(out_dim, in_dim, dtype=dtypes.float32) * 0.2
    bias = Tensor.randn(out_dim, dtype=dtypes.float32) * 0.2
    fp8_layer.weight.assign(weight)
    fp8_layer.bias.assign(bias)
    normal_layer.weight.assign(weight)
    normal_layer.bias.assign(bias)
    fp8_layer.weight.to_(GPUS)
    fp8_layer.bias.to_(GPUS)
    normal_layer.weight.to_(GPUS)
    normal_layer.bias.to_(GPUS)
    x = Tensor.randn(BS*2, in_dim, dtype=dtypes.float32) * 0.2
    x_sharded = x.detach()
    x = x.shard_(GPUS, axis=0)
    y_normal = normal_layer(x).realize()
    x_sharded.shard_(GPUS, axis=0)
    y_fp8 = fp8_layer(x_sharded).realize()
    np.testing.assert_allclose(y_fp8.numpy(), y_normal.numpy(), rtol=0.1, atol=0.1)

if __name__ == '__main__':
  unittest.main()
