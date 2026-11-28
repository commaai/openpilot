#!/usr/bin/env python
import unittest
import numpy as np
import torch
from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import Context
from tinygrad.nn import BatchNorm, ConvTranspose1d, Conv2d, LSTMCell
from tinygrad.nn.state import load_state_dict
from test.helpers import needs_devices

class TestNN(unittest.TestCase):
  def test_batchnorm_mixed_precision_running_stats(self):
    with Context(DEFAULT_FLOAT=dtypes.float16, TRAINING=1):
      bn = BatchNorm(1, momentum=0.5)
      bn(Tensor([[1.], [3.]], dtype=dtypes.float32))
      self.assertEqual(bn.running_mean.item(), 1.0)
      self.assertEqual(bn.running_var.item(), 1.5)

  def test_batchnorm_axis(self):
    x = Tensor.randn(2, 2, 2)
    weight = Tensor.randn(2, 2)
    bias = Tensor.randn(2, 2)
    mean = Tensor.randn(2, 2)
    invstd = Tensor.randn(2, 2)
    a = (x.batchnorm(weight, bias, mean, invstd, axis=(0, 2))
         .permute(1, 0, 2).reshape(2, 4))
    b = (x.permute(1, 0, 2).reshape(2, 4)
         .batchnorm(weight.flatten(), bias.flatten(), mean.flatten(), invstd.flatten()))
    np.testing.assert_allclose(a.numpy(), b.numpy())

  def _test_conv(self, tiny_conv, torch_conv, BS, C1, DIMS, C2, K, S, P, D=1):
    # create in tinygrad
    layer = tiny_conv(C1, C2, kernel_size=K, stride=S, padding=P, dilation=D)

    # create in torch
    with torch.no_grad():
      torch_layer = torch_conv(C1, C2, kernel_size=K, stride=S, padding=P, dilation=D).eval()
      torch_layer.weight[:] = torch.tensor(layer.weight.numpy(), dtype=torch.float32)
      torch_layer.bias[:] = torch.tensor(layer.bias.numpy(), dtype=torch.float32)

    # test
    x = Tensor.uniform(BS, C1, *DIMS)
    z = layer(x)
    torch_x = torch.tensor(x.numpy())
    torch_z = torch_layer(torch_x)
    np.testing.assert_allclose(z.numpy(), torch_z.detach().numpy(), atol=5e-4, rtol=1e-5)

  def test_conv2d_same_padding_with_dilation(self):
    self._test_conv(Conv2d, torch.nn.Conv2d, BS=1, C1=1, DIMS=[3, 3], C2=1, K=(1,3), S=1, P='same', D=(1,2))

  def test_conv_transpose1d(self):
    self._test_conv(ConvTranspose1d, torch.nn.ConvTranspose1d, BS=1, C1=1, DIMS=[3], C2=1, K=3, S=2, P=1)

  @needs_devices(4)
  def test_load_state_dict_sharded_model(self):
    devices = (f"{Device.DEFAULT}:1", f"{Device.DEFAULT}:2", f"{Device.DEFAULT}:3")

    layer = Conv2d(1, 5, kernel_size=3)
    layer.weight.shard_(devices, 3)
    layer.bias.shard_(devices, None)
    state_dict = {
      'weight': Tensor.randn(5, 1, 3, 3).realize(),
      'bias': Tensor.randn(5).realize(),
    }
    load_state_dict(layer, state_dict)

    # sharded model shards the state_dict
    self.assertEqual(layer.weight.device, devices)
    self.assertEqual(layer.weight.uop.axis, 3)
    self.assertEqual(layer.bias.device, devices)
    self.assertEqual(layer.bias.uop.axis, None)
    np.testing.assert_allclose(layer.weight.numpy(), state_dict['weight'].numpy())
    np.testing.assert_allclose(layer.bias.numpy(), state_dict['bias'].numpy())

  @needs_devices(4)
  def test_load_state_dict_sharded_dict(self):
    devices = (f"{Device.DEFAULT}:1", f"{Device.DEFAULT}:2", f"{Device.DEFAULT}:3")

    layer = Conv2d(1, 5, kernel_size=3)
    state_dict = {
      'weight': Tensor.randn(5, 1, 3, 3).shard(devices, 3),
      'bias': Tensor.randn(5).shard(devices, None),
    }
    load_state_dict(layer, state_dict)

    # NOTE: model is not sharded, still not sharded after load_state_dict
    self.assertEqual(layer.weight.device, Device.DEFAULT)
    self.assertEqual(layer.bias.device, Device.DEFAULT)
    np.testing.assert_allclose(layer.weight.numpy(), state_dict['weight'].numpy())
    np.testing.assert_allclose(layer.bias.numpy(), state_dict['bias'].numpy())

  @needs_devices(6)
  def test_load_state_dict_sharded_model_dict_different_axis(self):
    devices = (f"{Device.DEFAULT}:1", f"{Device.DEFAULT}:2", f"{Device.DEFAULT}:3")
    devices5 = (f"{Device.DEFAULT}:1", f"{Device.DEFAULT}:2", f"{Device.DEFAULT}:3", f"{Device.DEFAULT}:4", f"{Device.DEFAULT}:5")

    layer = Conv2d(1, 5, kernel_size=3)
    layer.weight.shard_(devices, 3)
    layer.bias.shard_(devices, None)

    # different shard axis
    state_dict = {
      'weight': Tensor.randn(5, 1, 3, 3).shard(devices, None),
      'bias': Tensor.randn(5).shard(devices5, 0),
    }
    load_state_dict(layer, state_dict)

    # NOTE: model and state_dict shard differently, use the state_dict sharding  # TODO: revisit this?
    self.assertEqual(layer.weight.device, devices)
    self.assertEqual(layer.weight.uop.axis, None)
    self.assertEqual(layer.bias.device, devices5)
    self.assertEqual(layer.bias.uop.axis, 0)
    np.testing.assert_allclose(layer.weight.numpy(), state_dict['weight'].numpy())
    np.testing.assert_allclose(layer.bias.numpy(), state_dict['bias'].numpy())

  def test_lstm_cell(self):
    layer = LSTMCell(1, 1)
    with torch.no_grad():
      torch_layer = torch.nn.LSTMCell(1, 1)
      layer.weight_hh.assign(torch_layer.weight_hh.numpy())
      layer.weight_ih.assign(torch_layer.weight_ih.numpy())
      layer.bias_hh.assign(torch_layer.bias_hh.numpy())
      layer.bias_ih.assign(torch_layer.bias_ih.numpy())

      inp = Tensor.randn(1, 1)
      out_h, out_c = layer(inp)
      torch_out_h, torch_out_c = torch_layer(torch.tensor(inp.numpy()))
      np.testing.assert_allclose(out_h.numpy(), torch_out_h.numpy(), atol=1e-6)
      np.testing.assert_allclose(out_c.numpy(), torch_out_c.numpy(), atol=1e-6)

      out_h, out_c = layer(inp, (out_h, out_c))
      torch_out_h, torch_out_c = torch_layer(torch.tensor(inp.numpy()), (torch_out_h, torch_out_c))
      np.testing.assert_allclose(out_h.numpy(), torch_out_h.numpy(), atol=1e-6)
      np.testing.assert_allclose(out_c.numpy(), torch_out_c.numpy(), atol=1e-6)

if __name__ == '__main__':
  unittest.main()
