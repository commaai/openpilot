#!/usr/bin/env python
import unittest
import numpy as np
import torch
from tinygrad import Tensor, Device, TinyJit
from tinygrad.uop.ops import Ops
from tinygrad.helpers import GlobalCounters, CI, Context
from tinygrad.nn import Conv1d, ConvTranspose1d, Conv2d, ConvTranspose2d, Linear, Embedding
from tinygrad.nn import BatchNorm, LayerNorm, LayerNorm2d, GroupNorm, InstanceNorm, RMSNorm, LSTMCell
from tinygrad.nn.state import load_state_dict
from tinygrad.engine.realize import run_schedule
from test.helpers import not_support_multi_device

@unittest.skipIf(CI and Device.DEFAULT in {"CUDA", "NV"}, "slow")
class TestNN(unittest.TestCase):
  def test_batchnorm2d(self, training=False, threed=False, track_running_stats=True):
    with Tensor.train(training):
      szs = [4, 8, 16, 32]
      for sz in szs:
        # create in tinygrad
        bn = BatchNorm(sz, eps=1e-5, track_running_stats=track_running_stats)
        bn.weight = Tensor.randn(sz)
        bn.bias = Tensor.randn(sz)
        if track_running_stats:
          bn.running_mean = Tensor.randn(sz)
          bn.running_var = Tensor.randn(sz)
          bn.running_var.numpy()[bn.running_var.numpy() < 0] = 0

        # create in torch
        with torch.no_grad():
          if threed:
            tbn = torch.nn.BatchNorm3d(sz, track_running_stats=track_running_stats).eval()
          else:
            tbn = torch.nn.BatchNorm2d(sz, track_running_stats=track_running_stats).eval()
          tbn.training = training
          tbn.weight[:] = torch.tensor(bn.weight.numpy())
          tbn.bias[:] = torch.tensor(bn.bias.numpy())
          if track_running_stats:
            tbn.running_mean[:] = torch.tensor(bn.running_mean.numpy())
            tbn.running_var[:] = torch.tensor(bn.running_var.numpy())

        if track_running_stats:
          np.testing.assert_allclose(bn.running_mean.numpy(), tbn.running_mean.detach().numpy(), rtol=1e-5, atol=1e-6)
          np.testing.assert_allclose(bn.running_var.numpy(), tbn.running_var.detach().numpy(), rtol=1e-5, atol=1e-6)

        # trial
        if threed:
          inn = Tensor.randn(2, sz, 3, 3, 3)
        else:
          inn = Tensor.randn(2, sz, 3, 3)

        # in tinygrad
        outt = bn(inn)

        # in torch
        toutt = tbn(torch.tensor(inn.numpy()))

        # close
        np.testing.assert_allclose(outt.numpy(), toutt.detach().numpy(), rtol=5e-4, atol=1e-6)
        if track_running_stats:
          np.testing.assert_allclose(bn.running_mean.numpy(), tbn.running_mean.detach().numpy(), rtol=1e-5, atol=1e-6)
          np.testing.assert_allclose(bn.running_var.numpy(), tbn.running_var.detach().numpy(), rtol=1e-5, atol=1e-6)

  def test_batchnorm2d_training(self): self.test_batchnorm2d(True, False, True)
  def test_batchnorm2d_no_running_stats(self): self.test_batchnorm2d(False, False, False)
  def test_batchnorm2d_training_no_running_stats(self): self.test_batchnorm2d(True, False, False)
  def test_batchnorm3d(self): self.test_batchnorm2d(False, True, True)
  def test_batchnorm3d_training(self): self.test_batchnorm2d(True, True, True)
  def test_batchnorm3d_no_running_stats(self): self.test_batchnorm2d(False, True, False)
  def test_batchnorm3d_training_no_running_stats(self): self.test_batchnorm2d(True, True, False)

  def test_batchnorm_axis(self):
    sz = (2, 4, 3, 2, 2)
    x = Tensor.randn(sz)
    weight = Tensor.randn(2, 3)
    bias = Tensor.randn(2, 3)
    mean = Tensor.randn(2, 3)
    invstd = Tensor.randn(2, 3)
    a = (x.batchnorm(weight, bias, mean, invstd, axis=(0, 2))
         .permute(1, 0, 2, 3, 4).reshape(4, 6, 2, 2))
    b = (x.permute(1, 0, 2, 3, 4).reshape(4, 6, 2, 2)
         .batchnorm(weight.flatten(), bias.flatten(), mean.flatten(), invstd.flatten()))
    t_x = torch.tensor(x.permute(1, 0, 2, 3, 4).reshape(4, 6, 2, 2).numpy())
    t_weight, t_bias = torch.tensor(weight.flatten().numpy()), torch.tensor(bias.flatten().numpy())
    t_mean, t_invstd = torch.tensor(mean.flatten().numpy()), torch.tensor(invstd.flatten().numpy())
    torch.nn.functional.batch_norm(t_x, t_mean, 1.0 / t_invstd**2, t_weight, t_bias)

    np.testing.assert_allclose(a.numpy(), b.numpy())

  def test_linear(self):
    def _test_linear(x, in_dim, out_dim):
      # create in tinygrad
      model = Linear(in_dim, out_dim)
      z = model(x)

      # create in torch
      with torch.no_grad():
        torch_layer = torch.nn.Linear(in_dim, out_dim).eval()
        torch_layer.weight[:] = torch.tensor(model.weight.numpy(), dtype=torch.float32)
        torch_layer.bias[:] = torch.tensor(model.bias.numpy(), dtype=torch.float32)
        torch_x = torch.tensor(x.numpy(), dtype=torch.float32)
        torch_z = torch_layer(torch_x)

      # test
      np.testing.assert_allclose(z.numpy(), torch_z.detach().numpy(), atol=5e-4, rtol=1e-5)

    BS, T, in_dim, out_dim = 4, 2, 8, 16
    _test_linear(Tensor.randn(BS, in_dim), in_dim, out_dim)
    _test_linear(Tensor.randn(BS, T, in_dim), in_dim, out_dim) # test with more dims

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

  def test_conv1d(self): self._test_conv(Conv1d, torch.nn.Conv1d, BS=4, C1=16, DIMS=[224//4], C2=64, K=7, S=2, P=1)
  def test_conv2d(self): self._test_conv(Conv2d, torch.nn.Conv2d, BS=4, C1=16, DIMS=[224//4, 224//4], C2=64, K=7, S=2, P=1)

  def test_conv1d_same_padding(self):
    self._test_conv(Conv1d, torch.nn.Conv1d, BS=8, C1=3, DIMS=[32], C2=16, K=3, S=1, P='same')
  def test_conv2d_same_padding_odd_input(self):
    self._test_conv(Conv2d, torch.nn.Conv2d, BS=16, C1=16, DIMS=[29, 31], C2=32, K=5, S=1, P='same')
  def test_conv2d_same_padding_large_kernel(self):
    self._test_conv(Conv2d, torch.nn.Conv2d, BS=16, C1=16, DIMS=[28, 33], C2=32, K=9, S=1, P='same')
  def test_conv2d_same_padding_with_dilation(self):
    self._test_conv(Conv2d, torch.nn.Conv2d, BS=16, C1=3, DIMS=[28, 28], C2=32, K=3, S=1, P='same', D=3)

  def test_conv2d_same_padding_invalid_stride(self):
    self.assertRaises(ValueError, Conv2d, in_channels=16, out_channels=32, kernel_size=2, stride=2, padding='same')
  def test_conv2d_same_padding_invalid_padding_str(self):
    self.assertRaises(ValueError, Conv2d, in_channels=16, out_channels=32, kernel_size=2, stride=1, padding='not_same')

  @unittest.skip("Takes too long to compile for Compiled backends")
  def test_conv2d_winograd(self):
    BS, C1, H, W = 2, 8, 16, 16
    C2, K, S, P = 8, 3, 1, 1

    # create in tinygrad
    layer = Conv2d(C1, C2, kernel_size=K, stride=S, padding=P)
    layer.weight.requires_grad = True
    layer.bias.requires_grad = True

    # create in torch
    torch_layer = torch.nn.Conv2d(C1, C2, kernel_size=K, stride=S, padding=P).eval()
    torch_layer.weight = torch.nn.Parameter(torch.tensor(layer.weight.numpy(), dtype=torch.float32))
    torch_layer.bias = torch.nn.Parameter(torch.tensor(layer.bias.numpy(), dtype=torch.float32))

    # test
    x = Tensor.uniform(BS, C1, H, W, requires_grad=True)

    with Context(WINO=1):
      z = layer(x)

    m = z.mean()
    m.backward()

    torch_x = torch.tensor(x.numpy(), requires_grad=True)
    torch_z = torch_layer(torch_x)
    np.testing.assert_allclose(z.numpy(), torch_z.detach().numpy(), atol=5e-4, rtol=1e-5)

    gw = layer.weight.grad.realize()
    gb = layer.bias.grad.realize()
    gx = x.grad.realize()

    torch_z.mean().backward()
    np.testing.assert_allclose(gw.numpy(), torch_layer.weight.grad.numpy(), atol=5e-4, rtol=1e-5)
    np.testing.assert_allclose(gb.numpy(), torch_layer.bias.grad.numpy(), atol=5e-4, rtol=1e-5)
    np.testing.assert_allclose(gx.numpy(), torch_x.grad.numpy(), atol=5e-4, rtol=1e-5)

  def test_conv_transpose1d(self):
    self._test_conv(ConvTranspose1d, torch.nn.ConvTranspose1d, BS=4, C1=16, DIMS=[224//4], C2=64, K=7, S=2, P=1)
  def test_conv_transpose2d(self):
    self._test_conv(ConvTranspose2d, torch.nn.ConvTranspose2d, BS=4, C1=16, DIMS=[224//4, 224//4], C2=64, K=7, S=2, P=1)

  def test_groupnorm(self):
    BS, H, W, C, G = 20, 10, 10, 6, 3

    # create in torch
    torch_layer = torch.nn.GroupNorm(G, C).eval()

    # create in tinygrad
    layer = GroupNorm(G, C)
    layer.weight = Tensor(torch_layer.weight.detach().numpy(), requires_grad=True)
    layer.bias = Tensor(torch_layer.bias.detach().numpy(), requires_grad=True)

    for _ in range(10):
      # forward
      x = Tensor.randn(BS, C, H, W, requires_grad=True)
      z = layer(x)
      z.sum().backward()

      torch_x = torch.tensor(x.numpy(), requires_grad=True)
      torch_z = torch_layer(torch_x)
      torch_z.sum().backward()

      np.testing.assert_allclose(z.numpy(), torch_z.detach().numpy(), atol=5e-6, rtol=5e-6)
      np.testing.assert_allclose(x.grad.numpy(), torch_x.grad.detach().numpy(), atol=5e-4, rtol=5e-4)
      np.testing.assert_allclose(layer.weight.grad.numpy(), torch_layer.weight.grad.detach().numpy(), atol=5e-4, rtol=5e-4)
      np.testing.assert_allclose(layer.bias.grad.numpy(), torch_layer.bias.grad.detach().numpy(), atol=5e-4, rtol=5e-4)

  def test_layernorm(self):
    N, C, H, W = 20, 5, 10, 10

    # create in torch
    torch_layer = torch.nn.LayerNorm([H, W]).eval()

    # create in tinygrad
    layer = LayerNorm([H, W])
    layer.weight = Tensor(torch_layer.weight.detach().numpy(), requires_grad=True)
    layer.bias = Tensor(torch_layer.bias.detach().numpy(), requires_grad=True)

    for _ in range(10):
      # forward
      x = Tensor.randn(N, C, H, W, requires_grad=True)
      z = layer(x)
      z.sum().backward()

      torch_x = torch.tensor(x.numpy(), requires_grad=True)
      torch_z = torch_layer(torch_x)
      torch_z.sum().backward()

      np.testing.assert_allclose(z.numpy(), torch_z.detach().numpy(), atol=5e-6, rtol=5e-6)
      np.testing.assert_allclose(x.grad.numpy(), torch_x.grad.detach().numpy(), atol=5e-4, rtol=5e-4)
      np.testing.assert_allclose(layer.weight.grad.numpy(), torch_layer.weight.grad.detach().numpy(), atol=5e-4, rtol=5e-4)
      np.testing.assert_allclose(layer.bias.grad.numpy(), torch_layer.bias.grad.detach().numpy(), atol=5e-4, rtol=5e-4)

  def test_layernorm_2d(self):
    N, C, H, W = 20, 5, 10, 10

    # create in torch
    torch_layer = torch.nn.LayerNorm([C]).eval()

    # create in tinygrad
    layer = LayerNorm2d(C)
    layer.weight = Tensor(torch_layer.weight.detach().numpy(), requires_grad=True)
    layer.bias = Tensor(torch_layer.bias.detach().numpy(), requires_grad=True)

    for _ in range(10):
      # forward
      x = Tensor.randn(N, C, H, W, requires_grad=True)
      z = layer(x)
      z.sum().backward()

      torch_x = torch.tensor(x.numpy(), requires_grad=True)
      torch_z = torch_layer(torch_x.permute(0,2,3,1)).permute(0,3,1,2)
      torch_z.sum().backward()

      np.testing.assert_allclose(z.numpy(), torch_z.detach().numpy(), atol=5e-6, rtol=5e-6)
      np.testing.assert_allclose(x.grad.numpy(), torch_x.grad.detach().numpy(), atol=5e-4, rtol=5e-4)
      np.testing.assert_allclose(layer.weight.grad.numpy(), torch_layer.weight.grad.detach().numpy(), atol=5e-4, rtol=5e-4)
      np.testing.assert_allclose(layer.bias.grad.numpy(), torch_layer.bias.grad.detach().numpy(), atol=5e-4, rtol=5e-4)

  def test_instancenorm_2d(self):
    N, C, H, W = 20, 10, 10, 10

    # create in torch
    torch_layer = torch.nn.InstanceNorm2d(C, affine=True).eval()

    # create in tinygrad
    layer = InstanceNorm(C)
    layer.weight = Tensor(torch_layer.weight.detach().numpy(), requires_grad=True)
    layer.bias = Tensor(torch_layer.bias.detach().numpy(), requires_grad=True)

    for _ in range(10):
      # forward
      x = Tensor.randn(N, C, H, W, requires_grad=True)
      z = layer(x)
      z.sum().backward()

      torch_x = torch.tensor(x.numpy(), requires_grad=True)
      torch_z = torch_layer(torch_x)
      torch_z.sum().backward()

      np.testing.assert_allclose(z.numpy(), torch_z.detach().numpy(), atol=5e-6, rtol=5e-6)
      np.testing.assert_allclose(x.grad.numpy(), torch_x.grad.detach().numpy(), atol=1e-3, rtol=1e-3)
      np.testing.assert_allclose(layer.weight.grad.numpy(), torch_layer.weight.grad.detach().numpy(), atol=1e-3, rtol=1e-3)
      np.testing.assert_allclose(layer.bias.grad.numpy(), torch_layer.bias.grad.detach().numpy(), atol=1e-3, rtol=1e-3)

  def test_instancenorm_3d(self):
    N, C, D, H, W = 20, 10, 10, 10, 10

    # create in torch
    torch_layer = torch.nn.InstanceNorm3d(C, affine=True).eval()

    # create in tinygrad
    layer = InstanceNorm(C)
    layer.weight = Tensor(torch_layer.weight.detach().numpy(), requires_grad=True)
    layer.bias = Tensor(torch_layer.bias.detach().numpy(), requires_grad=True)

    for _ in range(10):
      # forward
      x = Tensor.randn(N, C, D, H, W, requires_grad=True)
      z = layer(x)
      z.sum().backward()

      torch_x = torch.tensor(x.numpy(), requires_grad=True)
      torch_z = torch_layer(torch_x)
      torch_z.sum().backward()

      np.testing.assert_allclose(z.numpy(), torch_z.detach().numpy(), atol=5e-6, rtol=5e-6)
      np.testing.assert_allclose(x.grad.numpy(), torch_x.grad.detach().numpy(), atol=1e-3, rtol=1e-3)
      np.testing.assert_allclose(layer.weight.grad.numpy(), torch_layer.weight.grad.detach().numpy(), atol=2e-3, rtol=1e-3)
      np.testing.assert_allclose(layer.bias.grad.numpy(), torch_layer.bias.grad.detach().numpy(), atol=1e-3, rtol=1e-3)

  def test_rmsnorm(self):
    class TorchRMSNorm(torch.nn.Module):
      # https://github.com/meta-llama/llama/blob/be327c427cc5e89cc1d3ab3d3fec4484df771245/llama/model.py#L34C1-L77C36
      def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.weight = torch.nn.Parameter(torch.ones(dim)) if elementwise_affine else None

      def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

      def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output if self.weight is None else output * self.weight

    B, T, embed_size = 4, 10, 20
    torch_layer = TorchRMSNorm(embed_size)
    layer = RMSNorm(embed_size)
    layer.weight.requires_grad = True

    for _ in range(10):
      # forward
      x = Tensor.randn(B, T, embed_size, requires_grad=True)
      z = layer(x)
      z.sum().backward()

      torch_x = torch.tensor(x.numpy(), requires_grad=True)
      torch_z = torch_layer(torch_x)
      torch_z.sum().backward()

      np.testing.assert_allclose(z.numpy(), torch_z.detach().numpy(), atol=5e-6, rtol=5e-6)
      np.testing.assert_allclose(x.grad.numpy(), torch_x.grad.detach().numpy(), atol=1e-3, rtol=1e-3)
      np.testing.assert_allclose(layer.weight.grad.numpy(), torch_layer.weight.grad.detach().numpy(), atol=2e-3, rtol=1e-3)

    torch_layer = TorchRMSNorm(embed_size, elementwise_affine=False)
    layer = RMSNorm(embed_size, elementwise_affine=False)

    for _ in range(10):
      # forward
      x = Tensor.randn(B, T, embed_size, requires_grad=True)
      z = layer(x)
      z.sum().backward()

      torch_x = torch.tensor(x.numpy(), requires_grad=True)
      torch_z = torch_layer(torch_x)
      torch_z.sum().backward()

      np.testing.assert_allclose(z.numpy(), torch_z.detach().numpy(), atol=5e-6, rtol=5e-6)
      np.testing.assert_allclose(x.grad.numpy(), torch_x.grad.detach().numpy(), atol=1e-3, rtol=1e-3)

  def test_embedding(self):
    B, T, embed_size, vocab_size = 4, 10, 20, 28

    # create in tinygrad
    layer = Embedding(vocab_size, embed_size)

    with torch.no_grad():
      torch_layer = torch.nn.Embedding(vocab_size, embed_size).eval()
      torch_layer.weight[:] = torch.tensor(layer.weight.numpy(), dtype=torch.float32)

    # test
    x = Tensor(np.random.randint(0, vocab_size, (B, T), dtype=np.int32))
    z = layer(x)
    torch_x = torch.tensor(x.numpy())
    torch_z = torch_layer(torch_x)
    np.testing.assert_allclose(z.numpy(), torch_z.detach().numpy(), atol=1e-8, rtol=1e-8)

    # test with empty input length
    x = Tensor(np.random.randint(0, vocab_size, (B, 0), dtype=np.int32))
    z = layer(x)
    torch_x = torch.tensor(x.numpy())
    torch_z = torch_layer(torch_x)
    np.testing.assert_allclose(z.numpy(), torch_z.detach().numpy(), atol=1e-8, rtol=1e-8)

    # test with jit enabled
    @TinyJit
    def layer_jit(x):
      return layer(x).realize()

    for _ in range(3):
      x = Tensor(np.random.randint(0, vocab_size, (B, T), dtype=np.int32))
      z = layer_jit(x)
      torch_x = torch.tensor(x.numpy())
      torch_z = torch_layer(torch_x)
      np.testing.assert_allclose(z.numpy(), torch_z.detach().numpy(), atol=1e-8, rtol=1e-8)

  def test_embedding_one_kernel(self, ops=612000, kcount=2):
    GlobalCounters.reset()
    layer = Embedding(20, 30)
    layer.weight = Tensor.zeros_like(layer.weight).contiguous()
    a = Tensor([[1, 5, 9, 11],
                [12, 19, 8, 1]])
    result = layer(a)
    schedule = result.schedule()
    self.assertEqual(len([item for item in schedule if item.ast.op is Ops.SINK]), kcount, "first run realizes weight and embedding")
    run_schedule(schedule)

    b = Tensor([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
    result = layer(b)
    schedule = result.schedule()
    self.assertEqual(1, len([item for item in schedule if item.ast.op is Ops.SINK]), "second run realizes embedding only")
    run_schedule(schedule)
    print(f"Embedding used {GlobalCounters.global_ops} ops")
    self.assertLessEqual(GlobalCounters.global_ops, ops)

  # TODO: fused with opts uses more ops
  def test_embedding_one_kernel_fused(self):
    with Context(FUSE_ARANGE=1, NOOPT=0):
      self.test_embedding_one_kernel(ops=612_000, kcount=2)

  def test_embedding_one_kernel_fused_noopt(self):
    with Context(FUSE_ARANGE=1, NOOPT=1):
      self.test_embedding_one_kernel(ops=0, kcount=2)

  def test_embedding_shape(self):
    vocab_size, embed_size = 10, 16
    layer = Embedding(vocab_size, embed_size)
    for rank in range(5):
      shp = (1,) * rank
      a = Tensor([3]).reshape(shp)
      result = layer(a)
      self.assertEqual(result.shape, shp + (embed_size,))

  def test_embedding_regression(self):
    # used to fail bounds check
    with Context(FUSE_ARANGE=1):
      embedding = Embedding(100, 1024)
      input_ids = Tensor.empty(16, 16)
      embedding(input_ids).realize()

  def test_load_state_dict(self):
    layer = Conv2d(3, 5, kernel_size=3)

    state_dict = {
      'weight': Tensor.randn(5, 3, 3, 3),
      'bias': Tensor.randn(5),
    }
    load_state_dict(layer, state_dict)

    np.testing.assert_allclose(layer.weight.numpy(), state_dict['weight'].numpy())
    np.testing.assert_allclose(layer.bias.numpy(), state_dict['bias'].numpy())

  @unittest.skipIf(not_support_multi_device(), "no multi")
  def test_load_state_dict_sharded_model(self):
    devices = (f"{Device.DEFAULT}:1", f"{Device.DEFAULT}:2", f"{Device.DEFAULT}:3")

    layer = Conv2d(3, 5, kernel_size=3)
    layer.weight.shard_(devices, 3)
    layer.bias.shard_(devices, None)
    state_dict = {
      'weight': Tensor.randn(5, 3, 3, 3).realize(),
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

  @unittest.skipIf(not_support_multi_device, "no multi")
  def test_load_state_dict_sharded_dict(self):
    devices = (f"{Device.DEFAULT}:1", f"{Device.DEFAULT}:2", f"{Device.DEFAULT}:3")

    layer = Conv2d(3, 5, kernel_size=3)
    state_dict = {
      'weight': Tensor.randn(5, 3, 3, 3).shard(devices, 3),
      'bias': Tensor.randn(5).shard(devices, None),
    }
    load_state_dict(layer, state_dict)

    # NOTE: model is not sharded, still not sharded after load_state_dict
    self.assertEqual(layer.weight.device, Device.DEFAULT)
    self.assertEqual(layer.bias.device, Device.DEFAULT)
    np.testing.assert_allclose(layer.weight.numpy(), state_dict['weight'].numpy())
    np.testing.assert_allclose(layer.bias.numpy(), state_dict['bias'].numpy())

  @unittest.skipIf(not_support_multi_device(), "no multi")
  def test_load_state_dict_sharded_model_dict_same_axis(self):
    devices = (f"{Device.DEFAULT}:1", f"{Device.DEFAULT}:2", f"{Device.DEFAULT}:3")

    layer = Conv2d(3, 5, kernel_size=3)
    layer.weight.shard_(devices, 3)
    layer.bias.shard_(devices, None)

    state_dict = {
      'weight': Tensor.randn(5, 3, 3, 3).shard(devices, 3),
      'bias': Tensor.randn(5).shard(devices, None),
    }
    load_state_dict(layer, state_dict)

    self.assertEqual(layer.weight.device, devices)
    self.assertEqual(layer.weight.uop.axis, 3)
    self.assertEqual(layer.bias.device, devices)
    self.assertEqual(layer.bias.uop.axis, None)
    np.testing.assert_allclose(layer.weight.numpy(), state_dict['weight'].numpy())
    np.testing.assert_allclose(layer.bias.numpy(), state_dict['bias'].numpy())

  @unittest.skipIf(not_support_multi_device, "no multi")
  def test_load_state_dict_sharded_model_dict_different_axis(self):
    devices = (f"{Device.DEFAULT}:1", f"{Device.DEFAULT}:2", f"{Device.DEFAULT}:3")
    devices5 = (f"{Device.DEFAULT}:1", f"{Device.DEFAULT}:2", f"{Device.DEFAULT}:3", f"{Device.DEFAULT}:4", f"{Device.DEFAULT}:5")

    layer = Conv2d(3, 5, kernel_size=3)
    layer.weight.shard_(devices, 3)
    layer.bias.shard_(devices, None)

    # different shard axis
    state_dict = {
      'weight': Tensor.randn(5, 3, 3, 3).shard(devices, None),
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

  def test_load_state_dict_shape_mismatch(self):
    d1, d2 = 2, 4
    layer = Linear(d1, d1, bias=False)
    state_dict = {'weight': Tensor.randn(d2, d2)}
    with self.assertRaisesRegex(ValueError, r'Shape mismatch in layer `weight`: Expected shape \(2, 2\), but found \(4, 4\) in state dict.'):
      load_state_dict(layer, state_dict)

  def test_lstm_cell(self):
    layer = LSTMCell(32, 16)
    with torch.no_grad():
      torch_layer = torch.nn.LSTMCell(32, 16)
      layer.weight_hh.assign(torch_layer.weight_hh.numpy())
      layer.weight_ih.assign(torch_layer.weight_ih.numpy())
      layer.bias_hh.assign(torch_layer.bias_hh.numpy())
      layer.bias_ih.assign(torch_layer.bias_ih.numpy())

      inp = Tensor.randn(1, 32)
      out_h, out_c = layer(inp)
      torch_out_h, torch_out_c = torch_layer(torch.tensor(inp.numpy()))
      np.testing.assert_allclose(out_h.numpy(), torch_out_h.numpy(), atol=1e-6)
      np.testing.assert_allclose(out_c.numpy(), torch_out_c.numpy(), atol=1e-6)

      out_h, out_c = layer(inp, (out_h, out_c))
      torch_out_h, torch_out_c = torch_layer(torch.tensor(inp.numpy()), (torch_out_h, torch_out_c))
      np.testing.assert_allclose(out_h.numpy(), torch_out_h.numpy(), atol=1e-6)
      np.testing.assert_allclose(out_c.numpy(), torch_out_c.numpy(), atol=1e-6)

  def test_lstm_cell_no_bias(self):
    layer = LSTMCell(32, 16, bias=False)
    inp = Tensor.randn(1, 32)
    out_h, out_c = layer(inp)
    out_h.realize()
    out_c.realize()
    h = Tensor.randn(1, 16)
    c = Tensor.randn(1, 16)
    out_h, out_c = layer(inp, (h, c))
    out_h.realize()
    out_c.realize()
    assert layer.bias_hh is None
    assert layer.bias_ih is None

if __name__ == '__main__':
  unittest.main()
