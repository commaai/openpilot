import math
from typing import Optional, Union, Tuple
from tinygrad.tensor import Tensor
from tinygrad.helpers import prod, all_int

class BatchNorm2d:
  def __init__(self, sz, eps=1e-5, affine=True, track_running_stats=True, momentum=0.1):
    self.eps, self.track_running_stats, self.momentum = eps, track_running_stats, momentum

    if affine: self.weight, self.bias = Tensor.ones(sz), Tensor.zeros(sz)
    else: self.weight, self.bias = None, None

    self.running_mean, self.running_var = Tensor.zeros(sz, requires_grad=False), Tensor.ones(sz, requires_grad=False)
    self.num_batches_tracked = Tensor.zeros(1, requires_grad=False)

  def __call__(self, x:Tensor):
    if Tensor.training:
      # This requires two full memory accesses to x
      # https://github.com/pytorch/pytorch/blob/c618dc13d2aa23625cb0d7ada694137532a4fa33/aten/src/ATen/native/cuda/Normalization.cuh
      # There's "online" algorithms that fix this, like https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_Online_algorithm
      batch_mean = x.mean(axis=(0,2,3))
      y = (x - batch_mean.reshape(shape=[1, -1, 1, 1]))
      batch_var = (y*y).mean(axis=(0,2,3))
      batch_invstd = batch_var.add(self.eps).pow(-0.5)

      # NOTE: wow, this is done all throughout training in most PyTorch models
      if self.track_running_stats:
        self.running_mean.assign((1 - self.momentum) * self.running_mean + self.momentum * batch_mean.detach())
        self.running_var.assign((1 - self.momentum) * self.running_var + self.momentum * prod(y.shape)/(prod(y.shape) - y.shape[1]) * batch_var.detach() )
        self.num_batches_tracked += 1
    else:
      batch_mean = self.running_mean
      # NOTE: this can be precomputed for static inference. we expand it here so it fuses
      batch_invstd = self.running_var.reshape(1, -1, 1, 1).expand(x.shape).add(self.eps).rsqrt()

    return x.batchnorm(self.weight, self.bias, batch_mean, batch_invstd)

# TODO: these Conv lines are terrible
def Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
  return Conv2d(in_channels, out_channels, (kernel_size,), stride, padding, dilation, groups, bias)

class Conv2d:
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
    self.stride, self.padding, self.dilation, self.groups = stride, padding, dilation, groups
    self.weight = self.initialize_weight(out_channels, in_channels, groups)
    assert all_int(self.weight.shape), "does not support symbolic shape"
    bound = 1 / math.sqrt(prod(self.weight.shape[1:]))
    self.bias = Tensor.uniform(out_channels, low=-bound, high=bound) if bias else None

  def __call__(self, x:Tensor):
    return x.conv2d(self.weight, self.bias, padding=self.padding, stride=self.stride, dilation=self.dilation, groups=self.groups)

  def initialize_weight(self, out_channels, in_channels, groups): return Tensor.kaiming_uniform(out_channels, in_channels//groups, *self.kernel_size, a=math.sqrt(5))

def ConvTranspose1d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1, groups=1, bias=True):
  return ConvTranspose2d(in_channels, out_channels, (kernel_size,), stride, padding, output_padding, dilation, groups, bias)

class ConvTranspose2d(Conv2d):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1, groups=1, bias=True):
    super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
    self.output_padding = output_padding

  def __call__(self, x:Tensor):
    return x.conv_transpose2d(self.weight, self.bias, padding=self.padding, output_padding=self.output_padding, stride=self.stride, dilation=self.dilation, groups=self.groups)

  def initialize_weight(self, out_channels, in_channels, groups): return Tensor.kaiming_uniform(in_channels, out_channels//groups, *self.kernel_size, a=math.sqrt(5))

class Linear:
  def __init__(self, in_features, out_features, bias=True):
    self.weight = Tensor.kaiming_uniform(out_features, in_features, a=math.sqrt(5))
    # TODO: remove this once we can represent Tensor with int shape in typing
    assert isinstance(self.weight.shape[1], int), "does not support symbolic shape"
    bound = 1 / math.sqrt(self.weight.shape[1])
    self.bias = Tensor.uniform(out_features, low=-bound, high=bound) if bias else None

  def __call__(self, x:Tensor):
    return x.linear(self.weight.transpose(), self.bias)

class GroupNorm:
  def __init__(self, num_groups:int, num_channels:int, eps:float=1e-5, affine:bool=True):
    self.num_groups, self.num_channels, self.eps = num_groups, num_channels, eps
    self.weight: Optional[Tensor] = Tensor.ones(num_channels) if affine else None
    self.bias: Optional[Tensor] = Tensor.zeros(num_channels) if affine else None

  def __call__(self, x:Tensor):
    # reshape for layernorm to work as group norm
    # subtract mean and divide stddev
    x = x.reshape(x.shape[0], self.num_groups, -1).layernorm(eps=self.eps).reshape(x.shape)

    if self.weight is None or self.bias is None: return x
    # elementwise_affine on channels
    return x * self.weight.reshape(1, -1, *[1] * (len(x.shape)-2)) + self.bias.reshape(1, -1, *[1] * (len(x.shape)-2))

class InstanceNorm:
  def __init__(self, num_features:int, eps:float=1e-5, affine:bool=True):
    self.num_features, self.eps = num_features, eps
    self.weight: Optional[Tensor] = Tensor.ones(num_features) if affine else None
    self.bias: Optional[Tensor] = Tensor.zeros(num_features) if affine else None

  def __call__(self, x:Tensor):
    x = x.reshape(x.shape[0], self.num_features, -1).layernorm(eps=self.eps).reshape(x.shape)
    if self.weight is None or self.bias is None: return x
    return x * self.weight.reshape(1, -1, *[1] * (len(x.shape)-2)) + self.bias.reshape(1, -1, *[1] * (len(x.shape)-2))

class LayerNorm:
  def __init__(self, normalized_shape:Union[int, Tuple[int, ...]], eps:float=1e-5, elementwise_affine:bool=True):
    self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
    self.axis, self.eps, self.elementwise_affine = tuple(-1-i for i in range(len(self.normalized_shape))), eps, elementwise_affine
    self.weight, self.bias = (Tensor.ones(*self.normalized_shape), Tensor.zeros(*self.normalized_shape)) if elementwise_affine else (None, None)

  def __call__(self, x:Tensor):
    assert self.normalized_shape == x.shape[-len(self.normalized_shape):], f"last dimensions of {x.shape} must match {self.normalized_shape}"
    x = x.layernorm(eps=self.eps, axis=self.axis)
    if not self.elementwise_affine: return x
    return x * self.weight + self.bias

class LayerNorm2d(LayerNorm):
  def __call__(self, x): return super().__call__(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

class Embedding:
  def __init__(self, vocab_size:int, embed_size:int):
    self.vocab_size = vocab_size
    self.weight = Tensor.glorot_uniform(vocab_size, embed_size)

  def __call__(self, idx:Tensor) -> Tensor:
    if not hasattr(self, 'vocab_counter'): self.vocab_counter = Tensor.arange(self.vocab_size, requires_grad=False).reshape(1, 1, self.vocab_size)
    return (self.vocab_counter == idx.unsqueeze(2)).expand(*idx.shape, self.vocab_size) @ self.weight
