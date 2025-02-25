from __future__ import annotations
import math
from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinygrad.device import is_dtype_supported
from tinygrad.helpers import prod, make_tuple, flatten
from tinygrad.nn import optim, state, datasets  # noqa: F401

class BatchNorm:
  """
  Applies Batch Normalization over a 2D or 3D input.

  - Described: https://paperswithcode.com/method/batch-normalization
  - Paper: https://arxiv.org/abs/1502.03167v3

  See: `Tensor.batchnorm`

  ```python exec="true" session="tensor"
  from tinygrad import Tensor, dtypes, nn
  import numpy as np
  np.set_printoptions(precision=4)
  ```

  ```python exec="true" source="above" session="tensor" result="python"
  norm = nn.BatchNorm(3)
  t = Tensor.rand(2, 3, 4, 4)
  print(t.mean().item(), t.std().item())
  ```
  ```python exec="true" source="above" session="tensor" result="python"
  t = norm(t)
  print(t.mean().item(), t.std().item())
  ```
  """
  def __init__(self, sz:int, eps=1e-5, affine=True, track_running_stats=True, momentum=0.1):
    self.eps, self.track_running_stats, self.momentum = eps, track_running_stats, momentum

    self.weight: Tensor|None = Tensor.ones(sz) if affine else None
    self.bias: Tensor|None = Tensor.zeros(sz) if affine else None

    self.num_batches_tracked = Tensor.zeros(1, dtype='long' if is_dtype_supported(dtypes.long) else 'int', requires_grad=False)
    if track_running_stats: self.running_mean, self.running_var = Tensor.zeros(sz, requires_grad=False), Tensor.ones(sz, requires_grad=False)

  def calc_stats(self, x:Tensor) -> tuple[Tensor, Tensor]:
    shape_mask: list[int] = [1, -1, *([1]*(x.ndim-2))]
    if self.track_running_stats and not Tensor.training: return self.running_mean, self.running_var.reshape(shape=shape_mask).expand(x.shape)
    # This requires two full memory accesses to x
    # https://github.com/pytorch/pytorch/blob/c618dc13d2aa23625cb0d7ada694137532a4fa33/aten/src/ATen/native/cuda/Normalization.cuh
    # There's "online" algorithms that fix this, like https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_Online_algorithm
    batch_mean = x.mean(axis=(reduce_axes:=tuple(x for x in range(x.ndim) if x != 1)))
    y = (x - batch_mean.detach().reshape(shape=shape_mask))  # d(var)/d(mean) = 0
    batch_var = (y*y).mean(axis=reduce_axes)
    return batch_mean, batch_var

  def __call__(self, x:Tensor) -> Tensor:
    batch_mean, batch_var = self.calc_stats(x)
    # NOTE: wow, this is done all throughout training in most PyTorch models
    if self.track_running_stats and Tensor.training:
      self.running_mean.assign((1-self.momentum) * self.running_mean + self.momentum * batch_mean.detach())
      self.running_var.assign((1-self.momentum) * self.running_var + self.momentum * x.numel()/(x.numel()-x.shape[1]) * batch_var.detach())
      self.num_batches_tracked += 1
    return x.batchnorm(self.weight, self.bias, batch_mean, batch_var.add(self.eps).rsqrt())
BatchNorm2d = BatchNorm3d = BatchNorm

def Conv1d(in_channels:int, out_channels:int, kernel_size:int, stride=1, padding:int|str=0, dilation=1, groups=1, bias=True) -> Conv2d:
  """
  Applies a 1D convolution over an input signal composed of several input planes.

  See: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d

  ```python exec="true" source="above" session="tensor" result="python"
  conv = nn.Conv1d(1, 1, 3)
  t = Tensor.rand(1, 1, 4)
  print(t.numpy())
  ```
  ```python exec="true" source="above" session="tensor" result="python"
  t = conv(t)
  print(t.numpy())
  ```
  """
  return Conv2d(in_channels, out_channels, (kernel_size,), stride, padding, dilation, groups, bias)

class Conv2d:
  """
  Applies a 2D convolution over an input signal composed of several input planes.

  See: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d

  ```python exec="true" source="above" session="tensor" result="python"
  conv = nn.Conv2d(1, 1, 3)
  t = Tensor.rand(1, 1, 4, 4)
  print(t.numpy())
  ```
  ```python exec="true" source="above" session="tensor" result="python"
  t = conv(t)
  print(t.numpy())
  ```
  """
  def __init__(self, in_channels:int, out_channels:int, kernel_size:int|tuple[int, ...], stride=1, padding:int|tuple[int, ...]|str=0,
               dilation=1, groups=1, bias=True):
    self.kernel_size = make_tuple(kernel_size, 2)
    if isinstance(padding, str):
      if padding.lower() != 'same': raise ValueError(f"Invalid padding string {padding!r}, only 'same' is supported")
      if stride != 1: raise ValueError("padding='same' is not supported for strided convolutions")
      pad = [(d*(k-1)//2, d*(k-1) - d*(k-1)//2) for d,k in zip(make_tuple(dilation, len(self.kernel_size)), self.kernel_size[::-1])]
      padding = tuple(flatten(pad))
    self.stride, self.dilation, self.groups, self.padding = stride, dilation, groups, padding
    scale = 1 / math.sqrt(in_channels * prod(self.kernel_size))
    self.weight = Tensor.uniform(out_channels, in_channels//groups, *self.kernel_size, low=-scale, high=scale)
    self.bias: Tensor|None = Tensor.uniform(out_channels, low=-scale, high=scale) if bias else None

  def __call__(self, x:Tensor) -> Tensor: return x.conv2d(self.weight, self.bias, self.groups, self.stride, self.dilation, self.padding)

def ConvTranspose1d(in_channels:int, out_channels:int, kernel_size:int, stride=1, padding=0, output_padding=0, dilation=1,
                      groups=1, bias=True) -> ConvTranspose2d:
  """
  Applies a 1D transposed convolution operator over an input signal composed of several input planes.

  See: https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d

  ```python exec="true" source="above" session="tensor" result="python"
  conv = nn.ConvTranspose1d(1, 1, 3)
  t = Tensor.rand(1, 1, 4)
  print(t.numpy())
  ```
  ```python exec="true" source="above" session="tensor" result="python"
  t = conv(t)
  print(t.numpy())
  ```
  """
  return ConvTranspose2d(in_channels, out_channels, (kernel_size,), stride, padding, output_padding, dilation, groups, bias)

class ConvTranspose2d(Conv2d):
  """
  Applies a 2D transposed convolution operator over an input image.

  See: https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d

  ```python exec="true" source="above" session="tensor" result="python"
  conv = nn.ConvTranspose2d(1, 1, 3)
  t = Tensor.rand(1, 1, 4, 4)
  print(t.numpy())
  ```
  ```python exec="true" source="above" session="tensor" result="python"
  t = conv(t)
  print(t.numpy())
  ```
  """
  def __init__(self, in_channels:int, out_channels:int, kernel_size:int|tuple[int, ...], stride=1, padding=0, output_padding=0,
                dilation=1, groups=1, bias=True):
    super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
    scale = 1 / math.sqrt(in_channels * prod(self.kernel_size))
    self.weight = Tensor.uniform(in_channels, out_channels//groups, *self.kernel_size, low=-scale, high=scale)
    self.output_padding = output_padding

  def __call__(self, x:Tensor) -> Tensor:
    return x.conv_transpose2d(self.weight, self.bias, self.groups, self.stride, self.dilation, self.padding, self.output_padding)

class Linear:
  """
  Applies a linear transformation to the incoming data.

  See: https://pytorch.org/docs/stable/generated/torch.nn.Linear

  ```python exec="true" source="above" session="tensor" result="python"
  lin = nn.Linear(3, 4)
  t = Tensor.rand(2, 3)
  print(t.numpy())
  ```
  ```python exec="true" source="above" session="tensor" result="python"
  t = lin(t)
  print(t.numpy())
  ```
  """
  def __init__(self, in_features:int, out_features:int, bias=True):
    bound = 1 / math.sqrt(in_features)
    self.weight = Tensor.uniform(out_features, in_features, low=-bound, high=bound)
    self.bias = Tensor.uniform(out_features, low=-bound, high=bound) if bias else None

  def __call__(self, x:Tensor) -> Tensor: return x.linear(self.weight.transpose(), self.bias)

class GroupNorm:
  """
  Applies Group Normalization over a mini-batch of inputs.

  - Described: https://paperswithcode.com/method/group-normalization
  - Paper: https://arxiv.org/abs/1803.08494v3

  ```python exec="true" source="above" session="tensor" result="python"
  norm = nn.GroupNorm(2, 12)
  t = Tensor.rand(2, 12, 4, 4) * 2 + 1
  print(t.mean().item(), t.std().item())
  ```
  ```python exec="true" source="above" session="tensor" result="python"
  t = norm(t)
  print(t.mean().item(), t.std().item())
  ```
  """
  def __init__(self, num_groups:int, num_channels:int, eps=1e-5, affine=True):
    self.num_groups, self.num_channels, self.eps = num_groups, num_channels, eps
    self.weight: Tensor|None = Tensor.ones(num_channels) if affine else None
    self.bias: Tensor|None = Tensor.zeros(num_channels) if affine else None

  def __call__(self, x:Tensor) -> Tensor:
    # reshape for layernorm to work as group norm
    # subtract mean and divide stddev
    x = x.reshape(x.shape[0], self.num_groups, -1).layernorm(eps=self.eps).reshape(x.shape)

    if self.weight is None or self.bias is None: return x
    # elementwise_affine on channels
    return x * self.weight.reshape(1, -1, *[1] * (x.ndim-2)) + self.bias.reshape(1, -1, *[1] * (x.ndim-2))

class InstanceNorm:
  """
  Applies Instance Normalization over a mini-batch of inputs.

  - Described: https://paperswithcode.com/method/instance-normalization
  - Paper: https://arxiv.org/abs/1607.08022v3

  ```python exec="true" source="above" session="tensor" result="python"
  norm = nn.InstanceNorm(3)
  t = Tensor.rand(2, 3, 4, 4) * 2 + 1
  print(t.mean().item(), t.std().item())
  ```
  ```python exec="true" source="above" session="tensor" result="python"
  t = norm(t)
  print(t.mean().item(), t.std().item())
  ```
  """
  def __init__(self, num_features:int, eps=1e-5, affine=True):
    self.num_features, self.eps = num_features, eps
    self.weight: Tensor|None = Tensor.ones(num_features) if affine else None
    self.bias: Tensor|None = Tensor.zeros(num_features) if affine else None

  def __call__(self, x:Tensor) -> Tensor:
    x = x.reshape(x.shape[0], self.num_features, -1).layernorm(eps=self.eps).reshape(x.shape)
    if self.weight is None or self.bias is None: return x
    return x * self.weight.reshape(1, -1, *[1] * (x.ndim-2)) + self.bias.reshape(1, -1, *[1] * (x.ndim-2))

class LayerNorm:
  """
  Applies Layer Normalization over a mini-batch of inputs.

  - Described: https://paperswithcode.com/method/layer-normalization
  - Paper: https://arxiv.org/abs/1607.06450v1

  ```python exec="true" source="above" session="tensor" result="python"
  norm = nn.LayerNorm(3)
  t = Tensor.rand(2, 5, 3) * 2 + 1
  print(t.mean().item(), t.std().item())
  ```
  ```python exec="true" source="above" session="tensor" result="python"
  t = norm(t)
  print(t.mean().item(), t.std().item())
  ```
  """
  def __init__(self, normalized_shape:int|tuple[int, ...], eps=1e-5, elementwise_affine=True):
    self.normalized_shape: tuple[int, ...] = make_tuple(normalized_shape, 1)
    self.axis, self.eps, self.elementwise_affine = tuple(-1-i for i in range(len(self.normalized_shape))), eps, elementwise_affine
    self.weight: Tensor|None = Tensor.ones(*self.normalized_shape) if elementwise_affine else None
    self.bias: Tensor|None = Tensor.zeros(*self.normalized_shape) if elementwise_affine else None

  def __call__(self, x:Tensor) -> Tensor:
    assert self.normalized_shape == x.shape[-len(self.normalized_shape):], f"last dimensions of {x.shape} must match {self.normalized_shape}"
    x = x.layernorm(eps=self.eps, axis=self.axis)
    if not self.elementwise_affine: return x
    return x * self.weight + self.bias

class LayerNorm2d(LayerNorm):
  """
  Applies Layer Normalization over a mini-batch of 2D inputs.

  See: `LayerNorm`

  ```python exec="true" source="above" session="tensor" result="python"
  norm = nn.LayerNorm2d(3)
  t = Tensor.rand(2, 3, 4, 4) * 2 + 1
  print(t.mean().item(), t.std().item())
  ```
  ```python exec="true" source="above" session="tensor" result="python"
  t = norm(t)
  print(t.mean().item(), t.std().item())
  ```
  """
  def __call__(self, x: Tensor) -> Tensor: return super().__call__(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

class RMSNorm:
  """
  Applies Root Mean Square Normalization to input.

  - Described: https://paperswithcode.com/method/rmsnorm
  - Paper: https://arxiv.org/abs/1910.07467

  ```python exec="true" source="above" session="tensor" result="python"
  norm = nn.RMSNorm(4)
  t = Tensor.arange(12, dtype=dtypes.float).reshape(3, 4)
  print(t.numpy())
  ```
  ```python exec="true" source="above" session="tensor" result="python"
  print(norm(t).numpy())
  ```
  """
  def __init__(self, dim:int, eps=1e-6): self.eps, self.weight = eps, Tensor.ones(dim)

  def _norm(self, x:Tensor) -> Tensor: return x * (x.square().mean(-1, keepdim=True) + self.eps).rsqrt()

  def __call__(self, x:Tensor) -> Tensor: return self._norm(x.float()).cast(x.dtype) * self.weight

class Embedding:
  """
  A simple lookup table that stores embeddings of a fixed dictionary and size.

  See: https://pytorch.org/docs/stable/generated/torch.nn.Embedding

  ```python exec="true" source="above" session="tensor" result="python"
  emb = nn.Embedding(10, 3)
  print(emb(Tensor([1, 2, 3, 1])).numpy())
  ```
  """
  def __init__(self, vocab_size:int, embed_size:int):
    self.vocab_sz, self.embed_sz, self.weight = vocab_size, embed_size, Tensor.glorot_uniform(vocab_size, embed_size)

  def __call__(self, idx:Tensor) -> Tensor:
    if not hasattr(self, 'arange'): self.arange = Tensor.arange(self.vocab_sz, requires_grad=False, device=self.weight.device).unsqueeze(-1)
    big_shp = idx.shape+(self.vocab_sz, self.embed_sz)
    arange, idx, vals = self.arange.expand(big_shp), idx.reshape(idx.shape+(1, 1)).expand(big_shp), self.weight.expand(big_shp)
    return (arange == idx).mul(vals).sum(-2, acc_dtype=vals.dtype)

class LSTMCell:
  """
  A long short-term memory (LSTM) cell.

  Args:
    input_size: The number of expected features in the input `x`
    hidden_size: The number of features in the hidden state `h`
    bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`
  """
  def __init__(self, input_size:int, hidden_size:int, bias:bool=True):
    stdv = 1.0 / math.sqrt(hidden_size)
    self.weight_ih = Tensor.uniform(hidden_size*4, input_size, low=-stdv, high=stdv)
    self.weight_hh = Tensor.uniform(hidden_size*4, hidden_size, low=-stdv, high=stdv)
    self.bias_ih: Tensor|None = Tensor.zeros(hidden_size*4) if bias else None
    self.bias_hh: Tensor|None = Tensor.zeros(hidden_size*4) if bias else None

  def __call__(self, x:Tensor, hc:tuple[Tensor, Tensor]|None=None) -> tuple[Tensor, Tensor]:
    if hc is None: hc = (Tensor.zeros(x.size(0), self.weight_hh.size(1), dtype=x.dtype, device=x.device),)*2
    gates = x.linear(self.weight_ih.T, self.bias_ih) + hc[0].linear(self.weight_hh.T, self.bias_hh)
    i, f, g, o = gates.chunk(4, dim=1)
    i, f, g, o = i.sigmoid(), f.sigmoid(), g.tanh(), o.sigmoid()
    new_c = f * hc[1] + i * g
    new_h = o * new_c.tanh()
    return (new_h.contiguous(), new_c.contiguous())
