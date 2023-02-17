from tinygrad.tensor import Tensor

class BatchNorm2D:
  def __init__(self, sz, eps=1e-5, affine=True, track_running_stats=True, momentum=0.1):
    assert affine, "BatchNorm2D is only supported with affine"
    self.eps, self.track_running_stats, self.momentum = eps, track_running_stats, momentum

    self.weight, self.bias = Tensor.ones(sz), Tensor.zeros(sz)

    self.running_mean, self.running_var = Tensor.zeros(sz, requires_grad=False), Tensor.ones(sz, requires_grad=False)
    self.num_batches_tracked = Tensor.zeros(1, requires_grad=False)

  def __call__(self, x):
    if Tensor.training:
      # This requires two full memory accesses to x
      # https://github.com/pytorch/pytorch/blob/c618dc13d2aa23625cb0d7ada694137532a4fa33/aten/src/ATen/native/cuda/Normalization.cuh
      # There's "online" algorithms that fix this, like https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_Online_algorithm
      x_detached = x.detach()
      batch_mean = x_detached.mean(axis=(0,2,3))
      y = (x_detached - batch_mean.reshape(shape=[1, -1, 1, 1]))
      batch_var = (y*y).mean(axis=(0,2,3))
      batch_invstd = batch_var.add(self.eps)**-0.5
      self.batch_invstd = None

      # NOTE: wow, this is done all throughout training in most PyTorch models
      if self.track_running_stats:
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
        self.num_batches_tracked += 1
    else:
      batch_mean, batch_var = self.running_mean, self.running_var
      # NOTE: this can be precomputed for static inference. if you manually update running_var, you have to reset this
      if not hasattr(self, "batch_invstd") or not self.batch_invstd:
        self.batch_invstd = batch_var.add(self.eps)**-0.5
      batch_invstd = self.batch_invstd

    return x.batchnorm(self.weight, self.bias, batch_mean, batch_invstd)

# TODO: is this good weight init?
class Conv2d:
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
    self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else (kernel_size[0], kernel_size[1])
    self.stride = (stride, stride) if isinstance(stride, int) else (stride[0], stride[1])
    self.padding = (padding, ) * 4 if isinstance(padding, int) else ((padding[0], padding[0], padding[1], padding[1]) if len(padding) == 2 else padding)
    self.weight = Tensor.glorot_uniform(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1])
    self.bias = Tensor.zeros(out_channels) if bias else None

  def __call__(self, x):
    return x.conv2d(self.weight, self.bias, padding=self.padding, stride=self.stride)

class Linear:
  def __init__(self, in_features, out_features, bias=True):
    self.weight = Tensor.glorot_uniform(out_features, in_features)
    self.bias = Tensor.zeros(out_features) if bias else None

  def __call__(self, x):
    return x.linear(self.weight.transpose(), self.bias)
