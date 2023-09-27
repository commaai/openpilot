from tinygrad.tensor import Tensor
from tinygrad.helpers import prod
from extra.onnx import safe_numpy
import numpy as np
import functools

def Unsqueeze(data, axes):
  axes = [len(data.shape) + int(x) if x < 0 else int(x) for x in safe_numpy(axes)]
  ptr = 0
  new_shape = []
  for i in range(len(data.shape) + len(axes)):
    if i in axes: new_shape.append(1)
    else:
      new_shape.append(data.shape[ptr])
      ptr += 1
  return data.reshape(new_shape)

def Gemm(A, B, C=None, alpha=1.0, beta=1.0, transA=0, transB=0):
  ret = alpha * ((A.transpose() if transA == 1 else A) @ (B.transpose() if transB == 1 else B))
  if C is not None: ret += beta * C
  return ret

# TODO: this is copied from tinygrad/nn/__init__.py
# spatial is from opset 7 and has since been removed
def BatchNormalization(X, scale, B, input_mean, input_var, epsilon=1e-05, momentum=0.9, training_mode=0, spatial=1):
  if training_mode:
    x_detached = X.detach()
    current_mean = x_detached.mean(axis=(0,2,3))
    y = (x_detached - current_mean.reshape(shape=[1, -1, 1, 1]))
    current_var = (y*y).mean(axis=(0,2,3))
    current_invstd = current_var.add(epsilon).pow(-0.5)

    running_mean = input_mean * momentum + current_mean * (1 - momentum)
    running_var = input_var * momentum + current_var * (1 - momentum)

    return X.batchnorm(scale, B, current_mean, current_invstd), running_mean, running_var
  else:
    invstd = (input_var + epsilon)**-0.5
    return X.batchnorm(scale, B, input_mean, invstd)
  
def LayerNormalization(x: Tensor, scale, bias, axis=-1, epsilon=1e-05, stash_type=1):
  assert stash_type == 1, "only float32 is supported"
  axis = tuple(i for i in range(axis if axis >= 0 else len(x.shape) + axis, len(x.shape)))
  mean = x.mean(axis=axis, keepdim=True)
  return x.layernorm(axis, epsilon).mul(scale).add(bias), mean, (x.sub(mean)).pow(2).mean(axis=axis, keepdim=True).add(epsilon).sqrt().reciprocal()

def GroupNormalization(x: Tensor, scale: Tensor, bias: Tensor, num_groups, epsilon=1e-05):
  return x.reshape(x.shape[0], num_groups, -1).layernorm(axis=-1, eps=epsilon).mul(scale.unsqueeze(-1)).add(bias.unsqueeze(-1)).reshape(x.shape)
  
# TODO: expand to N-D
def _padding(X, pads=None, auto_pad="NOTSET"):
  assert auto_pad == "NOTSET"  # TODO: write this
  if pads is not None:
    return X.pad2d((pads[1], pads[3], pads[0], pads[2]))
  else:
    return X

def AveragePool(X, kernel_shape, auto_pad="NOTSET", ceil_mode=0, count_include_pad=0, dilations=1, pads=None, strides=1):
  # TODO: the padding shouldn't be counted in the average! this is causing a test failure
  assert ceil_mode == 0 and dilations == 1
  padding_included = _padding(X, pads, auto_pad).avg_pool2d(kernel_shape, stride=strides)
  if count_include_pad:
    return padding_included
  else:
    div = _padding(Tensor.ones(*X.shape), pads, auto_pad).avg_pool2d(kernel_shape, stride=strides)
    return padding_included / div

def MaxPool(X, kernel_shape, auto_pad="NOTSET", ceil_mode=0, dilations=1, pads=None, storage_order=0, strides=1):
  # TODO: the padding should be infinity, not 0!
  assert ceil_mode == 0 and storage_order == 0 and dilations == 1
  return _padding(X, pads, auto_pad).max_pool2d(kernel_shape, stride=strides)

def Conv(X, W, B=None, auto_pad="NOTSET", dilations=1, group=1, kernel_shape=None, pads=None, strides=1):
  return X.conv2d(W, B, stride=strides, groups=group, dilation=dilations, padding=(pads[1], pads[3], pads[0], pads[2]) if pads is not None else 0)

# TODO: copied from tensor.py
def Dropout(data, ratio=0.5, training_mode=False, seed=None):
  # TODO: mask should be a boolean tensor
  if not training_mode: return data, Tensor.ones(*data.shape)  # if mask is requested as output it will contain all ones.
  if seed is not None: Tensor.manual_seed(seed)
  _mask : np.ndarray = np.asarray(Tensor._rng.binomial(1, 1.0-ratio, size=data.shape), dtype=data.dtype)
  mask = Tensor(_mask, requires_grad=False, device=data.device)
  return data * mask * (1/(1.0 - ratio)), mask

def Shape(data, end=None, start=0): return list(data.shape)[start:end]
def Size(data): return prod(data.shape)

# TODO: this doesn't match Tensor.flatten behavior
def Flatten(input, axis=1):
  new_shape = (1, -1) if axis == 0 else (prod(input.shape[0:axis]), -1)
  return input.reshape(new_shape)

# TODO: abstract out the broadcast logic in tensor
def Expand(input, shape):
  x_shape, y_shape = input.shape, [int(x) for x in safe_numpy(shape)]
  # copied from _broadcasted
  x_shape, y_shape = [([1]*(max(len(x_shape), len(y_shape))-len(t_shape)) + list(t_shape)) for t_shape in [x_shape, y_shape]]
  shape_ret = tuple(max(sx, sy) for sx,sy in zip(x_shape, y_shape))
  return input.reshape(x_shape).expand(shape_ret)

def LRN(input, size, alpha=1e-4, beta=0.75, bias=1.0):
  bs, c, iy, ix = input.shape 
  return input / input.mul(input).reshape(bs,1,c,iy*ix).pad2d((0,0,(size-1)//2, size//2)).avg_pool2d((size, 1), 1).reshape(bs,c,iy,ix).mul(alpha).add(bias).pow(beta)

def Identity(input): return input
def Neg(input): return -input
def Reciprocal(input): return input.reciprocal()
def Sqrt(input): return input.sqrt()
def Sign(input): return input.sign()
def Softsign(input): return input / (1+input.abs())
def Abs(input): return input.abs()
def Exp(input): return input.exp()
def Log(input): return input.log()
def Mish(input): return input.mish()
def HardSigmoid(input, alpha=0.2, beta=0.5): return (alpha*input + beta).clip(0, 1)
def HardSwish(input): return input * HardSigmoid(input, 1/6, 0.5)
def Celu(X, alpha=1.0): return X.relu() - (-alpha*(X/alpha).exp()+1).relu()
def Selu(X, alpha=1.67326319217681884765625, gamma=1.05070102214813232421875): return gamma * (X.relu() - (-alpha*X.exp()+alpha).relu())
def Softplus(X): return X.softplus()
def PRelu(X, slope): return X.leakyrelu(slope)
def LeakyRelu(X, alpha=0.01): return X.leakyrelu(alpha)
def ThresholdedRelu(X, alpha=1.0): return (X-alpha).relu() + (X-alpha).relu().sign() * alpha
def Softmax_1(input, axis=1): return input.softmax(axis)
def Softmax_13(input, axis=-1): return input.softmax(axis)
Softmax = {1: Softmax_1, 13: Softmax_13}   # Softmax default axis changed
def LogSoftmax(input, axis=-1): return input.log_softmax(axis)
def Clip(input, min=-3.4e38, max=3.4e38): return input.clip(min, max)

def Less(x, y): return (x<y).numpy().astype(bool)
def LessOrEqual(x, y): return (x<=y).numpy().astype(bool)
def Greater(x, y): return (x>y).numpy().astype(bool)
def GreaterOrEqual(x, y): return (x>=y).numpy().astype(bool)
def Equal(x, y): return (x.eq(y)).numpy().astype(bool)

def Max(*data_0): return functools.reduce(Tensor.maximum, data_0)
def Min(*data_0): return -functools.reduce(Tensor.maximum, [-x for x in data_0])
def Sum(*data_0): return functools.reduce(Tensor.__add__, data_0)
def Mean(*data_0): return functools.reduce(Tensor.__add__, data_0) / len(data_0)

def _axes(axes, noop_with_empty_axes): return [int(x) for x in safe_numpy(axes)] if axes is not None else ([] if noop_with_empty_axes else None)

# ReduceProd would require a new llop
def ReduceMax(data, axes=None, keepdims=1, noop_with_empty_axes=0): return data.max(_axes(axes, noop_with_empty_axes), keepdim=keepdims)
def ReduceMin(data, axes=None, keepdims=1, noop_with_empty_axes=0): return data.min(_axes(axes, noop_with_empty_axes), keepdim=keepdims)
def ReduceSum(data, axes=None, keepdims=1, noop_with_empty_axes=0): return data.sum(_axes(axes, noop_with_empty_axes), keepdim=keepdims)
def ReduceMean(data, axes=None, keepdims=1, noop_with_empty_axes=0): return data.mean(_axes(axes, noop_with_empty_axes), keepdim=keepdims)
def ReduceSumSquare(data, axes=None, keepdims=1, noop_with_empty_axes=0): return data.square().sum(_axes(axes, noop_with_empty_axes), keepdim=keepdims)
def ReduceL1(data, axes=None, keepdims=1, noop_with_empty_axes=0): return data.abs().sum(_axes(axes, noop_with_empty_axes), keepdim=keepdims)
def ReduceL2(data, axes=None, keepdims=1, noop_with_empty_axes=0): return data.square().sum(_axes(axes, noop_with_empty_axes), keepdim=keepdims).sqrt()
def ReduceLogSum(data, axes=None, keepdims=1, noop_with_empty_axes=0): return data.sum(_axes(axes, noop_with_empty_axes), keepdim=keepdims).log()
def ReduceLogSumExp(data, axes=None, keepdims=1, noop_with_empty_axes=0): return data.exp().sum(_axes(axes, noop_with_empty_axes), keepdim=keepdims).log()

def GlobalAveragePool(X): return X.mean(axis=tuple(range(2, len(X.shape))), keepdim=True)
def GlobalMaxPool(X): return X.max(axis=tuple(range(2, len(X.shape))), keepdim=True)

def Tile(input, repeats):
  repeats_ = [int(x) for x in safe_numpy(repeats)]
  new_shape = [x for i in range(len(input.shape)) for x in [1,input.shape[i]]]
  expand_shape = [x for r,s in zip(repeats_, input.shape) for x in [r,s]]
  final_shape = [r*s for r,s in zip(repeats_, input.shape)]
  return input.reshape(new_shape).expand(expand_shape).reshape(final_shape)

def Range(start, limit, delta): return Tensor.arange(safe_numpy(limit)[0], safe_numpy(start)[0], safe_numpy(delta)[0])
def Where(condition, X, Y): return condition*X + (1-condition)*Y

def ConstantOfShape(input, value=0.0):
  shape = [int(x) for x in safe_numpy(input)]
  return Tensor.ones(*shape) * value

# this is obviously wrong, but since we don't have types, it's better than nothing
def Cast(input, to):
  print(f"WARNING: attempting to cast to {to}")
  return input

# NOTE: since we only have one type, this is valid!
def CastLike(input, target_type):
  assert isinstance(target_type, Tensor), "can only CastLike Tensor"
  return input
