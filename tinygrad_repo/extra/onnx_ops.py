from tinygrad.tensor import Tensor
from tinygrad.helpers import prod, dtypes, ImageDType
from extra.onnx import safe_numpy
from onnx.helper import tensor_dtype_to_np_dtype
from onnx.onnx_pb import TensorProto
import os
import numpy as np
import functools
from typing import Union, Tuple, Optional, List, Any
import math

# **************** Free Ops ****************

def Identity(input: Tensor): return input
def Neg(input: Tensor): return -input
def Add(input: Tensor, other: Tensor, broadcast=None): return input + other if input.dtype == dtypes.float or isinstance(input.dtype, ImageDType) else (input + other).cast(input.dtype)
def Sub(input: Union[Tensor, Any], other: Tensor): return input - other # some test has input as int
def Mul(input: Tensor, other: Tensor): return (input * other) if input.dtype == dtypes.float or isinstance(input.dtype, ImageDType) else (input * other).cast(input.dtype)
# in openpilot, due to SHUFFLE_PAD_OPS issues, we are spending an extra kernel
def Div(input: Tensor, other: Tensor): return input / other if input.dtype == dtypes.float or isinstance(input.dtype, ImageDType) else input.div(other).floor()
def Pow(input: Tensor, other: Tensor): return (input.float() ** other.float()).cast(input.dtype)
def Reciprocal(input: Tensor): return input.reciprocal()
def Sqrt(input: Tensor): return input.sqrt()
def Sign(input: Tensor): return input.sign()
def Abs(input: Tensor): return input.abs()
def Exp(input: Tensor): return input.exp()
def Log(input: Tensor): return input.log()
def Mish(input: Tensor): return input.mish()
def Sin(x: Tensor): return x.sin()
def Cos(x: Tensor): return x.cos()
def Tan(x: Tensor): return x.tan()
def Relu(input: Tensor): return input.relu()
def Sigmoid(input: Tensor): return input.sigmoid()
def Tanh(input: Tensor): return input.tanh()
def MatMul(input: Tensor, other: Tensor): return input.matmul(other)
def Floor(x:Tensor): return x.floor()
def Ceil(x:Tensor): return x.ceil()
def Less(x:Tensor,y:Tensor): return (x<y).cast(dtypes.bool)
def LessOrEqual(x:Tensor,y:Tensor): return (x<=y).cast(dtypes.bool)
def Greater(x:Tensor,y:Tensor): return (x>y).cast(dtypes.bool)
def GreaterOrEqual(x:Tensor,y:Tensor): return (x>=y).cast(dtypes.bool)
def Equal(x:Tensor,y:Tensor): return (x==y).cast(dtypes.bool)
def Max(*data_0): return functools.reduce(Tensor.maximum, data_0)
def Min(*data_0): return functools.reduce(Tensor.minimum, data_0)
def Sum(*data_0): return functools.reduce(Tensor.__add__, data_0)
def Mean(*data_0): return functools.reduce(Tensor.__add__, data_0) / len(data_0)
def Where(condition:Tensor,X:Tensor,Y:Tensor): return condition.where(X, Y).cast(X.dtype)
def Cast(input: Tensor, to): return input.cast(dtypes.from_np(tensor_dtype_to_np_dtype(to)))

# **************** Simple Ops ****************

def Constant(value: Tensor=None, value_float=None, value_floats=None, value_int=None, value_ints=None, value_string=None, value_strings=None):
  if value: return value
  elif value_float: return Tensor(value_float, dtype=dtypes.float32, requires_grad=False)
  elif value_floats: return Tensor(list(value_floats), dtype=dtypes.float32, requires_grad=False)
  elif value_int: return Tensor(value_int, dtype=dtypes.int64, requires_grad=False)
  elif value_ints: return Tensor(list(value_ints), dtype=dtypes.int64, requires_grad=False)
  elif value_string or value_strings: raise NotImplementedError(f'value_string or value_strings not implemented for Constant op')

def Softsign(input: Tensor): return input / (1+input.abs())
def Cosh(x): return (math.e ** x + math.e ** -x) / 2
def Sinh(x): return (math.e ** x - math.e ** -x) / 2
def Tanh(x): return x.tanh()

def HardSigmoid(input: Tensor, alpha=0.2, beta=0.5): return (alpha*input + beta).clip(0, 1)
def HardSwish(input: Tensor): return input * HardSigmoid(input, 1/6, 0.5)
def Celu(X: Tensor, alpha=1.0): return X.relu() - (-alpha*(X/alpha).exp()+1).relu()
def Selu(X: Tensor, alpha=1.67326319217681884765625, gamma=1.05070102214813232421875): return gamma * (X.relu() - (-alpha*X.exp()+alpha).relu())
def Softplus(X: Tensor): return X.softplus()
def PRelu(X:Tensor, slope:Tensor):
  slope = slope[0] if slope.shape[-1] != X.shape[-1] else slope # HACK OnnxBackendPyTorchConvertedModelTest HAS WEIRD SLOPE WHERE IT'S [0.25, 0.25, 0.25] FOR ANY X.SHAPE
  return X.clip(0, float("inf")) + X.clip(float("-inf"), 0) * slope
def LeakyRelu(X: Tensor, alpha=0.01): return X.leakyrelu(alpha)
def ThresholdedRelu(X: Tensor, alpha=1.0): return (X-alpha).relu() + (X-alpha).relu().sign() * alpha
def Softmax_1(input: Tensor, axis=1): return input.softmax(axis)
def Softmax_13(input: Tensor, axis=-1): return input.softmax(axis)
Softmax = {1: Softmax_1, 13: Softmax_13}   # Softmax default axis changed
def LogSoftmax(input: Tensor, axis=-1): return input.log_softmax(axis)
def Clip(input: Tensor, min=None, max=None): return input.clip(float('-inf') if min is None else min, float('inf') if max is None else max)

# NOTE ReduceProd would require a new llop
def _axes(axes, noop_with_empty_axes): return [int(x) for x in safe_numpy(axes)] if axes is not None and not (isinstance(axes, Tensor) and axes.shape == (0,)) else ([] if noop_with_empty_axes else None)
def ReduceMax(data: Tensor, axes=None, keepdims=1, noop_with_empty_axes=0): return data.max(_axes(axes, noop_with_empty_axes), keepdim=keepdims)
def ReduceMin(data: Tensor, axes=None, keepdims=1, noop_with_empty_axes=0): return data.min(_axes(axes, noop_with_empty_axes), keepdim=keepdims)
def ReduceSum(data: Tensor, axes=None, keepdims=1, noop_with_empty_axes=0): return data.sum(_axes(axes, noop_with_empty_axes), keepdim=keepdims)
def ReduceMean(data: Tensor, axes=None, keepdims=1, noop_with_empty_axes=0): return data.mean(_axes(axes, noop_with_empty_axes), keepdim=keepdims)
def ReduceSumSquare(data: Tensor, axes=None, keepdims=1, noop_with_empty_axes=0): return data.square().sum(_axes(axes, noop_with_empty_axes), keepdim=keepdims)
def ReduceL1(data: Tensor, axes=None, keepdims=1, noop_with_empty_axes=0): return data.abs().sum(_axes(axes, noop_with_empty_axes), keepdim=keepdims)
def ReduceL2(data: Tensor, axes=None, keepdims=1, noop_with_empty_axes=0): return data.square().sum(_axes(axes, noop_with_empty_axes), keepdim=keepdims).sqrt()
def ReduceLogSum(data: Tensor, axes=None, keepdims=1, noop_with_empty_axes=0): return data.sum(_axes(axes, noop_with_empty_axes), keepdim=keepdims).log()
def ReduceLogSumExp(data: Tensor, axes=None, keepdims=1, noop_with_empty_axes=0): return data.exp().sum(_axes(axes, noop_with_empty_axes), keepdim=keepdims).log()

def GlobalAveragePool(X: Tensor): return X.mean(axis=tuple(range(2, len(X.shape))), keepdim=True)
def GlobalMaxPool(X: Tensor): return X.max(axis=tuple(range(2, len(X.shape))), keepdim=True)
def OptionalHasElement(x: Tensor=None): return Tensor(x is not None and x.numel() > 0, dtype=dtypes.bool)
def OptionalGetElement(x: Tensor=None): return x if x is not None else Tensor([], dtype=dtypes.float32)

def Tile(input: Tensor, repeats): return input.repeat([int(x) for x in safe_numpy(repeats)])
def Range(start: Tensor, limit, delta): return Tensor.arange(start=int(safe_numpy(start)), stop=int(safe_numpy(limit)), step=int(safe_numpy(delta))).cast(dtype=start.dtype)
def Shape(data: Tensor, end=None, start=0): return Tensor(list(data.shape)[start:end], dtype=dtypes.int32 if os.path.isfile("/TICI") else dtypes.int64)  # TODO: really?
def Size(data: Tensor): return prod(data if isinstance(data, list) else data.shape)
def Flatten(input: Tensor, axis=1): return input.reshape(prod((1,) + input.shape[0:axis]), -1)
def Reshape(data: Tensor, shape: Tensor, allowzero=None): return data.reshape([int(x) if x != 0 else data.shape[i] for i,x in enumerate(safe_numpy(shape))])
def Shrink(input: Tensor, bias=0.0, lambd=0.5): return (input < -lambd)*(input+bias) + (input > lambd)*(input-bias)
def And(x:Tensor, y:Tensor): return Where((x==y), x, Tensor.zeros(*x.shape)).cast(dtypes.bool)
def Or(x:Tensor, y:Tensor): return Where((x==y), x, Tensor.ones(*x.shape)).cast(dtypes.bool)
def Xor(x:Tensor, y:Tensor): return Where((x==y), Tensor.zeros(*x.shape), Tensor.ones(*x.shape)).cast(dtypes.bool)
def Not(x:Tensor): return Where((x==1), Tensor.zeros(*x.shape), Tensor.ones(*x.shape)).cast(dtypes.bool)

def Asin(x): return Atan(x / Tensor.sqrt(1 - x * x))
def Asinh(x): return Tensor.log(x + Tensor.sqrt(x * x + 1))
def Acosh(x): return Tensor.log(x + Tensor.sqrt(x * x - 1))
def Atanh(x): return 0.5 * Tensor.log((1 + x)/(1 - x))
def Acos(x: Tensor):
  negate = (x < 0)
  x = x.abs()
  ret = ((((-0.0187293 * x) + 0.0742610)*x - 0.2121144) * x + 1.5707288) * Tensor.sqrt(1.0 - x)
  ret = ret - 2 * negate * ret
  return negate * 3.14159265358979 + ret
def Atan(y: Tensor):
  x = Tensor.ones(y.shape)
  t3 = x
  t1 = y.abs()
  t0 = (t3 > t1).where(t3, t1)
  t1 = (t3 < t1).where(t3, t1)
  t3 = t1 / t0
  t4 = t3 * t3
  t0 = ((((-0.013480470 * t4 + 0.057477314) * t4 - 0.121239071) * t4 + 0.195635925) * t4 - 0.332994597) * t4 + 0.999995630
  t3 = t0 * t3
  t3 = (y.abs() > x.abs()).where(1.570796327 - t3, t3)
  return (y < 0).where(-t3, t3)

def Trilu(x: Tensor, k: Union[Tensor, int]=0, upper=1):
  k = int(k.numpy().item()) if k != 0 else 0 # onnx passes k as a tensor int64 with one element, default is 0
  return x.triu(k) if upper else x.tril(k)

def Squeeze(input: Tensor, axes):
  if isinstance(axes, Tensor): axes = safe_numpy(axes)
  axes = [int(x) if x >= 0 else int(x+input.ndim) for x in axes]
  return input.reshape([s for i,s in enumerate(input.shape) if i not in axes])
def Unsqueeze(data: Tensor, axes):
  axes = [len(data.shape) + int(x) if x < 0 else int(x) for x in safe_numpy(axes)]
  new_shape = [1] * (len(data.shape) + len(axes))
  ptr = iter(data.shape)
  for i in range(len(new_shape)):
    if i not in axes:
      new_shape[i] = next(ptr)
  return data.reshape(new_shape)

def Binarizer(input, threshold=0.0): return input > threshold

def ArgMax(x: Tensor, axis=0, keepdims=1, select_last_index=0):
  axis = axis + x.ndim if axis < 0 else axis
  m = x == (x.max(axis=axis, keepdim=keepdims) if keepdims else x.max(axis=axis, keepdim=keepdims).unsqueeze(axis))
  c = Tensor.arange(x.shape[axis]).reshape(*[1]*(axis), x.shape[axis], *[1]*(x.ndim - axis-1)) * m
  return c.max(axis=axis,keepdim=keepdims).cast(dtypes.int64)
def ArgMin(x, axis=0, keepdims=1, select_last_index=0): return ArgMax(-x, axis=axis, keepdims=keepdims, select_last_index=select_last_index)

def Elu(input: Tensor, alpha=1.0): return input.elu(alpha=alpha)
def Concat(*inputs: List[Tensor], axis): return inputs[0].cat(*inputs[1:], dim=axis)
def Transpose(input: Tensor, perm=None): return input.permute(order=list(range(len(input.shape))[::-1]) if perm is None else perm)

# NOTE: since we only have one type, this is valid!
def CastLike(input, target_type):
  assert isinstance(target_type, Tensor), "can only CastLike Tensor"
  return input

def ConstantOfShape(input, value:Tensor=None):
  if value is None: value=Tensor([0.0])
  shape = [int(x) for x in safe_numpy(input)]
  return Tensor.ones(*shape, dtype=value.dtype) * (value if shape[0]!=0 else 1)

# TODO: abstract out the broadcast logic in tensor
def Expand(input: Tensor, shape):
  x_shape, y_shape = input.shape, [int(x) for x in safe_numpy(shape)]
  # copied from _broadcasted
  x_shape, y_shape = [([1]*(max(len(x_shape), len(y_shape))-len(t_shape)) + list(t_shape)) for t_shape in [x_shape, y_shape]]
  shape_ret = tuple(max(sx, sy) for sx,sy in zip(x_shape, y_shape))
  return input.reshape(x_shape).expand(shape_ret)

# **************** Complex Ops ****************

def Gemm(A: Tensor, B: Tensor, C: Tensor=None, alpha=1.0, beta=1.0, transA=0, transB=0, broadcast=0):
  ret = alpha * (A.transpose(transA) @ B.transpose(transB))
  if C is not None: ret += beta * (C if broadcast == 0 else C.reshape([-1 if i <  len(C.shape) else 1 for i in range(len(ret.shape))][::-1]))
  return ret

# works with Tensors.ndim != 4
def _batchnorm(self:Tensor, weight:Optional[Tensor], bias:Optional[Tensor], mean:Tensor, invstd:Tensor):
  shape = [1, -1] + [1] * (self.ndim-2)
  x = (self - mean.reshape(shape=shape))
  if weight: x = x * weight.reshape(shape=shape)
  ret = x.mul(invstd.reshape(shape=shape) if len(invstd.shape) == 1 else invstd)
  return (ret + bias.reshape(shape=shape)) if bias else ret

# TODO: this is copied from tinygrad/nn/__init__.py
# spatial is from opset 7 and has since been removed
def BatchNormalization(X: Tensor, scale, B, input_mean, input_var, epsilon=1e-05, momentum=0.9, training_mode=0, spatial=1, is_test=0):
  if training_mode:
    x_detached = X.detach()
    current_mean = x_detached.mean(axis=(0,2,3))
    y = (x_detached - current_mean.reshape(shape=[1, -1, 1, 1]))
    current_var = (y*y).mean(axis=(0,2,3))
    current_invstd = current_var.add(epsilon).pow(-0.5)

    running_mean = input_mean * momentum + current_mean * (1 - momentum)
    running_var = input_var * momentum + current_var * (1 - momentum)

    return _batchnorm(X, scale, B, current_mean, current_invstd), running_mean, running_var
  else:
    invstd = (input_var + epsilon)**-0.5
    return _batchnorm(X, scale, B, input_mean, invstd)

def InstanceNormalization(x: Tensor, scale: Tensor, bias: Tensor, epsilon=1e-05):
  axis = tuple(range(2, len(x.shape)))
  mean = x.mean(axis=axis, keepdim=True)
  invstd = x.sub(mean).pow(2).mean(axis=axis, keepdim=True).add(epsilon).pow(-0.5)
  return x.sub(mean).mul(scale.reshape(shape=[-1, 1, 1])).mul(invstd).add(bias.reshape(shape=[-1, 1, 1]))

def LayerNormalization(x: Tensor, scale, bias, axis=-1, epsilon=1e-05, stash_type=1):
  assert stash_type == 1, "only float32 is supported"
  axis = tuple(i for i in range(axis if axis >= 0 else len(x.shape) + axis, len(x.shape)))
  mean = x.mean(axis=axis, keepdim=True)
  return x.layernorm(axis, epsilon).mul(scale).add(bias), mean, (x.sub(mean)).pow(2).mean(axis=axis, keepdim=True).add(epsilon).sqrt().reciprocal()

def GroupNormalization(x: Tensor, scale: Tensor, bias: Tensor, num_groups, epsilon=1e-05):
  return x.reshape(x.shape[0], num_groups, -1).layernorm(axis=-1, eps=epsilon).mul(scale.unsqueeze(-1)).add(bias.unsqueeze(-1)).reshape(x.shape)

# onnx: [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
# numpy.pad: ((x1_begin, x1_end), (x2_begin, x2_end), ...)
def _format_padding(onnx_pads, ndims=None, axes=None):
  if ndims and len(onnx_pads)//2 != ndims:  onnx_pads = onnx_pads * ndims # for OnnxBackendPyTorchConvertedModelTest the len(onnx_pads) == 2
  if ndims is None: ndims = len(onnx_pads) // 2
  if axes is None: axes = list(range(ndims))
  num_axes = len(axes)
  np_pads = [(0,0)] * ndims
  for i in range(num_axes):
    np_pads[axes[i]] = (onnx_pads[i], onnx_pads[i + num_axes])
  return np_pads

def _padding(X: Tensor, pads=None, auto_pad="NOTSET", axes=None, constant_value=0., strides=None, kernel_shape=None, dilations=None):
  if auto_pad != "NOTSET": pads = _auto_pad(X, auto_pad, strides, kernel_shape, dilations)
  if pads is None: return X
  pads = _format_padding(pads, ndims=len(X.shape), axes=axes)
  return X.pad(tuple(pads), value=constant_value)

def _auto_pad(X, auto_pad, strides, kernel_shape, dilations):
  strides = [strides]*len(kernel_shape) if isinstance(strides, int) else strides if strides else [1]*len(kernel_shape)
  dilations = [1]*len(kernel_shape) if dilations == 1 else dilations
  pad_shape = [(math.ceil(sh/st)-1)*st+((ks-1)*di+1)-sh for sh, st, ks, di in zip(X.shape[-len(strides):], strides, kernel_shape, dilations)]
  if auto_pad == "SAME_UPPER": return [pad_shape[0]//2, pad_shape[1]//2, pad_shape[0]-pad_shape[0]//2, pad_shape[1]-pad_shape[1]//2]
  elif auto_pad == "SAME_LOWER": return [pad_shape[0]-pad_shape[0]//2, pad_shape[1]-pad_shape[1]//2, pad_shape[0]//2,  pad_shape[1]//2]
  else: raise NotImplementedError(f"auto_pad={auto_pad} not implemented, yet")

def Pad(x: Tensor, pads: Union[Tensor, Tuple[int, ...]], constant_value: Tensor=None, axes: Tensor=None, mode="constant", value: float=0.):
  constant_value = value if constant_value is None else float(safe_numpy(constant_value)[0])
  seq_pads = list(pads) if isinstance(pads, tuple) else safe_numpy(pads)
  seq_pads = [math.ceil(i) for i in seq_pads]
  seq_axes = safe_numpy(axes).astype(np.int32).tolist() if axes is not None else None
  base_shape = x.shape
  pads = _format_padding(seq_pads, ndims=len(x.shape), axes=seq_axes)
  if mode == "wrap":
    repeat_args = [math.ceil(dim[0]/sh) + math.ceil(dim[1]/sh) + 1 for dim, sh in zip(pads, base_shape)]
    new_shape = [s*r for s,r in zip(base_shape, repeat_args)]
    shrink_args = [(sh-dim[0]%sh if dim[0]%sh != 0 else 0, nsh-(sh-dim[1]%sh) if dim[1]%sh != 0 else nsh) for dim, sh, nsh in zip(pads, base_shape, new_shape)]
    return x.repeat(tuple(repeat_args)).shrink(tuple(shrink_args))
  elif mode == "reflect":
    for i,s in enumerate(x.shape):
      if pads[i] == (0,0): continue
      elif pads[i][0] and not pads[i][1]:
        x = x.flip(i).shrink(tuple([(0,s_) if i_ != i else (s-pads[i][0]-1, s_-1) for i_,s_ in enumerate(x.shape)])).pad(tuple([(0,0) if i_ != i else (0,s) for i_ in range(x.ndim)])) + \
            x.pad(tuple([(0,0) if i_ != i else pads[i] for i_ in range(x.ndim)]))
      elif not pads[i][0] and pads[i][1]:
        x = x.flip(i).shrink(tuple([(0,s_) if i_ != i else (1, pads[i][1]+1) for i_,s_ in enumerate(x.shape)])).pad(tuple([(0,0) if i_ != i else (s,0) for i_ in range(x.ndim)])) + \
            x.pad(tuple([(0,0) if i_ != i else pads[i] for i_ in range(x.ndim)]))
      else:
        x = x.flip(i).shrink(tuple([(0,s_) if i_ != i else (s-pads[i][0]-1, s_-1) for i_,s_ in enumerate(x.shape)])).pad(tuple([(0,0) if i_ != i else (0,s+pads[i][1]) for i_ in range(x.ndim)])) + \
            x.flip(i).shrink(tuple([(0,s_) if i_ != i else (1, pads[i][1]+1) for i_,s_ in enumerate(x.shape)])).pad(tuple([(0,0) if i_ != i else (s+pads[i][0],0) for i_ in range(x.ndim)])) + \
            x.pad(tuple([(0,0) if i_ != i else pads[i] for i_ in range(x.ndim)]))
    return x
  elif mode == "edge":
    for i,s in enumerate(x.shape):
      if pads[i] == (0,0): continue
      elif pads[i][0] and not pads[i][1]:
        x = x.shrink(tuple([(0,s_) if i_ != i else (0,1) for i_,s_ in enumerate(x.shape)])).expand([pads[i][0] if i_ == i else s_ for i_,s_ in enumerate(x.shape)]).pad(tuple([(0,0) if i_ != i else (0,s) for i_ in range(x.ndim)])) + \
            x.pad(tuple([(0,0) if i_ != i else pads[i] for i_ in range(x.ndim)]))
      elif not pads[i][0] and pads[i][1]:
        x = x.shrink(tuple([(0,s_) if i_ != i else (s_-1, s_) for i_,s_ in enumerate(x.shape)])).expand([pads[i][0] if i_ == i else s_ for i_,s_ in enumerate(x.shape)]).pad(tuple([(0,0) if i_ != i else (s+pads[i][0],0) for i_ in range(x.ndim)])) + \
            x.pad(tuple([(0,0) if i_ != i else pads[i] for i_ in range(x.ndim)]))
      else:
        x = x.shrink(tuple([(0,s_) if i_ != i else (0,1) for i_,s_ in enumerate(x.shape)])).expand([pads[i][0] if i_ == i else s_ for i_,s_ in enumerate(x.shape)]).pad(tuple([(0,0) if i_ != i else (0,s+pads[i][1]) for i_ in range(x.ndim)])) + \
            x.shrink(tuple([(0,s_) if i_ != i else (s_-1, s_) for i_,s_ in enumerate(x.shape)])).expand([pads[i][1] if i_ == i else s_ for i_,s_ in enumerate(x.shape)]).pad(tuple([(0,0) if i_ != i else (s+pads[i][0],0) for i_ in range(x.ndim)])) + \
            x.pad(tuple([(0,0) if i_ != i else pads[i] for i_ in range(x.ndim)]))
    return x
  elif mode == "constant":
    return _padding(x, seq_pads, axes=seq_axes, constant_value=constant_value)

def AveragePool(X: Tensor, kernel_shape, auto_pad="NOTSET", ceil_mode=0, count_include_pad=0, dilations=1, pads=None, strides=1):
  if dilations != 1: raise NotImplementedError(f"dilations != 1 not supported, dilations:{dilations}")
  pixel_axes = tuple(range(len(X.shape)))[-2:]
  if ceil_mode: auto_pad = "SAME_UPPER"
  padding_included = _padding(X, pads, auto_pad, axes=pixel_axes, strides=strides, kernel_shape=kernel_shape, dilations=dilations).avg_pool2d(kernel_shape, stride=strides)
  if count_include_pad:
    return padding_included
  else:
    div = _padding(Tensor.ones(*X.shape), pads, auto_pad, axes=pixel_axes, strides=strides, kernel_shape=kernel_shape, dilations=dilations).avg_pool2d(kernel_shape, stride=strides)
    return padding_included / div

def MaxPool(X: Tensor, kernel_shape, auto_pad="NOTSET", ceil_mode=0, dilations=1, pads=None, storage_order=0, strides=1):
  if ceil_mode: auto_pad = "SAME_UPPER"
  ret = _padding(X, pads, auto_pad, constant_value=-np.inf, axes=tuple(range(len(X.shape)))[-len(kernel_shape):], strides=strides, kernel_shape=kernel_shape, dilations=dilations)
  ret = ret.max_pool2d(kernel_shape, stride=strides, dilation=dilations)
  ret_len, X_len = ret.numel(), X.numel()
  indices = ((ret.flatten().unsqueeze(1).expand(ret_len, X_len) == X.flatten().reshape(1, X_len).expand(ret_len, X_len)) * Tensor.arange(X_len).reshape(1, X_len).expand(ret_len, X_len)).sum(1).reshape(ret.shape).cast(dtypes.int64)
  if storage_order: indices = indices.transpose(indices.ndim-2, indices.ndim-1)
  return ret, indices

def MaxUnpool(xT: Tensor, xI: Tensor, outshape: Tensor=None, kernel_shape=None, pads=None, strides=None):
  out_sh = [(ks//2)*2 + st * inps for inps, st, ks in zip(xI.shape, strides, kernel_shape)]
  outlength = prod(out_sh)
  xI = xI.flatten().unsqueeze(1).expand(prod(xT.shape), outlength)
  arange = Tensor.arange(outlength, requires_grad=False).reshape(1, outlength).expand(xI.shape)
  xT = xT.flatten().unsqueeze(1).expand(prod(xT.shape), outlength)
  ret = ((xI == arange) * xT).sum(0).reshape([1, 1] + out_sh)
  if outshape is not None:
    outshape = safe_numpy(outshape).tolist()
    if outshape != ret.shape:
      diff = [outshape[2] - ret.shape[2], outshape[3] - ret.shape[3]]
      pad_args = [diff[0]//2, diff[1]//2, diff[0]-diff[0]//2, diff[1]-diff[1]//2]
      ret = ret.pad2d((pad_args[1], pad_args[3], pad_args[0], pad_args[2]))
  return ret

def Conv(X: Tensor, W: Tensor, B=None, auto_pad="NOTSET", dilations=1, group=1, kernel_shape=None, pads=None, strides=1):
  if auto_pad != "NOTSET": padding = _auto_pad(X, auto_pad, strides, kernel_shape, dilations)
  else: padding = [p for ps in zip(pads[:len(pads)//2][::-1], pads[len(pads)//2:][::-1]) for p in ps] if pads is not None else 0 # reorder padding
  return X.conv2d(W, B, stride=strides, groups=group, dilation=dilations, padding=padding)

def ConvTranspose(X: Tensor, W: Tensor, B=None, auto_pad="NOTSET", dilations=1, group=1, kernel_shape=None, pads=None, output_shape=None, output_padding=0, strides=1):
  if not kernel_shape: kernel_shape = W.shape
  if pads is None and auto_pad != "NOTSET": pads = _auto_pad(X, auto_pad, strides, kernel_shape, dilations)
  elif pads is None and auto_pad == "NOTSET": pads = [0,0] * (X.ndim - 2)
  strides_ = [1]*(W.ndim-1) + [strides] if isinstance(strides, int) else [1]*(W.ndim-len(strides)) + list(strides)
  dilations_ = [1]*(W.ndim-1) + [dilations] if isinstance(dilations, int) else [1]*(W.ndim-len(dilations)) + list(dilations)
  if output_shape and not output_padding:
    out_sh = [st*(xs-1) + (ks-1)*di+1 if n < 2 else st*(xs-1) + (ks-1)*di+1 - pads[n-2] - pads[n-1] for n, (st, xs, ks, di) in enumerate(zip(strides_, X.shape, kernel_shape, dilations_))]
    output_padding = [os - rs for os, rs in zip(output_shape, out_sh[-len(output_shape):])]
  return X.conv_transpose2d(W, B, stride=strides, groups=group, dilation=dilations, padding=pads if pads is not None else 0, output_padding=output_padding)

# Reimplemented here because you need legacy RNG for passing ONNX tests.
def Dropout(data: Tensor, ratio=0.5, training_mode=False, seed=None):
  if isinstance(ratio, Tensor) and not ratio.shape: ratio = safe_numpy(ratio) # ratio and tensor is passed in as Tensor with shape: ()
  if isinstance(training_mode, Tensor) and not training_mode.shape: training_mode = safe_numpy(training_mode)
  if not training_mode: return data, Tensor.ones(*data.shape, dtype=dtypes.bool)  # if mask is requested as output it will contain all True's.
  rng = np.random.RandomState(seed)
  ratio = ratio.lazydata.realize().toCPU()[0] if isinstance(ratio, Tensor) else ratio
  mask = Tensor((rng.random(data.shape) >= ratio), requires_grad=False, device=data.device)
  return data * mask * (1/(1.0 - ratio)), mask

def LRN(input: Tensor, size, alpha=1e-4, beta=0.75, bias=1.0):
  bs, c, iy, ix = input.shape
  return input / input.mul(input).reshape(bs,1,c,iy*ix).pad2d((0,0,(size-1)//2, size//2)).avg_pool2d((size, 1), 1).reshape(bs,c,iy,ix).mul(alpha).add(bias).pow(beta)

def MeanVarianceNormalization(input: Tensor, axis=(0, 2, 3)):
  data_mean = input.mean(axis=axis, keepdim=True)
  std = ((input**2).mean(axis=axis, keepdim=True) - data_mean**2).sqrt()
  return (input - data_mean) / (std + 1e-9)

def NegativeLogLikelihoodLoss(input: Tensor, target: Tensor, weight=None, ignore_index=None, reduction="mean"):
  target = target.cast(dtypes.float32)
  N, C, i_shape = input.shape[0], input.shape[1], input.shape
  t_shape = target.shape
  if len(input.shape) != 3:
    input = input.reshape((N, C, -1))
    target = target.reshape((N, -1))
  if weight is not None:
    mask = target.unsqueeze(-1) == Tensor.arange(C).repeat((N, 1, 1))
    weight = (mask * weight).sum(axis=-1)
  if ignore_index is not None:
    cond = target == ignore_index
    weight = cond.where(0, weight) if weight is not None else cond.where(Tensor.zeros(*target.shape), 1)
  mask = target[:, None, :] ==  Tensor.arange(C).reshape([1, C] + [1]*(len(input.shape) -2))
  loss = (-mask * input).sum(axis=1) * (1 if weight is None else weight)
  if reduction == "mean": return loss.mean() if weight is None else loss.sum() / weight.sum()
  elif reduction == "sum": return loss.sum()
  return loss.reshape(t_shape) if len(i_shape) != 3 else loss

def SoftmaxCrossEntropyLoss(scores: Tensor, labels: Tensor, weights=None, ignore_index=None, reduction="mean"):
  N, C, *s_dimensions = scores.shape
  if ignore_index is not None: labels = (labels == ignore_index).where(C+1, labels)
  mask = labels.unsqueeze(1) == Tensor.arange(C).reshape(1, C, *[1]*len(s_dimensions))
  y = scores.log_softmax(axis=1)
  if weights is not None: weights = weights.__getitem__(tuple([labels, *[slice(None)]*(weights.ndim-1)]))
  loss = (mask * -y).sum(1) if weights is None else (mask * -y).sum(1) * weights
  if reduction == "mean": loss = loss.sum() / (loss == 0).where(0, 1).sum() if weights is None else loss.sum() / weights.sum()
  elif reduction == "sum": loss = loss.sum()
  return loss, y

def ArrayFeatureExtractor(input: Tensor, indices: Tensor): return input.__getitem__(tuple([slice(None) if i != (input.ndim-1) else indices for i in range(input.ndim)]))
def Gather(input: Tensor, indices: Tensor, axis=0):
  if indices.numel() < 9: # NOTE lessor kernels for smaller indices but kernel number increases depending on size of indices
    input_sh = list(input.shape)
    ret_shape = input_sh[:axis] + list(indices.shape) + input_sh[axis+1:]
    if indices.ndim > 1: indices = indices.flatten()
    indices = [int(safe_numpy(indices))] if indices.shape == () else [input_sh[axis]+int(x) if x<0 else int(x) for x in safe_numpy(indices)]
    args = [[(0,x) if j != axis else (i,i+1) for j, x in enumerate(input_sh)] for i in indices]
    return input.shrink(arg=tuple(args[0])).cat(*[input.shrink(arg=tuple(arg)) for arg in args[1:]], dim=axis).reshape(ret_shape)
  else: # NOTE faster gather, fixed number of kernels, but exceeds limited kernels for openpilot
    return input.__getitem__(tuple([slice(None) if i != axis else indices for i in range(input.ndim)]))

def GatherElements(input: Tensor, indices: Tensor, axis):
  indices = indices.sign().contiguous().__neg__().contiguous().relu() * input.shape[axis] + indices
  return input.gather(indices, axis)

def _round(x:Tensor, n:float, equidistant_case = "round_down") -> Tensor:
  def _and(cond1, cond2): return ((cond1 + cond2) == 2).where(1, 0)
  assert n <= 1, f"n:{n} shouldn't be larger than 1"
  b = x.cast(dtypes.int32).contiguous().cast(x.dtype)
  b = (b >= 0).where(b+n, b-n)
  if equidistant_case == "round_down":
    return (x > b).where(b+1-n, b-n)
  elif equidistant_case == "round_up":
    return (x >= b).where(b+1-n, b-n)
  elif equidistant_case == "round_to_even":
    x_ceil_fraction = x.ceil()/2
    cond_ceil_even = x_ceil_fraction.ceil() == x_ceil_fraction
    x = (_and(x == b, cond_ceil_even)).where(x+1-n, x)
    x = (x > b).where(b+1-n, b-n)
    return x

def Round(X:Tensor): return _round(X, 0.5, "round_to_even")

def Resize(X:Tensor, roi=None, scales=None, sizes=None, antialias=0, axes=None, coordinate_transformation_mode='half_pixel', cubic_coeff_a=-0.75, exclude_outside=0, extrapolation_value=0.0, keep_aspect_ratio_policy='stretch', mode='nearest', nearest_mode='round_prefer_floor'):
  def _nearest_gather(X: Tensor, x_out, y_out): return X[:,:,y_out,:][:,:,:,x_out]
  def _nearest_mode(x_resized: Tensor, nearest_mode: str, x_len):
    if nearest_mode == "round_prefer_floor": ret = _round(x_resized, 0.5, "round_down")
    elif nearest_mode == "round_prefer_ceil": ret = _round(x_resized, 0.5, "round_up")
    elif nearest_mode == "floor": ret = x_resized.floor()
    elif nearest_mode == "ceil": ret = x_resized.ceil()
    return ret.clip(0, x_len-1)
  def _coordinate_transformation(x_out, y_out, output_shape, scales_lol, roi=None):
    if coordinate_transformation_mode == "half_pixel":
      x_out = (x_out + 0.5)/Tensor(scales_lol[-1]) - 0.5 # TODO Tensor() because try (((Tensor([0,1,2,3,4,5])+0.5)/3.5 - 0.5)) with LLVM or METAL, inaccuacy.
      y_out = (y_out + 0.5)/Tensor(scales_lol[-2]) - 0.5
    elif coordinate_transformation_mode == "align_corners":
      x_out = x_out * (X.shape[-1] - 1) / (output_shape[-1] - 1)
      y_out = y_out * (X.shape[-2] - 1) / (output_shape[-2] - 1)
    elif coordinate_transformation_mode == "asymmetric":
      x_out = x_out/scales_lol[-1]
      y_out = y_out/scales_lol[-2]
    elif coordinate_transformation_mode == "half_pixel_symmetric":
      x_out = X.shape[-1] / 2 * (1 - int(output_shape[-1]) / output_shape[-1]) + (x_out + 0.5) / scales_lol[-1] - 0.5
      y_out = X.shape[-2] / 2 * (1 - int(output_shape[-2]) / output_shape[-2]) + (y_out + 0.5) / scales_lol[-2] - 0.5
    elif coordinate_transformation_mode == "pytorch_half_pixel":
      x_out = (x_out + 0.5)/scales_lol[-1] - 0.5 if output_shape[-1] > 1 else Tensor([0])
      y_out = (y_out + 0.5)/scales_lol[-2] - 0.5 if output_shape[-2] > 1 else Tensor([0])
    elif coordinate_transformation_mode == "tf_crop_and_resize":
      x_out = roi[-1][0] * (X.shape[-1] - 1) + x_out * ((roi[-1][1] - roi[-1][0]) * (X.shape[-1] - 1) / (output_shape[-1] - 1))  if output_shape[-1] > 1 else Tensor([0.5 * (roi[-1][0] + roi[-1][1]) * (X.shape[-1] - 1)])
      y_out = roi[-2][0] * (X.shape[-2] - 1) + y_out * ((roi[-2][1] - roi[-2][0]) * (X.shape[-2] - 1) / (output_shape[-2] - 1))  if output_shape[-2] > 1 else Tensor([0.5 * (roi[-2][0] + roi[-2][1]) * (X.shape[-2] - 1)])
    return x_out.clip(0, X.shape[-1]-1), y_out.clip(0, X.shape[-2]-1)
  if roi is not None:
    roi = safe_numpy(roi)
    roi = [(st,ed) for st, ed in zip(roi[:len(roi)//2], roi[len(roi)//2:])]
    roi_ = [(1,1)] * 4
    if axes is not None:
      for a,r in zip(axes, roi):
        roi_[a] = r
      roi = roi_
  if scales is not None:
    scales = safe_numpy(scales).tolist()
    if axes is not None:
      scales_ = [1]*X.ndim
      for a,s in zip(axes, scales):
        scales_[a] = s
      scales = scales_
  elif sizes is not None:
    sizes = [int(i) for i in safe_numpy(sizes)]
    scales = []
    if axes is not None:
      sizes_ = [1]*X.ndim
      for a,s in zip(axes, sizes):
        sizes_[a] = s
        scales.append(s/X.shape[a])
      sizes = sizes_
    else: scales = [si/xs for xs, si in zip(X.shape, sizes)]
    if keep_aspect_ratio_policy == "not_larger":
      scale = min(scales)
      sizes = _round(Tensor(list(X.shape[-2:]))*scale, 0.5, "round_up")
      sizes = list(X.shape[:-2]) + [int(i) for i in safe_numpy(sizes)]
    elif keep_aspect_ratio_policy == "not_smaller":
      scale = max(scales)
      sizes = _round(Tensor(list(X.shape[-2:]))*scale, 0.5, "round_up")
      sizes = list(X.shape[:-2]) + [int(i) for i in safe_numpy(sizes)]
  output_shape = sizes if sizes else [math.floor(x*s) for x,s in zip(X.shape, scales)]
  output_shape_ = sizes if sizes else [x*s for x,s in zip(X.shape, scales)]
  scales_lol = [os/xs for xs, os in zip(X.shape, output_shape)]
  x_out = Tensor.arange(output_shape[-1])
  y_out = Tensor.arange(output_shape[-2])
  if mode == "nearest":
    x_out, y_out = _coordinate_transformation(x_out, y_out, output_shape, scales_lol, roi)
    x_out = _nearest_mode(x_out, nearest_mode, X.shape[-1])
    y_out = _nearest_mode(y_out, nearest_mode, X.shape[-1])
    return _nearest_gather(X, x_out, y_out)
  elif mode == "linear":
    x_out, y_out = _coordinate_transformation(x_out, y_out, output_shape_, scales, roi)
    ret = []
    for y in safe_numpy(y_out):
      for x in safe_numpy(x_out):
        x_floor, y_floor = int(x), int(y)
        y_shrink = (0, X.shape[2]) if X.shape[2] == 1 else (y_floor, y_floor+2) if y != y_floor else (y_floor, y_floor+1)
        x_shrink = (x_floor, x_floor+2) if x != x_floor else (x_floor, x_floor+1)
        shrink_args = ((0, X.shape[0]), (0, X.shape[1]), y_shrink, x_shrink)
        corners = safe_numpy(X.shrink(shrink_args))
        x1, x2, y1, y2 = x_floor, x_floor+1, y_floor, y_floor+1
        if x == x_floor and y == y_floor: # TODO https://en.wikipedia.org/wiki/Bilinear_interpolation#Weighted_mean maybe do weighted mean?
          ret.append(corners[0,0,0,0])
        elif x == x_floor:
          ret.append((corners[0,0,0,0] * (y2 - y) + corners[0,0,1,0] * (y - y1)) / (y2 - y1))
        elif y == y_floor:
          ret.append((corners[0,0,0,0] * (x2 - x) + corners[0,0,0,1] * (x - x1)) / (x2 - x1))
        else:
          ret.append((corners[0,0,0,0] * (x2 - x) * (y2 - y) + corners[0,0,0,1] * (x - x1) * (y2 - y) + corners[0,0,1,0] * (x2 - x) * (y - y1) + corners[0,0,1,1] * (x - x1) * (y - y1)) / ((x2 - x1) * (y2 - y1)))
    return Tensor(ret).reshape(output_shape)
  elif mode == "cubic":
    raise Exception("cubic interpolation is not implemented")

def CenterCropPad(input: Tensor, shape: Tensor, axes=None):
  if not axes: axes = list(range(input.ndim))
  shrink_arg = [(0,i) for i in input.shape]
  pad_arg = [(0,0) for _ in range(input.ndim)]
  shape = safe_numpy(shape).tolist()
  for s, x in zip(shape, axes):
    if s < input.shape[x]: shrink_arg[x] = (input.shape[x]//2 - s//2, input.shape[x]//2 + s//2) if s%2 == 0 else (input.shape[x]//2 - s//2 - 1, input.shape[x]//2 + s//2)
    elif s > input.shape[x]: pad_arg[x] = ((s - input.shape[x])//2, (s - input.shape[x])//2)  if (s - input.shape[x])% 2 == 0 else ((s - input.shape[x])//2, (s - input.shape[x])//2 + 1)
  return input.shrink(tuple(shrink_arg)).pad(tuple(pad_arg))

def OneHot(indices: Tensor, depth: Tensor, values: Tensor, axis=-1):
  depth = int(safe_numpy(depth).item())
  indices, rank = (indices < 0).where(indices+depth, indices), len(indices.shape)
  if axis < 0: axis += rank + 1
  ls, rs = indices.shape[0:axis], indices.shape[axis: rank]
  cond = indices[:,None] == Tensor.arange(depth).reshape((1,) * len(ls) + (depth,) + (1,) * len(rs))
  return cond.where(values[1], values[0]).cast(values.dtype)

def Erf(x: Tensor):
  sign = x.sign()
  x = x.abs()
  t = 1.0 / (1.0 + 0.3275911 * x)
  term1 = 0.254829592 * t
  term2 = -0.284496736 * t ** 2
  term3 = 1.421413741 * t ** 3
  term4 = -1.453152027 * t ** 4
  term5 = 1.061405429 * t ** 5
  y = (term1 + term2 + term3 + term4 + term5)
  return sign * (1.0 - y * Tensor.exp(-x * x))

def Compress(inp: Tensor, condition: Tensor, axis=None):
  if axis is None:
    inp = inp.flatten()
    axis = 0

  axis = axis + inp.ndim if axis < 0 else axis

  con_np = safe_numpy(condition)
  con = Tensor(np.arange(condition.shape[0])[con_np]) # no boolean indexing in Tensor
  return inp.__getitem__(tuple([slice(None) if i != axis else con for i in range(inp.ndim)]))

type_map = {TensorProto.DOUBLE: dtypes.double, TensorProto.FLOAT: dtypes.float32}
def EyeLike(x: Tensor, dtype=None, k=0):
  if dtype is None: dtype = x.dtype
  else: dtype = type_map[dtype]
  shape = x.shape
  dim = min(x.shape)
  if shape[0] == shape[1]: return Tensor.eye(dim=dim, dtype=dtype)
  else:
    diff = (shape[0]-dim, shape[1]-dim)
    padarg = tuple([(d, d) if d == 0 else (k, d-k) for d in diff])
    return Tensor.eye(dim=dim, dtype=dtype).pad(padarg)

def Upsample(X, scales, mode): return Resize(X=X, scales=scales, mode=mode)

# Needs work
def IsInf(x,detect_negative=1,detect_positive=1):
  ret = (x == float("inf"))*detect_positive + (x == float("-inf"))*detect_negative + Tensor.zeros(*x.shape)
  return ret.cast(dtypes.bool)

# Needs work
def DequantizeLinear(x: Tensor, x_scale: Tensor, x_zero_point=0, axis=1):
  axis = axis + x.ndim if axis < 0 else axis
  x_sc = x_scale.reshape(*[1]*axis, *x_scale.shape, *[1]*(x.ndim - axis - x_scale.ndim))
  x_zer = x_zero_point.reshape(*[1]*axis, *x_scale.shape, *[1]*(x.ndim - axis - x_scale.ndim)) if isinstance(x_zero_point, Tensor) else x_zero_point
  return (x - x_zer) * x_sc

# Needs work
def IsNaN(x):
  return (x < float("-inf")).cast(dtypes.bool)

# **************** com.microsoft Ops ****************

def SkipLayerNormalization(input:Tensor, skip:Tensor, gamma, beta:Optional[Tensor]=None, bias:Optional[Tensor]=None, epsilon=None):
  if epsilon is None: epsilon=1e-12
  x = input + skip + bias
  return x.layernorm(eps=epsilon) * gamma + beta, None, None, x

def FastGelu(x:Tensor, bias:Optional[Tensor]=None):
  x = x + bias
  return 0.5 * x * (1 + (x * 0.797885 + 0.035677 * x ** 3).tanh())

def EmbedLayerNormalization(input_ids: Tensor, segment_ids:Optional[Tensor]=None, word_embedding:Tensor=None, position_embedding:Tensor=None, segment_embedding:Optional[Tensor]=None, gamma=None, beta=None, mask:Optional[Tensor]=None, position_ids:Optional[Tensor]=None, epsilon=None, mask_index_type=None):
  # https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.EmbedLayerNormalization
  assert (segment_ids is None) is (segment_embedding is None)
  assert (mask is None) is (mask_index_type is None)
  assert mask is None, "functionality not supported yet"  # TODO
  input_shape = input_ids.shape
  bsz, seq_length = input_shape[0], input_shape[1]
  compute_seg_emb = (segment_embedding is not None and segment_ids is not None)
  vocab_size, max_position_embeddings, type_vocab_size = word_embedding.shape[0], position_embedding.shape[0], (segment_embedding.shape[0] if compute_seg_emb else None)

  def embedding(x:Tensor, vocab_size, weight:Tensor)->Tensor:  # TODO from nn.Embedding. Could probably upstream this to Tensor
    vocab_counter = Tensor.arange(vocab_size, dtype=x.dtype, requires_grad=False).reshape(1, 1, vocab_size).expand(*x.shape, vocab_size)
    return (vocab_counter == x.unsqueeze(2).expand(*x.shape, vocab_size)) @ weight

  # bert embedding layer
  if epsilon is None: epsilon = 1e-12
  if position_ids is None: position_ids = Tensor.arange(seq_length, requires_grad=False).unsqueeze(0).expand(*input_shape)
  wrd_embedding_res = embedding(input_ids, vocab_size, word_embedding)
  pos_embedding_res = embedding(position_ids, max_position_embeddings, position_embedding)
  seg_embedding_res = embedding(segment_ids, type_vocab_size, segment_embedding) if compute_seg_emb else None

  embedding_sum = wrd_embedding_res + pos_embedding_res + seg_embedding_res
  out = embedding_sum.layernorm(eps=epsilon) * gamma + beta
  return out, None, embedding_sum

def Attention(input:Tensor, weights, bias:Optional[Tensor]=None, mask_index:Optional[Tensor]=None, past:Optional[Tensor]=None, relative_position_bias:Optional[Tensor]=None, past_sequence_length:Optional[Tensor]=None, do_rotary=None, mask_filter_value=None, num_heads=None, past_present_share_buffer=None, qkv_hidden_sizes=None, scale=None, unidirectional=None):
  # https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.Attention
  assert num_heads is not None  # required
  assert (qkv_hidden_sizes is None and past is not None) or (qkv_hidden_sizes is not None)
  assert relative_position_bias==do_rotary==past_sequence_length==mask_filter_value==past_present_share_buffer==scale==None, "functionality not supported yet"  # TODO strange params
  hidden_size, v_hidden_size = qkv_hidden_sizes[1:] if qkv_hidden_sizes is not None else 2*(weights.shape[1] // 3,)

  if unidirectional:  # gpt-style
    assert hidden_size == v_hidden_size
    xqkv = input.linear(weights, bias)
    xq, xk, xv = [xqkv.slice([None, None, (i*hidden_size, (i+1)*hidden_size)]) for i in range(3)]
  else:  # bert-style
    wq, wk, wv = weights[:,:hidden_size], weights[:,hidden_size:hidden_size+v_hidden_size], weights[:,hidden_size+v_hidden_size:]
    bq, bk, bv = (bias[:hidden_size], bias[hidden_size:hidden_size+v_hidden_size], bias[hidden_size+v_hidden_size]) if bias is not None else None
    xq, xk, xv = [input.linear(w, b) for w, b in zip((wq, wk, wv), (bq, bk, bv))]
  xq, xk, xv = [x.reshape(x.shape[0], x.shape[1], num_heads, -1).transpose(1, 2) for x in (xq, xk, xv)]

  if past is not None:
    xk, xv = Tensor.cat(past[0], xk, dim=-2), Tensor.cat(past[1], xv, dim=-2)
    present = Tensor.cat(xk.unsqueeze(0), xv.unsqueeze(0))

  def attn(query, key, value, attn_mask):
    query_length, key_length = query.shape[-2], key.shape[-2]
    cdim = max(query_length, key_length) + 1
    attn_weights = query @ key.transpose(-1, -2) / math.sqrt(value.shape[-1])
    # This is where Tensor.scaled_dot_product_attention differs:
    causal_mask = Tensor.ones((cdim, cdim), requires_grad=False).cast(dtypes.bool).tril(0)[key_length - query_length : key_length, :key_length].cast(dtypes.bool)
    return (Tensor.where(causal_mask, attn_weights, -float("inf")) + attn_mask).softmax(-1) @ value

  bsz, _, seq_len, _ = xq.shape
  out = attn(xq, xk, xv, mask_index).transpose(1, 2).reshape(bsz, seq_len, -1)
  return out, present

# **************** ai.onnx.preview.training Ops ****************

# TODO not entirely sure these optimizers are correct
def Adagrad(R, T, *inputs, decay_factor=0.0, epsilon=0.0, norm_coefficient=0.0):
  groups = len(inputs) // 3
  grouped_inputs = [inputs[i::groups] for i in range(groups)]
  T, R = safe_numpy(T)[0], safe_numpy(R)[0]
  r = R / (1 + T * decay_factor)
  ret = []
  for input in grouped_inputs:
    X, G, H = input
    X.grad = norm_coefficient * X + G
    X.grad.requires_grad, H.requires_grad = False, False # TODO manually turning off requires_grad, see TODO under (domain == "ai.onnx.preview.training") in onnx.py
    H.assign(H.detach() + X.grad * X.grad).realize()
    H_adaptive = H.sqrt() + epsilon
    X.assign(X.detach() - r * X.grad / H_adaptive)
    ret.extend([X, H])
  ret = ret[::2] + ret[1::2]
  return tuple(ret)

def Momentum(R, T, *inputs, alpha, beta, mode, norm_coefficient):
  groups = len(inputs) // 3
  grouped_inputs = [inputs[i::groups] for i in range(groups)]
  T, R = safe_numpy(T)[0], safe_numpy(R)[0]
  beta_adjusted = beta if T > 0 else 1
  ret = []
  for input in grouped_inputs:
    X, G, V = input
    X.grad = (norm_coefficient * X + G).realize()
    X.grad.requires_grad, V.requires_grad = False, False
    V.assign(alpha * V + beta_adjusted * X.grad).realize()
    if mode == "standard": X.assign(X.detach() - R * V).realize()
    elif mode == "nesterov": X.assign(X.detach() - R * (X.grad + alpha + V)).realize()
    ret.extend([X, V])
  ret = ret[::2] + ret[1::2]
  return tuple(ret)

# copied from tinygrad/nn/optim.py: LAMB with some edits
def Adam(R, T, *inputs, alpha=0.9, beta=0.999, epsilon=0.0, norm_coefficient=0.0, norm_coefficient_post=0.0):
  groups = len(inputs) // 4
  grouped_inputs = [inputs[i::groups] for i in range(groups)]
  T, R = safe_numpy(T)[0], safe_numpy(R)[0]
  ret = []
  for input in grouped_inputs:
    X, G, V, H = input
    X.grad = (norm_coefficient * X + G).realize()
    V.requires_grad, H.requires_grad, X.grad.requires_grad = False, False, False
    V.assign(alpha * V + (1.0 - alpha) * X.grad).realize()
    H.assign(beta * H + (1.0 - beta) * (X.grad * X.grad)).realize()
    up = (V / (1.0 - alpha**T)) / ((H / (1.0 - beta**T)).sqrt() + epsilon) if T > 0 else V / (H.sqrt() + epsilon)
    X.assign(X.detach() - R * up).realize()
    X = (1 - norm_coefficient_post) * X
    ret.extend([X, V, H])
  ret = ret[::3] + ret[1::3] + ret[2::3]
  return tuple(ret)
