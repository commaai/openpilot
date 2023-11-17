import numpy as np
import operator
from typing import Callable, Dict, Tuple, Optional
from tinygrad.helpers import dtypes, DType
from tinygrad.ops import BufferOps, UnaryOps, BinaryOps, MovementOps, ReduceOps, TernaryOps, Op, Interpreted
from tinygrad.runtime.lib import RawBuffer

def shape_to_axis(old_shape:Tuple[int, ...], new_shape:Tuple[int, ...]) -> Tuple[int, ...]:
  assert len(old_shape) == len(new_shape), "reduce shapes must have same dimensions"
  return tuple(i for i,(a,b) in enumerate(zip(old_shape, new_shape)) if a != b)

base_fxn_for_op: Dict[Op, Callable] = {
  BufferOps.MEM: lambda x: x._buf, UnaryOps.NEG: operator.neg, BinaryOps.ADD: operator.add, BinaryOps.SUB: operator.sub, BinaryOps.MUL: operator.mul, BinaryOps.DIV: operator.truediv,
  ReduceOps.SUM: lambda x, new_shape: x.sum(shape_to_axis(x.shape, new_shape), keepdims=True) if tuple(x.shape) != tuple(new_shape) else x[:],
  ReduceOps.MAX: lambda x, new_shape: (x.amax if hasattr(x, 'amax') else x.max)(shape_to_axis(x.shape, new_shape), keepdims=True) if tuple(x.shape) != tuple(new_shape) else x[:],
  MovementOps.RESHAPE: lambda x, arg: x.reshape(arg), MovementOps.SHRINK: lambda x, arg: x[tuple(slice(p[0], p[1], None) for p in arg)],
}

def match_types(x, y):
  up = x.dtype if dtypes.from_np(x.dtype).priority > dtypes.from_np(y.dtype).priority else y.dtype
  return x.astype(up, copy=False), y.astype(up, copy=False)

def einsum_mulacc(einsum, get_strides, expand):
  def einscripts(x): return ''.join(["abcdefghijklmnopqrstuvwxyz"[i] for i in x])
  def axes_slice(strides): return [i for i,s in enumerate(strides) if s != 0], tuple([slice(None) if s != 0 else 0 for i,s in enumerate(strides)])
  def mulacc(a, b, new_shape):
    (a_axes, a_slices), (b_axes, b_slices) = axes_slice(get_strides(a)), axes_slice(get_strides(b))
    out = [i for i in range(len(new_shape)) if a.shape[i] == new_shape[i] and (i in a_axes or i in b_axes)]
    ret = einsum(f"{einscripts(a_axes)}, {einscripts(b_axes)} -> {einscripts(out)}", a[a_slices], b[b_slices])
    return expand(ret.reshape([(1 if i not in a_axes and i not in b_axes else s) for i,s in enumerate(new_shape)]), new_shape)
  return mulacc

numpy_fxn_for_op: Dict[Op, Callable] = {**base_fxn_for_op, **{
  BufferOps.CONST: lambda val, dtype: np.array(val, dtype=dtype.np),
  UnaryOps.NOOP: lambda x: np.require(x, requirements='C'), UnaryOps.EXP2: np.exp2, UnaryOps.LOG2: np.log2, UnaryOps.SIN: np.sin,
  UnaryOps.CAST: lambda x,y: x.view(y[0].np) if y[1] else x.astype(y[0].np, copy=False),
  BinaryOps.MAX: np.maximum, BinaryOps.CMPLT: lambda x,y: (x<y).astype(np.promote_types(x.dtype,y.dtype)), BinaryOps.ADD: lambda x, y: np.add(*match_types(x, y)),
  BinaryOps.SUB: lambda x, y: np.subtract(*match_types(x, y)), BinaryOps.MUL: lambda x, y: np.multiply(*match_types(x, y)),
  BinaryOps.DIV: lambda x, y: np.divide(*match_types(x, y)), UnaryOps.SQRT: np.sqrt,
  MovementOps.PERMUTE: lambda x, order: x.transpose(order), MovementOps.PAD: np.pad, MovementOps.EXPAND: np.broadcast_to,
  MovementOps.STRIDE: lambda x, arg: x[tuple(slice(None, None, i) for i in arg)],
  MovementOps.AS_STRIDED: lambda x, arg: np.ndarray(arg[0], buffer=np.require(x, requirements='C'), dtype=x.dtype, offset=arg[2]*x.dtype.itemsize, strides=tuple(y*x.dtype.itemsize for y in arg[1])),
  TernaryOps.MULACC: einsum_mulacc(lambda s,a,b: np.einsum(s, *match_types(a.copy(), b.copy()), optimize=True), lambda x: x.strides, np.broadcast_to),
  TernaryOps.WHERE: np.where,
}}

class RawNumpyBuffer(RawBuffer):
  def __init__(self, size:int, dtype:DType, buf:Optional[np.ndarray]=None): super().__init__(size, dtype, buf if buf is not None else np.empty([size], dtype.np))
  @classmethod
  def fromCPU(cls, x): return cls(x.size, dtypes.from_np(x.dtype), x)
  def toCPU(self): return self._buf
CPUBuffer = Interpreted(RawNumpyBuffer, numpy_fxn_for_op, from_underlying=RawNumpyBuffer.fromCPU)
