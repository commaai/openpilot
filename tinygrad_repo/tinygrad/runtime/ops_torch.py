import torch
import numpy as np
from typing import Dict, Callable, Optional
from tinygrad.ops import BufferOps, UnaryOps, BinaryOps, MovementOps, TernaryOps, Op, Interpreted
from tinygrad.helpers import getenv, dtypes, prod, DType
from tinygrad.runtime.ops_cpu import base_fxn_for_op, einsum_mulacc
from tinygrad.runtime.lib import RawBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else ("mps" if getenv("MPS", 0) else "cpu"))
type_map = {torch.float64: dtypes.float64, torch.float16: dtypes.float16, torch.float32: dtypes.float32, torch.int8: dtypes.int8, torch.int32: dtypes.int32, torch.int64: dtypes.int64, torch.uint8: dtypes.uint8, torch.bool: dtypes.bool}
inverse_type_map = {v:k for k,v in type_map.items()}

def as_strided(x, arg):
  if any(i < 0 for i in arg[1]):
    return torch.as_strided(x.contiguous(), arg[0], tuple(abs(i) for i in arg[1]),
      arg[2] + sum((s-1)*a if a < 0 else 0 for (s,a) in zip(arg[0], arg[1]))).flip([i for i,a in enumerate(arg[1]) if a < 0])
  return torch.as_strided(x.contiguous(), arg[0], arg[1], arg[2])

torch_fxn_for_op: Dict[Op, Callable] = {**base_fxn_for_op, **{
  # TODO: torch.tensor should work here
  #BufferOps.CONST: lambda val, dtype: torch.tensor(val, dtype=inverse_type_map[dtype]),
  BufferOps.CONST: lambda val, dtype: torch.from_numpy(np.array(val, dtype=dtype.np)),
  UnaryOps.NOOP: lambda x: x.contiguous(), UnaryOps.SQRT: lambda x: x.sqrt(), UnaryOps.EXP2: lambda x: x.exp2(), UnaryOps.LOG2: lambda x: x.log2(), UnaryOps.SIN: torch.sin,
  UnaryOps.CAST: lambda x,y: (x.view if y[1] else x.type)(next(k for k,v in type_map.items() if v==y[0])),
  BinaryOps.MAX: torch.maximum, BinaryOps.CMPLT: lambda x,y: (x<y).type(torch.promote_types(x.dtype, y.dtype)), BinaryOps.SUB: lambda x,y: torch.logical_xor(x, y) if y.dtype is torch.bool else torch.sub(x, y),
  MovementOps.PAD: lambda x, padding: torch.nn.functional.pad(x, [item for sublist in padding[::-1] for item in sublist]), # pylint: disable=E1102
  TernaryOps.MULACC: einsum_mulacc(lambda s,a,b: torch.einsum(s, a.float(), b.float()).type(torch.promote_types(a.dtype, b.dtype)), lambda x: x.stride(), lambda x,s: x.expand(s)),
  TernaryOps.WHERE: lambda x, y, z: torch.where(x != 0, y, z),
  MovementOps.STRIDE: lambda x, arg: x[tuple(slice(None, None, abs(i)) for i in arg)].flip([i for i,a in enumerate(arg) if a < 0]),
  MovementOps.EXPAND: lambda x, arg: x.expand(arg), MovementOps.PERMUTE: lambda x, arg: x.permute(arg),
  MovementOps.AS_STRIDED: as_strided
}}

class RawTorchBuffer(RawBuffer):
  def __init__(self, size:int, dtype:DType, buf:Optional[torch.Tensor]=None): super().__init__(size, dtype, buf if buf is not None else torch.empty([size], dtype=inverse_type_map[dtype]))
  @classmethod
  def fromCPU(cls, x):
    buf = torch.from_numpy(x if all(s>=0 for s in x.strides) else x.copy()).requires_grad_(False).to(device)
    return cls(prod(x.shape), type_map[buf.dtype], buf)
  def toCPU(self): return self._buf.cpu().numpy()
TorchBuffer = Interpreted(RawTorchBuffer, torch_fxn_for_op, from_underlying=lambda x: RawTorchBuffer(prod(x.shape), type_map[x.dtype], x))
