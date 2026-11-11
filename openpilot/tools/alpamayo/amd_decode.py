from functools import partial

from tinygrad import Tensor, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp
from amd_attention import _softmax, _warp_reduce
from amd_gemm import pack


def _gemv(out, x, weight, x_scale, w_scale, *bias, packed=True):
  rows, width = weight.shape
  group = UOp.range(rows//(16 if packed else 4), 0, AxisType.GLOBAL)
  wave, lane = UOp.range(4, 1, AxisType.LOCAL), UOp.range(32, -1, AxisType.WARP)
  if packed:
    row, k = group*16+lane%16, UOp.range(width//64, 2, AxisType.LOOP)
    cols = [(k*4+wave)*16+(lane//16)*4+j%4+(j//4)*8 for j in range(8)]
    indices = [(group*(width//16)+c//16)*256+(c%16//4)*64+(row%16)*4+c%4 for c in cols]
  else:
    row, k = group*4+wave, UOp.range(width//128, 2, AxisType.LOOP)
    cols = [k*128+lane*4+j for j in range(4)]
    indices = [row*width+c for c in cols]
  value = sum(x.flatten().index(c).cast(dtypes.float32) * weight.flatten().index(i).cast(dtypes.float32)
              for c, i in zip(cols, indices, strict=True)).reduce(k, arg=Ops.ADD)
  if packed:
    shuffle = UOp.custom_function('llvm.amdgcn.ds.swizzle', dtype=dtypes.int32)
    value = value + shuffle.call(value.bitcast(dtypes.int32), UOp.const(0x1f | (16 << 10), dtypes.int32)).bitcast(dtypes.float32)
    partials = UOp.alloc((64,), dtypes.float32, addrspace=AddrSpace.LOCAL)
    stored = partials.index((wave*16+lane%16).valid(lane < 16)).store(value)
    value = sum(partials.after(stored).index(i*16+lane%16) for i in range(4))
    valid = wave.eq(0) & (lane < 16)
  else:
    value, valid = _warp_reduce(value), lane.eq(0)
  value = value * (x_scale.flatten().index(0) * w_scale.flatten().index(0))
  if bias:
    value = value + bias[0].flatten().index(row).cast(dtypes.float32)
  store = out.flatten().index(row.valid(valid)).store(value.cast(out.dtype))
  return store.end(group, wave, lane).sink(arg=KernelInfo(name='alpamayo_fp8_gemv', opts_to_apply=()))


def gemv(x, weight, scales, bias=None, packed=True, dtype=dtypes.bfloat16):
  out = Tensor.empty(weight.shape[0], dtype=dtype, device=x.device)
  return Tensor.custom_kernel(out, x, pack(weight) if packed else weight, *scales, *(() if bias is None else (bias,)),
                              fxn=partial(_gemv, packed=packed))[0]


def _scores(out, q, k, mask):
  batch, heads, _, width = q.shape
  kv_heads, keys = k.shape[1:3]
  tiles = (keys+3)//4
  group = UOp.range(batch*heads*tiles, 0, AxisType.GLOBAL)
  wave, lane = UOp.range(4, 1, AxisType.LOCAL), UOp.range(32, -1, AxisType.WARP)
  head, key = group//tiles, (group%tiles)*4+wave
  kv_head = (head//heads)*kv_heads + (head%heads)//(heads//kv_heads)
  value = sum(q.flatten().index(head*width+c).cast(dtypes.float32) *
              k.flatten().index(((kv_head*keys+key)*width+c).valid(key < keys)).cast(dtypes.float32) for c in (lane*4+j for j in range(4)))
  value = _warp_reduce(value).cast(dtypes.bfloat16) * UOp.const(width**-0.5, dtypes.bfloat16)
  value = value.cast(dtypes.float32) + mask.flatten().index(key.valid(key < keys))
  store = out.flatten().index((head*keys+key).valid((key < keys) & lane.eq(0))).store(value.cast(out.dtype))
  return store.end(group, wave, lane).sink(arg=KernelInfo(name='alpamayo_decode_scores', opts_to_apply=()))


def _values(out, probs, v):
  batch, heads, parts, width = out.shape
  kv_heads, keys = v.shape[1:3]
  group = UOp.range(batch*heads*parts, 0, AxisType.GLOBAL)
  wave, lane = UOp.range(4, 1, AxisType.LOCAL), UOp.range(32, -1, AxisType.WARP)
  head, part, col = group//parts, group%parts, wave*32+lane
  kv_head = (head//heads)*kv_heads + (head%heads)//(heads//kv_heads)
  r = UOp.range(128, 2, AxisType.LOOP)
  key = part*128+r
  value = (probs.flatten().index((head*keys+key).valid(key < keys)).cast(dtypes.float32) *
           v.flatten().index(((kv_head*keys+key)*width+col).valid(key < keys)).cast(dtypes.float32)).reduce(r, arg=Ops.ADD)
  return out.flatten().index(group*width+col).store(value).end(group, wave, lane).sink(
    arg=KernelInfo(name='alpamayo_decode_values', opts_to_apply=()))


def attention(q, k, v, mask):
  batch, heads, _, width = q.shape
  keys = k.shape[2]
  scores = Tensor.empty(batch, heads, 1, keys, dtype=q.dtype, device=q.device)
  scores = Tensor.custom_kernel(scores, q.contiguous(), k.contiguous(), mask.float().contiguous(), fxn=_scores)[0]
  probs = Tensor.empty(scores.shape, dtype=q.dtype, device=q.device)
  probs = Tensor.custom_kernel(probs, scores, fxn=partial(_softmax, packed=False))[0]
  partials = Tensor.empty(batch, heads, (keys+127)//128, width, dtype=dtypes.float32, device=q.device)
  partials = Tensor.custom_kernel(partials, probs, v.contiguous(), fxn=_values)[0]
  return partials.sum(2, keepdim=True).cast(q.dtype)
